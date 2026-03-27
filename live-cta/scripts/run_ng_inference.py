"""NG live inference: IBKR live ticks + daily context -> HybridMixtureNetwork -> position.

IBKR provides the live tick/bar stream. Daily context (EIA storage,
population-weighted weather) is either:
  - loaded from local files
  - pulled from S3/GCS
  - fetched fresh from EIA + NCEI APIs via macrOS-Int (--refresh-context)

Usage
-----
    # Pre-existing local caches
    python scripts/run_ng_inference.py --model-path model.pt --conid 462193585 \\
        --eia-key F:/Data/ng_eia_cache.hdf \\
        --weather-key F:/Data/new_weather.hdf

    # Pull from S3 bucket
    python scripts/run_ng_inference.py --model-path model.pt --conid 462193585 \\
        --context-backend s3 --context-bucket ctaflow-prod \\
        --eia-key model_data/new_ng_eia_cache.hdf \\
        --weather-key model_data/new_weather.hdf

    # Fetch fresh from APIs (requires EIA_API_KEY + NCEI_TOKEN env vars)
    python scripts/run_ng_inference.py --model-path model.pt --conid 462193585 \\
        --refresh-context

    # Continuous loop with fresh context refresh each day
    python scripts/run_ng_inference.py --model-path model.pt --conid 462193585 \\
        --refresh-context --loop --interval 1800
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Load environment variables from dotenv BEFORE any config reads
from dotenv import load_dotenv

_env_file = os.getenv("DOTENV_PATH", str(Path(__file__).resolve().parent.parent / "env" / "dot.env"))
load_dotenv(_env_file, override=False)

import numpy as np
import pandas as pd
import torch

from live_cta.sources.ibkr_client import IBKRConfig, IBKRContract, IBKRTickDataSource
from CTAFlow.models.deep_learning.multi_branch.ng_moe import (
    HybridConfig,
    HybridMixtureNetwork,
)
from CTAFlow.models.deep_learning.training.backtest import predictions_to_positions
from live_cta.core.ng_live import (
    DailyContextPaths,
    NatGasLiveInterface,
    ng_default_config,
)

logger = logging.getLogger("ng_inference")

TZ = "America/Chicago"
CACHE_DIR = Path(os.getenv("CTAFLOW_CACHE_DIR", Path.home() / ".ctaflow" / "live_context"))

# GCS / path defaults from env
_GCS_BUCKET = os.getenv("GCS_BUCKET", "ctaflow-prod-artifacts")
_GCS_PROJECT = os.getenv("GCS_PROJECT", "")
_GCS_REGION = os.getenv("GCS_REGION", "us-central1")
_GCS_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
_MODEL_DATA_DIR = os.getenv("MODEL_DATA_DIR", "model_data/")
_INTRADAY_CSV = os.getenv("INTRADAY_CSV", "model_data/NG/intraday_2.csv")
_RESULTS_DIR = os.getenv("RESULTS_DIR", "results/")
_running = True


def _handle_signal(signum, frame):
    global _running
    logger.info("Signal %s received, shutting down after current cycle...", signum)
    _running = False


# ---------------------------------------------------------------------------
# Download daily context files from cloud
# ---------------------------------------------------------------------------

def download_context_s3(
    bucket: str, keys: Dict[str, str], endpoint: str = "", prefix: str = "",
) -> Dict[str, Path]:
    """Download context files from S3 to local cache. Returns key -> local path."""
    from live_cta.storage.aws_client import AWSClient, S3Config

    cfg = S3Config(bucket_name=bucket, endpoint_url=endpoint or None, prefix=prefix)
    client = AWSClient(config=cfg, cache_dir=CACHE_DIR)

    local_paths = {}
    for label, s3_key in keys.items():
        if not s3_key:
            continue
        local = CACHE_DIR / Path(s3_key).name
        try:
            client.download_file(s3_key, str(local))
            local_paths[label] = local
            logger.info("S3 -> %s : %s", label, local)
        except Exception as exc:
            logger.warning("Failed to download %s from S3: %s", s3_key, exc)
    return local_paths


def download_context_gcs(
    bucket: str, keys: Dict[str, str], project: str = "", creds: str = "",
) -> Dict[str, Path]:
    """Download context files from GCS to local cache. Returns key -> local path."""
    try:
        from google.cloud import storage as gcs_storage
    except ImportError:
        logger.error("google-cloud-storage not installed")
        return {}

    kwargs = {}
    if project:
        kwargs["project"] = project
    if creds:
        from google.oauth2 import service_account
        kwargs["credentials"] = service_account.Credentials.from_service_account_file(creds)

    client = gcs_storage.Client(**kwargs)
    gcs_bucket = client.bucket(bucket)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    local_paths = {}
    for label, gcs_key in keys.items():
        if not gcs_key:
            continue
        local = CACHE_DIR / Path(gcs_key).name
        try:
            blob = gcs_bucket.blob(gcs_key)
            blob.download_to_filename(str(local))
            local_paths[label] = local
            logger.info("GCS -> %s : %s", label, local)
        except Exception as exc:
            logger.warning("Failed to download %s from GCS: %s", gcs_key, exc)
    return local_paths


# ---------------------------------------------------------------------------
# Fetch fresh EIA + weather from APIs (macrOS-Int)
# ---------------------------------------------------------------------------

def refresh_context_from_apis(
    cache_dir: Path,
    lookback_years: int = 3,
    weather_config_dir: Optional[str] = None,
) -> Dict[str, Path]:
    """Fetch fresh EIA storage + population-weighted weather via macrOS-Int.

    Requires ``EIA_API_KEY`` and ``NCEI_TOKEN`` environment variables.
    Returns dict of label -> local cached file path.
    """
    try:
        sys.path.insert(0, os.getenv(
            "MACROSINT_PATH",
            str(Path.home() / "PycharmProjects" / "macrOS-Int"),
        ))
        from MacrOSINT.data.sources.eia.api_tools import NatGasHelper
        from MacrOSINT.models.energy.natgas_storage_forecast import (
            NatGasStorageForecaster,
            fetch_storage_data,
        )
    except ImportError as exc:
        logger.error(
            "macrOS-Int not found. Set MACROSINT_PATH env var or install it. (%s)", exc
        )
        return {}

    from datetime import date, timedelta

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    today = date.today()
    start = today - timedelta(days=lookback_years * 365)

    # --- EIA storage ---
    eia_path = cache_dir / "ng_eia_cache.hdf"
    try:
        logger.info("Fetching EIA storage data [%s -> %s] ...", start, today)
        ng_helper = NatGasHelper()
        storage = fetch_storage_data(ng_helper, start=str(start), end=str(today))
        NatGasStorageForecaster.save_eia_cache(storage=storage, hdf_path=str(eia_path))
        paths["eia"] = eia_path
        logger.info("EIA cache: %d rows -> %s", len(storage), eia_path)
    except Exception as exc:
        logger.error("EIA fetch failed: %s", exc)

    # --- Population-weighted weather ---
    weather_path = cache_dir / "new_weather.hdf"
    try:
        logger.info("Fetching population-weighted weather [%s -> %s] ...", start, today)
        forecaster = NatGasStorageForecaster(
            config_dir=weather_config_dir or str(cache_dir / "weather_configs"),
        )
        daily_weather = forecaster._fetch_weather_by_epoch(start, today)
        NatGasStorageForecaster.save_weather_hdf(daily_weather, hdf_path=str(weather_path))
        paths["weather"] = weather_path
        logger.info("Weather cache: %d rows -> %s", len(daily_weather), weather_path)
    except Exception as exc:
        logger.error("Weather fetch failed: %s", exc)

    return paths


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    device: torch.device,
    config_path: Optional[str] = None,
) -> HybridMixtureNetwork:
    """Load a trained HybridMixtureNetwork from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if "config" in checkpoint:
        cfg = HybridConfig(**checkpoint["config"])
    elif config_path:
        with open(config_path) as f:
            cfg = HybridConfig(**json.load(f))
    else:
        cfg = HybridConfig()
        logger.warning("No config in checkpoint, using HybridConfig defaults")

    feature_group_sizes = checkpoint.get("feature_group_sizes", None)
    model = HybridMixtureNetwork(cfg, feature_group_sizes=feature_group_sizes)

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Loaded HybridMixtureNetwork from %s (%s)", model_path, device)
    return model


# ---------------------------------------------------------------------------
# Single inference cycle
# ---------------------------------------------------------------------------

def run_inference(
    interface: NatGasLiveInterface,
    model: HybridMixtureNetwork,
    device: torch.device,
    task: str = "regression",
    threshold: float = 0.0,
) -> Dict[str, Any]:
    """IBKR ticks + cloud context -> features -> forward pass -> position."""
    now = pd.Timestamp.now(tz=TZ)
    logger.info("=== Inference cycle @ %s ===", now.strftime("%Y-%m-%d %H:%M %Z"))

    # 1. Fetch live ticks from IBKR + merge with daily context -> feature tensors
    snapshot = interface.get_inference_snapshot(now)
    inputs = snapshot.to_model_inputs(add_batch_dim=True)

    # 2. HybridMixtureNetwork forward: (x_seq, ae_input)
    x_seq = inputs["tech_features"].to(device)
    ae_input = inputs["ae_input"].to(device)

    with torch.no_grad():
        out = model(x_seq, ae_input)

    # 3. Extract predictions
    pred_return = out["pred_return"].cpu().numpy()
    pred_std = out["pred_std"].cpu().numpy()

    # 4. Position
    if "class_logits" in out:
        logits = out["class_logits"].cpu().numpy()
        discrete_pos = predictions_to_positions(logits, task="classification")
    else:
        discrete_pos = predictions_to_positions(pred_return, task=task, threshold=threshold)

    continuous_pos = out["position"].cpu().item() if "position" in out else float(discrete_pos[0])

    result = {
        "timestamp": now.isoformat(),
        "anchor_ts": snapshot.anchor_ts.isoformat(),
        "anchor_close": snapshot.anchor_close,
        "pred_return": float(pred_return[0]),
        "pred_std": float(pred_std[0]),
        "position": continuous_pos,
        "discrete_position": float(discrete_pos[0]),
    }

    if "mdn_pred_return" in out:
        result["mdn_pred_return"] = float(out["mdn_pred_return"].cpu().item())
        result["mdn_pred_std"] = float(out["mdn_pred_std"].cpu().item())

    if "z_regime" in out:
        z = out["z_regime"].cpu().numpy()[0]
        result["regime_norm"] = float(np.linalg.norm(z))

    logger.info(
        "pred_return=%.4f  pred_std=%.4f  position=%.3f  discrete=%+.0f  anchor=%.3f",
        result["pred_return"],
        result["pred_std"],
        result["position"],
        result["discrete_position"],
        result["anchor_close"],
    )
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NG inference: IBKR live ticks + cloud daily context -> HybridMixtureNetwork -> position"
    )

    # IBKR (live tick stream)
    parser.add_argument("--ibkr-url", default=os.getenv("IBKR_BASE_URL", "https://localhost:5000"))
    parser.add_argument("--account-id", default=os.getenv("IBKR_ACCOUNT_ID", ""))
    parser.add_argument("--conid", required=True, help="IBKR contract ID for NG front month")

    # Cloud context (EIA, weather, features, spline transformer)
    parser.add_argument("--context-backend", choices=["s3", "gcs", "local"], default="gcs",
                        help="Where to pull daily context files from (default: gcs)")
    parser.add_argument("--context-bucket", default=_GCS_BUCKET)
    parser.add_argument("--s3-endpoint", default=os.getenv("S3_ENDPOINT", ""))
    parser.add_argument("--s3-prefix", default=os.getenv("S3_PREFIX", ""))
    parser.add_argument("--gcs-project", default=_GCS_PROJECT)
    parser.add_argument("--gcs-credentials", default=_GCS_CREDS)

    # Context file keys (cloud) or paths (local) — defaults use model_data/ prefix
    parser.add_argument("--eia-key", default=os.getenv("EIA_CACHE_DIR", _MODEL_DATA_DIR) + "ng_eia_cache.hdf",
                        help="EIA storage cache (S3/GCS key or local path)")
    parser.add_argument("--weather-key", default=os.getenv("WEATHER_CACHE_DIR", _MODEL_DATA_DIR) + "new_weather.hdf",
                        help="Population-weighted weather (S3/GCS key or local path)")
    parser.add_argument("--spline-key", default=None, help="SplineTransformer pickle (S3/GCS key or local path)")
    parser.add_argument("--daily-features-key", default=None, help="Pre-computed daily features (S3/GCS key or local path)")

    # Fresh API fetch (overrides --context-backend for EIA + weather)
    parser.add_argument("--refresh-context", action="store_true",
                        help="Fetch fresh EIA + weather from APIs (needs EIA_API_KEY + NCEI_TOKEN)")
    parser.add_argument("--refresh-lookback-years", type=int, default=3,
                        help="Years of history to fetch when refreshing")
    parser.add_argument("--weather-config-dir", default=None,
                        help="Directory for PopulationWeatherGrid config JSONs")

    # Model — default searches results/ directory
    parser.add_argument("--model-path", default=None,
                        help="HybridMixtureNetwork checkpoint (default: auto-detect from results/)")
    parser.add_argument("--config-path", default=None, help="HybridConfig JSON (if not in checkpoint)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Inference
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--bar-minutes", type=int, default=15)
    parser.add_argument("--lookback-days", type=int, default=10,
                        help="IBKR history lookback (API caps at ~30d)")

    # Execution
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=1800, help="Loop interval seconds (default 30min)")
    parser.add_argument("--output", default=None, help="Append JSON results to this file")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

    # Auto-detect model checkpoint from results/ if not specified
    if args.model_path is None:
        results_dir = Path(_RESULTS_DIR)
        candidates = sorted(results_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        candidates += sorted(results_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            args.model_path = str(candidates[0])
            logger.info("Auto-detected model checkpoint: %s", args.model_path)
        else:
            parser.error(f"--model-path not specified and no .pth/.pt files found in {results_dir}")

    device = torch.device(args.device)

    # --- Resolve daily context: fresh API fetch > cloud download > local paths ---
    eia_path = None
    weather_path = None
    spline_path = args.spline_key if args.context_backend == "local" else None
    daily_features_path = args.daily_features_key if args.context_backend == "local" else None

    if args.refresh_context:
        # Fetch fresh EIA + weather from APIs
        logger.info("Refreshing EIA + weather from APIs (%d year lookback)...", args.refresh_lookback_years)
        refreshed = refresh_context_from_apis(
            CACHE_DIR,
            lookback_years=args.refresh_lookback_years,
            weather_config_dir=args.weather_config_dir,
        )
        eia_path = str(refreshed["eia"]) if "eia" in refreshed else None
        weather_path = str(refreshed["weather"]) if "weather" in refreshed else None

        # Still allow cloud/local overrides for spline + daily features
        if args.spline_key and args.context_backend == "local":
            spline_path = args.spline_key
        if args.daily_features_key and args.context_backend == "local":
            daily_features_path = args.daily_features_key

    if not args.refresh_context or spline_path is None or daily_features_path is None:
        # Fill remaining context from cloud or local
        context_keys = {
            "eia": args.eia_key if eia_path is None else None,
            "weather": args.weather_key if weather_path is None else None,
            "spline": args.spline_key if spline_path is None else None,
            "daily_features": args.daily_features_key if daily_features_path is None else None,
        }
        has_cloud_keys = any(v for v in context_keys.values())

        if args.context_backend == "s3" and has_cloud_keys:
            logger.info("Downloading remaining context from S3 ...")
            downloaded = download_context_s3(
                args.context_bucket, context_keys, args.s3_endpoint, args.s3_prefix,
            )
            eia_path = eia_path or (str(downloaded["eia"]) if "eia" in downloaded else None)
            weather_path = weather_path or (str(downloaded["weather"]) if "weather" in downloaded else None)
            spline_path = spline_path or (str(downloaded["spline"]) if "spline" in downloaded else None)
            daily_features_path = daily_features_path or (str(downloaded["daily_features"]) if "daily_features" in downloaded else None)

        elif args.context_backend == "gcs" and has_cloud_keys:
            logger.info("Downloading remaining context from GCS ...")
            downloaded = download_context_gcs(
                args.context_bucket, context_keys, args.gcs_project, args.gcs_credentials,
            )
            eia_path = eia_path or (str(downloaded["eia"]) if "eia" in downloaded else None)
            weather_path = weather_path or (str(downloaded["weather"]) if "weather" in downloaded else None)
            spline_path = spline_path or (str(downloaded["spline"]) if "spline" in downloaded else None)
            daily_features_path = daily_features_path or (str(downloaded["daily_features"]) if "daily_features" in downloaded else None)

        elif args.context_backend == "local":
            eia_path = eia_path or args.eia_key
            weather_path = weather_path or args.weather_key
            spline_path = spline_path or args.spline_key
            daily_features_path = daily_features_path or args.daily_features_key

    # --- IBKR data source (live ticks) ---
    ibkr_cfg = IBKRConfig(base_url=args.ibkr_url, account_id=args.account_id)
    contracts = {"NG": IBKRContract(conid=args.conid, ticker="NG")}
    data_source = IBKRTickDataSource(config=ibkr_cfg, contracts=contracts, tz=TZ)

    # --- NatGasLiveInterface: IBKR ticks + daily context ---
    live_cfg = ng_default_config(
        bar_minutes=args.bar_minutes,
        history_lookback_days=args.lookback_days,
    )
    daily_paths = DailyContextPaths(
        eia_storage_path=eia_path,
        weather_path=weather_path,
        daily_features_path=daily_features_path,
        spline_transformer_path=spline_path,
    )
    interface = NatGasLiveInterface(live_cfg, data_source, daily_paths=daily_paths)

    # --- Load model ---
    model = load_model(args.model_path, device, config_path=args.config_path)

    # --- Output ---
    out_file = Path(args.output) if args.output else None
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("Ready: IBKR=%s | context=%s | model=%s", args.ibkr_url, args.context_backend, args.model_path)
    if eia_path:
        logger.info("  EIA    : %s", eia_path)
    if weather_path:
        logger.info("  Weather: %s", weather_path)

    last_refresh_date = pd.Timestamp.now(tz=TZ).date()

    # --- Run ---
    while _running:
        # Daily refresh of EIA + weather when looping
        today = pd.Timestamp.now(tz=TZ).date()
        if args.loop and args.refresh_context and today > last_refresh_date:
            logger.info("New trading day — refreshing EIA + weather ...")
            refreshed = refresh_context_from_apis(
                CACHE_DIR,
                lookback_years=args.refresh_lookback_years,
                weather_config_dir=args.weather_config_dir,
            )
            if "eia" in refreshed:
                interface.daily_paths.eia_storage_path = str(refreshed["eia"])
            if "weather" in refreshed:
                interface.daily_paths.weather_path = str(refreshed["weather"])
            interface.reload_daily_caches()
            last_refresh_date = today

        try:
            result = run_inference(interface, model, device, task=args.task, threshold=args.threshold)

            if out_file:
                with open(out_file, "a") as f:
                    f.write(json.dumps(result) + "\n")

            if not args.loop:
                print(json.dumps(result, indent=2))
                break

        except Exception:
            logger.exception("Inference cycle failed")
            if not args.loop:
                sys.exit(1)

        if args.loop and _running:
            logger.info("Next cycle in %ds", args.interval)
            time.sleep(args.interval)

    logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
