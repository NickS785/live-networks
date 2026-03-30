#!/usr/bin/env python
"""NG live inference server — GCS ticks → features → HybridMixtureNetwork → prediction.

Reads NG tick data uploaded by ``ng_client.py`` from GCS, builds features
via :class:`NatGasLiveInterface`, runs the best Optuna-trained
:class:`HybridMixtureNetwork`, and returns the prediction.

All defaults match the ``ng_hybrid_intraday_optuna.ipynb`` notebook:
  - BAR_MINUTES = 15
  - SESSION = USA (02:30–15:00)
  - SEQ_LEN = 20, AE_WINDOW = 510 bars (10 days × 51 bars/day)
  - n_features = 55, f_ae = 12, n_classes = 4
  - TARGET_HORIZON_BARS = 10 (150 min)
  - VSN feature groups: technical=34, storage=8, weather=13

Model artifacts are pulled from GCS at ``gs://<bucket>/results/``.  The
server auto-detects the newest ``.pth`` checkpoint, or you can pin a
specific file with ``--model-name``.

Usage
-----
::

    # One-shot inference (auto-downloads newest model from GCS)
    python scripts/ng_server.py

    # Pin a specific checkpoint in the bucket
    python scripts/ng_server.py --model-name ng_hybrid_intraday_best.pth

    # Continuous loop every 15 minutes
    python scripts/ng_server.py --loop --interval 900

    # Local model override (skip GCS download)
    python scripts/ng_server.py --model-path /local/path/model.pth

    # FastAPI server mode
    python scripts/ng_server.py --serve --port 8080
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Load dotenv before anything else
from dotenv import load_dotenv

_env_file = os.getenv("DOTENV_PATH", str(Path(__file__).resolve().parent.parent / "env" / "dot.env"))
load_dotenv(_env_file, override=False)

import numpy as np
import pandas as pd
import torch

from CTAFlow.models.deep_learning.multi_branch.ng_moe import (
    HybridConfig,
    HybridMixtureNetwork,
)
from CTAFlow.models.deep_learning.training.backtest import predictions_to_positions
from live_cta.core.live import LiveEvaluationConfig, SessionSpec
from live_cta.core.ng_live import (
    DailyContextPaths,
    NatGasLiveInterface,
)
from live_cta.sources.gcs_tick_source import GCSConfig, GCSTickerSpec, GCSTickDataSource
from live_cta.sources.history_backfill_source import HistoricalBackfillDataSource

logger = logging.getLogger("ng_server")

TZ = "America/Chicago"
_running = True

# ---------------------------------------------------------------------------
# Notebook defaults (ng_hybrid_intraday_optuna.ipynb)
# ---------------------------------------------------------------------------

# Pipeline
BAR_MINUTES = 15
SESSION = SessionSpec("USA", "02:30", "15:00")
TARGET_HORIZON_BARS = 10              # 150 min at 15-min bars
AE_WINDOW_DAYS = 10
BARS_PER_DAY = 51                     # (02:30–15:00) / 15min ≈ 51
AE_WINDOW_BARS = AE_WINDOW_DAYS * BARS_PER_DAY  # 510

# Feature dimensions
N_FEATURES = 55
F_AE = 12
SEQ_LEN = 20
N_CLASSES = 4

# VSN feature group sizes
FEATURE_GROUP_SIZES = {
    "technical": 34,
    "storage": 8,
    "weather": 13,
}

# ---------------------------------------------------------------------------
# Env defaults
# ---------------------------------------------------------------------------

_GCS_BUCKET = os.getenv("GCS_BUCKET", "ctaflow-prod-artifacts")
_GCS_PROJECT = os.getenv("GCS_PROJECT", "")
_GCS_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
_MODEL_DATA_DIR = os.getenv("MODEL_DATA_DIR", "model_data/")
_GCS_RESULTS_PREFIX = os.getenv("GCS_RESULTS_PREFIX", "results/")
_GCS_PREFIX = os.getenv("GCS_TICK_PREFIX", "live_data/")

# Known GCS artifact keys under the bucket
# gs://ctaflow-prod-artifacts/
#   results/*.pth              — model checkpoints (config + weights)
#   model_data/eia_cache.hdf   — EIA weekly storage cache
#   model_data/weather.hdf     — population-weighted daily weather (HDD/CDD)
#   model_data/NG/intraday_2.csv — cached 5-min NG bars (Sierra Chart export)
#   live_data/NG_latest.parquet — rolling front-month ticks (pushed by ng_client)
GCS_EIA_KEY = f"{_MODEL_DATA_DIR}eia_cache.hdf"
GCS_WEATHER_KEY = f"{_MODEL_DATA_DIR}weather.hdf"
GCS_INTRADAY_KEY = f"{_MODEL_DATA_DIR}NG/intraday_2.csv"
CACHE_DIR = Path(os.getenv("CTAFLOW_CACHE_DIR", Path.home() / ".ctaflow" / "live_context"))


def _handle_signal(signum, frame):
    global _running
    logger.info("Signal %s received, shutting down ...", signum)
    _running = False


# ---------------------------------------------------------------------------
# Download model artifacts from GCS
# ---------------------------------------------------------------------------

def download_model_from_gcs(
    gcs_config: GCSConfig,
    results_prefix: str = "results/",
    checkpoint_name: Optional[str] = None,
) -> Path:
    """Download the best model checkpoint from ``gs://<bucket>/<results_prefix>``.

    If *checkpoint_name* is given, downloads that exact file.  Otherwise
    lists objects under the prefix and picks the most recently modified
    ``.pth`` / ``.pt`` file.

    Returns the local path to the downloaded checkpoint.
    """
    try:
        from google.cloud import storage as gcs_storage
    except ImportError:
        raise ImportError("google-cloud-storage required: pip install google-cloud-storage")

    kwargs: dict = {}
    if gcs_config.project:
        kwargs["project"] = gcs_config.project
    if gcs_config.credentials_path:
        from google.oauth2 import service_account
        kwargs["credentials"] = service_account.Credentials.from_service_account_file(
            gcs_config.credentials_path,
        )

    client = gcs_storage.Client(**kwargs)
    bucket = client.bucket(gcs_config.bucket_name)
    cache = CACHE_DIR / "models"
    cache.mkdir(parents=True, exist_ok=True)

    if checkpoint_name:
        gcs_key = f"{results_prefix}{checkpoint_name}"
    else:
        # Auto-detect: list blobs under prefix, pick newest .pth/.pt
        blobs = list(bucket.list_blobs(prefix=results_prefix))
        model_blobs = [
            b for b in blobs
            if b.name.endswith(".pth") or b.name.endswith(".pt")
        ]
        if not model_blobs:
            raise FileNotFoundError(
                f"No .pth/.pt files found under gs://{gcs_config.bucket_name}/{results_prefix}"
            )
        # Sort by updated time, newest first
        model_blobs.sort(key=lambda b: b.updated, reverse=True)
        gcs_key = model_blobs[0].name
        logger.info(
            "Auto-detected model in GCS: gs://%s/%s (updated %s)",
            gcs_config.bucket_name, gcs_key, model_blobs[0].updated,
        )

    local_path = cache / Path(gcs_key).name
    blob = bucket.blob(gcs_key)
    blob.download_to_filename(str(local_path))
    logger.info(
        "Downloaded gs://%s/%s -> %s (%.1f MB)",
        gcs_config.bucket_name, gcs_key, local_path,
        local_path.stat().st_size / 1e6,
    )
    return local_path


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _detect_format(key: str) -> str:
    if key.endswith(".tar.gz"):
        return "tar.gz"
    if key.endswith(".csv"):
        return "csv"
    return "parquet"


def _parse_session(session_meta: Any) -> SessionSpec:
    if isinstance(session_meta, dict):
        return SessionSpec(
            session_meta.get("name", "USA"),
            session_meta.get("start", "02:30"),
            session_meta.get("end", "15:00"),
        )
    if isinstance(session_meta, str) and "_" in session_meta and "-" in session_meta:
        name, times = session_meta.split("_", 1)
        start, end = times.split("-", 1)
        return SessionSpec(name, start, end)
    return SESSION

def load_model(
    model_path: str,
    device: torch.device,
) -> Tuple[HybridMixtureNetwork, Dict[str, Any]]:
    """Load best HybridMixtureNetwork from an Optuna checkpoint.

    Checkpoint keys (from notebook):
        model_state_dict, config, best_params, n_features, feature_cols,
        regime_cols, feature_groups, bar_minutes, target_horizon_bars,
        session, ae_window_bars
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if "config" in checkpoint:
        cfg = HybridConfig(**checkpoint["config"])
    else:
        # Fallback: notebook locked architecture defaults
        cfg = HybridConfig(
            n_features=N_FEATURES,
            seq_len=SEQ_LEN,
            f_ae=F_AE,
            ae_window=AE_WINDOW_BARS,
            d_latent=32,
            d_ae_hidden=256,
            tcn_channels=[64, 64, 64, 64],
            tcn_kernel_size=3,
            stride=2,
            mdn_hidden_dims=[128, 64],
            head_hidden_dim=256,
            n_classes=N_CLASSES,
            use_positioning_head=True,
            positioning_hidden_dim=64,
            use_vsn=True,
        )
        logger.warning("No config in checkpoint — using notebook defaults")

    feature_group_sizes = checkpoint.get("feature_group_sizes")
    if feature_group_sizes is None and cfg.use_vsn:
        feature_group_sizes = FEATURE_GROUP_SIZES
        logger.info("Using default feature_group_sizes: %s", feature_group_sizes)

    model = HybridMixtureNetwork(cfg, feature_group_sizes=feature_group_sizes)

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Log checkpoint metadata if available
    if "bar_minutes" in checkpoint:
        logger.info(
            "Checkpoint: bar=%dmin  horizon=%d bars  ae_window=%d  session=%s",
            checkpoint.get("bar_minutes", BAR_MINUTES),
            checkpoint.get("target_horizon_bars", TARGET_HORIZON_BARS),
            checkpoint.get("ae_window_bars", AE_WINDOW_BARS),
            checkpoint.get("session", asdict(SESSION)),
        )
    logger.info("Loaded HybridMixtureNetwork from %s (%s)", model_path, device)
    checkpoint_meta = {
        "feature_cols": checkpoint.get("feature_cols"),
        "regime_cols": checkpoint.get("regime_cols"),
        "feature_groups": checkpoint.get("feature_groups"),
        "bar_minutes": checkpoint.get("bar_minutes", BAR_MINUTES),
        "target_horizon_bars": checkpoint.get("target_horizon_bars", TARGET_HORIZON_BARS),
        "session": checkpoint.get("session", asdict(SESSION)),
        "ae_window_bars": checkpoint.get("ae_window_bars", AE_WINDOW_BARS),
    }
    return model, checkpoint_meta


# ---------------------------------------------------------------------------
# GCS context download
# ---------------------------------------------------------------------------

def download_context_gcs(
    gcs_config: GCSConfig,
    keys: Dict[str, str],
) -> Dict[str, Path]:
    """Download EIA/weather context files from GCS to local cache."""
    try:
        from google.cloud import storage as gcs_storage
    except ImportError:
        logger.error("google-cloud-storage not installed")
        return {}

    kwargs = {}
    if gcs_config.project:
        kwargs["project"] = gcs_config.project
    if gcs_config.credentials_path:
        from google.oauth2 import service_account
        kwargs["credentials"] = service_account.Credentials.from_service_account_file(
            gcs_config.credentials_path,
        )

    client = gcs_storage.Client(**kwargs)
    bucket = client.bucket(gcs_config.bucket_name)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    local_paths = {}
    for label, gcs_key in keys.items():
        if not gcs_key:
            continue
        local = CACHE_DIR / Path(gcs_key).name
        try:
            blob = bucket.blob(gcs_key)
            blob.download_to_filename(str(local))
            local_paths[label] = local
            logger.info("GCS context -> %s : %s", label, local)
        except Exception as exc:
            logger.warning("Failed to download %s from GCS: %s", gcs_key, exc)
    return local_paths


# ---------------------------------------------------------------------------
# Single inference cycle
# ---------------------------------------------------------------------------

def run_inference(
    interface: NatGasLiveInterface,
    model: HybridMixtureNetwork,
    device: torch.device,
) -> Dict[str, Any]:
    """GCS ticks → features → HybridMixtureNetwork → prediction dict."""
    now = pd.Timestamp.now(tz=TZ)
    logger.info("=== Inference cycle @ %s ===", now.strftime("%Y-%m-%d %H:%M %Z"))

    # 1. Build feature snapshot from GCS tick data + daily context
    snapshot = interface.get_inference_snapshot(now)
    inputs = snapshot.to_model_inputs(add_batch_dim=True)

    # 2. Forward pass: (x_seq, ae_input)
    x_seq = inputs["tech_features"].to(device)
    ae_input = inputs["ae_input"].to(device)

    with torch.no_grad():
        out = model(x_seq, ae_input)

    # 3. Extract predictions
    pred_return = out["pred_return"].cpu().item()
    pred_std = out["pred_std"].cpu().item()

    # 4. Position signal
    if "position" in out:
        position = out["position"].cpu().item()
    else:
        position = float(predictions_to_positions(
            np.array([[pred_return]]), task="regression",
        )[0])

    # 5. Classification
    pred_class = None
    if "class_logits" in out:
        pred_class = int(out["class_logits"].argmax(dim=-1).cpu().item())

    result: Dict[str, Any] = {
        "timestamp": now.isoformat(),
        "anchor_ts": snapshot.anchor_ts.isoformat(),
        "anchor_close": snapshot.anchor_close,
        "pred_return": pred_return,
        "pred_std": pred_std,
        "position": position,
        "pred_class": pred_class,
    }

    # MDN mixture prediction
    if "mdn_pred_return" in out:
        result["mdn_pred_return"] = float(out["mdn_pred_return"].cpu().item())
        result["mdn_pred_std"] = float(out["mdn_pred_std"].cpu().item())

    # Regime latent norm
    if "z_regime" in out:
        z = out["z_regime"].cpu().numpy()[0]
        result["regime_norm"] = float(np.linalg.norm(z))

    # VSN attention weights
    if "vsn_weights" in out:
        w = out["vsn_weights"].cpu().numpy()[0]
        for i, name in enumerate(FEATURE_GROUP_SIZES):
            result[f"vsn_{name}"] = float(w[i])

    logger.info(
        "pred_ret=%.4f  std=%.4f  pos=%.3f  class=%s  anchor=%.3f",
        pred_return, pred_std, position,
        pred_class, result["anchor_close"],
    )
    return result


# ---------------------------------------------------------------------------
# FastAPI server mode
# ---------------------------------------------------------------------------

def run_fastapi_server(
    interface: NatGasLiveInterface,
    model: HybridMixtureNetwork,
    device: torch.device,
    host: str,
    port: int,
):
    """Launch a FastAPI HTTP server for on-demand inference."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import uvicorn
    except ImportError:
        logger.error("FastAPI/uvicorn not installed. pip install 'live-cta[server]'")
        sys.exit(1)

    app = FastAPI(title="NG Hybrid Inference Server")

    @app.get("/predict")
    def predict():
        try:
            result = run_inference(interface, model, device)
            return JSONResponse(content=result)
        except Exception as exc:
            logger.exception("Inference failed")
            return JSONResponse(content={"error": str(exc)}, status_code=500)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": "HybridMixtureNetwork",
            "device": str(device),
            "artifacts": {
                "eia": interface.daily_paths.eia_storage_path,
                "weather": interface.daily_paths.weather_path,
                "intraday_cache": getattr(interface, "_cached_intraday_csv", None),
            },
        }

    logger.info("Starting FastAPI server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NG inference server: GCS ticks -> HybridMixtureNetwork -> prediction",
    )

    # Model — downloaded from GCS results/ by default
    parser.add_argument("--model-path", default=None,
                        help="Local .pth path (skips GCS download if set)")
    parser.add_argument("--model-name", default=None,
                        help="Checkpoint filename in GCS results/ (e.g. ng_hybrid_best.pth). "
                             "If omitted, picks the newest .pth/.pt in the bucket.")
    parser.add_argument("--gcs-results-prefix", default=_GCS_RESULTS_PREFIX,
                        help="GCS prefix for model artifacts (default: results/)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # GCS tick data source
    parser.add_argument("--gcs-bucket", default=_GCS_BUCKET)
    parser.add_argument("--gcs-project", default=_GCS_PROJECT)
    parser.add_argument("--gcs-credentials", default=_GCS_CREDS)
    parser.add_argument("--gcs-tick-key", default=f"{_GCS_PREFIX}NG_latest.tar.gz",
                        help="GCS object path for the latest NG live segment")

    # Daily context — defaults point to known GCS artifacts
    parser.add_argument("--context-backend", choices=["gcs", "local"], default="gcs",
                        help="Where to load EIA/weather context from")
    parser.add_argument("--eia-key", default=GCS_EIA_KEY,
                        help="EIA storage cache (GCS key or local path)")
    parser.add_argument("--weather-key", default=GCS_WEATHER_KEY,
                        help="Weather cache (GCS key or local path)")
    parser.add_argument("--intraday-key", default=GCS_INTRADAY_KEY,
                        help="Cached 5-min NG intraday CSV in GCS (for history backfill)")

    # Pipeline overrides (defaults match notebook)
    parser.add_argument("--bar-minutes", type=int, default=BAR_MINUTES)
    parser.add_argument("--ae-window", type=int, default=AE_WINDOW_DAYS,
                        help="AE regime lookback in trading days")
    parser.add_argument("--lookback-days", type=int, default=400,
                        help="History lookback for feature building")

    # Execution mode
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=900,
                        help="Inference interval in seconds (default 15 min)")
    parser.add_argument("--serve", action="store_true", help="Run as FastAPI HTTP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output", default=None,
                        help="Append JSON results to this file")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

    device = torch.device(args.device)

    # --- GCS config (shared by tick source, context, and model download) ---
    gcs_config = GCSConfig(
        bucket_name=args.gcs_bucket,
        project=args.gcs_project or None,
        credentials_path=args.gcs_credentials or None,
    )

    # --- Download model from GCS (or use local override) ---
    if args.model_path:
        model_path = args.model_path
        logger.info("Using local model: %s", model_path)
    else:
        logger.info("Downloading model from gs://%s/%s ...", args.gcs_bucket, args.gcs_results_prefix)
        model_path = str(download_model_from_gcs(
            gcs_config,
            results_prefix=args.gcs_results_prefix,
            checkpoint_name=args.model_name,
        ))

    # --- Load model metadata and weights ---
    model, checkpoint_meta = load_model(model_path, device)
    checkpoint_bar_minutes = int(checkpoint_meta.get("bar_minutes") or args.bar_minutes)
    checkpoint_horizon_bars = int(
        checkpoint_meta.get("target_horizon_bars") or TARGET_HORIZON_BARS
    )
    checkpoint_session = _parse_session(checkpoint_meta.get("session"))
    checkpoint_feature_cols = checkpoint_meta.get("feature_cols") or None
    live_fmt = _detect_format(args.gcs_tick_key)
    gcs_source = GCSTickDataSource(
        config=gcs_config,
        ticker_map={"NG": GCSTickerSpec(args.gcs_tick_key, fmt=live_fmt)},
    )

    # --- Resolve daily context + cached intraday from GCS ---
    eia_path = None
    weather_path = None
    intraday_path = None

    if args.context_backend == "gcs":
        logger.info("Downloading context artifacts from GCS ...")
        downloaded = download_context_gcs(gcs_config, {
            "eia": args.eia_key,
            "weather": args.weather_key,
            "intraday": args.intraday_key,
        })
        eia_path = str(downloaded["eia"]) if "eia" in downloaded else None
        weather_path = str(downloaded["weather"]) if "weather" in downloaded else None
        intraday_path = str(downloaded["intraday"]) if "intraday" in downloaded else None
    else:
        eia_path = args.eia_key if Path(args.eia_key).exists() else None
        weather_path = args.weather_key if Path(args.weather_key).exists() else None
        intraday_path = args.intraday_key if Path(args.intraday_key).exists() else None

    if intraday_path:
        logger.info("  Intraday (5min cache): %s", intraday_path)
        data_source = HistoricalBackfillDataSource(
            gcs_source,
            history_map={"NG": intraday_path},
            tz=TZ,
        )
    else:
        data_source = gcs_source

    # --- NatGasLiveInterface with notebook-matched config ---
    live_cfg = LiveEvaluationConfig(
        ticker="NG",
        tick_size=0.001,
        tz=TZ,
        bar_minutes=checkpoint_bar_minutes,
        target_horizon_minutes=checkpoint_bar_minutes * checkpoint_horizon_bars,
        refresh_interval=f"{checkpoint_bar_minutes}min",
        history_lookback_days=args.lookback_days,
        sessions=[checkpoint_session],
        vpin_bucket_volume=150,
        vpin_window=60,
        vpin_start_time=checkpoint_session.start,
        vpin_end_time=checkpoint_session.end,
        profile_start_time="02:00",
        profile_end_time="09:30",
        ae_window=args.ae_window,
        ae_target_time="10:00",
    )

    daily_paths = DailyContextPaths(
        eia_storage_path=eia_path,
        weather_path=weather_path,
    )

    interface = NatGasLiveInterface(
        config=live_cfg,
        data_source=data_source,
        daily_paths=daily_paths,
        ae_window=args.ae_window,
        feature_cols=checkpoint_feature_cols,
    )

    # Stash cached intraday path on the interface for future backfill use.
    # NatGasLiveInterface doesn't consume this directly — it's available for
    # callers that need to seed history from the 5-min CSV when GCS live_data
    # doesn't have enough lookback (e.g. first deploy or after a long outage).
    interface._cached_intraday_csv = intraday_path  # type: ignore[attr-defined]

    # --- Output ---
    out_file = Path(args.output) if args.output else None
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(
        "NG Server ready: gcs=gs://%s/%s (%s) | model=%s | bar=%dmin | ae=%dd",
        args.gcs_bucket, args.gcs_tick_key, live_fmt, model_path,
        checkpoint_bar_minutes, args.ae_window,
    )
    if eia_path:
        logger.info("  EIA      : %s", eia_path)
    if weather_path:
        logger.info("  Weather  : %s", weather_path)
    if intraday_path:
        logger.info("  Intraday : %s (5-min cache for history backfill)", intraday_path)

    # --- FastAPI server mode ---
    if args.serve:
        run_fastapi_server(interface, model, device, args.host, args.port)
        return

    # --- Loop / one-shot mode ---
    last_context_date = pd.Timestamp.now(tz=TZ).date()

    while _running:
        # Refresh daily context on new trading day
        today = pd.Timestamp.now(tz=TZ).date()
        if args.loop and args.context_backend == "gcs" and today > last_context_date:
            logger.info("New trading day — refreshing context from GCS ...")
            downloaded = download_context_gcs(gcs_config, {
                "eia": args.eia_key,
                "weather": args.weather_key,
                "intraday": args.intraday_key,
            })
            if "eia" in downloaded:
                interface.daily_paths.eia_storage_path = str(downloaded["eia"])
            if "weather" in downloaded:
                interface.daily_paths.weather_path = str(downloaded["weather"])
            if "intraday" in downloaded:
                interface._cached_intraday_csv = str(downloaded["intraday"])  # type: ignore[attr-defined]
            interface.reload_daily_caches()
            last_context_date = today

        try:
            result = run_inference(interface, model, device)

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
            logger.info("Next inference in %ds", args.interval)
            time.sleep(args.interval)

    logger.info("Server shutdown complete.")


if __name__ == "__main__":
    main()
