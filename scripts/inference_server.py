"""Server-side inference: S3/GCS -> features -> HybridMixtureNetwork -> position.

Runs on RunPod (S3 backend) or Cloud Run (GCS backend). Fetches tick data
from cloud storage, builds feature tensors via NatGasLiveInterface, runs
the HybridMixtureNetwork forward pass, and returns the position signal.

Usage
-----
    # RunPod (S3)
    python scripts/inference_server.py --backend s3 --model-path /workspace/models/ng_hybrid.pt

    # Cloud Run (GCS)
    python scripts/inference_server.py --backend gcs --model-path /tmp/models/ng_hybrid.pt

    # Single-shot (no HTTP server)
    python scripts/inference_server.py --backend s3 --model-path model.pt --once
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

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

logger = logging.getLogger("inference_server")

# ---------------------------------------------------------------------------
# Environment defaults
# ---------------------------------------------------------------------------

# S3 (RunPod)
S3_BUCKET = os.getenv("S3_BUCKET", "ctaflow-prod")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "")
S3_PREFIX = os.getenv("S3_PREFIX", "")

# GCS (Cloud Run)
GCS_BUCKET = os.getenv("GCS_BUCKET", "ctaflow-prod-artifacts")
GCS_PROJECT = os.getenv("GCS_PROJECT", "")
GCS_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

LIVE_DATA_KEY = os.getenv("LIVE_DATA_KEY", "live_data/NG_latest.tar.gz")
TZ = "America/Chicago"


# ---------------------------------------------------------------------------
# Data source factories
# ---------------------------------------------------------------------------

def build_s3_source():
    from live_cta.storage.aws_client import S3Config
    from live_cta.sources.s3_tick_source import S3TickerSpec, S3TickDataSource

    cfg = S3Config(
        bucket_name=S3_BUCKET,
        endpoint_url=S3_ENDPOINT or None,
        prefix=S3_PREFIX,
    )
    return S3TickDataSource(
        s3_config=cfg,
        ticker_map={"NG": S3TickerSpec(LIVE_DATA_KEY)},
        tz=TZ,
    )


def _detect_format(key: str) -> str:
    """Infer data format from the GCS object key extension."""
    if key.endswith(".tar.gz"):
        return "tar.gz"
    elif key.endswith(".csv"):
        return "csv"
    return "parquet"


def build_gcs_source():
    from live_cta.sources.gcs_tick_source import GCSConfig, GCSTickerSpec, GCSTickDataSource

    cfg = GCSConfig(
        bucket_name=GCS_BUCKET,
        project=GCS_PROJECT or None,
        credentials_path=GCS_CREDENTIALS or None,
    )
    fmt = _detect_format(LIVE_DATA_KEY)
    return GCSTickDataSource(
        config=cfg,
        ticker_map={"NG": GCSTickerSpec(LIVE_DATA_KEY, fmt=fmt)},
        tz=TZ,
    )


# ---------------------------------------------------------------------------
# GCS artifact download
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.getenv("CTAFLOW_CACHE_DIR", Path.home() / ".ctaflow" / "live_context"))


def resolve_gcs_path(path_or_key: str) -> str:
    """If *path_or_key* doesn't exist locally, try downloading from GCS.

    Accepts either a local path (returned as-is if it exists) or a GCS key
    like ``results/ng_hybrid_intraday_best.pth``. Downloaded files are cached
    under ``CACHE_DIR``.
    """
    if Path(path_or_key).exists():
        return path_or_key

    # Treat as a GCS key — download to local cache
    try:
        from google.cloud import storage as gcs_storage
    except ImportError:
        return path_or_key  # can't download, let caller fail with FileNotFoundError

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local = CACHE_DIR / Path(path_or_key).name

    if local.exists():
        logger.info("Using cached %s", local)
        return str(local)

    logger.info("Downloading gs://%s/%s -> %s", GCS_BUCKET, path_or_key, local)
    kwargs = {}
    if GCS_PROJECT:
        kwargs["project"] = GCS_PROJECT
    if GCS_CREDENTIALS:
        from google.oauth2 import service_account
        kwargs["credentials"] = service_account.Credentials.from_service_account_file(GCS_CREDENTIALS)
    client = gcs_storage.Client(**kwargs)
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(path_or_key)
    blob.download_to_filename(str(local))
    logger.info("Downloaded %.1f KB", local.stat().st_size / 1024)
    return str(local)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    device: torch.device,
    config_path: Optional[str] = None,
) -> HybridMixtureNetwork:
    """Load a trained HybridMixtureNetwork from checkpoint."""
    model_path = resolve_gcs_path(model_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Support both raw state_dict and full checkpoint format
    if "config" in checkpoint:
        cfg = HybridConfig(**checkpoint["config"])
    elif config_path:
        with open(config_path) as f:
            cfg = HybridConfig(**json.load(f))
    else:
        cfg = HybridConfig()
        logger.warning("No config found in checkpoint, using HybridConfig defaults")

    # Extract feature group sizes: checkpoint stores either
    # "feature_group_sizes" (list) or "feature_groups" (dict of name -> cols)
    feature_group_sizes = checkpoint.get("feature_group_sizes", None)
    if feature_group_sizes is None and "feature_groups" in checkpoint:
        # Preserve as dict {group_name: n_features} for GroupedFeatureVSN
        feature_group_sizes = {
            name: len(cols) for name, cols in checkpoint["feature_groups"].items()
        }
        logger.info("Derived feature_group_sizes from feature_groups: %s", feature_group_sizes)
    model = HybridMixtureNetwork(cfg, feature_group_sizes=feature_group_sizes)

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Loaded HybridMixtureNetwork from %s", model_path)
    return model


# ---------------------------------------------------------------------------
# Inference cycle
# ---------------------------------------------------------------------------

def run_inference(
    interface: NatGasLiveInterface,
    model: HybridMixtureNetwork,
    device: torch.device,
    task: str = "regression",
    threshold: float = 0.0,
) -> Dict[str, Any]:
    """Run a single inference cycle: fetch data -> build features -> forward pass -> position."""
    now = pd.Timestamp.now(tz=TZ)
    logger.info("Running inference at %s", now)

    # 1. Build feature snapshot (fetches from S3/GCS internally)
    snapshot = interface.get_inference_snapshot(now)
    inputs = snapshot.to_model_inputs(add_batch_dim=True)

    # 2. Forward pass — HybridMixtureNetwork expects (x_seq, ae_input)
    x_seq = inputs["tech_features"].to(device)
    ae_input = inputs["ae_input"].to(device)

    with torch.no_grad():
        out = model(x_seq, ae_input)

    # 3. Extract prediction
    pred_return = out["pred_return"].cpu().numpy()
    pred_std = out["pred_std"].cpu().numpy()

    # 4. Convert to position
    if "class_logits" in out:
        logits = out["class_logits"].cpu().numpy()
        position = predictions_to_positions(logits, task="classification")
    else:
        position = predictions_to_positions(pred_return, task=task, threshold=threshold)

    # 5. Optional: if model has positioning head, use that directly
    if "position" in out:
        raw_position = out["position"].cpu().numpy().item()
    else:
        raw_position = float(position[0])

    result = {
        "timestamp": now.isoformat(),
        "anchor_ts": snapshot.anchor_ts.isoformat(),
        "anchor_close": snapshot.anchor_close,
        "pred_return": float(pred_return[0]),
        "pred_std": float(pred_std[0]),
        "position": raw_position,
        "discrete_position": float(position[0]),
    }

    # Include MDN density info
    if "mdn_pred_return" in out:
        result["mdn_pred_return"] = float(out["mdn_pred_return"].cpu().item())
        result["mdn_pred_std"] = float(out["mdn_pred_std"].cpu().item())

    logger.info(
        "Inference result: pred_return=%.4f, pred_std=%.4f, position=%.1f",
        result["pred_return"], result["pred_std"], result["discrete_position"],
    )
    return result


# ---------------------------------------------------------------------------
# FastAPI server (optional, for scheduled triggers)
# ---------------------------------------------------------------------------

def create_app(
    interface: NatGasLiveInterface,
    model: HybridMixtureNetwork,
    device: torch.device,
):
    """Create a FastAPI app for HTTP-triggered inference."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
    except ImportError:
        logger.error("FastAPI not installed. Use --once for single-shot mode.")
        sys.exit(1)

    app = FastAPI(title="CTAFlow NG Inference")

    @app.post("/infer")
    def infer():
        try:
            result = run_inference(interface, model, device)
            return JSONResponse(content=result)
        except Exception as exc:
            logger.exception("Inference failed")
            return JSONResponse(content={"error": str(exc)}, status_code=500)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NG inference server")
    parser.add_argument("--backend", choices=["s3", "gcs"], default="s3")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--config-path", default=None, help="Path to HybridConfig JSON (if not in checkpoint)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--eia-path", default=None, help="Path to cached EIA storage data")
    parser.add_argument("--weather-path", default=None, help="Path to cached population-weighted weather (wtd_TAVG)")
    parser.add_argument("--daily-features-path", default=None)
    parser.add_argument("--spline-transformer-path", default=None, help="Path to fitted SplineTransformer pickle")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Build data source
    data_source = build_s3_source() if args.backend == "s3" else build_gcs_source()

    # Build live interface — resolve context paths from GCS if not local
    live_cfg = ng_default_config()
    daily_paths = DailyContextPaths(
        eia_storage_path=resolve_gcs_path(args.eia_path) if args.eia_path else None,
        weather_path=resolve_gcs_path(args.weather_path) if args.weather_path else None,
        daily_features_path=resolve_gcs_path(args.daily_features_path) if args.daily_features_path else None,
        spline_transformer_path=resolve_gcs_path(args.spline_transformer_path) if args.spline_transformer_path else None,
    )
    interface = NatGasLiveInterface(live_cfg, data_source, daily_paths=daily_paths)

    # Load model (auto-downloads from GCS if not local)
    model = load_model(args.model_path, device, config_path=args.config_path)

    if args.once:
        result = run_inference(interface, model, device, task=args.task, threshold=args.threshold)
        print(json.dumps(result, indent=2))
        return

    # Start HTTP server
    app = create_app(interface, model, device)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
