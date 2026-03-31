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
from CTAFlow.models.prep.intraday_continuous import ContinuousIntradayPrep, SessionSpec
from live_cta.core.ng_live import (
    DailyContextPaths,
    NatGasLiveInterface,
    ng_default_config,
    process_weather_features,
    _normalize_dt_index,
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


def _parse_session(session_meta: Any) -> SessionSpec:
    """Normalize checkpoint session metadata into a SessionSpec."""
    if isinstance(session_meta, SessionSpec):
        return session_meta
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
    return SessionSpec("USA", "02:30", "15:00")


def _ensure_intraday_cst_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure intraday data is indexed in America/Chicago."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize(TZ)
    else:
        df.index = df.index.tz_convert(TZ)
    return df


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

def build_hybrid_inputs(
    bars_df: pd.DataFrame,
    eia_df: Optional[pd.DataFrame],
    weather_df: Optional[pd.DataFrame],
    checkpoint: Dict[str, Any],
    spline_transformer=None,
) -> Dict[str, Any]:
    """Replicate the notebook feature pipeline for HybridMixtureNetwork.

    Follows ``ng_hybrid_intraday_optuna.ipynb`` exactly:
      1. ContinuousIntradayPrep -> tech features
      2. Forward-fill storage features onto intraday bars
      3. Forward-fill weather features onto intraday bars
      4. Build 12-dim regime features from daily aggregation
      5. Window into (x_seq, ae_input) matching IntradayHybridDataset

    Returns dict with 'x_seq', 'ae_input' tensors and metadata.
    """
    # --- Extract checkpoint metadata ---
    feature_cols = checkpoint.get("feature_cols", [])
    feature_groups = checkpoint.get("feature_groups", {})
    regime_cols = checkpoint.get("regime_cols", [])
    bar_minutes = checkpoint.get("bar_minutes", 15)
    ae_window_bars = checkpoint.get("ae_window_bars", 510)
    seq_len = checkpoint["config"].get("seq_len", 20)

    session_spec = _parse_session(checkpoint.get("session"))

    target_horizon_bars = checkpoint.get("target_horizon_bars", 10)

    # --- Step 1: ContinuousIntradayPrep for base tech features ---
    bars_df = _ensure_intraday_cst_index(bars_df.copy())

    prep = ContinuousIntradayPrep(
        sessions=[session_spec],
        bar_minutes=bar_minutes,
    )

    df_prep, _, _ = prep.prepare(
        bars_df.copy(),
        steps_60m=target_horizon_bars,
        keep_only_active=False,
        add_daily=True,
        add_overnight=True,
        add_deseas=True,
        add_time_features=True,
        add_resample_precalc=False,
        apply_scaling=False,
        add_bid_ask="bidvol" in bars_df.columns or "askvol" in bars_df.columns,
        add_event_markers=False,
    )
    df_prep = _ensure_intraday_cst_index(df_prep)
    df_prep.columns = [c.lower() for c in df_prep.columns]

    tech_feature_cols = feature_groups.get("technical", [])
    storage_feature_cols = feature_groups.get("storage", [])
    weather_feature_cols = feature_groups.get("weather", [])

    # --- Step 2: Forward-fill storage features ---
    bar_dates = df_prep.index.normalize()
    bar_dates_naive = bar_dates.tz_localize(None) if bar_dates.tz else bar_dates
    if eia_df is not None and not eia_df.empty:
        sw = _normalize_dt_index(eia_df.copy())
        if "storage_level" in sw.columns:
            sl = sw["storage_level"]
            sw["sl_4wk_mean"] = sl.rolling(4, min_periods=2).mean()
            sw["sl_4wk_max"] = sl.rolling(4, min_periods=2).max()
            sw["sl_4wk_min"] = sl.rolling(4, min_periods=2).min()
            if "storage_change" in sw.columns:
                sw["sl_change_4wk_mean"] = sw["storage_change"].rolling(4, min_periods=2).mean()

        s_cols = [c for c in storage_feature_cols if c in sw.columns]
        if s_cols:
            daily_idx = pd.date_range(sw.index[0], bar_dates_naive[-1], freq="D")
            storage_daily = sw[s_cols].reindex(
                sw.index.union(daily_idx)
            ).sort_index().ffill()
            for col in s_cols:
                df_prep[col] = storage_daily[col].reindex(bar_dates_naive).values
        # Zero-fill any missing storage cols
        for col in storage_feature_cols:
            if col not in df_prep.columns:
                df_prep[col] = 0.0

    else:
        for col in storage_feature_cols:
            df_prep[col] = 0.0
        logger.warning("No EIA data — zero-filling storage features")

    # --- Step 3: Forward-fill weather features ---
    if weather_df is not None and not weather_df.empty:
        wx = _normalize_dt_index(weather_df.copy())
        wx_features, _ = process_weather_features(wx, spline_transformer=spline_transformer)
        wx_features = _normalize_dt_index(wx_features)
        for col in wx_features.columns:
            df_prep[col] = wx_features[col].reindex(bar_dates_naive).values
        # Zero-fill any missing weather cols
        for col in weather_feature_cols:
            if col not in df_prep.columns:
                df_prep[col] = 0.0
    else:
        for col in weather_feature_cols:
            df_prep[col] = 0.0
        logger.warning("No weather data — zero-filling weather features")

    # --- Step 4: Build 12-dim regime features ---
    daily_close = df_prep.groupby(df_prep.index.date)["close"].last()
    daily_close.index = pd.DatetimeIndex(daily_close.index)
    daily_log_ret = np.log(daily_close / daily_close.shift(1))

    regime_daily = pd.DataFrame(index=daily_close.index)
    regime_daily["regime_ret_1d"] = daily_log_ret
    regime_daily["regime_ret_5d"] = daily_log_ret.rolling(5).sum()
    regime_daily["regime_ret_21d"] = daily_log_ret.rolling(21).sum()
    regime_daily["regime_rv_5d"] = np.sqrt((daily_log_ret ** 2).rolling(5).mean()) * np.sqrt(252)
    regime_daily["regime_rv_21d"] = np.sqrt((daily_log_ret ** 2).rolling(21).mean()) * np.sqrt(252)

    # Storage-state regime features (5-year band per ISO week)
    if eia_df is not None and not eia_df.empty and "storage_level" in eia_df.columns:
        sw = _normalize_dt_index(eia_df.copy())
        sl = sw["storage_level"]
        wk_idx = sl.index.isocalendar().week.values

        hi_5y = pd.Series(np.nan, index=sl.index)
        lo_5y = pd.Series(np.nan, index=sl.index)
        mean_5y = pd.Series(np.nan, index=sl.index)

        for w in range(1, 54):
            mask = wk_idx == w
            if mask.sum() < 2:
                continue
            idx_pos = np.where(mask)[0]
            vals = sl.iloc[idx_pos]
            hi_5y.iloc[idx_pos] = vals.expanding().max().shift(1).values
            lo_5y.iloc[idx_pos] = vals.expanding().min().shift(1).values
            mean_5y.iloc[idx_pos] = vals.expanding().mean().shift(1).values

        hi_5y, lo_5y, mean_5y = hi_5y.ffill().bfill(), lo_5y.ffill().bfill(), mean_5y.ffill().bfill()
        band_w = (hi_5y - lo_5y).clip(lower=1)

        regime_wkly = pd.DataFrame({
            "regime_pct_in_5y_band": ((sl - lo_5y) / band_w).values,
            "regime_dev_5y_zscore": ((sl - mean_5y) / band_w).values,
            "regime_band_width_pct": (band_w / mean_5y.clip(lower=1)).values,
        }, index=sl.index)
        regime_wkly_daily = regime_wkly.reindex(
            regime_wkly.index.union(daily_close.index)
        ).sort_index().ffill().reindex(daily_close.index)
        for col in regime_wkly_daily.columns:
            regime_daily[col] = regime_wkly_daily[col].values

        # Forecast vs seasonal
        sc = sw["storage_change"] if "storage_change" in sw.columns else pd.Series(0, index=sw.index)
        sea_chg = pd.Series(np.nan, index=sc.index)
        for w in range(1, 54):
            mask = wk_idx == w
            if mask.sum() < 2:
                continue
            idx_pos = np.where(mask)[0]
            sea_chg.iloc[idx_pos] = sc.iloc[idx_pos].expanding().mean().shift(1).values
        sea_chg = sea_chg.ffill().bfill()
        sea_std = (sc - sea_chg).expanding().std().clip(lower=1)
        chg_vs_sea = (sc - sea_chg) / sea_std
        fc = sw["consensus_est"] if "consensus_est" in sw.columns else sea_chg
        fc_vs_sea = (fc - sea_chg) / sea_std

        fc_regime = pd.DataFrame({
            "regime_fc_vs_seasonal_z": fc_vs_sea.values,
            "regime_chg_vs_seasonal_z": chg_vs_sea.values,
        }, index=sc.index)
        fc_daily = fc_regime.reindex(
            fc_regime.index.union(daily_close.index)
        ).sort_index().ffill().reindex(daily_close.index)
        for col in fc_daily.columns:
            regime_daily[col] = fc_daily[col].values
    else:
        for col in ["regime_pct_in_5y_band", "regime_dev_5y_zscore",
                     "regime_band_width_pct", "regime_fc_vs_seasonal_z",
                     "regime_chg_vs_seasonal_z"]:
            regime_daily[col] = 0.0

    # Season features
    month = regime_daily.index.month
    regime_daily["regime_is_injection"] = ((month >= 4) & (month <= 10)).astype(np.float32)
    season_sign = np.where(regime_daily["regime_is_injection"].values > 0.5, 1.0, -1.0)
    regime_daily["regime_dev_x_season"] = regime_daily.get(
        "regime_dev_5y_zscore", pd.Series(0, index=regime_daily.index)
    ) * season_sign

    # Forward-fill regime to intraday bars
    for col in regime_cols:
        if col in regime_daily.columns:
            df_prep[col] = regime_daily[col].reindex(bar_dates_naive).values
        else:
            df_prep[col] = 0.0

    # --- Step 5: Fill NaNs and window ---
    all_cols = feature_cols + regime_cols
    for col in all_cols:
        if col not in df_prep.columns:
            df_prep[col] = 0.0
    df_prep[all_cols] = df_prep[all_cols].ffill().bfill().fillna(0.0)

    X = df_prep[feature_cols].values.astype(np.float32)
    R = df_prep[regime_cols].values.astype(np.float32)

    # Take the last valid window
    t = len(df_prep) - 1  # anchor = last bar
    x_start = max(0, t - seq_len + 1)
    ae_start = max(0, t - ae_window_bars + 1)

    x_seq = X[x_start: t + 1]
    ae_input = R[ae_start: t + 1]

    # Pad if insufficient history
    if x_seq.shape[0] < seq_len:
        pad = np.zeros((seq_len - x_seq.shape[0], x_seq.shape[1]), dtype=np.float32)
        x_seq = np.concatenate([pad, x_seq], axis=0)
    if ae_input.shape[0] < ae_window_bars:
        pad = np.zeros((ae_window_bars - ae_input.shape[0], ae_input.shape[1]), dtype=np.float32)
        ae_input = np.concatenate([pad, ae_input], axis=0)

    # Replace NaN/inf
    x_seq = np.nan_to_num(x_seq, nan=0.0, posinf=0.0, neginf=0.0)
    ae_input = np.nan_to_num(ae_input, nan=0.0, posinf=0.0, neginf=0.0)

    anchor_ts = df_prep.index[-1]
    anchor_close = float(df_prep["close"].iloc[-1])

    logger.info(
        "Built hybrid inputs: x_seq=%s, ae_input=%s, anchor=%s, close=%.3f",
        x_seq.shape, ae_input.shape, anchor_ts, anchor_close,
    )

    return {
        "x_seq": torch.from_numpy(x_seq).unsqueeze(0),       # (1, seq_len, n_features)
        "ae_input": torch.from_numpy(ae_input).unsqueeze(0),  # (1, ae_window_bars, 12)
        "anchor_ts": anchor_ts,
        "anchor_close": anchor_close,
        "n_bars": len(df_prep),
    }


def run_inference(
    data_source,
    model: HybridMixtureNetwork,
    checkpoint: Dict[str, Any],
    device: torch.device,
    eia_df: Optional[pd.DataFrame] = None,
    weather_df: Optional[pd.DataFrame] = None,
    spline_transformer=None,
    task: str = "regression",
    threshold: float = 0.0,
) -> Dict[str, Any]:
    """Run a single inference cycle matching the notebook pipeline exactly.

    1. Fetch bars from data source (GCS/S3)
    2. Build windowed (x_seq, ae_input) via ContinuousIntradayPrep + regime features
    3. Forward pass through HybridMixtureNetwork
    4. Convert to position signal
    """
    now = pd.Timestamp.now(tz=TZ)
    logger.info("Running inference at %s", now)

    # 1. Fetch bars
    lookback = pd.Timedelta(days=400)  # need history for regime features
    bars_df = data_source.get_ticks("NG", now - lookback, now)
    if bars_df.empty:
        raise RuntimeError("No bar data returned from data source")
    logger.info("Fetched %d bars [%s -> %s]", len(bars_df), bars_df.index[0], bars_df.index[-1])

    # 2. Build windowed inputs (replicates notebook pipeline)
    inputs = build_hybrid_inputs(
        bars_df, eia_df, weather_df, checkpoint,
        spline_transformer=spline_transformer,
    )

    # 3. Forward pass
    x_seq = inputs["x_seq"].to(device)
    ae_input = inputs["ae_input"].to(device)

    logger.info("x_seq shape: %s, ae_input shape: %s", x_seq.shape, ae_input.shape)

    with torch.no_grad():
        out = model(x_seq, ae_input)

    # 4. Extract prediction
    pred_return = out["pred_return"].cpu().numpy()
    pred_std = out["pred_std"].cpu().numpy()

    # 5. Convert to position
    if "class_logits" in out:
        logits = out["class_logits"].cpu().numpy()
        position = predictions_to_positions(logits, task="classification")
    else:
        position = predictions_to_positions(pred_return, task=task, threshold=threshold)

    if "position" in out:
        raw_position = out["position"].cpu().numpy().item()
    else:
        raw_position = float(position[0])

    result = {
        "timestamp": now.isoformat(),
        "anchor_ts": inputs["anchor_ts"].isoformat() if hasattr(inputs["anchor_ts"], "isoformat") else str(inputs["anchor_ts"]),
        "anchor_close": inputs["anchor_close"],
        "pred_return": float(pred_return[0]),
        "pred_std": float(pred_std[0]),
        "position": raw_position,
        "discrete_position": float(position[0]),
        "n_bars": inputs["n_bars"],
    }

    if "mdn_pred_return" in out:
        result["mdn_pred_return"] = float(out["mdn_pred_return"].cpu().item())
        result["mdn_pred_std"] = float(out["mdn_pred_std"].cpu().item())

    if "class_logits" in out:
        probs = torch.softmax(out["class_logits"], dim=-1).cpu().numpy()[0]
        result["class_probs"] = [float(p) for p in probs]

    logger.info(
        "Inference result: pred_return=%.4f, pred_std=%.4f, position=%.1f",
        result["pred_return"], result["pred_std"], result["discrete_position"],
    )
    return result


# ---------------------------------------------------------------------------
# FastAPI server (optional, for scheduled triggers)
# ---------------------------------------------------------------------------

def create_app(
    data_source,
    model: HybridMixtureNetwork,
    checkpoint: Dict[str, Any],
    device: torch.device,
    eia_df: Optional[pd.DataFrame],
    weather_df: Optional[pd.DataFrame],
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
            result = run_inference(
                data_source, model, checkpoint, device,
                eia_df=eia_df, weather_df=weather_df,
            )
            return JSONResponse(content=result)
        except Exception as exc:
            logger.exception("Inference failed")
            return JSONResponse(content={"error": str(exc)}, status_code=500)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


# ---------------------------------------------------------------------------
# Daily context loaders (download from GCS, cache locally)
# ---------------------------------------------------------------------------

def _load_daily_cache(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Load an HDF/parquet/CSV file into a DataFrame."""
    if path is None:
        return None
    from pathlib import Path as _P
    p = _P(path)
    if not p.exists():
        logger.warning("Daily cache not found: %s", path)
        return None
    try:
        if p.suffix in {".h5", ".hdf"}:
            df = pd.read_hdf(path)
        elif p.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, parse_dates=True, index_col=0)
        # Normalize index to avoid tz/resolution comparison issues
        df = _normalize_dt_index(df)
        logger.info("Loaded %s: %d rows", p.name, len(df))
        return df
    except Exception as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return None


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

    # Load model checkpoint (auto-downloads from GCS if not local)
    model_path = resolve_gcs_path(args.model_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Build model from checkpoint
    model = load_model(args.model_path, device, config_path=args.config_path)

    # Load daily context caches (download from GCS if needed)
    eia_path = resolve_gcs_path(args.eia_path) if args.eia_path else None
    weather_path = resolve_gcs_path(args.weather_path) if args.weather_path else None

    eia_df = _load_daily_cache(eia_path)
    weather_df = _load_daily_cache(weather_path)

    logger.info(
        "Context loaded — EIA: %s rows, Weather: %s rows",
        len(eia_df) if eia_df is not None else 0,
        len(weather_df) if weather_df is not None else 0,
    )

    if args.once:
        result = run_inference(
            data_source, model, checkpoint, device,
            eia_df=eia_df, weather_df=weather_df,
            task=args.task, threshold=args.threshold,
        )
        print(json.dumps(result, indent=2))
        return

    # Start HTTP server
    app = create_app(data_source, model, checkpoint, device, eia_df, weather_df)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
