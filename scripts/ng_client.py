#!/usr/bin/env python
"""NG live data client — Sierra Chart SCID → GCS feeder.

Reads .scid files from a local Sierra Chart data folder, stitches a
rolling front-month NG contract, and pushes the result to a GCS bucket
where the inference server can pick it up.  Runs on the local Windows
machine that has Sierra Chart installed.

Usage
-----
::

    # One-shot sync (last 30 days)
    python scripts/ng_client.py --scid-folder "F:/SierraChart/Data"

    # Continuous loop, re-sync every 5 minutes
    python scripts/ng_client.py --scid-folder "F:/SierraChart/Data" --loop --interval 300

    # Custom lookback + contract cache for fast restarts
    python scripts/ng_client.py --scid-folder "F:/SierraChart/Data" \
        --lookback-days 60 --contract-cache contract_map.pkl
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
from typing import Optional

# Load dotenv before anything else
from dotenv import load_dotenv

_env_file = os.getenv("DOTENV_PATH", str(Path(__file__).resolve().parent.parent / "env" / "dot.env"))
load_dotenv(_env_file, override=False)

import pandas as pd

from live_cta.sources.sierra_tick_source import SierraChartTickDataSource, SierraConfig
from live_cta.sources.gcs_tick_source import GCSConfig, GCSTickerSpec, GCSTickDataSource

logger = logging.getLogger("ng_client")

# ---------------------------------------------------------------------------
# Defaults from env / notebook
# ---------------------------------------------------------------------------

_GCS_BUCKET = os.getenv("GCS_BUCKET", "ctaflow-prod-artifacts")
_GCS_PROJECT = os.getenv("GCS_PROJECT", "")
_GCS_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
_SCID_FOLDER = os.getenv("SCID_FOLDER", "F:/SierraChart/Data")
_MODEL_DATA_DIR = os.getenv("MODEL_DATA_DIR", "model_data/")

# GCS object paths — must match what ng_server.py expects
# gs://ctaflow-prod-artifacts/
#   live_data/NG_latest.parquet   — rolling front-month ticks (this script writes it)
#   model_data/eia_cache.hdf      — EIA weekly storage
#   model_data/weather.hdf        — population-weighted daily weather
#   model_data/NG/intraday_2.csv  — cached 5-min bars (Sierra export)
_GCS_PREFIX = os.getenv("GCS_TICK_PREFIX", "live_data/")
_NG_TICK_KEY = f"{_GCS_PREFIX}NG_latest.parquet"
_NG_EIA_KEY = f"{_MODEL_DATA_DIR}eia_cache.hdf"
_NG_WEATHER_KEY = f"{_MODEL_DATA_DIR}weather.hdf"

_running = True


def _handle_signal(signum, frame):
    global _running
    logger.info("Signal %s received, stopping after current cycle ...", signum)
    _running = False


# ---------------------------------------------------------------------------
# Sync cycle
# ---------------------------------------------------------------------------

def sync_cycle(
    sierra: SierraChartTickDataSource,
    gcs: GCSTickDataSource,
    lookback_days: int,
) -> dict:
    """Run one sync: read SCID, push to GCS, return summary."""
    uri = sierra.sync_to_gcs(gcs, "NG", lookback_days=lookback_days)
    active = sierra.active_contract("NG")

    result = {
        "timestamp": pd.Timestamp.now(tz="America/Chicago").isoformat(),
        "active_contract": active,
        "gcs_uri": uri,
        "lookback_days": lookback_days,
    }
    return result


def sync_daily_context(
    gcs_config: GCSConfig,
    eia_path: Optional[str],
    weather_path: Optional[str],
) -> dict:
    """Upload local EIA/weather caches to GCS so the server can fetch them."""
    try:
        from google.cloud import storage as gcs_storage
        from google.oauth2 import service_account
    except ImportError:
        logger.warning("google-cloud-storage not installed, skipping context upload")
        return {}

    kwargs = {}
    if gcs_config.project:
        kwargs["project"] = gcs_config.project
    if gcs_config.credentials_path:
        creds = service_account.Credentials.from_service_account_file(gcs_config.credentials_path)
        kwargs["credentials"] = creds

    client = gcs_storage.Client(**kwargs)
    bucket = client.bucket(gcs_config.bucket_name)
    uploaded = {}

    for label, local_path, gcs_key in [
        ("eia", eia_path, _NG_EIA_KEY),
        ("weather", weather_path, _NG_WEATHER_KEY),
    ]:
        if not local_path or not Path(local_path).exists():
            continue
        try:
            blob = bucket.blob(gcs_key)
            blob.upload_from_filename(local_path)
            uri = f"gs://{gcs_config.bucket_name}/{gcs_key}"
            uploaded[label] = uri
            logger.info("Uploaded %s -> %s", local_path, uri)
        except Exception as exc:
            logger.warning("Failed to upload %s: %s", label, exc)

    return uploaded


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NG data client: Sierra Chart SCID -> GCS feeder",
    )

    # Sierra Chart
    parser.add_argument("--scid-folder", default=_SCID_FOLDER,
                        help="Sierra Chart data directory with .scid files")
    parser.add_argument("--contract-cache", default=None,
                        help="Pickle cache for contract map (faster restarts)")
    parser.add_argument("--lookback-days", type=int, default=60,
                        help="Calendar days of history to upload")

    # GCS
    parser.add_argument("--gcs-bucket", default=_GCS_BUCKET)
    parser.add_argument("--gcs-project", default=_GCS_PROJECT)
    parser.add_argument("--gcs-credentials", default=_GCS_CREDS)
    parser.add_argument("--gcs-tick-key", default=_NG_TICK_KEY,
                        help="GCS object path for NG tick data")

    # Daily context sync
    parser.add_argument("--sync-context", action="store_true",
                        help="Also upload EIA/weather caches to GCS")
    parser.add_argument("--eia-path", default=None,
                        help="Local path to EIA storage cache")
    parser.add_argument("--weather-path", default=None,
                        help="Local path to weather cache")

    # Loop
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=300,
                        help="Sync interval in seconds (default 5 min)")
    parser.add_argument("--output", default=None,
                        help="Append JSON sync results to this file")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # --- Build Sierra source ---
    sierra_cfg = SierraConfig(
        scid_folder=args.scid_folder,
        contract_cache=args.contract_cache,
    )
    sierra = SierraChartTickDataSource(sierra_cfg, tickers=["NG"])

    # Save contract cache on first run if not provided
    if args.contract_cache is None:
        cache_path = sierra.save_contract_cache()
        logger.info("Contract cache saved to %s (use --contract-cache for faster restarts)", cache_path)

    # --- Build GCS target ---
    gcs_config = GCSConfig(
        bucket_name=args.gcs_bucket,
        project=args.gcs_project or None,
        credentials_path=args.gcs_credentials or None,
    )
    gcs = GCSTickDataSource(
        config=gcs_config,
        ticker_map={"NG": GCSTickerSpec(args.gcs_tick_key, fmt="parquet")},
    )

    out_file = Path(args.output) if args.output else None
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(
        "NG Client ready: scid=%s | gcs=gs://%s/%s | lookback=%dd",
        args.scid_folder, args.gcs_bucket, args.gcs_tick_key, args.lookback_days,
    )

    # --- Run ---
    while _running:
        try:
            result = sync_cycle(sierra, gcs, args.lookback_days)
            logger.info(
                "Sync complete: contract=%s  uri=%s",
                result.get("active_contract"), result.get("gcs_uri"),
            )

            # Optional: also push daily context files
            if args.sync_context:
                ctx = sync_daily_context(gcs_config, args.eia_path, args.weather_path)
                result["context_uploaded"] = ctx

            if out_file:
                with open(out_file, "a") as f:
                    f.write(json.dumps(result) + "\n")

            if not args.loop:
                print(json.dumps(result, indent=2))
                break

        except Exception:
            logger.exception("Sync cycle failed")
            if not args.loop:
                sys.exit(1)

        if args.loop and _running:
            logger.info("Next sync in %ds", args.interval)
            time.sleep(args.interval)

    logger.info("Client shutdown complete.")


if __name__ == "__main__":
    main()
