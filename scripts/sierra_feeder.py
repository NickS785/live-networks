"""Sierra Chart -> GCS upload feeder.

Reads tick data from Sierra Chart CSV exports (default) or SCID binary files,
then uploads to GCS. Upload strategy depends on the model type:

- **mmtft**: Raw tick parquet (model needs VPIN, profile, numbars)
- **ng_hybrid**: Resample to 5-min bars -> tar.gz -> upload (model only needs bar features)

Only the lookback window is uploaded, not the full history.

Usage
-----
    # NG hybrid (resampled + compressed, CSV source)
    python scripts/sierra_feeder.py --ticker NG --model-type ng_hybrid --once

    # MMTFT ticker (raw ticks, CSV source)
    python scripts/sierra_feeder.py --ticker NG --model-type mmtft --once

    # Force SCID binary instead of CSV
    python scripts/sierra_feeder.py --ticker NG --model-type mmtft --no-csv --once

    # Continuous loop (default 300s interval)
    python scripts/sierra_feeder.py --ticker NG --model-type ng_hybrid --interval 300
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime

logger = logging.getLogger("sierra_feeder")

# ---------------------------------------------------------------------------
# Defaults (override via env vars or CLI args)
# ---------------------------------------------------------------------------

SCID_FOLDER = os.getenv("SCID_FOLDER", "F:/SierraChart/Data")
GCS_BUCKET = os.getenv("GCS_BUCKET", "ctaflow-prod-artifacts")
GCS_PROJECT = os.getenv("GCS_PROJECT", "")
GCS_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
LOOKBACK_DAYS = int(os.getenv("SIERRA_FEEDER_LOOKBACK_DAYS", "30"))
TZ = "America/Chicago"

# Default CSV filenames per ticker (in scid_folder)
DEFAULT_CSV_MAP = {
    "NG": "ng.csv",
    "CL": "cl.csv",
}

_running = True


def _handle_signal(signum, _frame):
    global _running
    logger.info("Received signal %s, shutting down...", signum)
    _running = False


# ---------------------------------------------------------------------------
# Upload dispatch
# ---------------------------------------------------------------------------

def run_sync(
    ticker: str,
    model_type: str,
    lookback_days: int,
    resample_rule: str,
    data_dir: str,
    use_csv: bool,
) -> str | None:
    """Run a single sync cycle for one ticker."""
    from live_cta.sources.sierra_tick_source import (
        NEEDS_TICK_DATA,
        SierraChartTickDataSource,
        SierraConfig,
    )
    from live_cta.sources.gcs_tick_source import GCSConfig, GCSTickerSpec, GCSTickDataSource

    needs_ticks = NEEDS_TICK_DATA.get(model_type, True)

    # Build Sierra source (CSV by default)
    csv_map = DEFAULT_CSV_MAP if use_csv else None
    sierra = SierraChartTickDataSource(
        SierraConfig(scid_folder=data_dir, tz=TZ, csv_map=csv_map),
        tickers=[ticker],
    )

    # Build GCS target
    if needs_ticks:
        gcs_key = f"live_data/{ticker}_latest.parquet"
        gcs_fmt = "parquet"
    else:
        gcs_key = f"live_data/{ticker}_latest.tar.gz"
        gcs_fmt = "tar.gz"

    gcs_cfg = GCSConfig(
        bucket_name=GCS_BUCKET,
        project=GCS_PROJECT or None,
        credentials_path=GCS_CREDENTIALS or None,
    )
    gcs = GCSTickDataSource(
        config=gcs_cfg,
        ticker_map={ticker: GCSTickerSpec(gcs_key, fmt=gcs_fmt)},
        tz=TZ,
    )

    # Dispatch based on model type
    if needs_ticks:
        logger.info("[%s] Uploading raw tick parquet (csv=%s)", ticker, use_csv)
        return sierra.sync_to_gcs(gcs, ticker, lookback_days=lookback_days)
    else:
        logger.info(
            "[%s] Uploading resampled %s bars as tar.gz (csv=%s)",
            ticker, resample_rule, use_csv,
        )
        return sierra.sync_to_gcs_compressed(
            gcs, ticker, lookback_days=lookback_days, resample_rule=resample_rule,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sierra Chart -> GCS feeder")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. NG)")
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["mmtft", "ng_hybrid"],
        help="Model type determines upload strategy: mmtft=raw ticks, ng_hybrid=resampled+compressed",
    )
    parser.add_argument("--no-csv", action="store_true", help="Force SCID binary instead of CSV")
    parser.add_argument("--interval", type=int, default=300, help="Sync interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS, help="Lookback in days")
    parser.add_argument("--resample", default="5min", help="Resample rule for ng_hybrid (default 5min)")
    parser.add_argument("--data-dir", default=SCID_FOLDER, help="Sierra Chart data directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    use_csv = not args.no_csv

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(
        "Starting Sierra feeder: ticker=%s, model_type=%s, csv=%s, interval=%ds",
        args.ticker, args.model_type, use_csv, args.interval,
    )

    while _running:
        try:
            uri = run_sync(
                args.ticker, args.model_type, args.lookback,
                args.resample, args.data_dir, use_csv,
            )
            if uri:
                logger.info("Sync complete: %s @ %s", uri, datetime.now().isoformat())
            else:
                logger.warning("Sync returned no URI (no data?)")
        except Exception:
            logger.exception("Sierra feeder cycle failed")

        if args.once:
            break

        time.sleep(args.interval)

    logger.info("Sierra feeder stopped.")


if __name__ == "__main__":
    main()
