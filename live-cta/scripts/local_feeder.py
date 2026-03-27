"""Local feeder: IBKR -> parquet -> S3/GCS upload.

Runs on the local machine with IBKR Client Portal Gateway authenticated.
Fetches the latest intraday bars, writes a parquet file, and pushes it to
the configured cloud storage bucket on a fixed interval.

Usage
-----
    python scripts/local_feeder.py --backend s3 --interval 300
    python scripts/local_feeder.py --backend gcs --interval 300
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from live_cta.sources.ibkr_client import IBKRConfig, IBKRContract, IBKRTickDataSource

logger = logging.getLogger("local_feeder")

# ---------------------------------------------------------------------------
# Defaults (override via env vars or CLI args)
# ---------------------------------------------------------------------------

IBKR_BASE_URL = os.getenv("IBKR_BASE_URL", "https://localhost:5000")
IBKR_ACCOUNT_ID = os.getenv("IBKR_ACCOUNT_ID", "")
NG_CONID = os.getenv("NG_CONID", "")

# S3
S3_BUCKET = os.getenv("S3_BUCKET", "ctaflow-prod")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "")
S3_PREFIX = os.getenv("S3_PREFIX", "")

# GCS
GCS_BUCKET = os.getenv("GCS_BUCKET", "ctaflow-prod-artifacts")
GCS_PROJECT = os.getenv("GCS_PROJECT", "")
GCS_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

UPLOAD_KEY = "live_data/ng_latest.parquet"
LOCAL_STAGING = Path(os.getenv("LOCAL_STAGING_DIR", Path.home() / ".ctaflow" / "staging"))
LOOKBACK_DAYS = int(os.getenv("FEEDER_LOOKBACK_DAYS", "10"))
TZ = "America/Chicago"

_running = True


def _handle_signal(signum, frame):
    global _running
    logger.info("Received signal %s, shutting down...", signum)
    _running = False


# ---------------------------------------------------------------------------
# IBKR fetch
# ---------------------------------------------------------------------------

def fetch_and_save(ibkr: IBKRTickDataSource, ticker: str, out_path: Path) -> Path:
    """Fetch latest bars from IBKR and save as parquet."""
    now = pd.Timestamp.now(tz=TZ)
    start = now - pd.Timedelta(days=LOOKBACK_DAYS)

    logger.info("Fetching %s bars [%s -> %s]", ticker, start, now)
    df = ibkr.get_ticks(ticker, start_time=start, end_time=now)

    if df.empty:
        raise RuntimeError(f"IBKR returned no data for {ticker}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    logger.info("Saved %d bars to %s (%.1f KB)", len(df), out_path, out_path.stat().st_size / 1024)
    return out_path


# ---------------------------------------------------------------------------
# Upload backends
# ---------------------------------------------------------------------------

def upload_s3(local_path: Path) -> str:
    from live_cta.storage.aws_client import S3Config
    from live_cta.sources.s3_tick_source import S3TickerSpec, S3TickDataSource

    cfg = S3Config(
        bucket_name=S3_BUCKET,
        endpoint_url=S3_ENDPOINT or None,
        prefix=S3_PREFIX,
    )
    source = S3TickDataSource(
        s3_config=cfg,
        ticker_map={"NG": S3TickerSpec(UPLOAD_KEY)},
    )
    return source.upload("NG", local_path)


def upload_gcs(local_path: Path) -> str:
    from live_cta.sources.gcs_tick_source import GCSConfig, GCSTickerSpec, GCSTickDataSource

    cfg = GCSConfig(
        bucket_name=GCS_BUCKET,
        project=GCS_PROJECT or None,
        credentials_path=GCS_CREDENTIALS or None,
    )
    source = GCSTickDataSource(
        config=cfg,
        ticker_map={"NG": GCSTickerSpec(UPLOAD_KEY)},
    )
    return source.upload("NG", local_path)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IBKR -> cloud storage feeder")
    parser.add_argument("--backend", choices=["s3", "gcs"], default="s3")
    parser.add_argument("--interval", type=int, default=300, help="Sync interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--ticker", default="NG")
    parser.add_argument("--conid", default=NG_CONID, help="IBKR contract ID")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    global LOOKBACK_DAYS
    LOOKBACK_DAYS = args.lookback

    if not args.conid:
        logger.error("Contract ID required. Set NG_CONID env var or pass --conid")
        sys.exit(1)

    # Build IBKR source
    ibkr_cfg = IBKRConfig(base_url=IBKR_BASE_URL, account_id=IBKR_ACCOUNT_ID)
    contracts = {
        args.ticker: IBKRContract(conid=args.conid, ticker=args.ticker),
    }
    ibkr = IBKRTickDataSource(config=ibkr_cfg, contracts=contracts, tz=TZ)

    upload_fn = upload_s3 if args.backend == "s3" else upload_gcs
    staging_path = LOCAL_STAGING / f"{args.ticker.lower()}_latest.parquet"

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(
        "Starting feeder: ticker=%s, backend=%s, interval=%ds",
        args.ticker, args.backend, args.interval,
    )

    while _running:
        try:
            fetch_and_save(ibkr, args.ticker, staging_path)
            uri = upload_fn(staging_path)
            logger.info("Upload complete: %s @ %s", uri, datetime.now().isoformat())
        except Exception:
            logger.exception("Feeder cycle failed")

        if args.once:
            break

        time.sleep(args.interval)

    logger.info("Feeder stopped.")


if __name__ == "__main__":
    main()
