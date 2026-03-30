"""Test Sierra Chart -> GCS upload for NG (ng_hybrid: resampled + compressed).

Measures total wall-clock time from SCID read through upload completion.

Usage
-----
    python scripts/test_sierra_upload.py
    python scripts/test_sierra_upload.py --model-type mmtft
    python scripts/test_sierra_upload.py --lookback 60 --resample 5min
"""

import sys
sys.path.insert(0, r"C:\Users\nicho\PycharmProjects\live-networks")
sys.path.insert(0, r"C:\Users\nicho\PycharmProjects\CTAFlow")

import argparse
import logging
import time

from live_cta.sources.sierra_tick_source import (
    NEEDS_TICK_DATA,
    SierraChartTickDataSource,
    SierraConfig,
)
from live_cta.sources.gcs_tick_source import GCSConfig, GCSTickerSpec, GCSTickDataSource

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("test_sierra_upload")

SCID_FOLDER = "F:/SierraChart/Data"
GCS_BUCKET = "ctaflow-prod-artifacts"
GCS_CREDENTIALS = "env/gcs_service_account.json"
GCS_PROJECT = "live-trader"
TICKER = "NG"
TZ = "America/Chicago"


def main():
    parser = argparse.ArgumentParser(description="Test Sierra -> GCS upload with timing")
    parser.add_argument("--model-type", default="ng_hybrid", choices=["mmtft", "ng_hybrid"])
    parser.add_argument("--lookback", type=int, default=30, help="Lookback in days")
    parser.add_argument("--resample", default="5min", help="Resample rule for ng_hybrid")
    parser.add_argument("--no-csv", action="store_true", help="Force SCID binary instead of CSV")
    parser.add_argument("--dry-run", action="store_true", help="Extract and compress but skip upload")
    args = parser.parse_args()

    needs_ticks = NEEDS_TICK_DATA.get(args.model_type, True)
    use_csv = not args.no_csv

    if needs_ticks:
        gcs_key = f"live_data/{TICKER}_latest.parquet"
        gcs_fmt = "parquet"
    else:
        gcs_key = f"live_data/{TICKER}_latest.tar.gz"
        gcs_fmt = "tar.gz"

    print(f"{'='*60}")
    print(f"Sierra Chart -> GCS Upload Test")
    print(f"{'='*60}")
    print(f"  Ticker:      {TICKER}")
    print(f"  Source:      {'CSV' if use_csv else 'SCID'}")
    print(f"  Model type:  {args.model_type}")
    print(f"  Strategy:    {'raw tick parquet' if needs_ticks else f'resample {args.resample} + tar.gz'}")
    print(f"  Lookback:    {args.lookback} days")
    print(f"  GCS key:     gs://{GCS_BUCKET}/{gcs_key}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"{'='*60}\n")

    t_total_start = time.perf_counter()

    # --- Step 1: Init Sierra source ---
    t0 = time.perf_counter()
    csv_map = {"NG": "ng.csv"} if use_csv else None
    sierra = SierraChartTickDataSource(
        SierraConfig(scid_folder=SCID_FOLDER, tz=TZ, csv_map=csv_map),
        tickers=[TICKER],
    )
    t_init = time.perf_counter() - t0
    print(f"[1/3] Sierra init:       {t_init:.2f}s")

    # --- Step 2: Build GCS target ---
    gcs_cfg = GCSConfig(
        bucket_name=GCS_BUCKET,
        project=GCS_PROJECT,
        credentials_path=GCS_CREDENTIALS,
    )
    gcs = GCSTickDataSource(
        config=gcs_cfg,
        ticker_map={TICKER: GCSTickerSpec(gcs_key, fmt=gcs_fmt)},
        tz=TZ,
    )

    # --- Step 3: Sync ---
    t0 = time.perf_counter()
    if args.dry_run:
        import pandas as pd
        from live_cta.core.live import _resample_ticks_to_bars

        end = pd.Timestamp.now(tz=TZ)
        start = end - pd.Timedelta(days=args.lookback)
        df = sierra.get_ticks(TICKER, start, end)
        t_extract = time.perf_counter() - t0
        print(f"[2/3] SCID extraction:   {t_extract:.2f}s  ({len(df):,} ticks)")

        if not needs_ticks and not df.empty:
            t0 = time.perf_counter()
            bar_minutes = int(args.resample.rstrip("minTSH"))
            bars = _resample_ticks_to_bars(df, bar_minutes)
            t_resample = time.perf_counter() - t0
            print(f"[3/3] Resample:          {t_resample:.2f}s  ({len(bars):,} bars)")

            tick_mb = df.memory_usage(deep=True).sum() / 1e6
            bar_mb = bars.memory_usage(deep=True).sum() / 1e6
            print(f"\n  Tick data memory:  {tick_mb:.2f} MB")
            print(f"  Bar data memory:   {bar_mb:.2f} MB")
            print(f"  Reduction:         {(1 - bar_mb/tick_mb)*100:.0f}%")
        else:
            print(f"[3/3] Upload:            SKIPPED (dry run)")

        uri = None
    else:
        if needs_ticks:
            uri = sierra.sync_to_gcs(gcs, TICKER, lookback_days=args.lookback)
        else:
            uri = sierra.sync_to_gcs_compressed(
                gcs, TICKER, lookback_days=args.lookback, resample_rule=args.resample,
            )
    t_sync = time.perf_counter() - t0

    t_total = time.perf_counter() - t_total_start

    if not args.dry_run:
        print(f"[2/3] Extract + process: {t_sync:.2f}s")
        print(f"[3/3] Upload URI:        {uri or 'FAILED'}")

    print(f"\n{'='*60}")
    print(f"  TOTAL TIME: {t_total:.2f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
