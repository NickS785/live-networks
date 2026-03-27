"""Quick local test of SierraChartTickDataSource with CSV fallback."""

import sys
sys.path.insert(0, r"C:\Users\nicho\PycharmProjects\live-networks")
sys.path.insert(0, r"C:\Users\nicho\PycharmProjects\CTAFlow")

from pathlib import Path
import pandas as pd
from CTAFlow.features.base_extractor import ScidBaseExtractor
from CTAFlow.data.raw_formatting.intraday_manager import read_exported_df
from CTAFlow.config import DLY_DATA_PATH

SCID_FOLDER = "F:/SierraChart/Data"
TICKER = "NG"
CST = "US/Central"
LOOKBACK_DAYS = 30

end = pd.Timestamp.now(tz=CST)
start = end - pd.Timedelta(days=LOOKBACK_DAYS)

print(f"Requesting {TICKER} ticks: {start.date()} -> {end.date()}")

# --- Try SCID binary first ---
extractor = ScidBaseExtractor(SCID_FOLDER, ticker=TICKER, tz="America/Chicago")
df = extractor.get_stitched_data(
    start_time=start, end_time=end,
    columns=["Close", "TotalVolume", "BidVolume", "AskVolume", "NumTrades"],
)

# Normalise index
if not df.empty and not isinstance(df.index, pd.DatetimeIndex):
    for col in ("DateTime", "Datetime"):
        if col in df.columns:
            df = df.set_index(col)
            break
    df.index = pd.to_datetime(df.index)

stale = df.empty or (pd.Timestamp.now(tz="UTC") - df.index[-1].tz_convert("UTC")) > pd.Timedelta(hours=6)
source = "SCID"

if stale:
    csv_path = DLY_DATA_PATH / f"{TICKER.lower()}.csv"
    print(f"SCID data stale/empty -- falling back to CSV: {csv_path}")
    raw = pd.read_csv(csv_path)
    raw.columns = raw.columns.str.strip()
    raw["Datetime"] = pd.to_datetime(
        raw["Date"].str.strip() + " " + raw["Time"].str.strip(), format="mixed"
    )
    raw.rename(columns={"Last": "Close"}, inplace=True)
    raw.set_index("Datetime", inplace=True)
    raw.drop(columns=["Date", "Time"], errors="ignore", inplace=True)
    df = raw[raw.index >= start.tz_localize(None)]
    source = "CSV"

# Ensure tz-aware CST
if df.index.tz is None:
    df.index = df.index.tz_localize("America/Chicago")
df.index = df.index.tz_convert(CST)

print(f"\nSource: {source}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB\n")

print("=== HEAD (first 5 rows) ===")
print(df.head())

print("\n=== TAIL (last 5 rows) ===")
print(df.tail())

print(f"\nFirst timestamp (CST): {df.index[0]}")
print(f"Last timestamp  (CST): {df.index[-1]}")
print(f"Date range span: {(df.index[-1] - df.index[0]).days} days")
print(f"Total rows:      {len(df):,}")
print(f"Unique dates:    {df.index.normalize().nunique()}")
