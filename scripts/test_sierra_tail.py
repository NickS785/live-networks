"""Check last day's timestamp distribution."""
import sys
sys.path.insert(0, r"C:\Users\nicho\PycharmProjects\live-networks")
sys.path.insert(0, r"C:\Users\nicho\PycharmProjects\CTAFlow")

import pandas as pd
from CTAFlow.features.base_extractor import ScidBaseExtractor

SCID_FOLDER = "F:/SierraChart/Data"
TICKER = "NG"
CST = "US/Central"

end = pd.Timestamp.now(tz=CST)
start = end - pd.Timedelta(days=30)

extractor = ScidBaseExtractor(SCID_FOLDER, ticker=TICKER, tz="America/Chicago")
df = extractor.get_stitched_data(start_time=start, end_time=end,
    columns=["Close", "TotalVolume", "BidVolume", "AskVolume", "NumTrades"])

if not isinstance(df.index, pd.DatetimeIndex):
    df = df.set_index("Datetime") if "Datetime" in df.columns else df.set_index("ts")
df.index = pd.to_datetime(df.index)
if df.index.tz is None:
    df.index = df.index.tz_localize("America/Chicago")
df.index = df.index.tz_convert(CST)

# Last 3 days breakdown
last_3 = df[df.index >= df.index.normalize().unique()[-3]]
for day, grp in last_3.groupby(last_3.index.date):
    print(f"\n--- {day} ---")
    print(f"  First tick: {grp.index[0]}")
    print(f"  Last tick:  {grp.index[-1]}")
    print(f"  Rows: {len(grp):,}")
    # Hourly distribution
    hourly = grp.groupby(grp.index.hour).size()
    print(f"  Hourly tick counts:")
    for h, c in hourly.items():
        print(f"    {h:02d}:00 -> {c:,}")
