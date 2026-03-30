from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import pandas as pd

from live_cta.core.live import (
    TickDataSource,
    TimestampLike,
    _ensure_bar_columns,
    _ensure_datetime_index,
    _ensure_tick_columns,
    _looks_like_bar_data,
)

logger = logging.getLogger(__name__)


def _load_local_market_frame(path: Path, tz: str) -> pd.DataFrame:
    if not path.exists():
        logger.warning("History cache not found: %s", path)
        return pd.DataFrame()

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df.columns = [str(col).strip() for col in df.columns]
    lowered = {str(col).strip().lower().replace(" ", ""): col for col in df.columns}

    if "datetime" in lowered:
        df = df.rename(columns={lowered["datetime"]: "Datetime"})
        df = df.set_index("Datetime")
    elif "ts" in lowered:
        df = df.rename(columns={lowered["ts"]: "ts"})
        df = df.set_index("ts")
    elif "timestamp" in lowered:
        df = df.rename(columns={lowered["timestamp"]: "ts"})
        df = df.set_index("ts")
    elif "date" in lowered and "time" in lowered:
        date_col = lowered["date"]
        time_col = lowered["time"]
        df["Datetime"] = pd.to_datetime(
            df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
            format="mixed",
        )
        df = df.set_index("Datetime")
        df = df.drop(columns=[date_col, time_col], errors="ignore")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"History cache {path} must have a Datetime index or Date/Time columns."
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return _ensure_datetime_index(df, tz)


class HistoricalBackfillDataSource:
    """Augment a live source with a local history cache before feature building.

    The server uses this wrapper so live GCS uploads can be treated as deltas
    against an existing historical base. The returned frame is sliced to the
    requested range, sorted, and deduplicated with live data winning on overlap.
    """

    def __init__(
        self,
        live_source: TickDataSource,
        history_map: Mapping[str, Union[str, Path]],
        tz: str = "America/Chicago",
    ) -> None:
        self.live_source = live_source
        self.history_map = {ticker: Path(path) for ticker, path in history_map.items()}
        self.tz = tz
        self._history_cache: Dict[str, pd.DataFrame] = {}
        self._history_mtime: Dict[str, Optional[float]] = {}

    def _get_history(self, ticker: str) -> pd.DataFrame:
        path = self.history_map.get(ticker)
        if path is None:
            return pd.DataFrame()

        mtime = path.stat().st_mtime if path.exists() else None
        if ticker not in self._history_cache or self._history_mtime.get(ticker) != mtime:
            self._history_cache[ticker] = _load_local_market_frame(path, self.tz)
            self._history_mtime[ticker] = mtime
        return self._history_cache[ticker]

    def get_ticks(
        self,
        ticker: str,
        start_time: TimestampLike,
        end_time: TimestampLike,
    ) -> pd.DataFrame:
        start_ts = pd.Timestamp(start_time)
        end_ts = pd.Timestamp(end_time)
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize(self.tz)
        else:
            start_ts = start_ts.tz_convert(self.tz)
        if end_ts.tz is None:
            end_ts = end_ts.tz_localize(self.tz)
        else:
            end_ts = end_ts.tz_convert(self.tz)

        history = self._get_history(ticker)
        if not history.empty:
            history = history.loc[(history.index >= start_ts) & (history.index <= end_ts)].copy()

        live = self.live_source.get_ticks(ticker, start_ts, end_ts)
        live = _ensure_datetime_index(live, self.tz) if not live.empty else live

        if history.empty:
            return live
        if live.empty:
            return history

        if _looks_like_bar_data(history) or _looks_like_bar_data(live):
            history = _ensure_bar_columns(history)
            live = _ensure_bar_columns(live)
        else:
            history = _ensure_tick_columns(history)
            live = _ensure_tick_columns(live)

        combined = pd.concat([history, live], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()
