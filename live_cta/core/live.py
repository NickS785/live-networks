from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from CTAFlow.data.datasets.tft import build_ticker_registry
from CTAFlow.data.datasets.v3_continuous import (
    compute_ae_daily_features,
    prepare_vpin_spatial_features,
    rasterize_vpin_to_grid,
    scale_numbars,
    scale_vpin_features,
)
from CTAFlow.data.raw_formatting.intraday_manager import read_exported_df
from CTAFlow.features.tick_extractor import _compute_profile_levels
from CTAFlow.features.volume.profile import MarketProfileExtractor, NumberBarsExtractor
from CTAFlow.features.volume.vpin import SequenceRasterizer, VPINExtractor
from CTAFlow.models.deep_learning.training.backtest import predictions_to_positions
from CTAFlow.models.prep.intraday_continuous import ContinuousIntradayPrep, SessionSpec


TimestampLike = Union[str, pd.Timestamp]


class TickDataSource(Protocol):
    """Provider contract for live or replayed tick data."""

    def get_ticks(
        self,
        ticker: str,
        start_time: TimestampLike,
        end_time: TimestampLike,
    ) -> pd.DataFrame:
        """Return tick-level data indexed by timestamp."""


def _ensure_datetime_index(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "Datetime" in out.columns:
            out = out.set_index("Datetime")
        elif "ts" in out.columns:
            out = out.set_index("ts")
        else:
            raise TypeError("Tick DataFrame must have a DatetimeIndex or a 'Datetime'/'ts' column.")

    out.index = pd.to_datetime(out.index).as_unit("ns")
    if out.index.tz is None:
        out.index = out.index.tz_localize(tz)
    else:
        out.index = out.index.tz_convert(tz)

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _rename_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map: Dict[str, str] = {}
    for col in out.columns:
        lower = str(col).strip().lower().replace(" ", "")
        if lower == "last":
            rename_map[col] = "Close"
        elif lower == "open":
            rename_map[col] = "Open"
        elif lower == "high":
            rename_map[col] = "High"
        elif lower == "low":
            rename_map[col] = "Low"
        elif lower in {"volume", "vol"}:
            rename_map[col] = "Volume"
        elif lower == "totalvolume":
            rename_map[col] = "TotalVolume"
        elif lower == "bidvolume":
            rename_map[col] = "BidVolume"
        elif lower == "askvolume":
            rename_map[col] = "AskVolume"
        elif lower in {"numtrades", "numberoftrades"}:
            rename_map[col] = "NumTrades"
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def _ensure_tick_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = _rename_price_columns(df)

    if "Close" not in out.columns:
        raise KeyError("Tick data must contain a 'Close' or 'Last' column.")

    if "BidVolume" not in out.columns:
        out["BidVolume"] = 0.0
    if "AskVolume" not in out.columns:
        out["AskVolume"] = 0.0
    # Derive TotalVolume from BidVolume + AskVolume; accept existing column as fallback
    if "TotalVolume" not in out.columns:
        out["TotalVolume"] = (
            pd.to_numeric(out["BidVolume"], errors="coerce").fillna(0.0)
            + pd.to_numeric(out["AskVolume"], errors="coerce").fillna(0.0)
        )
    if "NumTrades" not in out.columns:
        total = pd.to_numeric(out["TotalVolume"], errors="coerce").fillna(0.0)
        out["NumTrades"] = (total > 0).astype(np.int32)

    for col in ("Close", "BidVolume", "AskVolume", "TotalVolume", "NumTrades"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _looks_like_bar_data(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    cols = {str(col).strip().lower().replace(" ", "") for col in df.columns}
    required = {"open", "high", "low"}
    return required.issubset(cols)


def _ensure_bar_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = _rename_price_columns(df)
    if "Close" not in out.columns:
        raise KeyError("Bar data must contain a 'Close' or 'Last' column.")

    close = pd.to_numeric(out["Close"], errors="coerce")
    out["Open"] = pd.to_numeric(out.get("Open", close), errors="coerce").fillna(close)
    out["High"] = pd.to_numeric(out.get("High", close), errors="coerce").fillna(close)
    out["Low"] = pd.to_numeric(out.get("Low", close), errors="coerce").fillna(close)

    if "Volume" in out.columns:
        volume = pd.to_numeric(out["Volume"], errors="coerce").fillna(0.0)
    elif "TotalVolume" in out.columns:
        volume = pd.to_numeric(out["TotalVolume"], errors="coerce").fillna(0.0)
    else:
        bid = pd.to_numeric(out.get("BidVolume", 0.0), errors="coerce").fillna(0.0)
        ask = pd.to_numeric(out.get("AskVolume", 0.0), errors="coerce").fillna(0.0)
        volume = bid + ask
    out["Volume"] = volume

    for col in ("Open", "High", "Low", "Close", "Volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _merge_time_series_frames(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    lookback_start: pd.Timestamp,
) -> pd.DataFrame:
    if existing.empty:
        combined = incoming.copy()
    elif incoming.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, incoming], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

    if combined.empty:
        return combined
    return combined.loc[combined.index >= lookback_start].copy()


def _parse_hhmm(value: str) -> time:
    hour, minute = value.split(":")[:2]
    return time(int(hour), int(minute))


def _combine_date_time(d: date, hhmm: str, tz: str) -> pd.Timestamp:
    return pd.Timestamp(f"{d.isoformat()} {hhmm}").tz_localize(tz)


def _floor_timestamp(ts: pd.Timestamp, freq: str) -> pd.Timestamp:
    if ts.tz is None:
        raise ValueError("Refresh timestamps must be timezone-aware.")
    return ts.floor(freq)


def _as_naive(values: Union[pd.Index, pd.Series, Sequence[pd.Timestamp]]) -> np.ndarray:
    idx = pd.DatetimeIndex(values)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.values.astype("datetime64[ns]")


def _resample_ticks_to_bars(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    work = _ensure_tick_columns(df)
    rule = f"{int(bar_minutes)}min"
    close = work["Close"].astype(float)
    volume = pd.to_numeric(work["TotalVolume"], errors="coerce").fillna(0.0)

    bars = pd.DataFrame(
        {
            "Open": close.resample(rule, label="right", closed="right").first(),
            "High": close.resample(rule, label="right", closed="right").max(),
            "Low": close.resample(rule, label="right", closed="right").min(),
            "Close": close.resample(rule, label="right", closed="right").last(),
            "Volume": volume.resample(rule, label="right", closed="right").sum(),
        }
    )
    bars = bars.dropna(subset=["Open", "High", "Low", "Close"])
    return bars.sort_index()


def _pad_left_2d(arr: np.ndarray, target_rows: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if target_rows <= 0:
        return arr
    if arr.shape[0] >= target_rows:
        return arr[-target_rows:]
    pad = np.zeros((target_rows - arr.shape[0],) + arr.shape[1:], dtype=np.float32)
    return np.concatenate([pad, arr], axis=0)


class InMemoryTickDataSource:
    """Replay/live source backed by in-memory DataFrames."""

    def __init__(self, data: Mapping[str, pd.DataFrame], tz: str = "America/Chicago"):
        self.data = {ticker: _ensure_datetime_index(df, tz) for ticker, df in data.items()}
        self.tz = tz

    def get_ticks(
        self,
        ticker: str,
        start_time: TimestampLike,
        end_time: TimestampLike,
    ) -> pd.DataFrame:
        if ticker not in self.data:
            return pd.DataFrame()
        df = self.data[ticker]
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
        return df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()


class CsvTickDataSource(InMemoryTickDataSource):
    """Replay source for exported Sierra-style CSVs."""

    def __init__(self, csv_paths: Mapping[str, Union[str, Path]], tz: str = "America/Chicago"):
        data = {ticker: read_exported_df(str(path)) for ticker, path in csv_paths.items()}
        super().__init__(data=data, tz=tz)


class SierraTickDataSource:
    """Tick provider backed by SCID files through CTAFlow extractors."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        tz: str = "America/Chicago",
        contract_cache: Optional[Union[str, Path]] = None,
    ):
        self.data_dir = str(data_dir)
        self.tz = tz
        self.contract_cache = contract_cache
        self._extractors: Dict[str, Any] = {}

    def _get_extractor(self, ticker: str) -> Any:
        if ticker not in self._extractors:
            from CTAFlow.features.base_extractor import ScidBaseExtractor

            self._extractors[ticker] = ScidBaseExtractor(
                self.data_dir,
                ticker=ticker,
                tz=self.tz,
                contract_cache=self.contract_cache,
            )
        return self._extractors[ticker]

    def get_ticks(
        self,
        ticker: str,
        start_time: TimestampLike,
        end_time: TimestampLike,
    ) -> pd.DataFrame:
        extractor = self._get_extractor(ticker)
        return extractor.get_stitched_data(
            start_time=start_time,
            end_time=end_time,
            columns=["Close", "TotalVolume", "BidVolume", "AskVolume", "NumTrades"],
        )


class _InlineMarketProfileExtractor(MarketProfileExtractor):
    def __init__(self, tick_size: Optional[float] = None):
        self.tick_size = tick_size


class _InlineNumberBarsExtractor(NumberBarsExtractor):
    def __init__(
        self,
        tick_size: float,
        interval: Union[str, int],
        vwap_window: str,
        num_levels: int,
        centering_method: str,
    ):
        self.tick_size = tick_size
        self.interval = interval
        self.vwap_window = vwap_window
        self.num_levels = num_levels
        self.centering_method = centering_method


class _InlineVPINExtractor(VPINExtractor):
    def __init__(self, bucket_volume: int, window: int):
        self.bucket_volume = bucket_volume
        self.window = window


@dataclass
class LiveEvaluationConfig:
    ticker: str
    tick_size: float
    tz: str = "America/Chicago"
    refresh_interval: str = "30min"
    history_lookback_days: int = 400

    sessions: Sequence[SessionSpec] = field(
        default_factory=lambda: [SessionSpec("USA", "08:30", "16:00")]
    )
    bar_minutes: int = 5
    target_horizon_minutes: int = 30
    tech_lookback: int = 128
    seq_lookback_bars: int = 64
    numbars_lookback: int = 24
    use_fused_spatial: bool = True

    vpin_bucket_volume: int = 150
    vpin_window: int = 60
    vpin_start_time: str = "08:30"
    vpin_end_time: str = "16:00"

    profile_start_time: str = "02:00"
    profile_end_time: str = "09:30"
    value_area_pct: float = 0.7
    include_prev_24h_profile: bool = True
    include_ib: bool = True
    ib_minutes: int = 60

    num_bars_start_time: str = "08:30"
    num_bars_end_time: str = "16:00"
    num_bars_interval: str = "15min"
    num_bars_levels: int = 16
    num_bars_vwap_window: str = "1h"
    num_bars_centering: str = "rolling_vwap"

    raster_num_bars: int = 12
    raster_interval_mins: int = 30
    raster_bins: int = 128
    raster_span_pct: float = 0.01
    raster_session_start: str = "08:30"
    raster_vol_scale: float = 10.0
    raster_price_scale: float = 100.0

    ae_window: int = 21
    ae_target_time: str = "10:00"

    def __post_init__(self) -> None:
        if self.target_horizon_minutes % self.bar_minutes != 0:
            raise ValueError("target_horizon_minutes must be divisible by bar_minutes.")


@dataclass
class LiveFeatureSnapshot:
    ticker: str
    refreshed_at: pd.Timestamp
    anchor_ts: pd.Timestamp
    target_end_ts: pd.Timestamp
    sample: Dict[str, Any]
    bars: pd.DataFrame
    raw_ticks: pd.DataFrame
    vpin: pd.DataFrame

    @property
    def anchor_close(self) -> float:
        if self.anchor_ts in self.bars.index:
            return float(self.bars.loc[self.anchor_ts, "Close"])
        return float("nan")

    def to_model_inputs(self, add_batch_dim: bool = True) -> Dict[str, torch.Tensor]:
        sample = self.sample
        out: Dict[str, torch.Tensor] = {
            "tech_features": torch.tensor(sample["tech_features"], dtype=torch.float32),
            "tech_lens": torch.tensor(sample["tech_len"], dtype=torch.long),
            "seq_vpin": torch.tensor(sample["seq_vpin"], dtype=torch.float32),
            "seq_vpin_lens": torch.tensor(sample["seq_vpin_len"], dtype=torch.long),
            "ae_input": torch.tensor(sample["ae_input"], dtype=torch.float32),
            "ticker_id": torch.tensor(sample["ticker_id"], dtype=torch.long),
            "asset_class_id": torch.tensor(sample["asset_class_id"], dtype=torch.long),
            "asset_subclass_id": torch.tensor(sample["asset_subclass_id"], dtype=torch.long),
        }
        if "fused_spatial" in sample:
            out["fused_spatial"] = torch.tensor(sample["fused_spatial"], dtype=torch.float32)
        if "numbars_recent" in sample:
            out["numbars_recent"] = torch.tensor(sample["numbars_recent"], dtype=torch.float32)
        if "vpin_raster_recent" in sample:
            out["vpin_raster_recent"] = torch.tensor(sample["vpin_raster_recent"], dtype=torch.float32)

        if add_batch_dim:
            out = {key: value.unsqueeze(0) for key, value in out.items()}
        return out


@dataclass
class TradeRecord:
    ticker: str
    anchor_ts: pd.Timestamp
    target_end_ts: pd.Timestamp
    position: float
    prediction: float
    entry_price: float
    exit_price: Optional[float] = None
    realized_return: Optional[float] = None
    status: str = "open"
    meta: Dict[str, Any] = field(default_factory=dict)


class TradeLedger:
    """Simple trade-by-trade ledger for live or replay evaluation."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self._records: List[TradeRecord] = []

    @property
    def records(self) -> List[TradeRecord]:
        return list(self._records)

    def open_trade(
        self,
        snapshot: LiveFeatureSnapshot,
        prediction: Union[float, np.ndarray],
        *,
        task: str = "regression",
        threshold: float = 0.0,
        long_class: int = 2,
        short_class: int = 0,
        position: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> TradeRecord:
        pred_arr = np.asarray(prediction)
        pred_scalar = float(pred_arr.reshape(-1)[0]) if pred_arr.size else 0.0
        if position is None:
            inferred = predictions_to_positions(
                pred_arr.reshape(1, -1) if pred_arr.ndim > 1 else pred_arr.reshape(1),
                task=task,
                threshold=threshold,
                long_class=long_class,
                short_class=short_class,
            )
            position = float(inferred.reshape(-1)[0])

        record = TradeRecord(
            ticker=self.ticker,
            anchor_ts=snapshot.anchor_ts,
            target_end_ts=snapshot.target_end_ts,
            position=float(position),
            prediction=pred_scalar,
            entry_price=snapshot.anchor_close,
            meta=dict(meta or {}),
        )
        self._records.append(record)
        return record

    def close_due_trades(self, bars: pd.DataFrame, asof: Optional[pd.Timestamp] = None) -> List[TradeRecord]:
        if bars.empty:
            return []

        idx = bars.index
        if asof is None:
            latest_ts = idx.max()
        else:
            latest_ts = pd.Timestamp(asof)
            if idx.tz is not None and latest_ts.tz is None:
                latest_ts = latest_ts.tz_localize(idx.tz)
            elif idx.tz is not None and latest_ts.tz is not None:
                latest_ts = latest_ts.tz_convert(idx.tz)
            elif idx.tz is None and latest_ts.tz is not None:
                latest_ts = latest_ts.tz_localize(None)

        closed: List[TradeRecord] = []
        for record in self._records:
            if record.status != "open" or latest_ts < record.target_end_ts:
                continue

            if record.target_end_ts in bars.index:
                exit_px = float(bars.loc[record.target_end_ts, "Close"])
            else:
                match = bars.loc[bars.index <= record.target_end_ts]
                if match.empty:
                    continue
                exit_px = float(match["Close"].iloc[-1])

            if not np.isfinite(record.entry_price) or not np.isfinite(exit_px):
                continue

            if record.position >= 0:
                realized = (exit_px / record.entry_price) - 1.0
            else:
                realized = (record.entry_price / exit_px) - 1.0

            record.exit_price = exit_px
            record.realized_return = float(realized)
            record.status = "closed"
            closed.append(record)

        return closed

    def to_frame(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "anchor_ts",
                    "target_end_ts",
                    "position",
                    "prediction",
                    "entry_price",
                    "exit_price",
                    "realized_return",
                    "status",
                ]
            )
        rows = []
        for record in self._records:
            row = {
                "ticker": record.ticker,
                "anchor_ts": record.anchor_ts,
                "target_end_ts": record.target_end_ts,
                "position": record.position,
                "prediction": record.prediction,
                "entry_price": record.entry_price,
                "exit_price": record.exit_price,
                "realized_return": record.realized_return,
                "status": record.status,
            }
            row.update(record.meta)
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> Dict[str, float]:
        frame = self.to_frame()
        closed = frame[frame["status"] == "closed"].copy()
        if closed.empty:
            return {
                "n_trades": 0.0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "profit_factor": 0.0,
            }

        rets = pd.to_numeric(closed["realized_return"], errors="coerce").dropna()
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        gross_profit = float(wins.sum()) if not wins.empty else 0.0
        gross_loss = abs(float(losses.sum())) if not losses.empty else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else float("inf") if gross_profit > 0 else 0.0

        return {
            "n_trades": float(len(rets)),
            "win_rate": float((rets > 0).mean()),
            "avg_return": float(rets.mean()),
            "best_trade": float(rets.max()),
            "worst_trade": float(rets.min()),
            "profit_factor": float(profit_factor),
        }


class LiveV3FeatureInterface:
    """Rebuild `v3_continuous`-style samples on a fixed live refresh cadence."""

    def __init__(
        self,
        config: LiveEvaluationConfig,
        data_source: TickDataSource,
    ):
        self.config = config
        self.data_source = data_source
        self.prep = ContinuousIntradayPrep(
            sessions=config.sessions,
            bar_minutes=config.bar_minutes,
        )
        self.registry = build_ticker_registry([config.ticker])
        self.ticker_meta = self.registry[config.ticker]

        self.profile_extractor = _InlineMarketProfileExtractor(tick_size=config.tick_size)
        self.number_bars_extractor = _InlineNumberBarsExtractor(
            tick_size=config.tick_size,
            interval=config.num_bars_interval,
            vwap_window=config.num_bars_vwap_window,
            num_levels=config.num_bars_levels,
            centering_method=config.num_bars_centering,
        )
        self.vpin_extractor = _InlineVPINExtractor(
            bucket_volume=config.vpin_bucket_volume,
            window=config.vpin_window,
        )
        self.rasterizer = SequenceRasterizer(
            bins=config.raster_bins,
            span_pct=config.raster_span_pct,
            vol_scale=config.raster_vol_scale,
            price_scale=config.raster_price_scale,
        )

        self._raw_ticks = pd.DataFrame()
        self._bars = pd.DataFrame()
        self._last_refresh_bucket: Optional[pd.Timestamp] = None
        self._last_snapshot: Optional[LiveFeatureSnapshot] = None
        self.trade_ledger = TradeLedger(config.ticker)

    @property
    def last_snapshot(self) -> Optional[LiveFeatureSnapshot]:
        return self._last_snapshot

    @property
    def last_refresh_time(self) -> Optional[pd.Timestamp]:
        return self._last_refresh_bucket

    def should_refresh(self, now: Optional[TimestampLike] = None) -> bool:
        current = self._normalize_now(now)
        bucket = _floor_timestamp(current, self.config.refresh_interval)
        return self._last_refresh_bucket is None or bucket > self._last_refresh_bucket

    def maybe_refresh(self, now: Optional[TimestampLike] = None, force: bool = False) -> LiveFeatureSnapshot:
        if force or self.should_refresh(now):
            return self.refresh(now)
        if self._last_snapshot is None:
            return self.refresh(now)
        return self._last_snapshot

    def refresh(self, now: Optional[TimestampLike] = None) -> LiveFeatureSnapshot:
        current = self._normalize_now(now)
        refresh_bucket = _floor_timestamp(current, self.config.refresh_interval)

        self._update_tick_cache(current)
        if self._raw_ticks.empty:
            self._bars = _ensure_datetime_index(self._bars, self.config.tz)
            if not self._bars.empty:
                self._bars = _ensure_bar_columns(self._bars)
        else:
            self._bars = _resample_ticks_to_bars(self._raw_ticks, self.config.bar_minutes)
        snapshot = self._build_snapshot(current)

        self.trade_ledger.close_due_trades(snapshot.bars, asof=current)
        self._last_refresh_bucket = refresh_bucket
        self._last_snapshot = snapshot
        return snapshot

    def record_prediction(
        self,
        prediction: Union[float, np.ndarray],
        *,
        task: str = "regression",
        threshold: float = 0.0,
        long_class: int = 2,
        short_class: int = 0,
        position: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> TradeRecord:
        snapshot = self._last_snapshot
        if snapshot is None:
            raise RuntimeError("No live snapshot is available. Call refresh() first.")
        return self.trade_ledger.open_trade(
            snapshot,
            prediction,
            task=task,
            threshold=threshold,
            long_class=long_class,
            short_class=short_class,
            position=position,
            meta=meta,
        )

    def reset(self) -> None:
        self._raw_ticks = pd.DataFrame()
        self._bars = pd.DataFrame()
        self._last_refresh_bucket = None
        self._last_snapshot = None
        self.trade_ledger = TradeLedger(self.config.ticker)

    def _normalize_now(self, now: Optional[TimestampLike]) -> pd.Timestamp:
        ts = pd.Timestamp.now(tz=self.config.tz) if now is None else pd.Timestamp(now)
        if ts.tz is None:
            ts = ts.tz_localize(self.config.tz)
        else:
            ts = ts.tz_convert(self.config.tz)
        return ts

    def _update_tick_cache(self, now: pd.Timestamp) -> None:
        lookback_start = now - pd.Timedelta(days=self.config.history_lookback_days)
        overlap_start = lookback_start
        if not self._raw_ticks.empty:
            overlap_start = max(lookback_start, self._raw_ticks.index.max() - pd.Timedelta(hours=2))
        if not self._bars.empty:
            overlap_start = max(lookback_start, self._bars.index.max() - pd.Timedelta(days=2))

        fetched = self.data_source.get_ticks(
            self.config.ticker,
            start_time=overlap_start,
            end_time=now,
        )
        fetched = _ensure_datetime_index(fetched, self.config.tz)

        if _looks_like_bar_data(fetched):
            fetched = _ensure_bar_columns(fetched)
            self._bars = _merge_time_series_frames(
                self._bars,
                fetched,
                lookback_start=lookback_start,
            )
            self._raw_ticks = pd.DataFrame()
        else:
            fetched = _ensure_tick_columns(fetched)
            self._raw_ticks = _merge_time_series_frames(
                self._raw_ticks,
                fetched,
                lookback_start=lookback_start,
            )
            self._bars = pd.DataFrame()

    def _build_snapshot(self, now: pd.Timestamp) -> LiveFeatureSnapshot:
        if self._bars.empty:
            raise ValueError("No market data available to build live features.")

        tech_df, feat_cols = self._build_technical_frame()
        anchor_ts = self._select_anchor_ts(tech_df, now)

        numbars_idx, numbars_data, rasters, seq_vpin, vpin_spatial = self._build_orderflow_modalities(now)
        sample = self._assemble_sample(
            tech_df=tech_df,
            feat_cols=feat_cols,
            anchor_ts=anchor_ts,
            numbars_idx=numbars_idx,
            numbars_data=numbars_data,
            rasters=rasters,
            seq_vpin=seq_vpin,
            vpin_spatial=vpin_spatial,
        )

        return LiveFeatureSnapshot(
            ticker=self.config.ticker,
            refreshed_at=now,
            anchor_ts=anchor_ts,
            target_end_ts=sample["target_end_ts"],
            sample=sample,
            bars=self._bars.copy(),
            raw_ticks=self._raw_ticks.copy(),
            vpin=seq_vpin.copy(),
        )

    def _build_technical_frame(self) -> Tuple[pd.DataFrame, List[str]]:
        steps = self.config.target_horizon_minutes // self.config.bar_minutes
        df_out, _, _ = self.prep.prepare(
            self._bars,
            steps_60m=steps,
            keep_only_active=False,
            apply_scaling=True,
            scale_to_basis_points=True,
        )
        feat_cols = self.prep.get_feature_cols(
            steps_60m=steps,
            bar_minutes=self.config.bar_minutes,
        )
        feat_cols = [col for col in feat_cols if col in df_out.columns]
        df_out[feat_cols] = df_out[feat_cols].ffill().bfill().fillna(0.0)
        return df_out, feat_cols

    def _select_anchor_ts(self, tech_df: pd.DataFrame, now: pd.Timestamp) -> pd.Timestamp:
        eligible = tech_df.loc[tech_df.index <= now].copy()
        if eligible.empty:
            raise ValueError("No technical bars are available before the refresh timestamp.")

        if "is_active" in eligible.columns:
            active = eligible["is_active"].astype(bool)
            if active.any():
                eligible = eligible.loc[active]

        return eligible.index[-1]

    def _iter_feature_dates(self, now: pd.Timestamp) -> List[date]:
        source = self._raw_ticks if not self._raw_ticks.empty else self._bars
        idx = source.index[source.index <= now]
        return sorted(set(idx.date))

    def _build_orderflow_modalities(
        self,
        now: pd.Timestamp,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[date, np.ndarray], pd.DataFrame, pd.DataFrame]:
        dates = self._iter_feature_dates(now)
        rasters: Dict[date, np.ndarray] = {}
        numbars_idx_parts: List[np.ndarray] = []
        numbars_val_parts: List[np.ndarray] = []
        vpin_frames: List[pd.DataFrame] = []

        for d in dates:
            vpin_df, numbars_tuple, raster = self._build_daily_orderflow(d, now)
            if not vpin_df.empty:
                vpin_frames.append(vpin_df)
                rasters[d] = raster
            if numbars_tuple is not None:
                nb_idx, nb_val = numbars_tuple
                if len(nb_idx) > 0:
                    numbars_idx_parts.append(nb_idx)
                    numbars_val_parts.append(nb_val)

        if vpin_frames:
            raw_vpin = pd.concat(vpin_frames, axis=0).sort_index()
            raw_vpin = raw_vpin[~raw_vpin.index.duplicated(keep="last")]
            numeric_vpin = raw_vpin.select_dtypes(include="number")
            seq_vpin = scale_vpin_features(numeric_vpin)
            vpin_spatial = prepare_vpin_spatial_features(numeric_vpin)
        else:
            seq_vpin = pd.DataFrame()
            vpin_spatial = pd.DataFrame()

        if numbars_idx_parts:
            numbars_idx = np.concatenate(numbars_idx_parts)
            numbars_data = np.concatenate(numbars_val_parts, axis=0)
            order = np.argsort(numbars_idx)
            numbars_idx = numbars_idx[order]
            numbars_data = numbars_data[order]
        else:
            bins = (self.config.num_bars_levels * 2) + 1
            numbars_idx = np.array([], dtype="datetime64[ns]")
            numbars_data = np.empty((0, 4, bins), dtype=np.float32)

        return numbars_idx, numbars_data, rasters, seq_vpin, vpin_spatial

    def _build_daily_orderflow(
        self,
        d: date,
        now: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, Optional[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        empty_raster = np.zeros(
            (self.config.raster_num_bars, 4, self.config.raster_bins),
            dtype=np.float32,
        )
        if self._raw_ticks.empty:
            return pd.DataFrame(), None, empty_raster

        current_day_end = min(now, _combine_date_time(d, "23:59", self.config.tz))
        vpin_start = _combine_date_time(d, self.config.vpin_start_time, self.config.tz)
        vpin_end = min(_combine_date_time(d, self.config.vpin_end_time, self.config.tz), current_day_end)

        if vpin_end <= vpin_start:
            return pd.DataFrame(), None, empty_raster

        profile_start = _combine_date_time(d, self.config.profile_start_time, self.config.tz)
        profile_end = min(_combine_date_time(d, self.config.profile_end_time, self.config.tz), current_day_end)
        profile_vwap = None
        if profile_end > profile_start:
            profile_data = self._raw_ticks.loc[
                (self._raw_ticks.index >= profile_start) & (self._raw_ticks.index <= profile_end)
            ]
            _, _, _, profile_vwap = _compute_profile_levels(
                profile_data,
                self.profile_extractor,
                self.config.tick_size,
                self.config.value_area_pct,
            )
        else:
            profile_data = pd.DataFrame()

        ps_poc = ps_val = ps_vah = np.nan
        pd_poc = pd_val = pd_vah = np.nan
        if not profile_data.empty:
            prev_day = d - pd.Timedelta(days=1)
            ps_start = _combine_date_time(prev_day, self.config.profile_start_time, self.config.tz)
            ps_end = _combine_date_time(prev_day, self.config.profile_end_time, self.config.tz)
            ps_data = self._raw_ticks.loc[(self._raw_ticks.index >= ps_start) & (self._raw_ticks.index <= ps_end)]
            ps_poc, ps_val, ps_vah, _ = _compute_profile_levels(
                ps_data,
                self.profile_extractor,
                self.config.tick_size,
                self.config.value_area_pct,
            )

            if self.config.include_prev_24h_profile:
                pd_start = vpin_start - pd.Timedelta(hours=24)
                pd_data = self._raw_ticks.loc[
                    (self._raw_ticks.index >= pd_start) & (self._raw_ticks.index <= vpin_start)
                ]
                pd_poc, pd_val, pd_vah, _ = _compute_profile_levels(
                    pd_data,
                    self.profile_extractor,
                    self.config.tick_size,
                    self.config.value_area_pct,
                )

        ib_high = ib_low = np.nan
        if self.config.include_ib and profile_end > profile_start:
            ib_end = min(profile_start + pd.Timedelta(minutes=self.config.ib_minutes), current_day_end)
            ib_data = self._raw_ticks.loc[
                (self._raw_ticks.index >= profile_start) & (self._raw_ticks.index <= ib_end)
            ]
            if not ib_data.empty:
                ib_high = float(ib_data["Close"].max())
                ib_low = float(ib_data["Close"].min())

        vpin_data = self._raw_ticks.loc[(self._raw_ticks.index >= vpin_start) & (self._raw_ticks.index <= vpin_end)]
        if vpin_data.empty:
            return pd.DataFrame(), None, empty_raster

        vpin_df = self.vpin_extractor.calculate_vpin(
            vpin_data,
            bucket_volume=self.config.vpin_bucket_volume,
            window=self.config.vpin_window,
            include_sequence_features=True,
        )
        if not vpin_df.empty:
            vpin_df["ps_poc"] = ps_poc
            vpin_df["ps_val"] = ps_val
            vpin_df["ps_vah"] = ps_vah
            if self.config.include_prev_24h_profile:
                vpin_df["pd_poc"] = pd_poc
                vpin_df["pd_val"] = pd_val
                vpin_df["pd_vah"] = pd_vah
            if self.config.include_ib:
                vpin_df["ib_high"] = ib_high
                vpin_df["ib_low"] = ib_low

        raster = self.rasterizer.rasterize(
            vpin_df,
            num_bars=self.config.raster_num_bars,
            interval_mins=self.config.raster_interval_mins,
            session_start=self.config.raster_session_start,
        )
        raster_np = np.asarray(raster, dtype=np.float32)

        numbars = self._build_daily_numbars(
            d=d,
            current_day_end=current_day_end,
            profile_vwap=profile_vwap,
        )
        return vpin_df, numbars, raster_np

    def _build_daily_numbars(
        self,
        d: date,
        current_day_end: pd.Timestamp,
        profile_vwap: Optional[float],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        nb_start = _combine_date_time(d, self.config.num_bars_start_time, self.config.tz)
        nb_end = min(_combine_date_time(d, self.config.num_bars_end_time, self.config.tz), current_day_end)
        if nb_end <= nb_start:
            return None

        lookback = pd.Timedelta(self.config.num_bars_vwap_window) * 2
        nb_data = self._raw_ticks.loc[
            (self._raw_ticks.index >= (nb_start - lookback)) & (self._raw_ticks.index <= nb_end)
        ]
        if nb_data.empty:
            return None

        tensor, meta = self.number_bars_extractor.calculate_number_bars(
            df=nb_data,
            start_time=nb_start,
            interval=self.config.num_bars_interval,
            tick_size=self.config.tick_size,
            vwap_window=self.config.num_bars_vwap_window,
            num_levels=self.config.num_bars_levels,
            fixed_center=profile_vwap if profile_vwap is not None and np.isfinite(profile_vwap) else None,
        )
        if tensor.size == 0 or meta.empty:
            return None

        tensor = np.transpose(tensor, (0, 2, 1)).astype(np.float32)
        tensor = scale_numbars(tensor)
        meta_idx = _as_naive(pd.to_datetime(meta["Time"]))
        return meta_idx, tensor

    def _build_ae_input(self, anchor_ts: pd.Timestamp) -> np.ndarray:
        ae_feats = compute_ae_daily_features(
            self._bars,
            target_time=self.config.ae_target_time,
        )
        unique_dates = sorted(set(self._bars.index.date))
        current_date = anchor_ts.date()
        if current_date not in unique_dates:
            return np.zeros((self.config.ae_window, 4), dtype=np.float32)

        date_idx = unique_dates.index(current_date)
        prev_dates = unique_dates[max(0, date_idx - self.config.ae_window):date_idx]
        if len(prev_dates) < self.config.ae_window:
            prev_dates = ([None] * (self.config.ae_window - len(prev_dates))) + prev_dates

        rows: List[np.ndarray] = []
        last_vec = np.zeros(4, dtype=np.float32)
        for d in prev_dates:
            if d is None:
                rows.append(last_vec.copy())
                continue
            if d in ae_feats:
                last_vec = ae_feats[d]
                rows.append(last_vec.copy())
            else:
                rows.append(last_vec.copy())
        return np.asarray(rows, dtype=np.float32)

    def _assemble_sample(
        self,
        tech_df: pd.DataFrame,
        feat_cols: List[str],
        anchor_ts: pd.Timestamp,
        numbars_idx: np.ndarray,
        numbars_data: np.ndarray,
        rasters: Dict[date, np.ndarray],
        seq_vpin: pd.DataFrame,
        vpin_spatial: pd.DataFrame,
    ) -> Dict[str, Any]:
        anchor_pos = tech_df.index.get_loc(anchor_ts)
        tech_slice = tech_df.iloc[max(0, anchor_pos - self.config.tech_lookback):anchor_pos][feat_cols]
        tech_features = _pad_left_2d(
            tech_slice.values.astype(np.float32),
            self.config.tech_lookback,
        )

        if seq_vpin.empty:
            seq_data = np.zeros((1, 1), dtype=np.float32)
            seq_max_ts = pd.NaT
        else:
            seq_cut = seq_vpin.loc[seq_vpin.index < anchor_ts]
            seq_data = seq_cut.tail(self.config.seq_lookback_bars).values.astype(np.float32)
            seq_max_ts = seq_cut.index.max() if not seq_cut.empty else pd.NaT
            if seq_data.size == 0:
                seq_data = np.zeros((1, seq_vpin.shape[1]), dtype=np.float32)

        anchor64 = _as_naive([anchor_ts])[0]
        if len(numbars_idx) > 0:
            cut = np.searchsorted(numbars_idx, anchor64, side="left")
            if cut > 0:
                nb_selected = numbars_data[max(0, cut - self.config.numbars_lookback):cut]
                nb_max_ts = pd.Timestamp(numbars_idx[cut - 1]).tz_localize("UTC").tz_convert(self.config.tz)
            else:
                nb_selected = np.zeros((1, 4, numbars_data.shape[2]), dtype=np.float32)
                nb_max_ts = pd.NaT
        else:
            bins = (self.config.num_bars_levels * 2) + 1
            nb_selected = np.zeros((1, 4, bins), dtype=np.float32)
            nb_max_ts = pd.NaT

        prev_dates = sorted(d for d in set(tech_df.index.date) if d < anchor_ts.date())
        prev_date = prev_dates[-1] if prev_dates else None

        sample: Dict[str, Any] = {
            "tech_features": tech_features,
            "tech_feature_cols": list(feat_cols),
            "tech_len": int(len(tech_slice)),
            "seq_vpin": seq_data,
            "seq_vpin_len": int(len(seq_data)),
            "ae_input": self._build_ae_input(anchor_ts),
            "ticker_id": int(self.ticker_meta.ticker_id),
            "asset_class_id": int(self.ticker_meta.asset_class_id),
            "asset_subclass_id": int(self.ticker_meta.asset_subclass_id),
            "ticker": self.config.ticker,
            "date": anchor_ts.date(),
            "anchor_ts": anchor_ts,
            "target_end_ts": anchor_ts + pd.Timedelta(minutes=self.config.target_horizon_minutes),
            "_feature_meta": {
                "seq_vpin_max_ts": seq_max_ts,
                "numbars_max_ts": nb_max_ts,
                "prev_raster_date": prev_date,
            },
        }

        if self.config.use_fused_spatial:
            sample["fused_spatial"] = self._build_fused_spatial(nb_selected, anchor_ts, vpin_spatial)
        else:
            sample["numbars_recent"] = nb_selected.astype(np.float32)
            if prev_date is not None and prev_date in rasters:
                sample["vpin_raster_recent"] = np.asarray(rasters[prev_date], dtype=np.float32)
            else:
                sample["vpin_raster_recent"] = np.zeros(
                    (self.config.raster_num_bars, 4, self.config.raster_bins),
                    dtype=np.float32,
                )

        return sample

    def _build_fused_spatial(
        self,
        nb_selected: np.ndarray,
        anchor_ts: pd.Timestamp,
        vpin_spatial: pd.DataFrame,
    ) -> np.ndarray:
        fused_tail = (nb_selected.shape[1] + 3, nb_selected.shape[2])
        if nb_selected.size == 0:
            return np.zeros((1,) + fused_tail, dtype=np.float32)

        if vpin_spatial.empty:
            vpin_grids = np.zeros((nb_selected.shape[0], 3, nb_selected.shape[2]), dtype=np.float32)
            return np.concatenate([nb_selected, vpin_grids], axis=1).astype(np.float32)

        spatial_sorted = vpin_spatial.sort_index()
        spatial_idx = _as_naive(spatial_sorted.index)
        rolling_vwap = spatial_sorted["rolling_vwap_2h"].values.astype(np.float32)

        nb_end_ts = self._bars.loc[self._bars.index <= anchor_ts].index[-len(nb_selected):]
        nb_end_idx = _as_naive(nb_end_ts)
        one_hour = np.timedelta64(1, "h")
        vpin_lo = np.searchsorted(spatial_idx, nb_end_idx - one_hour, side="right")
        vpin_hi = np.searchsorted(spatial_idx, nb_end_idx, side="right")

        vpin_grids = np.zeros((nb_selected.shape[0], 3, nb_selected.shape[2]), dtype=np.float32)
        for i in range(nb_selected.shape[0]):
            hi = int(vpin_hi[i])
            if hi <= 0:
                continue
            lo = int(vpin_lo[i])
            window = spatial_sorted.iloc[lo:hi]
            center_price = float(rolling_vwap[hi - 1])
            vpin_grids[i] = rasterize_vpin_to_grid(
                vpin_window=window,
                price_offsets=nb_selected[i, 3],
                center_price=center_price,
            )

        return np.concatenate([nb_selected, vpin_grids], axis=1).astype(np.float32)


@dataclass
class SimulatedOrder:
    target_position: float
    prediction: Optional[float] = None
    submitted_at: Optional[pd.Timestamp] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatedFill:
    ticker: str
    ts: pd.Timestamp
    side: str
    quantity: float
    price: float
    target_position: float
    prediction: Optional[float] = None


@dataclass
class SimulatedTrade:
    ticker: str
    ts_in: pd.Timestamp
    ts_out: pd.Timestamp
    direction: float
    quantity: float
    px_in: float
    px_out: float
    pnl: float
    prediction: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayResult:
    steps: pd.DataFrame
    fills: pd.DataFrame
    trades: pd.DataFrame
    equity_curve: pd.DataFrame

    def assert_backward_looking(self) -> None:
        if self.steps.empty:
            return
        bad = self.steps.loc[~self.steps["backward_looking"].astype(bool)]
        if not bad.empty:
            raise AssertionError(f"Detected forward-looking features in {len(bad)} replay steps.")


class SimulatedTradingServer:
    """Minimal target-position execution server for replay and dry-run testing."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.reset()

    def reset(self) -> None:
        self.current_position = 0.0
        self.entry_ts: Optional[pd.Timestamp] = None
        self.entry_price: Optional[float] = None
        self.entry_prediction: Optional[float] = None
        self.entry_meta: Dict[str, Any] = {}
        self.realized_pnl = 0.0
        self._fills: List[SimulatedFill] = []
        self._trades: List[SimulatedTrade] = []
        self._equity_points: List[Dict[str, Any]] = []

    def process_order(
        self,
        order: SimulatedOrder,
        market_ts: pd.Timestamp,
        market_price: float,
    ) -> List[SimulatedFill]:
        if not np.isfinite(market_price):
            return []

        fills: List[SimulatedFill] = []
        target = float(order.target_position)
        current = float(self.current_position)
        if np.isclose(target, current):
            return fills

        if not np.isclose(current, 0.0):
            fills.append(
                SimulatedFill(
                    ticker=self.ticker,
                    ts=market_ts,
                    side="sell" if current > 0 else "buy",
                    quantity=abs(current),
                    price=float(market_price),
                    target_position=0.0,
                    prediction=order.prediction,
                )
            )
            pnl = self._realize_trade(market_ts, float(market_price))
            self.realized_pnl += pnl

        if not np.isclose(target, 0.0):
            fills.append(
                SimulatedFill(
                    ticker=self.ticker,
                    ts=market_ts,
                    side="buy" if target > 0 else "sell",
                    quantity=abs(target),
                    price=float(market_price),
                    target_position=target,
                    prediction=order.prediction,
                )
            )
            self.current_position = target
            self.entry_ts = market_ts
            self.entry_price = float(market_price)
            self.entry_prediction = order.prediction
            self.entry_meta = dict(order.meta)
        else:
            self.current_position = 0.0
            self.entry_ts = None
            self.entry_price = None
            self.entry_prediction = None
            self.entry_meta = {}

        self._fills.extend(fills)
        return fills

    def _realize_trade(self, exit_ts: pd.Timestamp, exit_price: float) -> float:
        if self.entry_ts is None or self.entry_price is None or np.isclose(self.current_position, 0.0):
            return 0.0

        qty = abs(self.current_position)
        direction = np.sign(self.current_position)
        if direction > 0:
            pnl = qty * ((exit_price / self.entry_price) - 1.0)
        else:
            pnl = qty * ((self.entry_price / exit_price) - 1.0)

        self._trades.append(
            SimulatedTrade(
                ticker=self.ticker,
                ts_in=self.entry_ts,
                ts_out=exit_ts,
                direction=float(direction),
                quantity=float(qty),
                px_in=float(self.entry_price),
                px_out=float(exit_price),
                pnl=float(pnl),
                prediction=self.entry_prediction,
                meta=dict(self.entry_meta),
            )
        )
        return float(pnl)

    def mark_to_market(self, ts: pd.Timestamp, price: float) -> float:
        unrealized = 0.0
        if self.entry_price is not None and not np.isclose(self.current_position, 0.0) and np.isfinite(price):
            qty = abs(self.current_position)
            direction = np.sign(self.current_position)
            if direction > 0:
                unrealized = qty * ((price / self.entry_price) - 1.0)
            else:
                unrealized = qty * ((self.entry_price / price) - 1.0)

        equity = self.realized_pnl + unrealized
        self._equity_points.append(
            {
                "ts": ts,
                "equity": float(equity),
                "realized_pnl": float(self.realized_pnl),
                "unrealized_pnl": float(unrealized),
                "position": float(self.current_position),
            }
        )
        return float(equity)

    def fills_frame(self) -> pd.DataFrame:
        if not self._fills:
            return pd.DataFrame(columns=["ticker", "ts", "side", "quantity", "price", "target_position", "prediction"])
        return pd.DataFrame([fill.__dict__ for fill in self._fills])

    def trades_frame(self) -> pd.DataFrame:
        if not self._trades:
            return pd.DataFrame(columns=["ticker", "ts_in", "ts_out", "direction", "quantity", "px_in", "px_out", "pnl", "prediction"])
        rows = []
        for trade in self._trades:
            row = trade.__dict__.copy()
            meta = row.pop("meta", {})
            row.update(meta)
            rows.append(row)
        return pd.DataFrame(rows)

    def equity_frame(self) -> pd.DataFrame:
        if not self._equity_points:
            return pd.DataFrame(columns=["ts", "equity", "realized_pnl", "unrealized_pnl", "position"])
        return pd.DataFrame(self._equity_points)


class ForwardReplayBacktester:
    """Play the live interface forward from a sample date using simulated execution."""

    def __init__(
        self,
        interface: LiveV3FeatureInterface,
        trading_server: Optional[SimulatedTradingServer] = None,
    ):
        self.interface = interface
        self.trading_server = trading_server or SimulatedTradingServer(interface.config.ticker)

    def run(
        self,
        start_time: TimestampLike,
        end_time: TimestampLike,
        decision_fn: Optional[Callable[[LiveFeatureSnapshot, SimulatedTradingServer], Optional[Union[SimulatedOrder, Dict[str, Any], float, int]]]] = None,
        *,
        flatten_at_end: bool = True,
    ) -> ReplayResult:
        self.interface.reset()
        self.trading_server.reset()

        start_ts = self.interface._normalize_now(start_time)
        end_ts = self.interface._normalize_now(end_time)
        if end_ts < start_ts:
            raise ValueError("end_time must be >= start_time.")

        timeline = pd.date_range(
            start=_floor_timestamp(start_ts, self.interface.config.refresh_interval),
            end=_floor_timestamp(end_ts, self.interface.config.refresh_interval),
            freq=self.interface.config.refresh_interval,
            tz=self.interface.config.tz,
        )

        step_rows: List[Dict[str, Any]] = []
        last_snapshot: Optional[LiveFeatureSnapshot] = None

        for ts in timeline:
            snapshot = self.interface.refresh(ts)
            last_snapshot = snapshot

            feature_meta = snapshot.sample.get("_feature_meta", {})
            seq_max_ts = feature_meta.get("seq_vpin_max_ts", pd.NaT)
            nb_max_ts = feature_meta.get("numbars_max_ts", pd.NaT)
            raw_max_ts = snapshot.raw_ticks.index.max() if not snapshot.raw_ticks.empty else pd.NaT

            backward_looking = True
            if pd.notna(raw_max_ts):
                backward_looking &= raw_max_ts <= snapshot.refreshed_at
            if pd.notna(seq_max_ts):
                backward_looking &= pd.Timestamp(seq_max_ts) < snapshot.anchor_ts
            if pd.notna(nb_max_ts):
                backward_looking &= pd.Timestamp(nb_max_ts) < snapshot.anchor_ts

            decision = decision_fn(snapshot, self.trading_server) if decision_fn is not None else None
            order = self._normalize_decision(decision, ts)
            fill_count = 0
            if order is not None:
                fills = self.trading_server.process_order(order, snapshot.anchor_ts, snapshot.anchor_close)
                fill_count = len(fills)
            equity = self.trading_server.mark_to_market(snapshot.anchor_ts, snapshot.anchor_close)

            step_rows.append(
                {
                    "refresh_ts": snapshot.refreshed_at,
                    "anchor_ts": snapshot.anchor_ts,
                    "target_end_ts": snapshot.target_end_ts,
                    "anchor_close": snapshot.anchor_close,
                    "raw_tick_max_ts": raw_max_ts,
                    "seq_vpin_max_ts": seq_max_ts,
                    "numbars_max_ts": nb_max_ts,
                    "backward_looking": bool(backward_looking),
                    "position": float(self.trading_server.current_position),
                    "equity": float(equity),
                    "fills": int(fill_count),
                    "prediction": order.prediction if order is not None else np.nan,
                    "target_position": order.target_position if order is not None else np.nan,
                }
            )

        if flatten_at_end and last_snapshot is not None and not np.isclose(self.trading_server.current_position, 0.0):
            self.trading_server.process_order(
                SimulatedOrder(target_position=0.0, submitted_at=last_snapshot.refreshed_at),
                last_snapshot.anchor_ts,
                last_snapshot.anchor_close,
            )
            self.trading_server.mark_to_market(last_snapshot.anchor_ts, last_snapshot.anchor_close)

        return ReplayResult(
            steps=pd.DataFrame(step_rows),
            fills=self.trading_server.fills_frame(),
            trades=self.trading_server.trades_frame(),
            equity_curve=self.trading_server.equity_frame(),
        )

    @staticmethod
    def _normalize_decision(
        decision: Optional[Union[SimulatedOrder, Dict[str, Any], float, int]],
        ts: pd.Timestamp,
    ) -> Optional[SimulatedOrder]:
        if decision is None:
            return None
        if isinstance(decision, SimulatedOrder):
            if decision.submitted_at is None:
                decision.submitted_at = ts
            return decision
        if isinstance(decision, dict):
            payload = dict(decision)
            payload.setdefault("submitted_at", ts)
            return SimulatedOrder(**payload)
        if isinstance(decision, (int, float, np.integer, np.floating)):
            return SimulatedOrder(target_position=float(decision), submitted_at=ts)
        raise TypeError(f"Unsupported decision type: {type(decision)!r}")


__all__ = [
    "CsvTickDataSource",
    "ForwardReplayBacktester",
    "InMemoryTickDataSource",
    "LiveEvaluationConfig",
    "LiveFeatureSnapshot",
    "LiveV3FeatureInterface",
    "ReplayResult",
    "SierraTickDataSource",
    "SimulatedFill",
    "SimulatedOrder",
    "SimulatedTrade",
    "SimulatedTradingServer",
    "TickDataSource",
    "TradeLedger",
    "TradeRecord",
]
