"""Sierra Chart tick data source with GCS sync.

Reads Sierra Chart data from CSV tick exports (default, denser) or .scid
binary files (fallback, requires CTAFlow). The ``csv_map`` field in
:class:`SierraConfig` controls which tickers use CSV; tickers without a
CSV mapping fall back to SCID extraction.

Usage
-----
::

    from live_cta.sources.sierra_tick_source import SierraChartTickDataSource, SierraConfig

    # CSV by default (denser tick data)
    sierra = SierraChartTickDataSource(
        SierraConfig(
            scid_folder="F:/SierraChart/Data",
            csv_map={"NG": "ng.csv"},
        ),
        tickers=["NG"],
    )
    df = sierra.get_ticks("NG", "2026-03-01", "2026-03-27")

    # Sync to GCS
    from live_cta.sources.gcs_tick_source import GCSConfig, GCSTickerSpec, GCSTickDataSource
    gcs = GCSTickDataSource(
        GCSConfig(bucket_name="ctaflow-prod-artifacts"),
        ticker_map={"NG": GCSTickerSpec("live_data/NG_latest.parquet")},
    )
    sierra.sync_to_gcs(gcs, "NG", lookback_days=30)
"""

from __future__ import annotations

import logging
import tarfile
import tempfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from live_cta.core.live import (
    TimestampLike,
    _ensure_datetime_index,
    _ensure_tick_columns,
)
from live_cta.pipelines import list_pipeline_names, model_requires_tick_data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Upload strategy per model type
# ---------------------------------------------------------------------------

NEEDS_TICK_DATA: Dict[str, bool] = {
    name: model_requires_tick_data(name)
    for name in list_pipeline_names(include_aliases=True)
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SierraConfig:
    """Settings for the Sierra Chart data source."""

    scid_folder: str
    """Path to the Sierra Chart data directory (contains .scid and/or CSV files)."""

    tz: str = "America/Chicago"
    """Timezone for output data."""

    csv_map: Optional[Dict[str, str]] = None
    """Maps ticker -> CSV filename for Sierra Chart tick exports.
    When set, ``get_ticks`` reads the CSV instead of SCID binary.
    Example: ``{"NG": "ng.csv"}``"""

    contract_cache: Optional[str] = None
    """Path to a pickled contract-map cache (avoids slow directory rescans).
    Pass ``None`` on first run; call :meth:`save_contract_cache` afterward."""

    resample_rule: Optional[str] = None
    """If set (e.g. ``'1min'``), resample raw ticks to OHLCV bars."""

    columns: List[str] = field(default_factory=lambda: [
        "Close", "TotalVolume", "BidVolume", "AskVolume", "NumTrades",
    ])
    """Columns to extract from .scid files."""


# ---------------------------------------------------------------------------
# Sierra Chart Tick Data Source
# ---------------------------------------------------------------------------

class SierraChartTickDataSource:
    """Tick provider that reads Sierra Chart data (CSV or SCID).

    When ``config.csv_map`` contains a mapping for a ticker, the CSV export
    is used (denser, true tick-by-tick). Otherwise falls back to SCID binary
    extraction via CTAFlow's ``ScidBaseExtractor``.

    Implements the :class:`~live_cta.core.live.TickDataSource` protocol.

    Parameters
    ----------
    config : SierraConfig
        Folder location, CSV mapping, timezone, and extraction settings.
    tickers : list[str], optional
        Pre-initialise SCID extractors for these tickers at construction time.
        CSV tickers don't need pre-initialisation.
    """

    def __init__(
        self,
        config: SierraConfig,
        tickers: Optional[List[str]] = None,
    ) -> None:
        self.config = config
        self._extractors: Dict[str, Any] = {}
        self._manager: Optional[Any] = None

        # Eagerly init SCID extractors for tickers without CSV mapping
        csv_map = config.csv_map or {}
        for t in (tickers or []):
            if t not in csv_map:
                self._get_extractor(t)

    # ------------------------------------------------------------------
    # CSV reading
    # ------------------------------------------------------------------

    def _read_csv(self, ticker: str) -> pd.DataFrame:
        """Read and parse the Sierra Chart CSV export for *ticker*."""
        csv_map = self.config.csv_map or {}
        filename = csv_map.get(ticker)
        if filename is None:
            return pd.DataFrame()

        csv_path = Path(self.config.scid_folder) / filename
        if not csv_path.exists():
            logger.warning("CSV file not found: %s", csv_path)
            return pd.DataFrame()

        logger.info("Reading CSV: %s", csv_path)
        raw = pd.read_csv(csv_path)
        raw.columns = raw.columns.str.strip()

        # Parse Date + Time -> DatetimeIndex
        raw["Datetime"] = pd.to_datetime(
            raw["Date"].str.strip() + " " + raw["Time"].str.strip(),
            format="mixed",
        )
        raw = raw.set_index("Datetime").drop(columns=["Date", "Time"], errors="ignore")

        # Rename to standard columns
        rename = {}
        for col in raw.columns:
            lc = col.lower().strip()
            if lc == "last":
                rename[col] = "Close"
            elif lc == "volume":
                rename[col] = "TotalVolume"
            elif lc == "numberoftrades":
                rename[col] = "NumTrades"
        if rename:
            raw = raw.rename(columns=rename)

        return raw

    # ------------------------------------------------------------------
    # Lazy SCID extractor initialisation
    # ------------------------------------------------------------------

    def _get_manager(self) -> Any:
        """Shared SmartScidManager across all tickers."""
        if self._manager is None:
            from CTAFlow.features.base_extractor import SmartScidManager

            self._manager = SmartScidManager(
                self.config.scid_folder,
                cache_path=self.config.contract_cache,
            )
        return self._manager

    def _get_extractor(self, ticker: str) -> Any:
        if ticker not in self._extractors:
            from CTAFlow.features.base_extractor import ScidBaseExtractor

            ext = ScidBaseExtractor.__new__(ScidBaseExtractor)
            ext.manager = self._get_manager()
            ext.ticker = ticker
            ext.tz = self.config.tz
            self._extractors[ticker] = ext
            logger.info("Initialised SCID extractor for %s", ticker)
        return self._extractors[ticker]

    # ------------------------------------------------------------------
    # TickDataSource protocol
    # ------------------------------------------------------------------

    def get_ticks(
        self,
        ticker: str,
        start_time: TimestampLike,
        end_time: TimestampLike,
    ) -> pd.DataFrame:
        """Read tick data, preferring CSV when available, else SCID.

        Returns a DataFrame indexed by timezone-aware timestamps with at
        minimum ``Close``, ``TotalVolume``, ``BidVolume``, ``AskVolume``,
        ``NumTrades`` columns.
        """
        csv_map = self.config.csv_map or {}

        if ticker in csv_map:
            df = self._read_csv(ticker)
            if df.empty:
                return df
            df = _ensure_datetime_index(df, self.config.tz)
            df = _ensure_tick_columns(df)

            # Filter to requested window
            start_ts = pd.Timestamp(start_time)
            end_ts = pd.Timestamp(end_time)
            if start_ts.tz is None:
                start_ts = start_ts.tz_localize(self.config.tz)
            else:
                start_ts = start_ts.tz_convert(self.config.tz)
            if end_ts.tz is None:
                end_ts = end_ts.tz_localize(self.config.tz)
            else:
                end_ts = end_ts.tz_convert(self.config.tz)

            df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
            logger.info("CSV: %d ticks for %s [%s -> %s]", len(df), ticker, start_ts, end_ts)
        else:
            # SCID fallback
            extractor = self._get_extractor(ticker)
            df = extractor.get_stitched_data(
                start_time=start_time,
                end_time=end_time,
                columns=self.config.columns,
            )
            if df.empty:
                return df
            df = _ensure_datetime_index(df, self.config.tz)
            df = _ensure_tick_columns(df)

        if self.config.resample_rule:
            from live_cta.core.live import _resample_ticks_to_bars
            df = _resample_ticks_to_bars(df, int(self.config.resample_rule.rstrip("min")))

        return df

    # ------------------------------------------------------------------
    # Contract info helpers
    # ------------------------------------------------------------------

    def available_tickers(self) -> List[str]:
        """Return tickers discovered in the .scid folder."""
        mgr = self._get_manager()
        return list(mgr._ticker_contracts.keys())

    def active_contract(self, ticker: str) -> Optional[str]:
        """Return the contract ID currently active for *ticker* (e.g. ``'N25'``)."""
        mgr = self._get_manager()
        contracts = mgr.get_contracts_for_ticker(ticker)
        if not contracts:
            return None

        now_utc = pd.Timestamp.now("UTC")
        from CTAFlow.data.contract_expiry_rules import get_roll_buffer_days

        try:
            roll_days = get_roll_buffer_days(f"{ticker}_F")
        except Exception:
            roll_days = 0
        buffer = pd.Timedelta(days=roll_days)

        for contract in contracts:
            expiry = mgr._calculate_contract_expiry(contract)
            if expiry.tz is None:
                expiry = expiry.tz_localize("UTC")
            roll_date = expiry - buffer
            if now_utc < roll_date:
                return contract.contract_id
        # Past all known contracts — return last
        return contracts[-1].contract_id if contracts else None

    def save_contract_cache(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Persist the contract map for fast reloading on next start."""
        return self._get_manager().save_contract_cache(path)

    # ------------------------------------------------------------------
    # GCS sync
    # ------------------------------------------------------------------

    def sync_to_gcs(
        self,
        gcs_source: "GCSTickDataSource",
        ticker: str,
        lookback_days: int = 30,
        *,
        fmt: str = "parquet",
    ) -> Optional[str]:
        """Extract recent data and upload to GCS.

        Only the lookback window is uploaded, not the full history.
        """
        end = pd.Timestamp.now(tz=self.config.tz)
        start = end - pd.Timedelta(days=lookback_days)

        logger.info(
            "Syncing %s to GCS [%s -> %s] (%d days)",
            ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), lookback_days,
        )

        df = self.get_ticks(ticker, start, end)
        if df.empty:
            logger.warning("No data for %s in [%s, %s] — skipping GCS sync", ticker, start, end)
            return None

        logger.info("Extracted %d rows for %s", len(df), ticker)

        suffix = ".parquet" if fmt == "parquet" else ".csv"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            if fmt == "parquet":
                df.to_parquet(tmp_path, engine="pyarrow")
            else:
                df.to_csv(tmp_path)

            size_kb = tmp_path.stat().st_size / 1024
            logger.info("Wrote %.1f KB for upload", size_kb)

            uri = gcs_source.upload(ticker, tmp_path, fmt=fmt)
            logger.info("GCS sync complete: %s (%d rows)", uri, len(df))
            return uri
        except Exception:
            logger.exception("GCS sync failed for %s", ticker)
            return None
        finally:
            tmp_path.unlink(missing_ok=True)

    def sync_to_gcs_compressed(
        self,
        gcs_source: "GCSTickDataSource",
        ticker: str,
        lookback_days: int = 30,
        resample_rule: str = "5min",
    ) -> Optional[str]:
        """Resample tick data to bars, compress as tar.gz, and upload to GCS.

        Only the lookback window is read and uploaded.
        """
        from live_cta.core.live import _resample_ticks_to_bars

        end = pd.Timestamp.now(tz=self.config.tz)
        start = end - pd.Timedelta(days=lookback_days)

        logger.info(
            "Syncing %s to GCS (compressed, %s bars) [%s -> %s]",
            ticker, resample_rule,
            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
        )

        df = self.get_ticks(ticker, start, end)
        if df.empty:
            logger.warning("No data for %s — skipping compressed GCS sync", ticker)
            return None

        bar_minutes = int(resample_rule.rstrip("minTSH"))
        bars = _resample_ticks_to_bars(df, bar_minutes)
        logger.info(
            "Resampled %d ticks -> %d bars (%s) for %s",
            len(df), len(bars), resample_rule, ticker,
        )

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            pq_buf = BytesIO()
            bars.to_parquet(pq_buf, engine="pyarrow")
            pq_bytes = pq_buf.getvalue()

            with tarfile.open(tmp_path, "w:gz") as tar:
                info = tarfile.TarInfo(name=f"{ticker.lower()}_bars.parquet")
                info.size = len(pq_bytes)
                tar.addfile(info, BytesIO(pq_bytes))

            logger.info(
                "Compressed archive: %.1f KB (%d ticks -> %d bars)",
                tmp_path.stat().st_size / 1024, len(df), len(bars),
            )

            uri = gcs_source.upload(ticker, tmp_path, fmt="tar.gz")
            logger.info("Compressed GCS sync complete: %s", uri)
            return uri
        except Exception:
            logger.exception("Compressed GCS sync failed for %s", ticker)
            return None
        finally:
            tmp_path.unlink(missing_ok=True)

    def sync_all_to_gcs(
        self,
        gcs_source: "GCSTickDataSource",
        lookback_days: int = 30,
        fmt: str = "parquet",
    ) -> Dict[str, Optional[str]]:
        """Sync all tickers in the GCS ticker_map that are also available locally."""
        results: Dict[str, Optional[str]] = {}
        csv_map = self.config.csv_map or {}
        for ticker in gcs_source.ticker_map:
            if ticker in csv_map or ticker in self.available_tickers() or ticker in self._extractors:
                results[ticker] = self.sync_to_gcs(
                    gcs_source, ticker, lookback_days=lookback_days, fmt=fmt,
                )
            else:
                logger.debug("Skipping %s — not in csv_map or .scid folder", ticker)
        return results
