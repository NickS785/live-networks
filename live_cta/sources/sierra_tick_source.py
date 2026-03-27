"""Sierra Chart SCID tick data source with GCS sync.

Reads .scid binary files from a local Sierra Chart data folder, stitches
rolling front-month contracts using CTAFlow's expiry rules, and optionally
pushes the result to a :class:`GCSTickDataSource` as a backup data path
for IBKR outages.

Usage
-----
::

    from live_cta.sources.sierra_tick_source import SierraChartTickDataSource, SierraConfig
    from live_cta.sources.gcs_tick_source import GCSConfig, GCSTickerSpec, GCSTickDataSource

    sierra = SierraChartTickDataSource(
        SierraConfig(scid_folder="F:/SierraChart/Data"),
        tickers=["NG", "CL"],
    )

    # Use directly as a TickDataSource
    df = sierra.get_ticks("NG", "2026-03-01", "2026-03-27")

    # Sync to GCS for cloud inference
    gcs = GCSTickDataSource(
        GCSConfig(bucket_name="ctaflow-prod-artifacts"),
        ticker_map={"NG": GCSTickerSpec("live_data/NG_latest.parquet")},
    )
    sierra.sync_to_gcs(gcs, "NG", lookback_days=30)
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from live_cta.core.live import (
    TimestampLike,
    _ensure_datetime_index,
    _ensure_tick_columns,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SierraConfig:
    """Settings for the Sierra Chart SCID data source."""

    scid_folder: str
    """Path to the Sierra Chart data directory containing .scid files."""

    tz: str = "America/Chicago"
    """Timezone for output data."""

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
    """Tick provider that reads Sierra Chart .scid files and stitches
    rolling front-month contracts using CTAFlow's expiry/roll rules.

    Implements the :class:`~live_cta.core.live.TickDataSource` protocol.

    Parameters
    ----------
    config : SierraConfig
        Folder location, timezone, and extraction settings.
    tickers : list[str], optional
        Pre-initialise extractors for these tickers at construction time.
        Additional tickers are initialised lazily on first ``get_ticks`` call.
    """

    def __init__(
        self,
        config: SierraConfig,
        tickers: Optional[List[str]] = None,
    ) -> None:
        self.config = config
        self._extractors: Dict[str, Any] = {}
        self._manager: Optional[Any] = None

        # Eagerly init requested tickers
        for t in (tickers or []):
            self._get_extractor(t)

    # ------------------------------------------------------------------
    # Lazy extractor initialisation
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
        """Read stitched front-month tick data from .scid files.

        Returns a DataFrame indexed by timezone-aware timestamps with at
        minimum ``Close``, ``TotalVolume``, ``BidVolume``, ``AskVolume``,
        ``NumTrades`` columns.
        """
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
        """Extract recent data and upload to GCS as a backup data path.

        Parameters
        ----------
        gcs_source : GCSTickDataSource
            Target GCS source with a ``ticker_map`` entry for *ticker*.
        ticker : str
            Ticker symbol to sync (e.g. ``"NG"``).
        lookback_days : int
            How many calendar days of history to upload.
        fmt : str
            Output format: ``"parquet"`` (default) or ``"csv"``.

        Returns
        -------
        str or None
            The ``gs://`` URI of the uploaded object, or ``None`` on failure.
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

        active = self.active_contract(ticker)
        logger.info(
            "Extracted %d rows for %s (active contract: %s)",
            len(df), ticker, active or "unknown",
        )

        # Write to a temp file, then upload via GCSTickDataSource.upload()
        suffix = ".parquet" if fmt == "parquet" else ".csv"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            if fmt == "parquet":
                df.to_parquet(tmp_path, engine="pyarrow")
            else:
                df.to_csv(tmp_path)

            uri = gcs_source.upload(ticker, tmp_path, fmt=fmt)
            logger.info("GCS sync complete: %s (%d rows)", uri, len(df))
            return uri
        except Exception:
            logger.exception("GCS sync failed for %s", ticker)
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
        for ticker in gcs_source.ticker_map:
            if ticker in self.available_tickers() or ticker in self._extractors:
                results[ticker] = self.sync_to_gcs(
                    gcs_source, ticker, lookback_days=lookback_days, fmt=fmt,
                )
            else:
                logger.debug("Skipping %s — not in local .scid folder", ticker)
        return results
