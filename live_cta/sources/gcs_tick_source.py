"""Google Cloud Storage tick data source for CTAFlow live pipelines.

Architecture: Local data (Sierra Chart) -> GCS bucket -> Cloud Run/VM inference.

Requires ``google-cloud-storage``: ``pip install google-cloud-storage``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from live_cta.core.live import (
    TimestampLike,
    _ensure_datetime_index,
    _ensure_tick_columns,
)

logger = logging.getLogger(__name__)

try:
    from google.cloud import storage as gcs_storage

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    gcs_storage = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GCSConfig:
    """Connection settings for a GCS-backed tick data source."""

    bucket_name: str
    prefix: str = "live_data/"
    project: Optional[str] = None
    credentials_path: Optional[str] = None


@dataclass
class GCSTickerSpec:
    """Maps a CTAFlow ticker to its GCS object path and format."""

    file_path: str  # e.g. "live_data/ng_latest.parquet"
    fmt: str = "parquet"  # "parquet" or "csv"


# ---------------------------------------------------------------------------
# GCS Tick Data Source
# ---------------------------------------------------------------------------

class GCSTickDataSource:
    """Live data provider fetching bars from a Google Cloud Storage bucket.

    Implements the :class:`~CTAFlow.models.evaluation.live.TickDataSource`
    protocol so it can be plugged directly into
    :class:`~CTAFlow.models.evaluation.live.LiveV3FeatureInterface`.

    Parameters
    ----------
    config : GCSConfig
        GCS bucket and project settings.
    ticker_map : dict[str, GCSTickerSpec]
        Mapping of CTAFlow ticker names to GCS object specs.
        Example: ``{"NG": GCSTickerSpec("live_data/ng_latest.parquet")}``
    tz : str
        Target timezone for output data (default ``America/Chicago``).
    """

    def __init__(
        self,
        config: GCSConfig,
        ticker_map: Dict[str, GCSTickerSpec],
        tz: str = "America/Chicago",
    ) -> None:
        if not GCS_AVAILABLE:
            raise ImportError(
                "google-cloud-storage is required for GCSTickDataSource. "
                "Install with: pip install google-cloud-storage"
            )
        self.config = config
        self.ticker_map = ticker_map
        self.tz = tz
        self._client: Optional[gcs_storage.Client] = None
        self._bucket = None

    # ------------------------------------------------------------------
    # Lazy client init
    # ------------------------------------------------------------------

    def _get_bucket(self):
        if self._bucket is None:
            kwargs = {}
            if self.config.project:
                kwargs["project"] = self.config.project
            if self.config.credentials_path:
                from google.oauth2 import service_account

                creds = service_account.Credentials.from_service_account_file(
                    self.config.credentials_path
                )
                kwargs["credentials"] = creds
            self._client = gcs_storage.Client(**kwargs)
            self._bucket = self._client.bucket(self.config.bucket_name)
        return self._bucket

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _download_df(self, spec: GCSTickerSpec) -> pd.DataFrame:
        """Download a single object from GCS and parse into a DataFrame."""
        bucket = self._get_bucket()
        blob = bucket.blob(spec.file_path)

        if not blob.exists():
            logger.warning("GCS object not found: gs://%s/%s", self.config.bucket_name, spec.file_path)
            return pd.DataFrame()

        data = blob.download_as_bytes()
        logger.info(
            "Downloaded %d bytes from gs://%s/%s",
            len(data), self.config.bucket_name, spec.file_path,
        )

        if spec.fmt == "parquet":
            return pd.read_parquet(BytesIO(data))
        elif spec.fmt == "csv":
            return pd.read_csv(BytesIO(data), parse_dates=True)
        else:
            raise ValueError(f"Unsupported format: {spec.fmt!r}")

    def get_ticks(
        self,
        ticker: str,
        start_time: TimestampLike,
        end_time: TimestampLike,
    ) -> pd.DataFrame:
        """Fetch tick/bar data for *ticker* within the given time window.

        Downloads the latest file from GCS, normalises columns/timezone,
        and filters to ``[start_time, end_time]``.
        """
        if ticker not in self.ticker_map:
            logger.warning("Ticker %r not in GCS ticker_map", ticker)
            return pd.DataFrame()

        df = self._download_df(self.ticker_map[ticker])
        if df.empty:
            return df

        df = _ensure_datetime_index(df, self.tz)
        df = _ensure_tick_columns(df)

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

    # ------------------------------------------------------------------
    # Upload helper (run on local feeder machine)
    # ------------------------------------------------------------------

    def upload(
        self,
        ticker: str,
        local_path: Union[str, Path],
        *,
        fmt: Optional[str] = None,
    ) -> str:
        """Upload a local file to GCS for the given ticker.

        Returns the ``gs://`` URI of the uploaded object.
        """
        spec = self.ticker_map[ticker]
        target_fmt = fmt or spec.fmt
        bucket = self._get_bucket()
        blob = bucket.blob(spec.file_path)

        content_type = "application/octet-stream"
        if target_fmt == "parquet":
            content_type = "application/octet-stream"
        elif target_fmt == "csv":
            content_type = "text/csv"

        blob.upload_from_filename(str(local_path), content_type=content_type)
        uri = f"gs://{self.config.bucket_name}/{spec.file_path}"
        logger.info("Uploaded %s -> %s", local_path, uri)
        return uri
