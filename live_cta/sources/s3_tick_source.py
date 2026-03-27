"""S3-compatible tick data source for CTAFlow live pipelines (RunPod / AWS).

Architecture: Local data (Sierra Chart) -> S3 bucket -> RunPod VM inference.

Reuses the existing :class:`~CTAFlow.data.storage.aws_client.AWSClient` for
all S3 operations, keeping a single boto3 dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from live_cta.storage.aws_client import AWSClient, S3Config, BOTO3_AVAILABLE
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
class S3TickerSpec:
    """Maps a CTAFlow ticker to its S3 object key and format."""

    key: str  # e.g. "live_data/ng_latest.parquet"
    fmt: str = "parquet"  # "parquet" or "csv"


# ---------------------------------------------------------------------------
# S3 Tick Data Source
# ---------------------------------------------------------------------------

class S3TickDataSource:
    """Live data provider fetching bars from an S3-compatible bucket.

    Implements the :class:`~CTAFlow.models.evaluation.live.TickDataSource`
    protocol so it can be plugged directly into
    :class:`~CTAFlow.models.evaluation.live.LiveV3FeatureInterface`.

    Works with AWS S3, RunPod network volumes (via S3-compatible endpoint),
    or any MinIO/Ceph endpoint.

    Parameters
    ----------
    s3_config : S3Config
        Bucket name, region, endpoint, and credentials.
    ticker_map : dict[str, S3TickerSpec]
        Mapping of CTAFlow ticker names to S3 object specs.
    tz : str
        Target timezone for output data (default ``America/Chicago``).
    cache_dir : str or Path, optional
        Local cache directory. When set, downloaded files are cached
        and only re-fetched when the remote object is newer.
    """

    def __init__(
        self,
        s3_config: S3Config,
        ticker_map: Dict[str, S3TickerSpec],
        tz: str = "America/Chicago",
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for S3TickDataSource. "
                "Install with: pip install boto3"
            )
        self.ticker_map = ticker_map
        self.tz = tz
        self._aws = AWSClient(config=s3_config, cache_dir=cache_dir)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _download_df(self, spec: S3TickerSpec) -> pd.DataFrame:
        """Download a single object from S3 and parse into a DataFrame."""
        full_key = self._aws._add_prefix(spec.key)

        if not self._aws.object_exists(spec.key):
            logger.warning(
                "S3 object not found: s3://%s/%s",
                self._aws.config.bucket_name, full_key,
            )
            return pd.DataFrame()

        # Download to a temporary in-memory buffer via the client
        data = self._aws.s3_client.get_object(
            Bucket=self._aws.config.bucket_name, Key=full_key
        )["Body"].read()
        logger.info(
            "Downloaded %d bytes from s3://%s/%s",
            len(data), self._aws.config.bucket_name, full_key,
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

        Downloads the latest file from S3, normalises columns/timezone,
        and filters to ``[start_time, end_time]``.
        """
        if ticker not in self.ticker_map:
            logger.warning("Ticker %r not in S3 ticker_map", ticker)
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
        """Upload a local file to S3 for the given ticker.

        Returns the ``s3://`` URI of the uploaded object.
        """
        spec = self.ticker_map[ticker]
        self._aws.upload_file(str(local_path), spec.key)
        full_key = self._aws._add_prefix(spec.key)
        uri = f"s3://{self._aws.config.bucket_name}/{full_key}"
        logger.info("Uploaded %s -> %s", local_path, uri)
        return uri
