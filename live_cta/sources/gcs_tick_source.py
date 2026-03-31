"""Google Cloud Storage tick data source for CTAFlow live pipelines.

Architecture: Local data (Sierra Chart) -> GCS bucket -> Cloud Run/VM inference.

Requires ``google-cloud-storage``: ``pip install google-cloud-storage``
"""

from __future__ import annotations

import logging
import tarfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
import tempfile
from typing import Dict, Optional, Union

import pandas as pd

from live_cta.core.live import (
    TimestampLike,
    _ensure_datetime_index,
    _ensure_tick_columns,
)

logger = logging.getLogger(__name__)


def _extract_archived_payload(data: bytes) -> Optional[pd.DataFrame]:
    try:
        with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            parquet_members = [m for m in members if m.name.endswith(".parquet")]
            csv_members = [m for m in members if m.name.endswith(".csv")]
            member = parquet_members[0] if parquet_members else csv_members[0] if csv_members else None
            if member is None:
                logger.warning("No supported data file found in tar.gz archive")
                return pd.DataFrame()
            extracted = tar.extractfile(member)
            if extracted is None:
                return pd.DataFrame()
            payload = extracted.read()
            if member.name.endswith(".parquet"):
                return pd.read_parquet(BytesIO(payload))
            return pd.read_csv(BytesIO(payload), parse_dates=True)
    except tarfile.ReadError:
        return None


def _compress_for_upload(local_path: Union[str, Path]) -> Path:
    source = Path(local_path)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        archive_path = Path(tmp.name)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source, arcname=source.name)
    return archive_path


def _is_tar_gz_path(local_path: Union[str, Path]) -> bool:
    source = Path(local_path)
    return source.suffixes[-2:] == [".tar", ".gz"]

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
                if "project" not in kwargs and getattr(creds, "project_id", None):
                    kwargs["project"] = creds.project_id
            try:
                self._client = gcs_storage.Client(**kwargs)
            except Exception as exc:
                detail = str(exc)
                if "google.auth.default" in detail or "Your default credentials" in detail:
                    raise RuntimeError(
                        "Failed to initialize Google Cloud Storage client. "
                        "Provide `GCS_PROJECT` and/or `GOOGLE_APPLICATION_CREDENTIALS`, "
                        "or pass `project`/`credentials_path` in `GCSConfig`."
                    ) from exc
                raise
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

        archived = _extract_archived_payload(data)
        if archived is not None:
            return archived
        if spec.fmt == "parquet":
            return pd.read_parquet(BytesIO(data))
        elif spec.fmt == "csv":
            return pd.read_csv(BytesIO(data), parse_dates=True)
        elif spec.fmt == "tar.gz":
            with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tar:
                members = [m for m in tar.getmembers() if m.name.endswith(".parquet")]
                if not members:
                    logger.warning("No .parquet file found in tar.gz archive")
                    return pd.DataFrame()
                f = tar.extractfile(members[0])
                return pd.read_parquet(BytesIO(f.read()))
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
        source_path = Path(local_path)
        archive_path = source_path if _is_tar_gz_path(source_path) else _compress_for_upload(source_path)
        try:
            blob.upload_from_filename(str(archive_path), content_type="application/gzip")
            uri = f"gs://{self.config.bucket_name}/{spec.file_path}"
            logger.info("Uploaded %s as compressed %s payload -> %s", local_path, target_fmt, uri)
            return uri
        finally:
            if archive_path != source_path:
                archive_path.unlink(missing_ok=True)
