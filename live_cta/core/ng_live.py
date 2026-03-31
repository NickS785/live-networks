"""NatGas-specific live inference interface.

Extends :class:`LiveV3FeatureInterface` with daily EIA storage,
weather (HDD/CDD), and VAE regime features required by the
:class:`~CTAFlow.models.deep_learning.multi_branch.ng_moe.HybridMixtureNetwork`
and :class:`~CTAFlow.models.deep_learning.multi_branch.ng_moe.NatGasMoE` models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .live import (
    LiveEvaluationConfig,
    LiveFeatureSnapshot,
    LiveV3FeatureInterface,
    SessionSpec,
    TickDataSource,
    TimestampLike,
)

logger = logging.getLogger(__name__)


def _normalize_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a DataFrame's DatetimeIndex uses nanosecond resolution (tz-naive).

    Prevents ``TypeError: Invalid comparison between dtype=datetime64[us] and
    Timestamp`` on pandas < 2.1 where mixed resolutions aren't auto-coerced.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    try:
        df.index = df.index.as_unit("ns")
    except AttributeError:
        pass  # pandas < 2.0 already uses ns
    return df


def _safe_date_mask(index: pd.DatetimeIndex, anchor: pd.Timestamp, op: str = "le") -> pd.Series:
    """Compare a DatetimeIndex against a Timestamp, handling tz/resolution mismatches.

    Strips timezone from both sides and compares as naive timestamps to avoid
    ``TypeError: Invalid comparison between dtype=datetime64[us] and Timestamp``.
    """
    idx = index.tz_localize(None) if index.tz is not None else index
    try:
        idx = idx.as_unit("ns")
    except AttributeError:
        pass
    ref = anchor.tz_localize(None) if anchor.tzinfo is not None else anchor
    if op == "le":
        return idx <= ref
    elif op == "ge":
        return idx >= ref
    return idx == ref

# ---------------------------------------------------------------------------
# Weather feature processing (self-contained for inference without macrOS-Int)
# ---------------------------------------------------------------------------

_BASE_TEMP_C = 18.33  # 65°F in Celsius

try:
    from sklearn.preprocessing import SplineTransformer

    _SPLINE_AVAILABLE = True
except ImportError:
    _SPLINE_AVAILABLE = False
    SplineTransformer = None  # type: ignore


def _compute_degree_days(
    daily_weather: pd.DataFrame,
    temp_col: str = "wtd_TAVG",
    base_temp: float = _BASE_TEMP_C,
) -> pd.DataFrame:
    """Compute HDD/CDD from population-weighted daily temperatures."""
    if temp_col not in daily_weather.columns:
        raise KeyError(
            f"Column {temp_col!r} not found. Available: {list(daily_weather.columns)}"
        )
    tavg = daily_weather[temp_col]
    hdd = (base_temp - tavg).clip(lower=0)
    cdd = (tavg - base_temp).clip(lower=0)
    return pd.DataFrame({"HDD": hdd, "CDD": cdd}, index=daily_weather.index)


def _compute_spline_hdd_basis(
    hdd: pd.Series,
    n_knots: int = 4,
    transformer=None,
) -> Tuple[pd.DataFrame, object]:
    """Natural cubic spline basis expansion for HDD."""
    if not _SPLINE_AVAILABLE:
        raise ImportError("scikit-learn required for spline HDD basis")
    values = hdd.values.reshape(-1, 1)
    if transformer is None:
        transformer = SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            extrapolation="linear",
            include_bias=False,
        )
        transformer.fit(values)
    basis = transformer.transform(values)
    cols = [f"HDD_sp{i}" for i in range(basis.shape[1])]
    return pd.DataFrame(basis, index=hdd.index, columns=cols), transformer


def process_weather_features(
    daily_weather: pd.DataFrame,
    spline_transformer=None,
    n_knots: int = 4,
) -> Tuple[pd.DataFrame, Optional[object]]:
    """Build the 13 weather features used by HybridMixtureNetwork.

    Input ``daily_weather`` must have a ``wtd_TAVG`` column (population-weighted
    average temperature from ``PopulationWeatherGrid``).

    Returns
    -------
    features : DataFrame
        Columns: ``dd_hdd``, ``dd_cdd``, ``dd_hdd_7d``, ``dd_cdd_7d``,
        ``dd_hdd_7d_chg``, ``dd_cdd_7d_chg``, ``dd_hdd_sp0`` .. ``dd_hdd_sp4``,
        ``dd_wtd_tavg``, ``dd_wtd_tavg_7d``.
    transformer : SplineTransformer or None
        Fitted transformer (pass back for consistent live inference).
    """
    dd = _compute_degree_days(daily_weather)

    dd["HDD_7d"] = dd["HDD"].rolling(7, min_periods=3).sum()
    dd["CDD_7d"] = dd["CDD"].rolling(7, min_periods=3).sum()
    dd["HDD_7d_chg"] = dd["HDD_7d"] - dd["HDD_7d"].shift(7)
    dd["CDD_7d_chg"] = dd["CDD_7d"] - dd["CDD_7d"].shift(7)

    dd_cols = ["HDD", "CDD", "HDD_7d", "CDD_7d", "HDD_7d_chg", "CDD_7d_chg"]

    # Spline basis on 7-day HDD
    fitted_transformer = spline_transformer
    if _SPLINE_AVAILABLE:
        try:
            hdd_7d = dd["HDD_7d"].fillna(0)
            spline_df, fitted_transformer = _compute_spline_hdd_basis(
                hdd_7d, n_knots=n_knots, transformer=spline_transformer,
            )
            for sc in spline_df.columns:
                dd[sc] = spline_df[sc].values
                dd_cols.append(sc)
        except Exception as exc:
            logger.warning("Spline HDD basis failed: %s", exc)

    # Population-weighted temperature
    if "wtd_TAVG" in daily_weather.columns:
        dd["wtd_tavg"] = daily_weather["wtd_TAVG"].values
        dd["wtd_tavg_7d"] = (
            daily_weather["wtd_TAVG"].rolling(7, min_periods=3).mean().values
        )
        dd_cols.extend(["wtd_tavg", "wtd_tavg_7d"])

    # Rename with dd_ prefix for feature group identification
    rename = {c: f"dd_{c.lower()}" for c in dd_cols}
    features = dd[dd_cols].rename(columns=rename)
    return features, fitted_transformer


# Weather feature column names (13 total, matching notebook training)
WEATHER_FEATURE_COLS: List[str] = [
    "dd_hdd", "dd_cdd", "dd_hdd_7d", "dd_cdd_7d",
    "dd_hdd_7d_chg", "dd_cdd_7d_chg",
    "dd_hdd_sp0", "dd_hdd_sp1", "dd_hdd_sp2", "dd_hdd_sp3", "dd_hdd_sp4",
    "dd_wtd_tavg", "dd_wtd_tavg_7d",
]


# ---------------------------------------------------------------------------
# Default config tuned for NG 30-min inference
# ---------------------------------------------------------------------------

def ng_default_config(**overrides: Any) -> LiveEvaluationConfig:
    """Return a :class:`LiveEvaluationConfig` pre-tuned for NG futures."""
    defaults = dict(
        ticker="NG",
        tick_size=0.001,
        tz="America/Chicago",
        bar_minutes=30,
        target_horizon_minutes=60,
        refresh_interval="30min",
        history_lookback_days=400,
        sessions=[SessionSpec("USA", "08:30", "16:00")],
        vpin_bucket_volume=150,
        vpin_window=60,
        vpin_start_time="08:30",
        vpin_end_time="16:00",
        profile_start_time="02:00",
        profile_end_time="09:30",
        ae_window=21,
        ae_target_time="10:00",
    )
    defaults.update(overrides)
    return LiveEvaluationConfig(**defaults)


# ---------------------------------------------------------------------------
# Daily context cache specification
# ---------------------------------------------------------------------------

@dataclass
class DailyContextPaths:
    """Paths to cached daily context files (refreshed by external cron)."""

    eia_storage_path: Optional[str] = None
    weather_path: Optional[str] = None
    daily_features_path: Optional[str] = None
    spline_transformer_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Regime feature spec (12 features expected by VAERegimeEncoder)
# ---------------------------------------------------------------------------

REGIME_FEATURES: List[str] = [
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "rv_5d",
    "rv_21d",
    "pct_in_5y_band",
    "dev_from_5y_mean_zscore",
    "band_width_pct",
    "forecast_vs_seasonal_zscore",
    "change_vs_seasonal_zscore",
    "is_injection_season",
    "dev_x_season",
]


# ---------------------------------------------------------------------------
# NatGas Live Interface
# ---------------------------------------------------------------------------

class NatGasLiveInterface(LiveV3FeatureInterface):
    """Live feature interface with NG-specific daily context injection.

    Wraps the standard :class:`LiveV3FeatureInterface` refresh cycle and
    appends daily EIA storage, weather, and VAE regime features to the
    snapshot before inference.

    Parameters
    ----------
    config : LiveEvaluationConfig
        Pipeline configuration (use :func:`ng_default_config` for defaults).
    data_source : TickDataSource
        Tick/bar provider (e.g. :class:`IBKRTickDataSource`).
    daily_paths : DailyContextPaths, optional
        Paths to pre-cached daily feature files.
    ae_window : int
        Lookback window for VAE regime encoder input (default 21).
    """

    def __init__(
        self,
        config: LiveEvaluationConfig,
        data_source: TickDataSource,
        daily_paths: Optional[DailyContextPaths] = None,
        ae_window: int = 21,
        feature_cols: Optional[Sequence[str]] = None,
        spline_transformer=None,
    ) -> None:
        super().__init__(config, data_source)
        self.daily_paths = daily_paths or DailyContextPaths()
        self.ae_window = ae_window
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self._missing_feature_cols_logged = False

        # Caches for daily data (loaded lazily or refreshed externally)
        self._eia_cache: Optional[pd.DataFrame] = None
        self._weather_cache: Optional[pd.DataFrame] = None
        self._weather_features_cache: Optional[pd.DataFrame] = None
        self._daily_features_cache: Optional[pd.DataFrame] = None
        self._cache_date: Optional[pd.Timestamp] = None
        self._spline_transformer = spline_transformer or self._load_spline_transformer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_inference_snapshot(
        self,
        now: Optional[TimestampLike] = None,
    ) -> LiveFeatureSnapshot:
        """Build a complete feature snapshot for NG model inference.

        1. Refreshes live tick data and builds technical/orderflow features
           via the parent :meth:`refresh`.
        2. Loads and appends daily EIA/weather context.
        3. Computes the VAE regime encoder input tensor.

        Returns
        -------
        LiveFeatureSnapshot
            Snapshot with ``sample["ae_input"]`` and enriched
            ``sample["tech_features"]`` containing daily context columns.
        """
        snapshot = self.refresh(now)
        anchor_ts = snapshot.anchor_ts

        # Refresh daily caches if date changed
        self._maybe_refresh_daily_caches(anchor_ts)

        # Inject daily context into tech features
        daily_context = self._get_daily_context(anchor_ts)
        if daily_context is not None:
            snapshot.sample["daily_context"] = daily_context
        self._apply_daily_context_features(snapshot, daily_context or {})

        # Build regime encoder input
        regime_input = self._compute_regime_input(anchor_ts, snapshot.bars)
        if regime_input is not None:
            snapshot.sample["ae_input"] = regime_input

        return snapshot

    # ------------------------------------------------------------------
    # Daily cache management
    # ------------------------------------------------------------------

    def _maybe_refresh_daily_caches(self, anchor_ts: pd.Timestamp) -> None:
        """Reload daily caches if the trading date has changed."""
        current_date = anchor_ts.normalize()
        if self._cache_date is not None and self._cache_date == current_date:
            return

        self._cache_date = current_date

        if self.daily_paths.eia_storage_path:
            self._eia_cache = self._load_eia(self.daily_paths.eia_storage_path)

        if self.daily_paths.weather_path:
            self._weather_cache = self._load_weather(self.daily_paths.weather_path)
            self._weather_features_cache = self._build_weather_features()

        if self.daily_paths.daily_features_path:
            self._daily_features_cache = self._load_daily_features(
                self.daily_paths.daily_features_path
            )

    def reload_daily_caches(self) -> None:
        """Force-reload all daily caches (call after cron refresh)."""
        self._cache_date = None
        if self.last_snapshot is not None:
            self._maybe_refresh_daily_caches(self.last_snapshot.anchor_ts)

    # ------------------------------------------------------------------
    # Daily feature loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_eia(path: str) -> Optional[pd.DataFrame]:
        """Load EIA natural gas storage data from cache file."""
        p = Path(path)
        if not p.exists():
            logger.warning("EIA cache not found: %s", path)
            return None
        try:
            if p.suffix in {".h5", ".hdf"}:
                df = pd.read_hdf(path)
            elif p.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, parse_dates=True, index_col=0)
            df = _normalize_dt_index(df)
            logger.info("Loaded EIA cache: %d rows from %s", len(df), path)
            return df
        except Exception as exc:
            logger.error("Failed to load EIA cache: %s", exc)
            return None

    @staticmethod
    def _load_weather(path: str) -> Optional[pd.DataFrame]:
        """Load HDD/CDD weather data from cache file."""
        p = Path(path)
        if not p.exists():
            logger.warning("Weather cache not found: %s", path)
            return None
        try:
            if p.suffix in {".h5", ".hdf"}:
                df = pd.read_hdf(path)
            elif p.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, parse_dates=True, index_col=0)
            df = _normalize_dt_index(df)
            logger.info("Loaded weather cache: %d rows from %s", len(df), path)
            return df
        except Exception as exc:
            logger.error("Failed to load weather cache: %s", exc)
            return None

    def _load_spline_transformer(self):
        """Load a pre-fitted SplineTransformer from disk (pickle)."""
        if self.daily_paths.spline_transformer_path is None:
            return None
        p = Path(self.daily_paths.spline_transformer_path)
        if not p.exists():
            logger.info("No spline transformer at %s, will fit on first load", p)
            return None
        try:
            import pickle

            with open(p, "rb") as f:
                transformer = pickle.load(f)
            logger.info("Loaded spline transformer from %s", p)
            return transformer
        except Exception as exc:
            logger.warning("Failed to load spline transformer: %s", exc)
            return None

    def save_spline_transformer(self, path: Union[str, Path]) -> None:
        """Persist the fitted SplineTransformer for consistent live inference."""
        if self._spline_transformer is None:
            logger.warning("No spline transformer to save")
            return
        import pickle

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self._spline_transformer, f)
        logger.info("Saved spline transformer to %s", p)

    def _build_weather_features(self) -> Optional[pd.DataFrame]:
        """Process raw population-weighted weather into the 13 training features."""
        if self._weather_cache is None or self._weather_cache.empty:
            return None
        if "wtd_TAVG" not in self._weather_cache.columns:
            logger.warning(
                "Weather cache missing 'wtd_TAVG' column (population-weighted "
                "temperature). Available: %s", list(self._weather_cache.columns),
            )
            return None
        try:
            features, transformer = process_weather_features(
                self._weather_cache,
                spline_transformer=self._spline_transformer,
            )
            if transformer is not None:
                self._spline_transformer = transformer
            logger.info(
                "Built weather features: %d rows x %d cols",
                len(features), len(features.columns),
            )
            return features
        except Exception as exc:
            logger.error("Failed to build weather features: %s", exc)
            return None

    @staticmethod
    def _load_daily_features(path: str) -> Optional[pd.DataFrame]:
        """Load pre-computed daily features from cache."""
        p = Path(path)
        if not p.exists():
            logger.warning("Daily features cache not found: %s", path)
            return None
        try:
            if p.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, parse_dates=True, index_col=0)
            df = _normalize_dt_index(df)
            logger.info("Loaded daily features: %d rows from %s", len(df), path)
            return df
        except Exception as exc:
            logger.error("Failed to load daily features: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------

    def _build_technical_frame(self) -> Tuple[pd.DataFrame, List[str]]:
        """Mirror the ng_hybrid notebook's ContinuousIntradayPrep settings."""
        steps = self.config.target_horizon_minutes // self.config.bar_minutes
        add_bid_ask = any(
            col in self._bars.columns
            for col in ("bidvol", "askvol", "BidVolume", "AskVolume")
        )
        df_out, _, _ = self.prep.prepare(
            self._bars,
            steps_60m=steps,
            keep_only_active=False,
            add_daily=True,
            add_overnight=True,
            add_deseas=True,
            add_time_features=True,
            add_resample_precalc=False,
            apply_scaling=False,
            add_bid_ask=add_bid_ask,
            add_event_markers=False,
        )
        df_out.columns = [str(col).lower() for col in df_out.columns]

        feat_cols = self.prep.get_feature_cols(
            steps_60m=steps,
            bar_minutes=self.config.bar_minutes,
            add_resample_precalc=False,
            add_bid_ask=False,
            add_event_markers=False,
        )
        feat_cols = [str(col).lower() for col in feat_cols if str(col).lower() in df_out.columns]
        df_out[feat_cols] = df_out[feat_cols].ffill().bfill().fillna(0.0)
        return df_out, feat_cols

    def _apply_daily_context_features(
        self,
        snapshot: LiveFeatureSnapshot,
        daily_context: Dict[str, float],
    ) -> None:
        tech_cols = list(snapshot.sample.get("tech_feature_cols", []))
        if not tech_cols:
            return

        tech_df = pd.DataFrame(snapshot.sample["tech_features"], columns=tech_cols)
        for key, value in daily_context.items():
            tech_df[key] = float(value)

        if self.feature_cols:
            missing = [col for col in self.feature_cols if col not in tech_df.columns]
            if missing and not self._missing_feature_cols_logged:
                logger.warning(
                    "Missing %d checkpoint feature columns in live assembly; zero-filling. "
                    "Examples: %s",
                    len(missing),
                    missing[:10],
                )
                self._missing_feature_cols_logged = True
            for col in missing:
                tech_df[col] = 0.0
            tech_df = tech_df.loc[:, self.feature_cols]
            snapshot.sample["tech_feature_cols"] = list(self.feature_cols)
        else:
            ordered = tech_cols + [col for col in daily_context if col not in tech_cols]
            tech_df = tech_df.loc[:, ordered]
            snapshot.sample["tech_feature_cols"] = ordered

        snapshot.sample["tech_features"] = tech_df.astype(np.float32).values

    def _get_daily_context(self, anchor_ts: pd.Timestamp) -> Optional[Dict[str, float]]:
        """Extract daily EIA/weather context for the anchor date.

        Uses forward-fill logic: if today's data hasn't been released yet,
        the most recent available value is used.

        Returns
        -------
        dict or None
            Dictionary of daily feature name -> value pairs.
        """
        context: Dict[str, float] = {}
        anchor_date = anchor_ts.normalize()

        if self._eia_cache is not None and not self._eia_cache.empty:
            eia = self._eia_cache
            if not isinstance(eia.index, pd.DatetimeIndex):
                eia.index = pd.to_datetime(eia.index)
            # Forward-fill: take most recent row <= anchor_date
            mask = _safe_date_mask(eia.index, anchor_date)
            if mask.any():
                latest = eia.loc[mask].iloc[-1]
                for col in eia.columns:
                    context[col] = float(latest[col]) if pd.notna(latest[col]) else 0.0

        # Use processed weather features (HDD/CDD/spline/wtd_tavg) if available,
        # otherwise fall back to raw weather cache
        wx_source = self._weather_features_cache
        wx_prefix = ""
        if wx_source is None or wx_source.empty:
            wx_source = self._weather_cache
            wx_prefix = "wx_"
        if wx_source is not None and not wx_source.empty:
            wx = wx_source
            if not isinstance(wx.index, pd.DatetimeIndex):
                wx.index = pd.to_datetime(wx.index)
            mask = _safe_date_mask(wx.index, anchor_date)
            if mask.any():
                latest = wx.loc[mask].iloc[-1]
                for col in wx.columns:
                    key = f"{wx_prefix}{col}" if wx_prefix else col
                    context[key] = float(latest[col]) if pd.notna(latest[col]) else 0.0

        if self._daily_features_cache is not None and not self._daily_features_cache.empty:
            df = self._daily_features_cache
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            mask = _safe_date_mask(df.index, anchor_date)
            if mask.any():
                latest = df.loc[mask].iloc[-1]
                for col in df.columns:
                    context[col] = float(latest[col]) if pd.notna(latest[col]) else 0.0

        return context if context else None

    def _compute_regime_input(
        self,
        anchor_ts: pd.Timestamp,
        bars: pd.DataFrame,
    ) -> Optional[np.ndarray]:
        """Build the VAE regime encoder input: ``(ae_window, 12)``.

        Computes the 12 regime features from available bar data and
        daily context caches. Returns ``None`` if insufficient data.
        """
        if bars is None or bars.empty:
            logger.warning("No bar data available for regime computation")
            return None

        close = bars["Close"].dropna()
        if len(close) < self.ae_window + 21:
            logger.warning(
                "Insufficient bar data for regime features: %d bars, need %d",
                len(close),
                self.ae_window + 21,
            )
            return None

        # Resample to daily close prices
        daily_close = close.resample("1D").last().dropna()
        if len(daily_close) < self.ae_window + 21:
            logger.warning("Insufficient daily closes for regime: %d", len(daily_close))
            return None

        # Compute return and volatility features
        log_ret = np.log(daily_close / daily_close.shift(1)).dropna()

        ret_1d = log_ret
        ret_5d = log_ret.rolling(5).sum()
        ret_21d = log_ret.rolling(21).sum()
        rv_5d = log_ret.rolling(5).std() * np.sqrt(252)
        rv_21d = log_ret.rolling(21).std() * np.sqrt(252)

        # Storage-state features from EIA cache
        storage_features = self._get_storage_regime_features(anchor_ts)

        # Seasonal features
        anchor_month = anchor_ts.month
        is_injection = 1.0 if 4 <= anchor_month <= 10 else 0.0

        # Build feature matrix for the last ae_window days
        n = len(daily_close)
        regime_rows = []
        for i in range(max(0, n - self.ae_window), n):
            idx = daily_close.index[i]
            row = np.zeros(12, dtype=np.float32)
            row[0] = ret_1d.get(idx, 0.0)
            row[1] = ret_5d.get(idx, 0.0)
            row[2] = ret_21d.get(idx, 0.0)
            row[3] = rv_5d.get(idx, 0.0)
            row[4] = rv_21d.get(idx, 0.0)

            # Storage features (from daily cache)
            if storage_features is not None:
                row[5] = storage_features.get("pct_in_5y_band", 0.0)
                row[6] = storage_features.get("dev_from_5y_mean_zscore", 0.0)
                row[7] = storage_features.get("band_width_pct", 0.0)
                row[8] = storage_features.get("forecast_vs_seasonal_zscore", 0.0)
                row[9] = storage_features.get("change_vs_seasonal_zscore", 0.0)

            row[10] = is_injection
            # dev_x_season: interaction of deviation and season
            row[11] = row[6] * (1.0 if is_injection else -1.0)

            regime_rows.append(row)

        regime_arr = np.array(regime_rows, dtype=np.float32)

        # Pad if we have fewer than ae_window rows
        if len(regime_arr) < self.ae_window:
            pad = np.zeros(
                (self.ae_window - len(regime_arr), 12), dtype=np.float32
            )
            regime_arr = np.concatenate([pad, regime_arr], axis=0)

        # Replace any NaN/inf with 0
        regime_arr = np.nan_to_num(regime_arr, nan=0.0, posinf=0.0, neginf=0.0)

        return regime_arr

    def _get_storage_regime_features(
        self, anchor_ts: pd.Timestamp
    ) -> Optional[Dict[str, float]]:
        """Extract EIA storage-based regime features for the anchor date."""
        if self._eia_cache is None or self._eia_cache.empty:
            return None

        eia = self._eia_cache
        if not isinstance(eia.index, pd.DatetimeIndex):
            eia.index = pd.to_datetime(eia.index)

        anchor_date = anchor_ts.normalize()
        mask = _safe_date_mask(eia.index, anchor_date)
        if not mask.any():
            return None

        features: Dict[str, float] = {}
        for col in [
            "pct_in_5y_band",
            "dev_from_5y_mean_zscore",
            "band_width_pct",
            "forecast_vs_seasonal_zscore",
            "change_vs_seasonal_zscore",
        ]:
            if col in eia.columns:
                val = eia.loc[mask, col].iloc[-1]
                features[col] = float(val) if pd.notna(val) else 0.0

        return features if features else None
