from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from CTAFlow.models.prep.intraday_continuous import ContinuousIntradayPrep, SessionSpec
from live_cta.core.ng_live import _normalize_dt_index, process_weather_features
from live_cta.pipelines.base import PipelineContract

logger = logging.getLogger(__name__)

TZ = "America/Chicago"


def _parse_session(session_meta: Any) -> SessionSpec:
    if isinstance(session_meta, SessionSpec):
        return session_meta
    if isinstance(session_meta, dict):
        return SessionSpec(
            session_meta.get("name", "USA"),
            session_meta.get("start", "02:30"),
            session_meta.get("end", "15:00"),
        )
    if isinstance(session_meta, str) and "_" in session_meta and "-" in session_meta:
        name, times = session_meta.split("_", 1)
        start, end = times.split("-", 1)
        return SessionSpec(name, start, end)
    return SessionSpec("USA", "02:30", "15:00")


def _ensure_intraday_cst_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize(TZ)
    else:
        df.index = df.index.tz_convert(TZ)
    return df


class HybridMixturePipeline:
    """Notebook-matched feature builder for HybridMixtureNetwork inference."""

    required_model_inputs = ("x_seq", "ae_input")

    @classmethod
    def build_inputs(
        cls,
        bars_df: pd.DataFrame,
        eia_df: Optional[pd.DataFrame],
        weather_df: Optional[pd.DataFrame],
        checkpoint: Dict[str, Any],
        spline_transformer=None,
    ) -> Dict[str, Any]:
        feature_cols = checkpoint.get("feature_cols", [])
        feature_groups = checkpoint.get("feature_groups", {})
        regime_cols = checkpoint.get("regime_cols", [])
        bar_minutes = checkpoint.get("bar_minutes", 15)
        ae_window_bars = checkpoint.get("ae_window_bars", 510)
        seq_len = checkpoint["config"].get("seq_len", 20)

        session_spec = _parse_session(checkpoint.get("session"))
        target_horizon_bars = checkpoint.get("target_horizon_bars", 10)

        bars_df = _ensure_intraday_cst_index(bars_df.copy())
        prep = ContinuousIntradayPrep(
            sessions=[session_spec],
            bar_minutes=bar_minutes,
        )

        df_prep, _, _ = prep.prepare(
            bars_df.copy(),
            steps_60m=target_horizon_bars,
            keep_only_active=False,
            add_daily=True,
            add_overnight=True,
            add_deseas=True,
            add_time_features=True,
            add_resample_precalc=False,
            apply_scaling=False,
            add_bid_ask="bidvol" in bars_df.columns or "askvol" in bars_df.columns,
            add_event_markers=False,
        )
        df_prep = _ensure_intraday_cst_index(df_prep)
        df_prep.columns = [c.lower() for c in df_prep.columns]

        tech_feature_cols = feature_groups.get("technical", [])
        storage_feature_cols = feature_groups.get("storage", [])
        weather_feature_cols = feature_groups.get("weather", [])

        bar_dates = df_prep.index.normalize()
        bar_dates_naive = bar_dates.tz_localize(None) if bar_dates.tz else bar_dates
        if eia_df is not None and not eia_df.empty:
            sw = _normalize_dt_index(eia_df.copy())
            if "storage_level" in sw.columns:
                sl = sw["storage_level"]
                sw["sl_4wk_mean"] = sl.rolling(4, min_periods=2).mean()
                sw["sl_4wk_max"] = sl.rolling(4, min_periods=2).max()
                sw["sl_4wk_min"] = sl.rolling(4, min_periods=2).min()
                if "storage_change" in sw.columns:
                    sw["sl_change_4wk_mean"] = sw["storage_change"].rolling(4, min_periods=2).mean()

            s_cols = [c for c in storage_feature_cols if c in sw.columns]
            if s_cols:
                daily_idx = pd.date_range(sw.index[0], bar_dates_naive[-1], freq="D")
                storage_daily = sw[s_cols].reindex(
                    sw.index.union(daily_idx)
                ).sort_index().ffill()
                for col in s_cols:
                    df_prep[col] = storage_daily[col].reindex(bar_dates_naive).values
            for col in storage_feature_cols:
                if col not in df_prep.columns:
                    df_prep[col] = 0.0
        else:
            for col in storage_feature_cols:
                df_prep[col] = 0.0
            logger.warning("No EIA data - zero-filling storage features")

        if weather_df is not None and not weather_df.empty:
            wx = _normalize_dt_index(weather_df.copy())
            wx_features, _ = process_weather_features(wx, spline_transformer=spline_transformer)
            wx_features = _normalize_dt_index(wx_features)
            for col in wx_features.columns:
                df_prep[col] = wx_features[col].reindex(bar_dates_naive).values
            for col in weather_feature_cols:
                if col not in df_prep.columns:
                    df_prep[col] = 0.0
        else:
            for col in weather_feature_cols:
                df_prep[col] = 0.0
            logger.warning("No weather data - zero-filling weather features")

        daily_close = df_prep.groupby(df_prep.index.date)["close"].last()
        daily_close.index = pd.DatetimeIndex(daily_close.index)
        daily_log_ret = np.log(daily_close / daily_close.shift(1))

        regime_daily = pd.DataFrame(index=daily_close.index)
        regime_daily["regime_ret_1d"] = daily_log_ret
        regime_daily["regime_ret_5d"] = daily_log_ret.rolling(5).sum()
        regime_daily["regime_ret_21d"] = daily_log_ret.rolling(21).sum()
        regime_daily["regime_rv_5d"] = np.sqrt((daily_log_ret ** 2).rolling(5).mean()) * np.sqrt(252)
        regime_daily["regime_rv_21d"] = np.sqrt((daily_log_ret ** 2).rolling(21).mean()) * np.sqrt(252)

        if eia_df is not None and not eia_df.empty and "storage_level" in eia_df.columns:
            sw = _normalize_dt_index(eia_df.copy())
            sl = sw["storage_level"]
            wk_idx = sl.index.isocalendar().week.values

            hi_5y = pd.Series(np.nan, index=sl.index)
            lo_5y = pd.Series(np.nan, index=sl.index)
            mean_5y = pd.Series(np.nan, index=sl.index)

            for w in range(1, 54):
                mask = wk_idx == w
                if mask.sum() < 2:
                    continue
                idx_pos = np.where(mask)[0]
                vals = sl.iloc[idx_pos]
                hi_5y.iloc[idx_pos] = vals.expanding().max().shift(1).values
                lo_5y.iloc[idx_pos] = vals.expanding().min().shift(1).values
                mean_5y.iloc[idx_pos] = vals.expanding().mean().shift(1).values

            hi_5y = hi_5y.ffill().bfill()
            lo_5y = lo_5y.ffill().bfill()
            mean_5y = mean_5y.ffill().bfill()
            band_w = (hi_5y - lo_5y).clip(lower=1)

            regime_wkly = pd.DataFrame(
                {
                    "regime_pct_in_5y_band": ((sl - lo_5y) / band_w).values,
                    "regime_dev_5y_zscore": ((sl - mean_5y) / band_w).values,
                    "regime_band_width_pct": (band_w / mean_5y.clip(lower=1)).values,
                },
                index=sl.index,
            )
            regime_wkly_daily = regime_wkly.reindex(
                regime_wkly.index.union(daily_close.index)
            ).sort_index().ffill().reindex(daily_close.index)
            for col in regime_wkly_daily.columns:
                regime_daily[col] = regime_wkly_daily[col].values

            sc = sw["storage_change"] if "storage_change" in sw.columns else pd.Series(0, index=sw.index)
            sea_chg = pd.Series(np.nan, index=sc.index)
            for w in range(1, 54):
                mask = wk_idx == w
                if mask.sum() < 2:
                    continue
                idx_pos = np.where(mask)[0]
                sea_chg.iloc[idx_pos] = sc.iloc[idx_pos].expanding().mean().shift(1).values
            sea_chg = sea_chg.ffill().bfill()
            sea_std = (sc - sea_chg).expanding().std().clip(lower=1)
            chg_vs_sea = (sc - sea_chg) / sea_std
            fc = sw["consensus_est"] if "consensus_est" in sw.columns else sea_chg
            fc_vs_sea = (fc - sea_chg) / sea_std

            fc_regime = pd.DataFrame(
                {
                    "regime_fc_vs_seasonal_z": fc_vs_sea.values,
                    "regime_chg_vs_seasonal_z": chg_vs_sea.values,
                },
                index=sc.index,
            )
            fc_daily = fc_regime.reindex(
                fc_regime.index.union(daily_close.index)
            ).sort_index().ffill().reindex(daily_close.index)
            for col in fc_daily.columns:
                regime_daily[col] = fc_daily[col].values
        else:
            for col in (
                "regime_pct_in_5y_band",
                "regime_dev_5y_zscore",
                "regime_band_width_pct",
                "regime_fc_vs_seasonal_z",
                "regime_chg_vs_seasonal_z",
            ):
                regime_daily[col] = 0.0

        month = regime_daily.index.month
        regime_daily["regime_is_injection"] = ((month >= 4) & (month <= 10)).astype(np.float32)
        season_sign = np.where(regime_daily["regime_is_injection"].values > 0.5, 1.0, -1.0)
        regime_daily["regime_dev_x_season"] = regime_daily.get(
            "regime_dev_5y_zscore", pd.Series(0, index=regime_daily.index)
        ) * season_sign

        for col in regime_cols:
            if col in regime_daily.columns:
                df_prep[col] = regime_daily[col].reindex(bar_dates_naive).values
            else:
                df_prep[col] = 0.0

        all_cols = feature_cols + regime_cols
        for col in all_cols:
            if col not in df_prep.columns:
                df_prep[col] = 0.0
        df_prep[all_cols] = df_prep[all_cols].ffill().bfill().fillna(0.0)

        x_seq = df_prep[feature_cols].values.astype(np.float32)
        ae_input = df_prep[regime_cols].values.astype(np.float32)

        t = len(df_prep) - 1
        x_start = max(0, t - seq_len + 1)
        ae_start = max(0, t - ae_window_bars + 1)
        x_seq = x_seq[x_start: t + 1]
        ae_input = ae_input[ae_start: t + 1]

        if x_seq.shape[0] < seq_len:
            pad = np.zeros((seq_len - x_seq.shape[0], x_seq.shape[1]), dtype=np.float32)
            x_seq = np.concatenate([pad, x_seq], axis=0)
        if ae_input.shape[0] < ae_window_bars:
            pad = np.zeros((ae_window_bars - ae_input.shape[0], ae_input.shape[1]), dtype=np.float32)
            ae_input = np.concatenate([pad, ae_input], axis=0)

        x_seq = np.nan_to_num(x_seq, nan=0.0, posinf=0.0, neginf=0.0)
        ae_input = np.nan_to_num(ae_input, nan=0.0, posinf=0.0, neginf=0.0)

        anchor_ts = df_prep.index[-1]
        anchor_close = float(df_prep["close"].iloc[-1])

        logger.info(
            "Built hybrid inputs: x_seq=%s, ae_input=%s, anchor=%s, close=%.3f",
            x_seq.shape, ae_input.shape, anchor_ts, anchor_close,
        )

        return {
            "x_seq": torch.from_numpy(x_seq).unsqueeze(0),
            "ae_input": torch.from_numpy(ae_input).unsqueeze(0),
            "anchor_ts": anchor_ts,
            "anchor_close": anchor_close,
            "n_bars": len(df_prep),
        }


HYBRID_CONTRACT = PipelineContract(
    key="hybrid_mixture_network",
    display_name="HybridMixtureNetwork",
    family="ng_intraday_hybrid",
    raw_input_mode="bars",
    requires_tick_data=False,
    canonical_entrypoint="scripts/inference_server.py",
    pipeline_class_name="HybridMixturePipeline",
    context_artifacts=(
        "intraday OHLCV bars",
        "EIA storage cache",
        "weather cache",
        "optional spline transformer",
    ),
    preprocessing_steps=(
        "normalize intraday bars to America/Chicago",
        "run ContinuousIntradayPrep with notebook-matched options",
        "forward-fill storage features onto intraday bars",
        "forward-fill weather features onto intraday bars",
        "build 12 regime features from daily aggregation",
        "window and pad x_seq and ae_input tensors",
    ),
    required_checkpoint_keys=(
        "config.seq_len",
        "feature_cols",
        "feature_groups",
        "regime_cols",
        "bar_minutes",
        "target_horizon_bars",
        "session",
        "ae_window_bars",
    ),
    required_model_inputs=HybridMixturePipeline.required_model_inputs,
    output_shapes={
        "x_seq": "(B, seq_len, n_features)",
        "ae_input": "(B, ae_window_bars, n_regime_features)",
    },
    aliases=("ng_hybrid", "hybrid"),
)
