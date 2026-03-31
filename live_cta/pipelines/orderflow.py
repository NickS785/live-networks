from __future__ import annotations

from typing import Optional

from live_cta.core.live import LiveFeatureSnapshot, LiveV3FeatureInterface, TimestampLike
from live_cta.pipelines.base import PipelineContract, SnapshotInputAdapter


class OrderflowLiveInterface(LiveV3FeatureInterface):
    """Named live interface for orderflow-driven model families."""

    def get_inference_snapshot(
        self,
        now: Optional[TimestampLike] = None,
    ) -> LiveFeatureSnapshot:
        return self.refresh(now)


class RecurrentWSPRLiveInterface(OrderflowLiveInterface):
    """Live interface for RecurrentWSPR/MMTFTv2-style inputs."""


class MMTFTv3LiveInterface(OrderflowLiveInterface):
    """Live interface for MMTFTv3-style inputs."""


class RecurrentWSPRPipeline(SnapshotInputAdapter):
    required_model_inputs = (
        "tech_features",
        "tech_lens",
        "seq_vpin",
        "seq_vpin_lens",
        "fused_spatial",
        "ticker_id",
        "asset_class_id",
        "asset_subclass_id",
    )


class MMTFTv3Pipeline(SnapshotInputAdapter):
    required_model_inputs = (
        "tech_features",
        "tech_lens",
        "seq_vpin",
        "seq_vpin_lens",
        "fused_spatial",
        "ticker_id",
        "asset_class_id",
        "asset_subclass_id",
    )


RECURRENT_WSPR_CONTRACT = PipelineContract(
    key="recurrent_wspr",
    display_name="RecurrentWSPR / MMTFTv2",
    family="orderflow_snapshot",
    raw_input_mode="ticks",
    requires_tick_data=True,
    canonical_entrypoint="live_cta.core.live.LiveV3FeatureInterface",
    pipeline_class_name="RecurrentWSPRPipeline",
    context_artifacts=(
        "tick data",
        "session configuration",
        "volume-profile settings",
        "VPIN settings",
        "raster settings",
    ),
    preprocessing_steps=(
        "fetch live ticks for the configured lookback window",
        "resample technical bars from ticks",
        "build sequential VPIN features",
        "build market-profile and number-bars context",
        "build fused spatial tensors",
        "assemble backward-looking snapshot tensors",
    ),
    required_checkpoint_keys=(
        "sessions",
        "bar_minutes",
        "target_horizon_minutes",
        "tech_lookback",
        "seq_lookback_bars",
        "numbars_lookback",
        "vpin_bucket_volume",
        "vpin_window",
        "profile_start_time",
        "profile_end_time",
        "raster_num_bars",
        "raster_bins",
    ),
    required_model_inputs=RecurrentWSPRPipeline.required_model_inputs,
    output_shapes={
        "tech_features": "(B, tech_lookback, F_tech)",
        "tech_lens": "(B,)",
        "seq_vpin": "(B, seq_lookback_bars, F_seq)",
        "seq_vpin_lens": "(B,)",
        "fused_spatial": "(B, numbars_lookback, 7, raster_bins_or_price_levels)",
        "ticker_id": "(B,)",
        "asset_class_id": "(B,)",
        "asset_subclass_id": "(B,)",
    },
    aliases=("wspr", "mmtftv2"),
)


MMTFTV3_CONTRACT = PipelineContract(
    key="mmtft_v3",
    display_name="MMTFTv3",
    family="orderflow_snapshot",
    raw_input_mode="ticks",
    requires_tick_data=True,
    canonical_entrypoint="live_cta.core.live.LiveV3FeatureInterface",
    pipeline_class_name="MMTFTv3Pipeline",
    context_artifacts=(
        "tick data",
        "session configuration",
        "volume-profile settings",
        "VPIN settings",
        "raster settings",
    ),
    preprocessing_steps=(
        "fetch live ticks for the configured lookback window",
        "resample technical bars from ticks",
        "build sequential VPIN features",
        "build market-profile and number-bars context",
        "build fused spatial tensors",
        "assemble backward-looking snapshot tensors",
    ),
    required_checkpoint_keys=(
        "sessions",
        "bar_minutes",
        "target_horizon_minutes",
        "tech_lookback",
        "seq_lookback_bars",
        "numbars_lookback",
        "vpin_bucket_volume",
        "vpin_window",
        "profile_start_time",
        "profile_end_time",
        "raster_num_bars",
        "raster_bins",
    ),
    required_model_inputs=MMTFTv3Pipeline.required_model_inputs,
    output_shapes={
        "tech_features": "(B, tech_lookback, F_tech)",
        "tech_lens": "(B,)",
        "seq_vpin": "(B, seq_lookback_bars, F_seq)",
        "seq_vpin_lens": "(B,)",
        "fused_spatial": "(B, numbars_lookback, 7, raster_bins_or_price_levels)",
        "ticker_id": "(B,)",
        "asset_class_id": "(B,)",
        "asset_subclass_id": "(B,)",
    },
    aliases=("mmtft", "mmtfv3"),
)
