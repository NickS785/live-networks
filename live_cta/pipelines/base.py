from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

from live_cta.core.live import LiveFeatureSnapshot


@dataclass(frozen=True)
class PipelineContract:
    """Structured description of a live inference pipeline contract."""

    key: str
    display_name: str
    family: str
    raw_input_mode: str
    requires_tick_data: bool
    canonical_entrypoint: str
    pipeline_class_name: str
    context_artifacts: Tuple[str, ...]
    preprocessing_steps: Tuple[str, ...]
    required_checkpoint_keys: Tuple[str, ...]
    required_model_inputs: Tuple[str, ...]
    output_shapes: Mapping[str, str]
    aliases: Tuple[str, ...] = ()


class SnapshotInputAdapter:
    """Selects the model input tensors required from a live feature snapshot."""

    required_model_inputs: Tuple[str, ...] = ()

    @classmethod
    def prepare_model_inputs(
        cls,
        snapshot: LiveFeatureSnapshot,
        *,
        add_batch_dim: bool = True,
    ) -> Dict[str, torch.Tensor]:
        inputs = snapshot.to_model_inputs(add_batch_dim=add_batch_dim)
        missing = [key for key in cls.required_model_inputs if key not in inputs]
        if missing:
            raise KeyError(
                f"{cls.__name__} requires snapshot tensors {missing}, "
                f"but only {sorted(inputs.keys())} were available."
            )
        return {key: inputs[key] for key in cls.required_model_inputs}

    @classmethod
    def describe_runtime_shapes(
        cls,
        snapshot: LiveFeatureSnapshot,
        *,
        add_batch_dim: bool = True,
    ) -> Dict[str, Tuple[int, ...]]:
        inputs = cls.prepare_model_inputs(snapshot, add_batch_dim=add_batch_dim)
        return {key: tuple(value.shape) for key, value in inputs.items()}


def alias_names(contract: PipelineContract) -> Sequence[str]:
    return (contract.key,) + contract.aliases
