from __future__ import annotations

from typing import Dict, Iterable, List

from live_cta.pipelines.base import PipelineContract, alias_names
from live_cta.pipelines.hybrid import HYBRID_CONTRACT, HybridMixturePipeline
from live_cta.pipelines.orderflow import (
    MMTFTV3_CONTRACT,
    RECURRENT_WSPR_CONTRACT,
    MMTFTv3LiveInterface,
    MMTFTv3Pipeline,
    RecurrentWSPRLiveInterface,
    RecurrentWSPRPipeline,
)


PIPELINE_CONTRACTS: Dict[str, PipelineContract] = {
    HYBRID_CONTRACT.key: HYBRID_CONTRACT,
    RECURRENT_WSPR_CONTRACT.key: RECURRENT_WSPR_CONTRACT,
    MMTFTV3_CONTRACT.key: MMTFTV3_CONTRACT,
}

PIPELINE_RUNTIME_CLASSES = {
    HYBRID_CONTRACT.key: HybridMixturePipeline,
    RECURRENT_WSPR_CONTRACT.key: RecurrentWSPRPipeline,
    MMTFTV3_CONTRACT.key: MMTFTv3Pipeline,
}

PIPELINE_INTERFACE_CLASSES = {
    RECURRENT_WSPR_CONTRACT.key: RecurrentWSPRLiveInterface,
    MMTFTV3_CONTRACT.key: MMTFTv3LiveInterface,
}


def _build_alias_index(contracts: Iterable[PipelineContract]) -> Dict[str, str]:
    alias_index: Dict[str, str] = {}
    for contract in contracts:
        for name in alias_names(contract):
            alias_index[name] = contract.key
    return alias_index


PIPELINE_NAME_INDEX = _build_alias_index(PIPELINE_CONTRACTS.values())


def get_pipeline_contract(model_type: str) -> PipelineContract:
    try:
        canonical = PIPELINE_NAME_INDEX[model_type]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model type {model_type!r}. Available names: {sorted(PIPELINE_NAME_INDEX)}"
        ) from exc
    return PIPELINE_CONTRACTS[canonical]


def get_pipeline_runtime(model_type: str):
    contract = get_pipeline_contract(model_type)
    return PIPELINE_RUNTIME_CLASSES[contract.key]


def get_pipeline_interface(model_type: str):
    contract = get_pipeline_contract(model_type)
    return PIPELINE_INTERFACE_CLASSES.get(contract.key)


def model_requires_tick_data(model_type: str) -> bool:
    return get_pipeline_contract(model_type).requires_tick_data


def list_pipeline_names(*, include_aliases: bool = False) -> List[str]:
    if include_aliases:
        return sorted(PIPELINE_NAME_INDEX)
    return sorted(PIPELINE_CONTRACTS)
