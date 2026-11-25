"""vLLM 0.18-compatible Ascend HCCL P2P KV connector."""

from __future__ import annotations

from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import P2pNcclConnector
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorRole
from vllm.distributed.parallel_state import get_world_group
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata

from dataset_pipeline.serving_dataset.pd_p2p_request_id import transfer_request_id

from .p2p_hccl_engine import P2pHcclEngine


class P2pHcclConnector(P2pNcclConnector):
    """Reuse vLLM's scheduler metadata and perform worker transfer on HCCL."""

    def __init__(self, vllm_config: Any, role: KVConnectorRole, kv_cache_config: Any = None):
        KVConnectorBase_V1.__init__(self, vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Any] = {}
        self.is_producer = self._kv_transfer_config.is_kv_producer
        self.chunked_prefill: dict[str, tuple[list[int], list[int] | None]] = {}
        self._rank = get_world_group().rank if role == KVConnectorRole.WORKER else 0
        self._local_rank = get_world_group().local_rank if role == KVConnectorRole.WORKER else 0
        self.p2p_hccl_engine = P2pHcclEngine(self._local_rank, self._kv_transfer_config, port_offset=self._rank) if role == KVConnectorRole.WORKER else None

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        if self.is_producer or forward_context.attn_metadata is None:
            return
        if self.p2p_hccl_engine is None:
            raise RuntimeError("P2pHcclConnector consumer has no HCCL engine")
        metadata = self._get_connector_metadata()
        if metadata is None:
            return
        for request in metadata.requests:
            ip, port = self.parse_request_id(request.request_id, False)
            remote_address = f"{ip}:{port + self._rank}"
            for layer_name, module in forward_context.no_compile_layers.items():
                if not hasattr(module, "kv_cache"):
                    continue
                components = _kv_tensor_components(
                    module.kv_cache[forward_context.virtual_engine], "load")
                for component_index, layer in components:
                    tensor = self.p2p_hccl_engine.recv_tensor(
                        _tensor_id(
                            request.request_id, layer_name, component_index),
                        remote_address,
                    )
                    _inject_kv_component(
                        layer, tensor, request.block_ids,
                        forward_context.attn_metadata,
                        component_index is not None,
                    )

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any) -> None:
        if not self.is_producer:
            return
        if self.p2p_hccl_engine is None:
            raise RuntimeError("P2pHcclConnector producer has no HCCL engine")
        metadata = self._get_connector_metadata()
        for request in metadata.requests:
            ip, port = self.parse_request_id(request.request_id, True)
            remote_address = f"{ip}:{port + self._rank}"
            for component_index, layer in _kv_tensor_components(kv_layer, "save"):
                self.p2p_hccl_engine.send_tensor(
                    _tensor_id(request.request_id, layer_name, component_index),
                    _extract_kv_component(
                        layer, request.block_ids, attn_metadata,
                        component_index is not None,
                    ),
                    remote_address,
                )

    def wait_for_save(self) -> None:
        if self.is_producer:
            if self.p2p_hccl_engine is None:
                raise RuntimeError("P2pHcclConnector producer has no HCCL engine")
            self.p2p_hccl_engine.wait_for_sent()

    def get_finished(self, finished_req_ids: set[str], **kwargs: Any) -> tuple[None, None]:
        if self.p2p_hccl_engine is None:
            raise RuntimeError("P2pHcclConnector has no HCCL engine")
        return self.p2p_hccl_engine.get_finished(
            {transfer_request_id(request_id) for request_id in finished_req_ids},
            self._vllm_config.compilation_config.static_forward_context,
        )


def _kv_tensor_components(value: Any, operation: str) -> tuple[tuple[int | None, torch.Tensor], ...]:
    if isinstance(value, torch.Tensor):
        return ((None, value),)
    if not isinstance(value, (tuple, list)) or not value:
        raise TypeError(
            f"P2pHcclConnector {operation} KV cache must be a tensor or "
            "a non-empty tensor sequence"
        )
    components = tuple(enumerate(value))
    if any(not isinstance(component, torch.Tensor) for _, component in components):
        raise TypeError("P2pHcclConnector KV cache sequence contains a non-tensor")
    return components


def _tensor_id(request_id: str, layer_name: str, component_index: int | None) -> str:
    value = f"{transfer_request_id(request_id)}#{layer_name}"
    return value if component_index is None else f"{value}#{component_index}"


def _extract_kv_component(
    layer: torch.Tensor, block_ids: Any, attn_metadata: Any,
    is_component_sequence: bool,
) -> torch.Tensor:
    if is_component_sequence:
        return layer[block_ids, ...]
    if isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2:
        return layer[block_ids, ...]
    if layer.shape[0] == 2:
        return layer[:, block_ids, ...]
    raise RuntimeError("P2pHcclConnector cannot send an unsupported KV cache layout")


def _inject_kv_component(
    layer: torch.Tensor, tensor: torch.Tensor, block_ids: Any, attn_metadata: Any,
    is_component_sequence: bool,
) -> None:
    if is_component_sequence:
        if len(block_ids) != tensor.shape[0]:
            raise RuntimeError("received HCCL KV block count differs from consumer allocation")
        layer[block_ids, ...] = tensor
        return
    if isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2:
        if len(block_ids) != tensor.shape[0]:
            raise RuntimeError("received HCCL KV block count differs from consumer allocation")
        layer[block_ids, ...] = tensor
        return
    if layer.shape[0] == 2:
        if len(block_ids) != tensor.shape[1]:
            raise RuntimeError("received HCCL KV block count differs from consumer allocation")
        layer[:, block_ids, ...] = tensor
        return
    raise RuntimeError("P2pHcclConnector received an unsupported KV cache layout")
