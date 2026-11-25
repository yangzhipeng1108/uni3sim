"""Explicit Mooncake producer-response adapter for real 1P1D runs.

The deployed vLLM 0.18.0 and 0.24.0 ``MooncakeConnector`` implementations consume
``do_remote_decode`` on the producer and uses the actual engine/bootstrap
objects to serve the transfer, but its upstream ``request_finished`` method
returns no HTTP handoff payload.  The PD router still needs that payload to
send the following Decode request.  This opt-in thin runtime adapter returns
only the route facts from the live producer connector; it neither transfers
KV itself nor invents a remote endpoint or token count.
"""

from __future__ import annotations

import os
import threading
from typing import Mapping


_ADAPTER_ENV = "MIDLAYER_PD_ROUTE_ADAPTER"
_CONNECTOR_ENV = "MIDLAYER_PD_HANDOFF_CONNECTOR"
_ROLE_ENV = "MIDLAYER_PD_INSTANCE_ROLE"
_ASCEND_HCCN_ENV = "MIDLAYER_MOONCAKE_HCCN_CONF"
_ADAPTER_IDS = (
    "mooncake_vllm_0_18",
    "mooncake_vllm_0_24",
)
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
_ASCEND_ENGINE_INSTALLED = False


def install_from_environment() -> bool:
    """Install the exact Mooncake adapter when a PD producer requests it."""

    global _ASCEND_ENGINE_INSTALLED, _INSTALLED
    if _CONNECTOR_ENV in os.environ:
        connector = os.environ[_CONNECTOR_ENV]
    else:
        connector = None
    ascend_runtime = _ASCEND_HCCN_ENV in os.environ
    if ascend_runtime:
        if connector != "MooncakeConnector":
            raise ValueError("Ascend Mooncake runtime requires MooncakeConnector")
        with _INSTALL_LOCK:
            if not _ASCEND_ENGINE_INSTALLED:
                _install_ascend_engine_identity_adapter()
                _ASCEND_ENGINE_INSTALLED = True
    if _ADAPTER_ENV not in os.environ:
        return ascend_runtime
    adapter = os.environ[_ADAPTER_ENV]
    if adapter not in _ADAPTER_IDS:
        raise ValueError("PD route adapter identity is unsupported")
    if connector != "MooncakeConnector":
        raise ValueError("Mooncake route adapter requires MooncakeConnector")
    if _ROLE_ENV not in os.environ:
        raise ValueError("Mooncake route adapter requires an instance role")
    role = os.environ[_ROLE_ENV]
    if role != "prefill":
        raise ValueError("Mooncake route adapter may only run on Prefill")
    with _INSTALL_LOCK:
        if _INSTALLED:
            return True
        _install_mooncake_producer_response_adapter()
        _INSTALLED = True
    return True


def _install_ascend_engine_identity_adapter() -> None:
    """Give vLLM's generic Mooncake worker the required physical NPU suffix."""

    try:
        import torch
        import torch_npu  # noqa: F401
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
            mooncake_connector,
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Ascend Mooncake identity adapter requires vLLM's MooncakeConnector"
        ) from error

    original_engine = mooncake_connector.TransferEngine

    def physical_npu_id() -> int:
        if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
            raise RuntimeError("Ascend Mooncake requires ASCEND_RT_VISIBLE_DEVICES")
        values = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
        if not values or any(not value.isdecimal() for value in values):
            raise ValueError("ASCEND_RT_VISIBLE_DEVICES must contain physical NPU IDs")
        logical_id = torch.npu.current_device()
        if logical_id < 0 or logical_id >= len(values):
            raise RuntimeError("current Ascend logical NPU has no physical mapping")
        return int(values[logical_id])

    class AscendTransferEngine:
        def __init__(self) -> None:
            self._engine = original_engine()

        def initialize(
            self,
            local_server_name: str,
            metadata_server: str,
            protocol: str,
            device_name: str,
        ) -> int:
            if not isinstance(local_server_name, str) or not local_server_name:
                raise TypeError("Mooncake local server name must be non-empty text")
            return self._engine.initialize(
                f"{local_server_name}:0:npu_{physical_npu_id()}",
                metadata_server,
                protocol,
                device_name,
            )

        def batch_register_memory(
            self, buffer_addresses: list[int], capacities: list[int]
        ) -> int:
            if len(buffer_addresses) != len(capacities):
                raise ValueError("Mooncake KV buffer address and capacity counts differ")
            for index, (address, capacity) in enumerate(
                zip(buffer_addresses, capacities)
            ):
                if isinstance(address, bool) or not isinstance(address, int) or address <= 0:
                    raise ValueError("Mooncake KV buffer address is invalid")
                if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
                    raise ValueError("Mooncake KV buffer capacity is invalid")
                print(
                    f"MIDLAYER_MOONCAKE_REGISTER_BEGIN index={index} capacity={capacity}",
                    flush=True,
                )
                status = self._engine.register_memory(address, capacity)
                print(
                    f"MIDLAYER_MOONCAKE_REGISTER_END index={index} status={status}",
                    flush=True,
                )
                if status != 0:
                    return status
            return 0

        def __getattr__(self, attribute: str) -> object:
            return getattr(self._engine, attribute)

    mooncake_connector.TransferEngine = AscendTransferEngine

    def register_block_major_kv_components(worker: object, kv_caches: object) -> object:
        if not isinstance(kv_caches, dict):
            raise TypeError("Mooncake KV caches must be a dictionary")
        kv_data_ptrs: list[int] = []
        kv_data_lens: list[int] = []
        seen_base_addresses: list[int] = []
        tensor_size_bytes: int | None = None
        split_k_and_v = worker.kv_topo.split_k_and_v
        for layer_name, value in kv_caches.items():
            if not isinstance(layer_name, str) or not layer_name:
                raise TypeError("Mooncake KV cache layer name must be non-empty text")
            if split_k_and_v:
                if not isinstance(value, (tuple, list)) or not value:
                    raise TypeError("Mooncake split-KV cache must be a non-empty tensor sequence")
                components = value
            else:
                if not isinstance(value, torch.Tensor):
                    raise TypeError("Mooncake fused KV cache must be a tensor")
                components = (value,)
            for component in components:
                if not isinstance(component, torch.Tensor):
                    raise TypeError("Mooncake Ascend KV cache component must be a tensor")
                base_address = component.data_ptr()
                if base_address in seen_base_addresses:
                    continue
                seen_base_addresses.append(base_address)
                component_size_bytes = component.nbytes
                if tensor_size_bytes is None:
                    tensor_size_bytes = component_size_bytes
                    worker.num_blocks = component.shape[0]
                if tensor_size_bytes != component_size_bytes:
                    raise RuntimeError("Mooncake KV cache components have inconsistent sizes")
                kernel_block_size = component.shape[-2 if worker.use_mla else -3]
                if worker.block_size != kernel_block_size:
                    raise RuntimeError("Mooncake KV cache block size differs from connector block size")
                kv_data_ptrs.append(base_address)
                kv_data_lens.append(component_size_bytes)
        if tensor_size_bytes is None or worker.num_blocks == 0:
            raise RuntimeError("Mooncake received no KV cache components")
        worker.kv_caches_base_addr = seen_base_addresses
        registration_status = worker.engine.batch_register_memory(
            kv_data_ptrs, kv_data_lens
        )
        if registration_status != 0:
            raise RuntimeError("Mooncake batch memory registration failed")
        if tensor_size_bytes % worker.num_blocks != 0:
            raise RuntimeError("Mooncake KV cache size is not divisible by its block count")
        worker.block_len = tensor_size_bytes // worker.num_blocks
        worker.device_kv_caches = kv_caches
        if worker.is_kv_consumer:
            return
        ready_event = threading.Event()
        import asyncio

        asyncio.run_coroutine_threadsafe(
            worker._mooncake_sender_listener(ready_event), worker.sender_loop
        )
        ready_event.wait()

    mooncake_connector.MooncakeConnectorWorker.register_kv_caches = (
        register_block_major_kv_components
    )


def _install_mooncake_producer_response_adapter() -> None:
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
            MooncakeConnector,
            get_mooncake_bootstrap_addr,
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Mooncake PD route adapter requires the supported vLLM "
            "MooncakeConnector implementation"
        ) from error

    def with_route_fields(
        connector: object, request: object, result: object
    ) -> object:
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("Mooncake request_finished returned an invalid result")
        delay_free_blocks, existing_params = result
        if existing_params is not None:
            if not isinstance(existing_params, Mapping):
                raise TypeError("Mooncake returned kv_transfer_params must be an object")
            return result
        _require_request_handoff(request)
        engine_id = _text_attribute(connector, "engine_id", "Mooncake connector")
        scheduler = _attribute(connector, "connector_scheduler", "Mooncake connector")
        vllm_config = _attribute(scheduler, "vllm_config", "Mooncake Scheduler")
        host, port = get_mooncake_bootstrap_addr(vllm_config)
        if not isinstance(host, str) or not host:
            raise TypeError("Mooncake bootstrap host must be non-empty text")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("Mooncake bootstrap port is invalid")
        input_params = _attribute(request, "kv_transfer_params", "Mooncake request")
        if not isinstance(input_params, Mapping):
            raise TypeError("Mooncake request handoff parameters must be an object")
        transfer_id = input_params["transfer_id"]
        if not isinstance(transfer_id, str) or not transfer_id:
            raise TypeError("Mooncake transfer_id must be non-empty text")
        return (
            delay_free_blocks,
            {
                "do_remote_prefill": True,
                "remote_engine_id": engine_id,
                "remote_bootstrap_addr": f"http://{host}:{port}",
                "transfer_id": transfer_id,
            },
        )

    original_request_finished = MooncakeConnector.request_finished

    def traced_request_finished(
        connector: object, request: object, block_ids: object
    ) -> object:
        return with_route_fields(
            connector,
            request,
            original_request_finished(connector, request, block_ids),
        )

    MooncakeConnector.request_finished = traced_request_finished
    if hasattr(MooncakeConnector, "request_finished_all_groups"):
        original_request_finished_all_groups = (
            MooncakeConnector.request_finished_all_groups
        )

        def traced_request_finished_all_groups(
            connector: object, request: object, block_ids: object
        ) -> object:
            return with_route_fields(
                connector,
                request,
                original_request_finished_all_groups(
                    connector, request, block_ids
                ),
            )

        MooncakeConnector.request_finished_all_groups = (
            traced_request_finished_all_groups
        )


def _require_request_handoff(request: object) -> None:
    params = _attribute(request, "kv_transfer_params", "Mooncake request")
    if not isinstance(params, Mapping):
        raise TypeError("Mooncake producer request must carry handoff parameters")
    if params["do_remote_decode"] is not True:
        raise ValueError("Mooncake producer request must explicitly request remote Decode")
    transfer_id = params["transfer_id"]
    if not isinstance(transfer_id, str) or not transfer_id:
        raise TypeError("Mooncake producer transfer_id must be non-empty text")


def _attribute(value: object, attribute: str, label: str) -> object:
    if not hasattr(value, attribute):
        raise RuntimeError(f"{label} lacks {attribute}")
    return getattr(value, attribute)


def _text_attribute(value: object, attribute: str, label: str) -> str:
    result = _attribute(value, attribute, label)
    if not isinstance(result, str) or not result:
        raise TypeError(f"{label}.{attribute} must be non-empty text")
    return result
