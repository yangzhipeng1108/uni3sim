"""Request-scoped PD cache transport, independent of model forward construction."""

from src.serving.deployment.deployment_config import worker_role_phase
from src.kernel.kernel_spec import DTYPE_BYTES
from src.serving.deployment.deployment_plan import DeploymentPlan, pd_kv_transfer_rank_pairs
from .model_config import ModelConfig
from .workload_graph import WorkloadGraph, WorkloadKind, WorkloadNode


def gqa_kv_interval_bytes_per_token(layers, heads, head_dim, cache_dtype):
    """Payload of one contiguous logical K/V layer/head interval."""
    if any(type(value) is not int or value <= 0 for value in (layers, heads, head_dim)):
        raise ValueError("GQA KV interval dimensions must be positive integers")
    if cache_dtype not in ("bf16", "fp16", "fp32"):
        raise ValueError("GQA KV cache requires a declared floating-point dtype")
    return 2 * layers * heads * head_dim * DTYPE_BYTES[cache_dtype]


def validate_pd_kv_layout(source, destination, attention_layout):
    if source.kv_transfer_layout != destination.kv_transfer_layout:
        raise NotImplementedError("PD KV cache layout conversion is not implemented")
    layout = source.kv_transfer_layout
    if source.hardware_id != destination.hardware_id and layout is None:
        raise ValueError("cross-hardware PD requires explicit cache layouts at both endpoints")
    if attention_layout not in ("standard_gqa", "mla", "dsa_mla"):
        raise ValueError("PD KV cache has an unsupported attention format")
    if layout is not None and ((attention_layout == "standard_gqa" and layout not in ("BLHNC", "BLNHC"))
            or (attention_layout != "standard_gqa" and layout != "LBHNC")):
        raise ValueError("declared cache layout is incompatible with the model's attention format")
    return layout


def pd_transfer_node_nic_ids(source, destination, route, source_rank, destination_rank):
    if bool(source.node_ids) != bool(destination.node_ids):
        raise ValueError("node-port transport requires explicit placement at both endpoints")
    if not source.node_ids:
        return ()  # Legacy route-scoped contract; no multinode certification.
    if route.link != "infiniband" or route.backend != "analytical_roofline_v1":
        raise NotImplementedError("node-port mapping currently supports analytical InfiniBand")
    if route.kv_transport_scope != "shared_fabric":
        raise ValueError("single node-port mapping cannot declare independent per-rank links")
    return (source.node_id_for_physical_rank(source_rank),
        destination.node_id_for_physical_rank(destination_rank))


def build_pd_kv_transfer_workload(
    request_id: str,
    context_tokens: int,
    model: ModelConfig,
    deployment: DeploymentPlan,
    source_pool_name: str,
    destination_pool_name: str,
    *, source_data_parallel_rank: int | None = None, destination_data_parallel_rank: int | None = None,
) -> WorkloadGraph:
    """Build one request-scoped P-to-D KV transfer after routing."""

    if not isinstance(request_id, str) or not request_id:
        raise ValueError("KV transfer request_id must be non-empty")
    if isinstance(context_tokens, bool) or not isinstance(context_tokens, int):
        raise TypeError("KV transfer context_tokens must be an integer")
    if context_tokens <= 0:
        raise ValueError("KV transfer context_tokens must be positive")
    if deployment.mode not in ("pd_disagg", "pd_af_disagg"):
        raise ValueError("KV transfer workload requires a PD deployment")
    if source_pool_name not in deployment.worker_pools:
        raise ValueError(f"unknown KV source pool {source_pool_name!r}")
    if destination_pool_name not in deployment.worker_pools:
        raise ValueError(
            f"unknown KV destination pool {destination_pool_name!r}")
    source = deployment.worker_pools[source_pool_name]
    destination = deployment.worker_pools[destination_pool_name]
    layout = validate_pd_kv_layout(source, destination, model.attention_layout)
    if worker_role_phase(source.role) != "prefill":
        raise ValueError("KV transfer source must have a prefill role")
    if worker_role_phase(destination.role) != "decode":
        raise ValueError("KV transfer destination must have a decode role")
    if model.attention_layout in ("mla", "dsa_mla"):
        # These are complete single-head compressed caches on every TP/PCP
        # rank. Query heads and routed experts never divide their payload.
        for endpoint in (source, destination):
            if model.num_attention_heads % endpoint.parallel_plan.config.tensor_parallel_size:
                raise ValueError("MLA query heads must be divisible by endpoint TP")
    routes = [
        route for route in deployment.routes
        if route.payload == "kv"
        and route.source_pool == source_pool_name
        and route.destination_pool == destination_pool_name
    ]
    if len(routes) != 1:
        raise ValueError(
            "request-scoped KV transfer requires exactly one route from "
            f"{source_pool_name!r} to {destination_pool_name!r}")
    route = routes[0]
    rank_pairs = pd_kv_transfer_rank_pairs(
        source, destination, model.num_key_value_heads, source_data_parallel_rank=source_data_parallel_rank,
        destination_data_parallel_rank=destination_data_parallel_rank, attention_layout=model.attention_layout, kv_transport_scope=route.kv_transport_scope)
    if source.hardware_id != destination.hardware_id and (
            route.link != "infiniband"
            or route.backend != "analytical_roofline_v1"):
        raise ValueError(
            "cross-hardware KV requires a declared analytical InfiniBand route; "
            "device-local fabrics cannot imply physical connectivity")
    source_tensor_parallel_size = (
        source.parallel_plan.config.tensor_parallel_size)
    destination_tensor_parallel_size = (
        destination.parallel_plan.config.tensor_parallel_size)
    base_node_id = (
        f"{request_id}:pd-kv-transfer:"
        f"{source_pool_name}-to-{destination_pool_name}"
    )
    nodes = []
    for edge_index, rank_pair in enumerate(rank_pairs):
        source_rank = rank_pair.source_rank
        destination_rank = rank_pair.destination_rank
        overlapping_kv_heads = (
            rank_pair.kv_head_end - rank_pair.kv_head_start)
        if model.attention_layout == "standard_gqa":
            bytes_per_token = gqa_kv_interval_bytes_per_token(
                rank_pair.layer_end - rank_pair.layer_start, overlapping_kv_heads,
                model.head_dim, model.compute_dtype)
        else:
            bytes_per_token = model.kv_bytes_per_token_for_layer_range(
                source_tensor_parallel_size, rank_pair.layer_start, rank_pair.layer_end)
        node_id = (
            base_node_id
            if len(rank_pairs) == 1
            else f"{base_node_id}:kv-edge-{edge_index}"
        )
        nodes.append(WorkloadNode(
            node_id=node_id,
            kind=WorkloadKind.PD_KV_TRANSFER,
            pool_name=source_pool_name,
            shape={},
            rank=source_rank,
            attributes={
                "algorithm": route.algorithm,
                "bytes": context_tokens * bytes_per_token,
                "token_start": 0,
                "token_end": context_tokens,
                "bytes_per_token": bytes_per_token,
                "attention_layout": model.attention_layout,
                "head_dim": model.head_dim if model.attention_layout == "standard_gqa" else None,
                "cache_dtype": model.compute_dtype,
                "cache_layout": layout,
                "bytes_semantics": "payload",
                "backend": route.backend,
                "backend_config_sha256": route.backend_config_sha256,
                "direction": "point_to_point",
                "link": route.link,
                "topology": route.topology,
                "participants": (source_rank, destination_rank),
                **({"participant_nic_ids": pd_transfer_node_nic_ids(
                    source, destination, route, source_rank, destination_rank)}
                    if source.node_ids or destination.node_ids else {}),
                "operation": "send_recv",
                "source_rank": source_rank,
                "destination_rank": destination_rank,
                "ring_order": (),
                "reduction": None,
                "resource_id": route.physical_resource_id(rank_pair.destination_logical_rank),
                "destination_pool": destination_pool_name,
                "rank_mapping": "kv_head_overlap" if model.attention_layout == "standard_gqa" else "replicated_mla_cache",
                "source_logical_rank": rank_pair.source_logical_rank,
                "destination_logical_rank": (
                    rank_pair.destination_logical_rank),
                "source_tensor_parallel_size": source_tensor_parallel_size,
                "destination_tensor_parallel_size": (
                    destination_tensor_parallel_size),
                "kv_head_start": rank_pair.kv_head_start,
                "kv_head_end": rank_pair.kv_head_end,
                "layer_start": rank_pair.layer_start,
                "layer_end": rank_pair.layer_end,
                "num_key_value_heads": model.num_key_value_heads if model.attention_layout == "standard_gqa" else 1,
                **({"model_num_key_value_heads": model.num_key_value_heads,
                    "compressed_dim": model.kv_lora_rank + model.qk_rope_head_dim,
                    "index_head_dim": model.index_head_dim,
                    "full_indexer_layers": tuple(layer for layer in range(rank_pair.layer_start, rank_pair.layer_end)
                        if model.indexer_types[layer] == "full") if model.attention_layout == "dsa_mla" else (),
                    "source_context_parallel_size": source.parallel_plan.config.context_parallel_size,
                    "destination_context_parallel_size": destination.parallel_plan.config.context_parallel_size}
                    if model.attention_layout in ("mla", "dsa_mla") else {}),
            },
        ))
    return WorkloadGraph(tuple(nodes))
