"""Lower the explicitly scoped Flash GPU/FPGA mathematical MoE design."""
from math import prod
from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_spec import DTYPE_BYTES, KernelKind, KernelSpec, PhysicalTransferConfig, linear_expert_partition
from src.serving.workload.traced_heterogeneous_moe import COMPONENT
from src.serving.workload.workload_graph import WorkloadKind


def is_heterogeneous_moe(node):
    return "source_component" in node.attributes and node.attributes["source_component"] == COMPONENT


def validate_heterogeneous_transfer(node, deployment):
    a = node.attributes
    if (is_heterogeneous_moe(node) and "heterogeneous_forward_scope" in a
            and a["heterogeneous_forward_scope"] == "gpu_decode_only"
            and a["gpu_decode_scheme"] == "dp8_ep8"):
        _validate_gpu_ep_transfer(node, deployment)
        return
    if (not is_heterogeneous_moe(node) or node.kind is not WorkloadKind.P2P
            or node.pool_name != "experts" or node.rank not in deployment.worker_pools["experts"].ranks
            or a["backend"] != "analytical_roofline_v1" or a["link"] != "roce"
            or a["algorithm"] != "send_recv" or a["source_rank"] == a["destination_rank"]):
        raise ValueError("heterogeneous payload transfer requires its FPGA-owned network service")
    pools = deployment.worker_pools
    if (not HWSpec.source_contract_supported("aggregate_gpu_source_roofline", pools["decode_gpu"].hardware_id)
            or not HWSpec.source_contract_supported("programmed_mxfp4_fp8_expert_service", pools["experts"].hardware_id)
            or set(pools["decode_gpu"].ranks) != set(range(8))):
        raise ValueError("heterogeneous link endpoint hardware or DP8 mapping disagrees")
    endpoints = a["participants"]
    if (endpoints != (a["source_rank"], a["destination_rank"])
            or not (sum(r in pools["decode_gpu"].ranks for r in endpoints) == 1
                and sum(r in pools["experts"].ranks for r in endpoints) == 1)
            or node.rank not in endpoints):
        raise ValueError("heterogeneous link must connect one GPU and its FPGA endpoint")
    expected = tuple("gpu-nic:" + str(r // 2) if r < 8 else "fpga-nic:" + str(r - 8) for r in endpoints)
    if a["participant_nic_ids"] != expected:
        raise ValueError("heterogeneous transfer lost the user-declared shared physical NICs")


def _validate_gpu_ep_transfer(node, deployment):
    """A concrete hop in the declared all-gatherv/sum-scatter ring."""
    a, pools = node.attributes, deployment.worker_pools
    if set(pools) != {"decode_gpu"}:
        raise ValueError("GPU EP ring cannot contain an FPGA or another device pool")
    pool, ring = pools["decode_gpu"], (0, 2, 4, 6, 1, 3, 5, 7)
    source, destination, phase = a["source_rank"], a["destination_rank"], a["collective_phase"]
    operation, stage, shape = a["collective_operation"], a["collective_stage"], a["payload_shape"]
    if (node.kind is not WorkloadKind.P2P or node.pool_name != pool.name or node.rank != source
            or source not in ring or destination != ring[(ring.index(source) + 1) % 8]
            or source // 2 == destination // 2 or type(phase) is not int or not 0 <= phase < 7
            or operation not in ("all_gatherv", "reduce_scatterv")
            or not HWSpec.source_contract_supported("aggregate_gpu_source_roofline", pool.hardware_id)
            or a["backend"] != "analytical_roofline_v1" or a["link"] != "roce"
            or a["algorithm"] != "send_recv" or a["operation"] != "send_recv"
            or a["direction"] != "point_to_point" or a["topology"] != "point_to_point"
            or a["bytes_semantics"] != "payload" or a["participants"] != (source, destination)
            or a["participant_nic_ids"] != ("flash-d-nic:" + str(source // 2), "flash-d-nic:" + str(destination // 2))
            or a["resource_id"] != "flash-d-roce-switch0" or a["native_provider_bound"] is not False
            or a["collective_software_startup_bound"] is not False):
        raise ValueError("GPU EP transfer lost its declared ring, shared NIC or analytical-only contract")
    expected_stage = {"hidden": "bf16", "topk_weights": "fp32", "topk_ids": "int32", "partial_output": "bf16"}
    if (stage not in expected_stage or a["payload_dtype"] != expected_stage[stage] or node.dtype != a["payload_dtype"]
            or (operation == "reduce_scatterv") != (stage == "partial_output")
            or not isinstance(shape, tuple) or len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape)
            or a["bytes"] != prod(shape) * DTYPE_BYTES[node.dtype]
            or a["nic_hbm_role_bytes"] != (a["bytes"], a["bytes"]) or a["nic_hbm_service_bound"] is not False):
        raise ValueError("GPU EP stage or typed payload bytes disagree")
    owner = ring[(ring.index(source) - phase - (operation == "reduce_scatterv")) % 8]
    if type(a["collective_segment_owner"]) is not int or a["collective_segment_owner"] != owner:
        raise ValueError("GPU EP ring forwarded the wrong collective segment")


def validate_heterogeneous_pd_transfer(node, deployment):
    a, pools = node.attributes, deployment.worker_pools
    batched = "heterogeneous_request_id" in a and a["heterogeneous_request_id"] == "fixed_batch"
    destination = node.rank if batched else 0
    if (set(pools) != {"prefill_gpu", "decode_gpu", "experts"}
            or node.kind is not WorkloadKind.PD_KV_TRANSFER or node.pool_name != "decode_gpu"
            or node.rank != destination or destination not in pools["decode_gpu"].ranks
            or a["handoff_program"] != "full_logical_arena_replica_copy_v1"
            or a["source_rank"] != pools["prefill_gpu"].ranks[0] or a["destination_rank"] != destination
            or a["participants"] != (a["source_rank"], destination)
            or a["backend"] != "analytical_roofline_v1" or a["link"] != "roce"
            or a["algorithm"] != "send_recv" or a["bytes_semantics"] != "payload"
            or a["native_provider_bound"] is not False or a["source_initialization_inferred"] is not False):
        raise ValueError("Flash logical PD requires its explicit TP replica to DP arena-copy contract")
    if (a["participant_nic_ids"] != ("prefill-gpu-nic:" + str(a["source_rank"] // 2), "gpu-nic:" + str(destination // 2))
            or not isinstance(a["payload_shape"], tuple) or not a["payload_shape"]
            or any(type(size) is not int or size <= 0 for size in a["payload_shape"])
            or a["payload_dtype"] not in ("uint8", "fp32", "int32")
            or a["bytes"] != prod(a["payload_shape"]) * DTYPE_BYTES[a["payload_dtype"]]):
        raise ValueError("Flash logical PD payload bytes or physical NIC mapping disagrees")
    role, shape, dtype = a["handoff_role"], a["payload_shape"], a["payload_dtype"]
    if not isinstance(role, str) or (role != "sampled-token" and role.count(":") != 1):
        raise ValueError("Flash PD role must name one cache object or sampled token")
    if role == "sampled-token":
        valid = (dtype == "int32" and (shape == (1, 1) if not batched else
            len(shape) == 2 and shape[1] == 1 and shape[0] <= a["heterogeneous_request_concurrency"]))
    else:
        layer, part = role.split(":", 1)
        if not layer.startswith("layer-") or not layer[6:].isdigit() or not 0 <= int(layer[6:]) < 43:
            raise ValueError("Flash PD cache role must belong to a checkpoint layer")
        if part in ("swa-cache", "compressed-cache", "indexer-cache"):
            valid = dtype == "uint8" and len(shape) == 2 and shape[-1] == (132 if part == "indexer-cache" else 584)
        elif part == "partial-state":
            valid = dtype == "fp32" and len(shape) == 3 and shape[1:] in ((4, 2048), (8, 1024))
        elif part == "indexer-partial-state":
            valid = dtype == "fp32" and len(shape) == 3 and shape[1:] == (4, 512)
        else:
            valid = False
    if not valid:
        raise ValueError("Flash PD role and typed arena layout disagree")


def _math_spec(kind, hardware_id, node, attributes):
    shape = dict(node.shape)
    if "elements" in shape:
        shape["element_count"] = shape.pop("elements")
    return KernelSpec(kind, hardware_id, shape=shape, dtype=node.dtype, variant=node.variant,
        attributes=attributes, execution_support="unsupported", unsupported_reason="launch_selector_unavailable")


def heterogeneous_moe_specs(node, hardware_id):
    if not is_heterogeneous_moe(node):
        raise ValueError("heterogeneous lowering requires its declared source component")
    a = node.attributes
    if node.kind in (WorkloadKind.P2P, WorkloadKind.COLLECTIVE, WorkloadKind.PD_KV_TRANSFER):
        transfer = PhysicalTransferConfig(**{key: a[key] for key in (
            "algorithm", "bytes", "direction", "link", "topology", "participants", "operation",
            "bytes_semantics", "source_rank", "destination_rank", "backend", "resource_id", "participant_nic_ids")},
            **{key: a[key] for key in ("ring_order", "reduction", "backend_config_sha256") if key in a})
        return (KernelSpec(KernelKind.COMMUNICATION, hardware_id, transfer=transfer),)
    if node.kind is WorkloadKind.MEMCPY:
        transfer = PhysicalTransferConfig(**{key: a[key] for key in ("algorithm", "bytes", "direction", "link")})
        return (KernelSpec(KernelKind.MEMCPY, hardware_id, transfer=transfer),)
    variant, dtype, shape = node.variant, node.dtype, dict(node.shape)
    if node.kind is WorkloadKind.FUSED_MOE:
        names = ("tokens_per_expert", "local_tokens_per_expert", "expert_parallel_size", "expert_rank",
            "owned_experts", "routed_scaling_factor", "routed_weight_dtype", "weight_group_size", "swiglu_limit")
        return (KernelSpec(KernelKind.FUSED_MOE, hardware_id, shape=node.shape, dtype="bf16", variant="routed_experts",
            attributes={**{key: a[key] for key in names}, "expert_weight_tile_rows": None,
                "mathematical_program": "dsv4_mxfp4_w4a16_swiglu_v1"},
            execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
    if variant == "vocab_parallel_embedding":
        return (_math_spec(KernelKind.VECTOR, hardware_id, node,
            {key: a[key] for key in ("vocab_size", "tensor_parallel_size", "tensor_parallel_rank", "owned_token_rows")}),)
    if variant == "dsv4_indexer_q_rope_quant":
        return (_math_spec(KernelKind.VECTOR, hardware_id, node, {}),)
    if node.kind is WorkloadKind.EAGER_ATTN and variant in ("dsv4_indexer_prefill", "dsv4_indexer_decode"):
        names = ("causal", "score_accumulator_dtype", "score_function", "cache_layout", "cache_quantization",
            "q_lens", "kv_lens", "request_phases", "topk_capacity", "topk_index_dtype", "topk_sentinel")
        return (_math_spec(KernelKind.SPARSE_ATTN_INDEXER, hardware_id, node,
            {**{key: a[key] for key in names}, "context_parallel_size": 1}),)
    if node.kind is WorkloadKind.EAGER_ATTN:
        names = ("causal", "indexer_type", "kv_layout", "kv_dist", "q_lens", "kv_lens",
            "selected_kv_lens", "swa_kv_lens", "topk_capacity", "topk_index_dtype", "compressed_selection")
        return (_math_spec(KernelKind.SPARSE_ATTN, hardware_id, node,
            {**{key: a[key] for key in names}, "mathematical_program": "dsv4_sparse_softmax_flops_bytes_v1"}),)
    if node.kind is WorkloadKind.KV_UPDATE and variant == "dsv4_qnorm_rope_kv_insert":
        return (_math_spec(KernelKind.KV_CACHE, hardware_id, node, {key: a[key] for key in (
            "cache_layout", "cache_write_semantics", "quantization", "slot_mapping_all_valid")}),)
    if node.kind is WorkloadKind.KV_UPDATE and variant in ("dsv4_compressor_state_cache", "dsv4_compressed_state_cache"):
        names = ("cache_layout", "cache_write_semantics") if variant == "dsv4_compressor_state_cache" else (
            "cache_layout", "cache_write_semantics", "quantization", "state_old_lens", "state_new_lens")
        attributes = {key: a[key] for key in names}
        if "finish_stage" in a:
            attributes["finish_stage"] = a["finish_stage"]
        return (_math_spec(KernelKind.KV_CACHE, hardware_id, node, attributes),)
    if variant == "dsv4_short_context_topk":
        return (_math_spec(KernelKind.VECTOR, hardware_id, node, {"position_bytes": a["position_bytes"]}),)
    if node.kind is WorkloadKind.NORM and variant == "dsv4_q_kv_rmsnorm":
        return (_math_spec(KernelKind.VECTOR, hardware_id, node, {"epsilon": a["epsilon"]}),)
    if variant in ("mhc_pre", "mhc_post_pre"):
        return (_math_spec(KernelKind.VECTOR, hardware_id, node, {"mathematical_program": a["mathematical_program"],
            "sinkhorn_iters": a["sinkhorn_iters"]}),)
    if variant == "dsv4_inv_rope_fp8_quant":
        return (_math_spec(KernelKind.VECTOR, hardware_id, node, {"quant_group_size": a["quant_group_size"]}),)
    if node.kind in (WorkloadKind.LINEAR, WorkloadKind.LM_HEAD) and dtype == "bf16":
        if "output_dtype" in a and a["output_dtype"] == "fp32":
            return (_math_spec(KernelKind.GEMM, hardware_id, node, {"output_dtype": "fp32"}),)
        if variant == "batched_gemm":
            attributes = {key: a[key] for key in (
                "batch_layout", "a_batch_stride_elements", "b_batch_stride_elements", "c_batch_stride_elements",
                "weight_dtype", "weight_group_size")}
            if attributes["weight_dtype"] == "fp8":
                attributes["mathematical_program"] = "batched_fp8_quantized_input_fp32_scales_v1"
            return (_math_spec(KernelKind.GEMM, hardware_id, node, attributes),)
        if "weight_dtype" in a and a["weight_dtype"] == "fp8":
            m, k = shape["m"], shape["k"]
            quant = KernelSpec(KernelKind.VECTOR, hardware_id, dtype="bf16", variant="fp8_group_quant",
                shape={"element_count": m * k, "rows": m, "columns": k}, attributes={"group_size": 128},
                execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
            return (quant, _math_spec(KernelKind.GEMM, hardware_id, node,
                {"mathematical_program": "block_fp8_fp32_scales_v1"}))
        b, m, n, k = (shape[key] for key in ("b", "m", "n", "k"))
        return (KernelSpec(KernelKind.GEMM, hardware_id, dtype="bf16", variant="batched_gemm", shape=shape,
            attributes={"batch_layout": "strided", "a_batch_stride_elements": m*k,
                "b_batch_stride_elements": k*n, "c_batch_stride_elements": m*n,
                "weight_dtype": "bf16", "weight_group_size": 0}, execution_support="unsupported",
            unsupported_reason="launch_selector_unavailable"),)
    if variant in ("mhc_post", "hc_head"):
        return (_math_spec(KernelKind.VECTOR, hardware_id, node, {}),)
    if variant == "rmsnorm":
        return (_math_spec(KernelKind.VECTOR, hardware_id, node, {"epsilon": a["epsilon"]}),)
    if variant == "vllm_embedding_id_widen":
        return (_math_spec(KernelKind.VECTOR, hardware_id, node, {"embedding_source": a["embedding_source"]}),)
    if variant == "embedding":
        return (KernelSpec(KernelKind.VECTOR, hardware_id, shape=shape, dtype=dtype, variant="gather",
            attributes={key: a[key] for key in ("gather_factor", "index_count", "index_bytes", "scattered", "materialized_output")},
            execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
    if variant == "vllm_logits_row_select":
        return (_math_spec(KernelKind.VECTOR, hardware_id, node, {key: a[key] for key in (
            "model_execution_version", "index_source", "selected_indices")}),)
    if variant == "vllm_logits_fp32_cast":
        return (_math_spec(KernelKind.VECTOR, hardware_id, node, {"sampling_source": a["sampling_source"]}),)
    if variant == "vllm_logits_gather_reorder":
        return (_math_spec(KernelKind.VECTOR, hardware_id, node,
            {"model_execution_version": a["model_execution_version"]}),)
    if variant == "vllm_plain_greedy_sample":
        return (_math_spec(KernelKind.VECTOR, hardware_id, node, {key: a[key] for key in ("sampling_source", "stage")}),)
    if node.kind in (WorkloadKind.LINEAR, WorkloadKind.ROUTER):
        kind = KernelKind.GEMM
        if node.kind is WorkloadKind.ROUTER:
            attributes = {"output_dtype": "fp32", "weight_dtype": "bf16", "weight_group_size": 0}
        else:
            attributes = {key: a[key] for key in ("weight_dtype", "weight_group_size", "has_bias")}
            attributes.update(output_dtype="bf16", mathematical_program=(
                "mxfp4_fp8_flops_bytes_v1" if HWSpec.source_contract_supported("programmed_mxfp4_fp8_expert_service", hardware_id)
                else "block_fp8_quantized_input_fp32_scales_v1"))
    else:
        kind = KernelKind.VECTOR
        if variant == "dsv4_moe_route":
            attributes = {key: a[key] for key in ("expert_id_selection", "routed_scaling_factor")}
        elif variant == "fp8_group_quant":
            attributes = {key: a[key] for key in ("group_size", "input_resource")}
        elif variant == "silu":
            attributes = {"swiglu_limit": a["swiglu_limit"]}
        elif variant == "gather":
            attributes = {key: a[key] for key in ("gather_factor", "index_count", "index_bytes", "scattered", "materialized_output")}
            if HWSpec.source_contract_supported("aggregate_gpu_source_roofline", hardware_id) and dtype == "bf16":
                attributes["mathematical_program"] = "materialized_bf16_gather_bytes_v1"
        elif variant == "moe_sum":
            attributes = {"work_fraction": a["work_fraction"], "routed_weights_dtype": "fp32"}
        elif variant == "add":
            variant = "elementwise"
            attributes = {key: a[key] for key in ("operation", "reads", "writes")}
        else:
            raise NotImplementedError(f"heterogeneous MoE has no mathematical binding for {variant!r}")
    return (KernelSpec(kind, hardware_id, shape=shape, dtype=dtype, variant=variant,
        attributes=attributes, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)


def validate_heterogeneous_graph(graph, deployment):
    if not graph.nodes or not all(is_heterogeneous_moe(n) for n in graph.nodes):
        raise ValueError("heterogeneous mathematical boundary requires its complete scoped graph")
    if set(deployment.worker_pools) == {"decode_gpu"}:
        pool = deployment.worker_pools["decode_gpu"]
        c = pool.parallel_plan.config
        schemes = {n.attributes["gpu_decode_scheme"] for n in graph.nodes}
        if len(schemes) != 1 or not schemes <= {"tp8", "dp8_ep8"}:
            raise ValueError("Flash D-only graph must bind one explicit GPU parallel scheme")
        scheme = next(iter(schemes))
        axes = ((c.tensor_parallel_size, c.data_parallel_size, c.enable_expert_parallel) ==
            ((8, 1, False) if scheme == "tp8" else (1, 8, True)))
        if (deployment.mode != "colocated" or not HWSpec.source_contract_supported(
                    "aggregate_gpu_source_roofline", pool.hardware_id)
                or pool.ranks != tuple(range(8)) or not axes
                or c.pipeline_parallel_size != 1 or c.context_parallel_size != 1
                or c.sequence_parallel_size != 1
                or any(n.pool_name != pool.name
                    or n.attributes["heterogeneous_forward_scope"] != "gpu_decode_only"
                    or n.attributes["gpu_decode_microbatch"] != 1 for n in graph.nodes)):
            raise ValueError("Flash D-only requires its explicit eight GPU pool and parallel axes")
        for node in graph.nodes:
            if node.kind is WorkloadKind.P2P:
                if scheme != "dp8_ep8":
                    raise ValueError("TP8 cannot import an EP dispatch hop")
                _validate_gpu_ep_transfer(node, deployment)
            elif node.kind is WorkloadKind.COLLECTIVE:
                if scheme != "tp8" or node.attributes["participant_nic_ids"] != tuple(
                        "flash-d-nic:" + str(r // 2) for r in pool.ranks):
                    raise ValueError("D-only collective lost the shared physical NIC mapping")
            elif node.kind is WorkloadKind.FUSED_MOE and scheme == "dp8_ep8":
                a, experts, tokens, topk = node.attributes, node.shape["experts"], node.shape["tokens"], node.shape["topk"]
                owned = tuple(linear_expert_partition(experts, 8, node.rank))
                global_counts, local_counts = a["tokens_per_expert"], a["local_tokens_per_expert"]
                if (a["expert_parallel_size"] != 8 or a["expert_rank"] != node.rank or a["owned_experts"] != owned
                        or len(global_counts) != experts or sum(global_counts) != tokens * topk
                        or any(type(count) is not int or not 0 <= count <= tokens for count in global_counts)
                        or local_counts != tuple(count if e in owned else 0 for e, count in enumerate(global_counts))
                        or len(a["gpu_decode_global_row_request_ids"]) != tokens
                        or sorted(a["gpu_decode_global_row_request_ids"]) != list(range(tokens))
                        or a["output_is_reduced"] is not False):
                    raise ValueError("GPU EP expert ownership, routing or global row order disagrees")
        return
    if "prefill_gpu" in deployment.worker_pools:
        pool = deployment.worker_pools["prefill_gpu"]
        c = pool.parallel_plan.config
        if (deployment.mode != "colocated" or not HWSpec.source_contract_supported("aggregate_gpu_source_roofline", pool.hardware_id)
                or pool.ranks != tuple(range(pool.ranks[0], pool.ranks[0] + 8)) or c.tensor_parallel_size != 8
                or c.pipeline_parallel_size != 1 or c.data_parallel_size != 1 or c.enable_expert_parallel
                or c.sequence_parallel_size != 1 or c.context_parallel_size != 1
                or any(n.attributes["heterogeneous_forward_scope"] != "isolated_prefill"
                    for n in graph.nodes if n.pool_name == pool.name)):
            raise ValueError("isolated Flash prefill requires its declared BW1100 TP8 GPU-only graph")
        for node in graph.nodes:
            if node.pool_name == pool.name and node.kind is WorkloadKind.COLLECTIVE and node.attributes["participant_nic_ids"] != tuple(
                    "prefill-gpu-nic:" + str(r // 2) for r in pool.ranks):
                raise ValueError("prefill collective lost two-GPU shared NIC mapping")
        if set(deployment.worker_pools) == {"prefill_gpu"}:
            if any(n.pool_name != pool.name for n in graph.nodes):
                raise ValueError("isolated prefill graph contains another pool")
            return
        if (set(deployment.worker_pools) != {"prefill_gpu", "decode_gpu", "experts"}
                or pool.ranks != tuple(range(8 + len(deployment.worker_pools["experts"].ranks),
                    16 + len(deployment.worker_pools["experts"].ranks)))
                or any("heterogeneous_request_id" not in n.attributes
                    or n.attributes["heterogeneous_request_id"] not in ("one", "fixed_batch") for n in graph.nodes)):
            raise ValueError("heterogeneous request requires disjoint TP8 prefill and its explicit request scope")
        scopes = {n.attributes["heterogeneous_request_id"] for n in graph.nodes}
        if len(scopes) != 1:
            raise ValueError("heterogeneous request cannot mix request scopes")
        if scopes == {"fixed_batch"}:
            fields = ("heterogeneous_request_concurrency", "heterogeneous_request_microbatches")
            if any(any(field not in n.attributes for field in fields) for n in graph.nodes):
                raise ValueError("fixed request graph requires explicit concurrency/microbatch scope")
            configurations = {tuple(n.attributes[field] for field in fields) for n in graph.nodes}
            if len(configurations) != 1:
                raise ValueError("fixed request graph has inconsistent concurrency/microbatch scope")
            concurrency, microbatches = next(iter(configurations))
            if (type(concurrency) is not int or concurrency < 1
                    or type(microbatches) is not int or microbatches not in (1, 2, 3)):
                raise ValueError("invalid fixed request concurrency/microbatch scope")
    if deployment.mode != "colocated" or not set(deployment.worker_pools) in (
            {"decode_gpu", "experts"}, {"prefill_gpu", "decode_gpu", "experts"}):
        raise ValueError("heterogeneous boundary requires exactly its two declared worker pools")
    gpu, fpga = (deployment.worker_pools[key] for key in ("decode_gpu", "experts"))
    if (not HWSpec.source_contract_supported("aggregate_gpu_source_roofline", gpu.hardware_id)
            or not HWSpec.source_contract_supported("programmed_mxfp4_fp8_expert_service", fpga.hardware_id)
            or gpu.ranks != tuple(range(8)) or len(fpga.ranks) not in (16, 32)
            or fpga.ranks != tuple(range(8, 8 + len(fpga.ranks)))):
        raise ValueError("heterogeneous boundary requires DP8 BW1100 and EP16/32 FPGA chip ranks")
    for pool, dp, ep in ((gpu, 8, False), (fpga, len(fpga.ranks), True)):
        c = pool.parallel_plan.config
        if (c.tensor_parallel_size != 1 or c.pipeline_parallel_size != 1
                or c.data_parallel_size != dp or c.enable_expert_parallel is not ep
                or c.sequence_parallel_size != 1 or c.context_parallel_size != 1):
            raise ValueError("heterogeneous boundary parallel axes disagree with the declared design")
    for node in graph.nodes:
        if node.kind is WorkloadKind.P2P:
            validate_heterogeneous_transfer(node, deployment)
        elif node.kind is WorkloadKind.PD_KV_TRANSFER:
            validate_heterogeneous_pd_transfer(node, deployment)
    if (not any(n.kind is WorkloadKind.P2P for n in graph.nodes)
            and not all(n.pool_name == "prefill_gpu" for n in graph.nodes)
            and not ("prefill_gpu" in deployment.worker_pools
                and any(n.kind is WorkloadKind.PD_KV_TRANSFER for n in graph.nodes)
                and all(n.pool_name in ("prefill_gpu", "decode_gpu")
                    and n.attributes["heterogeneous_request_id"] == "fixed_batch" for n in graph.nodes))):
        raise ValueError("heterogeneous boundary must contain actual GPU/FPGA exchanges")


def lower_flash_fp4_decode_scenario(graph, deployment, hardware_by_id):
    """Lower the explicitly tagged FP4 source through existing kernel owners."""
    from dataclasses import replace
    from .kernel_plan import KernelCall, KernelPlan
    from src.serving.workload.traced_flash_decode_scenario import COMPONENT as scenario_component
    if not graph.nodes or any(n.attributes["source_component"] != scenario_component for n in graph.nodes):
        raise ValueError("FP4 switch lowering requires one complete scenario source")
    pools = deployment.worker_pools
    gpu_id = pools["decode_gpu"].hardware_id
    if not HWSpec.source_contract_supported("aggregate_gpu_source_roofline", gpu_id):
        raise ValueError("FP4 scenario GPU is outside its declared hardware domain")
    node_calls, calls = {}, []
    for node_id in graph.topological_order():
        node = graph.node(node_id)
        if node.pool_name not in pools or node.rank not in pools[node.pool_name].ranks:
            raise ValueError("FP4 decode node rank/pool placement disagrees")
        hardware_id = pools[node.pool_name].hardware_id
        a = node.attributes
        hardware = hardware_by_id[hardware_id]
        if a["scenario_gpu_hardware_id"] != gpu_id:
            raise ValueError("FP4 source GPU identity differs from its deployment")
        scheme = a["scenario_scheme"]
        gpu_count = a["scenario_gpu_count"]
        config = pools["decode_gpu"].parallel_plan.config
        if (scheme not in ("tp8_ep8", "dp8_ep8", "dp8_fpga")
                or (config.tensor_parallel_size, config.data_parallel_size, config.enable_expert_parallel)
                    != ((8, 1, True) if scheme == "tp8_ep8" else (1, gpu_count, True))
                or pools["decode_gpu"].ranks != tuple(range(gpu_count))
                or (gpu_count != 8 and (gpu_count != 16 or scheme != "dp8_fpga" or a["scenario_gpu_compute_format"] != "fp4"))
                or ("experts" in pools) != (scheme == "dp8_fpga")):
            raise ValueError("FP4 decode parallel axes and expert placement disagree")
        dependencies = tuple(identifier for n in graph.predecessors(node.node_id) for identifier in node_calls[n])
        if node.kind in (WorkloadKind.LINEAR, WorkloadKind.LM_HEAD, WorkloadKind.ROUTER):
            shape = {name: node.shape[name] for name in ("b", "m", "n", "k")}
            b, m = shape["b"], shape["m"]
            row_counts = a["scenario_matrix_row_counts"] if "scenario_matrix_row_counts" in a else (m,) * b
            output = a["output_dtype"] if "output_dtype" in a and a["output_dtype"] is not None else "bf16"
            compute = a["scenario_gpu_compute_format"] if node.pool_name == "decode_gpu" else "fp8"
            hardware.resource_by_role("compute_tensor_" + compute)
            specs = (KernelSpec(KernelKind.GEMM, hardware_id, shape=shape, dtype=node.dtype, variant="gemm",
                attributes={"mathematical_program": "packed_fp4_source_v1", "weight_dtype": "mxfp4",
                    "weight_group_size": 32, "has_bias": False, "output_dtype": output,
                    "compute_format": compute, "weight_matrix_count": len(row_counts),
                    "matrix_row_counts": row_counts, "input_quantized": node.dtype == "fp8" and compute == "fp8"},
                execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
        elif node.kind is WorkloadKind.COLLECTIVE:
            if "traffic_matrix_bytes" in a:
                participants, matrix = tuple(a["participants"]), a["traffic_matrix_bytes"]
            else:
                participants = tuple(a["participants"])
                count, payload = len(participants), a["bytes"]
                if a["operation"] not in ("all_reduce", "all_gather"):
                    raise ValueError("unknown tensor direct collective")
                # Full-mesh broadcast-and-local-sum all-reduce; source-sized
                # all-gather. No ring traffic multiplier or switch reduction.
                matrix = tuple(tuple(0 if i == j else payload for j in range(count)) for i in range(count))
            pcie = a["scenario_gpu_direct_link"] == "pcie" and a["scenario_scheme"] != "dp8_fpga"
            if pcie:
                hardware.analytical_link_bandwidth_bytes_s("pcie")
            if pcie:
                if participants != tuple(range(8)):
                    raise ValueError("PCIe switch scenario requires the8 GPU endpoints")
                projected = [list(row) + [0, 0] for row in matrix] + [[0] * 10 for _ in range(2)]
                # Two four-GPU PCIe5 switches; each has one x16 upstream
                # link to a common nonblocking root. GPU ports and these
                # shared uplinks service the same streamed payload in parallel.
                projected[8][9] = sum(matrix[i][j] for i in range(4) for j in range(4, 8))
                projected[9][8] = sum(matrix[i][j] for i in range(4, 8) for j in range(4))
                matrix, participants = tuple(map(tuple, projected)), tuple(range(10))
                nic_ids = tuple("pcie-gpu:" + str(r) for r in range(8)) + ("pcie-uplink:0", "pcie-uplink:1")
            else:
                nic_ids = tuple("gpu-nic:" + str(r // 2) if r < gpu_count else "fpga-nic:" + str(r - gpu_count)
                    for r in participants)
            transfer = PhysicalTransferConfig(algorithm="pairwise", bytes=sum(map(sum, matrix)),
                direction="collective", link="pcie" if pcie else "roce", topology="fully_connected",
                participants=participants, operation="all_to_all", bytes_semantics="aggregate_input",
                backend="analytical_roofline_v1", resource_id="flash-direct-switch",
                participant_nic_ids=nic_ids, traffic_matrix_bytes=matrix)
            specs = (KernelSpec(KernelKind.COMMUNICATION, hardware_id, transfer=transfer),)
        else:
            old = replace(node, attributes={**a, "source_component": COMPONENT})
            specs = heterogeneous_moe_specs(old, hardware_id)
            if (node.pool_name == "decode_gpu" and node.variant in (
                    "embedding", "vocab_parallel_embedding", "mhc_pre", "mhc_post_pre", "hc_head")):
                specs = tuple(replace(spec, attributes={**spec.attributes,
                    "packed_weight_storage": "mxfp4_group32"}) for spec in specs)
        ids = []
        for ordinal, spec in enumerate(specs):
            identifier = node.node_id + f":kernel-{ordinal}"
            resource = spec.transfer.resource_id if spec.kind is KernelKind.COMMUNICATION else (
                spec.transfer.link if spec.transfer is not None else "compute")
            calls.append(KernelCall(identifier, spec, node.pool_name, node.rank, resource,
                (node.node_id,), dependencies if ordinal == 0 else (ids[-1],)))
            ids.append(identifier)
        if (node.kind is WorkloadKind.COLLECTIVE and "operation" in a
                and a["operation"] == "all_reduce"):
            local_sums = []
            for rank in a["participants"]:
                identifier = node.node_id + f":local-sum-rank-{rank}"
                spec = KernelSpec(KernelKind.VECTOR, hardware_id, dtype=node.dtype, variant="reduce",
                    shape={"element_count": a["bytes"] // DTYPE_BYTES[node.dtype]},
                    attributes={"reduction_factor": len(a["participants"])},
                    execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
                calls.append(KernelCall(identifier, spec, node.pool_name, rank, "compute", (node.node_id,), (ids[-1],)))
                local_sums.append(identifier)
            node_calls[node.node_id] = tuple(local_sums)
        else:
            node_calls[node.node_id] = (ids[-1],)
    return KernelPlan(calls)
