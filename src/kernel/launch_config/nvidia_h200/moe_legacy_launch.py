"""H200 vLLM0.11 BF16 expert host rules and Torch2.8 reduction geometry.

The pinned vLLM software snapshot supplies fused_moe.py/get_default_config,
invoke_fused_moe_kernel and moe_align_block_size.py. The runtime's own H200
config files and Triton3.4 CUDAOptions supply discrete tile/warp/stage facts.
This version has no0.30 naive assignment or small-allocation cap. No compiled
SM80 resource values or another hardware selector are imported.
"""

from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.nvidia_h200.moe_legacy_configs import H200_VLLM011_BF16_CONFIGS
from src.kernel.launch_config.selection_primitives import (
    ceil_div, fused_moe_local_expert_count, require_dtype, require_positive_integer, require_shape,
)


def legacy_moe_config(tokens, experts, intermediate):
    if (experts, intermediate) in H200_VLLM011_BF16_CONFIGS:
        rows = H200_VLLM011_BF16_CONFIGS[experts, intermediate]
        return min(rows, key=lambda row: abs(row[0] - tokens))[1:]
    if tokens <= experts:
        return 16, 32, 64, 1, 4, 3
    return 64, 64, 32, 8, 4, 3


def _require_legacy_source(attributes, hardware):
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("SM90 legacy MoE source requires H200 HWSpec")
    if attributes["model_execution_version"] != "vllm_0_11_0":
        raise NotImplementedError("legacy MoE source binds only vLLM0.11")


def select_legacy_moe_recipe(shape, attributes, hardware):
    """Single-chunk softmax BF16 submission roles, including Torch helpers.

    Router roles and expert-map roles select the existing elementwise,
    reduce and gather variants through source_moe_router_stage. The root
    public div contract/lowerer bridge is still needed for normalize_div;
    this stage sequence alone does not claim complete operator pricing.
    """
    from src.kernel.launch_config.nvidia_h200.moe_aux_launch import SOURCE_TOPK_EXPERT_COUNTS

    _require_legacy_source(attributes, hardware)
    required = {"source_routing", "routed_weight_dtype", "source_moe_config",
        "source_batch_invariant", "source_router_renormalize", "routing_input_dtype",
        "source_has_expert_map", "expert_parallel_size", "source_moe_chunk_size"}
    missing = required - set(attributes)
    if missing:
        raise NotImplementedError(
            "vLLM0.11 H200 MoE requires explicit runtime/backend source facts; "
            f"missing={sorted(missing)}")
    tokens, experts, topk = require_shape(shape, ("tokens", "experts", "topk"))
    if topk > experts:
        raise ValueError("topk exceeds global experts")
    if (attributes["source_routing"] != "softmax"
            or attributes["routed_weight_dtype"] != "bf16"
            or attributes["source_moe_config"] != "bundled_or_default"
            or attributes["source_batch_invariant"] is not False):
        raise NotImplementedError("legacy MoE recipe source is not bound")
    normalize = attributes["source_router_renormalize"]
    if type(normalize) is not bool:
        raise TypeError("source_router_renormalize must be a boolean")
    if attributes["routing_input_dtype"] not in ("fp32", "bf16"):
        raise NotImplementedError("legacy routing logits dtype is not bound")
    if (normalize and topk == 1 and ("source_router_state" not in attributes
            or attributes["source_router_state"] != "prepared_topk_weights_and_ids")):
        raise NotImplementedError("degenerate selected-weight row sum requires its own source binding")
    expert_map = attributes["source_has_expert_map"]
    if type(expert_map) is not bool:
        raise TypeError("source_has_expert_map must be a boolean")
    ep_size = require_positive_integer(attributes, "expert_parallel_size")
    if (ep_size > 1) != expert_map:
        raise ValueError("expert-map presence must match this source's EP program")
    chunk_size = require_positive_integer(attributes, "source_moe_chunk_size")
    if tokens > chunk_size:
        raise NotImplementedError("legacy chunked MoE requires one expert sequence per physical chunk")
    if experts not in SOURCE_TOPK_EXPERT_COUNTS:
        raise NotImplementedError("legacy source requires separate softmax/topk for this expert count")
    stages = (("moe_align_small_batch",) if tokens * topk < 1024 and experts <= 64
              else ("moe_align", "moe_count_and_sort"))
    sum_stage = "moe_sum" if topk in (2, 3, 4) else "moe_reduce"
    cast = ("logits_cast",) if attributes["routing_input_dtype"] == "bf16" else ()
    renormalization = ("normalize_sum", "normalize_div") if normalize else ()
    map_stages = ("expert_ids_long", "expert_map_lookup") if expert_map else ()
    return (*cast, "moe_topk", *renormalization, *stages, *map_stages,
            "w1", "moe_act_and_mul", "w2", sum_stage)


def select_legacy_chunked_moe_recipe(shape, dtype, attributes, hardware):
    """Complete functional0.11 BF16 source,with actual token intervals.

    Return ordered(role,token_begin,token_end,LaunchConfig) records. Router
    records cover the entire input once;expert records cover each source
    CHUNK_SIZE view. A final partial chunk reselects the H200 config;an
    exact multiple has no extra empty submission. Views/empty allocations
    and inplace output submit no copy or fill. cache13 aliases W1/W2 and
    is reused across chunks,not materialized again as payload traffic.

    Router prices (especially division),compiler resources and aggregate
    KernelSpec/consumer admission remain separate;this recipe does not
    invent a whole-operator launch or reinterpret0.11 as0.30.
    """
    from src.kernel.launch_config.nvidia_h200.moe_aux_launch import select_source_moe_aux_launch
    from src.kernel.launch_config.nvidia_h200.moe_router_launch import select_legacy_router_stage

    _require_legacy_source(attributes, hardware)
    require_dtype(dtype, ("bf16",))
    tokens, experts, topk, hidden, intermediate = require_shape(shape,
        ("tokens", "experts", "topk", "hidden", "intermediate"))
    if (attributes["source_component"] != "vllm_0_11_functional_unquantized_moe"
            or attributes["source_input_layout"] != "contiguous_aligned_16"
            or attributes["source_activation"] != "silu"
            or attributes["source_bias"] != "none"
            or attributes["source_apply_router_weight_on_input"] is not False):
        raise NotImplementedError("chunked MoE requires exact0.11 functional BF16 SiLU source without bias/input weight multiply")
    if type(attributes["source_inplace"]) is not bool:
        raise TypeError("chunked MoE requires the actual output-alias flag")
    if hidden % 8 or intermediate % 8:
        raise NotImplementedError("chunked MoE binds16-byte-aligned hidden/intermediate chunk views")
    chunk_size = require_positive_integer(attributes, "source_moe_chunk_size")
    first_shape = {**shape, "tokens": min(tokens, chunk_size)}
    # Reuse the existing single-chunk source validation and role decisions.
    first_roles = select_legacy_moe_recipe(first_shape, attributes, hardware)
    router_state = attributes["source_router_state"]
    if router_state not in ("softmax_logits_ready", "prepared_topk_weights_and_ids"):
        raise NotImplementedError("chunked MoE needs explicit router producer ownership")
    records = []
    def helper(role, local_shape, variant, helper_dtype, semantic, begin, end):
        launch = select_legacy_router_stage(local_shape, helper_dtype, variant,
            {**attributes, **semantic, "source_moe_router_stage": role}, hardware)
        records.append((role, begin, end, launch))
    router_roles = ("logits_cast", "moe_topk", "normalize_sum", "normalize_div")
    if router_state == "softmax_logits_ready":
        for role in first_roles:
            if role == "logits_cast":
                helper(role, {**shape, "element_count": tokens * experts}, "elementwise", "fp32",
                    {"operation": "cast_bf16_float32", "reads": 1, "writes": 1}, 0, tokens)
            elif role == "moe_topk":
                launch = select_source_moe_aux_launch(shape, "fp32", "moe_topk",
                    {**attributes, "routing_input_dtype": "fp32"}, hardware)
                records.append((role, 0, tokens, launch))
            elif role == "normalize_sum":
                helper(role, {**shape, "element_count": tokens}, "reduce", "fp32",
                    {"reduction_factor": topk}, 0, tokens)
            elif role == "normalize_div":
                helper(role, {**shape, "element_count": tokens * topk}, "elementwise", "fp32",
                    {"operation": "div", "reads": 2, "writes": 1}, 0, tokens)
    for begin in range(0, tokens, chunk_size):
        end = min(begin + chunk_size, tokens)
        chunk_shape = {**shape, "tokens": end - begin}
        roles = select_legacy_moe_recipe(chunk_shape, attributes, hardware)
        for role in roles:
            if role in router_roles:
                continue
            if role in ("expert_ids_long", "expert_map_lookup"):
                elements = legacy_expert_map_elements(chunk_shape, attributes)
                if role == "expert_ids_long":
                    semantic = {"operation": "cast_int32_int64", "reads": 1, "writes": 1,
                                "tensor_iterator_source": "pytorch_2_8_sm90"}
                    variant = "elementwise"
                else:
                    semantic = {"gather_factor": 1, "index_count": elements, "index_bytes": 8,
                                "scattered": True, "materialized_output": True}
                    variant = "gather"
                helper(role, {"element_count": elements}, variant, "int32", semantic, begin, end)
            elif role in ("w1", "w2"):
                records.append((role, begin, end,
                    select_legacy_moe_launch(chunk_shape, dtype, role, attributes, hardware)))
            else:
                records.append((role, begin, end,
                    select_legacy_moe_aux_launch(chunk_shape, dtype, role, attributes, hardware)))
    return tuple(records)


def legacy_expert_map_elements(shape, attributes):
    """Count the full allocated expert-ID tensor indexed by Python in0.11.

    Unlike executed EM, this allocation is never reduced by the small-A
    invocation optimization. Source: moe_align_block_size.py max_num_m_blocks.
    """
    if attributes["model_execution_version"] != "vllm_0_11_0":
        raise NotImplementedError("legacy expert-ID allocation binds only vLLM0.11")
    if (attributes["source_moe_config"] != "bundled_or_default"
            or attributes["routed_weight_dtype"] != "bf16"):
        raise NotImplementedError("legacy expert-ID allocation requires its bundled/default BF16 config")
    tokens, experts, topk, intermediate = require_shape(shape, ("tokens", "experts", "topk", "intermediate"))
    if topk > experts:
        raise ValueError("topk exceeds global experts")
    local = fused_moe_local_expert_count(experts, attributes)
    if local == 0:
        raise NotImplementedError("empty local expert map is not bound")
    bm = legacy_moe_config(tokens, local, intermediate)[0]
    return ceil_div(tokens * topk + experts * (bm - 1), bm)


def select_legacy_moe_launch(shape, dtype, variant, attributes, hardware):
    _require_legacy_source(attributes, hardware)
    require_dtype(dtype, ("bf16",))
    if (variant not in ("w1", "w2") or attributes["routed_weight_dtype"] != "bf16"
            or attributes["source_moe_config"] != "bundled_or_default"):
        raise NotImplementedError("legacy MoE source requires its unquantized BF16 program")
    if attributes["source_batch_invariant"] is not False:
        raise NotImplementedError("legacy batch-invariant MoE source is not bound")
    expert_map = attributes["source_has_expert_map"]
    if type(expert_map) is not bool:
        raise TypeError("source_has_expert_map must be a boolean")
    tokens, experts, topk, hidden, intermediate = require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    if topk > experts:
        raise ValueError("topk exceeds global experts")
    local = fused_moe_local_expert_count(experts, attributes)
    if local == 0:
        raise NotImplementedError("empty expert rank produces no grouped GEMM")
    if (attributes["expert_parallel_size"] > 1) != expert_map:
        raise ValueError("expert-map presence must match this source's EP program")
    bm, bn, bk, group, warps, stages = legacy_moe_config(tokens, local, intermediate)
    # The0.11 allocation uses global E without a routed<E cap. Only the
    # invocation's small A.size(0) condition reduces the executed EM.
    em = tokens * topk + experts * (bm - 1)
    input_rows = tokens if variant == "w1" else tokens * topk
    if input_rows < bm:
        em = min(em, tokens * topk * bm)
    width = 2 * intermediate if variant == "w1" else hidden
    parameters = {"group_size_m": float(group), "padded_rows": float(em),
        "mte2_tile_bytes": 0.0, "weight_tile_rows": 0.0,
        "operand_replay_a": 1.0, "operand_replay_b": 1.0,
        "num_warps": float(warps), "num_stages": float(stages)}
    if attributes["expert_parallel_size"] > 1:
        parameters["expert_parallel_size"] = float(attributes["expert_parallel_size"])
    return LaunchConfig(f"vllm_0_11_sm90_unquantized_{variant}",
        (ceil_div(em, bm) * ceil_div(width, bn), 1, 1), (32 * warps, 1, 1),
        tiling=TilingSpec(tile_shape={"m": bm, "n": bn, "k": bk}, pipeline_depth=stages),
        parameters=parameters, execution_variant=variant)


def select_legacy_moe_aux_launch(shape, dtype, variant, attributes, hardware):
    """Physical0.11 helper geometry; no0.30 scalar fill/vec8 sum substitution."""
    _require_legacy_source(attributes, hardware)
    require_dtype(dtype, ("bf16",))
    tokens, experts, topk = require_shape(shape, ("tokens", "experts", "topk"))
    if topk > experts:
        raise ValueError("topk exceeds global experts")
    if variant in ("moe_align", "moe_align_small_batch", "moe_count_and_sort"):
        if attributes["source_topk_id_dtype"] != "int32":
            raise NotImplementedError("legacy source alignment requires INT32 routed IDs")
        small = tokens * topk < 1024 and experts <= 64
        if ceil_div(experts, 32) * 32 >= 1024:
            raise ValueError("legacy alignment requires padded experts below1024")
        if (variant == "moe_align_small_batch") != small:
            raise ValueError("legacy alignment helper disagrees with its source branch")
        if small:
            grid, block = (1, 1, 1), (max(experts, 32), 1, 1)
        elif variant == "moe_align":
            grid, block = (1, 1, 1), (1024, 1, 1)
        else:
            grid, block = (min(ceil_div(tokens * topk, 256), 65535), 1, 1), (256, 1, 1)
    elif variant == "moe_act_and_mul":
        if attributes["source_input_layout"] != "contiguous_aligned_16":
            raise NotImplementedError("legacy activation source requires contiguous operands")
        intermediate, = require_shape(shape, ("intermediate",))
        grid, block = (tokens * topk, 1, 1), (min(intermediate, 1024), 1, 1)
    elif variant == "moe_sum":
        if (attributes["source_input_layout"] != "contiguous_aligned_16"
                or type(attributes["work_fraction"]) not in (int, float)
                or attributes["work_fraction"] != 1.0):
            raise NotImplementedError("legacy CUDA sum requires complete contiguous operands")
        hidden, = require_shape(shape, ("hidden",))
        if topk not in (2, 3, 4):
            raise ValueError("legacy source selects Torch reduction for this topk")
        grid, block = (tokens, 1, 1), (min(hidden, 1024), 1, 1)
    elif variant == "moe_reduce":
        return select_legacy_moe_reduce(shape, dtype, attributes, hardware)
    else:
        raise NotImplementedError("legacy auxiliary source variant is not bound")
    from src.kernel.launch_config.nvidia_h200.legacy_vector_resources import select_legacy_moe_aux_resources
    parameters = select_legacy_moe_aux_resources(shape, dtype, variant, attributes)
    return LaunchConfig(f"vllm_0_11_sm90_source_{variant}", grid, block,
                        parameters=parameters, execution_variant=variant)


def select_legacy_moe_reduce(shape, dtype, attributes, hardware):
    """Torch2.8 Reduce.cuh, contiguous [TOK,TOPK,H] with out[TOK,H]."""
    _require_legacy_source(attributes, hardware)
    require_dtype(dtype, ("bf16",))
    tokens, topk, hidden = require_shape(shape, ("tokens", "topk", "hidden"))
    if hidden == 1:
        raise NotImplementedError("legacy fastest-dimension reduction source is not bound")
    if topk in (2, 3, 4):
        raise ValueError("legacy source selects its specialized CUDA sum")
    if (attributes["source_input_layout"] != "contiguous_aligned_16"
            or type(attributes["work_fraction"]) not in (int, float)
            or attributes["work_fraction"] != 1.0):
        raise NotImplementedError("legacy Torch reduction requires its complete unsplit iterator")
    if (tokens * topk * hidden > 2**31 - 1
            or 1 + 2 * (tokens * topk * hidden - 1) > 2**31 - 1):
        raise NotImplementedError("legacy Torch reduction requires explicit32-bit sub-iterators")
    vector = 4
    extent = tokens * hidden if topk == 1 else hidden
    while extent % vector:
        vector //= 2
    outputs = tokens * hidden // vector
    maximum_threads = 512 // vector
    dim_x = min(1 << (outputs.bit_length() - 1), maximum_threads)
    dim_y = min(1 << (topk.bit_length() - 1), maximum_threads)
    x = min(dim_x, 32)
    y = min(dim_y, maximum_threads // x)
    x = min(dim_x, maximum_threads // y)
    reduce_y = topk >= min(y * 16, 256)
    grid_x = ceil_div(outputs, x if reduce_y else x * y)
    values_per_thread = ceil_div(topk, y) if reduce_y else topk
    if reduce_y and values_per_thread >= 256:
        target_grid = int(hardware.num_compute_units) * (int(hardware.max_threads_per_unit) // (x * y))
        if grid_x <= target_grid:
            ctas = max(min(ceil_div(target_grid, grid_x), ceil_div(values_per_thread, 16)),
                       ceil_div(values_per_thread, 256))
            if ctas > 1:
                raise NotImplementedError("legacy reduction global partial/semaphore stages are not bound")
    return LaunchConfig("pytorch_2_8_sm90_moe_sum_out", (grid_x, 1, 1), (x, y, 1),
                        execution_variant="moe_reduce")
