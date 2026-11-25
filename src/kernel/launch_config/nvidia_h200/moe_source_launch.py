"""vLLM 0.30 unquantized Triton expert host dispatch on H200.

Static semantics: fused_moe.py::try_get_optimal_moe_config,
get_default_config, _prepare_expert_assignment and invoke_fused_moe_triton_kernel.
Only source configuration and tensor extents enter selection. A versioned
prepared BF16 EP resource class is modeled where its source branch matches;
other compiler register/shared-memory resources remain unbound.
"""

from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.nvidia_h200.moe_source_configs import H200_BF16_CONFIGS, H200_FP8_BLOCK_CONFIGS
from src.kernel.launch_config.nvidia_h200.moe_aux_launch import (
    SOURCE_TOPK_EXPERT_COUNTS, source_alignment_stages, select_source_moe_aux_launch)
from src.kernel.launch_config.selection_primitives import (
    ceil_div, fused_moe_local_expert_count, require_dtype, require_shape,
)


# fused_moe_kernel first loads num_tokens_post_padded, then after its active
# block guard loads expert_ids and (on the aligned path) sorted_token_ids.
# Those second-level reads are independent; operand addresses consume them.
SOURCE_MOE_HEADER_CACHE_READ_LEVELS = 2


def source_moe_config(tokens, local_experts, intermediate):
    """Resolve source's nearest bundled batch key or its CUDA default branch."""
    if (local_experts, intermediate) in H200_BF16_CONFIGS:
        candidates = H200_BF16_CONFIGS[local_experts, intermediate]
        selected = min(candidates, key=lambda row: abs(row[0] - tokens))
        return selected[1:]
    block_m = 16 if tokens <= 32 else 32 if tokens <= 96 else 64 if tokens <= 512 else 128
    block_n = 64 if tokens <= 64 else 128
    block_k = 128 if tokens <= 64 else 64
    group_m = 16 if tokens // local_experts > 128 else 1
    warps = 4 if tokens <= 128 else 8
    stages = 4 if tokens <= 32 else 3
    return block_m, block_n, block_k, group_m, warps, stages


def observed_operand_shared_work(dtype, variant, assignment_mode, tiling, group_m,
                                 threads, shared_bytes, registers_per_thread,
                                 static_shared_bytes):
    """Compiled one-panel HMMA shared work; no duration or service rate.

    Tuples hold per-K read/write, prologue write, full-tile output read bound
    and output write bytes per CTA. Evidence: triton_operand_shared_work_sm90.json.
    """
    if assignment_mode not in ("naive", "aligned"):
        raise ValueError("shared work requires a known source assignment mode")
    if dtype != "bf16" or tiling.pipeline_depth != 1:
        return None
    observed = {
        ('w1', 'aligned', 16, 128, 128, 1, 128, 36864, 97, 0): (49152, 36864, 36864, 4096, 4096),
        ('w1', 'aligned', 16, 256, 128, 1, 256, 69632, 98, 0): (98304, 69632, 69632, 8192, 8192),
        ('w1', 'aligned', 16, 64, 128, 16, 128, 20480, 63, 0): (32768, 20480, 20480, 2048, 2048),
        ('w1', 'aligned', 16, 64, 256, 1, 128, 40960, 96, 0): (65536, 40960, 40960, 2048, 2048),
        ('w1', 'naive', 16, 64, 128, 16, 128, 20480, 64, 0): (32768, 20480, 20480, 2048, 2048),
        ('w2', 'aligned', 16, 128, 128, 1, 128, 36864, 99, 0): (49152, 36864, 36864, 4096, 4096),
        ('w2', 'aligned', 16, 256, 128, 1, 256, 69632, 100, 0): (98304, 69632, 69632, 8192, 8192),
        ('w2', 'aligned', 16, 64, 128, 16, 128, 20480, 64, 0): (32768, 20480, 20480, 2048, 2048),
        ('w2', 'aligned', 16, 64, 256, 1, 128, 40960, 94, 0): (65536, 40960, 40960, 2048, 2048),
    }
    tile = tiling.tile_shape
    key = (variant, assignment_mode, tile["m"], tile["n"], tile["k"], group_m,
           threads, shared_bytes, registers_per_thread, static_shared_bytes)
    return observed[key] if key in observed else None


def observed_matrix_preparation_issue(dtype, variant, assignment_mode, tiling, group_m,
                                      threads, shared_bytes, registers_per_thread,
                                      static_shared_bytes):
    """Unpredicated post-wait issue before first HGMMA per physical K tile.

    Evidence: triton_matrix_preparation_issue_sm90.json. No latency or rate.
    """
    if assignment_mode not in ("naive", "aligned"):
        raise ValueError("matrix preparation requires a known assignment mode")
    if dtype != "bf16":
        return None
    observed = {
        ('w1', 'aligned', 128, 128, 64, 1, 256, 98304, 116, 0): (43, 2),
        ('w1', 'aligned', 128, 128, 64, 16, 256, 98304, 116, 0): (43, 2),
        ('w1', 'aligned', 128, 256, 64, 16, 256, 196608, 210, 0): (47, 2),
        ('w1', 'aligned', 128, 256, 64, 32, 256, 196608, 210, 0): (47, 2),
        ('w1', 'aligned', 64, 128, 128, 1, 256, 196608, 80, 0): (31, 2),
        ('w1', 'aligned', 64, 128, 64, 1, 128, 73728, 120, 0): (29, 2),
        ('w1', 'aligned', 64, 128, 64, 1, 256, 73728, 64, 0): (25, 2),
        ('w2', 'aligned', 128, 128, 64, 1, 256, 98304, 116, 0): (43, 2),
        ('w2', 'aligned', 128, 128, 64, 16, 256, 98304, 116, 0): (43, 2),
        ('w2', 'aligned', 128, 256, 64, 1, 256, 196608, 214, 0): (47, 2),
        ('w2', 'aligned', 128, 256, 64, 16, 256, 196608, 214, 0): (47, 2),
        ('w2', 'aligned', 128, 256, 64, 32, 256, 196608, 214, 0): (47, 2),
        ('w2', 'aligned', 64, 128, 128, 1, 256, 196608, 94, 0): (41, 2),
        ('w2', 'aligned', 64, 128, 64, 1, 128, 73728, 122, 0): (25, 2),
        ('w2', 'aligned', 64, 128, 64, 1, 256, 73728, 64, 0): (26, 2),
    }
    tile = tiling.tile_shape
    if (tiling.pipeline_depth is None
            or shared_bytes != tiling.pipeline_depth * 2 * (tile["m"]+tile["n"]) * tile["k"]):
        return None
    key = (variant, assignment_mode, tile["m"], tile["n"], tile["k"], group_m,
           threads, shared_bytes, registers_per_thread, static_shared_bytes)
    return observed[key] if key in observed else None


def observed_startup_entry_issue(dtype, variant, assignment_mode, tiling, group_m, threads,
                                  shared_bytes, registers_per_thread,
                                  static_shared_bytes):
    """Conservative compiled entry issue before first LDGSTS, never time.

    Complete program-mode/physical signatures must bind one unique count.
    Predicated instructions are omitted; verified straight-line callees count on every call edge.
    Evidence: references/triton_startup_caller_issue_sm90.json.
    """
    if assignment_mode not in ("naive", "aligned"):
        raise ValueError("startup issue requires a known source assignment mode")
    if dtype != "bf16":
        return None
    observed = {
        ('w1', 'aligned', 16, 64, 64, 1, 128, 40960, 54, 0): 255,
        ('w1', 'aligned', 16, 64, 128, 1, 128, 40960, 64, 0): 352,
        ('w1', 'aligned', 16, 64, 128, 1, 128, 61440, 64, 0): 346,
        ('w1', 'aligned', 16, 64, 128, 16, 128, 20480, 63, 0): 351,
        ('w1', 'aligned', 16, 64, 256, 1, 128, 40960, 96, 0): 493,
        ('w1', 'aligned', 16, 64, 256, 16, 128, 81920, 106, 0): 541,
        ('w1', 'aligned', 16, 128, 64, 1, 128, 55296, 75, 0): 356,
        ('w1', 'aligned', 16, 128, 128, 1, 128, 36864, 97, 0): 561,
        ('w1', 'aligned', 16, 128, 128, 1, 128, 147456, 98, 0): 536,
        ('w1', 'aligned', 16, 128, 128, 1, 256, 73728, 64, 0): 363,
        ('w1', 'aligned', 16, 128, 128, 16, 256, 73728, 64, 0): 363,
        ('w1', 'aligned', 16, 128, 128, 64, 128, 73728, 98, 0): 529,
        ('w1', 'aligned', 16, 128, 128, 64, 256, 73728, 64, 0): 365,
        ('w1', 'aligned', 16, 256, 128, 1, 256, 69632, 98, 0): 552,
        ('w1', 'aligned', 32, 64, 128, 1, 128, 49152, 71, 0): 360,
        ('w1', 'aligned', 32, 64, 128, 1, 128, 98304, 72, 0): 450,
        ('w1', 'aligned', 32, 128, 64, 1, 128, 40960, 99, 0): 360,
        ('w1', 'aligned', 32, 128, 128, 64, 128, 81920, 128, 0): 552,
        ('w1', 'aligned', 64, 128, 64, 1, 128, 73728, 120, 0): 376,
        ('w1', 'aligned', 64, 128, 64, 1, 256, 73728, 64, 0): 303,
        ('w1', 'aligned', 64, 128, 128, 1, 256, 196608, 80, 0): 344,
        ('w1', 'aligned', 128, 128, 64, 1, 256, 98304, 116, 0): 312,
        ('w1', 'aligned', 128, 128, 64, 16, 256, 98304, 116, 0): 313,
        ('w1', 'aligned', 128, 256, 64, 16, 256, 196608, 210, 0): 419,
        ('w1', 'aligned', 128, 256, 64, 32, 256, 196608, 210, 0): 419,
        ('w1', 'naive', 16, 32, 64, 1, 128, 18432, 39, 0): 195,
        ('w1', 'naive', 16, 64, 64, 1, 128, 30720, 54, 0): 243,
        ('w1', 'naive', 16, 64, 64, 1, 128, 40960, 54, 0): 252,
        ('w1', 'naive', 16, 64, 128, 1, 128, 61440, 64, 0): 357,
        ('w1', 'naive', 16, 64, 128, 16, 128, 20480, 64, 0): 354,
        ('w1', 'naive', 16, 128, 64, 1, 128, 36864, 78, 0): 362,
        ('w1', 'naive', 16, 128, 64, 1, 128, 55296, 78, 0): 340,
        ('w1', 'naive', 16, 128, 128, 16, 128, 147456, 95, 0): 533,
        ('w2', 'aligned', 16, 64, 64, 1, 128, 40960, 56, 0): 263,
        ('w2', 'aligned', 16, 64, 128, 1, 128, 40960, 72, 0): 364,
        ('w2', 'aligned', 16, 64, 128, 1, 128, 61440, 72, 0): 361,
        ('w2', 'aligned', 16, 64, 128, 16, 128, 20480, 64, 0): 361,
        ('w2', 'aligned', 16, 64, 256, 1, 128, 40960, 94, 0): 534,
        ('w2', 'aligned', 16, 64, 256, 16, 128, 81920, 108, 0): 557,
        ('w2', 'aligned', 16, 128, 64, 1, 128, 55296, 77, 0): 363,
        ('w2', 'aligned', 16, 128, 128, 1, 128, 36864, 99, 0): 567,
        ('w2', 'aligned', 16, 128, 128, 1, 128, 147456, 100, 0): 582,
        ('w2', 'aligned', 16, 128, 128, 1, 256, 73728, 64, 0): 370,
        ('w2', 'aligned', 16, 128, 128, 16, 256, 73728, 64, 0): 371,
        ('w2', 'aligned', 16, 128, 128, 64, 128, 73728, 100, 0): 552,
        ('w2', 'aligned', 16, 128, 128, 64, 256, 73728, 64, 0): 371,
        ('w2', 'aligned', 16, 256, 128, 1, 256, 69632, 100, 0): 557,
        ('w2', 'aligned', 32, 64, 128, 1, 128, 49152, 80, 0): 372,
        ('w2', 'aligned', 32, 64, 128, 1, 128, 98304, 80, 0): 472,
        ('w2', 'aligned', 32, 128, 64, 1, 128, 40960, 102, 0): 372,
        ('w2', 'aligned', 32, 128, 128, 64, 128, 81920, 135, 0): 565,
        ('w2', 'aligned', 64, 128, 64, 1, 128, 73728, 122, 0): 395,
        ('w2', 'aligned', 64, 128, 64, 1, 256, 73728, 64, 0): 294,
        ('w2', 'aligned', 64, 128, 128, 1, 256, 196608, 94, 0): 358,
        ('w2', 'aligned', 128, 128, 64, 1, 256, 98304, 116, 0): 326,
        ('w2', 'aligned', 128, 128, 64, 16, 256, 98304, 116, 0): 327,
        ('w2', 'aligned', 128, 256, 64, 1, 256, 196608, 214, 0): 414,
        ('w2', 'aligned', 128, 256, 64, 16, 256, 196608, 214, 0): 415,
        ('w2', 'aligned', 128, 256, 64, 32, 256, 196608, 214, 0): 415,
        ('w2', 'naive', 16, 32, 64, 1, 128, 18432, 39, 0): 198,
        ('w2', 'naive', 16, 64, 64, 1, 128, 30720, 54, 0): 245,
        ('w2', 'naive', 16, 64, 64, 1, 128, 40960, 54, 0): 254,
        ('w2', 'naive', 16, 64, 128, 1, 128, 61440, 64, 0): 360,
        ('w2', 'naive', 16, 128, 128, 16, 128, 147456, 96, 0): 571,

    }
    tile = tiling.tile_shape
    signature = (variant, assignment_mode, tile["m"], tile["n"], tile["k"], group_m, threads,
        shared_bytes, registers_per_thread, static_shared_bytes)
    return observed[signature] if signature in observed else None


def observed_operand_pipeline_depth(shape, dtype, tiling, group_m, threads, shared_bytes,
                                    *, registers_per_thread=None, static_shared_bytes=None):
    """Bind a static instruction ring or matching bundled source stages."""
    if dtype != "bf16":
        return None
    # Compiled HMMA rings differ from newer bundled stage directives.
    # Discrete evidence: references/triton_bf16_bm{16,32}_operand_ring_sm90.json.
    observed_rings = {
        (16, 32, 64, 1, 128, 18432, 39, 0): 3,
        (16, 64, 128, 1, 128, 61440, 64, 0): 3,
        (16, 64, 64, 1, 128, 40960, 54, 0): 4,
        (16, 64, 64, 1, 128, 30720, 54, 0): 3,
        (16, 128, 64, 1, 128, 55296, 78, 0): 3,
        (16, 128, 64, 1, 128, 55296, 77, 0): 3,
        (16, 128, 64, 1, 128, 36864, 78, 0): 2,
        (16, 128, 64, 1, 128, 36864, 77, 0): 2,
        (16, 64, 128, 16, 128, 20480, 63, 0): 1,
        (16, 64, 128, 16, 128, 20480, 64, 0): 1,
        (16, 64, 128, 1, 128, 61440, 72, 0): 3,
        (16, 128, 128, 1, 128, 36864, 97, 0): 1,
        (16, 128, 128, 1, 128, 36864, 99, 0): 1,
        (16, 256, 128, 1, 256, 69632, 98, 0): 1,
        (16, 256, 128, 1, 256, 69632, 100, 0): 1,
        (16, 64, 256, 1, 128, 40960, 96, 0): 1,
        (16, 64, 256, 1, 128, 40960, 94, 0): 1,
        (16, 64, 128, 1, 128, 40960, 64, 0): 2,
        (16, 64, 128, 1, 128, 40960, 72, 0): 2,
        (16, 128, 128, 64, 256, 73728, 64, 0): 2,
        (32, 256, 128, 16, 128, 73728, 233, 0): 1,
        (32, 256, 128, 16, 128, 73728, 237, 0): 1,
        (32, 64, 128, 1, 128, 49152, 71, 0): 2,
        (32, 64, 128, 1, 128, 49152, 80, 0): 2,
        (16, 64, 64, 1, 128, 40960, 56, 0): 4,
        (16, 64, 256, 16, 128, 81920, 106, 0): 2,
        (16, 64, 256, 16, 128, 81920, 108, 0): 2,
        (16, 128, 64, 1, 128, 55296, 75, 0): 3,
        (16, 128, 128, 1, 128, 147456, 98, 0): 4,
        (16, 128, 128, 1, 128, 147456, 100, 0): 4,
        (16, 128, 128, 1, 256, 73728, 64, 0): 2,
        (16, 128, 128, 16, 128, 147456, 95, 0): 4,
        (16, 128, 128, 16, 128, 147456, 96, 0): 4,
        (16, 128, 128, 16, 256, 73728, 64, 0): 2,
        (16, 128, 128, 64, 128, 73728, 98, 0): 2,
        (16, 128, 128, 64, 128, 73728, 100, 0): 2,
        (32, 64, 128, 1, 128, 98304, 72, 0): 4,
        (32, 64, 128, 1, 128, 98304, 80, 0): 4,
        (32, 128, 64, 1, 128, 40960, 99, 0): 2,
        (32, 128, 64, 1, 128, 40960, 102, 0): 2,
        (32, 128, 128, 64, 128, 81920, 128, 0): 2,
        (32, 128, 128, 64, 128, 81920, 135, 0): 2,
    }
    signature = (*tuple(tiling.tile_shape[name] for name in ("m", "n", "k")),
        group_m, threads, shared_bytes, registers_per_thread, static_shared_bytes)
    if signature in observed_rings:
        return observed_rings[signature]
    bm, bn, bk, source_group, warps, stages = source_moe_config(
        shape["tokens"], shape["experts"], shape["intermediate"])
    if (tuple(tiling.tile_shape[name] for name in ("m", "n", "k")) != (bm, bn, bk)
            or group_m != source_group or threads != warps * 32
            or shared_bytes != stages * (bm + bn) * bk * 2):
        return None
    return stages


def observed_output_relayout_bytes(dtype, tiling, threads, shared_bytes,
                                   registers_per_thread, static_shared_bytes):
    """Return full-tile shared write+read bytes for compiled SS output layout.

    Static evidence: triton_bf16_output_shared_relayout_sm90.json. Predicate
    masks apply to global stores after this complete shared redistribution.
    """
    tm, tn, tk = (tiling.tile_shape[name] for name in ("m", "n", "k"))
    signature = (tm, tn, tk, threads, shared_bytes, registers_per_thread, static_shared_bytes)
    supported = {
        (128, 128, 64, 256, 98304, 116, 0),
        (128, 256, 64, 256, 196608, 210, 0),
        (128, 256, 64, 256, 196608, 214, 0),
    }
    if dtype != "bf16" or signature not in supported:
        return None
    return 2 * tm * tn * 2


def source_output_conversion_warp_instructions(dtype, tiling):
    """Return the packed BF16 output-conversion issue lower bound per CTA."""
    if dtype != "bf16":
        return None
    tm, tn, _ = (tiling.tile_shape[name] for name in ("m", "n", "k"))
    if tm * tn % 64:
        raise ValueError("BF16 output tile does not fill packed warp instructions")
    return tm * tn // 64


def observed_output_conversion_warp_instructions(dtype, tiling, threads, shared_bytes,
                                                 registers_per_thread, static_shared_bytes):
    """Return packed FP32-to-BF16 warp instructions before shared relayout."""
    if observed_output_relayout_bytes(dtype, tiling, threads, shared_bytes,
                                      registers_per_thread, static_shared_bytes) is None:
        return None
    return source_output_conversion_warp_instructions(dtype, tiling)


def select_source_moe_recipe(shape, attributes, hardware):
    """Full ordinary softmax/unquantized source sequence, including routing.

    The historical shape-only recipe ABI cannot identify runtime or expert
    maps. Consumers provide these source facts here; no runtime is inferred
    from the hardware name or an observed duration.
    """
    if attributes["model_execution_version"] == "vllm_0_11_0":
        from src.kernel.launch_config.nvidia_h200.moe_legacy_launch import select_legacy_moe_recipe
        return select_legacy_moe_recipe(shape, attributes, hardware)
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("SM90 expert recipe requires H200 HWSpec")
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_routing"] != "softmax"
            or attributes["source_moe_config"] != "bundled_or_default"
            or attributes["routed_weight_dtype"] != "bf16"
            or attributes["source_batch_invariant"] is not False):
        raise NotImplementedError("source MoE recipe is not bound")
    tokens, experts, topk = require_shape(shape, ("tokens", "experts", "topk"))
    if topk > experts:
        raise ValueError("topk exceeds global experts")
    if experts not in SOURCE_TOPK_EXPERT_COUNTS:
        raise NotImplementedError("source MoE recipe requires separate softmax/topk for this expert count")
    stages = source_alignment_stages(tokens, topk, experts, attributes["source_has_expert_map"])
    # fused_experts_impl zeroes the full routed W2 output before invoking
    # W2 when an expert map is present. Preserve that real producer; the
    # consumer must bind its actual runtime-specific ATen Fill source.
    output_zero = ("moe_zero_output",) if attributes["source_has_expert_map"] else ()
    return ("moe_topk", *stages, "w1", "moe_act_and_mul", *output_zero, "w2", "moe_sum")


def select_source_expert_launch_recipe(shape, dtype, attributes, hardware):
    """Prepared versioned BF16 experts; probabilities/IDs already exist.

    The caller declares the actual Triton bundled/default backend and its
    batch-invariant flag. The model checkpoint alone cannot choose them.
    Reuse host selectors without adding another routing submission.
    """
    require_dtype(dtype, ("bf16",))
    if attributes["model_execution_version"] == "vllm_0_11_0":
        required = {"source_component", "source_router_state", "source_inplace", "source_moe_chunk_size",
            "source_activation", "source_bias", "source_apply_router_weight_on_input", "source_moe_config",
            "source_batch_invariant", "source_has_expert_map", "expert_parallel_size", "source_input_layout",
            "routing_weight_dtype", "source_topk_id_dtype", "source_routing", "routing_input_dtype",
            "source_router_renormalize", "routed_weight_dtype", "work_fraction"}
        if (not required.issubset(attributes) or attributes["routing_weight_dtype"] != "fp32"
                or attributes["source_topk_id_dtype"] != "int32"):
            raise NotImplementedError("prepared0.11 experts require their complete functional BF16 source facts")
        if attributes["source_router_state"] != "prepared_topk_weights_and_ids":
            raise ValueError("prepared0.11 expert sidecar cannot repeat its external router")
        from src.kernel.launch_config.nvidia_h200.moe_legacy_launch import select_legacy_chunked_moe_recipe
        records = select_legacy_chunked_moe_recipe(shape, dtype, attributes, hardware)
        # Source intervals are physical launch facts. Encode them in unique
        # ordered roles without changing the aggregate mathematical program or
        # inventing a per-chunk histogram, which these host rules do not read.
        return tuple((f"chunk_{begin}_{end}_{role}", launch) for role, begin, end, launch in records)
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or type(attributes["expert_parallel_size"]) is not int
            or attributes["expert_parallel_size"] < 1
            or type(attributes["source_has_expert_map"]) is not bool
            or attributes["source_has_expert_map"] != (attributes["expert_parallel_size"] > 1)
            or attributes["source_input_layout"] != "contiguous_aligned_16"
            or attributes["routing_weight_dtype"] != "fp32"
            or attributes["source_topk_id_dtype"] != "int32"):
        raise NotImplementedError("prepared expert source requires exact0.30 BF16 expert-map operands")
    recipe = select_source_moe_recipe(shape, attributes, hardware)
    if recipe[0] != "moe_topk" or recipe.count("moe_topk") != 1:
        raise ValueError("prepared experts must remove exactly one external route producer")
    launches = []
    for stage in recipe[1:]:
        if stage in ("w1", "w2"):
            launch = select_source_moe_launch(shape, dtype, stage, attributes, hardware)
        elif stage == "moe_zero_output":
            if ("tensor_iterator_source" not in attributes
                    or attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"):
                raise NotImplementedError("prepared expert output zero requires explicit Torch2.13 Fill source")
            from src.kernel.launch_config.select_launch_config import select_launch_config
            launch = select_launch_config("nvidia_h200", "vector",
                {"element_count": shape["tokens"] * shape["topk"] * shape["hidden"]}, dtype,
                "elementwise", {**attributes, "operation": "fill", "reads": 0, "writes": 1}, hardware)
        else:
            stage_attrs = ({**attributes, "fill_role": "num_tokens_post_padded"}
                if stage == "moe_fill" else attributes)
            launch = select_source_moe_aux_launch(shape, dtype, stage, stage_attrs, hardware)
        launches.append((stage, launch))
    return tuple(launches)


def source_fp8_moe_config(tokens, local_experts, intermediate):
    """Pinned Hopper block-FP8 config; no per-tensor quantization projection."""
    if (local_experts, intermediate) in H200_FP8_BLOCK_CONFIGS:
        candidates = H200_FP8_BLOCK_CONFIGS[local_experts, intermediate]
        selected = min(candidates, key=lambda row: abs(row[0] - tokens))
        bm, bn, bk, group_m, warps, stages = selected[1:]
        return bm, bn, min(bk, 128), group_m, warps, stages
    return (16 if tokens <= 64 else 64,
            64 if tokens <= 8 else 128, 128,
            1 if tokens <= 16 else 32, 4,
            4 if tokens <= 4 else 3)


def select_source_moe_launch(shape, dtype, variant, attributes, hardware):
    if attributes["model_execution_version"] == "vllm_0_11_0":
        from src.kernel.launch_config.nvidia_h200.moe_legacy_launch import select_legacy_moe_launch
        return select_legacy_moe_launch(shape, dtype, variant, attributes, hardware)
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("SM90 expert source requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16"))
    if (variant not in ("w1", "w2")
            or attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_moe_config"] != "bundled_or_default"):
        raise NotImplementedError("H200 expert source is not bound")
    weight_dtype = attributes["routed_weight_dtype"]
    if weight_dtype not in (dtype, "fp8"):
        raise NotImplementedError("H200 expert weight source is not bound")
    if weight_dtype == "fp8" and tuple(attributes["source_weight_block_shape"]) != (128, 128):
        raise NotImplementedError("source FP8 expert requires explicit128x128 weight scale blocks")
    if attributes["source_batch_invariant"] is not False:
        raise NotImplementedError("batch-invariant expert source is not bound")
    expert_map = attributes["source_has_expert_map"]
    if type(expert_map) is not bool:
        raise TypeError("source_has_expert_map must be a boolean")
    tokens, experts, topk, hidden, intermediate = require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    if topk > experts:
        raise ValueError("topk cannot exceed global expert count")
    local_experts = fused_moe_local_expert_count(experts, attributes)
    if local_experts == 0:
        raise NotImplementedError("empty expert rank produces no grouped GEMM")
    if (attributes["expert_parallel_size"] > 1) != expert_map:
        raise ValueError("expert-map presence must match this source's EP program")
    config_selector = source_fp8_moe_config if weight_dtype == "fp8" else source_moe_config
    bm, bn, bk, group_m, warps, stages = config_selector(tokens, local_experts, intermediate)
    routed = tokens * topk
    naive = not expert_map and routed * 4 <= experts
    if naive:
        capacity = routed * bm
    else:
        capacity = routed + experts * (bm - 1)
        if routed < experts:
            capacity = min(capacity, routed * bm)
        input_rows = tokens if variant == "w1" else routed
        if input_rows < bm:
            capacity = min(capacity, routed * bm)
    output_width = 2 * intermediate if variant == "w1" else hidden
    parameters = {
        "group_size_m": float(group_m), "padded_rows": float(capacity),
        "mte2_tile_bytes": 0.0, "weight_tile_rows": 0.0,
        "operand_replay_a": 1.0, "operand_replay_b": 1.0,
        "num_warps": float(warps), "num_stages": float(stages),
    }
    if weight_dtype == "fp8":
        family = hardware.grouped_gemm_fp8_block_tile_pipelines
        program = f"bm{bm}_bn{bn}_bk{bk}_w{warps}_s{stages}"
        # Mirror the predictor's resolution: prefer the native v0.30 row
        # when the family carries it; only fall back to the measured SGLang
        # BN64 row when the native program has no mechanism. The launch's
        # dynamic shared-memory budget must come from the same row the
        # predictor prices, otherwise a native-only family would validate
        # against one row and execute the other.
        if family is not None and program in family:
            selected = program
        elif bn == 64:
            selected = f"sgl_{program}"
        else:
            selected = program
        if family is not None and selected in family:
            parameters["dynamic_shared_memory_bytes"] = float(
                family[selected]["dynamic_shared_memory_bytes_per_cta"])
    if attributes["expert_parallel_size"] > 1:
        parameters["expert_parallel_size"] = float(attributes["expert_parallel_size"])
    footprint = hardware.prepared_moe_operand_resources
    if footprint is not None and weight_dtype == dtype == "bf16" and expert_map:
        domain = footprint["domain"]
        if (domain["minimum_tokens"] <= tokens <= domain["maximum_tokens"]
                and (experts, local_experts, topk, hidden, intermediate, bm, bn, bk, group_m, 32 * warps,
                    attributes["expert_parallel_size"]) == tuple(domain[name] for name in (
                        "experts", "local_experts", "topk", "hidden", "intermediate", "tile_m", "tile_n", "tile_k",
                        "group_size_m", "threads_per_cta", "expert_parallel_size"))):
            resource = footprint["resources"][variant]
            parameters.update(registers_per_thread=float(resource["registers_per_thread"]),
                static_shared_bytes=float(resource["static_shared_bytes"]),
                dynamic_shared_memory_bytes=float(resource["dynamic_shared_bytes"]))
            # Keep source num_stages in parameters; physical shared slots
            # need not equal the compiler's source configuration argument.
            stages = resource["operand_ring_slots"]
    program = ("fp8_block128_swap_ab" if bm < 64 and bn >= 64 else "fp8_block128") if weight_dtype == "fp8" else "unquantized"
    return LaunchConfig(
        algorithm=f"vllm_0_30_sm90_{program}_{'naive' if naive else 'aligned'}_{variant}",
        grid=(ceil_div(capacity, bm) * ceil_div(output_width, bn), 1, 1),
        block=(32 * warps, 1, 1),
        tiling=TilingSpec(tile_shape={"m": bm, "n": bn, "k": bk}, pipeline_depth=stages),
        parameters=parameters, execution_variant=variant)
