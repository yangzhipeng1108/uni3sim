"""Complete vLLM0.30 TileLang/DeepGEMM mHC calls on H200.

The unique DeepGEMM revision is owned by cmake/external_projects/deepgemm.cmake.
Its SM90 TF32 hc-prenorm host fixes BM64/BN32/BK64,128 math+128 TMA threads,
12 stages and reserves the whole SM90 per-CTA shared capacity. The reserve
deliberately exceeds the arrays' used bytes; it must not be replaced by the
staging-size calculation. No SM100 or SM80 resource/selection fact is used.

TileLang fallback is an actual source program, selected only when its normal
DeepGEMM support predicate is false. Broadcast forbids that fallback even
when VLLM_USE_DEEP_GEMM=False. Small post/pre bypasses prenorm dispatch and
uses the source's fused tile/split rule, including its32-token boundary.
All compiler registers/static and TileLang dynamic shared remain unbound.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_positive_integer, require_shape


def mhc_fused_source_config(tokens, hidden, hc_mult):
    """The source's unconditional host rule,not a hardware tuning model."""
    if tokens > 32 or hidden % 128:
        return None
    tile_n, splits = (2 if tokens < 16 else 6), 8
    while tile_n > 1 and hc_mult * (hc_mult + 2) % tile_n:
        tile_n -= 1
    while splits > 1 and hidden % (splits * 128):
        splits //= 2
    return tile_n, splits


def compute_mhc_prenorm_splits(tokens, input_size, sms):
    """compute_num_split(64,K,ceil(T/64)) with the actual queried SM count."""
    return max(min(sms // ceil_div(tokens, 64), ceil_div(input_size, 64) // 4), 1)


def _tilelang_prenorm(tokens, input_size):
    # HcPrenormGemmTileLangKernel.dispatch defaults tile12/512threads/split1.
    if tokens >= 1024:
        grid, threads, tile, block_m = (ceil_div(tokens, 2), 2, 1), 512, 12, 2
    elif tokens < 128 and input_size % 1024 == 0:
        grid, threads, tile, block_m = (tokens, 6, 1), 1024, 4, 1
    else:
        grid, threads, tile, block_m = (tokens, 2, 1), 512, 12, 1
    if input_size % threads:
        raise NotImplementedError("mHC TileLang prenorm requires complete source thread slices")
    return LaunchConfig(f"vllm_0_30_sm90_mhc_tilelang_prenorm_m{block_m}_n{tile}", grid, (threads, 1, 1),
        parameters={"source_n_splits": 1.0, "source_tile_n": float(tile), "source_block_m": float(block_m)})


def _prenorm(tokens, hidden, hc_mult, broadcast, attributes, hardware):
    k = hidden if broadcast else hc_mult * hidden
    deepgemm = broadcast
    if not broadcast:
        enabled = attributes["source_use_deep_gemm"]
        if type(enabled) is not bool:
            raise TypeError("mHC source_use_deep_gemm must be an actual boolean")
        if enabled:
            present = attributes["source_has_deep_gemm"]
            if type(present) is not bool:
                raise TypeError("mHC source_has_deep_gemm must be an actual runtime boolean")
            # H200 SM90 satisfies the source's platform support predicate.
            deepgemm = present
    if not deepgemm:
        return _tilelang_prenorm(tokens, k), 1
    if (attributes["source_has_deep_gemm"] is not True
            or attributes["source_deepgemm_prenorm_available"] is not True
            or attributes["source_deepgemm_snapshot"] != "vllm_0_30_cmake_pin"):
        raise NotImplementedError("mHC selected DeepGEMM requires the actual pinned prenorm symbol;no fallback")
    sms = require_positive_integer(attributes, "source_split_device0_sms")
    if sms > hardware.num_compute_units:
        raise ValueError("mHC queried device0 SM count exceeds H200 HWSpec")
    splits = compute_mhc_prenorm_splits(tokens, k, sms)
    # SM90ArchSpec::smem_capacity is227KiB in the pinned source. The host
    # overrides its computed205KB staging size with this full reserve.
    shared_capacity = 227 * 1024
    block_m, block_n, block_k, stages = 64, 32, 64, 12
    used_shared = stages * (block_m * block_k * 2 + block_n * block_k * 4) + block_m * 128 + stages * 16
    if used_shared > shared_capacity:
        raise AssertionError("pinned mHC SM90 staging must fit its source capacity")
    launch = LaunchConfig("vllm_0_30_sm90_deepgemm_tf32_hc_prenorm", (splits * ceil_div(tokens, 64), 1, 1),
        (256, 1, 1), parameters={"dynamic_shared_bytes": float(shared_capacity),
            "source_n_splits": float(splits), "source_shape_n": 24.0, "source_shape_k": float(k),
            "source_block_m": 64.0, "source_block_n": 32.0, "source_block_k": 64.0,
            "source_num_stages": 12.0, "source_swizzle_cd_bytes": 128.0})
    return launch, splits


def select_mhc_source_recipe(shape, dtype, variant, attributes, hardware):
    """Return every submission for current V4 mHC,after prepared inputs.

    Supported roles:pre_broadcast,pre,post_pre,post,head. BF16 gamma is
    already contiguous when present; precomputed broadcast weights are
    caller-owned. Float gamma casts,delayed-pre/aux and other backends are
    explicit gaps rather than extra inferred kernels or missing stages.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("mHC source requires H200 HWSpec")
    require_dtype(dtype, ("bf16",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_mhc_tilelang"
            or attributes["source_input_layout"] != "contiguous_bf16_and_fp32_weights_coefficients"
            or attributes["source_mhc_api"] != "current"):
        raise NotImplementedError("mHC requires its exact0.30 current TileLang source and prepared layouts")
    if variant not in ("pre_broadcast", "pre", "post_pre", "post", "head"):
        raise NotImplementedError("mHC source operation is not bound")
    tokens, hidden, hc_mult = require_shape(shape, ("rows", "hidden", "hc_mult"))
    if hc_mult != 4 or hidden % 128:
        raise NotImplementedError("mHC source binds V4's four streams and128-aligned hidden width")
    if max(tokens, hidden, tokens * hidden * hc_mult) >= 2**31:
        raise NotImplementedError("mHC source requires bounded signed-int tensor extents")
    if variant in ("post", "head"):
        return ((variant, LaunchConfig("vllm_0_30_sm90_mhc_tilelang_" + variant,
            (tokens, 1, 1), (128, 1, 1))),)
    iters = require_positive_integer(attributes, "sinkhorn_iters")
    weight = attributes["source_norm_weight"]
    if weight not in ("none", "contiguous_bf16"):
        raise NotImplementedError("mHC source gamma conversion/materialization requires its own program")
    if variant == "pre_broadcast" and (weight != "contiguous_bf16"
            or attributes["source_broadcast_weight"] != "precomputed_contiguous_fp32"):
        raise NotImplementedError("broadcast mHC requires actual fused norm and precomputed summed projection")
    config = mhc_fused_source_config(tokens, hidden, hc_mult) if variant == "post_pre" else None
    if config is not None:
        tile_n, splits = config
        main = LaunchConfig("vllm_0_30_sm90_mhc_tilelang_fused_post_prenorm",
            (tokens, 24 // tile_n, splits), (128, 1, 1), parameters={
                "source_tile_n": float(tile_n), "source_n_splits": float(splits)})
        stages = (("fused_post_prenorm", main),)
    else:
        main, splits = _prenorm(tokens, hidden, hc_mult, variant == "pre_broadcast", attributes, hardware)
        post = (("post", LaunchConfig("vllm_0_30_sm90_mhc_tilelang_post", (tokens, 1, 1), (128, 1, 1))),) if variant == "post_pre" else ()
        stages = (*post, ("prenorm", main))
    epilogue = "broadcast_norm" if variant == "pre_broadcast" else "pre_norm" if weight == "contiguous_bf16" else "pre_reduce"
    finish = LaunchConfig("vllm_0_30_sm90_mhc_tilelang_" + epilogue, (tokens, 1, 1), (96, 1, 1),
        parameters={"source_n_splits": float(splits), "source_sinkhorn_iters": float(iters)})
    return (*stages, (epilogue, finish))
