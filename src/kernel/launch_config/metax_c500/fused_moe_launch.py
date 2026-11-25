"""MetaX C500 fused-MoE launch selection."""

import dataclasses

from typing import Any, Mapping, Optional

from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.fused_moe_tiling import derive_fused_moe_tiling
from src.kernel.launch_config.selection_primitives import (
    require_dtype, require_positive_integer, require_shape, require_variant,
)
from src.kernel.model_contract import (
    SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER,
)
from src.kernel.launch_config import hardware_skeleton as _skeleton


_FUSED_MOE_EP_EVIDENCE_REFERENCE = (
    "microbench/metax_c500/references/"
    "fused_moe_ep_vllm_metax_0_24_0.json")
_FUSED_MOE_EP_EVIDENCE_SHA256 = (
    "0f1bde42bc27b14a604c530a02125c9c2533b1315506fbc39b05f2c34d9a4f48")
_RUNTIME_CONFIG_EVIDENCE_REFERENCE = (
    "microbench/metax_c500/references/"
    "fused_moe_runtime_config_vllm_metax_0_24_0_g436274.json")

# The C500 deployment used by the retained selected-step capture runs
# vLLM-metax 0.24.0+g436274 with BF16 Qwen3-30B-A3B (H=2048, E=128,
# TP=EP=2).  Its plugin points VLLM_TUNED_CONFIG_FOLDER at H=2048, but the
# exact E=64,N=768 file is absent from both search locations.  The source
# therefore calls get_default_config.  Keep the exact runtime-resolver scope
# explicit: it must not claim that a different model's tuned JSON or
# environment has been resolved.
_DEPLOYED_GLOBAL_EXPERTS = 128
_DEPLOYED_EXPERT_PARALLEL_SIZE = 2
_SOURCE_FUSED_MOE_CHUNK_SIZE = 16 * 1024
# The live deployed cache contains a compiled BF16 BM128/BN128/BK64 default
# program (8 warps, 3 stages, 64 KiB dynamic shared memory).  Combined with
# the version-pinned source default below this closes the one-chunk default
# path; a multi-chunk invocation remains a separate runtime contract.
_MAX_SOURCE_BACKED_DEFAULT_TOKENS = _SOURCE_FUSED_MOE_CHUNK_SIZE

# vLLM-metax 0.24.0's C500 package contains these unquantized tuned JSON
# domains.  The plugin first looks up the H-scoped file and then falls back
# to the package-default E=256/N=256 file.  The generic analytical launch
# below does not reproduce their nearest-M, staged W1/W2 selections, so a
# matching runtime domain must never silently take that path.
_INSTALLED_H_SCOPED_TUNED_CONFIG_DOMAINS = frozenset({
    (1024, 128, 192),
    (2048, 128, 192),
    (2048, 128, 384),
    (2048, 512, 64),
    (2048, 512, 128),
    (2560, 64, 768),
    (2560, 64, 6144),
    (4096, 8, 1792),
    (4096, 8, 3584),
    (4096, 128, 96),
    (4096, 128, 176),
    (4096, 128, 192),
    (4096, 128, 384),
    (4096, 128, 1368),
    (5120, 160, 96),
    (5120, 160, 192),
    (7168, 256, 64),
    (8192, 64, 224),
    (8192, 64, 448),
    (8192, 64, 1792),
})
_INSTALLED_PACKAGE_DEFAULT_TUNED_CONFIG_DOMAIN = (256, 256)


@dataclasses.dataclass(frozen=True)
class FusedMoERuntimeConfig:
    """One fully resolved vLLM-metax Triton configuration.

    ``source`` mirrors the precedence in
    ``try_get_optimal_moe_config``: an explicit override wins, then the
    nearest key in a loaded tuned JSON, then the source default.  The runtime
    selector only instantiates the observed default path below, while this
    complete resolver makes all source branches directly testable and
    fail-closed.
    """

    source: str
    batch_key: Optional[int]
    block_size_m: int
    block_size_n: int
    block_size_k: int
    group_size_m: int
    num_warps: int
    num_stages: int
    dynamic_shared_memory_bytes: int


_DYNAMIC_SHARED_MEMORY_BY_COMPILED_CONFIG = {
    # These are compiler resource facts from the deployed AOT Triton cache,
    # not duration-derived values.  The tuple is BM, BN, BK, warps, stages.
    (16, 64, 128, 4, 4): 40_960,
    (32, 64, 128, 4, 3): 49_152,
    (32, 128, 64, 4, 3): 40_960,
    (64, 128, 64, 4, 3): 49_152,
    (64, 128, 64, 8, 3): 49_152,
    (128, 128, 64, 8, 3): 65_536,
}

_BLOCK_THREADS_BY_DEFAULT_TILE_M = {
    16: 256,
    32: 256,
    64: 512,
    128: 512,
}


def _require_config_integer(config: Mapping[str, Any], name: str) -> int:
    if name not in config:
        raise ValueError(f"MetaX FusedMoE runtime config lacks {name}")
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"MetaX FusedMoE runtime config {name} must be an integer")
    if value <= 0:
        raise ValueError(
            f"MetaX FusedMoE runtime config {name} must be positive")
    return value


def _normalize_runtime_stage(config: Mapping[str, Any]) -> tuple[int, ...]:
    if not isinstance(config, Mapping):
        raise TypeError("MetaX FusedMoE runtime stage must be a mapping")
    required = (
        "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
        "num_warps", "num_stages",
    )
    allowed = set(required) | {"SPLIT_K", "ACCF32"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(
            "MetaX FusedMoE runtime stage has unknown keys: "
            f"{sorted(unknown)}")
    values = tuple(_require_config_integer(config, name) for name in required)
    if "SPLIT_K" in config:
        split_k = _require_config_integer(config, "SPLIT_K")
        if split_k != 1:
            raise NotImplementedError(
                "MetaX C500 FusedMoE SPLIT_K != 1 requires a distinct "
                "reduction launch contract")
    return values


def _stage_for_projection(
        config: Mapping[str, Any], projection: str) -> tuple[int, ...]:
    """Replicate vLLM-metax ``initialize_staged_config`` without defaults."""

    has_stage1 = "stage1" in config
    has_stage2 = "stage2" in config
    if has_stage1 != has_stage2:
        raise ValueError(
            "MetaX FusedMoE staged config must provide both stage1 and "
            "stage2")
    if has_stage1:
        allowed = {"stage1", "stage2"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(
                "MetaX FusedMoE staged config has unknown keys: "
                f"{sorted(unknown)}")
        stage_name = "stage1" if projection == "w1" else "stage2"
        return _normalize_runtime_stage(config[stage_name])
    return _normalize_runtime_stage(config)


def _source_default_config(batch_tokens: int, local_experts: int) -> dict[str, int]:
    """Copy vLLM-metax 0.24.0's BF16 non-batch-invariant default rule."""

    if batch_tokens <= 32:
        block_m = 16
    elif batch_tokens <= 96:
        block_m = 32
    elif batch_tokens <= 512:
        block_m = 64
    else:
        block_m = 128
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": 64 if batch_tokens <= 64 else 128,
        "BLOCK_SIZE_K": 128 if batch_tokens <= 64 else 64,
        "GROUP_SIZE_M": 16 if batch_tokens // local_experts > 128 else 1,
        "SPLIT_K": 1,
        "num_warps": 4 if batch_tokens <= 128 else 8,
        "num_stages": 4 if batch_tokens <= 32 else 3,
    }


def resolve_fused_moe_runtime_config(
        *, batch_tokens: int, local_experts: int, projection: str,
        batch_invariant: bool, override_config: Optional[Mapping[str, Any]],
        tuned_configs: Optional[Mapping[int, Mapping[str, Any]]],
) -> FusedMoERuntimeConfig:
    """Resolve C500's override/tuned/default config precedence.

    This is deliberately parameterized for direct source-contract tests.  The
    deployed selector passes the observed environment (no override, no loaded
    tuned file and ``VLLM_BATCH_INVARIANT=0``); it does not make a latency or
    case-ID decision.
    """

    if isinstance(batch_tokens, bool) or not isinstance(batch_tokens, int):
        raise TypeError("MetaX FusedMoE batch_tokens must be an integer")
    if batch_tokens <= 0:
        raise ValueError("MetaX FusedMoE batch_tokens must be positive")
    if (isinstance(local_experts, bool)
            or not isinstance(local_experts, int)):
        raise TypeError("MetaX FusedMoE local_experts must be an integer")
    if local_experts <= 0:
        raise ValueError("MetaX FusedMoE local_experts must be positive")
    projection = require_variant(projection, ("w1", "w2"), "FUSED_MOE")
    if not isinstance(batch_invariant, bool):
        raise TypeError("MetaX FusedMoE batch_invariant must be boolean")
    if batch_invariant:
        raise NotImplementedError(
            "MetaX C500 batch-invariant FusedMoE needs compiler resource "
            "evidence for its fixed BM64/BN64/BK32 launch")

    if override_config is not None:
        if not isinstance(override_config, Mapping):
            raise TypeError("MetaX FusedMoE override config must be a mapping")
        if override_config:
            source = "override"
            batch_key = None
            selected = override_config
        else:
            source = "default"
            batch_key = None
            selected = _source_default_config(batch_tokens, local_experts)
    elif tuned_configs is not None:
        if not isinstance(tuned_configs, Mapping):
            raise TypeError("MetaX FusedMoE tuned configs must be a mapping")
        if tuned_configs:
            for key, value in tuned_configs.items():
                if isinstance(key, bool) or not isinstance(key, int) or key <= 0:
                    raise ValueError(
                        "MetaX FusedMoE tuned batch keys must be positive "
                        "integers")
                if not isinstance(value, Mapping):
                    raise TypeError(
                        "MetaX FusedMoE tuned config entries must be mappings")
            # Python's min preserves insertion order for equal keys, matching
            # vLLM's ``min(configs.keys(), key=...)`` nearest-key behavior.
            batch_key = min(tuned_configs, key=lambda key: abs(key - batch_tokens))
            source = "tuned"
            selected = tuned_configs[batch_key]
        else:
            source = "default"
            batch_key = None
            selected = _source_default_config(batch_tokens, local_experts)
    else:
        source = "default"
        batch_key = None
        selected = _source_default_config(batch_tokens, local_experts)

    (block_m, block_n, block_k, group_m, num_warps,
     num_stages) = _stage_for_projection(selected, projection)
    resource_key = (block_m, block_n, block_k, num_warps, num_stages)
    if resource_key not in _DYNAMIC_SHARED_MEMORY_BY_COMPILED_CONFIG:
        raise NotImplementedError(
            "MetaX C500 FusedMoE has no source-bound compiler resource "
            f"record for {resource_key}")
    return FusedMoERuntimeConfig(
        source=source,
        batch_key=batch_key,
        block_size_m=block_m,
        block_size_n=block_n,
        block_size_k=block_k,
        group_size_m=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
        dynamic_shared_memory_bytes=(
            _DYNAMIC_SHARED_MEMORY_BY_COMPILED_CONFIG[resource_key]),
    )


def select_fused_moe_operand_replay(tile_m: int) -> float:
    """Return the version-bound L2-ingress stream count for a MetaX tile.

    BM16, BM32, and BM128 use the source-issued base stream; this is not a
    claim that all physical L2 traffic is explained, because retained layer-1
    counters contain BM16 outliers.  BM64's two streams are supported by
    same-invocation non-duration counters for the frozen MACA/vLLM/Triton
    runner.  Do not infer a second stream for BM128 without the corresponding
    counter evidence.
    """

    if isinstance(tile_m, bool) or not isinstance(tile_m, int):
        raise TypeError("MetaX grouped-GEMM tile_m must be an integer")
    if tile_m in (16, 32, 128):
        return 1.0
    elif tile_m == 64:
        return 2.0
    else:
        raise ValueError(
            "MetaX grouped-GEMM operand replay is source-backed only for "
            f"tile_m 16, 32, 64, or 128, got {tile_m}")


def select_fused_moe_w2_routed_weight_program(variant: str) -> float:
    """Return whether the selected projection has C500's W2 router program."""

    if variant == "w1":
        return 0.0
    elif variant == "w2":
        return 1.0
    else:
        raise ValueError(
            f"unknown MetaX grouped-GEMM projection role: {variant!r}")


def select_fused_moe_recipe(
        shape: Mapping[str, int], *, hardware) -> tuple[str, ...]:
    tokens, _, topk, hidden, _ = require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    del hardware
    # vLLM-metax 0.24.0's FusedMoE calls self.activation(...), which reaches
    # torch.ops._C.silu_and_mul and launches one fused
    # vllm::act_and_mul_kernel over the W1 projection.  The earlier two-launch
    # SiLU-then-multiply pair is an ATen TensorIterator decomposition that the
    # retained offline runner's torch build performs; the deployed serving
    # runtime does not, and its retained step trace contains exactly one
    # act_and_mul launch per layer.  A TensorIterator index split therefore
    # does not apply to this kernel either.
    # The retained decode step runs vllm::moe::topkGatingSoftmax once per
    # layer per rank, ahead of the moe_align bookkeeping, and the model did
    # not emit it at all.  Its launch is a compile-time function of the
    # expert count and the wavefront width; see the vector launch.
    recipe = (
        "moe_router_logit_cast", "moe_topk", "moe_routing_weight_norm", "moe_routing_weight_scale",
        "moe_routed_id_widen",
        "moe_routed_id_extent", "moe_routed_id_extent",
        "moe_fill", "moe_fill", "moe_fill",
        "moe_count_and_sort", "moe_align_opt_128", "moe_expert_map_lookup",
        "w1", "moe_act_and_mul", "w2", "moe_sum",
    )
    # PyTorch TensorIterator partitions reductions above the signed 30-bit
    # indexing limit into two identical launches.
    if tokens * topk * hidden > 2 ** 30:
        recipe += ("moe_sum",)
    return recipe


# The retained C500 FusedMoE recipe has no typed Cast helper.
select_fused_moe_cast_roles = _skeleton.no_typed_cast_roles_selector(
    select_fused_moe_recipe)


# C500's retained recipe has no ``moe_reduce`` invocation.
select_fused_moe_reduce_work_fractions = (
    _skeleton.select_no_fused_moe_reduce_work_fractions)


def select_fused_moe_launch(shape: Mapping[str, int], dtype: Optional[str],
                            variant: Optional[str], attributes: Mapping[str, object],
                            hardware) -> LaunchConfig:
    tokens, experts, topk, hidden, intermediate = require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    require_dtype(dtype, ("bf16",))
    projection = require_variant(variant, ("w1", "w2"), "FUSED_MOE")
    del hardware
    expert_parallel_size = require_positive_integer(
        attributes, "expert_parallel_size")
    if expert_parallel_size not in (1, _DEPLOYED_EXPERT_PARALLEL_SIZE):
        raise NotImplementedError(
            "MetaX C500 FUSED_MOE source evidence supports only EP1 or EP2")
    if (expert_parallel_size == _DEPLOYED_EXPERT_PARALLEL_SIZE
            and experts != _DEPLOYED_GLOBAL_EXPERTS):
        raise NotImplementedError(
            "MetaX C500 FUSED_MOE EP2 source evidence requires 128 experts")
    if experts % expert_parallel_size:
        raise ValueError(
            "expert_parallel_size must divide FUSED_MOE experts")
    local_experts = experts // expert_parallel_size

    # vLLM-metax resolves every unquantized BF16 no-config case through its
    # own ``get_default_config``.  The old shared-GPU two-regime formula did
    # not reproduce this M=32/64/96/128/512 source decision tree, even when
    # no tuned JSON was present.  Reject a matching JSON and otherwise route
    # all supported domains through the version-pinned C500 resolver below.
    h_scoped_tuned_domain = (hidden, local_experts, intermediate)
    if h_scoped_tuned_domain in _INSTALLED_H_SCOPED_TUNED_CONFIG_DOMAINS:
        raise NotImplementedError(
            "MetaX C500 FusedMoE matching installed H-scoped tuned "
            "config is not encoded in the source-backed launch "
            f"contract: H={hidden}, E={local_experts}, N={intermediate}")
    if ((local_experts, intermediate)
            == _INSTALLED_PACKAGE_DEFAULT_TUNED_CONFIG_DOMAIN):
        raise NotImplementedError(
            "MetaX C500 FusedMoE matching installed package-default "
            "tuned config is not encoded in the source-backed launch "
            f"contract: E={local_experts}, N={intermediate}")

    if tokens > _SOURCE_FUSED_MOE_CHUNK_SIZE:
        raise NotImplementedError(
            "MetaX C500 FusedMoE multi-chunk config resolution is not "
            "represented")
    if tokens > _MAX_SOURCE_BACKED_DEFAULT_TOKENS:
        raise NotImplementedError(
            "MetaX C500 FusedMoE multi-chunk default launch is outside the "
            "retained source-backed launch scope")

    # The selected deployment records VLLM_BATCH_INVARIANT=0 and an unset
    # get_config() override.  The C500 source reaches this same default branch
    # whenever the checked H-scoped/package-default files are absent.
    runtime_config = resolve_fused_moe_runtime_config(
        batch_tokens=tokens,
        local_experts=local_experts,
        projection=projection,
        batch_invariant=False,
        override_config=None,
        tuned_configs=None,
    )
    projections = derive_fused_moe_tiling(
        tokens, experts, topk, hidden, intermediate,
        runtime_config.block_size_m,
        runtime_config.block_size_n,
        runtime_config.block_size_k,
        group_size_m=runtime_config.group_size_m,
        # vLLM-metax creates sorted_token_ids with the stage-1 BM and reuses
        # that allocation for W2 when the default config has one shared stage.
        # In particular, a tiny W1 batch retains its clipped routing buffer
        # rather than allocating a new W2-sized buffer.
        reuse_stage1_routing=True,
    )
    selected = projections[projection]
    tile_m = runtime_config.block_size_m
    if tile_m not in _BLOCK_THREADS_BY_DEFAULT_TILE_M:
        raise NotImplementedError(
            "MetaX C500 FusedMoE has no compiler launch record for "
            f"BLOCK_SIZE_M={tile_m}")
    operand_replay = select_fused_moe_operand_replay(tile_m)
    parameters = {
        "group_size_m": float(runtime_config.group_size_m),
        "padded_rows": float(selected["EM"]),
        "mte2_tile_bytes": 0.0,
        "weight_tile_rows": 0.0,
        # Keep this discrete source/counter fact with the selected launch
        # rather than hiding it in a latency-calibration profile.  It
        # describes L2 ingress multiplicity, not worker order or residency.
        "operand_replay_a": operand_replay,
        "operand_replay_b": operand_replay,
        "dynamic_shared_memory_bytes": float(
            runtime_config.dynamic_shared_memory_bytes),
        # Source/compiler emission facts, not timing.  BLOCK_SIZE_M=64 with
        # BN128/BK64 compiles to two distinct programs (four warps at most 128
        # batch tokens, eight above) that share tile shape, block threads and
        # dynamic shared memory, so a consumer cannot tell them apart without
        # these two fields.
        "num_warps": float(runtime_config.num_warps),
        "num_stages": float(runtime_config.num_stages),
    }
    if expert_parallel_size > 1:
        # This parameter is a repeated source fact for rank-local EP launches,
        # not an EP1 launch property.  Omitting it for EP1 lets KernelSpec
        # reject a forged rank-local routing contract.
        parameters["expert_parallel_size"] = float(expert_parallel_size)
    # ``self.activation(..., intermediate_cache2, ...)`` writes its result
    # directly into the buffer passed to W2.  This marker selects W2's
    # source-visible router-weight read and scale program; it is not a D2D
    # copy claim.
    parameters[
        SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER] = (
        select_fused_moe_w2_routed_weight_program(
            projection))
    return LaunchConfig(
        algorithm=f"triton_fused_moe_{projection}",
        grid=(selected["grid"], 1, 1),
        block=(_BLOCK_THREADS_BY_DEFAULT_TILE_M[tile_m], 1, 1),
        tiling=TilingSpec(tile_shape={
            "m": runtime_config.block_size_m,
            "n": runtime_config.block_size_n,
            "k": runtime_config.block_size_k,
        }),
        parameters=parameters,
        execution_variant=projection,
    )
