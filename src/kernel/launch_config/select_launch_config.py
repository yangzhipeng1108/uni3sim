"""The only public dispatcher for hardware launch selection."""

from dataclasses import dataclass, replace
from importlib import import_module
import math
from typing import Mapping, Optional, Tuple

from src.kernel.hardware_ids import canonical_hardware_id
from src.kernel.kernel_spec import KernelKind
from src.kernel.launch_config.collective_selection import (
    select_declared_collective,
)
from src.kernel.launch_config import (
    CollectiveSelection,
    FlashAttentionForwardPart,
    FlashAttentionSetupLaunch,
    LaunchConfig,
)


@dataclass(frozen=True)
class HardwareLaunchDeclaration:
    """Where one hardware's launch selectors live, as data.

    A hardware is a *directory of modules* plus, optionally, the set of
    selectors it owns itself.  ``package`` is the module path that resolves
    every ``(module, function)`` the dispatcher asks for; ``own_package`` is
    the hardware's own directory, and ``owned`` the ``(module, function)``
    pairs whose requests route there instead of to ``package``.

    Hardware whose ``package`` is another hardware's directory *projects*
    that package: it reuses those selectors verbatim and overrides only the
    ``owned`` entries.  The route is decided by the request domain before
    selection, never by a selector failure, and hardware packages never
    import each other.  Adding a hardware therefore means adding one entry
    here plus its data -- not a new dispatch path.
    """

    package: str
    own_package: Optional[str] = None
    owned: Tuple[Tuple[str, str], ...] = ()


_HARDWARE_LAUNCH_DECLARATIONS = {
    "nvidia_a800": HardwareLaunchDeclaration(
        package="src.kernel.launch_config.nvidia_a800"),
    # Every H200 request resolves within its own hardware package. Missing
    # source bindings remain explicit gaps, never another device's launch.
    "nvidia_h200": HardwareLaunchDeclaration(
        package="src.kernel.launch_config.nvidia_h200",
        own_package="src.kernel.launch_config.nvidia_h200",
        # Launch selectors H200 owns itself: the Hopper vLLM 0.24 stack
        # launches kernels (DeepGEMM FP8, FP8 Triton MoE tiles, FA3 MLA)
        # that the A800 package does not select.  For each (module,
        # function) listed here the hardware's own module defines the
        # selector and an ``owns_<function>`` predicate over the same
        # arguments; requests inside that domain use the own selector, all
        # others still resolve within the H200 package.
        owned=(
            ("vector_launch", "select_vector_launch"),
            # wp11_moe_fp8: vLLM 0.24 FP8 block-quantized Triton fused MoE.
            ("fused_moe_launch", "select_fused_moe_launch"),
            # wp10_fa3_mla: FA3 MLA prefill/decode and the split combine.
            ("flash_attn_launch", "select_target_mla_launch"),
            ("flash_attn_launch", "select_flash_attn_launch"),
            # Block-FP8 dense GEMMs (DeepGEMM / TensorRT-LLM FP8 blockscale).
            ("gemm_launch", "select_gemm_launch"),
            # Strided-batched BF16 GEMMs (MLA absorbed W_UK / W_UV).
            ("gemm_launch", "select_target_gemm_launch"),
            # The split-K reducer of an observed SGLang cuBLAS router launch.
            ("gemm_launch", "select_gemm_reducer_launch"),
            # wp_deepseek_v4: FlashMLA two-tier sparse attention launches
            # (dsv4_sparse_prefill/decode) that only SM90 evidence covers.
            ("vector_launch", "select_sparse_attn_launch"),
            # wp_deepseek_v4: the analytical per-request split-KV budget
            # derived from the SM90 split-KV decode launch structure.
            ("vector_launch", "select_dsv4_decode_split_counts"),
            # vLLM V4 horizontally fused QNorm/RoPE/quantized SWA insert.
            ("vector_launch", "select_kv_cache_update_launch"),
            ("vector_launch", "select_target_vector_launch"),
            ("vector_launch", "select_mhc_stage_launches"),
        ),
    ),
    "metax_c500": HardwareLaunchDeclaration(
        package="src.kernel.launch_config.metax_c500"),
    "ascend_910b": HardwareLaunchDeclaration(
        package="src.kernel.launch_config.ascend_910b"),
}

_HARDWARE_PACKAGES = {
    hardware_id: declaration.package
    for hardware_id, declaration in _HARDWARE_LAUNCH_DECLARATIONS.items()
}
_OWN_PACKAGES = {
    hardware_id: declaration.own_package
    for hardware_id, declaration in _HARDWARE_LAUNCH_DECLARATIONS.items()
    if declaration.own_package is not None
}
_OWN_PACKAGE_SELECTORS = {
    hardware_id: frozenset(declaration.owned)
    for hardware_id, declaration in _HARDWARE_LAUNCH_DECLARATIONS.items()
    if declaration.owned
}


def _require_launch_hardware_spec(canonical_id: str, hardware, noun: str) -> None:
    """Require the HWSpec to describe the hardware the selector was asked for.

    ``noun`` names the selector in the failure text (launch, reducer, recipe,
    ...), matching the wording each caller used before this check was shared.
    """

    if canonical_hardware_id(hardware.hardware_id) != canonical_id:
        raise ValueError(
            f"{noun} hardware_id {canonical_id!r} does not match "
            f"HWSpec {hardware.hardware_id!r}")


def _require_launch_hardware(hardware_id: str) -> str:
    """Return the canonical id, or fail loud when no package selects it.

    Every public selector needs the same two steps: canonicalise the requested
    id and refuse one no hardware package owns.  The refusal is
    ``NotImplementedError`` because an unowned id is unsupported rather than a
    malformed request.
    """

    canonical_id = canonical_hardware_id(hardware_id)
    if canonical_id not in _HARDWARE_PACKAGES:
        raise NotImplementedError(
            f"no production launch selector for {canonical_id}")
    return canonical_id


def _launch_function(canonical_id: str, module_name: str, function_name: str):
    if canonical_id == "nvidia_h200":
        module = import_module(f"{_HARDWARE_PACKAGES[canonical_id]}.{module_name}")
        if function_name not in vars(module):
            raise NotImplementedError(
                f"H200 source interface {module_name}.{function_name} is not bound")
        return getattr(module, function_name)
    projected = getattr(
        import_module(f"{_HARDWARE_PACKAGES[canonical_id]}.{module_name}"),
        function_name)
    if (canonical_id not in _OWN_PACKAGE_SELECTORS
            or (module_name, function_name)
            not in _OWN_PACKAGE_SELECTORS[canonical_id]):
        return projected
    own_module = import_module(f"{_OWN_PACKAGES[canonical_id]}.{module_name}")
    own = getattr(own_module, function_name)
    owns = getattr(own_module, f"owns_{function_name}")

    def route(*args, **kwargs):
        if owns(*args, **kwargs):
            return own(*args, **kwargs)
        return projected(*args, **kwargs)

    return route


def moe_alignment_capacity(hardware_id, tokens, topk, experts, block_size):
    """Read the hardware package's public CUDA alignment allocation ABI."""
    canonical_id = _require_launch_hardware(hardware_id)
    if canonical_id not in ("nvidia_a800", "nvidia_h200"):
        raise ValueError("CUDA alignment allocation is not declared for this hardware")
    if any(type(value) is not int or value <= 0
            for value in (tokens, topk, experts, block_size)):
        raise ValueError("Alignment allocation requires positive integer dimensions")
    return _launch_function(
        canonical_id, "fused_moe_launch", "moe_alignment_capacity")(
            tokens, topk, experts, block_size)


def moe_count_sort_source_program(hardware_id: str) -> str:
    """Read the projected hardware package's auxiliary program identity."""
    canonical_id = _require_launch_hardware(hardware_id)
    program = _launch_function(
        canonical_id, "vector_launch", "moe_count_sort_source_program")()
    if type(program) is not str or program not in (
            "cuda_moe_count_sort_int32_unmasked",
            "metax_routed_token_histogram"):
        raise ValueError("unknown routed-token auxiliary source program")
    return program


_KERNEL_SELECTORS = {
    KernelKind.GEMM: ("gemm_launch", "select_gemm_launch"),
    KernelKind.FLASH_ATTN: ("flash_attn_launch", "select_flash_attn_launch"),
    KernelKind.FUSED_MOE: ("fused_moe_launch", "select_fused_moe_launch"),
    KernelKind.VECTOR: ("vector_launch", "select_vector_launch"),
}


def select_collective_launch(
    hardware_id: str,
    *,
    communication_library: str,
    library_version: str,
    operation: str,
    message_bytes: int,
    participant_count: int,
    dtype: str,
    reduction: str | None,
    in_place: bool,
    topology: str,
) -> CollectiveSelection:
    """Select the profiled vendor algorithm/protocol or fail loud."""

    canonical_id = canonical_hardware_id(hardware_id)
    if canonical_id not in _HARDWARE_PACKAGES:
        raise NotImplementedError(
            f"no collective launch selector for {canonical_id}")
    for name, value in (
            ("communication_library", communication_library),
            ("library_version", library_version),
            ("operation", operation),
            ("dtype", dtype),
            ("topology", topology)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"collective launch selection requires non-empty {name}")
    if isinstance(message_bytes, bool) or not isinstance(message_bytes, int):
        raise TypeError("collective message_bytes must be an integer")
    if message_bytes <= 0:
        raise ValueError("collective message_bytes must be positive")
    if (isinstance(participant_count, bool)
            or not isinstance(participant_count, int)):
        raise TypeError("collective participant_count must be an integer")
    if participant_count <= 1:
        raise ValueError("collective participant_count must exceed one")
    if reduction is not None and (
            not isinstance(reduction, str) or not reduction.strip()):
        raise ValueError("collective reduction must be non-empty or null")
    if not isinstance(in_place, bool):
        raise TypeError("collective in_place must be boolean")

    # Every hardware package declares its profiled collective library as
    # ``COLLECTIVE_RULES``; one shared engine selects from that data.
    package = import_module(_HARDWARE_PACKAGES[canonical_id])
    if canonical_id == "nvidia_h200" and "COLLECTIVE_RULES" not in vars(package):
        raise NotImplementedError(
            "H200 collective algorithm/protocol observations are not bound")
    rules = package.COLLECTIVE_RULES
    selection = select_declared_collective(
        rules,
        communication_library=communication_library,
        library_version=library_version,
        operation=operation,
        message_bytes=message_bytes,
        participant_count=participant_count,
        dtype=dtype,
        reduction=reduction,
        in_place=in_place,
        topology=topology,
    )
    if not isinstance(selection, CollectiveSelection):
        raise TypeError(
            f"{canonical_id} collective selector must return "
            "CollectiveSelection")
    if rules.hardware_id != canonical_id:
        # A projected hardware reuses another package's profiled rules; the
        # selection is still reported for the hardware that requested it.
        selection = replace(selection, hardware_id=canonical_id)
    expected_identity = (
        canonical_id,
        communication_library,
        library_version,
        operation,
        message_bytes,
        participant_count,
        topology,
    )
    actual_identity = (
        selection.hardware_id,
        selection.communication_library,
        selection.library_version,
        selection.operation,
        selection.message_bytes,
        selection.participant_count,
        selection.topology,
    )
    if actual_identity != expected_identity:
        raise ValueError(
            f"{canonical_id} collective selector changed request identity")
    return selection


def select_mhc_stage_launches(hardware_id, shape, dtype, variant, attributes, hardware):
    """Dispatch a logical mHC call to its source-backed hardware stage owner."""
    canonical_id = _require_launch_hardware(hardware_id)
    _require_launch_hardware_spec(canonical_id, hardware, "mhc_stage_launches")
    selector_key = ("vector_launch", "select_mhc_stage_launches")
    if (canonical_id not in _OWN_PACKAGE_SELECTORS
            or selector_key not in _OWN_PACKAGE_SELECTORS[canonical_id]):
        raise NotImplementedError(f"no source-backed mHC stage selector for {canonical_id}")
    selector = getattr(import_module(f"{_OWN_PACKAGES[canonical_id]}.vector_launch"),
                       "select_mhc_stage_launches")
    return selector(shape, dtype, variant, attributes, hardware)


def select_dsv4_decode_split_counts(
    hardware_id: str,
    shape: Mapping[str, int],
    dtype: Optional[str],
    variant: str,
    attributes: Mapping[str, object],
    hardware,
) -> tuple[int, ...]:
    """Route a DSV4 split-KV decode plan request to its launch owner.

    The per-request split counts are a hardware launch decision owned by the
    hardware launch package (H200 ``vector_launch``); Serving lowering and
    predictors consume this decision and never derive it themselves.
    """

    canonical_id = _require_launch_hardware(hardware_id)
    _require_launch_hardware_spec(
        canonical_id, hardware, "dsv4_decode_split_counts")
    if not isinstance(shape, Mapping):
        raise TypeError("split-count shape must be a mapping")
    if not isinstance(attributes, Mapping):
        raise TypeError("split-count attributes must be a mapping")
    selector = _launch_function(
        canonical_id, "vector_launch", "select_dsv4_decode_split_counts")
    counts = selector(shape, dtype, variant, attributes, hardware)
    if (not isinstance(counts, tuple)
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                   for v in counts)):
        raise TypeError(
            f"{canonical_id} split-count selector must return a tuple of "
            "positive integers")
    return counts


def select_launch_config(
    hardware_id: str,
    kernel_kind: KernelKind | str,
    shape: Mapping[str, int],
    dtype: str,
    variant: str,
    attributes: Mapping[str, object],
    hardware,
) -> LaunchConfig:
    """Select one complete launch without importing any predictor module."""

    canonical_id = canonical_hardware_id(hardware_id)
    kind = KernelKind.parse(kernel_kind)
    if canonical_id not in _HARDWARE_PACKAGES:
        raise NotImplementedError(
            f"no production launch selector for {canonical_id}")
    if hardware is None:
        raise ValueError("launch selection requires an explicit HWSpec")
    _require_launch_hardware_spec(canonical_id, hardware, "launch")
    if dtype is None or not isinstance(dtype, str) or not dtype.strip():
        raise ValueError(f"{kind.value} launch selection requires dtype")
    if variant is None or not isinstance(variant, str) or not variant.strip():
        raise ValueError(f"{kind.value} launch selection requires variant")
    if not isinstance(shape, Mapping):
        raise TypeError("launch shape must be a mapping")
    if not isinstance(attributes, Mapping):
        raise TypeError("launch attributes must be a mapping")

    normalized_variant = variant.strip().lower()
    if normalized_variant == "fa3_prepare_varlen":
        if kind is not KernelKind.VECTOR or dtype.strip().lower() != "int32":
            raise ValueError("FA3 prepare launch requires VECTOR/INT32 source metadata")
        return hardware.fa3_prepare_varlen_launch(shape["batch"])
    if (kind is KernelKind.VECTOR
            and normalized_variant in (
                "kv_cache", "kvcache", "paged_kv_cache")):
        raise ValueError(
            "KV cache launch selection requires KernelKind.KV_CACHE with "
            "the paged_kv_cache variant")
    if (kind is KernelKind.KV_CACHE
            and normalized_variant in ("kv_cache", "kvcache")):
        raise ValueError(
            "KV cache launch selection requires the canonical "
            "paged_kv_cache variant")
    if kind is KernelKind.KV_CACHE and normalized_variant == (
            "paged_kv_cache"):
        module_name, function_name = "vector_launch", "select_vector_launch"
        selector = _launch_function(canonical_id, module_name, function_name)
        launch = selector(
            shape, dtype, "kv_cache", attributes, hardware=hardware)
        return replace(launch, execution_variant="paged_kv_cache")
    target_selectors = {
        KernelKind.GEMM: ("gemm_launch", "select_target_gemm_launch"),
        KernelKind.FLASH_ATTN: (
            "flash_attn_launch", "select_target_mla_launch"),
        KernelKind.VECTOR: ("vector_launch", "select_target_vector_launch"),
        KernelKind.SPARSE_ATTN_INDEXER: (
            "vector_launch", "select_sparse_attn_indexer_launch"),
        KernelKind.SPARSE_ATTN: (
            "vector_launch", "select_sparse_attn_launch"),
        KernelKind.KV_CACHE: (
            "vector_launch", "select_kv_cache_update_launch"),
        # The Kimi W4A16 aggregate is target-model specific even though its
        # selector lives beside the ordinary A800 FusedMoE projections.
        KernelKind.FUSED_MOE: (
            "fused_moe_launch", "select_fused_moe_launch"),
    }
    target_variants = {
        "batched_gemm", "moe_router", "layernorm", "mla_rope",
        "mla_prefill", "mla_decode", "mla_absorbed_prefill",
        "dsa_indexer_prefill", "dsa_indexer_decode", "dsa_indexer_mixed",
        "dsa_index_reuse", "dsa_sparse_prefill", "dsa_sparse_decode",
        "mla_kv_cache", "dsa_fp8_index_cache", "routed_experts",
        "dsv4_indexer_prefill", "dsv4_indexer_decode",
        "dsv4_sparse_prefill", "dsv4_sparse_decode",
        "dsv4_compressed_state_cache", "dsv4_compressor_state_cache",
        "dsv4_qnorm_rope_kv_insert", "dsv4_q_kv_rmsnorm", "dsv4_short_context_topk",
        "dsv32_fused_norm_rope", "dsv32_fused_q", "dsv32_eh_norm", "dsv32_embed_eh_norm", "dsv32_embed_norm",
        "dsv4_inv_rope_fp8_quant", "dsv4_moe_route",
        "mhc_post_prenorm", "mhc_pre_norm", "mhc_pre_norm_broadcast",
        "mhc_post", "hc_head",
    }
    if normalized_variant in target_variants:
        if canonical_id not in ("nvidia_a800", "nvidia_h200") or kind not in target_selectors:
            raise NotImplementedError(
                f"no source-backed {canonical_id} launch selector for "
                f"{kind.value}/{normalized_variant}")
        module_name, function_name = target_selectors[kind]
        selector = _launch_function(canonical_id, module_name, function_name)
        launch = selector(
            shape, dtype, normalized_variant, attributes, hardware=hardware)
        if launch.execution_variant != normalized_variant:
            raise ValueError(
                "target launch selector changed the requested variant")
        return launch
    if kind not in _KERNEL_SELECTORS:
        raise ValueError(f"{kind.value} does not use compute launch geometry")
    module_name, function_name = _KERNEL_SELECTORS[kind]
    selector = _launch_function(canonical_id, module_name, function_name)
    launch = selector(
        shape, dtype, variant, attributes, hardware=hardware)
    if launch.execution_variant is None:
        raise ValueError(
            f"{canonical_id} {kind.value} selector did not declare "
            "LaunchConfig.execution_variant")
    return launch


def select_workload_source_launch_recipe(hardware_id, shape, dtype, variant, attributes, hardware):
    """Select physical source stages beside an existing logical cost model.

    One logical math operation may have several native submissions. Return
    their existing LaunchConfig descriptors rather than mislabeling an
    aggregate launch. The logical spec and Roofline stay unchanged; Midlayer
    may explicitly expand a supported bound recipe into physical predictions.
    """
    canonical_id = _require_launch_hardware(hardware_id)
    if hardware is None:
        raise ValueError("source recipe requires an explicit matching HWSpec")
    _require_launch_hardware_spec(canonical_id, hardware, "source recipe")
    if canonical_id != "nvidia_h200":
        raise NotImplementedError("workload source recipe bridge currently owns H200 only")
    if variant == "vllm_mtp_eager":
        return _launch_function(canonical_id, "mtp_launch", "select_mtp_eager_stage_recipe")(
            shape, dtype, attributes, hardware)
    if variant == "mla_decode":
        return _launch_function(canonical_id, "flashmla_dense_launch", "select_dense_flashmla_decode_recipe")(
            shape, dtype, attributes, hardware)
    if variant == "dspark_prepare_query":
        return _launch_function(canonical_id, "dspark_launch", "select_dspark_prepare_query_recipe")(
            shape, dtype, attributes, hardware)
    if variant == "dspark_noncausal_indices":
        return _launch_function(canonical_id, "dspark_launch", "select_dspark_noncausal_indices_recipe")(
            shape, dtype, attributes, hardware)
    if variant == "dspark_primitive":
        return _launch_function(canonical_id, "dspark_launch", "select_dspark_primitive_recipe")(
            shape, dtype, attributes, hardware)
    if variant == "marlin_routed_experts":
        return _launch_function(canonical_id, "marlin_launch", "select_marlin_prepared_expert_launch_recipe")(
            shape, dtype, attributes, hardware)
    if variant == "routed_experts":
        if attributes["routed_weight_dtype"] == "fp8":
            return _launch_function(canonical_id, "moe_fp8_launch", "select_source_fp8_moe_recipe")(
                shape, dtype, attributes, hardware)
        if attributes["routed_weight_dtype"] == "bf16":
            return _launch_function(canonical_id, "moe_source_launch", "select_source_expert_launch_recipe")(
                shape, dtype, attributes, hardware)
        raise NotImplementedError("expert source recipe requires exact BF16 or block128 FP8 weight storage")
    if variant == "dsv4_indexer_q_rope_quant":
        launch = _launch_function(canonical_id, "v4_indexer_launch", "select_v4_indexer_q_launch")(
            shape, dtype, attributes, hardware)
        return (("q_rope_quant", launch),)
    if variant == "dsv4_short_context_topk":
        return _launch_function(canonical_id, "v4_indexer_launch", "select_v4_short_context_indexer_recipe")(
            shape, dtype, attributes, hardware)
    if variant in ("dsv4_indexer_prefill", "dsv4_indexer_decode"):
        if ((variant == "dsv4_indexer_prefill" and shape["num_decode_tokens"] != 0)
                or (variant == "dsv4_indexer_decode" and shape["num_prefill_tokens"] != 0)):
            raise ValueError("V4 source recipe phase disagrees with its actual token partition")
        return _launch_function(canonical_id, "v4_indexer_launch", "select_v4_prepared_indexer_recipe")(
            shape, dtype, attributes, hardware)
    if variant in ("dsa_indexer_decode", "dsa_indexer_prefill", "dsa_indexer_mixed"):
        decode, prefill = shape["num_decode_tokens"], shape["num_prefill_tokens"]
        if ((variant == "dsa_indexer_decode" and prefill != 0)
                or (variant == "dsa_indexer_prefill" and decode != 0)
                or (variant == "dsa_indexer_mixed" and (decode <= 0 or prefill <= 0))):
            raise ValueError("DSA source recipe phase disagrees with its actual token partition")
        return _launch_function(canonical_id, "dsa_launch", "select_dsa_prepared_indexer_recipe")(
            shape, dtype, attributes, hardware)
    if variant in ("dsa_sparse_decode", "dsa_sparse_prefill", "dsa_sparse_mixed"):
        if attributes["source_kv_cache_layout"] == "paged_bf16_rows_576":
            return _launch_function(canonical_id, "flashmla_sparse_launch", "select_dsa_bf16_flashmla_attention_recipe")(
                shape, dtype, attributes, hardware)
        return _launch_function(canonical_id, "flashmla_sparse_launch", "select_dsa_flashmla_attention_recipe")(
            shape, dtype, attributes, hardware)
    raise NotImplementedError("workload source recipe variant is not bound")


def dsa_cooperative_topk_cluster(hardware_id, recipe):
    """Validate the cooperative-only source-cost domain through its launch owner."""
    canonical_id = _require_launch_hardware(hardware_id)
    return _launch_function(canonical_id, "dsa_launch", "require_dsa_cooperative_topk_cluster")(recipe)


def dsa_decode_topk_source_route(hardware_id, recipe):
    """Return the launch-bound decode TopK route facts through its launch owner."""
    canonical_id = _require_launch_hardware(hardware_id)
    return _launch_function(canonical_id, "dsa_launch", "dsa_decode_topk_source_route")(recipe)


def select_flash_attn_copy_launches(
    hardware_id: str,
    shape: Mapping[str, int],
    attributes: Mapping[str, object],
    hardware,
) -> tuple[tuple[str, str, int, LaunchConfig], ...]:
    """Select backend-defined physical copies surrounding FlashAttention."""

    canonical_id = canonical_hardware_id(hardware_id)
    if canonical_id not in _HARDWARE_PACKAGES:
        raise NotImplementedError(
            f"no FlashAttention copy selector for {canonical_id}")
    if hardware is None:
        raise ValueError(
            "FlashAttention copy launch selection requires an explicit HWSpec")
    _require_launch_hardware_spec(canonical_id, hardware, "launch")
    selector = _launch_function(canonical_id, "flash_attn_launch", "select_flash_attn_copy_launches")
    selections = tuple(selector(shape, attributes, hardware=hardware))
    for position, dtype, element_count, launch in selections:
        if position not in ("before", "after"):
            raise ValueError(
                "FlashAttention copy position must be 'before' or 'after'")
        if dtype not in ("bf16", "float32"):
            raise ValueError(
                f"unsupported FlashAttention copy dtype: {dtype!r}")
        if (isinstance(element_count, bool)
                or not isinstance(element_count, int)
                or element_count <= 0):
            raise ValueError(
                "FlashAttention copy element_count must be a positive integer")
        if (not isinstance(launch, LaunchConfig)
                or launch.execution_variant != "elementwise"):
            raise ValueError(
                "FlashAttention copy selector must emit elementwise launches")
    return selections


def select_flash_attn_forward_partition(
    hardware_id: str,
    shape: Mapping[str, int],
    variant: str,
    attributes: Mapping[str, object],
) -> tuple[FlashAttentionForwardPart, ...]:
    """Select the backend pathways one logical FlashAttention call runs.

    An empty result means the backend executes the call whole.
    """

    canonical_id = canonical_hardware_id(hardware_id)
    if canonical_id not in _HARDWARE_PACKAGES:
        raise NotImplementedError(
            f"no FlashAttention forward partition for {canonical_id}")
    parts = tuple(_launch_function(
        canonical_id, "flash_attn_launch",
        "select_flash_attn_forward_partition")(shape, variant, attributes))
    if any(not isinstance(part, FlashAttentionForwardPart) for part in parts):
        raise TypeError(
            "FlashAttention forward partition must emit "
            "FlashAttentionForwardPart values")
    if parts:
        covered = sorted(
            index for part in parts for index in part.request_indexes)
        if covered != list(range(len(attributes["q_lens"]))):
            raise ValueError(
                "FlashAttention forward partition must cover every request "
                "exactly once")
    return parts


def select_flash_attn_output_fill_launches(
    hardware_id: str,
    shape: Mapping[str, int],
    dtype: str,
    attributes: Mapping[str, object],
    hardware,
) -> tuple[tuple[str, int, LaunchConfig], ...]:
    """Select the backend-defined output-buffer fill around FlashAttention."""

    canonical_id = canonical_hardware_id(hardware_id)
    if canonical_id not in _HARDWARE_PACKAGES:
        raise NotImplementedError(
            f"no FlashAttention output fill selector for {canonical_id}")
    if hardware is None:
        raise ValueError(
            "FlashAttention output fill selection requires an explicit "
            "HWSpec")
    _require_launch_hardware_spec(canonical_id, hardware, "launch")
    selector = _launch_function(canonical_id, "flash_attn_launch", "select_flash_attn_output_fill_launches")
    selections = tuple(selector(shape, dtype, attributes, hardware=hardware))
    for fill_dtype, element_count, launch in selections:
        if fill_dtype not in ("bf16", "fp16"):
            raise ValueError(
                "unsupported FlashAttention output fill dtype: "
                f"{fill_dtype!r}")
        if (isinstance(element_count, bool)
                or not isinstance(element_count, int)
                or element_count <= 0):
            raise ValueError(
                "FlashAttention output fill element_count must be a "
                "positive integer")
        if (not isinstance(launch, LaunchConfig)
                or launch.execution_variant != "elementwise"):
            raise ValueError(
                "FlashAttention output fill selector must emit elementwise "
                "launches")
    return selections


def select_source_flash_attn_partial_launch(hardware_id, shape, dtype, variant, attributes, hardware):
    """Ask the existing H200 fresh-prefill owner for its partial main.

    This source domain owns no general setup/copy/fill ABI. In particular,
    the one-INT32 scheduler reset remains an explicit small omission in the
    existing partial predictor, rather than an invented ATen build identity.
    """
    canonical_id = _require_launch_hardware(hardware_id)
    _require_launch_hardware_spec(canonical_id, hardware, "partial attention source")
    if (canonical_id != "nvidia_h200" or variant != "prefill" or dtype != "bf16"
            or attributes["source_component"] != "symbolic_gqa_mathematical"
            or attributes["source_model_execution_version"] != "vllm_0_30_0"
            or attributes["launch_policy"] != "analytical_launch"):
        return None
    owns = _launch_function(canonical_id, "flash_attn_launch", "owns_select_flash_attn_launch")
    if not owns(shape, dtype, variant, attributes, hardware):
        return None
    return _launch_function(canonical_id, "flash_attn_launch", "select_flash_attn_launch")(
        shape, dtype, variant, attributes, hardware)


def normalize_flash_attn_execution_attributes(
    hardware_id: str,
    shape: Mapping[str, int],
    attributes: Mapping[str, object],
) -> Mapping[str, object]:
    """Return source-level attributes actually passed to the backend ABI."""

    canonical_id = canonical_hardware_id(hardware_id)
    if canonical_id not in _HARDWARE_PACKAGES:
        raise NotImplementedError(
            f"no FlashAttention execution contract for {canonical_id}")
    normalized = _launch_function(
        canonical_id, "flash_attn_launch", "normalize_flash_attn_execution_attributes")(
        shape, attributes)
    if not isinstance(normalized, Mapping):
        raise TypeError(
            "FlashAttention execution attributes must be a mapping")
    return dict(normalized)


def select_flash_attn_setup_launches(
    hardware_id: str,
    shape: Mapping[str, int],
    dtype: str,
    variant: str,
    attributes: Mapping[str, object],
    hardware,
) -> tuple[FlashAttentionSetupLaunch, ...]:
    """Select source-defined setup kernels preceding FlashAttention."""

    canonical_id = canonical_hardware_id(hardware_id)
    if canonical_id not in _HARDWARE_PACKAGES:
        raise NotImplementedError(
            f"no FlashAttention setup selector for {canonical_id}")
    if hardware is None:
        raise ValueError(
            "FlashAttention setup launch selection requires an explicit "
            "HWSpec")
    _require_launch_hardware_spec(canonical_id, hardware, "launch")
    selections = tuple(_launch_function(
        canonical_id, "flash_attn_launch", "select_flash_attn_setup_launches")(
        shape, dtype, variant, attributes, hardware=hardware))
    if any(not isinstance(item, FlashAttentionSetupLaunch)
           for item in selections):
        raise TypeError(
            "FlashAttention setup selector must emit "
            "FlashAttentionSetupLaunch values")
    return selections


def select_gemm_reducer_launch(
    hardware_id: str,
    shape: Mapping[str, int],
    dtype: str,
    variant: str,
    attributes: Mapping[str, object],
    hardware,
) -> LaunchConfig | None:
    """Select the hardware-owned reducer paired with one GEMM launch."""

    canonical_id = _require_launch_hardware(hardware_id)
    if hardware is None:
        raise ValueError(
            "GEMM reducer selection requires an explicit HWSpec")
    _require_launch_hardware_spec(canonical_id, hardware, "reducer")
    if dtype is None or not isinstance(dtype, str) or not dtype.strip():
        raise ValueError("GEMM reducer selection requires dtype")
    if variant is None or not isinstance(variant, str) or not variant.strip():
        raise ValueError("GEMM reducer selection requires variant")
    if not isinstance(shape, Mapping):
        raise TypeError("GEMM reducer shape must be a mapping")
    if not isinstance(attributes, Mapping):
        raise TypeError("GEMM reducer attributes must be a mapping")

    reducer = _launch_function(
        canonical_id, "gemm_launch", "select_gemm_reducer_launch")(
        shape, dtype, variant, attributes, hardware=hardware)
    if reducer is None:
        return None
    if not isinstance(reducer, LaunchConfig):
        raise TypeError(
            f"{canonical_id} GEMM reducer selector must return "
            "LaunchConfig or None")
    if reducer.execution_variant != "gemm_reduce":
        raise ValueError(
            f"{canonical_id} GEMM reducer selector did not declare "
            "execution_variant='gemm_reduce'")
    return reducer


def select_block_fp8_input_quant_launch(
    hardware_id: str,
    gemm_launch: LaunchConfig,
    element_count: int,
    hardware,
) -> LaunchConfig | None:
    """Return the separate activation quantiser a block-FP8 launch needs.

    vLLM 0.24 quantises the BF16 activation of a DeepGEMM 1d2d linear with a
    separate ``per_token_group_quant_8bit`` launch (source geometry in
    ``block_fp8_gemm``); the swap-AB path quantises inside its own call and
    every other GEMM launch has no quantiser.  The launch follows from the
    selected GEMM algorithm, which only a hardware package that owns
    block-FP8 selection emits.
    """

    from src.kernel.launch_config.block_fp8_gemm import (
        DEEPGEMM_1D2D_ALGORITHM,
    )

    if gemm_launch.algorithm != DEEPGEMM_1D2D_ALGORITHM:
        return None
    canonical_id = canonical_hardware_id(hardware_id)
    _require_launch_hardware_spec(canonical_id, hardware, "quant")
    return per_token_group_quant_launch(element_count)


def select_fused_moe_input_quant_launch(
    hardware_id: str,
    moe_launch: LaunchConfig,
    element_count: int,
    hardware,
) -> LaunchConfig | None:
    """Return the activation quantiser in front of one fused-MoE GEMM.

    Only the FP8 block-128 Triton fused MoE (a launch only a hardware package
    that owns it emits) quantises its GEMM input with a separate
    ``per_token_group_quant_8bit`` launch.
    """

    from src.kernel.launch_config.block_fp8_gemm import (
        FP8_BLOCK_FUSED_MOE_ALGORITHM_PREFIX,
        SGLANG_FP8_BLOCK_FUSED_MOE_ALGORITHM_PREFIX,
    )

    if not moe_launch.algorithm.startswith((
            FP8_BLOCK_FUSED_MOE_ALGORITHM_PREFIX,
            SGLANG_FP8_BLOCK_FUSED_MOE_ALGORITHM_PREFIX)):
        return None
    canonical_id = canonical_hardware_id(hardware_id)
    _require_launch_hardware_spec(canonical_id, hardware, "quant")
    return per_token_group_quant_launch(element_count)


def per_token_group_quant_launch(element_count: int) -> LaunchConfig:
    """The vLLM 0.24 ``per_token_group_quant_8bit`` launch of one tensor."""

    from src.kernel.launch_config.block_fp8_gemm import (
        PER_TOKEN_GROUP_QUANT_ALGORITHM,
        per_token_group_quant_geometry,
    )

    ctas, threads = per_token_group_quant_geometry(element_count)
    return LaunchConfig(
        algorithm=PER_TOKEN_GROUP_QUANT_ALGORITHM,
        grid=(ctas, 1, 1),
        block=(threads, 1, 1),
        tiling=None,
        parameters={},
        execution_variant="fp8_group_quant",
    )


def select_fused_moe_recipe(
    hardware_id: str,
    shape: Mapping[str, int],
    hardware,
    selection_attributes: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    """Return one hardware-owned full fused-MoE kernel sequence."""

    canonical_id = _require_launch_hardware(hardware_id)
    if hardware is None:
        raise ValueError("fused-MoE recipe selection requires an explicit HWSpec")
    _require_launch_hardware_spec(canonical_id, hardware, "recipe")
    if not isinstance(shape, Mapping):
        raise TypeError("fused-MoE recipe shape must be a mapping")
    selector = _launch_function(
        canonical_id, "fused_moe_launch", "select_fused_moe_recipe")
    if canonical_id == "nvidia_h200":
        recipe = selector(shape, hardware=hardware, attributes=selection_attributes)
    else:
        recipe = selector(shape, hardware=hardware)
    if not isinstance(recipe, tuple) or not recipe:
        raise TypeError("fused-MoE recipe must be a non-empty tuple")
    if any(not isinstance(item, str) or not item for item in recipe):
        raise TypeError("fused-MoE recipe entries must be non-empty strings")
    if recipe.count("w1") != 1 or recipe.count("w2") != 1:
        raise ValueError("fused-MoE recipe must contain exactly one w1 and w2")
    helper_roles = ({"logits_cast", "normalize_sum", "normalize_div",
                     "expert_ids_long", "expert_map_lookup"}
        if canonical_id == "nvidia_h200" and selection_attributes is not None
        and selection_attributes["model_execution_version"] == "vllm_0_11_0" else set())
    invalid = [
        item for item in recipe
        if item not in ("w1", "w2") and not item.startswith("moe_") and item not in helper_roles
    ]
    if invalid:
        raise ValueError(f"invalid fused-MoE recipe entries: {invalid}")
    return recipe


def select_fused_moe_cast_roles(
    hardware_id: str,
    shape: Mapping[str, int],
    hardware,
    selection_attributes: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    """Return the typed Cast invocations in one hardware FusedMoE recipe.

    A recipe only records that a physical ``moe_cast`` kernel occurs.  The
    source ABI determines which logical tensor it converts, and that identity
    changes both its element count and its traffic.  Keep that dispatch beside
    the hardware-owned recipe rather than inferring a role from its ordinal in
    a consumer.
    """

    canonical_id = _require_launch_hardware(hardware_id)
    if hardware is None:
        raise ValueError("fused-MoE cast-role selection requires an explicit HWSpec")
    _require_launch_hardware_spec(canonical_id, hardware, "cast-role")
    if not isinstance(shape, Mapping):
        raise TypeError("fused-MoE cast-role shape must be a mapping")
    selector = _launch_function(
        canonical_id, "fused_moe_launch", "select_fused_moe_cast_roles")
    roles = (selector(shape, hardware=hardware, attributes=selection_attributes)
        if canonical_id == "nvidia_h200" else selector(shape, hardware=hardware))
    if not isinstance(roles, tuple):
        raise TypeError("fused-MoE cast roles must be a tuple")
    allowed_roles = {"topk_ids", "expert_tokens"}
    if any(role not in allowed_roles for role in roles):
        raise ValueError(
            "fused-MoE cast roles must be drawn from "
            f"{sorted(allowed_roles)}, got {roles!r}")
    if len(set(roles)) != len(roles):
        raise ValueError("fused-MoE cast roles cannot repeat")
    recipe = select_fused_moe_recipe(canonical_id, shape, hardware, selection_attributes)
    if recipe.count("moe_cast") != len(roles):
        raise ValueError(
            "fused-MoE recipe cast count disagrees with typed cast roles")
    return roles


def select_fused_moe_reduce_work_fractions(
    hardware_id: str,
    shape: Mapping[str, int],
    hardware,
    selection_attributes: Mapping[str, object] | None = None,
) -> tuple[float, ...]:
    """Return hardware-source-derived work shares for ``moe_reduce`` calls."""

    canonical_id = _require_launch_hardware(hardware_id)
    if hardware is None:
        raise ValueError(
            "fused-MoE reduction selection requires an explicit HWSpec")
    _require_launch_hardware_spec(canonical_id, hardware, "reduction")
    if not isinstance(shape, Mapping):
        raise TypeError("fused-MoE reduction shape must be a mapping")
    recipe = select_fused_moe_recipe(
        canonical_id, shape, hardware=hardware, selection_attributes=selection_attributes)
    expected_count = recipe.count("moe_reduce")
    if expected_count == 0:
        return ()
    selector = _launch_function(
        canonical_id, "fused_moe_launch", "select_fused_moe_reduce_work_fractions")
    fractions = (selector(shape, hardware=hardware, attributes=selection_attributes)
        if canonical_id == "nvidia_h200" else selector(shape, hardware=hardware))
    if not isinstance(fractions, tuple):
        raise TypeError(
            "fused-MoE reduction work fractions must be a tuple")
    if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 < float(value) <= 1
            for value in fractions):
        raise ValueError(
            "fused-MoE reduction work fractions must be finite values in "
            "(0, 1]")
    normalized = tuple(float(value) for value in fractions)
    if normalized and not math.isclose(
            sum(normalized), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "fused-MoE reduction work fractions must sum to one")
    if expected_count != len(normalized):
        raise ValueError(
            "fused-MoE recipe and reduction work fractions disagree: "
            f"recipe_count={expected_count}, "
            f"fraction_count={len(normalized)}")
    return normalized


_SOURCE_LAUNCH_FUNCTIONS = {"select_kda_chunk_stage_launch": "kda_chunk_launch"}


def source_launch_function(hardware_id, name):
    """Resolve one closed source API through its hardware-owned package."""
    from src.kernel.hardware_spec import HWSpec
    canonical = _require_launch_hardware(hardware_id)
    if not HWSpec.source_contract_supported("h200_partial_source_programs", canonical):
        raise ValueError("source launch API requires its declared hardware capability")
    if name not in _SOURCE_LAUNCH_FUNCTIONS:
        raise ValueError("unknown source launch API")
    module = import_module(f"{_OWN_PACKAGES[canonical]}.{_SOURCE_LAUNCH_FUNCTIONS[name]}")
    return getattr(module, name)
