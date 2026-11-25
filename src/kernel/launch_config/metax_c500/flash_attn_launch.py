"""MetaX C500 FlashAttention launch selection from pinned 2.6.3 source."""

from typing import Mapping, Optional

from src.kernel.launch_config import (
    FlashAttentionForwardPart,
    FlashAttentionSetupLaunch,
    LaunchConfig,
    TilingSpec,
)
from src.kernel.launch_config.flash_attn_workload import (
    flash_attn_workload_key,
)
from src.kernel.launch_config.selection_primitives import (
    ceil_div,
    launch_policy,
    require_dtype,
    require_positive_integer,
    require_shape,
    require_variant,
    select_analytical_gpu_flash_attention_splits,
)
from src.kernel.launch_config.tiling_spec import (
    METAX_MLA_SCHEDULER_PARTITIONS,
    derive_mla_scheduler_work,
    metax_mla_combine_block_m,
)
from src.kernel.launch_config import hardware_skeleton as _skeleton


# The canonical evidence artifact is the captured MACA FlashAttention binary.
# It is built against MACA 3.7.2 and has SHA256
# ``2a6f8a312dc8380c7435b4da1c4610030dcf49c7ee0a99b35ae47a423d643727``.
# ``get_hdim(int)`` in that binary resolves the source dispatch domain to
# these explicit template head dimensions.  The selector intentionally keeps
# only the dimensions whose XCORE1000 forward traits are present in the same
# binary; dimensions outside this set fail closed at the ABI boundary.
_SUPPORTED_HEAD_DIMS = frozenset({64, 128, 192})

# Source-backed ``get_tile_size(int, int, int)`` facts from the same binary
# (text address 0x1cf9f60).  The packed return is ``(block_m << 32) |
# block_n``.  The third argument is the effective query-head work amount used
# by FA2's dispatch; the second argument is not consulted by the routine.
# The final item is the XCORE1000 ``run_flash_splitkv_fwd`` warp count for the
# emitted trait.  ``None`` denotes the default branch in the disassembly.
_BINARY_SPLIT_TILE_RULES = {
    64: ((16, 16, 16, 1), (None, 64, 64, 4)),
    128: ((16, 16, 16, 1), (32, 32, 32, 2), (None, 64, 64, 4)),
    192: ((None, 64, 64, 4),),
}

# XCORE1000 direct-forward traits observed in ``run_mha_fwd``.  In the D192
# branch, the binary loads Flash_fwd_params+0x268 (the max_seqlen_q field) and
# selects the 128x64/8-warp trait only when that value is exactly 128; all
# other maximum query lengths use the 64x64/4-warp trait.  ``q_len`` below is
# the same canonical maximum-query field, not a rounded duration proxy.  This
# is a discrete launch fact, not a duration-derived tuning rule.
_BINARY_PREFILL_TRAITS = {
    192: {
        128: (128, 64, 8),
        None: (64, 64, 4),
    },
}
_PAGED_LAYOUTS = frozenset({"paged", "paged_varlen", "chunked_prefill"})
_SUPPORTED_LAYOUTS = _PAGED_LAYOUTS | {"contiguous"}
_TRAIT_DIRECT_PREFIX = "metax_flash_attention_trait_direct:"
_Q64_PACKED_PREFILL_ALGORITHM = (
    "metax_flash_attention_trait_packed:prefill_q64_full_tile")
_GRAPH_RUNTIME_MODES = frozenset({"NONE", "FULL", "PIECEWISE"})
# ``fused`` is one whole backend call (Kernel Dataset and pure-decode steps);
# ``prefill``/``decode`` name the two pathways of the vLLM-MetaX split forward
# selected by ``select_flash_attn_forward_partition``.
_FORWARD_ROLES = frozenset({"fused", "prefill", "decode"})
# vllm_metax 0.24 ``FlashAttentionMetadataBuilder.reorder_batch_threshold``
# (v1/attention/backends/flash_attn.py:424): without speculative decoding a
# request whose query length is at most one takes the decode pathway.
_SPLIT_FORWARD_DECODE_QUERY_THRESHOLD = 1


def normalize_flash_attn_execution_attributes(
        shape: Mapping[str, int],
        attributes: Mapping[str, object]) -> Mapping[str, object]:
    """Normalize MetaX FlashAttention execution facts at the ABI boundary."""

    del shape
    normalized = dict(attributes)
    # Kernel Dataset calls do not execute inside a captured graph.  Keep that
    # explicitly normalized execution fact separate from the dataset's
    # workload identity; Serving supplies its resolved GraphTokenShape mode.
    if "graph_runtime_mode" not in normalized:
        normalized["graph_runtime_mode"] = "NONE"
    graph_runtime_mode = normalized["graph_runtime_mode"]
    if graph_runtime_mode not in _GRAPH_RUNTIME_MODES:
        raise ValueError(
            "MetaX C500 FLASH_ATTN graph_runtime_mode must be one of "
            f"{sorted(_GRAPH_RUNTIME_MODES)!r}")
    # Only the Serving split-forward lowering names a pathway; every other
    # caller is one whole backend call.
    if "flash_attn_forward_role" not in normalized:
        normalized["flash_attn_forward_role"] = "fused"
    if normalized["flash_attn_forward_role"] not in _FORWARD_ROLES:
        raise ValueError(
            "MetaX C500 FLASH_ATTN flash_attn_forward_role must be one of "
            f"{sorted(_FORWARD_ROLES)!r}")
    if normalized["kv_layout"] == "mla":
        if normalized["causal"] is not False:
            raise ValueError(
                "canonical MetaX MLA input must retain causal=False before "
                "source execution normalization")
        normalized["causal"] = True
    return normalized


def _require_mla_decode_contract(
        shape: Mapping[str, int], dtype: str, variant: str,
        attributes: Mapping[str, object]):
    require_dtype(dtype, ("bf16",))
    batch, q_len, _, head_dim, query_heads, kv_heads = require_shape(
        shape, (
            "batch", "q_len", "kv_len", "head_dim",
            "num_query_heads", "num_kv_heads"))
    if variant != "decode":
        raise NotImplementedError(
            "MetaX source-bound MLA supports decode only")
    if (q_len, head_dim, query_heads, kv_heads) != (1, 512, 64, 1):
        raise NotImplementedError(
            "MetaX source-bound MLA requires Q1/D512/HQ64/HKV1")
    if attributes["kv_layout"] != "mla":
        raise ValueError("MetaX MLA execution requires kv_layout='mla'")
    if attributes["causal"] is not True:
        raise ValueError(
            "MetaX FlashMLA host ABI executes with causal=True")
    q_lens = tuple(attributes["q_lens"])
    kv_lens = tuple(attributes["kv_lens"])
    if len(q_lens) != batch or len(kv_lens) != batch:
        raise ValueError("MetaX MLA request lengths disagree with batch")
    if any(length != 1 for length in q_lens):
        raise ValueError("MetaX MLA decode requires one query per request")
    return batch, kv_lens, derive_mla_scheduler_work(kv_lens)


def select_flash_attn_setup_launches(
        shape: Mapping[str, int], dtype: str, variant: str,
        attributes: Mapping[str, object], *, hardware
) -> tuple[FlashAttentionSetupLaunch, ...]:
    """Rebuild the setup launches the C500 backend issues before its FA call.

    FlashMLA rebuilds its scheduler metadata and a bool fill.  The paged
    varlen prefill pathway of the vLLM-MetaX split forward is preceded by
    one CUB device scan -- ``DeviceScanInitKernel`` then ``DeviceScanKernel``,
    each one 128-thread CTA -- on every attention layer of every canonical
    prefill and mixed alignment phase, and on none of the decode phases.
    The scan runs inside the vendor FlashAttention extension, so the
    emission is bound to those discrete launch facts rather than to a
    source line.  Both kernels touch only the per-request prefix array of
    the call (``requests + 1`` int32 values): the init kernel writes the
    single tile's lookback state and the scan reads and writes the array,
    which prices them as a write-only and a read/write elementwise task on
    the observed geometry.  No duration participates.
    """

    del hardware
    if attributes["kv_layout"] != "mla":
        if (attributes["kv_layout"] not in _PAGED_LAYOUTS
                or attributes["flash_attn_forward_role"] != "prefill"):
            return ()
        prefix_values = len(tuple(attributes["q_lens"])) + 1
        return tuple(
            FlashAttentionSetupLaunch(
                variant="elementwise",
                dtype="int32",
                shape={"element_count": prefix_values},
                attributes={"reads": reads, "writes": 1,
                            "operation": operation},
                launch=LaunchConfig(
                    algorithm=f"observed_cub_device_scan:{kernel}",
                    grid=(1, 1, 1),
                    block=(128, 1, 1),
                    execution_variant="elementwise",
                ),
            )
            for kernel, operation, reads in (
                ("scan_init", "fill", 0), ("scan", "copy", 1)))
    batch, _, work = _require_mla_decode_contract(
        shape, dtype, variant, attributes)
    metadata_shape = {
        "batch": batch,
        "kv_block_count": work.kv_block_count,
        "scheduler_partitions": METAX_MLA_SCHEDULER_PARTITIONS,
        "scheduler_metadata_ints": METAX_MLA_SCHEDULER_PARTITIONS * 8,
        "split_metadata_ints": batch + 1,
        "scheduler_segment_count": work.segment_count,
        "scheduler_worker_run_count": (
            work.segment_count - work.same_worker_transitions),
    }
    metadata = FlashAttentionSetupLaunch(
        variant="flash_mla_metadata",
        dtype="int32",
        shape=metadata_shape,
        attributes={},
        launch=LaunchConfig(
            algorithm="observed_flash_mla_metadata:flash_mla_metadata",
            grid=(1, 1, 1),
            block=(32, 1, 1),
            execution_variant="flash_mla_metadata",
        ),
    )
    fill = FlashAttentionSetupLaunch(
        variant="elementwise",
        dtype="uint8",
        shape={"element_count": sum(tuple(attributes["q_lens"]))},
        attributes={"reads": 0, "writes": 1, "operation": "fill"},
        launch=LaunchConfig(
            algorithm="observed_flash_mla_bool_fill:elementwise",
            grid=(1, 1, 1),
            block=(256, 1, 1),
            execution_variant="elementwise",
        ),
    )
    return metadata, fill


def _mla_parameters() -> Mapping[str, float]:
    return {
        "unpadded_flops": 1.0,
        "mma_smem_bytes_per_flop": 0.0,
        "split_scratch_dram_bytes_per_element": 0.0,
        "page_table_bytes_per_kv_panel": 0.0,
    }


def _mla_main_launch(
        shape: Mapping[str, int], dtype: str, execution_variant: str,
        attributes: Mapping[str, object]) -> LaunchConfig:
    _, _, work = _require_mla_decode_contract(
        shape, dtype, execution_variant, attributes)
    splits = max(work.splits_per_request)
    return LaunchConfig(
        algorithm="observed_metax_flash_attention_persistent_queue",
        grid=(1, 1, METAX_MLA_SCHEDULER_PARTITIONS),
        block=(512, 1, 1),
        tiling=TilingSpec(tile_shape={}, num_kv_splits=splits),
        parameters=_mla_parameters(),
        execution_variant="decode",
    )


def _mla_combine_launch(
        shape: Mapping[str, int], dtype: str,
        attributes: Mapping[str, object]) -> LaunchConfig:
    batch, _, work = _require_mla_decode_contract(
        shape, dtype, "decode", attributes)
    splits = max(work.splits_per_request)
    if require_positive_integer(attributes, "num_kv_splits") != splits:
        raise ValueError(
            "MetaX MLA combine split count disagrees with source scheduler")
    query_heads = shape["num_query_heads"]
    block_m = metax_mla_combine_block_m(batch, 1, query_heads)
    return LaunchConfig(
        algorithm="observed_metax_flash_mla_combine",
        grid=(ceil_div(batch * query_heads, block_m), 1, 1),
        block=(256, 1, 1),
        tiling=TilingSpec(tile_shape={}, num_kv_splits=splits),
        parameters=_mla_parameters(),
        execution_variant="combine",
    )


def _parameters(
    splits: int, *, page_table_bytes_per_kv_panel: float,
) -> Mapping[str, float]:
    return {
        "unpadded_flops": 0.0,
        "mma_smem_bytes_per_flop": 0.0,
        "split_scratch_dram_bytes_per_element": (
            4.0 if splits > 1 else 0.0),
        "page_table_bytes_per_kv_panel": page_table_bytes_per_kv_panel,
    }


def _effective_decode_shape(
    q_len: int, query_heads: int, kv_heads: int,
) -> tuple[int, int]:
    if q_len != 1:
        raise ValueError("MetaX C500 FLASH_ATTN decode requires q_len 1")
    if query_heads % kv_heads:
        raise ValueError(
            "MetaX C500 FLASH_ATTN decode requires integral GQA head groups")
    return query_heads // kv_heads, kv_heads


def _split_tile(head_dim: int, effective_q: int) -> tuple[int, int, int]:
    """Return the XCORE1000 trait selected by binary ``get_tile_size``."""

    if head_dim not in _BINARY_SPLIT_TILE_RULES:
        raise NotImplementedError(
            "MetaX C500 binary split dispatch has no trait for head_dim "
            f"{head_dim}")
    if isinstance(effective_q, bool) or not isinstance(effective_q, int):
        raise TypeError("MetaX C500 FLASH_ATTN effective_q must be an integer")
    if effective_q <= 0:
        raise ValueError(
            "MetaX C500 FLASH_ATTN effective_q must be positive")
    for max_effective_q, block_m, block_n, warps in (
            _BINARY_SPLIT_TILE_RULES[head_dim]):
        if (max_effective_q is None
                or effective_q <= max_effective_q):
            return block_m, block_n, warps
    raise AssertionError(
        f"unhandled binary split tile rule for head_dim {head_dim}")


def _main_trait(
    *,
    batch: int,
    q_len: int,
    head_dim: int,
    query_heads: int,
    kv_heads: int,
    kv_layout: str,
    execution_variant: str,
    splits: int,
) -> tuple[int, int, int]:
    """Reproduce the pinned MACA 3.7.2 forward trait dispatch."""

    if execution_variant == "decode":
        effective_q = query_heads // kv_heads
        if (head_dim not in (128,)
                or kv_layout in _PAGED_LAYOUTS or splits > 1):
            return _split_tile(head_dim, effective_q)
        if batch == 1 or query_heads == kv_heads:
            return 128, 64, 4
        return 64, 64, 4
    if execution_variant != "prefill":
        raise AssertionError(
            f"unhandled FLASH_ATTN execution variant: {execution_variant!r}")
    binary_prefill = _BINARY_PREFILL_TRAITS.get(head_dim)
    if binary_prefill is not None:
        return binary_prefill.get(q_len, binary_prefill[None])
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(
            "MetaX C500 binary forward dispatch has no trait for head_dim "
            f"{head_dim}")
    if head_dim == 64:
        return 64, 64, 4
    if kv_layout == "contiguous":
        return 128, 64, 4
    if batch == 1 and q_len >= 1024:
        return 128, 64, 4
    return 64, 64, 4


def _native_graph_decode_splits(
    *,
    batch: int,
    effective_q: int,
    effective_heads: int,
    head_dim: int,
    kv_len: int,
    kv_layout: str,
    hardware,
) -> int:
    """Port the pinned FA2 SplitKV launch choice for C500 Graph replay.

    MACA 3.7.2's ``flash_splitkv.cpp::compute_params_numsplits`` invokes
    FA2's source-defined 80%-occupancy / 85%-of-best-wave heuristic when its
    caller supplies ``num_splits=0``.  vLLM supplies that native-auto value
    for the paged Full-Graph FA2 path.  The inputs below are source launch
    quantities only: GQA-effective head groups, source tile geometry, and AP
    capacity.  No Kernel Dataset or Serving duration participates.
    """

    if kv_layout not in _PAGED_LAYOUTS:
        return 1
    block_m, block_n, _ = _main_trait(
        batch=batch,
        q_len=1,
        head_dim=head_dim,
        query_heads=effective_q * effective_heads,
        kv_heads=effective_heads,
        kv_layout=kv_layout,
        execution_variant="decode",
        splits=1,
    )
    if head_dim == 128:
        blocks_per_ap = 8 if effective_q <= 16 else (
            4 if effective_q <= 32 else 2)
    elif head_dim == 64:
        blocks_per_ap = 16 if effective_q <= 16 else 4
    else:
        raise NotImplementedError(
            "MetaX C500 binary graph SplitKV occupancy facts are not "
            f"resolved for head_dim {head_dim}")
    work_items = batch * effective_heads * ceil_div(effective_q, block_m)
    launch_slots = hardware.num_compute_units * blocks_per_ap
    kv_panels = ceil_div(kv_len, block_n)
    return select_analytical_gpu_flash_attention_splits(
        work_items, launch_slots, kv_panels, max_splits=128)


def _main_launch(
    shape: Mapping[str, int],
    dtype: str,
    execution_variant: str,
    attributes: Mapping[str, object],
    hardware,
) -> LaunchConfig:
    flash_attn_workload_key(shape, dtype, execution_variant, attributes)
    batch, q_len, kv_len, head_dim, query_heads, kv_heads = require_shape(
        shape, (
            "batch", "q_len", "kv_len", "head_dim",
            "num_query_heads", "num_kv_heads"))
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(
            "MetaX C500 binary FLASH_ATTN supports head_dim values "
            f"{sorted(_SUPPORTED_HEAD_DIMS)}")
    if query_heads % kv_heads:
        raise ValueError(
            "MetaX C500 FLASH_ATTN requires integral GQA head groups")
    kv_layout = attributes["kv_layout"]
    if kv_layout not in _SUPPORTED_LAYOUTS:
        raise NotImplementedError(
            f"unsupported MetaX C500 FLASH_ATTN kv_layout: {kv_layout!r}")

    if execution_variant == "decode":
        effective_q, effective_heads = _effective_decode_shape(
            q_len, query_heads, kv_heads)
        split_policy = attributes["split_policy"]
        if split_policy == "forced":
            splits = require_positive_integer(attributes, "num_kv_splits")
            if splits > 128:
                raise ValueError(
                    "MetaX C500 FLASH_ATTN supports at most 128 KV splits")
        elif split_policy == "serving":
            splits = (
                _native_graph_decode_splits(
                    batch=batch,
                    effective_q=effective_q,
                    effective_heads=effective_heads,
                    head_dim=head_dim,
                    kv_len=kv_len,
                    kv_layout=kv_layout,
                    hardware=hardware,
                )
                if (attributes["graph_runtime_mode"] == "FULL"
                    or attributes["flash_attn_forward_role"] == "decode")
                else 1)
        else:
            raise ValueError(
                "MetaX C500 FLASH_ATTN split_policy must be 'forced' or "
                "'serving'")
    elif execution_variant == "prefill":
        effective_q = q_len
        effective_heads = query_heads
        split_policy = attributes["split_policy"]
        if not isinstance(split_policy, str) or not split_policy.strip():
            raise TypeError(
                "MetaX C500 FLASH_ATTN split_policy must be a non-empty "
                "string")
        if split_policy == "forced":
            splits = require_positive_integer(attributes, "num_kv_splits")
            if splits != 1:
                raise ValueError(
                    "MetaX C500 FLASH_ATTN prefill requires one KV split")
        elif split_policy == "serving":
            splits = 1
        else:
            raise ValueError(
                "MetaX C500 FLASH_ATTN split_policy must be 'forced' or "
                "'serving'")
    else:
        raise AssertionError(
            f"unhandled FLASH_ATTN execution variant: {execution_variant!r}")

    block_m, block_n, warps = _main_trait(
        batch=batch,
        q_len=q_len,
        head_dim=head_dim,
        query_heads=query_heads,
        kv_heads=kv_heads,
        kv_layout=kv_layout,
        execution_variant=execution_variant,
        splits=splits,
    )
    num_m_blocks = ceil_div(effective_q, block_m)
    uses_split_kernel = kv_layout in _PAGED_LAYOUTS or splits > 1
    if execution_variant == "prefill":
        if uses_split_kernel:
            grid = (effective_heads, num_m_blocks, batch)
        else:
            grid = (effective_heads, batch, num_m_blocks)
    elif head_dim == 64:
        if splits > 1:
            grid = (effective_heads * batch, splits, num_m_blocks)
        else:
            grid = (num_m_blocks, effective_heads, batch)
    elif splits > 1:
        grid = (num_m_blocks, effective_heads * batch, splits)
    else:
        grid = (effective_heads, batch, num_m_blocks)

    tile_shape = {"q": block_m, "kv": block_n, "head": head_dim}
    if block_m == 16:
        tile_shape["q"] = (
            32 if execution_variant == "prefill" and batch == 1 else 16)
        tile_shape["query_heads_per_workgroup"] = 4
    elif (execution_variant == "prefill"
          and kv_layout == "chunked_prefill" and block_m == 64):
        tile_shape["query_heads_per_workgroup"] = 1

    q_lens = tuple(attributes["q_lens"])
    kv_lens = tuple(attributes["kv_lens"])
    if (execution_variant == "prefill" and block_m == 16
            and kv_layout == "paged"):
        algorithm = (
            "metax_flash_attention_grouped_direct"
            if batch == 1 else
            "metax_flash_attention_persistent_queue")
    elif (execution_variant == "prefill"
          and kv_layout == "chunked_prefill" and block_m == 16):
        algorithm = "metax_flash_attention_packed_varlen_queue"
    elif (execution_variant == "prefill"
          and (block_m, block_n) == (64, 64)
          and kv_layout == "chunked_prefill"):
        algorithm = _Q64_PACKED_PREFILL_ALGORITHM
    elif (execution_variant == "prefill"
          and (block_m, block_n) == (64, 64)
          and kv_layout in ("paged", "paged_varlen")
          and attributes["causal"] is True
          and q_len % 64 == 0
          and all(q == kv and q % 64 == 0
                  for q, kv in zip(q_lens, kv_lens))):
        algorithm = (
            f"{_TRAIT_DIRECT_PREFIX}prefill_q64_full_tile")
    elif (execution_variant == "prefill"
          and (block_m, block_n) == (128, 64)
          and attributes["causal"] is True
          and q_len % 128 == 0
          and all(q % 128 == 0 and kv >= q and kv % 64 == 0
                  for q, kv in zip(q_lens, kv_lens))):
        suffix = (
            "prefill_q128_full_tile"
            if kv_layout == "contiguous" and all(
                q == kv for q, kv in zip(q_lens, kv_lens)) else
            "prefill_q128_split_full_tile")
        algorithm = f"{_TRAIT_DIRECT_PREFIX}{suffix}"
    elif (execution_variant == "decode"
          and (block_m, block_n) == (64, 64)
          and kv_layout == "contiguous"
          and attributes["causal"] is False
          and splits == 1
          and all(q == 1 for q in q_lens)
          and all(kv % 64 == 0 for kv in kv_lens)):
        algorithm = f"{_TRAIT_DIRECT_PREFIX}decode_q64_full_tile"
    else:
        algorithm = f"{_TRAIT_DIRECT_PREFIX}{execution_variant}"

    return LaunchConfig(
        algorithm=algorithm,
        grid=grid,
        block=(warps * 64, 1, 1),
        tiling=TilingSpec(
            tile_shape=tile_shape,
            num_kv_splits=splits,
        ),
        parameters=_parameters(
            splits,
            page_table_bytes_per_kv_panel=(
                4.0 if kv_layout in _PAGED_LAYOUTS and splits > 1 else 0.0),
        ),
        execution_variant=execution_variant,
    )


def _combine_launch(
    shape: Mapping[str, int],
    dtype: str,
    attributes: Mapping[str, object],
    hardware,
) -> LaunchConfig:
    batch, q_len, _, head_dim, query_heads, kv_heads = require_shape(
        shape, (
            "batch", "q_len", "kv_len", "head_dim",
            "num_query_heads", "num_kv_heads"))
    base_variant = "decode" if q_len == 1 else "prefill"
    normalized_attributes = dict(attributes)
    normalized_attributes["is_decode"] = base_variant == "decode"
    expected_main = _main_launch(
        shape, dtype, base_variant, normalized_attributes, hardware)
    splits = require_positive_integer(attributes, "num_kv_splits")
    if splits != expected_main.tiling.num_kv_splits:
        raise ValueError(
            "MetaX C500 FLASH_ATTN combine split count disagrees with the "
            "source-derived main launch")
    if splits <= 1:
        raise ValueError(
            "MetaX C500 FLASH_ATTN combine requires more than one KV split")

    effective_q, effective_heads = _effective_decode_shape(
        q_len, query_heads, kv_heads)
    combine_block_m_min = 1024 // head_dim
    max_block_count = ceil_div(
        batch * effective_heads * effective_q, combine_block_m_min)
    combine_block_m = (
        combine_block_m_min
        if (max_block_count <= 512
            or effective_q <= combine_block_m_min)
        else combine_block_m_min * 2)
    grid = (
        ceil_div(batch * effective_heads * effective_q, combine_block_m),
        1,
        1,
    )
    parameters = dict(_parameters(
        splits, page_table_bytes_per_kv_panel=0.0))
    parameters["unpadded_flops"] = 1.0
    return LaunchConfig(
        algorithm="metax_flash_attn_lut",
        grid=grid,
        block=(256, 1, 1),
        tiling=TilingSpec(tile_shape={}, num_kv_splits=splits),
        parameters=parameters,
        execution_variant="combine",
    )


def select_flash_attn_launch(
    shape: Mapping[str, int],
    dtype: Optional[str],
    variant: Optional[str],
    attributes: Mapping[str, object],
    hardware,
) -> LaunchConfig:
    """Select an independent analytical or source-derived FA launch."""

    normalized_attributes = normalize_flash_attn_execution_attributes(
        shape, attributes)
    execution_variant = require_variant(
        variant, ("prefill", "decode", "combine", "split_combine"),
        "FLASH_ATTN")
    normalized_dtype = require_dtype(dtype, ("bf16",))
    policy = launch_policy(normalized_attributes)
    if normalized_attributes["kv_layout"] == "mla":
        if policy != "injected_launch":
            raise NotImplementedError(
                "MetaX source-bound MLA requires injected_launch")
        if execution_variant in ("combine", "split_combine"):
            return _mla_combine_launch(
                shape, normalized_dtype, normalized_attributes)
        return _mla_main_launch(
            shape, normalized_dtype, execution_variant, normalized_attributes)
    if policy == "analytical_launch":
        if execution_variant not in ("prefill", "decode"):
            raise NotImplementedError(
                "MetaX C500 analytical FLASH_ATTN does not synthesize a "
                "split-combine launch")
        flash_attn_workload_key(
            shape, normalized_dtype, execution_variant, normalized_attributes)
        batch, q_len, _, head_dim, query_heads, kv_heads = require_shape(
            shape, (
                "batch", "q_len", "kv_len", "head_dim",
                "num_query_heads", "num_kv_heads"))
        if head_dim != 128:
            raise NotImplementedError(
                "MetaX C500 analytical FLASH_ATTN currently supports "
                "head_dim 128")
        if query_heads % kv_heads:
            raise ValueError(
                "MetaX C500 analytical FLASH_ATTN requires integral GQA "
                "head groups")
        logical_workgroups = batch * query_heads * q_len
        return LaunchConfig(
            algorithm="maca_flash_attention_analytical",
            grid=(min(
                int(hardware.num_compute_units), logical_workgroups), 1, 1),
            block=(256, 1, 1),
            tiling=TilingSpec(tile_shape={}, num_kv_splits=1),
            parameters={
                "unpadded_flops": 1.0,
                "mma_smem_bytes_per_flop": 0.0,
                "split_scratch_dram_bytes_per_element": 0.0,
                "page_table_bytes_per_kv_panel": 0.0,
            },
            execution_variant=execution_variant,
        )
    if policy != "injected_launch":
        raise AssertionError(f"unhandled launch policy: {policy!r}")
    if execution_variant in ("combine", "split_combine"):
        return _combine_launch(
            shape, normalized_dtype, normalized_attributes, hardware)
    return _main_launch(
        shape, normalized_dtype, execution_variant, normalized_attributes,
        hardware)


def select_flash_attn_forward_partition(
        shape: Mapping[str, int], variant: str,
        attributes: Mapping[str, object]) -> tuple:
    """Partition one logical attention call into the vLLM-MetaX split forward.

    ``VLLM_METAX_ENABLE_FA_SPLIT_FORWARD`` defaults to 1 (vllm_metax 0.24
    ``envs.py:76``), so ``FlashAttentionImpl.forward``
    (``v1/attention/backends/flash_attn.py:1179-1230``) runs the batch that
    ``split_decodes_and_prefills`` orders decode-first as two backend calls:

    - prefill requests through ``flash_attn_varlen_func``, whose freshly
      allocated result is slice-assigned into
      ``output[num_decode_tokens:num_actual_tokens]``: one contiguous D2D
      copy of every prefill query row;
    - decode requests through ``flash_attn_with_kvcache`` with its default
      ``num_splits=0`` native split heuristic.  Its output write-back is the
      decode copy already selected by ``select_flash_attn_copy_launches``.

    A pure-decode call is already that single decode pathway and is left
    whole.  ``q_lens`` is the request ABI the backend call sees: the logical
    requests for eager and PIECEWISE attention, the padded model-input
    requests inside a FULL Graph replay.  Only static request lengths
    participate; no duration is consumed.
    """

    execution_variant = require_variant(
        variant, ("prefill", "decode"), "FLASH_ATTN")
    if attributes["kv_layout"] == "mla" or execution_variant == "decode":
        return ()
    if attributes["kv_layout"] not in _PAGED_LAYOUTS:
        return ()
    head_dim, query_heads = require_shape(
        shape, ("head_dim", "num_query_heads"))
    q_lens = attributes["q_lens"]
    if (not isinstance(q_lens, tuple) or not q_lens
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in q_lens)):
        raise ValueError(
            "MetaX C500 split forward requires positive integer q_lens")
    decode_indexes = tuple(
        index for index, value in enumerate(q_lens)
        if value <= _SPLIT_FORWARD_DECODE_QUERY_THRESHOLD)
    prefill_indexes = tuple(
        index for index, value in enumerate(q_lens)
        if value > _SPLIT_FORWARD_DECODE_QUERY_THRESHOLD)
    parts = []
    if prefill_indexes:
        prefill_rows = sum(q_lens[index] for index in prefill_indexes)
        parts.append(FlashAttentionForwardPart(
            role="prefill",
            variant="prefill",
            request_indexes=prefill_indexes,
            output_copy_elements=prefill_rows * query_heads * head_dim,
            output_copy_launch=LaunchConfig(
                algorithm="native_d2d_flash_attention_copy",
                grid=(ceil_div(prefill_rows * head_dim, 64), 1, 1),
                block=(512, 1, 1),
                execution_variant="elementwise",
            ),
        ))
    if decode_indexes:
        parts.append(FlashAttentionForwardPart(
            role="decode",
            variant="decode",
            request_indexes=decode_indexes,
            output_copy_elements=0,
            output_copy_launch=None,
        ))
    return tuple(parts)


def select_flash_attn_copy_launches(
        shape: Mapping[str, int], attributes: Mapping[str, object],
        hardware) -> tuple[tuple[str, str, int, LaunchConfig], ...]:
    """Select MACA's transpose12 output copy emitted after decode."""

    del hardware
    batch, q_len, head_dim, query_heads, kv_heads = require_shape(
        shape, (
            "batch", "q_len", "head_dim", "num_query_heads",
            "num_kv_heads"))
    if (q_len != 1 or query_heads == kv_heads
            or attributes["kv_layout"] not in (
                "contiguous", "paged", "paged_varlen")):
        return ()
    element_count = batch * query_heads * head_dim
    launch = LaunchConfig(
        algorithm="native_d2d_flash_attention_copy",
        grid=(ceil_div(batch * q_len * head_dim, 64), 1, 1),
        block=(512, 1, 1),
        execution_variant="elementwise",
    )
    return (("after", "bf16", element_count, launch),)


# Every canonical C500 serving alignment phase (prefill, mixed, decode)
# launches one ``vectorized_elementwise_kernel`` nullary fill per attention
# layer, over the whole output buffer of the call, with a 1024-thread CTA
# covering 8192 values.  These are discrete launch facts of the versioned
# observation; no duration participates.
select_flash_attn_output_fill_launches = _skeleton.flash_attn_output_fill_selector(
    values_per_cta=8192, threads_per_cta=1024)
