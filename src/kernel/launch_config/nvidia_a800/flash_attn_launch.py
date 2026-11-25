"""NVIDIA A800 FlashAttention split-KV launch selection."""

from typing import Mapping, Optional

from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.selection_primitives import (
    ceil_div, require_boolean, require_dtype, require_positive_integer,
    require_shape, require_variant,
    select_analytical_gpu_flash_attention_splits,
)
from src.kernel.model_contract import (
    SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER,
)
from src.kernel.launch_config import hardware_skeleton as _skeleton


_DENSE_FA2_TILES = {
    64: (128, 128),
    128: (128, 64),
}
_SPLIT_FA2_TILES = {
    64: (64, 256),
    128: (64, 128),
}
_PAGED_LAYOUTS = {"paged", "paged_varlen", "chunked_prefill"}


# A800 forwards the public FlashAttention attributes unchanged.
normalize_flash_attn_execution_attributes = (
    _skeleton.normalize_flash_attn_execution_attributes)


# A800 FlashAttention has no separate host-side setup kernel.
select_flash_attn_setup_launches = _skeleton.select_flash_attn_setup_launches


# A800 FlashAttention runs one fused varlen call per layer.
select_flash_attn_forward_partition = (
    _skeleton.select_flash_attn_forward_partition)


def select_flash_attn_copy_launches(
        shape: Mapping[str, int], attributes: Mapping[str, object],
        hardware) -> tuple[tuple[str, str, int, LaunchConfig], ...]:
    """Select the three PyTorch TensorIterator copies around paged FA2."""

    del hardware
    batch, q_len, head_dim, query_heads = require_shape(
        shape, ("batch", "q_len", "head_dim", "num_query_heads"))
    kv_layout = attributes["kv_layout"]
    kv_heads, = require_shape(shape, ("num_kv_heads",))
    if (q_len != 1 or batch <= 1 or query_heads == kv_heads
            or kv_layout not in ("paged", "paged_varlen")):
        return ()

    def copy_launch(elements: int, values_per_thread: int,
                    tensor_rank: float) -> LaunchConfig:
        return LaunchConfig(
            algorithm="observed_flash_attention_copy:elementwise",
            grid=(ceil_div(elements, 128 * values_per_thread), 1, 1),
            block=(128, 1, 1),
            parameters={
                SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER: tensor_rank,
            },
            execution_variant="elementwise",
        )

    tensor_elements = batch * query_heads * head_dim
    lse_elements = batch * query_heads
    return (
        ("before", "bf16", tensor_elements,
         copy_launch(tensor_elements, 4, 4.0)),
        ("after", "bf16", tensor_elements,
         copy_launch(tensor_elements, 4, 4.0)),
        ("after", "float32", lse_elements,
         copy_launch(lse_elements, 2, 3.0)),
    )


# vLLM 0.11.0 ``Attention.forward`` zero-initializes the output buffer for
# backends accepting one (``vllm/attention/layer.py`` ``self.use_output``
# branch, ``FlashAttentionBackend.accept_output_buffer = True``), launching
# one PyTorch fill per attention call ahead of the framework copies.  A
# contiguous BF16 fill dispatches
# ``vectorized_elementwise_kernel<4, FillFunctor<BFloat16>>``: sm_80 clamps
# the vector width to four and the output-only iterator selects a thread
# work size of eight (PyTorch 2.8 ``CUDALoops.cuh``
# ``launch_vectorized_kernel``), so one 128-thread CTA covers 1024 values.
select_flash_attn_output_fill_launches = _skeleton.flash_attn_output_fill_selector(
    values_per_cta=1024, threads_per_cta=128)


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("power-of-two input must be positive")
    return 1 << (value - 1).bit_length()


def _select_dataset_mla_main_launch(
        shape: Mapping[str, int], dtype: str, execution_variant: str,
        attributes: Mapping[str, object]) -> LaunchConfig:
    """Rebuild the vLLM 0.11 A800 grouped-MLA stage-1 launch."""

    require_dtype(dtype, ("bf16",))
    batch, q_len, _, head_dim, query_heads, kv_heads = require_shape(
        shape, (
            "batch", "q_len", "kv_len", "head_dim",
            "num_query_heads", "num_kv_heads"))
    if execution_variant != "decode":
        raise NotImplementedError(
            "A800 source-bound MLA dataset supports decode only")
    if attributes["launch_policy"] != "injected_launch":
        raise NotImplementedError(
            "A800 MLA uses a two-stage Triton launch; generic analytical "
            "FLASH_ATTN must use the target MLA contract")
    if (q_len, head_dim, query_heads, kv_heads) != (1, 512, 64, 1):
        raise NotImplementedError(
            "A800 source-bound MLA dataset requires Q1/D512/HQ64/HKV1")
    if attributes["causal"] is not False:
        raise ValueError("A800 source-bound MLA dataset requires causal=False")
    if attributes["split_policy"] != "serving":
        raise ValueError(
            "A800 source-bound MLA dataset requires serving split policy")
    splits = 4
    return LaunchConfig(
        algorithm="triton_grouped_mla_stage1",
        grid=(batch, 4, splits),
        block=(128, 1, 1),
        tiling=TilingSpec(
            tile_shape={
                "query_heads": 16,
                "kv": 32,
                "head": 512,
                "qk_head": 576,
                "value_head": 512,
            },
            num_kv_splits=splits,
        ),
        parameters={
            "unpadded_flops": 0.0,
            "mma_smem_bytes_per_flop": 0.0,
            "split_scratch_dram_bytes_per_element": 0.0,
            "page_table_bytes_per_kv_panel": 0.0,
        },
        execution_variant="decode",
    )


def _select_mla_combine_launch(
        shape: Mapping[str, int], dtype: str,
        attributes: Mapping[str, object]) -> LaunchConfig:
    """Rebuild the Triton grouped-MLA stage-2 launch."""

    require_dtype(dtype, ("bf16",))
    batch, q_len, _, head_dim, query_heads, kv_heads = require_shape(
        shape, (
            "batch", "q_len", "kv_len", "head_dim",
            "num_query_heads", "num_kv_heads"))
    raw_q_lens = attributes["q_lens"]
    if (not isinstance(raw_q_lens, (list, tuple))
            or len(raw_q_lens) != batch
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in tuple(raw_q_lens))
            or max(raw_q_lens) != q_len):
        raise ValueError(
            "A800 grouped-MLA stage2 q_lens must contain one positive "
            "integer per request and agree with the shape")
    # The pinned vLLM Triton source (triton_decode_attention.py,
    # _decode_softmax_reducev_fwd) derives this launch as
    # grid = (batch, head_num), 4 warps, with BLOCK_DV =
    # next_power_of_2(Lv), where the source batch dimension is the query row
    # count (q.shape[0]): a multi-token verify window contributes one row per
    # window token.  The geometry is independent of the value head
    # dimension, which only sets the per-thread loop extent.  The
    # source-derived contract is therefore one query row per (row, head)
    # pair over the compressed single-KV-head MLA layout; the audited
    # deployments cover Lv 512 (GLM-5.2) and Lv 128 (Kimi-K2.6).
    if kv_heads != 1 or head_dim <= 0 or query_heads <= 0:
        raise NotImplementedError(
            "A800 grouped-MLA stage2 requires the single-KV-head MLA "
            f"layout; got head_dim={head_dim} "
            f"query_heads={query_heads} kv_heads={kv_heads}")
    splits = require_positive_integer(attributes, "num_kv_splits")
    return LaunchConfig(
        algorithm="triton_grouped_mla_stage2",
        grid=(sum(raw_q_lens), query_heads, 1),
        block=(128, 1, 1),
        tiling=TilingSpec(tile_shape={}, num_kv_splits=splits),
        parameters={
            "unpadded_flops": 1.0,
            "mma_smem_bytes_per_flop": 0.0,
            "split_scratch_dram_bytes_per_element": 4.0,
            "page_table_bytes_per_kv_panel": 0.0,
        },
        execution_variant="combine",
    )


def select_target_mla_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig:
    """Select the pinned vLLM 0.27.1 A800 MLA execution family.

    Prefill expands K/V to query heads and uses FlashAttention-2.  On SM80 its
    unequal 192/128 head dimensions take the source-defined V-padding path.
    Decode uses the Triton grouped MLA program with 16 query heads per CTA,
    32 KV rows per step, and the source split policy.
    """

    execution_variant = require_variant(
        variant, ("mla_prefill", "mla_decode"), "A800 target MLA")
    require_dtype(dtype, ("bf16",))
    batch, q_len, kv_len, query_heads, qk_head_dim, value_head_dim, kv_lora = (
        require_shape(shape, (
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "kv_lora_rank",
        )))
    if attributes["kv_layout"] != "mla":
        raise ValueError("A800 target MLA requires kv_layout='mla'")
    raw_q_lens = attributes["q_lens"]
    raw_kv_lens = attributes["kv_lens"]
    if (not isinstance(raw_q_lens, (list, tuple))
            or not isinstance(raw_kv_lens, (list, tuple))
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in tuple(raw_q_lens) + tuple(raw_kv_lens))):
        raise TypeError("A800 target MLA request lengths must be integers")
    q_lens = tuple(raw_q_lens)
    kv_lens = tuple(raw_kv_lens)
    if any(value <= 0 for value in q_lens + kv_lens):
        raise ValueError("A800 target MLA request lengths must be positive")
    if (len(q_lens) != batch or len(kv_lens) != batch
            or max(q_lens) != q_len or max(kv_lens) != kv_len):
        raise ValueError("A800 target MLA request lengths disagree with shape")
    if ("graph_model_input_tokens" in attributes
            and attributes["graph_model_input_tokens"] != sum(q_lens)):
        raise NotImplementedError(
            "A800 target MLA launch under CUDA-graph padding is not derived")

    parameters = {
        "unpadded_flops": 1.0,
        "mma_smem_bytes_per_flop": 0.0,
        "split_scratch_dram_bytes_per_element": 0.0,
        "page_table_bytes_per_kv_panel": 0.0,
    }
    if execution_variant == "mla_prefill":
        if (qk_head_dim, value_head_dim, kv_lora) != (192, 128, 512):
            raise NotImplementedError(
                "A800 Kimi MLA prefill requires QK=192, V=128, and "
                "KV-LoRA=512")
        return LaunchConfig(
            algorithm="vllm_flash_attn_v2_mla_prefill",
            grid=(ceil_div(q_len, 64), batch, query_heads),
            block=(256, 1, 1),
            tiling=TilingSpec(
                tile_shape={
                    "q": 64, "kv": 64,
                    "qk_head": 256, "value_head": 256,
                },
                num_kv_splits=1,
            ),
            parameters=parameters,
            execution_variant=execution_variant,
        )

    if (qk_head_dim, value_head_dim, kv_lora) != (576, 512, 512):
        raise NotImplementedError(
            "A800 Kimi MLA decode requires QK=576, V=512, and "
            "KV-LoRA=512")
    # The Triton grouped program (triton_decode_attention.py,
    # _decode_grouped_att_m_fwd) derives its grid batch dimension from
    # q.shape[0] -- one program row per query token, each with its own
    # causal sequence length -- so a multi-token MTP/EAGLE verify window
    # expands the row count instead of changing the per-CTA geometry.
    query_rows = sum(q_lens)
    ideal_splits = _next_power_of_two(max(1, kv_len // 512))
    splits = min(ideal_splits, int(hardware.num_compute_units) * 2)
    parameters["split_scratch_dram_bytes_per_element"] = (
        4.0 if splits > 1 else 0.0)
    return LaunchConfig(
        algorithm="vllm_triton_grouped_mla_decode",
        grid=(query_rows, ceil_div(query_heads, 16), splits),
        block=(128, 1, 1),
        tiling=TilingSpec(
            tile_shape={
                "query_heads": 16, "kv": 32,
                "qk_head": 576, "value_head": 512,
            },
            num_kv_splits=splits,
        ),
        parameters=parameters,
        execution_variant=execution_variant,
    )


def select_flash_attn_launch(shape: Mapping[str, int], dtype: Optional[str],
                             variant: Optional[str], attributes: Mapping[str, object],
                            hardware) -> LaunchConfig:
    execution_variant = require_variant(
        variant, ("prefill", "decode", "combine", "split_combine"),
        "FLASH_ATTN")
    normalized_dtype = require_dtype(dtype, ("bf16",))
    if execution_variant in ("combine", "split_combine"):
        if attributes["kv_layout"] == "mla":
            return _select_mla_combine_launch(
                shape, normalized_dtype, attributes)
        batch, head_dim, query_heads = require_shape(
            shape, ("batch", "head_dim", "num_query_heads"))
        splits = require_positive_integer(attributes, "num_kv_splits")
        kv_layout = attributes["kv_layout"]
        if not isinstance(kv_layout, str) or not kv_layout.strip():
            raise TypeError(
                "A800 FLASH_ATTN combine kv_layout must be non-empty")
        block = 128
        outputs_per_block = block * 4 // head_dim
        if outputs_per_block <= 0:
            raise ValueError(
                "FLASH_ATTN head_dim exceeds combine-kernel block capacity")
        return LaunchConfig(
            algorithm="flash_attn_splitkv_combine",
            grid=(ceil_div(batch * query_heads, outputs_per_block), 1, 1),
            block=(block, 1, 1),
            tiling=TilingSpec(tile_shape={}, num_kv_splits=splits),
            parameters={
                "unpadded_flops": 1.0,
                "mma_smem_bytes_per_flop": 0.0,
                "split_scratch_dram_bytes_per_element": (
                    4.0 if splits > 1 else 0.0),
                "page_table_bytes_per_kv_panel": 0.0,
            },
            execution_variant="combine",
        )
    if execution_variant not in ("prefill", "decode"):
        raise ValueError("split_combine is not a main FLASH_ATTN launch")
    batch, q_len, kv_len, head_dim, hq, hkv = require_shape(
        shape, (
            "batch", "q_len", "kv_len", "head_dim",
            "num_query_heads", "num_kv_heads",
        ))
    kv_layout = attributes["kv_layout"]
    if not isinstance(kv_layout, str) or not kv_layout.strip():
        raise TypeError("A800 FLASH_ATTN kv_layout must be a non-empty string")
    if kv_layout == "mla":
        return _select_dataset_mla_main_launch(
            shape, normalized_dtype, execution_variant, attributes)
    if kv_layout != "contiguous" and kv_layout not in _PAGED_LAYOUTS:
        raise ValueError(
            f"unsupported A800 FLASH_ATTN kv_layout: {kv_layout!r}")
    if head_dim not in _SPLIT_FA2_TILES:
        raise ValueError(
            "A800 source-closed FA2 launch supports head_dim 64 or 128, "
            f"got {head_dim}")
    num_sms = int(hardware.num_compute_units)
    split_tile_q, split_tile_kv = _SPLIT_FA2_TILES[head_dim]
    num_n_blocks = ceil_div(kv_len, split_tile_kv)
    is_decode = require_boolean(attributes, "is_decode")
    if is_decode != (execution_variant == "decode"):
        raise ValueError("FLASH_ATTN is_decode conflicts with variant")
    split_policy = attributes["split_policy"]
    if not isinstance(split_policy, str) or not split_policy.strip():
        raise TypeError(
            "A800 FLASH_ATTN split_policy must be a non-empty string")
    if split_policy not in ("forced", "serving"):
        raise ValueError(
            "A800 FLASH_ATTN split_policy must be 'forced' or 'serving'")
    if is_decode and hq % hkv:
        raise ValueError(
            "A800 FLASH_ATTN decode requires integral GQA head groups")
    effective_q_len = hq // hkv if is_decode else q_len
    launch_heads = hkv if is_decode else hq
    if split_policy == "forced":
        splits = require_positive_integer(attributes, "num_kv_splits")
        if not is_decode and splits != 1:
            raise ValueError(
                "A800 FLASH_ATTN prefill requires one forced KV split")
    elif (is_decode and q_len == 1 and kv_layout in _PAGED_LAYOUTS
          and hq == hkv):
        # vLLM's FA2 paged-MHA dispatch does not launch the split-KV
        # reduction path.  Paged GQA continues to use the analytical split
        # selection below because its query heads are grouped by KV head.
        splits = 1
    elif is_decode:
        splits = select_analytical_gpu_flash_attention_splits(
            batch * launch_heads * ceil_div(
                effective_q_len, split_tile_q),
            num_sms * 2,
            num_n_blocks,
        )
    else:
        splits = 1
    uses_split_kernel = kv_layout in _PAGED_LAYOUTS or splits > 1
    if uses_split_kernel:
        tile_q, tile_kv = split_tile_q, split_tile_kv
        algorithm = (
            "vllm_flash_attn_splitkv"
            if kv_layout in _PAGED_LAYOUTS else "flash_attn_splitkv")
    else:
        tile_q, tile_kv = _DENSE_FA2_TILES[head_dim]
        algorithm = "flash_attn_dense"
    num_m_blocks = ceil_div(effective_q_len, tile_q)
    if splits > 1:
        grid = (num_m_blocks, splits, batch * hkv)
    else:
        grid = (num_m_blocks, batch, launch_heads)
    return LaunchConfig(
        algorithm=algorithm,
        grid=grid,
        block=(128, 1, 1),
        tiling=TilingSpec(
            tile_shape={"q": tile_q, "kv": tile_kv, "head": head_dim},
            num_kv_splits=splits,
        ),
        # FA2 2.8.3 SM80 uses m16n8k16 MMA.  QK reads A+B from SMEM and
        # gemm_rs(P,V) keeps P in registers and reads only V from SMEM:
        # (768 + 256) bytes / (4096 + 4096) FLOPs = 1/8 byte/FLOP.
        parameters={
            "unpadded_flops": 0.0,
            "mma_smem_bytes_per_flop": 0.125,
            "split_scratch_dram_bytes_per_element": (
                4.0 if splits > 1 else 0.0),
            "page_table_bytes_per_kv_panel": (
                4.0 if kv_layout in ("paged", "paged_varlen")
                and splits > 1 else 0.0),
        },
        execution_variant=execution_variant,
    )
