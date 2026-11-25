"""NVIDIA H200 FlashAttention-3 source launch selection.

vLLM 0.24.0 on Hopper runs MLA attention through ``vllm.vllm_flash_attn``
FlashAttention-3, built from vllm-project/flash-attention at the revision
``cmake/external_projects/vllm_flash_attn.cmake`` pins
(``803020a8fa15407871341d41eba4919ade2ee1ee``).  Every discrete launch fact
below is a static fact of that source (``hopper/flash_api.cpp``,
``tile_size.h``, ``heuristics.h``, ``flash_fwd_launch_template.h``,
``flash_fwd_combine_launch_template.h``, ``flash_prepare_scheduler.cu``) and
of the vLLM 0.24.0 callers (``v1/attention/backends/mla/flashattn_mla.py``
for decode, the FLASH_ATTN MLA prefill backend for prefill):

* prefill: ``flash_attn_varlen_func(q[., H, 192], k[., H, 192],
  v[., H, 128], causal=True)`` with no ``num_splits`` -> the
  ``get_num_splits`` heuristic; ``tile_size_fwd_sm90(192, 128, causal)`` =
  (kBlockM 128, kBlockN 128, MmaPV_is_RS, IntraWGOverlap); two MMA
  warpgroups + one producer warpgroup (384 threads); varlen dynamic
  persistent scheduler (grid = SM count) with LPT order for causal.
* decode: absorbed MLA, ``q = q_pe`` (64), ``qv = q_nope`` (512),
  ``k = k_pe`` (64), ``v = kv_c`` (512), one KV head, paged cache,
  ``num_splits = flash_attn_max_num_splits_for_cuda_graph`` (32) when the
  deployment captures full CUDA graphs and the decode fits the capture
  size (``full_cuda_graphs`` / ``max_cudagraph_capture_size`` attributes),
  else 0 (heuristic); ``tile_size_fwd_sm90(64, 512)`` = (64, 64, SS,
  no intra-WG overlap); PackGQA is forced by ``num_splits > 1``; the paged
  cache uses cp.async (PagedKVNonTMA) because
  ``seqlen_q * H <= kBlockM``; 2 MMA warpgroups (LargeHeadDimV) + a
  128-thread producer warpgroup; ``seqlen_q == 1`` executes non-causal.
* combine (``num_splits > 1``): ``FlashAttnFwdCombine<(8, 128), log2
  max splits, 256 threads>``, varlen tile scheduler (STANDARD or
  LINEARIZE_M_AND_BATCH by tile density).
* prepare (``prepare_varlen_num_blocks_kernel<NumWarps>``): one CTA of
  ``32 * NumWarps`` threads, launched by every varlen call that does not
  carry precomputed scheduler metadata (prefill), and once per step by the
  decode metadata builder.

Owned MLA domains (``owns_*`` predicates, decided before selection): causal
fresh or noncausal context prefill with qk 192 / v 128 run unsplit; absorbed
MLA decode/verify (latent 512 + rope 64, uniform up to512 query tokens) and
its split combine. Multiple packed M tiles are retained. Wide combine
pricing still requires the shared source-memory guard to be widened.

The vLLM 0.30 ordinary BF16 D128 GQA fresh unsplit prefill branch is bound
separately to revision ``506341a143fcabd4bb79052a7605ada727d6b3f5``.
Its dynamic split heuristic rejects split/combine and one-MMA-warpgroup
branches; the first-stage predictor still reports partial CTA progress.
Canonical observations additionally bind the same revision's BF16 D64/D128
GQA main/combine templates and setup launches. Source call factories also
derive equal-head D256 calls, including GLM's expanded MLA prefill, and
larger KV pages using either cp.async or source-selected paged TMA. The
TMA algorithms carry the physical page extent in ``tile_shape["kv_page"]``.
Their predictor retains source-stage
ordering and explicit aggregate-capacity approximations; these extensions
do not establish a new measured service or native provider.

The selector records the source template in ``LaunchConfig.algorithm``
(``vllm_fa3_sm90_mla_prefill`` / ``vllm_fa3_sm90_mla_decode`` /
``vllm_fa3_sm90_combine``); the predictor re-derives the device-side tile
schedule from the same source semantics.
"""

import math
from typing import Mapping, Optional

from src.kernel.prediction_cache import RuntimeCache


from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.hardware_skeleton import (
    normalize_flash_attn_execution_attributes,
)
from src.kernel.launch_config.selection_primitives import (
    ceil_div, require_dtype, require_positive_integer, require_shape,
    require_variant,
)


FA3_SOURCE_REVISION = "803020a8fa15407871341d41eba4919ade2ee1ee"
FA3_MLA_PREFILL_027_SOURCE_REVISION = "28e862d21806bc3580207aa0ad4e2759151e9827"
# vLLM0.30 AttentionConfig default; an explicit caller scenario, not a rate.
FA3_GQA_DEFAULT_GRAPH_SPLITS_030 = 32
FA3_PREFILL_ALGORITHM = "vllm_fa3_sm90_mla_prefill"
FA3_PREFILL_SPLIT_ALGORITHM = "vllm_fa3_sm90_mla_prefill_split_partial"
FA3_PREFILL_COMBINE_ALGORITHM = "vllm_fa3_sm90_mla_prefill_combine_partial"
FA3_DECODE_ALGORITHM = "vllm_fa3_sm90_mla_decode"
FA3_COMBINE_ALGORITHM = "vllm_fa3_sm90_combine"
FA3_GQA_PREFILL_030_ALGORITHM = "vllm_030_fa3_sm90_gqa_prefill_partial"
FA3_GQA_OBSERVED_030_ALGORITHM = "vllm_030_fa3_sm90_gqa_observed_partial"
FA3_GQA_OBSERVED_SPLIT_030_ALGORITHM = "vllm_030_fa3_sm90_gqa_observed_split_partial"
FA3_GQA_COMBINE_030_ALGORITHM = "vllm_030_fa3_sm90_gqa_combine_partial"
FA3_GQA_TMA_030_ALGORITHM = "vllm_030_fa3_sm90_gqa_tma_partial"
FA3_GQA_TMA_SPLIT_030_ALGORITHM = "vllm_030_fa3_sm90_gqa_tma_split_partial"
FA3_GQA_TMA_COMBINE_030_ALGORITHM = "vllm_030_fa3_sm90_gqa_tma_combine_partial"
FA3_SPARSE_MLA_030_ALGORITHM = "vllm_030_fa3_sm90_sparse_mla_partial"
FA3_SPARSE_MLA_COMBINE_030_ALGORITHM = "vllm_030_fa3_sm90_sparse_mla_combine_partial"
FA3_GQA_PREFILL_030_SOURCE_REVISION = "506341a143fcabd4bb79052a7605ada727d6b3f5"
# SGLang's absorbed-MLA extend (flashinfer BatchMLAPagedAttentionWrapper,
# backend "auto" -> "fa3" on sm90): flashinfer 0.6.11
# include/flashinfer/attention/mla_hopper.cuh BatchMLAPageAttentionHopper.
FLASHINFER_MLA_PREFILL_ALGORITHM = "flashinfer_mla_hopper_sm90_paged"
_FLASHINFER_MLA_TILE = 64
_FLASHINFER_MLA_THREADS = 256
_FLASHINFER_MLA_STAGES = 2

# vLLM 0.24.0 vllm/config/attention.py
# (AttentionConfig.flash_attn_max_num_splits_for_cuda_graph).
_VLLM_CUDA_GRAPH_MAX_NUM_SPLITS = 32
# hopper/flash_api.cpp: splits above 1 require num_splits <= 256 and the
# heuristic's max_splits argument is 128.
_HEURISTIC_MAX_SPLITS = 128
# hopper/heuristics.h num_splits_heuristic: 50 MiB L2 assumption.
_HEURISTIC_L2_BYTES = 50 * 1024 * 1024
# hopper/flash_api.cpp PREPARE_VARLEN_MAX_BATCHES_1CTA.
_PREPARE_VARLEN_MAX_BATCHES_1CTA = 992
_ELEMENT_BYTES = 2
_MMA_WARPGROUPS = 2
_THREADS_PER_WARPGROUP = 128


def select_flash_attn_forward_partition(
        shape: Mapping[str, int], variant: str,
        attributes: Mapping[str, object]) -> tuple:
    """The pinned vLLM0.30 default FA3 backend issues one varlen call.

    FlashAttentionImpl.forward passes the complete query_start_loc/seq_lens
    and the caller-owned output view to its non-cascade attention call.
    The caller is pinned by source_v030_gqa_caller/manifest.json. Unlike the
    MetaX split-forward backend, it does not split mixed requests by phase.
    FULL consumes model-input metadata; eager and PIECEWISE use logical
    requests. Empty FULL tails stay empty. Returning no partitions after
    validation preserves one whole-call output fill and existing copy owners.
    This says nothing about the installed GPU provider or service accuracy.
    """
    require_variant(variant, ("prefill", "decode"), "H200 FA3 forward")
    if ("source_model_execution_version" not in attributes
            or attributes["source_model_execution_version"] != "vllm_0_30_0"
            or "attention_backend" in attributes
            or "flash_attn_version" in attributes and (
                type(attributes["flash_attn_version"]) is not int
                or attributes["flash_attn_version"] != 3)
            or attributes["kv_layout"] not in (
                "contiguous", "paged", "paged_varlen", "chunked_prefill")):
        raise NotImplementedError("H200 forward partition requires its vLLM0.30 default FA3 caller")
    if ("source_component" not in attributes
            or attributes["source_component"] != "symbolic_gqa_mathematical"):
        raise NotImplementedError("H200 forward partition does not own this source component")
    # SymbolicAttention is the ordinary decoder/DCP1/non-cascade caller.
    # Explicit contradictory caller facts must not inherit that source scope.
    if ("use_cascade" in attributes and attributes["use_cascade"] is not False
            or "dcp_world_size" in attributes and (
                type(attributes["dcp_world_size"]) is not int or attributes["dcp_world_size"] != 1)
            or "attn_type" in attributes and attributes["attn_type"] != "decoder"):
        raise NotImplementedError("H200 forward partition requires non-cascade decoder attention with DCP1")
    query_heads, kv_heads, head_dim = require_shape(
        shape, ("num_query_heads", "num_kv_heads", "head_dim"))
    if query_heads % kv_heads or head_dim not in (64, 128, 256):
        raise NotImplementedError("H200 forward partition requires its ordinary FA3 head geometry")
    mode = attributes["graph_runtime_mode"]
    if mode not in ("NONE", "PIECEWISE", "FULL"):
        raise ValueError("H200 forward partition requires a resolved graph runtime mode")
    if mode == "FULL":
        q_lens, kv_lens = attributes["model_input_q_lens"], attributes["model_input_kv_lens"]
        physical_rows = attributes["graph_model_input_tokens"]
        if type(physical_rows) is not int or physical_rows <= 0:
            raise ValueError("H200 FULL requires a positive physical query extent")
    else:
        q_lens, kv_lens = attributes["q_lens"], attributes["kv_lens"]
    if (not isinstance(q_lens, tuple) or not isinstance(kv_lens, tuple)
            or not q_lens or len(q_lens) != len(kv_lens)
            or any(type(value) is not int for value in (*q_lens, *kv_lens))):
        raise ValueError("H200 forward partition requires paired integer request lengths")
    if (any(q < 0 or k < q or (q == 0 and k != 0) for q, k in zip(q_lens, kv_lens))
            or not any(q_lens)):
        raise ValueError("H200 forward partition request lengths are outside its source domain")
    if mode == "FULL":
        if sum(q_lens) > physical_rows:
            raise ValueError("H200 FULL query metadata exceeds its physical query extent")
    elif (any(q == 0 for q in q_lens)
            or (len(q_lens), max(q_lens), max(kv_lens))
            != (shape["batch"], shape["q_len"], shape["kv_len"])):
        raise ValueError("H200 logical request lengths disagree with the call shape")
    return ()


def v030_gqa_dynamic_splits(q_lens, kv_lens, *, ratio: int, kv_heads: int,
                             block_m: int, block_n: int, split_cap: int,
                             num_sms: int, allow_source_padding=False) -> tuple[int, ...]:
    """Pinned flash_prepare_scheduler.cu, without duration-derived inputs.

    ``split_cap`` may be the largest observed dynamic split: capping at that
    value leaves every smaller per-request split unchanged. The compiler's
    Split flag is a separate fact, including launches whose dynamic split is
    one but which still materialize partial O/LSE and run combine.
    """

    if len(q_lens) != len(kv_lens) or not q_lens:
        raise ValueError("FA3 GQA requires paired nonempty request lengths")
    if any(type(value) is not int or value <= 0 for value in (
            ratio, kv_heads, block_m, block_n, split_cap, num_sms)):
        raise ValueError("FA3 GQA scheduler dimensions must be positive")
    if type(allow_source_padding) is not bool:
        raise ValueError("FA3 padding source flag must be boolean")
    if any(type(q) is not int or type(k) is not int or q < (0 if allow_source_padding else 1) or k < q
           or q == 0 and k != 0 for q, k in zip(q_lens, kv_lens)):
        raise ValueError("FA3 GQA scheduler requires positive Q <= KV lengths")
    if split_cap == 1 or len(q_lens) > 992:
        return (1,) * len(q_lens)
    import struct
    def f32(value):
        return struct.unpack("f", struct.pack("f", value))[0]
    m_blocks = tuple(0 if q == 0 else ceil_div(q * ratio, block_m) for q in q_lens)
    n_blocks = tuple(0 if k == 0 else ceil_div(k, block_n) for k in kv_lens)
    total = sum(m * n for m, n in zip(m_blocks, n_blocks))
    if total == 0:
        raise ValueError("FA3 dynamic prepare requires positive work before its source integer division")
    blocks_float = f32(f32(f32(float(total)) * f32(1.1)) * f32(float(kv_heads)))
    blocks_per_sm = max(1, math.ceil(f32(blocks_float / f32(float(num_sms)))))
    return tuple(max(1, min(0 if n == 0 else ceil_div(n, blocks_per_sm), split_cap))
                 for n in n_blocks)


def _v030_gqa_source_tile(head_dim: int, packed_q: int, causal: bool,
                          paged_non_tma: bool) -> tuple[int, int]:
    """Pinned tile_size_fwd_sm90 for the owned equal-QK/V nonlocal calls."""
    one_wg = packed_q <= 64
    if head_dim == 64:
        return ((64, 192) if one_wg
                else (192, 128 if causal else (160 if paged_non_tma else 192)))
    if head_dim == 128:
        return (64 if one_wg else 128,
                128 if causal or paged_non_tma else (176 if one_wg else 160))
    if head_dim == 256:
        # Use_one_mma_wg is disabled at D256 in the launch template.
        return 128, 80
    raise ValueError("FA3 equal-head source tile is outside the owned domain")


def _v030_gqa_source_causal(causal: bool, q_len: int, head_dim: int,
                            paged: bool) -> bool:
    # flash_api.cpp keeps the causal template for paged D128 Q1; the
    # attention mask itself is still equivalent to noncausal for one query.
    return causal and (q_len > 1 or paged and 64 < head_dim <= 128)


def validate_v030_gqa_observed_launch(kernel, hardware) -> None:
    """Validate the source-domain and discrete facts of a GQA observation.

    The existing canonical records retain actual grid/block and template
    flags. They select no empirical service rate. The D64/D128 observations
    and source-only D256 calls share the pinned PackGQA varlen program.
    """

    shape, attributes, launch = kernel.shape, kernel.attributes, kernel.launch
    batch, q_len, kv_len, query_heads, kv_heads, head_dim = require_shape(
        shape, ("batch", "q_len", "kv_len", "num_query_heads",
                "num_kv_heads", "head_dim"))
    full = ("source_program" in attributes and attributes["source_program"] == "vllm_0_30_fa3_gqa_full_decode")
    q_lens = tuple(attributes["model_input_q_lens"] if full else attributes["q_lens"])
    kv_lens = tuple(attributes["model_input_kv_lens"] if full else attributes["kv_lens"])
    if (hardware.hardware_id != "nvidia_h200" or kernel.dtype not in ("bf16", "fp16")
            or head_dim not in (64, 128, 256) or query_heads % kv_heads
            or len(q_lens) != batch or len(kv_lens) != batch
            or max(q_lens) != q_len or max(kv_lens) != kv_len
            or any(q < (0 if full else 1) or k < q or q == 0 and k != 0 for q, k in zip(q_lens, kv_lens))
            or attributes["kv_layout"] not in (
                "contiguous", "paged", "paged_varlen", "chunked_prefill")
            or launch is None or launch.tiling is None):
        raise ValueError("FA3 GQA observed source domain differs")
    tma = launch.algorithm in (FA3_GQA_TMA_030_ALGORITHM,
                               FA3_GQA_TMA_SPLIT_030_ALGORITHM,
                               FA3_GQA_TMA_COMBINE_030_ALGORITHM)
    paged = attributes["kv_layout"] in ("paged", "paged_varlen", "chunked_prefill")
    causal = _v030_gqa_source_causal(attributes["causal"], q_len, head_dim, paged)
    if tma:
        if not paged or "kv_page" not in launch.tiling.tile_shape:
            raise ValueError("FA3 paged TMA requires its explicit page size")
        page_size = launch.tiling.tile_shape["kv_page"]
        tma_m, tma_n = _v030_gqa_source_tile(
            head_dim, q_len * (query_heads // kv_heads), causal, False)
        if (int(page_size) != page_size or page_size < 16 or page_size % 16
                or page_size % tma_n or q_len * (query_heads // kv_heads) <= tma_m):
            raise ValueError("FA3 paged TMA page size or packed query differs from source")
        if launch.parameters != _source_parameters(launch.tiling.num_kv_splits, False):
            raise ValueError("FA3 paged TMA parameters differ from source")
    if launch.algorithm in (FA3_GQA_COMBINE_030_ALGORITHM, FA3_GQA_TMA_COMBINE_030_ALGORITHM,
                            FA3_PREFILL_COMBINE_ALGORITHM):
        if launch.algorithm == FA3_PREFILL_COMBINE_ALGORITHM:
            expected_grid = ((ceil_div(q_len * query_heads, 8), 1, batch)
                if 2 * ceil_div(sum(q_lens), 8) >= batch * ceil_div(q_len, 8)
                else (ceil_div(sum(q_lens) * query_heads, 8) + batch, 1, 1))
            if (kernel.dtype != "bf16" or head_dim != 128 or query_heads != kv_heads
                    or attributes["kv_layout"] != "contiguous" or attributes["causal"]
                    or launch.tiling.num_kv_splits <= 1
                    or launch.grid != expected_grid or launch.tiling.pipeline_depth != 4
                    or launch.parameters != _source_parameters(launch.tiling.num_kv_splits, False)):
                raise ValueError("FA3 split-prefill combine differs from its expanded MLA source")
        tile = launch.tiling.tile_shape
        if (kernel.variant != "combine"
                or tile != {"rows": 16 if head_dim == 64 else 8,
                            "value_head": min(head_dim, 128),
                            **({"kv_page": page_size} if tma else {})}
                or launch.block != (256, 1, 1)
                or launch.grid[1] != ceil_div(head_dim, min(head_dim, 128))):
            raise ValueError("FA3 GQA combine template differs from source")
        return
    if launch.algorithm not in (
            FA3_GQA_OBSERVED_030_ALGORITHM, FA3_GQA_OBSERVED_SPLIT_030_ALGORITHM,
            FA3_GQA_TMA_030_ALGORITHM, FA3_GQA_TMA_SPLIT_030_ALGORITHM):
        raise ValueError("FA3 GQA observation has an unrecognized program")
    tile = launch.tiling.tile_shape
    expected = _v030_gqa_source_tile(
        head_dim, q_len * (query_heads // kv_heads), causal, paged and not tma)
    if tma and q_len * (query_heads // kv_heads) <= expected[0]:
        raise ValueError("FA3 paged TMA requires more than one packed M tile")
    expected_threads = (expected[0] // 64 + 1) * 128
    if (set(tile) != ({"q", "kv", "head_dim", "kv_page"} if tma else {"q", "kv", "head_dim"})
            or tuple(tile[name] for name in ("q", "kv")) != expected
            or tile["head_dim"] != head_dim
            or launch.block != (expected_threads, 1, 1)
            or launch.grid != (int(hardware.num_compute_units), 1, 1)
            or launch.tiling.pipeline_depth != 2
            or not tma and launch.parameters != _source_parameters(
                launch.tiling.num_kv_splits, paged)):
        raise ValueError("FA3 GQA tile or persistent launch differs from source")


_GQA_FORWARD_SPEC_RECIPES = RuntimeCache()


def _immutable_gqa_fact_key(value):
    """Type-preserving keys; mutable inputs use the original factory."""
    if type(value) in (type(None), bool, int, float, str):
        return type(value), value
    if type(value) is tuple:
        parts = tuple(_immutable_gqa_fact_key(item) for item in value)
        if all(item is not None for item in parts):
            return tuple, parts
    return None


def _gqa_mapping_recipe_key(values):
    if not isinstance(values, Mapping):
        return None
    parts = tuple((_immutable_gqa_fact_key(name), _immutable_gqa_fact_key(value))
        for name, value in values.items())
    if any(name is None or value is None for name, value in parts):
        return None
    return parts


def select_v030_gqa_forward_specs(
        shape, dtype, attributes, hardware, *, page_size,
        num_splits: int, scheduler_metadata_precomputed: bool):
    """Reuse exact immutable source recipes within the existing run scope.

    This caches geometry/KernelSpecs, never observed or predicted duration.
    Complete facts, their types, hardware identity and SM count are keys;
    mutable caller inputs retain the same uncached validation and builder.
    """
    shape_key = _gqa_mapping_recipe_key(shape)
    attributes_key = _gqa_mapping_recipe_key(attributes)
    controls_key = _immutable_gqa_fact_key((dtype, page_size, num_splits, scheduler_metadata_precomputed))
    hardware_key = _immutable_gqa_fact_key((hardware.hardware_id, hardware.num_compute_units))
    if (shape_key is None or attributes_key is None or controls_key is None
            or hardware_key is None):
        return _select_v030_gqa_forward_specs(shape, dtype, attributes, hardware,
            page_size=page_size, num_splits=num_splits,
            scheduler_metadata_precomputed=scheduler_metadata_precomputed)
    key = (id(hardware), hardware_key, shape_key, attributes_key, controls_key)
    if key in _GQA_FORWARD_SPEC_RECIPES:
        cached_hardware, specs = _GQA_FORWARD_SPEC_RECIPES[key]
        if cached_hardware is not hardware:
            raise RuntimeError("FA3 source recipe hardware identity collision")
        _GQA_FORWARD_SPEC_RECIPES.move_to_end(key)
        return specs
    specs = _select_v030_gqa_forward_specs(shape, dtype, attributes, hardware,
        page_size=page_size, num_splits=num_splits,
        scheduler_metadata_precomputed=scheduler_metadata_precomputed)
    _GQA_FORWARD_SPEC_RECIPES[key] = hardware, specs
    if len(_GQA_FORWARD_SPEC_RECIPES) > 256:
        _GQA_FORWARD_SPEC_RECIPES.popitem(last=False)
    return specs


def _select_v030_gqa_forward_specs(
        shape, dtype, attributes, hardware, *, page_size,
        num_splits: int, scheduler_metadata_precomputed: bool):
    """Source-derived complete varlen FA3 call, with explicit caller facts.

    The pinned flash_api.cpp accepts zero for automatic split selection.
    Precomputed metadata belongs to the caller: when true its construction
    is outside this call, and an unsplit call resets its semaphore afterward.
    When false, the call prepares fresh metadata itself. No launch observation
    or duration is needed. The returned KernelSpecs use the common predict API.
    """
    from src.kernel.kernel_spec import KernelKind, KernelSpec

    if "source_program" in attributes and attributes["source_program"] == "vllm_0_30_fa3_gqa_full_decode":
        return select_v030_gqa_full_decode_specs(shape, dtype, attributes, hardware,
            page_size=page_size, num_splits=num_splits,
            scheduler_metadata_precomputed=scheduler_metadata_precomputed)
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("FA3 source selection requires H200")
    if "flash_attn_version" in attributes and (
            type(attributes["flash_attn_version"]) is not int or attributes["flash_attn_version"] != 3):
        raise NotImplementedError("FA3 source call cannot override the caller's FlashAttention version")
    if ("source_model_execution_version" in attributes
            and attributes["source_model_execution_version"] != "vllm_0_30_0"):
        raise NotImplementedError("FA3 GQA source call requires its pinned vLLM0.30 caller")
    if "attention_backend" in attributes:
        raise NotImplementedError("FA3 GQA source call cannot replace an explicit provider override")
    if "full_cuda_graphs" in attributes:
        if type(attributes["full_cuda_graphs"]) is not bool:
            raise TypeError("FA3 caller full_cuda_graphs must be boolean")
        if attributes["full_cuda_graphs"]:
            raise NotImplementedError("FA3 GQA captured/padded source contract is not yet bound")
    if "max_cudagraph_capture_size" in attributes:
        if type(attributes["max_cudagraph_capture_size"]) is not int:
            raise TypeError("FA3 caller capture size must be an integer")
        if attributes["max_cudagraph_capture_size"] != 0:
            raise NotImplementedError("FA3 GQA captured/padded source contract is not yet bound")
    if ("graph_runtime_mode" in attributes and attributes["graph_runtime_mode"] != "NONE"
            or any(name in attributes for name in (
                "graph_logical_tokens", "graph_model_input_tokens", "model_input_shape"))):
        raise NotImplementedError("FA3 GQA captured/padded source contract is not yet bound")
    if "return_softmax_lse" in attributes:
        if type(attributes["return_softmax_lse"]) is not bool:
            raise TypeError("FA3 caller LSE return flag must be boolean")
        if attributes["return_softmax_lse"]:
            raise NotImplementedError("FA3 GQA final-LSE caller binding is not yet defined")
    require_dtype(dtype, ("bf16", "fp16"))
    batch, q_len, kv_len, heads, kv_heads, d = require_shape(
        shape, ("batch", "q_len", "kv_len", "num_query_heads",
                "num_kv_heads", "head_dim"))
    if d not in (64, 128, 256) or heads < kv_heads or heads % kv_heads:
        raise NotImplementedError("FA3 source GQA requires D64/D128/D256 and integral GQA")
    if type(scheduler_metadata_precomputed) is not bool:
        raise TypeError("FA3 metadata lifetime must be explicit")
    if type(num_splits) is not int or not 0 <= num_splits <= 256:
        raise ValueError("FA3 requested splits must be in0..256")
    q_lens, kv_lens = _request_lengths(shape, attributes)
    layout = attributes["kv_layout"]
    if layout not in ("contiguous", "paged", "paged_varlen", "chunked_prefill"):
        raise NotImplementedError("FA3 source GQA layout is unbound")
    paged = layout in ("paged", "paged_varlen", "chunked_prefill")
    if (paged and (type(page_size) is not int or page_size < 16 or page_size % 16)
            or not paged and page_size is not None):
        raise NotImplementedError("FA3 source GQA requires an explicit aligned page ABI")
    model_fields = {"model_input_q_lens", "model_input_kv_lens"}
    if model_fields.intersection(attributes):
        if not model_fields.issubset(attributes) or "graph_runtime_mode" not in attributes:
            raise ValueError("FA3 GQA model-input metadata must be complete")
        if (tuple(attributes["model_input_q_lens"]) != q_lens
                or tuple(attributes["model_input_kv_lens"]) != kv_lens):
            raise NotImplementedError("FA3 padded model-input requests need a separate source contract")
    if type(attributes["causal"]) is not bool:
        raise TypeError("FA3 causal must be boolean")
    ratio = heads // kv_heads
    causal = _v030_gqa_source_causal(attributes["causal"], q_len, d, paged)
    paged_tma = False
    if paged:
        # flash_api.cpp:get_pagedkv_tma first queries the nonpaged tile,
        # then enables TMA only for a page multiple of N and packed Q>M.
        # Larger pages that keep cp.async reuse the existing program. TMA
        # calls retain their own tile, explicit page extent and algorithm.
        tma_m, tma_n = _v030_gqa_source_tile(d, q_len * ratio, causal, False)
        paged_tma = page_size % tma_n == 0 and q_len * ratio > tma_m
    bm, bn = _v030_gqa_source_tile(d, q_len * ratio, causal, paged and not paged_tma)
    static_splits = (num_splits if num_splits else _varlen_num_splits(
        batch=batch, max_seqlen_q=q_len, max_seqlen_k=kv_len,
        num_heads=heads, num_heads_k=kv_heads, head_dim=d, head_dim_v=d,
        block_m=bm, block_n=bn, causal=causal,
        num_sms=int(hardware.num_compute_units)))
    if batch > _PREPARE_VARLEN_MAX_BATCHES_1CTA:
        static_splits = 1
    # This revision's CMakeLists.txt builds FA3 with
    # FLASHATTENTION_PACKGQA_ONLY. flash_api.cpp therefore forces PackGQA
    # even when HQ == HKV; the ratio-one cases use this same source program.
    common = {name: attributes[name] for name in (
        "causal", "kv_layout", "kv_dist", "q_lens", "kv_lens")}
    common.update({"fa_ksteps": None, "fa_eff_mfrac": 1.0})
    if paged_tma:
        main_algorithm = (FA3_GQA_TMA_SPLIT_030_ALGORITHM if static_splits > 1
                          else FA3_GQA_TMA_030_ALGORITHM)
    else:
        main_algorithm = (FA3_GQA_OBSERVED_SPLIT_030_ALGORITHM if static_splits > 1
                          else FA3_GQA_OBSERVED_030_ALGORITHM)
    main_launch = LaunchConfig(
        algorithm=main_algorithm,
        grid=(int(hardware.num_compute_units), 1, 1),
        block=((bm // 64 + 1) * 128, 1, 1), cluster=(1, 1, 1),
        tiling=TilingSpec(tile_shape={"q": bm, "kv": bn, "head_dim": d,
                                     **({"kv_page": page_size} if paged_tma else {})},
                          pipeline_depth=2, num_kv_splits=static_splits),
        parameters=_source_parameters(static_splits, paged and not paged_tma),
        execution_variant="decode" if q_len == 1 else "prefill")
    main = KernelSpec(kind=KernelKind.FLASH_ATTN, hardware_id=hardware.hardware_id,
                      shape=shape, dtype=dtype, variant=main_launch.execution_variant,
                      launch=main_launch, attributes=common)
    validate_v030_gqa_observed_launch(main, hardware)
    parts = []
    if not scheduler_metadata_precomputed:
        parts.append(KernelSpec(
            kind=KernelKind.VECTOR, hardware_id=hardware.hardware_id,
            shape={"batch": batch}, dtype="int32", variant="fa3_prepare_varlen",
            launch=select_fa3_prepare_varlen_launch(batch),
            attributes={"num_splits_static": static_splits}))
    parts.append(main)
    if static_splits > 1:
        rows = 16 if d == 64 else 8
        value_tile = min(d, 128)
        value_blocks = ceil_div(d, value_tile)
        if 2 * ceil_div(sum(q_lens), rows) >= batch * ceil_div(q_len, rows):
            grid = (ceil_div(q_len * heads, rows), value_blocks, batch)
        else:
            grid = (ceil_div(sum(q_lens) * heads, rows) + batch, value_blocks, 1)
        combine = LaunchConfig(
            algorithm=(FA3_GQA_TMA_COMBINE_030_ALGORITHM if paged_tma
                       else FA3_GQA_COMBINE_030_ALGORITHM),
            grid=grid, block=(256, 1, 1),
            tiling=TilingSpec(tile_shape={"rows": rows, "value_head": value_tile,
                                         **({"kv_page": page_size} if paged_tma else {})},
                              pipeline_depth=4, num_kv_splits=static_splits),
            parameters=_source_parameters(static_splits, False),
            execution_variant="combine")
        parts.append(KernelSpec(
            kind=KernelKind.FLASH_ATTN, hardware_id=hardware.hardware_id,
            shape=shape, dtype=dtype, variant="combine", launch=combine,
            attributes=common))
    elif scheduler_metadata_precomputed:
        # The one-INT32 Fill is bound to the H200 observation/source geometry,
        # rather than selecting a projected SM80 program.
        parts.append(KernelSpec(
            kind=KernelKind.VECTOR, hardware_id=hardware.hardware_id,
            shape={"element_count": 1}, dtype="int32", variant="elementwise",
            launch=LaunchConfig(algorithm="elementwise", grid=(1, 1, 1),
                                block=(128, 1, 1), execution_variant="elementwise"),
            attributes={"reads": 0, "writes": 1, "operation": "fill"}))
    return tuple(parts)


def _require_v030_gqa_fresh_prefill_domain(shape, dtype, attributes, hardware):
    """Validate the pinned FA3 BF16 D128 fresh varlen GQA domain.

    The source fixes the tile and persistent grid.  Only equal-length fresh
    requests are bound; decode, mixed requests, split-KV and the separate
    0.24 MLA source remain outside this domain.
    """

    batch, q_len, kv_len, query_heads, kv_heads, head_dim = require_shape(
        shape, ("batch", "q_len", "kv_len", "num_query_heads",
                "num_kv_heads", "head_dim"))
    if (dtype != "bf16" or q_len != kv_len
            or head_dim != 128 or kv_heads > query_heads
            or query_heads % kv_heads != 0
            or attributes["causal"] is not True
            or attributes["kv_layout"] not in ("paged", "paged_varlen")
            or tuple(attributes["q_lens"]) != (q_len,) * batch
            or tuple(attributes["kv_lens"]) != (kv_len,) * batch
            or attributes["num_kv_splits"] != 1):
        raise NotImplementedError("vLLM 0.30 FA3 GQA fresh prefill domain is unbound")
    ratio = query_heads // kv_heads
    if q_len * ratio <= 64:
        raise NotImplementedError("vLLM 0.30 FA3 one-MMA-warpgroup domain is unbound")
    # The v0.30 pinned heuristics.h selects split-KV before dispatch.  A
    # split call needs its separate combine program, which is not bound here.
    if _varlen_num_splits(
            batch=1, max_seqlen_q=q_len, max_seqlen_k=kv_len,
            num_heads=query_heads, num_heads_k=kv_heads,
            head_dim=128, head_dim_v=128, block_m=128, block_n=128,
            causal=True, num_sms=int(hardware.num_compute_units)) != 1:
        raise NotImplementedError("vLLM 0.30 FA3 split GQA prefill is unbound")


def select_v030_gqa_fresh_prefill_launch(shape, dtype, attributes, hardware):
    """Select the fixed source launch after its shape/heuristic gate."""

    _require_v030_gqa_fresh_prefill_domain(
        shape, dtype, attributes, hardware)
    # hopper/tile_size.h: BF16 D128 causal, two MMA warpgroups -> 128x128.
    # flash_fwd_launch_template.h: varlen persistent scheduler launches one
    # CTA per SM; two MMA warpgroups plus one producer use 384 threads.
    return LaunchConfig(
        algorithm=FA3_GQA_PREFILL_030_ALGORITHM,
        grid=(int(hardware.num_compute_units), 1, 1),
        block=(384, 1, 1), cluster=(1, 1, 1),
        tiling=TilingSpec(tile_shape={"q": 128, "kv": 128,
                                     "qk_head": 128, "value_head": 128},
                          pipeline_depth=2, num_kv_splits=1),
        parameters=_source_parameters(1, True),
        execution_variant="prefill",
    )


def validate_v030_gqa_fresh_prefill_launch_contract(
        shape, dtype, variant, attributes, hardware, launch):
    """Check supplied source facts; prediction does not choose a launch."""

    if not isinstance(launch, LaunchConfig) or launch.tiling is None:
        raise ValueError("vLLM 0.30 FA3 GQA requires a source launch with tiling")
    _require_v030_gqa_fresh_prefill_domain(
        shape, dtype,
        {**attributes, "num_kv_splits": launch.tiling.num_kv_splits},
        hardware)
    source_tile = TilingSpec(
        tile_shape={"q": 128, "kv": 128,
                    "qk_head": 128, "value_head": 128},
        pipeline_depth=2, num_kv_splits=1)
    if (variant != "prefill"
            or launch.algorithm != FA3_GQA_PREFILL_030_ALGORITHM
            or launch.execution_variant != "prefill"
            or launch.grid != (int(hardware.num_compute_units), 1, 1)
            or launch.block != (384, 1, 1)
            or launch.cluster != (1, 1, 1)
            or launch.tiling != source_tile
            or launch.parameters != _source_parameters(1, True)
            or launch.source_program is not None):
        raise ValueError("vLLM 0.30 FA3 GQA launch disagrees with source contract")


def _source_parameters(splits: int, paged: bool) -> Mapping[str, float]:
    return {
        "unpadded_flops": 0.0,
        "mma_smem_bytes_per_flop": 0.0,
        "split_scratch_dram_bytes_per_element": 4.0 if splits > 1 else 0.0,
        "page_table_bytes_per_kv_panel": (
            4.0 if paged and splits > 1 else 0.0),
    }


def _request_lengths(shape: Mapping[str, int],
                     attributes: Mapping[str, object]) -> tuple:
    sparse_fa3 = ("source_program" in attributes
        and attributes["source_program"] == "vllm_0_30_fa3_sparse_mla")
    if "source_program" in attributes and attributes["source_program"] == "vllm_0_30_fa3_gqa_full_decode":
        return tuple(attributes["model_input_q_lens"]), tuple(attributes["model_input_kv_lens"])
    batch, q_len = require_shape(shape, ("batch", "q_len"))
    kv_len = shape["kv_len"]
    if type(kv_len) is not int or kv_len < (0 if sparse_fa3 else 1):
        raise ValueError("H200 FA3 maximum KV length is outside its source domain")
    raw_q = attributes["q_lens"]
    raw_kv = attributes["kv_lens"]
    if (not isinstance(raw_q, (list, tuple))
            or not isinstance(raw_kv, (list, tuple))
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in tuple(raw_q) + tuple(raw_kv))):
        raise TypeError("H200 FA3 MLA request lengths must be integers")
    q_lens, kv_lens = tuple(raw_q), tuple(raw_kv)
    if (any(value <= 0 for value in q_lens)
            or any(value < (0 if sparse_fa3 else 1) for value in kv_lens)):
        raise ValueError("H200 FA3 request lengths are outside their source domain")
    if (len(q_lens) != batch or len(kv_lens) != batch
            or max(q_lens) != q_len or max(kv_lens) != kv_len):
        raise ValueError("H200 FA3 MLA request lengths disagree with shape")
    if not sparse_fa3 and attributes["causal"] is True and any(query > key_value for query, key_value in zip(q_lens, kv_lens)):
        raise ValueError("H200 FA3 MLA requires q_len <= kv_len per request")
    return q_lens, kv_lens


def fa3_num_splits_heuristic(total_mblocks: int, num_sms: int,
                             num_n_blocks: int, num_m_blocks: int,
                             size_one_kv_head: int, is_causal_or_local: bool,
                             max_splits: int) -> int:
    """hopper/heuristics.h ``num_splits_heuristic`` verbatim."""

    if total_mblocks >= 0.8 * num_sms:
        if (size_one_kv_head > _HEURISTIC_L2_BYTES
                and num_m_blocks >= num_sms * 2
                and not is_causal_or_local):
            return min(ceil_div(size_one_kv_head, _HEURISTIC_L2_BYTES),
                       max_splits)
        return 1
    if num_n_blocks <= 4:
        return 1
    max_splits = min(max_splits, num_sms, num_n_blocks)
    efficiency = []
    for splits in range(1, max_splits + 1):
        waves = total_mblocks * splits / num_sms
        efficiency.append(waves / ceil_div(total_mblocks * splits, num_sms))
    best = max(efficiency)
    for splits in range(1, max_splits + 1):
        if efficiency[splits - 1] >= 0.85 * best:
            return splits
    return 1


def _varlen_num_splits(*, batch: int, max_seqlen_q: int, max_seqlen_k: int,
                       num_heads: int, num_heads_k: int, head_dim: int,
                       head_dim_v: int, block_m: int, block_n: int,
                       causal: bool, num_sms: int) -> int:
    """``get_num_splits`` for a varlen call (dynamic-split upper bound)."""

    qhead_per_khead = num_heads // num_heads_k
    num_n_blocks = ceil_div(max_seqlen_k, block_n)
    num_m_blocks = ceil_div(max_seqlen_q * qhead_per_khead, block_m)
    size_one_kv_head = max_seqlen_k * (head_dim + head_dim_v) * _ELEMENT_BYTES
    # Varlen sets num_splits_dynamic_ptr before the heuristic, which then
    # pretends batch = 1 (flash_api.cpp get_num_splits).
    total_mblocks = num_heads_k * num_m_blocks
    splits = fa3_num_splits_heuristic(
        total_mblocks, num_sms, num_n_blocks, num_m_blocks,
        size_one_kv_head, causal, _HEURISTIC_MAX_SPLITS)
    if batch > _PREPARE_VARLEN_MAX_BATCHES_1CTA:
        return 1
    return splits


def _decode_uses_cuda_graph_split_bound(attributes: Mapping[str, object],
                                        decode_tokens: int) -> bool:
    """vLLM FlashAttnMLAMetadataBuilder._build_decode split bound rule.

    ``max_num_splits`` is the CUDA-graph bound when the deployment enables
    full CUDA graphs (``cudagraph_mode.has_full_cudagraphs()``) and the
    decode token count fits ``max_cudagraph_capture_size``; otherwise 0
    (the FA3 heuristic).  Both are deployment facts the workload attaches
    to every MLA node; they are required.
    """

    full_graphs = attributes["full_cuda_graphs"]
    capture = attributes["max_cudagraph_capture_size"]
    if not isinstance(full_graphs, bool):
        raise TypeError("full_cuda_graphs must be boolean")
    if isinstance(capture, bool) or not isinstance(capture, int) or capture < 0:
        raise ValueError("max_cudagraph_capture_size must be a non-negative int")
    return full_graphs and decode_tokens <= capture


def _normalized_variant(variant) -> Optional[str]:
    if not isinstance(variant, str):
        return None
    return variant.strip().lower()


def _fa3_prefill_in_domain(shape, dtype, attributes, hardware) -> bool:
    """One unsplit FA3 MLA prefill call, fresh or noncausal context chunk.

    A context chunk is only the attention call. Its caller still owns the
    cache gather, K/V expansion and final merge with the fresh suffix.
    """

    if dtype != "bf16" or attributes["kv_layout"] != "mla":
        return False
    if (shape["qk_head_dim"], shape["v_head_dim"]) != (192, 128):
        return False
    q_lens, kv_lens = _request_lengths(shape, attributes)
    if q_lens != kv_lens and attributes["causal"]:
        # vLLM runs chunked-context prefill as a causal call plus context
        # chunk calls and merge_attn_states; that sequence is not derived.
        return False
    heads = shape["num_query_heads"]
    return _varlen_num_splits(
        batch=len(q_lens), max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens), num_heads=heads, num_heads_k=heads,
        head_dim=192, head_dim_v=128, block_m=128, block_n=128,
        causal=attributes["causal"], num_sms=int(hardware.num_compute_units)) == 1


def _uniform_packed_decode_queries(q_lens, query_heads: int) -> bool:
    """Uniform decode/verify rows handled by the pinned MLA caller.

    vLLM 0.24 FlashAttnMLA runs single-token decode and uniform speculative
    verify (up to its512-token reorder threshold) through the same FA3 call.
    PackGQA may produce multiple64-row M tiles; it does not round the
    query-head count down or silently route those tiles to FA2.
    """

    return (len(set(q_lens)) == 1
            and 1 <= q_lens[0] <= 512 and query_heads > 0)


def _fa3_decode_in_domain(shape, dtype, attributes) -> bool:
    """Absorbed MLA decode/verify with source-derived packed M tasks."""

    if dtype != "bf16" or attributes["kv_layout"] != "mla":
        return False
    if (shape["v_head_dim"], shape["kv_lora_rank"],
            shape["qk_head_dim"] - shape["v_head_dim"]) != (512, 512, 64):
        return False
    q_lens, _ = _request_lengths(shape, attributes)
    return _uniform_packed_decode_queries(q_lens, shape["num_query_heads"])


def _fa3_combine_in_domain(shape, dtype, attributes) -> bool:
    """The split combine of the absorbed MLA decode above."""

    if dtype != "bf16" or attributes["kv_layout"] != "mla":
        return False
    if (shape["head_dim"], shape["num_kv_heads"]) != (512, 1):
        return False
    q_lens, _ = _request_lengths(shape, attributes)
    return _uniform_packed_decode_queries(q_lens, shape["num_query_heads"])


def _flashinfer_mla_in_domain(shape, dtype, attributes) -> bool:
    if attributes["attention_backend"] != "flashinfer":
        raise ValueError(
            f"unknown MLA attention_backend {attributes['attention_backend']!r}")
    return (dtype == "bf16" and attributes["kv_layout"] == "mla"
            and shape["qk_head_dim"] - shape["v_head_dim"] == 64
            and shape["v_head_dim"] == 512)


def owns_select_target_mla_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> bool:
    """Whether the target MLA request is in the derived FA3 domain."""

    normalized = _normalized_variant(variant)
    if normalized == "mla_prefill":
        return _fa3_prefill_in_domain(shape, dtype, attributes, hardware)
    if normalized == "mla_decode" and "attention_backend" in attributes:
        return _flashinfer_mla_in_domain(shape, dtype, attributes)
    if normalized == "mla_decode":
        return _fa3_decode_in_domain(shape, dtype, attributes)
    if normalized == "mla_absorbed_prefill":
        return (dtype == "bf16" and attributes["kv_layout"] == "mla"
                and shape["qk_head_dim"] - shape["v_head_dim"] == 64
                and shape["v_head_dim"] == 512)
    return False


def select_target_mla_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig:
    """Select the vLLM 0.24 FA3 MLA prefill or absorbed-decode launch."""

    if not owns_select_target_mla_launch(
            shape, dtype, variant, attributes, hardware):
        raise ValueError(
            "H200 target MLA selector owns only the derived FA3 domain; the "
            "dispatcher routes other requests to the projected package")
    execution_variant = require_variant(
        variant, ("mla_prefill", "mla_decode", "mla_absorbed_prefill"),
        "H200 target MLA")
    require_dtype(dtype, ("bf16",))
    if attributes["causal"] is not True and execution_variant != "mla_prefill":
        raise ValueError("vLLM MLA attention is causal")
    batch, query_heads, qk_head_dim, value_head_dim = require_shape(
        shape, ("batch", "num_query_heads", "qk_head_dim", "v_head_dim"))
    q_lens, kv_lens = _request_lengths(shape, attributes)
    num_sms = int(hardware.num_compute_units)
    if (execution_variant == "mla_absorbed_prefill"
            or "attention_backend" in attributes):
        # Cooperative persistent launch dim3(num_blks_x = cluster size,
        # num_blks_y = num_sm / cluster size) from scheduler.cuh MLAPlan;
        # CTA_TILE_Q = CTA_TILE_KV = 64, NUM_STAGES = 2, 256 threads; the
        # split-KV merge runs inside the same kernel after grid.sync().
        return LaunchConfig(
            algorithm=FLASHINFER_MLA_PREFILL_ALGORITHM,
            grid=(num_sms, 1, 1),
            block=(_FLASHINFER_MLA_THREADS, 1, 1),
            cluster=(1, 1, 1),
            tiling=TilingSpec(
                tile_shape={
                    "q": _FLASHINFER_MLA_TILE, "kv": _FLASHINFER_MLA_TILE,
                    "qk_head": qk_head_dim - value_head_dim,
                    "value_head": value_head_dim,
                },
                pipeline_depth=_FLASHINFER_MLA_STAGES,
                num_kv_splits=1,
            ),
            parameters=_source_parameters(1, True),
            execution_variant=execution_variant,
        )
    block = (_MMA_WARPGROUPS * _THREADS_PER_WARPGROUP
             + _THREADS_PER_WARPGROUP, 1, 1)
    if execution_variant == "mla_prefill":
        # run_mha_fwd_<90, bf16, 192, 128>: q/k (nope+rope) 192, v 128, H
        # KV heads; tile_size_fwd_sm90(192, 128, causal) = 128 x 128.
        return LaunchConfig(
            algorithm=FA3_PREFILL_ALGORITHM,
            grid=(num_sms, 1, 1),
            block=block,
            cluster=(1, 1, 1),
            tiling=TilingSpec(
                tile_shape={
                    "q": 128, "kv": 128,
                    "qk_head": qk_head_dim, "value_head": value_head_dim,
                },
                pipeline_depth=2,
                num_kv_splits=1,
            ),
            parameters=_source_parameters(1, False),
            execution_variant=execution_variant,
        )

    # run_mha_fwd_<90, bf16, 64, 512>: q_pe/k_pe carry the rope head,
    # qv/v the latent; tile_size_fwd_sm90(64, 512) = 64 x 64.
    rope_dim = qk_head_dim - value_head_dim
    block_m, block_n = 64, 64
    if _decode_uses_cuda_graph_split_bound(attributes, sum(q_lens)):
        splits = _VLLM_CUDA_GRAPH_MAX_NUM_SPLITS
    else:
        splits = _varlen_num_splits(
            batch=batch, max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
            num_heads=query_heads, num_heads_k=1, head_dim=rope_dim,
            head_dim_v=value_head_dim, block_m=block_m, block_n=block_n,
            causal=attributes["causal"] and max(q_lens) > 1, num_sms=num_sms)
    if batch > _PREPARE_VARLEN_MAX_BATCHES_1CTA:
        splits = 1
    return LaunchConfig(
        algorithm=FA3_DECODE_ALGORITHM,
        grid=(num_sms, 1, 1),
        block=block,
        cluster=(1, 1, 1),
        tiling=TilingSpec(
            tile_shape={
                "q": block_m, "kv": block_n,
                "qk_head": rope_dim, "value_head": value_head_dim,
            },
            pipeline_depth=2,
            num_kv_splits=splits,
        ),
        parameters=_source_parameters(splits, True),
        execution_variant=execution_variant,
    )


def _combine_launch(shape: Mapping[str, int], dtype: str,
                    attributes: Mapping[str, object]) -> LaunchConfig:
    """``run_mha_fwd_combine_<bf16, float, 512 -> kBlockK 128>`` launch."""

    require_dtype(dtype, ("bf16",))
    batch, q_len, head_dim, query_heads = require_shape(
        shape, ("batch", "q_len", "head_dim", "num_query_heads"))
    q_lens, _ = _request_lengths(shape, attributes)
    splits = require_positive_integer(attributes, "num_kv_splits")
    if splits <= 1:
        raise ValueError("FA3 combine runs only when num_splits > 1")
    block_k = 128
    block_m = 8 if block_k % 128 == 0 else (16 if block_k % 64 == 0 else 32)
    total_q = sum(q_lens)
    if "graph_model_input_tokens" in attributes:
        # FULL-graph padding (vLLM gpu_model_runner / cudagraph_dispatcher):
        # the decode runs num_reqs_padded = padded tokens requests of query
        # length 0 and seq_len 0 beyond the real ones, and total_q is the
        # padded token count; the combine grid is sized from them and the
        # padded CTAs return at their seqlen check.
        if attributes["graph_logical_tokens"] != total_q:
            raise ValueError("combine graph domain disagrees with q_lens")
        if attributes["graph_runtime_mode"] != "FULL":
            raise ValueError("padded MLA combine requires a FULL graph")
        total_q = attributes["graph_model_input_tokens"]
        batch = total_q
        q_len = 1
    # StaticVarlenTileScheduler::choose_scheduling_algo.
    lower_bound_non_empty = ceil_div(total_q, block_m)
    dense_grid = batch * ceil_div(q_len, block_m)
    if 2 * lower_bound_non_empty >= dense_grid:
        grid = (ceil_div(q_len * query_heads, block_m),
                ceil_div(head_dim, block_k), batch)
    else:
        grid = (ceil_div(total_q * query_heads, block_m) + batch,
                ceil_div(head_dim, block_k), 1)
    return LaunchConfig(
        algorithm=FA3_COMBINE_ALGORITHM,
        grid=grid,
        block=(256, 1, 1),
        tiling=TilingSpec(
            tile_shape={"rows": block_m, "value_head": block_k},
            pipeline_depth=4,
            num_kv_splits=splits,
        ),
        parameters=_source_parameters(splits, False),
        execution_variant="combine",
    )


def owns_select_flash_attn_launch(shape: Mapping[str, int],
                                  dtype: Optional[str],
                                  variant: Optional[str],
                                  attributes: Mapping[str, object],
                                  hardware) -> bool:
    """Whether an explicitly versioned FA3 source program owns this call."""

    if (_normalized_variant(variant) == "prefill"
            and "source_component" in attributes
            and attributes["source_component"] == "symbolic_gqa_mathematical"
            and "source_model_execution_version" in attributes
            and attributes["source_model_execution_version"] == "vllm_0_30_0"
            and "launch_policy" in attributes
            and attributes["launch_policy"] == "analytical_launch"
            and ("flash_attn_version" not in attributes
                 or attributes["flash_attn_version"] == 3)):
        try:
            select_v030_gqa_fresh_prefill_launch(
                shape, dtype,
                {**attributes, "num_kv_splits": (
                    attributes["num_kv_splits"]
                    if "num_kv_splits" in attributes else 1)}, hardware)
        except NotImplementedError:
            return False
        return True
    return (_normalized_variant(variant) in ("combine", "split_combine")
            and _fa3_combine_in_domain(shape, dtype, attributes))


def select_flash_attn_launch(shape: Mapping[str, int], dtype: Optional[str],
                             variant: Optional[str],
                             attributes: Mapping[str, object],
                             hardware) -> LaunchConfig:
    """Select a source-bound FA3 GQA prefill or MLA split combine."""

    if not owns_select_flash_attn_launch(
            shape, dtype, variant, attributes, hardware):
        raise ValueError(
            "H200 FLASH_ATTN selector does not own this FA3 source domain; "
            "the dispatcher routes other requests to the projected package")
    if _normalized_variant(variant) == "prefill":
        return select_v030_gqa_fresh_prefill_launch(
            shape, dtype,
            {**attributes, "num_kv_splits": (
                attributes["num_kv_splits"]
                if "num_kv_splits" in attributes else 1)}, hardware)
    require_variant(variant, ("combine", "split_combine"), "FLASH_ATTN")
    return _combine_launch(shape, dtype, attributes)


def select_fa3_prepare_varlen_launch(batch: int) -> LaunchConfig:
    """``prepare_varlen_num_blocks`` launch (flash_prepare_scheduler.cu).

    ``num_warps = ceil(b / 31)`` rounded up by NUM_WARP_SWITCH to a power of
    two (at most 32), ``num_ctas = ceil(b / (31 * 32))``.
    """

    if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
        raise ValueError("prepare_varlen batch must be a positive integer")
    warps = ceil_div(batch, 31)
    switched = 1
    while switched < warps:
        switched *= 2
    if switched > 32:
        switched = 32
    return LaunchConfig(
        algorithm="vllm_fa3_prepare_varlen_num_blocks",
        grid=(ceil_div(batch, 31 * 32), 1, 1),
        block=(32 * switched, 1, 1),
        execution_variant="fa3_prepare_varlen",
    )


def select_fa3_mla_forward_specs(
        shape, dtype, variant, attributes, hardware, *,
        scheduler_metadata_precomputed: bool):
    """Source call sequence for the owned FA3 MLA domains.

    This function does not turn FlashMLA or FlashInfer into FA3. Context
    calls are individual unsplit/noncausal prefill chunks; their wrapper's
    gather/merge stays with its model owner. Uniform absorbed decode can
    have more than one64-row packed-Q tile per request. D256 expanded
    prefill returns ordinary MHA specs with partial source-stage pricing.
    Source selection derives the known wide-combine sequence; prediction
    must still validate its shared memory contract and may reject it.
    """
    from src.kernel.kernel_spec import KernelKind, KernelSpec

    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("FA3 MLA source selection requires H200")
    if type(scheduler_metadata_precomputed) is not bool:
        raise TypeError("FA3 MLA metadata lifetime must be explicit")
    if "return_softmax_lse" in attributes and type(attributes["return_softmax_lse"]) is not bool:
        raise TypeError("FA3 caller LSE return flag must be boolean")
    if variant not in ("mla_prefill", "mla_decode"):
        raise NotImplementedError("This source call is not a FA3 MLA program")
    if (variant == "mla_prefill" and dtype == "bf16"
            and attributes["kv_layout"] == "mla"
            and (shape["qk_head_dim"], shape["v_head_dim"], shape["kv_lora_rank"])
                == (256, 256, 512)):
        if "attention_backend" in attributes:
            raise NotImplementedError("Expanded MLA prefill requires the FA3 caller")
        if "return_softmax_lse" in attributes and attributes["return_softmax_lse"]:
            raise NotImplementedError("Expanded D256 FA3 final-LSE caller binding is not yet defined")
        # MLACommon.forward_mha expands latent K/V into one K/V head per
        # query head. GLM's256/256 call is therefore ordinary BF16 MHA,
        # rather than the192/128 MLA mechanism measured by the older probe.
        # Return its physical MHA specs and retain explicit partial pricing;
        # the model owner still binds expansion, gather and state merge.
        mha_shape = {name: shape[name] for name in (
            "batch", "q_len", "kv_len", "num_query_heads")}
        mha_shape.update({"num_kv_heads": shape["num_query_heads"], "head_dim": 256})
        return select_v030_gqa_forward_specs(
            mha_shape, dtype, {**attributes, "kv_layout": "contiguous"}, hardware,
            page_size=None, num_splits=0,
            scheduler_metadata_precomputed=scheduler_metadata_precomputed)
    if (variant == "mla_prefill" and dtype == "bf16"
            and attributes["kv_layout"] == "mla" and attributes["causal"] is False
            and (shape["qk_head_dim"], shape["v_head_dim"], shape["kv_lora_rank"])
                == (192, 128, 512)):
        q_lens, kv_lens = _request_lengths(shape, attributes)
        splits = _varlen_num_splits(
            batch=len(q_lens), max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
            num_heads=shape["num_query_heads"], num_heads_k=shape["num_query_heads"],
            head_dim=192, head_dim_v=128, block_m=128, block_n=128,
            causal=False, num_sms=int(hardware.num_compute_units))
        if splits > 1:
            return _select_fa3_split_prefill_specs(
                shape, dtype, attributes, hardware, splits=splits,
                scheduler_metadata_precomputed=scheduler_metadata_precomputed)
    if not owns_select_target_mla_launch(shape, dtype, variant, attributes, hardware):
        raise NotImplementedError("FA3 MLA call is outside the pinned source/mechanism domain")
    main_launch = select_target_mla_launch(shape, dtype, variant, attributes, hardware)
    if main_launch.algorithm not in (FA3_PREFILL_ALGORITHM, FA3_DECODE_ALGORITHM):
        raise NotImplementedError("The selected MLA backend is not FA3")
    main = KernelSpec(
        kind=KernelKind.FLASH_ATTN, hardware_id=hardware.hardware_id,
        shape=shape, dtype=dtype, variant=main_launch.execution_variant,
        launch=main_launch, attributes=attributes)
    parts = []
    if not scheduler_metadata_precomputed:
        parts.append(KernelSpec(
            kind=KernelKind.VECTOR, hardware_id=hardware.hardware_id,
            shape={"batch": shape["batch"]}, dtype="int32", variant="fa3_prepare_varlen",
            launch=select_fa3_prepare_varlen_launch(shape["batch"]),
            attributes={"num_splits_static": main_launch.tiling.num_kv_splits}))
    parts.append(main)
    splits = main_launch.tiling.num_kv_splits
    if splits > 1:
        combine_shape = {
            "batch": shape["batch"], "q_len": shape["q_len"], "kv_len": shape["kv_len"],
            "head_dim": shape["v_head_dim"], "num_query_heads": shape["num_query_heads"],
            "num_kv_heads": 1}
        combine_attributes = {name: attributes[name] for name in (
            "causal", "kv_layout", "kv_dist", "q_lens", "kv_lens")}
        graph_fields = ("graph_logical_tokens", "graph_model_input_tokens", "graph_runtime_mode")
        if any(name in attributes for name in graph_fields):
            if not all(name in attributes for name in graph_fields):
                raise ValueError("FA3 MLA graph metadata must be complete")
            combine_attributes.update({name: attributes[name] for name in graph_fields})
        combine_launch = _combine_launch(
            combine_shape, dtype, {**combine_attributes, "num_kv_splits": splits})
        parts.append(KernelSpec(
            kind=KernelKind.FLASH_ATTN, hardware_id=hardware.hardware_id,
            shape=combine_shape, dtype=dtype, variant="combine", launch=combine_launch,
            attributes={**combine_attributes, "fa_ksteps": None, "fa_eff_mfrac": 1.0}))
    elif scheduler_metadata_precomputed:
        parts.append(KernelSpec(
            kind=KernelKind.VECTOR, hardware_id=hardware.hardware_id,
            shape={"element_count": 1}, dtype="int32", variant="elementwise",
            launch=LaunchConfig(algorithm="elementwise", grid=(1, 1, 1),
                                block=(128, 1, 1), execution_variant="elementwise"),
            attributes={"reads": 0, "writes": 1, "operation": "fill"}))
    return tuple(parts)


def _select_fa3_split_prefill_specs(shape, dtype, attributes, hardware, *,
                                   splits, scheduler_metadata_precomputed):
    """Pinned expanded192/128 noncausal context call and128-wide combine.

    This factory is explicit. The ordinary MLA dispatcher remains unsplit
    until the shared model lowerer consumes this complete source sequence.
    """
    from src.kernel.kernel_spec import KernelKind, KernelSpec

    q_lens, _ = _request_lengths(shape, attributes)
    if "return_softmax_lse" in attributes and type(attributes["return_softmax_lse"]) is not bool:
        raise TypeError("FA3 caller LSE return flag must be boolean")
    # Only the final call exposes the library result. Main writes partial
    # LSE for split requests and direct final LSE for dynamic-one requests;
    # combine preserves those bypassed rows in the same final allocation.
    main_attributes = {name: value for name, value in attributes.items() if name != "return_softmax_lse"}
    main = KernelSpec(
        kind=KernelKind.FLASH_ATTN, hardware_id=hardware.hardware_id,
        shape=shape, dtype=dtype, variant="mla_prefill", attributes=main_attributes,
        launch=LaunchConfig(
            algorithm=FA3_PREFILL_SPLIT_ALGORITHM,
            grid=(int(hardware.num_compute_units), 1, 1), block=(384, 1, 1),
            cluster=(1, 1, 1), execution_variant="mla_prefill",
            tiling=TilingSpec(tile_shape={"q": 128, "kv": 128,
                "qk_head": 192, "value_head": 128}, pipeline_depth=2,
                num_kv_splits=splits), parameters=_source_parameters(splits, False)))
    rows, batch, heads = 8, shape["batch"], shape["num_query_heads"]
    grid = ((ceil_div(shape["q_len"] * heads, rows), 1, batch)
            if 2 * ceil_div(sum(q_lens), rows) >= batch * ceil_div(shape["q_len"], rows)
            else (ceil_div(sum(q_lens) * heads, rows) + batch, 1, 1))
    combine_shape = {name: shape[name] for name in ("batch", "q_len", "kv_len", "num_query_heads")}
    combine_shape.update({"head_dim": 128, "num_kv_heads": heads})
    combine_attributes = {name: attributes[name] for name in ("causal", "kv_dist", "q_lens", "kv_lens")}
    combine_attributes.update({"kv_layout": "contiguous", "fa_ksteps": None, "fa_eff_mfrac": 1.0})
    if "return_softmax_lse" in attributes:
        combine_attributes["return_softmax_lse"] = attributes["return_softmax_lse"]
    combine = KernelSpec(
        kind=KernelKind.FLASH_ATTN, hardware_id=hardware.hardware_id,
        shape=combine_shape, dtype=dtype, variant="combine", attributes=combine_attributes,
        launch=LaunchConfig(
            algorithm=FA3_PREFILL_COMBINE_ALGORITHM, grid=grid, block=(256, 1, 1),
            execution_variant="combine", parameters=_source_parameters(splits, False),
            tiling=TilingSpec(tile_shape={"rows": rows, "value_head": 128},
                              pipeline_depth=4, num_kv_splits=splits)))
    parts = []
    if not scheduler_metadata_precomputed:
        parts.append(KernelSpec(
            kind=KernelKind.VECTOR, hardware_id=hardware.hardware_id,
            shape={"batch": batch}, dtype="int32", variant="fa3_prepare_varlen",
            launch=select_fa3_prepare_varlen_launch(batch),
            attributes={"num_splits_static": splits}))
    return (*parts, main, combine)


def select_v030_sparse_mla_forward_specs(shape, dtype, attributes, hardware, *, topk_capacity):
    """FA3 library call inside FlashAttnMLASparseImpl._run_mqa_kernel.

    Each query token is a separate Q1 sequence. The wrapper supplies page1
    global indices and used lengths, but passes the top-k table width as
    max_seqlen_k. It omits num_splits/scheduler metadata, selecting source
    default0 and an internal prepare. Caller arange/index conversion and
    physical index/readiness bindings remain outside this library call.

    BF16, nonnegative used lengths, no graph padding. Explicit per-token
    lengths retain their own source tag and single prepare/main/combine
    call; do not regroup them or replace them with candidate capacity.
    All-zero dynamic prepare has a zero divisor in the pinned source and
    remains rejected; static-unsplit zero-K calls keep their zero epilogue.
    """
    from src.kernel.kernel_spec import KernelKind, KernelSpec

    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("FA3 sparse MLA source selection requires H200")
    if dtype != "bf16":
        raise NotImplementedError("FA3 sparse MLA mechanism domain is BF16")
    if type(topk_capacity) is not int or topk_capacity <= 0:
        raise ValueError("FA3 sparse MLA top-k capacity must be a positive integer")
    if (shape["qk_head_dim"], shape["v_head_dim"], shape["kv_lora_rank"]) != (576, 512, 512):
        raise NotImplementedError("FA3 sparse MLA requires latent512 plus rope64")
    if "source_program" in attributes and attributes["source_program"] != "vllm_0_30_fa3_sparse_mla":
        raise ValueError("FA3 sparse MLA source identity differs from its library caller")
    attributes = {**attributes, "source_program": "vllm_0_30_fa3_sparse_mla"}
    q_lens, used = _request_lengths(shape, attributes)
    if (shape["q_len"] != 1 or any(q != 1 for q in q_lens)
            or max(used) > topk_capacity
            or attributes["kv_layout"] != "mla" or attributes["kv_dist"] not in ("uniform", "explicit")
            or attributes["kv_dist"] == "uniform" and len(set(used)) != 1
            or attributes["causal"] is not True):
        raise NotImplementedError("FA3 sparse MLA requires one query per sequence and truthful nonnegative used lengths")
    if attributes["full_cuda_graphs"] is not False or attributes["max_cudagraph_capture_size"] != 0:
        raise NotImplementedError("FA3 sparse MLA graph/zero-row contract is unbound")
    if "attention_backend" in attributes or any(name in attributes for name in (
            "num_kv_splits",
            "graph_logical_tokens", "graph_model_input_tokens", "graph_runtime_mode",
            "model_input_shape", "model_input_q_lens", "model_input_kv_lens")):
        raise NotImplementedError("FA3 sparse MLA requires its own unpadded source caller")
    heads, batch = require_shape(shape, ("num_query_heads", "batch"))
    splits = _varlen_num_splits(
        batch=batch, max_seqlen_q=1, max_seqlen_k=topk_capacity,
        num_heads=heads, num_heads_k=1, head_dim=64, head_dim_v=512,
        block_m=64, block_n=64, causal=False, num_sms=int(hardware.num_compute_units))
    # Disable the generic one-index-per-compute-panel projection: page1
    # indirection reads one index per selected KV row. The sparse source
    # work owns that demand, including unsplit calls, in both predictors.
    main_parameters = _source_parameters(splits, False)
    if not any(used) and splits > 1:
        raise NotImplementedError("FA3 sparse MLA all-zero dynamic prepare has a source zero divisor")
    main = KernelSpec(
        kind=KernelKind.FLASH_ATTN, hardware_id=hardware.hardware_id,
        shape=shape, dtype=dtype, variant="mla_decode", attributes=attributes,
        launch=LaunchConfig(
            algorithm=FA3_SPARSE_MLA_030_ALGORITHM,
            grid=(int(hardware.num_compute_units), 1, 1), block=(384, 1, 1),
            cluster=(1, 1, 1), execution_variant="mla_decode",
            tiling=TilingSpec(tile_shape={"q": 64, "kv": 64, "qk_head": 64,
                "value_head": 512, "kv_page": 1, "kv_capacity": topk_capacity},
                pipeline_depth=2, num_kv_splits=splits), parameters=main_parameters))
    parts = [KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware.hardware_id,
        shape={"batch": batch}, dtype="int32", variant="fa3_prepare_varlen",
        launch=select_fa3_prepare_varlen_launch(batch),
        attributes={"num_splits_static": splits}), main]
    if splits > 1:
        combine_shape = {name: shape[name] for name in ("batch", "q_len", "kv_len", "num_query_heads")}
        combine_shape.update({"head_dim": 512, "num_kv_heads": 1})
        common = {name: attributes[name] for name in ("causal", "kv_layout", "kv_dist", "q_lens", "kv_lens")}
        common.update({"fa_ksteps": None, "fa_eff_mfrac": 1.0,
                       "source_program": "vllm_0_30_fa3_sparse_mla"})
        parts.append(KernelSpec(
            kind=KernelKind.FLASH_ATTN, hardware_id=hardware.hardware_id,
            shape=combine_shape, dtype=dtype, variant="combine", attributes=common,
            launch=LaunchConfig(
                algorithm=FA3_SPARSE_MLA_COMBINE_030_ALGORITHM,
                grid=((ceil_div(heads, 8), 4, batch)
                      if 2 * ceil_div(batch, 8) >= batch else
                      (ceil_div(batch * heads, 8) + batch, 4, 1)), block=(256, 1, 1),
                execution_variant="combine", parameters=_source_parameters(splits, False),
                tiling=TilingSpec(tile_shape={"rows": 8, "value_head": 128,
                    "kv_page": 1, "kv_capacity": topk_capacity},
                    pipeline_depth=4, num_kv_splits=splits))))
    return tuple(parts)


def select_v030_gqa_full_decode_specs(shape, dtype, attributes, hardware, *, page_size,
                                      num_splits, scheduler_metadata_precomputed):
    """Declared vLLM0.30 captured Q1 caller, keeping real zero request tails."""
    from dataclasses import replace
    required = {"source_program", "causal", "kv_layout", "kv_dist", "q_lens", "kv_lens",
        "graph_runtime_mode", "graph_model_input_tokens", "model_input_q_lens", "model_input_kv_lens"}
    optional = {"source_model_execution_version", "fa_ksteps", "fa_eff_mfrac"}
    if (not required.issubset(attributes) or set(attributes) - required - optional
            or type(scheduler_metadata_precomputed) is not bool or type(num_splits) is not int
            or "source_model_execution_version" in attributes and attributes["source_model_execution_version"] != "vllm_0_30_0"
            or "fa_ksteps" in attributes and attributes["fa_ksteps"] is not None
            or "fa_eff_mfrac" in attributes and (type(attributes["fa_eff_mfrac"]) not in (int,float)
                or attributes["fa_eff_mfrac"] != 1.0)):
        raise ValueError("FA3 FULL requires its complete exact caller fields without overrides")
    if (hardware.hardware_id != "nvidia_h200" or dtype not in ("bf16", "fp16") or shape["q_len"] != 1
            or shape["head_dim"] != 128 or attributes["causal"] is not True
            or attributes["kv_layout"] != "paged_varlen" or num_splits != FA3_GQA_DEFAULT_GRAPH_SPLITS_030):
        raise ValueError("FA3 FULL source requires its exact H200 BF16/FP16 Q1 default split caller")
    plain = {name: attributes[name] for name in ("causal", "kv_layout", "kv_dist", "q_lens", "kv_lens")}
    parts = select_v030_gqa_forward_specs({**shape, "batch": len(plain["q_lens"])}, dtype, plain, hardware,
        page_size=page_size, num_splits=(1 if shape["batch"] > _PREPARE_VARLEN_MAX_BATCHES_1CTA else num_splits),
        scheduler_metadata_precomputed=scheduler_metadata_precomputed)
    full = {name: attributes[name] for name in ("source_program", "graph_runtime_mode",
        "graph_model_input_tokens", "model_input_q_lens", "model_input_kv_lens")}
    result = []
    for stage in parts:
        if stage.kind.value == "VECTOR":
            result.append(replace(stage, shape={"batch": shape["batch"]},
                launch=select_fa3_prepare_varlen_launch(shape["batch"])) if stage.variant == "fa3_prepare_varlen" else stage)
            continue
        launch = stage.launch
        if stage.variant == "combine":
            physical = full["graph_model_input_tokens"]
            rows = launch.tiling.tile_shape["rows"]
            heads = shape["num_query_heads"]
            value_blocks = launch.grid[1]
            grid = ((ceil_div(heads, rows), value_blocks, shape["batch"])
                if 2 * ceil_div(physical, rows) >= shape["batch"] else
                (ceil_div(physical * heads, rows) + shape["batch"], value_blocks, 1))
            launch = replace(launch, grid=grid)
        stage = replace(stage, shape=shape, launch=launch, attributes={**stage.attributes, **full})
        validate_v030_gqa_observed_launch(stage, hardware)
        result.append(stage)
    return tuple(result)
