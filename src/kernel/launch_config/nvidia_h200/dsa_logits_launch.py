"""SM90 DSA logits from vLLM0.30's pinned DeepGEMM host wrappers.

Source identity is owned by vLLM0.30 cmake/external_projects/deepgemm.cmake.
In that revision, csrc/apis/attention.hpp and
csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp fix the dispatch, grid,
threads, cluster and dynamic shared allocation below. Registers and static
shared require separate compilation evidence. No production duration enters.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape


def _require_source(attributes, hardware):
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("SM90 DSA logits require H200 HWSpec")
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_deepgemm_snapshot"] != "vllm_0_30_cmake_pin"
            or attributes["source_q_scale"] != "none_folded_into_fp32_weights"
            or attributes["source_input_layout"] != "contiguous_aligned_16"
            or attributes["clean_logits"] is not False):
        raise NotImplementedError("DSA logits require the pinned0.30 FP8 source and explicit source layout")
    sms = attributes["source_num_sms"]
    if type(sms) is not int or not 0 < sms <= hardware.num_compute_units:
        raise ValueError("source_num_sms must be a positive H200 runtime SM budget")
    return sms


def _align(value, alignment):
    return ceil_div(value, alignment) * alignment


def select_dsa_prefill_logits_launch(shape, dtype, attributes, hardware):
    """One complete nonpaged logits call; output FP32, uncompressed layout."""
    require_dtype(dtype, ("fp8",))
    sms = _require_source(attributes, hardware)
    rows, kv_rows, heads, dim = require_shape(shape, ("rows", "kv_rows", "num_heads", "head_dim"))
    if (heads not in (32, 64) or dim != 128 or rows * heads >= 2**31
            or _align(kv_rows + 256, 256) >= 2**31):
        raise NotImplementedError("DSA prefill logits bind32/64 heads,head128 and signed-int row extents")
    if attributes["source_kv_scale_dtype"] != "fp32" or attributes["source_schedule_metadata"] != "none":
        raise NotImplementedError("SM90 nonpaged logits require FP32 KV scales and no schedule tensor")
    block_q, block_kv = 128 // heads, 256
    # Three Q/KV stages, FP8 values, FP32 weights and FP32 KV scales.
    smem = 3 * (block_q * heads * dim + block_q * heads * 4
                + block_kv * dim + block_kv * 4)
    smem += (3 * 2 + 3 * 2 + (512 // 128) * 2) * 8 + 4
    return LaunchConfig(f"deepgemm_vllm_0_30_sm90_fp8_mqa_logits_h{heads}_d128",
        (sms, 1, 1), (640, 1, 1), parameters={"dynamic_shared_bytes": float(smem)})


def dsa_prefill_logits_layout(shape):
    """Actual FP32 output allocation, including aligned rows and guard KV tile.

    attention.hpp adds a256-value KV tile before aligning the row stride;
    the public view exposes only kv_rows. Do not use kv_rows as its stride.
    """
    rows, kv_rows, heads = require_shape(shape, ("rows", "kv_rows", "num_heads"))
    if (heads not in (32, 64) or rows * heads >= 2**31
            or _align(kv_rows + 256, 256) >= 2**31):
        raise NotImplementedError("SM90 DSA logits layout requires32/64 heads")
    return _align(rows, 128 // heads), _align(kv_rows + 256, 256)


def select_dsa_decode_logits_launch(shape, dtype, attributes, hardware):
    """Paged logits with explicit two-dimensional context/schedule facts."""
    require_dtype(dtype, ("fp8",))
    sms = _require_source(attributes, hardware)
    batch, next_n, heads, dim, page, capacity = require_shape(shape, (
        "batch", "next_n", "num_heads", "head_dim", "block_size", "max_model_len"))
    if (heads not in (32, 64) or dim != 128 or next_n not in (1, 2, 4)
            or page not in (32, 64) or batch * next_n * heads >= 2**31
            or _align(capacity, 256) >= 2**31):
        raise NotImplementedError("SM90 paged logits require native next_n1/2/4,32/64 heads and32/64-token pages")
    if (attributes["source_context_lens_layout"] != "contiguous_int32_2d"
            or attributes["source_indices"] != "none"
            or attributes["source_kv_cache_layout"] != "paged_fp8_128_plus_fp32_scale"
            or type(attributes["source_block_table_inner_stride"]) is not int
            or attributes["source_block_table_inner_stride"] != 1):
        raise NotImplementedError("SM90 paged logits require their explicit cache/context/block-table source layout")
    multicast = 2 if next_n == 4 else 1
    if sms % multicast:
        raise ValueError("paged logits SM budget must fit complete multicast clusters")
    schedule_shape = attributes["source_schedule_shape"]
    if (not isinstance(schedule_shape, (tuple, list)) or len(schedule_shape) != 2
            or any(type(value) is not int for value in schedule_shape)
            or tuple(schedule_shape) != (sms // multicast + 1, 2)):
        raise ValueError("paged logits schedule shape must match actual SM90 cluster count")
    q_rows = next_n // multicast
    alignment = dim * 8
    q_pipe = 3 * (q_rows * heads * dim + _align(q_rows * heads * 4, alignment)) + _align(3 * 8 * 2, alignment)
    kv_pipe = 3 * (64 * dim + _align(64 * 4, alignment)) + _align(3 * 8 * 2, alignment)
    smem = q_pipe + 4 * kv_pipe + 4 * 2 * 8 + 4
    return LaunchConfig(f"deepgemm_vllm_0_30_sm90_fp8_paged_mqa_logits_n{next_n}_h{heads}_p{page}",
        (sms, 1, 1), (640, 1, 1), cluster=(multicast, 1, 1),
        parameters={"dynamic_shared_bytes": float(smem)})


def select_dsa_decode_metadata_launch(shape, dtype, attributes, hardware):
    """Metadata producer, outside the logits call when the caller provides it."""
    require_dtype(dtype, ("int32",))
    sms = _require_source(attributes, hardware)
    batch, next_n, page = require_shape(shape, ("batch", "next_n", "block_size"))
    if (batch >= 2**31 or next_n not in (1, 2, 4) or page not in (32, 64)
            or attributes["source_context_lens_layout"] != "contiguous_int32_2d"
            or attributes["source_indices"] != "none"):
        raise NotImplementedError("SM90 logits metadata requires native non-varlen context rows")
    multicast = 2 if next_n == 4 else 1
    if sms % multicast:
        raise ValueError("metadata SM budget must fit complete multicast clusters")
    shared = _align(batch, 32) * 4
    # SM90ArchSpec::smem_capacity in the same pinned host-source revision.
    if shared > 232448:
        raise NotImplementedError("metadata allocation exceeds the SM90 source shared-memory capacity")
    return LaunchConfig("deepgemm_vllm_0_30_sm90_paged_mqa_logits_metadata",
        (1, 1, 1), (32, 1, 1),
        parameters={"dynamic_shared_bytes": float(shared)})
