"""SM90 FlashKDA from the unique vLLM0.30 cmake/external_projects pin.

flash_kda.cpp transposes beta to contiguous[H,T], then launches prepare
and recurrence. Packed varlen prepare deliberately uses ceil(T/16)+N CTAs
per head, including early-return excess tiles. Recurrence uses VD64 iff
2*H*N <= the actual device SM count; otherwise VD128. Both paths are
source-defined, with no autotune or other-hardware selection table.

Dynamic shared follows fwd_kernel{1,2}.cuh SharedStorage and the pinned
CUTLASS gitlink's eight-byte barriers/pipeline arrays. Layout_K_INTER
tile_to_shape layouts have dense cosize here. Union lifetimes, per-field
alignment and final struct padding are preserved. This is source layout
derivation, not a register estimate or proof of the installed binary.
"""

import math

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.nvidia_h200.torch_vector_launch import select_torch213_strided_copy_launch
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_positive_integer, require_shape


def _aligned_struct_bytes(fields):
    offset, alignment = 0, 1
    for size, align in fields:
        offset = ((offset + align - 1) // align) * align + size
        alignment = max(alignment, align)
    return ceil_div(offset, alignment) * alignment


def _flashkda_shared_bytes(value_tile):
    qk, lm, beta, gtotal = 16 * 128 * 2, 16 * 16 * 2, 32 * 2, 128 * 4
    phase_a = 2 * qk + 16 * 128 * 4
    phase_b = 3 * qk + 16 * 16 * 4 + 2 * lm
    prepare = _aligned_struct_bytes(((max(phase_a, phase_b), 128), (beta, 128),
        (16 * 4, 16), (16 * 4, 16), (16 * 4, 16), (4, 4),
        (qk, 128), (gtotal, 128), (8, 16)))
    vo, state = 16 * value_tile * 2, value_tile * 128
    inputs = _aligned_struct_bytes(((vo, 128), (beta, 128), (qk, 128),
        (qk, 128), (qk, 128), (gtotal, 128), (lm, 128), (lm, 128)))
    recurrence = _aligned_struct_bytes(((state * 2, 128),
        (max(3 * inputs + 2 * vo, state * 4), 128),
        (3 * (8 + 8), 8), (2 * (8 + 8), 8), (8, 16)))
    return prepare, recurrence


def flashkda_workspace_extents(shape, attributes):
    """Source allocation upper bound; not traffic, Fill or normal capacity."""
    rows, heads, dim = require_shape(shape, ("rows", "heads", "head_dim"))
    if dim != 128 or heads not in (6, 12, 24, 48, 96):
        raise NotImplementedError("FlashKDA binds128-wide K3 TP1/2/4/8/16 heads")
    starts = attributes["query_start_loc"]
    if (not isinstance(starts, (tuple, list)) or len(starts) < 2
            or any(type(value) is not int for value in starts)
            or starts[0] != 0 or starts[-1] != rows
            or any(a >= b for a, b in zip(starts, starts[1:]))):
        raise ValueError("FlashKDA requires complete strictly increasing packed query segments")
    sequences = len(starts) - 1
    tiles = ceil_div(rows, 16) + sequences
    if max(rows, heads, sequences, tiles) >= 2**31 or sequences > 65535:
        raise NotImplementedError("FlashKDA packed host fields/grid exceed their integer domain")
    per_tile = 3 * 16 * dim * 2 + dim * 4 + 2 * 16 * 16 * 2
    size = heads * tiles * per_tile
    if size >= 2**63:
        raise NotImplementedError("FlashKDA workspace size exceeds signed int64")
    return {"bytes": size, "tiles_upper_bound": tiles, "per_head_tile_bytes": per_tile,
        "sequences": sequences}


def select_flashkda_prefill_recipe(shape, dtype, attributes, hardware):
    """Full prepared library call: beta materialization, prepare, recurrence.

    Normal K3 FP32 initial/final tensors are already prepared and supplied;
    checkpointing is disabled. Beta retains its actual packed row stride.
    The caller supplies the real workspace allocation, which may exceed the
    per-call requirement. No chunk64 metadata or Triton Q/K norms belong to
    this implementation; FlashKDA's prepare fuses normalization and gates.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("FlashKDA SM90 source requires H200 HWSpec")
    require_dtype(dtype, ("bf16",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_kimi_k3_flashkda"
            or attributes["source_flashkda_source"] != "vllm_0_30_cmake_pin"
            or attributes["source_input_layout"] != "contiguous_aligned_16_qkvg_out_and_state"
            or attributes["source_state_mode"] != "initial_and_final_fp32"
            or attributes["source_checkpoint_state"] != "disabled"
            or attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"):
        raise NotImplementedError("FlashKDA requires its exact0.30 pinned SM90 prepared source")
    levels = attributes["source_kernel_build_levels"]
    if (not isinstance(levels, (tuple, list)) or len(levels) != 2
            or any(type(value) is not int or value != 1 for value in levels)):
        raise NotImplementedError("FlashKDA requires the source's full default K1/K2 build")
    index_dtype = attributes["source_cu_seqlens_dtype"]
    if index_dtype not in ("int32", "int64"):
        raise NotImplementedError("FlashKDA query segments require actual INT32/INT64 metadata")
    lower_bound = attributes["gate_lower_bound"]
    if (type(lower_bound) not in (int, float) or not math.isfinite(lower_bound)
            or not -5 <= lower_bound < 0):
        raise ValueError("K3 FlashKDA requires its actual bounded gate in[-5,0)")
    extents = flashkda_workspace_extents(shape, attributes)
    allocated = require_positive_integer(attributes, "source_workspace_bytes")
    if allocated < extents["bytes"] or allocated >= 2**63:
        raise ValueError("FlashKDA actual workspace allocation is too small or exceeds signed int64")
    rows, heads = shape["rows"], shape["heads"]
    stride = require_positive_integer(attributes, "source_beta_row_stride_elements")
    if stride < heads:
        raise ValueError("FlashKDA beta rows must not overlap")
    if rows == 1:
        beta_stage = ()  # singleton time: the transpose is already contiguous
    else:
        beta = select_torch213_strided_copy_launch({"element_count": rows * heads}, "bf16", {
            "tensor_iterator_source": attributes["tensor_iterator_source"],
            "source_input_layout": "strided_nocast", "source_copy_flags": "same_nonnegated_real_dtype",
            "source_maximum_offset_bytes": 2 * ((rows - 1) * stride + heads - 1)}, hardware)
        beta_stage = (("beta_transpose_copy", beta),)
    split = 2 * heads * extents["sequences"] <= int(hardware.num_compute_units)
    value_tile = 64 if split else 128
    shared_prepare, shared_recurrence = _flashkda_shared_bytes(value_tile)
    index_id = "i32" if index_dtype == "int32" else "i64"
    prepare = LaunchConfig("vllm_0_30_sm90_flashkda_prepare_" + index_id,
        (extents["tiles_upper_bound"], heads, 1), (128, 1, 1),
        parameters={"dynamic_shared_bytes": float(shared_prepare), "source_chunk_size": 16.0})
    recurrence = LaunchConfig(f"vllm_0_30_sm90_flashkda_recurrence_fp32_v{value_tile}_" + index_id,
        (heads * (128 // value_tile), extents["sequences"], 1), (192, 1, 1),
        parameters={"dynamic_shared_bytes": float(shared_recurrence), "source_chunk_size": 16.0,
            "source_value_tile": float(value_tile), "source_input_stages": 3.0, "source_output_stages": 2.0})
    return (*beta_stage, ("flashkda_prepare", prepare), ("flashkda_recurrence", recurrence))
