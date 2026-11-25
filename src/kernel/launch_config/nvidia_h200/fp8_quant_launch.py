"""Complete vLLM0.30 dynamic group-FP8 producer calls on H200.

fp8_utils.py selects CUDA for a contiguous input and Triton otherwise.
QuantFP8.forward_cuda's normal H200 oracle retains FP32 scale storage;its
packed choice is gated to SM100/120 in deep_gemm.py. The explicit packed
UE8M0 wrapper itself supports SM90. Packed non-contiguous inputs first
materialize via ATen Copy;scale packing/padding is in the quant kernel.
Only source geometry/layout/allocation facts are supplied,not service rates,
compiler register estimates,physical traffic or deployment identity.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape


def group_quant_cuda_geometry(rows, groups_per_row, group_size, element_bytes):
    """per_token_group_quant_8bit's unconditional non-autotuned host rule."""
    groups = rows * groups_per_row
    policy = (rows + rows % 2) * groups_per_row
    per_block = next(size for size in (16, 8, 4, 2, 1) if policy % size == 0)
    return ceil_div(groups, per_block), 16 * per_block, per_block * group_size * element_bytes, per_block


def select_group_fp8_quant_recipe(shape, dtype, attributes, hardware):
    """Every submission for a selected2D group128 quantization producer.

    entry is quant_fp8_forward_cuda,per_token_group_quant_fp8,or the explicit
    per_token_group_quant_fp8_packed_for_deepgemm wrapper. Input row stride
    and resolved scale rounding are actual caller facts. out_q is fresh or
    an actual distinct contiguous same-shape FP8 tensor. No backend is
    inferred from duration and no missing oracle result receives a default.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("group FP8 producer source requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16", "fp32"))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_dynamic_group_fp8_quant"
            or attributes["source_input_layout"] != "row_major_unit_last_dim_aligned16"
            or attributes["source_output_q"] not in ("fresh_contiguous", "provided_distinct_contiguous_same_shape")
            or attributes["source_quant_output_dtype"] != "fp8_e4m3fn"):
        raise NotImplementedError("group FP8 quant requires its exact0.30 H2002D input and FP8 output source domain")
    columns, group, stride = require_shape(shape, ("columns", "group_size", "input_row_stride"))
    rows = shape["rows"]
    if type(rows) is not int or rows < 0:
        raise ValueError("group FP8 rows must be a non-negative integer")
    if group != 128 or columns % group or stride < columns:
        raise NotImplementedError("group FP8 producer binds non-overlapping2D rows with complete128-value groups")
    ue8m0 = attributes["source_scale_ue8m0"]
    if type(ue8m0) is not bool:
        raise TypeError("group FP8 source_scale_ue8m0 must be an actual resolved boolean")
    entry = attributes["source_entry"]
    packed = entry == "per_token_group_quant_fp8_packed_for_deepgemm"
    if entry == "quant_fp8_forward_cuda":
        # This is the dynamic group branch;static/per-token/per-tensor are
        # different operators. use_triton is not read by this source branch.
        if attributes["source_static_quant"] is not False or attributes["source_input_scale"] != "none":
            raise NotImplementedError("selected QuantFP8 source requires dynamic group quant withno input scale")
        if ue8m0:
            supported = attributes["source_use_deep_gemm_supported"]
            if type(supported) is not bool:
                raise TypeError("QuantFP8 requires its actual DeepGEMM support result")
            if supported:
                scale_format = attributes["source_deepgemm_scale_format"]
                if scale_format not in ("float32", "float32_ceil_ue8m0", "ue8m0"):
                    raise ValueError("QuantFP8 requires the actual DeepGEMM scale-format oracle result")
                if scale_format == "ue8m0":
                    raise NotImplementedError("normal H200 QuantFP8 oracle doesnot select SM100/120 packed storage;foreign cached-oracle scope is unbound")
    elif entry not in ("per_token_group_quant_fp8", "per_token_group_quant_fp8_packed_for_deepgemm"):
        raise NotImplementedError("group FP8 source entry is not bound")
    groups_per_row = columns // group
    if max(rows, columns, rows * groups_per_row) >= 2**31:
        raise NotImplementedError("group FP8 source counters must fit their signed-int host/kernel domain")
    contiguous = rows <= 1 or stride == columns
    width = 4 if dtype == "fp32" else 2
    if rows and width * ((rows - 1) * stride + columns - 1) >= 2**63:
        raise NotImplementedError("group FP8 source strided byte offsets exceed signed int64")
    if packed:
        require_dtype(dtype, ("bf16", "fp16"))
        if not ue8m0:
            raise ValueError("packed DeepGEMM source requires resolved UE8M0 scales")
        if rows == 0:
            raise NotImplementedError("packed wrapper hasno source zero-row guard before its zero-grid launch")
        prefix = ()
        if not contiguous:
            from src.kernel.launch_config.nvidia_h200.torch_vector_launch import select_torch213_strided_copy_launch
            copy = select_torch213_strided_copy_launch({"element_count": rows * columns}, dtype, {
                "tensor_iterator_source": attributes["tensor_iterator_source"], "source_input_layout": "strided_nocast",
                "source_copy_flags": "same_nonnegated_real_dtype",
                "source_maximum_offset_bytes": width * ((rows - 1) * stride + columns - 1)}, hardware)
            prefix = (("input_contiguous_copy", copy),)
        packed_columns = ceil_div(groups_per_row, 4)
        aligned_rows = ceil_div(rows, 4) * 4
        padded_groups = packed_columns * 4
        kx = 16 if padded_groups % 16 == 0 else 8 if padded_groups % 8 == 0 else 4
        ry = 16 // kx
        grid = (ceil_div(aligned_rows, ry), padded_groups // kx, 1)
        if grid[0] >= 2**31 or grid[1] > 65535:
            raise NotImplementedError("packed FP8 source grid exceeds CUDA dimension bounds")
        # empty_strided([rows,packed_columns],[1,aligned_rows]) has a minimal
        # storage extent,not the whole aligned rectangle. The kernel's
        # num_scale_elems uses this exact extent for in-line padding guards.
        scale_elements = rows + (packed_columns - 1) * aligned_rows
        launch = LaunchConfig(f"vllm_0_30_sm90_group128_fp8_packed_{dtype}_kx{kx}_ry{ry}", grid, (128, 1, 1),
            parameters={"dynamic_shared_bytes": 0.0, "source_programmatic_stream_serialization": 1.0,
                "source_scale_row_stride": 1.0, "source_scale_column_stride": float(aligned_rows),
                "source_scale_allocation_bytes": float(scale_elements * 4),
                "source_packed_scale_columns": float(packed_columns), "source_output_q_rows": float(rows)})
        return (*prefix, ("group_fp8_quant_packed", launch))
    column_major, tma = attributes["source_column_major_scales"], attributes["source_tma_aligned_scales"]
    if type(column_major) is not bool or type(tma) is not bool:
        raise TypeError("group FP8 scale layout flags must be actual booleans")
    if rows == 0:
        return ()
    if column_major:
        scale_col_stride = ceil_div(rows, 4) * 4 if tma else rows
        scale_row_stride = 1
        scale_elements = rows + (groups_per_row - 1) * scale_col_stride
    else:
        scale_col_stride, scale_row_stride = 1, groups_per_row
        scale_elements = rows * groups_per_row
    params = {"source_scale_row_stride": float(scale_row_stride), "source_scale_column_stride": float(scale_col_stride),
        "source_scale_allocation_bytes": float(scale_elements * 4), "source_output_q_rows": float(rows)}
    scale = "ue8m0" if ue8m0 else "fp32"
    if contiguous:
        blocks, threads, shared, per_block = group_quant_cuda_geometry(rows, groups_per_row, group, width)
        # C++ ignores the wrapper's dummy layout flags;the actual output
        # strides choose COL_MAJOR,including the single-row degenerate case.
        col_template = scale_row_stride < scale_col_stride
        bounds = "bounds" if rows * groups_per_row % per_block else "full"
        params.update(dynamic_shared_bytes=float(shared), source_programmatic_stream_serialization=1.0)
        launch = LaunchConfig(f"vllm_0_30_sm90_group128_fp8_cuda_{dtype}_{scale}_col{int(col_template)}_g{per_block}_{bounds}",
            (blocks, 1, 1), (threads, 1, 1), parameters=params)
    else:
        # The fallback consumes real strided rows directly;it does not first
        # materialize input. group128 fixes BLOCK128,one warp andone stage.
        layout = "col" if column_major else "row"
        launch = LaunchConfig(f"vllm_0_30_sm90_group128_fp8_triton_{dtype}_{scale}_{layout}",
            (rows * groups_per_row, 1, 1), (32, 1, 1), parameters=params)
    return (("group_fp8_quant", launch),)
