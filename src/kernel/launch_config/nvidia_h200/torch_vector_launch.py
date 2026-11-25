"""Versioned Torch SM90 TensorIterator host geometry from CUDALoops.cuh.

The pinned Torch2.8 CUDALoops.cuh/software identity is owned by
microbench/nvidia_a800/references/pytorch_moe_reduce_vllm_0_11_0_torch_2_8_0.json.
Only common header semantics are reused. Its SM90 vector cap is8 for BF16,
whereas SM80 caps the same expression at4. No other hardware selector or
compiled-resource numbers are used. Compiler resources remain unbound here.
Copy.cu routes homogeneous contiguous copies through allocator memcpyAsync
(or no work for the exact same view), not through this kernel selector.
The separate Torch2.13 Fill helper binds that exact official version's
Fill.cpp/FillKernel.cu/CUDALoops.cuh/thread_constants.h source independently.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape


_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "float32": 4,
                "int32": 4, "int64": 8, "bool": 1}


def _torch213_vectorized_gather_geometry(rows, row_bytes):
    """IndexKernelUtils.cu's common16-byte gather launcher."""
    return ((rows, min(ceil_div(row_bytes, 4096), 65535), 1),
            (min(ceil_div(row_bytes, 16 * 32) * 32, 256), 1, 1))


def select_torch213_query_last_row_launch(shape, attributes, hardware):
    """Complete INT32 query_start_loc[1:]-1 materialization on Torch2.13.

    BinaryOps.cpp sub_out invokes add_stub with negative alpha;ufunc/add.h
    supplies INT32 arithmetic. Loops.cuh captures the CPU scalar,leaving a
    no-cast unary contiguous iterator. CUDALoops.cuh uses8 values/thread and
    128 threads,including its unrolled vec1 branch. Actual slice alignment
    chooses vec1/2/4;the fresh output is16-byte aligned. No2.8 resources.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch2.13 query-last-row source requires H200 HWSpec")
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_query_start_layout"] != "contiguous_int32"):
        raise NotImplementedError("query-last-row requires exact Torch2.13 contiguous INT32 source")
    rows, = require_shape(shape, ("requests",))
    alignment = attributes["source_query_slice_address_alignment"]
    if type(alignment) is not int or alignment not in (4, 8, 16):
        raise ValueError("query_start_loc[1:] needs actual address alignment capped at16 bytes")
    if rows >= 2**31 or 4 * (rows - 1) >= 2**31 - 1:
        raise NotImplementedError("query-last-row split iterator requires every sublaunch")
    return LaunchConfig(f"pytorch_2_13_sm90_int32_sub_one_vec{alignment // 4}",
        (ceil_div(rows, 1024), 1, 1), (128, 1, 1),
        parameters={"dynamic_shared_bytes": 0, "source_values_per_thread": 8})


def select_torch213_int32_subtract_launch(shape, attributes, hardware):
    """Complete contiguous INT32 tensor-tensor subtraction,including inplace.

    Unlike the query-end scalar subtraction,this keeps two tensor inputs.
    The no-cast CUDALoops functor has12B I/O and8 values/thread. Pointer
    alignment affects vec1/2/4,but doesnot change128-thread/1024-value CTA
    geometry. The actual min alignment includes output and both inputs.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch2.13 INT32 subtraction requires H200 HWSpec")
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_subtract_layout"] != "contiguous_int32_same_shape"):
        raise NotImplementedError("INT32 subtraction requires exact Torch2.13 contiguous equal-shape source")
    elements, = require_shape(shape, ("element_count",))
    alignments = attributes["source_subtract_address_alignments"]
    if (not isinstance(alignments, (tuple, list)) or len(alignments) != 3
            or any(type(value) is not int or value not in (4, 8, 16) for value in alignments)):
        raise ValueError("INT32 subtraction requires actual output/input address alignments capped at16 bytes")
    if elements >= 2**31 or 4 * (elements - 1) >= 2**31 - 1:
        raise NotImplementedError("INT32 subtraction needs every split iterator launch")
    return LaunchConfig(f"pytorch_2_13_sm90_int32_subtract_vec{min(alignments) // 4}",
        (ceil_div(elements, 1024), 1, 1), (128, 1, 1),
        parameters={"dynamic_shared_bytes": 0, "source_values_per_thread": 8})


def select_torch213_float_stack0_recipe(shape, dtype, attributes, hardware):
    """Fresh dim0 stack of identical contiguous FP32 parameter tensors.

    TensorShape.cpp uses cat(dim0) then a view, including for one input.
    Shape.cu uses128-input batches for multiple inputs. A16B-aligned full
    slice selects fully vectorized input/output; otherwise alignedK handles
    aligned inputs, or the plain contiguous kernel handles misalignment.
    Both alignment flags accumulate across batches. One input follows the
    library's fresh-output copy path, not an alias or a made-up cat kernel.
    """
    from src.kernel.kernel_spec import PhysicalTransferConfig

    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch2.13 parameter stack requires H200 HWSpec")
    require_dtype(dtype, ("fp32",))
    inputs, = require_shape(shape, ("inputs",))
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_stack_layout"] != "contiguous_same_shape_dim0_fresh_output"):
        raise NotImplementedError("parameter stack requires exact Torch2.13 contiguous dim0 source")
    shapes = attributes["source_stack_input_shapes"]
    if (not isinstance(shapes, (tuple, list)) or len(shapes) != inputs
            or any(not isinstance(value, (tuple, list)) or not 1 <= len(value) <= 3
                   or any(type(size) is not int or size <= 0 for size in value) for value in shapes)
            or any(tuple(value) != tuple(shapes[0]) for value in shapes)):
        raise ValueError("parameter stack needs every actual matching positive input shape")
    elements = 1
    for size in shapes[0]:
        elements *= size
    if inputs * elements >= 2**31 - 1:
        raise NotImplementedError("parameter stack requires complete32-bit ELEMENT indexing")
    dtypes, alignments = attributes["source_stack_input_dtypes"], attributes["source_stack_input_address_alignments"]
    if (not isinstance(dtypes, (tuple, list)) or len(dtypes) != inputs or any(value != dtype for value in dtypes)
            or not isinstance(alignments, (tuple, list)) or len(alignments) != inputs
            or any(type(value) is not int or value not in (4, 8, 16) for value in alignments)):
        raise ValueError("parameter stack needs actual FP32 inputs and address alignments capped at16B")
    if inputs == 1:
        return (("stack_single_input_copy", PhysicalTransferConfig(
            algorithm="stream_copy", bytes=elements * 4, direction="d2d", link="dram")),)
    aligned, inout, stages = True, elements % 4 == 0, []
    for begin in range(0, inputs, 128):
        count = min(inputs - begin, 128)
        batch_aligned = all(value == 16 for value in alignments[begin:begin + count])
        aligned = aligned and batch_aligned
        inout = inout and batch_aligned
        if inout or aligned:
            grid = (min(ceil_div(elements, 512), int(hardware.num_compute_units) * 32), count, 1)
            block, name = (128, 1, 1), "vec16" if inout else "alignedK"
        else:
            grid, block, name = (2 * int(hardware.num_compute_units), count, 1), (512, 1, 1), "contig"
        launch = LaunchConfig(f"pytorch_2_13_sm90_stack_fp32_cat0_dims{1 if inout else len(shapes[0])}_{name}",
            grid, block, parameters={"dynamic_shared_bytes": 0, "source_input_begin": begin,
                "source_input_count": count, "source_elements_per_input": elements})
        stages.append((f"stack_inputs_{begin}_{begin + count}", launch))
    return tuple(stages)


def select_torch213_integer_stack_recipe(shape, dtype, attributes, hardware):
    """Complete stack of contiguous integer[B] vectors along dim1.

    TensorShape.cpp stack(dim==ndim) unsqueezes views and cats[B,1] along
    dim1. Shape.cu parallel_cat batches128 inputs. A one-element row slice
    is4/8B,so the fully vectorized16B IN+OUT path is never eligible. Aligned
    inputs use alignedK loads with strided output stores;otherwise the plain
    contiguous kernel uses2*SM CTAs and512 threads. isAligned accumulates
    across source batches,so a later aligned batch doesnot reset it.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch2.13 integer stack requires H200 HWSpec")
    require_dtype(dtype, ("int32", "int64"))
    inputs, = require_shape(shape, ("inputs",))
    rows = shape["rows"]
    if type(rows) is not int or rows < 0:
        raise ValueError("integer stack request count must be non-negative")
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_stack_layout"] != "contiguous_request_vectors_dim1_fresh_output"
            or inputs < 2):
        raise NotImplementedError("integer stack binds Torch2.13 multi-vector dim1 cat source")
    dtypes, lengths = attributes["source_stack_input_dtypes"], attributes["source_stack_input_lengths"]
    width = 4 if dtype == "int32" else 8
    if (not isinstance(dtypes, (tuple, list)) or len(dtypes) != inputs or any(value != dtype for value in dtypes)
            or not isinstance(lengths, (tuple, list)) or len(lengths) != inputs
            or any(type(value) is not int or value != rows for value in lengths)):
        raise ValueError("integer stack needs every actual same-dtype/same-length input vector")
    if rows == 0:
        return ()
    if rows * inputs >= 2**31 - 1:
        raise NotImplementedError("integer stack requires source32-bit ELEMENT indexing,not fallback copies")
    alignments = attributes["source_stack_input_address_alignments"]
    if (not isinstance(alignments, (tuple, list)) or len(alignments) != inputs
            or any(type(value) is not int or value not in ((4, 8, 16) if width == 4 else (8, 16)) for value in alignments)):
        raise ValueError("integer stack needs actual input address alignments capped at16 bytes")
    aligned, stages = True, []
    for begin in range(0, inputs, 128):
        count = min(inputs - begin, 128)
        aligned = aligned and all(value == 16 for value in alignments[begin:begin + count])
        if aligned:
            grid = (min(ceil_div(rows, 128 * (16 // width)), int(hardware.num_compute_units) * 32), count, 1)
            block, name = (128, 1, 1), "alignedK"
        else:
            grid, block, name = (2 * int(hardware.num_compute_units), count, 1), (512, 1, 1), "contig"
        launch = LaunchConfig(f"pytorch_2_13_sm90_stack_{dtype}_cat_{name}", grid, block,
            parameters={"dynamic_shared_bytes": 0, "source_input_begin": begin,
                        "source_input_count": count, "source_output_columns": inputs})
        stages.append((f"stack_inputs_{begin}_{begin + count}", launch))
    return tuple(stages)


def select_torch213_indexed_rows_launch(shape, dtype, attributes, hardware):
    """IndexSelect/IndexCopy of dense rows with actual row-strided storage.

    K3 mixed subgroup indices come from argsort and already have INT64
    dtype: neither operation inserts advanced-indexing's Long producer.
    IndexSelect's small path tests ELEMENT offsets including the original
    source row gaps; its larger gather_out tests restrided TensorIterator
    BYTE offsets. IndexCopy always uses IndexKernel<128,4>, not the16-byte
    gather fast path or ScatterGather<128,8>. No binary resources are bound.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch2.13 indexed rows require H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16", "fp32"))
    source_rows, columns = require_shape(shape, ("source_rows", "columns"))
    rows = shape["rows"]
    if type(rows) is not int or rows < 0:
        raise ValueError("indexed row count must be a non-negative integer")
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_row_layout"] != "dense_columns_strided_rows"
            or attributes["source_index_dtype"] != "int64"):
        raise NotImplementedError("indexed rows require actual Torch2.13 dense row views and INT64 indices")
    indices = attributes["source_row_indices"]
    if (not isinstance(indices, (tuple, list)) or len(indices) != rows
            or any(type(index) is not int or not 0 <= index < source_rows for index in indices)):
        raise ValueError("indexed rows need actual nonnegative row offsets")
    stride = attributes["source_row_stride_elements"]
    if type(stride) is not int or stride < columns:
        raise ValueError("indexed rows need nonoverlapping actual source/destination row strides")
    operation = attributes["source_operation"]
    if operation not in ("index_select_rows", "index_copy_rows"):
        raise NotImplementedError("indexed rows operation is not source-bound")
    if operation == "index_copy_rows" and len(set(indices)) != rows:
        raise ValueError("index_copy rows require unique destination indices")
    if rows == 0:
        return None
    width = 4 if dtype == "fp32" else 2
    elements = rows * columns
    source_offset = (source_rows - 1) * stride + columns - 1
    if width * (source_offset + 1) >= 2**63:
        raise NotImplementedError("indexed rows source address extent exceeds signed INT64")
    if operation == "index_select_rows":
        if rows <= 16 and max(source_rows * columns, elements, rows, source_offset) < 2**31 - 1:
            return LaunchConfig("pytorch_2_13_sm90_index_select_rows_small_int64",
                (min(ceil_div(columns, 128), int(hardware.num_compute_units) * 8), 1, 1),
                (min(columns, 128), 1, 1), parameters={"dynamic_shared_bytes": 0})
        if elements >= 2**31 or max(width * (elements - 1), 8 * (rows - 1)) >= 2**31 - 1:
            raise NotImplementedError("index_select gather_out needs its complete split iterator sequence")
        alignment = attributes["source_address_alignment"]
        if type(alignment) is not int or alignment not in (1, 2, 4, 8, 16):
            raise ValueError("index_select needs actual source address alignment capped at16 bytes")
        if rows > 1 and columns > 1 and alignment == 16 and width * columns % 16 == 0 and width * stride % 16 == 0:
            grid, block = _torch213_vectorized_gather_geometry(rows, width * columns)
            name = "pytorch_2_13_sm90_index_select_rows_gather_vec16_int64"
        else:
            grid, block = (ceil_div(elements, 1024), 1, 1), (128, 1, 1)
            name = "pytorch_2_13_sm90_index_select_rows_gather_scalar_int64"
    else:
        # Destination is restrided to zero on the indexed dimension;
        # actual selected-row addressing uses the separate INT64 index.
        if elements >= 2**31 or max(width * (elements - 1), 8 * (rows - 1)) >= 2**31 - 1:
            raise NotImplementedError("index_copy needs its complete split iterator sequence")
        grid, block = (ceil_div(elements, 512), 1, 1), (128, 1, 1)
        name = "pytorch_2_13_sm90_index_copy_rows_int64"
    return LaunchConfig(name, grid, block, parameters={"dynamic_shared_bytes": 0})


def select_vllm030_tp1_embedding_recipe(shape, dtype, attributes, hardware):
    """Complete unquantized TP1 VocabParallelEmbedding.forward onTorch2.13.

    The caller unconditionally calls long(),then F.embedding/index_select.
    Torch2.13 Indexing.cu uses smallIndex only for at most16 indices and
    three canUse32BitIndexMath checks in ELEMENT units. Otherwise it calls
    gather_out;its TensorIterator check is in BYTES. Singleton gather-out
    coalesces to1D andcannot use the16B gather fast path. No2.8 resource
    values orlarge-table small-index rejection are inherited here.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch2.13 embedding requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16", "fp32"))
    vocab, hidden = require_shape(shape, ("source_rows", "hidden"))
    rows = shape["rows"]
    if type(rows) is not int or rows < 0:
        raise ValueError("embedding rows must be a non-negative physical token count")
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_unquantized_tp1_embedding"
            or attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_input_layout"] != "contiguous_aligned_16"
            or attributes["source_added_vocab"] != "none"
            or type(attributes["source_tp_size"]) is not int or attributes["source_tp_size"] != 1):
        raise NotImplementedError("embedding binds exact0.30/Torch2.13 contiguous unquantized TP1 source")
    ids = attributes["source_token_ids"]
    index_dtype = attributes["source_input_ids_dtype"]
    if index_dtype not in ("int32", "int64"):
        raise NotImplementedError("embedding requires actual contiguous INT32/INT64 token IDs")
    if (not isinstance(ids, (tuple, list)) or len(ids) != rows
            or any(type(value) is not int or not 0 <= value < vocab for value in ids)):
        raise ValueError("embedding token IDs must be actual nonnegative source table offsets")
    if index_dtype == "int32" and any(value >= 2**31 for value in ids):
        raise ValueError("embedding token ID doesnot fit its actual INT32 dtype")
    if rows == 0:
        return ()
    width, elements = _DTYPE_BYTES[dtype], rows * hidden
    if width * vocab * hidden >= 2**63:
        raise NotImplementedError("embedding table exceeds signed source byte addressing")
    if index_dtype == "int32":
        prefix = (("token_ids_long", select_torch213_integer_cast_launch({"element_count": rows}, "int32",
            {**attributes, "source_output_dtype": "int64"}, hardware)),)
    else:
        prefix = ()
    # IndexingUtils.cpp's contiguous offset is numel-1,with the explicit
    # numel>=INT32_MAX rejection before that offset check. This differs from
    # TensorIterator's byte-offset proof used by gather_out below.
    small = rows <= 16 and max(elements, vocab * hidden, rows) < 2**31 - 1
    if small:
        grid = (min(ceil_div(hidden, 128), int(hardware.num_compute_units) * 8), 1, 1)
        block = (min(hidden, 128), 1, 1)
        algorithm = "pytorch_2_13_sm90_embedding_small_index_int64"
    else:
        if (elements >= 2**31 or max(width * (elements - 1), 8 * (rows - 1)) >= 2**31 - 1):
            raise NotImplementedError("embedding gather-out needs its complete split TensorIterator sequence")
        if rows > 1 and width * hidden % 16 == 0:
            row_bytes = width * hidden
            grid, block = _torch213_vectorized_gather_geometry(rows, row_bytes)
            algorithm = "pytorch_2_13_sm90_embedding_vectorized_gather_vec16_int64"
        else:
            # ScatterGatherKernel.cu uses num_threads()/thread_work_size()
            # from thread_constants.h (128/8),not IndexKernel.cu's128/4.
            grid, block = (ceil_div(elements, 1024), 1, 1), (128, 1, 1)
            algorithm = "pytorch_2_13_sm90_embedding_gather_out_scalar_int64"
    return (*prefix, ("embedding_index_select", LaunchConfig(algorithm, grid, block,
        parameters={"dynamic_shared_bytes": 0.0})))


def select_torch28_vector_launch(shape, dtype, variant, attributes, hardware):
    """Select a homogeneous no-cast iterator with an explicit layout proof."""
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch SM90 iterator requires H200 HWSpec")
    require_dtype(dtype, tuple(_DTYPE_BYTES))
    if variant != "elementwise" or attributes["tensor_iterator_source"] != "pytorch_2_8_sm90":
        raise NotImplementedError("Torch2.8 SM90 source program is not bound")
    operation = attributes["operation"]
    if operation in ("add", "mul"):
        if dtype not in ("bf16", "fp16", "fp32", "float32"):
            raise NotImplementedError("binary source binds only floating-point add/mul")
        reads, writes = 2, 1
    elif operation in ("copy", "unary"):
        if operation == "unary" and dtype not in ("bf16", "fp16", "fp32", "float32"):
            raise NotImplementedError("unary source binds only floating-point tensors")
        reads, writes = 1, 1
    elif operation == "fill":
        reads, writes = 0, 1
    else:
        raise NotImplementedError("Torch no-cast source operation is not bound")
    if (type(attributes["reads"]) is not int or type(attributes["writes"]) is not int
            or (attributes["reads"], attributes["writes"]) != (reads, writes)):
        raise ValueError("Torch source reads/writes disagree with its iterator arity")
    elements, = require_shape(shape, ("element_count",))
    width = _DTYPE_BYTES[dtype]
    layout = attributes["source_input_layout"]
    if layout == "contiguous_aligned_16":
        if operation == "copy":
            raise NotImplementedError("contiguous same-type ATen Copy requires MEMCPY or proven same-view no-op")
        maximum_offset = width * (elements - 1)
        # CUDALoops.cuh::elems_per_thread<io_size>, where io_size is the
        # sum of result and tensor-argument sizeof values. Fill<bool> alone
        # has io_size1 and16 values/lane; all other bound programs use8.
        values_per_thread = 16 if width * (reads + writes) == 1 else 8
        program = "vectorized_nocast"
    elif layout == "strided_nocast":
        maximum_offset = attributes["source_maximum_offset_bytes"]
        if type(maximum_offset) is not int or maximum_offset < 0:
            raise ValueError("strided iterator requires a non-negative integer maximum byte offset")
        if maximum_offset < width * (elements - 1):
            raise ValueError("strided iterator offset proof is smaller than its materialized output")
        values_per_thread = 2 if width >= 4 else 4
        program = "legacy_nocast"
    else:
        raise NotImplementedError("Torch iterator layout is not bound")
    # TensorIterator::can_use_32bit_indexing starts the byte offset at1.
    if elements > 2**31 - 1 or maximum_offset >= 2**31 - 1:
        raise NotImplementedError("split TensorIterator requires its full physical launch sequence")
    block_work = 128 * values_per_thread
    return LaunchConfig(f"pytorch_2_8_sm90_{program}",
        grid=(-(-elements // block_work), 1, 1), block=(128, 1, 1),
        execution_variant=variant)


def select_torch213_fill_launch(shape, dtype, attributes, hardware):
    """Official Torch v2.13 Fill/zero_ CUDA source on a contiguous SM90 view.

    Fill.cpp sends CUDA zero_ to fill_(0). FillKernel.cu uses FillFunctor
    with no tensor inputs, CUDALoops.cuh selects8 values/thread for all
    bound result types, and thread_constants.h fixes128 CUDA threads.
    Alignment chooses a vectorized or unrolled entry with the same grid.
    The BF16/FP16 vec8 entry is explicitly compiled for SM90/SM10x.
    This source rule is independently read from v2.13, not relabeled2.8
    compiler resources. Split/strided iterators remain outside this helper.
    """
    if hardware is not None and hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch2.13 SM90 Fill requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16", "fp32", "int32", "int64"))
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_input_layout"] not in ("contiguous_aligned_16", "contiguous")):
        raise NotImplementedError("Torch2.13 Fill requires its exact contiguous source")
    elements, = require_shape(shape, ("element_count",))
    width = {"bf16": 2, "fp16": 2, "fp32": 4, "int32": 4, "int64": 8}[dtype]
    if elements >= 2**31 or width * (elements - 1) >= 2**31 - 1:
        raise NotImplementedError("Torch2.13 split Fill requires its complete iterator launch sequence")
    return LaunchConfig(f"pytorch_2_13_sm90_fill_{dtype}",
        grid=(-(-elements // 1024), 1, 1), block=(128, 1, 1),
        parameters={"dynamic_shared_bytes": 0.0})


def select_torch213_integer_cast_launch(shape, dtype, attributes, hardware):
    """Copy.cu's contiguous INT32<->INT64 dynamic-cast source program.

    gpu_kernel uses the output-typed lambda with dynamic input conversion;
    CUDALoops.cuh dispatches128-thread/four-value launch_unrolled_kernel,not a homogeneous
    memcpy or no-cast vector kernel. The source has no compiled-resource
    estimate here. Values and narrowing validity remain caller semantics.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch2.13 integer cast requires H200 HWSpec")
    require_dtype(dtype, ("int32", "int64"))
    target = attributes["source_output_dtype"]
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_input_layout"] not in ("contiguous_aligned_16", "contiguous")
            or (dtype, target) not in (("int32", "int64"), ("int64", "int32"))):
        raise NotImplementedError("Torch2.13 cast requires an actual contiguous INT32/INT64 dtype change")
    elements, = require_shape(shape, ("element_count",))
    if elements >= 2**31 or 8 * (elements - 1) >= 2**31 - 1:
        raise NotImplementedError("Torch2.13 split cast requires every sub-iterator launch")
    return LaunchConfig(f"pytorch_2_13_sm90_{dtype}_to_{target}",
        (ceil_div(elements, 512), 1, 1), (128, 1, 1),
        parameters={"dynamic_shared_bytes": 0.0, "source_values_per_thread": 4.0})


def select_torch213_logits_fp32_cast_launch(shape, dtype, attributes, hardware):
    """BF16/FP16->FP32 Copy.cu's output-typed dynamic-cast CUDA kernel.

    The reverse FP32->16-bit specializations do not apply here. This is
    launch_unrolled_kernel with128 threads/four values,not a vectorized
    same-type copy. No2.8 compiled register/shared fact is transferred.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch2.13 logits cast requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16"))
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_input_layout"] not in ("contiguous_aligned_16", "contiguous")):
        raise NotImplementedError("Torch2.13 logits cast requires its actual contiguous16-bit source")
    elements, = require_shape(shape, ("element_count",))
    if elements >= 2**31 or 4 * (elements - 1) >= 2**31 - 1:
        raise NotImplementedError("Torch2.13 split logits cast requires every sub-iterator launch")
    return LaunchConfig(f"pytorch_2_13_sm90_{dtype}_to_fp32",
        (ceil_div(elements, 512), 1, 1), (128, 1, 1),
        parameters={"dynamic_shared_bytes": 0.0, "source_values_per_thread": 4.0})


def select_torch213_row_index_recipe(shape, dtype, attributes, hardware):
    """Advanced row indexing,including INT32->Long index normalization.

    IndexKernel.cu's gather fast path uses IndexKernelUtils.cu's16-byte
    vector copies. A singleton index dimension is removed by TensorIterator,
    so one selected row takes index_elementwise_kernel<128,4> instead. This
    remains materialized even for identity indices;it is never a view/copy_.
    Zero selected rows have no CUDA submission. Source rows and actual index
    values are independent caller inputs,not fabricated from output shape.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch2.13 row indexing requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16", "fp32", "int32"))
    source_rows, columns = require_shape(shape, ("source_rows", "columns"))
    rows = shape["rows"]
    if type(rows) is not int or rows < 0:
        raise ValueError("Torch2.13 row index count must be a non-negative integer")
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_input_layout"] not in ("contiguous", "contiguous_aligned_16")):
        raise NotImplementedError("Torch2.13 row indexing requires actual contiguous source/output rows")
    index_dtype, indices = attributes["source_index_dtype"], attributes["source_row_indices"]
    if index_dtype not in ("int32", "int64"):
        raise NotImplementedError("Torch2.13 advanced indexing binds INT32/INT64 source indices")
    if (not isinstance(indices, (tuple, list)) or len(indices) != rows
            or any(type(index) is not int or not -source_rows <= index < source_rows for index in indices)):
        raise ValueError("Torch2.13 row indices must be actual valid source row offsets")
    if index_dtype == "int32" and any(not -(2**31) <= index < 2**31 for index in indices):
        raise ValueError("Torch2.13 row index does not fit its actual INT32 dtype")
    if rows == 0:
        return ()
    width = 4 if dtype in ("fp32", "int32") else 2
    row_bytes, elements = width * columns, rows * columns
    if (max(width * (elements - 1), 8 * (rows - 1)) >= 2**31 - 1
            or elements >= 2**31 or width * source_rows * columns >= 2**63):
        raise NotImplementedError("Torch2.13 row indexing requires aligned complete rows andone32-bit iterator")
    prefix = ()
    if index_dtype == "int32":
        cast = select_torch213_integer_cast_launch({"element_count": rows}, "int32", {
            **attributes, "source_output_dtype": "int64"}, hardware)
        prefix = (("indices_long", cast),)
    if attributes["source_input_layout"] == "contiguous_aligned_16":
        aligned = True
    elif rows == 1 or row_bytes % 16:
        # These domains cannot satisfy the two-dimensional fast-gather
        # predicate, regardless of either pointer's alignment.
        aligned = False
    else:
        if "source_address_alignments" not in attributes:
            raise NotImplementedError("row indexing requires actual source/output pointer alignments to choose its gather program")
        alignments = attributes["source_address_alignments"]
        valid_alignments = (2, 4, 8, 16) if width == 2 else (4, 8, 16)
        if (not isinstance(alignments, (tuple, list)) or len(alignments) != 2
                or any(type(value) is not int or value not in valid_alignments for value in alignments)):
            raise ValueError("row indexing needs actual source/output alignments capped at16B")
        aligned = all(value == 16 for value in alignments)
    if rows == 1 or row_bytes % 16 or not aligned:
        launch = LaunchConfig("pytorch_2_13_sm90_row_index_singleton" if rows == 1 else "pytorch_2_13_sm90_row_index_scalar",
            (ceil_div(elements, 512), 1, 1), (128, 1, 1))
    else:
        grid, block = _torch213_vectorized_gather_geometry(rows, row_bytes)
        launch = LaunchConfig("pytorch_2_13_sm90_vectorized_row_index_vec16",
            grid, block)
    return (*prefix, ("row_index", launch))


def select_torch213_integer_vector_index_recipe(shape, dtype, attributes, hardware):
    """Complete advanced indexing of a contiguous1D INT32/INT64 vector.

    This is vector[indices],not index_select/gather_out. TensorIterator is
    one-dimensional,so IndexKernelUtils's two-dimensional16B gather path is
    ineligible even when pointers align. INT32 indices widen on EACH call;
    the original index tensor doesnot become Long for subsequent calls.
    Negative indices use this source vector's own actual extent.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch2.13 integer vector indexing requires H200 HWSpec")
    require_dtype(dtype, ("int32", "int64"))
    source_rows, = require_shape(shape, ("source_rows",))
    rows = shape["rows"]
    if type(rows) is not int or rows < 0:
        raise ValueError("integer vector index count must be non-negative")
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_input_layout"] not in ("contiguous", "contiguous_aligned_16")):
        raise NotImplementedError("integer vector indexing requires exact Torch2.13 contiguous source")
    index_dtype, indices = attributes["source_index_dtype"], attributes["source_row_indices"]
    if index_dtype not in ("int32", "int64"):
        raise NotImplementedError("integer vector indexing binds actual INT32/INT64 indices")
    if (not isinstance(indices, (tuple, list)) or len(indices) != rows
            or any(type(index) is not int or not -source_rows <= index < source_rows for index in indices)
            or (index_dtype == "int32" and any(not -2**31 <= index < 2**31 for index in indices))):
        raise ValueError("integer vector indices must fit actual source extent and index dtype")
    if rows == 0:
        return ()
    width = 8 if dtype == "int64" else 4
    if (rows >= 2**31 or max(width * (rows - 1), 8 * (rows - 1)) >= 2**31 - 1
            or source_rows * width >= 2**63):
        raise NotImplementedError("integer vector indexing needs complete source addressing/subiterator sequence")
    prefix = ()
    if index_dtype == "int32":
        prefix = (("indices_long", select_torch213_integer_cast_launch({"element_count": rows}, "int32",
            {**attributes, "source_output_dtype": "int64"}, hardware)),)
    return (*prefix, ("vector_index", LaunchConfig(f"pytorch_2_13_sm90_{dtype}_vector_index",
        (ceil_div(rows, 512), 1, 1), (128, 1, 1), parameters={"dynamic_shared_bytes": 0})))


def select_torch213_large_vocab_logsoftmax_launch(shape, dtype, attributes, hardware):
    """SoftMax.cu's FP32 inner_size1 log-softmax for a large vocabulary.

    The input is already contiguous FP32. If its row plus reduction scratch
    exceeds even the H200 opt-in shared-memory limit,all legal values of the
    smaller runtime sharedMemPerBlock select the ordinary global-memory
    source kernel. No default48KiB,occupancy query or cached logits value is
    assumed. Register/persistent/shared-fit domains remain separate programs.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch2.13 log-softmax requires H200 HWSpec")
    require_dtype(dtype, ("fp32",))
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_input_layout"] != "contiguous_aligned_16"):
        raise NotImplementedError("Torch2.13 log-softmax requires actual contiguousFP32 source")
    rows, vocab = require_shape(shape, ("rows", "vocab"))
    if (vocab <= 2048 or 4 * vocab + 128 <= hardware.max_smem_per_block_bytes
            or 4 * rows * vocab >= 2**31):
        raise NotImplementedError("log-softmax source requires its unsplit provably large-vocabulary global-memory branch")
    # SoftMaxForward_getBlockSize clamps to1024;warp-reduction scratch
    # is block.x/warp_size*sizeof(float),and CUDA use_fast_softmax=False.
    return LaunchConfig("pytorch_2_13_sm90_logsoftmax_float_global_ilp4",
        (rows, 1, 1), (1024, 1, 1), parameters={"dynamic_shared_bytes": 128.0})


def select_torch213_strided_copy_launch(shape, dtype, attributes, hardware):
    """A homogeneous Copy.cu iterator proven non-contiguous after coalescing.

    CUDALoops.cuh::gpu_kernel_impl_nocast uses legacy128-thread kernels
    with unroll4 for BF16/FP16 and unroll2 for32-bit values. Contiguous
    slices require a stream transfer (or proven same-view no-op) instead.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch2.13 strided Copy requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16", "fp32", "int32"))
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_input_layout"] != "strided_nocast"
            or attributes["source_copy_flags"] != "same_nonnegated_real_dtype"):
        raise NotImplementedError("Torch2.13 strided Copy requires an explicit non-contiguous no-cast iterator")
    elements, = require_shape(shape, ("element_count",))
    width = 2 if dtype in ("bf16", "fp16") else 4
    offset = attributes["source_maximum_offset_bytes"]
    if type(offset) is not int or offset < width * (elements - 1):
        raise ValueError("strided Copy requires its actual maximum byte-offset extent")
    if elements >= 2**31 or offset >= 2**31 - 1:
        raise NotImplementedError("Torch2.13 split Copy requires all iterator submissions")
    unroll = 4 if width == 2 else 2
    return LaunchConfig(f"pytorch_2_13_sm90_strided_copy_{dtype}",
        grid=(-(-elements // (128 * unroll)), 1, 1), block=(128, 1, 1),
        parameters={"dynamic_shared_bytes": 0.0})


def select_torch213_nn_embedding_launch(shape, dtype, attributes, hardware):
    """Default nn.Embedding's direct index_select, preserving INT32 indices.

    Embedding.cpp::embedding_symint accepts kInt/kLong and directly calls
    weight.index_select. Unlike VocabParallelEmbedding.forward it does not
    call long(). The smallIndex source path depends on tensor extents, not
    actual token values or pointer alignment. Larger batches use gather_out,
    whose vector branch requires actual source/output pointer alignment.
    The source-input alignment is the embedding weight table's pointer;
    it does not refer to the separate token-ID tensor.
    The declared index bounds
    come from an existing token producer, not a fabricated token sequence.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch2.13 nn.Embedding requires H200 HWSpec")
    require_dtype(dtype, ("bf16",))
    rows, vocab, hidden = require_shape(shape, ("rows", "source_rows", "hidden"))
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_embedding_program"] != "torch_nn_embedding_default"
            or attributes["source_input_layout"] != "contiguous"
            or attributes["source_weight_layout"] != "contiguous_rows"):
        raise NotImplementedError("nn.Embedding requires its exact Torch2.13 default contiguous source")
    index_dtype = attributes["source_index_dtype"]
    if index_dtype not in ("int32", "int64"):
        raise NotImplementedError("nn.Embedding requires the actual INT32/INT64 index dtype")
    bounds = attributes["source_index_bounds"]
    if (not isinstance(bounds, (tuple, list)) or len(bounds) != 2
            or any(type(value) is not int for value in bounds)
            or not 0 <= bounds[0] <= bounds[1] < vocab
            or index_dtype == "int32" and bounds[1] >= 2**31):
        raise ValueError("nn.Embedding index bounds must fit the actual source vocabulary and index dtype")
    elements = rows * hidden
    if max(elements, vocab * hidden, rows) >= 2**31 - 1:
        raise NotImplementedError("nn.Embedding requires its independently bounded INT64 source-address program outside32-bit element indexing")
    if rows <= 16:
        return LaunchConfig(f"pytorch_2_13_sm90_nn_embedding_small_index_{index_dtype}",
            (min(ceil_div(hidden, 128), int(hardware.num_compute_units) * 8), 1, 1),
            (min(hidden, 128), 1, 1), parameters={"dynamic_shared_bytes": 0.0})
    # Indexing.cu::index_select_out_cuda_impl expands the original index
    # along the row width and calls gather_out; no Long cast is introduced.
    index_width = 4 if index_dtype == "int32" else 8
    if max(2 * (elements - 1), index_width * (rows - 1)) >= 2**31 - 1:
        raise NotImplementedError("nn.Embedding gather_out requires its complete split iterator sequence")
    row_bytes = 2 * hidden
    vector = False
    if row_bytes % 16 == 0 and hidden > 1:
        fields = ("source_input_address_alignment_bytes", "source_output_address_alignment_bytes")
        if not set(fields).issubset(attributes):
            raise NotImplementedError("nn.Embedding gather_out requires actual source/output pointer alignments")
        alignments = tuple(attributes[name] for name in fields)
        if any(type(value) is not int or value not in (2, 4, 8, 16) for value in alignments):
            raise ValueError("nn.Embedding needs actual BF16 source/output alignments capped at16B")
        vector = all(value == 16 for value in alignments)
    if vector:
        grid, block = _torch213_vectorized_gather_geometry(rows, row_bytes)
        program = "gather_vec16"
    else:
        grid, block = (ceil_div(elements, 1024), 1, 1), (128, 1, 1)
        program = "gather_scalar"
    return LaunchConfig(f"pytorch_2_13_sm90_nn_embedding_{program}_{index_dtype}",
        grid, block, parameters={"dynamic_shared_bytes": 0.0})


def select_torch213_bf16_binary_add_launch(shape, attributes, hardware):
    """Two BF16 inputs and one fresh BF16 output, with actual row strides.

    Strided CUDALoops iterators use128 threads/unroll4. Contiguous inputs
    require actual three-pointer alignment: vec1 uses the512-value unrolled
    entry, while vec2/4/8 use1024 values per CTA. No allocator assumption or
    Torch2.8 resource metadata is used.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch2.13 BF16 add requires H200 HWSpec")
    rows, columns = require_shape(shape, ("rows", "columns"))
    if attributes["tensor_iterator_source"] != "pytorch_2_13_sm90":
        raise NotImplementedError("BF16 add requires exact Torch2.13 source")
    strides = attributes["source_input_strides_bytes"]
    if (not isinstance(strides, (tuple, list)) or len(strides) != 2
            or any(not isinstance(value, (tuple, list)) or len(value) != 2
                   or any(type(item) is not int or item <= 0 for item in value) for value in strides)
            or any(value[1] != 2 or value[0] < 2 * columns for value in strides)):
        raise ValueError("BF16 add needs actual nonoverlapping unit-final-stride input rows")
    elements = rows * columns
    maximum = max(2 * (elements - 1), *(value[0] * (rows - 1) + 2 * (columns - 1) for value in strides))
    if elements >= 2**31 or maximum >= 2**31 - 1:
        raise NotImplementedError("BF16 add requires its complete split iterator sequence")
    contiguous = rows == 1 or all(value[0] == 2 * columns for value in strides)
    if not contiguous:
        name, block_work = "strided_nocast", 512
    else:
        if "source_address_alignments" not in attributes:
            raise NotImplementedError("contiguous BF16 add requires actual output/input pointer alignments")
        alignments = attributes["source_address_alignments"]
        if (not isinstance(alignments, (tuple, list)) or len(alignments) != 3
                or any(type(value) is not int or value not in (2, 4, 8, 16) for value in alignments)):
            raise ValueError("BF16 add needs actual three-pointer alignments capped at16B")
        vector = min(alignments) // 2
        name, block_work = f"contiguous_vec{vector}", 512 if vector == 1 else 1024
    return LaunchConfig(f"pytorch_2_13_sm90_bf16_add_{name}",
        (ceil_div(elements, block_work), 1, 1), (128, 1, 1), parameters={"dynamic_shared_bytes": 0.0})
