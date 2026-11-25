"""SM90 DSpark's GPU-v2 DFlash input preparation, vLLM0.30 source.

v1/worker/gpu/spec_decode/dspark/speculator.py inherits DFlashSpeculator.
Its prepare_dflash_inputs wrapper in dflash/speculator.py fixes one fused
Triton submission. This prepared primitive is separate from the backbone,
context-KV insertion and sequential Markov sampling. Defaults are those of
Triton3.7.1 CUDAOptions; no compiled resources or timing are inferred.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import require_dtype, require_shape, ceil_div


def select_dspark_prepare_query_recipe(shape, dtype, attributes, hardware):
    """Select the complete query/context metadata preparation submission.

    Grid.y covers max(per-request scheduled target tokens)+query width,
    including rejected context, rather than total batch tokens, valid tokens
    or the allocation capacity. Token/page values cannot change this launch.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("DSpark query preparation requires H200 HWSpec")
    require_dtype(dtype, ("int32",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_gpu_v2_dspark_dflash_inputs"
            or attributes["source_compiler"] != "triton_3_7_1_cuda_sm90"):
        raise NotImplementedError("DSpark requires its exact GPU-v2 DFlash source and compiler")
    rows, depth, query, max_rows, max_tokens = require_shape(shape, (
        "rows", "draft_depth", "query_tokens_per_request", "max_requests", "max_tokens"))
    anchor = attributes["sample_from_anchor"]
    if type(anchor) is not bool:
        raise TypeError("DSpark anchor sampling must be an explicit boolean")
    if query != depth + (0 if anchor else 1):
        raise ValueError("DSpark query width disagrees with its anchor/fill-in source branch")
    scheduled = attributes["num_scheduled_tokens"]
    if (not isinstance(scheduled, (tuple, list)) or len(scheduled) != rows
            or any(type(value) is not int or value <= 0 for value in scheduled)):
        raise ValueError("DSpark requires positive scheduled target tokens for every actual request")
    block_size, max_len, columns, cp_size, interleave = require_shape(attributes, (
        "block_size", "max_model_len", "block_table_columns", "context_parallel_size", "cp_interleave"))
    cp_rank = attributes["cp_rank"]
    if type(cp_rank) is not int or not 0 <= cp_rank < cp_size:
        raise ValueError("DSpark CP rank must belong to its explicit context-parallel group")
    if (rows > max_rows or rows * query > max_tokens or sum(scheduled) > max_tokens
            or max(max_rows, max_tokens, max_len, block_size, columns, cp_size, interleave, depth) >= 2**31
            or max(scheduled) + query >= 2**31):
        raise ValueError("DSpark query preparation exceeds its source buffer or signed indexing extents")
    span = max(scheduled) + query
    tile = min(256, 1 << (span - 1).bit_length())
    launch = LaunchConfig("vllm_0_30_sm90_dspark_prepare_dflash_inputs",
        (rows, ceil_div(span, tile), 1), (128, 1, 1), parameters={
            "source_BLOCK_SIZE": tile, "source_num_stages": 3,
            "source_SAMPLE_FROM_ANCHOR": int(anchor), "source_CP_SIZE": cp_size,
            "source_CP_INTERLEAVE": interleave})
    return (("prepare_query_context_metadata", launch),)


def select_dspark_noncausal_indices_recipe(shape, dtype, attributes, hardware):
    """Original per-query SWA index producer for a complete draft block."""
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("DSpark noncausal metadata requires H200")
    require_dtype(dtype, ("int32",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_compiler"] != "triton_3_7_1_cuda_sm90"
            or attributes["source_component"] != "vllm_0_30_symbolic_dspark_eager"
            or attributes["operation"] != "noncausal_indices"):
        raise NotImplementedError("DSpark metadata requires exact GPU-v2 source identity")
    rows, width = require_shape(shape, ("rows", "width"))
    if (attributes["query_block_size"] != 5 or attributes["window_size"] != 128
            or width != 192 or rows != attributes["request_count"] * 5
            or rows >= 2**31):
        raise ValueError("DSpark metadata requires complete block5 query rows and192 indices")
    launch = LaunchConfig("vllm_0_30_sm90_dspark_noncausal_swa_indices",
        (rows, 1, 1), (128, 1, 1), parameters={"source_BLOCK_SIZE": 1024, "source_num_stages": 3})
    return (("noncausal_swa_indices_and_lengths", launch),)


def select_dspark_primitive_recipe(shape, dtype, attributes, hardware):
    """Source repeat/copy and explicitly prepared direct BF16 argmax stages.

    DSpark repeats its embedding through TensorShape.cpp::repeat's unfolded
    expanded view followed by copy_. The non-contiguous broadcast iterator
    uses the existing Torch2.13 no-cast Copy rule. Column assignment is a
    same-dtype copy_: batch1 is contiguous and uses a stream transfer;
    multiple rows keep the actual five-token destination stride.
    """
    from src.kernel.kernel_spec import PhysicalTransferConfig
    from src.kernel.launch_config.nvidia_h200.torch_vector_launch import (
        select_torch213_strided_copy_launch, select_torch213_nn_embedding_launch,
        select_torch213_integer_vector_index_recipe, select_torch213_bf16_binary_add_launch,
        select_torch213_row_index_recipe)
    from src.kernel.launch_config.nvidia_h200.sampler_launch import select_torch213_argmax_recipe

    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("DSpark primitive source requires H200 HWSpec")
    generic = attributes["source_component"] in (
        "vllm_0_30_symbolic_dspark_bias_add", "vllm_0_30_symbolic_dspark_sample_gather")
    runtime = attributes["source_geometry_tensor_iterator_source"] if generic else attributes["tensor_iterator_source"]
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] not in (
                "vllm_0_30_symbolic_dspark_eager", "vllm_0_30_symbolic_dspark_bias_add",
                "vllm_0_30_symbolic_dspark_sample_gather")
            or attributes["source_geometry_program"] != "vllm_0_30_torch_2_13_cuda_source"
            or runtime != "pytorch_2_13_sm90"):
        raise NotImplementedError("DSpark primitive requires its exact0.30/Torch2.13 source")
    operation = (attributes["source_geometry_operation"]
        if attributes["source_component"] == "vllm_0_30_symbolic_dspark_sample_gather" else attributes["operation"])
    if operation == "sample_gather":
        require_dtype(dtype, ("bf16",))
        rows, columns = require_shape(shape, ("rows", "hidden"))
        source_rows, = require_shape(attributes, ("source_geometry_source_rows",))
        if (attributes["source_component"] != "vllm_0_30_symbolic_dspark_sample_gather"
                or tuple(attributes["source_geometry_input_contiguous"]) != (True, True)
                or tuple(map(tuple, attributes["source_geometry_input_strides_bytes"])) != ((2 * columns, 2), (8,))
                or attributes["source_geometry_index_producer"] != "dspark_prepare_dflash_sample_from_anchor"):
            raise ValueError("DSpark sample gather requires its actual BF16 hidden rows and prepared INT64 indices")
        required = ("source_input_address_alignment_bytes", "source_output_address_alignment_bytes")
        if not set(required).issubset(attributes):
            raise NotImplementedError("DSpark sample gather needs actual input/output pointer alignments")
        return select_torch213_row_index_recipe({"rows": rows, "columns": columns, "source_rows": source_rows}, dtype, {
            "tensor_iterator_source": runtime, "source_input_layout": "contiguous", "source_index_dtype": "int64",
            "source_row_indices": attributes["source_geometry_row_indices"],
            "source_address_alignments": tuple(attributes[name] for name in required)}, hardware)
    if attributes["source_component"] == "vllm_0_30_symbolic_dspark_bias_add":
        require_dtype(dtype, ("bf16",))
        if operation != "add" or attributes["reads"] != 2 or attributes["writes"] != 1:
            raise ValueError("DSpark Markov bias requires the actual binary add arity")
        rows, columns = require_shape(attributes, ("source_geometry_rows", "source_geometry_columns"))
        if shape["element_count"] != rows * columns:
            raise ValueError("DSpark bias-add geometry must cover every actual logit")
        return (("markov_bias_add", select_torch213_bf16_binary_add_launch(
            {"rows": rows, "columns": columns}, {**attributes,
                "tensor_iterator_source": runtime,
                "source_input_strides_bytes": attributes["source_geometry_input_strides_bytes"]}, hardware)),)
    rows, = require_shape(shape, ("rows",))
    if operation == "anchor_gather":
        require_dtype(dtype, ("int32",))
        source_rows, = require_shape(shape, ("source_rows",))
        if (tuple(attributes["source_geometry_input_contiguous"]) != (True, True)
                or tuple(attributes["source_input_dtypes"]) != ("int32", "int64")
                or tuple(map(tuple, attributes["source_input_shapes"])) != ((source_rows,), (rows,))):
            raise ValueError("DSpark anchor gather requires its actual INT32 input IDs and persistent INT64 indices")
        return select_torch213_integer_vector_index_recipe({"rows": rows, "source_rows": source_rows}, "int32", {
            "tensor_iterator_source": attributes["tensor_iterator_source"], "source_input_layout": "contiguous",
            "source_index_dtype": "int64", "source_row_indices": attributes["source_geometry_row_indices"]}, hardware)
    if operation == "markov_embed":
        require_dtype(dtype, ("bf16",))
        vocab, width = require_shape(attributes, ("vocab", "markov_rank"))
        if (tuple(attributes["source_geometry_input_contiguous"]) != (True,)
                or tuple(attributes["source_input_dtypes"]) != (attributes["index_dtype"],)
                or tuple(map(tuple, attributes["source_input_shapes"])) != ((rows,),)
                or tuple(map(tuple, attributes["source_output_shapes"])) != ((rows, width),)
                or tuple(map(tuple, attributes["source_geometry_parameter_shapes"])) != ((vocab, width),)
                or tuple(map(tuple, attributes["source_geometry_parameter_strides_bytes"])) != ((2 * width, 2),)
                or tuple(attributes["source_geometry_parameter_dtypes"]) != ("bf16",)):
            raise ValueError("DSpark Markov embedding must retain the actual replicated BF16 table and index dtype")
        return (("markov_embedding_index_select", select_torch213_nn_embedding_launch(
            {"rows": rows, "source_rows": vocab, "hidden": width}, dtype, {
                "tensor_iterator_source": attributes["tensor_iterator_source"],
                "source_embedding_program": "torch_nn_embedding_default", "source_input_layout": "contiguous",
                "source_weight_layout": "contiguous_rows", "source_index_dtype": attributes["index_dtype"],
                "source_index_bounds": attributes["source_geometry_index_bounds"],
                **{name: attributes[name] for name in (
                    "source_input_address_alignment_bytes", "source_output_address_alignment_bytes") if name in attributes}}, hardware)),)
    if operation == "argmax":
        require_dtype(dtype, ("int64",))
        vocab, = require_shape(attributes, ("vocab",))
        if (tuple(attributes["source_input_dtypes"]) != ("bf16",)
                or tuple(map(tuple, attributes["source_input_shapes"])) != ((rows, vocab),)
                or tuple(map(tuple, attributes["source_output_shapes"])) != ((rows,),)):
            raise ValueError("DSpark greedy argmax requires actual complete BF16 logits and INT64 output")
        # Reduce.cuh handles the unaligned head in its device body. Actual
        # contiguous rows suffice for the source launch geometry; runtime
        # device properties remain explicit inputs checked against HWSpec.
        return select_torch213_argmax_recipe({"rows": rows, "vocab": vocab}, "bf16", attributes, hardware)
    if tuple(attributes["source_geometry_input_contiguous"]) != (True,):
        raise NotImplementedError("DSpark primitive requires its actual contiguous source value tensor")
    if operation == "repeat_hc":
        require_dtype(dtype, ("bf16",))
        hidden, copies = require_shape(attributes, ("hidden", "hc_mult"))
        elements, = require_shape(shape, ("element_count",))
        if (copies != 4 or elements != rows * copies * hidden
                or tuple(map(tuple, attributes["source_input_shapes"])) != ((rows, hidden),)
                or tuple(attributes["source_input_dtypes"]) != ("bf16",)
                or tuple(map(tuple, attributes["source_output_shapes"])) != ((rows, copies, hidden),)
                or tuple(map(tuple, attributes["source_geometry_input_strides_bytes"])) != ((2 * hidden, 2),)
                or attributes["source_geometry_output_contiguous"] is not True):
            raise ValueError("DSpark repeat requires the actual dense BF16 embedding and fresh four-copy output")
        launch = select_torch213_strided_copy_launch({"element_count": elements}, dtype, {
            "tensor_iterator_source": attributes["tensor_iterator_source"],
            "source_input_layout": "strided_nocast",
            "source_copy_flags": "same_nonnegated_real_dtype",
            "source_maximum_offset_bytes": 2 * (elements - 1)}, hardware)
        return (("repeat_hc_materialization", launch),)
    if operation == "draft_store":
        require_dtype(dtype, ("int64",))
        depth, = require_shape(attributes, ("draft_depth",))
        step = attributes["draft_step"]
        if (depth != 5 or type(step) is not int or not 0 <= step < depth
                or tuple(map(tuple, attributes["source_input_shapes"])) != ((rows,),)
                or tuple(attributes["source_input_dtypes"]) != ("int64",)
                or tuple(map(tuple, attributes["source_output_shapes"])) != ((rows,),)
                or tuple(map(tuple, attributes["source_geometry_input_strides_bytes"])) != ((8,),)
                or tuple(attributes["source_geometry_output_strides_bytes"]) != (8 * depth,)
                or attributes["source_geometry_output_offset_bytes"] != 8 * step
                or attributes["source_geometry_output_contiguous"] is not (rows == 1)):
            raise ValueError("DSpark draft store requires its actual INT64 five-column destination view")
        if rows == 1:
            return (("draft_column_copy", PhysicalTransferConfig(
                algorithm="stream_copy", bytes=8, direction="d2d", link="dram")),)
        # CUDALoops.cuh::gpu_kernel_impl_nocast uses sizeof(arg0_t)>=4
        # unroll2 for this8-byte integer Copy,without vector alignment.
        if rows >= 2**31 or (rows - 1) * 8 * depth >= 2**31 - 1:
            raise NotImplementedError("DSpark draft column Copy requires every split iterator launch")
        return (("draft_column_copy", LaunchConfig("pytorch_2_13_sm90_strided_copy_int64",
            (ceil_div(rows, 256), 1, 1), (128, 1, 1), parameters={"dynamic_shared_bytes": 0.0})),)
    raise NotImplementedError("DSpark primitive source operation is not bound")
