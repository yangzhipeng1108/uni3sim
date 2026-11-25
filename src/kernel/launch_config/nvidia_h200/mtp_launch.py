"""vLLM0.30 text MTP's complete padded-feedback/step helper source rules.

llm_base_proposer.py and spec_decode/utils.py own these operators. Their
warmup dispatchers return one program/request and no launch overrides;
Triton3.7.1 CUDAOptions defaults are four warps/three stages on SM90. No
compiler register/shared resources, provider identity or timing is inferred.
The draft model remains a separate caller program. Torch2.13's ordinary
first-pass shift/scatter/copies are selected below without changing the
padded helper's Triton/compiler identity or inventing aggregate admission.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import require_dtype, require_shape


def _source(dtype, attributes, hardware, expected_dtype):
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("MTP helper source requires H200 HWSpec")
    require_dtype(dtype, (expected_dtype,))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_base_proposer_padded_helpers"
            or attributes["source_compiler"] != "triton_3_7_1_cuda_sm90"
            or attributes["source_metadata_layout"] != "contiguous_int32_request_vectors"):
        raise NotImplementedError("MTP helpers require exact0.30 padded proposer/Triton3.7.1 source")


def _vector(attributes, key, rows, *, boolean=False):
    values = attributes[key]
    if (not isinstance(values, (tuple, list)) or len(values) != rows
            or any(type(value) is not (bool if boolean else int) for value in values)
            or (not boolean and any(not -2**31 <= value < 2**31 for value in values))):
        raise ValueError("MTP helper needs actual contiguous request vector: " + key)
    return values


def _metadata_launch(name, rows, **parameters):
    return LaunchConfig("vllm_0_30_sm90_mtp_" + name, (rows, 1, 1), (128, 1, 1),
        parameters={"source_num_stages": 3, **parameters})


def select_mtp_eager_stage_recipe(shape, dtype, attributes, hardware):
    """Bind existing typed MTP graph primitives without inventing token values.

    The three metadata kernels have value-independent geometry. The traced source
    already validates typed input/output extents and its explicit scenario;
    sampled IDs, positions or page contents need not be known to select a
    launch. This is the kernel primitive boundary, so backup H2D belongs to
    its own upstream producer rather than being submitted a second time.
    The eager step has input_batch_size==rows; padded graph steps retain
    their separate explicit prepared-helper contract.
    An explicitly declared Torch2.13 source also binds contiguous Copy,
    Fill and dynamic casts, whose geometry does not depend on pointer alignment.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("MTP eager stage requires H200 HWSpec")
    if (attributes["source_component"] != "vllm_0_30_symbolic_mtp_eager"
            or attributes["model_execution_version"] != "vllm_0_30_0"):
        raise NotImplementedError("MTP graph stages require their exact0.30 eager source")
    rows, width = require_shape(shape, ("rows", "width"))
    if max(rows, width) >= 2**31 or rows * width >= 2**63:
        raise ValueError("MTP source stage extents exceed signed source indexing")
    operation = attributes["operation"]
    if operation == "rejection":
        from src.kernel.launch_config.nvidia_h200.rejection_launch import _greedy_rejection_kernel_launch

        require_dtype(dtype, ("int32",))
        verified, accepted = attributes["verified_tokens"], attributes["accepted_tokens"]
        if (not isinstance(verified, (tuple, list)) or not isinstance(accepted, (tuple, list))
                or len(verified) != rows or len(accepted) != rows
                or any(type(v) is not int or type(a) is not int or not 0 <= a <= v < width
                    for v, a in zip(verified, accepted))
                or "source_rows" not in shape or type(shape["source_rows"]) is not int
                or shape["source_rows"] != sum(verified) or sum(verified) >= 2**31):
            raise ValueError("MTP greedy rejection requires its complete draft/acceptance row extents")
        # The graph already contains bonus sampling, target argmax and the
        # output placeholder fill. Bind this one kernel without replaying
        # the complete wrapper or interpreting assumed acceptance as values.
        return ((operation, _greedy_rejection_kernel_launch(rows, width - 1, hardware)),)
    if operation == "argmax":
        from src.kernel.launch_config.nvidia_h200.sampler_launch import select_torch213_argmax_recipe

        require_dtype(dtype, ("bf16", "fp32"))
        if (attributes["source_geometry_program"] != "vllm_0_30_torch_2_13_cuda_source"
                or tuple(attributes["source_geometry_input_contiguous"]) != (True,)
                or attributes["source_geometry_output_contiguous"] is not True):
            raise NotImplementedError("MTP argmax requires its declared contiguous Torch2.13 source")
        # Keep original BF16/FP32 input and INT64 output. The existing
        # recipe retains conditional semaphore zeroing; no cast or source
        # stage is dropped, and logical pricing remains independent.
        return select_torch213_argmax_recipe({"rows": rows, "vocab": width}, dtype,
            {"tensor_iterator_source": "pytorch_2_13_sm90", "source_input_layout": "contiguous",
             "source_num_compute_units": int(hardware.num_compute_units),
             "source_max_threads_per_unit": int(hardware.max_threads_per_unit)}, hardware)
    if operation in ("copy", "cast_ids", "cast_fp32", "fill"):
        from src.kernel.kernel_spec import PhysicalTransferConfig
        from src.kernel.launch_config.nvidia_h200.torch_vector_launch import (
            select_torch213_integer_cast_launch, select_torch213_logits_fp32_cast_launch,
            select_torch213_fill_launch)

        if attributes["source_geometry_program"] != "vllm_0_30_torch_2_13_cuda_source":
            raise NotImplementedError("MTP materialization requires its declared Torch2.13 source")
        contiguous = attributes["source_geometry_input_contiguous"]
        count = 0 if operation == "fill" else 1
        if (not isinstance(contiguous, (tuple, list)) or len(contiguous) != count
                or any(value is not True for value in contiguous)
                or attributes["source_geometry_output_contiguous"] is not True):
            raise NotImplementedError("MTP materialization needs its actual contiguous input/output layout")
        elements = rows * width
        source = {"tensor_iterator_source": "pytorch_2_13_sm90", "source_input_layout": "contiguous"}
        if operation == "copy":
            require_dtype(dtype, ("bool", "int32", "int64", "bf16", "fp32"))
            item_bytes = {"bool": 1, "int32": 4, "int64": 8, "bf16": 2, "fp32": 4}[dtype]
            if elements * item_bytes >= 2**63:
                raise ValueError("MTP contiguous copy exceeds signed byte extents")
            # Copy.cu's equal-dtype contiguous CUDA path submits memcpyAsync.
            # Do not manufacture a compute CTA for a physical transfer.
            return (("copy", PhysicalTransferConfig(algorithm="stream_copy", bytes=elements * item_bytes,
                direction="d2d", link="dram")),)
        if operation == "fill":
            launch = select_torch213_fill_launch({"element_count": elements}, dtype, source, hardware)
        elif operation == "cast_ids":
            require_dtype(dtype, ("int64",))
            if width != 1:
                raise ValueError("MTP draft IDs require one scalar ID per request")
            launch = select_torch213_integer_cast_launch({"element_count": elements}, dtype,
                {**source, "source_output_dtype": "int32"}, hardware)
        else:
            launch = select_torch213_logits_fp32_cast_launch({"element_count": elements}, dtype, source, hardware)
        return ((operation, launch),)
    if operation == "prepare_next_padded":
        require_dtype(dtype, ("int32",))
        block = 1 << (width - 1).bit_length()
        launch = _metadata_launch("prepare_next_token_padded", rows, source_BLOCK_SIZE_TOKENS=block)
    elif operation == "prepare_inputs_padded":
        require_dtype(dtype, ("int32",))
        if width != 1:
            raise ValueError("MTP input preparation needs scalar per-request output vectors")
        launch = _metadata_launch("prepare_inputs_padded", rows)
    elif operation == "step_metadata":
        require_dtype(dtype, ("int64",))
        if width != 1:
            raise ValueError("MTP metadata step needs scalar positions")
        block_size, limit, columns = require_shape(attributes, ("block_size", "max_model_len", "block_table_columns"))
        launch = _metadata_launch("step_slot_mapping_metadata", rows,
            source_block_size=block_size, source_max_model_len=limit, source_blocks_per_request=columns)
    else:
        raise NotImplementedError("MTP eager operation has no bound source launch recipe")
    return ((operation, launch),)


def select_mtp_prepare_next_token_recipe(shape, dtype, attributes, hardware):
    """Entire prepare_next_token_ids_padded: backup H2D then fused kernel.

    The CPU request lookup has already supplied each actual backup ID in
    the pinned CpuGpuBuffer. copy_to_gpu(num_reqs) always submits that4B/row
    transfer, even when no row is discarded. New empty output allocations
    add no fill. Counting and last-valid selection stay in the one kernel;
    valid positions need not form a prefix. Discarded rows always use backup
    and report count0. Source predicate is token!=-1 and token<vocab_size;
    it does not contain an extra token>=0 test.
    """
    from src.kernel.kernel_spec import PhysicalTransferConfig

    _source(dtype, attributes, hardware, "int32")
    rows, width, vocab = require_shape(shape, ("rows", "sampled_width", "vocab_size"))
    stride = attributes["source_sampled_row_stride"]
    if (max(rows, width, vocab) >= 2**31 or type(stride) is not int or not width <= stride < 2**31
            or rows * stride * 4 >= 2**63):
        raise ValueError("MTP sampled-token rows need actual signed host extents and valid row stride")
    if (attributes["source_sampled_layout"] != "dense_int32_columns_strided_rows"
            or attributes["source_backup_host_memory"] != "pinned_int32"
            or attributes["source_host_device_link"] != "pcie"
            or type(attributes["source_backup_rows"]) is not int or attributes["source_backup_rows"] != rows):
        raise NotImplementedError("MTP next-token source needs actual initialized pinned backup prefix and sampled row layout")
    sampled = attributes["sampled_token_ids"]
    if (not isinstance(sampled, (tuple, list)) or len(sampled) != rows
            or any(not isinstance(row, (tuple, list)) or len(row) != width
                or any(type(token) is not int or not -2**31 <= token < 2**31 for token in row) for row in sampled)):
        raise ValueError("MTP next-token source requires actual signed INT32 sampled-token rows")
    _vector(attributes, "discard_request_mask", rows, boolean=True)
    backup = _vector(attributes, "backup_next_token_ids", rows)
    if any(not 0 <= token < vocab for token in backup):
        raise ValueError("MTP backup token must be an actual valid vocabulary ID")
    copy = PhysicalTransferConfig(algorithm="stream_copy", bytes=rows * 4, direction="h2d", link="pcie")
    block = 1 << (width - 1).bit_length()
    kernel = _metadata_launch("prepare_next_token_padded", rows, source_BLOCK_SIZE_TOKENS=block)
    return (("backup_next_token_ids_h2d", copy), ("prepare_next_token_padded", kernel))


def select_mtp_prepare_inputs_launch(shape, dtype, attributes, hardware):
    """Complete prepare_inputs_padded GPU call;no target-row compaction.

    Inclusive cumulative draft counts are distinct from zero-prefixed query
    starts. Fresh rows with no draft force rejected=0 even if valid_count0;
    verification rows use draft+1-valid_count as written. Discarded verify
    rows may therefore select the row before their own query start. No
    additional clamp or invented accepted token is inserted by this helper.
    The original query/seq/slot tensors are reused in new CPU metadata.
    """
    _source(dtype, attributes, hardware, "int32")
    rows, tokens = require_shape(shape, ("rows", "target_rows"))
    if max(rows, tokens) >= 2**31:
        raise ValueError("MTP padded query extents must fit source INT32")
    cumulative = _vector(attributes, "cu_num_draft_tokens", rows)
    valid = _vector(attributes, "valid_sampled_tokens_count", rows)
    starts = attributes["query_start_loc"]
    if (not isinstance(starts, (tuple, list)) or len(starts) != rows + 1
            or any(type(value) is not int for value in starts)
            or starts[0] != 0 or starts[-1] != tokens
            or any(a >= b for a, b in zip(starts, starts[1:]))):
        raise ValueError("MTP padded inputs need actual complete positive target query segments")
    previous = 0
    for seq, end in enumerate(cumulative):
        drafts = end - previous
        if drafts < 0 or drafts >= starts[seq + 1] - starts[seq] or not 0 <= valid[seq] <= drafts + 1:
            raise ValueError("MTP draft/valid counts must match actual target request extents")
        previous = end
    return _metadata_launch("prepare_inputs_padded", rows)


def select_mtp_step_metadata_launch(shape, dtype, attributes, hardware):
    """Complete 1-D autoregressive position/slot/seq-len update kernel.

    Each graph padding program writes only slot=-1;it does not touch position
    or seq_len. Overflow at max_model_len writes position0/seq_len1/slot=-1,
    while still reading table column0. The source also clamps the block table
    column to its actual last column. No separate fill/cast is submitted.
    """
    _source(dtype, attributes, hardware, "int64")
    padded, columns, block_size, limit = require_shape(shape,
        ("input_batch_size", "blocks_per_request", "block_size", "max_model_len"))
    rows = shape["rows"]
    if (type(rows) is not int or not 0 <= rows <= padded or padded >= 2**31
            or max(columns, block_size, limit) >= 2**31 - 1
            or attributes["source_uses_mrope"] is not False
            or attributes["source_position_layout"] != "contiguous_int64_1d"
            or attributes["source_block_table_dtype"] != "int32"):
        raise NotImplementedError("MTP step requires actual text1D positions and valid padded/table host extents")
    positions, table = attributes["positions"], attributes["block_table"]
    lengths = _vector(attributes, "seq_lens", rows)
    if (not isinstance(positions, (tuple, list)) or len(positions) != rows
            or any(type(position) is not int or not 0 <= position < 2**63 - 1 for position in positions)
            or any(not 0 <= length <= limit for length in lengths)):
        raise ValueError("MTP step needs actual nonnegative INT64 positions and INT32 sequence lengths")
    if (not isinstance(table, (tuple, list)) or len(table) != rows
            or any(not isinstance(row, (tuple, list)) or len(row) != columns
                or any(type(value) is not int or not -2**31 <= value < 2**31 for value in row) for row in table)):
        raise ValueError("MTP step needs actual INT32 block-table rows")
    stride = attributes["source_block_table_row_stride"]
    if type(stride) is not int or stride < columns or rows * stride * 4 >= 2**63:
        raise ValueError("MTP step block-table addressing exceeds source extents")
    for position, row in zip(positions, table):
        if position + 1 >= limit:
            continue  # Column0 is read,but its slot product is discarded.
        block = row[min((position + 1) // block_size, columns - 1)]
        # block_id is loaded as INT32; its multiply precedes the INT64
        # position-offset add. Unread columns need no invented live pages.
        if block < 0 or block * block_size + block_size - 1 >= 2**31:
            raise ValueError("MTP step live slot addressing exceeds source INT32 product domain")
    return _metadata_launch("step_slot_mapping_metadata", padded,
        source_block_size=block_size, source_max_model_len=limit, source_blocks_per_request=columns)


def select_mtp_valid_count_copy_transfer(shape, dtype, attributes, hardware):
    """Runner's optional side-stream D2H;separate from proposer main chain.

    The copy stream waits on the prepare-next-token producer,then copies4B
    per request and records its event. The main prepare_inputs call doesnot
    wait for this copy. Consumers must preserve that dependency/direction;
    this descriptor doesnot assert a price or an extra default-stream wait.
    """
    from src.kernel.kernel_spec import PhysicalTransferConfig

    _source(dtype, attributes, hardware, "int32")
    rows, = require_shape(shape, ("rows",))
    event = attributes["source_valid_count_copy_event"]
    if type(event) is not bool:
        raise TypeError("MTP runner count-copy event presence must be an actual boolean")
    if not event:
        return None
    if attributes["source_count_host_memory"] != "pinned_int32" or attributes["source_host_device_link"] != "pcie":
        raise NotImplementedError("MTP runner count copy needs actual pinned host destination/link")
    return PhysicalTransferConfig(algorithm="stream_copy", bytes=rows * 4, direction="d2h", link="pcie")


def select_mtp_first_pass_inputs_recipe(shape, dtype, attributes, hardware):
    """Entire no-extra-slot/text set_inputs_first_pass before model forward.

    First pass keeps ALL target rows,shifts N-1 IDs,then scatters the actual per-request next
    IDs. Fresh None indices materialize query_end-1;provided verification
    indices retain their INT32/INT64 dtype and any legal negative offsets.
    Negative assignment indices address the full persistent ID buffer,not
    just its active target prefix;the index tensor itself is not changed.
    Advanced assignment widens INT32 indices before scalar IndexPut128x4.
    Repeated indices retain source nondeterministic-write semantics;no
    deterministic value equivalence is asserted. Positions/hidden copies
    depend on proven alias relations,not on equal shapes. Constructor zero
    initialization and the model itself lie outside this method boundary.
    """
    from src.kernel.kernel_spec import PhysicalTransferConfig
    from src.kernel.launch_config.nvidia_h200.torch_vector_launch import (
        select_torch213_integer_cast_launch, select_torch213_query_last_row_launch)

    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("MTP first-pass source requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16"))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_base_proposer_first_pass"
            or attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_use_heterogeneous_vocab"] is not False
            or attributes["source_needs_extra_input_slots"] is not False
            or attributes["source_draft_uses_mrope"] is not False
            or attributes["source_target_uses_mrope"] is not False
            or attributes["source_token_ids_dtype"] != "int32"
            or attributes["source_positions_dtype"] != "int64"
            or attributes["source_target_hidden_dtype"] != dtype
            or attributes["source_input_layout"] != "contiguous_ids_positions_hidden_rows"
            or attributes["source_token_buffers_alias"] != "distinct"
            or attributes["source_deterministic_algorithms"] is not False):
        raise NotImplementedError("MTP first-pass requires exact0.30/Torch2.13 no-extra-slot homogeneous text source")
    rows, requests, hidden = require_shape(shape, ("rows", "requests", "hidden"))
    if rows >= 2**31 or 8 * (requests - 1) >= 2**31 - 1 or rows * hidden * 2 >= 2**63:
        raise NotImplementedError("MTP first-pass addressing requires complete source iterator extents")
    capacities = attributes["source_buffer_row_capacities"]
    if (not isinstance(capacities, (tuple, list)) or len(capacities) != 3
            or any(type(value) is not int or not rows <= value < 2**63 for value in capacities)
            or max(capacities[0] * 4, capacities[1] * 8, capacities[2] * hidden * 2) >= 2**63):
        raise ValueError("MTP input/position/hidden persistent buffers need actual sufficient row capacities")
    starts = attributes["query_start_loc"]
    if (not isinstance(starts, (tuple, list)) or len(starts) != requests + 1
            or any(type(value) is not int for value in starts)
            or starts[0] != 0 or starts[-1] != rows
            or any(a >= b for a, b in zip(starts, starts[1:]))):
        raise ValueError("MTP first-pass needs every incoming request's complete target query segment")
    stages = []
    indices = attributes["token_indices_to_sample"]
    if indices is None:
        indices = tuple(end - 1 for end in starts[1:])
        index_dtype = "int32"
        stages.append(("query_last_rows", select_torch213_query_last_row_launch(
            {"requests": requests}, attributes, hardware)))
    else:
        index_dtype = attributes["source_sampling_indices_dtype"]
        if index_dtype not in ("int32", "int64") or attributes["source_sampling_indices_layout"] != "contiguous_aligned_16":
            raise NotImplementedError("MTP verification indices require actual contiguous aligned INT32/INT64 source")
        if (not isinstance(indices, (tuple, list)) or len(indices) != requests
                or any(type(index) is not int or not -capacities[0] <= index < rows for index in indices)
                or (index_dtype == "int32" and any(not -2**31 <= index < 2**31 for index in indices))):
            raise ValueError("MTP first-pass sampling indices must be actual valid signed target row offsets")
    if rows > 1:
        stages.append(("shift_target_ids", PhysicalTransferConfig(
            algorithm="stream_copy", bytes=(rows - 1) * 4, direction="d2d", link="dram")))
    if index_dtype == "int32":
        stages.append(("sampling_indices_long", select_torch213_integer_cast_launch({"element_count": requests}, "int32",
            {"tensor_iterator_source": "pytorch_2_13_sm90", "source_input_layout": "contiguous_aligned_16",
             "source_output_dtype": "int64"}, hardware)))
    stages.append(("next_token_index_put", LaunchConfig("pytorch_2_13_sm90_mtp_int32_index_put",
        ((requests + 511) // 512, 1, 1), (128, 1, 1), parameters={"dynamic_shared_bytes": 0})))
    for role, size, alias_key in (("positions_copy", rows * 8, "source_positions_copy_alias"),
                                  ("hidden_states_copy", rows * hidden * 2, "source_hidden_copy_alias")):
        alias = attributes[alias_key]
        if alias not in ("same_view", "distinct_contiguous"):
            raise NotImplementedError("MTP first-pass copy needs actual same-view or distinct-contiguous proof")
        if alias == "distinct_contiguous":
            stages.append((role, PhysicalTransferConfig(algorithm="stream_copy", bytes=size, direction="d2d", link="dram")))
    return tuple(stages)


def select_mtp_post_forward_rows_recipe(shape, dtype, attributes, hardware):
    """Complete ordinary row-selection block after first model/compaction.

    K0 still materializes sample_hidden before the source returns an empty
    output. K1 stops at that same selection and samples next. K>1 additionally
    indexes the persistent position vector and feedback hidden separately,
    before first sampling. Even when last_hidden_states is hidden_states,
    the source performs two independent advanced gathers and repeats the
    INT32-index widening. Model/logits/sample/compaction are producer or
    consumer calls outside this block,not invented aggregate launches.
    """
    from src.kernel.launch_config.nvidia_h200.torch_vector_launch import (
        select_torch213_row_index_recipe, select_torch213_integer_vector_index_recipe)

    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("MTP post-forward selection requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16"))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_base_proposer_post_forward_rows"
            or attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_parallel_drafting"] is not False
            or attributes["source_uses_mrope"] is not False
            or attributes["source_constant_draft_positions"] is not False
            or attributes["source_hidden_layout"] != "contiguous_aligned_16_2d"
            or attributes["source_hidden_return_relation"] not in ("same_tensor", "distinct_tensors")
            or attributes["source_topk_compaction_state"] not in ("not_requested", "completed_for_actual_sampling_indices")):
        raise NotImplementedError("MTP post-forward rows require exact0.30 sequential text/dense2D source after any compaction")
    requests, sample_rows, sample_hidden = require_shape(shape, ("requests", "sample_rows", "sample_hidden"))
    depth = shape["draft_depth"]
    if type(depth) is not int or not 0 <= depth < 2**31:
        raise ValueError("MTP depth must be the actual non-negative source draft count")
    if attributes["source_sampling_indices_layout"] != "contiguous_aligned_16":
        raise NotImplementedError("MTP row selection requires actual contiguous aligned indices")
    source_attrs = {"tensor_iterator_source": "pytorch_2_13_sm90", "source_input_layout": "contiguous_aligned_16",
                    "source_index_dtype": attributes["source_sampling_indices_dtype"],
                    "source_row_indices": attributes["token_indices_to_sample"]}
    sample = select_torch213_row_index_recipe({"rows": requests, "source_rows": sample_rows, "columns": sample_hidden},
        dtype, source_attrs, hardware)
    stages = [("sample_hidden_" + role, launch) for role, launch in sample]
    if depth <= 1:
        return tuple(stages)
    position_rows, feedback_rows, feedback_hidden = require_shape(shape,
        ("position_rows", "feedback_rows", "feedback_hidden"))
    if (attributes["source_hidden_return_relation"] == "same_tensor"
            and (sample_rows, sample_hidden) != (feedback_rows, feedback_hidden)):
        raise ValueError("same model-return tensor cannot have different sample/feedback extents")
    positions = select_torch213_integer_vector_index_recipe({"rows": requests, "source_rows": position_rows},
        "int64", source_attrs, hardware)
    feedback = select_torch213_row_index_recipe({"rows": requests, "source_rows": feedback_rows, "columns": feedback_hidden},
        dtype, source_attrs, hardware)
    return (*stages, *(("positions_" + role, launch) for role, launch in positions),
            *(("feedback_hidden_" + role, launch) for role, launch in feedback))


def _iteration_frame(shape, dtype, attributes, hardware):
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("MTP iteration inputs require H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16"))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_base_proposer_iteration_inputs"
            or attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_parallel_drafting"] is not False
            or attributes["source_constant_draft_positions"] is not False
            or attributes["source_uses_mrope"] is not False
            or attributes["source_use_heterogeneous_vocab"] is not False
            or attributes["source_supports_mm_inputs"] is not False
            or attributes["source_iteration_layout"] != "contiguous_ids_dense_hidden_rows"):
        raise NotImplementedError("MTP iteration requires exact0.30 ordinary sequential text source")
    rows, padded, hidden, depth = require_shape(shape, ("rows", "input_batch_size", "hidden", "draft_depth"))
    index = attributes["source_iteration_index"]
    if depth < 2 or type(index) is not int or not 0 <= index < depth - 1 or padded < rows:
        raise ValueError("MTP iteration must be an actual successor round with sufficient graph padding")
    capacities = attributes["source_buffer_row_capacities"]
    if (not isinstance(capacities, (tuple, list)) or len(capacities) != 3
            or any(type(value) is not int or value < padded for value in capacities)
            or max(capacities[0] * 4, capacities[1] * 8, capacities[2] * hidden * 2) >= 2**63):
        raise ValueError("MTP graph input buffers must hold actual logical and padded rows")
    token_dtype = attributes["source_previous_draft_ids_dtype"]
    if token_dtype not in ("int32", "int64"):
        raise NotImplementedError("MTP successor IDs require actual sampled INT32/INT64 dtype")
    return rows, hidden, token_dtype


def select_mtp_iteration_ids_recipe(shape, dtype, attributes, hardware):
    """Complete draft_token_ids.int() BEFORE position/metadata rebuilding.

    Return no submission for actual INT32 input,one dynamic cast for INT64.
    The position/slot helper and attention-metadata builder follow this
    operator;the post-metadata buffer copies are a separate recipe below.
    """
    from src.kernel.launch_config.nvidia_h200.torch_vector_launch import select_torch213_integer_cast_launch

    rows, _, token_dtype = _iteration_frame(shape, dtype, attributes, hardware)
    if token_dtype == "int32":
        return ()
    return (("draft_ids_int32", select_torch213_integer_cast_launch({"element_count": rows}, "int64",
        {"tensor_iterator_source": "pytorch_2_13_sm90", "source_input_layout": "contiguous_aligned_16",
         "source_output_dtype": "int32"}, hardware)),)


def select_mtp_iteration_buffer_recipe(shape, dtype, attributes, hardware):
    """All ordinary graph-buffer copies AFTER attention metadata rebuilding.

    Copies cover logical batch rows only;graph padding retains the prior
    buffer contents. Do not insert padding fills or normalize hidden again.
    A metadata builder may itself submit GPU work between the preceding
    ID cast/slot update and these copies. These two recipes must not erase
    that caller boundary or invent a complete model/iteration launch.
    """
    from src.kernel.kernel_spec import PhysicalTransferConfig

    rows, hidden, token_dtype = _iteration_frame(shape, dtype, attributes, hardware)
    stages = []
    for role, size, key in (("draft_ids_buffer_copy", rows * 4, "source_ids_copy_alias"),
                            ("feedback_hidden_buffer_copy", rows * hidden * 2, "source_hidden_copy_alias")):
        alias = attributes[key]
        if alias not in ("same_view", "distinct_contiguous"):
            raise NotImplementedError("MTP iteration copy requires proven same-view or distinct-contiguous buffers")
        if role == "draft_ids_buffer_copy" and token_dtype == "int64" and alias == "same_view":
            raise ValueError("fresh INT64->INT32 cast cannot alias a live persistent ID buffer")
        if alias == "distinct_contiguous":
            stages.append((role, PhysicalTransferConfig(algorithm="stream_copy", bytes=size, direction="d2d", link="dram")))
    return tuple(stages)


def select_mtp_seq_len_rollback_launch(shape, dtype, attributes, hardware):
    """One inplace seq_lens-=rejected operation before the successor loop.

    K0/K1 exit earlier and submit no rollback. A fresh proposal's absent
    rejected-count tensor also submits none. A present all-zero tensor still
    submits subtraction. This is not repeated once per successor iteration.
    """
    from src.kernel.launch_config.nvidia_h200.torch_vector_launch import select_torch213_int32_subtract_launch

    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("MTP rollback requires H200 HWSpec")
    require_dtype(dtype, ("int32",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_base_proposer_seq_len_rollback"
            or attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_parallel_drafting"] is not False):
        raise NotImplementedError("MTP rollback requires exact0.30 sequential source")
    rows, = require_shape(shape, ("rows",))
    depth = shape["draft_depth"]
    if type(depth) is not int or depth < 0:
        raise ValueError("MTP rollback requires actual non-negative draft depth")
    if depth <= 1:
        return None
    rejected = attributes["num_rejected_tokens"]
    if rejected is None:
        return None
    lengths = attributes["seq_lens"]
    if (not isinstance(rejected, (tuple, list)) or not isinstance(lengths, (tuple, list))
            or len(rejected) != rows or len(lengths) != rows
            or any(type(value) is not int or not 0 <= value < 2**31 for value in (*rejected, *lengths))
            or any(count > length for count, length in zip(rejected, lengths))):
        raise ValueError("MTP rollback requires actual INT32 rejected counts and current sequence lengths")
    alignments = attributes["source_subtract_address_alignments"]
    if isinstance(alignments, (tuple, list)) and len(alignments) == 3 and alignments[0] != alignments[1]:
        raise ValueError("inplace MTP rollback output and sequence-length input must share alignment")
    return select_torch213_int32_subtract_launch({"element_count": rows}, attributes, hardware)


def select_mtp_draft_output_recipe(shape, dtype, attributes, hardware):
    """Complete non-probabilistic sequential draft-ID output materialization.

    K0 allocates empty INT64 output;K1 returns a view,both with no output kernel.
    K>1 stacks the ORIGINAL sampled vectors,not the temporary INT32 inputs
    cast inside the loop. Actual sampled INT32 or INT64 dtype is retained.
    Probability stacking and parallel drafting are separate source programs.
    """
    from src.kernel.launch_config.nvidia_h200.torch_vector_launch import select_torch213_integer_stack_recipe

    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("MTP draft output requires H200 HWSpec")
    require_dtype(dtype, ("int32", "int64"))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_base_proposer_draft_output"
            or attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_parallel_drafting"] is not False
            or attributes["source_returned_draft_probs"] is not False):
        raise NotImplementedError("MTP output requires exact0.30 sequential no-probability source")
    rows = shape["rows"]
    if type(rows) is not int or rows < 0:
        raise ValueError("MTP output request count must be non-negative")
    depth = shape["draft_depth"]
    if type(depth) is not int or depth < 0:
        raise ValueError("MTP output requires actual non-negative draft depth")
    if depth == 0 and dtype != "int64":
        raise ValueError("MTP depth0 returns source-defined empty INT64 output")
    if depth <= 1:
        return ()
    return select_torch213_integer_stack_recipe({"rows": rows, "inputs": depth}, dtype, attributes, hardware)


def select_mtp_greedy_logits_recipe(shape, dtype, attributes, hardware):
    """Complete greedy draft reduction after model.compute_logits returns.

    SpecDecodeBaseProposer directly calls logits.argmax(-1). Unlike the
    target Sampler.forward, this neither casts the logits to FP32 nor
    narrows the INT64 result. FP16/BF16 accumulation happens inside the
    argmax kernel. The model's norm/head/logits processing and any TP
    communication must have completed before this prepared-input boundary;
    their source and launch programs remain independent caller obligations.
    Local argmax reduction, heterogeneous vocab and random drafting have
    different source programs and are not admitted here.
    """
    from src.kernel.launch_config.nvidia_h200.sampler_launch import select_torch213_argmax_recipe

    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("MTP greedy logits require H200 HWSpec")
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_base_proposer_greedy_logits"
            or attributes["source_logits_state"] != "prepared_compute_logits_output"
            or attributes["source_use_local_argmax_reduction"] is not False
            or attributes["source_use_heterogeneous_vocab"] is not False):
        raise NotImplementedError("MTP greedy logits require exact0.30 prepared ordinary argmax source")
    probabilistic = attributes["source_enable_probabilistic_draft_probs"]
    all_greedy = attributes["source_sampling_all_greedy"]
    if type(probabilistic) is not bool or type(all_greedy) is not bool:
        raise ValueError("MTP draft sampling flags must be actual source booleans")
    if probabilistic and not all_greedy:
        raise NotImplementedError("probabilistic draft sampling needs its separate source program")
    return select_torch213_argmax_recipe(shape, dtype, attributes, hardware)


def select_mtp_topk_compaction_recipe(shape, dtype, attributes, hardware):
    """Entire DeepSeekMultiTokenPredictor.compact_topk_indices call.

    The proposer calls this after the first model pass when sharing and
    set_skip_topk are enabled, before even the K0/K1 early exits. It visits
    ALL eligible MTP layers, not just the layer selected by spec_step_idx.
    Every layer gathers into fresh storage, then copies to its buffer prefix.
    INT32 indices widen separately for each layer. Identity indices do not
    remove either operation; negative indices address full buffer capacity.
    Actual buffer-presence facts encode the source attribute-chain guards;
    no layer or buffer is inferred from model names or proposal depth.
    """
    from src.kernel.kernel_spec import PhysicalTransferConfig
    from src.kernel.launch_config.nvidia_h200.torch_vector_launch import select_torch213_row_index_recipe

    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("MTP top-k compaction requires H200 HWSpec")
    require_dtype(dtype, ("int32",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_deepseek_mtp_topk_compaction"
            or attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"):
        raise NotImplementedError("MTP top-k compaction requires exact0.30 DeepSeek predictor source")
    sharing, has_set_skip = attributes["source_share_mtp_indices"], attributes["source_model_has_set_skip_topk"]
    if type(sharing) is not bool or type(has_set_skip) is not bool:
        raise ValueError("MTP top-k sharing/capability flags must be actual source booleans")
    if not sharing or not has_set_skip:
        return ()
    layers, = require_shape(shape, ("layers",))
    presence = attributes["source_layer_topk_buffer_presence"]
    if (not isinstance(presence, (tuple, list)) or len(presence) != layers
            or any(type(value) is not bool for value in presence)):
        raise ValueError("MTP compaction requires actual eligibility of every layer's top-k buffer")
    if not any(presence):
        return ()
    rows = shape["requests"]
    buffer_rows, topk = require_shape(shape, ("buffer_rows", "topk"))
    if type(rows) is not int or not 0 <= rows <= buffer_rows:
        raise ValueError("MTP compacted prefix must fit the actual persistent buffer")
    if attributes["source_topk_buffer_layout"] != "contiguous_aligned_16_int32_rows":
        raise NotImplementedError("MTP compaction requires actual contiguous aligned INT32 layer buffers")
    source_attrs = {"tensor_iterator_source": "pytorch_2_13_sm90", "source_input_layout": "contiguous_aligned_16",
        "source_index_dtype": attributes["source_sampling_indices_dtype"],
        "source_row_indices": attributes["token_indices_to_sample"]}
    stages = []
    for layer, present in enumerate(presence):
        if not present:
            continue
        gather = select_torch213_row_index_recipe({"rows": rows, "source_rows": buffer_rows, "columns": topk},
            "int32", source_attrs, hardware)
        stages.extend((f"layer_{layer}_" + role, launch) for role, launch in gather)
        if rows:
            stages.append((f"layer_{layer}_topk_prefix_copy", PhysicalTransferConfig(
                algorithm="stream_copy", bytes=rows * topk * 4, direction="d2d", link="dram")))
    return tuple(stages)
