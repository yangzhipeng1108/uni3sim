"""Complete plain vLLM0.30/Torch2.13 greedy sampler source on H200.

Sampler.forward converts16-bit logits toFP32,samples argmax,aliases the
INT64 result through view/long,then narrows toINT32 and returns an unsqueeze
view. Reduce.cuh's global split additionally submits cudaMemsetAsync before
the reduction. It is returned as a PhysicalTransferConfig source descriptor,
never a made-up CTA. Its stream_memset algorithm still needs shared consumer
admission before any aggregate native KernelSpec/price can be claimed.

All parameters follow official Torch2.13 Reduce.cuh/ReduceArgMaxKernel.cu
and Copy.cu;no2.8 binary resource,SM80 fact or duration is consumed.
The reusable argmax recipe also preserves native FP16/BF16 input for
prepared draft logits; the target sampler still performs its FP32 cast.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape
from src.kernel.launch_config.nvidia_h200.torch_vector_launch import (
    select_torch213_integer_cast_launch, select_torch213_logits_fp32_cast_launch)


def _source(attributes, hardware):
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch2.13 sampler requires H200 HWSpec")
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_input_layout"] != "contiguous_aligned_16"):
        raise NotImplementedError("Torch2.13 sampler requires actual contiguous aligned source logits")


def select_torch213_argmax_launch(shape, dtype, attributes, hardware):
    """Contiguous last-axis argmax,including exact source workspace.

    FP16/BF16 reduce directly with FP32 accumulator pairs in the same
    kernel; they do not materialize FP32 input. input_vec4/output_vec1
    and sizeof(pair<float,int64_t>)=16 determine
    geometry/shared/scratch. The runtime SM and thread limits are actual
    getCurrentDeviceProperties inputs,not occupancy measured from this
    workload. Compiler registers/static shared remain explicitly unbound.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch2.13 argmax requires H200 HWSpec")
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_input_layout"] not in ("contiguous", "contiguous_aligned_16")):
        raise NotImplementedError("Torch2.13 argmax requires actual contiguous source logits")
    # Reduce.cuh::input_vectorized_thread_reduce_impl handles an unaligned
    # input head before its aligned vector loop. setReduceConfig depends on
    # extents/strides and runtime limits, not the input pointer alignment.
    # This geometry admission does not broaden the complete sampler ABI.
    require_dtype(dtype, ("bf16", "fp16", "fp32"))
    rows, vocab = require_shape(shape, ("rows", "vocab"))
    width = 4 if dtype == "fp32" else 2
    if vocab < 128 or max(width * rows * vocab, 8 * rows) >= 2**31:
        raise NotImplementedError("Torch2.13 argmax requires vectorized unsplit32-bit contiguous rows")
    sms, threads = attributes["source_num_compute_units"], attributes["source_max_threads_per_unit"]
    if (type(sms) is not int or type(threads) is not int
            or not 0 < sms <= hardware.num_compute_units or threads != hardware.max_threads_per_unit):
        raise ValueError("Torch2.13 argmax runtime properties must match the declared H200 domain")
    # setReduceConfig<pair<float,int64_t>,scalar_t,4,4>;
    # mnt_wrapper<float/Half/BFloat16> fixes512 for all three inputs.
    dim0, dim1 = vocab // 4, rows
    x = min(1 << (dim0.bit_length() - 1), 512, 32)
    y = min(1 << (dim1.bit_length() - 1), 512 // x)
    x = min(1 << (dim0.bit_length() - 1), 512 // y)
    input_step = x
    split_warps = ceil_div(vocab, input_step) >= min(y * 16, 256)
    output_step = 1 if split_warps else y
    if split_warps:
        input_step *= y
    grid_x = ceil_div(rows, output_step)
    values = ceil_div(vocab, input_step)
    target_grid = sms * (threads // (x * y))
    ctas = 1
    if split_warps and values >= 256 and grid_x <= target_grid:
        ctas = max(min(ceil_div(target_grid, grid_x), ceil_div(values, 16)), ceil_div(values, 256))
    shared = 16 * x * y if split_warps or x > 32 else 0
    scalar = {"fp32": "float", "fp16": "half", "bf16": "bfloat16"}[dtype]
    return LaunchConfig(f"pytorch_2_13_sm90_argmax_{scalar}_long", (grid_x, ctas, 1), (x, y, 1), parameters={
        "dynamic_shared_bytes": float(shared), "source_input_vector_size": 4.0,
        "source_global_partial_allocation_bytes": float(16 * rows * ctas if ctas > 1 else 0),
        "source_global_semaphore_allocation_bytes": float(4 * grid_x if ctas > 1 else 0)})


def select_plain_greedy_sampler_recipe(shape, dtype, attributes, hardware):
    """Every CUDA submission of the plain legacy-runner0.30 sampler call.

    The caller selects the actual legacy Sampler entry,not the separate new
    gpu/sample sampler. Logprobs,penalties,masks,processors and thinking
    budgets must be absent. Temperature/top-p are not read by all-greedy.
    The returned memset describes one runtime operation,not a kernel or a
    supported transfer price. PhysicalTransferConfig has no memset typed
    consumer yet;consumers must reject that missing service rather than
    relabeling it stream_copy or silently dropping it.
    """
    _source(attributes, hardware)
    require_dtype(dtype, ("bf16", "fp16", "fp32"))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_legacy_plain_greedy_sampler"
            or attributes["source_all_greedy"] is not True
            or attributes["source_all_random"] is not False
            or attributes["source_max_num_logprobs"] is not None
            or attributes["source_logprob_token_ids"] != ()
            or attributes["source_allowed_token_ids_mask"] != "none"
            or attributes["source_bad_words"] != ()
            or attributes["source_non_argmax_processors"] != ()
            or attributes["source_no_penalties"] is not True
            or attributes["source_thinking_budget_holder"] != "none"):
        raise NotImplementedError("plain sampler requires its exact0.30 all-greedy source metadata without extra processing")
    rows, vocab = require_shape(shape, ("rows", "vocab"))
    # Output IDs are known vocab offsets and must fit the narrowed dtype.
    if vocab >= 2**31:
        raise NotImplementedError("plain greedy vocabulary must fit INT32 output IDs")
    stages = []
    if dtype != "fp32":
        stages.append(("logits_fp32_cast", select_torch213_logits_fp32_cast_launch(
            {"element_count": rows * vocab}, dtype, attributes, hardware)))
    stages.extend(select_torch213_argmax_recipe(shape, "fp32", attributes, hardware))
    stages.append(("sample_ids_int32", select_torch213_integer_cast_launch({"element_count": rows}, "int64", {
        **attributes, "source_output_dtype": "int32"}, hardware)))
    return tuple(stages)


def select_torch213_argmax_recipe(shape, dtype, attributes, hardware):
    """Complete library argmax call,including its conditional runtime zero."""
    from src.kernel.kernel_spec import PhysicalTransferConfig

    reduction = select_torch213_argmax_launch(shape, dtype, attributes, hardware)
    semaphore_bytes = int(reduction.parameters["source_global_semaphore_allocation_bytes"])
    prefix = (("argmax_semaphore_memset_zero", PhysicalTransferConfig(
        algorithm="stream_memset", bytes=semaphore_bytes, direction="d2d", link="dram", operation="zero")),) if semaphore_bytes else ()
    return (*prefix, ("argmax", reduction))
