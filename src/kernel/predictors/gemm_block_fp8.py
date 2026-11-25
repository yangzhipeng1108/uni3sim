"""Hopper FP8 block-scaled dense GEMM template (vLLM 0.24 stack).

Prices the two launches a hardware block-FP8 selector emits (the Hopper
package reproduces the vLLM 0.24 FlashInfer/DeepGEMM heuristics):

``trtllm_fp8_blockscale_swapab_gemm`` (M < 32)
    FlashInfer/TensorRT-LLM ``fp8_gemm_run``: ``scale_1x128_kernel`` over
    ``num_sms * 8`` CTAs, then the persistent ``fp8_gemm_kernel_swapAB``.
``deepgemm_sm90_fp8_gemm_1d2d`` (M >= 32)
    DeepGEMM persistent ``sm90_fp8_gemm_1d2d_impl`` (its separate
    ``per_token_group_quant`` activation quantiser is not part of this call).

Execution structure (source): a grid of ``num_sms`` persistent CTAs, one per
SM; CTA ``i`` owns tiles ``i, i + num_sms, ...``.  Per tile an elected TMA
thread streams ``ceil(K / 128)`` k-blocks of the weight box and the
activation box through a ``stages``-deep full/empty mbarrier ring; the math
warp-groups wait on each stage, issue the FP8 wgmma k-block and release the
stage; the epilogue stores the BF16 tile with TMA.

Timing template:

* one persistent wave (``tiles <= num_sms``, every decode launch): each
  working CTA runs exactly one tile, so the launch lasts one TMA-ring pass:
  a fill (launch shell, ring priming, BF16 tile store) plus ``ceil(K/128)``
  stage intervals.  The interval composes the per-SM ring stage service with
  the chip DRAM stream of the ``ceil(N / tile_n)`` distinct weight boxes per
  stage round; both, the composition exponent and the fill come from the
  hardware's MMA-free TMA-ring probe (``fp8_tma_operand_stream.cu`` under its
  ``microbench/<hardware>/operators/gemm``).
  The k-block tensor time at the public FP8 rate bounds the interval below.
* several waves (prefill): the busiest CTA runs ``ceil(tiles / num_sms)``
  tiles of ``ceil(K/128)`` k-blocks each at the public FP8 tensor rate of one
  SM (never below the per-SM ring stage service), bounded below by the
  compulsory A/B reads at the measured TMA DRAM stream rate plus the BF16
  output writes at the measured full-chip tile-store rate; the measured
  launch shell and the final tile-store drain (one BF16 tile per CTA at the
  tile-store rate) are added.

Every duration comes from the HWSpec ``block_fp8_gemm_tma_ring_service``
section (mechanism probes) or the HWSpec public FP8 tensor rate; the
launch span floor composes by maximum.  No production duration is read.

The optional linear-wrapper source program describes conditional vLLM 0.30
tensor objects only. It does not bind that runtime to this earlier launch
template or add traffic for unused, uninitialized placeholder allocations.
"""

from src.kernel.formula import small_kernel_span_floor_s
from src.kernel.kernel_spec import KernelKind, KernelPrediction, KernelSpec
from src.kernel.launch_config.block_fp8_gemm import (
    BLOCK_FP8_GEMM_ALGORITHMS,
    DEEPGEMM_1D2D_ALGORITHM,
    DEEPGEMM_FP8_EINSUM_BATCHED_ALGORITHM,
    TRTLLM_SCALE_1X128_CTAS_PER_SM,
    TRTLLM_SCALE_1X128_THREADS,
    TRTLLM_SWAP_AB_ALGORITHM,
)
from src.kernel.predictors.common import (
    graph_replayed_launch,
    prediction_from_ns,
)

_BLOCK_K = 128


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def ring_stage_interval_ns(ring, weight_box_bytes, streaming_weight_blocks):
    """Per-stage TMA-ring interval of one CTA (measured service form).

    Each CTA pays the per-SM ring stage service ``stage_floor_ns``; the chip
    streams ``streaming_weight_blocks * weight_box_bytes`` distinct DRAM bytes
    per stage round at ``dram_stream_bytes_per_ns``; the two compose by the
    measured p-norm.
    """

    stream_ns = (streaming_weight_blocks * weight_box_bytes
                 / ring["dram_stream_bytes_per_ns"])
    power = ring["combination_power"]
    return (ring["stage_floor_ns"] ** power
            + stream_ns ** power) ** (1.0 / power)


def _fp8_tensor_flops_per_sm_s(hardware) -> float:
    tensor = hardware.resource_by_role("compute_tensor_fp8")
    return float(tensor.capacity) / int(hardware.num_compute_units)


def predict_block_fp8_gemm(kernel: KernelSpec, hardware) -> KernelPrediction:
    """Price one block-FP8 dense GEMM call."""

    launch = kernel.launch
    if launch.algorithm not in BLOCK_FP8_GEMM_ALGORITHMS:
        raise ValueError(
            f"not a block-FP8 GEMM launch: {launch.algorithm!r}")
    service = hardware.block_fp8_gemm_tma_ring_service
    if service is None:
        raise ValueError(
            f"{hardware.hardware_id} publishes no block-FP8 GEMM TMA-ring "
            "service")
    from src.kernel.memory_access_builder import block_fp8_memory_access_program
    from src.kernel.memory_path_solver import block_fp8_source_demands
    source = block_fp8_source_demands(block_fp8_memory_access_program(kernel, hardware))
    b, m, n, k = source["geometry"]
    if launch.algorithm == DEEPGEMM_FP8_EINSUM_BATCHED_ALGORITHM:
        # DeepSeek-V4 wo_a einsum: one block-scaled problem per o_group;
        # the fused inverse-RoPE + 1x128 activation quantisation is a
        # separate kernel, so no in-call scale kernel is charged.
        if b < 1:
            raise ValueError("FP8 einsum batch must be positive")
    elif b != 1:
        raise ValueError("block-FP8 GEMM is a single (b=1) GEMM")
    tiling = launch.tiling
    tile_m = int(tiling.tile_shape["m"])
    tile_n = int(tiling.tile_shape["n"])
    if int(tiling.tile_shape["k"]) != _BLOCK_K or tiling.split_k != 1:
        raise ValueError("block-FP8 GEMM uses BLOCK_K=128 without split-K")
    grid = launch.grid_size
    if grid != int(hardware.num_compute_units):
        raise ValueError("block-FP8 GEMM launches one persistent CTA per SM")
    tiles = b * _ceil_div(m, tile_m) * _ceil_div(n, tile_n)
    kblocks = _ceil_div(k, _BLOCK_K)
    tensor_kblock_ns = (
        2.0 * tile_m * tile_n * _BLOCK_K
        / _fp8_tensor_flops_per_sm_s(hardware) * 1e9)
    groups = {}
    ring = service["ring"]
    if tiles <= grid:
        # One persistent wave: every working CTA streams its tile's weight
        # rows while the ceil(N / tile_n) distinct weight blocks stream from
        # DRAM concurrently (CTAs sharing a weight block hit L2).
        interval_ns = ring_stage_interval_ns(
            ring, tile_n * _BLOCK_K * source["weight_element_bytes"], b * _ceil_div(n, tile_n))
        fill_ns = (ring["fill_base_ns"]
                   + ring["fill_per_store_byte_ns"] * source["output_element_bytes"] * tile_m * tile_n)
        groups["tma_ring_fill"] = fill_ns
        groups["tma_ring_stages"] = kblocks * interval_ns
        groups["fp8_tensor"] = kblocks * tensor_kblock_ns
        gemm_ns = fill_ns + kblocks * max(interval_ns, tensor_kblock_ns)
    else:
        waves = _ceil_div(tiles, grid)
        shell_ns = service["launch_shell_ns"]
        # The last tile of every CTA stores after the mainloop ends.
        store_drain_ns = (grid * source["output_element_bytes"] * tile_m * tile_n
                          / service["full_chip_tile_store_bytes_per_ns"])
        # Compulsory operand/scale reads and BF16 output writes share the
        # HBM interface, so their measured stream times add.
        read_bytes = source["main_read_bytes"]
        write_bytes = source["main_write_bytes"]
        groups["fp8_tensor"] = waves * kblocks * max(
            tensor_kblock_ns, ring["stage_floor_ns"])
        groups["dram_stream"] = (
            read_bytes / ring["dram_stream_bytes_per_ns"]
            + write_bytes / service["full_chip_tile_store_bytes_per_ns"])
        groups["launch_shell"] = shell_ns
        groups["tile_store_drain"] = store_drain_ns
        gemm_ns = (shell_ns + max(groups["fp8_tensor"], groups["dram_stream"])
                   + store_drain_ns)
    bottleneck = max(groups, key=lambda name: groups[name])
    replayed = graph_replayed_launch(kernel)
    gemm_floor_ns = small_kernel_span_floor_s(
        hardware, grid, kernel.variant, replayed) * 1e9
    groups["launch_span_floor"] = gemm_floor_ns
    if gemm_floor_ns > gemm_ns:
        gemm_ns = gemm_floor_ns
        bottleneck = "launch_span_floor"
    scale_ns = 0.0
    if launch.algorithm == TRTLLM_SWAP_AB_ALGORITHM:
        if TRTLLM_SCALE_1X128_THREADS != 256:
            raise AssertionError("scale_1x128 geometry drifted")
        scale_grid = TRTLLM_SCALE_1X128_CTAS_PER_SM * grid
        # BF16 input read, FP8 output and FP32 1x128 scales written.
        scale_bytes = source["quantizer_total_bytes"]
        scale_ns = max(
            small_kernel_span_floor_s(
                hardware, scale_grid, kernel.variant, replayed) * 1e9,
            scale_bytes / ring["dram_stream_bytes_per_ns"])
        groups["trtllm_scale_1x128"] = scale_ns
    total_ns = scale_ns + gemm_ns
    per_group_s = {name: value * 1e-9 for name, value in groups.items()}
    # Read-only CONDITIONAL phase allocation. The persistent source assigns
    # tile i to CTA i%SMs; nonworking persistent CTAs do not run the MMA body.
    # The template assumes uniform tile service, so tail workers release
    # after their tile count instead of allocating every SM to the kernel
    # endpoint. scale_ns includes its launch floor; tma_ring_fill includes
    # shell/priming. These envelopes do not identify pure SM body activity.
    # No production duration or extra rate is introduced.
    from src.kernel.model_contract import PredictionDiagnostics, PipelineDiagnostics, PipelineTaskDiagnostic
    tasks = []
    worker_end = {str(sm):0. for sm in range(grid)}
    if scale_ns:
        # scale_1x128: one warp owns one full128 group, eight warps/CTA.
        # CUDA CTA placement is the same analytical round-robin placement
        # used by the source worker model, with the HW thread-slot limit.
        groups_count = b*m*kblocks
        scale_ctas = _ceil_div(groups_count, TRTLLM_SCALE_1X128_THREADS//32)
        resident = min(hardware.max_blocks_per_unit,
            hardware.max_threads_per_unit//TRTLLM_SCALE_1X128_THREADS)
        scale_waves = _ceil_div(scale_ctas, grid*resident)
        for sm in range(min(scale_ctas, grid)):
            ctas = _ceil_div(max(0, scale_ctas-sm), grid)
            duration = scale_ns*1e-9 * _ceil_div(ctas, resident)/scale_waves
            tasks.append(PipelineTaskDiagnostic('scale_phase_allocation',sm,{'conditional_phase':0.},{'conditional_phase':duration}))
            worker_end[str(sm)] = duration
    if tiles <= grid:
        matrix_phase_ns = groups['tma_ring_fill'] + max(groups['tma_ring_stages'],groups['fp8_tensor'])
        tile_ns = matrix_phase_ns
    else:
        # Existing source price's per-worker uniform-tile fluid service.
        tile_ns = max(groups['fp8_tensor'],groups['dram_stream'])/_ceil_div(tiles,grid)
    for sm in range(min(tiles,grid)):
        owned_tiles = _ceil_div(tiles-sm,grid)
        duration = owned_tiles*tile_ns*1e-9
        start = scale_ns*1e-9
        end = start+duration
        if end > total_ns*1e-9+1e-12:
            raise ValueError('source persistent work exceeds its unchanged price')
        tasks.append(PipelineTaskDiagnostic('persistent_phase_allocation',sm,{'conditional_phase':start},{'conditional_phase':end}))
        worker_end[str(sm)] = max(worker_end[str(sm)],end)
    pipeline = PipelineDiagnostics(total_ns*1e-9,0.,worker_end,{},(),tuple(tasks))
    diagnostics = PredictionDiagnostics({},(),(),(),pipeline,(), 'partial',(
        'source_persistent_CTA_to_SM_round_robin_assignment',
        'uniform_persistent_tile_service_is_the_existing_template_approximation',
        'scale_CTA_residency_uses_public_thread_and_block_slots_register_limits_unbound',
        'allocated_phase_envelopes_include_unpartitioned_launch_floor_and_TMA_fill_shell',
        'conditional_phase_allocation_not_pure_body_or_complete_SM_activity'))
    return KernelPrediction(
        t_total_s=total_ns * 1e-9,
        t_per_group=per_group_s,
        t_launch_s=0.0,
        bottleneck_group=bottleneck,
        t_per_resource=dict(per_group_s),
        diagnostics=diagnostics,
    )


def trtllm_scale_1x128_source_program(rows: int, columns: int):
    """Conditional SDK source expressions and layout, never a provider binding.

    A warp owns each full 128-value group. The source repeats scale arithmetic
    per lane; these counts do not assert compiler instruction counts or rates.
    The amax predicate is value dependent, including for finite inputs.
    """
    if type(rows) is not int or rows < 1 or type(columns) is not int or columns < 1 or columns % 128:
        raise ValueError("scale_1x128 source program requires positive rows and full 128-value groups")
    groups = rows * (columns // 128)
    scale_stride = _ceil_div(rows, 4) * 4
    return {
        "typed_steps": ["bf16_abs_and_local_max", "fp32_warp_max_then_bf16_round_each_step",
                        "broadcast_lane0_bf16_amax", "fp32_conditional_scale",
                        "bf16_to_fp32_value_multiply", "e4m3_cast_store", "fp32_inverse_scale_store"],
        "source_expression_counts": {
            "bf16_abs": 128 * groups, "bf16_local_max": 128 * groups,
            "fp32_warp_max": 160 * groups, "amax_zero_comparisons": 32 * groups,
            "fp32_value_multiply": 128 * groups, "fp32_inverse_scale_divide": groups},
        "forward_scale_divide": {"predicate": "bf16_amax != bf16(0)", "true_expressions_per_group": 32,
                                 "false_expressions_per_group": 0, "true_group_count": None,
                                 "true_expression": "448.f / fp32(amax)", "false_value": "1.f"},
        "scale_layout": {"dtype": "fp32", "logical_shape": [columns // 128, rows],
                         "strides_elements": [scale_stride, 1], "logical_written_elements": groups,
                         "required_address_span_elements": (columns // 128 - 1) * scale_stride + rows,
                         "padded_extent_elements": columns // 128 * scale_stride,
                         "padding_written_by_quantizer": False, "actual_allocation_capacity_bound": False},
        "scope": "full-group SDK source expressions and logical scale addressing; not compiler issue counts or provider dispatch",
    }


def trtllm_workspace_source_program(rows: int, columns: int, output_columns: int):
    """Conditional dense BF16/FP8 SDK workspace, with explicit cached state.

    H is the runner's post-request maximum aligned M, not this call's M.
    The binding requests the maximum of three dtype runners but the dense
    BF16/FP8 callee only touches A and its scales. Empty storage, reserve,
    live payload and address span are deliberately different quantities.
    No allocation request is added to the Roofline traffic account.
    """
    if any(type(value) is not int or value <= 0 for value in (rows, columns, output_columns)) or columns % 128:
        raise ValueError("FP8 workspace source requires positive dimensions and full 128-value groups")
    aligned4 = _ceil_div(rows, 4) * 4
    aligned32 = _ceil_div(rows, 32) * 32
    groups = columns // 128
    scale_reserve = 4 * aligned32 * groups
    unused_weights = output_columns * columns + 4 * _ceil_div(output_columns, 128) * groups
    return {
        "condition": "installed FlashInfer SM90 dense BF16 A / FP8 B source branch; no provider selection",
        "state": {"parameter": "H", "definition": "max(prior_max_shape_m_4_align, ceil(M/4)*4)",
                  "prior_domain": "nonnegative multiple of 4", "current_aligned_rows": aligned4,
                  "H_value": None, "expected_m_used_by_dense_gemm": False},
        "allocation": {"dtype": "uint8", "operation": "torch.empty", "initialized": False,
                       "initialization_write_bytes": 0, "request_bytes": {"H_coefficient": columns,
                           "constant": scale_reserve + unused_weights},
                       "request_rule": "maximum of BF16/BF16, BF16/FP8 and FP8/FP8 requests",
                       "shared_base_for_three_runners": True, "allocator_capacity_bytes": None},
        "runner_request_bytes": {"bf16_bf16": {"H_coefficient": columns, "constant": scale_reserve + unused_weights},
                                 "bf16_fp8": {"H_coefficient": columns, "constant": scale_reserve},
                                 "fp8_fp8": {"H_coefficient": 0, "constant": 0}},
        "activation_view": {"dtype": "fp8", "offset_bytes": 0, "shape": [rows, columns],
                            "strides_elements": [columns, 1], "reserved_bytes": {"H_coefficient": columns, "constant": 0},
                            "quantizer_write_bytes": rows * columns, "matmul_unique_read_bytes": rows * columns,
                            "reserved_padding_written_by_quantizer": False},
        "scale_view": {"dtype": "fp32", "offset_bytes": {"H_coefficient": columns, "constant": 0},
                       "shape": [groups, rows], "strides_elements": [aligned4, 1],
                       "reserved_bytes": scale_reserve, "address_span_bytes": 4 * ((groups - 1) * aligned4 + rows),
                       "quantizer_write_bytes": 4 * groups * rows, "matmul_unique_read_bytes": 4 * groups * rows,
                       "padding_written_by_quantizer": False},
        "unused_reservation_bytes": unused_weights,
        "borrowed_operands": ["FP8 weight", "FP32 weight scales"],
        "source_lifetime": "fresh Python local workspace is passed to cached runner and used by quantizer then matmul",
        "source_substage_order": ["quantize_bf16_and_write_scales", "packed_fp8_matmul"],
        "dense_matrix_storage": {
            "condition": "SM90 normal dense BF16/FP8 SDK source with ENABLE_FP8_BLOCK_SCALE, COMPILE_HOPPER_TMA_GEMMS and without PLACEHOLDER_KERNELS, after in-call activation quantization",
            "ports": [
                {"role": "A", "storage": "workspace", "dtype": "fp8", "offset_bytes": 0,
                 "shape": [rows, columns], "strides_elements": [columns, 1], "access": "read"},
                {"role": "As", "storage": "workspace", "dtype": "fp32",
                 "offset_bytes": {"H_coefficient": columns, "constant": 0},
                 "shape": [groups, rows], "strides_elements": [aligned4, 1], "access": "read"},
                {"role": "B", "storage": "borrowed_weight", "dtype": "fp8", "offset_bytes": 0,
                 "shape": [output_columns, columns], "strides_elements": [columns, 1], "access": "read"},
                {"role": "Bs", "storage": "borrowed_weight_scales", "dtype": "fp32", "offset_bytes": 0,
                 "shape": [_ceil_div(output_columns, 128), groups], "strides_elements": [groups, 1], "access": "read"},
                {"role": "D", "storage": "caller_output", "dtype": "bf16", "offset_bytes": 0,
                 "shape": [rows, output_columns], "strides_elements": [output_columns, 1], "access": "write"}],
            "dispatch_alternatives": [
                {"condition": "getDeepGemmEnabled() == false", "entry": "Fp8Gemm::run"},
                {"condition": "getDeepGemmEnabled() == true and M >= 32", "entry": "deep_gemm::runGemm"},
                {"condition": "getDeepGemmEnabled() == true and M < 32", "entry": "deep_gemm::runGemmSwapAB"}],
            "additional_source_global_tensor_bytes": 0,
            "descriptors": "host values passed to kernel; refer to the five existing tensor ports",
            "grouped_layout_pointer": None,
            "staging_and_barriers": "shared memory, not an additional global tensor",
            "accumulators": "FP32 register fragments, not a global partial-result tensor",
            "H_value_provider_compiler_spills_or_physical_traffic_bound": False},
        "stream_completion_or_allocator_reuse_proven": False, "actual_provider_bound": False,
        "allocation_request_is_traffic": False,
        "scope": "conditional source workspace and normal-dense tensor ports; not compiler spill, runtime allocation or normal capacity"}


def fp8_linear_wrapper_source_program(rows: int, columns: int, output_columns: int):
    """Conditional vLLM wrapper tensor objects, independent of launch timing.

    The unused As argument in the BF16-input path is not the FP32 scale
    buffer produced inside the SDK. Empty allocations carry no initialized
    payload traffic. Runtime class and library scratch are not selected.
    """
    if any(type(value) is not int or value <= 0 for value in (rows, columns, output_columns)) or columns % 128:
        raise ValueError("FP8 wrapper source requires positive dimensions and full 128-value groups")
    return {
        "source_symbol": "Fp8BlockScaledMMLinearKernel.apply_weights",
        "input_view": {"dtype": "bf16", "shape": [rows, columns], "aliases_input": True, "allocates": False},
        "output": {"dtype": "bf16", "shape": [rows, output_columns],
                   "to_dtype_allocates_if_mm_returns_configured_dtype": False, "final_view_allocates": False,
                   "bias": None},
        "program_alternatives": [
            {"condition": "apply_input_quant=True; dynamic full-group FP32-scale source template",
             "q_input": {"dtype": "fp8", "shape": [rows, columns], "aliases_input": False},
             "As": {"role": "quantizer_activation_scales", "dtype": "fp32",
                    "logical_shape": [rows, columns // 128], "read_by_matmul": True},
             "quantizer": "separate producer; FP8 values and FP32 scales are read by matmul"},
            {"condition": "FlashInferFp8BlockScaledMMKernel.apply_input_quant=False",
             "q_input": {"dtype": "bf16", "shape": [rows, columns], "aliases_input": True},
             "As": {"role": "unused_wrapper_argument", "read_by_matmul": False,
                    "when_input_scale_present": "alias existing input_scale; no new allocation",
                    "when_input_scale_absent": {"operation": "input_2d.new_empty(1)", "dtype": "bf16",
                        "shape": [1], "logical_storage_bytes": 2, "initialized": False,
                        "logical_read_bytes": 0, "logical_write_bytes": 0}},
             "quantizer": "inside SDK; its FP32 activation scales are a different object from As"}],
        "actual_wrapper_selected": None, "actual_provider_bound": False,
        "scope": "conditional no-bias vLLM BF16-output wrapper objects; not caller identity, packed-scale runtime, library scratch, peak capacity or extra payload traffic"}


def _block_fp8_matmul_payloads(b, m, n, k):
    kblocks = _ceil_div(k, _BLOCK_K)
    return ({"matmul_input": b * m * k, "weight_payload": b * n * k,
             "output": 2 * b * m * n},
            {"activation_scales": 4 * b * m * kblocks,
             "weight_scales": 4 * b * _ceil_div(n, _BLOCK_K) * kblocks})


def block_fp8_linear_logical_work(kernel: KernelSpec):
    """Ideal FP8 A/B, BF16 C and FP32 group-scale payload, without launch.

    vLLM0.30 create_fp8_scale_parameter uses ceil(N/128), including the
    unsharded merged projection's final partial output block. Input
    quantization is a separate producer; no library padding/scratch or
    actual scale layout/provider is inferred from these logical extents.
    """
    if (kernel.kind is not KernelKind.GEMM or kernel.variant != "gemm"
            or "mathematical_program" not in kernel.attributes
            or kernel.attributes["mathematical_program"] != "block_fp8_fp32_scales_v1"
            or kernel.dtype != "bf16" or kernel.execution_support != "unsupported"
            or kernel.launch is not None):
        raise ValueError("block-FP8 logical work requires its unbound mathematical program")
    b, m, n, k = (kernel.shape[name] for name in ("b", "m", "n", "k"))
    payloads, scales = _block_fp8_matmul_payloads(b, m, n, k)
    return {"mathematical_flops": 2 * b * m * n * k,
        "operand_payload_bytes": payloads, "scale_bytes": scales,
        "logical_bytes": sum(payloads.values()) + sum(scales.values()),
        "limitations": ("matmul_only_separate_BF16_to_FP8_quantization_producer",
            "logical_payload_not_HBM_traffic_or_scale_allocation_padding",
            "scale_application_conversion_and_comparison_instruction_services_unpriced",
            "compiler_provider_launch_library_workspace_and_native_unbound")}


def block_fp8_gemm_roofline_work(kernel: KernelSpec):
    """Expose the operand account actually consumed by the FP8 baseline.

    This is the existing template's FP32-scale storage account. It does not
    assert that a v0.30 ue8m0 packed-scale provider was selected at runtime.
    The BF16-to-FP8 producer is outside the matmul operand payloads. The
    TRTLLM call contains that producer, so its read/write roles are added
    to the whole-call byte ledger. DeepGEMM/einsum receive a separately
    prepared input and must not charge another producer here.
    """
    if kernel.launch.algorithm not in BLOCK_FP8_GEMM_ALGORITHMS:
        raise ValueError(
            f"not a block-FP8 GEMM launch: {kernel.launch.algorithm!r}")
    b, m, n, k = (int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    kblocks = _ceil_div(k, _BLOCK_K)
    payloads, scales = _block_fp8_matmul_payloads(b, m, n, k)
    wrapper = (None if (kernel.launch.algorithm == DEEPGEMM_FP8_EINSUM_BATCHED_ALGORITHM
                       or b != 1 or k % _BLOCK_K or kernel.dtype != "bf16")
               else fp8_linear_wrapper_source_program(m, k, n))
    quantizer = None
    quantizer_bytes = 0
    if kernel.launch.algorithm == TRTLLM_SWAP_AB_ALGORITHM:
        quantizer = {"placement": "inside_trtllm_call", "input_dtype": "bf16",
                     "quantized_dtype": "fp8", "scale_dtype": "fp32", "group_size": _BLOCK_K,
                     "bf16_input_read_bytes": 2 * b * m * k,
                     "fp8_output_write_bytes": b * m * k,
                     "scale_output_write_bytes": 4 * b * m * kblocks,
                     "common_fp_multiply_values": b * m * k,
                     "matmul_reads_quantized_output": True,
                     "source_program": trtllm_scale_1x128_source_program(b * m, k),
                     "arithmetic_and_actual_provider_complete": False}
        quantizer_bytes = (quantizer["bf16_input_read_bytes"] + quantizer["fp8_output_write_bytes"]
                           + quantizer["scale_output_write_bytes"])
    return {"mathematical_flops": 2 * b * m * n * k,
            "operand_payload_bytes": payloads, "scale_bytes": scales,
            "activation_quantizer": quantizer,
            "linear_wrapper_source_program": wrapper,
            "flashinfer_workspace_source_program": (None if wrapper is None
                                                    else trtllm_workspace_source_program(m, k, n)),
            "source_substage_order": (["quantize_bf16_and_write_scales", "packed_fp8_matmul"] if quantizer is not None
                                      else ["packed_fp8_matmul"]),
            "logical_bytes": sum(payloads.values()) + sum(scales.values()) + quantizer_bytes,
            "byte_scope": "matmul FP8 A/B, BF16 C, FP32 template scales and in-call TRTLLM quantizer roles; not measured HBM traffic",
            "mathematical_flops_scope": "matrix product only; quantizer common multiplies separate, scale-branch arithmetic and typed conversion incomplete",
            "vllm_0_30_packed_scale_runtime_bound": False}


def roofline_block_fp8_gemm(kernel: KernelSpec, hardware) -> KernelPrediction:
    """Roofline of one block-FP8 call: FP8 tensor peak vs compulsory DRAM."""

    work = block_fp8_gemm_roofline_work(kernel)
    from src.kernel.memory_access_builder import block_fp8_memory_access_program
    from src.kernel.memory_path_solver import block_fp8_source_demands
    source = block_fp8_source_demands(block_fp8_memory_access_program(kernel, hardware))
    tensor_s = (work["mathematical_flops"]
                / float(hardware.resource_by_role("compute_tensor_fp8")
                        .capacity))
    bandwidth = float(hardware.resource_by_role("dram").capacity)
    matmul_bytes = source["main_read_bytes"] + source["main_write_bytes"]
    matmul_s = max(tensor_s, matmul_bytes / bandwidth)
    quantizer = work["activation_quantizer"]
    quantizer_s = 0.0
    if quantizer is not None:
        quantizer_bytes = sum(quantizer[field] for field in (
            "bf16_input_read_bytes", "fp8_output_write_bytes", "scale_output_write_bytes"))
        quantizer_s = quantizer_bytes / bandwidth
        groups = quantizer["scale_output_write_bytes"] // 4
        # The zero/nonzero amax branch changes forward-scale divisions.
        # Establish a common ideal arithmetic/byte value instead of
        # choosing a data branch or silently overlapping the two kernels.
        for forward_divides in (0, 32 * groups):
            arithmetic_s = (quantizer["common_fp_multiply_values"] + groups + forward_divides) / float(
                hardware.resource_by_role("compute_fma").capacity)
            if arithmetic_s > quantizer_s:
                raise NotImplementedError("internal FP8 quantizer arithmetic is not memory dominated; bind the amax branch")
    return prediction_from_ns((quantizer_s + matmul_s) * 1e9, "roofline")


def is_block_fp8_gemm(kernel: KernelSpec) -> bool:
    return kernel.launch is not None and kernel.launch.algorithm in BLOCK_FP8_GEMM_ALGORITHMS


def block_fp8_source_traffic(kernel: KernelSpec, hardware):
    """Replay source-selected tile payloads/scales in one shared cold L2.

    This is a tile-object LRU approximation, not observed transactions or
    the logical operand sum. Persistent CTA i takes tiles i, i+SMs,...;
    equal K rounds interleave those resident CTAs. No time is an input.
    Dirty outputs retained at return are separate from in-call writeback.
    """
    from src.kernel.cache_traffic_sim import simulate_lru_trace_stateful_dual_accounted
    from src.kernel.memory_access_builder import block_fp8_memory_access_program
    from src.kernel.memory_path_solver import block_fp8_source_demands
    if not is_block_fp8_gemm(kernel):
        raise ValueError("FP8 source traffic requires its selected native family")
    work = block_fp8_gemm_roofline_work(kernel)
    b, m, n, k = block_fp8_source_demands(block_fp8_memory_access_program(kernel, hardware))["geometry"]
    tm, tn, tk = (kernel.launch.tiling.tile_shape[name] for name in ("m", "n", "k"))
    sms = hardware.num_compute_units
    if kernel.launch.grid_size != sms or tk != 128:
        raise ValueError("FP8 source traffic requires the persistent SM grid and K128")
    keys, sizes, kinds = [], [], []
    identities = {}

    def access(identity, size, write=False):
        if size <= 0:
            raise ValueError("source traffic object must be nonempty")
        if identity not in identities:
            identities[identity] = len(identities)
        keys.append(identities[identity])
        sizes.append(size)
        kinds.append(int(write))

    mt, nt, kt = _ceil_div(m, tm), _ceil_div(n, tn), _ceil_div(k, tk)
    if work["activation_quantizer"] is not None:
        for batch in range(b):
            for mi in range(mt):
                rows = min(tm, m-mi*tm)
                for ki in range(kt):
                    width = min(tk, k-ki*tk)
                    access(("input", batch, mi, ki), 2*rows*width)
                    access(("A", batch, mi, ki), rows*width, True)
                    access(("As", batch, mi, ki), 4*rows, True)
    total_tiles = b*mt*nt
    for wave in range(_ceil_div(total_tiles, sms)):
        tiles = []
        for sm in range(sms):
            ordinal = wave*sms+sm
            if ordinal >= total_tiles:
                continue
            batch, local = divmod(ordinal, mt*nt)
            mi, ni = divmod(local, nt)
            tiles.append((batch, mi, ni))
        for ki in range(kt):
            width = min(tk, k-ki*tk)
            for batch, mi, ni in tiles:
                rows, columns = min(tm, m-mi*tm), min(tn, n-ni*tn)
                access(("A", batch, mi, ki), rows*width)
                access(("As", batch, mi, ki), 4*rows)
                access(("B", batch, ni, ki), columns*width)
                for scale_group in range(ni*tn//128, _ceil_div(ni*tn+columns, 128)):
                    access(("Bs", batch, scale_group, ki), 4)
        for batch, mi, ni in tiles:
            access(("C", batch, mi, ni), 2*min(tm, m-mi*tm)*min(tn, n-ni*tn), True)
    reads = [kind == 0 for kind in kinds]
    writes = [kind == 1 for kind in kinds]
    capacity = hardware.resource_by_role("l2_cache").size_bytes
    read_requested, read_missed, write_requested, _, dirty_evicted, dirty_retained, _ = (
        simulate_lru_trace_stateful_dual_accounted(keys, sizes, kinds, reads, writes, int(capacity), 0))
    return {"l2_read_bytes":read_requested, "l2_write_bytes":write_requested,
        "hbm_read_bytes":read_missed, "hbm_write_bytes":dirty_evicted,
        "deferred_dirty_writeback_bytes":dirty_retained,
        "cache_capacity_bytes":capacity, "source_accesses":len(keys),
        "basis":"source-selected persistent CTA tile order, FP8/BF16/FP32 payload/scale ABI and shared tile-object LRU",
        "limitations":("standalone_cold_L2_entry_not_cross_call_cache_state",
            "equal_K_round_interleave_and_raster_order_are_analytical_schedule",
            "payload_byte_granularity_no_sector_overfetch",
            "single_call_conditional_component_not_complete_window_HBM_traffic",
            "deferred_dirty_bytes_not_charged_as_in_call_HBM_writes")}
