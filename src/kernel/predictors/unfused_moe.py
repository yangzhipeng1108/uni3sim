"""Prediction for aggregate, explicitly decomposed unfused MoE observations."""

from src.kernel.bank_stream_model import (
    bank_stream_ns, matmul_stream, pointwise_stream, softmax_stream)
from src.kernel.formula import PerformanceFormula
from src.kernel.kernel_base import KernelPredictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.predictors.common import (
    PredictionInput,
    attach_structured_contract,
    build_forward_demand,
    launch_wave_geometry,
    prediction_from_ns,
    roofline_ns,
)


def _model(kernel: KernelSpec, hardware):
    if kernel.variant != "decomposed":
        raise ValueError(
            f"unknown UNFUSED_MOE variant: {kernel.variant!r}")
    memory_model = hardware.kernel_memory_model
    if memory_model not in ("pim_bank", "systolic_scratchpad"):
        raise NotImplementedError(
            "UNFUSED_MOE currently requires a PIM-bank or "
            f"systolic-scratchpad execution model, got {memory_model!r}")

    from src.kernel.memory_access_builder import unfused_moe_memory_access_program
    from src.kernel.memory_path_solver import unfused_moe_primitive_demands
    source = unfused_moe_primitive_demands(
        unfused_moe_memory_access_program(kernel, hardware))
    tokens, experts, hidden, intermediate = source["geometry"]
    counts = source["tokens_per_expert"]
    routed_rows = float(sum(counts))
    hierarchy_bytes = float(source["primitive_requested_bytes"]
        if memory_model == "pim_bank" else source["expert_mm_bytes"])
    router_flops = 2.0 * tokens * hidden * experts if memory_model == "pim_bank" else 0.0
    flops = {
        "compute_tensor": float(
            6.0 * routed_rows * hidden * intermediate + router_flops),
        "compute_fma": 0.0,
        "compute_alu": 0.0,
        "compute_sfu": 0.0,
    }
    demand = build_forward_demand(
        hierarchy_bytes, hierarchy_bytes, False, flops)
    for role in ("xbar", "smem", "l1_cache"):
        demand[role] = 0.0

    execution_floor_ns = None
    if memory_model == "pim_bank":
        # Router, routing softmax, every active expert's W1/activation/W2, and
        # the output reduction run back to back on the bank-attached units.
        stages = [
            matmul_stream(tokens, hidden, experts),
            softmax_stream(experts * tokens),
        ]
        for count in counts:
            if count <= 0:
                continue
            stages.append(matmul_stream(count, hidden, 2 * intermediate))
            stages.append(pointwise_stream(count * intermediate))
            stages.append(matmul_stream(count, intermediate, hidden))
        stages.append(pointwise_stream(routed_rows * hidden))
        execution_floor_ns = bank_stream_ns(tuple(stages), hardware)
        # No cache stands between a bank-attached unit and its array, and the
        # bank stream above already prices every bank access and the single
        # SIMD datapath that consumes it.
        demand["l2_cache"] = 0.0
        demand["dram"] = 0.0
        for role in ("compute_fma", "compute_sfu", "compute_alu"):
            demand[role] = 0.0

    eta, wave_efficiency, active_units = launch_wave_geometry(
        kernel, hardware)
    if memory_model == "pim_bank":
        active_units = hardware.num_compute_units
        # A bank-attached launch has no CTA waves to quantize against: the
        # partitioner spreads the operand over every unit, and the bank
        # stream carries the resulting per-unit column count.
        eta, wave_efficiency = 1.0, 1.0
    elif memory_model == "systolic_scratchpad":
        pass
    else:
        raise AssertionError(
            f"validated UNFUSED_MOE memory model became invalid: {memory_model!r}")

    inp = PredictionInput(
        demand_per_resource=demand,
        eta=eta,
        wave_efficiency=wave_efficiency,
        active_units=active_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        grid_size=kernel.launch.grid_size,
        block_size=kernel.launch.block_size,
        launch_grid=kernel.launch.grid,
        launch_block=kernel.launch.block,
        variant=kernel.variant,
        mem_model=memory_model,
        compute_demands=(),
        memory_paths=(),
        pipeline_plan=None,
        dram_capacity_multiplier=(
            hardware.dram_protocol_efficiency_l1cached),
        support_level="compatibility",
        limitations=("outside_a800_c500_910b_acceptance_scope",
                     "unfused_moe_unresolved_routing_and_pointwise_packing",
                     "unfused_moe_bank_internal_passes_and_systolic_unpriced_glue"),
        execution_time_floor_ns=execution_floor_ns,
    )
    return attach_structured_contract(
        inp,
        hardware,
        transaction_bytes=None,
        path_name=None,
    ), hierarchy_bytes


class UnfusedMoEPredictor(KernelPredictor):
    kind = KernelKind.UNFUSED_MOE

    def resource_input(self, kernel: KernelSpec, hardware):
        return _model(kernel, hardware)

    def midlayer(self, kernel: KernelSpec, hardware):
        inp, _ = self.resource_input(kernel, hardware)
        return PerformanceFormula(hardware).predict_kernel(inp)

    def roofline(self, kernel: KernelSpec, hardware):
        inp, compulsory = self.resource_input(kernel, hardware)
        return prediction_from_ns(
            roofline_ns(inp.demand_per_resource, hardware, compulsory),
            "roofline")
