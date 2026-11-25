"""Prediction for decomposed, unfused eager attention."""

from src.kernel.bank_stream_model import (
    bank_stream_ns, matmul_stream, softmax_stream)
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


def _shape(kernel: KernelSpec):
    return tuple(int(kernel.shape[name]) for name in (
        "batch", "q_len", "kv_len", "head_dim",
        "num_query_heads", "num_kv_heads"))


def _model(kernel: KernelSpec, hardware):
    if kernel.variant != "decomposed":
        raise ValueError(
            f"unknown EAGER_ATTN variant: {kernel.variant!r}")
    memory_model = hardware.kernel_memory_model
    if memory_model not in ("pim_bank", "systolic_scratchpad"):
        raise NotImplementedError(
            "EAGER_ATTN currently requires a PIM-bank or systolic-scratchpad "
            f"execution model, got {memory_model!r}")

    from src.kernel.memory_access_builder import eager_attention_memory_access_program
    from src.kernel.memory_path_solver import eager_attention_interface_demands
    program = eager_attention_memory_access_program(kernel, hardware)
    qkt_bytes, softmax_bytes, pv_bytes, geometry = (
        eager_attention_interface_demands(program))
    rows, head_dim, kv_len = geometry
    softmax_elements = 0
    if memory_model == "pim_bank":
        softmax_elements = rows * kv_len
    elif memory_model == "systolic_scratchpad":
        # The actual Timeloop source gives software glue zero cost. Keep the
        # logical dataflow visible without charging an invented engine.
        softmax_bytes = 0
    else:
        raise AssertionError(
            f"validated EAGER_ATTN memory model became invalid: {memory_model!r}")

    hierarchy_bytes = float(qkt_bytes + softmax_bytes + pv_bytes)
    execution_floor_ns = None
    if memory_model == "pim_bank":
        # The three materialized stages run back to back on the bank-attached
        # units: scores, the softmax over them, then the value product.
        execution_floor_ns = bank_stream_ns((
            matmul_stream(rows, head_dim, kv_len),
            softmax_stream(rows * kv_len),
            matmul_stream(rows, kv_len, head_dim),
        ), hardware)
    flops = {
        "compute_tensor": float(
            4 * rows * head_dim * kv_len),
        "compute_fma": float(softmax_elements),
        "compute_alu": 0.0,
        "compute_sfu": float(softmax_elements),
    }
    demand = build_forward_demand(
        hierarchy_bytes, hierarchy_bytes, False, flops)
    for role in ("xbar", "smem", "l1_cache"):
        demand[role] = 0.0
    if memory_model == "pim_bank":
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
            f"validated EAGER_ATTN memory model became invalid: {memory_model!r}")

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
        dram_capacity_multiplier=hardware.dram_protocol_efficiency,
        support_level="compatibility",
        limitations=("outside_a800_c500_910b_acceptance_scope",),
        execution_time_floor_ns=execution_floor_ns,
    )
    return attach_structured_contract(
        inp,
        hardware,
        transaction_bytes=None,
        path_name=None,
    ), hierarchy_bytes


class EagerAttentionPredictor(KernelPredictor):
    kind = KernelKind.EAGER_ATTN

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
