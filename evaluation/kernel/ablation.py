"""
Ablation study: incremental technique breakdown for PPT (Forward only).

Naive Roofline:
  t = max(FLOPs/peak_compute, compulsory_bytes/peak_DRAM_BW)
  Compulsory = each matrix element accessed once (infinite cache)
  Single-pipe, no concurrency, no launch

+ Structured Capacity + Launch (no L2 cache):
  Same compute/memory contracts and canonical formula, unit compute efficiency,
  no tile pipeline, and DRAM traffic equal to L2 ingress traffic.

+ L2 Cache Model:
  dram = forward.dram (LRU simulation separates L2 hits from DRAM misses)
  Otherwise identical to the no-L2-cache predictor.

+ Worker/Tile Pipeline:
  Restore production compute efficiencies and the explicit worker-aware
  Load/Compute/Store plan.  Tail waves are charged only by the makespan.

Full Forward (Access-Class Capacity):
  Restore the independently evidenced DRAM access-class capacity multiplier.
  There is no beta, legacy DRAM-concurrency scale, serial-L2 correction, or
  pipeline floor/exposed term in any ablation.
"""
import dataclasses
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from evaluation.kernel.dataset import (
    get_hardware_config_path,
    load_kernel_dataset,
    parse_kernel_evaluation_timing,
    require_canonical_hardware_id,
)
from src.kernel.hardware_spec import HWSpec
from src.kernel.formula import PerformanceFormula
from src.kernel.kernel_base import create_predictor, predict
from src.kernel.predictors.common import replace_structured_demands
from evaluation.kernel.observation_normalizer import (
    normalize_kernel_spec,
    observed_flash_attention_splits,
    observed_gemm_split_k,
    observed_gemm_split_k_partial_bytes,
    observed_gemm_split_k_partial_dram_bytes,
    validate_kernel_observation_sequence,
)
from evaluation.kernel.profiler_observation import extract_ncu_metrics
from evaluation.kernel.observation_parser import is_non_tensor


def _ablation_input(
        inp_fwd, *, demand_replacements, unit_compute_efficiency,
        include_pipeline, include_access_class_capacity):
    if inp_fwd.mem_model != "gpu_mshr":
        raise NotImplementedError(
            "kernel ablation currently supports only gpu_mshr inputs, got "
            f"{inp_fwd.mem_model!r}")
    normalized = replace_structured_demands(
        inp_fwd, demand_replacements)
    if unit_compute_efficiency:
        compute_demands = tuple(
            dataclasses.replace(
                demand,
                shape_efficiency=1.0,
                schedule_efficiency=1.0,
            )
            for demand in normalized.compute_demands)
    else:
        compute_demands = normalized.compute_demands
    return dataclasses.replace(
        normalized,
        compute_demands=compute_demands,
        pipeline_plan=(
            normalized.pipeline_plan if include_pipeline else None),
        dram_capacity_multiplier=(
            normalized.dram_capacity_multiplier
            if include_access_class_capacity else 1.0),
    )


def predict_multiresource_no_cache(inp_fwd, hw):
    """Multi-Resource + Launch (no L2 cache).

    dram = l2_total (assume L2 hit rate = 0).
    This shows the formula structure contribution without cache modeling.
    """
    # Override: dram = l2_total (all L2 traffic spills to DRAM)
    inp = _ablation_input(
        inp_fwd,
        demand_replacements={
            "dram": inp_fwd.demand_per_resource["l2_cache"]},
        unit_compute_efficiency=True,
        include_pipeline=False,
        include_access_class_capacity=False,
    )
    formula = PerformanceFormula(hw)
    return formula.predict_kernel(inp).t_total_ns


def predict_with_l2_cache(inp_fwd, hw):
    """+ L2 Cache Model.

    dram = forward.dram (LRU simulation separates L2/DRAM traffic).
    Otherwise identical to predict_multiresource_no_cache (eta=1.0, no pipeline, no BW efficiency).
    """
    inp = _ablation_input(
        inp_fwd,
        demand_replacements={},
        unit_compute_efficiency=True,
        include_pipeline=False,
        include_access_class_capacity=False,
    )
    formula = PerformanceFormula(hw)
    return formula.predict_kernel(inp).t_total_ns


def predict_with_wave_and_serial(inp_fwd, hw):
    """Restore worker/tile makespan without access-class capacity scaling."""
    inp = _ablation_input(
        inp_fwd,
        demand_replacements={},
        unit_compute_efficiency=False,
        include_pipeline=True,
        include_access_class_capacity=False,
    )
    formula = PerformanceFormula(hw)
    return formula.predict_kernel(inp).t_total_ns


def predict_full_forward(inp_fwd, ncu, hw):
    """Full forward model."""
    formula = PerformanceFormula(hw)
    return formula.predict_kernel(inp_fwd).t_total_ns


def compute_mape(pairs):
    if not pairs:
        return None
    preds = np.array([p for p, _ in pairs])
    actuals = np.array([a for _, a in pairs])
    errs = np.abs(preds - actuals) / actuals * 100
    return errs.mean()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-id", required=True)
    args = parser.parse_args()

    hardware_id = require_canonical_hardware_id(args.hardware_id)
    raw_data = load_kernel_dataset(hardware_id)
    hw = HWSpec(get_hardware_config_path(hardware_id))

    technique_names = [
        "Naive Roofline",
        "+ Multi-Resource (no L2)",
        "+ L2 Cache Model",
        "+ Wave Quant + Pipeline",
        "+ BW Efficiency (Full)",
    ]

    op_types = ["GEMM", "FUSEDMOE", "FLASHATTN"]
    op_data = {}

    items = sorted(raw_data.items())
    for task_key, entry in items:
        if not isinstance(entry, dict):
            raise TypeError(f"dataset entry {task_key!r} must be a mapping")
        oi = entry["op_info"]
        op_type = oi["Type"]
        if op_type not in op_types:
            continue

        attr = oi["Attr"]
        kernels_raw = entry["kernels"]
        if isinstance(kernels_raw, dict):
            kernel_list = list(kernels_raw.values())
        elif isinstance(kernels_raw, list):
            kernel_list = kernels_raw
        else:
            raise TypeError(
                f"dataset entry {task_key!r} kernels must be a mapping or list")
        if not kernel_list:
            raise ValueError(f"dataset entry {task_key!r} has no kernels")
        validate_kernel_observation_sequence(
            kernel_list, hw.hardware_id, op_info=oi, hardware=hw)
        timing_observation = parse_kernel_evaluation_timing(
            entry, kernel_list, hw.hardware_id)

        production_attr = attr
        executed_gemm_split_k = None
        executed_gemm_partial_bytes = None
        executed_gemm_partial_dram_bytes = None
        if op_type == "FLASHATTN" and hw.hardware_id != "ascend_910b":
            production_attr = dict(attr)
            production_attr["SP"] = observed_flash_attention_splits(
                attr, kernel_list, hw.hardware_id)
        elif op_type == "GEMM":
            executed_gemm_split_k = observed_gemm_split_k(kernel_list, attr)
            executed_gemm_partial_bytes = (
                observed_gemm_split_k_partial_bytes(kernel_list, attr))
            executed_gemm_partial_dram_bytes = (
                observed_gemm_split_k_partial_dram_bytes(kernel_list, attr))

        technique_preds = {i: [] for i in range(5)}
        tensor_ordinal = 0
        moe_reduce_ordinal = 0

        for kernel_idx, kdata in enumerate(kernel_list):
            if not isinstance(kdata, dict):
                raise TypeError(
                    f"dataset entry {task_key!r} contains a non-mapping kernel")
            ncu = extract_ncu_metrics(kdata)
            ordinal = tensor_ordinal
            if (op_type == "FUSEDMOE" and not is_non_tensor(
                    op_type, ncu["kernel_name"],
                    ascend_op_type=ncu["op_type"])):
                tensor_ordinal += 1
            current_moe_reduce_ordinal = None
            if (op_type == "FUSEDMOE"
                    and hw.hardware_id == "nvidia_a800"
                    and "reduce_kernel" in ncu["kernel_name"].lower()):
                current_moe_reduce_ordinal = moe_reduce_ordinal
                moe_reduce_ordinal += 1
            kernel = normalize_kernel_spec(
                {"Type": op_type, "Attr": production_attr},
                kdata,
                hw.hardware_id,
                hw,
                tensor_ordinal=ordinal,
                kernel_ordinal=kernel_idx,
                moe_reduce_ordinal=current_moe_reduce_ordinal,
                gemm_split_k=executed_gemm_split_k,
                gemm_split_k_partial_bytes=executed_gemm_partial_bytes,
                gemm_split_k_partial_dram_bytes=(
                    executed_gemm_partial_dram_bytes),
                profiler_metrics=ncu,
            )
            inp_fwd, _ = create_predictor(
                kernel.kind).resource_input(kernel, hw)

            technique_preds[0].append(
                predict(kernel, hw, method="roofline").t_total_ns)
            technique_preds[1].append(
                predict_multiresource_no_cache(inp_fwd, hw))
            technique_preds[2].append(
                predict_with_l2_cache(inp_fwd, hw))
            technique_preds[3].append(
                predict_with_wave_and_serial(inp_fwd, hw))
            technique_preds[4].append(
                predict_full_forward(inp_fwd, ncu, hw))
        if not technique_preds[0]:
            raise ValueError(f"{task_key!r} produced no ablation predictions")
        op_data[task_key] = {
            "op_type": op_type,
            "technique_sums": {i: sum(technique_preds[i]) for i in range(5)},
            "actual_sum": timing_observation.total_duration_ns,
        }

    results = {}
    for technique_idx in range(5):
        results[technique_idx] = {}
        for ot in op_types:
            pairs = []
            for tk, od in op_data.items():
                if od["op_type"] == ot:
                    pairs.append((od["technique_sums"][technique_idx], od["actual_sum"]))
            results[technique_idx][ot] = compute_mape(pairs)

        all_pairs = []
        for tk, od in op_data.items():
            all_pairs.append((od["technique_sums"][technique_idx], od["actual_sum"]))
        results[technique_idx]["Overall"] = compute_mape(all_pairs)

    print()
    print("```")
    print("Forward MAE (%) — Incremental Technique Breakdown")
    print("=" * 80)
    header = f"{'Method':<28} {'GEMM':>8} {'FusedMoE':>10} {'FlashAttn':>10} {'Overall':>8}"
    print(header)
    print("-" * 80)
    for technique_idx, technique_name in enumerate(technique_names):
        g = results[technique_idx]["GEMM"]
        m = results[technique_idx]["FUSEDMOE"]
        f = results[technique_idx]["FLASHATTN"]
        o = results[technique_idx]["Overall"]
        g_s = f"{g:.1f}%" if g is not None else "  N/A "
        m_s = f"{m:.1f}%" if m is not None else "   N/A  "
        f_s = f"{f:.1f}%" if f is not None else "   N/A  "
        o_s = f"{o:.1f}%" if o is not None else "  N/A "
        print(f"{technique_name:<28} {g_s:>8} {m_s:>10} {f_s:>10} {o_s:>8}")
    print("=" * 80)
    print("```")

    for ot in op_types:
        n = sum(1 for od in op_data.values() if od["op_type"] == ot)
        print(f"  {ot}: n={n}")
