"""Check an independent short-K progress extrapolation against one W2 case."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import time

from evaluation.kernel.dataset import get_hardware_config_path, load_kernel_dataset
from evaluation.kernel.evaluation_runner import _get_hardware
from evaluation.kernel.observation_normalize_moe import _normalize_fused_moe
from src.kernel.kernel_base import predict


ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "FUSEDMOE_TOK32768_E128_TOPK16_H2048_INT768"
DATASET = ROOT / "kernel_dataset/nvidia_h200/fused_moe.json.zst"
PROBE = (ROOT / "experiment_logs/h200_accuracy_20261028/"
         "w2_grouped_timing_20261102.json")
INDEPENDENT_HOLDOUT = (ROOT / "profiler_output/microbench/nvidia_h200/"
                       "routed_panel_wgmma_overlap_20261029/"
                       "static_rounds_attempt/raw.json")
HOLDOUT_SOURCE = (ROOT / "microbench/nvidia_h200/compute/tensor_core/"
                  "routed_panel_wgmma_overlap.cu")
PROGRESS = (ROOT / "experiment_logs/h200_accuracy_20261028/"
            "moe_w2_source_progress_topcase_20261102.json")
COUNTERS = (ROOT / "experiment_logs/h200_accuracy_20261028/"
            "moe_topcase_source_counter_diagnostic_20261102.json")
OUTPUT = (ROOT / "experiment_logs/h200_accuracy_20261028/"
          "w2_topcase_pipeline_counterfactual_20261102.json")


def _pin(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)),
            "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    probe = json.loads(PROBE.read_text())
    if (probe["status"] != "independent_short_k_progress_only"
            or probe["production_duration_consumed"] is not False
            or probe["app_id"] != "capture-execution-3c89989aa6a154f1"):
        raise ValueError("independent W2 probe identity differs")
    four = probe["samples"]["gather_plus_wgmma_4"]["event_median_ms"]
    eight = probe["samples"]["gather_plus_wgmma_8"]["event_median_ms"]
    if not 0 < four < eight:
        raise ValueError("independent W2 4/8 event order differs")
    # Candidate is frozen before the production observation is read.  It is
    # only a same-grid, source-swizzled, short-K mechanism extrapolation.
    candidate_12_ms = eight + (eight - four)
    holdout = json.loads(INDEPENDENT_HOLDOUT.read_text())
    if (holdout["status"] != "complete"
            or holdout["source_sha256"] != _pin(HOLDOUT_SOURCE)["sha256"]
            or holdout["measurement_boundary"] !=
            "partial_k_synthetic_gather_cp_async_wgmma_no_moe_output"):
        raise ValueError("independent WGMMA round holdout identity differs")
    holdout_rows = [json.loads(line) for line in
                    holdout["execution"]["stdout"].splitlines()]
    holdout_combined = {row["rounds"]: row for row in holdout_rows[1:]
                        if row["mode"] == "gather_plus_wgmma"}
    if (set(holdout_combined) != {4, 8, 12}
            or any(row["grid"] != 2112 for row in holdout_combined.values())):
        raise ValueError("independent WGMMA round holdout coverage differs")
    holdout_events = {rounds: statistics.median(row["samples_ms"])
                      for rounds, row in holdout_combined.items()}
    holdout_prediction_12 = 2 * holdout_events[8] - holdout_events[4]

    dataset = load_kernel_dataset("nvidia_h200", ["fused_moe"])
    entry = dataset[CASE_ID]
    hardware = _get_hardware(get_hardware_config_path("nvidia_h200"))
    tensor_indices = [index for index, kernel in enumerate(entry["kernels"])
                      if kernel["Metadata"]["Kernel Name"] == "fused_moe_kernel"]
    if len(tensor_indices) != 2:
        raise ValueError("W2 case does not contain exactly two tensor launches")
    w2_index = tensor_indices[1]
    spec = _normalize_fused_moe(
        entry["op_info"]["Attr"], entry["kernels"][w2_index],
        "nvidia_h200", hardware, 1, w2_index, None)
    if (spec.variant != "w2" or spec.launch.grid != (33784, 1, 1)
            or spec.launch.block != (256, 1, 1)
            or tuple(spec.launch.tiling.tile_shape[name]
                     for name in ("m", "n", "k")) != (128, 256, 64)
            or spec.launch.parameters["group_size_m"] != 16):
        raise ValueError("W2 production launch differs from probe comparison")
    wall_samples_ns = []
    predictions = []
    for _ in range(5):
        start_ns = time.perf_counter_ns()
        predictions.append(predict(spec, hardware, "midlayer"))
        wall_samples_ns.append(time.perf_counter_ns() - start_ns)
    current = predictions[0]
    if any(item.t_total_s != current.t_total_s for item in predictions[1:]):
        raise ValueError("W2 Forward prediction changed between identical calls")

    progress = json.loads(PROGRESS.read_text())
    counters = json.loads(COUNTERS.read_text())
    actual_ns = progress["evaluation_only"]["old_actual_ns"]
    baseline_ns = progress["evaluation_only"]["old_prediction_ns"]
    if (progress["case_id"] != CASE_ID or counters["case_id"] != CASE_ID
            or abs(current.t_total_s * 1e9 - baseline_ns) > 1e-3
            or counters["quality_accepted"] is not False
            or counters["production_duration_consumed"] is not False):
        raise ValueError("W2 production baseline or raw-only counter differs")
    t = counters["traffic"]
    source_panel_bytes = progress["w2_program"][
        "unmasked_a_plus_b_bytes_over_all_ctas"]
    result = {
        "kind": "h200_w2_topcase_independent_progress_counterfactual",
        "status": "evaluation_only_source_equivalence_unproven",
        "case_id": CASE_ID,
        "inputs": {"dataset": _pin(DATASET), "probe": _pin(PROBE),
                   "independent_holdout": _pin(INDEPENDENT_HOLDOUT),
                   "progress": _pin(PROGRESS), "counters": _pin(COUNTERS)},
        "source_shape": {"grid": list(spec.launch.grid),
                         "block": list(spec.launch.block),
                         "tile": [128, 256, 64], "group_size_m": 16,
                         "k_rounds": progress["w2_program"]["k_loop_iterations"]},
        "independent_probe": {
            "four_round_event_median_ms": four,
            "eight_round_event_median_ms": eight,
            "incremental_four_round_ms": eight - four,
            "conditional_twelve_round_event_ms": candidate_12_ms,
            "extrapolation": "t12=t8+(t8-t4); fixed grid and linear rounds assumed",
        },
        "other_grid_independent_holdout": {
            "grid_ctas": 2112,
            "event_medians_ms": {str(key): value for key, value in
                                 holdout_events.items()},
            "from_four_eight_predicted_twelve_ms": holdout_prediction_12,
            "observed_twelve_ms": holdout_events[12],
            "extrapolation_ape_pct": abs(
                holdout_prediction_12 - holdout_events[12])
                / holdout_events[12] * 100,
            "scope": "independent smaller-footprint probe; no production duration",
        },
        "current_forward": {"predicted_ns": current.t_total_s * 1e9,
                            "simulator_wall_median_ns":
                                statistics.median(wall_samples_ns),
                            "simulator_wall_samples_ns": wall_samples_ns,
                            "bottleneck": current.bottleneck_group,
                            "l2_ms": current.t_per_group["l2"] * 1e3,
                            "tensor_ms": current.t_per_group["tensor_pipe"] * 1e3,
                            "dram_ms": current.t_per_group["dram"] * 1e3},
        "non_duration_traffic": {
            "production_l2_read_bytes": t["new_l2_read_bytes"],
            "production_dram_read_bytes": t["new_dram_read_bytes"],
            "source_unmasked_a_plus_b_panel_bytes": source_panel_bytes,
            "l2_read_over_source_panel": t["new_l2_read_bytes"] / source_panel_bytes,
            "dram_read_over_unique_backing":
                t["new_dram_read_bytes"] / (524288 * 768 * 2
                                             + 128 * 2048 * 768 * 2)},
        "evaluation_only": {
            "actual_ns": actual_ns,
            "baseline_ape_pct": abs(baseline_ns - actual_ns) / actual_ns * 100,
            "conditional_extrapolation_ape_pct":
                abs(candidate_12_ms * 1e6 - actual_ns) / actual_ns * 100,
            "conditional_extrapolation_signed_error_ns":
                candidate_12_ms * 1e6 - actual_ns,
        },
        "forward_changed": False,
        "limitations": [
            "The extrapolation assumes a stationary 4-to-12-round cadence; only four and eight rounds were measured.",
            "The independent 12-round holdout has a smaller grid and a different memory footprint; its linearity cannot prove the large-backplane probe's missing 12-round point.",
            "The probe omits routing and output and uses synthetic row IDs.",
            "Production/probe static LDGSTS sites and registers differ (48/214 versus 28/194).",
            "The independent event pass and production NSYS/NCU runs have unbound clock and cache equivalence.",
            "A one-case error comparison cannot select a production parameter or prove a cross-shape service.",
        ],
        "reproduce": "python -m evaluation.kernel.h200_w2_topcase_pipeline_counterfactual",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
