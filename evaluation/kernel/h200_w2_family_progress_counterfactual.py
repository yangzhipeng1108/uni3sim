"""Evaluate two physical overlap scenarios for one H200 W2 source family."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from evaluation.kernel.dataset import get_hardware_config_path, load_kernel_dataset
from evaluation.kernel.evaluation_runner import _get_hardware
from evaluation.kernel.observation_normalize_moe import _normalize_fused_moe
from src.kernel.kernel_base import predict


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "kernel_dataset/nvidia_h200/fused_moe.json.zst"
BASELINE = (ROOT / "experiment_logs/h200_accuracy_20261028/"
            "kernel_baseline_semantic.json")
PROBE = (ROOT / "experiment_logs/h200_accuracy_20261028/"
         "w2_topcase_pipeline_counterfactual_20261102.json")
NCU_CAMPAIGN = (ROOT / "profiler_output/kernel_dataset/nvidia_h200/"
                "ncu_1027_20261031")
OUTPUT = (ROOT / "experiment_logs/h200_accuracy_20261028/"
          "w2_family_progress_counterfactual_20261102.json")
REFERENCE_ACTIVE_CTAS = 33784


def _pin(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def _summary(rows: list[dict], field: str) -> dict:
    absolute = sum(abs(row[field] - row["actual_ns"]) for row in rows)
    actual = sum(row["actual_ns"] for row in rows)
    return {"absolute_error_ms": absolute / 1e6,
            "wmae_pct": absolute / actual * 100,
            "underpredicted_count": sum(row[field] < row["actual_ns"]
                                        for row in rows),
            "overpredicted_count": sum(row[field] > row["actual_ns"]
                                       for row in rows)}


def main() -> None:
    probe = json.loads(PROBE.read_text())
    if (probe["status"] != "evaluation_only_source_equivalence_unproven"
            or probe["forward_changed"] is not False
            or probe["case_id"] !=
            "FUSEDMOE_TOK32768_E128_TOPK16_H2048_INT768"):
        raise ValueError("W2 independent probe reference differs")
    source_event_ms = probe["independent_probe"][
        "conditional_twelve_round_event_ms"]
    dataset = load_kernel_dataset("nvidia_h200", ["fused_moe"])
    hardware = _get_hardware(get_hardware_config_path("nvidia_h200"))
    predictions = []
    for case_id, entry in dataset.items():
        attr = entry["op_info"]["Attr"]
        tile = attr["moe_tiling"]["w2"]
        if ((attr["E"], attr["H"], attr["INT"]) != (128, 2048, 768)
                or (tile["BM"], tile["BN"], tile["BK"], tile["GSM"])
                != (128, 256, 64, 16)):
            continue
        tensor_indices = [index for index, kernel in enumerate(entry["kernels"])
                          if kernel["Metadata"]["Kernel Name"] == "fused_moe_kernel"]
        if len(tensor_indices) != 2:
            raise ValueError(f"W2 family tensor pair differs: {case_id}")
        index = tensor_indices[1]
        spec = _normalize_fused_moe(attr, entry["kernels"][index],
                                    "nvidia_h200", hardware, 1, index, None)
        if (spec.variant != "w2" or spec.launch.block != (256, 1, 1)
                or spec.launch.tiling.tile_shape != {"m": 128, "n": 256, "k": 64}
                or spec.launch.parameters["group_size_m"] != 16):
            raise ValueError(f"W2 family source launch differs: {case_id}")
        active_ctas = sum(math.ceil(rows / 128)
                          for rows in attr["tok_per_expert"] if rows > 0) * 8
        if active_ctas <= 0 or active_ctas > spec.launch.grid_size:
            raise ValueError(f"W2 active CTA census differs: {case_id}")
        start_ns = time.perf_counter_ns()
        current = predict(spec, hardware, "midlayer")
        wall_ns = time.perf_counter_ns() - start_ns
        # These are two explicit overlap assumptions, fixed without reading
        # the case's production duration. Neither is a proposed Forward rule.
        core_ns = source_event_ms * 1e6 * active_ctas / REFERENCE_ACTIVE_CTAS
        output_ns = current.t_per_group["memory:output_write"] * 1e9
        predictions.append({
            "case_id": case_id, "w2_kernel_index": index,
            "active_ctas": active_ctas,
            "launch_grid_ctas": spec.launch.grid_size,
            "current_predicted_ns": current.t_total_s * 1e9,
            "core_only_conditional_ns": core_ns,
            "core_then_serial_output_conditional_ns": core_ns + output_ns,
            "current_output_path_ns": output_ns,
            "simulator_wall_ns": wall_ns,
        })
    if len(predictions) != 17:
        raise ValueError("W2 exact source family count changed")

    review_pins = {}
    reviewed = {}
    family_ids = {row["case_id"] for row in predictions}
    for path in sorted(NCU_CAMPAIGN.glob("*_review.json")):
        review = json.loads(path.read_text())
        if "statuses" not in review:
            continue
        common = family_ids.intersection(review["statuses"])
        if common:
            review_pins[path.name] = _pin(path)
        for case_id in common:
            if case_id in reviewed:
                raise ValueError(f"W2 NCU case reviewed twice: {case_id}")
            reviewed[case_id] = review["statuses"][case_id]["status"]
    for row in predictions:
        case_id = row["case_id"]
        row["ncu_candidate_status"] = (
            reviewed[case_id] if case_id in reviewed
            else "not_in_1027_capture")

    baseline = json.loads(BASELINE.read_text())
    if baseline["hardware_id"] != "nvidia_h200":
        raise ValueError("W2 family baseline hardware differs")
    by_id = {case["case_id"]: case for case in baseline["cases"]}
    for row in predictions:
        recorded = by_id[row["case_id"]]["kernel_results"][row["w2_kernel_index"]]
        if abs(recorded["pred_forward_ns"] - row["current_predicted_ns"]) > 1e-3:
            raise ValueError(f"W2 family baseline changed: {row['case_id']}")
        row["actual_ns"] = recorded["actual_ns"]
    ncu_complete = [row for row in predictions
                    if row["ncu_candidate_status"] == "complete"]
    if len(ncu_complete) != 10:
        raise ValueError("W2 family complete NCU candidate coverage changed")
    result = {
        "kind": "h200_w2_exact_source_family_progress_counterfactual",
        "status": "evaluation_only_overlap_unresolved",
        "inputs": {"dataset": _pin(DATASET), "baseline": _pin(BASELINE),
                   "independent_probe": _pin(PROBE),
                   "ncu_candidate_reviews": review_pins},
        "source_domain": {
            "hardware_id": "nvidia_h200", "expert_count": 128,
            "projection_k": 768, "projection_n": 2048,
            "tile_m_n_k": [128, 256, 64], "group_size_m": 16,
            "threads_per_cta": 256,
            "reference_active_ctas": REFERENCE_ACTIVE_CTAS,
        },
        "conditional_rule": {
            "core_only": "probe_12_round_event_ns * active_ctas / 33784",
            "core_then_serial_output": "core_only + current_output_path_ns",
            "probe_12_round_event_ms": source_event_ms,
        },
        "case_count": len(predictions),
        "ncu_complete_candidate_count": len(ncu_complete),
        "rows": predictions,
        "current": _summary(predictions, "current_predicted_ns"),
        "core_only_conditional": _summary(predictions,
                                          "core_only_conditional_ns"),
        "core_then_serial_output_conditional": _summary(
            predictions, "core_then_serial_output_conditional_ns"),
        "ncu_complete_candidate_subset": {
            "case_count": len(ncu_complete),
            "current": _summary(ncu_complete, "current_predicted_ns"),
            "core_only_conditional": _summary(ncu_complete,
                                              "core_only_conditional_ns"),
            "core_then_serial_output_conditional": _summary(
                ncu_complete, "core_then_serial_output_conditional_ns"),
        },
        "simulator_wall_median_ns": statistics.median(
            row["simulator_wall_ns"] for row in predictions),
        "forward_changed": False,
        "limitations": [
            "Both scenarios use an unverified 4-to-12-round extrapolation from an independent partial-K probe.",
            "The source family matches tile, K, N, group size and threads but not the probe's real routing, output, registers or static LDGSTS sites.",
            "Core-only and serial-output are scenarios, not proven bounds; inter-wave output overlap is unmeasured.",
            "Only one probe grid was measured, so active-CTA proportionality across the 17 cases is unverified.",
            "Production duration is read only after both physical scenarios are calculated and only for evaluation.",
            "Ten cases have worker-reviewed complete NCU candidates; seven KC cases are outside the 1027 capture and have no NCU candidate in this comparison.",
        ],
        "reproduce": "python -m evaluation.kernel.h200_w2_family_progress_counterfactual",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
