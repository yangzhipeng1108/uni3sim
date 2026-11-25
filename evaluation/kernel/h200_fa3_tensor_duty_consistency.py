"""Check a *saved candidate* FA3 tensor-capacity estimate against NCU duty.

Inputs are pinned, local evaluation snapshots.  The FA3 GQA partial predictor
that produced them is not published or committed with this diagnostic.  This
script can replay the arithmetic when those snapshots are available; it
does not rerun the predictor or certify its source equivalence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PARTIAL = (ROOT / "experiment_logs/h200_accuracy_20261028/"
           "fa3_gqa_prefill_partial_6case.json")
TILE_GAP = (ROOT / "experiment_logs/h200_accuracy_20261028/"
            "fa3_gqa_prefill_source_tile_gap_20261102.json")
OUTPUT = (ROOT / "experiment_logs/h200_accuracy_20261028/"
          "fa3_tensor_duty_consistency_20261102.json")


def _pin(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    partial = json.loads(PARTIAL.read_text())
    tile_gap = json.loads(TILE_GAP.read_text())
    if (partial["kind"] != "h200_fa3_gqa_prefill_partial_diagnostic"
            or partial["candidate_scope"] !=
            "five reviewed v0.30 unsplit FA3 main plus one-int32 semaphore Fill; split remains unsupported"
            or len(partial["results"]) != 6
            or tile_gap["kind"] != "h200_fa3_gqa_prefill_source_tile_gap"
            or tile_gap["status"] != "evaluation_only_partial"
            or tile_gap["summary"]["candidate_partial_cases"] != 5
            or tile_gap["summary"]["baseline_supported_cases"] != 0):
        raise ValueError("FA3 candidate snapshot scope changed")
    partial_by_id = {row["case_id"]: row for row in partial["results"]}
    if len(partial_by_id) != 6:
        raise ValueError("FA3 partial snapshot repeats a case")
    rows = []
    for source_row in tile_gap["rows"]:
        case_id = source_row["case_id"]
        if source_row["candidate_status"] == "unsupported":
            if partial_by_id[case_id]["main_candidate_ns"] is not None:
                raise ValueError(f"FA3 split candidate changed: {case_id}")
            continue
        if (source_row["candidate_status"] != "partial"
                or source_row["baseline_status"] != "unsupported"):
            raise ValueError(f"FA3 candidate status changed: {case_id}")
        prior = partial_by_id[case_id]
        if (prior["main_support_level"] != "partial"
                or abs(source_row["op_actual_ns"] - prior["op_actual_ns"]) > 1e-3
                or abs(source_row["op_candidate_ns"] - prior["op_candidate_ns"]) > 1e-3
                or abs(source_row["ncu_tensor_pipe_active_pct"]
                       - prior["ncu"]["tensor_pipe_active_pct"]) > 1e-9):
            raise ValueError(f"FA3 saved prediction or NCU snapshot differs: {case_id}")
        actual_ns = prior["main_actual_ns"]
        predicted_ns = prior["main_candidate_ns"]
        tensor_capacity_ns = source_row["main_tensor_capacity_ns"]
        l2_capacity_ns = source_row["main_l2_tile_read_capacity_ns"]
        if (actual_ns <= 0 or predicted_ns <= 0 or tensor_capacity_ns <= 0
                or l2_capacity_ns <= 0
                or predicted_ns < max(tensor_capacity_ns, l2_capacity_ns)):
            raise ValueError(f"FA3 saved capacity is invalid: {case_id}")
        ncu = prior["ncu"]
        sm_active_fraction = (ncu["sm_active_cycles_avg"]
                              / ncu["sm_elapsed_cycles_avg"])
        tensor_active_fraction = (ncu["tensor_pipe_active_pct"] / 100
                                  * sm_active_fraction)
        if not (0 < tensor_active_fraction <= 1):
            raise ValueError(f"FA3 NCU tensor duty is invalid: {case_id}")
        # Duration is consumed here solely to check the saved candidate's
        # source-tile work/capacity magnitude; it sets no Forward parameter.
        required_tensor_fraction = tensor_capacity_ns / actual_ns
        rows.append({
            "case_id": case_id,
            "main_actual_ns_evaluation_only": actual_ns,
            "main_saved_partial_prediction_ns": predicted_ns,
            "saved_candidate_tensor_capacity_ns": tensor_capacity_ns,
            "saved_candidate_l2_capacity_ns": l2_capacity_ns,
            "ncu_tensor_active_fraction_of_elapsed": tensor_active_fraction,
            "tensor_fraction_required_by_ns_duration_evaluation_only":
                required_tensor_fraction,
            "required_minus_measured_fraction":
                required_tensor_fraction - tensor_active_fraction,
            "ncu_sm_active_fraction_of_elapsed": sm_active_fraction,
            "tensor_capacity_divided_by_ncu_duty_ns_diagnostic_only":
                tensor_capacity_ns / tensor_active_fraction,
            "capacity_over_duty_to_actual_ratio_diagnostic_only":
                tensor_capacity_ns / tensor_active_fraction / actual_ns,
        })
    if len(rows) != 5 or len({row["case_id"] for row in rows}) != 5:
        raise ValueError("FA3 saved partial case coverage changed")
    output = {
        "kind": "h200_fa3_tensor_duty_consistency",
        "status": "saved_unpublished_candidate_evaluation_only",
        "inputs": {"partial": _pin(PARTIAL), "source_tile_gap": _pin(TILE_GAP),
                   "source_audit_sha256_recorded_in_partial":
                       partial["source_audit_sha256"],
                   "ncu_review_sha256_recorded_in_partial":
                       partial["ncu_review_sha256"]},
        "rows": rows,
        "interpretation": (
            "Saved candidate source-tile Tensor capacity is close to the "
            "Tensor activity implied by separate NCU and NSYS observations. "
            "This is a magnitude check, not a model of CTA progress."),
        "limits": [
            "The candidate predictor and launch selector are uncommitted and not published; this script replays saved values only.",
            "The two input snapshots live in ignored experiment_logs and must be retained with their SHA256 pins to replay this commit.",
            "NCU and NSYS are separate invocations; their cycle windows are not paired.",
            "Tensor capacity divided by NCU pipe duty is diagnostic, not a Forward prediction.",
            "No production duration or NCU duty is used to set a model parameter.",
        ],
        "reopen_conditions": [
            "A source-matched vLLM 0.30 D128 GQA mechanism probe must reproduce the one-CTA-per-SM TMA, QK/PV WGMMA, softmax and scheduler-barrier dependency graph without running the complete production operator.",
            "Its compiled launch resources and relevant instruction/traffic counters must be checked against the pinned FA3 source-bound cases before its independent service can enter Forward.",
            "The resulting Forward change must improve same-input Kernel and Serving errors without selecting a rate from production durations or NCU pipe duty.",
        ],
        "reproduce": "python -m evaluation.kernel.h200_fa3_tensor_duty_consistency",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(OUTPUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
