"""Partition frozen H200 MoE errors by observed source tiling.

Production durations are used only to rank evaluation errors.  The optional
counterfactual is an existing evaluation artifact; no values enter Forward.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

import zstandard


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "kernel_dataset/nvidia_h200/fused_moe.json.zst"
BASELINE = ROOT / "experiment_logs/h200_accuracy_20261028/kernel_baseline_semantic.json"
COUNTERFACTUAL = ROOT / "experiment_logs/h200_accuracy_20261028/moe_bm128_w2_l2_counterfactual_20261102.json"
OUTPUT = ROOT / "experiment_logs/h200_accuracy_20261028/moe_tiling_error_partition_20261102.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(rows: list[dict]) -> dict:
    return {
        "case_count": len(rows),
        "absolute_error_ms": sum(row["absolute_error_ns"] for row in rows) / 1e6,
        "signed_error_ms": sum(row["signed_error_ns"] for row in rows) / 1e6,
        "underpredicted_count": sum(row["signed_error_ns"] < 0 for row in rows),
        "overpredicted_count": sum(row["signed_error_ns"] > 0 for row in rows),
    }


def main() -> None:
    dataset = json.loads(zstandard.ZstdDecompressor().decompress(DATASET.read_bytes()))
    baseline = json.loads(BASELINE.read_bytes())
    if (baseline["hardware_id"] != "nvidia_h200"
            or baseline["case_count"] != len(baseline["cases"])):
        raise ValueError("H200 frozen Kernel baseline identity differs")
    by_id = {case["case_id"]: case for case in baseline["cases"]}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    signatures: dict[str, dict[str, tuple]] = {}
    w2_baseline: dict[str, tuple[float, float]] = {}
    for case_id, entry in dataset.items():
        if case_id not in by_id:
            raise ValueError(f"frozen MoE case is missing from baseline: {case_id}")
        case = by_id[case_id]
        kernels = entry["kernels"]
        actual_kernels = entry["op_info"]["evaluation"]["timing"]["kernels"]
        results = case["kernel_results"]
        if len(kernels) != len(actual_kernels) or len(kernels) != len(results):
            raise ValueError(f"MoE kernel count differs: {case_id}")
        for index, (actual, result) in enumerate(zip(actual_kernels, results)):
            if (actual["kernel_idx"] != index
                    or actual["duration_ns"] != result["actual_ns"]):
                raise ValueError(f"MoE kernel alignment differs: {case_id} {index}")
        tensor_indices = [index for index, kernel in enumerate(kernels)
                          if kernel["Metadata"]["Kernel Name"] == "fused_moe_kernel"]
        if len(tensor_indices) != 2:
            raise ValueError(f"MoE W1/W2 pair differs: {case_id}")
        signatures[case_id] = {}
        for role, index in zip(("w1", "w2"), tensor_indices):
            tiling = entry["op_info"]["Attr"]["moe_tiling"][role]
            signature = (role, tiling["BM"], tiling["BN"], tiling["BK"],
                         tiling["GSM"])
            signatures[case_id][role] = signature
            result = results[index]
            if role == "w2":
                w2_baseline[case_id] = (
                    result["pred_forward_ns"], result["actual_ns"])
            signed = result["pred_forward_ns"] - result["actual_ns"]
            groups[signature].append({
                "case_id": case_id,
                "signed_error_ns": signed,
                "absolute_error_ns": abs(signed),
            })
    counterfactual = json.loads(COUNTERFACTUAL.read_bytes())
    if (counterfactual["kind"] != "h200_moe_w2_l2_traffic_evaluation_counterfactual"
            or counterfactual["matched_case_count"] != len(counterfactual["rows"])):
        raise ValueError("W2 L2 evaluation candidate identity differs")
    cf_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in counterfactual["rows"]:
        case_id = row["case_id"]
        if case_id not in signatures:
            raise ValueError(f"counterfactual case is absent: {case_id}")
        signature = signatures[case_id]["w2"]
        actual = row["actual_ns"]
        old = row["baseline_predicted_ns"]
        if (old, actual) != w2_baseline[case_id]:
            raise ValueError(f"counterfactual baseline W2 differs: {case_id}")
        candidate = row["all_extra_l2_on_activation_predicted_ns"]
        cf_groups[signature].append({
            "case_id": case_id,
            "baseline_signed_error_ns": old - actual,
            "baseline_absolute_error_ns": abs(old - actual),
            "candidate_signed_error_ns": candidate - actual,
            "candidate_absolute_error_ns": abs(candidate - actual),
            "ncu_l2_over_model_l2": row["ncu_l2_read_bytes"] / row["model_l2_read_bytes"],
        })
    output = {
        "kind": "h200_moe_tiling_error_partition",
        "status": "evaluation_only_no_forward_change",
        "input_sha256": {
            "baseline": _sha(BASELINE),
            "dataset": _sha(DATASET),
            "counterfactual": _sha(COUNTERFACTUAL),
        },
        "groups": [
            {"role": key[0], "tile_m": key[1], "tile_n": key[2],
             "tile_k": key[3], "group_size_m": key[4],
             **_summary(rows)}
            for key, rows in sorted(groups.items(), key=lambda item:
                                     -sum(row["absolute_error_ns"] for row in item[1]))
        ],
        "counterfactual_by_source_signature": [
            {"role": key[0], "tile_m": key[1], "tile_n": key[2],
             "tile_k": key[3], "group_size_m": key[4],
             "case_count": len(rows),
             "baseline_absolute_error_ms": sum(
                 row["baseline_absolute_error_ns"] for row in rows) / 1e6,
             "candidate_absolute_error_ms": sum(
                 row["candidate_absolute_error_ns"] for row in rows) / 1e6,
             "candidate_improved_count": sum(
                 row["candidate_absolute_error_ns"]
                 < row["baseline_absolute_error_ns"] for row in rows),
             "median_ncu_l2_over_model_l2": median(
                 row["ncu_l2_over_model_l2"] for row in rows),
             "baseline_underpredicted_count": sum(
                 row["baseline_signed_error_ns"] < 0 for row in rows)}
            for key, rows in sorted(cf_groups.items())
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT),
                      "group_count": len(output["groups"]),
                      "counterfactual_group_count": len(
                          output["counterfactual_by_source_signature"])}))


if __name__ == "__main__":
    main()
