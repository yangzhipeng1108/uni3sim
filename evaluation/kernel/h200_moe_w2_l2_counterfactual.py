"""Test W2 L2 read replacement in evaluation without changing Forward."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

import zstandard

from evaluation.kernel.dataset import get_hardware_config_path
from evaluation.kernel.evaluation_runner import _get_hardware
from evaluation.kernel.h200_moe_w1_traffic_counterfactual import (
    _replace_one_l2_path, _summary,
)
from evaluation.kernel.observation_normalizer import normalize_kernel_spec
from src.kernel.formula import PerformanceFormula
from src.kernel.predictors.fused_moe import FusedMoEPredictor


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiment_logs/h200_accuracy_20261028/moe_bm128_w2_l2_source_diagnostic_20261102.json"
DATASET = ROOT / "kernel_dataset/nvidia_h200/fused_moe.json.zst"
BASELINE = ROOT / "experiment_logs/h200_accuracy_20261028/kernel_baseline_semantic.json"
OUTPUT = ROOT / "experiment_logs/h200_accuracy_20261028/moe_bm128_w2_l2_counterfactual_20261102.json"


def _pinned(path: Path, pin: dict) -> None:
    if (str(path.relative_to(ROOT)) != pin["path"]
            or path.stat().st_size != pin["size"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]):
        raise ValueError(f"W2 source diagnostic evidence differs: {path}")


def _ncu_w2(path: Path, kernel: dict, kernel_index: int,
            kernel_count: int) -> int:
    lines = path.read_text().splitlines()
    header = next(index for index, line in enumerate(lines)
                  if line.startswith('"ID"'))
    rows = [row for row in csv.DictReader(lines[header:])
            if row["ID"].isdecimal()]
    tensor_indices = [index for index, item in enumerate(rows)
                      if item["Kernel Name"] == "fused_moe_kernel"]
    if len(rows) != kernel_count or len(tensor_indices) != 2:
        raise ValueError(f"W2 NCU/canonical kernel count differs: {path}")
    if tensor_indices[1] != kernel_index:
        raise ValueError(f"W2 NCU/canonical order differs: {path}")
    row = rows[kernel_index]
    metadata = kernel["Metadata"]
    if (row["Kernel Name"] != metadata["Kernel Name"]
            or row["Grid Size"] != metadata["Grid Size"]
            or row["Block Size"] != metadata["Block Size"]):
        raise ValueError(f"W2 NCU/canonical launch differs: {path}")
    return int(row["lts__t_sectors_op_read.sum"]) * 32


def main() -> None:
    source_raw = SOURCE.read_bytes()
    source = json.loads(source_raw)
    if (source["kind"] != "h200_moe_bm128_w2_l2_source_diagnostic"
            or source["status"] != "evaluation_only_no_forward_change"):
        raise ValueError("W2 source diagnostic identity changed")
    dataset_raw = DATASET.read_bytes()
    dataset = json.loads(zstandard.ZstdDecompressor().decompress(dataset_raw))
    baseline_raw = BASELINE.read_bytes()
    baseline = {case["case_id"]: case for case in json.loads(baseline_raw)["cases"]}
    hardware = _get_hardware(get_hardware_config_path("nvidia_h200"))
    predictor = FusedMoEPredictor()
    formula = PerformanceFormula(hardware)
    rows = []
    unmatched = []
    for source_row in source["rows"]:
        case_id = source_row["case_id"]
        if case_id not in dataset or case_id not in baseline:
            unmatched.append(case_id)
            continue
        for name in ("review_pin", "validation_pin", "launch_pin", "ncu_csv_pin"):
            pin = source_row[name]
            _pinned(ROOT / pin["path"], pin)
        entry = dataset[case_id]
        tensor_indices = [index for index, item in enumerate(entry["kernels"])
                          if item["Metadata"]["Kernel Name"] == "fused_moe_kernel"]
        if len(tensor_indices) != 2:
            raise ValueError(f"W2 canonical kernel pair differs: {case_id}")
        index = tensor_indices[1]
        ncu_l2 = _ncu_w2(ROOT / source_row["ncu_csv_pin"]["path"],
                         entry["kernels"][index], index,
                         len(entry["kernels"]))
        if ncu_l2 != source_row["ncu_l2_read_bytes"]:
            raise ValueError(f"W2 source diagnostic NCU count differs: {case_id}")
        spec = normalize_kernel_spec(
            entry["op_info"], entry["kernels"][index], "nvidia_h200",
            hardware, tensor_ordinal=1, kernel_ordinal=index)
        started = time.perf_counter_ns()
        inp, _ = predictor.resource_input(spec, hardware)
        original = formula.predict_kernel(inp)
        frozen = baseline[case_id]["kernel_results"][index]
        if abs(original.t_total_s * 1e9 - frozen["pred_forward_ns"]) > 1e-5:
            raise ValueError(f"W2 frozen Forward prediction differs: {case_id}")
        model_l2 = sum(node.bytes for path in inp.memory_paths
                       if path.access_kind == "read" for node in path.nodes
                       if node.resource_role == "l2_cache")
        delta = ncu_l2 - model_l2
        if delta <= 0:
            raise ValueError(f"W2 counterfactual requires extra L2 read: {case_id}")
        activation = formula.predict_kernel(_replace_one_l2_path(
            inp, "activation_read", delta))
        weight = formula.predict_kernel(_replace_one_l2_path(
            inp, "weight_read", delta))
        rows.append({
            "case_id": case_id, "actual_ns": frozen["actual_ns"],
            "baseline_predicted_ns": original.t_total_s * 1e9,
            "all_extra_l2_on_activation_predicted_ns": activation.t_total_s * 1e9,
            "all_extra_l2_on_weight_predicted_ns": weight.t_total_s * 1e9,
            "model_l2_read_bytes": model_l2, "ncu_l2_read_bytes": ncu_l2,
            "baseline_bottleneck": original.bottleneck_group,
            "activation_counterfactual_bottleneck": activation.bottleneck_group,
            "weight_counterfactual_bottleneck": weight.bottleneck_group,
            "diagnostic_simulator_wall_ns": time.perf_counter_ns() - started,
        })
    if len(rows) + len(unmatched) != source["qualified_case_count"]:
        raise ValueError("W2 source/counterfactual case count differs")
    result = {
        "kind": "h200_moe_w2_l2_traffic_evaluation_counterfactual",
        "status": "evaluation_only_no_forward_change",
        "source_sha256": hashlib.sha256(source_raw).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset_raw).hexdigest(),
        "baseline_sha256": hashlib.sha256(baseline_raw).hexdigest(),
        "matched_case_count": len(rows), "unmatched_case_ids": unmatched,
        "baseline": _summary(rows, "baseline_predicted_ns"),
        "all_extra_l2_on_activation": _summary(
            rows, "all_extra_l2_on_activation_predicted_ns"),
        "all_extra_l2_on_weight": _summary(
            rows, "all_extra_l2_on_weight_predicted_ns"),
        "rows": rows,
        "limits": [
            "Only cases in the frozen local Kernel baseline are evaluated.",
            "NCU total L2 read does not identify activation or weight transactions.",
            "NCU duration is unused; production duration evaluates the counterfactual only.",
            "No NCU value enters Forward, HWSpec, launch selection, or canonical data.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "matched": len(rows),
                      "unmatched": unmatched, "baseline": result["baseline"],
                      "all_extra_l2_on_activation": result["all_extra_l2_on_activation"],
                      "all_extra_l2_on_weight": result["all_extra_l2_on_weight"]}))


if __name__ == "__main__":
    main()
