"""Evaluate H200 W1 read traffic against reviewed NCU without changing Forward."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
from pathlib import Path
import statistics
import time

import zstandard

from evaluation.kernel.dataset import get_hardware_config_path
from evaluation.kernel.evaluation_runner import _get_hardware
from evaluation.kernel.observation_normalizer import normalize_kernel_spec
from src.kernel.formula import PerformanceFormula
from src.kernel.predictors.fused_moe import FusedMoEPredictor


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "profiler_output/kernel_dataset/nvidia_h200/ncu_1027_20261031"
DATASET = ROOT / "kernel_dataset/nvidia_h200/fused_moe.json.zst"
BASELINE = ROOT / "experiment_logs/h200_accuracy_20261028/kernel_baseline_semantic.json"
OUTPUT = ROOT / "experiment_logs/h200_accuracy_20261028/moe_w1_traffic_counterfactual_20261102.json"


def _accepted_csvs(dataset: dict) -> dict[str, tuple[Path, dict]]:
    accepted = {}
    for review_path in CAMPAIGN.glob("*_review.json"):
        review = json.loads(review_path.read_text())
        pins = {item["path"]: item for item in review["raw_files"]}
        for case_id, status in review["statuses"].items():
            if status["status"] != "complete" or case_id not in dataset:
                continue
            if case_id in accepted:
                raise ValueError(f"multiple accepted NCU captures: {case_id}")
            suffix = f"/ncu_raw/{case_id}/ncu_stdout.csv"
            matches = [(relative, pin) for relative, pin in pins.items()
                       if relative.startswith("batches/") and relative.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"accepted NCU CSV pin is not unique: {case_id}")
            relative, pin = matches[0]
            path = CAMPAIGN / relative
            raw = path.read_bytes()
            if (len(raw) != pin["size"]
                    or hashlib.sha256(raw).hexdigest() != pin["sha256"]):
                raise ValueError(f"accepted NCU CSV changed: {relative}")
            accepted[case_id] = (path, pin)
    return accepted


def _ncu_w1_counts(path: Path, index: int, entry: dict) -> tuple[int, int]:
    lines = path.read_text().splitlines()
    header = next(i for i, line in enumerate(lines)
                  if line.startswith('"ID","Process ID",'))
    rows = [row for row in csv.DictReader(lines[header:]) if row["ID"].isdecimal()]
    if len(rows) != len(entry["kernels"]):
        raise ValueError(f"NCU/canonical kernel count differs: {path}")
    row = rows[index]
    canonical = entry["kernels"][index]["Metadata"]
    if (int(row["ID"]) != index
            or row["Kernel Name"] != "fused_moe_kernel"
            or canonical["Kernel Name"] != "fused_moe_kernel"
            or row["Grid Size"] != canonical["Grid Size"]
            or row["Block Size"] != canonical["Block Size"]):
        raise ValueError(f"NCU/canonical W1 launch differs: {path}")
    return int(row["lts__t_sectors_op_read.sum"]) * 32, int(row["dram__bytes_read.sum"])


def _replace_one_l2_path(inp, name: str, delta: float):
    paths = []
    found = False
    for path in inp.memory_paths:
        if path.name == name:
            found = True
            nodes = tuple(
                dataclasses.replace(
                    node, bytes=node.bytes + delta,
                    transactions=(node.bytes + delta) / node.transaction_bytes)
                if node.resource_role == "l2_cache" else node
                for node in path.nodes)
            path = dataclasses.replace(
                path, nodes=nodes, initial_transactions=nodes[0].transactions)
        paths.append(path)
    if not found or inp.phase_ledger is None:
        raise ValueError(f"W1 has no phase-bound {name} read path")
    paths = tuple(paths)
    demand = dict(inp.demand_per_resource)
    demand["l2_cache"] += delta
    return dataclasses.replace(
        inp, memory_paths=paths, demand_per_resource=demand,
        phase_ledger=inp.phase_ledger.rescaled(inp.compute_demands, paths))


def _summary(rows: list[dict], prediction: str) -> dict:
    return {
        "case_count": len(rows),
        "mape": statistics.mean(abs(row[prediction] - row["actual_ns"])
                                / row["actual_ns"] for row in rows),
        "wmae": sum(abs(row[prediction] - row["actual_ns"]) for row in rows)
        / sum(row["actual_ns"] for row in rows),
        "absolute_error_sum_ns": sum(
            abs(row[prediction] - row["actual_ns"]) for row in rows),
    }


def main() -> None:
    dataset_raw = DATASET.read_bytes()
    dataset = json.loads(zstandard.ZstdDecompressor().decompress(dataset_raw))
    baseline_raw = BASELINE.read_bytes()
    baseline = json.loads(baseline_raw)
    accepted = _accepted_csvs(dataset)
    hardware = _get_hardware(get_hardware_config_path("nvidia_h200"))
    predictor = FusedMoEPredictor()
    formula = PerformanceFormula(hardware)
    rows = []
    missing_ncu = []
    for case in baseline["cases"]:
        case_id = case["case_id"]
        if case_id not in dataset or dataset[case_id]["op_info"]["Type"] != "FUSEDMOE":
            continue
        entry = dataset[case_id]
        tensor = [i for i, item in enumerate(entry["kernels"])
                  if item["Metadata"]["Kernel Name"] == "fused_moe_kernel"]
        if len(tensor) != 2:
            continue
        index = tensor[0]
        frozen = case["kernel_results"][index]
        if frozen["forward_bottleneck"] != "dram":
            continue
        if case_id not in accepted:
            missing_ncu.append(case_id)
            continue
        path, pin = accepted[case_id]
        ncu_l2, ncu_dram = _ncu_w1_counts(path, index, entry)
        spec = normalize_kernel_spec(
            entry["op_info"], entry["kernels"][index], "nvidia_h200",
            hardware, tensor_ordinal=0, kernel_ordinal=index)
        started = time.perf_counter_ns()
        inp, _ = predictor.resource_input(spec, hardware)
        original = formula.predict_kernel(inp)
        if abs(original.t_total_s * 1e9 - frozen["pred_forward_ns"]) > 1e-5:
            raise ValueError(f"frozen Forward prediction differs: {case_id}")
        model_l2 = sum(node.bytes for memory in inp.memory_paths
                       if memory.access_kind == "read" for node in memory.nodes
                       if node.resource_role == "l2_cache")
        model_dram = sum(node.bytes for memory in inp.memory_paths
                         if memory.access_kind == "read" for node in memory.nodes
                         if node.resource_role == "dram")
        delta = ncu_l2 - model_l2
        if delta <= 0:
            raise ValueError(f"W1 NCU L2 is not above modeled demand: {case_id}")
        activation = formula.predict_kernel(_replace_one_l2_path(
            inp, "activation_read", delta))
        weight = formula.predict_kernel(_replace_one_l2_path(
            inp, "weight_read", delta))
        rows.append({
            "case_id": case_id,
            "actual_ns": frozen["actual_ns"],
            "baseline_predicted_ns": original.t_total_s * 1e9,
            "all_extra_l2_on_activation_predicted_ns": activation.t_total_s * 1e9,
            "all_extra_l2_on_weight_predicted_ns": weight.t_total_s * 1e9,
            "model_l2_read_bytes": model_l2,
            "ncu_l2_read_bytes": ncu_l2,
            "model_dram_read_bytes": model_dram,
            "ncu_dram_read_bytes": ncu_dram,
            "baseline_bottleneck": original.bottleneck_group,
            "activation_counterfactual_bottleneck": activation.bottleneck_group,
            "weight_counterfactual_bottleneck": weight.bottleneck_group,
            "ncu_csv": str(path.relative_to(ROOT)),
            "ncu_csv_sha256": pin["sha256"],
            "diagnostic_simulator_wall_ns": time.perf_counter_ns() - started,
        })
    result = {
        "kind": "h200_moe_w1_l2_traffic_evaluation_counterfactual",
        "dataset_sha256": hashlib.sha256(dataset_raw).hexdigest(),
        "baseline_sha256": hashlib.sha256(baseline_raw).hexdigest(),
        "ncu_covered_dram_bottleneck_w1_count": len(rows),
        "ncu_uncovered_dram_bottleneck_w1_count": len(missing_ncu),
        "ncu_uncovered_case_ids": missing_ncu,
        "median_ncu_over_model_dram_read": statistics.median(
            row["ncu_dram_read_bytes"] / row["model_dram_read_bytes"] for row in rows),
        "median_ncu_over_model_l2_read": statistics.median(
            row["ncu_l2_read_bytes"] / row["model_l2_read_bytes"] for row in rows),
        "baseline": _summary(rows, "baseline_predicted_ns"),
        "all_extra_l2_on_activation": _summary(
            rows, "all_extra_l2_on_activation_predicted_ns"),
        "all_extra_l2_on_weight": _summary(
            rows, "all_extra_l2_on_weight_predicted_ns"),
        "rows": rows,
        "limits": [
            "NCU traffic is from separate synthetic runs and only quality-accepted reviewed cases are included.",
            "NCU aggregate L2 read does not identify activation versus weight transactions; both extreme allocations are diagnostic only.",
            "No NCU counter or production duration enters Forward; duration only evaluates the counterfactual.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
