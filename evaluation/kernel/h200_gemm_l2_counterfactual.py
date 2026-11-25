"""Evaluation-only L2 read counterfactual for two H200 prefill GEMMs.

Run with the repository's configured Python environment.  The observed NCU bytes are
never passed to production launch selection or stored in a Forward parameter.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import zstandard

from evaluation.kernel.dataset import get_hardware_config_path
from evaluation.kernel.evaluation_runner import _get_hardware
from evaluation.kernel.observation_normalizer import normalize_kernel_spec
from src.kernel.formula import PerformanceFormula
from src.kernel.kernel_base import predict
from src.kernel.predictors.gemm import GEMMPredictor


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "experiment_logs/h200_accuracy_20261028/gemm_l2_counterfactual_20261102.json"
BOUNDS = ROOT / "experiment_logs/h200_accuracy_20261028/gemm_4x2_cluster_traffic_bounds_20261102.json"
DATASET = ROOT / "kernel_dataset/nvidia_h200/gemm.json.zst"


def _ncu_read_bytes(case: dict) -> int:
    path = ROOT / case["ncu_csv"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != case["ncu_csv_sha256"]:
        raise ValueError(f"NCU CSV changed: {path}")
    lines = raw.decode().splitlines()
    header = next(index for index, line in enumerate(lines)
                  if line.startswith('"ID","Process ID",'))
    rows = [row for row in csv.DictReader(lines[header:])
            if row["ID"].isdecimal()]
    if len(rows) != 1:
        raise ValueError(f"expected one NCU kernel for {path}, got {len(rows)}")
    row = rows[0]
    if (row["Kernel Name"] != "nvjet_tst_128x128_64x6_4x2_h_bz_TNT"
            or row["Grid Size"] != "(8, 15, 1)"
            or row["Block Size"] != "(384, 1, 1)"
            or case["case_id"] not in row[
                "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"]):
        raise ValueError(f"NCU launch or NVTX differs: {path}")
    read = int(row["lts__t_sectors_op_read.sum"]) * 32
    if read != case["observed_l2_read_bytes"]:
        raise ValueError(f"NCU L2 read differs from source diagnostic: {path}")
    return read


def _replace_l2_read(inp, read_bytes: int):
    paths = []
    old_read = None
    for path in inp.memory_paths:
        if path.access_kind == "read":
            if old_read is not None:
                raise ValueError("expected one GEMM read path")
            nodes = []
            for node in path.nodes:
                if node.resource_role == "l2_cache":
                    old_read = node.bytes
                    nodes.append(dataclasses.replace(
                        node, bytes=float(read_bytes),
                        transactions=read_bytes / node.transaction_bytes))
                else:
                    nodes.append(node)
            path = dataclasses.replace(
                path, nodes=tuple(nodes),
                initial_transactions=nodes[0].transactions)
        paths.append(path)
    if old_read is None:
        raise ValueError("GEMM read path lacks L2 service")
    demand = dict(inp.demand_per_resource)
    demand["l2_cache"] += read_bytes - old_read
    return dataclasses.replace(
        inp, demand_per_resource=demand, memory_paths=tuple(paths)), old_read


def main() -> None:
    source = json.loads(BOUNDS.read_text())
    raw_dataset = DATASET.read_bytes()
    dataset = json.loads(zstandard.ZstdDecompressor().decompress(raw_dataset))
    hardware = _get_hardware(get_hardware_config_path("nvidia_h200"))
    formula = PerformanceFormula(hardware)
    predictor = GEMMPredictor()
    output = []
    for case in source["cases"]:
        case_id = case["case_id"]
        entry = dataset[case_id]
        if len(entry["kernels"]) != 1:
            raise ValueError(f"expected one GEMM kernel: {case_id}")
        spec = normalize_kernel_spec(
            entry["op_info"], entry["kernels"][0], "nvidia_h200", hardware)
        started = time.perf_counter_ns()
        inp, _ = predictor.resource_input(spec, hardware)
        baseline = formula.predict_kernel(inp)
        forward = predict(spec, hardware, "midlayer")
        if abs(baseline.t_total_s - forward.t_total_s) > 1e-12:
            raise ValueError(f"resource input and public Forward differ: {case_id}")
        observed_read = _ncu_read_bytes(case)
        changed_input, model_read = _replace_l2_read(inp, observed_read)
        changed = formula.predict_kernel(changed_input)
        wall_ns = time.perf_counter_ns() - started
        actual_ns = entry["op_info"]["evaluation"]["timing"]["total_duration_ns"]
        output.append({
            "case_id": case_id,
            "actual_ns": actual_ns,
            "actual_source": "canonical M768 NSYS median; distinct from eager M729",
            "model_l2_read_bytes": model_read,
            "ncu_l2_read_bytes": observed_read,
            "baseline_predicted_ns": baseline.t_total_s * 1e9,
            "l2_only_counterfactual_ns": changed.t_total_s * 1e9,
            "baseline_ape": abs(baseline.t_total_s * 1e9 - actual_ns) / actual_ns,
            "counterfactual_ape": abs(changed.t_total_s * 1e9 - actual_ns) / actual_ns,
            "baseline_bottleneck": baseline.bottleneck_group,
            "counterfactual_bottleneck": changed.bottleneck_group,
            "diagnostic_simulator_wall_ns": wall_ns,
        })
    result = {
        "kind": "h200_gemm_l2_read_evaluation_counterfactual",
        "dataset_sha256": hashlib.sha256(raw_dataset).hexdigest(),
        "source_bounds_sha256": hashlib.sha256(BOUNDS.read_bytes()).hexdigest(),
        "cases": output,
        "limits": [
            "NCU is a separate CUDA12.9 synthetic run with matching shape and launch.",
            "Only aggregate L2 read demand changes; DRAM, tensor work, and CTA program stay as modeled.",
            "This uses measured production traffic for diagnosis, never for Forward or parameter selection.",
            "Canonical M768 duration, not the different eager M729 request, supplies the error denominator.",
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()
