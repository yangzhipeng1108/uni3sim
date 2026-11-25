"""Evaluate source/NCU-backed pair sharing without changing H200 Forward."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import time

import zstandard

from evaluation.kernel.dataset import get_hardware_config_path
from evaluation.kernel.evaluation_runner import _get_hardware
from evaluation.kernel.h200_gemm_l2_counterfactual import _replace_l2_read
from evaluation.kernel.observation_normalizer import normalize_kernel_spec
from src.kernel.formula import PerformanceFormula
from src.kernel.kernel_base import predict
from src.kernel.predictors.gemm import GEMMPredictor


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiment_logs/h200_accuracy_20261028/ncu_gemm_cluster_pair_analytic_candidate.json"
DATASET = ROOT / "kernel_dataset/nvidia_h200/gemm.json.zst"
OUTPUT = ROOT / "experiment_logs/h200_accuracy_20261028/gemm_cluster_pair_duration_counterfactual_20261102.json"


def _summary(rows: list[dict]) -> dict:
    observed = sum(row["observed_ns"] for row in rows)
    return {
        "case_count": len(rows),
        "baseline_mape": statistics.mean(row["baseline_ape"] for row in rows),
        "candidate_mape": statistics.mean(row["candidate_ape"] for row in rows),
        "baseline_wmae": sum(abs(row["baseline_predicted_ns"] - row["observed_ns"])
                              for row in rows) / observed,
        "candidate_wmae": sum(abs(row["candidate_predicted_ns"] - row["observed_ns"])
                               for row in rows) / observed,
        "improved_cases": sum(row["candidate_ape"] < row["baseline_ape"]
                              for row in rows),
        "regressed_cases": sum(row["candidate_ape"] > row["baseline_ape"]
                               for row in rows),
    }


def main() -> None:
    raw_source = SOURCE.read_bytes()
    source = json.loads(raw_source)
    raw_dataset = DATASET.read_bytes()
    dataset = json.loads(zstandard.ZstdDecompressor().decompress(raw_dataset))
    hardware = _get_hardware(get_hardware_config_path("nvidia_h200"))
    predictor = GEMMPredictor()
    formula = PerformanceFormula(hardware)
    seen = set()
    rows = []
    for case in source["cases"]:
        case_id = case["case_id"]
        if case_id in seen or case["cluster"] not in ([1, 2], [2, 1]):
            raise ValueError(f"invalid source cohort: {case_id}")
        seen.add(case_id)
        entry = dataset[case_id]
        if len(entry["kernels"]) != 1:
            raise ValueError(f"expected one GEMM kernel: {case_id}")
        spec = normalize_kernel_spec(
            entry["op_info"], entry["kernels"][0], "nvidia_h200", hardware)
        started = time.perf_counter_ns()
        inp, _ = predictor.resource_input(spec, hardware)
        baseline = formula.predict_kernel(inp)
        public = predict(spec, hardware, "midlayer")
        if abs(baseline.t_total_s - public.t_total_s) > 1e-12:
            raise ValueError(f"resource input and public Forward differ: {case_id}")
        candidate_input, old_l2_read = _replace_l2_read(
            inp, case["cluster_pair_analytic_l2_bytes"])
        candidate = formula.predict_kernel(candidate_input)
        wall_ns = time.perf_counter_ns() - started
        observed = entry["op_info"]["evaluation"]["timing"]["total_duration_ns"]
        baseline_ns = baseline.t_total_s * 1e9
        candidate_ns = candidate.t_total_s * 1e9
        rows.append({
            "case_id": case_id,
            "cluster": case["cluster"],
            "observed_ns": observed,
            "baseline_predicted_ns": baseline_ns,
            "candidate_predicted_ns": candidate_ns,
            "baseline_ape": abs(baseline_ns - observed) / observed,
            "candidate_ape": abs(candidate_ns - observed) / observed,
            "baseline_bottleneck": baseline.bottleneck_group,
            "candidate_bottleneck": candidate.bottleneck_group,
            "model_l2_read_bytes": old_l2_read,
            "analytic_pair_l2_read_bytes": case["cluster_pair_analytic_l2_bytes"],
            "ncu_l2_read_bytes": case["ncu_l2_read_bytes"],
            "diagnostic_simulator_wall_ns": wall_ns,
        })
    result = {
        "kind": "h200_gemm_cluster_pair_duration_counterfactual",
        "source_sha256": hashlib.sha256(raw_source).hexdigest(),
        "dataset_sha256": hashlib.sha256(raw_dataset).hexdigest(),
        "summary": {
            "all": _summary(rows),
            "cluster_1x2": _summary([row for row in rows if row["cluster"] == [1, 2]]),
            "cluster_2x1": _summary([row for row in rows if row["cluster"] == [2, 1]]),
        },
        "rows": rows,
        "interpretation": (
            "Pair sharing follows source geometry and non-duration NCU traffic, "
            "but changing only aggregate L2 read worsens duration. This is an "
            "evaluation-only counterfactual, not a Forward parameter or launch rule."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
