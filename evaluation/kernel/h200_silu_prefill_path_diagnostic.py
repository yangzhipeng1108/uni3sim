"""Compare observed and source-selected H200 SiLU paths without fitting."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import time

import zstandard

from evaluation.kernel.dataset import get_hardware_config_path
from evaluation.kernel.evaluation_runner import _get_hardware
from evaluation.kernel.observation_normalizer import normalize_kernel_spec
from src.kernel.kernel_base import predict
from src.kernel.launch_config.select_launch_config import select_launch_config


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "kernel_dataset/nvidia_h200/vector.json.zst"
BASELINE = ROOT / "experiment_logs/h200_accuracy_20261028/kernel_baseline_semantic.json"
EAGER = ROOT / "experiment_logs/h200_accuracy_20261028/eager_first729_kernel_error_comparison.json"
CAMPAIGN = ROOT / "profiler_output/kernel_dataset/nvidia_h200/ncu_1027_20261031"
OUTPUT = ROOT / "experiment_logs/h200_accuracy_20261028/silu_prefill_path_diagnostic_20261102.json"
NCU_CASES = {
    "VECTOR_SILU_T256_I9728_B1": "ncu_vector_022",
    "VECTOR_SILU_T1024_I9728_B1": "ncu_vector_020",
}


def _ncu_read(case_id: str, batch: str) -> dict:
    review_path = CAMPAIGN / f"{batch}_review.json"
    review_bytes = review_path.read_bytes()
    review = json.loads(review_bytes)
    if review["statuses"][case_id]["status"] != "complete":
        raise ValueError(f"NCU case is not accepted: {case_id}")
    matches = [pin for pin in review["raw_files"]
               if pin["path"].endswith(f"/ncu_raw/{case_id}/ncu_stdout.csv")]
    if len(matches) != 1:
        raise ValueError(f"NCU CSV pin is not unique: {case_id}")
    pin = matches[0]
    csv_path = CAMPAIGN / pin["path"]
    raw = csv_path.read_bytes()
    if len(raw) != pin["size"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError(f"NCU CSV differs from review: {case_id}")
    lines = raw.decode().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    rows = [row for row in csv.DictReader(lines[header:]) if row["ID"].isdecimal()]
    if len(rows) != 1 or "act_and_mul_kernel" not in rows[0]["Kernel Name"]:
        raise ValueError(f"NCU SiLU launch differs: {case_id}")
    return {
        "review_sha256": hashlib.sha256(review_bytes).hexdigest(),
        "csv_path": str(csv_path.relative_to(ROOT)),
        "csv_sha256": pin["sha256"],
        "l2_read_bytes": int(rows[0]["lts__t_sectors_op_read.sum"]) * 32,
        "dram_read_bytes": int(rows[0]["dram__bytes_read.sum"]),
    }


def main() -> None:
    dataset_bytes = DATASET.read_bytes()
    dataset = json.loads(zstandard.ZstdDecompressor().decompress(dataset_bytes))
    baseline_bytes = BASELINE.read_bytes()
    baseline = {case["case_id"]: case for case in json.loads(baseline_bytes)["cases"]}
    eager_bytes = EAGER.read_bytes()
    eager = json.loads(eager_bytes)
    hw = _get_hardware(get_hardware_config_path("nvidia_h200"))
    rows = []
    for case_id in sorted(dataset):
        if not (case_id.startswith("KC_SILU_T") and case_id.endswith("_I9728")) \
                and case_id not in NCU_CASES:
            continue
        entry = dataset[case_id]
        spec = normalize_kernel_spec(
            entry["op_info"], entry["kernels"][0], "nvidia_h200", hw)
        selected = select_launch_config(
            "nvidia_h200", spec.kind, spec.shape, spec.dtype,
            spec.variant, spec.attributes, hardware=hw)
        if spec.launch.grid != selected.grid or spec.launch.block != selected.block:
            raise ValueError(f"observed and selected SiLU geometry differs: {case_id}")
        started = time.perf_counter_ns()
        observed_prediction = predict(spec, hw)
        selected_prediction = predict(dataclasses.replace(spec, launch=selected), hw)
        frozen = baseline[case_id]["kernel_results"][0]
        if abs(observed_prediction.t_total_s * 1e9 - frozen["pred_forward_ns"]) > 1e-5:
            raise ValueError(f"observed SiLU path differs from frozen baseline: {case_id}")
        row = {
            "case_id": case_id,
            "tokens": spec.shape["tok"],
            "intermediate": spec.shape["intermediate"],
            "observed_ns": frozen["actual_ns"],
            "observed_launch_predicted_ns": observed_prediction.t_total_s * 1e9,
            "source_selected_predicted_ns": selected_prediction.t_total_s * 1e9,
            "source_selected_bottleneck": selected_prediction.bottleneck_group,
            # vLLM 0.30 BF16 sm90: use_vec=true, use_256b=false.  Each
            # ld128 holds four BF162 values, or eight BF16 elements.
            "source_vector_bf16_elements_per_thread": 8,
            "source_vector_load_rounds_per_cta": math.ceil(
                spec.shape["intermediate"] / (selected.block_size * 8)),
            "legacy_scalar_rounds_per_cta": math.ceil(
                spec.shape["intermediate"] / selected.block_size),
            "source_selected_read_path_ns": (
                selected_prediction.t_per_group["memory:global_read"] * 1e9
                if "memory:global_read" in selected_prediction.t_per_group else None),
            "source_selected_pipeline_ns": (
                selected_prediction.t_per_group["pipeline"] * 1e9
                if "pipeline" in selected_prediction.t_per_group else None),
            "diagnostic_simulator_wall_ns": time.perf_counter_ns() - started,
        }
        if case_id in NCU_CASES:
            row["ncu"] = _ncu_read(case_id, NCU_CASES[case_id])
        rows.append(row)
    eager_rows = [row for row in eager["matched"] if row["stage"] == "silu-mul"]
    if len(eager_rows) != 36 or len({row["shape"]["tok"] for row in eager_rows}) != 1:
        raise ValueError("eager SiLU layer coverage differs")
    result = {
        "kind": "h200_silu_prefill_path_evaluation_diagnostic",
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "eager_comparison_sha256": hashlib.sha256(eager_bytes).hexdigest(),
        "eager_first_layer": {
            "tokens": eager_rows[0]["shape"]["tok"],
            "intermediate": eager_rows[0]["shape"]["intermediate"],
            "observed_ns": eager_rows[0]["observed_us"] * 1000,
            "source_selected_predicted_ns": eager_rows[0]["predicted_us"] * 1000,
            "layer_count": len(eager_rows),
        },
        "rows": rows,
        "limits": [
            "The older canonical SiLU observations and eager vLLM 0.30 trace share a symbol and launch geometry; full source/binary equivalence is not established.",
            "Production durations are used only to assess existing prediction paths and never select a Forward service parameter.",
            "NCU traffic is from separate accepted synthetic runs, not the eager request.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
