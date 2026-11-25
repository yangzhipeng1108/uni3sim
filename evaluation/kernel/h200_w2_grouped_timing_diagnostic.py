"""Audit the independent H200 grouped-panel CTA phase timing probe."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = (ROOT / "profiler_output/microbench/nvidia_h200/"
            "routed_panel_wgmma_grouped_timing_20261102")
ATTEMPT = CAMPAIGN / "attempt_2_capture"
SOURCE = ROOT / "microbench/nvidia_h200/compute/tensor_core"
OUTPUT = (ROOT / "experiment_logs/h200_accuracy_20261028/"
          "w2_grouped_timing_20261102.json")
PRODUCTION_SASS = (ROOT / "experiment_logs/h200_accuracy_20261028/"
                   "moe_topcase_ncu_sass_audit_20261102.json")
PRODUCTION_PROGRESS = (ROOT / "experiment_logs/h200_accuracy_20261028/"
                       "moe_w2_source_progress_topcase_20261102.json")
MODES = ("gather_copy", "wgmma_only", "gather_plus_wgmma")


def _pin(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "sha256":
            hashlib.sha256(data).hexdigest(), "size": len(data)}


def _receipt_pin(receipt: dict, name: str) -> None:
    pin = receipt[name]
    path = ATTEMPT / pin["path"]
    if (path.stat().st_size != pin["size"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]):
        raise ValueError(f"W2 timing receipt pin differs: {name}")


def _traffic(raw: dict, mode: str, index: int) -> dict:
    captured = raw["counters"][mode]
    if captured["exit_code"] != 0:
        raise ValueError(f"W2 timing NCU failed: {mode}")
    lines = captured["stdout"].splitlines()
    start = next(position for position, line in enumerate(lines)
                 if line.startswith('"ID"'))
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    if (len(rows) != 2
            or f"routed_panel_probe<{index}, 8>" not in rows[1]["Kernel Name"]
            or rows[1]["Grid Size"] != "(33784, 1, 1)"
            or rows[1]["Block Size"] != "(256, 1, 1)"
            or rows[0]["lts__t_sectors_op_read.sum"] != "sector"
            or rows[0]["dram__bytes_read.sum"] != "byte"):
        raise ValueError(f"W2 timing NCU launch or units differ: {mode}")
    return {"l2_read_bytes": int(rows[1]["lts__t_sectors_op_read.sum"]) * 32,
            "dram_read_bytes": int(rows[1]["dram__bytes_read.sum"])}


def main() -> None:
    raw_path = ATTEMPT / "raw.json"
    receipt_path = ATTEMPT / "receipt.json"
    info_path = ATTEMPT / "app_info.json"
    raw = json.loads(raw_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    info = json.loads(info_path.read_text())
    for name in ("raw", "log", "app_info"):
        _receipt_pin(receipt, name)
    lifecycle = info["lifecycle"]
    if (receipt["probe_kind"] != "w2_panel_grouped_timing"
            or receipt["app_id"] != info["app_id"]
            or receipt["created_at"] != lifecycle["created_at"]
            or receipt["stopped_at"] != lifecycle["stopped_at"]
            or lifecycle["state"] != "stopped"
            or info["functions"]["capture"]["summary"]["gpu_config"]
            != [{"count": 1, "gpu_type": "H200"}]
            or raw["status"] != "complete"
            or raw["measurement_boundary"] !=
            "four_eight_k_rounds_source_grouped_cta_phase_timing_no_moe_output"
            or raw["sass_boundary"]["valid"] is not True
            or raw["execution"]["exit_code"] != 0):
        raise ValueError("W2 timing probe provenance or execution differs")
    for name, path in (
            ("source_sha256", SOURCE / "routed_panel_wgmma_grouped_timing.cu"),
            ("runner_sha256", SOURCE / "routed_panel_wgmma_grouped_timing_modal_runner.py"),
            ("generator_sha256", SOURCE / "wgmma_ss_wait1_issue_gen.py"),
            ("reference_sha256", ROOT / "microbench/nvidia_h200/operators/"
             "flash_attn/fa3_wgmma_service.cu")):
        if raw[name] != _pin(path)["sha256"]:
            raise ValueError(f"W2 timing source identity differs: {name}")
    for rounds in (4, 8):
        sass = raw["sass_boundary"]["counts_by_round"][str(rounds)]
        if (sass["hgmma"] != 4 or sass["ldgsts"] != 28
                or sass["wait1"] != 1 or sass["wait0"] != 1):
            raise ValueError(f"W2 timing SASS differs at {rounds} rounds")
    execution = [json.loads(line) for line in
                 raw["execution"]["stdout"].splitlines()]
    if (len(execution) != 7 or execution[0]["device"] != "NVIDIA H200"
            or execution[0]["sm_count"] != 132
            or execution[0]["synthetic_rows"] != 540544
            or execution[0]["k_full_rounds"] != 12):
        raise ValueError("W2 timing execution header differs")
    samples = {}
    for row in execution[1:]:
        mode, rounds = row["mode"], row["rounds"]
        if (mode not in MODES or rounds not in (4, 8)
                or row["grid"] != 33784 or row["shared_bytes"] != 196608
                or len(row["samples_ms"]) != 7
                or any(not math.isfinite(value) or value <= 0
                       for value in row["samples_ms"])):
            raise ValueError("W2 timing sample identity or value differs")
        for key in ("cta_prefill_issue_cycles_p10_p50_p90",
                    "cta_loop_wait_cycles_p10_p50_p90",
                    "cta_full_cycles_p10_p50_p90"):
            values = row[key]
            if (len(values) != 3 or values != sorted(values)
                    or values[0] <= 0):
                raise ValueError(f"W2 timing CTA cycles invalid: {key}")
        key = f"{mode}_{rounds}"
        if key in samples:
            raise ValueError(f"W2 timing duplicate sample: {key}")
        samples[key] = {
            "event_median_ms": statistics.median(row["samples_ms"]),
            "event_samples_ms": row["samples_ms"],
            "cta_setup_and_prefill_issue_cycles_p10_p50_p90":
                row["cta_prefill_issue_cycles_p10_p50_p90"],
            "cta_loop_wait_cycles_p10_p50_p90":
                row["cta_loop_wait_cycles_p10_p50_p90"],
            "cta_full_cycles_p10_p50_p90":
                row["cta_full_cycles_p10_p50_p90"],
        }
    if set(samples) != {f"{mode}_{rounds}" for mode in MODES
                       for rounds in (4, 8)}:
        raise ValueError("W2 timing mode/round coverage differs")
    traffic = {mode: _traffic(raw, mode, index) for mode, index in
               (("gather_copy", 0), ("gather_plus_wgmma", 2))}
    sass_audit = json.loads(PRODUCTION_SASS.read_text())
    progress = json.loads(PRODUCTION_PROGRESS.read_text())
    if (sass_audit["case_id"] != progress["case_id"]
            or sass_audit["kernel_sass_static_counts"][1]["role"] != "w2"
            or sass_audit["kernel_sass_static_counts"][1]["LDGSTS."] != 48
            or progress["w2_program"]["registers_per_thread"] != 214):
        raise ValueError("production W2 SASS comparator differs")
    logical_panel_bytes = 33784 * 49152 * 8
    unique_backing_bytes = (524288 * 512 * 2 + 128 * 2048 * 512 * 2)
    result = {
        "kind": "h200_w2_grouped_cta_phase_timing_diagnostic",
        "status": "independent_short_k_progress_only",
        "app_id": receipt["app_id"],
        "inputs": {"raw": _pin(raw_path), "receipt": _pin(receipt_path),
                   "app_info": _pin(info_path),
                   "production_sass": _pin(PRODUCTION_SASS),
                   "production_progress": _pin(PRODUCTION_PROGRESS)},
        "samples": samples,
        "ncu_8_round_traffic": traffic,
        "ncu_8_round_logical_panel_bytes": logical_panel_bytes,
        "ncu_8_round_unique_a_plus_b_backing_bytes": unique_backing_bytes,
        "ncu_8_round_dram_over_unique_backing": {
            mode: counters["dram_read_bytes"] / unique_backing_bytes
            for mode, counters in traffic.items()},
        "production_duration_consumed": False,
        "limitations": [
            "Synthetic grouped rows differ from the production routing histogram.",
            "Four and eight rounds omit the target twelve-round output and router.",
            "Probe combined SASS has 28 static LDGSTS sites and 194 registers; production W2 has 48 and 214.",
            "The first CTA interval includes accumulator initialization before asynchronous panel issue; it is not a pure copy service.",
            "The NCU traffic pass and unprofiled event pass have different cache and clock conditions.",
            "CTA cycles lack an independently measured active SM clock; event/wave scaling is not a transferable service.",
        ],
        "reproduce": "python -m evaluation.kernel.h200_w2_grouped_timing_diagnostic",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
