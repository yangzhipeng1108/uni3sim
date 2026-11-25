"""Evaluate a source-ordered FA3 instruction chain without fitting duration.

This is an evaluation-only hypothesis.  The mechanism reference measured
individual instructions in the older FA3 MLA program; this diagnostic tests
whether their shapes and the v0.30 GQA dependency order are useful before a
source-matched service is considered for Forward.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import time

from evaluation.kernel.dataset import get_hardware_config_path, load_kernel_dataset
from evaluation.kernel.evaluation_runner import _get_hardware
from evaluation.kernel.h200_fa3_gqa_prefill_diagnostic import (
    BATCHES, CAMPAIGN, SOURCE_AUDIT, _source_bound_case,
)
from src.kernel.predictors.flash_attn import (
    _fa3_gqa_prefill_030_tile_blocks, _fa3_list_schedule,
)


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "microbench/nvidia_h200/references/fa3_mla_mechanisms_sm90.json"
PARTIAL = ROOT / "experiment_logs/h200_accuracy_20261028/fa3_gqa_prefill_partial_6case.json"
OUTPUT = ROOT / "experiment_logs/h200_accuracy_20261028/fa3_gqa_instruction_chain_candidate_20261102.json"


def _pin(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    reference = json.loads(REFERENCE.read_text())
    if (reference["schema"] != "nvidia_h200_fa3_mla_mechanisms_reference_v1"
            or reference["hardware_id"] != "nvidia_h200"
            or reference["production_or_evaluation_duration_consumed"] is not False
            or reference["measured_sm_clock_mhz"] != 1980):
        raise ValueError("FA3 independent mechanism reference changed")
    audit = json.loads(SOURCE_AUDIT.read_text())
    if (audit["accepted_static_domain"]["block_m"] != 128
            or audit["accepted_static_domain"]["block_n"] != 128
            or audit["accepted_static_domain"]["mma_warpgroups"] != 2
            or audit["mechanism_reuse_audit"]["wgmma_instruction_shapes"] !=
            "QK SS m64n128k16 x 8 and PV RS m64n128k16 x 8 are probe-covered individually"
            or audit["mechanism_reuse_audit"]["softmax_fragment"] !=
            "two rows x 32 score values and two rows x 32 output values are probe-covered individually"):
        raise ValueError("vLLM 0.30 source/probe shape binding changed")
    qk = reference["wgmma_gemm_cycles"]["ss_n128_kmajor"]["8"]["2"]
    pv = reference["wgmma_gemm_cycles"]["rs_n128_mnmajor"]["8"]["2"]
    softmax = reference["softmax_step_cycles"]["32x32"]["2"]
    first_cycles = qk + softmax + pv
    steady_cycles = max(qk, pv) + softmax
    partial = json.loads(PARTIAL.read_text())
    prior = {row["case_id"]: row for row in partial["results"]}
    dataset = load_kernel_dataset("nvidia_h200", ["flash_attn"])
    hardware = _get_hardware(get_hardware_config_path("nvidia_h200"))
    rows = []
    for batch in BATCHES:
        review_path = CAMPAIGN / f"{batch}_review.json"
        review = json.loads(review_path.read_text())
        for case_id, status in review["statuses"].items():
            if "_PREFILL_" not in case_id or status["status"] != "complete":
                continue
            specs = _source_bound_case(case_id, batch, review, dataset[case_id],
                                       hardware)
            if specs is None:
                continue
            main, _ = specs
            blocks = _fa3_gqa_prefill_030_tile_blocks(main)
            start = time.perf_counter()
            tasks = _fa3_list_schedule(
                [first_cycles + (count - 1) * steady_cycles for count in blocks],
                main.launch.grid[0])
            main_ns = max(load for load, _ in tasks) * 1000 / reference["measured_sm_clock_mhz"]
            wall_s = time.perf_counter() - start
            baseline = prior[case_id]
            actual_ns = baseline["op_actual_ns"]
            candidate_ns = main_ns + baseline["fill_candidate_ns"]
            rows.append({
                "case_id": case_id, "review": _pin(review_path),
                "task_count": len(blocks), "kv_tile_visits": sum(blocks),
                "main_instruction_chain_ns": main_ns,
                "fill_existing_prediction_ns": baseline["fill_candidate_ns"],
                "operator_candidate_ns": candidate_ns,
                "operator_current_partial_ns": baseline["op_candidate_ns"],
                "operator_actual_ns_evaluation_only": actual_ns,
                "candidate_ape_pct": abs(candidate_ns - actual_ns) / actual_ns * 100,
                "current_partial_ape_pct": baseline["op_error_pct"],
                "simulation_wall_s": wall_s,
            })
    if len(rows) != 5 or len(prior) != 6:
        raise ValueError("FA3 source-bound evaluation case coverage changed")
    output = {
        "kind": "h200_fa3_gqa_instruction_chain_diagnostic",
        "status": "evaluation_only_not_forward_service",
        "inputs": [_pin(REFERENCE), _pin(SOURCE_AUDIT), _pin(PARTIAL)],
        "mechanism": {
            "qk_ss_cycles": qk, "pv_rs_cycles": pv,
            "softmax_fragment_cycles": softmax,
            "first_kv_tile_cycles": first_cycles,
            "subsequent_kv_tile_cycles": steady_cycles,
            "source_order": "first QK then softmax then PV; later QK and previous PV overlap before softmax",
            "clock_mhz": reference["measured_sm_clock_mhz"],
        },
        "summary": {
            "case_count": len(rows),
            "current_partial_mape_pct": statistics.mean(
                row["current_partial_ape_pct"] for row in rows),
            "instruction_chain_mape_pct": statistics.mean(
                row["candidate_ape_pct"] for row in rows),
            "improved_case_count": sum(row["candidate_ape_pct"]
                                       < row["current_partial_ape_pct"]
                                       for row in rows),
        },
        "limitations": [
            "Independent instruction reference is from older FA3 MLA revision, not a v0.30 GQA combined program.",
            "The source list schedule is a candidate mapping of tasks to 132 persistent CTAs, not a measured timeline.",
            "TMA, masking, output and cross-warpgroup overlap remain unpriced; production duration is used only for evaluation.",
        ],
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
