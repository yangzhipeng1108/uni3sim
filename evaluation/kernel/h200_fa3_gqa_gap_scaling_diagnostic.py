"""Contrast source-bound FA3 GQA residuals across query lengths and heads.

The pinned production durations are used only to choose the next independent
mechanism check.  This diagnostic produces no Forward parameter or fit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
INPUT = (ROOT / "experiment_logs/h200_accuracy_20261028/"
         "fa3_gqa_instruction_chain_candidate_20261102.json")
PARTIAL = (ROOT / "experiment_logs/h200_accuracy_20261028/"
           "fa3_gqa_prefill_partial_6case.json")
OUTPUT = (ROOT / "experiment_logs/h200_accuracy_20261028/"
          "fa3_gqa_gap_scaling_20261102.json")
CASE = re.compile(r"HQ(?P<hq>\d+)_HKV(?P<hkv>\d+)_D128_PREFILL_B1_QSUM(?P<q>\d+)_")


def _pin(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    chain = json.loads(INPUT.read_text())
    partial = json.loads(PARTIAL.read_text())
    if (chain["status"] != "evaluation_only_not_forward_service"
            or chain["summary"]["case_count"] != 5
            or len(partial["results"]) != 6):
        raise ValueError("FA3 diagnostic input scope changed")
    ncu_by_case = {row["case_id"]: row["ncu"] for row in partial["results"]}
    rows = []
    for row in chain["rows"]:
        match = CASE.search(row["case_id"])
        if match is None or row["case_id"] not in ncu_by_case:
            raise ValueError(f"FA3 source-bound case missing: {row['case_id']}")
        q = int(match["q"])
        hq = int(match["hq"])
        hkv = int(match["hkv"])
        if hq // hkv != 8:
            raise ValueError("FA3 GQA ratio changed")
        actual = row["operator_actual_ns_evaluation_only"]
        candidate = row["operator_candidate_ns"]
        ncu = ncu_by_case[row["case_id"]]
        rows.append({
            "case_id": row["case_id"], "q_len": q, "query_heads": hq,
            "kv_heads": hkv, "task_count": row["task_count"],
            "kv_tile_visits": row["kv_tile_visits"],
            "actual_operator_ns_evaluation_only": actual,
            "instruction_chain_operator_ns": candidate,
            "unexplained_ns_evaluation_only": actual - candidate,
            "ncu_tensor_pipe_active_pct": ncu["tensor_pipe_active_pct"],
            "ncu_sm_active_to_elapsed_cycle_ratio": (
                ncu["sm_active_cycles_avg"] / ncu["sm_elapsed_cycles_avg"]),
        })
    if sorted((row["q_len"], row["query_heads"]) for row in rows) != [
            (729, 32), (2048, 16), (2048, 32), (8192, 16), (8192, 32)]:
        raise ValueError("FA3 five-case domain changed")
    pairs = []
    for q in (2048, 8192):
        low = next(row for row in rows if row["q_len"] == q
                   and row["query_heads"] == 16)
        high = next(row for row in rows if row["q_len"] == q
                    and row["query_heads"] == 32)
        pairs.append({
            "q_len": q,
            "head_ratio": high["query_heads"] / low["query_heads"],
            "kv_tile_visit_ratio": high["kv_tile_visits"] / low["kv_tile_visits"],
            "actual_operator_ratio_evaluation_only": (
                high["actual_operator_ns_evaluation_only"]
                / low["actual_operator_ns_evaluation_only"]),
            "instruction_chain_operator_ratio": (
                high["instruction_chain_operator_ns"]
                / low["instruction_chain_operator_ns"]),
            "unexplained_ns_evaluation_only": {
                "hq16": low["unexplained_ns_evaluation_only"],
                "hq32": high["unexplained_ns_evaluation_only"],
            },
            "ncu_tensor_pipe_active_pct": {
                "hq16": low["ncu_tensor_pipe_active_pct"],
                "hq32": high["ncu_tensor_pipe_active_pct"],
            },
        })
    result = {
        "kind": "h200_fa3_gqa_gap_scaling_diagnostic",
        "status": "evaluation_only_no_forward_parameter",
        "input_pins": [_pin(INPUT), _pin(PARTIAL)],
        "rows": rows, "equal_length_head_pairs": pairs,
        "interpretation": (
            "At Q8192, doubling GQA heads nearly doubles both observed time "
            "and source-ordered instruction-chain work; the remaining gap "
            "also grows. At Q2048 the observed ratio is smaller, and Q729 "
            "has the largest residual relative to its candidate. These "
            "contrasts motivate separate short-launch/control and long "
            "per-tile progress checks; they do not identify a price."),
        "limits": [
            "Production duration is evaluation-only and is not fitted to a service rate.",
            "NCU active cycles and pipe duty come from separate profiler runs and are not converted into the NSYS duration.",
            "The instruction-chain candidate lacks v0.30 GQA TMA, masking, and cross-warpgroup overlap binding.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "pairs": pairs}, sort_keys=True))


if __name__ == "__main__":
    main()
