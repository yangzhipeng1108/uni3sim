"""Compare source-ordered short-K panel traffic with pinned H200 W2 counters."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROBE = (ROOT / "profiler_output/microbench/nvidia_h200/"
         "routed_panel_wgmma_grouped_20261102/attempt_1_capture")
PRODUCTION = (ROOT / "experiment_logs/h200_accuracy_20261028/"
              "moe_topcase_source_counter_diagnostic_20261102.json")
OLD_PROBE = (ROOT / "experiment_logs/h200_accuracy_20261028/"
             "w2_panel_footprint_20261102.json")
OUTPUT = (ROOT / "experiment_logs/h200_accuracy_20261028/"
          "w2_panel_grouped_20261102.json")


def _pin(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "sha256":
            hashlib.sha256(data).hexdigest(), "size": len(data)}


def _counter(raw: dict, label: str, mode: int) -> dict:
    record = raw["counters"][label]
    if record["exit_code"] != 0:
        raise ValueError(f"NCU counter failed: {label}")
    lines = record["stdout"].splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    if (len(rows) != 2 or rows[1]["Grid Size"] != "(33784, 1, 1)"
            or f"routed_panel_probe<{mode}, 4>" not in rows[1]["Kernel Name"]
            or rows[0]["lts__t_sectors_op_read.sum"] != "sector"
            or rows[0]["dram__bytes_read.sum"] != "byte"):
        raise ValueError(f"NCU launch or units differ: {label}")
    return {"l2_read_bytes": int(rows[1]["lts__t_sectors_op_read.sum"]) * 32,
            "dram_read_bytes": int(rows[1]["dram__bytes_read.sum"])}


def main() -> None:
    raw_path = PROBE / "raw.json"
    receipt_path = PROBE / "receipt.json"
    info_path = PROBE / "app_info.json"
    raw = json.loads(raw_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    info = json.loads(info_path.read_text())
    source = ROOT / "microbench/nvidia_h200/compute/tensor_core"
    if (receipt["probe_kind"] != "w2_panel_grouped"
            or receipt["app_id"] != info["app_id"]
            or receipt["created_at"] != info["lifecycle"]["created_at"]
            or receipt["stopped_at"] != info["lifecycle"]["stopped_at"]
            or receipt["raw"]["path"] != "raw.json"
            or receipt["raw"]["size"] != _pin(raw_path)["size"]
            or receipt["raw"]["sha256"] != _pin(raw_path)["sha256"]
            or receipt["app_info"]["path"] != "app_info.json"
            or receipt["app_info"]["size"] != _pin(info_path)["size"]
            or receipt["app_info"]["sha256"] != _pin(info_path)["sha256"]
            or raw["status"] != "complete"
            or raw["measurement_boundary"]
            != "four_k_rounds_source_grouped_cp_async_wgmma_no_moe_output"
            or raw["sass_boundary"]["valid"] is not True
            or raw["source_sha256"] != _pin(source / "routed_panel_wgmma_grouped.cu")["sha256"]
            or raw["runner_sha256"]
            != _pin(source / "routed_panel_wgmma_grouped_modal_runner.py")["sha256"]):
        raise ValueError("grouped probe provenance differs")
    execution = [json.loads(line) for line in raw["execution"]["stdout"].splitlines()]
    if (raw["execution"]["exit_code"] != 0 or len(execution) != 4
            or execution[0]["synthetic_rows"] != 540544
            or execution[0]["k_full_rounds"] != 12
            or {(row["mode"], row["rounds"], row["grid"], row["shared_bytes"])
                for row in execution[1:]}
            != {(mode, 4, 33784, 196608) for mode in
                ("gather_copy", "wgmma_only", "gather_plus_wgmma")}):
        raise ValueError("grouped probe execution differs")
    traffic = {label: _counter(raw, label, mode) for label, mode in
               (("gather_copy", 0), ("gather_plus_wgmma", 2))}
    panel_bytes = 33784 * 49152 * 4
    # The odd LCG multiplier permutes all 2^19 row IDs; each valid activation
    # row and every expert weight panel is touched once during four K rounds.
    unique_four_round_bytes = (524288 * 256 * 2
                               + 128 * 2048 * 256 * 2)
    unique_full_k_bytes = 524288 * 768 * 2 + 128 * 2048 * 768 * 2
    production = json.loads(PRODUCTION.read_text())
    old = json.loads(OLD_PROBE.read_text())
    if (production["case_id"] != "FUSEDMOE_TOK32768_E128_TOPK16_H2048_INT768"
            or old["status"] != "independent_probe_not_transferable"):
        raise ValueError("W2 comparator differs")
    production_traffic = production["traffic"]
    result = {
        "kind": "h200_w2_panel_grouped_traffic_diagnostic",
        "status": "source_swizzle_traffic_supported_timing_unbound",
        "app_id": receipt["app_id"],
        "inputs": {"raw": _pin(raw_path), "receipt": _pin(receipt_path),
                   "production_source_counters": _pin(PRODUCTION),
                   "old_independent_probe": _pin(OLD_PROBE)},
        "logical_panel_bytes": panel_bytes,
        "unique_four_round_a_plus_b_bytes": unique_four_round_bytes,
        "probe_traffic": traffic,
        "probe_dram_fraction": {label: row["dram_read_bytes"] / panel_bytes
                                for label, row in traffic.items()},
        "probe_dram_over_unique_backing": {
            label: row["dram_read_bytes"] / unique_four_round_bytes
            for label, row in traffic.items()},
        "old_probe_dram_fraction": old["probe"]["dram_over_logical_panel"],
        "production_dram_fraction": (
            production_traffic["new_dram_read_bytes"] /
            production_traffic["source_unmasked_a_plus_b_panel_bytes"]),
        "production_unique_full_k_a_plus_b_bytes": unique_full_k_bytes,
        "production_dram_over_unique_backing": (
            production_traffic["new_dram_read_bytes"] / unique_full_k_bytes),
        "production_ncu_l2_read_bytes": production_traffic["new_l2_read_bytes"],
        "limitations": [
            "Synthetic grouped rows are not the canonical per-expert histogram.",
            "The probe runs four of twelve K rounds without router, output, or a full MoE result.",
            "Uncontrolled caches and clocks differ between unprofiled event and NCU launches.",
            "The DRAM-fraction agreement does not identify finite CTA service or overlap.",
        ],
        "production_duration_consumed": False,
        "reproduce": "python -m evaluation.kernel.h200_w2_panel_grouped_diagnostic",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
