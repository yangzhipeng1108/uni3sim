"""Compare independent short-K panel traffic with pinned W2 counters."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
PROBE = (ROOT / "profiler_output/microbench/nvidia_h200/"
         "routed_panel_wgmma_footprint_20261102/attempt_1_capture")
PRODUCTION = (ROOT / "experiment_logs/h200_accuracy_20261028/"
              "moe_topcase_source_counter_diagnostic_20261102.json")
OUTPUT = (ROOT / "experiment_logs/h200_accuracy_20261028/"
          "w2_panel_footprint_20261102.json")


def _pin(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "sha256":
            hashlib.sha256(data).hexdigest(), "size": len(data)}


def _counter_row(raw: dict, label: str) -> dict:
    lines = raw["counters"][label]["stdout"].splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[first:]))))
    if len(rows) != 2 or rows[1]["Grid Size"] != "(2112, 1, 1)":
        raise ValueError(f"independent NCU launch differs: {label}")
    row = rows[1]
    expected = ("routed_panel_probe<0, 8>" if label == "gather_copy"
                else "routed_panel_probe<2, 8>")
    if expected not in row["Kernel Name"]:
        raise ValueError(f"independent NCU kernel differs: {label}")
    return {"l2_read_bytes": int(row["lts__t_sectors_op_read.sum"]) * 32,
            "dram_read_bytes": int(row["dram__bytes_read.sum"])}


def main() -> None:
    raw_path = PROBE / "raw.json"
    receipt_path = PROBE / "receipt.json"
    raw = json.loads(raw_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if (receipt["raw"]["sha256"] != _pin(raw_path)["sha256"]
            or receipt["raw"]["size"] != raw_path.stat().st_size
            or raw["status"] != "complete"
            or raw["sass_boundary"]["valid"] is not True
            or raw["source_sha256"] != _pin(ROOT / "microbench/nvidia_h200/compute/"
                                            "tensor_core/routed_panel_wgmma_footprint.cu")["sha256"]
            or raw["runner_sha256"] != _pin(ROOT / "microbench/nvidia_h200/compute/"
                                            "tensor_core/routed_panel_wgmma_footprint_modal_runner.py")["sha256"]):
        raise ValueError("independent probe provenance differs")
    measurements = [json.loads(line) for line in raw["execution"]["stdout"].splitlines()]
    if (len(measurements) != 7 or measurements[0]["synthetic_rows"] != 270336
            or measurements[0]["k_full_rounds"] != 12):
        raise ValueError("independent probe result set differs")
    events = {}
    for row in measurements[1:]:
        key = f"{row['mode']}_r{row['rounds']}"
        if key in events or row["grid"] != 2112 or row["shared_bytes"] != 196608:
            raise ValueError("independent probe event geometry differs")
        events[key] = {"event_median_ms": statistics.median(row["samples_ms"]),
                       "cta_cycles_p50": row["cta_cycles_p10_p50_p90"][1]}
    if set(events) != {f"{mode}_r{rounds}" for mode in
                      ("gather_copy", "wgmma_only", "gather_plus_wgmma")
                      for rounds in (4, 8)}:
        raise ValueError("independent probe modes differ")
    traffic = {label: _counter_row(raw, label) for label in
               ("gather_copy", "gather_plus_wgmma")}
    logical_panel_bytes = 2112 * 8 * 49152
    source = json.loads(PRODUCTION.read_text())
    production = source["traffic"]
    if source["case_id"] != "FUSEDMOE_TOK32768_E128_TOPK16_H2048_INT768":
        raise ValueError("production comparator differs")
    result = {
        "kind": "h200_w2_panel_footprint_diagnostic",
        "status": "independent_probe_not_transferable",
        "app_id": receipt["app_id"],
        "inputs": {"raw": _pin(raw_path), "receipt": _pin(receipt_path),
                   "production_source_counters": _pin(PRODUCTION)},
        "probe": {"grid_ctas": 2112, "rounds": [4, 8],
                  "activation_backing_bytes": 524288 * 768 * 2,
                  "logical_panel_bytes_for_8_rounds": logical_panel_bytes,
                  "events": events, "counter_bytes": traffic,
                  "dram_over_logical_panel": {
                      label: row["dram_read_bytes"] / logical_panel_bytes
                      for label, row in traffic.items()}},
        "production_w2": {
            "logical_unmasked_panel_bytes":
                production["source_unmasked_a_plus_b_panel_bytes"],
            "dram_read_bytes": production["new_dram_read_bytes"],
            "dram_over_logical_panel":
                production["new_dram_read_bytes"] /
                production["source_unmasked_a_plus_b_panel_bytes"]},
        "interpretation": (
            "The old small-backing limitation was removed, but this probe "
            "assigns a different synthetic row array to every CTA. The "
            "vLLM GROUP_SIZE_M swizzle reuses one M tile's rows across N "
            "tiles. The probe's much higher DRAM fraction cannot price the "
            "production W2 K-loop or justify a Forward parameter."),
        "production_duration_consumed": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
