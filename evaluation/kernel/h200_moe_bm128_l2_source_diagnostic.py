"""Compare source-issued BM128 MoE W2 reads with qualified H200 NCU counters.

This is an evaluation diagnostic. It does not use NCU duration or modify Forward.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "profiler_output/kernel_dataset/nvidia_h200/ncu_1027_20261031"
OUTPUT = ROOT / "experiment_logs/h200_accuracy_20261028/moe_bm128_w2_l2_source_diagnostic_20261102.json"


def _pin(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _reviewed_file(path: Path, pins: dict[str, dict]) -> None:
    relative = str(path.relative_to(CAMPAIGN))
    pin = pins[relative]
    if (path.stat().st_size != pin["size"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]):
        raise ValueError(f"NCU review pin differs: {relative}")


def _w2_ncu_counters(path: Path) -> dict[str, float | int]:
    lines = path.read_text().splitlines()
    header = next(index for index, line in enumerate(lines)
                  if line.startswith('"ID"'))
    kernels = [row for row in csv.DictReader(lines[header:])
               if row["Kernel Name"] == "fused_moe_kernel"]
    if len(kernels) != 2:
        raise ValueError(f"expected W1 and W2 in {path}")
    w2 = kernels[1]
    counters = {
        "l2_read_bytes": int(w2["lts__t_sectors_op_read.sum"]) * 32,
        "eligible_warps_per_cycle": float(
            w2["smsp__warps_eligible.avg.per_cycle_active"]),
        "active_warps_per_cycle": float(
            w2["smsp__warps_active.avg.per_cycle_active"]),
        "long_scoreboard_stall_pct": float(
            w2["smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"]),
        "tensor_pipe_active_pct": float(
            w2["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"]),
    }
    if any(not math.isfinite(value) or value < 0 for value in counters.values()):
        raise ValueError(f"invalid W2 NCU counter: {path}")
    return counters


def main() -> None:
    rows = []
    for validation in sorted(CAMPAIGN.glob(
            "batches/*/ncu_raw/FUSEDMOE_*/validation.json")):
        case_dir = validation.parent
        case_id = case_dir.name
        batch = case_dir.parents[1].name
        review_path = CAMPAIGN / f"{batch}_review.json"
        if not review_path.is_file():
            continue
        review = json.loads(review_path.read_text())
        if review["statuses"][case_id]["status"] != "complete":
            continue
        launch = (case_dir / "preflight/cases" / case_id /
                  "window_01.launch.json")
        ncu = case_dir / "ncu_stdout.csv"
        pins = {pin["path"]: pin for pin in review["raw_files"]}
        for path in (validation, launch, ncu):
            _reviewed_file(path, pins)
        if json.loads(validation.read_text())["status"] != "complete":
            raise ValueError(f"review/validation status differs: {case_id}")
        tiling = json.loads(launch.read_text())["telemetry"]["moe_tiling"]
        config = tiling["config"]
        if (config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                config["BLOCK_SIZE_K"]) != (128, 128, 64):
            continue
        w2 = tiling["w2"]
        if w2["K"] % 64:
            continue
        counts = tiling["tok_per_expert"]
        if sum(counts) != tiling["num_tokens"] * tiling["top_k"]:
            raise ValueError(f"routed-row count differs: {case_id}")
        m_tiles = sum(math.ceil(count / 128) for count in counts if count)
        n_tiles = math.ceil(w2["N"] / 128)
        if m_tiles * n_tiles > w2["grid"]:
            raise ValueError(f"active W2 CTAs exceed launch grid: {case_id}")
        a_bytes = sum(counts) * w2["K"] * n_tiles * 2
        b_bytes = m_tiles * 128 * w2["K"] * n_tiles * 2
        counters = _w2_ncu_counters(ncu)
        l2_bytes = counters["l2_read_bytes"]
        rows.append({
            "case_id": case_id, "batch": batch,
            "group_size_m": config["GROUP_SIZE_M"],
            "k_extent": w2["K"], "n_extent": w2["N"],
            "routed_rows": sum(counts), "populated_m_tiles": m_tiles,
            "n_tiles": n_tiles, "active_ctas": m_tiles * n_tiles,
            "launch_grid_ctas": w2["grid"],
            "valid_row_fraction": sum(counts) / (m_tiles * 128),
            "source_a_read_bytes": a_bytes,
            "source_b_read_bytes": b_bytes,
            "source_valid_a_plus_b_bytes": a_bytes + b_bytes,
            "full_tile_a_plus_b_bytes": 2 * b_bytes,
            "ncu_l2_read_bytes": l2_bytes,
            "ncu_service_counters": counters,
            "ncu_over_source_valid_ratio": l2_bytes / (a_bytes + b_bytes),
            "ncu_over_full_tile_ratio": l2_bytes / (2 * b_bytes),
            "review_pin": _pin(review_path), "validation_pin": _pin(validation),
            "launch_pin": _pin(launch), "ncu_csv_pin": _pin(ncu),
        })
    if not rows:
        raise ValueError("no qualified BM128/BN128/BK64 W2 counter cases")
    by_group = {}
    for group_size in sorted({row["group_size_m"] for row in rows}):
        selected = [row for row in rows if row["group_size_m"] == group_size]
        by_group[str(group_size)] = {
            "count": len(selected),
            "median_valid_row_fraction": statistics.median(
                row["valid_row_fraction"] for row in selected),
            "median_ncu_over_source_valid_ratio": statistics.median(
                row["ncu_over_source_valid_ratio"] for row in selected),
            "median_ncu_over_full_tile_ratio": statistics.median(
                row["ncu_over_full_tile_ratio"] for row in selected),
            "median_eligible_warps_per_cycle": statistics.median(
                row["ncu_service_counters"]["eligible_warps_per_cycle"]
                for row in selected),
            "median_long_scoreboard_stall_pct": statistics.median(
                row["ncu_service_counters"]["long_scoreboard_stall_pct"]
                for row in selected),
            "median_tensor_pipe_active_pct": statistics.median(
                row["ncu_service_counters"]["tensor_pipe_active_pct"]
                for row in selected),
        }
    result = {
        "kind": "h200_moe_bm128_w2_l2_source_diagnostic",
        "status": "evaluation_only_no_forward_change",
        "metric": "lts__t_sectors_op_read.sum * 32 B, excluding all duration metrics",
        "source_accounting": "valid routed A rows; one full B tile per populated M tile and N tile; BF16 inputs",
        "qualified_case_count": len(rows), "by_group_size_m": by_group,
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "qualified_case_count": len(rows),
                      "by_group_size_m": by_group}))


if __name__ == "__main__":
    main()
