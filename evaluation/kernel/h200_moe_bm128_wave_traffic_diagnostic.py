"""Check the BM128 W2 cache replay wave against qualified NCU read traffic.

The alternative wave width follows H200 shared/register occupancy.  Only
non-duration traffic counters enter this diagnostic; neither candidate is
selected for Forward by production timing or counter error.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import statistics

from evaluation.kernel.dataset import get_hardware_config_path
from evaluation.kernel.evaluation_runner import _get_hardware
from src.kernel.cache_traffic_sim import simulate_moe_tile_program


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "experiment_logs/h200_accuracy_20261028/"
          "moe_bm128_w2_l2_source_diagnostic_20261102.json")
OUTPUT = (ROOT / "experiment_logs/h200_accuracy_20261028/"
          "moe_bm128_wave_traffic_20261102.json")


def _pin(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _check_pin(pin: dict) -> Path:
    path = ROOT / pin["path"]
    if (path.stat().st_size != pin["size"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]):
        raise ValueError(f"qualified W2 input pin differs: {path}")
    return path


def _w2_dram_read(path: Path) -> int:
    lines = path.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    kernels = [row for row in csv.DictReader(lines[header:])
               if row["Kernel Name"] == "fused_moe_kernel"]
    if len(kernels) != 2:
        raise ValueError(f"W1/W2 NCU rows changed: {path}")
    return int(kernels[1]["dram__bytes_read.sum"])


def main() -> None:
    source = json.loads(SOURCE.read_text())
    if (source["qualified_case_count"] != 39
            or source["status"] != "evaluation_only_no_forward_change"):
        raise ValueError("BM128 source diagnostic scope changed")
    hardware = _get_hardware(get_hardware_config_path("nvidia_h200"))
    if hardware.num_compute_units != 132:
        raise ValueError("H200 SM count changed")
    rows = []
    for row in source["rows"]:
        for name in ("review_pin", "validation_pin", "launch_pin", "ncu_csv_pin"):
            _check_pin(row[name])
        review = json.loads(_check_pin(row["review_pin"]).read_text())
        validation = json.loads(_check_pin(row["validation_pin"]).read_text())
        if (review["statuses"][row["case_id"]]["status"] != "complete"
                or validation["status"] != "complete"):
            raise ValueError(f"W2 case lost qualified status: {row['case_id']}")
        launch = json.loads(_check_pin(row["launch_pin"]).read_text())
        tiling = launch["telemetry"]["moe_tiling"]
        config = tiling["config"]
        w2 = tiling["w2"]
        if (config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                config["BLOCK_SIZE_K"], config["GROUP_SIZE_M"]) != (
                    128, 128, 64, row["group_size_m"]):
            raise ValueError(f"W2 tile contract differs: {row['case_id']}")
        launches = [item for item in launch["launches"]
                    if item["name"] == "fused_moe_kernel"]
        if len(launches) != 2 or launches[1]["grid"] != [w2["grid"], 1, 1]:
            raise ValueError(f"W2 launch differs: {row['case_id']}")
        actual = launches[1]
        shared = (actual["dynamic_shared_memory_bytes"]
                  + actual["static_shared_memory_bytes"])
        registers = actual["registers_per_thread"] * actual["block"][0]
        resident = min(hardware.max_smem_per_unit_bytes // shared,
                       hardware.max_regs_per_unit // registers)
        if resident != 2 or actual["block"] != [256, 1, 1]:
            raise ValueError(f"W2 physical residency differs: {row['case_id']}")
        counts = tiling["tok_per_expert"]
        if sum(counts) != row["routed_rows"]:
            raise ValueError(f"W2 routed rows differ: {row['case_id']}")
        n_tiles = row["n_tiles"]
        if w2["grid"] % n_tiles:
            raise ValueError(f"W2 launch grid is not tiled: {row['case_id']}")
        actual_dram = _w2_dram_read(_check_pin(row["ncu_csv_pin"]))
        predictions = {}
        for wave in (132, 264):
            traffic = simulate_moe_tile_program(
                m_tiles=w2["grid"] // n_tiles, n_tiles=n_tiles,
                group_size_m=row["group_size_m"],
                k_extent=w2["K"], block_k=64,
                n_extent=w2["N"], block_n=128, element_bytes=2,
                c_row_bytes=256, block_m=128, wave_size=wave,
                tokens_per_expert=counts,
                capacity_bytes=hardware.l2_replacement_capacity_bytes,
                residency_profile=hardware.l2_replacement_residency_profile)
            # The C replay identifies the tail N object by its unique logical
            # columns.  The source issues a full tile with modulo-wrapped
            # columns, so its L2 request count can be larger at the tail.
            unique_operand_bytes = (
                row["source_a_read_bytes"]
                + row["populated_m_tiles"] * w2["K"] * w2["N"] * 2)
            if traffic.a_l2_bytes + traffic.b_l2_bytes != unique_operand_bytes:
                raise ValueError(f"W2 replay unique bytes differ: {row['case_id']}")
            predictions[str(wave)] = traffic.a_dram_bytes + traffic.b_dram_bytes
        rows.append({
            "case_id": row["case_id"], "group_size_m": row["group_size_m"],
            "resident_ctas_per_sm": resident,
            "active_ctas": row["active_ctas"],
            "ncu_dram_read_bytes": actual_dram,
            "wave132_operand_dram_bytes": predictions["132"],
            "wave264_operand_dram_bytes": predictions["264"],
            "wave132_fraction_of_ncu_read": predictions["132"] / actual_dram,
            "wave264_fraction_of_ncu_read": predictions["264"] / actual_dram,
        })
    if len(rows) != 39:
        raise ValueError("BM128 qualified case count changed")
    by_group = {}
    for group in sorted({row["group_size_m"] for row in rows}):
        selected = [row for row in rows if row["group_size_m"] == group]
        by_group[str(group)] = {
            "count": len(selected),
            "median_wave132_fraction_of_ncu_read": statistics.median(
                row["wave132_fraction_of_ncu_read"] for row in selected),
            "median_wave264_fraction_of_ncu_read": statistics.median(
                row["wave264_fraction_of_ncu_read"] for row in selected),
            "wave264_higher_dram_case_count": sum(
                row["wave264_operand_dram_bytes"] > row["wave132_operand_dram_bytes"]
                for row in selected),
            "wave264_closer_to_ncu_case_count": sum(
                abs(row["wave264_fraction_of_ncu_read"] - 1)
                < abs(row["wave132_fraction_of_ncu_read"] - 1)
                for row in selected),
            "wave132_closer_to_ncu_case_count": sum(
                abs(row["wave132_fraction_of_ncu_read"] - 1)
                < abs(row["wave264_fraction_of_ncu_read"] - 1)
                for row in selected),
        }
    result = {
        "kind": "h200_moe_bm128_wave_traffic_diagnostic",
        "status": "evaluation_only_no_forward_change",
        "input_pin": _pin(SOURCE), "qualified_case_count": len(rows),
        "by_group_size_m": by_group, "rows": rows,
        "interpretation": (
            "The source replay advances all CTAs in each wave at the same K step. "
            "Increasing the wave from 132 to the resource-resident 264 changes "
            "cache order, not the issued operand bytes. The effect changes "
            "sign across source groups and cannot justify a uniform occupancy "
            "substitution. Neither order proves actual CTA phase alignment; "
            "measured DRAM reads also include non-operand traffic. No "
            "production cache parameter is selected."),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "by_group_size_m": by_group},
                     sort_keys=True))


if __name__ == "__main__":
    main()
