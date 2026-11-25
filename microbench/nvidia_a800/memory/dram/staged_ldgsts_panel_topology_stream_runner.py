#!/usr/bin/env python3
"""Capture the SM80 staged-LDGSTS ring service across panel topologies.

The probe sweeps the declared operand-pair geometries the SM80 GEMM source
classes use (64x64x64 -> 16 KiB, 64x128x64 -> 24 KiB, 64x256x64 -> 40 KiB)
across CTA grids that keep one resident CTA per SM, and separately measures
what happens when part of the declared panel is issued with a zero source
size.  That zero-fill mode is what a cuBLAS staged GEMM emits for operand rows
outside a logical M smaller than its tile, so it isolates whether a round costs
its declared panel or only the bytes it moves.

The runner compiles every configuration from the committed source, refuses a
busy device, records the device state around the sweep, and writes an
auditable candidate.  No production shape and no production or evaluation
duration is consumed.
"""

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

NVCC = Path("/usr/local/cuda-12.2/bin/nvcc")
CUOBJDUMP = Path("/usr/local/cuda-12.2/bin/cuobjdump")
DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
SOURCE = Path(__file__).with_name("staged_ldgsts_panel_topology_stream.cu")
RUNNER = Path(__file__).resolve()
# Declared operand-pair bytes -> occupancy pad that keeps one resident CTA/SM.
PANELS = {16384: 49152, 24576: 24576, 40960: 16384}
GRIDS = (56, 64, 80, 86, 96, 100, 108)
ROUNDS = (64, 128, 256, 512)
TIERS = ("l2_resident", "dram_unique")
# Zero-filled prefix validation: same declared panel, decreasing real bytes.
ZFILL_PANEL = 24576
ZFILL_GRID = 64
ZFILL_BYTES = (0, 8192, 16384, 24576)
# The A panel of every SM80 64-row / K64 staged class is 64*64*2 bytes.  A
# launch whose logical M covers one tile issues exactly that panel with a zero
# source size, so this is the production-shaped ring: A predicated off, B
# fetched.  Sweep it over every panel and grid.
PRODUCTION_ZFILL_BYTES = 8192
MAXIMUM_RELATIVE_SPREAD = 0.10


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture(command, *, environment=None):
    return subprocess.run(
        command, check=True, text=True, capture_output=True,
        env=environment, timeout=3600)


def _device_state(device):
    result = _capture([
        "nvidia-smi", f"--id={device}",
        "--query-gpu=index,name,uuid,memory.used,utilization.gpu,"
        "clocks.current.sm,clocks.current.memory",
        "--format=csv,noheader,nounits",
    ])
    values = tuple(item.strip() for item in result.stdout.split(","))
    if len(values) != 7:
        raise ValueError("unexpected nvidia-smi device row")
    state = {
        "index": int(values[0]), "name": values[1], "uuid": values[2],
        "memory_used_mib": int(values[3]),
        "gpu_utilization_pct": int(values[4]),
        "sm_clock_mhz": int(values[5]), "memory_clock_mhz": int(values[6]),
    }
    if state["name"] != DEVICE_NAME:
        raise RuntimeError(f"device {device} is not {DEVICE_NAME}")
    return state


def _require_idle(device, *, settle_seconds=0.0):
    deadline = time.monotonic() + settle_seconds
    while True:
        state = _device_state(device)
        if state["gpu_utilization_pct"] == 0 and state["memory_used_mib"] <= 64:
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError(f"device {device} is not idle: {state}")
        time.sleep(0.5)


def _build(output_dir, panel, grid, zfill):
    binary = output_dir / f"panel{panel}_grid{grid}_zfill{zfill}"
    _capture([
        str(NVCC), "-O3", "-arch=sm_80",
        f"-DPROBE_PANEL_BYTES={panel}", "-DPROBE_SLOTS=3",
        f"-DPROBE_PAD_BYTES={PANELS[panel]}", f"-DPROBE_GRID={grid}",
        f"-DPROBE_ZFILL_BYTES={zfill}",
        "-o", str(binary), str(SOURCE)])
    return binary


def _secant_services(rows, panel, grid, zfill):
    """Return the finite intercept and conservative per-round service."""

    services = {}
    unstable = None
    for tier in TIERS:
        medians = {}
        for rounds in ROUNDS:
            cell = [
                float(row["duration_ms"]) * 1.0e6 for row in rows
                if int(row["panel_bytes"]) == panel
                and int(row["grid"]) == grid
                and int(row["zfill_bytes"]) == zfill
                and row["tier"] == tier and int(row["rounds"]) == rounds
            ]
            if not cell:
                raise ValueError(
                    f"missing cell panel={panel} grid={grid} tier={tier} "
                    f"rounds={rounds}")
            median = statistics.median(cell)
            spread = (max(cell) - min(cell)) / median
            if spread > MAXIMUM_RELATIVE_SPREAD:
                # Quarantine the cell rather than abort the sweep: an unstable
                # cell must not be published, but it also must not discard the
                # topologies that did measure cleanly.
                return None, {
                    "panel_bytes": panel, "grid_ctas": grid, "tier": tier,
                    "rounds": rounds, "relative_spread": spread,
                }
            medians[rounds] = median
        # Reference methodology: the intercept runs through the two longest
        # streams; the retained per-round service is the conservative value
        # that no measured completion in the domain falls below.
        slope = (medians[512] - medians[128]) / (512 - 128)
        intercept = medians[128] - 128 * slope
        conservative = min(
            (medians[rounds] - intercept) / rounds for rounds in ROUNDS)
        services[tier] = {
            "finite_intercept_ns": intercept,
            "conservative_ns_per_round": conservative,
            "median_duration_ns": {str(k): v for k, v in medians.items()},
        }
    return services, unstable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    arguments = parser.parse_args()
    if not 3 <= arguments.trials <= 11:
        raise ValueError("trials must be within 3..11")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    configurations = [
        (panel, grid, 0) for panel in sorted(PANELS) for grid in GRIDS
    ] + [
        (panel, grid, PRODUCTION_ZFILL_BYTES)
        for panel in sorted(PANELS) for grid in GRIDS
    ] + [
        (ZFILL_PANEL, ZFILL_GRID, zfill)
        for zfill in ZFILL_BYTES
        if zfill not in (0, PRODUCTION_ZFILL_BYTES)
    ]
    before = _require_idle(arguments.device)
    environment = {
        "CUDA_VISIBLE_DEVICES": str(arguments.device),
        "PATH": "/usr/bin:/bin",
    }
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    header = None
    lines = []
    sass_digests = {}
    for panel, grid, zfill in configurations:
        binary = _build(arguments.output_dir, panel, grid, zfill)
        sass_path = arguments.output_dir / f"{binary.name}.sass"
        sass_path.write_text(
            _capture([str(CUOBJDUMP), "-sass", str(binary)]).stdout,
            encoding="utf-8")
        sass_digests[binary.name] = _sha256(sass_path)
        for rounds in ROUNDS:
            # The probe validates one resident CTA per SM itself and exits
            # non-zero otherwise, so a changed residency fails the capture.
            result = _capture(
                [str(binary), "0", str(arguments.trials), str(rounds)],
                environment=environment)
            output = result.stdout.splitlines()
            if header is None:
                header = output[0]
            elif output[0] != header:
                raise ValueError("probe changed its output schema mid-sweep")
            lines.extend(output[1:])
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    after = _require_idle(arguments.device, settle_seconds=30.0)

    measurements_path = arguments.output_dir / "measurements.csv"
    measurements_path.write_text(
        "\n".join([header] + lines) + "\n", encoding="utf-8")
    rows = list(csv.DictReader([header] + lines))

    topologies = []
    excluded = []
    for panel in sorted(PANELS):
        for grid in GRIDS:
            services, unstable = _secant_services(rows, panel, grid, 0)
            if services is None:
                excluded.append(unstable)
                continue
            topologies.append({
                "operand_pair_bytes": panel,
                "grid_ctas": grid,
                "occupancy_pad_bytes": PANELS[panel],
                "services": services,
            })
    production_shaped = []
    for panel in sorted(PANELS):
        for grid in GRIDS:
            services, unstable = _secant_services(
                rows, panel, grid, PRODUCTION_ZFILL_BYTES)
            if services is None:
                excluded.append(unstable)
                continue
            production_shaped.append({
                "operand_pair_bytes": panel,
                "grid_ctas": grid,
                "zfill_bytes": PRODUCTION_ZFILL_BYTES,
                "fetched_bytes": panel - PRODUCTION_ZFILL_BYTES,
                "services": services,
            })
    zero_fill = []
    for zfill in ZFILL_BYTES:
        services, unstable = _secant_services(
            rows, ZFILL_PANEL, ZFILL_GRID, zfill)
        if services is None:
            excluded.append(unstable)
            continue
        zero_fill.append({
            "operand_pair_bytes": ZFILL_PANEL,
            "grid_ctas": ZFILL_GRID,
            "zfill_bytes": zfill,
            "fetched_bytes": ZFILL_PANEL - zfill,
            "services": services,
        })

    candidate = {
        "schema": "nvidia_a800_staged_ldgsts_panel_topology_candidate_v1",
        "hardware_id": "nvidia_a800",
        "mechanism": "staged_ldgsts_panel_topology_stream",
        "production_or_evaluation_duration_consumed": False,
        "probe_contract": {
            "threads_per_cta": 128,
            "complete_operand_pair_slots": 3,
            "load_instruction": "cp.async.cg.shared.global",
            "zero_fill_instruction": "cp.async.cg.shared.global ..., 0",
            "resident_ctas_per_sm": 1,
            "panels": {str(k): v for k, v in PANELS.items()},
            "grids": list(GRIDS),
            "rounds": list(ROUNDS),
            "zfill_bytes": list(ZFILL_BYTES),
            "timer_domain": "cuda_event_gpu_timeline",
        },
        "device_state": {"before": before, "after": after},
        "timing": {"started_at": started_at, "completed_at": completed_at},
        "trials": arguments.trials,
        "source_sha256": _sha256(SOURCE),
        "runner_sha256": _sha256(RUNNER),
        "sass_sha256": sass_digests,
        "measurements_sha256": _sha256(measurements_path),
        "topologies": topologies,
        "production_shaped_topologies": production_shaped,
        "zero_fill_validation": zero_fill,
        "excluded_unstable_cells": excluded,
    }
    candidate_path = arguments.output_dir / "candidate.json"
    candidate_path.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"wrote {candidate_path}")


if __name__ == "__main__":
    main()
