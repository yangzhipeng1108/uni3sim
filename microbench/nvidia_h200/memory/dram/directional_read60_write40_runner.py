#!/usr/bin/env python3
"""Capture the H200 exact 3:2 (read60_write40) mixed L2 and DRAM stream sweeps.

Runner for the owner copy ``directional_bandwidth.cu`` compiled with
``-DDIRECTIONAL_ENABLE_READ60_WRITE40=1`` (the A800 capture was a manual
``--single read60_write40 GRID`` loop recorded in its run record).  Two builds
are captured exactly as on A800:

* ``l2``: ``-DDIRECTIONAL_BUFFER_BYTES=15728640 -DDIRECTIONAL_ITERATIONS=256``
  (15 MiB source + 15 MiB destination, 30 MiB < 60 MiB H200 L2);
* ``dram``: the default 2 GiB buffers and 4 iterations (far beyond L2).

Each grid of the 132-multiple sweep runs as its own probe process (7 trials);
the sweep is repeated in ``ROTATIONS`` independent rotations with reversed grid
order (the A800 one-device capture becomes two independent rotations on the
owned GPU).  Every process is clock-gated.  No production shape, operator or
duration is consumed.
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_gated_capture as gate  # noqa: E402

NVCC = gate.NVCC
SOURCE = Path(__file__).with_name("directional_bandwidth.cu")
RUNNER = Path(__file__).resolve()
GRIDS = (132, 264, 528, 1056, 1024, 2112)
ROTATIONS = 2
TRIALS = 7
BUILDS = {
    "l2": ["-DDIRECTIONAL_BUFFER_BYTES=15728640", "-DDIRECTIONAL_ITERATIONS=256"],
    "dram": [],
}
HEADER = ("mode,read_fraction,grid,block,iterations,bytes_per_iteration,"
          "trial,event_ms,read_gbs,write_gbs,total_gbs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build", choices=sorted(BUILDS), required=True)
    arguments = parser.parse_args()
    out = arguments.output_dir
    out.mkdir(parents=True, exist_ok=True)
    binary = out / f"directional_bandwidth_{arguments.build}"
    compile_log = out / "compile.log"
    result = subprocess.run(
        [str(NVCC), "-O3", "-std=c++17", "-arch=sm_90", "-cudart", "static",
         "-lineinfo", "-Xptxas=-v", "-DDIRECTIONAL_ENABLE_READ60_WRITE40=1",
         *BUILDS[arguments.build], "-o", str(binary), str(SOURCE)],
        check=True, text=True, capture_output=True)
    compile_log.write_text(result.stdout + result.stderr)
    before = gate.require_idle(arguments.device)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rows, clocks, rejected, stderr_lines = [], [], [], []
    for rotation in range(ROTATIONS):
        order = GRIDS if rotation % 2 == 0 else GRIDS[::-1]
        for grid in order:
            stem = out / f"r{rotation}_g{grid}"
            stdout, accepted, rejects = gate.run_clock_gated(
                [str(binary), "--single", "read60_write40", str(grid)],
                device=arguments.device, log_stem=stem)
            clocks.append(dict(accepted, rotation=rotation, grid=grid))
            rejected.extend(dict(r, rotation=rotation, grid=grid) for r in rejects)
            stderr = stem.with_name(f"{stem.name}.stderr").read_text()
            if "validation=pass" not in stderr:
                raise RuntimeError(f"validation missing for grid {grid}: {stderr}")
            stderr_lines.append(stderr.strip())
            lines = stdout.splitlines()
            if lines[0] != HEADER or len(lines) != TRIALS + 1:
                raise ValueError(f"unexpected output for grid {grid}")
            for row in csv.DictReader(lines):
                rows.append(dict(row, rotation=rotation))
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    after = gate.require_idle(arguments.device, settle_seconds=30.0)
    raw = out / "raw.csv"
    raw.write_text("rotation," + HEADER + "\n" + "".join(
        f"{r['rotation']}," + ",".join(r[k] for k in HEADER.split(",")) + "\n" for r in rows))
    (out / "stderr.log").write_text("\n".join(stderr_lines) + "\n")
    candidate = {
        "schema": "nvidia_h200_directional_read60_write40_candidate_v1",
        "hardware_id": "nvidia_h200",
        "build": arguments.build,
        "build_flags": ["-DDIRECTIONAL_ENABLE_READ60_WRITE40=1", *BUILDS[arguments.build]],
        "production_or_evaluation_duration_consumed": False,
        "grids": list(GRIDS), "rotations": ROTATIONS, "trials_per_process": TRIALS,
        "device_state": {"before": before, "after": after},
        "timing": {"started_at": started_at, "completed_at": completed_at},
        "clock_gate": {"boost_mhz": gate.BOOST_MHZ, "accepted": clocks, "rejected": rejected},
        "source_sha256": gate.sha256(SOURCE), "runner_sha256": gate.sha256(RUNNER),
        "binary_sha256": gate.sha256(binary), "raw_sha256": gate.sha256(raw),
        "stderr_sha256": gate.sha256(out / "stderr.log"),
        "compile_log_sha256": gate.sha256(compile_log),
        "validation": "pass",
    }
    (out / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    for grid in GRIDS:
        values = [float(r["total_gbs"]) for r in rows if int(r["grid"]) == grid]
        median = statistics.median(values)
        print(grid, len(values), round(median, 3),
              round((max(values) - min(values)) / median * 100, 3))
    print("accepted", len(clocks), "rejected", len(rejected))


if __name__ == "__main__":
    main()
