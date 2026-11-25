#!/usr/bin/env python3
"""Capture the H200 serial same-device D2D-copy sweep with ``d2d_copy.cu``.

Runner for the owner copy (the A800 source had no runner).  Compiles for
sm_90, refuses a busy device and runs ``PROCESSES`` independent probe
processes under the SM-clock gate, writing ``raw.csv`` and ``candidate.json``.
No production shape, operator or duration is consumed.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "memory"))
import clock_gated_capture as gate  # noqa: E402

NVCC = gate.NVCC
SOURCE = Path(__file__).with_name("d2d_copy.cu")
RUNNER = Path(__file__).resolve()
PROCESSES = 2
HEADER = "run_id,trial,order_position,payload_bytes,source_role,destination_role,event_ns,validation"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    out = arguments.output_dir
    out.mkdir(parents=True, exist_ok=True)
    binary = out / "d2d_copy"
    subprocess.run([str(NVCC), "-O3", "-std=c++14", "-arch=sm_90", "-cudart", "static",
                    "-lineinfo", "-o", str(binary), str(SOURCE)], check=True)
    before = gate.require_idle(arguments.device)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    lines, clocks, rejected = [], [], []
    for process in range(PROCESSES):
        stdout, accepted, rejects = gate.run_clock_gated(
            [str(binary), "--run-id", f"h200_p{process}"], device=arguments.device,
            log_stem=out / f"p{process}")
        clocks.append(dict(accepted, process=process))
        rejected.extend(dict(r, process=process) for r in rejects)
        output = stdout.splitlines()
        if output[0] != HEADER:
            raise ValueError("unexpected d2d header")
        lines.extend(output[1:])
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    after = gate.require_idle(arguments.device, settle_seconds=30.0)
    raw = out / "raw.csv"
    raw.write_text(HEADER + "\n" + "\n".join(lines) + "\n")
    candidate = {
        "schema": "nvidia_h200_d2d_copy_candidate_v1", "hardware_id": "nvidia_h200",
        "production_or_evaluation_duration_consumed": False, "processes": PROCESSES,
        "device_state": {"before": before, "after": after},
        "timing": {"started_at": started_at, "completed_at": completed_at},
        "clock_gate": {"boost_mhz": gate.BOOST_MHZ, "accepted": clocks, "rejected": rejected},
        "source_sha256": gate.sha256(SOURCE), "runner_sha256": gate.sha256(RUNNER),
        "binary_sha256": gate.sha256(binary), "raw_sha256": gate.sha256(raw),
    }
    (out / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    print("rows", len(lines), "accepted", len(clocks), "rejected", len(rejected))


if __name__ == "__main__":
    main()
