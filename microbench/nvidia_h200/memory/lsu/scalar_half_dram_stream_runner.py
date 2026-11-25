#!/usr/bin/env python3
"""Capture the H200 scalar-half DRAM source stream.

Runner for the owner copy ``scalar_half_dram_stream.cu`` (the A800 capture
was one manual process recorded in its RUN.md).  The runner compiles for
sm_90, refuses a busy device and runs ``PROCESSES`` independent probe
processes under the SM-clock gate.  It writes ``results.csv`` (the probe's own
rows tagged with the process) and ``candidate.json``.  No production shape,
operator or duration is consumed.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_gated_capture as gate  # noqa: E402

NVCC = gate.NVCC
SOURCE = Path(__file__).with_name("scalar_half_dram_stream.cu")
RUNNER = Path(__file__).resolve()
PROCESSES = 2
RAW_HEADER = "raw,grid,groups_per_cta,trial,bytes,elapsed_ms,gb_per_s,sectors_per_active_sm_s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    out = arguments.output_dir
    out.mkdir(parents=True, exist_ok=True)
    binary = out / "scalar_half_dram_stream"
    compiled = subprocess.run(
        [str(NVCC), "-O3", "-std=c++14", "-arch=sm_90", "-cudart", "static",
         "-Xptxas=-v", "-o", str(binary), str(SOURCE)],
        check=True, text=True, capture_output=True)
    (out / "compile.log").write_text(compiled.stdout + compiled.stderr)
    before = gate.require_idle(arguments.device)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    raw_lines, meta, clocks, rejected = [], {}, [], []
    for process in range(PROCESSES):
        stdout, accepted, rejects = gate.run_clock_gated(
            [str(binary)], device=arguments.device, log_stem=out / f"p{process}")
        clocks.append(dict(accepted, process=process))
        rejected.extend(dict(r, process=process) for r in rejects)
        for line in stdout.splitlines():
            if line.startswith("meta,"):
                _, key, value = line.split(",", 2)
                meta[key] = value
            elif line.startswith("raw,") and line != RAW_HEADER:
                raw_lines.append(f"{process},{line[4:]}")
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    after = gate.require_idle(arguments.device, settle_seconds=30.0)
    results = out / "results.csv"
    results.write_text("process," + RAW_HEADER[4:] + "\n" + "\n".join(raw_lines) + "\n")
    candidate = {
        "schema": "nvidia_h200_scalar_half_dram_stream_candidate_v1",
        "hardware_id": "nvidia_h200",
        "production_or_evaluation_duration_consumed": False,
        "probe_meta": meta, "processes": PROCESSES,
        "device_state": {"before": before, "after": after},
        "timing": {"started_at": started_at, "completed_at": completed_at},
        "clock_gate": {"boost_mhz": gate.BOOST_MHZ, "accepted": clocks, "rejected": rejected},
        "source_sha256": gate.sha256(SOURCE), "runner_sha256": gate.sha256(RUNNER),
        "binary_sha256": gate.sha256(binary), "results_sha256": gate.sha256(results),
        "compile_log_sha256": gate.sha256(out / "compile.log"),
    }
    (out / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta), "rows", len(raw_lines), "accepted", len(clocks),
          "rejected", len(rejected))


if __name__ == "__main__":
    main()
