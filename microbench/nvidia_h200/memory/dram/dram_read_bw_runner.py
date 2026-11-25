#!/usr/bin/env python3
"""Capture the H200 saturated DRAM read capacity with ``dram_read_bw.cu``.

``bw_per_sm.cu`` launches one CTA per SM and, even at 1024 threads, holds too
few bytes in flight to saturate HBM3e (its per-SM rate still falls at 132
SMs), so the saturated read capacity that feeds ``dram_read_capacity_per_clk``
and the finite-stream ramp denominator is taken from the existing H200 probe
``dram_read_bw.cu`` (4 GiB buffer, 16-byte streaming loads, 1..8 resident
512-thread CTAs per SM).  The source is unchanged; this runner only adds the
clock gate and ``PROCESSES`` independent processes.  No production shape,
operator or duration is consumed.
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
SOURCE = Path(__file__).with_name("dram_read_bw.cu")
RUNNER = Path(__file__).resolve()
BUFFER_GIB = 4
PASSES = 20
PROCESSES = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    out = arguments.output_dir
    out.mkdir(parents=True, exist_ok=True)
    binary = out / "dram_read_bw"
    subprocess.run([str(NVCC), "-O3", "-arch=sm_90", "-cudart", "static",
                    "-o", str(binary), str(SOURCE)], check=True)
    before = gate.require_idle(arguments.device)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rows, clocks, rejected = [], [], []
    for process in range(PROCESSES):
        stdout, accepted, rejects = gate.run_clock_gated(
            [str(binary), str(BUFFER_GIB), str(PASSES)],
            device=arguments.device, log_stem=out / f"p{process}")
        clocks.append(dict(accepted, process=process))
        rejected.extend(dict(r, process=process) for r in rejects)
        lines = stdout.splitlines()
        header = lines.index("ctas_per_sm,grid,block,passes,ms,read_GBps")
        for row in csv.DictReader(lines[header:]):
            rows.append({"process": process, "ctas_per_sm": int(row["ctas_per_sm"]),
                         "grid": int(row["grid"]), "block": int(row["block"]),
                         "passes": int(row["passes"]), "ms": float(row["ms"]),
                         "read_gb_per_s": float(row["read_GBps"])})
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    after = gate.require_idle(arguments.device, settle_seconds=30.0)
    measurements = out / "measurements.csv"
    measurements.write_text("process,ctas_per_sm,grid,block,passes,ms,read_gb_per_s\n" + "".join(
        f"{r['process']},{r['ctas_per_sm']},{r['grid']},{r['block']},{r['passes']},"
        f"{r['ms']},{r['read_gb_per_s']}\n" for r in rows))
    medians = {}
    for k in sorted({r["ctas_per_sm"] for r in rows}):
        cell = [r["read_gb_per_s"] for r in rows if r["ctas_per_sm"] == k]
        if len(cell) != PROCESSES:
            raise ValueError(f"missing cell ctas_per_sm={k}")
        median = statistics.median(cell)
        medians[str(k)] = {"median_gb_per_s": median, "samples": cell,
                           "relative_spread": (max(cell) - min(cell)) / median}
    candidate = {
        "schema": "nvidia_h200_dram_read_bw_candidate_v1",
        "hardware_id": "nvidia_h200",
        "production_or_evaluation_duration_consumed": False,
        "probe_contract": {"buffer_bytes": BUFFER_GIB * 1024 ** 3, "passes": PASSES,
                           "block_threads": 512, "load": "__ldcs int4 (ld.global.cs.v4)",
                           "processes": PROCESSES,
                           "timer_domain": "cuda_event_gpu_timeline"},
        "device_state": {"before": before, "after": after},
        "timing": {"started_at": started_at, "completed_at": completed_at},
        "clock_gate": {"boost_mhz": gate.BOOST_MHZ, "accepted": clocks, "rejected": rejected},
        "source_sha256": gate.sha256(SOURCE), "runner_sha256": gate.sha256(RUNNER),
        "binary_sha256": gate.sha256(binary), "measurements_sha256": gate.sha256(measurements),
        "medians_by_ctas_per_sm": medians,
    }
    (out / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    for k, v in medians.items():
        print(k, round(v["median_gb_per_s"], 1), round(v["relative_spread"], 4))
    print("accepted", len(clocks), "rejected", len(rejected))


if __name__ == "__main__":
    main()
