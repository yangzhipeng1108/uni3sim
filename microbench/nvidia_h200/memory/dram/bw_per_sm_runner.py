#!/usr/bin/env python3
"""Capture the H200 per-SM DRAM stream sweep (.cg and .ca) behind
``dram_protocol_efficiency*``.

Owner runner for ``bw_per_sm.cu`` (the A800 source had no runner; the A800
configuration comment records a single full-chip ~1.79 TB/s reading for both
cache operators).  The runner compiles the owner copy for sm_90, refuses a
busy device, and runs ``PROCESSES`` independent probe processes for each
threads-per-block value, alternating the order, each under the SM-clock gate.
It writes ``measurements.csv`` and ``candidate.json`` into ``--output-dir``.
No production shape, operator or duration is consumed.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_gated_capture as gate  # noqa: E402

NVCC = gate.NVCC
SOURCE = Path(__file__).with_name("bw_per_sm.cu")
RUNNER = Path(__file__).resolve()
BUFFER_MB = 2048
THREADS_PER_BLOCK = (256, 1024)
PROCESSES = 5
ROW = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s*$")


def parse(stdout):
    rows = []
    label = None
    for line in stdout.splitlines():
        if line.strip() in ("L1-bypass", "L1-cached"):
            label = line.strip()
            continue
        match = ROW.match(line)
        if match:
            if label is None:
                raise ValueError("row before sweep label")
            rows.append({"path": label, "active_sms": int(match.group(1)),
                         "total_gb_per_s": float(match.group(2))})
    if len(rows) != 26:
        raise ValueError(f"expected 26 sweep rows, got {len(rows)}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    out = arguments.output_dir
    out.mkdir(parents=True, exist_ok=True)
    binary = out / "bw_per_sm"
    subprocess.run([str(NVCC), "-O3", "-arch=sm_90", "-cudart", "static",
                    "-o", str(binary), str(SOURCE)], check=True)
    before = gate.require_idle(arguments.device)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    records = []
    rejected = []
    for process in range(PROCESSES):
        order = THREADS_PER_BLOCK if process % 2 == 0 else THREADS_PER_BLOCK[::-1]
        for threads in order:
            stdout, accepted, rejects = gate.run_clock_gated(
                [str(binary), str(BUFFER_MB), str(threads)],
                device=arguments.device,
                log_stem=out / f"p{process}_t{threads}")
            rejected.extend(dict(r, process=process, threads=threads) for r in rejects)
            for row in parse(stdout):
                records.append(dict(row, process=process, threads_per_block=threads))
            records_clock = dict(accepted, process=process, threads=threads)
            records.append({"clock": records_clock})
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    after = gate.require_idle(arguments.device, settle_seconds=30.0)
    rows = [r for r in records if "clock" not in r]
    clocks = [r["clock"] for r in records if "clock" in r]
    measurements = out / "measurements.csv"
    measurements.write_text(
        "process,threads_per_block,path,active_sms,total_gb_per_s\n" + "".join(
            f"{r['process']},{r['threads_per_block']},{r['path']},"
            f"{r['active_sms']},{r['total_gb_per_s']}\n" for r in rows))
    medians = {}
    for threads in THREADS_PER_BLOCK:
        for path in ("L1-bypass", "L1-cached"):
            for sms in sorted({r["active_sms"] for r in rows}):
                cell = [r["total_gb_per_s"] for r in rows
                        if r["threads_per_block"] == threads and r["path"] == path
                        and r["active_sms"] == sms]
                if len(cell) != PROCESSES:
                    raise ValueError(f"missing cell {threads} {path} {sms}")
                medians[f"{threads}/{path}/{sms}"] = {
                    "median_gb_per_s": statistics.median(cell),
                    "relative_spread": (max(cell) - min(cell)) / statistics.median(cell),
                    "samples": cell}
    candidate = {
        "schema": "nvidia_h200_dram_bw_per_sm_candidate_v1",
        "hardware_id": "nvidia_h200",
        "production_or_evaluation_duration_consumed": False,
        "probe_contract": {
            "buffer_bytes": BUFFER_MB * 1024 * 1024,
            "threads_per_block": list(THREADS_PER_BLOCK),
            "processes": PROCESSES,
            "load_instructions": {"L1-bypass": "ld.global.cg.v4.f32",
                                  "L1-cached": "ld.global.ca.v4.f32"},
            "grid": "one CTA per active SM",
            "timer_domain": "cuda_event_gpu_timeline",
        },
        "device_state": {"before": before, "after": after},
        "timing": {"started_at": started_at, "completed_at": completed_at},
        "clock_gate": {"boost_mhz": gate.BOOST_MHZ, "accepted": clocks,
                       "rejected": rejected},
        "source_sha256": gate.sha256(SOURCE),
        "runner_sha256": gate.sha256(RUNNER),
        "binary_sha256": gate.sha256(binary),
        "measurements_sha256": gate.sha256(measurements),
        "medians": medians,
    }
    (out / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    for key, value in medians.items():
        if key.endswith("/132"):
            print(key, round(value["median_gb_per_s"], 1), round(value["relative_spread"], 4))
    print("accepted", len(clocks), "rejected", len(rejected))


if __name__ == "__main__":
    main()
