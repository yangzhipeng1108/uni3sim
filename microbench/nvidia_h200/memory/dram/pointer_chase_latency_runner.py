#!/usr/bin/env python3
"""Capture the H200 unloaded L2-hit and DRAM-miss pointer-chase latencies.

Runner for the owner copies ``dram_latency.cu`` and ``../l2/l2_latency.cu``
(the A800 sources had no runner).  Both probes time one dependent
``ld.global.cg`` chain with ``%globaltimer``; the DRAM chain touches 256 MiB of
incompressible sectors (more than four times the 60 MiB H200 L2) and the L2
chain 16 MiB after a complete warm walk.  The runner compiles both for sm_90,
refuses a busy device and runs ``PROCESSES`` independent processes of each
probe in alternating order (the A800 "two devices" minimum becomes two
independent processes on the one owned GPU), each under the SM-clock gate.
``%globaltimer`` nanoseconds do not depend on the SM clock; the gate is still
recorded.  No production shape, operator or duration is consumed.
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
SOURCES = {
    "dram": Path(__file__).with_name("dram_latency.cu"),
    "l2": Path(__file__).resolve().parents[1] / "l2" / "l2_latency.cu",
}
EXPECTED_TRIALS = {"dram": 3, "l2": 5}
RUNNER = Path(__file__).resolve()
PROCESSES = 2
TRIAL = re.compile(r"^trial=(\d+) steps=(\d+) elapsed_ns=(\d+) latency_ns=([0-9.]+)$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    out = arguments.output_dir
    out.mkdir(parents=True, exist_ok=True)
    binaries = {}
    for name, source in SOURCES.items():
        binaries[name] = out / f"{name}_latency"
        subprocess.run([str(NVCC), "-O3", "-arch=sm_90", "-cudart", "static",
                        "-o", str(binaries[name]), str(source)], check=True)
    before = gate.require_idle(arguments.device)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    trials, clocks, rejected, headers = [], [], [], {}
    for process in range(PROCESSES):
        order = ("dram", "l2") if process % 2 == 0 else ("l2", "dram")
        for name in order:
            stdout, accepted, rejects = gate.run_clock_gated(
                [str(binaries[name])], device=arguments.device,
                log_stem=out / f"p{process}_{name}")
            clocks.append(dict(accepted, process=process, probe=name))
            rejected.extend(dict(r, process=process, probe=name) for r in rejects)
            lines = stdout.splitlines()
            headers[f"{name}/{process}"] = lines[0]
            parsed = [TRIAL.match(line) for line in lines[1:]]
            if len(parsed) != EXPECTED_TRIALS[name] or not all(parsed):
                raise ValueError(f"unexpected {name} output: {stdout}")
            for match in parsed:
                trials.append({"probe": name, "process": process,
                               "trial": int(match.group(1)), "steps": int(match.group(2)),
                               "elapsed_ns": int(match.group(3)),
                               "latency_ns": int(match.group(3)) / int(match.group(2))})
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    after = gate.require_idle(arguments.device, settle_seconds=30.0)
    measurements = out / "measurements.csv"
    measurements.write_text("probe,process,trial,steps,elapsed_ns,latency_ns\n" + "".join(
        f"{t['probe']},{t['process']},{t['trial']},{t['steps']},{t['elapsed_ns']},"
        f"{t['latency_ns']}\n" for t in trials))
    summary = {}
    for name in SOURCES:
        values = [t["latency_ns"] for t in trials if t["probe"] == name]
        per_process = {str(p): statistics.median(
            t["latency_ns"] for t in trials if t["probe"] == name and t["process"] == p)
            for p in range(PROCESSES)}
        summary[name] = {"median_latency_ns": statistics.median(values),
                         "minimum_latency_ns": min(values),
                         "maximum_latency_ns": max(values),
                         "per_process_median_latency_ns": per_process,
                         "trials": len(values)}
    candidate = {
        "schema": "nvidia_h200_pointer_chase_latency_candidate_v1",
        "hardware_id": "nvidia_h200",
        "production_or_evaluation_duration_consumed": False,
        "probe_contract": {"load": "ld.global.cg.s32 single dependent chain, 1x1 launch",
                           "timer_domain": "globaltimer_ns",
                           "dram_touched_bytes": 256 * 1024 ** 2, "dram_stride_bytes": 128,
                           "l2_touched_bytes": 16 * 1024 ** 2, "l2_stride_bytes": 32,
                           "processes": PROCESSES},
        "probe_headers": headers,
        "device_state": {"before": before, "after": after},
        "timing": {"started_at": started_at, "completed_at": completed_at},
        "clock_gate": {"boost_mhz": gate.BOOST_MHZ, "accepted": clocks, "rejected": rejected},
        "source_sha256": {name: gate.sha256(path) for name, path in SOURCES.items()},
        "runner_sha256": gate.sha256(RUNNER),
        "binary_sha256": {name: gate.sha256(path) for name, path in binaries.items()},
        "measurements_sha256": gate.sha256(measurements),
        "summary": summary,
    }
    (out / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1))
    print("accepted", len(clocks), "rejected", len(rejected))


if __name__ == "__main__":
    main()
