#!/usr/bin/env python3
"""Derive the H200 ``memcpy_dtod_floor_ns`` decision from the D2D-copy capture.

Applies the contract stated in ``d2d_copy.cu``: the small-payload cells may
establish a fixed completion floor only if their raw interquartile intervals
have a common intersection, and the frozen D2D service term B/C_R + B/C_W is
checked independently with the 64, 128 and 256 MiB cells without refitting
its slope.  A floor is published only when both hold and the large-payload
residual does not grow with payload; otherwise the parameter stays null
(unsupported), as on A800.  C_R is the measured H200 read capacity
(dram_read_capacity_protocol_sm90.json); C_W is the H200 configuration's
dram_write_capacity_per_clk x dram clock (1536 B x 3.125 GHz, the public-spec
proxy; no H200 write capacity is measured in WP4).  No production shape,
operator or duration is consumed.
"""

import csv
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "d2d_copy_20260925"
RAW = ROOT / "profiler_output/microbench/nvidia_h200" / RUN_ID
REFS = ROOT / "microbench/nvidia_h200/references"
SMALL_PAYLOAD_MAX_BYTES = 65536
LARGE_PAYLOADS = (64 << 20, 128 << 20, 256 << 20)
WRITE_CAPACITY_BYTES_PER_S = 1536 * 3.125e9


def main():
    candidate = json.loads((RAW / "candidate.json").read_text())
    capacity = json.loads((REFS / "dram_read_capacity_protocol_sm90.json").read_text())
    read_capacity = capacity["dram_read_capacity"]["capacity_bytes_per_s"]
    rows = list(csv.DictReader((RAW / "raw.csv").open()))
    cells = {}
    for row in rows:
        if row["validation"] != "pass":
            raise ValueError("unvalidated observation")
        cells.setdefault(int(row["payload_bytes"]), []).append(float(row["event_ns"]))
    summary = {}
    for payload, values in sorted(cells.items()):
        q1, median, q3 = statistics.quantiles(values, n=4)
        summary[str(payload)] = {"median_ns": median, "q1_ns": q1, "q3_ns": q3,
                                 "observations": len(values)}
    small = [p for p in cells if p <= SMALL_PAYLOAD_MAX_BYTES]
    low = max(summary[str(p)]["q1_ns"] for p in small)
    high = min(summary[str(p)]["q3_ns"] for p in small)
    intersection = low <= high
    residuals = {}
    for payload in LARGE_PAYLOADS:
        service_ns = (payload / read_capacity + payload / WRITE_CAPACITY_BYTES_PER_S) * 1e9
        residuals[str(payload)] = summary[str(payload)]["median_ns"] - service_ns
    values = [residuals[str(p)] for p in LARGE_PAYLOADS]
    residual_grows = values[0] < values[1] < values[2]
    floor = None
    if intersection and not residual_grows:
        floor = statistics.median(
            statistics.median(cells[p]) for p in small)
    accepted = candidate["clock_gate"]["accepted"]
    reference = {
        "schema": "nvidia_h200_d2d_copy_floor_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "production_or_evaluation_duration_consumed": False,
        "cells": summary,
        "small_payload_max_bytes": SMALL_PAYLOAD_MAX_BYTES,
        "small_payload_iqr_intersection_ns": [low, high] if intersection else None,
        "service_check": {
            "read_capacity_bytes_per_s": read_capacity,
            "write_capacity_bytes_per_s": WRITE_CAPACITY_BYTES_PER_S,
            "write_capacity_source": "configs/hardware/nvidia_h200.yaml dram_write_capacity_per_clk "
                                     "1536 x 3.125 GHz (public-spec proxy, not measured in WP4)",
            "residual_ns_by_payload_bytes": residuals,
            "residual_grows_with_payload": residual_grows,
        },
        "memcpy_dtod_floor_ns": floor,
        "decision": ("unsupported: the large-payload residual over the frozen service term grows "
                     "with payload, so no fixed floor composes with it (same outcome as A800)"
                     if floor is None else "floor accepted"),
        "provenance": {
            "source": "microbench/nvidia_h200/runtime/d2d_copy/d2d_copy.cu",
            "source_sha256": candidate["source_sha256"],
            "runner": "microbench/nvidia_h200/runtime/d2d_copy/d2d_copy_runner.py",
            "runner_sha256": candidate["runner_sha256"],
            "binary_sha256": candidate["binary_sha256"],
            "raw": f"profiler_output/microbench/nvidia_h200/{RUN_ID}",
            "raw_sha256": candidate["raw_sha256"],
            "clock": {"boost_mhz": candidate["clock_gate"]["boost_mhz"],
                      "accepted_captures": len(accepted),
                      "rejected_captures": len(candidate["clock_gate"]["rejected"]),
                      "busy_sm_clock_mhz_min": min(a["busy_sm_clock_mhz_min"] for a in accepted)},
            "device_index": 1,
            "device_uuid": candidate["device_state"]["before"]["uuid"],
            "git_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                         check=True, text=True, capture_output=True).stdout.strip(),
        },
    }
    path = REFS / "d2d_copy_floor_sm90.json"
    path.write_text(json.dumps(reference, indent=2) + "\n")
    print("wrote", path.relative_to(ROOT), "floor", floor, "residuals", residuals,
          "iqr", reference["small_payload_iqr_intersection_ns"])


if __name__ == "__main__":
    main()
