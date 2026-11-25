#!/usr/bin/env python3
"""Summarise an nvidia-smi clock log taken during a whole-machine H200 job.

H200 clocks cannot be locked, so multi-GPU jobs sample
``nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,
utilization.gpu,power.draw,temperature.gpu,clocks_event_reasons.active
--format=csv,noheader,nounits -lms 500`` for their whole duration.  This
script reports, per GPU, how many busy samples (utilization > 0) ran at an
SM clock other than the expected 1980 MHz, and — given one or more runner
``report.json`` files whose summary rows carry ``t_start_unix`` /
``t_end_unix`` — lists the rows that overlap such a sample on any GPU, so
they can be excluded or re-measured.  It never edits the reports.

Usage:
    python3 summarize_clock_log.py --clock-log clocks.csv \
        --gpus 0,1,2,3,4,5,6,7 [--report run/report.json ...] \
        --output clock_summary.json
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

EXPECTED_SM_MHZ = 1980
FIELDS = ("timestamp", "index", "clocks_sm", "clocks_mem", "util_gpu",
          "power_w", "temp_c", "event_reasons")


def _parse(path: Path, gpus: set) -> list:
    samples = []
    with path.open() as handle:
        for line_no, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) != len(FIELDS):
                raise ValueError(f"{path}:{line_no}: expected {len(FIELDS)} "
                                 f"fields, got {len(row)}: {row}")
            values = dict(zip(FIELDS, (v.strip() for v in row)))
            index = int(values["index"])
            if index not in gpus:
                continue
            stamp = datetime.strptime(values["timestamp"],
                                      "%Y/%m/%d %H:%M:%S.%f").timestamp()
            samples.append({"t": stamp, "gpu": index,
                            "sm_mhz": int(values["clocks_sm"]),
                            "util": int(values["util_gpu"]),
                            "power_w": values["power_w"],
                            "temp_c": values["temp_c"],
                            "event_reasons": values["event_reasons"]})
    if not samples:
        raise ValueError(f"{path}: no samples for gpus {sorted(gpus)}")
    return samples


def _rows(report: dict):
    for world_size, block in report["world_sizes"].items():
        for row in block["summary"]:
            yield world_size, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-log", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--report", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gpus = {int(g) for g in args.gpus.split(",")}
    samples = _parse(args.clock_log, gpus)
    per_gpu = {}
    for gpu in sorted(gpus):
        mine = [s for s in samples if s["gpu"] == gpu]
        if not mine:
            raise ValueError(f"no samples for gpu {gpu}")
        busy = [s for s in mine if s["util"] > 0]
        off = [s for s in busy if s["sm_mhz"] != EXPECTED_SM_MHZ]
        per_gpu[str(gpu)] = {
            "samples": len(mine), "busy_samples": len(busy),
            "busy_non_1980_samples": len(off),
            # None means the GPU never showed utilization > 0 in the log.
            "busy_sm_mhz_min": (min(s["sm_mhz"] for s in busy)
                                if busy else None),
            "busy_sm_mhz_max": (max(s["sm_mhz"] for s in busy)
                                if busy else None),
            "non_1980_event_reasons": sorted({s["event_reasons"]
                                              for s in off}),
        }
    off_times = [s["t"] for s in samples
                 if s["util"] > 0 and s["sm_mhz"] != EXPECTED_SM_MHZ]
    flagged = []
    for report_path in args.report:
        report = json.loads(report_path.read_text())
        for world_size, row in _rows(report):
            hits = [t for t in off_times
                    if row["t_start_unix"] <= t <= row["t_end_unix"]]
            if hits:
                flagged.append({"report": str(report_path),
                                "world_size": world_size,
                                "row": {k: row[k] for k in row
                                        if k in ("series", "tokens",
                                                 "payload_bytes", "backend")},
                                "non_1980_busy_samples": len(hits)})
    result = {"expected_sm_mhz": EXPECTED_SM_MHZ,
              "clock_log": str(args.clock_log),
              "first_sample_unix": samples[0]["t"],
              "last_sample_unix": samples[-1]["t"],
              "per_gpu": per_gpu, "flagged_rows": flagged}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for gpu, info in per_gpu.items():
        status = "OK" if info["busy_non_1980_samples"] == 0 else "FLAG"
        print(f"gpu{gpu}: {status} busy={info['busy_samples']} "
              f"non1980={info['busy_non_1980_samples']} "
              f"min={info['busy_sm_mhz_min']} max={info['busy_sm_mhz_max']}")
    print(f"flagged rows: {len(flagged)}")


if __name__ == "__main__":
    main()
