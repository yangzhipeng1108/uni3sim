"""Validate exact-geometry two-read/one-write stream capacities."""
import csv
import math
import statistics

GRIDS = (131072, 262144, 524288)
BLOCKS = (96, 128)


def analyze(out):
    groups = {(grid, block, process): [] for grid in GRIDS for block in BLOCKS
              for process in (0, 1)}
    for process, run in enumerate(out["runs"]):
        if run["exit_code"] != 0:
            raise ValueError("failed mixed-stream process")
        for row in csv.DictReader(run["stdout"].splitlines()):
            key = (int(row["grid"]), int(row["block"]), process)
            if key not in groups:
                raise ValueError("unexpected mixed-stream cell")
            if (int(row["trial"]) < 0 or int(row["read_bytes"]) <= 0
                    or int(row["write_bytes"]) <= 0 or not math.isfinite(float(row["event_ms"]))
                    or float(row["event_ms"]) <= 0):
                raise ValueError("invalid mixed-stream sample")
            groups[key].append((int(row["trial"]), float(row["event_ms"]),
                                int(row["read_bytes"]), int(row["write_bytes"])))
    if any(len(values) != 7 or {trial for trial, *_ in values} != set(range(7))
           for values in groups.values()):
        raise ValueError("incomplete mixed-stream samples")
    points = []
    qualified = True
    for grid in GRIDS:
        for block in BLOCKS:
            medians = [statistics.median(ms for _, ms, *_ in groups[grid, block, process])
                       for process in (0, 1)]
            spread = max(medians) / min(medians) - 1
            qualified &= spread <= .03
            read, write = groups[grid, block, 0][0][2:]
            if any((r, w) != (read, write)
               for process in (0, 1) for _, _, r, w in groups[grid, block, process]):
                raise ValueError("mixed-stream byte identity changed")
            points.append({"grid": grid, "block": block,
                           "process_median_ms": medians, "process_spread": spread,
                           "logical_bytes_per_second": (read + write) * 1e3 / max(medians)})
    return {"mechanism_qualified": qualified, "points": points,
            "production_or_evaluation_duration_consumed": False}
