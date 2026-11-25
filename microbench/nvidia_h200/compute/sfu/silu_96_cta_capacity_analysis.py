"""Validate independent 96-thread register-SiLU capacity samples."""
import csv
import statistics


def analyze(out):
    grids = (1320, 13200, 132000, 524288)
    variants = ("arithmetic_control", "source_silu")
    groups = {(variant, grid, process): [] for variant in variants
              for grid in grids for process in (0, 1)}
    for process, run in enumerate(out["runs"]):
        if run["exit_code"] != 0:
            raise ValueError("failed SiLU capacity process")
        for row in csv.DictReader(run["stdout"].splitlines()):
            key = (row["variant"], int(row["grid"]), process)
            if key not in groups:
                raise ValueError("unexpected SiLU capacity cell")
            if (int(row["block"]) != 96 or int(row["chains_per_lane"]) != 8
                    or int(row["trial"]) < 0 or float(row["event_ms"]) <= 0):
                raise ValueError("invalid SiLU capacity sample")
            groups[key].append((int(row["trial"]), float(row["event_ms"])))
    if any(len(values) != 7 or {trial for trial, _ in values} != set(range(7))
           for values in groups.values()):
        raise ValueError("incomplete SiLU capacity samples")
    medians = {key: statistics.median(event for _, event in values)
               for key, values in groups.items()}
    points = []
    qualified = True
    for grid in grids:
        processes = [medians["source_silu", grid, process] for process in (0, 1)]
        spread = max(processes) / min(processes) - 1
        qualified &= spread <= .03
        elapsed_ms = max(processes)
        points.append({
            "grid": grid,
            "process_median_ms": processes,
            "process_spread": spread,
            "elements_per_second": grid * 96 * 8 * 1e3 / elapsed_ms,
            "control_median_ms": [medians["arithmetic_control", grid, process]
                                  for process in (0, 1)],
        })
    rate = min(point["elements_per_second"] for point in points)
    return {
        "mechanism_qualified": qualified,
        "elements_per_second": rate,
        "points": points,
        "production_or_evaluation_duration_consumed": False,
    }
