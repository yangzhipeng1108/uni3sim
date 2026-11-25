"""Validate independent one-CTA fill capacity samples without cloud imports."""
import csv
import statistics


def analyze(out):
    groups = {(items, repeat): [] for items in (262144, 524288, 1048576)
              for repeat in (0, 1)}
    for row in csv.DictReader(out["execution"]["stdout"].splitlines()):
        if row["repeat"] == "repeat":
            continue
        items = int(row["items"])
        key = (items, int(row["repeat"]))
        if key not in groups:
            raise ValueError("unexpected fill capacity sample domain")
        cycles = int(row["cta_cycles"])
        if cycles <= 0 or int(row["checksum"]) != 2 * items or float(row["event_ms"]) <= 0:
            raise ValueError("invalid fill sample")
        groups[key].append(cycles)
    if len(groups) != 6 or any(len(values) != 14 for values in groups.values()):
        raise ValueError("incomplete fill capacity samples")
    medians = {key: statistics.median(values) for key, values in groups.items()}
    points = []
    qualified = True
    for items in (262144, 524288, 1048576):
        first, second = medians[items, 0], medians[items, 1]
        spread = max(first, second) / min(first, second) - 1
        qualified &= spread <= .03
        cycles = max(first, second)
        points.append({"items": items, "repeat_median_cycles": [first, second],
                       "process_spread": spread, "cycles_per_item": cycles / items,
                       "bytes_per_second": 4 * items * 1.98e9 / cycles})
    return {"mechanism_qualified": qualified, "points": points,
            "production_or_evaluation_duration_consumed": False}
