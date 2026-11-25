"""Validate the independent E128 CUB BlockScan capture without cloud imports."""
import json
import statistics


def analyze(out):
    values = []
    for item in out["processes"]:
        rows = [json.loads(line) for line in item["measurement"]["stdout"].splitlines()
                if line.startswith('{"experts":')]
        if len(rows) != 1:
            raise ValueError("missing numerical fixture")
        row = rows[0]
        if (row["experts"], row["threads"], row["repeats"], row["cpu_reference_pass"]) != (128, 1024, 63, True):
            raise ValueError("changed numerical identity")
        if row["cycles"] <= 0:
            raise ValueError("invalid cycles")
        values.append(row["cycles"] / 63)
    slower = max(values) / min(values) - 1
    qualified = slower <= .03
    return {"mechanism_qualified": qualified, "cycles_per_scan_by_process": values,
            "slower_process_median_cycles_per_scan": statistics.median(values),
            "process_spread": slower,
            "production_or_evaluation_duration_consumed": False}
