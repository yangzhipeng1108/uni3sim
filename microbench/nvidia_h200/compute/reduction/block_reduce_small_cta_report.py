"""Validate a complete pure-CUB capture and report source/control limits."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import re
import statistics


def median_ci(values, seed):
    rng = random.Random(seed)
    medians = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(2000))
    return medians[49], medians[1949]


def summarize(directory, reference_path):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    receipt = json.loads((directory / "client_receipt.json").read_text())
    if manifest["status"] != "complete" or receipt["status"] != "complete":
        raise ValueError("incomplete capture is not a completed mechanism matrix")
    for name, digest in receipt["file_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError("returned file identity changed: " + name)
    for name, digest in manifest["source_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError("source identity changed: " + name)
    build = gzip.decompress((directory / "build.stderr.gz").read_bytes()).decode()
    spills = [int(value) for value in re.findall(r"(\d+) bytes spill (?:loads|stores)", build)]
    if not spills or any(spills):
        raise ValueError("compiler spill check failed")
    rows = []
    for process in (0, 1):
        for index, (hidden, threads, grid) in enumerate(manifest["selection"]["cells"]):
            label = f"p{process}_h{hidden}_b{threads}_g{grid}"
            raw = gzip.decompress((directory / (label + ".stdout.gz")).read_bytes())
            records = list(csv.DictReader(io.StringIO(raw.decode())))
            by_sample = {}
            for record in records:
                if record["numerical_validation"] != "pass":
                    raise ValueError("numerical check failed")
                sample = int(record["sample"])
                if sample not in by_sample:
                    by_sample[sample] = []
                by_sample[sample].append(record)
            if set(by_sample) != set(range(manifest["selection"]["samples_per_cell"])):
                raise ValueError("sample matrix changed")
            bodies, spans, sm_counts = [], [], []
            for sample in sorted(by_sample):
                items = by_sample[sample]
                if (sum(int(item["ctas"]) for item in items) != grid
                        or len({int(item["smid"]) for item in items}) != len(items)
                        or any((int(item["hidden"]), int(item["threads"]), int(item["grid"]))
                            != (hidden, threads, grid) for item in items)):
                    raise ValueError("physical CTA/SM/cell identity changed")
                sm_counts.append(len(items))
                bodies.append(statistics.median(int(item["median_block_cycles"]) for item in items))
                spans.append(statistics.median(int(item["span_cycles"]) for item in items))
            metadata = gzip.decompress((directory / (label + ".stderr.gz")).read_bytes()).decode().splitlines()
            resources, = (json.loads(line.removeprefix("RESOURCE ")) for line in metadata if line.startswith("RESOURCE "))
            if resources["threads"] != threads or resources["hidden"] != hidden:
                raise ValueError("resource metadata changed")
            body_ci = median_ci(bodies, 1000 * process + index)
            span_ci = median_ci(spans, 2000 * process + index)
            body, span = statistics.median(bodies), statistics.median(spans)
            rows.append({"process": process, "hidden": hidden, "threads": threads, "grid": grid,
                "samples": len(by_sample), "sm_count_range": [min(sm_counts), max(sm_counts)],
                "full_sm_coverage_passed": min(sm_counts) == (1 if grid == 1 else 132),
                "body_median_cycles": body, "span_median_cycles": span,
                "body_median_95_ci": body_ci, "span_median_95_ci": span_ci,
                "ci_3_percent_passed": (body_ci[1] - body_ci[0]) / (2 * body) <= .03
                    and (span_ci[1] - span_ci[0]) / (2 * span) <= .03,
                "resources": resources})
    process_checks = []
    for hidden, threads, grid in manifest["selection"]["cells"]:
        matches = [row for row in rows if (row["hidden"], row["threads"], row["grid"]) == (hidden, threads, grid)]
        left, right = matches
        deltas = {field: abs(left[field] - right[field]) / left[field] for field in ("body_median_cycles", "span_median_cycles")}
        process_checks.append({"hidden": hidden, "threads": threads, "grid": grid,
            "relative_delta": deltas, "passed": all(value <= .03 for value in deltas.values())})
    parameters = []
    for hidden, threads in ((128, 16), (4096, 256)):
        cells = [row for row in rows if row["hidden"] == hidden and row["threads"] == threads]
        resident_values = {row["resources"]["resident_ctas_per_sm"] for row in cells}
        resident, = resident_values
        first = statistics.median(row["body_median_cycles"] for row in cells if row["grid"] == 1)
        resident_span = statistics.median(row["span_median_cycles"] for row in cells if row["grid"] == resident * 132)
        maximum = max(row["grid"] for row in cells)
        maximum_span = statistics.median(row["span_median_cycles"] for row in cells if row["grid"] == maximum)
        resident_service = (resident_span - first) / (resident - 1)
        queued_service = (maximum_span - resident_span) / (maximum / 132 - resident)
        if not 0 < resident_service < first or not 0 < queued_service < first:
            raise ValueError("mechanism service derivation failed")
        parameters.append({"hidden": hidden, "threads": threads, "resident_ctas_per_sm": resident,
            "first_block_latency_cycles": first, "resident_block_service_cycles_per_sm": resident_service,
            "queued_block_service_cycles_per_sm": queued_service,
            "maximum_measured_grid": maximum, "status": "diagnostic_only_not_transfer_qualified"})
    reference = json.loads(Path(reference_path).read_text())
    old, = (row for row in reference["parameters"] if row["threads"] == 256)
    control, = (row for row in parameters if row["threads"] == 256)
    differences = {field: abs(control[field] - old[field]) / old[field] for field in (
        "first_block_latency_cycles", "resident_block_service_cycles_per_sm", "queued_block_service_cycles_per_sm")}
    old_control_passed = all(value <= .03 for value in differences.values())
    return {"status": "complete_capture_unqualified_for_parameter_transfer" if not old_control_passed else "complete_capture_pending_source_transfer_review",
        "app_id": receipt["app_id"], "source_commit": receipt["source_commit"], "rows": rows,
        "process_checks": process_checks, "diagnostic_cycle_parameters": parameters,
        "gates": {"all_cells_complete_and_numerical_valid": True, "zero_spills": True,
            "full_sm_coverage": all(row["full_sm_coverage_passed"] for row in rows),
            "median_ci_3_percent": all(row["ci_3_percent_passed"] for row in rows),
            "process_median_3_percent": all(row["passed"] for row in process_checks),
            "old_H4096_block256_control_3_percent": old_control_passed},
        "old_control_relative_difference": differences,
        "gpu_function_wall_s": manifest["gpu_function_wall_s"],
        "clock_scope": "clock64 within the same SM; GPU entry345MHz/end1980MHz do not establish per-call clocks; no conversion to measured nanoseconds",
        "transfer_limits": ["old source hash and nvcc12.9.86 match, but old control does not",
            "CUB header version/include-tree identity was not separately emitted; compiler version alone does not establish it",
            "driver575.57.08 versus610.57.04, warmup20 versus64 and lineinfo/protocol differ; cause unassigned",
            "CUDA occupancy32/8 describes this isolated primitive, not a full Norm body",
            "no reference/HWSpec/Forward parameter is adopted; no old endpoint is borrowed for block16"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.raw_root, args.reference)
    (args.raw_root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "gates", "old_control_relative_difference")}))
