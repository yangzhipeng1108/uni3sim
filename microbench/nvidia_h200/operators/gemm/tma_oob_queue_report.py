"""Validate complete independent OOB controls and retain all paired secants."""
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import statistics


def main():
    root = Path(__file__).resolve().parents[4]
    target = root / "profiler_output/microbench/nvidia_h200/tma_oob_queue_retry_20261107"
    preflight = json.loads((target / "preflight.json").read_text())
    raw = json.loads((target / "raw.json").read_text())
    cpu = json.loads((target / "cpu_smoke.json").read_text())
    if raw["build_exit_code"] != 0 or raw["run_exit_code"] != 0 or cpu["exit_code"] != 0:
        raise ValueError("build/run failure; no mechanism conclusion")
    source = (target / "source/translation_unit.cu").read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    if not source_hash == preflight["source_sha256"] == raw["source_sha256"] == cpu["source_sha256"]:
        raise ValueError("source identity mismatch")
    for name, key in (("tma_oob_queue_service.cu", "probe_sha256"),
                      ("tma_queue_service.cu", "helper_sha256"),
                      ("tma_oob_queue_service_modal_runner.py", "runner_sha256")):
        if hashlib.sha256((target / "source" / name).read_bytes()).hexdigest() != preflight[key]:
            raise ValueError("snapshot identity mismatch")
    if raw["runner_sha256"] != preflight["runner_sha256"]:
        raise ValueError("runner changed during capture")
    data = gzip.decompress((target / raw["stdout_path"]).read_bytes())
    sass = gzip.decompress((target / raw["sass_path"]).read_bytes())
    if hashlib.sha256(data).hexdigest() != raw["stdout_sha256"]:
        raise ValueError("measurement content mismatch")
    if hashlib.sha256(sass).hexdigest() != raw["sass_sha256"]:
        raise ValueError("SASS content mismatch")
    fields = ("valid_rows", "box_rows", "grid", "depth", "resident", "rounds")
    rows = [json.loads(line) for line in data.decode().splitlines()]
    observed = {tuple(row[field] for field in fields): row for row in rows}
    shapes = [(valid, box) for valid in (1, 2, 4) for box in (valid, 8)] + [(8, 8)]
    expected = {(valid, box, grid, depth, resident, rounds)
        for (valid, box), grid, depth, resident, rounds in itertools.product(
            shapes, (32, 132), (4, 12), (0, 1), (128, 512))}
    if len(rows) != len(observed) or set(observed) != expected or len(rows) != preflight["expected_controls"]:
        raise ValueError("missing, duplicated or unexpected controls")
    if any(not row["checks_passed"] or row["b_bytes"] != 8192
           or row["validation"] != "full_box_sum_xor_including_zero_fill"
           or not 100 < row["clock_mhz_p50"] < 3000 for row in rows):
        raise ValueError("payload/zero-fill/clock validation failed")
    slopes = {}
    for valid, box in shapes:
        for grid, depth, resident in itertools.product((32, 132), (4, 12), (0, 1)):
            lo = observed[(valid, box, grid, depth, resident, 128)]
            hi = observed[(valid, box, grid, depth, resident, 512)]
            slope = (hi["span_ns_p50"] - lo["span_ns_p50"]) / 384
            if slope <= 0:
                raise ValueError("nonpositive complete-copy secant")
            slopes[(valid, box, grid, depth, resident)] = slope
    pairs = []
    for valid, grid, depth, resident in itertools.product((1, 2, 4), (32, 132), (4, 12), (0, 1)):
        compact = slopes[(valid, valid, grid, depth, resident)]
        zero_fill = slopes[(valid, 8, grid, depth, resident)]
        pairs.append({"valid_rows": valid, "grid": grid, "depth": depth,
            "resident": resident, "compact_ns_per_round": compact,
            "zero_fill_ns_per_round": zero_fill,
            "zero_fill_over_compact": zero_fill / compact,
            "difference_ns_per_round": zero_fill - compact})
    result = {"scope": "Independent copy queue with identical valid global input; compact versus zero-filled boxes. No Tensor/GEMM or new Forward price. All controls and negative paired differences retained; no native descriptor claim.",
        "controls": len(rows), "all_checks_passed": True,
        "clock_mhz_range": [min(row["clock_mhz_p50"] for row in rows), max(row["clock_mhz_p50"] for row in rows)],
        "paired_ratio_median": statistics.median(pair["zero_fill_over_compact"] for pair in pairs),
        "paired_ratio_range": [min(pair["zero_fill_over_compact"] for pair in pairs), max(pair["zero_fill_over_compact"] for pair in pairs)],
        "nonpositive_differences": sum(pair["difference_ns_per_round"] <= 0 for pair in pairs),
        "pairs": pairs, "production_duration_consumed": False,
        "prediction_changed": False}
    (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in result if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
