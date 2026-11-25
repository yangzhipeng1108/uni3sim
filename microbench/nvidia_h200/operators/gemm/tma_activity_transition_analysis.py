"""Validate and compare all finite TMA activity-transition controls."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def analyze(directory, *, full_profile=False, wide_requests=False):
    raw = json.loads((directory / "raw.json").read_text())
    if raw["production_duration_consumed"] or len(raw["captures"]) != 2:
        raise ValueError("two independent activity domains required")
    source = directory / "source/tma_queue_service.cu"
    if hashlib.sha256(source.read_bytes()).hexdigest() != raw["source_sha256"]:
        raise ValueError("source identity mismatch")
    bs = (4096, 8192, 16384) if full_profile else (8192, 16384)
    if wide_requests:
        if not full_profile:
            raise ValueError("wide requests require the complete control grid")
        bs += (32768,)
    depths = (4, 6, 8, 12, 16) if full_profile else (4, 12)
    grids = (32, 64, 96, 132) if full_profile else (32, 132)
    expected = {(1024, b, depth, grid, resident, rounds)
        for b in bs for depth in depths if depth * (1024 + b) <= 212992
        for grid in grids for resident in (0, 1)
        for rounds in (16, 32, 64, 128, 512)}
    by_mode = {}
    for item in raw["captures"]:
        mode = item["activity_prelude"]
        if mode not in (1, 2) or mode in by_mode:
            raise ValueError("activity modes must be unique and exhaustive")
        if item["build_exit_code"] != 0 or item["run_exit_code"] != 0:
            raise ValueError("activity capture failed")
        points = {}
        for line in item["stdout"].splitlines():
            row = json.loads(line)
            key = tuple(row[name] for name in (
                "a_bytes", "b_bytes", "depth", "grid", "resident", "rounds"))
            if (key in points or row["activity_prelude"] != mode
                    or row["shared_operand_mask"] != 0
                    or row["row_stride_bytes"] != 128
                    or row["multicast_self"] != 0
                    or row["checks_passed"] is not True
                    or row["prelude_clock_mhz_p50"] <= 0
                    or row["clock_mhz_p50"] <= 0):
                raise ValueError("invalid/duplicate activity point")
            points[key] = row
        if set(points) != expected:
            raise ValueError("activity control set incomplete")
        by_mode[mode] = points
    pairs = []
    for key in sorted(expected):
        copy = by_mode[1][key]
        alu = by_mode[2][key]
        pairs.append({**{name: value for name, value in zip((
            "a_bytes", "b_bytes", "depth", "grid", "resident", "rounds"), key, strict=True)},
            "copy_prelude_span_ns": copy["span_ns_p50"],
            "alu_prelude_span_ns": alu["span_ns_p50"],
            "alu_over_copy_span_ratio": alu["span_ns_p50"] / copy["span_ns_p50"],
            "alu_minus_copy_span_ns": alu["span_ns_p50"] - copy["span_ns_p50"],
            "copy_interval_clock_mhz": copy["clock_mhz_p50"],
            "alu_interval_clock_mhz": alu["clock_mhz_p50"],
            "copy_prelude_clock_mhz": copy["prelude_clock_mhz_p50"],
            "alu_prelude_clock_mhz": alu["prelude_clock_mhz_p50"]})
    grouped = []
    for resident in (0, 1):
        for rounds in (16, 32, 64, 128, 512):
            rows = [row for row in pairs if row["resident"] == resident and row["rounds"] == rounds]
            ratios = [row["alu_over_copy_span_ratio"] for row in rows]
            deltas = [row["alu_minus_copy_span_ns"] for row in rows]
            grouped.append({"resident": resident, "rounds": rounds,
                "pair_count": len(rows), "span_ratio_min": min(ratios),
                "span_ratio_median": statistics.median(ratios),
                "span_ratio_max": max(ratios), "signed_delta_ns_min": min(deltas),
                "signed_delta_ns_median": statistics.median(deltas),
                "signed_delta_ns_max": max(deltas), "nonpositive_deltas": sum(d <= 0 for d in deltas)})
    clocks = [{"activity_prelude": mode,
        "prelude_clock_mhz_range": [min(r["prelude_clock_mhz_p50"] for r in points.values()),
            max(r["prelude_clock_mhz_p50"] for r in points.values())],
        "copy_clock_mhz_range": [min(r["clock_mhz_p50"] for r in points.values()),
            max(r["clock_mhz_p50"] for r in points.values())]}
        for mode, points in by_mode.items()]
    return {"scope": "Independent activity-domain control, not a GEMM price or production-clock profile.",
        "valid_points": sum(map(len, by_mode.values())), "paired_comparisons": len(pairs),
        "clocks": clocks, "by_length_and_residency": grouped, "pairs": pairs,
        "limits": ["ALU prelude is not native power-GEMM equivalence.",
            "Disjoint domains and common eviction control cache preparation, without asserting observed zero hits.",
            "Common cooperative grid sync can touch cached control memory.",
            "No per-interval DRAM clock, queue credits, native producer ownership or scalar wait price is inferred.",
            "All signed differences are retained; no slower cell is selected as a Forward parameter."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--full-profile", action="store_true")
    parser.add_argument("--wide-requests", action="store_true")
    arguments = parser.parse_args()
    result = analyze(arguments.campaign, full_profile=arguments.full_profile,
        wide_requests=arguments.wide_requests)
    (arguments.campaign / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
