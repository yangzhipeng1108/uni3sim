"""Summarize the independent producer-copy capacity probe, not operator prices."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text())
    if raw["status"] != "complete" or len(raw["runs"]) != 2:
        raise ValueError("complete compile and probe execution required")
    records = [json.loads(line) for line in raw["runs"][1]["stdout"].splitlines() if line.startswith('{"kind":')]
    identity = [row for row in records if row["kind"] == "identity"]
    samples = [row for row in records if row["kind"] == "sample"]
    expected = {(modifier, hot, depth, panel) for modifier in ("cg", "ca") for hot in (True, False)
                for panel in (16384, 32768, 65536) for depth in (1, 2, 4)
                if depth * panel <= 200 * 1024}
    if len(identity) != 1 or identity[0]["sms"] != 132 or len(samples) != 7 * len(expected):
        raise ValueError("probe must preserve its H200 identity and all 32x7 cells")
    groups = {}
    for row in samples:
        key = (row["cache_modifier"], row["hot"], row["depth"], row["panel_bytes"])
        if key not in expected or type(row["hot"]) is not bool:
            raise ValueError("unexpected probe geometry")
        if (row["blocks"], row["threads"], row["producer_threads"], row["shared_bytes"],
                row["resident_blocks_per_sm"], row["iterations"]) != (132, 384, 128, 204800, 1, 256):
            raise ValueError("producer geometry and single-CTA residency must be preserved")
        if row["logical_read_bytes"] != row["panel_bytes"] * 132 * 256:
            raise ValueError("copy payload count is inconsistent")
        if any(not math.isfinite(row[name]) or row[name] <= 0
               for name in ("event_ms", "maximum_cta_ns", "mean_sm_clock_ghz")):
            raise ValueError("copy measurements must be finite and positive")
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    rows = []
    if set(groups) != expected:
        raise ValueError("missing independent capacity cells")
    for key, values in groups.items():
        if {row["trial"] for row in values} != set(range(7)) or len(values) != 7:
            raise ValueError("every cell requires seven distinct trials")
        modifier, hot, depth, panel = key
        span_ms = statistics.median(row["event_ms"] for row in values)
        rows.append({"cache_modifier": modifier, "hot": hot, "depth": depth, "panel_bytes": panel,
                     "working_set": "cache_fit" if hot else "512MiB_cyclic_stream",
                     "median_event_ms": span_ms,
                     "median_logical_GB_per_s": panel * 132 * 256 / span_ms / 1e6,
                     "median_slowest_cta_steady_ns_per_panel": statistics.median(row["maximum_cta_ns"] for row in values) / 256,
                     "median_sm_clock_ghz": statistics.median(row["mean_sm_clock_ghz"] for row in values)})
    result = {"schema_version": 1, "identity": identity[0],
              "raw_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
              "sample_count": len(samples), "rows": rows,
              "production_or_evaluation_duration_consumed": False, "forward_parameter_changes": False,
              "limitations": ["copy-only resource-capacity screen, not a complete FA3 CTA service",
                              "steady streaming includes producer address/commit/wait instructions",
                              "no first-panel latency is extracted from the pipelined steady average",
                              "cache-fit design does not assert measured L1/L2 hit counters",
                              "ca may use L1; cg bypasses L1; cold stream may retain a finite wrap tail",
                              "no Q-TMA, consumer Tensor work or named producer-consumer barrier overlap"]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
