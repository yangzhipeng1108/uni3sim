"""Analyze only the independent queued CUDA launch-gap probe's NSYS trace."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics


def samples(text):
    records = [json.loads(line) for line in text.splitlines() if line.startswith('{"kind":')]
    identity = [row for row in records if row["kind"] == "identity"]
    values = [row for row in records if row["kind"] == "sample"]
    if len(identity) != 1 or len(values) != 24:
        raise ValueError("probe requires one identity and 24 synthetic cells")
    if any(row["host_ahead"] is not True or row["graph_nodes"] != 64
           or row["eager_nodes"] != 4 or row["block"] != 128 for row in values):
        raise ValueError("every probe sample must prove the declared queued launch domain")
    if {(row["grid"], row["trial"]) for row in values} != {
            (grid, trial) for grid in (1, 32, 132) for trial in range(8)}:
        raise ValueError("missing or repeated synthetic probe cell")
    return identity[0], values


def summary(values):
    ordered = sorted(values)
    return {"count": len(values), "minimum_ns": ordered[0],
            "median_ns": statistics.median(values), "mean_ns": statistics.mean(values),
            "p90_ns": ordered[(9 * (len(ordered) - 1)) // 10], "maximum_ns": ordered[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text())
    if raw["status"] != "complete" or len(raw["runs"]) != 3:
        raise ValueError("complete compiler, standalone and traced runs required")
    standalone_identity, standalone = samples(raw["runs"][1]["stdout"])
    traced_identity, traced = samples(raw["runs"][2]["stdout"])
    if standalone_identity != traced_identity:
        raise ValueError("standalone/traced GPU identity changed")
    connection = sqlite3.connect(f"file:{args.sqlite.resolve()}?mode=ro", uri=True)
    kernels = connection.execute("""SELECT k.start,k.end,s.value,k.gridX,k.blockX,k.streamId
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
        ORDER BY k.start""").fetchall()
    gates = [row for row in kernels if "hold_gpu(" in row[2]]
    if len(gates) != 24:
        raise ValueError("trace must preserve every independent gate")
    grouped = {grid: {name: [] for name in ("graph_node", "graph_entry", "graph_exit", "eager")}
               for grid in (1, 32, 132)}
    windows = []
    for index, (gate, sample) in enumerate(zip(gates, traced, strict=True)):
        end = gates[index + 1][0] if index + 1 < len(gates) else float("inf")
        launches = [row for row in kernels if row[0] >= gate[1] and row[0] < end
                    and "touch(" in row[2]][:68]
        if len(launches) != 68 or any(row[3:6] != (sample["grid"], 128, gate[5]) for row in launches):
            raise ValueError("queued chain requires exactly the declared first 68 touch launches")
        gaps = [current[0] - previous[1] for previous, current in zip(launches[:-1], launches[1:], strict=True)]
        if min(gaps) < 0 or min(row[1] - row[0] for row in launches) <= 0:
            raise ValueError("probe stream has overlapping or invalid task intervals")
        target = grouped[sample["grid"]]
        target["eager"].extend((gaps[0], gaps[66]))
        target["graph_entry"].append(gaps[1])
        target["graph_node"].extend(gaps[2:65])
        target["graph_exit"].append(gaps[65])
        span = launches[-1][1] - launches[0][0]
        union = sum(row[1] - row[0] for row in launches)
        if span != union + sum(gaps):
            raise ValueError("kernel spans and launch gaps must conserve the chain interval")
        windows.append({"grid": sample["grid"], "trial": sample["trial"],
                        "kernel_union_ns": union, "gap_sum_ns": sum(gaps), "chain_span_ns": span})
    rows = []
    for grid in grouped:
        off = statistics.median(row["device_chain_ms"] for row in standalone if row["grid"] == grid)
        on = statistics.median(row["device_chain_ms"] for row in traced if row["grid"] == grid)
        rows.append({"grid": grid, "block": 128, "gaps": {
            name: summary(values) for name, values in grouped[grid].items()},
            "standalone_chain_median_ms": off, "traced_chain_median_ms": on,
            "trace_chain_difference_percent": 100 * (on - off) / off})
    result = {"schema_version": 1, "purpose": __doc__, "identity": standalone_identity,
              "raw_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
              "report_sha256": raw["report_sha256"], "measurement_boundary": "device_side_host_ahead",
              "production_or_evaluation_duration_consumed": False,
              "rows": rows, "windows": windows,
              "limitations": ["independent CUDA C++ scalar-touch tasks, not a production operator",
                              "Nsight graph-node tracing perturbation retained per grid",
                              "gaps exclude task spans and Host enqueue; no timing parameter adopted"]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"identity": standalone_identity, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
