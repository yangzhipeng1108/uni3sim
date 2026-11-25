"""Validate independent sharing controls and report traffic/service evidence."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics


def analyze(directory):
    raw = json.loads((directory / "raw.json").read_text())
    if raw["production_duration_consumed"]:
        raise ValueError("independent mechanism required")
    source = directory / "source" / "tma_queue_service.cu"
    if hashlib.sha256(source.read_bytes()).hexdigest() != raw["source_sha256"]:
        raise ValueError("source snapshot identity mismatch")
    if len(raw["captures"]) != 4 or len(raw["counter_captures"]) != 4:
        raise ValueError("all address masks and counter controls required")
    indexed = {}
    rates = []
    geometries = tuple((a, b, depth)
        for a, b in ((128, 8192), (1024, 8192), (4096, 4096),
                     (1024, 16384), (1024, 32768))
        for depth in (4, 6, 8, 12, 16)
        if depth * (((a + 1023) // 1024) * 1024 + b) <= 212992)
    expected = {(a, b, depth, grid, resident, rounds)
        for a, b, depth in geometries for grid in (32, 132)
        for resident in (0, 1) for rounds in (128, 512)}
    for item in raw["captures"]:
        mask = item["shared_operand_mask"]
        if mask not in (0, 1, 2, 3) or mask in indexed:
            raise ValueError("address masks must be unique and exhaustive")
        if item["build_exit_code"] != 0 or item["run_exit_code"] != 0:
            raise ValueError("mechanism capture failed")
        points = {}
        for line in item["stdout"].splitlines():
            record = json.loads(line)
            key = tuple(record[name] for name in (
                "a_bytes", "b_bytes", "depth", "grid", "resident", "rounds"))
            if (key in points or not record["checks_passed"]
                    or record["shared_operand_mask"] != mask
                    or record["row_stride_bytes"] != 128
                    or record["multicast_self"] != 0):
                raise ValueError("unexpected or invalid mechanism point")
            points[key] = record
        if set(points) != expected:
            raise ValueError("missing/duplicate controls")
        indexed[mask] = points
        for a, b, depth in geometries:
            for grid in (32, 132):
                for resident in (0, 1):
                    short = points[a, b, depth, grid, resident, 128]
                    long = points[a, b, depth, grid, resident, 512]
                    difference = long["span_ns_p50"] - short["span_ns_p50"]
                    if difference <= 0:
                        raise ValueError("all secants must be positive")
                    rates.append({"mask": mask, "a_bytes": a, "b_bytes": b,
                        "depth": depth, "grid": grid, "resident": resident,
                        "ns_per_round": difference / 384,
                        "logical_bytes_per_s": grid * (a + b) * 384e9 / difference})
    envelopes = [{"mask": mask, "resident": resident,
        "capacity_upper_envelope_logical_bytes_per_s": max(
            row["logical_bytes_per_s"] for row in rates
            if row["mask"] == mask and row["resident"] == resident)}
        for mask in (0, 1, 2, 3) for resident in (0, 1)]
    paired = []
    for mask in (1, 2, 3):
        for resident in (0, 1):
            ratios = []
            for a, b, depth in geometries:
                for grid in (32, 132):
                    cell = lambda row: (row["a_bytes"], row["b_bytes"], row["depth"], row["grid"], row["resident"])
                    target = (a, b, depth, grid, resident)
                    baseline, = (row for row in rates if row["mask"] == 0 and cell(row) == target)
                    sharing, = (row for row in rates if row["mask"] == mask and cell(row) == target)
                    ratios.append(sharing["ns_per_round"] / baseline["ns_per_round"])
            paired.append({"mask": mask, "resident": resident,
                "paired_count": len(ratios), "shared_over_private_secant_span_min": min(ratios),
                "shared_over_private_secant_span_median": statistics.median(ratios),
                "shared_over_private_secant_span_max": max(ratios)})
    counters = []
    for item in raw["counter_captures"]:
        if item["build_exit_code"] != 0 or item["run_exit_code"] != 0:
            raise ValueError("counter capture failed")
        rows = list(csv.reader(io.StringIO(item["stdout"])))
        header_index, = (i for i, row in enumerate(rows) if row and row[0] == "ID")
        header = rows[header_index]
        data, = (row for row in rows[header_index + 1:]
            if len(row) == len(header) and row[0].isdigit())
        values = dict(zip(header, data, strict=True))
        payload = 132 * 512 * (4096 + 4096)
        l2 = float(values["lts__t_sectors_op_read.sum"].replace(",", "")) * 32
        dram = float(values["dram__bytes_read.sum"].replace(",", ""))
        counters.append({"mask": item["shared_operand_mask"], "payload_bytes": payload,
            "l2_read_bytes": l2, "l2_read_over_logical_payload": l2 / payload,
            "dram_read_bytes": dram, "dram_read_over_logical_payload": dram / payload})
    if {row["mask"] for row in counters} != {0, 1, 2, 3}:
        raise ValueError("counter masks incomplete")
    fixed_cell_timing = [row for row in rates if row["a_bytes"] == 4096
        and row["b_bytes"] == 4096 and row["depth"] == 4
        and row["grid"] == 132 and row["resident"] == 1]
    clocks = [row["clock_mhz_p50"] for points in indexed.values() for row in points.values()]
    return {"scope": "Independent address-sharing TMA service/traffic evidence only. No production durations or Forward parameter selection.",
        "valid_timing_points": sum(map(len, indexed.values())),
        "clock_mhz_range": [min(clocks), max(clocks)],
        "counter_cell_matched_timing": fixed_cell_timing,
        "counter_controls": counters, "capacity_upper_envelopes": envelopes,
        "paired_span_ratios": paired, "rates": rates,
        "limits": ["Advancing stream is not observed zero-hit DRAM traffic; shared inputs have smaller working sets.",
            "Counter binaries omit the device prelude, so exactly512 rounds define the payload. Timing binaries retain the device prelude.",
            "Synchronized fully shared CTA sources do not identify native partial-sharing, descriptor promotion or scheduler phase.",
            "Neither an address-sharing rate nor reduced sector count is a universal TMA bandwidth or multiplier."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    arguments = parser.parse_args()
    result = analyze(arguments.campaign)
    (arguments.campaign / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rates"}, indent=2))


if __name__ == "__main__":
    main()
