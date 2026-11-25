#!/usr/bin/env python3
"""Build a provenance-bearing, uniformly slowed production-trace window."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evaluation.json_shape import file_sha256 as sha256


FIELDS = (
    "Query Time Stamp [ms]",
    "In Token [tokens]",
    "Out Token [tokens]",
    "Cached hit [tokens]",
)
TOKEN_FIELDS = FIELDS[1:]




def _token_totals(rows: list[dict[str, str]]) -> dict[str, int]:
    totals = {}
    for field in TOKEN_FIELDS:
        values = []
        for index, row in enumerate(rows):
            raw = row[field]
            try:
                value = int(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{field} row {index} must be an integer") from error
            if value < 0:
                raise ValueError(f"{field} row {index} cannot be negative")
            values.append(value)
        totals[field] = sum(values)
    return totals


def build(source: Path, output: Path, provenance: Path, window_s: float, anchor_qps: float) -> None:
    source_sha = sha256(source)
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("trace is empty")
    if tuple(rows[0]) != FIELDS:
        raise ValueError(f"trace fields must be exactly {FIELDS}")
    source_token_totals = _token_totals(rows)
    timestamps_ms = [float(row[FIELDS[0]]) for row in rows]
    anchor_ms = timestamps_ms[0]
    duration_s = (timestamps_ms[-1] - anchor_ms) / 1000.0
    if duration_s <= 0:
        raise ValueError("trace duration must be positive")
    mean_qps = len(rows) / duration_s
    if mean_qps <= 0:
        raise ValueError("trace mean QPS must be positive")
    scale = anchor_qps / mean_qps
    source_window_s = window_s * scale
    selected = [
        row for timestamp, row in zip(timestamps_ms, rows, strict=True)
        if timestamp - anchor_ms <= source_window_s * 1000.0
    ]
    if not selected:
        raise ValueError("slow-window selection produced no rows")
    selected_token_totals = _token_totals(selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    output_sha = sha256(output)
    payload = {
        "schema_version": 2,
        "source_csv": str(source.resolve()),
        "source_sha256": source_sha,
        "candidate_csv": str(output.resolve()),
        "candidate_sha256": output_sha,
        "mean_qps_from_full_trace": mean_qps,
        "anchor_qps": anchor_qps,
        "arrival_rate_scale": scale,
        "window_seconds": window_s,
        "source_window_seconds": source_window_s,
        "anchor_timestamp_ms": anchor_ms,
        "selected_request_count": len(selected),
        "source_token_totals": source_token_totals,
        "selected_token_totals": selected_token_totals,
        "token_conservation": {
            "fields": list(TOKEN_FIELDS),
            "source_row_count": len(rows),
            "selected_row_count": len(selected),
            "selection_preserves_row_values": True,
        },
        "selection_rule": "timestamp-anchor <= window_seconds * arrival_rate_scale",
        "arrival_transform": "t_anchor + (t - t_anchor) / arrival_rate_scale",
        "generator": "scripts/evaluation/serving/build_slow_trace_candidate.py",
    }
    provenance.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--window-seconds", type=float, default=300.0)
    parser.add_argument("--anchor-qps", type=float, default=4.0)
    args = parser.parse_args()
    if args.window_seconds <= 0 or args.anchor_qps <= 0:
        raise ValueError("window-seconds and anchor-qps must be positive")
    build(args.source, args.output, args.provenance, args.window_seconds, args.anchor_qps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
