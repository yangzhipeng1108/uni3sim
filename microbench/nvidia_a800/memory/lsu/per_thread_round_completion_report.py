"""Publish the per-thread round-completion reference, fail-closed.

The probe runs one CTA on one SM, so its L1-resident cells are private to
that SM and unaffected by other tenants, while every L2-resident cell shares
the device L2 with whatever else is running.  The primary guard against that
is not here: it is the idle-device precondition the runner enforces before a
capture starts.  A busy device does not make the L2 cells noisy, it makes
them steadily wrong, and no statistic computed after the fact reliably
separates the two.  What this report adds is a withholding gate for genuine
noise and a recorded invariant check a reader can audit:

``spread``
    The ratio of the slowest to the fastest trial of a cell.  It catches
    bursty interference.

``width monotonicity`` (recorded, not withheld)
    At a fixed residency, stream count, store count and thread count, a round
    that moves more bytes through the same number of load instructions cannot
    complete sooner.  Two adjacent widths can still be physically
    indistinguishable -- at 1024 threads a 2-byte and a 4-byte round each
    issue one load per thread and move far less than the SM can absorb -- so
    an inversion is only reported when it exceeds the run's own
    repeatability, measured as the widest trial spread in that same row.

    This is a coarse screen, so it flags rather than withholds.  A verified
    clean capture still shows a few 1--2% inversions between the 2-byte and
    4-byte columns, and a capture taken under sustained interference can pass
    it while being uniformly wrong.  Read the flags; trust the runner's idle
    precondition.

Neither gate is derived from a production duration or a model error.
"""

import argparse
import collections
import csv
import hashlib
import json
import statistics
from pathlib import Path

WIDTHS = (2, 4, 8, 16)
DEFAULT_SPREAD_GATE = 1.10


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(raw_csv):
    cells = collections.defaultdict(list)
    with open(raw_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["residency"],
                int(row["streams"]),
                int(row["stores"]),
                int(row["width_bytes"]),
                int(row["threads"]),
            )
            cells[key].append(float(row["round_ns"]))
    if not cells:
        raise ValueError(f"no rows in {raw_csv}")
    return cells


def _evaluate(cells, spread_gate):
    medians = {key: statistics.median(v) for key, v in cells.items()}
    spreads = {key: max(v) / min(v) for key, v in cells.items()}
    verdict = {}
    for key, spread in spreads.items():
        verdict[key] = [] if spread <= spread_gate else ["spread"]
    # Width monotonicity over each (residency, streams, stores, threads) row.
    rows = collections.defaultdict(dict)
    for (res, streams, stores, width, threads) in medians:
        rows[(res, streams, stores, threads)][width] = (
            medians[(res, streams, stores, width, threads)])
    for row_key, by_width in rows.items():
        res, streams, stores, threads = row_key
        present = [w for w in WIDTHS if w in by_width]
        row_spread = max(
            spreads[(res, streams, stores, w, threads)] for w in present)
        for lower, upper in zip(present, present[1:]):
            inversion = by_width[lower] - by_width[upper]
            if inversion <= 0:
                continue
            if inversion / by_width[lower] <= row_spread - 1.0:
                # Inside the run's own repeatability: the two widths are not
                # resolved from each other, which is not evidence of
                # interference.
                continue
            for width in present:
                verdict[(res, streams, stores, width, threads)].append(
                    "width_monotonicity")
            break
    return medians, spreads, verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--probe-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--spread-gate", type=float, default=DEFAULT_SPREAD_GATE)
    args = parser.parse_args()

    cells = _load(args.raw_csv)
    medians, spreads, verdict = _evaluate(cells, args.spread_gate)

    published, withheld, flagged = [], [], 0
    for key in sorted(medians):
        residency, streams, stores, width, threads = key
        record = {
            "residency": residency,
            "loads_per_thread_per_round": streams,
            "stores_per_thread_per_round": stores,
            "access_width_bytes": width,
            "threads_per_cta": threads,
            "round_completion_ns": round(medians[key], 1),
            "trial_spread_ratio": round(spreads[key], 3),
        }
        failures = sorted(set(verdict[key]))
        if "width_monotonicity" in failures:
            record["width_monotonicity_flag"] = True
            flagged += 1
        if "spread" in failures:
            withheld.append({**record, "withheld_by": ["spread"]})
        else:
            published.append(record)

    document = {
        "reference": "per_thread_round_completion_sm80",
        "hardware_id": "nvidia_a800",
        "architecture": "sm_80",
        "measures": (
            "Completion time of one CTA-wide dependent source round in which "
            "every thread issues the declared number of global loads of the "
            "declared width and, when a store is declared, writes the other "
            "stream's word back through its own. Rounds are strictly "
            "serialized; requests inside a round are independent and "
            "coalesced. grid 1."),
        "timer_domain": "globaltimer_same_cta",
        "primary_guard": (
            "idle-device precondition enforced by "
            "per_thread_round_completion_runner.py before capture"),
        "gates": {
            "spread_ratio_max": args.spread_gate,
            "width_monotonicity": (
                "recorded as width_monotonicity_flag, not withheld: round "
                "time should be non-decreasing in access width at fixed "
                "residency, stream count, store count and thread count, "
                "beyond the row's own trial spread"),
        },
        "consumes_production_duration": False,
        "probe_source_path": args.probe_source,
        "probe_source_sha256": _sha256(args.probe_source),
        "raw_csv_sha256": _sha256(args.raw_csv),
        "published_cell_count": len(published),
        "withheld_cell_count": len(withheld),
        "width_monotonicity_flag_count": flagged,
        "cells": published,
        "withheld_cells": withheld,
    }
    Path(args.output).write_text(json.dumps(document, indent=2) + "\n")
    print(f"published {len(published)} cells, withheld {len(withheld)}, "
          f"width-monotonicity flags {flagged}")
    reasons = collections.Counter(
        reason for cell in withheld for reason in cell["withheld_by"])
    for reason, count in sorted(reasons.items()):
        print(f"  withheld by {reason}: {count}")


if __name__ == "__main__":
    main()
