#!/usr/bin/env python3
"""Turn the C500 MCCL collective span sweep into ProtoBridge protocol constants.

Reads the mcTracer kernel spans captured alongside the sweep written by
``collective_protocol_latency_runner.py`` and inverts the published ProtoBridge
development model for the one free constant of each protocol band: the band's
collective startup.  Every other input of that model -- link bandwidth, lane
count, per-phase fixed latency, spray chunk, topology -- is left exactly as the
hardware record already declares it, so the inversion has one unknown per band
and no freedom to trade one term against another.

A two-rank collective's kernel span is not symmetric: whichever rank reaches the
collective last runs it without waiting, and the rank that arrived first absorbs
the difference as occupancy.  Only the late rank's span is the mechanism; the
early rank's is the mechanism plus a skew this model represents separately by
serialising ranks on the execution timeline.  The extraction therefore takes,
per message size, the **minimum across ranks** of that rank's own median span,
which selects the late rank without needing to know which one it was.

The measured side is the microbenchmark's own synthetic message-size sweep.  No
production operator, kernel or Serving duration participates, and no candidate
is scored against a Midlayer accuracy metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.kernel.hardware_spec import HWSpec  # noqa: E402
from src.kernel.kernel_spec import (  # noqa: E402
    KernelKind, KernelSpec, PhysicalTransferConfig,
)
from src.kernel.launch_config import metax_c500 as metax_inputs  # noqa: E402
from src.kernel.predictors.protobridge import adapter  # noqa: E402

BAND_ORDER = ("small", "ll128", "simple")


_PROTOCOL_BAND = {"ll": "small", "ll128": "ll128", "simple": "simple"}


def _band_of(hardware_id: str, operation: str, algorithm: str,
             participants: int, message_bytes: int) -> str:
    """Bucket a measured size exactly as the prediction path does.

    A development profile does not take its protocol from the byte thresholds
    but from the vendor selector rules, so the bucketing has to ask the same
    function the prediction asks.  The measured vendor kernel names agree with
    it: MCCL keeps the ``FC2_LL`` entry point through 65,536 B and switches to
    ``FC2`` above.
    """

    profile = adapter.resolve_registered_profile(
        adapter.development_profile_config_sha256(hardware_id, operation))
    transfer = _transfer(hardware_id, operation, algorithm, participants,
                         message_bytes)
    protocol = adapter._selected_vendor_protocol(transfer, profile)
    if protocol not in _PROTOCOL_BAND:
        raise ValueError(f"unknown ProtoBridge protocol: {protocol!r}")
    return _PROTOCOL_BAND[protocol]


def _transfer(hardware_id, operation, algorithm, participants, message_bytes):
    return PhysicalTransferConfig(
        algorithm=algorithm,
        bytes=message_bytes,
        direction="collective",
        link="metaxlink",
        topology="fully_connected",
        participants=tuple(range(participants)),
        operation=operation,
        bytes_semantics="per_rank_input",
        reduction="sum" if operation == "all_reduce" else None,
        backend="protobridge",
        resource_id="tp",
        backend_config_sha256=adapter.development_profile_config_sha256(
            hardware_id, operation),
    )


def _predict_us(hardware, hardware_id, operation, algorithm, participants,
                message_bytes) -> float:
    transfer = _transfer(hardware_id, operation, algorithm, participants,
                         message_bytes)
    kernel = KernelSpec(
        kind=KernelKind.COMMUNICATION, hardware_id=hardware_id,
        transfer=transfer)
    return adapter.predict_protobridge(kernel, hardware).t_total_s * 1e6


def _reload_with_startups(startups: dict[str, float]) -> None:
    """Rebuild the registered profiles with one band's startup replaced."""

    inputs = dict(metax_inputs.DEVELOPMENT_PROTOBRIDGE_INPUTS)
    for band, value in startups.items():
        inputs[f"{band}_startup_us"] = value
    metax_inputs.DEVELOPMENT_PROTOBRIDGE_INPUTS = inputs
    import importlib
    from src.kernel.launch_config import communication_profile
    importlib.reload(communication_profile)
    importlib.reload(adapter)


def _solve_startup(hardware_id, operation, algorithm, participants,
                   message_bytes, target_us, band, other) -> float:
    low, high = 0.0, 4096.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        _reload_with_startups({**other, band: middle})
        hardware = HWSpec(
            REPOSITORY_ROOT / "configs" / "hardware" / f"{hardware_id}.yaml")
        value = _predict_us(hardware, hardware_id, operation, algorithm,
                            participants, message_bytes)
        if value < target_us:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="append", required=True,
                        help="a rank-N.json written by the runner; repeat it")
    parser.add_argument("--statistic", default="kernel_span_us_median",
                        choices=("kernel_span_us_min", "kernel_span_us_median"))
    parser.add_argument("--hardware-id", default="metax_c500")
    parser.add_argument("--algorithm", default="fc2")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    per_rank: dict[tuple[str, int, int], dict[int, float]] = {}
    provenance = []
    for path in args.sweep:
        document = json.loads(Path(path).read_text())
        if document["kind"] != "metax_c500_collective_kernel_span_sweep":
            raise ValueError(f"not a collective span sweep: {path}")
        provenance.append({
            "path": str(path),
            "run_id": document["run_id"],
            "source_commit": document["source_commit"],
            "rank": document["rank"],
            "world_size": document["world_size"],
            "operation": document["operation"],
            "timer_domain": document["timer_domain"],
        })
        for cell in document["cells"]:
            key = (document["operation"], document["world_size"],
                   cell["message_bytes"])
            per_rank.setdefault(key, {})[document["rank"]] = cell[
                args.statistic]
    # Late rank per size: the smaller median is the span without skew.
    cells = {key: [min(values.values())] for key, values in per_rank.items()}

    bands: dict[str, dict] = {}
    for band in BAND_ORDER:
        members = {
            key: statistics.median(values)
            for key, values in cells.items()
            if _band_of(args.hardware_id, key[0], args.algorithm, key[1],
                        key[2]) == band
        }
        if not members:
            continue
        bands[band] = {
            "measured_cells": [
                {"operation": key[0], "world_size": key[1],
                 "message_bytes": key[2], "per_collective_us": round(value, 4)}
                for key, value in sorted(members.items())
            ],
        }

    startups: dict[str, float] = {}
    for band, payload in bands.items():
        anchor = min(payload["measured_cells"],
                     key=lambda cell: cell["message_bytes"])
        startups[band] = _solve_startup(
            args.hardware_id, anchor["operation"], args.algorithm,
            anchor["world_size"], anchor["message_bytes"],
            anchor["per_collective_us"], band, dict(startups))
        payload["anchor"] = anchor
        payload["startup_us"] = round(startups[band], 4)

    _reload_with_startups(startups)
    hardware = HWSpec(
        REPOSITORY_ROOT / "configs" / "hardware" / f"{args.hardware_id}.yaml")
    for band, payload in bands.items():
        residuals = []
        for cell in payload["measured_cells"]:
            predicted = _predict_us(
                hardware, args.hardware_id, cell["operation"], args.algorithm,
                cell["world_size"], cell["message_bytes"])
            residuals.append({
                **cell,
                "predicted_us": round(predicted, 4),
                "signed_pct": round(
                    (predicted - cell["per_collective_us"])
                    / cell["per_collective_us"] * 100.0, 3),
            })
        payload["residuals"] = residuals
        payload["worst_abs_signed_pct"] = round(
            max(abs(row["signed_pct"]) for row in residuals), 3)

    document = {
        "schema_version": 1,
        "kind": "metax_c500_collective_protocol_policy_extraction",
        "hardware_id": args.hardware_id,
        "algorithm": args.algorithm,
        "statistic": args.statistic,
        "rank_selection": "per size, the minimum across ranks of that rank's "
                          "median kernel span, which is the rank that reached "
                          "the collective last and therefore did not wait",
        "unit": "microseconds",
        "derivation": "one unknown per protocol band, the band's collective "
                      "startup, solved by bisection so the published "
                      "ProtoBridge development model reproduces the measured "
                      "per-collective duration at that band's smallest swept "
                      "message; every other model input is the hardware "
                      "record's own declared value",
        "production_or_evaluation_duration_consumed": False,
        "sweeps": provenance,
        "bands": bands,
    }
    Path(args.output).write_text(json.dumps(document, indent=2) + "\n")
    for band, payload in bands.items():
        print("%-7s startup %9.3f us   worst cell %6.2f%%  (%d cells)" % (
            band, payload["startup_us"], payload["worst_abs_signed_pct"],
            len(payload["residuals"])))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
