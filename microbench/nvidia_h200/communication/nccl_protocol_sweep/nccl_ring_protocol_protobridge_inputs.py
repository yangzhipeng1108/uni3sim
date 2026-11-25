#!/usr/bin/env python3
"""Derive the H200 scale-up ProtoBridge NCCL inputs from the two-rank scan.

Input is the ``report.json`` of ``nccl_ring_protocol_runner.py`` (two H200s,
NCCL Ring forced to LL / LL128 / Simple, plus NCCL's own ``auto`` pass, every
collective a CUDA-graph node).  No Serving workload or production duration is
read.  The vendored surrogate prices a ring all-reduce at N ranks as
``startup + 2(N-1) * phase_fixed + 2(N-1) * serial(D, ring_bw)``; the serial
term is linear in ``1/ring_bw`` and the fixed part is linear in ``startup``.

* ``num_lanes`` / ``per_link_bytes_per_us`` come from the H200 NVLink4 spec
  (``configs/hardware/nvidia_h200.yaml``: 450 GB/s one-way = 18 x 25 GB/s).
* Simple two-rank ring bandwidth: chosen so the surrogate's Simple all-reduce
  slope between the two largest selector-domain sizes equals the measured one.
* A two-rank scan identifies only ``startup + 2 * phase_fixed``.  Each
  protocol keeps its current per-phase value (A800 NV8, unmeasured on H200;
  it only matters for N > 2) and its startup is solved so the surrogate
  reproduces the H200 measurement at the smallest measured size at which the
  registered NCCL selection picks the protocol at N=2 (LL: 256 B, Simple:
  2 MiB).  The registered selection never picks LL128 at N=2, so LL128 is
  solved at the smallest forced-LL128 measurement (256 B).

Usage:
    python nccl_ring_protocol_protobridge_inputs.py <report.json> \
        [--output <reference.json>]
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.kernel.hardware_spec import HWSpec  # noqa: E402
from src.kernel.kernel_base import predict  # noqa: E402
from src.kernel.kernel_spec import (  # noqa: E402
    KernelKind,
    KernelSpec,
    PhysicalTransferConfig,
)
from src.kernel.launch_config.communication_profile import (  # noqa: E402
    development_protobridge_inputs_for,
)
from src.kernel.predictors.protobridge import adapter  # noqa: E402
from src.kernel.predictors.protobridge.upstream import (  # noqa: E402
    load_upstream_module,
)

TheoryDerivedSurrogate = load_upstream_module().TheoryDerivedSurrogate

HARDWARE_ID = "nvidia_h200"
HARDWARE_CONFIG = "configs/hardware/nvidia_h200.yaml"
PARTICIPANTS = 2
PROTOCOLS = {"LL": "ll", "LL128": "ll128", "Simple": "simple"}
STARTUP_FIELD = {"ll": "small_startup_us", "ll128": "ll128_startup_us",
                 "simple": "simple_startup_us"}
PHASE_FIELD = {"ll": "small_phase_fixed_us", "ll128": "ll128_phase_fixed_us",
               "simple": "simple_phase_fixed_us"}
SPEC_PER_LINK_BYTES_PER_US = 25_000.0
SPEC_NUM_LANES = 18
SELECTOR_MAX_BYTES = 128 * 1024 * 1024


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measured(report, protocol_key):
    return {
        int(row["message_bytes"]): float(row["ns_per_call"]) / 1000.0
        for row in report["protocols"][protocol_key]["summary"]
        if row["operation"] == "all_reduce"
    }


def _forced_all_reduce_us(inputs, protocol, message_bytes, participants):
    """The adapter's ring all-reduce pricing with the protocol fixed."""

    profile = adapter._profile_from_hardware_inputs(inputs, "all_reduce", "ring")
    parameters = dict(profile.model_parameters)
    parameters["link_latency_us"] = (
        profile.protocol_policy.phase_fixed_for_protocol(protocol))
    parameters["ring_bw_bytes_per_us"] = profile.protocol_policy.ring_bandwidth_for(
        protocol, "all_reduce", participants,
        float(parameters["ring_bw_bytes_per_us"]))
    return TheoryDerivedSurrogate(**parameters).predict(
        data_bytes=message_bytes, num_gpus=participants, algo="ring",
        credits=profile.credits, ber=0.0, mem_bw=0.0,
        startup_us=profile.protocol_policy.startup_for_protocol(protocol),
        fec_enabled=False, llr_enabled=False, competing_bw=0.0,
        fec_n=None, fec_k=None, fec_t=None, retry_mode="gobackn",
        retry_buffer_entries=None, retry_buffer_bytes=None,
        optical_fraction=1.0, optical_hops=1.0, strict_reliability=False,
        retry_limit=None, include_credit=True, include_fec=True,
        include_retransmission=True, include_contention=True,
        topology=None, collective="allreduce", phase="full")


def _selected_protocol(message_bytes):
    selection = adapter.select_collective_launch(
        HARDWARE_ID, communication_library="NCCL", library_version="2.27.3",
        operation="all_reduce", message_bytes=message_bytes,
        participant_count=PARTICIPANTS, dtype="bfloat16", reduction="sum",
        in_place=True, topology="single_node_nvlink")
    if selection.protocol_status != "selected":
        raise ValueError(f"no selected protocol at {message_bytes} bytes")
    return selection.vendor_protocol


def derive(report_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    if report["world_size"] != PARTICIPANTS or report["dtype"] != "bfloat16":
        raise ValueError("report is not the two-rank BF16 scan")
    if report["environment"]["hardware_id"] != HARDWARE_ID:
        raise ValueError("report hardware is not nvidia_h200")
    measured = {protocol: _measured(report, key)
                for key, protocol in PROTOCOLS.items()}
    measured_auto = _measured(report, "auto")
    sizes = sorted(size for size in measured["ll"] if size <= SELECTOR_MAX_BYTES)
    selected = {size: _selected_protocol(size) for size in sizes}

    inputs = dict(development_protobridge_inputs_for(HARDWARE_ID, link="nvlink"))
    inputs["per_link_bytes_per_us"] = SPEC_PER_LINK_BYTES_PER_US
    inputs["num_lanes"] = SPEC_NUM_LANES

    # Simple two-rank ring bandwidth from the slope of the two largest sizes.
    simple_sizes = [size for size in sizes if selected[size] == "simple"]
    low, high = simple_sizes[-2], simple_sizes[-1]
    measured_slope = (measured["simple"][high] - measured["simple"][low]) / (high - low)
    reference_bw = 100_000.0
    inputs["simple_allreduce_ring_bw_by_participant"] = ((PARTICIPANTS, reference_bw),)
    reference_slope = (
        _forced_all_reduce_us(inputs, "simple", high, PARTICIPANTS)
        - _forced_all_reduce_us(inputs, "simple", low, PARTICIPANTS)) / (high - low)
    simple_bw = reference_bw * reference_slope / measured_slope
    inputs["simple_allreduce_ring_bw_by_participant"] = ((PARTICIPANTS, simple_bw),)

    anchors = {}
    for protocol in ("ll", "ll128", "simple"):
        chosen = [size for size in sizes if selected[size] == protocol]
        anchor = chosen[0] if chosen else min(measured[protocol])
        target = measured[protocol][anchor]
        predicted = _forced_all_reduce_us(inputs, protocol, anchor, PARTICIPANTS)
        startup = float(inputs[STARTUP_FIELD[protocol]]) + target - predicted
        if startup <= 0.0:
            raise ValueError(f"{protocol} solved startup is not positive: {startup}")
        inputs[STARTUP_FIELD[protocol]] = startup
        anchors[protocol] = {
            "message_bytes": anchor,
            "anchor_rule": ("smallest_selected_size_at_n2" if chosen
                            else "smallest_forced_size_protocol_not_selected_at_n2"),
            "measured_us": target,
            "startup_us": startup,
            "phase_fixed_us_kept": float(inputs[PHASE_FIELD[protocol]]),
        }
        check = _forced_all_reduce_us(inputs, protocol, anchor, PARTICIPANTS)
        if abs(check - target) > 1e-9 * target:
            raise AssertionError(f"{protocol} startup solve did not close")

    check_table = []
    for size in sizes:
        protocol = selected[size]
        model = _forced_all_reduce_us(inputs, protocol, size, PARTICIPANTS)
        check_table.append({
            "message_bytes": size,
            "selected_protocol": protocol,
            "predicted_us": model,
            "measured_forced_selected_us": measured[protocol][size],
            "measured_auto_us": measured_auto[size],
            "relative_error_vs_forced": model / measured[protocol][size] - 1.0,
        })
    return {
        "schema_version": 1,
        "hardware_id": HARDWARE_ID,
        "mechanism": "nccl_ring_all_reduce_two_rank_protobridge_inputs",
        "production_or_evaluation_duration_consumed": False,
        "raw_report": {
            "path": str(report_path.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(report_path),
            "environment": report["environment"],
            "runner": report["runner"],
            "statistic": report["statistic"],
            "timer_domain": report["timer_domain"],
        },
        "derivation_script": {
            "path": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "spec_inputs": {
            "source": HARDWARE_CONFIG,
            "source_sha256": _sha256(REPOSITORY_ROOT / HARDWARE_CONFIG),
            "per_link_bytes_per_us": SPEC_PER_LINK_BYTES_PER_US,
            "num_lanes": SPEC_NUM_LANES,
            "note": "NVLink4 450 GB/s one-way = 18 links x 25 GB/s",
        },
        "simple_ring_bandwidth": {
            "participants": PARTICIPANTS,
            "slope_sizes_bytes": [low, high],
            "measured_slope_us_per_byte": measured_slope,
            "surrogate_ring_bw_bytes_per_us": simple_bw,
        },
        "startup_anchors": anchors,
        "unmeasured_on_h200": [
            "small/ll128/simple phase_fixed_us (A800 NV8 values kept; a two-rank "
            "scan identifies only startup + 2 * phase_fixed)",
            "simple_allreduce_ring_bw_by_participant for 4 and 8 ranks (dropped; "
            "spec-derived default ring bandwidth applies)",
            "t_min_switch_us, spray_chunk_bytes, wire_efficiency, link_latency_us "
            "(A800 values; link_latency_us is replaced by phase_fixed per protocol)",
            "NCCL protocol selection table (A800 NCCL 2.27.3 tuning projection)",
        ],
        "derived_inputs": {
            "per_link_bytes_per_us": SPEC_PER_LINK_BYTES_PER_US,
            "num_lanes": SPEC_NUM_LANES,
            "small_startup_us": inputs["small_startup_us"],
            "ll128_startup_us": inputs["ll128_startup_us"],
            "simple_startup_us": inputs["simple_startup_us"],
            "small_phase_fixed_us": inputs["small_phase_fixed_us"],
            "ll128_phase_fixed_us": inputs["ll128_phase_fixed_us"],
            "simple_phase_fixed_us": inputs["simple_phase_fixed_us"],
            "simple_allreduce_ring_bw_by_participant": [[PARTICIPANTS, simple_bw]],
        },
        "check_table_n2": check_table,
    }


def registered_check(reference: dict) -> list:
    """Adapter predictions of the registered H200 record at the anchors."""

    hardware = HWSpec(str(REPOSITORY_ROOT / HARDWARE_CONFIG))
    sha = adapter.development_profile_config_sha256(HARDWARE_ID, "all_reduce")
    rows = []
    for protocol, anchor in reference["startup_anchors"].items():
        if anchor["anchor_rule"] != "smallest_selected_size_at_n2":
            continue
        transfer = PhysicalTransferConfig(
            algorithm="ring", bytes=anchor["message_bytes"],
            direction="collective", link="nvlink", topology="ring",
            participants=tuple(range(PARTICIPANTS)), operation="all_reduce",
            bytes_semantics="per_rank_input",
            ring_order=tuple(range(PARTICIPANTS)), reduction="sum",
            backend="protobridge", backend_config_sha256=sha,
            resource_id="node0-nvlink")
        prediction = predict(KernelSpec(KernelKind.COMMUNICATION, HARDWARE_ID,
                                        transfer=transfer), hardware, "midlayer")
        rows.append((protocol, anchor["measured_us"], prediction.t_total_s * 1e6))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-registered", action="store_true",
                        help="compare the registered H200 record to the anchors")
    args = parser.parse_args()
    reference = derive(args.report.resolve(strict=True))
    if args.output is not None:
        args.output.write_text(json.dumps(reference, indent=1, sort_keys=True) + "\n")
    print(json.dumps(reference["derived_inputs"], indent=1, sort_keys=True))
    worst = max(abs(row["relative_error_vs_forced"]) for row in reference["check_table_n2"])
    for row in reference["check_table_n2"]:
        print(f"{row['message_bytes']:>10} {row['selected_protocol']:>6} "
              f"pred={row['predicted_us']:9.3f} meas={row['measured_forced_selected_us']:9.3f} "
              f"auto={row['measured_auto_us']:9.3f} rel={row['relative_error_vs_forced']:+.4f}")
    print(f"max |relative error| at N=2 over selected sizes: {worst:.4f}")
    if args.check_registered:
        for protocol, measured, predicted in registered_check(reference):
            print(f"registered {protocol}: measured={measured:.6f} predicted={predicted:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
