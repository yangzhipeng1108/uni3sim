#!/usr/bin/env python3
"""Derive the H200 memory mechanism references (WP4) from retained raw captures.

Every reference is regenerated from the runner outputs under
``profiler_output/microbench/nvidia_h200/<run_id>/`` with the selection rule
of its A800 counterpart.  References that have an A800 counterpart keep the
A800 key set (checked by ``schema-check``); the others are new H200 evidence
files for parameters whose A800 value had no reference.  No production shape,
operator or duration is consumed.

Usage (from the repository root)::

    python microbench/nvidia_h200/memory/memory_reference_report.py all
"""

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "profiler_output/microbench/nvidia_h200"
REFS = ROOT / "microbench/nvidia_h200/references"
A800_REFS = ROOT / "microbench/nvidia_a800/references"
H200 = ROOT / "microbench/nvidia_h200/memory"
SM_CLOCK_HZ = 1.98e9            # measured H200 busy SM clock (clock gate)
DRAM_CLOCK_HZ = 3.125e9         # clock_domains.dram of configs/hardware/nvidia_h200.yaml
L2_BYTES = 62914560             # device-reported H200 L2
SMS = 132
DEVICE_INDEX = 1
RUN = {
    "dram_read_bw": "dram_read_bw_20260925",
    "bw_per_sm": "dram_bw_per_sm_20260925",
    "latency": "pointer_chase_latency_20260925",
    "mixed_l2": "l2_directional_3r2_20260925",
    "mixed_dram": "dram_directional_3r2_20260925",
    "ramp": ("dram_finite_stream_ramp_20260925_a", "dram_finite_stream_ramp_20260925_b"),
    "residency": ("l2_residency_knee_20260925_a", "l2_residency_knee_20260925_b"),
    "l2_bandwidth": "l2_bandwidth_20260925",
    "scalar_half": "scalar_half_dram_stream_20260925",
}
PAIRS = {
    "add_rmsnorm_mixed_l2_service_sm90.json": "add_rmsnorm_mixed_l2_service_sm80.json",
    "add_rmsnorm_mixed_dram_service_sm90.json": "add_rmsnorm_mixed_dram_service_sm80.json",
    "dram_finite_read_stream_ramp_sm90.json": "dram_finite_read_stream_ramp_sm80.json",
    "l2_operand_residency_curve_sm90.json": "l2_operand_residency_curve_sm80.json",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def load(run_id):
    return json.loads((RAW / run_id / "candidate.json").read_text())


def git_commit():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                          text=True, capture_output=True).stdout.strip()


def clock_summary(candidate):
    accepted = candidate["clock_gate"]["accepted"]
    return {
        "gate": "every busy nvidia-smi sample (utilization.gpu > 0, 50 ms period) "
                "must report at least boost_mhz; otherwise the process is rejected and re-run",
        "boost_mhz": candidate["clock_gate"]["boost_mhz"],
        "accepted_captures": len(accepted),
        "rejected_captures": len(candidate["clock_gate"]["rejected"]),
        "busy_sm_clock_mhz_min": min(a["busy_sm_clock_mhz_min"] for a in accepted),
        "busy_sm_clock_mhz_max": max(a["busy_sm_clock_mhz_max"] for a in accepted),
        "busy_samples": sum(a["busy_samples"] for a in accepted),
        "rejected": candidate["clock_gate"]["rejected"],
    }


def write(name, payload):
    path = REFS / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", rel(path))
    return payload


def spread_percent(values):
    median = statistics.median(values)
    return (max(values) - min(values)) / median * 100.0


# --------------------------------------------------------------------------- #1-3 capacity
def dram_capacity():
    read_bw = load(RUN["dram_read_bw"])
    per_sm = load(RUN["bw_per_sm"])
    medians = read_bw["medians_by_ctas_per_sm"]
    best = max(medians, key=lambda k: medians[k]["median_gb_per_s"])
    capacity_gbs = medians[best]["median_gb_per_s"]
    cells = per_sm["medians"]
    full = {f"{t}/{p}": cells[f"{t}/{p}/{SMS}"]["median_gb_per_s"]
            for t in (256, 1024) for p in ("L1-bypass", "L1-cached")}
    ratios = {str(t): full[f"{t}/L1-cached"] / full[f"{t}/L1-bypass"] for t in (256, 1024)}
    per_point = []
    for t in (256, 1024):
        for sms in sorted({int(k.split("/")[2]) for k in cells}):
            bypass = cells[f"{t}/L1-bypass/{sms}"]["median_gb_per_s"]
            cached = cells[f"{t}/L1-cached/{sms}"]["median_gb_per_s"]
            per_point.append({"threads_per_block": t, "active_sms": sms,
                              "l1_bypass_gb_per_s": bypass, "l1_cached_gb_per_s": cached,
                              "cached_over_bypass": cached / bypass})
    max_dev = max(abs(p["cached_over_bypass"] - 1.0) for p in per_point)
    return write("dram_read_capacity_protocol_sm90.json", {
        "schema": "nvidia_h200_dram_read_capacity_protocol_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "production_or_evaluation_duration_consumed": False,
        "dram_read_capacity": {
            "selection_rule": "maximum over resident-CTA counts of the per-count median "
                              "over five clock-gated processes (saturated plateau)",
            "capacity_gb_per_s": capacity_gbs,
            "capacity_bytes_per_s": capacity_gbs * 1e9,
            "selected_ctas_per_sm": int(best),
            "medians_by_ctas_per_sm": {k: v["median_gb_per_s"] for k, v in medians.items()},
            "relative_spread_by_ctas_per_sm": {k: v["relative_spread"] for k, v in medians.items()},
            "dram_clock_hz": DRAM_CLOCK_HZ,
            "dram_read_capacity_per_dram_clk": capacity_gbs * 1e9 / DRAM_CLOCK_HZ,
            "per_sm_clk_bytes": capacity_gbs * 1e9 / SM_CLOCK_HZ,
        },
        "dram_protocol_efficiency": {
            "probe": "bw_per_sm.cu (.cg vs .ca, 2 GiB incompressible, one CTA per active SM)",
            "full_chip_gb_per_s": full,
            "l1_cached_over_l1_bypass_full_chip": ratios,
            "maximum_abs_cached_over_bypass_deviation_all_points": max_dev,
            "per_point_medians": per_point,
            "dram_protocol_efficiency": 1.0,
            "dram_protocol_efficiency_l1cached": 1.0,
            "derivation": "The A800 contract keeps both efficiencies at 1.0 because the "
                          "sustained capacity is itself a measured stream rate that already "
                          "contains protocol overhead, and .cg and .ca streams achieve the "
                          "same rate.  On H200 .ca/.cg stays within the listed maximum "
                          "deviation at every active-SM point, so both remain 1.0 relative to "
                          "the measured read capacity above.  One CTA per SM (even at 1024 "
                          "threads) does not saturate HBM3e, so this probe supplies the "
                          ".cg/.ca ratio and dram_read_bw.cu supplies the capacity.",
        },
        "provenance": {
            "capacity_source": "microbench/nvidia_h200/memory/dram/dram_read_bw.cu",
            "capacity_source_sha256": read_bw["source_sha256"],
            "capacity_runner": "microbench/nvidia_h200/memory/dram/dram_read_bw_runner.py",
            "capacity_runner_sha256": read_bw["runner_sha256"],
            "capacity_raw": f"profiler_output/microbench/nvidia_h200/{RUN['dram_read_bw']}",
            "capacity_measurements_sha256": read_bw["measurements_sha256"],
            "capacity_clock": clock_summary(read_bw),
            "protocol_source": "microbench/nvidia_h200/memory/dram/bw_per_sm.cu",
            "protocol_source_sha256": per_sm["source_sha256"],
            "protocol_runner": "microbench/nvidia_h200/memory/dram/bw_per_sm_runner.py",
            "protocol_runner_sha256": per_sm["runner_sha256"],
            "protocol_raw": f"profiler_output/microbench/nvidia_h200/{RUN['bw_per_sm']}",
            "protocol_measurements_sha256": per_sm["measurements_sha256"],
            "protocol_clock": clock_summary(per_sm),
            "device_index": DEVICE_INDEX,
            "device_uuid": read_bw["device_state"]["before"]["uuid"],
            "git_commit": git_commit(),
        },
    })


# --------------------------------------------------------------------------- #2 latency
def latency():
    c = load(RUN["latency"])
    dram = c["summary"]["dram"]["median_latency_ns"]
    l2 = c["summary"]["l2"]["median_latency_ns"]
    return write("pointer_chase_latency_sm90.json", {
        "schema": "nvidia_h200_pointer_chase_latency_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "production_or_evaluation_duration_consumed": False,
        "selection_rule": "median of all %globaltimer trials pooled over two independent "
                          "clock-gated processes (DRAM 3 and L2 5 trials per process)",
        "probe_contract": c["probe_contract"],
        "l2_capacity_bytes": L2_BYTES,
        "summary": c["summary"],
        "dram_latency_ns": dram,
        "l2_latency_ns": l2,
        "derived": {
            "sm_clock_hz": SM_CLOCK_HZ,
            "l2_hit_latency_local_cycles": l2 * SM_CLOCK_HZ * 1e-9,
            "l2_hit_latency_remote_cycles": l2 * SM_CLOCK_HZ * 1e-9,
            "dram_miss_extra_cycles": (dram - l2) * SM_CLOCK_HZ * 1e-9,
            "formula": "l2_hit_latency_*_cycles = l2_latency_ns * 1.98 GHz; "
                       "dram_miss_extra_cycles = (dram_latency_ns - l2_latency_ns) * 1.98 GHz",
        },
        "provenance": {
            "sources": {"dram": "microbench/nvidia_h200/memory/dram/dram_latency.cu",
                        "l2": "microbench/nvidia_h200/memory/l2/l2_latency.cu"},
            "source_sha256": c["source_sha256"],
            "runner": "microbench/nvidia_h200/memory/dram/pointer_chase_latency_runner.py",
            "runner_sha256": c["runner_sha256"],
            "binary_sha256": c["binary_sha256"],
            "raw": f"profiler_output/microbench/nvidia_h200/{RUN['latency']}",
            "measurements_sha256": c["measurements_sha256"],
            "clock": clock_summary(c),
            "device_index": DEVICE_INDEX,
            "device_uuid": c["device_state"]["before"]["uuid"],
            "git_commit": git_commit(),
        },
    })


# --------------------------------------------------------------------------- #4 mixed 3:2
def _registers(compile_log, symbol):
    lines = compile_log.splitlines()
    for index, line in enumerate(lines):
        if line.endswith(f"Compiling entry function '{symbol}' for 'sm_90'"):
            for follow in lines[index + 1:index + 4]:
                if "Used" in follow and "registers" in follow:
                    return int(follow.split("Used ")[1].split(" registers")[0])
    raise ValueError(f"no register report for {symbol}")


def _mixed(kind):
    run_id = RUN[f"mixed_{kind}"]
    c = load(run_id)
    raw_path = RAW / run_id / "raw.csv"
    rows = list(csv.DictReader(raw_path.open()))
    grids = c["grids"]
    medians, spreads = {}, {}
    for grid in grids:
        values = [float(r["total_gbs"]) for r in rows if int(r["grid"]) == grid]
        if len(values) != c["rotations"] * c["trials_per_process"]:
            raise ValueError(f"grid {grid}: {len(values)} trials")
        medians[str(grid)] = statistics.median(values)
        spreads[str(grid)] = spread_percent(values)
    selected_grid = min(grids, key=lambda g: medians[str(g)])
    coupled = medians[str(selected_grid)] * 1e9
    buffer_bytes = int(rows[0]["bytes_per_iteration"])
    iterations = int(rows[0]["iterations"])
    before, after = c["device_state"]["before"], c["device_state"]["after"]
    compile_log = (RAW / run_id / "compile.log").read_text()
    a800 = json.loads((A800_REFS / f"add_rmsnorm_mixed_{kind}_service_sm80.json").read_text())
    contract = {
        "timer_domain": "cuda_event",
        "block_threads": 256,
        "vector_bytes_per_lane_operation": 16,
        "read_operations_per_group": 3,
        "write_operations_per_group": 2,
        "read_fraction": 0.6,
        "write_fraction": 0.4,
    }
    if kind == "l2":
        contract.update({
            "source_buffer_bytes": buffer_bytes,
            "destination_buffer_bytes": buffer_bytes,
            "combined_working_set_bytes": 2 * buffer_bytes,
        })
    else:
        contract.update({
            "allocated_source_buffer_bytes": buffer_bytes,
            "allocated_destination_buffer_bytes": buffer_bytes,
            "touched_source_bytes_per_iteration": buffer_bytes * 3 // 5,
            "touched_destination_bytes_per_iteration": buffer_bytes * 2 // 5,
            "combined_working_set_bytes": buffer_bytes,
        })
    contract.update({
        "hardware_l2_bytes": L2_BYTES,
        "iterations": iterations,
        "trials_per_grid": c["rotations"] * c["trials_per_process"],
        "grids": grids,
        "load_cache_operator": "ld.global.cg.v4.u32",
        "store_cache_operator": "st.global.wb.v4.u32",
    })
    if kind == "dram":
        contract.update({
            "kernel_registers_per_thread": _registers(compile_log, "a800_l2_read60_write40"),
            "kernel_static_shared_bytes": 0,
            "kernel_local_bytes": 0,
        })
    contract["validation"] = c["validation"]
    capture = {
        "owner": "h200-port-wp4_dram",
        "source_baseline_commit": git_commit(),
        "output": f"profiler_output/microbench/nvidia_h200/{run_id}",
        "physical_device_index": DEVICE_INDEX,
        "device_uuid": before["uuid"],
        "pci_bus_id": before["pci_bus_id"],
        "preflight_all_devices_idle": False,
        "preflight_owned_device_idle": True,
    }
    if kind == "dram":
        capture["capture_pids"] = None
    capture.update({
        "postflight_memory_mib": after["memory_used_mib"],
        "postflight_utilization_percent": after["gpu_utilization_pct"],
        "postflight_compute_processes": 0,
        "cleanup": "complete",
        "rotations": c["rotations"],
        "clock_gate": clock_summary(c),
        "note": "Only the owned GPU 1 was checked idle; the other H200 devices were in use by "
                "other workers.  capture_pids were not recorded by the runner.",
    })
    identity = {
        "probe_source": "microbench/nvidia_h200/memory/dram/directional_bandwidth.cu",
        "probe_source_sha256": c["source_sha256"],
        "runner": "microbench/nvidia_h200/memory/dram/directional_read60_write40_runner.py",
        "runner_sha256": c["runner_sha256"],
        "build_flags": c["build_flags"],
        "binary_sha256": c["binary_sha256"],
        "raw_sha256": c["raw_sha256"],
        "stderr_sha256": c["stderr_sha256"],
    }
    if kind == "dram":
        identity.update({
            "counter_sha256": None,
            "read_split_counter_sha256": None,
            "resource_dump_sha256": c["compile_log_sha256"],
            "sass_dump_sha256": None,
        })
    selected = {
        "selection_rule": "minimum per-grid median across the frozen full-chip grid sweep",
        "coupled_logical_byte_capacity_per_s": coupled,
        "read_logical_byte_capacity_per_s": coupled * 0.6,
        "write_logical_byte_capacity_per_s": coupled * 0.4,
        "composition": "For an exact 3:2 stream, read_bytes/read_capacity equals "
                       "write_bytes/write_capacity; their maximum is the one coupled service time.",
        "scope": "full_chip",
        "minimum_grid_ctas": min(grids),
        "maximum_grid_ctas": max(grids),
        "maximum_relative_trial_spread_percent": max(spreads.values()),
        "selected_grid_ctas": selected_grid,
        "read_logical_byte_capacity_per_s_by_grid_ctas": {
            g: medians[g] * 1e9 * 0.6 for g in medians},
        "write_logical_byte_capacity_per_s_by_grid_ctas": {
            g: medians[g] * 1e9 * 0.4 for g in medians},
    }
    applicability = dict(a800["applicability"])
    excluded = [x.replace("grids outside 108 through 1728 CTAs",
                          f"grids outside {min(grids)} through {max(grids)} CTAs")
                for x in a800["excluded_conclusions"]]
    payload = {
        "schema": f"nvidia_h200_add_rmsnorm_mixed_{kind}_service_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "evidence_kind": a800["evidence_kind"],
        "production_or_evaluation_duration_consumed": False,
        "capture": capture,
        "source_identity": identity,
    }
    if kind == "dram":
        payload["tool_identity"] = {
            "nvcc_sha256": None,
            "cuobjdump_sha256": None,
            "ncu_sha256": None,
            "nvcc_version": 'Cuda compilation tools, release 12.9 (capture toolchain)',
            "ncu_version": None,
            "driver_version": "575.57.08",
            "note": 'cuobjdump and Nsight Compute are not installed on capture host; no SASS dump or counter capture exists for this reference.',
        }
    payload.update({
        "probe_contract": contract,
        "median_total_capacity_gb_per_s": medians,
        "relative_trial_spread_percent": spreads,
    })
    if kind == "dram":
        payload["counter_validation"] = {
            **{key: None for key in a800["counter_validation"]},
            "status": "not_captured",
            "reason": "Nsight Compute is not installed on the H200 host, so the A800 "
                      "read-miss/write-sector counter check cannot be repeated.  The "
                      "working set (2 GiB) is 34 times the 60 MiB L2, and the same source "
                      "on A800 showed zero read-sector L2 hits at this geometry.",
        }
    payload["selected_service"] = selected
    payload["applicability"] = applicability
    reason = (f"The exact workload-independent 3:2 stream is stable across the frozen full-chip "
              f"{min(grids)}..{max(grids)}-CTA sweep, includes the 1024-CTA launch point, and "
              "supplies only a typed byte-service capacity.")
    if kind == "dram":
        reason = ("The exact workload-independent 3:2 stream has a working set 34 times larger "
                  "than the H200 L2, is stable across the frozen full-chip sweep, and supplies "
                  "only a typed byte-service capacity (no NCU counter proof on H200).")
    payload["decision"] = {"status": "accepted_mechanism_capacity", "forward_eligible": True,
                           "reason": reason}
    payload["excluded_conclusions"] = excluded
    return write(f"add_rmsnorm_mixed_{kind}_service_sm90.json", payload)


def mixed():
    return _mixed("l2"), _mixed("dram")


# --------------------------------------------------------------------------- #6 ramp
def ramp():
    captures = [load(run) for run in RUN["ramp"]]
    a800 = json.loads((A800_REFS / "dram_finite_read_stream_ramp_sm80.json").read_text())
    capacity = captures[0]["probe_contract"]["dram_read_capacity_bytes_per_s"]
    if any(c["probe_contract"]["dram_read_capacity_bytes_per_s"] != capacity for c in captures):
        raise ValueError("captures disagree on the capacity denominator")
    cells = {}
    for c in captures:
        for r in c["ramp"]:
            cells.setdefault((r["grid_ctas"], r["bytes_per_cta"]), []).append(
                r["achieved_fraction_of_dram_read_capacity"])
    published = sorted((k, max(v)) for k, v in cells.items()
                       if len(v) == len(captures) and k[0] <= SMS)
    # The saturated-capacity probe and this ramp probe measure the same
    # achievable DRAM read rate; long single-CTA-per-SM streams here run up
    # to ~2.4 % faster than dram_read_bw, so the published capacity is the
    # envelope of the two and every fraction is <= 1 by construction.  The
    # achieved rates (capacity x fraction) are unchanged.
    probe_capacity = capacity
    capacity = probe_capacity * max(1.0, max(f for _, f in published))
    published = [(k, f * probe_capacity / capacity) for k, f in published]
    ramp_rows = [{"achieved_fraction_of_dram_read_capacity": f, "bytes_per_cta": b,
                  "grid_ctas": g} for (g, b), f in published]
    grids = sorted({g for (g, _), _ in published})
    lengths = sorted({b for (_, b), _ in published})
    capture_rows = []
    for run, c in zip(RUN["ramp"], captures):
        capture_rows.append({
            "device_index": DEVICE_INDEX,
            "device_uuid": c["device_state"]["before"]["uuid"],
            "excluded_unstable_cells": c["excluded_unstable_cells"],
            "measurements_sha256": c["measurements_sha256"],
            "memory_clock_mhz": c["device_state"]["before"]["memory_clock_mhz"],
            "path": f"profiler_output/microbench/nvidia_h200/{run}",
            "sm_clock_mhz": min(a["busy_sm_clock_mhz_min"] for a in c["clock_gate"]["accepted"]),
            "trials": c["trials"],
            "clock_gate": clock_summary(c),
            "binary_sha256": c["binary_sha256"],
        })
    return write("dram_finite_read_stream_ramp_sm90.json", {
        "allowed_use": a800["allowed_use"],
        "derived_model": {
            "dram_read_capacity_bytes_per_s": capacity,
            "excluded_grids_reason": (
                "The probe allocates no shared memory, so above one CTA per SM several CTAs "
                f"co-schedule on a multiprocessor and the residency stops matching the "
                f"single-resident-CTA streams this ramp describes. Only grids up to {SMS} are "
                "published even where larger grids measured cleanly."),
            "grids": grids,
            "lengths_bytes_per_cta": lengths,
            "ramp": ramp_rows,
            "selection_rule": a800["derived_model"]["selection_rule"].replace(
                "two idle devices", "two independent clock-gated captures on the owned device"),
            "capacity_note": (
                "The denominator is the envelope of the H200 saturated read capacity from "
                "dram_read_capacity_protocol_sm90.json (dram_read_bw.cu, "
                f"{probe_capacity:.6g} B/s) and this probe's own long-stream asymptote: "
                "single-CTA-per-SM streams at 112-132 CTAs exceed dram_read_bw by up to about "
                "2.4 percent, so the larger measured rate is the capacity and every fraction "
                "is <= 1."),
            "saturated_probe_capacity_bytes_per_s": probe_capacity,
        },
        "forbidden_use": a800["forbidden_use"],
        "hardware_id": "nvidia_h200",
        "mechanism": "dram_finite_read_stream_ramp",
        "production_or_evaluation_duration_consumed": False,
        "provenance": {
            "captures": capture_rows,
            "runner": "microbench/nvidia_h200/memory/dram/dram_finite_stream_ramp_runner.py",
            "runner_sha256": captures[0]["runner_sha256"],
            "source": "microbench/nvidia_h200/memory/dram/dram_finite_stream_ramp.cu",
            "source_sha256": captures[0]["source_sha256"],
            "git_commit": git_commit(),
        },
        "schema": a800["schema"],
        "scope": {
            "in_flight_bytes_per_cta": captures[0]["probe_contract"]["in_flight_bytes_per_cta"],
            "load_instruction": "ld.global.cg.v4.u32",
            "maximum_bytes_per_cta": max(lengths),
            "maximum_grid_ctas": max(grids),
            "minimum_bytes_per_cta": min(lengths),
            "minimum_grid_ctas": min(grids),
            "threads_per_cta": 128,
        },
    })


# --------------------------------------------------------------------------- #7 residency
def residency():
    captures = [load(run) for run in RUN["residency"]]
    a800 = json.loads((A800_REFS / "l2_operand_residency_curve_sm80.json").read_text())
    resident_ctas = captures[0]["probe_contract"]["resident_ctas"]
    rates, excluded = {}, {}
    for c in captures:
        for e in c["curve"]:
            mib = e["total_footprint_mib"]
            words = mib * 1024 * 1024 // 3 // 16
            grid = min((words + 127) // 128, resident_ctas)
            if words / (grid * 128) < 2:
                excluded[str(mib * 1024 * 1024)] = (
                    "The probe's grid-stride inner loop performs fewer than two iterations per "
                    "thread at this footprint, so its rate reflects loop shape rather than "
                    "memory service. The measurement is retained in the capture but excluded "
                    "from the published domain.")
                continue
            rates.setdefault(mib, []).append(e["logical_byte_rate_per_s"])
    conservative = {m: min(v) for m, v in rates.items()}
    plateau_limit = 0.675 * L2_BYTES / 2 ** 20    # A800: 27 MiB of 40 MiB
    asymptote_limit = 1.35 * L2_BYTES / 2 ** 20   # A800: 54 MiB of 40 MiB
    plateau = max(v for m, v in conservative.items() if m <= plateau_limit)
    asymptote = min(v for m, v in conservative.items() if m >= asymptote_limit)
    running, curve = 0.0, []
    for mib in sorted(conservative):
        rate = conservative[mib]
        fraction = (1 / rate - 1 / plateau) / (1 / asymptote - 1 / plateau)
        running = max(running, min(1.0, max(0.0, fraction)))
        curve.append({"conservative_logical_byte_rate_per_s": rate,
                      "dram_served_byte_fraction": running,
                      "total_footprint_bytes": mib * 1024 * 1024})
    capture_rows = []
    for run, c in zip(RUN["residency"], captures):
        capture_rows.append({
            "binary_sha256": c["binary_sha256"],
            "device_index": DEVICE_INDEX,
            "device_uuid": c["device_state"]["before"]["uuid"],
            "maximum_relative_spread": max(e["relative_spread"] for e in c["curve"]),
            "measurements_sha256": c["measurements_sha256"],
            "memory_clock_mhz": c["device_state"]["before"]["memory_clock_mhz"],
            "path": f"profiler_output/microbench/nvidia_h200/{run}",
            "sass_sha256": None,
            "sm_clock_mhz": min(a["busy_sm_clock_mhz_min"] for a in c["clock_gate"]["accepted"]),
            "trials": c["trials"],
            "clock_gate": clock_summary(c),
            "spread_gate_exceeded_footprints_mib": [
                e["total_footprint_mib"] for e in c["curve"] if e["spread_gate_exceeded"]],
        })
    selection = a800["derived_model"]["selection_rule"].replace(
        "two idle devices", "two independent clock-gated captures on the owned device").replace(
        "at or below 27 MiB", f"at or below {plateau_limit:.1f} MiB (0.675x L2, the A800 27/40 ratio)").replace(
        "at or above 54 MiB", f"at or above {asymptote_limit:.1f} MiB (1.35x L2, the A800 54/40 ratio)")
    return write("l2_operand_residency_curve_sm90.json", {
        "allowed_use": a800["allowed_use"],
        "derived_model": {
            "dram_asymptote_logical_byte_rate_per_s": asymptote,
            "excluded_footprints": excluded,
            "formula": a800["derived_model"]["formula"],
            "residency_curve": curve,
            "resident_plateau_logical_byte_rate_per_s": plateau,
            "selection_rule": selection,
            "spread_gate_note": (
                "The A800 runner aborts a capture whose footprint trial spread exceeds 8 percent. "
                "On H200 the 45..54 MiB transition exceeds it in every capture (also with 256 "
                "warm-up passes), so the H200 runner records the footprints instead "
                "(provenance.captures[].spread_gate_exceeded_footprints_mib) and the median rate "
                "is used as on A800."),
        },
        "forbidden_use": a800["forbidden_use"],
        "hardware_id": "nvidia_h200",
        "mechanism": "l2_operand_residency_curve",
        "production_or_evaluation_duration_consumed": False,
        "provenance": {
            "captures": capture_rows,
            "runner": "microbench/nvidia_h200/memory/l2/l2_residency_knee_runner.py",
            "runner_sha256": captures[0]["runner_sha256"],
            "source": "microbench/nvidia_h200/memory/l2/l2_residency_knee.cu",
            "source_sha256": captures[0]["source_sha256"],
            "git_commit": git_commit(),
            "warmup_passes": captures[0]["probe_contract"]["warmup_passes"],
        },
        "schema": a800["schema"],
        "scope": {
            "l2_capacity_bytes": L2_BYTES,
            "maximum_footprint_bytes": max(e["total_footprint_bytes"] for e in curve),
            "minimum_footprint_bytes": min(e["total_footprint_bytes"] for e in curve),
            "read_instruction": "ld.global.cg.v4.u32",
            "stream": "two_read_one_write_vec4",
            "threads_per_cta": 128,
            "write_policy": "default_write_back",
        },
    })


# --------------------------------------------------------------------------- #1 L2 bandwidth
def l2_bandwidth():
    run_id = RUN["l2_bandwidth"]
    c = load(run_id)
    rows = list(csv.DictReader((RAW / run_id / "measurements.csv").open()))
    cells = {}
    for r in rows:
        cells.setdefault((int(r["footprint_mib"]), int(r["grid"])), []).append(
            float(r["logical_read_gbs"]))
    conservative = {k: min(v) for k, v in cells.items()}
    best = max(conservative, key=conservative.get)
    gbs = conservative[best]
    return write("l2_read_bandwidth_sm90.json", {
        "schema": "nvidia_h200_l2_read_bandwidth_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "production_or_evaluation_duration_consumed": False,
        "method": (
            "The A800 value (2921.842 GB/s) was read from NCU for this probe; NCU is not "
            "installed on the H200 host.  The probe times each cell itself with CUDA events "
            "(median of nine trials of one kernel reading every byte of an L2-resident "
            "footprint 16-32 GiB of logical traffic); logical bytes over event time measures "
            "the same sustained ld.global.cg L2 read service as the NCU L2 throughput because "
            "the footprints (8-32 MiB) are warmed into the 60 MiB L2 and bypass L1, so every "
            "logical byte is one L2 read."),
        "selection_rule": "per (footprint, grid) cell the smaller of the two process medians; "
                          "the published value is the largest such conservative cell "
                          "(sustained envelope over the sweep)",
        "cells_gb_per_s": {f"{m}MiB/{g}": v for (m, g), v in sorted(cells.items())},
        "selected_cell": {"footprint_mib": best[0], "grid": best[1]},
        "l2_read_gb_per_s": gbs,
        "sm_clock_hz": SM_CLOCK_HZ,
        "throughput_per_clk": gbs * 1e9 / SM_CLOCK_HZ,
        "provenance": {
            "source": "microbench/nvidia_h200/memory/l2/l2_bandwidth.cu",
            "source_sha256": c["source_sha256"],
            "runner": "microbench/nvidia_h200/memory/l2/l2_bandwidth_runner.py",
            "runner_sha256": c["runner_sha256"],
            "binary_sha256": c["binary_sha256"],
            "raw": f"profiler_output/microbench/nvidia_h200/{run_id}",
            "measurements_sha256": c["measurements_sha256"],
            "clock": clock_summary(c),
            "device_index": DEVICE_INDEX,
            "device_uuid": c["device_state"]["before"]["uuid"],
            "git_commit": git_commit(),
        },
    })


# --------------------------------------------------------------------------- #5 scalar half
def scalar_half(submit_ns=None, submit_source=None):
    run_id = RUN["scalar_half"]
    c = load(run_id)
    rows = list(csv.DictReader((RAW / run_id / "results.csv").open()))
    latency_ref = json.loads((REFS / "pointer_chase_latency_sm90.json").read_text())
    dram_latency_ns = latency_ref["dram_latency_ns"]
    resident_grid = SMS * 3
    long_groups = max(int(r["groups_per_cta"]) for r in rows if int(r["grid"]) == resident_grid)
    long_rate = statistics.median(
        float(r["sectors_per_active_sm_s"]) for r in rows
        if int(r["grid"]) == resident_grid and int(r["groups_per_cta"]) == long_groups)
    group3 = [r for r in rows if int(r["grid"]) == resident_grid and int(r["groups_per_cta"]) == 3]
    group3_ns = statistics.median(float(r["elapsed_ms"]) * 1e6 for r in group3)
    group3_bytes = int(group3[0]["bytes"])
    steady_ns = group3_bytes / (long_rate * SMS * 32) * 1e9
    medians = {}
    for key in sorted({(int(r["grid"]), int(r["groups_per_cta"])) for r in rows}):
        cell = [r for r in rows if (int(r["grid"]), int(r["groups_per_cta"])) == key]
        medians[f"{key[0]}/{key[1]}"] = {
            "median_elapsed_ns": statistics.median(float(r["elapsed_ms"]) * 1e6 for r in cell),
            "median_sectors_per_active_sm_s": statistics.median(
                float(r["sectors_per_active_sm_s"]) for r in cell),
            "bytes": int(cell[0]["bytes"]), "trials": len(cell)}
    fill = None if submit_ns is None else group3_ns - steady_ns - submit_ns - dram_latency_ns
    return write("scalar_half_dram_stream_sm90.json", {
        "schema": "nvidia_h200_scalar_half_dram_stream_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "production_or_evaluation_duration_consumed": False,
        "probe_meta": c["probe_meta"],
        "cell_medians": medians,
        "gemv_scalar_half_issue_group_elements_per_lane": 64,
        "gemv_scalar_half_resident_ctas_per_sm": 3,
        "gemv_scalar_half_dram_transactions_per_sm_s": long_rate,
        "three_group_median_ns": group3_ns,
        "three_group_bytes": group3_bytes,
        "three_group_steady_service_ns": steady_ns,
        "dram_latency_ns": dram_latency_ns,
        "submit_ns": submit_ns,
        "submit_source": submit_source if submit_ns is not None else "pending WP7 submit_ns",
        "gemv_scalar_half_source_fill_ns": fill,
        "formula": ("transactions_per_sm_s = median sectors_per_active_sm_s of the long stream "
                    "at grid 3 x 132; steady_ns = three_group_bytes / (rate * 132 * 32) ; "
                    "source_fill_ns = three_group_median_ns - steady_ns - submit_ns - "
                    "dram_latency_ns (A800 selection rule, same formula)"),
        "fill_before_submit_ns": group3_ns - steady_ns - dram_latency_ns,
        "provenance": {
            "source": "microbench/nvidia_h200/memory/lsu/scalar_half_dram_stream.cu",
            "source_sha256": c["source_sha256"],
            "runner": "microbench/nvidia_h200/memory/lsu/scalar_half_dram_stream_runner.py",
            "runner_sha256": c["runner_sha256"],
            "binary_sha256": c["binary_sha256"],
            "raw": f"profiler_output/microbench/nvidia_h200/{run_id}",
            "results_sha256": c["results_sha256"],
            "clock": clock_summary(c),
            "device_index": DEVICE_INDEX,
            "device_uuid": c["device_state"]["before"]["uuid"],
            "git_commit": git_commit(),
        },
    })


# --------------------------------------------------------------------------- schema check
def _keys(obj, prefix=""):
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            keys.add(path)
            if not k.isdigit():
                keys |= _keys(v, path)
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        keys |= _keys(obj[0], prefix + "[]")
    return keys


def schema_check():
    lines = ["| H200 reference | A800 reference | A800 keys | missing in H200 | extra in H200 | hardware_id |",
             "|---|---|---|---|---|---|"]
    for h200_name, a800_name in PAIRS.items():
        h = json.loads((REFS / h200_name).read_text())
        a = json.loads((A800_REFS / a800_name).read_text())
        ha, aa = _keys(h), _keys(a)
        # grid-keyed maps use numeric keys; compare them by name only
        aa = {k for k in aa if not any(p.isdigit() for p in k.replace("[]", "").split("."))}
        missing = sorted(aa - ha)
        extra = sorted(k for k in ha - aa
                       if not any(p.isdigit() for p in k.replace("[]", "").split(".")))
        extra_groups = sorted({".".join(k.split(".")[:2]).replace("[]", "") for k in extra})
        lines.append(f"| {h200_name} | {a800_name} | {len(aa)} | {', '.join(missing) or 'none'} "
                     f"| {len(extra)} (under {', '.join(extra_groups) or 'none'}) "
                     f"| {h['hardware_id']} |")
    text = "\n".join(lines)
    print(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["capacity", "latency", "mixed", "ramp", "residency",
                                           "l2_bandwidth", "scalar_half", "schema-check", "all"])
    parser.add_argument("--submit-ns", type=float)
    parser.add_argument("--submit-source")
    arguments = parser.parse_args()
    if arguments.target in ("capacity", "all"):
        dram_capacity()
    if arguments.target in ("latency", "all"):
        latency()
    if arguments.target in ("mixed", "all"):
        mixed()
    if arguments.target in ("ramp", "all"):
        ramp()
    if arguments.target in ("residency", "all"):
        residency()
    if arguments.target in ("l2_bandwidth", "all"):
        l2_bandwidth()
    if arguments.target in ("scalar_half", "all"):
        scalar_half(arguments.submit_ns, arguments.submit_source)
    if arguments.target in ("schema-check", "all"):
        schema_check()


if __name__ == "__main__":
    main()
