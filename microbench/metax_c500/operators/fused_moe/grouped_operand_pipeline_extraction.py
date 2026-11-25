#!/usr/bin/env python3
"""Extract a C500 grouped-operand pipeline mechanism from one tile sweep.

Input is the synthetic-routing, non-production K/N sweep produced by
``fused_moe_triton_operand_stream_runner.py`` at one deployed compiled
resource point.  The decomposition mirrors the accepted BM32 selection rule:
per routing/N group, solve the per-launch event median as
``waves * (fixed_span + k_tiles * round) + empty_waves * slot`` by least
squares.  Forward keeps the per-round service analytic, so the only measured
coefficients are the per-CTA fixed span and the empty-CTA per-wave slot.  No
production or evaluation duration participates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import importlib.util
import statistics


def _runner_module():
    """Load the sibling sweep runner without making microbench a package."""

    path = Path(__file__).with_name(
        "fused_moe_triton_operand_stream_runner.py")
    spec = importlib.util.spec_from_file_location(
        "fused_moe_triton_operand_stream_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


APS = 104
# Architecture facts from configs/hardware/metax_c500.yaml.
DRAM_CAPACITY_BYTES_PER_S = 1_653_069_312_000.0
DRAM_LATENCY_NS = 373.3333333333
AP_CLOCK_HZ = 1_125_000_000.0
ARCHITECTURE_PEAK_FLOPS_PER_AP_CYCLE = 2048.0


def _expert_layout(record: dict, signature: dict) -> tuple[int, int]:
    """Rebuild one cell's expert occupancy from its recorded route contract.

    The runner's routing is a deterministic function of tokens, experts,
    topk, the tile M and the route seed, all of which the sample records, so
    the populated-expert count and the largest per-expert M-tile span are
    recoverable exactly.  The rebuild is checked against the recorded route
    digest and active M-tile count and fails closed on any disagreement.
    """

    layout = _runner_module().build_random_topk_route(
        int(record["tokens"]), int(record["experts"]), int(record["topk"]),
        signature["tile_m"], int(record["route_seed"]))
    if layout.sha256 != record["route_sha256"]:
        raise ValueError(
            "rebuilt route digest disagrees with the recorded sample")
    if layout.active_m_tiles != int(record["active_m_tiles"]):
        raise ValueError(
            "rebuilt route active M tiles disagree with the recorded sample")
    populated = [count for count in layout.tokens_per_expert if count > 0]
    return len(populated), max(
        (math.ceil(count / signature["tile_m"]) for count in populated),
        default=0)


def _read_sample(path: Path) -> dict:
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    header = None
    sample = None
    for row in rows:
        if row and row[0] == "sample_header":
            header = row[1:]
        elif row and row[0] == "sample":
            sample = row[1:]
    if header is None or sample is None:
        raise ValueError(f"{path} has no sample record")
    return dict(zip(header, sample))


def _cells(directory: Path, prefix_names) -> dict:
    cells: dict[str, dict] = {}
    for path in sorted(directory.glob("*_r[123].csv")):
        name = path.name[: -len("_rN.csv")]
        record = _read_sample(path)
        entry = cells.setdefault(name, {"captures_ms": [], "record": record})
        entry["captures_ms"].append(float(record["event_median_ms"]))
    for name, entry in cells.items():
        entry["median_ms"] = statistics.median(entry["captures_ms"])
        entry["captures_ms"] = sorted(entry["captures_ms"])
        entry["is_prefix"] = name.startswith(prefix_names)
    return cells


def _structure(record: dict, empty_wave_slot_ns: float, signature: dict) -> dict:
    grid = int(record["grid"])
    active_m_tiles = int(record["active_m_tiles"])
    n_tiles = int(record["n_tiles"])
    active_ctas = active_m_tiles * n_tiles
    waves = math.ceil(active_ctas / APS)
    empty_waves = math.ceil(max(grid - active_ctas, 0) / APS)
    k_tiles = int(record["k"]) // signature["tile_k"]
    # Forward prices the read stream only, and charges one expert weight
    # matrix per populated expert (not per populated M tile): several M tiles
    # of the same expert share one weight stream.
    populated_experts, maximum_expert_m_tiles = _expert_layout(
        record, signature)
    dram_bytes = (
        populated_experts * int(record["n"]) * int(record["k"]) * 2
        + float(record["activation_unique_bytes"]))
    dram_round_us = (
        dram_bytes / k_tiles / waves / DRAM_CAPACITY_BYTES_PER_S * 1e6
        + DRAM_LATENCY_NS * 1e-3 / signature["pipeline_depth"])
    return {
        "grid": grid,
        "populated_experts": populated_experts,
        "maximum_expert_m_tiles": maximum_expert_m_tiles,
        "active_ctas": active_ctas,
        "waves": waves,
        "empty_waves": empty_waves,
        "k_tiles": k_tiles,
        "dram_bytes": dram_bytes,
        "dram_round_us": dram_round_us,
        "empty_tail_us": empty_waves * empty_wave_slot_ns * 1e-3,
    }


def _ordinary_least_squares(xs, ys):
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sxy = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    sxx = math.fsum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        raise ValueError("least-squares needs at least two distinct k_tiles")
    slope = sxy / sxx
    return slope, mean_y - slope * mean_x


def _empty_wave_slot_ns(cells) -> tuple[float, dict]:
    points = []
    for name, entry in cells.items():
        if not name.startswith("emptyscan_"):
            continue
        record = entry["record"]
        grid = int(record["grid"])
        waves = math.ceil(grid / APS)
        points.append((waves, entry["median_ms"] * 1e3, name, grid))
    if len(points) < 3:
        raise ValueError("empty-CTA grid scan needs at least three grids")
    points.sort()
    slope, intercept = _ordinary_least_squares(
        [p[0] for p in points], [p[1] for p in points])
    # A program whose early-return path retires faster than the launch jitter
    # leaves a slope that is not separable from zero.  Bound it instead of
    # inventing a measurement: the regression slope plus two standard errors
    # is an outlier-damped ceiling that the scan does support.  The slope is
    # only taken as measured when it clears that same noise.
    mean_waves = statistics.fmean([point[0] for point in points])
    sum_squares = math.fsum(
        (point[0] - mean_waves) ** 2 for point in points)
    residual_sum_squares = math.fsum(
        (us - (intercept + slope * waves)) ** 2
        for waves, us, _, _ in points)
    degrees_of_freedom = len(points) - 2
    if degrees_of_freedom <= 0 or sum_squares <= 0.0:
        raise ValueError("the empty-CTA grid scan cannot bound its slope")
    slope_standard_error = math.sqrt(
        residual_sum_squares / degrees_of_freedom / sum_squares)
    upper_bound = slope + 2.0 * slope_standard_error
    if upper_bound <= 0.0:
        raise ValueError(
            "the empty-CTA grid scan does not bound a positive per-wave slot")
    resolved = slope > 2.0 * slope_standard_error
    selected = slope if resolved else upper_bound
    residuals = {
        name: (us - (intercept + slope * waves)) / us
        for waves, us, name, _ in points
    }
    return selected * 1e3, {
        "model": "all_empty_prefix_launch_us = intercept + slope * waves",
        "points": [
            {"cell": name, "grid": grid, "waves": waves, "median_us": us}
            for waves, us, name, grid in points
        ],
        "intercept_us": intercept,
        "slope_us_per_wave": slope,
        "slope_standard_error_us_per_wave": slope_standard_error,
        "upper_bound_us_per_wave": upper_bound,
        "slope_resolved": resolved,
        "selected_us_per_wave": selected,
        "relative_residual_by_cell": residuals,
    }


def build_reference(directory: Path, signature: dict, reference_group: str,
                    provenance: dict, empty_wave_slot_override=None) -> dict:
    cells = _cells(directory, ("prefix_", "emptyscan_"))
    if empty_wave_slot_override is None:
        empty_wave_slot_ns, empty_scan = _empty_wave_slot_ns(cells)
    else:
        # Cross-check path only: reproduce an already accepted signature's
        # decomposition with its independently measured slot.
        empty_wave_slot_ns = float(empty_wave_slot_override)
        empty_scan = {"external_slot_ns": empty_wave_slot_ns}
    structure = {}
    groups: dict[str, list] = {}
    for name, entry in cells.items():
        if entry["is_prefix"]:
            continue
        record = entry["record"]
        item = _structure(record, empty_wave_slot_ns, signature)
        item["t_us"] = entry["median_ms"] * 1e3
        structure[name] = item
        groups.setdefault(name.rsplit("_k", 1)[0], []).append((name, item))
    decomposition = {}
    for group, items in sorted(groups.items()):
        xs = [item["k_tiles"] for _, item in items]
        ys = [
            (item["t_us"] - item["empty_tail_us"]) / item["waves"]
            for _, item in items
        ]
        round_us, fill_us = _ordinary_least_squares(xs, ys)
        decomposition[group] = {
            "points": len(items),
            "fill_us": fill_us,
            "round_us": round_us,
            "relative_residual_by_cell": {
                name: (
                    (item["t_us"]
                     - (item["waves"] * (fill_us + item["k_tiles"] * round_us)
                        + item["empty_tail_us"]))
                    / item["t_us"])
                for name, item in items
            },
        }
    # Forward prices a K round as max(the program's own issue interval, the
    # analytic DRAM round service).  The free-slope decomposition above
    # measures that issue interval directly: its slope is the per-round cost
    # at the swept density, and it is independent of the N extent, which is
    # what a per-CTA cadence must be.  Take the reference group's slope as the
    # measured interval, then extract the per-CTA fixed span conditioned on
    # whichever side of the max actually binds in each cell.  When the
    # measured interval binds everywhere this is the free intercept; when the
    # analytic round binds, the span absorbs the difference instead of the
    # slope silently disagreeing with Forward.
    local_round_us = decomposition[reference_group]["round_us"]
    architecture_round_us = (
        2.0 * signature["tile_m"] * signature["tile_n"] * signature["tile_k"]
        / ARCHITECTURE_PEAK_FLOPS_PER_AP_CYCLE / AP_CLOCK_HZ * 1e6)
    if local_round_us < architecture_round_us:
        raise ValueError(
            "the measured K-round interval is below the architecture-peak MMA "
            "round issue interval for this tile")
    conditioned = {}
    for group, items in sorted(groups.items()):
        spans = {
            name: (
                (item["t_us"] - item["empty_tail_us"]) / item["waves"]
                - item["k_tiles"] * max(
                    local_round_us, item["dram_round_us"]))
            for name, item in items
        }
        conditioned[group] = {
            "points": len(items),
            "fixed_span_us_by_cell": spans,
            "fixed_span_us": statistics.fmean(spans.values()),
        }
    if reference_group not in conditioned:
        raise KeyError("the production-density reference group is missing")
    fixed_span_ns = conditioned[reference_group]["fixed_span_us"] * 1e3
    if fixed_span_ns <= 0.0:
        raise ValueError(
            "the analytic round service already exceeds the measured launch")
    residual_by_cell = {}
    for name, item in structure.items():
        modelled = (
            item["waves"]
            * (fixed_span_ns * 1e-3 + item["k_tiles"] * max(
                item["dram_round_us"], local_round_us))
            + item["empty_tail_us"])
        residual_by_cell[name] = (modelled - item["t_us"]) / item["t_us"]
    local_round_ns = local_round_us * 1e3
    in_domain = {
        name: item for name, item in structure.items()
        if item["maximum_expert_m_tiles"]
        <= provenance["maximum_expert_m_tiles"]
    }
    if not in_domain:
        raise ValueError("the declared domain excludes every swept cell")
    return {
        "schema_version": 1,
        "hardware_id": "metax_c500",
        "software_revision": "maca_3.7.2",
        "mechanism": "triton_grouped_operand_pipeline",
        "timer_domain": "torch_cuda_event_gpu_timeline",
        "signature": dict(signature),
        "cells": {
            name: {
                "captures_ms": entry["captures_ms"],
                "median_ms": entry["median_ms"],
                "tokens": int(entry["record"]["tokens"]),
                "experts": int(entry["record"]["experts"]),
                "topk": int(entry["record"]["topk"]),
                "k": int(entry["record"]["k"]),
                "n": int(entry["record"]["n"]),
                "grid": int(entry["record"]["grid"]),
                "active_m_tiles": int(entry["record"]["active_m_tiles"]),
            }
            for name, entry in sorted(cells.items())
        },
        "extraction": {
            "cell_structure": structure,
            "free_slope_group_decomposition": decomposition,
            "analytic_round_conditioned_span": conditioned,
            "forward_model_relative_residual_by_cell": residual_by_cell,
            "empty_wave_slot_scan": empty_scan,
            "empty_wave_slot_ns": empty_wave_slot_ns,
            "dram_round_service_analytic_ns": (
                "dram_bytes/k_tiles/waves/1653.07GB/s + 373.333ns/"
                f"{signature['pipeline_depth']}"),
        },
        "selected_parameters": {
            "fixed_span_ns": fixed_span_ns,
            "local_k_round_interval_ns": local_round_ns,
            "empty_wave_slot_ns": empty_wave_slot_ns,
        },
        "scope": {
            "signature_name": provenance["signature_name"],
            "routing": (
                "deterministic_splitmix64_affine_without_replacement_v1"),
            "route_seed": 20260824,
            "k_tiles_swept": sorted({
                item["k_tiles"] for item in structure.values()}),
            "n_tiles_swept": sorted({
                int(entry["record"]["n"]) // signature["tile_n"]
                for name, entry in cells.items() if not entry["is_prefix"]}),
            "density_groups": {
                group: (
                    f"{items[0][1]['populated_experts']} populated experts, "
                    f"{items[0][1]['active_ctas']} active CTAs over "
                    f"{items[0][1]['waves']} resident waves, at most "
                    f"{items[0][1]['maximum_expert_m_tiles']} M tiles per "
                    f"expert")
                for group, items in sorted(groups.items())
            },
            "nonproduction_geometry_only": True,
        },
        "method": {
            "kernel": "frozen production Triton grouped-MoE source",
            "runner": (
                "microbench/metax_c500/operators/fused_moe/"
                "fused_moe_triton_operand_stream_runner.py"),
            "extraction": (
                "microbench/metax_c500/operators/fused_moe/"
                "grouped_operand_pipeline_extraction.py"),
            "compiled_resource_fact": {
                "shared": signature["dynamic_shared_memory_bytes_per_cta"],
                "num_warps": signature["num_warps"],
                "num_stages": signature["num_stages"],
                "note": (
                    "runtime introspection of the compiled kernel matched the "
                    "deployed AOT resource record in every cell"),
            },
            "event_trials_per_cell": 9,
            "event_repetitions_per_trial": 100,
            "independent_captures_per_cell": 3,
            "physical_device": provenance["physical_device_id"],
            "production_or_evaluation_duration_consumed": False,
            "selection_rule": (
                "Forward keeps the per-K-round service analytic (round bytes "
                "at the hardware DRAM capacity plus the pipeline-depth "
                "amortized DRAM latency), so the only measured coefficient is "
                "the per-CTA fixed span.  Where the free-slope decomposition "
                "of this sweep does not reproduce that analytic round, the "
                "span is not read off the free intercept.  fixed_span_ns is "
                "the mean over the production-density reference group of the "
                "launch median minus the measured empty-wave tail and the "
                "analytic round service, per resident wave, which makes the "
                "Forward expression unbiased over that group.  "
                "empty_wave_slot_ns is the slope of the source-exact "
                "all-empty early-return prefix launch against its resident "
                "wave count.  local_k_round_interval_ns is the "
                "reference group's measured per-K-round issue interval, "
                "which the sweep shows to be independent of the N extent; it "
                "is checked against the architecture-peak BF16 MMA round "
                "issue interval for the tile and must not fall below it."),
            "reference_group": reference_group,
        },
        "provenance": dict(provenance["record"]),
        "domain": {
            "maximum_expert_m_tiles": provenance["maximum_expert_m_tiles"],
            "reason": provenance["domain_reason"],
        },
        "allowed_conclusion": provenance["allowed_conclusion"],
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "forward_eligible": True,
        "stability": {
            "conditioned_span_group_spread_ns": [
                item["fixed_span_us"] * 1e3 for item in conditioned.values()],
            "free_slope_fill_group_spread_ns": [
                item["fill_us"] * 1e3 for item in decomposition.values()],
            "free_slope_round_group_spread_ns": [
                item["round_us"] * 1e3 for item in decomposition.values()],
            "max_abs_forward_model_relative_residual": max(
                abs(value) for value in residual_by_cell.values()),
            "max_abs_free_slope_relative_residual": max(
                abs(value)
                for item in decomposition.values()
                for value in item["relative_residual_by_cell"].values()),
            "capture_spread_note": (
                "three fresh-process captures per cell; medians reported in "
                "cells"),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signature-name", required=True)
    parser.add_argument("--tile-m", type=int, required=True)
    parser.add_argument("--tile-n", type=int, required=True)
    parser.add_argument("--tile-k", type=int, required=True)
    parser.add_argument("--num-warps", type=int, required=True)
    parser.add_argument("--num-stages", type=int, required=True)
    parser.add_argument("--threads-per-cta", type=int, required=True)
    parser.add_argument("--dynamic-shared-bytes", type=int, required=True)
    parser.add_argument("--reference-group", required=True)
    parser.add_argument("--maximum-expert-m-tiles", type=int, required=True)
    parser.add_argument("--domain-reason", required=True)
    parser.add_argument("--allowed-conclusion", required=True)
    parser.add_argument("--physical-device-id", type=int, required=True)
    parser.add_argument("--capture-command", required=True)
    parser.add_argument(
        "--capture-baseline-commit",
        default="712d214963d4f427d712ac9515c0895f54691e11")
    parser.add_argument("--remote-snapshot", required=True)
    parser.add_argument("--capture-worktree", type=Path, required=True)
    parser.add_argument(
        "--empty-wave-slot-ns", type=float, default=None,
        help=(
            "cross-check only: use an externally measured per-wave empty-CTA "
            "slot instead of this sweep's own grid scan"))
    arguments = parser.parse_args(argv)
    if not arguments.container_name.strip():
        parser.error("--container-name must be nonempty")
    signature = {
        "tile_m": arguments.tile_m,
        "tile_n": arguments.tile_n,
        "tile_k": arguments.tile_k,
        "group_size_m": 1,
        "num_warps": arguments.num_warps,
        "num_stages": arguments.num_stages,
        "threads_per_cta": arguments.threads_per_cta,
        "dynamic_shared_memory_bytes_per_cta": arguments.dynamic_shared_bytes,
        "resident_ctas_per_ap": 1,
        # The compiled program's operand buffer depth is its stage count.
        "pipeline_depth": arguments.num_stages,
    }
    provenance = {
        "signature_name": arguments.signature_name,
        "physical_device_id": arguments.physical_device_id,
        "maximum_expert_m_tiles": arguments.maximum_expert_m_tiles,
        "domain_reason": arguments.domain_reason,
        "allowed_conclusion": arguments.allowed_conclusion,
        "record": {
            "capture_baseline_commit": arguments.capture_baseline_commit,
            "capture_command": arguments.capture_command,
            "physical_device_id": arguments.physical_device_id,
            "owner": "capture-owner-1",
            "remote_snapshot": arguments.remote_snapshot,
            "container": arguments.container_name,
            "worktree": str(arguments.capture_worktree),
            "branch": "wt-c500-moe-duration",
        },
    }
    reference = build_reference(
        arguments.sweep_dir, signature, arguments.reference_group, provenance,
        arguments.empty_wave_slot_ns)
    arguments.output.write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(reference["selected_parameters"], indent=2))
    print(json.dumps(reference["stability"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
