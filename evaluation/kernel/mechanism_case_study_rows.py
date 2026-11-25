"""Hardware-specific mechanism case-study rows: Ascend MoE tile
replay and MetaX C500 SiLU mechanism evidence."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence
import numpy as np

from evaluation.kernel.mechanism_capture import (
    CASE_APE_LIMIT_PCT,
    canonical_capture_comparison_evidence,
)


def ascend_moe_tile_replay_rows(
        documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Compare production GroupedMatmul paths with matching interfaces."""

    rows = []
    for document in documents:
        if document["hardware_id"] != "ascend_910b":
            continue
        for task in document["tasks"]:
            if task["family"] != "FusedMoE":
                continue
            for kernel in task["kernels"]:
                mechanism = kernel["mechanisms"]
                variant = mechanism["variant"]
                if variant not in ("w1", "w2"):
                    continue
                launch = mechanism["observed_launch"]
                tiling = launch["tiling"]
                if tiling is None:
                    raise ValueError(
                        "Ascend GroupedMatmul replay check requires tiling")
                tile_shape = tiling["tile_shape"]
                tile_m = int(tile_shape["m"])
                tile_n = int(tile_shape["n"])
                tile_k = int(tile_shape["k"])
                if min(tile_m, tile_n, tile_k) <= 0:
                    raise ValueError(
                        "Ascend GroupedMatmul replay tiles must be positive")
                paths = mechanism["analytical_input"]["memory_paths"]
                if not isinstance(paths, list):
                    raise TypeError(
                        "Ascend GroupedMatmul memory paths must be a list")
                paths_by_name = {path["name"]: path for path in paths}
                if (len(paths_by_name) != len(paths)
                        or set(paths_by_name)
                        != {
                            "mte2_read", "operand_dram_read",
                            "aic_output_l2_write",
                        }):
                    raise ValueError(
                        "Ascend GroupedMatmul requires one MTE2 L2-read path, "
                        "one operand DRAM-read path, and one AIC output "
                        "L2-write path")
                read_path = paths_by_name["mte2_read"]
                dram_read_path = paths_by_name["operand_dram_read"]
                output_path = paths_by_name["aic_output_l2_write"]
                if (read_path["access_kind"] != "read"
                        or dram_read_path["access_kind"] != "read"
                        or output_path["access_kind"] != "write"):
                    raise ValueError(
                        "Ascend GroupedMatmul path access kinds are invalid")
                read_nodes = {
                    node["resource_role"]: float(node["bytes"])
                    for node in read_path["nodes"]
                }
                dram_read_nodes = {
                    node["resource_role"]: float(node["bytes"])
                    for node in dram_read_path["nodes"]
                }
                output_nodes = {
                    node["resource_role"]: float(node["bytes"])
                    for node in output_path["nodes"]
                }
                if set(read_nodes) != {"l2_cache"}:
                    raise ValueError(
                        "Ascend GroupedMatmul MTE2 read nodes are invalid")
                if set(dram_read_nodes) != {"dram"}:
                    raise ValueError(
                        "Ascend GroupedMatmul operand DRAM-read nodes are "
                        "invalid")
                if set(output_nodes) != {"l2_cache", "dram"}:
                    raise ValueError(
                        "Ascend GroupedMatmul output-write nodes are invalid")
                if (dram_read_nodes["dram"] > read_nodes["l2_cache"]
                        or output_nodes["dram"] > output_nodes["l2_cache"]):
                    raise ValueError(
                        "Ascend GroupedMatmul DRAM continuation cannot exceed "
                        "its directional L2 interface")
                predicted_read_bytes = read_nodes["l2_cache"]
                predicted_output_bytes = output_nodes["l2_cache"]
                observed_read_bytes = float(
                    mechanism["observation"]["native"]["gm_to_l1_bytes"])
                observed_output_bytes = float(
                    mechanism["observation"]["l2_interface_write_bytes"])
                if (predicted_read_bytes <= 0 or observed_read_bytes <= 0
                        or predicted_output_bytes <= 0
                        or observed_output_bytes <= 0):
                    raise ValueError(
                        "Ascend GroupedMatmul interface bytes must be positive")
                native = mechanism["observation"]["native"]
                if native["ub_to_gm_kb"] != "NA":
                    raise ValueError(
                        "Ascend GroupedMatmul UB-to-GM must remain unavailable")
                rows.append({
                    "hardware_id": document["hardware_id"],
                    "family": task["family"],
                    "case_id": task["case_id"],
                    "kernel_ordinal": mechanism["kernel_ordinal"],
                    "variant": variant,
                    "actual_ns": mechanism["timing_target"]["duration_ns"],
                    "timing_source": mechanism["timing_target"]["source"],
                    **canonical_capture_comparison_evidence(
                        document["hardware_id"]),
                    "tile_m": tile_m,
                    "tile_n": tile_n,
                    "tile_k": tile_k,
                    "predicted_mte2_l2_read_bytes": predicted_read_bytes,
                    "observed_gm_to_l1_read_bytes": observed_read_bytes,
                    "read_predicted_over_observed": (
                        predicted_read_bytes / observed_read_bytes),
                    "read_absolute_error_pct": (
                        abs(predicted_read_bytes - observed_read_bytes)
                        / observed_read_bytes * 100.0),
                    "predicted_aic_output_l2_write_bytes": (
                        predicted_output_bytes),
                    "observed_aic_main_memory_write_bytes": (
                        observed_output_bytes),
                    "output_predicted_over_observed": (
                        predicted_output_bytes / observed_output_bytes),
                    "output_absolute_error_pct": (
                        abs(predicted_output_bytes - observed_output_bytes)
                        / observed_output_bytes * 100.0),
                    "modeled_dram_read_bytes": dram_read_nodes["dram"],
                    "observed_dram_write_bytes": float(
                        mechanism["observation"]["dram_write_bytes"]),
                    "dram_residency_status": "evidence_pending",
                })
    rows.sort(key=lambda row: (
        row["actual_ns"] is None,
        -float(row["actual_ns"] or 0.0),
        row["case_id"],
        int(row["kernel_ordinal"]),
    ))
    return rows


def c500_silu_mechanism_rows(
        documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Keep C500 SiLU demand checks separate from replay timing counters."""

    rows = []
    for document in documents:
        if document["hardware_id"] != "metax_c500":
            continue
        for task in document["tasks"]:
            if task["family"] != "Vector":
                continue
            for kernel in task["kernels"]:
                mechanism = kernel["mechanisms"]
                if mechanism["variant"] != "silu":
                    continue
                if mechanism["kernel_kind"] != "VECTOR":
                    raise ValueError("C500 SiLU mechanism row must be VECTOR")
                analytical = mechanism["analytical_input"]
                observation = mechanism["observation"]
                native = observation["native"]
                if native["backend"] != "metax_mcprofiler":
                    raise ValueError(
                        "C500 SiLU mechanism row requires mcProfiler counters")
                timing_target = mechanism["timing_target"]
                if "availability" not in timing_target:
                    timing_availability = "observed"
                else:
                    timing_availability = timing_target["availability"]
                if timing_availability == "observed":
                    result_scope = "real_kernel"
                    production_duration_ns = float(
                        timing_target["duration_ns"])
                    predicted_duration_ns = float(
                        mechanism["forward"]["total_ns"])
                    timing_source = timing_target["source"]
                elif timing_availability == "unobserved":
                    if (timing_target["duration_ns"] is not None
                            or timing_target["source"] is not None
                            or timing_target["unavailable_reason"]
                            != "canonical_target_is_op_level"):
                        raise ValueError(
                            "C500 SiLU unobserved kernel target is malformed")
                    if task["metadata"]["actual_granularity"] != "op":
                        raise ValueError(
                            "C500 SiLU aggregate timing must be OP-level")
                    result_scope = "aggregate_task"
                    predicted_duration_ns, production_duration_ns = (
                        float(value) for value in task["forward_pair"])
                    timing_source = task["metadata"]["actual_source"]
                else:
                    raise ValueError(
                        "unknown C500 SiLU timing availability: "
                        f"{timing_availability!r}")
                if (timing_source != "device_event_op_median"
                        or native["duration_source"] != "mcTracer"):
                    raise ValueError(
                        "C500 SiLU canonical duration must come from the "
                        "device-event target while replay counters retain "
                        "their mcTracer duration provenance")
                shape = mechanism["shape"]
                tokens = int(shape["tok"])
                intermediate = int(shape["intermediate"])
                predicted_instructions = float(
                    analytical["demand_per_resource"]["warp_issue"])
                observed_instructions = float(native["total_instructions"])
                predicted_global_bytes = float(
                    analytical["demand_per_resource"]["dram"])
                observed_global_bytes = float(
                    native["global_memory_access_bytes"])
                instrumented_duration_ns = float(
                    native["instrumented_during_ns"])
                if native["core_clock_ghz"] is None:
                    raise ValueError(
                        "C500 SiLU mechanism row requires mcProfiler "
                        "core_clk")
                instrumented_core_clock_ghz = float(
                    native["core_clock_ghz"])
                instrumented_active_cycles = float(
                    observation["active_cycles"])
                instrumented_cycles = (
                    instrumented_duration_ns
                    * instrumented_core_clock_ghz)
                positive_values = {
                    "production_duration_ns": production_duration_ns,
                    "predicted_duration_ns": predicted_duration_ns,
                    "observed_total_instructions": observed_instructions,
                    "predicted_global_bytes": predicted_global_bytes,
                    "observed_global_bytes": observed_global_bytes,
                    "instrumented_duration_ns": instrumented_duration_ns,
                    "instrumented_core_clock_ghz": (
                        instrumented_core_clock_ghz),
                    "instrumented_ap_active_cycles": instrumented_active_cycles,
                    "instrumented_cycles": instrumented_cycles,
                    "waves": float(native["waves"]),
                    "instrumented_average_wave_life_cycles": float(
                        native["average_wave_life_cycles"]),
                }
                invalid = {
                    name: value for name, value in positive_values.items()
                    if not math.isfinite(value) or value <= 0
                }
                if invalid:
                    raise ValueError(
                        "C500 SiLU mechanism row has invalid positive values: "
                        f"{invalid}")
                if (not math.isfinite(predicted_instructions)
                        or predicted_instructions < 0):
                    raise ValueError(
                        "C500 SiLU predicted instruction lower bound must be "
                        "finite and non-negative")
                if predicted_instructions > 0:
                    instruction_ratio = (
                        predicted_instructions / observed_instructions)
                    instruction_status = "comparable"
                    instruction_reason = ""
                else:
                    instruction_ratio = None
                    instruction_status = "unavailable"
                    instruction_reason = (
                        "Forward publishes no static instruction lower bound "
                        "for this C500 SiLU source revision")
                non_negative_values = {
                    "instrumented_dnoc_read_latency_cycles": float(
                        native["dnoc_read_average_latency_cycles"]),
                    "instrumented_vls_pipeline_stall_cycles": float(
                        native["vls_pipeline_stall_cycles"]),
                    "instrumented_vls_wdata_stall_cycles": float(
                        native["vls_wdata_stall_cycles"]),
                }
                invalid_non_negative = {
                    name: value
                    for name, value in non_negative_values.items()
                    if not math.isfinite(value) or value < 0
                }
                if invalid_non_negative:
                    raise ValueError(
                        "C500 SiLU mechanism row has invalid non-negative "
                        f"values: {invalid_non_negative}")
                instrumented_mean_active_aps = (
                    instrumented_active_cycles / instrumented_cycles)
                active_units = int(analytical["active_units"])
                if instrumented_mean_active_aps > active_units * 1.01:
                    raise ValueError(
                        "C500 SiLU replay active-AP count exceeds the launch "
                        "capacity")
                rows.append({
                    "result_scope": result_scope,
                    "case_id": task["case_id"],
                    "tokens": tokens,
                    "intermediate": intermediate,
                    "production_duration_ns": production_duration_ns,
                    "timing_source": timing_source,
                    "predicted_duration_ns": predicted_duration_ns,
                    "predicted_over_production": (
                        predicted_duration_ns / production_duration_ns),
                    "absolute_error_pct": (
                        abs(predicted_duration_ns - production_duration_ns)
                        / production_duration_ns * 100.0),
                    "grid_size": int(analytical["grid_size"]),
                    "waves": positive_values["waves"],
                    "predicted_instruction_lower_bound": (
                        predicted_instructions),
                    "observed_total_instructions": observed_instructions,
                    "instruction_predicted_over_observed": instruction_ratio,
                    "instruction_comparison_status": instruction_status,
                    "instruction_comparison_reason": instruction_reason,
                    "predicted_global_bytes": predicted_global_bytes,
                    "observed_global_bytes": observed_global_bytes,
                    "traffic_predicted_over_observed": (
                        predicted_global_bytes / observed_global_bytes),
                    "instrumented_duration_ns": instrumented_duration_ns,
                    "instrumented_core_clock_ghz": (
                        instrumented_core_clock_ghz),
                    "instrumented_ap_active_cycles": (
                        instrumented_active_cycles),
                    "instrumented_mean_active_aps": (
                        instrumented_mean_active_aps),
                    "instrumented_instruction_per_active_cycle": (
                        observed_instructions / instrumented_active_cycles),
                    "instrumented_average_wave_life_cycles": float(
                        positive_values[
                            "instrumented_average_wave_life_cycles"]),
                    "instrumented_dnoc_read_latency_cycles": float(
                        non_negative_values[
                            "instrumented_dnoc_read_latency_cycles"]),
                    "instrumented_vls_pipeline_stall_cycles": float(
                        non_negative_values[
                            "instrumented_vls_pipeline_stall_cycles"]),
                    "instrumented_vls_wdata_stall_cycles": float(
                        non_negative_values[
                            "instrumented_vls_wdata_stall_cycles"]),
                    "timing_scope_note": (
                        "canonical duration and Forward prediction use the "
                        f"{result_scope} boundary; AP/wave/DNoC/stall counters "
                        "and their mcTracer duration are a separate mcProfiler "
                        "replay and are never divided by the canonical "
                        "duration"),
                })
    rows.sort(key=lambda row: (
        -float(row["production_duration_ns"]), row["case_id"]))
    return rows


def c500_silu_mechanism_summary(
        rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Summarize only directly comparable demand and same-replay evidence."""

    if not rows:
        raise ValueError("C500 SiLU mechanism summary requires rows")
    duration_apes = np.asarray([
        float(row["absolute_error_pct"]) for row in rows])
    instruction_rows = [
        row for row in rows
        if row["instruction_comparison_status"] == "comparable"
    ]
    instruction_apes = np.asarray([
        abs(float(row["instruction_predicted_over_observed"]) - 1.0) * 100.0
        for row in instruction_rows
    ])
    traffic_apes = np.asarray([
        abs(float(row["traffic_predicted_over_observed"]) - 1.0) * 100.0
        for row in rows
    ])
    largest_absolute_delta = max(
        rows,
        key=lambda row: abs(
            float(row["predicted_duration_ns"])
            - float(row["production_duration_ns"])),
    )
    return {
        "case_count": len(rows),
        "underpredicted_count": sum(
            float(row["predicted_duration_ns"])
            < float(row["production_duration_ns"])
            for row in rows),
        "case_gate_passed_count": int(np.sum(duration_apes <= CASE_APE_LIMIT_PCT)),
        "duration_mape_pct": float(np.mean(duration_apes)),
        "duration_max_ape_pct": float(np.max(duration_apes)),
        "instruction_comparison_status": (
            "comparable" if instruction_rows else "unavailable"),
        "instruction_comparable_count": len(instruction_rows),
        "instruction_exact_count": sum(
            float(row["predicted_instruction_lower_bound"])
            == float(row["observed_total_instructions"])
            for row in instruction_rows),
        "instruction_mape_pct": (
            float(np.mean(instruction_apes)) if instruction_rows else None),
        "instruction_max_ape_pct": (
            float(np.max(instruction_apes)) if instruction_rows else None),
        "traffic_mape_pct": float(np.mean(traffic_apes)),
        "traffic_max_ape_pct": float(np.max(traffic_apes)),
        "replay_mean_active_aps_min": min(
            float(row["instrumented_mean_active_aps"]) for row in rows),
        "replay_mean_active_aps_max": max(
            float(row["instrumented_mean_active_aps"]) for row in rows),
        "replay_core_clock_ghz_min": min(
            float(row["instrumented_core_clock_ghz"]) for row in rows),
        "replay_core_clock_ghz_max": max(
            float(row["instrumented_core_clock_ghz"]) for row in rows),
        "replay_wave_life_cycles_min": min(
            float(row["instrumented_average_wave_life_cycles"])
            for row in rows),
        "replay_wave_life_cycles_max": max(
            float(row["instrumented_average_wave_life_cycles"])
            for row in rows),
        "replay_dnoc_latency_cycles_min": min(
            float(row["instrumented_dnoc_read_latency_cycles"])
            for row in rows),
        "replay_dnoc_latency_cycles_max": max(
            float(row["instrumented_dnoc_read_latency_cycles"])
            for row in rows),
        "largest_absolute_delta_case_id": largest_absolute_delta["case_id"],
        "largest_absolute_delta_production_ns": float(
            largest_absolute_delta["production_duration_ns"]),
        "largest_absolute_delta_predicted_ns": float(
            largest_absolute_delta["predicted_duration_ns"]),
        "largest_absolute_delta_ape_pct": float(
            largest_absolute_delta["absolute_error_pct"]),
    }
