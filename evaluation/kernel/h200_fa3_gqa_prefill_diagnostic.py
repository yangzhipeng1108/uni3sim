"""Compare the partial vLLM 0.30 FA3 GQA candidate with six pinned cases."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
from pathlib import Path
import statistics
import time

from evaluation.kernel.dataset import get_hardware_config_path, load_kernel_dataset
from evaluation.kernel.evaluation_runner import _get_hardware
from evaluation.kernel.observation_normalize_flash_attention import _normalize_flash_attention
from evaluation.kernel.observation_parser import observed_launch_config
from evaluation.kernel.profiler_observation import extract_ncu_metrics
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.predictors.flash_attn import (
    _fa3_gqa_prefill_030_tile_blocks,
    _fa3_list_schedule,
)
from src.kernel.launch_config.nvidia_h200.flash_attn_launch import (
    FA3_GQA_PREFILL_030_SOURCE_REVISION,
    select_v030_gqa_fresh_prefill_launch,
)


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "profiler_output/kernel_dataset/nvidia_h200/ncu_1027_20261031"
OUTPUT = ROOT / "experiment_logs/h200_accuracy_20261028/fa3_gqa_prefill_partial_6case.json"
BATCHES = ("ncu_flash_attn_002", "ncu_flash_attn_004")
SOURCE_AUDIT = ROOT / "experiment_logs/h200_accuracy_20261028/fa3_gqa_prefill_source_audit.json"
BAR_PROBE = (ROOT / "profiler_output/microbench/nvidia_h200/"
             "fa3_wg_scheduler_barrier_20261102/attempt_02")


def _pinned_file(review: dict, path: Path) -> None:
    relative = str(path.relative_to(CAMPAIGN))
    pins = [item for item in review["raw_files"] if item["path"] == relative]
    if len(pins) != 1:
        raise ValueError(f"NCU review does not pin {relative}")
    pin = pins[0]
    if (path.stat().st_size != pin["size"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]):
        raise ValueError(f"NCU raw changed after review: {relative}")


def _source_bound_case(case_id: str, batch: str, review: dict, entry: dict,
                       hardware) -> tuple[KernelSpec, KernelSpec] | None:
    """Bind only a reviewed v0.30 raw case; canonical lacks source identity."""

    batch_path = CAMPAIGN / "batches" / batch
    identity_path = batch_path / "identity.json"
    _pinned_file(review, identity_path)
    identity = json.loads(identity_path.read_text())
    if (identity["data"]["versions"]["vllm"] != "0.30.0+cu129"
            or identity["plan_sha256"] != review["plan_sha256"]):
        raise ValueError(f"NCU raw is outside vLLM 0.30 source identity: {batch}")
    audit = json.loads(SOURCE_AUDIT.read_text())
    if audit["source_commit"] != FA3_GQA_PREFILL_030_SOURCE_REVISION:
        raise ValueError("FA3 source audit revision changed")
    if review["statuses"][case_id]["status"] != "complete":
        raise ValueError(f"NCU case lacks accepted validation: {case_id}")
    case_path = batch_path / "ncu_raw" / case_id
    validation_path = case_path / "validation.json"
    _pinned_file(review, validation_path)
    if json.loads(validation_path.read_text())["status"] != "complete":
        raise ValueError(f"NCU validation differs from review: {case_id}")
    preflight_path = (case_path / "preflight/cases" / case_id
                      / "window_01.launch.json")
    _pinned_file(review, preflight_path)
    preflight = json.loads(preflight_path.read_text())
    if (preflight["case_id"] != case_id
            or preflight["runner_relpath"] !=
            "scripts/kernel_dataset/nvidia_h200/flash_attn_runner.py"):
        raise ValueError(f"preflight source runner differs: {case_id}")
    kernels = entry["kernels"]
    if len(kernels) != len(preflight["launches"]):
        raise ValueError(f"canonical/raw kernel count differs: {case_id}")
    for canonical, raw in zip(kernels, preflight["launches"]):
        metrics = extract_ncu_metrics(canonical)
        if (metrics["kernel_name"] != raw["name"]
                or [metrics["grid_x"], metrics["grid_y"], metrics["grid_z"]]
                != raw["grid"]
                or [metrics["block_x"], metrics["block_y"], metrics["block_z"]]
                != raw["block"]
                or metrics["dynamic_smem_per_block"] != raw["dynamic_shared_memory_bytes"]
                or metrics["static_smem_per_block"] != raw["static_shared_memory_bytes"]
                or metrics["regs_per_thread"] != raw["registers_per_thread"]):
            raise ValueError(f"canonical/raw launch differs: {case_id}")
    main_metrics = extract_ncu_metrics(kernels[0])
    main_smem = (main_metrics["dynamic_smem_per_block"]
                 + main_metrics["static_smem_per_block"])
    if (main_smem > hardware.max_smem_per_block_bytes
            or main_smem * 2 <= hardware.max_smem_per_unit_bytes):
        raise ValueError(f"FA3 main is not a one-CTA-per-SM resource launch: {case_id}")
    attributes = entry["op_info"]["Attr"]
    if attributes["SP"] != 1:
        return None
    main = _normalize_flash_attention(
        attributes, kernels[0], "nvidia_h200", hardware)
    selected = select_v030_gqa_fresh_prefill_launch(
        main.shape, main.dtype,
        {**main.attributes, "num_kv_splits": attributes["SP"]}, hardware)
    if (main.launch.grid != selected.grid or main.launch.block != selected.block
            or "FlashAttnFwdSm90" not in preflight["launches"][0]["name"]):
        raise ValueError(f"source main launch differs: {case_id}")
    source_main = dataclasses.replace(main, launch=selected)
    fill_name = preflight["launches"][1]["name"]
    if not fill_name.startswith(
            "void at::native::vectorized_elementwise_kernel<(int)4, "
            "at::native::FillFunctor<int>, "):
        raise ValueError(f"source semaphore Fill symbol differs: {case_id}")
    fill_shape = {"element_count": 1, "dtype_bytes": 4}
    fill_metrics = extract_ncu_metrics(kernels[1])
    fill_launch = observed_launch_config(
        fill_metrics, KernelKind.VECTOR, fill_shape,
        algorithm="vllm_030_fa3_scheduler_semaphore_fill",
        variant="elementwise")
    if fill_launch.grid != (1, 1, 1) or fill_launch.block != (128, 1, 1):
        raise ValueError(f"source semaphore Fill geometry differs: {case_id}")
    fill = KernelSpec(
        kind=KernelKind.VECTOR, hardware_id="nvidia_h200",
        shape=fill_shape, dtype="int32", variant="elementwise",
        launch=fill_launch,
        attributes={"reads": 0, "writes": 1, "operation": "fill"},
    )
    return source_main, fill


def _main_ncu_row(case_id: str, batch: str, review: dict) -> dict:
    case = CAMPAIGN / "batches" / batch / "ncu_raw" / case_id
    csv_path = case / "ncu_stdout.csv"
    _pinned_file(review, csv_path)
    lines = csv_path.read_text().splitlines()
    header = next(index for index, line in enumerate(lines)
                  if line.startswith('"ID"'))
    rows = list(csv.DictReader(lines[header:]))
    matches = [row for row in rows if "FlashAttnFwdSm90" in row["Kernel Name"]]
    if len(matches) != 1:
        raise ValueError(f"NCU case lacks one FA3 main kernel: {case_id}")
    row = matches[0]
    return {
        "l2_read_bytes": int(row["lts__t_sectors_op_read.sum"]) * 32,
        "dram_read_bytes": int(row["dram__bytes_read.sum"]),
        "tensor_pipe_active_pct": float(
            row["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"]),
        "sm_active_cycles_avg": float(row["sm__cycles_active.avg"]),
        "sm_elapsed_cycles_avg": float(row["sm__cycles_elapsed.avg"]),
    }


def _source_task_balance(main: KernelSpec) -> dict:
    """Screen tile-count imbalance without assigning a time price to it."""

    blocks = _fa3_gqa_prefill_030_tile_blocks(main)
    workers = min(len(blocks), main.launch.grid[0])
    assigned = _fa3_list_schedule(blocks, workers)
    total = sum(blocks)
    if sum(load for load, _ in assigned) != total:
        raise ValueError("FA3 source tile assignment lost KV blocks")
    mean = total / workers
    return {
        "task_count": len(blocks),
        "active_cta_count": workers,
        "kv_tile_visits": total,
        "mean_visits_per_active_cta": mean,
        "max_visits_per_cta_in_proportional_service_schedule": max(
            load for load, _ in assigned),
        "max_to_mean_visit_ratio": max(load for load, _ in assigned) / mean,
        "scope": "source task order; equal per-KV-block service and earliest-free CTA assignment only",
    }


def _barrier_probe_decision(main: KernelSpec, main_gap_ns: float) -> dict:
    """Test the fixed barrier instruction fee against source KV-block count.

    This is an evaluation diagnostic, not a Forward service parameter: the
    register-only probe cannot reproduce WGMMA/TMA/softmax overlap.
    """

    receipt_path = BAR_PROBE / "receipt.json"
    raw_path = BAR_PROBE / "raw.json"
    app_path = BAR_PROBE / "app_info.json"
    receipt = json.loads(receipt_path.read_text())
    if (raw_path.stat().st_size != receipt["raw_size"]
            or hashlib.sha256(raw_path.read_bytes()).hexdigest()
            != receipt["raw_sha256"]
            or app_path.stat().st_size != receipt["app_info_size"]
            or hashlib.sha256(app_path.read_bytes()).hexdigest()
            != receipt["app_info_sha256"]):
        raise ValueError("FA3 barrier probe differs from pinned receipt")
    raw = json.loads(raw_path.read_text())
    app = json.loads(app_path.read_text())
    if (raw["sass_boundary_valid"] is not True
            or raw["execution"]["exit_code"] != 0
            or app["app_id"] != receipt["app_id"]
            or app["lifecycle"]["state"] != "stopped"
            or app["functions"]["capture"]["summary"]["gpu_config"]
            != [{"count": 1, "gpu_type": "H200"}]):
        raise ValueError("FA3 barrier probe execution or app state invalid")
    readings = {}
    for line in raw["execution"]["stdout"].splitlines():
        values = tuple(int(value) for value in line.split(","))
        if len(values) != 8:
            raise ValueError("FA3 barrier probe reading width differs")
        grid, mode, hold, rounds, cta, wg, cycles, _ = values
        key = (grid, mode, hold, rounds, cta, wg)
        if key in readings or cycles < 0:
            raise ValueError("FA3 barrier probe readings duplicated or invalid")
        readings[key] = cycles
    if len(readings) != receipt["execution_line_count"]:
        raise ValueError("FA3 barrier probe reading count differs")
    deltas = {}
    for rounds in (1, 8, 64):
        deltas[rounds] = statistics.median(
            readings[(132, 1, 0, rounds, cta, 0)]
            - readings[(132, 0, 0, rounds, cta, 0)]
            for cta in range(132))
    steady_cycles = (deltas[64] - deltas[8]) / (64 - 8)
    blocks = _fa3_gqa_prefill_030_tile_blocks(main)
    per_task_rounds = [max(block - 1, 0) for block in blocks]
    if len(blocks) != 184 or sum(blocks) != 624:
        raise ValueError("FA3 Q729 source tile census changed")
    # This is the candidate's list schedule, not a measured CTA timeline.
    scheduled = _fa3_list_schedule(per_task_rounds, main.launch.grid[0])
    proxy_critical_rounds = max(load for load, _ in scheduled)
    reference_clock_mhz = 1980  # Existing H200 NSYS/NCU quality target.
    proxy_ns = proxy_critical_rounds * steady_cycles / reference_clock_mhz * 1000
    return {
        "probe_raw_sha256": receipt["raw_sha256"],
        "probe_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "grid132_hold0_wg0_median_mode1_minus_mode0_cycles": deltas,
        "steady_cycles_per_round_from_8_to_64": steady_cycles,
        "source_occurrence": "one sync/arrive pair for each KV block after the first within each Q tile",
        "q729_task_count": len(blocks),
        "q729_kv_block_count": sum(blocks),
        "q729_sync_arrive_pairs_total_across_grid": sum(per_task_rounds),
        "q729_candidate_schedule_critical_pairs": proxy_critical_rounds,
        "reference_sm_clock_mhz": reference_clock_mhz,
        "candidate_schedule_fixed_fee_ns": proxy_ns,
        "main_prediction_gap_ns_for_comparison_only": main_gap_ns,
        "decision": (
            "Fixed empty-handshake instruction fee is too small to explain "
            "the Q729 main-kernel gap under the source candidate schedule. "
            "Hold-64/256 probe differences include serialized work between "
            "warpgroups; no barrier fee is added to Forward. Actual QK/PV "
            "WGMMA, softmax and TMA overlap remains unbound."),
    }


def main() -> None:
    dataset = load_kernel_dataset("nvidia_h200", ["flash_attn"])
    hardware = _get_hardware(get_hardware_config_path("nvidia_h200"))
    results = []
    barrier_main = None
    for batch in BATCHES:
        review = json.loads((CAMPAIGN / f"{batch}_review.json").read_text())
        for case_id, status in review["statuses"].items():
            if "_PREFILL_" not in case_id:
                continue
            if status["status"] != "complete":
                raise ValueError(f"review rejects selected case: {case_id}")
            entry = dataset[case_id]
            start = time.perf_counter()
            source_specs = _source_bound_case(
                case_id, batch, review, entry, hardware)
            predictions = (tuple(predict(spec, hardware, "midlayer")
                                 for spec in source_specs)
                           if source_specs is not None else None)
            if "HQ32_HKV4_D128_PREFILL_B1_QSUM729" in case_id:
                if source_specs is None:
                    raise ValueError("selected FA3 barrier case is unsupported")
                barrier_main = source_specs[0]
            simulation_wall_s = time.perf_counter() - start
            if source_specs is not None:
                warm_start = time.perf_counter()
                repeated = tuple(predict(spec, hardware, "midlayer")
                                 for spec in source_specs)
                simulation_wall_warm_s = time.perf_counter() - warm_start
                if any(a.t_total_s != b.t_total_s
                       for a, b in zip(predictions, repeated)):
                    raise ValueError(f"candidate changed between repetitions: {case_id}")
            else:
                simulation_wall_warm_s = None
            timing = entry["op_info"]["evaluation"]["timing"]
            actual_kernels = timing["kernels"]
            if len(actual_kernels) != len(entry["kernels"]):
                raise ValueError(f"canonical timing lacks kernels: {case_id}")
            main_actual_ns = actual_kernels[0]["duration_ns"]
            if predictions is not None:
                main_candidate_ns = predictions[0].t_total_s * 1e9
                fill_candidate_ns = predictions[1].t_total_s * 1e9
                op_candidate_ns = main_candidate_ns + fill_candidate_ns
                main_error_pct = abs(main_candidate_ns - main_actual_ns) / main_actual_ns * 100
                op_error_pct = abs(op_candidate_ns - timing["total_duration_ns"]) / timing["total_duration_ns"] * 100
            else:
                main_candidate_ns = fill_candidate_ns = op_candidate_ns = None
                main_error_pct = op_error_pct = None
            results.append({
                "case_id": case_id,
                "ncu_batch": batch,
                "observed_main_resources": {
                    "registers_per_thread": extract_ncu_metrics(
                        entry["kernels"][0])["regs_per_thread"],
                    "dynamic_smem_per_block_bytes": extract_ncu_metrics(
                        entry["kernels"][0])["dynamic_smem_per_block"],
                    "static_smem_per_block_bytes": extract_ncu_metrics(
                        entry["kernels"][0])["static_smem_per_block"],
                    "resident_cta_slots_per_sm_upper_bound": 1,
                },
                "main_actual_ns": main_actual_ns,
                "main_candidate_ns": main_candidate_ns,
                "main_error_pct": main_error_pct,
                "main_support_level": "partial" if predictions is not None else "unsupported",
                "main_limitations": predictions[0].diagnostics.limitations
                if predictions is not None else (),
                "fill_actual_ns": actual_kernels[1]["duration_ns"]
                if predictions is not None else None,
                "fill_candidate_ns": fill_candidate_ns,
                "op_actual_ns": timing["total_duration_ns"],
                "op_candidate_ns": op_candidate_ns,
                "op_error_pct": op_error_pct,
                "op_forward_status": "candidate_partial" if predictions is not None
                else "unsupported_split_combine",
                "simulation_wall_s": simulation_wall_s,
                "simulation_wall_warm_s": simulation_wall_warm_s,
                "source_task_balance": _source_task_balance(source_specs[0])
                if source_specs is not None else None,
                "ncu": _main_ncu_row(case_id, batch, review),
            })
    if len(results) != 6:
        raise ValueError(f"expected six NCU accepted fresh prefill cases, got {len(results)}")
    if sum(row["main_candidate_ns"] is not None for row in results) != 5:
        raise ValueError("expected five unsplit candidates and one split case")
    if barrier_main is None:
        raise ValueError("FA3 barrier source case missing")
    barrier_row = next(row for row in results
                       if "HQ32_HKV4_D128_PREFILL_B1_QSUM729" in row["case_id"])
    document = {
        "kind": "h200_fa3_gqa_prefill_partial_diagnostic",
        "source_audit": "experiment_logs/h200_accuracy_20261028/fa3_gqa_prefill_source_audit.json",
        "source_audit_sha256": hashlib.sha256(SOURCE_AUDIT.read_bytes()).hexdigest(),
        "ncu_review_sha256": {
            batch: hashlib.sha256((CAMPAIGN / f"{batch}_review.json").read_bytes()).hexdigest()
            for batch in BATCHES},
        "source_identity_limit": (
            "vLLM 0.30 CMake pins FA3 source commit 506341a; pinned Modal "
            "raw records show vLLM 0.30.0+cu129 and matching runtime launch "
            "shape, but the prebuilt FA3 binary has no independently recorded "
            "build SHA tying it to that commit"),
        "baseline_main_status": "unsupported; no same-case baseline Midlayer time",
        "candidate_scope": "five reviewed v0.30 unsplit FA3 main plus one-int32 semaphore Fill; split remains unsupported",
        "barrier_probe_decision": _barrier_probe_decision(
            barrier_main,
            barrier_row["main_actual_ns"] - barrier_row["main_candidate_ns"]),
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
