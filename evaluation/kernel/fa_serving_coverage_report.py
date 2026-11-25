"""Forward-only prediction report for the published FA serving capture set."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

import zstandard

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import (
    KernelKind,
    KernelSpec,
    PhysicalTransferConfig,
)
from src.kernel.launch_config.select_launch_config import (
    select_flash_attn_copy_launches,
    select_launch_config,
)
from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS


_HARDWARE_IDS = EVALUATED_HARDWARE_IDS
_TIMING_SOURCES = {
    "nvidia_a800": "nsys_nvtx_kernel_median",
    "metax_c500": "device_event_op_median",
    "ascend_910b": "msprof_timeline_op_median",
}


def _load_release(path: Path, hardware_id: str) -> dict:
    with path.open("rb") as stream:
        payload = zstandard.ZstdDecompressor().decompress(stream.read())
    value = json.loads(payload)
    if value["schema_version"] != 1 or value["kind"] != "fa_serving_capture_release":
        raise ValueError(f"invalid FA serving release schema: {path}")
    if value["hardware_id"] != hardware_id or value["case_count"] != 40:
        raise ValueError(f"invalid FA serving release hardware/count: {path}")
    if len(value["records"]) != 40:
        raise ValueError(f"invalid FA serving release records: {path}")
    return value


def validate_fa_serving_capture_record(record: dict, hardware_id: str) -> float:
    """Validate one canonical serving-aligned FA record and return its target."""

    if hardware_id not in _HARDWARE_IDS:
        raise ValueError(f"unsupported hardware: {hardware_id!r}")
    if not isinstance(record, dict):
        raise TypeError("FA serving capture record must be a mapping")
    if set(record) != {"capture_evidence", "op_info"}:
        raise ValueError(
            "FA serving capture record fields must be exactly "
            "capture_evidence and op_info")
    op_info = record["op_info"]
    if not isinstance(op_info, dict) or set(op_info) != {
            "Attr", "Type", "evaluation"}:
        raise ValueError("FA serving capture op_info fields are invalid")
    if op_info["Type"] != "FLASHATTN":
        raise ValueError("FA serving capture record must have Type=FLASHATTN")
    timing = op_info["evaluation"]["timing"]
    if not isinstance(timing, dict) or set(timing) != {
            "granularity", "source", "total_duration_ns"}:
        raise ValueError("FA serving capture timing fields are invalid")
    if timing["granularity"] != "op":
        raise ValueError("FA serving capture timing must be op-level")
    if timing["source"] != _TIMING_SOURCES[hardware_id]:
        raise ValueError(
            f"FA serving capture timing source for {hardware_id} must be "
            f"{_TIMING_SOURCES[hardware_id]!r}")
    target = timing["total_duration_ns"]
    if (isinstance(target, bool) or not isinstance(target, (int, float))
            or not math.isfinite(target) or target <= 0):
        raise ValueError("FA serving capture target must be finite and positive")
    evidence = record["capture_evidence"]
    if not isinstance(evidence, dict):
        raise TypeError("FA serving capture evidence must be a mapping")
    required_evidence = {
        "logical_identity_sha256", "workload_digest", "phase", "timing",
        "counter",
    }
    if not required_evidence <= set(evidence):
        raise ValueError("FA serving capture evidence is incomplete")
    for name in ("logical_identity_sha256", "workload_digest"):
        value = evidence[name]
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"FA serving capture {name} must be a SHA256")
    counter = evidence["counter"]
    if not isinstance(counter, dict):
        raise TypeError("FA serving capture counter evidence must be a mapping")
    status = counter["status"]
    if status == "available":
        required_counter = {
            "status", "candidate", "candidate_path", "candidate_sha256"}
    elif status == "unavailable":
        required_counter = {
            "status", "reason", "candidate_path", "candidate_sha256"}
        if counter["candidate_path"] is not None or counter["candidate_sha256"] is not None:
            raise ValueError(
                "unavailable FA serving counter must not name a candidate")
    else:
        raise ValueError(f"invalid FA serving counter status: {status!r}")
    if set(counter) != required_counter:
        raise ValueError("FA serving capture counter fields are invalid")
    return float(target)


def build_fa_serving_specs(
        attributes: dict, hardware_id: str,
        hardware: HWSpec) -> tuple[KernelSpec, ...]:
    variant = "decode" if attributes["Q_LEN"] == 1 else "prefill"
    shape = {
        "batch": attributes["B"],
        "q_len": attributes["Q_LEN"],
        "kv_len": attributes["KV_LEN"],
        "head_dim": attributes["D"],
        "num_query_heads": attributes["HQ"],
        "num_kv_heads": attributes["HKV"],
    }
    selection_attributes = {
        "is_decode": variant == "decode",
        "causal": attributes["causal"],
        "kv_layout": attributes["kv_layout"],
        "kv_dist": attributes["kv_dist"],
        "q_lens": attributes["Q_LENS"],
        "kv_lens": attributes["KV_LENS"],
        "split_policy": attributes["split_policy"],
        "launch_policy": "analytical_launch",
    }
    if hardware_id == "metax_c500":
        # The Event timer uses a Graph replay envelope, while the captured FA
        # call itself is the source's ordinary one-split dispatch.  FULL would
        # incorrectly select the graph-token multi-split path.
        selection_attributes["launch_policy"] = "injected_launch"
        selection_attributes["graph_runtime_mode"] = "NONE"
    launch = select_launch_config(
        hardware_id, KernelKind.FLASH_ATTN, shape, attributes["dtype"],
        variant, selection_attributes, hardware)
    production_attributes = {
        "causal": attributes["causal"],
        "kv_layout": attributes["kv_layout"],
        "kv_dist": attributes["kv_dist"],
        "q_lens": tuple(attributes["Q_LENS"]),
        "kv_lens": tuple(attributes["KV_LENS"]),
        "fa_ksteps": None,
        "fa_eff_mfrac": 1.0,
    }
    main = KernelSpec(
        kind=KernelKind.FLASH_ATTN,
        hardware_id=hardware_id,
        shape=shape,
        dtype=attributes["dtype"],
        variant=launch.execution_variant,
        launch=launch,
        attributes=production_attributes,
    )
    specs = [main]
    tiling = launch.tiling
    if tiling is None or tiling.num_kv_splits is None:
        raise ValueError("FLASH_ATTN launch lacks a split contract")
    if hardware_id == "nvidia_a800" and tiling.num_kv_splits > 1:
        combine_attributes = dict(selection_attributes)
        combine_attributes["num_kv_splits"] = tiling.num_kv_splits
        combine_launch = select_launch_config(
            hardware_id, KernelKind.FLASH_ATTN, shape, attributes["dtype"],
            "combine", combine_attributes, hardware)
        specs.append(KernelSpec(
            kind=KernelKind.FLASH_ATTN,
            hardware_id=hardware_id,
            shape=shape,
            dtype=attributes["dtype"],
            variant=combine_launch.execution_variant,
            launch=combine_launch,
            attributes=production_attributes,
        ))
    for _, dtype, elements, copy_launch in select_flash_attn_copy_launches(
            hardware_id, shape, selection_attributes, hardware):
        if copy_launch.algorithm == "native_d2d_flash_attention_copy":
            element_bytes = {"bf16": 2, "float32": 4}
            if dtype not in element_bytes:
                raise ValueError(f"unsupported FA copy dtype: {dtype!r}")
            specs.append(KernelSpec(
                kind=KernelKind.MEMCPY,
                hardware_id=hardware_id,
                transfer=PhysicalTransferConfig(
                    algorithm="dma", bytes=elements * element_bytes[dtype],
                    direction="d2d", link="dram"),
            ))
        else:
            specs.append(KernelSpec(
                kind=KernelKind.VECTOR,
                hardware_id=hardware_id,
                shape={"element_count": elements},
                dtype=dtype,
                variant="elementwise",
                launch=copy_launch,
                attributes={"reads": 1, "writes": 1, "operation": "copy"},
            ))
    return tuple(specs)


def build_report(release_paths: dict[str, Path]) -> dict:
    """Predict every published case through the production kernel API."""
    if set(release_paths) != set(_HARDWARE_IDS):
        raise ValueError(f"release paths must cover exactly {_HARDWARE_IDS}")
    results = {}
    for hardware_id in _HARDWARE_IDS:
        release = _load_release(release_paths[hardware_id], hardware_id)
        hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
        rows = []
        for case_id, record in sorted(release["records"].items()):
            attributes = record["op_info"]["Attr"]
            target = record["op_info"]["evaluation"]["timing"]["total_duration_ns"]
            if not isinstance(target, (int, float)) or target <= 0:
                raise ValueError(f"invalid timing target: {hardware_id}/{case_id}")
            validate_fa_serving_capture_record(record, hardware_id)
            kernel_specs = build_fa_serving_specs(
                attributes, hardware_id, hardware)
            predictions = [
                predict(kernel_spec, hardware, method="midlayer")
                for kernel_spec in kernel_specs
            ]
            predicted = math.fsum(prediction.t_total_ns for prediction in predictions)
            if not math.isfinite(predicted) or predicted <= 0:
                raise ValueError(f"invalid Forward prediction: {hardware_id}/{case_id}")
            absolute_error = abs(predicted - target)
            rows.append({
                "case_id": case_id,
                "phase": record["capture_evidence"]["phase"],
                "counter_status": record["capture_evidence"]["counter"]["status"],
                "actual_ns": target,
                "predicted_ns": predicted,
                "absolute_error_ns": absolute_error,
                "absolute_percentage_error": absolute_error / target * 100.0,
                "launches": [
                    {
                        "kind": kernel_spec.kind.value,
                        "algorithm": (
                            kernel_spec.launch.algorithm
                            if kernel_spec.launch is not None else "dma"),
                        "grid": (
                            list(kernel_spec.launch.grid)
                            if kernel_spec.launch is not None else None),
                        "block": (
                            list(kernel_spec.launch.block)
                            if kernel_spec.launch is not None else None),
                        "execution_variant": kernel_spec.variant,
                    }
                    for kernel_spec in kernel_specs
                ],
                "support_levels": [
                    (prediction.diagnostics.support_level
                     if prediction.diagnostics is not None else None)
                    for prediction in predictions],
                "limitations": [
                    (prediction.diagnostics.limitations
                     if prediction.diagnostics is not None else None)
                    for prediction in predictions],
            })
        errors = [row["absolute_percentage_error"] for row in rows]
        absolute_errors = [row["absolute_error_ns"] for row in rows]
        results[hardware_id] = {
            "case_count": len(rows),
            "counter_available_case_count": sum(
                row["counter_status"] == "available" for row in rows),
            "counter_unavailable_case_count": sum(
                row["counter_status"] == "unavailable" for row in rows),
            "mae_ns": statistics.fmean(absolute_errors),
            "mape_pct": statistics.fmean(errors),
            "median_ape_pct": statistics.median(errors),
            "max_ape_pct": max(errors),
            "rows": rows,
        }
    return {"schema_version": 1, "kind": "fa_serving_forward_report", "hardware": results}


def write_report(report: dict, output_root: Path) -> dict[str, Path]:
    """Write the machine-readable and Markdown Forward report once."""
    if output_root.exists():
        raise FileExistsError(f"report output already exists: {output_root}")
    output_root.mkdir(parents=True)
    json_path = output_root / "kernel_report.json"
    markdown_path = output_root / "kernel_report.md"
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with markdown_path.open("x", encoding="utf-8") as stream:
        stream.write("# FA serving Forward kernel report\n\n")
        stream.write("All predictions call `src.kernel.kernel_base.predict` with the production `midlayer` method. No profiler duration or counter is an input to Forward.\n\n")
        stream.write("| Hardware | Cases | Counter available | Counter unavailable | MAE (µs) | MAPE | Median APE | Max APE |\n")
        stream.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for hardware_id in _HARDWARE_IDS:
            summary = report["hardware"][hardware_id]
            stream.write(
                f"| {hardware_id} | {summary['case_count']} | "
                f"{summary['counter_available_case_count']} | "
                f"{summary['counter_unavailable_case_count']} | "
                f"{summary['mae_ns'] / 1000.0:.3f} | "
                f"{summary['mape_pct']:.2f}% | {summary['median_ape_pct']:.2f}% | "
                f"{summary['max_ape_pct']:.2f}% |\n")
        stream.write("\nC500 B8/KV8192 is included in its timing accuracy metric; its per-invocation counter is explicitly unavailable and was not replaced by an aggregate report.\n")
        for hardware_id in _HARDWARE_IDS:
            stream.write(f"\n## {hardware_id} cases\n\n")
            stream.write("| Case | Phase | Counter | Golden (µs) | Predicted (µs) | Absolute error (µs) | APE |\n")
            stream.write("| --- | --- | --- | ---: | ---: | ---: | ---: |\n")
            for row in report["hardware"][hardware_id]["rows"]:
                stream.write(
                    f"| {row['case_id']} | {row['phase']} | "
                    f"{row['counter_status']} | "
                    f"{row['actual_ns'] / 1000.0:.3f} | "
                    f"{row['predicted_ns'] / 1000.0:.3f} | "
                    f"{row['absolute_error_ns'] / 1000.0:.3f} | "
                    f"{row['absolute_percentage_error']:.2f}% |\n")
    return {"json": json_path, "markdown": markdown_path}
