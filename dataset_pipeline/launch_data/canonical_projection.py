"""Project expanded launch evidence into compact production selector facts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Mapping

from src.kernel.launch_config.gemm_launch_selection import (
    ASCEND_910B_GEMM_PROGRAM_FIELDS,
    GEMM_BOOLEAN_ENCODING,
    GEMM_LAUNCH_ARTIFACT_TYPE,
    GEMM_LAUNCH_CSV_FILENAME,
    GEMM_LAUNCH_COMPRESSION,
    GEMM_LAUNCH_MANIFEST_FILENAME,
    GEMM_NULL_ENCODING,
    compress_gemm_launch_csv,
    gemm_launch_csv_fields,
    gemm_launch_fields,
    gemm_launch_schema_version,
    gemm_launch_hardware_contract_sha256,
    gemm_launch_record_layout,
    load_gemm_launch_hardware_contract,
    load_gemm_launch_selection_files,
)
from src.kernel.launch_config.gemm_workload import (
    GEMM_LAUNCH_SELECTION_KEY_FIELDS,
    GEMM_SEMANTIC_CONTEXT_FIELDS,
    gemm_launch_selection_key_from_workload_key,
)


GEMM_DOMAIN_SHA256 = (
    "8a935c5226fc5b719fac1ab0e8dd78df130b9deeaa0725c9c936e5eeb9d14bee")


def compact_json_text(value: object) -> str:
    """Serialize canonical launch facts with scalar arrays on one line."""

    def render(item: object, level: int) -> list[str]:
        if isinstance(item, Mapping):
            if not item:
                return ["{}"]
            lines = ["{"]
            keys = sorted(item)
            for index, key in enumerate(keys):
                if not isinstance(key, str):
                    raise TypeError("canonical JSON mapping keys must be strings")
                child = render(item[key], level + 1)
                prefix = (
                    "  " * (level + 1)
                    + json.dumps(key, ensure_ascii=True)
                    + ": "
                )
                lines.append(prefix + child[0])
                lines.extend(child[1:])
                if index != len(keys) - 1:
                    lines[-1] += ","
            lines.append("  " * level + "}")
            return lines
        if isinstance(item, list):
            if all(not isinstance(element, (Mapping, list)) for element in item):
                return [json.dumps(item, ensure_ascii=True)]
            if not item:
                return ["[]"]
            lines = ["["]
            for index, element in enumerate(item):
                child = render(element, level + 1)
                lines.append("  " * (level + 1) + child[0])
                lines.extend(child[1:])
                if index != len(item) - 1:
                    lines[-1] += ","
            lines.append("  " * level + "]")
            return lines
        return [json.dumps(item, ensure_ascii=True)]

    return "\n".join(render(value, 0)) + "\n"


def expanded_evidence_sha256(value: Mapping[str, object]) -> str:
    """Hash the exact canonical serialization archived in profiler_output."""

    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(
        "utf-8")
    return hashlib.sha256(payload).hexdigest()


_A800_REDUCTION_NAMES = {
    0: "none",
    1: "in_place",
    2: "fp32_external",
    4: "output_dtype_external",
}


def project_nvidia_a800_gemm(
        expanded: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """Project expanded cuBLASLt evidence to launch-selection fields."""

    reduction_names = _A800_REDUCTION_NAMES
    operations = expanded["operations"]
    workloads = expanded["workloads"]
    if not isinstance(operations, dict) or not isinstance(workloads, dict):
        raise TypeError("expanded A800 GEMM operations/workloads must be mappings")
    entries = {}
    for workload_key, operation_key in workloads.items():
        operation = operations[operation_key]
        projection = operation["projection"]
        execution_variant = projection["execution_variant"]
        if execution_variant == "gemv":
            tile = None
        elif execution_variant == "gemm":
            tile = [
                projection["logical_tile_m"],
                projection["logical_tile_n"],
                projection["tile_k"],
            ]
        else:
            raise ValueError(
                f"unsupported A800 GEMM execution variant {execution_variant!r}")
        reduction_scheme = projection["reduction_scheme"]
        if reduction_scheme not in reduction_names:
            raise ValueError(
                f"unsupported A800 GEMM reduction scheme {reduction_scheme}")
        entries[workload_key] = {
            "tile": tile,
            "pipeline_depth": projection["pipeline_depth"],
            "split_k": projection["split_k"],
            "reduction": reduction_names[reduction_scheme],
            "cta_raster": dict(projection["cta_raster"]),
        }
    return entries


def project_nvidia_a800_kernel_dataset_gemm(
        expanded: Mapping[str, object]) -> dict[tuple[object, ...], object]:
    """Project the kernel-dataset launch domain onto physical CSV keys."""

    operations = expanded["operations"]
    if not isinstance(operations, dict):
        raise TypeError("expanded A800 GEMM operations must be a mapping")
    entries = {}
    for operation_key, selection_key in expanded["kernel_dataset_selection"]:
        projection = operations[operation_key]["projection"]
        execution_variant = projection["execution_variant"]
        if execution_variant == "gemv":
            tile = None
        elif execution_variant == "gemm":
            tile = [
                projection["logical_tile_m"],
                projection["logical_tile_n"],
                projection["tile_k"],
            ]
        else:
            raise ValueError(
                f"unsupported A800 GEMM execution variant {execution_variant!r}")
        reduction_scheme = projection["reduction_scheme"]
        if reduction_scheme not in _A800_REDUCTION_NAMES:
            raise ValueError(
                f"unsupported A800 GEMM reduction scheme {reduction_scheme}")
        entries[tuple(selection_key)] = {
            "tile": tile,
            "pipeline_depth": projection["pipeline_depth"],
            "split_k": projection["split_k"],
            "reduction": _A800_REDUCTION_NAMES[reduction_scheme],
            "cta_raster": dict(projection["cta_raster"]),
        }
    return entries


def project_metax_c500_gemm(
        expanded: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """Project expanded mcBLAS evidence to launch-selection fields."""

    operations = expanded["operations"]
    workloads = expanded["workloads"]
    if not isinstance(operations, dict) or not isinstance(workloads, dict):
        raise TypeError("expanded C500 GEMM operations/workloads must be mappings")
    entries = {}
    for workload_key, operation_key in workloads.items():
        projection = operations[operation_key]["projection"]
        tile_value = projection["tile"]
        if tile_value is None:
            tile = None
        else:
            tile = [tile_value["m"], tile_value["n"], tile_value["k"]]
        partial = projection["partial_buffer"]
        if partial is None:
            reduction = "none"
        elif partial["contract"] == (
                "mcblas_devdstelementwiseadd_bf16_fp32_partial"):
            reduction = "fp32_external"
        else:
            raise ValueError(
                f"unsupported C500 GEMM reduction contract {partial['contract']!r}")
        entries[workload_key] = {
            "tile": tile,
            "pipeline_depth": projection["pipeline_depth"],
            "split_k": projection["split_k"],
            "reduction": reduction,
        }
    return entries


def project_ascend_910b_gemm(
        expanded: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """Project expanded CANN evidence to launch-selection fields."""

    raw_entries = expanded["entries"]
    if not isinstance(raw_entries, dict):
        raise TypeError("expanded 910B GEMM entries must be a mapping")
    entries = {}
    from src.kernel.launch_config.ascend_910b.gemm_launch import (
        decode_matmul_tiling_payload,
    )
    for workload_key, entry in raw_entries.items():
        tiling = entry["tiling"]
        if tiling["pipeline_depth"] is not None or tiling["split_k"] != 1:
            raise ValueError(
                "910B compact GEMM schema requires the fixed non-split domain")
        values = json.loads(workload_key)
        _, decoded = decode_matmul_tiling_payload(
            bytes.fromhex(entry["payload_hex"]),
            {"m": values[3], "n": values[4], "k": values[5]},
            entry["grid"][0], entry["kernel_symbol"])
        program = decoded["source_program"]
        entries[workload_key] = {
            "tile": [tiling["m"], tiling["n"], tiling["k"]],
            "pipeline_depth": None,
            "split_k": 1,
            "reduction": "none",
            "source_program": {
                field: getattr(program, field)
                for field in ASCEND_910B_GEMM_PROGRAM_FIELDS
            },
        }
    return entries


def _canonical_bool(value: object, context: str) -> str:
    if not isinstance(value, bool):
        raise TypeError(f"{context} must be boolean")
    return "true" if value else "false"


def _positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _optional_positive_integer(value: object, context: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, context)


# Semantic-entry field that holds each projected launch-record extension.
_PROJECTED_EXTENSION_FIELDS = {
    "cta_raster": "cta_raster",
    "cann_source_program": "source_program",
}


def _normalize_launch(
        value: object, context: str,
        hardware_id: str,
        ) -> tuple[object, ...]:
    # The projected launch carries the one record extension its hardware
    # declares (see ``gemm_launch_record_layout``).
    extension = gemm_launch_record_layout(hardware_id).extension
    expected_fields = {"tile", "pipeline_depth", "split_k", "reduction"}
    if extension in _PROJECTED_EXTENSION_FIELDS:
        expected_fields.add(_PROJECTED_EXTENSION_FIELDS[extension])
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError(f"{context} fields differ")
    tile = value["tile"]
    if tile is None:
        tile_values = (None, None, None)
    else:
        if not isinstance(tile, list) or len(tile) != 3:
            raise ValueError(f"{context}.tile must contain three values")
        tile_values = tuple(
            _positive_integer(item, f"{context}.tile[{index}]")
            for index, item in enumerate(tile))
    pipeline_depth = _optional_positive_integer(
        value["pipeline_depth"], f"{context}.pipeline_depth")
    if tile is None and pipeline_depth is not None:
        raise ValueError(f"{context} untiled launch declares a pipeline")
    split_k = _positive_integer(value["split_k"], f"{context}.split_k")
    reduction = value["reduction"]
    if reduction not in {
            "none", "in_place", "fp32_external", "output_dtype_external"}:
        raise ValueError(f"{context}.reduction is unsupported")
    if ((split_k == 1) != (reduction == "none")
            or tile is None and split_k != 1):
        raise ValueError(f"{context} split/reduction contract differs")
    compact = (*tile_values, pipeline_depth, split_k, reduction)
    if extension == "cta_raster":
        raster = value["cta_raster"]
        if (not isinstance(raster, Mapping)
                or set(raster) != {
                    "convention", "log_group", "serpentine", "k_slices"}):
            raise ValueError(f"{context}.cta_raster fields differ")
        convention = raster["convention"]
        if convention not in {
                "none", "cublaslt_m_group", "cutlass_n_group"}:
            raise ValueError(f"{context}.cta_raster.convention is unsupported")
        log_group = raster["log_group"]
        if (isinstance(log_group, bool) or not isinstance(log_group, int)
                or log_group < 0):
            raise ValueError(
                f"{context}.cta_raster.log_group must be a non-negative "
                f"integer")
        serpentine = _canonical_bool(
            raster["serpentine"], f"{context}.cta_raster.serpentine")
        if convention == "none" and (log_group or serpentine == "true"):
            raise ValueError(
                f"{context}.cta_raster declares a program without a "
                f"convention")
        if convention == "cutlass_n_group" and serpentine == "true":
            raise ValueError(
                f"{context}.cta_raster CUTLASS map has no serpentine")
        if tile is None and convention != "none":
            raise ValueError(
                f"{context}.cta_raster is set on an untiled launch")
        k_slices = _positive_integer(
            raster["k_slices"], f"{context}.cta_raster.k_slices")
        if k_slices > split_k:
            raise ValueError(
                f"{context}.cta_raster K slices exceed the K split")
        return (*compact, convention, log_group, serpentine, k_slices)
    if extension != "cann_source_program":
        return compact
    source_program = value["source_program"]
    if (not isinstance(source_program, Mapping)
            or set(source_program) != set(ASCEND_910B_GEMM_PROGRAM_FIELDS)):
        raise ValueError(f"{context}.source_program fields differ")
    integer_fields = ASCEND_910B_GEMM_PROGRAM_FIELDS[:20]
    program_values = tuple(
        _positive_integer(source_program[field], f"{context}.{field}")
        for field in integer_fields)
    traversal_order = source_program["traversal_order"]
    if traversal_order not in {
            "persistent_n_then_m", "m_then_n_snake", "n_then_m_snake"}:
        raise ValueError(f"{context}.traversal_order is unsupported")
    l2_flags = tuple(
        _canonical_bool(source_program[field], f"{context}.{field}")
        for field in ("a_uses_l2", "b_uses_l2"))
    if program_values[0] != program_values[1] * program_values[2] * program_values[3]:
        raise ValueError(f"{context} program grid differs from workers")
    if program_values[3] != split_k:
        raise ValueError(f"{context} program split-K differs from launch")
    if program_values[7:10] != tile_values:
        raise ValueError(f"{context} program base tile differs from launch")
    return (*compact, *program_values, traversal_order, *l2_flags)


def build_gemm_launch_selection_artifact(
        semantic_entries: Mapping[str, object], *, hardware_id: str,
        software_revision: str, generated_by: str, evidence_sha256: str,
        evidence_sources: list[Mapping[str, str]],
        capture_baseline_commit: str, domain_sha256: str,
        repository_root: Path,
        physical_entries: Mapping[tuple[object, ...], object] | None = None,
        ) -> tuple[str, dict[str, object]]:
    """Build deterministic CSV and manifest bytes from semantic launch facts.

    ``physical_entries`` carries launches whose capture domain is stated by a
    physical selection key rather than by a semantic workload key.  They share
    the one schema and must agree with any semantic entry for the same key.
    """

    if not isinstance(semantic_entries, Mapping) or not semantic_entries:
        raise ValueError("GEMM semantic launch entries must be non-empty")
    physical: dict[
        tuple[object, ...],
        tuple[object, ...],
    ] = {}
    for workload_key, launch_value in semantic_entries.items():
        key = gemm_launch_selection_key_from_workload_key(workload_key)
        launch = _normalize_launch(
            launch_value, f"GEMM launch entry {workload_key}", hardware_id)
        if key in physical and physical[key] != launch:
            raise ValueError(
                "conflicting GEMM launches for physical key "
                f"{key!r}: {physical[key]!r} != {launch!r}")
        physical[key] = launch
    for key, launch_value in (physical_entries or {}).items():
        if not isinstance(key, tuple) or len(key) != len(
                GEMM_LAUNCH_SELECTION_KEY_FIELDS):
            raise ValueError("GEMM physical launch key shape differs")
        launch = _normalize_launch(
            launch_value, f"GEMM physical launch entry {key!r}", hardware_id)
        if key in physical and physical[key] != launch:
            raise ValueError(
                "conflicting GEMM launches for physical key "
                f"{key!r}: {physical[key]!r} != {launch!r}")
        physical[key] = launch

    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(gemm_launch_csv_fields(hardware_id))
    for key in sorted(physical):
        launch = physical[key]
        key_fields = [
            _canonical_bool(value, str(name)) if isinstance(value, bool)
            else value
            for name, value in zip(
                GEMM_LAUNCH_SELECTION_KEY_FIELDS, key, strict=True)
        ]
        launch_fields = ["" if value is None else value for value in launch]
        writer.writerow([*key_fields, *launch_fields])
    csv_text = stream.getvalue()
    csv_bytes = csv_text.encode("utf-8")
    compressed_csv_bytes = compress_gemm_launch_csv(csv_bytes)
    csv_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    config_path = f"configs/hardware/{hardware_id}.yaml"
    launch_contract = load_gemm_launch_hardware_contract(
        Path(repository_root) / config_path, hardware_id=hardware_id)
    manifest = {
        "schema_version": gemm_launch_schema_version(hardware_id),
        "artifact_type": GEMM_LAUNCH_ARTIFACT_TYPE,
        "hardware_id": hardware_id,
        "hardware_revision": {
            "config_path": config_path,
            "launch_contract": launch_contract,
            "launch_contract_sha256": (
                gemm_launch_hardware_contract_sha256(launch_contract)),
        },
        "software_revision": software_revision,
        "generated_by": generated_by,
        "evidence_sha256": evidence_sha256,
        "evidence_sources": [dict(item) for item in evidence_sources],
        "capture_baseline_commit": capture_baseline_commit,
        "domain_sha256": domain_sha256,
        "csv_sha256": csv_sha256,
        "csv_compressed_sha256": hashlib.sha256(
            compressed_csv_bytes).hexdigest(),
        "compression": GEMM_LAUNCH_COMPRESSION,
        "row_count": len(physical),
        "selection_key_fields": list(GEMM_LAUNCH_SELECTION_KEY_FIELDS),
        "launch_fields": list(gemm_launch_fields(hardware_id)),
        "semantic_context_fields": list(GEMM_SEMANTIC_CONTEXT_FIELDS),
        "boolean_encoding": GEMM_BOOLEAN_ENCODING,
        "null_encoding": GEMM_NULL_ENCODING,
        "duration_consumed": False,
    }
    return csv_text, manifest


def write_gemm_launch_selection_artifact(
        output_directory: Path, csv_text: str,
        manifest: Mapping[str, object]) -> None:
    """Write one canonical pair after all in-memory validation succeeds."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / GEMM_LAUNCH_CSV_FILENAME
    manifest_path = output_directory / GEMM_LAUNCH_MANIFEST_FILENAME
    temporary_csv = output_directory / f".{GEMM_LAUNCH_CSV_FILENAME}.tmp"
    temporary_manifest = (
        output_directory / f".{GEMM_LAUNCH_MANIFEST_FILENAME}.tmp")
    try:
        temporary_csv.write_bytes(
            compress_gemm_launch_csv(csv_text.encode("utf-8")))
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        load_gemm_launch_selection_files(
            temporary_csv,
            temporary_manifest,
            hardware_id=manifest["hardware_id"],
            software_revision=manifest["software_revision"],
        )
        temporary_csv.replace(csv_path)
        temporary_manifest.replace(manifest_path)
    finally:
        if temporary_csv.exists():
            temporary_csv.unlink()
        if temporary_manifest.exists():
            temporary_manifest.unlink()


def compact_ascend_910b_swiglu(
        expanded: Mapping[str, object]) -> dict[str, object]:
    """Keep only the source-selected SwiGLU tile and queue depth."""

    raw_entries = expanded["entries"]
    if not isinstance(raw_entries, dict):
        raise TypeError("expanded 910B SwiGLU entries must be a mapping")
    entries = {}
    for workload_key, entry in raw_entries.items():
        entries[workload_key] = {
            "tile": [entry["base_row_len"], entry["base_col_len"]],
            "queue_depth": entry["queue_depth"],
        }
    return {
        "schema_version": 5,
        "artifact_type": "ascend_910b_swiglu_launch_selection",
        "hardware_id": expanded["hardware_id"],
        "software_revision": expanded["software_revision"],
        "generated_by": expanded["generated_by"],
        "duration_consumed": False,
        "target_kernel_submitted": False,
        "workload_key_fields": expanded["workload_key_fields"],
        "aiv_capacity": 48,
        "entry_count": len(entries),
        "provenance_groups": [{
            "group_id": "dense_exact_domain",
            "coverage_kind": "dense_cartesian",
            "m_min": expanded["exact_domain"]["m_min"],
            "m_max": expanded["exact_domain"]["m_max"],
            "intermediate_values": expanded["exact_domain"][
                "intermediate_values"],
            "workload_keys": [],
            "candidate_manifest_sha256": expanded[
                "candidate_manifest_sha256"],
            "capture_baseline_commit": expanded["capture_baseline_commit"],
            "source_snapshot_sha256": expanded["source_snapshot_sha256"],
            "cann_snapshot_sha256": expanded["cann_provenance_sha256"],
            "evidence_sha256": expanded_evidence_sha256(expanded),
        }],
        "entries": entries,
    }


def _compact_flash_decode(value: Mapping[str, object]) -> dict[str, object]:
    head_fields = (
        value["head_b_n2"],
        value["head_g_s1"],
        value["head_s2_split_count"],
        value["head_g_s1_split_count"],
        value["head_g_s1_last_size"],
    )
    vector_fields = (
        value["vector_worker_head_end"],
        value["vector_worker_head_split_end"],
    )
    return {
        "heads": [list(items) for items in zip(*head_fields)],
        "vector_worker_ends": [
            list(items) for items in zip(*vector_fields)],
    }


def compact_ascend_910b_flash_attn(
        expanded: Mapping[str, object]) -> dict[str, object]:
    """Keep only CANN variant and dynamic AIC/AIV schedule decisions."""

    raw_entries = expanded["entries"]
    if not isinstance(raw_entries, dict):
        raise TypeError("expanded 910B FlashAttention entries must be a mapping")
    entries = {}
    for workload_key, entry in raw_entries.items():
        source_schedule = entry["source_schedule"]
        if source_schedule is None:
            schedule = None
        else:
            worker_ends = source_schedule["worker_exclusive_ends"]
            schedule = {
                "tile": [
                    source_schedule["inner_split"]["m_base_size"],
                    source_schedule["inner_split"]["s2_base_size"],
                ],
                "worker_endpoints": [
                    list(items) for items in zip(
                        worker_ends["b_n2"],
                        worker_ends["g_s1"],
                        worker_ends["s2"],
                    )
                ],
                "flash_decode": (
                    None
                    if source_schedule["flash_decode"] is None
                    else _compact_flash_decode(
                        source_schedule["flash_decode"])),
            }
        entries[workload_key] = {
            "runtime_variant": entry["runtime_variant"],
            "aic_count": entry["block_dim"],
            "schedule": schedule,
        }
    return {
        "schema_version": 4,
        "artifact_type": "ascend_910b_flash_attn_launch_selection",
        "hardware_id": expanded["hardware_id"],
        "software_revision": expanded["software_revision"],
        "generated_by": expanded["generated_by"],
        "evidence_sha256": expanded_evidence_sha256(expanded),
        "source_sha256": expanded["source_sha256"],
        "workload_contract_sha256": expanded["workload_contract_sha256"],
        "source_schedule_reference_sha256": expanded[
            "source_schedule_reference_sha256"],
        "duration_consumed": False,
        "workload_key_fields": expanded["workload_key_fields"],
        "aiv_per_aic": 2,
        "output_layout": 0,
        "softmax_lse_flag": 0,
        "flash_decode_g_s1_base_size": 8,
        "entry_count": len(entries),
        "entries": entries,
    }
