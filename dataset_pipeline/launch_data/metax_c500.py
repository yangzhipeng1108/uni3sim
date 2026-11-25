#!/usr/bin/env python3
"""Validate/package exact MetaX C500 GEMM launch observations.

The GEMM input is the allocation-free mcBLAS host-selector capture produced by
``microbench/metax_c500/operators/gemm/dry_launch_runner.py``. This module
accepts only the complete ShareGPT launch domain and retains ordered launch
facts; it never consumes a kernel duration or executes GEMM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from dataset_pipeline.launch_data.gemm_workload_candidates import (
    build_gemm_workload_candidate,
)
from dataset_pipeline.launch_data.json_validators import (
    require_array,
    require_exact_fields,
    require_integer,
    require_nonempty_string,
    require_object,
    require_sha256_hex,
    sha256_bytes,
)
from src.kernel.launch_config.gemm_workload import (
    GEMM_WORKLOAD_KEY_FIELDS,
    validate_gemm_workload_key,
)
from dataset_pipeline.launch_data.canonical_projection import (
    GEMM_DOMAIN_SHA256,
    build_gemm_launch_selection_artifact,
    expanded_evidence_sha256,
    project_metax_c500_gemm,
    write_gemm_launch_selection_artifact,
)
from src.kernel.launch_config.gemm_launch_selection import (
    GEMM_LAUNCH_CSV_FILENAME,
    GEMM_LAUNCH_MANIFEST_FILENAME,
    ARTIFACT_MANIFEST_FIELDS,
    METAX_CANONICAL_FIELDS,
    METAX_LAUNCH_FIELDS,
    METAX_RESULT_FIELDS,
    METAX_SCHEDULER_FIELDS,
    METAX_SOURCE_PATHS,
    PROHIBITED_TIMING_FIELD_PARTS,
    load_gemm_launch_selection_files,
)
from src.kernel.launch_config.tiling_spec import parse_complete_gemm_tile


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "launch_data" / "metax_c500"
GEMM_CSV_PATH = DATA_DIR / GEMM_LAUNCH_CSV_FILENAME
GEMM_MANIFEST_PATH = DATA_DIR / GEMM_LAUNCH_MANIFEST_FILENAME
DOMAIN_CONFIG = (
    ROOT / "configs" / "evaluation" / "experiments"
    / "gemm_launch_domain.json"
)

EXPECTED_SEMANTIC_KEY_COUNT = 140544
EXPECTED_PHYSICAL_OPERATION_COUNT = 140544
EXPECTED_MCBLAS_VERSION = 10
EXPECTED_MACA_PATH = "/opt/maca-3.7.2"
SOFTWARE_REVISION = "maca_3.7.2_mcblas_10"
SELECTOR_CONTRACT = "mcblas_default_host_selector_dry_launch"

_EXPECTED_WORKLOAD_MANIFEST_SHA256 = (
    "60c39f865cf0acff07c3906ab8167f2bba20bfad90dad2eb65e1827c371c9af4")

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_OPERATION_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_MAIN_GEMM_PATTERN = re.compile(
    r"^mcblas__Mck_bf16gemm_tn_[0-9]+x[0-9]+x[0-9]+_")
_UNALIGNED_GEMM_PATTERN = re.compile(
    r"^_ZN8cpp_gemm12unalign_gemmL6LaunchINS0_19unalign_gemm_kernelI")
_GEMV_IDENTITIES = ("b16gemvt_kernel", "gemv")
_MANGLED_REDUCER_PREFIX = (
    "_ZN6mcblas20DevDstElementwiseAddI15__maca_bfloat16ffLi0EEEvPT_"
    "PKS2_PKT0_"
)
_DEMANGLED_REDUCER_PREFIX = (
    "void mcblas::DevDstElementwiseAdd<__maca_bfloat16, float, float, 0>("
)
_CANDIDATE_FIELDS = {
    "schema_version", "artifact_type", "hardware_id", "mechanism",
    "gemm_executed", "duration_consumed",
    "production_or_evaluation_duration_consumed", "input_manifest",
    "environment", "interposition", "selector_operations",
    "selector_results", "exact_key_map",
}
_INPUT_MANIFEST_FIELDS = {
    "path", "sha256", "schema_version", "artifact_type", "source_config",
    "source_config_sha256", "workload_key_fields", "hardware_id",
    "exact_key_count", "selected_exact_key_count",
}
_ENVIRONMENT_FIELDS = {
    "source_identity", "source_sha256", "maca_path", "mcblas_version",
    "device", "compiler", "host_platform",
}
_SOURCE_IDENTITY_FIELDS = {"mode", "baseline_commit", "marker_path"}
_INTERPOSITION_FIELDS = {
    "required_launch_apis", "unsupported_execution_apis_blocked",
    "initial_expected_launch_count", "positive_count_mismatch_policy",
    "zero_launch_policy", "initialization_launch_policy",
    "active_marker_launch_policy", "full_selector_operation_count",
    "selected_selector_operation_count", "selector_operation_order",
    "selector_operation_range", "selector_attempt_count",
    "captured_launch_count",
}
_OPERATION_FIELDS = {"operation_key", "workload"}
_WORKLOAD_FIELDS = {
    "batch", "m", "n", "k", "dtype", "layout", "algorithm",
    "algorithm_value",
}
_HARDWARE_DOMAIN_FIELDS = {
    "status", "domain_sha256", "semantic_key_count",
    "physical_operation_count",
}
_ALLOWED_TIMING_BOUNDARY_FIELDS = {
    "duration_consumed",
    "production_or_evaluation_duration_consumed",
    "timing_consumed",
}


def _reject_duration_fields(value: object, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{context} contains a non-string field")
            lowered_key = key.lower()
            if (any(part in lowered_key
                    for part in PROHIBITED_TIMING_FIELD_PARTS)
                    and key not in _ALLOWED_TIMING_BOUNDARY_FIELDS):
                raise ValueError(
                    f"{context} contains forbidden timing field {key!r}")
            if (key in _ALLOWED_TIMING_BOUNDARY_FIELDS
                    and item is not False):
                raise ValueError(f"{context}.{key} must be false")
            _reject_duration_fields(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_duration_fields(item, f"{context}[{index}]")


def _dim3(value: object, context: str) -> list[int]:
    dimensions = require_array(value, context)
    if len(dimensions) != 3:
        raise ValueError(f"{context} must contain exactly three dimensions")
    return [
        require_integer(item, f"{context}[{index}]", positive=True)
        for index, item in enumerate(dimensions)
    ]


def _validate_manifest(manifest: Mapping[str, object]) -> None:
    require_exact_fields(manifest, ARTIFACT_MANIFEST_FIELDS, "workload_manifest")
    _reject_duration_fields(manifest, "workload_manifest")
    if manifest["source_config"] != str(DOMAIN_CONFIG.relative_to(ROOT)):
        raise ValueError("candidate workload manifest targets an unknown config")
    if manifest["source_config_sha256"] != sha256_bytes(DOMAIN_CONFIG.read_bytes()):
        raise ValueError("candidate workload source config digest differs")
    if manifest["schema_version"] != 2:
        raise ValueError("workload manifest must use schema_version 2")
    if manifest["artifact_type"] != "gemm_workload_candidate":
        raise ValueError("workload manifest has the wrong artifact_type")
    if manifest["duration_consumed"] is not False:
        raise ValueError("workload manifest consumed duration evidence")
    if manifest["semantic_key_count"] != EXPECTED_SEMANTIC_KEY_COUNT:
        raise ValueError(
            "workload manifest does not contain the full 140544-key domain")
    if (manifest["physical_operation_count"]
            != EXPECTED_PHYSICAL_OPERATION_COUNT):
        raise ValueError(
            "workload manifest physical-operation count is incomplete")
    fields = require_array(
        manifest["workload_key_fields"],
        "workload_manifest.workload_key_fields")
    if tuple(fields) != GEMM_WORKLOAD_KEY_FIELDS:
        raise ValueError("workload manifest does not use the 16-field key")
    physical_fields = require_array(
        manifest["physical_operation_fields"],
        "workload_manifest.physical_operation_fields")
    if tuple(physical_fields) != ("m", "n", "k", "dtype", "has_bias"):
        raise ValueError("workload manifest physical-operation fields differ")

    config = require_object(json.loads(DOMAIN_CONFIG.read_bytes()), "domain config")
    scheduler_expected = require_object(
        config["scheduler_contract"], "domain config scheduler_contract")
    scheduler = require_object(
        manifest["scheduler_contract"],
        "workload_manifest.scheduler_contract")
    require_exact_fields(
        scheduler, METAX_SCHEDULER_FIELDS,
        "workload_manifest.scheduler_contract")
    if scheduler != scheduler_expected:
        raise ValueError("workload manifest scheduler_contract differs")

    workload_keys = require_array(
        manifest["workload_keys"], "workload_manifest.workload_keys")
    if (len(workload_keys) != EXPECTED_SEMANTIC_KEY_COUNT
            or workload_keys != sorted(set(workload_keys))):
        raise ValueError("workload manifest keys are not complete and exact")
    for index, semantic_key in enumerate(workload_keys):
        if not isinstance(semantic_key, str):
            raise TypeError(
                f"workload_manifest.workload_keys[{index}] must be a string")
        validate_gemm_workload_key(semantic_key)

    domains = require_object(
        manifest["hardware_domains"], "workload_manifest.hardware_domains")
    domain = require_object(
        domains["metax_c500"],
        "workload_manifest.hardware_domains.metax_c500")
    require_exact_fields(
        domain, _HARDWARE_DOMAIN_FIELDS,
        "workload_manifest.hardware_domains.metax_c500")
    if (domain["status"] != "exact"
            or domain["semantic_key_count"] != EXPECTED_SEMANTIC_KEY_COUNT
            or domain["physical_operation_count"]
            != EXPECTED_PHYSICAL_OPERATION_COUNT):
        raise ValueError("MetaX C500 workload domain is not complete and exact")

    manifest_digest = sha256_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    if manifest_digest != _EXPECTED_WORKLOAD_MANIFEST_SHA256:
        raise ValueError("workload manifest differs from the frozen exact domain")


def _expected_operation(semantic_key: str) -> tuple[str, str, dict[str, object]]:
    validate_gemm_workload_key(semantic_key)
    values = json.loads(semantic_key)
    fields = dict(zip(GEMM_WORKLOAD_KEY_FIELDS, values, strict=True))
    operation_key = json.dumps([
        fields["b"], fields["m"], fields["n"], fields["k"],
        "bfloat16", "row_major_linear_tn", "MCBLAS_GEMM_DEFAULT",
    ], separators=(",", ":"))
    operation_id = "sha256:" + sha256_bytes(operation_key.encode("utf-8"))
    workload = {
        "batch": fields["b"],
        "m": fields["m"],
        "n": fields["n"],
        "k": fields["k"],
        "dtype": "bfloat16",
        "layout": "row_major_linear_tn",
        "algorithm": "MCBLAS_GEMM_DEFAULT",
        "algorithm_value": -1,
    }
    return operation_id, operation_key, workload


def _validate_workload(
        value: object, operation_key: str, context: str) -> dict[str, object]:
    workload = require_object(value, context)
    require_exact_fields(workload, _WORKLOAD_FIELDS, context)
    for field in ("batch", "m", "n", "k"):
        require_integer(workload[field], f"{context}.{field}", positive=True)
    if workload["batch"] != 1:
        raise ValueError(f"{context}.batch must be one")
    if (workload["dtype"] != "bfloat16"
            or workload["layout"] != "row_major_linear_tn"
            or workload["algorithm"] != "MCBLAS_GEMM_DEFAULT"
            or workload["algorithm_value"] != -1):
        raise ValueError(f"{context} differs from the frozen selector ABI")
    rebuilt_key = json.dumps([
        workload["batch"], workload["m"], workload["n"], workload["k"],
        workload["dtype"], workload["layout"], workload["algorithm"],
    ], separators=(",", ":"))
    if rebuilt_key != operation_key:
        raise ValueError(f"{context} disagrees with operation_key")
    return dict(workload)


def _validate_launch(value: object, ordinal: int, context: str) -> dict[str, object]:
    launch = require_object(value, context)
    require_exact_fields(launch, METAX_LAUNCH_FIELDS, context)
    if launch["record_type"] != "launch" or launch["ordinal"] != ordinal:
        raise ValueError(f"{context} has a non-contiguous launch ordinal")
    if launch["api"] not in ("mcLaunchKernel", "mcModuleLaunchKernel"):
        raise ValueError(f"{context} uses an unsupported launch API")
    if launch["identity_source"] not in (
            "mcModuleGetFunction", "mcFuncGetName", "dladdr"):
        raise ValueError(f"{context} has unresolved kernel identity provenance")
    require_nonempty_string(launch["kernel_identity"], f"{context}.kernel_identity")
    grid = _dim3(launch["grid"], f"{context}.grid")
    block = _dim3(launch["block"], f"{context}.block")
    dynamic_smem = require_integer(
        launch["dynamic_shared_memory_bytes"],
        f"{context}.dynamic_shared_memory_bytes")
    if launch["static_shared_memory_bytes"] is not None:
        raise ValueError(f"{context} invents static shared memory")
    if (launch["static_shared_memory_status"]
            != "not_exposed_by_launch_api"):
        raise ValueError(f"{context} has invalid static shared-memory status")
    return {
        "record_type": "launch",
        "ordinal": ordinal,
        "api": launch["api"],
        "identity_source": launch["identity_source"],
        "kernel_identity": launch["kernel_identity"],
        "grid": grid,
        "block": block,
        "dynamic_shared_memory_bytes": dynamic_smem,
        "static_shared_memory_bytes": None,
        "static_shared_memory_status": "not_exposed_by_launch_api",
    }


def _is_source_backed_reducer(name: str) -> bool:
    return (
        name.startswith(_MANGLED_REDUCER_PREFIX)
        or name.startswith(_DEMANGLED_REDUCER_PREFIX)
    )


def _project_launches(
        workload: Mapping[str, object], launches: Sequence[Mapping[str, object]],
        context: str) -> dict[str, object]:
    if len(launches) not in (1, 2):
        raise NotImplementedError(
            f"{context} has unsupported mcBLAS launch count {len(launches)}")
    main = launches[0]
    main_name = main["kernel_identity"]
    lowered_name = main_name.lower()
    if (_MAIN_GEMM_PATTERN.match(main_name) is not None
            or _UNALIGNED_GEMM_PATTERN.match(main_name) is not None):
        execution_variant = "gemm"
        parsed = parse_complete_gemm_tile(main_name)
        if parsed is None:
            raise ValueError(
                f"{context} main kernel has no source-backed parsed tile")
        tile_m, tile_n, tile_k, pipeline_depth = parsed
        if tile_k is None:
            raise ValueError(f"{context} main kernel has no K tile")
        tile = {"m": tile_m, "n": tile_n, "k": tile_k}
    elif any(identity in lowered_name for identity in _GEMV_IDENTITIES):
        execution_variant = "gemv"
        tile = None
        pipeline_depth = None
    else:
        raise NotImplementedError(
            f"{context} has unsupported mcBLAS main kernel {main_name!r}")
    split_k = main["grid"][2]
    if len(launches) == 1:
        if split_k != 1:
            raise NotImplementedError(
                f"{context} split-K main has no captured reducer")
        reducer_ordinal = None
        partial_buffer = None
    else:
        if execution_variant != "gemm" or split_k <= 1:
            raise ValueError(
                f"{context} reducer requires a tensor main with split_k > 1")
        reducer = launches[1]
        if not _is_source_backed_reducer(reducer["kernel_identity"]):
            raise NotImplementedError(
                f"{context} reducer ABI is unsupported: "
                f"{reducer['kernel_identity']!r}")
        if reducer["api"] != "mcLaunchKernel":
            raise ValueError(f"{context} reducer launch API differs from evidence")
        reducer_ordinal = 2
        partial_buffer = {
            "contract": "mcblas_devdstelementwiseadd_bf16_fp32_partial",
            "element_bytes": 4,
            "dram_bytes_per_element": 4,
        }
    return {
        "execution_variant": execution_variant,
        "main_ordinal": 1,
        "tile": tile,
        "pipeline_depth": pipeline_depth,
        "split_k": split_k,
        "reducer_ordinal": reducer_ordinal,
        "partial_buffer": partial_buffer,
    }


def _validate_result(
        value: object, workload: Mapping[str, object], context: str,
        ) -> tuple[dict[str, object], dict[str, object]]:
    result = require_object(value, context)
    require_exact_fields(result, METAX_RESULT_FIELDS, context)
    if result["gemm_executed"] is not False:
        raise ValueError(f"{context} executed GEMM")
    if result["duration_consumed"] is not False:
        raise ValueError(f"{context} consumed duration evidence")
    attempts = require_integer(
        result["selector_attempt_count"],
        f"{context}.selector_attempt_count", positive=True)
    initial = require_integer(
        result["initial_expected_launch_count"],
        f"{context}.initial_expected_launch_count", positive=True)
    validated = require_integer(
        result["validated_launch_count"],
        f"{context}.validated_launch_count", positive=True)
    stability = require_integer(
        result["stability_capture_count"],
        f"{context}.stability_capture_count", positive=True)
    raw_launches = require_array(result["launches"], f"{context}.launches")
    launches = [
        _validate_launch(item, ordinal, f"{context}.launches[{ordinal - 1}]")
        for ordinal, item in enumerate(raw_launches, start=1)
    ]
    if initial != 1 or validated != len(launches):
        raise ValueError(f"{context} launch-count evidence is inconsistent")
    if len(launches) == 1:
        if attempts != 1 or stability != 1:
            raise ValueError(f"{context} single launch has invalid stability evidence")
    elif attempts != 3 or stability != 2:
        raise ValueError(f"{context} discovered launch count lacks two captures")
    projection = _project_launches(workload, launches, context)
    canonical_result = {
        "gemm_executed": False,
        "duration_consumed": False,
        "selector_attempt_count": attempts,
        "initial_expected_launch_count": initial,
        "validated_launch_count": validated,
        "stability_capture_count": stability,
        "launches": launches,
    }
    return canonical_result, projection


def build(
        candidate_path: Path, *, expected_baseline: str,
        models_root: Path) -> dict[str, object]:
    """Build canonical launch facts from one complete dry-launch capture."""

    candidate_path = candidate_path.resolve()
    candidate_bytes = candidate_path.read_bytes()
    candidate = require_object(json.loads(candidate_bytes), str(candidate_path))
    require_exact_fields(candidate, _CANDIDATE_FIELDS, "candidate")
    _reject_duration_fields(candidate, "candidate")
    if (candidate["schema_version"] != 1
            or candidate["artifact_type"]
            != "metax_c500_mcblas_gemm_dry_launch_candidate"
            or candidate["hardware_id"] != "metax_c500"
            or candidate["mechanism"] != SELECTOR_CONTRACT):
        raise ValueError("candidate identity or selector mechanism differs")
    if (candidate["gemm_executed"] is not False
            or candidate["duration_consumed"] is not False
            or candidate["production_or_evaluation_duration_consumed"]
            is not False):
        raise ValueError("candidate crossed the execution/duration boundary")
    if _COMMIT_PATTERN.fullmatch(expected_baseline) is None:
        raise ValueError("expected_baseline must be a full lowercase commit")

    environment = require_object(candidate["environment"], "candidate.environment")
    require_exact_fields(environment, _ENVIRONMENT_FIELDS, "candidate.environment")
    source_identity = require_object(
        environment["source_identity"], "candidate.environment.source_identity")
    require_exact_fields(
        source_identity, _SOURCE_IDENTITY_FIELDS,
        "candidate.environment.source_identity")
    if source_identity["baseline_commit"] != expected_baseline:
        raise ValueError("candidate baseline_commit does not match expectation")
    if source_identity["mode"] not in (
            "clean_exact_git_checkout", "audited_source_snapshot"):
        raise ValueError("candidate source identity mode is unsupported")
    marker_path = source_identity["marker_path"]
    if source_identity["mode"] == "clean_exact_git_checkout":
        if marker_path is not None:
            raise ValueError("clean Git source identity must have null marker_path")
    elif not isinstance(marker_path, str) or not marker_path:
        raise TypeError("audited source snapshot requires marker_path")
    else:
        pass
    source_sha256 = require_object(
        environment["source_sha256"], "candidate.environment.source_sha256")
    if set(source_sha256) != set(METAX_SOURCE_PATHS):
        raise ValueError("candidate source SHA256 set is incomplete")
    for path in METAX_SOURCE_PATHS:
        digest = require_sha256_hex(
            source_sha256[path], f"candidate.environment.source_sha256.{path}")
        if digest != sha256_bytes((ROOT / path).read_bytes()):
            raise ValueError(f"candidate source differs from current {path}")
    if (environment["maca_path"] != EXPECTED_MACA_PATH
            or environment["mcblas_version"] != EXPECTED_MCBLAS_VERSION):
        raise ValueError("candidate MACA/mcBLAS software identity differs")
    require_integer(environment["device"], "candidate.environment.device")
    require_nonempty_string(environment["compiler"], "candidate.environment.compiler")
    require_nonempty_string(
        environment["host_platform"], "candidate.environment.host_platform")

    manifest_record = require_object(
        candidate["input_manifest"], "candidate.input_manifest")
    require_exact_fields(
        manifest_record, _INPUT_MANIFEST_FIELDS, "candidate.input_manifest")
    rebuilt_manifest = build_gemm_workload_candidate(
        ROOT, DOMAIN_CONFIG, models_root=models_root)
    _validate_manifest(rebuilt_manifest)
    manifest_bytes = (
        json.dumps(rebuilt_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_sha256 = sha256_bytes(manifest_bytes)
    manifest_path = require_nonempty_string(
        manifest_record["path"], "candidate.input_manifest.path")
    if (Path(manifest_path).is_absolute()
            or not manifest_path.startswith("profiler_output/launch_data/")):
        raise ValueError("candidate input manifest path is outside its owner")
    if (manifest_record["sha256"] != manifest_sha256
            or manifest_record["schema_version"] != 2
            or manifest_record["artifact_type"] != "gemm_workload_candidate"
            or manifest_record["source_config"]
            != str(DOMAIN_CONFIG.relative_to(ROOT))
            or manifest_record["source_config_sha256"]
            != rebuilt_manifest["source_config_sha256"]
            or manifest_record["hardware_id"] != "metax_c500"
            or tuple(require_array(
                manifest_record["workload_key_fields"],
                "candidate.input_manifest.workload_key_fields"))
            != GEMM_WORKLOAD_KEY_FIELDS
            or manifest_record["exact_key_count"]
            != EXPECTED_SEMANTIC_KEY_COUNT
            or manifest_record["selected_exact_key_count"]
            != EXPECTED_SEMANTIC_KEY_COUNT):
        raise ValueError("candidate input manifest is not the complete exact domain")

    interposition = require_object(
        candidate["interposition"], "candidate.interposition")
    require_exact_fields(
        interposition, _INTERPOSITION_FIELDS, "candidate.interposition")
    expected_interposition = {
        "required_launch_apis": ["mcLaunchKernel", "mcModuleLaunchKernel"],
        "unsupported_execution_apis_blocked": [
            "mcGraphLaunch", "mcLaunchCooperativeKernel"],
        "initial_expected_launch_count": 1,
        "positive_count_mismatch_policy": (
            "discover_positive_count_then_require_two_exact_captures"),
        "zero_launch_policy": "fail_loud_without_retry",
        "initialization_launch_policy": "forwarded_outside_active_marker",
        "active_marker_launch_policy": "recorded_and_not_submitted",
        "full_selector_operation_count": EXPECTED_PHYSICAL_OPERATION_COUNT,
        "selected_selector_operation_count": EXPECTED_PHYSICAL_OPERATION_COUNT,
        "selector_operation_order": "lexicographic_operation_id",
        "selector_operation_range": {
            "start_index": 0,
            "end_index": EXPECTED_PHYSICAL_OPERATION_COUNT,
        },
    }
    for field, expected in expected_interposition.items():
        if interposition[field] != expected:
            raise ValueError(
                f"candidate.interposition.{field} differs from full capture")
    require_integer(
        interposition["selector_attempt_count"],
        "candidate.interposition.selector_attempt_count", positive=True)
    require_integer(
        interposition["captured_launch_count"],
        "candidate.interposition.captured_launch_count", positive=True)

    expected_keys = require_array(
        rebuilt_manifest["workload_keys"], "workload_manifest.workload_keys")
    exact_key_map = require_object(candidate["exact_key_map"], "candidate.exact_key_map")
    if set(exact_key_map) != set(expected_keys):
        raise ValueError("candidate exact_key_map does not cover the semantic set")
    operations = require_object(
        candidate["selector_operations"], "candidate.selector_operations")
    results = require_object(candidate["selector_results"], "candidate.selector_results")
    if (len(operations) != EXPECTED_PHYSICAL_OPERATION_COUNT
            or set(results) != set(operations)):
        raise ValueError("candidate selector operation/result set is incomplete")

    workloads = {}
    referenced_operations = set()
    for semantic_key in expected_keys:
        operation_id, _, _ = _expected_operation(semantic_key)
        mapping = require_object(
            exact_key_map[semantic_key], f"candidate.exact_key_map[{semantic_key}]")
        require_exact_fields(
            mapping, {"operation_id", "result_id"},
            f"candidate.exact_key_map[{semantic_key}]")
        if (mapping["operation_id"] != operation_id
                or mapping["result_id"] != operation_id):
            raise ValueError(
                f"candidate exact key maps to wrong operation: {semantic_key}")
        workloads[semantic_key] = operation_id
        referenced_operations.add(operation_id)
    if referenced_operations != set(operations):
        raise ValueError("candidate contains unmapped or missing selector operations")

    canonical_operations = {}
    total_attempts = 0
    total_launches = 0
    for operation_id in sorted(operations):
        if _OPERATION_ID_PATTERN.fullmatch(operation_id) is None:
            raise ValueError(f"invalid selector operation id {operation_id!r}")
        record = require_object(
            operations[operation_id], f"selector_operations[{operation_id}]")
        require_exact_fields(record, _OPERATION_FIELDS, f"selector_operations[{operation_id}]")
        operation_key = require_nonempty_string(
            record["operation_key"],
            f"selector_operations[{operation_id}].operation_key")
        if "sha256:" + sha256_bytes(operation_key.encode("utf-8")) != operation_id:
            raise ValueError(f"selector operation id/hash mismatch: {operation_id}")
        workload = _validate_workload(
            record["workload"], operation_key,
            f"selector_operations[{operation_id}].workload")
        selector_result, projection = _validate_result(
            results[operation_id], workload,
            f"selector_results[{operation_id}]")
        total_attempts += selector_result["selector_attempt_count"]
        total_launches += len(selector_result["launches"])
        canonical_operations[operation_id] = {
            "operation_key": operation_key,
            "workload": workload,
            "selector_result": selector_result,
            "projection": projection,
        }
    if (total_attempts != interposition["selector_attempt_count"]
            or total_launches != interposition["captured_launch_count"]):
        raise ValueError("candidate aggregate launch/attempt counts disagree")

    return {
        "schema_version": 2,
        "artifact_type": "metax_c500_gemm_launch_facts",
        "hardware_id": "metax_c500",
        "software_revision": SOFTWARE_REVISION,
        "maca_version": "3.7.2",
        "mcblas_version": EXPECTED_MCBLAS_VERSION,
        "source": f"candidate_sha256:{_sha256(candidate_bytes)}",
        "generated_by": "scripts/launch_data/metax_c500_launch_generator.py",
        "candidate_sha256": sha256_bytes(candidate_bytes),
        "capture_baseline_commit": expected_baseline,
        "source_sha256": dict(source_sha256),
        "workload_manifest_sha256": manifest_sha256,
        "workload_manifest": dict(rebuilt_manifest),
        "workload_key_fields": list(GEMM_WORKLOAD_KEY_FIELDS),
        "selector_contract": SELECTOR_CONTRACT,
        "duration_consumed": False,
        "gemm_executed": False,
        "semantic_key_count": EXPECTED_SEMANTIC_KEY_COUNT,
        "physical_operation_count": EXPECTED_PHYSICAL_OPERATION_COUNT,
        "workloads": workloads,
        "operations": canonical_operations,
    }


def validate_gemm(path: Path) -> tuple[dict[str, object], int]:
    """Validate the v2 canonical root and exact domain cardinality."""

    data = require_object(json.loads(path.read_bytes()), str(path))
    require_exact_fields(data, METAX_CANONICAL_FIELDS, str(path))
    if (data["schema_version"] != 2
            or data["artifact_type"] != "metax_c500_gemm_launch_facts"
            or data["hardware_id"] != "metax_c500"
            or data["software_revision"] != SOFTWARE_REVISION
            or data["maca_version"] != "3.7.2"
            or data["mcblas_version"] != EXPECTED_MCBLAS_VERSION
            or data["selector_contract"] != SELECTOR_CONTRACT
            or data["duration_consumed"] is not False
            or data["gemm_executed"] is not False):
        raise ValueError(f"{path}: canonical identity or evidence boundary differs")
    _reject_duration_fields(data, str(path))
    candidate_sha256 = require_sha256_hex(
        data["candidate_sha256"], f"{path}.candidate_sha256")
    if data["source"] != f"candidate_sha256:{candidate_sha256}":
        raise ValueError(f"{path}: source does not bind the candidate identity")
    if (_COMMIT_PATTERN.fullmatch(data["capture_baseline_commit"])
            is None):
        raise ValueError(f"{path}: capture baseline is not a full commit")
    if (data["generated_by"]
            != "scripts/launch_data/metax_c500_launch_generator.py"):
        raise ValueError(f"{path}: generated_by differs")
    source_sha256 = require_object(data["source_sha256"], f"{path}.source_sha256")
    if set(source_sha256) != set(METAX_SOURCE_PATHS):
        raise ValueError(f"{path}: source SHA256 set is incomplete")
    for source_path in METAX_SOURCE_PATHS:
        digest = require_sha256_hex(
            source_sha256[source_path], f"{path}.source_sha256.{source_path}")
        if digest != sha256_bytes((ROOT / source_path).read_bytes()):
            raise ValueError(f"{path}: source digest differs for {source_path}")
    manifest = require_object(data["workload_manifest"], f"{path}.workload_manifest")
    if set(manifest) != ARTIFACT_MANIFEST_FIELDS:
        raise ValueError(f"{path}: workload manifest fields differ")
    _validate_manifest(manifest)
    manifest_digest = sha256_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    if data["workload_manifest_sha256"] != manifest_digest:
        raise ValueError(f"{path}: workload manifest digest differs")
    if tuple(data["workload_key_fields"]) != GEMM_WORKLOAD_KEY_FIELDS:
        raise ValueError(f"{path}: workload key fields differ")
    expected_keys = require_array(
        manifest["workload_keys"], f"{path}.workload_manifest.workload_keys")
    if expected_keys != sorted(set(expected_keys)):
        raise ValueError(f"{path}: workload keys are not sorted and unique")
    workloads = require_object(data["workloads"], f"{path}.workloads")
    operations = require_object(data["operations"], f"{path}.operations")
    if (data["semantic_key_count"] != EXPECTED_SEMANTIC_KEY_COUNT
            or len(workloads) != EXPECTED_SEMANTIC_KEY_COUNT):
        raise ValueError(f"{path}: semantic workload coverage is incomplete")
    if (data["physical_operation_count"]
            != EXPECTED_PHYSICAL_OPERATION_COUNT
            or len(operations) != EXPECTED_PHYSICAL_OPERATION_COUNT):
        raise ValueError(f"{path}: physical operation coverage is incomplete")
    if set(workloads.values()) != set(operations):
        raise ValueError(f"{path}: semantic-to-physical mapping is incomplete")
    if set(workloads) != set(expected_keys):
        raise ValueError(f"{path}: semantic workload set differs from manifest")
    referenced_operations = set()
    for semantic_key in expected_keys:
        expected_operation_id, _, _ = _expected_operation(semantic_key)
        operation_id = workloads[semantic_key]
        if operation_id != expected_operation_id:
            raise ValueError(
                f"{path}: semantic key maps to wrong operation {semantic_key}")
        referenced_operations.add(operation_id)
    if referenced_operations != set(operations):
        raise ValueError(f"{path}: operation reference set differs")
    for operation_id in sorted(operations):
        if _OPERATION_ID_PATTERN.fullmatch(operation_id) is None:
            raise ValueError(f"{path}: invalid operation id {operation_id!r}")
        record = require_object(
            operations[operation_id], f"{path}.operations[{operation_id}]")
        require_exact_fields(
            record,
            {"operation_key", "workload", "selector_result", "projection"},
            f"{path}.operations[{operation_id}]")
        operation_key = require_nonempty_string(
            record["operation_key"],
            f"{path}.operations[{operation_id}].operation_key")
        if "sha256:" + sha256_bytes(operation_key.encode("utf-8")) != operation_id:
            raise ValueError(f"{path}: operation id/hash mismatch")
        workload = _validate_workload(
            record["workload"], operation_key,
            f"{path}.operations[{operation_id}].workload")
        result, projection = _validate_result(
            record["selector_result"], workload,
            f"{path}.operations[{operation_id}].selector_result")
        del result
        if record["projection"] != projection:
            raise ValueError(
                f"{path}: operation projection differs from raw launches")
    return dict(data), EXPECTED_SEMANTIC_KEY_COUNT


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--expected-baseline")
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    generation_arguments = (
        args.candidate, args.expected_baseline, args.models_root)
    if any(value is not None for value in generation_arguments):
        if any(value is None for value in generation_arguments):
            parser.error(
                "--candidate, --expected-baseline, and --models-root are "
                "required together")
        expanded = build(
            args.candidate, expected_baseline=args.expected_baseline,
            models_root=args.models_root)
        entries = project_metax_c500_gemm(expanded)
        evidence_sha256 = expanded_evidence_sha256(expanded)
        csv_text, manifest = build_gemm_launch_selection_artifact(
            entries,
            hardware_id="metax_c500",
            software_revision=SOFTWARE_REVISION,
            generated_by="scripts/launch_data/metax_c500_launch_generator.py",
            evidence_sha256=evidence_sha256,
            evidence_sources=[
                {"kind": "expanded_evidence", "sha256": evidence_sha256},
                {
                    "kind": "capture_candidate",
                    "sha256": expanded["candidate_sha256"],
                },
            ],
            capture_baseline_commit=expanded["capture_baseline_commit"],
            domain_sha256=GEMM_DOMAIN_SHA256,
            repository_root=ROOT,
        )
        if args.write:
            write_gemm_launch_selection_artifact(
                DATA_DIR, csv_text, manifest)
        gemm_count = manifest["row_count"]
    else:
        if args.write:
            parser.error("--write requires --candidate")
        selection = load_gemm_launch_selection_files(
            GEMM_CSV_PATH,
            GEMM_MANIFEST_PATH,
            hardware_id="metax_c500",
            software_revision=SOFTWARE_REVISION,
        )
        gemm_count = len(selection)
    print(json.dumps({"gemm": gemm_count}, sort_keys=True))
    return 0
