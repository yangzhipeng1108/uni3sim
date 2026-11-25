#!/usr/bin/env python3
"""Validate and package exact NVIDIA A800 GEMM launch observations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

from dataset_pipeline.launch_data.gemm_workload_candidates import (
    HARDWARE_IDS,
    PHYSICAL_OPERATION_FIELDS,
    build_gemm_workload_candidate,
)
from dataset_pipeline.launch_data.json_validators import (
    canonical_sequence_sha256,
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
    SERVING_MODEL_PARALLELISM as MODEL_PARALLELISM,
    SERVING_SCHEDULER_CONTRACT as EXPECTED_SCHEDULER_CONTRACT,
    gemm_physical_operation_key,
    parse_cublaslt_stages,
    reject_timing_evidence_fields,
    validate_gemm_workload_key,
)
from dataset_pipeline.launch_data.canonical_projection import (
    project_nvidia_a800_kernel_dataset_gemm,
    GEMM_DOMAIN_SHA256,
    build_gemm_launch_selection_artifact,
    expanded_evidence_sha256,
    project_nvidia_a800_gemm,
    write_gemm_launch_selection_artifact,
)
from src.kernel.launch_config.gemm_launch_selection import (
    GEMM_LAUNCH_CSV_FILENAME,
    GEMM_LAUNCH_MANIFEST_FILENAME,
    ALGORITHM_FIELDS,
    ARTIFACT_MANIFEST_FIELDS,
    ALGORITHM_INTEGER_FIELDS,
    CANONICAL_OPERATION_FIELDS,
    COVERAGE_SCOPE_FIELDS,
    CROSS_HARDWARE_FIELDS,
    DEVICE_EXECUTION_FIELDS,
    HARDWARE_DOMAIN_FIELDS,
    LIMITATION_FIELDS,
    MODEL_DOMAIN_FIELDS,
    OPERATION_FIELDS,
    POSITION_DOMAIN_FIELDS,
    SELECTION_FIELDS,
    load_gemm_launch_selection_files,
)


ROOT = Path(__file__).resolve().parents[2]
GEMM_DIR = ROOT / "launch_data" / "nvidia_a800"
GEMM_CSV_PATH = GEMM_DIR / GEMM_LAUNCH_CSV_FILENAME
GEMM_MANIFEST_PATH = GEMM_DIR / GEMM_LAUNCH_MANIFEST_FILENAME
DOMAIN_CONFIG = (
    ROOT / "configs" / "evaluation" / "experiments"
    / "gemm_launch_domain.json"
)
RUNNER = (
    ROOT / "microbench" / "nvidia_a800" / "operators" / "gemm"
    / "cublaslt_heuristic_runner.py"
)
QUERY_SOURCE = RUNNER.with_name("cublaslt_heuristic_query.cpp")

EXPECTED_SEMANTIC_KEY_COUNT = 140544
EXPECTED_PHYSICAL_OPERATION_COUNT = 140544
KERNEL_DATASET_GEMM = (
    ROOT / "kernel_dataset" / "nvidia_a800" / "gemm.json.zst"
)
EXPECTED_CUBLASLT_VERSION = 120205
EXPECTED_MAX_WORKSPACE_BYTES = 32 * 1024 * 1024
EXPECTED_SELECTOR_CONTRACT = "cublaslt_first_feasible_rank_0"
SOFTWARE_REVISION = "cuda_12.2"

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_CANDIDATE_FIELDS = {
    "schema_version", "artifact_type", "hardware_id", "baseline_commit",
    "runner_sha256", "query_source_sha256", "workload_manifest_sha256",
    "workload_manifest", "workload_key_fields", "cublaslt_version",
    "requested_devices", "device_identities", "device_telemetry",
    "max_workspace_bytes", "selector_contract", "duration_consumed",
    "matmul_executed", "production_kernel_submitted", "workload_key_count",
    "operation_count", "kernel_dataset_sha256",
    "kernel_dataset_operation_keys",
    "workloads", "operations", "limitations",
}
_QUERY_RESULT_FIELDS = {
    "schema_version", "hardware_id", "device_name", "compute_capability",
    "cublaslt_version", "operation", "max_workspace_bytes",
    "matmul_executed", "production_kernel_submitted", "dry_launch",
    "algorithms",
}
_CANONICAL_FIELDS = {
    "schema_version", "artifact_type", "hardware_id", "software_revision",
    "cublaslt_version", "source", "generated_by", "candidate_sha256",
    "capture_baseline_commit", "runner_sha256", "query_source_sha256",
    "workload_manifest_sha256", "workload_manifest",
    "workload_key_fields", "selector_contract", "duration_consumed",
    "matmul_executed", "production_kernel_submitted", "max_workspace_bytes",
    "semantic_key_count", "physical_operation_count", "workloads",
    "operations", "kernel_dataset_sha256", "kernel_dataset_selection",
}
_PROJECTION_FIELDS = {
    "execution_variant", "logical_tile_m", "logical_tile_n", "tile_k",
    "pipeline_depth", "split_k", "reduction_scheme", "workspace_bytes",
    "cta_raster",
}
_CUTLASS_KERNEL_PREFIX = "_ZN7cutlass"
_CUBLASLT_RASTER_KERNEL_PREFIX = "ampere_"
_DRY_LAUNCH_FIELDS = {
    "attempted", "replacement_kernel_submitted", "replacement_launch_count",
    "launches", "device_execution",
}
_LAUNCH_FIELDS = {
    "api", "symbol", "grid", "block", "shared_memory_bytes",
    "cta_raster_log_group", "cta_raster_serpentine",
}
_LAUNCH_APIS = {
    "cuLaunchKernel", "cuLaunchKernel_ptsz",
    "cuLaunchKernelEx", "cuLaunchKernelEx_ptsz",
}
_TELEMETRY_FIELDS = {
    "command", "idle_requirement", "preflight", "postflight",
    "postflight_attempts",
}
_TELEMETRY_RECORD_FIELDS = {
    "uuid", "name", "utilization_gpu_percent", "memory_used_mib",
}
_SELECTION_REASON = "cublaslt_rank_0_under_captured_preference"
_GEMV_KERNEL_PREFIX = "_ZN8internal5gemvx6kernelI"


def _operation_key(operation: Mapping[str, object], context: str) -> str:
    require_exact_fields(operation, OPERATION_FIELDS, context)
    m = require_integer(operation["m"], f"{context}.m", positive=True)
    n = require_integer(operation["n"], f"{context}.n", positive=True)
    k = require_integer(operation["k"], f"{context}.k", positive=True)
    dtype = operation["dtype"]
    if dtype not in ("bf16", "fp16"):
        raise ValueError(f"{context}.dtype is unsupported")
    if operation["layout"] != "row_major":
        raise ValueError(f"{context}.layout must be row_major")
    if operation["trans_a"] is not False or operation["trans_b"] is not True:
        raise ValueError(f"{context} has unsupported transpose semantics")
    has_bias = operation["has_bias"]
    if not isinstance(has_bias, bool):
        raise TypeError(f"{context}.has_bias must be boolean")
    return json.dumps(
        [m, n, k, dtype, has_bias], separators=(",", ":"))


def _validate_algorithm(
        value: object, expected_rank: int, max_workspace_bytes: int,
        context: str) -> Mapping[str, object]:
    algorithm = require_object(value, context)
    require_exact_fields(algorithm, ALGORITHM_FIELDS, context)
    for field in ALGORITHM_INTEGER_FIELDS:
        require_integer(algorithm[field], f"{context}.{field}")
    if algorithm["rank"] != expected_rank:
        raise ValueError(f"{context}.rank is not contiguous")
    if algorithm["workspace_bytes"] > max_workspace_bytes:
        raise ValueError(f"{context} exceeds the workspace limit")
    for field in ("tile", "stages"):
        item = algorithm[field]
        if not isinstance(item, str) or not item:
            raise ValueError(f"{context}.{field} must be a non-empty string")
    waves_count = algorithm["waves_count"]
    if (isinstance(waves_count, bool)
            or not isinstance(waves_count, (int, float))
            or not math.isfinite(waves_count) or waves_count < 0):
        raise ValueError(f"{context}.waves_count must be finite and non-negative")
    return algorithm


def _validate_launch(value: object, context: str) -> dict[str, object]:
    launch = require_object(value, context)
    require_exact_fields(launch, _LAUNCH_FIELDS, context)
    for field in ("api", "symbol"):
        item = launch[field]
        if not isinstance(item, str) or not item:
            raise ValueError(f"{context}.{field} must be non-empty")
    if launch["api"] not in _LAUNCH_APIS:
        raise ValueError(f"{context}.api is unsupported")
    if "midlayer_noop" in launch["symbol"]:
        raise ValueError(f"{context} records the replacement no-op")
    normalized = {}
    for field in ("grid", "block"):
        dimensions = require_array(launch[field], f"{context}.{field}")
        if len(dimensions) != 3:
            raise ValueError(f"{context}.{field} must have three dimensions")
        normalized[field] = [
            require_integer(item, f"{context}.{field}[{index}]", positive=True)
            for index, item in enumerate(dimensions)
        ]
    shared_memory = require_integer(
        launch["shared_memory_bytes"], f"{context}.shared_memory_bytes")
    raster_declared = launch["symbol"].startswith(
        _CUBLASLT_RASTER_KERNEL_PREFIX) and "gemm" in launch["symbol"]
    log_group = launch["cta_raster_log_group"]
    serpentine = launch["cta_raster_serpentine"]
    if not raster_declared:
        if log_group is not None or serpentine is not None:
            raise ValueError(
                f"{context} reports a raster word for a kernel family whose "
                f"parameter layout does not define one")
    else:
        log_group = require_integer(log_group, f"{context}.cta_raster_log_group")
        serpentine = require_integer(serpentine, f"{context}.cta_raster_serpentine")
        if log_group < 0 or log_group > 15:
            raise ValueError(f"{context}.cta_raster_log_group is out of range")
        if serpentine not in (0, 1):
            raise ValueError(f"{context}.cta_raster_serpentine is not a flag")
    return {
        "api": launch["api"],
        "symbol": launch["symbol"],
        "grid": normalized["grid"],
        "block": normalized["block"],
        "shared_memory_bytes": shared_memory,
        "cta_raster_log_group": log_group,
        "cta_raster_serpentine": serpentine,
    }


def _cta_raster_projection(
        execution_variant: str, main_launch: Mapping[str, object],
        operation: Mapping[str, object], logical_tile_m: int | None,
        logical_tile_n: int | None, split_k: int,
        context: str) -> dict[str, object]:
    """Project the CTA-to-tile raster program of one rank-0 launch.

    Two kernel families reach this launch domain and each states its map
    statically.  A cuBLASLt ``ampere_*gemm*`` kernel reads a raster word from
    its own parameter block and keeps a grid that is one CTA per output tile
    and K slice.  A CUTLASS kernel instead encodes its grouped map in the grid
    itself, widening the M axis by the group and folding N by the same factor.
    Anything else keeps no raster program and rasters its grid directly.
    """

    if execution_variant == "gemv":
        return {
            "convention": "none", "log_group": 0, "serpentine": False,
            "k_slices": split_k,
        }
    if execution_variant != "gemm":
        raise ValueError(
            f"{context} has an unsupported execution variant for a raster")
    grid_x, grid_y, grid_z = main_launch["grid"]
    m_tiles = -(-operation["m"] // logical_tile_m)
    n_tiles = -(-operation["n"] // logical_tile_n)
    if grid_z > split_k:
        raise ValueError(f"{context} grid Z exceeds the retained K split")
    symbol = main_launch["symbol"]
    if main_launch["cta_raster_log_group"] is not None:
        if (grid_x, grid_y) != (n_tiles, m_tiles):
            raise ValueError(
                f"{context} cuBLASLt grid is not one CTA per output tile")
        if grid_z != split_k:
            raise ValueError(
                f"{context} cuBLASLt grid Z differs from its K split")
        return {
            "convention": "cublaslt_m_group",
            "log_group": main_launch["cta_raster_log_group"],
            "serpentine": bool(main_launch["cta_raster_serpentine"]),
            "k_slices": grid_z,
        }
    if symbol.startswith(_CUTLASS_KERNEL_PREFIX):
        if grid_x % m_tiles:
            raise ValueError(
                f"{context} CUTLASS grid X is not a multiple of its M tiles")
        group = grid_x // m_tiles
        if group & (group - 1):
            raise ValueError(
                f"{context} CUTLASS raster group is not a power of two")
        if grid_y != -(-n_tiles // group):
            raise ValueError(
                f"{context} CUTLASS grid Y differs from its folded N tiles")
        return {
            "convention": "cutlass_n_group",
            "log_group": group.bit_length() - 1,
            "serpentine": False,
            "k_slices": grid_z,
        }
    # A kernel family whose grid is neither one CTA per output tile nor the
    # CUTLASS grouped encoding states no raster map this projection can read.
    # It keeps no raster program and its traffic stays on the schedule-free
    # envelope rather than on an invented traversal.
    if (grid_x, grid_y) != (n_tiles, m_tiles) and grid_x != n_tiles * m_tiles:
        return {
            "convention": "none", "log_group": 0, "serpentine": False,
            "k_slices": split_k,
        }
    return {
        "convention": "none", "log_group": 0, "serpentine": False,
        "k_slices": grid_z,
    }


def _validate_device_execution(
        value: object, replacement_count: int, context: str,
        ) -> dict[str, object]:
    evidence = require_object(value, context)
    require_exact_fields(evidence, DEVICE_EXECUTION_FIELDS, context)
    if (evidence["evidence_kind"] != "cupti_activity_kernel"
            or evidence["evidence_version"] != 1
            or evidence["all_replacements_observed"] is not True):
        raise ValueError(f"{context} lacks affirmative CUPTI evidence")
    activity_count = require_integer(
        evidence["activity_record_count"], f"{context}.activity_record_count")
    dropped_count = require_integer(
        evidence["dropped_record_count"], f"{context}.dropped_record_count")
    symbols = require_array(
        evidence["executed_symbols"], f"{context}.executed_symbols")
    if (dropped_count != 0 or activity_count != replacement_count
            or len(symbols) != activity_count
            or any(symbol != "midlayer_noop" for symbol in symbols)):
        raise ValueError(f"{context} does not prove every no-op execution")
    return {
        "evidence_kind": "cupti_activity_kernel",
        "evidence_version": 1,
        "activity_record_count": activity_count,
        "executed_symbols": list(symbols),
        "all_replacements_observed": True,
        "dropped_record_count": 0,
    }


def _validate_embedded_manifest(
        value: object, workload_keys: set[str], context: str,
        ) -> Mapping[str, object]:
    manifest = require_object(value, context)
    require_exact_fields(manifest, ARTIFACT_MANIFEST_FIELDS, context)
    if (manifest["schema_version"] != 2
            or manifest["artifact_type"] != "gemm_workload_candidate"
            or manifest["source_config"]
            != str(DOMAIN_CONFIG.relative_to(ROOT))
            or manifest["duration_consumed"] is not False):
        raise ValueError(f"{context} identity differs")
    require_sha256_hex(
        manifest["source_config_sha256"], f"{context}.source_config_sha256")
    if tuple(require_array(
            manifest["workload_key_fields"],
            f"{context}.workload_key_fields")) != GEMM_WORKLOAD_KEY_FIELDS:
        raise ValueError(f"{context} workload key fields differ")
    if tuple(require_array(
            manifest["physical_operation_fields"],
            f"{context}.physical_operation_fields")) != (
                PHYSICAL_OPERATION_FIELDS):
        raise ValueError(f"{context} physical operation fields differ")
    scheduler = require_object(
        manifest["scheduler_contract"], f"{context}.scheduler_contract")
    require_exact_fields(
        scheduler, set(EXPECTED_SCHEDULER_CONTRACT),
        f"{context}.scheduler_contract")
    if scheduler != EXPECTED_SCHEDULER_CONTRACT:
        raise ValueError(f"{context} scheduler contract differs")
    coverage = require_object(
        manifest["coverage_scope"], f"{context}.coverage_scope")
    require_exact_fields(coverage, COVERAGE_SCOPE_FIELDS, f"{context}.coverage_scope")
    if coverage["kind"] != "static_complete_legal_launch_selector_domain":
        raise ValueError(f"{context} coverage kind differs")
    for field in COVERAGE_SCOPE_FIELDS - {"kind"}:
        if not isinstance(coverage[field], bool):
            raise TypeError(f"{context}.coverage_scope.{field} must be boolean")
    if any(coverage[field] is not False for field in (
            "request_trajectory_consumed", "scheduler_alignment_consumed",
            "operator_trace_consumed", "timing_consumed",
            "fused_moe_included",
            "serving_end_to_end_executability_claimed")):
        raise ValueError(f"{context} coverage boundary differs")

    expected_models = set(MODEL_PARALLELISM)
    inventory = require_object(
        manifest["dataset_model_inventory"],
        f"{context}.dataset_model_inventory")
    if set(inventory) != set(HARDWARE_IDS):
        raise ValueError(f"{context} hardware inventory differs")
    for hardware_id in HARDWARE_IDS:
        models = require_array(
            inventory[hardware_id], f"{context}.inventory.{hardware_id}")
        if (models != sorted(models) or set(models) != expected_models
                or len(models) != len(expected_models)):
            raise ValueError(
                f"{context} model inventory for {hardware_id} differs")

    model_domains = require_array(
        manifest["model_domains"], f"{context}.model_domains")
    if len(model_domains) != len(expected_models):
        raise ValueError(f"{context} model domain count differs")
    seen_models = set()
    expected_model_types = {
        "Qwen3-4B": "qwen3",
        "Qwen3-8B": "qwen3",
        "Llama-3.1-8B": "llama",
        "Qwen3-32B": "qwen3",
        "Qwen3-30B-A3B": "qwen3_moe",
    }
    for index, item in enumerate(model_domains):
        domain = require_object(item, f"{context}.model_domains[{index}]")
        require_exact_fields(
            domain, MODEL_DOMAIN_FIELDS,
            f"{context}.model_domains[{index}]")
        model_id = require_nonempty_string(
            domain["model_id"], f"{context}.model_domains[{index}].model_id")
        if (model_id not in expected_models or model_id in seen_models
                or domain["model_type"] != expected_model_types[model_id]):
            raise ValueError(f"{context} model domain identity differs")
        seen_models.add(model_id)
        require_nonempty_string(
            domain["model_path"], f"{context}.model_domains[{index}].model_path")
        require_sha256_hex(
            domain["model_config_sha256"],
            f"{context}.model_domains[{index}].model_config_sha256")
        expected_tp, expected_ep = MODEL_PARALLELISM[model_id]
        if (domain["tensor_parallel_size"] != expected_tp
                or domain["expert_parallel_size"] != expected_ep):
            raise ValueError(f"{context} model parallelism differs")
        raw_positions = require_array(
            domain["positions"], f"{context}.model_domains[{index}].positions")
        if domain["model_type"] == "qwen3_moe":
            expected_positions = ("qkv_proj", "o_proj", "router", "lm_head")
        elif domain["model_type"] in ("qwen3", "llama"):
            expected_positions = (
                "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head")
        else:
            raise AssertionError(
                f"unhandled model type {domain['model_type']!r}")
        position_names = []
        position_key_count = 0
        for position_index, item in enumerate(raw_positions):
            position_context = (
                f"{context}.model_domains[{index}].positions[{position_index}]")
            position = require_object(item, position_context)
            require_exact_fields(position, POSITION_DOMAIN_FIELDS, position_context)
            name = require_nonempty_string(
                position["position"], f"{position_context}.position")
            position_names.append(name)
            if name == "lm_head":
                expected_m_max = 256
                expected_variant = "m1_gemv_else_gemm"
            elif name in expected_positions:
                expected_m_max = 8192
                expected_variant = "gemm"
            else:
                raise ValueError(f"{context} model position is unknown")
            if (position["m_min"] != 1
                    or position["m_max"] != expected_m_max
                    or position["semantic_key_count"] != expected_m_max
                    or position["dtype"] != "bf16"
                    or position["has_bias"] is not False
                    or position["variant_contract"] != expected_variant):
                raise ValueError(f"{context} model position contract differs")
            require_integer(position["n"], f"{position_context}.n", positive=True)
            require_integer(position["k"], f"{position_context}.k", positive=True)
            position_key_count += position["semantic_key_count"]
        if tuple(position_names) != expected_positions:
            raise ValueError(f"{context} model positions are invalid")
        semantic_key_count = require_integer(
            domain["semantic_key_count"],
            f"{context}.model_domains[{index}].semantic_key_count",
            positive=True)
        physical_operation_count = require_integer(
            domain["physical_operation_count"],
            f"{context}.model_domains[{index}].physical_operation_count",
            positive=True)
        if (semantic_key_count != position_key_count
                or physical_operation_count > semantic_key_count):
            raise ValueError(f"{context} model domain counts differ")
    if seen_models != expected_models:
        raise ValueError(f"{context} model domain set differs")

    limitations = require_array(
        manifest["limitations"], f"{context}.limitations")
    for index, item in enumerate(limitations):
        limitation = require_object(item, f"{context}.limitations[{index}]")
        require_exact_fields(
            limitation, LIMITATION_FIELDS,
            f"{context}.limitations[{index}]")
        require_nonempty_string(
            limitation["scope"], f"{context}.limitations[{index}].scope")
        require_nonempty_string(
            limitation["reason"], f"{context}.limitations[{index}].reason")
        affected_hardware = require_array(
            limitation["affected_hardware_ids"],
            f"{context}.limitations[{index}].affected_hardware_ids")
        affected_models = require_array(
            limitation["affected_model_ids"],
            f"{context}.limitations[{index}].affected_model_ids")
        if (not affected_hardware
                or any(item not in HARDWARE_IDS for item in affected_hardware)
                or len(set(affected_hardware)) != len(affected_hardware)
                or not affected_models
                or any(item not in expected_models for item in affected_models)
                or len(set(affected_models)) != len(affected_models)):
            raise ValueError(f"{context} limitation scope is invalid")

    keys = require_array(manifest["workload_keys"], f"{context}.workload_keys")
    if (keys != sorted(keys) or len(set(keys)) != len(keys)
            or set(keys) != workload_keys):
        raise ValueError(f"{context} workload key set differs")
    for key in keys:
        validate_gemm_workload_key(key)
    if (manifest["semantic_key_count"] != EXPECTED_SEMANTIC_KEY_COUNT
            or len(keys) != EXPECTED_SEMANTIC_KEY_COUNT
            or manifest["physical_operation_count"]
            != EXPECTED_PHYSICAL_OPERATION_COUNT):
        raise ValueError(f"{context} coverage counts differ")
    domain_sha256 = canonical_sequence_sha256(keys)
    domains = require_object(
        manifest["hardware_domains"], f"{context}.hardware_domains")
    if set(domains) != set(HARDWARE_IDS):
        raise ValueError(f"{context} hardware domains differ")
    for hardware_id in HARDWARE_IDS:
        domain = require_object(
            domains[hardware_id], f"{context}.hardware_domains.{hardware_id}")
        require_exact_fields(
            domain, HARDWARE_DOMAIN_FIELDS,
            f"{context}.hardware_domains.{hardware_id}")
        if domain != {
                "status": "exact",
                "domain_sha256": domain_sha256,
                "semantic_key_count": EXPECTED_SEMANTIC_KEY_COUNT,
                "physical_operation_count": EXPECTED_PHYSICAL_OPERATION_COUNT,
        }:
            raise ValueError(f"{context} hardware domain {hardware_id} differs")
    cross_hardware = require_object(
        manifest["cross_hardware"], f"{context}.cross_hardware")
    require_exact_fields(
        cross_hardware, CROSS_HARDWARE_FIELDS,
        f"{context}.cross_hardware")
    if cross_hardware != {
            "status": "exact_equal",
            "hardware_ids": list(HARDWARE_IDS),
            "domain_sha256": domain_sha256,
            "semantic_key_count": EXPECTED_SEMANTIC_KEY_COUNT,
            "physical_operation_count": EXPECTED_PHYSICAL_OPERATION_COUNT,
    }:
        raise ValueError(f"{context} cross-hardware contract differs")
    return manifest


def _select_algorithm(
        algorithms: Sequence[Mapping[str, object]], context: str,
        ) -> tuple[Mapping[str, object], int, int | None]:
    algorithm = algorithms[0]
    undefined_tile = (
        algorithm["tile"] == "undefined"
        and algorithm["stages"] == "undefined")
    if undefined_tile:
        if (algorithm["tile_id"] != 0
                or algorithm["logical_tile_m"] != 0
                or algorithm["logical_tile_n"] != 0
                or algorithm["stages_id"] != 0):
            raise ValueError(
                f"{context} undefined rank-0 tile metadata is inconsistent")
        tile_k = None
        pipeline_depth = None
    else:
        if "undefined" in (algorithm["tile"], algorithm["stages"]):
            raise ValueError(
                f"{context} rank-0 tile metadata is only partly defined")
        tile_k, pipeline_depth = parse_cublaslt_stages(algorithm["stages"])
        if (algorithm["logical_tile_m"] <= 0
                or algorithm["logical_tile_n"] <= 0):
            raise ValueError(f"{context} rank-0 tile is incomplete")
    if algorithm["split_k"] <= 0:
        raise ValueError(f"{context} rank-0 heuristic is incomplete")
    split_k = algorithm["split_k"]
    reduction_scheme = algorithm["reduction_scheme"]
    if (split_k == 1 and reduction_scheme != 0) or (
            split_k > 1 and reduction_scheme not in (1, 2, 4)):
        raise ValueError(
            f"{context} rank-0 split/reduction contract is inconsistent")
    return algorithm, tile_k, pipeline_depth


def _execution_variant(
        operation: Mapping[str, object], algorithm: Mapping[str, object],
        tile_k: int | None, main_launch: Mapping[str, object],
        context: str) -> str:
    is_gemv = main_launch["symbol"].startswith(_GEMV_KERNEL_PREFIX)
    if tile_k is None:
        if (operation["m"] != 1 or not is_gemv
                or algorithm["split_k"] != 1
                or algorithm["reduction_scheme"] != 0):
            raise NotImplementedError(
                f"{context} has an unsupported undefined-tile rank-0 launch")
        return "gemv"
    if is_gemv:
        raise ValueError(f"{context} GEMV symbol unexpectedly declares a tile")
    return "gemm"


def _logical_tile_projection(
        execution_variant: str,
        algorithm: Mapping[str, object],
        ) -> tuple[int | None, int | None]:
    if execution_variant == "gemv":
        return None, None
    if execution_variant == "gemm":
        return algorithm["logical_tile_m"], algorithm["logical_tile_n"]
    raise ValueError(
        f"unsupported NVIDIA GEMM execution variant: {execution_variant!r}")


def _validate_canonical_operation(
        operation_key: str, value: object, max_workspace_bytes: int,
        context: str) -> None:
    record = require_object(value, context)
    require_exact_fields(record, CANONICAL_OPERATION_FIELDS, context)
    operation = require_object(record["operation"], f"{context}.operation")
    if _operation_key(operation, f"{context}.operation") != operation_key:
        raise ValueError(f"{context} payload identity differs")
    queried_device = require_nonempty_string(
        record["queried_device"], f"{context}.queried_device")
    if not queried_device.isdigit() or str(int(queried_device)) != queried_device:
        raise ValueError(f"{context}.queried_device is not canonical")
    raw_algorithms = require_array(record["algorithms"], f"{context}.algorithms")
    if not raw_algorithms or len(raw_algorithms) >= 64:
        raise ValueError(f"{context}.algorithms must contain 1..63 entries")
    algorithms = [
        _validate_algorithm(
            item, rank, max_workspace_bytes,
            f"{context}.algorithms[{rank}]")
        for rank, item in enumerate(raw_algorithms)
    ]
    selection = require_object(record["selection"], f"{context}.selection")
    require_exact_fields(selection, SELECTION_FIELDS, f"{context}.selection")
    if selection != {
            "contract": "vendor_rank_0_heuristic",
            "selected_rank": 0,
            "reason": _SELECTION_REASON,
    }:
        raise ValueError(f"{context}.selection differs")
    selected, tile_k, pipeline_depth = _select_algorithm(algorithms, context)
    projection = require_object(record["projection"], f"{context}.projection")
    require_exact_fields(projection, _PROJECTION_FIELDS, f"{context}.projection")
    main_launch = _validate_launch(record["main_launch"], f"{context}.main_launch")
    execution_variant = _execution_variant(
        operation, selected, tile_k, main_launch, context)
    logical_tile_m, logical_tile_n = _logical_tile_projection(
        execution_variant, selected)
    expected_projection = {
        "execution_variant": execution_variant,
        "logical_tile_m": logical_tile_m,
        "logical_tile_n": logical_tile_n,
        "tile_k": tile_k,
        "pipeline_depth": pipeline_depth,
        "split_k": selected["split_k"],
        "reduction_scheme": selected["reduction_scheme"],
        "workspace_bytes": selected["workspace_bytes"],
        "cta_raster": _cta_raster_projection(
            execution_variant, main_launch, operation, logical_tile_m,
            logical_tile_n, selected["split_k"], context),
    }
    if projection != expected_projection:
        raise ValueError(f"{context}.projection differs from rank-0 launch")
    reducer_value = record["reducer_launch"]
    reducer_launch = None if reducer_value is None else _validate_launch(
        reducer_value, f"{context}.reducer_launch")
    if selected["split_k"] == 1:
        if selected["reduction_scheme"] != 0 or reducer_launch is not None:
            raise ValueError(f"{context} non-split launch contract differs")
    elif selected["reduction_scheme"] == 1:
        if reducer_launch is not None:
            raise ValueError(f"{context} in-place split launch has a reducer")
    elif selected["reduction_scheme"] in (2, 4):
        if (reducer_launch is None
                or "splitKreduce_kernel" not in reducer_launch["symbol"]):
            raise ValueError(f"{context} separate reducer facts differ")
    else:
        raise ValueError(f"{context} split/reduction contract differs")
    _validate_device_execution(
        record["device_execution"],
        1 + (1 if reducer_launch is not None else 0),
        f"{context}.device_execution")
    if selected["reduction_scheme"] == 2:
        partial_element_bytes = 4
    elif selected["reduction_scheme"] == 4:
        partial_element_bytes = 2
    else:
        partial_element_bytes = 0
    required_workspace = (
        main_launch["grid"][2] * operation["m"] * operation["n"]
        * partial_element_bytes)
    if selected["workspace_bytes"] < required_workspace:
        raise ValueError(f"{context} observed-partial workspace is insufficient")


def _validate_manifest(
        manifest: Mapping[str, object], models_root: Path,
        ) -> Mapping[str, object]:
    if manifest["source_config"] != str(DOMAIN_CONFIG.relative_to(ROOT)):
        raise ValueError("candidate workload manifest targets an unknown config")
    rebuilt = build_gemm_workload_candidate(
        ROOT, DOMAIN_CONFIG, models_root=models_root)
    if manifest != rebuilt:
        raise ValueError(
            "candidate workload manifest does not match the current exact "
            "ShareGPT launch domain")
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
        manifest["workload_key_fields"], "workload_manifest.workload_key_fields")
    if tuple(fields) != GEMM_WORKLOAD_KEY_FIELDS:
        raise ValueError("workload manifest does not use the 16-field key")
    domains = require_object(
        manifest["hardware_domains"], "workload_manifest.hardware_domains")
    domain = require_object(
        domains["nvidia_a800"],
        "workload_manifest.hardware_domains.nvidia_a800")
    if domain["status"] != "exact":
        raise ValueError("NVIDIA A800 workload domain is not exact")
    if domain["semantic_key_count"] != EXPECTED_SEMANTIC_KEY_COUNT:
        raise ValueError("NVIDIA A800 workload domain key count is incomplete")
    if (domain["physical_operation_count"]
            != EXPECTED_PHYSICAL_OPERATION_COUNT):
        raise ValueError(
            "NVIDIA A800 workload domain operation count is incomplete")
    return rebuilt


def _validate_telemetry(
        value: object, requested_devices: Sequence[object],
        identities: Mapping[str, object]) -> None:
    telemetry = require_object(value, "device_telemetry")
    require_exact_fields(telemetry, _TELEMETRY_FIELDS, "device_telemetry")
    if telemetry["command"] != [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
    ]:
        raise ValueError("candidate used an unexpected telemetry command")
    if telemetry["idle_requirement"] != {
            "utilization_gpu_percent": 0,
            "memory_used_mib": 0,
    }:
        raise ValueError("candidate used an unexpected idle requirement")

    def validate_snapshot(snapshot_value: object, context: str, *, idle: bool):
        snapshot = require_object(snapshot_value, context)
        if set(snapshot) != set(requested_devices):
            raise ValueError(f"{context} does not cover requested devices")
        for device in requested_devices:
            record = require_object(snapshot[device], f"{context}.{device}")
            require_exact_fields(
                record, _TELEMETRY_RECORD_FIELDS, f"{context}.{device}")
            identity = require_object(
                identities[device], f"device_identities.{device}")
            if (record["uuid"] != identity["uuid"]
                    or record["name"] != identity["device_name"]):
                raise ValueError(f"{context}.{device} identity differs")
            utilization = require_integer(
                record["utilization_gpu_percent"],
                f"{context}.{device}.utilization_gpu_percent")
            memory = require_integer(
                record["memory_used_mib"],
                f"{context}.{device}.memory_used_mib")
            if idle and (utilization != 0 or memory != 0):
                raise ValueError(f"{context}.{device} is not idle")

    validate_snapshot(telemetry["preflight"], "device_telemetry.preflight", idle=True)
    validate_snapshot(
        telemetry["postflight"], "device_telemetry.postflight", idle=True)
    attempts = require_array(
        telemetry["postflight_attempts"], "device_telemetry.postflight_attempts")
    if not attempts:
        raise ValueError("candidate has no postflight telemetry attempts")
    for index, snapshot in enumerate(attempts):
        validate_snapshot(
            snapshot, f"device_telemetry.postflight_attempts[{index}]",
            idle=index + 1 == len(attempts))


def build(
        candidate_path: Path, *, expected_baseline: str,
        models_root: Path) -> dict[str, object]:
    """Build canonical launch data from one exact allocation-free capture."""

    candidate_path = candidate_path.resolve()
    candidate_bytes = candidate_path.read_bytes()
    candidate_sha256 = sha256_bytes(candidate_bytes)
    candidate = require_object(json.loads(candidate_bytes), str(candidate_path))
    reject_timing_evidence_fields(candidate, "candidate")
    require_exact_fields(candidate, _CANDIDATE_FIELDS, "candidate")
    if candidate["schema_version"] != 2:
        raise ValueError("candidate must use schema_version 2")
    if candidate["artifact_type"] != "nvidia_a800_gemm_launch_candidate":
        raise ValueError("candidate has the wrong artifact_type")
    if candidate["hardware_id"] != "nvidia_a800":
        raise ValueError("candidate has the wrong hardware_id")
    if (_COMMIT_PATTERN.fullmatch(expected_baseline) is None
            or candidate["baseline_commit"] != expected_baseline):
        raise ValueError("candidate baseline_commit does not match expectation")
    for field in (
            "runner_sha256", "query_source_sha256",
            "workload_manifest_sha256"):
        require_sha256_hex(candidate[field], f"candidate.{field}")
    if candidate["runner_sha256"] != sha256_bytes(RUNNER.read_bytes()):
        raise ValueError("candidate was not produced by the current runner")
    if candidate["query_source_sha256"] != sha256_bytes(
            QUERY_SOURCE.read_bytes()):
        raise ValueError("candidate was not produced by the current query source")
    if candidate["cublaslt_version"] != EXPECTED_CUBLASLT_VERSION:
        raise ValueError("candidate has the wrong cuBLASLt software version")
    if candidate["max_workspace_bytes"] != EXPECTED_MAX_WORKSPACE_BYTES:
        raise ValueError("candidate has the wrong workspace preference")
    if candidate["selector_contract"] != EXPECTED_SELECTOR_CONTRACT:
        raise ValueError("candidate has the wrong selector contract")
    if candidate["duration_consumed"] is not False:
        raise ValueError("candidate consumed duration evidence")
    if candidate["matmul_executed"] is not False:
        raise ValueError("candidate executed a matrix multiplication")
    if candidate["production_kernel_submitted"] is not False:
        raise ValueError("candidate submitted a production kernel")

    manifest = require_object(candidate["workload_manifest"], "workload_manifest")
    rebuilt_manifest = _validate_manifest(manifest, models_root)
    rebuilt_manifest_bytes = (
        json.dumps(rebuilt_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if candidate["workload_manifest_sha256"] != sha256_bytes(
            rebuilt_manifest_bytes):
        raise ValueError(
            "candidate workload_manifest_sha256 does not match the exact "
            "generated manifest")
    if tuple(require_array(
            candidate["workload_key_fields"], "workload_key_fields")) != (
                GEMM_WORKLOAD_KEY_FIELDS):
        raise ValueError("candidate does not use the 16-field workload key")
    expected_keys = require_array(
        manifest["workload_keys"], "workload_manifest.workload_keys")
    if len(expected_keys) != EXPECTED_SEMANTIC_KEY_COUNT:
        raise ValueError("workload manifest key array is incomplete")
    if candidate["workload_key_count"] != EXPECTED_SEMANTIC_KEY_COUNT:
        raise ValueError("candidate workload key count is incomplete")
    kernel_dataset_selection, kernel_dataset_sha256 = (
        _kernel_dataset_gemm_domain())
    if candidate["kernel_dataset_sha256"] != kernel_dataset_sha256:
        raise ValueError(
            "candidate kernel dataset digest differs from the canonical "
            "kernel dataset it claims to cover")
    kernel_dataset_operation_keys = sorted(
        {key for key, _selection in kernel_dataset_selection})
    if sorted(require_array(
            candidate["kernel_dataset_operation_keys"],
            "kernel_dataset_operation_keys")) != kernel_dataset_operation_keys:
        raise ValueError(
            "candidate kernel dataset operations differ from the dataset")
    expected_operation_count = len(
        set(kernel_dataset_operation_keys)
        | {gemm_physical_operation_key(key) for key in expected_keys})
    if candidate["operation_count"] != expected_operation_count:
        raise ValueError("candidate operation count is incomplete")

    requested_devices = require_array(
        candidate["requested_devices"], "requested_devices")
    if (not requested_devices
            or any(not isinstance(device, str) or not device.isdigit()
                   or str(int(device)) != device
                   for device in requested_devices)
            or len(set(requested_devices)) != len(requested_devices)):
        raise ValueError("candidate requested_devices are invalid")
    identities = require_object(candidate["device_identities"], "device_identities")
    if set(identities) != set(requested_devices):
        raise ValueError("candidate device identities are incomplete")
    for device in requested_devices:
        identity = require_object(identities[device], f"device_identities.{device}")
        require_exact_fields(
            identity, {"uuid", "device_name", "compute_capability"},
            f"device_identities.{device}")
        if (not isinstance(identity["uuid"], str) or not identity["uuid"]
                or not isinstance(identity["device_name"], str)
                or not identity["device_name"].startswith("NVIDIA A800")
                or identity["compute_capability"] != "8.0"):
            raise ValueError(f"device identity {device!r} is not an A800 sm80")
    _validate_telemetry(
        candidate["device_telemetry"], requested_devices, identities)
    limitations = require_array(candidate["limitations"], "limitations")
    if (not limitations
            or any(not isinstance(item, str) or not item for item in limitations)):
        raise ValueError("candidate limitations must be non-empty strings")

    workloads = require_object(candidate["workloads"], "workloads")
    if set(workloads) != set(expected_keys):
        raise ValueError("candidate workloads do not cover the exact semantic set")
    operations = require_object(candidate["operations"], "operations")
    if len(operations) != expected_operation_count:
        raise ValueError("candidate operations do not cover the exact domain")
    referenced_operations = set()
    canonical_workloads = {}
    for semantic_key in expected_keys:
        validate_gemm_workload_key(semantic_key)
        workload = require_object(workloads[semantic_key], f"workloads[{semantic_key}]")
        require_exact_fields(
            workload, {"operation_key", "result_key"},
            f"workloads[{semantic_key}]")
        operation_key = workload["operation_key"]
        if (not isinstance(operation_key, str)
                or workload["result_key"] != operation_key):
            raise ValueError(
                f"workloads[{semantic_key}] has inconsistent result identity")
        expected_operation_key = gemm_physical_operation_key(semantic_key)
        if operation_key != expected_operation_key:
            raise ValueError(
                f"workloads[{semantic_key}] maps to the wrong physical operation")
        referenced_operations.add(operation_key)
        canonical_workloads[semantic_key] = operation_key
    if referenced_operations | set(kernel_dataset_operation_keys) != set(
            operations):
        raise ValueError("candidate contains unmapped or missing operations")

    canonical_operations = {}
    for operation_key in sorted(operations):
        record = require_object(operations[operation_key], f"operations[{operation_key}]")
        require_exact_fields(record, {"device", "result"}, f"operations[{operation_key}]")
        device = record["device"]
        if device not in requested_devices:
            raise ValueError(f"operations[{operation_key}] used an unknown device")
        result = require_object(record["result"], f"operations[{operation_key}].result")
        require_exact_fields(result, _QUERY_RESULT_FIELDS, f"operations[{operation_key}].result")
        if (result["schema_version"] != 2
                or result["hardware_id"] != "nvidia_a800"
                or result["device_name"] != identities[device]["device_name"]
                or result["compute_capability"] != "8.0"
                or result["cublaslt_version"] != EXPECTED_CUBLASLT_VERSION
                or result["max_workspace_bytes"]
                != EXPECTED_MAX_WORKSPACE_BYTES):
            raise ValueError(f"operations[{operation_key}] software identity differs")
        if result["matmul_executed"] is not False:
            raise ValueError(f"operations[{operation_key}] executed matmul")
        if result["production_kernel_submitted"] is not False:
            raise ValueError(
                f"operations[{operation_key}] submitted a production kernel")
        operation = require_object(
            result["operation"], f"operations[{operation_key}].result.operation")
        if _operation_key(operation, f"operations[{operation_key}].operation") != (
                operation_key):
            raise ValueError(f"operations[{operation_key}] payload identity differs")
        raw_algorithms = require_array(
            result["algorithms"], f"operations[{operation_key}].algorithms")
        if not raw_algorithms or len(raw_algorithms) >= 64:
            raise ValueError(
                f"operations[{operation_key}] must contain 1..63 heuristics")
        algorithms = [
            _validate_algorithm(
                value, rank, EXPECTED_MAX_WORKSPACE_BYTES,
                f"operations[{operation_key}].algorithms[{rank}]")
            for rank, value in enumerate(raw_algorithms)
        ]
        selected, tile_k, pipeline_depth = _select_algorithm(
            algorithms, f"operations[{operation_key}]")
        dry_launch = require_object(
            result["dry_launch"], f"operations[{operation_key}].dry_launch")
        require_exact_fields(
            dry_launch, _DRY_LAUNCH_FIELDS,
            f"operations[{operation_key}].dry_launch")
        attempted = dry_launch["attempted"]
        replacement_submitted = dry_launch["replacement_kernel_submitted"]
        if not isinstance(attempted, bool) or not isinstance(
                replacement_submitted, bool):
            raise TypeError(
                f"operations[{operation_key}] dry-launch flags must be boolean")
        replacement_count = require_integer(
            dry_launch["replacement_launch_count"],
            f"operations[{operation_key}].replacement_launch_count")
        launches = [
            _validate_launch(
                launch, f"operations[{operation_key}].launches[{index}]")
            for index, launch in enumerate(require_array(
                dry_launch["launches"],
                f"operations[{operation_key}].dry_launch.launches"))
        ]
        if attempted is not True or replacement_submitted is not True:
            raise ValueError(
                f"operations[{operation_key}] did not dry-launch rank-0")
        if replacement_count != len(launches):
            raise ValueError(
                f"operations[{operation_key}] did not replace every launch")
        device_execution = _validate_device_execution(
            dry_launch["device_execution"], replacement_count,
            f"operations[{operation_key}].device_execution")
        reducers = [
            launch for launch in launches
            if "splitKreduce_kernel" in launch["symbol"]
        ]
        if selected["split_k"] == 1:
            if len(launches) != 1 or reducers:
                raise ValueError(
                    f"operations[{operation_key}] non-split launch differs")
            main_launch = launches[0]
            reducer_launch = None
        elif selected["reduction_scheme"] == 1:
            if len(launches) != 1 or reducers:
                raise ValueError(
                    f"operations[{operation_key}] in-place split launch differs")
            main_launch = launches[0]
            reducer_launch = None
        elif selected["reduction_scheme"] in (2, 4):
            if (len(launches) != 2 or len(reducers) != 1
                    or launches[-1] != reducers[0]):
                raise ValueError(
                    f"operations[{operation_key}] separate reducer differs")
            main_launch = launches[0]
            reducer_launch = launches[1]
        else:
            raise AssertionError("unhandled rank-0 reduction scheme")
        if selected["reduction_scheme"] == 2:
            partial_element_bytes = 4
        elif selected["reduction_scheme"] == 4:
            partial_element_bytes = 2
        else:
            partial_element_bytes = 0
        execution_variant = _execution_variant(
            operation, selected, tile_k, main_launch,
            f"operations[{operation_key}]")
        logical_tile_m, logical_tile_n = _logical_tile_projection(
            execution_variant, selected)
        required_partial_workspace = (
            main_launch["grid"][2] * operation["m"] * operation["n"]
            * partial_element_bytes)
        if selected["workspace_bytes"] < required_partial_workspace:
            raise ValueError(
                f"operations[{operation_key}] rank-0 workspace is smaller "
                "than its observed partial buffer")
        canonical_operations[operation_key] = {
            "operation": dict(operation),
            "queried_device": device,
            "algorithms": [dict(algorithm) for algorithm in algorithms],
            "selection": {
                "contract": "vendor_rank_0_heuristic",
                "selected_rank": selected["rank"],
                "reason": _SELECTION_REASON,
            },
            "main_launch": main_launch,
            "reducer_launch": reducer_launch,
            "device_execution": device_execution,
            "projection": {
                "execution_variant": execution_variant,
                "logical_tile_m": logical_tile_m,
                "logical_tile_n": logical_tile_n,
                "tile_k": tile_k,
                "pipeline_depth": pipeline_depth,
                "split_k": selected["split_k"],
                "reduction_scheme": selected["reduction_scheme"],
                "workspace_bytes": selected["workspace_bytes"],
                "cta_raster": _cta_raster_projection(
                    execution_variant, main_launch, operation,
                    logical_tile_m, logical_tile_n, selected["split_k"],
                    f"operations[{operation_key}]"),
            },
        }

    return {
        "schema_version": 3,
        "artifact_type": "nvidia_a800_gemm_launch_facts",
        "hardware_id": "nvidia_a800",
        "software_revision": SOFTWARE_REVISION,
        "cublaslt_version": EXPECTED_CUBLASLT_VERSION,
        "source": f"candidate_sha256:{candidate_sha256}",
        "generated_by": "scripts/launch_data/nvidia_a800_launch_generator.py",
        "candidate_sha256": candidate_sha256,
        "capture_baseline_commit": expected_baseline,
        "runner_sha256": candidate["runner_sha256"],
        "query_source_sha256": candidate["query_source_sha256"],
        "workload_manifest_sha256": candidate["workload_manifest_sha256"],
        "workload_manifest": dict(manifest),
        "workload_key_fields": list(GEMM_WORKLOAD_KEY_FIELDS),
        "selector_contract": "vendor_rank_0_heuristic",
        "duration_consumed": False,
        "matmul_executed": False,
        "production_kernel_submitted": False,
        "max_workspace_bytes": EXPECTED_MAX_WORKSPACE_BYTES,
        "semantic_key_count": EXPECTED_SEMANTIC_KEY_COUNT,
        "physical_operation_count": EXPECTED_PHYSICAL_OPERATION_COUNT,
        "workloads": canonical_workloads,
        "operations": canonical_operations,
        "kernel_dataset_sha256": kernel_dataset_sha256,
        "kernel_dataset_selection": [
            [operation_key, list(selection_key)]
            for operation_key, selection_key in kernel_dataset_selection
        ],
    }


def _kernel_dataset_gemm_domain(
        ) -> tuple[tuple[tuple[str, tuple[object, ...]], ...], str]:
    """Return the physical launch domain the A800 kernel dataset measures.

    Kernel evaluation prices exactly the GEMM launches this canonical dataset
    captured.  Their selector facts come from the same rank-zero heuristic the
    ShareGPT domain queries, so they join the one launch-selection artifact
    under physical keys the dataset itself states.  The domain is therefore
    content-addressed by the dataset file.
    """

    import zstandard

    payload = KERNEL_DATASET_GEMM.read_bytes()
    digest = sha256_bytes(payload)
    entries = json.loads(zstandard.ZstdDecompressor().decompress(payload))
    domain = {}
    for case_id, entry in sorted(entries.items()):
        info = entry["op_info"]
        if info["Type"] != "GEMM":
            raise ValueError(
                f"kernel dataset case {case_id!r} is not a GEMM operation")
        attributes = info["Attr"]
        dtype = attributes["dtype"]
        if dtype not in ("bf16", "fp16"):
            raise ValueError(
                f"kernel dataset case {case_id!r} has an unsupported dtype")
        b = require_integer(attributes["B"], f"{case_id} B", positive=True)
        m = require_integer(attributes["M"], f"{case_id} M", positive=True)
        n = require_integer(attributes["N"], f"{case_id} N", positive=True)
        k = require_integer(attributes["K"], f"{case_id} K", positive=True)
        selection_key = (
            dtype, "gemm", b, m, n, k,
            "row_major", "row_major", "row_major", False, True, "none", False,
        )
        operation_key = json.dumps(
            [b * m, n, k, dtype, False], separators=(",", ":"))
        if (selection_key in domain
                and domain[selection_key] != operation_key):
            raise ValueError(
                f"kernel dataset case {case_id!r} has an ambiguous operation")
        domain[selection_key] = operation_key
    if not domain:
        raise ValueError("kernel dataset declares no GEMM launch")
    return tuple(
        (operation_key, selection_key)
        for selection_key, operation_key in sorted(domain.items())
    ), digest


def validate(path: Path) -> tuple[dict[str, object], int]:
    """Validate the canonical root and exact semantic/physical coverage."""

    path = Path(path)
    context = str(path)
    data = require_object(json.loads(path.read_bytes()), context)
    reject_timing_evidence_fields(data, context)
    require_exact_fields(data, _CANONICAL_FIELDS, context)
    if (data["schema_version"] != 3
            or data["artifact_type"] != "nvidia_a800_gemm_launch_facts"
            or data["hardware_id"] != "nvidia_a800"
            or data["software_revision"] != SOFTWARE_REVISION
            or data["cublaslt_version"] != EXPECTED_CUBLASLT_VERSION
            or data["selector_contract"]
            != "vendor_rank_0_heuristic"
            or data["duration_consumed"] is not False
            or data["matmul_executed"] is not False
            or data["production_kernel_submitted"] is not False
            or data["max_workspace_bytes"] != EXPECTED_MAX_WORKSPACE_BYTES):
        raise ValueError(f"{path}: canonical identity or evidence boundary differs")
    require_nonempty_string(data["source"], f"{context}.source")
    if data["generated_by"] != (
            "scripts/launch_data/nvidia_a800_launch_generator.py"):
        raise ValueError(f"{context}: generated_by differs")
    for field in (
            "candidate_sha256", "runner_sha256", "query_source_sha256",
            "workload_manifest_sha256"):
        require_sha256_hex(data[field], f"{context}.{field}")
    expected_source = f"candidate_sha256:{data['candidate_sha256']}"
    if data["source"] != expected_source:
        raise ValueError(
            f"{context}.source must identify the exact candidate SHA256")
    baseline = data["capture_baseline_commit"]
    if (not isinstance(baseline, str)
            or _COMMIT_PATTERN.fullmatch(baseline) is None):
        raise ValueError(f"{context}.capture_baseline_commit is malformed")
    if tuple(require_array(
            data["workload_key_fields"],
            f"{context}.workload_key_fields")) != GEMM_WORKLOAD_KEY_FIELDS:
        raise ValueError(f"{context}: workload key fields differ")

    workloads = require_object(data["workloads"], f"{context}.workloads")
    operations = require_object(data["operations"], f"{context}.operations")
    if (data["semantic_key_count"] != EXPECTED_SEMANTIC_KEY_COUNT
            or len(workloads)
            != EXPECTED_SEMANTIC_KEY_COUNT):
        raise ValueError(f"{path}: semantic workload coverage is incomplete")
    kernel_dataset_selection, kernel_dataset_sha256 = (
        _kernel_dataset_gemm_domain())
    if data["kernel_dataset_sha256"] != kernel_dataset_sha256:
        raise ValueError(f"{context}: kernel dataset digest differs")
    if [[operation_key, list(selection_key)]
            for operation_key, selection_key in kernel_dataset_selection] != (
                data["kernel_dataset_selection"]):
        raise ValueError(f"{context}: kernel dataset launch domain differs")
    kernel_dataset_operation_keys = {
        operation_key for operation_key, _key in kernel_dataset_selection}
    expected_operation_count = len(
        kernel_dataset_operation_keys
        | {gemm_physical_operation_key(key) for key in workloads})
    if (data["physical_operation_count"]
            != EXPECTED_PHYSICAL_OPERATION_COUNT
            or len(operations) != expected_operation_count):
        raise ValueError(f"{path}: physical operation coverage is incomplete")

    manifest = _validate_embedded_manifest(
        data["workload_manifest"], set(workloads),
        f"{context}.workload_manifest")
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if data["workload_manifest_sha256"] != sha256_bytes(manifest_bytes):
        raise ValueError(f"{context}: embedded workload manifest hash differs")

    referenced_operations = set()
    for semantic_key, operation_key in workloads.items():
        validate_gemm_workload_key(semantic_key)
        if not isinstance(operation_key, str):
            raise TypeError(
                f"{context}.workloads[{semantic_key}] must be a string")
        if operation_key != gemm_physical_operation_key(semantic_key):
            raise ValueError(
                f"{context}.workloads[{semantic_key}] maps to the wrong "
                "physical operation")
        if operation_key not in operations:
            raise ValueError(
                f"{context}.workloads[{semantic_key}] references an unknown "
                "operation")
        referenced_operations.add(operation_key)
    if referenced_operations | kernel_dataset_operation_keys != set(
            operations):
        raise ValueError(f"{context}: canonical operations are unreferenced")
    for operation_key, operation in operations.items():
        if not isinstance(operation_key, str):
            raise TypeError(f"{context}: operation keys must be strings")
        _validate_canonical_operation(
            operation_key, operation, EXPECTED_MAX_WORKSPACE_BYTES,
            f"{context}.operations[{operation_key}]")
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
        entries = project_nvidia_a800_gemm(expanded)
        kernel_dataset_entries = (
            project_nvidia_a800_kernel_dataset_gemm(expanded))
        csv_text, manifest = build_gemm_launch_selection_artifact(
            entries,
            physical_entries=kernel_dataset_entries,
            hardware_id="nvidia_a800",
            software_revision=SOFTWARE_REVISION,
            generated_by=(
                "scripts/launch_data/nvidia_a800_launch_generator.py"),
            evidence_sha256=expanded_evidence_sha256(expanded),
            evidence_sources=[
                {
                    "kind": "expanded_evidence",
                    "sha256": expanded_evidence_sha256(expanded),
                },
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
                GEMM_DIR, csv_text, manifest)
        count = manifest["row_count"]
    else:
        if args.write:
            parser.error("--write requires --candidate")
        selection = load_gemm_launch_selection_files(
            GEMM_CSV_PATH,
            GEMM_MANIFEST_PATH,
            hardware_id="nvidia_a800",
            software_revision=SOFTWARE_REVISION,
        )
        count = len(selection)
    print(json.dumps({"gemm": count}, sort_keys=True))
    return 0
