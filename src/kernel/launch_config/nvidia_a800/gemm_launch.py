"""NVIDIA A800 GEMM launch selection from exact cuBLASLt facts."""

from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
import re
from typing import Mapping, NamedTuple, Optional, Sequence

from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.launch_program import GpuCtaRasterProgram
from src.kernel.launch_config.gemm_workload import (
    GEMM_WORKLOAD_KEY_FIELDS,
    LAUNCH_DOMAIN_MODELS,
    SERVING_SCHEDULER_CONTRACT,
    analytical_cublas_tiling,
    cublaslt_main_block_threads,
    gemm_launch_selection_key,
    parse_cublaslt_stages,
    validate_gemm_workload_key,
)
from src.kernel.launch_config.gemm_launch_selection import (
    GemmLaunchSelection,
    ALGORITHM_FIELDS,
    ARTIFACT_MANIFEST_FIELDS,
    ALGORITHM_INTEGER_FIELDS,
    COVERAGE_SCOPE_FIELDS,
    CROSS_HARDWARE_FIELDS,
    DEVICE_EXECUTION_FIELDS,
    HARDWARE_DOMAIN_FIELDS,
    LIMITATION_FIELDS,
    MODEL_DOMAIN_FIELDS,
    OPERATION_FIELDS,
    POSITION_DOMAIN_FIELDS,
    load_gemm_launch_selection,
)
from src.kernel.launch_config.selection_primitives import (
    ceil_div, launch_policy, require_dtype, require_exact_fields as _exact_fields,
    require_mapping as _mapping, require_sequence as _sequence,
    require_sha256 as _require_sha256, require_shape,
    require_variant,
)
from src.kernel.model_contract import (
    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
    SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
)
from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS


_EXPECTED_SEMANTIC_KEY_COUNT = 140544
# The launch-selection CSV carries the ShareGPT semantic domain plus the
# physical keys of the canonical A800 kernel dataset's own GEMM launches.
_EXPECTED_LAUNCH_ROW_COUNT = 140765
_EXPECTED_PHYSICAL_OPERATION_COUNT = 140544
_EXPECTED_SOURCE_CONFIG_SHA256 = (
    "884f01a3efdd980c9730c3a033b89f61968b025bc3b9c37f356312abdcb6072a")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_CANONICAL_FIELDS = {
    "schema_version", "artifact_type", "hardware_id", "software_revision",
    "cublaslt_version", "source", "generated_by", "candidate_sha256",
    "capture_baseline_commit", "runner_sha256", "query_source_sha256",
    "workload_manifest_sha256", "workload_manifest",
    "workload_key_fields", "selector_contract", "duration_consumed",
    "matmul_executed", "production_kernel_submitted", "max_workspace_bytes",
    "semantic_key_count", "physical_operation_count", "workloads",
    "operations",
}
_PROJECTION_FIELDS = {
    "execution_variant", "logical_tile_m", "logical_tile_n", "tile_k",
    "pipeline_depth", "split_k", "reduction_scheme", "workspace_bytes",
}
_SELECTION_REASON = "cublaslt_rank_0_under_captured_preference"
_GEMV_KERNEL_PREFIX = "_ZN8internal5gemvx6kernelI"
_HARDWARE_IDS = EVALUATED_HARDWARE_IDS


# The thread table and analytical tiler are shared with the projecting
# hardware packages (src/kernel/launch_config/gemm_workload.py).
_cublaslt_main_block_threads = cublaslt_main_block_threads




class _ObservedKernelLaunch(NamedTuple):
    algorithm: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_memory_bytes: int


def _integer(value: object, context: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{context} must be {qualifier}")
    return value


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _canonical_sequence_sha256(values: Sequence[str]) -> str:
    payload = json.dumps(
        list(values), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_root_provenance(raw: Mapping[str, object]) -> None:
    _nonempty_string(raw["source"], "NVIDIA A800 GEMM source")
    if raw["generated_by"] != (
            "scripts/launch_data/nvidia_a800_launch_generator.py"):
        raise ValueError("NVIDIA A800 GEMM generator identity differs")
    for field in (
            "candidate_sha256", "runner_sha256", "query_source_sha256",
            "workload_manifest_sha256"):
        value = raw[field]
        if (not isinstance(value, str)
                or _SHA256_PATTERN.fullmatch(value) is None):
            raise ValueError(f"NVIDIA A800 GEMM {field} is malformed")
    baseline = raw["capture_baseline_commit"]
    if (not isinstance(baseline, str)
            or _COMMIT_PATTERN.fullmatch(baseline) is None):
        raise ValueError(
            "NVIDIA A800 GEMM capture_baseline_commit is malformed")


def _validate_embedded_manifest(
        value: object, workload_keys: set[str], expected_sha256: object,
        ) -> None:
    manifest = _mapping(value, "workload_manifest")
    _exact_fields(manifest, ARTIFACT_MANIFEST_FIELDS, "workload_manifest")
    if (manifest["schema_version"] != 2
            or manifest["artifact_type"] != "gemm_workload_candidate"
            or manifest["source_config"] != (
                "configs/evaluation/experiments/gemm_launch_domain.json")
            or manifest["duration_consumed"] is not False
            or manifest["semantic_key_count"]
            != _EXPECTED_SEMANTIC_KEY_COUNT
            or manifest["physical_operation_count"]
            != _EXPECTED_PHYSICAL_OPERATION_COUNT):
        raise ValueError("NVIDIA A800 embedded workload manifest differs")
    if (_require_sha256(
            manifest["source_config_sha256"],
            "workload_manifest.source_config_sha256")
            != _EXPECTED_SOURCE_CONFIG_SHA256):
        raise ValueError("embedded source config hash differs")
    if tuple(_sequence(
            manifest["workload_key_fields"],
            "workload_manifest.workload_key_fields")) != (
                GEMM_WORKLOAD_KEY_FIELDS):
        raise ValueError("embedded workload manifest key fields differ")
    if tuple(_sequence(
            manifest["physical_operation_fields"],
            "workload_manifest.physical_operation_fields")) != (
                "m", "n", "k", "dtype", "has_bias"):
        raise ValueError("embedded physical operation fields differ")
    scheduler = _mapping(
        manifest["scheduler_contract"],
        "workload_manifest.scheduler_contract")
    _exact_fields(
        scheduler, set(SERVING_SCHEDULER_CONTRACT),
        "workload_manifest.scheduler_contract")
    if scheduler != SERVING_SCHEDULER_CONTRACT:
        raise ValueError("embedded scheduler contract differs")
    coverage = _mapping(
        manifest["coverage_scope"], "workload_manifest.coverage_scope")
    _exact_fields(
        coverage, COVERAGE_SCOPE_FIELDS,
        "workload_manifest.coverage_scope")
    if coverage["kind"] != "static_complete_legal_launch_selector_domain":
        raise ValueError("embedded coverage kind differs")
    for field in COVERAGE_SCOPE_FIELDS - {"kind"}:
        if not isinstance(coverage[field], bool):
            raise TypeError(
                f"workload_manifest.coverage_scope.{field} must be boolean")
    if any(coverage[field] is not False for field in (
            "request_trajectory_consumed", "scheduler_alignment_consumed",
            "operator_trace_consumed", "timing_consumed",
            "fused_moe_included",
            "serving_end_to_end_executability_claimed")):
        raise ValueError("embedded coverage evidence boundary differs")

    expected_models = set(LAUNCH_DOMAIN_MODELS)
    inventory = _mapping(
        manifest["dataset_model_inventory"],
        "workload_manifest.dataset_model_inventory")
    if set(inventory) != set(_HARDWARE_IDS):
        raise ValueError("embedded hardware inventory differs")
    for hardware_id in _HARDWARE_IDS:
        models = _sequence(
            inventory[hardware_id],
            f"workload_manifest.inventory.{hardware_id}")
        if (models != sorted(models) or set(models) != expected_models
                or len(models) != len(expected_models)):
            raise ValueError(
                f"embedded model inventory for {hardware_id} differs")

    model_domains = _sequence(
        manifest["model_domains"], "workload_manifest.model_domains")
    if len(model_domains) != len(expected_models):
        raise ValueError("embedded model domain count differs")
    seen_models = set()
    for index, value in enumerate(model_domains):
        context = f"workload_manifest.model_domains[{index}]"
        model = _mapping(value, context)
        _exact_fields(model, MODEL_DOMAIN_FIELDS, context)
        model_id = _nonempty_string(model["model_id"], f"{context}.model_id")
        if model_id not in expected_models or model_id in seen_models:
            raise ValueError("embedded model domain identity differs")
        seen_models.add(model_id)
        launch_model = LAUNCH_DOMAIN_MODELS[model_id]
        expected_type = launch_model.model_type
        if (model["model_type"] != expected_type
                or model["model_path"] != launch_model.model_path
                or _require_sha256(
                    model["model_config_sha256"],
                    f"{context}.model_config_sha256")
                != launch_model.model_config_sha256):
            raise ValueError("embedded model provenance differs")
        if (model["tensor_parallel_size"] != launch_model.tensor_parallel_size
                or model["expert_parallel_size"]
                != launch_model.expert_parallel_size):
            raise ValueError("embedded model parallelism differs")
        raw_positions = _sequence(
            model["positions"], f"{context}.positions")
        if expected_type == "qwen3_moe":
            expected_positions = ("qkv_proj", "o_proj", "router", "lm_head")
        elif expected_type in ("qwen3", "llama"):
            expected_positions = (
                "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head")
        else:
            raise AssertionError(f"unhandled model type {expected_type!r}")
        position_names = []
        position_key_count = 0
        for position_index, value in enumerate(raw_positions):
            position_context = f"{context}.positions[{position_index}]"
            position = _mapping(value, position_context)
            _exact_fields(position, POSITION_DOMAIN_FIELDS, position_context)
            name = _nonempty_string(
                position["position"], f"{position_context}.position")
            position_names.append(name)
            if name == "lm_head":
                expected_m_max = 256
                expected_variant = "m1_gemv_else_gemm"
            elif name in expected_positions:
                expected_m_max = 8192
                expected_variant = "gemm"
            else:
                raise ValueError("embedded model position is unknown")
            if (position["m_min"] != 1
                    or position["m_max"] != expected_m_max
                    or position["semantic_key_count"] != expected_m_max
                    or position["dtype"] != "bf16"
                    or position["has_bias"] is not False
                    or position["variant_contract"] != expected_variant):
                raise ValueError("embedded model position contract differs")
            _integer(position["n"], f"{position_context}.n", positive=True)
            _integer(position["k"], f"{position_context}.k", positive=True)
            position_key_count += position["semantic_key_count"]
        if tuple(position_names) != expected_positions:
            raise ValueError("embedded model positions are invalid")
        semantic_key_count = _integer(
            model["semantic_key_count"], f"{context}.semantic_key_count",
            positive=True)
        physical_operation_count = _integer(
            model["physical_operation_count"],
            f"{context}.physical_operation_count", positive=True)
        if (semantic_key_count != position_key_count
                or physical_operation_count > semantic_key_count):
            raise ValueError("embedded model domain counts differ")
    if seen_models != expected_models:
        raise ValueError("embedded model domain set differs")

    limitations = _sequence(
        manifest["limitations"], "workload_manifest.limitations")
    for index, value in enumerate(limitations):
        context = f"workload_manifest.limitations[{index}]"
        limitation = _mapping(value, context)
        _exact_fields(limitation, LIMITATION_FIELDS, context)
        _nonempty_string(limitation["scope"], f"{context}.scope")
        _nonempty_string(limitation["reason"], f"{context}.reason")
        affected_hardware = _sequence(
            limitation["affected_hardware_ids"],
            f"{context}.affected_hardware_ids")
        affected_models = _sequence(
            limitation["affected_model_ids"],
            f"{context}.affected_model_ids")
        if (not affected_hardware
                or any(item not in _HARDWARE_IDS for item in affected_hardware)
                or len(set(affected_hardware)) != len(affected_hardware)
                or not affected_models
                or any(item not in expected_models for item in affected_models)
                or len(set(affected_models)) != len(affected_models)):
            raise ValueError("embedded limitation scope is invalid")
    manifest_keys = _sequence(
        manifest["workload_keys"], "workload_manifest.workload_keys")
    if (len(manifest_keys) != _EXPECTED_SEMANTIC_KEY_COUNT
            or manifest_keys != sorted(manifest_keys)
            or len(set(manifest_keys)) != len(manifest_keys)
            or set(manifest_keys) != workload_keys):
        raise ValueError("embedded workload manifest key set differs")
    for key in manifest_keys:
        validate_gemm_workload_key(key)
    domain_sha256 = _canonical_sequence_sha256(manifest_keys)
    domains = _mapping(
        manifest["hardware_domains"], "workload_manifest.hardware_domains")
    if set(domains) != set(_HARDWARE_IDS):
        raise ValueError("embedded hardware domains differ")
    for hardware_id in _HARDWARE_IDS:
        context = f"workload_manifest.hardware_domains.{hardware_id}"
        domain = _mapping(domains[hardware_id], context)
        _exact_fields(domain, HARDWARE_DOMAIN_FIELDS, context)
        if domain != {
                "status": "exact",
                "domain_sha256": domain_sha256,
                "semantic_key_count": _EXPECTED_SEMANTIC_KEY_COUNT,
                "physical_operation_count": _EXPECTED_PHYSICAL_OPERATION_COUNT,
        }:
            raise ValueError(f"embedded hardware domain {hardware_id} differs")
    cross_hardware = _mapping(
        manifest["cross_hardware"], "workload_manifest.cross_hardware")
    _exact_fields(
        cross_hardware, CROSS_HARDWARE_FIELDS,
        "workload_manifest.cross_hardware")
    if cross_hardware != {
            "status": "exact_equal",
            "hardware_ids": list(_HARDWARE_IDS),
            "domain_sha256": domain_sha256,
            "semantic_key_count": _EXPECTED_SEMANTIC_KEY_COUNT,
            "physical_operation_count": _EXPECTED_PHYSICAL_OPERATION_COUNT,
    }:
        raise ValueError("embedded cross-hardware contract differs")
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if (_require_sha256(expected_sha256, "workload_manifest_sha256")
            != hashlib.sha256(manifest_bytes).hexdigest()):
        raise ValueError("embedded workload manifest hash differs")


def _validate_algorithm(
        value: object, expected_rank: int, max_workspace_bytes: int,
        context: str) -> Mapping[str, object]:
    algorithm = _mapping(value, context)
    _exact_fields(algorithm, ALGORITHM_FIELDS, context)
    for field in ALGORITHM_INTEGER_FIELDS:
        _integer(algorithm[field], f"{context}.{field}")
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


def _validate_operation_identity(
        operation_key: str, operation: Mapping[str, object], context: str,
        ) -> None:
    _exact_fields(operation, OPERATION_FIELDS, context)
    m = _integer(operation["m"], f"{context}.m", positive=True)
    n = _integer(operation["n"], f"{context}.n", positive=True)
    k = _integer(operation["k"], f"{context}.k", positive=True)
    if operation["dtype"] not in ("bf16", "fp16"):
        raise ValueError(f"{context}.dtype is unsupported")
    if (operation["layout"] != "row_major"
            or operation["trans_a"] is not False
            or operation["trans_b"] is not True
            or not isinstance(operation["has_bias"], bool)):
        raise ValueError(f"{context} has unsupported matrix semantics")
    expected = json.dumps([
        m, n, k, operation["dtype"], operation["has_bias"],
    ], separators=(",", ":"))
    if operation_key != expected:
        raise ValueError(f"{context} does not match its operation key")


def _validate_observed_launch(
        value: object, context: str) -> _ObservedKernelLaunch | None:
    if value is None:
        return None
    launch = _mapping(value, context)
    _exact_fields(
        launch, {"api", "symbol", "grid", "block", "shared_memory_bytes"},
        context)
    if launch["api"] not in (
            "cuLaunchKernel", "cuLaunchKernel_ptsz",
            "cuLaunchKernelEx", "cuLaunchKernelEx_ptsz"):
        raise ValueError(f"{context}.api is unsupported")
    symbol = launch["symbol"]
    if (not isinstance(symbol, str) or not symbol
            or "midlayer_noop" in symbol):
        raise ValueError(f"{context}.symbol is invalid")
    dimensions = {}
    for field in ("grid", "block"):
        raw_dimensions = _sequence(launch[field], f"{context}.{field}")
        if len(raw_dimensions) != 3:
            raise ValueError(f"{context}.{field} must have three dimensions")
        dimensions[field] = tuple(
            _integer(item, f"{context}.{field}[{index}]", positive=True)
            for index, item in enumerate(raw_dimensions)
        )
    shared_memory = _integer(
        launch["shared_memory_bytes"], f"{context}.shared_memory_bytes")
    return _ObservedKernelLaunch(
        algorithm=symbol,
        grid=dimensions["grid"],
        block=dimensions["block"],
        shared_memory_bytes=shared_memory,
    )


def _selected_projection(
        operation: Mapping[str, object], selected: Mapping[str, object],
        main_launch: _ObservedKernelLaunch, context: str,
        ) -> tuple[str, int | None, int | None, int | None, int | None]:
    undefined_tile = (
        selected["tile"] == "undefined"
        and selected["stages"] == "undefined")
    is_gemv = main_launch.algorithm.startswith(_GEMV_KERNEL_PREFIX)
    if undefined_tile:
        if (selected["tile_id"] != 0
                or selected["logical_tile_m"] != 0
                or selected["logical_tile_n"] != 0
                or selected["stages_id"] != 0
                or operation["m"] != 1
                or selected["split_k"] != 1
                or selected["reduction_scheme"] != 0
                or not is_gemv):
            raise NotImplementedError(
                f"{context} has an unsupported undefined-tile rank-0 launch")
        return "gemv", None, None, None, None
    if "undefined" in (selected["tile"], selected["stages"]):
        raise ValueError(f"{context} rank-0 tile metadata is only partly defined")
    tile_k, pipeline_depth = parse_cublaslt_stages(selected["stages"])
    tile_m = selected["logical_tile_m"]
    tile_n = selected["logical_tile_n"]
    if tile_m <= 0 or tile_n <= 0:
        raise ValueError(f"{context} selected an incomplete heuristic")
    if is_gemv:
        raise ValueError(f"{context} GEMV symbol unexpectedly declares a tile")
    return "gemm", tile_m, tile_n, tile_k, pipeline_depth


def _validate_device_execution(
        value: object, expected_count: int, context: str) -> None:
    evidence = _mapping(value, context)
    _exact_fields(evidence, DEVICE_EXECUTION_FIELDS, context)
    if (evidence["evidence_kind"] != "cupti_activity_kernel"
            or evidence["evidence_version"] != 1
            or evidence["all_replacements_observed"] is not True):
        raise ValueError(f"{context} lacks affirmative CUPTI evidence")
    activity_count = _integer(
        evidence["activity_record_count"], f"{context}.activity_record_count")
    dropped_count = _integer(
        evidence["dropped_record_count"], f"{context}.dropped_record_count")
    symbols = _sequence(
        evidence["executed_symbols"], f"{context}.executed_symbols")
    if (dropped_count != 0 or activity_count != expected_count
            or len(symbols) != activity_count
            or any(symbol != "midlayer_noop" for symbol in symbols)):
        raise ValueError(f"{context} does not prove every no-op execution")


@lru_cache(maxsize=1)
def _tiling_lut() -> Mapping[tuple[object, ...], GemmLaunchSelection]:
    result = load_gemm_launch_selection(
        "nvidia_a800", software_revision="cuda_12.2")
    if len(result) != _EXPECTED_LAUNCH_ROW_COUNT:
        raise ValueError(
            "NVIDIA A800 GEMM launch selection must contain "
            f"{_EXPECTED_LAUNCH_ROW_COUNT} rows")
    return result


def select_gemm_launch(shape: Mapping[str, int], dtype: Optional[str],
                       variant: Optional[str], attributes: Mapping[str, object],
                       hardware) -> LaunchConfig:
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    execution_variant = require_variant(variant, ("gemm", "gemv"), "GEMM")
    dtype_name = require_dtype(dtype, ("bf16", "fp16"))
    policy = launch_policy(attributes)
    m_total = b * m
    if policy == "analytical_launch":
        tile_m, tile_n, tile_k, depth = analytical_cublas_tiling(
            m_total, n, k)
        algorithm = f"cublas_analytical_{execution_variant}"
        split_k = 1
        partial_element_bytes = 0.0
        observed_main = None
        selected_execution_variant = execution_variant
        # The analytical tiler stands in for a launch nobody captured, so it
        # states no vendor raster map and the traffic model rasters the grid
        # directly.
        cta_raster = GpuCtaRasterProgram("none", 0, False, 1)
    elif policy == "injected_launch":
        workload_key = gemm_launch_selection_key(
            shape, dtype_name, execution_variant, attributes)
        lut = _tiling_lut()
        if workload_key not in lut:
            raise NotImplementedError(
                "uncharacterized NVIDIA A800 GEMM launch key "
                f"{workload_key}")
        injected = lut[workload_key]
        selected_execution_variant = injected.execution_variant
        tile_m = injected.tile_m
        tile_n = injected.tile_n
        tile_k = injected.tile_k
        depth = injected.pipeline_depth
        algorithm = f"cublaslt_injected_{injected.execution_variant}"
        split_k = injected.split_k
        reduction = injected.reduction
        if reduction in ("none", "in_place"):
            partial_element_bytes = 0.0
        elif reduction == "fp32_external":
            partial_element_bytes = 4.0
        elif reduction == "output_dtype_external":
            partial_element_bytes = 2.0
        else:
            raise AssertionError(
                f"unhandled cuBLASLt reduction contract {reduction!r}")
        raster = injected.cta_raster
        if raster is None:
            if injected.execution_variant != "gemv":
                raise ValueError(
                    "NVIDIA A800 GEMM launch selection lost its CTA raster "
                    "map")
            cta_raster = None
        else:
            cta_raster = GpuCtaRasterProgram(
                raster.convention, raster.log_group, raster.serpentine,
                raster.k_slices)
        observed_main = None
    else:
        raise ValueError(f"unhandled launch policy: {policy!r}")
    if selected_execution_variant == "gemv":
        grid = (m_total * n, 1, 1)
        block = (hardware.parallel_width, 1, 1)
        shared_memory_bytes = 0
    elif selected_execution_variant == "gemm":
        grid = (
            ceil_div(m_total, tile_m) * ceil_div(n, tile_n), 1, split_k)
        block = (_cublaslt_main_block_threads(tile_m, tile_n), 1, 1)
        shared_memory_bytes = 0
    else:
        raise AssertionError(
            f"unhandled injected execution variant {selected_execution_variant!r}")
    if tile_m is None or tile_n is None or tile_k is None:
        if (selected_execution_variant != "gemv"
                or any(value is not None for value in (
                    tile_m, tile_n, tile_k, depth))
                or split_k != 1):
            raise ValueError("NVIDIA A800 launch has incomplete tiling facts")
        tiling = None
    else:
        tiling = TilingSpec(
            tile_shape={"m": tile_m, "n": tile_n, "k": tile_k},
            pipeline_depth=depth,
            split_k=split_k,
        )
    return LaunchConfig(
        algorithm=algorithm,
        grid=grid,
        block=block,
        tiling=tiling,
        parameters={
            "compulsory_traffic": 0.0,
            "weight_tile_rows": 0.0,
            "mte2_tile_bytes": 0.0,
            "aiv_mixed": 0.0,
            "n_major_traffic": 0.0,
            "split_k_partial_bytes_per_element": partial_element_bytes,
            "split_k_partial_dram_bytes_per_element": 0.0,
        },
        execution_variant=selected_execution_variant,
        source_program=(
            cta_raster if selected_execution_variant == "gemm" else None),
    )


def select_target_gemm_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig:
    """Project target-model GEMM semantics onto the BF16 analytical tiler."""

    execution_variant = require_variant(
        variant, ("batched_gemm", "moe_router"), "A800 target GEMM")
    if execution_variant == "batched_gemm":
        require_dtype(dtype, ("bf16",))
    elif execution_variant == "moe_router":
        require_dtype(dtype, ("bf16", "fp32"))
        if attributes["weight_dtype"] != "bf16":
            raise ValueError("target-model MoE router requires BF16 weights")
    else:
        raise AssertionError(
            f"unhandled target GEMM variant: {execution_variant!r}")
    base = select_gemm_launch(
        shape, "bf16", "gemm", attributes, hardware)
    return replace(
        base,
        algorithm=f"{base.algorithm}:{execution_variant}",
        execution_variant=execution_variant,
    )


def select_gemm_reducer_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig | None:
    """Return the exact separate cuBLASLt split-K reducer, when present."""

    require_shape(shape, ("b", "m", "n", "k"))
    execution_variant = require_variant(variant, ("gemm", "gemv"), "GEMM")
    dtype_name = require_dtype(dtype, ("bf16", "fp16"))
    if launch_policy(attributes) != "injected_launch":
        return None
    workload_key = gemm_launch_selection_key(
        shape, dtype_name, execution_variant, attributes)
    lut = _tiling_lut()
    if workload_key not in lut:
        raise NotImplementedError(
            "uncharacterized NVIDIA A800 GEMM launch key "
            f"{workload_key}")
    injected = lut[workload_key]
    if injected.reduction in ("none", "in_place"):
        return None
    if injected.reduction == "fp32_external":
        partial_element_bytes = 4.0
    elif injected.reduction == "output_dtype_external":
        partial_element_bytes = 2.0
    else:
        raise ValueError(
            "separate reducer requires compute-type or output-type partials")
    b, m, n, _ = require_shape(shape, ("b", "m", "n", "k"))
    element_count = b * m * n
    if injected.reduction == "output_dtype_external":
        grid = (ceil_div(n, 32), ceil_div(b * m, 16), 1)
        block = (32, 16, 1)
        source_parameters = (
            cublaslt_output_dtype_reducer_source_parameters(
                element_count=element_count,
                split_k=injected.split_k,
                grid=grid,
                block=block,
            ))
    elif injected.reduction == "fp32_external":
        grid = (ceil_div(element_count, 256), 1, 1)
        block = (256, 1, 1)
        source_parameters = {}
    else:
        raise AssertionError("validated cuBLASLt reduction kind disappeared")
    return LaunchConfig(
        algorithm="logical_gemm_aux:gemm_reduce",
        grid=grid,
        block=block,
        tiling=None,
        parameters={
            "split_k_partial_bytes_per_element": partial_element_bytes,
            "split_k_partial_dram_bytes_per_element": 0.0,
            **source_parameters,
        },
        execution_variant="gemm_reduce",
    )


def cublaslt_output_dtype_reducer_source_parameters(
        *, element_count: int, split_k: int,
        grid: tuple[int, int, int], block: tuple[int, int, int],
        ) -> dict[str, float]:
    """Derive the fixed SM80 output-dtype split-K reducer program.

    The cuBLASLt ``splitKreduce_kernel<32, 16, ..., __half, __half, ...>``
    binary launches sixteen physical warps per CTA.  Each active warp reduces
    one 32-element output segment per row iteration, using two 32-byte L2
    sectors per FP16 partial.  Static SASS groups partials in 16/8/4/tail
    batches and contains at least one load, conversion/FMA, and loop/control
    issue per partial followed by pack/store.
    """

    if (isinstance(element_count, bool)
            or not isinstance(element_count, int)
            or element_count <= 0):
        raise ValueError("cuBLASLt reducer element_count must be positive")
    if (isinstance(split_k, bool)
            or not isinstance(split_k, int)
            or split_k <= 1):
        raise ValueError("cuBLASLt reducer split_k must exceed one")
    if block != (32, 16, 1):
        raise ValueError(
            "cuBLASLt output-dtype reducer requires block=(32, 16, 1)")
    if (len(grid) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int)
            or value <= 0 for value in grid)
            or grid[2] != 1):
        raise ValueError(
            "cuBLASLt output-dtype reducer requires a positive 2D grid")
    if element_count % 32 != 0:
        raise ValueError(
            "cuBLASLt output-dtype reducer requires complete warp segments")
    output_warp_programs = element_count // 32
    physical_warps_per_cta = 16
    row_iterations_per_warp = ceil_div(
        output_warp_programs,
        grid[0] * grid[1] * physical_warps_per_cta,
    )
    active_warps_per_cta = ceil_div(
        output_warp_programs,
        grid[0] * grid[1] * row_iterations_per_warp,
    )
    if active_warps_per_cta > physical_warps_per_cta:
        raise ValueError(
            "cuBLASLt reducer output programs exceed physical warp capacity")
    if split_k == 3:
        # The pinned scalar-FP16/FP32-accumulator SM80 entry takes the
        # three-partial tail directly.  Its host-scalar, beta-zero path
        # executes 82 warp instructions per output segment: 18 entry/guard,
        # 44 reduction setup and tail dispatch, nine load/convert/add
        # instructions, and 11 output/exit instructions.  Other split counts
        # retain the common static lower bound because they take different
        # 16/8/4-way paths.
        warp_instructions_per_output_segment = 82
    else:
        warp_instructions_per_output_segment = 3 * split_k + 64
    return {
        SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER: float(
            2 * split_k * row_iterations_per_warp),
        SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER: float(
            active_warps_per_cta),
        "warp_instructions_lower_bound": float(
            output_warp_programs * warp_instructions_per_output_segment),
    }


def observed_gemm_cta_raster(
        shape: Mapping[str, int], dtype: str, variant: str,
        tile: tuple[int, int, int], split_k: int,
        ) -> Optional[GpuCtaRasterProgram]:
    """Return the retained CTA raster map of one observed A800 GEMM launch.

    A profiler record states the grid, the tile, and the K split of the launch
    it measured, but not the CTA-to-tile map the library applies first: that
    map lives in the kernel's own parameter block.  The canonical launch
    selection retains it for every launch its capture domain covers, keyed by
    the same physical identity.  A launch outside that domain, or one whose
    retained tiling disagrees with the measured launch, mounts no raster map
    and keeps the schedule-free traffic envelope.
    """

    if variant != "gemm":
        return None
    key = (
        dtype, "gemm", int(shape["b"]), int(shape["m"]), int(shape["n"]),
        int(shape["k"]), "row_major", "row_major", "row_major",
        False, True, "none", False,
    )
    lut = _tiling_lut()
    if key not in lut:
        return None
    selection = lut[key]
    if selection.cta_raster is None:
        return None
    if (selection.tile_m, selection.tile_n, selection.tile_k) != tile:
        return None
    if selection.split_k != split_k:
        return None
    raster = selection.cta_raster
    return GpuCtaRasterProgram(
        raster.convention, raster.log_group, raster.serpentine,
        raster.k_slices)
