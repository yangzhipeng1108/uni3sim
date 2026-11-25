"""Aggregate-only comparison of Serving simulation and production measurements."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path, PurePath
import statistics
from typing import Any, Mapping, Sequence

from evaluation.serving.metrics.sharegpt_compact_evaluation import (
    _EXECUTION_ARTIFACT_COUNT_FIELDS,
    STREAMING_EXECUTION_ARTIFACT_HASH_METHODS,
)
from evaluation.serving.metrics.latency_distribution import linear_percentile
from evaluation.serving.simulation.execution_artifact import (
    DEVELOPMENT_PROTOBRIDGE_TIMING_STATUS,
    PROJECTED_PERFORMANCE_TIMING_STATUS,
    UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS,
    performance_timing_status,
)


class ProductionGroundTruthComparisonError(ValueError):
    """Raised when comparison inputs violate their cross-run contract."""


_LATENCY_FIELDS = {
    "ttft": "ttft_s",
    "tpot": "tpot_s",
    "e2e": "latency_s",
}
_THROUGHPUT_FIELDS = (
    "request_throughput_per_s",
    "output_token_throughput_per_s",
)
_PERCENTILES = tuple(range(101))
_LATENCY_DISTRIBUTION_CONTRACT = {
    "schema_version": 2,
    "quantile_representation": (
        "linear_interpolated_quantile_points_p0_p100"
    ),
    "percentile_method": "linear_interpolation_rank_n_minus_1_v1",
    "cdf_representation": "empirical_right_continuous_step_points",
    "cdf_method": "empirical_right_continuous_less_than_or_equal_v1",
}
_CANONICAL_REQUEST_COLUMNS = [
    "Query Time Stamp [ms]",
    "In Token [tokens]",
    "Out Token [tokens]",
    "Cached hit [tokens]",
]
_EXECUTION_ARTIFACT_HASH_METHOD = "canonical_json_sort_keys_compact_utf8_v1"
_SUPPORTED_EXECUTION_ARTIFACT_SCHEMA_VERSIONS = {5, 7, 8}
_EXECUTION_ARTIFACT_SUMMARY_FIELDS = {
    "schema_version",
    "hash_method",
    "performance_timing_status",
    "counts",
    "artifact_sha256",
    "request_level_artifact",
}
_MAIN_CHAIN_FIELDS = {"status", "chain", "execution_artifact_sha256"}
_MAIN_CHAIN = (
    "RequestStream -> SchedulerBackend/BackendStep/ScheduleStep "
    "-> build_workload -> WorkloadGraph -> lower_workload -> "
    "KernelSpec -> predict"
)
_RUNTIME_FIELDS = (
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "expert_parallel_size",
    "dtype",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "max_num_batched_tokens",
    "max_num_seqs",
    "max_model_len",
    "kv_block_size_tokens",
    "scheduler_policy",
)
_ATTESTATION_IDENTITY_FIELDS = (
    "source_kind",
    "hardware_id",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "case_id",
)
_SCHEDULER_POLICY_ATTESTATION_FIELDS = {
    "schema_version",
    "kind",
    "attestor",
    "basis",
    "cases",
}
_SCHEDULER_POLICY_ATTESTATION_CASE_FIELDS = {
    "identity",
    "ground_truth_sha256",
    "scheduler_policy",
    "evidence_kind",
}
_SCHEDULER_POLICY_ATTESTATION_KIND = (
    "sharegpt_ground_truth_scheduler_policy_attestation"
)
_SCHEDULER_POLICY_ATTESTATION_BASIS = "capture_operator_confirmation_v1"
_SCHEDULER_POLICY_ATTESTATION_EVIDENCE_KIND = (
    "ground_truth_runner_configuration_review_v1"
)
_EXPECTED_SCHEDULER_POLICY = "fcfs"


def compare_production_ground_truth(
    ground_truth_inputs: Sequence[str | Path],
    simulation_inputs: Sequence[str | Path],
    *,
    scheduler_policy_attestation: str | Path | None = None,
) -> dict[str, Any]:
    """Compare matched production measurements and simulation aggregates."""

    ground_truth_paths = _discover_inputs(
        ground_truth_inputs, filename="ground_truth.json", label="ground truth"
    )
    simulation_paths = _discover_simulations(simulation_inputs)
    attestation = _load_scheduler_policy_attestation(
        scheduler_policy_attestation
    )
    ground_truth = _load_index(
        ground_truth_paths,
        lambda path: _load_ground_truth(path, attestation),
    )
    _validate_scheduler_policy_attestation_coverage(
        tuple(ground_truth.values()), attestation
    )
    simulations = _load_index(simulation_paths, _load_simulation)
    if set(ground_truth) != set(simulations):
        missing_simulation = sorted(set(ground_truth) - set(simulations))
        missing_ground_truth = sorted(set(simulations) - set(ground_truth))
        raise ProductionGroundTruthComparisonError(
            "comparison identity sets disagree: "
            f"missing_simulation={missing_simulation!r}, "
            f"missing_ground_truth={missing_ground_truth!r}"
        )

    cells = []
    for identity in sorted(ground_truth):
        measured = ground_truth[identity]
        simulated = simulations[identity]
        _validate_binding(measured, simulated)
        cells.append(_compare_cell(measured, simulated))
    if not cells:
        raise ProductionGroundTruthComparisonError(
            "comparison requires at least one matched cell"
        )
    hardware_ids = sorted({cell["identity"]["hardware_id"] for cell in cells})
    groups = [
        _summarize_cells(
            [cell for cell in cells if cell["identity"]["hardware_id"] == hardware_id],
            scope="hardware",
            hardware_id=hardware_id,
        )
        for hardware_id in hardware_ids
    ]
    groups.append(_summarize_cells(cells, scope="overall", hardware_id=None))
    unsupported_cells = [
        cell["identity"]
        for cell in cells
        if cell["simulation_contract"]["accuracy_claim_status"]
        == "unsupported_development_only"
    ]
    return {
        "schema_version": 1,
        "kind": "serving_simulation_production_ground_truth_comparison",
        "comparison_contract": {
            "error_definition": "(simulation - measured) / measured",
            "signed_error_unit": "percent",
            "throughput_units": {
                "request_throughput_per_s": "requests/second",
                "output_token_throughput_per_s": "tokens/second",
            },
            "latency_unit": "millisecond",
            "percentiles": list(_PERCENTILES),
            "percentile_method": _LATENCY_DISTRIBUTION_CONTRACT[
                "percentile_method"
            ],
            "tpot_population": "requests_with_more_than_one_output_token",
            "tpot_population_mismatch": "unsupported_fail_loud",
            "request_payload_consumed": False,
            "forward_inputs_modified": False,
        },
        "accuracy_claim_status": (
            "unsupported_cells_present"
            if unsupported_cells
            else "all_cells_supported_for_accuracy_interpretation"
        ),
        "scheduler_policy_attestation": (
            None
            if attestation is None
            else {
                "source_path": attestation["source_path"],
                "sha256": attestation["sha256"],
                "kind": _SCHEDULER_POLICY_ATTESTATION_KIND,
                "basis": _SCHEDULER_POLICY_ATTESTATION_BASIS,
            }
        ),
        "unsupported_cells": unsupported_cells,
        "cell_count": len(cells),
        "cells": cells,
        "summary": groups,
    }


def write_production_ground_truth_comparison(
    report: Mapping[str, Any], output_directory: str | Path
) -> dict[str, Path]:
    """Write JSON, two CSV views, and a compact Markdown report."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    paths = {
        "json": destination / "comparison.json",
        "case_csv": destination / "per_case_metrics.csv",
        "percentile_csv": destination / "latency_percentiles_p0_p100.csv",
        "markdown": destination / "report.md",
    }
    paths["json"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    _write_case_csv(report, paths["case_csv"])
    _write_percentile_csv(report, paths["percentile_csv"])
    paths["markdown"].write_text(_render_markdown(report))
    return paths


def _discover_inputs(
    inputs: Sequence[str | Path], *, filename: str, label: str
) -> tuple[Path, ...]:
    if not inputs:
        raise ProductionGroundTruthComparisonError(
            f"at least one {label} input is required"
        )
    discovered = []
    for value in inputs:
        path = Path(value)
        if path.is_file():
            if path.name != filename:
                raise ProductionGroundTruthComparisonError(
                    f"{label} file must be named {filename}: {path}"
                )
            discovered.append(path.resolve())
        elif path.is_dir():
            discovered.extend(candidate.resolve() for candidate in path.rglob(filename))
        else:
            raise ProductionGroundTruthComparisonError(
                f"{label} input does not exist: {path}"
            )
    return _unique_paths(discovered, label)


def _discover_simulations(inputs: Sequence[str | Path]) -> tuple[Path, ...]:
    if not inputs:
        raise ProductionGroundTruthComparisonError(
            "at least one simulation input is required"
        )
    discovered = []
    for value in inputs:
        path = Path(value)
        if path.is_file():
            if path.suffix != ".json" or path.name.endswith(".summary.json"):
                raise ProductionGroundTruthComparisonError(
                    f"simulation input must be a non-summary JSON file: {path}"
                )
            discovered.append(path.resolve())
        elif path.is_dir():
            discovered.extend(
                candidate.resolve()
                for candidate in path.rglob("*.json")
                if not candidate.name.endswith(".summary.json")
            )
        else:
            raise ProductionGroundTruthComparisonError(
                f"simulation input does not exist: {path}"
            )
    return _unique_paths(discovered, "simulation")


def _unique_paths(paths: Sequence[Path], label: str) -> tuple[Path, ...]:
    unique = sorted(set(paths))
    if not unique:
        raise ProductionGroundTruthComparisonError(f"no {label} JSON files found")
    if len(unique) != len(paths):
        raise ProductionGroundTruthComparisonError(
            f"duplicate {label} input paths were supplied"
        )
    return tuple(unique)


def _load_index(paths: Sequence[Path], loader: Any) -> dict[tuple[str, str, str], Any]:
    result = {}
    for path in paths:
        cell = loader(path)
        identity = cell["identity_tuple"]
        if identity in result:
            raise ProductionGroundTruthComparisonError(
                f"duplicate comparison identity {identity!r}"
            )
        result[identity] = cell
    return result


def _load_scheduler_policy_attestation(
    path: str | Path | None,
) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        source_path = Path(path).resolve(strict=True)
    except OSError as error:
        raise ProductionGroundTruthComparisonError(
            f"cannot resolve scheduler-policy attestation {path}: {error}"
        ) from error
    if not source_path.is_file():
        raise ProductionGroundTruthComparisonError(
            f"scheduler-policy attestation must be a file: {source_path}"
        )
    source_bytes = _read_bytes(source_path, "scheduler-policy attestation")
    payload = _parse_json_bytes(
        source_bytes, source_path, "scheduler-policy attestation"
    )
    _exact_fields(
        payload,
        _SCHEDULER_POLICY_ATTESTATION_FIELDS,
        "scheduler-policy attestation",
    )
    if payload["schema_version"] != 1:
        raise ProductionGroundTruthComparisonError(
            "scheduler-policy attestation schema_version must equal 1"
        )
    if payload["kind"] != _SCHEDULER_POLICY_ATTESTATION_KIND:
        raise ProductionGroundTruthComparisonError(
            "scheduler-policy attestation kind is unsupported"
        )
    _nonempty_string(payload["attestor"], "scheduler-policy attestor")
    if payload["basis"] != _SCHEDULER_POLICY_ATTESTATION_BASIS:
        raise ProductionGroundTruthComparisonError(
            "scheduler-policy attestation basis must equal "
            f"{_SCHEDULER_POLICY_ATTESTATION_BASIS}"
        )
    cases = _sequence(payload["cases"], "scheduler-policy attestation cases")
    if not cases:
        raise ProductionGroundTruthComparisonError(
            "scheduler-policy attestation requires at least one case"
        )
    cases_by_identity = {}
    for index, value in enumerate(cases):
        label = f"scheduler-policy attestation case {index}"
        case = _mapping(value, label)
        _exact_fields(case, _SCHEDULER_POLICY_ATTESTATION_CASE_FIELDS, label)
        identity = _attestation_identity(case["identity"], f"{label} identity")
        identity_tuple = tuple(
            identity[field] for field in _ATTESTATION_IDENTITY_FIELDS
        )
        if identity_tuple in cases_by_identity:
            raise ProductionGroundTruthComparisonError(
                "scheduler-policy attestation contains duplicate identity"
            )
        ground_truth_sha256 = _sha256(
            case["ground_truth_sha256"], f"{label} ground_truth_sha256"
        )
        if case["scheduler_policy"] != _EXPECTED_SCHEDULER_POLICY:
            raise ProductionGroundTruthComparisonError(
                "scheduler-policy attestation must assert fcfs"
            )
        if case["evidence_kind"] != _SCHEDULER_POLICY_ATTESTATION_EVIDENCE_KIND:
            raise ProductionGroundTruthComparisonError(
                "scheduler-policy attestation evidence_kind is unsupported"
            )
        cases_by_identity[identity_tuple] = {
            "identity": identity,
            "ground_truth_sha256": ground_truth_sha256,
            "scheduler_policy": _EXPECTED_SCHEDULER_POLICY,
            "evidence_kind": _SCHEDULER_POLICY_ATTESTATION_EVIDENCE_KIND,
        }
    return {
        "source_path": str(source_path),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "cases_by_identity": cases_by_identity,
    }


def _validate_scheduler_policy_attestation_coverage(
    ground_truth: Sequence[Mapping[str, Any]],
    attestation: Mapping[str, Any] | None,
) -> None:
    required = {
        value["attestation_identity_tuple"]
        for value in ground_truth
        if value["attestation_identity_tuple"] is not None
    }
    if attestation is None:
        if required:
            raise AssertionError("required scheduler-policy attestation is missing")
        return
    supplied = set(
        _mapping(
            attestation["cases_by_identity"],
            "scheduler-policy attestation case index",
        )
    )
    if supplied != required:
        missing = sorted(required - supplied, key=repr)
        unexpected = sorted(supplied - required, key=repr)
        raise ProductionGroundTruthComparisonError(
            "scheduler-policy attestation identities must exactly equal the "
            "ground truths that lack recorded scheduler_policy: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )


def _load_ground_truth(
    path: Path,
    scheduler_policy_attestation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_bytes = _read_bytes(path, "ground truth")
    ground_truth_sha256 = hashlib.sha256(source_bytes).hexdigest()
    payload = _parse_json_bytes(source_bytes, path, "ground truth")
    if payload["schema_version"] != 1:
        raise ProductionGroundTruthComparisonError(
            f"ground truth schema_version must equal 1: {path}"
        )
    if payload["status"] != "success":
        raise ProductionGroundTruthComparisonError(
            f"ground truth status must be success: {path}"
        )
    workload = _mapping(payload["workload"], "ground truth workload")
    if workload["kind"] != "sharegpt_canonical_schedule":
        raise ProductionGroundTruthComparisonError(
            f"ground truth workload kind is unsupported: {path}"
        )
    identity = _identity(workload["identity"], "ground truth identity")
    if identity["source_kind"] != "sharegpt":
        raise ProductionGroundTruthComparisonError(
            f"ground truth source_kind must be sharegpt: {path}"
        )
    server = _mapping(payload["server"], "ground truth server")
    if server["hardware_id"] != identity["hardware_id"]:
        raise ProductionGroundTruthComparisonError(
            f"ground truth server hardware identity mismatch: {path}"
        )
    runtime_contract, scheduler_policy_binding = _ground_truth_runtime_contract(
        server,
        path,
        identity_value=workload["identity"],
        ground_truth_sha256=ground_truth_sha256,
        scheduler_policy_attestation=scheduler_policy_attestation,
    )
    request_count = _positive_integer(
        workload["request_count"], "ground truth request_count"
    )
    metrics = _mapping(payload["metrics"], "ground truth metrics")
    if _positive_integer(
        metrics["completed_requests"], "completed_requests"
    ) != request_count:
        raise ProductionGroundTruthComparisonError(
            f"ground truth completed request count mismatch: {path}"
        )
    per_request = _sequence(metrics["per_request"], "ground truth per_request")
    if len(per_request) != request_count:
        raise ProductionGroundTruthComparisonError(
            f"ground truth per_request count mismatch: {path}"
        )
    latency_values = {name: [] for name in _LATENCY_FIELDS}
    for index, value in enumerate(per_request):
        record = _mapping(value, f"ground truth per_request[{index}]")
        output_tokens = _positive_integer(
            record["output_tokens"], f"per_request[{index}].output_tokens"
        )
        latency_values["ttft"].append(
            _nonnegative_number(record["ttft_ms"], "ttft_ms")
        )
        latency_values["e2e"].append(
            _nonnegative_number(record["e2e_ms"], "e2e_ms")
        )
        tpot = _nonnegative_number(record["tpot_ms"], "tpot_ms")
        if output_tokens > 1:
            latency_values["tpot"].append(tpot)
        elif tpot != 0:
            raise ProductionGroundTruthComparisonError(
                "single-output-token ground truth request must report tpot_ms=0"
            )
    distributions = {}
    for name, values in latency_values.items():
        if not values:
            raise ProductionGroundTruthComparisonError(
                f"ground truth has no eligible {name} latency values: {path}"
            )
        distributions[name] = _distribution_from_samples(values)
        _validate_ground_truth_summary(
            metrics[name], distributions[name], name, path
        )
    throughputs = {
        name: _positive_number(metrics[name], f"ground truth {name}")
        for name in _THROUGHPUT_FIELDS
    }
    if scheduler_policy_binding["kind"] == "capture_operator_attestation":
        attestation_identity_tuple = tuple(
            _attestation_identity(
                workload["identity"],
                "ground truth attested identity",
            )[field]
            for field in _ATTESTATION_IDENTITY_FIELDS
        )
    elif scheduler_policy_binding["kind"] == "ground_truth_payload":
        attestation_identity_tuple = None
    else:
        raise AssertionError(
            "validated scheduler-policy binding has an unknown kind: "
            f"{scheduler_policy_binding['kind']!r}"
        )
    return {
        "identity_tuple": (
            identity["hardware_id"], identity["model_id"], identity["case_id"]
        ),
        "identity": identity,
        "path": str(path),
        "ground_truth_sha256": ground_truth_sha256,
        "attestation_identity_tuple": attestation_identity_tuple,
        "request_count": request_count,
        "requests_sha256": _sha256(
            workload["requests_sha256"], "ground truth requests_sha256"
        ),
        "request_stream_sha256": _sha256(
            workload["request_stream_sha256"],
            "ground truth request_stream_sha256",
        ),
        "model_config_sha256": _sha256(
            workload["model_config_sha256"],
            "ground truth model_config_sha256",
        ),
        "runtime_contract": runtime_contract,
        "scheduler_policy_binding": scheduler_policy_binding,
        "throughputs": throughputs,
        "latencies": distributions,
    }


def _load_simulation(path: Path) -> dict[str, Any]:
    payload = _load_json(path, "simulation")
    expected = {
        "schema_version": 1,
        "version": 3,
        "kind": "canonical_requests_simulation_aggregate",
        "backend": "vllm",
        "deployment": "colocated",
    }
    for field, value in expected.items():
        if payload[field] != value:
            raise ProductionGroundTruthComparisonError(
                f"simulation {field} must equal {value!r}: {path}"
            )
    request_source = _mapping(payload["request_source"], "simulation request_source")
    if request_source["kind"] != "canonical_requests":
        raise ProductionGroundTruthComparisonError(
            f"simulation request source must be canonical_requests: {path}"
        )
    input_schema = _mapping(
        request_source["input_schema"], "simulation request input schema"
    )
    if (
        input_schema["schema_version"] != 1
        or input_schema["format"] != "csv"
        or input_schema["columns"] != _CANONICAL_REQUEST_COLUMNS
    ):
        raise ProductionGroundTruthComparisonError(
            f"simulation canonical request units/schema mismatch: {path}"
        )
    identity = _simulation_identity(request_source["source_path"])
    input_identity = _simulation_identity(request_source["input_path"])
    if input_identity != identity:
        raise ProductionGroundTruthComparisonError(
            f"simulation source_path and input_path identities disagree: {path}"
        )
    hardware_ids = _sequence(payload["hardware_ids"], "simulation hardware_ids")
    if list(hardware_ids) != [identity["hardware_id"]]:
        raise ProductionGroundTruthComparisonError(
            f"simulation hardware identity mismatch: {path}"
        )
    runtime_contract = _simulation_runtime_contract(payload, path)
    request_count = _positive_integer(
        request_source["row_count"], "simulation request row_count"
    )
    requests_sha256 = _sha256(
        request_source["source_sha256"], "simulation requests source_sha256"
    )
    stream_sha256 = _sha256(
        request_source["normalized_request_stream_sha256"],
        "simulation normalized request stream SHA256",
    )
    workload_binding = _mapping(
        payload["workload_assumption_binding"],
        "simulation workload_assumption_binding",
    )
    if workload_binding["request_stream_sha256"] != stream_sha256:
        raise ProductionGroundTruthComparisonError(
            f"simulation workload binding SHA mismatch: {path}"
        )
    arrival_process = _mapping(payload["arrival_process"], "simulation arrival_process")
    if (
        arrival_process["kind"] != "recorded_canonical_arrivals"
        or arrival_process["request_stream_sha256"] != stream_sha256
    ):
        raise ProductionGroundTruthComparisonError(
            f"simulation arrival process is not bound to request stream: {path}"
        )
    model = _mapping(payload["model_config_provenance"], "simulation model provenance")
    if model["source_config_sha256"] != model["config_sha256"]:
        raise ProductionGroundTruthComparisonError(
            f"simulation model source/config SHA256 values disagree: {path}"
        )
    metrics = _mapping(payload["metrics"], "simulation metrics")
    latency_manifest = _mapping(
        metrics["latency_distributions"], "simulation latency distributions"
    )
    for field, value in _LATENCY_DISTRIBUTION_CONTRACT.items():
        if latency_manifest[field] != value:
            raise ProductionGroundTruthComparisonError(
                f"simulation latency contract {field} mismatch: {path}"
            )
    distributions = {
        name: _simulation_distribution(
            latency_manifest[field], name=name, request_count=request_count, path=path
        )
        for name, field in _LATENCY_FIELDS.items()
    }
    throughputs = {
        name: _positive_number(metrics[name], f"simulation {name}")
        for name in _THROUGHPUT_FIELDS
    }
    launch_policy, timing_status = _validate_timing_contract(
        payload, tuple(hardware_ids), path
    )
    return {
        "identity_tuple": (
            identity["hardware_id"], identity["model_id"], identity["case_id"]
        ),
        "identity": identity,
        "path": str(path),
        "request_count": request_count,
        "requests_sha256": requests_sha256,
        "request_stream_sha256": stream_sha256,
        "model_config_sha256": _sha256(
            model["config_sha256"], "simulation model config SHA256"
        ),
        "runtime_contract": runtime_contract,
        "performance_timing_status": timing_status,
        "launch_policy": launch_policy,
        "throughputs": throughputs,
        "latencies": distributions,
    }


def _simulation_identity(source_path: Any) -> dict[str, str]:
    source = PurePath(_nonempty_string(source_path, "simulation source_path"))
    parts = source.parts
    matches = [
        index
        for index in range(len(parts) - 1)
        if parts[index : index + 2] == ("serving_dataset", "sharegpt")
    ]
    if len(matches) != 1:
        raise ProductionGroundTruthComparisonError(
            "simulation source_path must contain exactly one "
            "serving_dataset/sharegpt segment"
        )
    tail = parts[matches[0] + 2 :]
    if len(tail) != 4 or tail[3] != "requests.csv":
        raise ProductionGroundTruthComparisonError(
            "simulation source_path must end with "
            "<hardware>/<model>/<case>/requests.csv"
        )
    return {
        "source_kind": "sharegpt",
        "hardware_id": tail[0],
        "model_id": tail[1],
        "case_id": tail[2],
    }


def _ground_truth_runtime_contract(
    server: Mapping[str, Any],
    path: Path,
    *,
    identity_value: Any,
    ground_truth_sha256: str,
    scheduler_policy_attestation: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    engine_label = f"ground truth server.engine in {path}"
    engine = _mapping(
        _required_field(server, "engine", "ground truth server"),
        engine_label,
    )
    contract = {
        field: _required_field(engine, field, engine_label)
        for field in _RUNTIME_FIELDS
        if field != "scheduler_policy"
    }
    for field in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "expert_parallel_size",
        "max_num_batched_tokens",
        "max_num_seqs",
        "max_model_len",
    ):
        contract[field] = _positive_integer(
            contract[field], f"ground truth server.engine.{field}"
        )
    for field in ("enable_chunked_prefill", "enable_prefix_caching"):
        contract[field] = _boolean(
            contract[field], f"ground truth server.engine.{field}"
        )
    if contract["dtype"] != "bfloat16":
        raise ProductionGroundTruthComparisonError(
            f"ground truth engine dtype must be bfloat16: {path}"
        )
    if "scheduler_policy" in engine:
        contract["scheduler_policy"] = _nonempty_string(
            engine["scheduler_policy"],
            "ground truth server.engine.scheduler_policy",
        )
        return contract, {
            "kind": "ground_truth_payload",
            "ground_truth_sha256": ground_truth_sha256,
            "attestation_source_path": None,
            "attestation_sha256": None,
            "evidence_kind": None,
        }
    if scheduler_policy_attestation is None:
        raise ProductionGroundTruthComparisonError(
            "ground truth server.engine is missing required field "
            "'scheduler_policy'; a structured scheduler-policy attestation "
            f"is required: {path}"
        )
    identity = _attestation_identity(
        identity_value, "ground truth identity requiring scheduler-policy attestation"
    )
    identity_tuple = tuple(identity[field] for field in _ATTESTATION_IDENTITY_FIELDS)
    cases = _mapping(
        scheduler_policy_attestation["cases_by_identity"],
        "scheduler-policy attestation case index",
    )
    if identity_tuple not in cases:
        raise ProductionGroundTruthComparisonError(
            "scheduler-policy attestation must contain exactly one matching case "
            f"for ground truth {path}"
        )
    attested_case = _mapping(
        cases[identity_tuple], "matching scheduler-policy attestation case"
    )
    if attested_case["ground_truth_sha256"] != ground_truth_sha256:
        raise ProductionGroundTruthComparisonError(
            "scheduler-policy attestation ground_truth_sha256 does not match "
            f"raw ground truth: {path}"
        )
    contract["scheduler_policy"] = attested_case["scheduler_policy"]
    return contract, {
        "kind": "capture_operator_attestation",
        "ground_truth_sha256": ground_truth_sha256,
        "attestation_source_path": scheduler_policy_attestation["source_path"],
        "attestation_sha256": scheduler_policy_attestation["sha256"],
        "evidence_kind": attested_case["evidence_kind"],
    }


def _simulation_runtime_contract(
    payload: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    scheduler_runtime = _mapping(
        _required_field(payload, "scheduler_runtime", "simulation"),
        "simulation scheduler_runtime",
    )
    if set(scheduler_runtime) != {"main"}:
        raise ProductionGroundTruthComparisonError(
            f"colocated simulation scheduler_runtime must contain only main: {path}"
        )
    runtime = _mapping(
        scheduler_runtime["main"], "simulation scheduler_runtime.main"
    )
    field_mapping = {
        "tensor_parallel_size": "effective_tensor_parallel_size",
        "pipeline_parallel_size": "effective_pipeline_parallel_size",
        "dtype": "effective_dtype",
        "enable_chunked_prefill": "effective_chunked_prefill",
        "enable_prefix_caching": "requested_prefix_caching",
        "max_num_batched_tokens": "max_num_batched_tokens",
        "max_num_seqs": "max_num_seqs",
        "max_model_len": "max_model_len",
        "kv_block_size_tokens": "kv_block_size_tokens",
        "scheduler_policy": "scheduler_policy",
    }
    contract = {
        field: _required_field(runtime, runtime_field, "scheduler_runtime.main")
        for field, runtime_field in field_mapping.items()
    }
    for field in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "max_num_batched_tokens",
        "max_num_seqs",
        "max_model_len",
        "kv_block_size_tokens",
    ):
        contract[field] = _positive_integer(
            contract[field], f"simulation scheduler_runtime.main.{field}"
        )
    for field in ("enable_chunked_prefill", "enable_prefix_caching"):
        contract[field] = _boolean(
            contract[field], f"simulation scheduler_runtime.main.{field}"
        )
    effective_expert_parallel = _boolean(
        _required_field(
            runtime, "effective_expert_parallel", "scheduler_runtime.main"
        ),
        "simulation scheduler_runtime.main.effective_expert_parallel",
    )
    contract["expert_parallel_size"] = (
        contract["tensor_parallel_size"] if effective_expert_parallel else 1
    )
    if contract["dtype"] != "bfloat16":
        raise ProductionGroundTruthComparisonError(
            f"simulation effective dtype must be bfloat16: {path}"
        )
    if payload["model_compute_dtype"] != "bf16":
        raise ProductionGroundTruthComparisonError(
            f"simulation model_compute_dtype must be bf16: {path}"
        )
    contract["scheduler_policy"] = _nonempty_string(
        contract["scheduler_policy"],
        "simulation scheduler_runtime.main.scheduler_policy",
    )
    return contract


def _uses_development_protobridge(value: Any) -> bool:
    if value is None:
        return False
    inputs = _mapping(value, "simulation external predictor inputs")
    _exact_fields(
        inputs,
        {"protobridge"},
        "simulation external predictor inputs",
    )
    protobridge = _mapping(
        inputs["protobridge"], "simulation ProtoBridge predictor inputs"
    )
    _exact_fields(
        protobridge,
        {"backend", "profile_sha256_by_operation", "source_kind"},
        "simulation ProtoBridge predictor inputs",
    )
    if protobridge["backend"] != "protobridge":
        raise ProductionGroundTruthComparisonError(
            "simulation external predictor backend must be protobridge"
        )
    if protobridge["source_kind"] != "development_only":
        raise ProductionGroundTruthComparisonError(
            "simulation ProtoBridge source_kind is unsupported"
        )
    return True


def _validate_timing_contract(
    payload: Mapping[str, Any], hardware_ids: tuple[Any, ...], path: Path
) -> tuple[str, str]:
    launch_policy = _nonempty_string(payload["launch_policy"], "launch_policy")
    execution = _mapping(
        payload["execution_provenance"], "simulation execution_provenance"
    )
    try:
        expected_status = performance_timing_status(
            launch_policy,
            hardware_ids,
            uses_development_protobridge=_uses_development_protobridge(
                execution["external_predictor_inputs"]
            ),
        )
    except (TypeError, ValueError) as error:
        raise ProductionGroundTruthComparisonError(
            f"invalid canonical timing policy in {path}: {error}"
        ) from error
    timing_status = _nonempty_string(
        payload["performance_timing_status"], "performance_timing_status"
    )
    if timing_status != expected_status:
        raise ProductionGroundTruthComparisonError(
            f"simulation timing status disagrees with canonical policy: {path}"
        )
    launch_inputs = _mapping(
        execution["launch_inputs"], "simulation execution launch_inputs"
    )
    artifact = _mapping(payload["execution_artifact"], "simulation execution_artifact")
    _exact_fields(
        artifact,
        _EXECUTION_ARTIFACT_SUMMARY_FIELDS,
        f"simulation execution_artifact summary in {path}",
    )
    if (
        execution["performance_timing_status"] != timing_status
        or launch_inputs["launch_policy"] != launch_policy
        or artifact["performance_timing_status"] != timing_status
    ):
        raise ProductionGroundTruthComparisonError(
            f"simulation execution timing/launch contract is inconsistent: {path}"
        )
    schema_version = artifact["schema_version"]
    if schema_version in _SUPPORTED_EXECUTION_ARTIFACT_SCHEMA_VERSIONS:
        expected_hash_method = _EXECUTION_ARTIFACT_HASH_METHOD
    elif schema_version in STREAMING_EXECUTION_ARTIFACT_HASH_METHODS:
        expected_hash_method = STREAMING_EXECUTION_ARTIFACT_HASH_METHODS[
            schema_version
        ]
    else:
        raise ProductionGroundTruthComparisonError(
            f"unsupported simulation execution artifact schema: {path}"
        )
    if artifact["hash_method"] != expected_hash_method:
        raise ProductionGroundTruthComparisonError(
            f"simulation execution artifact hash method mismatch: {path}"
        )
    if artifact["request_level_artifact"] != "omitted":
        raise ProductionGroundTruthComparisonError(
            f"aggregate simulation must omit request-level execution artifact: {path}"
        )
    counts = _mapping(
        artifact["counts"], "simulation execution_artifact counts"
    )
    _exact_fields(
        counts,
        _EXECUTION_ARTIFACT_COUNT_FIELDS,
        f"simulation execution_artifact counts in {path}",
    )
    for field in _EXECUTION_ARTIFACT_COUNT_FIELDS:
        _nonnegative_integer(
            counts[field], f"simulation execution_artifact counts.{field}"
        )
    artifact_sha256 = _sha256(
        artifact["artifact_sha256"], "execution artifact SHA256"
    )
    main_chain = _mapping(payload["main_chain_status"], "simulation main_chain_status")
    _exact_fields(
        main_chain,
        _MAIN_CHAIN_FIELDS,
        f"simulation main_chain_status in {path}",
    )
    if (
        main_chain["status"] != "executed"
        or main_chain["chain"] != _MAIN_CHAIN
        or main_chain["execution_artifact_sha256"] != artifact_sha256
    ):
        raise ProductionGroundTruthComparisonError(
            f"simulation main chain is not bound to the execution artifact: {path}"
        )
    return launch_policy, timing_status


def _simulation_distribution(
    value: Any, *, name: str, request_count: int, path: Path
) -> dict[str, Any]:
    distribution = _mapping(value, f"simulation {name} distribution")
    if distribution["availability"] != "available":
        raise ProductionGroundTruthComparisonError(
            f"simulation {name} distribution must be available: {path}"
        )
    eligible = _positive_integer(
        distribution["eligible_request_count"],
        f"simulation {name} eligible_request_count",
    )
    excluded = _nonnegative_integer(
        distribution["excluded_request_count"],
        f"simulation {name} excluded_request_count",
    )
    if eligible + excluded != request_count:
        raise ProductionGroundTruthComparisonError(
            f"simulation {name} request count mismatch: {path}"
        )
    quantiles = _mapping(distribution["quantiles"], f"simulation {name} quantiles")
    percentiles = list(
        _sequence(quantiles["percentiles"], f"simulation {name} percentiles")
    )
    if percentiles != list(_PERCENTILES):
        raise ProductionGroundTruthComparisonError(
            f"simulation {name} percentiles must be exactly P0-P100: {path}"
        )
    values_s = _numeric_sequence(
        quantiles["values_s"], f"simulation {name} quantile values_s"
    )
    if len(values_s) != len(_PERCENTILES) or values_s != sorted(values_s):
        raise ProductionGroundTruthComparisonError(
            f"simulation {name} P0-P100 values_s are invalid: {path}"
        )
    cdf = _mapping(distribution["empirical_cdf"], f"simulation {name} CDF")
    cdf_values_s = _numeric_sequence(
        cdf["values_s"], f"simulation {name} CDF values_s"
    )
    probabilities = _numeric_sequence(
        cdf["cumulative_probabilities"],
        f"simulation {name} CDF cumulative probabilities",
    )
    samples_s = _expand_empirical_cdf(
        cdf_values_s, probabilities, eligible, f"simulation {name} CDF"
    )
    expected_quantiles = [
        linear_percentile(samples_s, percentile) for percentile in _PERCENTILES
    ]
    for percentile, actual, expected in zip(
        _PERCENTILES, values_s, expected_quantiles
    ):
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ProductionGroundTruthComparisonError(
                f"simulation {name} P{percentile} disagrees with empirical CDF: {path}"
            )
    return {
        "eligible_request_count": eligible,
        "mean_ms": statistics.fmean(samples_s) * 1000.0,
        "percentiles_ms": [value * 1000.0 for value in values_s],
    }


def _expand_empirical_cdf(
    values: Sequence[float],
    probabilities: Sequence[float],
    eligible_count: int,
    label: str,
) -> list[float]:
    if not values or len(values) != len(probabilities) or values != sorted(values):
        raise ProductionGroundTruthComparisonError(f"{label} arrays are invalid")
    samples = []
    previous_count = 0
    previous_probability = 0.0
    for value, probability in zip(values, probabilities):
        if probability <= previous_probability or probability > 1:
            raise ProductionGroundTruthComparisonError(
                f"{label} probabilities must be strictly increasing through 1"
            )
        cumulative_count = round(probability * eligible_count)
        if not math.isclose(
            probability, cumulative_count / eligible_count, abs_tol=1e-12
        ):
            raise ProductionGroundTruthComparisonError(
                f"{label} probability is not bound to eligible_request_count"
            )
        samples.extend([value] * (cumulative_count - previous_count))
        previous_count = cumulative_count
        previous_probability = probability
    if previous_count != eligible_count or not math.isclose(
        previous_probability, 1.0, abs_tol=1e-12
    ):
        raise ProductionGroundTruthComparisonError(
            f"{label} must terminate at the eligible request count"
        )
    return samples


def _validate_binding(measured: Mapping[str, Any], simulated: Mapping[str, Any]) -> None:
    for field in (
        "identity_tuple",
        "request_count",
        "requests_sha256",
        "request_stream_sha256",
        "model_config_sha256",
    ):
        if measured[field] != simulated[field]:
            raise ProductionGroundTruthComparisonError(
                f"matched cell {measured['identity_tuple']!r} has conflicting {field}"
            )
    for field in _RUNTIME_FIELDS:
        if measured["runtime_contract"][field] != simulated["runtime_contract"][field]:
            raise ProductionGroundTruthComparisonError(
                f"matched cell {measured['identity_tuple']!r} has conflicting "
                f"runtime field {field}: "
                f"measured={measured['runtime_contract'][field]!r}, "
                f"simulated={simulated['runtime_contract'][field]!r}"
            )
    for name in _LATENCY_FIELDS:
        if (
            measured["latencies"][name]["eligible_request_count"]
            != simulated["latencies"][name]["eligible_request_count"]
        ):
            if name == "tpot":
                raise ProductionGroundTruthComparisonError(
                    f"matched cell {measured['identity_tuple']!r} has different "
                    "TPOT eligible populations; comparison when output_tokens<=1 "
                    "membership differs is currently unsupported"
                )
            raise ProductionGroundTruthComparisonError(
                f"matched cell {measured['identity_tuple']!r} has conflicting "
                f"{name} eligible request counts"
            )


def _compare_cell(
    measured: Mapping[str, Any], simulated: Mapping[str, Any]
) -> dict[str, Any]:
    throughput = {
        name: _comparison(
            measured["throughputs"][name], simulated["throughputs"][name]
        )
        for name in _THROUGHPUT_FIELDS
    }
    latency = {}
    for name in _LATENCY_FIELDS:
        measured_distribution = measured["latencies"][name]
        simulated_distribution = simulated["latencies"][name]
        latency[name] = {
            "eligible_request_count": measured_distribution[
                "eligible_request_count"
            ],
            "mean": _comparison(
                measured_distribution["mean_ms"],
                simulated_distribution["mean_ms"],
            ),
            "percentiles": [
                {
                    "percentile": percentile,
                    **_comparison(measured_value, simulated_value),
                }
                for percentile, measured_value, simulated_value in zip(
                    _PERCENTILES,
                    measured_distribution["percentiles_ms"],
                    simulated_distribution["percentiles_ms"],
                )
            ],
        }
    timing_status = simulated["performance_timing_status"]
    if timing_status == PROJECTED_PERFORMANCE_TIMING_STATUS:
        accuracy_claim_status = "supported_for_accuracy_interpretation"
    elif timing_status in {
        DEVELOPMENT_PROTOBRIDGE_TIMING_STATUS,
        UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS,
    }:
        accuracy_claim_status = "unsupported_development_only"
    else:
        raise ProductionGroundTruthComparisonError(
            f"unknown canonical performance timing status: {timing_status!r}"
        )
    return {
        "identity": {
            key: measured["identity"][key]
            for key in ("hardware_id", "model_id", "case_id")
        },
        "binding": {
            "request_count": measured["request_count"],
            "requests_sha256": measured["requests_sha256"],
            "request_stream_sha256": measured["request_stream_sha256"],
            "model_config_sha256": measured["model_config_sha256"],
            "runtime": measured["runtime_contract"],
            "scheduler_policy_provenance": measured[
                "scheduler_policy_binding"
            ],
        },
        "sources": {
            "ground_truth_json": measured["path"],
            "simulation_json": simulated["path"],
        },
        "simulation_contract": {
            "performance_timing_status": timing_status,
            "launch_policy": simulated["launch_policy"],
            "accuracy_claim_status": accuracy_claim_status,
        },
        "metrics": {
            "throughput": throughput,
            "latency_ms": latency,
        },
    }


def _summarize_cells(
    cells: Sequence[Mapping[str, Any]], *, scope: str, hardware_id: str | None
) -> dict[str, Any]:
    metrics = {}
    for name in _THROUGHPUT_FIELDS:
        metrics[name] = _mape(
            [cell["metrics"]["throughput"][name] for cell in cells]
        )
    for name in _LATENCY_FIELDS:
        metrics[f"{name}_mean_ms"] = _mape(
            [cell["metrics"]["latency_ms"][name]["mean"] for cell in cells]
        )
        metrics[f"{name}_p0_p100_ms"] = _mape(
            [
                percentile
                for cell in cells
                for percentile in cell["metrics"]["latency_ms"][name][
                    "percentiles"
                ]
            ]
        )
    timing_statuses = sorted({
        cell["simulation_contract"]["performance_timing_status"]
        for cell in cells
    })
    if ({
        DEVELOPMENT_PROTOBRIDGE_TIMING_STATUS,
        UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS,
    } & set(timing_statuses)):
        accuracy_claim_status = "unsupported_cells_present"
    else:
        accuracy_claim_status = "supported_for_accuracy_interpretation"
    return {
        "scope": scope,
        "hardware_id": hardware_id,
        "cell_count": len(cells),
        "performance_timing_statuses": timing_statuses,
        "accuracy_claim_status": accuracy_claim_status,
        "metrics": metrics,
    }


def _comparison(measured: float, simulated: float) -> dict[str, float]:
    if measured <= 0:
        raise ProductionGroundTruthComparisonError(
            "relative comparison requires a positive measured value"
        )
    relative = (simulated - measured) / measured
    return {
        "measured": measured,
        "simulated": simulated,
        "signed_relative_error_pct": relative * 100.0,
        "absolute_relative_error_pct": abs(relative) * 100.0,
    }


def _mape(comparisons: Sequence[Mapping[str, float]]) -> dict[str, float]:
    return {
        "mean_absolute_relative_error_pct": statistics.fmean(
            value["absolute_relative_error_pct"] for value in comparisons
        ),
        "mean_signed_relative_error_pct": statistics.fmean(
            value["signed_relative_error_pct"] for value in comparisons
        ),
    }


def _distribution_from_samples(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "eligible_request_count": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "percentiles_ms": [
            linear_percentile(ordered, percentile) for percentile in _PERCENTILES
        ],
    }


def _validate_ground_truth_summary(
    value: Any,
    distribution: Mapping[str, Any],
    name: str,
    path: Path,
) -> None:
    summary = _mapping(value, f"ground truth {name} summary")
    expected = {
        "mean_ms": distribution["mean_ms"],
        "p50_ms": distribution["percentiles_ms"][50],
        "p90_ms": distribution["percentiles_ms"][90],
        "p99_ms": distribution["percentiles_ms"][99],
    }
    for field, expected_value in expected.items():
        actual = _nonnegative_number(summary[field], f"ground truth {name}.{field}")
        if not math.isclose(actual, expected_value, rel_tol=1e-10, abs_tol=1e-9):
            raise ProductionGroundTruthComparisonError(
                f"ground truth {name}.{field} disagrees with per-request metrics: {path}"
            )


def _write_case_csv(report: Mapping[str, Any], path: Path) -> None:
    fields = ["hardware_id", "model_id", "case_id"]
    metric_names = [*_THROUGHPUT_FIELDS, "ttft_mean_ms", "tpot_mean_ms", "e2e_mean_ms"]
    for name in metric_names:
        fields.extend(
            (
                f"{name}_measured",
                f"{name}_simulated",
                f"{name}_signed_relative_error_pct",
                f"{name}_absolute_relative_error_pct",
            )
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in report["cells"]:
            row = dict(cell["identity"])
            comparisons = {
                **cell["metrics"]["throughput"],
                **{
                    f"{name}_mean_ms": cell["metrics"]["latency_ms"][name]["mean"]
                    for name in _LATENCY_FIELDS
                },
            }
            for name, comparison in comparisons.items():
                for field, value in comparison.items():
                    row[f"{name}_{field}"] = value
            writer.writerow(row)


def _write_percentile_csv(report: Mapping[str, Any], path: Path) -> None:
    fields = [
        "hardware_id",
        "model_id",
        "case_id",
        "latency_metric",
        "percentile",
        "measured_ms",
        "simulated_ms",
        "signed_relative_error_pct",
        "absolute_relative_error_pct",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in report["cells"]:
            for name in _LATENCY_FIELDS:
                for comparison in cell["metrics"]["latency_ms"][name]["percentiles"]:
                    writer.writerow({
                        **cell["identity"],
                        "latency_metric": name,
                        "percentile": comparison["percentile"],
                        "measured_ms": comparison["measured"],
                        "simulated_ms": comparison["simulated"],
                        "signed_relative_error_pct": comparison[
                            "signed_relative_error_pct"
                        ],
                        "absolute_relative_error_pct": comparison[
                            "absolute_relative_error_pct"
                        ],
                    })


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Serving simulation versus production ground truth",
        "",
        (
            "Error is `(simulation - measured) / measured`; positive values "
            "mean the simulation is higher."
        ),
        "",
    ]
    scheduler_attestation = report["scheduler_policy_attestation"]
    if scheduler_attestation is not None:
        lines.extend([
            (
                "Scheduler policy for legacy ground truth is bound to "
                f"`{scheduler_attestation['source_path']}` "
                f"(SHA256 `{scheduler_attestation['sha256']}`)."
            ),
            "",
        ])
    if report["accuracy_claim_status"] == "unsupported_cells_present":
        lines.extend([
            (
                "**WARNING: This report contains development-only timing "
                "results. Cells marked `unsupported_development_only` must "
                "not be used for formal accuracy claims.**"
            ),
            "",
        ])
    elif report["accuracy_claim_status"] == (
        "all_cells_supported_for_accuracy_interpretation"
    ):
        pass
    else:
        raise ProductionGroundTruthComparisonError(
            "unknown report accuracy_claim_status"
        )
    lines.extend([
        "## Summary",
        "",
        (
            "| Scope | Timing status | Cells | Req/s MAPE | Output tok/s MAPE | TTFT mean "
            "MAPE | TPOT mean MAPE | E2E mean MAPE |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for group in report["summary"]:
        if group["scope"] == "hardware":
            scope = group["hardware_id"]
        elif group["scope"] == "overall":
            scope = "overall"
        else:
            raise ProductionGroundTruthComparisonError(
                f"unknown report summary scope: {group['scope']!r}"
            )
        metrics = group["metrics"]
        timing = ", ".join(group["performance_timing_statuses"])
        lines.append(
            f"| {scope} | {timing} | {group['cell_count']} | "
            f"{_format_pct(metrics['request_throughput_per_s'])} | "
            f"{_format_pct(metrics['output_token_throughput_per_s'])} | "
            f"{_format_pct(metrics['ttft_mean_ms'])} | "
            f"{_format_pct(metrics['tpot_mean_ms'])} | "
            f"{_format_pct(metrics['e2e_mean_ms'])} |"
        )
    lines.extend([
        "",
        "## Per-case aggregate metrics",
        "",
        (
            "| Hardware | Model | Case | Timing status | Req/s measured→simulated | Output "
            "tok/s measured→simulated | TTFT mean ms measured→simulated | "
            "TPOT mean ms measured→simulated | E2E mean ms "
            "measured→simulated |"
        ),
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for cell in report["cells"]:
        identity = cell["identity"]
        throughput = cell["metrics"]["throughput"]
        latency = cell["metrics"]["latency_ms"]
        lines.append(
            f"| {identity['hardware_id']} | {identity['model_id']} | {identity['case_id']} | "
            f"{cell['simulation_contract']['performance_timing_status']} | "
            f"{_format_comparison(throughput['request_throughput_per_s'])} | "
            f"{_format_comparison(throughput['output_token_throughput_per_s'])} | "
            f"{_format_comparison(latency['ttft']['mean'])} | "
            f"{_format_comparison(latency['tpot']['mean'])} | "
            f"{_format_comparison(latency['e2e']['mean'])} |"
        )
    return "\n".join(lines) + "\n"


def _format_pct(summary: Mapping[str, float]) -> str:
    return f"{summary['mean_absolute_relative_error_pct']:.2f}%"


def _format_comparison(comparison: Mapping[str, float]) -> str:
    return (
        f"{comparison['measured']:.3f}→{comparison['simulated']:.3f} "
        f"({comparison['signed_relative_error_pct']:+.2f}%)"
    )


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    source_bytes = _read_bytes(path, label)
    return _parse_json_bytes(source_bytes, path, label)


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ProductionGroundTruthComparisonError(
            f"cannot read {label} JSON {path}: {error}"
        ) from error


def _parse_json_bytes(
    source_bytes: bytes, path: Path, label: str
) -> Mapping[str, Any]:
    try:
        value = json.loads(source_bytes, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductionGroundTruthComparisonError(
            f"cannot load {label} JSON {path}: {error}"
        ) from error
    return _mapping(value, label)


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProductionGroundTruthComparisonError(
                f"duplicate JSON object key: {key!r}"
            )
        result[key] = value
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProductionGroundTruthComparisonError(f"{label} must be an object")
    return value


def _required_field(mapping: Mapping[str, Any], field: str, label: str) -> Any:
    if field not in mapping:
        raise ProductionGroundTruthComparisonError(
            f"{label} is missing required field {field!r}"
        )
    return mapping[field]


def _exact_fields(
    mapping: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(mapping)
    if actual != expected:
        raise ProductionGroundTruthComparisonError(
            f"{label} fields mismatch: "
            f"missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}"
        )


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ProductionGroundTruthComparisonError(f"{label} must be an array")
    return value


def _identity(value: Any, label: str) -> dict[str, str]:
    identity = _mapping(value, label)
    return {
        field: _nonempty_string(identity[field], f"{label}.{field}")
        for field in ("source_kind", "hardware_id", "model_id", "case_id")
    }


def _attestation_identity(value: Any, label: str) -> dict[str, Any]:
    identity = _mapping(value, label)
    _exact_fields(identity, set(_ATTESTATION_IDENTITY_FIELDS), label)
    result = {
        field: _nonempty_string(identity[field], f"{label}.{field}")
        for field in ("source_kind", "hardware_id", "model_id", "case_id")
    }
    if result["source_kind"] != "sharegpt":
        raise ProductionGroundTruthComparisonError(
            f"{label}.source_kind must equal sharegpt"
        )
    for field in ("model_revision", "tokenizer_revision"):
        revision = identity[field]
        if revision is not None:
            revision = _nonempty_string(revision, f"{label}.{field}")
        result[field] = revision
    return {
        field: result[field] for field in _ATTESTATION_IDENTITY_FIELDS
    }


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProductionGroundTruthComparisonError(
            f"{label} must be a non-empty string"
        )
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ProductionGroundTruthComparisonError(f"{label} must be boolean")
    return value


def _sha256(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ProductionGroundTruthComparisonError(
            f"{label} must be a lowercase SHA256"
        )
    return text


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProductionGroundTruthComparisonError(
            f"{label} must be a positive integer"
        )
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProductionGroundTruthComparisonError(
            f"{label} must be a non-negative integer"
        )
    return value


def _positive_number(value: Any, label: str) -> float:
    number = _nonnegative_number(value, label)
    if number <= 0:
        raise ProductionGroundTruthComparisonError(f"{label} must be positive")
    return number


def _nonnegative_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ProductionGroundTruthComparisonError(
            f"{label} must be a finite non-negative number"
        )
    return float(value)


def _numeric_sequence(value: Any, label: str) -> list[float]:
    return [
        _nonnegative_number(item, f"{label}[{index}]")
        for index, item in enumerate(_sequence(value, label))
    ]
