"""Materialize one real 1P1D ShareGPT capture plan from a frozen case."""

from __future__ import annotations

from evaluation.json_shape import file_sha256

import argparse
import json
import os
from pathlib import Path
import shutil
import time
import uuid
from typing import Any, Mapping, Sequence

from evaluation.serving.experiments import sharegpt_ground_truth
from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS

from evaluation.json_shape import require_object_value as _mapping

from evaluation.json_shape import require_text as _text


_ROLES = ("prefill", "decode")
_ALLOWED_HARDWARE = EVALUATED_HARDWARE_IDS
_ALLOWED_CONNECTORS = {
    "nvidia_a800": ("P2pNcclConnector",),
    "metax_c500": ("P2pMcclConnector",),
    "ascend_910b": ("P2pHcclConnector", "MooncakeConnector"),
}
# The highest device ordinal each hardware exposes.  The executor validates
# the executed plan against the same map, so it is declared here and imported
# there rather than restated.
MAX_DEVICE_ID = {
    "nvidia_a800": 5,
    "metax_c500": 3,
    "ascend_910b": 5,
}
_MOONCAKE_ROUTE_ADAPTER = {
    "ascend_910b": "mooncake_vllm_0_18",
}
_CAPTURE_LIMIT_TOKENS = {
    "nvidia_a800": 8192,
    "metax_c500": 512,
    "ascend_910b": 512,
}
_KV_ROLE = {"prefill": "kv_producer", "decode": "kv_consumer"}
_KV_RANK = {"prefill": 0, "decode": 1}
_MOONCAKE_RUNTIME_ROOT_ENV = "MIDLAYER_MOONCAKE_RUNTIME_ROOT"
_COLOCATED_ENGINE_FIELDS = (
    "async_scheduling",
    "compilation_level",
    "configured_cudagraph_mode",
    "cpu_offload_gb",
    "cudagraph_metrics",
    "dtype",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "enforce_eager",
    "expert_parallel_size",
    "gpu_memory_utilization",
    "graph_mode",
    "kv_block_size_tokens",
    "max_concurrent_batches",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "pipeline_parallel_size",
    "safetensors_load_strategy",
    "safetensors_prefetch_num_threads",
    "scheduler_policy",
    "seed",
    "shutdown_timeout_seconds",
    "tensor_parallel_size",
    "vllm_engine_generation",
)
_A800_PD_CASES = {
    "llama-3.1-8b": ("tp1_qps4", "tp1_qps16", "tp1_qps64"),
    "qwen3-4b": ("tp1_qps4", "tp1_qps16", "tp1_qps64"),
    "qwen3-8b": ("tp1_qps4", "tp1_qps16", "tp1_qps64"),
    "qwen3-30b-a3b": ("tp2_ep2_qps1", "tp2_ep2_qps4", "tp2_ep2_qps16"),
    "qwen3-32b": ("tp2_ep1_qps1", "tp2_ep1_qps4", "tp2_ep1_qps16"),
}


def freeze_a800_pd_source_suite(
    *,
    dataset_root: str | Path,
    source_suite_paths: Sequence[str | Path],
    output_path: str | Path,
) -> Mapping[str, object]:
    """Freeze the current 15-case A800 source set as a schema-v3 suite.

    The existing historical suites retain their original LFS manifest object
    identities.  This capture-local snapshot records the current, checked-out
    contents that will actually be copied into each immutable PD raw run.
    """

    root = Path(dataset_root).resolve(strict=True)
    expected_root = (
        Path(__file__).resolve().parents[3] / "serving_dataset" / "sharegpt"
    ).resolve(strict=True)
    if root != expected_root:
        raise ValueError("PD source-suite dataset root must be canonical ShareGPT data")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to replace PD source suite: {output}")
    if not source_suite_paths:
        raise ValueError("PD source-suite freeze requires source suite manifests")
    source_models: dict[str, Mapping[str, object]] = {}
    source_hardware: Mapping[str, object] | None = None
    for suite_path_value in source_suite_paths:
        suite_path = Path(suite_path_value).resolve(strict=True)
        suite = _load_json_object(suite_path, "PD source suite")
        if suite["schema_version"] not in (3, 4):
            raise ValueError("PD source suite schema version is invalid")
        for model_value in _sequence(suite["models"], "PD source suite models"):
            model = _mapping(model_value, "PD source suite model")
            model_id = _text(model["dataset_model_id"], "PD source suite model ID")
            if model_id in source_models and source_models[model_id] != model:
                raise ValueError("PD source suites disagree about a model contract")
            source_models[model_id] = dict(model)
        matching_hardware = [
            _mapping(value, "PD source suite hardware")
            for value in _sequence(suite["hardware"], "PD source suite hardware")
            if _mapping(value, "PD source suite hardware")["hardware_id"] == "nvidia_a800"
        ]
        if len(matching_hardware) > 1:
            raise ValueError("PD source suite has duplicate A800 hardware contracts")
        if matching_hardware:
            hardware = matching_hardware[0]
            if source_hardware is not None and source_hardware != hardware:
                raise ValueError("PD source suites disagree about A800 hardware contract")
            source_hardware = dict(hardware)
    if source_hardware is None:
        raise ValueError("PD source suites have no A800 hardware contract")
    expected_cells = {
        (model_id, case_id)
        for model_id, case_ids in _A800_PD_CASES.items()
        for case_id in case_ids
    }
    manifest_paths = tuple(sorted((root / "nvidia_a800").glob("*/*/manifest.json")))
    cells: list[Mapping[str, object]] = []
    observed_cells: set[tuple[str, str]] = set()
    for manifest_path in manifest_paths:
        manifest = _load_json_object(manifest_path, "PD source case manifest")
        identity = _mapping(manifest["identity"], "PD source case identity")
        if identity["hardware_id"] != "nvidia_a800":
            raise ValueError("PD source case hardware is invalid")
        model_id = _text(identity["model_id"], "PD source case model ID")
        case_id = _text(identity["case_id"], "PD source case ID")
        cell_key = (model_id, case_id)
        if cell_key not in expected_cells:
            raise ValueError("PD source suite found an unsupported A800 case")
        if cell_key in observed_cells:
            raise ValueError("PD source suite found a duplicate A800 case")
        observed_cells.add(cell_key)
        requests = _mapping(manifest["requests"], "PD source case requests")
        requests_path = manifest_path.parent / _text(
            requests["path"], "PD source requests path"
        )
        if _sha256_file(requests_path) != requests["sha256"]:
            raise ValueError("PD source requests CSV disagrees with source case manifest")
        cells.append({
            "hardware_id": "nvidia_a800",
            "model_id": model_id,
            "case_id": case_id,
            "manifest_sha256": _sha256_file(manifest_path),
            "requests_sha256": _sha256_file(requests_path),
        })
    if observed_cells != expected_cells:
        raise ValueError("PD source suite does not contain exactly fifteen A800 cases")
    model_ids = tuple(sorted(_A800_PD_CASES))
    if set(source_models) & set(model_ids) != set(model_ids):
        raise ValueError("PD source suites omit a required model contract")
    suite = {
        "schema_version": 3,
        "suite_id": "a800_pd_capture_source_v1",
        "dataset_root": "serving_dataset/sharegpt",
        "models": [source_models[model_id] for model_id in model_ids],
        "hardware": [source_hardware],
        "qps_values": [1, 4, 16, 64],
        "cells": sorted(
            cells,
            key=lambda cell: (
                _text(cell["model_id"], "PD frozen model ID"),
                _text(cell["case_id"], "PD frozen case ID"),
            ),
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_new(output, suite)
    return suite


def materialize_pd_capture_plan(
    *,
    model: str | Path,
    dataset: str | Path,
    case_manifest: str | Path,
    suite_manifest: str | Path,
    colocated_ground_truth: str | Path,
    output_root: str | Path,
    prefill_devices: Sequence[int],
    decode_devices: Sequence[int],
    prefill_http_port: int,
    decode_http_port: int,
    router_port: int,
    prefill_kv_ip: str,
    decode_kv_ip: str,
    prefill_kv_port: int,
    decode_kv_port: int,
    connector: str,
    observer_effect_validation: Mapping[str, object],
    runtime_source_commit: str | None = None,
    trace_enabled: bool = True,
    canary_observer_pair: bool = False,
) -> Mapping[str, object]:
    """Create commands and raw inputs for one measured, frozen PD run.

    This only writes an immutable capture plan and copies the two frozen input
    files.  It never starts a process; callers must first record the Goal's
    sixty-second idle evidence before executing these commands.
    """

    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"refusing to replace PD output root: {root}")
    case = sharegpt_ground_truth._load_case_contract(
        case_manifest, model_path=model, suite_manifest_path=suite_manifest
    )
    hardware_id = case["identity"]["hardware_id"]
    if hardware_id not in _ALLOWED_HARDWARE:
        raise ValueError("PD capture only supports A800, C500, or Ascend 910B")
    if connector not in _ALLOWED_CONNECTORS[hardware_id]:
        raise ValueError("PD connector does not match the selected hardware")
    runtime = case["runtime"]
    colocated_settings = _colocated_execution_settings(
        colocated_ground_truth=colocated_ground_truth,
        case=case,
        model=model,
    )
    executor_source_commit = _executor_source_commit(
        runtime, runtime_source_commit)
    # The frozen source manifest is immutable evidence.  Its optional runtime
    # source identity can be absent in historical captures, so create an
    # execution-only contract for the live vLLM process rather than mutating
    # the source manifest or manufacturing a default identity.
    execution_case = {
        **case,
        "runtime": {**runtime, "source_commit": executor_source_commit},
    }
    p_devices = _validate_role_devices(
        prefill_devices, "prefill", runtime, hardware_id)
    d_devices = _validate_role_devices(
        decode_devices, "decode", runtime, hardware_id)
    if set(p_devices) & set(d_devices):
        raise ValueError("PD Prefill and Decode device allocations overlap")
    for port in (
        prefill_http_port, decode_http_port, router_port,
        prefill_kv_port, decode_kv_port,
    ):
        _validate_port(port)
    _validate_kv_port_ranges(
        prefill_kv_ip=prefill_kv_ip,
        decode_kv_ip=decode_kv_ip,
        prefill_kv_port=prefill_kv_port,
        decode_kv_port=decode_kv_port,
        prefill_rank_count=len(p_devices),
        decode_rank_count=len(d_devices),
        connector=connector,
    )
    if len({prefill_http_port, decode_http_port, router_port}) != 3:
        raise ValueError("PD HTTP ports must be distinct")
    for value, label in ((prefill_kv_ip, "prefill KV IP"), (decode_kv_ip, "decode KV IP")):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be non-empty")
    if not isinstance(trace_enabled, bool):
        raise TypeError("PD trace_enabled must be boolean")
    if not isinstance(canary_observer_pair, bool):
        raise TypeError("PD canary_observer_pair must be boolean")

    run_identity = _run_identity(case["identity"]["case_id"])
    raw_root = root / "raw"
    raw_root.mkdir(parents=True)
    frozen_manifest = raw_root / "frozen_manifest.json"
    frozen_requests = raw_root / "requests.csv"
    shutil.copyfile(case_manifest, frozen_manifest)
    shutil.copyfile(case["requests_path"], frozen_requests)
    p_trace = raw_root / "scheduler_steps_prefill.raw.jsonl"
    d_trace = raw_root / "scheduler_steps_decode.raw.jsonl"
    handoff_events = raw_root / "connector_events"
    handoff_events.mkdir()
    router_handoffs = raw_root / "router_handoffs.raw.jsonl"
    run_contract = raw_root / "run_contract.json"
    observer_effect = _observer_effect_for_case(
        observer_effect_validation,
        case["identity"],
        canary_observer_pair=canary_observer_pair,
    )
    observer_effect_path = raw_root / "observer_effect.json"
    _write_json_new(observer_effect_path, observer_effect)

    role_plans = {
        "prefill": _role_server_plan(
            model=model,
            case=execution_case,
            devices=p_devices,
            port=prefill_http_port,
            trace_path=p_trace,
            role="prefill",
            run_identity=run_identity,
            handoff_events=handoff_events,
            connector=connector,
            kv_ip=prefill_kv_ip,
            kv_port=prefill_kv_port,
            peer_kv_ip=decode_kv_ip,
            peer_kv_port=decode_kv_port,
            colocated_settings=colocated_settings,
            trace_enabled=trace_enabled,
        ),
        "decode": _role_server_plan(
            model=model,
            case=execution_case,
            devices=d_devices,
            port=decode_http_port,
            trace_path=d_trace,
            role="decode",
            run_identity=run_identity,
            handoff_events=handoff_events,
            connector=connector,
            kv_ip=decode_kv_ip,
            kv_port=decode_kv_port,
            peer_kv_ip=prefill_kv_ip,
            peer_kv_port=prefill_kv_port,
            colocated_settings=colocated_settings,
            trace_enabled=trace_enabled,
        ),
    }
    contract = _run_contract(
        case=execution_case,
        role_plans=role_plans,
        run_identity=run_identity,
        observer_effect_validation=observer_effect,
    )
    _write_json_new(run_contract, contract)
    parity_report = _setting_parity_report(
        case=case,
        colocated_ground_truth=colocated_ground_truth,
        settings=colocated_settings,
        role_plans=role_plans,
    )
    parity_path = raw_root / "setting_parity.json"
    _write_json_new(parity_path, parity_report)
    if parity_report["passed"] is not True:
        raise ValueError("PD setting parity differs outside the topology allowlist")
    router_argv = [
        os.fspath(Path(__file__).resolve().parents[3] / "scripts" / "experiments" / "request_router.py"),
        "--host", "127.0.0.1", "--port", str(router_port),
        "--prefill-base-url", role_plans["prefill"]["endpoint"]["base_url"],
        "--decode-base-url", role_plans["decode"]["endpoint"]["base_url"],
        "--model", case["identity"]["model_id"],
        "--handoff-transport", _handoff_transport(connector),
        "--prefill-kv-ip", prefill_kv_ip,
        "--prefill-kv-port", str(prefill_kv_port),
        "--decode-kv-ip", decode_kv_ip,
        "--decode-kv-port", str(decode_kv_port),
        "--handoff-log", str(router_handoffs),
        "--run-id", run_identity["run_id"],
        "--run-origin-monotonic-ns", str(run_identity["run_origin_monotonic_ns"]),
        "--host-boot-id", run_identity["host_boot_id"],
    ]
    benchmark_argv = [
        os.fspath(Path(__file__).resolve().parents[3] / "scripts" / "experiments" / "run_sharegpt_pd_ground_truth.py"),
        "--model", str(Path(model).resolve(strict=True)),
        "--dataset", str(Path(dataset).resolve(strict=True)),
        "--case-manifest", str(Path(case_manifest).resolve(strict=True)),
        "--suite-manifest", str(Path(suite_manifest).resolve(strict=True)),
        "--router-base-url", f"http://127.0.0.1:{router_port}",
        "--run-contract", str(run_contract),
        "--output", str(raw_root / "ground_truth.json"),
        "--seed", str(colocated_settings["seed"]),
        "--max-model-len", str(colocated_settings["max_model_len"]),
        "--kv-block-size-tokens", str(colocated_settings["kv_block_size_tokens"]),
        "--trace-enabled" if trace_enabled else "--no-trace-enabled",
    ]
    plan = {
        "schema_version": 1,
        "kind": "sharegpt_pd_capture_plan",
        "identity": contract["identity"],
        "run_contract": str(run_contract),
        "raw_root": str(raw_root),
        "devices": {"prefill": list(p_devices), "decode": list(d_devices)},
        "commands": {
            "prefill": role_plans["prefill"],
            "decode": role_plans["decode"],
            "router": {
                "argv": [os.fspath(Path(os.sys.executable)), *router_argv],
                "environment": {
                    "ASCEND_RT_VISIBLE_DEVICES": "",
                    "CUDA_VISIBLE_DEVICES": "",
                    "MACA_VISIBLE_DEVICES": "",
                },
                "endpoint": {
                    "base_url": f"http://127.0.0.1:{router_port}",
                    "health_url": f"http://127.0.0.1:{router_port}/v1/models",
                },
            },
            "benchmark": {"argv": [os.fspath(Path(os.sys.executable)), *benchmark_argv]},
        },
        "artifacts": {
            "frozen_manifest": str(frozen_manifest),
            "frozen_requests": str(frozen_requests),
            "prefill_scheduler": None if not trace_enabled else str(p_trace),
            "decode_scheduler": None if not trace_enabled else str(d_trace),
            "router_handoffs": str(router_handoffs),
            "connector_events_directory": str(handoff_events),
            "ground_truth": str(raw_root / "ground_truth.json"),
            "observer_effect": str(observer_effect_path),
            "setting_parity": str(parity_path),
        },
        "profiler_attached": False,
    }
    _write_json_new(root / "capture_plan.json", plan)
    return plan


def _observer_effect_for_case(
    validation: Mapping[str, object],
    identity: Mapping[str, object],
    *,
    canary_observer_pair: bool,
) -> Mapping[str, object]:
    result = dict(validation)
    current_case_id = "/".join(
        _text(identity[field], f"PD identity {field}")
        for field in ("hardware_id", "model_id", "case_id")
    )
    if result["status"] == "pending_canary_comparison":
        if not canary_observer_pair:
            raise ValueError(
                "pending observer effect is valid only for a canary pair"
            )
        if set(result) != {"status"}:
            raise ValueError("pending canary observer effect has extra fields")
        result["validation_case_id"] = current_case_id
        return result
    if canary_observer_pair:
        raise ValueError("canary pair requires pending observer-effect evidence")
    if result["status"] != "measured_for_this_case":
        raise ValueError("PD capture requires a measured observer-effect report")
    validation_case_id = result["validation_case_id"]
    if not isinstance(validation_case_id, str) or not validation_case_id:
        raise ValueError("PD observer-effect validation_case_id must be text")
    if validation_case_id != current_case_id:
        result["status"] = "inherited_from_representative_case"
    return result


def _colocated_execution_settings(
    *,
    colocated_ground_truth: str | Path,
    case: Mapping[str, object],
    model: str | Path,
) -> Mapping[str, object]:
    """Load the measured colocated execution contract for one PD case.

    Compact case manifests deliberately do not repeat every vLLM launch
    setting.  A PD run must therefore inherit them from the measured
    colocated ground truth, rather than reintroducing PD-specific CLI
    defaults.  The source is validated before any immutable PD output is
    created, so a missing or mismatched setting fails closed.
    """

    source_path = Path(colocated_ground_truth).resolve(strict=True)
    source = _load_json_object(source_path, "colocated ground truth")
    if source["schema_version"] != 1 or source["status"] != "success":
        raise ValueError("colocated ground truth must be a successful schema-v1 run")
    workload = _mapping(source["workload"], "colocated workload")
    source_identity = _mapping(workload["identity"], "colocated identity")
    case_identity = _mapping(case["identity"], "source case identity")
    if dict(source_identity) != dict(case_identity):
        raise ValueError("colocated ground-truth identity differs from source case")
    runtime = _mapping(case["runtime"], "source case runtime")
    if workload["request_count"] != case["requests_row_count"]:
        raise ValueError("colocated ground-truth request count differs from source case")
    if workload["expected_qps"] != runtime["target_qps"]:
        raise ValueError("colocated ground-truth QPS differs from source case")
    if workload["seed"] != runtime["seed"]:
        raise ValueError("colocated ground-truth seed differs from source case")
    if workload["requests_sha256"] != case["requests_sha256"]:
        raise ValueError("colocated ground-truth requests differ from source case")
    server = _mapping(source["server"], "colocated server")
    if Path(_text(server["model_path"], "colocated model path")).resolve(
        strict=True
    ) != Path(model).resolve(strict=True):
        raise ValueError("colocated ground-truth model path differs from PD model")
    if server["client_vllm_version"] != runtime["runtime_version"]:
        raise ValueError("colocated ground-truth runtime version differs from source case")
    trace = _mapping(server["scheduler_trace"], "colocated Scheduler trace")
    if trace["enabled"] is not True:
        raise ValueError("colocated ground truth must include Scheduler trace evidence")
    profiler = _mapping(server["profiler"], "colocated profiler")
    if profiler["requested"] is not False:
        raise ValueError("colocated ground truth must not attach a profiler")
    engine = _mapping(server["engine"], "colocated engine")
    _exact_keys(engine, _COLOCATED_ENGINE_FIELDS, "colocated engine")
    for field in (
        "dtype",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "expert_parallel_size",
        "seed",
    ):
        if engine[field] != runtime[field]:
            raise ValueError(f"colocated engine {field} differs from source case")
    if engine["enable_prefix_caching"] is not runtime["capture_enable_prefix_caching"]:
        raise ValueError("colocated engine prefix-cache setting differs from source case")
    for field in (
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "kv_block_size_tokens",
        "seed",
    ):
        value = engine[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"colocated engine {field} must be positive")
    return {
        **dict(engine),
        "colocated_ground_truth_path": str(source_path),
        "colocated_ground_truth_sha256": _sha256_file(source_path),
    }


def _setting_parity_report(
    *,
    case: Mapping[str, object],
    colocated_ground_truth: str | Path,
    settings: Mapping[str, object],
    role_plans: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    """Return a fail-closed comparison of every non-topology engine setting."""

    source_path = Path(colocated_ground_truth).resolve(strict=True)
    source = _load_json_object(source_path, "colocated ground truth")
    expected_engine = _mapping(
        _mapping(source["server"], "colocated server")["engine"],
        "colocated engine",
    )
    _exact_keys(expected_engine, _COLOCATED_ENGINE_FIELDS, "colocated engine")
    differences: dict[str, list[str]] = {}
    for role in _ROLES:
        actual_engine = _mapping(role_plans[role]["engine"], f"PD {role} engine")
        _exact_keys(actual_engine, _COLOCATED_ENGINE_FIELDS, f"PD {role} engine")
        role_differences = [
            field for field in _COLOCATED_ENGINE_FIELDS
            if actual_engine[field] != expected_engine[field]
        ]
        differences[role] = role_differences
    if settings["colocated_ground_truth_sha256"] != _sha256_file(source_path):
        raise ValueError("colocated ground truth changed while materializing PD plan")
    return {
        "schema_version": 1,
        "kind": "sharegpt_pd_setting_parity",
        "identity": dict(_mapping(case["identity"], "source case identity")),
        "source": {
            "ground_truth_path": str(source_path),
            "ground_truth_sha256": settings["colocated_ground_truth_sha256"],
            "requests_sha256": case["requests_sha256"],
        },
        "allowed_differences": [
            "deployment_topology",
            "role_devices",
            "http_ports",
            "kv_connector",
            "kv_endpoints",
            "run_identity",
            "trace_output_paths",
        ],
        "engine_fields": list(_COLOCATED_ENGINE_FIELDS),
        "differences": differences,
        "passed": all(not fields for fields in differences.values()),
    }


def _load_json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is invalid JSON") from error
    return _mapping(value, label)


def _sha256_file(path: Path) -> str:
    return file_sha256(path)


def _role_server_plan(
    *,
    model: str | Path,
    case: Mapping[str, Any],
    devices: tuple[int, ...],
    port: int,
    trace_path: Path,
    role: str,
    run_identity: Mapping[str, object],
    handoff_events: Path,
    connector: str,
    kv_ip: str,
    kv_port: int,
    peer_kv_ip: str,
    peer_kv_port: int,
    colocated_settings: Mapping[str, object],
    trace_enabled: bool,
) -> Mapping[str, object]:
    args = argparse.Namespace(
        model=str(model), devices=list(devices), port=port,
        seed=colocated_settings["seed"],
        gpu_memory_utilization=colocated_settings["gpu_memory_utilization"],
        max_model_len=colocated_settings["max_model_len"],
        max_num_batched_tokens=colocated_settings["max_num_batched_tokens"],
        max_num_seqs=colocated_settings["max_num_seqs"],
        kv_block_size_tokens=colocated_settings["kv_block_size_tokens"],
        max_dispatch_lag_ms=100.0,
        scheduler_trace_output=(str(trace_path) if trace_enabled else None),
        a800_compilation_counter_output=None,
    )
    base = sharegpt_ground_truth._server_plan(args, dict(case))
    engine = base["engine"]
    if engine["async_scheduling"] is not True or engine["max_concurrent_batches"] != 2:
        raise ValueError("PD server plan did not explicitly configure depth=2")
    if engine["enable_prefix_caching"] is not True or engine["cpu_offload_gb"] not in (None, 0):
        raise ValueError("PD server plan violates prefix-cache or offload contract")
    environment = dict(base["environment"])
    environment.update({
        "MIDLAYER_PD_RUN_ID": _text(run_identity["run_id"], "PD run ID"),
        "MIDLAYER_PD_RUN_ORIGIN_MONOTONIC_NS": str(run_identity["run_origin_monotonic_ns"]),
        "MIDLAYER_PD_HOST_BOOT_ID": _text(run_identity["host_boot_id"], "PD host boot ID"),
        "MIDLAYER_PD_INSTANCE_ROLE": role,
        "MIDLAYER_PD_INSTANCE_ID": f"{run_identity['run_id']}:{role}:0",
        "MIDLAYER_PD_HANDOFF_EVENTS_DIR": str(handoff_events),
        "MIDLAYER_PD_HANDOFF_CONNECTOR": connector,
    })
    if trace_enabled:
        # The bootstrap installs rich Scheduler instrumentation before the
        # Executor exists.  Carry the source-defined planned bound here; the
        # emitted runtime sidecar remains the acceptance evidence for the
        # live Executor's actual max_concurrent_batches value.
        environment["MIDLAYER_SCHEDULER_TRACE_MAX_IN_FLIGHT"] = str(
            engine["max_concurrent_batches"]
        )
    if connector == "P2pHcclConnector":
        # The verified private ProcessGroupHCCL smoke uses the same local
        # device-P2P policy.  This is a transport requirement, not a Host
        # fallback: KV payloads still stay in NPU memory end to end.
        environment["HCCL_WHITELIST_DISABLE"] = "1"
    if connector == "P2pNcclConnector":
        # vLLM 0.11's P2pNcclConnector does not consume KVTransferConfig's
        # kv_ip when it binds its ZMQ listener.  It calls vllm.utils.get_ip()
        # instead, which is controlled by VLLM_HOST_IP.  Keep the listener
        # address identical to the explicit address placed in the paired
        # request IDs by the router; otherwise NCCL rank 0 waits forever for
        # a peer at a different address.
        environment["VLLM_HOST_IP"] = kv_ip
    if connector == "MooncakeConnector":
        # Mooncake obtains its producer bootstrap endpoint from this runtime
        # setting rather than KVTransferConfig.kv_port.  It must therefore be
        # explicit and unique per Prefill instance; the Decode role receives
        # the observed endpoint in the producer's response.
        if role == "prefill":
            environment["VLLM_MOONCAKE_BOOTSTRAP_PORT"] = str(kv_port)
            hardware_id = _text(
                case["identity"]["hardware_id"], "PD hardware ID")
            environment["MIDLAYER_PD_ROUTE_ADAPTER"] = (
                _MOONCAKE_ROUTE_ADAPTER[hardware_id]
            )
        # The deployed Mooncake Python extension links ascend_transport.so
        # from /usr/local/lib.  docker exec's login shell omits that directory;
        # preserve the inherited loader path while making the runtime library
        # location explicit for the server process.
        if case["identity"]["hardware_id"] == "ascend_910b":
            runtime = _ascend_mooncake_runtime()
            environment.update(runtime["environment"])
    if connector in (
        "P2pNcclConnector",
        "P2pMcclConnector",
        "MooncakeConnector",
        "P2pHcclConnector",
    ):
        bootstrap_paths = [
            str(sharegpt_ground_truth._SCHEDULER_TRACE_BOOTSTRAP),
            str(sharegpt_ground_truth.ROOT),
        ]
        if connector == "MooncakeConnector":
            runtime_package = _text(
                _ascend_mooncake_runtime()["runtime_package"],
                "Mooncake runtime package",
            )
            bootstrap_paths.append(runtime_package)
        if case["identity"]["hardware_id"] == "ascend_910b":
            for path in (
                f"{environment['ASCEND_HOME_PATH']}/python/site-packages",
                f"{environment['ASCEND_OPP_PATH']}/built-in/op_impl/ai_core/tbe",
            ):
                if path not in bootstrap_paths:
                    bootstrap_paths.append(path)
        if "PYTHONPATH" in environment and environment["PYTHONPATH"]:
            for path in environment["PYTHONPATH"].split(os.pathsep):
                if path not in bootstrap_paths:
                    bootstrap_paths.append(path)
        environment["PYTHONPATH"] = os.pathsep.join(bootstrap_paths)
    kv_config = {
        "kv_connector": connector,
        "engine_id": f"{run_identity['run_id']}:{role}:engine",
        "kv_role": _KV_ROLE[role],
        "kv_rank": _KV_RANK[role],
        "kv_parallel_size": 2,
        "kv_ip": kv_ip,
        "kv_port": kv_port,
        "kv_connector_extra_config": _connector_extra_config(
            connector, role, endpoint_port=port, peer_ip=peer_kv_ip,
            peer_port=peer_kv_port,
            hardware_id=_text(case["identity"]["hardware_id"], "PD hardware ID"),
        ),
    }
    if connector == "P2pMcclConnector":
        kv_config["kv_buffer_size"] = 8 * 1024**3
        kv_config["kv_connector_module_path"] = (
            "dataset_pipeline.serving_dataset.metax_p2p_mccl"
        )
    if connector == "P2pHcclConnector":
        kv_config["kv_buffer_size"] = 8 * 1024**3
        kv_config["kv_connector_module_path"] = (
            "dataset_pipeline.serving_dataset.ascend_p2p_hccl"
        )
    return {
        "argv": [*base["argv"], "--kv-transfer-config", json.dumps(kv_config, sort_keys=True)],
        "environment": environment,
        "endpoint": base["endpoint"],
        "engine": engine,
        "scheduler_trace": base["scheduler_trace"],
        "profiler_attached": False,
    }


def _ascend_mooncake_runtime() -> Mapping[str, object]:
    if _MOONCAKE_RUNTIME_ROOT_ENV not in os.environ:
        raise RuntimeError(
            f"Ascend Mooncake requires {_MOONCAKE_RUNTIME_ROOT_ENV} to name the verified runtime snapshot"
        )
    root_value = os.environ[_MOONCAKE_RUNTIME_ROOT_ENV]
    if not root_value:
        raise RuntimeError("Ascend Mooncake runtime root is empty")
    root = Path(root_value).resolve(strict=True)
    runtime_package = root / "runtime_package"
    library = runtime_package / "mooncake"
    shim = library / "libmooncake_hccl_abi_shim.so"
    hccn = root / "hccn.conf"
    for path, label in (
        (runtime_package, "Mooncake runtime package"),
        (library, "Mooncake runtime library"),
        (shim, "Mooncake HCCL ABI shim"),
        (hccn, "Mooncake HCCN configuration"),
    ):
        if not path.exists():
            raise RuntimeError(f"{label} is absent: {path}")
    if "LD_LIBRARY_PATH" not in os.environ or not os.environ["LD_LIBRARY_PATH"]:
        raise RuntimeError("Ascend Mooncake runtime requires inherited LD_LIBRARY_PATH")
    return {
        "runtime_package": str(runtime_package),
        "environment": {
            _MOONCAKE_RUNTIME_ROOT_ENV: str(root),
            "MIDLAYER_MOONCAKE_HCCN_CONF": str(hccn),
            "LD_PRELOAD": str(shim),
            "LD_LIBRARY_PATH": f"{library}{os.pathsep}{os.environ['LD_LIBRARY_PATH']}",
        },
    }


def _connector_extra_config(
    connector: str,
    role: str,
    *,
    endpoint_port: int,
    peer_ip: str,
    peer_port: int,
    hardware_id: str,
) -> Mapping[str, object]:
    if connector == "P2pNcclConnector":
        return {
            "http_ip": "127.0.0.1",
            "http_port": str(endpoint_port),
            # P2pNcclConnector dispatches on each local engine's send_type.
            # A Decode GET is served from Prefill's send_store, so both ends
            # must use GET.  Mixing Prefill PUT_ASYNC with Decode GET makes
            # Prefill receive a GET request without a send_store.
            "send_type": "GET",
            "peer_ip": peer_ip,
            "peer_port": str(peer_port),
        }
    if connector == "P2pMcclConnector":
        if hardware_id != "metax_c500":
            raise ValueError("P2pMcclConnector is only valid for metax_c500")
        return {
            "http_port": str(endpoint_port),
            "send_type": "PUT_ASYNC",
            "nccl_num_channels": "8",
        }
    if connector == "P2pHcclConnector":
        if hardware_id != "ascend_910b":
            raise ValueError("P2pHcclConnector is only valid for ascend_910b")
        return {"send_type": "PUT_ASYNC", "pg_timeout_s": 120}
    if connector == "MooncakeConnector":
        return {
            "mooncake_protocol": "rdma",
            "num_workers": 10,
        }
    raise AssertionError("PD connector was validated")


def _run_contract(
    *,
    case: Mapping[str, Any],
    role_plans: Mapping[str, Mapping[str, object]],
    run_identity: Mapping[str, object],
    observer_effect_validation: Mapping[str, object],
) -> Mapping[str, object]:
    source_identity = _mapping(case["identity"], "source identity")
    source_runtime = _mapping(case["runtime"], "source runtime")
    instances = {}
    for role in _ROLES:
        engine = _mapping(role_plans[role]["engine"], f"PD {role} engine")
        configured_mode = engine["configured_cudagraph_mode"]
        if not isinstance(configured_mode, str) or not configured_mode:
            raise ValueError("PD server plan has no explicit Graph mode")
        instances[role] = {
            "tensor_parallel_size": engine["tensor_parallel_size"],
            "expert_parallel_size": engine["expert_parallel_size"],
            "pipeline_parallel_size": engine["pipeline_parallel_size"],
            "graph": {
                "configured_mode": configured_mode,
                "capture_limit_tokens": _CAPTURE_LIMIT_TOKENS[
                    source_identity["hardware_id"]
                ],
            },
        }
    identity = {
        "source_kind": "sharegpt_pd_disaggregated",
        "topology": "1p1d",
        "case_id": source_identity["case_id"],
        "hardware_id": source_identity["hardware_id"],
        "model_id": source_identity["model_id"],
        "raw_model_id": source_identity["raw_model_id"],
        "model_revision": source_identity["model_revision"],
        "tokenizer_revision": source_identity["tokenizer_revision"],
    }
    return {
        "schema_version": 1,
        "kind": "sharegpt_pd_run_contract",
        "identity": identity,
        "runtime": {
            "dtype": source_runtime["dtype"],
            "deployment": "pd_disaggregated",
            "target_qps": source_runtime["target_qps"],
            "seed": source_runtime["seed"],
            "runtime_version": source_runtime["runtime_version"],
            "source_commit": source_runtime["source_commit"],
            "capture_enable_prefix_caching": source_runtime[
                "capture_enable_prefix_caching"
            ],
            "instances": instances,
        },
        "capture_identity": dict(run_identity),
        "observer_effect_validation": dict(observer_effect_validation),
        "provenance_gaps": [],
    }


def _executor_source_commit(
    runtime: Mapping[str, object], explicit_source_commit: str | None,
) -> str:
    observed = runtime["source_commit"]
    if observed is not None:
        if not isinstance(observed, str) or not observed:
            raise ValueError("source runtime source_commit is invalid")
    if explicit_source_commit is not None:
        if not _is_full_commit(explicit_source_commit):
            raise ValueError("explicit runtime_source_commit must be a full commit")
        return explicit_source_commit
    if isinstance(observed, str) and _is_full_commit(observed):
        return observed
    raise ValueError(
        "PD Scheduler trace requires explicit runtime_source_commit when "
        "the frozen source runtime has no full source commit"
    )


def _run_identity(case_id: object) -> Mapping[str, object]:
    case = _text(case_id, "case ID")
    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    if not boot_id_path.is_file():
        raise RuntimeError("PD capture requires a host boot ID")
    boot_id = boot_id_path.read_text(encoding="utf-8").strip()
    _text(boot_id, "host boot ID")
    return {
        "run_id": f"pd-{case}-{uuid.uuid4().hex}",
        "run_origin_monotonic_ns": time.perf_counter_ns(),
        "host_boot_id": boot_id,
    }


def _validate_role_devices(
    devices: Sequence[int],
    role: str,
    runtime: Mapping[str, object],
    hardware_id: str,
) -> tuple[int, ...]:
    if isinstance(devices, (str, bytes, bytearray)) or not isinstance(devices, Sequence):
        raise TypeError(f"PD {role} devices must be a sequence")
    expected = runtime["tensor_parallel_size"]
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        raise ValueError("source runtime has an invalid TP size")
    values = tuple(devices)
    if len(values) != expected or len(set(values)) != len(values):
        raise ValueError(f"PD {role} device count must equal TP without duplicates")
    maximum = MAX_DEVICE_ID[hardware_id]
    for device in values:
        if (
            isinstance(device, bool)
            or not isinstance(device, int)
            or not 0 <= device <= maximum
        ):
            raise ValueError(
                f"PD {role} on {hardware_id} may only use device IDs "
                f"0 through {maximum}"
            )
    return values


def _handoff_transport(connector: str) -> str:
    if connector in ("P2pNcclConnector", "P2pMcclConnector", "P2pHcclConnector"):
        return "p2p_request_id"
    if connector == "MooncakeConnector":
        return "openai_kv_transfer_params"
    raise AssertionError("PD connector was validated")


def _validate_port(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError("PD port must be in [1, 65535]")


def _validate_kv_port_ranges(
    *,
    prefill_kv_ip: str,
    decode_kv_ip: str,
    prefill_kv_port: int,
    decode_kv_port: int,
    prefill_rank_count: int,
    decode_rank_count: int,
    connector: str,
) -> None:
    if connector not in ("P2pNcclConnector", "P2pMcclConnector", "P2pHcclConnector"):
        return
    prefill_end = prefill_kv_port + prefill_rank_count - 1
    decode_end = decode_kv_port + decode_rank_count - 1
    if prefill_end > 65535 or decode_end > 65535:
        raise ValueError("PD per-rank KV port range exceeds 65535")
    if prefill_kv_ip == decode_kv_ip and not (
        prefill_end < decode_kv_port or decode_end < prefill_kv_port
    ):
        raise ValueError("PD Prefill and Decode per-rank KV port ranges overlap")


def _git_commit() -> str:
    import subprocess

    repository_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ("git", "rev-parse", "--verify", "HEAD"),
        cwd=repository_root, check=False,
        capture_output=True, text=True,
    )
    if completed.returncode == 0:
        commit = completed.stdout.strip()
        if _is_full_commit(commit):
            return commit
        raise RuntimeError("git did not return a full commit")
    snapshot_identity = repository_root / "SOURCE_COMMIT"
    if snapshot_identity.is_file():
        commit = snapshot_identity.read_text(encoding="utf-8").strip()
        if _is_full_commit(commit):
            return commit
        raise RuntimeError("SOURCE_COMMIT does not contain a full commit")
    raise RuntimeError("cannot determine PD source commit")


def _is_full_commit(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _write_json_new(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace PD artifact: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")




def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence")
    return value


def _exact_keys(value: Mapping[str, object], expected: Sequence[str], label: str) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{label} field set is invalid")
