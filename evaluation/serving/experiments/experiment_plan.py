"""Materialize deterministic workload, command, output, and provenance plans."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from evaluation.serving.experiments.experiment_manifest import (
    ExperimentCell,
    ManifestValidationError,
    ProductionTraceSweepCell,
    SimulationCell,
    VLLM_CONFORMANCE_SCHEDULER_DEFAULTS,
    validate_sampling_config,
)
from src.serving.backend.vllm import (
    VLLMSchedulerConfig,
    mtp_scenario_manifest,
)
from src.serving.backend.vllm_scheduler_process import (
    VLLM_SCHEDULER_SOURCE_COMMIT,
    VLLM_SCHEDULER_VERSION,
)
from src.kernel.hardware_spec import HWSpec
from src.serving.workload.expert_frequency import (
    load_expert_frequency_profile,
)
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import (
    ExpertRoutingInput,
    load_expert_routing_replay,
)
from evaluation.serving.simulation.execution_artifact import (
    performance_timing_status,
)


@dataclass(frozen=True)
class WorkloadArtifact:
    version: int
    workload: dict[str, Any]
    request_ids: list[str]
    prompts: list[str]
    prompt_lengths: list[int]
    sampling: dict[str, Any]
    token_controls: dict[str, Any]
    num_requests: int
    requested_num_requests: int
    seed: int
    hash: str
    path: Path | None = None

    @property
    def name(self) -> str:
        return self.workload["name"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "workload": dict(self.workload),
            "request_ids": list(self.request_ids),
            "prompts": list(self.prompts),
            "prompt_lengths": list(self.prompt_lengths),
            "sampling": dict(self.sampling),
            "token_controls": dict(self.token_controls),
            "num_requests": self.num_requests,
            "requested_num_requests": self.requested_num_requests,
            "seed": self.seed,
            "hash": self.hash,
        }

PROJECT_ROOT = ROOT


_SIMULATION_PREVIEW_FIELDS = (
    "backend", "deployment", "prediction_method", "launch_policy",
    "serving_timing_mode", "execution_mode", "execution_workers", "step_retention",
    "graph_mode", "graph_capture_sizes", "prefix_cache_mode", "enable_prefix_caching",
    "enable_chunked_prefill", "max_model_len", "max_batched_tokens", "max_requests",
    "kv_block_size_tokens", "prefill_tp_size", "prefill_pp_size", "decode_tp_size",
    "ffn_tp_size", "num_prefill_instances", "num_decode_instances", "enable_expert_parallel",
    "vllm_async_scheduling", "mtp_num_speculative_tokens",
    "vllm_scheduler_version", "attention_kv_layout", "attention_kv_distribution",
)


_SIMULATION_EXPLICIT_FILE_OPTIONS = frozenset({
    "workload", "production-trace", "canonical-requests", "model",
    "hardware-config", "decode-hardware-config", "additional-colocated-hardware-config",
    "runtime-memory-policy", "kv-lifecycle-replay", "mtp-acceptance-replay",
    "mtp-draft-token-ids-by-step", "expert-routing-replay", "expert-routing-profile-manifest",
    "host-launch-profile", "scheduler-host-profile", "host-request-path-profile",
    "host-runtime-prepare-profile", "slo-config", "interactivity-tgs-config",
    "vllm-sdk-python", "vllm-source-root",
})


def simulation_explicit_input_files(options: dict[str, Any]) -> dict[str, Any]:
    """Observe explicit files with the existing file provenance owner.

    This deliberately does not assert complete dependency coverage: directory
    inputs, implicit profiles, and the Python environment need separate identity.
    """
    from evaluation.serving.simulation.simulation_artifacts import directory_provenance, file_provenance

    files = {}
    for name in sorted(_SIMULATION_EXPLICIT_FILE_OPTIONS & options.keys()):
        raw = options[name]
        values = raw if isinstance(raw, list) else [raw]
        entries = []
        for value in values:
            if not isinstance(value, str):
                raise ManifestValidationError(f"{name} must name a file or directory")
            path = Path(value).resolve()
            if path.is_file():
                size = path.stat().st_size
                if size <= 64 * 1024 * 1024:
                    entries.append({"status": "file", "provenance": file_provenance(path)})
                else:
                    entries.append({"status": "large_file_not_fingerprinted", "path": str(path), "size_bytes": size})
            elif path.is_dir():
                if name == "vllm-source-root" and (path / ".git").exists():
                    # Use the Scheduler owner's package manifest so ignored
                    # Python sources are observed as well as Git contents.
                    from src.serving.backend.vllm_scheduler_process import _package_python_manifest

                    package_root = path / "vllm"
                    if package_root.is_dir():
                        count, digest = _package_python_manifest(package_root)
                        python_sources = {"status": "observed", "file_count": count,
                                          "manifest_sha256": digest}
                        package_entries = tuple(package_root.rglob("*"))
                        if any(candidate.is_symlink() for candidate in package_entries):
                            package_auxiliary = {"status": "symlinks_not_fingerprinted"}
                        elif any(candidate.is_file() for candidate in package_entries):
                            auxiliary = directory_provenance(
                                package_root, allowed_suffixes=("*",))
                            package_auxiliary = {"status": "observed",
                                "file_count": auxiliary["file_count"],
                                "tree_sha256": auxiliary["tree_sha256"],
                                "scope": "all_regular_package_files"}
                        else:
                            package_auxiliary = {"status": "no_package_files"}
                    else:
                        python_sources = {"status": "package_directory_missing"}
                        package_auxiliary = {"status": "package_directory_missing"}
                    entries.append({"status": "git_directory_partial_identity",
                                    "identity": git_code_identity(path),
                                    "scope": "git_contents_and_vllm_package_files",
                                    "package_python_sources": python_sources,
                                    "package_auxiliary": package_auxiliary})
                else:
                    entries.append({"status": "directory_not_fingerprinted", "path": str(path)})
            else:
                entries.append({"status": "unavailable", "path": str(path)})
            if name == "vllm-sdk-python":
                # A venv's interpreter is commonly a symlink to its base
                # Python; resolving that link does not bind the venv config.
                venv_config = Path(value).absolute().parent.parent / "pyvenv.cfg"
                if venv_config.is_file():
                    entries.append({"status": "file", "provenance": file_provenance(venv_config)})
                else:
                    entries.append({"status": "venv_config_not_present", "path": str(venv_config)})
        files[name] = entries
    return {"scope": "explicit_file_arguments_only_not_complete_dependency_identity", "files": files}


def preview_simulation_requests(args: Any) -> dict[str, Any]:
    """Summarize small inputs using the same loader used by actual execution."""
    from evaluation.serving.simulation.simulation_artifacts import load_request_source
    from src.serving.request_gen.request_stream import request_stream_sha256

    if args.workload is not None:
        kind, source = "workload", args.workload
    elif args.production_trace is not None:
        kind, source = "production_trace", args.production_trace
    elif args.canonical_requests is not None:
        kind, source = "canonical_requests", args.canonical_requests
    else:
        raise ValueError("simulation request source is missing")
    source_path = Path(source)
    if source_path.stat().st_size > 64 * 1024 * 1024:
        return {"status": "not_materialized_large_input", "kind": kind,
                "path": str(source_path), "reason": "preview avoids loading input larger than 64 MiB"}
    stream, arrival, workload_hash, _, _ = load_request_source(args)
    requests = stream.requests
    preview = {
        "status": "input_parsed_not_executed", "kind": kind, "path": str(source_path),
        "workload_hash": workload_hash, "request_stream_sha256": request_stream_sha256(stream),
        "request_count": len(requests),
        "arrival_time_range_s": [min(r.arrival_time_s for r in requests),
                                 max(r.arrival_time_s for r in requests)],
        "input_tokens_range": [min(r.input_tokens for r in requests), max(r.input_tokens for r in requests)],
        "output_tokens_range": [min(r.output_tokens for r in requests), max(r.output_tokens for r in requests)],
        "declared_prefix_hit_tokens": sum(r.prefix_hit_tokens for r in requests),
        "prefix_hit_source": ("synthetic_length_only_zero_assumption" if kind == "workload"
                              else "input_field_not_runtime_observation"),
        "effective_prefix_hits": "unknown_until_execution",
        "input_generation_seed": "unknown_or_not_applicable",
        "prompt_token_ids_present_request_count": sum(r.prompt_token_ids is not None for r in requests),
        "output_token_ids_present_request_count": sum(r.output_token_ids is not None for r in requests),
        "arrival_process": arrival,
    }
    if kind == "workload":
        artifact = load_workload_artifact(source_path)
        preview["input_generation_seed"] = artifact.seed
        preview["workload_generation"] = {"seed": artifact.seed,
            "requested_num_requests": artifact.requested_num_requests,
            "token_controls": artifact.token_controls}
    return preview


def materialize_cpu_simulation_plan(cell: SimulationCell, cell_root: Path) -> dict[str, Any]:
    """Use the simulation CLI as the single owner of option normalization."""
    from evaluation.serving.simulation.simulation_runner import _parse_args
    from evaluation.serving.simulation.simulation_execution import use_streaming_step_retention

    options = []
    for key in sorted(cell.options):
        value = cell.options[key]
        options.append(f"--{key}")
        if value is None:
            continue
        if isinstance(value, list):
            options.extend(str(item) for item in value)
        else:
            options.append(str(value))
    result_path = (cell_root / "result.json").resolve()
    summary_path = (cell_root / "settings.md").resolve()
    options.extend(("--output", str(result_path), "--settings-summary", str(summary_path), "--quiet"))
    try:
        parsed = _parse_args(options)
        external_request_source = (parsed.production_trace is not None
                                   or parsed.canonical_requests is not None)
        use_streaming_step_retention(parsed, True if external_request_source else None)
    except SystemExit as error:
        raise ManifestValidationError(f"{cell.id}: invalid simulation settings") from error
    return {
        "cell_id": cell.id,
        "execution_type": "cpu_simulation",
        "requested": cell.options,
        "normalized_preview": {name: vars(parsed)[name] for name in _SIMULATION_PREVIEW_FIELDS},
        "request_preview": preview_simulation_requests(parsed),
        "command": [sys.executable, "-m", "scripts.evaluation.serving.simulation_runner_entrypoint", *options],
        "output": str(result_path),
        "settings_summary": str(summary_path),
        "report_enabled": "interactive-report" in cell.options,
        "interactive_report": str(Path(str(cell.options["interactive-output"])).resolve()
            if "interactive-output" in cell.options else result_path.with_suffix(".html")),
    }

_VLLM_CUSTOM_DATASET_SHUFFLE_SEED = 0
VLLM_BENCH_SERVE_SHA256 = (
    "5dcfbc9cb735450d9399cc65d7a7fecad8e9b841c5f7ea0fad90f0eb0b768d97"
)

def create_workload_artifact(
    *,
    name: str,
    num_requests: int,
    prompt_len: int,
    output_len: int,
    sampling: dict[str, Any],
    tokenizer: Any,
    seed: int,
    prompts: list[str] | None = None,
    request_ids: list[str] | None = None,
) -> WorkloadArtifact:
    if not isinstance(name, str) or not name:
        raise ManifestValidationError("workload.name must be a non-empty string")
    requested_num_requests = _require_positive_int_value(
        num_requests, "workload.num_requests")
    prompt_len = _require_positive_int_value(
        prompt_len, "workload.prompt_len")
    output_len = _require_positive_int_value(
        output_len, "workload.output_len")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ManifestValidationError("workload.seed must be a non-negative integer")
    if tokenizer is None:
        raise ManifestValidationError(
            "workload materialization requires the model tokenizer")

    sampling = validate_sampling_config(dict(sampling), "workload artifact")
    if sampling["max_tokens"] != output_len:
        raise ManifestValidationError(
            "workload.output_len must match sampling.max_tokens")

    if prompts is None:
        prompts = _generate_prompts(requested_num_requests, prompt_len, seed, tokenizer)
    else:
        prompts = _validate_prompt_list(prompts)
    if not prompts:
        raise ManifestValidationError("prompts must be a non-empty list")

    actual_num_requests = len(prompts)
    if actual_num_requests != requested_num_requests:
        raise ManifestValidationError(
            "prompt count must match workload.num_requests: "
            f"expected {requested_num_requests}, got {actual_num_requests}")
    if request_ids is None:
        request_ids = _default_request_ids(actual_num_requests)
    else:
        request_ids = _validate_request_ids(request_ids, actual_num_requests)
    prompt_lengths = [_token_length(prompt, tokenizer) for prompt in prompts]
    for index, actual_prompt_len in enumerate(prompt_lengths):
        if actual_prompt_len != prompt_len:
            raise ManifestValidationError(
                f"prompt {index} must contain exactly {prompt_len} model tokens; "
                f"got {actual_prompt_len}")

    workload = {
        "name": name,
        "prompt_len": prompt_len,
        "output_len": output_len,
    }
    token_controls = _token_controls(sampling)
    artifact = WorkloadArtifact(
        version=1,
        workload=workload,
        request_ids=request_ids,
        prompts=prompts,
        prompt_lengths=prompt_lengths,
        sampling=sampling,
        token_controls=token_controls,
        num_requests=actual_num_requests,
        requested_num_requests=requested_num_requests,
        seed=seed,
        hash="",
    )
    return _artifact_with_hash(artifact)


def load_model_tokenizer(model: str | Path) -> Any:
    """Load the exact local tokenizer used to define Serving token lengths."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model),
        local_files_only=True,
    )
    if not callable(tokenizer) and not hasattr(tokenizer, "encode"):
        raise ManifestValidationError(
            f"model tokenizer is not callable and has no encode(): {model}")
    return tokenizer


def save_workload_artifact(artifact: WorkloadArtifact, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True))


def load_workload_artifact(path: str | Path) -> WorkloadArtifact:
    artifact_path = Path(path)
    with open(artifact_path) as f:
        raw = json.load(f)
    artifact = _load_workload_artifact_dict(raw, path=artifact_path)
    return artifact


def materialize_workload_commands(
    *,
    workload_path: str | Path,
    model: str | Path,
    tokenizer: Any,
    deployment: str,
    sampling: dict[str, Any],
    request_rate: float,
    attention_kv_layout: str,
    attention_kv_distribution: str,
    expert_routing: dict[str, Any] | None,
    expected_workload_path: str | Path | None = None,
    colocated_hw_config: str | Path | None = None,
    colocated_engine_args: dict[str, Any] | None = None,
    colocated_prediction_output: str | Path | None = None,
    colocated_tp_size: int | None = None,
    disaggregated_prefill_hw_config: str | Path | None = None,
    disaggregated_decode_hw_config: str | Path | None = None,
    prefill_engine_args: dict[str, Any] | None = None,
    decode_engine_args: dict[str, Any] | None = None,
    prefill_tp_size: int | None = None,
    decode_tp_size: int | None = None,
    kv_connector: str | None = None,
    disaggregated_prediction_output: str | Path | None = None,
    vllm_dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize real benchmark and production Simulation commands.

    Historical fake-trace capture and trace-pricing commands are deliberately
    absent. Both colocated and PD predictions enter through the same
    ``src.serving`` composition root used by unit and full-eval gates.
    """
    path = Path(workload_path)
    if expected_workload_path is not None:
        expected = Path(expected_workload_path)
        if path.resolve(strict=False) != expected.resolve(strict=False):
            raise ManifestValidationError("divergent workload artifact path")
    if not path.exists():
        raise ManifestValidationError(f"workload artifact does not exist: {path}")
    artifact = load_workload_artifact(path)
    _validate_workload_tokenization(artifact, tokenizer)
    expected_sampling = validate_sampling_config(
        dict(sampling), "materialized workload")
    if expected_sampling != artifact.sampling:
        raise ManifestValidationError("divergent sampling config")
    if (not isinstance(request_rate, (int, float))
            or isinstance(request_rate, bool)
            or not math.isfinite(request_rate)
            or request_rate <= 0):
        raise ManifestValidationError(
            "request_rate must be a finite positive number")
    if deployment not in ("colocated", "disaggregated"):
        raise ManifestValidationError(f"unsupported deployment: {deployment!r}")
    if deployment != "colocated":
        raise ManifestValidationError(
            "the current vLLM simulation baseline supports only colocated "
            "deployment")
    if attention_kv_layout not in (
            "contiguous", "paged", "paged_varlen", "chunked_prefill", "mla"):
        raise ManifestValidationError(
            f"unsupported attention_kv_layout: {attention_kv_layout!r}")
    if attention_kv_layout != "paged_varlen":
        raise ManifestValidationError(
            "the pinned vLLM FlashAttention path requires "
            "attention_kv_layout='paged_varlen'")
    if attention_kv_distribution not in ("uniform", "skew"):
        raise ManifestValidationError(
            "unsupported attention_kv_distribution: "
            f"{attention_kv_distribution!r}")

    # Validate the complete deployment contract before writing the derived
    # vLLM dataset.  A missing hardware config must fail without leaving a
    # partially materialized experiment behind.
    if deployment == "colocated":
        hw_config = _require_existing_file(
            colocated_hw_config, "colocated_hw_config")
        tp_size = _require_positive_int_value(
            colocated_tp_size, "colocated_tp_size")
        engine_args = _require_vllm_baseline_engine_args(
            colocated_engine_args,
            "colocated_engine_args",
            tensor_parallel_size=tp_size,
            num_requests=artifact.num_requests,
        )
        prediction_output = _require_path_value(
            colocated_prediction_output, "colocated_prediction_output")
        if engine_args["enable_expert_parallel"]:
            _require_validated_ep2_model_hardware(model, hw_config)
    elif deployment == "disaggregated":
        prefill_hw_config = _require_existing_file(
            disaggregated_prefill_hw_config,
            "disaggregated_prefill_hw_config")
        decode_hw_config = _require_existing_file(
            disaggregated_decode_hw_config,
            "disaggregated_decode_hw_config")
        prefill_size = _require_positive_int_value(
            prefill_tp_size, "prefill_tp_size")
        decode_size = _require_positive_int_value(
            decode_tp_size, "decode_tp_size")
        prefill_args = _require_engine_args(
            prefill_engine_args, "prefill_engine_args")
        decode_args = _require_engine_args(
            decode_engine_args, "decode_engine_args")
        if not isinstance(kv_connector, str) or not kv_connector:
            raise ManifestValidationError(
                "kv_connector must be a non-empty string")
        prediction_output = _require_path_value(
            disaggregated_prediction_output,
            "disaggregated_prediction_output")
    else:
        raise AssertionError(
            "deployment validation allowed an unknown value")

    normalized_expert_routing = _require_materialized_expert_routing(
        expert_routing,
        routing_required=engine_args["enable_expert_parallel"],
        model=model,
    )
    dataset_path = Path(vllm_dataset_path) if vllm_dataset_path is not None else _default_vllm_dataset_path(path)
    dataset_seed = artifact.seed
    sidecar_request_ids = _write_vllm_custom_dataset(artifact, dataset_path)
    arrival_process = vllm_bench_arrival_process(
        request_rate=request_rate,
        seed=dataset_seed,
    )
    mtp_scenario = mtp_scenario_manifest(VLLMSchedulerConfig(str(model)))
    routing_manifest = _materialized_expert_routing_identity(
        normalized_expert_routing)
    execution_mode = "accelerated"
    if (normalized_expert_routing is not None
            and normalized_expert_routing["source_kind"]
            == "replay_observation"):
        execution_mode = "reference"
    validation_identity = {
        "workload_hash": artifact.hash,
        "request_order_sha256": hashlib.sha256(
            _canonical_json(artifact.request_ids).encode()).hexdigest(),
        "arrival_contract_sha256": hashlib.sha256(
            _canonical_json(arrival_process).encode()).hexdigest(),
        "scheduler_contract_sha256": hashlib.sha256(
            _canonical_json({
                "engine_args": engine_args,
                "mtp_scenario": mtp_scenario,
                "expert_routing": routing_manifest,
            }).encode()).hexdigest(),
    }
    model_str = str(model)
    max_tokens = str(artifact.token_controls["max_tokens"])
    simulation_script = str(PROJECT_ROOT / "scripts" / "run_serving_simulation.py")
    common_simulation = [
        sys.executable, simulation_script,
        "--workload", str(path), "--model", model_str,
        "--backend", "vllm", "--request-rate", str(request_rate),
        "--arrival-process", "vllm_bench_gamma",
        "--arrival-seed", str(dataset_seed),
        "--arrival-burstiness", "1.0",
        "--prediction-method", "midlayer",
        "--launch-policy", "injected_launch",
        "--max-requests", str(engine_args["max_num_seqs"]),
        "--max-batched-tokens", str(
            engine_args["max_num_batched_tokens"]),
        "--scheduler-pipeline-depth", "1",
        "--kv-block-size-tokens", str(engine_args["block_size"]),
        "--debug-num-kv-blocks", str(engine_args["num_gpu_blocks_override"]),
        "--attention-kv-layout", attention_kv_layout,
        "--attention-kv-distribution", attention_kv_distribution,
        "--execution-mode", execution_mode,
    ]
    if normalized_expert_routing is not None:
        if normalized_expert_routing["source_kind"] == "explicit_assumption":
            common_simulation.extend([
                "--expert-routing-policy", "seeded_assumption",
                "--expert-routing-source-id",
                normalized_expert_routing["source_id"],
                "--expert-routing-seed",
                str(normalized_expert_routing["root_seed"]),
            ])
        elif normalized_expert_routing["source_kind"] == "replay_observation":
            common_simulation.extend([
                "--expert-routing-policy", "replay_observation",
                "--expert-routing-replay",
                normalized_expert_routing["path"],
                "--expert-routing-source-sha256",
                normalized_expert_routing["artifact"]["source_sha256"],
                "--expert-routing-canonical-payload-sha256",
                normalized_expert_routing["artifact"][
                    "canonical_payload_sha256"],
            ])
        elif normalized_expert_routing["source_kind"] == "profile_observation":
            common_simulation.extend([
                "--expert-routing-policy", "profile_observation",
                "--expert-routing-source-id",
                normalized_expert_routing["source_id"],
                "--expert-routing-profile-manifest",
                normalized_expert_routing["manifest_path"],
            ])
        else:
            raise ManifestValidationError(
                "unknown materialized expert routing source")
    benchmark_argv = [
            sys.executable, "-m", "vllm.entrypoints.cli.main",
            "bench", "serve", "--model", model_str,
            "--dataset-name", "custom", "--dataset-path", str(dataset_path),
            "--num-prompts", str(artifact.num_requests),
            "--custom-output-len", max_tokens, "--custom-skip-chat-template",
            "--ignore-eos", "--seed", str(dataset_seed),
            "--request-rate", str(request_rate),
            "--burstiness", "1.0",
        ] + _vllm_sampling_args(artifact.sampling) + [
            "--save-detailed",
            "--metadata",
            *(
                f"{key}={value}"
                for key, value in validation_identity.items()
            ),
        ]
    benchmark = {
        "kind": "real_vllm_benchmark",
        "argv": benchmark_argv,
        "workload_path": str(path), "vllm_dataset_path": str(dataset_path),
        "vllm_dataset_seed": dataset_seed,
        "vllm_dataset_shuffle_seed": _VLLM_CUSTOM_DATASET_SHUFFLE_SEED,
        "vllm_sidecar_request_ids": sidecar_request_ids,
        "vllm_consumed_request_ids": list(artifact.request_ids),
        "arrival_process": dict(arrival_process),
        "mtp_scenario": dict(mtp_scenario),
        "validation_identity": dict(validation_identity),
        "workload_hash": artifact.hash, "sampling": dict(artifact.sampling),
    }
    if deployment == "colocated":
        simulation = {
            "kind": "colocated_midlayer_simulation",
            "argv": common_simulation + (
                ["--enable-expert-parallel"]
                if engine_args["enable_expert_parallel"] else []
            ) + [
                "--deployment", "colocated",
                "--hardware-config", hw_config,
                "--prefill-tp-size", str(tp_size),
                "--output", prediction_output,
            ],
            "workload_path": str(path),
            "workload_hash": artifact.hash,
            "sampling": dict(artifact.sampling),
            "arrival_process": dict(arrival_process),
            "mtp_scenario": dict(mtp_scenario),
            "expert_routing": (
                None if routing_manifest is None else dict(routing_manifest)),
            "validation_identity": dict(validation_identity),
            "tensor_parallel_size": tp_size,
            "engine_args": engine_args,
            "outputs": {"prediction": prediction_output},
        }
        commands = {
            "real_vllm_benchmark": benchmark,
            "colocated_midlayer_simulation": simulation,
        }
    elif deployment == "disaggregated":
        simulation = {
            "kind": "disaggregated_midlayer_simulation",
            "argv": common_simulation + [
                "--deployment", "pd_disagg",
                "--hardware-config", prefill_hw_config,
                "--decode-hardware-config", decode_hw_config,
                "--prefill-tp-size", str(prefill_size),
                "--decode-tp-size", str(decode_size),
                "--num-prefill-instances", "1",
                "--num-decode-instances", "1",
                "--transfer-algorithm", "send_recv",
                "--transfer-link", "nvlink",
                "--output", prediction_output,
            ],
            "workload_path": str(path),
            "workload_hash": artifact.hash,
            "sampling": dict(artifact.sampling),
            "arrival_process": dict(arrival_process),
            "mtp_scenario": dict(mtp_scenario),
            "prefill_engine_args": prefill_args,
            "decode_engine_args": decode_args,
            "prefill_tp_size": prefill_size,
            "decode_tp_size": decode_size,
            "kv_connector": kv_connector,
            "outputs": {"prediction": prediction_output},
        }
        commands = {
            "real_vllm_benchmark": benchmark,
            "disaggregated_midlayer_simulation": simulation,
        }
    else:
        raise AssertionError(
            "deployment validation allowed an unknown value")
    return {
        "version": 2, "deployment": deployment, "workload_path": str(path),
        "vllm_dataset_path": str(dataset_path), "workload_hash": artifact.hash,
        "sampling": dict(artifact.sampling),
        "token_controls": dict(artifact.token_controls),
        "mtp_scenario": dict(mtp_scenario),
        **commands, "commands": commands,
    }

def _default_vllm_dataset_path(workload_path: Path) -> Path:
    return workload_path.with_suffix(".vllm-bench.jsonl")


def _write_vllm_custom_dataset(
    artifact: WorkloadArtifact,
    dataset_path: Path,
) -> list[str]:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    shuffle_indices = _vllm_shuffle_indices(
        len(artifact.prompts),
        _VLLM_CUSTOM_DATASET_SHUFFLE_SEED,
    )
    sidecar_prompts: list[str | None] = [None] * len(artifact.prompts)
    sidecar_request_ids: list[str | None] = [None] * len(artifact.prompts)
    for consumed_index, sidecar_index in enumerate(shuffle_indices):
        sidecar_prompts[sidecar_index] = artifact.prompts[consumed_index]
        sidecar_request_ids[sidecar_index] = artifact.request_ids[consumed_index]
    if any(prompt is None for prompt in sidecar_prompts):
        raise ManifestValidationError("vLLM dataset sidecar prompt order is incomplete")
    if any(request_id is None for request_id in sidecar_request_ids):
        raise ManifestValidationError("vLLM dataset sidecar request ID order is incomplete")
    lines = [
        json.dumps({"prompt": prompt}, sort_keys=True)
        for prompt in sidecar_prompts
    ]
    dataset_path.write_text("\n".join(lines) + "\n")
    return list(sidecar_request_ids)


def _vllm_shuffle_indices(num_prompts: int, seed: int) -> list[int]:
    indices = list(range(num_prompts))
    random.Random(seed).shuffle(indices)
    return indices


def vllm_bench_arrival_process(
    *,
    request_rate: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "kind": "vllm_bench_gamma",
        "request_rate_per_s": float(request_rate),
        "burstiness": 1.0,
        "seed": seed,
        "normalization": "total_requests_over_request_rate",
        "vllm_version": VLLM_SCHEDULER_VERSION,
        "vllm_source_commit": VLLM_SCHEDULER_SOURCE_COMMIT,
        "source_symbol": "vllm.benchmarks.serve.get_request",
        "source_sha256": VLLM_BENCH_SERVE_SHA256,
    }


def vllm_bench_arrival_times(
    *,
    num_requests: int,
    request_rate: float,
    burstiness: float,
    seed: int,
) -> tuple[float, ...]:
    """Reproduce pinned vLLM ``bench serve`` request send timestamps."""

    if (not isinstance(num_requests, int) or isinstance(num_requests, bool)
            or num_requests < 1):
        raise ManifestValidationError("num_requests must be a positive integer")
    if (not isinstance(request_rate, (int, float))
            or isinstance(request_rate, bool)
            or not math.isfinite(request_rate)
            or request_rate <= 0):
        raise ManifestValidationError(
            "request_rate must be a finite positive number")
    if (not isinstance(burstiness, (int, float))
            or isinstance(burstiness, bool)
            or not math.isfinite(burstiness)
            or burstiness <= 0):
        raise ManifestValidationError(
            "arrival burstiness must be a finite positive number")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ManifestValidationError(
            "arrival seed must be a non-negative integer")

    random_state = np.random.RandomState(seed)
    theta = 1.0 / (float(request_rate) * float(burstiness))
    intervals = random_state.gamma(
        shape=float(burstiness),
        scale=theta,
        size=num_requests,
    )
    arrivals = np.cumsum(intervals)
    if arrivals[-1] == 0:
        raise ManifestValidationError(
            "vLLM benchmark arrival schedule has zero total delay")
    target_total_delay_s = num_requests / float(request_rate)
    arrivals *= target_total_delay_s / arrivals[-1]
    return tuple(float(value) for value in arrivals)


def _vllm_sampling_args(sampling: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, flag in (
        ("temperature", "--temperature"),
        ("top_p", "--top-p"),
        ("top_k", "--top-k"),
        ("min_p", "--min-p"),
    ):
        if key in sampling:
            args.extend([flag, str(sampling[key])])
    return args


def _require_engine_args(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{label} must be an explicit object")
    return dict(value)


def _require_vllm_baseline_engine_args(
    value: Any,
    label: str,
    *,
    tensor_parallel_size: int,
    num_requests: int,
) -> dict[str, Any]:
    engine_args = _require_engine_args(value, label)
    allowed = {
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "enable_expert_parallel",
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "max_num_seqs",
        "max_num_batched_tokens",
        "block_size",
        "num_gpu_blocks_override",
    }
    unknown = sorted(set(engine_args) - allowed)
    if unknown:
        raise ManifestValidationError(
            f"{label} contains engine arguments outside the current vLLM "
            f"baseline: {', '.join(unknown)}")
    if "tensor_parallel_size" in engine_args:
        configured_tp = _require_positive_int_value(
            engine_args["tensor_parallel_size"],
            f"{label}.tensor_parallel_size",
        )
        if configured_tp != tensor_parallel_size:
            raise ManifestValidationError(
                f"{label}.tensor_parallel_size conflicts with the simulation "
                "tensor parallel size")
    else:
        engine_args["tensor_parallel_size"] = tensor_parallel_size
    if "pipeline_parallel_size" in engine_args:
        pipeline_parallel_size = _require_positive_int_value(
            engine_args["pipeline_parallel_size"],
            f"{label}.pipeline_parallel_size",
        )
    else:
        pipeline_parallel_size = 1
        engine_args["pipeline_parallel_size"] = pipeline_parallel_size
    if pipeline_parallel_size != 1:
        raise ManifestValidationError(
            "the current vLLM simulation baseline requires PP=1")
    if "data_parallel_size" in engine_args:
        data_parallel_size = _require_positive_int_value(
            engine_args["data_parallel_size"],
            f"{label}.data_parallel_size",
        )
    else:
        data_parallel_size = 1
        engine_args["data_parallel_size"] = data_parallel_size
    if data_parallel_size != 1:
        raise ManifestValidationError(
            "the current vLLM simulation baseline requires DP=1")
    if "enable_expert_parallel" in engine_args:
        enable_expert_parallel = engine_args["enable_expert_parallel"]
        if not isinstance(enable_expert_parallel, bool):
            raise ManifestValidationError(
                f"{label}.enable_expert_parallel must be boolean")
    else:
        enable_expert_parallel = False
        engine_args["enable_expert_parallel"] = enable_expert_parallel
    if enable_expert_parallel:
        if tensor_parallel_size != 2:
            raise ManifestValidationError(
                "the validated expert parallel simulation requires TP=2")
    elif tensor_parallel_size != 1:
        raise ManifestValidationError(
            "the current non-EP vLLM simulation baseline requires TP=1")
    scheduler_fields = tuple(VLLM_CONFORMANCE_SCHEDULER_DEFAULTS)
    if num_requests > 1:
        missing = tuple(
            field for field in scheduler_fields
            if field not in engine_args)
        if missing:
            raise ManifestValidationError(
                f"{label} must explicitly define scheduler capacity for a "
                "multi-request workload: " + ", ".join(missing))
    for field, default in VLLM_CONFORMANCE_SCHEDULER_DEFAULTS.items():
        if field in engine_args:
            engine_args[field] = _require_positive_int_value(
                engine_args[field], f"{label}.{field}")
        else:
            engine_args[field] = default
    if engine_args["block_size"] != 16:
        raise ManifestValidationError(
            "the pinned CPU-only vLLM Scheduler baseline requires "
            "block_size=16")
    if "enable_chunked_prefill" in engine_args:
        enable_chunked_prefill = engine_args["enable_chunked_prefill"]
        if not isinstance(enable_chunked_prefill, bool):
            raise ManifestValidationError(
                f"{label}.enable_chunked_prefill must be boolean")
        if not enable_chunked_prefill:
            raise ManifestValidationError(
                "pinned vLLM 0.11 EngineArgs forces chunked prefill on its "
                "V1 generative path; enable_chunked_prefill must be true")
    else:
        engine_args["enable_chunked_prefill"] = True
    if "enable_prefix_caching" in engine_args:
        enable_prefix_caching = engine_args["enable_prefix_caching"]
        if not isinstance(enable_prefix_caching, bool):
            raise ManifestValidationError(
                f"{label}.enable_prefix_caching must be boolean")
        if enable_prefix_caching:
            raise ManifestValidationError(
                "trace-authoritative prefix injection requires vLLM content "
                "prefix caching to be disabled")
    else:
        engine_args["enable_prefix_caching"] = False
    return engine_args


def _require_validated_ep2_model_hardware(
    model: str | Path,
    hardware_config: str | Path,
) -> None:
    try:
        model_config = load_model_config(str(model))
        hardware_id = HWSpec(str(hardware_config)).hardware_id
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise ManifestValidationError(
            "expert-parallel materialization could not validate its "
            f"model/hardware contract: {error}") from error
    if (hardware_id != "nvidia_a800"
            or model_config.model_type != "qwen3_moe"
            or model_config.attention_layout != "standard_gqa"
            or model_config.num_experts != 2
            or model_config.num_experts_per_tok != 1
            or model_config.num_shared_experts != 0
            or model_config.compute_dtype != "bf16"
            or model_config.weight_dtype != "bf16"):
        raise ManifestValidationError(
            "expert-parallel materialization requires the validated "
            "nvidia_a800 BF16 Qwen3MoE two-expert top-k=1 slice")


def _require_vllm_baseline_moe_config(
    value: Any,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(
            f"{label} must define the expert communication backend")
    required = {"all2all_backend", "eplb_enabled"}
    if set(value) != required:
        raise ManifestValidationError(
            f"{label} must contain exactly all2all_backend and eplb_enabled")
    backend = value["all2all_backend"]
    if backend != "allgather_reducescatter":
        raise ManifestValidationError(
            "the validated expert parallel simulation requires "
            "all2all_backend='allgather_reducescatter'")
    eplb_enabled = value["eplb_enabled"]
    if eplb_enabled is not False:
        raise ManifestValidationError(
            "the validated expert parallel simulation requires EPLB to be "
            "disabled")


def _require_path_value(value: str | Path | None, label: str) -> str:
    if value is None:
        raise ManifestValidationError(f"{label} is required")
    normalized = str(value)
    if not normalized:
        raise ManifestValidationError(f"{label} must not be empty")
    return normalized


def _require_existing_file(value: str | Path | None, label: str) -> str:
    normalized = _require_path_value(value, label)
    if not Path(normalized).is_file():
        raise ManifestValidationError(
            f"{label} does not exist or is not a file: {normalized}")
    return normalized


def _generate_prompts(
    num_requests: int,
    prompt_len: int,
    seed: int,
    tokenizer: Any,
) -> list[str]:
    vocab = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "artificial", "intelligence", "machine", "learning", "neural",
        "network", "transformer", "attention", "mechanism", "language",
        "model", "generation", "inference", "serving", "batching",
        "throughput", "latency", "optimization", "accelerator", "gpu",
        "tensor", "parallel", "data", "pipeline", "distributed", "system",
    ]
    rng = random.Random(seed)
    prompts = [
        " ".join(rng.choices(vocab, k=prompt_len))
        for _ in range(num_requests)
    ]
    return [
        _adjust_prompt_length(prompt, prompt_len, tokenizer, rng, vocab)
        for prompt in prompts
    ]


def _adjust_prompt_length(
    prompt: str,
    prompt_len: int,
    tokenizer: Any,
    rng: random.Random,
    vocab: list[str],
) -> str:
    words = prompt.split()
    for _ in range(10):
        current_len = _token_length(" ".join(words), tokenizer)
        if current_len == prompt_len:
            return " ".join(words)
        if current_len < prompt_len:
            words.extend(rng.choices(vocab, k=prompt_len - current_len))
        else:
            words = words[:max(1, len(words) - (current_len - prompt_len))]
    adjusted = " ".join(words)
    actual_length = _token_length(adjusted, tokenizer)
    if actual_length != prompt_len:
        raise ManifestValidationError(
            f"could not construct prompt with {prompt_len} tokens; got {actual_length}")
    return adjusted


def _validate_prompt_list(prompts: list[str]) -> list[str]:
    if not isinstance(prompts, list) or not prompts:
        raise ManifestValidationError("prompts must be a non-empty list")
    if not all(isinstance(prompt, str) and prompt for prompt in prompts):
        raise ManifestValidationError("prompts must contain non-empty strings")
    return list(prompts)


def _default_request_ids(num_requests: int) -> list[str]:
    return [f"req-{idx:06d}" for idx in range(num_requests)]


def _validate_request_ids(request_ids: list[str], expected_count: int) -> list[str]:
    if not isinstance(request_ids, list) or len(request_ids) != expected_count:
        raise ManifestValidationError("request_ids must match prompt count")
    if not all(isinstance(request_id, str) and request_id for request_id in request_ids):
        raise ManifestValidationError("request_ids must contain non-empty strings")
    if len(set(request_ids)) != len(request_ids):
        raise ManifestValidationError("duplicate request_ids")
    return list(request_ids)


def _token_length(prompt: str, tokenizer: Any) -> int:
    if tokenizer is None:
        raise ManifestValidationError(
            "token length cannot be computed without the model tokenizer")
    if callable(tokenizer):
        tokenized = tokenizer(prompt, add_special_tokens=False)
    elif hasattr(tokenizer, "encode"):
        tokenized = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        raise ManifestValidationError("tokenizer must be callable or expose encode()")
    if isinstance(tokenized, Mapping):
        if "input_ids" not in tokenized:
            raise ManifestValidationError("tokenizer result is missing input_ids")
        token_ids = tokenized["input_ids"]
    else:
        token_ids = tokenized
    if not isinstance(token_ids, list):
        raise ManifestValidationError("tokenizer must return input_ids as a list")
    if not all(isinstance(token_id, int) and not isinstance(token_id, bool)
               for token_id in token_ids):
        raise ManifestValidationError(
            "tokenizer input_ids must contain only integer token IDs")
    return len(token_ids)


def _validate_workload_tokenization(
    artifact: WorkloadArtifact,
    tokenizer: Any,
) -> None:
    actual_lengths = [
        _token_length(prompt, tokenizer)
        for prompt in artifact.prompts
    ]
    if actual_lengths != artifact.prompt_lengths:
        raise ManifestValidationError(
            "workload prompt_lengths do not match the model tokenizer: "
            f"stored={artifact.prompt_lengths}, actual={actual_lengths}")
    prompt_len = artifact.workload["prompt_len"]
    mismatches = [
        (index, length)
        for index, length in enumerate(actual_lengths)
        if length != prompt_len
    ]
    if mismatches:
        raise ManifestValidationError(
            f"workload prompts must each contain exactly {prompt_len} model "
            f"tokens; mismatches={mismatches[:8]}")


def _token_controls(sampling: dict[str, Any]) -> dict[str, Any]:
    return {
        "min_tokens": sampling["min_tokens"],
        "max_tokens": sampling["max_tokens"],
        "ignore_eos": sampling["ignore_eos"],
    }


def _artifact_with_hash(artifact: WorkloadArtifact) -> WorkloadArtifact:
    payload = _artifact_hash_payload(artifact.to_dict())
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    return WorkloadArtifact(
        version=artifact.version,
        workload=dict(artifact.workload),
        request_ids=list(artifact.request_ids),
        prompts=list(artifact.prompts),
        prompt_lengths=list(artifact.prompt_lengths),
        sampling=dict(artifact.sampling),
        token_controls=dict(artifact.token_controls),
        num_requests=artifact.num_requests,
        requested_num_requests=artifact.requested_num_requests,
        seed=artifact.seed,
        hash=digest,
        path=artifact.path,
    )


def _load_workload_artifact_dict(
    raw: Any,
    path: Path | None = None,
) -> WorkloadArtifact:
    if not isinstance(raw, dict):
        raise ManifestValidationError("workload artifact root must be an object")
    expected_fields = {
        "version", "workload", "request_ids", "prompts", "prompt_lengths",
        "sampling", "token_controls", "num_requests",
        "requested_num_requests", "seed", "hash",
    }
    if set(raw) != expected_fields:
        raise ManifestValidationError(
            "workload artifact schema mismatch: "
            f"missing={sorted(expected_fields - set(raw))}, "
            f"unknown={sorted(set(raw) - expected_fields)}")
    if raw["version"] != 1:
        raise ManifestValidationError("workload artifact version must be 1")

    workload = raw["workload"]
    if not isinstance(workload, dict):
        raise ManifestValidationError("workload must be an object")
    if set(workload) != {"name", "prompt_len", "output_len"}:
        raise ManifestValidationError("workload artifact workload schema mismatch")
    if not isinstance(workload["name"], str) or not workload["name"]:
        raise ManifestValidationError("workload.name must be a non-empty string")
    prompt_len = _require_positive_int(workload, "prompt_len", "workload")
    output_len = _require_positive_int(workload, "output_len", "workload")

    prompts = _validate_prompt_list(raw["prompts"])
    request_ids = _validate_request_ids(raw["request_ids"], len(prompts))

    prompt_lengths = raw["prompt_lengths"]
    if not isinstance(prompt_lengths, list) or len(prompt_lengths) != len(prompts):
        raise ManifestValidationError("prompt_lengths must match prompt count")
    if not all(isinstance(length, int) and not isinstance(length, bool) and length > 0
               for length in prompt_lengths):
        raise ManifestValidationError("prompt_lengths must contain positive integers")

    num_requests = raw["num_requests"]
    if not isinstance(num_requests, int) or isinstance(num_requests, bool):
        raise ManifestValidationError("num_requests must be an integer")
    if num_requests != len(prompts):
        raise ManifestValidationError("num_requests must match actual prompt count")

    requested_num_requests = raw["requested_num_requests"]
    if (not isinstance(requested_num_requests, int)
            or isinstance(requested_num_requests, bool)
            or requested_num_requests < 1):
        raise ManifestValidationError(
            "requested_num_requests must be a positive integer")
    if requested_num_requests != num_requests:
        raise ManifestValidationError(
            "requested_num_requests must match num_requests")

    seed = raw["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ManifestValidationError("seed must be a non-negative integer")

    sampling = validate_sampling_config(_require_dict(raw, "sampling", "workload artifact"), "workload artifact")
    if sampling["max_tokens"] != output_len:
        raise ManifestValidationError("workload.output_len must match sampling.max_tokens")
    token_controls = raw["token_controls"]
    if not isinstance(token_controls, dict):
        raise ManifestValidationError("token_controls must be an object")
    if token_controls != _token_controls(sampling):
        raise ManifestValidationError("token_controls must match sampling")

    stored_hash = raw["hash"]
    if not isinstance(stored_hash, str) or not stored_hash:
        raise ManifestValidationError("hash must be a non-empty string")

    artifact = WorkloadArtifact(
        version=1,
        workload={
            "name": workload["name"],
            "prompt_len": prompt_len,
            "output_len": output_len,
        },
        request_ids=request_ids,
        prompts=prompts,
        prompt_lengths=list(prompt_lengths),
        sampling=sampling,
        token_controls=_token_controls(sampling),
        num_requests=num_requests,
        requested_num_requests=requested_num_requests,
        seed=seed,
        hash=stored_hash,
        path=path,
    )
    expected_hash = _artifact_with_hash(artifact).hash
    if stored_hash != expected_hash:
        raise ManifestValidationError("hash does not match workload artifact content")
    return artifact


def _artifact_hash_payload(raw: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw)
    del payload["hash"]
    return payload


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def materialize_experiment_plan(
    cell: ExperimentCell | ProductionTraceSweepCell,
    output_root: Path,
    *,
    project_root: Path,
    invocation: list[str],
    tokenizer: Any,
) -> dict[str, Any]:
    """Materialize one case and attach the environment needed to reproduce it."""
    if isinstance(cell, ProductionTraceSweepCell):
        command_manifest = _materialize_production_trace_sweep_cell(
            cell, output_root)
    elif cell.deployment == "colocated":
        command_manifest = _materialize_colocated_cell(
            cell, output_root, tokenizer=tokenizer)
    elif cell.deployment == "disaggregated":
        raise ManifestValidationError(
            "the current vLLM scheduler/reference baseline does not yet "
            "materialize disaggregated experiments")
    else:
        raise ManifestValidationError(
            f"unsupported experiment deployment: {cell.deployment!r}")
    _attach_provenance(command_manifest, project_root, invocation)
    return command_manifest


def _materialize_production_trace_sweep_cell(
    cell: ProductionTraceSweepCell,
    output_root: Path,
) -> dict[str, Any]:
    outputs = _production_trace_sweep_outputs(output_root)
    trace = cell.trace
    field_mapping = trace["field_mapping"]
    success_filter = trace["success_filter"]
    simulation = cell.simulation
    scheduler = simulation["scheduler"]
    if scheduler["capacity_mode"] != "analytical":
        raise ManifestValidationError(
            "Production Trace plans require an analytical runtime memory "
            "policy; debug KV block counts are forbidden")
    mtp = simulation["mtp"]
    hardware = simulation["hardware"]
    normalized_expert_routing = _require_materialized_expert_routing(
        simulation["expert_routing"],
        routing_required=(
            scheduler["enable_expert_parallel"]
            or mtp["acceptance_policy"] != "disabled"),
        model=cell.model,
    )
    routing_manifest = _materialized_expert_routing_identity(
        normalized_expert_routing)
    simulation_argv = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_serving_simulation.py"),
        "--production-trace", str(cell.trace_path),
        "--trace-source-sha256", trace["source_sha256"],
        "--model", str(cell.model),
        "--model-config-sha256", cell.model_config_sha256,
        "--hardware-config", str(hardware["config_path"]),
        "--hardware-config-sha256", hardware["config_sha256"],
        "--deployment", simulation["deployment"],
        "--backend", simulation["backend"],
        "--prediction-method", simulation["prediction_method"],
        "--launch-policy", simulation["launch_policy"],
        "--execution-mode", "accelerated",
        "--attention-kv-layout", simulation["attention_kv_layout"],
        "--attention-kv-distribution",
        simulation["attention_kv_distribution"],
        "--trace-format", trace["source_format"],
        "--trace-time-unit", trace["input_time_unit"],
        "--trace-arrival-field", field_mapping["arrival_time"],
        "--trace-input-tokens-field", field_mapping["input_tokens"],
        "--trace-output-tokens-field", field_mapping["output_tokens"],
        "--trace-prefix-hit-tokens-field",
        field_mapping["prefix_hit_tokens"],
        "--trace-success-filter-mode", success_filter["mode"],
        "--trace-success-provenance", success_filter["provenance"],
        "--trace-replay-mode", "arrival_rate_sweep",
        "--trace-arrival-rate-scales",
        *(str(scale) for scale in cell.arrival_rate_scales),
        "--max-requests", str(scheduler["max_requests"]),
        "--max-batched-tokens", str(scheduler["max_batched_tokens"]),
        "--scheduler-pipeline-depth",
        str(scheduler["scheduler_pipeline_depth"]),
        "--kv-block-size-tokens", str(scheduler["kv_block_size_tokens"]),
        "--max-model-len", str(scheduler["max_model_len"]),
        "--prefill-tp-size", str(scheduler["prefill_tp_size"]),
        "--prefill-pp-size", str(scheduler["prefill_pp_size"]),
    ]
    simulation_argv.extend((
        "--runtime-memory-policy",
        str(scheduler["runtime_memory_policy"]),
    ))
    if mtp["acceptance_policy"] == "disabled":
        pass
    elif mtp["acceptance_policy"] == "fixed_count":
        simulation_argv.extend((
            "--mtp-num-speculative-tokens",
            str(mtp["num_speculative_tokens"]),
            "--mtp-acceptance-policy", "fixed_count",
            "--mtp-fixed-accepted-tokens",
            str(mtp["fixed_accepted_tokens"]),
            "--mtp-acceptance-source-id", mtp["acceptance_source_id"],
        ))
    elif mtp["acceptance_policy"] == "maximum_entropy":
        assumption = mtp["acceptance_assumption"]
        simulation_argv.extend((
            "--mtp-num-speculative-tokens",
            str(mtp["num_speculative_tokens"]),
            "--mtp-acceptance-policy", "maximum_entropy",
            "--mtp-acceptance-source-id", mtp["acceptance_source_id"],
            "--mtp-mean-acceptance-rate",
            assumption["mean_acceptance_rate"],
            "--mtp-acceptance-seed", str(assumption["root_seed"]),
        ))
    else:
        raise ManifestValidationError(
            "Production Trace materialization received an unsupported MTP "
            f"acceptance policy: {mtp['acceptance_policy']!r}")
    if normalized_expert_routing is not None:
        if normalized_expert_routing["source_kind"] != "explicit_assumption":
            raise ManifestValidationError(
                "Production Trace expert routing currently requires an "
                "explicit seeded assumption")
        if scheduler["enable_expert_parallel"]:
            simulation_argv.append("--enable-expert-parallel")
        simulation_argv.extend((
            "--expert-routing-policy", "seeded_assumption",
            "--expert-routing-source-id",
            normalized_expert_routing["source_id"],
            "--expert-routing-seed",
            str(normalized_expert_routing["root_seed"]),
        ))
        if "hot_expert_count" in normalized_expert_routing:
            simulation_argv.extend((
                "--expert-routing-hot-expert-count",
                str(normalized_expert_routing["hot_expert_count"]),
                "--expert-routing-hot-expert-weight",
                str(normalized_expert_routing["hot_expert_weight"]),
            ))
    kv_lifecycle_replay = simulation["kv_lifecycle_replay"]
    if kv_lifecycle_replay is not None:
        if simulation["deployment"] not in ("colocated", "af_disagg"):
            raise ManifestValidationError(
                "Host-KV lifecycle replay currently requires colocated or "
                "standalone AF Production Trace deployment")
        replay_path = Path(kv_lifecycle_replay["path"])
        try:
            replay_payload = replay_path.read_bytes()
        except OSError as error:
            raise ManifestValidationError(
                "Host-KV lifecycle replay cannot be read during Production "
                "Trace materialization") from error
        actual_replay_sha256 = hashlib.sha256(replay_payload).hexdigest()
        if actual_replay_sha256 != kv_lifecycle_replay["source_sha256"]:
            raise ManifestValidationError(
                "Host-KV lifecycle replay changed after manifest validation")
        simulation_argv.extend((
            "--kv-lifecycle-replay", str(replay_path),
            "--kv-lifecycle-source-sha256",
            kv_lifecycle_replay["source_sha256"],
        ))
    if simulation["deployment"] == "colocated":
        if (simulation["pd_disagg"] is not None
                or simulation["af_disagg"] is not None):
            raise AssertionError(
                "colocated Production Trace simulation retained "
                "disaggregated config")
        decode_hardware_config = None
    elif simulation["deployment"] == "pd_disagg":
        if simulation["af_disagg"] is not None:
            raise AssertionError(
                "PD Production Trace simulation retained AF config")
        pd_disagg = simulation["pd_disagg"]
        if pd_disagg is None:
            raise AssertionError(
                "PD Production Trace simulation lost its deployment config")
        decode_hardware_config = str(hardware["config_path"])
        kv_transfer = pd_disagg["kv_transfer"]
        router = pd_disagg["router"]
        simulation_argv.extend((
            "--decode-hardware-config", str(hardware["config_path"]),
            "--decode-tp-size", str(scheduler["decode_tp_size"]),
            "--num-prefill-instances",
            str(pd_disagg["num_prefill_instances"]),
            "--num-decode-instances",
            str(pd_disagg["num_decode_instances"]),
            "--transfer-algorithm", kv_transfer["algorithm"],
            "--transfer-link", kv_transfer["link"],
            "--router-resource-id", router["resource_id"],
            "--router-dispatch-service-time-s",
            str(router["dispatch_service_time_s"]),
        ))
    elif simulation["deployment"] == "af_disagg":
        if simulation["pd_disagg"] is not None:
            raise AssertionError(
                "AF Production Trace simulation retained PD config")
        af_disagg = simulation["af_disagg"]
        if af_disagg is None:
            raise AssertionError(
                "AF Production Trace simulation lost its deployment config")
        decode_hardware_config = None
        activation_transfer = af_disagg["activation_transfer"]
        router = af_disagg["router"]
        simulation_argv.extend((
            "--ffn-tp-size", str(af_disagg["ffn_tp_size"]),
            "--activation-transfer-algorithm",
            activation_transfer["algorithm"],
            "--activation-transfer-link", activation_transfer["link"],
            "--router-resource-id", router["resource_id"],
            "--router-dispatch-service-time-s",
            str(router["dispatch_service_time_s"]),
        ))
    elif simulation["deployment"] == "pd_af_disagg":
        pd_disagg = simulation["pd_disagg"]
        af_disagg = simulation["af_disagg"]
        if pd_disagg is None or af_disagg is None:
            raise AssertionError(
                "PD+AF Production Trace simulation lost its deployment "
                "configuration")
        decode_hardware_config = str(hardware["config_path"])
        kv_transfer = pd_disagg["kv_transfer"]
        activation_transfer = af_disagg["activation_transfer"]
        router = pd_disagg["router"]
        simulation_argv.extend((
            "--decode-hardware-config", str(hardware["config_path"]),
            "--decode-tp-size", str(scheduler["decode_tp_size"]),
            "--num-prefill-instances",
            str(pd_disagg["num_prefill_instances"]),
            "--num-decode-instances",
            str(pd_disagg["num_decode_instances"]),
            "--transfer-algorithm", kv_transfer["algorithm"],
            "--transfer-link", kv_transfer["link"],
            "--ffn-tp-size", str(af_disagg["ffn_tp_size"]),
            "--activation-transfer-algorithm",
            activation_transfer["algorithm"],
            "--activation-transfer-link", activation_transfer["link"],
            "--router-resource-id", router["resource_id"],
            "--router-dispatch-service-time-s",
            str(router["dispatch_service_time_s"]),
        ))
    else:
        raise ManifestValidationError(
            "unknown Production Trace deployment during materialization: "
            f"{simulation['deployment']!r}")
    request_id_field = field_mapping["request_id"]
    if request_id_field is not None:
        simulation_argv.extend(("--trace-request-id-field", request_id_field))
    if success_filter["mode"] == "field_equals":
        simulation_argv.extend((
            "--trace-success-field", success_filter["field_name"],
            "--trace-success-value-json",
            json.dumps(success_filter["accepted_value"]),
        ))
    elif success_filter["mode"] == "upstream_attested":
        pass
    else:
        raise ManifestValidationError(
            f"unknown Production Trace success filter: "
            f"{success_filter['mode']!r}")
    for option, contract in (
        ("slo", cell.analysis["slo_contract"]),
        ("interactivity-tgs", cell.analysis["interactivity_tgs_contract"]),
    ):
        if contract is not None:
            simulation_argv.extend((
                f"--{option}-config", str(contract["path"]),
                f"--{option}-config-sha256", contract["source_sha256"],
            ))
    simulation_argv.extend(("--output", outputs["sweep"]))
    simulation_inputs = {
        "trace": str(cell.trace_path),
        "model": str(cell.model),
        "hardware_config": str(hardware["config_path"]),
        "kv_lifecycle_replay": (
            None
            if kv_lifecycle_replay is None
            else str(kv_lifecycle_replay["path"])
        ),
        "slo_contract": (
            None
            if cell.analysis["slo_contract"] is None
            else str(cell.analysis["slo_contract"]["path"])
        ),
        "interactivity_tgs_contract": (
            None
            if cell.analysis["interactivity_tgs_contract"] is None
            else str(cell.analysis["interactivity_tgs_contract"]["path"])
        ),
    }
    if decode_hardware_config is not None:
        simulation_inputs["decode_hardware_config"] = decode_hardware_config
    commands = {
        "production_trace_simulation": {
            "kind": "production_trace_simulation",
            "argv": simulation_argv,
            "inputs": simulation_inputs,
            "expert_routing": (
                None if routing_manifest is None else dict(routing_manifest)),
            "mtp_scenario": dict(mtp),
            "outputs": {"sweep": outputs["sweep"]},
        },
        "production_trace_report": {
            "kind": "production_trace_report",
            "argv": [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluation" / "serving"
                    / "production_trace_report.py"),
                "--input", outputs["sweep"],
                "--output", outputs["report_markdown"],
            ],
            "inputs": {"sweep": outputs["sweep"]},
            "outputs": {"report_markdown": outputs["report_markdown"]},
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    return {
        "version": 2,
        "experiment_type": "production_trace",
        "workflow": cell.input_kind,
        "cell_id": cell.id,
        "output_root": str(output_root),
        "input": {
            "kind": cell.input_kind,
            "trace": dict(trace),
            "arrival_rate_scales": cell.arrival_rate_scales,
        },
        "model": {
            "path": str(cell.model),
            "config_sha256": cell.model_config_sha256,
        },
        "simulation": _json_ready(simulation),
        "analysis": _json_ready(cell.analysis),
        "environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        },
        "validation_status": {
            "trace_replay": "validated_conservation",
            "scheduler_control_plane": "native_pinned_vllm",
            "expert_routing": (
                "not_applicable"
                if routing_manifest is None
                else routing_manifest["validation_status"]
            ),
            "mtp_acceptance": mtp["validation_status"],
            "performance_timing": performance_timing_status(
                simulation["launch_policy"],
                (hardware["hardware_id"],),
            ),
        },
        "outputs": outputs,
        "commands": commands,
        "command_manifest_path": outputs["command_manifest"],
    }


def _materialize_colocated_cell(
    cell: ExperimentCell,
    output_root: Path,
    *,
    tokenizer: Any,
) -> dict[str, Any]:
    if cell.parallelism is None or cell.gpus is None:
        raise ManifestValidationError(f"{cell.id}: colocated config is incomplete")
    parallelism = dict(cell.parallelism)
    colocated_tp_size = parallelism["tensor_parallel_size"]
    normalized_parallelism = _require_vllm_baseline_engine_args(
        parallelism,
        f"{cell.id}.parallelism",
        tensor_parallel_size=colocated_tp_size,
        num_requests=cell.workload["num_requests"],
    )
    hardware_config = _require_existing_file(
        cell.prediction["hw_config"],
        f"{cell.id}.prediction.hw_config",
    )
    if normalized_parallelism["enable_expert_parallel"]:
        _require_vllm_baseline_moe_config(
            cell.moe,
            f"{cell.id}.moe",
        )
        _require_validated_ep2_model_hardware(cell.model, hardware_config)
    artifact = create_workload_artifact(
        name=cell.workload["name"],
        num_requests=cell.workload["num_requests"],
        prompt_len=cell.workload["prompt_len"],
        output_len=cell.workload["output_len"],
        sampling=cell.sampling,
        tokenizer=tokenizer,
        seed=cell.workload["seed"],
    )
    output_root.mkdir(parents=True, exist_ok=True)
    workload_path = output_root / "workload.json"
    save_workload_artifact(artifact, workload_path)
    outputs = _cell_outputs(output_root)
    endpoint = _http_endpoint(port=8000)
    colocated_trace = output_root / "traces/colocated_trace.json"
    colocated_trace.parent.mkdir(parents=True, exist_ok=True)
    commands = materialize_workload_commands(
        workload_path=workload_path,
        model=cell.model,
        tokenizer=tokenizer,
        deployment=cell.deployment,
        sampling=cell.sampling,
        colocated_hw_config=cell.prediction["hw_config"],
        colocated_prediction_output=outputs["prediction"],
        colocated_tp_size=colocated_tp_size,
        colocated_engine_args=normalized_parallelism,
        request_rate=cell.benchmark["request_rate"],
        attention_kv_layout="paged_varlen",
        attention_kv_distribution="uniform",
        expert_routing=cell.expert_routing,
    )
    normalized_engine_args = dict(
        commands["colocated_midlayer_simulation"]["engine_args"])
    benchmark_command = dict(commands["real_vllm_benchmark"])
    benchmark_command["argv"] = list(benchmark_command["argv"]) + [
        "--base-url", endpoint["base_url"],
        "--save-result",
        "--result-dir", str(output_root),
        "--result-filename", "ground_truth.json",
    ]
    benchmark_command["endpoint"] = endpoint
    benchmark_command["outputs"] = {"ground_truth": outputs["ground_truth"]}
    environment = _cell_environment(cell)
    materialized = {
        "version": 1,
        "cell_id": cell.id,
        "case_id": cell.case_id,
        "deployment": cell.deployment,
        "model": str(cell.model),
        "model_type": cell.model_type,
        "output_root": str(output_root),
        "gpus": list(cell.gpus),
        "parallelism": normalized_engine_args,
        "moe": {} if cell.moe is None else dict(cell.moe),
        "environment": environment,
        "endpoint": endpoint,
        "workload": {
            "path": str(workload_path),
            "hash": artifact.hash,
            "num_requests": artifact.num_requests,
            "requested_num_requests": artifact.requested_num_requests,
        },
        "sampling": dict(artifact.sampling),
        "token_controls": dict(artifact.token_controls),
        "mtp_scenario": dict(
            commands["colocated_midlayer_simulation"]["mtp_scenario"]),
        "expert_routing": (
            None
            if commands["colocated_midlayer_simulation"][
                "expert_routing"] is None
            else dict(commands["colocated_midlayer_simulation"][
                "expert_routing"])
        ),
        "traces": {
            "colocated_trace": str(colocated_trace),
            "colocated_sidecar": f"{colocated_trace}_scheduler.jsonl",
        },
        "outputs": outputs,
        "commands": {
            "vllm_serve": {
                "kind": "vllm_serve",
                "argv": _vllm_serve_argv(
                    cell,
                    endpoint=endpoint,
                    engine_args=normalized_engine_args,
                ),
                "environment": environment,
                "endpoint": endpoint,
                "mtp_scenario": dict(
                    commands["colocated_midlayer_simulation"][
                        "mtp_scenario"]),
            },
            "vllm_benchmark": benchmark_command,
            "colocated_midlayer_simulation": commands["colocated_midlayer_simulation"],
            "comparison": _comparison_command(
                output_root / "command_manifest.json",
                outputs,
                diagnostic_mode="colocated",
            ),
        },
        "command_manifest_path": str(output_root / "command_manifest.json"),
    }
    return materialized





def _cell_environment(cell: ExperimentCell) -> dict[str, str]:
    if cell.deployment == "disaggregated":
        if cell.prefill is None or cell.decode is None:
            raise ManifestValidationError(
                f"{cell.id}: disaggregated GPU config is incomplete")
        gpus = list(cell.prefill["gpus"]) + list(cell.decode["gpus"])
    elif cell.deployment == "colocated":
        if cell.gpus is None:
            raise ManifestValidationError(f"{cell.id}: colocated GPUs are missing")
        gpus = list(cell.gpus)
    else:
        raise ManifestValidationError(
            f"unsupported experiment deployment: {cell.deployment!r}")
    return _environment_for_gpus(gpus)



def _environment_for_gpus(gpus: list[Any]) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpus),
        "VLLM_WORKER_MULTIPROC_METHOD": "fork",
    }


def _http_endpoint(*, port: int, host: str = "127.0.0.1") -> dict[str, Any]:
    base_url = f"http://{host}:{port}"
    return {
        "host": host,
        "port": int(port),
        "base_url": base_url,
        "health_url": f"{base_url}/v1/models",
    }


def _vllm_serve_argv(
    cell: ExperimentCell,
    *,
    endpoint: dict[str, Any],
    engine_args: dict[str, Any],
) -> list[str]:
    argv = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "serve", str(cell.model),
        "--host", str(endpoint["host"]),
        "--port", str(endpoint["port"]),
    ]
    normalized_engine_args = dict(engine_args)
    _append_engine_args(argv, normalized_engine_args)
    _append_moe_engine_args(argv, cell.moe, normalized_engine_args)
    argv.append("--no-enable-prefix-caching")
    return argv


def _pd_serve_argv(
    cell: ExperimentCell,
    *,
    instance: str,
    endpoint: dict[str, Any],
    router: dict[str, Any],
) -> list[str]:
    if instance == "prefill":
        instance_config = cell.prefill
        role = "kv_producer"
        kv_rank = 0
    elif instance == "decode":
        instance_config = cell.decode
        role = "kv_consumer"
        kv_rank = 1
    else:
        raise ManifestValidationError(f"unknown PD instance role: {instance!r}")
    if instance_config is None:
        raise ManifestValidationError(f"{cell.id}: {instance} config is missing")
    kv_ip = instance_config["kv_ip"]
    if not isinstance(kv_ip, str) or not kv_ip:
        raise ManifestValidationError(
            f"{cell.id}: {instance}.kv_ip must be explicit for real PD")
    handoff_transport = _pd_handoff_transport(cell.kv_connector)
    extra_config = {
        "http_ip": endpoint["host"],
        "http_port": str(endpoint["port"]),
    }
    if handoff_transport == "p2p_request_id":
        extra_config["send_type"] = "PUT_ASYNC"
    elif handoff_transport == "openai_kv_transfer_params":
        pass
    else:
        raise AssertionError(f"unsupported PD transport: {handoff_transport}")
    kv_config = {
        "kv_connector": cell.kv_connector,
        "kv_role": role,
        "kv_rank": kv_rank,
        "kv_parallel_size": 2,
        "kv_ip": kv_ip,
        "kv_port": instance_config["kv_port"],
        "kv_connector_extra_config": extra_config,
    }
    if instance_config["kv_buffer_size"] is not None:
        kv_config["kv_buffer_size"] = instance_config["kv_buffer_size"]
    argv = [
        "vllm", "serve", str(cell.model),
        "--host", str(endpoint["host"]),
        "--port", str(endpoint["port"]),
        "--kv-transfer-config", json.dumps(kv_config, sort_keys=True),
    ]
    engine_args = dict(instance_config["engine_args"])
    _append_engine_args(argv, engine_args)
    _append_moe_engine_args(argv, cell.moe, engine_args)
    return argv


def _pd_proxy_serve_argv(
    cell: ExperimentCell,
    *,
    endpoint: dict[str, Any],
    prefill_endpoint: dict[str, Any],
    decode_endpoint: dict[str, Any],
    router: dict[str, Any],
    handoff_log: Path,
    run_identity: dict[str, object],
) -> list[str]:
    if cell.prefill is None or cell.decode is None:
        raise ManifestValidationError(f"{cell.id}: PD instances are missing")
    prefill_kv_ip = cell.prefill["kv_ip"]
    decode_kv_ip = cell.decode["kv_ip"]
    if not isinstance(prefill_kv_ip, str) or not prefill_kv_ip:
        raise ManifestValidationError(f"{cell.id}: prefill.kv_ip must be explicit")
    if not isinstance(decode_kv_ip, str) or not decode_kv_ip:
        raise ManifestValidationError(f"{cell.id}: decode.kv_ip must be explicit")
    return [
        sys.executable, str(ROOT / "scripts" / "experiments" / "request_router.py"),
        "--host", str(endpoint["host"]),
        "--port", str(endpoint["port"]),
        "--prefill-base-url", prefill_endpoint["base_url"],
        "--decode-base-url", decode_endpoint["base_url"],
        "--model", str(cell.model),
        "--handoff-transport", _pd_handoff_transport(cell.kv_connector),
        "--prefill-kv-ip", prefill_kv_ip,
        "--prefill-kv-port", str(cell.prefill["kv_port"]),
        "--decode-kv-ip", decode_kv_ip,
        "--decode-kv-port", str(cell.decode["kv_port"]),
        "--handoff-log", str(handoff_log),
        "--run-id", _require_text_value(run_identity, "run_id"),
        "--run-origin-monotonic-ns", str(_require_positive_int_value(
            run_identity["run_origin_monotonic_ns"], "PD run origin")),
        "--host-boot-id", _require_text_value(run_identity, "host_boot_id"),
    ]


def _pd_handoff_transport(kv_connector: str | None) -> str:
    if kv_connector == "P2pNcclConnector":
        return "p2p_request_id"
    if kv_connector == "MooncakeConnectorV1":
        return "openai_kv_transfer_params"
    raise ManifestValidationError(
        "real PD routing supports only P2pNcclConnector or MooncakeConnectorV1")




def _append_engine_args(
    argv: list[str],
    engine_args: dict[str, Any],
) -> None:
    for key, value in engine_args.items():
        if value is None:
            raise ManifestValidationError(
                f"engine argument {key!r} cannot be null")
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend([flag, str(value)])


def _append_moe_engine_args(
    argv: list[str],
    moe: dict[str, Any] | None,
    engine_args: dict[str, Any],
) -> None:
    if engine_args["enable_expert_parallel"]:
        if moe is None:
            raise ManifestValidationError(
                "expert-parallel execution requires explicit MoE settings")
        argv.extend(["--all2all-backend", moe["all2all_backend"]])
        if moe["eplb_enabled"]:
            argv.append("--enable-eplb")



def _cell_outputs(output_root: Path) -> dict[str, str]:
    return {
        "command_manifest": str(output_root / "command_manifest.json"),
        "status": str(output_root / "status.json"),
        "ground_truth": str(output_root / "ground_truth.json"),
        "pd_handoffs": str(output_root / "pd_handoffs.jsonl"),
        "prediction": str(output_root / "prediction.json"),
        "comparison_report": str(output_root / "comparison_report.json"),
        "comparison_markdown": str(output_root / "comparison_report.md"),
        "logs": str(output_root / "logs"),
    }


def _production_trace_sweep_outputs(output_root: Path) -> dict[str, str]:
    return {
        "command_manifest": str(output_root / "command_manifest.json"),
        "status": str(output_root / "status.json"),
        "sweep": str(output_root / "production_trace_sweep.json"),
        "report_markdown": str(
            output_root / "production_trace_report.md"),
        "logs": str(output_root / "logs"),
    }




def _comparison_command(
    command_manifest_path: Path,
    outputs: dict[str, str],
    *,
    diagnostic_mode: str,
) -> dict[str, Any]:
    return {
        "kind": "real_vllm_comparison",
        "argv": [
            sys.executable,
            str(ROOT / "scripts" / "evaluation" / "serving" / "run_serving_metric_evaluation.py"),
            "cell",
            "--command-manifest", str(command_manifest_path),
            "--ground-truth", outputs["ground_truth"],
            "--prediction", outputs["prediction"],
            "--output", outputs["comparison_report"],
            "--markdown", outputs["comparison_markdown"],
            "--diagnostic-mode", diagnostic_mode,
        ],
        "inputs": {
            "command_manifest": str(command_manifest_path),
            "ground_truth": outputs["ground_truth"],
            "prediction": outputs["prediction"],
        },
        "outputs": {
            "comparison_report": outputs["comparison_report"],
            "comparison_markdown": outputs["comparison_markdown"],
        },
        "diagnostic_mode": diagnostic_mode,
    }



def _attach_provenance(
    command_manifest: dict[str, Any],
    project_root: Path,
    invocation: list[str],
) -> None:
    environment = dict(command_manifest["environment"])
    if "experiment_type" in command_manifest:
        experiment_type = command_manifest["experiment_type"]
        if experiment_type != "production_trace":
            raise ManifestValidationError(
                f"unknown experiment type: {experiment_type!r}")
        code_identity = git_code_identity(ROOT)
        git_commit = code_identity["git_commit"]
        git_status_short = list(code_identity["git_status_short"])
        git_diff_hash = code_identity["git_diff_hash"]
        cuda_driver_summary = {
            "nvidia_smi": "not_queried_cpu_only",
            "device_access": "forbidden",
        }
    else:
        git_commit = _git_commit(project_root)
        git_status_short = _git_status_short(project_root)
        git_diff_hash = _git_diff_hash(project_root)
        cuda_driver_summary = _cuda_driver_summary()
    command_manifest["provenance"] = {
        "version": 1,
        "git_commit": git_commit,
        "git_dirty": bool(git_status_short),
        "git_status_short": git_status_short,
        "git_diff_hash": git_diff_hash,
        "hostname": socket.gethostname(),
        "cuda_driver_summary": cuda_driver_summary,
        "python_executable": sys.executable,
        "vllm_version": _vllm_version(),
        "midlayer_command": list(invocation),
        "environment_overrides": environment,
        "gpu_ids": _manifest_gpu_ids(command_manifest),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _manifest_gpu_ids(command_manifest: dict[str, Any]) -> list[int]:
    if "experiment_type" in command_manifest:
        experiment_type = command_manifest["experiment_type"]
        if experiment_type == "production_trace":
            return []
        raise ManifestValidationError(
            f"unknown experiment type: {experiment_type!r}")
    deployment = command_manifest["deployment"]
    if deployment == "disaggregated":
        return list(command_manifest["prefill"]["gpus"]) + list(
            command_manifest["decode"]["gpus"])
    elif deployment == "colocated":
        return list(command_manifest["gpus"])
    else:
        raise ManifestValidationError(
            f"unknown experiment deployment: {deployment!r}")


def _git_commit(project_root: Path) -> str:
    for cwd in (project_root, ROOT):
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(cwd),
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "git is required to record experiment provenance") from exc
        if result.returncode == 0:
            commit = result.stdout.strip()
            if not commit:
                raise RuntimeError("git returned an empty commit identifier")
            return commit
    raise RuntimeError(
        f"cannot determine Git commit from {project_root} or {ROOT}")


def git_code_identity(repository_root: str | Path) -> dict[str, Any]:
    """Hash HEAD and worktree contents without writing shared Git metadata."""
    root = Path(repository_root).resolve()
    if not (root / ".git").exists() and (root / "SOURCE_COMMIT").is_file():
        return _snapshot_code_identity(root)
    with tempfile.TemporaryDirectory(prefix="midlayer-git-readonly-") as directory:
        sandbox = Path(directory)
        (sandbox / "hooks").mkdir()
        (sandbox / "lfs").mkdir()
        return _git_code_identity(root, sandbox)


def _git_code_identity(root: Path, sandbox: Path) -> dict[str, Any]:
    read_git = partial(_git_bytes, sandbox=sandbox)
    index_bytes = read_git(root, ("rev-parse", "--git-path", "index"), allow_empty=False)
    index_path = Path(os.fsdecode(index_bytes.rstrip(b"\n")))
    if not index_path.is_absolute():
        index_path = root / index_path
    private_index = sandbox / "worktree-index"
    try:
        shutil.copyfile(index_path, private_index)
    except FileNotFoundError:
        # Git treats a missing index as empty; point it at the private path so
        # any later refresh remains in the sandbox.
        pass
    git = partial(_git_bytes, sandbox=sandbox, index_file=private_index)
    commit_bytes = git(
        root, ("rev-parse", "--verify", "HEAD"), allow_empty=False)
    try:
        commit = commit_bytes.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise RuntimeError("git returned a non-ASCII commit identifier") from error
    if (len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)):
        raise RuntimeError(
            f"git returned an invalid commit identifier: {commit!r}")

    tracked_paths = git(root, ("ls-files", "-z"), allow_empty=True)
    _validate_git_content_attributes(root, tracked_paths, sandbox)
    common_bytes = git(root, ("rev-parse", "--git-common-dir"), allow_empty=False)
    common_path = Path(os.fsdecode(common_bytes.rstrip(b"\n")))
    if not common_path.is_absolute():
        common_path = root / common_path
    lfs_objects = _git_lfs_objects(root, common_path.resolve(), sandbox)
    try:
        lfs_metadata = lfs_objects.lstat()
    except FileNotFoundError:
        lfs_metadata = None
    if lfs_metadata is not None:
        if stat.S_ISLNK(lfs_metadata.st_mode):
            lfs_metadata = lfs_objects.stat()
        if not stat.S_ISDIR(lfs_metadata.st_mode):
            raise RuntimeError(f"Git LFS object storage is not a directory: {lfs_objects}")
    # Git LFS clean writes only to the private, initially empty store configured
    # by _git_bytes. It needs no objects from the shared store to hash files.
    index_flags = git(root, ("ls-files", "-v", "-z"), allow_empty=True)
    for entry in index_flags.split(b"\0"):
        if not entry:
            continue
        if len(entry) < 3 or entry[1:2] != b" ":
            raise RuntimeError("git returned malformed index flags")
        tag = entry[0]
        if tag == ord("S") or ord("a") <= tag <= ord("z"):
            raise RuntimeError(
                "git code identity cannot verify files marked assume-unchanged "
                "or skip-worktree")

    status_bytes = git(
        root,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        allow_empty=True,
    )
    try:
        status_text = status_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("git status contains a non-UTF-8 path") from error
    status_lines = tuple(status_text.splitlines())

    tracked_diff = git(
        root,
        ("diff", "HEAD", "--binary", "--no-ext-diff", "--no-textconv"),
        allow_empty=True,
    )
    staged_diff = git(
        root,
        ("diff", "--cached", "HEAD", "--binary", "--no-ext-diff", "--no-textconv"),
        allow_empty=True,
    )
    # An index stat cache can hide same-size edits with a restored mtime. A
    # fresh temporary index forces Git to compare tracked worktree contents.
    with tempfile.TemporaryDirectory(prefix="midlayer-git-identity-") as directory:
        fresh_index = Path(directory) / "index"
        git(root, ("read-tree", "HEAD"), allow_empty=True, index_file=fresh_index)
        verified_diff = git(
            root, ("diff", "HEAD", "--binary", "--no-ext-diff", "--no-textconv"),
            allow_empty=True, index_file=fresh_index)
        verified_names = git(
            root, ("diff", "HEAD", "--name-only", "-z", "--no-ext-diff", "--no-textconv"),
            allow_empty=True, index_file=fresh_index) if verified_diff else b""
    status_paths: set[bytes] = set()
    if verified_diff:
        raw_status = git(
            root, ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
            allow_empty=True)
        entries = iter(raw_status.split(b"\0"))
        for entry in entries:
            if not entry:
                continue
            if len(entry) < 4 or entry[2:3] != b" ":
                raise RuntimeError("git returned malformed porcelain status")
            status_paths.add(entry[3:])
            if b"R" in entry[:2] or b"C" in entry[:2]:
                original = next(entries, b"")
                if not original:
                    raise RuntimeError("git returned an incomplete rename status")
                status_paths.add(original)
    verified_paths = {name for name in verified_names.split(b"\0") if name}
    raw_untracked = git(
        root,
        ("ls-files", "--others", "--exclude-standard", "-z"),
        allow_empty=True,
    )
    untracked_paths = tuple(
        sorted(path for path in raw_untracked.split(b"\0") if path))
    digest = hashlib.sha256()
    digest.update(b"midlayer-git-worktree-content-v1\0")
    _update_framed_digest(digest, b"tracked_diff", tracked_diff)
    _update_framed_digest(digest, b"staged_diff", staged_diff)
    _update_framed_digest(digest, b"verified_worktree_diff", verified_diff)
    for raw_path in untracked_paths:
        relative_path = Path(os.fsdecode(raw_path))
        path = root / relative_path
        try:
            metadata = path.lstat()
        except OSError as error:
            raise RuntimeError(
                f"cannot read untracked path for code identity: {path}") from error
        if stat.S_ISREG(metadata.st_mode):
            kind = b"regular"
            try:
                content = path.read_bytes()
            except OSError as error:
                raise RuntimeError(
                    f"cannot read untracked file for code identity: {path}") \
                    from error
        elif stat.S_ISLNK(metadata.st_mode):
            kind = b"symlink"
            try:
                content = os.fsencode(os.readlink(path))
            except OSError as error:
                raise RuntimeError(
                    f"cannot read untracked symlink for code identity: {path}") \
                    from error
        else:
            raise RuntimeError(
                "code identity supports only regular files and symlinks for "
                f"untracked paths: {path}")
        _update_framed_digest(digest, b"untracked_path", raw_path)
        _update_framed_digest(digest, b"untracked_kind", kind)
        _update_framed_digest(
            digest,
            b"untracked_mode",
            str(stat.S_IMODE(metadata.st_mode)).encode("ascii"),
        )
        _update_framed_digest(digest, b"untracked_content", content)

    if status_lines or verified_diff:
        worktree_state = "dirty_not_reproducible_by_commit_alone"
    else:
        worktree_state = "clean_reproducible_by_commit"
    return {
        "repository_root": str(root),
        "git_commit": commit,
        "worktree_state": worktree_state,
        "changed_path_count": len(status_lines) + len(verified_paths - status_paths),
        "git_status_short": status_lines,
        "status_porcelain_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "git_diff_hash": digest.hexdigest(),
    }


def _snapshot_code_identity(root: Path) -> dict[str, Any]:
    source_commit = (root / "SOURCE_COMMIT").read_text(encoding="utf-8").strip()
    if (
        len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise RuntimeError("SOURCE_COMMIT does not contain a full commit")
    status_bytes = b""
    digest = hashlib.sha256()
    digest.update(b"midlayer-execution-snapshot-v1\0")
    digest.update(source_commit.encode("ascii"))
    return {
        "repository_root": str(root),
        "git_commit": source_commit,
        "worktree_state": "execution_snapshot_declared_commit",
        "changed_path_count": 0,
        "git_status_short": (),
        "status_porcelain_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "git_diff_hash": digest.hexdigest(),
    }


def _update_framed_digest(
    digest: Any,
    label: bytes,
    payload: bytes,
) -> None:
    for value in (label, payload):
        digest.update(len(value).to_bytes(8, byteorder="big"))
        digest.update(value)


def _validate_git_content_attributes(root: Path, tracked_paths: bytes, sandbox: Path) -> None:
    """Reject transformations that could hide actual checked-out code bytes."""
    paths = tuple(path for path in tracked_paths.split(b"\0") if path)
    names = (b"filter", b"text", b"eol", b"working-tree-encoding", b"ident")
    if not paths:
        return
    raw = _git_bytes(
        root, ("check-attr", "-z", "--stdin", *(name.decode("ascii") for name in names)),
        allow_empty=False, stdin=tracked_paths, sandbox=sandbox,
        index_file=sandbox / "worktree-index")
    fields = raw.split(b"\0")
    if fields[-1:] != [b""] or len(fields) != len(paths) * len(names) * 3 + 1:
        raise RuntimeError("git returned malformed content attributes")
    offset = 0
    for path in paths:
        for name in names:
            actual_path, actual_name, value = fields[offset:offset + 3]
            offset += 3
            if actual_path != path or actual_name != name:
                raise RuntimeError("git returned mismatched content attributes")
            if value in (b"unspecified", b"unset"):
                continue
            if (name == b"filter" and value == b"lfs"
                    and path.startswith((b"kernel_dataset/", b"launch_data/", b"serving_dataset/"))):
                continue
            raise RuntimeError("git code identity cannot verify unsupported Git content attributes")


def _git_lfs_objects(root: Path, common_path: Path, sandbox: Path) -> Path:
    """Resolve Git LFS's optional storage configuration without writing it."""
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
        if name in environment:
            del environment[name]
    completed = subprocess.run(
        ("git", "-C", str(root), "--work-tree", str(root),
         "-c", f"core.hooksPath={sandbox / 'hooks'}",
         "config", "--path", "--get", "lfs.storage"),
        capture_output=True, env=environment,
    )
    if completed.returncode == 1:
        return common_path / "lfs" / "objects"
    if completed.returncode != 0:
        raise RuntimeError("cannot determine Git LFS storage configuration")
    value = os.fsdecode(completed.stdout.rstrip(b"\n"))
    if not value:
        raise RuntimeError("Git LFS storage configuration is empty")
    storage = Path(value)
    if not storage.is_absolute():
        storage = common_path / storage
    return storage.resolve() / "objects"


def _git_bytes(
    repository_root: Path,
    arguments: tuple[str, ...],
    *,
    allow_empty: bool,
    sandbox: Path,
    index_file: Path | None = None,
    stdin: bytes | None = None,
) -> bytes:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
        if name in environment:
            del environment[name]
    if index_file is not None:
        environment["GIT_INDEX_FILE"] = str(index_file)
    try:
        completed = subprocess.run(
            ("git", "-C", str(repository_root), "--work-tree", str(repository_root),
             "-c", f"core.hooksPath={sandbox / 'hooks'}",
             "-c", f"lfs.storage={sandbox / 'lfs'}",
             "-c", "core.trustctime=true", "-c", "core.checkStat=default",
             "-c", "core.fsmonitor=false", "-c", "core.autocrlf=false",
             "-c", "filter.lfs.clean=git-lfs clean -- %f",
             "-c", "filter.lfs.process=git-lfs filter-process",
             "-c", "filter.lfs.required=true", *arguments),
            input=stdin,
            capture_output=True,
            env=environment,
        )
    except FileNotFoundError as error:
        raise RuntimeError("git is required to record code identity") from error
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"git {' '.join(arguments)} failed for {repository_root}: {stderr}")
    output = completed.stdout
    if not allow_empty and not output.strip():
        raise RuntimeError(
            f"git {' '.join(arguments)} returned empty output for "
            f"{repository_root}")
    return output


def _git_status_short(project_root: Path) -> list[str]:
    for cwd in (project_root, ROOT):
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=str(cwd),
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "git is required to record experiment provenance") from exc
        if result.returncode == 0:
            return [line for line in result.stdout.splitlines() if line]
    raise RuntimeError(
        f"cannot determine Git status from {project_root} or {ROOT}")


def _git_diff_hash(project_root: Path) -> str:
    for cwd in (project_root, ROOT):
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD", "--binary"],
                cwd=str(cwd),
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "git is required to record experiment provenance") from exc
        if result.returncode == 0:
            import hashlib
            return hashlib.sha256(result.stdout.encode()).hexdigest()
    raise RuntimeError(
        f"cannot determine Git diff from {project_root} or {ROOT}")


def _cuda_driver_summary() -> dict[str, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {"nvidia_smi": "unavailable", "stderr": "nvidia-smi not found"}
    if result.returncode != 0:
        stderr = result.stderr
        if not isinstance(stderr, str):
            raise TypeError("nvidia-smi stderr must be captured as text")
        return {
            "nvidia_smi": "unavailable",
            "stderr": stderr.strip(),
        }
    stdout = result.stdout
    if not isinstance(stdout, str):
        raise TypeError("nvidia-smi stdout must be captured as text")
    return {"nvidia_smi": stdout.strip()}


def _vllm_version() -> str:
    try:
        return importlib_metadata.version("vllm")
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"


def validate_vllm_experiment_environment() -> None:
    """Fail before a real run if its Python cannot execute audited vLLM."""

    version = _vllm_version()
    if version != VLLM_SCHEDULER_VERSION:
        raise ManifestValidationError(
            "real Serving experiments require the audited vLLM environment: "
            f"expected={VLLM_SCHEDULER_VERSION}, current={version}")
    distribution = importlib_metadata.distribution("vllm")
    serve_path = Path(distribution.locate_file("vllm/benchmarks/serve.py"))
    if not serve_path.is_file():
        raise ManifestValidationError(
            f"audited vLLM benchmark source is missing: {serve_path}")
    actual_sha256 = hashlib.sha256(serve_path.read_bytes()).hexdigest()
    if actual_sha256 != VLLM_BENCH_SERVE_SHA256:
        raise ManifestValidationError(
            "vLLM benchmark source hash disagrees with the audited contract: "
            f"expected={VLLM_BENCH_SERVE_SHA256}, actual={actual_sha256}")
    cli_path = Path(distribution.locate_file("vllm/entrypoints/cli/main.py"))
    if not cli_path.is_file():
        raise ManifestValidationError(
            f"vLLM CLI module is missing from the active Python: {cli_path}")


def validate_vllm_scheduler_environment() -> None:
    """Preflight only the pinned CPU Scheduler package, not benchmark/CLI."""

    version = _vllm_version()
    if version != VLLM_SCHEDULER_VERSION:
        raise ManifestValidationError(
            "Production Trace experiments require the audited vLLM "
            f"Scheduler environment: expected={VLLM_SCHEDULER_VERSION}, "
            f"current={version}")
    distribution = importlib_metadata.distribution("vllm")
    scheduler_path = Path(distribution.locate_file(
        "vllm/v1/core/sched/scheduler.py"))
    if not scheduler_path.is_file():
        raise ManifestValidationError(
            f"audited vLLM Scheduler source is missing: {scheduler_path}")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _require_dict(raw: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    if key not in raw:
        raise ManifestValidationError(f"{label}: {key} is required")
    value = raw[key]
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{label}: {key} must be an object")
    return dict(value)


def _require_positive_int(raw: dict[str, Any], key: str, label: str) -> int:
    if key not in raw:
        raise ManifestValidationError(f"{label}.{key} is required")
    return _require_positive_int_value(raw[key], f"{label}.{key}")


def _require_text_value(raw: Mapping[str, object], key: str) -> str:
    if key not in raw:
        raise ManifestValidationError(f"PD run identity: {key} is required")
    value = raw[key]
    if not isinstance(value, str) or not value:
        raise ManifestValidationError(
            f"PD run identity: {key} must be non-empty text")
    return value


def _require_positive_int_value(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ManifestValidationError(f"{label} must be an integer >= 1")
    return value


def _require_materialized_expert_routing(
    value: dict[str, Any] | None,
    *,
    routing_required: bool,
    model: str | Path,
) -> dict[str, Any] | None:
    if not isinstance(routing_required, bool):
        raise ManifestValidationError(
            "expert routing requirement must be boolean")
    if not routing_required:
        if value is not None:
            raise ManifestValidationError(
                "expert routing input requires a validated MoE execution")
        return None
    if not isinstance(value, dict):
        raise ManifestValidationError(
            "expert parallel simulation requires explicit expert routing")
    if "source_kind" not in value:
        raise ManifestValidationError(
            "expert routing source_kind is required")
    source_kind = value["source_kind"]
    if source_kind == "explicit_assumption":
        uniform_fields = {"source_kind", "source_id", "root_seed"}
        skew_fields = uniform_fields | {
            "hot_expert_count", "hot_expert_weight"}
        if set(value) not in (uniform_fields, skew_fields):
            raise ManifestValidationError(
                "explicit expert routing fields must be source_kind, "
                "source_id, and root_seed, with both or neither hot-expert "
                "parameters")
        source_id = value["source_id"]
        root_seed = value["root_seed"]
        if not isinstance(source_id, str) or not source_id:
            raise ManifestValidationError(
                "expert routing source_id must be non-empty")
        if (isinstance(root_seed, bool) or not isinstance(root_seed, int)
                or root_seed < 0):
            raise ManifestValidationError(
                "expert routing root_seed must be a non-negative integer")
        if set(value) == skew_fields:
            hot_expert_count = value["hot_expert_count"]
            hot_expert_weight = value["hot_expert_weight"]
            if (isinstance(hot_expert_count, bool)
                    or not isinstance(hot_expert_count, int)
                    or hot_expert_count < 1):
                raise ManifestValidationError(
                    "expert routing hot_expert_count must be a positive "
                    "integer")
            if (isinstance(hot_expert_weight, bool)
                    or not isinstance(hot_expert_weight, int)
                    or hot_expert_weight <= 1):
                raise ManifestValidationError(
                    "expert routing hot_expert_weight must be an integer "
                    "greater than one")
        return dict(value)
    if source_kind == "replay_observation":
        expected = {"source_kind", "path"}
        if set(value) != expected:
            raise ManifestValidationError(
                "replayed expert routing fields must be source_kind and path")
        path = value["path"]
        if not isinstance(path, str) or not path:
            raise ManifestValidationError(
                "expert routing replay path must be non-empty")
        try:
            replay = load_expert_routing_replay(path)
        except (OSError, TypeError, ValueError) as error:
            raise ManifestValidationError(
                f"expert routing replay is invalid: {error}") from error
        return {
            "source_kind": source_kind,
            "path": str(Path(path).resolve()),
            "artifact": replay.manifest(),
        }
    if source_kind == "profile_observation":
        expected = {"source_kind", "source_id", "manifest_path"}
        if set(value) != expected:
            raise ManifestValidationError(
                "profile expert routing fields must be source_kind, "
                "source_id, and manifest_path")
        source_id = value["source_id"]
        manifest_path = value["manifest_path"]
        if not isinstance(source_id, str) or not source_id:
            raise ManifestValidationError(
                "profile expert routing source_id must be non-empty")
        if not isinstance(manifest_path, str) or not manifest_path:
            raise ManifestValidationError(
                "profile expert routing manifest_path must be non-empty")
        model_config = load_model_config(model)
        if model_config.model_id is None:
            executing_model_id = Path(model).resolve().name
        else:
            executing_model_id = model_config.model_id
        try:
            profile = load_expert_frequency_profile(
                manifest_path,
                source_id,
                expected_model_id=executing_model_id,
                expected_moe_layers=sum(
                    model_config.layer_uses_moe(layer_index)
                    for layer_index in range(model_config.num_hidden_layers)
                ),
                expected_num_experts=model_config.num_experts,
            )
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ManifestValidationError(
                f"expert routing profile is invalid: {error}") from error
        return {
            "source_kind": source_kind,
            "source_id": profile.canonical_model_id,
            "manifest_path": str(Path(manifest_path).resolve()),
            "profile": profile.manifest(),
            "runtime_manifest": (
                ExpertRoutingInput.profile_observation(profile).manifest()),
        }
    raise ManifestValidationError(
        f"unknown expert routing source_kind: {source_kind!r}")


def _materialized_expert_routing_identity(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if value["source_kind"] == "explicit_assumption":
        if "hot_expert_count" in value:
            distribution = "stable_hotset_skew_sensitivity_v1"
            hot_expert_count = value["hot_expert_count"]
            hot_expert_weight = value["hot_expert_weight"]
        else:
            distribution = "maximum_entropy_uniform_topk_v1"
            hot_expert_count = None
            hot_expert_weight = None
        return {
            "source_kind": "explicit_assumption",
            "source_id": value["source_id"],
            "algorithm": (
                "sha256_weighted_rejection_topk_without_replacement_"
                "model_step_v2"),
            "distribution": distribution,
            "hot_expert_count": hot_expert_count,
            "hot_expert_weight": hot_expert_weight,
            "root_seed": value["root_seed"],
            "validation_status": "Projected",
        }
    if value["source_kind"] == "replay_observation":
        return {
            "source_kind": "replay_observation",
            "path": value["path"],
            "artifact": value["artifact"],
            "validation_status": "SchemaValidatedInput",
        }
    if value["source_kind"] == "profile_observation":
        runtime_manifest = value["runtime_manifest"]
        return {
            "source_kind": "profile_observation",
            "source_id": value["source_id"],
            "manifest_path": value["manifest_path"],
            "profile": value["profile"],
            "algorithm": runtime_manifest["algorithm"],
            "distribution": runtime_manifest["distribution"],
            "policy_descriptor_sha256": runtime_manifest[
                "policy_descriptor_sha256"],
            "sharing_scope": "model_step_block_signature",
            "validation_status": "SchemaValidatedInput",
        }
    raise ManifestValidationError(
        f"unknown materialized expert routing source: "
        f"{value['source_kind']!r}")
