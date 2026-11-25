"""Assemble the H200 Host timing profiles from one-shot Host profile runs.

The input is one output directory of
``microbench/nvidia_h200/runtime/host_profiles/run_nvidia_h200_host_profiles.sh`` (its
``steps.tsv``, ``provenance/`` and ``profiler_output/microbench/nvidia_h200``
tree).  This module only parses and reduces those microbenchmark results into
the published profile schemas (``src/serving/host_timing.py``); it reads no
Serving duration and derives no prediction.  Every selected value keeps the
rule the A800 profiles were assembled with:

* Host launch: ``eager:512`` and ``graph_replay:512`` submit medians of
  ``host_launch_r1`` (the run whose eager value the runtime-prepare runners
  subtracted); later repeats are kept as evidence only.
* Request path: primitives run ``r1``; the admission/IPC regions take the
  median across their independent runs; delivery is the delivery-region
  probe's own selection (per-fanout median of repeated setups, least squares).
* Runtime prepare: the A800 profile's mechanism set, each at its
  ``runtime_prepare_ns_per_call``; the collective prepares average both ranks
  and both message sizes, as the A800 profile does.

A measurement is used for a vLLM version other than the one it was measured
under only through an explicit ``substitution`` record naming the reason.
"""

from __future__ import annotations

from dataset_pipeline.serving_dataset.json_shape import file_sha256

import csv
import hashlib
import json
from pathlib import Path
from statistics import median


HARDWARE_ID = "nvidia_h200"
VLLM_VERSION_TAGS = {"0.11.0": "vllm011", "0.30.0": "vllm030"}
_RAW_PREFIX = Path("profiler_output/microbench/nvidia_h200")
_HOST_LAUNCH_RUNNER = "microbench/nvidia_h200/runtime/launch/torch_eager_graph_runner.py"
# The A800 runtime-prepare profile's enabled mechanism set.
_RUNTIME_PREPARE_MECHANISMS = (
    "compiled_fx_backend_entry",
    "compiled_fx_moe_wrapper_control",
    "fused_attention_source_wrapper_prepare",
    "moe_gating_prepare",
    "moe_unquant_mlp_wrapper_prepare",
    "reshape_cache_prepare",
    "unquantized_gemm_prepare",
)
_COLLECTIVE_MECHANISMS = {
    "all_gather": "collective_all_gather_prepare",
    "all_reduce": "collective_all_reduce_prepare",
}
_ADMISSION_RUNS = (
    "admission_ipc_region_r1", "admission_ipc_region_r2", "admission_ipc_region_r3")


def sha256_file(path: Path) -> str:
    return file_sha256(path)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must hold a JSON object")
    return value


class HostProfileRun:
    """One validated ``run_nvidia_h200_host_profiles.sh`` output directory."""

    def __init__(self, run_dir: Path, repository_root: Path) -> None:
        self.run_dir = run_dir.resolve(strict=True)
        self.repository_root = repository_root.resolve(strict=True)
        rows = list(csv.DictReader(
            (self.run_dir / "steps.tsv").read_text().splitlines(), delimiter="\t"))
        self.step_returncodes = {row["label"]: int(row["rc"]) for row in rows}
        if len(self.step_returncodes) != len(rows):
            raise ValueError(f"duplicate step labels in {self.run_dir}/steps.tsv")
        provenance = self.run_dir / "provenance"
        self.source_commit = (provenance / "source_commit").read_text().strip()
        self.versions = _json(provenance / "versions.json")
        self.vllm_version = str(self.versions["vllm"]).split("+")[0]
        self.hostname = (provenance / "hostname").read_text().strip()
        self.host_cpu = self._lscpu_model_name(provenance / "lscpu.txt")
        self.operating_system = self._operating_system(provenance)
        self._verify_checksums()

    @staticmethod
    def _lscpu_model_name(path: Path) -> str:
        names = [line.split(":", 1)[1].strip() for line in path.read_text().splitlines()
                 if line.startswith("Model name:")]
        if len(names) != 1:
            raise ValueError(f"{path} must name exactly one CPU model")
        return names[0]

    @staticmethod
    def _operating_system(provenance: Path) -> str:
        pretty = [line.split("=", 1)[1].strip('"')
                  for line in (provenance / "os-release").read_text().splitlines()
                  if line.startswith("PRETTY_NAME=")]
        if len(pretty) != 1:
            raise ValueError("os-release must hold exactly one PRETTY_NAME")
        uname = (provenance / "uname").read_text().split()
        return f"{pretty[0]}; Linux {uname[2]} {uname[-2]}"

    def _verify_checksums(self) -> None:
        for line in (self.run_dir / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            if sha256_file(self.run_dir / name) != digest:
                raise ValueError(f"raw checksum mismatch: {self.run_dir / name}")

    def require_step(self, label: str) -> None:
        if self.step_returncodes[label] != 0:
            raise ValueError(f"{self.run_dir} step {label} failed")

    def raw(self, relative: str) -> Path:
        return (self.run_dir / _RAW_PREFIX / relative).resolve(strict=True)

    def raw_record(self, path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": sha256_file(path)}

    def identity(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "hostname": self.hostname,
            "source_commit": self.source_commit,
            "versions": self.versions,
        }


def _substitution(run: HostProfileRun, target_version: str,
                  reason: str | None) -> dict[str, object] | None:
    if run.vllm_version == target_version:
        if reason is not None:
            raise ValueError("a substitution reason needs a version mismatch")
        return None
    if reason is None:
        raise ValueError(
            f"measurement under vLLM {run.vllm_version} is bound to vLLM "
            f"{target_version} only with an explicit substitution reason")
    return {
        "target_vllm_version": target_version,
        "measured_vllm_version": run.vllm_version,
        "reason": reason,
    }


def host_launch(run: HostProfileRun, target_version: str,
                substitution_reason: str | None) -> tuple[dict, dict]:
    """Return ``(reference, profile)`` for the Host launch profile.

    The profile's ``measurement`` section binds the written reference and is
    added by the caller.
    """

    run.require_step("host_launch_r1")
    manifest_path = run.raw("host_launch_r1/manifest.json")
    manifest = _json(manifest_path)
    if manifest["status"] != "complete" or manifest["hardware_id"] != HARDWARE_ID:
        raise ValueError(f"{manifest_path} is not a complete H200 host launch run")
    contract = manifest["contract"]
    if contract["boundary"] != "host_submit_only_no_device_sync":
        raise ValueError("host launch run boundary is not submit-only")
    runner = run.repository_root / _HOST_LAUNCH_RUNNER
    expected = manifest["command"][manifest["command"].index("--expected-source-sha256") + 1]
    if sha256_file(runner) != expected:
        raise ValueError("host launch runner changed since the measurement")
    summary = manifest["summary"]
    eager_s = summary["eager:512"]["median"] / 1e9
    graph_s = summary["graph_replay:512"]["median"] / 1e9
    repeats = []
    for label in sorted(run.step_returncodes):
        if label.startswith("host_launch_r") and label != "host_launch_r1":
            if run.step_returncodes[label] != 0:
                continue
            repeat_path = run.raw(f"{label}/manifest.json")
            repeat = _json(repeat_path)["summary"]
            repeats.append({
                "label": label,
                **run.raw_record(repeat_path),
                "eager_512_median_ns": repeat["eager:512"]["median"],
                "graph_replay_512_median_ns": repeat["graph_replay:512"]["median"],
            })
    environment = manifest["environment"]
    reference = {
        "schema_version": 1,
        "hardware_id": HARDWARE_ID,
        "measurement_boundary": "host_submit_only_no_device_sync",
        "selection": {
            "eager_enqueue_s": eager_s,
            "eager_source": "host_launch_r1 eager:512 host_submit_ns_per_dispatch median",
            "graph_replay_enqueue_s": graph_s,
            "graph_source": "host_launch_r1 graph_replay:512 host_submit_ns median",
        },
        "raw": {**run.raw_record(manifest_path), "retention": "local_raw_not_committed"},
        "repeats_evidence_only": repeats,
        "run": run.identity(),
        "environment": environment,
        "device_before": manifest["device_before"],
        "sample_contract": {
            "chain_length": 512,
            "trials": contract["trials_per_cell"],
            "operation": contract["operation"],
            "production_or_evaluation_duration_consumed": False,
        },
        "substitution": _substitution(run, target_version, substitution_reason),
        "device_launch_gap": (
            "not measured on H200: the profile omits the optional "
            "device_launch_gap section, so every device launch gap is zero"),
    }
    profile = {
        "schema_version": 2,
        "hardware_id": HARDWARE_ID,
        "host_cpu": run.host_cpu,
        "operating_system": run.operating_system,
        "driver_version": manifest["device_before"]["driver_version"],
        "runtime": {"name": "CUDA", "version": environment["cuda_runtime"]},
        "framework": {"name": "PyTorch", "version": environment["torch"]},
        "service_time_s": {"eager_enqueue": eager_s, "graph_replay_enqueue": graph_s},
        "source": {
            "commit": run.source_commit,
            "path": _HOST_LAUNCH_RUNNER,
            "sha256": expected,
        },
    }
    return reference, profile


def request_path(run: HostProfileRun, target_version: str,
                 substitution_reason: str | None) -> tuple[dict, dict]:
    """Return ``(reference, service_time_s)`` for the request-path profile."""

    for label in ("request_path_primitives_r1", "request_path_admission_ipc_r1",
                  "request_path_admission_ipc_r2", "request_path_admission_ipc_r3",
                  "request_path_delivery_region"):
        run.require_step(label)
    primitives_path = run.raw("host_request_path_r1/manifest.json")
    primitives = _json(primitives_path)
    measurements = primitives["measurements"]
    window = measurements["tokenizer_microbatch_window"]
    if window["status"] != "available":
        raise ValueError("tokenizer microbatch window is unavailable")
    processor = measurements["input_processor"]
    if processor["status"] != "available":
        raise ValueError("input processor measurement is unavailable")
    admission_runs = []
    for name in _ADMISSION_RUNS:
        path = run.raw(f"{name}/manifest.json")
        regions = _json(path)["measurements"]
        admission_runs.append({
            "label": name,
            **run.raw_record(path),
            "admission_enqueue_s": regions["admission_enqueue_region"]["median_s"],
            "engine_core_ipc_s": regions["engine_core_ipc_region"]["median_s"],
        })
    delivery_path = run.raw("output_delivery_region_r1.json")
    delivery = _json(delivery_path)
    repeat = None
    if run.step_returncodes["request_path_primitives_r2"] == 0:
        repeat_path = run.raw("host_request_path_r2/manifest.json")
        repeat_measurements = _json(repeat_path)["measurements"]
        repeat = {
            **run.raw_record(repeat_path),
            "api_entry_s": repeat_measurements["api_entry_and_stream"]["api_entry"]["median_s"],
            "tokenizer_async_call_s": repeat_measurements["tokenizer_microbatch_window"]["median_s"],
        }
    services = {
        "admission_enqueue": median(row["admission_enqueue_s"] for row in admission_runs),
        "api_entry": measurements["api_entry_and_stream"]["api_entry"]["median_s"],
        "engine_core_ipc": median(row["engine_core_ipc_s"] for row in admission_runs),
        "first_token_delivery": delivery["first_token_delivery_s"],
        "input_processor_intercept": processor["intercept_s"],
        "input_processor_per_token": processor["per_token_s"],
        "output_delivery_per_output": delivery["output_delivery_per_output_s"],
        "tokenizer_async_call": window["median_s"],
        "tokenizer_encode_intercept": measurements["tokenizer_encode"]["intercept_s"],
        "tokenizer_encode_per_token": measurements["tokenizer_encode"]["per_token_s"],
        "tokenizer_render_region_intercept": None,
        "tokenizer_render_region_per_token": None,
    }
    source_identity = (
        f"{window['source_symbol']}(batch_wait_timeout_s="
        f"{window['declared_batch_wait_timeout_s']},max_batch_size="
        f"{window['declared_max_batch_size']})")
    reference = {
        "schema_version": 1,
        "hardware_id": HARDWARE_ID,
        "measurement_boundary": "host_only_no_device_no_model",
        "production_or_evaluation_duration_consumed": False,
        "host": {**primitives["host"], "hostname": run.hostname},
        "selection": {
            "service_time_s": services,
            "rules": {
                "api_entry": "primitives r1 api_entry median",
                "tokenizer_async_call": "primitives r1 tokenizer_microbatch_window median",
                "tokenizer_encode_*": "primitives r1 tokenizer_encode affine fit",
                "input_processor_*": "primitives r1 input_processor affine through measured endpoints",
                "admission_enqueue / engine_core_ipc": "median across admission/IPC region runs r1-r3",
                "first_token_delivery / output_delivery_per_output": (
                    "delivery-region probe selection (per-fanout median of repeated setups, "
                    "least-squares single-output and marginal terms)"),
                "tokenizer_render_region_*": "null: this runtime reaches the tokenizer through "
                                             "the batching wrapper, not a render region",
            },
            "source_identity": source_identity,
            "runtime_version": primitives["host"]["vllm_version"],
        },
        "raw": {
            "primitives_r1": run.raw_record(primitives_path),
            "primitives_r2_evidence_only": repeat,
            "admission_ipc_regions": admission_runs,
            "delivery_region": run.raw_record(delivery_path),
            "retention": "local_raw_not_committed",
        },
        "run": run.identity(),
        "substitution": _substitution(run, target_version, substitution_reason),
    }
    return reference, services


def runtime_prepare(run: HostProfileRun, target_version: str,
                    substitution_reason: str | None) -> tuple[dict, dict]:
    """Return ``(reference, service_time_s)`` for the runtime-prepare profile."""

    run.require_step("runtime_prepare_r1")
    run.require_step("collective_prepare_r1")
    summary_path = run.raw("host_runtime_prepare_r1/summary.json")
    summary = _json(summary_path)
    if summary["hardware_id"] != HARDWARE_ID:
        raise ValueError("runtime prepare summary hardware differs")
    services = {
        mechanism: summary["results"][mechanism]["runtime_prepare_ns_per_call"] / 1e9
        for mechanism in _RUNTIME_PREPARE_MECHANISMS
    }
    rank_records = []
    per_operation: dict[str, list[float]] = {name: [] for name in _COLLECTIVE_MECHANISMS}
    for rank in (0, 1):
        path = run.raw(f"collective_host_prepare_r1/summary-rank-{rank}.json")
        payload = _json(path)
        rank_records.append(run.raw_record(path))
        for fit in payload["fits"]:
            per_operation[fit["operation"]].append(fit["runtime_prepare_ns_per_call"])
    for operation, mechanism in _COLLECTIVE_MECHANISMS.items():
        values = per_operation[operation]
        if len(values) != 4:
            raise ValueError(f"expected 2 ranks x 2 message sizes for {operation}")
        services[mechanism] = sum(values) / len(values) / 1e9
    services = dict(sorted(services.items()))
    reference = {
        "schema_version": 1,
        "hardware_id": HARDWARE_ID,
        "measurement_boundary": "independent_runtime_mechanism_microbenchmark",
        "selection": {
            "service_time_s": services,
            "rules": {
                "runtime mechanisms": "host_runtime_prepare_r1 runtime_prepare_ns_per_call "
                                      "(chain slope minus the run's eager enqueue)",
                "collective_*_prepare": "mean over both ranks and both message sizes of "
                                        "runtime_prepare_ns_per_call",
                "mechanism set": "identical to the A800 runtime-prepare profile",
            },
            "eager_enqueue_us_subtracted": summary["contract"]["eager_enqueue_us"],
        },
        "raw": {
            "runtime_prepare": run.raw_record(summary_path),
            "collective_prepare_ranks": rank_records,
            "retention": "local_raw_not_committed",
        },
        "environment": summary["environment"],
        "run": run.identity(),
        "substitution": _substitution(run, target_version, substitution_reason),
    }
    return reference, services


# Fields the Scheduler Host fitter (``scheduler_host_profile.py``) reads from
# each Scheduler trace step, plus the step index kept for traceability.
_SCHEDULER_SAMPLE_FIELDS = (
    "step_index",
    "scheduled_request_count",
    "prefill_request_count",
    "decode_request_count",
    "schedule_start_ns",
    "schedule_end_ns",
    "scheduler_update_start_ns",
    "scheduler_update_end_ns",
)


def reduce_serving_scheduler_trace(case_dir: Path, destination: Path) -> dict[str, object]:
    """Write the Scheduler Host samples of one H200 candidate case.

    The candidate's ``scheduler_steps.jsonl.zst`` is reduced to the fields the
    Scheduler Host fitter consumes; the written ``manifest.json`` carries the
    identity/runtime fields the fitter reads and binds the source candidate by
    SHA256.  No duration other than the Scheduler's own plan/update intervals
    is kept.
    """

    import zstandard

    from dataset_pipeline.serving_dataset.compact_case import load_compact_case

    case = load_compact_case(case_dir)
    manifest = case.manifest
    if manifest["identity"]["hardware_id"] != HARDWARE_ID:
        raise ValueError(f"{case_dir} is not an H200 case")
    if case.scheduler_steps is None:
        raise ValueError(f"{case_dir} has no Scheduler trace")
    lines = [
        json.dumps({field: step[field] for field in _SCHEDULER_SAMPLE_FIELDS},
                   sort_keys=True, separators=(",", ":"))
        for step in case.scheduler_steps
    ]
    payload = ("\n".join(lines) + "\n").encode()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "scheduler_steps.jsonl.zst").write_bytes(
        zstandard.ZstdCompressor(level=19).compress(payload))
    runtime = manifest["runtime"]
    reduced_manifest = {
        "schema_version": 1,
        "kind": "scheduler_host_samples_from_serving_candidate",
        "identity": {
            "hardware_id": HARDWARE_ID,
            "model_id": manifest["identity"]["model_id"],
            "case_id": manifest["identity"]["case_id"],
        },
        "runtime": {
            "runtime_version": runtime["runtime_version"],
            "source_commit": runtime["source_commit"],
            "target_qps": runtime["target_qps"],
        },
        "fields": list(_SCHEDULER_SAMPLE_FIELDS),
        "sample_sha256": hashlib.sha256(payload).hexdigest(),
        "source": {
            "candidate_manifest_sha256": sha256_file(case_dir / "manifest.json"),
            "scheduler_steps_sha256": manifest["scheduler_steps"]["sha256"],
            "scheduler_timing_boundary": manifest["scheduler_steps"]["timing_boundary"],
            "converter": manifest["converter"],
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(reduced_manifest, indent=1, sort_keys=True) + "\n")
    return reduced_manifest
