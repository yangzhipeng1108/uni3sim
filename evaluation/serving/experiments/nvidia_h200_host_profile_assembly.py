#!/usr/bin/env python3
"""Assemble the four H200 Host timing profiles for one vLLM version.

One command re-assembles every H200 Host profile of ``--vllm-version`` from the
Host profile runs of ``microbench/nvidia_h200/runtime/host_profiles/
run_nvidia_h200_host_profiles.sh`` and the H200 ShareGPT candidate root of the same
version (Scheduler Host samples):

* ``configs/host_timing/host_launch_profiles/nvidia_h200_<tag>.json``
* ``configs/host_timing/host_request_path_profiles/nvidia_h200_<tag>.json``
* ``configs/host_timing/host_runtime_prepare_profiles/nvidia_h200_<tag>_eager.json``
* ``configs/host_timing/scheduler_host_profiles/nvidia_h200_<tag>.json``

with ``<tag>`` = ``vllm011`` / ``vllm030``, their committed references under
``microbench/nvidia_h200/runtime/`` and an assembly record under
``experiment_logs/h200_serving_eval/profiles/``.  ``--host-launch-run`` and
``--host-mechanism-run`` may be the same directory.  A run measured under a
different vLLM version than ``--vllm-version`` is accepted only with
``--substitution-reason``; the reason is written into every affected
reference.  When a clean run of the target version exists, pass it as both
runs and drop the reason.

The Scheduler Host profile is fitted by ``scheduler_host_profile.py`` (QPS 16
held out) on the candidate root's own Scheduler traces plus, when the mechanism
run was measured under the target version, its native prefill sweep
(``rep1`` trains, ``rep2`` is held out at QPS 16).
"""

from __future__ import annotations

from functools import partial
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]

from src.serving.host_timing import (  # noqa: E402
    load_host_launch_profile,
    load_host_request_path_profile,
    load_host_runtime_prepare_profile,
    load_scheduler_host_profile,
)

_HOST_TIMING = "configs/host_timing"
_RUNTIME_REFERENCES = "microbench/nvidia_h200/runtime"
_CANDIDATE_CASES = (
    "qwen3-4b/tp1_qps4", "qwen3-4b/tp1_qps16", "qwen3-4b/tp1_qps64",
    "qwen3-8b/tp1_qps4", "qwen3-8b/tp1_qps16", "qwen3-8b/tp1_qps64",
    "llama-3.1-8b/tp1_qps4", "llama-3.1-8b/tp1_qps16", "llama-3.1-8b/tp1_qps64",
    "qwen3-32b/tp2_ep1_qps1", "qwen3-32b/tp2_ep1_qps4", "qwen3-32b/tp2_ep1_qps16",
    "qwen3-30b-a3b/tp2_ep2_qps1", "qwen3-30b-a3b/tp2_ep2_qps4",
    "qwen3-30b-a3b/tp2_ep2_qps16",
)
_PREFILL_SWEEPS = ("prefill_sweep_rep1", "prefill_sweep_rep2")
_HELD_OUT_QPS = 16


def _write_json(relative: str, payload: object, *, profile_pipeline) -> dict[str, str]:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {"path": relative, "sha256": profile_pipeline.sha256_file(path)}


def main(*, profile_pipeline) -> int:
    h200 = profile_pipeline
    write_json = partial(_write_json, profile_pipeline=h200)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vllm-version", choices=tuple(h200.VLLM_VERSION_TAGS),
                        required=True)
    parser.add_argument("--host-launch-run", type=Path, required=True)
    parser.add_argument("--host-mechanism-run", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True,
                        help="H200 candidate dataset root of --vllm-version")
    parser.add_argument("--substitution-reason")
    args = parser.parse_args()
    version = args.vllm_version
    tag = h200.VLLM_VERSION_TAGS[version]
    launch_run = h200.HostProfileRun(args.host_launch_run, ROOT)
    mechanism_run = h200.HostProfileRun(args.host_mechanism_run, ROOT)
    launch_reason = (None if launch_run.vllm_version == version
                     else args.substitution_reason)
    mechanism_reason = (None if mechanism_run.vllm_version == version
                        else args.substitution_reason)
    if args.substitution_reason is not None and launch_reason is None \
            and mechanism_reason is None:
        raise ValueError("--substitution-reason given but both runs match the version")
    outputs = {}

    # Host launch.
    reference, profile = h200.host_launch(launch_run, version, launch_reason)
    reference_record = write_json(
        f"{_RUNTIME_REFERENCES}/launch/references/host_launch_{tag}.json", reference)
    profile["measurement"] = {
        "artifact": reference_record["path"],
        "sha256": reference_record["sha256"],
        "boundary": "host_submit_only_no_device_sync",
    }
    outputs["host_launch"] = write_json(
        f"{_HOST_TIMING}/host_launch_profiles/nvidia_h200_{tag}.json", profile)

    # Request path.
    reference, services = h200.request_path(mechanism_run, version, mechanism_reason)
    reference_record = write_json(
        f"{_RUNTIME_REFERENCES}/host_request_path/references/host_request_path_{tag}.json",
        reference)
    outputs["host_request_path"] = write_json(
        f"{_HOST_TIMING}/host_request_path_profiles/nvidia_h200_{tag}.json", {
            "schema_version": 1,
            "hardware_id": h200.HARDWARE_ID,
            "host": {"cpu": mechanism_run.host_cpu},
            "runtime": {
                "version": reference["selection"]["runtime_version"],
                "source_identity": reference["selection"]["source_identity"],
            },
            "service_time_s": services,
            "measurement": {
                "artifacts": [reference_record],
                "boundary": "host_only_no_device_no_model",
                "statistic": "median_of_independent_primitive_samples",
                "time_unit": "seconds",
            },
        })

    # Runtime prepare.
    reference, services = h200.runtime_prepare(mechanism_run, version, mechanism_reason)
    reference_record = write_json(
        f"{_RUNTIME_REFERENCES}/launch/references/host_runtime_prepare_{tag}.json",
        reference)
    outputs["host_runtime_prepare"] = write_json(
        f"{_HOST_TIMING}/host_runtime_prepare_profiles/nvidia_h200_{tag}_eager.json", {
            "schema_version": 1,
            "hardware_id": h200.HARDWARE_ID,
            "runtime": {"name": "CUDA", "version": mechanism_run.versions["torch_cuda"]},
            "framework": {"name": "PyTorch", "version": mechanism_run.versions["torch"]},
            "service_time_s": services,
            "measurement": {
                "boundary": "independent_runtime_mechanism_microbenchmark",
                "artifacts": [reference_record],
            },
        })

    # Scheduler Host samples and fit.
    sample_root_relative = f"{_RUNTIME_REFERENCES}/scheduler_host/references/{tag}"
    sample_root = ROOT / sample_root_relative
    if sample_root.exists():
        shutil.rmtree(sample_root)
    candidate_root = args.candidate_root.resolve(strict=True)
    for case in _CANDIDATE_CASES:
        h200.reduce_serving_scheduler_trace(
            candidate_root / h200.HARDWARE_ID / case,
            sample_root / h200.HARDWARE_ID / case)
    prefill_sweeps = []
    if mechanism_run.vllm_version == version:
        for label, sweep in zip(("scheduler_prefill_rep1", "scheduler_prefill_rep2"),
                                _PREFILL_SWEEPS):
            mechanism_run.require_step(label)
            source = (mechanism_run.run_dir / "scheduler_host_prefill/dataset_root"
                      / h200.HARDWARE_ID / "scheduler-microbench" / sweep)
            destination = sample_root / h200.HARDWARE_ID / "scheduler-microbench" / sweep
            destination.mkdir(parents=True)
            for name in ("manifest.json", "scheduler_steps.jsonl.zst"):
                shutil.copy2(source / name, destination / name)
            prefill_sweeps.append(sweep)
    report_path = ROOT / f"experiment_logs/h200_serving_eval/profiles/scheduler_fit_{tag}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    scheduler_relative = f"{_HOST_TIMING}/scheduler_host_profiles/nvidia_h200_{tag}.json"
    fit_command = [
        sys.executable, "-m", "dataset_pipeline.serving_dataset.scheduler_host_profile",
        "--dataset-root", str(sample_root), "--hardware-id", h200.HARDWARE_ID,
        "--held-out-qps", str(_HELD_OUT_QPS),
        "--profile-output", str(ROOT / scheduler_relative),
        "--report-output", str(report_path),
    ]
    prefill_substitute = None
    if not prefill_sweeps:
        # The Serving traces admit one request per pure-prefill step, so the
        # prefill line needs the native sweep; without a same-version sweep the
        # measured version's published line is bound explicitly.
        prefill_substitute = (
            f"{_HOST_TIMING}/scheduler_host_profiles/nvidia_h200_"
            f"{h200.VLLM_VERSION_TAGS[mechanism_run.vllm_version]}.json")
        (ROOT / prefill_substitute).resolve(strict=True)
        fit_command += ["--prefill-line-substitute", str(ROOT / prefill_substitute),
                        "--prefill-line-substitution-reason", mechanism_reason]
    subprocess.run(fit_command, cwd=ROOT, check=True)
    outputs["scheduler_host"] = {
        "path": scheduler_relative,
        "sha256": h200.sha256_file(ROOT / scheduler_relative),
    }

    # Every profile must load through its published contract.
    launch = load_host_launch_profile(ROOT / outputs["host_launch"]["path"])
    request = load_host_request_path_profile(ROOT / outputs["host_request_path"]["path"])
    prepare = load_host_runtime_prepare_profile(
        ROOT / outputs["host_runtime_prepare"]["path"])
    scheduler = load_scheduler_host_profile(ROOT / outputs["scheduler_host"]["path"])
    if scheduler.vllm_version != version:
        raise ValueError("Scheduler Host profile version differs from --vllm-version")
    for loaded in (launch, request, prepare, scheduler):
        if loaded.hardware_id != h200.HARDWARE_ID:
            raise ValueError("assembled profile hardware differs")
    record = {
        "schema_version": 1,
        "hardware_id": h200.HARDWARE_ID,
        "vllm_version": version,
        "tag": tag,
        "command": sys.argv,
        "host_launch_run": launch_run.identity(),
        "host_mechanism_run": mechanism_run.identity(),
        "substitution": {
            "host_launch": launch_reason,
            "host_request_path": mechanism_reason,
            "host_runtime_prepare": mechanism_reason,
            "scheduler_host": (
                None if prefill_substitute is None
                else {"prefill_line_from": prefill_substitute, "reason": mechanism_reason}),
        },
        "scheduler_host_training": {
            "candidate_root": str(candidate_root),
            "prefill_sweeps": prefill_sweeps,
            "held_out_qps": _HELD_OUT_QPS,
            "fit_report": str(report_path),
        },
        "outputs": outputs,
    }
    record_path = report_path.with_name(f"assembly_{tag}.json")
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(record_path)
    return 0
