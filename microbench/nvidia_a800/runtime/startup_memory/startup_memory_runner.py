"""Plan by default; execute one expressly authorized A800 startup capture."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
OWNER_CONFIG = REPOSITORY_ROOT / "configs/model/qwen3-8b.json"
CAMPAIGN_RAW = (
    REPOSITORY_ROOT
    / "profiler_output/serving/nvidia_a800/"
    "startup_native_memory_20261026"
)
EXPECTED_REPORT = (
    REPOSITORY_ROOT
    / "experiment_logs/evaluation/operator_support/"
    "analytical_performance_closure/normal_acceptance/a800"
)
CAPTURE_SIZES = [1, 2, 4, 8, 16, *range(24, 513, 8)]
REQUIRED_PHASES = {
    "worker_init.end",
    "worker_model_load.begin",
    "worker_model_load.end",
    "model_load.begin",
    "model_load.end",
    "profile.begin",
    "profile.end",
    "available_KV.end",
    "runtime_peak_reset.after",
    "KV_allocate.end",
    "runner_KV_initialize.begin",
    "runner_KV_initialize.end",
    "all_capture.begin",
    "all_capture.end",
    "startup_warmup.end",
}


def command_for(model, port):
    compilation = json.dumps(
        {
            "level": 3,
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": CAPTURE_SIZES,
        },
        separators=(",", ":"),
    )
    return [
        sys.executable,
        "-B",
        "-c",
        "from vllm.entrypoints.cli.main import main; main()",
        "serve",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        "Qwen3-8B",
        "--dtype",
        "bfloat16",
        "--load-format",
        "auto",
        "--seed",
        "42",
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--block-size",
        "16",
        "--gpu-memory-utilization",
        "0.8",
        "--max-model-len",
        "8192",
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "256",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--compilation-config",
        compilation,
    ]


def records(raw):
    for path in raw.glob("memory.rank-process-*.jsonl"):
        for line in path.read_text().splitlines():
            yield json.loads(line)


def observations(raw):
    phases_by_pid = {}
    modules_by_pid = {}
    for record in records(raw):
        pid = record["pid"]
        if record["record_type"] == "native_startup_memory_bytes":
            if pid not in phases_by_pid:
                phases_by_pid[pid] = set()
            phases_by_pid[pid].add(record["phase"])
        elif record["record_type"] == "installed_source_binding":
            if pid not in modules_by_pid:
                modules_by_pid[pid] = set()
            modules_by_pid[pid].add(record["module"])
        else:
            raise ValueError("Unknown startup memory record type")
    return phases_by_pid, modules_by_pid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--startup-timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--execute-authorized-device-startup", action="store_true"
    )
    args = parser.parse_args()
    raw = args.raw_output.resolve()
    if raw.parent != CAMPAIGN_RAW or not raw.name.startswith("run"):
        raise ValueError(
            "Raw output must be a fresh run child of " + str(CAMPAIGN_RAW)
        )
    if args.report_output.resolve() != EXPECTED_REPORT:
        raise ValueError("This campaign report is frozen to " + str(EXPECTED_REPORT))
    if args.device_index != 7:
        raise ValueError("This authorized capture is frozen to physical GPU 7")
    if not 1024 <= args.port <= 65535:
        raise ValueError("Explicit unprivileged port required")
    if not 1 <= args.startup_timeout_seconds <= 900:
        raise ValueError("Startup timeout must be in [1, 900] seconds")
    command = command_for(args.model, args.port)
    plan = {
        "mode": "plan_only_no_model_or_device_import",
        "command": command,
        "required_phases": sorted(REQUIRED_PHASES),
        "raw_output": str(args.raw_output.resolve()),
        "report_output": str(args.report_output.resolve()),
        "physical_device_index": args.device_index,
        "capture_scope": {
            "hardware": "nvidia_a800",
            "vllm": "0.11.0",
            "model_family": "Qwen3",
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "speculative_or_mtp_tokens": 0,
            "capacity_mode": "normal_vllm_startup",
            "block_size_tokens": 16,
            "timing_or_pricing_evidence": False,
        },
    }
    if not args.execute_authorized_device_startup:
        print(json.dumps(plan, indent=2))
        return 0

    model_config = args.model / "config.json"
    if not model_config.is_file():
        raise FileNotFoundError(model_config)
    if not OWNER_CONFIG.is_file():
        raise FileNotFoundError(OWNER_CONFIG)
    model_config_bytes = model_config.read_bytes()
    if model_config_bytes != OWNER_CONFIG.read_bytes():
        raise RuntimeError("Qwen3-8B config identity changed")
    config_sha256 = hashlib.sha256(model_config_bytes).hexdigest()
    report = args.report_output.resolve()
    if raw.exists():
        raise FileExistsError("Fresh raw campaign directory required: " + str(raw))
    raw.mkdir(parents=True)
    report.mkdir(parents=True, exist_ok=True)
    identity = {
        "model": str(args.model),
        "model_config_path": str(model_config),
        "owner_config_path": str(OWNER_CONFIG),
        "config_sha256": config_sha256,
        "device_index": args.device_index,
        "command": command,
        "capture_scope": plan["capture_scope"],
    }
    identity_path = raw / "capture_identity.json"
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    bindings = json.loads((HERE / "bindings.json").read_text())
    expected_modules = set(bindings["modules"])
    runtime_root = Path(tempfile.mkdtemp(prefix="a8-", dir="/tmp"))
    runtime_names = (
        "tmp",
        "vllm_cache",
        "xdg_cache",
        "inductor_cache",
        "triton_cache",
        "torch_cache",
    )
    runtime_paths = [runtime_root / name for name in runtime_names]
    for path in runtime_paths:
        path.mkdir()
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES=str(args.device_index),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        MIDLAYER_STARTUP_MEMORY_RAW=str(raw),
        MIDLAYER_STARTUP_MEMORY_BINDINGS=str(HERE / "bindings.json"),
        MIDLAYER_STARTUP_MEMORY_IDENTITY=str(identity_path),
        TMPDIR=str(runtime_root / "tmp"),
        VLLM_CACHE_ROOT=str(runtime_root / "vllm_cache"),
        XDG_CACHE_HOME=str(runtime_root / "xdg_cache"),
        TORCHINDUCTOR_CACHE_DIR=str(runtime_root / "inductor_cache"),
        TRITON_CACHE_DIR=str(runtime_root / "triton_cache"),
        TORCH_HOME=str(runtime_root / "torch_cache"),
    )
    bootstrap = str(HERE / "bootstrap")
    if "PYTHONPATH" in environment:
        environment["PYTHONPATH"] = bootstrap + os.pathsep + environment["PYTHONPATH"]
    else:
        environment["PYTHONPATH"] = bootstrap
    process = None
    complete_pid = None
    phases_by_pid = {}
    modules_by_pid = {}
    started = time.monotonic()
    identity["runtime_root"] = str(runtime_root)
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    with (raw / "startup.stdout.log").open("w") as stdout, (
        raw / "startup.stderr.log"
    ).open("w") as stderr:
        try:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            while time.monotonic() - started < args.startup_timeout_seconds:
                phases_by_pid, modules_by_pid = observations(raw)
                for pid, phases in phases_by_pid.items():
                    if (
                        REQUIRED_PHASES <= phases
                        and expected_modules <= modules_by_pid[pid]
                    ):
                        complete_pid = pid
                        break
                if complete_pid is not None or process.poll() is not None:
                    break
                time.sleep(0.25)
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            shutil.rmtree(runtime_root)
    seen = set().union(*phases_by_pid.values()) if phases_by_pid else set()
    receipt = {
        "collected_required_phases": complete_pid is not None,
        "complete_observer_pid": complete_pid,
        "seen_phases_by_pid": {
            str(pid): sorted(values) for pid, values in phases_by_pid.items()
        },
        "missing_phases": sorted(REQUIRED_PHASES - seen),
        "seen_source_modules_by_pid": {
            str(pid): sorted(values) for pid, values in modules_by_pid.items()
        },
        "runtime_root": str(runtime_root),
        "runtime_root_removed": not runtime_root.exists(),
        "normal_capacity_accepted": False,
        "timing_or_pricing_evidence": False,
    }
    (raw / "collection_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    report_path = report / ("startup_capture_execution." + raw.name + ".json")
    if report_path.exists():
        raise FileExistsError("Fresh report output required: " + str(report_path))
    report_path.write_text(
        json.dumps({**plan, "receipt": receipt}, indent=2) + "\n"
    )
    return 0 if complete_pid is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
