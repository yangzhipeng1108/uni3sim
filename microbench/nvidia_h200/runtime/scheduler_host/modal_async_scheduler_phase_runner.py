"""Measure labeled vLLM 0.30 AsyncScheduler service on a Modal H200 Host.

The H200 allocation selects the previously authorized Modal Host environment.
The benchmark uses synthetic request lengths and no model runner or weights.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

import modal


SOURCE = Path(__file__).resolve()
_REMOTE_MODULE = len(SOURCE.parents) < 5
ROOT = Path("/opt/midlayer") if _REMOTE_MODULE else SOURCE.parents[4]
if _REMOTE_MODULE:
    IMAGE = modal.Image.debian_slim()
else:
    IMAGE = (
        modal.Image.from_registry("vllm/vllm-openai:v0.30.0", add_python="3.12")
        .entrypoint([])
        .add_local_dir(ROOT / "dataset_pipeline",
                       remote_path="/opt/midlayer/dataset_pipeline")
        .add_local_file(ROOT / "scripts/measure_external_vllm_scheduler_service.py",
                        remote_path="/opt/midlayer/scripts/measure_external_vllm_scheduler_service.py")
        .add_local_file(ROOT / "configs/model/qwen3-4b.json",
                        remote_path="/opt/midlayer/configs/model/qwen3-4b.json")
    )
APP = modal.App("midlayer-h200-async-scheduler-phase-20261102")
MEASUREMENT_PYTHON = "/usr/bin/python3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], timeout: int) -> dict:
    started = time.monotonic()
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout)
    except subprocess.TimeoutExpired as error:
        return {"command": command, "returncode": None,
                "error": f"timeout after {timeout}s: {error}",
                "wall_seconds": time.monotonic() - started}
    return {"command": command, "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "wall_seconds": time.monotonic() - started}


@APP.function(gpu="H200", image=IMAGE, timeout=600, retries=0,
              max_containers=1, single_use_containers=True)
def capture() -> dict:
    result = {"status": "started", "hardware_id": "nvidia_h200",
              "measurement_boundary":
              "native_vllm030_async_scheduler_schedule_and_update_only"}
    root = Path("/opt/midlayer")
    runner = root / "scripts/measure_external_vllm_scheduler_service.py"
    owner = root / "dataset_pipeline/profiler/external_vllm_scheduler_service.py"
    model = root / "configs/model/qwen3-4b.json"
    result["input_sha256"] = {
        "thin_runner": _sha256(runner), "measurement_owner": _sha256(owner),
        "model_geometry": _sha256(model),
    }
    result["gpu"] = _run([
        "nvidia-smi", "--query-gpu=name,driver_version",
        "--format=csv,noheader"], 20)
    result["host_identity"] = _run([
        MEASUREMENT_PYTHON, "-c",
        "import json,os,pathlib,platform,vllm; "
        "print(json.dumps({'vllm':vllm.__version__,"
        "'cpuinfo':pathlib.Path('/proc/cpuinfo').read_text().split('\\n\\n',1)[0],"
        "'affinity':sorted(os.sched_getaffinity(0)),"
        "'platform':platform.platform()}))"], 30)
    if (result["gpu"]["returncode"] != 0
            or not result["gpu"]["stdout"].startswith("NVIDIA H200,")
            or result["host_identity"]["returncode"] != 0
            or '"vllm": "0.30.0' not in result["host_identity"]["stdout"]):
        result["status"] = "identity_failed"
        return result
    with tempfile.TemporaryDirectory(prefix="midlayer-h200-phase-",
                                     dir="/tmp") as temporary:
        directory = Path(temporary)
        result["runs"] = []
        cells = ((2, 32, 8, False),
                 (1, 32, 64, False), (1, 256, 64, False),
                 (8, 32, 64, False), (8, 256, 64, False),
                 (32, 32, 64, False), (32, 256, 64, False),
                 (64, 32, 64, False), (64, 256, 64, False),
                 (83, 32, 64, False), (83, 256, 64, False),
                 (83, 256, 64, True))
        for max_requests, prompt_tokens, sample_count, mixed in cells:
            suffix = "_mixed" if mixed else ""
            output = directory / f"scheduler_{max_requests}_{prompt_tokens}{suffix}.json"
            command = [
                MEASUREMENT_PYTHON, str(runner),
                "--hardware-id", "nvidia_h200", "--model-config", str(model),
                "--output", str(output), "--sample-count", str(sample_count),
                "--max-batched-tokens", "32768", "--max-requests", str(max_requests),
                "--num-kv-blocks", "8192", "--max-model-len", "4096",
                "--prompt-tokens", str(prompt_tokens), "--output-tokens", "8",
                "--async-scheduling", "--platform", "cpu",
            ]
            if mixed:
                command.extend(["--admission-batch-sizes", "41",
                                "--admit-after-first-step", "42"])
            run = _run(command, 40)
            run["fanout"] = max_requests
            run["prompt_tokens"] = prompt_tokens
            run["mixed_admission"] = mixed
            if run["returncode"] == 0 and output.is_file():
                raw = json.loads(output.read_text())
                rows = raw["steps"]
                phases = {"prefill": 0, "mixed": 0, "decode": 0}
                for row in rows:
                    phases[row["phase"]] += 1
                run["phase_counts"] = phases
                run["fanout_match"] = (
                    any(row["phase"] == "mixed"
                        and row["scheduled_request_count"] == 83
                        and row["prefill_request_count"] == 42
                        and row["decode_request_count"] == 41
                        for row in rows) if mixed else all(
                            row["scheduled_request_count"] == max_requests
                            for row in rows))
                run["raw"] = raw
            result["runs"].append(run)
            if (run["returncode"] != 0 or "raw" not in run
                    or not run["fanout_match"]
                    or (max_requests != 2 and not mixed
                        and (run["phase_counts"]["prefill"] < 4
                             or run["phase_counts"]["decode"] < 20))
                    or (mixed and run["phase_counts"]["mixed"] < 1)):
                result["status"] = "smoke_failed" if max_requests == 2 else "cell_failed"
                return result
        result["status"] = "complete"
        return result


@APP.local_entrypoint()
def main(output_dir: str) -> None:
    if ("MIDLAYER_H200_PAID_PROBE_CONTROLLER" not in os.environ
            or os.environ["MIDLAYER_H200_PAID_PROBE_CONTROLLER"] != "1"):
        raise ValueError("paid probe requires the budget controller")
    target = Path(output_dir).resolve()
    expected_parent = (ROOT / "profiler_output/microbench/nvidia_h200/"
                       "modal_async_scheduler_phase_20261102").resolve()
    if (target.parent != expected_parent
            or re.fullmatch(r"attempt_[A-Za-z0-9_]+", target.name) is None):
        raise ValueError("output must be a new registered scheduler probe attempt")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite phase raw: {target}")
    result = capture.remote()
    target.mkdir(parents=True)
    result["wrapper_sha256"] = _sha256(SOURCE)
    raw = target / "raw.json"
    raw.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "raw_path": str(raw),
                      "run_count": len(result["runs"]) if "runs" in result else 0}))
