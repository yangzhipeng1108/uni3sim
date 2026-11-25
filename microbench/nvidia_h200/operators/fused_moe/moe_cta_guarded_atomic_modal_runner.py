"""Capture independent H200 one-CTA guarded shared-atomic comparison."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import modal


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "moe_cta_guarded_atomic.cu"
TASK_DIR = Path("/tmp/midlayer-h200-moe-cta-guarded-20261029")
IMAGE = (modal.Image.from_registry(
    "nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .add_local_file(SOURCE, remote_path=str(TASK_DIR / SOURCE.name)))
APP = modal.App("midlayer-h200-moe-cta-guarded-20261029")


def _run(command: list[str], timeout: int) -> dict:
    completed = subprocess.run(command, cwd=TASK_DIR, capture_output=True,
                               text=True, timeout=timeout)
    return {"command": command, "exit_code": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr}


@APP.function(gpu="H200", image=IMAGE, timeout=180, retries=0,
              max_containers=1, single_use_containers=True)
def capture() -> dict:
    started = time.monotonic()
    result = {}
    try:
        result["gpu_identity"] = _run([
            "nvidia-smi", "--query-gpu=name,clocks.sm,driver_version",
            "--format=csv,noheader"], 20)
        result["nvcc_version"] = _run(["nvcc", "--version"], 20)
        if result["gpu_identity"]["exit_code"] or result["nvcc_version"]["exit_code"]:
            return result
        binary = TASK_DIR / "moe_cta_guarded_atomic"
        result["build"] = _run([
            "nvcc", "-O3", "-std=c++17", "-arch=sm_90", "-Xptxas=-v",
            "-o", str(binary), SOURCE.name], 100)
        if result["build"]["exit_code"]:
            return result
        result["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        result["sass"] = _run(["cuobjdump", "--dump-sass", str(binary)], 45)
        if result["sass"]["exit_code"]:
            return result
        result["execution"] = _run([str(binary)], 40)
        return result
    finally:
        result["gpu_function_wall_s"] = time.monotonic() - started
        shutil.rmtree(TASK_DIR, ignore_errors=True)


@APP.local_entrypoint()
def main(output_dir: str):
    root = Path(__file__).resolve().parents[4]
    target = Path(output_dir).resolve()
    expected = (root / "profiler_output/microbench/nvidia_h200"
                / "moe_cta_guarded_atomic_20261029")
    if target != expected.resolve() or target.exists():
        raise ValueError("output must be the new registered campaign directory")
    started = time.monotonic()
    result = capture.remote()
    result["client_wall_s"] = time.monotonic() - started
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["measurement_boundary"] = (
        "one_CTA_synthetic_uniform_ID_load_guard_shared_atomic;"
        "no_prefix_scan_sort_expert_output_or_complete_MoE_operator")
    target.mkdir(parents=True)
    raw = target / "raw.json"
    raw.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (target / "manifest.json").write_text(json.dumps({
        "campaign": target.name,
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "source_sha256": result["source_sha256"],
        "runner_sha256": result["runner_sha256"],
        "binary_sha256": result["binary_sha256"] if "binary_sha256" in result else None,
        "measurement_boundary": result["measurement_boundary"],
        "gpu_count": 1, "gpu_model": "H200", "retries": 0,
        "function_timeout_s": 180, "client_wall_s": result["client_wall_s"],
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(target),
                      "execution_exit_code": result["execution"]["exit_code"]
                      if "execution" in result else None,
                      "client_wall_s": result["client_wall_s"]}, sort_keys=True))
