"""Capture isolated H200 single-counter persistent queue service."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import modal


SOURCE = Path(__file__).with_name("nvjet_atomic_queue.cu")
IMAGE = (modal.Image.from_registry(
    "nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .add_local_file(SOURCE, remote_path="/tmp/nvjet_atomic_queue.cu"))
APP = modal.App("midlayer-h200-nvjet-atomic-queue-20261030")


@APP.function(gpu="H200", image=IMAGE, timeout=240, retries=0,
              max_containers=1, single_use_containers=True)
def capture() -> dict:
    started = time.monotonic()
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version",
         "--format=csv,noheader"], capture_output=True, text=True,
        timeout=20, check=True).stdout.strip()
    if not gpu.startswith("NVIDIA H200,"):
        raise ValueError("probe requires one NVIDIA H200")
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True,
                          text=True, timeout=20, check=True).stdout
    with tempfile.TemporaryDirectory(prefix="midlayer-nvjet-atomic-queue-",
                                     dir="/tmp") as temporary:
        binary = Path(temporary) / "atomic_queue"
        compile_command = [
            "nvcc", "-O3", "-std=c++17", "-arch=sm_90", "-Xptxas=-v",
            "-o", str(binary), "/tmp/nvjet_atomic_queue.cu"]
        build = subprocess.run(compile_command, capture_output=True,
                               text=True, timeout=90)
        result = {
            "gpu_identity": gpu, "nvcc_version": nvcc,
            "compile_command": compile_command,
            "build_exit_code": build.returncode,
            "build_stdout": build.stdout,
            "build_stderr": build.stderr,
        }
        if build.returncode == 0:
            sass = subprocess.run(["cuobjdump", "--dump-sass", str(binary)],
                                  capture_output=True, text=True, timeout=45)
            result["sass_exit_code"] = sass.returncode
            result["sass_stdout"] = sass.stdout
            result["sass_stderr"] = sass.stderr
            result["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
            runs = []
            for _ in range(2):
                run = subprocess.run([str(binary)], capture_output=True,
                                     text=True, timeout=60)
                runs.append({"exit_code": run.returncode,
                             "stdout": run.stdout, "stderr": run.stderr})
                if run.returncode != 0:
                    break
            result["runs"] = runs
        result["function_wall_s"] = time.monotonic() - started
        return result


@APP.local_entrypoint()
def main(output_dir: str) -> None:
    root = Path(__file__).resolve().parents[4]
    expected = (root / "profiler_output/microbench/nvidia_h200/"
                "nvjet_atomic_queue_20261030").resolve()
    target = Path(output_dir).resolve()
    if target != expected or target.exists():
        raise ValueError("output must be the new registered atomic queue campaign")
    started = time.monotonic()
    result = capture.remote()
    result["client_wall_s"] = time.monotonic() - started
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["measurement_boundary"] = (
        "independent_66_cluster_leaders_single_global_counter_no_gemm_or_tma")
    target.mkdir(parents=True)
    (target / "raw.json").write_text(json.dumps(result, indent=2,
                                                sort_keys=True) + "\n")
    print(json.dumps({"raw": str(target / "raw.json"),
                      "build_exit_code": result["build_exit_code"],
                      "function_wall_s": result["function_wall_s"]}))
