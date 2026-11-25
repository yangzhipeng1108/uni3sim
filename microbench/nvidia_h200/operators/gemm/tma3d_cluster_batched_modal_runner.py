"""Capture eight-in-flight cluster-2 TMA 3D multicast versus duplicate loads."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import modal


SOURCE = Path(__file__).with_name("tma3d_cluster_batched.cu")
IMAGE = (modal.Image.from_registry(
    "nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .add_local_file(SOURCE, remote_path="/tmp/tma3d_cluster_batched.cu"))
APP = modal.App("midlayer-h200-tma3d-cluster-batched-20261030")


@APP.function(gpu="H200", image=IMAGE, timeout=180, retries=0,
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
    with tempfile.TemporaryDirectory(prefix="midlayer-tma3d-batched-",
                                     dir="/tmp") as temporary:
        binary = Path(temporary) / "operand_stream"
        command = ["nvcc", "-O3", "-std=c++17", "-arch=sm_90a",
                   "-Xptxas=-v", "-o", str(binary),
                   "/tmp/tma3d_cluster_batched.cu"]
        build = subprocess.run(command, capture_output=True, text=True,
                               timeout=90)
        result = {
            "gpu_identity": gpu,
            "nvcc_version": nvcc,
            "compile_command": command,
            "build_exit_code": build.returncode,
            "build_stdout": build.stdout,
            "build_stderr": build.stderr,
        }
        if build.returncode == 0:
            result["binary_sha256"] = hashlib.sha256(
                binary.read_bytes()).hexdigest()
            run = subprocess.run([str(binary)], capture_output=True,
                                 text=True, timeout=60)
            result["run_exit_code"] = run.returncode
            result["run_stdout"] = run.stdout
            result["run_stderr"] = run.stderr
        result["function_wall_s"] = time.monotonic() - started
        return result


@APP.local_entrypoint()
def main(output_dir: str) -> None:
    root = Path(__file__).resolve().parents[4]
    expected = (root / "profiler_output/microbench/nvidia_h200/"
                "tma3d_cluster_batched_20261030").resolve()
    target = Path(output_dir).resolve()
    if target != expected or target.exists():
        raise ValueError("output must be the registered new campaign")
    started = time.monotonic()
    result = capture.remote()
    result["client_wall_s"] = time.monotonic() - started
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["measurement_boundary"] = (
        "independent_3d_tma_cluster2_66_clusters_80_steps_8_in_flight_"
        "no_matrix_no_output_tile_no_production_scheduler")
    target.mkdir(parents=True)
    (target / "raw.json").write_text(json.dumps(result, indent=2,
                                                sort_keys=True) + "\n")
    print(json.dumps({"raw": str(target / "raw.json"),
                      "build_exit_code": result["build_exit_code"],
                      "function_wall_s": result["function_wall_s"]}))
