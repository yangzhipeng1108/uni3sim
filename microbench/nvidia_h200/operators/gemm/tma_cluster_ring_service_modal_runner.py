"""Capture an independent 3D TMA remote-release ring on one H200."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import modal

IMAGE = modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
if modal.is_local():
    SOURCE = Path(__file__).with_name("tma_cluster_ring_service.cu")
    IMAGE = IMAGE.add_local_file(SOURCE, remote_path="/tmp/tma_cluster_ring_service.cu")
APP = modal.App("midlayer-h200-tma-cluster-ring-20261107")

@APP.function(gpu="H200", image=IMAGE, timeout=300, retries=0, single_use_containers=True)
def capture(boundary_only: bool):
    started = time.monotonic()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=20).stdout
    if not gpu.startswith("NVIDIA H200,"):
        raise ValueError("one H200 required")
    with tempfile.TemporaryDirectory(prefix="midlayer-cluster-ring-") as directory:
        command = ["nvcc", "-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v",
            "/tmp/tma_cluster_ring_service.cu", "-o", directory + "/probe"]
        environment = dict(os.environ)
        removed = []
        for name in ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"):
            if name in environment:
                del environment[name]
                removed.append(name)
        build = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=90)
        result = {"gpu": gpu, "build_command": command, "compiler_flag_environment_removed": removed,
            "build_exit_code": build.returncode, "build_stdout": build.stdout, "build_stderr": build.stderr}
        if build.returncode == 0:
            result["binary_sha256"] = hashlib.sha256(Path(directory + "/probe").read_bytes()).hexdigest()
            result["clock_before"] = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,pstate", "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=20).stdout
            arguments = [directory + "/probe"] + (["--boundary-only"] if boundary_only else [])
            run = subprocess.run(arguments, capture_output=True, text=True, timeout=180)
            result.update({"run_exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr})
            result["clock_after"] = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,pstate", "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=20).stdout
        result["function_wall_s"] = time.monotonic() - started
    return result

@APP.local_entrypoint()
def main(retry_fixed_module_paths: bool = False, prearmed_release: bool = False, boundary_only: bool = False):
    root = Path(__file__).resolve().parents[4]
    campaign = "tma_cluster_ring_service_retry_20261107" if retry_fixed_module_paths else "tma_cluster_ring_service_20261107"
    if prearmed_release:
        if retry_fixed_module_paths:
            raise ValueError("choose one campaign identity")
        campaign = "tma_cluster_ring_prearmed_20261107"
    if boundary_only:
        if retry_fixed_module_paths or prearmed_release:
            raise ValueError("choose one campaign identity")
        campaign = "tma_ring_boundary_20261107"
    target = root / "profiler_output/microbench/nvidia_h200" / campaign
    if target.exists():
        raise ValueError("refuse to overwrite captured evidence")
    result = capture.remote(boundary_only)
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["production_duration_consumed"] = False
    target.mkdir(parents=True)
    (target / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    (target / SOURCE.name).write_bytes(SOURCE.read_bytes())
    (target / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"build_exit_code": result["build_exit_code"], "function_wall_s": result["function_wall_s"]}))
