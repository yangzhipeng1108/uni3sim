"""Capture one independent H200 four-stage cp.async L2 panel service."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import modal


HERE = Path(__file__).resolve().parent
TASK_DIR = Path("/tmp/midlayer-h200-bm128-cp-async-ring-20261029")
SOURCE = (HERE / "bm128_cp_async_ring.cu" if HERE.name == "l2"
          else TASK_DIR / "bm128_cp_async_ring.cu")
IMAGE = (modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("cuda-nsight-compute-12-6")
    .add_local_file(SOURCE, remote_path=str(TASK_DIR / SOURCE.name)))
APP = modal.App("midlayer-h200-bm128-cp-async-ring-20261029")


def _run(command: list[str], timeout: int) -> dict:
    result = subprocess.run(command, cwd=TASK_DIR, capture_output=True,
                            text=True, timeout=timeout)
    return {"command": command, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


@APP.function(gpu="H200", image=IMAGE, timeout=360, retries=0,
              max_containers=1, single_use_containers=True)
def capture() -> dict:
    started = time.monotonic()
    result = {}
    try:
        result["gpu_identity"] = _run([
            "nvidia-smi", "--query-gpu=name,driver_version",
            "--format=csv,noheader"], 20)
        result["nvcc_version"] = _run(["nvcc", "--version"], 20)
        result["ncu_version"] = _run(["ncu", "--version"], 20)
        if any(result[key]["exit_code"] for key in (
                "gpu_identity", "nvcc_version", "ncu_version")):
            return result
        binary = TASK_DIR / "bm128_cp_async_ring"
        result["build"] = _run([
            "nvcc", "-O3", "-std=c++17", "-arch=sm_90",
            "-Xptxas=-v", "-o", str(binary), SOURCE.name], 120)
        if result["build"]["exit_code"]:
            return result
        result["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        result["sass"] = _run(["cuobjdump", "--dump-sass", str(binary)], 40)
        if result["sass"]["exit_code"]:
            return result
        sass = result["sass"]["stdout"]
        result["sass_boundary_valid"] = (
            "LDGSTS.E.BYPASS.128" in sass and "HGMMA." not in sass
            and "HMMA." not in sass)
        if not result["sass_boundary_valid"]:
            return result
        result["execution"] = _run([str(binary)], 60)
        if result["execution"]["exit_code"]:
            return result
        result["counter"] = _run([
            "ncu", "--csv", "--page", "raw", "--cache-control", "none",
            "--clock-control", "none", "--kernel-name", "regex:cp_async_ring",
            "--launch-skip", "2", "--launch-count", "1", "--metrics",
            "lts__t_sectors_op_read.sum,dram__bytes_read.sum",
            str(binary)], 180)
        return result
    finally:
        result["gpu_function_wall_s"] = time.monotonic() - started
        shutil.rmtree(TASK_DIR, ignore_errors=True)


@APP.local_entrypoint()
def main(output_dir: str):
    root = Path(__file__).resolve().parents[4]
    target = Path(output_dir).resolve()
    parent = root / "profiler_output/microbench/nvidia_h200/bm128_cp_async_ring_20261029"
    if target != parent.resolve() or (target.exists() and (target / "raw.json").exists()):
        raise ValueError("output must be the registered cp.async ring campaign without a capture")
    start = time.monotonic()
    result = capture.remote()
    result["client_wall_s"] = time.monotonic() - start
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["measurement_boundary"] = (
        "independent_four_stage_48kib_cp_async_l2_ring_no_gemm_or_route")
    target.mkdir(parents=True, exist_ok=True)
    (target / "raw.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(target),
                      "sass_boundary_valid": result["sass_boundary_valid"]
                      if "sass_boundary_valid" in result else None,
                      "counter_exit_code": result["counter"]["exit_code"]
                      if "counter" in result else None,
                      "client_wall_s": result["client_wall_s"]}, sort_keys=True))
