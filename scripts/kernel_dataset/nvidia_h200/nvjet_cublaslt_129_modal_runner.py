"""Capture a CUDA 12.9 cuBLASLt nvjet candidate without timing calibration."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import zlib

import modal


SOURCE = Path(__file__).with_name("nvjet_cublaslt_129_case.cu")
IMAGE = (modal.Image.from_registry(
    "nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("cuda-nsight-compute-12-9")
    .add_local_file(SOURCE, remote_path="/tmp/nvjet_cublaslt_129_case.cu"))
APP = modal.App("midlayer-h200-nvjet-cublaslt-129-20261030")


def _record(command: list[str], timeout: int) -> dict:
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, timeout=timeout)
    if len(completed.stdout) > 40_000_000 or len(completed.stderr) > 4_000_000:
        raise RuntimeError("nvjet report exceeded declared return size")
    return {
        "command": command,
        "exit_code": completed.returncode,
        "wall_s": time.monotonic() - started,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "stdout_zlib_hex": zlib.compress(completed.stdout, level=9).hex(),
        "stderr_zlib_hex": zlib.compress(completed.stderr, level=9).hex(),
    }


@APP.function(gpu="H200", image=IMAGE, timeout=240, retries=0,
              max_containers=1, single_use_containers=True)
def capture() -> dict:
    started = time.monotonic()
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version",
         "--format=csv,noheader"], capture_output=True, text=True,
        timeout=20, check=True).stdout.strip()
    if not gpu.startswith("NVIDIA H200,"):
        raise ValueError("candidate requires one H200")
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True,
                          text=True, timeout=20, check=True).stdout
    with tempfile.TemporaryDirectory(prefix="midlayer-nvjet-129-",
                                     dir="/tmp") as temporary:
        binary = Path(temporary) / "nvjet_129_case"
        build_command = [
            "nvcc", "-O2", "-std=c++17", "-arch=sm_90", "-o", str(binary),
            "/tmp/nvjet_cublaslt_129_case.cu", "-lcublasLt", "-lcublas"]
        build = _record(build_command, 90)
        result = {"gpu_identity": gpu, "nvcc_version": nvcc,
                  "build": build, "stages": []}
        if build["exit_code"] != 0:
            result["status"] = "build_failed"
            result["function_wall_s"] = time.monotonic() - started
            return result
        result["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        common = ["--clock-control", "none", "--cache-control", "none",
                  "--kernel-name", "regex:nvjet", "--launch-count", "1"]
        stages = [
            ("traffic_and_atomics", [
                "--csv", "--page", "raw", "--metrics",
                "lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,"
                "dram__bytes_read.sum,dram__bytes_write.sum,"
                "smsp__inst_executed_op_generic_atom_pred_on_any.sum,"
                "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum"]),
            ("sass", ["--page", "source", "--print-source", "sass",
                      "--metrics", "smsp__inst_executed.sum"]),
        ]
        for name, options in stages:
            row = _record(["ncu", *options, *common, str(binary)], 100)
            row["stage"] = name
            result["stages"].append(row)
            if row["exit_code"] != 0:
                result["status"] = "capture_failed"
                result["function_wall_s"] = time.monotonic() - started
                return result
        result["status"] = "complete"
        result["function_wall_s"] = time.monotonic() - started
        return result


@APP.local_entrypoint()
def main(output_dir: str) -> None:
    root = Path(__file__).resolve().parents[3]
    expected = (root / "profiler_output/kernel_dataset/nvidia_h200/"
                "nvjet_129_candidate_20261030").resolve()
    target = Path(output_dir).resolve()
    if target != expected or target.exists():
        raise ValueError("output must be the new registered CUDA12.9 candidate")
    started = time.monotonic()
    result = capture.remote()
    result["client_wall_s"] = time.monotonic() - started
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["measurement_boundary"] = (
        "synthetic_bf16_cublaslt_rank0_cuda129_non_duration_program_counters")
    target.mkdir(parents=True)
    (target / "raw.json").write_text(json.dumps(result, indent=2,
                                                sort_keys=True) + "\n")
    print(json.dumps({"raw": str(target / "raw.json"),
                      "status": result["status"],
                      "function_wall_s": result["function_wall_s"]}))
