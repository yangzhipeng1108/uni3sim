"""Capture CUDA 13.0 H200 nvjet non-duration program evidence."""

import hashlib
import json
from pathlib import Path
import subprocess
import time
import zlib

import modal


CASE = Path(__file__).with_name("nvjet_static_case.py")
IMAGE = (modal.Image.from_registry("vllm/vllm-openai:v0.30.0", add_python="3.12")
         .entrypoint([]).apt_install("cuda-nsight-compute-13-0")
         .add_local_file(CASE, remote_path="/tmp/nvjet_static_case.py"))
APP = modal.App("midlayer-h200-nvjet-static-20261030")


@APP.function(gpu="H200", image=IMAGE, timeout=240, retries=0,
              max_containers=1, single_use_containers=True)
def capture(stage: str) -> dict:
    started = time.monotonic()
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    if stage == "sass":
        report_args = ["--page", "source", "--print-source", "sass",
                       "--metrics", "smsp__inst_executed.sum"]
    elif stage == "launch":
        report_args = ["--csv", "--page", "raw", "--section", "LaunchStats"]
    elif stage == "traffic":
        report_args = [
            "--csv", "--page", "raw", "--metrics",
            "lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,"
            "dram__bytes_read.sum,dram__bytes_write.sum",
        ]
    elif stage == "atomics":
        report_args = [
            "--csv", "--page", "raw", "--metrics",
            "smsp__sass_thread_inst_executed_op_global_atom_pred_on.sum",
        ]
    elif stage == "atomic_requests":
        report_args = [
            "--csv", "--page", "raw", "--metrics",
            "smsp__inst_executed_op_generic_atom_pred_on_any.sum,"
            "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
        ]
    else:
        raise ValueError("stage must be sass, launch, traffic, atomics or atomic_requests")
    command = ["ncu", *report_args,
               "--clock-control", "none", "--cache-control", "none",
               "--kernel-name", "regex:nvjet", "--launch-count", "1",
               "/usr/bin/python3", "/tmp/nvjet_static_case.py"]
    completed = subprocess.run(command, capture_output=True, timeout=180)
    stdout, stderr = completed.stdout, completed.stderr
    if len(stdout) > 40_000_000 or len(stderr) > 4_000_000:
        raise RuntimeError("nvjet report exceeded the declared return size")
    return {
        "stage": stage,
        "gpu_identity": gpu,
        "command": command,
        "exit_code": completed.returncode,
        "function_wall_s": time.monotonic() - started,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stdout_zlib_hex": zlib.compress(stdout, level=9).hex(),
        "stderr_zlib_hex": zlib.compress(stderr, level=9).hex(),
    }


@APP.local_entrypoint()
def main(output_dir: str, stage: str = "sass") -> None:
    root = Path(__file__).resolve().parents[3]
    target = Path(output_dir).resolve()
    expected = (root / "profiler_output/kernel_dataset/nvidia_h200/"
                "nvjet_static_candidate_20261030").resolve()
    if target != expected or stage not in (
            "sass", "launch", "traffic", "atomics", "atomic_requests"):
        raise ValueError("output must be the registered nvjet candidate campaign")
    raw_name = {"sass": "raw.json", "launch": "launch_raw.json",
                "traffic": "traffic_raw.json", "atomics": "atomics_raw.json",
                "atomic_requests": "atomic_requests_raw.json"}[stage]
    if (target / raw_name).exists() or (stage == "sass" and target.exists()):
        raise ValueError("stage output already exists")
    started = time.monotonic()
    result = capture.remote(stage)
    result["client_wall_s"] = time.monotonic() - started
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["case_sha256"] = hashlib.sha256(CASE.read_bytes()).hexdigest()
    result["measurement_boundary"] = (
        "synthetic_bf16_gemm_dynamic_atomic_counter_no_duration_consumption"
        if stage in ("atomics", "atomic_requests") else
        "production_shape_synthetic_inputs_static_sass_no_duration_consumption")
    target.mkdir(parents=True, exist_ok=True)
    (target / raw_name).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"raw": str(target / raw_name),
                      "exit_code": result["exit_code"],
                      "function_wall_s": result["function_wall_s"]}, sort_keys=True))
