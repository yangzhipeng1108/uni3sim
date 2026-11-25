"""Capture independent H200 FA3 two-warpgroup scheduler barrier service."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import modal


HERE = Path(__file__).resolve().parent
TASK_DIR = Path("/tmp/midlayer-h200-fa3-wg-scheduler-barrier-20261102")
SOURCE = (HERE / "fa3_wg_scheduler_barrier.cu" if HERE.name == "flash_attn"
          else TASK_DIR / "fa3_wg_scheduler_barrier.cu")
IMAGE = (modal.Image.from_registry(
    "nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .add_local_file(SOURCE, remote_path=str(TASK_DIR / SOURCE.name)))
APP = modal.App("midlayer-h200-fa3-wg-scheduler-barrier-20261102")


def _run(command: list[str], timeout: int) -> dict:
    result = subprocess.run(command, cwd=TASK_DIR, capture_output=True,
                            text=True, timeout=timeout)
    return {"command": command, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


@APP.function(gpu="H200", image=IMAGE, timeout=480, retries=0,
              max_containers=1, single_use_containers=True)
def capture() -> dict:
    start = time.monotonic()
    result = {}
    try:
        result["gpu_identity"] = _run([
            "nvidia-smi", "--query-gpu=name,clocks.sm,driver_version",
            "--format=csv,noheader"], 20)
        result["nvcc_version"] = _run(["nvcc", "--version"], 20)
        if result["gpu_identity"]["exit_code"] or result["nvcc_version"]["exit_code"]:
            return result
        binary = TASK_DIR / "fa3_wg_scheduler_barrier"
        result["build"] = _run([
            "nvcc", "-O3", "-std=c++17",
            "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v",
            "-o", str(binary), SOURCE.name], 120)
        if result["build"]["exit_code"]:
            return result
        result["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        result["sass"] = _run(["cuobjdump", "--dump-sass", str(binary)], 60)
        if result["sass"]["exit_code"]:
            return result
        sass = result["sass"]["stdout"]
        result["sass_boundary_valid"] = (
            re.search(r"BAR\.SYNC(?:\.DEFER_BLOCKING)? 0x1, 0x100", sass) is not None
            and re.search(r"BAR\.SYNC(?:\.DEFER_BLOCKING)? 0x2, 0x100", sass) is not None
            and "BAR.ARV 0x1, 0x100" in sass
            and "BAR.ARV 0x2, 0x100" in sass
            and "HGMMA" not in sass and "LDGSTS" not in sass)
        if result["sass_boundary_valid"]:
            result["execution"] = _run([str(binary)], 90)
        return result
    finally:
        result["gpu_function_wall_s"] = time.monotonic() - start
        shutil.rmtree(TASK_DIR, ignore_errors=True)


@APP.local_entrypoint()
def main(output_dir: str):
    if ("MIDLAYER_H200_PAID_PROBE_CONTROLLER" not in os.environ
            or os.environ["MIDLAYER_H200_PAID_PROBE_CONTROLLER"] != "1"):
        raise ValueError("paid probe requires the budget controller")
    root = Path(__file__).resolve().parents[4]
    target = Path(output_dir).resolve()
    expected_parent = (root / "profiler_output/microbench/nvidia_h200/"
                       "fa3_wg_scheduler_barrier_20261102")
    if (target.parent != expected_parent.resolve()
            or re.fullmatch(r"attempt_[A-Za-z0-9_]+", target.name) is None
            or target.exists()):
        raise ValueError("output must be the new registered FA3 scheduler probe campaign")
    started = time.monotonic()
    result = capture.remote()
    result["client_wall_s"] = time.monotonic() - started
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["source_revision"] = "506341a143fcabd4bb79052a7605ada727d6b3f5"
    result["source_semantics"] = (
        "two warpgroup bar.sync/bar.arrive alternating handshake; "
        "no production attention, tensor operand or WGMMA")
    target.mkdir(parents=True)
    (target / "raw.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_dir": str(target),
        "sass_boundary_valid": result["sass_boundary_valid"]
        if "sass_boundary_valid" in result else None,
        "execution_exit_code": result["execution"]["exit_code"]
        if "execution" in result else None,
        "client_wall_s": result["client_wall_s"],
    }, sort_keys=True))
