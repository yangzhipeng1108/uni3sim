"""Capture one independent H200 SS WGMMA wait-group mechanism comparison."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import modal


HERE = Path(__file__).resolve().parent
TASK_DIR = Path("/tmp/midlayer-h200-wgmma-wait1-20261029")
if HERE.name == "tensor_core":
    SOURCE = HERE / "wgmma_ss_wait1_issue.cu"
    GENERATOR = HERE / "wgmma_ss_wait1_issue_gen.py"
    REFERENCE = HERE.parents[1] / "operators/flash_attn/fa3_wgmma_service.cu"
else:
    SOURCE = TASK_DIR / "wgmma_ss_wait1_issue.cu"
    GENERATOR = TASK_DIR / "wgmma_ss_wait1_issue_gen.py"
    REFERENCE = TASK_DIR / "fa3_wgmma_service.cu"
IMAGE = (modal.Image.from_registry(
    "nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .add_local_file(SOURCE, remote_path=str(TASK_DIR / SOURCE.name))
    .add_local_file(GENERATOR, remote_path=str(TASK_DIR / GENERATOR.name))
    .add_local_file(REFERENCE, remote_path=str(TASK_DIR / REFERENCE.name)))
APP = modal.App("midlayer-h200-wgmma-wait1-20261029")


def _run(command: list[str], timeout: int) -> dict:
    result = subprocess.run(command, cwd=TASK_DIR, capture_output=True,
                            text=True, timeout=timeout)
    return {"command": command, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def _wait1_sass_has_four_uninterrupted_wgmma(sass: str) -> bool:
    symbol = "Function : _Z10wait_probeILi1EEviiPyPf"
    if sass.count(symbol) != 1:
        return False
    body = sass.split(symbol, 1)[1].split("Function : ", 1)[0]
    consecutive = 0
    for line in body.splitlines():
        if "HGMMA.64x256x16.F32.BF16" in line:
            consecutive += 1
            if consecutive >= 4:
                return True
        elif "WARPGROUP.DEPBAR.LE" in line and "0x0" in line:
            consecutive = 0
    return False


@APP.function(gpu="H200", image=IMAGE, timeout=480, retries=0,
              max_containers=1, single_use_containers=True)
def capture() -> dict:
    started = time.monotonic()
    result = {}
    try:
        result.update({"gpu_identity": _run([
            "nvidia-smi", "--query-gpu=name,clocks.sm,driver_version",
            "--format=csv,noheader"], 20),
            "nvcc_version": _run(["nvcc", "--version"], 20)})
        result["generation"] = _run(["python3", GENERATOR.name,
                                      REFERENCE.name, "wgmma_ss_helper.cuh"], 20)
        if result["generation"]["exit_code"]:
            return result
        result["generated_helper_sha256"] = hashlib.sha256(
            (TASK_DIR / "wgmma_ss_helper.cuh").read_bytes()).hexdigest()
        binary = TASK_DIR / "wgmma_ss_wait1_issue"
        result["build"] = _run(["nvcc", "-O3", "-std=c++17",
                                "-gencode=arch=compute_90a,code=sm_90a",
                                "-Xptxas=-v", "--keep", "--keep-dir", str(TASK_DIR),
                                "-o", str(binary), SOURCE.name], 120)
        if result["build"]["exit_code"]:
            result["ptx_targets"] = [
                line for path in TASK_DIR.glob("*.ptx")
                for line in path.read_text().splitlines()
                if line.startswith(".target")]
            return result
        result["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        result["sass"] = _run(["cuobjdump", "--dump-sass", str(binary)], 60)
        if result["sass"]["exit_code"]:
            return result
        result["wait1_four_uninterrupted_wgmma"] = (
            _wait1_sass_has_four_uninterrupted_wgmma(result["sass"]["stdout"]))
        if result["wait1_four_uninterrupted_wgmma"]:
            result["execution"] = _run([str(binary)], 90)
        return result
    finally:
        result["gpu_function_wall_s"] = time.monotonic() - started
        shutil.rmtree(TASK_DIR, ignore_errors=True)


@APP.local_entrypoint()
def main(output_dir: str):
    root = Path(__file__).resolve().parents[4]
    target = Path(output_dir).resolve()
    parent = root / "profiler_output/microbench/nvidia_h200/wgmma_wait1_issue_20261029"
    if target != parent.resolve() or (target.exists() and (target / "raw.json").exists()):
        raise ValueError("output must be the registered WGMMA wait1 campaign without a capture")
    start = time.monotonic()
    result = capture.remote()
    result["client_wall_s"] = time.monotonic() - start
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["generator_sha256"] = hashlib.sha256(GENERATOR.read_bytes()).hexdigest()
    result["reference_sha256"] = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["measurement_boundary"] = (
        "independent_onchip_ss_m64n256k16_wait0_wait1_no_global_operand_or_route")
    target.mkdir(parents=True, exist_ok=True)
    (target / "raw.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(target),
                      "wait1_four_uninterrupted_wgmma": (
                          result["wait1_four_uninterrupted_wgmma"]
                          if "wait1_four_uninterrupted_wgmma" in result else None),
                      "client_wall_s": result["client_wall_s"]}, sort_keys=True))
