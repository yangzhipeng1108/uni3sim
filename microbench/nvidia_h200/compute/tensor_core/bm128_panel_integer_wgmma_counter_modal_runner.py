"""Capture non-duration INT32 issue for the panel/WGMMA probe modes."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import modal


HERE = Path(__file__).resolve().parent
TASK_DIR = Path("/tmp/midlayer-h200-panel-integer-wgmma-counter-20261029")
if HERE.name == "tensor_core":
    SOURCE = HERE / "bm128_panel_integer_wgmma_overlap.cu"
    GENERATOR = HERE / "wgmma_ss_wait1_issue_gen.py"
    REFERENCE = HERE.parents[1] / "operators/flash_attn/fa3_wgmma_service.cu"
else:
    SOURCE = TASK_DIR / "bm128_panel_integer_wgmma_overlap.cu"
    GENERATOR = TASK_DIR / "wgmma_ss_wait1_issue_gen.py"
    REFERENCE = TASK_DIR / "fa3_wgmma_service.cu"
IMAGE = (modal.Image.from_registry(
    "nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("cuda-nsight-compute-12-9")
    .add_local_file(SOURCE, remote_path=str(TASK_DIR / SOURCE.name))
    .add_local_file(GENERATOR, remote_path=str(TASK_DIR / GENERATOR.name))
    .add_local_file(REFERENCE, remote_path=str(TASK_DIR / REFERENCE.name)))
APP = modal.App("midlayer-h200-panel-integer-wgmma-counter-20261029")


def _run(command: list[str], timeout: int) -> dict:
    result = subprocess.run(command, cwd=TASK_DIR, capture_output=True,
                            text=True, timeout=timeout)
    return {"command": command, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def _sass_boundary_valid(sass: str) -> bool:
    for mode in (2, 3, 4):
        symbol = f"Function : _Z11panel_probeILi{mode}EEvPKhPyPfPj"
        if sass.count(symbol) != 1:
            return False
        body = sass.split(symbol, 1)[1].split("Function : ", 1)[0]
        if (body.count("HGMMA.64x256x16.F32.BF16") != 4
                or "LDGSTS.E.BYPASS.128" not in body
                or "WARPGROUP.DEPBAR.LE gsb0, 0x1" not in body
                or "WARPGROUP.DEPBAR.LE gsb0, 0x0" not in body):
            return False
        if mode in (3, 4) and body.count("IMAD") < 50:
            return False
    return True


@APP.function(gpu="H200", image=IMAGE, timeout=480, retries=0,
              max_containers=1, single_use_containers=True)
def capture(metric_set: str, backing_mib: int) -> dict:
    if backing_mib not in (8, 48):
        raise ValueError("panel backing must be 8 or 48 MiB")
    if metric_set == "integer":
        metrics = "smsp__sass_thread_inst_executed_op_integer_pred_on.sum"
    elif metric_set == "traffic":
        metrics = "lts__t_sectors_op_read.sum,dram__bytes_read.sum"
    else:
        raise ValueError("counter metric set must be integer or traffic")
    started = time.monotonic()
    result = {}
    try:
        result["gpu_identity"] = _run([
            "nvidia-smi", "--query-gpu=name,clocks.sm,driver_version",
            "--format=csv,noheader"], 20)
        result["nvcc_version"] = _run(["nvcc", "--version"], 20)
        result["ncu_version"] = _run(["ncu", "--version"], 20)
        if result["gpu_identity"]["exit_code"] or result["nvcc_version"]["exit_code"]:
            return result
        if result["ncu_version"]["exit_code"]:
            return result
        result["generation"] = _run([
            "python3", GENERATOR.name, REFERENCE.name,
            "wgmma_ss_helper.cuh"], 20)
        if result["generation"]["exit_code"]:
            return result
        result["generated_helper_sha256"] = hashlib.sha256(
            (TASK_DIR / "wgmma_ss_helper.cuh").read_bytes()).hexdigest()
        binary = TASK_DIR / "bm128_panel_integer_wgmma_overlap"
        result["build"] = _run([
            "nvcc", "-O3", "-std=c++17",
            "-gencode=arch=compute_90a,code=sm_90a",
            f"-DPANEL_BACKING_MIB={backing_mib}",
            "-Xptxas=-v", "-o", str(binary), SOURCE.name], 120)
        if result["build"]["exit_code"]:
            return result
        result["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        result["sass"] = _run(["cuobjdump", "--dump-sass", str(binary)], 60)
        if result["sass"]["exit_code"]:
            return result
        result["sass_boundary_valid"] = _sass_boundary_valid(result["sass"]["stdout"])
        if not result["sass_boundary_valid"]:
            return result
        result["counters"] = {}
        for mode, skip in (("copy_plus_wgmma", 20),
                           ("copy_plus_wgmma_integer_chain", 29),
                           ("copy_plus_wgmma_integer_four_chains", 38)):
            result["counters"][mode] = _run([
                "ncu", "--csv", "--page", "raw", "--cache-control", "none",
                "--clock-control", "none", "--kernel-name", "regex:panel_probe",
                "--launch-skip", str(skip), "--launch-count", "1",
                "--metrics", metrics,
                str(binary)], 180)
        return result
    finally:
        result["gpu_function_wall_s"] = time.monotonic() - started
        shutil.rmtree(TASK_DIR, ignore_errors=True)


@APP.local_entrypoint()
def main(output_dir: str, metric_set: str = "integer", backing_mib: int = 48):
    if metric_set not in ("integer", "traffic"):
        raise ValueError("counter metric set must be integer or traffic")
    if backing_mib not in (8, 48):
        raise ValueError("panel backing must be 8 or 48 MiB")
    root = Path(__file__).resolve().parents[4]
    target = Path(output_dir).resolve()
    campaign = ("bm128_panel_integer_l2_20261029" if backing_mib == 8
                else "bm128_panel_integer_wgmma_overlap_20261029")
    expected = (root /
        "profiler_output/microbench/nvidia_h200" / campaign /
        ("counter_attempt" if metric_set == "integer" else "traffic_attempt"))
    if target != expected.resolve():
        raise ValueError("output must be the registered overlap campaign stage")
    if target.exists() and (target / "raw.json").exists():
        raise ValueError("overlap campaign stage already has a capture")
    started = time.monotonic()
    result = capture.remote(metric_set, backing_mib)
    result["client_wall_s"] = time.monotonic() - started
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["generator_sha256"] = hashlib.sha256(GENERATOR.read_bytes()).hexdigest()
    result["reference_sha256"] = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["measurement_boundary"] = (
        "independent_panel_wgmma_integer_issue_counter_only_no_production_duration"
        if metric_set == "integer" else
        "independent_panel_wgmma_l2_dram_traffic_counter_only_no_production_duration")
    result["metric_set"] = metric_set
    result["backing_mib"] = backing_mib
    target.mkdir(parents=True, exist_ok=True)
    (target / "raw.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(target),
                      "sass_boundary_valid": result["sass_boundary_valid"]
                      if "sass_boundary_valid" in result else None,
                      "counter_exit_codes": {
                          name: row["exit_code"] for name, row in result["counters"].items()
                      } if "counters" in result else None,
                      "client_wall_s": result["client_wall_s"]}, sort_keys=True))
