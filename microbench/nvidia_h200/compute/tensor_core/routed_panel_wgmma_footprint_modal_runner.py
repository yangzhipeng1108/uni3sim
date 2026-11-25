"""Capture short-K H200 panel overlap with a backing larger than L2."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import modal


HERE = Path(__file__).resolve().parent
TASK_DIR = Path("/tmp/midlayer-h200-routed-panel-wgmma-footprint-20261102")
if HERE.name == "tensor_core":
    SOURCE = HERE / "routed_panel_wgmma_footprint.cu"
    GENERATOR = HERE / "wgmma_ss_wait1_issue_gen.py"
    REFERENCE = HERE.parents[1] / "operators/flash_attn/fa3_wgmma_service.cu"
else:
    SOURCE = TASK_DIR / "routed_panel_wgmma_footprint.cu"
    GENERATOR = TASK_DIR / "wgmma_ss_wait1_issue_gen.py"
    REFERENCE = TASK_DIR / "fa3_wgmma_service.cu"
IMAGE = (modal.Image.from_registry(
    "nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("cuda-nsight-compute-12-9")
    .add_local_file(SOURCE, remote_path=str(TASK_DIR / SOURCE.name))
    .add_local_file(GENERATOR, remote_path=str(TASK_DIR / GENERATOR.name))
    .add_local_file(REFERENCE, remote_path=str(TASK_DIR / REFERENCE.name)))
APP = modal.App("midlayer-h200-routed-panel-wgmma-footprint-20261102")


def _run(command: list[str], timeout: int) -> dict:
    started = time.monotonic()
    try:
        result = subprocess.run(command, cwd=TASK_DIR, capture_output=True,
                                text=True, timeout=timeout, check=False)
        return {"command": command, "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr,
                "wall_s": time.monotonic() - started}
    except subprocess.TimeoutExpired as error:
        return {"command": command, "exit_code": None,
                "error": f"timeout after {timeout}s: {error}",
                "wall_s": time.monotonic() - started}


def _sass_boundary(sass: str) -> dict:
    counts_by_round = {}
    for rounds in (4, 8):
        symbol = ("Function : _Z18routed_panel_probeILi2ELi"
                  f"{rounds}EEvPKhS1_PKiPyPfPj")
        if sass.count(symbol) != 1:
            return {"valid": False, "reason":
                    f"combined {rounds}-round function missing or duplicated"}
        body = sass.split(symbol, 1)[1].split("Function : ", 1)[0]
        counts = {"hgmma": body.count("HGMMA.64x256x16.F32.BF16"),
                  "ldgsts": body.count("LDGSTS.E.BYPASS.128"),
                  "wait1": body.count("WARPGROUP.DEPBAR.LE gsb0, 0x1"),
                  "wait0": body.count("WARPGROUP.DEPBAR.LE gsb0, 0x0")}
        counts_by_round[str(rounds)] = counts
    return {"valid": all(counts["hgmma"] == 4 and counts["ldgsts"] > 0
                          and counts["wait1"] > 0 and counts["wait0"] > 0
                          for counts in counts_by_round.values()),
            "counts_by_round": counts_by_round}


@APP.function(gpu="H200", image=IMAGE, timeout=480, retries=0,
              max_containers=1, single_use_containers=True)
def capture() -> dict:
    started = time.monotonic()
    result = {"measurement_boundary":
              "short_k_large_backing_gather_cp_async_wgmma_no_moe_output"}
    try:
        result["gpu_identity"] = _run([
            "nvidia-smi", "--query-gpu=name,clocks.sm,driver_version",
            "--format=csv,noheader"], 20)
        result["nvcc_version"] = _run(["nvcc", "--version"], 20)
        result["ncu_version"] = _run(["ncu", "--version"], 20)
        if any(result[name]["exit_code"] != 0 for name in
               ("gpu_identity", "nvcc_version", "ncu_version")):
            result["status"] = "environment_failed"
            return result
        result["generation"] = _run([
            "python3", GENERATOR.name, REFERENCE.name,
            "wgmma_ss_helper.cuh"], 20)
        if result["generation"]["exit_code"] != 0:
            result["status"] = "generation_failed"
            return result
        helper = TASK_DIR / "wgmma_ss_helper.cuh"
        result["helper_sha256"] = hashlib.sha256(helper.read_bytes()).hexdigest()
        binary = TASK_DIR / "routed_panel_wgmma_footprint"
        result["build"] = _run([
            "nvcc", "-O3", "-std=c++17",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-Xptxas=-v", "-o", str(binary), SOURCE.name], 120)
        if result["build"]["exit_code"] != 0:
            result["status"] = "build_failed"
            return result
        result["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        result["sass"] = _run(["cuobjdump", "--dump-sass", str(binary)], 60)
        if result["sass"]["exit_code"] != 0:
            result["status"] = "sass_failed"
            return result
        result["sass_boundary"] = _sass_boundary(result["sass"]["stdout"])
        if not result["sass_boundary"]["valid"]:
            result["status"] = "sass_boundary_failed"
            return result
        result["execution"] = _run([str(binary)], 90)
        if result["execution"]["exit_code"] != 0:
            result["status"] = "execution_failed"
            return result
        result["counters"] = {}
        for mode, label in ((0, "gather_copy"), (2, "gather_plus_wgmma")):
            row = _run([
                "ncu", "--csv", "--page", "raw", "--cache-control", "none",
                "--clock-control", "none", "--metrics",
                "lts__t_sectors_op_read.sum,dram__bytes_read.sum",
                str(binary), "--single", str(mode), "8"], 120)
            result["counters"][label] = row
            if row["exit_code"] != 0:
                result["status"] = "counter_failed"
                return result
        result["status"] = "complete"
        return result
    finally:
        result["gpu_function_wall_s"] = time.monotonic() - started
        shutil.rmtree(TASK_DIR, ignore_errors=True)


@APP.local_entrypoint()
def main(output_dir: str) -> None:
    if os.environ["MIDLAYER_H200_PAID_PROBE_CONTROLLER"] != "1":
        raise ValueError("W2 footprint probe requires the H200 spend controller")
    root = Path(__file__).resolve().parents[4]
    campaign = root / "profiler_output/microbench/nvidia_h200/" \
        "routed_panel_wgmma_footprint_20261102"
    target = Path(output_dir).resolve()
    if target.parent != campaign.resolve() or not target.name.startswith("attempt_"):
        raise ValueError("probe output must be a registered attempt")
    if target.exists():
        raise FileExistsError(target)
    started = time.monotonic()
    result = capture.remote()
    result["client_wall_s"] = time.monotonic() - started
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["generator_sha256"] = hashlib.sha256(GENERATOR.read_bytes()).hexdigest()
    result["reference_sha256"] = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target.mkdir(parents=True)
    (target / "raw.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(target / "raw.json"),
                      "status": result["status"],
                      "gpu_function_wall_s": result["gpu_function_wall_s"]}))


if __name__ == "__main__":
    main()
