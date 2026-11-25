"""Measure independent BM128 BF16 conversion and scattered output tail."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import modal


HERE = Path(__file__).resolve().parent
TASK_DIR = Path("/tmp/midlayer-h200-bm128-output-tail-20261029")
SOURCE = (HERE / "bm128_output_tail.cu" if HERE.name == "fused_moe"
          else TASK_DIR / "bm128_output_tail.cu")
IMAGE = (modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("cuda-nsight-compute-12-6")
    .add_local_file(SOURCE, remote_path=str(TASK_DIR / SOURCE.name)))
APP = modal.App("midlayer-h200-bm128-output-tail-20261029")


def _run(command: list[str], timeout: int) -> dict:
    result = subprocess.run(command, cwd=TASK_DIR, capture_output=True,
                            text=True, timeout=timeout)
    return {"command": command, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


@APP.function(gpu="H200", image=IMAGE, timeout=420, retries=0,
              max_containers=1, single_use_containers=True)
def capture(layout: str) -> dict:
    started = time.monotonic()
    result = {}
    try:
        result["gpu_identity"] = _run([
            "nvidia-smi", "--query-gpu=name,driver_version,clocks.sm",
            "--format=csv,noheader"], 20)
        result["nvcc_version"] = _run(["nvcc", "--version"], 20)
        result["ncu_version"] = _run(["ncu", "--version"], 20)
        if any(result[key]["exit_code"] for key in (
                "gpu_identity", "nvcc_version", "ncu_version")):
            return result
        binary = TASK_DIR / "bm128_output_tail"
        build_command = [
            "nvcc", "-O3", "-std=c++17", "-arch=sm_90", "-Xptxas=-v",
            "-o", str(binary), SOURCE.name]
        if layout == "source_columns":
            build_command.insert(5, "-DSOURCE_COLUMN_LAYOUT=1")
        elif layout == "source_shared":
            build_command.insert(5, "-DSOURCE_SHARED_LAYOUT=1")
        elif layout != "thread_row":
            raise ValueError("unknown output-tail layout")
        result["build"] = _run(build_command, 120)
        if result["build"]["exit_code"]:
            return result
        result["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        result["sass"] = _run(["cuobjdump", "--dump-sass", str(binary)], 40)
        if result["sass"]["exit_code"]:
            return result
        sass = result["sass"]["stdout"]
        result["sass_boundary_valid"] = (
            sass.count("F2FP.BF16.F32.PACK_AB") == 64
            and sass.count("STG.E.128") >= 16
            and (layout != "source_shared" or (
                sass.count("STS.128") == 16
                and sass.count("LDSM") == 16
                and sass.count("BAR.SYNC") == 3))
            and "HGMMA." not in sass and "HMMA." not in sass)
        if not result["sass_boundary_valid"]:
            return result
        result["events"] = []
        result["counters"] = []
        for variant in ("w1", "w2"):
            for mode in ("control", "store"):
                record = _run([str(binary), variant, mode], 90)
                result["events"].append({"variant": variant, "mode": mode,
                                          **record})
                if (record["exit_code"] or record["stdout"].count("sample=") != 7
                        or "occupancy=1" not in record["stderr"]):
                    result["event_boundary_valid"] = False
                    return result
        result["event_boundary_valid"] = True
        for variant in ("w1", "w2"):
            record = _run([
                "ncu", "--csv", "--page", "raw", "--cache-control", "none",
                "--clock-control", "none", "--kernel-name", "regex:output_tail",
                "--launch-skip", "2", "--launch-count", "1", "--metrics",
                "lts__t_sectors_op_write.sum,dram__bytes_write.sum",
                str(binary), variant, "store"], 90)
            result["counters"].append({"variant": variant, **record})
            if record["exit_code"]:
                return result
        return result
    finally:
        result["gpu_function_wall_s"] = time.monotonic() - started
        shutil.rmtree(TASK_DIR, ignore_errors=True)


@APP.local_entrypoint()
def main(output_dir: str, layout: str = "thread_row"):
    root = Path(__file__).resolve().parents[4]
    parent = root / "profiler_output/microbench/nvidia_h200"
    target = Path(output_dir).resolve()
    expected_name = ("bm128_output_tail_source_shared_20261029"
                     if layout == "source_shared" else
                     "bm128_output_tail_source_columns_20261029"
                     if layout == "source_columns" else
                     "bm128_output_tail_20261029"
                     if layout == "thread_row" else None)
    if (expected_name is None or target.parent != parent.resolve()
            or target.exists() or target.name != expected_name):
        raise ValueError("output must be a new registered H200 output-tail campaign")
    started = time.monotonic()
    result = capture.remote(layout)
    result["client_wall_s"] = time.monotonic() - started
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["layout"] = layout
    result["baseline_commit"] = "ef73f6a476b8184d1cf5faad3c5c5a86dc9d8049"
    result["measurement_boundary"] = (
        "independent_bm128_bf16_conversion_scatter_no_mma_or_moe_operand_loop")
    result["production_or_evaluation_duration_consumed"] = False
    target.mkdir(parents=True)
    raw = target / "raw.json"
    raw.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({
        "output_dir": str(target),
        "gpu_function_wall_s": result["gpu_function_wall_s"],
        "client_wall_s": result["client_wall_s"],
        "sass_boundary_valid": result["sass_boundary_valid"]
        if "sass_boundary_valid" in result else None,
        "event_boundary_valid": result["event_boundary_valid"]
        if "event_boundary_valid" in result else None,
        "counter_exit_codes": [row["exit_code"] for row in result["counters"]]
        if "counters" in result else [],
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True))
