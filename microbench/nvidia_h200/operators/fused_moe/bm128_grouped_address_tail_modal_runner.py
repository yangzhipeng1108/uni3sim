"""Measure source-ordered A/B staging followed by an independent output tail."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import modal


HERE = Path(__file__).resolve().parent
TASK_DIR = Path("/tmp/midlayer-h200-grouped-address-tail-20261029")
SOURCE = (HERE / "bm128_grouped_address_stream.cu" if HERE.name == "fused_moe"
          else TASK_DIR / "bm128_grouped_address_stream.cu")
IMAGE = (modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("cuda-nsight-compute-12-6")
    .add_local_file(SOURCE, remote_path=str(TASK_DIR / SOURCE.name)))
APP = modal.App("midlayer-h200-grouped-address-tail-20261029")


def _run(command: list[str], timeout: int) -> dict:
    result = subprocess.run(command, cwd=TASK_DIR, capture_output=True,
                            text=True, timeout=timeout)
    return {"command": command, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


@APP.function(gpu="H200", image=IMAGE, timeout=420, retries=0,
              max_containers=1, single_use_containers=True)
def capture() -> dict:
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
        binary = TASK_DIR / "bm128_grouped_address_tail"
        result["build"] = _run([
            "nvcc", "-O3", "-std=c++17", "-arch=sm_90",
            "-DWITH_OUTPUT_TAIL=1", "-Xptxas=-v", "-o", str(binary),
            SOURCE.name], 120)
        if result["build"]["exit_code"]:
            return result
        result["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        result["sass"] = _run(["cuobjdump", "--dump-sass", str(binary)], 40)
        if result["sass"]["exit_code"]:
            return result
        sass = result["sass"]["stdout"]
        result["sass_boundary_valid"] = (
            sass.count("LDGSTS.E.BYPASS.128") > 0
            and sass.count("F2FP.BF16.F32.PACK_AB") == 64
            and sass.count("STG.E.128") == 16
            and sass.count("STS.128") == 16
            and sass.count("LDSM") == 16
            and "HGMMA." not in sass and "HMMA." not in sass)
        if not result["sass_boundary_valid"]:
            return result
        result["events"] = []
        result["counters"] = []
        for mode in ("control", "tail"):
            record = _run([str(binary), "w1", "both", mode], 90)
            result["events"].append({"mode": mode, **record})
            if (record["exit_code"] or record["stdout"].count("sample=") != 7
                    or "validated checksum=" not in record["stderr"]
                    or "occupancy=1" not in record["stderr"]):
                result["event_boundary_valid"] = False
                return result
        result["event_boundary_valid"] = True
        for mode in ("control", "tail"):
            record = _run([
                "ncu", "--csv", "--page", "raw", "--cache-control", "none",
                "--clock-control", "none", "--kernel-name",
                "regex:grouped_address_stream", "--launch-skip", "2",
                "--launch-count", "1", "--metrics",
                "lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,"
                "dram__bytes_read.sum,dram__bytes_write.sum",
                str(binary), "w1", "both", mode], 90)
            result["counters"].append({"mode": mode, **record})
            if record["exit_code"]:
                return result
        return result
    finally:
        result["gpu_function_wall_s"] = time.monotonic() - started
        shutil.rmtree(TASK_DIR, ignore_errors=True)


@APP.local_entrypoint()
def main(output_dir: str):
    root = Path(__file__).resolve().parents[4]
    parent = root / "profiler_output/microbench/nvidia_h200"
    target = Path(output_dir).resolve()
    if (target.parent != parent.resolve() or target.exists()
            or target.name != "bm128_grouped_address_tail_20261029"):
        raise ValueError("output must be a new registered H200 grouped-tail campaign")
    started = time.monotonic()
    result = capture.remote()
    result["client_wall_s"] = time.monotonic() - started
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["baseline_commit"] = "ef73f6a476b8184d1cf5faad3c5c5a86dc9d8049"
    result["measurement_boundary"] = (
        "independent_grouped_address_cp_async_then_ptx_derived_output_tail_no_mma")
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
