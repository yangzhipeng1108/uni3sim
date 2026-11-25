"""Distinguish within-WG scheduling from cross-WG primitive interference."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "tensor_scalar_warpgroup_partition.cu"
REFERENCE = HERE / ".." / ".." / "operators/flash_attn/fa3_wgmma_service.cu"
TASK = Path("/tmp/midlayer-tensor-scalar-partition-20261107")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
).add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name)).add_local_file(
    REFERENCE, remote_path=str(TASK / REFERENCE.name))
APP = modal.App("midlayer-tensor-scalar-partition-20261107")


def helper(source):
    first = "__device__ __forceinline__ void wgmma_ss_n64_kmajor"
    start = "__device__ __forceinline__ void wgmma_ss_n128_kmajor"
    end = "__device__ __forceinline__ void wgmma_ss_n256_mnmajor"
    if any(source.count(x) != 1 for x in (first, start, end)):
        raise ValueError("Independent WGMMA helper boundary changed")
    return source[:source.index(first)] + source[source.index(start):source.index(end)]


def code_gate(sass):
    for matrices in (4, 8, 16):
        for values in (16, 64, 128):
            for exp in (0, 1):
                marker = f"Function : _Z14partition_pairILi{matrices}ELi{values}ELb{exp}EE"
                if sass.count(marker) != 1:
                    return False
                body = sass.split(marker, 1)[1].split("Function : ", 1)[0]
                if body.count("HGMMA.") != 2 * matrices or body.count("MUFU.EX2") != (2 * values if exp else 0):
                    return False
                windows = body.split("WARPGROUP.DEPBAR")
                if exp and not any("HGMMA." in window and "MUFU.EX2" in window
                                   and window.index("HGMMA.") < window.rindex("MUFU.EX2")
                                   for window in windows):
                    return False
    return True


@APP.function(gpu="H200", image=IMAGE, timeout=240, retries=0,
              single_use_containers=True)
def capture():
    result = {}
    started = time.monotonic()
    try:
        reference = (TASK / REFERENCE.name).read_text()
        header = helper(reference)
        (TASK / "partition_wgmma_helper.cuh").write_text(header)
        result["helper_source"] = header
        result["reference_sha256"] = hashlib.sha256(reference.encode()).hexdigest()
        commands = {
            "identity": ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm", "--format=csv,noheader"],
            "compiler": ["nvcc", "--version"],
            "build": ["nvcc", "-O3", "-arch=sm_90a", "-Xptxas=-v", str(TASK / SOURCE.name), "-o", str(TASK / "probe")],
            "sass": ["cuobjdump", "--dump-sass", str(TASK / "probe")],
            "measurement": [str(TASK / "probe")],
        }
        for name, command in commands.items():
            p = subprocess.run(command, capture_output=True, text=True, timeout=90)
            result[name] = {"command": command, "exit_code": p.returncode,
                            "stdout": p.stdout, "stderr": p.stderr}
            if p.returncode:
                break
            if name == "build":
                entries = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", p.stderr)
                result["spill_free"] = len(entries) == 18 and all(a == b == "0" for a, b in entries)
                if not result["spill_free"]:
                    break
            if name == "sass":
                result["code_gate"] = code_gate(p.stdout)
                if not result["code_gate"]:
                    break
        return result
    finally:
        result["gpu_function_wall_s"] = time.monotonic() - started
        if TASK.exists():
            shutil.rmtree(TASK)


@APP.local_entrypoint()
def main(output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "raw.json").exists():
        raise ValueError("capture already exists")
    result = capture.remote()
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    if not all(name in result and result[name]["exit_code"] == 0 for name in (
            "identity", "compiler", "build", "sass", "measurement")):
        raise RuntimeError("WG partition capture failed; inspect saved raw.json")
    print({"wall_s": result["gpu_function_wall_s"], "spill_free": result["spill_free"]})
