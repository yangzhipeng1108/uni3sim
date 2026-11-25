"""Independent Tensor/scalar concurrent versus serial resource primitives."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "tensor_scalar_overlap.cu"
REFERENCE = HERE / ".." / ".." / "operators/flash_attn/fa3_wgmma_service.cu"
TASK = Path("/tmp/midlayer-tensor-scalar-overlap-20261107")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
).add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name)).add_local_file(
    REFERENCE, remote_path=str(TASK / REFERENCE.name))
APP = modal.App("midlayer-tensor-scalar-overlap-20261107")


def helper(source):
    first = "__device__ __forceinline__ void wgmma_ss_n64_kmajor"
    start = "__device__ __forceinline__ void wgmma_ss_n128_kmajor"
    end = "__device__ __forceinline__ void wgmma_ss_n256_mnmajor"
    if any(source.count(x) != 1 for x in (first, start, end)):
        raise ValueError("Independent WGMMA helper boundary changed")
    return source[:source.index(first)] + source[source.index(start):source.index(end)]


def overlaps(sass):
    for rs in (0, 1):
        marker = f"Function : _Z13resource_pairILi2ELb{rs}ELb1EE"
        if sass.count(marker) != 1:
            return False
        body = sass.split(marker, 1)[1].split("Function : ", 1)[0]
        instructions = [line for line in body.splitlines() if "/*" in line]
        first = next((i for i, line in enumerate(instructions) if "HGMMA." in line), None)
        if first is None:
            return False
        stops = [i for i, line in enumerate(instructions) if i > first and "WARPGROUP.DEPBAR" in line]
        if not stops:
            return False
        window = instructions[first:stops[0]]
        if sum("HGMMA." in line for line in window) < 8 or not any("MUFU.EX2" in line for line in window):
            return False
        serial_marker = f"Function : _Z13resource_pairILi3ELb{rs}ELb1EE"
        if sass.count(serial_marker) != 1:
            return False
        serial = sass.split(serial_marker, 1)[1].split("Function : ", 1)[0]
        first_mma = serial.index("HGMMA.")
        first_wait = serial.index("WARPGROUP.DEPBAR", first_mma)
        if "MUFU.EX2" in serial[first_mma:first_wait] or "MUFU.EX2" not in serial[first_wait:]:
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
        (TASK / "overlap_wgmma_helper.cuh").write_text(header)
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
                counts = re.findall(r"(\d+) bytes (?:spill stores|spill loads|stack frame)", p.stderr)
                result["spill_free"] = bool(counts) and all(int(x) == 0 for x in counts)
                if not result["spill_free"]:
                    break
            if name == "sass":
                result["sass_overlap_gate"] = overlaps(p.stdout)
                if not result["sass_overlap_gate"]:
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
        raise RuntimeError("Resource-overlap capture failed; inspect saved raw.json")
    print({"wall_s": result["gpu_function_wall_s"], "overlap_gate": result["sass_overlap_gate"]})
