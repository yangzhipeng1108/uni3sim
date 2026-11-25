"""Capture independent WGMMA enqueue/retire windows with SASS timer gates."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "wgmma_enqueue_window.cu"
REFERENCE = HERE / ".." / ".." / "operators/flash_attn/fa3_wgmma_service.cu"
TASK = Path("/tmp/midlayer-wgmma-enqueue-window-20261107")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
).add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name)).add_local_file(
    REFERENCE, remote_path=str(TASK / REFERENCE.name))
APP = modal.App("midlayer-wgmma-enqueue-window-20261107")


def helper(source):
    first = "__device__ __forceinline__ void wgmma_ss_n64_kmajor"
    start = "__device__ __forceinline__ void wgmma_ss_n128_kmajor"
    end = "__device__ __forceinline__ void wgmma_ss_n256_mnmajor"
    if any(source.count(x) != 1 for x in (first, start, end)):
        raise ValueError("Independent WGMMA helper boundary changed")
    return source[:source.index(first)] + source[source.index(start):source.index(end)]


def code_gate(sass):
    for matrices in (0, 1, 4, 8, 16):
        for rs in (0, 1):
            marker = f"Function : _Z14enqueue_windowILi{matrices}ELb{rs}EE"
            if sass.count(marker) != 1:
                return False
            body = sass.split(marker, 1)[1].split("Function : ", 1)[0]
            instructions = [line for line in body.splitlines() if "/*" in line]
            clocks = [index for index, line in enumerate(instructions) if "SR_CLOCKLO" in line]
            mmas = [index for index, line in enumerate(instructions) if "HGMMA." in line]
            # ptxas materializes an empty committed group with a false-
            # predicated dummy HGMMA. This is the empty-group control,
            # not a BF16 matrix operation or a bare timer calibration.
            if matrices == 0 and (len(mmas) != 1 or not all(
                    "HGMMA.64x8x16.F16 RZ" in instructions[index] and "!UPT" in instructions[index]
                    for index in mmas)):
                return False
            if len(clocks) != 3 or matrices and len(mmas) != matrices:
                return False
            if matrices and not clocks[0] < mmas[0] <= mmas[-1] < clocks[1]:
                return False
            if not any("WARPGROUP.DEPBAR" in line for line in instructions[clocks[1]:clocks[2]]):
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
        (TASK / "enqueue_wgmma_helper.cuh").write_text(header)
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
                result["spill_free"] = len(entries) == 10 and all(a == b == "0" for a, b in entries)
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
        raise RuntimeError("WGMMA enqueue-window capture failed; inspect saved raw.json")
    print({"wall_s": result["gpu_function_wall_s"], "code_gate": result["code_gate"]})
