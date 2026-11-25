"""Capture staggered independent SS/SFU bursts; requires paid authorization."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "tensor_sfu_phase_offset.cu"
REFERENCE = HERE / "../../operators/flash_attn/fa3_wgmma_service.cu"
TASK = Path("/tmp/midlayer-tensor-sfu-phase-offset-20261107")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
).add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name)).add_local_file(
    REFERENCE, remote_path=str(TASK / REFERENCE.name))
APP = modal.App("midlayer-tensor-sfu-phase-offset-20261107")


def helper(source):
    first = "__device__ __forceinline__ void wgmma_ss_n64_kmajor"
    start = "__device__ __forceinline__ void wgmma_ss_n128_kmajor"
    end = "__device__ __forceinline__ void wgmma_ss_n256_mnmajor"
    if any(source.count(marker) != 1 for marker in (first, start, end)):
        raise ValueError("Independent WGMMA helper boundary changed")
    return source[:source.index(first)] + source[source.index(start):source.index(end)]


def code_gate(sass):
    bodies = sass.split("Function : ")[1:]
    if len(bodies) != 6:
        return False
    for body in bodies:
        match = re.search(r"_Z10phase_pairILi(4|8|16)ELi(64|128)EE", body)
        if match is None:
            return False
        matrices, values = (int(match[1]), int(match[2]))
        if body.count("HGMMA.") != matrices or body.count("MUFU.EX2") != values:
            return False
        batches = [part.count("HGMMA.") for part in body.split("WARPGROUP.DEPBAR") if "HGMMA." in part]
        if batches != [matrices] or "WG.DP" in body:
            return False
        timers = [match.start() for match in re.finditer("SR_GLOBALTIMERLO", body)]
        if len(timers) != 4:
            return False
        tensor_window = next((body[a:b] for a, b in zip(timers, timers[1:])
            if a < body.index("HGMMA.") < body.rindex("HGMMA.") < b), None)
        scalar_window = next((body[a:b] for a, b in zip(timers, timers[1:])
            if a < body.index("MUFU.EX2") < body.rindex("MUFU.EX2") < b), None)
        if tensor_window is None or scalar_window is None:
            return False
        if ("MUFU.EX2" in tensor_window or "HGMMA." in scalar_window
                or "WARPGROUP.DEPBAR" not in tensor_window):
            return False
    return True


@APP.function(gpu="H200", image=IMAGE, timeout=120, retries=0,
              single_use_containers=True)
def capture():
    result = {}
    start = time.monotonic()
    try:
        reference = (TASK / REFERENCE.name).read_text()
        header = helper(reference)
        (TASK / "phase_wgmma_helper.cuh").write_text(header)
        result["helper_source"] = header
        result["reference_sha256"] = hashlib.sha256(reference.encode()).hexdigest()
        commands = {
            "identity": ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm", "--format=csv,noheader"],
            "compiler": ["nvcc", "--version"],
            "build": ["nvcc", "-O3", "-arch=sm_90a", "-Xptxas=-v", str(TASK / SOURCE.name), "-o", str(TASK / "probe")],
            "sass": ["cuobjdump", "--dump-sass", str(TASK / "probe")],
            "measurement": [str(TASK / "probe")],
        }
        result["required_commands"] = list(commands)
        for name, command in commands.items():
            process = subprocess.run(command, cwd=TASK, capture_output=True, text=True, timeout=60)
            result[name] = {"command": command, "exit_code": process.returncode,
                            "stdout": process.stdout, "stderr": process.stderr}
            if process.returncode:
                break
            if name == "build":
                spills = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", process.stderr)
                result["zero_spill_gate"] = len(spills) == 6 and all(a == b == "0" for a, b in spills)
                if not result["zero_spill_gate"]:
                    break
            if name == "sass":
                result["code_gate"] = code_gate(process.stdout)
                if not result["code_gate"]:
                    break
        return result
    finally:
        result["gpu_function_wall_s"] = time.monotonic() - start
        if TASK.exists():
            shutil.rmtree(TASK)


@APP.local_entrypoint()
def main(output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "raw.json").exists():
        raise ValueError("Capture already exists")
    result = capture.remote()
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    if not all(name in result and result[name]["exit_code"] == 0 for name in result["required_commands"]):
        raise RuntimeError("Phase capture command failed; inspect retained raw")
    if not result["zero_spill_gate"] or not result["code_gate"]:
        raise RuntimeError("Phase capture validity gate failed; inspect retained raw")
    print({"function_wall_s": result["gpu_function_wall_s"], "validity_gates": True})
