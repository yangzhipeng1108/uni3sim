"""Measure an independent SM90a register-pool handoff, not an operator."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "register_pool_handoff.cu"
TASK = Path("/tmp/midlayer-register-pool-handoff-20261107")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
).add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name))
APP = modal.App("midlayer-register-pool-handoff-20261107")


def pressure_helper():
    lines = ["__device__ __forceinline__ float pressure(float seed) {", "float v[160];"]
    lines.extend(f"v[{i}] = seed + {float(i)}f;" for i in range(160))
    operands = ", ".join(f'"+f"(v[{i}])' for i in range(160))
    lines.extend(['asm volatile("bar.warp.sync 0xffffffff;" : ' + operands + ");", "float sum=0;"])
    lines.extend(f"sum += v[{i}];" for i in range(160))
    lines.extend(["return sum;", "}"])
    return "\n".join(lines) + "\n"


@APP.function(gpu="H200", image=IMAGE, timeout=240, retries=0,
              single_use_containers=True)
def capture():
    started = time.monotonic()
    result = {}
    try:
        header = pressure_helper()
        (TASK / "register_pool_pressure.cuh").write_text(header)
        result["generated_helper"] = header
        commands = {
            "identity": ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm", "--format=csv,noheader"],
            "compiler": ["nvcc", "--version"],
            "build": ["nvcc", "-O3", "-arch=sm_90a", "-maxrregcount=168", "-Xptxas=-v",
                      str(TASK / SOURCE.name), "-o", str(TASK / "probe")],
            "sass": ["cuobjdump", "--dump-sass", str(TASK / "probe")],
            "measurement": [str(TASK / "probe")],
        }
        for name, command in commands.items():
            p = subprocess.run(command, capture_output=True, text=True, timeout=30)
            result[name] = {"command": command, "exit_code": p.returncode,
                            "stdout": p.stdout, "stderr": p.stderr}
            if p.returncode:
                break
            if name == "build":
                body = p.stderr.split("Compiling entry function '_Z4poolILb1EEvPyPf'", 1)[1]
                body = body.split("Compiling entry function", 1)[0]
                counts = re.findall(r"Used (\d+) registers", body)
                result["pool_allocation_valid"] = len(counts) == 1 and 160 < int(counts[0]) <= 168
                if not result["pool_allocation_valid"]:
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
    required = ("identity", "compiler", "build", "sass", "measurement")
    if not all(name in result and result[name]["exit_code"] == 0 for name in required):
        raise RuntimeError("Register-pool capture failed; inspect saved raw.json")
    print({"wall_s": result["gpu_function_wall_s"], "measurement_exit_code": result["measurement"]["exit_code"]})
