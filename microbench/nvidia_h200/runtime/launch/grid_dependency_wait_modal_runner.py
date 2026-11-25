"""Capture independent ready-predecessor SM90 grid-dependency waits."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "grid_dependency_wait.cu"
TASK = Path("/tmp/midlayer-grid-dependency-wait-20261107")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
).add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name))
APP = modal.App("midlayer-grid-dependency-wait-20261107")


def timed_acquire(sass):
    marker = "Function : _Z10ready_waitILi1EEvPy"
    if sass.count(marker) != 1:
        return False
    body = sass.split(marker, 1)[1].split("Function : ", 1)[0]
    if body.count("ACQBULK") != 1:
        return False
    before, after = body.split("ACQBULK", 1)
    return before.count("CS2R") == 2 and after.count("CS2R") == 2


@APP.function(gpu="H200", image=IMAGE, timeout=240, retries=0,
              single_use_containers=True)
def capture():
    started = time.monotonic()
    result = {}
    try:
        commands = {
            "identity": ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm", "--format=csv,noheader"],
            "compiler": ["nvcc", "--version"],
            "build": ["nvcc", "-O3", "-arch=sm_90", "-Xptxas=-v", str(TASK / SOURCE.name), "-o", str(TASK / "probe")],
            "sass": ["cuobjdump", "--dump-sass", str(TASK / "probe")],
            "measurement": [str(TASK / "probe")],
        }
        for name, command in commands.items():
            p = subprocess.run(command, capture_output=True, text=True, timeout=90)
            result[name] = {"command": command, "exit_code": p.returncode,
                            "stdout": p.stdout, "stderr": p.stderr}
            if p.returncode:
                break
            if name == "sass":
                result["timed_acquire_gate"] = timed_acquire(p.stdout)
                if not result["timed_acquire_gate"]:
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
        raise RuntimeError("Grid-dependency capture failed; inspect saved raw.json")
    print({"wall_s": result["gpu_function_wall_s"], "timed_acquire_gate": result["timed_acquire_gate"]})
