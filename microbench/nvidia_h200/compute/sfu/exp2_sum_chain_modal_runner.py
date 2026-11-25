"""Run an independent bounded EX2/local-sum register probe on one H200."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent
TASK = Path("/tmp/midlayer-fa-exp2-sum-20261107")
SOURCE = HERE / "exp2_sum_chain.cu"
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
).add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name))
APP = modal.App("midlayer-fa-exp2-sum-20261107")


@APP.function(gpu="H200", image=IMAGE, timeout=240, retries=0,
              single_use_containers=True)
def capture():
    started = time.monotonic()
    result = {}
    try:
        commands = {
            "identity": ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm",
                         "--format=csv,noheader"],
            "compiler": ["nvcc", "--version"],
            "build": ["nvcc", "-O3", "-arch=sm_90", "-Xptxas=-v",
                      str(TASK / SOURCE.name), "-o", str(TASK / "probe")],
            "sass": ["cuobjdump", "--dump-sass", str(TASK / "probe")],
            "measurement": [str(TASK / "probe")],
            "final_clock": ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"],
        }
        for name, command in commands.items():
            process = subprocess.run(command, capture_output=True, text=True, timeout=90)
            result[name] = {"command": command, "exit_code": process.returncode,
                            "stdout": process.stdout, "stderr": process.stderr}
            if process.returncode:
                break
            if name == "sass":
                valid = True
                for width in (20, 32, 40, 44, 48):
                    symbol = f"Function : _Z5chainILi{width}ELi1EEvPyPf"
                    if process.stdout.count(symbol) != 1:
                        valid = False
                        break
                    body = process.stdout.split(symbol, 1)[1].split("Function : ", 1)[0]
                    if body.count("MUFU.EX2") != 2 * width:
                        valid = False
                        break
                result["sass_boundary_valid"] = valid
                if not valid:
                    break
        return result
    finally:
        result["gpu_function_wall_s"] = time.monotonic() - started
        if TASK.exists():
            shutil.rmtree(TASK)


@APP.local_entrypoint()
def main(output_dir: str):
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    if (target / "raw.json").exists():
        raise ValueError("capture already exists")
    result = capture.remote()
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (target / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    required = ("identity", "compiler", "build", "sass", "measurement", "final_clock")
    if (not all(name in result and result[name]["exit_code"] == 0 for name in required)
            or result["sass_boundary_valid"] is not True):
        raise RuntimeError("EX2 capture failed; inspect the saved raw.json")
    print({"wall_s": result["gpu_function_wall_s"], "measurement_exit_code": result["measurement"]["exit_code"] if "measurement" in result else None})
