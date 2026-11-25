"""Independent cold/repeated strided copy readiness, with direct SI timing."""
import hashlib
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "strided_async_copy_ready.cu"
TASK = Path("/tmp/midlayer-strided-copy-ready-20261107")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
).apt_install("cuda-nsight-compute-12-6").add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name))
APP = modal.App("midlayer-strided-copy-ready-20261107")


def timed_copy_gate(sass):
    for tma in (0, 1):
        marker = f"Function : _Z5readyILb{tma}EE"
        if sass.count(marker) != 1:
            return False
        body = sass.split(marker, 1)[1].split("Function : ", 1)[0]
        instructions = [line for line in body.splitlines() if "/*" in line]
        clocks = [i for i, line in enumerate(instructions) if "SR_CLOCKLO" in line]
        global_clocks = [i for i, line in enumerate(instructions) if "SR_GLOBALTIMERLO" in line]
        copies = [i for i, line in enumerate(instructions) if ("UTMALDG." if tma else "LDGSTS.") in line]
        if len(clocks) != 2 or len(global_clocks) != 2 or not copies:
            return False
        if not global_clocks[0] < clocks[0] < copies[0] <= copies[-1] < clocks[1] < global_clocks[1]:
            return False
        wait = "SYNCS.PHASECHK" if tma else "DEPBAR.LE"
        if not any(wait in line for line in instructions[copies[-1]:clocks[1]]):
            return False
    return True


def counter_pairs(stdout):
    lines = stdout.splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith('"ID","Process ID"')]
    if len(headers) != 1:
        raise ValueError("Expected one wide-format NCU counter table")
    rows = [row for row in csv.DictReader(lines[headers[0]:]) if row["ID"] != ""]
    if len(rows) != 4 or [row["ID"] for row in rows] != ["0", "1", "2", "3"]:
        raise ValueError("Incomplete strided-copy counter pairs")
    for i, row in enumerate(rows):
        if f"ready<{i // 2}>" not in row["Kernel Name"]:
            raise ValueError("Counter copy-path order mismatch")
        for metric in ("lts__t_sectors_op_read.sum", "dram__bytes_read.sum"):
            if float(row[metric]) < 0:
                raise ValueError("Invalid traffic counter")
    return rows


@APP.function(gpu="H200", image=IMAGE, timeout=240, retries=0,
              single_use_containers=True)
def capture():
    result = {}
    started = time.monotonic()
    try:
        commands = {
            "identity": ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm", "--format=csv,noheader"],
            "compiler": ["nvcc", "--version"],
            "build": ["nvcc", "-O3", "-arch=sm_90", "-Xptxas=-v", str(TASK / SOURCE.name),
                      "-L/usr/local/cuda/lib64/stubs", "-lcuda", "-o", str(TASK / "probe")],
            "sass": ["cuobjdump", "--dump-sass", str(TASK / "probe")],
            "measurement": [str(TASK / "probe")],
            "counter": ["ncu", "--csv", "--page", "raw", "--cache-control", "none", "--clock-control", "none",
                        "--kernel-name", "regex:ready", "--launch-count", "4", "--metrics",
                        "lts__t_sectors_op_read.sum,dram__bytes_read.sum", str(TASK / "probe"), "counter"],
        }
        for name, command in commands.items():
            p = subprocess.run(command, capture_output=True, text=True, timeout=90)
            result[name] = {"command": command, "exit_code": p.returncode,
                            "stdout": p.stdout, "stderr": p.stderr}
            if p.returncode:
                break
            if name == "build":
                spills = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", p.stderr)
                result["zero_spill_gate"] = len(spills) == 4 and all(a == b == "0" for a, b in spills)
                if not result["zero_spill_gate"]:
                    break
            if name == "sass":
                result["timed_copy_gate"] = timed_copy_gate(p.stdout)
                if not result["timed_copy_gate"]:
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
            "identity", "compiler", "build", "sass", "measurement", "counter")):
        raise RuntimeError("Strided-copy capture failed; inspect saved raw.json")
    if len(result["measurement"]["stdout"].splitlines()) != 96881:
        raise RuntimeError("Incomplete strided-copy sample population")
    counter_pairs(result["counter"]["stdout"])
    print({"wall_s": result["gpu_function_wall_s"], "counter_exit": result["counter"]["exit_code"]})
