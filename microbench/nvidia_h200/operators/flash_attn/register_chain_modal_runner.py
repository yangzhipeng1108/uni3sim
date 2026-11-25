"""Run an independent FP32 register-chain probe on one H200."""
import hashlib
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import time
import statistics

import modal

HERE = Path(__file__).resolve().parent
TASK = Path("/tmp/midlayer-fa-register-chain-20261107")
SOURCE = HERE / "register_chain.cu"
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
).add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name))
APP = modal.App("midlayer-fa-register-chain-20261107")


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
        return result
    finally:
        result["gpu_function_wall_s"] = time.monotonic() - started
        shutil.rmtree(TASK, ignore_errors=True)


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
    if "measurement" in result and result["measurement"]["exit_code"] == 0:
        rows = list(csv.DictReader(io.StringIO(result["measurement"]["stdout"])))
        entries = []
        for width in (20, 40, 44, 48):
            medians = {mode: statistics.median(
                int(row["cycles"]) / 512 for row in rows
                if int(row["width"]) == width and int(row["mode"]) == mode)
                for mode in (0, 1, 2)}
            entries.append({"score_values_per_lane_local_row": width,
                "fp32_max_quad4_cycles": medians[1] - medians[0],
                "fp32_local_sum_cycles": medians[2] - medians[0]})
        summary = {"hardware_id": "nvidia_h200",
            "mechanism": "two_row_fp32_register_chains", "threads_per_cta": 128,
            "lane_local_rows_per_thread": 2, "timer_domain": "clock64_sm_cycles",
            "production_or_evaluation_duration_consumed": False,
            "source_sha256": result["source_sha256"],
            "raw_path": str(target / "raw.json"),
            "raw_sha256": hashlib.sha256((target / "raw.json").read_bytes()).hexdigest(),
            "derivation": "Median over three trials and 132 CTA clocks / 512 repeats, minus same-width empty-loop median. Exact widths only; N32 is a control, existing N32 prices retained.",
            "entries": entries}
        (target / "register_chains.json").write_text(json.dumps(summary, indent=2) + "\n")
    print({"wall_s": result["gpu_function_wall_s"], "measurement_exit_code": result["measurement"]["exit_code"] if "measurement" in result else None})
