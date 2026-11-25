"""Independent full CTA envelope versus internally timed synthetic body."""
import hashlib
import csv
import json
from pathlib import Path
import re
import shutil
import sqlite3
import statistics
import subprocess
import time
import zlib

import modal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "register_task_envelope.cu"
TASK = Path("/tmp/midlayer-register-task-envelope-20261107")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
).apt_install("cuda-nsight-systems-12-6").add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name))
APP = modal.App("midlayer-register-task-envelope-20261107")


def pressure_helper():
    lines = ["template<int Threads> __device__ __forceinline__ float pressure(float seed);"]
    for threads, count in ((256, 224), (384, 144), (512, 128)):
        lines.extend([f"template<> __device__ __forceinline__ float pressure<{threads}>(float seed) {{",
                      f"float v[{count}];"])
        lines.extend(f"v[{i}] = seed + {float(i)}f;" for i in range(count))
        operands = ", ".join(f'"+f"(v[{i}])' for i in range(count))
        lines.extend(['asm volatile("bar.warp.sync 0xffffffff;" : ' + operands + ");", "float sum=0;"])
        lines.extend(f"sum += v[{i}];" for i in range(count))
        lines.extend(["return sum;", "}"])
    return "\n".join(lines) + "\n"


def spill_free_build(log):
    entries = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", log)
    return len(entries) == 18 and all(stores == "0" and loads == "0" for stores, loads in entries)


def analyze(result):
    """Join ordered CUPTI launches and device spans; no predictor prices."""
    lines = result["profile"]["stdout"].splitlines()
    header = "launch_id,grid,threads,high,payload_bytes,extended,hold_cycles,trial,registers,begin_ns,end_ns,body_cycles"
    if lines.count(header) != 1:
        raise ValueError("Expected exactly one CTA trace header")
    rows = [{key: int(value) for key, value in row.items()} for row in csv.DictReader(
        [header] + [line for line in lines if re.fullmatch(r"[0-9]+(?:,-?[0-9]+){11}", line)])]
    groups = {}
    for row in rows:
        index = row["launch_id"]
        if index not in groups:
            groups[index] = []
        groups[index].append(row)
    if len(rows) != 71820 or set(groups) != set(range(1080)) or len(result["kernel_timestamps"]) != 1080:
        raise ValueError("Incomplete CTA or CUPTI capture")
    launches = []
    fields = ("launch_id", "grid", "threads", "high", "payload_bytes", "extended", "hold_cycles", "trial", "registers")
    for index, kernel in enumerate(result["kernel_timestamps"]):
        group = groups[index]
        row = group[0]
        start, end, grid, threads, shared, symbol = kernel
        if (len(group) != grid or row["grid"] != grid or row["threads"] != threads or shared != 131072
                or any(any(item[field] != row[field] for field in fields) for item in group)
                or f'<(int){threads}, (bool){row["high"]}, (int){row["payload_bytes"] // 8}>' not in symbol):
            raise ValueError("CUPTI/device trace geometry or order mismatch")
        body = max(item["end_ns"] for item in group) - min(item["begin_ns"] for item in group)
        if not 0 < body <= end - start or any(item["body_cycles"] <= 0 for item in group):
            raise ValueError("Invalid body or outer time interval")
        launches.append({**{field: row[field] for field in fields}, "kernel_ns": end - start,
                         "body_span_ns": body, "outer_ns": end - start - body,
                         "median_body_cycles": statistics.median(item["body_cycles"] for item in group)})
    cells = {}
    for row in launches:
        if row["trial"] < 0:
            continue
        key = tuple(row[field] for field in fields[1:7])
        if key not in cells:
            cells[key] = []
        cells[key].append(row["outer_ns"])
    if len(cells) != 216 or any(len(values) != 3 for values in cells.values()):
        raise ValueError("Incomplete timed cell population")
    summary = {"scope": "Independent CTA resource/parameter envelope, not a native operator ABI or Forward price",
               "kernel_count": len(launches), "cta_trace_rows": len(rows),
               "outer_min_ns": min(row["outer_ns"] for row in launches),
               "outer_max_ns": max(row["outer_ns"] for row in launches),
               "cells": [{**dict(zip(fields[1:7], key)), "median_outer_ns": statistics.median(values),
                          "outer_samples_ns": values} for key, values in cells.items()]}
    return launches, summary


@APP.function(gpu="H200", image=IMAGE, timeout=240, retries=0,
              single_use_containers=True)
def capture():
    result = {"reports": {}}
    started = time.monotonic()
    try:
        header = pressure_helper()
        (TASK / "envelope_pressure.cuh").write_text(header)
        result["generated_helper"] = header
        stem = TASK / "trace"
        commands = {
            "identity": ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm", "--format=csv,noheader"],
            "compiler": ["nvcc", "--version"],
            "build": ["nvcc", "-O3", "-arch=sm_90a", "-Xptxas=-v",
                      str(TASK / SOURCE.name), "-o", str(TASK / "probe")],
            "sass": ["cuobjdump", "--dump-sass", str(TASK / "probe")],
            "profiler_version": ["nsys", "--version"],
            "plain": [str(TASK / "probe")],
            "profile": ["nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                        "--force-overwrite=true", "--output", str(stem), str(TASK / "probe")],
            "export": ["nsys", "export", "--type=sqlite", "--force-overwrite=true",
                       "--output", str(stem) + ".sqlite", str(stem) + ".nsys-rep"],
        }
        result["required_commands"] = list(commands)
        for name, command in commands.items():
            p = subprocess.run(command, capture_output=True, text=True, timeout=90, cwd=TASK)
            result[name] = {"command": command, "exit_code": p.returncode,
                            "stdout": p.stdout, "stderr": p.stderr}
            if p.returncode:
                break
            if name == "build":
                result["spill_free_build_gate"] = spill_free_build(p.stderr)
                if not result["spill_free_build_gate"]:
                    break
            if name == "export":
                with sqlite3.connect(str(stem) + ".sqlite") as db:
                    result["kernel_timestamps"] = db.execute(
                        "SELECT k.start,k.end,k.gridX,k.blockX,k.dynamicSharedMemory,s.value "
                        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id "
                        "WHERE s.value LIKE '%register_envelope%' ORDER BY k.start").fetchall()
                result["reports"]["trace.nsys-rep.zlib"] = zlib.compress(
                    Path(str(stem) + ".nsys-rep").read_bytes())
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
    files = {}
    for name, data in result["reports"].items():
        (out / name).write_bytes(data)
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "compression": "zlib"}
    result["reports"] = files
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    if not all(name in result and result[name]["exit_code"] == 0 for name in result["required_commands"]):
        raise RuntimeError("CTA envelope capture failed; inspect saved raw.json")
    launches, summary = analyze(result)
    summary["analysis_runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "launch_spans.json").write_text(json.dumps(launches, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print({"wall_s": result["gpu_function_wall_s"], "reports": files})
