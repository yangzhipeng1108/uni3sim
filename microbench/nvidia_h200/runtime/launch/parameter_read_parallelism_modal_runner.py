"""Independent parameter-read parallelism with live load groups and NSYS spans."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time
import zlib

import modal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "parameter_read_parallelism.cu"
TASK = Path("/tmp/midlayer-parameter-parallelism-20261107")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
).apt_install("cuda-nsight-systems-12-6").add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name))
APP = modal.App("midlayer-parameter-parallelism-20261107")


def parameter_groups():
    lines = ["template<int Words,int Group> __device__ __forceinline__ unsigned long long grouped_parameter_reads(const Payload<Words>& p);"]
    for words in (128, 480):
        count = words // 8
        for group in (1, 8, count):
            lines += [f"template<> __device__ __forceinline__ unsigned long long grouped_parameter_reads<{words},{group}>(const Payload<{words}>& p) {{", "unsigned long long sum=0;"]
            for start in range(0, count, group):
                end = min(start + group, count)
                lines.append("{")
                lines += [f"unsigned long long v{i} = p.words[{i * 8}];" for i in range(start, end)]
                operands = ", ".join(f'"+l"(v{i})' for i in range(start, end))
                lines.append('asm volatile("bar.warp.sync 0xffffffff;" : ' + operands + ');')
                lines += [f"sum ^= v{i};" for i in range(start, end)]
                lines.append("}")
            lines += ["return sum;", "}"]
    return "\n".join(lines) + "\n"


@APP.function(gpu="H200", image=IMAGE, timeout=240, retries=0,
              single_use_containers=True)
def capture():
    result = {"reports": {}}
    started = time.monotonic()
    try:
        generated = parameter_groups()
        (TASK / "parameter_read_groups.cuh").write_text(generated)
        result["generated_helper"] = generated
        stem = TASK / "trace"
        commands = {
            "identity": ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm", "--format=csv,noheader"],
            "compiler": ["nvcc", "--version"],
            "build": ["nvcc", "-O3", "-arch=sm_90", "-Xptxas=-v", str(TASK / SOURCE.name), "-o", str(TASK / "probe")],
            "sass": ["cuobjdump", "--dump-sass", str(TASK / "probe")],
            "profiler_version": ["nsys", "--version"],
            "plain": [str(TASK / "probe"), "reuse"],
            "profile": ["nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                        "--force-overwrite=true", "--output", str(stem), str(TASK / "probe"), "reuse"],
            "export": ["nsys", "export", "--type=sqlite", "--force-overwrite=true",
                       "--output", str(stem) + ".sqlite", str(stem) + ".nsys-rep"],
        }
        result["required_commands"] = list(commands)
        for name, command in commands.items():
            p = subprocess.run(command, capture_output=True, text=True, timeout=90, cwd=TASK)
            result[name] = {"command": command, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
            if p.returncode:
                break
            if name == "build":
                entries = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", p.stderr)
                result["zero_spill_gate"] = len(entries) == 8 and all(a == b == "0" for a, b in entries)
                if not result["zero_spill_gate"]:
                    break
            if name == "export":
                with sqlite3.connect(str(stem) + ".sqlite") as db:
                    result["kernel_timestamps"] = db.execute(
                        "SELECT k.start,k.end,k.gridX,k.blockX,k.dynamicSharedMemory,s.value "
                        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id "
                        "WHERE s.value LIKE '%parameter_parallel%' ORDER BY k.start").fetchall()
                result["reports"]["trace.nsys-rep.zlib"] = zlib.compress(Path(str(stem) + ".nsys-rep").read_bytes())
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
        raise RuntimeError("Parameter read-footprint capture failed; inspect saved raw.json")
    if len(result["kernel_timestamps"]) != 288:
        raise ValueError("Incomplete parameter read-footprint launch population")
    print({"wall_s": result["gpu_function_wall_s"], "reports": files})
