"""Independent uniform/regular parameter-read controls and NSYS spans."""
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
SOURCE = HERE / "parameter_read_pipes.cu"
TASK = Path("/tmp/midlayer-parameter-pipes-20261107")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
).apt_install("cuda-nsight-systems-12-6").add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name))
APP = modal.App("midlayer-parameter-pipes-20261107")


def parameter_groups():
    lines = []
    for name in ('uniform_reads', 'regular_reads'):
        extra = ',unsigned idx' if name == 'regular_reads' else ''
        lines.append(f'template<int Regions> __device__ __forceinline__ unsigned long long {name}(const Payload& p{extra});')
        for count in (2, 16, 60):
            lines += [f'template<> __device__ __forceinline__ unsigned long long {name}<{count}>(const Payload& p{extra}) {{', 'unsigned long long sum=0;']
            for i in range(count):
                word = (59 if i == count - 1 else i) * 8
                index = f'idx+{word}' if name == 'regular_reads' else str(word)
                lines.append(f'unsigned long long v{i}=p.words[{index}];')
            operands = ', '.join(f'"+l"(v{i})' for i in range(count))
            lines.append('asm volatile("bar.warp.sync 0xffffffff;" : ' + operands + ');')
            lines += [f'sum^=v{i};' for i in range(count)]
            lines += ['return sum;', '}']
    return '\n'.join(lines)+'\n'


def code_gate(sass):
    bodies = sass.split('Function : ')[1:]
    if len(bodies) != 18:
        return False
    for body in bodies:
        match = re.search(r'parameter_pipesILi([0-5])ELi(2|16|60)EE', body)
        if match is None:
            return False
        mode, count = map(int, match.groups())
        # Only the opaque per-lane index can produce an indexed LDC. Input
        # payload must stay in constant memory rather than become LDG.
        indexed = len(re.findall(r'\bLDC(?:\.[A-Z0-9]+)*\s+[^;]*c\[0x0\]\[R', body))
        uniform = len(re.findall(r'\bULDC(?:\.[A-Z0-9]+)*\s', body))
        if 'LDG.' in body or 'LDL' in body or 'STL' in body:
            return False
        if mode in (0, 4) and (indexed != 0 or uniform < count - 1):
            return False
        if mode in (1, 2, 3, 5) and indexed < count:
            return False
        if mode in (2, 3) and uniform < count - 1:
            return False
    return True


@APP.function(gpu="H200", image=IMAGE, timeout=120, retries=0,
              single_use_containers=True)
def capture():
    result = {"reports": {}}
    started = time.monotonic()
    try:
        generated = parameter_groups()
        (TASK / "pipe_read_groups.cuh").write_text(generated)
        result["generated_helper"] = generated
        stem = TASK / "trace"
        commands = {
            "identity": ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm", "--format=csv,noheader"],
            "compiler": ["nvcc", "--version"],
            "build": ["nvcc", "-O3", "-arch=sm_90", "-Xptxas=-v", str(TASK / SOURCE.name), "-o", str(TASK / "probe")],
            "sass": ["cuobjdump", "--dump-sass", str(TASK / "probe")],
            "profiler_version": ["nsys", "--version"],
            "plain": [str(TASK / "probe"), "pipes"],
            "profile": ["nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                        "--force-overwrite=true", "--output", str(stem), str(TASK / "probe"), "pipes"],
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
                result["zero_spill_gate"] = len(entries) == 18 and all(a == b == "0" for a, b in entries)
                if not result["zero_spill_gate"]:
                    break
            if name == "sass":
                result["code_gate"] = code_gate(p.stdout)
                if not result["code_gate"]:
                    break
            if name == "export":
                with sqlite3.connect(str(stem) + ".sqlite") as db:
                    result["kernel_timestamps"] = db.execute(
                        "SELECT k.start,k.end,k.gridX,k.blockX,k.dynamicSharedMemory,s.value "
                        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id "
                        "WHERE s.value LIKE '%parameter_pipes%' ORDER BY k.start").fetchall()
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
    if len(result["kernel_timestamps"]) != 1728:
        raise ValueError("Incomplete parameter read-footprint launch population")
    print({"wall_s": result["gpu_function_wall_s"], "reports": files})
