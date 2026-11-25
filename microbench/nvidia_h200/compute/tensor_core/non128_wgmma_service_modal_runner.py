"""Independent non-N128 retire/clock evidence with registered-A footprints."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "non128_wgmma_service.cu"
REFERENCE = HERE / "../../operators/flash_attn/fa3_wgmma_service.cu"
GENERATOR = HERE / "../../operators/flash_attn/fa3_wgmma_service_gen.py"
TASK = Path("/tmp/midlayer-non128-wgmma-service-20261107")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
).add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name)).add_local_file(
    REFERENCE, remote_path=str(TASK / REFERENCE.name)).add_local_file(
    GENERATOR, remote_path=str(TASK / GENERATOR.name))
APP = modal.App("midlayer-non128-wgmma-service-20261107")


def helper(reference, generator):
    marker = "__device__ __forceinline__ void wgmma_ss_n64_kmajor"
    if reference.count(marker) != 1:
        raise ValueError("Independent helper preamble changed")
    functions = [node for node in ast.parse(generator).body
                 if isinstance(node, ast.FunctionDef) and node.name == "wgmma_function"]
    if len(functions) != 1:
        raise ValueError("Independent PTX generator changed")
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<primitive-generator>", "exec"), namespace)
    header = reference[:reference.index(marker)]
    for n, register_a in ((128, False), (176, False), (192, False), (64, True), (128, True)):
        name = ("rs" if register_a else "ss") + f"_n{n}_" + ("mnmajor" if register_a else "kmajor")
        header += namespace["wgmma_function"](name, n, register_a, int(register_a))
    return header


def code_gate(sass):
    bodies = sass.split("Function : ")[1:]
    if len(bodies) != 6 or "WG.DP" in sass:
        return False
    for body in bodies:
        match = re.search(r"shape_serviceILi(\d+)ELi(\d+)ELb([01])ELi(\d+)EE", body)
        if match is None:
            return False
        n, steps, _, values = map(int, match.groups())
        if body.count(f"HGMMA.64x{n}x16.") != 2 * steps or body.count("MUFU.EX2") != 2 * values:
            return False
        batches = [part.count("HGMMA.") for part in body.split("WARPGROUP.DEPBAR") if "HGMMA." in part]
        if batches != [steps, steps]:
            return False
    return True


@APP.function(gpu="H200", image=IMAGE, timeout=120, retries=0, single_use_containers=True)
def capture():
    result = {}; started = time.monotonic()
    try:
        reference = (TASK / REFERENCE.name).read_text()
        generator = (TASK / GENERATOR.name).read_text()
        header = helper(reference, generator)
        (TASK / "non128_wgmma_helper.cuh").write_text(header)
        result["helper_source"] = header
        result["reference_sha256"] = hashlib.sha256(reference.encode()).hexdigest()
        result["generator_sha256"] = hashlib.sha256(generator.encode()).hexdigest()
        commands = {
            "identity": ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm", "--format=csv,noheader"],
            "compiler": ["nvcc", "--version"],
            "build": ["nvcc", "-O3", "-arch=sm_90a", "-Xptxas=-v", str(TASK / SOURCE.name), "-o", str(TASK / "probe")],
            "sass": ["cuobjdump", "--dump-sass", str(TASK / "probe")],
            "measurement": [str(TASK / "probe")],
        }
        result["required_commands"] = list(commands)
        for name, command in commands.items():
            p = subprocess.run(command, capture_output=True, text=True, timeout=60, cwd=TASK)
            result[name] = {"command": command, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
            if p.returncode:
                break
            if name == "build":
                spills = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", p.stderr)
                result["zero_spill_gate"] = len(spills) == 6 and all(a == b == "0" for a, b in spills)
                if not result["zero_spill_gate"]:
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
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if (out / "raw.json").exists():
        raise ValueError("Capture already exists")
    result = capture.remote()
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    if not all(name in result and result[name]["exit_code"] == 0 for name in result["required_commands"]):
        raise RuntimeError("Primitive capture command failed")
    if not result["zero_spill_gate"] or not result["code_gate"]:
        raise RuntimeError("Primitive compiler gate failed")
    print({"function_wall_s": result["gpu_function_wall_s"], "compiler_gates": True})
