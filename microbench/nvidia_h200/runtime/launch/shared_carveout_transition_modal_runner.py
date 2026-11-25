"""Compare fixed versus alternating launch carveouts with independent kernels."""
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
import zlib

import modal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "shared_carveout_transition.cu"
TASK = Path("/tmp/midlayer-shared-transition-20261107")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
).apt_install("cuda-nsight-systems-12-6").add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name))
APP = modal.App("midlayer-shared-transition-20261107")


@APP.function(gpu="H200", image=IMAGE, timeout=240, retries=0,
              single_use_containers=True)
def capture():
    result = {"reports": {}}
    started = time.monotonic()
    try:
        commands = {
            "identity": ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm", "--format=csv,noheader"],
            "compiler": ["nvcc", "--version"],
            "build": ["nvcc", "-O3", "-arch=sm_90", str(TASK / SOURCE.name), "-o", str(TASK / "probe")],
            "profiler_version": ["nsys", "--version"],
        }
        for size in (166912, 196608):
            stem = TASK / f"trace_{size}"
            commands[f"plain_{size}"] = [str(TASK / "probe"), str(size)]
            commands[f"profile_{size}"] = ["nsys", "profile", "--trace=cuda", "--sample=none",
                "--cpuctxsw=none", "--cuda-graph-trace=node", "--force-overwrite=true",
                "--output", str(stem), str(TASK / "probe"), str(size)]
            commands[f"export_{size}"] = ["nsys", "export", "--type=sqlite", "--force-overwrite=true",
                "--output", str(stem) + ".sqlite", str(stem) + ".nsys-rep"]
        result["required_commands"] = list(commands)
        for name, command in commands.items():
            p = subprocess.run(command, capture_output=True, text=True, timeout=90, cwd=TASK)
            result[name] = {"command": command, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
            if p.returncode:
                break
            if name.startswith("export_"):
                size = int(name.split("_")[1])
                stem = TASK / f"trace_{size}"
                with sqlite3.connect(str(stem) + ".sqlite") as db:
                    result[f"kernel_timestamps_{size}"] = db.execute(
                        "SELECT k.start,k.end,k.gridX,k.blockX,k.dynamicSharedMemory,s.value "
                        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id "
                        "WHERE s.value LIKE '%cache_transition%' ORDER BY k.start").fetchall()
                result["reports"][f"trace_{size}.nsys-rep.zlib"] = zlib.compress(
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
        raise RuntimeError("Shared-carveout capture failed; inspect saved raw.json")
    print({"wall_s": result["gpu_function_wall_s"], "reports": files})
