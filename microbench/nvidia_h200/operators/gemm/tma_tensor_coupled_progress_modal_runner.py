"""Independent TMA / Tensor-consumer progress with explicit source arguments."""
import gzip
import hashlib
import json
from pathlib import Path

import modal

IMAGE = modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
APP = modal.App("midlayer-h200-tma-tensor-coupled-progress-20261107")


@APP.function(image=IMAGE, serialized=True, include_source=False, memory=1024,
              timeout=30, startup_timeout=30, retries=0, single_use_containers=True)
def precompile(source):
    import hashlib
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="midlayer-tma-precompile-") as directory:
        path = Path(directory) / "probe.cu"
        path.write_bytes(source)
        command = ["nvcc", "-O3", "-std=c++17",
                   "-gencode=arch=compute_90a,code=sm_90a", str(path), "-o", directory + "/probe"]
        run = subprocess.run(command, capture_output=True, text=True, timeout=25)
        return {"gpu": None, "source_sha256": hashlib.sha256(source).hexdigest(),
                "command": command, "exit_code": run.returncode,
                "stdout": run.stdout, "stderr": run.stderr}


@APP.function(gpu="H200", image=IMAGE, serialized=True, include_source=False,
              memory=2048, timeout=300, startup_timeout=60, retries=0, single_use_containers=True)
def capture(source):
    import hashlib
    import os
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    started = time.monotonic()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,power.limit",
                          "--format=csv,noheader"], capture_output=True, text=True,
                         check=True, timeout=20).stdout
    if not gpu.startswith("NVIDIA H200,"):
        raise ValueError("one H200 required")
    environment = dict(os.environ)
    for name in ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"):
        if name in environment:
            del environment[name]
    with tempfile.TemporaryDirectory(prefix="midlayer-tma-finite-") as directory:
        path = Path(directory) / "probe.cu"
        path.write_bytes(source)
        command = ["nvcc", "-O3", "-std=c++17",
                   "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v",
                   str(path), "-o", directory + "/probe"]
        build = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=60)
        result = {"gpu": gpu, "source_sha256": hashlib.sha256(source).hexdigest(),
                  "build_command": command, "build_exit_code": build.returncode,
                  "build_stdout": build.stdout, "build_stderr": build.stderr}
        if build.returncode == 0:
            result["binary_sha256"] = hashlib.sha256(Path(directory + "/probe").read_bytes()).hexdigest()
            dump = subprocess.run(["cuobjdump", "--dump-sass", directory + "/probe"],
                                  capture_output=True, text=True, check=True, timeout=20)
            result["sass"] = dump.stdout
            try:
                run = subprocess.run([directory + "/probe"], capture_output=True, text=True,
                                     env=environment, timeout=85)
                result.update({"run_exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr})
            except subprocess.TimeoutExpired as error:
                output = error.stdout
                result.update({"run_exit_code": 124,
                    "stdout": output.decode() if isinstance(output, bytes) else ("" if output is None else output),
                    "stderr": "subprocess timeout; no retry"})
        result["function_wall_s"] = time.monotonic() - started
        return result


@APP.local_entrypoint()
def main():
    import subprocess
    import sys
    import tempfile
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    target = root / "profiler_output/microbench/nvidia_h200/tma_tensor_coupled_progress_20261107"
    if {p.name for p in target.iterdir()} != {"preflight.json", "local_build.log.gz"}:
        raise ValueError("refuse to overwrite captured evidence")
    primary_path = here / "tma_tensor_coupled_progress.cu"
    dependency_path = here / "tma_queue_service.cu"
    generator = root / "microbench/nvidia_h200/compute/tensor_core/shared_tensor_activity_gen.py"
    primitives = root / "microbench/nvidia_h200/operators/flash_attn/fa3_wgmma_service.cu"
    primary, dependency = primary_path.read_text(), dependency_path.read_text()
    with tempfile.TemporaryDirectory(prefix="midlayer-coupled-helper-") as directory:
        helper = Path(directory) / "shared_tensor_activity_helper.cuh"
        subprocess.run([sys.executable, str(generator), str(primitives), str(helper)], check=True, timeout=10)
        generated = helper.read_text()
    source = primary.replace('#include "tma_queue_service.cu"', dependency).replace(
        '#include "shared_tensor_activity_helper.cuh"', generated).encode()
    snapshot = target / "source"
    snapshot.mkdir()
    (snapshot / "translation_unit.cu").write_bytes(source)
    for path in (primary_path, dependency_path, generator, primitives, Path(__file__)):
        (snapshot / path.name).write_bytes(path.read_bytes())
    (snapshot / "shared_tensor_activity_helper.cuh").write_text(generated)
    smoke = precompile.remote(source)
    (target / "cpu_smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
    if smoke["exit_code"] != 0 or smoke["source_sha256"] != hashlib.sha256(source).hexdigest():
        raise ValueError("CPU source/compile failed; GPU not called")
    result = capture.remote(source)
    if result["source_sha256"] != hashlib.sha256(source).hexdigest():
        raise ValueError("GPU source changed")
    for key, name in (("sass", "queue.sass.txt.gz"), ("stdout", "timings.jsonl.gz")):
        if key in result:
            data = result[key].encode()
            (target / name).write_bytes(gzip.compress(data, mtime=0))
            result[key + "_path"] = name
            result[key + "_sha256"] = hashlib.sha256(data).hexdigest()
            del result[key]
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["production_duration_consumed"] = False
    (target / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"build_exit_code": result["build_exit_code"], "function_wall_s": result["function_wall_s"],
        "run_exit_code": result["run_exit_code"] if "run_exit_code" in result else None}))
