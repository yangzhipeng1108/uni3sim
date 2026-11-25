"""Capture launch parameters for a source-bound raster question, no timings."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import modal

IMAGE = modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
if modal.is_local():
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parents[3]
    SOURCE = HERE / "nvjet_launch_params.cu"
    IMAGE = IMAGE.add_local_file(SOURCE, remote_path="/tmp/nvjet_launch_params.cu")
APP = modal.App("midlayer-h200-nvjet-launch-params-20261107")


@APP.function(gpu="H200", image=IMAGE, memory=4096, timeout=300, retries=0, single_use_containers=True)
def capture(shape_csv: str | None):
    started = time.monotonic()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,power.limit", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=20).stdout
    if not gpu.startswith("NVIDIA H200,"):
        raise ValueError("one H200 required")
    environment = dict(os.environ)
    removed = []
    for name in ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"):
        if name in environment:
            del environment[name]
            removed.append(name)
    libraries = "/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib"
    if "LD_LIBRARY_PATH" in environment:
        libraries += ":" + environment["LD_LIBRARY_PATH"]
    environment["LD_LIBRARY_PATH"] = libraries
    with tempfile.TemporaryDirectory(prefix="midlayer-nvjet-params-") as directory:
        arguments = []
        if shape_csv is not None:
            plan = Path(directory) / "shapes.csv"
            plan.write_text(shape_csv)
            arguments = [str(plan)]
        command = ["nvcc", "-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v",
            "-I/usr/local/cuda/extras/CUPTI/include", "-L/usr/local/cuda/extras/CUPTI/lib64",
            "/tmp/nvjet_launch_params.cu", "-lcupti", "-lcublas", "-lcublasLt", "-ldl", "-o", directory + "/probe"]
        build = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=90)
        result = {"gpu": gpu, "compiler_flag_environment_removed": removed, "build_command": command,
            "build_exit_code": build.returncode, "build_stdout": build.stdout, "build_stderr": build.stderr}
        if build.returncode == 0:
            result["binary_sha256"] = hashlib.sha256(Path(directory + "/probe").read_bytes()).hexdigest()
            marker = subprocess.run(["cuobjdump", "--dump-sass", "--function", "_Z16parameter_markerPij", directory + "/probe"],
                capture_output=True, text=True, check=True, timeout=20)
            if "Function : _Z16parameter_markerPij" not in marker.stdout:
                raise ValueError("parameter marker SASS missing")
            result["marker_sass_stdout"] = marker.stdout
            result["marker_sass_stderr"] = marker.stderr
            run = subprocess.run([directory + "/probe", *arguments], capture_output=True, text=True, env=environment, timeout=180)
            result.update({"run_exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr})
        result["function_wall_s"] = time.monotonic() - started
    return result


@APP.local_entrypoint()
def main(retry: bool = False, batch: bool = False):
    if retry and batch:
        raise ValueError("choose one-shape retry or fixed221 capture")
    if batch:
        name = "nvjet_launch_params_20261107_all221"
        shape_csv = (ROOT / "experiment_logs/h200_gemm_accuracy_20261107/nvjet_shapes_221.csv").read_text()
    else:
        name = "nvjet_launch_params_20261107_retry" if retry else "nvjet_launch_params_20261107"
        shape_csv = None
    target = ROOT / "profiler_output/microbench/nvidia_h200" / name
    if target.exists():
        raise ValueError("refuse to overwrite captured evidence")
    result = capture.remote(shape_csv)
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["production_duration_consumed"] = False
    result["manual_context_independent_handle_retry"] = retry
    result["fixed221_shape_capture"] = batch
    target.mkdir(parents=True)
    if "marker_sass_stdout" in result:
        data = result["marker_sass_stdout"].encode()
        (target / "parameter_marker.sass.txt.gz").write_bytes(gzip.compress(data, mtime=0))
        result["marker_sass_stdout_sha256"] = hashlib.sha256(data).hexdigest()
        del result["marker_sass_stdout"]
    if batch and "stdout" in result:
        data = result["stdout"].encode()
        (target / "launches.jsonl.gz").write_bytes(gzip.compress(data, mtime=0))
        result["stdout_path"] = "launches.jsonl.gz"
        result["stdout_sha256"] = hashlib.sha256(data).hexdigest()
        del result["stdout"]
    (target / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    snapshot = target / "source"
    snapshot.mkdir()
    for source in (SOURCE, Path(__file__)):
        (snapshot / source.name).write_bytes(source.read_bytes())
    if shape_csv is not None:
        data = shape_csv.encode()
        (snapshot / "shapes.csv").write_bytes(data)
        (target / "shape_input_identity.json").write_text(json.dumps({"sha256": hashlib.sha256(data).hexdigest(),
            "scope": "Only frozen case IDs and FP16 flattened M/N/K; no production durations or observations uploaded."}, indent=2) + "\n")
    print(json.dumps({"build_exit_code": result["build_exit_code"], "function_wall_s": result["function_wall_s"]}))
