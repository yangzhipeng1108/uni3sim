"""Independent strided multicast operand queues, no Tensor/GEMM or timing fit."""
import gzip
import hashlib
import json
from pathlib import Path

import modal

IMAGE = modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
APP = modal.App("midlayer-h200-cluster-multicast-stride-20261107")


@APP.function(image=IMAGE, serialized=True, include_source=False, memory=1024,
              timeout=30, startup_timeout=60, retries=0, single_use_containers=True)
def precompile(source):
    import hashlib
    import subprocess
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory(prefix="midlayer-cluster-stride-cpu-") as directory:
        path = Path(directory) / "probe.cu"
        path.write_bytes(source)
        command = ["nvcc", "-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
                   "-DTMA_PAYLOAD_A_BYTES=1024", "-DTMA_ROW_STRIDE_BYTES=65536",
                   str(path), "-o", directory + "/probe"]
        run = subprocess.run(command, capture_output=True, text=True, timeout=25)
        return {"gpu": None, "source_sha256": hashlib.sha256(source).hexdigest(),
                "command": command, "exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr}


@APP.function(gpu="H200", image=IMAGE, serialized=True, include_source=False, memory=2048,
              timeout=300, startup_timeout=60, retries=0, single_use_containers=True)
def capture(source):
    import hashlib
    import os
    import subprocess
    import tempfile
    import time
    from pathlib import Path
    started = time.monotonic()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,power.limit",
                          "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=20).stdout
    if not gpu.startswith("NVIDIA H200,"):
        raise ValueError("one H200 required")
    environment = dict(os.environ)
    for name in ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"):
        if name in environment:
            del environment[name]
    result = {"gpu": gpu, "source_sha256": hashlib.sha256(source).hexdigest(), "captures": []}
    with tempfile.TemporaryDirectory(prefix="midlayer-cluster-stride-") as directory:
        path = Path(directory) / "probe.cu"
        path.write_bytes(source)
        for stride in (128,4096,16384,65536):
            command = ["nvcc", "-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
                       "-DTMA_PAYLOAD_A_BYTES=1024", "-DTMA_ROW_STRIDE_BYTES=" + str(stride),
                       "-Xptxas=-v", str(path), "-o", directory + "/probe"]
            build = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=35)
            item = {"row_stride_bytes": stride, "build_command": command, "build_exit_code": build.returncode,
                    "build_stdout": build.stdout, "build_stderr": build.stderr, "run_exit_code": None}
            result["captures"].append(item)
            if build.returncode != 0:
                break
            item["binary_sha256"] = hashlib.sha256(Path(directory + "/probe").read_bytes()).hexdigest()
            dump = subprocess.run(["cuobjdump", "--dump-sass", directory + "/probe"],
                                  capture_output=True, text=True, check=True, timeout=15)
            item["sass"] = dump.stdout
            try:
                run = subprocess.run([directory + "/probe"], capture_output=True, text=True,
                                     env=environment, timeout=50)
                item.update({"run_exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr})
            except subprocess.TimeoutExpired as error:
                output = error.stdout
                item.update({"run_exit_code": 124, "stdout": output.decode() if isinstance(output, bytes) else ("" if output is None else output),
                             "stderr": "subprocess timeout; capture stopped, no later controls attempted"})
            if item["run_exit_code"] != 0:
                break
        result["function_wall_s"] = time.monotonic() - started
        return result


@APP.local_entrypoint()
def main():
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    target = root / "profiler_output/microbench/nvidia_h200/cluster_multicast_stride_20261107"
    if {p.name for p in target.iterdir()} != {"preflight.json", "local_build.log.gz"}:
        raise ValueError("refuse to overwrite captured evidence")
    primary = (here / "cluster_multicast_operand_copy.cu").read_text()
    dependency = (root / "microbench/nvidia_h200/runtime/launch/cluster_operand_token_ring.cu").read_text()
    source = primary.replace('#include "../../runtime/launch/cluster_operand_token_ring.cu"', dependency).encode()
    smoke = precompile.remote(source)
    (target / "cpu_smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
    if smoke["exit_code"] != 0 or smoke["source_sha256"] != hashlib.sha256(source).hexdigest():
        raise ValueError("CPU compile/source failed; GPU not called")
    result = capture.remote(source)
    if result["source_sha256"] != hashlib.sha256(source).hexdigest():
        raise ValueError("source changed")
    for item in result["captures"]:
        for key, suffix in (("sass", ".sass.txt.gz"), ("stdout", ".timings.jsonl.gz")):
            if key in item:
                data = item[key].encode()
                name = str(item["row_stride_bytes"]) + suffix
                (target / name).write_bytes(gzip.compress(data, mtime=0))
                item[key + "_path"] = name
                item[key + "_sha256"] = hashlib.sha256(data).hexdigest()
                del item[key]
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["production_duration_consumed"] = False
    (target / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    snapshot = target / "source"
    snapshot.mkdir()
    (snapshot / "translation_unit.cu").write_bytes(source)
    (snapshot / "cluster_multicast_operand_copy.cu").write_text(primary)
    (snapshot / "cluster_operand_token_ring.cu").write_text(dependency)
    (snapshot / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"build_codes": [x["build_exit_code"] for x in result["captures"]],
                      "run_codes": [x["run_exit_code"] for x in result["captures"]], "function_wall_s": result["function_wall_s"]}))
