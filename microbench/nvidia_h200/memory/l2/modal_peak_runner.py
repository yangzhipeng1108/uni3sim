"""Capture one H200 NCU report for static peak metadata; no bandwidth fitting."""
import hashlib
import json
from pathlib import Path

import modal


PROBE = r'''
#include <cuda_runtime.h>
#include <cstdio>
#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s\n", cudaGetErrorString(e)); return 1; } } while (0)
__global__ void peak_metadata_probe(float* data) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  data[i] = data[i] + 1.0f;
}
int main() {
  float* data;
  CHECK(cudaMalloc(&data, 256 * 256 * sizeof(float)));
  CHECK(cudaMemset(data, 0, 256 * 256 * sizeof(float)));
  peak_metadata_probe<<<256,256>>>(data);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaFree(data));
  return 0;
}
'''
METRICS = ",".join((
    "lts__lts2xbar_cycles_active.sum.peak_sustained",
    "lts__t_sectors_srcunit_tex_op_read.avg.peak_sustained",
    "lts__t_sectors_srcunit_tex_op_write.avg.peak_sustained",
    "lts__t_sectors.avg.peak_sustained",
    "lts__t_bytes.sum.peak_sustained",
))
app = modal.App("midlayer-h200-l2-peak-metadata")
image = (modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("cuda-nsight-compute-12-6"))


@app.function(image=image, gpu="H200", cpu=1, memory=2048, timeout=180,
    startup_timeout=600, max_containers=1, retries=0, scaledown_window=2,
    serialized=True, include_source=False)
def capture():
    import shutil
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    started = time.time()
    logs = []
    files = {}
    with tempfile.TemporaryDirectory(prefix="h200-l2-peak-") as temporary:
        directory = Path(temporary)
        def run(argv):
            result = subprocess.run(argv, cwd=directory, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            logs.append({"argv": argv, "returncode": result.returncode,
                "output": result.stdout})
            return result.returncode
        run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,clocks.max.sm,clocks.current.sm", "--format=csv"])
        ncu = shutil.which("ncu")
        if ncu is None:
            candidates = sorted(Path("/opt/nvidia/nsight-compute").glob("*/ncu"))
            if len(candidates) != 1:
                raise RuntimeError("Expected exactly one installed NCU executable")
            ncu = str(candidates[0])
        run([ncu, "--version"])
        (directory / "probe.cu").write_text(PROBE)
        code = run(["nvcc", "-O2", "-arch=sm_90", "probe.cu", "-o", "probe"])
        if code == 0:
            code = run([ncu, "--clock-control", "none", "--target-processes", "all", "--launch-count", "1",
                "--metrics", METRICS, "--force-overwrite", "--export", "peak", "./probe"])
        if code == 0:
            run([ncu, "--import", "peak.ncu-rep", "--page", "raw", "--csv", "--print-units", "base"])
        for path in directory.iterdir():
            if path.suffix in (".ncu-rep", ".cu"):
                files[path.name] = path.read_bytes()
    return {"started_unix_s": started, "finished_unix_s": time.time(),
        "returncode": code, "logs": logs, "files": files}


@app.local_entrypoint()
def main(output: str):
    import subprocess
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    result = capture.remote()
    files = result.pop("files")
    hashes = {}
    for name, data in files.items():
        (directory / name).write_bytes(data)
        hashes[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    result["files"] = hashes
    result["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result["gpu_count"] = 1
    result["function_gpu_hours"] = (result["finished_unix_s"] - result["started_unix_s"]) / 3600
    result["budget_gpu_hours"] = 0.25
    result["canonical_publication"] = False
    (directory / "capture.json").write_text(json.dumps(result, indent=2))
    for log in result["logs"]:
        print(json.dumps(log), flush=True)
    print(json.dumps({"output": output, "files": hashes,
        "function_gpu_hours": result["function_gpu_hours"]}), flush=True)
    if result["returncode"] != 0:
        raise RuntimeError("NCU capture failed; inspect saved capture.json")
