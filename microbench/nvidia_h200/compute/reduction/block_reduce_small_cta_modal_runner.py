"""Bounded pure-CUB small-CTA scope screen; never prices a complete Norm."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import modal

HERE = Path(__file__).resolve().parent
SOURCES = ("block_reduce_latency.cu", "block_reduce_small_cta.cu")
CELLS = tuple((128, 16, grid) for grid in (1, 132, 4224, 8448)) + tuple(
    (4096, 256, grid) for grid in (1, 132, 1056, 2112))
SAMPLES = 63
SELECTION = {
    "purpose": "first/resident/queued CUB<float,1024> division/rsqrt/publication scope at H128/block16, with H4096/block256 control",
    "cells": CELLS, "samples_per_cell": SAMPLES, "process_rotations": 2,
    "second_order": "reverse", "warmup": "250ms full-grid clock warmup then64 same-cell untimed launches",
    "timer": "clock64 cycles, only within the same SM; no cross-SM subtraction or nominal-clock claim",
    "production_operator_body": "absent: no tensor input loads, square accumulation, residual update, scale or output tensor streams",
    "acceptance": ["all16cells complete", "every output validates against scalar CPU reference",
        "every sample has exactly the declared CTA count", "full-SM cells cover132SMs",
        "CUDA occupancy metadata uses compiler resources, not a thread-only bound", "zero spills",
        "95% median CI half-width<=3% and between-process medians<=3%",
        "H4096/block256 control must match the existing cycle reference before transferring new parameters"],
    "model_benefit_basis": "current full-model H128 q/k-Norm sums0.567–1.034ms prefill/mixed and0.086ms decode; this is a component scale, not an attainable saving or a bound on missing cost",
    "adoption": "no automatic reference, rate, HWSpec or Forward update; failed cells retained; no selection by Serving error",
}
IMAGE = modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
for source in SOURCES:
    IMAGE = IMAGE.add_local_file(HERE / source, remote_path="/opt/midlayer-cub-probe/" + source)
APP = modal.App("midlayer-serving-cub-small-cta-20261108")


@APP.function(gpu="H200", image=IMAGE, timeout=300, retries=0, max_containers=1, single_use_containers=True)
def capture():
    started = time.monotonic()
    work = Path(tempfile.mkdtemp(prefix="midlayer-serving-cub-small-"))
    manifest = {"status": "started", "selection": SELECTION, "commands": [], "source_sha256": {}}

    def run(command, name, limit):
        began = time.monotonic()
        try:
            result = subprocess.run(command, cwd=work, capture_output=True, timeout=limit)
            code, stdout, stderr = result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired as error:
            code = 124
            stdout = b"" if error.stdout is None else error.stdout
            stderr = b"" if error.stderr is None else error.stderr
        (work / (name + ".stdout.gz")).write_bytes(gzip.compress(stdout, mtime=0))
        (work / (name + ".stderr.gz")).write_bytes(gzip.compress(stderr, mtime=0))
        manifest["commands"].append({"name": name, "command": command, "exit_code": code,
            "wall_s": time.monotonic() - began, "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest()})
        return code, stdout

    try:
        for name in SOURCES:
            raw = Path("/opt/midlayer-cub-probe", name).read_bytes()
            (work / name).write_bytes(raw)
            manifest["source_sha256"][name] = hashlib.sha256(raw).hexdigest()
        manifest["status"] = "complete"
        for name, command, limit in (
                ("gpu", ["nvidia-smi", "--query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem", "--format=csv,noheader"], 15),
                ("nvcc", ["nvcc", "--version"], 15),
                ("build", ["nvcc", "-O3", "-std=c++17", "-arch=sm_90", "-DA800_REDUCTION_CLOCK64",
                    "-Xptxas=-v", "block_reduce_small_cta.cu", "-o", "probe"], 90),
                ("resources", ["cuobjdump", "--dump-resource-usage", "probe"], 15),
                ("sass", ["cuobjdump", "--dump-sass", "probe"], 15)):
            code, stdout = run(command, name, limit)
            if code != 0 or name == "gpu" and b"H200" not in stdout:
                manifest["status"] = name + "_failed"
                break
        if manifest["status"] == "complete":
            manifest["binary_sha256"] = hashlib.sha256((work / "probe").read_bytes()).hexdigest()
            for process in (0, 1):
                for hidden, threads, grid in (CELLS if process == 0 else tuple(reversed(CELLS))):
                    label = f"p{process}_h{hidden}_b{threads}_g{grid}"
                    code, stdout = run([str(work / "probe"), str(threads), str(grid), str(SAMPLES), str(hidden)], label, 20)
                    if code != 0:
                        manifest["status"] = label + "_failed"
                        break
                if manifest["status"] != "complete":
                    break
                print(json.dumps({"completed_process_rotation": process}), flush=True)
        run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.mem,utilization.gpu", "--format=csv,noheader"], "gpu_end", 15)
        manifest["gpu_function_wall_s"] = time.monotonic() - started
        (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        files = {path.name: path.read_bytes() for path in work.iterdir() if path.is_file() and path.name != "probe"}
        return {"status": manifest["status"], "files": files,
            "file_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}}
    finally:
        shutil.rmtree(work)


@APP.local_entrypoint()
def main(output_root: str, source_commit: str):
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"app_id": APP.app_id, "output": str(out)}), flush=True)
    result = capture.remote()
    for name, raw in result["files"].items():
        if Path(name).name != name or hashlib.sha256(raw).hexdigest() != result["file_sha256"][name]:
            raise ValueError("capture file identity changed")
        (out / name).write_bytes(raw)
    receipt = {"status": result["status"], "app_id": APP.app_id, "source_commit": source_commit,
        "selection": SELECTION, "file_sha256": result["file_sha256"], "remote_temporary_directory_removed": True}
    (out / "client_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "app_id": APP.app_id}), flush=True)
    if result["status"] != "complete":
        raise RuntimeError("bounded capture failed; original outputs retained")
