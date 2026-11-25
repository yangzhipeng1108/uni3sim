"""Capture one cuBLASLt descriptor trace for two old eager prefill GEMMs.

This records launch and cuBLAS API facts only. Production kernel durations in
the report are evaluation evidence and must not set Forward parameters.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import modal

_REMOTE_MODULE = len(Path(__file__).resolve().parents) < 4
if _REMOTE_MODULE:
    IMAGE = modal.Image.debian_slim()
else:
    from scripts.kernel_dataset.nvidia_h200.ncu_coverage_modal_runner import IMAGE


APP = modal.App("midlayer-h200-gemm-descriptor-20261102")
GATE = "MIDLAYER_H200_PAID_PROBE_CONTROLLER"
_ROOT = (Path("/opt/midlayer") if _REMOTE_MODULE
         else Path(__file__).resolve().parents[4])
CAMPAIGN = (_ROOT /
            "profiler_output/microbench/nvidia_h200/"
            "cublas_descriptor_20261102")

WORKLOAD = r"""
import json
import torch

torch.set_grad_enabled(False)
shapes = ((768, 4096, 2560), (768, 9728, 2560))
for m, k, n in shapes:
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    layer = torch.nn.Linear(k, n, bias=False, device="cuda",
                            dtype=torch.bfloat16)
    for _ in range(3):
        layer(x)
    torch.cuda.synchronize()
    marker = f"midlayer-gemm-descriptor:M{m}:K{k}:N{n}"
    torch.cuda.nvtx.range_push(marker)
    layer(x)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print(json.dumps({"marker": marker, "torch": str(torch.__version__),
                      "gpu": torch.cuda.get_device_name()}), flush=True)
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@APP.function(image=IMAGE, gpu="H200", timeout=600,
              max_containers=1, single_use_containers=True)
def capture():
    import zstandard

    sys.path.insert(0, "/opt/midlayer")
    from scripts.kernel_dataset.nvidia_h200.ncu_coverage_modal_runner import (
        _check_identity,
    )

    identity = _check_identity()
    with tempfile.TemporaryDirectory(prefix="h200-gemm-descriptor-") as temporary:
        root = Path(temporary)
        report = root / "descriptor.nsys-rep"
        command = ["nsys", "profile", "--trace=cuda,nvtx,cublas-verbose",
                   "--sample=none", "--cpuctxsw=none", "--force-overwrite=true",
                   "-o", str(root / "descriptor"), sys.executable, "-c", WORKLOAD]
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=480)
        if result.returncode != 0 or not report.is_file():
            raise RuntimeError(json.dumps({
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-3000:],
                "stderr_tail": result.stderr[-3000:],
            }))
        compressed = root / "descriptor.nsys-rep.zst"
        with report.open("rb") as source, compressed.open("wb") as target:
            zstandard.ZstdCompressor(level=3).copy_stream(source, target)
        metadata = {
            "schema_version": 1,
            "kind": "h200_cublas_descriptor_probe_raw",
            "identity": identity,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "workload_sha256": hashlib.sha256(WORKLOAD.encode()).hexdigest(),
            "shapes_mkn": [[768, 4096, 2560], [768, 9728, 2560]],
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "files": [{"path": compressed.name, "size": compressed.stat().st_size,
                       "sha256": _sha(compressed)}],
        }
        yield {"kind": "metadata", "value": metadata}
        yield {"kind": "file_start", **metadata["files"][0]}
        with compressed.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                yield {"kind": "file_chunk", "data": chunk}
        yield {"kind": "file_end"}


@APP.local_entrypoint()
def main(output_dir: str) -> None:
    if os.environ[GATE] != "1":
        raise PermissionError("paid descriptor probe requires spend controller")
    target = Path(output_dir).resolve()
    if (target.parent != CAMPAIGN.resolve()
            or re.fullmatch(r"attempt_[A-Za-z0-9_]+", target.name) is None
            or target.exists()):
        raise ValueError("output must be a new registered descriptor attempt")
    target.mkdir(parents=True, exist_ok=False)
    pending = None
    expected = None
    metadata = None
    for event in capture.remote_gen():
        kind = event["kind"]
        if kind == "metadata":
            if metadata is not None:
                raise ValueError("duplicate descriptor metadata")
            metadata = event["value"]
        elif kind == "file_start":
            if event["path"] != "descriptor.nsys-rep.zst" or pending is not None:
                raise ValueError("unexpected descriptor evidence path")
            expected = event
            pending = (target / event["path"]).open("xb")
        elif kind == "file_chunk":
            if pending is None:
                raise ValueError("descriptor evidence chunk without start")
            pending.write(event["data"])
        elif kind == "file_end":
            if pending is None or expected is None:
                raise ValueError("descriptor evidence end without start")
            pending.close()
            pending = None
            path = target / expected["path"]
            if path.stat().st_size != expected["size"] or _sha(path) != expected["sha256"]:
                raise ValueError("descriptor evidence transfer differs")
        else:
            raise ValueError(f"unknown descriptor evidence event: {kind}")
    if pending is not None or metadata is None or expected is None:
        raise ValueError("descriptor evidence transfer incomplete")
    (target / "raw.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")
