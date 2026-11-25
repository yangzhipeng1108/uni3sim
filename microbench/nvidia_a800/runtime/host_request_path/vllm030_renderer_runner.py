"""Run the existing0.30 standalone renderer probe on the local CPU SDK.

The H200 probe implementation is unchanged; this wrapper supplies the same
guarded, offline CPU environment already used by the native Scheduler SDK.
It does not run Modal or initialize a GPU/model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--input-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = args.source_root.resolve(strict=True)
if args.output.exists():
    raise FileExistsError(args.output)
os.environ.update(CUDA_VISIBLE_DEVICES="", VLLM_TARGET_DEVICE="cpu", VLLM_PLUGINS="",
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
    VLLM_NO_USAGE_STATS="1", VLLM_CACHE_ROOT=str(args.output.parent.resolve() / "cache"),
    PYTHONPATH=str(source) + os.pathsep + str(ROOT))
sys.path[:0] = (str(ROOT), str(source))
from src.serving.backend.vllm_scheduler_process import (
    _install_cpu_device_guard, _verify_no_accelerator_access)
baseline = _install_cpu_device_guard()
from microbench.nvidia_h200.runtime.host_request_path import vllm030_render_service as probe
sys.argv = [probe.__file__, "--input-dir", str(args.input_dir), "--output", str(args.output),
    "--samples", "128", "--warmup", "16", "--local-source-preflight"]
probe.main()
_verify_no_accelerator_access(baseline)
receipt = {"kind": "local_cpu_wrapper_of_existing_renderer_probe",
    "cpu_device_guard_passed": True, "production_or_evaluation_duration_consumed": False,
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "probe_path": str(Path(probe.__file__).relative_to(ROOT)),
    "probe_sha256": hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest(),
    "upstream_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=source, text=True).strip(),
    "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in (
        "vllm/renderers/base.py", "vllm/renderers/hf.py", "vllm/renderers/inputs/preprocess.py",
        "vllm/entrypoints/openai/completion/protocol.py")},
    "raw_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
(args.output.parent / "guard_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
