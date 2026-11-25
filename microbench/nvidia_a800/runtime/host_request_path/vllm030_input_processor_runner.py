"""Independent CPU-only vLLM0.30 pretokenized InputProcessor service.

No tokenizer, model weights, Scheduler, tensor execution or GPU is involved.
This boundary does not include HTTP, text rendering, IPC or output delivery.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--model-config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
source = args.source_root.resolve(strict=True)
os.environ.update(CUDA_VISIBLE_DEVICES="", VLLM_TARGET_DEVICE="cpu", VLLM_PLUGINS="",
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
    VLLM_NO_USAGE_STATS="1", VLLM_CACHE_ROOT=str(args.output.parent.resolve() / "cache"),
    PYTHONPATH=str(source) + os.pathsep + str(ROOT))
sys.path[:0] = (str(ROOT), str(source))
from src.serving.backend.vllm_scheduler_process import (
    _install_cpu_device_guard, _verify_no_accelerator_access)

baseline = _install_cpu_device_guard()
import torch
import vllm
from vllm.config import VllmConfig, ModelConfig, DeviceConfig, SchedulerConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor

args.output.parent.mkdir(parents=True, exist_ok=True)
inputs = args.output.parent / "source_inputs"
inputs.mkdir(exist_ok=False)
(inputs / "config.json").write_bytes(args.model_config.read_bytes())
config = VllmConfig(model_config=ModelConfig(model=str(inputs), tokenizer=str(inputs),
    skip_tokenizer_init=True, dtype="bfloat16", max_model_len=4096, enforce_eager=True, runner="generate"),
    device_config=DeviceConfig(device="cpu"), scheduler_config=SchedulerConfig(
        max_model_len=4096, is_encoder_decoder=False, max_num_batched_tokens=4096,
        max_num_seqs=32, async_scheduling=False))
processor = InputProcessor(config)
params = SamplingParams(temperature=0, max_tokens=8)
rows = []
for count in (32, 128, 729, 2048):
    ids = [17] * count
    prompt = {"type": "token", "prompt_token_ids": ids}
    samples = []
    for index in range(144):
        request_id = f"independent-{count}-{index}"
        started = time.perf_counter_ns()
        request = processor.process_inputs(request_id, prompt, params, ("generate",), arrival_time=0.0)
        elapsed = time.perf_counter_ns() - started
        if request.prompt_token_ids != ids or request.request_id != request_id:
            raise RuntimeError("InputProcessor changed the synthetic request")
        if index >= 16:
            samples.append(elapsed)
    rows.append({"prompt_tokens": count, "samples_ns": samples,
        "median_ns": statistics.median(samples), "max_ns": max(samples)})
_verify_no_accelerator_access(baseline)
paths = ("vllm/v1/engine/input_processor.py", "vllm/renderers/base.py", "vllm/sampling_params.py")
payload = {"kind": "independent_cpu_vllm030_pretokenized_input_processor",
    "production_or_evaluation_duration_consumed": False, "model_weights_loaded": False,
    "cuda_initialized": torch.cuda.is_initialized(), "software": {"vllm": vllm.__version__,
        "torch": torch.__version__, "python": platform.python_version()},
    "source_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=source, text=True).strip(),
    "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in paths},
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "model_config_sha256": hashlib.sha256(args.model_config.read_bytes()).hexdigest(),
    "cpuinfo_first_processor": Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0],
    "affinity": sorted(os.sched_getaffinity(0)), "rows": rows,
    "boundary": "InputProcessor.process_inputs on already-rendered token IDs",
    "limitations": ["skip_tokenizer_init=True; no text/tokenizer or rendering service",
        "CPU platform; input validation differs from a real GPU platform",
        "no API, IPC, HTTP client or output delivery; not a full ingress bound"]}
args.output.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"status": "complete", "rows": [
    {key: value for key, value in row.items() if key != "samples_ns"} for row in rows]}))
