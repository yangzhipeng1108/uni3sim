"""Measure vLLM 0.30 completion rendering without a model or GPU execution.

This is an independent Host mechanism probe.  It exercises the tagged
``Renderer.render_cmpl_async`` source path on synthetic Qwen3 text and token
IDs, and records every call rather than fitting a Serving duration.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(directory: Path) -> dict:
    manifest_path = directory / "inputs_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["kind"] != "public_qwen3_4b_tokenizer_and_hf_config_for_host_only_probe"
            or manifest["repository"] != "Qwen/Qwen3-4B"
            or manifest["no_model_weights"] is not True):
        raise ValueError("request-path input manifest differs")
    expected = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if {item["path"] for item in manifest["files"]} != expected:
        raise ValueError("request-path input file set differs")
    for item in manifest["files"]:
        path = directory / item["path"]
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"request-path input size differs: {path}")
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"request-path input SHA differs: {path}")
    return manifest


def _cpu_platform():
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before Host-only rendering")
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    import vllm.platforms as platforms
    from vllm.platforms.cpu import CpuPlatform

    platforms._current_platform = CpuPlatform()
    if not platforms.current_platform.is_cpu():
        raise RuntimeError("vLLM CPU platform selection failed")
    return torch


def _summary(samples: list[int]) -> dict:
    if len(samples) < 20 or any(value <= 0 for value in samples):
        raise ValueError("renderer samples are incomplete")
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "median_ns": int(statistics.median(ordered)),
        "p10_ns": ordered[(len(ordered) - 1) // 10],
        "p90_ns": ordered[(len(ordered) - 1) * 9 // 10],
        "samples_ns": samples,
    }


def _host_identity() -> dict:
    cpuinfo = Path("/proc/cpuinfo")
    first = cpuinfo.read_text().split("\n\n", 1)[0] if cpuinfo.is_file() else ""
    return {
        "cpuinfo_first_processor": first,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "affinity": sorted(os.sched_getaffinity(0)),
    }


async def _measure(renderer, config, lengths: tuple[int, ...],
                   samples: int, warmup: int) -> list[dict]:
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.renderers.inputs.preprocess import parse_model_prompt

    rows = []
    for length in lengths:
        text = " hello" * length
        ids = renderer.get_tokenizer().encode(text, add_special_tokens=True)
        if len(ids) != length:
            raise ValueError(f"synthetic Qwen tokenizer length differs: {length}")
        request = CompletionRequest(
            model="host-only-probe", prompt=text, max_tokens=8, stream=True)
        tok_params = request.build_tok_params(config.model_config)
        prompts = {
            "text": parse_model_prompt(config.model_config, text),
            "token_ids": parse_model_prompt(config.model_config, ids),
        }
        readings = {"text": [], "token_ids": []}
        for index in range(warmup + samples):
            modes = ("text", "token_ids") if index % 2 == 0 else ("token_ids", "text")
            for mode in modes:
                start = time.perf_counter_ns()
                output = await renderer.render_cmpl_async(
                    [prompts[mode]], tok_params)
                elapsed = time.perf_counter_ns() - start
                if (len(output) != 1
                        or len(output[0]["prompt_token_ids"]) != length):
                    raise ValueError(f"vLLM renderer output differs: {length} {mode}")
                if index >= warmup:
                    readings[mode].append(elapsed)
        rows.append({"prompt_tokens": length,
                     "text_render": _summary(readings["text"]),
                     "token_id_render": _summary(readings["token_ids"])})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--local-source-preflight", action="store_true")
    args = parser.parse_args()
    if args.samples < 20 or args.warmup < 1 or args.output.exists():
        raise ValueError("invalid renderer sample count or output path")
    directory = args.input_dir.resolve()
    manifest = _inputs(directory)
    torch = _cpu_platform()
    import transformers
    import vllm
    from vllm.engine.arg_utils import EngineArgs
    from vllm.renderers.registry import renderer_from_config

    if (not args.local_source_preflight
            and vllm.__version__.split("+", 1)[0] != "0.30.0"):
        raise RuntimeError("request-path probe requires installed vLLM 0.30.0")
    config = EngineArgs(
        model=str(directory), tokenizer=str(directory), dtype="bfloat16",
        max_model_len=4096, skip_tokenizer_init=False, async_scheduling=False,
        enforce_eager=True, trust_remote_code=False,
    ).create_engine_config()
    renderer = renderer_from_config(config)
    if type(renderer).__name__ != "HfRenderer":
        raise RuntimeError("Qwen3 request path did not select HfRenderer")
    rows = asyncio.run(_measure(renderer, config, (32, 128, 729, 2048),
                                args.samples, args.warmup))
    if torch.cuda.is_initialized():
        raise RuntimeError("Host-only request path initialized CUDA")
    payload = {
        "schema_version": 1,
        "kind": "vllm030_host_only_completion_renderer_service",
        "measurement_boundary": "Renderer.render_cmpl_async synthetic text/token-id input to EngineInput",
        "production_or_evaluation_duration_consumed": False,
        "model_weights_loaded": False,
        "input_manifest_sha256": _sha256(directory / "inputs_manifest.json"),
        "input_revision": manifest["revision"],
        "host": _host_identity(),
        "software": {"vllm": vllm.__version__, "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "renderer": {"class": type(renderer).__name__,
                     "num_workers": config.model_config.renderer_num_workers},
        "sample_count_per_mode_and_length": args.samples,
        "warmup_per_mode_and_length": args.warmup,
        "local_source_preflight": args.local_source_preflight,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "output": str(args.output),
                      "rows": len(rows)}))


if __name__ == "__main__":
    main()
