#!/usr/bin/env python3
"""Capture the device launch graph of the public ATB RoPE entry.

This probe records launch identities, shapes, and block dimensions only.  Its
output is launch evidence and must not be used as a Forward duration source.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch_npu


SCHEMA_VERSION = 1
HARDWARE_ID = "ascend_910b"
SOFTWARE_REVISION = "CANN 8.5.1"
DEFAULT_TOKENS = 64
DEFAULT_CACHE_ROWS = None
QUERY_HEADS = 32
KV_HEADS = 8
DEFAULT_HEAD_DIM = 128
WARMUPS = 3
CAPTURE_DIRECTORY_ENVIRONMENT = "MIDLAYER_ASCEND_GATHER_V3_CAPTURE_DIR"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--tiling-capture-dir", type=Path)
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--cache-rows", type=int, default=DEFAULT_CACHE_ROWS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    args = parser.parse_args()
    if args.device_id < 0:
        raise ValueError("device id must be non-negative")
    if args.tokens <= 0:
        raise ValueError("tokens must be positive")
    cache_rows = args.tokens if args.cache_rows is None else args.cache_rows
    if cache_rows <= 0:
        raise ValueError("cache rows must be positive")
    if cache_rows < args.tokens:
        raise ValueError("cache rows must cover every generated position")
    if args.head_dim <= 0 or args.head_dim % 2 != 0:
        raise ValueError("head dim must be a positive even integer")
    if (len(args.baseline_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in args.baseline_commit)):
        raise ValueError("baseline commit must be a lowercase SHA-1")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir() or any(output_dir.iterdir()):
        raise ValueError("output directory must exist and be empty")

    torch.npu.set_device(args.device_id)
    tokens = args.tokens
    head_dim = args.head_dim
    dtype = torch.bfloat16
    query = torch.randn(
        tokens, QUERY_HEADS * head_dim, dtype=dtype, device="npu")
    key = torch.randn(
        tokens, KV_HEADS * head_dim, dtype=dtype, device="npu")
    positions = torch.arange(tokens, dtype=torch.int32, device="npu")
    theta = 1.0 / (10000 ** (
        torch.arange(0, head_dim, 2, device="npu", dtype=torch.float32)
        / head_dim))
    frequencies = (
        torch.arange(cache_rows, device="npu", dtype=torch.float32)
        .unsqueeze(1) * theta).float()
    cos_sin_cache = torch.cat(
        [frequencies.cos(), frequencies.sin()], dim=-1).to(dtype)

    def operation():
        return torch_npu._npu_rotary_embedding(
            positions, query, key, head_dim, cos_sin_cache, True)

    for _ in range(WARMUPS):
        operation()
    torch.npu.synchronize()

    if args.tiling_capture_dir is not None:
        tiling_capture_dir = args.tiling_capture_dir.resolve()
        if (not tiling_capture_dir.is_dir()
                or any(tiling_capture_dir.iterdir())):
            raise ValueError(
                "tiling capture directory must exist and be empty")
        os.environ[CAPTURE_DIRECTORY_ENVIRONMENT] = str(tiling_capture_dir)
        operation()
        torch.npu.synchronize()
        del os.environ[CAPTURE_DIRECTORY_ENVIRONMENT]

    trace_dir = output_dir / "trace"
    handler = torch_npu.profiler.tensorboard_trace_handler(
        str(trace_dir), analyse_flag=True, async_mode=False)
    experimental = torch_npu.profiler._ExperimentalConfig(
        profiler_level="Level0", aic_metrics="ACL_AICORE_NONE",
        l2_cache=False, data_simplification=True, record_op_args=False)
    with torch_npu.profiler.profile(
            activities=(
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU),
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=handler,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental) as profiler:
        operation()
        torch.npu.synchronize()
        profiler.step()

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "owner": "kernel_accuracy_vector",
        "software_revision": SOFTWARE_REVISION,
        "baseline_commit": args.baseline_commit,
        "device_id": args.device_id,
        "process_pid": os.getpid(),
        "command": sys.argv,
        "source": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(__file__),
        },
        "workload": {
            "tokens": tokens,
            "cache_rows": cache_rows,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": head_dim,
            "dtype": "bf16",
        },
        "warmups_before_capture": WARMUPS,
        "captured_public_calls": 1,
        "tiling_capture_enabled": args.tiling_capture_dir is not None,
        "consumption_contract": (
            "only launch identities, shapes, and block dimensions may enter "
            "Forward; profiler durations are evaluation-only"),
        "production_or_evaluation_duration_consumed": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
