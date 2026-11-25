#!/usr/bin/env python3
"""Observe one synthetic sampler helper's discrete dispatch, never a rate.

Execute under mcTracer after an owned-device preflight. Only the final two
explicit device synchronizations delimit the selected helper. Profiler raw
may include durations; this runner has no timing API or Forward rate output.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re


ENVIRONMENT_KEYS = (
    "PYTORCH_DISABLE_UNROLL_FLOAT_OPT", "PYTORCH_DISABLE_REDUCE_CONTINUOUS",
    "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_NO_CUDA_MEMORY_CACHING",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--operation", choices=("cast", "argmax"), required=True)
    parser.add_argument("--elements", type=int, required=True)
    parser.add_argument("--expected-torch-version", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device_id < 0 or not 1 <= args.elements <= 2**20:
        parser.error("device must be nonnegative; synthetic elements in [1,2**20]")
    if re.fullmatch(r"[0-9a-f]{40}", args.baseline) is None:
        parser.error("baseline must be an exact commit")
    if args.output.exists():
        raise FileExistsError(args.output)
    if importlib.metadata.version("torch") != args.expected_torch_version:
        raise RuntimeError("installed Torch differs from the frozen target")
    initial_environment = {
        key: os.environ[key] if key in os.environ else None
        for key in ENVIRONMENT_KEYS
    }
    os.environ["MACA_VISIBLE_DEVICES"] = str(args.device_id)
    import torch

    if "metax" not in str(torch.__version__).lower():
        raise RuntimeError("this observation requires the MetaX Torch runtime")
    torch.cuda.set_device(0)
    values = [float(index % 97 - 48) for index in range(args.elements)]
    source = torch.tensor(values, dtype=(
        torch.bfloat16 if args.operation == "cast" else torch.float32),
        device="cuda:0")
    if args.operation == "cast":
        warm = source.float()
        actual = warm.cpu().tolist()
        if actual != values:
            raise RuntimeError("synthetic BF16 to FP32 values differ")
    else:
        source = source.view(1, args.elements)
        warm = source.argmax(dim=-1)
        if warm.cpu().tolist() != [values.index(max(values))]:
            raise RuntimeError("synthetic argmax/tie result differs")
    # Source preparation, lazy dispatch setup and validation are outside the
    # marker. Retain warm to avoid making the target an allocator reuse test.
    torch.cuda.synchronize()
    destination = source.float() if args.operation == "cast" else source.argmax(dim=-1)
    torch.cuda.synchronize()
    actual = destination.cpu().tolist()
    expected = values if args.operation == "cast" else [values.index(max(values))]
    if actual != expected:
        raise RuntimeError("marker helper output differs from its CPU reference")
    result = {
        "scope": "synthetic_helper_discrete_dispatch_only",
        "baseline": args.baseline,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__),
        "hardware_id": "metax_c500", "device_id": args.device_id,
        "operation": args.operation, "input_domain": "finite_repeated_integer_values",
        "elements": args.elements, "environment_before_runtime_import": initial_environment,
        "allocator_backend": torch.cuda.memory.get_allocator_backend(),
        "input": {"dtype": str(source.dtype), "shape": list(source.shape),
            "strides": list(source.stride()), "contiguous": source.is_contiguous(),
            "data_ptr": source.data_ptr(), "mod4": source.data_ptr() % 4,
            "mod16": source.data_ptr() % 16, "mod256": source.data_ptr() % 256},
        "output": {"dtype": str(destination.dtype), "shape": list(destination.shape),
            "strides": list(destination.stride()), "contiguous": destination.is_contiguous(),
            "data_ptr": destination.data_ptr(), "mod4": destination.data_ptr() % 4,
            "mod16": destination.data_ptr() % 16, "mod256": destination.data_ptr() % 256},
        "warmup_validated": True, "marker_output_validated": True,
        "duration_consumed": False,
        "whole_operator_rate_eligible": False, "actual_serving_dispatch_verified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
