#!/usr/bin/env python3
"""Issue bounded BF16 AddRMSNorm calls for launch capture, without timing."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu


CAPTURE_DIRECTORY_ENVIRONMENT = "MIDLAYER_ASCEND_ADDRMS_CAPTURE_DIR"
EXPECTED_ROWS = (1, 64, 256, 1024)
EXPECTED_HIDDEN = 16384
REPETITIONS = 3


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    args = parser.parse_args(argv)
    if args.device_id < 0:
        raise ValueError("device id must be non-negative")
    output_root = args.output_root.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"output root is missing: {output_root}")
    torch.npu.set_device(args.device_id)
    weight = torch.empty(
        EXPECTED_HIDDEN, dtype=torch.bfloat16, device="npu")
    for rows in EXPECTED_ROWS:
        case_key = f"T{rows}_H{EXPECTED_HIDDEN}"
        case_directory = output_root / case_key
        case_directory.mkdir()
        os.environ[CAPTURE_DIRECTORY_ENVIRONMENT] = str(case_directory)
        x = torch.empty(
            rows, EXPECTED_HIDDEN, dtype=torch.bfloat16, device="npu")
        residual = torch.empty_like(x)
        outputs = []
        for _ in range(REPETITIONS):
            outputs.append(torch_npu.npu_add_rms_norm(
                x, residual, weight, 1.0e-6))
        torch.npu.synchronize()
        del outputs
        del residual
        del x
        print(json.dumps({
            "case_key": case_key,
            "dtype": "bf16",
            "hidden": EXPECTED_HIDDEN,
            "repetitions": REPETITIONS,
            "rows": rows,
            "state": "dry_launch_complete",
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
