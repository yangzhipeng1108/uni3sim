#!/usr/bin/env python3
"""Report GLM residency, V4 parameters or Kimi active outer parameters."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.simulation.model_operator_probe import write_model_weight_residency_diagnostic


def layout(value):
    fields = value.split(":")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("layout must be TP:PP")
    try:
        tp, pp = (int(field) for field in fields)
    except ValueError as error:
        raise argparse.ArgumentTypeError("layout must contain two integers") from error
    if min(tp, pp) < 1:
        raise argparse.ArgumentTypeError("layout extents must be positive")
    return tp, pp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layout", type=layout, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vision-tensor-parallel-mode", choices=("weights", "data"),
                        help="explicit Kimi enabled-tower source scenario; no loader/runtime/capacity acceptance")
    parser.add_argument("--model-execution-version", choices=("vllm_0_30_0",),
                        help="explicit GLM v0.30 source parameters; FP8 loader payload excludes normal residency")
    parser.add_argument("--mtp-mode", choices=("off", "native"),
                        help="GLM target-off or native parameter inventory; runtime acceptance remains unbound")
    parser.add_argument("--source-max-batched-tokens", type=int,
                        help="explicit GLM-5.3 eager constructor-buffer scenario, not observed limits")
    parser.add_argument("--source-max-requests", type=int,
                        help="paired explicit GLM-5.3 buffer scenario; no reserve or normal capacity acceptance")
    parser.add_argument("--source-fp8-backend", choices=("deep_gemm_sm90",),
                        help="explicit GLM-5.3 SM90 DeepGEMM post-load storage profile; not observed provider selection")
    parser.add_argument("--source-moe-workspace-tokens", type=int,
                        help="optional explicit EP-off unchunked MoE workspace request at M>=128; requires SM90 DeepGEMM profile; not total peak")
    args = parser.parse_args()
    write_model_weight_residency_diagnostic(args.model, args.layout, args.output,
        vision_tensor_parallel_mode=args.vision_tensor_parallel_mode,
        model_execution_version=args.model_execution_version, mtp_mode=args.mtp_mode,
        source_fp8_backend=args.source_fp8_backend,
        source_moe_workspace_tokens=args.source_moe_workspace_tokens,
        constructor_buffer_limits=(None if args.source_max_batched_tokens is None and args.source_max_requests is None
                                   else (args.source_max_batched_tokens, args.source_max_requests)))
    print(args.output)


if __name__ == "__main__":
    main()
