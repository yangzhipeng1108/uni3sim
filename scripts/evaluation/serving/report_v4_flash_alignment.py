#!/usr/bin/env python3
"""Generate independent Flash source expectations and compare probe artifacts."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.simulation.v4_flash_alignment import write_flash_alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--candidate", action="append", nargs=2, required=True,
                        metavar=("PREFILL_TOKENS", "PROBE_JSON"))
    parser.add_argument("--case-input", action="append", nargs=2, default=[],
                        metavar=("PREFILL_TOKENS", "TOKEN_IDS_JSON"),
                        help="independent synthetic embedding inputs; missing declarations remain unchecked")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--torch-include", type=Path,
                        help="genuine Torch 2.11 include directory for static GreaterOrNan source binding; never imports Torch")
    parser.add_argument("--flashinfer-source-root", type=Path,
                        help="genuine installed FlashInfer package root; reads build metadata/C++ only, never loads a provider")
    parser.add_argument("--require-aligned", action="store_true")
    parser.add_argument("--compress-artifacts", action="store_true",
                        help="write complete reference/comparison as compact lossless .json.gz; keep all checks")
    parser.add_argument("--v4-source-max-model-len", type=int,
                        help="independent source metadata constructor cap, separate from HF context limit")
    parser.add_argument("--v4-source-use-deep-gemm", action=argparse.BooleanOptionalAction, default=None,
                        help="declared process-lifetime VLLM_USE_DEEP_GEMM before cached dispatch; not observed provider support")
    args = parser.parse_args()
    result = write_flash_alignment(args.model, args.source_root,
        [(int(tokens), Path(path)) for tokens, path in args.candidate], args.output,
        torch_include=args.torch_include, flashinfer_source_root=args.flashinfer_source_root,
        case_inputs=[(int(tokens), Path(path)) for tokens, path in args.case_input],
        v4_source_max_model_len=args.v4_source_max_model_len,
        v4_source_use_deep_gemm=args.v4_source_use_deep_gemm,
        compress_artifacts=args.compress_artifacts)
    print(result["summary"])
    print(args.output / "report.md")
    if args.require_aligned and not result["alignment_passed"]:
        raise SystemExit("Flash independent alignment remains incomplete")


if __name__ == "__main__":
    main()
