#!/usr/bin/env python3
"""Check consecutive Flash source graphs on CPU-only virtual tensor metadata."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "source-root", "torch-include", "flashinfer-source-root", "sequence", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-batched-tokens", type=int, required=True)
    parser.add_argument("--use-deep-gemm", choices=("enabled", "disabled"), required=True)
    parser.add_argument("--source-workspace-initial-capacity-bytes", type=int, nargs=4)
    parser.add_argument("--indexer-selection-source", choices=("declared_causal_topk",))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    # Enforce the execution identity before loading any model adapters.
    for name in ("torch", "triton", "cupy", "pycuda", "cuda"):
        sys.modules[name] = None
    from evaluation.serving.metrics.v4_sequence import write_v4_source_sequence
    declaration = json.loads(args.sequence.read_text())
    options = {name: str(getattr(args, name)) for name in ("model", "source_root", "torch_include", "flashinfer_source_root")}
    options.update(max_model_len=args.max_model_len, max_batched_tokens=args.max_batched_tokens,
        use_deep_gemm=args.use_deep_gemm == "enabled", indexer_selection_source=args.indexer_selection_source,
        workspace_initial_capacity_bytes=(None if args.source_workspace_initial_capacity_bytes is None
            else tuple(args.source_workspace_initial_capacity_bytes)))
    result = write_v4_source_sequence(options, declaration["steps"], initial_prefixes=declaration["initial_prefixes"],
        output=args.output, workers=args.workers)
    print(args.output / "report.md")
    if not result["logical_sequence_passed"]:
        raise SystemExit("source sequence has unmatched scopes; inspect comparison.json")


if __name__ == "__main__":
    main()
