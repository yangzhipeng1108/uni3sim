#!/usr/bin/env python3
"""Report exact model/native-mode identity through the real vLLM 0.30 SDK."""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    sys.path.insert(0, str(args.source))
    from evaluation.serving.simulation.model_operator_probe import write_vllm030_model_identity_probe

    write_vllm030_model_identity_probe(args.model, args.source, args.output, draft_path=args.draft)
    print(args.output)


if __name__ == "__main__":
    main()
