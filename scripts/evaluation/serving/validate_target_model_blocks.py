"""Run source-pinned tiny GLM-5.2 and Kimi-K2.6 CUDA block validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--transformers-root", type=Path, required=True)
    parser.add_argument("--glm-root", type=Path, required=True)
    parser.add_argument("--kimi-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from evaluation.serving.simulation.target_model_block_validation import (
        run_target_model_block_validation,
    )

    result = run_target_model_block_validation(
        vllm_root=args.vllm_root,
        transformers_root=args.transformers_root,
        glm_root=args.glm_root,
        kimi_root=args.kimi_root,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
