#!/usr/bin/env python3
"""Write the aligned ShareGPT GEMM workload universe as candidate data."""

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_pipeline.launch_data.gemm_workload_candidates import (  # noqa: E402
    build_gemm_workload_candidate,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/experiments/gemm_launch_domain.json"),
    )
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    owner = (ROOT / "profiler_output/launch_data").resolve()
    if owner not in output.parents:
        raise ValueError(f"--output must be below {owner}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate = build_gemm_workload_candidate(
        ROOT, ROOT / args.config, models_root=args.models_root)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps({
        "output": str(output),
        "semantic_keys": candidate["semantic_key_count"],
        "physical_operations": candidate["physical_operation_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
