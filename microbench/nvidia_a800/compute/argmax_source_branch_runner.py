#!/usr/bin/env python3
"""Offline check of the pinned SM80 ArgMax's value-dependent branch.

Reads decoded static evidence, checks its exact branch def/use and exercises
synthetic scalar comparator states on CPU. No GPU, model tensor or duration.
This does not price ArgMax, infer its dynamic merge count or approve provider.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import re


def check_program(program):
    if program["arch"] != "sm_80":
        raise ValueError("requires independently decoded SM80 entry")
    resources = program["resources"]
    if (resources["registers_per_thread"], resources["static_shared_bytes"],
            resources["stack_bytes"], resources["local_bytes"]) != (32, 16, 0, 0):
        raise ValueError("resource evidence differs from retained exact SM80 program")
    instructions = {}
    for line in program["code"].splitlines():
        match = re.search(r"/\*([0-9a-f]+)\*/", line)
        if match is not None:
            instructions[int(match.group(1), 16)] = line.split("*/", 1)[1].split("/*", 1)[0].strip()
    if not instructions[0x13e0].startswith("FSETP.GTU.AND P1, PT, |R8|, +INF"):
        raise ValueError("branch predicate is not retained ordered NaN classification")
    if instructions[0x1430] != "@P1 BRA 0x14d0 ;":
        raise ValueError("NaN-dependent branch was changed")
    if instructions[0x14c0] != "BRA 0x1500 ;" or instructions[0x1500] != "BSYNC B2 ;":
        raise ValueError("branch join changed")
    numeric = [instructions[a] for a in range(0x1440, 0x14d0, 0x10)]
    nan_path = [instructions[a] for a in range(0x14d0, 0x1500, 0x10)]
    if len(numeric) == len(nan_path):
        raise ValueError("counterexample no longer demonstrates distinct instruction paths")
    # Small scalar source states, independent of checkpoint and model execution.
    cases = []
    for label, a, b in (("finite_pair", 1.0, 0.0), ("nan_accumulator", math.nan, 0.0)):
        a_nan = math.isnan(a)
        choose_a = (True if a_nan and not math.isnan(b) else 3 < 7 if a_nan or a == b else a > b)
        cases.append({"synthetic_state": label, "accumulator_is_nan": a_nan,
                      "branch_taken_at_0x1430": a_nan,
                      "selected_path_static_instructions": len(nan_path) if a_nan else len(numeric),
                      "source_selects_accumulator": choose_a,
                      "checkpoint_or_GPU_observation": False})
    return {"schema": "sm80_argmax_source_branch_check_v1", "passed": True,
            "GPU_executed": False, "duration_consumed": False,
            "resources": resources, "numeric_path": numeric, "nan_path": nan_path,
            "synthetic_counterexamples": cases,
            "conclusion": "Same-shape legal scalar states reach different retained instruction paths; no source proof of equal physical cost or universal non-NaN domain",
            "not_proven": ["hardware duration difference", "complete dynamic instruction counts", "complete sampler or provider", "actual checkpoint NaNs"],
            "minimal_missing_dynamic_fact": "For each actual warp/merge invocation, NaN branch execution/active masks; or an independently proved source/checkpoint input invariant that selects a single applicable path. Additional typed service units still required.",
            "current_caller_contract": "SymbolicGreedy accepts version, sample_logits, copy_dispatch; argmax KernelSpec accepts sampling_source/stage and optional hardware source properties only; lowering does not transport logits or branch-domain facts",
            "point_prediction_bound": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    with gzip.open(args.program, "rt") as stream:
        result = check_program(json.load(stream))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "GPU_executed": False,
                      "numeric_path_instructions": len(result["numeric_path"]),
                      "nan_path_instructions": len(result["nan_path"])}))


if __name__ == "__main__":
    main()
