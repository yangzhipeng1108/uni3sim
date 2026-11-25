#!/usr/bin/env python3
"""Evaluate unpublished, duration-free launch facts through the ordinary CLI.

No canonical data is changed. Linux fork workers inherit an immutable mapping
from physical launch inputs to source programs; predictions use public predict.
"""
import argparse
import collections
import dataclasses
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.nvidia_h200_nvjet_raster import (
    derive_nvjet_raster_programs, load_nvjet_parameter_records,
    source_bound_nvjet_entries,
)
from evaluation.kernel import evaluation_runner as runner
from src.kernel.kernel_spec import KernelKind
from src.kernel.launch_config.tiling_spec import parse_nvjet_gemm_symbol

_PROGRAMS = {}
_ORIGINAL_PREDICT = runner.predict_production


def _physical_key(dtype, m, n, k, tile, depth, grid, block, cm, cn):
    return (dtype, m, n, k, tuple(tile), depth, tuple(grid), tuple(block), cm, cn)


def _predict_candidate(kernel, hardware, **kwargs):
    if (kernel.kind is KernelKind.GEMM and kernel.variant == "gemm"
            and kernel.launch.tiling is not None
            and kernel.launch.tiling.split_k == 1
            and "cluster_output_m_tiles" in kernel.launch.parameters):
        tiling = kernel.launch.tiling
        key = _physical_key(
            kernel.dtype, kernel.shape["b"] * kernel.shape["m"],
            kernel.shape["n"], kernel.shape["k"],
            tuple(tiling.tile_shape[axis] for axis in ("m", "n", "k")),
            tiling.pipeline_depth, kernel.launch.grid, kernel.launch.block,
            int(kernel.launch.parameters["cluster_output_m_tiles"]),
            int(kernel.launch.parameters["cluster_output_n_tiles"]))
        if key in _PROGRAMS:
            program = _PROGRAMS[key]
            if kernel.launch.source_program is not None and kernel.launch.source_program != program:
                raise ValueError("candidate conflicts with an existing source program")
            kernel = dataclasses.replace(kernel, launch=dataclasses.replace(
                kernel.launch, source_program=program))
    return _ORIGINAL_PREDICT(kernel, hardware, **kwargs)


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--parameter-campaign", type=Path, required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--semantic-output", type=Path, required=True)
    parser.add_argument("--binding-summary", type=Path, required=True)
    parser.add_argument("--hardware-id", choices=("nvidia_h200",), required=True)
    arguments, remaining = parser.parse_known_args()
    if multiprocessing.get_start_method() != "fork":
        raise RuntimeError("candidate CLI requires the existing Linux fork executor")
    expected = json.loads(arguments.dataset.read_text())
    records, origin = load_nvjet_parameter_records(arguments.parameter_campaign)
    entries = source_bound_nvjet_entries(arguments.source_review)
    programs, status = derive_nvjet_raster_programs(records, origin, expected, entries)
    indexed = {r["case_id"]: r for r in records}
    for case, program in programs.items():
        record = indexed[case]
        symbol = parse_nvjet_gemm_symbol(record["kernel"])
        if symbol is None:
            raise ValueError("candidate source symbol missing")
        attr = expected[case]["op_info"]["Attr"]
        key = _physical_key(attr["dtype"], attr["B"] * attr["M"], attr["N"], attr["K"],
            (symbol.tile_m, symbol.tile_n, symbol.tile_k), symbol.pipeline_depth,
            record["grid"], record["block"], symbol.cluster_m_tiles, symbol.cluster_n_tiles)
        if key in _PROGRAMS and _PROGRAMS[key] != program:
            raise ValueError("one physical key has conflicting raster candidates")
        _PROGRAMS[key] = program
    prepared = time.perf_counter()
    runner.predict_production = _predict_candidate
    sys.argv = [sys.argv[0], "--dataset", str(arguments.dataset),
        "--semantic-output", str(arguments.semantic_output),
        "--hardware-id", arguments.hardware_id, *remaining]
    result = runner.main()
    evaluated = time.perf_counter()
    if result not in (None, 0):
        return result
    report = json.loads(arguments.semantic_output.read_text())
    if report["hardware_id"] != "nvidia_h200":
        raise ValueError("the candidate observations belong to H200")
    applied = {row["case_id"] for row in report["cases"]
        if "cluster_grouped_source_raster_macro_tile_cache_approximation"
        in row["kernel_results"][0]["forward_limitations"]}
    if applied != set(programs):
        raise ValueError("not every verified raster candidate reached public predict")
    report["scope"] = "Fixed221 evaluation with unpublished discrete raster candidates; canonical untouched"
    arguments.semantic_output.write_text(json.dumps(report, indent=2) + "\n")
    arguments.binding_summary.write_text(json.dumps({
        "scope": "Candidate launch observations only; no duration or error selects a program",
        "parameter_campaign": str(arguments.parameter_campaign),
        "source_review": str(arguments.source_review),
        "constant_bank_parameter_origin": hex(origin),
        "bound_cases": sorted(programs), "physical_key_count": len(_PROGRAMS),
        "status_counts": dict(collections.Counter(status.values())), "status": status,
        "programs": {case: dataclasses.asdict(program) for case, program in programs.items()},
        "candidate_preparation_wall_s": prepared - started,
        "evaluation_cli_wall_s": evaluated - prepared,
        "total_command_wall_s": time.perf_counter() - started,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
