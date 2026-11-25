#!/usr/bin/env python3
"""Project the H200 cuBLASLt rank-0 capture into a candidate launch LUT.

H200 counterpart of the A800 path
``scripts/launch_data/nvidia_a800_launch_generator.py`` ->
``dataset_pipeline/launch_data/nvidia_a800.py`` (``build`` + projection) ->
``canonical_projection.build_gemm_launch_selection_artifact``.  It writes a
*candidate* pair in the A800 schema v3 (13 selection-key columns, launch
columns tile_m/tile_n/tile_k/pipeline_depth/split_k/reduction and the four
CTA-raster columns) under ``profiler_output/launch_data/nvidia_h200/``.
Canonical publication under ``launch_data/`` is the main agent's job.

Inputs (produced by ``cublaslt_heuristic_runner.py``):
  * the domain JSON (``domain`` subcommand): the 140,765 selection keys of the
    A800 CSV and their 140,752 physical operations;
  * the compact rank-0 launch-fact file and candidate summary of the H200
    ``capture`` sweep (GPUs 0 and 3 on target host).

Projection rules (fail-loud; every rule states which fact it reads):
  * tile_m / tile_n: the rank-0 CUBLASLT_ALGO_CONFIG_TILE_ID mapped through
    cublasLt.h 12.9 (row-major M = second extent, N = first, as on A800);
  * tile_k: the K of the rank-0 CUBLASLT_ALGO_CONFIG_STAGES_ID.  On H200 every
    rank-0 algorithm reports ``64xAUTO`` (stage count chosen inside the
    kernel), so pipeline_depth is read from the launched nvjet kernel symbol
    ``nvjet_<kind>_<TN>x<TM>_<K>x<STAGES>_<CX>x<CY>_<v|h>_...``; the symbol's
    tile and K must equal the algorithm enum or the build fails;
  * split_k / reduction: rank-0 SPLITK_NUM / REDUCTION_SCHEME with the A800
    names (0 none, 1 in_place, 2 fp32_external, 4 output_dtype_external); a
    separate reduction must launch exactly one trailing splitKreduce kernel;
  * CTA raster: the A800 conventions are kernel-family facts
    (``ampere_*gemm*`` raster word, CUTLASS 2.x grouped grid).  Hopper nvjet
    kernels have neither: the grid is (cluster-rounded) output tiles times K
    slices, or a persistent grid of <= 132 CTAs, and their raster/cluster
    program has no verified parameter layout.  They are written with
    convention ``none``, log_group 0, serpentine false and k_slices =
    split_k (the K slices the launch folds into its grid), which is the A800
    schema's "no raster program" encoding.  The raw cluster shape and
    direction letter stay in the compact capture for later use.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kernel.launch_config.gemm_launch_selection import (  # noqa: E402
    GEMM_BOOLEAN_ENCODING, GEMM_LAUNCH_ARTIFACT_TYPE, GEMM_LAUNCH_COMPRESSION,
    GEMM_LAUNCH_CSV_FILENAME, GEMM_LAUNCH_MANIFEST_FILENAME,
    GEMM_NULL_ENCODING, compress_gemm_launch_csv, gemm_launch_csv_fields,
    gemm_launch_fields, gemm_launch_hardware_contract_sha256,
    gemm_launch_schema_version, load_gemm_launch_hardware_contract,
)
from src.kernel.launch_config.gemm_workload import (  # noqa: E402
    GEMM_LAUNCH_SELECTION_KEY_FIELDS, GEMM_SEMANTIC_CONTEXT_FIELDS,
)

HARDWARE_ID = "nvidia_h200"
SCHEMA_TEMPLATE_HARDWARE = "nvidia_a800"
SOFTWARE_REVISION = "cuda_12.9"
EXPECTED_CUBLASLT_VERSION = 120901
OUTPUT_DIR = ROOT / "profiler_output" / "launch_data" / HARDWARE_ID
GENERATED_BY = (
    "microbench/nvidia_h200/operators/gemm/"
    "build_gemm_launch_selection_candidate.py")
REDUCTION_NAMES = {
    0: "none", 1: "in_place", 2: "fp32_external", 4: "output_dtype_external",
}
NVJET_SYMBOL = re.compile(
    r"^nvjet_(?P<kind>[a-z]+)_(?P<a>\d+)x(?P<b>\d+)_(?P<k>\d+)x(?P<stages>\d+)"
    r"_(?P<cx>\d+)x(?P<cy>\d+)_(?P<raster>[vh])_(?P<suffix>\w+)$")
SM_COUNT = 132


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _project(operation_key: str, fact: dict) -> tuple[tuple, dict]:
    m, n, k, dtype, has_bias = json.loads(operation_key)
    rank0 = fact["rank0"]
    launches = fact["launches"]
    main = launches[0]
    symbol = main["symbol"]
    match = NVJET_SYMBOL.fullmatch(symbol)
    if match is None:
        raise NotImplementedError(
            f"{operation_key}: rank-0 kernel family {symbol!r} has no H200 "
            "projection rule")
    if rank0["tile"] != f"{match['a']}x{match['b']}":
        raise ValueError(f"{operation_key}: symbol tile differs from TILE_ID")
    stage_k, stage_depth = rank0["stages"].split("x")
    if stage_depth != "AUTO" or int(stage_k) != int(match["k"]):
        raise ValueError(f"{operation_key}: symbol K differs from STAGES_ID")
    tile_m, tile_n = rank0["logical_tile_m"], rank0["logical_tile_n"]
    tile_k = int(stage_k)
    pipeline_depth = int(match["stages"])
    split_k = rank0["split_k"]
    reduction_scheme = rank0["reduction_scheme"]
    if reduction_scheme not in REDUCTION_NAMES:
        raise ValueError(f"{operation_key}: unsupported reduction scheme")
    if (split_k == 1) != (reduction_scheme == 0):
        raise ValueError(f"{operation_key}: split/reduction contract differs")
    reducers = [launch for launch in launches
                if "splitKreduce_kernel" in launch["symbol"]]
    if reduction_scheme in (2, 4):
        if len(launches) != 2 or len(reducers) != 1 or launches[1] is not reducers[0]:
            raise ValueError(f"{operation_key}: separate reducer differs")
    elif len(launches) != 1 or reducers:
        raise ValueError(f"{operation_key}: direct launch count differs")
    if ("splitK" in match["suffix"]) != (split_k > 1):
        raise ValueError(f"{operation_key}: symbol split-K flag differs")
    # The symbol and CLUSTER_SHAPE_ID state the 2-D tile cluster; the launch
    # attribute flattens it onto cluster X.
    if (rank0["cluster_shape"] != f"{match['cx']}x{match['cy']}x1"
            or list(main["cluster"]) != [
                int(match["cx"]) * int(match["cy"]), 1, 1]):
        raise ValueError(f"{operation_key}: symbol cluster differs from launch")
    ctas = main["grid"][0] * main["grid"][1] * main["grid"][2]
    output_tiles = -(-m // tile_m) * -(-n // tile_n)
    persistent = ctas <= SM_COUNT and ctas < output_tiles * split_k
    launch = (tile_m, tile_n, tile_k, pipeline_depth, split_k,
              REDUCTION_NAMES[reduction_scheme], "none", 0, "false", split_k)
    diagnostics = {
        "persistent": persistent,
        "grid_equals_tiles_times_splits": ctas >= output_tiles * split_k,
        "kind": match["kind"],
    }
    return launch, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    summary_bytes = args.summary.read_bytes()
    summary = json.loads(summary_bytes)
    facts_bytes = args.facts.read_bytes()
    domain_bytes = args.domain.read_bytes()
    if summary["compact_sha256"] != _sha256(facts_bytes):
        raise ValueError("compact facts do not match the candidate summary")
    if summary["domain_sha256"] != _sha256(domain_bytes):
        raise ValueError("domain does not match the candidate summary")
    if (summary["hardware_id"] != HARDWARE_ID
            or summary["cublaslt_version"] != EXPECTED_CUBLASLT_VERSION
            or summary["duration_consumed"] is not False
            or summary["matmul_executed"] is not False
            or summary["production_kernel_submitted"] is not False
            or summary["max_workspace_bytes"] != 32 * 1024 * 1024):
        raise ValueError("candidate summary identity or evidence boundary differs")
    domain = json.loads(domain_bytes)
    facts = {}
    for line in gzip.decompress(facts_bytes).decode().splitlines():
        record = json.loads(line)
        if record["operation_key"] in facts:
            raise ValueError("duplicate operation in compact facts")
        facts[record["operation_key"]] = record
    expected_operations = {
        json.dumps(op, separators=(",", ":")) for op in domain["operations"]}
    if set(facts) != expected_operations:
        raise ValueError("capture does not cover exactly the domain operations")

    projected = {}
    diagnostics = {"persistent": 0, "tile_grid": 0, "kinds": {}}
    for key in sorted(facts):
        launch, diag = _project(key, facts[key])
        projected[key] = launch
        diagnostics["persistent"] += diag["persistent"]
        diagnostics["tile_grid"] += diag["grid_equals_tiles_times_splits"]
        kinds = diagnostics["kinds"]
        kinds[diag["kind"]] = (kinds[diag["kind"]] if diag["kind"] in kinds
                               else 0) + 1

    fields = gemm_launch_csv_fields(SCHEMA_TEMPLATE_HARDWARE)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(fields)
    rows = []
    for key_fields, operation_key in domain["selection_keys"]:
        rows.append(key_fields + list(projected[operation_key]))
    if len({tuple(row[:13]) for row in rows}) != len(rows):
        raise ValueError("duplicate physical selection key")
    for row in rows:
        writer.writerow(row)
    csv_text = stream.getvalue()
    csv_bytes = csv_text.encode()
    compressed = compress_gemm_launch_csv(csv_bytes)
    config_path = f"configs/hardware/{HARDWARE_ID}.yaml"
    launch_contract = load_gemm_launch_hardware_contract(
        ROOT / config_path, hardware_id=HARDWARE_ID)
    baseline = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    manifest = {
        "schema_version": gemm_launch_schema_version(SCHEMA_TEMPLATE_HARDWARE),
        "artifact_type": GEMM_LAUNCH_ARTIFACT_TYPE,
        "hardware_id": HARDWARE_ID,
        "hardware_revision": {
            "config_path": config_path,
            "launch_contract": launch_contract,
            "launch_contract_sha256": gemm_launch_hardware_contract_sha256(
                launch_contract),
        },
        "software_revision": SOFTWARE_REVISION,
        "generated_by": GENERATED_BY,
        "evidence_sha256": _sha256(facts_bytes),
        "evidence_sources": [
            {"kind": "compact_rank0_launch_facts", "sha256": _sha256(facts_bytes)},
            {"kind": "capture_candidate_summary", "sha256": _sha256(summary_bytes)},
        ],
        "capture_baseline_commit": baseline,
        "domain_sha256": domain["source_domain_sha256"],
        "csv_sha256": _sha256(csv_bytes),
        "csv_compressed_sha256": _sha256(compressed),
        "compression": GEMM_LAUNCH_COMPRESSION,
        "row_count": len(rows),
        "selection_key_fields": list(GEMM_LAUNCH_SELECTION_KEY_FIELDS),
        "launch_fields": list(gemm_launch_fields(SCHEMA_TEMPLATE_HARDWARE)),
        "semantic_context_fields": list(GEMM_SEMANTIC_CONTEXT_FIELDS),
        "boolean_encoding": GEMM_BOOLEAN_ENCODING,
        "null_encoding": GEMM_NULL_ENCODING,
        "duration_consumed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / GEMM_LAUNCH_CSV_FILENAME).write_bytes(compressed)
    (args.output_dir / GEMM_LAUNCH_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir),
                      "row_count": len(rows),
                      "operations": len(projected),
                      "diagnostics": diagnostics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
