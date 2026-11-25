"""Build NVIDIA H200 dense and grid-block-only router launch selection.

The source is the H200 ``kcampaign`` launch records of the DSR1 router op
(``scripts/kernel_dataset/nvidia_h200/kc_deepseek_r1_decode_runner.py --op router``:
``torch.mm(x_bf16, W_bf16.T, out_dtype=torch.float32)`` above the DSV3 router
kernel's 16-token domain).  Each record states the cuBLASLt main kernel's
grid and block and the separate ``cublasLt::splitKreduce_kernel`` launch; it
states neither the tile nor the split count, so those columns stay null.
Only launch geometry is read: the records' durations are never consumed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from dataset_pipeline.launch_data.json_validators import (
    sha256_bytes,
    sorted_json_sha256,
)
from src.kernel.launch_config.gemm_launch_selection import (
    GEMM_BOOLEAN_ENCODING,
    GEMM_LAUNCH_ARTIFACT_TYPE,
    GEMM_LAUNCH_COMPRESSION,
    GEMM_LAUNCH_CSV_FILENAME,
    GEMM_LAUNCH_MANIFEST_FILENAME,
    GEMM_NULL_ENCODING,
    compress_gemm_launch_csv,
    gemm_launch_csv_fields,
    gemm_launch_fields,
    gemm_launch_hardware_contract_sha256,
    NVIDIA_H200_GEMM_OBSERVED_SCHEMA_VERSION,
    load_gemm_launch_hardware_contract,
    load_gemm_launch_selection_files,
)
from src.kernel.launch_config.gemm_workload import (
    GEMM_LAUNCH_SELECTION_KEY_FIELDS,
    GEMM_SEMANTIC_CONTEXT_FIELDS,
)


HARDWARE_ID = "nvidia_h200"
SOFTWARE_REVISION = "cuda_12.9"
GENERATED_BY = "scripts/launch_data/nvidia_h200_launch_generator.py"
CONSUMPTION_SCOPE = (
    "sglang_moe_router_cublas_bf16_fp32_only: captured under torch "
    "2.11.0+cu129 / vLLM 0.24.0+cu129, not the SGLang v0.5.19 cu130 image; "
    "every prediction consuming a row carries the limitation "
    "launch_observed_on_cuda_12_9_not_serving_cuda_13_0")
_ROOT = Path(__file__).resolve().parents[2]
_CASE_PATTERN = re.compile(r"DSR1_ROUTER_M(\d+)_H(\d+)_E(\d+)")
_MAIN_KERNEL_PATTERN = re.compile(r"nvjet_[A-Za-z0-9_]+_splitK_[A-Z]+")
# cublasLt::splitKreduce_kernel<32, 16, int, float(partial), ...>: the fourth
# template argument is the partial (compute) type.
_REDUCER_PREFIX = (
    "void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, ")
_REDUCER_BLOCK = (32, 16, 1)


def _window_geometry(record: Mapping[str, object], context: str):
    launches = record["launches"]
    if not isinstance(launches, list) or len(launches) != 2:
        raise ValueError(
            f"{context} must record exactly the main GEMM and its reducer")
    main, reducer = launches
    if _MAIN_KERNEL_PATTERN.fullmatch(main["name"]) is None:
        raise ValueError(f"{context} main launch is not a split-K nvjet GEMM")
    if not reducer["name"].startswith(_REDUCER_PREFIX):
        raise ValueError(
            f"{context} second launch is not the FP32-partial split-K reducer")
    main_grid = tuple(main["grid"])
    main_block = tuple(main["block"])
    reducer_grid = tuple(reducer["grid"])
    reducer_block = tuple(reducer["block"])
    if (len(main_grid) != 3 or main_block[1:] != (1, 1)
            or reducer_block != _REDUCER_BLOCK or reducer_grid[2] != 1):
        raise ValueError(f"{context} launch dimensions differ from the contract")
    return {
        "main_name": main["name"],
        "grid": list(main_grid),
        "block_threads": main_block[0],
        "reducer_grid": list(reducer_grid[:2]),
        "reducer_block": list(reducer_block[:2]),
    }


def expand_router_evidence(capture_dir: Path) -> list[dict[str, object]]:
    """Return every split-K router case with window-identical geometry."""

    cases = []
    runtimes = set()
    for case_dir in sorted(Path(capture_dir).iterdir()):
        match = _CASE_PATTERN.fullmatch(case_dir.name)
        if match is None:
            continue
        m, hidden, experts = (int(value) for value in match.groups())
        windows = sorted(case_dir.glob("window_*.launch.json"))
        if not windows:
            raise ValueError(f"{case_dir} has no launch windows")
        records = [json.loads(path.read_text(encoding="utf-8"))
                   for path in windows]
        expected_args = [
            "--op", "router", "--tokens", str(m), "--hidden", str(hidden),
            "--experts", str(experts)]
        for path, record in zip(windows, records):
            if record["runner_args"][:len(expected_args)] != expected_args:
                raise ValueError(f"{path} runner arguments differ from its case")
            telemetry = record["telemetry"]
            if (telemetry["hardware_id"] != HARDWARE_ID
                    or telemetry["clock_accepted"] is not True):
                raise ValueError(f"{path} is not an accepted H200 window")
            runtimes.add(tuple(sorted(telemetry["environment"].items())))
        if records[0]["launches"][0]["name"].startswith(
                "void router_gemm_kernel_float_output"):
            # M <= 16: the source-selected DSV3 router kernel, not cuBLAS.
            continue
        geometries = [
            _window_geometry(record, str(path))
            for path, record in zip(windows, records)]
        if any(geometry != geometries[0] for geometry in geometries[1:]):
            raise ValueError(f"{case_dir} windows disagree on launch geometry")
        cases.append({
            "case_id": case_dir.name,
            "m": m, "n": experts, "k": hidden,
            "windows": len(windows),
            "window_sha256": [
                sha256_bytes(path.read_bytes()) for path in windows],
            **geometries[0],
        })
    if len(runtimes) != 1:
        raise ValueError(f"capture mixes runtimes: {sorted(runtimes)}")
    runtime = dict(next(iter(runtimes)))
    if runtime["torch_cuda_version"] != "12.9":
        raise ValueError("router capture runtime is not CUDA 12.9")
    if not cases:
        raise ValueError(f"{capture_dir} contains no split-K router case")
    return [{"capture_runtime": runtime}, *cases]


def expanded_evidence_sha256(expanded: Sequence[object]) -> str:
    return sorted_json_sha256(list(expanded))


def _row(case: Mapping[str, object]) -> list[str]:
    key = [
        "bf16", "gemm", "1", str(case["m"]), str(case["n"]), str(case["k"]),
        "row_major", "row_major", "row_major", "false", "true", "none",
        "false",
    ]
    launch = ["", "", "", "", "", "fp32_external"]
    grid = case["grid"]
    observed = [
        "grid_block_only", *(str(value) for value in grid),
        str(case["block_threads"]),
        *(str(value) for value in case["reducer_grid"]),
        *(str(value) for value in case["reducer_block"]),
        "fp32", "fp32",
    ]
    return key + launch + observed


def build_launch_selection(
        capture_dir: Path, output_dir: Path,
        capture_baseline_commit: str) -> int:
    expanded = expand_router_evidence(capture_dir)
    runtime, cases = expanded[0]["capture_runtime"], expanded[1:]
    rows = sorted(
        (_row(case) for case in cases),
        key=lambda row: tuple(int(v) if v.isdecimal() else v
                              for v in row[:13]))
    header = list(gemm_launch_csv_fields(HARDWARE_ID))
    csv_bytes = (
        "\n".join(",".join(row) for row in [header, *rows]) + "\n"
    ).encode("utf-8")
    compressed = compress_gemm_launch_csv(csv_bytes)
    keys = [json.dumps(row[:13], separators=(",", ":")) for row in rows]
    contract = load_gemm_launch_hardware_contract(
        _ROOT / "configs" / "hardware" / f"{HARDWARE_ID}.yaml",
        hardware_id=HARDWARE_ID)
    evidence_sha256 = expanded_evidence_sha256(expanded)
    raw_sha256 = sorted_json_sha256(
        [sha for case in cases for sha in case["window_sha256"]])
    manifest = {
        "schema_version": NVIDIA_H200_GEMM_OBSERVED_SCHEMA_VERSION,
        "artifact_type": GEMM_LAUNCH_ARTIFACT_TYPE,
        "hardware_id": HARDWARE_ID,
        "hardware_revision": {
            "config_path": f"configs/hardware/{HARDWARE_ID}.yaml",
            "launch_contract": contract,
            "launch_contract_sha256": (
                gemm_launch_hardware_contract_sha256(contract)),
        },
        "software_revision": SOFTWARE_REVISION,
        "generated_by": GENERATED_BY,
        "evidence_sha256": evidence_sha256,
        "evidence_sources": [
            {"kind": "expanded_evidence", "sha256": evidence_sha256},
            {"kind": "kcampaign_launch_windows", "sha256": raw_sha256},
        ],
        "capture_baseline_commit": capture_baseline_commit,
        "domain_sha256": sha256_bytes("\n".join(keys).encode("utf-8")),
        "csv_sha256": sha256_bytes(csv_bytes),
        "csv_compressed_sha256": sha256_bytes(compressed),
        "compression": GEMM_LAUNCH_COMPRESSION,
        "row_count": len(rows),
        "selection_key_fields": list(GEMM_LAUNCH_SELECTION_KEY_FIELDS),
        "launch_fields": list(gemm_launch_fields(HARDWARE_ID)),
        "semantic_context_fields": list(GEMM_SEMANTIC_CONTEXT_FIELDS),
        "boolean_encoding": GEMM_BOOLEAN_ENCODING,
        "null_encoding": GEMM_NULL_ENCODING,
        "duration_consumed": False,
        "capture_runtime": runtime,
        "consumption_scope": CONSUMPTION_SCOPE,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / GEMM_LAUNCH_CSV_FILENAME
    manifest_path = output_dir / GEMM_LAUNCH_MANIFEST_FILENAME
    csv_path.write_bytes(compressed)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    loaded = load_gemm_launch_selection_files(
        csv_path, manifest_path, hardware_id=HARDWARE_ID,
        software_revision=SOFTWARE_REVISION)
    return len(loaded)


# A separate no-matmul query captured rank-0 cuBLASLt choices for the full
# shared input domain. Keep these facts in the same H200 artifact as the
# router rows; they carry complete tiles/splits instead of guessing from grid.
_DENSE_SYMBOL = re.compile(
    r"nvjet_(?P<kind>[a-z]+)_(?P<n>\d+)x(?P<m>\d+)_"
    r"(?P<k>\d+)x(?P<depth>\d+)_(?P<cx>\d+)x(?P<cy>\d+)_"
    r"(?P<raster>[vh])_(?P<suffix>\w+)")


def _dense_launch_row(operation_key: str, fact: Mapping) -> list[str]:
    _, _, _, dtype, _ = json.loads(operation_key)
    rank = fact["rank0"]
    launches = fact["launches"]
    if not launches or rank["rank"] != 0:
        raise ValueError("dense observation lacks rank-zero launch")
    main = launches[0]
    match = _DENSE_SYMBOL.fullmatch(main["symbol"])
    if match is None:
        raise ValueError("dense observation is not a supported nvjet symbol")
    tile_m, tile_n = rank["logical_tile_m"], rank["logical_tile_n"]
    tile_k, depth = int(match["k"]), int(match["depth"])
    if (rank["tile"] != f"{match['n']}x{match['m']}"
            or (tile_m, tile_n) != (int(match["m"]), int(match["n"]))
            or rank["stages"] != f"{tile_k}xAUTO"):
        raise ValueError("dense symbol tile/stages disagree with rank-zero facts")
    if (rank["cluster_shape"] != f"{match['cx']}x{match['cy']}x1"
            or main["cluster"] != [int(match["cx"]) * int(match["cy"]), 1, 1]):
        raise ValueError("dense symbol cluster disagrees with launch")
    split = rank["split_k"]
    reduction_code = rank["reduction_scheme"]
    if (type(split) is not int or split <= 0
            or reduction_code not in (0, 2)
            or (split == 1) != (reduction_code == 0)
            or ("splitK" in match["suffix"]) != (split > 1)):
        raise ValueError("dense rank-zero split/reduction contract differs")
    if main["block"][1:] != [1, 1] or main["grid"][2] != 1:
        raise ValueError("dense main block/grid geometry is unsupported")
    if reduction_code == 2:
        if len(launches) != 2 or "splitKreduce_kernel" not in launches[1]["symbol"]:
            raise ValueError("dense observation lacks the separate split reducer")
        reducer = launches[1]
        if reducer["grid"][2] != 1 or reducer["block"][2] != 1:
            raise ValueError("dense reducer geometry is unsupported")
        reducer_fields = [*reducer["grid"][:2], *reducer["block"][:2]]
        reduction, partial_dtype = "fp32_external", "fp32"
    else:
        if len(launches) != 1:
            raise ValueError("non-split dense observation has extra launches")
        reducer_fields = ["", "", "", ""]
        reduction, partial_dtype = "none", ""
    evidence = fact["device_execution"]
    if (evidence["evidence_kind"] != "cupti_activity_kernel"
            or evidence["all_replacements_observed"] is not True
            or evidence["dropped_record_count"] != 0
            or evidence["activity_record_count"] != len(launches)):
        raise ValueError("dense observation lacks complete no-matmul CUPTI evidence")
    return [str(x) for x in (
        tile_m, tile_n, tile_k, depth, split, reduction,
        "tiled_grid", *main["grid"], main["block"][0],
        *reducer_fields, partial_dtype, dtype)]


def build_dense_launch_selection(
        domain_path: Path, facts_path: Path, summary_path: Path,
        router_dir: Path, output_dir: Path, capture_baseline_commit: str) -> int:
    """Package the full rank-zero capture plus existing router observations.

    This runs entirely offline. The captured CUDA revision and rank-zero
    preference are not evidence that another Serving runtime chooses alike.
    """
    domain_bytes = domain_path.read_bytes()
    facts_bytes = facts_path.read_bytes()
    summary_bytes = summary_path.read_bytes()
    domain, summary = json.loads(domain_bytes), json.loads(summary_bytes)
    if (summary["compact_sha256"] != sha256_bytes(facts_bytes)
            or summary["domain_sha256"] != sha256_bytes(domain_bytes)):
        raise ValueError("dense capture domain/facts SHA mismatch")
    if (summary["hardware_id"] != HARDWARE_ID
            or summary["cublaslt_version"] != 120901
            or summary["selector_contract"] != "cublaslt_first_feasible_rank_0"
            or summary["max_workspace_bytes"] != 32 * 1024 * 1024
            or summary["duration_consumed"] is not False
            or summary["matmul_executed"] is not False
            or summary["production_kernel_submitted"] is not False):
        raise ValueError("dense capture identity or evidence boundary differs")
    if domain["selection_key_fields"] != list(GEMM_LAUNCH_SELECTION_KEY_FIELDS):
        raise ValueError("dense domain selection key fields differ")
    projected = {}
    for line in gzip.decompress(facts_bytes).decode().splitlines():
        fact = json.loads(line)
        key = fact["operation_key"]
        if key in projected:
            raise ValueError("duplicate dense captured operation")
        projected[key] = _dense_launch_row(key, fact)
    expected = {json.dumps(op, separators=(",", ":")) for op in domain["operations"]}
    if (len(expected) != len(domain["operations"])
            or set(projected) != expected
            or len(projected) != summary["operation_count"]
            or len(projected) != domain["operation_count"]):
        raise ValueError("dense capture does not cover its exact operation domain")
    router_csv = router_dir / GEMM_LAUNCH_CSV_FILENAME
    router_manifest = router_dir / GEMM_LAUNCH_MANIFEST_FILENAME
    routers = load_gemm_launch_selection_files(
        router_csv, router_manifest, hardware_id=HARDWARE_ID,
        software_revision=SOFTWARE_REVISION)
    if any(r.observed_grid is None or r.observed_grid.observability != "grid_block_only"
           for r in routers.values()):
        raise ValueError("router input must contain only grid-block-only observations")
    from src.kernel.launch_config.gemm_launch_selection import decompress_gemm_launch_csv
    rows = list(csv.reader(io.StringIO(decompress_gemm_launch_csv(
        router_csv.read_bytes()).decode())))[1:]
    for fields, operation_key in domain["selection_keys"]:
        if len(fields) != len(GEMM_LAUNCH_SELECTION_KEY_FIELDS):
            raise ValueError("dense selection key is incomplete")
        dtype, variant, b, m, n, k = fields[:6]
        if (dtype not in ("bf16", "fp16") or variant not in ("gemm", "gemv")
                or fields[6:] != ["row_major", "row_major", "row_major",
                                  "false", "true", "none", "false"]):
            raise ValueError("dense key call semantics differ from captured row-major linear")
        expected_operation = json.dumps(
            [int(b) * int(m), int(n), int(k), dtype, fields[12] == "true"],
            separators=(",", ":"))
        if expected_operation != operation_key:
            raise ValueError("dense key-to-operation mapping differs")
        rows.append([*fields, *projected[operation_key]])
    if len(domain["selection_keys"]) != domain["selection_key_count"]:
        raise ValueError("dense domain key count differs")
    keys = [tuple(row[:13]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate physical selection key across dense/router domains")
    rows.sort(key=lambda row: tuple(int(v) if v.isdecimal() else v for v in row[:13]))
    csv_bytes = ("\n".join(",".join(row) for row in
        [list(gemm_launch_csv_fields(HARDWARE_ID)), *rows]) + "\n").encode()
    compressed = compress_gemm_launch_csv(csv_bytes)
    manifest = json.loads(router_manifest.read_bytes())
    manifest.update({
        "schema_version": NVIDIA_H200_GEMM_OBSERVED_SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "capture_baseline_commit": capture_baseline_commit,
        "evidence_sha256": sha256_bytes(facts_bytes),
        "evidence_sources": [
            {"kind": "compact_rank0_launch_facts", "sha256": sha256_bytes(facts_bytes)},
            {"kind": "capture_candidate_summary", "sha256": sha256_bytes(summary_bytes)},
            {"kind": "capture_operation_domain", "sha256": sha256_bytes(domain_bytes)},
            {"kind": "router_manifest", "sha256": sha256_bytes(router_manifest.read_bytes())},
            {"kind": "router_csv", "sha256": sha256_bytes(router_csv.read_bytes())},
        ],
        "domain_sha256": sha256_bytes(json.dumps(
            [row[:13] for row in rows], separators=(",", ":")).encode()),
        "csv_sha256": sha256_bytes(csv_bytes),
        "csv_compressed_sha256": sha256_bytes(compressed),
        "row_count": len(rows),
        "consumption_scope": (
            "capture_runtime describes retained router observations only; "
            "tiled_grid: standalone cuBLASLt 12.9 rank-zero heuristic under 32MiB workspace "
            "for row-major unbiassed linear; target runtime call equivalence unverified; "
            "cluster/TMA/persistent task program retained in raw but not modeled. "
            "grid_block_only: " + CONSUMPTION_SCOPE),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, manifest_path = output_dir / GEMM_LAUNCH_CSV_FILENAME, output_dir / GEMM_LAUNCH_MANIFEST_FILENAME
    csv_path.write_bytes(compressed)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    loaded = load_gemm_launch_selection_files(
        csv_path, manifest_path, hardware_id=HARDWARE_ID, software_revision=SOFTWARE_REVISION)
    return len(loaded)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--domain", type=Path)
    parser.add_argument("--facts", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--router-dir", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=_ROOT / "launch_data" / HARDWARE_ID)
    parser.add_argument("--capture-baseline-commit", required=True)
    args = parser.parse_args(argv)
    dense_args = (args.domain, args.facts, args.summary, args.router_dir)
    if args.capture_dir is not None:
        if any(value is not None for value in dense_args):
            parser.error("router capture and dense packaging modes are exclusive")
        count = build_launch_selection(
            args.capture_dir, args.output_dir, args.capture_baseline_commit)
    else:
        if any(value is None for value in dense_args):
            parser.error("dense packaging requires --domain --facts --summary --router-dir")
        count = build_dense_launch_selection(
            *dense_args, args.output_dir, args.capture_baseline_commit)
    print(json.dumps({"gemm": count}, sort_keys=True))
    return 0
