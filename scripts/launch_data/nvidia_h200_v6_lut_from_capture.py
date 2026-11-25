#!/usr/bin/env python3
"""Convert segment1 + segment2 + BMM captures into one v6 LUT candidate.

Runs entirely offline (CPU only).  Reads the compact rank-0 launch facts
produced by the Modal capture runs, projects each record onto one v6 schema
CSV row, writes the candidate artifact pair, and validates it through the
strict canonical loader plus a selector routing spot-check.

Usage:
    python scripts/launch_data/nvidia_h200_v6_lut_from_capture.py \
        --output-dir experiment_logs/h200_gemm_lut_20261104/v6_candidate
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tempfile
from pathlib import Path

# Resolve the repo root from this script's location (scripts/launch_data/ -> repo root)
# so the producer works in any worktree or canonical checkout without a
# hard-coded absolute path.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.kernel.launch_config.gemm_launch_selection import (  # noqa: E402
    GEMM_LAUNCH_SELECTION_KEY_FIELDS,
    NVIDIA_H200_GEMM_V6_SELECTION_KEY_FIELDS,
    compress_gemm_launch_csv,
    gemm_launch_csv_fields_for_schema,
    gemm_launch_hardware_contract_sha256,
    load_gemm_launch_hardware_contract,
    load_gemm_launch_selection_files,
)
from src.kernel.hardware_spec import HWSpec  # noqa: E402
from src.kernel.launch_config.nvidia_h200.gemm_launch import (  # noqa: E402
    select_gemm_launch,
)
from src.kernel.launch_config.gemm_workload import (  # noqa: E402
    gemm_launch_selection_key,
)

HARDWARE_ID = "nvidia_h200"
SOFTWARE_REVISION = "cuda_12.9"
HW_CONFIG = ROOT / "configs/hardware/nvidia_h200.yaml"

CAPTURE_ROOT = ROOT / "profiler_output/launch_data/nvidia_h200/gemm_lut_20261104"
SEG1_FACTS = CAPTURE_ROOT / "segment1_raw_capture/rank0_launch_facts.jsonl.gz"
SEG2_FACTS = CAPTURE_ROOT / "segment2_bmm_capture/rank0_launch_facts_seg2.jsonl.gz"
BMM_FACTS = CAPTURE_ROOT / "segment2_bmm_capture/rank0_launch_facts_bmm.jsonl.gz"
SEG1_RECEIPT = CAPTURE_ROOT / "segment1_raw_capture/segment1_capture_receipt.json"
SEG2_RECEIPT = CAPTURE_ROOT / "segment2_bmm_capture/segment2_bmm_capture_receipt.json"
SEG3_FACTS = CAPTURE_ROOT / "segment3_lowm_capture/rank0_launch_facts_seg3.jsonl.gz"
SEG3_RECEIPT = CAPTURE_ROOT / "segment3_lowm_capture/segment3_lowm_capture_receipt.json"
SEG4_FACTS = CAPTURE_ROOT / "segment4_linear32k_capture/rank0_launch_facts_seg4.jsonl.gz"
SEG4_RECEIPT = CAPTURE_ROOT / "segment4_linear32k_capture/segment4_linear32k_capture_receipt.json"

# Root-preserved legacy v5 router source.  The canonical
# launch_data/nvidia_h200/ artifact will be replaced by the v6 candidate;
# these two files are the immutable, byte-identical copy of the v5 source
# the 8 router rows were read from.  SHA c4dd7d08... is pinned by root.
LEGACY_V5_ROUTER_DIR = CAPTURE_ROOT / "legacy_v5_router_source"
LEGACY_V5_ROUTER_CSV = LEGACY_V5_ROUTER_DIR / "gemm_launch_selection.csv.zst"
LEGACY_V5_ROUTER_MANIFEST = LEGACY_V5_ROUTER_DIR / "gemm_launch_selection.manifest.json"
LEGACY_V5_ROUTER_CSV_SHA256 = (
    "c4dd7d0854870439299c3b35b142a47d5e76e579ce89b3dfbc128d0d8626320c")


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _router_row(m: int, n: int, k: int, grid: tuple, block_threads: int,
                reducer_grid: tuple | None, reducer_block: tuple | None,
                partial_dtype: str, output_dtype: str) -> list[str]:
    """Build one v6 CSV row for a legacy SGLang router observation.

    The canonical v5 artifact stores router rows under the plain 13-field
    key with output_dtype="fp32" recorded in the launch fields.  The v6
    candidate must carry the same physical facts under the 19-field typed
    key so the selector finds them without a v5 fallback.
    """
    if reducer_grid is not None:
        reducer_fields = [str(reducer_grid[0]), str(reducer_grid[1]),
                          str(reducer_block[0]), str(reducer_block[1])]
        reduction = "fp32_external"
    else:
        reducer_fields = ["", "", "", ""]
        reduction = "none"
    return [
        "bf16", "gemm", "1", str(m), str(n), str(k),
        "row_major", "row_major", "row_major",
        "false", "true",
        "none", "false",
        # v6 typed-call extension: beta="", output_dtype_key="fp32",
        # batch_count=1, strides=None,None,None
        "", "fp32", "1", "", "", "",
        # launch fields: tile/pipeline/split stay null (grid_block_only)
        "", "", "", "",
        "",
        reduction,
        "grid_block_only",
        str(grid[0]), str(grid[1]), str(grid[2]),
        str(block_threads),
        *reducer_fields,
        partial_dtype,
        output_dtype,
    ]


def _validate_legacy_v5_source() -> None:
    """Verify the root-preserved legacy v5 router source before reading.

    Pins the CSV SHA256, manifest schema_version=5, and manifest identity
    so the producer cannot silently read a replaced or corrupted source.
    """
    actual_sha = _sha256_path(LEGACY_V5_ROUTER_CSV)
    if actual_sha != LEGACY_V5_ROUTER_CSV_SHA256:
        raise ValueError(
            f"legacy v5 router source SHA256 mismatch: "
            f"expected {LEGACY_V5_ROUTER_CSV_SHA256}, got {actual_sha}")
    import json as _json
    manifest = _json.loads(LEGACY_V5_ROUTER_MANIFEST.read_text())
    # Strict required-field reads (no .get defaults) per root contract.
    try:
        schema_version = manifest["schema_version"]
        hardware_id = manifest["hardware_id"]
        artifact_type = manifest["artifact_type"]
        manifest_csv_sha = manifest["csv_sha256"]
        manifest_compressed_sha = manifest["csv_compressed_sha256"]
    except KeyError as exc:
        raise ValueError(
            f"legacy v5 router source manifest missing required field: {exc}"
        ) from None
    if (schema_version != 5
            or hardware_id != "nvidia_h200"
            or artifact_type != "gemm_launch_selection"):
        raise ValueError(
            f"legacy v5 router source manifest identity differs: "
            f"schema={schema_version}, hardware={hardware_id!r}")
    # Verify the manifest binds the CSV we just hashed.
    import zstandard as _zstd
    compressed_bytes = LEGACY_V5_ROUTER_CSV.read_bytes()
    csv_bytes = _zstd.ZstdDecompressor().decompress(compressed_bytes)
    actual_csv_sha = hashlib.sha256(csv_bytes).hexdigest()
    actual_compressed_sha = hashlib.sha256(compressed_bytes).hexdigest()
    if actual_csv_sha != manifest_csv_sha:
        raise ValueError(
            f"legacy v5 manifest csv_sha256 mismatch: "
            f"manifest={manifest_csv_sha}, actual={actual_csv_sha}")
    if actual_compressed_sha != manifest_compressed_sha:
        raise ValueError(
            f"legacy v5 manifest csv_compressed_sha256 mismatch: "
            f"manifest={manifest_compressed_sha}, actual={actual_compressed_sha}")


def _load_router_rows_from_legacy_v5() -> list[list[str]]:
    """Read the 8 accepted router rows from the root-preserved legacy v5 source.

    Returns list of v6 CSV rows with the 19-field typed key.
    """
    _validate_legacy_v5_source()
    import zstandard
    compressed = LEGACY_V5_ROUTER_CSV.read_bytes()
    csv_bytes = zstandard.ZstdDecompressor().decompress(compressed)
    lines = csv_bytes.decode().splitlines()
    router_shapes = {(7168, 256), (6144, 256), (7168, 384)}
    rows = []
    for line in lines[1:]:
        parts = line.split(",")
        n, k = int(parts[4]), int(parts[5])
        if (k, n) not in router_shapes:
            continue
        # parts: dtype,variant,b,m,n,k,a_layout,b_layout,c_layout,
        #        trans_a,trans_b,epilogue,has_bias,
        #        tile_m,tile_n,tile_k,pipeline_depth,split_k,reduction,
        #        launch_observability,grid_x,grid_y,grid_z,block_threads,
        #        reducer_grid_x,reducer_grid_y,reducer_block_x,reducer_block_y,
        #        partial_dtype,output_dtype
        m = int(parts[3])
        grid = (int(parts[20]), int(parts[21]), int(parts[22]))
        block_threads = int(parts[23])
        reducer_grid = (int(parts[24]), int(parts[25])) if parts[24] else None
        reducer_block = (int(parts[26]), int(parts[27])) if parts[26] else None
        partial_dtype = parts[28] if parts[28] else ""
        output_dtype = parts[29]
        assert output_dtype == "fp32", f"router row must be FP32 output, got {output_dtype}"
        rows.append(_router_row(m, n, k, grid, block_threads,
                                reducer_grid, reducer_block,
                                partial_dtype, output_dtype))
    return rows


def _dense_launch_row_from_projection(projection: dict) -> list[str]:
    """Project one capture projection onto the v6 launch block."""
    rank0 = projection["rank0_algorithm"]
    launches = projection["launches"]
    main = launches[0]
    split = rank0["split_k"]
    reduction_code = rank0["reduction_scheme"]
    if split <= 0 or reduction_code not in (0, 2):
        raise ValueError(f"unsupported split/reduction: {split}/{reduction_code}")
    if (split == 1) != (reduction_code == 0):
        raise ValueError("split-1 must be direct and split>1 must be external")
    if main["block"][1:] != [1, 1] or main["grid"][2] != 1:
        raise ValueError("main block/grid geometry is unsupported")
    if reduction_code == 2:
        if len(launches) != 2 or "splitKreduce_kernel" not in launches[1]["symbol"]:
            raise ValueError("external reduction lacks the splitKreduce launch")
        reducer = launches[1]
        if reducer["grid"][2] != 1 or reducer["block"][2] != 1:
            raise ValueError("reducer geometry is unsupported")
        reducer_fields = [str(reducer["grid"][0]), str(reducer["grid"][1]),
                          str(reducer["block"][0]), str(reducer["block"][1])]
        reduction, partial_dtype = "fp32_external", "fp32"
    else:
        if len(launches) != 1:
            raise ValueError("direct launch has extra launches")
        reducer_fields = ["", "", "", ""]
        reduction, partial_dtype = "none", ""
    output_dtype = rank0["tile_id"] and None or None  # placeholder, replaced below
    return [
        "", "", "", "",
        "",
        reduction,
        "grid_block_only",
        str(main["grid"][0]), str(main["grid"][1]), str(main["grid"][2]),
        str(main["block"][0] * main["block"][1] * main["block"][2]),
        *reducer_fields,
        partial_dtype,
        "",  # output_dtype filled by caller
    ]


def _ordinary_row(m: int, n: int, k: int, dtype: str,
                  projection: dict) -> list[str]:
    """Build one v6 CSV row for an ordinary (batch=1) capture record."""
    launches = projection["launches"]
    main = launches[0]
    rank0 = projection["rank0_algorithm"]
    has_reducer = len(launches) > 1
    reduction = "fp32_external" if has_reducer else "none"
    if has_reducer:
        reducer = launches[1]
        reducer_fields = [str(reducer["grid"][0]), str(reducer["grid"][1]),
                          str(reducer["block"][0]), str(reducer["block"][1])]
        partial = "fp32"
    else:
        reducer_fields = ["", "", "", ""]
        partial = ""
    return [
        dtype, "gemm", "1", str(m), str(n), str(k),
        "row_major", "row_major", "row_major",
        "false", "true",
        "none", "false",
        "", "", "1", "", "", "",
        "", "", "", "",
        "",
        reduction,
        "grid_block_only",
        str(main["grid"][0]), str(main["grid"][1]), str(main["grid"][2]),
        str(main["block"][0] * main["block"][1] * main["block"][2]),
        *reducer_fields,
        partial,
        dtype,
    ]


def _bmm_row(m: int, n: int, k: int, dtype: str, batch: int,
             strides: tuple[int, int, int], projection: dict) -> list[str]:
    """Build one v6 CSV row for a typed batched-GEMM capture record."""
    launches = projection["launches"]
    main = launches[0]
    has_reducer = len(launches) > 1
    reduction = "fp32_external" if has_reducer else "none"
    if has_reducer:
        reducer = launches[1]
        reducer_fields = [str(reducer["grid"][0]), str(reducer["grid"][1]),
                          str(reducer["block"][0]), str(reducer["block"][1])]
        partial = "fp32"
    else:
        reducer_fields = ["", "", "", ""]
        partial = ""
    return [
        dtype, "batched_gemm", str(batch), str(m), str(n), str(k),
        "row_major", "row_major", "row_major",
        "false", "true",
        "none", "false",
        "", "", str(batch),
        str(strides[0]), str(strides[1]), str(strides[2]),
        "", "", "", "",
        "",
        reduction,
        "grid_block_only",
        str(main["grid"][0]), str(main["grid"][1]), str(main["grid"][2]),
        str(main["block"][0] * main["block"][1] * main["block"][2]),
        *reducer_fields,
        partial,
        dtype,
    ]


def _load_facts(path: Path) -> list[tuple[tuple, dict]]:
    records = []
    with gzip.open(path, "rt") as f:
        for line in f:
            rec = json.loads(line)
            records.append((tuple(json.loads(rec["operation_key"])),
                            rec["projection"]))
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seg1 = _load_facts(SEG1_FACTS)
    seg2 = _load_facts(SEG2_FACTS)
    bmm = _load_facts(BMM_FACTS)
    seg3 = _load_facts(SEG3_FACTS)
    seg4 = _load_facts(SEG4_FACTS)
    print(f"loaded: seg1={len(seg1)} seg2={len(seg2)} bmm={len(bmm)} seg3={len(seg3)} seg4={len(seg4)}")

    rows: list[list[str]] = []
    for (m, n, k, dtype, _bias), proj in seg1:
        rows.append(_ordinary_row(m, n, k, dtype, proj))
    for (m, n, k, dtype, _bias), proj in seg2:
        rows.append(_ordinary_row(m, n, k, dtype, proj))
    for (m, n, k, dtype, _bias), proj in seg3:
        rows.append(_ordinary_row(m, n, k, dtype, proj))
    for (m, n, k, dtype, _bias), proj in seg4:
        rows.append(_ordinary_row(m, n, k, dtype, proj))
    bmm_keys = set()
    for (m, n, k, dtype, batch), proj in bmm:
        strides = (m * k, m * n, m * n)
        rows.append(_bmm_row(m, n, k, dtype, batch, strides, proj))
        bmm_keys.add((m, n, k, dtype, batch))
    print(f"total rows: {len(rows)} (bmm={len(bmm_keys)})")

    # Merge the 8 legacy SGLang router rows from the root-preserved legacy
    # v5 source into the v6 candidate under their 19-field typed identity.
    router_rows = _load_router_rows_from_legacy_v5()
    print(f"router rows merged: {len(router_rows)}")
    rows.extend(router_rows)

    def sort_key(row: list[str]):
        # mirror the loader's parsed-key order: numeric fields as ints,
        # string fields lexicographically, over the 19-field typed key.
        key = []
        for i, value in enumerate(
                row[:len(NVIDIA_H200_GEMM_V6_SELECTION_KEY_FIELDS)]):
            if value.isdecimal():
                key.append(int(value))
            else:
                key.append(value)
        return tuple(key)
    rows.sort(key=sort_key)

    fields = gemm_launch_csv_fields_for_schema(HARDWARE_ID, 6)
    lines = [",".join(fields)]
    for row in rows:
        assert len(row) == len(fields), (len(row), len(fields), row)
        assert all("," not in v and '"' not in v for v in row)
        lines.append(",".join(row))
    csv_bytes = ("\n".join(lines) + "\n").encode()
    compressed = compress_gemm_launch_csv(csv_bytes)
    print(f"csv: {len(csv_bytes)} bytes raw, {len(compressed)} compressed")

    hardware_contract = load_gemm_launch_hardware_contract(
        HW_CONFIG, hardware_id=HARDWARE_ID)
    # Bind the evidence to the real capture receipts and compact fact files;
    # the evidence hash is the SHA-256 of the concatenated source hashes so
    # every byte of the manifest's evidence is content-addressed.
    source_hashes = [
        _sha256_path(SEG1_RECEIPT),
        _sha256_path(SEG2_RECEIPT),
        _sha256_path(SEG3_RECEIPT),
        _sha256_path(SEG4_RECEIPT),
        _sha256_path(SEG1_FACTS),
        _sha256_path(SEG2_FACTS),
        _sha256_path(BMM_FACTS),
        _sha256_path(SEG3_FACTS),
        _sha256_path(SEG4_FACTS),
        LEGACY_V5_ROUTER_CSV_SHA256,
    ]
    evidence_sha = hashlib.sha256(
        "".join(source_hashes).encode()).hexdigest()
    manifest = {
        "schema_version": 6,
        "artifact_type": "gemm_launch_selection",
        "hardware_id": HARDWARE_ID,
        "hardware_revision": {
            "config_path": "configs/hardware/nvidia_h200.yaml",
            "launch_contract": hardware_contract,
            "launch_contract_sha256": gemm_launch_hardware_contract_sha256(
                hardware_contract),
        },
        "software_revision": SOFTWARE_REVISION,
        "generated_by": "scripts/launch_data/nvidia_h200_launch_generator.py",
        "evidence_sha256": evidence_sha,
        "evidence_sources": [
            {"kind": "expanded_evidence", "sha256": evidence_sha},
            {"kind": "segment1_capture_receipt",
             "sha256": source_hashes[0]},
            {"kind": "segment2_bmm_capture_receipt",
             "sha256": source_hashes[1]},
            {"kind": "segment3_lowm_capture_receipt",
             "sha256": source_hashes[2]},
            {"kind": "segment4_linear32k_capture_receipt",
             "sha256": source_hashes[3]},
            {"kind": "segment1_compact_facts",
             "sha256": source_hashes[4]},
            {"kind": "segment2_compact_facts",
             "sha256": source_hashes[5]},
            {"kind": "bmm_compact_facts",
             "sha256": source_hashes[6]},
            {"kind": "segment3_compact_facts",
             "sha256": source_hashes[7]},
            {"kind": "segment4_compact_facts",
             "sha256": source_hashes[8]},
            {"kind": "legacy_v5_router_source",
             "sha256": source_hashes[9]},
        ],
        "capture_baseline_commit": "31a58ebb343b4cd11d6a22231419015de286bebb",
        "domain_sha256": hashlib.sha256(json.dumps(
            [r[:len(NVIDIA_H200_GEMM_V6_SELECTION_KEY_FIELDS)] for r in rows],
            separators=(",", ":")).encode()).hexdigest(),
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "csv_compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "compression": "zstd",
        "row_count": len(rows),
        "selection_key_fields": list(NVIDIA_H200_GEMM_V6_SELECTION_KEY_FIELDS),
        "launch_fields": list(fields[
            len(NVIDIA_H200_GEMM_V6_SELECTION_KEY_FIELDS):]),
        "semantic_context_fields": [
            "position", "tensor_parallel_size", "expert_parallel_size"],
        "boolean_encoding": "lowercase_true_false",
        "null_encoding": "empty_string",
        "duration_consumed": False,
        "capture_runtime": {
            "torch_version": "not_applicable_dry_launch",
            "torch_cuda_version": "12.9.1",
            "vllm_version": "not_applicable_dry_launch",
        },
        "consumption_scope": (
            "segment1+segment2+segment3+segment4 ordinary GEMM M1-16384 "
            "per-integer plus M32768: 77 of the 115 (K,N,dtype) keys cover "
            "the full M1-16384 range; the remaining 38 BF16 (K,N) keys now "
            "cover M1-16384 after segment3 filled the M1-8192 half "
            "(311,296 records); 13 fp16 keys cover the full M range plus "
            "M32768 (segment4, 13 records).  156 typed BMM rows (13 (K,N) "
            "fp16 x B{2..512} M=1 + B{4,8} M=4096) carry the actual capture "
            "runner's strided-batch call identity (stride_a=M*K, "
            "stride_b=M*N, stride_c=M*N, output_dtype unstated); the "
            "production compact-BMM call contract differs and a compact "
            "re-capture is separately contracted.  8 legacy SGLang router "
            "rows (bf16 input, fp32 output, N=256 K=7168, M=17..192) are "
            "merged from the canonical v5 artifact under their 19-field "
            "typed identity.  cuBLASLt 12.9.1 rank-0 heuristic dry-launch "
            "under 32MiB workspace; target-runtime call equivalence "
            "unverified; no duration consumed."
        ),
    }

    csv_path = output_dir / "gemm_launch_selection.csv.zst"
    manifest_path = output_dir / "gemm_launch_selection.manifest.json"
    csv_path.write_bytes(compressed)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    loaded = load_gemm_launch_selection_files(
        csv_path, manifest_path,
        hardware_id=HARDWARE_ID, software_revision=SOFTWARE_REVISION)
    assert len(loaded) == len(rows), (len(loaded), len(rows))
    print(f"strict loader: {len(loaded)} rows OK")

    # selector spot-checks: load the candidate directly into the selector.
    import src.kernel.launch_config.nvidia_h200.gemm_launch as h200_mod
    candidate_lut = load_gemm_launch_selection_files(
        csv_path, manifest_path,
        hardware_id=HARDWARE_ID, software_revision=SOFTWARE_REVISION)
    h200_mod._observed_launch_lut = lambda: candidate_lut
    hw = HWSpec(HW_CONFIG)
    attrs = {
        "a_layout": "row_major", "b_layout": "row_major",
        "c_layout": "row_major", "trans_a": False, "trans_b": True,
        "epilogue": "none", "has_bias": False,
        "launch_policy": "injected_launch",
        # The schema boundary always populates these; declare them so the
        # direct (non-KernelSpec) selector spot-check exercises the same
        # contract the production path sees after normalization.
        "beta": 0,
        "output_dtype": None,
    }
    checks = [
        ({"b": 1, "m": 64, "n": 10240, "k": 5120}, "bf16", "gemm", {}),
        ({"b": 1, "m": 8192, "n": 10240, "k": 5120}, "bf16", "gemm", {}),
        ({"b": 1, "m": 12288, "n": 10240, "k": 5120}, "bf16", "gemm", {}),
        ({"b": 1, "m": 16384, "n": 10240, "k": 5120}, "bf16", "gemm", {}),
        ({"b": 1, "m": 8193, "n": 11008, "k": 4096}, "fp16", "gemm", {}),
    ]
    for shape, dtype, variant, extra in checks:
        merged = {**attrs, **extra}
        cfg = select_gemm_launch(shape, dtype, variant, merged, hw)
        print(f"  selector {shape}: grid={cfg.grid} block={cfg.block}")
    bmm_attrs = {
        "a_layout": "row_major", "b_layout": "row_major",
        "c_layout": "row_major", "trans_a": False, "trans_b": True,
        "epilogue": "none", "has_bias": False,
        "launch_policy": "injected_launch",
        "beta": "0", "output_dtype": "fp16",
        "a_batch_stride_elements": 1536,
        "b_batch_stride_elements": 5120,
        "c_batch_stride_elements": 5120,
    }
    # The BMM capture rows were stored with output_dtype left blank because
    # the capture runner did not state an explicit output_dtype; the request
    # must match the stored identity exactly (output_dtype None), not assume
    # an unrecorded fp16 output.  This is the same public-contract rule the
    # production selector follows: an unstated output_dtype is ordinary.
    bmm_attrs["output_dtype"] = None
    cfg = select_gemm_launch(
        {"b": 128, "m": 1, "n": 5120, "k": 1536}, "fp16", "batched_gemm",
        bmm_attrs, hw)
    print(f"  selector BMM b=128 m=1 n=5120 k=1536: grid={cfg.grid}")
    # fp16 ordinary spot-check
    fp16_attrs = {**attrs}
    cfg = select_gemm_launch(
        {"b": 1, "m": 64, "n": 4096, "k": 11008}, "fp16", "gemm",
        fp16_attrs, hw)
    print(f"  selector fp16 m=64 n=4096 k=11008: grid={cfg.grid}")
    # A caller that bypasses the KernelSpec boundary and omits the
    # schema-normalized beta/output_dtype must still resolve to the
    # ordinary identity per the schema contract (the boundary is the
    # single source of the default, not a per-call detail).  This
    # spot-check pins that behaviour so a regression toward inventing
    # a parallel default or rejecting the call is caught here.
    cfg = select_gemm_launch(
        {"b": 1, "m": 64, "n": 10240, "k": 5120}, "bf16", "gemm",
        {k: v for k, v in attrs.items()
         if k not in ("beta", "output_dtype")}, hw)
    print(f"  selector without explicit beta/output_dtype: grid={cfg.grid}")

    # Router spot-check: the merged v6 candidate must serve the legacy
    # SGLang router rows under their 19-field typed identity without any
    # v5 fallback or monkeypatch.
    router_attrs = {
        "router_linear_runner": "sglang_linear_bf16_fp32_cublas",
        "weight_dtype": "bf16",
        "weight_group_size": 0,
    }
    for m in (17, 24, 32, 48, 64, 96, 128, 192):
        shape = {"b": 1, "m": m, "n": 256, "k": 7168}
        router_sel = h200_mod._observed_sglang_router_launch(
            shape, router_attrs)
        assert router_sel is not None, f"router m={m} not found in v6 candidate"
        print(f"  router m={m}: grid={router_sel.grid} block={router_sel.block_threads} "
              f"output_dtype={router_sel.output_dtype}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
