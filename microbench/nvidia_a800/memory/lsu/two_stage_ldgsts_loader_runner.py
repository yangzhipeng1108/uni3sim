#!/usr/bin/env python3
"""Capture and audit the A800 grouped-MLA two-stage LDGSTS loader probe."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


HARDWARE_ID = "nvidia_a800"
EXPECTED_DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
EXPECTED_SM_COUNT = 108
EXPECTED_L2_BYTES = 40 * 1024 * 1024
BLOCK_THREADS = 128
WARPS_PER_CTA = 4
PIPELINE_STAGES = 2
BLOCK_H = 16
HEAD_GROUP_COUNT = 4
NUM_KV_SPLITS = 4
STREAMS_PER_STAGE = 3
DYNAMIC_SHARED_BYTES = 89088
K_NOPE_SHARED_BASE = 0
V_SHARED_BASE = 32768
K_ROPE_SHARED_BASE = 81920
SWIZZLE_VECTOR_ELEMENTS = 8
SWIZZLE_PER_PHASE = 1
SWIZZLE_MAX_PHASE = 8
QK_HEAD_DIM = 576
VALUE_HEAD_DIM = 512
KV_TILE = 32
ELEMENT_BYTES = 2
K_NOPE_BYTES = 32768
K_ROPE_BYTES = 4096
V_BYTES = 32768
STAGE_PAYLOAD_BYTES = 69632
SOURCE_TILE_BYTES = 36864
CP_ASYNC_BYTES_PER_THREAD = 16
COPIES_PER_THREAD_PER_STAGE = 34
LDGSTS_WARP_INSTRUCTIONS_PER_STAGE = 136
XBAR_BYTES_PER_LDGSTS_WARP_INSTRUCTION = 1024
XBAR_TRANSACTION_BYTES = 32
L2_SECTOR_BYTES = 32
L2_LINE_BYTES = 128
MAX_TRIAL_RELATIVE_SPREAD = 0.05
MAX_PLATEAU_RELATIVE_DELTA = 0.05
MIN_CANDIDATE_TRIALS_PER_CELL = 7
WARMUP_LAUNCHES = 3
GRID_POINTS = (1, 2, 8, 32, 64, 108)
TILE_POINTS = (2, 4, 16, 64)
SMOKE_CELL = (1, 4)
NCU_CELLS = ((1, 2), (1, 64), (108, 2), (108, 64))
RATE_UNIT = "xbar_accounted_32B_transactions_per_active_SM_per_SM_cycle"
PAYLOAD_RATE_UNIT = "payload_bytes_per_active_SM_per_SM_cycle"
SOURCE_NAME = "two_stage_ldgsts_loader.cu"
RUNNER_NAME = "two_stage_ldgsts_loader_runner.py"
STATIC_MANIFEST_SHA256 = (
    "61ebecd21687c5f806d938dfa73c463184ddf56a52d9c5dce82c2c1f5164a9f0")
VLLM_SOURCE_SHA256 = (
    "8bc1fe6d8ef29e34c0bf833373dc309c705872b4b24d9239952284d10a5d208a")
STATIC_CUBIN_SHA256 = (
    "7f2c5e8096a70832d68744d4fd574f267c9a18f7c889ad654c8039ef0ad99293")
STATIC_TTGIR_SHA256 = (
    "3346bae446f4bb33222d09323cbdb8540829225dbbe1c3fc26c76abe48abf703")
STATIC_METADATA_SHA256 = (
    "73bedfe84159f18de18632f4350c428cb2e728fd1d18708671c89af48ea93dde")
EXPECTED_HEADER = (
    "expected", "grid", "tiles", "ldgsts_warp_instructions",
    "payload_bytes", "expected_xbar_bytes",
    "expected_xbar_32b_transactions",
)
RAW_HEADER = (
    "raw", "grid", "tiles", "trial", "ldgsts_warp_instructions",
    "payload_bytes", "sum_cta_sm_cycles", "min_cta_sm_cycles",
    "max_cta_sm_cycles",
    "xbar_32b_transactions_per_active_sm_cycle",
)
NCU_METRICS = (
    "sm__sass_inst_executed_op_ldgsts.sum",
    "sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "dram__bytes_read.sum",
)
NCU_METRIC_UNITS = {
    "sm__sass_inst_executed_op_ldgsts.sum": "inst",
    "sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass.sum": "byte",
    "lts__t_sectors_srcunit_tex_op_read.sum": "sector",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum": "sector",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum": "sector",
    "dram__bytes_read.sum": "byte",
}
REUSE_ACTIVE_SOURCES = 108
REUSE_TILES_PER_SPLIT = (1, 4, 8, 10, 11, 12, 16)
REUSE_NCU_METRICS = (
    "sm__sass_inst_executed_op_ldgsts.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "dram__bytes_read.sum",
)
REUSE_NCU_METRIC_UNITS = {
    "sm__sass_inst_executed_op_ldgsts.sum": "inst",
    "lts__t_sectors_srcunit_tex_op_read.sum": "sector",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum": "sector",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum": "sector",
    "dram__bytes_read.sum": "byte",
}
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
DEVICE_QUERY_FIELDS = (
    "uuid,name,pci.bus_id,driver_version,pstate,memory.used,memory.total,"
    "utilization.gpu,utilization.memory,clocks.sm,clocks.mem")
COMPUTE_QUERY_FIELDS = "gpu_uuid,pid,process_name,used_gpu_memory"


def _require_nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _parse_uint(text, label):
    normalized = _require_nonempty(text, label).replace(",", "")
    if re.fullmatch(r"[0-9]+", normalized) is None:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(normalized)


def _parse_float(text, label):
    normalized = _require_nonempty(text, label).replace(",", "")
    value = float(normalized)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _run(command, *, cwd, env=None, stdout_path=None, stderr_path=None):
    if not isinstance(command, list) or not command:
        raise TypeError("command must be a non-empty list")
    if any(not isinstance(value, str) or not value for value in command):
        raise ValueError("command entries must be non-empty strings")
    stdout_stream = subprocess.PIPE
    stderr_stream = subprocess.PIPE
    opened = []
    if stdout_path is not None:
        stdout_stream = stdout_path.open("w", encoding="utf-8")
        opened.append(stdout_stream)
    if stderr_path is not None:
        stderr_stream = stderr_path.open("w", encoding="utf-8")
        opened.append(stderr_stream)
    try:
        return subprocess.run(
            command, cwd=cwd, env=env, text=True, stdout=stdout_stream,
            stderr=stderr_stream, check=True)
    finally:
        for stream in opened:
            stream.close()


def _git_output(repo_root, *arguments):
    return _run(["git", *arguments], cwd=repo_root).stdout.strip()


def _normalize_pci_bus_id(text):
    value = _require_nonempty(text, "PCI bus ID")
    match = re.fullmatch(
        r"([0-9A-Fa-f]{4}|[0-9A-Fa-f]{8}):"
        r"([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.([0-7])", value)
    if match is None:
        raise ValueError(f"PCI bus ID has invalid syntax: {value!r}")
    domain = int(match.group(1), 16)
    bus = int(match.group(2), 16)
    device = int(match.group(3), 16)
    function = int(match.group(4), 16)
    if domain > 0xffff or device > 0x1f:
        raise ValueError(f"PCI bus ID is outside PCI address bounds: {value!r}")
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function:x}"


def _expected_cells(mode):
    if mode == "full_matrix":
        return tuple((grid, tiles) for grid in GRID_POINTS for tiles in TILE_POINTS)
    if mode == "smoke_cell":
        return (SMOKE_CELL,)
    raise ValueError(f"unknown native capture mode {mode!r}")


def _expected_counts(grid, tiles):
    instructions = grid * tiles * LDGSTS_WARP_INSTRUCTIONS_PER_STAGE
    payload = grid * tiles * STAGE_PAYLOAD_BYTES
    xbar_bytes = instructions * XBAR_BYTES_PER_LDGSTS_WARP_INSTRUCTION
    return {
        "ldgsts_warp_instructions": instructions,
        "payload_bytes": payload,
        "expected_xbar_bytes": xbar_bytes,
        "expected_xbar_32b_transactions": xbar_bytes // XBAR_TRANSACTION_BYTES,
    }


def _validate_meta(meta, mode):
    required = {
        "schema", "device_index", "device_name", "device_uuid", "pci_bus_id",
        "compute_capability", "sm_count", "l2_bytes", "block_threads",
        "resident_ctas_per_sm", "pipeline_stages",
        "dynamic_shared_bytes_per_cta", "k_nope_shared_base",
        "v_shared_base", "k_rope_shared_base", "k_krope_shared_layout",
        "v_shared_layout", "qk_head_dim", "value_head_dim",
        "kv_tile", "element_bytes", "k_panel_bytes", "v_panel_bytes",
        "stage_payload_bytes", "cp_async_bytes_per_thread",
        "copies_per_thread_per_stage", "ldgsts_warp_instructions_per_stage",
        "warmup_launches", "grid_points", "tile_points", "capture_mode",
        "timer_domain", "timed_barriers_included", "timer_start",
        "timer_end", "cross_sm_clocks_compared",
        "production_or_evaluation_duration_consumed",
        "production_parameter_published",
    }
    if set(meta) != required:
        raise ValueError(
            "native meta schema mismatch: "
            f"missing={sorted(required-set(meta))}, "
            f"extra={sorted(set(meta)-required)}")
    expected = {
        "schema": "nvidia_a800_two_stage_ldgsts_native_v3",
        "device_index": "0",
        "device_name": EXPECTED_DEVICE_NAME,
        "compute_capability": "sm_80",
        "sm_count": str(EXPECTED_SM_COUNT),
        "l2_bytes": str(EXPECTED_L2_BYTES),
        "block_threads": str(BLOCK_THREADS),
        "resident_ctas_per_sm": "1",
        "pipeline_stages": str(PIPELINE_STAGES),
        "dynamic_shared_bytes_per_cta": str(DYNAMIC_SHARED_BYTES),
        "k_nope_shared_base": str(K_NOPE_SHARED_BASE),
        "v_shared_base": str(V_SHARED_BASE),
        "k_rope_shared_base": str(K_ROPE_SHARED_BASE),
        "k_krope_shared_layout": "vec8_perPhase1_maxPhase8_order01",
        "v_shared_layout": "vec8_perPhase1_maxPhase8_order10",
        "qk_head_dim": str(QK_HEAD_DIM),
        "value_head_dim": str(VALUE_HEAD_DIM),
        "kv_tile": str(KV_TILE),
        "element_bytes": str(ELEMENT_BYTES),
        "k_panel_bytes": str(K_NOPE_BYTES + K_ROPE_BYTES),
        "v_panel_bytes": str(V_BYTES),
        "stage_payload_bytes": str(STAGE_PAYLOAD_BYTES),
        "cp_async_bytes_per_thread": str(CP_ASYNC_BYTES_PER_THREAD),
        "copies_per_thread_per_stage": str(COPIES_PER_THREAD_PER_STAGE),
        "ldgsts_warp_instructions_per_stage": str(
            LDGSTS_WARP_INSTRUCTIONS_PER_STAGE),
        "warmup_launches": str(WARMUP_LAUNCHES),
        "grid_points": "|".join(str(value) for value in GRID_POINTS),
        "tile_points": "|".join(str(value) for value in TILE_POINTS),
        "capture_mode": mode,
        "timer_domain": "clock64_sm_local_warp_envelope_cycles",
        "timed_barriers_included": "false",
        "timer_start": "per_warp_before_prologue_copy",
        "timer_end": "per_warp_after_final_wait_group_zero",
        "cross_sm_clocks_compared": "false",
        "production_or_evaluation_duration_consumed": "false",
        "production_parameter_published": "false",
    }
    for key, value in expected.items():
        if meta[key] != value:
            raise ValueError(
                f"native meta {key!r} is {meta[key]!r}, expected {value!r}")
    if re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
            meta["device_uuid"]) is None:
        raise ValueError("native device UUID has invalid syntax")
    _normalize_pci_bus_id(meta["pci_bus_id"])


def parse_native_output(text, expected_trials=7, expected_mode="full_matrix"):
    """Parse a complete native matrix and recompute every physical rate."""

    if not isinstance(text, str) or not text:
        raise ValueError("native output must be non-empty text")
    if (isinstance(expected_trials, bool) or not isinstance(expected_trials, int)
            or expected_trials <= 0):
        raise ValueError("expected_trials must be a positive integer")
    cells = set(_expected_cells(expected_mode))
    meta = {}
    expected = {}
    raw = {}
    summary = {}
    expected_header = None
    raw_header = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        columns = tuple(next(csv.reader([line])))
        kind = columns[0]
        if kind == "meta":
            if len(columns) != 3:
                raise ValueError(f"native meta row {line_number} is malformed")
            key = _require_nonempty(columns[1], "native meta key")
            if key in meta:
                raise ValueError(f"native meta repeats {key!r}")
            meta[key] = columns[2]
        elif kind == "expected" and columns[1:] == EXPECTED_HEADER[1:]:
            if expected_header is not None:
                raise ValueError("native output repeats expected header")
            expected_header = columns
        elif kind == "raw" and columns[1:] == RAW_HEADER[1:]:
            if raw_header is not None:
                raise ValueError("native output repeats raw header")
            raw_header = columns
        elif kind == "expected":
            if len(columns) != len(EXPECTED_HEADER):
                raise ValueError(f"native expected row {line_number} is malformed")
            grid = _parse_uint(columns[1], "expected grid")
            tiles = _parse_uint(columns[2], "expected tiles")
            key = (grid, tiles)
            if key in expected:
                raise ValueError(f"native expected repeats {key!r}")
            expected[key] = {
                "grid": grid,
                "tiles": tiles,
                "ldgsts_warp_instructions": _parse_uint(
                    columns[3], "expected LDGSTS instructions"),
                "payload_bytes": _parse_uint(columns[4], "expected payload"),
                "expected_xbar_bytes": _parse_uint(
                    columns[5], "expected Xbar bytes"),
                "expected_xbar_32b_transactions": _parse_uint(
                    columns[6], "expected Xbar transactions"),
            }
        elif kind == "raw":
            if len(columns) != len(RAW_HEADER):
                raise ValueError(f"native raw row {line_number} is malformed")
            grid = _parse_uint(columns[1], "raw grid")
            tiles = _parse_uint(columns[2], "raw tiles")
            trial = _parse_uint(columns[3], "raw trial")
            key = (grid, tiles, trial)
            if key in raw:
                raise ValueError(f"native raw repeats {key!r}")
            raw[key] = {
                "grid": grid,
                "tiles": tiles,
                "trial": trial,
                "ldgsts_warp_instructions": _parse_uint(
                    columns[4], "raw LDGSTS instructions"),
                "payload_bytes": _parse_uint(columns[5], "raw payload"),
                "sum_cta_sm_cycles": _parse_uint(columns[6], "sum CTA cycles"),
                "min_cta_sm_cycles": _parse_uint(columns[7], "minimum CTA cycles"),
                "max_cta_sm_cycles": _parse_uint(columns[8], "maximum CTA cycles"),
                "rate": _parse_float(columns[9], "raw Xbar transaction rate"),
            }
        elif kind == "summary":
            if len(columns) != 4:
                raise ValueError(f"native summary row {line_number} is malformed")
            grid = _parse_uint(columns[1], "summary grid")
            tiles = _parse_uint(columns[2], "summary tiles")
            key = (grid, tiles)
            if key in summary:
                raise ValueError(f"native summary repeats {key!r}")
            summary[key] = _parse_float(columns[3], "summary rate")
        else:
            raise ValueError(
                f"native output has unknown row kind at line {line_number}: {kind!r}")
    _validate_meta(meta, expected_mode)
    if expected_header != EXPECTED_HEADER or raw_header != RAW_HEADER:
        raise ValueError("native CSV header changed")
    if set(expected) != cells or set(summary) != cells:
        raise ValueError("native expected/summary cell matrix mismatch")
    expected_raw = {
        (grid, tiles, trial) for grid, tiles in cells
        for trial in range(expected_trials)}
    if set(raw) != expected_raw:
        raise ValueError("native raw cell matrix mismatch")
    for cell in sorted(cells):
        counts = _expected_counts(*cell)
        if any(expected[cell][name] != value for name, value in counts.items()):
            raise ValueError(f"native static transaction count changed for {cell!r}")
        rates = []
        for trial in range(expected_trials):
            record = raw[(*cell, trial)]
            if any(record[name] != counts[name] for name in (
                    "ldgsts_warp_instructions", "payload_bytes")):
                raise ValueError(f"native raw transaction count changed for {cell!r}")
            if not (0 < record["min_cta_sm_cycles"]
                    <= record["max_cta_sm_cycles"]
                    <= record["sum_cta_sm_cycles"]):
                raise ValueError(f"native cycle interval is invalid for {cell!r}")
            rate = counts["expected_xbar_32b_transactions"] / record[
                "sum_cta_sm_cycles"]
            if not math.isclose(rate, record["rate"], rel_tol=5e-12, abs_tol=5e-13):
                raise ValueError(f"native rate does not conserve transactions for {cell!r}")
            rates.append(record["rate"])
        median = sorted(rates)[len(rates) // 2]
        if not math.isclose(median, summary[cell], rel_tol=5e-12, abs_tol=5e-13):
            raise ValueError(f"native median is inconsistent for {cell!r}")
    return {"meta": meta, "expected": expected, "raw": raw, "summary": summary}


def _swizzled_contiguous_element(contiguous_element, orthogonal_element):
    if contiguous_element % SWIZZLE_VECTOR_ELEMENTS != 0:
        raise ValueError("swizzled shared coordinate must be vector aligned")
    return contiguous_element ^ (
        ((orthogonal_element // SWIZZLE_PER_PHASE) % SWIZZLE_MAX_PHASE)
        * SWIZZLE_VECTOR_ELEMENTS)


def _shared_vector_offset(base, contiguous_element, orthogonal_element,
                          contiguous_extent):
    return base + (
        orthogonal_element * contiguous_extent
        + _swizzled_contiguous_element(
            contiguous_element, orthogonal_element)) * ELEMENT_BYTES


def validate_swizzle_geometry():
    """Prove the two frozen TTGIR shared encodings are aligned bijections."""

    streams = (
        ("k_nope_order_01", K_NOPE_SHARED_BASE, VALUE_HEAD_DIM, KV_TILE),
        ("v_order_10", V_SHARED_BASE, VALUE_HEAD_DIM, KV_TILE),
        ("k_rope_order_01", K_ROPE_SHARED_BASE,
         QK_HEAD_DIM - VALUE_HEAD_DIM, KV_TILE),
    )
    audit = {}
    all_offsets = set()
    for name, base, contiguous_extent, orthogonal_extent in streams:
        offsets = tuple(
            _shared_vector_offset(base, contiguous, orthogonal,
                                  contiguous_extent)
            for orthogonal in range(orthogonal_extent)
            for contiguous in range(
                0, contiguous_extent, SWIZZLE_VECTOR_ELEMENTS))
        expected_count = (
            contiguous_extent * orthogonal_extent
            // SWIZZLE_VECTOR_ELEMENTS)
        if len(offsets) != expected_count or len(set(offsets)) != expected_count:
            raise ValueError(f"{name} shared swizzle is not bijective")
        if any(offset % CP_ASYNC_BYTES_PER_THREAD != 0 for offset in offsets):
            raise ValueError(f"{name} shared swizzle breaks 16-byte alignment")
        end = base + contiguous_extent * orthogonal_extent * ELEMENT_BYTES
        if min(offsets) < base or max(offsets) + CP_ASYNC_BYTES_PER_THREAD > end:
            raise ValueError(f"{name} shared swizzle escapes its allocation")
        if all_offsets.intersection(offsets):
            raise ValueError(f"{name} shared allocation aliases another stream")
        all_offsets.update(offsets)
        audit[name] = {
            "base_byte": base,
            "end_byte": end,
            "vector_count": expected_count,
            "vector_bytes": CP_ASYNC_BYTES_PER_THREAD,
            "encoding": "vec=8,perPhase=1,maxPhase=8",
        }
    anchors = {
        "k_nope_feature0_token1": _shared_vector_offset(
            K_NOPE_SHARED_BASE, 0, 1, VALUE_HEAD_DIM),
        "v_token1_value0": _shared_vector_offset(
            V_SHARED_BASE, 0, 1, VALUE_HEAD_DIM),
        "k_rope_feature0_token1": _shared_vector_offset(
            K_ROPE_SHARED_BASE, 0, 1, QK_HEAD_DIM - VALUE_HEAD_DIM),
    }
    expected_anchors = {
        "k_nope_feature0_token1": 1040,
        "v_token1_value0": 33808,
        "k_rope_feature0_token1": 82064,
    }
    if anchors != expected_anchors:
        raise ValueError("frozen TTGIR shared swizzle anchors changed")
    return {"streams": audit, "anchors": anchors}


def validate_ttgir_shared_layout(text):
    if not isinstance(text, str) or not text:
        raise ValueError("TTGIR must be non-empty text")
    required = (
        "#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, "
        "maxPhase = 8, order = [1, 0]}>",
        "#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, "
        "maxPhase = 8, order = [0, 1]}>",
        "!ttg.memdesc<1x512x32xf16, #shared1, #smem, mutable>",
        "!ttg.memdesc<1x64x32xf16, #shared1, #smem, mutable>",
        "!ttg.memdesc<1x32x512xf16, #shared, #smem, mutable>",
    )
    missing = tuple(fragment for fragment in required if fragment not in text)
    if missing:
        raise ValueError("pinned TTGIR shared layout changed")
    return {
        "k_krope_encoding": "vec=8,perPhase=1,maxPhase=8,order=[0,1]",
        "v_encoding": "vec=8,perPhase=1,maxPhase=8,order=[1,0]",
        "geometry": validate_swizzle_geometry(),
    }


def validate_static_manifest(document, manifest_sha256):
    """Bind the probe to static MLA source/compiler facts, never duration."""

    if manifest_sha256 != STATIC_MANIFEST_SHA256:
        raise ValueError("MLA static manifest SHA256 changed")
    if not isinstance(document, dict):
        raise TypeError("MLA static manifest must be an object")
    if document["schema"] != "nvidia_a800_mla_stage1_static_compile_manifest_v1":
        raise ValueError("MLA static manifest schema changed")
    if document["hardware_id"] != HARDWARE_ID:
        raise ValueError("MLA static manifest hardware changed")
    source = document["source"]
    artifacts = document["artifacts"]
    contract = document["static_contract"]
    boundary = document["evidence_boundary"]
    if source["vllm_module_sha256"] != VLLM_SOURCE_SHA256:
        raise ValueError("pinned vLLM MLA source changed")
    expected_artifacts = {
        "cubin_sha256": STATIC_CUBIN_SHA256,
        "ttgir_sha256": STATIC_TTGIR_SHA256,
        "metadata_sha256": STATIC_METADATA_SHA256,
    }
    if artifacts != expected_artifacts:
        raise ValueError("pinned MLA compiler artifacts changed")
    expected_contract = {
        "block_n": KV_TILE,
        "block_h": 16,
        "num_warps": WARPS_PER_CTA,
        "threads_per_cta": BLOCK_THREADS,
        "num_stages": PIPELINE_STAGES,
        "block_dmodel": VALUE_HEAD_DIM,
        "block_dpe": QK_HEAD_DIM - VALUE_HEAD_DIM,
        "block_dv": VALUE_HEAD_DIM,
        "dynamic_shared_bytes": 89088,
        "registers_per_thread": 255,
        "per_step_k_nope_bytes": K_NOPE_BYTES,
        "per_step_k_rope_bytes": K_ROPE_BYTES,
        "per_step_v_bytes": V_BYTES,
        "per_step_async_copy_bytes": STAGE_PAYLOAD_BYTES,
        "async_copy_opcode": "LDGSTS.E.BYPASS.128",
        "static_ldgsts_sites": 68,
    }
    if contract != expected_contract:
        raise ValueError("pinned grouped-MLA static loader contract changed")
    if boundary["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError("MLA static evidence consumed a forbidden duration")
    return {"source": source, "artifacts": artifacts, "static_contract": contract}


def validate_static_artifact_files(manifest_path, document):
    """Verify the files named by the pinned compile manifest in place."""

    artifact_relative = Path(document["stage1_artifact_directory"])
    if artifact_relative.is_absolute() or ".." in artifact_relative.parts:
        raise ValueError("MLA static artifact directory must be relative and contained")
    artifact_dir = (manifest_path.parent / artifact_relative).resolve()
    if manifest_path.parent.resolve() not in artifact_dir.parents:
        raise ValueError("MLA static artifact directory escapes its capture root")
    expected = {
        "_fwd_grouped_kernel_stage1.cubin": STATIC_CUBIN_SHA256,
        "_fwd_grouped_kernel_stage1.ttgir": STATIC_TTGIR_SHA256,
        "_fwd_grouped_kernel_stage1.json": STATIC_METADATA_SHA256,
    }
    observed = {}
    for name, digest in expected.items():
        path = artifact_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[name] = _sha256(path)
        if observed[name] != digest:
            raise ValueError(f"pinned MLA static artifact changed: {name}")
    observed["ttgir_shared_layout"] = validate_ttgir_shared_layout(
        (artifact_dir / "_fwd_grouped_kernel_stage1.ttgir").read_text(
            encoding="utf-8"))
    module = Path(document["source"]["vllm_module"])
    if not module.is_file():
        raise FileNotFoundError(module)
    observed["vllm_module"] = _sha256(module)
    if observed["vllm_module"] != VLLM_SOURCE_SHA256:
        raise ValueError("installed pinned vLLM MLA source changed")
    return observed


def validate_static_snapshot_files(manifest_path, document):
    """Verify the self-contained compiler artifacts without installed vLLM."""

    if _sha256(manifest_path) != STATIC_MANIFEST_SHA256:
        raise ValueError("snapshotted MLA static manifest changed")
    validate_static_manifest(document, STATIC_MANIFEST_SHA256)
    artifact_relative = Path(document["stage1_artifact_directory"])
    if artifact_relative.is_absolute() or ".." in artifact_relative.parts:
        raise ValueError("MLA static artifact directory must be relative and contained")
    artifact_dir = (manifest_path.parent / artifact_relative).resolve()
    if manifest_path.parent.resolve() not in artifact_dir.parents:
        raise ValueError("MLA static artifact directory escapes its snapshot")
    expected = {
        "_fwd_grouped_kernel_stage1.cubin": STATIC_CUBIN_SHA256,
        "_fwd_grouped_kernel_stage1.ttgir": STATIC_TTGIR_SHA256,
        "_fwd_grouped_kernel_stage1.json": STATIC_METADATA_SHA256,
    }
    observed = {}
    for name, digest in expected.items():
        path = artifact_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[name] = _sha256(path)
        if observed[name] != digest:
            raise ValueError(f"snapshotted MLA static artifact changed: {name}")
    observed["ttgir_shared_layout"] = validate_ttgir_shared_layout(
        (artifact_dir / "_fwd_grouped_kernel_stage1.ttgir").read_text(
            encoding="utf-8"))
    observed["vllm_module"] = VLLM_SOURCE_SHA256
    observed["manifest_sha256"] = STATIC_MANIFEST_SHA256
    return observed


def _sass_kernel_body(text):
    if not isinstance(text, str) or not text:
        raise ValueError("SASS must be non-empty text")
    match = re.search(
        r"Function\s*:\s*two_stage_ldgsts_loader\b(?P<body>.*?)(?=\n\s*Function\s*:|\Z)",
        text, flags=re.DOTALL)
    if match is None:
        raise ValueError("SASS lacks two_stage_ldgsts_loader")
    return match.group("body")


def validate_sass(text):
    body = _sass_kernel_body(text)
    ldgsts = tuple(re.findall(r"\bLDGSTS(?:\.[A-Z0-9]+)*\b", body))
    expected_sites = PIPELINE_STAGES * COPIES_PER_THREAD_PER_STAGE
    if len(ldgsts) != expected_sites:
        raise ValueError(
            "probe SASS must contain 68 prologue/body copy sites, got "
            f"{len(ldgsts)}")
    if any(opcode != "LDGSTS.E.BYPASS.128" for opcode in ldgsts):
        raise ValueError("probe SASS changed the LDGSTS cache/width policy")
    destinations = tuple(re.findall(
        r"\bLDGSTS\.E\.BYPASS\.128\s+\[R([0-9]+)(?:\+(0x[0-9a-fA-F]+))?\]",
        body))
    if len(destinations) != expected_sites:
        raise ValueError("probe SASS LDGSTS destination syntax changed")
    immediates = tuple(
        int(immediate_text, 16) if immediate_text else 0
        for _, immediate_text in destinations)
    for stage in range(PIPELINE_STAGES):
        begin = stage * COPIES_PER_THREAD_PER_STAGE
        k_sites = immediates[begin:begin + 16]
        rope_sites = immediates[begin + 16:begin + 18]
        v_sites = immediates[begin + 18:begin + 34]
        if any(value not in (0, 0x10) for value in k_sites):
            raise ValueError("probe SASS K-nope shared base changed")
        # ptxas may fold the 0x14000 base into the two body address registers;
        # the prologue must retain it and the source/TTGIR gates prove both.
        if any(value not in (
                0, 0x10, K_ROPE_SHARED_BASE,
                K_ROPE_SHARED_BASE + 0x10) for value in rope_sites):
            raise ValueError("probe SASS K-rope shared base changed")
        if any(value not in (
                V_SHARED_BASE, V_SHARED_BASE + 0x10) for value in v_sites):
            raise ValueError("probe SASS V shared base changed")
    explicit_rope_sites = sum(
        value in (K_ROPE_SHARED_BASE, K_ROPE_SHARED_BASE + 0x10)
        for value in immediates)
    if explicit_rope_sites < 2:
        raise ValueError("probe SASS lacks the explicit K-rope shared base")
    base_counts = {"k_nope": 32, "v": 32, "k_rope": 4}
    if len({register for register, _ in destinations}) < 16:
        raise ValueError("probe SASS collapsed the swizzled destination address set")
    commits = len(re.findall(r"\bLDGDEPBAR\b", body))
    if commits != PIPELINE_STAGES * STREAMS_PER_STAGE:
        raise ValueError(
            "probe SASS must contain three prologue and three body commits")
    waits_zero = len(re.findall(r"DEPBAR\.LE\s+SB0,\s*0x0\b", body))
    if waits_zero != 2:
        raise ValueError("probe SASS changed the two-stage wait/drain structure")
    wait_positions = tuple(
        match.start() for match in re.finditer(
            r"DEPBAR\.LE\s+SB0,\s*0x0\b", body))
    clock_reads = len(re.findall(r"SR_CLOCKLO\b", body))
    if clock_reads != 2:
        raise ValueError("probe must contain exactly two clock64 reads")
    clock_positions = tuple(
        match.start() for match in re.finditer(r"SR_CLOCKLO\b", body))
    ldgsts_positions = tuple(
        match.start() for match in re.finditer(
            r"\bLDGSTS\.E\.BYPASS\.128\b", body))
    if not (clock_positions[0] < ldgsts_positions[0]
            and ldgsts_positions[-1] < wait_positions[-1]
            < clock_positions[1]):
        raise ValueError("clock64 reads do not bracket the loader instruction stream")
    timed_body = body[clock_positions[0]:clock_positions[1]]
    if re.search(r"\bBAR(?:\.SYNC)?\b", timed_body):
        raise ValueError("probe timed interval contains a CTA barrier")
    if re.search(r"\b(?:LDL|STL)(?:\.|\s)", body):
        raise ValueError("probe spills into local memory")
    if re.search(r"\bLDG(?:\.|\s)", body):
        raise ValueError("probe contains ordinary global loads in its target kernel")
    return {
        "ldgsts_static_sites": len(ldgsts),
        "ldgsts_opcode": "LDGSTS.E.BYPASS.128",
        "ldgsts_shared_base_site_counts": base_counts,
        "explicit_k_rope_base_immediate_sites": explicit_rope_sites,
        "ldgsts_distinct_destination_registers": len({
            register for register, _ in destinations}),
        "async_commit_sites": commits,
        "wait_group_zero_sites": waits_zero,
        "clock64_reads": clock_reads,
        "timed_cta_barriers": 0,
        "local_memory_instructions": 0,
        "ordinary_global_load_instructions": 0,
    }


def validate_reuse_sass(text):
    if not isinstance(text, str) or not text:
        raise ValueError("reuse SASS must be non-empty text")
    match = re.search(
        r"Function\s*:\s*grouped_mla_kv_reuse_loader\b"
        r"(?P<body>.*?)(?=\n\s*Function\s*:|\Z)",
        text, flags=re.DOTALL)
    if match is None:
        raise ValueError("SASS lacks grouped_mla_kv_reuse_loader")
    body = match.group("body")
    ldgsts = tuple(re.findall(r"\bLDGSTS(?:\.[A-Z0-9]+)*\b", body))
    expected_sites = PIPELINE_STAGES * COPIES_PER_THREAD_PER_STAGE
    if len(ldgsts) != expected_sites:
        raise ValueError(
            "reuse SASS must contain the prologue/body LDGSTS sites")
    if any(opcode != "LDGSTS.E.BYPASS.128" for opcode in ldgsts):
        raise ValueError("reuse SASS changed the LDGSTS cache/width policy")
    destinations = tuple(re.findall(
        r"\bLDGSTS\.E\.BYPASS\.128\s+\[R([0-9]+)(?:\+(0x[0-9a-fA-F]+))?\]",
        body))
    if len(destinations) != expected_sites:
        raise ValueError("reuse SASS LDGSTS destination syntax changed")
    immediates = tuple(
        int(immediate_text, 16) if immediate_text else 0
        for _, immediate_text in destinations)
    for stage in range(PIPELINE_STAGES):
        begin = stage * COPIES_PER_THREAD_PER_STAGE
        k_sites = immediates[begin:begin + 16]
        rope_sites = immediates[begin + 16:begin + 18]
        v_sites = immediates[begin + 18:begin + 34]
        if any(value not in (0, 0x10) for value in k_sites):
            raise ValueError("reuse SASS K-nope shared base changed")
        if any(value not in (
                0, 0x10, K_ROPE_SHARED_BASE,
                K_ROPE_SHARED_BASE + 0x10) for value in rope_sites):
            raise ValueError("reuse SASS K-rope shared base changed")
        if any(value not in (
                V_SHARED_BASE, V_SHARED_BASE + 0x10) for value in v_sites):
            raise ValueError("reuse SASS V shared base changed")
    explicit_rope_sites = sum(
        value in (K_ROPE_SHARED_BASE, K_ROPE_SHARED_BASE + 0x10)
        for value in immediates)
    if explicit_rope_sites < 2:
        raise ValueError("reuse SASS lacks the explicit K-rope shared base")
    distinct_destinations = len({register for register, _ in destinations})
    if distinct_destinations < 16:
        raise ValueError("reuse SASS collapsed the swizzled destination set")
    commits = len(re.findall(r"\bLDGDEPBAR\b", body))
    waits_zero = len(re.findall(r"DEPBAR\.LE\s+SB0,\s*0x0\b", body))
    if commits != PIPELINE_STAGES * STREAMS_PER_STAGE or waits_zero != 2:
        raise ValueError("reuse SASS changed the two-stage three-stream schedule")
    if re.search(r"\b(?:LDL|STL)(?:\.|\s)", body):
        raise ValueError("reuse probe spills into local memory")
    if re.search(r"\bLDG(?:\.|\s)", body):
        raise ValueError("reuse probe contains ordinary global loads")
    return {
        "ldgsts_static_sites": len(ldgsts),
        "ldgsts_opcode": "LDGSTS.E.BYPASS.128",
        "ldgsts_shared_base_site_counts": {
            "k_nope": 32, "v": 32, "k_rope": 4},
        "explicit_k_rope_base_immediate_sites": explicit_rope_sites,
        "ldgsts_distinct_destination_registers": distinct_destinations,
        "async_commit_sites": commits,
        "wait_group_zero_sites": waits_zero,
        "local_memory_instructions": 0,
        "ordinary_global_load_instructions": 0,
    }


def validate_ptxas(text):
    if not isinstance(text, str) or not text:
        raise ValueError("PTXAS output must be non-empty text")
    if "Function properties for two_stage_ldgsts_loader" not in text:
        raise ValueError("PTXAS output lacks two_stage_ldgsts_loader")
    spills = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", text)
    if not spills or any(record != ("0", "0") for record in spills):
        raise ValueError("two-stage LDGSTS probe has register spills")


def _parse_ncu_dimension(text, label):
    match = re.fullmatch(
        r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)",
        _require_nonempty(text, label))
    if match is None:
        raise ValueError(f"{label} has invalid syntax")
    return tuple(int(match.group(index)) for index in range(1, 4))


def _validate_ncu_launch(row, grid):
    if row["Kernel Name"] != "two_stage_ldgsts_loader":
        raise ValueError("NCU selected the wrong kernel")
    if _parse_ncu_dimension(row["Grid Size"], "NCU grid") != (grid, 1, 1):
        raise ValueError("NCU grid does not match the selected cell")
    if _parse_ncu_dimension(row["Block Size"], "NCU block") != (
            BLOCK_THREADS, 1, 1):
        raise ValueError("NCU block does not match the static contract")


def parse_ncu_csv(text, grid):
    lines = text.splitlines()
    header_index = next(
        (index for index, line in enumerate(lines)
         if line.startswith('"ID","Process ID"')), None)
    if header_index is None:
        raise ValueError("NCU CSV lacks its metric header")
    reader = csv.DictReader(lines[header_index:])
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("NCU CSV header is absent or duplicated")
    rows = tuple(reader)
    fields = set(reader.fieldnames)
    launch_fields = {"Kernel Name", "Grid Size", "Block Size"}
    tall_fields = launch_fields | {"Metric Name", "Metric Unit", "Metric Value"}
    wide_fields = launch_fields | set(NCU_METRICS)
    if wide_fields <= fields:
        units = tuple(row for row in rows if not row["Kernel Name"])
        data = tuple(row for row in rows if row["Kernel Name"])
        if len(units) != 1 or len(data) != 1:
            raise ValueError("wide NCU CSV requires one unit row and one data row")
        _validate_ncu_launch(data[0], grid)
        metrics = {}
        for metric in NCU_METRICS:
            if units[0][metric] != NCU_METRIC_UNITS[metric]:
                raise ValueError(f"NCU {metric} unit changed")
            metrics[metric] = _parse_uint(data[0][metric], f"NCU {metric}")
    elif tall_fields <= fields:
        metrics = {}
        for row in rows:
            _validate_ncu_launch(row, grid)
            metric = row["Metric Name"]
            if metric not in NCU_METRIC_UNITS or metric in metrics:
                raise ValueError(f"NCU metric identity is invalid: {metric!r}")
            if row["Metric Unit"] != NCU_METRIC_UNITS[metric]:
                raise ValueError(f"NCU {metric} unit changed")
            metrics[metric] = _parse_uint(row["Metric Value"], f"NCU {metric}")
    else:
        raise ValueError("NCU CSV lacks the required tall or wide schema")
    if set(metrics) != set(NCU_METRICS):
        raise ValueError("NCU metric schema mismatch")
    return metrics


def validate_ncu_contract(grid, tiles, metrics):
    counts = _expected_counts(grid, tiles)
    instruction_metric = NCU_METRICS[0]
    xbar_metric = NCU_METRICS[1]
    l2_sector_metric = NCU_METRICS[2]
    l2_hit_metric = NCU_METRICS[3]
    l2_miss_metric = NCU_METRICS[4]
    dram_metric = NCU_METRICS[5]
    if metrics[instruction_metric] != counts["ldgsts_warp_instructions"]:
        raise ValueError("NCU dynamic LDGSTS count disagrees with source geometry")
    if metrics[xbar_metric] != counts["expected_xbar_bytes"]:
        raise ValueError("NCU Xbar bytes do not prove the 32B transaction mapping")
    expected_l2_sectors = counts["payload_bytes"] // L2_SECTOR_BYTES
    l2_sectors = metrics[l2_sector_metric]
    l2_hits = metrics[l2_hit_metric]
    l2_misses = metrics[l2_miss_metric]
    if l2_sectors != expected_l2_sectors:
        raise ValueError("NCU L2 sectors disagree with the LDGSTS payload")
    if l2_hits + l2_misses != l2_sectors:
        raise ValueError("NCU L2 hit/miss sectors do not conserve requests")
    # With cache-control disabled, three untimed launches precede the target.
    # Admit at most one 128-B L2 line per CTA as counter-granularity noise.
    # This is 576x smaller than the two-tile cold source footprint per CTA and
    # cannot silently accept a cold refill.
    maximum_miss_sectors = grid * (L2_LINE_BYTES // L2_SECTOR_BYTES)
    maximum_dram_bytes = grid * L2_LINE_BYTES
    if l2_misses > maximum_miss_sectors or metrics[dram_metric] > maximum_dram_bytes:
        raise ValueError("NCU shows that the loader cell was not hot in L2")
    if metrics[dram_metric] % L2_SECTOR_BYTES != 0:
        raise ValueError("NCU DRAM bytes violate the 32-byte counter granularity")
    return {
        "payload_bytes_per_ldgsts_warp_instruction": 512,
        "xbar_bytes_per_ldgsts_warp_instruction": (
            XBAR_BYTES_PER_LDGSTS_WARP_INSTRUCTION),
        "xbar_32b_transactions_per_ldgsts_warp_instruction": 32,
        "expected_l2_read_sectors": expected_l2_sectors,
        "l2_read_hit_sectors": l2_hits,
        "l2_read_miss_sectors": l2_misses,
        "l2_read_hit_rate": l2_hits / l2_sectors,
        "maximum_admitted_l2_miss_sectors": maximum_miss_sectors,
        "maximum_admitted_dram_bytes": maximum_dram_bytes,
        "warm_bound_basis": "one_128B_L2_line_per_CTA_counter_granularity",
    }


def _reuse_cases():
    cases = []
    for tiles_per_split in REUSE_TILES_PER_SPLIT:
        synthetic_unique_bytes_per_split = (
            REUSE_ACTIVE_SOURCES * tiles_per_split * SOURCE_TILE_BYTES)
        case_id = f"a{REUSE_ACTIVE_SOURCES}_t{tiles_per_split}"
        cases.append({
            "case_id": case_id,
            "active_sources": REUSE_ACTIVE_SOURCES,
            "tiles_per_split": tiles_per_split,
            "head_group_count": HEAD_GROUP_COUNT,
            "num_kv_splits": NUM_KV_SPLITS,
            "logical_ctas": (
                REUSE_ACTIVE_SOURCES * HEAD_GROUP_COUNT * NUM_KV_SPLITS),
            "grid_x": REUSE_ACTIVE_SOURCES,
            "grid_y": HEAD_GROUP_COUNT,
            "grid_z": NUM_KV_SPLITS,
            "synthetic_unique_address_bytes_per_split": (
                synthetic_unique_bytes_per_split),
            "synthetic_unique_address_cache_lines_per_split": (
                synthetic_unique_bytes_per_split // L2_LINE_BYTES),
            "allocated_unique_backing_bytes": (
                synthetic_unique_bytes_per_split * NUM_KV_SPLITS),
            "allocated_unique_backing_cache_lines": (
                synthetic_unique_bytes_per_split * NUM_KV_SPLITS
                // L2_LINE_BYTES),
            "requested_k_nope_k_rope_bytes": (
                REUSE_ACTIVE_SOURCES * HEAD_GROUP_COUNT * NUM_KV_SPLITS
                * tiles_per_split * SOURCE_TILE_BYTES),
            "requested_v_bytes": (
                REUSE_ACTIVE_SOURCES * HEAD_GROUP_COUNT * NUM_KV_SPLITS
                * tiles_per_split * V_BYTES),
            "requested_source_bytes": (
                REUSE_ACTIVE_SOURCES * HEAD_GROUP_COUNT * NUM_KV_SPLITS
                * tiles_per_split * STAGE_PAYLOAD_BYTES),
        })
    return tuple(cases)


def build_reuse_case_manifest(*, baseline_commit, source_sha256,
                              runner_sha256):
    for label, value, pattern in (
            ("baseline commit", baseline_commit, r"[0-9a-f]{40}"),
            ("source SHA256", source_sha256, r"[0-9a-f]{64}"),
            ("runner SHA256", runner_sha256, r"[0-9a-f]{64}")):
        if re.fullmatch(pattern, value) is None:
            raise ValueError(f"reuse {label} has invalid syntax")
    cases = _reuse_cases()
    return {
        "schema": "nvidia_a800_grouped_mla_kv_reuse_case_manifest_v1",
        "hardware_id": HARDWARE_ID,
        "mechanism": "grouped_mla_kv_source_reuse",
        "code_provenance": {
            "baseline_commit": baseline_commit,
            "source_sha256": source_sha256,
            "runner_sha256": runner_sha256,
            "grouped_mla_static_manifest_sha256": STATIC_MANIFEST_SHA256,
        },
        "static_access_contract": {
            "block_h": BLOCK_H,
            "block_n": KV_TILE,
            "query_heads": BLOCK_H * HEAD_GROUP_COUNT,
            "head_group_count": HEAD_GROUP_COUNT,
            "num_kv_splits": NUM_KV_SPLITS,
            "threads_per_cta": BLOCK_THREADS,
            "pipeline_stages": PIPELINE_STAGES,
            "k_nope_source_bytes_per_tile": K_NOPE_BYTES,
            "k_rope_source_bytes_per_tile": K_ROPE_BYTES,
            "k_nope_k_rope_source_bytes_per_tile": SOURCE_TILE_BYTES,
            "v_source_bytes_per_tile": V_BYTES,
            "requested_source_bytes_per_tile": STAGE_PAYLOAD_BYTES,
            "unique_aliased_backing_bytes_per_tile": SOURCE_TILE_BYTES,
            "k_v_source_addresses_alias": True,
            "ldgsts_warp_instructions_per_tile": (
                LDGSTS_WARP_INSTRUCTIONS_PER_STAGE),
        },
        "cuda_grid_index_mapping": {
            "cuda_grid": "(active_sources, head_group_count, num_kv_splits)",
            "linearization": "x_fastest_then_y_then_z",
            "one_dimensional_equivalent": (
                "logical_cta=((kv_split*head_group_count+head_group)*"
                "active_sources)+source_index"),
            "source_segment": "(kv_split,source_index)",
            "reuse_statement": (
                "four head-group CTAs replay the same source segment within "
                "each KV split at the logical address level"),
            "scheduler_entry_order_observed": False,
            "scheduler_entry_order_claimed": False,
            "scheduler_entry_order_status": (
                "not_observed_requires_independent_entry_probe"),
        },
        "synthetic_address_sweep": {
            "active_sources": REUSE_ACTIVE_SOURCES,
            "tiles_per_split": list(REUSE_TILES_PER_SPLIT),
            "runtime_active_residency_observed": False,
            "runtime_active_residency_status": (
                "not_observed_requires_independent_residency_probe"),
            "l2_capacity_transition_claimed": False,
            "interpretation": (
                "exact launch-wide aggregate counters only; CUDA grid index "
                "mapping does not establish CTA entry or residency order"),
        },
        "counter_contract": {
            "metrics": list(REUSE_NCU_METRICS),
            "l2_counter_unit_bytes": L2_SECTOR_BYTES,
            "reported_cache_line_bytes": L2_LINE_BYTES,
            "l2_source_read_bytes_conservation_required": True,
            "l2_lookup_attempts_are_not_source_read_sectors": True,
            "dram_read_counter_unit": "byte",
            "ncu_cache_control": "all",
            "ncu_clock_control": "none",
        },
        "cases": list(cases),
        "evidence_boundary": {
            "evidence_status": "exact_launch_aggregate_only",
            "loader_only": True,
            "synthetic_address_footprint": True,
            "production_or_evaluation_duration_consumed": False,
            "duration_metric_requested": False,
            "forward_parameter_published": False,
        },
    }


def _validate_reuse_ncu_launch(row, case):
    if row["Kernel Name"] != "grouped_mla_kv_reuse_loader":
        raise ValueError("reuse NCU selected the wrong kernel")
    if _parse_ncu_dimension(row["Grid Size"], "reuse NCU grid") != (
            case["grid_x"], case["grid_y"], case["grid_z"]):
        raise ValueError("reuse NCU grid does not match the case manifest")
    if _parse_ncu_dimension(row["Block Size"], "reuse NCU block") != (
            BLOCK_THREADS, 1, 1):
        raise ValueError("reuse NCU block does not match the static contract")


def parse_reuse_ncu_csv(text, case):
    lines = text.splitlines()
    header_index = next(
        (index for index, line in enumerate(lines)
         if tuple(next(csv.reader([line])))[:2] == ("ID", "Process ID")),
        None)
    if header_index is None:
        raise ValueError("reuse NCU CSV lacks its metric header")
    parsed = tuple(csv.reader(lines[header_index:]))
    if not parsed or len(parsed[0]) != len(set(parsed[0])):
        raise ValueError("reuse NCU CSV header is absent or duplicated")
    header = parsed[0]
    tall_fields = {
        "Kernel Name", "Grid Size", "Block Size", "Metric Name",
        "Metric Unit", "Metric Value",
    }
    metrics = {}
    if tall_fields.issubset(set(header)):
        for values in parsed[1:]:
            if len(values) != len(header):
                raise ValueError("reuse NCU tall row width changed")
            row = dict(zip(header, values))
            _validate_reuse_ncu_launch(row, case)
            metric = row["Metric Name"]
            if metric not in REUSE_NCU_METRIC_UNITS or metric in metrics:
                raise ValueError(
                    f"reuse NCU metric identity is invalid: {metric!r}")
            if row["Metric Unit"] != REUSE_NCU_METRIC_UNITS[metric]:
                raise ValueError(f"reuse NCU {metric} unit changed")
            metrics[metric] = _parse_uint(
                row["Metric Value"], f"reuse NCU {metric}")
    else:
        required = {
            "Kernel Name", "Grid Size", "Block Size", *REUSE_NCU_METRICS}
        if not required.issubset(set(header)) or len(parsed) != 3:
            raise ValueError("reuse NCU lacks one supported metric record")
        units = dict(zip(header, parsed[1]))
        row = dict(zip(header, parsed[2]))
        if len(parsed[1]) != len(header) or len(parsed[2]) != len(header):
            raise ValueError("reuse NCU wide row width changed")
        _validate_reuse_ncu_launch(row, case)
        for metric in REUSE_NCU_METRICS:
            if units[metric] != REUSE_NCU_METRIC_UNITS[metric]:
                raise ValueError(f"reuse NCU {metric} unit changed")
            metrics[metric] = _parse_uint(
                row[metric], f"reuse NCU {metric}")
    if set(metrics) != set(REUSE_NCU_METRICS):
        raise ValueError("reuse NCU metric schema mismatch")
    return metrics


def validate_reuse_ncu_contract(case, metrics):
    if case not in _reuse_cases():
        raise ValueError(
            "reuse case is outside the frozen synthetic address sweep")
    if set(metrics) != set(REUSE_NCU_METRICS):
        raise ValueError("reuse NCU metric schema mismatch")
    instruction_metric = REUSE_NCU_METRICS[0]
    l2_read_metric = REUSE_NCU_METRICS[1]
    l2_hit_metric = REUSE_NCU_METRICS[2]
    l2_miss_metric = REUSE_NCU_METRICS[3]
    dram_read_metric = REUSE_NCU_METRICS[4]
    expected_instructions = (
        case["logical_ctas"] * case["tiles_per_split"]
        * LDGSTS_WARP_INSTRUCTIONS_PER_STAGE)
    if metrics[instruction_metric] != expected_instructions:
        raise ValueError(
            "reuse NCU dynamic LDGSTS count disagrees with static geometry")
    expected_l2_sectors = (
        case["requested_source_bytes"] // L2_SECTOR_BYTES)
    l2_read_sectors = metrics[l2_read_metric]
    l2_hit_sectors = metrics[l2_hit_metric]
    l2_miss_sectors = metrics[l2_miss_metric]
    if l2_read_sectors != expected_l2_sectors:
        raise ValueError("reuse NCU L2 reads disagree with requested source bytes")
    lookup_sectors = l2_hit_sectors + l2_miss_sectors
    if lookup_sectors <= 0:
        raise ValueError("reuse NCU L2 lookup counters are empty")
    dram_read_bytes = metrics[dram_read_metric]
    if dram_read_bytes % L2_SECTOR_BYTES != 0:
        raise ValueError("reuse DRAM bytes violate 32-byte counter granularity")
    return {
        "case_id": case["case_id"],
        "synthetic_unique_address_bytes_per_split": (
            case["synthetic_unique_address_bytes_per_split"]),
        "allocated_unique_backing_bytes": case["allocated_unique_backing_bytes"],
        "requested_k_nope_k_rope_bytes": (
            case["requested_k_nope_k_rope_bytes"]),
        "requested_v_bytes": case["requested_v_bytes"],
        "requested_source_bytes": case["requested_source_bytes"],
        "l2_source_read_sectors": l2_read_sectors,
        "l2_lookup_hit_sectors": l2_hit_sectors,
        "l2_lookup_miss_sectors": l2_miss_sectors,
        "l2_lookup_attempt_sectors": lookup_sectors,
        "l2_source_read_cache_line_equivalents": (
            l2_read_sectors * L2_SECTOR_BYTES / L2_LINE_BYTES),
        "l2_lookup_hit_cache_line_equivalents": (
            l2_hit_sectors * L2_SECTOR_BYTES / L2_LINE_BYTES),
        "l2_lookup_miss_cache_line_equivalents": (
            l2_miss_sectors * L2_SECTOR_BYTES / L2_LINE_BYTES),
        "l2_lookup_attempt_cache_line_equivalents": (
            lookup_sectors * L2_SECTOR_BYTES / L2_LINE_BYTES),
        "l2_lookup_attempt_amplification": lookup_sectors / l2_read_sectors,
        "l2_lookup_hit_fraction": l2_hit_sectors / lookup_sectors,
        "dram_read_bytes": dram_read_bytes,
        "dram_read_cache_line_equivalents": dram_read_bytes / L2_LINE_BYTES,
    }


def qualify_steady_state(native):
    if native["meta"]["capture_mode"] != "full_matrix":
        return {
            "qualified": False,
            "reason": "full native grid/tile matrix is required",
            "minimum_trials_per_cell": MIN_CANDIDATE_TRIALS_PER_CELL,
            "observed_trials_per_cell": {},
            "trial_relative_spread": {},
            "tile16_to_tile64_relative_delta": {},
        }
    trial_spread = {}
    trial_counts = {}
    for grid, tiles in _expected_cells("full_matrix"):
        rates = tuple(
            record["rate"] for key, record in native["raw"].items()
            if key[0] == grid and key[1] == tiles)
        median = native["summary"][(grid, tiles)]
        trial_counts[f"g{grid}_t{tiles}"] = len(rates)
        trial_spread[f"g{grid}_t{tiles}"] = (max(rates) - min(rates)) / median
    plateau = {}
    for grid in GRID_POINTS:
        tile16 = native["summary"][(grid, 16)]
        tile64 = native["summary"][(grid, 64)]
        plateau[f"g{grid}"] = abs(tile16 - tile64) / tile64
    qualified = (
        min(trial_counts.values()) >= MIN_CANDIDATE_TRIALS_PER_CELL
        and len(set(trial_counts.values())) == 1
        and max(trial_spread.values()) <= MAX_TRIAL_RELATIVE_SPREAD
        and max(plateau.values()) <= MAX_PLATEAU_RELATIVE_DELTA)
    return {
        "qualified": qualified,
        "reason": (
            "all cells satisfy repeated-trial dispersion and tile16/tile64 "
            "service plateau gates" if qualified else
            "repeated-trial dispersion or tile16/tile64 plateau is not proven"),
        "maximum_trial_relative_spread": MAX_TRIAL_RELATIVE_SPREAD,
        "maximum_plateau_relative_delta": MAX_PLATEAU_RELATIVE_DELTA,
        "minimum_trials_per_cell": MIN_CANDIDATE_TRIALS_PER_CELL,
        "observed_trials_per_cell": trial_counts,
        "trial_relative_spread": trial_spread,
        "tile16_to_tile64_relative_delta": plateau,
    }


def build_candidate(native, ncu_audit, raw_sha256, static_manifest_sha256):
    if re.fullmatch(r"[0-9a-f]{64}", raw_sha256) is None:
        raise ValueError("raw SHA256 must contain 64 lowercase hexadecimal digits")
    if static_manifest_sha256 != STATIC_MANIFEST_SHA256:
        raise ValueError("candidate static manifest SHA256 changed")
    mode = native["meta"]["capture_mode"]
    if mode == "full_matrix":
        expected_ncu_cells = NCU_CELLS
    elif mode == "smoke_cell":
        expected_ncu_cells = (SMOKE_CELL,)
    else:
        raise ValueError(f"unknown candidate capture mode {mode!r}")
    expected_ncu_names = {
        f"g{grid}_t{tiles}" for grid, tiles in expected_ncu_cells}
    if set(ncu_audit) != expected_ncu_names:
        raise ValueError("candidate lacks the frozen NCU cell matrix")
    for grid, tiles in expected_ncu_cells:
        name = f"g{grid}_t{tiles}"
        record = ncu_audit[name]
        if record["grid"] != grid or record["tiles"] != tiles:
            raise ValueError(f"candidate NCU identity changed for {name}")
        expected_geometry = validate_ncu_contract(
            grid, tiles, record["metrics"])
        if record["counter_geometry"] != expected_geometry:
            raise ValueError(f"candidate NCU geometry changed for {name}")
    if mode == "full_matrix":
        status = "candidate"
    elif mode == "smoke_cell":
        status = "smoke_only"
    else:
        raise ValueError(f"unknown candidate capture mode {mode!r}")
    rates = {
        f"g{grid}_t{tiles}": native["summary"][(grid, tiles)]
        for grid, tiles in sorted(native["summary"])
    }
    payload_rates = {
        name: rate * XBAR_TRANSACTION_BYTES / 2.0
        for name, rate in rates.items()
    }
    long_tile = max(TILE_POINTS)
    parameters = None
    steady_state = qualify_steady_state(native)
    if status == "candidate" and not steady_state["qualified"]:
        status = "diagnostic_only"
    if status == "candidate" and steady_state["qualified"]:
        single = native["summary"][(1, long_tile)]
        scaling_values = [
            native["summary"][(grid, long_tile)]
            for grid in (2, 8, 32)]
        scaling = sorted(scaling_values)[len(scaling_values) // 2]
        full_per_sm = native["summary"][(108, long_tile)]
        parameters = {
            "single_sm_xbar_32b_transactions_per_sm_cycle": single,
            "scaling_region_xbar_32b_transactions_per_active_sm_cycle": scaling,
            "full_chip_xbar_32b_transactions_per_cycle": full_per_sm * 108,
            "selection_rule": (
                "tile64 steady-service point; grid1 is single-SM, the fixed "
                "median of grids 2/8/32 is the scaling region, and grid108 "
                "times 108 is full-chip service"),
        }
    return {
        "schema": "nvidia_a800_two_stage_ldgsts_candidate_v1",
        "evidence_status": status,
        "mechanism": "grouped_mla_two_stage_l2_resident_ldgsts_loader_service",
        "raw_sha256": raw_sha256,
        "static_manifest_sha256": static_manifest_sha256,
        "static_structure": {
            "threads_per_cta": BLOCK_THREADS,
            "pipeline_stages": PIPELINE_STAGES,
            "shared_bytes_per_cta": DYNAMIC_SHARED_BYTES,
            "streams_per_stage": (
                "K-nope 32768 B, K-rope 4096 B, V 32768 B"),
            "k_v_source_alias": True,
            "payload_bytes_per_stage": STAGE_PAYLOAD_BYTES,
            "source_bytes_per_stage": SOURCE_TILE_BYTES,
        },
        "rate_unit": RATE_UNIT,
        "payload_rate_unit": PAYLOAD_RATE_UNIT,
        "xbar_counter_geometry": {
            "payload_bytes_per_warp_ldgsts": 512,
            "xbar_bytes_per_warp_ldgsts": 1024,
            "xbar_accounted_32b_transactions_per_warp_ldgsts": 32,
            "note": (
                "the NCU Xbar counter is twice payload bytes for this "
                "LDGSTS source; payload and counter units remain explicit"),
        },
        "rates": rates,
        "payload_rates": payload_rates,
        "steady_state_qualification": steady_state,
        "mechanism_parameters": parameters,
        "applicability": (
            "NVIDIA A800 sm_80; 128-thread one-CTA-per-SM loader; two-stage "
            "three-stream grouped-MLA transaction structure; L2-resident "
            "source; measured interval includes pipeline fill and drain"),
        "allowed_use": (
            "source-bound grouped-MLA LDGSTS ingress service and its finite-"
            "tile fill/drain curve, with downstream L2/DRAM service modeled "
            "separately"),
        "forbidden_use": (
            "generic LSU/L2/DRAM capacity, ordinary LDG/STG, attention "
            "compute duration, per-shape correction, or production-duration fit"),
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
        "ncu_audit": ncu_audit,
    }


def _query_device(nvidia_smi, device_uuid, cwd):
    result = _run(_device_query_argv(nvidia_smi, device_uuid), cwd=cwd)
    rows = tuple(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 11:
        raise ValueError("nvidia-smi returned an unexpected device row")
    row = tuple(value.strip() for value in rows[0])
    return {
        "uuid": row[0], "name": row[1], "pci_bus_id": row[2],
        "driver_version": row[3], "pstate": row[4],
        "memory_used_mib": _parse_uint(row[5], "GPU memory used"),
        "memory_total_mib": _parse_uint(row[6], "GPU memory total"),
        "utilization_gpu_pct": _parse_uint(row[7], "GPU utilization"),
        "utilization_memory_pct": _parse_uint(row[8], "memory utilization"),
        "sm_clock_mhz": _parse_uint(row[9], "SM clock"),
        "memory_clock_mhz": _parse_uint(row[10], "memory clock"),
    }


def _require_device_identity(device, expected_uuid, expected_pci):
    if device["uuid"] != expected_uuid or device["name"] != EXPECTED_DEVICE_NAME:
        raise ValueError("nvidia-smi device identity changed")
    if _normalize_pci_bus_id(device["pci_bus_id"]) != _normalize_pci_bus_id(expected_pci):
        raise ValueError("nvidia-smi PCI identity changed")


def _require_idle(device, expected_uuid, expected_pci):
    _require_device_identity(device, expected_uuid, expected_pci)
    if (device["memory_used_mib"] != 0
            or device["utilization_gpu_pct"] != 0
            or device["utilization_memory_pct"] != 0):
        raise RuntimeError(
            "requested A800 is not idle: "
            f"memory={device['memory_used_mib']} MiB, "
            f"gpu_utilization={device['utilization_gpu_pct']}%, "
            f"memory_utilization={device['utilization_memory_pct']}%")


def _query_compute_processes(nvidia_smi, device_uuid, cwd):
    result = _run(_compute_query_argv(nvidia_smi, device_uuid), cwd=cwd)
    text = result.stdout.strip()
    if not text:
        return []
    if text == "No running processes found":
        return []
    rows = tuple(csv.reader(text.splitlines(), skipinitialspace=True))
    processes = []
    for row in rows:
        if len(row) != 4:
            raise ValueError("nvidia-smi returned an unexpected compute-app row")
        uuid, pid, name, used_memory = (value.strip() for value in row)
        if uuid != device_uuid:
            raise ValueError("nvidia-smi compute-app device identity changed")
        processes.append({
            "gpu_uuid": uuid,
            "pid": _parse_uint(pid, "compute process PID"),
            "process_name": _require_nonempty(name, "compute process name"),
            "used_gpu_memory_mib": _parse_uint(
                used_memory, "compute process memory"),
        })
    return processes


def _execution_state(nvidia_smi, device_uuid, cwd):
    return {
        "unix_s": time.time(),
        "device": _query_device(nvidia_smi, device_uuid, cwd),
        "compute_processes": _query_compute_processes(
            nvidia_smi, device_uuid, cwd),
    }


def _require_known_sm_clock(device, expected_sm_clock_mhz):
    if expected_sm_clock_mhz is None:
        observed = device["sm_clock_mhz"]
        if (isinstance(observed, bool) or not isinstance(observed, int)
                or observed <= 0):
            raise RuntimeError(
                "A800 counter-only capture reported an invalid SM clock")
        return
    if (isinstance(expected_sm_clock_mhz, bool)
            or not isinstance(expected_sm_clock_mhz, int)
            or expected_sm_clock_mhz <= 0):
        raise ValueError("expected SM clock must be a positive integer MHz value")
    if device["sm_clock_mhz"] != expected_sm_clock_mhz:
        raise RuntimeError(
            "A800 SM clock is not at the required fixed frequency: "
            f"observed={device['sm_clock_mhz']} MHz, "
            f"required={expected_sm_clock_mhz} MHz")


def _require_quiescent(
        state, expected_uuid, expected_pci, expected_sm_clock_mhz):
    _require_idle(state["device"], expected_uuid, expected_pci)
    _require_known_sm_clock(state["device"], expected_sm_clock_mhz)
    if state["device"]["memory_clock_mhz"] <= 0:
        raise RuntimeError("nvidia-smi reported a zero memory clock")
    if state["compute_processes"]:
        raise RuntimeError("requested A800 has a live compute process")


def _descendant_pids(root_pid):
    if isinstance(root_pid, bool) or not isinstance(root_pid, int) or root_pid <= 0:
        raise ValueError("root process PID must be a positive integer")
    parents = {}
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            text = status_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            text = None
        if text is not None:
            pid = int(status_path.parent.name)
            ppid_lines = tuple(
                line for line in text.splitlines() if line.startswith("PPid:"))
            if len(ppid_lines) != 1:
                raise ValueError(f"process status lacks one PPid: {status_path}")
            parents[pid] = int(ppid_lines[0].split(":", 1)[1].strip())
    owned = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in owned and pid not in owned:
                owned.add(pid)
                changed = True
    return owned


def _profile_target_executable(command):
    positions = tuple(
        index for index, value in enumerate(command)
        if value == "--force-overwrite")
    if not positions:
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError("NCU command has an invalid profiled executable")
    target = Path(command[positions[0] + 1])
    if not target.is_absolute():
        raise ValueError("NCU profiled executable must be absolute")
    return target.resolve()


def _is_profile_target_process(record, target):
    if target is None:
        return False
    proc_path = Path(f"/proc/{record['pid']}")
    if record["process_name"] == "[Not Found]" and not proc_path.exists():
        return True
    process_name = Path(record["process_name"])
    if process_name.is_absolute() and process_name.resolve() == target:
        return True
    try:
        return Path(os.readlink(proc_path / "exe")).resolve() == target
    except FileNotFoundError:
        return False


def _run_monitored_command(
        command, *, cwd, env, stdout_path, stderr_path, nvidia_smi,
        device_uuid, expected_sm_clock_mhz):
    if not isinstance(command, list) or not command:
        raise TypeError("command must be a non-empty list")
    stdout_stream = stdout_path.open("w", encoding="utf-8")
    stderr_stream = stderr_path.open("w", encoding="utf-8")
    samples = []
    started_unix_s = time.time()
    profile_target = _profile_target_executable(command)
    try:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, text=True, stdout=stdout_stream,
            stderr=stderr_stream)
        try:
            while process.poll() is None:
                owned_before = _descendant_pids(process.pid)
                state = _execution_state(nvidia_smi, device_uuid, cwd)
                owned_after = _descendant_pids(process.pid)
                owned = owned_before | owned_after
                compute_pids = {
                    record["pid"] for record in state["compute_processes"]}
                unexpected = compute_pids - owned
                for record in state["compute_processes"]:
                    if (record["pid"] in unexpected
                            and _is_profile_target_process(
                                record, profile_target)):
                        owned.add(record["pid"])
                unexpected = compute_pids - owned
                vanished = {
                    pid for pid in unexpected
                    if not Path(f"/proc/{pid}").exists()
                }
                if vanished:
                    refreshed = _execution_state(
                        nvidia_smi, device_uuid, cwd)
                    refreshed_pids = {
                        record["pid"]
                        for record in refreshed["compute_processes"]}
                    owned.update(vanished - refreshed_pids)
                    unexpected = compute_pids - owned
                if unexpected:
                    unexpected_records = [
                        record for record in state["compute_processes"]
                        if record["pid"] in unexpected]
                    raise RuntimeError(
                        "in-flight clock sample contains an unrelated compute "
                        f"process: {unexpected_records}")
                _require_known_sm_clock(
                    state["device"], expected_sm_clock_mhz)
                samples.append({
                    "state": state,
                    "owner_process_tree_pids": sorted(owned),
                })
                try:
                    process.wait(timeout=0.25)
                except subprocess.TimeoutExpired as timeout:
                    if timeout.timeout <= 0:
                        raise RuntimeError("clock monitor timeout must be positive")
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            raise
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        stdout_stream.close()
        stderr_stream.close()
    if not samples:
        raise RuntimeError("capture command ended before an in-flight clock sample")
    return {
        "command_started_unix_s": started_unix_s,
        "command_finished_unix_s": time.time(),
        "command_pid": process.pid,
        "samples": samples,
    }


def _post_command_state(
        nvidia_smi, device_uuid, pci_bus_id, expected_sm_clock_mhz, cwd):
    attempts = 40 if expected_sm_clock_mhz is None else 1
    for attempt in range(attempts):
        state = _execution_state(nvidia_smi, device_uuid, cwd)
        if state["compute_processes"]:
            _require_quiescent(
                state, device_uuid, pci_bus_id, expected_sm_clock_mhz)
        try:
            _require_quiescent(
                state, device_uuid, pci_bus_id, expected_sm_clock_mhz)
            return state
        except RuntimeError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.25)
    raise AssertionError("post-command state loop did not terminate")


def _run_with_device_boundary(
        command, *, cwd, env, stdout_path, stderr_path, boundary_path,
        label, nvidia_smi, device_uuid, pci_bus_id, expected_sm_clock_mhz):
    before = _execution_state(nvidia_smi, device_uuid, cwd)
    _require_quiescent(
        before, device_uuid, pci_bus_id, expected_sm_clock_mhz)
    try:
        monitored = _run_monitored_command(
            command, cwd=cwd, env=env, stdout_path=stdout_path,
            stderr_path=stderr_path, nvidia_smi=nvidia_smi,
            device_uuid=device_uuid,
            expected_sm_clock_mhz=expected_sm_clock_mhz)
    except BaseException as error:
        after = _post_command_state(
            nvidia_smi, device_uuid, pci_bus_id,
            expected_sm_clock_mhz, cwd)
        _write_json(boundary_path, {
            "schema": "nvidia_a800_execution_boundary_v3",
            "label": label,
            "status": "command_failed",
            "required_sm_clock_mhz": expected_sm_clock_mhz,
            "command": command,
            "before": before,
            "during": [],
            "after": after,
        })
        try:
            _require_quiescent(
                after, device_uuid, pci_bus_id, expected_sm_clock_mhz)
        except BaseException as cleanup_error:
            raise RuntimeError(
                f"{label} failed and left the A800 non-quiescent") from cleanup_error
        raise error
    after = _post_command_state(
        nvidia_smi, device_uuid, pci_bus_id, expected_sm_clock_mhz, cwd)
    _write_json(boundary_path, {
        "schema": "nvidia_a800_execution_boundary_v3",
        "label": label,
        "status": "completed_and_released",
        "required_sm_clock_mhz": expected_sm_clock_mhz,
        "command": command,
        "before": before,
        "during": monitored["samples"],
        "after": after,
        "command_started_unix_s": monitored["command_started_unix_s"],
        "command_finished_unix_s": monitored["command_finished_unix_s"],
        "command_pid": monitored["command_pid"],
    })
    return monitored


def _write_manifest(output_dir):
    entries = []
    for path in sorted(output_dir.rglob("*")):
        relative = path.relative_to(output_dir).as_posix()
        if not path.is_file() or relative in (
                "manifest.sha256", "manifest_anchor.json"):
            continue
        entries.append(f"{_sha256(path)}  {relative}")
    manifest = output_dir / "manifest.sha256"
    manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
    anchor = {
        "schema": "sha256_manifest_anchor_v1",
        "manifest_path": "manifest.sha256",
        "manifest_sha256": _sha256(manifest),
        "entry_count": len(entries),
    }
    _write_json(output_dir / "manifest_anchor.json", anchor)
    return anchor


def _compile_argv(nvcc, source, binary):
    return [
        str(nvcc), "-O3", "-std=c++14", "-arch=sm_80", "-Xptxas=-v",
        "-Xlinker=--build-id=none",
        "-Xcompiler=-frandom-seed=midlayer_a800_loader",
        str(source), "-o", str(binary),
    ]


def _strip_argv(strip_tool, binary):
    return [str(strip_tool), "--strip-all", str(binary)]


def _sass_argv(cuobjdump, binary):
    return [str(cuobjdump), "--dump-sass", str(binary)]


def _native_argv(binary, capture_mode, trials):
    return [
        str(binary), "--mode", capture_mode, "--trials", str(trials)]


def _ncu_argv(ncu, binary, ncu_dir, grid, tiles):
    name = f"g{grid}_t{tiles}"
    return [
        str(ncu), "--kernel-name-base", "function",
        "--kernel-name", "regex:^two_stage_ldgsts_loader$",
        "--cache-control", "none", "--clock-control", "none",
        "--launch-skip", str(WARMUP_LAUNCHES), "--launch-count", "1",
        "--metrics", ",".join(NCU_METRICS), "--page", "raw", "--csv",
        "--print-units", "base", "--log-file", str(ncu_dir / f"{name}.csv"),
        "--export", str(ncu_dir / f"{name}.ncu-rep"),
        "--force-overwrite", str(binary), "--mode", "profile_cell",
        "--grid", str(grid), "--tiles", str(tiles),
    ]


def _reuse_ncu_argv(ncu, binary, ncu_dir, case):
    case_id = case["case_id"]
    return [
        str(ncu), "--kernel-name-base", "function",
        "--kernel-name", "regex:^grouped_mla_kv_reuse_loader$",
        "--cache-control", "all", "--clock-control", "none",
        "--launch-count", "1", "--metrics", ",".join(REUSE_NCU_METRICS),
        "--page", "raw", "--csv", "--print-units", "base",
        "--log-file", str(ncu_dir / f"{case_id}.csv"),
        "--export", str(ncu_dir / f"{case_id}.ncu-rep"),
        "--force-overwrite", str(binary), "--mode", "reuse_profile_case",
        "--active-sources", str(case["active_sources"]),
        "--tiles-per-split", str(case["tiles_per_split"]),
    ]


def build_reuse_command_manifest(*, repo_root, output_dir, ncu, binary):
    paths = tuple(Path(path).resolve() for path in (
        repo_root, output_dir, ncu, binary))
    if any(not path.is_absolute() for path in paths):
        raise ValueError("reuse command paths must be absolute")
    resolved_repo, resolved_output, resolved_ncu, resolved_binary = paths
    ncu_dir = resolved_output / "ncu"
    return {
        "schema": "nvidia_a800_grouped_mla_kv_reuse_commands_v1",
        "source": str(
            resolved_repo / "microbench/nvidia_a800/memory/lsu" / SOURCE_NAME),
        "runner": str(
            resolved_repo / "microbench/nvidia_a800/memory/lsu" / RUNNER_NAME),
        "ncu_argv": {
            case["case_id"]: _reuse_ncu_argv(
                resolved_ncu, resolved_binary, ncu_dir, case)
            for case in _reuse_cases()
        },
        "counter_only": True,
        "production_or_evaluation_duration_consumed": False,
    }


def _reuse_capture_command_document(identity, tool_paths):
    output_dir = Path(identity["output_path"])
    repo_root = Path(identity["repo_root"])
    snapshot_dir = output_dir / "source_snapshot"
    binary = output_dir / "build/two_stage_ldgsts_loader"
    commands = build_reuse_command_manifest(
        repo_root=repo_root, output_dir=output_dir,
        ncu=tool_paths["ncu"], binary=binary)
    normalized_capture_argv = [
        str(repo_root / "microbench/nvidia_a800/memory/lsu" / RUNNER_NAME),
        "reuse-capture", "--output-dir", str(output_dir), "--run-id",
        identity["run_id"], "--owner", identity["owner"], "--device-uuid",
        identity["device_uuid"], "--pci-bus-id", identity["pci_bus_id"],
        "--mla-static-manifest", identity["static_manifest_source_path"],
        "--repo-root", str(repo_root), "--nvcc", tool_paths["nvcc"],
        "--cuobjdump", tool_paths["cuobjdump"], "--strip",
        tool_paths["strip"], "--ncu", tool_paths["ncu"], "--nvidia-smi",
        tool_paths["nvidia_smi"],
    ]
    if identity["required_sm_clock_mhz"] is not None:
        insert_at = normalized_capture_argv.index("--mla-static-manifest")
        normalized_capture_argv[insert_at:insert_at] = [
            "--sm-clock-mhz", str(identity["required_sm_clock_mhz"])]
    commands["normalized_capture_argv"] = normalized_capture_argv
    commands["compile_argv"] = _compile_argv(
        tool_paths["nvcc"], snapshot_dir / SOURCE_NAME, binary)
    commands["strip_argv"] = _strip_argv(tool_paths["strip"], binary)
    commands["sass_argv"] = _sass_argv(tool_paths["cuobjdump"], binary)
    commands["tool_version_argv"] = _tool_version_argvs(tool_paths)
    commands["device_query_argv"] = _device_query_argv(
        tool_paths["nvidia_smi"], identity["device_uuid"])
    commands["compute_query_argv"] = _compute_query_argv(
        tool_paths["nvidia_smi"], identity["device_uuid"])
    return commands


def _tool_version_argvs(tool_paths):
    version_argument = {
        "nvcc": "--version",
        "cuobjdump": "--version",
        "strip": "--version",
        "ncu": "--version",
        "nvidia_smi": "-h",
    }
    if set(tool_paths) != set(version_argument):
        raise ValueError("tool path registry does not match version commands")
    return {
        name: [path, version_argument[name]]
        for name, path in tool_paths.items()
    }


def _device_query_argv(nvidia_smi, device_uuid):
    return [
        str(nvidia_smi), f"--id={device_uuid}",
        f"--query-gpu={DEVICE_QUERY_FIELDS}",
        "--format=csv,noheader,nounits",
    ]


def _compute_query_argv(nvidia_smi, device_uuid):
    return [
        str(nvidia_smi), f"--id={device_uuid}",
        f"--query-compute-apps={COMPUTE_QUERY_FIELDS}",
        "--format=csv,noheader,nounits",
    ]


def _command_document(identity, capture_mode, trials, tool_paths):
    output_dir = Path(identity["output_path"])
    repo_root = Path(identity["repo_root"])
    snapshot_dir = output_dir / "source_snapshot"
    build_dir = output_dir / "build"
    ncu_dir = output_dir / "ncu"
    binary = build_dir / "two_stage_ldgsts_loader"
    capture_argv = [
        str(repo_root / "microbench/nvidia_a800/memory/lsu" / RUNNER_NAME),
        "--output-dir", str(output_dir), "--run-id", identity["run_id"],
        "--owner", identity["owner"], "--device-uuid", identity["device_uuid"],
        "--pci-bus-id", identity["pci_bus_id"], "--sm-clock-mhz",
        str(identity["required_sm_clock_mhz"]), "--mla-static-manifest",
        identity["static_manifest_source_path"], "--repo-root", str(repo_root),
        "--capture-mode", capture_mode, "--trials", str(trials),
        "--nvcc", tool_paths["nvcc"], "--cuobjdump", tool_paths["cuobjdump"],
        "--strip", tool_paths["strip"], "--ncu", tool_paths["ncu"],
        "--nvidia-smi", tool_paths["nvidia_smi"],
    ]
    if capture_mode == "full_matrix":
        cells = NCU_CELLS
    elif capture_mode == "smoke_cell":
        cells = (SMOKE_CELL,)
    else:
        raise ValueError(f"unknown capture mode {capture_mode!r}")
    return {
        "schema": "nvidia_a800_two_stage_ldgsts_command_manifest_v1",
        "normalized_capture_argv": capture_argv,
        "git_argv": {
            "baseline": ["git", "rev-parse", "HEAD"],
            "status": ["git", "status", "--short"],
            "source_tracked": [
                "git", "ls-files", "--error-unmatch",
                f"microbench/nvidia_a800/memory/lsu/{SOURCE_NAME}"],
            "runner_tracked": [
                "git", "ls-files", "--error-unmatch",
                f"microbench/nvidia_a800/memory/lsu/{RUNNER_NAME}"],
        },
        "compile_argv": _compile_argv(
            tool_paths["nvcc"], snapshot_dir / SOURCE_NAME, binary),
        "strip_argv": _strip_argv(tool_paths["strip"], binary),
        "sass_argv": _sass_argv(tool_paths["cuobjdump"], binary),
        "tool_version_argv": _tool_version_argvs(tool_paths),
        "native_argv": _native_argv(binary, capture_mode, trials),
        "ncu_argv": {
            f"g{grid}_t{tiles}": _ncu_argv(
                tool_paths["ncu"], binary, ncu_dir, grid, tiles)
            for grid, tiles in cells
        },
        "device_query_argv": _device_query_argv(
            tool_paths["nvidia_smi"], identity["device_uuid"]),
        "compute_query_argv": _compute_query_argv(
            tool_paths["nvidia_smi"], identity["device_uuid"]),
    }


def _verify_manifest(capture_dir, detached_manifest_sha256):
    if re.fullmatch(r"[0-9a-f]{64}", detached_manifest_sha256) is None:
        raise ValueError("detached manifest SHA256 must be 64 lowercase hex digits")
    manifest_path = capture_dir / "manifest.sha256"
    if _sha256(manifest_path) != detached_manifest_sha256:
        raise ValueError("capture manifest does not match detached SHA256")
    entries = {}
    previous_relative = None
    for line_number, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        if match is None:
            raise ValueError(f"manifest line {line_number} is malformed")
        relative_text = match.group(2)
        relative = Path(relative_text)
        if (relative.is_absolute() or ".." in relative.parts
                or relative.as_posix() != relative_text):
            raise ValueError("manifest path is not a contained POSIX path")
        if relative_text in entries:
            raise ValueError("manifest repeats a file")
        if previous_relative is not None and relative_text <= previous_relative:
            raise ValueError("manifest inventory is not strictly sorted")
        entries[relative_text] = match.group(1)
        previous_relative = relative_text
    actual = set()
    for path in capture_dir.rglob("*"):
        relative = path.relative_to(capture_dir).as_posix()
        if path.is_file() and relative not in (
                "manifest.sha256", "manifest_anchor.json"):
            actual.add(relative)
    if set(entries) != actual:
        raise ValueError(
            "manifest inventory mismatch: "
            f"missing={sorted(actual-set(entries))}, "
            f"extra={sorted(set(entries)-actual)}")
    for relative, digest in entries.items():
        path = capture_dir / relative
        if path.is_symlink():
            raise ValueError(f"manifest payload must not be a symlink: {relative}")
        if _sha256(path) != digest:
            raise ValueError(f"manifest payload hash changed: {relative}")
    anchor = _read_json(capture_dir / "manifest_anchor.json")
    expected_anchor = {
        "schema": "sha256_manifest_anchor_v1",
        "manifest_path": "manifest.sha256",
        "manifest_sha256": detached_manifest_sha256,
        "entry_count": len(entries),
    }
    if anchor != expected_anchor:
        raise ValueError("manifest sidecar disagrees with detached SHA256")
    return entries


def _expected_capture_inventory(static_document, mode):
    if mode == "full_matrix":
        cells = NCU_CELLS
    elif mode == "smoke_cell":
        cells = (SMOKE_CELL,)
    else:
        raise ValueError(f"unknown capture mode {mode!r}")
    artifact_relative = Path(static_document["stage1_artifact_directory"])
    if artifact_relative.is_absolute() or ".." in artifact_relative.parts:
        raise ValueError("snapshot artifact directory is not contained")
    files = {
        "build/compile.stdout",
        "build/cuobjdump.stderr",
        "build/ptxas.stderr",
        "build/sass_audit.json",
        "build/two_stage_ldgsts_loader",
        "build/two_stage_ldgsts_loader.sass",
        "candidate.json",
        "capture_identity.json",
        "commands.json",
        "native/execution_boundary.json",
        "native/sweep.csv",
        "native/sweep.stderr",
        "ncu/audit.json",
        "postflight_execution_state.json",
        "preflight_execution_state.json",
        "report.json",
        f"source_snapshot/{RUNNER_NAME}",
        f"source_snapshot/{SOURCE_NAME}",
        "source_snapshot/mla_stage1_static_manifest.json",
        "static_artifact_audit.json",
        "static_contract_audit.json",
        "static_snapshot_audit.json",
        "toolchain.json",
    }
    for name in (
            "_fwd_grouped_kernel_stage1.cubin",
            "_fwd_grouped_kernel_stage1.ttgir",
            "_fwd_grouped_kernel_stage1.json"):
        files.add(
            f"source_snapshot/{artifact_relative.as_posix()}/{name}")
    for grid, tiles in cells:
        name = f"g{grid}_t{tiles}"
        files.update({
            f"ncu/{name}.csv",
            f"ncu/{name}.execution_boundary.json",
            f"ncu/{name}.ncu-rep",
            f"ncu/{name}.stderr",
            f"ncu/{name}.stdout",
        })
    return files, cells


def _expected_reuse_capture_inventory(static_document):
    artifact_relative = Path(static_document["stage1_artifact_directory"])
    if artifact_relative.is_absolute() or ".." in artifact_relative.parts:
        raise ValueError("reuse snapshot artifact directory is not contained")
    files = {
        "build/compile.stdout",
        "build/cuobjdump.stderr",
        "build/ptxas.stderr",
        "build/sass_audit.json",
        "build/two_stage_ldgsts_loader",
        "build/two_stage_ldgsts_loader.sass",
        "capture_identity.json",
        "case_manifest.json",
        "commands.json",
        "ncu/audit.json",
        "postflight_execution_state.json",
        "preflight_execution_state.json",
        "report.json",
        f"source_snapshot/{RUNNER_NAME}",
        f"source_snapshot/{SOURCE_NAME}",
        "source_snapshot/mla_stage1_static_manifest.json",
        "static_artifact_audit.json",
        "static_contract_audit.json",
        "static_snapshot_audit.json",
        "toolchain.json",
    }
    for name in (
            "_fwd_grouped_kernel_stage1.cubin",
            "_fwd_grouped_kernel_stage1.ttgir",
            "_fwd_grouped_kernel_stage1.json"):
        files.add(
            f"source_snapshot/{artifact_relative.as_posix()}/{name}")
    for case in _reuse_cases():
        case_id = case["case_id"]
        files.update({
            f"ncu/{case_id}.csv",
            f"ncu/{case_id}.execution_boundary.json",
            f"ncu/{case_id}.ncu-rep",
            f"ncu/{case_id}.stderr",
            f"ncu/{case_id}.stdout",
        })
    return files


def _validate_state_schema(state):
    if set(state) != {"unix_s", "device", "compute_processes"}:
        raise ValueError("execution-state schema mismatch")
    if (isinstance(state["unix_s"], bool)
            or not isinstance(state["unix_s"], (int, float))
            or not math.isfinite(state["unix_s"])):
        raise ValueError("execution-state timestamp is invalid")
    expected_device = {
        "uuid", "name", "pci_bus_id", "driver_version", "pstate",
        "memory_used_mib", "memory_total_mib", "utilization_gpu_pct",
        "utilization_memory_pct", "sm_clock_mhz", "memory_clock_mhz",
    }
    if set(state["device"]) != expected_device:
        raise ValueError("execution-state device schema mismatch")
    if not isinstance(state["compute_processes"], list):
        raise ValueError("execution-state compute processes must be a list")
    for process in state["compute_processes"]:
        if set(process) != {
                "gpu_uuid", "pid", "process_name", "used_gpu_memory_mib"}:
            raise ValueError("execution-state process schema mismatch")


def validate_execution_boundary(
        document, label, device_uuid, pci_bus_id, sm_clock_mhz,
        expected_command):
    required = {
        "schema", "label", "status", "required_sm_clock_mhz", "before",
        "during", "after", "command_started_unix_s",
        "command_finished_unix_s", "command", "command_pid",
    }
    if set(document) != required:
        raise ValueError("execution-boundary schema mismatch")
    if (document["schema"] != "nvidia_a800_execution_boundary_v3"
            or document["label"] != label
            or document["status"] != "completed_and_released"
            or document["required_sm_clock_mhz"] != sm_clock_mhz
            or document["command"] != expected_command):
        raise ValueError("execution-boundary identity changed")
    if not isinstance(document["during"], list) or not document["during"]:
        raise ValueError("execution boundary lacks an in-flight clock sample")
    if (isinstance(document["command_pid"], bool)
            or not isinstance(document["command_pid"], int)
            or document["command_pid"] <= 0):
        raise ValueError("execution-boundary command PID is invalid")
    before = document["before"]
    after = document["after"]
    during_states = []
    for sample in document["during"]:
        if set(sample) != {"state", "owner_process_tree_pids"}:
            raise ValueError("in-flight clock sample schema mismatch")
        owned = sample["owner_process_tree_pids"]
        if (not isinstance(owned, list) or document["command_pid"] not in owned
                or any(isinstance(pid, bool) or not isinstance(pid, int)
                       or pid <= 0 for pid in owned)
                or len(owned) != len(set(owned)) or owned != sorted(owned)):
            raise ValueError("in-flight owner process tree is invalid")
        compute_pids = {
            record["pid"] for record in sample["state"]["compute_processes"]}
        if not compute_pids <= set(owned):
            raise ValueError("in-flight sample contains an unrelated compute PID")
        during_states.append(sample["state"])
    for state in (before, *during_states, after):
        _validate_state_schema(state)
        _require_device_identity(state["device"], device_uuid, pci_bus_id)
        _require_known_sm_clock(state["device"], sm_clock_mhz)
    _require_quiescent(before, device_uuid, pci_bus_id, sm_clock_mhz)
    _require_quiescent(after, device_uuid, pci_bus_id, sm_clock_mhz)
    started = document["command_started_unix_s"]
    finished = document["command_finished_unix_s"]
    if not (before["unix_s"] <= started <= finished <= after["unix_s"]):
        raise ValueError("execution-boundary timestamps are not ordered")
    if any(not (started <= state["unix_s"] <= finished)
           for state in during_states):
        raise ValueError("in-flight clock sample is outside the command interval")
    return {
        "label": label,
        "required_sm_clock_mhz": sm_clock_mhz,
        "in_flight_sample_count": len(during_states),
    }


def _validate_capture_identity(document):
    required = {
        "schema", "hardware_id", "owner", "baseline_commit", "run_id",
        "device_uuid", "pci_bus_id", "required_sm_clock_mhz", "repo_root",
        "static_manifest_source_path", "pid", "started_unix_s", "output_path",
        "production_or_evaluation_duration_consumed",
        "production_parameter_published",
    }
    if set(document) != required:
        raise ValueError("capture identity schema mismatch")
    if (document["schema"] != "nvidia_a800_two_stage_ldgsts_capture_identity_v3"
            or document["hardware_id"] != HARDWARE_ID
            or document["production_or_evaluation_duration_consumed"] is not False
            or document["production_parameter_published"] is not False):
        raise ValueError("capture identity semantic contract changed")
    if re.fullmatch(r"[0-9a-f]{40}", document["baseline_commit"]) is None:
        raise ValueError("capture baseline commit is invalid")
    if re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
            document["device_uuid"]) is None:
        raise ValueError("capture device UUID is invalid")
    if RUN_ID_PATTERN.fullmatch(document["run_id"]) is None:
        raise ValueError("capture run ID is invalid")
    _require_nonempty(document["owner"], "capture owner")
    _require_nonempty(document["output_path"], "capture output path")
    for name in ("repo_root", "static_manifest_source_path"):
        if not Path(_require_nonempty(document[name], name)).is_absolute():
            raise ValueError(f"capture {name} is not absolute")
    if (isinstance(document["pid"], bool) or not isinstance(document["pid"], int)
            or document["pid"] <= 0):
        raise ValueError("capture PID is invalid")
    if (isinstance(document["started_unix_s"], bool)
            or not isinstance(document["started_unix_s"], (int, float))
            or not math.isfinite(document["started_unix_s"])):
        raise ValueError("capture start timestamp is invalid")
    _normalize_pci_bus_id(document["pci_bus_id"])
    _require_known_sm_clock(
        {"sm_clock_mhz": document["required_sm_clock_mhz"]},
        document["required_sm_clock_mhz"])


def _validate_reuse_capture_identity(document):
    required = {
        "schema", "hardware_id", "owner", "baseline_commit", "run_id",
        "device_uuid", "pci_bus_id", "required_sm_clock_mhz", "repo_root",
        "static_manifest_source_path", "pid", "started_unix_s", "output_path",
        "production_or_evaluation_duration_consumed",
        "production_parameter_published",
    }
    if set(document) != required:
        raise ValueError("reuse capture identity schema mismatch")
    if (document["schema"]
            != "nvidia_a800_grouped_mla_kv_reuse_capture_identity_v1"
            or document["hardware_id"] != HARDWARE_ID
            or document["production_or_evaluation_duration_consumed"] is not False
            or document["production_parameter_published"] is not False):
        raise ValueError("reuse capture identity semantic contract changed")
    if re.fullmatch(r"[0-9a-f]{40}", document["baseline_commit"]) is None:
        raise ValueError("reuse capture baseline commit is invalid")
    if re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
            document["device_uuid"]) is None:
        raise ValueError("reuse capture device UUID is invalid")
    if RUN_ID_PATTERN.fullmatch(document["run_id"]) is None:
        raise ValueError("reuse capture run ID is invalid")
    _require_nonempty(document["owner"], "reuse capture owner")
    _require_nonempty(document["output_path"], "reuse capture output path")
    for name in ("repo_root", "static_manifest_source_path", "output_path"):
        if not Path(_require_nonempty(document[name], name)).is_absolute():
            raise ValueError(f"reuse capture {name} is not absolute")
    if (isinstance(document["pid"], bool)
            or not isinstance(document["pid"], int) or document["pid"] <= 0):
        raise ValueError("reuse capture PID is invalid")
    if (isinstance(document["started_unix_s"], bool)
            or not isinstance(document["started_unix_s"], (int, float))
            or not math.isfinite(document["started_unix_s"])):
        raise ValueError("reuse capture start timestamp is invalid")
    _normalize_pci_bus_id(document["pci_bus_id"])
    if document["required_sm_clock_mhz"] is not None:
        _require_known_sm_clock(
            {"sm_clock_mhz": document["required_sm_clock_mhz"]},
            document["required_sm_clock_mhz"])


def _validate_toolchain_document(document):
    if set(document) != {"nvcc", "cuobjdump", "strip", "ncu", "nvidia_smi"}:
        raise ValueError("toolchain inventory changed")
    for name, record in document.items():
        if set(record) != {
                "path", "sha256", "version_stdout", "version_stderr"}:
            raise ValueError(f"toolchain {name} schema mismatch")
        if not Path(record["path"]).is_absolute():
            raise ValueError(f"toolchain {name} path is not absolute")
        if re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None:
            raise ValueError(f"toolchain {name} SHA256 is invalid")
        if (not isinstance(record["version_stdout"], str)
                or not isinstance(record["version_stderr"], str)):
            raise ValueError(f"toolchain {name} version evidence is invalid")


def _git_file_bytes(repo_root, baseline_commit, relative_path):
    if re.fullmatch(r"[0-9a-f]{40}", baseline_commit) is None:
        raise ValueError("baseline commit is invalid")
    result = subprocess.run(
        ["git", "show", f"{baseline_commit}:{relative_path}"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True)
    return result.stdout


def _normalized_kernel_sass(text):
    body = _sass_kernel_body(text)
    instructions = tuple(
        line.strip() for line in body.splitlines()
        if re.search(r"/\*[0-9a-fA-F]+\*/", line))
    if not instructions:
        raise ValueError("SASS has no encoded kernel instructions")
    return "\n".join(instructions) + "\n"


def _normalized_reuse_kernel_sass(text):
    if not isinstance(text, str) or not text:
        raise ValueError("reuse SASS must be non-empty text")
    match = re.search(
        r"Function\s*:\s*grouped_mla_kv_reuse_loader\b"
        r"(?P<body>.*?)(?=\n\s*Function\s*:|\Z)",
        text, flags=re.DOTALL)
    if match is None:
        raise ValueError("SASS lacks grouped_mla_kv_reuse_loader")
    instructions = tuple(
        line.strip() for line in match.group("body").splitlines()
        if re.search(r"/\*[0-9a-fA-F]+\*/", line))
    if not instructions:
        raise ValueError("reuse SASS has no encoded kernel instructions")
    return "\n".join(instructions) + "\n"


def _verify_rebuilt_artifacts(
        capture_dir, baseline_source, nvcc, cuobjdump, strip_tool,
        toolchain):
    local_tools = {
        "nvcc": Path(nvcc).resolve(),
        "cuobjdump": Path(cuobjdump).resolve(),
        "strip": Path(strip_tool).resolve(),
    }
    for name, path in local_tools.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != toolchain[name]["sha256"]:
            raise ValueError(f"local {name} does not match capture toolchain")
    with tempfile.TemporaryDirectory(prefix="midlayer_a800_loader_verify_") as temporary:
        temporary_dir = Path(temporary)
        source = temporary_dir / SOURCE_NAME
        binary = temporary_dir / "two_stage_ldgsts_loader"
        ptxas = temporary_dir / "ptxas.stderr"
        rebuilt_sass = temporary_dir / "rebuilt.sass"
        captured_binary_sass = temporary_dir / "captured_binary.sass"
        source.write_bytes(baseline_source)
        build_env = os.environ.copy()
        build_env["SOURCE_DATE_EPOCH"] = "0"
        _run(
            _compile_argv(local_tools["nvcc"], source, binary),
            cwd=temporary_dir, env=build_env,
            stdout_path=temporary_dir / "compile.stdout", stderr_path=ptxas)
        validate_ptxas(ptxas.read_text(encoding="utf-8"))
        _run(_strip_argv(local_tools["strip"], binary), cwd=temporary_dir)
        captured_binary = capture_dir / "build/two_stage_ldgsts_loader"
        if _sha256(binary) != _sha256(captured_binary):
            raise ValueError("captured binary is not reproducible from Git source")
        _run(
            _sass_argv(local_tools["cuobjdump"], binary), cwd=temporary_dir,
            stdout_path=rebuilt_sass,
            stderr_path=temporary_dir / "rebuilt_cuobjdump.stderr")
        _run(
            _sass_argv(local_tools["cuobjdump"], captured_binary),
            cwd=temporary_dir, stdout_path=captured_binary_sass,
            stderr_path=temporary_dir / "captured_cuobjdump.stderr")
        recorded_sass = (
            capture_dir / "build/two_stage_ldgsts_loader.sass").read_text(
                encoding="utf-8")
        normalized = _normalized_kernel_sass(recorded_sass)
        if (_normalized_kernel_sass(rebuilt_sass.read_text(encoding="utf-8"))
                != normalized):
            raise ValueError("recorded SASS is not reproducible from Git source")
        if (_normalized_kernel_sass(
                captured_binary_sass.read_text(encoding="utf-8"))
                != normalized):
            raise ValueError("captured binary does not contain the recorded SASS")
    return {
        "binary_sha256": _sha256(capture_dir / "build/two_stage_ldgsts_loader"),
        "normalized_sass_sha256": hashlib.sha256(
            normalized.encode("utf-8")).hexdigest(),
    }


def _verify_rebuilt_reuse_artifacts(
        capture_dir, baseline_source, nvcc, cuobjdump, strip_tool,
        toolchain):
    local_tools = {
        "nvcc": Path(nvcc).resolve(),
        "cuobjdump": Path(cuobjdump).resolve(),
        "strip": Path(strip_tool).resolve(),
    }
    for name, path in local_tools.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != toolchain[name]["sha256"]:
            raise ValueError(f"local {name} does not match reuse toolchain")
    with tempfile.TemporaryDirectory(
            prefix="midlayer_a800_mla_reuse_verify_") as temporary:
        temporary_dir = Path(temporary)
        source = temporary_dir / SOURCE_NAME
        binary = temporary_dir / "two_stage_ldgsts_loader"
        ptxas = temporary_dir / "ptxas.stderr"
        rebuilt_sass = temporary_dir / "rebuilt.sass"
        captured_binary_sass = temporary_dir / "captured_binary.sass"
        source.write_bytes(baseline_source)
        build_env = os.environ.copy()
        build_env["SOURCE_DATE_EPOCH"] = "0"
        _run(
            _compile_argv(local_tools["nvcc"], source, binary),
            cwd=temporary_dir, env=build_env,
            stdout_path=temporary_dir / "compile.stdout", stderr_path=ptxas)
        validate_ptxas(ptxas.read_text(encoding="utf-8"))
        _run(_strip_argv(local_tools["strip"], binary), cwd=temporary_dir)
        captured_binary = capture_dir / "build/two_stage_ldgsts_loader"
        if _sha256(binary) != _sha256(captured_binary):
            raise ValueError(
                "captured reuse binary is not reproducible from Git source")
        _run(
            _sass_argv(local_tools["cuobjdump"], binary), cwd=temporary_dir,
            stdout_path=rebuilt_sass,
            stderr_path=temporary_dir / "rebuilt_cuobjdump.stderr")
        _run(
            _sass_argv(local_tools["cuobjdump"], captured_binary),
            cwd=temporary_dir, stdout_path=captured_binary_sass,
            stderr_path=temporary_dir / "captured_cuobjdump.stderr")
        recorded_text = (
            capture_dir / "build/two_stage_ldgsts_loader.sass").read_text(
                encoding="utf-8")
        rebuilt_text = rebuilt_sass.read_text(encoding="utf-8")
        binary_text = captured_binary_sass.read_text(encoding="utf-8")
        for normalizer in (
                _normalized_kernel_sass, _normalized_reuse_kernel_sass):
            normalized = normalizer(recorded_text)
            if normalizer(rebuilt_text) != normalized:
                raise ValueError(
                    "recorded reuse SASS is not reproducible from Git source")
            if normalizer(binary_text) != normalized:
                raise ValueError(
                    "captured binary does not contain the recorded reuse SASS")
        loader_audit = validate_sass(recorded_text)
        reuse_audit = validate_reuse_sass(recorded_text)
    return {
        "binary_sha256": _sha256(
            capture_dir / "build/two_stage_ldgsts_loader"),
        "loader_normalized_sass_sha256": hashlib.sha256(
            _normalized_kernel_sass(recorded_text).encode("utf-8")).hexdigest(),
        "reuse_normalized_sass_sha256": hashlib.sha256(
            _normalized_reuse_kernel_sass(recorded_text).encode(
                "utf-8")).hexdigest(),
        "loader_sass_audit": loader_audit,
        "reuse_sass_audit": reuse_audit,
    }


def verify_capture(
        capture_dir, detached_manifest_sha256, *, repo_root=Path.cwd(),
        nvcc=None, cuobjdump=None, strip_tool=None):
    """Offline, read-only verification rooted in an externally stored hash."""

    capture_dir = Path(capture_dir).resolve()
    if not capture_dir.is_dir():
        raise FileNotFoundError(capture_dir)
    manifest_entries = _verify_manifest(
        capture_dir, detached_manifest_sha256)
    identity = _read_json(capture_dir / "capture_identity.json")
    _validate_capture_identity(identity)
    repo_root = Path(repo_root).resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(repo_root)
    report = _read_json(capture_dir / "report.json")
    if not isinstance(report, dict):
        raise TypeError("capture report must be an object")
    mode = report["capture_mode"]
    trials = report["trials_per_cell"]
    if (mode == "full_matrix"
            and trials < MIN_CANDIDATE_TRIALS_PER_CELL):
        raise ValueError("full capture has fewer than 7 trials per cell")
    static_manifest_path = (
        capture_dir / "source_snapshot/mla_stage1_static_manifest.json")
    static_document = _read_json(static_manifest_path)
    expected_inventory, cells = _expected_capture_inventory(
        static_document, mode)
    if set(manifest_entries) != expected_inventory:
        raise ValueError("capture file inventory does not match its mode")
    static_contract = validate_static_manifest(
        static_document, _sha256(static_manifest_path))
    static_snapshot = validate_static_snapshot_files(
        static_manifest_path, static_document)
    expected_artifact_audit = dict(static_snapshot)
    expected_artifact_audit.pop("manifest_sha256")
    if _read_json(capture_dir / "static_contract_audit.json") != static_contract:
        raise ValueError("static contract audit is not reproducible")
    if _read_json(capture_dir / "static_artifact_audit.json") != expected_artifact_audit:
        raise ValueError("static artifact audit is not reproducible")
    if _read_json(capture_dir / "static_snapshot_audit.json") != static_snapshot:
        raise ValueError("static snapshot audit is not reproducible")
    toolchain = _read_json(capture_dir / "toolchain.json")
    _validate_toolchain_document(toolchain)
    tool_paths = {
        name: record["path"] for name, record in toolchain.items()}
    expected_commands = _command_document(
        identity, mode, trials, tool_paths)
    if _read_json(capture_dir / "commands.json") != expected_commands:
        raise ValueError("capture command manifest is not reproducible")
    source_relative = (
        f"microbench/nvidia_a800/memory/lsu/{SOURCE_NAME}")
    runner_relative = (
        f"microbench/nvidia_a800/memory/lsu/{RUNNER_NAME}")
    baseline_source = _git_file_bytes(
        repo_root, identity["baseline_commit"], source_relative)
    baseline_runner = _git_file_bytes(
        repo_root, identity["baseline_commit"], runner_relative)
    if ((capture_dir / f"source_snapshot/{SOURCE_NAME}").read_bytes()
            != baseline_source):
        raise ValueError("source snapshot differs from Git baseline")
    if ((capture_dir / f"source_snapshot/{RUNNER_NAME}").read_bytes()
            != baseline_runner):
        raise ValueError("runner snapshot differs from Git baseline")
    runner_snapshot_text = (
        capture_dir / f"source_snapshot/{RUNNER_NAME}").read_text(
            encoding="utf-8")
    if ('"--cache-control", "none"' not in runner_snapshot_text
            or '"--clock-control", "none"' not in runner_snapshot_text
            or 'f"{name}.ncu-rep"' not in runner_snapshot_text):
        raise ValueError("snapshotted runner lacks the frozen NCU policy")
    ptxas_path = capture_dir / "build/ptxas.stderr"
    validate_ptxas(ptxas_path.read_text(encoding="utf-8"))
    sass_path = capture_dir / "build/two_stage_ldgsts_loader.sass"
    sass_audit = validate_sass(sass_path.read_text(encoding="utf-8"))
    if _read_json(capture_dir / "build/sass_audit.json") != sass_audit:
        raise ValueError("SASS audit is not reproducible")
    rebuild_audit = _verify_rebuilt_artifacts(
        capture_dir, baseline_source,
        toolchain["nvcc"]["path"] if nvcc is None else nvcc,
        toolchain["cuobjdump"]["path"] if cuobjdump is None else cuobjdump,
        toolchain["strip"]["path"] if strip_tool is None else strip_tool,
        toolchain)
    native_path = capture_dir / "native/sweep.csv"
    native = parse_native_output(
        native_path.read_text(encoding="utf-8"), trials, mode)
    if (native["meta"]["device_uuid"] != identity["device_uuid"]
            or _normalize_pci_bus_id(native["meta"]["pci_bus_id"])
            != _normalize_pci_bus_id(identity["pci_bus_id"])):
        raise ValueError("native output device identity changed")
    clock_mhz = identity["required_sm_clock_mhz"]
    boundaries = {
        "native": validate_execution_boundary(
            _read_json(capture_dir / "native/execution_boundary.json"),
            f"native_{mode}", identity["device_uuid"],
            identity["pci_bus_id"], clock_mhz,
            expected_commands["native_argv"]),
    }
    ncu_audit = {}
    for grid, tiles in cells:
        name = f"g{grid}_t{tiles}"
        metrics = parse_ncu_csv(
            (capture_dir / f"ncu/{name}.csv").read_text(encoding="utf-8"),
            grid)
        geometry = validate_ncu_contract(grid, tiles, metrics)
        ncu_audit[name] = {
            "grid": grid,
            "tiles": tiles,
            "metrics": metrics,
            "counter_geometry": geometry,
        }
        boundaries[name] = validate_execution_boundary(
            _read_json(capture_dir / f"ncu/{name}.execution_boundary.json"),
            f"ncu_{name}", identity["device_uuid"], identity["pci_bus_id"],
            clock_mhz, expected_commands["ncu_argv"][name])
    if _read_json(capture_dir / "ncu/audit.json") != ncu_audit:
        raise ValueError("NCU audit is not reproducible from raw CSV")
    candidate = build_candidate(
        native, ncu_audit, _sha256(native_path),
        _sha256(static_manifest_path))
    if _read_json(capture_dir / "candidate.json") != candidate:
        raise ValueError("candidate is not reproducible from raw evidence")
    preflight = _read_json(capture_dir / "preflight_execution_state.json")
    postflight = _read_json(capture_dir / "postflight_execution_state.json")
    for state in (preflight, postflight):
        _validate_state_schema(state)
        _require_quiescent(
            state, identity["device_uuid"], identity["pci_bus_id"],
            clock_mhz)
    expected_report = {
        "schema": "nvidia_a800_two_stage_ldgsts_capture_v3",
        "baseline_commit": identity["baseline_commit"],
        "source_sha256": _sha256(
            capture_dir / f"source_snapshot/{SOURCE_NAME}"),
        "runner_sha256": _sha256(
            capture_dir / f"source_snapshot/{RUNNER_NAME}"),
        "binary_sha256": _sha256(
            capture_dir / "build/two_stage_ldgsts_loader"),
        "sass_sha256": _sha256(sass_path),
        "native_sha256": _sha256(native_path),
        "static_manifest_sha256": _sha256(static_manifest_path),
        "static_snapshot_manifest_sha256": _sha256(static_manifest_path),
        "capture_mode": mode,
        "trials_per_cell": trials,
        "required_sm_clock_mhz": clock_mhz,
        "ncu_cache_control": "none",
        "ncu_clock_control": "none",
        "rate_unit": RATE_UNIT,
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
    }
    if report != expected_report:
        raise ValueError("capture report is not reproducible")
    return {
        "schema": "nvidia_a800_two_stage_ldgsts_offline_verification_v1",
        "capture_dir": str(capture_dir),
        "detached_manifest_sha256": detached_manifest_sha256,
        "file_count": len(manifest_entries),
        "capture_mode": mode,
        "trials_per_cell": trials,
        "required_sm_clock_mhz": clock_mhz,
        "candidate_evidence_status": candidate["evidence_status"],
        "rebuild_audit": rebuild_audit,
        "boundaries": boundaries,
    }


def verify_reuse_capture(
        capture_dir, detached_manifest_sha256, *, repo_root=Path.cwd(),
        nvcc=None, cuobjdump=None, strip_tool=None):
    """Verify a counter-only grouped-MLA reuse capture from raw evidence."""

    capture_dir = Path(capture_dir).resolve()
    if not capture_dir.is_dir():
        raise FileNotFoundError(capture_dir)
    manifest_entries = _verify_manifest(
        capture_dir, detached_manifest_sha256)
    identity = _read_json(capture_dir / "capture_identity.json")
    _validate_reuse_capture_identity(identity)
    if Path(identity["output_path"]).resolve() != capture_dir:
        raise ValueError("reuse capture identity output path changed")
    repo_root = Path(repo_root).resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(repo_root)
    static_manifest_path = (
        capture_dir / "source_snapshot/mla_stage1_static_manifest.json")
    static_document = _read_json(static_manifest_path)
    expected_inventory = _expected_reuse_capture_inventory(static_document)
    if set(manifest_entries) != expected_inventory:
        raise ValueError("reuse capture file inventory changed")
    static_contract = validate_static_manifest(
        static_document, _sha256(static_manifest_path))
    static_snapshot = validate_static_snapshot_files(
        static_manifest_path, static_document)
    expected_artifact_audit = dict(static_snapshot)
    expected_artifact_audit.pop("manifest_sha256")
    if _read_json(capture_dir / "static_contract_audit.json") != static_contract:
        raise ValueError("reuse static contract audit is not reproducible")
    if (_read_json(capture_dir / "static_artifact_audit.json")
            != expected_artifact_audit):
        raise ValueError("reuse static artifact audit is not reproducible")
    if (_read_json(capture_dir / "static_snapshot_audit.json")
            != static_snapshot):
        raise ValueError("reuse static snapshot audit is not reproducible")

    toolchain = _read_json(capture_dir / "toolchain.json")
    _validate_toolchain_document(toolchain)
    tool_paths = {
        name: record["path"] for name, record in toolchain.items()}
    expected_commands = _reuse_capture_command_document(identity, tool_paths)
    if _read_json(capture_dir / "commands.json") != expected_commands:
        raise ValueError("reuse command manifest is not reproducible")
    source_relative = f"microbench/nvidia_a800/memory/lsu/{SOURCE_NAME}"
    runner_relative = f"microbench/nvidia_a800/memory/lsu/{RUNNER_NAME}"
    baseline_source = _git_file_bytes(
        repo_root, identity["baseline_commit"], source_relative)
    baseline_runner = _git_file_bytes(
        repo_root, identity["baseline_commit"], runner_relative)
    source_snapshot = capture_dir / f"source_snapshot/{SOURCE_NAME}"
    runner_snapshot = capture_dir / f"source_snapshot/{RUNNER_NAME}"
    if source_snapshot.read_bytes() != baseline_source:
        raise ValueError("reuse source snapshot differs from Git baseline")
    if runner_snapshot.read_bytes() != baseline_runner:
        raise ValueError("reuse runner snapshot differs from Git baseline")
    runner_text = runner_snapshot.read_text(encoding="utf-8")
    if ('"--cache-control", "all"' not in runner_text
            or 'regex:^grouped_mla_kv_reuse_loader$' not in runner_text
            or '"--launch-count", "1"' not in runner_text):
        raise ValueError("snapshotted reuse runner lacks the counter policy")

    ptxas_path = capture_dir / "build/ptxas.stderr"
    validate_ptxas(ptxas_path.read_text(encoding="utf-8"))
    sass_path = capture_dir / "build/two_stage_ldgsts_loader.sass"
    sass_text = sass_path.read_text(encoding="utf-8")
    expected_sass_audit = {
        "loader_service": validate_sass(sass_text),
        "kv_reuse": validate_reuse_sass(sass_text),
    }
    if _read_json(capture_dir / "build/sass_audit.json") != expected_sass_audit:
        raise ValueError("reuse SASS audit is not reproducible")
    rebuild_audit = _verify_rebuilt_reuse_artifacts(
        capture_dir, baseline_source,
        toolchain["nvcc"]["path"] if nvcc is None else nvcc,
        toolchain["cuobjdump"]["path"] if cuobjdump is None else cuobjdump,
        toolchain["strip"]["path"] if strip_tool is None else strip_tool,
        toolchain)

    expected_case_manifest = build_reuse_case_manifest(
        baseline_commit=identity["baseline_commit"],
        source_sha256=_sha256(source_snapshot),
        runner_sha256=_sha256(runner_snapshot))
    if _read_json(capture_dir / "case_manifest.json") != expected_case_manifest:
        raise ValueError("reuse case manifest is not reproducible")
    clock_mhz = identity["required_sm_clock_mhz"]
    audits = {}
    boundaries = {}
    for case in _reuse_cases():
        case_id = case["case_id"]
        metrics = parse_reuse_ncu_csv(
            (capture_dir / f"ncu/{case_id}.csv").read_text(
                encoding="utf-8"),
            case)
        audits[case_id] = {
            "case": case,
            "metrics": metrics,
            "cache_line_report": validate_reuse_ncu_contract(case, metrics),
        }
        boundaries[case_id] = validate_execution_boundary(
            _read_json(
                capture_dir / f"ncu/{case_id}.execution_boundary.json"),
            f"ncu_reuse_{case_id}", identity["device_uuid"],
            identity["pci_bus_id"], clock_mhz,
            expected_commands["ncu_argv"][case_id])
    if _read_json(capture_dir / "ncu/audit.json") != audits:
        raise ValueError("reuse NCU audit is not reproducible from raw CSV")
    for state_name in (
            "preflight_execution_state.json",
            "postflight_execution_state.json"):
        state = _read_json(capture_dir / state_name)
        _validate_state_schema(state)
        _require_quiescent(
            state, identity["device_uuid"], identity["pci_bus_id"],
            clock_mhz)
    expected_report = {
        "schema": "nvidia_a800_grouped_mla_kv_reuse_capture_v1",
        "evidence_status": "exact_launch_aggregate_only",
        "baseline_commit": identity["baseline_commit"],
        "source_sha256": _sha256(source_snapshot),
        "runner_sha256": _sha256(runner_snapshot),
        "binary_sha256": _sha256(
            capture_dir / "build/two_stage_ldgsts_loader"),
        "sass_sha256": _sha256(sass_path),
        "static_manifest_sha256": _sha256(static_manifest_path),
        "case_manifest_sha256": _sha256(capture_dir / "case_manifest.json"),
        "ncu_audit_sha256": _sha256(capture_dir / "ncu/audit.json"),
        "case_count": len(audits),
        "required_sm_clock_mhz": clock_mhz,
        "scheduler_entry_order_observed": False,
        "scheduler_entry_order_status": (
            "not_observed_requires_independent_entry_probe"),
        "runtime_active_residency_observed": False,
        "runtime_active_residency_status": (
            "not_observed_requires_independent_residency_probe"),
        "l2_capacity_transition_claimed": False,
        "ncu_cache_control": "all",
        "ncu_clock_control": "none",
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
    }
    if _read_json(capture_dir / "report.json") != expected_report:
        raise ValueError("reuse capture report is not reproducible")
    return {
        "schema": "nvidia_a800_grouped_mla_kv_reuse_offline_verification_v1",
        "capture_dir": str(capture_dir),
        "detached_manifest_sha256": detached_manifest_sha256,
        "file_count": len(manifest_entries),
        "case_count": len(audits),
        "evidence_status": "exact_launch_aggregate_only",
        "rebuild_audit": rebuild_audit,
        "boundaries": boundaries,
    }


def capture(arguments):
    repo_root = arguments.repo_root.resolve()
    output_dir = arguments.output_dir.resolve()
    allowed_root = (
        repo_root / "profiler_output/microbench/nvidia_a800").resolve()
    if output_dir.parent != allowed_root:
        raise ValueError(f"output_dir must be one direct child of {allowed_root}")
    source = repo_root / "microbench/nvidia_a800/memory/lsu" / SOURCE_NAME
    runner = repo_root / "microbench/nvidia_a800/memory/lsu" / RUNNER_NAME
    static_manifest = arguments.mla_static_manifest.resolve()
    tool_paths = {
        "nvcc": str(arguments.nvcc.resolve()),
        "cuobjdump": str(arguments.cuobjdump.resolve()),
        "strip": str(arguments.strip.resolve()),
        "ncu": str(arguments.ncu.resolve()),
        "nvidia_smi": str(arguments.nvidia_smi.resolve()),
    }
    for path in (source, runner, static_manifest, arguments.nvcc,
                 arguments.cuobjdump, arguments.strip, arguments.ncu,
                 arguments.nvidia_smi):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    baseline = _git_output(repo_root, "rev-parse", "HEAD")
    if _git_output(repo_root, "status", "--short"):
        raise RuntimeError("capture requires a clean tracked worktree")
    for path in (source, runner):
        relative = str(path.relative_to(repo_root))
        if _git_output(repo_root, "ls-files", "--error-unmatch", relative) != relative:
            raise RuntimeError(f"capture source is not tracked: {relative}")
    static_document = json.loads(static_manifest.read_text(encoding="utf-8"))
    static_audit = validate_static_manifest(static_document, _sha256(static_manifest))
    static_file_audit = validate_static_artifact_files(
        static_manifest, static_document)
    preflight_state = _execution_state(
        tool_paths["nvidia_smi"], arguments.device_uuid, repo_root)
    _require_quiescent(
        preflight_state, arguments.device_uuid, arguments.pci_bus_id,
        arguments.sm_clock_mhz)

    output_dir.mkdir(parents=True)
    build_dir = output_dir / "build"
    native_dir = output_dir / "native"
    ncu_dir = output_dir / "ncu"
    snapshot_dir = output_dir / "source_snapshot"
    for directory in (build_dir, native_dir, ncu_dir, snapshot_dir):
        directory.mkdir()
    shutil.copy2(source, snapshot_dir / SOURCE_NAME)
    shutil.copy2(runner, snapshot_dir / RUNNER_NAME)
    snapshot_static_manifest = snapshot_dir / "mla_stage1_static_manifest.json"
    shutil.copy2(static_manifest, snapshot_static_manifest)
    artifact_relative = Path(static_document["stage1_artifact_directory"])
    artifact_source_dir = (static_manifest.parent / artifact_relative).resolve()
    artifact_snapshot_dir = snapshot_dir / artifact_relative
    artifact_snapshot_dir.mkdir(parents=True)
    for name in (
            "_fwd_grouped_kernel_stage1.cubin",
            "_fwd_grouped_kernel_stage1.ttgir",
            "_fwd_grouped_kernel_stage1.json"):
        shutil.copy2(artifact_source_dir / name, artifact_snapshot_dir / name)
    static_snapshot_audit = validate_static_snapshot_files(
        snapshot_static_manifest, static_document)
    identity = {
        "schema": "nvidia_a800_two_stage_ldgsts_capture_identity_v3",
        "hardware_id": HARDWARE_ID,
        "owner": arguments.owner,
        "baseline_commit": baseline,
        "run_id": arguments.run_id,
        "device_uuid": arguments.device_uuid,
        "pci_bus_id": _normalize_pci_bus_id(arguments.pci_bus_id),
        "required_sm_clock_mhz": arguments.sm_clock_mhz,
        "repo_root": str(repo_root),
        "static_manifest_source_path": str(static_manifest),
        "pid": os.getpid(),
        "started_unix_s": time.time(),
        "output_path": str(output_dir),
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
    }
    commands = _command_document(
        identity, arguments.capture_mode, arguments.trials, tool_paths)
    _write_json(output_dir / "capture_identity.json", identity)
    _write_json(output_dir / "commands.json", commands)
    _write_json(output_dir / "preflight_execution_state.json", preflight_state)
    _write_json(output_dir / "static_contract_audit.json", static_audit)
    _write_json(output_dir / "static_artifact_audit.json", static_file_audit)
    _write_json(output_dir / "static_snapshot_audit.json", static_snapshot_audit)

    binary = build_dir / "two_stage_ldgsts_loader"
    ptxas_path = build_dir / "ptxas.stderr"
    build_env = os.environ.copy()
    build_env["SOURCE_DATE_EPOCH"] = "0"
    _run(
        commands["compile_argv"], cwd=repo_root, env=build_env,
        stdout_path=build_dir / "compile.stdout", stderr_path=ptxas_path)
    validate_ptxas(ptxas_path.read_text(encoding="utf-8"))
    _run(commands["strip_argv"], cwd=repo_root)
    sass_path = build_dir / "two_stage_ldgsts_loader.sass"
    _run(
        commands["sass_argv"], cwd=repo_root, stdout_path=sass_path,
        stderr_path=build_dir / "cuobjdump.stderr")
    sass_audit = validate_sass(sass_path.read_text(encoding="utf-8"))
    _write_json(build_dir / "sass_audit.json", sass_audit)
    toolchain = {}
    for name, path_text in tool_paths.items():
        path = Path(path_text)
        version = _run(commands["tool_version_argv"][name], cwd=repo_root)
        toolchain[name] = {
            "path": path_text, "sha256": _sha256(path),
            "version_stdout": version.stdout, "version_stderr": version.stderr,
        }
    _write_json(output_dir / "toolchain.json", toolchain)

    execution_env = os.environ.copy()
    execution_env["CUDA_VISIBLE_DEVICES"] = arguments.device_uuid
    native_path = native_dir / "sweep.csv"
    _run_with_device_boundary(
       commands["native_argv"], cwd=repo_root, env=execution_env,
       stdout_path=native_path,
       stderr_path=native_dir / "sweep.stderr",
       boundary_path=native_dir / "execution_boundary.json",
       label=f"native_{arguments.capture_mode}",
       nvidia_smi=tool_paths["nvidia_smi"],
       device_uuid=arguments.device_uuid, pci_bus_id=arguments.pci_bus_id,
       expected_sm_clock_mhz=arguments.sm_clock_mhz)
    native = parse_native_output(
        native_path.read_text(encoding="utf-8"), arguments.trials,
        arguments.capture_mode)
    if native["meta"]["device_uuid"] != arguments.device_uuid:
        raise ValueError("native CUDA UUID disagrees with preflight")
    if _normalize_pci_bus_id(native["meta"]["pci_bus_id"]) != _normalize_pci_bus_id(
            arguments.pci_bus_id):
        raise ValueError("native CUDA PCI identity disagrees with preflight")

    if arguments.capture_mode == "full_matrix":
        cells = NCU_CELLS
    elif arguments.capture_mode == "smoke_cell":
        cells = (SMOKE_CELL,)
    else:
        raise ValueError(f"unknown capture mode {arguments.capture_mode!r}")
    ncu_audit = {}
    for grid, tiles in cells:
        name = f"g{grid}_t{tiles}"
        csv_path = ncu_dir / f"{name}.csv"
        _run_with_device_boundary(
           commands["ncu_argv"][name], cwd=repo_root, env=execution_env,
           stdout_path=ncu_dir / f"{name}.stdout",
           stderr_path=ncu_dir / f"{name}.stderr",
           boundary_path=ncu_dir / f"{name}.execution_boundary.json",
           label=f"ncu_{name}", nvidia_smi=tool_paths["nvidia_smi"],
           device_uuid=arguments.device_uuid,
           pci_bus_id=arguments.pci_bus_id,
           expected_sm_clock_mhz=arguments.sm_clock_mhz)
        metrics = parse_ncu_csv(csv_path.read_text(encoding="utf-8"), grid)
        geometry = validate_ncu_contract(grid, tiles, metrics)
        ncu_audit[name] = {
            "grid": grid, "tiles": tiles, "metrics": metrics,
            "counter_geometry": geometry,
        }
    _write_json(ncu_dir / "audit.json", ncu_audit)
    candidate = build_candidate(
        native, ncu_audit, _sha256(native_path), _sha256(static_manifest))
    _write_json(output_dir / "candidate.json", candidate)
    postflight_state = _execution_state(
        tool_paths["nvidia_smi"], arguments.device_uuid, repo_root)
    _require_quiescent(
        postflight_state, arguments.device_uuid, arguments.pci_bus_id,
        arguments.sm_clock_mhz)
    if postflight_state["device"]["uuid"] != preflight_state["device"]["uuid"]:
        raise ValueError("postflight device identity changed")
    _write_json(output_dir / "postflight_execution_state.json", postflight_state)
    _write_json(output_dir / "report.json", {
        "schema": "nvidia_a800_two_stage_ldgsts_capture_v3",
        "baseline_commit": baseline,
        "source_sha256": _sha256(snapshot_dir / SOURCE_NAME),
        "runner_sha256": _sha256(snapshot_dir / RUNNER_NAME),
        "binary_sha256": _sha256(binary),
        "sass_sha256": _sha256(sass_path),
        "native_sha256": _sha256(native_path),
        "static_manifest_sha256": _sha256(static_manifest),
        "static_snapshot_manifest_sha256": _sha256(snapshot_static_manifest),
        "capture_mode": arguments.capture_mode,
        "trials_per_cell": arguments.trials,
        "required_sm_clock_mhz": arguments.sm_clock_mhz,
        "ncu_cache_control": "none",
        "ncu_clock_control": "none",
        "rate_unit": RATE_UNIT,
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
    })
    anchor = _write_manifest(output_dir)
    return {
        "capture_dir": str(output_dir),
        "detached_manifest_sha256": anchor["manifest_sha256"],
    }


def capture_reuse(arguments):
    """Capture only cache traffic for the frozen grouped-MLA reuse sweep."""

    repo_root = arguments.repo_root.resolve()
    output_dir = arguments.output_dir.resolve()
    allowed_root = (
        repo_root / "profiler_output/microbench/nvidia_a800").resolve()
    if output_dir.parent != allowed_root:
        raise ValueError(f"output_dir must be one direct child of {allowed_root}")
    source = repo_root / "microbench/nvidia_a800/memory/lsu" / SOURCE_NAME
    runner = repo_root / "microbench/nvidia_a800/memory/lsu" / RUNNER_NAME
    static_manifest = arguments.mla_static_manifest.resolve()
    tool_paths = {
        "nvcc": str(arguments.nvcc.resolve()),
        "cuobjdump": str(arguments.cuobjdump.resolve()),
        "strip": str(arguments.strip.resolve()),
        "ncu": str(arguments.ncu.resolve()),
        "nvidia_smi": str(arguments.nvidia_smi.resolve()),
    }
    for path in (source, runner, static_manifest, arguments.nvcc,
                 arguments.cuobjdump, arguments.strip, arguments.ncu,
                 arguments.nvidia_smi):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    baseline = _git_output(repo_root, "rev-parse", "HEAD")
    if _git_output(repo_root, "status", "--short"):
        raise RuntimeError("reuse capture requires a clean tracked worktree")
    for path in (source, runner):
        relative = str(path.relative_to(repo_root))
        if _git_output(
                repo_root, "ls-files", "--error-unmatch", relative) != relative:
            raise RuntimeError(f"reuse capture source is not tracked: {relative}")
        if path.read_bytes() != _git_file_bytes(repo_root, baseline, relative):
            raise RuntimeError(f"reuse capture source differs from HEAD: {relative}")
    static_document = _read_json(static_manifest)
    static_audit = validate_static_manifest(
        static_document, _sha256(static_manifest))
    static_file_audit = validate_static_artifact_files(
        static_manifest, static_document)
    preflight_state = _execution_state(
        tool_paths["nvidia_smi"], arguments.device_uuid, repo_root)
    _require_quiescent(
        preflight_state, arguments.device_uuid, arguments.pci_bus_id,
        arguments.sm_clock_mhz)

    output_dir.mkdir(parents=True)
    build_dir = output_dir / "build"
    ncu_dir = output_dir / "ncu"
    snapshot_dir = output_dir / "source_snapshot"
    for directory in (build_dir, ncu_dir, snapshot_dir):
        directory.mkdir()
    shutil.copy2(source, snapshot_dir / SOURCE_NAME)
    shutil.copy2(runner, snapshot_dir / RUNNER_NAME)
    snapshot_static_manifest = snapshot_dir / "mla_stage1_static_manifest.json"
    shutil.copy2(static_manifest, snapshot_static_manifest)
    artifact_relative = Path(static_document["stage1_artifact_directory"])
    artifact_source_dir = (static_manifest.parent / artifact_relative).resolve()
    artifact_snapshot_dir = snapshot_dir / artifact_relative
    artifact_snapshot_dir.mkdir(parents=True)
    for name in (
            "_fwd_grouped_kernel_stage1.cubin",
            "_fwd_grouped_kernel_stage1.ttgir",
            "_fwd_grouped_kernel_stage1.json"):
        shutil.copy2(artifact_source_dir / name, artifact_snapshot_dir / name)
    static_snapshot_audit = validate_static_snapshot_files(
        snapshot_static_manifest, static_document)
    identity = {
        "schema": "nvidia_a800_grouped_mla_kv_reuse_capture_identity_v1",
        "hardware_id": HARDWARE_ID,
        "owner": arguments.owner,
        "baseline_commit": baseline,
        "run_id": arguments.run_id,
        "device_uuid": arguments.device_uuid,
        "pci_bus_id": _normalize_pci_bus_id(arguments.pci_bus_id),
        "required_sm_clock_mhz": arguments.sm_clock_mhz,
        "repo_root": str(repo_root),
        "static_manifest_source_path": str(static_manifest),
        "pid": os.getpid(),
        "started_unix_s": time.time(),
        "output_path": str(output_dir),
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
    }
    case_manifest = build_reuse_case_manifest(
        baseline_commit=baseline, source_sha256=_sha256(source),
        runner_sha256=_sha256(runner))
    binary = build_dir / "two_stage_ldgsts_loader"
    commands = _reuse_capture_command_document(identity, tool_paths)
    _write_json(output_dir / "capture_identity.json", identity)
    _write_json(output_dir / "case_manifest.json", case_manifest)
    _write_json(output_dir / "commands.json", commands)
    _write_json(output_dir / "preflight_execution_state.json", preflight_state)
    _write_json(output_dir / "static_contract_audit.json", static_audit)
    _write_json(output_dir / "static_artifact_audit.json", static_file_audit)
    _write_json(output_dir / "static_snapshot_audit.json", static_snapshot_audit)

    ptxas_path = build_dir / "ptxas.stderr"
    build_env = os.environ.copy()
    build_env["SOURCE_DATE_EPOCH"] = "0"
    _run(
        commands["compile_argv"], cwd=repo_root, env=build_env,
        stdout_path=build_dir / "compile.stdout", stderr_path=ptxas_path)
    validate_ptxas(ptxas_path.read_text(encoding="utf-8"))
    _run(commands["strip_argv"], cwd=repo_root)
    sass_path = build_dir / "two_stage_ldgsts_loader.sass"
    _run(
        commands["sass_argv"], cwd=repo_root, stdout_path=sass_path,
        stderr_path=build_dir / "cuobjdump.stderr")
    sass_text = sass_path.read_text(encoding="utf-8")
    _write_json(build_dir / "sass_audit.json", {
        "loader_service": validate_sass(sass_text),
        "kv_reuse": validate_reuse_sass(sass_text),
    })
    toolchain = {}
    for name, path_text in tool_paths.items():
        path = Path(path_text)
        version = _run(commands["tool_version_argv"][name], cwd=repo_root)
        toolchain[name] = {
            "path": path_text,
            "sha256": _sha256(path),
            "version_stdout": version.stdout,
            "version_stderr": version.stderr,
        }
    _write_json(output_dir / "toolchain.json", toolchain)

    execution_env = os.environ.copy()
    execution_env["CUDA_VISIBLE_DEVICES"] = arguments.device_uuid
    audits = {}
    for case in _reuse_cases():
        case_id = case["case_id"]
        command = commands["ncu_argv"][case_id]
        csv_path = ncu_dir / f"{case_id}.csv"
        _run_with_device_boundary(
            command, cwd=repo_root, env=execution_env,
            stdout_path=ncu_dir / f"{case_id}.stdout",
            stderr_path=ncu_dir / f"{case_id}.stderr",
            boundary_path=ncu_dir / f"{case_id}.execution_boundary.json",
            label=f"ncu_reuse_{case_id}",
            nvidia_smi=tool_paths["nvidia_smi"],
            device_uuid=arguments.device_uuid,
            pci_bus_id=arguments.pci_bus_id,
            expected_sm_clock_mhz=arguments.sm_clock_mhz)
        metrics = parse_reuse_ncu_csv(
            csv_path.read_text(encoding="utf-8"), case)
        audits[case_id] = {
            "case": case,
            "metrics": metrics,
            "cache_line_report": validate_reuse_ncu_contract(case, metrics),
        }
    _write_json(ncu_dir / "audit.json", audits)
    postflight_state = _execution_state(
        tool_paths["nvidia_smi"], arguments.device_uuid, repo_root)
    _require_quiescent(
        postflight_state, arguments.device_uuid, arguments.pci_bus_id,
        arguments.sm_clock_mhz)
    _write_json(output_dir / "postflight_execution_state.json", postflight_state)
    _write_json(output_dir / "report.json", {
        "schema": "nvidia_a800_grouped_mla_kv_reuse_capture_v1",
        "evidence_status": "exact_launch_aggregate_only",
        "baseline_commit": baseline,
        "source_sha256": _sha256(snapshot_dir / SOURCE_NAME),
        "runner_sha256": _sha256(snapshot_dir / RUNNER_NAME),
        "binary_sha256": _sha256(binary),
        "sass_sha256": _sha256(sass_path),
        "static_manifest_sha256": _sha256(snapshot_static_manifest),
        "case_manifest_sha256": _sha256(output_dir / "case_manifest.json"),
        "ncu_audit_sha256": _sha256(ncu_dir / "audit.json"),
        "case_count": len(audits),
        "required_sm_clock_mhz": arguments.sm_clock_mhz,
        "scheduler_entry_order_observed": False,
        "scheduler_entry_order_status": (
            "not_observed_requires_independent_entry_probe"),
        "runtime_active_residency_observed": False,
        "runtime_active_residency_status": (
            "not_observed_requires_independent_residency_probe"),
        "l2_capacity_transition_claimed": False,
        "ncu_cache_control": "all",
        "ncu_clock_control": "none",
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
    })
    anchor = _write_manifest(output_dir)
    return {
        "capture_dir": str(output_dir),
        "detached_manifest_sha256": anchor["manifest_sha256"],
    }


def _arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--device-uuid", required=True)
    parser.add_argument("--pci-bus-id", required=True)
    parser.add_argument(
        "--sm-clock-mhz", type=int, required=True,
        help="externally fixed SM clock required before, during, and after capture")
    parser.add_argument("--mla-static-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--capture-mode", choices=("full_matrix", "smoke_cell"),
                        default="full_matrix")
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--nvcc", type=Path,
                        default=Path("/usr/local/cuda-12.2/bin/nvcc"))
    parser.add_argument("--cuobjdump", type=Path,
                        default=Path("/usr/local/cuda-12.2/bin/cuobjdump"))
    parser.add_argument("--strip", type=Path, default=Path("/usr/bin/strip"))
    parser.add_argument("--ncu", type=Path,
                        default=Path("/usr/local/cuda-12.2/bin/ncu"))
    parser.add_argument("--nvidia-smi", type=Path,
                        default=Path("/usr/bin/nvidia-smi"))
    arguments = parser.parse_args(argv)
    arguments.run_id = _require_nonempty(arguments.run_id, "run_id")
    if RUN_ID_PATTERN.fullmatch(arguments.run_id) is None:
        raise ValueError("run_id must match [A-Za-z0-9._-]{1,96}")
    arguments.owner = _require_nonempty(arguments.owner, "owner")
    arguments.device_uuid = _require_nonempty(arguments.device_uuid, "device_uuid")
    arguments.pci_bus_id = _require_nonempty(arguments.pci_bus_id, "pci_bus_id")
    if (isinstance(arguments.trials, bool) or not isinstance(arguments.trials, int)
            or arguments.trials <= 0 or arguments.trials % 2 == 0):
        raise ValueError("trials must be a positive odd integer")
    if (arguments.capture_mode == "full_matrix"
            and arguments.trials < MIN_CANDIDATE_TRIALS_PER_CELL):
        raise ValueError("full_matrix capture requires at least 7 trials per cell")
    if (isinstance(arguments.sm_clock_mhz, bool)
            or arguments.sm_clock_mhz <= 0):
        raise ValueError("sm_clock_mhz must be a positive integer")
    return arguments


def _verification_arguments(argv):
    parser = argparse.ArgumentParser(prog=f"{RUNNER_NAME} verify")
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--detached-manifest-sha256", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--nvcc", type=Path)
    parser.add_argument("--cuobjdump", type=Path)
    parser.add_argument("--strip", dest="strip_tool", type=Path)
    return parser.parse_args(argv)


def _reuse_verification_arguments(argv):
    parser = argparse.ArgumentParser(prog=f"{RUNNER_NAME} verify-reuse")
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--detached-manifest-sha256", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--nvcc", type=Path)
    parser.add_argument("--cuobjdump", type=Path)
    parser.add_argument("--strip", dest="strip_tool", type=Path)
    return parser.parse_args(argv)


def _reuse_plan_arguments(argv):
    parser = argparse.ArgumentParser(prog=f"{RUNNER_NAME} reuse-plan")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--ncu", type=Path, default=Path("/usr/local/cuda-12.2/bin/ncu"))
    return parser.parse_args(argv)


def _reuse_capture_arguments(argv):
    parser = argparse.ArgumentParser(prog=f"{RUNNER_NAME} reuse-capture")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--device-uuid", required=True)
    parser.add_argument("--pci-bus-id", required=True)
    parser.add_argument(
        "--sm-clock-mhz", type=int,
        help=("optional externally fixed SM clock; omit for this "
              "counter-only capture"))
    parser.add_argument("--mla-static-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--nvcc", type=Path,
                        default=Path("/usr/local/cuda-12.2/bin/nvcc"))
    parser.add_argument("--cuobjdump", type=Path,
                        default=Path("/usr/local/cuda-12.2/bin/cuobjdump"))
    parser.add_argument("--strip", type=Path, default=Path("/usr/bin/strip"))
    parser.add_argument("--ncu", type=Path,
                        default=Path("/usr/local/cuda-12.2/bin/ncu"))
    parser.add_argument("--nvidia-smi", type=Path,
                        default=Path("/usr/bin/nvidia-smi"))
    arguments = parser.parse_args(argv)
    arguments.run_id = _require_nonempty(arguments.run_id, "run_id")
    if RUN_ID_PATTERN.fullmatch(arguments.run_id) is None:
        raise ValueError("run_id must match [A-Za-z0-9._-]{1,96}")
    arguments.owner = _require_nonempty(arguments.owner, "owner")
    arguments.device_uuid = _require_nonempty(
        arguments.device_uuid, "device_uuid")
    arguments.pci_bus_id = _require_nonempty(
        arguments.pci_bus_id, "pci_bus_id")
    if (arguments.sm_clock_mhz is not None
            and (isinstance(arguments.sm_clock_mhz, bool)
                 or arguments.sm_clock_mhz <= 0)):
        raise ValueError("sm_clock_mhz must be a positive integer")
    return arguments


def build_reuse_plan(arguments):
    repo_root = arguments.repo_root.resolve()
    source = repo_root / "microbench/nvidia_a800/memory/lsu" / SOURCE_NAME
    runner = repo_root / "microbench/nvidia_a800/memory/lsu" / RUNNER_NAME
    for path in (source, runner):
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline = _git_output(repo_root, "rev-parse", "HEAD")
    source_relative = str(source.relative_to(repo_root))
    runner_relative = str(runner.relative_to(repo_root))
    if source.read_bytes() != _git_file_bytes(
            repo_root, baseline, source_relative):
        raise RuntimeError("reuse-plan source differs from the Git baseline")
    if runner.read_bytes() != _git_file_bytes(
            repo_root, baseline, runner_relative):
        raise RuntimeError("reuse-plan runner differs from the Git baseline")
    return {
        "case_manifest": build_reuse_case_manifest(
            baseline_commit=baseline,
            source_sha256=_sha256(source),
            runner_sha256=_sha256(runner)),
        "command_manifest": build_reuse_command_manifest(
            repo_root=repo_root, output_dir=arguments.output_dir,
            ncu=arguments.ncu, binary=arguments.binary),
    }


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "verify-reuse":
        arguments = _reuse_verification_arguments(values[1:])
        audit = verify_reuse_capture(
            arguments.capture_dir, arguments.detached_manifest_sha256,
            repo_root=arguments.repo_root, nvcc=arguments.nvcc,
            cuobjdump=arguments.cuobjdump, strip_tool=arguments.strip_tool)
        print(json.dumps(audit, indent=2, sort_keys=True))
    elif values and values[0] == "reuse-capture":
        print(json.dumps(
            capture_reuse(_reuse_capture_arguments(values[1:])),
            indent=2, sort_keys=True))
    elif values and values[0] == "reuse-plan":
        print(json.dumps(
            build_reuse_plan(_reuse_plan_arguments(values[1:])),
            indent=2, sort_keys=True))
    elif values and values[0] == "verify":
        arguments = _verification_arguments(values[1:])
        audit = verify_capture(
            arguments.capture_dir, arguments.detached_manifest_sha256,
            repo_root=arguments.repo_root, nvcc=arguments.nvcc,
            cuobjdump=arguments.cuobjdump, strip_tool=arguments.strip_tool)
        print(json.dumps(audit, indent=2, sort_keys=True))
    else:
        result = capture(_arguments(values))
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
