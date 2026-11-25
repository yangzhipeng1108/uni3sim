#!/usr/bin/env python3
"""Capture independent and chained H8192 A800 Add-RMS phase service."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = REPOSITORY_ROOT / "profiler_output/microbench/nvidia_a800"
RUNNER_PATH = Path(__file__).resolve()
SOURCE_PATH = RUNNER_PATH.with_name("add_rmsnorm_round_service_1024_h8192.cu")
REPORT_PATH = RUNNER_PATH.with_name("add_rmsnorm_round_service_1024_h8192_report.py")
SOURCE_PIN_PATH = RUNNER_PATH.with_name(
    "add_rmsnorm_round_service_vllm_v0_11_0_excerpt.cu")
BASE_RUNNER_PATH = RUNNER_PATH.with_name("add_rmsnorm_round_service_runner.py")
UPSTREAM_REFERENCE_PATH = (
    REPOSITORY_ROOT
    / "microbench/nvidia_a800/references/vllm_rmsnorm_v0_11_0_sm80.json")

_BASE_SPEC = importlib.util.spec_from_file_location(
    "add_rmsnorm_round_service_base", BASE_RUNNER_PATH)
BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(BASE)

DEFAULT_NVCC = BASE.DEFAULT_NVCC
DEFAULT_CUOBJDUMP = BASE.DEFAULT_CUOBJDUMP
DEFAULT_NVIDIA_SMI = BASE.DEFAULT_NVIDIA_SMI
DEFAULT_PRODUCTION_BINARY = BASE.DEFAULT_PRODUCTION_BINARY
EXPECTED_SOURCE_PIN_SHA256 = (
    "4439703d34351e08d32825b05f04086bc30d28d590fb490f6f15222fe21bade6")
EXPECTED_UPSTREAM_REFERENCE_SHA256 = (
    "5cfc88825fe890d0881a9d870e04d19dbd59c9c9af5a9d0feddacb8add72a53e")
EXPECTED_UPSTREAM_SOURCE_SHA256 = (
    "fb3b6102d6c0efdf66ed7a2030d0bebf6a25aa46e07fe7274ec545f39e200409")
EXPECTED_THREADS = 1024
EXPECTED_HIDDEN_SIZE = 8192
EXPECTED_ACTIVE_THREADS = 1024
EXPECTED_ACTIVE_WARPS = 32
EXPECTED_REGISTERS = 32
EXPECTED_STATIC_SHARED_BYTES = 168
EXPECTED_RESIDENT_CTAS = 2
EXPECTED_TRIALS = 3
EXPECTED_SAMPLES = 200
EXPECTED_GRIDS = (1, 4, 8, 16, 64, 108)
EXPECTED_FUNCTIONS = (
    "add_square", "scale_multiply", "publication_barrier", "phase_chain")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
RESOURCE_PATTERN = re.compile(
    r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+) CONSTANT\[0\]:(\d+)")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_output(run_id):
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run-id contains unsupported characters")
    output = (OUTPUT_ROOT / run_id).resolve()
    if output.parent != OUTPUT_ROOT.resolve():
        raise ValueError("run-id escaped the A800 profiler output root")
    if output.exists():
        raise FileExistsError(output)
    return output


def _validate_probe_source(path=SOURCE_PATH):
    text = path.read_text(encoding="utf-8")
    required = (
        "constexpr int kThreads = 1024;",
        "constexpr int kHiddenSize = 8192;",
        "constexpr int kActiveThreads = kHiddenSize / kVectorWidth;",
        "constexpr int kActiveWarps = kActiveThreads / 32;",
        "constexpr int kResidentCtas = 2;",
        "constexpr int kStaticSharedBytes = 168;",
        "constexpr int kTrials = 3;",
        "constexpr int kSamples = 200;",
        'asm volatile("mov.u64 %0, %%clock64;"',
        "__launch_bounds__(kThreads, kResidentCtas)",
        "RoundFunction::ADD_SQUARE",
        "RoundFunction::SCALE_MULTIPLY",
        "RoundFunction::PUBLICATION_BARRIER",
        "RoundFunction::PHASE_CHAIN",
        "void add_rmsnorm_publication_barrier_1024_h8192(",
        "void add_rmsnorm_phase_chain_1024_h8192(",
        "__hadd2(left_pairs[pair], right_pairs[pair])",
        "const float2 values = __bfloat1622float2(added);",
        "*reinterpret_cast<volatile float*>(&shared.s_variance)",
        "__floats2bfloat162_rn(scaled.x, scaled.y)",
        "__hmul2(rounded, weight_pairs[pair])",
        "if (threadIdx.x < kActiveThreads)",
        "const Pack right = residual[offset];",
        "residual[offset] = result;",
        "const Pack value = residual[offset];",
        "input[offset] = result;",
        "for (int sample = 0; sample < kSamples; ++sample)",
        "{RoundFunction::ADD_SQUARE, 1}",
        "{RoundFunction::SCALE_MULTIPLY, 108}",
    )
    missing = tuple(token for token in required if token not in text)
    forbidden = tuple(
        token for token in (
            "BlockReduce", "rsqrt", "MUFU", "production duration", "0.875f",
            "fmaf(", "kRounds", "kBufferSlots", "residual_input",
            "residual_output", "Pack* __restrict__ output",
            "read_globaltimer", "%globaltimer")
        if token in text)
    if missing or forbidden or text.count("read_clock64();") != 6:
        raise ValueError(
            f"probe source partition changed: missing={missing}, "
            f"forbidden={forbidden}")
    for grid in EXPECTED_GRIDS:
        for function in (
                "ADD_SQUARE", "SCALE_MULTIPLY", "PUBLICATION_BARRIER",
                "PHASE_CHAIN"):
            token = f"{{RoundFunction::{function}, {grid}}}"
            if text.count(token) != 1:
                raise ValueError(f"probe grid/function matrix changed: {token}")
    smid_store = text.index("smids[blockIdx.x] = read_smid();")
    initial_barrier = text.index("__syncthreads();")
    start = text.index("begin_cycles = read_clock64();")
    add = text.index(
        "if constexpr (function == RoundFunction::ADD_SQUARE)", start)
    sink = text.index("sink[blockIdx.x * kActiveThreads + threadIdx.x]")
    end = text.index("finish_cycles = read_clock64();")
    post_barrier = text.index("__syncthreads();", end)
    timestamp_store = text.index(
        "start_cycles[timer_offset] = begin_cycles;", end)
    padding = text.index("enforce_resource_identity(&shared)", end)
    if (not smid_store < initial_barrier < start < add < end
            < post_barrier < timestamp_store < padding < sink
            or "__syncthreads" in text[start:end]
            or "enforce_resource_identity" in text[start:end]
            or "start_cycles[" in text[start:end]
            or "end_cycles[" in text[start:end]
            or "%globaltimer" in text):
        raise ValueError("probe timed source-round interval changed")
    publication = text.index(
        "void add_rmsnorm_publication_barrier_1024_h8192(")
    publication_start = text.index("begin_cycles = read_clock64();", publication)
    publication_store = text.index("shared.s_variance =", publication_start)
    publication_barrier = text.index("__syncthreads();", publication_store)
    publication_load = text.index(
        "*reinterpret_cast<volatile float*>(&shared.s_variance)",
        publication_barrier)
    publication_end = text.index(
        "finish_cycles = read_clock64();", publication_load)
    phase_chain = text.index("void add_rmsnorm_phase_chain_1024_h8192(")
    phase_start = text.index("begin_cycles = read_clock64();", phase_chain)
    phase_add = text.index("const Pack left = input[offset];", phase_start)
    phase_residual_store = text.index("residual[offset] = added_pack;", phase_add)
    phase_publish = text.index("shared.s_variance =", phase_residual_store)
    phase_barrier = text.index("__syncthreads();", phase_publish)
    phase_residual_load = text.index(
        "const Pack value = residual[offset];", phase_barrier)
    phase_scale_load = text.index(
        "*reinterpret_cast<volatile float*>(&shared.s_variance)",
        phase_residual_load)
    phase_input_store = text.index("input[offset] = result;", phase_scale_load)
    phase_end = text.index("finish_cycles = read_clock64();", phase_input_store)
    if not (
            publication_start < publication_store < publication_barrier
            < publication_load < publication_end
            and phase_start < phase_add < phase_residual_store < phase_publish
            < phase_barrier < phase_residual_load < phase_scale_load
            < phase_input_store < phase_end):
        raise ValueError("phase-chain dependency partition changed")
    return {
        "threads_per_cta": EXPECTED_THREADS,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "active_threads": EXPECTED_ACTIVE_THREADS,
        "active_warps": EXPECTED_ACTIVE_WARPS,
        "registers_per_thread": EXPECTED_REGISTERS,
        "static_shared_bytes": EXPECTED_STATIC_SHARED_BYTES,
        "resident_ctas_per_sm": EXPECTED_RESIDENT_CTAS,
        "grids": list(EXPECTED_GRIDS),
        "trials_per_cell": EXPECTED_TRIALS,
        "samples_per_trial_cell": EXPECTED_SAMPLES,
        "functions": list(EXPECTED_FUNCTIONS),
        "source_rounds_per_kernel_invocation": 1,
        "phase_chain_source_rounds_per_kernel_invocation": 2,
        "timer_source": "clock64",
        "timer_unit": "cycles",
        "timer_comparison_scope": "same_cta_same_sm",
        "input_and_residual_restored_before_each_sample": True,
        "add_residual_load_store_same_address": True,
        "scale_residual_load_and_input_store": True,
        "timer_values_stay_in_registers_until_post_timed_barrier": True,
        "all_threads_reach_post_timed_barrier": True,
        "scale_value_source": "runtime_kernel_argument_written_to_shared_s_variance",
        "phase_chain_middle_stage": (
            "thread0_dependency_publication_then_cta_barrier"),
        "phase_chain_dependency_source": "thread0_add_square_variance",
        "publication_control_dependency_source": "runtime_kernel_argument",
        "reduction_or_rsqrt_absent": True,
        "checksum_or_resource_padding_in_timed_region": False,
    }


def _validate_source_pin(path=SOURCE_PIN_PATH):
    digest = _sha256(path)
    if digest != EXPECTED_SOURCE_PIN_SHA256:
        raise ValueError("pinned vLLM Add-RMSNorm excerpt changed")
    text = path.read_text(encoding="utf-8")
    required = (
        "Commit: b8b302cde434df8c9289a2b465406b47ebab1c2d",
        f"csrc/layernorm_kernels.cu {EXPECTED_UPSTREAM_SOURCE_SHA256}",
        "csrc/type_convert.cuh 1c7cc528f08f14eb4fcb7cee50255d4437db8301335412af3bb86b3a248ad5e2",
        "fused_add_rms_norm_kernel(",
        "_f16Vec<scalar_t, width> temp = input_v[strided_id];",
        "temp += residual_v[id];",
        "variance += temp.sum_squares();",
        "residual_v[id] = temp;",
        "using BlockReduce = cub::BlockReduce<float, 1024>;",
        "s_variance = rsqrtf(variance / hidden_size + epsilon);",
        "temp *= s_variance;",
        "temp *= weight_v[idx];",
        "input_v[strided_id] = temp;",
        "const int max_block_size = (num_tokens < 256) ? 1024 : 256;",
        "dim3 block(std::min(hidden_size, max_block_size));",
        "constexpr int vector_width = 8;",
        "temp += T2{other.data[i], other.data[i + 1]};",
        "temp_f.x *= scale;",
        "T2 temp = Converter::convert(temp_f);",
        "temp *= T2{other.data[i], other.data[i + 1]};",
    )
    missing = tuple(token for token in required if token not in text)
    if missing:
        raise ValueError(f"pinned production source is incomplete: {missing}")
    add_begin = text.index("temp += residual_v[id];")
    add_end = text.index("residual_v[id] = temp;")
    reduction_begin = text.index("using BlockReduce")
    reduction_end = text.index("__syncthreads();", reduction_begin)
    scale_begin = text.index("temp *= s_variance;")
    scale_end = text.index("input_v[strided_id] = temp;")
    if not add_begin < add_end < reduction_begin < reduction_end < scale_begin < scale_end:
        raise ValueError("production phase order changed")
    return {
        "upstream_repository": "https://github.com/vllm-project/vllm.git",
        "upstream_commit": "b8b302cde434df8c9289a2b465406b47ebab1c2d",
        "upstream_path": "csrc/layernorm_kernels.cu",
        "upstream_full_source_sha256": EXPECTED_UPSTREAM_SOURCE_SHA256,
        "type_convert_path": "csrc/type_convert.cuh",
        "type_convert_full_source_sha256": (
            "1c7cc528f08f14eb4fcb7cee50255d4437db8301335412af3bb86b3a248ad5e2"),
        "pinned_excerpt_sha256": digest,
        "phase_order": [
            "add_square_source_round", "reduction_and_publication",
            "scale_multiply_source_round"],
        "launch_rule": "num_tokens < 256 selects block=1024",
        "vector_width_elements": 8,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "active_source_threads": EXPECTED_ACTIVE_THREADS,
    }


def _validate_upstream_reference(path=UPSTREAM_REFERENCE_PATH):
    if _sha256(path) != EXPECTED_UPSTREAM_REFERENCE_SHA256:
        raise ValueError("versioned vLLM RMSNorm/Add-RMSNorm reference changed")
    reference = json.loads(path.read_text(encoding="utf-8"))
    if (reference["production_or_evaluation_duration_consumed"] is not False
            or reference["runtime_stack"]["vllm_source_commit"]
            != "b8b302cde434df8c9289a2b465406b47ebab1c2d"
            or reference["upstream_sources"][0]["sha256"]
            != EXPECTED_UPSTREAM_SOURCE_SHA256):
        raise ValueError("versioned production source reference changed")
    kernel = reference["add_rmsnorm_kernel"]
    if (kernel["vector_width_elements"] != 8
            or kernel["registers_per_thread_sm80"] != EXPECTED_REGISTERS
            or kernel["static_shared_memory_bytes_per_cta_sm80"]
            != EXPECTED_STATIC_SHARED_BYTES
            or kernel["resident_ctas_per_sm_by_block_threads"]["1024"]
            != EXPECTED_RESIDENT_CTAS):
        raise ValueError("versioned production Add-RMSNorm resources changed")
    return {
        "path": str(UPSTREAM_REFERENCE_PATH.relative_to(REPOSITORY_ROOT)),
        "sha256": EXPECTED_UPSTREAM_REFERENCE_SHA256,
        "production_or_evaluation_duration_consumed": False,
    }


def _parse_probe_resources(text):
    resources = {}
    current = None
    for line in text.splitlines():
        function = BASE.SYMBOL_PATTERN.search(line)
        if function is not None:
            current = function.group(1)
            continue
        if (current is None
                or ("add_rmsnorm_round_service_1024_h8192" not in current
                    and "add_rmsnorm_publication_barrier_1024_h8192"
                    not in current
                    and "add_rmsnorm_phase_chain_1024_h8192" not in current)):
            continue
        match = RESOURCE_PATTERN.search(line)
        if match is None:
            continue
        if "RoundFunction0" in current:
            name = "add_square"
        elif "RoundFunction1" in current:
            name = "scale_multiply"
        elif "add_rmsnorm_publication_barrier_1024_h8192" in current:
            name = "publication_barrier"
        elif "add_rmsnorm_phase_chain_1024_h8192" in current:
            name = "phase_chain"
        else:
            raise ValueError(f"unknown probe specialization: {current}")
        if name in resources:
            raise ValueError(f"duplicate probe specialization: {name}")
        resources[name] = {
            "symbol": current,
            "registers_per_thread": int(match.group(1)),
            "stack_bytes": int(match.group(2)),
            "static_shared_bytes": int(match.group(3)),
            "local_bytes": int(match.group(4)),
            "constant_bytes": int(match.group(5)),
        }
    if set(resources) != set(EXPECTED_FUNCTIONS):
        raise ValueError(
            "probe must contain two source-round, publication-control, and "
            "phase-chain kernels")
    for name in EXPECTED_FUNCTIONS:
        resource = resources[name]
        if (resource["registers_per_thread"] != EXPECTED_REGISTERS
                or resource["stack_bytes"] != 0
                or resource["static_shared_bytes"]
                != EXPECTED_STATIC_SHARED_BYTES
                or resource["local_bytes"] != 0):
            raise ValueError(f"probe resource identity changed for {name}")
    return resources


def _sass_functions(text):
    functions = {}
    matches = tuple(re.finditer(r"^\s*Function\s*:\s*(\S+)\s*$", text, re.MULTILINE))
    for index, match in enumerate(matches):
        begin = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        symbol = match.group(1)
        if ("add_rmsnorm_round_service_1024_h8192" in symbol
                or "add_rmsnorm_publication_barrier_1024_h8192" in symbol
                or "add_rmsnorm_phase_chain_1024_h8192" in symbol):
            functions[symbol] = text[begin:end]
    return functions


def _validate_probe_sass(text, resources):
    functions = _sass_functions(text)
    by_name = {}
    for name in EXPECTED_FUNCTIONS:
        symbol = resources[name]["symbol"]
        if symbol not in functions:
            raise ValueError(f"probe SASS is missing {name}")
        body = functions[symbol]
        required = {
            "add_square": ("LDG.E.128", "STG.E.128", "BAR.SYNC"),
            "scale_multiply": (
                "LDG.E.128", "STG.E.128", "LDS", "BAR.SYNC"),
            "publication_barrier": ("STS", "LDS", "BAR.SYNC"),
            "phase_chain": (
                "LDG.E.128", "STG.E.128", "STS", "LDS", "BAR.SYNC"),
        }[name]
        missing = tuple(opcode for opcode in required if opcode not in body)
        forbidden = tuple(
            opcode for opcode in ("MUFU.RSQ", "SHFL", "ATOM", "LDL", "STL")
            if opcode in body)
        timer_matches = tuple(re.finditer(
            r"^.*\b(?:CS2R|S2R)\b.*\bSR_CLOCKLO\b.*$", body, re.MULTILINE))
        if len(timer_matches) == 2:
            first_timer = timer_matches[0].start()
            second_timer = timer_matches[1].start()
            timed_body = body[first_timer:second_timer]
            post_timer_body = body[timer_matches[1].end():]
        else:
            first_timer = -1
            second_timer = -1
            timed_body = ""
            post_timer_body = ""
        opcodes = re.findall(
            r"/\*[0-9a-f]+\*/\s+(?:@[!P0-9]+\s+)?([A-Z][A-Z0-9_.]*)",
            timed_body)
        counts = Counter(opcodes)
        expected_counts = {
            "add_square": {
                "CS2R": 1, "MOV": 1, "IMAD.WIDE": 2,
                "LDG.E.128.CONSTANT": 1, "LDG.E.128": 1, "STG.E.128": 1,
                "HFMA2.BF16_V2": 4, "PRMT": 8,
                "FMUL": 4, "FFMA": 4, "FADD": 4,
                "LDS": 0, "IMAD.WIDE.U32": 0,
                "F2FP.BF16.PACK_AB": 0,
            },
            "scale_multiply": {
                "CS2R": 1, "MOV": 0, "IMAD.WIDE": 1,
                "IMAD.WIDE.U32": 1,
                "LDG.E.128.CONSTANT": 2, "STG.E.128": 1,
                "LDS": 1, "PRMT": 8, "FMUL": 8,
                "F2FP.BF16.PACK_AB": 4, "HFMA2.BF16_V2": 4,
                "LDG.E.128": 0, "FFMA": 0, "FADD": 0,
            },
            "publication_barrier": {
                "CS2R": 2, "MOV": 2, "FFMA": 1,
                "ISETP.GT.U32.AND": 1, "STS": 1, "BSSY": 1,
                "BAR.SYNC": 1, "LDS": 1, "BRA": 1, "LOP3.LUT": 1,
            },
            "phase_chain": {
                "CS2R": 2, "MOV": 1, "LDG.E.128": 3,
                "HFMA2.BF16_V2": 8, "PRMT": 16, "FMUL": 12,
                "FFMA": 5, "FADD": 4, "STG.E.128": 2,
                "BSYNC": 1, "ISETP.NE.AND": 1, "BSSY": 1,
                "STS": 1, "BAR.SYNC": 1, "BRA": 1,
                "IMAD.WIDE.U32": 1, "LDS": 1,
                "LDG.E.128.CONSTANT": 1, "LOP3.LUT": 1,
                "F2FP.BF16.PACK_AB": 4,
            },
        }[name]
        wrong_counts = {
            opcode: (counts[opcode], expected)
            for opcode, expected in expected_counts.items()
            if counts[opcode] != expected}
        unexpected_opcodes = tuple(sorted(set(counts) - set(expected_counts)))
        if name == "add_square":
            key_order = (
                "LDG.E.128.CONSTANT", "LDG.E.128", "HFMA2.BF16_V2",
                "PRMT", "STG.E.128")
        elif name == "scale_multiply":
            key_order = (
                "LDS", "FMUL", "F2FP.BF16.PACK_AB", "HFMA2.BF16_V2",
                "STG.E.128")
        elif name == "publication_barrier":
            key_order = ("STS", "BAR.SYNC", "LDS")
        elif name == "phase_chain":
            key_order = (
                "HFMA2.BF16_V2", "STG.E.128", "STS", "BAR.SYNC", "LDS",
                "F2FP.BF16.PACK_AB")
        else:
            raise ValueError(f"unknown source-round function {name!r}")
        if all(opcode in opcodes for opcode in key_order):
            first_positions = tuple(opcodes.index(opcode) for opcode in key_order)
            ordered = tuple(sorted(first_positions)) == first_positions
        else:
            ordered = False
        packed_lines = tuple(
            line for line in timed_body.splitlines()
            if "HFMA2.BF16_V2" in line)
        if name == "add_square":
            packed_semantics = (
                len(packed_lines) == 4
                and all(", 1, 1," in line for line in packed_lines))
        elif name == "scale_multiply":
            lds_match = re.search(r"\bLDS\s+(R\d+),", timed_body)
            multiply_lines = tuple(
                line for line in timed_body.splitlines()
                if re.search(r"\bFMUL\b", line))
            packed_semantics = (
                lds_match is not None and len(multiply_lines) == 8
                and all(lds_match.group(1) in line for line in multiply_lines)
                and len(packed_lines) == 4
                and all("-RZ.H0_H0" in line for line in packed_lines))
        elif name == "publication_barrier":
            packed_semantics = not packed_lines
        elif name == "phase_chain":
            packed_semantics = (
                len(packed_lines) == 8
                and all(", 1, 1," in line for line in packed_lines[:4])
                and all("-RZ.H0_H0" in line for line in packed_lines[4:]))
        else:
            raise ValueError(f"unknown source-round function {name!r}")
        timed_global_stores = tuple(
            opcode for opcode in opcodes if opcode.startswith("STG"))
        timed_shared_stores = tuple(
            opcode for opcode in opcodes if opcode.startswith("STS"))
        timed_shared_loads = tuple(
            opcode for opcode in opcodes if opcode.startswith("LDS"))
        if name == "add_square":
            expected_shared_loads = ()
        elif name == "scale_multiply":
            expected_shared_loads = ("LDS",)
        elif name in ("publication_barrier", "phase_chain"):
            expected_shared_loads = ("LDS",)
        else:
            raise ValueError(f"unknown source-round function {name!r}")
        expected_global_stores = {
            "add_square": ("STG.E.128",),
            "scale_multiply": ("STG.E.128",),
            "publication_barrier": (),
            "phase_chain": ("STG.E.128", "STG.E.128"),
        }[name]
        expected_shared_stores = {
            "add_square": (),
            "scale_multiply": (),
            "publication_barrier": ("STS",),
            "phase_chain": ("STS",),
        }[name]
        expected_timed_barriers = {
            "add_square": 0,
            "scale_multiply": 0,
            "publication_barrier": 1,
            "phase_chain": 1,
        }[name]
        post_timer_barrier = post_timer_body.find("BAR.SYNC")
        post_timer_barrier_line = next(
            (line for line in post_timer_body.splitlines()
             if "BAR.SYNC" in line), "")
        post_timer_barrier_unpredicated = (
            bool(post_timer_barrier_line)
            and "@" not in post_timer_barrier_line)
        first_post_timer_memory = re.search(
            r"\b(?:LDS|STS|LDG|STG)(?:\.[A-Z0-9_.]+)?\b", post_timer_body)
        barrier_precedes_post_timer_memory = (
            post_timer_barrier >= 0
            and first_post_timer_memory is not None
            and post_timer_barrier < first_post_timer_memory.start())
        if (missing or forbidden or len(timer_matches) != 2
                or body.count("SR_CLOCKLO") != 2
                or "SR_GLOBALTIMER" in body
                or timed_body.count("BAR.SYNC") != expected_timed_barriers
                or wrong_counts
                or unexpected_opcodes
                or not ordered or not packed_semantics
                or timed_global_stores != expected_global_stores
                or timed_shared_stores != expected_shared_stores
                or timed_shared_loads != expected_shared_loads
                or not barrier_precedes_post_timer_memory
                or not post_timer_barrier_unpredicated):
            raise ValueError(
                f"probe SASS contract failed for {name}: missing={missing}, "
                f"forbidden={forbidden}, clock64_reads={len(timer_matches)}, "
                f"globaltimer_present={'SR_GLOBALTIMER' in body}, "
                f"timed_barriers={timed_body.count('BAR.SYNC')}, "
                f"wrong_counts={wrong_counts}, ordered={ordered}, "
                f"unexpected_opcodes={unexpected_opcodes}, "
                f"packed_semantics={packed_semantics}, "
                f"timed_global_stores={timed_global_stores}, "
                f"timed_shared_stores={timed_shared_stores}, "
                f"timed_shared_loads={timed_shared_loads}, "
                f"post_timer_barrier_before_memory="
                f"{barrier_precedes_post_timer_memory}, "
                f"post_timer_barrier_unpredicated="
                f"{post_timer_barrier_unpredicated}")
        by_name[name] = {
            "symbol": symbol,
            "required_opcodes": list(required),
            "timed_opcode_counts": expected_counts,
            "timed_key_opcode_order": list(key_order),
            "packed_bf16_semantics_exact": True,
            "forbidden_opcodes_absent": [
                "MUFU.RSQ", "SHFL", "ATOM", "LDL", "STL"],
            "clock64_reads": 2,
            "globaltimer_reads": 0,
            "timed_barrier_absent": expected_timed_barriers == 0,
            "timed_dependency_barrier_count": expected_timed_barriers,
            "timed_global_stores": list(expected_global_stores),
            "timed_shared_stores": list(expected_shared_stores),
            "timed_shared_loads": list(expected_shared_loads),
            "post_timer_barrier_precedes_padding_and_timestamp_stores": True,
            "post_timer_block_barrier_unpredicated": True,
            "reduction_or_rsqrt_absent": True,
        }
    return by_name


def _occupancy_proof(resources):
    limits = {
        "registers_per_sm": 65_536,
        "shared_bytes_per_sm": 163_840,
        "threads_per_sm": 2_048,
        "blocks_per_sm": 32,
    }
    by_function = {}
    for function in EXPECTED_FUNCTIONS:
        resource = resources[function]
        boundaries = {
            "register_limit": (
                limits["registers_per_sm"]
                // (EXPECTED_THREADS * resource["registers_per_thread"])),
            "shared_memory_limit": (
                limits["shared_bytes_per_sm"]
                // resource["static_shared_bytes"]),
            "thread_limit": limits["threads_per_sm"] // EXPECTED_THREADS,
            "block_limit": limits["blocks_per_sm"],
        }
        resident = min(boundaries.values())
        if resident != EXPECTED_RESIDENT_CTAS:
            raise ValueError(f"probe occupancy boundary changed for {function}")
        by_function[function] = {
            **boundaries, "resident_ctas_per_sm": resident}
    return {"sm_limits": limits, "by_function": by_function}


def _git_identity():
    commit = BASE._run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
        capture_output=True).stdout.strip()
    return commit


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _static_contract(output, args):
    source_contract = _validate_probe_source()
    source_pin = _validate_source_pin()
    upstream_reference = _validate_upstream_reference()
    tools = {
        "nvcc": BASE._tool_identity(
            args.nvcc.resolve(), ("--version",), BASE.EXPECTED_NVCC_SHA256),
        "cuobjdump": BASE._tool_identity(
            args.cuobjdump.resolve(), ("--version",),
            BASE.EXPECTED_CUOBJDUMP_SHA256),
    }
    snapshot = output / "source_snapshot"
    build = output / "build"
    snapshot.mkdir()
    build.mkdir()
    for path in (
            SOURCE_PATH, RUNNER_PATH, REPORT_PATH, SOURCE_PIN_PATH,
            UPSTREAM_REFERENCE_PATH):
        shutil.copy2(path, snapshot / path.name)
    production = BASE._audit_production_artifact(
        args.production_binary.resolve(), args.cuobjdump.resolve(), build)
    binary = build / "add_rmsnorm_round_service_1024_h8192"
    compile_command = [
        str(args.nvcc.resolve()), "-O3", "-std=c++17", "-arch=sm_80",
        "-lineinfo", "-Xptxas=-v", str(snapshot / SOURCE_PATH.name),
        "-o", str(binary),
    ]
    with (build / "compile.stdout").open("w", encoding="utf-8") as stdout, (
            build / "compile.stderr").open("w", encoding="utf-8") as stderr:
        BASE._run(
            compile_command, cwd=REPOSITORY_ROOT, stdout=stdout, stderr=stderr)
    resource_result = BASE._run([
        str(args.cuobjdump.resolve()), "--dump-resource-usage", str(binary)],
        cwd=REPOSITORY_ROOT, capture_output=True)
    resource_path = build / "probe_resource_usage.txt"
    resource_path.write_text(resource_result.stdout, encoding="utf-8")
    resources = _parse_probe_resources(resource_result.stdout)
    sass_result = BASE._run([
        str(args.cuobjdump.resolve()), "--dump-sass", str(binary)],
        cwd=REPOSITORY_ROOT, capture_output=True)
    sass_path = build / "add_rmsnorm_round_service_1024_h8192.sass"
    sass_path.write_text(sass_result.stdout, encoding="utf-8")
    sass_contract = _validate_probe_sass(sass_result.stdout, resources)
    return {
        "schema": "a800_add_rmsnorm_phase_chain_1024_h8192_clock64_static_v2",
        "baseline_commit": _git_identity(),
        "production_or_evaluation_duration_consumed": False,
        "source_contract": source_contract,
        "production_source_pin": source_pin,
        "upstream_reference": upstream_reference,
        "tool_identities": tools,
        "production_artifact": production,
        "compile_command": shlex.join(compile_command),
        "source_sha256": _sha256(snapshot / SOURCE_PATH.name),
        "runner_sha256": _sha256(snapshot / RUNNER_PATH.name),
        "report_sha256": _sha256(snapshot / REPORT_PATH.name),
        "source_pin_sha256": _sha256(snapshot / SOURCE_PIN_PATH.name),
        "upstream_reference_sha256": _sha256(
            snapshot / UPSTREAM_REFERENCE_PATH.name),
        "base_runner_sha256": _sha256(BASE_RUNNER_PATH),
        "probe_binary_sha256": _sha256(binary),
        "probe_resource_sha256": _sha256(resource_path),
        "probe_sass_sha256": _sha256(sass_path),
        "probe_resources": resources,
        "occupancy_proof": _occupancy_proof(resources),
        "probe_sass_contract": sass_contract,
    }, binary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", type=int)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--nvcc", type=Path, default=DEFAULT_NVCC)
    parser.add_argument("--cuobjdump", type=Path, default=DEFAULT_CUOBJDUMP)
    parser.add_argument("--nvidia-smi", type=Path, default=DEFAULT_NVIDIA_SMI)
    parser.add_argument(
        "--production-binary", type=Path, default=DEFAULT_PRODUCTION_BINARY)
    args = parser.parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", args.owner) is None:
        raise ValueError("owner contains unsupported characters")
    if args.device is not None and args.device < 0:
        raise ValueError("device must be non-negative")
    if not args.static_only and args.device is None:
        raise ValueError("device is required unless --static-only is used")
    output = _validate_output(args.run_id)
    output.mkdir(parents=True)
    static, binary = _static_contract(output, args)
    _write_json(output / "static_contract.json", static)
    if args.static_only:
        return

    nvidia_smi = args.nvidia_smi.resolve()
    nvidia_smi_identity = BASE._tool_identity(
        nvidia_smi,
        ("--query-gpu=driver_version", "--format=csv,noheader"))
    preflight = BASE._require_idle_device(args.device, nvidia_smi)
    run_command = [str(binary), str(args.device), "0.875"]
    started_at = _utc_now()
    with (output / "raw.csv").open("w", encoding="utf-8") as stdout, (
            output / "native.stderr").open("w", encoding="utf-8") as stderr:
        native_pid = BASE._run_native(
            run_command, cwd=REPOSITORY_ROOT, stdout=stdout, stderr=stderr)
    postflight = BASE._wait_for_idle_device(args.device, nvidia_smi)
    manifest = {
        "schema": "a800_add_rmsnorm_phase_chain_1024_h8192_clock64_capture_v2",
        "owner": args.owner,
        "runner_pid": os.getpid(),
        "native_pid": native_pid,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "native_command": shlex.join(run_command),
        "trials": EXPECTED_TRIALS,
        "samples_per_trial_cell": EXPECTED_SAMPLES,
        "runtime_variance_scale": 0.875,
        "timer_source": "clock64",
        "measurement_unit": "cycles",
        "device": preflight,
        "postflight_device": postflight,
        "preflight_cleanup": "target_device_idle_no_compute_processes",
        "postflight_cleanup": "target_device_idle_no_compute_processes",
        "nvidia_smi_identity": nvidia_smi_identity,
        "static_contract": static,
        "static_contract_sha256": _sha256(output / "static_contract.json"),
        "raw_sha256": _sha256(output / "raw.csv"),
        "production_or_evaluation_duration_consumed": False,
        "forward_eligible": False,
    }
    _write_json(output / "manifest.json", manifest)
    report_command = [
        sys.executable, str(REPORT_PATH),
        "--input", str(output / "raw.csv"),
        "--manifest", str(output / "manifest.json"),
        "--output", str(output / "report.json"),
    ]
    subprocess.run(
        report_command, cwd=REPOSITORY_ROOT, check=True, timeout=20)


if __name__ == "__main__":
    main()
