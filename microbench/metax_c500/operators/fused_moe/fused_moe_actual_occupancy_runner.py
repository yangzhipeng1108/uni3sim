#!/usr/bin/env python3
"""Capture dry-launch resource facts for short C500 FusedMoE programs.

This probe compiles the frozen C500 dataset program for the short BM16 cases
which dominate the current C500 Max-APE cluster.  The installed serving vLLM
is deliberately not used: its current tuner selects a different
``BN64/BK128`` program rather than the canonical ``BN32/BK64`` program.
Instead, ``JITFunction.warmup`` compiles the frozen program without launching
it, and the loaded function handle is queried through the MACA occupancy API.
Consequently this writes no device-kernel duration and cannot be used as a
production duration measurement.

The W1 and W2 calls preserve the source roles: W1 has no routed-weight scale;
W2 has it.  The operand storage for B is a shape-only zero-stride view because
warmup compiles but never dereferences it.  B's runtime strides are scalar ABI
arguments, not Triton constexpr values, so this does not affect the compiled
function resources being observed.
"""

import argparse
import ctypes
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
QUERY_HELPER = Path(__file__).with_name("fused_moe_actual_occupancy_query.cpp")
EXPECTED_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
}


@dataclass(frozen=True)
class Case:
    name: str
    tokens: int
    experts: int
    topk: int
    hidden: int
    intermediate: int


CASES = (
    Case("tok1_e128_topk8_h2048_i768", 1, 128, 8, 2048, 768),
    Case("tok1_e256_topk8_h2048_i512", 1, 256, 8, 2048, 512),
    Case("tok1_e512_topk10_h2048_i512", 1, 512, 10, 2048, 512),
    Case("tok2_e256_topk8_h2048_i512", 2, 256, 8, 2048, 512),
)
CASE_BY_NAME = {case.name: case for case in CASES}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, environment=None, cwd=None):
    result = subprocess.run(
        command, check=False, capture_output=True, text=True,
        env=environment, cwd=cwd)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command!r}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def _parse_worker_output(stdout):
    payloads = [
        json.loads(line) for line in stdout.splitlines()
        if line.startswith("{")
    ]
    if len(payloads) != 1:
        raise ValueError(
            "worker output must contain exactly one JSON payload; "
            f"got {len(payloads)}")
    return payloads[0]


def _build(args, build_dir):
    compiler = shutil.which(args.compiler)
    if compiler is None:
        raise FileNotFoundError(f"compiler is unavailable: {args.compiler}")
    header = args.maca_path / "include/mcr/mc_runtime.h"
    if not header.is_file():
        raise FileNotFoundError(f"missing MACA runtime header: {header}")
    if not QUERY_HELPER.is_file():
        raise FileNotFoundError(f"missing occupancy query helper: {QUERY_HELPER}")
    output = build_dir / "libmidlayer_fused_moe_actual_occupancy_query.so"
    _run([
        compiler, "-std=c++14", "-O2", "-Wall", "-Wextra", "-Werror",
        "-fPIC", "-shared", "-isystem", str(args.maca_path / "include/mcr"),
        "-isystem", str(args.maca_path / "include"), str(QUERY_HELPER),
        "-o", str(output),
    ])
    return output


def _expected_grid(case, projection):
    if projection == "w1":
        columns = 2 * case.intermediate
    elif projection == "w2":
        columns = case.hidden
    else:
        raise ValueError(f"unknown projection: {projection!r}")
    return [case.tokens * case.topk * (columns // EXPECTED_CONFIG["BLOCK_SIZE_N"]),
            1, 1]


def _shape_only_weight(torch, experts, columns, rows):
    # Warmup never launches a kernel, so this view is never dereferenced.
    return torch.empty((1, 1, 1), device="cuda", dtype=torch.bfloat16).expand(
        experts, columns, rows)


def _worker(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(CASE_BY_NAME))
    args = parser.parse_args(argv)
    case = CASE_BY_NAME[args.case]
    import torch
    import triton
    import triton.language as tl
    from dataset_pipeline.kernel_dataset.metax_c500_fused_moe_kernels import (
        fused_moe_kernel,
        fused_moe_kernel_w2,
    )

    torch.cuda.set_device(0)
    padded_route_count = case.tokens * case.topk
    launch_route_count = padded_route_count * EXPECTED_CONFIG["BLOCK_SIZE_M"]
    sorted_ids = torch.empty(
        (launch_route_count,), dtype=torch.int32, device="cuda")
    expert_ids = torch.empty(
        (padded_route_count,), dtype=torch.int32, device="cuda")
    padded_routes = torch.empty(
        (1,), dtype=torch.int32, device="cuda")
    weights = torch.empty(
        (case.tokens, case.topk), dtype=torch.bfloat16, device="cuda")
    w1_input = torch.empty(
        (case.tokens, case.hidden), dtype=torch.bfloat16, device="cuda")
    w1_weight = _shape_only_weight(
        torch, case.experts, 2 * case.intermediate, case.hidden)
    w1_output = torch.empty(
        (case.tokens, case.topk, 2 * case.intermediate),
        dtype=torch.bfloat16, device="cuda")
    w2_input = torch.empty(
        (case.tokens * case.topk, case.intermediate),
        dtype=torch.bfloat16, device="cuda")
    w2_weight = _shape_only_weight(
        torch, case.experts, case.hidden, case.intermediate)
    w2_output = torch.empty(
        (case.tokens, case.topk, case.hidden),
        dtype=torch.bfloat16, device="cuda")

    query_helper_path = Path(
        os.environ["MIDLAYER_C500_FUSED_MOE_OCCUPANCY_QUERY"])
    if not query_helper_path.is_file():
        raise FileNotFoundError(
            f"occupancy query helper is missing: {query_helper_path}")
    query_library = ctypes.CDLL(str(query_helper_path))
    query = query_library.midlayer_c500_fused_moe_query_occupancy
    query.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int),
    ]
    query.restype = ctypes.c_int

    def compile_and_query(kernel, source, weight, output, topk_weights,
                          projection, source_topk, mul_routed_weight):
        grid = (
            triton.cdiv(launch_route_count, EXPECTED_CONFIG["BLOCK_SIZE_M"])
            * triton.cdiv(weight.size(1), EXPECTED_CONFIG["BLOCK_SIZE_N"]), )
        compiled = kernel.warmup(
            source, weight, output, None, None, None, topk_weights,
            sorted_ids, expert_ids, padded_routes,
            weight.size(1), weight.size(2), launch_route_count,
            source.size(0) * source_topk,
            source.stride(0), source.stride(1),
            weight.stride(0), weight.stride(2), weight.stride(1),
            output.stride(1), output.stride(2),
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            grid=grid,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=source_topk,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            per_channel_quant=False,
            HAS_BIAS=False,
            BLOCK_SIZE_K=EXPECTED_CONFIG["BLOCK_SIZE_K"],
            BLOCK_SIZE_M=EXPECTED_CONFIG["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=EXPECTED_CONFIG["BLOCK_SIZE_N"],
            GROUP_SIZE_M=EXPECTED_CONFIG["GROUP_SIZE_M"],
        )
        if compiled is None:
            raise RuntimeError("Triton warmup returned no compiled kernel")
        # This only loads the binary so its registered function can be queried;
        # JITFunction.run() is intentionally never called.
        compiled._init_handles()
        block_threads = compiled.metadata.num_warps * 64
        if block_threads != 256:
            raise RuntimeError(
                f"canonical short-BM16 block changed: {block_threads}")
        if isinstance(compiled.function, int):
            function = ctypes.c_void_p(compiled.function)
        elif isinstance(compiled.function, ctypes.c_void_p):
            function = compiled.function
        else:
            raise TypeError(
                "Triton compiled function handle has unsupported type: "
                f"{type(compiled.function)!r}")
        attribute_status = ctypes.c_int()
        occupancy_status = ctypes.c_int()
        registers = ctypes.c_int()
        static_shared = ctypes.c_size_t()
        resident = ctypes.c_int()
        query_status = query(
            function, block_threads, compiled.metadata.shared,
            ctypes.byref(attribute_status), ctypes.byref(occupancy_status),
            ctypes.byref(registers), ctypes.byref(static_shared),
            ctypes.byref(resident))
        if query_status != 0:
            raise RuntimeError(f"occupancy query helper failed: {query_status}")
        runtime_query_available = (
            attribute_status.value == 0 and occupancy_status.value == 0)
        if runtime_query_available:
            if registers.value <= 0 or resident.value <= 0:
                raise RuntimeError("runtime returned non-positive resource data")
            if registers.value != compiled.n_regs:
                raise RuntimeError(
                    f"runtime/compiler register disagreement: {registers.value} "
                    f"!= {compiled.n_regs}")
            if static_shared.value != compiled.metadata.shared:
                raise RuntimeError(
                    "runtime/compiler shared-memory disagreement: "
                    f"{static_shared.value} != {compiled.metadata.shared}")
        expected_grid = _expected_grid(case, projection)
        observed_grid = [grid[0], 1, 1]
        if observed_grid != expected_grid:
            raise RuntimeError(
                f"{projection} grid differs from source contract: "
                f"{observed_grid!r} != {expected_grid!r}")
        result = {
            "projection": projection,
            "mul_routed_weight": mul_routed_weight,
            "source_topk": source_topk,
            "grid": observed_grid,
            "block": [block_threads, 1, 1],
            "compiler_registers_per_thread": compiled.n_regs,
            "compiler_shared_memory_bytes": compiled.metadata.shared,
            "runtime_resource_query": {
                "available": runtime_query_available,
                "attribute_status": attribute_status.value,
                "occupancy_status": occupancy_status.value,
            },
        }
        if runtime_query_available:
            result["runtime_registers_per_thread"] = registers.value
            result["runtime_static_shared_memory_bytes"] = static_shared.value
            result["resident_ctas_per_ap"] = resident.value
        return result

    launches = [
        compile_and_query(
            fused_moe_kernel, w1_input, w1_weight, w1_output, None, "w1",
            case.topk, False),
        compile_and_query(
            fused_moe_kernel_w2, w2_input, w2_weight, w2_output, weights,
            "w2", 1, True),
    ]
    source = ROOT / "dataset_pipeline/kernel_dataset/metax_c500_fused_moe_kernels.py"
    if not source.is_file():
        raise FileNotFoundError(f"frozen canonical source is missing: {source}")
    print(json.dumps({
        "case": asdict(case),
        "source": {"path": str(source), "sha256": _sha256(source)},
        "torch": torch.__version__,
        "device_kernel_submitted": False,
        "admission_evidence_eligible": all(
            launch["runtime_resource_query"]["available"]
            for launch in launches),
        "launches": launches,
    }, sort_keys=True))
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--maca-path", type=Path, default=Path("/opt/maca"))
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args(argv)
    if args.device < 0:
        raise ValueError("device must be non-negative")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    build_dir = args.output_dir / "build"
    build_dir.mkdir()
    query_helper = _build(args, build_dir)
    base_environment = os.environ.copy()
    if "LD_PRELOAD" in base_environment and base_environment["LD_PRELOAD"]:
        raise ValueError("LD_PRELOAD must be empty for attributable warmup")
    base_environment["MACA_VISIBLE_DEVICES"] = str(args.device)
    base_environment["MIDLAYER_C500_FUSED_MOE_OCCUPANCY_QUERY"] = str(
        query_helper)
    reports = []
    for case in CASES:
        environment = base_environment.copy()
        result = _run([
            sys.executable, str(Path(__file__).resolve()), "_worker",
            "--case", case.name], environment=environment)
        (args.output_dir / f"{case.name}.stdout.txt").write_text(result.stdout)
        (args.output_dir / f"{case.name}.stderr.txt").write_text(result.stderr)
        worker = _parse_worker_output(result.stdout)
        reports.append(worker)
    report = {
        "schema_version": 1,
        "hardware_id": "metax_c500",
        "mechanism": "fused_moe_short_bm16_runtime_occupancy",
        "purpose": "static source-selected resource and CTA-admission evidence",
        "production_or_evaluation_duration_consumed": False,
        "device_kernel_submitted": False,
        "query_helper": {
            "path": str(QUERY_HELPER), "sha256": _sha256(QUERY_HELPER),
            "query_after_binary_load": True,
        },
        "runner": {"path": str(Path(__file__).resolve()),
                   "sha256": _sha256(Path(__file__))},
        "admission_evidence_eligible": all(
            report["admission_evidence_eligible"] for report in reports),
        "cases": reports,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(args.output_dir / "report.json"),
                      "case_count": len(reports)}, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        _worker(sys.argv[2:])
    else:
        main()
