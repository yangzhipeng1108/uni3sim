"""Independent finite BF16 panel pipeline, without routing or matrix outputs.

Each CTA loads synthetic indexed A rows and strided B panels, accumulates a
16x64 tile through a short K loop, and writes only one checksum. There is no
router, expert selection, activation, production GEMM/MoE output or workload
duration. Source configuration is fixed before collecting any measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import triton
import triton.language as tl


@triton.jit
def panel_pipeline(a_ptr, b_ptr, row_ptr, out_ptr, stamp_ptr, rounds, valid_rows,
                   LEADING: tl.constexpr, ROWS: tl.constexpr,
                   BM: tl.constexpr = 16, BN: tl.constexpr = 64, BK: tl.constexpr = 128):
    pid = tl.program_id(0)
    m = tl.arange(0, BM)
    n = tl.arange(0, BN)
    k = tl.arange(0, BK)
    selected = tl.load(row_ptr + pid * BM + m)
    accum = tl.full((BM, BN), 0, tl.float32)
    start = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l", args=[],
                                    dtype=tl.uint64, is_pure=False, pack=1)
    clock_start = tl.inline_asm_elementwise("mov.u64 $0, %clock64;", constraints="=l", args=[],
                                          dtype=tl.uint64, is_pure=False, pack=1)
    for step in range(rounds):
        offset = step * BK + k
        a = tl.load(a_ptr + selected[:, None] * LEADING + offset[None, :],
                    mask=m[:, None] < valid_rows, other=0)
        # Disjoint synthetic B matrices, stored [N, K]. This is a panel
        # mechanism, with no global-expert map or source GEMM output.
        b = tl.load(b_ptr + pid * BN * LEADING + n[None, :] * LEADING + offset[:, None])
        accum += tl.dot(a, b)
    checksum = tl.sum(tl.sum(accum, axis=0), axis=0)
    tl.store(out_ptr + pid, checksum)
    stop = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l", args=[],
                                   dtype=tl.uint64, is_pure=False, pack=1)
    clock_stop = tl.inline_asm_elementwise("mov.u64 $0, %clock64;", constraints="=l", args=[],
                                         dtype=tl.uint64, is_pure=False, pack=1)
    tl.store(stamp_ptr + pid * 4, start)
    tl.store(stamp_ptr + pid * 4 + 1, stop)
    tl.store(stamp_ptr + pid * 4 + 2, clock_start)
    tl.store(stamp_ptr + pid * 4 + 3, clock_stop)


@triton.jit
def queue_gate(duration_ns: tl.constexpr):
    begin = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l", args=[],
                                    dtype=tl.uint64, is_pure=False, pack=1)
    now = begin
    while now - begin < duration_ns:
        now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l", args=[],
                                       dtype=tl.uint64, is_pure=False, pack=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 3:
        raise ValueError("existing output or insufficient samples")
    if "H200" not in torch.cuda.get_device_name(0):
        raise ValueError("authorized H200 required")
    torch.manual_seed(20261107)
    leading, rows, maximum_grid = 4096, 4096, 6336
    a = torch.randn((rows, leading), dtype=torch.bfloat16, device="cuda") * 0.125
    b = torch.randn((maximum_grid, 64, leading), dtype=torch.bfloat16, device="cuda") * 0.125
    mapping = torch.randint(rows, (maximum_grid, 16), dtype=torch.int32, device="cuda")
    output = torch.zeros(maximum_grid, dtype=torch.float32, device="cuda")
    stamps = torch.zeros((maximum_grid, 4), dtype=torch.int64, device="cuda")
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    artifacts = args.output.parent / "compiler"
    artifacts.mkdir(exist_ok=False)
    # One code object: valid rows, loop rounds and grid are runtime values.
    compiled = panel_pipeline[(132,)](a, b, mapping, output, stamps, 4, 16,
        LEADING=leading, ROWS=rows, num_warps=4, num_stages=4)
    for name in ("ptx", "cubin"):
        data = compiled.asm[name]
        (artifacts / ("panel_pipeline." + name)).write_bytes(data if isinstance(data, bytes) else data.encode())
    metadata = {"num_warps": compiled.metadata.num_warps, "num_stages": compiled.metadata.num_stages,
                "shared_bytes": compiled.metadata.shared, "registers_per_thread": compiled.n_regs,
                "spill_count": compiled.n_spills}
    torch.cuda.synchronize()
    checks = []
    for valid in (1, 4, 16):
        panel_pipeline[(132,)](a, b, mapping, output, stamps, 4, valid,
            LEADING=leading, ROWS=rows, num_warps=4, num_stages=4)
        indexes = mapping[0, :valid].long()
        reference = (a[indexes, :512].float() @ b[0, :, :512].float().T).sum()
        error = float((output[0] - reference).abs())
        if error > max(0.03, abs(float(reference)) * 0.01):
            raise ValueError(f"panel checksum numerical error: {error}")
        checks.append({"valid_rows": valid, "checksum_absolute_error": error})
    results = []
    for grid in (132, 396, 1584, 6336):
        for valid in (1, 4, 16):
            for rounds in (4, 8, 16, 32):
                graph = torch.cuda.CUDAGraph()
                # Compile/warm before capture, outside all measured windows.
                panel_pipeline[(grid,)](a, b, mapping, output, stamps, rounds, valid,
                    LEADING=leading, ROWS=rows, num_warps=4, num_stages=4)
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    panel_pipeline[(grid,)](a, b, mapping, output, stamps, rounds, valid,
                        LEADING=leading, ROWS=rows, num_warps=4, num_stages=4)
                samples = []
                for sample in range(args.samples):
                    flush.fill_(sample)
                    queue_gate[(1,)](duration_ns=20_000_000, num_warps=1)
                    gate_done = torch.cuda.Event()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    gate_done.record()
                    start.record()
                    graph.replay()
                    end.record()
                    host_ahead = not gate_done.query()
                    end.synchronize()
                    if not host_ahead:
                        raise ValueError("host-ahead boundary not established")
                    timing = stamps[:grid].cpu().tolist()
                    samples.append({"event_ns": start.elapsed_time(end) * 1e6, "host_ahead": host_ahead,
                                    "cta_stamps_ns_and_cycles": timing})
                row = {"grid": grid, "valid_rows": valid, "rounds": rounds, "block_threads": 128,
                       "logical_b_read_bytes": grid * 64 * 128 * rounds * 2,
                       "logical_a_read_bytes": grid * valid * 128 * rounds * 2,
                       "samples": samples, "median_event_ns": statistics.median(s["event_ns"] for s in samples)}
                results.append(row)
                print(json.dumps({key: value for key, value in row.items() if key != "samples"}), flush=True)
                del graph
    payload = {"status": "complete_independent_panel_pipeline", "production_duration_consumed": False,
               "model_or_operator_executed": False, "gpu": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "triton": triton.__version__, "metadata": metadata,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "parameters": {"bm": 16, "bn": 64, "bk": 128, "leading": leading,
                              "synthetic_rows": rows, "cold_preamble_bytes": flush.numel()},
               "numerical_checks": checks, "rows": results,
               "limitations": ["independent panel checksum, not full GEMM, MoE, activation or routing",
                               "source addressing, output reduction and compiler resources differ from native expert kernels",
                               "event includes one graph entry/exit; CTA stamps exclude final stamp stores and global launch envelope",
                               "cold preamble declares state; cache hit rates and DRAM bytes are not observed",
                               "no empirical hardware rate or production price adopted by this collector"]}
    args.output.write_text(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
