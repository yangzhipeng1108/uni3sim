"""
Capture nsys ground truth for GPU kernel profiling.

Runs vLLM on real GPU with nsys profiling (inference only, warmup excluded via
cudaProfilerApi), parses the sqlite output, and saves structured nsys data to JSON.

Usage:
    python scripts/evaluation/serving/capture_nsys.py \
        --hardware-id nvidia_a800 \
        --output-dir profiler_output/serving/nvidia_a800/e2e_comparison
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sqlite3
import sys
from pathlib import Path
from collections import defaultdict

from src.kernel.hardware_ids import canonical_hardware_id

_ROOT = Path(__file__).resolve().parents[2]

SEQ_LEN = 128
MAX_TOKENS = 8


def get_capture_output_dir(hardware_id):
    """Return the explicit local workspace for NVIDIA Serving captures."""
    canonical_id = canonical_hardware_id(hardware_id)
    if canonical_id != "nvidia_a800":
        raise NotImplementedError(
            f"Serving Nsight capture is not implemented for {canonical_id}")
    return _ROOT / "profiler_output" / "serving" / canonical_id


def require_capture_output_dir(output_dir, hardware_id):
    """Resolve one capture directory inside the canonical profiler workspace."""

    capture_root = get_capture_output_dir(hardware_id).resolve(strict=False)
    path = Path(output_dir)
    if not path.is_absolute():
        path = _ROOT / path
    resolved = path.resolve(strict=False)
    if resolved != capture_root and capture_root not in resolved.parents:
        raise ValueError(
            "Serving Nsight output must stay under "
            f"{capture_root}, got {resolved}"
        )
    return resolved


def capture_nsys_trace(
    output_dir,
    hardware_id,
    model,
    seq_len=SEQ_LEN,
    max_tokens=MAX_TOKENS,
    tp_size=1,
    pp_size=1,
    ep_size=1,
):
    """Run vLLM on real GPU(s), nsys captures ONLY inference (not warmup).

    Multi-GPU (tp_size/pp_size/ep_size > 1): vLLM spawns one worker per rank;
    nsys traces all ranks in a single sqlite. We partition kernels by deviceId
    so each rank's per-kernel trace can be compared against the analytic mirror,
    which models ONE rank's work (consistent with the per-PID emulation trace).

    Expert parallelism uses vLLM's enable_expert_parallel over the same world
    size as tensor parallelism (EP is layered onto the TP process group in vLLM;
    ep_size sets the EP degree and requires tp_size==ep_size for the dispatch).
    """
    output_dir = require_capture_output_dir(output_dir, hardware_id)
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty path or identifier")
    if min(seq_len, max_tokens, tp_size, pp_size, ep_size) <= 0:
        raise ValueError("sequence lengths and parallel sizes must be positive")
    if ep_size > 1 and ep_size != tp_size:
        raise ValueError("expert parallel size must be 1 or equal tensor parallel size")
    python_path = Path(sys.executable)
    if not python_path.is_file():
        raise FileNotFoundError(
            f"current Python executable not found: {python_path}")
    if "NSYS_BIN" in os.environ:
        configured_nsys = os.environ["NSYS_BIN"]
        if not configured_nsys:
            raise ValueError("NSYS_BIN must not be empty")
        resolved_nsys = shutil.which(configured_nsys)
    else:
        resolved_nsys = shutil.which("nsys")
    if resolved_nsys is None:
        raise FileNotFoundError(
            "nsys is not on PATH; set NSYS_BIN to an explicit executable")
    nsys_path = Path(resolved_nsys)

    output_dir.mkdir(parents=True, exist_ok=True)
    nsys_output = str(output_dir / "nsys_profile")
    for f in output_dir.glob("nsys_profile*"):
        f.unlink()

    world_size = tp_size * pp_size
    enable_ep = ep_size > 1
    script = f"""
import torch
from vllm import LLM, SamplingParams
import time

llm = LLM(model='{model}', dtype='float16', gpu_memory_utilization=0.85,
           enforce_eager=True, max_model_len=512, trust_remote_code=True,
           tensor_parallel_size={tp_size}, pipeline_parallel_size={pp_size},
           enable_expert_parallel={enable_ep})

# Warmup (OUTSIDE profiler capture)
_ = llm.generate(['test'], SamplingParams(temperature=0.0, max_tokens=1))
torch.cuda.synchronize()

# === START PROFILING ===
torch.cuda.cudart().cudaProfilerStart()
torch.cuda.synchronize()

prompt = ' '.join(['hello'] * {seq_len})
params = SamplingParams(temperature=0.0, max_tokens={max_tokens},
                        min_tokens={max_tokens}, ignore_eos=True)
start = time.perf_counter()
out = llm.generate([prompt], params)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

torch.cuda.cudart().cudaProfilerStop()
# === END PROFILING ===

print('WALL_MS:%.3f' % (elapsed * 1000))
print('TOKENS:%d' % len(out[0].outputs[0].token_ids))
"""

    cmd = [
        str(nsys_path), "profile",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--trace=cuda,nvtx",
        "--export=sqlite",
        "-f", "true",
        "-o", nsys_output,
        str(python_path), "-c", script,
    ]

    print(f"[nsys] profiling real GPU inference "
          f"(tp={tp_size} pp={pp_size} ep={ep_size}, world={world_size})...")
    env = os.environ.copy()
    # Respect an explicit device binding. Otherwise expose the first world-size
    # devices deterministically.
    if "CUDA_VISIBLE_DEVICES" in env:
        visible_devices = [
            value.strip()
            for value in env["CUDA_VISIBLE_DEVICES"].split(",")
            if value.strip()
        ]
        if len(visible_devices) < world_size:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES exposes fewer devices than the requested "
                f"world size: {visible_devices} vs {world_size}")
    else:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(index) for index in range(world_size))
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
        check=True,
    )

    wall_values = []
    for line in result.stdout.splitlines():
        if line.startswith("WALL_MS:"):
            try:
                value = float(line.partition(":")[2].strip())
            except ValueError as error:
                raise ValueError(
                    f"profiled process reported invalid WALL_MS: {line!r}") \
                    from error
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"profiled process WALL_MS must be positive: {value!r}")
            wall_values.append(value)
        if line.startswith("TOKENS:"):
            print(f"[nsys] Output tokens: {line.split(':')[1]}")
    if not wall_values:
        raise RuntimeError("profiled process did not report WALL_MS")
    if len(wall_values) != 1:
        raise RuntimeError(
            f"profiled process reported WALL_MS {len(wall_values)} times")
    wall_ms = wall_values[0]

    sqlite_path = output_dir / "nsys_profile.sqlite"
    if not sqlite_path.exists():
        nsys_rep = output_dir / "nsys_profile.nsys-rep"
        if nsys_rep.exists():
            subprocess.run(
                [str(nsys_path), "export", "--type=sqlite", "-o",
                 str(sqlite_path), str(nsys_rep)],
                capture_output=True,
                text=True,
                timeout=180,
                check=True,
            )

    if not sqlite_path.exists():
        raise RuntimeError(
            "nsys produced neither an SQLite export nor an exportable report; "
            f"stderr tail: {result.stderr[-500:]}")

    conn = sqlite3.connect(str(sqlite_path))
    rows = conn.execute("""
        SELECT s.value as name, k.start, k.end, k.end - k.start as dur,
               k.deviceId as dev
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    """).fetchall()
    conn.close()

    # Partition by device (rank). The analytic mirror models one rank, so for
    # multi-GPU runs we compare against a single rank (rank 0 = lowest deviceId).
    by_dev = defaultdict(list)
    for name, start, end, dur, dev in rows:
        by_dev[dev].append((name, start, end, dur))
    devs = sorted(by_dev)
    if not devs:
        raise RuntimeError("nsys SQLite contains no GPU kernel launches")
    print(f"[nsys] {len(rows)} kernels across {len(devs)} device(s): "
          f"{ {d: len(by_dev[d]) for d in devs} }")

    def _summarize(dev_rows):
        first_start = min(r[1] for r in dev_rows)
        last_end = max(r[2] for r in dev_rows)
        span_ns = last_end - first_start
        gpu_ns = sum(r[3] for r in dev_rows)
        kernels = defaultdict(lambda: {"count": 0, "ns": 0})
        for name, start, end, dur in dev_rows:
            kernels[name]["count"] += 1
            kernels[name]["ns"] += dur
        return span_ns, gpu_ns, dict(kernels)

    # Primary nsys_data.json = rank 0 (single rank), matching the analytic mirror.
    rank0_rows = by_dev[devs[0]]
    span_ns, gpu_ns, kern0 = _summarize(rank0_rows)
    print(f"[nsys] rank0: {len(rank0_rows)} kernels, "
          f"span={span_ns/1e6:.2f}ms, GPU_sum={gpu_ns/1e6:.2f}ms, "
          f"wall={wall_ms:.2f}ms")

    nsys_data = {
        "wall_ms": wall_ms,
        "span_ns": span_ns,
        "gpu_ns": gpu_ns,
        "n_kernels": len(rank0_rows),
        # Raw per-kernel facts are the capture contract. Evaluation owns any
        # later classification and must fail on unexplained names.
        "kernels": kern0,
        "model": model,
        "seq_len": seq_len,
        "max_tokens": max_tokens,
        "tp_size": tp_size,
        "pp_size": pp_size,
        "ep_size": ep_size,
        "world_size": world_size,
        "num_devices": len(devs),
    }

    nsys_json_path = output_dir / "nsys_data.json"
    with open(nsys_json_path, "w") as f:
        json.dump(nsys_data, f, indent=2)
    print(f"[nsys] Saved rank0 → {nsys_json_path}")

    # Also persist every rank's per-kernel counts for cross-rank inspection.
    if len(devs) > 1:
        per_rank = {}
        for i, d in enumerate(devs):
            sp, gp, kn = _summarize(by_dev[d])
            per_rank[f"rank{i}"] = {"deviceId": d, "n_kernels": len(by_dev[d]),
                                    "span_ns": sp, "gpu_ns": gp, "kernels": kn}
        ranks_path = output_dir / "nsys_per_rank.json"
        with open(ranks_path, "w") as f:
            json.dump(per_rank, f, indent=2)
        print(f"[nsys] Saved {len(devs)} ranks → {ranks_path}")

    return nsys_data


def main():
    parser = argparse.ArgumentParser(description="Capture nsys ground truth")
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--expert-parallel-size", type=int, default=1,
                        help="EP degree (>1 enables vLLM expert parallelism)")
    args = parser.parse_args()
    output_dir = (
        get_capture_output_dir(args.hardware_id)
        if args.output_dir is None
        else Path(args.output_dir)
    )
    capture_nsys_trace(
        output_dir, args.hardware_id, args.model,
        args.seq_len, args.max_tokens,
        tp_size=args.tensor_parallel_size,
        pp_size=args.pipeline_parallel_size,
        ep_size=args.expert_parallel_size)
