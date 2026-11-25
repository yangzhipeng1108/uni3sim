#!/usr/bin/env python3
"""Rebuild nsys_kernels_full.json (per_rank, with ts_ns) from mctracer chrome
traces — for C500 multi-rank cases whose shipped nsys_kernels_full.json is a
flat list WITHOUT ts_ns (e.g. Qwen3-32B-TP2).

The chrome trace (mctracer_out/mctracer-<pid>.json) carries, per kernel exec
event (ph=X, cat=0):
  name, ts (ns), dur, args.grid{x,y,z}, args.block{x,y,z},
  args.mem.{static_shared, dynamic_shared, registers_per_thread},
  args.device_id.

The canonical native Chrome parser streams one JSON event at a time and emits a
per_rank dict: {per_rank: {"0": [{name,ts_ns,grid,block,smem,...}]}}
mirroring the Qwen3-4B/8B format, so _load_nsys_full_json takes the ts_ns
per_rank path and _partition can cut by ts (instead of the broken count-based
flat split). Although this compatibility converter emits only kernels, it
validates every typed device-memory event and fails on unknown native enums.

Usage:
    python scripts/evaluation/serving/mctracer_to_nsys_converter.py <case_dir>
"""
import argparse
import json
import re
import sys
import glob
import os
from pathlib import Path
from collections import defaultdict

from dataset_pipeline.conversion.mctracer_chrome import (
    McTracerKernelEvent,
    iter_mctracer_device_events,
)


def stream_kernels(path):
    """Yield typed kernel events from a native mcTracer Chrome trace."""
    for event in iter_mctracer_device_events(path):
        if isinstance(event, McTracerKernelEvent):
            yield event


def rebuild(case_dir):
    cd = Path(case_dir)
    mdir = cd / "mctracer_out"
    # Sort by size desc; multi-rank → one big file per rank. Identify rank by
    # the dominant device_id in each file.
    cands = sorted(glob.glob(str(mdir / "mctracer-*.json")),
                   key=os.path.getsize, reverse=True)
    big = [c for c in cands if os.path.getsize(c) > 50_000_000]  # >50MB = real trace
    if not big:
        raise FileNotFoundError(f"no large mctracer traces in {mdir}")
    per_rank = {}
    rank_files = {}
    for path in big:
        dev_counts = defaultdict(int)
        # quick scan of device_id distribution (sample first ~5M)
        with open(path, "rb", buffering=1 << 20) as f:
            sampled = 0
            for raw in f:
                for m in re.finditer(rb'"device_id":\s*(\d+)', raw):
                    d = int(m.group(1))
                    dev_counts[d] += 1
                    sampled += 1
                if sampled > 20000:
                    break
        if not dev_counts:
            raise ValueError(f"mctracer trace has no device_id samples: {path}")
        rank = max(dev_counts, key=dev_counts.__getitem__)
        if rank in rank_files:
            raise ValueError(
                f"multiple mctracer traces resolve to device/rank {rank}: "
                f"{rank_files[rank]} and {path}")
        rank_files[rank] = path

    if not rank_files:
        raise ValueError(f"no rank traces could be identified in {mdir}")

    print(f"  [rebuild] rank->file: " +
          ", ".join(f"rank{r}={os.path.basename(p)}" for r, p in sorted(rank_files.items())),
          file=sys.stderr)

    for rank, path in rank_files.items():
        kerns = []
        for event in stream_kernels(path):
            if event.device_id != rank:
                continue
            kerns.append({
                "name": event.name,
                "ts_ns": event.ts_ns,
                "dur_ns": event.dur_ns,
                "grid": list(event.grid),
                "block": list(event.block),
                "smem": (
                    event.static_shared_memory_bytes
                    + event.dynamic_shared_memory_bytes
                ),
                "static_smem": event.static_shared_memory_bytes,
                "dynamic_smem": event.dynamic_shared_memory_bytes,
                "registers_per_thread": event.registers_per_thread,
            })
        kerns.sort(key=lambda k: k["ts_ns"])
        if not kerns:
            raise ValueError(
                f"mctracer trace {path} has no kernel events for rank {rank}")
        per_rank[str(rank)] = {"kernels": kerns}
        print(f"  [rebuild] rank{rank}: {len(kerns)} kernels extracted",
              file=sys.stderr)

    out = {
        "kernels": [],  # legacy flat (empty) — consumers use per_rank
        "per_rank": {k: v["kernels"] for k, v in per_rank.items()},
    }
    out_path = cd / "nsys_kernels_full.json"
    # Back up the flat original once.
    bak = cd / "nsys_kernels_full.flat.json.bak"
    if out_path.exists() and not bak.exists():
        os.rename(str(out_path), str(bak))
        print(f"  [rebuild] backed up flat original -> {bak.name}", file=sys.stderr)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"  [rebuild] wrote {out_path} (per_rank, ts_ns) "
          f"ranks={list(per_rank.keys())}", file=sys.stderr)
    # Invalidate the cached sync_ends (rank0-only old logic) so it rebuilds.
    for r in per_rank:
        se = cd / f".mctracer_sync_ends_{r}.json"
        if se.exists():
            os.remove(str(se))
            print(f"  [rebuild] removed stale {se.name}", file=sys.stderr)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert mcTracer observations to the Nsight schema.")
    parser.add_argument("case_dir", nargs="+", help="Serving case directory")
    args = parser.parse_args(argv)
    for case_dir in args.case_dir:
        rebuild(case_dir)
    return 0
