#!/usr/bin/env python3
"""Per-(n, k, m) mcBLAS program table from one launch-identity run.

Reads ``trace.<pid>.json`` (kineto chrome trace) and ``meta.<pid>.json``
written by ``launchid_patch.py``; splits kernels per forward on the marker
kernels; maps each forward's decoder GEMMs to (n, k) cells by in-layer
position (qkv, o, gate_up, down), validates n against the launch grid,
excludes lm_head by its vocabulary extent and skips forwards that use the
GEMV path.  The profiler drops a few activity records under load, so each
GEMM is classified on its own: n from its launch grid, and the cells that
share n told apart by the kernel launched just before it (silu/act -> down,
attention region -> o_proj, norm -> qkv or gate_up).  The key m is the
padded bucket for FULL and the exact scheduled token count for PIECEWISE
(NONE is not graph-replayed and is skipped).
Launch identity only: durations in the trace are never read.

Usage: launchid_table.py RUN_DIR MODEL OUT.json
"""
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

GEMM_RE = re.compile(
    r"mcblas__Mck_(?:bf16|h)gemm_tn_(\d+)x(\d+)x(\d+)_(\w+?)_(\d+)t(_\d+stage)?"
    r"_(?:bf16|fp16)_(?:bf16|fp16)_fp32_sb_0_(\d+)")
GEMV_RE = re.compile(r"gemvt")
MARKER = "spin_kernel"
# in-layer positions: 0 qkv, 1 o_proj, 2 gate_up, 3 down
ROLE_POSITIONS = {"act": {3}, "attention": {1}, "norm": {0, 2}}


def _role(previous):
    lowered = previous.lower()
    if "silu" in lowered or "act_and_mul" in lowered:
        return "act"
    if "norm" in lowered:
        return "norm"
    return "attention"

MODELS = {
    "qwen3-4b": {"layers": 36, "vocab": 151936,
                 "cells": [(6144, 2560), (2560, 4096), (19456, 2560), (2560, 9728)]},
    "qwen3-8b": {"layers": 36, "vocab": 151936,
                 "cells": [(6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288)]},
    "llama-3.1-8b": {"layers": 32, "vocab": 128256,
                     "cells": [(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336)]},
    # per tensor-parallel rank (TP2)
    "qwen3-32b": {"layers": 64, "vocab": 151936 // 2,
                  "cells": [(5120, 5120), (5120, 4096), (25600, 5120), (5120, 12800)]},
}


def main():
    run_dir, model, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = MODELS[model]
    cells, layers, vocab_n = cfg["cells"], cfg["layers"], cfg["vocab"]
    tables = {}
    stats = Counter()
    for trace_path in sorted(glob.glob(os.path.join(run_dir, "trace.*.json"))):
        pid = trace_path.rsplit(".", 2)[1]
        meta = json.load(open(os.path.join(run_dir, f"meta.{pid}.json")))
        events = [e for e in json.load(open(trace_path))["traceEvents"]
                  if e.get("cat") == "kernel"]
        events.sort(key=lambda e: e["ts"])
        forwards, current, inside = [], None, False
        for event in events:
            if MARKER in event["name"]:
                if not inside:
                    current, inside = [], True
                else:
                    forwards.append(current)
                    inside = False
                continue
            if inside:
                current.append(event)
        if len(forwards) != len(meta):
            raise RuntimeError(f"{trace_path}: {len(forwards)} forwards vs {len(meta)} meta")
        table = defaultdict(Counter)
        for info, kernels in zip(meta, forwards):
            if info["mode"] == "NONE":
                stats["skip_none"] += 1
                continue
            if any(GEMV_RE.search(k["name"]) for k in kernels):
                stats["skip_gemv"] += 1
                continue
            m = info["padded_tokens"] if info["mode"] == "FULL" else info["scheduled_tokens"]
            previous = ""
            for kernel in kernels:
                match = GEMM_RE.search(kernel["name"])
                if not match:
                    if "DevDstElementwiseAdd" not in kernel["name"]:
                        previous = kernel["name"]
                    continue
                gx, gy, gz = kernel["args"]["grid"]
                g1, g2, g3 = (int(match.group(i)) for i in (1, 2, 3))
                if any(axis * g1 >= vocab_n for axis in (gx, gy)):
                    continue  # lm_head
                role = _role(previous)
                candidates = [
                    (position, n, k) for position, (n, k) in enumerate(cells)
                    if any(e in (n, 2 * n) or e * 2 == n or (e >= n and e - g1 < n)
                           for e in (gx * g1, gy * g1))
                    and ROLE_POSITIONS[role] & {position}]
                if len(candidates) != 1:
                    stats["unclassified_gemm"] += 1
                    continue
                _, n, k = candidates[0]
                sig = (f"{g2}x{g1}x{g3}_{match.group(4)}_{match.group(5)}t"
                       f"{'_4stage' if match.group(6) else ''}_blk{kernel['args']['block'][0]}"
                       f"_s{gz if gz > 1 else 1}_sb{match.group(7)}")
                table[(info["mode"], n, k, m)][sig] += 1
                stats["classified_gemm"] += 1
            stats["kept_forwards"] += 1
        tables[pid] = table
    # Merge ranks; record per-mode conflicts and FULL/PIECEWISE disagreement at equal m.
    merged = defaultdict(Counter)
    for table in tables.values():
        for key, sigs in table.items():
            merged[key].update(sigs)
    conflicts = {str(k): dict(v) for k, v in merged.items() if len(v) > 1}
    by_m = defaultdict(dict)
    for (mode, n, k, m), sigs in merged.items():
        by_m[(n, k, m)][mode] = sigs.most_common(1)[0][0]
    mode_disagree = {f"{n},{k},{m}": v for (n, k, m), v in by_m.items()
                     if len(set(v.values())) > 1}
    flat = {f"{n},{k},{m}": dict(merged[(mode, n, k, m)])
            for (mode, n, k, m) in sorted(merged)}
    json.dump({"model": model, "run_dir": run_dir, "stats": dict(stats),
               "ranks": sorted(tables), "conflicts": conflicts,
               "full_vs_piecewise_disagreement": mode_disagree,
               "table_by_mode": {f"{mode}|{n},{k},{m}": dict(v)
                                 for (mode, n, k, m), v in sorted(merged.items())},
               "table": flat}, open(out_path, "w"), indent=1)
    print(model, dict(stats), "keys", len(merged), "conflicts", len(conflicts),
          "full_vs_piecewise_disagree", len(mode_disagree))


if __name__ == "__main__":
    main()
