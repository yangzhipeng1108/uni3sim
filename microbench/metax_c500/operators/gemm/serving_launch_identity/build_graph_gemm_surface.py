#!/usr/bin/env python3
"""Build the C500 serving graph GEMM surface from launch-identity runs.

Input: ``table_<model>.json`` from ``launchid_table.py`` over the deployed
serving stack (container ``configured serving container``: vLLM 0.24.0,
vllm_metax 0.24.0, torch 2.10.0+metax3.8.2.2), one run per model:
qwen3-4b / qwen3-8b / llama-3.1-8b (TP1) and qwen3-32b (TP2, both ranks).
Keys are (n, k, m) with m the padded bucket for FULL and the exact scheduled
token count for PIECEWISE; the runs show no FULL/PIECEWISE disagreement at
equal m.  Rules (fail-closed):
- a key needs one program with >= 80% of its observations (the profiler drops
  a few records, which can shift a GEMM's neighbour-based classification);
- the entry must satisfy the loader's own program contract (name/tile/stage,
  split class vs split factor, thread block class); otherwise it is dropped;
- nothing from earlier nsys/mcTracer extractions is carried over: those came
  from a different software stack.
Launch identity only; no duration is read.

Usage: build_graph_gemm_surface.py TABLE_DIR OUT.json
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.insert(0, ROOT)
from src.kernel.launch_config.metax_c500.gemm_launch import (  # noqa: E402
    _mcblas_main_block_threads,
)

SIG_RE = re.compile(
    r"^(\d+)x(\d+)x(\d+)_(\w+?)_(\d+)t(_4stage)?_blk(\d+)_s(\d+)_sb(\d+)$")
MODELS = ("qwen3-4b", "qwen3-8b", "llama-3.1-8b", "qwen3-32b")
MAJORITY = 0.8


def main():
    table_dir, out_path = sys.argv[1], sys.argv[2]
    cells = defaultdict(dict)
    evidence = defaultdict(lambda: {"models": set(), "dropped": Counter()})
    for model in MODELS:
        document = json.load(open(os.path.join(table_dir, f"table_{model}.json")))
        if document["full_vs_piecewise_disagreement"]:
            raise ValueError(f"{model}: FULL/PIECEWISE disagree at equal m")
        merged = defaultdict(Counter)
        for key, sigs in document["table_by_mode"].items():
            _, nkm = key.split("|")
            merged[tuple(int(x) for x in nkm.split(","))].update(sigs)
        for (n, k, m), sigs in merged.items():
            evidence[(n, k)]["models"].add(model)
            sig, count = sigs.most_common(1)[0]
            if count / sum(sigs.values()) < MAJORITY:
                evidence[(n, k)]["dropped"]["no_majority"] += 1
                continue
            parsed = SIG_RE.match(sig)
            g2, g1, g3 = (int(parsed.group(i)) for i in (1, 2, 3))
            threads, block = int(parsed.group(5)), int(parsed.group(7))
            split, sb = int(parsed.group(8)), int(parsed.group(9))
            tile = [g2, g1, g3]
            try:
                tile_block = _mcblas_main_block_threads(*tile)
            except ValueError:
                tile_block = None  # tile class outside the loader's contract
            if (threads != block or (split == 1) != (sb == 0)
                    or tile_block != block):
                evidence[(n, k)]["dropped"]["program_contract"] += 1
                continue
            entry = {
                "m": m, "tile": tile, "block": block, "split_k": split,
                "stage4": bool(parsed.group(6)),
                "program": (f"mcblas__Mck_bf16gemm_tn_{g1}x{g2}x{g3}_{parsed.group(4)}_"
                            f"{threads}t{parsed.group(6) or ''}_bf16_bf16_fp32_sb_0_{sb}"),
            }
            if m in cells[(n, k)] and cells[(n, k)][m] != entry:
                raise ValueError(f"({n},{k},{m}) differs across models")
            cells[(n, k)][m] = entry
    document = {
        "schema": "mcblas_serving_graph_gemm_surface_v1",
        "hardware_id": "metax_c500",
        "software_revision": "vllm_metax_0.24.0_maca_3.8.2.2",
        "duration_consumed": False,
        "provenance": {
            "purpose": ("vLLM-metax 0.24.0 serving deployment mcBLAS program selection "
                        "for graph-replay (FULL/PIECEWISE) token GEMMs; PIECEWISE keyed "
                        "by exact scheduled tokens, FULL by padded capture bucket"),
            "sources": [
                "microbench/metax_c500/operators/gemm/serving_launch_identity "
                "launch-identity runs on the deployed serving container "
                "(kernel name/grid/block per forward; T=1..512 PIECEWISE and every "
                "decode capture bucket FULL; qwen3-4b/qwen3-8b/llama-3.1-8b TP1, "
                "qwen3-32b TP2)"],
            "builder": ("microbench/metax_c500/operators/gemm/serving_launch_identity/"
                        "build_graph_gemm_surface.py"),
            "known_gaps": ("eager NONE steps (m > capture limit), GEMV (m = 1) and "
                           "qwen3-30b-a3b are not overlaid; keys without a program "
                           "majority or violating the program contract keep the "
                           "canonical CSV selection"),
        },
        "cells": [
            {"n": n, "k": k,
             "entries": [cells[(n, k)][m] for m in sorted(cells[(n, k)])],
             "evidence": {"models": sorted(evidence[(n, k)]["models"]),
                          "dropped": dict(evidence[(n, k)]["dropped"]),
                          "entry_count": len(cells[(n, k)])}}
            for (n, k) in sorted(cells)
        ],
    }
    with open(out_path, "w") as handle:
        json.dump(document, handle, indent=1)
        handle.write("\n")
    for cell in document["cells"]:
        print(cell["n"], cell["k"], cell["evidence"])


if __name__ == "__main__":
    main()
