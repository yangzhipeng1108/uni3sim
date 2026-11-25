#!/usr/bin/env python3
"""Plan and drive the WP1 H200 re-measurement of the A800 Triton MoE probes.

``plan`` (local, needs the canonical A800 kernel dataset for the exact
per-expert routing censuses) writes one JSON capture list that restates, per
A800 reference, the A800 probe cells: the same runner, routing (seed, exact
canonical ``tok_per_expert`` or M-tile census), K/N, source top-k, routed
weight flag, workspace depth and fresh-process count.  Only the hardware
coupling changes: A800 device 2 -> H200 GPU 0 and A800 device 6 -> H200 GPU
3, and the deep-queue gate 2*108*8 = 1728 M tiles becomes 2*132*8 = 2112
(the only affected probe is the E32 depth of the deep-H2880 reference, see
``DEEP_GATE_RECHOSEN``).  Where an A800 reference does not record a probe
argument, the value chosen here is listed in ``INFERRED`` and copied into the
published H200 reference.

``run`` (remote, stdlib only) executes the captures of one GPU sequentially,
one fresh process per capture, and repeats a process whose clock gate
rejected it (up to ``MAX_ATTEMPTS``); rejected attempts stay on disk.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


A800_TO_H200_DEVICE = {2: 0, 6: 3}
MAX_ATTEMPTS = 4
POLLUTER_BYTES = 128 * 1024 * 1024
DEEP_GATE_A800 = 2 * 108 * 8
DEEP_GATE_H200 = 2 * 132 * 8
TRI = "microbench/nvidia_h200/operators/fused_moe/fused_moe_triton_operand_stream_runner.py"
W2R = "microbench/nvidia_h200/operators/fused_moe/sm80_bf16_bm16_sparse_w2_source_runner.py"
W1R = "microbench/nvidia_h200/operators/fused_moe/sm80_bf16_bm16_sparse_w1_source_vllm011_runner.py"
CANONICAL_DATASET = "kernel_dataset/nvidia_a800/fused_moe.json.zst"
DEEP_GATE_RECHOSEN = {
    "bm64_triton_h2880_source_depth": (
        "A800 measured E32 at 32768 tokens (maximum 2080 M tiles), which "
        "passes the A800 gate 1728 but not the H200 gate 2112; H200 measures "
        "E32 at 33792 tokens (maximum ceil((33792*4+32*63)/64) = 2144 M "
        "tiles). The E128 depth 2174 already passes 2112 and is unchanged."),
}
INFERRED = {
    "bm64_triton_h2880_source_depth": (
        "A800 route seeds are not recorded; H200 uses seeds 2026092501.. "
        "(one per capture). W1 runs without and W2 with MUL_ROUTED_WEIGHT "
        "(production source roles); source_topk = topk = 4."),
    "bm64_triton_e256_h7168_mid_workspace_source_depth": (
        "The A800 routed-weight flags are not recorded; H200 follows the "
        "same-family E256 references captured before it (deep/exact source "
        "order): W1 with and W2 without MUL_ROUTED_WEIGHT, source_topk 8."),
    "bm64_e256_h7168_5_3_source_group": (
        "The A800 routed-weight flag is not recorded; H200 follows the "
        "E256 W1 family convention (MUL_ROUTED_WEIGHT)."),
    "bm64_e32_h2880_143_active_160_max_source_depth": (
        "The A800 seed is not recorded; seed 2026090017 (the seed of the "
        "same-period E128 574-active probe) reproduces 143 active M tiles."),
    "bm64_e512_1001_active_1144_max_source_depth": (
        "The A800 seed is not recorded; seed 2026090003 is the first seed "
        "from 2026090000 that reproduces 1001 active M tiles."),
    "bm64_triton_e256_h7168_2166_active_2300_w2_source_depth": (
        "The A800 reference records only the M-tile histogram {8:141, 9:112, "
        "10:3}; the canonical TOK16384 E256 tok_per_expert has exactly this "
        "histogram and is used as the routing."),
    "bm64_triton_e128_h2880_1084_active_1150_exact_source_order": (
        "The A800 fresh-process count per cell is not recorded; H200 uses 3."),
}


def _canonical_counts(root, key):
    import zstandard
    with (root / CANONICAL_DATASET).open("rb") as stream:
        document = json.loads(
            zstandard.ZstdDecompressor().stream_reader(stream).read())
    return document[key]["op_info"]["Attr"]["tok_per_expert"]


def _csv(values):
    return ",".join(str(value) for value in values)


def _tri(reference, name, device, *, tokens, experts, topk, k, n,
         source_topk=None, mul=False, seed=None, tokens_per_expert=None,
         m_tile_counts=None, maximum_m_tiles=None, trials=7, refresh=False,
         polluter=False, cell=None, process=1):
    arguments = [
        "--tokens", str(tokens), "--experts", str(experts),
        "--topk", str(topk), "--k", str(k), "--n", str(n),
        "--trials", str(trials)]
    if source_topk is not None:
        arguments += ["--source-topk", str(source_topk)]
    if mul:
        arguments.append("--mul-routed-weight")
    if seed is not None:
        arguments += ["--route-seed", str(seed)]
    if tokens_per_expert is not None:
        arguments += ["--tokens-per-expert", _csv(tokens_per_expert)]
    if m_tile_counts is not None:
        arguments += ["--m-tile-counts", _csv(m_tile_counts)]
    if maximum_m_tiles is not None:
        arguments += ["--maximum-m-tiles", str(maximum_m_tiles)]
    if refresh:
        arguments.append("--refresh-activation")
    if polluter:
        arguments += ["--polluter-bytes", str(POLLUTER_BYTES)]
    return {
        "reference": reference, "name": name, "runner": "tri",
        "gpu": A800_TO_H200_DEVICE[device], "a800_device": device,
        "cell": cell, "process": process, "arguments": arguments}


def _w2r(reference, name, device, *, tokens, experts, topk, active, k, n,
         role, source_topk=None, seed=None, expected_active=None,
         cell=None, process=1):
    arguments = [
        "--tokens", str(tokens), "--experts", str(experts),
        "--topk", str(topk), "--active-experts", str(active),
        "--k", str(k), "--n", str(n), "--role", role, "--trials", "9"]
    if source_topk is not None:
        arguments += ["--source-topk", str(source_topk)]
    if seed is not None:
        arguments += ["--route-seed", str(seed)]
    if expected_active is not None:
        arguments += ["--expected-active-m-tiles", str(expected_active)]
    return {
        "reference": reference, "name": name, "runner": "w2r",
        "gpu": A800_TO_H200_DEVICE[device], "a800_device": device,
        "cell": cell, "process": process, "arguments": arguments}


def build_plan(root):
    captures = []
    add = captures.append
    counts = {key: _canonical_counts(root, key) for key in (
        "FUSEDMOE_TOK32768_E256_TOPK8_H7168_INT2048",
        "FUSEDMOE_TOK16384_E256_TOPK8_H7168_INT2048",
        "FUSEDMOE_TOK8192_E256_TOPK8_H7168_INT2048",
        "FUSEDMOE_TOK32768_E128_TOPK8_H4096_INT1536",
        "FUSEDMOE_TOK32768_E128_TOPK16_H2048_INT768",
        "FUSEDMOE_TOK16384_E8_TOPK2_H4096_INT14336",
        "FUSEDMOE_TOK32768_E8_TOPK2_H4096_INT14336",
        "FUSEDMOE_TOK32768_E512_TOPK10_H4096_INT1024",
        "FUSEDMOE_TOK16384_E128_TOPK4_H2880_INT2880")}
    e256_32768 = counts["FUSEDMOE_TOK32768_E256_TOPK8_H7168_INT2048"]
    e256_16384 = counts["FUSEDMOE_TOK16384_E256_TOPK8_H7168_INT2048"]
    e256_8192 = counts["FUSEDMOE_TOK8192_E256_TOPK8_H7168_INT2048"]

    # ---- E256 H7168 family (DeepSeek-V3 shapes) ----
    reference = "bm64_triton_e256_h7168_deep_source_depth"
    for device in (2, 6):
        for role, k, widths, mul in (("w1", 7168, (4032, 4160), True),
                                     ("w2", 2048, (7104, 7232), False)):
            for n in widths:
                for process in (1, 2, 3):
                    add(_tri(reference, f"d{device}_{role}_n{n}_c{process}",
                             device, tokens=32768, experts=256, topk=8, k=k,
                             n=n, mul=mul, seed=35, maximum_m_tiles=4348,
                             cell=f"{role}_n{n}_d{device}", process=process))
    reference = "bm64_triton_e256_h7168_exact_source_order"
    for device in (2, 6):
        for role, k, widths, mul in (("w1", 7168, (4032, 4160), True),
                                     ("w2", 2048, (7104, 7232), False)):
            for n in widths:
                for process in (1, 2, 3):
                    add(_tri(reference, f"d{device}_{role}_n{n}_c{process}",
                             device, tokens=32768, experts=256, topk=8, k=k,
                             n=n, mul=mul, tokens_per_expert=e256_32768,
                             maximum_m_tiles=4348,
                             cell=f"{role}_n{n}_d{device}", process=process))
    reference = "bm64_triton_e256_h7168_exact_w2_router_weight_source_order"
    for n in (7104, 7232):
        for device in (2, 6):
            add(_tri(reference, f"n{n}_d{device}", device, tokens=32768,
                     experts=256, topk=8, source_topk=1, k=2048, n=n,
                     mul=True, tokens_per_expert=e256_32768,
                     maximum_m_tiles=4348, cell=f"n{n}_d{device}"))
    reference = "bm64_triton_e256_h7168_2166_active_2300_w2_source_depth"
    for device in (2, 6):
        for n in (7104, 7232):
            for process in (1, 2, 3):
                add(_tri(reference, f"d{device}_n{n}_p{process}", device,
                         tokens=16384, experts=256, topk=8, source_topk=1,
                         k=2048, n=n, mul=True, tokens_per_expert=e256_16384,
                         maximum_m_tiles=2300, refresh=True,
                         cell=f"n{n}_d{device}", process=process))
    reference = "bm64_triton_e256_h7168_2166_exact_w1_source_order"
    for device in (2, 6):
        for n in (4032, 4160):
            for process in (1, 2, 3):
                add(_tri(reference, f"d{device}_w1_n{n}_c{process}", device,
                         tokens=16384, experts=256, topk=8, k=7168, n=n,
                         mul=True, tokens_per_expert=e256_16384,
                         maximum_m_tiles=2300, cell=f"w1_n{n}_d{device}",
                         process=process))
    reference = "bm64_triton_e256_h7168_mid_workspace_source_depth"
    for role, k, widths, mul in (("w1", 7168, (4032, 4160), True),
                                 ("w2", 2048, (7104, 7232), False)):
        for n in widths:
            for device in (2, 6):
                add(_tri(reference, f"exact1155_{role}_n{n}_d{device}",
                         device, tokens=8192, experts=256, topk=8, k=k, n=n,
                         mul=mul, tokens_per_expert=e256_8192,
                         maximum_m_tiles=1276,
                         cell=f"{role}_n{n}_d{device}_exact"))
        add(_tri(reference, f"seed7_{role}_n{widths[0]}_d2", 2, tokens=8192,
                 experts=256, topk=8, k=k, n=widths[0], mul=mul, seed=7,
                 maximum_m_tiles=1276, cell=f"{role}_n{widths[0]}_d2_seed7"))
    reference = "bm64_e256_h7168_5_3_source_group"
    for n in (4032, 4160):
        for process in (1, 2, 3):
            add(_tri(reference, f"n{n}_r{process}", 2, tokens=8192,
                     experts=256, topk=8, source_topk=8, k=7168, n=n,
                     mul=True, m_tile_counts=[5, 3] * 128,
                     maximum_m_tiles=1276, cell=f"n{n}", process=process))

    # ---- E8 (Mixtral-class eight-expert) family ----
    reference = "bm64_triton_source_depth"
    for tokens in (2048, 4096):
        for k, n in ((6080, 32640), (6208, 32896)):
            for process in (1, 2, 3):
                add(_tri(reference, f"tok{tokens}_k{k}_n{n}_r{process}", 2,
                         tokens=tokens, experts=8, topk=2, k=k, n=n,
                         cell=f"tok{tokens}_k{k}_n{n}_r{process}",
                         process=process))
    reference = "bm64_triton_e8_h6144_8_active_9_max_source_depth"
    for device in (2, 6):
        for role, k, widths, mul in (("w1", 6144, (32704, 32832), False),
                                     ("w2", 16384, (6080, 6208), True)):
            for n in widths:
                for process in (1, 2, 3):
                    add(_tri(reference,
                             f"device{device}_{role}_n{n}_rep{process}",
                             device, tokens=16, experts=8, topk=2, k=k, n=n,
                             mul=mul, seed=0,
                             cell=f"{role}_n{n}_d{device}", process=process))
    reference = "bm64_triton_e8_h6144_deep_w1_source_depth"
    census = [128, 129, 127, 128, 127, 130, 129, 130]
    for device in (2, 6):
        for n in (32704, 32832):
            for process, label in ((1, "a"), (2, "b"), (3, "c")):
                add(_tri(reference, f"d{device}_n{n}_{label}", device,
                         tokens=32704, experts=8, topk=2, source_topk=2,
                         k=6144, n=n, mul=True, m_tile_counts=census,
                         maximum_m_tiles=1032, cell=f"n{n}_d{device}",
                         process=process))
    for reference, k, widths in (
            ("bm64_triton_e8_h6144_deep_w2_source_depth", 16384, (6080, 6208)),
            ("bm64_triton_e8_h4096_deep_w2_source_depth", 14336, (4032, 4160))):
        for tokens, seed, depth in ((16384, 2026091016, 520),
                                    (32768, 2026091032, 1032)):
            for n in widths:
                for device in (2, 6):
                    for process in (1, 2, 3):
                        add(_tri(reference,
                                 f"tok{tokens}_n{n}_d{device}_c{process}",
                                 device, tokens=tokens, experts=8, topk=2,
                                 k=k, n=n, mul=True, seed=seed,
                                 maximum_m_tiles=depth,
                                 cell=f"tok{tokens}_n{n}_d{device}",
                                 process=process))
    reference = "bm64_triton_e8_h6144_deep_w2_source_topk1_depth"
    for tokens, seed, depth in ((16384, 2026091016, 520),
                                (32768, 2026091032, 1032)):
        for n in (6080, 6208):
            for device in (2, 6):
                add(_tri(reference, f"tok{tokens}_n{n}_d{device}", device,
                         tokens=tokens, experts=8, topk=2, source_topk=1,
                         k=16384, n=n, mul=True, seed=seed,
                         maximum_m_tiles=depth,
                         cell=f"tok{tokens}_n{n}_d{device}"))
    reference = "bm64_triton_e8_h6144_260_active_264_max_w2_source_topk1"
    for n in (6080, 6208):
        for device in (2, 6):
            add(_tri(reference, f"tok8192_n{n}_d{device}", device,
                     tokens=8192, experts=8, topk=2, source_topk=1, k=16384,
                     n=n, m_tile_counts=[33, 32, 33, 32, 32, 33, 32, 33],
                     maximum_m_tiles=264, cell=f"n{n}_d{device}"))
    reference = "bm64_triton_e8_h4096_exact_w2_topk1_source_order"
    for tokens, depth in ((16384, 520), (32768, 1032)):
        exact = counts[f"FUSEDMOE_TOK{tokens}_E8_TOPK2_H4096_INT14336"]
        for n in (4032, 4160):
            for device in (2, 6):
                add(_tri(reference, f"tok{tokens}_n{n}_d{device}", device,
                         tokens=tokens, experts=8, topk=2, source_topk=1,
                         k=14336, n=n, mul=True, tokens_per_expert=exact,
                         maximum_m_tiles=depth,
                         cell=f"tok{tokens}_n{n}_d{device}"))

    # ---- deep H2880 and remaining E32/E128/E512 families ----
    reference = "bm64_triton_h2880_source_depth"
    seed = 2026092501
    h2880 = []
    for experts, tokens in ((128, 32768), (32, 33792)):
        cells = (
            (("w2", 89, 44, 2), ("w2", 89, 44, 6), ("w2", 89, 45, 2),
             ("w2", 91, 45, 6), ("w2", 91, 46, 6), ("w2", 91, 46, 2),
             ("w1", 89, 89, 2), ("w1", 89, 89, 6), ("w1", 91, 91, 6),
             ("w1", 91, 91, 2)) if experts == 128 else
            (("w2", 89, 44, 2), ("w2", 91, 46, 6), ("w1", 89, 89, 2),
             ("w1", 91, 91, 6)))
        for role, k_tiles, n_tiles, device in cells:
            h2880.append((experts, tokens, role, k_tiles, n_tiles, device,
                          seed))
            seed += 1
    for experts, tokens, role, k_tiles, n_tiles, device, seed in h2880:
        add(_tri(reference,
                 f"e{experts}_{role}_k{k_tiles}_n{n_tiles}_d{device}", device,
                 tokens=tokens, experts=experts, topk=4, k=k_tiles * 32,
                 n=n_tiles * 64, mul=role == "w2", seed=seed,
                 cell=f"e{experts}_{role}_k{k_tiles}_n{n_tiles}_d{device}"))
    reference = "bm64_triton_e128_h2880_1084_active_1150_exact_source_order"
    exact = counts["FUSEDMOE_TOK16384_E128_TOPK4_H2880_INT2880"]
    for device in (2, 6):
        for role, source_topk, k, widths, mul, refresh in (
                ("w1", 4, 2880, (5696, 5824), False, False),
                ("w2", 1, 2880, (2816, 2944), True, True)):
            for n in widths:
                for process in (1, 2, 3):
                    add(_tri(reference, f"d{device}_{role}_n{n}_c{process}",
                             device, tokens=16384, experts=128, topk=4,
                             source_topk=source_topk, k=k, n=n, mul=mul,
                             tokens_per_expert=exact, maximum_m_tiles=1150,
                             refresh=refresh, cell=f"{role}_n{n}_d{device}",
                             process=process))
    reference = "bm64_e32_h2880_32_active_36_max_source_depth"
    for device in (2, 6):
        for role, widths, mul in (("w1", (5696, 5824), True),
                                  ("w2", (2816, 2944), False)):
            for n in widths:
                for process in (1, 2, 3):
                    add(_tri(reference, f"d{device}_{role}_n{n}_c{process}",
                             device, tokens=64, experts=32, topk=4, k=2880,
                             n=n, mul=mul, seed=2026091001,
                             cell=f"{role}_n{n}_d{device}", process=process))
    reference = "bm64_e32_h2880_143_active_160_max_source_depth"
    for role, endpoints, mul in (("w1", ((89, 89), (91, 91)), False),
                                 ("w2", ((89, 44), (91, 46)), True)):
        for k_tiles, n_tiles in endpoints:
            for device in (2, 6):
                add(_tri(reference,
                         f"d{device}_{role}_k{k_tiles}_n{n_tiles}", device,
                         tokens=2048, experts=32, topk=4, k=k_tiles * 32,
                         n=n_tiles * 64, mul=mul, seed=2026090017, trials=9,
                         cell=f"{role}_k{k_tiles}_n{n_tiles}_d{device}"))
    reference = "bm64_e512_1001_active_1144_max_source_depth"
    for role, endpoints, mul in (("w1", ((127, 31), (129, 33)), False),
                                 ("w2", ((31, 63), (33, 65)), True)):
        for k_tiles, n_tiles in endpoints:
            for device in (2, 6):
                add(_tri(reference,
                         f"d{device}_{role}_k{k_tiles}_n{n_tiles}", device,
                         tokens=4096, experts=512, topk=10, k=k_tiles * 32,
                         n=n_tiles * 64, mul=mul, seed=2026090003, trials=9,
                         cell=f"{role}_k{k_tiles}_n{n_tiles}_d{device}"))
    reference = "bm64_e128_574_active_638_max_source_depth"
    for device in (2, 6):
        for role, source_topk, k, widths, mul, refresh in (
                ("w1", 16, 2048, (1472, 1600), False, False),
                ("w2", 1, 768, (1984, 2112), True, True)):
            for n in widths:
                for process in (1, 2, 3):
                    add(_tri(reference, f"d{device}_{role}_n{n}_c{process}",
                             device, tokens=2048, experts=128, topk=16,
                             source_topk=source_topk, k=k, n=n, mul=mul,
                             seed=2026090017, trials=9, refresh=refresh,
                             polluter=True, cell=f"{role}_n{n}_d{device}",
                             process=process))
    reference = "bm64_triton_e128_deep_w2_source_depth"
    for tokens, seed, depth in ((131072, 2026090017, 2174),
                                (262144, 2026090002, 4222)):
        for n in (4032, 4160):
            for device in (2, 6):
                add(_tri(reference, f"depth{depth}_n{n}_d{device}", device,
                         tokens=tokens, experts=128, topk=1, k=1536, n=n,
                         mul=True, seed=seed, cell=f"depth{depth}_n{n}_d{device}"))
    reference = "bm64_triton_e128_h4096_4156_active_4222_max_w1_source_topk8"
    exact = counts["FUSEDMOE_TOK32768_E128_TOPK8_H4096_INT1536"]
    for n in (3008, 3136):
        for device in (2, 6):
            add(_tri(reference, f"tok32768_n{n}_d{device}", device,
                     tokens=32768, experts=128, topk=8, source_topk=8,
                     k=4096, n=n, tokens_per_expert=exact,
                     maximum_m_tiles=4222, cell=f"n{n}_d{device}"))
    exact = counts["FUSEDMOE_TOK32768_E128_TOPK16_H2048_INT768"]
    for reference, source_topk, k, widths, mul in (
            ("bm64_e128_h2048_8250_active_8318_max_w1_source_depth", 16,
             2048, (1472, 1600), False),
            ("bm64_e128_h2048_8250_active_8318_max_w2_source_topk1", 1, 768,
             (1984, 2112), True)):
        for n in widths:
            for device in (2, 6):
                add(_tri(reference, f"tok32768_n{n}_d{device}", device,
                         tokens=32768, experts=128, topk=16,
                         source_topk=source_topk, k=k, n=n, mul=mul,
                         tokens_per_expert=exact, maximum_m_tiles=8318,
                         cell=f"n{n}_d{device}"))
    reference = "bm64_triton_e512_w2_source_depth"
    for tokens, seed, n, device in (
            (163840, 2026090867, 4032, 2), (163840, 2026090867, 4160, 6),
            (163840, 2026090867, 4032, 6), (163840, 2026090867, 4160, 2),
            (163840, 2026090968, 4032, 2), (163840, 2026090968, 4160, 6),
            (327680, 2026090804, 4032, 2), (327680, 2026090804, 4160, 6),
            (327680, 2026090827, 4032, 2), (327680, 2026090827, 4160, 6),
            (327680, 2026090801, 4032, 2), (327680, 2026090801, 4160, 6),
            (327680, 2026090802, 4032, 2), (327680, 2026090802, 4160, 6),
            (327680, 2026090803, 4032, 2), (327680, 2026090803, 4160, 6)):
        add(_tri(reference, f"tok{tokens}_seed{seed}_n{n}_d{device}", device,
                 tokens=tokens, experts=512, topk=1, k=1024, n=n, mul=True,
                 seed=seed, cell=f"tok{tokens}_seed{seed}_n{n}_d{device}"))
    reference = "bm64_e512_h4096_deep_w1_source_depth"
    census = [math.ceil(value / 64) for value in
              counts["FUSEDMOE_TOK32768_E512_TOPK10_H4096_INT1024"]]
    for n in (1984, 2112):
        for device in (2, 6):
            add(_tri(reference, f"n{n}_d{device}", device, tokens=32704,
                     experts=512, topk=10, source_topk=10, k=4096, n=n,
                     m_tile_counts=census, maximum_m_tiles=5624,
                     cell=f"n{n}_d{device}"))

    # ---- BM16 sparse W2/W1 source runner (W2R) ----
    for reference, cases in (
            ("bm16_e128_177_active_248_max_source_depth", (
                ("w1", dict(tokens=128, experts=128, topk=16, active=128,
                            seed=29, expected_active=177, k=2048),
                 (1504, 1568)),
                ("w2", dict(tokens=128, experts=128, topk=16, active=128,
                            seed=29, expected_active=177, k=768,
                            source_topk=1), (2016, 2080)))),
            ("bm16_e128_88_active_128_max_source_depth", (
                ("w1", dict(tokens=16, experts=128, topk=8, active=88,
                            k=4096), (3008, 3136)),
                ("w2", dict(tokens=16, experts=128, topk=8, active=88,
                            k=1536, source_topk=1), (4064, 4128)))),
            ("bm16_e8_7_active_8_max_source_depth", (
                ("w1", dict(tokens=4, experts=8, topk=2, active=7, k=4096),
                 (28160, 29184)),
                ("w2", dict(tokens=4, experts=8, topk=2, active=7, k=14336,
                            source_topk=1), (4064, 4128)))),
            ("bm16_e8_h6144_7_active_8_max_source_depth", (
                ("w1", dict(tokens=4, experts=8, topk=2, active=7, k=6144),
                 (32256, 33280)),
                ("w2", dict(tokens=4, experts=8, topk=2, active=7, k=16384,
                            source_topk=1), (6080, 6208)))),
            ("bm16_sparse_w2_source_depth", (
                ("e256_16_active_241_max", dict(
                    tokens=16, experts=256, topk=1, active=16, k=512),
                 (1984, 2112)),
                ("e512_20_active_482_max", dict(
                    tokens=20, experts=512, topk=1, active=20, k=512),
                 (1984, 2112))))):
        for service, case, widths in cases:
            role = "w1" if service == "w1" else "w2"
            for device in (2, 6):
                for n in widths:
                    for process in (1, 2, 3):
                        add(_w2r(reference,
                                 f"{service}_d{device}_n{n}_c{process}",
                                 device, n=n, role=role, **case,
                                 cell=f"{service}_n{n}_d{device}",
                                 process=process))

    # ---- BM16 hard-coded W1R (one process runs every A800 case) ----
    for device in (2, 6):
        add({"reference": "bm16_grouped_gemm_h2880_source_depth",
             "name": f"d{device}_r1", "runner": "w1r",
             "gpu": A800_TO_H200_DEVICE[device], "a800_device": device,
             "cell": f"d{device}", "process": 1,
             "arguments": ["--trials", "9"]})
    names = [(item["reference"], item["name"]) for item in captures]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate capture names")
    return captures


def _order(captures):
    """DeepSeek-relevant references first, as the WP1 Goal requires."""
    priority = [
        "bm64_triton_e256_h7168_deep_source_depth",
        "bm64_triton_e256_h7168_exact_source_order",
        "bm64_triton_e256_h7168_exact_w2_router_weight_source_order",
        "bm64_triton_e256_h7168_2166_active_2300_w2_source_depth",
        "bm64_triton_e256_h7168_2166_exact_w1_source_order",
        "bm64_triton_e256_h7168_mid_workspace_source_depth",
        "bm64_e256_h7168_5_3_source_group",
        "bm64_triton_source_depth",
        "bm64_triton_e8_h6144_8_active_9_max_source_depth",
        "bm64_triton_e8_h6144_deep_w1_source_depth",
        "bm64_triton_e8_h6144_deep_w2_source_depth",
        "bm64_triton_e8_h4096_deep_w2_source_depth",
        "bm64_triton_e8_h6144_deep_w2_source_topk1_depth",
        "bm64_triton_e8_h6144_260_active_264_max_w2_source_topk1",
        "bm64_triton_e8_h4096_exact_w2_topk1_source_order",
        "bm64_triton_h2880_source_depth",
    ]
    rank = {name: index for index, name in enumerate(priority)}
    return sorted(captures, key=lambda item: rank.get(
        item["reference"], len(priority)))


def _plan(arguments):
    root = Path(__file__).resolve().parents[4]
    captures = _order(build_plan(root))
    payload = {
        "schema": "h200_wp1_triton_moe_campaign_plan_v1",
        "a800_to_h200_device": A800_TO_H200_DEVICE,
        "deep_gate_m_tiles": {"a800": DEEP_GATE_A800, "h200": DEEP_GATE_H200},
        "deep_gate_rechosen": DEEP_GATE_RECHOSEN,
        "inferred": INFERRED,
        "captures": captures,
    }
    arguments.output.write_text(json.dumps(payload, indent=1) + "\n")
    by_gpu = {}
    for item in captures:
        by_gpu[item["gpu"]] = by_gpu.get(item["gpu"], 0) + 1
    print(f"{len(captures)} captures, by GPU {by_gpu}")


def _command(item, output, arguments):
    if item["runner"] == "tri":
        return [sys.executable, TRI, "--output-dir", str(output),
                "--device-id", str(item["gpu"]), *item["arguments"],
                "--vllm-011-source-dir", arguments.vllm_011_source_dir,
                "--baseline-commit", arguments.baseline_commit]
    if item["runner"] == "w2r":
        return [sys.executable, W2R, "--output", str(output / "result.json"),
                "--device-id", str(item["gpu"]), *item["arguments"],
                "--vllm-011-source-dir", arguments.vllm_011_source_dir,
                "--baseline-commit", arguments.baseline_commit]
    if item["runner"] == "w1r":
        return [sys.executable, W1R, *item["arguments"],
                "--vllm-011-source-dir", arguments.vllm_011_source_dir,
                "--clock-log", str(output / "clock.csv")]
    raise ValueError(f"unknown runner {item['runner']!r}")


def _accepted(item, output):
    if item["runner"] == "w1r":
        rows = [line.split(",") for line in
                (output / "raw.csv").read_text().splitlines()
                if line.startswith("clock,")
                and not line.startswith("clock,tokens,")]
        if not rows:
            raise RuntimeError(f"{output} has no clock verdicts")
        return all(row[-1] == "1" for row in rows)
    payload = json.loads((output / "result.json").read_text())
    return bool(payload["result"]["clock_gate"]["accepted"])


def _run(arguments):
    plan = json.loads(Path(arguments.plan).read_text())
    run_root = Path(arguments.run_root).resolve()
    status_path = run_root / f"status_gpu{arguments.gpu}.jsonl"
    selected = [item for item in plan["captures"]
                if item["gpu"] == arguments.gpu
                and (arguments.reference is None
                     or item["reference"] in arguments.reference)]
    environment = dict(os.environ)
    for item in selected:
        base = run_root / item["reference"] / item["name"]
        if (base / "DONE").exists():
            continue
        for attempt in range(1, MAX_ATTEMPTS + 1):
            output = base if attempt == 1 else base.with_name(
                f"{item['name']}_attempt{attempt}")
            if output.exists():
                if (output / "ACCEPTED").exists():
                    break
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            if item["runner"] == "w1r":
                output.mkdir()
            command = _command(item, output, arguments)
            started = time.time()
            log_path = output.parent / f"{output.name}.log"
            with log_path.open("w") as log:
                if item["runner"] == "w1r":
                    with (output / "raw.csv").open("w") as raw:
                        code = subprocess.run(
                            command, stdout=raw, stderr=log,
                            env=dict(environment,
                                     CUDA_VISIBLE_DEVICES=str(item["gpu"])),
                            cwd=arguments.repo).returncode
                else:
                    code = subprocess.run(
                        command, stdout=subprocess.DEVNULL, stderr=log,
                        env=environment, cwd=arguments.repo).returncode
            record = {"reference": item["reference"], "name": item["name"],
                      "output": str(output), "attempt": attempt,
                      "returncode": code,
                      "seconds": round(time.time() - started, 1)}
            if code != 0:
                record["accepted"] = False
                record["error"] = log_path.read_text()[-2000:]
                with status_path.open("a") as status:
                    status.write(json.dumps(record) + "\n")
                raise RuntimeError(f"capture failed: {record}")
            record["accepted"] = _accepted(item, output)
            with status_path.open("a") as status:
                status.write(json.dumps(record) + "\n")
            if record["accepted"]:
                (output / "ACCEPTED").write_text("")
                break
        else:
            raise RuntimeError(
                f"{item['reference']}/{item['name']} rejected "
                f"{MAX_ATTEMPTS} times by the clock gate")
        if not (base / "DONE").exists():
            base.mkdir(parents=True, exist_ok=True)
            (base / "DONE").write_text(str(output))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--run-root", required=True)
    run.add_argument("--repo", required=True)
    run.add_argument("--gpu", required=True, type=int, choices=(0, 3))
    run.add_argument("--vllm-011-source-dir", required=True)
    run.add_argument("--baseline-commit", required=True)
    run.add_argument("--reference", nargs="*")
    arguments = parser.parse_args()
    if arguments.action == "plan":
        _plan(arguments)
    else:
        _run(arguments)


if __name__ == "__main__":
    main()
