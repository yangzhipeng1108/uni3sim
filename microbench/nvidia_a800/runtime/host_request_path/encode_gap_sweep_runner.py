#!/usr/bin/env python3
"""Tokenizer encode cost versus idle gap between successive calls (host only).

The published encode probe (``measure_tokenizer_encode`` in
``vllm_host_request_path_runner.py``) times back-to-back calls on one repeated
word, so every call finds the tokenizer's code and tables hot in the core's
caches and predictors and its word cache pre-warmed.  In vLLM 0.11.0 each
serving request is encoded once, by ``AsyncMicrobatchTokenizer``'s executor
worker calling the same ``tokenizer(prompt)``, and requests reach it tens of
milliseconds apart.  This sweep encodes fresh held-out natural-language
prompts (first user turns from the tail of ShareGPT, disjoint from the
head-sampled serving workloads) with a fixed idle gap before each timed call,
for several gaps and prompt lengths, and reports the median wall and process
CPU per (gap, length).  The repeated-word prompt of the published probe is
re-run back to back as a control.  Gaps are an experimental axis, not a
deployment parameter; no serving or evaluation duration is read.

Usage (A800 host, no device):
  python encode_gap_sweep_runner.py TOKENIZER_PATH OUT.json [GAPS_MS]

``GAPS_MS`` is an optional comma-separated gap list (default: the full sweep);
``0`` alone reproduces the published back-to-back regime on natural prompts.
Each cell records the encoded token count of the prompts it timed.
"""
import json
import os
import statistics
import sys
import time

GAPS_MS = (0, 1, 3, 10, 30, 100, 250)
LENGTHS = (16, 128, 512)
TRIALS = 40
WARMUP = 5
if "MIDLAYER_SHAREGPT_DATASET" not in os.environ:
    raise SystemExit("MIDLAYER_SHAREGPT_DATASET must name the ShareGPT JSON input")
DATASET = os.environ["MIDLAYER_SHAREGPT_DATASET"]
HELD_OUT_FROM = 60000


def _held_out_texts(tokenizer, length, count, used):
    data = json.load(open(DATASET))
    texts = []
    for index in range(len(data) - 1, HELD_OUT_FROM, -1):
        turns = data[index]["conversations"]
        if not turns or turns[0]["from"] != "human":
            continue
        ids = tokenizer.encode(turns[0]["value"], add_special_tokens=False)
        if len(ids) < length or index in used:
            continue
        used.add(index)
        texts.append(tokenizer.decode(ids[:length]))
        if len(texts) == count:
            return texts
    raise RuntimeError(f"not enough held-out prompts for length {length}")


def _timed(operation):
    wall0, cpu0 = time.perf_counter_ns(), time.process_time_ns()
    operation()
    return time.perf_counter_ns() - wall0, time.process_time_ns() - cpu0


def main():
    model_path, out_path = sys.argv[1], sys.argv[2]
    gaps = (tuple(int(value) for value in sys.argv[3].split(","))
            if len(sys.argv) > 3 else GAPS_MS)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    used = set()
    need = len(gaps) * (TRIALS + WARMUP)
    texts = {length: _held_out_texts(tokenizer, length, need, used)
             for length in LENGTHS}
    control = []
    for length in LENGTHS:
        text = " ".join(["token"] * length)
        walls = []
        for index in range(TRIALS + WARMUP):
            wall, _ = _timed(lambda: tokenizer(text))
            if index >= WARMUP:
                walls.append(wall)
        control.append({"prompt_tokens": len(tokenizer(text)["input_ids"]),
                        "median_wall_s": statistics.median(walls) / 1e9})
        print("control", control[-1], flush=True)
    cells = []
    for length in LENGTHS:
        pool = iter(texts[length])
        for gap in gaps:
            walls, cpus, counts = [], [], []
            for index in range(TRIALS + WARMUP):
                if gap:
                    time.sleep(gap / 1000)
                text = next(pool)
                wall, cpu = _timed(lambda: tokenizer(text))
                if index >= WARMUP:
                    walls.append(wall)
                    cpus.append(cpu)
                    counts.append(len(tokenizer(text)["input_ids"]))
            cells.append({"gap_ms": gap, "prompt_tokens": length,
                          "median_encoded_tokens": statistics.median(counts),
                          "median_wall_s": statistics.median(walls) / 1e9,
                          "median_process_cpu_s": statistics.median(cpus) / 1e9})
            print(gap, length, round(cells[-1]["median_wall_s"] * 1e6, 1),
                  round(cells[-1]["median_process_cpu_s"] * 1e6, 1), flush=True)
    with open(out_path, "w") as handle:
        json.dump({"schema_version": 1, "hardware_id": "nvidia_a800",
                   "tokenizer_path": model_path,
                   "region": "tokenizer(prompt) as called by "
                             "AsyncMicrobatchTokenizer (vLLM 0.11.0)",
                   "prompt_source": "held-out ShareGPT first user turns, "
                                    "fresh per call",
                   "production_or_evaluation_duration_consumed": False,
                   "trials": TRIALS, "warmup": WARMUP,
                   "back_to_back_repeated_word_control": control,
                   "cells": cells}, handle, indent=1)


if __name__ == "__main__":
    main()
