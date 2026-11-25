#!/usr/bin/env python3
"""Render-region cost versus idle gap between successive calls (host only).

The published render-region probe times back-to-back calls, so every call
finds the tokenizer and handler code hot in the core's caches and predictors.
Serving requests arrive tens of milliseconds apart.  This sweep renders fresh
held-out natural-language prompts (``natural_render_region_runner``) with a
fixed idle gap before each timed call, for several gaps and prompt lengths,
and reports the median wall and process CPU per (gap, length).  Gaps are an
experimental axis, not a deployment parameter; no serving or evaluation
duration is read.

Usage (inside the C500 vLLM container):
  python render_gap_sweep_runner.py TOKENIZER OUT.json
"""
import asyncio, json, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from natural_render_region_runner import _held_out_texts  # noqa: E402

GAPS_MS = (0, 1, 3, 10, 30, 100, 250)
LENGTHS = (16, 128, 512)
TRIALS = 40
WARMUP = 5


def main():
    model_path, out_path = sys.argv[1], sys.argv[2]
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    from transformers import AutoTokenizer
    from vllm.engine.arg_utils import EngineArgs
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender
    from vllm.renderers.registry import renderer_from_config

    tok = AutoTokenizer.from_pretrained(model_path)
    used = set()
    need = len(GAPS_MS) * (TRIALS + WARMUP)
    texts = {length: _held_out_texts(tok, length, need, used) for length in LENGTHS}
    config = EngineArgs(model=model_path, load_format="dummy", max_model_len=4096,
                        enforce_eager=True, gpu_memory_utilization=0.1).create_engine_config()
    registry = OpenAIModelRegistry(config.model_config, [BaseModelPath(name=model_path, model_path=model_path)])
    serving = OpenAIServingRender(config.model_config, renderer_from_config(config), registry,
                                  request_logger=None, chat_template=None, chat_template_content_format="auto")

    async def run():
        cells = []
        for length in LENGTHS:
            pool = iter(texts[length])
            for gap in GAPS_MS:
                walls, cpus = [], []
                for index in range(TRIALS + WARMUP):
                    if gap:
                        await asyncio.sleep(gap / 1000)
                    request = CompletionRequest(model=model_path, prompt=next(pool), max_tokens=16)
                    w0, c0 = time.perf_counter_ns(), time.process_time_ns()
                    await registry.check_model(request.model)
                    await serving.render_completion(request)
                    if index >= WARMUP:
                        walls.append(time.perf_counter_ns() - w0)
                        cpus.append(time.process_time_ns() - c0)
                cells.append({"gap_ms": gap, "prompt_tokens": length,
                              "median_wall_s": statistics.median(walls) / 1e9,
                              "median_process_cpu_s": statistics.median(cpus) / 1e9})
                print(gap, length, round(cells[-1]["median_wall_s"] * 1e6), round(cells[-1]["median_process_cpu_s"] * 1e6), flush=True)
        return cells

    cells = asyncio.run(run())
    with open(out_path, "w") as handle:
        json.dump({"schema_version": 1, "hardware_id": "metax_c500", "tokenizer_path": model_path,
                   "region": "check_model + OpenAIServingRender.render_completion",
                   "prompt_source": "held-out ShareGPT first user turns, fresh per call",
                   "production_or_evaluation_duration_consumed": False,
                   "trials": TRIALS, "warmup": WARMUP, "cells": cells}, handle, indent=1)


if __name__ == "__main__":
    main()
