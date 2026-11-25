#!/usr/bin/env python3
"""Completion render region on fresh natural-language prompts (host only).

Same region as ``measure_completion_render_region`` in
``vllm_host_request_path_runner.py`` (``check_model`` +
``OpenAIServingRender.render_completion``), but every timed call renders a
*different* natural-language prompt, so the tokenizer's word cache is not
pre-warmed by repeating one text, and the text is not a single repeated word.
Prompts are the first user turns of ShareGPT conversations taken from the
tail of the file (held out from the head-sampled serving workloads), truncated
to the target token length with the served model's own tokenizer.  Only the
prompt text is used; no serving or evaluation duration is read.  The repeated
``"token"`` prompt of the published probe is run alongside as a control.

Usage (inside the C500 vLLM container, host only):
  python natural_render_region_runner.py TOKENIZER_PATH OUT.json
"""

import asyncio
import json
import os
import statistics
import sys
import time

LENGTHS = (16, 128, 512, 1024)
TRIALS = 60
WARMUP = 10
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

    probe_tokenizer = AutoTokenizer.from_pretrained(model_path)
    used = set()
    natural = {length: _held_out_texts(probe_tokenizer, length, TRIALS + WARMUP, used)
               for length in LENGTHS}
    config = EngineArgs(
        model=model_path, load_format="dummy", max_model_len=4096,
        enforce_eager=True, gpu_memory_utilization=0.1).create_engine_config()
    registry = OpenAIModelRegistry(
        config.model_config, [BaseModelPath(name=model_path, model_path=model_path)])
    serving = OpenAIServingRender(
        config.model_config, renderer_from_config(config), registry,
        request_logger=None, chat_template=None,
        chat_template_content_format="auto")

    async def region(text):
        request = CompletionRequest(model=model_path, prompt=text, max_tokens=16)
        started = time.perf_counter_ns()
        await registry.check_model(request.model)
        await serving.render_completion(request)
        return time.perf_counter_ns() - started

    async def run():
        result = {}
        for length in LENGTHS:
            samples = [await region(text) for text in natural[length]][WARMUP:]
            repeated = " ".join(["token"] * length)
            control = [await region(repeated) for _ in range(TRIALS + WARMUP)][WARMUP:]
            result[length] = {
                "natural_fresh_median_s": statistics.median(samples) / 1e9,
                "repeated_token_median_s": statistics.median(control) / 1e9,
            }
        return result

    result = asyncio.run(run())

    def fit(key):
        (l0, c0), (l1, c1) = ((LENGTHS[0], result[LENGTHS[0]][key]),
                              (LENGTHS[-1], result[LENGTHS[-1]][key]))
        slope = (c1 - c0) / (l1 - l0)
        return {"intercept_s": c0 - slope * l0, "per_token_s": slope}

    payload = {
        "schema_version": 1,
        "hardware_id": "metax_c500",
        "tokenizer_path": model_path,
        "region": "check_model + OpenAIServingRender.render_completion",
        "prompt_source": {"dataset": DATASET, "held_out_index_from_tail_above": HELD_OUT_FROM,
                          "fresh_prompt_per_call": True},
        "trials": TRIALS, "warmup": WARMUP,
        "production_or_evaluation_duration_consumed": False,
        "by_length": {str(k): v for k, v in result.items()},
        "natural_fresh_fit": fit("natural_fresh_median_s"),
        "repeated_token_fit": fit("repeated_token_median_s"),
    }
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=1)
    print("RESULT", json.dumps({k: payload[k] for k in ("by_length", "natural_fresh_fit", "repeated_token_fit")}))


if __name__ == "__main__":
    main()
