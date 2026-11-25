#!/usr/bin/env python3
"""Drive a vLLM 0.24 serving engine through every graph token domain.

Launch-identity collection on the deployed C500 serving stack (container
``configured serving container``).  The recorder (``launchid_patch.py``, loaded via
``sitecustomize.py`` on PYTHONPATH in every process) keeps kernel name/grid/
block per forward; this driver only chooses which token domains run:

- PIECEWISE: one fresh random-token prompt of T tokens, T = 1..512
  (the capture limit), so the prefill forward schedules exactly T tokens;
- FULL: B concurrent short requests for every decode capture size B, so
  decode forwards replay the B bucket.

Engine arguments follow the canonical ShareGPT capture.  No duration is read.

Usage: MACA_LAUNCHID_DIR=OUT python launchid_driver.py MODEL TP
"""
import json
import os
import random
import sys
import time


def main():
    model, tp = sys.argv[1], int(sys.argv[2])
    out = os.environ["MACA_LAUNCHID_DIR"]
    from vllm import EngineArgs, SamplingParams
    from vllm.v1.engine.llm_engine import LLMEngine

    engine = LLMEngine.from_engine_args(EngineArgs(
        model=model, dtype="bfloat16", tensor_parallel_size=tp,
        gpu_memory_utilization=0.8, max_model_len=8192,
        max_num_batched_tokens=8192, max_num_seqs=256,
        enable_prefix_caching=True, seed=42, async_scheduling=True,
        enable_chunked_prefill=True))
    compilation = engine.vllm_config.compilation_config
    sizes = sorted(size for size in compilation.cudagraph_capture_sizes if size <= 256)
    rng = random.Random(20260926)
    rid = [0]

    def run(prompts, max_tokens):
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=0.0)
        for length in prompts:
            engine.add_request(f"r{rid[0]}", {"prompt_token_ids": [
                rng.randint(1000, 30000) for _ in range(length)]}, params)
            rid[0] += 1
        while engine.has_unfinished_requests():
            engine.step()

    run([32] * 4, 4)  # warmup, not recorded
    open(os.path.join(out, "profile.on"), "w").close()
    for tokens in range(1, 513):
        run([tokens], 1)
    for batch in sizes:
        run([4] * batch, 4)
    os.remove(os.path.join(out, "profile.on"))
    deadline = time.time() + 900
    while time.time() < deadline:
        run([8], 2)  # one more forward per rank lets each recorder stop
        if len([f for f in os.listdir(out) if f.startswith("done.")]) >= tp:
            break
        time.sleep(2)
    with open(os.path.join(out, "driver.json"), "w") as handle:
        json.dump({"model": model, "tensor_parallel_size": tp,
                   "decode_capture_sizes": sizes,
                   "piecewise_tokens": [1, 512],
                   "compilation_capture_sizes": list(compilation.cudagraph_capture_sizes)},
                  handle)
    print("LAUNCHID_DRIVER_DONE", flush=True)


if __name__ == "__main__":
    main()
