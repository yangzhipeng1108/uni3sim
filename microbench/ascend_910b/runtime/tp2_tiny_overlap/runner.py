"""Run a one-layer, original-width Qwen3 MoE through Ascend TP2/EP2 vLLM."""

import asyncio
import json
import os
from pathlib import Path
import time

from vllm import AsyncEngineArgs, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


ROOT = Path(os.environ["MID_TP2_TINY_TASK_DIR"])


async def main():
    args = AsyncEngineArgs(
        model=str(ROOT / "model"), tokenizer=os.environ["MIDLAYER_ASCEND_TOKENIZER"],
        load_format="dummy", dtype="bfloat16", tensor_parallel_size=2,
        pipeline_parallel_size=1, distributed_executor_backend="mp",
        enable_expert_parallel=True, async_scheduling=True,
        enforce_eager=True, gpu_memory_utilization=0.1,
        kv_cache_memory_bytes=64 * 1024 * 1024, max_model_len=1024,
        max_num_batched_tokens=1024, max_num_seqs=4,
        disable_log_stats=True, generation_config="vllm",
    )
    engine = AsyncLLM.from_engine_args(args)
    parameters = SamplingParams(max_tokens=8, temperature=0, ignore_eos=True)
    sequential_warmup = os.environ["MID_TP2_TINY_SEQUENTIAL_WARMUP"] == "1"
    events = []

    async def request(request_id, token_id):
        prompt = {"prompt_token_ids": [token_id] * 729}
        async for output in engine.generate(prompt, parameters, request_id):
            events.append({"request_id": request_id,
                           "output_tokens": len(output.outputs[0].token_ids),
                           "finished": output.finished,
                           "elapsed_ms": (time.monotonic_ns() - start_ns) / 1e6})

    try:
        if sequential_warmup:
            warmup_parameters = SamplingParams(
                max_tokens=2, temperature=0, ignore_eos=True)
            async for _ in engine.generate(
                    {"prompt_token_ids": [5] * 729}, warmup_parameters,
                    "warmup"):
                pass
        start_ns = time.monotonic_ns()
        await asyncio.gather(request("tiny-0", 3), request("tiny-1", 4))
    finally:
        engine.shutdown()
    print("MIDLAYER_TP2_TINY=" + json.dumps({"schema_version": 1,
          "devices": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
          "sequential_warmup": sequential_warmup,
          "events": events}, sort_keys=True), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
