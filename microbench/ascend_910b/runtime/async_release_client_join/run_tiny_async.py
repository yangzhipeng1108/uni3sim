"""Small source-path probe for vLLM 0.18 async scheduling on 910B."""

import asyncio
import json
import os
import time

from vllm import AsyncEngineArgs, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


ROOT = os.environ["MIDLAYER_ASCEND_CONTAINER_TASK_ROOT"]
MODEL = f"{ROOT}/model"
TOKENIZER = os.environ["MIDLAYER_ASCEND_TOKENIZER"]


async def main() -> None:
    args = AsyncEngineArgs(
        model=MODEL,
        tokenizer=TOKENIZER,
        load_format="dummy",
        dtype="bfloat16",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        distributed_executor_backend="mp",
        async_scheduling=True,
        enforce_eager=True,
        gpu_memory_utilization=0.04,
        kv_cache_memory_bytes=32 * 1024 * 1024,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        disable_log_stats=True,
        generation_config="vllm",
    )
    engine = AsyncLLM.from_engine_args(args)
    params = SamplingParams(max_tokens=4, temperature=0)
    start = time.monotonic_ns()
    events = []

    async def request(request_id: str, prompt: str) -> None:
        async for output in engine.generate(prompt, params, request_id):
            events.append({
                "request_id": request_id,
                "output_tokens": len(output.outputs[0].token_ids),
                "finished": output.finished,
                "relative_ns": time.monotonic_ns() - start,
            })

    try:
        await asyncio.gather(
            request("tiny-0", "Hello"),
            request("tiny-1", "The answer is"),
        )
    finally:
        engine.shutdown()
    print(json.dumps({
        "schema_version": 1,
        "kind": "vllm_018_tiny_async_source_path_probe",
        "device": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        "start_monotonic_ns": start,
        "events": events,
    }, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
