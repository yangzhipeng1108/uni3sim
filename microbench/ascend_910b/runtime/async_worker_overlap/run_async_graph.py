"""Tiny 36-layer Qwen3 async MP graph path for worker overlap ordering."""

import asyncio
import json
import os
from pathlib import Path
import time

from vllm import AsyncEngineArgs, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


ROOT = Path(os.environ["MIDLAYER_ASCEND_MODEL_WORK_ROOT"])
TOKENIZER = os.environ["MIDLAYER_ASCEND_TOKENIZER"]


async def main() -> None:
    if os.environ["ASCEND_RT_VISIBLE_DEVICES"] != "5":
        raise RuntimeError("async graph probe is assigned to physical NPU 5")
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=str(ROOT / "layer_36"),
        tokenizer=TOKENIZER,
        load_format="dummy",
        dtype="bfloat16",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        distributed_executor_backend="mp",
        async_scheduling=True,
        enforce_eager=False,
        gpu_memory_utilization=0.1,
        kv_cache_memory_bytes=256 * 1024 * 1024,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=4,
        disable_log_stats=True,
        generation_config="vllm",
    ))
    parameters = SamplingParams(
        max_tokens=4, temperature=0, ignore_eos=True)
    start_ns = time.monotonic_ns()
    rows = []

    async def request(request_id: str, token_id: int) -> None:
        prompt = {"prompt_token_ids": [token_id] * 650}
        async for output in engine.generate(prompt, parameters, request_id):
            rows.append({
                "request_id": request_id,
                "output_tokens": len(output.outputs[0].token_ids),
                "finished": output.finished,
                "relative_ns": time.monotonic_ns() - start_ns,
            })

    try:
        await asyncio.gather(
            request("tiny-0", 3), request("tiny-1", 4))
    finally:
        engine.shutdown()
    print("MIDLAYER_ASYNC_GRAPH=" + json.dumps({
        "schema_version": 1,
        "physical_device_id": 5,
        "model_layers": 36,
        "hidden_size": 128,
        "prompt_tokens_per_request": 650,
        "rows": rows,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
