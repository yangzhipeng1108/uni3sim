#!/usr/bin/env python3
"""The OpenAI completion route around the handler body, host only (vLLM 0.11.0).

The published ``api_entry`` probe (``measure_http_path`` in
``vllm_host_request_path_runner.py``) serves a bare Starlette route whose
handler reads the body and returns at once.  The deployment reaches the
completion handler through vLLM's own FastAPI application
(``api_server.build_app``: its middleware, the ``validate_json_request``
dependency, pydantic ``CompletionRequest`` validation, and the
``with_cancellation`` / ``load_aware_call`` route decorators, whose handler
runs as its own task), and leaves it the same way: the route wraps the
returned stream generator in a ``StreamingResponse``, Starlette sends the
headers and starts iterating ``completion_stream_generator``, which pulls
``merge_async_iterators`` over ``engine_client.generate()``, and only then does
``generate()`` reach ``Processor.process_inputs``.

This runner serves the real application object with one substitution: the
``openai_serving_completion`` state is an ``OpenAIServingCompletion`` whose
``create_completion`` skips the render/tokenize body (priced by the
tokenizer terms) and whose engine client's ``generate`` records its entry and
ends.  The stream generator is the real ``completion_stream_generator`` and
the merge is the real ``merge_async_iterators``.  Requests are sent by the
real benchmark client function (``async_request_openai_completions``) from a
separate process over a shared ``aiohttp`` session, one at a time.

Two regions are reported per request, on the one monotonic clock both
processes read:

``route_entry``   client ``start_time`` -> handler body entry (what the
                  published ``api_entry`` stands for).
``route_to_generate``  handler body return -> ``generate()`` entry, i.e. the
                  capture's ``openai_create_completion`` end ->
                  ``engine_input_processor`` start less the few statements
                  ``AsyncLLM.generate``/``add_request`` run before
                  ``process_inputs``.

No serving or evaluation duration is read.

Usage (A800 host, no device):
  python completion_route_region_runner.py MODEL_PATH OUT.json
"""
import asyncio
import json
import multiprocessing
import os
import socket
import statistics
import sys
import time

TRIALS = 200
WARMUP = 20
PROMPT = ("Midlayer host request path probe. " * 12).strip()


def _serve(model_path, port, record_path, ready):
    import uvicorn
    import uvloop
    import vllm.platforms
    from vllm.utils import resolve_obj_by_qualname

    # The deployment platform, named rather than probed: the route never
    # consults it, and probing it would need the device this runner avoids.
    vllm.platforms._current_platform = resolve_obj_by_qualname(
        "vllm.platforms.cuda.CudaPlatform")()
    from vllm.entrypoints.openai.api_server import build_app
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.entrypoints.openai.protocol import RequestResponseMetadata
    from vllm.entrypoints.openai.serving_completion import (
        OpenAIServingCompletion)
    from vllm.utils import FlexibleArgumentParser, merge_async_iterators

    args = make_arg_parser(FlexibleArgumentParser()).parse_args(
        ["--model", model_path])
    app = build_app(args)
    records = open(record_path, "w")

    async def generate(request_id, marks):
        marks["generate_entry_ns"] = time.perf_counter_ns()
        records.write(json.dumps(marks) + "\n")
        records.flush()
        return
        yield  # an async generator, as ``AsyncLLM.generate`` is

    serving = object.__new__(OpenAIServingCompletion)
    serving.enable_prompt_tokens_details = False

    async def create_completion(request, raw_request=None):
        marks = {"request_id": raw_request.headers["x-request-id"],
                 "body_entry_ns": time.perf_counter_ns()}
        request_id = f"cmpl-{marks['request_id']}"
        result_generator = merge_async_iterators(
            generate(f"{request_id}-0", marks))
        stream = serving.completion_stream_generator(
            request, [{"prompt_token_ids": [1]}], result_generator,
            request_id, int(time.time()), request.model, num_prompts=1,
            tokenizer=None,
            request_metadata=RequestResponseMetadata(request_id=request_id),
            enable_force_include_usage=False)
        marks["body_return_ns"] = time.perf_counter_ns()
        return stream

    serving.create_completion = create_completion
    app.state.openai_serving_completion = serving
    app.state.log_stats = False
    app.state.enable_server_load_tracking = args.enable_server_load_tracking
    config = uvicorn.Config(app, host="127.0.0.1", port=port,
                            log_level="error")
    server = uvicorn.Server(config)

    async def run():
        task = asyncio.get_running_loop().create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.01)
        ready.set()
        await task

    uvloop.run(run())


async def _drive(model_path, port):
    import aiohttp
    from vllm.benchmarks.lib.endpoint_request_func import (
        RequestFuncInput, async_request_openai_completions)

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0,
                                     keepalive_timeout=60, force_close=False,
                                     ssl=False)
    starts = {}
    async with aiohttp.ClientSession(connector=connector,
                                     trust_env=False) as session:
        for index in range(WARMUP + TRIALS):
            request_id = f"probe-{index:06d}"
            output = await async_request_openai_completions(
                RequestFuncInput(
                    prompt=PROMPT, api_url=f"http://127.0.0.1:{port}/v1/completions",
                    prompt_len=86, output_len=16, model=model_path,
                    model_name=model_path, ignore_eos=True,
                    request_id=request_id),
                session=session)
            starts[request_id] = round(output.start_time * 1e9)
    return starts


def _summary(samples):
    ordered = sorted(samples)
    return {"median_s": statistics.median(ordered) / 1e9,
            "p10_s": ordered[len(ordered) // 10] / 1e9,
            "p90_s": ordered[(9 * len(ordered)) // 10] / 1e9,
            "sample_count": len(ordered)}


def main():
    model_path, out_path = sys.argv[1], sys.argv[2]
    record_path = out_path + ".records.jsonl"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    server = context.Process(target=_serve,
                             args=(model_path, port, record_path, ready),
                             daemon=True)
    server.start()
    if not ready.wait(120):
        raise RuntimeError("probe server did not start")
    try:
        starts = asyncio.run(_drive(model_path, port))
    finally:
        server.terminate()
        server.join(30)
    marks = [json.loads(line) for line in open(record_path)]
    by_id = {mark["request_id"]: mark for mark in marks}
    measured = [f"probe-{index:06d}" for index in range(WARMUP, WARMUP + TRIALS)]
    route_entry = [by_id[rid]["body_entry_ns"] - starts[rid] for rid in measured]
    route_to_generate = [by_id[rid]["generate_entry_ns"] - by_id[rid]["body_return_ns"]
                         for rid in measured]
    import vllm
    result = {
        "schema_version": 1, "hardware_id": sys.argv[3],
        "vllm_version": vllm.__version__,
        "measurement_boundary": "host_only_no_device_no_model",
        "production_or_evaluation_duration_consumed": False,
        "trials": TRIALS, "warmup": WARMUP,
        "route_entry": _summary(route_entry),
        "route_to_generate": _summary(route_to_generate),
    }
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps({k: (v["median_s"] * 1e6 if isinstance(v, dict) else v)
                      for k, v in result.items()}))


if __name__ == "__main__":
    main()
