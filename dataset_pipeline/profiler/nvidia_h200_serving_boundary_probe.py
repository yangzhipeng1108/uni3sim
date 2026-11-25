"""Bounded Host-boundary Serving diagnostics.

Original128-request arrival/length geometry is preserved. Prompt IDs are
explicit independent representatives with distinct first blocks. Results
cannot replace the canonical ShareGPT metrics or price Forward mechanisms.
The explicit selective-MoE mode instead observes one complete first request
under reference/selective/reference code events without changing functions.
"""
import argparse
from array import array
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

EVENT_FIELDS = ("label", "start_ns", "end_ns", "thread_cpu_ns", "step", "batch",
                "tokens", "qmax", "thread_id", "object_id")

def load_geometry(path):
    data = json.loads(path.read_text())
    if data["kind"] != "h200_qwen4_qps64_diagnostic_geometry" or len(data["rows"]) != 128:
        raise ValueError("requires the fixed128-request diagnostic geometry")
    for index, row in enumerate(data["rows"]):
        if (set(row) != {"index", "arrival_s", "input_tokens", "output_tokens"}
                or type(row["index"]) is not int or row["index"] != index
                or type(row["arrival_s"]) not in (int, float)
                or not math.isfinite(row["arrival_s"]) or row["arrival_s"] < 0
                or type(row["input_tokens"]) is not int or row["input_tokens"] < 1
                or type(row["output_tokens"]) is not int or row["output_tokens"] < 1
                or row["input_tokens"] + row["output_tokens"] > 8192):
            raise ValueError("invalid original request geometry")
        if index and row["arrival_s"] < data["rows"][index - 1]["arrival_s"]:
            raise ValueError("request geometry must preserve original arrival order")
    digest = data["canonical_requests_sha256"]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("canonical request content identity is invalid")
    return data


async def replay(rows, endpoint, *, preflight=False, model_alias='boundary-qwen4',
                 cache_salt=None, temperature=0.6):
    import aiohttp
    origin_ns = time.perf_counter_ns()
    origin = origin_ns * 1e-9
    timeout = aiohttp.ClientTimeout(total=240)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        async def one(row):
            remaining = origin + row["arrival_s"] - time.perf_counter()
            if remaining > 0:
                await asyncio.sleep(remaining)
            started = time.perf_counter()
            payload = {"model": model_alias, "prompt": [1000 + row["index"]] + [7777] * (row["input_tokens"] - 1),
                "max_tokens": row["output_tokens"], "ignore_eos": True,
                "temperature": temperature, "top_k": 20, "top_p": 0.95,
                "stream": True, "stream_options": {"include_usage": True}}
            if cache_salt is not None:
                payload['cache_salt'] = cache_salt
            first_choice, first_text, usage, pending, done = None, None, None, b"", False
            text_parts = []
            async with session.post(endpoint + "/v1/completions", json=payload,
                    headers={"X-Request-Id": f"boundary-{row['index']}"}) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP{response.status}: {(await response.text())[:1000]}")
                async for chunk in response.content.iter_any():
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        if not line.startswith(b"data: "):
                            raise ValueError("unexpected completion stream framing")
                        value = line[6:]
                        if value == b"[DONE]":
                            done = True
                            continue
                        message = json.loads(value)
                        if "usage" in message and message["usage"] is not None:
                            usage = message["usage"]
                        if message["choices"]:
                            text_parts.append(message['choices'][0]['text'])
                            now = time.perf_counter()
                            if first_choice is None:
                                first_choice = now
                            if message["choices"][0]["text"] and first_text is None:
                                first_text = now
            ended = time.perf_counter()
            if pending.strip() or not done or usage is None or first_choice is None:
                raise ValueError("incomplete SSE completion/usage")
            if (usage["prompt_tokens"] != row["input_tokens"]
                    or usage["completion_tokens"] != row["output_tokens"]):
                raise ValueError("actual prompt/output token count differs from the fixed request")
            return {**row, "actual_start_s": started - origin,
                "arrival_lateness_ms": (started - origin - row["arrival_s"]) * 1000,
                "first_choice_ms": (first_choice - started) * 1000,
                "first_text_ms": None if first_text is None else (first_text - started) * 1000,
                "e2e_ms": (ended - started) * 1000, "usage": usage,
                'output_text_sha256': hashlib.sha256(''.join(text_parts).encode()).hexdigest()}
        results = await asyncio.gather(*(one(row) for row in rows))
    return {"requests": results, "wall_s": time.perf_counter() - origin,
            "preflight_only": preflight, "clock": {"name": "time.perf_counter_ns",
                "origin_ns": origin_ns, "implementation": time.get_clock_info("perf_counter").implementation}}


async def preflight():
    from aiohttp import web
    async def complete(request):
        body = await request.json()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        messages = ({"choices": [{"text": "ok"}], "usage": None},
                    {"choices": [], "usage": {"prompt_tokens": len(body["prompt"]),
                     "completion_tokens": body["max_tokens"]}})
        for message in messages:
            # Deliberately fragment frames to exercise the exact remote parser.
            content = ("data: " + json.dumps(message) + "\n\n").encode()
            await response.write(content[:7]); await response.write(content[7:])
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response
    app = web.Application(); app.router.add_post("/v1/completions", complete)
    runner = web.AppRunner(app); await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        result = await replay([{"index": i, "arrival_s": i * 0.002,
            "input_tokens": i + 1, "output_tokens": i + 2} for i in range(3)],
            f"http://127.0.0.1:{port}", preflight=True)
        assert len(result["requests"]) == 3
    finally:
        await runner.cleanup()
    print(json.dumps({"status": "local_client_preflight_passed", "requests": 3}))


def stop_tree(process):
    """Stop a dedicated session (or its direct child) using only POSIX tools.

    Modal collectors own a session. Selective servers inherit that group so
    an outer timeout also reaches their workers, without a psutil dependency.
    """
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.kill(); process.wait(timeout=10)
    # An owned group can outlive its parent; force any remaining members down.
    # An inherited direct child has no group named after its PID.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def set_mode(path, mode):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(mode)
    temporary.replace(path)


def observer_summary(events):
    counts = {"execute_model": 0, "sample_tokens": 0, "all_rows": 0}
    metadata = []
    for path in events.glob("*/metadata.json"):
        item = json.loads(path.read_text())
        if (item["itemsize"] != 8 or item["byteorder"] != sys.byteorder
                or item["unclosed_profile_frames"] != 0
                or tuple(item["fields"]) != EVENT_FIELDS):
            raise ValueError("incomplete or incompatible observer records")
        if "observer" in item:
            if item["observer"] != "native_cupti_host_api_callbacks_and_outer_markers":
                raise ValueError("unknown native observer contract")
            if (any(type(item[name]) is not int or item[name] != 0
                    for name in ("native_init_status", "native_dump_status", "error_code"))
                    or type(item["attempted_rows"]) is not int
                    or item["attempted_rows"] != item["rows"]):
                raise ValueError("native API observer failed or lost records")
        values = array("q")
        values.frombytes((path.parent / "events.bin").read_bytes())
        width = len(item["fields"])
        if len(values) != item["rows"] * width:
            raise ValueError("observer record extent differs from metadata")
        counts["all_rows"] += item["rows"]
        counts["execute_model"] += sum(values[index] == 0 for index in range(0, len(values), width))
        counts["sample_tokens"] += sum(values[index] == 1 for index in range(0, len(values), width))
        metadata.append(item)
    if counts["execute_model"] < 128 or counts["sample_tokens"] < 128:
        raise ValueError("actual V2 worker calls are missing from the diagnostic")
    return counts, metadata


def ready(process, endpoint, timeout_s=900):
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        if process.poll() is not None:
            raise RuntimeError("vLLM exited before readiness; see retained server logs")
        try:
            with urllib.request.urlopen(endpoint + "/health", timeout=2) as response:
                if response.status == 200:
                    return time.monotonic() - started
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("bounded vLLM readiness timeout")


def selective_moe(args):
    """Full first-request reference/selective/reference diagnostic, never a price."""
    if os.getpgrp() != os.getpid():
        raise ValueError('selective MoE requires a collector launched with start_new_session=True')
    if args.output is None or args.output.exists() or args.hooks is None or args.geometry is None:
        raise ValueError('selective MoE requires new output and frozen hook/input files')
    geometry = json.loads(args.geometry.read_text())
    if (geometry['kind'] != 'h200_moe_single_request_selective_control'
            or geometry['input_tokens'] != 729 or geometry['output_tokens'] != 121
            or geometry['tensor_parallel_size'] != 2 or geometry['expert_parallel_size'] != 2):
        raise ValueError('unexpected complete-model selective-control geometry')
    args.output.mkdir(parents=True)
    from huggingface_hub import snapshot_download
    model = Path(snapshot_download('Qwen/Qwen3-30B-A3B', cache_dir=args.output / 'hf',
        allow_patterns=('*.json', '*.safetensors', '*.txt', '*.model', '*.tiktoken')))
    config = json.loads((model / 'config.json').read_text())
    expected = {'model_type': 'qwen3_moe', 'hidden_size': 2048, 'num_hidden_layers': 48,
        'num_attention_heads': 32, 'num_key_value_heads': 4, 'num_experts': 128,
        'num_experts_per_tok': 8, 'moe_intermediate_size': 768, 'vocab_size': 151936}
    if any(config[name] != value for name, value in expected.items()):
        raise ValueError('public checkpoint geometry differs from original full MoE model')
    (args.output / 'checkpoint_config.json').write_bytes((model / 'config.json').read_bytes())
    hooks = args.output / 'hooks'; hooks.mkdir()
    shutil.copyfile(args.hooks, hooks / 'sitecustomize.py')
    events = args.output / 'events'; events.mkdir()
    flag = args.output / 'observer_mode.json'

    def mode(value, epoch):
        temporary = flag.with_suffix('.tmp')
        temporary.write_text(json.dumps({'mode': value, 'epoch': epoch}))
        temporary.replace(flag)

    mode('disabled', -1)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(hooks) + (os.pathsep + env['PYTHONPATH'] if 'PYTHONPATH' in env else '')
    env.update(MIDLAYER_SELECTIVE_DIRECTORY=str(events), MIDLAYER_SELECTIVE_FLAG=str(flag),
        VLLM_CACHE_ROOT=str(args.output / 'compile_cache'), VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1')
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', str(model),
        '--served-model-name', 'selective-moe', '--host', '127.0.0.1', '--port', '18400',
        '--dtype', 'bfloat16', '--tensor-parallel-size', '2', '--enable-expert-parallel',
        '--seed', '42', '--max-model-len', '8192', '--max-num-batched-tokens', '8192',
        '--max-num-seqs', '256', '--gpu-memory-utilization', '0.8', '--block-size', '16',
        '--enable-prefix-caching', '--async-scheduling', '--cudagraph-metrics',
        '--compilation-config', json.dumps({'mode': 3, 'cudagraph_mode': 'FULL_AND_PIECEWISE',
                                           'max_cudagraph_capture_size': 512})]
    receipt = {'kind': geometry['kind'], 'model_revision': model.name, 'config': expected,
        'geometry': geometry, 'command': command, 'runs': [], 'status': 'running',
        'compiled_model_function_objects_modified': False,
        'historical_provider_or_guard_equivalence_certified': False,
        'scope': 'Full BF16 48-layer TP2EP2; original first input/output lengths; synthetic IDs and greedy output.',
        'cache_protocol': 'Prefix caching stays enabled; distinct request cache_salt prevents cross-control hits.',
        'production_duration_priced': False, 'canonical_changed': False}
    server = None
    try:
        with (args.output / 'server.stdout.log').open('w') as stdout, (args.output / 'server.stderr.log').open('w') as stderr:
            # Inherit the collector's owned session so its timeout reaches all
            # worker processes. The normal legacy collector still owns a new
            # server session below.
            server = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
            receipt['readiness_s'] = ready(server, 'http://127.0.0.1:18400', timeout_s=360)
            rows = [{'index': 0, 'arrival_s': 0.0, 'input_tokens': 729, 'output_tokens': 121}]
            for epoch, state in enumerate(('reference', 'selective', 'reference')):
                mode(state, epoch); time.sleep(.25)
                result = asyncio.run(replay(rows, 'http://127.0.0.1:18400', model_alias='selective-moe',
                    cache_salt=f'midlayer-selective-control-{epoch}', temperature=0.0))
                path = args.output / f'client_{epoch}_{state}.json'
                path.write_text(json.dumps(result, indent=2) + '\n')
                receipt['runs'].append({'epoch': epoch, 'mode': state,
                    'completed_requests': len(result['requests']), 'client_result': path.name,
                    'output_text_sha256': result['requests'][0]['output_text_sha256']})
                time.sleep(.25)  # retain any legal in-flight tail before changing event modes
                mode('disabled', epoch); time.sleep(.05)
            mode('stop', 3); time.sleep(.25)
            receipt['status'] = 'complete_candidate'
    except BaseException as error:
        receipt['status'] = 'failed_retained'
        receipt['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        if server is not None:
            mode('stop', 3); time.sleep(.1)
            stop_tree(server)
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'runs': len(receipt['runs'])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--hooks", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument('--selective-moe', action='store_true')
    args = parser.parse_args()
    if args.cpu_preflight:
        asyncio.run(preflight()); return
    if args.selective_moe:
        selective_moe(args); return
    if args.output is None or args.output.exists() or args.geometry is None or args.hooks is None:
        raise ValueError("requires new output and explicit input/hook files")
    geometry = load_geometry(args.geometry)
    args.output.mkdir(parents=True)
    from huggingface_hub import snapshot_download
    model = Path(snapshot_download("Qwen/Qwen3-4B", cache_dir=args.output / "hf",
        allow_patterns=("*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken")))
    config = json.loads((model / "config.json").read_text())
    if (config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"],
            config["num_attention_heads"], config["num_key_value_heads"], config["vocab_size"]) != (2560, 9728, 36, 32, 8, 151936):
        raise ValueError("public checkpoint geometry differs from Qwen3-4B")
    # A snapshot directory names an immutable HF revision; preserve it explicitly.
    receipt = {"kind": "bounded_qwen4_host_boundary_candidate", "model_revision": model.name,
        "geometry": geometry, "synthetic_ids": "distinct_first_token_then7777; no claimed historical token content",
        "sampling": {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "ignore_eos": True},
        "canonical_changed": False, "production_duration_priced": False, "runs": []}
    endpoint = "http://127.0.0.1:18400"
    for mode in ("reference", "detailed"):
        stage = args.output / mode; stage.mkdir()
        hooks = stage / "hooks"; hooks.mkdir()
        shutil.copyfile(args.hooks, hooks / "sitecustomize.py")
        events = stage / "events"; events.mkdir()
        flag = stage / "observer_mode"; set_mode(flag, "disabled")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(hooks) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
        env["MIDLAYER_BOUNDARY_DIRECTORY"] = str(events)
        env["MIDLAYER_BOUNDARY_FLAG"] = str(flag)
        env["VLLM_CACHE_ROOT"] = str(stage / "compile_cache")
        env["VLLM_NO_USAGE_STATS"] = "1"
        env["DO_NOT_TRACK"] = "1"
        command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model),
            "--served-model-name", "boundary-qwen4", "--host", "127.0.0.1", "--port", "18400",
            "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--seed", "42",
            "--max-model-len", "8192", "--max-num-batched-tokens", "8192", "--max-num-seqs", "256",
            "--gpu-memory-utilization", "0.8", "--block-size", "16", "--enable-prefix-caching",
            "--async-scheduling", "--cudagraph-metrics", "--compilation-config",
            json.dumps({"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"})]
        started = time.monotonic()
        with (stage / "server.stdout.log").open("w") as stdout, (stage / "server.stderr.log").open("w") as stderr:
            server = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                readiness_s = ready(server, endpoint)
                set_mode(flag, mode); time.sleep(0.3)
                result = asyncio.run(replay(geometry["rows"], endpoint))
                set_mode(flag, "dump"); time.sleep(2)
                set_mode(flag, "stop"); time.sleep(0.5)
                (stage / "client_result.json").write_text(json.dumps(result, indent=2) + "\n")
                counts, metadata = observer_summary(events)
                receipt["runs"].append({"mode": mode, "command": command, "readiness_s": readiness_s,
                    "wall_s": time.monotonic() - started, "completed_requests": len(result["requests"]),
                    "observer_files": len(metadata), "observer_counts": counts,
                    "source_method_sha256": [item["source_method_sha256"] for item in metadata]})
            finally:
                stop_tree(server)
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    # The cloud wrapper returns only evidence, never the temporary weights/caches.
    print(json.dumps({"status": "complete_candidate", "runs": len(receipt["runs"])}))


if __name__ == "__main__":
    main()
