"""HTTP benchmark and ground-truth assembly for ShareGPT captures."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Sequence

from evaluation.serving.experiments.experiment_plan import git_code_identity
from src.serving.request_gen import load_canonical_requests
from src.serving.request_gen.request_stream import (
    RequestStream,
    RequestSpec,
    request_stream_sha256,
)

from evaluation.serving.experiments.ground_truth_case_contract import (
    ROOT,
    sha256_file,
)


def _sample_sharegpt(
    *,
    model: str,
    dataset: str,
    seed: int,
    requests: Sequence[RequestSpec],
) -> tuple[Any, list[Any]]:
    from transformers import AutoTokenizer
    from vllm.benchmarks.datasets import ShareGPTDataset

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    samples = ShareGPTDataset(
        dataset_path=dataset,
        random_seed=seed,
    ).sample(
        tokenizer,
        num_requests=len(requests),
        output_len=None,
    )
    if len(samples) != len(requests):
        raise ValueError(
            "ShareGPT sample count disagrees with canonical requests: "
            f"{len(samples)} != {len(requests)}"
        )
    for index, (sample, request) in enumerate(zip(samples, requests)):
        if sample.prompt_len != request.input_tokens:
            raise ValueError(
                f"ShareGPT input length mismatch at request {index}: "
                f"{sample.prompt_len} != {request.input_tokens}"
            )
        if sample.expected_output_len != request.output_tokens:
            raise ValueError(
                f"ShareGPT output length mismatch at request {index}: "
                f"{sample.expected_output_len} != {request.output_tokens}"
            )
    return tokenizer, samples


def _diagnostic_output_capped_requests(
    requests: Sequence[RequestSpec],
    *,
    diagnostic_request_count: int | None,
    diagnostic_output_token_cap: int | None,
) -> tuple[RequestSpec, ...]:
    """Derive a short-output diagnostic stream after canonical sample checks."""
    if diagnostic_output_token_cap is None:
        return tuple(requests)
    if diagnostic_request_count is None:
        raise ValueError("diagnostic output cap requires diagnostic request count")
    if (type(diagnostic_output_token_cap) is not int
            or not 4 <= diagnostic_output_token_cap <= 8):
        raise ValueError("diagnostic output cap must be an integer from four to eight")
    if any(request.output_token_ids is not None for request in requests):
        raise ValueError("diagnostic output cap requires length-only requests")
    return tuple(
        replace(request, output_tokens=min(request.output_tokens,
                                           diagnostic_output_token_cap))
        for request in requests
    )


def _prefix_cache_reuse_evidence(
    tokenizer: Any,
    samples: Sequence[Any],
    *,
    kv_block_size_tokens: int,
) -> dict[str, Any]:
    if (
        isinstance(kv_block_size_tokens, bool)
        or not isinstance(kv_block_size_tokens, int)
        or kv_block_size_tokens < 1
    ):
        raise ValueError("kv_block_size_tokens must be a positive integer")
    token_sequences = []
    for index, sample in enumerate(samples):
        encoded = tokenizer(sample.prompt)
        token_ids = list(encoded.input_ids)
        if len(token_ids) != sample.prompt_len:
            raise ValueError(
                "prefix-cache evidence token count disagrees with sampled "
                f"prompt {index}: {len(token_ids)} != {sample.prompt_len}"
            )
        if any(isinstance(token_id, bool) or not isinstance(token_id, int)
               for token_id in token_ids):
            raise TypeError("tokenizer input_ids must contain integers")
        token_sequences.append(token_ids)

    shared_prefix_block_pairs = 0
    max_shared_prefix_blocks = 0
    for left_index, left in enumerate(token_sequences):
        for right in token_sequences[left_index + 1:]:
            shared_tokens = 0
            for left_token, right_token in zip(left, right):
                if left_token != right_token:
                    break
                shared_tokens += 1
            shared_blocks = shared_tokens // kv_block_size_tokens
            if shared_blocks:
                shared_prefix_block_pairs += 1
                max_shared_prefix_blocks = max(
                    max_shared_prefix_blocks, shared_blocks
                )
    token_stream_bytes = json.dumps(
        token_sequences,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    evidence = {
        "kind": "static_prompt_token_prefix_analysis_v1",
        "logical_prefix_hit_observability": "unavailable_without_instrumentation",
        "enable_prefix_caching": True,
        "kv_block_size_tokens": kv_block_size_tokens,
        "request_count": len(token_sequences),
        "prompt_token_stream_sha256": hashlib.sha256(token_stream_bytes).hexdigest(),
        "shared_prefix_block_pair_count": shared_prefix_block_pairs,
        "max_shared_prefix_blocks": max_shared_prefix_blocks,
        "effective_reuse_representative_tokens": 0,
        "representative_provenance": "scheduler_equivalent_representative",
    }
    if shared_prefix_block_pairs:
        raise ValueError(
            "canonical prompts contain a shared complete prefix-cache block; "
            "a zero effective-reuse representative is not valid"
        )
    return evidence


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values_s: Sequence[float]) -> dict[str, float]:
    if not values_s:
        raise ValueError("latency distribution must not be empty")
    return {
        "mean_ms": sum(values_s) * 1000.0 / len(values_s),
        "p50_ms": _percentile(values_s, 50.0) * 1000.0,
        "p90_ms": _percentile(values_s, 90.0) * 1000.0,
        "p99_ms": _percentile(values_s, 99.0) * 1000.0,
    }


# Highest descriptor the replay client pre-sizes its fd table to (see below).
CLIENT_FD_TABLE_RESERVE = 1023


def _reserve_client_fd_table() -> None:
    """Grow this process's fd table once, before warmup and replay.

    On Linux 5.15 the first use of fd 64, 128, 256, ... in a
    multithreaded process expands the fd table and waits for a system-wide RCU
    grace period inside socket().  During replay that froze the asyncio loop
    for 12-338 ms when about 35 requests were first in flight, so every request
    due in that window went out late (dispatch-lag failures on H200).  The
    table never shrinks, so paying the expansion here removes the stall without
    changing the arrival schedule or when connections are opened.
    """

    if sys.platform != "win32":
        import resource

        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_limit != resource.RLIM_INFINITY and soft_limit <= CLIENT_FD_TABLE_RESERVE:
            raise RuntimeError(
                f"RLIMIT_NOFILE soft limit {soft_limit} does not allow fd "
                f"{CLIENT_FD_TABLE_RESERVE}")
    reserve_fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(reserve_fd, CLIENT_FD_TABLE_RESERVE, inheritable=False)
    os.close(CLIENT_FD_TABLE_RESERVE)
    os.close(reserve_fd)


async def _run_http_benchmark(
    *,
    base_url: str,
    served_model_name: str,
    samples: Sequence[Any],
    requests: Sequence[RequestSpec],
    require_request_id: bool,
    tokenizer: Any | None = None,
    max_model_len: int = 8192,
    benchmark_boundary_callback: Callable[[], None] | None = None,
    arrival_probe_per_token_s: float = 0.0,
) -> dict[str, Any]:
    import aiohttp
    try:
        from vllm.benchmarks.lib.endpoint_request_func import (
            RequestFuncInput,
            async_request_openai_completions,
        )
    except ModuleNotFoundError as error:
        if error.name not in {
            "vllm.benchmarks.lib",
            "vllm.benchmarks.lib.endpoint_request_func",
        }:
            raise
        from vllm.benchmarks.endpoint_request_func import (
            RequestFuncInput,
            async_request_openai_completions,
        )

    request_fields = RequestFuncInput.__dataclass_fields__
    request_function_parameters = inspect.signature(
        async_request_openai_completions
    ).parameters
    supports_request_id = "request_id" in request_fields
    supports_shared_session = "session" in request_function_parameters
    if supports_request_id != supports_shared_session:
        raise RuntimeError(
            "unsupported vLLM benchmark client API: request_id and session "
            "support must change together"
        )
    if require_request_id and not supports_request_id:
        raise RuntimeError(
            "Scheduler trace requires vLLM benchmark request_id support"
        )

    def request_input(
        *,
        prompt: str,
        prompt_len: int,
        output_len: int,
        request_id: str,
    ) -> Any:
        kwargs = {
            "prompt": prompt,
            "api_url": api_url,
            "prompt_len": prompt_len,
            "output_len": output_len,
            "model": served_model_name,
            "model_name": served_model_name,
            "ignore_eos": True,
        }
        if supports_request_id:
            kwargs["request_id"] = request_id
        return RequestFuncInput(**kwargs)

    async def send(request: Any, session: Any) -> Any:
        if supports_shared_session:
            return await async_request_openai_completions(
                request_func_input=request,
                session=session,
            )
        return await async_request_openai_completions(
            request_func_input=request,
        )

    _reserve_client_fd_table()
    connector = aiohttp.TCPConnector(
        limit=0,
        limit_per_host=0,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True,
        force_close=False,
        ssl=False,
    )
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    api_url = f"{base_url.rstrip('/')}/v1/completions"
    async with aiohttp.ClientSession(
        connector=connector,
        trust_env=False,
        timeout=timeout,
    ) as session:
        if tokenizer is None:
            warmup_specs = [("sharegpt-warmup", "Warmup.", 1, 8)]
            mixed_warmup_specs: list[tuple[str, str, int, int]] = []
            isolated_warmup_specs: list[tuple[str, str, int, int]] = []
        else:
            eligible = [
                (index, sample)
                for index, sample in enumerate(samples)
                if sample.prompt_len <= max_model_len - 256
            ]
            if len(eligible) < 8:
                raise RuntimeError(
                    "deterministic warmup requires eight prompts with model-length "
                    "headroom"
                )
            eligible.sort(key=lambda value: value[1].prompt_len)
            positions = [
                round(position * (len(eligible) - 1) / 7)
                for position in range(8)
            ]

            def isolated_warmup_spec(
                wave: str,
                ordinal: int,
                output_len: int,
            ) -> tuple[str, str, int, int]:
                sample_index, sample = eligible[positions[ordinal]]
                prefix = (
                    f"Midlayer isolated {wave} warmup request {ordinal}; "
                    f"canonical sample {sample_index}; no prefix reuse. " * 3
                )
                prompt = prefix + sample.prompt
                prompt_len = len(tokenizer(prompt).input_ids)
                if prompt_len + output_len > max_model_len:
                    raise RuntimeError(
                        "warmup prompt exceeds configured maximum model length"
                    )
                return (
                    f"warmup-{wave}-{ordinal:02d}",
                    prompt,
                    prompt_len,
                    output_len,
                )

            warmup_specs = [
                isolated_warmup_spec("batch", ordinal, 8)
                for ordinal in range(8)
            ]
            mixed_warmup_specs = [
                isolated_warmup_spec(
                    "mixed",
                    ordinal,
                    64 if ordinal == 0 else 8,
                )
                for ordinal in range(8)
            ]
            isolated_warmup_specs = [
                isolated_warmup_spec("isolated", 7, 64)
            ]

        async def send_warmup(
            specification: tuple[str, str, int, int]
        ) -> Any:
            request_id, prompt, prompt_len, output_len = specification
            output = await send(request_input(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
                request_id=request_id,
            ), session)
            if not output.success:
                raise RuntimeError(
                    f"vLLM warmup request {request_id} failed: {output.error}"
                )
            if output.output_tokens != output_len:
                raise RuntimeError(
                    f"vLLM warmup request {request_id} returned "
                    f"{output.output_tokens} tokens instead of {output_len}"
                )
            return output

        warmup_started = time.perf_counter()
        await asyncio.gather(*(
            send_warmup(specification) for specification in warmup_specs
        ))
        if mixed_warmup_specs:
            anchor = asyncio.create_task(send_warmup(mixed_warmup_specs[0]))
            await asyncio.sleep(0.25)
            followers = await asyncio.gather(*(
                send_warmup(specification)
                for specification in mixed_warmup_specs[1:]
            ))
            await anchor
            if len(followers) != 7:
                raise RuntimeError("mixed warmup follower count is inconsistent")
        for specification in isolated_warmup_specs:
            await send_warmup(specification)
        warmup_duration_s = time.perf_counter() - warmup_started
        if benchmark_boundary_callback is not None:
            benchmark_boundary_callback()

        dispatch_start = asyncio.Event()
        all_requests_ready = asyncio.Event()
        ready_request_count = 0
        benchmark_start_monotonic_ns: int | None = None

        async def submit_one(index: int) -> tuple[Any, int]:
            nonlocal ready_request_count
            ready_request_count += 1
            if ready_request_count == len(requests):
                all_requests_ready.set()
            await dispatch_start.wait()
            if benchmark_start_monotonic_ns is None:
                raise RuntimeError("benchmark clock was not initialized")
            request = requests[index]
            # Opt-in controlled probe: delay this request's submission in
            # proportion to its prompt length, which is the only shape of
            # perturbation that changes the relative spacing between arrivals
            # and can therefore reach the Scheduler's batching decisions.  A
            # zero rate reproduces the canonical arrival replay exactly.
            probe_offset_s = arrival_probe_per_token_s * request.input_tokens
            delay_ns = (
                benchmark_start_monotonic_ns
                + round(
                    (request.arrival_time_s + probe_offset_s) * 1_000_000_000)
                - time.perf_counter_ns()
            )
            if delay_ns > 0:
                await asyncio.sleep(delay_ns / 1_000_000_000)
            submitted_at_ns = time.perf_counter_ns()
            completion_request = request_input(
                prompt=samples[index].prompt,
                prompt_len=request.input_tokens,
                output_len=request.output_tokens,
                request_id=request.request_id,
            )
            output = await send(completion_request, session)
            return output, submitted_at_ns

        request_tasks = [
            asyncio.create_task(submit_one(index))
            for index in range(len(requests))
        ]
        await all_requests_ready.wait()
        benchmark_start_monotonic_ns = time.perf_counter_ns()
        dispatch_start.set()
        completed = await asyncio.gather(*request_tasks)
        duration_s = (
            time.perf_counter_ns() - benchmark_start_monotonic_ns
        ) / 1_000_000_000

    if benchmark_start_monotonic_ns is None:
        raise RuntimeError("benchmark clock was not initialized")
    host_boot_id = Path(
        "/proc/sys/kernel/random/boot_id"
    ).read_text(encoding="utf-8").strip()
    if not host_boot_id:
        raise RuntimeError("host boot ID is empty")

    per_request = []
    ttft_s = []
    tpot_s = []
    e2e_s = []
    total_output_tokens = 0
    failures = []
    for index, ((output, submitted_at_ns), request) in enumerate(
            zip(completed, requests)):
        if not output.success:
            failures.append({
                "request_id": request.request_id,
                "error": output.error,
            })
            continue
        if output.output_tokens != request.output_tokens:
            failures.append({
                "request_id": request.request_id,
                "error": "output token count disagrees with canonical request",
                "expected_output_tokens": request.output_tokens,
                "actual_output_tokens": output.output_tokens,
            })
            continue
        request_tpot_s = 0.0
        if output.output_tokens > 1:
            request_tpot_s = (
                output.latency - output.ttft
            ) / (output.output_tokens - 1)
            tpot_s.append(request_tpot_s)
        ttft_s.append(output.ttft)
        e2e_s.append(output.latency)
        total_output_tokens += output.output_tokens
        request_start_s = output.start_time
        if (
            isinstance(request_start_s, bool)
            or not isinstance(request_start_s, (int, float))
            or not math.isfinite(request_start_s)
        ):
            raise TypeError("vLLM benchmark request start time is invalid")
        request_start_ns = round(float(request_start_s) * 1_000_000_000)
        if request_start_ns < submitted_at_ns:
            raise RuntimeError(
                "vLLM benchmark TTFT start predates request submission"
            )
        first_token_ns = request_start_ns + round(
            float(output.ttft) * 1_000_000_000
        )
        submit_offset_s = (
            submitted_at_ns - benchmark_start_monotonic_ns
        ) / 1_000_000_000
        per_request.append({
            "request_id": request.request_id,
            "scheduled_arrival_s": request.arrival_time_s,
            "arrival_probe_offset_s": (
                arrival_probe_per_token_s * request.input_tokens),
            "submit_offset_s": submit_offset_s,
            # Dispatch lag measures how late the client was against the
            # schedule it was asked to hit, so the requested probe offset is
            # part of that schedule and is not counted as lag.
            "dispatch_lag_ms": (
                submit_offset_s
                - request.arrival_time_s
                - arrival_probe_per_token_s * request.input_tokens
            ) * 1000.0,
            "input_tokens": request.input_tokens,
            "output_tokens": output.output_tokens,
            "submit_monotonic_ns": submitted_at_ns,
            "ttft_start_monotonic_ns": request_start_ns,
            "first_token_monotonic_ns": first_token_ns,
            "ttft_ms": output.ttft * 1000.0,
            "tpot_ms": request_tpot_s * 1000.0,
            "e2e_ms": output.latency * 1000.0,
        })
    if failures:
        raise RuntimeError(json.dumps({
            "message": "one or more production Serving requests failed",
            "failures": failures,
        }, sort_keys=True))
    return {
        "warmup": {
            "request_count": (
                len(warmup_specs)
                + len(mixed_warmup_specs)
                + len(isolated_warmup_specs)
            ),
            "batch_wave_request_count": len(warmup_specs),
            "mixed_wave_request_count": len(mixed_warmup_specs),
            "isolated_wave_request_count": len(isolated_warmup_specs),
            "duration_s": warmup_duration_s,
            "request_ids": [
                specification[0]
                for specification in (
                    *warmup_specs,
                    *mixed_warmup_specs,
                    *isolated_warmup_specs,
                )
            ],
            "prompt_tokens": [
                specification[2]
                for specification in (
                    *warmup_specs,
                    *mixed_warmup_specs,
                    *isolated_warmup_specs,
                )
            ],
            "output_tokens": [
                specification[3]
                for specification in (
                    *warmup_specs,
                    *mixed_warmup_specs,
                    *isolated_warmup_specs,
                )
            ],
        },
        "duration_s": duration_s,
        "benchmark_clock": {
            "clock_domain": "shared_host_time.perf_counter_ns",
            "host_boot_id": host_boot_id,
            "benchmark_start_monotonic_ns": benchmark_start_monotonic_ns,
        },
        "completed_requests": len(per_request),
        "total_input_tokens": sum(request.input_tokens for request in requests),
        "total_output_tokens": total_output_tokens,
        "request_throughput_per_s": len(per_request) / duration_s,
        "output_token_throughput_per_s": total_output_tokens / duration_s,
        "ttft": _distribution(ttft_s),
        "tpot": _distribution(tpot_s),
        "e2e": _distribution(e2e_s),
        "per_request": per_request,
    }


def build_ground_truth(
    *,
    model: str,
    dataset: str,
    seed: int,
    case: dict[str, Any],
    server_plan: dict[str, Any],
    dry_run: bool,
    diagnostic_request_count: int | None = None,
    diagnostic_output_token_cap: int | None = None,
    benchmark_boundary_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    requests_path = case["requests_path"]
    canonical_stream = load_canonical_requests(requests_path)
    canonical_requests = canonical_stream.requests
    if len(canonical_requests) != case["requests_row_count"]:
        raise ValueError("requests.csv row count disagrees with case manifest")
    if diagnostic_request_count is not None:
        if (type(diagnostic_request_count) is not int
                or not 8 <= diagnostic_request_count < len(canonical_requests)):
            raise ValueError("diagnostic request count must be 8 through canonical count - 1")
    requests = (canonical_requests if diagnostic_request_count is None
                else canonical_requests[:diagnostic_request_count])
    if requests[0].arrival_time_s != 0:
        raise ValueError("canonical request schedule must start at zero seconds")
    if any(request.output_tokens < 1 for request in requests):
        raise ValueError("ground-truth requests must generate at least one token")
    if any(request.prefix_hit_tokens != 0 for request in requests):
        raise ValueError(
            "ShareGPT ground-truth runner does not support prefix-hit requests"
        )
    tokenizer, samples = _sample_sharegpt(
        model=model,
        dataset=dataset,
        seed=seed,
        requests=canonical_requests,
    )
    if diagnostic_request_count is not None:
        samples = samples[:diagnostic_request_count]
    requests = _diagnostic_output_capped_requests(
        requests,
        diagnostic_request_count=diagnostic_request_count,
        diagnostic_output_token_cap=diagnostic_output_token_cap,
    )
    request_stream = RequestStream(requests)
    prefix_cache_evidence = _prefix_cache_reuse_evidence(
        tokenizer,
        samples,
        kv_block_size_tokens=server_plan["engine"]["kv_block_size_tokens"],
    )
    import vllm

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "dry_run" if dry_run else "running",
        "workload": {
            "kind": ("sharegpt_canonical_schedule" if diagnostic_request_count is None
                     else "sharegpt_diagnostic_prefix_schedule"),
            "request_count": len(requests),
            "expected_qps": case["runtime"]["target_qps"],
            "scheduled_duration_s": requests[-1].arrival_time_s,
            "seed": seed,
            "case_manifest_path": case["manifest_path"],
            "case_manifest_sha256": case["manifest_sha256"],
            "suite_manifest_path": case["suite_manifest_path"],
            "suite_manifest_sha256": case["suite_manifest_sha256"],
            "identity": case["identity"],
            "model_config_path": case["model_config_path"],
            "model_config_sha256": case["model_config_sha256"],
            "requests_path": requests_path,
            "requests_sha256": case["requests_sha256"],
            "request_stream_sha256": request_stream_sha256(request_stream),
            "dataset_path": str(Path(dataset).resolve(strict=True)),
            "dataset_sha256": sha256_file(dataset),
            "prompt_and_output_lengths_validated": True,
            "prefix_cache_reuse_evidence": prefix_cache_evidence,
            "request_template": case["request_template"],
        },
        "server": {
            "model_path": str(Path(model).resolve(strict=True)),
            "client_vllm_version": vllm.__version__,
            **server_plan,
        },
        "provenance": {
            **git_code_identity(ROOT),
            "argv": list(sys.argv),
        },
        "metrics": None,
    }
    if diagnostic_request_count is not None:
        result["workload"]["canonical_request_stream_sha256"] = (
            request_stream_sha256(canonical_stream))
        result["workload"]["canonical_request_count"] = len(canonical_requests)
    if diagnostic_output_token_cap is not None:
        result["workload"]["diagnostic_output_token_cap"] = (
            diagnostic_output_token_cap
        )
    if dry_run:
        return result
    result["metrics"] = asyncio.run(_run_http_benchmark(
        base_url=server_plan["endpoint"]["base_url"],
        served_model_name=case["identity"]["model_id"],
        samples=samples,
        requests=requests,
        require_request_id=server_plan["scheduler_trace"]["enabled"],
        tokenizer=tokenizer,
        max_model_len=server_plan["engine"]["max_model_len"],
        benchmark_boundary_callback=benchmark_boundary_callback,
        arrival_probe_per_token_s=(
            server_plan["arrival_probe_per_1000_tokens_ms"] / 1000.0 / 1000.0),
    ))
    max_dispatch_lag_ms = max(
        request["dispatch_lag_ms"]
        for request in result["metrics"]["per_request"]
    )
    result["metrics"]["max_dispatch_lag_ms"] = max_dispatch_lag_ms
    result["status"] = "success"
    return result
