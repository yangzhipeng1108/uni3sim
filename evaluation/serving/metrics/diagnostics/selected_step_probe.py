"""Run a deterministic pure-decode selected-step probe on vLLM hardware."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

from evaluation.serving.metrics.diagnostics.selected_step_injection import (
    materialize_selected_step_token_prefixes,
    write_json,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-step-manifest", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--canonical-ground-truth", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--target-following-decode-steps",
        type=int,
        default=1,
        help=(
            "number of decode Graph steps after the selected Graph step; the "
            "runner also executes one prefill sampling token before that step"
        ),
    )
    parser.add_argument("--cuda-profiler-api", action="store_true")
    parser.add_argument("--target-start-delay-seconds", type=float, default=0.0)
    parser.add_argument("--post-target-delay-seconds", type=float, default=0.0)
    parser.add_argument(
        "--token-prefixes-input",
        type=Path,
        help="reuse a previously materialized token-prefix artifact",
    )
    parser.add_argument(
        "--independent-eager",
        action="store_true",
        help=(
            "force eager execution while preserving the selected request/context "
            "workload; this is an independent replay, not the canonical Graph step"
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_requests(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("canonical request table is empty")
    required = {
        "Query Time Stamp [ms]",
        "In Token [tokens]",
        "Out Token [tokens]",
        "Cached hit [tokens]",
    }
    if set(rows[0]) != required:
        raise ValueError("canonical request table columns are not exact")
    return rows


def seed_inputs(
    runtime: dict[str, object], canonical_workload: dict[str, object]
) -> tuple[int, dict[str, int]]:
    """Keep dataset sampling identity separate from the vLLM engine seed."""

    dataset_seed = canonical_workload["seed"]
    if isinstance(dataset_seed, bool) or not isinstance(dataset_seed, int):
        raise ValueError("canonical workload seed must be an integer")
    runtime_seed = runtime["seed"]
    if runtime_seed is None:
        return dataset_seed, {}
    if isinstance(runtime_seed, bool) or not isinstance(runtime_seed, int):
        raise ValueError("runtime engine seed must be null or an integer")
    return dataset_seed, {"seed": runtime_seed}


def target_max_tokens(following_decode_steps: int) -> int:
    """Return output tokens needed for the selected Graph and its anchors.

    vLLM samples one output token in the initial prefill execution.  The
    selected pure-decode Graph is therefore the second token, so one following
    decode Graph needs three output tokens in total.
    """

    if isinstance(following_decode_steps, bool) or not isinstance(
        following_decode_steps, int
    ):
        raise TypeError("following decode steps must be an integer")
    if following_decode_steps < 0:
        raise ValueError("following decode steps must be non-negative")
    return 2 + following_decode_steps


def drain_inprocess_async_pipeline(llm: object) -> int:
    """Drain completed reconstruction work before formal request injection."""

    if not hasattr(llm, "llm_engine"):
        raise RuntimeError("selected-step runner requires an LLMEngine")
    client = llm.llm_engine.engine_core
    if not hasattr(client, "engine_core"):
        raise RuntimeError("selected-step runner requires an in-process EngineCore")
    core = client.engine_core
    if not hasattr(core, "batch_queue") or core.batch_queue is None:
        raise RuntimeError("selected-step runner requires the async batch queue")
    drained = 0
    while core.batch_queue:
        client.get_output()
        drained += 1
        if drained > 2:
            raise RuntimeError(
                "selected-step reconstruction pipeline exceeded depth 2"
            )
    if core.scheduler.has_requests():
        raise RuntimeError(
            "selected-step reconstruction left Scheduler requests active"
        )
    return drained


def _inprocess_core(llm: object) -> object:
    if not hasattr(llm, "llm_engine"):
        raise RuntimeError("selected-step runner requires an LLMEngine")
    client = llm.llm_engine.engine_core
    if not hasattr(client, "engine_core"):
        raise RuntimeError("selected-step runner requires an in-process EngineCore")
    return client.engine_core


def warm_c500_full_decode_graph(
    llm: object,
    capture_bucket_tokens: int,
    target_params: object,
    tokens_prompt_type: object,
) -> int:
    """Materialize the C500 FULL decode bucket before the formal step.

    The C500 vLLM runtime can JIT ``_compute_slot_mapping_kernel`` on the
    first inference for a decode-batch size even after engine Graph capture.
    That compilation must be outside the selected-step measurement.  The
    Scheduler tracer recognizes ``warmup-`` request IDs and records them only
    in its separate warmup evidence, never as formal trace steps.
    """

    if isinstance(capture_bucket_tokens, bool) or not isinstance(
        capture_bucket_tokens, int
    ):
        raise TypeError("C500 capture bucket must be an integer")
    if capture_bucket_tokens < 1:
        raise ValueError("C500 capture bucket must be positive")
    core = _inprocess_core(llm)
    if core.scheduler.has_requests():
        raise RuntimeError("C500 Graph warmup requires no active Scheduler requests")
    if core.batch_queue is None or core.batch_queue:
        raise RuntimeError("C500 Graph warmup requires an empty async batch queue")
    request_ids = {
        f"warmup-c500-full-decode-{index}"
        for index in range(capture_bucket_tokens)
    }
    for request_id in sorted(request_ids):
        llm.llm_engine.add_request(
            request_id,
            tokens_prompt_type(prompt_token_ids=[0]),
            target_params,
        )
    completed_request_ids = set()
    while llm.llm_engine.has_unfinished_requests():
        for output in llm.llm_engine.step():
            if output.finished:
                completed_request_ids.add(output.request_id)
    if completed_request_ids != request_ids:
        raise RuntimeError("C500 Graph warmup did not finish exactly its requests")
    drained = drain_inprocess_async_pipeline(llm)
    llm.reset_prefix_cache()
    return drained


def warm_c500_standalone_prefill(
    llm: object,
    prefill_token_count: int,
    target_params: object,
    tokens_prompt_type: object,
    request_id: str,
) -> int:
    """Compile C500's selected prefill shape outside its formal trace step."""

    if isinstance(prefill_token_count, bool) or not isinstance(
        prefill_token_count, int
    ):
        raise TypeError("C500 standalone prefill token count must be an integer")
    if prefill_token_count < 1:
        raise ValueError("C500 standalone prefill token count must be positive")
    if not isinstance(request_id, str) or not request_id.startswith("warmup-"):
        raise ValueError("C500 standalone prefill warmup ID must be reserved")
    core = _inprocess_core(llm)
    if core.scheduler.has_requests():
        raise RuntimeError("C500 prefill warmup requires no active Scheduler requests")
    if core.batch_queue is None or core.batch_queue:
        raise RuntimeError("C500 prefill warmup requires an empty async batch queue")
    llm.llm_engine.add_request(
        request_id,
        tokens_prompt_type(prompt_token_ids=[0] * prefill_token_count),
        target_params,
    )
    completed_request_ids = set()
    while llm.llm_engine.has_unfinished_requests():
        for output in llm.llm_engine.step():
            if output.finished:
                completed_request_ids.add(output.request_id)
    if completed_request_ids != {request_id}:
        raise RuntimeError("C500 standalone prefill warmup did not finish exactly once")
    drained = drain_inprocess_async_pipeline(llm)
    llm.reset_prefix_cache()
    return drained


def request_status_name(status: object) -> str:
    """Normalize the two explicit status representations accepted by vLLM."""

    if isinstance(status, str):
        name = status
    else:
        if not hasattr(status, "name"):
            raise TypeError("selected-step request status must expose a name")
        name = getattr(status, "name")
    if not isinstance(name, str) or not name:
        raise TypeError("selected-step request status name must be non-empty text")
    return name


def prepare_mixed_decode_state(
    llm: object,
    decode_requests: list[dict[str, object]],
    target_params: object,
    tokens_prompt_type: object,
) -> None:
    """Materialize only selected decode KV state, then pause admission."""

    core = _inprocess_core(llm)
    if core.batch_queue is None or core.batch_queue_size != 2:
        raise RuntimeError(
            "mixed selected-step runner requires an empty depth-2 batch queue"
        )
    if core.batch_queue:
        raise RuntimeError("mixed selected-step decode materialization queue is not empty")
    for request in decode_requests:
        llm.llm_engine.add_request(
            request["request_id"],
            tokens_prompt_type(
                prompt_token_ids=request["injection_prompt_token_ids"]
            ),
            target_params,
        )
    original_depth = core.batch_queue_size
    core.batch_queue_size = 1
    try:
        llm.llm_engine.step()
    finally:
        core.batch_queue_size = original_depth
    if core.batch_queue:
        raise RuntimeError(
            "mixed selected-step decode materialization left a pending batch"
        )
    expected_context = {
        request["request_id"]: request["computed_tokens_before_step"]
        for request in decode_requests
    }
    actual_requests = core.scheduler.requests
    if set(actual_requests) != set(expected_context):
        raise RuntimeError(
            "mixed selected-step decode materialization request set differs"
        )
    for request_id, expected_tokens in expected_context.items():
        request = actual_requests[request_id]
        status = request_status_name(request.status)
        if status != "RUNNING":
            raise RuntimeError(
                "mixed selected-step decode request is not RUNNING: "
                f"{request_id}={status}"
            )
        if request.num_computed_tokens != expected_tokens:
            raise RuntimeError(
                "mixed selected-step decode context differs: "
                f"{request_id}={request.num_computed_tokens}, "
                f"expected={expected_tokens}"
            )
    core.pause_scheduler(mode="keep", clear_cache=False)


def admit_mixed_prefill_requests(
    llm: object,
    prefill_requests: list[dict[str, object]],
    target_params: object,
    tokens_prompt_type: object,
) -> None:
    """Atomically add fresh prefills while selected decode state is paused."""

    core = _inprocess_core(llm)
    if not core.is_scheduler_paused():
        raise RuntimeError("mixed selected-step Scheduler was not paused")
    existing_ids = set(core.scheduler.requests)
    for request in prefill_requests:
        llm.llm_engine.add_request(
            request["request_id"],
            tokens_prompt_type(
                prompt_token_ids=request["injection_prompt_token_ids"]
            ),
            target_params,
        )
    expected_ids = existing_ids | {
        request["request_id"] for request in prefill_requests
    }
    if set(core.scheduler.requests) != expected_ids:
        raise RuntimeError("mixed selected-step atomic prefill request set differs")
    waiting_ids = {
        request_id
        for request_id, request in core.scheduler.requests.items()
        if request_status_name(request.status) == "WAITING"
    }
    if waiting_ids != {request["request_id"] for request in prefill_requests}:
        raise RuntimeError("mixed selected-step fresh prefill requests are not WAITING")
    core.resume_scheduler()


def require_environment(
    expected_request_count: int,
    configured_mode: str,
    hardware_id: str,
    expected_rank_count: int,
) -> None:
    expected = {
        "MIDLAYER_SCHEDULER_TRACE_REQUEST_COUNT": str(expected_request_count),
        "MIDLAYER_SCHEDULER_TRACE_CONFIGURED_CUDAGRAPH_MODE": configured_mode,
        "MIDLAYER_SCHEDULER_TRACE_MAX_IN_FLIGHT": "2",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
        "MIDLAYER_SCHEDULER_TRACE_HARDWARE_ID": hardware_id,
        "MIDLAYER_SCHEDULER_TRACE_WORKER_RANK_COUNT": str(expected_rank_count),
    }
    for key, value in expected.items():
        if key not in os.environ:
            raise RuntimeError(f"selected-step runner requires {key}={value}")
        if os.environ[key] != value:
            raise RuntimeError(
                f"selected-step runner environment mismatch: "
                f"{key}={os.environ[key]!r}, expected={value!r}"
            )
    if "MIDLAYER_SCHEDULER_TRACE_OUTPUT" not in os.environ:
        raise RuntimeError("selected-step runner requires a Scheduler trace output")
    if "MIDLAYER_SCHEDULER_TRACE_SOURCE_IDENTITY" not in os.environ:
        raise RuntimeError("selected-step runner requires a runtime source identity")
    if (
        "VLLM_ENABLE_V1_MULTIPROCESSING" not in os.environ
        or os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] != "0"
    ):
        raise RuntimeError(
            "selected-step runner requires in-process EngineCore so all target "
            "requests can be inserted before scheduling"
        )
    if hardware_id == "metax_c500":
        expected_metax = {
            "MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE": "0",
            "USE_VLLM_TRITON_EXPERT": "0",
        }
        for key, value in expected_metax.items():
            if key not in os.environ or os.environ[key] != value:
                raise RuntimeError(
                    f"selected-step C500 runner requires {key}={value}"
                )
    elif hardware_id == "ascend_910b":
        profiler_boundary_names = (
            "MIDLAYER_910B_MSPROF_STEP_TIMESTAMP",
            "MIDLAYER_910B_MSPROFTX_STEP_MARK",
        )
        enabled_boundaries = [
            name for name in profiler_boundary_names if name in os.environ
        ]
        if len(enabled_boundaries) != 1:
            raise RuntimeError(
                "selected-step 910B runner requires exactly one profiler "
                "step boundary mechanism"
            )
        name = enabled_boundaries[0]
        if os.environ[name] != "1":
            raise RuntimeError(f"selected-step 910B runner requires {name}=1")
    elif hardware_id != "nvidia_a800":
        raise ValueError(
            f"unsupported pure-decode selected-step hardware: {hardware_id!r}"
        )


def validate_target_graph_execution(
    hardware_id: str,
    runtime_mode: str,
    fallback_reason: str | None,
    *,
    mixed_target: bool,
    standalone_prefill_target: bool = False,
) -> None:
    if standalone_prefill_target:
        if (
            hardware_id not in {"ascend_910b", "metax_c500"}
            or runtime_mode != "NONE"
            or fallback_reason != "above_capture_limit"
        ):
            raise ValueError(
                "standalone prefill selected-step runner requires a supported "
                "above-capture-limit eager target"
            )
        return
    if not mixed_target:
        if runtime_mode != "FULL":
            raise ValueError(
                "pure-decode selected-step runner requires a FULL Graph target"
            )
        return
    if (
        hardware_id not in {"ascend_910b", "metax_c500"}
        or runtime_mode != "NONE"
        or fallback_reason != "above_capture_limit"
    ):
        raise ValueError(
            "mixed selected-step runner requires a supported "
            "above-capture-limit eager target"
        )


def main(
    *,
    scheduler_trace_pending_step_count_from_environment,
    finalize_scheduler_trace_from_environment,
) -> None:
    arguments = _arguments()
    selected_path = arguments.selected_step_manifest.resolve(strict=True)
    case_dir = arguments.case_dir.resolve(strict=True)
    ground_truth_path = arguments.canonical_ground_truth.resolve(strict=True)
    dataset_path = arguments.dataset.resolve(strict=True)
    model_path = arguments.model.resolve(strict=True)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    case_manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    fingerprint = selected["expected_fingerprint"]
    workload = fingerprint["request_workload"]
    decode_workload = [row for row in workload if row["decode_tokens"] == 1]
    prefill_workload = [row for row in workload if row["prefill_tokens"] > 0]
    if len(decode_workload) + len(prefill_workload) != len(workload):
        raise ValueError(
            "selected-step runner accepts one-token decode and prefill requests"
        )
    if any(row["prefill_tokens"] != 0 for row in decode_workload):
        raise ValueError("one selected request cannot mix prefill and decode tokens")
    if any(
        row["decode_tokens"] != 0
        or row["computed_tokens_before_step"] != 0
        or row["scheduled_tokens"] != row["prefill_tokens"]
        for row in prefill_workload
    ):
        raise ValueError(
            "selected-step runner requires each prefill request to be fresh"
        )
    mixed_target = bool(decode_workload and prefill_workload)
    standalone_prefill_target = bool(prefill_workload) and not bool(decode_workload)
    graph = fingerprint["graph_execution"]
    if arguments.independent_eager and mixed_target:
        raise ValueError("independent eager selected-step replay requires pure decode")
    if not arguments.independent_eager:
        validate_target_graph_execution(
            fingerprint["hardware_id"],
            graph["runtime_mode"],
            graph["fallback_reason"],
            mixed_target=mixed_target,
            standalone_prefill_target=standalone_prefill_target,
        )
    minimum_following_steps = 0 if arguments.independent_eager else 1
    if arguments.target_following_decode_steps < minimum_following_steps:
        raise ValueError(
            "target following decode steps are below the selected execution minimum"
        )
    if arguments.target_start_delay_seconds < 0:
        raise ValueError("target start delay seconds must be non-negative")
    if arguments.post_target_delay_seconds < 0:
        raise ValueError("post-target delay seconds must be non-negative")
    hardware_id = fingerprint["hardware_id"]
    require_environment(
        len(workload),
        "NONE" if arguments.independent_eager else graph["configured_mode"],
        hardware_id,
        len(graph["rank_ids"]),
    )

    runtime = case_manifest["runtime"]
    identity = case_manifest["identity"]
    if runtime["pipeline_parallel_size"] != 1:
        raise ValueError("selected-step runner requires PP=1")
    if runtime["capture_enable_prefix_caching"] is not True:
        raise ValueError("canonical selected case did not enable prefix caching")
    canonical_workload = ground_truth["workload"]
    if _sha256(dataset_path) != canonical_workload["dataset_sha256"]:
        raise ValueError("dataset SHA256 disagrees with canonical ground truth")
    if _sha256(model_path / "config.json") != canonical_workload["model_config_sha256"]:
        raise ValueError("model config SHA256 disagrees with canonical ground truth")
    dataset_sampling_seed, engine_seed_kwargs = seed_inputs(
        runtime, canonical_workload
    )

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.benchmarks.datasets import ShareGPTDataset
    from vllm.inputs import TokensPrompt

    request_rows = _load_requests(case_dir / "requests.csv")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    samples = ShareGPTDataset(
        dataset_path=str(dataset_path), random_seed=dataset_sampling_seed
    ).sample(tokenizer, num_requests=len(request_rows), output_len=None)
    prompt_token_sequences = [
        list(tokenizer(sample.prompt).input_ids) for sample in samples
    ]
    expected_prompt_hash = canonical_workload[
        "prefix_cache_reuse_evidence"
    ]["prompt_token_stream_sha256"]
    actual_prompt_hash = _canonical_sha256(prompt_token_sequences)
    if actual_prompt_hash != expected_prompt_hash:
        raise ValueError(
            "reconstructed prompt token stream disagrees with canonical ground truth"
        )
    for index, (tokens, row) in enumerate(zip(prompt_token_sequences, request_rows)):
        if len(tokens) != int(row["In Token [tokens]"]):
            raise ValueError(f"request {index} prompt token count mismatch")

    generation_prompts = []
    generation_params = []
    prompt_by_request = {}
    generated_by_request = {row["request_idx"]: [] for row in decode_workload}
    for row in decode_workload:
        request_idx = row["request_idx"]
        prompt_tokens = prompt_token_sequences[request_idx]
        generated_count = row["computed_tokens_before_step"] - len(prompt_tokens)
        if generated_count < 0:
            raise ValueError(f"request {request_idx} decode context predates its prompt")
        if generated_count > int(request_rows[request_idx]["Out Token [tokens]"]):
            raise ValueError(
                f"request {request_idx} target exceeds canonical output length"
            )
        prompt_by_request[request_idx] = prompt_tokens
        if generated_count:
            generation_prompts.append(TokensPrompt(prompt_token_ids=prompt_tokens))
            generation_params.append(SamplingParams(
                temperature=0.0,
                repetition_penalty=1.0,
                max_tokens=generated_count,
                ignore_eos=True,
            ))

    total_context_tokens = sum(
        row["computed_tokens_before_step"] for row in workload
    )
    if hardware_id == "nvidia_a800":
        compilation_config = {
            "level": 3,
            "cudagraph_mode": fingerprint["graph_execution"]["configured_mode"],
        }
        platform_kwargs = {}
    elif hardware_id == "metax_c500":
        compilation_config = {"mode": 3}
        platform_kwargs = {
            "cudagraph_metrics": True,
            "safetensors_load_strategy": "prefetch",
            "safetensors_prefetch_num_threads": 1,
        }
    elif hardware_id == "ascend_910b":
        compilation_config = {
            "cudagraph_mode": graph["configured_mode"],
            "max_cudagraph_capture_size": graph["capture_limit_tokens"],
        }
        platform_kwargs = {"cudagraph_metrics": True}
    else:
        raise AssertionError("selected-step hardware was validated")

    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=runtime["tensor_parallel_size"],
        pipeline_parallel_size=runtime["pipeline_parallel_size"],
        dtype=runtime["dtype"],
        gpu_memory_utilization=arguments.gpu_memory_utilization,
        max_model_len=8192,
        block_size=16,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        max_num_batched_tokens=max(8192, total_context_tokens),
        # Keep the canonical scheduler/Graph capacity.  Reducing this to the
        # selected request count changes vLLM's capture-size set (for example,
        # 66 requests caps A800 at 128 instead of the canonical 512).
        max_num_seqs=256,
        async_scheduling=True,
        enforce_eager=arguments.independent_eager,
        compilation_config=compilation_config,
        enable_expert_parallel=runtime["expert_parallel_size"] > 1,
        **engine_seed_kwargs,
        **platform_kwargs,
    )
    if generation_prompts and arguments.token_prefixes_input is None:
        generated = llm.generate(
            generation_prompts, generation_params, use_tqdm=True
        )
        generated_indexes = [
            row["request_idx"]
            for row in decode_workload
            if row["computed_tokens_before_step"]
            > len(prompt_by_request[row["request_idx"]])
        ]
        for request_idx, output in zip(generated_indexes, generated, strict=True):
            generated_by_request[request_idx] = list(output.outputs[0].token_ids)
    reconstruction_drained_pipeline_steps = drain_inprocess_async_pipeline(llm)
    if arguments.token_prefixes_input is None:
        content = materialize_selected_step_token_prefixes(
            selected,
            {
                row["request_idx"]: prompt_token_sequences[row["request_idx"]]
                for row in workload
            },
            generated_by_request,
            {
                "method": "greedy_reconstruction_same_model_and_prompt_stream",
                "canonical_prompt_token_stream_sha256": expected_prompt_hash,
                "dataset_path": str(dataset_path),
                "dataset_sha256": _sha256(dataset_path),
                "model_path": str(model_path),
                "model_config_sha256": _sha256(model_path / "config.json"),
                "temperature": 0.0,
                "ignore_eos": True,
                "historical_output_token_ids_observed": False,
            },
        )
    else:
        token_prefix_path = arguments.token_prefixes_input.resolve(strict=True)
        content = json.loads(token_prefix_path.read_text(encoding="utf-8"))
        if content["kind"] != "selected_step_token_prefixes":
            raise ValueError("token-prefix input has the wrong kind")
        if content["cell_id"] != selected["cell_id"]:
            raise ValueError("token-prefix input has the wrong cell identity")
        if content["expected_step_fingerprint_sha256"] != _canonical_sha256(
            fingerprint
        ):
            raise ValueError("token-prefix input has the wrong step fingerprint")
        content_indexes = {
            row["request_idx"] for row in content["requests"]
        }
        if content_indexes != {row["request_idx"] for row in workload}:
            raise ValueError("token-prefix input has the wrong request coverage")
    llm.reset_prefix_cache()

    target_params = SamplingParams(
        temperature=0.0,
        repetition_penalty=1.0,
        max_tokens=target_max_tokens(arguments.target_following_decode_steps),
        ignore_eos=True,
    )
    c500_graph_warmup_drained_pipeline_steps = 0
    if (
        hardware_id == "metax_c500"
        and not arguments.independent_eager
        and graph["capture_bucket_tokens"] is not None
    ):
        c500_graph_warmup_drained_pipeline_steps = warm_c500_full_decode_graph(
            llm,
            graph["capture_bucket_tokens"],
            target_params,
            TokensPrompt,
        )
    c500_standalone_prefill_warmup_drained_pipeline_steps = 0
    if hardware_id == "metax_c500" and standalone_prefill_target:
        for index, request in enumerate(prefill_workload):
            c500_standalone_prefill_warmup_drained_pipeline_steps += (
                warm_c500_standalone_prefill(
                    llm,
                    request["prefill_tokens"],
                    target_params,
                    TokensPrompt,
                    f"warmup-c500-standalone-prefill-{index}",
                )
            )
    state_materialization_steps = 0
    if mixed_target:
        content_by_index = {
            request["request_idx"]: request for request in content["requests"]
        }
        decode_requests = [
            content_by_index[row["request_idx"]] for row in decode_workload
        ]
        prefill_requests = [
            content_by_index[row["request_idx"]] for row in prefill_workload
        ]
        prepare_mixed_decode_state(
            llm, decode_requests, target_params, TokensPrompt
        )
        state_materialization_steps = 1
        admit_mixed_prefill_requests(
            llm, prefill_requests, target_params, TokensPrompt
        )
    else:
        for request in content["requests"]:
            llm.llm_engine.add_request(
                request["request_id"],
                TokensPrompt(
                    prompt_token_ids=request["injection_prompt_token_ids"]
                ),
                target_params,
            )
    if arguments.target_start_delay_seconds:
        time.sleep(arguments.target_start_delay_seconds)
    if arguments.cuda_profiler_api:
        import torch

        torch.cuda.cudart().cudaProfilerStart()
    target_started_ns = time.perf_counter_ns()
    completed_request_ids = []
    try:
        while llm.llm_engine.has_unfinished_requests():
            for output in llm.llm_engine.step():
                if output.finished:
                    completed_request_ids.append(output.request_id)
        target_ended_ns = time.perf_counter_ns()
        if set(completed_request_ids) != {row["request_id"] for row in workload}:
            raise RuntimeError(
                "target injection did not complete exactly the selected requests"
            )
        drained_pipeline_steps = 0
        while scheduler_trace_pending_step_count_from_environment() > 0:
            # async_scheduling can report no unfinished frontend requests while
            # one depth-2 batch is already submitted.  Consume that EngineCore
            # result without handing its stale output to the finished frontend.
            llm.llm_engine.engine_core.get_output()
            drained_pipeline_steps += 1
            if drained_pipeline_steps > 2:
                raise RuntimeError(
                    "selected-step Scheduler trace did not drain within depth 2"
                )
        if not finalize_scheduler_trace_from_environment():
            raise RuntimeError("selected-step Scheduler trace was not enabled")
    finally:
        if arguments.cuda_profiler_api:
            torch.cuda.cudart().cudaProfilerStop()
    scheduler_trace_path = Path(
        os.environ["MIDLAYER_SCHEDULER_TRACE_OUTPUT"]
    ).resolve()
    if not scheduler_trace_path.is_file():
        raise RuntimeError("selected-step Scheduler trace was not sealed")

    content["execution"] = {
        "role": (
            "independent_eager_selected_state_injection"
            if arguments.independent_eager
            else "deterministic_selected_state_injection"
        ),
        "target_wall_envelope_ms": (target_ended_ns - target_started_ns) / 1_000_000,
        "following_decode_steps": arguments.target_following_decode_steps,
        "reconstruction_drained_pipeline_steps": (
            reconstruction_drained_pipeline_steps
        ),
        "c500_graph_warmup_drained_pipeline_steps": (
            c500_graph_warmup_drained_pipeline_steps
        ),
        "c500_standalone_prefill_warmup_drained_pipeline_steps": (
            c500_standalone_prefill_warmup_drained_pipeline_steps
        ),
        "state_materialization_steps": state_materialization_steps,
        "drained_pipeline_steps": drained_pipeline_steps,
        "scheduler_trace": str(scheduler_trace_path),
        "configured_graph_mode": (
            "NONE"
            if arguments.independent_eager
            else fingerprint["graph_execution"]["configured_mode"]
        ),
        "canonical_graph_mode": fingerprint["graph_execution"]["configured_mode"],
        "expected_runtime_graph_mode": fingerprint["graph_execution"]["runtime_mode"],
        "expected_capture_bucket_tokens": fingerprint[
            "graph_execution"
        ]["capture_bucket_tokens"],
        "historical_canonical_output_token_identity_confirmed": False,
        "comparison_status": (
            "independent_eager_replay_requires_workload_identity_audit"
            if arguments.independent_eager
            else "requires_post_run_trace_and_content_identity_audit"
        ),
    }
    write_json(arguments.output.resolve(), content)
    if arguments.post_target_delay_seconds:
        time.sleep(arguments.post_target_delay_seconds)
    llm.llm_engine.engine_core.shutdown()
