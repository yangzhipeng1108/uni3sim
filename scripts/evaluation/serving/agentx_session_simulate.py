"""Thin CLI for AgentX corpus workload build and session simulation.

Subcommands:

* ``build``   — parse the raw ``traces.jsonl`` corpus, validate it against
  the pinned sha256, and write a deterministic, seeded workload JSON
  (schema ``agentx_session_workload/2``).
* ``replay``  — run the closed-loop session simulation against the pinned
  native vLLM 0.11 Scheduler backend (CPU only; no GPU pricing is
  performed here, kernel prediction is delegated to the common
  ``ServingExecutionContext``).

The ``replay`` subcommand is the Phase A short-E2E entry point.  It
supports:

* one or more live-session lanes (``--concurrency``);
* session rollover to spare traces when a trajectory ends
  (``--spares`` at build time);
* a per-run ``--turn-cap`` that closes each lane after the given number
  of turns (for short tests; default is unbounded);
* ``--think-cap-s`` mirroring the InferenceX driver's idle-gap cap
  (default 300 s).

It writes ``requests.jsonl``, ``summary.json`` and a metrics snapshot
under ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset.agentx_corpus import (
    BLOCK_TOKENS,
    MIN_KEPT_TURNS,
    build_workload,
    describe_workload,
    load_corpus,
    load_workload,
    save_workload,
)
from src.serving.request_gen.agentx_sessions import (
    AgentxSessionController,
    AgentxSessionControllerConfig,
)
from src.serving.request_gen.request_stream import RequestStream
from src.serving.schedule.single_pool_engine import run_single_pool_schedule
from src.serving.metrics.serving_metrics import compute_serving_metrics


# --------------------------------------------------------------------- build

def cmd_build(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.traces, max_model_len=args.max_model_len)
    workload = build_workload(
        corpus,
        seed=args.seed,
        lanes=args.lanes,
        spares=args.spares,
        start_min_ratio=args.start_min_ratio,
        start_max_ratio=args.start_max_ratio,
        think_cap_s=args.think_cap_s,
    )
    save_workload(workload, args.output)
    print(json.dumps({
        "output": args.output,
        "corpus_stats": workload.corpus_stats,
        "lanes": [
            {"lane": plan.lane, "trace": plan.trace_id,
             "start": plan.start_turn, "turns": plan.num_turns}
            for plan in workload.lanes
        ],
        "distribution": describe_workload(workload),
    }, indent=1))
    return 0


# -------------------------------------------------------------------- replay

def cmd_replay(args: argparse.Namespace) -> int:
    workload = load_workload(args.workload)
    if args.concurrency > len(workload.lanes):
        raise SystemExit(
            f"C={args.concurrency} > {len(workload.lanes)} planned lanes")
    max_model_len = min(workload.max_model_len, args.max_model_len)

    from src.serving.schedule.backend_protocol import BackendConfig

    # Load the model config to clamp synthetic token IDs to its vocabulary.
    from src.serving.workload.model_config import load_model_config
    model = load_model_config(args.model)
    if args.full_context:
        # Full-context replay: every recorded prompt that fits the served
        # max_model_len is replayed at its corpus length; turns whose
        # recorded in exceeds it replay as prompt-less stubs.  Requires a
        # deployment with enough KV capacity for the recorded context.
        max_prompt_tokens: Optional[int] = None
        max_output_tokens: Optional[int] = args.max_output_tokens
    else:
        max_prompt_tokens = max_model_len - args.max_output_tokens
        max_output_tokens = args.max_output_tokens
    if model.model_type == "glm_moe_dsa" and args.backend != "sglang":
        # The vLLM0.30 V32 BF16/FP8 GLM forward is only source-complete for
        # TP8 H200-class pools (validated scope); TP1 lacks its collective
        # program.  The AgentX short-E2E gate instead uses its fixture
        # fixture_qwen3_256k (TP1 PP1, 1-layer dense Qwen3-style model) as
        # the "complete supported model": every recorded turn is replayed
        # and priced through the same ServingExecutionContext path, and the
        # canonical-formula boundary is unchanged.  The 78-layer GLM-5.2
        # config remains supported through the standard evaluation
        # simulation entry with its declared TP8 scope.
        raise SystemExit(
            "agentx replay does not serve glm_moe_dsa; use the fixture "
            "model or the standard evaluation simulation entry for GLM-5.2")
    kv_bytes_per_token = model.kv_bytes_per_token_per_parallel_rank(1, 1)
    backend_config = BackendConfig(
        pool_name="main",
        role="colocated",
        max_batched_tokens=args.max_batched_tokens,
        max_requests=args.max_requests,
        num_kv_blocks=args.num_kv_blocks,
        kv_block_size_tokens=args.kv_block_size_tokens,
        kv_bytes_per_token=kv_bytes_per_token,
        scheduler_pipeline_depth=1,
    )
    native_scheduler = None
    backend = None
    try:
        if args.backend == "sglang":
            from src.serving.backend.sglang import SGLangBackend
            from src.serving.backend.sglang_native import (
                build_native_sglang_scheduler,
            )
            if args.sglang_mtp_num_speculative_tokens:
                # The dense0.11 pricing source used by this replay has no
                # MTP-enabled program: pricing speculative verify rows through
                # it would silently degrade them to ordinary decode
                # (root-forbidden).  Surface the combination as explicitly
                # unsupported before any scheduler construction.
                raise SystemExit(
                    "agentx replay --backend sglang does not support "
                    "--sglang-mtp-num-speculative-tokens with the dense0.11 "
                    "pricing source (no MTP-enabled program); the combination "
                    "is unsupported, not a partial run")
            native_scheduler = build_native_sglang_scheduler(
                max_total_num_tokens=(
                    args.num_kv_blocks * args.kv_block_size_tokens),
                max_running_requests=args.max_requests,
                chunked_prefill_size=args.max_batched_tokens,
                max_model_len=max_model_len,
                mtp_num_speculative_tokens=0,
                mtp_acceptance_assumption=None,
            )
            backend = SGLangBackend.from_scheduler(
                backend_config,
                native_scheduler,
                mtp_acceptance_assumption=None,
                price_verify_steps=True,
            )
        else:
            from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig
            scheduler_config = VLLMSchedulerConfig(
                args.model,
                max_model_len=max_model_len,
                enable_chunked_prefill=True,
                enable_prefix_caching=True,
                prefix_cache_mode="token_ids",
            )
            backend = VLLMBackend.from_model(backend_config, scheduler_config)
        controller = AgentxSessionController(
            AgentxSessionControllerConfig(
                workload=workload,
                concurrency=args.concurrency,
                nonce=args.nonce,
                think_cap_s=args.think_cap_s,
                max_turns=args.turn_cap,
                vocab_size=model.vocab_size,
                max_prompt_tokens=max_prompt_tokens,
                max_output_tokens=max_output_tokens,
                replay_subagents=args.replay_subagents,
            ),
        )
        initial = controller.initial_requests()
        print(f"[agentx] C={args.concurrency} lanes planned; "
              f"{len(initial)} initial requests; "
              f"think cap {args.think_cap_s}s; turn cap {args.turn_cap}",
              flush=True)
        if args.dry_run:
            for request in initial:
                print(f"  [lane] {request.request_id} in={request.input_tokens} "
                      f"out={request.output_tokens} t={request.arrival_time_s}")
            for lane in range(args.concurrency):
                chain_len = min(
                    args.turn_cap or 10**9,
                    len(controller._chain_order[lane]))
                for position in range(chain_len):
                    info = controller.lane_turn_info(lane, position)
                    print(f"  [plan] lane={lane} pos={position} {info}")
            return 0

        engine_requests = RequestStream(initial)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()

        prediction = {"method": args.prediction_method, "wall_s": None,
                      "phase_wall_s": None}
        if args.diagnostic_fixed_1s:
            # Explicit diagnostic mode: every step costs 1.0 s and the run
            # prices nothing.  Output is clearly labelled so it can never
            # be mistaken for a predicted result.
            prediction["method"] = "diagnostic_fixed_1s"
            def step_duration(step):
                return 1.0
        else:
            if args.prediction_method is None:
                raise SystemExit(
                    "replay requires --prediction-method roofline|midlayer; the "
                    "1.0 s placeholder clock is only available under the "
                    "explicit --diagnostic-fixed-1s flag")
            # Price every step through the common ServingExecutionContext
            # (dynamic graph lowering + predict against the HWSpec).  This
            # is the supported short-model path; the fixture model is a
            # real ModelConfig executed by the normal lowering/predict
            # chain, not a hand-priced stub.
            from src.kernel.hardware_spec import HWSpec
            from src.serving.deployment.deployment_config import (
                DeploymentConfig, WorkerPoolConfig)
            from src.serving.deployment.deployment_plan import (
                build_deployment_plan)
            from src.serving.execution import ServingExecutionContext
            from src.serving.parallelism.parallel_config import ParallelConfig
            from src.serving.workload.runtime_inputs import (
                ExpertRoutingInput, RuntimeInputs)
            hardware = HWSpec(args.hardware_config)
            pool = WorkerPoolConfig("main", "colocated", hardware.hardware_id,
                                    ParallelConfig(), kv_cache=None)
            deployment = build_deployment_plan(
                DeploymentConfig("colocated", (pool,)),
                num_layers=model.num_hidden_layers)
            if args.backend == "sglang":
                # The real-SGLang backend is a *scheduler* replay: the
                # pricing path keeps the ordinary validated dense forward
                # (same source the vLLM branch uses) so the run measures
                # scheduling differences, not a new model-source claim.
                # MTP (EAGLE) configurations are rejected before scheduler
                # construction (no MTP-enabled pricing program).
                runtime = RuntimeInputs(
                    ExpertRoutingInput.not_applicable(), "paged_varlen", "uniform",
                    model_execution_version="vllm_0_11_0",
                    target_sampling_source="vllm_0_11_plain_greedy_bf16",
                    symbolic_forward_source="cuda_eager_inplace")
            elif model.model_type == "glm_moe_dsa":
                # GLM-5 fixtures serve the native MLA KV layout through the
                # vLLM0.30 V32 BF16 virtual forward; the pinned qwen3-style
                # scheduler defaults target the 0.11 paged-MHA eager source.
                runtime = RuntimeInputs(
                    ExpertRoutingInput.seeded_assumption(
                        "operator_support_static_seeded_routing_v1", 0),
                    "mla", "uniform",
                    model_execution_version="vllm_0_30_0",
                    target_sampling_source="vllm_0_30_plain_greedy_bf16",
                    glm_indexer_selection_source="declared_causal_topk",
                    glm_prefill_attention_source="force_sparse_mqa",
                    symbolic_workspace_initial_capacity_bytes=(0,),
                    symbolic_forward_source=(
                        "vllm_0_30_v32_bf16_sync_explicit_allreduce"
                        if model.weight_dtype == "bf16"
                        else "vllm_0_30_v32_fp8_sync_explicit_allreduce"),
                    symbolic_source_max_model_len=max_model_len,
                    symbolic_source_max_batched_tokens=args.max_batched_tokens)
            else:
                runtime = RuntimeInputs(
                    ExpertRoutingInput.not_applicable(), "paged_varlen", "uniform",
                    model_execution_version="vllm_0_11_0",
                    target_sampling_source="vllm_0_11_plain_greedy_bf16",
                    symbolic_forward_source="cuda_eager_inplace")
            execution = ServingExecutionContext(
                model, deployment, {hardware.hardware_id: hardware},
                "analytical_launch", args.prediction_method, runtime,
                materialize_timeline_details=False)

            def step_duration(step):
                components = execution.execute_step(
                    step.schedule,
                    origin_time_s=0.0, device_floor_s=0.0,
                    kv_cache_events=step.kv_cache_events,
                    embedding_token_ids=step.embedding_token_ids,
                    model_source_inputs=step.model_source_inputs)
                return components.execution_timeline.duration_s

        # Closed-loop successor injection.  Both modes pass the controller
        # as the engine's closed_loop: the engine calls ``successors_of``
        # once per finished request and appends every returned successor
        # (concurrent siblings share the parent's finish instant) to its
        # closed-loop arrival queue, which admits each at its true arrival
        # instant.
        trace = run_single_pool_schedule(
            backend,
            engine_requests,
            step_duration,
            1,
            async_output_release="self",
            closed_loop=controller,
        )
        wall_s = time.perf_counter() - started
        if not args.diagnostic_fixed_1s:
            prediction["wall_s"] = wall_s
            stats = execution.performance_stats()
            phases = stats["phase_wall_time_s"]
            prediction["phase_wall_s"] = {
                name: phases[name] for name in (
                    "workload_construction", "kernel_lowering",
                    "kernel_prediction", "timeline_construction")
                if name in phases}
        # Token conservation closes over every *scheduled* request.  Turns
        # the engine never scheduled (e.g. a lane's final successors that
        # never entered the Scheduler because the run ended first) carry no
        # output and are dropped from the metrics view instead of failing
        # conservation; ``controller.completed_count``/``issued_count``
        # remain the authoritative turn bookkeeping.
        scheduled_ids = {
            request.request_id
            for entry in trace.entries
            for request in entry.backend_step.schedule.requests
        }
        metrics_stream = RequestStream(tuple(
            request for request in controller.metrics_stream
            if request.request_id in scheduled_ids))
        metrics = compute_serving_metrics(metrics_stream, trace, 0.0)
        with (out_dir / "requests.jsonl").open("w") as handle:
            for entry in trace.entries:
                for request in entry.backend_step.schedule.requests:
                    handle.write(json.dumps({
                        "step_id": entry.backend_step.schedule.step_id,
                        "request_id": request.request_id,
                        "phase": request.phase,
                        "input_tokens": request.input_tokens,
                        "output_tokens": request.output_tokens,
                        "computed_tokens": request.computed_tokens,
                        "content_prefix_hit_tokens": request.content_prefix_hit_tokens,
                        "start_time_s": entry.start_time_s,
                        "end_time_s": entry.end_time_s,
                    }) + "\n")
        summary = {
            "concurrency": args.concurrency,
            "workload": str(args.workload),
            "think_cap_s": args.think_cap_s,
            "turn_cap": args.turn_cap,
            "full_context": args.full_context,
            "replay_subagents": args.replay_subagents,
            "issued_turns": controller.issued_count,
            "completed_turns": controller.completed_count,
            "truncated_output_turns": controller.truncated_output_turns,
            "recorded_subagent_fanout_total": controller.subagent_fanout_total,
            "subagent_requests_total": controller.subagent_requests_total,
            "subagent_completed_total": controller.subagent_completed_total,
            "schedule_entries": len(trace.entries),
            "simulation_wall_s": wall_s,
            "prediction": prediction,
            "metrics": {
                "request_throughput_per_s": metrics.request_throughput_per_s,
                "output_token_throughput_per_s": metrics.output_token_throughput_per_s,
                "token_throughput_per_s": metrics.token_throughput_per_s,
            },
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=1))
    finally:
        try:
            if backend is not None:
                backend.close()
        finally:
            if native_scheduler is not None:
                from src.serving.backend.sglang_native import (
                    destroy_native_sglang_scheduler)
                destroy_native_sglang_scheduler(native_scheduler)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--traces", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--max-model-len", type=int, required=True)
    b.add_argument("--seed", type=int, required=True)
    b.add_argument("--lanes", type=int, required=True)
    b.add_argument("--spares", type=int, required=True)
    b.add_argument("--start-min-ratio", type=float, required=True)
    b.add_argument("--start-max-ratio", type=float, required=True)
    b.add_argument("--think-cap-s", type=float, required=True)
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("replay")
    r.add_argument("--workload", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--max-model-len", type=int, required=True)
    r.add_argument("--concurrency", type=int, required=True)
    r.add_argument("--nonce", required=True)
    r.add_argument("--think-cap-s", type=float, required=True)
    r.add_argument("--turn-cap", type=int, default=None)
    r.add_argument("--max-batched-tokens", type=int, required=True)
    r.add_argument("--max-requests", type=int, required=True)
    r.add_argument("--num-kv-blocks", type=int, required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--prediction-method", choices=("roofline", "midlayer"),
                   default=None,
                   help="price every step through ServingExecutionContext "
                        "with this predict method (required; the fixed "
                        "1.0 s clock is only available under "
                        "--diagnostic-fixed-1s)")
    r.add_argument("--diagnostic-fixed-1s", action="store_true",
                   help="explicit diagnostic mode: every step costs 1.0 s "
                        "and the run prices nothing; the summary labels "
                        "the clock so it cannot be mistaken for a "
                        "predicted result")
    r.add_argument("--hardware-config", default=str(ROOT / "configs/hardware/nvidia_a800.yaml"),
                   help="HWSpec yaml used by the prediction method")
    r.add_argument("--max-output-tokens", type=int, default=64,
                   help="upper bound on a turn's generated tokens reserved "
                        "against max_model_len (default 64)")
    r.add_argument("--kv-block-size-tokens", type=int, default=16,
                   help="KV cache block size in tokens (16 outside the "
                        "Qwen3 BF16 colocated TP1 PP1 block128 slice)")
    r.add_argument("--backend", choices=("vllm", "sglang"),
                   default="vllm",
                   help="serving backend: native vLLM 0.11 Scheduler or the "
                        "real SGLang 0.5.4.post2 Scheduler (Dummy+Patch "
                        "construction)")
    r.add_argument("--sglang-mtp-num-speculative-tokens", type=int, default=0,
                   help="SGLang EAGLE/MTP proposal depth (0=disabled)")
    r.add_argument("--sglang-mtp-acceptance-rate", default=None,
                   help="mean MTP acceptance rate in [0,1] (e.g. 0.55)")
    r.add_argument("--full-context", action="store_true",
                   help="replay every recorded prompt at its corpus length "
                        "(turns over the served max_model_len replay as "
                        "prompt-less stubs); requires enough KV capacity")
    r.add_argument("--replay-subagents", action="store_true",
                   help="execute recorded subagent (child) requests with "
                        "their real agent_id identities; children start at "
                        "the parent turn's finish instant and the lane "
                        "resumes after the last child (full fan-out)")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=cmd_replay)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
