#!/usr/bin/env python3
"""Emit static full-model Roofline operator facts for source comparison."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.simulation.model_operator_probe import (
    write_model_operator_probe, write_glm_gpu_v2_storage_probe,
    require_logical_roofline_forward, build_model_operator_probe_source_inputs,
    normalize_model_operator_probe_source)
from evaluation.serving.simulation.simulation_configuration import decode_typed_step_source_json
from src.serving.workload.runtime_inputs import V32PaddedMTPInputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-link", choices=("nvlink",))
    parser.add_argument("--pipeline-layer-partition")
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--mtp-mode", choices=("off", "native"), default="off")
    parser.add_argument("--target-sampling-source", choices=("vllm_0_30_plain_sampler_bf16", "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_greedy_fp16",
                        "vllm_0_30_rejection_raw_logits_bf16", "vllm_0_27_1_plain_greedy_bf16",
                        "vllm_0_11_plain_greedy_bf16", "vllm_0_11_plain_sampler_bf16"),
                        help="ordinary/rejection sampler typed materializations; full sampling/provider remain unbound")
    parser.add_argument("--mtp-draft-sampling-source", choices=("vllm_0_30_legacy_full_logits_greedy",),
                        help="explicit source greedy argmax/INT32 feedback; not numerical model or runtime binding")
    parser.add_argument("--prefill-tokens", type=int, default=8,
                        help="prompt length before continuous decode (127 exercises the C128 boundary)")
    parser.add_argument("--source-max-model-len", type=int)
    parser.add_argument("--source-max-batched-tokens", type=int)
    parser.add_argument("--source-max-requests", type=int)
    parser.add_argument("--glm-mtp-phase-source-input", action="append", nargs=2, metavar=("PHASE", "JSON"),
        help="repeat explicit V32PaddedMTPInputs JSON per phase; every field required, no inferred draft IDs or acceptance")
    parser.add_argument("--source-workspace-initial-capacity-bytes", type=int, nargs="+")
    parser.add_argument("--glm-indexer-selection-source", choices=("declared_causal_topk",))
    parser.add_argument("--glm-prefill-attention-source", choices=("force_sparse_mqa",))
    token_inputs = parser.add_mutually_exclusive_group()
    token_inputs.add_argument("--embedding-token-ids", type=Path,
                        help="JSON mapping of phases to IDs in the declared model-input order; V3 mixed is decode before prefill; not runtime observations")
    token_inputs.add_argument("--synthetic-input-token-ids", action="store_true",
                        help="explicit synthetic workload: sequential IDs for prefill/decode/mixed; no checkpoint or sampled tokens")
    parser.add_argument("--v4-eager-metadata", action="store_true",
                        help="explicit source-only eager dict/all-valid V4 metadata scenario; not runtime observation")
    parser.add_argument("--v4-source-max-model-len", type=int,
                        help="explicit source metadata constructor cap; not an observed runtime value")
    parser.add_argument("--v4-source-max-batched-tokens", type=int,
                        help="explicit batch constructor cap for the shared virtual forward")
    parser.add_argument("--v4-indexer-selection-source", choices=("declared_causal_topk",),
                        help="explicit cardinality domain for scored V4 indexer inputs")
    parser.add_argument("--v4-routing-source", choices=("vllm_0_30_triton_top6_pure_text",
                      "vllm_0_30_modular_cuda_top6_pure_text"),
                        help="conditional CUDA routing source; modular scenario also includes INT32 hash routing")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-only", action="store_true",
                        help="write compact phase latencies instead of full operator details; requires all four phases priced")
    parser.add_argument("--require-logical-roofline-forward", action="store_true",
                        help="fail unless all four declared forward phases have complete prices and DAG timelines; not native/runtime acceptance")
    parser.add_argument("--gpu-v2-storage-only", action="store_true",
                        help="unpriced GLM source storage subset; normal GPU-v2 workload stays unsupported")
    parser.add_argument("--source-graph-only", action="store_true",
                        help="V4 provenance nodes/edges only; skips lowering, launch selection and prediction")
    parser.add_argument("--prefill-context-parallel-size", type=int)
    parser.add_argument("--v4-source-use-deep-gemm", action=argparse.BooleanOptionalAction, default=None,
                        help="declared process-lifetime VLLM_USE_DEEP_GEMM before cached dispatch; not observed provider support")
    args = parser.parse_args()
    model_path = Path(args.model)
    args.model = str(model_path if model_path.is_absolute() else ROOT / model_path)
    args.symbolic_forward_source, source_normalization = normalize_model_operator_probe_source(
        args.model, args.target_sampling_source)
    ordinary_source_options = (args.source_max_model_len, args.source_max_batched_tokens,
        args.source_workspace_initial_capacity_bytes, args.glm_indexer_selection_source, args.glm_prefill_attention_source,
        args.source_max_requests, args.glm_mtp_phase_source_input)
    if args.symbolic_forward_source is None and any(value is not None for value in ordinary_source_options):
        raise ValueError("ordinary source options require a matching explicit versioned target sampling source")
    glm_pipeline_source = (args.symbolic_forward_source in (
        "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
        and args.tensor_parallel_size == 8 and args.pipeline_parallel_size == 2
        and not args.enable_expert_parallel and args.pipeline_link == "nvlink"
        and args.pipeline_layer_partition == "38,40"
        and args.source_workspace_initial_capacity_bytes is not None
        and len(args.source_workspace_initial_capacity_bytes) == 16)
    glm_native_source = (args.mtp_mode == "native" and args.glm_mtp_phase_source_input is not None
        and args.symbolic_forward_source in ("vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
        and args.pipeline_parallel_size == 1 and not args.enable_expert_parallel)
    if args.symbolic_forward_source is not None and ((args.mtp_mode != "off" and not glm_native_source)
            or (args.pipeline_parallel_size != 1 and not glm_pipeline_source)
            or args.gpu_v2_storage_only or args.source_graph_only or args.v4_eager_metadata
            or args.v4_routing_source is not None or args.v4_source_max_model_len is not None
            or args.v4_source_max_batched_tokens is not None or args.v4_source_use_deep_gemm is not None
            or args.v4_indexer_selection_source is not None):
        raise ValueError("ordinary shared source requires target-off PP1/declared GLM TP8 PP2 or explicit GLM native PP1 typed phase inputs; no V4/storage-only options")
    if args.glm_mtp_phase_source_input is not None and not glm_native_source:
        raise ValueError("GLM MTP phase inputs require an explicit native GLM source")
    if args.v4_indexer_selection_source is not None and args.v4_source_max_batched_tokens is None:
        raise ValueError("indexer source requires the explicit virtual batch constructor cap")
    if args.v4_source_max_batched_tokens is not None and (
            args.gpu_v2_storage_only or args.mtp_mode != "off" or args.pipeline_parallel_size != 1
            or not args.v4_eager_metadata or args.v4_source_max_model_len is None
            or args.v4_source_use_deep_gemm is None
            or args.v4_routing_source != "vllm_0_30_modular_cuda_top6_pure_text"
            or args.target_sampling_source != "vllm_0_30_plain_greedy_bf16"):
        raise ValueError("virtual probe requires target-off PP1 and explicit eager metadata, constructor caps, modular routing, DeepGEMM branch and greedy source")
    if args.synthetic_input_token_ids and (args.mtp_mode != "off" or args.gpu_v2_storage_only):
        raise ValueError("synthetic input IDs require the target MTP-off profile")
    if (args.require_logical_roofline_forward or args.summary_only) and (args.gpu_v2_storage_only or args.source_graph_only):
        raise ValueError("complete logical forward check cannot accept storage or source-graph-only output")
    if args.gpu_v2_storage_only:
        if (args.source_graph_only or args.mtp_mode != "native" or args.enable_expert_parallel
                or args.mtp_draft_sampling_source is not None or args.target_sampling_source is not None
                or args.v4_routing_source is not None or args.v4_eager_metadata or args.v4_source_use_deep_gemm is not None
                or args.prefill_context_parallel_size is None
                or args.pipeline_parallel_size != 1 or args.pipeline_link is not None
                or args.pipeline_layer_partition is not None or args.embedding_token_ids is not None):
            raise ValueError("GPU-v2 storage source probe requires native, explicit PCP, EP-off and no V4 flag")
        write_glm_gpu_v2_storage_probe(args.model, args.tensor_parallel_size, args.output,
            prefill_context_parallel_size=args.prefill_context_parallel_size)
        print(args.output)
        return
    if args.prefill_context_parallel_size is not None:
        raise ValueError("explicit PCP applies only to the GPU-v2 storage source probe")
    tokens = None
    if args.embedding_token_ids is not None:
        raw_tokens = json.loads(args.embedding_token_ids.read_text(encoding="utf-8"))
        if not isinstance(raw_tokens, dict) or any(not isinstance(values, list) for values in raw_tokens.values()):
            raise ValueError("embedding token input must map phases to token-ID arrays")
        tokens = {phase: tuple(values) for phase, values in raw_tokens.items()}
    elif args.synthetic_input_token_ids:
        count = args.prefill_tokens
        tokens = {"prefill": tuple(range(count)), "decode1": (count,), "decode2": (count + 1,),
                  "mixed": tuple(range(count + 2, 2 * count + 3))}
    source_inputs = None
    if args.symbolic_forward_source is not None:
        if args.symbolic_forward_source == "cuda_eager_inplace" and tokens is None:
            # Four known shape-only input ports, with no dummy token values.
            # The shared model owner still rejects every value-dependent
            # model/parallel/routing domain for these unknown IDs.
            tokens = {phase: None for phase in ("prefill", "decode1", "decode2", "mixed")}
        mtp_inputs = None
        if args.glm_mtp_phase_source_input is not None:
            mtp_inputs = {}
            for phase, encoded in args.glm_mtp_phase_source_input:
                if phase in mtp_inputs:
                    raise ValueError("GLM MTP source phase names must be unique")
                mtp_inputs[phase] = decode_typed_step_source_json(encoded, V32PaddedMTPInputs)
        source_inputs = build_model_operator_probe_source_inputs(symbolic_forward_source=args.symbolic_forward_source,
            embedding_token_ids_by_phase=tokens, source_max_model_len=args.source_max_model_len,
            source_max_batched_tokens=args.source_max_batched_tokens, target_sampling_source=args.target_sampling_source,
            workspace_initial_capacity_bytes=(None if args.source_workspace_initial_capacity_bytes is None else
                tuple(args.source_workspace_initial_capacity_bytes)), glm_indexer_selection_source=args.glm_indexer_selection_source,
            glm_prefill_attention_source=args.glm_prefill_attention_source, source_max_requests=args.source_max_requests,
            mtp_draft_sampling_source=args.mtp_draft_sampling_source, model_source_inputs_by_phase=mtp_inputs)
    elif args.v4_source_max_batched_tokens is not None:
        if tokens is None:
            raise ValueError("virtual probe requires explicit or declared synthetic input token IDs")
    write_model_operator_probe(
        args.model, args.tensor_parallel_size, args.enable_expert_parallel,
        args.output, mtp_mode=args.mtp_mode, prefill_tokens=args.prefill_tokens,
        v4_eager_metadata=args.v4_eager_metadata if source_inputs is None else False,
        v4_routing_source=args.v4_routing_source if source_inputs is None else None,
        v4_source_max_model_len=args.v4_source_max_model_len if source_inputs is None else None,
        v4_source_max_batched_tokens=args.v4_source_max_batched_tokens if source_inputs is None else None,
        v4_indexer_selection_source=args.v4_indexer_selection_source if source_inputs is None else None,
        v4_source_use_deep_gemm=args.v4_source_use_deep_gemm if source_inputs is None else None,
        pipeline_parallel_size=args.pipeline_parallel_size,
        pipeline_link=args.pipeline_link, pipeline_layer_partition=args.pipeline_layer_partition,
        source_graph_only=args.source_graph_only,
        mtp_draft_sampling_source=args.mtp_draft_sampling_source if source_inputs is None else None,
        target_sampling_source=args.target_sampling_source if source_inputs is None else None,
        embedding_token_ids_by_phase=tokens if source_inputs is None else None,
        source_runtime_inputs_by_phase=source_inputs,
        source_normalization=source_normalization,
        summary_only=args.summary_only)
    if args.require_logical_roofline_forward and not args.summary_only:
        require_logical_roofline_forward(json.loads(args.output.read_text(encoding="utf-8")))
    print(args.output)


if __name__ == "__main__":
    main()
