#!/usr/bin/env python3
"""Report a metadata-only full model graph, live storage and Roofline costs."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.metrics.symbolic_model import write_symbolic_model_report, write_symbolic_serving_report
from src.serving.workload.runtime_inputs import ExpertRoutingInput


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model")
    source.add_argument("--serving-result", type=Path)
    source.add_argument("--serving-matrix-output", type=Path)
    parser.add_argument("--hardware-config", type=Path, default=ROOT / "configs/hardware/nvidia_h200.yaml",
        help="target HWSpec for the static model report; matrix hardware comes from each frozen case")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=0,
        help="parallel Serving case workers; 0 uses one worker per selected case (default)")
    parser.add_argument("--serving-timing-mode", choices=("device_only", "device_plus_launch",
        "device_plus_launch_plus_scheduler"), default="device_only")
    parser.add_argument("--serving-execution-mode", choices=("accelerated", "fast"),
        default="accelerated")
    parser.add_argument("--serving-execution-workers", type=int, default=0,
        help="per-case streaming tail workers; 0 selects the CPU-aware default")
    parser.add_argument("--serving-state-audit", action="store_true")
    parser.add_argument("--h200-serving-scenario", choices=("analytical_bf16_v1",),
        help="explicit H200 analytical BF16/Host scenario with declared TP2 zero-ID locality approximation")
    parser.add_argument("--serving-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--vllm-sdk-python", type=Path)
    parser.add_argument("--vllm-source-root", type=Path)
    parser.add_argument("--h200-capture-version", choices=("0.30.0", "none"), default="0.30.0")
    parser.add_argument("--reuse-completed-virtual", action="store_true",
        help="rebuild only metrics/report from exact-command completed virtual CLI results")
    parser.add_argument("--query-lengths", nargs="+", type=int)
    parser.add_argument("--context-lengths", nargs="+", type=int)
    parser.add_argument("--upstream-vllm-root", type=Path)
    parser.add_argument("--torch-include", type=Path,
        help="explicit Torch2.11 source headers for V4 argmax semantics; does not select a runtime/provider")
    parser.add_argument("--flashinfer-source-root", type=Path,
        help="genuine local SDK source for conditional FP8 quantizer programs; never loads a provider")
    parser.add_argument("--symbolic-forward-source", choices=("cuda_eager_inplace", "vllm_ascend_eager", "vllm_0_30_v4_sync_fp8_ds_mla",
        "vllm_0_11_glm4_sync",
        "vllm_0_27_1_deepseek_v3_sync_fp8",
        "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce",
        "vllm_0_30_k2_6_int4_sync_explicit_allreduce", "vllm_0_30_k3_triton_marlin_bf16"))
    parser.add_argument("--k3-state-indices", nargs="+", type=int,
        help="explicit distinct positive KDA pool slots in request order; not native Scheduler observations")
    parser.add_argument("--k3-state-pool-slots", type=int)
    parser.add_argument("--k3-prefill-output-source", choices=("supports_out",),
        help="explicit source backend capability, without selecting a real provider")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--expert-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-batched-tokens", type=int)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--source-block-size-tokens", type=int)
    parser.add_argument("--mtp-draft-sampling-source", choices=("vllm_0_27_1_eager_full_logits_greedy",))
    parser.add_argument("--mtp-verified-tokens", type=int)
    parser.add_argument("--mtp-accepted-tokens", type=int)
    parser.add_argument("--mtp-proposer-tokens", type=int)
    parser.add_argument("--mtp-acceptance-source-id")
    parser.add_argument("--embedding-token-ids", nargs="+", type=int,
        help="explicit target IDs for the V3 MTP source scenario")
    parser.add_argument("--v3-mtp-draft-input-token-ids", nargs="+", type=int, action="append",
        help="independent declared IDs for one draft pass; repeat for both passes, including padding state")
    parser.add_argument("--v3-mtp-attention-source", choices=("vllm_0_27_1_causal_triton_mla_source",
        "vllm_0_27_1_causal_flashmla_bf16_source"))
    parser.add_argument("--v3-mtp-batch-invariant", choices=("enabled", "disabled"),
        help="explicit source VLLM_BATCH_INVARIANT value; FlashMLA currently requires disabled")
    parser.add_argument("--v4-source-use-deep-gemm", choices=("enabled", "disabled"))
    parser.add_argument("--v4-indexer-selection-source", choices=("declared_causal_topk",),
        help="explicit min(K, causal completed states) distinct valid selected indices per row; not observed values")
    parser.add_argument("--glm-indexer-selection-source", choices=("declared_causal_topk",),
        help="explicit min(K, uncompressed causal context) distinct valid indices per GLM decode row")
    parser.add_argument("--glm-prefill-attention-source", choices=("force_sparse_mqa",),
        help="explicit vLLM attention_config.sparse_mla_force_mqa=True source branch for GLM prefill/mixed")
    parser.add_argument("--target-sampling-source", choices=("vllm_0_30_plain_greedy_bf16", "vllm_0_11_plain_greedy_bf16",
                        "vllm_0_27_1_plain_greedy_bf16", "vllm_0_27_1_greedy_rejection_bf16"),
        help="include the matching source's plain greedy chain; omission retains that source's declared unsampled scope")
    parser.add_argument("--source-workspace-initial-capacity-bytes", nargs="+", type=int,
        help="explicit workspace entry capacity per rank for this source projection; no runtime history is inferred")
    parser.add_argument("--expert-routing-source-id", help="explicit synthetic routing assumption identity")
    parser.add_argument("--expert-routing-seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interactive-report", action="store_true",
        help="also write the offline HTML report plus its visualization projection")
    args = parser.parse_args()
    mtp_inputs = (args.max_requests, args.source_block_size_tokens, args.mtp_verified_tokens,
        args.mtp_accepted_tokens, args.mtp_proposer_tokens, args.mtp_acceptance_source_id,
        args.embedding_token_ids, args.v3_mtp_draft_input_token_ids, args.v3_mtp_attention_source)
    if args.mtp_draft_sampling_source is not None:
        if (args.model is None or args.symbolic_forward_source != "vllm_0_27_1_deepseek_v3_sync_fp8"
                or args.target_sampling_source != "vllm_0_27_1_greedy_rejection_bf16"
                or any(value is None for value in mtp_inputs)):
            parser.error("V3 MTP requires its model/source pair, acceptance counts, independent IDs and constructor limits")
    elif any(value is not None for value in mtp_inputs) or args.v3_mtp_batch_invariant is not None:
        parser.error("MTP inputs require the explicit V3 MTP sampling source")
    if args.model is None and args.expert_parallel_size != 1:
        parser.error("static expert-parallel size requires --model")
    if args.serving_matrix_output is not None:
        if args.vllm_sdk_python is None or args.vllm_source_root is None:
            parser.error("virtual Serving matrix requires explicit SDK Python and vLLM source root")
        if not args.vllm_sdk_python.is_file():
            parser.error("virtual Serving SDK Python does not exist")
        from dataset_pipeline.serving_dataset import load_compact_case
        from evaluation.serving.metrics.symbolic_model import (
            run_virtual_serving_evaluation, write_virtual_serving_evaluation_report)
        root = Path(__file__).resolve().parents[3]
        result_path = run_virtual_serving_evaluation(root, args.serving_matrix_output.resolve(),
            args.case, args.jobs, args.serving_python.resolve(strict=True),
            args.vllm_sdk_python.absolute(), args.vllm_source_root.resolve(strict=True),
            load_compact_case, h200_version=None if args.h200_capture_version == "none" else args.h200_capture_version,
            reuse_completed=args.reuse_completed_virtual, serving_timing_mode=args.serving_timing_mode,
            execution_mode=args.serving_execution_mode, state_audit=args.serving_state_audit,
            execution_workers=args.serving_execution_workers, interactive_report=args.interactive_report,
            h200_scenario=args.h200_serving_scenario)
        write_virtual_serving_evaluation_report(result_path, args.output)
        print(result_path)
        return 0
    if args.serving_result is not None:
        if any(value is not None for value in (args.query_lengths, args.context_lengths, args.upstream_vllm_root,
                args.symbolic_forward_source, args.tensor_parallel_size, args.max_model_len, args.max_batched_tokens,
                args.v4_source_use_deep_gemm, args.expert_routing_source_id, args.expert_routing_seed,
                args.v4_indexer_selection_source, args.glm_indexer_selection_source,
                args.target_sampling_source, args.source_workspace_initial_capacity_bytes, args.glm_prefill_attention_source,
                args.torch_include, args.flashinfer_source_root, args.k3_state_indices, args.k3_state_pool_slots,
                args.k3_prefill_output_source)):
            parser.error("saved Serving result mode does not rerun a static model or source comparison")
        print(write_symbolic_serving_report(args.serving_result, args.output))
        return 0
    if args.torch_include is not None and (args.upstream_vllm_root is None
            or args.symbolic_forward_source != "vllm_0_30_v4_sync_fp8_ds_mla"):
        parser.error("--torch-include requires the V4 source profile and upstream source root")
    if args.flashinfer_source_root is not None and (args.upstream_vllm_root is None
            or args.symbolic_forward_source != "vllm_0_30_v4_sync_fp8_ds_mla"):
        parser.error("--flashinfer-source-root requires V4 and the upstream source root")
    if args.query_lengths is None or args.context_lengths is None:
        parser.error("static model mode requires query and context lengths")
    source_profile = "cuda_eager_inplace" if args.symbolic_forward_source is None else args.symbolic_forward_source
    if source_profile != "vllm_0_30_k3_triton_marlin_bf16" and (
            args.k3_state_indices is not None or args.k3_state_pool_slots is not None or args.k3_prefill_output_source is not None):
        parser.error("K3 state inputs require the K3 source program")
    routing = None
    from src.serving.workload.model_config import load_model_config
    if source_profile in ("cuda_eager_inplace", "vllm_ascend_eager") and load_model_config(args.model).model_type == "qwen3_moe":
        if any(value is None for value in (args.expert_routing_source_id, args.expert_routing_seed)):
            parser.error("Qwen3 MoE requires explicit routing identity and seed")
        if any(value is not None for value in (args.max_model_len, args.max_batched_tokens,
                args.v4_source_use_deep_gemm, args.v4_indexer_selection_source, args.glm_indexer_selection_source,
                args.target_sampling_source, args.source_workspace_initial_capacity_bytes, args.glm_prefill_attention_source)):
            parser.error("Qwen3 MoE does not consume V4/GLM source options")
        routing = ExpertRoutingInput.seeded_assumption(args.expert_routing_source_id, args.expert_routing_seed)
    elif source_profile == "vllm_0_11_glm4_sync":
        if args.expert_routing_source_id is None or args.expert_routing_seed is None:
            parser.error("GLM4 requires an explicit routing assumption identity/seed")
        routing = ExpertRoutingInput.seeded_assumption(args.expert_routing_source_id, args.expert_routing_seed)
    elif source_profile == "vllm_0_27_1_deepseek_v3_sync_fp8":
        if args.tensor_parallel_size != 8 or args.expert_parallel_size != 1:
            parser.error("DeepSeek-V3 source requires explicit TP8 and EP-off")
        if any(value is None for value in (args.max_model_len, args.max_batched_tokens,
                args.expert_routing_source_id, args.expert_routing_seed)):
            parser.error("DeepSeek-V3 requires explicit context/batch limits and routing assumption identity/seed")
        if args.target_sampling_source not in (("vllm_0_27_1_greedy_rejection_bf16",)
                if args.mtp_draft_sampling_source is not None else (None, "vllm_0_27_1_plain_greedy_bf16")):
            parser.error("DeepSeek-V3 source requires its matching0.27.1 sampling identity")
        if any(value is not None for value in (
                args.upstream_vllm_root, args.v4_source_use_deep_gemm, args.v4_indexer_selection_source,
                args.glm_indexer_selection_source, args.glm_prefill_attention_source,
                args.source_workspace_initial_capacity_bytes, args.vllm_sdk_python, args.vllm_source_root)):
            parser.error("DeepSeek-V3 static source does not bind legacy/source audit, SDK/provider or V4/GLM options")
        routing = ExpertRoutingInput.seeded_assumption(args.expert_routing_source_id, args.expert_routing_seed)
    elif source_profile == "vllm_0_30_k3_triton_marlin_bf16":
        if any(value is None for value in (args.k3_state_indices, args.k3_state_pool_slots,
                args.expert_routing_source_id, args.expert_routing_seed)):
            parser.error("K3 requires explicit KDA state slots/pool capacity and routing identity/seed")
        if args.target_sampling_source != "vllm_0_30_plain_greedy_bf16":
            parser.error("K3 requires the explicit plain greedy chain")
        routing = ExpertRoutingInput.seeded_assumption(args.expert_routing_source_id, args.expert_routing_seed)
    elif source_profile == "vllm_0_30_v4_sync_fp8_ds_mla":
        if (args.glm_indexer_selection_source is not None or args.target_sampling_source is not None
                or args.glm_prefill_attention_source is not None):
            parser.error("V4 does not consume the GLM selection/target-chain options")
        if any(value is None for value in (args.tensor_parallel_size, args.max_model_len, args.max_batched_tokens,
                args.v4_source_use_deep_gemm, args.expert_routing_source_id, args.expert_routing_seed)):
            parser.error("V4 requires explicit TP, context/batch limits, DeepGEMM source branch and synthetic routing identity/seed")
        routing = ExpertRoutingInput.seeded_assumption(args.expert_routing_source_id, args.expert_routing_seed)
    elif source_profile in ("vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce"):
        if args.v4_source_use_deep_gemm is not None or args.v4_indexer_selection_source is not None:
            parser.error("GLM decoder mode does not consume V4 options")
        if any(value is None for value in (args.tensor_parallel_size, args.max_model_len, args.max_batched_tokens,
                args.expert_routing_source_id, args.expert_routing_seed, args.glm_indexer_selection_source,
                args.source_workspace_initial_capacity_bytes)):
            parser.error("GLM decoder requires explicit TP, limits, routing/cardinality and per-rank workspace history")
        routing = ExpertRoutingInput.seeded_assumption(args.expert_routing_source_id, args.expert_routing_seed)
    elif source_profile == "vllm_0_30_k2_6_int4_sync_explicit_allreduce":
        if any(value is not None for value in (args.v4_source_use_deep_gemm, args.v4_indexer_selection_source,
                args.glm_indexer_selection_source, args.glm_prefill_attention_source,
                args.source_workspace_initial_capacity_bytes, args.upstream_vllm_root)):
            parser.error("K2.6 does not bind V4/indexer/arena options or an independent whole-source comparison")
        if any(value is None for value in (args.tensor_parallel_size, args.max_model_len, args.max_batched_tokens,
                args.expert_routing_source_id, args.expert_routing_seed)):
            parser.error("K2.6 requires explicit TP, context/batch limits and routing assumption identity/seed")
        routing = ExpertRoutingInput.seeded_assumption(args.expert_routing_source_id, args.expert_routing_seed)
    elif any(value is not None for value in (args.max_model_len, args.max_batched_tokens,
            args.v4_source_use_deep_gemm, args.expert_routing_source_id, args.expert_routing_seed,
            args.v4_indexer_selection_source, args.glm_indexer_selection_source, args.target_sampling_source,
            args.source_workspace_initial_capacity_bytes, args.glm_prefill_attention_source)):
        parser.error("Qwen3 source does not consume V4-only source inputs")
    result = write_symbolic_model_report(args.model, tuple(args.query_lengths),
        tuple(args.context_lengths), args.output, upstream_root=args.upstream_vllm_root,
        report=args.interactive_report,
        hardware_config=args.hardware_config,
        source_profile=source_profile, tensor_parallel_size=1 if args.tensor_parallel_size is None else args.tensor_parallel_size,
        expert_parallel_size=args.expert_parallel_size,
        source_max_model_len=args.max_model_len, max_batched_tokens=args.max_batched_tokens, torch_include=args.torch_include,
        flashinfer_source_root=args.flashinfer_source_root,
        source_use_deep_gemm=None if args.v4_source_use_deep_gemm is None else args.v4_source_use_deep_gemm == "enabled",
        expert_routing=routing, target_sampling_source=args.target_sampling_source,
        mtp_draft_sampling_source=args.mtp_draft_sampling_source,
        mtp_verified_tokens=args.mtp_verified_tokens, mtp_accepted_tokens=args.mtp_accepted_tokens,
        mtp_proposer_tokens=args.mtp_proposer_tokens, mtp_acceptance_source_id=args.mtp_acceptance_source_id,
        embedding_token_ids=None if args.embedding_token_ids is None else tuple(args.embedding_token_ids),
        v3_mtp_draft_input_token_ids=(None if args.v3_mtp_draft_input_token_ids is None
            else tuple(tuple(ids) for ids in args.v3_mtp_draft_input_token_ids)),
        v3_mtp_attention_source=args.v3_mtp_attention_source,
        v3_mtp_batch_invariant=None if args.v3_mtp_batch_invariant is None else args.v3_mtp_batch_invariant == "enabled",
        source_max_requests=args.max_requests, source_block_size_tokens=args.source_block_size_tokens,
        prefill_attention_source=args.glm_prefill_attention_source,
        indexer_selection_source=(args.glm_indexer_selection_source
            if source_profile in ("vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
            else args.v4_indexer_selection_source),
        workspace_initial_capacity_bytes=(None if args.source_workspace_initial_capacity_bytes is None
            else tuple(args.source_workspace_initial_capacity_bytes)),
        k3_state_indices=None if args.k3_state_indices is None else tuple(args.k3_state_indices),
        k3_state_pool_slots=args.k3_state_pool_slots, k3_prefill_output_source=args.k3_prefill_output_source)
    print(f"nodes={result['graph_nodes']} calls={result['kernel_calls']} unpriced={len(result['unpriced_calls'])} "
          f"activation_peak_bytes={result['activation_storage_peak_bytes']} trace_cpu_s={result['trace_cpu_seconds']:.6f}")
    failed = len(result["unpriced_calls"])
    if "legacy_path_comparison" in result:
        failed += not result["legacy_path_comparison"]["equivalent"]
    if "source_forward_comparison" in result:
        failed += result["source_forward_comparison"]["failed"]
    if "source_caller_comparison" in result:
        comparison = result["source_caller_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_caller_comparison" in result:
        comparison = result["source_v4_caller_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_mixed_projection_comparison" in result:
        comparison = result["source_v4_mixed_projection_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_compressor_dependency_comparison" in result:
        comparison = result["source_v4_compressor_dependency_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_dependency_comparison" in result:
        comparison = result["source_v4_dependency_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_prepare_comparison" in result:
        comparison = result["source_v4_prepare_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_partial_state_comparison" in result:
        comparison = result["source_v4_partial_state_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_finish_comparison" in result:
        comparison = result["source_v4_finish_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_ffn_comparison" in result:
        comparison = result["source_v4_ffn_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_fp8_dataflow_comparison" in result:
        comparison = result["source_v4_fp8_dataflow_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_quantizer_comparison" in result:
        comparison = result["source_v4_quantizer_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_ffn_vector_comparison" in result:
        comparison = result["source_v4_ffn_vector_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_output_comparison" in result:
        comparison = result["source_v4_output_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_indexer_comparison" in result:
        comparison = result["source_v4_indexer_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_attention_math_comparison" in result:
        comparison = result["source_v4_attention_math_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_mhc_comparison" in result:
        comparison = result["source_v4_mhc_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_qkv_norm_comparison" in result:
        comparison = result["source_v4_qkv_norm_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_target_comparison" in result:
        comparison = result["source_v4_target_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_final_norm_comparison" in result:
        comparison = result["source_v4_final_norm_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_argmax_comparison" in result:
        comparison = result["source_v4_argmax_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_routing_objects_comparison" in result:
        comparison = result["source_v4_routing_objects_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_v4_routing_math_comparison" in result:
        comparison = result["source_v4_routing_math_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    if "source_indexer_gather_workspace_comparison" in result:
        comparison = result["source_indexer_gather_workspace_comparison"]
        failed += comparison["mismatched"] + comparison["not_checked"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
