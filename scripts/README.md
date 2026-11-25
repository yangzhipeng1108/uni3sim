# Script index

`scripts/` contains thin CLIs that call `src/`, `evaluation/`, or `dataset_pipeline/`, and hardware profiler workloads (`*_runner.py`). Active modules must not import scripts. Capture tools may be invoked manually or remotely; an absent in-repository caller alone does not make them unused.

Generate standard Kernel and Serving reports through `scripts/evaluation/run_standard_evaluation_report.py`. Generated outputs belong in `experiment_logs/`; raw capture evidence belongs in `profiler_output/`.

### `scripts/`

| Script | Purpose |
|---|---|
| [export_memory_access.py](export_memory_access.py) | Export source-owned memory programs and their untimed diagnostics. |
| [measure_external_vllm_scheduler_service.py](measure_external_vllm_scheduler_service.py) | Measure a container's installed vLLM Scheduler without model execution. |
| [measure_vllm_scheduler_service.py](measure_vllm_scheduler_service.py) | Capture an independently reproducible native-vLLM Scheduler CPU profile. |
| [run_serving_simulation.py](run_serving_simulation.py) | CLI for the production no-GPU serving simulation. |

### `scripts/baselines/pipeweave/`

| Script | Purpose |
|---|---|
| [benchmark_converter.py](baselines/pipeweave/benchmark_converter.py) | Convert benchmark-runner JSON into PipeWeave training CSVs. |
| [benchmark_runner.py](baselines/pipeweave/benchmark_runner.py) | Run optional PipeWeave-specific operator benchmarks on target hardware. |
| [kernel_dataset_converter.py](baselines/pipeweave/kernel_dataset_converter.py) | Convert canonical Midlayer kernel datasets into PipeWeave training CSVs. |
| [model_trainer.py](baselines/pipeweave/model_trainer.py) | Train hardware-specific PipeWeave MLPs from generated CSV datasets. |

### `scripts/collective_selection/`

| Script | Purpose |
|---|---|
| [build_hccl_observations.py](collective_selection/build_hccl_observations.py) | Validate one HCCL INFO run and write normalized selection facts. |
| [build_mccl_observations.py](collective_selection/build_mccl_observations.py) | Validate one MCCL mcTracer run and write normalized selection facts. |
| [build_nccl_observations.py](collective_selection/build_nccl_observations.py) | Validate one NCCL profiler run and write normalized selection facts. |
| [validate_collective_rules.py](collective_selection/validate_collective_rules.py) | Validate normalized collective observations against selection rules. |

### `scripts/evaluation/`

| Script | Purpose |
|---|---|
| [run_full_evaluation.py](evaluation/run_full_evaluation.py) | CLI for the cross-hardware full evaluation. |
| [run_standard_evaluation_report.py](evaluation/run_standard_evaluation_report.py) | Update the canonical Kernel/full-Serving report from existing evaluators. |

### `scripts/evaluation/kernel/`

| Script | Purpose |
|---|---|
| [ascend_910b_dram_derivation_audit.py](evaluation/kernel/ascend_910b_dram_derivation_audit.py) | CLI for the Ascend 910B DRAM-derivation audit. |
| [check_prediction_order.py](evaluation/kernel/check_prediction_order.py) | Thin CLI for deterministic three-hardware kernel order checks. |
| [gemm_launch_comparison_report.py](evaluation/kernel/gemm_launch_comparison_report.py) | Write the observed-versus-injected GEMM launch comparison report. |
| [kernel_golden_report.py](evaluation/kernel/kernel_golden_report.py) | CLI for deterministic kernel-golden sampling and Event comparison. |
| [kernel_latency_scatter_report.py](evaluation/kernel/kernel_latency_scatter_report.py) | CLI for the cross-hardware kernel latency scatter report. |
| [kernel_mechanism_validation_report.py](evaluation/kernel/kernel_mechanism_validation_report.py) | CLI for the cross-hardware kernel mechanism-validation report. |
| [kernel_memory_traffic_report.py](evaluation/kernel/kernel_memory_traffic_report.py) | CLI for the cross-hardware kernel memory-traffic scatter report. |
| [kernel_standard_report.py](evaluation/kernel/kernel_standard_report.py) | CLI for the report-only single-kernel Forward standard report assembly. |
| [rank_gemm_mechanism_clusters.py](evaluation/kernel/rank_gemm_mechanism_clusters.py) | Rank current GEMM Forward error by static launch-program mechanism. |
| [run_h200_nvjet_raster_candidate.py](evaluation/kernel/run_h200_nvjet_raster_candidate.py) | Evaluate unpublished, duration-free launch facts through the ordinary CLI. |
| [run_kernel_ablation.py](evaluation/kernel/run_kernel_ablation.py) | CLI for the kernel predictor ablation study. |
| [run_kernel_evaluation.py](evaluation/kernel/run_kernel_evaluation.py) | CLI for single-kernel evaluation. |

### `scripts/evaluation/serving/`

| Script | Purpose |
|---|---|
| [agentx_session_replay.py](evaluation/serving/agentx_session_replay.py) | AgentX-derived multi-turn session replay (KV host-offload group, DSR1 TP8). |
| [agentx_session_simulate.py](evaluation/serving/agentx_session_simulate.py) | Thin CLI for AgentX corpus workload build and session simulation. |
| [aggregate_nvidia_h200_nsys_window_kernels.py](evaluation/serving/aggregate_nvidia_h200_nsys_window_kernels.py) | Read-only aggregation of one nsys window (sqlite export), device 0, for the family tables. |
| [ascend_910b_qwen3_4b_tp1_qps16_reduction.py](evaluation/serving/ascend_910b_qwen3_4b_tp1_qps16_reduction.py) | Run the audited 910B Qwen3-4B QPS16 timing reduction. |
| [assemble_nvidia_h200_host_profiles.py](evaluation/serving/assemble_nvidia_h200_host_profiles.py) | Assemble the four H200 Host timing profiles for one vLLM version. |
| [build_ground_truth_metrics_candidates.py](evaluation/serving/build_ground_truth_metrics_candidates.py) | Build candidate ShareGPT metrics from production ground-truth results. |
| [build_metax_c500_mctracer_sync_boundaries.py](evaluation/serving/build_metax_c500_mctracer_sync_boundaries.py) | Create one C500 native mcTracer structural-boundary sidecar for a case. |
| [build_nvidia_h200_sharegpt_candidate.py](evaluation/serving/build_nvidia_h200_sharegpt_candidate.py) | Convert one H200 ShareGPT capture into a compact Serving case candidate. |
| [build_nvidia_h200_sharegpt_candidate_root.py](evaluation/serving/build_nvidia_h200_sharegpt_candidate_root.py) | Build one H200 ShareGPT candidate dataset root for one vLLM version. |
| [build_pd_compact_candidate.py](evaluation/serving/build_pd_compact_candidate.py) | Build one immutable 1P1D Serving compact candidate from measured raw data. |
| [build_production_candidates.py](evaluation/serving/build_production_candidates.py) | Build strict local Production request or target-alignment candidates. |
| [build_sharegpt_candidates.py](evaluation/serving/build_sharegpt_candidates.py) | CLI for building compact ShareGPT Serving candidates. |
| [build_slow_trace_candidate.py](evaluation/serving/build_slow_trace_candidate.py) | Build a provenance-bearing, uniformly slowed production-trace window. |
| [capture_metax_c500_sharegpt_runner.py](evaluation/serving/capture_metax_c500_sharegpt_runner.py) | Capture one C500 ShareGPT cell with a frozen request stream and optional hooks. |
| [capture_nsys.py](evaluation/serving/capture_nsys.py) | CLI for capturing an Nsight Systems serving trace. |
| [check_af_run_artifact.py](evaluation/serving/check_af_run_artifact.py) | Validate a finished standalone AF run-output artifact (Gate A2 checks). |
| [compare_serving_execution_modes.py](evaluation/serving/compare_serving_execution_modes.py) | Compare baseline, detailed, and fast standard Serving result files. |
| [compare_trace_with_nsys.py](evaluation/serving/compare_trace_with_nsys.py) | CLI for comparing a simulated trace with Nsight Systems. |
| [derive_nvidia_h200_runtime_memory_policy.py](evaluation/serving/derive_nvidia_h200_runtime_memory_policy.py) | Derive the H200 runtime memory policies from real vLLM startup logs. |
| [extract_nvidia_h200_attention_kernel_calls.py](evaluation/serving/extract_nvidia_h200_attention_kernel_calls.py) | Extract real rank-0 attention-family kernel calls from H200 nsys traces. |
| [extract_nvidia_h200_fp8_gemm_real_calls.py](evaluation/serving/extract_nvidia_h200_fp8_gemm_real_calls.py) | Extract real H200 FP8 block-scaled dense GEMM calls from nsys GPU traces. |
| [extract_nvidia_h200_fused_moe_trace_calls.py](evaluation/serving/extract_nvidia_h200_fused_moe_trace_calls.py) | Extract H200 Triton fused-MoE launch geometry and per-call durations. |
| [llmservingsim_profile_runner.py](evaluation/serving/llmservingsim_profile_runner.py) | Run an explicitly owned native layerwise capture in an isolated execution copy. |
| [mctracer_to_nsys_converter.py](evaluation/serving/mctracer_to_nsys_converter.py) | CLI for converting mcTracer observations to the Nsight schema. |
| [metax_c500_qwen3_4b_tp1_qps16_reduction.py](evaluation/serving/metax_c500_qwen3_4b_tp1_qps16_reduction.py) | Run the audited C500 Qwen3-4B QPS16 timing reduction. |
| [msprof_summary_converter.py](evaluation/serving/msprof_summary_converter.py) | CLI for converting msprof operator summaries. |
| [msprof_timeline_converter.py](evaluation/serving/msprof_timeline_converter.py) | CLI for converting msprof timelines. |
| [nvidia_h200_deepseek_r1_nsys_decode_capture.sh](evaluation/serving/nvidia_h200_deepseek_r1_nsys_decode_capture.sh) | Capture pure-decode nsys windows of the real vLLM DeepSeek-R1 server on H200 |
| [nvidia_h200_deepseek_r1_nsys_prefill_capture.sh](evaluation/serving/nvidia_h200_deepseek_r1_nsys_prefill_capture.sh) | Capture one prefill-only nsys window of the real vLLM DeepSeek-R1 server on |
| [nvidia_h200_deepseek_r1_rank_proxy_config.py](evaluation/serving/nvidia_h200_deepseek_r1_rank_proxy_config.py) | Write a single-GPU DeepSeek-R1 model directory with the per-rank shapes of TP=N. |
| [nvidia_h200_deepseek_r1_tp8_capture_runner.py](evaluation/serving/nvidia_h200_deepseek_r1_tp8_capture_runner.py) | DeepSeek-R1 TP8 real-serving capture on 8x H200 with vLLM 0.30 (dummy weights). |
| [nvidia_h200_deepseek_r1_tp8_day2.sh](evaluation/serving/nvidia_h200_deepseek_r1_tp8_day2.sh) | DeepSeek-R1 on 8x H200, 2026-09-27 8-GPU capture window. |
| [nvidia_h200_deepseek_r1_tp8_offload_serve.sh](evaluation/serving/nvidia_h200_deepseek_r1_tp8_offload_serve.sh) | vLLM 0.30 DeepSeek-R1-0528 TP8 server for the KV host-offload group. |
| [nvidia_h200_deepseek_r1_tp8_serve.sh](evaluation/serving/nvidia_h200_deepseek_r1_tp8_serve.sh) | Real vLLM 0.30 DeepSeek-R1-0528 server on 8x H200, TP8, dummy weights. |
| [nvidia_h200_export_nsys_windows.sh](evaluation/serving/nvidia_h200_export_nsys_windows.sh) | sqlite-export every window_N.nsys-rep of one capture directory and write the |
| [nvidia_h200_glm_5_2_tp8_capture_runner.py](evaluation/serving/nvidia_h200_glm_5_2_tp8_capture_runner.py) | GLM-5.2 TP8 real-serving capture on 8x H200 with vLLM 0.30 (dummy weights). |
| [nvidia_h200_glm_5_2_tp8_serve.sh](evaluation/serving/nvidia_h200_glm_5_2_tp8_serve.sh) | Real vLLM 0.30 GLM-5.2 server on 8x H200, TP8, dummy weights. |
| [nvidia_h200_memory_ledger.py](evaluation/serving/nvidia_h200_memory_ledger.py) | Emit a reproducible projected H200 PD per-rank memory ledger. |
| [nvidia_h200_nsys_prefill_capture.sh](evaluation/serving/nvidia_h200_nsys_prefill_capture.sh) | Capture prefill-only windows of the real vLLM DSV3-L32 TP4 server on 4x H200. |
| [nvidia_h200_nsys_pure_decode_capture.sh](evaluation/serving/nvidia_h200_nsys_pure_decode_capture.sh) | Capture pure-decode windows of the real vLLM DSV3-L32 TP4 server on 4x H200. |
| [nvidia_h200_serve_deepseek_r1.sh](evaluation/serving/nvidia_h200_serve_deepseek_r1.sh) | Real vLLM 0.24 DeepSeek-R1 architecture (dummy FP8 block-128 weights) on H200. |
| [nvidia_h200_tp8_serving_capture.py](evaluation/serving/nvidia_h200_tp8_serving_capture.py) | Compatibility imports for H200 capture runners; orchestration belongs to profiler. |
| [probe_first_schedule.py](evaluation/serving/probe_first_schedule.py) | Capture first-admission membership during a paired virtual Serving case. |
| [production_trace_report.py](evaluation/serving/production_trace_report.py) | Render a projected production-Trace arrival-rate sweep as Markdown. |
| [report_activation_memory.py](evaluation/serving/report_activation_memory.py) | Compare source-derived activation storage with existing startup records. |
| [report_glm53_fp8_memory_budget.py](evaluation/serving/report_glm53_fp8_memory_budget.py) | Emit GLM-5.3 FP8 per-rank HBM and native KV allocation budgets. |
| [report_glm_official_fp8_family.py](evaluation/serving/report_glm_official_fp8_family.py) | Compare official GLM-5.2-FP8/5.3 source graphs, capacity terms and math. |
| [report_interactive.py](evaluation/serving/report_interactive.py) | Regenerate an offline Serving report from retained visualization evidence. |
| [report_kimi_k3_serving.py](evaluation/serving/report_kimi_k3_serving.py) | Run ordinary Kimi-K3 text Serving or its native H200 capacity plan. |
| [report_kimi_k3_source_components.py](evaluation/serving/report_kimi_k3_source_components.py) | Publish independent K3 source components; no complete model/runtime claim. |
| [report_model_operator_probe.py](evaluation/serving/report_model_operator_probe.py) | Emit static full-model Roofline operator facts for source comparison. |
| [report_model_source_identity.py](evaluation/serving/report_model_source_identity.py) | Report exact model/native-mode identity through the real vLLM 0.30 SDK. |
| [report_model_weight_residency.py](evaluation/serving/report_model_weight_residency.py) | Report GLM residency, V4 parameters or Kimi active outer parameters. |
| [report_native_startup_memory.py](evaluation/serving/report_native_startup_memory.py) | Compare a captured Ascend startup with analytical activation and KV capacity. |
| [report_operator_support_checkpoint.py](evaluation/serving/report_operator_support_checkpoint.py) | Regenerate the seven-model Goal checkpoint from static/source artifacts. |
| [report_symbolic_model.py](evaluation/serving/report_symbolic_model.py) | Report a metadata-only full model graph, live storage and Roofline costs. |
| [report_v4_flash_alignment.py](evaluation/serving/report_v4_flash_alignment.py) | Generate independent Flash source expectations and compare probe artifacts. |
| [report_v4_source_sequence.py](evaluation/serving/report_v4_source_sequence.py) | Check consecutive Flash source graphs on CPU-only virtual tensor metadata. |
| [report_weight_placement.py](evaluation/serving/report_weight_placement.py) | Generate a logical startup weight inventory for colocated instances. |
| [run_interactive_example.py](evaluation/serving/run_interactive_example.py) | Run a CPU-only Serving example and generate its offline report. |
| [run_llmservingsim.py](evaluation/serving/run_llmservingsim.py) | Explicit entry point for the independent LLMServingSim CPU baseline. |
| [run_paired_virtual_case.py](evaluation/serving/run_paired_virtual_case.py) | Execute one virtual Serving case and retain its identity-checked comparison. |
| [run_pd_disaggregation_study.py](evaluation/serving/run_pd_disaggregation_study.py) | Run colocated-versus-PD ShareGPT experiments with separately supplied PD datasets. |
| [run_production_ground_truth_comparison.py](evaluation/serving/run_production_ground_truth_comparison.py) | Compare Serving simulation aggregates with production ground truth. |
| [run_serving_metric_evaluation.py](evaluation/serving/run_serving_metric_evaluation.py) | CLI for serving-metric evaluation and aggregation. |
| [run_sharegpt_compact_evaluation.py](evaluation/serving/run_sharegpt_compact_evaluation.py) | Run aggregate-only ShareGPT compact-suite evaluation. |
| [simulation_runner_entrypoint.py](evaluation/serving/simulation_runner_entrypoint.py) | Spawn-safe command-line entry point for the Serving simulator. |
| [summarize_nvidia_h200_nsys_decode_windows.py](evaluation/serving/summarize_nvidia_h200_nsys_decode_windows.py) | Summarise nsys decode windows of the real vLLM DSV3-L32 TP4 server. |
| [validate_serving_trace.py](evaluation/serving/validate_serving_trace.py) | CLI for validating a serving trace. |
| [validate_sharegpt_compact_alignment.py](evaluation/serving/validate_sharegpt_compact_alignment.py) | Validate all compact ShareGPT cases and run structural alignment. |
| [validate_target_model_blocks.py](evaluation/serving/validate_target_model_blocks.py) | Run source-pinned tiny GLM-5.2 and Kimi-K2.6 CUDA block validation. |
| [validate_trace_alignment.py](evaluation/serving/validate_trace_alignment.py) | CLI for validating serving-trace alignment. |
| [write_nvidia_h200_sharegpt_suites.py](evaluation/serving/write_nvidia_h200_sharegpt_suites.py) | Write the three H200 ShareGPT suite configs of one vLLM version. |

### `scripts/evaluation/serving/diagnostics/`

| Script | Purpose |
|---|---|
| [analyze_ascend_910b_selected_step_msprof.py](evaluation/serving/diagnostics/analyze_ascend_910b_selected_step_msprof.py) | Analyze a same-run 910B selected-step msprof capture. |
| [analyze_metax_c500_selected_step_mctracer.py](evaluation/serving/diagnostics/analyze_metax_c500_selected_step_mctracer.py) | Correlate a validated C500 selected-step capture with mcTracer tasks. |
| [analyze_nvidia_a800_selected_step_nsys.py](evaluation/serving/diagnostics/analyze_nvidia_a800_selected_step_nsys.py) | Extract correlated A800 selected-step CUDA Graph intervals from NSYS. |
| [audit_scheduler_depth_metadata.py](evaluation/serving/diagnostics/audit_scheduler_depth_metadata.py) | Audit the complete Scheduler-depth metadata-only migration candidate. |
| [compare_dataset_serving_inputs.py](evaluation/serving/diagnostics/compare_dataset_serving_inputs.py) | Compare exact Dataset and Serving predictor inputs for three hardware. |
| [inspect_runtime_prepare_occurrences.py](evaluation/serving/diagnostics/inspect_runtime_prepare_occurrences.py) | Inspect source-authored runtime-prepare occurrence signatures for one cell. |
| [materialize_selected_step.py](evaluation/serving/diagnostics/materialize_selected_step.py) | Write complete C_step rankings and one selected-step injection input. |
| [migrate_scheduler_depth_metadata.py](evaluation/serving/diagnostics/migrate_scheduler_depth_metadata.py) | Create a complete Scheduler-depth metadata candidate for one hardware. |
| [replay_selected_step_injection.py](evaluation/serving/diagnostics/replay_selected_step_injection.py) | Derive a full prediction fingerprint for one injected Scheduler step. |
| [run_admission_sensitivity_sweep.py](evaluation/serving/diagnostics/run_admission_sensitivity_sweep.py) | Sweep a controlled admission shift across selected Serving cells. |
| [run_graph_padding_paired_steps.py](evaluation/serving/diagnostics/run_graph_padding_paired_steps.py) | Price one case's observed steps under exact and real graph capture sets. |
| [run_kernel_coverage.py](evaluation/serving/diagnostics/run_kernel_coverage.py) | Build a Serving-to-Kernel coverage census for one hardware. |
| [run_oracle_ttft_reconstruction.py](evaluation/serving/diagnostics/run_oracle_ttft_reconstruction.py) | Reconstruct client TTFT from an absolute-clock Scheduler capture. |
| [run_scheduler_depth_breakdown.py](evaluation/serving/diagnostics/run_scheduler_depth_breakdown.py) | Run the exact Scheduler-depth TTFT/TPOT signed error breakdown. |
| [run_scheduler_error_propagation.py](evaluation/serving/diagnostics/run_scheduler_error_propagation.py) | Measure how dynamic Scheduling propagates synthetic step-duration errors. |
| [run_scheduler_pipeline_depth_study.py](evaluation/serving/diagnostics/run_scheduler_pipeline_depth_study.py) | Run the 45-case native-vs-depth-one Scheduler capacity study. |
| [selected_step_runner.py](evaluation/serving/diagnostics/selected_step_runner.py) | Run a deterministic pure-decode selected-step probe on vLLM hardware. |
| [validate_scheduler_alignment.py](evaluation/serving/diagnostics/validate_scheduler_alignment.py) | Run aggregate-only Scheduler alignment for an explicit suite subset. |
| [validate_selected_step_injection.py](evaluation/serving/diagnostics/validate_selected_step_injection.py) | Validate a selected-step hardware injection against its canonical state. |

### `scripts/evaluation/serving/scheduler/`

| Script | Purpose |
|---|---|
| [ascend_910b_canonical_trajectory_counterfactual.py](evaluation/serving/scheduler/ascend_910b_canonical_trajectory_counterfactual.py) | Run the formal 910B canonical-trajectory counterfactual. |
| [ascend_910b_same_trajectory_reduction.py](evaluation/serving/scheduler/ascend_910b_same_trajectory_reduction.py) | Run the audited 910B matched-trajectory timing reduction. |
| [build_scheduler_capture_candidate.py](evaluation/serving/scheduler/build_scheduler_capture_candidate.py) | Build one schema-v3 Serving case from an accepted Scheduler capture. |
| [compare_scheduler_trace_ab.py](evaluation/serving/scheduler/compare_scheduler_trace_ab.py) | Compare trace-off and trace-on Serving ground-truth runs. |
| [publish_scheduler_capture_candidates.py](evaluation/serving/scheduler/publish_scheduler_capture_candidates.py) | Publish fully validated Scheduler capture candidates to one dataset root. |
| [reconcile_scheduler_capture_suite.py](evaluation/serving/scheduler/reconcile_scheduler_capture_suite.py) | Validate and summarize a complete Scheduler/Graph Serving capture suite. |
| [report_scheduling_trajectory_replay.py](evaluation/serving/scheduler/report_scheduling_trajectory_replay.py) | Plot Scheduling Trajectory Replay and optional A/B/C/D attribution results. |
| [run_scheduling_trajectory_replay.py](evaluation/serving/scheduler/run_scheduling_trajectory_replay.py) | Run Scheduling Trajectory Replay for one canonical observed Scheduler trace. |

### `scripts/experiments/`

| Script | Purpose |
|---|---|
| [ascend_910b_mooncake_device_transfer_runner.py](experiments/ascend_910b_mooncake_device_transfer_runner.py) | Verify one Ascend Mooncake NPU-to-NPU KV-style transfer without host staging. |
| [ascend_910b_serving_alignment_runner.py](experiments/ascend_910b_serving_alignment_runner.py) | Capture one real Ascend 910B vLLM eager-prefill window for kernel alignment. |
| [compare_pd_observer_effect.py](experiments/compare_pd_observer_effect.py) | Write strict paired trace-off/trace-on PD observer-effect evidence. |
| [deploy_nvidia_h200_serving_capture.sh](experiments/deploy_nvidia_h200_serving_capture.sh) | Usage: deploy_nvidia_h200_serving_capture.sh  (from a clean checkout; prints the snapshot dir) |
| [execute_pd_capture_plan.py](experiments/execute_pd_capture_plan.py) | Execute one materialized real 1P1D PD capture plan. |
| [freeze_nvidia_a800_pd_source_suite.py](experiments/freeze_nvidia_a800_pd_source_suite.py) | Freeze the current 15-case A800 ShareGPT source contract for PD capture. |
| [materialize_pd_capture_plan.py](experiments/materialize_pd_capture_plan.py) | CLI for creating one frozen ShareGPT 1P1D PD capture plan. |
| [metax_c500_mccl_device_transfer_runner.py](experiments/metax_c500_mccl_device_transfer_runner.py) | Validate one direct MCCL transfer between two MetaX C500 devices. |
| [metax_c500_mooncake_device_transfer_runner.py](experiments/metax_c500_mooncake_device_transfer_runner.py) | Run one isolated same-host Mooncake transfer on two MetaX C500 devices. |
| [modal_nvidia_h200_moe_small_traffic_runner.py](experiments/modal_nvidia_h200_moe_small_traffic_runner.py) | One budgeted H200 capture of non-duration BF16 W1/W2 counters. |
| [nvidia_a800_serving_alignment_runner.py](experiments/nvidia_a800_serving_alignment_runner.py) | Capture one real A800 vLLM eager-prefill window for kernel alignment. |
| [nvidia_h200_gemm_module_modal_runner.py](experiments/nvidia_h200_gemm_module_modal_runner.py) | One bounded, untimed H200 GEMM module/launch observation. |
| [nvidia_h200_moe_small_traffic_runner.py](experiments/nvidia_h200_moe_small_traffic_runner.py) | Counter-only small BF16 prepared-expert fixture; no timing-based pricing. |
| [nvidia_h200_serving_alignment_runner.py](experiments/nvidia_h200_serving_alignment_runner.py) | Capture one H200 ShareGPT alignment raw case (operators + Scheduler + Nsight). |
| [nvidia_h200_serving_api_boundary_modal_runner.py](experiments/nvidia_h200_serving_api_boundary_modal_runner.py) | One bounded H200 native Host-API diagnostic; original model/CLI retained. |
| [nvidia_h200_serving_boundary_modal_runner.py](experiments/nvidia_h200_serving_boundary_modal_runner.py) | One authorized1xH200 reference/detailed Serving boundary campaign. |
| [nvidia_h200_serving_selective_modal_runner.py](experiments/nvidia_h200_serving_selective_modal_runner.py) | One bounded2xH200 full MoE local-code observation, with shared-pool accounting. |
| [request_router.py](experiments/request_router.py) | CLI for the OpenAI-compatible evaluation request router. |
| [run_nvidia_h200_sharegpt_capture.py](experiments/run_nvidia_h200_sharegpt_capture.py) | Run the full H200 ShareGPT capture for one lane of cases (stdlib only). |
| [run_serving_experiments.py](experiments/run_serving_experiments.py) | CLI for manifest-driven serving experiments. |
| [run_sharegpt_ground_truth.py](experiments/run_sharegpt_ground_truth.py) | CLI for canonical ShareGPT production Serving ground truth. |
| [run_sharegpt_pd_ground_truth.py](experiments/run_sharegpt_pd_ground_truth.py) | CLI for one frozen ShareGPT 1P1D Serving ground-truth benchmark. |

### `scripts/kernel_dataset/`

| Script | Purpose |
|---|---|
| [fa_serving_capture_ascend_910b_full_runner.py](kernel_dataset/fa_serving_capture_ascend_910b_full_runner.py) | Capture frozen Ascend 910B FA cases as Timeline and op-mode evidence. |
| [fa_serving_capture_audit_runner.py](kernel_dataset/fa_serving_capture_audit_runner.py) | Create fail-closed candidate audit artifacts for FA serving capture. |
| [fa_serving_capture_metax_c500_full_runner.py](kernel_dataset/fa_serving_capture_metax_c500_full_runner.py) | Capture frozen FA serving candidates on MetaX C500 one case at a time. |
| [fa_serving_capture_metax_c500_timing_audit.py](kernel_dataset/fa_serving_capture_metax_c500_timing_audit.py) | Emit C500 FA Event/trace timing evidence when counters are unavailable. |
| [fa_serving_capture_nvidia_a800_full_runner.py](kernel_dataset/fa_serving_capture_nvidia_a800_full_runner.py) | Capture frozen A800 FA cases as NSYS windows plus NCU raw-page evidence. |
| [fa_serving_capture_report.py](kernel_dataset/fa_serving_capture_report.py) | Write final candidate-only FA serving publication-audit artifacts. |
| [fa_serving_capture_runner.py](kernel_dataset/fa_serving_capture_runner.py) | Validate and materialize frozen cross-hardware FA serving-capture inputs. |
| [fa_serving_forward_report.py](kernel_dataset/fa_serving_forward_report.py) | Run Forward accuracy on the published FA serving capture datasets. |
| [fa_serving_release.py](kernel_dataset/fa_serving_release.py) | Publish one hardware's FA serving capture evidence dataset. |
| [merge_fa_serving_release.py](kernel_dataset/merge_fa_serving_release.py) | Merge one published FA serving campaign into the canonical FA dataset. |

### `scripts/kernel_dataset/ascend_910b/`

| Script | Purpose |
|---|---|
| [attach_gemm_launch_observations_runner.py](kernel_dataset/ascend_910b/attach_gemm_launch_observations_runner.py) | Attach duration-free CANN launch observations to a GEMM Dataset. |
| [build_dataset.py](kernel_dataset/ascend_910b/build_dataset.py) | CLI for building the Ascend 910B kernel dataset. |
| [event_timer_sanity_runner.py](kernel_dataset/ascend_910b/event_timer_sanity_runner.py) | Validate Ascend 910B ACL-backed NPU Event timing without a profiler. |
| [flash_attn_runner.py](kernel_dataset/ascend_910b/flash_attn_runner.py) | FlashAttention runner for 910B dataset profiling. |
| [fused_moe_runner.py](kernel_dataset/ascend_910b/fused_moe_runner.py) | Standalone vLLM-Ascend fused-MoE runner for 910B dataset profiling. |
| [gemm_runner.py](kernel_dataset/ascend_910b/gemm_runner.py) | GEMM runner for 910B dataset profiling (torch.matmul on NPU). |
| [kernel_golden_runner.py](kernel_dataset/ascend_910b/kernel_golden_runner.py) | Run the fixed no-profiler Ascend 910B kernel-golden sample serially. |
| [profile_parallel.sh](kernel_dataset/ascend_910b/profile_parallel.sh) | Parallel msprof-op profiling across the 8 Ascend 910B NPUs. |
| [vector_runner.py](kernel_dataset/ascend_910b/vector_runner.py) | Run one Ascend 910B Vector workload or a persistent Event batch. |

### `scripts/kernel_dataset/metax_c500/`

| Script | Purpose |
|---|---|
| [audit_full_event_capture_runner.py](kernel_dataset/metax_c500/audit_full_event_capture_runner.py) | CLI for the MetaX C500 full OP Event campaign audit. |
| [build_dataset.py](kernel_dataset/metax_c500/build_dataset.py) | CLI for building the MetaX C500 kernel dataset. |
| [build_full_event_campaign_manifest_runner.py](kernel_dataset/metax_c500/build_full_event_campaign_manifest_runner.py) | CLI for assembling one immutable MetaX C500 Event campaign manifest. |
| [build_full_event_candidate_runner.py](kernel_dataset/metax_c500/build_full_event_candidate_runner.py) | CLI for the candidate-only MetaX C500 987-case Event dataset builder. |
| [build_full_event_selection_runner.py](kernel_dataset/metax_c500/build_full_event_selection_runner.py) | CLI for building the audited MetaX C500 Device Event selection. |
| [collect_process_isolation_snapshot_runner.py](kernel_dataset/metax_c500/collect_process_isolation_snapshot_runner.py) | CLI for collecting one complete MetaX C500 host process snapshot. |
| [event_timer_sanity_runner.py](kernel_dataset/metax_c500/event_timer_sanity_runner.py) | Validate MetaX C500 current-stream Event timing without a profiler. |
| [flash_attn_runner.py](kernel_dataset/metax_c500/flash_attn_runner.py) | Serving-aligned MetaX C500 FlashAttention profiler runner. |
| [fused_moe_runner.py](kernel_dataset/metax_c500/fused_moe_runner.py) | Standalone vLLM-equivalent MetaX fused-MoE runner without importing vLLM. |
| [fused_moe_vllm024_runner.py](kernel_dataset/metax_c500/fused_moe_vllm024_runner.py) | Capture vLLM-metax 0.24 FusedMoE launch geometry on C500. |
| [gemm_runner.py](kernel_dataset/metax_c500/gemm_runner.py) | Run the canonical MetaX Linear/GEMM workload with device Events. |
| [kernel_golden_graph_runner.py](kernel_dataset/metax_c500/kernel_golden_graph_runner.py) | Measure canonical C500 cases with graph-amortized device Events. |
| [kernel_golden_runner.py](kernel_dataset/metax_c500/kernel_golden_runner.py) | Run the fixed no-profiler C500 kernel-golden sample serially. |
| [profile_all.sh](kernel_dataset/metax_c500/profile_all.sh) | Run full mcProfiler + mcTracer profiling for all kernel types. |
| [profile_fused_moe_parallel.sh](kernel_dataset/metax_c500/profile_fused_moe_parallel.sh) | Compatibility entry point for segmented fused-MoE profiling. |
| [run_event_campaign_attempt_runner.py](kernel_dataset/metax_c500/run_event_campaign_attempt_runner.py) | CLI for one auditable MetaX C500 Device Event attempt. |
| [vector_runner.py](kernel_dataset/metax_c500/vector_runner.py) | MetaX C500 Serving-aligned vector operator runner. |

### `scripts/kernel_dataset/nvidia_a800/`

| Script | Purpose |
|---|---|
| [build_dataset.py](kernel_dataset/nvidia_a800/build_dataset.py) | CLI for building the NVIDIA A800 kernel dataset. |
| [event_timer_sanity_runner.py](kernel_dataset/nvidia_a800/event_timer_sanity_runner.py) | Validate NVIDIA A800 CUDA Event timing without a profiler. |
| [flash_attn_runner.py](kernel_dataset/nvidia_a800/flash_attn_runner.py) | FlashAttention runner for the A800 single-op NCU dataset. |
| [fused_moe_runner.py](kernel_dataset/nvidia_a800/fused_moe_runner.py) | Fused-MoE runner for the A800 single-op NCU dataset (vLLM fused_experts). |
| [gemm_runner.py](kernel_dataset/nvidia_a800/gemm_runner.py) | GEMM runner for the A800 single-op NCU dataset (torch.matmul on CUDA). |
| [kernel_golden_runner.py](kernel_dataset/nvidia_a800/kernel_golden_runner.py) | Run the fixed no-profiler NVIDIA A800 kernel-golden sample serially. |
| [vector_runner.py](kernel_dataset/nvidia_a800/vector_runner.py) | Serving-aligned A800 Vector runner for NCU and no-profiler CUDA Events. |

### `scripts/kernel_dataset/nvidia_h200/`

| Script | Purpose |
|---|---|
| [build_dataset.py](kernel_dataset/nvidia_h200/build_dataset.py) | CLI for NVIDIA H200 kernel-dataset candidates (plans / assemble / check / publish). |
| [build_existing_evidence_candidate.py](kernel_dataset/nvidia_h200/build_existing_evidence_candidate.py) | Build a review-only H200 kernel dataset from already pinned evidence. |
| [fa_instruction_counter_modal_runner.py](kernel_dataset/nvidia_h200/fa_instruction_counter_modal_runner.py) | Three source-matched native FA non-duration instruction counter points. |
| [flash_attn_runner.py](kernel_dataset/nvidia_h200/flash_attn_runner.py) | NVIDIA H200 owner copy of ``scripts/kernel_dataset/nvidia_a800/flash_attn_runner.py``. |
| [fused_moe_counter_case.py](kernel_dataset/nvidia_h200/fused_moe_counter_case.py) | One synthetic complete BF16 FusedMoE call for H200 traffic counters. |
| [fused_moe_counter_modal_runner.py](kernel_dataset/nvidia_h200/fused_moe_counter_modal_runner.py) | Capture non-duration L2 and DRAM traffic for a synthetic vLLM 0.30 MoE call. |
| [fused_moe_runner.py](kernel_dataset/nvidia_h200/fused_moe_runner.py) | NVIDIA H200 owner copy of ``scripts/kernel_dataset/nvidia_a800/fused_moe_runner.py``. |
| [gemm_runner.py](kernel_dataset/nvidia_h200/gemm_runner.py) | NVIDIA H200 owner copy of ``scripts/kernel_dataset/nvidia_a800/gemm_runner.py``. |
| [kc_deepseek_r1_decode_runner.py](kernel_dataset/nvidia_h200/kc_deepseek_r1_decode_runner.py) | H200 kcampaign runner for the DeepSeek-R1/V3 decode-path operators vLLM 0.24 launches. |
| [kc_dsa_cache_write_runner.py](kernel_dataset/nvidia_h200/kc_dsa_cache_write_runner.py) | NVIDIA H200 DeepSeek Sparse Attention (DSA) KV-cache write runner. |
| [kc_dsa_indexer_runner.py](kernel_dataset/nvidia_h200/kc_dsa_indexer_runner.py) | NVIDIA H200 DeepSeek Sparse Attention (DSA) indexer runner. |
| [kc_dsa_sparse_mla_runner.py](kernel_dataset/nvidia_h200/kc_dsa_sparse_mla_runner.py) | NVIDIA H200 DeepSeek Sparse Attention (DSA) sparse-MLA attention runner. |
| [kc_k3_mla_runner.py](kernel_dataset/nvidia_h200/kc_k3_mla_runner.py) | NVIDIA H200 Kimi-K3 gated-MLA decode runner (vLLM 0.30.0 only). |
| [kc_kda_runner.py](kernel_dataset/nvidia_h200/kc_kda_runner.py) | NVIDIA H200 Kimi Delta Attention (KDA) kernel runner. |
| [kc_v4attn_runner.py](kernel_dataset/nvidia_h200/kc_v4attn_runner.py) | NVIDIA H200 DeepSeek-V4 attention-component runner (CSA / HCA / SWA, |
| [kc_wq_moe_runner.py](kernel_dataset/nvidia_h200/kc_wq_moe_runner.py) | NVIDIA H200 kernel-dataset runner for weight-only-quantized (W4A16) MoE. |
| [missing39_modal_runner.py](kernel_dataset/nvidia_h200/missing39_modal_runner.py) | Run the prepared 39-case H200 NSYS plan on one Modal GPU. |
| [moe_request_layers_modal_runner.py](kernel_dataset/nvidia_h200/moe_request_layers_modal_runner.py) | Native non-duration request-layer attribution, never a Forward rate. |
| [ncu_coverage_modal_runner.py](kernel_dataset/nvidia_h200/ncu_coverage_modal_runner.py) | Capture pinned non-duration NCU evidence for one bounded H200 batch. |
| [ncu_subset_modal_runner.py](kernel_dataset/nvidia_h200/ncu_subset_modal_runner.py) | Capture NCU counters for three pinned H200 dataset cases on Modal. |
| [nsys_campaign_runner.py](kernel_dataset/nvidia_h200/nsys_campaign_runner.py) | Remote NSYS campaign for the H200 kernel dataset (A800 canonical protocol). |
| [nvjet_cublaslt_129_modal_runner.py](kernel_dataset/nvidia_h200/nvjet_cublaslt_129_modal_runner.py) | Capture a CUDA 12.9 cuBLASLt nvjet candidate without timing calibration. |
| [nvjet_static_case.py](kernel_dataset/nvidia_h200/nvjet_static_case.py) | One synthetic BF16 GEMM for non-duration nvjet program inspection. |
| [nvjet_static_modal_runner.py](kernel_dataset/nvidia_h200/nvjet_static_modal_runner.py) | Capture CUDA 13.0 H200 nvjet non-duration program evidence. |
| [occupancy_latency_modal_runner.py](kernel_dataset/nvidia_h200/occupancy_latency_modal_runner.py) | Modal runner for independent occupancy-latency probe on H200. |
| [occupancy_latency_probe_runner.py](kernel_dataset/nvidia_h200/occupancy_latency_probe_runner.py) | Independent occupancy-latency probe: L2 latency vs warp occupancy. |
| [prepare_missing15_ncu.py](kernel_dataset/nvidia_h200/prepare_missing15_ncu.py) | Prepare raw-only H200 NCU plans for the 15 missing NSYS shapes. |
| [prepare_ncu_clock_settle_retry.py](kernel_dataset/nvidia_h200/prepare_ncu_clock_settle_retry.py) | Prepare one H200 FA NCU retry after adding pre-NVTX clock settling. |
| [prepare_ncu_coverage_runner.py](kernel_dataset/nvidia_h200/prepare_ncu_coverage_runner.py) | Prepare pinned H200 NCU case batches for A800's kernel case IDs. |
| [prepare_ncu_quality_retry.py](kernel_dataset/nvidia_h200/prepare_ncu_quality_retry.py) | Prepare bounded H200 NCU retries for first-pass quality failures. |
| [publish_missing39_runner.py](kernel_dataset/nvidia_h200/publish_missing39_runner.py) | Publish the validated H200 missing-case and NCU candidate from local raw. |
| [review_missing6_nsys.py](kernel_dataset/nvidia_h200/review_missing6_nsys.py) | Write a pinned worker review for one returned H200 NSYS retry. |
| [run_ncu_coverage_batches.py](kernel_dataset/nvidia_h200/run_ncu_coverage_batches.py) | Run explicitly named H200 NCU batches one at a time under a spend cap. |
| [silu_768_modal_runner.py](kernel_dataset/nvidia_h200/silu_768_modal_runner.py) | Modal runner for 768-thread SiLU scalar latency measurement on H200. |
| [single_shot_latency_modal_runner.py](kernel_dataset/nvidia_h200/single_shot_latency_modal_runner.py) | Modal runner for single-shot occupancy-latency probe on H200. |
| [single_shot_latency_probe_runner.py](kernel_dataset/nvidia_h200/single_shot_latency_probe_runner.py) | Single-shot occupancy-latency probe for H200. |
| [vector_elementwise_case_runner.py](kernel_dataset/nvidia_h200/vector_elementwise_case_runner.py) | Vector elementwise case for NCU stall capture. |
| [vector_elementwise_stall_modal_runner.py](kernel_dataset/nvidia_h200/vector_elementwise_stall_modal_runner.py) | Native non-duration Vector kernel stall attribution, never a Forward rate. |
| [vector_runner.py](kernel_dataset/nvidia_h200/vector_runner.py) | NVIDIA H200 owner copy of ``scripts/kernel_dataset/nvidia_a800/vector_runner.py``. |

### `scripts/kernel_dataset/samsung_hbm_pim/`

| Script | Purpose |
|---|---|
| [build_dataset.py](kernel_dataset/samsung_hbm_pim/build_dataset.py) | Build Samsung HBM-PIM data with a separately supplied `--config`. |

### `scripts/launch_data/`

| Script | Purpose |
|---|---|
| [ascend_910b_launch_generator.py](launch_data/ascend_910b_launch_generator.py) | CLI for validating or generating Ascend 910B launch data. |
| [build_gemm_workload_candidates.py](launch_data/build_gemm_workload_candidates.py) | Write the aligned ShareGPT GEMM workload universe as candidate data. |
| [metax_c500_launch_generator.py](launch_data/metax_c500_launch_generator.py) | CLI for validating or generating MetaX C500 launch-selection CSV. |
| [nvidia_a800_launch_generator.py](launch_data/nvidia_a800_launch_generator.py) | CLI for validating or generating NVIDIA A800 launch-selection CSV. |
| [nvidia_h200_launch_generator.py](launch_data/nvidia_h200_launch_generator.py) | CLI for validating or generating NVIDIA H200 launch-selection CSV. |
| [nvidia_h200_v6_lut_from_capture.py](launch_data/nvidia_h200_v6_lut_from_capture.py) | Convert segment1 + segment2 + BMM captures into one v6 LUT candidate. |
