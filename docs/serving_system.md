# Uni³Sim Serving system design

The Serving layer predicts how request workloads, model structure, deployment, and hardware interact through scheduling, computation, communication, and migration to produce latency, throughput, and bottlenecks. It produces complete operator/launch semantics and consumes invocation predictions from the [Kernel model](kernel_performance_model.md). Neither layer duplicates the other's responsibilities.

## Running the examples

The [README quickstart](../README.md#quickstart) runs a CPU-only synthetic conformance example with the base dependencies in [requirements.txt](../requirements.txt). It exercises scheduling, workload lowering, prediction and reporting without model weights or a native framework SDK. Its support marker remains `unsupported_a800_analytical_launch`; it validates the execution path rather than hardware accuracy.

Install [requirements-reports.txt](../requirements-reports.txt) for plots.

Hardware profiles consume hash-bound, versioned mechanism references. Raw capture, source-snapshot, and trial replay checks belong to parameter publication; raw profiler artifacts are not runtime inputs.

Native framework modes use separate, version-matched environments: vLLM Scheduler contracts select `0.11.0` or `0.30.0`, and SGLang execution profiles name their supported version. Install the selected runtime using the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/) or [SGLang installation guide](https://docs.sglang.ai/get_started/install.html). Supply the SDK Python and pinned source checkout required by the selected mode; the 0.30 CLI requires `--vllm-sdk-python`, `--vllm-source-root` and `--vllm-cache-layout`. These dependencies are intentionally separate from the CPU-only requirements because runtime releases and vendor builds have different PyTorch and accelerator requirements. A missing or incompatible SDK is rejected.

## 1. Goals and boundaries

Preserve three kinds of fidelity: explicit request/runtime inputs; the target framework's control-plane semantics; and complete operator execution semantics, including shape, dtype/layout, rank, placement, dependencies, and bytes. Arrivals, lengths, prefixes, acceptance, and routing come from declared workloads, replay, or explicit assumptions, never inferred performance.

Uni³Sim does not load weights, execute numerical tensors, generate real tokens, or reconstruct an unknown deployment from a request trace. Production Forward never reads measured latency, TTFT, throughput, or other outcome columns. Observations belong only to evaluation and cannot select scheduling policies, operator prices, or system parameters.

SGLang's mirror may accept complete `prompt_token_ids`/`output_token_ids` transcripts through `add_request`, exporting `BackendStep.embedding_token_ids` in actual chunked-prefill/decode row order with source `sglang_explicit_request_token_ids`. Length-only inputs remain unknown and cannot mix with value-bearing requests in one backend. EAGLE additionally requires draft/tree state; accepted counts cannot recover token values.

## 2. Two-layer architecture

```text
Workload + model + deployment
    → requests and runtime inputs
    → scheduling/routing
    → logical operator/communication/migration DAG
    → complete KernelSpec + HWSpec
    → unified invocation prediction
    → resource-constrained global timeline
    → completion/token/KV/queue feedback
    → next Scheduler decision and request metrics
```

Serving prices logical work rather than matching a historical profiler name. Input, control plane, workload lowering, Kernel prediction, timeline orchestration, and analysis are separate responsibilities. Completion feedback is essential: predicted service changes subsequent batching, so a fixed offline batch table is a different experiment.

### Virtual forward and storage lifetimes

Model integration may use official, Transformers, or framework forward code with exact configuration/source/branch identity. Adapters replace initialization, weight loading, and primitives while retaining model control flow. New structures extend existing virtual primitives instead of adding a separate DAG or release table.

`src/serving/workload/traced_forward.py` creates only shape/dtype/stride/storage/view metadata. Shared primitive allocation/alias/in-place semantics let `ForwardTrace` derive read/write dependencies and synchronous CPython lifetime from references/scopes. `WorkloadGraph`, `MemoryAllocation`, and `TensorView` provide both performance and storage views. Value-dependent branches need explicit dynamic input or a restricted domain.

Activation peak is the maximum simultaneously live **independent storage** bytes. Views/aliases do not add storage, and caller-held tensors outlive suboperator returns. Weights, persistent buffers, and KV are accounted separately. Startup model/sampler profiling is a different program whose shared objects live across calls; summing component peaks is invalid. Compiled release timing, async completion, and unknown library workspace need independent evidence.

`ServingExecutionContext` passes the selected method unchanged through `lower_workload → KernelPlan → predict_kernel_plan → predict(KernelSpec, HWSpec, method)`. Each Kernel owner decides method support. Launchless mathematical nodes may support Roofline while rejecting Uni³Sim locally; the Serving entry must neither reject all virtual graphs nor silently substitute a method or graph.

### Source scenarios and reference programs

Source profiles declare a specific program; they do not prove the installed provider or historical runtime environment.

| Scenario | Defined behavior and limits |
|---|---|
| Dense Qwen3/Llama `cuda_eager_inplace` | BF16 CUDA eager, H200, PP1/EP-off/MTP-off, greedy; shared rank-local TP/collective primitives. Llama omits Q/K norm and retains its local-variable/`maybe_fused_act_quant` lifetime |
| Explicit vLLM 0.11 dense | `model_execution_version=vllm_0_11_0` with `vllm_0_11_plain_greedy_bf16` covers A800 BF16 Qwen/Llama colocated TP1/TP2, sync eager PP1. Embedding uses native mask→INT64 gather→in-place mask→all-reduce rather than the 0.30 fused CUDA branch |
| CUDA RMSNorm | Leading-axis gaps in Q/K views read original storage; only a nonunit final stride materializes input. Fresh output allocates before the internal copy. ROCm flattening, IR-native, and batch-invariant paths are separate programs |
| `norm_source_batch_invariant=False` | `--norm-source-batch-invariant disabled` binds actual Norm shape/layout/source to the existing lowerer. Missing remains unknown; True is unsupported. False reflects upstream `VLLM_BATCH_INVARIANT=0`, not historical capture proof. Duplicate step declarations reject |
| H200 BF16 GEMM `tiled_reference` | `--bf16-gemm-model tiled_reference` uses existing static tiler/pipeline for homogeneous unbiased BF16 Linear/head, split=1. Preserves graph rows, storage, dependencies, communication, and capacity. Mixed FP32-output routers, quantized and biased paths keep their models. No cuBLAS geometry/provider claim, error-based tuning, or per-case reference selection |
| H200 BF16 experts `triton_reference` | `--bf16-expert-model triton_reference` expands uncaptured 0.30 bundled/default prepared-expert alignment, two GEMMs, activation, complete zero producer, and sum; external routing stays once. Restores per-call Host enqueue and readiness. FULL/PIECEWISE keeps capture aggregates without internal Host preparation; chunk limit 32768 tokens. Compiler/callee workspace/provider remain partial |

Production CLI normalization selects existing virtual sources and declared greedy scenarios only in implemented version/model/hardware domains: ordinary A800 0.11 dense/GLM4 and applicable 0.30 dense/MoE, GLM, K2.6, and V4. Required routing, workspace/indexer history, and physical branch facts remain mandatory. Model source 0.27.1 and CPU Scheduler 0.30 identities stay distinct; other modes retain explicit migration gaps.

Programmatic Serving/typed probes may omit `symbolic_forward_source` for H200 colocated vLLM0.30 with explicit BF16 plain greedy and complete MLA constructor inputs. The common binder selects existing GLM5.2 BF16, official GLM5.2-FP8/5.3, or exact K2.6 programs. It never fills context/batch limits, routing, indexer/workspace history, or per-step tokens, changes sampler/provider, or falls back to a legacy builder. Existing H200 PD pools reuse these same sources for GLM TP8/EP1 or EP8 and K2.6 TP=EP8, with PP/SP/CP/DP1, graph/MTP-off, and both Prefill/Decode roles. Pool-local graph/lowering/storage matches colocated execution; request-scoped KV handoff remains shared.

The 0.30 CPU metadata bridge uses the same complete Request/KV-group handling. GLM/Kimi EP partitions FFN experts only; DP1 does not activate MoE SP, and MLA/indexer cache geometry remains unchanged. GLM reduces MoE partials before the next input norm; Kimi retains its FFN late sum. Common producer-region/ownership/port validation and reported effective TP/EP/SP prevent duplicate model/transfer formulas. Other parallelism, Flash native PD, actual connectors, provider, and normal capacity retain separate boundaries.

### Model-specific source coverage

**Qwen3-MoE.** Qwen3-30B-A3B reuses dense GQA/residual/target/TP code, adding BF16 GateLinear, explicit softmax top-k, and typed experts. `norm_topk_prob` must be a boolean. FP32 weights, INT32 IDs, and INT32 `token_expert_indices` have separate wrapper lifetimes. Explicit routing replaces exactly one former top-k stage; missing selector pricing still rejects complete timing. TP1 embedding materializes the source `input_.long()` INT32→INT64 conversion; TP fused embedding retains its own branch.

**Kimi K2.6.** `DeepseekV2MLAInput` follows 0.30 unfused RMSNorm MLA wrapper semantics: fused Q/KV-A, two norms, Q-B, in-place RoPE, and wrapper-held locals. Unsqueeze preserves stride/offset, including single-row slices. Exact module exclusions keep attention/dense/shared FFN BF16 despite the mixed INT4/BF16 checkpoint; routed experts remain INT4/group32. `DeepseekV2MLP` requires explicit `reduce_results`: ordinary dense down-projection reduces in `SymbolicRowParallelLinear`'s frame, while shared/GLM branches retain partials. Full rank groups and collective config are mandatory.

`DeepseekV2MLADecodeAttention` adds BF16 cache append, token-major Q absorption, FlashMLA concat, latent output/FP32 LSE, V-up `out=`, and O-projection TP reduction under explicit `vllm_0_30_flashmla_bf16`. Logical cache pools and source empty(0) establish ordering only; physical pages/metadata/scratch/strides/launch remain unbound. `bf16_mla_decode_softmax_v1` prices useful latent math and parameter payload; LSE follows upstream `test_flashmla.py`, without compiled-instruction acceptance.

`trace_kimi_k2_decode_forward` selects `vllm_0_30_k2_6_int4_sync_explicit_allreduce` for the full 61-layer all-decode model and optional plain-greedy target chain, reusing residual/model and GLM target/routing helpers. The BF16 gate keeps router factor 1; INT4 expert output applies checkpoint `routed_scaling_factor` before shared-add/late TP sum. CLI requires exact K2.6 config, TP8, constructor limits, and explicit routing; `--target-sampling-source vllm_0_30_plain_greedy_bf16` selects targets. Existing Marlin pricing does not establish 0.30 native expert binding. All 480 Kimi routing calls still reject complete pricing, making whole-model time null/nonzero CLI exit. Prefill/mixed, full worker memory, normal capacity, and actual CUDA providers are separate unaccepted scopes.

**GLM BF16 attention/decoder.** `DeepseekV32AttentionInput` covers the 0.30 source prefix through fused A, full-layer indexer projections, fused Norm/RoPE/cache writes, Q absorption for all input rows, and query preparation. An explicit context keeps original projections/index-Q locals alive while the caller executes the backend. `DeepseekV32Attention` covers CP1/TP8, unpadded single-token all-decode with `declared_causal_topk`: indexer, arena concat, latent sparse math, V absorption, and partial O projection. Shared layers reuse one top-k; all layers share a declared-history 64-head backing arena. Views and `out=` retain real backing. Logical page conversion, third backend result, and callee scratch remain gaps.

`trace_glm_dsa_decoder` uses exact GLM5.2 BF16's 78-layer forward (three dense, 75 MoE), final norm and TP8 reduction, all-decode EP/MTP-off and PP/SP/CP/DP1 with explicit workspace entry state. Dense/shared MLP shares ordinary SiluAndMul; router is BF16 A/B→FP32 C; grouped routing returns FP32 weights/INT32 IDs. The declared modular/nonoverlap, unfused-shared, all-reduce→RMSNorm branch retains O/MLP partials until next input/post-attention/final norm. `SymbolicAllReduceRMSNorm` keeps its reduced local only in its own frame, and equal get_rope arguments share caches. Expert workspace remains incomplete; decoder scope excludes vocabulary/sampling and other unselected branches.

**Official GLM FP8.** GLM5.2-FP8 and GLM5.3 share the exact structure/quantization path, resolving FP8 exclusions by module path. Indexer wk/weights stay BF16; Q/KV-A, Q-B, indexer Q-B, and O reuse block-FP8 Linear. Fused unsharded A may have a final block below 128; scales use `ceil(N/128)` FP32 entries. `block_fp8_fp32_scales_v1` declares FP8 A/B, BF16 C, logical FP32 scales, and 2MNK Roofline, preceded by launchless quantization. Both remain `execution_support=unsupported` for native/Uni³Sim and reject launch/FP32-output conflicts. Source allocation order preserves quantizer→C temporary lifetime; padding/workspace remain unknown.

`vllm_0_30_v32_fp8_sync_explicit_allreduce` selects the full shared 78-layer decoder/target only for matching FP8 checkpoints, never through a BF16 profile. Dense/shared FFN selects BF16/block-FP8 by module path; GateLinear remains BF16 A/B→FP32 C. Routed expert metadata preserves full expert IDs, weight type, FP32 route weights/INT32 IDs/partial output. `block_fp8_swiglu_v1` prices useful W13/SwiGLU/W2/weighted top-k math, without choosing native tile/padding/workspace or borrowing DeepSeek-V3-only H200 launch tables. Explicit assumption identity may share structure, but checkpoint/config/replay identities remain separate. Grouped-route pricing and FFN workspace gaps keep total duration null and normal capacity unaccepted.

The same GLM decoder/greedy chain supports H200 TP1: INT32→INT64 embedding without TP collective, direct final norm, full-vocabulary head, and exactly one rank's workspace history. FP8 target-off comparisons reuse it with explicit per-phase history/limits/synthetic tokens. A complete logical price requires every call priced and a declared sampler; it does not establish native launch/provider or normal capacity.

**GLM padded MTP.** `V32PaddedMTPInputs` declares request order, two independent draft-ID passes, discard state, block size, and sequential/full-logits/greedy conditions, reusing 78 targets and the additional V32 predictor. Target/draft use spec2; indexer threshold 3 makes q2 decode and q8 prefill, while mixed shapes are distinct. Shared V3 helpers retain raw target logits, greedy rejection, padded preparation, pass-0 full top-k with real retained-row gather/copy compaction, and pass-1 reuse of old selection; KV growth does not increase the old count. Final all-reduce/RMSNorm hidden feeds both head and next pass without an extra head norm. TP1 uses fused embedding/EH; TP8 uses sharded embedding/reduction then fused EH. Typed ports/lifetimes remain source-derived.

`SimulationConfig.model_source_inputs_by_step` freezes independent frames and rejects missing frames, duplicate ownership, stale run-level state, or mismatched request/token/depth. The CPU0.30 bridge supports only explicit fixed-count/spec2/sequential/padded/full-logits/greedy GLM through real SpeculativeConfig and target/draft cache owners, including layer 78 latent/indexer storage. It cannot infer initial draft contents or equate logical pages with native allocation; test KV budgets/source Roofline do not establish GPU, provider, or normal capacity.

**GLM prefill/mixed.** Explicit `--glm-prefill-attention-source force_sparse_mqa` binds 0.30 `attention_config.sparse_mla_force_mqa=True`. Without it, previously unsupported multi-token branches cannot be guessed from shape except the separately defined pure-fresh dense-MHA domain below. Reuse source decode-prefix packing to permute IDs/positions/slots and select target rows at packed cumulative-query-end−1, without an invented GPU copy. All rows still perform Q absorption. BF16 latent cache/shared top-k/64-head backing uses each query's causal work.

Prefill indexer reuses Flash FP8/UINT8 gather and same-rank arena; GLM reserves `runtime_max_model_len*40` rows, Flash C4 divides by four, and FP32 scales alias existing storage. Constructor query backing, growth overlap, and layer reuse follow virtual references; decode-only emits no gather. `--upstream-vllm-root` independently audits the two ports and scale alias in constructor/helper AST, not the entire call inventory. CP1 payload gathers full KV once/request, reusing query chunks. Candidate counts≤top-k skip score reads but retain score producers, two INT32 row-bound reads, and full output-capacity index/sentinel writes. Actual chunking, rectangular score backing, general selector, and internal workspace remain unknown; routing still prevents complete price, true Scheduler/provider, or normal-capacity claims.

**Target/sampling chain.** Static `report_model_operator_probe.py` no longer accepts `--symbolic-forward-source`. Explicit versioned sampler plus model/checkpoint type selects implemented GLM, K2.6, or DeepSeek-V3 source; configuration alone cannot infer a backend. TP, limits, tokens, GLM `--glm-indexer-selection-source declared_causal_topk`, and per-rank workspace remain required. Ordinary dense/MoE without sampler remains LM-head-only and fails a full-forward gate. Per-phase typed RuntimeInputs use the same source binder and reject conflicts with old separate parameters.

`--target-sampling-source vllm_0_30_plain_greedy_bf16` or `trace_glm_dsa_forward` includes INT32 query-start row indices, fresh BF16 sample hidden, TP head, default CUDA all-gather, vocabulary slicing, FP32 cast, INT64 argmax, and INT32 output. `SymbolicTPLogitsProcessor` distinguishes rank-major backing from last-axis API order: multiple rows materialize one row-major reorder, one row is a view. Helper backing is released before the cast; all rank producers/shards/payloads/reorder are validated. `full_model_graph_complete=True` describes endpoints, not successful pricing/source/Serving acceptance. ATen index conversion, compiler/library scratch, and metadata producers remain unbound.

Qwen3 TP1 and V4 Flash TP4/Pro TP8 share `SymbolicLogitsRowSelect`/`SymbolicGreedy`; V4 shares TP logits processing. Ordinary request selection uses INT32 metadata even for identity selection, producing fresh output; startup `_dummy_run` NumPy-cumsum INT64 is a distinct program. Qwen retains unquantized local head; V4 multi-request gather adds one reorder/rank, single request no copy. Complete worker lifetimes/capacity/provider remain separate.

Static probes price each call only through public `predict`. Missing optional resource diagnostics add limitations without overwriting a successful prediction. Unmigrated sources stay `legacy_unmigrated`. Any unpriced call yields `serial_roofline_seconds=null` and nonzero CLI exit, never zero-fill or a partial sum. `execution_scope` distinguishes decoder from full-model endpoints; production plan prediction remains fail-fast.

**DeepSeek V4.** Flash/TP4 and Pro/TP8 use exact configs, H200 text-only eager colocated PP1/EP1, with MTP/DSpark/multimodal/other deployments rejected. `vllm_0_30_v4_sync_fp8_ds_mla` requires declared eager metadata, modular routing, DeepGEMM choice, constructor limits, plain greedy, and explicit/synthetic token IDs; long scored indexer needs `--v4-indexer-selection-source declared_causal_topk`. Scheduled row count cannot infer constructor capacity. SWA, C4/C128 compression/indexing, MHC, FP8 projections, and MXFP4 experts lower through common KernelSpec/predict. C128 selects completed states; compressor emission uses differences of completed windows and emits nothing without a new eager boundary. Seven cache-lifecycle groups distinguish windows, append streams, and FP32 partial state. Page-aligned charged/raw accounting is not native allocator equivalence. Kernel evaluation's observed split plan and Serving's named analytical split budget have separate accuracy evidence. Without V4 hardware observations, TTFT/TPOT remain projected; residency and Marlin transfer remain outside the validated scope.

The 0.30 cache factory creates target-layer SWA/compressed/indexer/FP32 groups from exact `compress_ratios`, without draft-tail cache, under PP/CP/DP1, EP/MTP-off, BF16 compute/fp8_ds_mla. Native allocator/placement receives rank budgets and actual group/arena objects; wrong TP, missing policy, or legacy GLM-only input rejects. Source weights/buffers and complete normal policy remain independent requirements.

**Kimi K3.** `vllm_0_30_k3_mxfp4_sync` retains 93 text layers: 69 KDA, 24 NoPE MLA, Attention Residual, latent MoE, vocabulary, greedy, BF16 compute, MXFP4/group32 experts, BF16 convolution/FP32 recurrent state and MLA KV. A real CPU0.30 Scheduler handles admission/decode/free with per-layer/rank KDA indices and MLA slots. Initial complete requests cover H200 colocated TP1 PP1 or PP16, fresh prompts≥512, 1–8 outputs, total within constructor `max_model_len=640`. TP1 is a test-budget mathematical chain, not one-GPU capacity. PP16 assigns five layers to the first two/final stages and six to the other 13, with adjacent P2P. TP16 has weight/mixed-cache capacity reporting only, without complete request/source-memory support. Non-64 KDA tail chunks price full tiles; detailed masked lanes, longer chunked prefill, multimodal/MTP/prefix combinations, and real providers remain unvalidated.

```bash
MIDLAYER_VLLM030_PYTHON=/path/to/.venv-vllm030/bin/python \
MIDLAYER_VLLM030_SOURCE=/path/to/upstream/vllm/v0.30.0 \
python scripts/evaluation/serving/report_kimi_k3_serving.py \
  --layout pp16 --prediction-method midlayer \
  --prompt-tokens 512 --output-tokens 4 \
  --output experiment_logs/serving/kimi_k3/pp16_minimal.json
```

The unified CLI selects the same K3 source. PP16 requires `configs/runtime_memory_policy/nvidia_h200/kimi-k3-pp16.json`, `--prefill-pp-size 16 --pipeline-ranks-per-node 8`; price cross-node links from placement. Both layouts require H200/colocated/EP-off/uniform MLA/analytical launch/version-matched greedy. Repeated `--prompt-tokens` supports a few concurrent requests; `--layout tp16 --capacity-only` selects the separate capacity policy. `--source-memory-only` reports known activations. Complete requests require every output, no unpriced calls, and positive time; report variants, critical paths, FLOPs, simulator wall time, and unavailable aggregate logical bytes rather than zero. Cross-rank/call duration sums are diagnostic, not stage latency. Routing is seeded assumption and feedback uses lengths, not numerical generation.

PP16 capacity itemizes weights, native mixed arenas, KDA/Marlin workspaces, P2P, worker, allocator, fragmentation, and framework. KDA includes q/k, w/u/kg, v_new, chunk_h, FP32 final state/g, Aqk, beta, metadata, and a varlen chunk upper bound `floor((tokens+63*requests)/64)`. For 2048 tokens/two requests, 33 chunks give 538,706,472B KDA and 1,202,504,400B KDA+Marlin. Persistent `worker.intermediate` hidden/residual uses `ceil(start_layer/12)` AttnRes blocks in framework; transient block-residual uses `ceil(end_layer/12)` in activation; the largest rank has 264,241,152B for each. Replace the transient witness with its constructor bound without double counting; communication reserve remains 128MiB. Complete policy/subitem validation establishes a bounded source budget only, not real H200 capacity or accuracy.

**SGLang DeepSeek-V3.** `sglang_0_5_19_deepseek_v3_decode_fp8` declares H200 TP8 Cutlass FP8, Triton grouped top-k, FlashMLA BF16, BF16 absorption BMM, and unfused communication. `_flashinfer_ar_norm` selects the explicit fused all-reduce/residual/RMSNorm branch. Profiles retain INT64 IDs, BF16 A/B→FP32 router, separate FP32 weights/INT32 IDs, shared-expert extra slot, and BF16→FP32 logits. Branch metadata does not prove installed/executed providers.

`sglang_0_5_19_deepseek_v3_target_fp8` and its fused-norm variant add ragged/eager/CP1/no-hook fresh-MHA prefill, then ordinary decode. Shared concat/cast feeds O projection; source INT32 cumsum→INT64−1 materializes only last-request rows before head, without vLLM output-copy. Default scope ends at logits; explicit `sglang_0_5_19_plain_greedy_fp32` declares all-greedy CUDA with custom processors/logprob/grammar/return-mask, NaN checks/sanitization, and token-sync disabled. It produces fresh INT64 argmax directly from FP32 logits, without vLLM INT32 conversion or numerical execution.

For pure prefill with `0<sum(prefix)<8192`, the target source uses absorbed MLA with ragged queries and latent O without LSE. Fresh prefill uses MHA. At prefix≥8192 with total KV within declared chunk capacity, `SGLangPrefillInputs` must supply complete combined-KV INT32 indices whose suffix matches slots; source writes cache, separately gathers latent/RoPE, expands full KV, and runs one causal MHA without public LSE/merge. Larger KV needs explicit per-chunk indices: causal suffix then noncausal prefix chunks, materializing O/LSE merges. Empty prefix rows preserve zero BF16 O and negative-infinity FP32 LSE; merges still read full states. Actual req_to_token producers/providers/compiler traffic and unmigrated mixed/EAGLE branches remain unclaimed.

`SGLangSchedulerMirrorBackend(request_token_transcripts=...)` retains an explicit source ID and ordinary `add_request(RequestSpec)`. Missing transcripts stay length-only; missing values in transcript mode reject. A separate single-request topk1/steps1 EAGLE domain uses `SGLangEagleChainState` bonus/prior draft/target argmax/tree/accept path/draft slots. Prefill runs target+NextN extend; verify runs two target rows/tree verify/bonus/extend. Prefill bonus is INT64, later bonus INT32; accept_lens includes bonus, Scheduler accepted count excludes it. Exact-step `model_source_inputs_by_step` validates request order/candidates/acceptance/transcript and passes frames unchanged through `BackendStep.model_source_inputs`. This is declared synthetic source simulation, not actual SGLang, normal capacity/provider, larger batch/tree, or graph acceptance.

### Source-memory accounting and capacity

The source observer consumes the **same WorkloadGraph** later lowered; it does not rerun a model for memory. Optional `source_activation_memory` in components/steps records represented rank-local peaks and witnesses; summaries retain coverage, profile, and one maximum/pool/rank. Full/streaming/CPU-worker modes share the source. Missing virtual source means absent/None, not zero.

External canonical/Production persistence strips request content, step series/IDs, and allocation/event witnesses while retaining coverage/profile/rank peaks; in-memory results retain details. Snapshots remain `complete=False`, `normal_capacity_accepted=False`: they exclude unknown workspace/compiler/async runtime, actual KV allocation, and cross-forward async peaks. Independent rank maxima cannot be summed into one card's capacity.

Flash `--source-workspace-initial-capacity-bytes` declares each TP rank's entry single-lane unlocked arena; unknown remains None, four explicit zeros mean a caller-declared cold scenario rather than warmed workers. Preserve grown capacity across steps, including large-prefill→small-decode transitions. Stateful CPU tails stay sequential instead of forking independent histories. Long-C4 indexer prefill shares this manager, reserving FP8 gather/UINT8 scale constructor capacity at ten times max context, with an FP32 alias. Empty compression groups retain reservation without nonexistent traffic; decode-only has no gather. Shared/layer-reused arenas are not multiplied by 43 or converted from capacity to traffic; external scores/selector storage remain unknown.

`report_activation_memory.py --source-only --program symbolic_startup` selects Flash TP4/Pro TP8 profile `vllm_0_30_v4_sync_no_metadata_pytorch_sort_fp32_noise`, requiring TP, max context/batched tokens/requests, DeepGEMM branch, indexer-logit limit, per-rank arena history, routing source/seed, and H200 config. The existing V4 forward adapts only no-metadata padding/reservation/zero-output behavior. Common worker/sampler/storage scanning reports activation peak, strict-profile total peak, retained end storage, and transient headroom, without real tensors/GPU. Storage-only nodes cannot become a completely priced request or normal V4 capacity policy.

Startup ledgers mark profile boundaries and separately compute total peak, end-live bytes, and headroom. Sampler cache created/retained during the program contributes both to intermediate peak and end-live bytes. Triton sampler growth, views retaining backing, shared lookup/split buffers follow references; SM count comes from HWSpec, not device queries.

`run_serving_simulation.py --activation-source-profile` invokes `resolve_symbolic_kv_capacity_plan` through the real CPU0.30 allocator/Scheduler without test block overrides. Current source domain is BF16 Qwen3/Llama, A800/H200, TP1/TP2 PP1, MTP-off, eager CUDA startup; model/hardware/parallelism/constructor limits must match policy. `kv_capacity.source_lifetime_projection` retains schema2 policy and rank ledgers. Unknown workspace/runtime and missing comparable startup logs block normal acceptance even with positive blocks/completed requests. Llama preserves no-QK-norm/CUDA-RoPE and early gate-up release; existing framework reserve below represented residency fails instead of filling the difference. TP2 runs independent rank profiles/caches; all-gather leaves full-vocabulary sampler storage per rank, so TP1 peak cannot simply be halved. Declared fresh collective output does not cover transport workspace/provider.

Source-only policy projection accepts explicit base/derived policy paths. `project_symbolic_activation_policy` separates end-live framework storage from the simultaneous nonresident activation peak, charging arenas once; this is a declared source budget bound, not observed unknown-runtime coverage. V4 requires `--framework-source-epoch before_profile|after_profile`: before-profile validates initial objects then adds represented retained growth once, after-profile validates final residency and retains the base value. Qwen uses after-profile. External reserves/provenance stay intact and candidates remain `unverified_analytical_assumption`; no default/baseline overwrite or normal/log-10% upgrade follows.

Qwen/Flash/Pro share `VLLM030WorkerStorage` and `vllm030_worker_storage_bytes` for known text-only CP1/MTP-off runner buffers and InputBatch sampling arrays. CpuGpuBuffer counts device storage once, without charging CPU/pinned/NumPy to HBM. Optional `--upstream-vllm-root` audits constructor shape/dtype/forwarding before candidate export. Block tables, optional multimodal/speculative state, and other runtime memory remain outside coverage.

### Compilation reuse and fast execution

Dynamic forward integrates/compiles a model; simulation iterations reuse source structure while rebinding current shapes, KV, routing, and slots. Validated unpadded single-token dual-rank programs use immutable binding frames; representative nodes materialize current KernelSpec while complete public node tuples remain available. Validated 0.11/MACA0.24/Ascend0.18 warm decode binds directly to cold-compiled recipes; other structures/padding retain full compilation. Reuse KernelPlan DAG only while call set/hardware/resources/dependencies remain unchanged; recompile changed source/template/bindings/branch.

Exact node contract plus HWSpec keys bound LRU caches for Norm/activation/RoPE/KV recipes. Current ports/graphs/branches are still validated each step; no timing result is cached or call removed. H200 FA3 GQA additionally caches immutable complete-source KernelSpec recipes with all shapes/source attributes/control values **and types**, HWSpec identity, and SM count; mutable inputs still run the original factory. The 256-entry bound limits simulator metadata, not hardware capacity/accuracy.

H200 GLM 0.30 BF16/FP8 fast TP8/EP1/MTP-off single-token decode records full source once, then binds a real one-layer prefix and all MoE routes while retaining all layers/rank communication/storage. First full lowering establishes the DAG; validate every routing packet and group only exact ordered histograms/Attention-KV/embedding shard contracts through the ordinary dispatcher. Structural changes recompile. Per-step times still recur normally; caches stay run-local and never persist prediction LUTs. Kimi K2.6/K2.7 TP8/EP1 fast decode uses the same source-binding path; prefill/mixed/unsupported changes retain full source.

Closed bindings hold read-only storage/events and separately cache whole-lifetime versus strict-capture peaks. Initial native projection validation retains immutable scalar/tuple facts; every frame validates current KV/slots/backing. Public exports preserve IDs/field equivalence without promising Python object identity. Mutating an old export cannot mutate templates; resubmission still validates. Unknown mutable nested metadata uses full projection, explicit dynamic metadata is copied each frame; changed allocations/events/windows/templates revalidate.

`fast` retains Scheduler, full logical graph, lowering, launch, chosen `midlayer` predictor, timeline, source identity/lifetime statistics, and all requests/tokens/major compute/communication. It omits separate per-call `TraceMemoryContext`/bindings/access programs and sends identical KernelSpec/HWSpec directly to unified `predict`. Detailed memory-access diagnostics require `accelerated`. Fast is not Roofline and never reads production durations or previous-run predictions. Source-recipe implementation identity is checked at run boundaries and all dynamic fields rebind.

Fast's pinned 0.11 CPU Scheduler runs in process under the same device guard, native plan/update, capacities, and source validation, eliminating RPC serialization without removing Scheduler work. Other constructors/accelerated retain isolated processes. `execution_mode=fast` records this choice; shutdown checks no CUDA context/handles and removes temporary model config.

`--execution-workers 1` is sequential. The standard report exposes `--serving-execution-workers`; source-validated independent tail decode after the final arrival may run in parallel for colocated Qwen3-MoE TP2/EP2 or Qwen3/Llama TP1/TP2 PP1, depth two, Graph, and full Host timing. The parent reconstructs FIFO Host/capture history and audit chains in Scheduler order, preserving cross-batch pending input and stable full run-local source namespaces. Workers change simulator execution, not hardware capacity. Other geometry/history remains sequential or uses its existing supported route.

`model_profile_layer_mean_v1` preserves SHA256-weighted sampling without replacement, original random stream/floating summation/layer provenance. Float profiles use `profiled_routing.c` for the same selection loop while Python constructs exact integer weights and validates conservation; compilation is temporary, not a cross-run cache. Other types retain Python; tests compare exact choices.

### Current capability scope

Executable support, mechanism origin, and validation evidence are independent.

| Axis | States |
|---|---|
| Origin | Native framework state machine; captured normal input; modeled routing/lowering/timeline |
| Coverage | Supported declared combinations; partial listed combinations; unsupported missing semantics/physics |
| Evidence | Validated against independent references/conservation/hardware; projected without target observations |

Native batching does not validate downstream timing. Adjacent supported cases do not establish an untested combination.

| Capability | Scope and remaining boundary |
|---|---|
| Production/synthetic requests | Same normalized request/Scheduler/lowering chain; production success-only arrivals/lengths/prefix, no latency or inferred routes; synthetic distribution remains explicit |
| vLLM Scheduler | Pinned CPU continuous batching/chunking/KV/finish/preemption/recompute without weights/tokenizer/GPU executor |
| Prefix | Trace-hit lengths by default; token mode uses native content state under its explicit domain |
| Routing | Multiple P/D `least_queued_tokens`, immediate reservations; no cache-aware inference |
| Deployments | Colocated, PD, standalone AF, PD+AF slices; instances own exclusive device sets |
| TP/PP/EP/DP/SP/PCP | Model/version-specific sharding and communication; provider/numerical/pipeline equivalence separate |
| MTP | Explicit propose/verify/acceptance/handoff; assumptions remain projected |
| Communication | Explicit collective/P2P physical scenarios; configured development mappings limit accuracy claims |
| Host KV | Lifecycle replay and narrow native SimpleCPU connector domain; not generic eviction/prefetch/SSD/controller support |
| Host CPU | Router dispatch and declared runtime lanes; no invented universal CPU residual |
| Models/hardware | Dense/MoE/MLA/MTP slices on A800/C500/910B/H200 inherit Kernel limitations; other research hardware and combinations are not implicit |
| Metrics | TTFT/TPOT/E2E/distributions/throughput/goodput/queues/resources/critical paths from one timeline; success-only traces cannot establish availability |

### 2.1 Runtime Graph padding and token domains

Distinguish Scheduler logical tokens, padded model-input rows, and operator active rows. Production consumes actual runtime-resolved `capture_sizes`, normalized uniquely/ascending, then selects the smallest sufficient bucket. Observed `padded_total_tokens`/`capture_bucket_tokens` remain evaluation holdout, never step-wise Forward inputs. Eager preserves unchanged unpadded work.

Configuration supplies `configured_mode`, `uniform_decode_query_len`, capture sizes, and compiler-excluded `piecewise_eager_workload_kinds`. A common table resolves NONE/PIECEWISE/FULL/FULL_DECODE_ONLY/FULL_AND_PIECEWISE by uniform query length, with NONE when disabled/above the largest bucket. Attention/KV remain logical; head uses real selection; offload uses logical block ranges. FULL produces one Host replay per pool/rank lane covering all expanded occurrences. PIECEWISE splits at declared eager kinds and replays contiguous intervening occurrences. Source/lane/repeat/coverage mismatch fails. Defaults and splitting ops are runtime configuration, not hardware/version branches.

CLI: `--graph-mode`, `--graph-uniform-decode-query-len`, `--graph-capture-sizes`, `--graph-piecewise-eager-workload-kinds`. Nonzero padding covers declared standard-GQA/MTP-off colocated dense and TP=EP2 MoE slices; other combinations fail at workload/lowering boundaries.

Canonical profiles are A800 FULL-uniform/PIECEWISE-nonuniform with capture increments of eight above eight through 512; C500 FULL/PIECEWISE with steps of eight below 256 and 16 from 256 through 512; 910B FULL/FULL with the C500 set. Audited 0.11 A800, 0.10.2 C500, and 0.18/plugin 910B use the same ceiling/actual-token/logits-compaction mechanism. Actual capture sets remain bound runtime input. 910B dummy-query overhead remains partial.

Padded nodes require `graph_logical_tokens`, `graph_model_input_tokens`, `graph_runtime_mode`; compute specs retain `model_input_shape`. Reject missing fields, physical<logical, padding in NONE, or insufficient physical shape. Unpadded serialization does not manufacture default domains. Attention instead requires complete logical and `model_input_q_lens`/`model_input_kv_lens` plus mode.

| Family | Active prediction | Launch/storage/payload |
|---|---|---|
| Embedding/Gather | Logical elements/rows/indices | Physical captured input |
| QKV/O/MLP/router GEMM | Logical M | Physical M/grid/workspace |
| Norm/RoPE/activation/elementwise | Logical work | Physical launch/temporary buffers; disable worker schedules assuming every padded slot valid |
| Attention main/combine | Actual logical request lengths/masks/KV | Physical batch/lengths/grid/split; padded requests excluded from source CTA work |
| KV write/offload | Logical slots/block ranges | No dummy KV expansion; sentinel tails do not write |
| MoE | Logical histogram = tokens×top-k | Physical recipe/grid/temporary backing |
| TP/EP collective | No dummy useful FLOPs | Captured physical bytes and matching producer domain |
| PP/AF transfer | Logical producer identity | Physical activation/hidden/residual bytes, destination gather and producer checks |
| Final norm/head/sampling | Norm logical rows; real selected requests/tokens | Norm physical backing; head has no Graph row padding |

Launch provenance remains separate: `injected_launch` consumes versioned discrete facts; `analytical_launch` derives static/HWSpec constraints and is not per-shape observed source-exact A800/C500 GEMM. Available 910B CANN typed schedules retain source identity; missing combinations cannot borrow adjacent launches. Every helper/reducer/materialization uses physical ABI while prediction restores active work.

Representative checks include A800 decode 66→72 FA grid `[1,72,4]`, C500 36→40 main/combine `[1,80,9]`/`[80,1,1]`, and 910B dense/MoE 11→16. C500's head_dim64 dense fixture remains unsupported because its current source-backed FA selector admits 128 only. Above-limit dispatch becomes NONE, including 910B 747 tokens beyond 512. The 910B Llama TP1/QPS4 `vllm:120` has one decode plus ten prefill tokens, FULL16, five dummy rows, two head rows, and actual Attention lengths `((1,10),(33,10))`. Its 2067-step holdout, including 1454 padded steps, matched mode/physical/bucket without inventing dummy KV work.

## 3. Workloads, models, and evidence

### 3.1 Request workload

Minimum input is arrival, prompt/output lengths, optional prefix hit, and explicit runtime fields. Sensitive content stays local. Both production and synthetic streams use identical downstream semantics. Success-only replay answers performance conditional on those requests; it cannot estimate failure/timeout/availability. Capacity studies vary arrival rate while retaining shape distribution rather than calling one replay peak capacity.

For prompt P, output O, trace hit H, require P>0, O≥0, 0≤H≤P. P−H is logical miss, not necessarily executed work because blocks/last-token/recompute matter. O=0 is valid: keep prompt work and mark TTFT/TPOT inapplicable rather than zero.

ShareGPT schema-v3 cases contain `requests.csv`, `metrics.json`, `scheduler_steps.jsonl.zst`, `alignment_steps.json.zst`, and `manifest.json`. Requests/metrics/Scheduler content comes from one profiler-free trace-on run passing the same-config four-metric 3% observer gate. Use actual HTTP submission arrivals relative to the first request and that run's TTFT/TPOT/E2E/throughput. Scheduler records complete boundaries, requests, tokens, and progress. Both compressed files use level-1 zstd; manifests bind compressed and canonical-content hashes, run ID, and instrumentation version. Alignment retains only representative prefill/decode/mixed **structure**, recaptured when operator/launch order changes, not timing alignment.

Target GLM/Kimi alignment requires caller-supplied canonical cases through `--target-model-source-root`; private Production Trace datasets are excluded from this distribution. Schema-v4 partial `sharegpt_alignment_steps` records source/environment/hooks/tensor contracts, production-size operators and complete KernelSpec, explicit phase/layer/rank/differences/omissions/support. Unexecuted mixed is `not_observed`; no duration or false launch-capture claim is allowed. Evaluation alone consumes it. Candidates go to `profiler_output/target_model_alignment/` before primary-reviewed publication.

V1 async step intervals can overlap. Each boundary is step-monotonic, with `step_start ≤ schedule_end ≤ update_start ≤ step_end` within a step; conversion must not serialize/crop real overlap. Lightweight wrappers buffer records and write after canonical completion without external profilers. Raw/candidates are separate from validated formal dataset inputs.

**Scheduling Trajectory Replay** fixes canonical step order, batch membership, and token progress, predicts each step with Forward, then reconstructs TTFT/TPOT/E2E/throughput on that same trajectory. Serialized measured service is comparison-only, recorded as `forward_duration_observation_consumed=false`. Dense needs no routing; MoE requires separately supplied expert-frequency profiles with the same assumption/provenance as Serving evaluation, not claimed step observations.

```bash
python scripts/evaluation/serving/scheduler/run_scheduling_trajectory_replay.py \
  --case ascend_910b/qwen3-4b/tp1_qps16 \
  --host-launch-profile configs/host_timing/host_launch_profiles/ascend_910b.json
```

### 3.2 Model semantics and fixed target-model evidence

Model configs provide layers/hidden/heads/experts/top-k/MTP/weight and compute types, never hardware time. Dense/MoE/MLA/MTP retain distinct real dataflow. Exact source/checkpoint identities are maintained by the existing canonical provenance and validation runner rather than duplicated hashes here.

| Target | Fixed structural/control domain | Capacity and timing boundary |
|---|---|---|
| Kimi-K2.6 | 61 text layers, first dense then MoE, 384 routed+one shared expert, top-k8, MLA, BF16 compute, INT4/groupwise routed weights. Fixed A800 TP=EP8/PP2, 16 ranks/eight per node, DP/CP/SP1. vLLM0.11 CPU Scheduler with 0.27.1 model semantics; no MTP/offload/non-MLA layout. Prefill KV-B and decode Q-down/MLA/V-up remain distinct; compressed KV has 576 elements. Symmetric TP uses a shard representative; MoE expands eight ranks then one TP sum | BF16 plus symmetric INT4 packed weights/BF16 scales, 90% HBM, explicit 2GiB reserve, native allocator: 64,745 blocks / 1,035,920 tokens. This reserve is projected, never a debug production override. A800 BF16/Vector/canonical launches plus source FA2/grouped-MLA/Marlin/router are projected; no production-weight vLLM GPU accuracy claim or C500/910B fallback |
| GLM-5.2 | 78 layers, first three dense, 256 routed+one shared, top-k8, MLA/DSA/full/shared indexer, BF16 and routed scale2.5. Fixed A800 TP=EP8/PP4, stages `[0,18)`, `[18,38)`, `[38,58)`, `[58,78)`, DP/CP/SP1; stage starts at full indexer. CPU0.11 with 0.27.1 model source, no offload. Strict sparse/indexer/FP8 index-cache/MLA/Norm/RoPE/router/expert specs; launch=None where unknown | BF16 stage-local weights and both MLA/full-indexer cache, 90% HBM/2GiB reserve/native allocator: 65,586 blocks MTP-off, 57,296 with terminal draft embedding/layer/head. MTP proposal/acceptance/source/seed explicit and projected. DSA/FlashMLA upstream needs SM90/100, so SM80 A800 timing is projected math, not runnable upstream CUDA |

Kimi source binds the official `text_config` and exact moonshot checkpoint, excluding the vision tower. GLM uses its official config and cross-layer shared-indexer source; audited 0.16 registration was not sufficient, hence fixed 0.27.1 semantics. `validate_target_model_blocks.py` checks vLLM/Transformers/model identities before tiny CUDA tests. Historical tiny-model validation used Python3.12.9/PyTorch2.7.1+cu126/Transformers5.10.2. BF16 three-token prefill plus one-token decode produced logits `(1,3,32)`/`(1,1,32)` and cache length4 for tiny GLM/Kimi, covering dense/MoE/shared experts, GLM full/shared indexer cache, and Kimi YaRN control flow. Transformers expands K/V before DynamicCache, so compressed MLA/FP8 index-cache/decode absorption remain source/manifest validation. vLLM0.27.1 requires PyTorch2.13; the tiny smoke does not claim its CUDA execution or production weights.

`target_model_support_matrix()` provides model×phase×layer-type×validation-level; production-shape manifests cover the 12 dense/MoE prefill/decode/mixed combinations. `model_semantics_validation` compares full request/step identities, logical kind/shape/dependencies, source nodes/KernelSpecs/launches, and evaluation-only latency tolerances. `structural_lowering_manifest()` exports hashes, graph/edges/cache effects/mappings/support without prediction. No measured TTFT/TPOT/kernel duration/launch observation enters ModelConfig or production selection.

### Streaming retention and simulator measurements

External canonical/Production requests use `--step-retention auto` to select `streaming_aggregate` where supported: one colocated pool, external requests, no lifecycle replay. Each step still constructs, lowers, predicts, timelines, hashes its ordered contract, then releases IR. `full` is available for small audits; unsupported explicit streaming fails. GLM MTP retains acceptance for conservation. After final arrival, supported independent work may project in workers; the Scheduler still decides/updates in original order, and the parent aggregates duration/hash in bounded windows with worker recycling. Schema13 retains transfer digests/counts, metrics, token/call counts, cache/parallelism, and wall time but no full step/resource/critical-path artifacts.

Explicit `Simulation`/standalone-AF `MultiSimulation` Host-KV lifecycle streaming retains transfer steps until completion validation/tier commit; other steps use compact timelines. Schema16 adds completion hashes, bytes/counts, and final tiers. Fixed HBM/Host prefix proportions are whole-block scenario assumptions; both copies after LOAD count until request release. They are not native victim/LRU/future reuse. Generic CLI auto/replay boundaries remain separate.

Each case records simulation monotonic start/finish plus wall time. Matrix `simulation_batch_timing` spans earliest simulation start through latest finish, separately from outer `evaluation_wall_time_seconds`. Overlapping postprocessing prevents subtracting summed phase time from outer elapsed. `compare_serving_execution_modes.py --baseline ... --detailed ... --fast ... --output-dir ...` compares saved completed metrics/conservation/cost without rerunning or reusing old predictions; optional `--supplement` adds source gates/evidence/benefit bounds. One invocation may share parsed results for validation/plots/reporting while retaining original files/checks; complete report cost belongs in outer wall time.

Optional `--state-audit-output` / `--serving-state-audit` writes a duration-free sidecar of rank-local logical calls, communication/bytes, source storage domains/capacities/allocation-free counts/state-chain hashes, and transfers. Every step/source/count must be present. It disables immediate projection; parallel tails return snapshots so the parent reconstructs the ordered chain. Audit cost belongs to that run, and audit runs cannot supply unaudited speed denominators.

Length-only CPU Scheduler requests explicitly disable content hashing and use compact equal-value dummy sequences, materializing only requested slices while preserving exact indexing/append/extend. FCFS can inspect at most its next `max_requests` waiting prefix, so run-local deferred materialization avoids constructing invisible long-tail native requests. Arrivals/order/preemption reinsertion and each real decision remain unchanged.

## 4. Scheduling, routing, and request state

### 4.1 Scheduler and pipeline depth

The loop admits currently arrived/migration-ready work, calls each real Scheduler under sequence/token/KV budgets, lowers/prices it, schedules dependencies/resources, advances the earliest event, updates tokens/KV/queues/completions, and repeats. The CLI emits progress every 100 steps without changing result/API behavior.

`scheduler_pipeline_depth` counts decided batches not yet applied through executor-output update, not tokens/blocks/PP stages. Formal inputs require the integer and raw `max_concurrent_batches` evidence; no Forward default is supplied. Each pool owns a slot ledger and serialized device-ready time. Acquire on schedule submission, release only after that batch's update completes.

Pinned A8000.11/C5000.24/910B0.18 expose executor capacity to EngineCore's bounded batch deque. While space exists, schedule then enqueue execution; workers dequeue/execute serially, so Host futures may overlap without overlapping same-pool device steps. When full/no new work, wait the oldest FIFO output and update Scheduler. A800/C500 async depth is two, otherwise PP size; 910B uses PP size if PP>1, else async2/sync1 and separately handles execute/sample futures/aborts. Source snapshots/manifests own exact identities.

Events are `schedule_start → schedule_end/slot acquire → device_queue_wait → device_start → device_end/completion → update_start → update_end/slot release`. Independent reference fixtures compare ledger/FIFO/update behavior. PP1 async depth2 is observed; depth1 and expanded depth3/4 are explicit counterfactual sensitivities, never canonical runtime evidence.

H2000.30 single-rank `vllm_executor_backend=uni` invokes Host `run_method` on EngineCore before its async device-output future. Next scheduling waits Host return, while device may overlap next Host submit; FIFO waits only its own output. Canonical executor-class evidence selects this discrete setting without durations. Multiproc enqueue need not wait worker Host; worker Host stays serial and output handling asynchronous. Independent workload/lowering/prediction can prepare in existing CPU workers, then FIFO supplies actual starts to the unchanged timeline. Unsupported versions/multirank UniProc reject.

KV blocks, max sequences, and max batched tokens are independent budgets. Derive capacity from HWSpec HBM, sourced utilization, rank-local weight placement, and explicit activation/Graph/communication/allocator/fragmentation/framework reserves, then use the pinned **native** KV allocator. Debug/conformance block overrides are never production defaults. Measured available bytes/block counts are evaluation-only equality checks, never overrides/defaults.

Runtime policy evidence pairs are exact: `verified_uniform_across_pool_ranks` + `all_non_kv_reserves_for_bound_runtime` means `verified_runtime_evidence`; `assumed_uniform_across_pool_ranks_unverified` + `reserves_inherited_unverified_not_complete_for_bound_runtime` means `unverified_analytical_assumption`. Mixed/unknown pairs reject. Rebound AF reserves remain unverified; completion cannot promote them and verified ShareGPT evaluation rejects them.

### 4.2 Prefix semantics

For block size B, prompt P, and trace logical hit H, reusable native prefix is:

$$
C=\left\lfloor\frac{\min(H,P-1)}{B}\right\rfloor B,
\qquad P-C=(P-H)+(H-C).
$$

Distinguish trace hit, reusable whole blocks, boundary/last-token replay, and preemption recompute. Without identities, H is a routed system-level oracle; it cannot establish local affinity/sharing/refcounts/eviction.

Enable real-content replay with `--enable-prefix-caching --prefix-cache-mode token_ids` and trace prompt/output ID field options. JSONL uses integer arrays; CSV JSON-array text. RequestSpec tuples must match lengths and nonnegative vocabulary bounds. Positive output needs the complete real transcript; zero-output may omit it. `prefix_hit_tokens=0` means no injected trace hit, not observed zero. Native Scheduler derives resident whole-block hits and still recomputes the last prompt token. First content hits use `content_prefix_hit_tokens`, separate from trace hits; preemption does not count them twice. Reports expose requested/effective caching, aggregate hits and actual prefill, without token content.

Current content mode uses sync CPU0.11 and one FullAttention/MLA group; model/parallel/deployment restrictions still apply. Validated slices include dense TP2×PP2, Qwen3-MoE, 1P1D/2P2D, GLM4 MTP target/proposer, and official 61-layer DeepSeek-V3 H200 TP1 colocated MLA with explicit `vllm_0_11_deepseek_v3_sync_fp8`/`vllm_0_11_plain_greedy_bf16`, source limits, Roofline, and test KV budget. These do not validate providers or normal capacity. Caches are per instance; PD preserves token identity and counts transferred KV separately from D local hits. 0.30/V4 mixed KV, sliding windows, async, closed-loop admission, SGLang, and external tiers/replay remain rejected combinations.

MTP content needs exact-step `mtp_draft_token_ids_by_step` / `--mtp-draft-token-ids-by-step`: `<pool>:vllm:<index>` maps every proposer batch request to exactly speculative-depth vocabulary IDs, including terminal/prefill proposer steps. Accepted draft prefixes must match output transcripts. Missing/inconsistent IDs reject; counts cannot invent values. PD drafts belong to D, with P MTP disabled. GLM4 API validation uses explicit test capacity; Production CLI still rejects missing complete MTP draft-weight residency rather than bypassing with debug blocks. Reports expose frame/source counts only. Supplying IDs while leaving trace mode permits a no-content-cache control.

A three-request demonstration with 32-token prompts, two identical prefixes, block16, and two outputs reduces total prefill from96 to80 by a16-token second-request hit. This work reduction is not a TTFT/E2E speedup ratio. Use actual model-tokenizer transcripts for real workload evaluation.

### 4.3 Multi-instance routing

`least_queued_tokens` sees only current unfinished work. Stable tie-break and immediate reservation prevent simultaneous arrivals observing a stale queue. PD performs ingress→P, then P→D when prefill/KV transfer permit; result provenance records policy/load/reservation semantics. This is modeled routing, not recovered production routes. Other policies need explicit observed-state definitions.

An optional shared serial CPU dispatch resource delays each route before destination admission. Zero service explicitly omits the mechanism rather than asserting no CPU cost. Nonzero service needs an independent mechanism or labeled projection, never an E2E residual. Generic Scheduler Python/sampling/output costs must not be hidden in this dispatch price.

### 4.4 Runtime-unknown inputs

Expert distributions and MTP acceptance require bound replay, versioned independent distributions, reproducible assumptions, or bounds/sensitivity. They cannot be calibrated from target duration. Preserve `proposed=accepted+rejected`, `verified≥accepted`, and `emitted≤accepted+terminal token`; proposer/verify/remaining outputs/handoff share one state machine. Fixed-count acceptance without observations is projected, not predicted model quality.

## 5. Deployment, communication, and migration

Deployment declares instance roles/hardware/capacity, TP/PP/EP, links/transfers, and Scheduler limits. Reusing a workload across configurations compares systems; routing/scheduling outcomes are not intrinsic trace properties. Each instance owns its device set, without same-card multi-instance competition.

Colocated keeps prompt/decode/Attention/FFN local. PD transfers prompt KV before Decode. AF transfers hidden/residual at layer-role boundaries. PD+AF models both transfer classes separately, not a total-latency multiplier.

Standalone AF behavioral validation covers A800 Qwen3-4B/Qwen3-30B-A3B AttentionTP1+FFNTP1, Qwen3-32B TP2+TP2, full ShareGPT QPS16, and a four-pool TP1 PD+AF slice. Routes are one-to-one inverses and conserve requests/tokens/ranks/bytes/dependencies. Asymmetric TP, shared FFN, many-to-many/heterogeneous AF and source-SiLU-unsupported combinations are not inferred supported. Communication remains a development projection without AF hardware truth. Policies are under `configs/runtime_memory_policy/af_disagg/`; `--deployment af_disagg` requires explicit TP/transfer/policy. `check_af_run_artifact.py --run-output ... --model ... --requests ...` checks structure/conservation, not accuracy.

### PD request lifecycle

P prompt completion produces full prompt KV and the first sampled token. Stop P decoding, transfer that KV exactly once, reserve D blocks before transfer, keep source blocks pinned during transfer, and admit D only after every physical completion. D continues from already-computed prompt/first token without repeating prefill. Global request/output/KV progress remains continuous. If no output remains, finish without D handoff.

P trace-prefix hit belongs only to P. Construct D's full-KV request with `prefix_hit_tokens=0` for both reservation and handoff, while retaining original P/hit evidence and true token/first-output identities. D receives full prompt KV, without subtracting P hit or double-counting transfer as local content hit.

### Heterogeneous PD and parallelism

The initial 0.30 BF16 standard-GQA Qwen3 A800/H200 domain uses independent P/D TP/PP via `--decode-hardware-config`, `--prefill-tp-size`, `--prefill-pp-size`, `--decode-tp-size`, `--decode-pp-size`; cross-hardware requires `--transfer-link infiniband --kv-transfer-backend analytical_roofline_v1`. Map layer/head ranges through physical placement, copying only necessary source replicas. `pd_kv_lifecycle` records pools/ranks/requests/block groups/times. Transfer is a streaming lower bound across both HBM/PCIe/network constraints, with unknown startup/propagation omitted explicitly. Reserve evidence is not upgraded by completion; arbitrary KV format conversion is unsupported.

`VLLMBackend.from_data_parallel_replicas` coordinates independent native CPU Schedulers/KV managers and retains each ScheduledRequest's `data_parallel_rank`. Dense DP routes independent replicas; Qwen3-MoE allows independent TP/EP/PP at each end. EP-on uses EP=TP×DP and SP=TP; TP1 adds no SP transfer. EP-off shards expert intermediate width by TP×DP and performs DP dispatch/combine then local TP sum. Validated request slices include TP2/EP4, TP2/EP-off, TP1/EP2, DP2/TP2/PP2/EP4, nondivisible EP3, and empty-expert CPU cases. Idle replicas still service experts without inventing requests/Attention/PP payload. PP sends hidden/residual per real DP owner, through independent destination backing, TP gather, and runner copy; SP restore sends only real-row prefixes. Padded rows affect MoE gate/dispatch/expert/combine, never request/KV progress. Common COMMUNICATION typed ports price all_gatherv/reduce_scatterv under explicit pairwise physical scenarios.

CLI independent `--prefill-dp-size`/`--decode-dp-size` uses the same coordinator. Optional policy data_parallel_size normalizes to1; derive weights/budgets across DP×PP×TP, preserve actual allocator receipts per DP, and check Scheduler startup. Expert width shards by TP×DP when EP-off; attention/router/vocabulary replicate within local TP/PP, and KV is not divided by DP/EP. Source-memory reconciliation uses actual DP backing. 0.30/EPLB-off linear placement handles base/remainder and empty expert ranks: count actual ownership, omit zero-extent parameters, and retain zero-output producers for combine. DP capacities may differ; public scalar blocks are the minimum, while native receipts/Schedulers retain each value. MoE EP-off DP is not dense independent expert replication.

GLM5.2 BF16 0.30 A800/H200 MTP-off explicit sparse-MLA reuses the same heterogeneous entry with independent TP/PP/DP/EP/CP. `--prefill-cp-size`/`--decode-cp-size` and continuous `--prefill-pp-layer-partition`/`--decode-pp-layer-partition` map to pool/Scheduler/policy fields; conflicting global/pool partitions reject. Schema2 optional CP/partition normalizes without changing old CP1/None round-trips.

PCP is MLA-only, DCP1/PP1. Mirror queries into 2×CP ceiling chunks with absolute causal positions; padding writes no KV. Gather normalized latent/RoPE and full-indexer K before separate cache insertion, then restore hidden order. Every TP/CP rank retains full compressed cache; full-indexer adds132B/token unaffected by EP. Handoff chooses source CP0/TP0 by real DP/layer ownership and charges every required D TP/CP copy. CPU allocators project per DP to actual CP1 workers, reporting full physical CP replicas/original identity/projection separately rather than inventing native PCP receipts.

GLM EP-off shards routed I by TP×CP×DP then DP/CP sum-scatter. EP-on DP>1/TP>1 uses SP=TP: PP1 model-level SP retains full dense/shared weights; PP>1 limits SP to MoE, dense uses ordinary TP, shared remains full, and restored real rows retain decoder late sum. TP1×CP>1×DP>1 uses explicit `analytical_global_expert_projection`, since upstream naive DP groups do not prove global exchange; `native_provider_equivalence=False`. DP1 PCP EP-on retains both Runner/decoder TP sums; numerical/provider equivalence remains unvalidated.

CP1 GLM PP supports continuous partitions and stage-local groups. Shared-indexer-first stages are legal constructors, but sparse/decode/mixed programs need a real top-k producer, never invented initialization/transfer. Pure fresh BF16 dense-MHA prefill with q=KV for all rows and context≤checkpoint top-k does not read top-k and admits shared-first partitions such as39/39. At CP1, omitted `--glm-prefill-attention-source` may choose this strict existing shape domain; explicit force_sparse retains dependency checks, and PCP requires it. Full-indexer-first examples26/24/28 or38/40 retain sparse/decode support. TP-partial send/gather/late sum remain declared source; no new top-k-transfer/numerical/provider/capacity claim follows CPU completion.

### Communication and Host KV

Communication is first-class work with operation, participants, per-rank payload, topology/link, algorithm, and dependencies. TP/PP/EP/AF first conserve ranks/bytes, then price physical service. Analytical/external backends replace each other, never add. Overlap follows resources/dependencies, not a global ratio. Timing type (PCIe/fabric/Host/algorithm) and mutex resource identity are orthogonal: different routes/directions may share an endpoint, and equal timing types may own independent links.

KV movement declares source/destination tier, blocks/request, owner rank, reservations, completion, physical resources, and compute dependencies. LOAD/EVICT intent reserves capacity; **completion** commits residency and releases consumers. Lifecycle replay is not native victim selection/prefetch/Host-controller simulation.

The separate autonomous source is pinned 0.30 `SimpleCPUOffloadConnector`: `kv_both`, CPU buffer/backend, `lazy_offload=false`, token-content prefix, per-rank CPU bytes multiplied by native world size. Later steps store hashed complete GPU blocks. Store completion makes CPU hashes visible and unpins temporary source without evicting GPU copies. Lookup checks GPU then contiguous CPU hashes; async load blocks request progress until feedback. Full CPU capacity suppresses stores; upstream refcounts/free queues independently own copies. Ordinary `Simulation.run()` integrates stores/cross-request hits/load-only steps/rank D2H-H2D/completion, frees overwritten CPU slots, honors GPU pin-delayed request free, and timelines transfer tails. Validated only for explicit transcripts, one whole-block group, eager CPU single-pool Qwen3 TP1/TP2, short loose/tight CPU budgets and zero-output tails. Fixed-ratio traces do not use this autonomous policy.

`production_trace_fixed_tier_request` requires HBM percentage, capacity, wall limit, and data path. Request and arrival-window selectors are half-open; `--max-requests` limits concurrent scheduling. Default streaming checkpoints on wall limit record completed requests/tokens/steps/Host-placement bytes and step-boundary-sampled parent/Scheduler RSS with `checkpoint_wall_limit`. Complete runs include rank H2D service/load gates. Original lengths/hits/arrivals remain unless explicit `--synthetic-output-cap` truncates output, marked synthetic with original `canonical_requests`; it cannot claim unchanged trace completion. Explicit `--execution-mode accelerated|fast` retains the entry's documented default.

## 6. Lowering and unified Kernel prediction

A ScheduleStep determines phases, layers/ranks, token work, and KV effects, then expands logical kind/shape/dtype/layout/variant/placement/bytes/dependencies. Math/deployment names identify work; hardware launch selection supplies variant/tile/grid/block/split. One logical op may lower to several calls. No Serving-case/name/E2E error selects alternate formulas.

Source graphs preserve aliases/lifetimes/dependencies. Geometry templates share only validated immutable fields; dynamic routes/layer identity rebind. `KernelBlockRepeat.trace_node_ids` maps all logical origins; `KernelPlan.logical_calls()` lazily expands source/parameters/dependencies and rejects legacy repeats lacking maps. Full mapped artifacts use schema15; streaming schema14/chainv6. Historical full8/stream9v1/13v5 retain their own read contracts and cannot be relabeled.

Ordinary A800 BF16 Qwen/Llama eager supports `injected_launch` without changing source graphs/storage/dependencies. Row-major Linear A/weight/C with transB=True and single-row head GEMV keys use existing canonical selectors; GQA main/setup/combine does not duplicate former wrapper fill/copies. Missing keys fail. Embedding/row selection/sampling may remain Roofline-only despite other native launches.

SGLang0.5.19 FULL_DECODE_ONLY captures decoder/head/FP32 logits, while active slicing/greedy stays outside. Underfilled buckets require explicit `SGLangGraphInputs` retained-tail IDs; unknown IDs cannot determine rank embedding rows. FlashMLA dummy query uses seq_len1/slot0 and a declared reserved64-token empty page; logical active arena starts after it without claiming native KV allocation. Other source sentinel/no-write rules remain unchanged. EAGLE+Graph requires separate support.

Kernel returns invocation time/resources/bottleneck/support/limitations. Serving does not inspect its compute/cache/transaction/pipeline formulas; Kernel does not read arrivals/TTFT/router queues/truth.

## 7. Global virtual time and resources

A work item starts when all predecessors finish **and** its physical resources are available. Independent ranks/engines/links may overlap; shared GPU/fabric/PCIe/Host resources compete. Batch work is shared execution, not a copy of its full cost per participating request. Merge pool-local timelines, routing/admission queues, communication, and migration into one horizon and feed completions back into native control state.

## 8. Metrics and diagnosis

Diagnosis traces paired request errors through steps, runtime intervals, and hardware kernels.

TTFT spans arrival to first client-visible output; E2E spans arrival to completion; TPOT uses eligible consecutive outputs and explicit endpoints. With Host delivery enabled, first/last output envelopes affect client TTFT/E2E/TPOT; different batch sizes do not guarantee canceling delivery costs. Preserve request output order without adding an invented CPU queue or Scheduler feedback. Disabled delivery retains engine endpoints. Report empirical CDF/declared percentiles and no inapplicable fabricated zeros.

Throughput may be requests, output tokens, or input+output tokens per declared horizon. SLO goodput counts successful eligible requests/tokens under versioned TTFT/TPOT thresholds, not production success rate. Interactivity/generation-speed tradeoffs require explicit units/denominators.

Resource/critical-path analysis separates scheduling/routing wait, compute/communication/migration, shared batch steps, dependency wait, and unexplained gaps. Unknown causes remain unknown; summed service/resource busy time may overlap and is not end-to-end latency.

## 9. Support, validation, and result identity

Supported means legal executable structure in the declared domain; validated means independent semantic/physical evidence; projected means missing target truth; unsupported requires explicit rejection. A case can validate shape/ranks/bytes yet retain projected timing. Kernel accuracy does not validate Scheduler feedback, and native control does not validate device timing.

Reproducibility binds workload/allowed fields; model revision/structure/dtype; pools/roles/hardware/parallelism/links; Scheduler version and all three budgets; routing/tie-break/CPU assumptions; hardware/launch profiles; MTP/routing/KV replay or assumptions; code identity; units/metrics/SLO. Keep sensitive content out of reports. Incomplete identity supports only a temporary observation, not cross-deployment/hardware comparison.

## 10. Extension boundaries

New Schedulers produce steps/KV/native decisions, not Kernel formulas or hidden timing. New models produce conserved logical work/dependencies. New communication/memory backends replace only their owned resources. Expand evidence for DP/SP/PCP/heterogeneous PD/low precision/hardware independently. Training/RL share some primitives but have different state machines and communication/metrics, and are not implicit Serving support.

## 11. Lowering helper-launch contract

A call already in KernelPlan but missing Host preparation belongs to `runtime_prepare.py`. A helper absent from KernelPlan receives neither device nor Host pricing. Lowering therefore declares **0/1/N actual launches**, not a residual constant. Predicates come from versioned vLLM/vllm_ascend branches; where lower dispatch differs, use duration-free discrete observations. Owners are `workload/helper_launch.py`, `workload/build_workload.py`, and `configs/validation/serving/helper_emission_observation.yaml`.

| Call point | Source/observed emission contract |
|---|---|
| Attention output writeback | Slice assignment is one ATen copy; paged `out=` is zero; dim0 slicing is a view and zero. 910B canonical dispatch counts/rank/phase are decode0, Qwen3-30B prefill0, Qwen3-4B/8B prefill/mixed `2L−1`, reflecting torch_npu/aclnn behavior that Python cannot infer. Versioned observation supplies count and `source_output_copy_plan` logical/physical domains; lowerer materializes TensorMove only for declared emissions |
| MoE routing mask | EP-off0. EP-on has Index→NotEqual→Cast→Mul; source shows three ops, Cast arises from boolean/weight type promotion, with48 emissions/string/layer in phase observations. Apply-router-weight-on-input adds Mul; masked-fill/where/renormalization remain conditional. A800/H200 emits expert-map-index, 910B all four, C500 recipe already owns `moe_expert_map_lookup` and workload does not duplicate it |
| Model input/output | Sampling steps materialize exactly one `hidden_states[logits_indices]` gather; ordinary slicing is a zero-launch view. Final-norm→logits-boundary-gather→head carries source predicates. Token embedding uses its call-point count; A800 prefill observed zero requires explicit observation. MTP token-embedding/position-zero-mask pairs remain explicit |

The contract records `source_symbol`, source-referencable `predicate`, `count`, kind/variant, and existing shape inputs. **Zero requires explicit authorization**; absence is not zero. Unknown branches/missing required observations fail closed without source-predicate fallback. Acceptance is an exact per-step trace count equality, independent of duration. Shape/model/card constants cannot encode source branch identity and would erase falsifiability, typed shape work, and cross-version provenance.

## 12. Evaluation workflows

### Independent LLMServingSim baseline

Only explicit `scripts/evaluation/serving/run_llmservingsim.py` loads this baseline; default Uni³Sim never imports its checkout/Python/Chakra/ASTRA. `configs/baselines/llmservingsim/source.json` uniquely owns source identity/five bundled settings, with original requests/TP/EP/memory. Its Scheduler, measured LUT interpolation/extrapolation, and ASTRA analytical **network** remain independent; LUT durations never enter Uni³Sim Forward/HWSpec.

Prepare an isolated runtime copy under `experiment_logs/evaluation/llmservingsim/runtime`; keep original source read-only. Initialize ASTRA/Chakra/fmt/spdlog/memory-analytical/network-analytical/yaml-cpp, without NS3/GPU. Independent Python3.10 dependencies are `numpy==1.23.5 pandas==1.5.3 pyyaml pyinstrument rich grpcio-tools setuptools-grpc networkx`; install fixed Chakra with `uv pip install --python <python> --no-deps <Chakra>`, without vLLM/weights/optional HTA. Record Python/build locks.

A separate build Conda prefix provides `protobuf=3.20 cmake<4 make`; generate Chakra `et_def.pb.cc/.h` with its protoc and matching-schema Python via grpcio-tools. Configure ASTRA with `cmake -S <ASTRA>/build/astra_analytical -B <ASTRA>/build/astra_analytical/build -DBUILDTARGET=congestion_unaware -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-DNDEBUG -DCMAKE_PREFIX_PATH=<prefix>`, build `-j 4`, and link `AnalyticalAstra/bin/AnalyticalAstra` to `../../bin/AstraSim_Analytical_Congestion_Unaware`. NDEBUG matches upstream overwritten release flags to Protobuf release, without third-party source edits.

CLI modes are `--bundled-suite`, report-only `--report`, read-only `--prepare-targets`, independent `--midlayer-targets` (local lru_sim required), and CPU static `--capacity-check`, with explicit `--runtime`, `--python`, and fresh `--output`. Per-case `run.json` binds commands/config/input/binary/exit; new `sim.csv` is raw output; `normalized.json` checks original-row identity/order/completeness/arrivals/lengths/instances/finite metrics and converts ns→s. Comparisons separate historical reproducibility and vLLM error, with means/P50/P90/P99/MAE/MAPE/full denominators. TPOT divides by output−1, single-output native zero retained; throughput windows use each side's first arrival→finish, separately preserving Uni³Sim workload-zero horizon. No per-token ITL truth means no ITL accuracy claim.

`targets.json` exhaustively marks historical hardware `active` or `deferred_by_user`. Current A800/C500/910B two-model six cases are active, H200 two cases deferred. Active denominators retain missing profiles; deferred historical results stay separate and do not count success. All collection/native/Uni³Sim/capacity paths honor deferral before SSH/device/capture. Preserve canonical eight-case input/cache mapping; nearest/half-even ns rounding retains original ms and≤0.5ns delta. Missing matching hardware/model/precision/TP/implementation profiles cannot be renamed/scaled/filled with predictions/empty tables.

Use matching Llama Instruct config. Synthetic identities instantiate only structured-evidence scheduler-equivalent **zero-net-prefix-work** representatives, not original prompts or observed logical zero hits. Match actual KV byte budgets/block geometry rather than assuming equal utilization means equal reserves. Report software, cache representative, engine geometry, eager/uniform/plugin/extrapolation limits and warnings; missing lookup counts stay unavailable. Uni³Sim comparison uses task-local regenerable code/data materialization and read-only validated raw references, without modifying production/canonical/main. Request metrics remain evaluation-only.

Capacity checks extract static fields from existing allocation manifests, build native `MemoryModel` in independent Python, and compare blocks using `(native weight bytes + available KV bytes)/physical HBM bytes`. Preserve policy/weight-count differences; do not change configs or substitute timing. Equal blocks do not prove full resource/profile equivalence.

`llmservingsim_profile_runner.py` requires current `--owner`, native one-layer dummy-weight BF16/TP1 in a task execution copy, no model download. Preflight verifies idle allocated device/model/UUID; manifest records owner/device/baseline/source/config/command/PID. Temporary build/IPC is task-owned and cleaned. Resume requires the same task/hardware/model and stopped process, with content-addressed reused measurements. `--attention-grid-factor` changes sampling density, not domain; `--skip-skew` remains uniform-only. Deferred hardware fails before device reads. Hardware-only NVIDIA collection records UUID-bound NVML HBM/clock/bus and native DDR bandwidth conversion, no kernels/communication, with unmeasured interconnect explicit. Legacy Kineto mappings require unique CUPTI launch correlation even for resource0.

`dataset_pipeline/serving_dataset/llmservingsim_profile.py` creates private adaptations: A800 old request/KV API via `--vllm-011`; MetaX source-proven catalog aliases and real Kineto correlation; Ascend native torch_to_npu module/task mapping, per-invocation device-interval union, raw audit of excluded runtime/preparation. Ascend high-token factor-two sampling is an explicit cost choice rather than native default; engine block geometry rejects128-versus16 mismatches without changing targets.

An explicit Ascend `block16_capability=True` requires same-stack BF16 numerical checks of native FIA/paged attention at16/128, source fingerprints of refresh/support functions, and `--block16-evidence`. It changes only private policy/support declarations, preserving device operators/metadata, limited to verified BF16/TP1 Qwen/Llama32 query/8 KV heads/head_dim128. Actual blocks must be16 with full three-stage coverage; retain TND mixed causal control and original128 tables, never rename timings.

After valid raw return, `--target-case <hardware>/<model>/tp1_qps16 --profile-capture ... --hardware-capture ... --runtime ... --python ... --output ...` validates all identities/geometry/acquisition before running the full original case. Cases write independent subdirectories and may run concurrently. Reports distinguish LUT compute+analytical network from unavailable pure analytical Transformer compute. Profile differences stay evaluation-only; final acceptance remains primary-owned.

### Logical weight placement

`deployment_weight_placements(model, deployment)` produces logical startup inventory without weights/device allocation. Supported unquantized BF16/FP16 Qwen3/Llama and residency-admitted Qwen3MoE reuse TP/PP/EP/multiple-colocated placement. Reject TP>KV heads, tied embedding across PP, custom partitions, unsupported compression, and split pools. IDs are pool/rank logical identities; fused QKV/gate-up/expert objects retain `object_view`. Physical addresses/layout/UUID remain unknown. Vocabulary logical shape excludes padding while storage/shards include it; same-rank tied aliases count once. Inventory excludes KV/activation/workspace/reserves.

```bash
python scripts/evaluation/serving/report_weight_placement.py \
  --model configs/model/qwen3-8b.json --hardware nvidia_a800 --tp 1 --pp 2 \
  --instance main:2,7 --output experiment_logs/weight_placement/qwen3-8b-pp2.json \
  --table experiment_logs/weight_placement/qwen3-8b-pp2.tsv
```

`Simulation`/`MultiSimulation(..., weight_placement_consumer=callback)` emits the same validated inventory only when requested. Physical rank numbers are deployment declarations rather than UUIDs.

`dsv4_routed_mxfp4_placements` is a loader-only EP-off subset: packed MXFP4/UE8M0 scales, gate/up logical ranges, sizes, and no-padding constraints from `dsv4_routed_loader_payload_bytes`. It excludes nonrouted/deployment/full-model/post-load Marlin layout. `dsv4_postload_inventory_diagnostic` checks complete target-off bytes using existing SM90 DeepGEMM/Marlin source, module-local loader shapes, replication/TP, block128 scales, and sink padding. Explicit `v4_weight_storage_profile="sm90_deepgemm_marlin"` / CLI `--v4-weight-storage-profile` accepts H200 EP-off PP1 but does not assert selected runtime provider. MXFP4 local shape remains logical, loader shape packed; scale grids separately identify represented weight intervals. FP8 loader UE8M0 scales become FP32; routed packed values/UE8M0 remain, with post-load physical layouts unknown.

## 13. Offline interactive reports

The simulation CLI writes an adjacent HTML to `--output`, or a new `experiment_logs/serving_reports/` run, overridable with `--interactive-report`. Current default is `--interactive-view swimlanes --interactive-detail-mode summary`; it does not enable execution recording or change mode/workers/retention/source state. Missing Device/Host intervals mean unknown/null, not zero busy or an envelope disguised as activity. Both full and streaming retention use existing execution projections exactly once/step.

`sampled --interactive-max-events 100000` retains bounded **complete step** details across phases/requests, preferring coverage/first-last/large batches with deterministic ties, never selecting by predicted duration. Full-run resource aggregates remain independent of sampled details. Reports omit request text/token IDs/kernel attributes. A request appears covered if it participated in one retained batch, not necessarily its full lifetime; filters select participating batches whose compute stays shared. Resource tables aggregate all work by pool/rank/window, independent of request filters. Omitted envelopes include gaps and cannot imply continuous busy.

Summary shows recorded model/hardware/parallelism, completed requests/throughput/eligible means, simulated full-run span, and separate simulator wall time. Full span is not mean E2E. Group totals sum predicted operation time and can overlap; maximum resource demand is a rank-local whole-run mean, never a sum across cards. Unknown failure/support/cost/overlap stays unknown. Configuration snapshots are read/bound at run start, exclude credentials/content, and never use current files to impersonate historical inputs.

`<report>.report.json` measures recording/projection/compression/HTML/JSON writes and producer wall time. HTML-only generation is not full reporting cost; overlapping phases cannot be added blindly. The embedded total stops at first result write; final receipts/outer wall include metadata updates. `visualization_run_costs` accepts sourced finite nonnegative HTML seconds/scope only. Compact transfer metadata caps at1024, retaining full count/net bytes/omission totals separately; Forward/request index size still grows with workload. Explicit `--no-interactive-report --no-report-reason ...` records an opt-out; failures/unsupported produce truthful partial summaries. Paths cannot collide with result files.

HTML is standalone, without CDN/fonts/services, storing gzip/Base64 data for `DecompressionStream`; bad/unsupported decoding disables controls and reports failure. Compression is not equivalent browser-memory reduction. Scroll pans, Alt+wheel zooms around cursor, drag/range/navigation synchronize the shared simulated axis; resource dialogs support Shift-drag selection. Pool/rank groups collapse, labels stay fixed, and the full-run navigator remains visible. UI labels may be translated without changing source IDs/evidence; view details retain the same data/time/filter state.

Operation breakdown separates predicted compute/communication/KV/Host by category, not independent hardware queues. Logical HBM demand is uniformly spread over recorded intervals and is not an observed HBM operation. `bottleneck_group`, nonzero service, and optional per-operator model critical paths come directly from predictors; missing historical fields stay unknown. The path catalog caps4096 and per-step/group paths32, recording omitted counts. These internal paths and summed group times are not request-level critical paths or serial service intervals.

The older breakdown resource plot uses Device interval unions/window for busy, typed operations/resource capacity/window for compute, and compulsory logical bytes/HBM bandwidth/window for a lower bound. Uniform step-envelope spreading is a time-distribution approximation; 48 equal bins do not create hardware counters. Preserve values above100% and missing-resource counts. Endpoint algorithm bytes exclude protocol/retry overhead and are counted once per physical call, not summed across endpoints then divided by one endpoint capacity. Unmodeled collectives remain unavailable. HBM/KV/known activation capacities remain separate and incomplete.

`visualization_storage` deduplicates descriptor/pool/rank/resource/call tables inside regenerable report storage, not execution/canonical schemas. Readers accept historical inline/new indexed encoding and reject bad references. Store original absolute endpoints without subtract/add reconstruction. Sparse omission of explicit zero service is distinct from missing evidence. Default projections omit full source IDs/group times/Host dependency DAG and cannot reconstruct those omitted chains.

Swimlanes synchronize Forward/CPU/Device/Tensor/Vector/L2/HBM/scale-up/out. Whole-step aggregates, rather than retained samples, drive curves. Tensor normalizes each dtype's work; Vector takes the maximum normalized role rather than summing unrelated units. L2/HBM read/write bytes come from existing `PredictionInput.memory_paths` `l2_cache`/`dram`, including existing cache/writeback, not logical bytes or time-inferred traffic. H200 display reference `microbench/nvidia_h200/references/l2_peak_metadata_sm90.json` uses L2 read6016/write3008B/clk at1.98GHz =11.91168/5.95584TB/s, and common HBM4.8TB/s. L2 combined fraction is max(read%,write%); HBM sums directional fractions; total GB/s always sums bytes, not unlike denominators. These display-only references are source-exported/persisted and do not alter pricing/HWSpec. Other hardware uses explicitly labeled model bandwidth; mixed reports without row hardware identity cannot apply H200 references globally.

Directional ledgers distinguish modeled downstream zero (e.g. cache hit/no writeback) from missing ledgers. Count aggregate paths once, without readding pipeline stages or running another report-owned LRU. Predictor-local cache contexts do not establish full-run interoperator cache continuity. Single-card missing links are N/A. Untimed CPU Scheduler events remain zero-width markers with an untimed label; plotting never changes timing configuration.

Forward selectors report pool/iteration/phase/request count and scheduled input+output token work excluding padding, not final request lengths/generated totals. Auto-scaled resource heights cannot be compared across axes. Hover/pinned values show exact bin windows; zoom keeps each value a window mean rather than creating instantaneous counters.

`summary` stores no per-operator detail; `sampled` obeys event budget; swimlanes `all` retains every Forward and ignores that sampling cap. Each step's safe projection is independently written to `<report>.iterations/iteration-*.json.gz` and embedded in HTML; load only overview first, then the chosen iteration, replacing previous detail. This needs both CompressionStream/DecompressionStream. Missing sample details stay summary-only. Regeneration requires indexed chunks with validated path/length/SHA and original absolute shift, but the final HTML needs no sidecars. Fresh task directories avoid silent chunk overwrites.

```bash
python scripts/evaluation/serving/report_interactive.py \
  experiment_logs/<run>/report.visualization.json experiment_logs/<run>/report.html
```

`run_interactive_example.py` supplies CPU-only `single`, `tp2`, `pd` full-Qwen3-4B examples (CPU0.30/Roofline/existing H200 Host profile) and the existing four-pool 0.11 token-prefix fixture without Host pricing. Provide output/SDK Python/source root; preserve SDK virtual-environment paths. Test KV budgets do not validate normal/provider capacity. `--no-record` compares the same scenario; `state.json` retains completion/token/routing/transfer/activation state. Real case configuration, not filename, determines runtime.

Static model summaries show recorded graph size/serial prices/unpriced calls/configuration without invented Serving intervals. Historical results lacking projections get partial metric summaries, without inferred means from percentiles or fabricated curves. Saved views regenerate by declared type; unknown types fail.
