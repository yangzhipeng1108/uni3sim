#!/usr/bin/env python3
"""DeepSeek-R1 TP8 real-serving capture on 8x H200 with vLLM 0.30 (dummy weights).

H200 ground truth for the InferenceX DSR1 8K/1K TGS-Interactivity comparison
(InferenceX configs ``dsr1-fp8-h200-sglang`` / ``dsr1-fp8-h200-sglang-mtp``:
DeepSeek-R1-0528 FP8, one TP8 EP1 instance, C in {4, 8, 16, 32, 64}; raw
points in ``profiler_output/serving/inferencex_alignment/raw/dsr1_all_20260924.json``;
the one-off protocol-filter script was retired in the C4 scripts cleanup and
remains recoverable from git history).  Same orchestration
as the GLM-5.2 capture (``nvidia_h200_tp8_serving_capture.py``; smoke, closed loop,
per-C pure-decode and prefill nsys windows); what this entry point fixes:

* Model: config/tokenizer-only DeepSeek-R1-0528 (``models/DeepSeek-R1-0528-
  config``; its config.json equals the DeepSeek-V3 config apart from
  ``transformers_version``, so execution is that of DeepSeek-V3; the R1-0528
  tokenizer builds the random prompts as in InferenceX), FP8 e4m3 block
  [128,128] weights, ``--load-format dummy``, TP8, MoE routing
  ``VLLM_MOE_ROUTING_SIMULATION_STRATEGY`` (``uniform_random``: dummy weights
  skew the router).
* Client: InferenceX ``benchmark_lib.sh run_benchmark_serving`` =
  ``--num-prompts 10*C --num-warmups 2*C --ignore-eos --request-rate inf``,
  input/output [0.8*L, L] for L = 8192/1024 (vLLM's symmetric ratio: 7373 /
  922, ratio 1/9); its client sends temperature 0, so every request here
  carries ``--temperature 0`` (vLLM 0.30 ``bench serve`` otherwise leaves the
  temperature to the server).  InferenceX's MTP client adds
  ``--use-chat-template`` (a few template tokens per 7.4K prompt); not
  reproduced.
* Server: InferenceX SGLang recipe ``--chunked-prefill-size 32768``
  (``--max-num-batched-tokens 32768``), radix cache off (prefix caching off),
  no KV dtype flag (BF16 KV, ``--kv-cache-dtype auto`` -> FLASH_ATTN_MLA on
  SM90, the path of the dsr1_mtpoff_fa3 / dsr1_mtpon_verify kernel plans).
* MTP on: InferenceX ``SPECULATIVE_NUM_STEPS=2 SPECULATIVE_DRAFT_TOKENS=3
  SPECULATIVE_EAGLE_TOPK=1`` = 2 draft tokens per verify, i.e. vLLM
  ``num_speculative_tokens 2`` on the single NextN layer (method ``mtp``).
  Dummy weights accept ~no draft, so acceptance is imposed with vLLM 0.30
  synthetic rejection sampling at the repo's DSR1 comparison profile
  ``inferencex_sglang_h200_mean`` (run_inferencex_deepseek_v3_8k1024_closedloop_
  comparison.py: per-token acceptance 0.673 = mean over C=4..64 of the
  InferenceX SGLang server logs' accept length, run 34450728410) with the
  simulator's maximum-entropy count distribution
  (src/serving/backend/mtp_acceptance_assumption.py, depth 2, rate 0.673:
  P(K=0,1,2) = 0.1756, 0.3027, 0.5216), given to vLLM as unconditional
  per-position rates [P(K>=1), P(K=2)] = [0.8244, 0.5216] (mean acceptance
  length 2.346).  Achieved acceptance is read back from the spec-decode
  counters per C and per window.

Memory at TP8 (per GPU, estimated from config.json): FP8 weights ~79.5 GiB
(+ ~2 GiB for the MTP layer); BF16 MLA KV 61 x 576 x 2 B = 70,272 B per token
(71,424 B with the MTP layer), so C=64 x 9,216 tokens = 38.6 GiB (39.2 GiB
MTP on).  ``--gpu-memory-utilization`` is a required argument (runbook: 0.95);
the smoke prints the vLLM KV capacity, the selected backends and the
utilization at which the KV pool would just hold max(C) x (9,216 + drafts)
tokens; runs abort before measuring when the capacity is short.

Wall-time estimates (not measurements; 12 request rounds per client x the
InferenceX SGLang median E2E latency, x1.0-1.4 for vLLM vs SGLang):
closed loop MTP off C=1/4/8/16/32/64 ~1.7/2.1/2.6/3.8/6.0/9.5 min (26-36 min),
MTP on ~1.1/1.4/2.1/3.1/4.7/7.9 min (20-28 min), + 3-8 min per server start;
nsys per MTP setting 5-10 min start + 8 windows x 1.5-4 min (17-42 min).
Full run (closed loop off+on, nsys off+on) ~1.5-2.7 h; smoke 10-25 min cold
per MTP setting (5-10 min warm).

KV host-offload group (``run --phases offload``): same
model, weights, routing, BF16 KV, chunk and utilization, but
``nvidia_h200_deepseek_r1_tp8_offload_serve.sh`` (prefix caching on, --max-model-len 163840,
per-request cached tokens / metrics), MTP off, and an AgentX-derived
multi-turn replay (InferenceX ``agentic-coding`` corpus
cc-traces-weka-062126, main-agent stream, see ``agentx_session_replay.py``).
Arms: ``native`` = ``--kv-offloading-size 256 --kv-offloading-backend native``
(OffloadingConnector, CPUOffloadingSpec, one replicated MLA copy in
/dev/shm = 3.9M tokens); ``simple`` = the same flag with
VLLM_USE_SIMPLE_KV_OFFLOAD=1 and 768 GiB (SimpleCPUOffloadConnector, the vLLM
backend InferenceX AgentX uses, 96 GiB = 1.47M tokens per rank); ``gpu`` =
prefix caching only (control).  Per point 150 s warm-up + 300 s measured + up
to 60 s grace, recorded idle gaps capped at 60 s.

Evaluation-only measurements; no timing feeds any model parameter.
"""

from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dataset_pipeline.profiler import nvidia_h200_serving_capture as capture

PROFILE = capture.ModelProfile(
    key="dsr1",
    log_tag="dsr1_tp8",
    doc=__doc__,
    serve_sh=Path(__file__).resolve().with_name("nvidia_h200_deepseek_r1_tp8_serve.sh"),
    model_dir=Path("models") / "DeepSeek-R1-0528-config",
    served_model="dsr1-fp8",
    port=8013,
    expected_config={"architectures": ["DeepseekV3ForCausalLM"],
                     "model_type": "deepseek_v3", "num_hidden_layers": 61,
                     "num_nextn_predict_layers": 1, "n_routed_experts": 256,
                     "num_attention_heads": 128},
    kv_cache_dtypes=("auto", "fp8"),
    kv_cache_dtype_help="auto (BF16, InferenceX SGLang default) -> FLASH_ATTN_MLA "
                        "on SM90; fp8 only if BF16 KV cannot hold max(C)",
    env_prefix="DSR1",
    gpu_memory_utilization_arg=True,
    speculative_config_mtp_on={
        "method": "mtp", "num_speculative_tokens": 2,
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_rates": [0.8244, 0.5216]},
    separate_nsys_concurrency=True,
    bench_sampling_args=("--temperature", "0"),
    extra_backend_patterns=(r"non-default args", r"Model loading took",
                            r"Available KV cache memory", r"CUDA graph pool memory",
                            r"[Ss]peculative", r"[Rr]ejection"),
    extended_record=True,
    sigterm_cleanup=True,
    offload=capture.OffloadGroup(
        serve_sh=Path(__file__).resolve().with_name("nvidia_h200_deepseek_r1_tp8_offload_serve.sh"),
        max_model_len=163840,
        arms={
            # OffloadingConnector (v1/kv_offload/cpu); MLA at TP8 keeps one
            # replicated copy in /dev/shm/vllm_offload_<engine>.mmap, so
            # 256 GiB / 70,272 B = 3.9M tokens of host KV.
            "native": capture.OffloadArm(
                serve_args=("--kv-offloading-size", "256",
                            "--kv-offloading-backend", "native"),
                use_simple_kv_offload="0", connector="OffloadingConnector",
                activation_patterns=(
                    r"Creating offloading spec with name: CPUOffloadingSpec",
                    r"Created mmap file /dev/shm/vllm_offload_\S+ \([\d.]+ GB\)",
                    r"'kv_offloading_size': 256"),
                host_loads_required=True),
            # SimpleCPUOffloadConnector (InferenceX "vllm-simple", eager):
            # per rank 768/8 = 96 GiB, each rank a full MLA copy = 1.47M tokens.
            "simple": capture.OffloadArm(
                serve_args=("--kv-offloading-size", "768",
                            "--kv-offloading-backend", "native"),
                use_simple_kv_offload="1", connector="SimpleCPUOffloadConnector",
                activation_patterns=(
                    r"SimpleCPUOffloadConnector: role=\w+, per_rank=96\.00 GB, "
                    r"world_size=8, mode=eager",),
                host_loads_required=True),
            # control: GPU prefix caching only.
            "gpu": capture.OffloadArm(
                serve_args=(), use_simple_kv_offload="0", connector="",
                activation_patterns=(), host_loads_required=False),
        },
        replay_script=Path(__file__).resolve().with_name("agentx_session_replay.py"),
        warmup_s=150.0, measure_s=300.0, grace_s=60.0, think_cap_s=60.0,
        sock_read_timeout_s=1800.0,
        server_extra_patterns=(r"connector", r"[Oo]ffload", r"mmap file",
                               r"cudaHostRegister", r"max_model_len")),
)


if __name__ == "__main__":
    capture.main(PROFILE)
