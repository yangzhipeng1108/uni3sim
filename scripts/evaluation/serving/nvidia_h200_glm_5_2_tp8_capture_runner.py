#!/usr/bin/env python3
"""GLM-5.2 TP8 real-serving capture on 8x H200 with vLLM 0.30 (dummy weights).

Modeled on the DSV3-L32 TP4 H200 captures (``nvidia_h200_nsys_pure_decode_capture.sh``,
``nvidia_h200_nsys_prefill_capture.sh``, and the remote ``sweep_closedloop.sh``):

* ``smoke``: start the server (same flags as the real run, no nsys), check the
  startup facts (KV capacity, attention / MoE / all-reduce backends, routing
  simulation active), send one completion request, measure server start ->
  ready and request -> response, shut down, verify all eight GPUs are free.
  Also warms the DeepGEMM / FlashInfer / Triton JIT caches used by ``run``.
* ``run --phases closedloop``: InferenceX-style 8K/1K closed loop without
  nsys: ``vllm bench serve`` random dataset, input 7373 +-11.1% (6554..8192),
  output 922 +-11.1% (820..1024), ``--ignore-eos``, ``--max-concurrency C``,
  ``10*C`` measured prompts after ``2*C`` warm-ups, per C in ``--concurrency``.
* ``run --phases nsys``: one server under ``nsys --capture-range=cudaProfilerApi
  --cuda-graph-trace=node`` (vLLM ``--profiler-config '{"profiler": "cuda"}'``,
  windows through ``/start_profile`` / ``/stop_profile``).  Per C one
  pure-decode window (exactly C requests, captured after all C prefills have
  produced their first token -- checked on the TTFT histogram count -- and
  before the shortest request, >= 820 output tokens, can finish) and one
  prefill-only window (C closed-loop clients, fixed 7373-token prompts, one
  output token, window starts 3 s into the main run).  Windows are 1.5 s.
  ``windows.txt`` uses the DSV3 ``<label> start|stop <time>`` format so
  ``summarize_nvidia_h200_nsys_decode_windows.py`` can read it; ``windows.json``
  adds unix times, metric deltas (purity checks) and clock flags.

Every GPU use must hold the eight whole-GPU locks; launch through
``microbench/nvidia_h200/kcampaign/remote/with_gpu_locks.sh 0,1,2,3,4,5,6,7
<wait> -- <this script> ...``.  The script verifies that it inherited the
lock fds (with_gpu_locks.sh opens them as fds 200..207) and that no compute
process is on any GPU before it starts.  SM clocks cannot be locked: an
``nvidia-smi`` clock log is taken for the whole job and every closed-loop C
and every window reports its busy samples at a clock other than 1980 MHz
(flag only; nothing is dropped).

``--dry-run`` prints every command and output path without launching any
process, touching a GPU or calling nvidia-smi.

The orchestration lives in ``nvidia_h200_tp8_serving_capture.py``; this entry point
only supplies the GLM-5.2 ``ModelProfile`` (same CLI, commands and outputs as
before the split).

Evaluation-only measurements; no timing feeds any model parameter.

Wall-time estimates (not measurements; scaled from the DSV3-L32 TP4 vLLM
0.24 H200 runs of 2026-09-25, whose closed loop took ~1.2/1.7/3.5/8.5 min
at C=1/4/16/64 and whose server started in ~1 min): smoke 10-25 min (first
start JIT-compiles DeepGEMM/FlashInfer kernels for GLM shapes); closedloop
~5-20 min startup + ~2.5/3/6/13 min for C=1/4/16/64; nsys ~5-15 min startup
+ 8 windows x (~0.5-2 min load ramp + ~1-6 min report flush).  Total run
~1.3-2.0 h.
"""

from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dataset_pipeline.profiler import nvidia_h200_serving_capture as capture

PROFILE = capture.ModelProfile(
    key="glm52",
    log_tag="glm52_tp8",
    doc=__doc__,
    serve_sh=Path(__file__).resolve().with_name("nvidia_h200_glm_5_2_tp8_serve.sh"),
    model_dir=Path("models") / "GLM-5.2-FP8-config",
    served_model="glm52-fp8",
    port=8012,
    expected_config={"architectures": ["GlmMoeDsaForCausalLM"]},
    kv_cache_dtypes=("fp8_ds_mla", "auto"),
    kv_cache_dtype_help="fp8_ds_mla -> FLASHMLA_SPARSE; auto (bf16) -> "
                        "FLASH_ATTN_MLA_SPARSE on SM90",
    env_prefix="GLM52",
    gpu_memory_utilization_arg=False,  # fixed 0.90 in nvidia_h200_glm_5_2_tp8_serve.sh
    speculative_config_mtp_on={},      # MTP off only
    separate_nsys_concurrency=False,   # nsys windows at every --concurrency C
    bench_sampling_args=(),            # server-side default sampling
    extra_backend_patterns=(),
    extended_record=False,
    sigterm_cleanup=False,
    offload=None,
)


if __name__ == "__main__":
    capture.main(PROFILE)
