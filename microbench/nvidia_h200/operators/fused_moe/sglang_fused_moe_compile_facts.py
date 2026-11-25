#!/usr/bin/env python3
"""Record the compile facts of SGLang 0.5.12's production fused_moe_kernel for the DSR1 TP8 configs.

Launch facts only (registers, shared memory, warps, stages, constexpr
specialisation); no timing is taken or recorded.  Runs SGLang's own
``fused_moe`` on synthetic FP8 E=257 (256 routed + 1 fused shared expert),
N=256 (intermediate per TP8 rank), hidden 7168, [128, 128]-block weights for
each decode- and prefill-range token count, so SGLang resolves the same
``E=257,N=256,device_name=NVIDIA_H200,...json`` / ``_down.json`` configs as the
served model, then reads every compiled specialisation from the Triton JIT
cache of ``fused_moe_kernel``.  Must run in the SGLang 0.5.12 environment.
Usage: sglang_fused_moe_compile_facts.py <out.json>
"""

import json
import sys
import types

import torch
import triton

import sglang.srt.server_args as server_args

from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils import (
    fused_moe_triton_kernels as kernels)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.topk import StandardTopKOutput

EXPERTS, ROUTED, TOPK, HIDDEN, INTERMEDIATE, BLOCK = 257, 256, 9, 7168, 256, 128
TOKENS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256,
          512, 1024, 4096)


def main():
    # The MoE path reads only these two server flags; both are False in the
    # InferenceX v0.5.19 and the target host v0.5.12 server logs.
    server_args._global_server_args = types.SimpleNamespace(
        enable_deterministic_inference=False,
        enable_fused_moe_sum_all_reduce=False)
    torch.manual_seed(0)
    fp8 = torch.float8_e4m3fn
    w1 = (torch.randn(EXPERTS, 2 * INTERMEDIATE, HIDDEN, device="cuda",
                      dtype=torch.bfloat16) * 0.05).to(fp8)
    w2 = (torch.randn(EXPERTS, HIDDEN, INTERMEDIATE, device="cuda",
                      dtype=torch.bfloat16) * 0.05).to(fp8)
    w1_scale = torch.rand(EXPERTS, 2 * INTERMEDIATE // BLOCK, HIDDEN // BLOCK,
                          device="cuda", dtype=torch.float32)
    w2_scale = torch.rand(EXPERTS, HIDDEN // BLOCK, INTERMEDIATE // BLOCK,
                          device="cuda", dtype=torch.float32)
    config = MoeRunnerConfig(
        num_experts=EXPERTS, num_local_experts=EXPERTS, hidden_size=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE, top_k=TOPK,
        num_fused_shared_experts=1, params_dtype=torch.bfloat16,
        inplace=False)
    for tokens in TOKENS:
        x = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
        ids = torch.randint(0, ROUTED, (tokens, TOPK), device="cuda",
                            dtype=torch.int32)
        ids[:, -1] = ROUTED
        weights = torch.rand(tokens, TOPK, device="cuda", dtype=torch.float32)
        logits = torch.zeros(tokens, ROUTED, device="cuda")
        fused_moe(x, w1, w2, StandardTopKOutput(weights, ids, logits), config,
                  use_fp8_w8a8=True, w1_scale=w1_scale, w2_scale=w2_scale,
                  block_shape=[BLOCK, BLOCK])
    torch.cuda.synchronize()
    facts = []
    for device_cache in kernels.fused_moe_kernel.device_caches.values():
        for compiled in device_cache[0].values():
            constants = {
                str(key): value for key, value in
                compiled.src.constants.items()
                if isinstance(value, (bool, int, float, str))}
            facts.append({
                "registers_per_thread": compiled.n_regs,
                "spilled_registers": compiled.n_spills,
                "shared_memory_bytes": compiled.metadata.shared,
                "num_warps": compiled.metadata.num_warps,
                "num_stages": compiled.metadata.num_stages,
                "constants": constants,
            })
    out = {"triton": triton.__version__, "torch": torch.__version__,
           "kernel": "sglang fused_moe_kernel (moe_runner/triton_utils)",
           "timing_recorded": False, "specialisations": facts}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for fact in facts:
        c = fact["constants"]
        print({k: c[k] for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N",
                                 "BLOCK_SIZE_K", "swap_ab", "c_sorted",
                                 "MUL_ROUTED_WEIGHT", "even_Ks") if k in c},
              fact["registers_per_thread"], fact["shared_memory_bytes"],
              fact["num_stages"])


if __name__ == "__main__":
    main()
