"""Reproduce the deployed MoE block so mcTracer can record its launch facts.

The C500 decode step issues about twenty-one kernels per MoE layer.  Only
`topkGatingSoftmax` is reachable from open source; the routing bookkeeping,
the torch reduce quartet, the histogram and the elementwise glue are all
issued from inside the compiled `vllm_metax` extension, so their grid and
block cannot be derived by reading anything.

The contract's sanctioned alternative is a versioned observation that records
only discrete launch facts.  This probe drives the deployed `fused_topk` and
`fused_experts` at the production routing geometry so one mcTracer invocation
captures those facts.  It measures no duration and emits none: the output is
kernel name, grid and block only.
"""

import argparse
import torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=22)
    p.add_argument("--global-experts", type=int, default=128)
    p.add_argument("--local-experts", type=int, default=64)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--intermediate", type=int, default=768)
    p.add_argument("--iters", type=int, default=3)
    a = p.parse_args()

    from vllm.model_executor.layers.fused_moe import fused_topk
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    torch.cuda.set_device(0)
    dev, dt = "cuda", torch.bfloat16
    hidden = torch.randn((a.tokens, a.hidden), device=dev, dtype=dt)
    gating = torch.randn((a.tokens, a.global_experts), device=dev,
                         dtype=torch.float32)
    w1 = torch.randn((a.local_experts, 2 * a.intermediate, a.hidden),
                     device=dev, dtype=dt) * 0.02
    w2 = torch.randn((a.local_experts, a.hidden, a.intermediate),
                     device=dev, dtype=dt) * 0.02
    # EP2 rank 0 owns the first half of the global expert range.
    expert_map = torch.full((a.global_experts,), -1, device=dev,
                            dtype=torch.int32)
    expert_map[:a.local_experts] = torch.arange(
        a.local_experts, device=dev, dtype=torch.int32)

    for _ in range(a.iters):
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states=hidden, gating_output=gating, topk=a.topk,
            renormalize=True)
        fused_experts(
            hidden, w1, w2, topk_weights, topk_ids,
            global_num_experts=a.global_experts, expert_map=expert_map)
    torch.cuda.synchronize()
    print(f"ok tokens={a.tokens} global_experts={a.global_experts} "
          f"local_experts={a.local_experts} topk={a.topk} "
          f"hidden={a.hidden} intermediate={a.intermediate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
