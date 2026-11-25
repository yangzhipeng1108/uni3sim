"""Counter-only small BF16 prepared-expert fixture; no timing-based pricing."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, choices=(1, 8, 32), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["VLLM_BATCH_INVARIANT"] = "0"
    import torch
    import vllm
    owner = importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")
    if vllm.__version__ != "0.30.0":
        raise ValueError("this fixture requires vLLM0.30.0")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    hidden, intermediate, experts, local_experts, topk = 2048, 768, 128, 64, 8
    x = torch.randn(args.tokens, hidden, device=device, dtype=torch.bfloat16) * 0.1
    w1 = torch.randn(local_experts, 2 * intermediate, hidden, device=device, dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(local_experts, hidden, intermediate, device=device, dtype=torch.bfloat16) * 0.02
    linear = torch.arange(args.tokens * topk, device=device, dtype=torch.int32) % experts
    ids = ((linear % 2) * local_experts + linear // 2).reshape(args.tokens, topk)
    weights = torch.full((args.tokens, topk), 1 / topk, device=device, dtype=torch.float32)
    expert_map = torch.full((experts,), -1, device=device, dtype=torch.int32)
    expert_map[:local_experts] = torch.arange(local_experts, device=device, dtype=torch.int32)

    def call():
        return owner.fused_experts(x, w1, w2, weights, ids,
            global_num_experts=experts, expert_map=expert_map)

    warm = call()
    call()
    reference = torch.zeros_like(x)
    host_ids = ids.cpu().tolist()
    for token, selected in enumerate(host_ids):
        partials = []
        for expert in selected:
            if expert < local_experts:
                projected = torch.mv(w1[expert].float(), x[token].float()).to(torch.bfloat16)
                gate, up = projected.chunk(2)
                activated = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
                down = torch.mv(w2[expert].float(), activated.float()) / topk
                partials.append(down.to(torch.bfloat16).float())
        if partials:
            reference[token] = torch.stack(partials).sum(0).to(torch.bfloat16)
    torch.testing.assert_close(warm, reference, rtol=0.03, atol=1e-5)
    if not torch.isfinite(warm).all() or warm.abs().max().item() <= 1e-5:
        raise ValueError("prepared-expert fixture must have finite nonzero output")
    drain = torch.empty(256 * 1024 * 1024, device=device, dtype=torch.uint8)
    drain.fill_(17)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("counter_frame")
    output = call()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, rtol=0.03, atol=1e-5)
    counts = [0] * experts
    for row in host_ids:
        for expert in row:
            counts[expert] += 1
    source = Path(owner.__file__)
    payload = {
        "status": "numerically_validated", "torch": torch.__version__,
        "vllm": vllm.__version__, "cuda": torch.version.cuda,
        "source_path": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tokens": args.tokens, "experts": experts, "local_experts": local_experts,
        "expert_rank": 0, "topk": topk, "hidden": hidden, "intermediate": intermediate,
        "tokens_per_expert": counts, "routing_ids": host_ids,
        "local_routed_rows": sum(counts[:local_experts]),
        "active_local_experts": sum(value > 0 for value in counts[:local_experts]),
        "parameter_shapes": {"w1": list(w1.shape), "w2": list(w2.shape)},
        "max_absolute_reference_error": (output.float() - reference.float()).abs().max().item(),
        "operator_duration_measured": False,
        "counter_scope": "two W1/W2 launches inside counter_frame; synthetic balanced global IDs, explicit EP map and cold disjoint traffic",
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"tokens": args.tokens, "status": payload["status"]}), flush=True)


if __name__ == "__main__":
    main()
