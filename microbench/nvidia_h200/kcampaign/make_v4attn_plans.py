"""Plans for the H200 DeepSeek-V4 attention-component kernel campaign.

Shapes come from config.json (no weights) mirrored under
<workspace>/models/DeepSeek-V4-{Flash,Pro}-config/config.json
(hf-mirror, fetched 2026-09-26), copied into MODEL_CONFIGS below:
head_dim 512 (448 NoPE + 64 RoPE), one latent KV head, sliding_window 128,
compress_ratios 4 (CSA / C4A) and 128 (HCA / C128A) plus SWA-only layers,
index_topk 512 (Flash) / 1024 (Pro), hc_mult 4, hc_sinkhorn_iters 20.

Deployment assumption (vLLM recipes, recipes.vllm.ai/deepseek-ai/...): on
8 x H200 both models run DP attention + EP (Flash --data-parallel-size 4,
Pro --data-parallel-size 8, --max-model-len 800000 for Pro), so each rank
holds all query heads: Flash 64, Pro 128 -- exactly the FlashMLA FP8 decode
padded head counts {64, 128}.  A TP deployment pads its local heads up to
64 (``get_padded_num_q_heads``), i.e. it executes the h_q=64 kernel.  Flash
max_model_len is taken as its max_position_embeddings 1048576 (only the
C128A index width depends on it).  --block-size 256, --kv-cache-dtype fp8
(``fp8_ds_mla``).

Decode cost of a C4A layer stops depending on the context once
context/4 >= index_topk; C128A scales with context/128; SWA-only layers do
not depend on context.  Decode tokens of a multi-token (MTP) step are
separate FlashMLA queries, so batch x q_len = tokens covers MTP.  All cases
use scripts/kernel_dataset/nvidia_h200/kc_v4attn_runner.py under the
canonical nsys_nvtx_kernel_median protocol.
"""

import json
from pathlib import Path

RUNNER = "scripts/kernel_dataset/nvidia_h200/kc_v4attn_runner.py"
OUT = Path(__file__).with_name("plans")
FAMILY = "v4attn"
HEAD_DIM = 512  # config head_dim (448 NoPE + 64 RoPE)
KV_BLOCK_SIZE = 256  # --block-size of the V4 recipes (FLASHMLA_SPARSE_DSV4)

MODEL_CONFIGS = {
    "deepseek-v4-flash": {"num_attention_heads": 64, "index_topk": 512,
                          "sliding_window": 128, "hidden_size": 4096,
                          "hc_mult": 4, "hc_sinkhorn_iters": 20,
                          "max_model_len": 1048576},
    "deepseek-v4-pro": {"num_attention_heads": 128, "index_topk": 1024,
                        "sliding_window": 128, "hidden_size": 7168,
                        "hc_mult": 4, "hc_sinkhorn_iters": 20,
                        "max_model_len": 800000},
}

DECODE_BATCH = (1, 2, 4, 8, 16, 32, 64, 128)
C4_CONTEXT = (1024, 4096, 65536, 262144)
C128_CONTEXT = (4096, 16384, 65536, 262144, 786432)
PREFILL_TOKENS = (512, 1024, 2048, 4096, 8192, 16384)
PREFIX_CONTEXT = 65536
PREFIX_TOKENS = (2048, 8192)
COMPRESSOR_BATCH = (1, 8, 32, 128)
COMPRESSOR_CONTEXT = 65536
MHC_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
              16384)


def plan(case_id, args):
    marker = f"H200_KERNEL::{FAMILY}::{case_id}"
    return {"family": FAMILY, "case_id": case_id, "marker": marker,
            "runner_relpath": RUNNER,
            "runner_args": [*args, "--nvtx_range_name", marker]}


def attn_case(op, model, layer, batch, q_len, context):
    c = MODEL_CONFIGS[model]
    heads = c["num_attention_heads"]
    tag = (f"KC_V4ATTN_{op.upper()}_{layer.upper()}_HQ{heads}_TOPK{c['index_topk']}"
           f"_B{batch}_Q{q_len}_CTX{context}")
    return plan(tag, [
        "--op", op, "--layer-type", layer, "--batch", str(batch),
        "--q-len", str(q_len), "--context", str(context),
        "--local-heads", str(heads), "--padded-heads", str(heads),
        "--index-topk", str(c["index_topk"]), "--window", str(c["sliding_window"]),
        "--max-model-len", str(c["max_model_len"]),
        "--head-dim", str(HEAD_DIM), "--kv-block-size", str(KV_BLOCK_SIZE)])


def v4attn_plans():
    plans = []
    for model in MODEL_CONFIGS:
        for b in DECODE_BATCH:
            plans.append(attn_case("decode", model, "swa", b, 1, 4096))
            for s in C4_CONTEXT:
                plans.append(attn_case("decode", model, "c4", b, 1, s))
            for s in C128_CONTEXT:
                plans.append(attn_case("decode", model, "c128", b, 1, s))
        for layer in ("swa", "c4", "c128"):
            for t in PREFILL_TOKENS:
                plans.append(attn_case("prefill", model, layer, 1, t, 0))
            if layer != "swa":
                for t in PREFIX_TOKENS:
                    plans.append(attn_case("prefill", model, layer, 1, t, PREFIX_CONTEXT))
    for layer in ("c4", "c128"):
        for b in COMPRESSOR_BATCH:
            plans.append(plan(
                f"KC_V4ATTN_COMPRESSOR_{layer.upper()}_B{b}_Q1_CTX{COMPRESSOR_CONTEXT}",
                ["--op", "compressor", "--layer-type", layer, "--batch", str(b),
                 "--q-len", "1", "--context", str(COMPRESSOR_CONTEXT),
                 "--head-dim", str(HEAD_DIM), "--kv-block-size", str(KV_BLOCK_SIZE)]))
        for t in PREFILL_TOKENS:
            plans.append(plan(
                f"KC_V4ATTN_COMPRESSOR_{layer.upper()}_B1_Q{t}_CTX0",
                ["--op", "compressor", "--layer-type", layer, "--batch", "1",
                 "--q-len", str(t), "--context", "0",
                 "--head-dim", str(HEAD_DIM), "--kv-block-size", str(KV_BLOCK_SIZE)]))
    for model, c in MODEL_CONFIGS.items():
        for op in ("mhc_pre", "mhc_post_pre"):
            for t in MHC_TOKENS:
                plans.append(plan(
                    f"KC_V4ATTN_{op.upper()}_H{c['hidden_size']}_HC{c['hc_mult']}_T{t}",
                    ["--op", op, "--batch", str(t), "--q-len", "1", "--context", "0",
                     "--hidden-size", str(c["hidden_size"]),
                     "--hc-mult", str(c["hc_mult"]),
                     "--sinkhorn-iters", str(c["hc_sinkhorn_iters"])]))
    return plans


def write(name, plans, **extra):
    ids = [p["case_id"] for p in plans]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name}: duplicate case ids")
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps({"schema_version": 1, "hardware_id": "nvidia_h200",
                                "campaign": name, "case_count": len(plans),
                                **extra, "plans": plans}, indent=1),
                    encoding="utf-8")
    print(path.name, len(plans))


def main():
    write("kc_v4attn_models", v4attn_plans(), model_configs=MODEL_CONFIGS)


if __name__ == "__main__":
    main()
