"""Plans for the H200 DeepSeek Sparse Attention (DSA) kernel campaign.

Covers the per-layer DSA kernels vLLM 0.24.0 runs on H200 for GLM-5.2
(``glm_moe_dsa``, configs/model/glm-5.2.json) and DeepSeek-V3.2 at TP8 per
rank (see the DSA family section of README.md for the call sites):

* ``kc_dsa_sparse_mla_decode``  -- FLASHMLA_SPARSE attention, decode
  (q_len 1 and 1+MTP), both KV-cache dtypes (``bf16`` = vLLM ``auto`` for a
  BF16 checkpoint, ``fp8_ds_mla`` = ``--kv-cache-dtype fp8``);
* ``kc_dsa_sparse_mla_prefill`` -- the same backend for prefill chunks;
* ``kc_dsa_indexer``            -- indexer logits + top-k (paged decode with
  next_n 1/2 native and 4/6 flattened; non-paged chunked prefill);
* ``kc_dsa_cache_write``        -- indexer K quant+insert and the MLA cache
  insert (fp8_ds_mla and bf16).

MLA attention heads are sharded by TP (``num_heads // tp``); the indexer is
replicated (``ReplicatedLinear`` wq_b), so it keeps the full index_n_heads.
Every plan runs through scripts/kernel_dataset/nvidia_h200/
nsys_campaign_runner.py; nothing here changes how a case is measured.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNERS = "scripts/kernel_dataset/nvidia_h200"
OUT = Path(__file__).with_name("plans")

TP = 8
# One deployment max_model_len for every case (>= the largest context); it
# only sizes the paged-logits row stride and the fp8 cache_seqlens tensor.
MAX_MODEL_LEN = 131072
DECODE_BATCH = (1, 2, 4, 8, 16, 32, 64, 128)
CONTEXTS = (2048, 4096, 8192, 16384, 32768, 65536)
PREFILL_TOKENS = (512, 1024, 2048, 4096, 8192, 16384)
KV_CACHE_DTYPES = ("fp8_ds_mla", "bf16")
CACHE_WRITE_PREFILL_TOKENS = PREFILL_TOKENS
CACHE_WRITE_DECODE_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768)

# DeepSeek-V3.2 (HF deepseek-ai/DeepSeek-V3.2 config.json); no local config
# file exists for it in configs/model.
DEEPSEEK_V32 = {
    "num_attention_heads": 128, "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64, "v_head_dim": 128, "kv_lora_rank": 512,
    "index_n_heads": 64, "index_head_dim": 128, "index_topk": 2048,
}


def load_glm():
    return json.loads((ROOT / "configs/model/glm-5.2.json").read_text())


def models():
    glm = load_glm()
    keys = ("num_attention_heads", "qk_nope_head_dim", "qk_rope_head_dim",
            "v_head_dim", "kv_lora_rank", "index_n_heads", "index_head_dim",
            "index_topk")
    out = {"glm52": {key: glm[key] for key in keys},
           "dsv32": dict(DEEPSEEK_V32)}
    for name, config in out.items():
        if config["kv_lora_rank"] != 512 or config["qk_rope_head_dim"] != 64:
            raise ValueError(f"{name}: FlashMLA sparse needs 512+64 latent")
        if config["num_attention_heads"] % TP != 0:
            raise ValueError(f"{name}: heads not divisible by TP{TP}")
    return out


def plan(family, case_id, runner, args):
    marker = f"H200_KERNEL::{family}::{case_id}"
    return {"family": family, "case_id": case_id, "marker": marker,
            "runner_relpath": f"{RUNNERS}/{runner}",
            "runner_args": [*args, "--nvtx_range_name", marker]}


def mla_case(tag, config, kv_dtype, batch, q_len, context, spec_tokens):
    heads = config["num_attention_heads"] // TP
    case_id = (f"KC_DSA_MLA_{tag.upper()}_TP{TP}_H{heads}_{kv_dtype.upper()}"
               f"_B{batch}_Q{q_len}_C{context}")
    return plan("dsa_sparse_mla", case_id, "kc_dsa_sparse_mla_runner.py", [
        "--batch", str(batch), "--q-len", str(q_len),
        "--context", str(context), "--num-heads", str(heads),
        "--qk-nope-head-dim", str(config["qk_nope_head_dim"]),
        "--kv-cache-dtype", kv_dtype, "--topk", str(config["index_topk"]),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--num-speculative-tokens", str(spec_tokens)])


def prefill_contexts(tokens):
    return (tokens, *[c for c in CONTEXTS if c > tokens])


def sparse_mla_decode_plans(cfgs):
    # GLM-5.2 is served with MTP (num_nextn_predict_layers=1; the local
    # InferenceX recipe uses 5 speculative tokens) -> q_len 1, 2, 4, 6.
    q_lens = {"glm52": (1, 2, 4, 6), "dsv32": (1, 2)}
    plans = []
    for tag, config in cfgs.items():
        for kv_dtype in KV_CACHE_DTYPES:
            for q_len in q_lens[tag]:
                for batch in DECODE_BATCH:
                    for context in CONTEXTS:
                        plans.append(mla_case(tag, config, kv_dtype, batch,
                                              q_len, context, q_len - 1))
    return plans


def sparse_mla_prefill_plans(cfgs):
    plans = []
    for tag, config in cfgs.items():
        for kv_dtype in KV_CACHE_DTYPES:
            for tokens in PREFILL_TOKENS:
                for context in prefill_contexts(tokens):
                    plans.append(mla_case(tag, config, kv_dtype, 1, tokens,
                                          context, 0))
    return plans


def indexer_case(tag, config, phase, batch, q_len, context):
    heads = config["index_n_heads"]
    case_id = (f"KC_DSA_INDEXER_{phase.upper()}_{tag.upper()}_IH{heads}"
               f"_B{batch}_Q{q_len}_C{context}")
    return plan("dsa_indexer", case_id, "kc_dsa_indexer_runner.py", [
        "--phase", phase, "--batch", str(batch), "--q-len", str(q_len),
        "--context", str(context), "--index-n-heads", str(heads),
        "--index-head-dim", str(config["index_head_dim"]),
        "--topk", str(config["index_topk"]),
        "--max-model-len", str(MAX_MODEL_LEN)])


def indexer_plans(cfgs):
    plans = []
    for tag, config in cfgs.items():
        for next_n in (1, 2, 4, 6):
            for batch in DECODE_BATCH:
                for context in CONTEXTS:
                    plans.append(indexer_case(tag, config, "decode", batch,
                                              next_n, context))
        for tokens in PREFILL_TOKENS:
            for context in prefill_contexts(tokens):
                plans.append(indexer_case(tag, config, "prefill", 1, tokens,
                                          context))
    return plans


def cache_write_plans():
    plans = []
    for op in ("indexer_k", "mla_fp8_ds_mla", "mla_bf16"):
        for layout, tokens_list in (("prefill", CACHE_WRITE_PREFILL_TOKENS),
                                    ("decode", CACHE_WRITE_DECODE_TOKENS)):
            for tokens in tokens_list:
                case_id = (f"KC_DSA_CACHE_{op.upper()}_{layout.upper()}"
                           f"_T{tokens}")
                plans.append(plan(
                    "dsa_cache_write", case_id, "kc_dsa_cache_write_runner.py",
                    ["--op", op, "--tokens", str(tokens),
                     "--slot-layout", layout]))
    return plans


def fused_norm_rope_plans(glm):
    """vLLM 0.30-only: the fused norm/RoPE/cache-insert kernel that replaces
    the standalone inserts in the DeepSeek-V3.2 / GLM-5.2 CUDA model.

    GLM-5.2 only (its config is local): index layers (has_indexer 1) and the
    shared top-k layers (index_topk_freq 4 -> has_indexer 0)."""
    plans = []
    for has_indexer in (1, 0):
        for kv_dtype in KV_CACHE_DTYPES:
            for layout, tokens_list in (
                    ("prefill", CACHE_WRITE_PREFILL_TOKENS),
                    ("decode", CACHE_WRITE_DECODE_TOKENS)):
                for tokens in tokens_list:
                    case_id = (f"KC_DSA_FUSED_NORM_ROPE_GLM52_IDX{has_indexer}"
                               f"_{kv_dtype.upper()}_{layout.upper()}"
                               f"_T{tokens}")
                    plans.append(plan(
                        "dsa_cache_write", case_id,
                        "kc_dsa_cache_write_runner.py",
                        ["--op", "fused_norm_rope", "--tokens", str(tokens),
                         "--slot-layout", layout, "--kv-cache-dtype",
                         kv_dtype, "--q-lora-rank", str(glm["q_lora_rank"]),
                         "--has-indexer", str(has_indexer),
                         "--index-n-heads", str(glm["index_n_heads"]),
                         "--index-rope-interleave",
                         str(int(glm["indexer_rope_interleave"])),
                         "--topk", str(glm["index_topk"])]))
    return plans


def write(name, plans, extra):
    ids = [item["case_id"] for item in plans]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name}: duplicate case ids")
    payload = {"plans": plans, "tp": TP, "max_model_len": MAX_MODEL_LEN,
               **extra}
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{name}: {len(plans)} cases")


def main():
    cfgs = models()
    source = {"models": cfgs,
              "model_sources": {"glm52": "configs/model/glm-5.2.json",
                                "dsv32": "HF deepseek-ai/DeepSeek-V3.2"}}
    write("kc_dsa_sparse_mla_decode", sparse_mla_decode_plans(cfgs), source)
    write("kc_dsa_sparse_mla_prefill", sparse_mla_prefill_plans(cfgs), source)
    write("kc_dsa_indexer", indexer_plans(cfgs), source)
    write("kc_dsa_cache_write", cache_write_plans(), source)
    write("kc_dsa_fused_norm_rope__v030", fused_norm_rope_plans(load_glm()),
          {**source, "vllm_stack": "vllm030"})


if __name__ == "__main__":
    main()
