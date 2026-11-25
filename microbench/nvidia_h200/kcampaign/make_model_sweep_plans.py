"""Plans for the H200 kernel campaign beyond the A800 canonical case list.

Shapes are derived from configs/model/*.json at the tensor-parallel degrees
the models are served with (serving-dataset models at TP1/TP2, GLM-5.2 and
Kimi-K2.6 BF16 layers at TP8), with denser token grids than the canonical
dataset.  Every plan runs an existing H200 dataset runner under the canonical
nsys_nvtx_kernel_median protocol (scripts/kernel_dataset/nvidia_h200/
nsys_campaign_runner.py); nothing here changes how a case is measured.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNERS = "scripts/kernel_dataset/nvidia_h200"
OUT = Path(__file__).with_name("plans")

GEMM_M = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384,
          512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384)
VECTOR_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
MOE_TOKENS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256,
              384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288,
              16384, 32768)
FA_DECODE_B = (1, 2, 4, 8, 16, 32, 64, 128, 256)
FA_DECODE_KV = (256, 512, 1024, 2048, 4096, 8192)
FA_PREFILL_S = (128, 256, 512, 1024, 2048, 4096, 8192)

DENSE_MODELS = ("qwen3-4b", "qwen3-8b", "llama-3.1-8b-instruct", "qwen3-32b",
                "qwen3-30b-a3b")


def load(model):
    config = json.loads((ROOT / "configs/model" / f"{model}.json").read_text())
    return config["text_config"] if "text_config" in config else config


def plan(family, case_id, runner, args):
    marker = f"H200_KERNEL::{family}::{case_id}"
    return {"family": family, "case_id": case_id, "marker": marker,
            "runner_relpath": f"{RUNNERS}/{runner}",
            "runner_args": [*args, "--nvtx_range_name", marker]}


def gqa_dims(config):
    head_dim = (config["head_dim"] if "head_dim" in config
                else config["hidden_size"] // config["num_attention_heads"])
    return (config["hidden_size"], config["num_attention_heads"],
            config["num_key_value_heads"], head_dim)


def dense_gemm_pairs():
    """(K, N, tag) per rank for the serving-dataset models at TP1 and TP2."""
    pairs = []
    for model in DENSE_MODELS:
        c = load(model)
        hidden, heads, kv_heads, head_dim = gqa_dims(c)
        for tp in (1, 2):
            q, kv = heads * head_dim // tp, kv_heads * head_dim // tp
            layers = [("qkv", hidden, q + 2 * kv), ("o", q, hidden),
                      ("lm_head", hidden, c["vocab_size"] // tp)]
            if "num_experts" in c:
                layers.append(("router", hidden, c["num_experts"]))
            else:
                inter = c["intermediate_size"] // tp
                layers += [("gate_up", hidden, 2 * inter), ("down", inter, hidden)]
            pairs += [(k, n, f"{model}_tp{tp}_{name}") for name, k, n in layers]
    return pairs


def mla_bf16_gemm_pairs():
    """BF16 linears of GLM-5.2 (BF16 checkpoint) and Kimi-K2.6 (non-expert
    BF16 layers) at TP8 per rank.  MLA q/kv down projections are fused
    (q_lora_rank + kv_lora_rank + rope) and replicated, as in vLLM."""
    tp = 8
    pairs = []
    for model in ("glm-5.2", "kimi-k2.6"):
        c = load(model)
        hidden, heads = c["hidden_size"], c["num_attention_heads"]
        qk = c["qk_nope_head_dim"] + c["qk_rope_head_dim"]
        layers = [
            ("fused_qkv_a", hidden,
             c["q_lora_rank"] + c["kv_lora_rank"] + c["qk_rope_head_dim"]),
            ("q_b", c["q_lora_rank"], heads * qk // tp),
            ("kv_b", c["kv_lora_rank"],
             heads * (c["qk_nope_head_dim"] + c["v_head_dim"]) // tp),
            ("o", heads * c["v_head_dim"] // tp, hidden),
            ("shared_gate_up", hidden,
             2 * c["n_shared_experts"] * c["moe_intermediate_size"] // tp),
            ("shared_down", c["n_shared_experts"] * c["moe_intermediate_size"] // tp,
             hidden),
            ("dense_gate_up", hidden, 2 * c["intermediate_size"] // tp),
            ("dense_down", c["intermediate_size"] // tp, hidden),
            ("router", hidden, c["n_routed_experts"]),
            ("lm_head", hidden, c["vocab_size"] // tp),
        ]
        if "index_n_heads" in c:
            layers += [
                ("indexer_wq_b", c["q_lora_rank"],
                 c["index_n_heads"] * c["index_head_dim"]),
                ("indexer_wk", hidden, c["index_head_dim"]),
                ("indexer_weights_proj", hidden, c["index_n_heads"]),
            ]
        pairs += [(k, n, f"{model}_tp{tp}_{name}") for name, k, n in layers]
    return pairs


def gemm_plans(pairs):
    seen, plans = {}, []
    for k, n, tag in pairs:
        if (k, n) not in seen:
            seen[(k, n)] = []
        seen[(k, n)].append(tag)
    for (k, n), tags in sorted(seen.items()):
        for m in GEMM_M:
            plans.append(plan("gemm", f"KC_GEMM_BF16_M{m}_K{k}_N{n}",
                              "gemm_runner.py",
                              ["--bs", "1", "--sl", str(m), "--in_feat", str(k),
                               "--out_feat", str(n), "--dtype", "bf16"]))
    return plans, {f"K{k}_N{n}": tags for (k, n), tags in sorted(seen.items())}


def fa_plans():
    heads = set()
    for model in DENSE_MODELS:
        _, q, kv, head_dim = gqa_dims(load(model))
        for tp in (1, 2):
            heads.add((q // tp, kv // tp, head_dim))
    plans = []
    for hq, hkv, d in sorted(heads):
        common = ["--num_heads_q", str(hq), "--num_heads_kv", str(hkv),
                  "--head_dim", str(d), "--dtype", "bf16", "--kv_layout", "paged",
                  "--kv_dist", "uniform"]
        for b in FA_DECODE_B:
            for s in FA_DECODE_KV:
                plans.append(plan(
                    "flash_attn", f"KC_FA_PAGED_DECODE_HQ{hq}_HKV{hkv}_D{d}_B{b}_S{s}",
                    "flash_attn_runner.py",
                    ["--batch", str(b), "--seq_len_cache", str(s), *common,
                     "--seq_len_q", "1", "--q-lens-json", json.dumps([1] * b),
                     "--kv-lens-json", json.dumps([s] * b), "--causal", "0",
                     "--num_splits", "1"]))
        for b in (1, 4):
            for s in FA_PREFILL_S:
                plans.append(plan(
                    "flash_attn", f"KC_FA_PAGED_PREFILL_HQ{hq}_HKV{hkv}_D{d}_B{b}_S{s}",
                    "flash_attn_runner.py",
                    ["--batch", str(b), "--seq_len_cache", str(s), *common,
                     "--seq_len_q", str(s), "--q-lens-json", json.dumps([s] * b),
                     "--kv-lens-json", json.dumps([s] * b), "--causal", "1",
                     "--num_splits", "1"]))
    return plans


def vector_args(op, tok, hidden=4096, heads=32, head_dim=128, inter=6144,
                kv_heads=4):
    return ["--op", op, "--tok", str(tok), "--dtype", "bf16", "--batch", "1",
            "--op2", "add", "--hidden", str(hidden), "--num_heads", str(heads),
            "--head_dim", str(head_dim), "--intermediate", str(inter),
            "--num_kv_heads", str(kv_heads)]


def vector_plans():
    hiddens, inters, ropes, kvs = set(), set(), set(), set()
    for model in DENSE_MODELS:
        c = load(model)
        hidden, q, kv, head_dim = gqa_dims(c)
        hiddens.add(hidden)
        for tp in (1, 2):
            ropes.add((q // tp, head_dim, kv // tp))
            kvs.add((kv // tp, head_dim))
            if "num_experts" not in c:
                inters.add(c["intermediate_size"] // tp)
    for model in ("glm-5.2", "kimi-k2.6", "deepseek-v3"):
        hiddens.add(load(model)["hidden_size"])
    plans = []
    for tok in VECTOR_TOKENS:
        for h in sorted(hiddens):
            for op in ("rmsnorm", "add_rmsnorm"):
                plans.append(plan("vector", f"KC_{op.upper()}_T{tok}_H{h}",
                                  "vector_runner.py", vector_args(op, tok, hidden=h)))
        for inter in sorted(inters):
            plans.append(plan("vector", f"KC_SILU_T{tok}_I{inter}",
                              "vector_runner.py", vector_args("silu", tok, inter=inter)))
        for heads, head_dim, kv_heads in sorted(ropes):
            plans.append(plan(
                "vector", f"KC_ROPE_T{tok}_HQ{heads}_D{head_dim}_HKV{kv_heads}",
                "vector_runner.py",
                vector_args("rope", tok, heads=heads, head_dim=head_dim,
                            kv_heads=kv_heads)))
        for kv_heads, head_dim in sorted(kvs):
            plans.append(plan(
                "vector", f"KC_KVCACHE_T{tok}_HKV{kv_heads}_D{head_dim}",
                "vector_runner.py",
                vector_args("kvcache", tok, head_dim=head_dim, kv_heads=kv_heads)))
    return plans


def moe_plans():
    """BF16 Triton fused MoE.  Routing is the runner's seeded random router
    (seed 42), whose tok_per_expert is recorded in the launch evidence."""
    c30 = load("qwen3-30b-a3b")
    glm = load("glm-5.2")
    configs = [
        ("qwen3-30b-a3b_tp1", c30["num_experts"], c30["num_experts_per_tok"],
         c30["hidden_size"], c30["moe_intermediate_size"]),
        ("qwen3-30b-a3b_tp2", c30["num_experts"], c30["num_experts_per_tok"],
         c30["hidden_size"], c30["moe_intermediate_size"] // 2),
        ("glm-5.2_tp8", glm["n_routed_experts"], glm["num_experts_per_tok"],
         glm["hidden_size"], glm["moe_intermediate_size"] // 8),
    ]
    plans = []
    for _, experts, topk, hidden, inter in configs:
        for tok in MOE_TOKENS:
            plans.append(plan(
                "fused_moe",
                f"KC_FUSEDMOE_BF16_TOK{tok}_E{experts}_TOPK{topk}_H{hidden}_INT{inter}",
                "fused_moe_runner.py",
                ["--num_tokens", str(tok), "--num_experts", str(experts),
                 "--top_k", str(topk), "--hidden_size", str(hidden),
                 "--intermediate_size", str(inter), "--dtype", "bf16"]))
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
    dense, dense_tags = gemm_plans(dense_gemm_pairs())
    mla, mla_tags = gemm_plans(mla_bf16_gemm_pairs())
    write("kc_gemm_bf16_serving_models", dense, shape_origin=dense_tags)
    write("kc_gemm_bf16_mla_models", mla, shape_origin=mla_tags)
    write("kc_fa_paged_serving_models", fa_plans())
    write("kc_vector_models", vector_plans())
    write("kc_moe_bf16_models", moe_plans())


if __name__ == "__main__":
    main()


# Dense decode-range M sweep (filler that keeps GPUs busy between new-op
# plans): every M in 1..128 plus 136..256 step 8, excluding the M values the
# coarse grid already covers, for the TP1 per-layer GEMMs of the serving models.
DENSE_DECODE_M = tuple(m for m in (*range(1, 129), *range(136, 257, 8))
                       if m not in GEMM_M)


def dense_decode_gemm_plans(models):
    pairs = [(k, n, tag) for k, n, tag in dense_gemm_pairs()
             if tag.split("_tp")[0] in models and "_tp1_" in tag
             and not tag.endswith("lm_head")]
    seen = {}
    for k, n, tag in pairs:
        if (k, n) not in seen:
            seen[(k, n)] = []
        seen[(k, n)].append(tag)
    plans = [plan("gemm", f"KC_GEMM_BF16_M{m}_K{k}_N{n}", "gemm_runner.py",
                  ["--bs", "1", "--sl", str(m), "--in_feat", str(k),
                   "--out_feat", str(n), "--dtype", "bf16"])
             for (k, n) in sorted(seen) for m in DENSE_DECODE_M]
    return plans, {f"K{k}_N{n}": tags for (k, n), tags in sorted(seen.items())}


def main_dense_decode():
    for name, models in (("kc_gemm_bf16_dense_decode_a", ("qwen3-4b", "qwen3-8b")),
                         ("kc_gemm_bf16_dense_decode_b",
                          ("llama-3.1-8b-instruct", "qwen3-32b", "qwen3-30b-a3b"))):
        plans, tags = dense_decode_gemm_plans(models)
        write(name, plans, shape_origin=tags)


if __name__ == "__main__" and "--dense-decode" in __import__("sys").argv:
    main_dense_decode()
