"""Plans for the H200 Kimi Delta Attention (KDA) kernel campaign.

Shapes are per tensor-parallel rank, derived from config.json (no weights)
mirrored under <workspace>/models/<name>-config/config.json
(hf-mirror, fetched 2026-09-26) and copied into MODEL_CONFIGS below:

* Kimi-K3 (moonshotai/Kimi-K3, text_config.linear_attn_config): 96 KDA heads,
  head_dim 128, short_conv_kernel_size 4, gate_lower_bound -5.0,
  use_full_rank_gate.  The vLLM recipe (recipes.vllm.ai/moonshotai/Kimi-K3,
  vLLM >= 0.29) serves it at TP8 (12 local heads) or TEP16 (assumed TP16
  attention, 6 local heads).
* Kimi-Linear-48B-A3B-Instruct (the KDA model vLLM 0.24 ships,
  KimiLinearForCausalLM): 32 KDA heads, head_dim 128, conv 4, at TP1.

The KDA recurrent state is a fixed (H, D, D) FP32 matrix per request, so KDA
decode cost has no KV-context dimension: decode plans sweep batch (and,
for the FLA call only, q_len 2/4 multi-token verification), prefill plans
sweep packed tokens with one or four requests and with/without an initial
state.

``vllm_*`` cases run the vLLM 0.24 KimiLinear KDA kernels at these shapes
(K3's lower-bound gate is not implemented there); ``fla_*`` cases run the
Kimi-K3 reference call (flash-linear-attention 0.5.2, safe_gate +
lower_bound) from FLA_PYDEPS.  All cases use
scripts/kernel_dataset/nvidia_h200/kc_kda_runner.py under the canonical
nsys_nvtx_kernel_median protocol.

Kimi-K3 "Gated MLA" (full-attention layers: mla_use_nope, mla_use_output_gate)
is DeepSeek-style MLA (kv_lora_rank 512, qk_rope_head_dim 64 -- carried but
not rotated, v_head_dim 128) followed by ``o * sigmoid(g_proj(x))``; the
attention kernel itself is the standard MLA decode kernel.  The
kc_kda_k3_gated_mla plan therefore reuses the existing
scripts/kernel_dataset/nvidia_h200/flash_attn_runner.py ``mla`` layout (vLLM
0.24 FLASH_ATTN_MLA decode on SM90) at K3's per-rank head counts.
"""

import json
import os
from pathlib import Path

RUNNER = "scripts/kernel_dataset/nvidia_h200/kc_kda_runner.py"
OUT = Path(__file__).with_name("plans")
FAMILY = "kda"
if "FLA_PYDEPS" not in os.environ or not os.environ["FLA_PYDEPS"].strip():
    raise SystemExit("FLA_PYDEPS must name the caller's FLA 0.5.2 dependency path")
FLA_PYDEPS = os.environ["FLA_PYDEPS"]
# vLLM 0.24 FLA_CHUNK_SIZE and fla 0.5.2 chunk_kda default chunk_size.
CHUNK_SIZE = 64

MODEL_CONFIGS = {
    "kimi-k3": {"num_heads": 96, "head_dim": 128, "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0},
    "kimi-linear-48b-a3b": {"num_heads": 32, "head_dim": 128,
                            "short_conv_kernel_size": 4,
                            "gate_lower_bound": None},
}
# (model, tp) -> per-rank shape.
DEPLOYMENTS = (("kimi-k3", 8), ("kimi-k3", 16), ("kimi-linear-48b-a3b", 1))

DECODE_BATCH = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SPEC_BATCH = (1, 8, 32, 128)
SPEC_Q_LEN = (2, 4)
PREFILL_TOKENS = (512, 1024, 2048, 4096, 8192, 16384)
NORM_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
               8192, 16384)


def plan(case_id, args):
    marker = f"H200_KERNEL::{FAMILY}::{case_id}"
    return {"family": FAMILY, "case_id": case_id, "marker": marker,
            "runner_relpath": RUNNER,
            "runner_args": [*args, "--nvtx_range_name", marker]}


def case(op, heads, head_dim, conv, batch, q_len, init, lower_bound):
    tag = f"KC_KDA_{op.upper()}_H{heads}_D{head_dim}_B{batch}_Q{q_len}"
    if op in ("vllm_conv_prefill", "vllm_chunk", "fla_chunk",
              "k3_prefill_flashkda", "k3_prefill_triton"):
        tag += f"_INIT{init}"
    args = ["--op", op, "--num-heads", str(heads), "--head-dim", str(head_dim),
            "--conv-kernel", str(conv), "--batch", str(batch),
            "--q-len", str(q_len), "--has-initial-state", str(init),
            "--chunk-size", str(CHUNK_SIZE)]
    if op.startswith("fla_"):
        args += ["--gate-lower-bound", str(lower_bound),
                 "--fla-pydeps", FLA_PYDEPS]
    if op.startswith("k3_"):
        args += ["--gate-lower-bound", str(lower_bound)]
    return plan(tag, args)


def kda_plans():
    plans, origin = [], {}
    for model, tp in DEPLOYMENTS:
        c = MODEL_CONFIGS[model]
        if c["num_heads"] % tp:
            raise ValueError(f"{model}: {c['num_heads']} heads not divisible by TP{tp}")
        h, d, w = c["num_heads"] // tp, c["head_dim"], c["short_conv_kernel_size"]
        origin[f"H{h}_D{d}"] = f"{model}_tp{tp}"
        ops = ["vllm"] + (["fla"] if c["gate_lower_bound"] is not None else [])
        for stack in ops:
            lb = c["gate_lower_bound"]
            recurrent = "vllm_decode" if stack == "vllm" else "fla_recurrent"
            chunk = "vllm_chunk" if stack == "vllm" else "fla_chunk"
            for b in DECODE_BATCH:
                plans.append(case(recurrent, h, d, w, b, 1, 1, lb))
            if stack == "fla":
                # Multi-token verification: only the FLA recurrent call takes
                # [B, q_len] natively (vLLM 0.24 KDA decode is q_len 1).
                for b in SPEC_BATCH:
                    for q in SPEC_Q_LEN:
                        plans.append(case(recurrent, h, d, w, b, q, 1, lb))
            for t in PREFILL_TOKENS:
                plans.append(case(chunk, h, d, w, 1, t, 0, lb))
                plans.append(case(chunk, h, d, w, 1, t, 1, lb))
                plans.append(case(chunk, h, d, w, 4, t // 4, 0, lb))
            if stack != "vllm":
                continue
            for b in DECODE_BATCH:
                plans.append(case("vllm_conv_decode", h, d, w, b, 1, 1, None))
            for t in PREFILL_TOKENS:
                plans.append(case("vllm_conv_prefill", h, d, w, 1, t, 0, None))
                plans.append(case("vllm_conv_prefill", h, d, w, 4, t // 4, 1, None))
            for t in NORM_TOKENS:
                plans.append(case("vllm_o_norm", h, d, w, t, 1, 1, None))
    return plans, origin


MLA_BATCH = (1, 2, 4, 8, 16, 32, 64, 128)
MLA_CONTEXT = (1024, 4096, 16384, 65536, 131072)
K3_MLA = {"num_attention_heads": 96, "kv_lora_rank": 512}


def gated_mla_plans():
    plans = []
    for tp in (8, 16):
        hq = K3_MLA["num_attention_heads"] // tp
        for b in MLA_BATCH:
            for s in MLA_CONTEXT:
                case_id = f"KC_KDA_K3_GATED_MLA_DECODE_HQ{hq}_B{b}_S{s}"
                marker = f"H200_KERNEL::flash_attn::{case_id}"
                plans.append({
                    "family": "flash_attn", "case_id": case_id, "marker": marker,
                    "runner_relpath": "scripts/kernel_dataset/nvidia_h200/flash_attn_runner.py",
                    "runner_args": [
                        "--batch", str(b), "--seq_len_cache", str(s),
                        "--num_heads_q", str(hq), "--num_heads_kv", "1",
                        "--head_dim", str(K3_MLA["kv_lora_rank"]), "--dtype", "bf16",
                        "--kv_layout", "mla", "--kv_dist", "uniform",
                        "--seq_len_q", "1", "--q-lens-json", json.dumps([1] * b),
                        "--kv-lens-json", json.dumps([s] * b), "--causal", "1",
                        "--num_splits", "1", "--nvtx_range_name", marker]})
    return plans


# ---- vLLM 0.30.0 Kimi-K3 native paths (plans named *__v030) ----
# KimiK3DeltaAttention._forward (vllm/models/kimi_k3/nvidia/kda.py).  The
# fused native decode (torch.ops._C.fused_kda_decode, 12/24/48/96 local heads)
# is compiled only by CUDA >= 13 builds (CMakeLists.txt FUSED_KDA_DECODE_ARCHS);
# the vllm-0.30.0+cu129 wheel the H200 (driver 575 / CUDA 12.9) can run lacks
# it, so resolve_kda_decode_backend("auto") returns "triton" at every head
# count and the plan has no k3_decode_native cases (the runner op fails loudly
# on this build).  k3_prefill_triton is the kda_prefill_backend="triton" path
# (auto picks FlashKDA on SM90).
K3_NATIVE_DECODE_BUILT = False
K3_NATIVE_DECODE_HEADS = (12, 24, 48, 96)


def k3_native_plans():
    plans = []
    c = MODEL_CONFIGS["kimi-k3"]
    for tp in (8, 16):
        h, d, w = c["num_heads"] // tp, c["head_dim"], c["short_conv_kernel_size"]
        lb = c["gate_lower_bound"]
        for op in ("k3_prefill_flashkda", "k3_prefill_triton"):
            for t in PREFILL_TOKENS:
                plans.append(case(op, h, d, w, 1, t, 0, lb))
                plans.append(case(op, h, d, w, 1, t, 1, lb))
                plans.append(case(op, h, d, w, 4, t // 4, 0, lb))
        decode_ops = ["k3_decode_triton"]
        if K3_NATIVE_DECODE_BUILT and h in K3_NATIVE_DECODE_HEADS:
            decode_ops.insert(0, "k3_decode_native")
        for op in decode_ops:
            for b in DECODE_BATCH:
                plans.append(case(op, h, d, w, b, 1, 1, lb))
        for b in SPEC_BATCH:
            for q in SPEC_Q_LEN:
                plans.append(case("k3_spec_decode", h, d, w, b, q, 1, lb))
    return plans


# Kimi-K3 MLA (kimi_k3/nvidia/mla.py) from config.json: q_lora_rank 1536,
# kv_lora_rank 512, qk_nope 128, qk_rope 64, v 128; bf16 KV cache
# (kv_cache_scheme null).  Block size = Platform._align_hybrid_block_size:
# MLA kernel alignment max(16, 128) = 128 tokens times
# cdiv(KDA state page, 128 * 576 * 2 B); KDA page = conv (w-1) * 3 * H*D * 2 B
# + recurrent H * D * D * 4 B per rank (no speculative tokens).
K3_MLA_FULL = {"num_attention_heads": 96, "q_lora_rank": 1536,
               "kv_lora_rank": 512, "qk_nope_head_dim": 128,
               "qk_rope_head_dim": 64, "v_head_dim": 128}


def k3_mla_block_size(tp):
    c = MODEL_CONFIGS["kimi-k3"]
    h, d, w = c["num_heads"] // tp, c["head_dim"], c["short_conv_kernel_size"]
    kda_page = (w - 1) * 3 * h * d * 2 + h * d * d * 4
    token_page = (K3_MLA_FULL["kv_lora_rank"] + K3_MLA_FULL["qk_rope_head_dim"]) * 2
    align = 128
    return align * -(-kda_page // (align * token_page))


def k3_mla_plans():
    plans = []
    runner = "scripts/kernel_dataset/nvidia_h200/kc_k3_mla_runner.py"
    for tp in (8, 16):
        hq = K3_MLA_FULL["num_attention_heads"] // tp
        bs = k3_mla_block_size(tp)
        for b in MLA_BATCH:
            for s in MLA_CONTEXT:
                case_id = f"KC_K3_MLA_DECODE_HQ{hq}_B{b}_S{s}_BLK{bs}"
                marker = f"H200_KERNEL::k3_mla::{case_id}"
                plans.append({
                    "family": "k3_mla", "case_id": case_id, "marker": marker,
                    "runner_relpath": runner,
                    "runner_args": [
                        "--op", "decode", "--num-heads", str(hq), "--batch", str(b),
                        "--context", str(s), "--kv-block-size", str(bs),
                        "--kv-lora-rank", str(K3_MLA_FULL["kv_lora_rank"]),
                        "--qk-nope-head-dim", str(K3_MLA_FULL["qk_nope_head_dim"]),
                        "--qk-rope-head-dim", str(K3_MLA_FULL["qk_rope_head_dim"]),
                        "--v-head-dim", str(K3_MLA_FULL["v_head_dim"]),
                        "--q-lora-rank", str(K3_MLA_FULL["q_lora_rank"]),
                        "--nvtx_range_name", marker]})
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
    plans, origin = kda_plans()
    write("kc_kda_models", plans, shape_origin=origin,
          model_configs=MODEL_CONFIGS)
    write("kc_kda_k3_gated_mla", gated_mla_plans(), model_configs={"kimi-k3": K3_MLA})
    write("kc_kda_k3_native__v030", k3_native_plans(),
          model_configs={"kimi-k3": MODEL_CONFIGS["kimi-k3"]})
    write("kc_kda_k3_mla__v030", k3_mla_plans(),
          model_configs={"kimi-k3": K3_MLA_FULL},
          kv_block_size={f"tp{tp}": k3_mla_block_size(tp) for tp in (8, 16)})


if __name__ == "__main__":
    main()
