"""Plans for the H200 weight-only-quantized MoE kernel campaign.

Shapes come from the models' config.json (no weights):

* Kimi-K2.6 (configs/model/kimi-k2.6.json): compressed-tensors pack-quantized
  INT4 W4A16, group 32, routed experts only.  Served on 8 x H200 at TP8
  (per-rank expert intermediate 2048/8) or with expert parallelism EP8
  (48 local experts of the full intermediate 2048).
* DeepSeek-V4-Flash / DeepSeek-V4-Pro (expert_dtype fp4, swiglu_limit 10)
  and Kimi-K3 (compressed-tensors mxfp4-pack-quantized, routed experts run at
  routed_expert_hidden_size): MXFP4 W4A16 Marlin on SM90.  The configs are
  mirrored under <workspace>/models/<name>-config/config.json and
  copied into the MODEL_CONFIGS constants below (only the fields used).

Every case runs scripts/kernel_dataset/nvidia_h200/kc_wq_moe_runner.py under
the canonical nsys_nvtx_kernel_median protocol; routing is the runner's
seeded softmax/top-k router, whose tok_per_expert is launch evidence.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = "scripts/kernel_dataset/nvidia_h200/kc_wq_moe_runner.py"
OUT = Path(__file__).with_name("plans")
FAMILY = "wq_fused_moe"

TOKENS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384,
          512, 1024, 2048, 4096, 8192, 16384)

# Fields copied from the hf-mirror config.json files (fetched 2026-09-26 into
# <workspace>/models/<name>-config/config.json).
MODEL_CONFIGS = {
    "deepseek-v4-flash": {"hidden_size": 4096, "moe_intermediate_size": 2048,
                          "n_routed_experts": 256, "num_experts_per_tok": 6,
                          "swiglu_limit": 10.0, "expert_dtype": "fp4"},
    "deepseek-v4-pro": {"hidden_size": 7168, "moe_intermediate_size": 3072,
                        "n_routed_experts": 384, "num_experts_per_tok": 6,
                        "swiglu_limit": 10.0, "expert_dtype": "fp4"},
    # text_config: routed experts operate at routed_expert_hidden_size;
    # hidden_act "situ" is not implemented in vLLM 0.24 (no Kimi-K3 model,
    # no MoEActivation for it), so the plan uses SiLU: the two Marlin GEMMs
    # are the K3 ones, the activation kernel is a SiLU stand-in.
    "kimi-k3": {"hidden_size": 3584, "moe_intermediate_size": 3072,
                "n_routed_experts": 896, "num_experts_per_tok": 16,
                "swiglu_limit": None, "expert_dtype": "mxfp4-pack-quantized"},
}

# Kimi-K3 text_config hidden_act "situ" with its betas (same config.json).
# vLLM 0.30 runs it as MoEActivation.SITU (torch.ops._C.situ_and_mul), so
# the __v030 plan below measures K3's own activation instead of the SiLU
# stand-in of kc_wq_moe_mxfp4_kimi_k3.
K3_SITU = {"hidden_act": "situ", "activation_situ_beta": 4.0,
           "activation_situ_linear_beta": 25.0}


def load_k26():
    config = json.loads((ROOT / "configs/model/kimi-k2.6.json").read_text())
    text = config["text_config"] if "text_config" in config else config
    quant = text["quantization_config"] if "quantization_config" in text \
        else config["quantization_config"]
    weights = quant["config_groups"]["group_0"]["weights"]
    if (quant["format"] != "pack-quantized" or weights["num_bits"] != 4
            or weights["group_size"] != 32 or weights["type"] != "int"
            or not weights["symmetric"] or weights["actorder"] is not None):
        raise ValueError("kimi-k2.6 is not INT4 g32 symmetric pack-quantized")
    return text


def plan(case_id, args):
    marker = f"H200_KERNEL::{FAMILY}::{case_id}"
    return {"family": FAMILY, "case_id": case_id, "marker": marker,
            "runner_relpath": RUNNER,
            "runner_args": [*args, "--nvtx_range_name", marker]}


def sweep(tag, fmt, experts, topk, hidden, inter, ep, swiglu_limit,
          situ=None):
    if experts % ep:
        raise ValueError(f"{tag}: experts {experts} not divisible by EP{ep}")
    plans = []
    for tok in TOKENS:
        case_id = (f"KC_WQMOE_{fmt.upper()}_{tag}_TOK{tok}_E{experts}_EP{ep}"
                   f"_TOPK{topk}_H{hidden}_INT{inter}")
        args = ["--weight_format", fmt, "--num_tokens", str(tok),
                "--num_experts", str(experts), "--top_k", str(topk),
                "--hidden_size", str(hidden), "--intermediate_size", str(inter),
                "--group_size", "32", "--ep_size", str(ep), "--ep_rank", "0"]
        if swiglu_limit is not None:
            args += ["--swiglu_limit", str(swiglu_limit)]
        if situ is not None:
            args += ["--activation", "situ",
                     "--activation-situ-beta",
                     str(situ["activation_situ_beta"]),
                     "--activation-situ-linear-beta",
                     str(situ["activation_situ_linear_beta"])]
        plans.append(plan(case_id, args))
    return plans


def k26_plans():
    c = load_k26()
    e, k, h, i = (c["n_routed_experts"], c["num_experts_per_tok"],
                  c["hidden_size"], c["moe_intermediate_size"])
    return (sweep("KIMIK26TP8", "int4_g32", e, k, h, i // 8, 1, None)
            + sweep("KIMIK26EP8", "int4_g32", e, k, h, i, 8, None))


def mxfp4_plans(model, tag, layouts, situ=None):
    c = MODEL_CONFIGS[model]
    plans = []
    for kind, degree in layouts:
        inter = (c["moe_intermediate_size"] // degree if kind == "TP"
                 else c["moe_intermediate_size"])
        ep = degree if kind == "EP" else 1
        plans += sweep(f"{tag}{kind}{degree}", "mxfp4", c["n_routed_experts"],
                       c["num_experts_per_tok"], c["hidden_size"], inter, ep,
                       c["swiglu_limit"], situ)
    return plans


def write(name, plans, **extra):
    ids = [p["case_id"] for p in plans]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name}: duplicate case ids")
    if len(plans) > 500:
        raise ValueError(f"{name}: {len(plans)} cases exceed 500")
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps({"schema_version": 1, "hardware_id": "nvidia_h200",
                                "campaign": name, "case_count": len(plans),
                                **extra, "plans": plans}, indent=1),
                    encoding="utf-8")
    print(path.name, len(plans))


def main():
    write("kc_wq_moe_int4_kimi_k26", k26_plans(),
          shape_origin={"KIMIK26": "configs/model/kimi-k2.6.json"})
    # V4-Flash fits 4 x H200 (TP4) and 8 x H200 (TP8/EP8); V4-Pro needs 8.
    write("kc_wq_moe_mxfp4_deepseek_v4",
          mxfp4_plans("deepseek-v4-flash", "DSV4FLASH",
                      (("TP", 4), ("TP", 8), ("EP", 8)))
          + mxfp4_plans("deepseek-v4-pro", "DSV4PRO", (("TP", 8), ("EP", 8))),
          shape_origin={m: MODEL_CONFIGS[m] for m in
                        ("deepseek-v4-flash", "deepseek-v4-pro")})
    # Kimi-K3 MXFP4 experts (~1.4 TB) exceed one 8 x H200 node: per-rank
    # expert shapes of TP8 (with PP2 across two nodes) and EP16.
    write("kc_wq_moe_mxfp4_kimi_k3",
          mxfp4_plans("kimi-k3", "KIMIK3", (("TP", 8), ("EP", 16))),
          shape_origin={"kimi-k3": MODEL_CONFIGS["kimi-k3"]})
    # vLLM 0.30 only: the same K3 shapes with K3's own SiTU activation.
    write("kc_wq_moe_mxfp4_kimi_k3_situ__v030",
          mxfp4_plans("kimi-k3", "KIMIK3SITU", (("TP", 8), ("EP", 16)),
                      situ=K3_SITU),
          shape_origin={"kimi-k3": {**MODEL_CONFIGS["kimi-k3"], **K3_SITU}},
          vllm_stack="vllm030")


if __name__ == "__main__":
    main()
