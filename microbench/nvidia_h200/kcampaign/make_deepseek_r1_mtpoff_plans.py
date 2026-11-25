"""kcampaign plans for the DeepSeek-R1 H200 MTP-off calibration (TP8 per-rank shapes).

Plan format: the one consumed by
scripts/kernel_dataset/nvidia_h200/nsys_campaign_runner.py.  Token counts
M cover MTP-off decode (M = C) and MTP-on verify (M = 3C, SGLang EAGLE
3 draft tokens) for C in {1, 2, 4, 8, 16, 32, 64}, plus the vLLM dispatch
boundaries in between (router tier switch after 16, FP8 linear swapAB
switch at 32, MoE naive/aligned switch after 8).  A few DSV3-L32 TP4
shapes (32 heads, MoE intermediate 512) are included to validate the
runner against the real TP4 server trace.  Evaluation input only.
Usage: python make_deepseek_r1_mtpoff_plans.py  -> plans/dsr1_mtpoff_{ops,fa3}.json, plans/dsr1_flashinfer_mla.json
(The dsr1_* plan names are stable campaign identifiers referenced by the
deployed remote snapshots and the runbook result tables.)
"""

import json
from pathlib import Path

RUNNERS = "scripts/kernel_dataset/nvidia_h200"
TOKENS = (1, 2, 3, 4, 6, 8, 9, 12, 16, 17, 24, 32, 48, 64, 96, 128, 192)
TP4_CHECK = (1, 4, 16, 64)
DENSE_K = (7168, 1536, 2048, 2304)   # q_a/kv_a, q_b, o_proj, dense down (TP8)


def plan(family, case_id, runner, args):
    marker = f"H200_KERNEL::{family}::{case_id}"
    return {"family": family, "case_id": case_id, "marker": marker,
            "runner_relpath": f"{RUNNERS}/{runner}",
            "runner_args": [*args, "--nvtx_range_name", marker]}


def op(case_id, *args):
    return plan("dsr1_decode_op", case_id, "kc_deepseek_r1_decode_runner.py", [str(a) for a in args])


def ops_plans():
    plans = []
    for m in TOKENS:
        for heads in (16,) + ((32,) if m in TP4_CHECK else ()):
            plans.append(op(f"DSR1_BMM_UK_H{heads}_M{m}", "--op", "mla_bmm_uk", "--heads", heads, "--tokens", m))
            plans.append(op(f"DSR1_BMM_UV_H{heads}_M{m}", "--op", "mla_bmm_uv", "--heads", heads, "--tokens", m))
        plans.append(op(f"DSR1_ROUTER_M{m}_H7168_E256", "--op", "router", "--tokens", m,
                        "--hidden", 7168, "--experts", 256))
        for inter in (256,) + ((512,) if m in TP4_CHECK else ()):
            plans.append(op(f"DSR1_MOE_FP8_M{m}_E256_K8_H7168_I{inter}", "--op", "fused_moe_fp8",
                            "--tokens", m, "--experts", 256, "--top-k", 8, "--hidden", 7168,
                            "--intermediate", inter))
        plans.append(op(f"DSR1_MOESUM_M{m}_K8_H7168", "--op", "moe_sum", "--tokens", m,
                        "--top-k", 8, "--hidden", 7168))
        for k in DENSE_K:
            plans.append(op(f"DSR1_ACTQ_ROW_M{m}_K{k}", "--op", "act_quant", "--tokens", m, "--k", k))
            if m >= 32:
                plans.append(op(f"DSR1_ACTQ_COL_M{m}_K{k}", "--op", "act_quant", "--tokens", m,
                                "--k", k, "--column-major-scales", 1))
    return plans


def fa3_plans():
    plans = []
    cases = [(16, 1, b, kv) for b in (1, 2, 4, 8, 16, 32, 64) for kv in (6656, 7424, 8192, 8960)]
    cases += [(16, 3, b, kv) for b in (1, 2, 4, 8, 16, 32, 64) for kv in (7424, 8960)]
    cases += [(32, 1, b, 7424) for b in TP4_CHECK]
    for heads, q, batch, kv in cases:
        plans.append(plan(
            "flash_attn", f"DSR1_MLA_DECODE_HQ{heads}_Q{q}_B{batch}_S{kv}", "flash_attn_runner.py",
            ["--batch", str(batch), "--seq_len_cache", str(kv), "--num_heads_q", str(heads),
             "--num_heads_kv", "1", "--head_dim", "512", "--dtype", "bf16", "--kv_layout", "mla",
             "--kv_dist", "uniform", "--seq_len_q", str(q),
             "--q-lens-json", json.dumps([q] * batch), "--kv-lens-json", json.dumps([kv] * batch),
             "--causal", "1", "--num_splits", "4"]))  # as dsr1_decode_phase1: the MLA branch calls FA3 with num_splits=0 (auto); 4 only satisfies the positive-dimension check
    return plans


def flashinfer_mla_plans():
    """SGLang's MLA decode backend (FlashInfer, page size 1) at TP8 per-rank shapes."""
    cases = [(16, 1, b) for b in (1, 2, 4, 8, 16, 32, 64)]
    cases += [(16, 3, b) for b in (1, 4, 16, 64)]
    cases += [(32, 1, b) for b in TP4_CHECK]
    return [op(f"DSR1_FI_MLA_HQ{h}_Q{q}_B{b}_S7424", "--op", "flashinfer_mla_decode",
               "--tokens", b, "--heads", h, "--q-len", q, "--kv-len", 7424, "--page-size", 1)
            for h, q, b in cases]


def main():
    out_dir = Path(__file__).with_name("plans")
    out_dir.mkdir(exist_ok=True)
    for name, plans in (("dsr1_mtpoff_ops", ops_plans()), ("dsr1_mtpoff_fa3", fa3_plans()),
                        ("dsr1_flashinfer_mla", flashinfer_mla_plans())):
        ids = [p["case_id"] for p in plans]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate case ids in {name}")
        out = out_dir / f"{name}.json"
        out.write_text(json.dumps({"schema_version": 1, "hardware_id": "nvidia_h200",
                                   "campaign": name, "case_count": len(plans), "plans": plans},
                                  indent=1) + "\n", encoding="utf-8")
        print(out, len(plans))


if __name__ == "__main__":
    main()
