"""Plans for the DeepSeek-R1 MTP-off calibration sweep on H200 (phase 1).

Only shapes the existing H200 dataset runners can already execute: the
router GEMM, the MLA-path RMSNorms and the FA3 MLA decode, at DeepSeek-R1
TP8 per-rank shapes (16 heads per rank).  Token counts cover MTP-off decode
(C = 1..64 tokens) and MTP-on verify (3*C up to 192).  The plan format is
the one consumed by scripts/kernel_dataset/nvidia_h200/nsys_campaign_runner.py.
"""

import json
from pathlib import Path

RUNNERS = "scripts/kernel_dataset/nvidia_h200"
TOKENS = sorted(set(range(1, 65)) | set(range(3, 193, 3)))
MLA_KV = (6656, 7424, 8192, 8960)   # 8K/1K decode context range, block 16


def plan(family, case_id, runner, args):
    marker = f"H200_KERNEL::{family}::{case_id}"
    return {"family": family, "case_id": case_id, "marker": marker,
            "runner_relpath": f"{RUNNERS}/{runner}",
            "runner_args": [*args, "--nvtx_range_name", marker]}


def vector_args(op, tok, hidden):
    # The unused dimensions carry the values of the existing vector plans.
    return ["--op", op, "--tok", str(tok), "--dtype", "bf16", "--batch", "1",
            "--op2", "add", "--hidden", str(hidden), "--num_heads", "32",
            "--head_dim", "128", "--intermediate", "6144",
            "--num_kv_heads", "4"]


def main():
    plans = []
    for tok in TOKENS:
        plans.append(plan("gemm", f"DSR1_ROUTER_GEMM_M{tok}_K7168_N256",
                          "gemm_runner.py",
                          ["--bs", "1", "--sl", str(tok), "--in_feat", "7168",
                           "--out_feat", "256", "--dtype", "bf16"]))
        for op, hidden in (("rmsnorm", 7168), ("rmsnorm", 1536),
                           ("rmsnorm", 512), ("add_rmsnorm", 7168)):
            plans.append(plan("vector",
                              f"DSR1_{op.upper()}_T{tok}_H{hidden}",
                              "vector_runner.py", vector_args(op, tok, hidden)))
    mla = [(16, b, kv) for b in range(1, 65) for kv in MLA_KV]
    mla += [(32, b, kv) for b in (1, 4, 16, 64) for kv in MLA_KV]
    for heads, batch, kv in mla:
        plans.append(plan(
            "flash_attn", f"DSR1_MLA_DECODE_HQ{heads}_B{batch}_S{kv}",
            "flash_attn_runner.py",
            ["--batch", str(batch), "--seq_len_cache", str(kv),
             "--num_heads_q", str(heads), "--num_heads_kv", "1",
             "--head_dim", "512", "--dtype", "bf16", "--kv_layout", "mla",
             "--kv_dist", "uniform", "--seq_len_q", "1",
             "--q-lens-json", json.dumps([1] * batch),
             "--kv-lens-json", json.dumps([kv] * batch),
             "--causal", "1", "--num_splits", "4"]))  # MLA branch always calls FA3 with num_splits=0 (auto); 4 only satisfies the positive-dimension check, as in WP9
    ids = [p["case_id"] for p in plans]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")
    out = Path(__file__).with_name("plans") / "dsr1_decode_phase1.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"schema_version": 1, "hardware_id": "nvidia_h200",
                               "campaign": "dsr1_decode_phase1",
                               "case_count": len(plans), "plans": plans},
                              indent=1), encoding="utf-8")
    print(out, len(plans))


if __name__ == "__main__":
    main()
