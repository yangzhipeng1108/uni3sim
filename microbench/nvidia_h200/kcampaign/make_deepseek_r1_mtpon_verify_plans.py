"""Plan for the DeepSeek-R1 MTP-on verify FA3 MLA decode on H200 (vLLM 0.30).

The 14 ``DSR1_MLA_DECODE_HQ16_Q3_*`` cases of ``dsr1_mtpoff_fa3`` (generated
by ``make_deepseek_r1_mtpoff_plans.py`` on ``main``, commit a7e5fcec1): TP8
per-rank shapes (16 heads, one latent KV head, kv_lora_rank 512), MTP verify
of 1 + 2 draft tokens (Q_LEN=3 per request) at B in 1..64 and KV 7424 / 8960.
Case IDs, markers and runner arguments are identical to that plan; only the
plan name differs, so these cases run with the multi-token MLA branch of
``flash_attn_runner.py`` without touching the running plans.  ``--num_splits
4`` is kept as in every DSR1 MLA case: the MLA branch never forwards it (FA3
runs ``num_splits=0`` like vLLM's ``FlashAttnMLAImpl``).
Usage: python make_deepseek_r1_mtpon_verify_plans.py -> plans/dsr1_mtpon_verify_mla__v030.json
"""

import json
from pathlib import Path

RUNNERS = "scripts/kernel_dataset/nvidia_h200"
NAME = "dsr1_mtpon_verify_mla__v030"
HEADS = 16
QUERY_LEN = 3
BATCHES = (1, 2, 4, 8, 16, 32, 64)
KV_LENS = (7424, 8960)


def plan(family, case_id, runner, args):
    marker = f"H200_KERNEL::{family}::{case_id}"
    return {"family": family, "case_id": case_id, "marker": marker,
            "runner_relpath": f"{RUNNERS}/{runner}",
            "runner_args": [*args, "--nvtx_range_name", marker]}


def verify_mla_plans():
    plans = []
    for batch in BATCHES:
        for kv in KV_LENS:
            plans.append(plan(
                "flash_attn",
                f"DSR1_MLA_DECODE_HQ{HEADS}_Q{QUERY_LEN}_B{batch}_S{kv}",
                "flash_attn_runner.py",
                ["--batch", str(batch), "--seq_len_cache", str(kv),
                 "--num_heads_q", str(HEADS), "--num_heads_kv", "1",
                 "--head_dim", "512", "--dtype", "bf16",
                 "--kv_layout", "mla", "--kv_dist", "uniform",
                 "--seq_len_q", str(QUERY_LEN),
                 "--q-lens-json", json.dumps([QUERY_LEN] * batch),
                 "--kv-lens-json", json.dumps([kv] * batch),
                 "--causal", "1", "--num_splits", "4"]))
    return plans


def main():
    plans = verify_mla_plans()
    ids = [p["case_id"] for p in plans]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")
    out = Path(__file__).with_name("plans") / f"{NAME}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"schema_version": 1, "hardware_id": "nvidia_h200",
                               "campaign": NAME, "case_count": len(plans),
                               "plans": plans}, indent=1) + "\n",
                   encoding="utf-8")
    print(out, len(plans))


if __name__ == "__main__":
    main()
