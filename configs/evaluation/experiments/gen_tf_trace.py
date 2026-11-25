"""Token 工厂 production trace 生成器（96% 命中率）。

为什么是 trace 而不是 workload：
    experiment_plan.py:1131-1140 的 workload artifact 允许字段是严格匹配的
    {version, workload, request_ids, prompts, prompt_lengths, sampling,
     token_controls, num_requests, requested_num_requests, seed, hash}
    —— 里面没有 prefix_hit_tokens。
    experiment_plan.py:215 明确把 synthetic workload 的命中标记为
    "synthetic_length_only_zero_assumption"，即强制 0。

    唯一能精确表达命中率的路径是 production trace：
        --production-trace <file>
        --trace-prefix-hit-tokens-field prefix_hit_tokens
    这样每行请求自带命中 token 数，命中率可精确控制。

用法：
    py -3.11 gen_tf_trace.py                    # 生成 96% 命中 trace + 报告
    py -3.11 gen_tf_trace.py --hit 0.94         # 指定命中率
    py -3.11 gen_tf_trace.py --jsonl out.jsonl  # 指定输出

口径纪律：
    prefix_hit_tokens 表达「已缓存的提示前缀长度」，是声明值不是观测值。
    Uni3Sim preview 会把它标为 input_field_not_runtime_observation。
"""
from __future__ import annotations

import argparse
import json
import math
import random

# ---- TF-DESIGN-03 §1.1 业务输入 ----
T_DAY = 155e12
AVG_P = 77000
AVG_O = 512
PEAK_MULT = 2.5


def derive_qps(peak: bool = False) -> float:
    r_day = T_DAY / (AVG_P + AVG_O)
    qps = r_day / 86400.0
    return qps * PEAK_MULT if peak else qps


def generate_trace(path, *, hit=0.96, num_requests=200000, qps=None,
                   prompt_len=AVG_P, output_len=AVG_O, seed=0,
                   arrival="poisson"):
    """写 JSONL trace：每行一个请求，含 prefix_hit_tokens。"""
    if qps is None:
        qps = derive_qps(False)
    rng = random.Random(seed)
    now = 0.0
    written = 0
    total_in = total_hit = 0
    with open(path, "w", encoding="utf-8") as f:
        for i in range(num_requests):
            if i:
                if arrival == "poisson":
                    now += rng.expovariate(qps)
                else:
                    now += 1.0 / qps
            rec = {
                "request_id": f"req-{i:07d}",
                "arrival_time": round(now, 6),
                "input_tokens": prompt_len,
                "output_tokens": output_len,
                "prefix_hit_tokens": int(prompt_len * hit),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_in += prompt_len
            total_hit += rec["prefix_hit_tokens"]
            written += 1
    return {
        "path": path,
        "num_requests": written,
        "qps": qps,
        "span_s": now,
        "input_tokens_total": total_in,
        "hit_tokens_total": total_hit,
        "actual_hit_rate": total_hit / total_in,
        "peak_qps": derive_qps(True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hit", type=float, default=0.96)
    ap.add_argument("--num-requests", type=int, default=200000)
    ap.add_argument("--jsonl", default="tf_trace_96.jsonl")
    ap.add_argument("--prompt-len", type=int, default=AVG_P)
    ap.add_argument("--output-len", type=int, default=AVG_O)
    ap.add_argument("--arrival", choices=("poisson", "constant"), default="poisson")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    r = generate_trace(args.jsonl, hit=args.hit, num_requests=args.num_requests,
                       prompt_len=args.prompt_len, output_len=args.output_len,
                       seed=args.seed, arrival=args.arrival)

    print("=== trace 生成完成 ===")
    print("  文件            %s" % r["path"])
    print("  请求数          %d" % r["num_requests"])
    print("  QPS             %.0f  (峰值 %.0f)" % (r["qps"], r["peak_qps"]))
    print("  时间跨度        %.1f 秒" % r["span_s"])
    print("  输入 token 总   %d" % r["input_tokens_total"])
    print("  命中 token 总   %d" % r["hit_tokens_total"])
    print("  实际命中率      %.4f  (目标 %.2f) %s" % (
        r["actual_hit_rate"], args.hit,
        "OK" if abs(r["actual_hit_rate"] - args.hit) < 1e-6 else "FAIL"))
    print()
    print("=== 对应 CLI 参数 ===")
    print("  --production-trace %s" % r["path"])
    print("  --trace-format jsonl")
    print("  --trace-arrival-field arrival_time")
    print("  --trace-input-tokens-field input_tokens")
    print("  --trace-output-tokens-field output_tokens")
    print("  --trace-prefix-hit-tokens-field prefix_hit_tokens")
    print("  --trace-request-id-field request_id")
    print("  --prefix-cache-mode trace")
    print("  --enable-prefix-caching")


if __name__ == "__main__":
    main()
