"""Token 工厂 workload 生成器 + 96% 命中率验证。

核心事实（已实测验证）：
    Uni3Sim 的 SyntheticRequestConfig 原生支持 prefix_hit_fraction，
    实现为  prefix_hit_tokens = int(input_length * prefix_hit_fraction)。
    因此「缓存预热」不需要构造共享前缀 workload —— 一个参数即可表达。

用法：
    py -3.11 gen_tf_workload.py            # 打印验证矩阵
    py -3.11 gen_tf_workload.py --write    # 生成 manifest 到 uni3sim 配置目录

口径纪律：
    - prefix_hit_fraction 表达的是「请求流前缀重合度」，不是「缓存命中率」本身。
      真实引擎还可能因 KV 淘汰、块对齐而略低，本参数是上界。
    - 本脚本只生成 uni3sim 的 workload/experiment 文件，不参与仿真结果。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types

UNI3SIM = r"E:\python\trae\pac\optimization\uni3sim"
REGEN = os.path.join(UNI3SIM, "src", "serving", "request_gen")
EXPDIR = os.path.join(UNI3SIM, "configs", "evaluation", "experiments")

# ---- TF-DESIGN-03 §1.1 业务输入 ----
T_DAY = 155e12
AVG_P = 77000
AVG_O = 512
HIT = 0.96
PEAK_MULT = 2.5


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _synthetic_module():
    """隔离加载 synthetic_requests，绕开 src.serving.__init__ 的 lru_sim 依赖。"""
    if "rg.request_stream" not in sys.modules:
        pkg = types.ModuleType("rg")
        pkg.__path__ = [REGEN]
        sys.modules["rg"] = pkg
        _load("rg.request_stream", os.path.join(REGEN, "request_stream.py"))
        _load("rg.synthetic_requests", os.path.join(REGEN, "synthetic_requests.py"))
    return sys.modules["rg.synthetic_requests"]


def derive_qps(peak: bool = False) -> float:
    r_day = T_DAY / (AVG_P + AVG_O)
    qps = r_day / 86400.0
    return qps * PEAK_MULT if peak else qps


def verify(hit_rates=(0.90, 0.94, 0.96, 0.98), n=1000):
    sr = _synthetic_module()
    rows = []
    for h in hit_rates:
        cfg = sr.SyntheticRequestConfig(
            num_requests=n, request_rate=5000.0,
            input_lengths=(AVG_P,), output_lengths=(AVG_O,),
            prefix_hit_fraction=h, arrival_distribution="poisson", seed=42)
        reqs = list(sr.generate_synthetic_requests(cfg).requests)
        total = sum(r.input_tokens for r in reqs)
        hit = sum(r.prefix_hit_tokens for r in reqs)
        rows.append((h, n, total, hit, hit / total))
    return rows


def build_cell(cell_id, *, hit=HIT, peak=False, deployment="pd_disagg",
               hw="configs/hardware/nvidia_h20.yaml", d_hw=None,
               p_inst=1, d_inst=1, link="nvlink", max_model_len=131072,
               note=""):
    qps = derive_qps(peak)
    sim = {
        "model": "configs/model/deepseek-v3.json",
        "hardware-config": hw,
        "deployment": deployment,
        "backend": "vllm",
        "prediction-method": "roofline",
        "launch-policy": "analytical_launch",
        "attention-kv-layout": "paged_varlen",
        "attention-kv-distribution": "uniform",
        "scheduler-pipeline-depth": 1,
        "prefill-tp-size": 8,
        "max-model-len": max_model_len,
        "max-batched-tokens": 32768,
        "max-requests": 256,
        "kv-block-size-tokens": 16,
        "vllm-scheduler-version": "0.11.0",
        "symbolic-forward-source": "cuda_eager_inplace",
        "target-sampling-source": "vllm_0_11_plain_greedy_bf16",
        "collective-backend": "analytical_roofline_v1",
        "execution-mode": "reference",
        "graph-mode": "NONE",
        "workload": "configs/evaluation/experiments/nvidia_h20_pd_tf03_workload.json",
        "request-rate": int(qps),
        "arrival-process": "poisson",
    }
    if deployment == "pd_disagg":
        sim["decode-tp-size"] = 8
        sim["num-prefill-instances"] = p_inst
        sim["num-decode-instances"] = d_inst
        sim["decode-hardware-config"] = d_hw or hw
        sim["transfer-algorithm"] = "send_recv"
        sim["transfer-link"] = link
        if link == "infiniband":
            sim["kv-transfer-backend"] = "analytical_roofline_v1"
    cell = {"id": cell_id, "simulation": sim}
    if note:
        cell["note"] = note
    return cell


def build_hit_matrix():
    """固定 QPS，只扫命中率 —— 验证 TF-03「每降1% +3.3万卡」。"""
    cells = []
    for h in (0.90, 0.94, 0.96, 0.98):
        cells.append(build_cell(
            f"hit{int(h*100)}_avg",
            hit=h, peak=False,
            note=f"命中率 {h:.0%}，QPS={int(derive_qps(False))}。用于验证命中率敏感性。"))
    return {"version": 9, "cells": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    print("=== 命中率验证（prefix_hit_fraction 实测）===")
    print("%-8s %-6s %-14s %-13s %s" % ("目标", "请求数", "输入token", "命中token", "实际"))
    for h, n, tot, hit, actual in verify():
        flag = "OK" if abs(actual - h) < 1e-9 else "FAIL"
        print("%-8.2f %-6d %-14d %-13d %.4f %s" % (h, n, tot, hit, actual, flag))

    print("\n=== 派生 QPS ===")
    print("  均值 QPS: %d   峰值 QPS: %d" % (int(derive_qps(False)), int(derive_qps(True))))

    matrix = build_hit_matrix()
    print("\n=== 命中率矩阵 manifest: %d cells ===" % len(matrix["cells"]))
    for c in matrix["cells"]:
        print("  %-16s hit=%.2f  QPS=%s" % (
            c["id"], c["simulation"]["prefix-hit-fraction"], c["simulation"]["request-rate"]))

    if args.write:
        out = os.path.join(EXPDIR, "tf_hit_matrix.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(matrix, f, ensure_ascii=False, indent=2)
        print("\n已写入:", out)


if __name__ == "__main__":
    main()
