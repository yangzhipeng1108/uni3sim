"""Build the HBM-PIM single-op dataset using the UniNDP simulator as groundtruth.

PIM hardware is unavailable for NCU/mcProfiler/msprof sampling. This builder uses UniNDP (HPCA25)
to simulate each GEMM shape and uses its cycle-accurate latency (ns) as
ground truth, producing an NCU-compatible JSON dataset for kernel evaluation.

Stages (shares the `aggregate` publication interface with other builders; UniNDP supplies latency directly):
  - aggregate: run UniNDP compile.py per shape, checkpoint in profiler_output, then
    publish canonical `kernel_dataset/samsung_hbm_pim/<kernel>.json.zst`.

GEMM uses a single UniNDP mm; EagerAttention, UnfusedMoE, and vector operations explicitly
decompose into supported mm/elewise/softmax/layernorm primitives, preserving their unfused upper-bound provenance.
If large-M prefill simulation times out or real A800 shape/tiling is missing, retain only candidates and skip logs;
do not publish an incomplete canonical dataset.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Pin each UniNDP subprocess to a single thread so N-way parallel generation uses
# exactly N cores (not N × numpy's internal threads), avoiding oversubscription.
_SIM_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", **os.environ}

from dataset_pipeline.kernel_dataset.config import (
    load_pipeline_config,
    load_workload_config,
    get_kernel_pipeline_settings,
    get_workload_config_path,
    get_kernel_dataset_path,
    get_profiler_output_dir,
    parse_kernel_list,
    REPO_ROOT,
    require_mapping,
    require_environment_directory,
    require_non_empty_list,
    require_positive_int,
    to_float,
)
from dataset_pipeline.kernel_dataset.dataset_writer import (
    DatasetCompletenessError,
    build_kernel_evaluation_timing,
    publish_complete_dataset,
)
from src.kernel.dataset_storage import (
    KERNEL_DATASET_SUFFIX,
    load_kernel_dataset_document,
)


# ─── NCU-compatible entry skeleton ───────────────────────────────────
# evaluation/kernel/profiler_observation.py::extract_ncu_metrics reads
# the following ~25 NCU leaf fields per kernel; any missing field raises KeyError. Header duration is retained only as
# raw UniNDP evidence; evaluation reads its sole scoring target from op_info.evaluation.timing.
# Other leaf values do not enter prediction (PIM uses analytical traffic), so use 0.0 / nominal placeholders.
def _ncu_sections_skeleton(duration_ns: float) -> dict:
    """Build a minimal NCU Sections block carrying the real groundtruth latency.

    All non-Duration leaves are placeholders — they satisfy extract_ncu_metrics'
    hard indexing but their values are never used as groundtruth or consumed
    by the forward predictor on a PIM hw spec
    (kernel_name_encodes_tile=false -> analytical traffic path).
    """
    if (isinstance(duration_ns, bool)
            or not isinstance(duration_ns, (int, float))):
        raise TypeError("ground-truth duration must be numeric")
    if not math.isfinite(duration_ns) or duration_ns <= 0:
        raise ValueError(
            f"ground-truth duration must be finite and positive: {duration_ns!r}")
    return {
        "GPU Speed Of Light Throughput": {
            "Header": {
                "Duration [nsecond]": float(duration_ns),
                "SM Frequency [cycle/second]": 800000000.0,   # PIM core 800MHz
                "DRAM Frequency [cycle/second]": 800000000.0,
                "Elapsed Cycles [cycle]": float(duration_ns) * 0.8,  # 800MHz
                "SM Active Cycles [cycle]": float(duration_ns) * 0.8,
            }
        },
        "Compute Workload Analysis": {
            "Pipe Utilization (% of active cycles)": {
                "Tensor (All) [%]": 0.0,
                "FMA [%]": 0.0,
                "ALU [%]": 0.0,
            },
            "Pipe Utilization (% of peak instructions executed)": {
                "XU [%]": 0.0,
            },
        },
        "Memory Workload Analysis Chart": {
            "Memory Chart": {
                "l1tex__m_xbar2l1tex_read_bytes.sum [byte]": 0.0,
                "l1tex__m_l1tex2xbar_write_bytes.sum [byte]": 0.0,
                "dram__bytes_read.sum [byte]": 0.0,
                "dram__bytes_write.sum [byte]": 0.0,
                "l1tex__t_sector_hit_rate.pct [%]": 0.0,
                "lts__t_sector_hit_rate.pct [%]": 0.0,
                "sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass.sum [byte]": 0.0,
            }
        },
        "Launch Statistics": {
            "Header": {
                "Block Size": 1,
                "Registers Per Thread [register/thread]": 32,
                "Dynamic Shared Memory Per Block [byte/block]": 0,
                "Static Shared Memory Per Block [byte/block]": 0,
                "Driver Shared Memory Per Block [byte/block]": 0,
                "Achieved Occupancy [%]": 100.0,
                "Theoretical Occupancy [%]": 100.0,
            }
        },
        "Occupancy": {
            "Header": {
                "Block Limit Registers [block]": 1,
            }
        },
        "Memory Workload Analysis Tables": {
            "Device Memory": {
                "dram__bytes_read.sum.pct_of_peak_sustained_elapsed [%]": 0.0,
                "dram__bytes_write.sum.pct_of_peak_sustained_elapsed [%]": 0.0,
            }
        },
    }


def _make_gemv_entry(B: int, M: int, K: int, N: int, dtype: str,
                     latency_ns: float) -> dict:
    """Build one NCU-compat GEMM dataset entry with real UniNDP latency."""
    return _make_entry(
        "GEMM", {"B": B, "M": M, "K": K, "N": N, "dtype": dtype},
        latency_ns, "pim_hbm_gemm")


# ─── UniNDP simulation bridge ─────────────────────────────────────────
def _run_unindp_op(unindp_repo: str, arch: str, topk: int, quicksearch: bool,
                   timeout_s: int, workspace_rel: str, name: str,
                   workload: str, sizes: list) -> float:
    """Simulate one UniNDP primitive op (mm/elewise/softmax/...); return latency ns.

    Output: {workspace}/{outdir}/csv/_{name}.csv, latency = field 3 (optimal_lat),
    already in ns (config/hbm-pim.yaml tick=1ns). NO unit conversion.

    Gotcha: compile.py:70 prepends `./` to the log path, so -WS MUST be relative
    to the UniNDP repo cwd (absolute paths break with `.//abs/...`).

    Raises TimeoutError (sim too slow) / RuntimeError (no valid partition for the
    shape, nonzero exit, or no csv) / AssertionError (non-positive latency).
    """
    outdir = f"out_{name}"
    cmd = [
        "python3", "compile.py",
        "-A", arch,
        "-W", workload,
        "-N", name,
        "-S", *[str(s) for s in sizes],
        "-O", outdir,
        "-WS", workspace_rel,
        "-K", str(topk),
    ]
    if quicksearch:
        cmd.append("-Q")

    csv_path = os.path.join(unindp_repo, workspace_rel, outdir, "csv", f"_{name}.csv")
    try:
        proc = subprocess.run(
            cmd, cwd=unindp_repo,
            capture_output=True, text=True, timeout=timeout_s, env=_SIM_ENV,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"UniNDP sim timed out >{timeout_s}s")

    if proc.returncode != 0:
        raise RuntimeError(
            f"UniNDP compile.py exited {proc.returncode}: {proc.stderr[-300:]}"
        )
    if not os.path.exists(csv_path):
        raise RuntimeError(f"UniNDP produced no csv at {csv_path}")

    # csv row: name,workload,optimal_lat,...  -> field index 2 (0-based) = optimal_lat
    with open(csv_path) as f:
        row = next(csv.reader(f))
    if len(row) <= 2:
        raise ValueError(f"UniNDP CSV row is missing optimal_lat: {row!r}")
    latency_ns = to_float(row[2])
    if latency_ns <= 0:
        raise ValueError(f"non-positive latency {latency_ns} for {name}")
    return latency_ns


def _run_unindp_gemm(unindp_repo, arch, topk, quicksearch, timeout_s,
                     workspace_rel, task_key, M_eff, K, N):
    """Simulate one GEMM. UniNDP `mm` -S=[M,K(reduce),N,B]; compile.py:78 folds B
    into M, so our (B,M,K,N) -> -S (B*M) K N 1."""
    return _run_unindp_op(unindp_repo, arch, topk, quicksearch, timeout_s,
                          workspace_rel, task_key, "mm", [M_eff, K, N, 1])


def _run_unindp_glue(run_fn, workload, sizes, name):
    """Run a required UniNDP glue stage without fabricating zero latency."""
    return run_fn(workload, sizes, name)


# ─── EagerAttn decomposition (unfused attention; NOT FlashAttention) ───────
# Groundtruth = SUM of UniNDP per-stage latencies for the decomposed attention:
#   qkt mm + (row softmax over KV) + pv mm.
# This is an UNFUSED "eager" upper bound: UniNDP has no fused FA kernel —
# softmax is a two-pass materializing reduce (writes exp/P back to bank/DRAM),
# mm ships S=QK^T to host between stages, no causal mask, sim() cold-starts per
# op. Named EAGERATTN (not FLASHATTN) on purpose.
#
# Geometry: the A800 FA attr (cloned for ALL 306 shapes — plain + serving
# variants paged/chunked/pagedvarlen/mla) carries authoritative `Q_LEN` (query
# count) and `KV_LEN` (KV cache length). The decomposition uses those, NOT `S`
# (S is the legacy config seq_len; for decode/paged shapes Q_LEN=1 ≪ KV_LEN, so
# using S as the query length would model the wrong attention). Plain square
# prefill attrs also carry Q_LEN/KV_LEN (= S), so the path is uniform. Variants
# differ only in KV memory layout (paged block table / chunked Q-rechunk) — the
# core matmul structure is standard attention; the decomposition ignores the
# paged-scatter memory penalty (an upper-bound approximation, see notes). MLA is
# latent-space attention (D=512, HKV=1) — qkt/sm/pv apply in latent dim, but the
# fused v up-projection (latent→hidden) is absorbed in the A800 MLA kernel and
# NOT in the naive decomposition → MLA is a known lower approximation (documented, not added,
# since `hidden` isn't carried by the MLA attr).
def _decomp_eager_attn_latency(attr, run_fn, name_stem):
    """Return (latency_ns, stages). attr: {B,Q_LEN?,KV_LEN?,S,HQ,HKV,D,...}."""
    B = require_positive_int(attr, "B", "EAGERATTN Attr")
    HQ = require_positive_int(attr, "HQ", "EAGERATTN Attr")
    D = require_positive_int(attr, "D", "EAGERATTN Attr")
    # Q_LEN = query count (decode=1, prefill=S); KV_LEN = KV cache length.
    Q = require_positive_int(attr, "Q_LEN", "EAGERATTN Attr")
    KV = require_positive_int(attr, "KV_LEN", "EAGERATTN Attr")
    # qkt: mm [M=Q, K=D, N=KV, B=HQ*B] -> (Q*HQ*B, D, KV, 1)
    l_qkt = run_fn("mm", [Q, D, KV, HQ * B], f"{name_stem}_qkt")
    # row softmax over KV, rows=Q*HQ*B -> -S [KV, Q*HQ*B, 1, 1]
    l_sm = _run_unindp_glue(run_fn, "softmax", [KV, Q * HQ * B, 1, 1],
                             f"{name_stem}_sm")
    # pv: mm [M=Q, K=KV, N=D, B=HQ*B] -> (Q*HQ*B, KV, D, 1)
    l_pv = run_fn("mm", [Q, KV, D, HQ * B], f"{name_stem}_pv")
    return l_qkt + l_sm + l_pv, {"qkt": l_qkt, "softmax": l_sm, "pv": l_pv}


# ─── MoE decomposition (unfused; real A800 expert routing) ─────────────────
# Groundtruth = SUM of UniNDP per-stage latencies:
#   router mm + router softmax + Σ_active_experts (w1 mm + swiglu + w2 mm) + reduce
# tok_per_expert (real A800-measured routing) determines per-expert token counts —
# NOT an even split. UniNDP has no expert/grouped-mm/topk-scatter (zero hits repo-
# wide), so this is E separate dense MVMs, an unfused upper bound (NOT a fused
# grouped-MoE kernel latency).
def _decomp_moe_latency(attr, run_fn, name_stem):
    """Return (latency_ns, stages). attr: {TOK,E,TOPK,H,INT,tok_per_expert}."""
    tokens = require_positive_int(attr, "TOK", "UNFUSEDMOE Attr")
    experts = require_positive_int(attr, "E", "UNFUSEDMOE Attr")
    topk = require_positive_int(attr, "TOPK", "UNFUSEDMOE Attr")
    H = require_positive_int(attr, "H", "UNFUSEDMOE Attr")
    INT = require_positive_int(attr, "INT", "UNFUSEDMOE Attr")
    raw_tpe = attr["tok_per_expert"]
    if not isinstance(raw_tpe, list):
        raise TypeError("UNFUSEDMOE Attr.tok_per_expert must be a list")
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in raw_tpe):
        raise TypeError(
            "UNFUSEDMOE Attr.tok_per_expert must contain integers")
    if any(value < 0 for value in raw_tpe):
        raise ValueError(
            "UNFUSEDMOE Attr.tok_per_expert cannot contain negative values")
    if len(raw_tpe) != experts:
        raise ValueError(
            "UNFUSEDMOE Attr.tok_per_expert length must equal E")
    if sum(raw_tpe) != tokens * topk:
        raise ValueError(
            "UNFUSEDMOE Attr.tok_per_expert sum must equal TOK * TOPK")
    tpe = list(raw_tpe)
    # router: mm [M=TOK, K=H, N=E] (TOK from attr for shape; routing in tpe)
    l_router = _run_unindp_glue(run_fn, "mm", [tokens, H, experts, 1],
                                 f"{name_stem}_router")
    l_rsm = _run_unindp_glue(run_fn, "softmax",
                             [experts, tokens, 1, 1],
                             f"{name_stem}_rsm")
    # Per-expert latency depends only on tok_e (w1/swiglu/w2 shapes are tok_e-driven,
    # not expert-index) — cache by distinct tok_e to minimize UniNDP calls.
    per_tok_e = {}
    for tok_e in sorted({t for t in tpe if t > 0}):
        # w1: mm [M=tok_e, K=H, N=2*INT]; w2: mm [M=tok_e, K=INT, N=H]
        l_w1 = run_fn("mm", [tok_e, H, 2 * INT, 1], f"{name_stem}_w1_t{tok_e}")
        l_sw = _run_unindp_glue(run_fn, "elewise", [1, tok_e * INT, 1, 1],
                                 f"{name_stem}_sw_t{tok_e}")
        l_w2 = run_fn("mm", [tok_e, INT, H, 1], f"{name_stem}_w2_t{tok_e}")
        per_tok_e[tok_e] = l_w1 + l_sw + l_w2
    # reduce: sum topk expert outputs -> elewise over Σ tok_e * H
    total_tok = sum(tpe)
    l_red = _run_unindp_glue(run_fn, "elewise", [1, total_tok * H, 1, 1],
                             f"{name_stem}_red")
    # Σ over experts of per-expert latency; per_tok_e cached by distinct tok_e.
    expert_sum = sum(per_tok_e[t] * tpe.count(t) for t in per_tok_e)
    total = l_router + l_rsm + expert_sum + l_red
    return total, {"router": l_router, "softmax": l_rsm,
                   "expert_sum": expert_sum, "reduce": l_red,
                   "active_experts": sum(1 for t in tpe if t > 0)}


# ─── Vector decomposition (LLM glue ops: norm/rotary/activation/ew/kvcache) ────
# UniNDP has NO native rmsnorm/rope/silu/kvcache (compile.py -W accepts only
# mm/elewise/softmax/layernorm/batchnorm — base.py:271 else NotImplementedError).
# So every vector op is decomposed into the runnable UniNDP primitives, and the
# groundtruth latency = SUM of per-primitive UniNDP sim latencies (real per-hw
# simulation, same spirit as EagerAttn/MoE). Shapes are CLONED from the A800
# measured vector attr (344 shapes) so the dataset stays horizontally comparable
# to 910B/A800; only the Duration is replaced by the UniNDP decomposed sum.
#
#   elewise -S [C,H,W,B] -> mm_size (1, C*H*W*B, 1, 1): streams the full element
#            count through PIM compute (K = element_count, no real reduce).
#   layernorm -S [C(norm),H,W,B] -> mm_size (1, C*H*W, B, 1): norms over C*H*W
#            per batch B. rmsnorm has no UniNDP primitive; layernorm is the
#            closest (norm = reduce-then-scale) and is a defensible UPPER BOUND
#            (layernorm does mean+var, rmsnorm does mean(x²) only — ~1 fewer
#            pass, so the sim marginally over-estimates rmsnorm).
_VECTOR_OP_TYPE = {
    "rmsnorm": "NORM", "add_rmsnorm": "NORM", "rope": "ROTARY",
    "silu": "ACTIVATION", "elementwise": "ELEMENTWISE", "kvcache": "KV_CACHE",
}


def _decomp_vector_latency(attr, run_fn, name_stem):
    """Return (latency_ns, stages). Decompose a vector op into UniNDP primitives.

    attr: the cloned A800 vector attr {op, tok, hidden, intermediate, num_heads,
    head_dim, num_kv_heads, batch, ...}. run_fn raises on UniNDP failure (→ the
    case is skipped by the worker, mirroring GEMM/EagerAttn — never fabricated).
    """
    op = attr["op"]
    tok = require_positive_int(attr, "tok", "vector Attr")
    H = require_positive_int(attr, "hidden", "vector Attr")
    INT = require_positive_int(attr, "intermediate", "vector Attr")
    NH = require_positive_int(attr, "num_heads", "vector Attr")
    D = require_positive_int(attr, "head_dim", "vector Attr")
    KVH = require_positive_int(attr, "num_kv_heads", "vector Attr")
    B = require_positive_int(attr, "batch", "vector Attr")

    def elwise(n, suf):
        # elewise -S [C=elem, 1, 1, 1] -> K = n (full element stream)
        return run_fn("elewise", [n, 1, 1, 1], f"{name_stem}_{suf}")

    def lnorm(dim, suf):
        # layernorm -S [dim, 1, 1, tok] -> norm over dim per token (batch=tok)
        return run_fn("layernorm", [dim, 1, 1, tok], f"{name_stem}_{suf}")

    if op == "rmsnorm":
        l = lnorm(H, "norm")
        return l, {"layernorm": l}
    elif op == "add_rmsnorm":
        # residual add (elewise over tok*hidden) + norm (layernorm). PIM has no
        # unified buffer, so the residual add is a real elewise pass (unlike the
        # GPU forward model which folds it in-place — a genuine per-hw difference).
        l_add = elwise(tok * H, "add")
        l_nm = lnorm(H, "norm")
        return l_add + l_nm, {"residual_add": l_add, "layernorm": l_nm}
    elif op == "rope":
        # The serving operator rotates both query heads and KV heads in one
        # invocation.  Preserve the same Q+K workload contract as the GPU/NPU
        # runners instead of simulating only the query tensor.
        l = elwise(tok * (NH + KVH) * D, "rope")
        return l, {"elewise": l}
    elif op == "silu":
        l = elwise(tok * INT, "silu")
        return l, {"elewise": l}
    elif op == "elementwise":
        l = elwise(tok * H, "ew")
        return l, {"elewise": l}
    elif op == "kvcache":
        # paged K+V write for B sequences: 2 (K,V) × B × tok × KVH × D elements
        l = elwise(2 * B * tok * KVH * D, "kvc")
        return l, {"elewise": l}
    else:
        raise ValueError(f"unsupported vector op: {op}")


# ─── entry makers (NCU-compat; Duration carries the UniNDP decomposed latency) ───
def _make_entry(op_type, attr, latency_ns, kernel_name, grid_size="1,1,1"):
    kernels = [{
            "Metadata": {"Kernel Name": kernel_name, "Grid Size": grid_size,
                         "Block Size": "1,1,1", "Device": "HBM-PIM", "CC": "pim"},
            "Sections": _ncu_sections_skeleton(latency_ns),
        }]
    return {
        "op_info": {
            "Type": op_type,
            "Attr": attr,
            "evaluation": build_kernel_evaluation_timing(
                "samsung_hbm_pim", kernels, [float(latency_ns)]),
        },
        "kernels": kernels,
    }


def _make_gemv_entry(B, M, K, N, dtype, latency_ns):
    return _make_entry(
        "GEMM", {"B": B, "M": M, "K": K, "N": N, "dtype": dtype},
        latency_ns, "pim_hbm_gemm")


# ─── case matrix ───────────────────────────────────────────
def _enumerate_gemv_cases(kernel_cfg: dict):
    """Yield (task_key, gpu_key, (B,M,K,N,dtype)) for the GEMM grid
    product. gpu_key=None (GEMM shapes are hw-agnostic, no A800 routing to clone).

    Mirrors the NVIDIA A800 builder naming: GEMM_B{B}_M{M}_K{K}_N{N}.
    """
    workloads = require_non_empty_list(
        kernel_cfg, "workload_configs", "gemm config")
    layers = require_non_empty_list(kernel_cfg, "layers", "gemm config")
    for wc in workloads:
        require_mapping(wc, "each gemm workload")
        B = require_positive_int(wc, "batch", "gemm workload")
        M = require_positive_int(wc, "seq_len", "gemm workload")
        for layer in layers:
            require_mapping(layer, "each gemm layer")
            K = require_positive_int(layer, "k", "gemm layer")
            N = require_positive_int(layer, "n", "gemm layer")
            dtype = layer["dtype"]
            if not isinstance(dtype, str) or not dtype:
                raise TypeError("gemm layer.dtype must be a non-empty string")
            task_key = f"GEMM_B{B}_M{M}_K{K}_N{N}"
            yield task_key, None, (B, M, K, N, dtype)


def _enumerate_moe_cases(kernel_cfg: dict):
    """Yield (key, gpu_key, (TOK,E,TOPK,H,INT)) for the MoE case matrix.
    The published case is UNFUSEDMOE, while the routing/tiling source remains
    the measured NVIDIA A800 FUSEDMOE observation for the same shape."""
    workloads = require_non_empty_list(
        kernel_cfg, "workload_configs", "fused_moe source config")
    layers = require_non_empty_list(
        kernel_cfg, "layers", "fused_moe source config")
    for wc in workloads:
        require_mapping(wc, "each fused_moe source workload")
        TOK = require_positive_int(
            wc, "num_tokens", "fused_moe source workload")
        for layer in layers:
            require_mapping(layer, "each fused_moe source layer")
            E = require_positive_int(
                layer, "num_experts", "fused_moe source layer")
            TOPK = require_positive_int(
                layer, "top_k", "fused_moe source layer")
            H = require_positive_int(
                layer, "hidden_size", "fused_moe source layer")
            INT = require_positive_int(
                layer, "intermediate_size", "fused_moe source layer")
            suffix = f"TOK{TOK}_E{E}_TOPK{TOPK}_H{H}_INT{INT}"
            task_key = f"UNFUSEDMOE_{suffix}"
            source_key = f"FUSEDMOE_{suffix}"
            yield task_key, source_key, (TOK, E, TOPK, H, INT)


def _eager_attn_case_id(source_key: str) -> str:
    """Map one measured FLASHATTN source ID to its EAGERATTN output ID."""
    prefix = "FLASHATTN_"
    if not isinstance(source_key, str) or not source_key.startswith(prefix):
        raise ValueError(
            "eager-attention source case ID must start with "
            f"{prefix!r}, got {source_key!r}")
    suffix = source_key[len(prefix):]
    if not suffix:
        raise ValueError("eager-attention source case ID has an empty suffix")
    return f"EAGERATTN_{suffix}"


def _load_a800_attr_map(gpu_kernel: str) -> dict:
    """Load A800 dataset {key: op_attr} for a kernel (e.g. 'flash_attn' ->
    flash_attn.json.zst). PIM EagerAttn/MoE CLONE the A800-measured op_attr — real
    FA shapes and, for MoE, real expert routing (tok_per_expert) + fused-kernel
    tiling (moe_tiling) — so the GPU predictors run correctly. Only the Duration
    (groundtruth latency) is replaced by the UniNDP decomposed sum. This keeps
    the dataset self-consistent: measured routing/tiling + UniNDP-modeled latency.
    Missing source data is an error because publishing a partially cloned
    dataset would violate the workload contract.
    """
    path = os.path.join(
        REPO_ROOT, "kernel_dataset", "nvidia_a800",
        f"{gpu_kernel}{KERNEL_DATASET_SUFFIX}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"required NVIDIA A800 routing dataset does not exist: {path}")
    data = load_kernel_dataset_document(path)
    return {k: v["op_info"]["Attr"] for k, v in data.items()}


# Map PIM output operator -> (shape workload, A800 source dataset, enumerator).
# vector/eager_attn: enumerate_fn=None — their cases are cloned verbatim from the
# A800 attr map, NOT enumerated from a workload config. eager_attn uses the
# flash_attn shape/source contract because
# the A800 flash_attn keys carry serving-kernel variant prefixes (PAGED/CHUNKED/
# PAGEDVARLEN/MLA) that the shared flash_attn config's plain keys don't match
# (only 100/306 overlap); cloning the A800 attr directly covers all 306 shapes.
_KERNEL_DISPATCH = {
    "gemm": ("gemm", "gemm", _enumerate_gemv_cases),
    "eager_attn": ("flash_attn", "flash_attn", None),
    # Output operator, workload-shape source, measured attribute source.
    "unfused_moe": ("fused_moe", "fused_moe", _enumerate_moe_cases),
    "vector": ("vector", "vector", None),
}


# ─── stages ────────────────────────────────────────────────
def stage_aggregate(cfg, kernel: str, max_tasks: int, start_idx: int, end_idx: int,
                    jobs: int, timeout_s: int):
    """Simulate every case via UniNDP and publish a complete kernel dataset.

    GEMM: one UniNDP mm per case, latency = real single-op sim measurement.
    EAGERATTN / UNFUSEDMOE: decompose into UniNDP primitives
    (mm/softmax/elewise),
    latency = SUM of per-stage sim latencies (a decomposed UNFUSED upper bound,
    NOT a fused-kernel measurement). EagerAttn/MoE clone the A800-measured
    op_attr (real expert routing + tiling for MoE) for predictor self-consistency.

    Runs `jobs` cases concurrently (each UniNDP subprocess is single-threaded, so
    jobs = core budget). Resume-safe: cached entries skipped; each completion
    persisted under a lock so an interrupt never loses finished work.
    """
    if kernel not in _KERNEL_DISPATCH:
        raise ValueError(f"unsupported kernel: {kernel}")
    config_kernel_name, a800_json, enumerate_fn = _KERNEL_DISPATCH[kernel]

    pipeline_settings = get_kernel_pipeline_settings(cfg)
    unindp_repo = require_environment_directory("UNINDP_ROOT")
    arch = pipeline_settings["unindp_arch"]
    topk = require_positive_int(
        pipeline_settings, "unindp_topk", "kernel pipeline settings")
    quicksearch = pipeline_settings["unindp_quicksearch"]
    if not isinstance(quicksearch, bool):
        raise TypeError("kernel pipeline unindp_quicksearch must be bool")
    workspace_path = os.path.join(
        get_profiler_output_dir(cfg),
        pipeline_settings["unindp_workspace_subdir"],
    )
    # UniNDP requires -WS relative to its own cwd. Compute that relative path
    # from the canonical Midlayer profiler workspace instead of creating a
    # second output root inside the external checkout.
    workspace_rel = os.path.relpath(workspace_path, unindp_repo)

    kernel_cfg_path = get_workload_config_path(config_kernel_name)
    kernel_cfg = load_workload_config(
        kernel_cfg_path, config_kernel_name)

    # UniNDP run closure shared by all decomposition stages.
    run_fn = lambda workload, sizes, name: _run_unindp_op(
        unindp_repo, arch, topk, quicksearch, timeout_s, workspace_rel, name,
        workload, sizes)

    # NVIDIA A800 op_attr map (real measured routing/tiling) for
    # EagerAttn/MoE/Vector. GEMM is generated from its own workload contract.
    if kernel == "gemm":
        a800_map = {}
    elif kernel in ("eager_attn", "unfused_moe", "vector"):
        if a800_json is None:
            raise ValueError(
                f"{kernel} requires an NVIDIA A800 routing dataset owner")
        a800_map = _load_a800_attr_map(a800_json)
        if not a800_map:
            raise ValueError(
                f"NVIDIA A800 {a800_json} routing dataset is empty")
    else:
        raise ValueError(f"unsupported kernel: {kernel!r}")

    def simulate(task_key, gpu_key, case):
        """Return one complete case or fail on missing source observations."""
        if kernel == "gemm":
            B, M, K, N, dtype = case
            lat = _run_unindp_gemm(unindp_repo, arch, topk, quicksearch, timeout_s,
                                    workspace_rel, task_key, B * M, K, N)
            return _make_gemv_entry(B, M, K, N, dtype, lat), lat
        elif kernel in ("eager_attn", "unfused_moe", "vector"):
            if gpu_key not in a800_map:
                raise KeyError(
                    "NVIDIA A800 routing dataset has no source case "
                    f"{gpu_key!r}")
            attr = a800_map[gpu_key]
        else:
            raise ValueError(f"unsupported kernel: {kernel!r}")

        if kernel == "eager_attn":
            lat, _ = _decomp_eager_attn_latency(attr, run_fn, task_key)
            return _make_entry("EAGERATTN", attr, lat, "pim_eagerattn"), lat
        elif kernel == "vector":
            lat, _ = _decomp_vector_latency(attr, run_fn, task_key)
            return _make_entry(_VECTOR_OP_TYPE[attr["op"]], attr, lat,
                                "pim_vector"), lat
        elif kernel == "unfused_moe":
            # grid_size must match the cloned w1 tiling grid so projection
            # selection resolves to the source observation.
            lat, _ = _decomp_moe_latency(attr, run_fn, task_key)
            moe_tiling = require_mapping(
                attr["moe_tiling"], "UNFUSEDMOE Attr.moe_tiling")
            w1_tiling = require_mapping(
                moe_tiling["w1"], "UNFUSEDMOE Attr.moe_tiling.w1")
            w1_grid = require_positive_int(
                w1_tiling, "grid", "UNFUSEDMOE Attr.moe_tiling.w1")
            return _make_entry(
                "UNFUSEDMOE", attr, lat, "pim_moe_decomposed",
                grid_size=f"{w1_grid},1,1"), lat
        else:
            raise ValueError(f"unsupported kernel: {kernel!r}")

    if enumerate_fn is not None:
        all_cases = list(enumerate_fn(kernel_cfg))
    else:
        # vector/eager_attn use the A800 observation as the shape source,
        # including serving-variant Q_LEN/KV_LEN geometry.  The source and
        # output IDs are deliberately distinct for EAGERATTN.
        if kernel == "vector":
            all_cases = [(key, key, attr) for key, attr in a800_map.items()]
        elif kernel == "eager_attn":
            all_cases = [
                (_eager_attn_case_id(source_key), source_key, attr)
                for source_key, attr in a800_map.items()
            ]
        else:
            raise AssertionError(
                "enumerate_fn=None is valid only for vector/eager_attn")
    if not all_cases:
        raise RuntimeError(
            f"no source cases available for {kernel}; refusing to publish an empty dataset")
    expected_keys = tuple(task_key for task_key, _, _ in all_cases)
    # slice for --start-idx / --end-idx / --max-tasks
    lo = start_idx - 1 if start_idx > 0 else 0
    hi = len(all_cases) if end_idx <= 0 else min(end_idx, len(all_cases))
    cases = all_cases[lo:hi]
    if max_tasks > 0:
        cases = cases[:max_tasks]

    out_path = get_kernel_dataset_path(cfg, kernel)
    candidate_dir = os.path.join(get_profiler_output_dir(cfg), "candidate")
    candidate_path = os.path.join(candidate_dir, f"{kernel}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(candidate_dir, exist_ok=True)

    # Resume from both the last canonical dataset and the newer candidate.
    # Candidate entries override matching canonical entries. Only the ignored
    # candidate path is used as a per-case checkpoint while simulation runs.
    existing = {}
    for resume_path in (out_path, candidate_path):
        if os.path.exists(resume_path):
            if resume_path.endswith(KERNEL_DATASET_SUFFIX):
                payload = load_kernel_dataset_document(resume_path)
            elif resume_path.endswith(".json"):
                with open(resume_path) as stream:
                    payload = json.load(stream)
            else:
                raise ValueError(
                    f"unsupported resume dataset format: {resume_path}")
            cached = require_mapping(
                payload, f"resume dataset {resume_path}")
            existing.update(cached)

    results = dict(existing)
    skipped = []
    n_new = 0
    todo = [(tk, gk, c) for (tk, gk, c) in cases if tk not in existing]
    print(f"\n>>> Stage: Aggregate [{kernel}] — {len(cases)} cases "
          f"({len(existing)} cached, {len(todo)} to run) via UniNDP {arch}", flush=True)
    print(f"    repo={unindp_repo}  timeout={timeout_s}s  workspace={workspace_rel}  "
          f"jobs={jobs}", flush=True)

    def worker(task_key, gpu_key, case):
        """Run one case; return (task_key, entry|None, latency|None, reason|None)."""
        try:
            entry, lat = simulate(task_key, gpu_key, case)
            if entry is None:
                raise TypeError(
                    f"simulation returned no entry for {task_key!r}")
            return task_key, entry, lat, None
        except (TimeoutError, RuntimeError, AssertionError) as error:
            return task_key, None, None, str(error)

    # Concurrent generation: each UniNDP subprocess is single-threaded, so `jobs`
    # is the core budget. Results are lock-protected and checkpointed per
    # completion so an interrupt never loses finished work.
    lock = threading.Lock()
    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(worker, tk, gk, c) for (tk, gk, c) in todo]
        for fut in as_completed(futs):
            task_key, entry, lat, reason = fut.result()
            done += 1
            with lock:
                if entry is not None:
                    results[task_key] = entry
                    n_new += 1
                    print(f"  [{done:3d}/{len(todo)}] {task_key}: {lat:.0f} ns",
                          flush=True)
                else:
                    skipped.append((task_key, reason))
                    print(f"  [{done:3d}/{len(todo)}] {task_key}: SKIP ({reason})",
                          flush=True)
                # checkpoint after each completion (lock-protected)
                with open(candidate_path, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    completeness_error = None
    try:
        publish_complete_dataset(
            results, expected_keys, candidate_path, out_path)
    except DatasetCompletenessError as error:
        completeness_error = error
        print(f"\n  candidate only: {candidate_path} ({len(results)} cached, "
              f"{len(error.missing)} missing, {len(skipped)} skipped)",
              flush=True)
    else:
        print(f"\n  published {out_path}: {len(expected_keys)} entries "
              f"({n_new} new)", flush=True)

    if skipped:
        skip_log = os.path.join(candidate_dir, f"{kernel}_skipped.log")
        with open(skip_log, "w") as f:
            for task_key, reason in skipped:
                f.write(f"{task_key}\t{reason}\n")
        print(f"  skip log: {skip_log}", flush=True)

    if completeness_error is not None:
        raise completeness_error


def main():
    parser = argparse.ArgumentParser(
        description="HBM-PIM dataset generation via UniNDP simulator (groundtruth latency)"
    )
    parser.add_argument("--config", required=True,
                        help="path to a separately supplied dataset configuration")
    parser.add_argument("--kernel", required=True,
                        help="Comma-separated: gemm,eager_attn,unfused_moe,vector "
                             "(eager_attn/unfused_moe/vector use UniNDP decomposition)")
    parser.add_argument(
        "--stage", required=True, choices=["aggregate"],
        help="Run simulator ground truth and publish canonical dataset JSON.")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Cap cases per kernel (0 = all)")
    parser.add_argument(
        "--start-idx", type=int, default=0,
        help="Start case index (1-based; 0 starts at the first case).")
    parser.add_argument(
        "--end-idx", type=int, default=0,
        help="End case index (inclusive; 0 runs through the last case).")
    parser.add_argument("-j", "--jobs", type=int, default=32,
                        help="Concurrent UniNDP sims (each single-threaded = 1 core; "
                             "default 32). Set higher on core-rich machines.")
    parser.add_argument("--timeout", type=int, default=0,
                        help="Per-op UniNDP sim timeout in seconds (0 = use config "
                             "sim_timeout_s). Bump to recover slow mid-size cases.")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    timeout_s = (
        args.timeout if args.timeout > 0
        else require_positive_int(
            get_kernel_pipeline_settings(cfg), "sim_timeout_s",
            "kernel pipeline settings")
    )
    kernels = parse_kernel_list(cfg, args.kernel)

    if args.stage == "aggregate":
        for kernel in kernels:
            stage_aggregate(cfg, kernel, args.max_tasks, args.start_idx, args.end_idx,
                            args.jobs, timeout_s)
    else:
        raise ValueError(f"unsupported dataset stage: {args.stage!r}")
