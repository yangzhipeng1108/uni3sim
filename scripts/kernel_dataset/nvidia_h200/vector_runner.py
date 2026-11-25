"""NVIDIA H200 owner copy of ``scripts/kernel_dataset/nvidia_a800/vector_runner.py``.

Executes the H200 deployment stack (vLLM 0.24.0 custom ops / torch 2.11 in
the caller-selected runtime) under the A800 canonical NSYS NVTX
timing protocol.  Changes versus the A800 runner, all hardware/stack coupling:

* Only the NVTX (normal) mode is kept: it is the mode the A800 canonical
  timing (``nsys_nvtx_kernel_median``) used.  The device-Event rotating-ring,
  batched and CUDA-Graph modes (A800 L2-coupled contracts in
  ``nvidia_a800_event_capture``) are not ported, so the ring/batch/graph CLI
  options and their helpers are removed; every op keeps its single-call path.
* ``H200CaptureSession`` (``dataset_pipeline/kernel_dataset/nvidia_h200_capture.py``):
  GPU lock before the first CUDA call, device identity check (NVIDIA H200,
  132 SMs, 60 MiB L2), BF16 GEMM power warm-up before the runner's WARMUP=5
  loop, SM-clock query before NVTX push / after NVTX pop.
* Optional pre-NVTX cache sweep minimum is 480 MiB (8 x 60 MiB L2) instead of
  320 MiB; the canonical protocol leaves it at 0.
* vLLM 0.11 -> 0.24: ``RotaryEmbedding.forward_cuda``, ``SiluAndMul.forward_cuda``,
  ``a+b``/``a*b`` and ``ops.reshape_and_cache_flash`` keep their 0.11 call form
  and still dispatch the vLLM CUDA custom kernels (``rotary_embedding_kernel``,
  ``act_and_mul_kernel``, ``reshape_and_cache_flash_kernel``).  0.24
  ``RMSNorm.forward_cuda`` no longer calls a custom kernel (it returns
  ``forward_native``, an 8-11 kernel eager ATen chain), so rmsnorm /
  add_rmsnorm execute the 0.11 ``forward_cuda`` body with the 0.24 custom ops
  it called: ``ops.rms_norm(empty_like(x), x, w, eps)`` and in-place
  ``ops.fused_add_rms_norm(x, residual, w, eps)``.  vLLM 0.24 ``CustomOp`` construction requires a
  current ``VllmConfig``, so the case runs inside
  ``set_current_vllm_config(VllmConfig())`` (default config); the explicit
  ``forward_cuda`` calls keep selecting the vLLM CUDA custom kernels exactly
  as the A800 runner did (a torch.compile deployment may instead fuse these
  ops into Inductor kernels; that path is not what the A800 case measures).

Unchanged: shapes, dtypes, module construction, WARMUP=5, one exact public
op call inside the NVTX range.

vLLM 0.30 (``installed_vllm_stack() == "vllm030"``): every call above keeps
its 0.24 form and kernel.  ``RMSNorm.forward_cuda`` still returns
``forward_native`` (vLLM IR ``ir.ops.rms_norm``; the eager platform default
priority ``["vllm_c", "native"]`` maps to the same ``torch.ops._C.rms_norm``
/ ``fused_add_rms_norm`` that ``ops.rms_norm`` / ``ops.fused_add_rms_norm``
call), ``RotaryEmbedding`` (``rotary_embedding/base.py`` unchanged),
``SiluAndMul.forward_cuda`` (``torch.ops._C.silu_and_mul``) and
``ops.reshape_and_cache_flash`` have the same signatures.  The stack and the
per-op call path are recorded in the ``environment`` evidence.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.nvidia_h200_capture import (
    H200_MINIMUM_CACHE_EVICTION_MIB,
    H200_HARDWARE_ID,
    add_h200_capture_arguments,
    installed_vllm_stack,
    open_h200_capture_session,
)

import torch
import vllm
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

from evaluation.kernel.device_event import (
    prepare_bf16_cache_eviction_sweep,
    validate_cache_eviction_mib,
)

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}
WARMUP = 5
# Executed public op per --op; the same call form on vLLM 0.24 and 0.30.
OP_CALL_PATH = {
    "rmsnorm": "vllm._custom_ops.rms_norm(empty_like(x), x, w, eps) -> torch.ops._C.rms_norm",
    "add_rmsnorm": ("vllm._custom_ops.fused_add_rms_norm(x, residual, w, eps) "
                    "-> torch.ops._C.fused_add_rms_norm (in place)"),
    "rope": "RotaryEmbedding(is_neox_style=True).forward_cuda(positions, q, k)",
    "silu": "SiluAndMul.forward_cuda(x) -> torch.ops._C.silu_and_mul",
    "elementwise": "torch a+b / a*b",
    "kvcache": ("vllm._custom_ops.reshape_and_cache_flash(key, value, key_cache, "
                "value_cache, slot_mapping, 'auto', k_scale, v_scale)"),
}
VLLM_STACKS = ("vllm024", "vllm030")


def _measure(fn, nvtx_range_name, cache_eviction_operation, capture):
    """Warm outside NVTX, then expose exactly one post-warm launch."""
    torch.cuda.synchronize()
    capture.power_warmup(torch)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    if cache_eviction_operation is not None:
        cache_eviction_operation()
        torch.cuda.current_stream().synchronize()
    capture.before_nvtx()
    torch.cuda.nvtx.range_push(nvtx_range_name)
    fn()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    capture.after_nvtx()


def _environment(op):
    vllm_stack = installed_vllm_stack()
    if vllm_stack not in VLLM_STACKS:
        raise RuntimeError(f"H200 Vector runner has no call path for {vllm_stack}")
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": vllm.__version__,
        "input_seed": None,
        "vllm_stack": vllm_stack,
        "vllm_call_path": OP_CALL_PATH[op],
    }


def _run(fn, *, op, nvtx_range_name, capture, pre_nvtx_cache_eviction_mib=0):
    if not isinstance(nvtx_range_name, str) or not nvtx_range_name:
        raise ValueError("H200 Vector runner requires an NVTX name")
    validate_cache_eviction_mib(
        pre_nvtx_cache_eviction_mib,
        minimum_nonzero_mib=H200_MINIMUM_CACHE_EVICTION_MIB,
        hardware_id=H200_HARDWARE_ID)
    cache_eviction_operation, _ = prepare_bf16_cache_eviction_sweep(
        torch_module=torch, device="cuda",
        cache_eviction_mib=pre_nvtx_cache_eviction_mib)
    _measure(fn, nvtx_range_name, cache_eviction_operation, capture)
    capture.add_launch_evidence("environment", _environment(op))


def run_rmsnorm(tok, hidden, nvtx, capture, dtype="bf16", add_residual=False,
                pre_nvtx_cache_eviction_mib=0):
    dt = DTYPE_MAP[dtype.lower()]
    rn = RMSNorm(hidden).to("cuda").to(dt)
    x = torch.randn(tok, hidden, device="cuda", dtype=dt)
    # vLLM 0.24 RMSNorm.forward_cuda delegates to forward_native (eager ATen
    # chain); the vLLM 0.11 forward_cuda body that the A800 runner executed
    # is reproduced here with the 0.24 custom ops it called.
    if add_residual:
        residual = torch.randn(tok, hidden, device="cuda", dtype=dt)

        def fn():
            ops.fused_add_rms_norm(
                x, residual, rn.weight.data, rn.variance_epsilon)
            return x, residual
    else:
        def fn():
            out = torch.empty_like(x)
            ops.rms_norm(out, x, rn.weight.data, rn.variance_epsilon)
            return out
    _run(fn, op="add_rmsnorm" if add_residual else "rmsnorm",
         nvtx_range_name=nvtx, capture=capture,
         pre_nvtx_cache_eviction_mib=pre_nvtx_cache_eviction_mib)


def run_rope(tok, num_heads, num_kv_heads, head_dim, nvtx, capture,
             dtype="bf16", pre_nvtx_cache_eviction_mib=0):
    dt = DTYPE_MAP[dtype.lower()]
    re = RotaryEmbedding(head_size=head_dim, rotary_dim=head_dim,
                         max_position_embeddings=max(tok, 4096), base=10000,
                         is_neox_style=True, dtype=dt).to("cuda")
    positions = torch.arange(tok, device="cuda", dtype=torch.long)
    q = torch.randn(tok, num_heads * head_dim, device="cuda", dtype=dt)
    k = torch.randn(tok, num_kv_heads * head_dim, device="cuda", dtype=dt)
    fn = lambda: re.forward_cuda(positions, q, k)
    _run(fn, op="rope", nvtx_range_name=nvtx, capture=capture,
         pre_nvtx_cache_eviction_mib=pre_nvtx_cache_eviction_mib)


def run_silu(tok, intermediate, nvtx, capture, dtype="bf16",
             pre_nvtx_cache_eviction_mib=0):
    dt = DTYPE_MAP[dtype.lower()]
    sm = SiluAndMul().to("cuda")
    x = torch.randn(tok, 2 * intermediate, device="cuda", dtype=dt)
    fn = lambda: sm.forward_cuda(x)
    _run(fn, op="silu", nvtx_range_name=nvtx, capture=capture,
         pre_nvtx_cache_eviction_mib=pre_nvtx_cache_eviction_mib)


def run_elementwise(tok, hidden, nvtx, capture, dtype="bf16", op2="add",
                    pre_nvtx_cache_eviction_mib=0):
    dt = DTYPE_MAP[dtype.lower()]
    a = torch.randn(tok, hidden, device="cuda", dtype=dt)
    b = torch.randn(tok, hidden, device="cuda", dtype=dt)
    if op2 == "add":
        fn = lambda: a + b
    elif op2 == "mul":
        fn = lambda: a * b
    else:
        raise ValueError(f"unknown elementwise operation: {op2!r}")
    _run(fn, op="elementwise", nvtx_range_name=nvtx, capture=capture,
         pre_nvtx_cache_eviction_mib=pre_nvtx_cache_eviction_mib)


def run_kvcache(tok, num_kv_heads, head_dim, nvtx, capture, batch=1,
                dtype="bf16", pre_nvtx_cache_eviction_mib=0):
    """Paged KV write via reshape_and_cache_flash (A800 single-call path)."""
    if min(tok, num_kv_heads, head_dim, batch) <= 0:
        raise ValueError("KV-cache workload dimensions must be positive")
    dt = DTYPE_MAP[dtype.lower()]
    key = torch.randn(tok, num_kv_heads, head_dim, device="cuda", dtype=dt)
    value = torch.randn(tok, num_kv_heads, head_dim, device="cuda", dtype=dt)
    block_size = 16
    num_blocks = (tok + block_size - 1) // block_size + 1
    key_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim,
                            device="cuda", dtype=dt)
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.arange(tok, device="cuda", dtype=torch.long)
    k_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    v_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    fn = lambda: ops.reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping, "auto",
        k_scale, v_scale)
    _run(fn, op="kvcache", nvtx_range_name=nvtx, capture=capture,
         pre_nvtx_cache_eviction_mib=pre_nvtx_cache_eviction_mib)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True,
                   choices=["rmsnorm", "add_rmsnorm", "rope", "silu", "elementwise", "kvcache"])
    for name in ("tok", "hidden", "num_heads", "head_dim", "intermediate",
                 "num_kv_heads", "batch"):
        p.add_argument(f"--{name}", type=int, required=True)
    p.add_argument("--dtype", required=True, choices=sorted(DTYPE_MAP))
    p.add_argument("--op2", required=True, choices=["add", "mul"])
    p.add_argument("--nvtx_range_name", type=str, required=True)
    p.add_argument("--pre-nvtx-cache-eviction-mib", type=int, default=0)
    add_h200_capture_arguments(p)
    a = p.parse_args()
    capture = open_h200_capture_session(a, a.nvtx_range_name)
    torch.cuda.set_device(0)
    capture.validate_device(torch)
    common = {"pre_nvtx_cache_eviction_mib": a.pre_nvtx_cache_eviction_mib}
    vllm_config_context = set_current_vllm_config(VllmConfig())
    vllm_config_context.__enter__()
    if a.op == "rmsnorm":
        run_rmsnorm(a.tok, a.hidden, a.nvtx_range_name, capture, a.dtype,
                    add_residual=False, **common)
    elif a.op == "add_rmsnorm":
        run_rmsnorm(a.tok, a.hidden, a.nvtx_range_name, capture, a.dtype,
                    add_residual=True, **common)
    elif a.op == "rope":
        run_rope(a.tok, a.num_heads, a.num_kv_heads, a.head_dim,
                 a.nvtx_range_name, capture, a.dtype, **common)
    elif a.op == "silu":
        run_silu(a.tok, a.intermediate, a.nvtx_range_name, capture, a.dtype,
                 **common)
    elif a.op == "elementwise":
        run_elementwise(a.tok, a.hidden, a.nvtx_range_name, capture, a.dtype,
                        a.op2, **common)
    elif a.op == "kvcache":
        run_kvcache(a.tok, a.num_kv_heads, a.head_dim, a.nvtx_range_name,
                    capture, a.batch, a.dtype, **common)
    else:
        raise AssertionError(f"argparse accepted an unknown vector op: {a.op!r}")
    vllm_config_context.__exit__(None, None, None)
    capture.finish()
    print("VEC done")
