"""NVIDIA H200 owner copy of ``scripts/kernel_dataset/nvidia_a800/gemm_runner.py``.

Executes the caller-selected H200 deployment stack (torch 2.11 + cuBLAS 12.9; ``torch.nn.Linear`` dispatches the
same cuBLAS GEMM path vLLM's unquantized linear layers use) under the A800
canonical NSYS NVTX timing protocol.  Changes versus the A800 runner, all
hardware/stack coupling:

* Only the NVTX (normal) mode is kept: it is the mode the A800 canonical
  timing (``nsys_nvtx_kernel_median``) used.  The device-Event mode is not
  ported.
* ``H200CaptureSession`` (``dataset_pipeline/kernel_dataset/nvidia_h200_capture.py``):
  GPU lock before the first CUDA call, device identity check (NVIDIA H200,
  132 SMs, 60 MiB L2), BF16 GEMM power warm-up after input construction,
  SM-clock query before NVTX push / after NVTX pop.
* Optional pre-NVTX cache sweep minimum is 480 MiB (8 x 60 MiB L2) instead of
  320 MiB; the canonical protocol leaves it at 0.
* The canonical GEMM job never seeded the RNG in NVTX mode; the H200 copy
  keeps that (inputs are ``torch.randn`` of the case shape).

Unchanged: input and ``torch.nn.Linear(bias=False)`` construction, no
pre-NVTX warm-up of the op (A800 canonical GEMM job had none), one exact
``linear_layer(input_tensor)`` call inside the NVTX range.

vLLM stacks (``installed_vllm_stack``): the call is the same on vLLM 0.24
(torch 2.11) and vLLM 0.30 (torch 2.13).
In both, ``UnquantizedLinearMethod`` on SM90 dispatches
``default_unquantized_gemm`` = ``torch.nn.functional.linear`` (0.30's
``--linear-backend auto`` only switches to the FlashInfer CuTe-DSL BF16 GEMM
on SM100), so ``torch.nn.Linear`` stays the deployment GEMM call; the stack
and call path are recorded in the ``environment`` evidence.
"""

import argparse
import importlib.metadata
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

from evaluation.kernel.device_event import (
    prepare_bf16_cache_eviction_sweep,
    validate_cache_eviction_mib,
)


# Deployment call per supported vLLM stack (identical on both).
VLLM_CALL_PATH = {
    "vllm024": ("torch.nn.Linear(bias=False) -> F.linear (cuBLAS); vLLM 0.24 "
                "UnquantizedLinearMethod -> default_unquantized_gemm on SM90"),
    "vllm030": ("torch.nn.Linear(bias=False) -> F.linear (cuBLAS); vLLM 0.30 "
                "UnquantizedLinearMethod -> dispatch_unquantized_gemm('auto') "
                "-> default_unquantized_gemm on SM90"),
}

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def run_gemm(batch_size, seq_len, in_features, out_features,
             nvtx_range_name, capture, dtype=torch.bfloat16,
             pre_nvtx_cache_eviction_mib=0):
    vllm_stack = installed_vllm_stack()
    input_tensor = torch.randn(batch_size, seq_len, in_features, device='cuda', dtype=dtype)
    linear_layer = torch.nn.Linear(in_features, out_features, bias=False).cuda().to(dtype)

    def operation():
        with torch.no_grad():
            return linear_layer(input_tensor)

    if not isinstance(nvtx_range_name, str) or not nvtx_range_name:
        raise ValueError("H200 GEMM runner requires an NVTX name")
    validate_cache_eviction_mib(
        pre_nvtx_cache_eviction_mib,
        minimum_nonzero_mib=H200_MINIMUM_CACHE_EVICTION_MIB,
        hardware_id=H200_HARDWARE_ID)
    cache_eviction_operation, _ = prepare_bf16_cache_eviction_sweep(
        torch_module=torch, device="cuda",
        cache_eviction_mib=pre_nvtx_cache_eviction_mib)
    torch.cuda.synchronize()
    capture.power_warmup(torch)
    if cache_eviction_operation is not None:
        cache_eviction_operation()
        torch.cuda.current_stream().synchronize()
    capture.before_nvtx()
    torch.cuda.nvtx.range_push(nvtx_range_name)
    output = operation()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    capture.after_nvtx()
    capture.add_launch_evidence("environment", {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": importlib.metadata.version("vllm"),
        "vllm_stack": vllm_stack,
        "vllm_call_path": VLLM_CALL_PATH[vllm_stack],
    })
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--sl", type=int, required=True)
    parser.add_argument("--in_feat", type=int, required=True)
    parser.add_argument("--out_feat", type=int, required=True)
    parser.add_argument(
        "--dtype", choices=sorted(DTYPE_MAP), default="bf16")
    parser.add_argument("--nvtx_range_name", type=str, required=True)
    parser.add_argument("--pre-nvtx-cache-eviction-mib", type=int, default=0)
    add_h200_capture_arguments(parser)

    args = parser.parse_args()
    capture = open_h200_capture_session(args, args.nvtx_range_name)
    torch.cuda.set_device(0)
    capture.validate_device(torch)

    run_gemm(
        args.bs, args.sl, args.in_feat, args.out_feat,
        args.nvtx_range_name, capture, DTYPE_MAP[args.dtype],
        pre_nvtx_cache_eviction_mib=args.pre_nvtx_cache_eviction_mib)
    capture.finish()
