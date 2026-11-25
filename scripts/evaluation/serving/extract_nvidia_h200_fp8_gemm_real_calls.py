#!/usr/bin/env python3
'Extract real H200 FP8 block-scaled dense GEMM calls from nsys GPU traces.\n\nEvaluation only: per-call durations printed here are compared with Midlayer\npredictions in WP_REPORT.md and are never read by a predictor, HWSpec or\nlaunch selector.\n\nFor every ``--window LABEL:CSV:M`` (an nsys ``cuda_gpu_trace`` CSV) this\nkeeps one device, walks its kernels in start order and records\n\n* ``deep_gemm::fp8_gemm_kernel_swapAB<N, K, ...>`` (TensorRT-LLM/FlashInfer\n  swap-AB DeepGEMM, M < 32 path) together with the\n  ``tensorrt_llm::...::scale_1x128_kernel`` launched immediately before it on\n  the same CUDA stream by the same ``fp8_gemm_run`` call;\n* ``deep_gemm::sm90_fp8_gemm_1d2d_impl<...>`` (DeepGEMM ``fp8_gemm_nt``,\n  M >= 32 path) with its GEMM M recovered from the launch geometry of the\n  ``per_token_group_quant_8bit_kernel`` that quantised its activation.\n\nKernels are grouped by (kernel, N, K, template tuple, launch geometry); the\noutput holds count, median, p10 and p90 of the per-call duration and the\nlaunch geometry columns.  Standard library only (runs with the remote system\npython3).'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset.nvidia_h200_fp8_gemm_calls import main

if __name__ == "__main__":
    raise SystemExit(main())
