"""One synthetic BF16 GEMM for non-duration nvjet program inspection."""

import json

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("H200 CUDA device is required")
    device = torch.cuda.get_device_name(0)
    if "H200" not in device:
        raise RuntimeError(f"expected H200, got {device}")
    m, k, n = 16384, 5120, 151936
    left = torch.ones((m, k), device="cuda", dtype=torch.bfloat16)
    # Match a linear layer's contiguous (N, K) weight and transposed view.
    weight = torch.ones((n, k), device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        output = torch.mm(left, weight.T)
        torch.cuda.synchronize()
    print(json.dumps({
        "boundary": "synthetic_bf16_gemm_static_program_only",
        "device": device,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "shape": [m, k, n],
        "output_shape": list(output.shape),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
