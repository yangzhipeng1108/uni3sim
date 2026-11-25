"""Vector elementwise case for NCU stall capture.

Runs a single vectorized_elementwise_kernel launch with representative
small-grid shape (T=8, H=2048, grid=16, block=128) and exits.
No timing measurement, no duration pricing - only stall attribution.
"""
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tok', type=int, default=8)
    parser.add_argument('--hidden', type=int, default=2048)
    parser.add_argument('--operation', default='add', choices=['add', 'mul'])
    args = parser.parse_args()

    device = torch.device('cuda')
    elements = args.tok * args.hidden

    # BF16 tensors matching production dtype
    if args.operation == 'add':
        a = torch.randn(elements, device=device, dtype=torch.bfloat16)
        b = torch.randn(elements, device=device, dtype=torch.bfloat16)
        # Warmup launches (skipped by --launch-skip)
        for _ in range(3):
            c = a + b
        torch.cuda.synchronize()
        # Profiled launches
        for _ in range(4):
            c = a + b
        torch.cuda.synchronize()
    else:  # mul
        a = torch.randn(elements, device=device, dtype=torch.bfloat16)
        b = torch.randn(elements, device=device, dtype=torch.bfloat16)
        for _ in range(3):
            c = a * b
        torch.cuda.synchronize()
        for _ in range(4):
            c = a * b
        torch.cuda.synchronize()


if __name__ == '__main__':
    main()
