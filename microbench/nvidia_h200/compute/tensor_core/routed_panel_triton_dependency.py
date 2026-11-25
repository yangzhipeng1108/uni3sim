"""Independent routed-panel K-loop, without MoE router or matrix output.

The workload preserves vLLM 0.30's grouped M/N program order, sorted-row
lookup, expert lookup, masked A load, B load, and BF16 dot dependency.  It
stores one checksum per CTA solely to keep the K-loop live.  The 4/8/12-round
depths and two synthetic footprints test an independent mechanism; this does
not measure a production operator or set a production duration parameter.
"""

import argparse
import json

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["K_ROUNDS"])
def routed_panel_kloop(
    a_ptr,
    b_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    checksum_ptr,
    N,
    K,
    EM,
    NUM_VALID_TOKENS,
    TOP_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    K_ROUNDS,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    if pid_m * BLOCK_SIZE_M >= tl.load(num_tokens_post_padded_ptr):
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < NUM_VALID_TOKENS
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        return

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // TOP_K * K + offs_k[None, :])
    b_ptrs = (b_ptr + off_experts * N * K +
              offs_k[:, None] + offs_bn[None, :] * K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K_ROUNDS):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] &
            (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    checksum = tl.sum(tl.sum(accumulator, 0), 0)
    tl.store(checksum_ptr + pid, checksum)


CONDITIONS = {
    "small": {"valid_rows": 32768, "expert_count": 8,
              "m_tiles": 264, "grid": 2112},
    "large": {"valid_rows": 524288, "expert_count": 128,
              "m_tiles": 4223, "grid": 33784},
}


def run_once(condition: str, rounds: int, samples: int) -> dict:
    """Launch synthetic routed panels; only this probe's CUDA events are timed."""
    import torch

    if condition not in CONDITIONS or rounds not in (4, 8, 12):
        raise ValueError("unknown independent panel condition or K depth")
    if samples < 1 or samples > 7:
        raise ValueError("invalid bounded sample count")
    row = CONDITIONS[condition]
    n = 2048
    k = 768
    em = row["m_tiles"] * 128
    grid = row["grid"]
    if grid != row["m_tiles"] * (n // 256):
        raise ValueError("grid differs from grouped M/N source program")
    valid = row["valid_rows"]
    device = "cuda"
    a = torch.full((valid, k), 0.25, device=device, dtype=torch.bfloat16)
    b = torch.full((row["expert_count"], n, k), 0.125,
                   device=device, dtype=torch.bfloat16)
    positions = torch.arange(em, device=device, dtype=torch.int32)
    # Odd affine permutation preserves unique rows inside the valid power-of-two
    # range.  Padded positions retain the original invalid sentinel.
    sorted_rows = torch.where(positions < valid,
                              (positions * 1103515245 + 12345) % valid,
                              torch.full_like(positions, valid))
    expert_ids = (torch.arange(row["m_tiles"], device=device,
                               dtype=torch.int32) // 33)
    padded = torch.tensor(em, device=device, dtype=torch.int32)
    checksums = torch.empty(grid, device=device, dtype=torch.float32)
    if int(expert_ids.max().item()) != row["expert_count"] - 1:
        raise ValueError("expert ID construction differs from source domain")
    def launch():
        routed_panel_kloop[(grid,)](
            a, b, sorted_rows, expert_ids, padded, checksums,
            n, k, em, valid, 1, 128, 256, 64, 16, rounds,
            num_warps=8, num_stages=4)
    for _ in range(2):
        launch()
    torch.cuda.synchronize()
    times_ms = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        stop.record()
        stop.synchronize()
        times_ms.append(start.elapsed_time(stop))
    return {"condition": condition, "rounds": rounds,
            "source_grid": grid, "valid_rows": valid,
            "expert_count": row["expert_count"], "top_k_for_w2": 1,
            "activation_bytes": a.numel() * a.element_size(),
            "weight_bytes": b.numel() * b.element_size(),
            "checksum_finite": bool(torch.isfinite(checksums).all().item()),
            "independent_probe_event_ms": times_ms}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=tuple(CONDITIONS))
    parser.add_argument("--rounds", required=True, type=int, choices=(4, 8, 12))
    parser.add_argument("--samples", type=int, default=1)
    arguments = parser.parse_args()
    print(json.dumps(run_once(arguments.condition, arguments.rounds,
                              arguments.samples)))
