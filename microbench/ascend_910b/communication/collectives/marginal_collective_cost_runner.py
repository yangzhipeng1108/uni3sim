#!/usr/bin/env python3
"""Ascend 910B marginal collective cost: HCCL all_reduce inside a busy stream.

The standalone HCCL sweep already published in
``microbench/ascend_910b/references/hccl_fullmesh_collective_startup_cann_8_5_1.json``
measures one collective back-to-back at 127-137 us from 4 KiB to 128 KiB.  That
figure cannot be charged to a decode step: the step runs on one device serially,
so its device time cannot exceed the measured token cadence, and 126 us per
collective pushes the predicted step past that bound on both TP=2 models.  What
the model charges is the *marginal* cost of a collective issued into a stream
that already holds compute, and the same sweep bounds it from the other side --
its isolated pattern costs 220 us against 127 us back-to-back, so roughly 93 us
of the standalone figure is launch and rank skew that surrounding compute
absorbs.

This probe measures that marginal cost directly, as a matched A/B on one
compute chain:

``arm A``
    ``chain_length`` identical synthetic matmuls, bounded by one device-event
    pair.
``arm B``
    the same chain, with one ``all_reduce`` of ``message_bytes`` inserted after
    every ``insert_every``-th matmul, bounded the same way.

``(T_B - T_A) / collective_count`` is the marginal cost.  Both arms issue
byte-identical compute, so the difference isolates what the collective adds on
top of an already-busy stream.

It is a communication mechanism sweep: synthetic byte counts, a constant filled
payload, workload-independent participation.  No production operator, model
shape, kernel or Serving duration takes part, and nothing it emits is selected
by comparing a Midlayer prediction against a production measurement.

Design and acceptance criteria are frozen in
``MARGINAL_COLLECTIVE_REGISTRATION.md`` beside this file.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
from pathlib import Path

MESSAGE_BYTES = (4 * 1024, 16 * 1024, 64 * 1024, 143 * 1024,
                 256 * 1024, 512 * 1024)
INSERT_EVERY = (1, 4)
CHAIN_LENGTH = 64
MATMUL_DIM = 2048
STANDALONE_BACK_TO_BACK_US = 127.0


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trials", type=_positive_integer, default=9)
    parser.add_argument("--warmup", type=_positive_integer, default=20)
    parser.add_argument("--chain-length", type=_positive_integer,
                        default=CHAIN_LENGTH)
    return parser.parse_args()


def _timed_chain(torch, stream_compute, collective, count, insert_every,
                 chain_length):
    """Return the device-event span of one chain, in microseconds."""
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for index in range(chain_length):
        stream_compute()
        if collective is not None and (index + 1) % insert_every == 0:
            collective()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000.0


def main() -> int:
    args = _parse_args()
    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError("marginal collective probe requires exactly two ranks")
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

    dim = MATMUL_DIM
    left = torch.ones((dim, dim), dtype=torch.bfloat16, device=f"npu:{rank}")
    right = torch.ones((dim, dim), dtype=torch.bfloat16, device=f"npu:{rank}")

    def compute():
        torch.matmul(left, right)

    records = []
    for message_bytes in MESSAGE_BYTES:
        elements = message_bytes // 2
        payload = torch.ones(elements, dtype=torch.bfloat16,
                             device=f"npu:{rank}")

        def collective():
            dist.all_reduce(payload)

        for insert_every in INSERT_EVERY:
            collective_count = args.chain_length // insert_every
            for _ in range(args.warmup):
                _timed_chain(torch, compute, collective, collective_count,
                             insert_every, args.chain_length)
            dist.barrier()
            a_us, b_us = [], []
            for _ in range(args.trials):
                a_us.append(_timed_chain(torch, compute, None, 0,
                                         insert_every, args.chain_length))
                b_us.append(_timed_chain(torch, compute, collective,
                                         collective_count, insert_every,
                                         args.chain_length))
            dist.barrier()
            a_med, b_med = st.median(a_us), st.median(b_us)
            marginal = (b_med - a_med) / collective_count
            records.append({
                "message_bytes": message_bytes,
                "insert_every": insert_every,
                "collective_count": collective_count,
                "chain_length": args.chain_length,
                "matmul_dim": dim,
                "dtype": "bfloat16",
                "arm_a_us": round(a_med, 4),
                "arm_b_us": round(b_med, 4),
                "arm_a_samples_us": [round(v, 4) for v in a_us],
                "arm_b_samples_us": [round(v, 4) for v in b_us],
                "marginal_us_per_collective": round(marginal, 5),
                "absorbed_fraction_vs_standalone": round(
                    1.0 - marginal / STANDALONE_BACK_TO_BACK_US, 5),
            })
            print(f"rank={rank} bytes={message_bytes} every={insert_every} "
                  f"A={a_med:.1f}us B={b_med:.1f}us "
                  f"marginal={marginal:.3f}us/collective", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"marginal_collective_{args.run_id}_rank{rank}.json"
    out.write_text(json.dumps({
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "run_id": args.run_id,
        "source_commit": args.source_commit,
        "rank": rank,
        "world_size": world_size,
        "link": "hccs",
        "operation": "all_reduce",
        "backend": "hccl",
        "trials": args.trials,
        "warmup": args.warmup,
        "standalone_back_to_back_us": STANDALONE_BACK_TO_BACK_US,
        "records": records,
    }, indent=1) + "\n", encoding="utf-8")
    print(f"rank={rank} wrote {out}", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
