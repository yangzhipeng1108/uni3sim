#!/usr/bin/env python3
"""Control descriptor and pointer reuse in an independent GMM Host submit.

Only the asynchronous API call is timed. Device synchronization, allocations,
and tensor destruction stay outside. Fixed tiny synthetic shapes do not read
Serving inputs or durations. Conditions name controlled inputs, not inferred
cache hits; results alone do not authorize a Forward cache policy or price.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    visible = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
    if not visible.isdecimal():
        raise ValueError("select exactly one physical NPU")
    import torch
    import torch_npu

    torch.npu.set_device("npu:0")
    opts = {"device": "npu:0", "dtype": torch.bfloat16}
    groups, hidden, output = 16, 64, 128

    def operands(rows):
        return (torch.empty((rows, hidden), **opts),
                torch.empty((groups, hidden, output), **opts),
                torch.full((groups,), rows // groups,
                           dtype=torch.int64, device="npu:0"))

    def submit(values):
        x, weight, counts = values
        return torch_npu.npu_grouped_matmul(
            x=[x], weight=[weight], split_item=2,
            group_list_type=1, group_type=0, group_list=counts)[0]

    # Exclude process initialization and first API/library use from the
    # subsequent descriptor/pointer controls, while recording them separately.
    prime = operands(128)
    torch.npu.synchronize()
    first_start = time.perf_counter_ns()
    result = submit(prime)
    first_ns = time.perf_counter_ns() - first_start
    torch.npu.synchronize()
    del result
    for _ in range(10):
        result = submit(prime)
        torch.npu.synchronize()
        del result

    records = []
    for rows in (16, 32, 48, 64):
        base = operands(rows)
        replacements = tuple(operands(rows) for _ in range(12))
        cases = (
            ("same_operands", (base,) * 12),
            ("new_input_pointer", tuple((v[0], base[1], base[2]) for v in replacements)),
            ("new_weight_pointer", tuple((base[0], v[1], base[2]) for v in replacements)),
            ("new_count_pointer", tuple((base[0], base[1], v[2]) for v in replacements)),
            ("all_new_pointers", replacements),
        )
        for condition, sequence in cases:
            for index, values in enumerate(sequence):
                torch.npu.synchronize()
                wall_start = time.perf_counter_ns()
                cpu_start = time.thread_time_ns()
                result = submit(values)
                cpu_ns = time.thread_time_ns() - cpu_start
                wall_ns = time.perf_counter_ns() - wall_start
                torch.npu.synchronize()
                records.append({"rows": rows, "condition": condition,
                    "index": index, "host_wall_ns": wall_ns,
                    "host_thread_cpu_ns": cpu_ns,
                    "input_pointers": [v.data_ptr() for v in values]})
                del result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema_version": 1,
        "kind": "independent_gmm_host_descriptor_pointer_controls",
        "forward_pricing_allowed": False,
        "production_or_evaluation_duration_consumed": False,
        "boundary": "Host API entry to asynchronous return; synchronization excluded",
        "physical_npu": int(visible), "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "groups": groups,
        "hidden": hidden, "output": output, "split_item": 2,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "first_api_host_wall_ns": first_ns, "records": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
