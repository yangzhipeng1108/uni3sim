#!/usr/bin/env python3
"""Measure one A800 launch at both span granularities the datasets use.

The three canonical kernel datasets report a "measured" duration from three
different instruments: A800 uses nsys_nvtx_kernel_median at kernel
granularity, C500 device_event_op_median and 910B msprof_timeline_op_median,
both at op granularity.  Comparing family MAPE across those cards therefore
compares predictions against different physical spans.

This probe quantifies that difference on one card, so the comparison can be
corrected rather than assumed away.  For each shape it records, over the same
launches:

  kernel span -- the CUPTI kernel duration, what nsys reports per kernel;
  op span     -- a CUDA-event interval enclosing the op, what a device-event
                 op median reports.

Both come from the same synthetic torch op; no production operator, kernel or
Serving duration is consumed, and nothing here feeds a predictor parameter.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time

import torch


def _op(name, a, b, out):
    if name == "add":
        torch.add(a, b, out=out)
    elif name == "mul":
        torch.mul(a, b, out=out)
    else:
        raise ValueError(f"unsupported op {name!r}")


def _op_span_back_to_back_ns(name, a, b, out, iters, device):
    """Amortised op span: one event pair around a back-to-back run.

    A capture's op median is taken over invocations that queue behind one
    another, so each op does not pay a cold submission.  Timing one event pair
    around `iters` ops and dividing keeps the inter-op gap but removes the
    per-iteration synchronisation the isolated form imposes.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        _op(name, a, b, out)
    torch.cuda.synchronize(device)
    start.record()
    for _ in range(iters):
        _op(name, a, b, out)
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) * 1e6 / iters


def _op_span_ns(name, a, b, out, iters, device):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        _op(name, a, b, out)
    torch.cuda.synchronize(device)
    spans = []
    for _ in range(iters):
        start.record()
        _op(name, a, b, out)
        end.record()
        torch.cuda.synchronize(device)
        spans.append(start.elapsed_time(end) * 1e6)
    return statistics.median(spans)


def _kernel_span_ns(name, a, b, out, iters, device):
    for _ in range(10):
        _op(name, a, b, out)
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            _op(name, a, b, out)
        torch.cuda.synchronize(device)
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as fh:
        path = fh.name
    prof.export_chrome_trace(path)
    events = json.load(open(path))
    events = events["traceEvents"] if isinstance(events, dict) else events
    os.unlink(path)
    durations = [
        float(event["dur"]) * 1000.0
        for event in events
        if "elementwise_kernel" in event.get("name", "")
        and "dur" in event and (event.get("args") or {}).get("grid")
    ]
    if not durations:
        raise RuntimeError("no elementwise kernel events captured")
    return statistics.median(durations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--elements", type=int, nargs="+", required=True)
    parser.add_argument("--ops", nargs="+", default=["add", "mul"])
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)
    rows = []
    for elements in args.elements:
        a = torch.rand(elements, dtype=torch.bfloat16, device=device)
        b = torch.rand(elements, dtype=torch.bfloat16, device=device)
        out = torch.empty_like(a)
        for name in args.ops:
            kernel_ns = _kernel_span_ns(name, a, b, out, args.iters, device)
            op_ns = _op_span_ns(name, a, b, out, args.iters, device)
            b2b_ns = _op_span_back_to_back_ns(
                name, a, b, out, args.iters, device)
            rows.append({
                "elements": elements, "op": name,
                "kernel_span_ns": kernel_ns,
                "op_span_isolated_ns": op_ns,
                "op_span_back_to_back_ns": b2b_ns,
                "b2b_minus_kernel_ns": b2b_ns - kernel_ns,
                "b2b_over_kernel": b2b_ns / kernel_ns,
            })
        del a, b, out
        torch.cuda.empty_cache()

    document = {
        "schema": "nvidia_a800_span_granularity_raw_v1",
        "hardware_id": "nvidia_a800",
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "physical_device_index": args.device_index,
        "kernel_span_source": "cupti_kernel_duration",
        "op_span_source": "cuda_event_interval_around_op",
        "iterations_per_measurement": args.iters,
        "target_operator_used": False,
        "production_or_evaluation_duration_consumed": False,
        "samples": rows,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=1, sort_keys=True)
    print(f"{'elements':>10s} {'op':>4s} {'kernel':>9s} {'isolated':>9s} "
          f"{'b2b':>9s} {'b2b-kern':>9s} {'b2b/kern':>9s}")
    for row in rows:
        print(f"{row['elements']:10d} {row['op']:>4s} "
              f"{row['kernel_span_ns']:9.1f} {row['op_span_isolated_ns']:9.1f} "
              f"{row['op_span_back_to_back_ns']:9.1f} "
              f"{row['b2b_minus_kernel_ns']:9.1f} {row['b2b_over_kernel']:9.3f}")


if __name__ == "__main__":
    main()
