#!/usr/bin/env python3
"""Measure the Host submit cost of one real aclnn operator dispatch.

The Ascend serving Host profile charges every eager-step kernel call one
``compiled_fx_*_node_dispatch`` service.  The value production consumed for
those six mechanisms came from ``compiled_fx_executor_20260916_r2``, whose
own summary declares ``forward_eligibility: diagnostic_only_until ...`` and
whose ``mechanism`` is a *synthetic opaque generic backend node* -- not an
aclnn dispatch.  This runner measures the mechanism that is actually
charged: a real aclnn operator submitted from the Host through the same
torch_npu build the engine runs, one operator per FX node, which is what
``fallback_to_eager`` -> ``fx_run_eagerly`` executes on a step whose batch
exceeds the capture limit (``runtime_cudagraph_mode`` NONE).

Protocol.  Submit N operators back to back with no synchronization inside
the timed interval, sweep N, and take the slope; the intercept absorbs the
one-time call overhead so the slope is the marginal Host cost of one more
node.  Each operator class is measured separately because the census charges
six distinct classes.

Back-pressure control.  A Host-submit measurement is only valid while the
stream queue never fills, otherwise the loop would time Device service
instead.  The control re-measures one class at three shapes spanning ~1000x
of Device FLOPs per task: if the slope is invariant the Host is the
bottleneck, and if the queue were back-pressuring the slope would track
Device time.  The control is emitted into the summary so the reader can
check that invariance rather than assume it.

No Serving workload, model or production duration is read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import time

import torch
import torch_npu

SCHEMA_VERSION = 1
HARDWARE_ID = "ascend_910b"
CHAIN_LENGTHS = (1, 8, 64, 256, 512)
TRIALS = 24
WARMUP_CALLS = 200
# Dense decoder dimensions of the evaluated 910B models (hidden 4096,
# 32 query heads of 128), so each class is measured at the shape family it
# is charged for rather than at an arbitrary one.
HIDDEN = 4096
HEADS = 32
HEAD_DIM = 128
TOKENS = 256
CONTROL_SHAPES = ((256, 512), (1024, 2048), (4096, 4096))
# Occurrence counts of each dispatch class in the canonical qps64 prefill
# kernel census, used only to draw the distinct-op sequence in the same
# proportion the engine dispatches it.  No timing is read from the census.
CENSUS = {
    "compiled_fx_aten_linear_node_dispatch": 145,
    "compiled_fx_torch_npu_norm_node_dispatch": 181,
    "compiled_fx_aten_copy_node_dispatch": 217,
    "compiled_fx_rope_node_dispatch": 36,
    "compiled_fx_fia_node_dispatch": 36,
    "compiled_fx_reshape_cache_node_dispatch": 36,
}


def _fit(by_n: dict[int, int]) -> tuple[float, float]:
    xs = list(by_n)
    ys = [by_n[n] for n in xs]
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    slope = (
        sum((a - mean_x) * (b - mean_y) for a, b in zip(xs, ys))
        / sum((a - mean_x) ** 2 for a in xs))
    return slope, mean_y - slope * mean_x


def _measure(call) -> dict:
    for _ in range(WARMUP_CALLS):
        call()
    torch.npu.synchronize()
    by_n: dict[int, int] = {}
    for length in CHAIN_LENGTHS:
        samples = []
        for _ in range(TRIALS):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            for _ in range(length):
                call()
            end = time.perf_counter_ns()
            torch.npu.synchronize()
            samples.append(end - start)
        by_n[length] = int(statistics.median(samples))
    slope, intercept = _fit(by_n)
    return {
        "wall_ns_by_chain_length": {str(k): v for k, v in by_n.items()},
        "per_node_us_by_chain_length": {
            str(n): round(v / n / 1000, 4) for n, v in by_n.items()},
        "slope_ns_per_node": slope,
        "intercept_ns": intercept,
    }


def _classes(device: str) -> dict:
    activation = torch.randn(
        TOKENS, HIDDEN, dtype=torch.float16, device=device)
    weight = torch.randn(HIDDEN, HIDDEN, dtype=torch.float16, device=device)
    gain = torch.randn(HIDDEN, dtype=torch.float16, device=device)
    destination = torch.empty_like(activation)
    query = torch.randn(
        1, HEADS, TOKENS, HEAD_DIM, dtype=torch.float16, device=device)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    rotary = torch.randn(
        1, TOKENS, HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    cosine = torch.randn(
        1, TOKENS, 1, HEAD_DIM, dtype=torch.float16, device=device)
    sine = torch.randn_like(cosine)
    entries = torch.randn(
        1024, 8, HEAD_DIM, dtype=torch.float16, device=device)
    cache = torch.zeros(4096, 8, HEAD_DIM, dtype=torch.float16, device=device)
    slots = torch.arange(1024, dtype=torch.int32, device=device).view(-1, 1)
    return {
        "compiled_fx_aten_linear_node_dispatch":
            lambda: torch.nn.functional.linear(activation, weight),
        "compiled_fx_torch_npu_norm_node_dispatch":
            lambda: torch_npu.npu_rms_norm(activation, gain, 1e-6),
        "compiled_fx_aten_copy_node_dispatch":
            lambda: destination.copy_(activation),
        "compiled_fx_fia_node_dispatch":
            lambda: torch_npu.npu_fused_infer_attention_score(
                query, key, value, num_heads=HEADS, input_layout="BNSD",
                scale=HEAD_DIM ** -0.5),
        "compiled_fx_rope_node_dispatch":
            lambda: torch_npu.npu_rotary_mul(rotary, cosine, sine),
        "compiled_fx_reshape_cache_node_dispatch":
            lambda: torch_npu.npu_scatter_nd_update_(cache, slots, entries),
    }


def _control(device: str) -> dict:
    control = {}
    for rows, width in CONTROL_SHAPES:
        activation = torch.randn(
            rows, width, dtype=torch.float16, device=device)
        weight = torch.randn(width, width, dtype=torch.float16, device=device)
        entry = _measure(
            lambda: torch.nn.functional.linear(activation, weight))
        entry["device_flop_per_task"] = 2 * rows * width * width
        control[f"linear_{rows}x{width}"] = entry
    return control


def _sequence_control(calls: dict) -> dict:
    """Contrast a repeated single op against a sequence of distinct ops.

    Sweeping one op N times leaves the aclnn tiling and executor caches hot
    for that one op, which is the best case; an eager step instead walks a
    sequence of *different* ops.  If a distinct-op sequence costs more per
    node than the census-weighted mean of the per-class slopes, the repeated
    sweep understates the mechanism and the difference is what a step pays
    for not repeating itself.  If the two agree, the repeated sweep stands.

    The sequence is drawn round-robin in census proportion, so it differs
    from the per-class sweeps only in the *order* the same ops are issued.
    """
    order = []
    for mechanism, count in CENSUS.items():
        order.extend([mechanism] * count)
    order.sort(key=lambda m: -CENSUS[m])
    interleaved = []
    remaining = dict(CENSUS)
    while any(remaining.values()):
        for mechanism in CENSUS:
            if remaining[mechanism] > 0:
                interleaved.append(mechanism)
                remaining[mechanism] -= 1
    cycle = interleaved

    def step(index: int = [0]) -> None:
        calls[cycle[index[0] % len(cycle)]]()
        index[0] += 1

    return {
        "sequence_length": len(cycle),
        "census": dict(CENSUS),
        "measurement": _measure(step),
    }


def _per_class_in_sequence(calls: dict, rounds: int = 40) -> dict:
    """Price each class as it is dispatched inside a realistic step sequence.

    The per-class sweeps repeat one op, which leaves that op's tiling and
    executor caches hot and therefore measures the best case; the sequence
    control shows a distinct-op sequence costs substantially more per node.
    Neither number is the one production charges: a step walks a *sequence of
    distinct ops* and pays, for each one, whatever that op costs in that
    context.

    So this measures exactly that.  One pass issues every class in census
    proportion, round-robin so classes interleave the way a decoder layer
    interleaves them, and each individual dispatch is timed.  There is still
    no synchronization inside the timed region, so each sample is the Host
    submit of that op with the caches in the state the preceding distinct op
    left them.  Per-class medians over many passes give the price of one
    dispatch of that class under production-representative conditions.
    """
    order = []
    remaining = dict(CENSUS)
    while any(remaining.values()):
        for mechanism in CENSUS:
            if remaining[mechanism] > 0:
                order.append(mechanism)
                remaining[mechanism] -= 1

    for mechanism in order:
        calls[mechanism]()
    torch.npu.synchronize()

    samples: dict[str, list[int]] = {m: [] for m in CENSUS}
    for _ in range(rounds):
        torch.npu.synchronize()
        for mechanism in order:
            start = time.perf_counter_ns()
            calls[mechanism]()
            samples[mechanism].append(time.perf_counter_ns() - start)
        torch.npu.synchronize()

    return {
        "rounds": rounds,
        "sequence_length": len(order),
        "census": dict(CENSUS),
        "classes": {
            mechanism: {
                "median_ns": statistics.median(values),
                "mean_ns": statistics.fmean(values),
                "sample_count": len(values),
            }
            for mechanism, values in samples.items()},
    }


def _shape_family(device: str, tokens: int) -> dict:
    """Build the six class calls at one token count.

    Every allocation happens here, outside any timed region, so switching
    token counts during the sweep costs nothing but the dispatch itself.
    """
    activation = torch.randn(
        tokens, HIDDEN, dtype=torch.float16, device=device)
    weight = torch.randn(HIDDEN, HIDDEN, dtype=torch.float16, device=device)
    gain = torch.randn(HIDDEN, dtype=torch.float16, device=device)
    destination = torch.empty_like(activation)
    query = torch.randn(
        1, HEADS, tokens, HEAD_DIM, dtype=torch.float16, device=device)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    rotary = torch.randn(
        1, tokens, HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    cosine = torch.randn(
        1, tokens, 1, HEAD_DIM, dtype=torch.float16, device=device)
    sine = torch.randn_like(cosine)
    entries = torch.randn(
        tokens, 8, HEAD_DIM, dtype=torch.float16, device=device)
    cache = torch.zeros(4096, 8, HEAD_DIM, dtype=torch.float16, device=device)
    slots = torch.arange(tokens, dtype=torch.int32, device=device).view(-1, 1)
    return {
        "compiled_fx_aten_linear_node_dispatch":
            lambda: torch.nn.functional.linear(activation, weight),
        "compiled_fx_torch_npu_norm_node_dispatch":
            lambda: torch_npu.npu_rms_norm(activation, gain, 1e-6),
        "compiled_fx_aten_copy_node_dispatch":
            lambda: destination.copy_(activation),
        "compiled_fx_fia_node_dispatch":
            lambda: torch_npu.npu_fused_infer_attention_score(
                query, key, value, num_heads=HEADS, input_layout="BNSD",
                scale=HEAD_DIM ** -0.5),
        "compiled_fx_rope_node_dispatch":
            lambda: torch_npu.npu_rotary_mul(rotary, cosine, sine),
        "compiled_fx_reshape_cache_node_dispatch":
            lambda: torch_npu.npu_scatter_nd_update_(cache, slots, entries),
    }


def _shape_residency_control(device: str, rounds: int = 24) -> dict:
    """Contrast a repeated token count against a fresh one every round.

    Every measurement above holds the operand shapes fixed, so each class is
    dispatched at a shape whose tiling and executor selection were resolved
    on an earlier call and cached.  A step that falls out of the captured
    graph does not get that: it runs at its own unpadded token count, which
    differs from step to step, so every dispatch presents a shape the caches
    have not seen.

    Both modes issue the identical census-proportion sequence for the same
    number of rounds and differ only in whether the token count is reused or
    fresh, so the difference is the Host cost of resolving a new shape and
    nothing else.  All operands for all token counts are allocated before
    timing starts.
    """
    tokens = [512 + 37 * index for index in range(rounds)]
    families = {count: _shape_family(device, count) for count in tokens}
    fixed_family = families[tokens[0]]

    order = []
    remaining = dict(CENSUS)
    while any(remaining.values()):
        for mechanism in CENSUS:
            if remaining[mechanism] > 0:
                order.append(mechanism)
                remaining[mechanism] -= 1

    # Warm every shape once so the contrast is between a *second* use of a
    # shape and a second use of a different shape, not between warm and the
    # first touch of a freshly allocated buffer.
    for family in families.values():
        for mechanism in order:
            family[mechanism]()
    torch.npu.synchronize()

    def sweep(pick) -> dict:
        samples: dict[str, list[int]] = {m: [] for m in CENSUS}
        total = 0
        for index in range(rounds):
            family = pick(index)
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            for mechanism in order:
                sample = time.perf_counter_ns()
                family[mechanism]()
                samples[mechanism].append(time.perf_counter_ns() - sample)
            total += time.perf_counter_ns() - start
            torch.npu.synchronize()
        return {
            "wall_ns_per_dispatch": total / (rounds * len(order)),
            "classes": {
                mechanism: {
                    "median_ns": statistics.median(values),
                    "sample_count": len(values),
                }
                for mechanism, values in samples.items()},
        }

    repeated = sweep(lambda index: fixed_family)
    fresh = sweep(lambda index: families[tokens[index]])
    return {
        "rounds": rounds,
        "sequence_length": len(order),
        "token_counts": tokens,
        "repeated_shape": repeated,
        "fresh_shape_each_round": fresh,
        "fresh_minus_repeated_ns_per_dispatch": (
            fresh["wall_ns_per_dispatch"] - repeated["wall_ns_per_dispatch"]),
    }


def _device_id() -> int:
    text = os.environ.get("ASCEND_DEVICE_ID")
    if text is None:
        raise SystemExit(
            "ASCEND_DEVICE_ID must explicitly select an idle NPU")
    return int(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if not arguments.container_name.strip():
        parser.error("--container-name must be nonempty")
    device_id = _device_id()
    device = f"npu:{device_id}"
    torch.npu.set_device(device)

    classes = {}
    for mechanism, call in _classes(device).items():
        classes[mechanism] = _measure(call)
        print(f"{mechanism:46s} "
              f"{classes[mechanism]['slope_ns_per_node'] / 1000:8.3f} us/node")
    class_calls = _classes(device)
    sequence = _sequence_control(class_calls)
    weighted = sum(
        CENSUS[m] * classes[m]["slope_ns_per_node"] for m in CENSUS) / sum(
            CENSUS.values())
    observed = sequence["measurement"]["slope_ns_per_node"]
    sequence["census_weighted_repeated_slope_ns_per_node"] = weighted
    sequence["distinct_sequence_excess_ns_per_node"] = observed - weighted
    print(f"sequence control: census-weighted repeated "
          f"{weighted / 1000:.3f} us/node vs distinct sequence "
          f"{observed / 1000:.3f} us/node "
          f"(excess {(observed - weighted) / 1000:+.3f})")
    in_sequence = _per_class_in_sequence(class_calls)
    print("\nper-class price inside the census-proportion distinct sequence:")
    for mechanism, entry in in_sequence["classes"].items():
        print(f"  {mechanism:48s} {entry['median_ns'] / 1000:8.3f} us "
              f"(n={entry['sample_count']})")
    control = _control(device)
    for name, entry in control.items():
        print(f"control {name:22s} "
              f"{entry['device_flop_per_task'] / 1e6:10.1f} MFLOP/task "
              f"{entry['slope_ns_per_node'] / 1000:8.3f} us/node")
    residency = _shape_residency_control(device)
    print("\nshape residency control (same sequence, same rounds):")
    print(f"  repeated token count      "
          f"{residency['repeated_shape']['wall_ns_per_dispatch'] / 1000:8.3f}"
          f" us/dispatch")
    print(f"  fresh token count / round "
          f"{residency['fresh_shape_each_round']['wall_ns_per_dispatch'] / 1000:8.3f}"
          f" us/dispatch")
    print(f"  fresh - repeated          "
          f"{residency['fresh_minus_repeated_ns_per_dispatch'] / 1000:+8.3f}"
          f" us/dispatch")
    for mechanism in CENSUS:
        rep = residency["repeated_shape"]["classes"][mechanism]["median_ns"]
        new = residency[
            "fresh_shape_each_round"]["classes"][mechanism]["median_ns"]
        print(f"    {mechanism:48s} {rep / 1000:8.3f} -> {new / 1000:8.3f} us")

    arguments.output.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": (
            "real aclnn operator Host submit, one operator per FX node, no "
            "synchronization inside the timed interval"),
        "forward_eligibility": "forward_eligible",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "container": arguments.container_name,
        "device_id": device_id,
        "host": platform.node(),
        "torch_version": torch.__version__,
        "framework": {"name": "torch_npu", "version": torch_npu.__version__},
        "trials_per_point": TRIALS,
        "warmup_calls": WARMUP_CALLS,
        "chain_lengths": list(CHAIN_LENGTHS),
        "decoder_shape": {
            "hidden": HIDDEN, "heads": HEADS, "head_dim": HEAD_DIM,
            "tokens": TOKENS},
        "classes": classes,
        "back_pressure_control": control,
        "distinct_sequence_control": sequence,
        "per_class_in_sequence": in_sequence,
        "shape_residency_control": residency,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
