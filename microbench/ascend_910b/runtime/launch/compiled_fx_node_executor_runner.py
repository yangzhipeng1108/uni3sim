#!/usr/bin/env python3
"""Measure the per-node cost of executing a compiled FX graph, not a bare op.

``src/serving/runtime_prepare.py`` charges every eager-step node one
``compiled_fx_*_node_dispatch`` service, and the mapping table
``_ASCEND_910B_DENSE_NODE_CLASSES`` names the object: one dispatch of one
*compiled-FX graph node*.  Two existing artifacts each measure half of it.

``aclnn_eager_dispatch_20260926_r2``/``_r3`` calls the real operators
straight from Python with no graph executor walking nodes, so it measures
the operator submit alone and is a lower bound.
``compiled_fx_executor_20260916_r2`` walks an executor over nodes but hangs
a synthetic opaque generic operator on each one, so it measures the walk
with the wrong operator underneath; its own summary gates it
``diagnostic_only_until a source-static compiled-FX node descriptor is
added``.  That descriptor now exists -- it is the mapping table above.

This runner measures the composite the table actually charges: the same six
real node classes, in the same census proportion, executed *through a graph
executor*, priced per node.  The contrast against the identical operator
sequence issued directly from Python isolates what the executor adds per
node, which is the half the aclnn runner cannot see.

Execution modes, all over one identical traced graph:

  direct       the module's own Python forward (the aclnn-runner control)
  graph_call   the traced GraphModule's generated forward
  interpreter  torch.fx.Interpreter walking the nodes one at a time
  dynamo       torch.compile(backend="eager"), the real Dynamo entry

vLLM-Ascend reaches this path with ``config.debug.run_eagerly = True``
(``vllm_ascend/compilation/compiler_interface.py``), so an above-capture-limit
step runs a compiled callable rather than replaying a captured graph.

No synchronization occurs inside a timed region, so each measurement is Host
submit time.  No Serving workload, model or production duration is read.
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
TRIALS = 20
WARMUP_ROUNDS = 10
HIDDEN = 4096
HEADS = 32
HEAD_DIM = 128
TOKENS = 768
# Decoder layers issued per measured graph.  The census the mapping table
# pins is per step across all layers; one layer is the repeating unit, so
# the per-node price is layer-count independent and this only sets how many
# nodes each timed region contains.
LAYERS = 12


class DenseLayer(torch.nn.Module):
    """One dense decoder layer built from the six charged node classes.

    The node classes and their per-layer multiplicity follow
    ``_ASCEND_910B_DENSE_NODE_CLASSES``: norm nodes for the RmsNorm and
    AddRmsNorm members, linear nodes for qkv/o/gate_up/down, one rope, one
    fused-infer-attention, one reshape-and-cache, and the copy-shape nodes
    the table charges alongside them.
    """

    def __init__(self, device: str) -> None:
        super().__init__()
        opts = {"dtype": torch.float16, "device": device}
        self.gain_in = torch.nn.Parameter(torch.randn(HIDDEN, **opts))
        self.gain_post = torch.nn.Parameter(torch.randn(HIDDEN, **opts))
        self.qkv = torch.nn.Parameter(torch.randn(HIDDEN, HIDDEN, **opts))
        self.out = torch.nn.Parameter(torch.randn(HIDDEN, HIDDEN, **opts))
        self.gate_up = torch.nn.Parameter(torch.randn(HIDDEN, HIDDEN, **opts))
        self.down = torch.nn.Parameter(torch.randn(HIDDEN, HIDDEN, **opts))

    def forward(self, hidden, cosine, sine, cache, slots):
        tokens = hidden.shape[0]
        normed = torch_npu.npu_rms_norm(hidden, self.gain_in, 1e-6)[0]
        fused = torch.nn.functional.linear(normed, self.qkv)
        rotary = fused.view(1, tokens, HEADS, HEAD_DIM)
        rotated = torch_npu.npu_rotary_mul(rotary, cosine, sine)
        query = rotated.view(1, tokens, HEADS, HEAD_DIM).transpose(1, 2)
        key = query.clone()
        value = query.clone()
        entries = key.reshape(tokens * HEADS // 8, 8, HEAD_DIM)[:tokens]
        torch_npu.npu_scatter_nd_update_(cache, slots, entries)
        attended = torch_npu.npu_fused_infer_attention_score(
            query, key, value, num_heads=HEADS, input_layout="BNSD",
            scale=HEAD_DIM ** -0.5)[0]
        merged = attended.transpose(1, 2).reshape(tokens, HIDDEN)
        projected = torch.nn.functional.linear(merged, self.out)
        residual = hidden + projected
        post = torch_npu.npu_rms_norm(residual, self.gain_post, 1e-6)[0]
        gated = torch.nn.functional.linear(post, self.gate_up)
        activated = torch.nn.functional.silu(gated)
        return residual + torch.nn.functional.linear(activated, self.down)


class DenseStack(torch.nn.Module):
    def __init__(self, device: str, layers: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            DenseLayer(device) for _ in range(layers))

    def forward(self, hidden, cosine, sine, cache, slots):
        for layer in self.layers:
            hidden = layer(hidden, cosine, sine, cache, slots)
        return hidden


def _operands(device: str, tokens: int = TOKENS) -> tuple:
    opts = {"dtype": torch.float16, "device": device}
    return (
        torch.randn(tokens, HIDDEN, **opts),
        torch.randn(1, tokens, 1, HEAD_DIM, **opts),
        torch.randn(1, tokens, 1, HEAD_DIM, **opts),
        torch.zeros(8192, 8, HEAD_DIM, **opts),
        torch.arange(tokens, dtype=torch.int32, device=device).view(-1, 1),
    )


def _time(callable_under_test, operands, node_count: int) -> dict:
    for _ in range(WARMUP_ROUNDS):
        callable_under_test(*operands)
    torch.npu.synchronize()
    samples = []
    for _ in range(TRIALS):
        torch.npu.synchronize()
        start = time.perf_counter_ns()
        callable_under_test(*operands)
        samples.append(time.perf_counter_ns() - start)
        torch.npu.synchronize()
    median = statistics.median(samples)
    return {
        "node_count": node_count,
        "median_ns": median,
        "mean_ns": statistics.fmean(samples),
        "stdev_ns": statistics.stdev(samples),
        "per_node_us": median / node_count / 1000,
        "sample_count": len(samples),
    }


def _node_count(graph_module) -> int:
    return sum(
        1 for node in graph_module.graph.nodes
        if node.op in ("call_function", "call_method", "call_module"))


def _shape_recompile_control(module, device: str, nodes: int,
                             rounds: int = 12) -> dict:
    """Price a compiled callable entered at a token count it has not seen.

    Every mode above is entered at one fixed token count, so after warm-up
    the compiled artifact is always a cache hit.  A step that falls out of
    graph capture is not: it runs at its own unpadded token count, which
    differs from step to step, so each entry presents the compiled frame a
    shape its guards were not built for.

    Both arms issue the identical graph for the same number of rounds and
    differ only in whether the token count is reused or fresh, so the
    difference is whatever entering at an unseen shape costs.  All operands
    are allocated before timing starts.  ``dynamic=False`` is measured
    because it is the arm that must recompile; ``dynamic=True`` is measured
    beside it because a symbolic batch dimension is the configuration that
    would avoid the recompile, and the contrast says which one production
    behaves like.
    """
    counts = [512 + 64 * index for index in range(rounds)]
    families = {count: _operands(device, count) for count in counts}
    result = {"rounds": rounds, "token_counts": counts, "arms": {}}
    for label, dynamic in (("static", False), ("symbolic", True)):
        compiled = torch.compile(module, backend="eager", dynamic=dynamic)
        # Warm the fixed shape so the repeated arm is a pure cache hit.
        for _ in range(3):
            compiled(*families[counts[0]])
        torch.npu.synchronize()
        arm = {}
        for mode, pick in (("repeated", lambda index: counts[0]),
                           ("fresh", lambda index: counts[index])):
            samples = []
            for index in range(rounds):
                operands = families[pick(index)]
                torch.npu.synchronize()
                start = time.perf_counter_ns()
                compiled(*operands)
                samples.append(time.perf_counter_ns() - start)
                torch.npu.synchronize()
            arm[mode] = {
                "median_ns": statistics.median(samples),
                "first_ns": samples[0],
                "max_ns": max(samples),
                "per_node_us": statistics.median(samples) / nodes / 1000,
                "sample_count": len(samples),
            }
        arm["fresh_minus_repeated_us_per_node"] = (
            arm["fresh"]["per_node_us"] - arm["repeated"]["per_node_us"])
        result["arms"][label] = arm
    return result


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
    torch.npu.manual_seed(0)

    module = DenseStack(device, LAYERS).eval()
    operands = _operands(device)

    traced = torch.fx.symbolic_trace(module)
    nodes = _node_count(traced)
    print(f"traced graph: {nodes} call nodes over {LAYERS} layers "
          f"({nodes / LAYERS:.1f} per layer)")

    modes = {}
    with torch.no_grad():
        modes["direct"] = _time(module, operands, nodes)
        modes["graph_call"] = _time(traced, operands, nodes)
        interpreter = torch.fx.Interpreter(traced)
        modes["interpreter"] = _time(
            lambda *args: interpreter.run(*args), operands, nodes)
        try:
            compiled = torch.compile(module, backend="eager", dynamic=False)
            modes["dynamo"] = _time(compiled, operands, nodes)
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            modes["dynamo"] = {"unavailable": repr(error)}

    for name, entry in modes.items():
        if "unavailable" in entry:
            print(f"{name:14s} unavailable: {entry['unavailable'][:90]}")
            continue
        print(f"{name:14s} {entry['median_ns'] / 1e6:9.3f} ms "
              f"{entry['per_node_us']:8.3f} us/node")

    priced = {
        name: entry["per_node_us"]
        for name, entry in modes.items() if "per_node_us" in entry}
    excess = {
        name: value - priced["direct"]
        for name, value in priced.items() if name != "direct"}
    for name, value in excess.items():
        print(f"  {name} minus direct: {value:+8.3f} us/node")

    with torch.no_grad():
        recompile = _shape_recompile_control(module, device, nodes)
    print("\nshape recompile control (fresh token count per entry):")
    for label, arm in recompile["arms"].items():
        print(f"  {label:9s} repeated {arm['repeated']['per_node_us']:8.3f} "
              f"fresh {arm['fresh']['per_node_us']:8.3f} us/node  "
              f"(delta {arm['fresh_minus_repeated_us_per_node']:+8.3f}; "
              f"fresh first {arm['fresh']['first_ns'] / 1e6:8.3f} ms, "
              f"max {arm['fresh']['max_ns'] / 1e6:8.3f} ms)")

    arguments.output.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": (
            "per-node Host cost of executing a compiled FX graph of the six "
            "charged dense node classes, contrasted against the identical "
            "operator sequence issued directly from Python"),
        "forward_eligibility": "forward_eligible",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "container": arguments.container_name,
        "device_id": device_id,
        "host": platform.node(),
        "torch_version": torch.__version__,
        "framework": {"name": "torch_npu", "version": torch_npu.__version__},
        "decoder_shape": {
            "hidden": HIDDEN, "heads": HEADS, "head_dim": HEAD_DIM,
            "tokens": TOKENS, "layers": LAYERS},
        "traced_call_node_count": nodes,
        "trials": TRIALS,
        "modes": modes,
        "executor_excess_us_per_node": excess,
        "shape_recompile_control": recompile,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
