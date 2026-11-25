#!/usr/bin/env python3
"""Price entering the torchair npugraph_ex backend, the real production path.

``eager_step_backend_identification_runner.py`` established on card that a
910B engine built the way the canonical capture path builds it resolves to
``CompilationMode.VLLM_COMPILE`` with vLLM-Ascend's OOT ``AscendCompiler``,
which takes ``npugraph_ex_compile``: torchair's NPU backend with
``config.mode = "reduce-overhead"`` and ``config.debug.run_eagerly = True``.
torchair is importable because it ships inside torch_npu.

Every previous synthetic microbenchmark in this repository entered the
compiled callable through ``torch.compile(backend="eager")`` -- plain Dynamo,
a different backend -- so none of them measured the path production runs.
This runner uses the same torchair CompilerConfig the engine builds.

A step inside the capture limit replays a captured graph.  A step above it
(``graph_fallback_reason == "above_capture_limit"``) runs at its own unpadded
token count, which differs step to step, so it enters the compiled callable
at a shape the backend has not been entered at before.  The two arms differ
only in that: one re-enters at a fixed token count, the other at a fresh one
each round.  The plain-Dynamo arm is measured beside them to quantify how far
the backend all earlier artifacts used sits from the production backend.

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
HIDDEN = 4096
HEADS = 32
HEAD_DIM = 128
LAYERS = 12
ROUNDS = 12
FIXED_TOKENS = 512


class DenseLayer(torch.nn.Module):
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


def _operands(device: str, tokens: int) -> tuple:
    opts = {"dtype": torch.float16, "device": device}
    return (
        torch.randn(tokens, HIDDEN, **opts),
        torch.randn(1, tokens, 1, HEAD_DIM, **opts),
        torch.randn(1, tokens, 1, HEAD_DIM, **opts),
        torch.zeros(8192, 8, HEAD_DIM, **opts),
        torch.arange(tokens, dtype=torch.int32, device=device).view(-1, 1),
    )


def _npugraph_ex_backend():
    """Build the backend exactly as vllm_ascend.npugraph_ex_compile does."""
    import torch_npu.dynamo.torchair as torchair

    torch.npu.set_compile_mode(jit_compile=False)
    config = torchair.CompilerConfig()
    config.mode = "reduce-overhead"
    config.debug.run_eagerly = True
    config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
    return torchair.get_npu_backend(compiler_config=config)


def _sweep(compiled, families, counts, mode: str) -> dict:
    """Time one entry per round; no synchronization inside the timed region."""
    samples = []
    for index in range(ROUNDS):
        count = counts[0] if mode == "fixed" else counts[index]
        torch.npu.synchronize()
        start = time.perf_counter_ns()
        compiled(*families[count])
        samples.append(time.perf_counter_ns() - start)
        torch.npu.synchronize()
    return {
        "mode": mode,
        "median_ms": statistics.median(samples) / 1e6,
        "first_ms": samples[0] / 1e6,
        "max_ms": max(samples) / 1e6,
        "min_ms": min(samples) / 1e6,
        "all_ms": [round(s / 1e6, 3) for s in samples],
        "sample_count": len(samples),
    }


def _arm(module, backend, label: str, device: str, counts, families,
         dynamic: bool = False) -> dict:
    compiled = torch.compile(module, backend=backend, dynamic=dynamic)
    # Warm the fixed shape so the fixed arm is a genuine re-entry.
    for _ in range(3):
        compiled(*families[counts[0]])
    torch.npu.synchronize()
    fixed = _sweep(compiled, families, counts, "fixed")
    fresh = _sweep(compiled, families, counts, "fresh")
    return {
        "fixed_shape": fixed,
        "fresh_shape": fresh,
        "fresh_minus_fixed_ms": fresh["median_ms"] - fixed["median_ms"],
        "label": label,
    }


def _device_id() -> int:
    text = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if text is None:
        raise SystemExit("ASCEND_RT_VISIBLE_DEVICES must select an idle NPU")
    return int(text.split(",")[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if not arguments.container_name.strip():
        parser.error("--container-name must be nonempty")
    _device_id()
    device = "npu:0"
    torch.npu.set_device(device)
    torch.npu.manual_seed(0)

    module = DenseStack(device, LAYERS).eval()
    counts = [FIXED_TOKENS + 64 * index for index in range(ROUNDS)]
    families = {count: _operands(device, count) for count in counts}
    traced = torch.fx.symbolic_trace(module)
    nodes = sum(
        1 for node in traced.graph.nodes
        if node.op in ("call_function", "call_method", "call_module"))
    print(f"traced graph: {nodes} call nodes over {LAYERS} layers")

    arms = {}
    with torch.no_grad():
        arms["npugraph_ex"] = _arm(
            module, _npugraph_ex_backend(), "torchair npugraph_ex "
            "(reduce-overhead, run_eagerly) -- production backend",
            device, counts, families)
        arms["npugraph_ex_dynamic"] = _arm(
            module, _npugraph_ex_backend(), "torchair npugraph_ex with a "
            "symbolic batch dimension -- the analogue of vLLM's single "
            "compile range (1, 8192)", device, counts, families, dynamic=True)
        arms["plain_dynamo_eager"] = _arm(
            module, "eager", "torch.compile(backend='eager') -- the backend "
            "every earlier artifact used", device, counts, families)

    for name, arm in arms.items():
        print(f"\n{name}  ({arm['label']})")
        for key in ("fixed_shape", "fresh_shape"):
            entry = arm[key]
            print(f"  {key:12s} median {entry['median_ms']:9.3f} ms  "
                  f"first {entry['first_ms']:9.3f}  max {entry['max_ms']:9.3f}"
                  f"  per-node {entry['median_ms'] * 1000 / nodes:7.2f} us")
        print(f"  fresh - fixed: {arm['fresh_minus_fixed_ms']:+9.3f} ms")

    arguments.output.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": (
            "Host cost of entering the torchair npugraph_ex compiled callable "
            "at a fixed versus a fresh token count, contrasted against plain "
            "Dynamo eager backend"),
        "forward_eligibility": "forward_eligible",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "container": arguments.container_name,
        "host": platform.node(),
        "torch_version": torch.__version__,
        "framework": {"name": "torch_npu", "version": torch_npu.__version__},
        "decoder_shape": {
            "hidden": HIDDEN, "heads": HEADS, "head_dim": HEAD_DIM,
            "layers": LAYERS},
        "traced_call_node_count": nodes,
        "rounds": ROUNDS,
        "token_counts": counts,
        "arms": arms,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
