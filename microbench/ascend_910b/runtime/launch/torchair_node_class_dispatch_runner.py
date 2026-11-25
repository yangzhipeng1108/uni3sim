#!/usr/bin/env python3
"""Measure torchair fallback per-node Host dispatch cost by op class, at the
real qwen3-8b tp1 launch domains.

The canonical Ascend serving engine compiles one unsplit FX graph
(enable_npugraph_ex, splitting_ops empty); steps above the capture limit run
``ACLGraphWrapper`` passthrough -> the codegen'd FX kernel, one Host dispatch
per graph node.  This runner measures that per-node Host submit for the op
classes of the dense qwen3-8b fallback graph, with vehicles that carry each
class's real dispatch machinery:

* ``aten_linear``      -- ``F.linear`` blocks over the real gate_up (24576) and
  down (12288) weight domains, one block = two aten Matmul nodes;
* ``aten_copy``        -- one ``copy_`` from a distinct seed into a distinct
  sink per node (the class of the TensorMove/Slice copies; distinct
  destinations defeat functional elimination and in-place folding);
* ``torch_npu_light``  -- ``torch_npu.npu_rms_norm`` (the class of the
  torch_npu pybind norm ops used by vllm-ascend);
* ``library_custom``   -- a ``torch.library.custom_op`` node (the machinery
  baseline of the vllm-registered custom-op class);
* ``fia_tnd``          -- the real ``torch_npu.npu_fused_infer_attention_score``
  prefill call, TND layout over a real paged KV shape;
* ``reshape_cache``    -- the real ``torch_npu._npu_reshape_and_cache`` over a
  real paged cache and slot mapping;
* ``rope_custom``      -- the real ``torch.ops.vllm.npu_rotary_embedding``
  custom op (its impl launches the triton rope kernel, as in the engine).

Each class is measured at every real M point both as a standalone eager call
and as nodes of a torchair fallback chain, with device synchronization after
every timed trial so queue backpressure cannot contaminate the Host submit
timing.  Chain fits use M=729 with shape checks at M=78 and M=2612, the real
prefill-M range of the qps64 case.  Values are Host submit times only; no
Serving workload, model, or production duration is consumed.

A second instrument, the mixed composite, mirrors the per-layer node census
of the real fallback graph (4 Matmul, 5 torch_npu norm/swiglu, 6 copy, 1
rope, 1 reshape-and-cache, 1 FIA per unit, 18 nodes) so allocation lifetimes
and dispatch interleaving match the engine lane: class-pure chains keep L
same-class outputs simultaneously live, which the real lane never does.
Count-differencing (one extra node of one class per unit) gives the in-mix
per-class marginal; the composite is a measurement instrument, not a fit --
no evaluation or trace value enters anywhere.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu
import torch_npu.dynamo.torchair as torchair

import vllm  # noqa: F401  (registers the torch.ops.vllm custom-op namespace)
import vllm_ascend.ops.register_custom_ops  # noqa: F401  (npu_rotary_embedding et al.)
from vllm_ascend.ops.triton.triton_utils import (  # noqa: F401
    init_device_properties_triton,
)


HARDWARE_ID = "ascend_910b"
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", HARDWARE_ID)

# Real qwen3-8b tp1 domains (configs/model/qwen3-8b.json).
HIDDEN = 4096
GATE_UP_OUT = 24576
DOWN_IN = 12288
QKV_OUT = 6144
VOCAB = 151936
DTYPE = torch.bfloat16
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 128
COS_SIN_MAX_POS = 4096
FIA_SCALE = HEAD_DIM ** -0.5

# Real prefill-M domain of the qps64 compact case (78 .. 2612).
M_DOMAIN = (78, 729, 2612)
FIT_M = 729
CHECK_MS = (78, 2612)
WARMUP_M = 257  # distinct from every measured M, trips the capture limit

# Chain lengths in class nodes (the linear class counts one gate_up+down
# block as one chain unit = two Matmul nodes).  r4 showed the dependent
# backpressure regime starts beyond L=8 for the torch_npu and FIA vehicles,
# so the dispatch-cost regime is fitted on L<=8 with L=64 as holdout.
CHAIN_LENGTHS = (1, 2, 4, 8, 64)
FIT_CHAIN_LENGTHS = frozenset((1, 2, 4, 8))
HOLDOUT_CHAIN_LENGTH = 64
TRIALS = 12
WARMUP_REPEATS = 4
CAPTURE_SIZE_LIMIT = 1
SCHEMA_VERSION = 3

# Mixed composite: the per-unit census mirrors the qps64 qwen3-8b prefill
# fallback graph census divided by 36 layers (144 Matmul, 181 torch_npu
# norm/add-rms-norm/swiglu, 217 TensorMove/Slice copy, 36 rope, 36
# reshape-and-cache, 36 FIA per forward).  One extra node of one class per
# unit enables count-differenced in-mix marginals.
COMPOSITE_CENSUS = {
    "aten_linear": 4,
    "torch_npu_light": 5,
    "aten_copy": 6,
    "rope_custom": 1,
    "reshape_cache": 1,
    "fia_tnd": 1,
}
COMPOSITE_BASE_NODES = sum(COMPOSITE_CENSUS.values())
COMPOSITE_EXTRA_CLASSES = tuple(COMPOSITE_CENSUS)
COMPOSITE_LAYER_LENGTHS = (4, 8)
COMPOSITE_LAYERS_PER_FORWARD = 36


@torch.library.custom_op(
    "midlayer_torchair_class_probe::functional_custom_node", mutates_args=())
def _functional_custom_node(source: torch.Tensor) -> torch.Tensor:
    """A vllm-custom-op-class node: library registration machinery + clone."""
    return source.clone()


@torch.library.register_fake(
    "midlayer_torchair_class_probe::functional_custom_node")
def _functional_custom_node_fake(source: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(source)


def _paged_blocks(m_tokens: int) -> int:
    return (m_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 1


class _LinearBlockChain(torch.nn.Module):
    """One block = one real gate_up Matmul (24576 out) plus the back
    projection that restores the (M, hidden) shape; the block slope divided
    by two gives the per-Matmul node cost in the gate_up domain."""

    def __init__(self, block_count: int) -> None:
        super().__init__()
        self.block_count = block_count
        self.register_buffer(
            "gate_up_weight", torch.randn(GATE_UP_OUT, HIDDEN, dtype=DTYPE))
        self.register_buffer(
            "back_weight", torch.randn(HIDDEN, GATE_UP_OUT, dtype=DTYPE))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        value = source
        for _ in range(self.block_count):
            value = torch.nn.functional.linear(value, self.gate_up_weight)
            value = torch.nn.functional.linear(value, self.back_weight)
        return value


class _LinearDomainCheckChain(torch.nn.Module):
    """Two-Matmul block over one further real weight domain pair."""

    def __init__(self, block_count: int, out_features: int) -> None:
        super().__init__()
        self.block_count = block_count
        self.register_buffer(
            "forward_weight",
            torch.randn(out_features, HIDDEN, dtype=DTYPE))
        self.register_buffer(
            "back_weight", torch.randn(HIDDEN, out_features, dtype=DTYPE))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        value = source
        for _ in range(self.block_count):
            value = torch.nn.functional.linear(value, self.forward_weight)
            value = torch.nn.functional.linear(value, self.back_weight)
        return value


LINEAR_DOMAIN_CHECKS = {
    # Real qkv (6144 out) and o-proj (square 4096) domains; the real down
    # (12288 in) and lm-head (151936 out) domains are covered by the eager
    # standalone measurements plus the wide check below.
    "qkv": QKV_OUT,
    "o_proj": HIDDEN,
    "lm_head": VOCAB,
}


class _AtenCopyRingChain(torch.nn.Module):
    """One node = one ``copy_`` from a distinct seed row into a distinct sink
    row; every node writes a distinct destination, so no functional or
    in-place pass can fold the chain, and the per-node cost is the copy-class
    dispatch cost directly (r4's clone+copy_ vehicle collapsed: a chain of
    clones feeding copies is semantically one copy)."""

    def __init__(self, node_count: int, m_tokens: int) -> None:
        super().__init__()
        self.node_count = node_count
        self.register_buffer(
            "seeds",
            torch.randn(node_count, m_tokens, HIDDEN, dtype=DTYPE))
        self.register_buffer(
            "sinks", torch.randn(node_count, m_tokens, HIDDEN, dtype=DTYPE))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        for index in range(self.node_count):
            self.sinks[index].copy_(self.seeds[index])
        return source


class _TorchNpuNormChain(torch.nn.Module):
    def __init__(self, node_count: int) -> None:
        super().__init__()
        self.node_count = node_count
        self.register_buffer("weight", torch.ones(HIDDEN, dtype=DTYPE))
        self.epsilon = 1e-6

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        value = source
        for _ in range(self.node_count):
            value = torch_npu.npu_rms_norm(value, self.weight, self.epsilon)[0]
        return value


class _LibraryCustomChain(torch.nn.Module):
    def __init__(self, node_count: int) -> None:
        super().__init__()
        self.node_count = node_count

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        value = source
        for _ in range(self.node_count):
            value = _functional_custom_node(value)
        return value


class _FiaTndChain(torch.nn.Module):
    """The real prefill FIA call over a real paged KV shape, TND layout,
    consuming its own output so every node stays observable."""

    def __init__(self, node_count: int, m_tokens: int) -> None:
        super().__init__()
        self.node_count = node_count
        self.m_tokens = m_tokens
        self.register_buffer(
            "query", torch.randn(m_tokens, NUM_HEADS, HEAD_DIM, dtype=DTYPE))
        num_blocks = _paged_blocks(m_tokens)
        self.register_buffer(
            "key_cache",
            torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS * HEAD_DIM,
                        dtype=DTYPE))
        self.register_buffer(
            "value_cache", self.key_cache.clone())
        self.register_buffer(
            "block_table",
            torch.zeros(1, num_blocks, dtype=torch.int32))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        query = self.query
        for _ in range(self.node_count):
            query = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=self.key_cache,
                value=self.value_cache,
                atten_mask=None,
                block_table=self.block_table,
                input_layout="TND",
                block_size=BLOCK_SIZE,
                actual_seq_lengths=[self.m_tokens],
                actual_seq_lengths_kv=[self.m_tokens],
                num_key_value_heads=NUM_KV_HEADS,
                num_heads=NUM_HEADS,
                scale=FIA_SCALE,
                sparse_mode=0,
            )[0]
        return query


class _ReshapeCacheChain(torch.nn.Module):
    """The real ``torch_npu._npu_reshape_and_cache`` over a real paged cache
    and slot mapping; the in-place cache write anchors every node."""

    def __init__(self, node_count: int, m_tokens: int) -> None:
        super().__init__()
        self.node_count = node_count
        self.register_buffer(
            "key", torch.randn(m_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=DTYPE))
        self.register_buffer("value", self.key.clone())
        num_blocks = _paged_blocks(m_tokens)
        self.register_buffer(
            "key_cache",
            torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                        dtype=DTYPE))
        self.register_buffer("value_cache", self.key_cache.clone())
        self.register_buffer(
            "slot_mapping",
            torch.arange(m_tokens, dtype=torch.int64)
            % (num_blocks * BLOCK_SIZE))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        for _ in range(self.node_count):
            torch_npu._npu_reshape_and_cache(
                self.key, self.value, self.key_cache, self.value_cache,
                self.slot_mapping)
        return source

class _RopeCustomChain(torch.nn.Module):
    """The real ``torch.ops.vllm.npu_rotary_embedding`` custom op; its
    registered impl launches the triton rope kernel exactly as in the
    engine, and the returned tensors chain into the next node."""

    def __init__(self, node_count: int, m_tokens: int) -> None:
        super().__init__()
        self.node_count = node_count
        self.head_size = HEAD_DIM
        self.rotary_dim = HEAD_DIM
        self.is_neox_style = True
        self.register_buffer(
            "query",
            torch.randn(m_tokens, NUM_HEADS * HEAD_DIM, dtype=DTYPE))
        self.register_buffer(
            "key",
            torch.randn(m_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=DTYPE))
        self.register_buffer(
            "positions", torch.arange(m_tokens, dtype=torch.int64))
        self.register_buffer(
            "cos_sin_cache",
            torch.randn(COS_SIN_MAX_POS, HEAD_DIM, dtype=DTYPE))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        query = self.query
        key = self.key
        for _ in range(self.node_count):
            query, key = torch.ops.vllm.npu_rotary_embedding(
                self.positions, query, key, self.cos_sin_cache,
                self.head_size, self.rotary_dim, self.is_neox_style)
        return query


class _MixedLayerChain(torch.nn.Module):
    """One unit mirrors the per-layer node census of the qwen3-8b prefill
    fallback graph with the real op vehicles and transient (per-unit)
    intermediate lifetimes, matching the engine lane's allocation pattern
    that class-pure chains cannot reproduce.  ``extra`` adds exactly one
    node of one class per unit for count-differenced in-mix marginals."""

    def __init__(self, layer_count: int, m_tokens: int, extra: str | None) -> None:
        super().__init__()
        self.layer_count = layer_count
        self.m_tokens = m_tokens
        if extra is not None and extra not in COMPOSITE_CENSUS:
            raise ValueError(f"unknown composite extra class: {extra}")
        self.extra = extra
        self.epsilon = 1e-6
        self.register_buffer(
            "norm_weight", torch.ones(HIDDEN, dtype=DTYPE))
        # The composite's qkv projection is square (hidden -> hidden) so its
        # output can chain straight into the rope query operand; the real qkv
        # width (6144) is covered by the Part A linear domain checks and the
        # host dispatch cost does not depend on the output width.
        self.register_buffer(
            "qkv_weight", torch.randn(HIDDEN, HIDDEN, dtype=DTYPE))
        self.register_buffer(
            "o_weight", torch.randn(HIDDEN, HIDDEN, dtype=DTYPE))
        self.register_buffer(
            "gate_up_weight", torch.randn(GATE_UP_OUT, HIDDEN, dtype=DTYPE))
        self.register_buffer(
            "down_weight", torch.randn(HIDDEN, DOWN_IN, dtype=DTYPE))
        # rope operands (the rope output feeds the FIA query so the rope
        # node stays live without extra consumers).
        self.register_buffer(
            "positions", torch.arange(m_tokens, dtype=torch.int64))
        self.register_buffer(
            "rope_key",
            torch.randn(m_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=DTYPE))
        self.register_buffer(
            "cos_sin_cache",
            torch.randn(COS_SIN_MAX_POS, HEAD_DIM, dtype=DTYPE))
        # paged KV cache and slot mapping for the reshape-and-cache node.
        num_blocks = _paged_blocks(m_tokens)
        self.register_buffer(
            "flat_key",
            torch.randn(m_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=DTYPE))
        self.register_buffer("flat_value", self.flat_key.clone())
        self.register_buffer(
            "key_cache",
            torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                        dtype=DTYPE))
        self.register_buffer("value_cache", self.key_cache.clone())
        self.register_buffer(
            "slot_mapping",
            torch.arange(m_tokens, dtype=torch.int64)
            % (num_blocks * BLOCK_SIZE))
        # FIA operands.
        self.register_buffer(
            "fia_key_cache",
            torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS * HEAD_DIM,
                        dtype=DTYPE))
        self.register_buffer("fia_value_cache", self.fia_key_cache.clone())
        self.register_buffer(
            "block_table", torch.zeros(1, num_blocks, dtype=torch.int32))
        # Fixed-M seed for the value chain: the composite ignores its input
        # argument entirely, which keeps every internal tensor at the fixed
        # M.  The input stays a pure shape-tripper: the warm-up call at
        # WARMUP_M compiles and captures graph key #1, the measured call at
        # M trips the capture-size limit and latches fallback_to_eager --
        # exactly the r4-verified buffer-class pattern.
        self.register_buffer(
            "seed_value", torch.randn(m_tokens, HIDDEN, dtype=DTYPE))
        # Copy ring: 6 distinct (seed, sink) pairs; slot 5 is reserved for
        # the aten_copy extra variant.  The attention output copy uses its
        # own destination because contiguous() on an already-contiguous
        # view is elided by dynamo, while copy_ is a real dispatch.
        self.register_buffer(
            "copy_seeds", torch.randn(6, m_tokens, HIDDEN, dtype=DTYPE))
        self.register_buffer(
            "copy_sinks", torch.randn(6, m_tokens, HIDDEN, dtype=DTYPE))
        self.register_buffer(
            "attention_out", torch.randn(m_tokens, HIDDEN, dtype=DTYPE))

    def _fia(self, query: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=self.fia_key_cache,
            value=self.fia_value_cache,
            atten_mask=None,
            block_table=self.block_table,
            input_layout="TND",
            block_size=BLOCK_SIZE,
            actual_seq_lengths=[self.m_tokens],
            actual_seq_lengths_kv=[self.m_tokens],
            num_key_value_heads=NUM_KV_HEADS,
            num_heads=NUM_HEADS,
            scale=FIA_SCALE,
            sparse_mode=0,
        )[0]

    def _unit(self, value: torch.Tensor) -> torch.Tensor:
        # 2 light (input + q norm stand-ins)
        value = torch_npu.npu_rms_norm(
            value, self.norm_weight, self.epsilon)[0]
        if self.extra == "torch_npu_light":
            value = torch_npu.npu_rms_norm(
                value, self.norm_weight, self.epsilon)[0]
        value = torch_npu.npu_rms_norm(
            value, self.norm_weight, self.epsilon)[0]
        # 1 linear (qkv)
        query = torch.nn.functional.linear(value, self.qkv_weight)
        # 1 copy (qkv contiguous)
        self.copy_sinks[0].copy_(self.copy_seeds[0])
        # 1 rope (on the qkv projection so the linear node stays live)
        rope_q, _ = torch.ops.vllm.npu_rotary_embedding(
            self.positions, query, self.rope_key, self.cos_sin_cache,
            HEAD_DIM, HEAD_DIM, True)
        if self.extra == "rope_custom":
            rope_q, _ = torch.ops.vllm.npu_rotary_embedding(
                self.positions, rope_q, self.rope_key, self.cos_sin_cache,
                HEAD_DIM, HEAD_DIM, True)
        # 1 FIA (rope output chained so the rope node stays live)
        attention = self._fia(
            rope_q.view(self.m_tokens, NUM_HEADS, HEAD_DIM))
        if self.extra == "fia_tnd":
            attention = self._fia(attention)
        # 1 copy (attention output)
        self.attention_out.copy_(
            attention.reshape(self.m_tokens, HIDDEN))
        # 1 linear (o-proj)
        o = torch.nn.functional.linear(self.attention_out, self.o_weight)
        # 1 light (post-attention norm)
        value = torch_npu.npu_rms_norm(o, self.norm_weight, self.epsilon)[0]
        # 1 linear (gate_up) + 1 light (swiglu) + 1 linear (down)
        gate_up = torch.nn.functional.linear(value, self.gate_up_weight)
        activated = torch_npu.npu_swiglu(gate_up)
        down = torch.nn.functional.linear(activated, self.down_weight)
        if self.extra == "aten_linear":
            down = torch.nn.functional.linear(down, self.o_weight)
        # 1 reshape-and-cache
        torch_npu._npu_reshape_and_cache(
            self.flat_key, self.flat_value, self.key_cache, self.value_cache,
            self.slot_mapping)
        if self.extra == "reshape_cache":
            torch_npu._npu_reshape_and_cache(
                self.flat_key, self.flat_value, self.key_cache,
                self.value_cache, self.slot_mapping)
        # 4 glue copies
        self.copy_sinks[1].copy_(self.copy_seeds[1])
        self.copy_sinks[2].copy_(self.copy_seeds[2])
        self.copy_sinks[3].copy_(self.copy_seeds[3])
        self.copy_sinks[4].copy_(self.copy_seeds[4])
        if self.extra == "aten_copy":
            self.copy_sinks[5].copy_(self.copy_seeds[5])
        # 1 light (final add-rms-norm stand-in)
        return torch_npu.npu_add_rms_norm(
            down, value, self.norm_weight, self.epsilon)[0]

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        # The input argument is a pure shape-tripper (see __init__); the
        # value chain runs entirely on fixed-M buffers.
        value = self.seed_value
        for _ in range(self.layer_count):
            value = self._unit(value)
        return value


CLASS_CHAIN_FACTORIES = {
    "aten_linear": (lambda count: _LinearBlockChain(count), 2),
    "torch_npu_light": (lambda count: _TorchNpuNormChain(count), 1),
    "library_custom": (lambda count: _LibraryCustomChain(count), 1),
}

# Buffer-based classes whose vehicles are fixed to one M per module instance.
BUFFER_CHAIN_CLASSES = {
    "aten_copy": (lambda count, m: _AtenCopyRingChain(count, m), 1),
    "fia_tnd": (_FiaTndChain, 1),
    "reshape_cache": (_ReshapeCacheChain, 1),
    "rope_custom": (_RopeCustomChain, 1),
}

EAGER_LABELS = {
    "aten_linear": "linear_gate_up",
    "aten_copy": "aten_copy_",
    "torch_npu_light": "torch_npu_rms_norm",
    "library_custom": "library_custom_clone",
    "fia_tnd": "fia_tnd",
    "reshape_cache": "reshape_cache",
    "rope_custom": "rope_custom",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--baseline-commit", type=str, required=True)
    arguments = parser.parse_args()
    if not arguments.container_name.strip():
        parser.error("--container-name must be nonempty")
    return arguments


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    owner_size = len(OUTPUT_OWNER_PARTS)
    offsets = tuple(
        index
        for index in range(len(resolved.parts) - owner_size + 1)
        if resolved.parts[index:index + owner_size] == OUTPUT_OWNER_PARTS
    )
    if len(offsets) != 1 or len(resolved.parts) <= offsets[0] + owner_size:
        raise ValueError("--output-dir must be a run below the Ascend owner")
    if resolved.exists():
        raise FileExistsError("--output-dir must not already exist")
    return resolved


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _append_json_line(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("median requires samples")
    return float(statistics.median(values))


def _linear_fit(points: list[tuple[int, float]]) -> tuple[float, float]:
    count = len(points)
    x_sum = sum(point[0] for point in points)
    y_sum = sum(point[1] for point in points)
    xx_sum = sum(point[0] * point[0] for point in points)
    xy_sum = sum(point[0] * point[1] for point in points)
    denominator = count * xx_sum - x_sum * x_sum
    if denominator <= 0:
        raise RuntimeError("class dispatch fit is singular")
    slope = (count * xy_sum - x_sum * y_sum) / denominator
    intercept = (y_sum - slope * x_sum) / count
    if not math.isfinite(intercept) or not math.isfinite(slope) or slope <= 0:
        raise RuntimeError("class dispatch fit is invalid")
    return float(intercept), float(slope)


def _compiled_graph(module: torch.nn.Module) -> torch.nn.Module:
    # Module-held tensors must live on the target device before tracing:
    # fake-tensor propagation fails on mixed npu/cpu module attributes.
    module = module.to(f"npu:{torch.npu.current_device()}").eval()
    torch.npu.set_compile_mode(jit_compile=False)
    config = torchair.CompilerConfig()
    config.mode = "reduce-overhead"
    # No debug.run_eagerly: that flag made npu_fx_compiler return the raw fx
    # GraphModule, so every "per-node" number it produced was FX-interpreter
    # overhead (~200us/node), not the codegen'd dispatch the engine pays.
    # r4/r5 measurements taken under that flag are invalidated; the real
    # executor vehicle is r8a_real_opaque_probe.py (experiment_logs).
    config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
    config.debug.aclgraph.static_capture_size_limit = CAPTURE_SIZE_LIMIT
    backend = torchair.get_npu_backend(compiler_config=config)
    return torch.compile(module.eval(), fullgraph=True, dynamic=False,
                         backend=backend)


def _measure_eager_call(
    label: str,
    call,
    measured: torch.Tensor,
    trials_path: Path,
) -> dict[str, object]:
    """Standalone eager call submit time; synchronization stays outside."""
    for _ in range(WARMUP_REPEATS):
        call(measured)
    torch.npu.synchronize()
    wall_values = []
    cpu_values = []
    for trial_index in range(TRIALS):
        cpu_start = time.thread_time_ns()
        wall_start = time.perf_counter_ns()
        call(measured)
        wall_end = time.perf_counter_ns()
        cpu_end = time.thread_time_ns()
        # Drain the device queue so backpressure cannot leak into the next
        # trial's Host submit interval.
        torch.npu.synchronize()
        wall_ns = wall_end - wall_start
        cpu_ns = cpu_end - cpu_start
        if wall_ns <= 0 or cpu_ns <= 0:
            raise RuntimeError("eager call timed interval must be positive")
        _append_json_line(trials_path, {
            "label": label,
            "trial_index": trial_index,
            "host_call_wall_ns": wall_ns,
            "host_call_thread_cpu_ns": cpu_ns,
        })
        wall_values.append(float(wall_ns))
        cpu_values.append(float(cpu_ns))
    return {
        "label": label,
        "host_call_thread_cpu_ns": {"median": _median(cpu_values)},
        "host_call_wall_ns": {"median": _median(wall_values)},
        "sample_count": TRIALS,
    }


def _measure_chain(
    class_name: str,
    factory,
    nodes_per_unit: int,
    chain_length: int,
    warmup: torch.Tensor,
    measured: torch.Tensor,
    trials_path: Path,
) -> dict[str, object]:
    compiled = _compiled_graph(factory(chain_length))
    for _ in range(WARMUP_REPEATS):
        compiled(warmup)
    torch.npu.synchronize()
    compiled(measured)
    torch.npu.synchronize()
    wall_values = []
    cpu_values = []
    for trial_index in range(TRIALS):
        cpu_start = time.thread_time_ns()
        wall_start = time.perf_counter_ns()
        result = compiled(measured)
        wall_end = time.perf_counter_ns()
        cpu_end = time.thread_time_ns()
        if result.device != measured.device:
            raise RuntimeError("class dispatch probe changed the device")
        # Drain the device queue so backpressure cannot leak into the next
        # trial's Host submit interval.
        torch.npu.synchronize()
        wall_ns = wall_end - wall_start
        cpu_ns = cpu_end - cpu_start
        if wall_ns <= 0 or cpu_ns <= 0:
            raise RuntimeError("class dispatch timed interval must be positive")
        _append_json_line(trials_path, {
            "class": class_name,
            "chain_length": chain_length,
            "trial_index": trial_index,
            "host_call_wall_ns": wall_ns,
            "host_call_thread_cpu_ns": cpu_ns,
        })
        wall_values.append(float(wall_ns))
        cpu_values.append(float(cpu_ns))
    return {
        "chain_length": chain_length,
        "graph_node_units": chain_length,
        "kernel_nodes": chain_length * nodes_per_unit,
        "host_call_thread_cpu_ns": {"median": _median(cpu_values)},
        "host_call_wall_ns": {"median": _median(wall_values)},
        "sample_count": TRIALS,
    }


def _measure_class(
    class_name: str,
    factory,
    nodes_per_unit: int,
    m_tokens: int,
    warmup: torch.Tensor,
    trials_path: Path,
) -> list[dict[str, object]]:
    """Full chain-length ladder for one class at one M point."""
    records = []
    measured = torch.randn(m_tokens, HIDDEN, dtype=DTYPE,
                           device=f"npu:{torch.npu.current_device()}")
    for chain_length in CHAIN_LENGTHS:
        records.append({
            "class": class_name,
            "m_tokens": m_tokens,
            **_measure_chain(class_name, factory, nodes_per_unit,
                             chain_length, warmup, measured, trials_path),
        })
    return records


def _eager_real_op_calls(m_tokens: int, device: str) -> dict[str, object]:
    """Eager standalones of the real op vehicles at one M point."""
    num_blocks = _paged_blocks(m_tokens)
    paged_key_cache = torch.randn(
        num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE,
        device=device)
    paged_value_cache = paged_key_cache.clone()
    flat_key = torch.randn(m_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=DTYPE,
                           device=device)
    flat_value = flat_key.clone()
    flat_query = torch.randn(m_tokens, NUM_HEADS * HEAD_DIM, dtype=DTYPE,
                             device=device)
    fia_key_cache = torch.randn(
        num_blocks, BLOCK_SIZE, NUM_KV_HEADS * HEAD_DIM, dtype=DTYPE,
        device=device)
    fia_value_cache = fia_key_cache.clone()
    block_table = torch.zeros(1, num_blocks, dtype=torch.int32, device=device)
    slot_mapping = (torch.arange(m_tokens, dtype=torch.int64, device=device)
                    % (num_blocks * BLOCK_SIZE))
    positions = torch.arange(m_tokens, dtype=torch.int64, device=device)
    cos_sin_cache = torch.randn(COS_SIN_MAX_POS, HEAD_DIM, dtype=DTYPE,
                                device=device)
    return {
        "fia_tnd": lambda t: torch_npu.npu_fused_infer_attention_score(
            query=t.view(m_tokens, NUM_HEADS, HEAD_DIM),
            key=fia_key_cache,
            value=fia_value_cache,
            atten_mask=None,
            block_table=block_table,
            input_layout="TND",
            block_size=BLOCK_SIZE,
            actual_seq_lengths=[m_tokens],
            actual_seq_lengths_kv=[m_tokens],
            num_key_value_heads=NUM_KV_HEADS,
            num_heads=NUM_HEADS,
            scale=FIA_SCALE,
            sparse_mode=0,
        ),
        "reshape_cache": lambda t: torch_npu._npu_reshape_and_cache(
            flat_key, flat_value, paged_key_cache, paged_value_cache,
            slot_mapping),
        "rope_custom": lambda t: torch.ops.vllm.npu_rotary_embedding(
            positions, flat_query, flat_key, cos_sin_cache, HEAD_DIM,
            HEAD_DIM, True),
    }


def _warm_reshape_cache(device: str) -> None:
    """Absorb the ATB ReshapeAndCache first-call setup failure.

    The first ``torch_npu._npu_reshape_and_cache`` call of a process fails
    inside ATB operation setup and every later call succeeds (reproduced in
    isolation on device 0 of this container); this untimed warm-up pays that
    one-time failure outside every measured interval.
    """
    key = torch.zeros(1, NUM_KV_HEADS * HEAD_DIM, dtype=DTYPE, device=device)
    value = key.clone()
    cache = torch.zeros(1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE,
                        device=device)
    slots = torch.zeros(1, dtype=torch.int64, device=device)
    try:
        torch_npu._npu_reshape_and_cache(key, value, cache, cache.clone(),
                                         slots)
        torch.npu.synchronize()
    except RuntimeError:
        torch_npu._npu_reshape_and_cache(key, value, cache, cache.clone(),
                                         slots)
        torch.npu.synchronize()


def main() -> None:
    arguments = _arguments()
    if isinstance(arguments.device_id, bool) or arguments.device_id < 0:
        raise ValueError("--device-id must be a non-negative integer")
    if (len(arguments.baseline_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in arguments.baseline_commit)):
        raise ValueError("--baseline-commit must be a lowercase full Git SHA")
    output_dir = _validate_output_dir(arguments.output_dir)
    output_dir.mkdir(parents=True)
    trials_path = output_dir / "trials.jsonl"
    device = f"npu:{arguments.device_id}"
    torch.npu.set_device(arguments.device_id)
    # Each chain length compiles a fresh module instance of the same forward
    # code object; the torchair backend fails closed on the default
    # recompilation budget, so raise it before the predeclared compiles
    # (r5 adds the mixed composite: 3 M x 7 variants x 2 lengths more).
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 2048
    # The triton rope impl reads device properties through this one-time
    # initializer before its first launch.
    init_device_properties_triton()
    _warm_reshape_cache(device)

    # Eager standalone measurements over the real M domain.
    eager_records = []
    for m_value in M_DOMAIN:
        hidden_input = torch.randn(m_value, HIDDEN, dtype=DTYPE, device=device)
        gate_up_weight = torch.randn(GATE_UP_OUT, HIDDEN, dtype=DTYPE,
                                     device=device)
        down_weight = torch.randn(HIDDEN, DOWN_IN, dtype=DTYPE, device=device)
        qkv_weight = torch.randn(QKV_OUT, HIDDEN, dtype=DTYPE, device=device)
        vocab_weight = torch.randn(VOCAB, HIDDEN, dtype=DTYPE, device=device)
        norm_weight = torch.ones(HIDDEN, dtype=DTYPE, device=device)
        residual = torch.randn(m_value, HIDDEN, dtype=DTYPE, device=device)
        down_input = torch.randn(m_value, DOWN_IN, dtype=DTYPE,
                                 device=device)
        swiglu_in = torch.randn(m_value, GATE_UP_OUT, dtype=DTYPE,
                                device=device)
        domain_calls = {
            "linear_gate_up": lambda t: torch.nn.functional.linear(
                t, gate_up_weight),
            "linear_down": lambda t: torch.nn.functional.linear(
                down_input, down_weight),
            "linear_qkv": lambda t: torch.nn.functional.linear(t, qkv_weight),
            "linear_lm_head": lambda t: torch.nn.functional.linear(
                t, vocab_weight),
            "aten_clone": lambda t: t.clone(),
            "aten_copy_": lambda t: residual.copy_(t),
            "aten_slice": lambda t: t[:, :NUM_KV_HEADS * HEAD_DIM].contiguous(),
            "torch_npu_rms_norm": lambda t: torch_npu.npu_rms_norm(
                t, norm_weight, 1e-6),
            "torch_npu_add_rms_norm": lambda t: torch_npu.npu_add_rms_norm(
                t, residual, norm_weight, 1e-6),
            "torch_npu_swiglu": lambda t: torch_npu.npu_swiglu(swiglu_in),
            "library_custom_clone": lambda t: _functional_custom_node(t),
        }
        domain_calls.update(_eager_real_op_calls(m_value, device))
        for label, call in domain_calls.items():
            eager_records.append(_measure_eager_call(
                f"eager:{label}:m{m_value}", call, hidden_input,
                trials_path))
        del gate_up_weight, down_weight, qkv_weight, vocab_weight, swiglu_in
        torch.npu.synchronize()

    # Chain fits at M=729 plus shape checks at the other real M points.
    chain_records = []
    fit_warmup = torch.randn(WARMUP_M, HIDDEN, dtype=DTYPE, device=device)
    for class_name, (factory, nodes_per_unit) in CLASS_CHAIN_FACTORIES.items():
        chain_records.extend(_measure_class(
            class_name, factory, nodes_per_unit, FIT_M, fit_warmup,
            trials_path))
        for check_m in CHECK_MS:
            check_warmup = fit_warmup
            measured = torch.randn(check_m, HIDDEN, dtype=DTYPE, device=device)
            chain_records.append({
                "class": class_name,
                "m_tokens": check_m,
                **_measure_chain(class_name, factory, nodes_per_unit, 8,
                                 check_warmup, measured, trials_path),
            })
    # Buffer-based classes: one module instance per (class, M); the source
    # input is unused by these vehicles, so the warmup shape is free.
    for class_name, (factory, nodes_per_unit) in BUFFER_CHAIN_CLASSES.items():
        for m_value in (FIT_M, *CHECK_MS):
            warmup = torch.randn(WARMUP_M, HIDDEN, dtype=DTYPE, device=device)
            # Bind the class's fixed M here so the chain factory takes only
            # the chain length, matching _measure_chain's call signature.
            bound_factory = (lambda count, f=factory, m=m_value: f(count, m))
            chain_records.extend(_measure_class(
                class_name, bound_factory, nodes_per_unit, m_value, warmup,
                trials_path))

    # Weight-domain checks: 8-block Matmul chains over the further real
    # linear output domains (qkv, o-proj square, lm-head wide).
    for domain_name, out_features in LINEAR_DOMAIN_CHECKS.items():
        factory = (lambda count, out=out_features:
                   _LinearDomainCheckChain(count, out))
        class_name = f"aten_linear_domain_check:{domain_name}"
        chain_records.append({
            "class": class_name,
            "m_tokens": FIT_M,
            **_measure_chain(class_name, factory, 2, 8, fit_warmup,
                             torch.randn(FIT_M, HIDDEN, dtype=DTYPE,
                                         device=device), trials_path),
        })

    # Mixed composite: the in-mix instrument at every real M point.  The
    # composite ignores its input (all tensors fixed-M); the warm-up call
    # at WARMUP_M captures graph key #1 and the measured call at M trips
    # the capture-size limit and latches fallback_to_eager -- the same
    # r4-verified pattern as the buffer-based chain classes.
    # Count-differencing against the base census gives the in-mix per-class
    # marginal node cost.
    composite_records = []
    for m_value in M_DOMAIN:
        for extra in (None, *COMPOSITE_EXTRA_CLASSES):
            nodes_per_unit = COMPOSITE_BASE_NODES + (extra is not None)
            for layer_count in COMPOSITE_LAYER_LENGTHS:
                factory = (lambda count, m=m_value, e=extra:
                           _MixedLayerChain(count, m, e))
                composite_records.append({
                    "composite_extra": extra,
                    "m_tokens": m_value,
                    **_measure_chain(
                        f"composite:{extra or 'base'}", factory,
                        nodes_per_unit, layer_count, fit_warmup,
                        torch.randn(m_value, HIDDEN, dtype=DTYPE,
                                    device=device),
                        trials_path),
                })

    # Per-class fits: slope is per chain unit; convert to per kernel node.
    fits = {}
    for class_name, (_, nodes_per_unit) in {
        **CLASS_CHAIN_FACTORIES,
        **{name: (None, nodes) for name, (_, nodes)
           in BUFFER_CHAIN_CLASSES.items()},
    }.items():
        fit_points = [
            (record["chain_length"],
             record["host_call_thread_cpu_ns"]["median"])
            for record in chain_records
            if record["class"] == class_name and record["m_tokens"] == FIT_M
            and record["chain_length"] in FIT_CHAIN_LENGTHS
        ]
        intercept, unit_slope = _linear_fit(fit_points)
        node_slope = unit_slope / nodes_per_unit
        holdout = next(
            record for record in chain_records
            if record["class"] == class_name and record["m_tokens"] == FIT_M
            and record["chain_length"] == HOLDOUT_CHAIN_LENGTH)
        holdout_predicted = (intercept + unit_slope * HOLDOUT_CHAIN_LENGTH)
        eager_label = EAGER_LABELS[class_name]
        eager_fit_m = next(
            record for record in eager_records
            if record["label"] == f"eager:{eager_label}:m{FIT_M}")
        check_slopes = {}
        for check_m in CHECK_MS:
            check_record = next(
                record for record in chain_records
                if record["class"] == class_name
                and record["m_tokens"] == check_m
                and record["chain_length"] == 8)
            eager_check = next(
                record for record in eager_records
                if record["label"] == f"eager:{eager_label}:m{check_m}")
            check_slopes[str(check_m)] = {
                "chain8_per_node_thread_cpu_ns":
                    (check_record["host_call_thread_cpu_ns"]["median"]
                     / nodes_per_unit),
                "eager_thread_cpu_ns":
                    eager_check["host_call_thread_cpu_ns"]["median"],
            }
        fits[class_name] = {
            "thread_cpu_intercept_ns": intercept,
            "thread_cpu_unit_slope_ns": unit_slope,
            "kernel_nodes_per_unit": nodes_per_unit,
            "thread_cpu_node_slope_ns": node_slope,
            "eager_standalone_fit_m_thread_cpu_ns":
                eager_fit_m["host_call_thread_cpu_ns"]["median"],
            "machinery_delta_node_ns":
                node_slope - eager_fit_m["host_call_thread_cpu_ns"]["median"],
            "holdout_thread_cpu_signed_error_percent": (
                (holdout["host_call_thread_cpu_ns"]["median"]
                 - holdout_predicted) / holdout_predicted * 100.0),
            "shape_checks": check_slopes,
        }

    graph_entry_ns = statistics.median(
        fits[class_name]["thread_cpu_intercept_ns"]
        for class_name in fits)

    # Composite analysis: count-differenced in-mix per-class marginals, the
    # base census projection per 36-layer forward, and cross-length
    # consistency of every marginal.
    composite_marginals = {}
    for m_value in M_DOMAIN:
        base_by_length = {
            record["chain_length"]:
                record["host_call_thread_cpu_ns"]["median"]
            for record in composite_records
            if record["composite_extra"] is None
            and record["m_tokens"] == m_value
        }
        # Fail closed if the composite was ACLGraph-replayed instead of
        # per-node dispatch: replay erases the length slope, while every
        # real per-node dispatch class costs well above this floor.
        base_slope_per_node_ns = (
            (base_by_length[max(COMPOSITE_LAYER_LENGTHS)]
             - base_by_length[min(COMPOSITE_LAYER_LENGTHS)])
            / ((max(COMPOSITE_LAYER_LENGTHS) - min(COMPOSITE_LAYER_LENGTHS))
               * COMPOSITE_BASE_NODES))
        if base_slope_per_node_ns < 5000.0:
            raise RuntimeError(
                "composite base census shows no per-node dispatch slope; "
                "the ACLGraph replay regime was measured instead")
        per_m = {
            "base_per_layer_thread_cpu_ns": {
                str(length): base / length
                for length, base in base_by_length.items()},
            # Diagnostic only: extrapolates the measured base census to the
            # real 36-layer forward scale for instrument-vs-lane comparison.
            "census_projection_36_layers_thread_cpu_ns": {
                str(length): base / length * COMPOSITE_LAYERS_PER_FORWARD
                for length, base in base_by_length.items()},
        }
        for extra in COMPOSITE_EXTRA_CLASSES:
            marginal_by_length = {}
            for layer_count in COMPOSITE_LAYER_LENGTHS:
                variant = next(
                    record for record in composite_records
                    if record["composite_extra"] == extra
                    and record["m_tokens"] == m_value
                    and record["chain_length"] == layer_count)
                marginal_by_length[str(layer_count)] = (
                    (variant["host_call_thread_cpu_ns"]["median"]
                     - base_by_length[layer_count]) / layer_count)
            per_m[f"extra:{extra}:marginal_node_thread_cpu_ns"] = (
                marginal_by_length)
        composite_marginals[str(m_value)] = per_m

    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "ascend_torchair_node_class_dispatch_microbenchmark",
        "hardware_id": HARDWARE_ID,
        "mechanism": (
            "torchair reduce-overhead AclGraph latched to fallback_to_eager "
            "via static_capture_size_limit; per-node Host dispatch measured "
            "by op class with vehicles carrying each class's real dispatch "
            "machinery, paired with standalone eager calls of the same "
            "vehicle so node slope minus eager standalone isolates the "
            "per-node machinery cost"),
        "measurement_boundary": (
            "compiled graph call entry to return (chains) and single call "
            "submit (eager); device synchronization is outside the timed "
            "interval in both cases"),
        "measurement_semantics": {
            "m_domain_tokens": list(M_DOMAIN),
            "fit_m_tokens": FIT_M,
            "warmup_m_tokens": WARMUP_M,
            "dtype": str(DTYPE).replace("torch.", ""),
            "model_domains": {
                "hidden": HIDDEN,
                "gate_up_out": GATE_UP_OUT,
                "down_in": DOWN_IN,
                "qkv_out": QKV_OUT,
                "vocab": VOCAB,
                "num_heads": NUM_HEADS,
                "num_kv_heads": NUM_KV_HEADS,
                "head_dim": HEAD_DIM,
                "kv_block_size": BLOCK_SIZE,
            },
            "composite_census_per_unit": dict(COMPOSITE_CENSUS),
            "composite_layers_per_forward": COMPOSITE_LAYERS_PER_FORWARD,
            "static_capture_size_limit": CAPTURE_SIZE_LIMIT,
            "backend_config": {
                "mode": "reduce-overhead",
                "run_eagerly": True,
                "jit_compile": False,
            },
            "class_vehicles": {
                "aten_linear": "F.linear gate_up(24576)/down(12288) block",
                "aten_copy": (
                    "one aten.copy_ from a distinct seed into a distinct "
                    "sink per node (distinct destinations defeat functional "
                    "and in-place folding; r4's clone+copy_ chain was "
                    "semantically collapsible and collapsed)"),
                "torch_npu_light": "torch_npu.npu_rms_norm chain",
                "library_custom": (
                    "torch.library.custom_op functional node wrapping clone"),
                "fia_tnd": (
                    "torch_npu.npu_fused_infer_attention_score, TND layout "
                    "over a real paged KV shape (sparse_mode=0 vehicle; the "
                    "engine's prefill call uses sparse_mode=3 with the same "
                    "pybind machinery and arg count)"),
                "reshape_cache": (
                    "torch_npu._npu_reshape_and_cache over a real paged "
                    "cache and slot mapping"),
                "rope_custom": (
                    "torch.ops.vllm.npu_rotary_embedding custom op; the "
                    "registered impl launches the triton rope kernel"),
            },
            "class_equivalence_notes": (
                "torch_npu.npu_rms_norm stands in for the torch_npu pybind "
                "norm-op class (the canonical engine's AddRmsNormBias is the "
                "vllm_ascend_C extension of the same dispatch class); "
                "aten.copy_ stands in for the TensorMove/Slice "
                "aclnnInPlaceCopy and aclnnContiguous dispatch classes"),
        },
        "predeclared": {
            "chain_lengths": list(CHAIN_LENGTHS),
            "fit_chain_lengths": sorted(FIT_CHAIN_LENGTHS),
            "holdout_chain_length": HOLDOUT_CHAIN_LENGTH,
            "trials": TRIALS,
            "warmup_repeats": WARMUP_REPEATS,
            "composite_layer_lengths": list(COMPOSITE_LAYER_LENGTHS),
            "composite_extra_classes": list(COMPOSITE_EXTRA_CLASSES),
            "eager_domains": [
                "linear_gate_up", "linear_down", "linear_qkv",
                "linear_lm_head", "aten_clone", "aten_copy_", "aten_slice",
                "torch_npu_rms_norm", "torch_npu_add_rms_norm",
                "torch_npu_swiglu", "library_custom_clone", "fia_tnd",
                "reshape_cache", "rope_custom",
            ],
        },
        "eager_calls": eager_records,
        "chains": chain_records,
        "composite_chains": composite_records,
        "composite_marginals": composite_marginals,
        "fits": fits,
        "graph_entry_thread_cpu_ns": graph_entry_ns,
        "environment": {
            "container": arguments.container_name,
            "device_id": arguments.device_id,
            "python": __import__("platform").python_version(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "vllm": vllm.__version__,
        },
        "baseline_commit": arguments.baseline_commit,
        "completed_at": _utc_now(),
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    print(json.dumps({
        "summary_path": str(summary_path),
        "fits": {
            class_name: {
                "node_slope_ns": value["thread_cpu_node_slope_ns"],
                "machinery_delta_ns": value["machinery_delta_node_ns"],
            }
            for class_name, value in fits.items()
        },
        "graph_entry_ns": graph_entry_ns,
        "composite_marginals_fit_m": composite_marginals[str(FIT_M)],
    }))


if __name__ == "__main__":
    main()
