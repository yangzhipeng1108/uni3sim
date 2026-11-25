"""Serving-aligned A800 Vector runner for NCU and no-profiler CUDA Events."""
import argparse
from pathlib import Path
import sys

import torch
import vllm
from vllm import _custom_ops as ops
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel import device_event as event_backend
from evaluation.kernel.device_event import (
    add_event_cli_arguments,
    parse_event_run_options,
    prepare_bf16_cache_eviction_sweep,
    validate_cache_eviction_mib,
)
from dataset_pipeline.kernel_dataset.nvidia_a800_vector_event import (
    allocate_ring_tensor,
    make_rotating_operation,
    measure_nvidia_a800_vector_event,
)
from dataset_pipeline.kernel_dataset.nvidia_a800_event_capture import (
    A800_VECTOR_POWER_WARMUP_GEMM_DIMENSION,
    A800_VECTOR_POWER_WARMUP_GEMM_ITERATIONS,
    A800_VECTOR_EVENT_TIMING_MODES,
    build_nvidia_a800_vector_differential_graph_contract,
    build_nvidia_a800_vector_event_batch_contract,
    build_nvidia_a800_vector_graph_contract,
    build_nvidia_a800_vector_power_warmup_provenance,
)

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}
WARMUP = 5
def _allocate_ring_tensor(
        shape, dtype, contract, contract_section, operand, *, empty=False):
    return allocate_ring_tensor(
        torch_module=torch, shape=shape, dtype=dtype, contract=contract,
        contract_section=contract_section, operand=operand, empty=empty)


def _event_batch_contract(workload, event_options, event_input_ring_mib,
                          event_operation_repetitions_per_sample,
                          event_vector_timing_mode):
    if (isinstance(event_input_ring_mib, bool)
            or not isinstance(event_input_ring_mib, int)):
        raise TypeError("event_input_ring_mib must be an integer")
    if (isinstance(event_operation_repetitions_per_sample, bool)
            or not isinstance(event_operation_repetitions_per_sample, int)):
        raise TypeError(
            "event_operation_repetitions_per_sample must be an integer")
    if event_vector_timing_mode not in A800_VECTOR_EVENT_TIMING_MODES:
        raise ValueError(
            f"unsupported A800 Vector timing mode: {event_vector_timing_mode}")
    uses_batch = event_operation_repetitions_per_sample != 1
    uses_ring = event_input_ring_mib != 0
    uses_graph = event_vector_timing_mode in {
        "cuda_graph_full_ring_replay", "cuda_graph_differential"}
    if event_options is None and (uses_batch or uses_ring or uses_graph):
        raise ValueError("A800 Vector rotating batches require Event mode")
    if uses_graph:
        if not uses_ring or uses_batch:
            raise ValueError(
                "A800 Vector CUDA-Graph replay requires an input ring and "
                "exactly one public call per direct Event option")
        if event_vector_timing_mode == "cuda_graph_differential":
            return build_nvidia_a800_vector_differential_graph_contract(
                workload, minimum_distinct_input_mib=event_input_ring_mib)
        return build_nvidia_a800_vector_graph_contract(
            workload, input_ring_mib=event_input_ring_mib)
    if uses_batch != uses_ring:
        raise ValueError(
            "A800 Vector Event batching requires both a nonzero input ring "
            "and an explicitly selected public-call count")
    if not uses_batch:
        return None
    return build_nvidia_a800_vector_event_batch_contract(
        workload, input_ring_mib=event_input_ring_mib,
        public_call_count=event_operation_repetitions_per_sample)

def _measure(fn, nvtx_range_name, cache_eviction_operation):
    """Warm outside NVTX, then expose exactly one post-warm launch to NCU."""
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    if cache_eviction_operation is not None:
        cache_eviction_operation()
        torch.cuda.current_stream().synchronize()
    torch.cuda.nvtx.range_push(nvtx_range_name)
    fn()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def _environment():
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": vllm.__version__,
        "device_index": device_index,
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "input_seed": 20260826,
    }


def _prepare_device_power_warmup():
    dimension = A800_VECTOR_POWER_WARMUP_GEMM_DIMENSION
    left = torch.randn(dimension, dimension, dtype=torch.bfloat16, device="cuda")
    right = torch.randn(dimension, dimension, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(dimension, dimension, dtype=torch.bfloat16, device="cuda")

    def operation():
        return torch.mm(left, right, out=output)

    for _ in range(A800_VECTOR_POWER_WARMUP_GEMM_ITERATIONS):
        operation()
    torch.cuda.synchronize()
    return operation, build_nvidia_a800_vector_power_warmup_provenance()


def _run(fn, *, nvtx_range_name, event_options, workload,
         allocation_evidence, pre_nvtx_cache_eviction_mib=0,
         input_ring_contract=None,
         event_operation_repetitions_per_sample=1,
         event_vector_timing_mode="direct_batched_event"):
    if event_options is None:
        if (input_ring_contract is not None
                or event_operation_repetitions_per_sample != 1):
            raise ValueError(
                "A800 Vector rotating batches require Event mode")
        if not isinstance(nvtx_range_name, str) or not nvtx_range_name:
            raise ValueError("normal A800 runner mode requires an NVTX name")
        validate_cache_eviction_mib(
            pre_nvtx_cache_eviction_mib, minimum_nonzero_mib=320,
            hardware_id="nvidia_a800")
        cache_eviction_operation, _ = prepare_bf16_cache_eviction_sweep(
            torch_module=torch, device="cuda",
            cache_eviction_mib=pre_nvtx_cache_eviction_mib)
        _measure(fn, nvtx_range_name, cache_eviction_operation)
        return
    if nvtx_range_name is not None:
        raise ValueError("A800 Event mode cannot enable an NVTX range")
    if pre_nvtx_cache_eviction_mib:
        raise ValueError(
            "A800 Event mode cannot enable a pre-NVTX cache sweep")
    uses_differential_graph = event_vector_timing_mode == "cuda_graph_differential"
    if uses_differential_graph:
        power_maintenance_operation, power_warmup = (
            _prepare_device_power_warmup())
    else:
        power_maintenance_operation, power_warmup = None, None
    measure_nvidia_a800_vector_event(
        fn, torch_module=torch, event_options=event_options,
        event_backend=event_backend,
        workload=workload, allocation_evidence=allocation_evidence,
        input_ring_contract=input_ring_contract,
        operation_repetitions_per_sample=event_operation_repetitions_per_sample,
        timing_mode=event_vector_timing_mode, runner_path=Path(__file__),
        environment=_environment(),
        power_maintenance_operation=power_maintenance_operation,
        power_warmup=power_warmup,
    )


def _finish(fn, nvtx, event_options, workload, allocation_evidence,
            pre_nvtx_cache_eviction_mib, ring_contract,
            event_operation_repetitions_per_sample, event_vector_timing_mode):
    _run(
        fn, nvtx_range_name=nvtx, event_options=event_options,
        workload=workload, allocation_evidence=allocation_evidence,
        pre_nvtx_cache_eviction_mib=pre_nvtx_cache_eviction_mib,
        input_ring_contract=ring_contract,
        event_operation_repetitions_per_sample=event_operation_repetitions_per_sample,
        event_vector_timing_mode=event_vector_timing_mode)


def run_rmsnorm(tok, hidden, nvtx, dtype="bf16", add_residual=False,
                event_options=None, pre_nvtx_cache_eviction_mib=0,
                event_input_ring_mib=0,
                event_operation_repetitions_per_sample=1,
                event_vector_timing_mode="direct_batched_event"):
    dt = DTYPE_MAP[dtype.lower()]
    rn = RMSNorm(hidden).to("cuda").to(dt)
    op_name = "add_rmsnorm" if add_residual else "rmsnorm"
    workload = {
        "op": op_name, "tok": tok, "hidden": hidden,
        "dtype": dtype, "dtype_bytes": dt.itemsize,
    }
    ring_contract = _event_batch_contract(
        workload, event_options, event_input_ring_mib,
        event_operation_repetitions_per_sample, event_vector_timing_mode)
    slot_count = 1 if ring_contract is None else ring_contract["slot_count"]
    if add_residual:
        if ring_contract is None:
            x = torch.randn(tok, hidden, device="cuda", dtype=dt)
            residual = torch.randn(tok, hidden, device="cuda", dtype=dt)
            fn = lambda: rn.forward_cuda(x, residual)
        else:
            x = _allocate_ring_tensor((tok, hidden), dt, ring_contract,
                                      "dynamic_input_bytes_per_operand", "x")
            residual = _allocate_ring_tensor(
                (tok, hidden), dt, ring_contract,
                "dynamic_input_bytes_per_operand", "residual")
            fn = make_rotating_operation(
                slot_count, ring_contract["slot_stride"],
                lambda index: ops.fused_add_rms_norm(
                    x[index], residual[index], rn.weight.data,
                    rn.variance_epsilon))
        _finish(
            fn, nvtx, event_options, workload, (
                "the single-call path uses RMSNorm.forward_cuda; the batched "
                "path calls its exact fused_add_rms_norm custom op over "
                "preallocated disjoint x/residual slots and the shared module weight"),
            pre_nvtx_cache_eviction_mib, ring_contract,
            event_operation_repetitions_per_sample, event_vector_timing_mode)
    else:
        if ring_contract is None:
            x = torch.randn(tok, hidden, device="cuda", dtype=dt)
            fn = lambda: rn.forward_cuda(x)
        else:
            x = _allocate_ring_tensor((tok, hidden), dt, ring_contract,
                                      "dynamic_input_bytes_per_operand", "x")
            output = _allocate_ring_tensor(
                (tok, hidden), dt, ring_contract,
                "preallocated_output_bytes_per_operand", "output", empty=True)
            fn = make_rotating_operation(
                slot_count, ring_contract["slot_stride"],
                lambda index: ops.rms_norm(
                    output[index], x[index], rn.weight.data,
                    rn.variance_epsilon))
        _finish(
            fn, nvtx, event_options, workload, (
                "the single-call path uses RMSNorm.forward_cuda; the batched "
                "path calls its exact rms_norm custom op over preallocated "
                "disjoint input/output slots and the shared module weight"),
            pre_nvtx_cache_eviction_mib, ring_contract,
            event_operation_repetitions_per_sample, event_vector_timing_mode)


def run_rope(tok, num_heads, num_kv_heads, head_dim, nvtx, dtype="bf16",
             event_options=None, pre_nvtx_cache_eviction_mib=0,
             event_input_ring_mib=0,
             event_operation_repetitions_per_sample=1,
             event_vector_timing_mode="direct_batched_event"):
    dt = DTYPE_MAP[dtype.lower()]
    re = RotaryEmbedding(head_size=head_dim, rotary_dim=head_dim,
                         max_position_embeddings=max(tok, 4096), base=10000,
                         is_neox_style=True, dtype=dt).to("cuda")
    positions = torch.arange(tok, device="cuda", dtype=torch.long)
    workload = {
        "op": "rope", "tok": tok, "num_heads": num_heads,
        "num_kv_heads": num_kv_heads, "head_dim": head_dim,
        "dtype": dtype, "dtype_bytes": dt.itemsize,
    }
    ring_contract = _event_batch_contract(
        workload, event_options, event_input_ring_mib,
        event_operation_repetitions_per_sample, event_vector_timing_mode)
    if ring_contract is None:
        q = torch.randn(tok, num_heads * head_dim, device="cuda", dtype=dt)
        k = torch.randn(tok, num_kv_heads * head_dim, device="cuda", dtype=dt)
        fn = lambda: re.forward_cuda(positions, q, k)
    else:
        slot_count = ring_contract["slot_count"]
        q = _allocate_ring_tensor((tok, num_heads * head_dim), dt, ring_contract,
                                  "dynamic_input_bytes_per_operand", "q")
        k = _allocate_ring_tensor((tok, num_kv_heads * head_dim), dt, ring_contract,
                                  "dynamic_input_bytes_per_operand", "k")
        fn = make_rotating_operation(
            slot_count, ring_contract["slot_stride"],
            lambda index: re.forward_cuda(positions, q[index], k[index]))
    _finish(
        fn, nvtx, event_options, workload, (
            "RotaryEmbedding cache dtype is established before timing and "
            "the exact custom op updates either the single preallocated q/k "
            "pair or disjoint rotating q/k slots in place"),
        pre_nvtx_cache_eviction_mib, ring_contract,
        event_operation_repetitions_per_sample, event_vector_timing_mode)


def run_silu(
        tok, intermediate, nvtx, dtype="bf16", event_options=None,
        pre_nvtx_cache_eviction_mib=0, event_input_ring_mib=0,
        event_operation_repetitions_per_sample=1,
        event_vector_timing_mode="direct_batched_event"):
    dt = DTYPE_MAP[dtype.lower()]
    sm = SiluAndMul().to("cuda")
    workload = {
        "op": "silu", "tok": tok, "intermediate": intermediate,
        "dtype": dtype, "dtype_bytes": dt.itemsize,
    }
    ring_contract = _event_batch_contract(
        workload, event_options, event_input_ring_mib,
        event_operation_repetitions_per_sample, event_vector_timing_mode)
    if ring_contract is None:
        x = torch.randn(tok, 2 * intermediate, device="cuda", dtype=dt)
        fn = lambda: sm.forward_cuda(x)
    else:
        slot_count = ring_contract["slot_count"]
        x = _allocate_ring_tensor((tok, 2 * intermediate), dt, ring_contract,
                                  "dynamic_input_bytes_per_operand", "input_tensor")
        output = _allocate_ring_tensor(
            (tok, intermediate), dt, ring_contract,
            "preallocated_output_bytes_per_operand", "output", empty=True)
        fn = make_rotating_operation(
            slot_count, ring_contract["slot_stride"],
            lambda index: sm.op(output[index], x[index]))
    _finish(
        fn, nvtx, event_options, workload, (
            "the single-call path uses SiluAndMul.forward_cuda; the batched "
            "path calls the same module custom op with disjoint preallocated "
            "input and output slots"),
        pre_nvtx_cache_eviction_mib, ring_contract,
        event_operation_repetitions_per_sample, event_vector_timing_mode)


def run_elementwise(tok, hidden, nvtx, dtype="bf16", op2="add",
                    event_options=None, pre_nvtx_cache_eviction_mib=0,
                    event_input_ring_mib=0,
                    event_operation_repetitions_per_sample=1,
                    event_vector_timing_mode="direct_batched_event"):
    dt = DTYPE_MAP[dtype.lower()]
    workload = {
        "op": "elementwise", "op2": op2, "tok": tok,
        "hidden": hidden, "dtype": dtype, "dtype_bytes": dt.itemsize,
    }
    ring_contract = _event_batch_contract(
        workload, event_options, event_input_ring_mib,
        event_operation_repetitions_per_sample, event_vector_timing_mode)
    if ring_contract is None:
        a = torch.randn(tok, hidden, device="cuda", dtype=dt)
        b = torch.randn(tok, hidden, device="cuda", dtype=dt)
        if op2 == "add":
            fn = lambda: a + b
        elif op2 == "mul":
            fn = lambda: a * b
        else:
            raise ValueError(f"unknown elementwise operation: {op2!r}")
    else:
        slot_count = ring_contract["slot_count"]
        a = _allocate_ring_tensor((tok, hidden), dt, ring_contract,
                                  "dynamic_input_bytes_per_operand", "left")
        b = _allocate_ring_tensor((tok, hidden), dt, ring_contract,
                                  "dynamic_input_bytes_per_operand", "right")
        output = _allocate_ring_tensor(
            (tok, hidden), dt, ring_contract,
            "preallocated_output_bytes_per_operand", "output", empty=True)
        if op2 == "add":
            operation = lambda index: torch.add(
                a[index], b[index], out=output[index])
        elif op2 == "mul":
            operation = lambda index: torch.mul(
                a[index], b[index], out=output[index])
        else:
            raise ValueError(f"unknown elementwise operation: {op2!r}")
        fn = make_rotating_operation(
            slot_count, ring_contract["slot_stride"], operation)
    _finish(
        fn, nvtx, event_options, workload, (
            "the single-call path preserves canonical a+b/a*b; the batched "
            "path uses equivalent torch.add/torch.mul out= calls with "
            "disjoint preallocated input and output slots"),
        pre_nvtx_cache_eviction_mib, ring_contract,
        event_operation_repetitions_per_sample, event_vector_timing_mode)


def run_kvcache(tok, num_kv_heads, head_dim, nvtx, batch=1, dtype="bf16",
                event_options=None, pre_nvtx_cache_eviction_mib=0,
                event_input_ring_mib=0,
                event_operation_repetitions_per_sample=1,
                event_vector_timing_mode="direct_batched_event"):
    """Paged KV write via the v1 FlashAttention serving path
    (reshape_and_cache_flash). Writes (tok, num_kv_heads, head_dim) new KV into a
    paged pool of shape (num_blocks, block_size, num_kv_heads, head_dim) via
    slot_mapping. `batch` labels the serving decode batch (attr/filename only);
    the measured work is `tok` token-writes, matching ascend_910b/vector run_kvcache."""
    if min(tok, num_kv_heads, head_dim, batch) <= 0:
        raise ValueError("KV-cache workload dimensions must be positive")
    dt = DTYPE_MAP[dtype.lower()]
    workload = {
        "op": "kvcache", "tok": tok,
        "num_kv_heads": num_kv_heads, "head_dim": head_dim,
        "batch": batch, "dtype": dtype, "dtype_bytes": dt.itemsize,
    }
    ring_contract = _event_batch_contract(
        workload, event_options, event_input_ring_mib,
        event_operation_repetitions_per_sample, event_vector_timing_mode)
    slot_count = 1 if ring_contract is None else ring_contract["slot_count"]
    if ring_contract is None:
        key = torch.randn(tok, num_kv_heads, head_dim, device="cuda", dtype=dt)
        value = torch.randn(tok, num_kv_heads, head_dim, device="cuda", dtype=dt)
    else:
        key = _allocate_ring_tensor(
            (tok, num_kv_heads, head_dim), dt, ring_contract,
            "dynamic_input_bytes_per_operand", "key")
        value = _allocate_ring_tensor(
            (tok, num_kv_heads, head_dim), dt, ring_contract,
            "dynamic_input_bytes_per_operand", "value")
    block_size = 16  # v1 FA backend requires block_size % 16 == 0
    total_slots = slot_count * tok
    if ring_contract is None:
        num_blocks = (tok + block_size - 1) // block_size + 1
    else:
        num_blocks = max(1, (total_slots + block_size - 1) // block_size)
    key_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim,
                            device="cuda", dtype=dt)
    value_cache = torch.zeros_like(key_cache)
    if ring_contract is None:
        slot_mapping = torch.arange(tok, device="cuda", dtype=torch.long)
    else:
        slot_mapping = torch.arange(
            total_slots, device="cuda", dtype=torch.long).reshape(
                slot_count, tok)
    k_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    v_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    if ring_contract is None:
        fn = lambda: ops.reshape_and_cache_flash(
            key, value, key_cache, value_cache, slot_mapping, "auto",
            k_scale, v_scale)
    else:
        actual_output_bytes = (
            key_cache.numel() * key_cache.element_size()
            + value_cache.numel() * value_cache.element_size())
        if actual_output_bytes != ring_contract["allocated_output_ring_bytes"]:
            raise RuntimeError(
                "KV-cache destination allocation disagrees with the Event "
                "batch contract")
        fn = make_rotating_operation(
            slot_count, ring_contract["slot_stride"],
            lambda index: ops.reshape_and_cache_flash(
                key[index], value[index], key_cache, value_cache,
                slot_mapping[index], "auto", k_scale, v_scale))
    _finish(
        fn, nvtx, event_options, workload, (
            "reshape_and_cache_flash reads either one key/value pair or "
            "disjoint rotating key/value slots and writes into preallocated "
            "nonoverlapping ranges of shared key/value cache tensors"),
        pre_nvtx_cache_eviction_mib, ring_contract,
        event_operation_repetitions_per_sample, event_vector_timing_mode)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True,
                   choices=["rmsnorm", "add_rmsnorm", "rope", "silu", "elementwise", "kvcache"])
    for name in ("tok", "hidden", "num_heads", "head_dim", "intermediate",
                 "num_kv_heads", "batch"):
        p.add_argument(f"--{name}", type=int, required=True)
    p.add_argument("--dtype", required=True, choices=sorted(DTYPE_MAP))
    p.add_argument("--op2", required=True, choices=["add", "mul"])
    p.add_argument("--nvtx_range_name", type=str)
    p.add_argument("--pre-nvtx-cache-eviction-mib", type=int, default=0)
    p.add_argument("--event-input-ring-mib", type=int, default=0)
    p.add_argument("--event-operation-repetitions-per-sample", type=int, default=1)
    p.add_argument("--event-vector-timing-mode",
                   choices=sorted(A800_VECTOR_EVENT_TIMING_MODES), default="direct_batched_event")
    add_event_cli_arguments(p)
    a = p.parse_args()
    event_options = parse_event_run_options(a)

    torch.cuda.set_device(0)
    if event_options is not None:
        torch.manual_seed(20260826)
        torch.cuda.manual_seed(20260826)
    common = {
        "event_options": event_options,
        "pre_nvtx_cache_eviction_mib": a.pre_nvtx_cache_eviction_mib,
        "event_input_ring_mib": a.event_input_ring_mib,
        "event_operation_repetitions_per_sample": a.event_operation_repetitions_per_sample,
        "event_vector_timing_mode": a.event_vector_timing_mode,
    }
    if a.op == "rmsnorm":
        run_rmsnorm(a.tok, a.hidden, a.nvtx_range_name, a.dtype,
                    add_residual=False, **common)
    elif a.op == "add_rmsnorm":
        run_rmsnorm(a.tok, a.hidden, a.nvtx_range_name, a.dtype,
                    add_residual=True, **common)
    elif a.op == "rope":
        run_rope(a.tok, a.num_heads, a.num_kv_heads, a.head_dim,
                 a.nvtx_range_name, a.dtype, **common)
    elif a.op == "silu":
        run_silu(a.tok, a.intermediate, a.nvtx_range_name, a.dtype, **common)
    elif a.op == "elementwise":
        run_elementwise(a.tok, a.hidden, a.nvtx_range_name, a.dtype, a.op2,
                        **common)
    elif a.op == "kvcache":
        run_kvcache(a.tok, a.num_kv_heads, a.head_dim, a.nvtx_range_name,
                    a.batch, a.dtype, **common)
    else:
        raise AssertionError(f"argparse accepted an unknown vector op: {a.op!r}")
    print("VEC done")
