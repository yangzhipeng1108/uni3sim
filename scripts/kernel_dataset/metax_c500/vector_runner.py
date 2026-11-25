"""MetaX C500 Serving-aligned vector operator runner."""

import argparse
from pathlib import Path
import sys
import time

import torch
import vllm_metax_plugin._C  # noqa: F401 - registers torch.ops._C kernels

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.device_event import (
    EventRunOptions,
    add_event_cli_arguments,
    measure_differential_rotating_graph_replay_events,
    measure_graph_replay_events,
    measure_rotating_graph_replay_events,
    measure_current_stream_events,
    parse_event_run_options,
    prepare_bf16_cache_eviction_sweep,
    validate_cache_eviction_mib,
    write_event_case_record,
)
from dataset_pipeline.kernel_dataset.metax_c500_event_capture import (
    build_vector_buffer_contract,
)


DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}
C500_L2_BYTES = 8 * 1024 * 1024

def _warm_run(fn, *, warmup=3, use_marker=False, profiler_wait=0):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if use_marker:
        time.sleep(profiler_wait)
        torch.cuda.profiler.start()
    elif profiler_wait:
        time.sleep(profiler_wait)
    fn()
    torch.cuda.synchronize()
    if use_marker:
        torch.cuda.profiler.stop()

def _run(operations, *, use_marker, profiler_wait,
         event_options: EventRunOptions | None, workload, allocation_evidence,
         graph_repetitions, minimum_distinct_input_mib):
    if (not isinstance(operations, tuple) or not operations
            or any(not callable(operation) for operation in operations)):
        raise TypeError("operations must be a non-empty tuple of callables")
    operation = operations[0]
    if event_options is None:
        if len(operations) != 1:
            raise ValueError("rotating buffers require Event graph mode")
        _warm_run(
            operation, use_marker=use_marker, profiler_wait=profiler_wait)
        return
    validate_cache_eviction_mib(
        event_options.cache_eviction_mib, minimum_nonzero_mib=64,
        hardware_id="metax_c500")
    if use_marker or profiler_wait != 0:
        raise ValueError(
            "MetaX Event mode cannot enable profiler marker/wait options")
    cache_eviction_operation, cache_eviction_bytes = (
        prepare_bf16_cache_eviction_sweep(
            torch_module=torch, device="cuda",
            cache_eviction_mib=event_options.cache_eviction_mib))
    buffer_contract = build_vector_buffer_contract(workload)
    if graph_repetitions is None:
        if len(operations) != 1:
            raise ValueError("rotating buffers require graph repetitions")
        measurement = measure_current_stream_events(
            operation, device_module=torch.cuda,
            warmup_iterations=event_options.warmup_iterations,
            sample_count=event_options.sample_count,
            cache_eviction_operation=cache_eviction_operation,
            cache_eviction_bytes=cache_eviction_bytes)
        timer_mode = "one_exact_public_op_per_event_interval"
    elif len(operations) == 1:
        if minimum_distinct_input_mib is not None:
            raise ValueError(
                "minimum distinct input requires rotating graph operations")
        measurement = measure_graph_replay_events(
            operation, device_module=torch.cuda,
            warmup_iterations=event_options.warmup_iterations,
            sample_count=event_options.sample_count,
            repetition_counts=graph_repetitions,
            cache_eviction_operation=cache_eviction_operation,
            cache_eviction_bytes=cache_eviction_bytes)
        timer_mode = "external_events_around_cuda_graph_replay"
    elif minimum_distinct_input_mib is None:
        measurement = measure_rotating_graph_replay_events(
            operations, device_module=torch.cuda,
            warmup_iterations=event_options.warmup_iterations,
            sample_count=event_options.sample_count,
            repetition_counts=graph_repetitions,
            cache_eviction_operation=cache_eviction_operation,
            cache_eviction_bytes=cache_eviction_bytes)
        timer_mode = "external_events_around_rotating_cuda_graph_replay"
    else:
        if event_options.cache_eviction_mib != 256:
            raise ValueError(
                "C500 cold Vector differential measurement requires a "
                "256 MiB queued sweep before every replay")
        minimum_distinct_input_bytes = minimum_distinct_input_mib * 1024 * 1024
        measurement = measure_differential_rotating_graph_replay_events(
            operations, device_module=torch.cuda,
            warmup_iterations=event_options.warmup_iterations,
            sample_count=event_options.sample_count,
            repetition_counts=graph_repetitions,
            per_call_distinct_input_bytes=buffer_contract[
                "per_call_distinct_input_bytes"],
            minimum_distinct_input_bytes=minimum_distinct_input_bytes,
            hardware_l2_bytes=C500_L2_BYTES,
            cache_eviction_operation=cache_eviction_operation,
            cache_eviction_bytes=cache_eviction_bytes)
        timer_mode = (
            "paired_differential_events_around_rotating_cuda_graph_replay")
    normalized_workload = dict(workload)
    normalized_workload["buffer_count"] = len(operations)
    normalized_workload.update(buffer_contract)
    normalized_workload["rotating_working_set_bytes"] = (
        buffer_contract["per_call_distinct_working_set_bytes"]
        * len(operations))
    if graph_repetitions is None or len(operations) == 1:
        rotation_mode = "none"
        selected_distinct_buffer_count = 1
    elif max(graph_repetitions) <= len(operations):
        rotation_mode = "unique_buffer_per_selected_replay_call"
        selected_distinct_buffer_count = max(graph_repetitions)
    else:
        rotation_mode = "complete_ring_before_reuse"
        selected_distinct_buffer_count = len(operations)
    normalized_workload["rotation_mode"] = rotation_mode
    normalized_workload["selected_replay_distinct_buffer_count"] = (
        selected_distinct_buffer_count)
    normalized_workload["selected_replay_distinct_working_set_bytes"] = (
        buffer_contract["per_call_distinct_working_set_bytes"]
        * selected_distinct_buffer_count)
    if minimum_distinct_input_mib is not None:
        normalized_workload["distinct_input_contract"] = measurement[
            "distinct_input_contract"]
    if graph_repetitions is None:
        normalized_workload["cache_eviction_bytes_before_sample"] = (
            cache_eviction_bytes)
    else:
        normalized_workload["cache_eviction_bytes_before_replay"] = (
            cache_eviction_bytes)
    write_event_case_record(
        options=event_options,
        hardware_id="metax_c500",
        family="vector",
        runner_path=Path(__file__),
        timer={
            "api": "torch.cuda.Event",
            "implementation": "wcudaEvent_to_maca_mcEvent",
            "returned_unit": "millisecond",
            "nanoseconds_per_unit": 1_000_000,
            "cupti_identity": "not_proven",
            "mode": timer_mode,
            "cache_condition": (
                "queued_pre_sample_device_sweep"
                if cache_eviction_bytes and graph_repetitions is None
                else (
                    "queued_pre_replay_device_sweep_without_intra_replay_"
                    "buffer_reuse"
                    if (cache_eviction_bytes and len(operations) > 1
                        and max(graph_repetitions) <= len(operations))
                    else (
                        "queued_pre_replay_device_sweep_with_intra_replay_reuse"
                        if cache_eviction_bytes
                        else (
                            "rotating_buffer_ring_without_explicit_sweep"
                            if len(operations) > 1
                            else "ordinary_warm_state")))),
        },
        stream_contract={
            "event_stream": "torch.cuda.current_stream",
            "all_op_streams_joined": True,
            "evidence": (
                "canonical case has one kernel and the runner/operator call "
                "contains no explicit secondary stream"),
        },
        allocation_contract={
            "inputs_preallocated": True,
            "output_behavior": "exact_runner_call",
            "workspace_behavior": "exact_runner_call",
            "evidence": allocation_evidence,
        },
        workload=normalized_workload,
        environment=_environment(),
        measurement=measurement,
    )

def _environment():
    properties = torch.cuda.get_device_properties(0)
    return {"torch_version": torch.__version__,
            "torch_cuda_version_field": torch.version.cuda,
            "device_index": 0, "device_name": properties.name}

def run_rmsnorm(tok, hidden, dtype="bf16", *, use_marker=False,
                profiler_wait=0, event_options=None, graph_repetitions=None,
                buffer_count=1, minimum_distinct_input_mib=None):
    dt = DTYPE_MAP[dtype]
    x = torch.randn(buffer_count, tok, hidden, dtype=dt, device="cuda")
    weight = torch.randn(buffer_count, hidden, dtype=dt, device="cuda")
    output = torch.empty_like(x)
    operations = tuple(
        lambda index=index: torch.ops._C.rms_norm(
            output[index], x[index].contiguous(), weight[index], 1e-6)
        for index in range(buffer_count))
    _run(
        operations,
        use_marker=use_marker, profiler_wait=profiler_wait,
        event_options=event_options,
        graph_repetitions=graph_repetitions,
        minimum_distinct_input_mib=minimum_distinct_input_mib,
        workload={"op": "rmsnorm", "tok": tok, "hidden": hidden,
                  "dtype": dtype, "dtype_bytes": x.element_size()},
        allocation_evidence="rms_norm uses indexed x, weight, and output")

def run_add_rmsnorm(tok, hidden, dtype="bf16", *, use_marker=False,
                    profiler_wait=0, event_options=None,
                    graph_repetitions=None, buffer_count=1,
                    minimum_distinct_input_mib=None):
    dt = DTYPE_MAP[dtype]
    x = torch.randn(buffer_count, tok, hidden, dtype=dt, device="cuda")
    residual = torch.randn(
        buffer_count, tok, hidden, dtype=dt, device="cuda")
    weight = torch.randn(buffer_count, hidden, dtype=dt, device="cuda")
    operations = tuple(
        lambda index=index: torch.ops._C.fused_add_rms_norm(
            x[index], residual[index], weight[index], 1e-6)
        for index in range(buffer_count))
    _run(
        operations,
        use_marker=use_marker, profiler_wait=profiler_wait,
        event_options=event_options,
        graph_repetitions=graph_repetitions,
        minimum_distinct_input_mib=minimum_distinct_input_mib,
        workload={"op": "add_rmsnorm", "tok": tok, "hidden": hidden,
                  "dtype": dtype, "dtype_bytes": x.element_size()},
        allocation_evidence=("fused_add_rms_norm uses indexed x, residual, "
                             "and weight tensors in place"))

def run_rope(tok, num_heads, num_kv_heads, head_dim, dtype="bf16", *,
             use_marker=False, profiler_wait=0, event_options=None,
             graph_repetitions=None, buffer_count=1,
             minimum_distinct_input_mib=None):
    dt = DTYPE_MAP[dtype]
    q = torch.randn(
        buffer_count, tok, num_heads, head_dim, dtype=dt, device="cuda")
    k = torch.randn(
        buffer_count, tok, num_kv_heads, head_dim,
        dtype=dt, device="cuda")
    positions = torch.arange(
        tok, dtype=torch.int64, device="cuda").repeat(buffer_count, 1)
    theta = 1.0 / (10000 ** (
        torch.arange(
            0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim))
    frequencies = (
        torch.arange(tok, device="cuda", dtype=torch.float32).unsqueeze(1)
        * theta)
    cos_sin_cache = torch.cat(
        [frequencies.cos(), frequencies.sin()], dim=-1).to(dt)
    cos_sin_cache = cos_sin_cache.repeat(buffer_count, 1, 1)
    q_flat = q.reshape(buffer_count, tok, -1)
    k_flat = k.reshape(buffer_count, tok, -1)
    operations = tuple(
        lambda index=index: torch.ops._C.rotary_embedding(
            positions[index], q_flat[index], k_flat[index], head_dim,
            cos_sin_cache[index], True)
        for index in range(buffer_count))
    _run(
        operations,
        use_marker=use_marker, profiler_wait=profiler_wait,
        event_options=event_options,
        graph_repetitions=graph_repetitions,
        minimum_distinct_input_mib=minimum_distinct_input_mib,
        workload={
            "op": "rope", "tok": tok, "num_heads": num_heads,
            "num_kv_heads": num_kv_heads, "head_dim": head_dim,
            "dtype": dtype, "dtype_bytes": q.element_size(),
        },
        allocation_evidence=("rotary_embedding uses indexed positions, q, k, "
                             "and cos/sin cache tensors in place"))


def run_silu(tok, intermediate, dtype="bf16", *, use_marker=False,
             profiler_wait=0, event_options=None, graph_repetitions=None,
             buffer_count=1, minimum_distinct_input_mib=None):
    dt = DTYPE_MAP[dtype]
    input_tensor = torch.randn(
        buffer_count, tok, 2 * intermediate, dtype=dt, device="cuda")
    output = torch.empty(
        buffer_count, tok, intermediate, dtype=dt, device="cuda")
    operations = tuple(
        lambda index=index: torch.ops._C.silu_and_mul(
            output[index], input_tensor[index])
        for index in range(buffer_count))
    _run(
        operations,
        use_marker=use_marker, profiler_wait=profiler_wait,
        event_options=event_options,
        graph_repetitions=graph_repetitions,
        minimum_distinct_input_mib=minimum_distinct_input_mib,
        workload={"op": "silu", "tok": tok, "intermediate": intermediate,
                  "dtype": dtype, "dtype_bytes": output.element_size()},
        allocation_evidence="silu_and_mul uses indexed input and output")


def run_elementwise(tok, hidden, op2, dtype="bf16", *,
                    use_marker=False, profiler_wait=0, event_options=None,
                    graph_repetitions=None, buffer_count=1,
                    minimum_distinct_input_mib=None):
    dt = DTYPE_MAP[dtype]
    left = torch.randn(
        buffer_count, tok, hidden, dtype=dt, device="cuda")
    right = torch.randn(
        buffer_count, tok, hidden, dtype=dt, device="cuda")
    output = torch.empty_like(left)
    if op2 == "add":
        operations = tuple(
            lambda index=index: torch.add(
                left[index], right[index], out=output[index])
            for index in range(buffer_count))
    elif op2 == "mul":
        operations = tuple(
            lambda index=index: torch.mul(
                left[index], right[index], out=output[index])
            for index in range(buffer_count))
    else:
        raise ValueError(f"unsupported elementwise operation: {op2!r}")
    _run(
        operations,
        use_marker=use_marker,
        profiler_wait=profiler_wait,
        event_options=event_options,
        graph_repetitions=graph_repetitions,
        minimum_distinct_input_mib=minimum_distinct_input_mib,
        workload={
            "op": "elementwise", "op2": op2, "tok": tok,
            "hidden": hidden, "dtype": dtype,
            "dtype_bytes": left.element_size(),
        },
        allocation_evidence=("torch.add/torch.mul uses indexed left, right, "
                             "and preallocated output tensors"),
    )


def run_kvcache(tok, num_kv_heads, head_dim, dtype="bf16", *,
                use_marker=False, profiler_wait=0, event_options=None,
                graph_repetitions=None, buffer_count=1,
                minimum_distinct_input_mib=None):
    dt = DTYPE_MAP[dtype]
    key = torch.randn(
        buffer_count, tok, num_kv_heads, head_dim,
        dtype=dt, device="cuda")
    value = torch.randn_like(key)
    block_size = 128
    total_slots = buffer_count * tok
    num_blocks = max(1, (total_slots + block_size - 1) // block_size)
    key_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, head_dim,
        dtype=dt, device="cuda")
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.arange(
        total_slots, dtype=torch.int64, device="cuda").reshape(
            buffer_count, tok)
    key_scale = torch.ones(buffer_count, device="cuda")
    value_scale = torch.ones(buffer_count, device="cuda")
    operations = tuple(
        lambda index=index: torch.ops._C_cache_ops.reshape_and_cache_flash(
            key[index], value[index], key_cache, value_cache,
            slot_mapping[index], "auto", key_scale[index], value_scale[index])
        for index in range(buffer_count))
    buffer_contract = build_vector_buffer_contract({
        "op": "kvcache", "tok": tok, "num_kv_heads": num_kv_heads,
        "head_dim": head_dim, "dtype_bytes": key.element_size()})
    _run(
        operations,
        use_marker=use_marker, profiler_wait=profiler_wait,
        event_options=event_options,
        graph_repetitions=graph_repetitions,
        minimum_distinct_input_mib=minimum_distinct_input_mib,
        workload={
            "op": "kvcache", "tok": tok,
            "num_kv_heads": num_kv_heads, "head_dim": head_dim,
            "dtype": dtype, "dtype_bytes": key.element_size(),
            "cache_backing_allocated_bytes": (
                key_cache.numel() * key_cache.element_size()
                + value_cache.numel() * value_cache.element_size()),
            "unique_written_address_bytes_per_call": (
                buffer_contract["per_call_distinct_output_storage_bytes"]),
            "slot_mapping_policy": (
                "contiguous_nonoverlapping_ranges_in_shared_cache_backing"),
        },
        allocation_evidence=("reshape_and_cache_flash uses indexed key, value, "
                             "slot mapping, and scales with disjoint cache slots"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op", required=True,
        choices=("rmsnorm", "add_rmsnorm", "rope", "silu",
                 "elementwise", "kvcache"))
    parser.add_argument("--tok", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--head_dim", type=int, required=True)
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--num_kv_heads", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--dtype", required=True, choices=sorted(DTYPE_MAP))
    parser.add_argument("--op2", required=True, choices=("add", "mul"))
    parser.add_argument("--profiler_wait", type=int, default=0)
    parser.add_argument("--use_marker", action="store_true")
    add_event_cli_arguments(parser)
    parser.add_argument("--event-graph-repetitions", nargs=2, type=int)
    parser.add_argument("--event-buffer-count", type=int, default=1)
    parser.add_argument("--event-minimum-distinct-input-mib", type=int)
    args = parser.parse_args()
    event_options = parse_event_run_options(args)
    graph_repetitions = None
    if args.event_graph_repetitions is not None:
        if event_options is None:
            raise ValueError(
                "event_graph_repetitions requires complete Event options")
        graph_repetitions = tuple(args.event_graph_repetitions)
    if args.event_buffer_count <= 0:
        raise ValueError("event_buffer_count must be positive")
    if args.event_buffer_count != 1 and graph_repetitions is None:
        raise ValueError(
            "event_buffer_count greater than one requires graph repetitions")
    if args.event_buffer_count != 1:
        if (event_options.cache_eviction_mib == 0
                and min(graph_repetitions) < args.event_buffer_count):
            raise ValueError(
                "every rotating graph must traverse the complete buffer "
                "cycle: each graph repetition count must be at least "
                "event_buffer_count")
        if (event_options.cache_eviction_mib != 0
                and max(graph_repetitions) > args.event_buffer_count):
            raise ValueError(
                "sweep-assisted rotating graphs must not reuse an operator "
                "buffer within one replay: event_buffer_count must be at "
                "least each graph repetition count")
    if args.event_minimum_distinct_input_mib is not None:
        if graph_repetitions is None or args.event_buffer_count == 1:
            raise ValueError(
                "event_minimum_distinct_input_mib requires rotating graph "
                "repetitions")
        if args.event_minimum_distinct_input_mib < 64:
            raise ValueError(
                "event_minimum_distinct_input_mib must be at least 64 "
                "(8x C500 L2)")
        if args.event_minimum_distinct_input_mib % 8 != 0:
            raise ValueError(
                "event_minimum_distinct_input_mib must be an integer multiple "
                "of the 8 MiB C500 L2")

    common = {
        "use_marker": args.use_marker,
        "profiler_wait": args.profiler_wait,
        "event_options": event_options,
        "graph_repetitions": graph_repetitions,
        "buffer_count": args.event_buffer_count,
        "minimum_distinct_input_mib": args.event_minimum_distinct_input_mib,
    }
    if args.op == "rmsnorm":
        run_rmsnorm(args.tok, args.hidden, args.dtype, **common)
    elif args.op == "add_rmsnorm":
        run_add_rmsnorm(args.tok, args.hidden, args.dtype, **common)
    elif args.op == "rope":
        run_rope(
            args.tok, args.num_heads, args.num_kv_heads, args.head_dim,
            args.dtype, **common)
    elif args.op == "silu":
        run_silu(args.tok, args.intermediate, args.dtype, **common)
    elif args.op == "elementwise":
        run_elementwise(
            tok=args.tok, hidden=args.hidden, op2=args.op2,
            dtype=args.dtype, **common)
    elif args.op == "kvcache":
        if args.batch != 1:
            raise ValueError(
                "KV-cache batch is not a physical runner dimension; expected 1")
        run_kvcache(
            args.tok, args.num_kv_heads, args.head_dim, args.dtype, **common)
    else:
        raise AssertionError(f"argparse accepted an unknown op: {args.op!r}")


if __name__ == "__main__":
    main()
