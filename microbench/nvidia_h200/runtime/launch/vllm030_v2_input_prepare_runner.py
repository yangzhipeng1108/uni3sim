"""Independent V2 runner preparation with synthetic state, no model or sampler.

CUDA mode calls three unmodified installed-source methods and real V2 buffers,
request state and block tables. CPU mode validates the fixture with explicit
tensor-operation shims; its timings must never be used as hardware prices.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import statistics
import textwrap
import time
from types import SimpleNamespace

import numpy as np
import torch

METHODS = ("gather_batch_req_state", "prepare_inputs", "prepare_attn")
CELLS = ((1, 1), (8, 1), (32, 1), (128, 1), (1, 32), (8, 32),
         (32, 32), (1, 256), (8, 256))


def cpu_namespace():
    """Fixture-only operators; the CUDA measurement never enters these shims."""
    def h2d(array, *, device=None, out=None):
        tensor = torch.from_numpy(array.copy())
        return tensor if out is None else out.copy_(tensor)

    def prefill(ids, next_ids, mapping, starts, all_ids, lengths, computed):
        for row, slot in enumerate(mapping.tolist()):
            start, end = starts[row:row + 2].tolist()
            pos = int(computed[slot])
            ids[start:end] = all_ids[slot, pos:pos + end - start]

    def positions(mapping, starts, computed, output, lengths):
        lengths.zero_()
        for row, slot in enumerate(mapping.tolist()):
            start, end = starts[row:row + 2].tolist()
            output[start:end] = torch.arange(int(computed[slot]),
                                             int(computed[slot]) + end - start)
            lengths[row] = computed[slot] + end - start

    def combine(ids, mapping, last, starts, lengths, prefill_len, drafts, cumulative,
                total, new_tokens):
        assert drafts.shape[1] == 0 and total == len(mapping) and new_tokens == 1
        indices = starts[1:len(mapping) + 1].to(torch.int64) - 1
        for row, slot in enumerate(mapping.tolist()):
            if int(lengths[row]) > int(prefill_len[slot]):
                ids[indices[row]] = last[slot, 0]
        return indices

    return {"np": np, "torch": torch, "InputBatch": SimpleNamespace,
            "BatchReqState": SimpleNamespace, "async_tensor_h2d": h2d,
            "sort_batch_req_ids": lambda counts, drafts, length: sorted(counts),
            "get_uniform_decode_token_count": lambda *args, **kwargs: None,
            "prepare_prefill_inputs": prefill, "prepare_pos_seq_lens": positions,
            "combine_sampled_and_draft_tokens": combine,
            "envs": SimpleNamespace(VLLM_MOE_SKIP_PADDING=False),
            "pcp": SimpleNamespace(maybe_partition_pcp_batch=lambda manager, batch, desc: batch)}


def source_class(path, cuda):
    source = path.read_text()
    tree = ast.parse(source)
    owner, = (node for node in tree.body if isinstance(node, ast.ClassDef)
              and node.name == "GPUModelRunner")
    definitions = {node.name: ast.get_source_segment(source, node)
                   for node in owner.body if isinstance(node, ast.FunctionDef)
                   and node.name in METHODS}
    if set(definitions) != set(METHODS):
        raise ValueError("three required V2 methods missing")
    namespace = (dict(vars(importlib.import_module("vllm.v1.worker.gpu.model_runner")))
                 if cuda else cpu_namespace())
    code = "from __future__ import annotations\n" + "\n\n".join(
        textwrap.dedent("    " + definitions[name]) for name in METHODS)
    exec(compile(code, str(path), "exec"), namespace)
    return (type("SyntheticV2Preparation", (), {name: namespace[name] for name in METHODS}),
            hashlib.sha256(source.encode()).hexdigest(),
            {name: hashlib.sha256(definitions[name].encode()).hexdigest() for name in METHODS})


def fixture(cls, requests, query, device):
    capacity, tokens, max_len = 256, 8192, 8192
    has_prefill = query > 1
    prompt_len = 1024 + query + 2 if has_prefill else 32
    names = [f"synthetic-{i:03d}" for i in range(requests)]
    if device.type == "cuda":
        from vllm.v1.worker.gpu.block_table import BlockTables
        from vllm.v1.worker.gpu.input_batch import InputBuffers
        from vllm.v1.worker.gpu.states import RequestState
        state = RequestState(capacity, max_len, tokens, 0, 256, device)
        blocks = BlockTables([16], capacity, tokens, [max_len // 16], device, [16])
        for i, name in enumerate(names):
            state.add_request(name, prompt_len, [i + 1] * prompt_len, 1024, 256)
            slot = state.req_id_to_index[name]
            state.last_sampled_tokens[slot, 0] = 17
            blocks.append_block_ids(slot, (list(range(i * 512, (i + 1) * 512)),), True)
        state.apply_staged_writes()
        blocks.apply_staged_writes()
        buffers = InputBuffers(capacity, tokens, device)
    else:
        all_ids = torch.zeros((capacity, max_len), dtype=torch.int32)
        for i in range(requests):
            all_ids[i, :prompt_len] = i + 1
        state = SimpleNamespace(
            req_id_to_index={name: i for i, name in enumerate(names)},
            prefill_len=SimpleNamespace(np=np.full(capacity, prompt_len, dtype=np.int32),
                                       gpu=torch.full((capacity,), prompt_len, dtype=torch.int32)),
            num_computed_prefill_tokens=np.full(capacity, 1024 if has_prefill else 32, dtype=np.int32),
            num_computed_tokens_np=np.full(capacity, 1024, dtype=np.int32),
            num_computed_tokens=SimpleNamespace(gpu=torch.full((capacity,), 1024, dtype=torch.int32)),
            all_token_ids=SimpleNamespace(gpu=all_ids),
            last_sampled_tokens=torch.full((capacity, 1), 17, dtype=torch.int64),
            draft_tokens=torch.zeros((capacity, 0), dtype=torch.int64),
            next_prefill_tokens=torch.zeros((1, capacity), dtype=torch.int32))
        buffers = SimpleNamespace(
            input_ids=torch.zeros(tokens, dtype=torch.int32), positions=torch.zeros(tokens, dtype=torch.int64),
            is_padding=torch.zeros(tokens, dtype=torch.bool), query_start_loc=torch.zeros(capacity + 1, dtype=torch.int32),
            seq_lens=torch.zeros(capacity, dtype=torch.int32))
        blocks = SimpleNamespace(
            gather_block_tables=lambda mapping, num_reqs_padded: (),
            compute_slot_mappings=lambda mapping, starts, positions, num_tokens_padded: None)
    runner = cls()
    runner.device, runner.max_num_reqs = device, capacity
    runner.input_buffers, runner.req_states, runner.block_tables = buffers, state, blocks
    runner.adaptive_verification, runner.fast_prefill, runner.pcp_manager = None, None, None
    runner.decode_query_len = 1
    runner.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1)
    runner.model_config = SimpleNamespace(rswa_window=None)
    output = SimpleNamespace(num_scheduled_tokens={name: query for name in names},
                             total_num_scheduled_tokens=requests * query,
                             scheduled_spec_decode_tokens={}, has_structured_output_requests=False)
    desc = SimpleNamespace(num_tokens=requests * query, num_reqs=requests if not has_prefill else None)
    return runner, output, desc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 4:
        raise ValueError("existing output or insufficient samples")
    path = args.source_root / "vllm/v1/worker/gpu/model_runner.py"
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    if cuda and "H200" not in torch.cuda.get_device_name(0):
        raise ValueError("requires authorized H200")
    cls, source_sha, method_shas = source_class(path, cuda)
    rows = []
    for requests, query in CELLS:
        runner, output, desc = fixture(cls, requests, query, device)
        host, completion, event = [], [], []
        for sample in range(12 + args.samples):
            if cuda:
                torch.cuda.synchronize()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
            start = time.perf_counter_ns()
            state, _ = runner.gather_batch_req_state(output, False)
            batch = runner.prepare_inputs(output, state, desc)
            block_tables, slots = runner.prepare_attn(batch)
            enqueued = time.perf_counter_ns()
            if cuda:
                end.record()
                end.synchronize()
            stop = time.perf_counter_ns()
            if sample >= 12:
                host.append(enqueued - start)
                completion.append(stop - start)
                if cuda:
                    event.append(begin.elapsed_time(end) * 1e6)
        assert batch.logits_indices.tolist() == [(i + 1) * query - 1 for i in range(requests)]
        assert batch.positions.tolist() == [1024 + j for i in range(requests) for j in range(query)]
        assert batch.input_ids.tolist() == ([17] * requests if query == 1 else
                                           [i + 1 for i in range(requests) for j in range(query)])
        assert batch.seq_lens.tolist() == [1024 + query] * requests
        if cuda:
            expected = [i * 8192 + 1024 + j for i in range(requests) for j in range(query)]
            assert slots[0, :requests * query].tolist() == expected
            assert block_tables[0][:, 0].tolist() == [i * 512 for i in range(requests)]
        rows.append({"requests": requests, "query_tokens_each": query,
                     "host_enqueue_ns": host, "synchronized_completion_ns": completion,
                     "cuda_event_ns": event, "median_host_enqueue_ns": statistics.median(host),
                     "median_completion_ns": statistics.median(completion)})
        print(json.dumps({k: v for k, v in rows[-1].items() if not isinstance(v, list)}), flush=True)
    payload = {"kind": "independent_vllm030_v2_input_preparation", "cpu_preflight_only": not cuda,
               "production_duration_consumed": False, "model_or_sampler_executed": False,
               "source_path": str(path), "source_sha256": source_sha, "method_sha256": method_shas,
               "measurement_boundary": "gather_batch_req_state + prepare_inputs + prepare_attn; no model/weights",
               "limitations": ["excludes add/update/free requests, attention metadata builder, sampler and output processing",
                               "completion and CUDA-event windows include CPU enqueue idle, not pure device service",
                               "synthetic independent state; native provider and historical CPU equivalence unbound"],
               "torch": torch.__version__, "host_platform": platform.platform(),
               "cpuinfo_first": Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0],
               "affinity": sorted(os.sched_getaffinity(0)),
               "gpu": torch.cuda.get_device_name(0) if cuda else None, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
