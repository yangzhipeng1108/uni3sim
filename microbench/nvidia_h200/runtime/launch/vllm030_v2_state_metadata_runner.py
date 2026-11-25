"""Independent V2 request-state and FA3 metadata Host preparation, no model.

Actual source methods and data structures run on synthetic request state. No
Attention forward, logits or sampling kernel executes. Timers separate the
already measured input primitive from the two missing Host boundaries.
"""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_check(root):
    files = ("vllm/v1/worker/gpu/model_runner.py", "vllm/v1/worker/gpu/states.py",
        "vllm/v1/worker/gpu/block_table.py", "vllm/v1/worker/gpu/model_states/default.py",
        "vllm/v1/attention/backends/flash_attn.py", "vllm/v1/attention/backend.py",
        "vllm/v1/worker/gpu/sample/sampler.py", "vllm/v1/worker/gpu/lora_utils.py",
        "vllm/v1/worker/gpu/sample/prompt_logprob.py")
    tree = ast.parse((root / files[0]).read_text())
    owner, = (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner")
    methods = {node.name for node in owner.body if isinstance(node, ast.FunctionDef)}
    required = {"finish_requests", "free_states", "add_requests", "update_requests",
        "gather_batch_req_state", "prepare_inputs", "prepare_attn"}
    if not required <= methods:
        raise ValueError("required V2 source preparation methods missing")
    return {name: sha(root / name) for name in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-check-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_hashes = source_check(args.source_root)
    if args.source_check_only:
        args.output.write_text(json.dumps({"source_checked": True, "cuda_initialized": torch.cuda.is_initialized(),
            "source_sha256": source_hashes, "no_timing_parameter": True}, indent=2) + "\n")
        return
    if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0):
        raise ValueError("requires the authorized single H200")

    import vllm
    from vllm.sampling_params import SamplingParams
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.worker.gpu.lora_utils import LoraState
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState
    from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    import vllm030_v2_input_prepare_runner as input_probe

    if Path(inspect.getsourcefile(GPUModelRunner)).resolve() != (args.source_root / "vllm/v1/worker/gpu/model_runner.py").resolve():
        raise ValueError("imported V2 source differs from the selected installed source")
    device = torch.device("cuda")
    cls, _, _ = input_probe.source_class(args.source_root / "vllm/v1/worker/gpu/model_runner.py", True)
    rows = []
    method_names = ("finish_requests", "free_states", "add_requests", "update_requests", "_remove_request")
    for mode in ("cached", "replace"):
        for count, query in input_probe.CELLS:
            runner, output, desc = input_probe.fixture(cls, count, query, device)
            for name in method_names:
                setattr(runner, name, getattr(GPUModelRunner, name).__get__(runner))
            state = object.__new__(DefaultModelState)
            state.rope_state, state.prompt_embeds_state = None, None
            state.num_new_sampled_tokens_per_step = 1
            runner.model_state = state
            runner.is_last_pp_rank = True
            runner.pooling_runner = runner.pp_handler = runner.encoder_cache = None
            runner.adaptive_verification = None
            runner.lora_state = LoraState(256)
            runner.prompt_logprobs_worker = PromptLogprobsWorker(256)
            runner.sampler = Sampler(256, 256, device, runner.req_states, num_speculative_tokens=0)
            sampling = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)

            # Source-ready metadata state, not constructor/provider certification.
            # The bound ordinary D128/FA3 path has DCP1, no cascade/MM/R-SWA.
            builder = object.__new__(FlashAttentionMetadataBuilder)
            builder.aot_schedule = True
            builder.aot_sliding_window = (-1, -1)
            builder.use_full_cuda_graph = True
            builder.max_cudagraph_size = 512
            builder.max_num_splits = 32
            builder.dcp_world_size, builder.dcp_rank = 1, 0
            builder.num_heads_q, builder.num_heads_kv, builder.headdim, builder.block_size = 32, 8, 128, 16
            builder.kv_cache_dtype = torch.bfloat16
            builder.cache_config = SimpleNamespace(cache_dtype="auto")
            builder.scheduler_metadata = torch.zeros(2049, dtype=torch.int32, device=device)
            builder.rswa_window = builder.mm_prefix_query_ranges_np = None
            samples = []
            metadata_extent = None
            names = list(output.num_scheduled_tokens)
            for trial in range(12 + 64):
                cached_ids = names if mode == "cached" else []
                output.finished_req_ids = set(names) if mode == "replace" else set()
                output.preempted_req_ids = set()
                output.free_encoder_mm_hashes = []
                output.new_block_ids_to_zero = []
                output.kv_cache_block_copies = []
                output.block_table_updates = None
                output.scheduled_cached_reqs = SimpleNamespace(req_ids=cached_ids,
                    num_computed_tokens=[1024] * len(cached_ids), new_block_ids=[None] * len(cached_ids))
                if mode == "replace":
                    names = [f"independent-{trial}-{index}" for index in range(count)]
                    prompt_len = 1024 + query + 2 if query > 1 else 32
                    output.scheduled_new_reqs = [NewRequestData(name, [index + 1] * prompt_len,
                        [], sampling, None, (list(range(index * 512, (index + 1) * 512)),), 1024, None,
                        prefill_token_ids=[index + 1] * prompt_len) for index, name in enumerate(names)]
                    output.num_scheduled_tokens = {name: query for name in names}
                else:
                    output.scheduled_new_reqs = []
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                runner.finish_requests(output)
                runner.free_states(output)
                runner.add_requests(output)
                runner.update_requests(output)
                runner.block_tables.apply_staged_writes()
                state_end = time.perf_counter_ns()
                batch_state, _ = runner.gather_batch_req_state(output, False)
                batch = runner.prepare_inputs(output, batch_state, desc)
                blocks, slots = runner.prepare_attn(batch)
                input_end = time.perf_counter_ns()
                common = CommonAttentionMetadata(query_start_loc=batch.query_start_loc[:count + 1],
                    query_start_loc_cpu=torch.from_numpy(batch.query_start_loc_np[:count + 1]),
                    seq_lens=batch.seq_lens, num_reqs=count, num_actual_tokens=count * query,
                    max_query_len=query, max_seq_len=1024 + query,
                    block_table_tensor=blocks[0], slot_mapping=slots[0, :count * query])
                metadata = builder.build(0, common)
                metadata_end = time.perf_counter_ns()
                torch.cuda.synchronize()
                if trial >= 12:
                    samples.append({"state_host_ns": state_end - start, "input_host_ns": input_end - state_end,
                        "metadata_host_ns": metadata_end - input_end, "total_host_ns": metadata_end - start})
                if len(runner.req_states.req_id_to_index) != count or metadata.num_actual_tokens != count * query:
                    raise ValueError("state/metadata result count changed")
                if metadata.query_start_loc.tolist() != [index * query for index in range(count + 1)]:
                    raise ValueError("metadata query offsets changed")
                if batch.positions.tolist() != [1024 + index for _ in range(count) for index in range(query)]:
                    raise ValueError("prepared positions changed")
                value = metadata.scheduler_metadata
                if value is None or value.dtype != torch.int32 or value.ndim != 1 or not 0 < value.numel() <= builder.scheduler_metadata.numel():
                    raise ValueError("FA3 metadata must fit its actual typed source-owned storage")
                if metadata_extent is None:
                    metadata_extent = value.numel()
                    print(json.dumps({"mode": mode, "requests": count, "query": query,
                        "actual_metadata_extent": metadata_extent, "source_comment_extent": 1 + ((count + 3) // 4) * 16}), flush=True)
                elif value.numel() != metadata_extent:
                    raise ValueError("FA3 metadata extent changed within one fixed cell")
                if torch.count_nonzero(builder.scheduler_metadata[metadata_extent:]).item() != 0:
                    raise ValueError("source metadata tail clearing failed")
                if value.data_ptr() != builder.scheduler_metadata.data_ptr():
                    raise ValueError("metadata does not use the source-owned replay buffer")
            rows.append({"mode": mode, "requests": count, "query_tokens_each": query, "samples": samples,
                "actual_metadata_extent": metadata_extent,
                "median_ns": {name: statistics.median(row[name] for row in samples)
                    for name in ("state_host_ns", "input_host_ns", "metadata_host_ns", "total_host_ns")}})
            print(json.dumps({name: value for name, value in rows[-1].items() if name != "samples"}), flush=True)
            args.output.write_text(json.dumps({"status": "partial", "rows": rows}, indent=2) + "\n")
    if source_check(args.source_root) != source_hashes:
        raise ValueError("source changed during the mechanism measurement")
    payload = {"status": "complete", "kind": "independent_v2_state_and_metadata_host_preparation", "production_duration_consumed": False,
        "model_weights_or_attention_or_logits_or_sampling_executed": False,
        "boundary": "finish/free/add/update/staged writes; existing input primitive; ordinary FA3 metadata build, separately timed",
        "fixture": "synthetic ready V2 state; source ModelRunner/ModelState constructors bypassed; real Sampler state only",
        "source_sha256": source_hashes, "runner_sha256": sha(Path(__file__)),
        "input_fixture_sha256": sha(Path(input_probe.__file__)), "torch": torch.__version__, "vllm": vllm.__version__,
        "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
        "cpuinfo_first": Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0],
        "affinity": sorted(os.sched_getaffinity(0)), "rows": rows,
        "limitations": ["Host enqueue only; no GPU duration pricing or completion-fence parameter",
            "source-ready synthetic objects do not certify actual provider or target-host equality",
            "metadata D128 QH32/KVH8/page16/DCP1 only; no cascade/MM/R-SWA; no numerical Attention output",
            "replace includes finish and re-admission, not an independently identified pure add price",
            "cache zeroing/copy-on-write paths are explicitly empty; metadata Device work is already represented separately"]}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
