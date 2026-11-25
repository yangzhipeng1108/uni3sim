"""Separate pure V2 request-state admission, finish and cached-update Host work.

All request data are synthetic. Actual native source methods and buffers run;
there is no model, input preparation, attention metadata, logits or sampling.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from vllm030_v2_state_metadata_runner import source_check


COUNTS = (1, 8, 32, 128)
PROMPTS = (32, 256, 1024)
CELLS = (("empty", 0, 0, 0, 32),
    *(("cached", count, 0, 0, 32) for count in COUNTS),
    *(("finish", 0, count, 0, 32) for count in COUNTS),
    *(("admit", 0, 0, count, prompt) for prompt in PROMPTS for count in COUNTS),
    ("mixed", 32, 1, 1, 256), ("mixed", 32, 8, 8, 256),
    ("mixed", 32, 1, 1, 1024), ("mixed", 32, 8, 8, 1024))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-check-only", action="store_true")
    parser.add_argument("--rotation", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    hashes = source_check(args.source_root)
    if args.source_check_only:
        args.output.write_text(json.dumps({"source_sha256": hashes, "cells": CELLS,
            "timing_not_executed": True, "cuda_initialized": torch.cuda.is_initialized()}, indent=2) + "\n")
        return
    if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0):
        raise ValueError("requires authorized H200")

    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.worker.gpu.lora_utils import LoraState
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState
    from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    import vllm030_v2_input_prepare_runner as fixture_source

    if Path(inspect.getsourcefile(GPUModelRunner)).resolve() != (args.source_root / "vllm/v1/worker/gpu/model_runner.py").resolve():
        raise ValueError("installed source differs")
    cls, _, _ = fixture_source.source_class(args.source_root / "vllm/v1/worker/gpu/model_runner.py", True)
    device = torch.device("cuda")
    rows = []
    cells = CELLS if args.rotation == 0 else tuple(reversed(CELLS))
    for mode, cached_count, finish_count, new_count, prompt in cells:
        runner, _, _ = fixture_source.fixture(cls, 0, 1, device)
        for name in ("finish_requests", "free_states", "add_requests", "update_requests", "_remove_request"):
            setattr(runner, name, getattr(GPUModelRunner, name).__get__(runner))
        state = object.__new__(DefaultModelState)
        state.rope_state = state.prompt_embeds_state = None
        state.num_new_sampled_tokens_per_step = 1
        runner.model_state = state
        runner.is_last_pp_rank = True
        runner.pooling_runner = runner.pp_handler = runner.encoder_cache = None
        runner.adaptive_verification = None
        runner.lora_state = LoraState(256)
        runner.prompt_logprobs_worker = PromptLogprobsWorker(256)
        runner.sampler = Sampler(256, 256, device, runner.req_states, num_speculative_tokens=0)
        sampling = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)

        def output(new_records, finished_ids, cached_ids):
            return SimpleNamespace(scheduled_new_reqs=new_records, finished_req_ids=set(finished_ids),
                preempted_req_ids=set(), free_encoder_mm_hashes=[], new_block_ids_to_zero=[],
                kv_cache_block_copies=[], block_table_updates=None,
                scheduled_cached_reqs=SimpleNamespace(req_ids=cached_ids,
                    num_computed_tokens=[prompt] * len(cached_ids), new_block_ids=[None] * len(cached_ids)))

        def records(names, computed_tokens):
            blocks = (prompt + 15) // 16
            return [NewRequestData(name, [index % 255 + 1] * prompt, [], sampling, None,
                (list(range(index * 512, index * 512 + blocks)),), computed_tokens, None,
                prefill_token_ids=[index % 255 + 1] * prompt) for index, name in enumerate(names)]

        samples = []
        for trial in range(12 + 32):
            # Clear/seed outside the timer. Admit-only never includes finish;
            # finish-only never includes a source add or a reused input buffer.
            old_ids = tuple(runner.req_states.req_id_to_index)
            runner.finish_requests(output([], old_ids, []))
            resident = [f"resident-{trial}-{index}" for index in range(cached_count + finish_count)]
            runner.add_requests(output(records(resident, prompt), (), []))
            runner.block_tables.apply_staged_writes()
            cached = resident[:cached_count]
            finished = resident[cached_count:]
            fresh = [f"fresh-{trial}-{index}" for index in range(new_count)]
            request = output(records(fresh, 0), finished, cached)
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            runner.finish_requests(request)
            after_finish = time.perf_counter_ns()
            runner.free_states(request)
            runner.add_requests(request)
            after_add = time.perf_counter_ns()
            runner.update_requests(request)
            runner.block_tables.apply_staged_writes()
            after_update = time.perf_counter_ns()
            torch.cuda.synchronize()
            expected = set(cached + fresh)
            if set(runner.req_states.req_id_to_index) != expected:
                raise ValueError("source lifecycle produced wrong resident IDs")
            for record in request.scheduled_new_reqs:
                index = runner.req_states.req_id_to_index[record.req_id]
                if runner.req_states.all_token_ids.gpu[index, :prompt].cpu().tolist() != record.prefill_token_ids:
                    raise ValueError("staged token identity changed")
                if runner.req_states.num_computed_tokens.gpu[index].item() != 0:
                    raise ValueError("new request computed-token state changed")
            if trial >= 12:
                samples.append({"finish_ns": after_finish - started, "add_ns": after_add - after_finish,
                    "update_ns": after_update - after_add, "state_ns": after_update - started})
        row = {"mode": mode, "cached_requests": cached_count, "finished_requests": finish_count,
            "new_requests": new_count, "new_prompt_tokens_each": prompt, "samples": samples,
            "median_ns": {name: statistics.median(value[name] for value in samples)
                for name in ("finish_ns", "add_ns", "update_ns", "state_ns")}}
        rows.append(row)
        print(json.dumps({name: value for name, value in row.items() if name != "samples"}), flush=True)
        args.output.write_text(json.dumps({"status": "partial", "rows": rows}, indent=2) + "\n")
    if source_check(args.source_root) != hashes:
        raise ValueError("source changed during state mechanisms")
    args.output.write_text(json.dumps({"status": "complete", "kind": __doc__, "rotation": args.rotation,
        "source_sha256": hashes, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_fixture_sha256": hashlib.sha256(Path(fixture_source.__file__).read_bytes()).hexdigest(),
        "model_weights_or_attention_or_input_or_metadata_or_sampling_math_executed": False,
        "production_duration_consumed": False, "rows": rows,
        "torch": torch.__version__, "cpuinfo_first": Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0],
        "limitations": ["source-ready runner/model-state constructors bypassed",
            "greedy ordinary sampler state only; no sampler invocation or logprobs",
            "no block append/copy-on-write/zeroing; synthetic no-hit initial token state",
            "Host enqueue timing only; no GPU duration price or completion fence"]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
