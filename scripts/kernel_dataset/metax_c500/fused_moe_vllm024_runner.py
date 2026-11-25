"""Capture vLLM-metax 0.24 FusedMoE launch geometry on C500.

This runner deliberately calls the deployment package's unquantized BF16
implementation. It is not a replacement for the canonical dataset runner:
it records configuration and launch facts for an EP rank before its output
AllReduce, so rank-local output values are not an end-to-end correctness
oracle.
"""

import argparse
import heapq
import json
import math
from pathlib import Path
import time


def _positive_integer(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _route_ids(counts: list[int], tokens: int, topk: int, torch):
    if len(counts) != 128:
        raise ValueError("route counts must contain exactly 128 global experts")
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0
           for count in counts):
        raise ValueError("route counts must be nonnegative integers")
    if sum(counts) != tokens * topk:
        raise ValueError("route-count sum must equal tokens * topk")
    if max(counts) > tokens:
        raise ValueError("an expert cannot occur twice in one token top-k row")
    queue = [(-count, expert) for expert, count in enumerate(counts) if count]
    heapq.heapify(queue)
    rows = []
    for _ in range(tokens):
        if len(queue) < topk:
            raise ValueError("route counts cannot form distinct top-k rows")
        selected = [heapq.heappop(queue) for _ in range(topk)]
        rows.append([expert for _, expert in selected])
        for negative_count, expert in selected:
            remaining = negative_count + 1
            if remaining < 0:
                heapq.heappush(queue, (remaining, expert))
    if queue:
        raise ValueError("route-count construction left unassigned rows")
    return torch.tensor(rows, dtype=torch.int32, device="cuda")


def _run_once(
    *,
    torch,
    fused_experts_impl,
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    expert_map,
):
    with torch.no_grad():
        output = fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            False,
            "silu",
            global_num_experts=128,
            expert_map=expert_map,
        )
    if output.shape != hidden_states.shape:
        raise RuntimeError("FusedMoE output shape differs from hidden states")
    return output


def _write_observation(
    *,
    path: Path,
    tokens: int,
    topk: int,
    hidden: int,
    intermediate: int,
    expert_rank: int,
    counts: list[int],
    stage1_config: dict,
    stage2_config: dict,
    stage1_padded_rows: int,
    stage2_padded_rows: int,
    mctlass_fused_moe_enabled: bool,
    mctlass_python_api_enabled: bool,
) -> None:
    stage_records = {}
    for name, config, padded_rows, projection_n in (
        ("w1", stage1_config, stage1_padded_rows, 2 * intermediate),
        ("w2", stage2_config, stage2_padded_rows, hidden),
    ):
        block_m = config["BLOCK_SIZE_M"]
        block_n = config["BLOCK_SIZE_N"]
        stage_records[name] = {
            "config": config,
            "padded_rows": padded_rows,
            "triton_if_selected_grid_x": (
                math.ceil(padded_rows / block_m)
                * math.ceil(projection_n / block_n)
            ),
            "triton_if_selected_grid_y": config["SPLIT_K"],
        }
    payload = {
        "schema_version": 1,
        "runtime": "vllm-metax 0.24.0 deployed unquantized BF16 path",
        "mctlass_fused_moe_enabled": mctlass_fused_moe_enabled,
        "mctlass_python_api_enabled": mctlass_python_api_enabled,
        "source_default_config_scope": (
            "Triton fallback only when MCTlass fused-MoE dispatch is disabled"
            if mctlass_fused_moe_enabled else "active Triton dispatch"
        ),
        "global_experts": 128,
        "local_experts": 64,
        "expert_rank": expert_rank,
        "tokens": tokens,
        "topk": topk,
        "hidden": hidden,
        "intermediate": intermediate,
        "tok_per_expert": counts,
        "stages": stage_records,
        "output_contract": (
            "rank-local projection output precedes EP AllReduce and is not "
            "an end-to-end numerical correctness result"
        ),
    }
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", required=True, type=_positive_integer)
    parser.add_argument("--topk", type=_positive_integer, default=8)
    parser.add_argument("--hidden", type=_positive_integer, default=2048)
    parser.add_argument("--intermediate", type=_positive_integer, default=768)
    parser.add_argument("--expert-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--tok-per-expert-json", required=True)
    parser.add_argument("--observation-path", required=True, type=Path)
    parser.add_argument("--profiler-attach-wait-seconds", type=float, default=30.0)
    arguments = parser.parse_args()
    if arguments.profiler_attach_wait_seconds < 0:
        raise ValueError("profiler attach wait must be nonnegative")
    counts = json.loads(arguments.tok_per_expert_json)
    if not isinstance(counts, list):
        raise TypeError("route counts must decode to a JSON list")

    import torch
    from vllm_metax.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl,
        initialize_staged_config,
        try_get_optimal_moe_config,
    )
    import vllm_metax.envs as mx_envs

    torch.cuda.set_device(0)
    tokens = arguments.num_tokens
    hidden_states = torch.randn(
        (tokens, arguments.hidden), dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn(
        (64, 2 * arguments.intermediate, arguments.hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    w2 = torch.randn(
        (64, arguments.hidden, arguments.intermediate),
        dtype=torch.bfloat16,
        device="cuda",
    )
    topk_ids = _route_ids(counts, tokens, arguments.topk, torch)
    topk_weights = torch.full(
        (tokens, arguments.topk),
        1.0 / arguments.topk,
        dtype=torch.bfloat16,
        device="cuda",
    )
    shard_start = arguments.expert_rank * 64
    expert_map = torch.tensor(
        [expert - shard_start if shard_start <= expert < shard_start + 64 else -1
         for expert in range(128)],
        dtype=torch.int32,
        device="cuda",
    )
    config = try_get_optimal_moe_config(
        tuple(w1.size()), tuple(w2.size()), arguments.topk, None, tokens)
    stage1_config, stage2_config = initialize_staged_config(config)

    _run_once(
        torch=torch,
        fused_experts_impl=fused_experts_impl,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_map=expert_map,
    )
    torch.cuda.synchronize()
    time.sleep(arguments.profiler_attach_wait_seconds)
    torch.cuda.profiler.start()
    _run_once(
        torch=torch,
        fused_experts_impl=fused_experts_impl,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_map=expert_map,
    )
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()

    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    _, _, stage1_padded = moe_align_block_size(
        topk_ids, stage1_config["BLOCK_SIZE_M"], 128, expert_map)
    _, _, stage2_padded = moe_align_block_size(
        topk_ids, stage2_config["BLOCK_SIZE_M"], 128, expert_map)
    _write_observation(
        path=arguments.observation_path,
        tokens=tokens,
        topk=arguments.topk,
        hidden=arguments.hidden,
        intermediate=arguments.intermediate,
        expert_rank=arguments.expert_rank,
        counts=counts,
        stage1_config=stage1_config,
        stage2_config=stage2_config,
        stage1_padded_rows=int(stage1_padded.item()),
        stage2_padded_rows=int(stage2_padded.item()),
        mctlass_fused_moe_enabled=mx_envs.MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE,
        mctlass_python_api_enabled=mx_envs.MACA_VLLM_ENABLE_MCTLASS_PYTHON_API,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
