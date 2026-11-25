"""Normalized parallel plan for one worker pool."""

from dataclasses import dataclass
from typing import Tuple

from .parallel_config import ParallelConfig
from .rank_groups import RankGroups, build_rank_groups


@dataclass(frozen=True)
class ParallelPlan:
    config: ParallelConfig
    rank_groups: RankGroups
    num_layers: int
    num_experts: int = 0
    layer_to_pipeline_stage: Tuple[int, ...] = ()
    expert_to_shard: Tuple[int, ...] = ()
    enabled_strategies: Tuple[str, ...] = ()
    communication_rules: Tuple[str, ...] = ()
    replica_routing: str = "single"

    def logical_ranks_for_pipeline_stage(
        self,
        stage_index: int,
    ) -> Tuple[int, ...]:
        """Return every logical rank whose pipeline coordinate is a stage."""

        if isinstance(stage_index, bool) or not isinstance(stage_index, int):
            raise TypeError("pipeline stage index must be an integer")
        pipeline_size = self.config.pipeline_parallel_size
        if not 0 <= stage_index < pipeline_size:
            raise ValueError(
                "pipeline stage index is outside the configured pipeline: "
                f"{stage_index} vs {pipeline_size}")
        groups = self.rank_groups.pipeline
        if any(len(group) != pipeline_size for group in groups):
            raise RuntimeError(
                "pipeline rank groups disagree with pipeline parallel size")
        ranks = tuple(group[stage_index] for group in groups)
        if len(ranks) != len(set(ranks)):
            raise RuntimeError("pipeline stage contains duplicate logical ranks")
        return ranks


def build_parallel_plan(config: ParallelConfig, *, num_layers: int,
                        num_experts: int = 0,
                        pipeline_layer_partition=None) -> ParallelPlan:
    for name, value in (("num_layers", num_layers), ("num_experts", num_experts)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    if num_experts < 0:
        raise ValueError("num_experts cannot be negative")
    plan = ParallelPlan(
        config=config,
        rank_groups=build_rank_groups(config),
        num_layers=num_layers,
        num_experts=num_experts,
        layer_to_pipeline_stage=tuple(0 for _ in range(num_layers)),
        expert_to_shard=tuple(0 for _ in range(num_experts)),
    )
    from . import cp, dp, ep, pp, sp, tp
    for strategy in (tp, pp, ep, sp, cp, dp):
        if strategy is pp:
            plan = strategy.apply(plan, config, pipeline_layer_partition)
        else:
            plan = strategy.apply(plan, config)
    return plan
