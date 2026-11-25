"""Expert-parallel plan rules."""

from dataclasses import replace
from src.kernel.kernel_spec import linear_expert_partition


def apply(plan, config):
    if not config.enable_expert_parallel:
        return plan
    if plan.num_experts <= 0:
        raise ValueError("expert parallelism requires num_experts > 0")
    ep_size = config.expert_parallel_size
    mapping = []
    for ep_rank in range(ep_size):
        mapping.extend(ep_rank for _ in linear_expert_partition(plan.num_experts, ep_size, ep_rank))
    if len(mapping) != plan.num_experts:
        raise RuntimeError("expert placement does not cover global experts")
    communication_rules = ()
    if config.data_parallel_size * config.context_parallel_size > 1:
        communication_rules += (
            "expert_dispatch_all_gatherv",
            "expert_combine_reduce_scatterv",
        )
    if config.tensor_parallel_size > 1:
        communication_rules += ("expert_output_all_reduce",)
    return replace(
        plan,
        expert_to_shard=tuple(mapping),
        enabled_strategies=plan.enabled_strategies + ("expert_parallel",),
        communication_rules=(
            plan.communication_rules + communication_rules),
    )
