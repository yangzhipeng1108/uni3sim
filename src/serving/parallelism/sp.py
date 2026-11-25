"""Sequence-parallel plan rules."""

from dataclasses import replace


def apply(plan, config):
    if config.sequence_parallel_size == 1:
        return plan
    return replace(
        plan,
        enabled_strategies=plan.enabled_strategies + ("sequence_parallel",),
        communication_rules=plan.communication_rules + ("sequence_all_gather",),
    )
