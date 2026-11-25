"""Data-parallel replica plan rules."""

from dataclasses import replace


def apply(plan, config):
    if config.data_parallel_size == 1:
        return plan
    return replace(
        plan,
        enabled_strategies=plan.enabled_strategies + ("data_parallel",),
        replica_routing="least_loaded",
    )
