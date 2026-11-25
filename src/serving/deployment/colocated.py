"""Independent colocated-instance deployment builder."""

from .deployment_config import (
    DeploymentConfig,
    worker_role_operator_group,
    worker_role_phase,
)
from .deployment_plan import build_deployment_plan


def build_colocated_plan(config: DeploymentConfig, *, num_layers: int,
                         num_experts: int = 0,
                         pipeline_layer_partition=None):
    if config.mode != "colocated":
        raise ValueError("colocated deployment requires colocated mode")
    if not config.worker_pools:
        raise ValueError("colocated deployment requires at least one worker pool")
    if config.routes:
        raise ValueError(
            "independent colocated instances cannot declare transfer routes")
    invalid_roles = [
        pool.role for pool in config.worker_pools
        if (worker_role_phase(pool.role) != "colocated"
            or worker_role_operator_group(pool.role) is not None)
    ]
    if invalid_roles:
        raise ValueError(
            "colocated deployment worker pools require the colocated role: "
            f"{invalid_roles}")
    return build_deployment_plan(
        config,
        num_layers=num_layers,
        num_experts=num_experts,
        pipeline_layer_partition=pipeline_layer_partition,
    )
