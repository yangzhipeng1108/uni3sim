"""Prefill/decode disaggregated deployment builder."""

from .af_disagg import validate_af_config
from .deployment_config import (
    DeploymentConfig,
    worker_role_operator_group,
    worker_role_phase,
)
from .deployment_plan import build_deployment_plan


def build_pd_disagg_plan(config: DeploymentConfig, *, num_layers: int,
                         num_experts: int = 0):
    if config.mode not in ("pd_disagg", "pd_af_disagg"):
        raise ValueError("PD builder requires pd_disagg or pd_af_disagg mode")
    if config.mode == "pd_af_disagg":
        validate_af_config(config)
    elif config.mode == "pd_disagg":
        invalid_roles = tuple(
            pool.role for pool in config.worker_pools
            if pool.role not in ("prefill", "decode"))
        if invalid_roles:
            raise ValueError(
                "pure PD deployment accepts only prefill/decode worker roles; "
                f"got {invalid_roles!r}")
        invalid_payloads = tuple(
            route.payload for route in config.routes
            if route.payload != "kv")
        if invalid_payloads:
            raise ValueError(
                "pure PD deployment accepts only the complete P-to-D KV "
                f"route matrix; got payloads {invalid_payloads!r}")
    else:
        raise AssertionError(f"unhandled PD deployment mode: {config.mode!r}")
    phases = {worker_role_phase(pool.role) for pool in config.worker_pools}
    if "prefill" not in phases or "decode" not in phases:
        raise ValueError("PD deployment requires prefill and decode worker roles")
    pools_by_name = {pool.name: pool for pool in config.worker_pools}
    prefill_engines = tuple(
        pool.name for pool in config.worker_pools
        if worker_role_phase(pool.role) == "prefill"
        and worker_role_operator_group(pool.role) != "ffn"
    )
    decode_engines = tuple(
        pool.name for pool in config.worker_pools
        if worker_role_phase(pool.role) == "decode"
        and worker_role_operator_group(pool.role) != "ffn"
    )
    for route in config.routes:
        if route.payload != "kv":
            continue
        if route.source_pool not in pools_by_name:
            raise ValueError(
                f"KV route has unknown source pool {route.source_pool!r}")
        if route.destination_pool not in pools_by_name:
            raise ValueError(
                "KV route has unknown destination pool "
                f"{route.destination_pool!r}")
        source = pools_by_name[route.source_pool]
        destination = pools_by_name[route.destination_pool]
        if (worker_role_phase(source.role) != "prefill"
                or worker_role_operator_group(source.role) == "ffn"):
            raise ValueError("KV route source must be a prefill scheduler engine")
        if (worker_role_phase(destination.role) != "decode"
                or worker_role_operator_group(destination.role) == "ffn"):
            raise ValueError("KV route destination must be a decode scheduler engine")
    for source in prefill_engines:
        for destination in decode_engines:
            matches = [
                route for route in config.routes
                if route.payload == "kv"
                and route.source_pool == source
                and route.destination_pool == destination
            ]
            if len(matches) != 1:
                raise ValueError(
                    "PD deployment requires exactly one KV route for every "
                    f"P/D engine pair; {source!r}->{destination!r} has "
                    f"{len(matches)}")
    return build_deployment_plan(config, num_layers=num_layers, num_experts=num_experts)
