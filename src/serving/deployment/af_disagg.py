"""Attention/FFN disaggregated deployment validation and builder."""

from .deployment_config import (
    DeploymentConfig,
    worker_role_operator_group,
    worker_role_phase,
)
from .deployment_plan import build_deployment_plan


def validate_af_config(config: DeploymentConfig) -> None:
    """Require one-to-one Attention/FFN pairs with reciprocal routes."""

    if not isinstance(config, DeploymentConfig):
        raise TypeError("AF validation requires a DeploymentConfig")
    if config.mode == "af_disagg":
        expected_phases = ("colocated",)
    elif config.mode == "pd_af_disagg":
        expected_phases = ("prefill", "decode")
    else:
        raise ValueError("AF validation requires af_disagg or pd_af_disagg mode")

    if config.mode == "af_disagg":
        allowed_payloads = {"activation"}
    elif config.mode == "pd_af_disagg":
        allowed_payloads = {"activation", "kv"}
    else:
        raise ValueError("AF route validation received an unknown mode")
    unexpected_payloads = sorted({
        route.payload for route in config.routes
        if route.payload not in allowed_payloads
    })
    if unexpected_payloads:
        raise ValueError(
            f"{config.mode} has unsupported route payloads: "
            f"{unexpected_payloads}")

    pools_by_name = {pool.name: pool for pool in config.worker_pools}
    pairs = []
    expected_pool_names = set()
    for phase in expected_phases:
        by_group = {}
        for operator_group in ("attention", "ffn"):
            matches = [
                pool for pool in config.worker_pools
                if worker_role_phase(pool.role) == phase
                and worker_role_operator_group(pool.role) == operator_group
            ]
            if not matches:
                raise ValueError(
                    f"AF deployment requires at least one {phase} "
                    f"{operator_group} pool")
            by_group[operator_group] = tuple(matches)
            expected_pool_names.update(pool.name for pool in matches)
        if len(by_group["attention"]) != len(by_group["ffn"]):
            raise ValueError(
                f"AF deployment requires equal {phase} Attention and FFN "
                "pool counts; found "
                f"{len(by_group['attention'])} and {len(by_group['ffn'])}")
        pairs.append((phase, by_group["attention"], by_group["ffn"]))

    unexpected_pools = sorted(set(pools_by_name) - expected_pool_names)
    if unexpected_pools:
        raise ValueError(
            f"AF deployment has non-AF or duplicate-phase pools: {unexpected_pools}")

    activation_routes = tuple(
        route for route in config.routes if route.payload == "activation")
    for route in activation_routes:
        if (route.source_pool not in pools_by_name
                or route.destination_pool not in pools_by_name):
            raise ValueError(
                "activation route references a pool outside the AF deployment: "
                f"{route.source_pool!r}->{route.destination_pool!r}")
        source = pools_by_name[route.source_pool]
        destination = pools_by_name[route.destination_pool]
        source_phase = worker_role_phase(source.role)
        destination_phase = worker_role_phase(destination.role)
        source_group = worker_role_operator_group(source.role)
        destination_group = worker_role_operator_group(destination.role)
        if (source_phase != destination_phase
                or {source_group, destination_group} != {"attention", "ffn"}):
            raise ValueError(
                "activation routes must connect Attention and FFN pools in "
                f"the same phase: {route.source_pool!r}->{route.destination_pool!r}")

    expected_directed_routes = set()
    for phase, attention_pools, ffn_pools in pairs:
        ffn_names = {pool.name for pool in ffn_pools}
        paired_ffn_names = set()
        for attention_pool in attention_pools:
            forward = [
                route for route in activation_routes
                if route.source_pool == attention_pool.name
                and route.destination_pool in ffn_names
            ]
            if len(forward) != 1:
                raise ValueError(
                    "AF deployment requires exactly one outgoing activation "
                    f"route for {phase} Attention pool "
                    f"{attention_pool.name!r}; found {len(forward)}")
            ffn_name = forward[0].destination_pool
            if ffn_name in paired_ffn_names:
                raise ValueError(
                    f"AF deployment pairs {phase} FFN pool {ffn_name!r} "
                    "with more than one Attention pool")
            paired_ffn_names.add(ffn_name)
            reverse = [
                route for route in activation_routes
                if route.source_pool == ffn_name
                and route.destination_pool == attention_pool.name
            ]
            if len(reverse) != 1:
                raise ValueError(
                    "AF deployment requires exactly one activation route for "
                    f"{ffn_name!r}->{attention_pool.name!r}; found "
                    f"{len(reverse)}")
            expected_directed_routes.update((
                (attention_pool.name, ffn_name),
                (ffn_name, attention_pool.name),
            ))
        if paired_ffn_names != ffn_names:
            raise ValueError(
                f"AF deployment must pair every {phase} FFN pool exactly "
                f"once; unpaired={sorted(ffn_names - paired_ffn_names)}")

    actual_directed_routes = [
        (route.source_pool, route.destination_pool)
        for route in activation_routes
    ]
    if (len(actual_directed_routes) != len(expected_directed_routes)
            or set(actual_directed_routes) != expected_directed_routes):
        raise ValueError(
            "AF deployment activation routes must be exactly the reciprocal "
            "one-to-one Attention/FFN pairs")


def build_af_disagg_plan(config: DeploymentConfig, *, num_layers: int,
                         num_experts: int = 0):
    validate_af_config(config)
    if config.mode == "pd_af_disagg":
        from .pd_disagg import build_pd_disagg_plan
        return build_pd_disagg_plan(
            config, num_layers=num_layers, num_experts=num_experts)
    return build_deployment_plan(config, num_layers=num_layers, num_experts=num_experts)
