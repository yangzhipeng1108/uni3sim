"""System resource activity, queue timing, and critical-path analysis."""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort_right
import math
from numbers import Real

from src.kernel.kernel_spec import KernelKind
from src.kernel.predictors.protobridge.adapter import (
    is_development_profile_sha256,
)
from src.serving.deployment.deployment_config import (
    worker_role_operator_group,
)
from src.serving.deployment.deployment_plan import (
    DeploymentPlan,
    pd_kv_transfer_rank_pairs,
)
from src.serving.kernel_lowering.execution_timeline import (
    call_reserved_resource_ids,
)
from src.serving.kernel_lowering.kv_offload_completion import (
    build_kv_cache_transfer_completion,
)
from src.serving.kv_cache.kv_cache_events import KVCacheAction
from src.serving.request_gen.request_stream import RequestStream
from src.serving.simulation import (
    SimulationKVCacheTransfer,
    SimulationResult,
)
from src.serving.schedule.schedule_step import is_non_mtp_async_terminal_placeholder
from src.serving.workload.workload_graph import WorkloadKind


_PREDECESSOR_CAUSE_PRIORITY = {
    "dependency": 0,
    "dependency_and_resource": 0,
    "resource": 1,
}


def system_resource_manifest(
    deployment: DeploymentPlan,
    requests: RequestStream,
    result: SimulationResult,
) -> dict[str, object]:
    if not isinstance(deployment, DeploymentPlan):
        raise TypeError("system analysis deployment must be DeploymentPlan")
    if not isinstance(requests, RequestStream):
        raise TypeError("system analysis requests must be RequestStream")
    if not isinstance(result, SimulationResult):
        raise TypeError("system analysis result must be SimulationResult")
    observation_window_s = result.schedule_trace.end_time_s
    if (isinstance(observation_window_s, bool)
            or not isinstance(observation_window_s, Real)
            or not math.isfinite(observation_window_s)
            or observation_window_s <= 0):
        raise ValueError(
            "system analysis observation window must be finite and positive")
    request_by_id = {
        request.request_id: request for request in requests
    }
    if set(request_by_id) != set(result.metrics.requests):
        raise ValueError(
            "system analysis requests disagree with result metrics")
    entries_by_step_id = {}
    for entry in result.schedule_trace.entries:
        step_id = entry.backend_step.schedule.step_id
        if step_id in entries_by_step_id:
            raise ValueError(
                f"system analysis found duplicate step ID: {step_id!r}")
        entries_by_step_id[step_id] = entry
    simulation_step_ids = {
        step.backend_step.schedule.step_id for step in result.steps
    }
    if len(simulation_step_ids) != len(result.steps):
        raise ValueError("system analysis found duplicate simulation step IDs")
    if simulation_step_ids != set(entries_by_step_id):
        raise ValueError(
            "system analysis simulation steps and schedule entries disagree")

    intervals_by_pool = {
        pool_name: [] for pool_name in deployment.worker_pools
    }
    interval_count_by_pool = {
        pool_name: 0 for pool_name in deployment.worker_pools
    }
    for step in result.steps:
        step_id = step.backend_step.schedule.step_id
        entry = entries_by_step_id[step_id]
        if entry.backend_step != step.backend_step:
            raise ValueError(
                f"system analysis step {step_id!r} disagrees with its "
                "schedule trace entry")
        execution_start_time_s = (
            entry.start_time_s
            + entry.scheduler_plan_cpu_s
            + entry.device_queue_wait_s)
        execution_end_time_s = (
            execution_start_time_s + entry.execution_wall_s)
        if not math.isclose(
                step.execution_timeline.duration_s,
                entry.execution_wall_s,
                rel_tol=1e-12,
                abs_tol=_timestamp_subtraction_tolerance_s(
                    execution_start_time_s, execution_end_time_s)):
            raise ValueError(
                f"system analysis step {step_id!r} duration disagrees with "
                "its schedule entry")
        if step.execution_timeline.pool_activity:
            pool_activities = step.execution_timeline.pool_activity
        elif step.kernel_plan.block_repeats:
            raise ValueError(
                "compact execution timeline lacks pool activity")
        else:
            pool_activities = ()
        if pool_activities:
            for activity in pool_activities:
                pool_name = activity.activity_id
                if pool_name not in deployment.worker_pools:
                    raise ValueError(
                        "system analysis activity references unknown pool: "
                        f"{pool_name!r}")
                for start_time_s, end_time_s in activity.intervals:
                    _validate_relative_interval(
                        start_time_s,
                        end_time_s,
                        step.execution_timeline.duration_s,
                        f"pool activity {pool_name!r}",
                        source_start_time_s=execution_start_time_s,
                        source_end_time_s=execution_end_time_s,
                    )
                    intervals_by_pool[pool_name].append((
                        execution_start_time_s + start_time_s,
                        execution_start_time_s + end_time_s,
                    ))
                interval_count_by_pool[pool_name] += activity.interval_count
            continue
        for interval in step.execution_timeline.intervals:
            pool_name = interval.pool_name
            if pool_name not in deployment.worker_pools:
                raise ValueError(
                    f"system analysis interval references unknown pool: "
                    f"{pool_name!r}")
            pool = deployment.worker_pools[pool_name]
            if interval.rank not in pool.ranks:
                raise ValueError(
                    f"system analysis interval rank {interval.rank} is not "
                    f"assigned to pool {pool_name!r}")
            _validate_relative_interval(
                interval.start_time_s,
                interval.end_time_s,
                step.execution_timeline.duration_s,
                f"kernel interval {interval.call_id!r}",
                source_start_time_s=execution_start_time_s,
                source_end_time_s=execution_end_time_s,
            )
            intervals_by_pool[pool_name].append((
                execution_start_time_s + interval.start_time_s,
                execution_start_time_s + interval.end_time_s,
            ))
            interval_count_by_pool[pool_name] += 1

    worker_pool_utilization = {}
    for pool_name in sorted(deployment.worker_pools):
        busy_time_s = _interval_union_duration(
            intervals_by_pool[pool_name], observation_window_s)
        idle_time_s = _nonnegative_difference(
            observation_window_s, busy_time_s)
        worker_pool_utilization[pool_name] = {
            "kernel_interval_count": interval_count_by_pool[pool_name],
            "busy_time_s": busy_time_s,
            "idle_time_s": idle_time_s,
            "utilization": busy_time_s / observation_window_s,
        }

    (
        communication_resource_utilization,
        transfer_kernel_bindings,
    ) = _communication_resource_utilization(
        result,
        entries_by_step_id,
        observation_window_s,
    )
    host_kv_link_utilization = _host_kv_link_utilization(
        deployment,
        result,
        entries_by_step_id,
        observation_window_s,
    )

    known_pool_ids = set(deployment.worker_pools)
    known_request_ids = set(request_by_id)
    link_by_resource_id = {}
    for route in deployment.routes:
        if route.resource_id in link_by_resource_id:
            if link_by_resource_id[route.resource_id] != route.link:
                raise ValueError(
                    "system analysis found an inconsistent communication "
                    f"resource contract: {route.resource_id!r}")
        else:
            link_by_resource_id[route.resource_id] = route.link
    transfers_by_resource = {
        resource_id: [] for resource_id in link_by_resource_id
    }
    transfer_bytes_by_resource = {
        resource_id: 0 for resource_id in link_by_resource_id
    }
    intervals_by_reserved_resource = {}
    for transfer in result.schedule_trace.transfers:
        if transfer.source_pool_id not in known_pool_ids:
            raise ValueError(
                "system analysis transfer references unknown source pool: "
                f"{transfer.source_pool_id!r}")
        if transfer.destination_pool_id not in known_pool_ids:
            raise ValueError(
                "system analysis transfer references unknown destination "
                f"pool: {transfer.destination_pool_id!r}")
        unknown_request_ids = (
            set(transfer.request_ids) - known_request_ids)
        if unknown_request_ids:
            raise ValueError(
                "system analysis transfer references unknown requests: "
                f"{sorted(unknown_request_ids)}")
        resource_id = transfer.resource_id
        if resource_id not in transfers_by_resource:
            raise ValueError(
                "system analysis transfer references an undeclared "
                f"communication resource: {resource_id!r}")
        if transfer.link != link_by_resource_id[resource_id]:
            raise ValueError(
                "system analysis transfer link disagrees with its "
                "communication resource")
        matching_routes = [
            route for route in deployment.routes
            if route.source_pool == transfer.source_pool_id
            and route.destination_pool == transfer.destination_pool_id
            and route.payload == transfer.payload
            and route.link == transfer.link
            and route.resource_id == resource_id
        ]
        if len(matching_routes) != 1:
            raise ValueError(
                "system analysis transfer must match exactly one deployment "
                "route by source, destination, payload, link, and resource: "
                f"transfer={transfer.transfer_id!r}, "
                f"matches={len(matching_routes)}")
        binding = transfer_kernel_bindings[transfer.transfer_id]
        call = binding["call"]
        physical_transfer = binding["physical_transfer"]
        if call.pool_name != transfer.source_pool_id:
            raise ValueError(
                "system analysis transfer source pool disagrees with its "
                "canonical communication call")
        if (call.resource != transfer.resource_id
                or physical_transfer.resource_id != transfer.resource_id):
            raise ValueError(
                "system analysis transfer resource disagrees with its "
                "canonical communication call")
        if physical_transfer.link != transfer.link:
            raise ValueError(
                "system analysis transfer link disagrees with its canonical "
                "communication call")
        if physical_transfer.bytes != transfer.byte_count:
            raise ValueError(
                "system analysis transfer bytes disagree with its canonical "
                "communication call")
        if (physical_transfer.direction != "point_to_point"
                or physical_transfer.operation != "send_recv"
                or physical_transfer.bytes_semantics != "payload"):
            raise ValueError(
                "system analysis trace transfer requires a canonical "
                "point-to-point payload call")
        source_pool = deployment.worker_pools[transfer.source_pool_id]
        destination_pool = deployment.worker_pools[
            transfer.destination_pool_id]
        if transfer.payload == "kv":
            source_nodes = tuple(
                node
                for simulation_transfer in result.transfers
                for node in simulation_transfer.workload.nodes
                if node.node_id == transfer.transfer_id
                and node.kind is WorkloadKind.PD_KV_TRANSFER
            )
            if len(source_nodes) != 1:
                raise ValueError(
                    "system analysis KV transfer requires one workload node")
            source_node = source_nodes[0]
            rank_pairs = tuple(
                (rank_pair.source_rank, rank_pair.destination_rank)
                for rank_pair in pd_kv_transfer_rank_pairs(
                    source_pool,
                    destination_pool,
                    source_node.attributes["num_key_value_heads"],
                )
            )
        elif len(source_pool.ranks) == 1 and len(destination_pool.ranks) == 1:
            rank_pairs = ((source_pool.ranks[0], destination_pool.ranks[0]),)
        elif len(source_pool.ranks) == 2 and len(destination_pool.ranks) == 2:
            parallel_configs = (
                source_pool.parallel_plan.config,
                destination_pool.parallel_plan.config,
            )
            common_tp2 = not any(
                    parallelism.worker_count != 2
                    or parallelism.tensor_parallel_size != 2
                    or parallelism.pipeline_parallel_size != 1
                    or parallelism.sequence_parallel_size != 1
                    or parallelism.context_parallel_size != 1
                    or parallelism.data_parallel_size != 1
                    for parallelism in parallel_configs)
            dense_tp2 = all(
                parallelism.expert_parallel_size == 1
                for parallelism in parallel_configs)
            operator_parallelism = {
                worker_role_operator_group(source_pool.role):
                    source_pool.parallel_plan.config,
                worker_role_operator_group(destination_pool.role):
                    destination_pool.parallel_plan.config,
            }
            af_ep2 = (
                deployment.mode == "af_disagg"
                and set(operator_parallelism) == {"attention", "ffn"}
                and operator_parallelism[
                    "attention"].expert_parallel_size == 1
                and operator_parallelism["ffn"].expert_parallel_size == 2
            )
            if not common_tp2 or not (dense_tp2 or af_ep2):
                raise NotImplementedError(
                    "system analysis two-rank route endpoints require "
                    "matching TP=2 lanes in the validated dense or FFN-EP2 "
                    "scope")
            rank_pairs = tuple(zip(
                source_pool.ranks,
                destination_pool.ranks,
                strict=True,
            ))
        else:
            raise NotImplementedError(
                "system analysis transfer reservation validation currently "
                "requires single-rank or matching two-lane TP=2 endpoints")
        physical_lane = (
            physical_transfer.source_rank,
            physical_transfer.destination_rank,
        )
        if (physical_lane not in rank_pairs
                or physical_transfer.participants != physical_lane
                or call.rank != physical_transfer.source_rank):
            raise ValueError(
                "system analysis canonical communication call disagrees with "
                "the deployment endpoint lanes")
        if (transfer.reserved_resource_ids
                != binding["reserved_resource_ids"]):
            raise ValueError(
                "system analysis transfer reserved resources disagree with "
                "its canonical communication call")
        if (not math.isclose(
                transfer.start_time_s,
                binding["start_time_s"],
                rel_tol=1e-12,
                abs_tol=1e-15)
                or not math.isclose(
                    transfer.end_time_s,
                    binding["end_time_s"],
                    rel_tol=1e-12,
                    abs_tol=1e-15)):
            raise ValueError(
                "system analysis transfer interval disagrees with its "
                "canonical communication interval")
        _validate_absolute_interval(
            transfer.start_time_s,
            transfer.end_time_s,
            observation_window_s,
            f"transfer {transfer.transfer_id!r}",
        )
        transfers_by_resource[resource_id].append((
            transfer.start_time_s,
            transfer.end_time_s,
        ))
        transfer_bytes_by_resource[resource_id] += transfer.byte_count
        for reserved_resource_id in transfer.reserved_resource_ids:
            if reserved_resource_id not in intervals_by_reserved_resource:
                intervals_by_reserved_resource[reserved_resource_id] = []
            intervals_by_reserved_resource[reserved_resource_id].append((
                transfer.start_time_s,
                transfer.end_time_s,
            ))

    link_utilization = {}
    for resource_id in sorted(link_by_resource_id):
        intervals = transfers_by_resource[resource_id]
        _validate_exclusive_resource_intervals(intervals, resource_id)
        busy_time_s = _interval_union_duration(
            intervals, observation_window_s)
        idle_time_s = _nonnegative_difference(
            observation_window_s, busy_time_s)
        link_utilization[resource_id] = {
            "link": link_by_resource_id[resource_id],
            "transfer_count": len(intervals),
            "transferred_bytes": transfer_bytes_by_resource[resource_id],
            "busy_time_s": busy_time_s,
            "idle_time_s": idle_time_s,
            "utilization": busy_time_s / observation_window_s,
        }
    for resource_id, intervals in intervals_by_reserved_resource.items():
        _validate_exclusive_resource_intervals(intervals, resource_id)

    router_queue, scheduler_queue = _queue_manifests(
        deployment,
        requests,
        result,
    )
    return {
        "schema_version": 5,
        "observation_window": {
            "basis": "simulation_origin_to_last_completion",
            "duration_s": float(observation_window_s),
        },
        "worker_pool_activity_basis": (
            "union_of_emitted_step_kernel_intervals"),
        "link_activity_basis": (
            "fabric_resource_union_of_canonical_transfer_trace_intervals"),
        "communication_resource_activity_basis": (
            "fabric_resource_union_of_canonical_communication_kernel_"
            "intervals"),
        "host_kv_link_activity_basis": (
            "per_rank_resource_union_of_canonical_kv_offload_physical_"
            "completion_intervals"),
        "worker_pool_utilization": worker_pool_utilization,
        "link_utilization": link_utilization,
        "communication_resource_utilization": (
            communication_resource_utilization),
        "host_kv_link_utilization": host_kv_link_utilization,
        "router_queue": router_queue,
        "scheduler_queue": scheduler_queue,
    }


def _host_kv_link_utilization(
    deployment: DeploymentPlan,
    result: SimulationResult,
    entries_by_step_id: dict[str, object],
    observation_window_s: float,
) -> dict[str, object]:
    steps_by_id = {}
    transfer_events_by_key = {}
    all_kv_call_keys = set()
    for step in result.steps:
        step_id = step.backend_step.schedule.step_id
        if step_id in steps_by_id:
            raise ValueError(
                f"Host-KV analysis found duplicate step ID: {step_id!r}")
        steps_by_id[step_id] = step
        for event in step.backend_step.kv_cache_events:
            if event.action not in (
                    KVCacheAction.LOAD, KVCacheAction.EVICT):
                continue
            key = (step_id, event.ready_event)
            if key in transfer_events_by_key:
                raise ValueError(
                    "Host-KV analysis found duplicate transfer intent")
            transfer_events_by_key[key] = event
        for call in step.kernel_plan.calls:
            if call.kernel_spec.kind is KernelKind.KV_OFFLOAD:
                all_kv_call_keys.add((step_id, call.call_id))

    if result.kv_offload is None:
        if transfer_events_by_key or all_kv_call_keys:
            raise ValueError(
                "Host-KV analysis found offload intent/call without "
                "canonical completion state")
        return {}

    records_by_key = {}
    for record in result.kv_offload.transfers:
        if not isinstance(record, SimulationKVCacheTransfer):
            raise TypeError(
                "Host-KV analysis requires canonical transfer completions")
        key = (record.step_id, record.event.ready_event)
        if key in records_by_key:
            raise ValueError(
                "Host-KV analysis found duplicate transfer completion")
        records_by_key[key] = record
    if set(records_by_key) != set(transfer_events_by_key):
        raise ValueError(
            "Host-KV intent/completion coverage disagrees")
    physical_counts = {
        len(record.completion.physical_transfers)
        for record in records_by_key.values()
    }
    if physical_counts not in ({1}, {2}):
        raise NotImplementedError(
            "Host-KV link utilization supports the validated single-rank "
            "or TP=2 lockstep scope")

    records_by_resource = {}
    covered_kv_call_keys = set()
    for key in sorted(records_by_key):
        step_id, ready_event = key
        if step_id not in steps_by_id or step_id not in entries_by_step_id:
            raise ValueError(
                "Host-KV completion references an unknown step")
        step = steps_by_id[step_id]
        event = transfer_events_by_key[key]
        record = records_by_key[key]
        if record.event != event:
            raise ValueError(
                "Host-KV completion intent changed after scheduling")
        owner_nodes = tuple(
            node for node in step.workload.nodes
            if (node.kind is WorkloadKind.KV_OFFLOAD
                and node.attributes["ready_event"] == ready_event)
        )
        if (not owner_nodes
                or any(node.rank is None for node in owner_nodes)):
            raise ValueError(
                "Host-KV workload owner ranks are incomplete")
        expected_pool_name = step.backend_step.schedule.pool_name
        expected_ranks = tuple(sorted(node.rank for node in owner_nodes))
        expected_completion = build_kv_cache_transfer_completion(
            event,
            step.workload,
            step.kernel_plan,
            step.predictions,
            step.execution_timeline,
            expected_pool_name=expected_pool_name,
            expected_ranks=expected_ranks,
        )
        if record.completion != expected_completion:
            raise ValueError(
                "Host-KV completion disagrees with the canonical timeline")
        calls_by_id = {
            call.call_id: call for call in step.kernel_plan.calls
        }
        intervals_by_id = {
            interval.call_id: interval
            for interval in step.execution_timeline.intervals
        }
        entry = entries_by_step_id[step_id]
        for physical in record.completion.physical_transfers:
            call_key = (step_id, physical.call_id)
            if call_key in covered_kv_call_keys:
                raise ValueError(
                    "Host-KV physical call belongs to multiple completions")
            covered_kv_call_keys.add(call_key)
            if (physical.call_id not in calls_by_id
                    or physical.call_id not in intervals_by_id
                    or physical.call_id not in step.predictions):
                raise ValueError(
                    "Host-KV physical completion lacks call, prediction, or "
                    "interval")
            call = calls_by_id[physical.call_id]
            interval = intervals_by_id[physical.call_id]
            prediction = step.predictions[physical.call_id]
            if physical.pool_name not in deployment.worker_pools:
                raise ValueError(
                    "Host-KV physical completion references an unknown pool")
            pool = deployment.worker_pools[physical.pool_name]
            if (physical.rank not in pool.ranks
                    or call.kernel_spec.hardware_id != pool.hardware_id):
                raise ValueError(
                    "Host-KV physical call disagrees with deployment rank/"
                    "hardware ownership")
            resource_id = (
                f"pool:{physical.pool_name}:rank:{physical.rank}:"
                f"{physical.link}")
            expected_reserved_resources = call_reserved_resource_ids(call)
            if (expected_reserved_resources != (resource_id,)
                    or interval.reserved_resource_ids
                    != expected_reserved_resources):
                raise ValueError(
                    "Host-KV physical interval is not reserved on its "
                    "rank-local link resource")
            execution_start_time_s = (
                entry.start_time_s
                + entry.scheduler_plan_cpu_s
                + entry.device_queue_wait_s)
            absolute_start_s = execution_start_time_s + physical.start_time_s
            absolute_end_s = execution_start_time_s + physical.end_time_s
            _validate_absolute_interval(
                absolute_start_s,
                absolute_end_s,
                observation_window_s,
                f"Host-KV interval {physical.call_id!r}",
            )
            if not math.isclose(
                    absolute_end_s - absolute_start_s,
                    prediction.t_total_s,
                    rel_tol=1e-12,
                    abs_tol=_timestamp_subtraction_tolerance_s(
                        absolute_start_s, absolute_end_s)):
                raise ValueError(
                    "Host-KV physical interval disagrees with its canonical "
                    "prediction")
            contract = (
                physical.pool_name,
                physical.rank,
                call.kernel_spec.hardware_id,
                physical.link,
            )
            if resource_id not in records_by_resource:
                records_by_resource[resource_id] = {
                    "contract": contract,
                    "intervals": [],
                    "physical_transfer_count": 0,
                    "action_counts": {},
                    "bytes_by_action": {},
                }
            resource_record = records_by_resource[resource_id]
            if resource_record["contract"] != contract:
                raise ValueError(
                    "Host-KV resource has inconsistent physical contract: "
                    f"{resource_id!r}")
            resource_record["intervals"].append((
                absolute_start_s, absolute_end_s))
            resource_record["physical_transfer_count"] += 1
            action = event.action.value
            if action not in resource_record["action_counts"]:
                resource_record["action_counts"][action] = 0
                resource_record["bytes_by_action"][action] = 0
            resource_record["action_counts"][action] += 1
            resource_record["bytes_by_action"][action] += physical.bytes

    if covered_kv_call_keys != all_kv_call_keys:
        raise ValueError(
            "Host-KV completion coverage disagrees with canonical "
            "KV_OFFLOAD calls")

    manifest = {}
    for resource_id in sorted(records_by_resource):
        record = records_by_resource[resource_id]
        pool_name, rank, hardware_id, link = record["contract"]
        intervals = record["intervals"]
        _validate_exclusive_resource_intervals(intervals, resource_id)
        busy_time_s = _interval_union_duration(
            intervals, observation_window_s)
        bytes_by_action = {
            action: record["bytes_by_action"][action]
            for action in sorted(record["bytes_by_action"])
        }
        manifest[resource_id] = {
            "pool_name": pool_name,
            "rank": rank,
            "hardware_id": hardware_id,
            "link": link,
            "physical_transfer_count": record["physical_transfer_count"],
            "action_counts": {
                action: record["action_counts"][action]
                for action in sorted(record["action_counts"])
            },
            "bytes_by_action": bytes_by_action,
            "transferred_bytes": sum(bytes_by_action.values()),
            "busy_time_s": busy_time_s,
            "idle_time_s": _nonnegative_difference(
                observation_window_s, busy_time_s),
            "utilization": busy_time_s / observation_window_s,
        }
    return manifest


def _communication_resource_utilization(
    result: SimulationResult,
    entries_by_step_id: dict[str, object],
    observation_window_s: float,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    trace_transfers_by_id = {}
    for transfer in result.schedule_trace.transfers:
        if transfer.transfer_id in trace_transfers_by_id:
            raise ValueError(
                "system analysis found duplicate transfer ID: "
                f"{transfer.transfer_id!r}")
        trace_transfers_by_id[transfer.transfer_id] = transfer

    owners = []
    for step in result.steps:
        step_id = step.backend_step.schedule.step_id
        if step_id not in entries_by_step_id:
            raise ValueError(
                "communication analysis step has no schedule entry: "
                f"{step_id!r}")
        owners.append((
            f"step:{step_id}",
            entries_by_step_id[step_id].start_time_s
            + entries_by_step_id[step_id].scheduler_plan_cpu_s
            + entries_by_step_id[step_id].device_queue_wait_s,
            step.workload,
            step.kernel_plan,
            step.execution_timeline,
        ))
    for transfer in result.transfers:
        source_node_ids = {
            node_id
            for call in transfer.kernel_plan.calls
            for node_id in call.source_node_ids
        }
        matching_transfer_ids = tuple(
            transfer_id
            for transfer_id in trace_transfers_by_id
            if transfer_id in source_node_ids
        )
        if len(matching_transfer_ids) != 1:
            raise ValueError(
                "communication analysis SimulationTransfer must match one "
                "trace transfer, found "
                f"{matching_transfer_ids!r}")
        transfer_id = matching_transfer_ids[0]
        calls_by_id = {
            call.call_id: call for call in transfer.kernel_plan.calls
        }
        intervals = transfer.execution_timeline.intervals
        if (len({interval.call_id for interval in intervals}) != len(intervals)
                or {interval.call_id for interval in intervals} != set(calls_by_id)):
            raise ValueError(
                "communication transfer timeline must cover every call once")
        matching_intervals = tuple(
            interval for interval in intervals
            if transfer_id in calls_by_id[interval.call_id].source_node_ids)
        if len(matching_intervals) != 1:
            raise ValueError("communication transfer must have one canonical interval")
        # The trace begins at the physical transfer, after Host submission;
        # timeline intervals remain relative to execution entry.
        interval, = matching_intervals
        owners.append((
            f"transfer:{transfer_id}",
            trace_transfers_by_id[transfer_id].start_time_s - interval.start_time_s,
            transfer.workload,
            transfer.kernel_plan,
            transfer.execution_timeline,
        ))

    records_by_resource = {}
    intervals_by_reserved_resource = {}
    fabric_intervals_by_resource = {}
    transfer_kernel_bindings = {}
    for owner_id, origin_time_s, workload, plan, timeline in owners:
        calls_by_id = {call.call_id: call for call in plan.calls}
        intervals_by_id = {
            interval.call_id: interval for interval in timeline.intervals
        }
        if (len(intervals_by_id) != len(timeline.intervals)
                or set(intervals_by_id) != set(calls_by_id)):
            raise ValueError(
                "communication analysis timeline coverage disagrees with "
                f"KernelPlan for {owner_id!r}")
        if timeline.call_occurrence_counts:
            occurrence_count_by_call_id = dict(
                timeline.call_occurrence_counts)
        elif plan.block_repeats:
            raise ValueError(
                "compact communication timeline lacks occurrence counts")
        else:
            occurrence_count_by_call_id = {
                call_id: 1 for call_id in calls_by_id
            }
        repeated_trace_occurrences_by_call_id = {}
        repeated_template_ids = set()
        timeline_repeats_by_id = {
            repeat.block_id: repeat for repeat in timeline.block_repeats
        }
        workload_repeats = (
            workload.block_repeats if plan.block_repeats else ()
        )
        for workload_repeat in workload_repeats:
            if not workload_repeat.trace_node_ids:
                continue
            if workload_repeat.block_id not in timeline_repeats_by_id:
                raise ValueError(
                    "communication analysis trace repeat lacks timeline")
            occurrence_intervals_by_call_id = dict(
                timeline_repeats_by_id[
                    workload_repeat.block_id].call_intervals)
            for template_node_id, logical_node_ids in (
                    workload_repeat.trace_node_ids):
                matching_calls = tuple(
                    call for call in plan.calls
                    if template_node_id in call.source_node_ids
                )
                if len(matching_calls) != 1:
                    raise ValueError(
                        "communication analysis repeated trace node must "
                        "lower to one call")
                repeated_call_id = matching_calls[0].call_id
                occurrence_intervals = occurrence_intervals_by_call_id[
                    repeated_call_id]
                repeated_trace_occurrences_by_call_id[
                    repeated_call_id] = tuple(zip(
                        logical_node_ids,
                        occurrence_intervals,
                        strict=True,
                    ))
                repeated_template_ids.add(template_node_id)
        if timeline.reserved_resource_activity:
            for activity in timeline.reserved_resource_activity:
                absolute_intervals = tuple(
                    (
                        origin_time_s + start_time_s,
                        origin_time_s + end_time_s,
                    )
                    for start_time_s, end_time_s in activity.intervals
                )
                if activity.activity_id not in intervals_by_reserved_resource:
                    intervals_by_reserved_resource[activity.activity_id] = []
                intervals_by_reserved_resource[activity.activity_id].extend(
                    absolute_intervals)
                fabric_prefix = "communication:fabric:"
                if activity.activity_id.startswith(fabric_prefix):
                    resource_id = activity.activity_id[len(fabric_prefix):]
                    if resource_id not in fabric_intervals_by_resource:
                        fabric_intervals_by_resource[resource_id] = []
                    fabric_intervals_by_resource[resource_id].extend(
                        absolute_intervals)
        for call_id in intervals_by_id:
            call = calls_by_id[call_id]
            if call.kernel_spec.kind is not KernelKind.COMMUNICATION:
                continue
            transfer = call.kernel_spec.transfer
            if transfer is None:
                raise ValueError(
                    "COMMUNICATION KernelSpec requires a physical transfer")
            if call.resource != transfer.resource_id:
                raise ValueError(
                    "communication call resource disagrees with its physical "
                    "transfer")
            interval = intervals_by_id[call_id]
            expected_reserved_resources = call_reserved_resource_ids(call)
            if interval.reserved_resource_ids != expected_reserved_resources:
                raise ValueError(
                    "communication interval reserved resources disagree with "
                    f"its canonical call: {call_id!r}")
            absolute_start_s = origin_time_s + interval.start_time_s
            absolute_end_s = origin_time_s + interval.end_time_s
            _validate_absolute_interval(
                absolute_start_s,
                absolute_end_s,
                observation_window_s,
                f"communication interval {call_id!r}",
            )
            if not timeline.reserved_resource_activity:
                for reserved_resource_id in interval.reserved_resource_ids:
                    if reserved_resource_id not in intervals_by_reserved_resource:
                        intervals_by_reserved_resource[
                            reserved_resource_id] = []
                    intervals_by_reserved_resource[
                        reserved_resource_id].append((
                            absolute_start_s,
                            absolute_end_s,
                        ))
                if transfer.resource_id not in fabric_intervals_by_resource:
                    fabric_intervals_by_resource[transfer.resource_id] = []
                fabric_intervals_by_resource[transfer.resource_id].append((
                    absolute_start_s,
                    absolute_end_s,
                ))
            occurrence_count = occurrence_count_by_call_id[call_id]

            matching_trace_ids = tuple(
                node_id for node_id in call.source_node_ids
                if node_id in trace_transfers_by_id
            )
            if call_id in repeated_trace_occurrences_by_call_id:
                matching_trace_ids = tuple(
                    transfer_id for transfer_id in matching_trace_ids
                    if transfer_id not in repeated_template_ids
                )
                trace_occurrences = repeated_trace_occurrences_by_call_id[
                    call_id]
            else:
                trace_occurrences = tuple(
                    (transfer_id, (
                        interval.start_time_s, interval.end_time_s))
                    for transfer_id in matching_trace_ids
                )
            if len(matching_trace_ids) > 1:
                raise ValueError(
                    "one communication call cannot own multiple trace "
                    f"transfers: {matching_trace_ids!r}")
            for transfer_id, relative_interval in trace_occurrences:
                if transfer_id in transfer_kernel_bindings:
                    raise ValueError(
                        "system analysis trace transfer maps to multiple "
                        f"communication calls: {transfer_id!r}")
                transfer_kernel_bindings[transfer_id] = {
                    "call": call,
                    "physical_transfer": transfer,
                    "reserved_resource_ids": expected_reserved_resources,
                    "start_time_s": (
                        origin_time_s + relative_interval[0]),
                    "end_time_s": (
                        origin_time_s + relative_interval[1]),
                }

            resource_id = transfer.resource_id
            contract = (
                call.kernel_spec.hardware_id,
                transfer.link,
                transfer.backend,
            )
            if resource_id not in records_by_resource:
                records_by_resource[resource_id] = {
                    "contract": contract,
                    "intervals": [],
                    "call_count": 0,
                    "operation_counts": {},
                    "topology_counts": {},
                    "bytes_by_semantics": {},
                    "backend_config_sha256_by_operation": {},
                }
            record = records_by_resource[resource_id]
            if record["contract"] != contract:
                raise ValueError(
                    "communication resource has inconsistent physical "
                    f"contracts: {resource_id!r}")
            record["call_count"] += occurrence_count
            operation = transfer.operation
            profile_by_operation = record[
                "backend_config_sha256_by_operation"]
            if operation in profile_by_operation:
                if (profile_by_operation[operation]
                        != transfer.backend_config_sha256):
                    raise ValueError(
                        "communication operation has inconsistent backend "
                        f"profiles: {resource_id!r}/{operation}")
            else:
                profile_by_operation[operation] = (
                    transfer.backend_config_sha256)
            if operation not in record["operation_counts"]:
                record["operation_counts"][operation] = 0
            record["operation_counts"][operation] += occurrence_count
            topology = transfer.topology
            if topology not in record["topology_counts"]:
                record["topology_counts"][topology] = 0
            record["topology_counts"][topology] += occurrence_count
            bytes_semantics = transfer.bytes_semantics
            if bytes_semantics not in record["bytes_by_semantics"]:
                record["bytes_by_semantics"][bytes_semantics] = 0
            record["bytes_by_semantics"][bytes_semantics] += (
                transfer.bytes * occurrence_count)

    for resource_id in intervals_by_reserved_resource:
        _validate_exclusive_resource_intervals(
            intervals_by_reserved_resource[resource_id],
            resource_id,
        )

    for resource_id in records_by_resource:
        if resource_id not in fabric_intervals_by_resource:
            raise ValueError(
                "communication resource lacks compact fabric activity: "
                f"{resource_id!r}")
        records_by_resource[resource_id]["intervals"] = (
            fabric_intervals_by_resource[resource_id])

    if set(transfer_kernel_bindings) != set(trace_transfers_by_id):
        raise ValueError(
            "system analysis trace transfer coverage disagrees with canonical "
            "communication calls: "
            f"trace={sorted(trace_transfers_by_id)!r}, "
            f"calls={sorted(transfer_kernel_bindings)!r}")

    manifest = {}
    for resource_id in sorted(records_by_resource):
        record = records_by_resource[resource_id]
        hardware_id, link, backend = record["contract"]
        if backend == "analytical_roofline_v1":
            timing_status = "projected_analytical_roofline"
        elif backend == "protobridge":
            profile_digests = tuple(
                digest for digest in record[
                    "backend_config_sha256_by_operation"].values()
                if digest is not None
            )
            if profile_digests and any(
                    is_development_profile_sha256(digest)
                    for digest in profile_digests):
                timing_status = "predicted_by_protobridge_development_only"
            else:
                timing_status = "predicted_by_protobridge"
        else:
            raise ValueError(
                f"unknown communication backend in result: {backend!r}")
        intervals = record["intervals"]
        resource_profile_digests = set(
            record["backend_config_sha256_by_operation"].values())
        if len(resource_profile_digests) == 1:
            backend_config_sha256 = next(iter(resource_profile_digests))
        else:
            backend_config_sha256 = None
        _validate_exclusive_resource_intervals(intervals, resource_id)
        busy_time_s = _interval_union_duration(
            intervals, observation_window_s)
        manifest[resource_id] = {
            "hardware_id": hardware_id,
            "link": link,
            "backend": backend,
            "backend_config_sha256": backend_config_sha256,
            "timing_status": timing_status,
            "call_count": record["call_count"],
            "operation_counts": {
                operation: record["operation_counts"][operation]
                for operation in sorted(record["operation_counts"])
            },
            "topology_counts": {
                topology: record["topology_counts"][topology]
                for topology in sorted(record["topology_counts"])
            },
            "bytes_by_semantics": {
                semantics: record["bytes_by_semantics"][semantics]
                for semantics in sorted(record["bytes_by_semantics"])
            },
            "busy_time_s": busy_time_s,
            "idle_time_s": _nonnegative_difference(
                observation_window_s, busy_time_s),
            "utilization": busy_time_s / observation_window_s,
        }
    return manifest, transfer_kernel_bindings


def critical_path_manifest(
    deployment: DeploymentPlan,
    requests: RequestStream,
    result: SimulationResult,
) -> dict[str, object]:
    """Explain modeled request latency without inventing runtime causality.

    Kernel intervals belong to a scheduler step shared by every request in its
    batch.  They are therefore wall-clock contributions, not per-request
    kernel ownership.  Gaps between steps are observable but their scheduler
    cause is not, so they remain explicitly named ``not_scheduled_wait``.
    """

    if not isinstance(deployment, DeploymentPlan):
        raise TypeError("critical path deployment must be DeploymentPlan")
    if not isinstance(requests, RequestStream):
        raise TypeError("critical path requests must be RequestStream")
    if not isinstance(result, SimulationResult):
        raise TypeError("critical path result must be SimulationResult")
    observation_window_s = result.schedule_trace.end_time_s
    if (isinstance(observation_window_s, bool)
            or not isinstance(observation_window_s, Real)
            or not math.isfinite(observation_window_s)
            or observation_window_s <= 0):
        raise ValueError(
            "critical path observation window must be finite and positive")

    requests_by_id = {
        request.request_id: request for request in requests
    }
    if set(requests_by_id) != set(result.metrics.requests):
        raise ValueError("critical path requests disagree with result metrics")
    entries_by_step_id = {}
    for entry in result.schedule_trace.entries:
        step_id = entry.backend_step.schedule.step_id
        if step_id in entries_by_step_id:
            raise ValueError(
                f"critical path found duplicate step ID: {step_id!r}")
        entries_by_step_id[step_id] = entry
    steps_by_id = {}
    for step in result.steps:
        step_id = step.backend_step.schedule.step_id
        if step_id in steps_by_id:
            raise ValueError(
                f"critical path found duplicate simulation step: {step_id!r}")
        steps_by_id[step_id] = step
    if set(steps_by_id) != set(entries_by_step_id):
        raise ValueError(
            "critical path simulation steps and schedule entries disagree")

    step_paths = []
    step_path_by_id = {}
    for step_id in sorted(
            steps_by_id,
            key=lambda identifier: (
                entries_by_step_id[identifier].start_time_s,
                entries_by_step_id[
                    identifier].backend_step.schedule.pool_name,
                identifier,
            )):
        path = _critical_step_path(
            steps_by_id[step_id],
            entries_by_step_id[step_id],
            observation_window_s,
        )
        step_paths.append(path)
        step_path_by_id[step_id] = path

    request_paths = {}
    for request_id in sorted(requests_by_id):
        request = requests_by_id[request_id]
        metric = result.metrics.requests[request_id]
        blocks = _request_execution_blocks(
            request_id,
            deployment,
            result,
            entries_by_step_id,
            step_path_by_id,
            observation_window_s,
        )
        completion_entries = [
            entry for entry in result.schedule_trace.entries
            if request_id in entry.backend_step.finished_request_ids
        ]
        if not completion_entries:
            raise ValueError(
                f"critical path request {request_id!r} has no completion")
        canonical_e2e_target_s = max(
            entry.end_time_s for entry in completion_entries)
        terminal_completion_entries = [
            entry for entry in completion_entries
            if math.isclose(
                entry.end_time_s,
                canonical_e2e_target_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ]
        if len(terminal_completion_entries) != 1:
            raise ValueError(
                f"critical path request {request_id!r} has ambiguous "
                "terminal completion")
        e2e_target_s = request.arrival_time_s + metric.latency_s
        e2e = _request_target_path(
            request_id,
            request.arrival_time_s,
            e2e_target_s,
            blocks,
            "e2e",
            canonical_terminal_time_s=canonical_e2e_target_s,
        )
        if metric.ttft_s is None:
            ttft = {
                "availability": "not_applicable_zero_output_tokens",
                "duration_s": None,
                "target_time_s": None,
                "segments": [],
                "contribution_s_by_category": {},
                "dominant_category": None,
            }
        else:
            ttft_target_s = request.arrival_time_s + metric.ttft_s
            ttft = _request_target_path(
                request_id,
                request.arrival_time_s,
                ttft_target_s,
                blocks,
                "ttft",
            )
        request_paths[request_id] = {
            "arrival_time_s": request.arrival_time_s,
            "attribution": "shared_batch_wall_clock_not_kernel_ownership",
            "ttft": ttft,
            "e2e": e2e,
        }

    request_completion_horizon_s = max(
        path["e2e"]["target_time_s"]
        for path in request_paths.values()
    )
    terminal_request_ids = tuple(sorted(
        request_id
        for request_id, path in request_paths.items()
        if math.isclose(
            path["e2e"]["target_time_s"],
            request_completion_horizon_s,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    ))
    if not terminal_request_ids:
        raise ValueError(
            "critical path found no request completing at the system horizon")
    system_paths = []
    for request_id in terminal_request_ids:
        arrival_time_s = request_paths[request_id]["arrival_time_s"]
        contributions = dict(
            request_paths[request_id]["e2e"][
                "contribution_s_by_category"])
        e2e_duration_s = request_paths[request_id]["e2e"]["duration_s"]
        if not math.isclose(
                arrival_time_s + e2e_duration_s,
                request_completion_horizon_s,
                rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(
                f"terminal request path for {request_id!r} does not end at "
                "the observation horizon")
        system_paths.append({
            "request_id": request_id,
            "path_start_time_s": arrival_time_s,
            "path_duration_s": e2e_duration_s,
            "unattributed_pre_arrival_window_s": arrival_time_s,
            "contribution_s_by_category": contributions,
            "dominant_category": _dominant_category(contributions),
        })

    return {
        "schema_version": 2,
        "semantics": {
            "path_basis": "canonical_simulator_timeline",
            "step_attribution": (
                "shared_batch_wall_clock_not_per_request_kernel_share"),
            "kernel_bottleneck_basis": "modeled_kernel_prediction_group",
            "gap_attribution": (
                "observable_not_scheduled_wait_without_scheduler_cause"),
            "router_attribution": (
                "modeled_serial_dispatch_queue_and_service"),
            "system_path_scope": (
                "terminal_request_e2e_not_global_pre_arrival_causal_path"),
            "tie_break": (
                "explicit_dependency_then_resource_predecessor_then_call_id"),
        },
        "steps": step_paths,
        "requests": request_paths,
        "system_completion": {
            "time_s": request_completion_horizon_s,
            "terminal_request_ids": terminal_request_ids,
            "terminal_observation_paths": system_paths,
        },
    }


def _critical_step_path(step, entry, observation_window_s: float):
    step_id = step.backend_step.schedule.step_id
    if entry.backend_step.schedule.step_id != step_id:
        raise ValueError("critical path step and trace entry IDs disagree")
    pool_name = step.backend_step.schedule.pool_name
    if entry.backend_step.schedule.pool_name != pool_name:
        raise ValueError("critical path step and trace entry pools disagree")
    entry_duration_s = entry.end_time_s - entry.start_time_s
    scheduler_plan_end_time_s = (
        entry.start_time_s + entry.scheduler_plan_cpu_s)
    execution_start_time_s = (
        scheduler_plan_end_time_s + entry.device_queue_wait_s)
    execution_end_time_s = execution_start_time_s + entry.execution_wall_s
    scheduler_update_start_time_s = (
        execution_end_time_s + entry.completion_queue_wait_s)
    if not math.isclose(
            step.execution_timeline.duration_s,
            entry.execution_wall_s,
            rel_tol=1e-12,
            abs_tol=_timestamp_subtraction_tolerance_s(
                execution_start_time_s, execution_end_time_s)):
        raise ValueError(
            f"critical path step {step_id!r} duration disagrees with trace")
    if (not step.kernel_plan.block_repeats
            and step.execution_timeline.host_intervals):
        calls_by_id = {call.call_id: call for call in step.kernel_plan.calls}
        intervals = step.execution_timeline.intervals
        if (len({interval.call_id for interval in intervals}) != len(intervals)
                or {interval.call_id for interval in intervals} != set(calls_by_id)
                or set(step.predictions) != set(calls_by_id)):
            raise ValueError(
                "critical path timeline and predictions must cover every call once")
        for interval in intervals:
            if interval.reserved_resource_ids != call_reserved_resource_ids(
                    calls_by_id[interval.call_id]):
                raise ValueError(
                    "critical path interval reserved resources disagree with its call")
            if not math.isclose(
                    interval.end_time_s - interval.start_time_s,
                    step.predictions[interval.call_id].t_total_s,
                    rel_tol=1e-12,
                    abs_tol=_timestamp_subtraction_tolerance_s(
                        execution_start_time_s + interval.start_time_s,
                        execution_start_time_s + interval.end_time_s)):
                raise ValueError(
                    "critical path interval disagrees with its prediction")
    if step.kernel_plan.block_repeats or step.execution_timeline.host_intervals:
        compact = bool(step.kernel_plan.block_repeats)
        contributions = dict(
            step.execution_timeline.critical_path_contributions)
        if (not contributions
                or len(contributions) != len(
                    step.execution_timeline.critical_path_contributions)):
            raise ValueError(
                f"critical path step {step_id!r} requires unique contributions")
        if any(not math.isfinite(value) or value < 0
                for value in contributions.values()):
            raise ValueError(
                f"critical path step {step_id!r} contributions must be finite and nonnegative")
        contribution_total_s = sum(contributions.values())
        if not math.isclose(
                contribution_total_s,
                entry.execution_wall_s,
                rel_tol=1e-12,
                abs_tol=_timestamp_subtraction_tolerance_s(
                    execution_start_time_s, execution_end_time_s)):
            raise ValueError(
                f"critical path step {step_id!r} contributions "
                "disagree with duration")
        cursor_s = execution_start_time_s
        segments = []
        if entry.scheduler_plan_cpu_s > 0.0:
            segments.append({
                "type": "host:scheduler_plan",
                "category": "host:scheduler_plan",
                "call_id": None,
                "pool_name": pool_name,
                "rank": None,
                "resource": None,
                "kernel_kind": None,
                "bottleneck_group": None,
                "predecessor_cause": "scheduler_boundary",
                "start_time_s": entry.start_time_s,
                "end_time_s": scheduler_plan_end_time_s,
                "duration_s": entry.scheduler_plan_cpu_s,
            })
        if entry.device_queue_wait_s > 0.0:
            segments.append({
                "type": "scheduler_device_queue_wait",
                "category": "scheduler_device_queue_wait",
                "call_id": None,
                "pool_name": pool_name,
                "rank": None,
                "resource": None,
                "kernel_kind": None,
                "bottleneck_group": None,
                "predecessor_cause": "scheduler_pipeline_slot",
                "start_time_s": scheduler_plan_end_time_s,
                "end_time_s": execution_start_time_s,
                "duration_s": entry.device_queue_wait_s,
            })
        ordered_contributions = sorted(contributions.items())
        for index, (category, duration_s) in enumerate(
                ordered_contributions):
            end_time_s = (
                execution_end_time_s
                if index == len(ordered_contributions) - 1
                else cursor_s + duration_s
            )
            segments.append({
                "type": (
                    ("kernel_repeat_aggregate" if compact
                     else "kernel_timeline_aggregate")
                    if category.startswith("kernel:")
                    else category
                ),
                "category": category,
                "call_id": None,
                "pool_name": pool_name,
                "rank": None,
                "resource": None,
                "kernel_kind": None,
                "bottleneck_group": (
                    category[len("kernel:"):]
                    if category.startswith("kernel:")
                    else None
                ),
                "predecessor_cause": (
                    "compact_repeat_transition" if compact
                    else "execution_timeline_contribution"),
                "start_time_s": cursor_s,
                "end_time_s": end_time_s,
                "duration_s": end_time_s - cursor_s,
            })
            cursor_s = end_time_s
        if entry.completion_queue_wait_s > 0.0:
            segments.append({
                "type": "scheduler_completion_queue_wait",
                "category": "scheduler_completion_queue_wait",
                "call_id": None,
                "pool_name": pool_name,
                "rank": None,
                "resource": None,
                "kernel_kind": None,
                "bottleneck_group": None,
                "predecessor_cause": "scheduler_pipeline_slot",
                "start_time_s": execution_end_time_s,
                "end_time_s": scheduler_update_start_time_s,
                "duration_s": entry.completion_queue_wait_s,
            })
        if entry.scheduler_update_cpu_s > 0.0:
            segments.append({
                "type": "host:scheduler_update",
                "category": "host:scheduler_update",
                "call_id": None,
                "pool_name": pool_name,
                "rank": None,
                "resource": None,
                "kernel_kind": None,
                "bottleneck_group": None,
                "predecessor_cause": "scheduler_boundary",
                "start_time_s": scheduler_update_start_time_s,
                "end_time_s": entry.end_time_s,
                "duration_s": entry.scheduler_update_cpu_s,
            })
        contributions = _segment_contributions(segments)
        _validate_contiguous_segments(
            segments,
            entry.start_time_s,
            entry.end_time_s,
            f"critical path step {step_id!r}",
        )
        _validate_absolute_interval(
            entry.start_time_s,
            entry.end_time_s,
            observation_window_s,
            f"critical path step {step_id!r}",
        )
        return {
            "step_id": step_id,
            "pool_name": pool_name,
            "request_ids": tuple(
                request.request_id
                for request in step.backend_step.schedule.requests
            ),
            "start_time_s": entry.start_time_s,
            "end_time_s": entry.end_time_s,
            "duration_s": entry_duration_s,
            "representation": (
                "compact_repeat_transition_v1" if compact
                else "timeline_contributions_v1"),
            "segment_time_basis": "category_packing_not_event_chronology",
            "canonical_path_unique": (
                step.execution_timeline.critical_path_unique),
            "critical_call_ids": (),
            "critical_call_count": (
                step.execution_timeline.critical_call_count),
            "total_call_count": step.kernel_plan.logical_call_count,
            "segments": segments,
            "contribution_s_by_category": contributions,
            "dominant_category": _dominant_category(contributions),
        }
    calls = step.kernel_plan.topological_order()
    calls_by_id = {call.call_id: call for call in calls}
    intervals_by_id = {
        interval.call_id: interval
        for interval in step.execution_timeline.intervals
    }
    if len(intervals_by_id) != len(step.execution_timeline.intervals):
        raise ValueError(
            f"critical path step {step_id!r} has duplicate intervals")
    call_ids = set(calls_by_id)
    if set(intervals_by_id) != call_ids:
        raise ValueError(
            f"critical path step {step_id!r} intervals do not cover calls")
    if set(step.predictions) != call_ids:
        raise ValueError(
            f"critical path step {step_id!r} predictions do not cover calls")

    predecessor_by_id = {}
    wait_by_id = {}
    cause_by_id = {}
    predecessor_tie_by_id = {}
    resource_history = {}
    for call in calls:
        interval = intervals_by_id[call.call_id]
        expected_reserved_resources = call_reserved_resource_ids(call)
        if interval.reserved_resource_ids != expected_reserved_resources:
            raise ValueError(
                f"critical path call {call.call_id!r} reserved resources "
                "disagree with its KernelCall")
        prediction = step.predictions[call.call_id]
        interval_duration_s = interval.end_time_s - interval.start_time_s
        if not math.isclose(
                interval_duration_s,
                prediction.t_total_s,
                rel_tol=1e-12,
                abs_tol=_timestamp_subtraction_tolerance_s(
                    execution_start_time_s + interval.start_time_s,
                    execution_start_time_s + interval.end_time_s)):
            raise ValueError(
                f"critical path call {call.call_id!r} interval disagrees "
                "with its prediction")
        candidates = {}
        for dependency in call.dependencies:
            candidates[dependency] = "dependency"
        # Only the latest eligible users of each reserved resource can set
        # this call's ready time. Keep all near-equal users so the canonical
        # predecessor tie semantics remain identical without rescanning every
        # earlier call in the step.
        for resource_id in interval.reserved_resource_ids:
            if resource_id not in resource_history:
                resource_history[resource_id] = []
            history = resource_history[resource_id]
            eligible_stop = bisect_right(
                history,
                interval.start_time_s + 1e-15,
                key=lambda item: item[0],
            )
            if eligible_stop == 0:
                continue
            resource_ready_time_s = history[eligible_stop - 1][0]
            close_tolerance_s = max(
                1e-12 * abs(resource_ready_time_s),
                1e-15,
            )
            close_start = bisect_left(
                history,
                resource_ready_time_s - close_tolerance_s,
                0,
                eligible_stop,
                key=lambda item: item[0],
            )
            for earlier_end_time_s, earlier_call_id in history[
                    close_start:eligible_stop]:
                if not math.isclose(
                        earlier_end_time_s,
                        resource_ready_time_s,
                        rel_tol=1e-12,
                        abs_tol=1e-15):
                    continue
                if (earlier_call_id in candidates
                        and candidates[earlier_call_id] == "dependency"):
                    candidates[earlier_call_id] = (
                        "dependency_and_resource")
                elif earlier_call_id not in candidates:
                    candidates[earlier_call_id] = "resource"
        ready_time_s = (
            max(intervals_by_id[identifier].end_time_s
                for identifier in candidates)
            if candidates else 0.0
        )
        if interval.start_time_s < ready_time_s - 1e-15:
            raise ValueError(
                f"critical path call {call.call_id!r} starts before a "
                "predecessor is ready")
        wait_s = max(0.0, interval.start_time_s - ready_time_s)
        if wait_s > 1e-15 and call.kernel_spec.transfer is None:
            raise ValueError(
                f"critical path call {call.call_id!r} contains unexplained "
                "non-transfer idle time")
        ready_candidates = sorted(
            identifier for identifier in candidates
            if math.isclose(
                intervals_by_id[identifier].end_time_s,
                ready_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )
        if ready_candidates:
            ready_candidates.sort(key=lambda identifier: (
                _PREDECESSOR_CAUSE_PRIORITY[candidates[identifier]],
                identifier,
            ))
            predecessor = ready_candidates[0]
            predecessor_by_id[call.call_id] = predecessor
            cause_by_id[call.call_id] = candidates[predecessor]
            predecessor_tie_by_id[call.call_id] = (
                len(ready_candidates) > 1)
        elif ready_time_s > 1e-15:
            raise ValueError(
                f"critical path call {call.call_id!r} has no ready "
                "predecessor")
        else:
            predecessor_by_id[call.call_id] = None
            cause_by_id[call.call_id] = None
            predecessor_tie_by_id[call.call_id] = False
        wait_by_id[call.call_id] = wait_s
        for resource_id in interval.reserved_resource_ids:
            history = resource_history[resource_id]
            item = (interval.end_time_s, call.call_id)
            if not history or history[-1] <= item:
                history.append(item)
            else:
                insort_right(history, item)

    final_time_s = max(
        interval.end_time_s for interval in intervals_by_id.values())
    if not math.isclose(
            final_time_s,
            step.execution_timeline.duration_s,
            rel_tol=1e-12,
            abs_tol=1e-15):
        raise ValueError(
            f"critical path step {step_id!r} final interval disagrees "
            "with timeline duration")
    final_candidates = sorted(
        identifier for identifier, interval in intervals_by_id.items()
        if math.isclose(
            interval.end_time_s,
            final_time_s,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    )
    unique = len(final_candidates) == 1
    current_id = final_candidates[0]
    reversed_segments = []
    visited = set()
    while current_id is not None:
        if current_id in visited:
            raise ValueError("critical path predecessor chain contains a cycle")
        visited.add(current_id)
        if predecessor_tie_by_id[current_id]:
            unique = False
        call = calls_by_id[current_id]
        interval = intervals_by_id[current_id]
        prediction = step.predictions[current_id]
        reversed_segments.append({
            "type": "kernel",
            "category": f"kernel:{prediction.bottleneck_group}",
            "call_id": current_id,
            "pool_name": call.pool_name,
            "rank": call.rank,
            "resource": call.resource,
            "kernel_kind": call.kernel_spec.kind.value,
            "bottleneck_group": prediction.bottleneck_group,
            "predecessor_cause": cause_by_id[current_id],
            "start_time_s": execution_start_time_s + interval.start_time_s,
            "end_time_s": execution_start_time_s + interval.end_time_s,
            "duration_s": interval.end_time_s - interval.start_time_s,
        })
        wait_s = wait_by_id[current_id]
        if wait_s > 1e-15:
            reversed_segments.append({
                "type": "shared_transfer_resource_wait",
                "category": "shared_transfer_resource_wait",
                "call_id": None,
                "pool_name": call.pool_name,
                "rank": call.rank,
                "resource": call.resource,
                "kernel_kind": None,
                "bottleneck_group": None,
                "predecessor_cause": "external_resource_calendar",
                "start_time_s": (
                    execution_start_time_s + interval.start_time_s - wait_s),
                "end_time_s": execution_start_time_s + interval.start_time_s,
                "duration_s": wait_s,
            })
        current_id = predecessor_by_id[current_id]
    segments = list(reversed(reversed_segments))
    if entry.scheduler_plan_cpu_s > 0.0:
        segments.insert(0, {
            "type": "host:scheduler_plan",
            "category": "host:scheduler_plan",
            "call_id": None,
            "pool_name": pool_name,
            "rank": None,
            "resource": None,
            "kernel_kind": None,
            "bottleneck_group": None,
            "predecessor_cause": "scheduler_boundary",
            "start_time_s": entry.start_time_s,
            "end_time_s": scheduler_plan_end_time_s,
            "duration_s": entry.scheduler_plan_cpu_s,
        })
    if entry.device_queue_wait_s > 0.0:
        segments.insert(1 if entry.scheduler_plan_cpu_s > 0.0 else 0, {
            "type": "scheduler_device_queue_wait",
            "category": "scheduler_device_queue_wait",
            "call_id": None,
            "pool_name": pool_name,
            "rank": None,
            "resource": None,
            "kernel_kind": None,
            "bottleneck_group": None,
            "predecessor_cause": "scheduler_pipeline_slot",
            "start_time_s": scheduler_plan_end_time_s,
            "end_time_s": execution_start_time_s,
            "duration_s": entry.device_queue_wait_s,
        })
    if entry.completion_queue_wait_s > 0.0:
        segments.append({
            "type": "scheduler_completion_queue_wait",
            "category": "scheduler_completion_queue_wait",
            "call_id": None,
            "pool_name": pool_name,
            "rank": None,
            "resource": None,
            "kernel_kind": None,
            "bottleneck_group": None,
            "predecessor_cause": "scheduler_pipeline_slot",
            "start_time_s": execution_end_time_s,
            "end_time_s": scheduler_update_start_time_s,
            "duration_s": entry.completion_queue_wait_s,
        })
    if entry.scheduler_update_cpu_s > 0.0:
        segments.append({
            "type": "host:scheduler_update",
            "category": "host:scheduler_update",
            "call_id": None,
            "pool_name": pool_name,
            "rank": None,
            "resource": None,
            "kernel_kind": None,
            "bottleneck_group": None,
            "predecessor_cause": "scheduler_boundary",
            "start_time_s": scheduler_update_start_time_s,
            "end_time_s": entry.end_time_s,
            "duration_s": entry.scheduler_update_cpu_s,
        })
    _validate_contiguous_segments(
        segments,
        entry.start_time_s,
        entry.end_time_s,
        f"critical path step {step_id!r}",
    )
    contributions = _segment_contributions(segments)
    _validate_absolute_interval(
        entry.start_time_s,
        entry.end_time_s,
        observation_window_s,
        f"critical path step {step_id!r}",
    )
    return {
        "step_id": step_id,
        "pool_name": pool_name,
        "request_ids": tuple(
            request.request_id
            for request in step.backend_step.schedule.requests
        ),
        "start_time_s": entry.start_time_s,
        "end_time_s": entry.end_time_s,
        "duration_s": entry_duration_s,
        "canonical_path_unique": unique,
        "critical_call_ids": tuple(
            segment["call_id"]
            for segment in segments
            if segment["type"] == "kernel"
        ),
        "critical_call_count": sum(
            1 for segment in segments if segment["type"] == "kernel"),
        "total_call_count": len(calls),
        "segments": segments,
        "contribution_s_by_category": contributions,
        "dominant_category": _dominant_category(contributions),
    }


def _request_execution_blocks(
    request_id,
    deployment,
    result,
    entries_by_step_id,
    step_path_by_id,
    observation_window_s,
):
    blocks = []
    for decision in result.schedule_trace.route_decisions:
        if decision.request_id != request_id:
            continue
        if decision.deployment_id != deployment.deployment_id:
            raise ValueError(
                "critical path route decision deployment ID disagrees with "
                "the deployment plan")
        service_time_s = (
            decision.dispatch_time_s - decision.queue_exit_time_s)
        if not math.isclose(
                service_time_s,
                deployment.router.dispatch_service_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15):
            raise ValueError(
                "critical path route dispatch service disagrees with the "
                "deployment router config")
        _validate_absolute_interval(
            decision.queue_enter_time_s,
            decision.dispatch_time_s,
            observation_window_s,
            f"critical path route decision for {request_id!r}",
        )
        if decision.dispatch_time_s > decision.queue_enter_time_s + 1e-15:
            blocks.append({
                "type": "router_dispatch",
                "step_id": None,
                "stage": decision.stage,
                "resource": deployment.router.resource_id,
                "start_time_s": decision.queue_enter_time_s,
                "queue_exit_time_s": decision.queue_exit_time_s,
                "end_time_s": decision.dispatch_time_s,
            })
    for step_id, entry in entries_by_step_id.items():
        matching_requests = tuple(
            request
            for request in entry.backend_step.schedule.requests
            if request.request_id == request_id
        )
        if matching_requests:
            scheduled, = matching_requests
            if (
                is_non_mtp_async_terminal_placeholder(scheduled)
                and entry.backend_step.generated_tokens[request_id] == 0
                and request_id not in entry.backend_step.finished_request_ids
            ):
                continue
            path = step_path_by_id[step_id]
            blocks.append({
                "type": "step",
                "step_id": step_id,
                "start_time_s": entry.start_time_s,
                "end_time_s": entry.end_time_s,
                "segments": path["segments"],
            })
    for transfer in result.schedule_trace.transfers:
        if transfer.payload != "kv" or request_id not in transfer.request_ids:
            continue
        simulation_matches = [
            candidate for candidate in result.transfers
            if candidate.request_id == request_id
            and candidate.source_engine_id == transfer.source_pool_id
            and candidate.destination_engine_id == transfer.destination_pool_id
            and len(candidate.workload.nodes) == 1
            and candidate.workload.nodes[0].node_id == transfer.transfer_id
        ]
        if len(simulation_matches) != 1:
            raise ValueError(
                "critical path requires exactly one modeled physical PD "
                f"transfer for {transfer.transfer_id!r}")
        simulation_transfer = simulation_matches[0]
        route_matches = [
            route for route in deployment.routes
            if route.source_pool == transfer.source_pool_id
            and route.destination_pool == transfer.destination_pool_id
            and route.payload == "kv"
            and route.link == transfer.link
            and route.resource_id == transfer.resource_id
        ]
        if len(route_matches) != 1:
            raise ValueError(
                "critical path PD transfer must match exactly one deployment "
                "route by source, destination, payload, link, and resource")
        route = route_matches[0]
        calls = simulation_transfer.kernel_plan.calls
        if len(calls) != 1:
            raise ValueError(
                "critical path PD transfer must lower to exactly one call")
        call = calls[0]
        if call.pool_name != transfer.source_pool_id:
            raise ValueError(
                "critical path PD transfer call has the wrong source pool")
        if call.resource != transfer.resource_id:
            raise ValueError(
                "critical path PD transfer call resource disagrees with trace")
        source_pool = deployment.worker_pools[transfer.source_pool_id]
        destination_pool = deployment.worker_pools[
            transfer.destination_pool_id]
        transfer_node = simulation_transfer.workload.nodes[0]
        rank_pairs = tuple(
            (rank_pair.source_rank, rank_pair.destination_rank)
            for rank_pair in pd_kv_transfer_rank_pairs(
                source_pool,
                destination_pool,
                transfer_node.attributes["num_key_value_heads"],
            )
        )
        physical_transfer = call.kernel_spec.transfer
        if physical_transfer is None:
            raise ValueError(
                "critical path PD transfer call lacks a physical transfer")
        expected_participants = physical_transfer.participants
        if expected_participants not in rank_pairs:
            raise ValueError(
                "critical path PD transfer participants disagree with "
                "deployment rank lanes")
        if call.rank != expected_participants[0]:
            raise ValueError(
                "critical path PD transfer call has the wrong source rank")
        if physical_transfer.bytes != transfer.byte_count:
            raise ValueError(
                "critical path PD transfer bytes disagree with trace")
        if physical_transfer.link != transfer.link:
            raise ValueError(
                "critical path PD transfer link disagrees with trace")
        if physical_transfer.resource_id != transfer.resource_id:
            raise ValueError(
                "critical path PD transfer resource disagrees with trace")
        if physical_transfer.direction != "point_to_point":
            raise ValueError(
                "critical path PD transfer must be point-to-point")
        if (physical_transfer.algorithm != route.algorithm
                or physical_transfer.topology != route.topology):
            raise ValueError(
                "critical path PD transfer algorithm/topology disagree with "
                "the deployment route")
        if set(simulation_transfer.predictions) != {call.call_id}:
            raise ValueError(
                "critical path PD transfer prediction does not cover its call")
        prediction = simulation_transfer.predictions[call.call_id]
        duration_s = transfer.end_time_s - transfer.start_time_s
        timeline = simulation_transfer.execution_timeline
        if len(timeline.intervals) != 1:
            raise ValueError(
                "critical path PD transfer timeline must contain one interval")
        interval = timeline.intervals[0]
        if (interval.call_id != call.call_id
                or interval.pool_name != call.pool_name
                or interval.rank != call.rank
                or interval.resource != call.resource
                or not math.isclose(
                    interval.end_time_s, timeline.duration_s,
                    rel_tol=1e-12, abs_tol=1e-15)):
            raise ValueError(
                "critical path PD transfer timeline disagrees with its call")
        if interval.reserved_resource_ids != transfer.reserved_resource_ids:
            raise ValueError(
                "critical path PD transfer reserved resources disagree with "
                "trace")
        if not math.isclose(
                duration_s,
                prediction.t_total_s,
                rel_tol=1e-12,
                abs_tol=_timestamp_subtraction_tolerance_s(
                    transfer.start_time_s, transfer.end_time_s)):
            raise ValueError(
                "critical path PD transfer trace disagrees with prediction")
        if not math.isclose(
                interval.end_time_s - interval.start_time_s,
                prediction.t_total_s,
                rel_tol=1e-12,
                abs_tol=1e-15):
            raise ValueError(
                "critical path PD transfer timeline disagrees with prediction")
        if timeline.host_intervals:
            decisions = tuple(
                decision for decision in result.schedule_trace.route_decisions
                if decision.request_id == request_id
                and decision.stage == "prefill_to_decode"
                and decision.source_engine_id == transfer.source_pool_id
                and decision.selected_engine_id == transfer.destination_pool_id)
            if len(decisions) != 1:
                raise ValueError(
                    "critical path PD submission requires one handoff and Host timeline")
            decision, = decisions
            submission_end_s = decision.dispatch_time_s + max(
                event.end_time_s for event in timeline.host_intervals)
            if submission_end_s > transfer.start_time_s + 1e-15:
                raise ValueError(
                    "critical path PD transfer begins before Host submission completes")
            blocks.append({
                "type": "host_pd_transfer_submit",
                "step_id": None,
                "transfer_id": transfer.transfer_id,
                "start_time_s": decision.dispatch_time_s,
                "end_time_s": submission_end_s,
            })
        _validate_absolute_interval(
            transfer.start_time_s,
            transfer.end_time_s,
            observation_window_s,
            f"critical path transfer {transfer.transfer_id!r}",
        )
        blocks.append({
            "type": "pd_kv_transfer",
            "step_id": None,
            "transfer_id": transfer.transfer_id,
            "start_time_s": transfer.start_time_s,
            "end_time_s": transfer.end_time_s,
            "source_pool_id": transfer.source_pool_id,
            "destination_pool_id": transfer.destination_pool_id,
            "resource": transfer.resource_id,
            "kernel_kind": call.kernel_spec.kind.value,
            "bottleneck_group": prediction.bottleneck_group,
        })
    blocks.sort(key=lambda block: (
        block["start_time_s"],
        block["end_time_s"],
        block["type"],
    ))
    normalized_blocks = []
    previous_end = None
    for block in blocks:
        if (previous_end is not None
                and block["start_time_s"] < previous_end - 1e-15):
            if (block["type"] != "step"
                    or block["end_time_s"] <= previous_end + 1e-15):
                raise ValueError(
                    f"critical path request {request_id!r} execution overlaps")
            # With AsyncScheduler, planning and device-queue residence for the
            # next step overlap the preceding in-flight step.  Those prefixes
            # are real system activity but are not additive on one request's
            # causal latency path.  Retain only the non-overlapping suffix.
            clipped_segments = []
            for segment in block["segments"]:
                if segment["end_time_s"] <= previous_end + 1e-15:
                    continue
                clipped = dict(segment)
                if clipped["start_time_s"] < previous_end:
                    clipped["start_time_s"] = previous_end
                    clipped["duration_s"] = (
                        clipped["end_time_s"] - previous_end)
                clipped_segments.append(clipped)
            if (not clipped_segments
                    or not math.isclose(
                        clipped_segments[0]["start_time_s"],
                        previous_end,
                        rel_tol=1e-12,
                        abs_tol=1e-15)):
                raise ValueError(
                    "critical path async overlap has no causal step suffix")
            block = dict(block)
            block["start_time_s"] = previous_end
            block["segments"] = clipped_segments
        normalized_blocks.append(block)
        previous_end = block["end_time_s"]
    return normalized_blocks


def _request_target_path(
    request_id,
    arrival_time_s,
    target_time_s,
    blocks,
    target_name,
    *,
    canonical_terminal_time_s=None,
):
    if canonical_terminal_time_s is not None:
        if not math.isclose(
                target_time_s,
                canonical_terminal_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15):
            raise ValueError(
                f"critical path {target_name} metric target for "
                f"{request_id!r} disagrees with its canonical terminal "
                "completion")
        target_time_s = canonical_terminal_time_s
    if target_time_s < arrival_time_s - 1e-15:
        raise ValueError(
            f"critical path {target_name} precedes arrival for {request_id!r}")
    included = []
    ttft_delivery_block = None
    for block in blocks:
        if block["end_time_s"] <= target_time_s + 1e-15:
            included.append(block)
        elif block["start_time_s"] < target_time_s - 1e-15:
            if target_name == "ttft" and included:
                previous_block = included[-1]
                if previous_block["type"] != "step":
                    raise ValueError(
                        "critical path ttft delivery does not follow a "
                        "scheduler step")
                ttft_delivery_block = {
                    "type": "host_first_token_delivery",
                    "start_time_s": previous_block["end_time_s"],
                    "end_time_s": target_time_s,
                }
                break
            raise ValueError(
                f"critical path {target_name} cuts through an execution block")
    if (target_name == "ttft"
            and ttft_delivery_block is None
            and included
            and included[-1]["end_time_s"] < target_time_s - 1e-15):
        previous_block = included[-1]
        if previous_block["type"] != "step":
            raise ValueError(
                "critical path ttft delivery does not follow a scheduler step")
        ttft_delivery_block = {
            "type": "host_first_token_delivery",
            "start_time_s": previous_block["end_time_s"],
            "end_time_s": target_time_s,
        }
    if ttft_delivery_block is not None:
        included.append(ttft_delivery_block)
    if (not included
            or not math.isclose(
                included[-1]["end_time_s"],
                target_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )):
        raise ValueError(
            f"critical path {target_name} for {request_id!r} has no "
            "terminal execution block")
    cursor_s = arrival_time_s
    segments = []
    for block in included:
        if block["start_time_s"] < cursor_s - 1e-15:
            raise ValueError(
                f"critical path blocks overlap for request {request_id!r}")
        if block["start_time_s"] > cursor_s + 1e-15:
            if block["type"] == "pd_kv_transfer":
                wait_type = "transfer_resource_wait"
                resource = block["resource"]
            elif block["type"] in (
                    "router_dispatch", "step", "host_pd_transfer_submit"):
                wait_type = "not_scheduled_wait"
                resource = None
            else:
                raise ValueError(
                    f"critical path found unknown execution block: "
                    f"{block['type']!r}")
            segments.append({
                "type": wait_type,
                "category": wait_type,
                "step_id": None,
                "call_id": None,
                "resource": resource,
                "kernel_kind": None,
                "bottleneck_group": None,
                "start_time_s": cursor_s,
                "end_time_s": block["start_time_s"],
                "duration_s": block["start_time_s"] - cursor_s,
            })
        if block["type"] == "router_dispatch":
            queue_exit_time_s = block["queue_exit_time_s"]
            if queue_exit_time_s > block["start_time_s"] + 1e-15:
                segments.append({
                    "type": "router_queue_wait",
                    "category": "router_queue_wait",
                    "step_id": None,
                    "call_id": None,
                    "stage": block["stage"],
                    "resource": block["resource"],
                    "kernel_kind": None,
                    "bottleneck_group": None,
                    "start_time_s": block["start_time_s"],
                    "end_time_s": queue_exit_time_s,
                    "duration_s": (
                        queue_exit_time_s - block["start_time_s"]),
                })
            if block["end_time_s"] > queue_exit_time_s + 1e-15:
                segments.append({
                    "type": "router_dispatch_service",
                    "category": "router_dispatch_service",
                    "step_id": None,
                    "call_id": None,
                    "stage": block["stage"],
                    "resource": block["resource"],
                    "kernel_kind": None,
                    "bottleneck_group": None,
                    "start_time_s": queue_exit_time_s,
                    "end_time_s": block["end_time_s"],
                    "duration_s": block["end_time_s"] - queue_exit_time_s,
                })
        elif block["type"] == "step":
            step_contributions = _segment_contributions(block["segments"])
            segments.append({
                "type": "shared_batch_step",
                "category": None,
                "step_id": block["step_id"],
                "call_id": None,
                "resource": None,
                "kernel_kind": None,
                "bottleneck_group": None,
                "request_attribution": "shared_batch_wall_clock",
                "step_contribution_s_by_category": step_contributions,
                "start_time_s": block["start_time_s"],
                "end_time_s": block["end_time_s"],
                "duration_s": block["end_time_s"] - block["start_time_s"],
            })
        elif block["type"] == "pd_kv_transfer":
            segments.append({
                "type": "pd_kv_transfer",
                "category": f"kernel:{block['bottleneck_group']}",
                "step_id": None,
                "call_id": None,
                "transfer_id": block["transfer_id"],
                "source_pool_id": block["source_pool_id"],
                "destination_pool_id": block["destination_pool_id"],
                "resource": block["resource"],
                "kernel_kind": block["kernel_kind"],
                "bottleneck_group": block["bottleneck_group"],
                "start_time_s": block["start_time_s"],
                "end_time_s": block["end_time_s"],
                "duration_s": block["end_time_s"] - block["start_time_s"],
            })
        elif block["type"] == "host_pd_transfer_submit":
            segments.append({
                "type": "host:launch",
                "category": "host:launch",
                "step_id": None,
                "call_id": None,
                "transfer_id": block["transfer_id"],
                "resource": None,
                "kernel_kind": None,
                "bottleneck_group": None,
                "start_time_s": block["start_time_s"],
                "end_time_s": block["end_time_s"],
                "duration_s": block["end_time_s"] - block["start_time_s"],
            })
        elif block["type"] == "host_first_token_delivery":
            segments.append({
                "type": "host_first_token_delivery",
                "category": "host:first_token_delivery",
                "step_id": None,
                "call_id": None,
                "resource": None,
                "kernel_kind": None,
                "bottleneck_group": None,
                "start_time_s": block["start_time_s"],
                "end_time_s": block["end_time_s"],
                "duration_s": block["end_time_s"] - block["start_time_s"],
            })
        else:
            raise ValueError(
                f"critical path found unknown execution block: "
                f"{block['type']!r}")
        cursor_s = block["end_time_s"]
    if not math.isclose(
            cursor_s, target_time_s,
            rel_tol=1e-12,
            abs_tol=1e-15):
        raise ValueError(
            f"critical path {target_name} for {request_id!r} is incomplete")
    _validate_contiguous_segments(
        segments,
        arrival_time_s,
        target_time_s,
        f"critical path {target_name} for {request_id!r}",
    )
    contributions = _segment_contributions(segments)
    return {
        "availability": "available",
        "duration_s": target_time_s - arrival_time_s,
        "target_time_s": target_time_s,
        "segments": segments,
        "contribution_s_by_category": contributions,
        "dominant_category": _dominant_category(contributions),
    }


def _timestamp_subtraction_tolerance_s(
    start_time_s: float,
    end_time_s: float,
) -> float:
    timestamp_scale_s = max(abs(start_time_s), abs(end_time_s))
    return 8.0 * math.ulp(max(1.0, timestamp_scale_s))


def _validate_contiguous_segments(
    segments,
    expected_start_s,
    expected_end_s,
    label,
):
    if not segments:
        if math.isclose(
                expected_start_s, expected_end_s,
                rel_tol=1e-12, abs_tol=1e-15):
            return
        raise ValueError(f"{label} has no segments")
    cursor_s = expected_start_s
    for segment in segments:
        if not math.isclose(
                segment["start_time_s"], cursor_s,
                rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"{label} segments are not contiguous")
        segment_start_s = segment["start_time_s"]
        segment_end_s = segment["end_time_s"]
        duration_s = segment_end_s - segment_start_s
        subtraction_tolerance_s = _timestamp_subtraction_tolerance_s(
            segment_start_s, segment_end_s)
        if duration_s < -subtraction_tolerance_s or not math.isclose(
                segment["duration_s"], max(0.0, duration_s),
                rel_tol=1e-12, abs_tol=subtraction_tolerance_s):
            raise ValueError(f"{label} segment duration is invalid")
        cursor_s = segment["end_time_s"]
    if not math.isclose(
            cursor_s, expected_end_s,
            rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{label} does not cover its full interval")


def _segment_contributions(segments):
    contributions = {}
    for segment in segments:
        if segment["type"] == "shared_batch_step":
            additions = segment["step_contribution_s_by_category"]
        elif segment["type"] in (
                "kernel",
                "kernel_repeat_aggregate",
                "kernel_timeline_aggregate",
                "router_queue_wait",
                "router_dispatch_service",
                "host:launch",
                "device:launch_gap",
                "host:scheduler_plan",
                "scheduler_device_queue_wait",
                "scheduler_completion_queue_wait",
                "host:scheduler_update",
                "host_first_token_delivery",
                "shared_transfer_resource_wait",
                "not_scheduled_wait",
                "transfer_resource_wait",
                "pd_kv_transfer"):
            additions = {segment["category"]: segment["duration_s"]}
        else:
            raise ValueError(
                f"critical path found unknown contribution segment: "
                f"{segment['type']!r}")
        for category, duration_s in additions.items():
            if category in contributions:
                contributions[category] += duration_s
            else:
                contributions[category] = duration_s
    return contributions


def _dominant_category(contributions):
    if not contributions:
        return None
    return sorted(
        contributions,
        key=lambda category: (-contributions[category], category),
    )[0]


def _queue_manifests(
    deployment: DeploymentPlan,
    requests: RequestStream,
    result: SimulationResult,
) -> tuple[dict[str, object], dict[str, object]]:
    entries = result.schedule_trace.entries
    decisions = result.schedule_trace.route_decisions
    request_ids = {
        request.request_id for request in requests
    }
    router_samples_by_stage: dict[str, list[float]] = {}
    router_service_by_stage: dict[str, list[float]] = {}
    scheduler_samples_by_stage: dict[str, list[float]] = {}
    scheduler_samples_by_pool: dict[str, list[float]] = {}
    scheduler_samples = []
    if decisions:
        router_samples = []
        router_service_samples = []
        router_service_intervals = []
        scheduler_pool_names = {
            engine_id
            for decision in decisions
            for engine_id, queued_tokens in decision.candidate_queued_tokens
        }
        unknown_pools = scheduler_pool_names - set(deployment.worker_pools)
        if unknown_pools:
            raise ValueError(
                "system analysis route candidates reference unknown pools: "
                f"{sorted(unknown_pools)}")
        for pool_name in scheduler_pool_names:
            scheduler_samples_by_pool[pool_name] = []
        for decision in decisions:
            router_wait_s = (
                decision.queue_exit_time_s - decision.queue_enter_time_s)
            router_wait_s = _normalize_wait(router_wait_s, "router queue")
            router_samples.append(router_wait_s)
            if decision.stage not in router_samples_by_stage:
                router_samples_by_stage[decision.stage] = []
            router_samples_by_stage[decision.stage].append(router_wait_s)
            service_time_s = _normalize_wait(
                decision.dispatch_time_s - decision.queue_exit_time_s,
                "router dispatch service",
            )
            if not math.isclose(
                    service_time_s,
                    deployment.router.dispatch_service_time_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15):
                raise ValueError(
                    "system analysis route dispatch service disagrees with "
                    "the deployment router config")
            router_service_samples.append(service_time_s)
            if decision.stage not in router_service_by_stage:
                router_service_by_stage[decision.stage] = []
            router_service_by_stage[decision.stage].append(service_time_s)
            router_service_intervals.append((
                decision.queue_exit_time_s,
                decision.dispatch_time_s,
            ))
            if decision.request_id not in request_ids:
                raise ValueError(
                    "system analysis route decision references unknown request")
            admission_time_s = _scheduler_admission_time(
                decision,
                result,
            )
            schedule_start_s = _first_schedule_start(
                decision.request_id,
                decision.selected_engine_id,
                admission_time_s,
                entries,
            )
            scheduler_wait_s = schedule_start_s - admission_time_s
            scheduler_wait_s = _normalize_wait(
                scheduler_wait_s, "scheduler queue")
            scheduler_samples.append(scheduler_wait_s)
            if decision.stage not in scheduler_samples_by_stage:
                scheduler_samples_by_stage[decision.stage] = []
            scheduler_samples_by_stage[decision.stage].append(
                scheduler_wait_s)
            scheduler_samples_by_pool[decision.selected_engine_id].append(
                scheduler_wait_s)
        _validate_exclusive_resource_intervals(
            router_service_intervals,
            deployment.router.resource_id,
        )
        router_busy_time_s = _interval_union_duration(
            router_service_intervals,
            result.schedule_trace.end_time_s,
        )
        router_queue = {
            "availability": "modeled_serial_dispatch_resource",
            "timing_semantics": (
                "least_load_decision_then_serial_dispatch_queue_and_service"),
            "resource_id": deployment.router.resource_id,
            "configured_dispatch_service_time_s": (
                deployment.router.dispatch_service_time_s),
            "overall": _wait_summary(router_samples),
            "by_stage": {
                stage: _wait_summary(router_samples_by_stage[stage])
                for stage in sorted(router_samples_by_stage)
            },
            "dispatch_service": {
                "overall": _wait_summary(router_service_samples),
                "by_stage": {
                    stage: _wait_summary(router_service_by_stage[stage])
                    for stage in sorted(router_service_by_stage)
                },
                "busy_time_s": router_busy_time_s,
                "idle_time_s": _nonnegative_difference(
                    result.schedule_trace.end_time_s,
                    router_busy_time_s),
                "utilization": (
                    router_busy_time_s
                    / result.schedule_trace.end_time_s),
            },
        }
    else:
        if (deployment.mode != "colocated"
                or len(deployment.worker_pools) != 1):
            raise ValueError(
                "system analysis found no route decisions outside the "
                "single-pool colocated path")
        pool_name = next(iter(deployment.worker_pools))
        stage = "ingress_to_colocated"
        scheduler_samples_by_pool[pool_name] = []
        scheduler_samples_by_stage[stage] = []
        for request in requests:
            schedule_start_s = _first_schedule_start(
                request.request_id,
                pool_name,
                request.arrival_time_s,
                entries,
            )
            scheduler_wait_s = schedule_start_s - request.arrival_time_s
            scheduler_wait_s = _normalize_wait(
                scheduler_wait_s, "scheduler queue")
            scheduler_samples.append(scheduler_wait_s)
            scheduler_samples_by_pool[pool_name].append(scheduler_wait_s)
            scheduler_samples_by_stage[stage].append(scheduler_wait_s)
        router_queue = {
            "availability": "not_applicable_single_pool_without_router",
            "overall": _wait_summary(()),
            "by_stage": {},
        }
    scheduler_queue = {
        "availability": "available",
        "admission_basis": (
            "router_dispatch_or_kv_transfer_completion_or_single_pool_arrival"),
        "overall": _wait_summary(scheduler_samples),
        "by_stage": {
            stage: _wait_summary(scheduler_samples_by_stage[stage])
            for stage in sorted(scheduler_samples_by_stage)
        },
        "by_pool": {
            pool_name: _wait_summary(scheduler_samples_by_pool[pool_name])
            for pool_name in sorted(scheduler_samples_by_pool)
        },
    }
    return router_queue, scheduler_queue


def _scheduler_admission_time(decision, result: SimulationResult) -> float:
    if decision.stage in ("ingress_to_colocated", "ingress_to_prefill"):
        return decision.dispatch_time_s
    elif decision.stage == "prefill_to_decode":
        matches = [
            transfer for transfer in result.schedule_trace.transfers
            if transfer.payload == "kv"
            and transfer.source_pool_id == decision.source_engine_id
            and transfer.destination_pool_id == decision.selected_engine_id
            and decision.request_id in transfer.request_ids
        ]
        if not matches:
            raise ValueError(
                "system analysis requires KV transfers for each "
                "prefill_to_decode route decision")
        return max(transfer.end_time_s for transfer in matches)
    else:
        raise ValueError(
            f"system analysis found unknown route stage: {decision.stage!r}")


def _first_schedule_start(
    request_id: str,
    pool_name: str,
    admission_time_s: float,
    entries,
) -> float:
    candidates = []
    for entry in entries:
        if entry.backend_step.schedule.pool_name != pool_name:
            continue
        if entry.start_time_s + 1e-15 < admission_time_s:
            continue
        if any(
                scheduled.request_id == request_id
                for scheduled in entry.backend_step.schedule.requests):
            candidates.append(entry.start_time_s)
    if not candidates:
        raise ValueError(
            f"system analysis found no schedule admission for request "
            f"{request_id!r} on pool {pool_name!r}")
    return min(candidates)


def _wait_summary(values) -> dict[str, object]:
    normalized = tuple(float(value) for value in values)
    if normalized:
        return {
            "sample_count": len(normalized),
            "total_wait_s": sum(normalized),
            "mean_wait_s": sum(normalized) / len(normalized),
            "max_wait_s": max(normalized),
        }
    return {
        "sample_count": 0,
        "total_wait_s": 0.0,
        "mean_wait_s": None,
        "max_wait_s": None,
    }


def _normalize_wait(value: float, label: str) -> float:
    if (isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value < -1e-15):
        raise ValueError(f"system analysis {label} must be non-negative")
    return max(0.0, float(value))


def _validate_relative_interval(
    start_time_s: float,
    end_time_s: float,
    window_s: float,
    label: str,
    *,
    source_start_time_s: float,
    source_end_time_s: float,
) -> None:
    _validate_absolute_interval(
        start_time_s,
        end_time_s,
        window_s,
        label,
        boundary_tolerance_s=_timestamp_subtraction_tolerance_s(
            source_start_time_s, source_end_time_s),
    )


def _validate_absolute_interval(
    start_time_s: float,
    end_time_s: float,
    window_s: float,
    label: str,
    *,
    boundary_tolerance_s: float = 1e-15,
) -> None:
    if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            for value in (start_time_s, end_time_s)):
        raise ValueError(f"system analysis {label} must be finite")
    if (isinstance(boundary_tolerance_s, bool)
            or not isinstance(boundary_tolerance_s, Real)
            or not math.isfinite(boundary_tolerance_s)
            or boundary_tolerance_s < 0.0):
        raise ValueError(
            "system analysis interval boundary tolerance must be finite and "
            "non-negative")
    reversed_interval = (
        end_time_s < start_time_s
        and not math.isclose(
            end_time_s, start_time_s,
            rel_tol=1e-12, abs_tol=boundary_tolerance_s)
    )
    outside_window = (
        end_time_s > window_s
        and not math.isclose(
            end_time_s, window_s,
            rel_tol=1e-12, abs_tol=boundary_tolerance_s)
    )
    if (start_time_s < -boundary_tolerance_s
            or reversed_interval
            or outside_window):
        raise ValueError(
            f"system analysis {label} lies outside its observation window")


def _validate_exclusive_resource_intervals(intervals, resource_id: str) -> None:
    previous_end = None
    for start_time_s, end_time_s in sorted(intervals):
        if (previous_end is not None
                and start_time_s < previous_end - 1e-15):
            raise ValueError(
                f"exclusive resource {resource_id!r} has overlapping "
                "intervals")
        previous_end = end_time_s


def _interval_union_duration(intervals, observation_window_s: float) -> float:
    if not intervals:
        return 0.0
    normalized = sorted(intervals)
    for start_time_s, end_time_s in normalized:
        _validate_absolute_interval(
            start_time_s,
            end_time_s,
            observation_window_s,
            "utilization interval",
        )
    current_start, current_end = normalized[0]
    duration_s = 0.0
    for start_time_s, end_time_s in normalized[1:]:
        if start_time_s <= current_end:
            current_end = max(current_end, end_time_s)
        else:
            duration_s += current_end - current_start
            current_start = start_time_s
            current_end = end_time_s
    duration_s += current_end - current_start
    return duration_s


def _nonnegative_difference(total_s: float, part_s: float) -> float:
    difference = total_s - part_s
    if (difference < 0.0
            and not math.isclose(
                total_s, part_s, rel_tol=1e-12, abs_tol=1e-15)):
        raise ValueError("system analysis busy time exceeds observation window")
    return max(0.0, difference)
