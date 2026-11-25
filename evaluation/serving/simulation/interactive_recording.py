"""Optional bounded details plus complete resource accounting for the viewer.

Extends the existing execution artifact owner; default simulation does not
construct this recorder. Detail selection never limits aggregate accounting.
"""
from collections import Counter
from contextlib import contextmanager
import math
import os
import threading
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

from evaluation.serving.simulation.execution_artifact import (
    StreamingExecutionArtifactAccumulator, _step_manifest, _transfer_manifest, _identity_digest_is_immutable,
)
from src.kernel.resource_usage import kernel_resource_usage, kernel_link_usage, _project_resource_usage
from src.kernel.prediction_cache import _observe_public_resource_inputs
from src.kernel.memory_access_builder import memory_prediction_source_scope
from src.serving.simulation import StreamingSimulationResult


def run_recorded_simulation(simulation, recorder):
    """Own capture only while this simulation (including its workers) runs."""
    if recorder is None:
        return simulation.run()
    with recorder.observe_resource_inputs(simulation.hardware_by_id):
        return simulation.run()


def _union(intervals):
    result = []
    for start, end in sorted((start, end) for start, end in intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def _add_l2_demand(lane, usage, count):
    """Optional old evidence stays unknown; only explicit modeled reads count."""
    if ("l2_traffic" not in usage or usage["l2_traffic"]["availability"] == "unavailable"):
        lane["l2_unavailable"] += count
        return
    traffic = usage["l2_traffic"]
    if traffic["availability"] != "available":
        raise ValueError("unknown L2 traffic availability")
    lane["l2_read_bytes"] += count * traffic["read_bytes"]
    lane["l2_read_capacity"] = traffic["read_capacity_bytes_per_s"]
    lane["l2_available"] += count


def _empty_memory_traffic():
    return {name: {"read_bytes": 0.0, "write_bytes": 0.0, "capacity_bytes_per_s": None,
        "available": 0, "unavailable": 0} for name in ("l2", "hbm")}


def _add_memory_traffic(lane, usage, count):
    for name in ("l2", "hbm"):
        target = lane["memory_traffic"][name]
        if "memory_traffic" not in usage or usage["memory_traffic"][name]["availability"] == "unavailable":
            target["unavailable"] += count
            continue
        source = usage["memory_traffic"][name]
        if source["availability"] != "available":
            raise ValueError("unknown memory traffic availability")
        target["read_bytes"] += count * source["read_bytes"]
        target["write_bytes"] += count * source["write_bytes"]
        target["capacity_bytes_per_s"] = source["capacity_bytes_per_s"]
        target["available"] += count


def _add_model_bottleneck(lane, prediction, count, path_id):
    """Aggregate the predictor's declared group, never observed durations."""
    group = prediction.bottleneck_group
    groups = lane["model_bottleneck_groups"]
    if group not in groups:
        groups[group] = {"operation_count": 0, "predicted_operation_time_s": 0.0,
            "critical_paths": {}, "critical_path_unavailable_or_omitted": 0}
    groups[group]["operation_count"] += count
    groups[group]["predicted_operation_time_s"] += count * prediction.t_total_s
    paths = groups[group]["critical_paths"]
    if path_id is None or (path_id not in paths and len(paths) >= 32):
        groups[group]["critical_path_unavailable_or_omitted"] += count
    else:
        if path_id not in paths:
            paths[path_id] = {"operation_count": 0, "predicted_operation_time_s": 0.0}
        paths[path_id]["operation_count"] += count
        paths[path_id]["predicted_operation_time_s"] += count * prediction.t_total_s


class InteractiveRecording(StreamingExecutionArtifactAccumulator):
    """Keep request-aware bounded samples and every step's resource totals."""

    def __init__(self, hardware_by_id, max_detail_events=100000, *, detail_archive_dir=None):
        super().__init__()
        if type(max_detail_events) is not int or max_detail_events < 1:
            raise ValueError("max_detail_events must be a positive integer")
        self.compact_mode = False
        self.hardware = hardware_by_id
        self.max_detail_events = max_detail_events
        self.detail_events = 0
        self.total_events = 0
        self.detail_steps = []
        self.summaries = []
        self.resources = {}
        self.links = []
        self.recording_wall_s = 0.0
        self._usage_cache = {}
        self._capture_cost_by_worker = {}
        self._resource_observer_active = False
        self._usage_by_identity = {}
        self._recorded_origins = {}
        self._critical_path_lookup = {}
        self._critical_path_catalog = []
        self._prediction_path_ids = {}
        self._prediction_path_missing = {}
        self._detail_members = {}
        self._seen_request_ids = set()
        self._first_steps = {}
        self._latest_steps = {}
        self.detail_archive_dir = Path(detail_archive_dir).resolve() if detail_archive_dir is not None else None
        self._archive_records = []
        self._step_durations = {}
        self._step_event_counts = {}

    def _select_detail(self, step, count):
        """Prefer request coverage, phase coverage and endpoints within the cap.

        The remaining tie-breaks favor larger batches and a deterministic hash
        sample. No predicted duration or observation is used for selection.
        """
        schedule = step.backend_step.schedule
        step_id = schedule.step_id
        requests = frozenset(request.request_id for request in schedule.requests)
        self._seen_request_ids.update(requests)
        pools = tuple(sorted({call.pool_name for call in step.kernel_plan.calls}))
        if pools not in self._first_steps:
            self._first_steps[pools] = step_id
        self._latest_steps[pools] = step_id
        if count == 0 or count > self.max_detail_events:
            return False
        member = {"count": count, "requests": requests, "pools": pools,
            "phases": frozenset((pools, request.phase) for request in schedule.requests),
            "call_ids": frozenset(call.call_id for call in step.kernel_plan.calls),
            "duration_s": step.execution_timeline.duration_s,
            "priority": int.from_bytes(hashlib.sha256(step_id.encode()).digest()[:8], "big")}
        candidates = {**self._detail_members, step_id: member}
        total = self.detail_events + count
        while total > self.max_detail_events:
            request_counts = Counter(request for value in candidates.values() for request in value["requests"])
            phase_counts = Counter(phase for value in candidates.values() for phase in value["phases"])
            def priority(item):
                key, value = item
                return (sum(request_counts[request] == 1 for request in value["requests"]),
                    sum(phase_counts[phase] == 1 for phase in value["phases"]),
                    key in (self._first_steps[value["pools"]], self._latest_steps[value["pools"]]),
                    len(value["requests"]), value["priority"])
            victim, value = min(candidates.items(), key=priority)
            del candidates[victim]
            total -= value["count"]
        if step_id not in candidates:
            return False
        evicted = set(self._detail_members) - set(candidates)
        if evicted:
            self.detail_steps = [row for row in self.detail_steps if row["step_id"] not in evicted]
            for row in self.summaries:
                if row["step_id"] in evicted:
                    row["detail_retained"] = False
            retained_calls = {call_id for value in candidates.values() for call_id in value["call_ids"]}
            self.resources = {key: value for key, value in self.resources.items() if key in retained_calls}
            self._prediction_path_ids = {key: value for key, value in self._prediction_path_ids.items()
                if key in retained_calls}
            self._prediction_path_missing = {key: value for key, value in self._prediction_path_missing.items()
                if key in retained_calls}
        self._detail_members = candidates
        # record() adds the new member's events after constructing its manifest.
        self.detail_events = total - count
        return True

    def _critical_path_id(self, prediction):
        diagnostic = prediction.diagnostics
        if diagnostic is None or not diagnostic.critical_path:
            return None
        key = (diagnostic.critical_path, diagnostic.support_level, diagnostic.limitations)
        if key not in self._critical_path_lookup:
            if len(self._critical_path_catalog) >= 4096:
                return None
            self._critical_path_lookup[key] = str(len(self._critical_path_catalog))
            self._critical_path_catalog.append({"critical_path": list(diagnostic.critical_path),
                "support_level": diagnostic.support_level, "limitations": list(diagnostic.limitations)})
        return self._critical_path_lookup[key]

    def _record_prediction_path(self, call_id, prediction, path_id):
        if path_id is not None:
            self._prediction_path_ids[call_id] = path_id
        elif prediction.diagnostics is None or not prediction.diagnostics.critical_path:
            self._prediction_path_missing[call_id] = "Model did not provide an operator critical path"
        else:
            self._prediction_path_missing[call_id] = "Critical-path catalog reached its recording limit"

    @contextmanager
    def observe_resource_inputs(self, hardware_by_id):
        """Bind this run's recorder; fork and copied thread contexts inherit it.

        Only projected usage enters the existing bounded cache. Public inputs
        and HWSpec are consumed synchronously and never saved for later lookup.
        """
        if self._resource_observer_active:
            raise RuntimeError("report resource observer is already active")
        self.hardware.update(hardware_by_id)
        self._resource_observer_active = True
        try:
            with _observe_public_resource_inputs(self._observe_resource_input):
                yield
        finally:
            # Full-retention runs project after Simulation returns. Their parent
            # capture cost is charged here; worker costs travel in projections.
            self.recording_wall_s += self._take_capture_cost()
            self._capture_cost_by_worker.clear()
            self._resource_observer_active = False

    def _observe_resource_input(self, spec, hardware, value, *, capture_wall_s):
        started = time.perf_counter()
        if (type(capture_wall_s) not in (int, float)
                or not math.isfinite(capture_wall_s) or capture_wall_s < 0):
            raise ValueError("report capture cost must be finite and nonnegative")
        try:
            if (spec.hardware_id not in self.hardware
                    or self.hardware[spec.hardware_id] is not hardware):
                return
            # A mathematical-only launch must retain the original unavailable
            # export; captured analytical work is not native launch evidence.
            if spec.launch is None:
                return
            key = repr(spec)
            if key not in self._usage_cache:
                usage = _project_resource_usage(spec, hardware, *value)
                if len(self._usage_cache) >= 4096:
                    self._usage_cache.clear()
                self._usage_cache[key] = usage
        finally:
            owner = (os.getpid(), threading.get_ident())
            if owner not in self._capture_cost_by_worker:
                self._capture_cost_by_worker[owner] = 0.0
            self._capture_cost_by_worker[owner] += capture_wall_s + time.perf_counter() - started

    def _take_capture_cost(self):
        owner = (os.getpid(), threading.get_ident())
        if owner not in self._capture_cost_by_worker:
            return 0.0
        cost = self._capture_cost_by_worker[owner]
        del self._capture_cost_by_worker[owner]
        return cost

    def _usage(self, spec):
        # Use the artifact owner's recursive immutability proof, since KernelSpec's
        # mapping wrapper alone does not freeze nested values. Retain proven objects
        # so Python cannot recycle their ids while this run-local cache lives.
        identity = id(spec)
        if identity in self._usage_by_identity:
            return self._usage_by_identity[identity][1]
        # The unchanged execution digest already recursively proves and retains
        # immutable specs. Reuse only that exact object's proof, never a digest
        # as a resource-equivalence key. Unproven/mutable specs are checked now.
        proven = identity in self._contract_digest_by_identity
        if proven and self._contract_digest_by_identity[identity][0] is not spec:
            raise RuntimeError("report resource contract identity collision")
        key = repr(spec)
        if key not in self._usage_cache:
            if len(self._usage_cache) >= 4096:
                self._usage_cache.clear()
            self._usage_cache[key] = kernel_resource_usage(spec, self.hardware[spec.hardware_id])
        usage = self._usage_cache[key]
        if proven or _identity_digest_is_immutable(spec):
            if len(self._usage_by_identity) >= 4096:
                self._usage_by_identity.clear()
            self._usage_by_identity[identity] = (spec, usage)
        return usage

    def record(self, step, entry):
        started = time.perf_counter()
        timeline = step.execution_timeline
        counts = dict(timeline.call_occurrence_counts) if timeline.block_repeats else {
            call.call_id: 1 for call in step.kernel_plan.calls}
        count = sum(counts.values()) + len(timeline.host_intervals)
        if self.detail_archive_dir is None:
            keep = self._select_detail(step, count)
        else:
            keep = True
            self._seen_request_ids.update(request.request_id for request in step.backend_step.schedule.requests)
        self._step_durations[step.backend_step.schedule.step_id] = timeline.duration_s
        self._step_event_counts[step.backend_step.schedule.step_id] = count
        self.total_events += count
        origin = entry.start_time_s + entry.scheduler_plan_cpu_s + entry.device_queue_wait_s
        self._recorded_origins[step.backend_step.schedule.step_id] = origin
        intervals = {}
        for interval in timeline.intervals:
            intervals[interval.call_id] = [(interval.start_time_s, interval.end_time_s)]
        for repeat in timeline.block_repeats:
            for call_id, spans in repeat.call_intervals:
                intervals[call_id] = list(spans)
        lanes = {}
        collective_participation = []
        for call in step.kernel_plan.calls:
            prediction = step.predictions[call.call_id]
            path_id = self._critical_path_id(prediction)
            if keep:
                self._record_prediction_path(call.call_id, prediction, path_id)
            key = (call.pool_name, call.rank)
            if key not in lanes:
                lanes[key] = {"pool_name": call.pool_name, "rank": call.rank,
                    "start_time_s": origin, "end_time_s": origin + timeline.duration_s,
                    "step_id": step.backend_step.schedule.step_id,
                    "busy_intervals": [], "compute": {}, "logical_bytes": 0.0,
                    "hbm_capacity": None, "available": 0, "unavailable": 0,
                    "l2_read_bytes": 0.0, "l2_read_capacity": None, "l2_available": 0, "l2_unavailable": 0,
                    "memory_traffic": _empty_memory_traffic(),
                    "device_events": 0, "detail_retained": keep, "operation_counts": {},
                    "model_bottleneck_groups": {},
                    "phases": sorted({r.phase for r in step.backend_step.schedule.requests}),
                    "host_intervals": [[origin + h.start_time_s, origin + h.end_time_s]
                        for h in timeline.host_intervals if h.pool_name == call.pool_name and h.rank == call.rank]}
            lane = lanes[key]
            if call.kernel_spec.transfer is not None and call.kernel_spec.kind.value == "COMMUNICATION":
                for participant in call.kernel_spec.transfer.participants:
                    if participant != call.rank:
                        collective_participation.append((call.pool_name, participant,
                            intervals[call.call_id], counts[call.call_id], key, prediction, path_id))
            lane["busy_intervals"].extend((origin + a, origin + b) for a, b in intervals[call.call_id])
            n = counts[call.call_id]
            lane["device_events"] += n
            _add_model_bottleneck(lane, prediction, n, path_id)
            kind = call.kernel_spec.kind.value
            if kind not in lane["operation_counts"]:
                lane["operation_counts"][kind] = 0
            lane["operation_counts"][kind] += n
            usage = self._usage(call.kernel_spec)
            _add_l2_demand(lane, usage, n)
            _add_memory_traffic(lane, usage, n)
            link = kernel_link_usage(call.kernel_spec, self.hardware[call.kernel_spec.hardware_id])
            if link["availability"] != "not_applicable":
                self.links.append({**link, "pool_name": call.pool_name,
                    "rank": call.rank, "call_id": call.call_id,
                    "step_id": step.backend_step.schedule.step_id,
                    "intervals": [[origin + a, origin + b] for a, b in intervals[call.call_id]]})
            if keep:
                self.resources[call.call_id] = usage
            if usage["availability"] == "unavailable":
                lane["unavailable"] += n
                continue
            lane["available"] += n
            lane["logical_bytes"] += n * usage["logical_bytes"]
            lane["hbm_capacity"] = usage["hbm_capacity_bytes_per_s"]
            for demand in usage["compute"]:
                role = usage["dtype"] + " / " + demand["role"]
                if role not in lane["compute"]:
                    lane["compute"][role] = {"operations": 0.0,
                        "capacity_operations_per_s": demand["capacity_operations_per_s"]}
                lane["compute"][role]["operations"] += n * demand["operations"]
        for pool, rank, spans, participation_count, source_key, prediction, path_id in collective_participation:
            key = (pool, rank)
            if key not in lanes:
                source = lanes[source_key]
                lanes[key] = {**source, "rank": rank, "busy_intervals": [], "compute": {},
                    "logical_bytes": 0.0, "available": 0, "unavailable": 0,
                    "l2_read_bytes": 0.0, "l2_read_capacity": None, "l2_available": 0, "l2_unavailable": 0,
                    "memory_traffic": _empty_memory_traffic(),
                    "device_events": 0, "operation_counts": {}, "host_intervals": [],
                    "model_bottleneck_groups": {}}
            lane = lanes[key]
            lane["busy_intervals"].extend((origin + a, origin + b) for a, b in spans)
            lane["device_events"] += participation_count
            lane["unavailable"] += participation_count
            lane["l2_unavailable"] += participation_count
            for memory in lane["memory_traffic"].values():
                memory["unavailable"] += participation_count
            _add_model_bottleneck(lane, prediction, participation_count, path_id)
            if "COMMUNICATION" not in lane["operation_counts"]:
                lane["operation_counts"]["COMMUNICATION"] = 0
            lane["operation_counts"]["COMMUNICATION"] += participation_count
        for lane in lanes.values():
            lane["busy_intervals"] = _union(lane["busy_intervals"])
            self.summaries.append(lane)
        if keep:
            manifest = _step_manifest(step, {step.backend_step.schedule.step_id: entry})
            if self.detail_archive_dir is None:
                self.detail_steps.append(manifest)
            else:
                from evaluation.reporting.serving_swimlanes import write_iteration_chunk
                record = write_iteration_chunk(manifest, self.resources,
                    {"by_call_id": self._prediction_path_ids, "missing_by_call_id": self._prediction_path_missing},
                    self.detail_archive_dir, len(self._archive_records))
                self._archive_records.append(record)
                self.resources = {}
                self._prediction_path_ids = {}
                self._prediction_path_missing = {}
            self.detail_events += count
        self.recording_wall_s += self._take_capture_cost() + time.perf_counter() - started

    def consume(self, step, entry):
        super().consume(step, entry)
        self.record(step, entry)

    def visualization_payload(self, payload, result):
        entries = {entry.backend_step.schedule.step_id: entry for entry in result.schedule_trace.entries}
        origins = {step_id: entry.start_time_s + entry.scheduler_plan_cpu_s + entry.device_queue_wait_s
            for step_id, entry in entries.items()}
        for row in self.summaries:
            delta = origins[row["step_id"]] - self._recorded_origins[row["step_id"]]
            row["start_time_s"] += delta
            row["end_time_s"] += delta
            for field in ("busy_intervals", "host_intervals"):
                if row[field] is not None:
                    row[field] = [[a + delta, b + delta] for a, b in row[field]]
        for link in self.links:
            delta = origins[link["step_id"]] - self._recorded_origins[link["step_id"]]
            link["intervals"] = [[a + delta, b + delta] for a, b in link["intervals"]]
        def shift(value, delta):
            if isinstance(value, dict):
                for name, child in value.items():
                    if name in ("absolute_start_time_s", "absolute_end_time_s", "origin_time_s"):
                        value[name] += delta
                    else:
                        shift(child, delta)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    shift(child, delta)
        for step in self.detail_steps:
            entry = entries[step["step_id"]]
            delta = origins[step["step_id"]] - self._recorded_origins[step["step_id"]]
            shift(step["execution_timeline"], delta)
            step["schedule_interval"] = {"start_time_s": entry.start_time_s, "end_time_s": entry.end_time_s}
            step["timing"] = {"mode": entry.timing_mode, "scheduler_plan_cpu_s": entry.scheduler_plan_cpu_s,
                "scheduler_update_cpu_s": entry.scheduler_update_cpu_s, "execution_wall_s": entry.execution_wall_s,
                "step_wall_s": entry.end_time_s - entry.start_time_s}
        schedule = [{"step_id": step_id, "pool_name": entry.backend_step.schedule.pool_name,
            "start_time_s": entry.start_time_s, "end_time_s": entry.end_time_s,
            "scheduler_plan_cpu_s": entry.scheduler_plan_cpu_s,
            "scheduler_update_cpu_s": entry.scheduler_update_cpu_s,
            "device_queue_wait_s": entry.device_queue_wait_s,
            "completion_queue_wait_s": entry.completion_queue_wait_s,
            "execution_wall_s": entry.execution_wall_s, "timing_mode": entry.timing_mode}
            for step_id, entry in entries.items()]
        trace_count = len(result.schedule_trace.transfers)
        trace_payload_bytes = sum(item.byte_count for item in result.schedule_trace.transfers)
        trace_metadata = (result.schedule_trace.transfers[:self.max_transfer_metadata]
            if self.compact_mode else result.schedule_trace.transfers)
        traces = {item.transfer_id: asdict(item) for item in trace_metadata}
        transfer_records = (() if self.compact_mode else
            result.recorded_transfers if isinstance(result, StreamingSimulationResult) else result.transfers)
        if not self.compact_mode and len(transfer_records) != len(traces):
            raise ValueError("interactive output requires all transfer execution records")
        transfers = [_transfer_manifest(transfer, traces) for transfer in transfer_records]
        transfer_event_count = 0
        for transfer in transfer_records:
            transfer_event_count += len(transfer.execution_timeline.intervals) + len(transfer.execution_timeline.host_intervals)
            trace = traces[transfer.workload.nodes[0].node_id]
            for call in transfer.kernel_plan.calls:
                link = kernel_link_usage(call.kernel_spec, self.hardware[call.kernel_spec.hardware_id])
                self.links.append({**link, "pool_name": call.pool_name, "rank": call.rank,
                    "call_id": call.call_id, "intervals": [[trace["start_time_s"], trace["end_time_s"]]],
                    "source_pool_id": trace["source_pool_id"], "destination_pool_id": trace["destination_pool_id"]})
            call, = transfer.kernel_plan.calls
            spec = call.kernel_spec.transfer
            path_id = self._critical_path_id(transfer.predictions[call.call_id])
            self._record_prediction_path(call.call_id, transfer.predictions[call.call_id], path_id)
            for pool, rank in ((transfer.source_engine_id, spec.source_rank),
                    (transfer.destination_engine_id, spec.destination_rank)):
                row = {"pool_name": pool, "rank": rank,
                    "step_id": trace["transfer_id"], "start_time_s": trace["start_time_s"],
                    "end_time_s": trace["end_time_s"],
                    "busy_intervals": [[trace["start_time_s"], trace["end_time_s"]]],
                    "compute": {}, "logical_bytes": 0.0, "hbm_capacity": None,
                    "available": 0, "unavailable": 1, "device_events": 1,
                    "l2_read_bytes": 0.0, "l2_read_capacity": None, "l2_available": 0, "l2_unavailable": 1,
                    "memory_traffic": {name: {**value, "unavailable": 1}
                        for name, value in _empty_memory_traffic().items()},
                    "detail_retained": True, "operation_counts": {"COMMUNICATION": 1},
                    "phases": ["kv_handoff"], "host_intervals": [],
                    "model_bottleneck_groups": {}}
                _add_model_bottleneck(row, transfer.predictions[call.call_id], 1, path_id)
                self.summaries.append(row)
        selected_ids = ([record["step_id"] for record in self._archive_records]
            if self.detail_archive_dir is not None else list(self._detail_members))
        detail_intervals = _union((origins[step_id], origins[step_id] + self._step_durations[step_id])
            for step_id in selected_ids)
        detail_intervals = _union([*detail_intervals,
            *[(traces[transfer.workload.nodes[0].node_id]["start_time_s"],
               traces[transfer.workload.nodes[0].node_id]["end_time_s"]) for transfer in transfer_records]])
        detailed_requests = (self._seen_request_ids if self.detail_archive_dir is not None else
            {request for value in self._detail_members.values() for request in value["requests"]})
        metadata_limit = self.max_transfer_metadata if self.compact_mode else len(traces)
        transfer_metadata = list(traces.values())[:metadata_limit]
        extra = {"visualization_transfer_metadata": transfer_metadata,
            "visualization_transfer_summary": {"total_records": trace_count, "retained_records": len(transfer_metadata),
                "omitted_records": trace_count - len(transfer_metadata),
                "total_payload_bytes": trace_payload_bytes,
                "scope": "schedule-trace payload bytes, not wire transactions or bandwidth utilization"}, "visualization_iterations": [{"step_id": step_id,
            "pool_name": entry.backend_step.schedule.pool_name,
            "device_start_time_s": origins[step_id],
            "device_end_time_s": origins[step_id] + self._step_durations[step_id],
            "event_count": self._step_event_counts[step_id],
            "phases": sorted({request.phase for request in entry.backend_step.schedule.requests}),
            "requests": [{"request_id": request.request_id, "phase": request.phase,
                "input_tokens": request.input_tokens, "output_tokens": request.output_tokens}
                for request in entry.backend_step.schedule.requests]}
            for step_id, entry in entries.items()]}
        if self.detail_archive_dir is not None:
            extra["visualization_detail_archive"] = {"directory": str(self.detail_archive_dir),
                "records": [{**record, "delta_s": origins[record["step_id"]] - self._recorded_origins[record["step_id"]]}
                    for record in self._archive_records]}
        return {**{key: value for key, value in payload.items() if key != "transfers"}, **extra, "execution_artifact": {
            "steps": self.detail_steps, "transfers": transfers,
            "schedule_trace": {"route_decisions": [asdict(item) for item in result.schedule_trace.route_decisions]}},
            "visualization_resources": self.resources,
            "visualization_diagnostics": {"critical_paths": self._critical_path_catalog,
                "by_call_id": self._prediction_path_ids,
                "missing_by_call_id": self._prediction_path_missing, "catalog_limit": 4096,
                "per_group_step_limit": 32},
            "visualization_links": self.links,
            "visualization_schedule": schedule,
            "visualization_hardware": {key: {"hbm_capacity_bytes": hw.resource_by_role("dram").size_bytes,
                "hbm_bandwidth_bytes_per_s": hw.resource_by_role("dram").capacity}
                for key, hw in self.hardware.items()},
            "visualization_summary": self.summaries,
            "visualization_coverage": {"detail_events": self.detail_events + transfer_event_count,
                "total_events": self.total_events + transfer_event_count,
                "transfer_events": transfer_event_count,
                "transfer_metadata_count": trace_count,
                "max_transfer_metadata": self.max_transfer_metadata if self.compact_mode else None,
                "transfer_metadata_retained": len(transfer_metadata),
                "transfer_metadata_omitted": trace_count - len(transfer_metadata),
                "transfer_event_scope": "retained operator/Host details; compact mode retains metadata only",
                "activity_scope": ("compact projection; activity intervals unavailable; logical device call counts only, Host event count unavailable" if self.compact_mode else "per rank; collective reservations appear on every participating rank; transfer details are retained outside the model-step budget"),
                "detail_interval_s": ([min(step["schedule_interval"]["start_time_s"] for step in self.detail_steps),
                    max(step["schedule_interval"]["end_time_s"] for step in self.detail_steps)] if self.detail_steps else None),
                "omitted_detail_events": self.total_events - self.detail_events,
                "aggregate_interval_s": [0.0, result.schedule_trace.end_time_s],
                "selection": ("compact summary; no operator details" if self.compact_mode else
                    "all complete forward iterations, independently compressed on disk"
                    if self.detail_archive_dir is not None else
                    "request coverage, phase coverage, endpoints, batch size and deterministic sampling within event budget"),
                "detail_intervals_s": detail_intervals,
                "detailed_requests": len(detailed_requests), "total_requests": len(self._seen_request_ids),
                "detailed_steps": len(selected_ids), "total_steps": len(self._recorded_origins),
                "max_detail_events": self.max_detail_events if self.detail_archive_dir is None else None,
                "resource_scope": ("all completed steps; per-step uniform demand; activity intervals unavailable in compact projection" if self.compact_mode else "all completed steps; per-step uniform demand inside intersected windows; exact busy interval union"),
                "activity_availability": "unavailable" if self.compact_mode else "available",
                "link_interval_availability": "unavailable" if self.compact_mode else "available",
                "transfer_detail_availability": "unavailable" if self.compact_mode else "available",
                "resource_recording_s": self.recording_wall_s}}


class CompactInteractiveRecording(InteractiveRecording):
    """Project resource totals without asking execution to materialize activity.

    The projector may run in a worker. Only consume_prehashed mutates the parent
    accumulator; the original digest and source-state snapshot pass through once.
    """

    def __init__(self, hardware_by_id, *, max_transfer_metadata=1024):
        if type(max_transfer_metadata) is not int or max_transfer_metadata < 1:
            raise ValueError("max_transfer_metadata must be a positive record budget")
        super().__init__(hardware_by_id)
        self.max_transfer_metadata = max_transfer_metadata
        self.compact_mode = True
        self.max_detail_events = 0

    def project_step(self, step):
        digest = self.execution_contract_digest(step)
        projection = self._timed_step_summary(step)
        return (digest, projection)

    def _timed_step_summary(self, step):
        started = time.perf_counter()
        with memory_prediction_source_scope():
            projection = self._project_step_summary(step)
        # Include both source checks in reporting cost. The existing artifact
        # digest belongs to the baseline path and remains outside this timer.
        projection["projection_wall_s"] = self._take_capture_cost() + time.perf_counter() - started
        return projection

    def _project_step_summary(self, step):
        timeline = step.execution_timeline
        counts = dict(timeline.call_occurrence_counts)
        if not counts:
            counts = {call.call_id: 1 for call in step.kernel_plan.calls}
        lanes = {}
        schedule = step.backend_step.schedule
        def lane_for(pool, rank):
            key = (pool, rank)
            if key not in lanes:
                lanes[key] = {"pool_name": pool, "rank": rank, "step_id": schedule.step_id,
                    "start_time_s": 0.0, "end_time_s": timeline.duration_s,
                    "busy_intervals": None, "host_intervals": None,
                    "activity_availability": "unavailable", "compute": {},
                    "logical_bytes": 0.0, "hbm_capacity": None, "available": 0, "unavailable": 0,
                    "l2_read_bytes": 0.0, "l2_read_capacity": None, "l2_available": 0, "l2_unavailable": 0,
                    "memory_traffic": _empty_memory_traffic(), "device_events": 0,
                    "detail_retained": False, "operation_counts": {}, "model_bottleneck_groups": {},
                    "phases": sorted({r.phase for r in schedule.requests})}
            return lanes[key]
        for call in step.kernel_plan.calls:
            lane = lane_for(call.pool_name, call.rank)
            n = counts[call.call_id]
            lane["device_events"] += n
            kind = call.kernel_spec.kind.value
            if kind not in lane["operation_counts"]:
                lane["operation_counts"][kind] = 0
            lane["operation_counts"][kind] += n
            _add_model_bottleneck(lane, step.predictions[call.call_id], n, None)
            usage = self._usage(call.kernel_spec)
            _add_l2_demand(lane, usage, n)
            _add_memory_traffic(lane, usage, n)
            if usage["availability"] == "available":
                lane["available"] += n
                lane["logical_bytes"] += n * usage["logical_bytes"]
                lane["hbm_capacity"] = usage["hbm_capacity_bytes_per_s"]
                for demand in usage["compute"]:
                    role = usage["dtype"] + " / " + demand["role"]
                    if role not in lane["compute"]:
                        lane["compute"][role] = {"operations": 0.0,
                            "capacity_operations_per_s": demand["capacity_operations_per_s"]}
                    lane["compute"][role]["operations"] += n * demand["operations"]
            elif usage["availability"] == "unavailable":
                lane["unavailable"] += n
            else:
                raise ValueError("unknown resource evidence availability")
            transfer = call.kernel_spec.transfer
            if transfer is not None and kind == "COMMUNICATION":
                for rank in transfer.participants:
                    if rank != call.rank:
                        participant = lane_for(call.pool_name, rank)
                        participant["unavailable"] += n
                        participant["device_events"] += n
                        participant["l2_unavailable"] += n
                        if "COMMUNICATION" not in participant["operation_counts"]:
                            participant["operation_counts"]["COMMUNICATION"] = 0
                        participant["operation_counts"]["COMMUNICATION"] += n
                        _add_model_bottleneck(participant, step.predictions[call.call_id], n, None)
                        for memory in participant["memory_traffic"].values():
                            memory["unavailable"] += n
        return {"summary": list(lanes.values()), "duration_s": timeline.duration_s,
            "event_count": sum(counts.values())}

    def consume_prehashed(self, backend_step, entry, execution_contract_digest, logical_kernel_call_count,
            compact, state_audit_snapshot=None):
        digest, projection = execution_contract_digest
        super().consume_prehashed(backend_step, entry, digest, logical_kernel_call_count,
            compact, state_audit_snapshot)
        started = time.perf_counter()
        step_id = backend_step.schedule.step_id
        origin = entry.start_time_s + entry.scheduler_plan_cpu_s + entry.device_queue_wait_s
        self._recorded_origins[step_id] = origin
        self._step_durations[step_id] = projection["duration_s"]
        self._step_event_counts[step_id] = projection["event_count"]
        self.total_events += projection["event_count"]
        self._seen_request_ids.update(r.request_id for r in backend_step.schedule.requests)
        for row in projection["summary"]:
            self.summaries.append({**row, "start_time_s": row["start_time_s"] + origin,
                "end_time_s": row["end_time_s"] + origin})
        self.recording_wall_s += projection["projection_wall_s"] + time.perf_counter() - started

    def consume(self, step, entry):
        if step.source_state_summary is not None:
            self._state_audit.consume(step)
        self.consume_prehashed(step.backend_step, entry, self.project_step(step),
            step.kernel_plan.logical_call_count, bool(step.kernel_plan.block_repeats))

    def record(self, step, entry):
        """Keep the original validation for callers without a full artifact."""
        self._require_prediction_coverage(step)
        _, projection = self.project_step(step)
        self._record_summary(step, entry, projection)

    @staticmethod
    def _require_prediction_coverage(step):
        if set(step.predictions) != {call.call_id for call in step.kernel_plan.calls}:
            raise ValueError("compact report predictions must cover KernelPlan exactly")

    def _record_after_artifact_validation(self, step, entry):
        """Only the ordinary finalizer calls this after its full manifest passes."""
        self._require_prediction_coverage(step)
        projection = self._timed_step_summary(step)
        self._record_summary(step, entry, projection)

    def _record_summary(self, step, entry, projection):
        origin = entry.start_time_s + entry.scheduler_plan_cpu_s + entry.device_queue_wait_s
        step_id = step.backend_step.schedule.step_id
        self._recorded_origins[step_id] = origin
        self._step_durations[step_id] = projection["duration_s"]
        self._step_event_counts[step_id] = projection["event_count"]
        self.total_events += projection["event_count"]
        self._seen_request_ids.update(r.request_id for r in step.backend_step.schedule.requests)
        self.summaries.extend({**row, "start_time_s": origin, "end_time_s": origin + projection["duration_s"]}
            for row in projection["summary"])
        self.recording_wall_s += projection["projection_wall_s"]
