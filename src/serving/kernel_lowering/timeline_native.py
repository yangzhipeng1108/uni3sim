import ctypes
import functools
import math
from collections import OrderedDict
import os
from pathlib import Path
import subprocess
import tempfile
import threading

from .execution_timeline import _ExperimentCache


class _NativePlan(ctypes.Structure):
    _fields_ = (
        ("occurrence_count", ctypes.c_int32),
        ("call_count", ctypes.c_int32),
        ("resource_count", ctypes.c_int32),
        ("digest_count", ctypes.c_int32),
        ("repeat_count", ctypes.c_int32),
        ("call_index", ctypes.POINTER(ctypes.c_int32)),
        ("dependency_offsets", ctypes.POINTER(ctypes.c_int32)),
        ("dependencies", ctypes.POINTER(ctypes.c_int32)),
        ("resource_offsets", ctypes.POINTER(ctypes.c_int32)),
        ("resources", ctypes.POINTER(ctypes.c_int32)),
        ("repeat_slot", ctypes.POINTER(ctypes.c_int32)),
        ("repeat_first", ctypes.POINTER(ctypes.c_int32)),
        ("repeat_last", ctypes.POINTER(ctypes.c_int32)),
        ("digest_slot", ctypes.POINTER(ctypes.c_int32)),
    )


class _NativeHostProgram(ctypes.Structure):
    _fields_ = (
        ("lane_count", ctypes.c_int32),
        ("identity_count", ctypes.c_int32),
        ("operation_count", ctypes.c_int32),
        ("offsets", ctypes.POINTER(ctypes.c_int32)),
        ("lanes", ctypes.POINTER(ctypes.c_int32)),
        ("kinds", ctypes.POINTER(ctypes.c_int32)),
        ("identities", ctypes.POINTER(ctypes.c_int32)),
        ("sync_before_replay", ctypes.POINTER(ctypes.c_int32)),
        ("extra_enqueues", ctypes.POINTER(ctypes.c_int32)),
        ("services", ctypes.POINTER(ctypes.c_double)),
    )


@functools.lru_cache(maxsize=1)
def _library():
    source = Path(__file__).with_name("timeline_native.c")
    with tempfile.TemporaryDirectory(prefix="midlayer-timeline-native-") as directory:
        output = Path(directory) / "timeline_native.so"
        compiler = os.environ["CC"] if "CC" in os.environ else "cc"
        result = subprocess.run((compiler, "-O3", "-std=c99", "-shared", "-fPIC",
            str(source), "-o", str(output)), capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("native summary timeline build failed: " + result.stderr)
        library = ctypes.CDLL(str(output))
    function = library.midlayer_summary_timeline
    function.argtypes = (ctypes.POINTER(_NativePlan), ctypes.POINTER(ctypes.c_double),
        ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(_NativeHostProgram),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_size_t)))
    function.restype = ctypes.c_int
    release = library.midlayer_summary_timeline_free
    release.argtypes = (ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_size_t))
    release.restype = None
    order = library.midlayer_earliest_ready_order
    order.argtypes = (ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        *(ctypes.POINTER(ctypes.c_int32),) * 7, ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int32))
    order.restype = ctypes.c_int
    function.resource_order = order
    return function, release


def native_resource_predecessors(calls, resources, predictions):
    """Bind the current expanded DAG to the existing native timeline helper."""
    positions = {call.call_id: index for index, call in enumerate(calls)}
    resource_ids, groups, group_rows, call_groups = {}, {}, [], []
    for row in resources:
        if row not in groups:
            groups[row] = len(group_rows)
            group_rows.append(row)
        call_groups.append(groups[row])
        for resource in row:
            if resource not in resource_ids:
                resource_ids[resource] = len(resource_ids)

    def rows_array(rows):
        offsets, flat = [0], []
        for row in rows:
            flat.extend(row)
            offsets.append(len(flat))
        return offsets, flat

    dep_offsets, deps = rows_array(tuple(tuple(positions[parent] for parent in dict.fromkeys(call.dependencies))
        for call in calls))
    res_offsets, res = rows_array(tuple(tuple(resource_ids[name] for name in row) for row in resources))
    group_offsets, group_res = rows_array(tuple(tuple(resource_ids[name] for name in row) for row in group_rows))
    values = (dep_offsets, deps, res_offsets, res, call_groups, group_offsets, group_res)
    if any(len(row) >= 2**31 for row in values) or max(len(calls), len(resource_ids), len(groups)) >= 2**31:
        raise ValueError("native resource order exceeds int32 index space")
    arrays = tuple((ctypes.c_int32 * max(1, len(row)))(*row) for row in values)
    durations = (ctypes.c_double * len(calls))(*(predictions[call.call_id].t_total_s for call in calls))
    output = (ctypes.c_int32 * len(res))()
    status = _library()[0].resource_order(len(calls), len(resource_ids), len(groups), *arrays, durations, output)
    if status:
        raise ValueError(f"native earliest-ready resource order failed: {status}")
    return tuple(tuple(sorted({output[index] for index in range(res_offsets[call], res_offsets[call + 1])
        if output[index] >= 0})) for call in range(len(calls)))


class _NativePlanInput:
    def __init__(self, schedule_plan, repeats):
        if any(type(value) is not int or value <= 0 or value > 2**31 - 1
                for value in (schedule_plan.occurrence_count, schedule_plan.resource_count,
                    len(repeats) if repeats else 1)):
            raise ValueError("native summary schedule dimensions must be positive int32 values")
        if any(type(value) is not int or value < 0 or value > 2**31 - 1
                for value in (schedule_plan.digest_count, len(repeats))):
            raise ValueError("native summary digest dimensions must be nonnegative int32 values")
        owned = []

        def array(values):
            if (not isinstance(values, tuple)
                    or any(type(value) is not int or not -2**31 <= value < 2**31 for value in values)):
                raise TypeError("native summary indices must be exact int32 tuples")
            result = (ctypes.c_int32 * max(1, len(values)))(*values)
            owned.append(result)
            return result

        def nested(rows):
            flattened = []
            offsets = [0]
            for row in rows:
                if not isinstance(row, tuple):
                    raise TypeError("native summary index groups must be tuples")
                flattened.extend(row)
                offsets.append(len(flattened))
            return array(tuple(offsets)), array(tuple(flattened))

        dependency_offsets, dependencies = nested(schedule_plan.dependency_slots)
        resource_offsets, resources = nested(schedule_plan.resource_slots)
        self.owned = owned
        self.structure = _NativePlan(
            schedule_plan.occurrence_count, max(schedule_plan.call_index) + 1,
            schedule_plan.resource_count, schedule_plan.digest_count, len(repeats),
            array(schedule_plan.call_index), dependency_offsets, dependencies,
            resource_offsets, resources, array(schedule_plan.repeat_slot),
            array(tuple(int(value) for value in schedule_plan.repeat_first)),
            array(tuple(int(value) for value in schedule_plan.repeat_last)),
            array(schedule_plan.digest_slot))


class _NativeHostInput:
    def __init__(self, schedule_plan, host_lane_program, hardware_by_call_index,
            eager_service_s_by_hardware, graph_replay_service_s_by_hardware):
        from .execution_timeline import _HostPrepareServices
        from src.serving.host_timing import RuntimeGraphSegmentDescriptor
        owned = []

        def integers(values):
            result = (ctypes.c_int32 * max(1, len(values)))(*values)
            owned.append(result)
            return result

        lane_indices = {}
        identity_indices = {}
        offsets = [0]
        lanes, kinds, identities, sync, extra, services = [], [], [], [], [], []
        for slot, logical_key in enumerate(schedule_plan.logical_keys):
            hardware_id = hardware_by_call_index[schedule_plan.call_index[slot]]
            eager_service = eager_service_s_by_hardware[hardware_id]
            graph_service = graph_replay_service_s_by_hardware[hardware_id]
            for lane, segment, service_times in host_lane_program[logical_key]:
                if (type(lane) is not tuple or len(lane) != 2 or type(lane[0]) is not str
                        or type(lane[1]) is not int):
                    return None
                if lane not in lane_indices:
                    lane_indices[lane] = len(lane_indices)
                lane_index = lane_indices[lane]
                if service_times is not None:
                    if type(service_times) is not _HostPrepareServices:
                        return None
                    entries = tuple(service_times.service_entries(eager_service))
                    if not entries:
                        entries = ((0.0, 0),)
                    for service_s, extra_enqueues in entries:
                        if (type(service_s) is not float or not math.isfinite(service_s)
                                or type(extra_enqueues) is not int):
                            return None
                        lanes.append(lane_index)
                        kinds.append(0)
                        identities.append(-1)
                        sync.append(0)
                        extra.append(extra_enqueues)
                        services.append(service_s)
                if segment is None:
                    service_s = 0.0 if service_times is not None and service_times.replaces_eager_enqueue else eager_service
                    lanes.append(lane_index)
                    kinds.append(1)
                    identities.append(-1)
                    sync.append(0)
                    extra.append(1)
                    services.append(service_s)
                    continue
                if type(segment) is not RuntimeGraphSegmentDescriptor:
                    return None
                identity = (lane[0], lane[1], segment.segment_id)
                if identity not in identity_indices:
                    identity_indices[identity] = len(identity_indices)
                lanes.append(lane_index)
                kinds.append(2)
                identities.append(identity_indices[identity])
                sync.append(int(segment.sync_before_replay))
                extra.append(0)
                services.append(graph_service)
            offsets.append(len(lanes))
        if not lanes or any(type(value) is not float or not math.isfinite(value) or value < 0 for value in services):
            return None
        self.owned = owned
        self.structure = _NativeHostProgram(len(lane_indices), len(identity_indices), len(lanes),
            integers(tuple(offsets)), integers(tuple(lanes)), integers(tuple(kinds)),
            integers(tuple(identities)), integers(tuple(sync)), integers(tuple(extra)),
            (ctypes.c_double * len(services))(*services))
        owned.append(self.structure.services)


_HOST_CACHE = _ExperimentCache()


def _cached_host_program(schedule_plan, host_lane_program, hardware_by_call_index,
        eager_service_s_by_hardware, graph_replay_service_s_by_hardware):
    # The caller rebuilds its service dictionaries for every step. Bind their
    # exact values, including hardware order, rather than container identity.
    # Mutable caller dictionaries are read again on every lookup.
    hardware = tuple(hardware_by_call_index)
    services = tuple((hardware_id, eager_service_s_by_hardware[hardware_id],
        graph_replay_service_s_by_hardware[hardware_id])
        for hardware_id in sorted(set(hardware)))
    if any(type(value) is not float or not math.isfinite(value) or value < 0.0
            for _, eager, graph in services for value in (eager, graph)):
        # Do not let bool/int equality with cached floats bypass the native
        # input checks. Preserve the constructor's behavior for unused fields.
        return _NativeHostInput(schedule_plan, host_lane_program, hardware_by_call_index,
            eager_service_s_by_hardware, graph_replay_service_s_by_hardware)
    key = (id(schedule_plan), id(host_lane_program), hardware, services)
    with _CACHE_LOCK:
        if key in _HOST_CACHE:
            retained, result = _HOST_CACHE[key]
            if retained[0] is not schedule_plan or retained[1] is not host_lane_program:
                raise RuntimeError("native Host program identity collision")
            _HOST_CACHE.move_to_end(key)
            return result
        result = _NativeHostInput(schedule_plan, host_lane_program, hardware_by_call_index,
            eager_service_s_by_hardware, graph_replay_service_s_by_hardware)
        _HOST_CACHE[key] = ((schedule_plan, host_lane_program), result)
        if len(_HOST_CACHE) > _CACHE_LIMIT:
            _HOST_CACHE.popitem(last=False)
        return result


_PLAN_CACHE = _ExperimentCache()
_ARRAY_CACHE = _ExperimentCache()
_CACHE_LOCK = threading.Lock()
_CACHE_LIMIT = 16


def _cached_plan(schedule_plan, repeats):
    identity = id(schedule_plan)
    with _CACHE_LOCK:
        if identity in _PLAN_CACHE:
            retained, result = _PLAN_CACHE[identity]
            if retained is not schedule_plan:
                raise RuntimeError("native summary schedule identity collision")
            _PLAN_CACHE.move_to_end(identity)
            return result
        result = _NativePlanInput(schedule_plan, repeats)
        _PLAN_CACHE[identity] = schedule_plan, result
        if len(_PLAN_CACHE) > _CACHE_LIMIT:
            _PLAN_CACHE.popitem(last=False)
        return result


def _float_array(values, expected_type):
    identity = id(values)
    with _CACHE_LOCK:
        if identity in _ARRAY_CACHE:
            retained, result = _ARRAY_CACHE[identity]
            if retained is not values:
                raise RuntimeError("native summary input identity collision")
            _ARRAY_CACHE.move_to_end(identity)
            return result
        if (not isinstance(values, expected_type)
                or any(type(value) is not float or not math.isfinite(value) for value in values)):
            raise TypeError("native summary timing inputs must be immutable finite float sequences")
        result = (ctypes.c_double * max(1, len(values)))(*values)
        _ARRAY_CACHE[identity] = values, result
        if len(_ARRAY_CACHE) > _CACHE_LIMIT:
            _ARRAY_CACHE.popitem(last=False)
        return result


def native_summary_recurrence(plan, predictions, calls, origin_time_s,
        host_launch_profiles, static_host_timeline, launch_gap_by_slot,
        shared_resource_calendar, schedule_plan, host_lane_program,
        hardware_by_call_index, eager_service_s_by_hardware,
        graph_replay_service_s_by_hardware):
    host_timing = host_launch_profiles is not None
    if (shared_resource_calendar is not None or not schedule_plan.occurrence_count
            or not host_timing and static_host_timeline is not None
            or type(schedule_plan.occurrence_count) is not int
            or type(schedule_plan.resource_count) is not int
            or type(schedule_plan.digest_count) is not int):
        return None
    durations = [predictions[call.call_id].t_total_s for call in calls]
    if any(type(value) is not float or not math.isfinite(value) or value < 0 for value in durations):
        return None
    plan_input = _cached_plan(schedule_plan, plan.block_repeats)
    native_durations = (ctypes.c_double * len(durations))(*durations)
    dynamic_host = None
    host_ready = None
    if host_timing and static_host_timeline is not None:
        host_ready = _float_array(static_host_timeline.ready_time_s, tuple)
        if len(host_ready) != schedule_plan.occurrence_count:
            raise ValueError("native summary static Host inputs must cover every occurrence")
    elif host_timing:
        if host_lane_program is None:
            return None
        dynamic_host = _cached_host_program(schedule_plan, host_lane_program,
            hardware_by_call_index, eager_service_s_by_hardware,
            graph_replay_service_s_by_hardware)
        if dynamic_host is None:
            return None
    launch_gaps = _float_array(launch_gap_by_slot, tuple) if host_timing else None
    if host_timing and len(launch_gaps) != schedule_plan.occurrence_count:
        raise ValueError("native summary Host launch gaps must cover every occurrence")
    maximum_end = ctypes.c_double()
    maximum_device_end = ctypes.c_double()
    launch_cpu_time_s = ctypes.c_double()
    launch_critical_path_s = ctypes.c_double()
    eager_dispatch_count = ctypes.c_int32()
    graph_replay_count = ctypes.c_int32()
    launch_kind_flags = ctypes.c_int32()
    bounds = (ctypes.c_double * max(1, 2 * len(plan.block_repeats)))()
    records = ctypes.POINTER(ctypes.c_char)()
    counts = ctypes.POINTER(ctypes.c_size_t)()
    function, release = _library()
    status = function(ctypes.byref(plan_input.structure), native_durations,
        origin_time_s, int(host_timing), host_ready, launch_gaps,
        None if dynamic_host is None else ctypes.byref(dynamic_host.structure),
        ctypes.byref(maximum_end), ctypes.byref(maximum_device_end), bounds,
        ctypes.byref(launch_cpu_time_s), ctypes.byref(launch_critical_path_s),
        ctypes.byref(eager_dispatch_count), ctypes.byref(graph_replay_count),
        ctypes.byref(launch_kind_flags), ctypes.byref(records), ctypes.byref(counts))
    if status:
        raise RuntimeError(f"native summary timeline rejected its validated input: {status}")
    try:
        repeat_bounds = tuple((bounds[2 * index], bounds[2 * index + 1])
            for index in range(len(plan.block_repeats)))
        stride = schedule_plan.occurrence_count * 16
        base = ctypes.addressof(records.contents)
        interval_records = tuple(ctypes.string_at(
                base + index * stride, counts[index] * 16)
            for index in range(schedule_plan.digest_count))
    finally:
        release(records, counts)
    duration_s = maximum_end.value - origin_time_s
    predicted_device_time_s = (maximum_device_end.value - origin_time_s
        if host_timing else duration_s)
    if static_host_timeline is not None:
        launch_cpu = static_host_timeline.launch_cpu_time_s
        launch_critical = static_host_timeline.launch_critical_path_s
        eager = static_host_timeline.eager_dispatch_count
        graphs = static_host_timeline.graph_replay_count
        kinds = static_host_timeline.launch_kinds
    else:
        launch_cpu = launch_cpu_time_s.value
        launch_critical = launch_critical_path_s.value
        eager = eager_dispatch_count.value
        graphs = graph_replay_count.value
        kinds = tuple(name for bit, name in ((1, "runtime_prepare"), (2, "eager"),
            (4, "graph_replay")) if launch_kind_flags.value & bit)
    return (duration_s, predicted_device_time_s, repeat_bounds, interval_records,
        launch_cpu, launch_critical, eager, graphs, kinds)
