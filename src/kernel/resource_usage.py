"""Read-only analytical demand evidence for optional execution recording.

Demand and compulsory bytes come from the predictor owner. No timing is
modified and unavailable source contracts remain explicitly unavailable.
"""
import json
import math
from pathlib import Path

from src.kernel.kernel_base import create_predictor


def kernel_tensor_work(kernel, hardware):
    """Useful Tensor math coverage, independently of native launch coverage.

    A covered zero is a non-Tensor operation, not missing launch evidence.
    Issued padded instructions remain separate from mathematical products.
    """
    from src.kernel.kernel_spec import KernelKind
    if kernel.hardware_id != hardware.hardware_id:
        raise ValueError("Tensor work hardware identity mismatch")
    if kernel.kind in (KernelKind.VECTOR, KernelKind.MEMCPY, KernelKind.COMMUNICATION):
        return {"availability": "available", "operations_by_role": {},
            "basis": "non-Tensor kernel-kind semantics; scalar/vector and transfer work excluded",
            "work_scope": "useful_mathematical_tensor_operations", "issued_operations_by_role": {}}
    if kernel.kind is KernelKind.GEMM:
        from src.kernel.predictors.gemm_block_fp8 import is_block_fp8_gemm, block_fp8_gemm_roofline_work
        if is_block_fp8_gemm(kernel):
            work = block_fp8_gemm_roofline_work(kernel)
            b, m, n, k = (kernel.shape[name] for name in ("b", "m", "n", "k"))
            tiles = kernel.launch.tiling.tile_shape
            issued = 2 * b * math.prod(-(-size // tiles[axis]) * tiles[axis]
                for size, axis in ((m, "m"), (n, "n"), (k, "k")))
            return {"availability": "available", "operations_by_role": {"compute_tensor_fp8": work["mathematical_flops"]},
                "issued_operations_by_role": {"compute_tensor_fp8": issued},
                "basis": "block_fp8_gemm_roofline_work plus source-selected fixed MMA tile extents",
                "work_scope": "useful_mathematical_tensor_operations; issued padding separate"}
        if ("mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] == "block_fp8_fp32_scales_v1"):
            from src.kernel.predictors.gemm_block_fp8 import block_fp8_linear_logical_work
            work = block_fp8_linear_logical_work(kernel)
            return {"availability":"available", "operations_by_role":{"compute_tensor_fp8":work["mathematical_flops"]},
                "issued_operations_by_role":None, "basis":"block_fp8_linear_logical_work source program",
                "work_scope":"matrix work includes declared model-input rows; native tile padding unbound"}
        if kernel.variant in ("gemm", "batched_gemm") and all(name in kernel.shape for name in ("b", "m", "n", "k")):
            if kernel.dtype not in ("bf16", "fp16"):
                return {"availability": "unavailable", "reason": "matrix dtype Tensor role not bound"}
            operations = 2 * math.prod(kernel.shape[name] for name in ("b", "m", "n", "k"))
            role = "compute_tensor" if kernel.dtype == "bf16" else "compute_tensor_fp16"
            hardware.resource_by_role(role)
            return {"availability": "available", "operations_by_role": {role: operations},
                "issued_operations_by_role": None, "basis": "declared complete matrix-product shape",
                "work_scope": "useful_mathematical_tensor_operations; backend padding unbound"}
    if kernel.kind is KernelKind.SPARSE_ATTN and kernel.variant in ("dsa_sparse_decode", "dsa_sparse_prefill", "dsa_sparse_mixed"):
        from src.kernel.predictors.sparse_attn import dsa_sparse_attention_logical_work
        work = dsa_sparse_attention_logical_work(kernel)
        operations = {"compute_tensor": work["tensor_flops"]}
        basis = "dsa_sparse_attention_logical_work selected causal pair cardinality"
    elif kernel.kind is KernelKind.SPARSE_ATTN_INDEXER and kernel.variant in ("dsa_indexer_decode", "dsa_indexer_prefill", "dsa_indexer_mixed"):
        from src.kernel.predictors.sparse_attn_indexer import dsa_indexer_logical_work
        work = dsa_indexer_logical_work(kernel)
        operations = {"compute_tensor_fp8": work["arithmetic"]["qk_dot_flops"]}
        basis = "DSA scores prepared_q/key FP8 operands; dsa_indexer_logical_work qk_dot_flops"
    elif (kernel.kind is KernelKind.FUSED_MOE and kernel.variant == "routed_experts"
            and "mathematical_program" in kernel.attributes
            and kernel.attributes["mathematical_program"] == "block_fp8_swiglu_v1"):
        from src.kernel.predictors.fused_moe import block_fp8_experts_logical_work
        work = block_fp8_experts_logical_work(kernel)
        operations = {"compute_tensor_fp8": sum(stage["mathematical_fp8_flops"] for stage in work["matrix_stages"])}
        basis = "block_fp8_experts_logical_work all routed matrix stages"
    else:
        return {"availability": "unavailable", "reason": "Tensor source work owner not bound for this variant"}
    for role in operations:
        hardware.resource_by_role(role)
    return {"availability": "available", "operations_by_role": operations,
        "issued_operations_by_role": None, "basis": basis,
        "work_scope": "useful_mathematical_tensor_operations; compiler padding unbound"}


def kernel_sm_activity(kernel, hardware, prediction, *, resource_input=None):
    """Integrate modeled per-SM resident task lifetimes, including waits.

    Pipeline workers must be the predictor's physical GPU execution units.
    This consumes scheduled task endpoints, never grid times kernel span.
    """
    from src.kernel.model_contract import PipelineWorkPlan, PipelineTaskProgramDemand
    if hardware.kernel_memory_model != "gpu_mshr":
        return {"availability": "unavailable", "reason": "SM activity requires GPU execution units"}
    if prediction.diagnostics is None or prediction.diagnostics.pipeline is None:
        return {"availability": "unavailable", "reason": "prediction has no scheduled resident-task diagnostics"}
    from src.kernel.predictors.gemm_block_fp8 import is_block_fp8_gemm
    from src.kernel.kernel_spec import KernelSpec
    source_persistent = isinstance(kernel, KernelSpec) and is_block_fp8_gemm(kernel)
    if source_persistent:
        pipeline = prediction.diagnostics.pipeline
        if (not all(0 <= task.worker < hardware.num_compute_units for task in pipeline.tasks)
                or "source_persistent_CTA_to_SM_round_robin_assignment" not in prediction.diagnostics.limitations):
            raise ValueError("FP8 source SM diagnostics lost their worker assignment contract")
        spans = {}
        for task in pipeline.tasks:
            start, end = min(task.stage_start_s.values()),max(task.stage_end_s.values())
            if not 0 <= start <= end <= prediction.t_total_s+1e-12:
                raise ValueError("FP8 source SM residency outside kernel")
            if task.worker not in spans:spans[task.worker]=[]
            spans[task.worker].append((start,end))
        intervals=[]
        for sm, values in spans.items():
            lane=[]
            for start,end in sorted(values):
                if lane and start<=lane[-1][1]:lane[-1][1]=max(end,lane[-1][1])
                else:lane.append([start,end])
            intervals.extend({"SM":sm,"start_s":start,"end_s":end} for start,end in lane)
        busy=sum(row["end_s"]-row["start_s"] for row in intervals)
        return {"availability":"available", "SM_conditionally_allocated_seconds":busy,"SM_count":hardware.num_compute_units,
            "conditional_phase_intervals":intervals,
            "SM_conditionally_allocated_phase_fraction":busy/(hardware.num_compute_units*prediction.t_total_s) if prediction.t_total_s else 0.,
            "SM_active_fraction":None, "activity_completeness":"conditional_phase_allocation",
            "basis":"source tile ownership with conditional per-SM allocation of existing price phase envelopes",
            "scope":"scale phase includes its launch floor; matrix phase includes TMA fill/shell. This conditional allocation is neither pure body busy nor complete SM-active",
            "limitations":list(prediction.diagnostics.limitations)}
    if resource_input is None:
        try:
            resource_input, _ = create_predictor(kernel.kind).resource_input(kernel, hardware)
        except NotImplementedError as error:
            return {"availability": "unavailable", "reason": str(error)}
    plan = resource_input.pipeline_plan
    if (not isinstance(plan, PipelineWorkPlan) or plan.worker_count != resource_input.active_units
            or not 0 < plan.worker_count <= hardware.num_compute_units):
        return {"availability": "unavailable", "reason": "pipeline workers not bound one-to-one to physical SMs"}
    physical_workers = {}
    program_workers = set()
    for task in plan.tasks:
        if isinstance(task, PipelineTaskProgramDemand):
            workers = task.physical_workers if task.physical_workers else (task.worker,)
            if len(workers) != task.worker_multiplicity:
                raise ValueError("source physical-worker multiplicity is incomplete")
            program_workers.add(task.worker)
        else:
            workers = (task.worker,)
        if task.worker in physical_workers and physical_workers[task.worker] != workers:
            raise ValueError("source worker has conflicting physical SM classes")
        physical_workers[task.worker] = workers
    pipeline = prediction.diagnostics.pipeline
    if not pipeline.tasks:
        return {"availability":"unavailable", "reason":"source pipeline omitted resident task endpoints"}
    spans = {}
    for task in pipeline.tasks:
        if not 0 <= task.worker < plan.worker_count or not task.stage_start_s or not task.stage_end_s:
            raise ValueError("invalid scheduled SM task identity or endpoints")
        start, end = min(task.stage_start_s.values()), max(task.stage_end_s.values())
        if not 0 <= start <= end <= prediction.t_total_s + 1e-12:
            raise ValueError("resident task lifetime lies outside its predicted kernel")
        if task.worker not in physical_workers:
            raise ValueError("scheduled source worker has no physical assignment")
        for sm in physical_workers[task.worker]:
            if not 0 <= sm < hardware.num_compute_units:
                raise ValueError("source physical worker lies outside SM hardware")
            if sm not in spans:
                spans[sm] = []
            spans[sm].append((start, end))
    # A compressed worker program retains its nonpreemptive source worker
    # through the queued pattern, including operand waits. Endpoint capture
    # supplies first acquire / last release without multiplying one interval
    # by the repeat count. This is the existing static worker-queue model;
    # it does not claim native CTA placement or issue-pipe activity.
    for worker in program_workers:
        for sm in physical_workers[worker]:
            if sm in spans:
                spans[sm] = [(min(start for start, _ in spans[sm]), max(end for _, end in spans[sm]))]
    merged = []
    for sm, intervals in sorted(spans.items()):
        lane = []
        for start, end in sorted(intervals):
            if lane and start <= lane[-1][1]:
                lane[-1][1] = max(end, lane[-1][1])
            else:
                lane.append([start, end])
        merged.extend({"SM": sm, "start_s": start, "end_s": end} for start, end in lane)
    sm_seconds = sum(row["end_s"] - row["start_s"] for row in merged)
    return {"availability": "available", "SM_busy_seconds": sm_seconds,
        "SM_count": hardware.num_compute_units, "resident_intervals": merged,
        "SM_body_busy_fraction": (sm_seconds / (hardware.num_compute_units * prediction.t_total_s)
            if prediction.t_total_s else 0.),
        "SM_active_fraction":None, "activity_completeness":"body_only",
        "limitations":["source resident task intervals only; complete outer-kernel coverage not proved"],
        "basis": "source predictor PipelineWorkPlan physical-worker binding and scheduled task acquire-to-release endpoints; union only within each SM",
        "scope": "analytical resident-task busy including memory waits; compressed programs retain their static worker lease; not measured native counters, issue activity or grid coverage"}


def report_memory_bandwidth_reference(hardware_id):
    """Independent peak evidence for display only; never changes prediction."""
    if hardware_id != "nvidia_h200":
        return None
    path = Path(__file__).resolve().parents[2] / "microbench/nvidia_h200/references/l2_peak_metadata_sm90.json"
    reference = json.loads(path.read_text())
    if reference["version"] != 1 or reference["hardware_id"] != hardware_id:
        raise ValueError("invalid report memory bandwidth reference identity")
    clock = reference["clock_hz"]
    l2 = reference["l2"]
    if not all(math.isfinite(value) and value > 0 for value in (
            clock, l2["read_bytes_per_cycle"], l2["write_bytes_per_cycle"])):
        raise ValueError("invalid report memory bandwidth reference peak")
    source = l2["source"]
    if (l2["read_bytes_per_cycle"] != source["read_sectors_per_cycle"] * source["sector_bytes"]
            or l2["write_bytes_per_cycle"] != source["write_sectors_per_cycle"] * source["sector_bytes"]
            or l2["read_bytes_per_cycle"] != source["xbar_peak"] * source["sector_bytes"]):
        raise ValueError("report memory bandwidth peak disagrees with NCU metadata")
    hbm = reference["hbm"]
    if not math.isfinite(hbm["shared_bytes_per_second"]) or hbm["shared_bytes_per_second"] <= 0:
        raise ValueError("invalid HBM shared bandwidth peak")
    return {"l2": {"read_bytes_per_second": l2["read_bytes_per_cycle"] * clock,
        "write_bytes_per_second": l2["write_bytes_per_cycle"] * clock,
        "combined": "max_directional", "basis": l2["basis"], "source": l2["source"]},
        "hbm": {"read_bytes_per_second": hbm["shared_bytes_per_second"],
        "write_bytes_per_second": hbm["shared_bytes_per_second"],
        "combined": "shared_sum", "basis": hbm["basis"], "source": hbm["source"]}}


def kernel_sm_coverage_bound(kernel, hardware) -> dict:
    """Source-grid instantaneous SM coverage bound, never SM-active duty.

    A grid with B CTAs cannot occupy more than min(B, SMs) SMs at once.
    Wave tails, placement, compiler residency and per-SM active times are
    not provided by this static bound. No predicted duration enters it.
    """
    if hardware.hardware_id not in ("nvidia_h200", "nvidia_a800"):
        return {"availability": "unavailable", "reason": "CUDA SM grid bound requires a supported NVIDIA hardware source"}
    if kernel.launch is None:
        return {"availability": "unavailable", "reason": "no native launch grid; mathematical work does not define SM coverage"}
    grid = kernel.launch.grid
    sms = hardware.num_compute_units
    if (type(grid) is not tuple or len(grid) != 3
            or any(type(value) is not int or value <= 0 for value in grid)
            or type(sms) is not int or sms <= 0):
        raise ValueError("SM grid coverage requires positive source grid and SM count")
    blocks = math.prod(grid)
    return {"availability": "geometry_bound_only", "source_grid": list(grid),
        "CTA_count": blocks, "SM_count": sms,
        "maximum_covered_SM_fraction": min(blocks, sms)/sms,
        "SM_active_fraction": None,
        "basis": "source launch grid and HWSpec SM count; instantaneous upper bound, not measured or modeled SM-active duty"}


def kernel_resource_usage(kernel, hardware) -> dict:
    """Export modeled work/traffic and configured HWSpec bandwidths, with units."""
    from src.kernel.kernel_spec import KernelSpec
    from src.kernel.predictors.gemm_block_fp8 import is_block_fp8_gemm, block_fp8_source_traffic, block_fp8_gemm_roofline_work
    if isinstance(kernel, KernelSpec) and is_block_fp8_gemm(kernel):
        source = block_fp8_source_traffic(kernel, hardware)
        tensor = kernel_tensor_work(kernel, hardware)
        traffic = {}
        for name, role in (("l2", "l2_cache"), ("hbm", "dram")):
            traffic[name] = {"availability":"available", "read_bytes":source[name+"_read_bytes"],
                "write_bytes":source[name+"_write_bytes"],
                "capacity_bytes_per_s":hardware.resource_by_role(role).capacity,
                "basis":source["basis"], "source":"block_fp8_source_traffic shared LRU replay",
                "limitations":list(source["limitations"])}
        return {"availability":"available", "dtype":kernel.dtype, "compute":[{"role":role, "operations":amount,
                "capacity_operations_per_s":hardware.resource_by_role(role).capacity}
                for role, amount in tensor["operations_by_role"].items()],
            "tensor_work":tensor, "memory_traffic":traffic,
            "logical_bytes":block_fp8_gemm_roofline_work(kernel)["logical_bytes"],
            "hbm_capacity_bytes_per_s":hardware.resource_by_role("dram").capacity,
            "l2_traffic":{"availability":"available", "read_bytes":source["l2_read_bytes"],
                "write_bytes":source["l2_write_bytes"], "read_capacity_bytes_per_s":hardware.resource_by_role("l2_cache").capacity,
                "basis":source["basis"], "source":"block_fp8_source_traffic"},
            "deferred_dirty_writeback_bytes":source["deferred_dirty_writeback_bytes"],
            "source":source["basis"], "limitations":list(source["limitations"])}
    if kernel.launch is None:
        return {"availability": "unavailable", "reason": "resource-input export requires native launch metadata; this operation has only mathematical execution support"}
    predictor = create_predictor(kernel.kind)
    try:
        inputs, compulsory_bytes = predictor.resource_input(kernel, hardware)
    except NotImplementedError as error:
        return {"availability": "unavailable", "reason": str(error)}
    return _project_resource_usage(kernel, hardware, inputs, compulsory_bytes)


def _project_resource_usage(kernel, hardware, inputs, compulsory_bytes) -> dict:
    """Project an already constructed public input using the same evidence rules."""
    compute = []
    for role, amount in inputs.demand_per_resource.items():
        if role.startswith("compute_") and amount > 0:
            compute.append({"role": role, "operations": float(amount),
                "capacity_operations_per_s": hardware.resource_by_role(role).capacity})
    # These aggregate nodes are produced by the predictor's memory accounting,
    # including its cache/LRU replay where that path uses replay. Never use
    # compulsory logical bytes or resource durations as replacement traffic.
    memory_traffic = {}
    directional_pair = {path.access_kind for path in inputs.memory_paths} == {"read", "write"}
    for name, role in (("l2", "l2_cache"), ("hbm", "dram")):
        directional = {}
        for direction in ("read", "write"):
            nodes = [node for path in inputs.memory_paths if path.access_kind == direction
                for node in path.nodes if node.resource_role == role]
            # MemoryNodeDemand requires strictly positive bytes. The owner
            # omits a downstream node when the exported read/write ledger has
            # zero traffic there (e.g. an LRU hit or retained dirty output).
            # This is a proven zero only with both directional paths present.
            directional[direction + "_bytes"] = sum(node.bytes for node in nodes) if nodes or directional_pair else None
        complete = all(value is not None for value in directional.values())
        memory_traffic[name] = {**directional,
            "availability": "available" if complete else "unavailable",
            "capacity_bytes_per_s": hardware.resource_by_role(role).capacity if complete else None,
            "basis": "predictor aggregate memory-path bytes; same HWSpec configured bandwidth denominator for read, write and total",
            "source": "PredictionInput.memory_paths; MemoryNodeDemand.bytes; role=" + role,
            "limitations": ([] if complete else ["read_or_write_path_not_exported; unknown is not zero"])}
    read = memory_traffic["l2"]["read_bytes"]
    l2_traffic = ({"availability": "available", "read_bytes": read,
        "write_bytes": memory_traffic["l2"]["write_bytes"],
        "read_capacity_bytes_per_s": hardware.resource_by_role("l2_cache").capacity,
        "basis": "modeled aggregate read-path bytes / configured l2_cache bandwidth",
        "source": "PredictionInput.memory_paths; MemoryNodeDemand.bytes; role=l2_cache"}
        if read is not None else {"availability": "unavailable",
            "reason": "predictor did not export an aggregate L2 read path"})
    return {"availability": "available", "compute": compute, "l2_traffic": l2_traffic,
        "memory_traffic": memory_traffic,
        "dtype": kernel.dtype, "logical_bytes": float(compulsory_bytes),
        "hbm_capacity_bytes_per_s": hardware.resource_by_role("dram").capacity,
        "traffic_basis": "compulsory logical work reference only; use memory_traffic for modeled L2/HBM bandwidth",
        "source": "KernelPredictor.resource_input; HWSpec configured resource capacity",
        "limitations": list(inputs.limitations)}


def kernel_link_usage(kernel, hardware) -> dict:
    """Expose algorithmic per-endpoint wire demand from transfer semantics.

    Ring all-reduce sends and receives 2*(P-1)/P payload bytes per rank;
    all-gather's payload is the input shard. These are algorithmic bytes,
    excluding packet headers, retransmits and routing overhead, not observed
    network transactions. No duration enters this accounting.
    """
    from src.kernel.kernel_spec import KernelKind
    if kernel.kind is not KernelKind.COMMUNICATION:
        return {"availability": "not_applicable"}
    transfer = kernel.transfer
    payload = transfer.bytes
    ranks = len(transfer.participants)
    if transfer.operation == "send_recv":
        wire_bytes = float(payload)
    elif transfer.operation == "all_reduce" and transfer.algorithm == "ring":
        wire_bytes = 2.0 * (ranks - 1) / ranks * payload
    elif transfer.operation == "all_gather" and transfer.algorithm == "ring":
        wire_bytes = float((ranks - 1) * payload)
    elif transfer.operation == "all_to_all":
        wire_bytes = (ranks - 1) / ranks * payload
    else:
        return {"availability": "unavailable", "link": transfer.link,
            "reason": "per-endpoint wire accounting unavailable for this collective algorithm"}
    return {"availability": "available", "link": transfer.link,
        "resource_id": transfer.resource_id,
        "participants": list(transfer.participants),
        "payload_bytes": payload, "wire_bytes_per_endpoint": wire_bytes,
        "capacity_bytes_per_s": hardware.analytical_link_bandwidth_bytes_s(transfer.link),
        "basis": "algorithmic bytes per participating endpoint; excludes packet overhead; configured analytical link bandwidth"}
