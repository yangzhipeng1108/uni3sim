"""Attention operands projected from actual forward allocations and views.

Logical cache selections preserve their real arena without inventing page IDs,
slot contents, or library workspace. Native ABIs lacking source ports stay partial.
"""
from dataclasses import fields, replace
from math import prod

from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import EventKind, EventRef, LogicalSelectionView, ProgramEvent
from src.kernel.memory_access_builder import (
    CacheUpdateBindings, ElementwiseBindings, has_public_attention_source,
)
from src.kernel.memory_cache_update import paged_kv_source_index_type
from src.serving.workload.workload_graph import WorkloadKind
from .memory_bindings import source_memory_events


def fa3_gqa_caller_facts(attributes,hardware):
    """Normalize complete optional graph-caller facts at the source boundary."""
    fields=("fa3_gqa_graph_configured_mode","fa3_gqa_max_capture_tokens","fa3_gqa_query_tokens")
    present=tuple(field in attributes for field in fields)
    if any(present) and not all(present):
        raise ValueError("FA3 caller graph facts must be complete")
    graph=({"graph_configured_mode":attributes[fields[0]],"max_capture_tokens":attributes[fields[1]],
        "query_tokens":attributes[fields[2]]} if all(present) else {})
    facts = hardware.fa3_gqa_model_call_facts(attributes["source_model_execution_version"],
        page_size=None if attributes["fa3_gqa_page_origin"]=="vllm030_CacheConfig_default" else attributes["fa3_gqa_page_size"],
        **graph)
    if "source_program" in attributes and attributes["source_program"] == "vllm_0_30_fa3_gqa_full_decode":
        if not all(present) or attributes[fields[0]] not in ("FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"):
            raise ValueError("FA3 FULL caller requires its exact capture configuration")
        facts["fa3_gqa_caller_scenario"] = "vllm030_full_decode_zero_request_tail_default_aot"
    return facts


class _FA3SourceCallContext:
    """One plan's immutable call references and source-readiness proofs."""

    def __init__(self, calls):
        self.calls = tuple(calls)
        self.by_id = {call.call_id: call for call in self.calls}
        if len(self.by_id) != len(self.calls):
            raise ValueError("FA3 source frame contains duplicate call identities")
        self.completions = {call.source_node_ids[0]: call.call_id for call in self.calls
            if len(call.source_node_ids) == 1}
        self._entries = None
        self.validated_writers = {}

    def attention_entries(self):
        if self._entries is None:
            entries = {}
            for call in self.calls:
                if (call.kernel_spec.kind is not KernelKind.FLASH_ATTN or call.memory_source is None
                        or "fa3_full_aot_layout" not in call.memory_source.attributes
                        or any(name in self.by_id and self.by_id[name].memory_source is call.memory_source
                            for name in call.dependencies)):
                    continue
                identity = call.memory_source.node_id
                if identity in entries:
                    raise ValueError("FA3 AOT source has multiple attention entry calls")
                entries[identity] = call
            self._entries = entries
        return self._entries


def fa3_plan_call_groups(calls, *, source_context=None):
    """Retain explicitly source-bound library groups through deferred pricing."""
    groups = {}
    for call in calls:
        source = call.memory_source
        if source is not None and "fa3_full_aot_component" in source.attributes:
            continue
        if source is None or not ("fa3_source_revision" in source.attributes
                or "fa3_gqa_caller_scenario" in source.attributes or "fa3_expanded_source_revision" in source.attributes):
            continue
        key = (source.pool_name, source.rank, source.node_id)
        if key not in groups:
            groups[key] = []
        groups[key].append(call)
    context = _FA3SourceCallContext(calls) if source_context is None else source_context
    if (type(context) is not _FA3SourceCallContext
            or source_context is not None and context.calls is not calls):
        raise ValueError("FA3 source context does not bind this actual call plan")
    by_id, source_completions = context.by_id, context.completions
    for group in groups.values():
        main = group[0]
        attrs = main.memory_source.attributes
        full = ("source_program" in attrs
            and attrs["source_program"] == "vllm_0_30_fa3_gqa_full_decode")
        if ("fa3_gqa_caller_scenario" in attrs and main.kernel_spec.kind is KernelKind.FLASH_ATTN
                and not full):
            producer_id = attrs["fa3_gqa_metadata_prepare_node"]
            if producer_id not in by_id:
                raise ValueError("FA3 GQA lost its actual shared metadata producer")
            producer = by_id[producer_id]
            if (producer.kernel_spec.kind is not KernelKind.VECTOR
                    or producer.kernel_spec.variant != "fa3_prepare_varlen"
                    or producer.kernel_spec.dtype != "int32"
                    or producer.kernel_spec.shape["batch"] != main.kernel_spec.shape["batch"]
                    or producer.rank != main.rank or producer.pool_name != main.pool_name
                    or producer.kernel_spec.hardware_id != main.kernel_spec.hardware_id
                    or producer.source_node_ids != (producer_id,)
                    or producer_id not in main.dependencies
                    or producer_id not in main.memory_source.predecessors):
                raise ValueError("FA3 GQA metadata readiness requires its actual same-rank prepare")
        if "source_program" in attrs and attrs["source_program"] == "vllm_0_30_fa3_gqa_full_decode":
            if "fa3_full_aot_layout" in attrs:
                _validate_full_fa3_aot_frame(group,by_id,source_completions,context=context)
            producer_id = attrs["fa3_gqa_metadata_prepare_node"]
            if producer_id not in by_id:
                raise ValueError("FA3 FULL lost its actual metadata producer")
            producer = by_id[producer_id]
            if (producer.kernel_spec.kind is not KernelKind.VECTOR
                    or producer.kernel_spec.variant != "fa3_prepare_varlen"
                    or producer.kernel_spec.shape["batch"] != main.kernel_spec.shape["batch"]
                    or producer.rank != main.rank or producer.pool_name != main.pool_name
                    or producer.kernel_spec.hardware_id != main.kernel_spec.hardware_id
                    or (producer_id not in main.dependencies and "fa3_full_aot_layout" not in attrs)):
                raise ValueError("FA3 FULL metadata readiness requires its actual same-rank prepare")
    return {call.call_id: tuple(group) for group in groups.values() for call in group}


def _same_fa3_port(actual, role):
    return (actual.allocation_id, actual.offset_bytes, actual.shape, actual.strides_bytes,
        actual.dtype, actual.element_bytes, actual.region()) == (
        role.allocation_id, role.offset_bytes, role.shape, role.strides_bytes,
        role.dtype, role.element_bytes, role.region())


def predict_fa3_plan_call(call, calls, hardware, method, evaluate, *, source_calls=None):
    """Validate actual parents and enter the library scope before memo lookup.

    Only binding identity is projected back to the original source call. The
    plan retains every stage, dependency, allocation and readiness reference.
    """
    if "fa3_gqa_caller_scenario" in call.memory_source.attributes:
        return _predict_fa3_gqa_plan_call(call, calls, hardware, method, evaluate, source_calls=source_calls)
    if "fa3_expanded_source_revision" in call.memory_source.attributes:
        return _predict_fa3_expanded_plan_call(call, calls, hardware, method, evaluate)
    if ("source_component" in call.memory_source.attributes
            and call.memory_source.attributes["source_component"] == "fa3_mla_unsplit_prefill_call"):
        return _predict_fa3_unsplit_plan_call(call, calls, hardware, method, evaluate)
    from src.kernel.memory_access_builder import prediction_memory_bindings
    from src.kernel.memory_flash_attention import (
        current_fa3_split_prefill_prediction_work,
        fa3_split_prefill_call_prediction_scope,
    )
    source = call.memory_source
    hardware.require_fa3_mla_source_revision(source.attributes["fa3_source_revision"])
    model_source = ("source_component" in source.attributes
        and source.attributes["source_component"] == "fa3_mla_split_prefill_call")
    if model_source:
        revision = hardware.fa3_mla_source_revision_for_model(source.attributes["source_model_execution_version"])
        if (source.attributes["fa3_source_revision"] != revision
                or source.attributes["fa3_backend_source"] != hardware.fa3_mla_model_backend_source(
                    source.attributes["source_model_execution_version"])
                or source.attributes["fa3_runtime_binary_bound"] is not False
                or source.attributes["fa3_length_array_transport_bound"] is not False):
            raise ValueError("FA3 model call changed its declared source/backend scope")
    if "scheduler_metadata_precomputed" not in source.attributes:
        raise ValueError("FA3 source call must preserve its actual metadata lifetime")
    precomputed = source.attributes["scheduler_metadata_precomputed"]
    if type(precomputed) is not bool:
        raise TypeError("FA3 metadata lifetime must be boolean")
    if len(calls) != (2 if precomputed else 3):
        raise ValueError("FA3 plan must preserve the complete ordered library call")
    main, combine = (stage.kernel_spec for stage in calls[-2:])
    stage_binding = calls[-2].memory_bindings
    if not isinstance(stage_binding, ElementwiseBindings):
        raise ValueError("FA3 plan requires actual library source parents")
    bindings = replace(stage_binding, binding_id=source.node_id + ":library-memory", call_id=source.node_id)
    roles = {view.view_id: view for view in bindings.views}
    output_roles = (source.attributes["fa3_source_output_roles"]
        if "fa3_source_output_roles" in source.attributes else ("output", "lse"))
    if (len(roles) != len(bindings.views)
            or any(role not in roles for role in (*bindings.read_views, "output", "lse"))
            or len(source.inputs) != len(bindings.read_views) or len(source.outputs) != len(output_roles)
            or any(not _same_fa3_port(port, roles[role]) for port, role in zip(
                source.inputs, bindings.read_views, strict=True))
            or any(not _same_fa3_port(port, roles[role]) for port, role in zip(
                source.outputs, output_roles, strict=True))):
        raise ValueError("FA3 plan source ports must retain the actual library parents")
    if {item.allocation_id: item.with_lifetime(item.allocation_id, item.allocated_event, None)
            for item in source.allocations} != {
            item.allocation_id: item for item in bindings.allocations}:
        raise ValueError("FA3 plan source allocation lifetimes changed")
    for index, stage in enumerate(calls):
        if (stage.memory_source is not source or stage.source_node_ids != (source.node_id,)
                or stage.rank != source.rank or stage.pool_name != source.pool_name
                or stage.kernel_spec.hardware_id != hardware.hardware_id or stage.resource != "compute"):
            raise ValueError("FA3 plan stages crossed source or hardware ownership")
        if index and calls[index - 1].call_id not in stage.dependencies:
            raise ValueError("FA3 plan lost its actual stage dependency")
        if stage.kernel_spec.kind is KernelKind.FLASH_ATTN:
            actual = stage.memory_bindings
            if not isinstance(actual, ElementwiseBindings) or stage.memory_unbound_reason is not None:
                raise ValueError("FA3 stage requires its own bound source identity")
            for field in fields(ElementwiseBindings):
                if field.name not in ("binding_id", "call_id") and getattr(actual, field.name) != getattr(bindings, field.name):
                    raise ValueError("FA3 stage changed shared library operands")
        elif (index != 0 or precomputed or stage.kernel_spec.kind is not KernelKind.VECTOR
                or stage.kernel_spec.variant != "fa3_prepare_varlen" or stage.memory_bindings is not None):
            raise ValueError("FA3 plan has an invalid prepare stage")
    attributes = dict(main.attributes)
    if "return_softmax_lse" in combine.attributes:
        attributes["return_softmax_lse"] = combine.attributes["return_softmax_lse"]
    expected = hardware.fa3_mla_forward_specs(main.shape, main.dtype, main.variant, attributes,
        scheduler_metadata_precomputed=precomputed)
    if tuple(stage.kernel_spec for stage in calls) != expected:
        raise ValueError("FA3 plan stages differ from the complete source factory call")
    with fa3_split_prefill_call_prediction_scope(main, combine, hardware, bindings,
            scheduler_metadata_precomputed=precomputed) as work:
        if model_source and source.attributes["fa3_dynamic_splits"] != work["splits"]:
            raise ValueError("FA3 model metadata differs from its actual source split program")
        actual = bindings if call.kernel_spec.kind is KernelKind.FLASH_ATTN else None
        if actual is not None:
            with prediction_memory_bindings(call.kernel_spec, actual, method):
                current_fa3_split_prefill_prediction_work(call.kernel_spec, hardware)
        prediction = evaluate(call.kernel_spec, hardware, method, memory_bindings=actual)
        if model_source and actual is not None and prediction.diagnostics is not None:
            prediction = replace(prediction, diagnostics=replace(prediction.diagnostics,
                limitations=prediction.diagnostics.limitations + (
                    "fa3_v030_declared_default_source_not_installed_GPU_binary_binding"
                        if source.attributes["source_model_execution_version"] == "vllm_0_30_0"
                        else "fa3_v027_declared_default_source_not_installed_GPU_binary_binding",
                    "fa3_caller_length_array_transport_unbound",)))
        return prediction


def _predict_fa3_expanded_plan_call(call, calls, hardware, method, evaluate):
    """Validate complete D256 MHA calls without inventing native scratch."""
    from src.kernel.memory_access import Strided
    from src.kernel.memory_access_builder import bound_prediction_memory_program
    source, attrs = call.memory_source, call.memory_source.attributes
    revision = hardware.fa3_mla_source_revision_for_model(attrs["source_model_execution_version"])
    if (attrs["fa3_expanded_source_revision"] != revision
            or attrs["source_component"] != "fa3_expanded_mla_prefill_call"
            or attrs["scheduler_metadata_precomputed"] is not False
            or attrs["fa3_runtime_binary_bound"] is not False
            or attrs["fa3_length_array_transport_bound"] is not False
            or attrs["fa3_native_workspace_bound"] is not False
            or attrs["fa3_backend_source"] != "vllm_0_30_default_hopper_flash_attn3"
            or attrs["causal"] is not True or attrs["return_softmax_lse"] is not False
            or attrs["q_lens"] != attrs["kv_lens"] or attrs["full_cuda_graphs"] is not False
            or type(attrs["max_cudagraph_capture_size"]) is not int or attrs["max_cudagraph_capture_size"] != 0
            or "flash_attn_version" in attrs and (type(attrs["flash_attn_version"]) is not int or attrs["flash_attn_version"] != 3)
            or any(field in attrs for field in ("attention_backend", "graph_runtime_mode", "graph_logical_tokens",
                "graph_model_input_tokens", "model_input_shape", "model_input_q_lens", "model_input_kv_lens"))
            or len(calls) not in (2, 3)
            or len(source.inputs) != 3 or len(source.outputs) != 1):
        raise ValueError("expanded FA3 plan changed its source identity or public parents")
    main = calls[1].kernel_spec
    if (main.kind is not KernelKind.FLASH_ATTN or main.variant != "prefill"
            or main.dtype != "bf16" or main.shape["head_dim"] != 256
            or main.shape["num_query_heads"] != main.shape["num_kv_heads"]):
        raise ValueError("expanded FA3 requires its D256 expanded MHA program")
    expected = hardware.fa3_gqa_forward_specs(main.shape, main.dtype,
        {**{name: attrs[name] for name in ("causal", "kv_dist", "q_lens", "kv_lens")},
            "kv_layout": "contiguous", "return_softmax_lse": attrs["return_softmax_lse"]},
        page_size=None, num_splits=0, scheduler_metadata_precomputed=False)
    if tuple(stage.kernel_spec for stage in calls) != expected:
        raise ValueError("expanded FA3 plan lost a source stage")
    for index, stage in enumerate(calls):
        if (stage.memory_source is not source or stage.rank != source.rank or stage.pool_name != source.pool_name
                or stage.source_node_ids != (source.node_id,) or stage.resource != "compute"
                or stage.kernel_spec.hardware_id != hardware.hardware_id
                or index and calls[index-1].call_id not in stage.dependencies):
            raise ValueError("expanded FA3 plan lost its source stage dependency")
    shape = (sum(main.attributes["q_lens"]), main.shape["num_query_heads"], 256)
    if any(view.shape != shape or view.dtype != "bf16" for view in (*source.inputs, *source.outputs)):
        raise ValueError("expanded FA3 actual Q/K/V/O extents disagree with its source call")
    for view in (*source.inputs, *source.outputs):
        if view.region() != Strided(view.offset_bytes, view.shape, view.strides_bytes, view.element_bytes):
            raise ValueError("expanded FA3 public parent geometry was corrupted")
    if call.kernel_spec.kind is KernelKind.FLASH_ATTN:
        expected_binding, reason = _bind_expanded_attention(call, source)
        if call.memory_bindings != expected_binding or call.memory_unbound_reason != reason:
            raise ValueError("expanded FA3 bound operands lost actual parents, readiness or lifetimes")
    if call.memory_bindings is not None:
        bound_prediction_memory_program(call.kernel_spec, hardware, call.memory_bindings, method)
    if call.memory_bindings is None and call.memory_unbound_reason is None:
        raise ValueError("expanded FA3 native ABI requires an explicit unbound reason")
    prediction = evaluate(call.kernel_spec, hardware, method, memory_bindings=call.memory_bindings)
    if call.kernel_spec.kind is KernelKind.FLASH_ATTN and prediction.diagnostics is not None:
        prediction = replace(prediction, diagnostics=replace(prediction.diagnostics,
            limitations=prediction.diagnostics.limitations + (
                "fa3_expanded_mla_complete_source_call_and_public_parents_validated",
                "fa3_expanded_mla_native_scratch_and_metadata_transport_unbound",
                "fa3_declared_vllm030_default_source_not_observed_runtime_binary",)))
    return prediction


def _predict_fa3_gqa_plan_call(call, calls, hardware, method, evaluate, *, source_calls=None):
    """Keep the complete GQA source sequence; unbound native ABI stays partial."""
    from src.kernel.memory_access import Strided
    from src.kernel.memory_access_builder import bound_prediction_memory_program
    source = call.memory_source
    attrs = source.attributes
    native_aot = "fa3_full_aot_layout" in attrs
    if native_aot:
        if source_calls is None:
            raise ValueError("FA3 native AOT prediction requires its complete source producer frame")
        context = (source_calls if type(source_calls) is _FA3SourceCallContext
            else _FA3SourceCallContext(source_calls))
        _validate_full_fa3_aot_frame(calls, context.by_id, context.completions, context=context)
    facts = fa3_gqa_caller_facts(attrs,hardware)
    if any(type(attrs[name]) is not type(value) or attrs[name] != value for name, value in facts.items()):
        raise ValueError("GQA plan lost its actual source scenario facts")
    main = calls[0].kernel_spec
    if (main.kind is not KernelKind.FLASH_ATTN or main.variant not in ("prefill", "decode")
            or len(source.inputs) != (7 if "fa3_full_aot_layout" in attrs else 4)
            or len(source.outputs) != (2 if "fa3_full_aot_layout" in attrs else 1)
            or (attrs["fa3_gqa_metadata_prepare_node"] not in calls[0].dependencies and not native_aot)):
        raise ValueError("GQA plan lost public ports or its shared metadata producer")
    expected = hardware.fa3_gqa_forward_specs(main.shape, attrs["fa3_gqa_operand_dtype"], main.attributes,
        page_size=facts["fa3_gqa_page_size"], num_splits=facts["fa3_gqa_num_splits"],
        scheduler_metadata_precomputed=facts["fa3_gqa_scheduler_metadata_precomputed"])
    if tuple(stage.kernel_spec for stage in calls) != expected:
        raise ValueError("GQA plan differs from the complete source library call")
    for index, stage in enumerate(calls):
        if (stage.memory_source is not source or stage.rank != source.rank or stage.pool_name != source.pool_name
                or stage.source_node_ids != (source.node_id,) or stage.resource != "compute"
                or stage.kernel_spec.hardware_id != hardware.hardware_id
                or index and calls[index - 1].call_id not in stage.dependencies):
            raise ValueError("GQA plan crossed source ownership or lost a stage dependency")
    if any(view.dtype != main.dtype for view in (*source.inputs[:4], source.outputs[0])):
        raise ValueError("FA3 GQA public Q/K/V/cache/O dtype must match the actual operand dtype")
    for view in (*source.inputs, *source.outputs):
        if view.region() != Strided(view.offset_bytes, view.shape, view.strides_bytes, view.element_bytes):
            raise ValueError("GQA caller public storage geometry was corrupted")
    rows = sum(main.attributes["q_lens"])
    full = ("source_program" in main.attributes
        and main.attributes["source_program"] == "vllm_0_30_fa3_gqa_full_decode")
    caller_rows = main.attributes["graph_model_input_tokens"] if full else rows
    if "fa3_gqa_query_tokens" in facts and caller_rows!=facts["fa3_gqa_query_tokens"]:
        raise ValueError("FA3 caller split policy lost the actual caller token extent")
    if full:
        if (attrs["graph_runtime_mode"] != "FULL" or attrs["fa3_gqa_execution_origin"] != "full_captured"
                or type(attrs["fa3_full_max_num_seqs"]) is not int or attrs["fa3_full_max_num_seqs"] <= 0
                or main.shape["batch"] != min(caller_rows,attrs["fa3_full_max_num_seqs"])
                or type(attrs["fa3_full_capture_max_seq_len"]) is not int
                or main.shape["kv_len"] > attrs["fa3_full_capture_max_seq_len"]
                or attrs["fa3_gqa_native_workspace_bound"] is not False
                or any(type(attrs[field]) is not type(main.attributes[field]) or attrs[field] != main.attributes[field]
                    for field in ("source_program", "graph_runtime_mode", "graph_model_input_tokens",
                        "model_input_q_lens", "model_input_kv_lens", "q_lens", "kv_lens"))
                or not isinstance(attrs["fa3_full_capture_id"],str) or not attrs["fa3_full_capture_id"]):
            raise ValueError("FA3 FULL source lost its actual capture descriptor/bounds")
        if source.input_producers != attrs["fa3_full_input_producers"]:
            raise ValueError("FA3 FULL public input readiness changed")
        allocations = {item.allocation_id:item for item in source.allocations}
        for port,parent in zip((source.inputs[0],source.outputs[0]), attrs["fa3_full_physical_parents"],strict=True):
            allocation,offset,shape,strides,allocated,retired = parent
            if (allocation != port.allocation_id
                    or (allocations[allocation].allocated_event, allocations[allocation].freed_event) != (allocated,retired) or offset != port.offset_bytes or shape != (caller_rows,*port.shape[1:])
                    or strides != port.strides_bytes or offset+sum((n-1)*s for n,s in zip(shape,strides,strict=True))
                        +port.element_bytes > allocations[allocation].size_bytes):
                raise ValueError("FA3 FULL physical Q/O parent alias or capacity changed")
    heads, dimension = main.shape["num_query_heads"], main.shape["head_dim"]
    if prod(source.inputs[0].shape) != rows * heads * dimension or prod(source.outputs[0].shape) != rows * heads * dimension:
        raise ValueError("GQA public input/output extents disagree with the source call")
    if not full:
        if (source.kind is not WorkloadKind.FLASH_ATTN or source.variant != main.variant
                or attrs["source_component"] != "symbolic_gqa_mathematical"):
            raise ValueError("GQA source lost its actual ordinary attention operation")
        for role, port in zip(("key", "value"), source.inputs[1:3], strict=True):
            _reshape(port, role, (rows, main.shape["num_kv_heads"], dimension))
        if call.kernel_spec.kind is KernelKind.FLASH_ATTN:
            # Main and combine retain the same source wrapper but use
            # different operand ABIs. Revalidate through the binding
            # dispatcher used by TraceMemoryContext, including its exact
            # unbound reason, before either direct or cached prediction.
            binding, reason = bind_attention_memory(call, source)
            if call.memory_bindings != binding or call.memory_unbound_reason != reason:
                raise ValueError("GQA public operands changed actual parents, readiness or lifetimes")
    if call.memory_bindings is not None:
        bound_prediction_memory_program(call.kernel_spec, hardware, call.memory_bindings, method)
    elif call.memory_unbound_reason is None:
        raise ValueError("GQA unbound native operands require an explicit reason")
    prediction = evaluate(call.kernel_spec, hardware, method, memory_bindings=call.memory_bindings)
    if call.kernel_spec.kind is KernelKind.FLASH_ATTN and prediction.diagnostics is not None:
        prediction = replace(prediction, diagnostics=replace(prediction.diagnostics,
            limitations=prediction.diagnostics.limitations + (
                "fa3_gqa_complete_source_call_and_shared_metadata_dependency_validated",
                "fa3_gqa_native_workspace_pages_and_metadata_transport_unbound",
                "fa3_gqa_declared_default_caller_not_observed_runtime_binary",)))
    return prediction


def _predict_fa3_unsplit_plan_call(call, calls, hardware, method, evaluate):
    from src.kernel.memory_flash_attention import fa3_prefill_prediction_work
    source = call.memory_source
    attributes = source.attributes
    revision = hardware.fa3_mla_source_revision_for_model(attributes["source_model_execution_version"])
    if (attributes["fa3_source_revision"] != revision
            or attributes["scheduler_metadata_precomputed"] is not False
            or attributes["fa3_backend_source"] != hardware.fa3_mla_model_backend_source(
                attributes["source_model_execution_version"])
            or attributes["fa3_runtime_binary_bound"] is not False
            or attributes["fa3_length_array_transport_bound"] is not False or len(calls) != 2):
        raise ValueError("FA3 unsplit model call changed its source/backend/lifetime scope")
    prepare, main_call = calls
    main = main_call.kernel_spec
    binding = main_call.memory_bindings
    if not isinstance(binding, ElementwiseBindings):
        raise ValueError("FA3 unsplit model call lost actual native ports")
    expected = hardware.fa3_mla_forward_specs(main.shape, main.dtype, main.variant, main.attributes,
        scheduler_metadata_precomputed=False)
    if (tuple(stage.kernel_spec for stage in calls) != expected
            or prepare.call_id not in main_call.dependencies or prepare.memory_bindings is not None):
        raise ValueError("FA3 unsplit model call lost its complete source stage sequence")
    for stage in calls:
        if (stage.memory_source is not source or stage.source_node_ids != (source.node_id,)
                or stage.rank != source.rank or stage.pool_name != source.pool_name
                or stage.kernel_spec.hardware_id != hardware.hardware_id or stage.resource != "compute"):
            raise ValueError("FA3 unsplit stages crossed actual source ownership")
    parents = {view.view_id: view for view in binding.views}
    output_roles = ("output", "lse", "scheduler_metadata")
    if (attributes["fa3_source_output_roles"] != output_roles
            or len(source.inputs) != len(binding.read_views) or len(source.outputs) != 3
            or any(not _same_fa3_port(port, parents[role]) for port, role in zip(
                source.inputs, binding.read_views, strict=True))
            or any(not _same_fa3_port(port, parents[role]) for port, role in zip(
                source.outputs, output_roles, strict=True))):
        raise ValueError("FA3 unsplit source ports differ from actual caller operands")
    if {item.allocation_id: item.with_lifetime(item.allocation_id, item.allocated_event, None)
            for item in source.allocations} != {item.allocation_id: item for item in binding.allocations}:
        raise ValueError("FA3 unsplit source allocation lifetimes changed")
    # Validation runs even when the evaluator is about to return a memo hit.
    fa3_prefill_prediction_work(main, hardware, binding)
    prediction = evaluate(call.kernel_spec, hardware, method,
        memory_bindings=binding if call is main_call else None)
    if call is main_call and prediction.diagnostics is not None:
        prediction = replace(prediction, diagnostics=replace(prediction.diagnostics,
            limitations=prediction.diagnostics.limitations + (
                "fa3_unsplit_model_source_aliases_and_readiness_validated",
                "fa3_v030_declared_default_source_not_installed_GPU_binary_binding"
                    if attributes["source_model_execution_version"] == "vllm_0_30_0"
                    else "fa3_v027_declared_default_source_not_installed_GPU_binary_binding",
                "fa3_caller_length_array_transport_unbound",)))
    return prediction


def _bind_fa3_unsplit_prefill(call, source):
    input_roles = ("query", "key", "value", "cu_query", "cu_key")
    output_roles = ("output", "lse", "scheduler_metadata")
    if len(source.inputs) != 5 or len(source.outputs) != 3 or source.attributes["fa3_source_output_roles"] != output_roles:
        raise ValueError("FA3 unsplit model call requires its exact library ports")
    parents = {role: replace(view, view_id=role) for role, view in zip(
        (*input_roles, *output_roles), (*source.inputs, *source.outputs), strict=True)}
    library_call = replace(call, call_id=source.node_id, memory_bindings=None,
        memory_source=None, memory_unbound_reason=None)
    allocations, events, ready, visibility = source_memory_events(library_call, source,
        tuple(zip((parents[role] for role in input_roles), source.input_producers, strict=True)))
    return ElementwiseBindings(call.call_id + ":source-memory", call.call_id, allocations,
        tuple(parents.values()), events, input_roles, ("output", "lse"), ready,
        call.rank, "source-forward", visibility), None


def _bind_fa3_split_prefill(call, source):
    """Project real model/library ports, including source-owned scratch roots."""
    input_roles = ("query", "key", "value", "cu_query", "cu_key")
    output_roles = ("output", "lse", "scheduler_metadata", "partial_output", "partial_lse")
    if (source.attributes["scheduler_metadata_precomputed"] is not False
            or source.attributes["fa3_source_output_roles"] != output_roles
            or len(source.inputs) != len(input_roles) or len(source.outputs) != len(output_roles)):
        raise ValueError("FA3 model call requires its exact eager library ports")
    parents = {role: replace(view, view_id=role) for role, view in zip(
        (*input_roles, *output_roles), (*source.inputs, *source.outputs), strict=True)}
    q_lens = source.attributes["q_lens"]
    splits = source.attributes["fa3_dynamic_splits"]
    if len(q_lens) != len(splits):
        raise ValueError("FA3 model call lost per-request dynamic split facts")
    batch = len(q_lens)
    rounded = -(-batch // 4) * 4
    meta = parents["scheduler_metadata"]
    views = list(parents.values())
    for role, offset, shape in (("dynamic_splits", rounded * 4, (batch,)),
            ("semaphore", 2 * rounded * 4, (1,))):
        views.append(replace(meta, view_id=role, offset_bytes=meta.offset_bytes + offset,
            shape=shape, strides_bytes=(4,)))
    prefix = 0
    for index, (length, count) in enumerate(zip(q_lens, splits, strict=True)):
        if count > 1:
            for role in ("partial_output", "partial_lse"):
                parent = parents[role]
                row_axis = 2
                shape = (count, parent.shape[1], length, *parent.shape[3:])
                views.append(replace(parent, view_id=f"{role}.r{index}", shape=shape,
                    offset_bytes=parent.offset_bytes + prefix * parent.strides_bytes[row_axis]))
        prefix += length
    # Shared source IDs give every stage equal readiness facts. Stage-specific
    # call IDs are retained on the binding and never replace storage identity.
    library_call = replace(call, call_id=source.node_id, memory_bindings=None,
        memory_source=None, memory_unbound_reason=None)
    allocations, events, ready, visibility = source_memory_events(library_call, source,
        tuple(zip((parents[role] for role in input_roles), source.input_producers, strict=True)))
    return ElementwiseBindings(call.call_id + ":source-memory", call.call_id, allocations,
        tuple(views), events, input_roles, ("output", "lse", "semaphore"), ready,
        call.rank, "source-forward", visibility), None


def _reshape(view, name, shape):
    """Reinterpret only actual dense trailing dimensions, preserving row pitch."""
    if view.shape == shape:
        return replace(view, view_id=name)
    if len(view.shape) == 2 and len(shape) == 3 and view.shape == (shape[0], prod(shape[1:])):
        if view.strides_bytes[-1] != view.element_bytes:
            raise ValueError("attention source requires dense inner channels")
        return replace(view, view_id=name, shape=shape,
            strides_bytes=(view.strides_bytes[0], shape[2] * view.element_bytes, view.element_bytes))
    if len(view.shape) == 4 and len(shape) == 3 and prod(view.shape[:2]) == shape[0] and view.shape[2:] == shape[1:]:
        if view.strides_bytes[0] != view.shape[1] * view.strides_bytes[1]:
            raise ValueError("attention source batch rows must be contiguous")
        return replace(view, view_id=name, shape=shape, strides_bytes=view.strides_bytes[1:])
    raise ValueError(f"attention source shape {view.shape} does not match {name} {shape}")


def _logical_read_events(call, source, reads, producers):
    # Derive producer/barrier dependencies from the actual parent ports first.
    parents = tuple((parent, producer) for (_, parent), producer in zip(reads, producers, strict=True))
    allocations, events, ready, visibility = source_memory_events(call, source, parents)
    known_views = {view.view_id for view, _ in reads}
    events = [replace(event, view_id=None) if event.view_id is not None and event.view_id not in known_views
        else event for event in events]
    refs = []
    for (view, parent), ref in zip(reads, ready, strict=True):
        if isinstance(view, LogicalSelectionView):
            identity = call.call_id + ":selected:" + view.view_id
            events.append(ProgramEvent(identity, EventKind.LOGICAL_AVAILABLE, view.allocation_id,
                view.region(), visibility, (ref,)))
            refs.append(EventRef(identity, EventKind.LOGICAL_AVAILABLE))
        else:
            # The ready region is the actual parent and covers its role view.
            refs.append(ref)
    return allocations, tuple(events), tuple(refs), visibility


def _elementwise(call, source, reads, writes, producers):
    allocations, events, ready, visibility = _logical_read_events(call, source, reads, producers)
    views = tuple(view for view, _ in reads) + tuple(writes)
    return ElementwiseBindings(call.call_id + ":source-memory", call.call_id, allocations,
        views, events, tuple(view.view_id for view, _ in reads),
        tuple(view.view_id for view in writes), ready, call.rank, "source-forward", visibility), None


def _bind_paged_cache(call, source):
    if len(source.inputs) == 2:
        return None, "paged_kv_source_slot_tensor_unavailable"
    if len(source.inputs) != 3 or len(source.outputs) != 1:
        raise ValueError("paged KV source requires K/V/slots and one actual cache arena")
    spec = call.kernel_spec
    shape = tuple(spec.shape[name] for name in ("seq_len", "num_kv_heads", "head_dim"))
    key, value = (_reshape(view, name, shape) for view, name in zip(source.inputs[:2], ("key", "value")))
    slots = replace(source.inputs[2], view_id="slots")
    if slots.shape != (shape[0],) or slots.dtype != "int64":
        raise ValueError("paged KV source slot tensor must use its declared INT64 ABI")
    slot_abi = paged_kv_source_index_type(spec)
    if slot_abi is None:
        return None, "paged_kv_source_tensor_only_compatibility_abi_unavailable"
    if (slots.dtype, slots.element_bytes) != slot_abi:
        return None, "paged_kv_source_slot_cast_unbound"
    cache, = source.outputs
    allocations, events, ready, visibility = source_memory_events(call, source,
        tuple(zip((key, value, slots), source.input_producers, strict=True)))
    return CacheUpdateBindings(call.call_id + ":source-memory", call.call_id, allocations,
        (key, value, slots), events, "key", "value", "slots", cache.allocation_id,
        cache.allocation_id, *ready, call.rank, "source-forward", visibility), None


def _bind_public_attention(call, source):
    spec = call.kernel_spec
    if not has_public_attention_source(spec):
        return None, "attention_native_metadata_or_workspace_ports_unavailable"
    if "model_input_q_lens" in spec.attributes and (
            spec.attributes["model_input_q_lens"] != spec.attributes["q_lens"]
            or spec.attributes["model_input_kv_lens"] != spec.attributes["kv_lens"]):
        return None, "attention_graph_physical_operand_mapping_unavailable"
    if spec.launch.tiling.num_kv_splits not in (None, 1):
        return None, "attention_split_workspace_source_ports_unavailable"
    if len(source.inputs) != 4 or len(source.outputs) != 1:
        return None, "attention_source_public_operand_layout_unavailable"
    query, key, value, cache = source.inputs
    q, kv = sum(spec.attributes["q_lens"]), sum(spec.attributes["kv_lens"])
    heads, kv_heads, dimension = (spec.shape[name] for name in ("num_query_heads", "num_kv_heads", "head_dim"))
    query = _reshape(query, "query", (q, heads, dimension))
    output = _reshape(source.outputs[0], "output", (q, heads, dimension))
    reads = [(query, query)]
    producers = [source.input_producers[0]]
    # The actual cache call consumes prior-context K/V even when fresh K/V
    # tensors also appear on the public wrapper. Unknown native page selection
    # must not be substituted by those shorter fresh tensors.
    for role in ("key", "value"):
        selected = LogicalSelectionView(role, cache.allocation_id, (kv, kv_heads, dimension),
            spec.dtype, spec.shape["dtype_bytes"], call.call_id + ":cache:" + role, None)
        reads.append((selected, cache))
        producers.append(source.input_producers[3])
    return _elementwise(call, source, tuple(reads), (output,), tuple(producers))


def _bind_expanded_attention(call, source):
    if call.kernel_spec.variant == "combine" or call.kernel_spec.launch.tiling.num_kv_splits not in (None, 1):
        return None, "expanded_fa3_native_split_workspace_ports_unavailable"
    if len(source.inputs) != 3 or len(source.outputs) != 1:
        raise ValueError("expanded attention requires actual Q/K/V and O parents")
    reads = tuple((replace(view, view_id=role), view) for view, role in zip(
        source.inputs, ("query", "key", "value"), strict=True))
    output = replace(source.outputs[0], view_id="output")
    return _elementwise(call, source, reads, (output,), source.input_producers)


def _bind_mla_decode(call, source):
    spec = call.kernel_spec
    if spec.launch is None:
        return None, "mla_mathematical_memory_source_unavailable"
    if spec.launch.tiling.num_kv_splits not in (None, 1) or spec.launch.algorithm == "vllm_fa3_sm90_mla_decode":
        return None, "mla_split_workspace_source_ports_unavailable"
    if len(source.inputs) != 2 or len(source.outputs) != 2:
        return None, "mla_source_public_operand_layout_unavailable"
    query, cache = source.inputs
    q_lens, kv_lens = spec.attributes["q_lens"], spec.attributes["kv_lens"]
    rows, heads, dimension = sum(q_lens), spec.shape["num_query_heads"], spec.shape["qk_head_dim"]
    if query.shape[0] != rows and len(query.shape) != 4:
        return None, "mla_padded_public_query_mapping_unavailable"
    query = _reshape(query, "query", (rows, heads, dimension))
    output = _reshape(source.outputs[0], "output", (rows, heads, spec.shape["v_head_dim"]))
    rank = spec.shape["kv_lora_rank"]
    reads, producers, offset = [], [], 0
    for index, (count, length) in enumerate(zip(q_lens, kv_lens, strict=True)):
        suffix = "" if len(q_lens) == 1 else f".request{index}"
        qview = replace(query, view_id="query" + suffix, shape=(count, heads, dimension),
            offset_bytes=query.offset_bytes + offset * query.strides_bytes[0])
        reads.append((qview, query)); producers.append(source.input_producers[0]); offset += count
        for role, width in (("compressed_kv", rank), ("key_rope", dimension - rank)):
            if not width:
                continue
            name = role + suffix
            view = LogicalSelectionView(name, cache.allocation_id, (length, width), spec.dtype,
                spec.shape["dtype_bytes"], call.call_id + ":" + name, None)
            reads.append((view, cache)); producers.append(source.input_producers[1])
    return _elementwise(call, source, tuple(reads), (output,), tuple(producers))


def _bind_mathematical_attention(call, source):
    spec = call.kernel_spec
    rows = sum(spec.attributes["q_lens"])
    cache_rows = sum(spec.attributes["kv_lens"])
    heads = spec.shape["num_query_heads"]
    if spec.variant in ("prefill", "decode"):
        dimension, kv_heads = spec.shape["head_dim"], spec.shape["num_kv_heads"]
        if len(source.inputs) not in (2, 3, 4) or not source.outputs:
            return None, "gqa_mathematical_source_operand_ports_unavailable"
        if (prod(source.inputs[0].shape) != rows * heads * dimension
                or prod(source.outputs[0].shape) != rows * heads * dimension):
            return None, "gqa_padded_public_operand_mapping_unavailable"
        query = _reshape(source.inputs[0], "query", (rows, heads, dimension))
        output = _reshape(source.outputs[0], "output", (rows, heads, dimension))
        reads = [(query, query)]
        producers = [source.input_producers[0]]
        if len(source.inputs) == 3:
            if spec.attributes["q_lens"] != spec.attributes["kv_lens"]:
                return None, "gqa_direct_history_source_mapping_unavailable"
            for index, role in enumerate(("key", "value"), 1):
                view = _reshape(source.inputs[index], role, (cache_rows, kv_heads, dimension))
                reads.append((view, view)); producers.append(source.input_producers[index])
        else:
            cache_index = 1 if len(source.inputs) == 2 else 3
            cache = source.inputs[cache_index]
            for role in ("key", "value"):
                view = LogicalSelectionView(role, cache.allocation_id, (cache_rows, kv_heads, dimension),
                    spec.dtype, spec.shape["dtype_bytes"], call.call_id + ":cache:" + role, None)
                reads.append((view, cache)); producers.append(source.input_producers[cache_index])
        return _elementwise(call, source, tuple(reads), (output,), tuple(producers))
    if spec.variant in ("mla_decode", "mla_absorbed_prefill"):
        if len(source.inputs) != 2 or len(source.outputs) not in (1, 2):
            return None, "mla_mathematical_source_operand_ports_unavailable"
        dimension, value_dim = spec.shape["qk_head_dim"], spec.shape["v_head_dim"]
        if (prod(source.inputs[0].shape) != rows * heads * dimension
                or prod(source.outputs[0].shape) != rows * heads * value_dim):
            return None, "mla_padded_public_operand_mapping_unavailable"
        query = _reshape(source.inputs[0], "query", (rows, heads, dimension))
        cache = source.inputs[1]
        if cache.dtype != spec.dtype or cache.shape[-1] != dimension:
            raise ValueError("MLA source cache must retain the declared latent/RoPE record")
        selected = LogicalSelectionView("compressed_kv", cache.allocation_id, (cache_rows, dimension),
            spec.dtype, spec.shape["dtype_bytes"], call.call_id + ":cache:compressed_kv", None)
        output = _reshape(source.outputs[0], "output", (rows, heads, value_dim))
        writes = (output,)
        if len(source.outputs) == 2:
            writes += (replace(source.outputs[1], view_id="lse"),)
        return _elementwise(call, source, ((query, query), (selected, cache)), writes,
            source.input_producers)
    return None, "attention_mathematical_variant_ports_unavailable"


def _bind_mla_cache(call, source):
    spec = call.kernel_spec
    if len(source.inputs) != 3 or len(source.outputs) != 1:
        return None, "mla_cache_source_latent_rope_slots_unavailable"
    latent, rope, slots = source.inputs
    cache, = source.outputs
    tokens, dimension = spec.shape["seq_len"], spec.shape["compressed_dim"]
    if (latent.shape[0] != tokens or len(latent.shape) != 2 or rope.shape[:2] != (tokens, 1)
            or latent.shape[1] + rope.shape[2] != dimension or slots.shape != (tokens,)
            or slots.dtype != "int64" or cache.dtype != spec.dtype
            or latent.dtype != spec.dtype or rope.dtype != spec.dtype):
        raise ValueError("MLA cache source latent/RoPE/slot geometry disagrees with the kernel")
    latent = _reshape(latent, "latent", (tokens, 1, latent.shape[1]))
    rope, slots = replace(rope, view_id="rope"), replace(slots, view_id="slots")
    selection = f"target_cache_tiles.mla_kv_cache.cache_payload.T{tokens}.D{dimension}.S0"
    output = LogicalSelectionView("cache_payload", cache.allocation_id, (tokens, 1, dimension),
        spec.dtype, spec.shape["dtype_bytes"], selection, None)
    return _elementwise(call, source, ((latent, latent), (rope, rope), (slots, slots)),
        (output,), source.input_producers)


def _bind_v4_insert(call, source):
    if ("mathematical_program" in call.kernel_spec.attributes and call.kernel_spec.attributes["mathematical_program"]
            == "vllm_0_30_dspark_partial_context_insert_v1"):
        return None, "dspark_partial_context_native_slot_values_unbound"
    if len(source.inputs) == 4:
        return None, "v4_insert_source_slot_tensor_unavailable"
    if len(source.inputs) != 5 or len(source.outputs) != 2:
        raise ValueError("V4 insert requires actual Q/KV/position/RoPE/slot inputs and Q/cache outputs")
    spec = call.kernel_spec
    rows, heads, padded = (spec.shape[name] for name in ("seq_len", "num_query_heads", "padded_query_heads"))
    query, kv, positions, rotary, slots = source.inputs
    output, cache = source.outputs
    query = _reshape(query, "query", (rows, heads, 512))
    kv = _reshape(kv, "kv", (rows, 1, 512))
    positions, slots = replace(positions, view_id="positions"), replace(slots, view_id="slots")
    if (positions.shape != (rows,) or slots.shape != (rows,)
            or positions.dtype != "int64" or slots.dtype != "int64"
            or output.shape != (rows, padded, 512) or cache.dtype != "uint8" or cache.shape[-1] != 584
            or rotary.dtype != "fp32" or rotary.shape[-1] != 64):
        raise ValueError("V4 insert actual source ports disagree with native typed operands")
    def selection(role, parent, shape, dtype, width):
        identity = f"target_cache_tiles.dsv4_qnorm_rope_kv_insert.{role}.T{rows}.D512.S8.H{heads}.P{padded}"
        return LogicalSelectionView(role, parent.allocation_id, shape, dtype, width, identity, None)
    rope = selection("rope", rotary, (rows, 1, 64), "fp32", 4)
    live = replace(output, view_id="query_live", shape=(rows, heads, 512))
    writes = (live,)
    if heads < padded:
        writes += (replace(output, view_id="query_pad", shape=(rows, padded-heads, 512),
            offset_bytes=output.offset_bytes + heads * output.strides_bytes[1]),)
    writes += tuple(selection(role, cache, (rows, 1, dimension), dtype, width)
        for role, dimension, dtype, width in (("cache_payload", 448, "fp8", 1),
            ("cache_rope", 64, "bf16", 2), ("cache_scale", 8, "uint8", 1)))
    reads = ((query, query), (kv, kv), (positions, positions), (slots, slots), (rope, rotary))
    producers = tuple(source.input_producers[index] for index in (0, 1, 2, 4, 3))
    return _elementwise(call, source, reads, writes, producers)


def _bind_dsa_index_cache(call, source):
    if len(source.inputs) != 2 or len(source.outputs) != 1:
        return None, "dsa_index_cache_source_key_and_slot_ports_unavailable"
    key, slots = source.inputs
    cache, = source.outputs
    spec = call.kernel_spec
    rows, width, scale = (spec.shape[name] for name in ("seq_len", "key_dim", "scale_bytes_per_token"))
    if (key.shape != (rows, width) or key.dtype != "bf16" or slots.shape != (rows,) or slots.dtype != "int64"
            or cache.dtype != "uint8" or cache.shape[-1] != width + scale or scale != 4):
        raise ValueError("DSA cache insert must preserve real BF16 keys, slots and the shared FP8/scale backing")
    key, slots = _reshape(key, "key", (rows, 1, width)), replace(slots, view_id="slots")
    def field(role, shape, dtype, element_bytes):
        return LogicalSelectionView(role, cache.allocation_id, shape, dtype, element_bytes,
            f"target_cache_tiles.dsa_fp8_index_cache.{role}.T{rows}.D{width}.S{scale}", None)
    return _elementwise(call, source, ((key, key), (slots, slots)),
        (field("cache_payload", (rows, 1, width), "fp8", 1), field("cache_scale", (rows, 1, 1), "fp32", 4)),
        source.input_producers)


def bind_attention_memory(call, source):
    """Return typed actual operands or a bounded reason for missing source ABI."""
    spec = call.kernel_spec
    if (source.kind is WorkloadKind.FLASH_ATTN and spec.kind is KernelKind.FLASH_ATTN
            and "fa3_expanded_source_revision" in source.attributes):
        return _bind_expanded_attention(call, source)
    if (source.kind is WorkloadKind.FLASH_ATTN and spec.kind is KernelKind.FLASH_ATTN
            and "source_component" in source.attributes
            and source.attributes["source_component"] == "fa3_mla_split_prefill_call"):
        return _bind_fa3_split_prefill(call, source)
    if (source.kind is WorkloadKind.FLASH_ATTN and spec.kind is KernelKind.FLASH_ATTN
            and "source_component" in source.attributes
            and source.attributes["source_component"] == "fa3_mla_unsplit_prefill_call"):
        return _bind_fa3_unsplit_prefill(call, source)
    if source.kind is WorkloadKind.KV_UPDATE and spec.kind is KernelKind.KV_CACHE:
        if source.variant != spec.variant:
            return None, "cache_source_variant_mapping_unavailable"
        if spec.variant == "paged_kv_cache":
            return _bind_paged_cache(call, source)
        if spec.variant == "mla_kv_cache":
            return _bind_mla_cache(call, source)
        if spec.variant == "dsa_fp8_index_cache":
            return _bind_dsa_index_cache(call, source)
        if spec.variant == "dsv4_qnorm_rope_kv_insert":
            return _bind_v4_insert(call, source)
        reasons = {
            "dsv4_compressor_state_cache": "v4_compressor_source_slots_and_ape_unavailable",
            "dsv4_compressed_state_cache": "v4_finish_source_metadata_and_gamma_unavailable",
            "dsa_fp8_index_cache": "dsa_index_cache_source_key_and_slot_ports_unavailable",
        }
        if spec.variant in reasons:
            return None, reasons[spec.variant]
        return None, "cache_source_native_operand_mapping_unavailable"
    if source.kind is WorkloadKind.FLASH_ATTN and spec.kind is KernelKind.FLASH_ATTN:
        from src.kernel.memory_access_builder import has_mathematical_memory_source
        if has_mathematical_memory_source(spec):
            return _bind_mathematical_attention(call, source)
        if spec.variant in ("prefill", "decode") and source.variant in ("prefill", "decode"):
            return _bind_public_attention(call, source)
        if spec.variant == source.variant == "mla_decode":
            return _bind_mla_decode(call, source)
        return None, "attention_source_native_operand_mapping_unavailable"
    if spec.kind is KernelKind.SPARSE_ATTN_INDEXER:
        return None, "dsa_indexer_source_score_backing_unavailable"
    if spec.kind is KernelKind.SPARSE_ATTN:
        if spec.variant.startswith("dsv4_"):
            return None, "v4_sparse_source_sink_controls_and_native_workspace_unavailable"
        return None, "dsa_sparse_mathematical_memory_source_unavailable"
    if spec.kind is KernelKind.EAGER_ATTN:
        return None, "eager_attention_source_score_and_probability_backings_unavailable"
    return None, "attention_source_family_operand_mapping_unavailable"


def _validate_full_fa3_aot_frame(group,by_id,source_completions,*,context=None):
    """Consume the real AOT writer/copy/clear and per-call reset parents."""
    from math import prod
    main=group[0]
    source=main.memory_source
    attrs=source.attributes
    if context is None:
        context=_FA3SourceCallContext(tuple(by_id.values()))
    if any(call.call_id not in by_id or by_id[call.call_id] is not call
            or call.memory_source is not source or call.rank!=main.rank or call.pool_name!=main.pool_name
            or call.kernel_spec.hardware_id!=main.kernel_spec.hardware_id for call in group):
        raise ValueError("FA3 AOT source stages crossed their actual call frame")
    b,r,n,capacity,dynamic,offsets,holes=attrs["fa3_full_aot_layout"]
    if (any(type(value) is not int or value<=0 for value in (b,r,n,capacity)) or type(dynamic) is not bool
            or b!=main.kernel_spec.shape["batch"] or r!=(b+3)//4*4 or n!=(2+int(dynamic))*r+1 or dynamic!=(b<=992)
            or capacity!=1+4*((max(attrs["fa3_full_max_num_seqs"],attrs["fa3_gqa_max_capture_tokens"])+3)//4*4)):
        raise ValueError("FA3 AOT native packet layout changed")
    expected_offsets=(0,r,2*r,3*r) if dynamic else (0,r,2*r)
    expected_holes=tuple(offset+i for offset in expected_offsets[:-1] for i in range(b,r))
    if offsets!=expected_offsets or holes!=expected_holes:
        raise ValueError("FA3 AOT rounded holes cannot be initialized by padding")
    selected=[]
    for role in ("prepare","copy","clear"):
        identity=attrs["fa3_full_aot_"+role+"_node"]
        if identity not in by_id:
            raise ValueError("FA3 AOT actual helper is missing")
        call=by_id[identity]
        if call.memory_source is None or call.rank!=main.rank or call.pool_name!=main.pool_name:
            raise ValueError("FA3 AOT helper crossed actual source ownership")
        helper=call.memory_source
        if (helper.attributes["fa3_full_aot_component"]!=role or helper.attributes["fa3_full_aot_layout"]!=attrs["fa3_full_aot_layout"]
                or helper.attributes["fa3_full_aot_group"]!=attrs["fa3_full_aot_group"]
                or helper.attributes["fa3_full_aot_hole_validity"]!="uninitialized_copied_not_zero"):
            raise ValueError("FA3 AOT helper validity/layout changed")
        if (helper.attributes["fa3_full_aot_head_shape"] != tuple(main.kernel_spec.shape[field]
                    for field in ("num_query_heads", "num_kv_heads", "head_dim"))
                or helper.attributes["fa3_gqa_operand_dtype"] != main.kernel_spec.dtype):
            raise ValueError("FA3 AOT helper must preserve actual rank-local heads and operand dtype")
        selected.append(call)
    prepare,copy,clear=selected
    p,c,z=(call.memory_source for call in selected)
    if (prepare.kernel_spec.variant!="fa3_prepare_varlen" or prepare.kernel_spec.shape["batch"]!=b
            or prepare.kernel_spec.attributes!={"source_program":"vllm_0_30_fa3_full_aot_prepare",
                "use_dynamic_split":dynamic,"head_swizzle":True,"num_splits_static":None}
            or len(p.inputs)!=2 or tuple(view.shape for view in p.inputs)!=((b+1,),(b,))
            or len(p.outputs)!=len(offsets)):
        raise ValueError("FA3 AOT prepare lost its native input/field ABI")
    if any(view.dtype!="int32" or view.strides_bytes!=(4,) or view.offset_bytes!=0 for view in p.inputs):
        raise ValueError("FA3 AOT caller input dtype or storage layout changed")
    from itertools import accumulate
    if (p.attributes["fa3_full_aot_cu_query"]!=(0,*accumulate(main.kernel_spec.attributes["model_input_q_lens"]))
            or p.attributes["fa3_full_aot_used_key"]!=main.kernel_spec.attributes["model_input_kv_lens"]):
        raise ValueError("FA3 AOT actual input values changed before preparation")
    packet=p.outputs[0].allocation_id
    for i,view in enumerate(p.outputs):
        if (view.allocation_id!=packet or view.dtype!="int32" or view.strides_bytes!=(4,)
                or view.offset_bytes!=offsets[i]*4 or view.shape!=((b,) if i<len(offsets)-1 else (1,))):
            raise ValueError("FA3 AOT prepare cannot write rounded holes")
    if (copy.kernel_spec.kind is not KernelKind.MEMCPY or copy.kernel_spec.transfer.algorithm!="stream_copy"
            or copy.kernel_spec.transfer.bytes!=n*4 or copy.kernel_spec.transfer.direction!="d2d"
            or copy.kernel_spec.transfer.link!="dram" or len(c.inputs)!=1 or len(c.outputs)!=1
            or c.inputs[0].allocation_id!=packet or c.inputs[0].offset_bytes!=0 or c.inputs[0].shape!=(n,)
            or c.inputs[0].dtype!="int32" or c.inputs[0].strides_bytes!=(4,)
            or prepare.call_id not in copy.dependencies):
        raise ValueError("FA3 AOT actual prefix copy changed")
    if len(source.inputs)!=7 or len(source.outputs)!=2 or len(source.input_producers)!=7:
        raise ValueError("FA3 AOT native caller requires seven inputs and two outputs")
    metadata=source.inputs[6]
    if (metadata.allocation_id!=attrs["fa3_full_aot_group"] or metadata.dtype!="int32"
            or metadata.shape!=(n,) or metadata.offset_bytes!=0 or metadata.strides_bytes!=(4,)
            or not _same_fa3_port(metadata,c.outputs[0])
            or not _same_fa3_port(source.inputs[4],p.inputs[0]) or not _same_fa3_port(source.inputs[5],p.inputs[1])
            or source.outputs[1].allocation_id!=metadata.allocation_id
            or source.outputs[1].offset_bytes!=offsets[-1]*4 or source.outputs[1].shape!=(1,)
            or source.outputs[1].dtype!="int32" or source.outputs[1].strides_bytes!=(4,)):
        raise ValueError("FA3 AOT native caller lost copied parents or reset alias")
    allocation=next(item for item in c.allocations if item.allocation_id==metadata.allocation_id)
    if (capacity<n or allocation.size_bytes!=capacity*4 or len(z.inputs)!=0 or len(z.outputs)!=1
            or z.outputs[0].allocation_id!=metadata.allocation_id or z.outputs[0].offset_bytes!=n*4
            or z.outputs[0].shape!=(capacity-n,) or z.outputs[0].dtype!="int32" or z.outputs[0].strides_bytes!=(4,)
            or copy.call_id not in clear.dependencies
            or clear.kernel_spec.attributes!={"operation":"fill","reads":0,"writes":1,
                "mathematical_program":"vllm_0_30_fa3_aot_tail_clear"}):
        raise ValueError("FA3 AOT tail clear cannot initialize the internal holes or overwrite live metadata")
    # A same-buffer reset supplies a direct, source-proven readiness edge.
    producer=source.input_producers[-1]
    if producer not in source_completions or source_completions[producer] not in main.dependencies:
        raise ValueError("FA3 AOT metadata was consumed before its real current writer")
    ready_call=by_id[source_completions[producer]]
    entries=context.attention_entries()
    seen=set()
    while producer!=z.node_id:
        if producer in seen:
            raise ValueError("FA3 AOT semaphore readiness contains a cycle")
        seen.add(producer)
        previous=ready_call.memory_source
        if (previous is None or "fa3_full_aot_layout" not in previous.attributes
                or previous.rank!=source.rank or previous.pool_name!=source.pool_name
                or ready_call.kernel_spec.hardware_id!=main.kernel_spec.hardware_id
                or previous.attributes["fa3_full_aot_layout"]!=attrs["fa3_full_aot_layout"]
                or previous.attributes["fa3_full_aot_group"]!=attrs["fa3_full_aot_group"]
                or len(previous.inputs)!=7 or len(previous.input_producers)!=7
                or len(previous.outputs)!=2 or not _same_fa3_port(previous.outputs[1],source.outputs[1])
                or not _same_fa3_port(previous.inputs[6],metadata)):
            raise ValueError("FA3 AOT readiness requires actual prefix clear or previous same-buffer reset")
        # A cached prefix proves its ancestors, never this consumer's actual
        # rank/group/alias relationship to the prefix's completion writer.
        if producer in context.validated_writers and context.validated_writers[producer] is ready_call:
            break
        if producer not in entries:
            raise ValueError("FA3 AOT semaphore writer lost its attention entry")
        entry=entries[producer]
        if entry is not ready_call and entry.call_id not in ready_call.dependencies:
            raise ValueError("FA3 AOT semaphore writer lost its library completion dependency")
        producer=previous.input_producers[-1]
        if producer not in source_completions or source_completions[producer] not in entry.dependencies:
            raise ValueError("FA3 AOT semaphore readiness lost its actual writer chain")
        ready_call=by_id[source_completions[producer]]
    main_allocations={item.allocation_id:item for item in source.allocations}
    if main_allocations[metadata.allocation_id]!=allocation or allocation.freed_event is not None:
        raise ValueError("FA3 AOT metadata parent lifetime changed")
    prepared_allocations={item.allocation_id:item for item in p.allocations}
    for port in source.inputs[4:6]:
        if (main_allocations[port.allocation_id]!=prepared_allocations[port.allocation_id]
                or main_allocations[port.allocation_id].freed_event is not None):
            raise ValueError("FA3 AOT caller input lifetime changed")
    context.validated_writers[source.node_id]=by_id[source_completions[source.node_id]]
