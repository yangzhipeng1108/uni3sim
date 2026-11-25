"""One live Flash request on disjoint virtual GPU/FPGA placements.

The PD strategy explicitly copies full logical cache arenas from one TP
replica. It preserves unknown/unused bytes, not zero-initialized values.
This is a simulator transport contract, not an installed vLLM PD connector.
"""
from dataclasses import replace
from math import prod

from .traced_heterogeneous_moe import (COMPONENT, FlashHeterogeneousDecodeState,
    HeterogeneousMoETrace, _flash_heterogeneous_placement, _tag, trace_flash_heterogeneous_decode_layers)
from .traced_heterogeneous_prefill import trace_flash_heterogeneous_prefill
from .workload_graph import WorkloadEdge, WorkloadGraph, WorkloadKind, WorkloadNode


class FlashHeterogeneousRequest:
    """Reusable prefill/PD/decode virtual program; no scheduling or time stub."""

    def __init__(self, model, *, input_ids, output_tokens, fpga_cards=8, root_seed=0,
            gpu_hardware_id="hygon_bw1100"):
        if type(output_tokens) is not int or output_tokens < 1:
            raise ValueError("heterogeneous request requires at least one output token")
        if not isinstance(input_ids, tuple) or not input_ids:
            raise ValueError("heterogeneous request requires explicit prompt IDs")
        context = len(input_ids) + output_tokens - 1
        if context > model.max_position_embeddings:
            raise ValueError("heterogeneous request exceeds the checkpoint context")
        self.model, self.input_ids, self.output_tokens = model, input_ids, output_tokens
        self.gpu_hardware_id = gpu_hardware_id
        self.fpga_cards, self.root_seed, self.max_context_tokens = fpga_cards, root_seed, context
        self.deployment, self.traces, _, _ = _flash_heterogeneous_placement(
            model, 1, 1, fpga_cards, root_seed, 0, gpu_hardware_id)
        self.prefill_graph = None
        self.prefill_traces = ()
        self.prefill_owners = ()
        self.decode_state = None
        self.emitted_tokens = 0
        self.handoff_bytes = 0
        self.step_records = []

    def prefill(self):
        if self.prefill_graph is not None:
            raise ValueError("a request may prefill only once")
        owners = []
        graph, traces, deployment, outputs = trace_flash_heterogeneous_prefill(self.model,
            input_ids=self.input_ids, root_seed=self.root_seed, rank_start=len(self.traces),
            max_context_tokens=self.max_context_tokens,
            gpu_hardware_id=self.gpu_hardware_id,
            state_observer=lambda trace, layers: owners.append((trace, layers)))
        self.prefill_graph, self.prefill_traces, self.prefill_owners = graph, traces, tuple(owners)
        self.deployment = replace(self.deployment, deployment_id=self.deployment.deployment_id + ":request",
            worker_pools={**self.deployment.worker_pools, **deployment.worker_pools},
            data_parallel_groups={**self.deployment.data_parallel_groups, **deployment.data_parallel_groups})
        self.emitted_tokens = 1
        self._prefill_output = outputs[0]
        self.step_records.append({"phase": "prefill", "query_tokens": len(self.input_ids),
            "kv_length": len(self.input_ids), "generated_tokens": 1})
        return outputs[0]

    def _handoff_tensor(self, tensor, *, role, frontier, destination_rank=0, name_prefix=""):
        source, destination = tensor.trace, self.traces[destination_rank]
        source._check_tensor(tensor)
        source._check_tensor(frontier)
        if not tensor.is_contiguous:
            raise ValueError("full-arena handoff requires a contiguous source object")
        name = destination.qualify("pd:" + name_prefix + role)
        received = destination.tensor(name + ":received", tensor.shape, tensor.dtype,
            persistent=not role.endswith("sampled-token"))
        size = prod(tensor.shape) * tensor.view.element_bytes
        node = WorkloadNode(name, WorkloadKind.PD_KV_TRANSFER, "decode_gpu", {},
            rank=destination.rank, attributes={"source_component": COMPONENT,
                "algorithm": "send_recv", "bytes": size, "bytes_semantics": "payload",
                "direction": "point_to_point", "link": "roce", "topology": "point_to_point",
                "participants": (source.rank, destination.rank), "operation": "send_recv",
                "source_rank": source.rank, "destination_rank": destination.rank,
                "backend": "analytical_roofline_v1", "resource_id": "heterogeneous_roce_switch0",
                "participant_nic_ids": ("prefill-gpu-nic:" + str(source.rank // 2),
                    "gpu-nic:" + str(destination.rank // 2)),
                "handoff_program": "full_logical_arena_replica_copy_v1", "handoff_role": role,
                "payload_shape": tensor.shape, "payload_dtype": tensor.dtype,
                "source_view": tensor.view.view_id, "destination_view": received.view.view_id,
                "symbolic_tensor_inputs": (tensor.view.view_id,), "symbolic_tensor_outputs": (received.view.view_id,),
                "native_provider_bound": False, "source_initialization_inferred": False})
        destination._nodes[name] = node
        dependencies = {source._writers[tensor.view.allocation_id], source._writers[frontier.view.allocation_id]}
        dependencies.discard(None)
        if not dependencies:
            raise ValueError("PD handoff requires the completed source forward frontier")
        destination._edges.extend(WorkloadEdge(writer, name) for writer in sorted(dependencies))
        source._readers[tensor.view.allocation_id].add(name)
        destination._writers[received.view.allocation_id] = name
        source._event("send:" + name)
        destination._event("receive:" + name)
        self.handoff_bytes += size
        return received

    def handoff(self):
        if self.prefill_graph is None or self.decode_state is not None or self.emitted_tokens != 1:
            raise ValueError("PD handoff requires one completed prefill and no previous handoff")
        # KV and compression state are MQA-replicated on the TP ranks;
        # attention/output head shards are not transferred as cache heads.
        source, layers = self.prefill_owners[0]
        if source.rank != len(self.traces) or len(layers) != self.model.num_hidden_layers:
            raise ValueError("PD source must be the selected complete TP8 replica")
        cache_tensors = tuple(tuple((role, self._handoff_tensor(tensor,
            role=f"layer-{layer}:{role}", frontier=self._prefill_output))
            for role, tensor in decoder.attn.cache_tensors) for layer, decoder in enumerate(layers))
        token = self._handoff_tensor(self._prefill_output, role="sampled-token", frontier=self._prefill_output)
        trace = self.traces[0]
        capacity = max(128, -(-self.max_context_tokens // 128) * 128)
        self.decode_state = FlashHeterogeneousDecodeState(self.deployment, self.traces, cache_tensors, token,
            trace.tensor("request:topk", (1, self.model.index_topk), "int32", persistent=True),
            trace.tensor("request:rope", (max(128, self.max_context_tokens), self.model.qk_rope_head_dim), "fp32", persistent=True),
            capacity, self.max_context_tokens, len(self.input_ids), input_graph=self.prefill_graph)
        return token

    def decode(self):
        if self.decode_state is None or self.emitted_tokens >= self.output_tokens:
            raise ValueError("decode requires handed-off state and an unfinished request")
        kv_length = self.decode_state.kv_length + 1
        result = trace_flash_heterogeneous_decode_layers(self.model, fpga_cards=self.fpga_cards,
            root_seed=self.root_seed, layer_count=self.model.num_hidden_layers, kv_length=kv_length,
            include_model_boundary=True, execution_state=self.decode_state,
            gpu_hardware_id=self.gpu_hardware_id)
        self.emitted_tokens += 1
        self.step_records.append({"phase": "decode", "query_tokens": 1, "kv_length": kv_length, "generated_tokens": 1})
        return result.outputs[0]

    @property
    def graph(self):
        if self.prefill_graph is None:
            raise ValueError("request has not executed prefill")
        # Prefill collectives have already been joined by their real group.
        nodes = (*self.prefill_graph.nodes, *(n for t in self.traces for n in t._nodes.values()))
        edges = (*self.prefill_graph.edges, *(e for t in self.traces for e in t._capture_program_edges()))
        graph = WorkloadGraph(tuple(replace(node, attributes={**node.attributes,
            "source_component": COMPONENT, "heterogeneous_request_id": "one"}) for node in nodes), tuple(edges))
        graph.topological_order()
        return graph


class FlashHeterogeneousBatchRequest:
    """Bounded virtual program for fixed-concurrency synchronous requests.

    Each fixed DP-local request group is prefetched on the same TP8
    pool, then copied to its real Decode cache owners. ``prepare`` yields one
    such graph at a time. FPGA microbatches never partition GPU model batches. Every ``decode``
    executes the full model layer-by-layer across all groups;
    the caller prices it before calling ``retire_step``. Retirement requires
    an external whole-step frontier barrier: it does not model a native PD
    scheduler or retain an ever-growing cross-step graph.
    """

    def __init__(self, model, *, prompt_tokens, output_tokens, concurrency,
            microbatches, fpga_cards=8, root_seed=0, gpu_hardware_id="hygon_bw1100"):
        if (type(prompt_tokens) is not int or prompt_tokens < 1
                or type(output_tokens) is not int or output_tokens < 1):
            raise ValueError("fixed request lengths must be positive integers")
        context = prompt_tokens + output_tokens - 1
        if context > model.max_position_embeddings:
            raise ValueError("fixed requests exceed the checkpoint context")
        self.model, self.prompt_tokens, self.output_tokens = model, prompt_tokens, output_tokens
        self.gpu_hardware_id = gpu_hardware_id
        self.concurrency, self.microbatches = concurrency, microbatches
        self.fpga_cards, self.root_seed = fpga_cards, root_seed
        self.max_context_tokens = context
        self.deployment, self.traces, _, self.batches = _flash_heterogeneous_placement(
            model, concurrency, microbatches, fpga_cards, root_seed, 0, gpu_hardware_id)
        self.groups = tuple((0, gpu, tuple(t for t in range(concurrency) if t % 8 == gpu))
            for gpu in range(min(8, concurrency)))
        self.decode_states = []
        self.emitted_tokens = 0
        self.handoff_bytes = 0
        self._preparing = False
        self._pending_prefill_group = False
        self._pending_step = False

    def _graph(self):
        graph = WorkloadGraph(tuple(self._batch_node(n) for t in self.traces for n in t._nodes.values()),
            tuple(e for t in self.traces for e in t._capture_program_edges()))
        graph.topological_order()
        return graph

    def _batch_node(self, node):
        updates = {
            "heterogeneous_request_id": "fixed_batch",
            "heterogeneous_request_concurrency": self.concurrency,
            "heterogeneous_request_microbatches": self.microbatches}
        return (node.with_updates({}, updates, compose=True) if "_module_source_frame" in node.__dict__
            else node.with_attributes({**node.attributes, **updates}))

    def prepare(self):
        """Yield complete TP8 Prefill plus full-arena PD for each local group."""
        if self._preparing or self.emitted_tokens:
            raise ValueError("fixed requests may prepare only once")
        self._preparing = True
        stride = max(128, -(-self.max_context_tokens // 128) * 128)
        for mb, gpu, local in self.groups:
            owners = []
            prefix = f"mb-{mb}:gpu-{gpu}"
            ids = tuple(i % self.model.vocab_size for _ in local for i in range(self.prompt_tokens))
            graph, prefill_traces, placement, outputs = trace_flash_heterogeneous_prefill(
                self.model, input_ids=ids, request_q_lens=(self.prompt_tokens,) * len(local),
                root_seed=self.root_seed, rank_start=len(self.traces),
                max_context_tokens=self.max_context_tokens, namespace="heterogeneous-prefill:" + prefix,
                gpu_hardware_id=self.gpu_hardware_id,
                state_observer=lambda trace, layers: owners.append((trace, layers)))
            self.deployment = replace(self.deployment,
                worker_pools={**self.deployment.worker_pools, **placement.worker_pools},
                data_parallel_groups={**self.deployment.data_parallel_groups, **placement.data_parallel_groups})
            source, layers = owners[0]
            frontier = outputs[0]
            cache_tensors = tuple(tuple((role, FlashHeterogeneousRequest._handoff_tensor(self, tensor,
                role=f"layer-{layer}:{role}", frontier=frontier, destination_rank=gpu,
                name_prefix=prefix + ":"))
                for role, tensor in decoder.attn.cache_tensors) for layer, decoder in enumerate(layers))
            token = FlashHeterogeneousRequest._handoff_tensor(self, frontier,
                role="sampled-token", frontier=frontier, destination_rank=gpu, name_prefix=prefix + ":")
            trace = self.traces[gpu]
            state = FlashHeterogeneousDecodeState(self.deployment, self.traces, cache_tensors, token,
                trace.tensor(prefix + ":topk", (len(local), self.model.index_topk), "int32", persistent=True),
                trace.tensor(prefix + ":rope", (max(128, self.max_context_tokens),
                    self.model.qk_rope_head_dim), "fp32", persistent=True),
                stride * len(local), self.max_context_tokens, self.prompt_tokens,
                input_graph=None, local_token_ids=local, gpu_rank=gpu,
                microbatch_index=mb, routing_concurrency=self.concurrency, cache_stride_tokens=stride)
            self.decode_states.append(state)
            joined = WorkloadGraph(tuple(self._batch_node(n) for n in
                (*graph.nodes, *(n for t in self.traces for n in t._nodes.values()))),
                (*graph.edges, *(e for t in self.traces for e in t._capture_program_edges())))
            joined.topological_order()
            self._pending_prefill_group = True
            yield HeterogeneousMoETrace(joined, (*prefill_traces, *self.traces), self.deployment,
                (), (token,), (local,))
            if self._pending_prefill_group:
                self.retire_prepared_group()
        self.emitted_tokens = 1
        self._preparing = False

    def retire_prepared_group(self):
        """Release a priced/aborted group while preserving its live cache ports.

        Legacy iteration still acknowledges on resume; the common execution
        owner can acknowledge before leaving its source metadata scope.
        """
        if not self._preparing or not self._pending_prefill_group:
            raise ValueError("no prepared Prefill/PD group is waiting for retirement")
        self._retire_traces()
        self._pending_prefill_group = False

    def decode(self):
        if self._preparing or not self.emitted_tokens or self.emitted_tokens >= self.output_tokens:
            raise ValueError("decode requires prepared unfinished fixed requests")
        if self._pending_step:
            raise ValueError("price and retire the preceding complete decode step first")
        trace_flash_heterogeneous_decode_layers(self.model, concurrency=self.concurrency,
            microbatches=self.microbatches, fpga_cards=self.fpga_cards, root_seed=self.root_seed,
            layer_count=self.model.num_hidden_layers, kv_length=self.decode_states[0].kv_length + 1,
            include_model_boundary=True, execution_states=tuple(self.decode_states), defer_graph=True,
            gpu_hardware_id=self.gpu_hardware_id)
        outputs = tuple(state.token for state in self.decode_states)
        for trace in self.traces:
            _tag(trace)
        self.emitted_tokens += 1
        self._pending_step = True
        return HeterogeneousMoETrace(self._graph(), self.traces, self.deployment, (), tuple(outputs), self.batches)

    def _retire_traces(self):
        # Layer closures and temporary symbolic modules must release their
        # handles before dropping dead allocation metadata. Cache/token
        # objects themselves are preserved verbatim across these barriers.
        for state in self.decode_states:
            state.layer_owners = ()
        retained = tuple(t for state in self.decode_states for t in
            (state.token, state.topk, state.rope, *(tensor for layer in state.cache_tensors for _, tensor in layer)))
        for trace in self.traces:
            live_views = {t.view.view_id for t in retained if t.trace is trace}
            live_storage = {name for name, references in trace._references.items() if references > 0}
            trace._nodes.clear()
            trace._edges.clear()
            trace._events.clear()
            trace._views = {name: trace._views[name] for name in trace._views if name in live_views}
            trace._references = {name: count for name, count in trace._references.items() if name in live_storage}
            trace._allocations = {name: replace(trace._allocations[name],
                allocated_event=trace._event("retain:" + name), freed_event=None)
                for name in trace._allocations if name in live_storage}
            trace._writers = {name: None for name in live_storage}
            trace._readers = {name: set() for name in live_storage}
            trace._persistent_ids.intersection_update(live_storage)
            # Shape/view aliases use a rank-global generic counter, and the
            # latest sampled token may retain its alias across the barrier.
            # Keep that counter monotonic while dropping per-step names.
            trace._names = {name: count for name, count in trace._names.items()
                if name == trace.qualify("view")}
            trace._checkpoints.clear()

    def retire_step(self):
        """Release priced transient work after the caller's step barrier."""
        if not self._pending_step:
            raise ValueError("no complete decode step is waiting for retirement")
        self._retire_traces()
        self._pending_step = False
