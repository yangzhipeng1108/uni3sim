"""Explicit per-microbatch decode groups for explicit packed-FP4 switch scenarios.

Source math and cache lifetime owners are the existing complete V4 decoder.
Only linear precision, routed-expert placement and the declared transport
are projected. No prefill, native provider or resident FP8 copy is inferred.
"""
from collections import Counter
from dataclasses import replace
from math import prod

from src.serving.deployment.deployment_config import CollectiveConfig, DeploymentConfig, WorkerPoolConfig
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.parallelism.parallel_config import ParallelConfig
from .traced_flash_gpu_decode import FlashGPUDecode, _requires_ep_coordinator
from .traced_forward import ForwardTrace, ForwardLocalGroup, ForwardCollectiveGroup
from .traced_models import DeepseekV4DecoderLayer
from .traced_operators import (SymbolicEmbedding, SymbolicLinear, SymbolicV4Routing,
    SymbolicSiluAndMul, SymbolicAdd, SymbolicMHC, SymbolicRMSNorm,
    SymbolicLogitsRowSelect, SymbolicGreedy, SymbolicTPLogitsProcessor)
from .traced_heterogeneous_moe import _staging
from .workload_graph import WorkloadGraph, WorkloadKind, WorkloadNode, WorkloadEdge
from .runtime_inputs import seeded_expert_routes

COMPONENT = "flash_fp4_decode_switch_v1"


def _node(trace, name, kind, shape, dtype, variant, attributes):
    pool = "decode_gpu" if trace.rank < trace.scenario_gpu_count else "experts"
    return WorkloadNode(trace.unique_name(name), kind, pool, shape, dtype, variant,
        rank=trace.rank, attributes=attributes)


def _matrix(trace, name, x, width, row_counts):
    node = _node(trace, name, WorkloadKind.LINEAR,
        {"b": 1, "m": x.shape[0], "n": width, "k": x.shape[1]}, "bf16", "gemm",
        {"scenario_matrix_row_counts": row_counts, "weight_dtype": "mxfp4",
            "weight_group_size": 32, "has_bias": False})
    return trace.call(node, (x,), shape=(x.shape[0], width), dtype="bf16")


def _mlp(trace, name, x, width, row_counts, limit):
    up = _matrix(trace, name + ":w13", x, 2 * width, row_counts)
    activated = SymbolicSiluAndMul(trace, name + ":swiglu",
        "decode_gpu" if trace.rank < trace.scenario_gpu_count else "experts", swiglu_limit=limit)(up)
    return _matrix(trace, name + ":w2", activated, x.shape[1], row_counts)


def _gather(trace, name, inputs, rows, hidden, *, controls=()):
    dtype = inputs[0].dtype
    node = _node(trace, name, WorkloadKind.GATHER,
        {"elements": rows * hidden, "rows": rows, "hidden": hidden}, dtype, "gather",
        {"gather_factor": 1, "index_count": rows if trace.rank < trace.scenario_gpu_count else 0, "index_bytes": 4,
            "scattered": trace.rank < trace.scenario_gpu_count, "materialized_output": True})
    return trace.call(node, inputs, shape=(rows, hidden), dtype=dtype, controls=controls)


def _exchange(traces, tensors, output_rows, traffic, *, name, operation):
    """One whole collective, direct endpoint traffic, no ring or hop costs."""
    leader = traces[0]
    participants = tuple(t.rank for t in traces)
    matrix = tuple(tuple(int(value) for value in row) for row in traffic)
    node = _node(leader, name, WorkloadKind.COLLECTIVE, {}, "bf16", None,
        {"scenario_collective_operation": operation, "participants": participants,
            "traffic_matrix_bytes": matrix, "bytes": sum(map(sum, matrix)),
            "scenario_compute_endpoint": False})
    identifier = leader.qualify(node.node_id)
    leader._nodes[identifier] = node.with_attributes(node.attributes, node_id=identifier)
    for tensor in tensors:
        trace = tensor.trace
        writer = trace._writers[tensor.view.allocation_id]
        if writer is None:
            raise ValueError("direct collective requires a produced source")
        leader._edges.append(WorkloadEdge(writer, identifier))
        trace._readers[tensor.view.allocation_id].add(identifier)
    outputs = []
    hidden = tensors[0].shape[1]
    for trace, rows in zip(traces, output_rows, strict=True):
        if rows == 0:
            outputs.append(None)
            continue
        output = trace.tensor(name + ":received", (rows, hidden), "bf16")
        trace._writers[output.view.allocation_id] = identifier
        outputs.append(output)
    return tuple(outputs)


class FlashDecodeScenario(FlashGPUDecode):
    """One real request group; physical resources are shared across groups."""

    def __init__(self, model, *, hardware_id, scheme, microbatch_index=0,
                 initial_context=100000, max_steps=32, root_seed=0, layer_count=None, fpga_cards=16,
                 gpu_count=8, microbatch_size=128):
        if hardware_id not in ("hygon_bw1100", "nvidia_rtx_pro_5500"):
            raise ValueError("FP4 decode requires an explicit supported GPU scenario")
        if scheme not in ("tp8_ep8", "dp8_ep8", "dp8_fpga"):
            raise ValueError("unknown FP4 decode deployment")
        if type(fpga_cards) is not int or fpga_cards not in (16, 32):
            raise ValueError("FPGA cards must be 16 or 32 (two chips per card)")
        if scheme != "dp8_fpga" and fpga_cards != 16:
            raise ValueError("FPGA card override requires dp8_fpga")
        if type(gpu_count) is not int or gpu_count not in (8, 16):
            raise ValueError("GPU count must be 8 or 16")
        if gpu_count != 8 and (scheme != "dp8_fpga" or hardware_id != "nvidia_rtx_pro_5500"):
            raise ValueError("16 GPUs require the declared RTX FPGA direct-switch scenario")
        if type(microbatch_size) is not int or microbatch_size not in (32, 64, 128, 256):
            raise ValueError("microbatch size must be 32, 64, 128 or 256")
        if gpu_count == 8 and microbatch_size != 128:
            raise ValueError("8-GPU compatibility scenarios require microbatch size 128")
        self.gpu_count, self.microbatch_size = gpu_count, microbatch_size
        self.local_rows = microbatch_size // gpu_count
        self.fpga_chips = fpga_cards * 2 if scheme == "dp8_fpga" else 0
        if type(microbatch_index) is not int or microbatch_index not in (0, 1, 2):
            raise ValueError("microbatch group index must be0,1,2")
        if type(initial_context) is not int or initial_context < 128:
            raise ValueError("decode requires explicit already-prefilled context")
        if type(max_steps) is not int or not 1 <= max_steps <= 128:
            raise ValueError("decode window must contain1..128 actual steps")
        if initial_context + max_steps > model.max_position_embeddings:
            raise ValueError("decode exceeds model context limit")
        if (model.model_id, model.num_hidden_layers, model.num_experts) not in (("deepseek-ai/DeepSeek-V4-Flash", 43, 256), ("deepseek-ai/DeepSeek-V4-Pro", 61, 384)):
            raise ValueError("FP4 scenario requires the unchanged Flash geometry")
        self.model, self.hardware_id, self.scheme = model, hardware_id, scheme
        self.gpu_compute_format = "fp4" if hardware_id == "nvidia_rtx_pro_5500" else "fp8"
        self.gpu_direct_link = "pcie" if hardware_id == "nvidia_rtx_pro_5500" else "roce"
        self.concurrency, self.microbatch_index = microbatch_size, microbatch_index
        self.initial_context, self.max_steps, self.root_seed = initial_context, max_steps, root_seed
        self.layer_count = model.num_hidden_layers if layer_count is None else layer_count
        if type(self.layer_count) is not int or not 1 <= self.layer_count <= model.num_hidden_layers:
            raise ValueError("invalid target layer count")
        self.step_index, self.pending, self.caches, self.layers = 0, False, None, ()
        self.stride = -(-(initial_context + max_steps) // 128) * 128
        tp = scheme == "tp8_ep8"
        self.local_request_ids = tuple(tuple(range(self.microbatch_size)) if tp else tuple(range(rank, self.microbatch_size, self.gpu_count))
            for rank in range(self.gpu_count))
        configs = tuple(CollectiveConfig(op, "mesh", "roce", "fully_connected", "flash-direct-switch",
            "analytical_roofline_v1") for op in ("all_reduce", "all_gather"))
        pools = [WorkerPoolConfig("decode_gpu", "colocated", hardware_id,
            parallelism=(ParallelConfig(tensor_parallel_size=self.gpu_count, enable_expert_parallel=True) if tp
                else ParallelConfig(data_parallel_size=self.gpu_count, enable_expert_parallel=True)),
            ranks=tuple(range(self.gpu_count)), collectives=configs if tp else ())]
        if scheme == "dp8_fpga":
            pools.append(WorkerPoolConfig("experts", "colocated", "intel_agmf039r47",
                parallelism=ParallelConfig(data_parallel_size=self.fpga_chips, enable_expert_parallel=True),
                ranks=tuple(range(self.gpu_count, self.gpu_count + self.fpga_chips))))
        self.deployment = build_deployment_plan(DeploymentConfig("colocated", tuple(pools)),
            num_layers=model.num_hidden_layers, num_experts=model.num_experts)
        self.pool = self.deployment.worker_pools["decode_gpu"]
        self.traces = tuple(ForwardTrace(rank=r, namespace=f"flash-fp4:mb-{microbatch_index}:rank-{r}")
            for r in range(self.gpu_count + self.fpga_chips))
        rows = self.microbatch_size if tp else self.local_rows
        for trace in self.traces:
            trace.scenario_gpu_count = self.gpu_count
        self.tokens = tuple(t.tensor("declared-first-input-token", (rows,), "int32") for t in self.traces[:self.gpu_count])
        self.topks = tuple(t.tensor("topk", (rows, model.index_topk), "int32", persistent=True)
            for t in self.traces[:self.gpu_count])
        self.ropes = tuple({kind: t.tensor("rope-" + kind, (self.stride, model.qk_rope_head_dim),
            "fp32", persistent=True) for kind in ("swa", "compressed")} for t in self.traces[:self.gpu_count])

    def step(self):
        if self.pending or self.step_index >= self.max_steps:
            raise ValueError("retire the previous source step before advancing")
        model, index = self.model, self.step_index
        kv, tp = self.initial_context + index + 1, self.scheme == "tp8_ep8"
        rows = self.microbatch_size if tp else self.local_rows
        group = (ForwardCollectiveGroup(self.traces[:self.gpu_count], namespace=f"mb-{self.microbatch_index}:step-{index}",
            participant_nic_ids=tuple("gpu-nic:" + str(r // 2) for r in range(self.gpu_count))) if tp else None)
        config = self.pool.collective_for("all_reduce") if tp else None
        contexts = []
        for rank, trace in enumerate(self.traces[:self.gpu_count]):
            prefix = f"step-{index}"
            ids = self.tokens[rank].view_as_shape(rows)
            positions = trace.tensor(prefix + ":positions", (rows,), "int64")
            hidden = SymbolicEmbedding(trace, prefix + ":embedding", self.pool.name, model.hidden_size,
                "bf16", vocab_size=model.vocab_size)(ids)
            if tp:
                writer = trace._writers[hidden.view.allocation_id]
                trace._nodes[writer] = trace._nodes[writer].with_attributes({**trace._nodes[writer].attributes,
                    "sampled_token_values": "unknown",
                    "embedding_payload_scope": "all_row_upper_bound_per_tp_vocabulary_shard"})
                hidden = group.call(prefix + ":embedding-all-reduce", hidden,
                    pool_name=self.pool.name, config=config, inplace=False)
            layers = tuple(DeepseekV4DecoderLayer(trace, model, self.pool.name, prefix + f":layer-{layer}",
                group if tp else ForwardLocalGroup(trace), config,
                self.ropes[rank]["swa" if model.compress_ratios[layer] <= 1 else "compressed"],
                self.topks[rank], layer_index=layer, q_lens=(1,) * rows, kv_lens=(kv,) * rows,
                request_phases=("decode",) * rows, slot_mapping=tuple(r * self.stride + kv - 1 for r in range(rows)),
                cache_capacity_tokens=self.stride * rows, source_max_model_len=self.stride,
                linear_source_profile="vllm_0_30_logical_group128_fp32", source_use_deep_gemm=None,
                hardware=None, resolution=None, invocation_id=None, indexer_selection_source="declared_causal_topk",
                source_max_batched_tokens=rows, ffn=_requires_ep_coordinator, mhc_workspace_scope="logical_math_only",
                cache_tensors=None if self.caches is None else self.caches[rank][layer])
                for layer in range(self.layer_count))
            contexts.append({"trace": trace, "ids": ids, "positions": positions, "hidden": hidden,
                "layers": layers, "residual": None, "post_mix": None, "res_mix": None})
        histograms = []
        for layer in range(self.layer_count):
            routes = seeded_expert_routes(self.microbatch_size, model.num_experts, model.num_experts_per_tok, source_id="flash_fp4_decode_common_routes",
                root_seed=self.root_seed, moe_invocation_id=f"mb-{self.microbatch_index}:step-{index}:layer-{layer}")
            hist = Counter(e for row in routes for e in row)
            histograms.append(tuple(hist[e] for e in range(model.num_experts)))
            inputs, shared = [], []
            prefix = f"step-{index}:layer-{layer}"
            for rank, ctx in enumerate(contexts):
                trace = ctx["trace"]
                x, ctx["residual"], ctx["post_mix"], ctx["res_mix"] = ctx["layers"][layer].prepare_ffn(
                    ctx["hidden"], ctx["positions"], ctx["post_mix"], ctx["res_mix"], ctx["residual"])
                logits, _ = SymbolicLinear(trace, prefix + ":router", self.pool.name,
                    model.hidden_size, model.num_experts, kind=WorkloadKind.ROUTER, output_dtype="fp32")(x)
                weights, selected = SymbolicV4Routing(trace, prefix + ":route", self.pool.name,
                    model, layer_index=layer)(x, logits, ctx["ids"])
                writer = trace._writers[selected.view.allocation_id]
                trace._nodes[writer] = trace._nodes[writer].with_attributes({**trace._nodes[writer].attributes,
                    "assumed_topk_ids": routes if tp else tuple(routes[t] for t in self.local_request_ids[rank])})
                local = tuple(range(rank, self.microbatch_size, self.gpu_count))
                source_rows = local if tp else tuple(range(self.local_rows))
                fragments = tuple(x.narrow(0, row, 1) for row in source_rows)
                lane = _gather(trace, prefix + ":expert-lane", fragments, self.local_rows, model.hidden_size,
                    controls=(selected,))
                lane_weights = (_gather(trace, prefix + ":expert-lane-weights",
                    tuple(weights.narrow(0, row, 1) for row in source_rows), self.local_rows, 6) if tp else weights)
                inputs.append((lane, local, lane_weights))
                output = _mlp(trace, prefix + ":shared", x, model.moe_intermediate_size // (self.gpu_count if tp else 1),
                    (rows,), model.swiglu_limit)
                if tp:
                    output = group.call(prefix + ":shared-all-reduce", output, pool_name=self.pool.name,
                        config=config, inplace=False)
                shared.append(output)
            routed = self._experts(tuple(inputs), routes, prefix)
            if tp:
                # Each GPU owns its original16 requests after expert combine.
                # All-gather restores the replicated TP attention row order.
                traffic = tuple(tuple(0 if i == j else self.local_rows * model.hidden_size * 2 for j in range(self.gpu_count))
                    for i in range(self.gpu_count))
                routed = _exchange(self.traces[:self.gpu_count], routed, (self.microbatch_size,) * self.gpu_count, traffic,
                    name=prefix + ":restore-tp-rows", operation="all_gather")
                routed = tuple(_gather(trace, prefix + ":restore-natural-request-order",
                    tuple(output.narrow(0, request % self.gpu_count * self.local_rows + request // self.gpu_count, 1) for request in range(self.microbatch_size)),
                    self.microbatch_size, model.hidden_size) for trace, output in zip(self.traces[:self.gpu_count], routed, strict=True))
            for rank, ctx in enumerate(contexts):
                ctx["hidden"] = SymbolicAdd(ctx["trace"], prefix + ":shared-add", self.pool.name)(
                    routed[rank], shared[rank])
        outputs = []
        for rank, ctx in enumerate(contexts):
            trace, hidden, prefix = ctx["trace"], ctx["hidden"], f"step-{index}"
            hidden = SymbolicMHC(trace, prefix + ":final-mhc-post", self.pool.name, model, None,
                variant="mhc_post", source_use_deep_gemm=None, workspace_scope="logical_math_only")(
                    hidden, ctx["residual"], ctx["post_mix"], ctx["res_mix"])
            hidden = SymbolicMHC(trace, prefix + ":hc-head", self.pool.name, model, None,
                variant="hc_head", source_use_deep_gemm=None, workspace_scope="logical_math_only")(hidden)
            hidden = SymbolicRMSNorm(trace, prefix + ":final-norm", self.pool.name, model.hidden_size,
                model.target_semantics.rms_norm_eps)(hidden)
            indices = trace.tensor(prefix + ":logits-indices", (rows,), "int32")
            selected = SymbolicLogitsRowSelect(trace, prefix + ":logits-select", self.pool.name)(
                hidden, indices, selected_indices=tuple(range(rows)))
            if tp:
                logits = SymbolicTPLogitsProcessor(trace, prefix + ":lm-head", self.pool.name,
                    hidden_size=model.hidden_size, vocab_size=model.vocab_size, group=group,
                    config=self.pool.collective_for("all_gather"))(selected)
            else:
                logits, _ = SymbolicLinear(trace, prefix + ":lm-head", self.pool.name, model.hidden_size,
                    model.vocab_size, kind=WorkloadKind.LM_HEAD)(selected)
            outputs.append(SymbolicGreedy(trace, prefix + ":sampler", self.pool.name)(logits))
        self.layers = tuple(ctx["layers"] for ctx in contexts)
        self.caches = tuple(tuple(d.attn.cache_tensors for d in layers) for layers in self.layers)
        self.tokens = tuple(outputs)
        if tp:
            graph = group.graph
        else:
            graph = WorkloadGraph(tuple(n for t in self.traces for n in t._nodes.values()),
                tuple(e for t in self.traces for e in t._capture_program_edges()))
        tagged = tuple(replace(n, attributes={**n.attributes, "source_component": COMPONENT,
            "scenario_scheme": self.scheme, "scenario_gpu_count": self.gpu_count, "scenario_gpu_hardware_id": self.hardware_id,
            "scenario_gpu_compute_format": self.gpu_compute_format,
            "scenario_gpu_direct_link": self.gpu_direct_link,
            "microbatch_index": self.microbatch_index, "decode_step_index": index}) for n in graph.nodes)
        graph = WorkloadGraph(tagged, graph.edges)
        self.pending, self.step_index = True, index + 1
        return graph, {"step_index": index, "kv_length": kv, "position": kv - 1, "q_len": 1,
            "microbatch_index": self.microbatch_index, "global_histograms": tuple(histograms),
            "initial_context_tokens": self.initial_context, "requests": self.microbatch_size,
            "sampler_node_ids": tuple(t._writers[o.view.allocation_id] for t, o in zip(self.traces[:self.gpu_count], outputs)),
            "input_token_source": "declared_first_token" if index == 0 else "retained_previous_sampler",
            "cache_input_source": "declared_prefilled_KV" if index == 0 else "retained_previous_KV"}

    def _experts(self, inputs, routes, prefix):
        model, fpga = self.model, self.scheme == "dp8_fpga"
        owners = self.traces[self.gpu_count:] if fpga else self.traces
        width = model.num_experts // len(owners)
        records = [tuple(sorted((expert, gpu, row) for gpu, (_, local, _) in enumerate(inputs)
            for row, token in enumerate(local) for expert in routes[token]
            if expert // width == owner)) for owner in range(len(owners))]
        traffic = [[0] * len(self.traces) for _ in self.traces]
        packed = []
        for gpu, (x, local, weights) in enumerate(inputs):
            rows = tuple((row, expert) for row, token in enumerate(local) for expert in routes[token])
            packed.append(_gather(x.trace, prefix + ":dispatch-pack",
                tuple(x.narrow(0, row, 1) for row, _ in rows), len(rows), model.hidden_size))
            for row, expert in rows:
                target = (self.gpu_count if fpga else 0) + expert // width
                if target != gpu:
                    traffic[gpu][target] += model.hidden_size * 2
        output_rows = ((0,) * self.gpu_count if fpga else ()) + tuple(len(r) for r in records)
        received = _exchange(self.traces, tuple(packed), output_rows, traffic,
            name=prefix + ":expert-dispatch", operation="routed_dispatch")
        expert_outputs = []
        for owner, trace in enumerate(owners):
            rank = trace.rank
            x = received[rank]
            if x is None:
                continue
            name = prefix + ":expert-group"
            if fpga:
                x = _staging(trace, name + ":receive-ddr", x, "network_to_ddr", output_resource="ddr")
                x = _staging(trace, name + ":read-ddr", x, "ddr_to_buffer", output_resource="buffer")
            x = _gather(trace, name + ":pack-experts", (x,), len(records[owner]), model.hidden_size)
            counts = Counter(record[0] for record in records[owner])
            row_counts = tuple(counts[e] for e in range(owner * width, (owner + 1) * width) if counts[e])
            x = _mlp(trace, name, x, model.moe_intermediate_size, row_counts, model.swiglu_limit)
            if fpga:
                x = _staging(trace, name + ":store-ddr", x, "hbm_to_ddr", output_resource="ddr")
                x = _staging(trace, name + ":send-ddr", x, "ddr_to_network", output_resource="buffer")
            expert_outputs.append(x)
        reverse = tuple(tuple(traffic[j][i] for j in range(len(traffic))) for i in range(len(traffic)))
        returned = _exchange(self.traces, tuple(expert_outputs), (self.local_rows * model.num_experts_per_tok,) * self.gpu_count + ((0,) * self.fpga_chips if fpga else ()),
            reverse, name=prefix + ":expert-combine", operation="routed_combine")
        outputs = []
        for gpu, (x, local, weights) in enumerate(inputs):
            trace = x.trace
            node = _node(trace, prefix + ":weighted-merge", WorkloadKind.VECTOR,
                {"elements": self.local_rows * model.hidden_size, "tokens": self.local_rows, "hidden": model.hidden_size,
                    "experts": model.num_experts, "intermediate": model.moe_intermediate_size, "topk": 6},
                "bf16", "moe_sum", {"work_fraction": 1.0})
            outputs.append(trace.call(node, (returned[gpu], weights), shape=(self.local_rows, model.hidden_size), dtype="bf16"))
        return tuple(outputs)
