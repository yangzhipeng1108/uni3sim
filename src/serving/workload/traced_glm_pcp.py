"""vLLM0.30 MLA context/data rows on the existing virtual tensor owners."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PCPQueryChunk:
    request_id: str
    phase: str
    query_offset: int
    rows: int
    local_start: int
    original_row_start: int
    computed_tokens: int
    cache_base: int
    context_tokens: int

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id or self.phase not in ("prefill", "decode"):
            raise ValueError("PCP chunks require an original request identity and phase")
        values = (self.query_offset, self.rows, self.local_start, self.original_row_start,
            self.computed_tokens, self.cache_base, self.context_tokens)
        if any(type(value) is not int or value < 0 for value in values) or not self.rows:
            raise ValueError("PCP chunk extents must be nonnegative integers with positive query rows")
        if self.computed_tokens + self.query_offset + self.rows > self.context_tokens:
            raise ValueError("PCP chunk causal positions must fit its original request context")

    @property
    def absolute_start(self):
        return self.computed_tokens + self.query_offset


@dataclass(frozen=True)
class PCPModelRows:
    """One original batch; chunks refer to its original requests and arena."""
    chunks_by_rank: tuple[tuple[PCPQueryChunk, ...], ...]
    input_ids_by_rank: tuple[tuple[int, ...], ...]
    positions_by_rank: tuple[tuple[int, ...], ...]
    slots_by_rank: tuple[tuple[int, ...], ...]
    is_padding_by_rank: tuple[tuple[bool, ...], ...]
    hidden_restore_indices: tuple[int, ...]
    physical_rows: int
    cache_capacity_tokens: int

    @property
    def real_rows_by_rank(self):
        return tuple(sum(chunk.rows for chunk in chunks) for chunks in self.chunks_by_rank)


def pcp_model_rows(step, input_ids, context_parallel_size):
    """PCPManager._iter_rank_chunks with DCP=1, including short prefills.

    With PCP, prefill takes chunks c and2*C-1-c from2*C ceil-sized chunks. Decode
    is replicated. IDs/positions are padded with0 to the actual maximum M;
    PAD_SLOT=-1 cannot become a real cache update or request.
    CP1 is the ordinary unsharded request batch, used by the shared DP path.
    """
    cp = context_parallel_size
    from src.serving.schedule.schedule_step import ScheduleStep
    if not isinstance(step, ScheduleStep):
        raise TypeError("PCP source rows require a canonical ScheduleStep")
    if type(cp) is not int or cp < 1:
        raise ValueError("MLA source rows require a positive context group size")
    if (not step.requests or not isinstance(input_ids, tuple) or len(input_ids) != step.token_count
            or any(type(token) is not int or token < 0 for token in input_ids)):
        raise ValueError("PCP source rows require the complete nonempty original token batch")
    requests = []
    offset = capacity = 0
    for request in step.requests:
        if (request.phase not in ("decode", "prefill") or request.scheduled_tokens < 1
                or (request.phase == "decode" and request.scheduled_tokens != 1)
                or request.mtp_proposed_tokens or request.mtp_proposer_tokens or request.mtp_verified_tokens):
            raise ValueError("PCP source requires ordinary nonempty prefill/decode requests")
        requests.append((request, offset, capacity))
        offset += request.scheduled_tokens
        capacity += request.computed_tokens + request.scheduled_tokens
    chunks_by_rank, ids_by_rank, positions_by_rank, slots_by_rank = [], [], [], []
    global_rows = [None] * step.token_count
    for rank in range(cp):
        chunks, ids, positions, slots = [], [], [], []
        for request, original, base in requests:
            count = request.scheduled_tokens
            chunk_size = -(-count // (2 * cp)) if request.phase == "prefill" and cp > 1 else count
            indices = (rank, 2 * cp - 1 - rank) if request.phase == "prefill" and cp > 1 else (0,)
            for index in indices:
                start, rows = index * chunk_size, min(chunk_size, count - index * chunk_size)
                if rows <= 0:
                    continue
                local = len(ids)
                chunks.append(PCPQueryChunk(request.request_id, request.phase, start, rows, local,
                    original + start, request.computed_tokens, base, request.computed_tokens + count))
                ids.extend(input_ids[original + start:original + start + rows])
                positions.extend(range(request.computed_tokens + start, request.computed_tokens + start + rows))
                slots.extend(range(base + request.computed_tokens + start, base + request.computed_tokens + start + rows))
                # Decode rows have multiple real copies. Only the canonical
                # CP0 copy belongs to global logits/hidden restore order.
                for row in range(rows):
                    global_row = original + start + row
                    if request.phase == "prefill" or rank == 0:
                        if global_rows[global_row] is not None:
                            raise ValueError("PCP query chunks overlap an original prefill row")
                        global_rows[global_row] = (rank, local + row)
        chunks_by_rank.append(tuple(chunks))
        ids_by_rank.append(tuple(ids))
        positions_by_rank.append(tuple(positions))
        slots_by_rank.append(tuple(slots))
    maximum = max(map(len, ids_by_rank))
    if not maximum or any(row is None for row in global_rows):
        raise ValueError("PCP chunks must cover every original global row exactly once before replica selection")
    padding = tuple((False,) * len(ids) + (True,) * (maximum - len(ids)) for ids in ids_by_rank)
    return PCPModelRows(tuple(chunks_by_rank),
        tuple(ids + (0,) * (maximum - len(ids)) for ids in ids_by_rank),
        tuple(values + (0,) * (maximum - len(values)) for values in positions_by_rank),
        tuple(values + (-1,) * (maximum - len(values)) for values in slots_by_rank),
        padding, tuple(rank * maximum + local for rank, local in global_rows), maximum, capacity)


class GLMPCPContext:
    """One rank's existing context group and original batch row projection."""
    def __init__(self, rows, rank, group, collective, *, dispatch=None, combine=None,
            tensor_group=None, tensor_collective=None, data_group=None,
            data_parallel_rank=0, physical_rows_by_data_rank=None,
            sequence_parallel=False, expert_group=None, analytical_global_expert_parallel=False,
            moe_sequence_parallel=None, dummy_run=False, request_rows_by_data_rank=None):
        from .traced_forward import ForwardCollectiveGroup
        if group is not None and not isinstance(group, ForwardCollectiveGroup):
            raise TypeError("GLM context group must be the actual source collective group")
        cp = 1 if group is None else len(group.traces)
        if type(dummy_run) is not bool or dummy_run and (cp != 1 or rows is not None):
            raise ValueError("GLM dummy context requires CP1 without fabricated request/cache coordinates")
        self.dummy_run = dummy_run
        if ((group is not None and not isinstance(group, ForwardCollectiveGroup)) or type(rank) is not int
                or not 0 <= rank < cp or (collective is not None and collective.operation != "all_gather")
                or (group is not None and collective is None)
                or (rows is not None and (not isinstance(rows, PCPModelRows)
                    or cp != len(rows.chunks_by_rank)))):
            raise ValueError("GLM PCP context requires its complete row layout and actual ordered context group")
        if physical_rows_by_data_rank is None:
            if rows is None or data_group is not None or data_parallel_rank != 0:
                raise ValueError("empty or distributed PCP rows require an explicit complete DP row declaration")
            physical_rows_by_data_rank = (rows.physical_rows,)
        if (not isinstance(physical_rows_by_data_rank, tuple) or not physical_rows_by_data_rank
                or any(type(value) is not int or value < 0 for value in physical_rows_by_data_rank)
                or not sum(physical_rows_by_data_rank) or type(data_parallel_rank) is not int
                or not 0 <= data_parallel_rank < len(physical_rows_by_data_rank)
                or (physical_rows_by_data_rank[data_parallel_rank] < 1 if dummy_run else
                    physical_rows_by_data_rank[data_parallel_rank] != (0 if rows is None else rows.physical_rows))
                or (len(physical_rows_by_data_rank) > 1 and (not isinstance(data_group, ForwardCollectiveGroup)
                    or len(data_group.traces) != len(physical_rows_by_data_rank)))):
            raise ValueError("GLM PCP DP context requires actual rank rows including explicit idle replicas")
        self.rows, self.rank, self.group, self.collective = rows, rank, group, collective
        self.dispatch, self.combine = dispatch, combine
        self.tensor_group, self.tensor_collective = tensor_group, tensor_collective
        self.data_group, self.data_parallel_rank = data_group, data_parallel_rank
        self.physical_rows_by_data_rank = physical_rows_by_data_rank
        self.request_rows_by_data_rank = physical_rows_by_data_rank if request_rows_by_data_rank is None else request_rows_by_data_rank
        if (type(self.request_rows_by_data_rank) is not tuple or len(self.request_rows_by_data_rank) != len(physical_rows_by_data_rank)
                or any(type(real) is not int or real < 0 or real != work and (real != 0 or work < 1)
                    for real, work in zip(self.request_rows_by_data_rank, physical_rows_by_data_rank, strict=True))
                or dummy_run and self.request_rows_by_data_rank[data_parallel_rank] != 0):
            raise ValueError("GLM dummy/request geometry must distinguish actual rows from one-row dummy work")
        if group is None and (data_group is None or len(physical_rows_by_data_rank) < 2):
            raise ValueError("CP1 distributed context requires its actual nontrivial data group")
        if type(sequence_parallel) is not bool or type(analytical_global_expert_parallel) is not bool:
            raise TypeError("GLM PCP sequence-parallel source must be explicit boolean")
        self.sequence_parallel, self.expert_group = sequence_parallel, expert_group
        self.moe_sequence_parallel = sequence_parallel if moe_sequence_parallel is None else moe_sequence_parallel
        if type(self.moe_sequence_parallel) is not bool or sequence_parallel and not self.moe_sequence_parallel:
            raise ValueError("GLM model SP requires MoE SP; each source flag must be boolean")
        self.analytical_global_expert_parallel = analytical_global_expert_parallel
        tp = 1 if tensor_group is None else len(tensor_group.traces)
        physical_rank = group.participants[rank] if group is not None else data_group.participants[data_parallel_rank]
        self.tensor_rank = 0 if tensor_group is None else tensor_group.participants.index(physical_rank)
        self.expert_rank = (data_parallel_rank * cp + rank) * tp + self.tensor_rank
        if expert_group is not None and (not isinstance(expert_group, ForwardCollectiveGroup)
                or len(expert_group.traces) != tp * cp * len(physical_rows_by_data_rank)
                or expert_group.participants[self.expert_rank] != physical_rank):
            raise ValueError("GLM distributed experts require their complete ordered physical group")
        if self.moe_sequence_parallel and (tp < 2 or len(physical_rows_by_data_rank) < 2
                or not isinstance(expert_group, ForwardCollectiveGroup)
                or len(expert_group.traces) != tp * cp * len(physical_rows_by_data_rank)
                or expert_group.participants[self.expert_rank] != physical_rank):
            raise ValueError("GLM PCP SP requires its complete ordered TP*PCP*DP expert group")
        if analytical_global_expert_parallel and (sequence_parallel or tp != 1 or cp == 1
                or len(physical_rows_by_data_rank) < 2 or not isinstance(expert_group, ForwardCollectiveGroup)
                or len(expert_group.traces) != cp * len(physical_rows_by_data_rank)
                or expert_group.participants[self.expert_rank] != physical_rank):
            raise ValueError("GLM PCP analytical EP projection requires its actual TP1 PCP*DP expert group")

    @property
    def physical_rows(self):
        return self.physical_rows_by_data_rank[self.data_parallel_rank]

    @property
    def tensor_parallel_size(self):
        return 1 if self.tensor_group is None else len(self.tensor_group.traces)

    @property
    def context_parallel_size(self):
        return 1 if self.group is None else len(self.group.traces)

    @property
    def shard_rows(self):
        return -(-self.physical_rows // self.tensor_parallel_size) if self.sequence_parallel else self.physical_rows

    @property
    def moe_shard_rows(self):
        return -(-self.physical_rows // self.tensor_parallel_size) if self.moe_sequence_parallel else self.physical_rows

    @property
    def expert_rows_by_rank(self):
        tp, cp = self.tensor_parallel_size, self.context_parallel_size
        return tuple((-(-rows // tp) if self.moe_sequence_parallel else rows)
            for rows in self.physical_rows_by_data_rank for _ in range(cp * tp))

    def _pad_hidden(self, trace, hidden, *, pool, name, for_moe=False):
        from .traced_operators import SymbolicCopy, SymbolicFillZero
        target = (self.moe_shard_rows if for_moe else self.shard_rows) * self.tensor_parallel_size
        if (not (self.moe_sequence_parallel if for_moe else self.sequence_parallel)
                or hidden.shape[0] != self.physical_rows or hidden.dtype != "bf16"):
            raise ValueError("GLM SP padding requires the actual full PCP hidden rows")
        if target == hidden.shape[0]:
            return hidden
        padded = trace.tensor(trace.unique_name(name + ":padded"), (target, hidden.shape[1]), "bf16")
        SymbolicCopy(trace, name + ":copy", pool)(hidden, padded.narrow(0, 0, hidden.shape[0]))
        SymbolicFillZero(trace, name + ":pad-zero", pool)(padded.narrow(0, hidden.shape[0], target - hidden.shape[0]))
        return padded

    def shard_embedding(self, trace, hidden, *, pool):
        padded = self._pad_hidden(trace, hidden, pool=pool, name="sp-embedding")
        return padded.narrow(0, self.tensor_rank * self.shard_rows, self.shard_rows)

    def gather_hidden(self, trace, hidden, *, pool, name):
        if (not self.sequence_parallel or hidden.shape[0] != self.shard_rows
                or trace not in self.tensor_group.traces):
            raise ValueError("GLM SP gather requires this rank's complete local hidden shard")
        gathered = self.tensor_group.call(name, hidden, pool_name=pool, config=self.collective,
            inplace=False, dim=0)
        return gathered if gathered.shape[0] == self.physical_rows else gathered.narrow(0, 0, self.physical_rows)

    def scatter_attention(self, trace, hidden, *, pool, name):
        padded = self._pad_hidden(trace, hidden, pool=pool, name=name)
        return self.tensor_group.call_variable(name + ":reduce-scatter", trace, padded, pool_name=pool,
            config=self.combine, rows_by_rank=(self.shard_rows,) * self.tensor_parallel_size,
            row_shape=(hidden.shape[1],), dtype="bf16", group_dimension="tensor")

    def shard_moe_input(self, trace, hidden, *, pool, name):
        padded = self._pad_hidden(trace, hidden, pool=pool, name=name, for_moe=True)
        return padded.narrow(0, self.tensor_rank * self.moe_shard_rows, self.moe_shard_rows)

    def gather_moe_output(self, trace, hidden, *, pool, name):
        if (not self.moe_sequence_parallel or self.sequence_parallel
                or hidden.shape[0] != self.moe_shard_rows or trace not in self.tensor_group.traces):
            raise ValueError("GLM internal MoE SP restore requires its actual local token shard")
        gathered = self.tensor_group.call(name, hidden, pool_name=pool, config=self.collective,
            inplace=False, dim=0)
        descriptor = {"logical_rows": self.physical_rows, "gathered_rows": gathered.shape[0],
            "tensor_parallel_size": self.tensor_parallel_size}
        writer = trace._writers[gathered.view.allocation_id]
        node = trace._nodes[writer]
        trace._nodes[writer] = node.with_attributes({**node.attributes, "symbolic_glm_moe_sp_restore": descriptor})
        identifiers = tuple(identifier for identifier, submissions in self.tensor_group._submissions.items()
            if trace.rank in submissions and submissions[trace.rank] == writer)
        if len(identifiers) != 1:
            raise ValueError("GLM MoE SP restore must belong to its actual TP gather submission")
        identifier, = identifiers
        # Keep the in-progress collective contract comparable for subsequent
        # ranks; publish its shared descriptor when its last source rank joins.
        if len(self.tensor_group._submissions[identifier]) == len(self.tensor_group.participants):
            contract = self.tensor_group._contracts[identifier]
            self.tensor_group._contracts[identifier] = contract.with_attributes({**contract.attributes,
                "symbolic_glm_moe_sp_restore": descriptor})
        return gathered if gathered.shape[0] == self.physical_rows else gathered.narrow(0, 0, self.physical_rows)

    def attention(self, *args, **kwargs):
        if self.context_parallel_size == 1:
            from .traced_models import DeepseekV32Attention
            return DeepseekV32Attention(*args, **kwargs)
        return GLMPCPAttention(*args, context=self, **kwargs)


class GLMPCPAttention:
    """PCP normalization, gathered cache insertion and original causal queries.

    Source cache updates are aggregated by real logical query interval. PAD
    slots never become writes; small negative-slot metadata checks are omitted.
    These intervals do not claim a separate native launch per source chunk.
    """
    def __init__(self, trace, model, pool, prefix, *, context, layer_index, tp_size, q_lens, kv_lens,
            request_phases, cache, index_cache, topk_buffer, rotary_cache, index_rotary_cache, slot_mapping,
            dummy_index_q, dummy_index_weights, workspace, max_batched_tokens, selection_source,
            prefill_attention_source=None, source_max_model_len=None, num_speculative_tokens=0):
        from .traced_models import DeepseekV32AttentionInput, _source_linear
        from .traced_operators import SymbolicConcat, SymbolicBatchedLinear
        if (not isinstance(context, GLMPCPContext) or model.weight_dtype != "bf16" or num_speculative_tokens
                or selection_source != "declared_causal_topk" or prefill_attention_source != "force_sparse_mqa"):
            raise NotImplementedError("PCP GLM currently requires the explicit BF16, MTP-off causal sparse-MQA source")
        self.trace, self.model, self.pool, self.prefix, self.context = trace, model, pool, prefix, context
        self.layer_index, self.tp_size = layer_index, tp_size
        self.cache, self.index_cache, self.topk_buffer, self.workspace = cache, index_cache, topk_buffer, workspace
        self.source_max_model_len = source_max_model_len
        self.inputs = DeepseekV32AttentionInput(trace, model, pool, prefix, layer_index=layer_index, tp_size=tp_size,
            cache=cache, index_cache=index_cache, topk_buffer=topk_buffer, rotary_cache=rotary_cache,
            index_rotary_cache=index_rotary_cache, slot_mapping=slot_mapping, dummy_index_q=dummy_index_q,
            dummy_index_weights=dummy_index_weights, pcp_outputs_only=True)
        self.heads = self.inputs.local_heads
        self.concat = SymbolicConcat(trace, prefix + ":concat-query", pool)
        self.query_buffer, = workspace.get_simultaneous(
            ((max_batched_tokens, 64, model.kv_lora_rank + model.qk_rope_head_dim), "bf16"))
        self.v_up = SymbolicBatchedLinear(trace, prefix + ":v-up", pool, self.heads, model.kv_lora_rank, model.v_head_dim)
        self.o_proj = _source_linear(trace, model, pool, prefix + ":o-proj",
            self.heads * model.v_head_dim, model.hidden_size, f"model.layers.{layer_index}.self_attn.o_proj")

    def _insert_cache(self, latent, rope, index_key):
        from .traced_operators import SymbolicMLACacheUpdate, SymbolicDSAIndexCacheUpdate
        context, trace, model = self.context, self.trace, self.model
        rows, rank = context.rows, context.rank
        local_chunks = rows.chunks_by_rank[rank]
        decodes = sum(chunk.rows for chunk in local_chunks if chunk.phase == "decode")
        if any(chunk.phase == "decode" and chunk.local_start >= decodes for chunk in local_chunks):
            raise ValueError("PCP cache source requires the original decode-prefix batch order")
        has_prefill = any(chunk.phase == "prefill" for chunks in rows.chunks_by_rank for chunk in chunks)
        gathered = None
        if has_prefill:
            extent = rows.physical_rows - decodes
            if extent <= 0:
                raise ValueError("PCP prefill requires its padded non-decode rows")
            def gather(name, tensor):
                return context.group.call(self.prefix + ":pcp-gather:" + name,
                    tensor.narrow(0, decodes, extent), pool_name=self.pool, config=context.collective,
                    inplace=False, dim=0, group_dimension="context")
            # Indexer K is normalized BF16 here; cache quantization follows
            # this gather. MLA and index-cache insertion are never repeated
            # inside the outputs-only fused norm.
            index_gather = gather("index-k", index_key) if index_key is not None else None
            gathered = (gather("latent", latent), gather("rope", rope), index_gather)
        slot_values = (rows.slots_by_rank[rank][:decodes] + tuple(slot for values in rows.slots_by_rank
            for slot in values[decodes:])) if has_prefill else rows.slots_by_rank[rank]
        slots = trace.tensor(self.prefix + ":pcp-cache-slots", (len(slot_values),), "int64", persistent=True)
        sources = [(chunk, latent, rope, index_key, chunk.local_start, chunk.local_start)
            for chunk in local_chunks if chunk.phase == "decode"]
        if has_prefill:
            extent = rows.physical_rows - decodes
            for cp_rank, chunks in enumerate(rows.chunks_by_rank):
                for chunk in chunks:
                    if chunk.phase == "prefill":
                        offset = cp_rank * extent + chunk.local_start - decodes
                        sources.append((chunk, *gathered, offset, decodes + offset))
        for ordinal, (chunk, key, kpe, index, offset, slot_offset) in enumerate(sources):
            values = slot_values[slot_offset:slot_offset + chunk.rows]
            if values != tuple(range(chunk.cache_base + chunk.absolute_start,
                    chunk.cache_base + chunk.absolute_start + chunk.rows)):
                raise ValueError("PCP cache insertion must preserve each original request's absolute slot interval")
            mapping = slots.narrow(0, slot_offset, chunk.rows)
            if index is not None:
                SymbolicDSAIndexCacheUpdate(trace, self.prefix + f":pcp-index-insert:{ordinal}", self.pool,
                    self.index_cache, mapping, slots=values)(index.narrow(0, offset, chunk.rows))
            SymbolicMLACacheUpdate(trace, self.prefix + f":pcp-mla-insert:{ordinal}", self.pool, self.cache,
                mapping, slots=values, latent_dim=model.kv_lora_rank, rope_dim=model.qk_rope_head_dim)(
                    key.narrow(0, offset, chunk.rows), kpe.narrow(0, offset, chunk.rows).unsqueeze(1))

    def __call__(self, positions, hidden_states):
        from .traced_operators import SymbolicDSAIndexer, SymbolicDSASparseAttention, SymbolicFillZero
        context, trace, model = self.context, self.trace, self.model
        with self.inputs(positions, hidden_states) as prepared:
            _, _, _, ql_nope, index_q, index_weights, mqa_q, latent, rope, output, index_key = prepared
            self._insert_cache(latent, rope, index_key)
            for ordinal, chunk in enumerate(context.rows.chunks_by_rank[context.rank]):
                start, count = chunk.local_start, chunk.rows
                lengths = (chunk.absolute_start + count,)
                prefix = self.prefix + f":pcp-query:{ordinal}"
                topk = self.topk_buffer.narrow(0, start, count)
                if self.inputs.has_indexer:
                    SymbolicDSAIndexer(trace, prefix + ":indexer", self.pool, model, q_lens=(count,), kv_lens=lengths,
                        request_phases=(chunk.phase,), cache=self.index_cache.narrow(0, chunk.cache_base, chunk.context_tokens),
                        topk_buffer=topk, selection_source="declared_causal_topk", prefill_attention_source="force_sparse_mqa",
                        workspace=self.workspace, source_max_model_len=self.source_max_model_len)(
                            index_q.narrow(0, start, count), index_weights.narrow(0, start, count))
                packed = self.query_buffer.narrow(0, start, count).narrow(1, 0, self.heads)
                query = self.concat(ql_nope.narrow(0, start, count), mqa_q.narrow(0, start, count), out=packed)
                sparse = SymbolicDSASparseAttention(trace, prefix + ":sparse", self.pool, model, tp_size=self.tp_size,
                    q_lens=(count,), kv_lens=lengths, request_phases=(chunk.phase,),
                    cache=self.cache.narrow(0, chunk.cache_base, chunk.context_tokens), topk_buffer=topk,
                    indexer_type="full" if self.inputs.has_indexer else "shared", selection_source="declared_causal_topk",
                    prefill_attention_source="force_sparse_mqa", pcp_source=True)
                first = set(trace._nodes)
                sparse_output = sparse(query)
                for identifier in set(trace._nodes) - first:
                    node = trace._nodes[identifier]
                    trace._nodes[identifier] = node.with_attributes({**node.attributes,
                        "pcp_query_origin": {"request_id": chunk.request_id, "absolute_start": chunk.absolute_start,
                            "rows": count, "cache_base": chunk.cache_base, "context_tokens": chunk.context_tokens}})
                self.v_up(sparse_output.transpose(0, 1),
                    out=output.narrow(0, start, count).view_as_shape(count, self.heads, model.v_head_dim).transpose(0, 1))
            actual = context.rows.real_rows_by_rank[context.rank]
            if actual < context.rows.physical_rows:
                SymbolicFillZero(trace, self.prefix + ":pcp-output-padding", self.pool)(
                    output.narrow(0, actual, context.rows.physical_rows - actual))
            return self.o_proj(output)[0]


@dataclass(frozen=True)
class _GLMLegacyInputState:
    pool: object
    model_sha256: str
    step_ids: tuple
    first_token_ids: tuple
    value_origins: tuple
    runner_limits: tuple


def _glm_legacy_input_state(model, step, pool, ids_by_replica, source_program_cache, *, runner_limits):
    """The default legacy dummy reads its own last prepared ID-buffer prefix."""
    from .traced_forward import SourceProgramCache
    if source_program_cache is not None and type(source_program_cache) is not SourceProgramCache:
        raise TypeError("GLM retained inputs require the run-owned SourceProgramCache")
    previous = None
    key = id(pool)
    if source_program_cache is not None and key in source_program_cache.glm_legacy_inputs:
        previous = source_program_cache.glm_legacy_inputs[key]
        if (type(previous) is not _GLMLegacyInputState or previous.pool is not pool
                or previous.model_sha256 != model.config_sha256
                or len(previous.first_token_ids) != len(ids_by_replica)
                or len(previous.value_origins) != len(ids_by_replica)
                or previous.runner_limits != runner_limits
                or step.step_id in previous.step_ids):
            raise ValueError("GLM retained inputs require their own unreplayed model/pool history")
    # CpuGpuBuffer.__init__ uses zeros. _prepare_inputs writes only the
    # current scheduled prefix; _dummy_run reads it without resetting it.
    # The pinned default VLLM_RANDOMIZE_DP_DUMMY_INPUTS is false. An isolated
    # warm graph has no constructor/history declaration and stays unknown.
    retained = ((None,) * len(ids_by_replica) if source_program_cache is None
        else (0,) * len(ids_by_replica)) if previous is None else previous.first_token_ids
    first = tuple(ids[0] if ids else retained[replica] for replica, ids in enumerate(ids_by_replica))
    origins = tuple("provided_source_input" if ids else
        ("unknown_warm_input" if source_program_cache is None else "declared_constructor_zero")
        if previous is None else ("declared_constructor_zero" if previous.value_origins[replica] == "declared_constructor_zero"
            else "retained_source_input") for replica, ids in enumerate(ids_by_replica))
    return _GLMLegacyInputState(pool, model.config_sha256,
        (() if previous is None else previous.step_ids) + (step.step_id,), first, origins, runner_limits)


def trace_glm_pcp_forward(model, step, pool, expert_routing, *, input_ids, positions, slot_mapping,
        cache_capacity_tokens, source_max_model_len, max_batched_tokens, workspace_initial_capacity_bytes,
        selection_source, source_profile, target_sampling_source=None, source_prompt_lengths=None,
        prefill_attention_source=None, include_lm_head=True, indexer_max_logits_mb=512, source_program_cache=None):
    """Full decoder with original request rows and replicated PCP cache.

    Communication uses the existing physical process groups. Independent
    mathematical attention intervals retain each original causal offset;
    provider launch/scratch and metadata preparation remain unbound.
    """
    from .traced_forward import ForwardCollectiveGroup, ForwardTrace
    from .traced_models import DeepseekV32Model, _source_expert_resolutions, _plain_greedy_target
    from .traced_operators import SymbolicGather
    from .workload_graph import WorkloadKind
    from .runtime_inputs import vllm_0_30_source_model_inputs
    config = pool.parallel_plan.config
    tp, cp, dp = config.tensor_parallel_size, config.context_parallel_size, config.data_parallel_size
    sequence_parallel = config.enable_expert_parallel and dp > 1 and tp > 1
    analytical_global_ep = config.enable_expert_parallel and dp > 1 and tp == 1 and cp > 1
    from src.serving.deployment.validated_scope import is_glm_fp8_data_parallel_source_scope
    fp8_target = is_glm_fp8_data_parallel_source_scope(pool, model)
    if fp8_target and any(request.mtp_proposer_mode != "disabled" or request.mtp_proposed_tokens
            or request.mtp_proposer_tokens or request.mtp_verified_tokens or request.mtp_next_proposed_tokens
            for request in step.requests):
        raise NotImplementedError("ordinary GLM FP8 DP source requires MTP-off requests")
    if (model.model_type != "glm_moe_dsa" or model.weight_dtype != "bf16" and not fp8_target or model.compute_dtype != "bf16"
            or pool.hardware_id not in ("nvidia_a800", "nvidia_h200") or cp < 1 or (cp == 1 and dp == 1)
            or config.pipeline_parallel_size != 1
            or config.sequence_parallel_size not in ((1, tp) if sequence_parallel else (1,))
            or model.num_attention_heads % tp or (not config.enable_expert_parallel and model.moe_intermediate_size % (tp * cp * dp))
            or source_profile != f"vllm_0_30_v32_{model.weight_dtype}_sync_explicit_allreduce"
            or selection_source != "declared_causal_topk" or prefill_attention_source != "force_sparse_mqa"):
        raise NotImplementedError("GLM PCP requires explicit BF16 sparse-MQA, PP1 and its source TP/DP/EP/SP relation")
    if target_sampling_source not in (None, "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16"):
        raise NotImplementedError("GLM PCP target requires its explicit plain BF16 source")
    expected_positions = tuple(p for r in step.requests for p in range(r.computed_tokens, r.computed_tokens + r.scheduled_tokens))
    if (step.pool_name != pool.name or positions != expected_positions or not isinstance(input_ids, tuple)
            or len(input_ids) != step.token_count or any(type(value) is not int or not 0 <= value < model.vocab_size for value in input_ids)):
        raise ValueError("GLM PCP must consume original request IDs, token IDs and absolute positions")
    expected_slots, base = [], 0
    for request in step.requests:
        expected_slots.extend(range(base + request.computed_tokens, base + request.computed_tokens + request.scheduled_tokens))
        base += request.computed_tokens + request.scheduled_tokens
    if (slot_mapping != tuple(expected_slots) or type(cache_capacity_tokens) is not int or cache_capacity_tokens < base
            or type(source_max_model_len) is not int or not max(r.computed_tokens + r.scheduled_tokens for r in step.requests)
                <= source_max_model_len <= model.max_position_embeddings):
        raise ValueError("GLM PCP source requires the complete declared logical request cache arenas")
    step, input_ids, input_order = vllm_0_30_source_model_inputs(step, embedding_token_ids=input_ids,
        prompt_lengths=source_prompt_lengths)
    phases_by_request = dict(zip((request.request_id for request in step.requests),
        input_order["model_attention_phases"], strict=True))
    from dataclasses import replace
    requests_by_dp, ids_by_dp = [[] for _ in range(dp)], [[] for _ in range(dp)]
    offset = 0
    for request in step.requests:
        if request.data_parallel_rank >= dp:
            raise ValueError("GLM PCP request DP owner is outside its deployment")
        requests_by_dp[request.data_parallel_rank].append(request)
        ids_by_dp[request.data_parallel_rank].extend(input_ids[offset:offset + request.scheduled_tokens])
        offset += request.scheduled_tokens
    steps_by_dp = tuple(replace(step, requests=tuple(requests), finished_request_ids=(), preempted_request_ids=())
        if requests else None for requests in requests_by_dp)
    input_state = (_glm_legacy_input_state(model, step, pool, ids_by_dp, source_program_cache,
        runner_limits=(source_max_model_len, max_batched_tokens))
        if cp == 1 else None)
    rows_by_dp = tuple(pcp_model_rows(local_step, tuple(ids), cp) if local_step is not None else None
        for local_step, ids in zip(steps_by_dp, ids_by_dp, strict=True))
    request_rows_by_dp = tuple(0 if rows is None else rows.physical_rows for rows in rows_by_dp)
    physical_rows_by_dp = tuple(1 if cp==1 and count==0 else count for count in request_rows_by_dp)
    if (type(max_batched_tokens) is not int or max_batched_tokens < max(physical_rows_by_dp)
            or not isinstance(workspace_initial_capacity_bytes, tuple) or len(workspace_initial_capacity_bytes) != len(pool.ranks)
            or any(type(value) is not int or value < 0 for value in workspace_initial_capacity_bytes)):
        raise ValueError("GLM PCP needs actual per-rank source batch and workspace capacities")
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    by_rank = {trace.rank: trace for trace in traces}
    def groups(dimension):
        unique = tuple(dict.fromkeys(pool.physical_rank_group(dimension, pool.logical_rank_for_physical_rank(rank)) for rank in pool.ranks))
        return tuple(ForwardCollectiveGroup(tuple(by_rank[rank] for rank in ranks),
            namespace=f"{step.step_id}:{dimension}-{index}") for index, ranks in enumerate(unique))
    context_groups = groups("context") if cp > 1 else ()
    tensor_groups = groups("tensor") if tp > 1 else ()
    data_groups = groups("data") if dp > 1 else ()
    root = ForwardCollectiveGroup(traces, namespace=step.step_id, subgroups=(*context_groups, *tensor_groups, *data_groups))
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    expert_rows = cp * sum(-(-rows // tp) * tp if sequence_parallel else rows for rows in physical_rows_by_dp)
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, expert_rows)
    gather = pool.collective_for("all_gather") if cp > 1 or tp > 1 else None
    reduce = pool.collective_for("all_reduce") if tp > 1 else None
    for trace, capacity in zip(traces, workspace_initial_capacity_bytes, strict=True):
        context_group = next(group for group in context_groups if trace in group.traces) if cp > 1 else None
        tensor_group = next(group for group in tensor_groups if trace in group.traces) if tp > 1 else None
        data_group = next(group for group in data_groups if trace in group.traces) if dp > 1 else None
        data_rank = 0 if data_group is None else data_group.participants.index(trace.rank)
        rows, local_step = rows_by_dp[data_rank], steps_by_dp[data_rank]
        context_rank = context_group.participants.index(trace.rank) if context_group is not None else 0
        context = GLMPCPContext(rows, context_rank, context_group, gather,
            dispatch=pool.collective_for("all_gatherv"), combine=pool.collective_for("reduce_scatterv"),
            tensor_group=tensor_group, tensor_collective=reduce, data_group=data_group,
            data_parallel_rank=data_rank, physical_rows_by_data_rank=physical_rows_by_dp,
            sequence_parallel=sequence_parallel, expert_group=root if config.enable_expert_parallel and dp > 1 else None,
            analytical_global_expert_parallel=analytical_global_ep,dummy_run=cp==1 and rows is None,
            request_rows_by_data_rank=request_rows_by_dp)
        if rows is None and cp>1:
            # No request, hidden row, attention cache or sampling state exists
            # on this replica. It still owns expert parameter shards and
            # services the other replicas through zero-row DP submissions.
            from .traced_forward import ForwardWorkspace
            idle_workspace = ForwardWorkspace(trace, "worker.workspace", alignment_bytes=256,
                initial_capacity_bytes=capacity, locked=False)
            for layer in range(model.num_hidden_layers):
                if model.layer_uses_moe(layer):
                    GLMPCPMoE(trace, model, pool.name, f"layer-{layer}:mlp", context=context,
                        layer_index=layer, tp_size=tp, resolution=resolutions[layer],
                        invocation_id=invocations[layer], expert_parallel_size=config.expert_parallel_size,
                        expert_rank=context.expert_rank if config.enable_expert_parallel else 0)(None)
            for identifier, node in tuple(trace._nodes.items()):
                trace._nodes[identifier] = node.with_attributes({**node.attributes,
                    "model_execution_version": "vllm_0_30_0", "model_source_profile": source_profile,
                    "source_projection_only": True, **({"pcp_source": True} if cp > 1 else {"glm_parallel_source": True}),
                **({"source_glm_fp8_target_only": True} if fp8_target else {}),
                    "source_glm_context_parallel_size": cp,
                    "pcp_context_rank": context_rank, "pcp_data_parallel_rank": data_rank,
                    "pcp_physical_rows": 0, "pcp_real_rows": 0,
                    "source_pcp_sequence_parallel": sequence_parallel,
                    **({"source_pcp_expert_communication_scope": "analytical_global_expert_projection",
                        "native_provider_equivalence": False} if analytical_global_ep else {}),
                    "pcp_original_model_input_order": input_order})
            del idle_workspace
            continue
        dummy=rows is None
        from .traced_glm_mtp import GLMLegacyDummyAttentionFactory
        count=physical_rows_by_dp[data_rank]
        network = DeepseekV32Model(trace, model, pool.name, tensor_group, reduce,
            token_ids=((None if input_state.first_token_ids[data_rank] is None else
                (input_state.first_token_ids[data_rank],)) if dummy else rows.input_ids_by_rank[context_rank]),
            q_lens=(1,) if dummy else (tuple(request.scheduled_tokens for request in local_step.requests) if cp == 1 else (rows.physical_rows,)),
            kv_lens=(1,) if dummy else (tuple(request.computed_tokens + request.scheduled_tokens for request in local_step.requests) if cp == 1 else
                (max(r.computed_tokens + r.scheduled_tokens for r in local_step.requests),)),
            request_phases=("decode",) if dummy else (tuple(phases_by_request[request.request_id] for request in local_step.requests) if cp == 1 else ("prefill",)),
            slot_mapping=None if dummy else rows.slots_by_rank[context_rank],
            cache_capacity_tokens=None if dummy else (cache_capacity_tokens if dp == 1 else rows.cache_capacity_tokens),
            source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens,
            resolutions=resolutions, invocation_ids=invocations, workspace_initial_capacity_bytes=capacity,
            selection_source=None if dummy else selection_source, source_profile=source_profile,
            prefill_attention_source=None if dummy else prefill_attention_source,
            model_input_order=None, pcp_context=context, expert_parallel_size=config.expert_parallel_size,
            startup_attention_factory=GLMLegacyDummyAttentionFactory(tokens=1,source_max_model_len=source_max_model_len,
                max_batched_tokens=max_batched_tokens,indexer_max_logits_mb=indexer_max_logits_mb) if dummy else None)
        ids = trace.tensor("input_ids", (count,), "int32")
        positions_tensor = trace.tensor("positions", (count,), "int64")
        hidden = network(ids, positions_tensor)
        if context_group is not None:
            gathered = context_group.call("pcp-hidden-gather", hidden, pool_name=pool.name, config=gather,
                inplace=False, dim=0, group_dimension="context")
            hidden = SymbolicGather(trace, "pcp-hidden-restore", pool.name)(gathered, local_step.token_count,
                indices=rows.hidden_restore_indices)
        if dummy:
            SymbolicGather(trace,"dummy:selected-hidden",pool.name)(hidden,1,indices=(0,))
        elif include_lm_head:
            indices = trace.tensor("logits_indices", (len(local_step.requests),), "int32")
            _plain_greedy_target(trace, model, pool.name, tensor_group, gather, hidden, indices,
                tuple(request.scheduled_tokens for request in local_step.requests),
                sample_logits=target_sampling_source == "vllm_0_30_plain_greedy_bf16",
                cast_logits=target_sampling_source == "vllm_0_30_plain_sampler_bf16")
        for identifier, node in tuple(trace._nodes.items()):
            trace._nodes[identifier] = node.with_attributes({**node.attributes,
                "model_execution_version": "vllm_0_30_0", "model_source_profile": source_profile,
                "source_projection_only": True, **({"pcp_source": True} if cp > 1 else {"glm_parallel_source": True}),
                **({"source_glm_fp8_target_only": True} if fp8_target else {}),
                "source_glm_context_parallel_size": cp,
                **({"source_dummy_run":True,"source_actual_request_count":0} if dummy
                    and node.kind not in (WorkloadKind.COLLECTIVE,WorkloadKind.P2P) else {}),
                "pcp_context_rank": context_rank, "pcp_data_parallel_rank": data_rank,
                "pcp_physical_rows": count,
                "source_pcp_sequence_parallel": sequence_parallel,
                **({"source_pcp_expert_communication_scope": "analytical_global_expert_projection",
                    "native_provider_equivalence": False} if analytical_global_ep else {}),
                "pcp_real_rows": 0 if dummy else rows.real_rows_by_rank[context_rank],
                "pcp_original_model_input_order": input_order})
    # Joined collective nodes originate in the shared contracts rather than
    # the per-rank submission nodes. Preserve the same explicit source scope.
    for group in (root, *root.subgroups):
        for identifier, node in tuple(group._contracts.items()):
            group._contracts[identifier] = node.with_attributes({**node.attributes,
                "model_execution_version": "vllm_0_30_0", "model_source_profile": source_profile,
                "source_projection_only": True, **({"pcp_source": True} if cp > 1 else {"glm_parallel_source": True}),
                **({"source_glm_fp8_target_only": True} if fp8_target else {}),
                "source_glm_context_parallel_size": cp,
                "source_pcp_sequence_parallel": sequence_parallel,
                **({"source_pcp_expert_communication_scope": "analytical_global_expert_projection",
                    "native_provider_equivalence": False} if analytical_global_ep else {})})
    root.source_scope={"model_execution_version":"vllm_0_30_0","source_projection_only":True,
        "source_data_parallel_rows":request_rows_by_dp,"source_model_rows_including_dummy":physical_rows_by_dp,
        "source_expert_physical_rows":expert_rows,"indexer_max_logits_mb":indexer_max_logits_mb,
        "idle_source_scope":"full legacy metadata=None CP1 target; PCP>1 idle remains expert-only",
        "normal_capacity_accepted":False,"actual_provider_bound":False,
        "dummy_input_history_bound":cp == 1 and source_program_cache is not None,
        "dummy_input_randomization":False,
        "dummy_input_randomization_source":"pinned_default_environment_false",
        "dummy_input_runtime_observation_bound":False,
        "retained_input_value_origins":None if input_state is None else input_state.value_origins,
        "retained_first_token_ids":None if input_state is None else input_state.first_token_ids}
    if cp == 1 and source_program_cache is not None:
        root.graph  # Validate joined source before committing run history.
        source_program_cache.glm_legacy_inputs[id(pool)] = input_state
    return root


def glm_fp8_deferred_expert_source_metadata():
    """Pinned0.30 FlashInferCutlassExperts block128 input/prepare facts.

    expects_unquantized_inputs is true for W8A8 block quantization. Naive
    prepare therefore dispatches BF16 and no activation scales; the callee
    owns its quantization. This is a declared source variant, not a selected
    GPU provider or the mapped Triton EP mathematical pricing program.
    """
    return {"source_fp8_expert_implementation": "vllm_0_30_flashinfer_cutlass_block128",
        "source_fp8_defer_input_quant": True, "source_fp8_dispatch_input_dtype": "bf16",
        "source_fp8_weight_block_shape": (128, 128), "source_fp8_activation_group_size": 128,
        "source_fp8_provider_bound": False}


class GLMPCPMoERunner:
    """PCP modular Runner's local shared branch and dispatched routed branch.

    EP-off gathers hidden rows and FP32 gate logits over PCP before routing;
    DP additionally gathers hidden rows and the resulting top-k weights/IDs.
    Experts execute one TP*PCP*DP intermediate shard; sum-scatter reverses
    those two collectives in DP-then-PCP order. EP-on DP1 routes locally, then the
    modular prepare/finalize path gathers hidden rows, FP32 weights and
    INT32 expert IDs over PCP before running complete owned experts.
    Both paths sum-scatter routed outputs to their original PCP row owner.
    Shared experts consume only local rows with ordinary TP weight shards.

    The EP-on source Runner also performs its final TP all-reduce because
    the naive modular output_is_reduced() is false. The decoder's following
    late TP all-reduce remains present in source; these two source costs are
    retained without claiming numerical equivalence of a real provider.

    TP1 with multiple DP and PCP ranks uses an explicit analytical global
    expert dependency projection. It includes required cross-PCP reduction,
    and does not claim equivalence to the naive DP-only modular provider.

    The variable collective projection uses the existing explicit backend
    contract, preserving exact per-peer payloads and producer dependencies;
    it does not assert selection of a real communication provider.
    """

    def __init__(self, trace, model, pool, prefix, *, context, layer_index, tp_size,
                 resolution, invocation_id, expert_parallel_size=1, expert_rank=0):
        from .traced_models import DeepseekV2MLP
        from .traced_operators import SymbolicAdd, SymbolicGroupedRouting, SymbolicLinear, SymbolicRoutedExperts
        from .workload_graph import WorkloadKind

        if (not isinstance(context, GLMPCPContext) or model.model_type != "glm_moe_dsa"
                or model.compute_dtype != "bf16" or model.weight_dtype not in ("bf16", "fp8")
                or model.num_shared_experts != 1 or type(tp_size) is not int or tp_size < 1
                or type(expert_parallel_size) is not int or type(expert_rank) is not int):
            raise NotImplementedError("PCP GLM Runner requires BF16 DP1 and ordinary TP shared experts")
        cp_size, dp_size = context.context_parallel_size, len(context.physical_rows_by_data_rank)
        if model.weight_dtype == "fp8":
            from .model_config import is_glm_fp8_tensor_model_scope
            if (cp_size != 1 or dp_size < 2 or not is_glm_fp8_tensor_model_scope(model, tp_size)
                    or expert_parallel_size == 1 and (context.sequence_parallel or context.moe_sequence_parallel
                        or model.moe_intermediate_size % (tp_size * dp_size * 128))
                    or expert_parallel_size > 1 and (model.num_experts % expert_parallel_size
                        or (tp_size > 1) != context.moe_sequence_parallel)):
                raise NotImplementedError("GLM block-FP8 DP requires whole128 blocks or uniform EP with its actual TP sequence shards")
        if (expert_parallel_size not in (1, tp_size * cp_size * dp_size)
                or not 0 <= expert_rank < expert_parallel_size
                or (expert_parallel_size == 1 and expert_rank != 0)):
            raise ValueError("PCP GLM experts must use EP-off or the complete ordinary TP times PCP expert axis")
        self.fp8_source = glm_fp8_deferred_expert_source_metadata() if model.weight_dtype == "fp8" else {}
        self.enable_expert_parallel = expert_parallel_size > 1
        if (cp_size < 1
                or context.dispatch is None or context.dispatch.operation != "all_gatherv"
                or context.combine is None or context.combine.operation != "reduce_scatterv"
                or (context.rows is not None and cp_size != len(context.rows.chunks_by_rank))):
            raise ValueError("PCP GLM Runner requires its complete context group and explicit dispatch/combine contracts")
        if (not self.enable_expert_parallel and model.moe_intermediate_size % (tp_size * cp_size * dp_size)):
            raise ValueError("PCP GLM routed intermediate dimension must divide ordinary TP times PCP")
        self.trace, self.model, self.pool, self.prefix = trace, model, pool, prefix
        self.context, self.tp_size, self.invocation_id = context, tp_size, invocation_id
        self.rows_by_rank = (context.physical_rows,) * cp_size
        self.data_rows = tuple(cp_size * rows for rows in context.physical_rows_by_data_rank)
        self.cp_dimension = "context" if dp_size > 1 else "data_context"
        self.gate = SymbolicLinear(trace, prefix + ":router", pool, model.hidden_size, model.num_experts,
            kind=WorkloadKind.ROUTER, output_dtype="fp32")
        self.shared_experts = DeepseekV2MLP(trace, model, pool, prefix + ":shared", tp_size=1 if context.moe_sequence_parallel else tp_size,
            intermediate_size=model.moe_intermediate_size * model.num_shared_experts,
            module_path=f"model.layers.{layer_index}.mlp.shared_experts", reduce_results=False)
        self.router = SymbolicGroupedRouting(trace, prefix + ":route", pool, model)
        self.routed_experts = SymbolicRoutedExperts(trace, prefix + ":routed", pool, model,
            tp_size=tp_size * cp_size * dp_size, layer_index=layer_index, resolution=resolution,
            invocation_id=invocation_id, routing_source_stage="dsv32_moe_route",
            expert_parallel_size=expert_parallel_size, expert_rank=expert_rank, pcp_source=cp_size > 1)
        self.add = SymbolicAdd(trace, prefix + ":shared-add", pool)

    def __call__(self, *, hidden_states, router_logits):
        if router_logits is not hidden_states:
            raise ValueError("PCP GLM Runner owns its gate and requires the source hidden-state placeholder")
        if (self.context.moe_sequence_parallel and not self.context.sequence_parallel and hidden_states is not None):
            hidden_states = self.context.shard_moe_input(self.trace, hidden_states,
                pool=self.pool, name=self.prefix + ":moe-sp-input")
        rows = self.context.moe_shard_rows
        if (rows and (hidden_states is None or hidden_states.shape != (rows, self.model.hidden_size) or hidden_states.dtype != "bf16")):
            raise ValueError("PCP GLM Runner requires the complete padded rows of its original context rank")
        if not rows and hidden_states is not None:
            raise ValueError("idle PCP replicas cannot invent a hidden-state payload")
        logits = self.gate(hidden_states)[0] if rows else None
        context = self.context
        if context.expert_group is not None:
            if not self.enable_expert_parallel:
                raise ValueError("GLM PCP sequence-parallel Runner requires its source EP group")
            if rows:
                shared_output = self.shared_experts(hidden_states)
                local_weights, local_ids = self.router(hidden_states, logits)
            else:
                shared_output = local_weights = local_ids = None
            def expert_dispatch(name, tensor, width, dtype):
                return context.expert_group.call_variable(self.prefix + ":ep-dispatch-" + name,
                    self.trace, tensor, pool_name=self.pool, config=context.dispatch,
                    rows_by_rank=context.expert_rows_by_rank, row_shape=(width,), dtype=dtype,
                    group_dimension="expert")
            dispatched = expert_dispatch("hidden", hidden_states, self.model.hidden_size, "bf16")
            weights = expert_dispatch("weights", local_weights, self.model.num_experts_per_tok, "fp32")
            ids = expert_dispatch("ids", local_ids, self.model.num_experts_per_tok, "int32")
            partial = self.routed_experts(dispatched, weights, ids)
            producer = self.trace._writers[partial.view.allocation_id]
            node = self.trace._nodes[producer]
            self.trace._nodes[producer] = node.with_attributes({**node.attributes, **self.fp8_source,
                "source_pcp_physical_rows": self.rows_by_rank,
                "source_pcp_data_rows": self.data_rows,
                "source_pcp_request_rows": context.request_rows_by_data_rank,
                "source_pcp_expert_rows": context.expert_rows_by_rank,
                "source_pcp_sequence_parallel": context.sequence_parallel,
                "source_pcp_row_order": ("data_then_context_then_tensor_lane_then_local_row_including_padding"
                    if context.moe_sequence_parallel else "data_then_context_then_local_row_including_padding"),
                "source_pcp_routing_payload": "hidden_weights_ids",
                "source_tensor_parallel_size": self.tp_size,
                "source_padding_routing": "declared_routing_for_physical_rows_including_padding_not_observed_topk"})
            reduced = context.expert_group.call_variable(self.prefix + ":ep-combine", self.trace, partial,
                pool_name=self.pool, config=context.combine, rows_by_rank=context.expert_rows_by_rank,
                row_shape=(self.model.hidden_size,), dtype="bf16", moe_invocation_id=self.invocation_id,
                group_dimension="expert")
            if not rows:
                return None
            result = self.add(shared_output, reduced)
            if context.moe_sequence_parallel and not context.sequence_parallel:
                result = context.gather_moe_output(self.trace, result,
                    pool=self.pool, name=self.prefix + ":moe-sp-restore")
            return result
        def dispatch(name, tensor, width, dtype):
            return context.group.call_variable(self.prefix + ":pcp-dispatch-" + name, self.trace, tensor,
                pool_name=self.pool, config=context.dispatch, rows_by_rank=self.rows_by_rank,
                row_shape=(width,), dtype=dtype, group_dimension=self.cp_dimension)
        if self.enable_expert_parallel:
            shared_output = self.shared_experts(hidden_states)
            local_weights, local_ids = self.router(hidden_states, logits)
            dispatched = dispatch("hidden", hidden_states, self.model.hidden_size, "bf16")
            weights = dispatch("weights", local_weights, self.model.num_experts_per_tok, "fp32")
            ids = dispatch("ids", local_ids, self.model.num_experts_per_tok, "int32")
        else:
            if rows:
                dispatched = dispatch("hidden", hidden_states, self.model.hidden_size, "bf16") if context.group is not None else hidden_states
                dispatched_logits = dispatch("logits", logits, self.model.num_experts, "fp32") if context.group is not None else logits
                shared_output = self.shared_experts(hidden_states)
                weights, ids = self.router(dispatched, dispatched_logits)
            else:
                dispatched = weights = ids = shared_output = None
            if context.data_group is not None:
                def data_dispatch(name, tensor, width, dtype):
                    return context.data_group.call_variable(self.prefix + ":dp-dispatch-" + name,
                        self.trace, tensor, pool_name=self.pool, config=context.dispatch,
                        rows_by_rank=self.data_rows, row_shape=(width,), dtype=dtype,
                        group_dimension="data")
                dispatched = data_dispatch("hidden", dispatched, self.model.hidden_size, "bf16")
                weights = data_dispatch("weights", weights, self.model.num_experts_per_tok, "fp32")
                ids = data_dispatch("ids", ids, self.model.num_experts_per_tok, "int32")
        partial = self.routed_experts(dispatched, weights, ids)
        producer = self.trace._writers[partial.view.allocation_id]
        node = self.trace._nodes[producer]
        self.trace._nodes[producer] = node.with_attributes({**node.attributes, **self.fp8_source,
            "source_pcp_physical_rows": self.rows_by_rank,
            "source_pcp_row_order": ("data_then_context_then_local_row_including_padding"
                if context.data_group is not None else "context_then_local_row_including_padding"),
            "source_pcp_data_rows": self.data_rows,
            "source_pcp_request_rows": context.request_rows_by_data_rank,
            "source_pcp_routing_payload": "hidden_weights_ids" if self.enable_expert_parallel else "hidden_logits",
            "source_tensor_parallel_size": self.tp_size,
            "source_padding_routing": "declared_routing_for_physical_rows_including_padding_not_observed_topk"})
        if context.data_group is not None:
            partial = context.data_group.call_variable(self.prefix + ":dp-combine", self.trace, partial,
                pool_name=self.pool, config=context.combine, rows_by_rank=self.data_rows,
                row_shape=(self.model.hidden_size,), dtype="bf16", moe_invocation_id=self.invocation_id,
                group_dimension="data")
        if not rows:
            return None
        reduced = (context.group.call_variable(self.prefix + ":pcp-combine", self.trace, partial,
            pool_name=self.pool, config=context.combine, rows_by_rank=self.rows_by_rank,
            row_shape=(self.model.hidden_size,), dtype="bf16", moe_invocation_id=self.invocation_id,
            group_dimension=self.cp_dimension) if context.group is not None else partial)
        result = self.add(shared_output, reduced)
        if self.enable_expert_parallel and self.tp_size > 1:
            if (context.tensor_group is None or len(context.tensor_group.traces) != self.tp_size
                    or context.tensor_collective is None or context.tensor_collective.operation != "all_reduce"):
                raise ValueError("PCP EP Runner final all-reduce requires its complete ordinary TP group")
            result = context.tensor_group.call(self.prefix + ":runner-all-reduce", result,
                pool_name=self.pool, config=context.tensor_collective, inplace=False,
                moe_invocation_id=self.invocation_id)
        return result


class GLMPCPMoE:
    """DeepseekV2MoE row-preserving wrapper around the PCP Runner."""

    def __init__(self, trace, model, pool, prefix, *, context, layer_index, tp_size,
                 resolution, invocation_id, expert_parallel_size=1, expert_rank=0):
        self.experts = GLMPCPMoERunner(trace, model, pool, prefix, context=context,
            layer_index=layer_index, tp_size=tp_size, resolution=resolution,
            invocation_id=invocation_id, expert_parallel_size=expert_parallel_size, expert_rank=expert_rank)

    def __call__(self, hidden_states):
        if hidden_states is None:
            return self.experts(hidden_states=None, router_logits=None)
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view_as_shape(num_tokens, hidden_dim)
        output = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
        return output.view_as_shape(num_tokens, hidden_dim)
