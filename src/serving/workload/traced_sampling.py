"""Source storage program for native sampling, using virtual metadata only.

vLLM0.11/0.30 Sampler.forward/sample and TopKTopPSampler.forward_native,
apply_top_k_top_p_pytorch, random_sample and sample_with_exponential_noise.
Explicit profiles select PyTorch-sort, CUDA Triton filtering or the0.11 worker
with FlashInfer0.4.1 top-k-first sampling, no-logprobs, no-penalties and mixed
greedy/random dummy sampler metadata. Native noise is explicit FP32.
It does not establish that a captured runtime selected this source version.
Unrepresented library workspace, compiled fusion/lifetimes and other
FlashInfer source versions remain open.
No random numbers, sorts, arithmetic values or real tensors are computed.
"""

from functools import partial
from math import prod

from .traced_forward import ForwardTrace
from .traced_operators import SymbolicOperator
from .workload_graph import WorkloadKind


class MTPSourceOps(SymbolicOperator):
    """Versioned metadata-only eager Tensor operations for target/proposer.

    Layouts come from real virtual ports, never computed token values.
    The kernel owner prices the explicit useful typed-payload program.
    An explicit source geometry profile can bind contiguous materialization
    descriptors beside that cost model; backend temporaries, instruction
    issue and deployed/provider identity remain independent.
    """

    def __init__(self, trace, name, pool_name, *, model_execution_version="vllm_0_11_0",
                 source_geometry_program=None):
        if model_execution_version not in ("vllm_0_11_0", "vllm_0_27_1", "vllm_0_30_0"):
            raise NotImplementedError("MTP eager primitives require an explicit supported source version")
        super().__init__(trace, name, pool_name)
        self.model_execution_version = model_execution_version
        if source_geometry_program is not None and (
                source_geometry_program != "vllm_0_30_torch_2_13_cuda_source"
                or model_execution_version != "vllm_0_30_0"):
            raise ValueError("MTP source geometry requires its exact declared0.30/Torch2.13 source")
        self.source_geometry_program = source_geometry_program

    def _node(self, operation, *, rows, width, source_rows, dtype, attributes=None):
        return self.node(WorkloadKind.VECTOR, {"elements": rows * width, "rows": rows,
            "width": width, "source_rows": source_rows}, dtype, "vllm_mtp_eager", {
                "source_component": ("vllm_0_30_symbolic_mtp_eager"
                    if self.model_execution_version == "vllm_0_30_0" else "vllm_0_27_1_symbolic_mtp_eager"
                    if self.model_execution_version == "vllm_0_27_1" else "symbolic_mtp_eager"),
                "model_execution_version": self.model_execution_version,
                "operation": operation, "symbolic_workspace_complete": False,
                **({} if attributes is None else attributes)})

    def _call(self, operation, inputs, *, rows, width, source_rows, dtype, shape, output_dtype,
              out=None, attributes=None):
        attributes = {} if attributes is None else dict(attributes)
        if self.source_geometry_program is not None and operation in ("copy", "cast_ids", "cast_fp32", "fill", "argmax"):
            attributes.update({"source_geometry_program": self.source_geometry_program,
                "source_geometry_input_contiguous": tuple(value.is_contiguous for value in inputs),
                "source_geometry_output_contiguous": out is None or out.is_contiguous})
        node = self._node(operation, rows=rows, width=width, source_rows=source_rows,
            dtype=dtype, attributes=attributes)
        return self.trace.call(node, inputs, shape=shape, dtype=output_dtype, out=out)

    def _require_027(self):
        if self.model_execution_version not in ("vllm_0_27_1", "vllm_0_30_0"):
            raise NotImplementedError("padded MTP source primitives require explicit vLLM0.27.1 or0.30.0")

    def cast_fp32(self, x):
        self._require_027()
        if x.dtype != "bf16" or len(x.shape) != 2:
            raise ValueError("MTP logits conversion requires a BF16 matrix")
        return self._call("cast_fp32", (x,), rows=x.shape[0], width=x.shape[1],
            source_rows=x.shape[0], dtype=x.dtype, shape=x.shape, output_dtype="fp32")

    def prepare_next_padded(self, sampled, discard, backup, *, vocab_size,
                            valid_sampled_counts, discarded_rows):
        """utils.py::eagle_prepare_next_token_padded_kernel, no token values."""
        self._require_027()
        if (sampled.dtype != "int32" or len(sampled.shape) != 2
                or discard.dtype != "bool" or discard.shape != (sampled.shape[0],)
                or backup.dtype != "int32" or backup.shape != (sampled.shape[0],)
                or type(vocab_size) is not int or not 0 < vocab_size <= 2**31 - 1):
            raise ValueError("padded next-token preparation requires typed sampled rows, mask, backup and vocabulary")
        rows, width = sampled.shape
        if (not isinstance(valid_sampled_counts, tuple) or len(valid_sampled_counts) != rows
                or any(type(count) is not int or not 0 <= count <= width for count in valid_sampled_counts)
                or not isinstance(discarded_rows, tuple) or len(discarded_rows) != rows
                or any(type(discarded) is not bool for discarded in discarded_rows)
                or any(discarded and count != 0 for discarded, count in zip(discarded_rows, valid_sampled_counts))):
            raise ValueError("padded MTP next-token preparation requires explicit valid counts and discard scenario")
        node = self._node("prepare_next_padded", rows=rows, width=width, source_rows=rows,
            dtype="int32", attributes={"vocab_size": vocab_size,
                "valid_sampled_counts": valid_sampled_counts, "discarded_rows": discarded_rows})
        return self.trace.call_outputs(node, (sampled, discard, backup), outputs=(
            ("next_token_ids", (rows,), "int32"), ("valid_sampled_tokens_count", (rows,), "int32")))

    def prepare_inputs_padded(self, prefix, count, query_start):
        """utils.py::eagle_prepare_inputs_padded_kernel's two INT32 outputs."""
        self._require_027()
        if (prefix.dtype != "int32" or len(prefix.shape) != 1 or count.dtype != "int32"
                or count.shape != prefix.shape or query_start.dtype != "int32"
                or query_start.shape != (prefix.shape[0] + 1,)):
            raise ValueError("padded MTP preparation requires INT32 prefix/count and B+1 query starts")
        rows = prefix.shape[0]
        node = self._node("prepare_inputs_padded", rows=rows, width=1, source_rows=rows, dtype="int32")
        return self.trace.call_outputs(node, (prefix, count, query_start), outputs=(
            ("token_indices_to_sample", (rows,), "int32"), ("num_rejected_tokens", (rows,), "int32")))

    def step_metadata(self, positions, block_table, seq_lens, *, out_positions, out_slots,
                      block_size, max_model_len):
        """utils.py:26–94: write positions/slots and mutate sequence lengths."""
        self._require_027()
        if (positions.dtype != "int64" or len(positions.shape) != 1
                or block_table.dtype != "int32" or len(block_table.shape) != 2
                or block_table.shape[0] != positions.shape[0] or block_table.shape[1] < 1
                or seq_lens.dtype != "int32" or seq_lens.shape != positions.shape
                or any(out.dtype != "int64" or out.shape != positions.shape for out in (out_positions, out_slots))
                or out_positions.view.allocation_id == out_slots.view.allocation_id
                or any(type(value) is not int or value <= 0 for value in (block_size, max_model_len))):
            raise ValueError("MTP step metadata requires typed positions/block table/lengths and distinct INT64 outputs")
        rows = positions.shape[0]
        node = self._node("step_metadata", rows=rows, width=1, source_rows=rows, dtype="int64",
            attributes={"block_table_columns": block_table.shape[1], "block_size": block_size,
                "max_model_len": max_model_len})
        return self.trace.call(node, (positions, block_table, seq_lens), shape=positions.shape, dtype="int64",
            out=out_positions, writes=(out_slots, seq_lens))

    def copy(self, x, out=None):
        if out is not None and (out.shape != x.shape or out.dtype != x.dtype):
            raise ValueError("MTP copy requires matching typed extents")
        if out is not None and out.view.allocation_id == x.view.allocation_id:
            raise NotImplementedError("MTP copy requires independent source/destination backing")
        return self._call("copy", (x,), rows=x.shape[0], width=prod(x.shape[1:]),
            source_rows=x.shape[0], dtype=x.dtype, shape=x.shape, output_dtype=x.dtype, out=out)

    def gather(self, x, indices, *, selected_indices):
        rows = indices.shape[0]
        if (indices.shape != (rows,) or indices.dtype not in ("int32", "int64")
                or not isinstance(selected_indices, tuple) or len(selected_indices) != rows
                or any(type(i) is not int or not 0 <= i < x.shape[0] for i in selected_indices)):
            raise ValueError("MTP gather requires explicit typed in-bounds indices")
        return self._call("gather", (x, indices), rows=rows, width=prod(x.shape[1:]), source_rows=x.shape[0],
            dtype=x.dtype, shape=(rows, *x.shape[1:]), output_dtype=x.dtype,
            attributes={"index_bytes": indices.view.element_bytes, "selected_indices": selected_indices})

    def scatter(self, x, indices, out, *, selected_indices):
        rows = x.shape[0]
        if (x.shape != (rows,) or x.dtype != "int32" or indices.shape != (rows,)
                or indices.dtype not in ("int32", "int64") or out.dtype != "int32" or len(out.shape) != 1):
            raise ValueError("MTP ID scatter requires scalar INT32 IDs and typed row indices")
        if (not isinstance(selected_indices, tuple) or len(selected_indices) != rows
                or any(type(i) is not int or not 0 <= i < out.shape[0] for i in selected_indices)
                or len(set(selected_indices)) != rows):
            raise ValueError("MTP ID scatter requires distinct valid destination indices")
        return self._call("scatter", (x, indices), rows=rows, width=1, source_rows=out.shape[0],
            dtype="int32", shape=out.shape, output_dtype="int32", out=out,
            attributes={"index_bytes": indices.view.element_bytes, "selected_indices": selected_indices})

    def fill(self, out, value=0):
        return self._call("fill", (), rows=out.shape[0], width=prod(out.shape[1:]), source_rows=out.shape[0],
            dtype=out.dtype, shape=out.shape, output_dtype=out.dtype, out=out, attributes={"source_constant": value})

    def unary(self, x, operation="unary", *, inplace=False, expression=None, scalar=None):
        if operation in ("compare", "unary"):
            expressions = (("not_equal", "less", "greater_equal", "greater") if operation == "compare" else
                ("add", "subtract", "clamp_min", "floor_divide", "multiply", "remainder"))
            if expression not in expressions or type(scalar) is not int:
                raise ValueError("MTP integer source expression requires its explicit operation and scalar")
        if operation == "compare":
            output_dtype, shape = "bool", x.shape
        elif operation == "sum":
            if x.dtype != "bool" or len(x.shape) != 2:
                raise ValueError("MTP mask sum requires boolean rows")
            output_dtype, shape = "int64", (x.shape[0],)
        elif operation == "argmax":
            if x.dtype not in ("bf16", "fp32") or len(x.shape) != 2:
                raise ValueError("MTP argmax requires floating-point logit rows")
            output_dtype, shape = "int64", (x.shape[0],)
        elif operation == "cast_ids":
            if x.dtype != "int64" or len(x.shape) != 1:
                raise ValueError("MTP draft-ID cast requires INT64 scalar IDs")
            output_dtype, shape = "int32", x.shape
        elif operation == "unary":
            if x.dtype not in ("int32", "int64"):
                raise ValueError("MTP integer unary operation requires INT32/INT64")
            output_dtype, shape = x.dtype, x.shape
        else:
            raise ValueError("unsupported MTP unary source operation")
        if inplace and (shape != x.shape or output_dtype != x.dtype):
            raise ValueError("MTP dtype/shape change cannot alias its input")
        return self._call(operation, (x,), rows=x.shape[0], width=prod(x.shape[1:]), source_rows=x.shape[0],
            dtype=x.dtype, shape=shape, output_dtype=output_dtype, out=x if inplace else None,
            attributes={} if expression is None else {"source_expression": expression, "source_scalar": scalar})

    def binary(self, x, y, operation="subtract", *, inplace=False):
        if type(inplace) is not bool:
            raise TypeError("MTP binary inplace requires an explicit boolean")
        if x.shape != y.shape:
            raise ValueError("MTP binary inputs require equal shapes")
        if operation in ("promote_subtract", "promote_add"):
            if x.dtype != "int32" or y.dtype != "int64":
                raise ValueError("MTP promoted arithmetic requires INT32 and INT64 operands")
            dtype = "int64"
        elif operation == "logical_and":
            if x.dtype != "bool" or y.dtype != "bool":
                raise ValueError("MTP logical conjunction requires boolean masks")
            dtype = "bool"
        elif operation == "subtract":
            if x.dtype != y.dtype or x.dtype not in ("int32", "int64"):
                raise ValueError("MTP integer binary operation requires matching integer types")
            dtype = x.dtype
        else:
            raise ValueError("unsupported MTP binary source operation")
        if inplace and dtype != x.dtype:
            raise ValueError("MTP promoted binary output cannot alias its input storage")
        return self._call(operation, (x, y), rows=x.shape[0], width=prod(x.shape[1:]), source_rows=x.shape[0],
            dtype=x.dtype, shape=x.shape, output_dtype=dtype, out=x if inplace else None)

    def where(self, mask, x, y=None):
        if mask.shape != x.shape or mask.dtype != "bool" or x.dtype not in ("int32", "int64"):
            raise ValueError("MTP where requires matching boolean condition and integer values")
        promote = y is not None and x.dtype == "int64" and y.dtype == "int32"
        if y is not None and (y.shape != x.shape or (y.dtype != x.dtype and not promote)):
            raise ValueError("MTP where branches must share integer dtype and shape")
        # The two-input form is source torch.where(mask, 0, x).
        return self._call("where_zero" if y is None else "where_promote" if promote else "where",
            (mask, x) if y is None else (mask, x, y),
            rows=x.shape[0], width=prod(x.shape[1:]), source_rows=x.shape[0], dtype=x.dtype,
            shape=x.shape, output_dtype=x.dtype)

    def concat(self, values):
        if not values or any(len(x.shape) != 1 or x.dtype != values[0].dtype for x in values):
            raise ValueError("MTP prefix concat requires matching scalar vectors")
        rows = sum(x.shape[0] for x in values)
        return self._call("concat", values, rows=rows, width=1, source_rows=rows,
            dtype=values[0].dtype, shape=(rows,), output_dtype=values[0].dtype)

    def masked_fill(self, x, mask, *, masked_rows):
        if x.dtype not in ("int32", "int64") or mask.shape != (x.shape[0],) or mask.dtype != "bool":
            raise ValueError("MTP masked fill requires integer rows and boolean row mask")
        return self._call("masked_fill", (mask,), rows=x.shape[0], width=prod(x.shape[1:]), source_rows=x.shape[0],
            dtype=x.dtype, shape=x.shape, output_dtype=x.dtype, out=x, attributes={"masked_rows": masked_rows})

    def row_gather(self, x, indices):
        if len(x.shape) != 2 or x.dtype != "int32" or indices.shape != (x.shape[0],) or indices.dtype != "int64":
            raise ValueError("MTP row gather requires INT32 rows and INT64 per-row indices")
        return self._call("row_gather", (x, indices), rows=x.shape[0], width=x.shape[1], source_rows=x.shape[0],
            dtype=x.dtype, shape=(x.shape[0],), output_dtype=x.dtype)

    def stack(self, values):
        if (not isinstance(values, tuple) or len(values) < 2 or len(values[0].shape) != 1
                or any(v.shape != values[0].shape or v.dtype != "int64" for v in values)):
            raise ValueError("MTP stack requires multiple matching INT64 draft-ID vectors")
        rows = values[0].shape[0]
        return self._call("stack", values, rows=rows, width=len(values), source_rows=rows,
            dtype="int64", shape=(rows, len(values)), output_dtype="int64")

    def rejection(self, out, prefix, draft, target, bonus, *, verified, accepted):
        rows = out.shape[0]
        if (out.dtype != "int32" or len(out.shape) != 2 or prefix.shape != (rows,) or prefix.dtype != "int32"
                or draft.dtype != "int32" or target.dtype != "int64" or draft.shape != target.shape
                or bonus.shape != (rows,) or bonus.dtype != "int32" or draft.shape != (sum(verified),)):
            raise ValueError("MTP rejection requires typed prefix/draft/argmax/bonus ports")
        return self._call("rejection", (prefix, draft, target, bonus), rows=rows, width=out.shape[1],
            source_rows=draft.shape[0], dtype="int32", shape=out.shape, output_dtype="int32", out=out,
            attributes={"verified_tokens": verified, "accepted_tokens": accepted})

    def h2d(self, out, *, host_memory_kind=None):
        if host_memory_kind is None:
            algorithm = "dma"
        elif self.model_execution_version == "vllm_0_27_1" and host_memory_kind == "pinned":
            algorithm = "stream_copy_pinned"
        else:
            raise ValueError("MTP host copy requires the explicit0.27.1 pinned CUDA source scenario")
        node = self.node(WorkloadKind.MEMCPY, {}, None, None, {"algorithm": algorithm, "direction": "h2d",
            "link": "pcie", "bytes": prod(out.shape) * out.view.element_bytes})
        return self.trace.call(node, (), shape=out.shape, dtype=out.dtype, out=out)


def sample_padded_raw_logits(ops, logits, *, prefix, previous_draft, target_indices,
                             bonus_indices, verified, accepted, target_values, bonus_values):
    """Shared0.27/0.30 all-greedy rejection with raw-logit preservation.

    This is the source sampler chain used by both adapted MLA families;
    token values/acceptance remain caller-declared inputs, never computed.
    """
    from .traced_operators import SymbolicGreedy
    bonus_logits = ops.gather(logits, bonus_indices, selected_indices=bonus_values)
    bonus_raw_logits = ops.cast_fp32(bonus_logits)
    bonus_ids = SymbolicGreedy(ops.trace, "target-bonus-sampler", ops.pool_name)(bonus_logits,
        model_execution_version=ops.model_execution_version).view_as_shape(len(verified))
    raw_target_logits = ops.cast_fp32(ops.gather(logits, target_indices, selected_indices=target_values))
    target_logits = ops.copy(raw_target_logits)
    target_argmax = ops.unary(target_logits, "argmax")
    sampled = ops.fill(ops.trace.tensor("target.rejection-output", (len(verified), max(verified) + 1), "int32"), value=-1)
    return ops.rejection(sampled, prefix, previous_draft, target_argmax, bonus_ids,
        verified=verified, accepted=accepted)


def prepare_padded_drafter(ops, sampled, discard, backup, query_start, *, vocab_size,
                           valid_sampled_counts, discarded_rows, prefix):
    """Shared legacy padded next-token and retained-query preparation."""
    next_ids, valid_count = ops.prepare_next_padded(sampled, discard, backup,
        vocab_size=vocab_size, valid_sampled_counts=valid_sampled_counts, discarded_rows=discarded_rows)
    if prefix is None:
        batch = len(valid_sampled_counts)
        sample_indices = ops.unary(query_start.narrow(0, 1, batch), expression="subtract", scalar=1)
        rejected = None
    else:
        sample_indices, rejected = ops.prepare_inputs_padded(prefix, valid_count, query_start)
    return next_ids, sample_indices, rejected


class _VLLMWorkerStorage:
    """Version-bound known device buffers of pure-text CP1/MTP-off workers.

    Adapt GPUModelRunner.__init__ and InputBatch's unconditional sampling
    arrays. CpuGpuBuffer owns one device allocation; its CPU/pinned/NumPy
    storage is not another HBM allocation. Block tables, backend metadata,
    optional multimodal/speculative buffers and library state stay separate.
    No source module import, real tensor, initialization or numerical values.
    """

    def __init__(self, trace, *, tokens, max_requests, inputs_embeds_size, dtype, model_execution_version):
        if not isinstance(trace, ForwardTrace):
            raise TypeError("worker storage requires a ForwardTrace")
        if any(type(value) is not int or value <= 0 for value in (tokens, max_requests, inputs_embeds_size)):
            raise ValueError("worker storage requires explicit positive token/request/embedding capacities")
        if dtype not in ("bf16", "fp16"):
            raise NotImplementedError("worker storage requires a supported 16-bit pure-text source")
        if model_execution_version not in ("vllm_0_11_0", "vllm_0_24_0", "vllm_0_30_0"):
            raise NotImplementedError("worker storage requires its version-bound constructor")
        t, r = tokens, max_requests
        layouts = {
            "input_ids": ((t,), "int32"), "positions": ((t,), "int64"),
            "query_start_loc": ((r + 1,), "int32"), "seq_lens": ((r,), "int32"),
            "inputs_embeds": ((t, inputs_embeds_size), dtype), "is_token_ids": ((t,), "bool")}
        if model_execution_version == "vllm_0_11_0":
            # v0.11 GPUModelRunner.__init__: no0.30 req_indices/query_pos
            # buffers; discard indices and accepted counts are both INT64.
            layouts.update({"discard_request_indices": ((r,), "int64"),
                "num_decode_draft_tokens": ((r,), "int32"),
                "num_accepted_tokens": ((r,), "int64")})
        else:
            #0.24 and0.30 share these16 unconditional device constructors.
            layouts = {
                "input_ids": layouts["input_ids"], "positions": layouts["positions"],
                "query_start_loc": layouts["query_start_loc"], "seq_lens": layouts["seq_lens"],
                "num_computed_tokens": ((r,), "int32"), "prev_num_draft_tokens": ((r,), "int32"),
                "req_indices": ((t,), "int64"), "prev_positions": ((r,), "int64"),
                "num_scheduled_tokens": ((r,), "int32"), "encoder_seq_lens": ((r,), "int32"),
                "inputs_embeds": ((t, inputs_embeds_size), dtype), "is_token_ids": ((t,), "bool"),
                "discard_request_mask": ((r,), "bool"), "num_decode_draft_tokens": ((r,), "int32"),
                "num_accepted_tokens": ((r,), "int32"), "query_pos": ((max(r + 1, t),), "int64")}
        self.buffers = {role: trace.tensor("profile." + role, shape, item_dtype, persistent=True)
            for role, (shape, item_dtype) in layouts.items()}
        self.input_batch_buffers = {role: trace.tensor("profile.input_batch." + role, (r,), item_dtype, persistent=True)
            for role, item_dtype in (("temperature", "fp32"), ("top_p", "fp32"), ("top_k", "int32"),
                ("frequency_penalties", "fp32"), ("presence_penalties", "fp32"), ("repetition_penalties", "fp32"))}
        self.input_ids, self.positions = self.buffers["input_ids"], self.buffers["positions"]


class VLLM030WorkerStorage(_VLLMWorkerStorage):
    """Known v0.30 device constructors; unknown runtime storage stays outside."""
    def __init__(self, trace, *, tokens, max_requests, inputs_embeds_size, dtype):
        super().__init__(trace, tokens=tokens, max_requests=max_requests, inputs_embeds_size=inputs_embeds_size,
            dtype=dtype, model_execution_version="vllm_0_30_0")


class VLLM011WorkerStorage(_VLLMWorkerStorage):
    """v0.11 GPUModelRunner/InputBatch unconditional device constructors.

    Bound to local vLLM0.11 gpu_model_runner.py:345–368 and
    gpu_input_batch.py:150–212. CPU/pinned arrays are not HBM copies.
    Block tables, compiler/library workspace and allocator state are uncovered.
    """
    def __init__(self, trace, *, tokens, max_requests, inputs_embeds_size, dtype):
        super().__init__(trace, tokens=tokens, max_requests=max_requests, inputs_embeds_size=inputs_embeds_size,
            dtype=dtype, model_execution_version="vllm_0_11_0")


class VLLM024WorkerStorage(_VLLMWorkerStorage):
    """v0.24 GPUModelRunner:721–801 device buffers; CPU sides excluded."""
    def __init__(self, trace, *, tokens, max_requests, inputs_embeds_size, dtype):
        super().__init__(trace, tokens=tokens, max_requests=max_requests, inputs_embeds_size=inputs_embeds_size,
            dtype=dtype, model_execution_version="vllm_0_24_0")


def vllm030_worker_storage_bytes(*, tokens, max_requests, inputs_embeds_size, dtype):
    """Known source residency using the same descriptors, not a peak formula.

    Intended for an explicitly before-profile base policy. These descriptors
    alone do not establish complete worker memory or an actual provider.
    """
    trace = ForwardTrace()
    owner = VLLM030WorkerStorage(trace, tokens=tokens, max_requests=max_requests,
        inputs_embeds_size=inputs_embeds_size, dtype=dtype)
    return {a.allocation_id: a.size_bytes for a in trace.allocations}


def vllm011_worker_storage_bytes(*, tokens, max_requests, inputs_embeds_size, dtype):
    """Known0.11 constructor storage only; no complete-runtime assertion."""
    trace = ForwardTrace()
    owner = VLLM011WorkerStorage(trace, tokens=tokens, max_requests=max_requests,
        inputs_embeds_size=inputs_embeds_size, dtype=dtype)
    return {a.allocation_id: a.size_bytes for a in trace.allocations}


class StorageOnlyOps(SymbolicOperator):
    """Allocation/alias contracts for source operators without timing claims."""
    def operation(self, operation, inputs, *, shape, dtype, out=None, writes=()):
        node = self.node(WorkloadKind.VECTOR, {"elements": prod(shape)}, dtype,
            "symbolic_storage_only", {"source_operation": operation})
        return self.trace.call(node, inputs, shape=shape, dtype=dtype, out=out, writes=writes)

    def cast(self, x, dtype):
        if x.dtype == dtype:
            return x
        return self.operation("to", (x,), shape=x.shape, dtype=dtype)

    def sort(self, x):
        node = self.node(WorkloadKind.VECTOR, {"elements": prod(x.shape)}, x.dtype,
            "symbolic_storage_only", {"source_operation": "sort"})
        return self.trace.call_outputs(node, (x,), outputs=(
            ("values", x.shape, x.dtype), ("indices", x.shape, "int64")))

    def inplace(self, operation, x, *inputs):
        return self.operation(operation, (x, *inputs), shape=x.shape, dtype=x.dtype, out=x)

    def unary(self, operation, x, *, dtype=None, shape=None):
        return self.operation(operation, (x,), shape=x.shape if shape is None else shape,
            dtype=x.dtype if dtype is None else dtype)

    def compare(self, operation, x, y):
        # The source profile compares a [B,V] tensor with [B,1].
        if len(x.shape) != 2 or y.shape not in (x.shape, (x.shape[0], 1)):
            raise NotImplementedError("native source comparison requires its explicit row broadcast")
        return self.operation(operation, (x, y), shape=x.shape, dtype="bool")


class TritonSamplingStorageCache:
    """Version-bound0.24/0.30 cache allocations; no kernel execution.

    One cache belongs to one physical rank/trace. Source constants are the
    two 200-entry FP32 lookup tables; only0.30 adds split [64*32,4] plus
    [64,5,32,8,3]. Per-program buffers are keyed by vocabulary and grow to
    min(next_power_of_2(min(SM,B)),SM), retaining their full backing on a
    smaller view. Cache creation and replacement follow Python ownership.
    """
    def __init__(self, trace, *, num_compute_units, model_execution_version="vllm_0_30_0"):
        if not isinstance(trace, ForwardTrace):
            raise TypeError("sampler cache requires ForwardTrace")
        if type(num_compute_units) is not int or num_compute_units <= 0:
            raise ValueError("Triton sampler needs an explicit positive SM count")
        if model_execution_version not in ("vllm_0_24_0", "vllm_0_30_0"):
            raise NotImplementedError("Triton sampler cache requires its0.24 or0.30 source")
        self.model_execution_version = model_execution_version
        self.trace = trace
        self.num_compute_units = num_compute_units
        self._buffers = {}
        self._tables = None
        self._split = None

    def _allocate(self, role, shape):
        return self.trace.tensor(self.trace.unique_name("sampler.triton." + role),
            shape, "fp32", persistent=True)

    def apply(self, ops, logits, k, p):
        for tensor in (logits, k, p):
            self.trace._check_tensor(tensor)
        if ops.trace is not self.trace or logits.dtype != "fp32" or len(logits.shape) != 2:
            raise ValueError("Triton storage cache requires its own FP32 row logits")
        rows, vocab = logits.shape
        if k.shape != (rows,) or p.shape != (rows,):
            raise ValueError("Triton sampling metadata must match its row count")
        if self.model_execution_version == "vllm_0_24_0" and rows < 8:
            #0.24 apply_top_k_top_p uses the in-place PyTorch sort path
            # below eight rows even when HAS_TRITON is true.
            return _apply_top_k_top_p(ops, logits, k, p)
        if logits.view.strides_bytes[-1] != logits.view.element_bytes:
            logits = ops.unary("contiguous_last_axis", logits)
        k_ptr, p_ptr = ops.cast(k, "int32"), ops.cast(p, "fp32")
        programs = min(self.num_compute_units, rows)
        buffer = self._buffers[vocab] if vocab in self._buffers else None
        if buffer is None or buffer.shape[0] < programs:
            size = min(1 << (programs - 1).bit_length(), self.num_compute_units)
            buffer = self._allocate("buffer", (size, vocab))
            self._buffers[vocab] = buffer
        if buffer.shape[0] > programs:
            buffer = self.trace.alias(buffer, self.trace.unique_name("sampler.triton.buffer-view"),
                shape=(programs, vocab), strides_bytes=buffer.view.strides_bytes)
        if self._tables is None:
            self._tables = (self._allocate("normal-cdf-table", (200,)),
                self._allocate("percentile-table", (200,)))
        ops.operation("triton_topk_topp", (logits, buffer, *self._tables, k_ptr, p_ptr),
            shape=logits.shape, dtype=logits.dtype, out=logits, writes=(buffer,))
        # CUDA+p enables the split pipeline for B<=64 even with active k;
        # value-dependent work inside the kernels does not skip allocation.
        if self.model_execution_version == "vllm_0_30_0" and rows <= 64:
            if self._split is None:
                self._split = (self._allocate("split-stats", (64 * 32, 4)),
                    self._allocate("split-parts", (64, 5, 32, 8, 3)))
            stats, parts = self._split
            ops.operation("triton_topp_split_stats", (logits, k_ptr, p_ptr),
                shape=stats.shape, dtype=stats.dtype, out=stats)
            for round_index in range(5):
                ops.operation("triton_topp_split_step_" + str(round_index),
                    (logits, stats, parts, k_ptr, p_ptr), shape=parts.shape, dtype=parts.dtype, out=parts)
            ops.operation("triton_topp_split_mask", (logits, stats, parts, k_ptr, p_ptr),
                shape=logits.shape, dtype=logits.dtype, out=logits)
        return logits

    def clear(self):
        """Drop source cache ownership; external aliases retain their backing."""
        self._buffers.clear()
        self._tables = None
        self._split = None


def _apply_top_k_top_p(ops, logits, k, p, *, inplace_scatter=True):
    """Allocation-faithful adaptation of apply_top_k_top_p_pytorch's path."""
    logits_sort, logits_idx = ops.sort(logits)
    top_k_mask = ops.unary("vocab_minus_k", ops.cast(k, "int64"))
    top_k_mask = ops.operation("gather", (logits_sort, top_k_mask.view_as_shape(k.shape[0], 1)),
        shape=(k.shape[0], 1), dtype=logits.dtype)
    top_k_mask = ops.compare("less_than", logits_sort, top_k_mask)
    ops.inplace("masked_fill", logits_sort, top_k_mask)

    probs_sort = ops.unary("softmax", logits_sort)
    probs_sum = ops.inplace("cumsum_out", probs_sort)
    top_p_mask = ops.compare("less_equal", probs_sum,
        ops.unary("one_minus_p", p.view_as_shape(p.shape[0], 1)))
    ops.inplace("mask_last_column_false", top_p_mask)
    ops.inplace("masked_fill", logits_sort, top_p_mask)
    if inplace_scatter:
        return ops.inplace("scatter", logits, logits_idx, logits_sort)
    # v0.11 apply_top_k_top_p returns logits_sort.scatter(...), retaining
    # the incoming logits in its caller while materializing a fresh result.
    return ops.operation("scatter", (logits_sort, logits_idx, logits_sort), shape=logits.shape, dtype=logits.dtype)


def _random_sample(ops, probs):
    q = ops.unary("empty_exponential_noise", probs)
    ops.inplace("exponential_random", q)
    # Per-request generator overwrite changes values, not the allocation.
    scores = ops.inplace("divide", probs, q)
    return ops.unary("argmax", scores, dtype="int64", shape=(probs.shape[0],))


def _forward_native(ops, logits, k, p, filter_logits):
    logits = filter_logits(ops, logits, k, p)
    probs = ops.unary("softmax", logits)
    return _random_sample(ops, probs)


def _flashinfer_041_top_k_first(ops, logits, k, p):
    """FlashInfer 0.4.1 sampling.py's default logits path, storage only.

    top_k_mask_logits converts the row thresholds to INT32 inside its
    helper, returns a fresh FP32 masked matrix, then torch.softmax creates
    another matrix while masked_logits remains live. The top-p custom op
    returns INT32 rows. This path does not call flashinfer.softmax or its
    cached workspace. Generator state tensors are CPU objects, not HBM.
    This explicitly selected source version does not identify historical
    logs which merely name FlashInfer without a library version.
    """
    def top_k_mask_logits():
        thresholds = ops.cast(k, "int32")
        return ops.operation("flashinfer_0_4_1_top_k_mask_logits", (logits, thresholds),
            shape=logits.shape, dtype=logits.dtype)

    masked_logits = top_k_mask_logits()
    probs = ops.unary("softmax", masked_logits)
    return ops.operation("flashinfer_0_4_1_top_p_sampling", (probs, p),
        shape=(logits.shape[0],), dtype="int32")


def _sample(ops, logits, temperature, k, p, random_sampler):
    rows = logits.shape[0]
    greedy_sampled = ops.unary("argmax", logits, dtype="int64", shape=(rows,))

    def apply_temperature():
        condition = ops.unary("temperature_less_than_eps", temperature, dtype="bool")
        adjusted = ops.operation("where_temperature", (condition, temperature), shape=temperature.shape, dtype=temperature.dtype)
        return ops.inplace("divide_by_temperature", logits, adjusted.view_as_shape(rows, 1))

    logits = apply_temperature()
    random_sampled = random_sampler(ops, logits, k, p)
    condition = ops.unary("temperature_less_than_eps", temperature, dtype="bool")
    return ops.inplace("where_reuse_greedy", greedy_sampled, condition, random_sampled)


def _sampling_filter(trace, source_profile, triton_cache):
    if source_profile in ("vllm_0_11_pytorch_sort_mixed_no_logprobs_fp32_noise",
            "vllm_0_30_pytorch_sort_mixed_no_logprobs_fp32_noise"):
        if triton_cache is not None:
            raise ValueError("PyTorch sort does not consume Triton cache state")
        filter_logits = (_apply_top_k_top_p if source_profile == "vllm_0_30_pytorch_sort_mixed_no_logprobs_fp32_noise"
            else partial(_apply_top_k_top_p, inplace_scatter=False))
    elif source_profile == "vllm_0_30_triton_mixed_no_logprobs_fp32_noise":
        if (not isinstance(triton_cache, TritonSamplingStorageCache) or triton_cache.trace is not trace
                or triton_cache.model_execution_version != "vllm_0_30_0"):
            raise ValueError("Triton sampler requires explicitly owned cache state")
        filter_logits = triton_cache.apply
    elif source_profile == "vllm_0_24_native_triton_or_sort_mixed_no_logprobs_fp32_noise":
        if (not isinstance(triton_cache, TritonSamplingStorageCache) or triton_cache.trace is not trace
                or triton_cache.model_execution_version != "vllm_0_24_0"):
            raise ValueError("0.24 native sampler requires its explicitly versioned cold cache")
        filter_logits = triton_cache.apply
    else:
        raise NotImplementedError("native sampler storage requires a fully declared source profile")
    return filter_logits


def _sampling_program(trace, source_profile, triton_cache):
    if source_profile == "vllm_0_11_flashinfer_0_4_1_top_k_first_mixed_no_logprobs":
        if triton_cache is not None:
            raise ValueError("FlashInfer top-k-first does not consume Triton cache state")
        return _flashinfer_041_top_k_first
    return partial(_forward_native, filter_logits=_sampling_filter(trace, source_profile, triton_cache))


def native_sampler_storage_forward(trace, logits, *, source_profile, temperature, top_k, top_p,
                                   pool_name="main", triton_cache=None):
    """Return a symbolic INT32 sampler output while tracing real source locals.

    All sampler metadata is explicit. The profile covers nonempty top-k/top-p,
    FP32 noise and the raw-logprobs/no-requested-logprobs branch. No output
    token value is computed. Storage-only nodes must not be timed by lowering.
    """
    random_sampler = _sampling_program(trace, source_profile, triton_cache)
    if (logits.dtype not in ("bf16", "fp16") or len(logits.shape) != 2
            or logits.dtype == "fp16" and source_profile not in (
                "vllm_0_11_pytorch_sort_mixed_no_logprobs_fp32_noise",
                "vllm_0_30_pytorch_sort_mixed_no_logprobs_fp32_noise")):
        raise NotImplementedError("native sampler storage expects supported 16-bit row logits; FP16 requires PyTorch-sort")
    rows = logits.shape[0]
    for value in (temperature, top_k, top_p):
        trace._check_tensor(value)
        if value.shape != (rows,):
            raise ValueError("sampling metadata must have one value per row")
    if temperature.dtype != "fp32" or top_p.dtype != "fp32" or top_k.dtype != "int64":
        raise ValueError("dummy sampler source requires FP32 temperature/top-p and INT64 top-k")
    ops = StorageOnlyOps(trace, "native-sampler", pool_name)
    # Both bound Sampler.forward versions use logits.to(torch.float32).
    # The model head's 16-bit dtype is retained by its caller; this cast
    # allocates fresh FP32 logits, with all later sort/noise storage FP32.
    logits = ops.cast(logits, "fp32")
    sampled = _sample(ops, logits, temperature, top_k, top_p, random_sampler)
    sampled = ops.cast(sampled, "int64")
    sampled = ops.cast(sampled, "int32")
    return sampled.view_as_shape(rows, 1)


def worker_profile_storage_forward(trace, model_forward, compute_logits, *,
                                   input_ids, positions, query_lengths, sampler_profile, pool_name="main",
                                   triton_cache=None, drafter_dummy_run=None, model_execution_version="vllm_0_30_0"):
    """Adapt version-bound profile_run/_dummy_run/_dummy_sampler_run scopes.

    Model and head are supplied symbolic callables; there is no model-specific
    peak formula. The caller retains both full and gathered hidden states
    while the sampler replaces its own hidden input with rand_like. First
    sampler output remains live across0.24/0.30's second native warmup;0.11 only
    calls its sampler once. No random
    values, logits, probabilities or tokens are generated.
    """
    if model_execution_version not in ("vllm_0_11_0", "vllm_0_24_0", "vllm_0_30_0"):
        raise NotImplementedError("worker profile requires its version-bound source")
    if ((model_execution_version == "vllm_0_11_0")
            != (sampler_profile in ("vllm_0_11_pytorch_sort_mixed_no_logprobs_fp32_noise",
                "vllm_0_11_flashinfer_0_4_1_top_k_first_mixed_no_logprobs"))):
        raise ValueError("worker and sampler source versions disagree")
    if ((model_execution_version == "vllm_0_24_0")
            != (sampler_profile == "vllm_0_24_native_triton_or_sort_mixed_no_logprobs_fp32_noise")):
        raise ValueError("0.24 worker and sampler source versions disagree")
    if model_execution_version in ("vllm_0_11_0", "vllm_0_24_0") and drafter_dummy_run is not None:
        raise NotImplementedError("vLLM0.11/0.24 startup storage currently requires MTP-off")
    _sampling_program(trace, sampler_profile, triton_cache)
    if drafter_dummy_run is not None and not callable(drafter_dummy_run):
        raise TypeError("worker profile drafter must be an explicit metadata-only callable")
    trace._check_tensor(input_ids)
    trace._check_tensor(positions)
    if (not isinstance(query_lengths, tuple) or not query_lengths
            or any(type(n) is not int or n <= 0 for n in query_lengths)
            or input_ids.shape != positions.shape or input_ids.shape != (sum(query_lengths),)):
        raise ValueError("worker profile requires explicit positive query lengths covering its input rows")
    ops = StorageOnlyOps(trace, "profile.worker", pool_name)

    def dummy_run():
        hidden_states = model_forward(input_ids, positions)
        if drafter_dummy_run is not None:
            # GPUModelRunner._dummy_run calls the drafter before selecting
            # last_hidden_states. The target output stays live during every
            # sequential draft forward; no acceptance or draft sampling runs.
            drafter_dummy_run()
        # gpu_model_runner uses cumsum - 1 as an INT64 device gather index.
        indices = trace.tensor("profile.logit_indices", (len(query_lengths),), "int64")
        selected = ops.operation("profile_last_hidden_gather", (hidden_states, indices),
            shape=(len(query_lengths), *hidden_states.shape[1:]), dtype=hidden_states.dtype)
        return hidden_states, selected

    def dummy_sampler_run(hidden_states):
        hidden_states = ops.unary("rand_like", hidden_states)
        logits = compute_logits(hidden_states)
        rows = logits.shape[0]
        temperature = trace.tensor("profile.temperature", (rows,), "fp32")
        top_p = trace.tensor("profile.top_p", (rows,), "fp32")
        top_k = trace.tensor("profile.top_k", (rows,), "int64")
        # no_penalties=True skips the operations; dummy metadata still owns
        # these three device arrays throughout its version's sampling calls.
        penalty_metadata = tuple(trace.tensor("profile." + name, (rows,), "fp32")
            for name in ("frequency", "presence", "repetition"))
        sampler_output = native_sampler_storage_forward(trace, logits,
            source_profile=sampler_profile, temperature=temperature, top_k=top_k, top_p=top_p,
            pool_name=pool_name, triton_cache=triton_cache)
        if model_execution_version in ("vllm_0_24_0", "vllm_0_30_0"):
            native_sampler_storage_forward(trace, logits,
                source_profile=sampler_profile, temperature=temperature, top_k=top_k, top_p=top_p,
                pool_name=pool_name, triton_cache=triton_cache)
        return sampler_output

    hidden_states, last_hidden_states = dummy_run()
    output = dummy_sampler_run(last_hidden_states)
    # profile_run synchronizes before releasing these source-owned objects.
    # The metadata-only program describes that synchronous retirement order.
    del hidden_states, output


def ascend018_profile_storage_forward(trace, model_forward, compute_logits, *,
        input_ids, positions, query_lengths, pool_name="main"):
    """NPUModelRunner0.18 profile_run's head-only dummy sampler.

    The Ascend override returns (hidden_states, hidden_states) from
    _dummy_run. _dummy_sampler_run gathers the last rows and only calls
    compute_logits: no rand_like, logits cast, filtering or random sampler.
    Source indices originate in CPU NumPy metadata; the logical device
    index port is represented, without claiming its runtime copy lowering.
    Other preallocated worker objects, compilation and library workspace
    are separate from this caller-storage scope.
    """
    trace._check_tensor(input_ids)
    trace._check_tensor(positions)
    if (not isinstance(query_lengths, tuple) or not query_lengths
            or any(type(n) is not int or n <= 0 for n in query_lengths)
            or input_ids.shape != positions.shape or input_ids.shape != (sum(query_lengths),)
            or input_ids.dtype != "int32" or positions.dtype != "int64"):
        raise ValueError("Ascend profile needs typed source ports and complete positive query lengths")
    ops = StorageOnlyOps(trace, "profile.worker", pool_name)
    hidden_states = model_forward(input_ids, positions)
    last_hidden_states = hidden_states

    def dummy_head(hidden_states):
        indices = trace.tensor("profile.logit_indices", (len(query_lengths),), "int64")
        hidden_states = ops.operation("ascend018_profile_last_hidden_gather", (hidden_states, indices),
            shape=(len(query_lengths), *hidden_states.shape[1:]), dtype=hidden_states.dtype)
        return compute_logits(hidden_states)

    output = dummy_head(last_hidden_states)
    # profile_run synchronizes before deleting its returned full hidden and
    # output. last_hidden_states still aliases the original model result.
    del hidden_states, output


def _mtp_retained_last_rows(requests):
    """vLLM 0.30 selects retained rows in padded queries.

    Source: vLLM 0.30 autoregressive/speculator.py query_len -= num_rejected,
    last_token_index = query_start + query_len - 1. Query starts retain the
    full scheduled extents; accepted output does not resize the draft input.
    Legacy eagle_prepare_inputs_padded_kernel uses the same query-end minus
    rejected-row rule and produces INT32 token_indices_to_sample.
    """
    rows = []
    query_start = 0
    for request in requests:
        retained = request.scheduled_tokens - request.mtp_rejected_tokens
        if retained <= 0:
            raise ValueError("MTP row selection requires a retained anchor row")
        rows.append(query_start + retained - 1)
        query_start += request.scheduled_tokens
    return tuple(rows)



def glm_gpu_v2_draft_update_work(*, num_requests, hidden_size, step, proposal_depth,
                               sampled_token_dtype, advance_draft_positions):
    """Logical roles of vLLM 0.30 _update_draft_inputs_kernel, without timing.

    InputBuffers stores INT32 ids/seq_lens and INT64 positions; speculator
    stores INT64 draft tokens, sampling positions and current step. Sampled
    token dtype is an explicit source-scenario operand, not inferred from a
    missing sampler binding. The source supplies grid=N and BLOCK_SIZE=1024
    but no num_warps, so this is not a complete LaunchConfig.
    """
    if any(type(value) is not int for value in (num_requests, hidden_size, step, proposal_depth)):
        raise TypeError("draft update dimensions/step/depth must be integers")
    if num_requests <= 0 or hidden_size <= 0 or not 1 <= step < proposal_depth:
        raise ValueError("draft update requires live requests and a uniform proposal step")
    if sampled_token_dtype not in ("int32", "int64") or type(advance_draft_positions) is not bool:
        raise ValueError("draft update requires explicit sampled-token dtype and advance flag")
    token_bytes = {"int32": 4, "int64": 8}[sampled_token_dtype]

    def role(name, dtype, elements, element_bytes, access):
        return {"role": name, "dtype": dtype, "elements": elements,
                "bytes": elements * element_bytes, "access": access}

    roles = [role("sampled_token", sampled_token_dtype, num_requests, token_bytes, "read"),
             role("current_step", "int64", 1, 8, "read"),
             role("draft_token_column", "int64", num_requests, 8, "write")]
    final = step == proposal_depth - 1
    additions = minimums = 0
    if not final:
        roles.extend((
            role("sample_src_positions", "int64", num_requests, 8, "read_write"),
            role("next_input_ids", "int32", num_requests, 4, "write"),
            role("post_norm_hidden", "bf16", num_requests * hidden_size, 2, "read"),
            role("persistent_feedback_hidden", "bf16", num_requests * hidden_size, 2, "write")))
        additions = num_requests
        if advance_draft_positions:
            roles.extend((role("input_positions", "int64", num_requests, 8, "read_write"),
                          role("seq_lens", "int32", num_requests, 4, "read_write")))
            additions += 2 * num_requests
            minimums = 2 * num_requests
    return {
        "step": step, "proposal_depth": proposal_depth, "final_step": final,
        "source_grid": (num_requests, 1, 1), "source_block_elements": 1024,
        "source_threads": None, "launch": None, "seconds": None, "demand": None,
        "operands": tuple(roles), "integer_additions": additions, "integer_minimums": minimums,
        "sampled_to_output_dtype": "int64",
        "sampled_to_next_input_dtype": None if final else "int32",
        "feedback_operation_id": None if final else f"uniform_{step}_feedback_update_role",
        "limitations": (
            "logical_object_role_bytes_not_repeated_transactions_or_observed_dram",
            "current_step_scalar_is_one_object_not_per_cta_physical_reads",
            "feedback_role_is_same_event_as_storage_subset_not_an_extra_copy",
            "runtime_max_model_len_position_values_and_saturating_results_unbound",
            "sampler_dtype_is_declared_source_scenario_not_runtime_binding",
            "triton_default_warps_compiler_local_services_and_launch_binding_unavailable",
            "address_predicate_conversion_and_instruction_work_not_priced",
        ),
    }



def glm_gpu_v2_prepare_decode_work(*, target_seq_lens, num_rejected, positions,
                                 sample_src_positions, max_num_requests, max_model_len,
                                 proposal_depth, num_rejected_dtype,
                                 advance_draft_positions):
    """Unpriced source state/roles for 0.30 prepare_decode_inputs.

    InputBuffers owns INT32 ids/query starts/seq_lens and INT64 positions.
    DraftModelSpeculator owns INT64 draft_tokens; this producer reads that
    buffer after sampling, regardless of sampler return dtype. Rejection
    count width is declared separately. The extra CTA writes full request capacity, even in eager.
    Depth1 returns before this producer. No Triton warps are specified by
    the source, so grid and BLOCK_SIZE do not form an actual launch.
    """
    if any(type(value) is not int for value in (max_num_requests, max_model_len, proposal_depth)):
        raise TypeError("prepare decode capacity/length/depth must be integers")
    if min(max_num_requests, max_model_len, proposal_depth) <= 0:
        raise ValueError("prepare decode capacity/length/depth must be positive")
    vectors = (target_seq_lens, num_rejected, positions, sample_src_positions)
    if any(not isinstance(values, (tuple, list)) or not values for values in vectors):
        raise ValueError("prepare decode requires nonempty state vectors")
    batch = len(target_seq_lens)
    if any(len(values) != batch for values in vectors) or batch > max_num_requests:
        raise ValueError("prepare decode state vectors must match live request capacity")
    if any(type(value) is not int for values in vectors for value in values):
        raise TypeError("prepare decode states must be integers")
    if (any(not 1 <= length <= max_model_len for length in target_seq_lens)
            or any(not 0 <= rejected < length for rejected, length in zip(num_rejected, target_seq_lens))
            or any(not 0 <= position < max_model_len for position in positions)
            or any(position < 0 for position in sample_src_positions)):
        raise ValueError("prepare decode requires valid lengths, retained anchors and positions")
    if (num_rejected_dtype not in ("int32", "int64")
            or type(advance_draft_positions) is not bool):
        raise ValueError("prepare decode requires explicit count dtype and advance flag")
    if proposal_depth == 1:
        return None

    def role(name, dtype, elements, access):
        return {"role": name, "dtype": dtype, "elements": elements,
                "bytes": elements * {"int32": 4, "int64": 8}[dtype], "access": access}

    roles = [role("draft_token_column_zero", "int64", batch, "read"),
             role("target_seq_lens", "int32", batch, "read"),
             role("num_rejected", num_rejected_dtype, batch, "read"),
             role("sample_src_positions", "int64", batch, "read_write"),
             role("next_input_ids", "int32", batch, "write"),
             role("query_start_loc_capacity", "int32", max_num_requests + 1, "write"),
             role("seq_lens_capacity", "int32", max_num_requests, "write")]
    next_positions = tuple(positions)
    next_lens = tuple(length - rejected for length, rejected in zip(target_seq_lens, num_rejected))
    if advance_draft_positions:
        roles.append(role("input_positions", "int64", batch, "read_write"))
        next_positions = tuple(min(position + 1, max_model_len - 1) for position in positions)
        next_lens = tuple(min(length + 1, max_model_len) for length in next_lens)
    return {
        "source": "vllm_0_30_autoregressive_speculator.prepare_decode_inputs",
        "source_grid": (batch + 1, 1, 1), "source_block_elements": 1024,
        "source_threads": None, "launch": None, "seconds": None, "demand": None,
        "operands": tuple(roles), "positions_after": next_positions,
        "sample_src_positions_after": tuple(position + 1 for position in sample_src_positions),
        "seq_lens_after_with_padding": next_lens + (0,) * (max_num_requests - batch),
        "query_start_loc_after_with_padding": tuple(range(batch)) + (batch,) * (max_num_requests + 1 - batch),
        "integer_subtractions": batch, "integer_additions": batch * (3 if advance_draft_positions else 1),
        "integer_minimums": 2 * batch if advance_draft_positions else 0,
        "producer_binding": None,
        "limitations": (
            "declared_state_scenario_not_observed_runtime_or_sampler_count_binding",
            "capacity_padding_stores_are_logical_addresses_not_dram_transactions",
            "grid_and_block_elements_do_not_bind_compiler_warps_or_actual_launch",
            "integer_counts_exclude_address_predicate_and_conversion_instructions",
            "slot_mapping_metadata_page_ownership_and_cache_rollback_are_separate_events",
        ),
    }



def glm_gpu_v2_slot_mapping_work(*, idx_mapping, query_start_loc, positions,
                               block_tables_by_group, kv_block_sizes, kernel_block_sizes,
                               mapping_enabled, slot_capacity, returned_tokens, cp_size):
    """Unpriced CP1 source roles/state of GPU-v2 compute_slot_mappings.

    BlockTables owns INT32 block numbers, INT64 slots/strides, UINT64 table
    pointers and BOOL mapping flags. The returned view does not limit the
    extra CTA's writes: padding starts at actual tokens and covers capacity.
    CP requires a different local-position mapping and is rejected here.
    Source masked position loads use zero for SIMD tails; CP1 block-number
    loads are unmasked, so partial 1024-element blocks also read row index0.
    Distinct source objects do not prove physical transactions or residency.
    """
    if any(type(value) is not int for value in (slot_capacity, returned_tokens, cp_size)):
        raise TypeError("slot mapping capacity/returned tokens/CP must be integers")
    if cp_size != 1:
        raise NotImplementedError("slot mapping source work currently requires CP1")
    vectors = (idx_mapping, query_start_loc, positions, block_tables_by_group,
               kv_block_sizes, kernel_block_sizes, mapping_enabled)
    if any(not isinstance(values, (tuple, list)) for values in vectors):
        raise TypeError("slot mapping requires explicit state/group vectors")
    batch, groups = len(idx_mapping), len(block_tables_by_group)
    if (not batch or len(query_start_loc) != batch + 1
            or any(len(values) != groups for values in (kv_block_sizes, kernel_block_sizes, mapping_enabled))):
        raise ValueError("slot mapping request/group vectors must match")
    if any(type(value) is not int for values in (idx_mapping, query_start_loc, positions) for value in values):
        raise TypeError("slot mapping request states must be integers")
    if (query_start_loc[0] != 0 or any(a > b for a, b in zip(query_start_loc, query_start_loc[1:]))
            or query_start_loc[-1] != len(positions)
            or not 0 <= len(positions) <= returned_tokens <= slot_capacity
            or any(value < 0 for value in (*idx_mapping, *positions))):
        raise ValueError("slot mapping requires valid query starts, positions and slice/capacity")
    if (any(type(value) is not bool for value in mapping_enabled)
            or any(type(value) is not int or not 0 < value < 2**31
                   for value in (*kv_block_sizes, *kernel_block_sizes))
            or any(kv_size % kernel_size for kv_size, kernel_size in zip(kv_block_sizes, kernel_block_sizes))):
        raise ValueError("slot mapping group sizes/flags must be explicit and kernel blocks must divide KV blocks")
    if not groups:
        return None  # source returns an empty group view without a kernel
    actual = len(positions)
    output, group_reads, unique_block_reads = [], [], 0
    for table, kernel_size, enabled in zip(block_tables_by_group, kernel_block_sizes, mapping_enabled):
        if (not isinstance(table, (tuple, list)) or not table
                or any(not isinstance(row, (tuple, list)) or not row for row in table)
                or len({len(row) for row in table}) != 1
                or any(type(value) is not int or not 0 <= value < 2**31 for row in table for value in row)
                or any(row >= len(table) for row in idx_mapping)):
            raise ValueError("slot mapping requires rectangular INT32 block tables and valid request rows")
        slots, reads, tail_reads, unique_reads = [], [], [], set()
        for request, row in enumerate(idx_mapping):
            start, end = query_start_loc[request:request + 2]
            request_reads = []
            for position in positions[start:end]:
                block_index = position // kernel_size if enabled else 0
                if block_index >= len(table[row]):
                    raise ValueError("slot mapping position exceeds declared block table")
                request_reads.append(block_index)
                unique_reads.add((row, block_index))
                slot = table[row][block_index] * kernel_size + position % kernel_size
                slots.append(slot if enabled else -1)
            reads.append(tuple(request_reads))
            tail = end > start and (end - start) % 1024 != 0
            tail_reads.append(0 if tail else None)
            if tail:
                unique_reads.add((row, 0))
        unique_block_reads += len(unique_reads)
        group_reads.append({"valid_lane_block_indices_by_request": tuple(reads),
                            "simd_tail_block_index_by_request": tuple(tail_reads),
                            "distinct_block_objects": tuple(sorted(unique_reads))})
        output.append(tuple(slots) + (-1,) * (slot_capacity - actual))

    def role(name, dtype, elements, access):
        widths = {"int32": 4, "int64": 8, "uint64": 8, "bool": 1}
        return {"role": name, "dtype": dtype, "elements": elements,
                "bytes": elements * widths[dtype], "access": access}

    return {
        "source": "vllm_0_30_gpu.BlockTables.compute_slot_mappings",
        "source_grid": (groups, batch + 1, 1), "source_block_elements": 1024,
        "source_threads": None, "launch": None, "seconds": None, "demand": None,
        "actual_tokens": actual, "returned_tokens": returned_tokens, "slot_capacity": slot_capacity,
        "slot_values_capacity": tuple(output), "group_reads": tuple(group_reads),
        "operands": (
            role("idx_mapping", "int32", batch, "read"),
            role("query_start_loc", "int32", batch + 1, "read"),
            role("positions", "int64", actual, "read"),
            role("block_table_ptrs", "uint64", groups, "read"),
            role("block_table_strides", "int64", groups, "read"),
            role("kv_block_sizes", "int32", groups, "read"),
            role("kernel_block_sizes", "int32", groups, "read"),
            role("mapping_enabled", "bool", groups, "read"),
            role("distinct_block_numbers", "int32", unique_block_reads, "read"),
            role("live_slot_ids", "int64", groups * actual, "write"),
            role("capacity_padding_slot_ids", "int64", groups * (slot_capacity - actual), "write")),
        "padding_writes_inside_returned_slice_bytes": groups * (returned_tokens - actual) * 8,
        "padding_writes_outside_returned_slice_bytes": groups * (slot_capacity - returned_tokens) * 8,
        "producer_binding": None,
        "limitations": (
            "declared_block_ids_and_requests_not_normal_allocator_or_runtime_evidence",
            "logical_distinct_objects_and_slot_stores_not_dram_transactions_or_local_services",
            "shared_input_arrays_are_counted_once_not_per_group_or_cta_reads",
            "SIMD_tail_index0_object_not_an_extra_scalar_or_transaction_count",
            "compiler_warps_and_actual_launch_unbound",
            "layer_views_alias_group_slots_not_additional_stores",
            "slot_mapping_is_separate_from_KV_insert_rejected_token_rollback_and_page_lifecycle",
        ),
    }



def glm_gpu_v2_metadata_refresh_plan(*, proposal_depth, metadata_update_supported_by_groups):
    """Source event dependencies for GLM's advancing, non-dummy eager loop.

    AutoRegressiveSpeculator.advance_draft_positions=True applies to GLM.
    _configure_fused_multi_step_decode chooses fused iff every attention
    group supports draft metadata updates. These explicit capability bits
    do not bind actual backend identity. Events include host/view/library
    and model roles, not a kernel launch count or a priced native graph.
    """
    if type(proposal_depth) is not int or proposal_depth < 1:
        raise ValueError("metadata refresh plan requires positive proposal depth")
    if (not isinstance(metadata_update_supported_by_groups, (tuple, list))
            or not metadata_update_supported_by_groups
            or any(type(value) is not bool for value in metadata_update_supported_by_groups)):
        raise ValueError("metadata refresh plan requires explicit attention-group capability flags")
    fused = proposal_depth > 1 and all(metadata_update_supported_by_groups)
    events = []

    def append(identifier, kind, predecessors, *, step=None, group=None, final_step=None):
        events.append({
            "event_id": identifier, "source_role": kind,
            "predecessors": tuple(dict.fromkeys(predecessors)),
            "step": step, "attention_group": group, "final_step": final_step,
            "launch": None, "demand": None, "seconds": None,
        })
        return identifier

    def initial_metadata(step, predecessor):
        slots = append(f"uniform_{step}_slot_mapping", "compute_slot_mappings", (predecessor,), step=step)
        views = append(f"uniform_{step}_layer_slot_views", "build_slot_mappings_by_layer", (slots,), step=step)
        return (append(f"uniform_{step}_metadata_build", "build_uniform_attn_metadata", (views,), step=step),)

    if proposal_depth > 1:
        last_update = append("prepare_decode_inputs", "prepare_decode_inputs", ("on_prefill_end",))
        metadata_frontier = initial_metadata(1, last_update) if fused else ()
        for step in range(1, proposal_depth):
            if not fused:
                metadata_frontier = initial_metadata(step, last_update)
            fill = append(f"uniform_{step}_current_step_fill", "current_draft_step.fill_",
                          (last_update, *metadata_frontier), step=step)
            forward = append(f"uniform_{step}_draft_forward", "draft_model_forward",
                             (fill, *metadata_frontier), step=step)
            sample = append(f"uniform_{step}_draft_sample", "sample_draft", (forward,), step=step)
            final = step == proposal_depth - 1
            last_update = append(f"uniform_{step}_input_update", "update_draft_inputs",
                                 (sample, forward), step=step, final_step=final)
            if fused and not final:
                slots = append(f"uniform_{step + 1}_slot_mapping", "compute_slot_mappings",
                               (last_update,), step=step + 1)
                group_updates = []
                previous_group_update = slots
                for group in range(len(metadata_update_supported_by_groups)):
                    previous_group_update = append(f"uniform_{step + 1}_group_{group}_metadata_update",
                        "update_draft_decode_metadata", (previous_group_update,), step=step + 1, group=group)
                    group_updates.append(previous_group_update)
                metadata_frontier = tuple(group_updates)
    return {
        "source": "vllm_0_30_AutoRegressiveSpeculator_advancing_eager",
        "proposal_depth": proposal_depth, "advance_draft_positions": True,
        "cudagraph_runtime_mode": "NONE", "dummy_run": False,
        "metadata_update_supported_by_groups": tuple(metadata_update_supported_by_groups),
        "execution_kind": "fused" if fused else "ordinary", "source_events": tuple(events),
        "actual_attention_backend_binding": None,
        "scope": "declared_capability_source_control_flow_not_kernel_call_count_or_priced_native",
        "limitations": (
            "current_step_fill_sampler_and_group_updates_do_not_bind_library_kernel_launches",
            "host_alias_and_backend_events_are_not_assumed_free",
            "prefill_on_prefill_end_predecessor_is_external_to_this_uniform_loop_subset",
            "EPLB_DP_sync_model_fusion_and_KV_page_lifecycle_not_covered",
            "FULL_graph_capture_replay_and_nonadvancing_Gemma4_are_outside_this_GLM_eager_scope",
        ),
    }



def glm_gpu_v2_sparse_backend_metadata_plan(*, proposal_depth, backend_name):
    """Conditional source constraint for NVIDIA sparse MLA draft builders.

    These vLLM0.30 builders inherit AttentionMetadataBuilder's False flag
    through SparseMLACommonMetadataBuilder without an override. One such
    draft attention group excludes fused advancing decode regardless of
    other groups. This does not choose/observe a backend or assert the
    total group count; the ordinary path has no per-group update callbacks.
    """
    builders = {
        "FLASH_ATTN_MLA_SPARSE": ("FlashAttnMLASparseMetadataBuilder",),
        "FLASHMLA_SPARSE": ("FlashMLASparseMetadataBuilder",),
        "FLASHINFER_MLA_SPARSE_SM90": (
            "FlashInferMLASparseSM90Builder", "FlashInferMLASparseMetadataBuilder"),
    }
    if not isinstance(backend_name, str):
        raise TypeError("sparse metadata source requires an explicit backend name")
    if backend_name not in builders:
        raise NotImplementedError("sparse backend metadata capability is not source-bound")
    plan = glm_gpu_v2_metadata_refresh_plan(
        proposal_depth=proposal_depth, metadata_update_supported_by_groups=(False,))
    plan["source_backend_constraint"] = {
        "declared_sparse_backend": backend_name,
        "builder_inheritance": (*builders[backend_name],
            "SparseMLACommonMetadataBuilder", "AttentionMetadataBuilder"),
        "supports_draft_decode_metadata_update": False,
        "capability_source": "vllm/v1/attention/backend.py:AttentionMetadataBuilder",
        "capability_scope": "one_known_sparse_group_suffices_not_total_group_count",
        "backend_selection_observed": False,
        "actual_group_membership": None,
        "source_runtime_requirements": "GLM advancing eager; at least one group uses declared unmodified 0.30 builder",
    }
    plan["scope"] = "conditional_sparse_backend_source_constraint_not_observed_runtime_or_priced_native"
    return plan



def glm_gpu_v2_mtp_storage_work(model, requests, runtime, *, prefill_context_parallel_size):
    """Source-only storage roles for the eager GLM GPU-v2 proposer subset.

    vLLM 0.30 autoregressive/speculator.py _prefill gathers sampled hidden,
    then copies it into persistent feedback storage. GLM nvidia/mtp.py
    returns (post_norm_hidden, post_norm_hidden), so no second gather is
    needed. MTPSpeculator.on_prefill_end compacts shared top-k via fancy
    indexing and assignment. These are logical object roles and lifetimes,
    never library launch counts, observed traffic, demand or prediction.
    """
    from .model_config import ModelConfig
    from .runtime_inputs import RuntimeInputs
    from src.serving.schedule.schedule_step import ScheduledRequest
    if not isinstance(model, ModelConfig) or not isinstance(runtime, RuntimeInputs):
        raise TypeError("GLM storage work requires typed model/runtime inputs")
    if (model.mtp_architecture != "glm_moe_dsa_mtp" or model.compute_dtype != "bf16"
            or model.num_mtp_layers != 1):
        raise NotImplementedError("GLM storage work requires the single-layer BF16 V32 source")
    if runtime.mtp_proposer_execution_source != "vllm_0_30_gpu_v2":
        raise ValueError("GLM storage work requires explicit GPU-v2 source selection")
    if type(prefill_context_parallel_size) is not int or prefill_context_parallel_size != 1:
        raise NotImplementedError("GLM storage work currently requires PCP1")
    if not isinstance(requests, (tuple, list)) or not requests:
        raise ValueError("GLM storage work requires a nonempty request sequence")
    if any(not isinstance(request, ScheduledRequest) for request in requests):
        raise TypeError("GLM storage work requires ScheduledRequest objects")
    depths = {request.mtp_next_proposed_tokens for request in requests}
    if len(depths) != 1 or next(iter(depths)) <= 0:
        raise ValueError("GLM storage work requires one positive proposal depth")
    if any(request.mtp_proposer_mode != "padded" for request in requests):
        raise ValueError("GLM storage work requires padded proposer queries")
    depth = next(iter(depths))
    retained_rows = _mtp_retained_last_rows(requests)
    batch, hidden = len(requests), model.hidden_size
    hidden_bytes = batch * hidden * model.dtype_bytes
    index_bytes = batch * 8  # last_token_indices is INT64 in the source.

    def operation(name, kind, source_role, destination_role, dtype, value_bytes,
                  predecessors, *, selected_rows=None, consumers=()):
        return {
            "operation_id": name, "operation": kind,
            "source_role": source_role, "destination_role": destination_role,
            "dtype": dtype, "read_value_bytes": value_bytes,
            "read_index_bytes": index_bytes if selected_rows is not None else 0,
            "write_value_bytes": value_bytes, "selected_rows": selected_rows,
            "predecessors": predecessors, "consumers": consumers,
            "launch": None, "seconds": None, "demand": None,
            "execution_support": "unsupported_launch_unbound",
        }

    operations = [
        operation("prefill_hidden_gather", "materialized_gather",
                  "draft_post_norm_hidden", "sample_hidden_temporary", "bf16",
                  hidden_bytes, ("draft_prefill_forward",), selected_rows=retained_rows,
                  consumers=("draft_sample", "prefill_feedback_copy")),
        operation("prefill_feedback_copy", "copy",
                  "sample_hidden_temporary", "persistent_feedback_hidden_front", "bf16",
                  hidden_bytes, ("prefill_hidden_gather", "draft_sample")),
    ]
    if model.index_share_for_mtp_iteration and depth > 1:
        topk_bytes = batch * model.index_topk * 4  # topk_indices_buffer is INT32.
        operations.extend((
            operation("topk_compaction_gather", "materialized_gather",
                      "draft_topk_buffer", "topk_compaction_temporary", "int32",
                      topk_bytes, ("draft_prefill_forward", "prefill_feedback_copy"),
                      selected_rows=retained_rows, consumers=("topk_front_copy",)),
            operation("topk_front_copy", "copy",
                      "topk_compaction_temporary", "draft_topk_buffer_front", "int32",
                      topk_bytes, ("topk_compaction_gather",), consumers=("uniform_draft_forward",)),
        ))
    # _generate_draft samples last_hidden[:num_reqs], a view, not a fancy
    # gather. update_draft_inputs copies feedback only before the final step.
    for step_index in range(1, depth - 1):
        operations.append(operation(f"uniform_{step_index}_feedback_update_role",
            "copy_role_within_update_draft_inputs", "uniform_post_norm_hidden_front",
            "persistent_feedback_hidden_front", "bf16", hidden_bytes,
            (f"uniform_{step_index}_draft_forward", f"uniform_{step_index}_draft_sample"),
            consumers=(f"uniform_{step_index + 1}_draft_forward",)))
    return {
        "scope": "GLM GPU-v2 PCP1 eager selected-hidden/compaction/feedback storage subset",
        "retained_rows": retained_rows, "padded_query_rows": sum(r.scheduled_tokens for r in requests),
        "sample_rows": batch, "proposal_depth": depth,
        "post_norm_logits_feedback_source_alias": True,
        "uniform_sampling_is_view": True,
        "final_uniform_step_feedback_update": False,
        "sampled_token_dtype_scenario": "int64",
        "sampled_token_producer_binding": None,
        "operations": tuple(operations),
        "uniform_updates": tuple(glm_gpu_v2_draft_update_work(
            num_requests=batch, hidden_size=hidden, step=step_index, proposal_depth=depth,
            sampled_token_dtype="int64", advance_draft_positions=True)
            for step_index in range(1, depth)),
        "limitations": (
            "source_tensor_roles_not_observed_dram_or_library_launch_counts",
            "actual_storage_addresses_strides_aliases_and_allocator_lifetimes_unbound",
            "indexed_temporary_and_front_copy_are_distinct_logical_roles",
            "feedback_update_role_is_within_fused_update_not_an_independent_launch",
            "sampler_rng_logits_prefill_token_position_slot_and_other_control_roles_not_covered",
            "no_rate_demand_or_duration_assigned_to_storage_work",
            "normal_gpu_v2_workload_remains_unsupported_until_launch_binding",
        ),
    }


def vllm030_v2_greedy_rejection_forward(trace, logits, *, input_ids, positions, logits_indices,
        cu_num_logits, idx_mapping, expanded_idx_mapping, expanded_local_pos, temperature, seeds,
        seq_lens, prefill_len, logits_per_request, selected_indices, request_state_indices,
        accepted_tokens, sequence_lengths, prompt_lengths, num_speculative_tokens,
        source_profile, pool_name, name):
    """V2 plain-greedy verifier, with declared acceptance and real partial writes.

    This source profile has no sampling-parameter transforms, draft logits,
    logprobs, NaN scan, adaptive/block verification, synthetic rates or watermark.
    It records the native fixed launches; it computes neither IDs nor acceptance.
    The short-request profile binds one native verification chunk explicitly.
    Core trace dependencies remain allocation-level, not initialization proofs.
    """
    from .workload_graph import WorkloadEdge
    profile='vllm_0_30_V2_plain_allgreedy_no_sampling_params'
    if source_profile!=profile or not isinstance(trace,ForwardTrace):
        raise NotImplementedError('V2 rejection source requires its explicit plain all-greedy profile')
    if trace._recorded_graph is not None:
        raise ValueError('V2 rejection requires an open forward trace')
    for port in (logits,input_ids,positions,logits_indices,cu_num_logits,idx_mapping,expanded_idx_mapping,
            expanded_local_pos,temperature,seeds,seq_lens,prefill_len):
        trace._check_tensor(port)
    if len(input_ids.shape)!=1 or len(temperature.shape)!=1:
        raise ValueError('V2 rejection requires vector input IDs and temperature request state')
    if (type(num_speculative_tokens) is not int or num_speculative_tokens<1
            or type(logits_per_request) is not tuple or not logits_per_request
            or any(type(q) is not int or not 1<=q<=num_speculative_tokens+1 for q in logits_per_request)):
        raise ValueError('V2 rejection requires explicit positive bounded request logit extents')
    batch=len(logits_per_request);rows=sum(logits_per_request)
    if (logits.dtype!='bf16' or len(logits.shape)!=2 or logits.shape[0]!=rows or logits.shape[1]<1
            or not logits.is_contiguous or type(name) is not str or not name or type(pool_name) is not str or not pool_name):
        raise ValueError('V2 rejection requires its complete dense BF16 target logits and source names')
    vocab=logits.shape[1]
    if rows>max(1,2**30//(vocab*4)):
        raise NotImplementedError('V2 rejection multi-chunk source remains separate from this short-request profile')
    declarations=(selected_indices,request_state_indices,accepted_tokens,sequence_lengths,prompt_lengths)
    if any(type(values) is not tuple for values in declarations):
        raise TypeError('V2 rejection source declarations must be explicit tuples')
    if (len(selected_indices)!=rows or any(type(row) is not int or not 0<=row<input_ids.shape[0] for row in selected_indices)
            or len(set(selected_indices))!=rows
            or any(len(values)!=batch for values in declarations[1:])
            or any(type(a) is not int or not 0<=a<q for a,q in zip(accepted_tokens,logits_per_request,strict=True))
            or any(type(n) is not int or n<0 for values in (sequence_lengths,prompt_lengths) for n in values)):
        raise ValueError('V2 rejection requires distinct selected rows and explicit bounded acceptance/length declarations')
    states=temperature.shape[0]
    layouts=((input_ids,(input_ids.shape[0],),'int32'),(positions,input_ids.shape,'int64'),
        (logits_indices,(rows,),'int64'),(cu_num_logits,(batch+1,),'int32'),(idx_mapping,(batch,),'int64'),
        (expanded_idx_mapping,(rows,),'int64'),(expanded_local_pos,(rows,),'int32'),
        (temperature,(states,),'fp32'),(seeds,(states,),'int64'),(seq_lens,(batch,),'int32'),(prefill_len,(states,),'int32'))
    for tensor,shape,dtype in ((logits,logits.shape,'bf16'),*layouts):
        trace._check_tensor(tensor)
        if tensor.shape!=shape or tensor.dtype!=dtype or not tensor.is_contiguous:
            raise ValueError('V2 rejection input ports disagree with native dtype/extent/owner')
    if (len(input_ids.shape)!=1 or states<batch or len(set(request_state_indices))!=batch
            or any(type(index) is not int or not 0<=index<states for index in request_state_indices)):
        raise ValueError('V2 rejection request state indices must name distinct existing native states')
    ops=SymbolicOperator(trace,name,pool_name);previous=None;launches=[]
    def node(operation,shape,dtype='bf16',**attributes):
        return ops.node(WorkloadKind.VECTOR,shape,dtype,'vllm_gpu_v2_greedy_rejection',{
            'source_component':'vllm_0_30_gpu_V2_rejection_sampler','source_operation':operation,
            'source_profile':profile,'model_execution_version':'vllm_0_30_0','source_projection_only':True,
            'acceptance_scope':'caller-declared branch scenario; no numerical verification',
            'accepted_prefix_tokens':accepted_tokens,'logits_per_request':logits_per_request,
            'num_speculative_tokens':num_speculative_tokens,'workspace_and_valid_regions_declared':True,
            'native_launch_selected':False,'complete_priced_request':False,**attributes})
    def effect(operation,inputs,writes,shape,*,dtype='bf16',**attributes):
        nonlocal previous
        current=trace.call_effect(node(operation,shape,dtype,**attributes),tuple(inputs),writes=tuple(writes))
        if previous is not None:
            edge=WorkloadEdge(previous,current,'barrier')
            if edge not in trace._edges:trace._edges.append(edge)
        previous=current;launches.append(current)
        return current
    def tensor(role,shape,dtype):return trace.tensor(trace.unique_name(name+'.'+role),shape,dtype)
    def state_regions(port):return tuple(port.narrow(0,index,1) for index in request_state_indices)
    def row_region(port,start,count):return port.narrow(0,start,count)
    def sample_region(port,row,column,count):return port.narrow(0,row,1).narrow(1,column,count)
    offsets=[];cursor=0
    for q in logits_per_request:offsets.append(cursor);cursor+=q
    # Native advanced-index gathers use INT64 logits_indices twice.
    gather_attributes={'gather_factor':1,'index_count':rows,'index_bytes':8,'scattered':True,
        'materialized_output':True,'selected_indices':selected_indices,
        'source_component':'vllm_0_30_V2_rejection_sampler_advanced_index'}
    gathered=[]
    old_nodes=set(trace._nodes)
    frontier=old_nodes-{edge.source for edge in trace._edges if edge.source in old_nodes and edge.destination in old_nodes}
    for role,port in (('draft-sampled',input_ids),('selected-positions',positions)):
        op=SymbolicOperator(trace,name+':'+role,pool_name)
        gather=op.node(WorkloadKind.GATHER,{'elements':rows,'rows':rows,'hidden':1},port.dtype,'gather',gather_attributes)
        result=trace.call(gather,(port,logits_indices),shape=(rows,),dtype=port.dtype)
        current=trace._writers[result.view.allocation_id]
        for source in sorted(frontier if previous is None else (previous,)):
            edge=WorkloadEdge(source,current,'barrier')
            if edge not in trace._edges:trace._edges.append(edge)
        previous=current
        gathered.append(result)
    draft_sampled,pos=gathered
    # Even the single chunk materializes cu_num_logits - 0.
    chunk_cu=tensor('chunk-cu-num-logits',(batch+1,),'int32')
    effect('chunk_cu_subtract',(cu_num_logits,),(chunk_cu,),{'elements':batch+1,'rows':batch+1},
        dtype='int32',source_scalar=0,fixed_kernel_launch=False)
    c8=-(-vocab//8192);c1=-(-vocab//1024)
    argmax=tensor('target-local-argmax',(rows,c8),'int64')
    maximum=tensor('target-local-max',(rows,c8),'fp32')
    untouched=tuple(tensor(role,(rows,c8),'fp32') for role in
        ('target-local-sumexp','draft-local-max','draft-local-sumexp'))
    stat_inputs=[expanded_local_pos,*state_regions(temperature)];stat_writes=[]
    valid_stats=[]
    for offset,q in zip(offsets,logits_per_request,strict=True):
        valid=min(q,num_speculative_tokens);valid_stats.append((offset,valid))
        stat_inputs.extend((row_region(logits,offset,valid),row_region(expanded_idx_mapping,offset,valid)))
        stat_writes.extend((row_region(argmax,offset,valid),row_region(maximum,offset,valid)))
    effect('local_stats',stat_inputs,stat_writes,{'elements':sum(n for _,n in valid_stats)*vocab,'rows':rows,'vocab':vocab},
        grid=(rows,c8),vocab_block_size=8192,stats_written_row_spans=tuple(valid_stats),
        untouched_statistics=tuple(value.view.allocation_id for value in untouched),fixed_kernel_launch=True)
    sampled=tensor('sampled',(batch,num_speculative_tokens+1),'int64')
    count=tensor('num-sampled',(batch,),'int32')
    target_lse=tensor('target-rejected-lse',(batch,),'fp32')
    draft_lse=tensor('draft-rejected-lse',(batch,),'fp32')
    reject_inputs=[idx_mapping,chunk_cu,*state_regions(temperature),*state_regions(seeds)]
    reject_writes=[count,target_lse,draft_lse];written_prefix=[]
    for row,(offset,q,a) in enumerate(zip(offsets,logits_per_request,accepted_tokens,strict=True)):
        d=q-1;visited=min(a+1,d);written=d if a==d else a+1;written_prefix.append(written)
        if d:reject_inputs.append(row_region(draft_sampled,offset+1,d))
        if visited:reject_inputs.extend(row_region(value,offset,visited) for value in (pos,argmax,maximum))
        if written:reject_writes.append(sample_region(sampled,row,0,written))
    effect('rejection',reject_inputs,reject_writes,{'elements':sum(q-1 for q in logits_per_request),'rows':batch,'vocab':vocab},
        grid=(batch,),num_warps=1,sampled_written_prefixes=tuple(written_prefix),
        lse_stores=(0.0,0.0),fixed_kernel_launch=True)
    local_argmax=tensor('resampled-local-argmax',(batch,c1),'int64')
    local_max=tensor('resampled-local-max',(batch,c1),'fp32')
    bonus=tuple(row for row,(q,a) in enumerate(zip(logits_per_request,accepted_tokens,strict=True)) if a==q-1)
    resample_inputs=[count,chunk_cu,*state_regions(temperature)];resample_writes=[]
    for row,(offset,a) in enumerate(zip(offsets,accepted_tokens,strict=True)):
        resample_inputs.append(row_region(expanded_idx_mapping,offset+a,1))
        if row in bonus:
            resample_inputs.extend((row_region(logits,offset+a,1),row_region(pos,offset+a,1),seeds.narrow(0,request_state_indices[row],1)))
            resample_writes.extend((row_region(local_argmax,row,1),row_region(local_max,row,1)))
    effect('resample',resample_inputs,resample_writes,{'elements':len(bonus)*vocab,'rows':batch,'vocab':vocab},
        grid=(batch,c1),vocab_block_size=1024,resampled_written_requests=bonus,
        zero_write_launch=not bonus,fixed_kernel_launch=True)
    insert_inputs=[count,chunk_cu,*state_regions(temperature)];insert_writes=[count]
    for row,(offset,a) in enumerate(zip(offsets,accepted_tokens,strict=True)):
        insert_inputs.append(row_region(expanded_idx_mapping,offset+a,1))
        if row in bonus:
            insert_inputs.extend((row_region(local_argmax,row,1),row_region(local_max,row,1)))
            insert_writes.append(sample_region(sampled,row,logits_per_request[row]-1,1))
    effect('insert',insert_inputs,insert_writes,{'elements':len(bonus)*c1,'rows':batch},dtype='int64',
        grid=(batch,),bonus_written_requests=bonus,counts_before_final_mask=tuple(a+1 for a in accepted_tokens),
        fixed_kernel_launch=True)
    # rejection_sample returns before the outer input_batch count kernel.
    # Release its true scratch AND our alias/input lists at that boundary.
    del stat_inputs,stat_writes,reject_inputs,reject_writes,resample_inputs,resample_writes,insert_inputs,insert_writes
    del argmax,maximum,untouched,target_lse,draft_lse,local_argmax,local_max
    rejected=tensor('num-rejected',(batch,),'int32')
    counts=tuple(0 if seq<prompt else a+1 for seq,prompt,a in zip(sequence_lengths,prompt_lengths,accepted_tokens,strict=True))
    rejections=tuple(0 if seq<prompt else q-count for seq,prompt,q,count in
        zip(sequence_lengths,prompt_lengths,logits_per_request,counts,strict=True))
    effect('sampled_rejected_counts',(count,seq_lens,cu_num_logits,idx_mapping,*state_regions(prefill_len)),(count,rejected),
        {'elements':batch,'rows':batch},dtype='int32',grid=(batch,),
        declared_num_sampled=counts,declared_num_rejected=rejections,fixed_kernel_launch=True)
    # Retain only real returned buffers; all unused/partially initialized
    # scratch follows the native function lifetime and is never fabricated.
    return {'sampled_token_ids':sampled,'num_sampled':count,'num_rejected':rejected,
        'sampler_dispatch':'rejection',
        'declared_num_sampled':counts,'declared_num_rejected':rejections,'launch_nodes':tuple(launches),
        'source_profile':profile,'values_computed':False,'normal_capacity_accepted':False,
        'complete_priced_request':False,'scratch_initialization_scope':'explicit partial regions only; no core element-level initialization tracking'}


def vllm030_v2_plain_greedy_forward(trace, logits, *, input_ids, positions, logits_indices,
        cu_num_logits, idx_mapping, expanded_idx_mapping, expanded_local_pos, temperature, seeds,
        seq_lens, prefill_len, logits_per_request, selected_indices, request_state_indices,
        accepted_tokens, sequence_lengths, prompt_lengths, num_speculative_tokens,
        source_profile, pool_name, name):
    """Native ordinary V2 Sampler branch for a batch with no draft tokens.

    All-greedy/no-logit-processing/no-top-k-p/no-logprobs/no-NaNs/no-watermark,
    no trace replay/sampling mask/grammar/batch sharder, FP32 reduction
    (use_fp64=False). Outputs [B,1], not rejection's [B,N+1].
    Inputs and sampled values remain virtual; phase decisions are source facts.
    """
    from .workload_graph import WorkloadEdge
    profile='vllm_0_30_V2_plain_allgreedy_no_sampling_params'
    if not isinstance(trace,ForwardTrace) or source_profile!=profile:
        raise NotImplementedError('V2 ordinary sampler requires its explicit plain all-greedy profile')
    if trace._recorded_graph is not None:
        raise ValueError('V2 ordinary sampler requires an open forward trace')
    if (type(logits_per_request) is not tuple or not logits_per_request or any(q!=1 or type(q) is not int for q in logits_per_request)
            or type(accepted_tokens) is not tuple or accepted_tokens!=(0,)*len(logits_per_request)
            or any(type(a) is not int for a in accepted_tokens)
            or type(num_speculative_tokens) is not int or num_speculative_tokens<1):
        raise ValueError('V2 ordinary sampler requires a complete batch without draft logits rows')
    for port in (logits,input_ids,positions,logits_indices,cu_num_logits,idx_mapping,expanded_idx_mapping,
            expanded_local_pos,temperature,seeds,seq_lens,prefill_len):trace._check_tensor(port)
    batch=len(logits_per_request)
    if len(logits.shape)!=2 or logits.shape[0]!=batch or logits.dtype!='bf16' or len(temperature.shape)!=1 or len(input_ids.shape)!=1:
        raise ValueError('V2 ordinary sampler requires BF16 per-request logits and vector metadata')
    states=temperature.shape[0];vocab=logits.shape[1]
    layouts=((logits,logits.shape,'bf16'),(input_ids,input_ids.shape,'int32'),(positions,input_ids.shape,'int64'),
        (logits_indices,(batch,),'int64'),(cu_num_logits,(batch+1,),'int32'),(idx_mapping,(batch,),'int64'),
        (expanded_idx_mapping,(batch,),'int64'),(expanded_local_pos,(batch,),'int32'),
        (temperature,(states,),'fp32'),(seeds,(states,),'int64'),(seq_lens,(batch,),'int32'),(prefill_len,(states,),'int32'))
    if any(port.shape!=shape or port.dtype!=dtype or not port.is_contiguous for port,shape,dtype in layouts):
        raise ValueError('V2 ordinary sampler native ports have inconsistent extent/dtype/layout')
    if (any(type(values) is not tuple or len(values)!=batch for values in
                (selected_indices,request_state_indices,sequence_lengths,prompt_lengths))
            or len(set(selected_indices))!=batch or any(type(index) is not int or not 0<=index<input_ids.shape[0] for index in selected_indices)
            or len(set(request_state_indices))!=batch or any(type(index) is not int or not 0<=index<states for index in request_state_indices)
            or any(type(n) is not int or n<0 for values in (sequence_lengths,prompt_lengths) for n in values)
            or type(pool_name) is not str or not pool_name or type(name) is not str or not name):
        raise ValueError('V2 ordinary sampler declarations must bind actual selected rows, request states and lengths')
    ops=SymbolicOperator(trace,name,pool_name);previous=None;launches=[]
    old=set(trace._nodes);frontier=old-{edge.source for edge in trace._edges if edge.source in old and edge.destination in old}
    def link(current):
        nonlocal previous
        for source in sorted(frontier if previous is None else (previous,)):
            edge=WorkloadEdge(source,current,'barrier')
            if edge not in trace._edges:trace._edges.append(edge)
        previous=current
    def fresh(role,shape,dtype):return trace.tensor(trace.unique_name(name+'.'+role),shape,dtype)
    gathered=[]
    # Sampler.__call__ gathers positions first, input IDs second.
    for role,port in (('selected-positions',positions),('selected-input-ids',input_ids)):
        op=SymbolicOperator(trace,name+':'+role,pool_name)
        node=op.node(WorkloadKind.GATHER,{'elements':batch,'rows':batch,'hidden':1},port.dtype,'gather',{
            'gather_factor':1,'index_count':batch,'index_bytes':8,'scattered':True,'materialized_output':True,
            'selected_indices':selected_indices,'source_component':'vllm_0_30_V2_plain_sampler_advanced_index'})
        value=trace.call(node,(port,logits_indices),shape=(batch,),dtype=port.dtype)
        link(trace._writers[value.view.allocation_id]);gathered.append(value)
    pos,unused_input_ids=gathered
    def effect(operation,inputs,writes,shape,dtype,**attributes):
        node=ops.node(WorkloadKind.VECTOR,shape,dtype,'vllm_gpu_v2_greedy_sampler',{
            'source_component':'vllm_0_30_gpu_V2_plain_sampler','source_operation':operation,'source_profile':profile,
            'model_execution_version':'vllm_0_30_0','source_projection_only':True,
            'source_operator_call':True,'fixed_kernel_launch':operation in ('gumbel_block_argmax','sampled_rejected_counts'),
            'ATen_internal_launch_and_workspace_bound':False,
            'native_launch_selected':False,'values_computed':False,'complete_priced_request':False,**attributes})
        current=trace.call_effect(node,tuple(inputs),writes=tuple(writes));link(current);launches.append(current)
    c1=-(-vocab//1024)
    local_argmax=fresh('local-argmax',(batch,c1),'int64');local_max=fresh('local-max',(batch,c1),'fp32')
    state_ports=tuple(port.narrow(0,index,1) for port in (temperature,seeds) for index in request_state_indices)
    effect('gumbel_block_argmax',(logits,expanded_idx_mapping,pos,*state_ports),(local_argmax,local_max),
        {'elements':batch*vocab,'rows':batch,'vocab':vocab},'bf16',grid=(batch,c1),vocab_block_size=1024,
        is_drafting=False,apply_temperature=False,all_greedy=True)
    blocks=fresh('max-block-idx',(batch,1),'int64')
    effect('gumbel_reduce_argmax',(local_max,),(blocks,),{'elements':batch*c1,'rows':batch},'fp32',axis=1,keepdim=True)
    sampled=fresh('sampled',(batch,1),'int64')
    effect('gumbel_gather',(local_argmax,blocks),(sampled,),{'elements':batch,'rows':batch},'int64',
        axis=1,dynamic_index=True,index_count=batch,index_bytes=8,source_columns=c1)
    # gumbel_sample returns before Sampler creates new_ones and calls counts.
    del local_argmax,local_max,blocks
    count=fresh('num-sampled',(batch,),'int32')
    effect('seed_num_sampled_ones',(),(count,),{'elements':batch,'rows':batch},'int32',source_constant=1)
    rejected=fresh('num-rejected',(batch,),'int32')
    prefill_ports=tuple(prefill_len.narrow(0,index,1) for index in request_state_indices)
    counts=tuple(int(seq>=prompt) for seq,prompt in zip(sequence_lengths,prompt_lengths,strict=True))
    effect('sampled_rejected_counts',(count,seq_lens,cu_num_logits,idx_mapping,*prefill_ports),(count,rejected),
        {'elements':batch,'rows':batch},'int32',grid=(batch,),declared_num_sampled=counts,declared_num_rejected=(0,)*batch)
    return {'sampled_token_ids':sampled,'num_sampled':count,'num_rejected':rejected,'sampler_dispatch':'plain',
        'declared_num_sampled':counts,'declared_num_rejected':(0,)*batch,'launch_nodes':tuple(launches),
        'source_profile':profile,'values_computed':False,'normal_capacity_accepted':False,'complete_priced_request':False}


from dataclasses import dataclass


@dataclass(frozen=True)
class _VLLM030PostUpdateState:
    trace: object
    request_ids: tuple
    state_indices: tuple
    limits: tuple
    all_token_ids_storage: str
    step_ids: tuple
    total_lengths: tuple
    computed_lengths: tuple
    num_computed: object
    last_sampled: object
    all_token_ids: object
    total_len: object
    output_bin_counts: object
    next_prefill_tokens: object
    declared_sampled_ids: tuple
    finished_request_ids: tuple = ()
    last_sampled_values: tuple = ()


def validate_vllm030_last_sampled_constructor(trace,port,max_requests):
    trace._check_tensor(port)
    writer=trace._writers[port.view.allocation_id]
    if (port.shape!=(max_requests,1) or port.dtype!='int64' or not port.is_contiguous
            or port.view.allocation_id not in trace._persistent_ids
            or trace._allocations[port.view.allocation_id].freed_event is not None
            or writer is None or writer not in trace._nodes
            or 'source_component' not in trace._nodes[writer].attributes
            or trace._nodes[writer].attributes['source_component']!='vllm_0_30_RequestState_SamplingStates_constructor'
            or port.view.view_id not in trace._nodes[writer].attributes['symbolic_tensor_outputs']):
        raise ValueError('V2 initial last-sampled alias needs its unchanged source constructor-zero backing')


def vllm030_v2_post_update_forward(trace, sampler_output, *, request_ids, state_indices,
        query_lengths, computed_before, sampled_token_ids, initial_total_lengths, max_requests,
        max_model_len, vocab_size, all_token_ids_storage, idx_mapping, query_start_loc,
        step_id, pool_name, prior_state=None, _preflight_only=False, finished_request_ids=(),initialize_last_sampled=False,
        admitted_requests=(),admission_input_ids=None,initial_last_sampled=None):
    """Native last-PP post_update with declared sampled IDs, no model execution.

    The optional last-sampled zero constructor is source bound. Other resident
    initialization and request staged writes remain external declarations.
    Explicit UVA uses logical worker-owned DDR;
    physical host identity, pointer mapping and PCIe traffic remain unbound.
    GPU fallback is a separate explicit storage choice, never inferred.
    """
    if not isinstance(trace,ForwardTrace) or trace._recorded_graph is not None:
        raise ValueError('V2 post_update requires its open source trace')
    if type(_preflight_only) is not bool:
        raise TypeError('V2 post_update preflight flag must be boolean')
    if type(initialize_last_sampled) is not bool or initialize_last_sampled and prior_state is not None:
        raise ValueError('V2 last-sampled constructor initialization belongs only to the first resident revision')
    if initial_last_sampled is not None:
        if prior_state is not None or initialize_last_sampled:
            raise ValueError('V2 initial last-sampled alias cannot replace or reinitialize a live request state')
        validate_vllm030_last_sampled_constructor(trace,initial_last_sampled,max_requests)
    if (prior_state is None and '_v2_post_update_state' in vars(trace)
            or prior_state is not None and ('_v2_post_update_state' not in vars(trace)
                or trace._v2_post_update_state is not prior_state)):
        raise ValueError('V2 post_update requires its current unconsumed request-state revision')
    if (type(request_ids) is not tuple or not request_ids or len(set(request_ids))!=len(request_ids)
            or any(type(key) is not str or not key for key in request_ids)
            or any(type(n) is not int or n<1 for n in (max_requests,max_model_len,vocab_size))
            or all_token_ids_storage not in ('declared_uva_host','declared_gpu_fallback')
            or type(step_id) is not str or not step_id or type(pool_name) is not str or not pool_name):
        raise ValueError('V2 post_update requires explicit request owners, limits and resident storage branch')
    batch=len(request_ids)
    if (type(admitted_requests) is not tuple
            or any(type(row) is not tuple or len(row)!=3 or type(row[0]) is not str
                or type(row[1]) is not int or type(row[2]) is not tuple for row in admitted_requests)
            or len({row[0] for row in admitted_requests})!=len(admitted_requests)):
        raise ValueError('V2 request admission requires distinct (ID, resident row, prompt IDs) declarations')
    if (type(finished_request_ids) is not tuple or len(set(finished_request_ids))!=len(finished_request_ids)
            or any(type(key) is not str or key not in request_ids for key in finished_request_ids)):
        raise ValueError('V2 post_update finished IDs must belong to the current sampled batch')
    if (batch>max_requests or any(type(values) is not tuple or len(values)!=batch for values in
                (state_indices,query_lengths,computed_before,sampled_token_ids))
            or len(set(state_indices))!=batch or any(type(i) is not int or not 0<=i<max_requests for i in state_indices)
            or any(type(q) is not int or q<1 for q in query_lengths)
            or any(type(n) is not int or n<0 for n in computed_before)):
        raise ValueError('V2 post_update requires per-request distinct state indices, query lengths and declared old context')
    current_indices=dict(zip(request_ids,state_indices,strict=True));query_by_id=dict(zip(request_ids,query_lengths,strict=True))
    computed_by_id=dict(zip(request_ids,computed_before,strict=True))
    if (any(key not in current_indices or current_indices[key]!=index or computed_by_id[key]!=0
                or not tokens or len(tokens)!=query_by_id[key]
                or any(type(token) is not int or not 0<=token<vocab_size for token in tokens)
                for key,index,tokens in admitted_requests)
            or bool(admitted_requests)!=(admission_input_ids is not None)):
        raise ValueError('V2 admitted requests must bind their exact fresh whole prompt and current resident row')
    if admitted_requests:
        trace._check_tensor(admission_input_ids)
        if admission_input_ids.shape!=(sum(query_lengths),) or admission_input_ids.dtype!='int32' or not admission_input_ids.is_contiguous:
            raise ValueError('V2 request admission requires the actual packed target input-ID port')
    if type(sampler_output) is not dict or any(key not in sampler_output for key in
            ('sampled_token_ids','num_sampled','num_rejected','declared_num_sampled','declared_num_rejected')):
        raise ValueError('V2 post_update requires the complete actual sampler output owner')
    sampled=sampler_output['sampled_token_ids'];counts=sampler_output['num_sampled'];rejected=sampler_output['num_rejected']
    for port in (sampled,counts,rejected,idx_mapping,query_start_loc):trace._check_tensor(port)
    if len(sampled.shape)!=2:
        raise ValueError('V2 post_update sampled rows must be the actual 2D output')
    lengths=sampler_output['declared_num_sampled'];rejects=sampler_output['declared_num_rejected']
    if (type(lengths) is not tuple or type(rejects) is not tuple or len(lengths)!=batch or len(rejects)!=batch
            or any(type(n) is not int or n<0 for values in (lengths,rejects) for n in values)
            or any(type(ids) is not tuple or len(ids)!=count or any(type(i) is not int or not 0<=i<vocab_size for i in ids)
                for ids,count in zip(sampled_token_ids,lengths,strict=True))):
        raise ValueError('V2 post_update needs explicit valid sampled-ID prefixes, not shape-inferred values')
    ports=((sampled,(batch,sampled.shape[-1]),'int64'),(counts,(batch,),'int32'),(rejected,(batch,),'int32'),
        (idx_mapping,(batch,),'int64'),(query_start_loc,(batch+1,),'int32'))
    for port,shape,dtype in ports:
        trace._check_tensor(port)
        if port.shape!=shape or port.dtype!=dtype or not port.is_contiguous:
            raise ValueError('V2 post_update actual input extent/dtype/trace differs from native source')
    if any(count>sampled.shape[1] or reject>q for count,reject,q in zip(lengths,rejects,query_lengths,strict=True)):
        raise ValueError('V2 post_update cannot read an unwritten sampled tail or reject beyond its query')
    producer=trace._writers[counts.view.allocation_id]
    token_producer=trace._writers[sampled.view.allocation_id]
    if (producer is None or producer!=trace._writers[rejected.view.allocation_id] or token_producer is None
            or producer not in trace._nodes or token_producer not in trace._nodes):
        raise ValueError('V2 post_update requires actual produced sampled/count ports')
    counter_node=trace._nodes[producer];token_node=trace._nodes[token_producer]
    components=('vllm_0_30_gpu_V2_plain_sampler','vllm_0_30_gpu_V2_rejection_sampler')
    if (any(key not in counter_node.attributes for key in ('source_component','source_operation','declared_num_sampled','declared_num_rejected'))
            or counter_node.attributes['source_component'] not in components or counter_node.attributes['source_operation']!='sampled_rejected_counts'
            or counter_node.attributes['declared_num_sampled']!=lengths or counter_node.attributes['declared_num_rejected']!=rejects
            or idx_mapping.view.view_id not in counter_node.attributes['symbolic_tensor_inputs']
            or 'source_component' not in token_node.attributes or token_node.attributes['source_component'] not in components):
        raise ValueError('V2 post_update sampler ports differ from their current source producer or declared feedback')
    limits=(max_requests,max_model_len,vocab_size)
    admission_indices={key:index for key,index,_ in admitted_requests}
    admission_totals={key:len(tokens) for key,_,tokens in admitted_requests}
    if prior_state is not None:
        if type(prior_state) is not _VLLM030PostUpdateState or prior_state.trace is not trace:
            raise ValueError('V2 admission requires its original request-state owner')
        retained_indices={key:index for key,index in zip(prior_state.request_ids,prior_state.state_indices,strict=True)
            if key not in prior_state.finished_request_ids}
        retained_computed={key:count for key,count in zip(prior_state.request_ids,prior_state.computed_lengths,strict=True)
            if key not in prior_state.finished_request_ids}
        if (type(prior_state) is not _VLLM030PostUpdateState or prior_state.trace is not trace
                or set(retained_indices).intersection(admission_indices)
                or set(retained_indices.values()).intersection(admission_indices.values())
                or {**retained_indices,**admission_indices}!=current_indices
                or prior_state.limits!=limits or prior_state.all_token_ids_storage!=all_token_ids_storage
                or {**retained_computed,**{key:0 for key in admission_indices}}!=computed_by_id
                or step_id in prior_state.step_ids
                or initial_total_lengths is not None):
            raise ValueError('V2 post_update continuation changed current request state, owner, limits or step revision')
        total_by_id={**dict(zip(prior_state.request_ids,prior_state.total_lengths,strict=True)),**admission_totals}
        totals=tuple(total_by_id[key] for key in request_ids)
        last_values=prior_state.last_sampled_values
        if (type(last_values) is not tuple or len(last_values)!=max_requests
                or any(value is not None and (type(value) is not int or not 0<=value<vocab_size) for value in last_values)):
            raise ValueError('V2 post_update lost its resident last-sampled value declarations')
        resident=(prior_state.num_computed,prior_state.last_sampled,prior_state.all_token_ids,
            prior_state.total_len,prior_state.output_bin_counts,prior_state.next_prefill_tokens)
        for port in resident:
            trace._check_tensor(port)
            if port.view.allocation_id not in trace._persistent_ids or trace._allocations[port.view.allocation_id].freed_event is not None:
                raise ValueError('V2 post_update lost its original live persistent state backing')
    else:
        if (type(initial_total_lengths) is not tuple or len(initial_total_lengths)!=batch
                or any(type(n) is not int or n<0 for n in initial_total_lengths)):
            raise ValueError('V2 post_update first entry requires explicit resident initial total lengths')
        totals=initial_total_lengths
        if any(totals[request_ids.index(key)]!=count for key,count in admission_totals.items()):
            raise ValueError('V2 admission prompt length differs from initial resident total length')
        last_values=(0,)*max_requests if initialize_last_sampled or initial_last_sampled is not None else (None,)*max_requests
    after_total=tuple(old+count for old,count in zip(totals,lengths,strict=True))
    after_computed=tuple(old+q-r for old,q,r in zip(computed_before,query_lengths,rejects,strict=True))
    if any(not 0<=computed<=total<=max_model_len for computed,total in zip(after_computed,after_total,strict=True)):
        raise ValueError('V2 post_update declared history/computed extents exceed the resident request state')
    if admitted_requests and '_v2_resident_draft_tokens' in vars(trace):
        drafts,_=trace._v2_resident_draft_tokens
        trace._check_tensor(drafts)
        if (drafts.shape!=(max_requests,5) or drafts.dtype!='int64' or not drafts.is_contiguous
                or drafts.view.allocation_id not in trace._persistent_ids
                or trace._allocations[drafts.view.allocation_id].freed_event is not None
                or trace._writers[drafts.view.allocation_id] is None):
            raise ValueError('V2 admission cannot replace its original resident draft table')
    if _preflight_only:
        return after_total,after_computed
    if prior_state is None:
        def resident_port(role,shape,dtype,domain='dram'):
            return trace.tensor(trace.unique_name('V2-reqstate.'+role),shape,dtype,persistent=True,memory_domain=domain)
        resident=(resident_port('num-computed',(max_requests,),'int32'),
            resident_port('last-sampled',(max_requests,1),'int64') if initial_last_sampled is None else initial_last_sampled,
            resident_port('all-token-ids',(max_requests,max_model_len),'int32',
                'ddr' if all_token_ids_storage=='declared_uva_host' else 'dram'),
            resident_port('total-len',(max_requests,),'int32'),resident_port('output-bin-counts',(max_requests,vocab_size),'int32'),
            resident_port('next-prefill-tokens',(1,max_requests),'int32'))
    computed,last,history,total,bins,next_prefill=resident
    if initialize_last_sampled:
        from .traced_operators import SymbolicFillZero
        SymbolicFillZero(trace,'V2-last-sampled-constructor',pool_name)(last)
        writer=trace._writers[last.view.allocation_id];node=trace._nodes[writer]
        trace._nodes[writer]=node.with_attributes({**node.attributes,
            'source_component':'vllm_0_30_RequestState_last_sampled_constructor','source_operation':'torch_zeros',
            'source_projection_only':True,'normal_capacity_accepted':False})
    last_values=list(last_values)
    if admitted_requests:
        op=SymbolicOperator(trace,'V2-stage-requests',pool_name)
        node=op.node(WorkloadKind.VECTOR,{'rows':len(admitted_requests),'elements':sum(admission_totals.values())+2*len(admitted_requests)},
            'int32','vllm_gpu_v2_stage_requests',{
            'source_component':'vllm_0_30_RequestState_add_request','source_operation':'project_staged_writes',
            'admitted_request_ids':tuple(key for key,_,_ in admitted_requests),
            'request_state_indices':tuple(index for _,index,_ in admitted_requests),
            'declared_prompt_ids':tuple(tokens for _,_,tokens in admitted_requests),
            'last_sampled_cleared':False,'history_tail_cleared':False,'penalty_bins_cleared':False,
            'staging_scope':'resident writes consumed by post_update/draft; earlier native input preparation remains declared',
            'source_projection_only':True,'internal_launch_bound':False,'normal_capacity_accepted':False})
        starts={key:sum(query_lengths[:row]) for row,key in enumerate(request_ids)}
        reads=tuple(admission_input_ids.narrow(0,starts[key],len(tokens)) for key,_,tokens in admitted_requests)
        staged=tuple(port for key,index,tokens in admitted_requests for port in (
            total.narrow(0,index,1),computed.narrow(0,index,1),history.narrow(0,index,1).narrow(1,0,len(tokens))))
        trace.call_effect(node,reads,writes=staged)
        if '_v2_resident_draft_tokens' in vars(trace):
            drafts,_=trace._v2_resident_draft_tokens
            trace._check_tensor(drafts)
            reset=op.node(WorkloadKind.ELEMENTWISE,{'elements':5*len(admitted_requests)},'int64','elementwise',{
                'operation':'fill','reads':0,'writes':1,'source_constant':0,
                'source_component':'vllm_0_30_RequestState_add_request','source_operation':'draft_row_zero',
                'request_state_indices':tuple(index for _,index,_ in admitted_requests),'source_projection_only':True})
            trace.call_effect(reset,(),writes=tuple(drafts.narrow(0,index,1) for _,index,_ in admitted_requests))
    inputs=[idx_mapping,counts,rejected,query_start_loc];writes=[];increments=[]
    for row,(state,old,count,q,reject,ids) in enumerate(zip(state_indices,totals,lengths,query_lengths,rejects,sampled_token_ids,strict=True)):
        total_cell=total.narrow(0,state,1);inputs.append(total_cell)
        if count:
            last_values[state]=ids[-1]
            inputs.append(sampled.narrow(0,row,1).narrow(1,0,count))
            writes.extend((last.narrow(0,state,1),total_cell,history.narrow(0,state,1).narrow(1,old,count)))
            for token in sorted(set(ids)):
                cell=bins.narrow(0,state,1).narrow(1,token,1)
                inputs.append(cell);writes.append(cell);increments.append((state,token,ids.count(token)))
        if q-reject:
            cell=computed.narrow(0,state,1);inputs.append(cell);writes.append(cell)
    op=SymbolicOperator(trace,'V2-post-update',pool_name)
    node=op.node(WorkloadKind.VECTOR,{'rows':batch,'elements':sum(lengths)},'int32','vllm_gpu_v2_post_update',{
        'source_component':'vllm_0_30_gpu_post_update','source_operation':'post_update','grid':(batch,),'num_warps':1,
        'source_projection_only':True,'fixed_kernel_launch':True,'native_launch_selected':False,
        'request_state_indices':state_indices,'declared_sampled_ids':sampled_token_ids,'declared_bin_increments':tuple(increments),
        'finished_request_ids_after_current_round':finished_request_ids,
        'computed_before':computed_before,'computed_after':after_computed,'total_before':totals,'total_after':after_total,
        'all_token_ids_storage':all_token_ids_storage,'initialization_scope':
            'explicit last-sampled zero constructor; remaining staged writes unbound' if initialize_last_sampled else
            'retained or declared resident state; remaining constructor/staged writes unbound',
        'host_device_pointer_mapping_and_PCIe_bound':False,'values_computed':False,'normal_capacity_accepted':False})
    trace.call_effect(node,tuple(inputs),writes=tuple(writes))
    state=_VLLM030PostUpdateState(trace,request_ids,state_indices,limits,all_token_ids_storage,
        (step_id,) if prior_state is None else (*prior_state.step_ids,step_id),after_total,after_computed,*resident,sampled_token_ids,
        finished_request_ids,tuple(last_values))
    trace._v2_post_update_state=state
    return state


def vllm030_store_draft_tokens_forward(trace, draft_tokens, *, request_state, step_id, pool_name):
    """GPUModelRunner's indexed assignment into resident RequestState drafts.

    The proposer output is batch ordered. PPHandler packs from the resident
    state table using idx_mapping; it cannot gather state rows from that
    smaller batch-ordered proposer output.
    """
    if (type(request_state) is not _VLLM030PostUpdateState or request_state.trace is not trace
            or '_v2_post_update_state' not in vars(trace) or trace._v2_post_update_state is not request_state
            or request_state.step_ids[-1]!=step_id or type(pool_name) is not str or not pool_name):
        raise ValueError('V2 draft storage requires its current request-state revision')
    trace._check_tensor(draft_tokens)
    batch=len(request_state.request_ids);capacity=request_state.limits[0]
    if (draft_tokens.shape!=(batch,5) or draft_tokens.dtype!='int64' or not draft_tokens.is_contiguous
            or trace._writers[draft_tokens.view.allocation_id] is None):
        raise ValueError('V2 draft storage requires the produced complete N5 proposer output')
    if '_v2_resident_draft_tokens' in vars(trace):
        resident,steps=trace._v2_resident_draft_tokens
        trace._check_tensor(resident)
        if (step_id in steps or resident.shape!=(capacity,5) or resident.dtype!='int64'
                or resident.view.allocation_id not in trace._persistent_ids
                or trace._allocations[resident.view.allocation_id].freed_event is not None
                or trace._writers[resident.view.allocation_id] is None):
            raise ValueError('V2 draft storage lost its original resident backing or step revision')
    else:
        resident=trace.tensor(trace.unique_name('V2-reqstate.draft-tokens'),(capacity,5),'int64',persistent=True)
        from .traced_operators import SymbolicFillZero
        SymbolicFillZero(trace,'V2-request-drafts-constructor',pool_name)(resident)
        writer=trace._writers[resident.view.allocation_id];node=trace._nodes[writer]
        trace._nodes[writer]=node.with_attributes({**node.attributes,
            'source_component':'vllm_0_30_RequestState_draft_constructor','source_operation':'torch_zeros',
            'source_projection_only':True,'normal_capacity_accepted':False})
        steps=()
    op=SymbolicOperator(trace,'V2-store-drafts',pool_name)
    node=op.node(WorkloadKind.VECTOR,{'elements':batch*5,'rows':batch},'int64','vllm_gpu_v2_store_drafts',{
        'source_component':'vllm_0_30_RequestState_drafts','source_operation':'indexed_assignment',
        'request_ids':request_state.request_ids,'request_state_indices':request_state.state_indices,
        'batch_rows':batch,'resident_capacity':capacity,'source_step_id':step_id,
        'source_projection_only':True,'internal_launch_bound':False,'values_computed':False,
        'normal_capacity_accepted':False})
    trace.call_effect(node,(draft_tokens,),writes=tuple(resident.narrow(0,index,1) for index in request_state.state_indices))
    trace._v2_resident_draft_tokens=(resident,(*steps,step_id))
    return resident


def dspark_prepare_input_plan(*, target_query_lengths, target_positions, rejected_counts,
        group_ids, layer_group_indices, kernel_block_sizes, block_table_ids,
        max_requests, max_tokens, max_model_len, cache_capacity_tokens, num_query_per_request,dummy_run=False):
    """Pure CP1/no-eviction/no-clamp plan for source prepare_dflash_inputs.

    Tables and cache groups are explicit declarations; no native page receipt
    is inferred from token positions. One prepare invocation per UNIQUE group.
    """
    if (type(target_query_lengths) is not tuple or not target_query_lengths
            or any(type(q) is not int or q<1 for q in target_query_lengths)
            or type(rejected_counts) is not tuple or len(rejected_counts)!=len(target_query_lengths)
            or any(type(r) is not int or not 0<=r<q for q,r in zip(target_query_lengths,rejected_counts,strict=True))):
        raise ValueError('DSpark prepare needs physical target query spans and an accepted context row per request')
    batch=len(target_query_lengths);physical=sum(target_query_lengths)
    if (type(dummy_run) is not bool or dummy_run and (target_query_lengths!=(6,) or rejected_counts!=(0,)
            or target_positions!=(0,)*6 or num_query_per_request!=5)):
        raise ValueError('DSpark dummy prepare requires its actual six zero-position target rows and zero rejection')
    if (any(type(n) is not int or n<1 for n in (max_requests,max_tokens,max_model_len,cache_capacity_tokens,num_query_per_request))
            or batch>max_requests or max(physical,batch*num_query_per_request)>max_tokens
            or type(target_positions) is not tuple or len(target_positions)!=physical
            or any(type(pos) is not int or not 0<=pos<max_model_len for pos in target_positions)):
        raise ValueError('DSpark prepare target/query extents and positions must fit declared source buffers')
    if (type(group_ids) is not tuple or not group_ids or any(type(g) is not int or g<0 for g in group_ids)
            or tuple(sorted(set(group_ids)))!=group_ids or type(layer_group_indices) is not tuple or len(layer_group_indices)!=3
            or any(type(g) is not int or not 0<=g<len(group_ids) for g in layer_group_indices)
            or set(layer_group_indices)!=set(range(len(group_ids)))
            or type(kernel_block_sizes) is not tuple or len(kernel_block_sizes)!=len(group_ids)
            or any(type(size) is not int or size<1 for size in kernel_block_sizes)
            or type(block_table_ids) is not tuple or len(block_table_ids)!=len(group_ids)):
        raise ValueError('DSpark prepare requires its actual unique groups, three-layer mapping and per-group kernel block sizes')
    valid=tuple(q-r for q,r in zip(target_query_lengths,rejected_counts,strict=True))
    starts=[];cursor=0;contexts=[];ctx_positions=[];query_positions=[]
    for q,count in zip(target_query_lengths,valid,strict=True):
        starts.append(cursor);span=target_positions[cursor:cursor+q]
        if not dummy_run and span!=tuple(range(span[0],span[0]+q)):
            raise ValueError('DSpark prepare requires original contiguous target rows in request order')
        context=span[count-1]+1
        if context+num_query_per_request>max_model_len:
            raise NotImplementedError('DSpark prepare clamped-query source remains unbound; reject before any writes')
        contexts.append(context);ctx_positions.extend((*span[:count],*((0,)*(q-count))))
        query_positions.extend(range(context,context+num_query_per_request));cursor+=q
    cslots=[];qslots=[];full_context=[]
    for size,table in zip(kernel_block_sizes,block_table_ids,strict=True):
        if (type(table) is not tuple or len(table)!=batch or any(type(row) is not tuple or not row for row in table)
                or len({len(row) for row in table})!=1 or any(type(block) is not int or block<0 for row in table for block in row)):
            raise ValueError('DSpark prepare requires a rectangular actual request-packed block table per group')
        if dummy_run and any(block!=0 for row in table for block in row):
            raise ValueError('DSpark dummy tables must clear every column to the native null block')
        group_context=[];group_query=[];group_full=[]
        for index,(context,q,count,start) in enumerate(zip(contexts,target_query_lengths,valid,starts,strict=True)):
            row=table[index]
            if len(row)*size<context+num_query_per_request:
                raise ValueError('DSpark prepare table cannot silently clamp a missing block column')
            def slot(pos):
                block=row[pos//size]
                if block==0:
                    if dummy_run:return -1
                    raise NotImplementedError('DSpark declared no-eviction/all-valid profile cannot use native null-block slots')
                if dummy_run:raise ValueError('DSpark dummy block tables must be entirely null')
                result=block*size+pos%size
                if result>=cache_capacity_tokens:
                    raise ValueError('DSpark prepare table projects beyond its actual source cache capacity')
                return result
            group_full.extend(slot(pos) for pos in range(context))
            group_context.extend((*tuple(slot(pos) for pos in target_positions[start:start+count]),*((-1,)*(q-count))))
            group_query.extend(slot(pos) for pos in range(context,context+num_query_per_request))
        if not dummy_run and len(set((*group_full,*group_query)))!=len(group_full)+len(group_query):
            raise ValueError('DSpark prepare cannot alias distinct same-group request context/query slots')
        cslots.append(tuple(group_context));qslots.append(tuple(group_query));full_context.append(tuple(group_full))
    longest=max(target_query_lengths)+num_query_per_request
    block=min(256,1<<(max(1,longest)-1).bit_length())
    return {'dummy_run':dummy_run,'group_ids':group_ids,'layer_group_indices':layer_group_indices,'kernel_block_sizes':kernel_block_sizes,
        'block_table_ids':block_table_ids,'target_query_lengths':target_query_lengths,'target_positions':target_positions,
        'rejected_counts':rejected_counts,'contexts':tuple(contexts),'context_positions':tuple(ctx_positions),'query_positions':tuple(query_positions),
        'context_slots_by_group':tuple(cslots),'query_slots_by_group':tuple(qslots),'full_context_slots_by_group':tuple(full_context),
        'limits':(max_requests,max_tokens,max_model_len,cache_capacity_tokens,num_query_per_request),
        'source_grid':(batch,-(-longest//block)),'source_BLOCK_SIZE':block,'source_scope':'declared CP1 noCG/no-eviction/no-clamp source tables; native physical pages unbound'}


@dataclass(frozen=True)
class _DSparkPrepareBuffers:
    trace: object
    limits: tuple
    hidden_size: int
    group_ids: tuple
    layer_group_indices: tuple
    kernel_block_sizes: tuple
    shared: tuple
    context_slots: tuple
    query_slots: tuple
    table_ports: tuple
    step_ids: tuple
    attention_ports: tuple
    input_token_values: tuple
    position_values: tuple


def dspark_input_buffer_values(buffers, *, physical_rows, vocab_size):
    """Known source values require live constructor/preparation-owned storage."""
    if (type(buffers) is not _DSparkPrepareBuffers or type(physical_rows) is not int
            or not 0<physical_rows<=buffers.limits[1] or type(vocab_size) is not int or vocab_size<1
            or '_dspark_prepare_buffers' not in vars(buffers.trace)
            or buffers.trace._dspark_prepare_buffers is not buffers
            or type(buffers.input_token_values) is not tuple or len(buffers.input_token_values)!=buffers.limits[1]
            or any(type(token) is not int or not 0<=token<vocab_size for token in buffers.input_token_values[:physical_rows])
            or type(buffers.position_values) is not tuple or len(buffers.position_values)!=buffers.limits[1]
            or any(type(pos) is not int or not 0<=pos<buffers.limits[2] for pos in buffers.position_values)):
        raise ValueError('DSpark physical input values require their actual constructor/prepare history')
    for role in ('input_ids','positions'):
        port=dict(buffers.shared)[role];buffers.trace._check_tensor(port)
        writer=buffers.trace._writers[port.view.allocation_id]
        if (port.view.allocation_id not in buffers.trace._persistent_ids or writer is None
                or writer not in buffers.trace._nodes
                or 'source_component' not in buffers.trace._nodes[writer].attributes
                or buffers.trace._nodes[writer].attributes['source_component'] not in (
                    'vllm_0_30_InputBuffers_constructor','vllm_0_30_prepare_dflash_inputs')):
            raise ValueError('DSpark input history lost its actual constructor/prepare writer')
    return buffers.input_token_values[:physical_rows],buffers.position_values[:physical_rows]


def validate_dspark_prepare_ports(trace, plan, *, state_indices, target_positions,
        target_query_start_loc, idx_mapping, temperature, seeds, hidden_size, noise_token_id,
        step_id, pool_name, prior_buffers):
    """Pure whole-program checks used before any post_update or group writes."""
    if (not isinstance(trace,ForwardTrace) or trace._recorded_graph is not None or type(plan) is not dict
            or type(hidden_size) is not int or hidden_size<1 or type(noise_token_id) is not int or noise_token_id<0
            or type(step_id) is not str or not step_id or type(pool_name) is not str or not pool_name):
        raise ValueError('DSpark prepare requires an open owner and valid source dimensions')
    plan_fields=('dummy_run','target_query_lengths','target_positions','rejected_counts','group_ids','layer_group_indices',
        'kernel_block_sizes','block_table_ids','limits')
    if any(key not in plan for key in plan_fields) or type(plan['limits']) is not tuple or len(plan['limits'])!=5:
        raise ValueError('DSpark prepare input plan is incomplete')
    expected=dspark_prepare_input_plan(**{key:plan[key] for key in plan_fields if key!='limits'},
        max_requests=plan['limits'][0],max_tokens=plan['limits'][1],max_model_len=plan['limits'][2],
        cache_capacity_tokens=plan['limits'][3],num_query_per_request=plan['limits'][4])
    if plan!=expected:
        raise ValueError('DSpark prepare plan differs from its fully validated source projection')
    max_requests,max_tokens,max_len,capacity,N=plan['limits'];batch=len(plan['target_query_lengths']);physical=sum(plan['target_query_lengths'])
    for port in (target_positions,target_query_start_loc,idx_mapping,temperature,seeds):
        trace._check_tensor(port)
    if (target_positions.shape!=(physical,) or target_positions.dtype!='int64'
            or target_query_start_loc.shape!=(batch+1,) or target_query_start_loc.dtype!='int32'
            or idx_mapping.shape!=(batch,) or idx_mapping.dtype!='int64'
            or temperature.dtype!='fp32' or len(temperature.shape)!=1 or seeds.shape!=temperature.shape or seeds.dtype!='int64'
            or any(state>=temperature.shape[0] for state in state_indices)):
        raise ValueError('DSpark prepare target metadata ports differ from physical query/state extents')
    if prior_buffers is not None:
        if (type(prior_buffers) is not _DSparkPrepareBuffers or prior_buffers.trace is not trace
                or '_dspark_prepare_buffers' not in vars(trace) or trace._dspark_prepare_buffers is not prior_buffers
                or prior_buffers.limits!=plan['limits'] or prior_buffers.hidden_size!=hidden_size
                or prior_buffers.group_ids!=plan['group_ids'] or prior_buffers.layer_group_indices!=plan['layer_group_indices']
                or prior_buffers.kernel_block_sizes!=plan['kernel_block_sizes'] or step_id in prior_buffers.step_ids
                or any(port.shape!=(max_requests,len(table[0])) for port,table in zip(prior_buffers.table_ports,plan['block_table_ids'],strict=True))):
            raise ValueError('DSpark prepare must retain its current original group/metadata backing and limits')
        if (len(prior_buffers.attention_ports)!=len(plan['group_ids'])
                or any(type(pair) is not tuple or len(pair)!=2 for pair in prior_buffers.attention_ports)):
            raise ValueError('DSpark prepare requires its original per-group attention metadata buffers')
        if (type(prior_buffers.input_token_values) is not tuple or len(prior_buffers.input_token_values)!=max_tokens
                or any(value is not None and (type(value) is not int or value<0) for value in prior_buffers.input_token_values)
                or type(prior_buffers.position_values) is not tuple or len(prior_buffers.position_values)!=max_tokens
                or any(type(value) is not int or not 0<=value<max_len for value in prior_buffers.position_values)):
            raise ValueError('DSpark prepare lost its constructor and retained input-value history')
        for pair in prior_buffers.attention_ports:
            for port,dtype in zip(pair,('int32','bool'),strict=True):
                trace._check_tensor(port)
                if (port.shape!=(max_tokens,) or port.dtype!=dtype or not port.is_contiguous
                        or port.view.allocation_id not in trace._persistent_ids
                        or trace._allocations[port.view.allocation_id].freed_event is not None):
                    raise ValueError('DSpark attention metadata lost its live persistent backing')
    elif '_dspark_prepare_buffers' in vars(trace):
        raise ValueError('DSpark prepare cannot replace live source buffers with a new owner')
    return None


def _dspark_prepare_storage(trace,plan,hidden_size,pool_name,prior_buffers):
    max_requests,max_tokens,max_len,capacity,N=plan['limits']
    if prior_buffers is None:
        def buf(role,shape,dtype):return trace.tensor(trace.unique_name('DFlash.'+role),shape,dtype,persistent=True)
        shared=tuple((role,buf(role,shape,dtype)) for role,shape,dtype in (
            ('input_ids',(max_tokens,),'int32'),('positions',(max_tokens,),'int64'),('query_start',(max_requests+1,),'int32'),
            ('seq_lens',(max_requests,),'int32'),('context_positions',(max_tokens,),'int64'),
            ('sample_indices',(max_requests*N,),'int64'),('sample_pos',(max_requests*N,),'int64'),
            ('sample_idx_mapping',(max_requests*N,),'int32'),('temperature',(max_requests,),'fp32'),
            ('seeds',(max_requests,),'int64'),('hidden_states',(max_tokens,hidden_size),'bf16'),('anchor_indices',(max_requests,),'int64')))
        context_slots=tuple(buf(f'group-{gid}.context-slots',(max_tokens,),'int64') for gid in plan['group_ids'])
        query_slots=tuple(buf(f'group-{gid}.query-slots',(max_tokens,),'int64') for gid in plan['group_ids'])
        tables=tuple(buf(f'group-{gid}.block-table',(max_requests,len(table[0])),'int32') for gid,table in zip(plan['group_ids'],plan['block_table_ids'],strict=True))
        attention_ports=tuple((buf(f'group-{gid}.token-to-request',(max_tokens,),'int32'),
            buf(f'group-{gid}.is-valid-token',(max_tokens,),'bool')) for gid in plan['group_ids'])
        # Native InputBuffers.__init__ uses torch.zeros for both full buffers.
        # Bind these writes before retaining a zero-valued tail; prepare never
        # clears that tail on subsequent rounds.
        from .traced_operators import SymbolicFillZero
        for role in ('input_ids','positions'):
            port=dict(shared)[role]
            SymbolicFillZero(trace,'DFlash-constructor-zero-'+role,pool_name)(port)
            writer=trace._writers[port.view.allocation_id];node=trace._nodes[writer]
            trace._nodes[writer]=node.with_attributes({**node.attributes,
                'source_component':'vllm_0_30_InputBuffers_constructor','source_operation':'torch_zeros',
                'input_buffer_role':role,'source_projection_only':True,
                'full_constructor_bound':False,'normal_capacity_accepted':False})
        input_values=(0,)*max_tokens;position_values=(0,)*max_tokens
        if plan['dummy_run']:
            for role,shape,dtype in (('dummy-last-sampled',(max_requests,1),'int64'),
                    ('dummy-temperature',(max_requests,),'fp32'),('dummy-seeds',(max_requests,),'int64')):
                port=buf(role,shape,dtype);shared=(*shared,(role,port))
                SymbolicFillZero(trace,'DFlash-dummy-constructor-'+role,pool_name)(port)
                writer=trace._writers[port.view.allocation_id];node=trace._nodes[writer]
                trace._nodes[writer]=node.with_attributes({**node.attributes,
                    'source_component':'vllm_0_30_RequestState_SamplingStates_constructor','source_dummy_run':True,
                    'source_operation':'torch_zeros','physical_UVA_mapping_bound':False})
    else:
        shared=prior_buffers.shared;context_slots=prior_buffers.context_slots;query_slots=prior_buffers.query_slots;tables=prior_buffers.table_ports
        attention_ports=prior_buffers.attention_ports
        input_values=prior_buffers.input_token_values;position_values=prior_buffers.position_values
    return shared,context_slots,query_slots,tables,attention_ports,input_values,position_values


def dspark_prepare_inputs_forward(trace, plan, *, request_state, sampler_output, target_positions,
        target_query_start_loc, idx_mapping, temperature, seeds, hidden_size, noise_token_id,
        step_id, pool_name, prior_buffers=None, _preflight_only=False, _allocate_only=False):
    """Actual shared prepare buffers, one kernel per declared cache group.

    Input ID/position constructor zeros are explicit. The rest of constructor
    setup and block-table staging remain external. Buffers retain genuine
    active/padding write regions; no CG replay is claimed.
    """
    if (type(_preflight_only) is not bool or type(_allocate_only) is not bool or _allocate_only and _preflight_only
            or type(request_state) is not _VLLM030PostUpdateState or request_state.trace is not trace):
        raise ValueError('DSpark prepare requires its actual post_update state and preflight mode')
    validate_dspark_prepare_ports(trace,plan,state_indices=request_state.state_indices,target_positions=target_positions,
        target_query_start_loc=target_query_start_loc,idx_mapping=idx_mapping,temperature=temperature,seeds=seeds,
        hidden_size=hidden_size,noise_token_id=noise_token_id,step_id=step_id,pool_name=pool_name,prior_buffers=prior_buffers)
    max_requests,max_tokens,max_len,capacity,N=plan['limits'];batch=len(plan['target_query_lengths']);physical=sum(plan['target_query_lengths'])
    if ('_v2_post_update_state' not in vars(trace) or trace._v2_post_update_state is not request_state
            or request_state.computed_lengths!=plan['contexts'] or request_state.limits[0]!=max_requests
            or request_state.limits[1]!=max_len or sampler_output['declared_num_rejected']!=plan['rejected_counts']
            or len(request_state.request_ids)!=batch):
        raise ValueError('DSpark prepare contexts must match actual current post_update and sampler feedback')
    if _preflight_only:return None
    shared,context_slots,query_slots,tables,attention_ports,input_values,position_values=\
        _dspark_prepare_storage(trace,plan,hidden_size,pool_name,prior_buffers)
    if _allocate_only:
        if prior_buffers is not None:return prior_buffers
        state=_DSparkPrepareBuffers(trace,plan['limits'],hidden_size,plan['group_ids'],plan['layer_group_indices'],
            plan['kernel_block_sizes'],shared,context_slots,query_slots,tables,(),attention_ports,input_values,position_values)
        trace._dspark_prepare_buffers=state
        return state
    ports=dict(shared);query_rows=batch*N;op=SymbolicOperator(trace,'DFlash-prepare-inputs',pool_name)
    # Values below describe the declared addresses/IDs of the symbolic program,
    # never numerically inferred sampler outputs. A no-sample prefill anchor
    # stays unknown until its actual staged input values are represented.
    active_ids=tuple(token for row,ids in enumerate(request_state.declared_sampled_ids)
        for token in ((ids[-1] if sampler_output['declared_num_sampled'][row] else None),*((noise_token_id,)*(N-1))))
    input_values=(*active_ids,*input_values[query_rows:])
    position_values=(*plan['query_positions'],*position_values[query_rows:])
    for group,gid in enumerate(plan['group_ids']):
        inputs=[target_positions,target_query_start_loc,idx_mapping,sampler_output['num_sampled'],sampler_output['num_rejected']]
        for row,state in enumerate(request_state.state_indices):
            if sampler_output['declared_num_sampled'][row]>0:inputs.append(request_state.last_sampled.narrow(0,state,1))
            else:inputs.append(request_state.next_prefill_tokens.narrow(1,state,1))
            inputs.extend((temperature.narrow(0,state,1),seeds.narrow(0,state,1)))
            # Only valid context/query columns are read. Null/short columns
            # were rejected by the pure no-eviction/no-clamp plan.
            start=sum(plan['target_query_lengths'][:row]);valid=plan['target_query_lengths'][row]-plan['rejected_counts'][row]
            accessed=sorted({pos//plan['kernel_block_sizes'][group] for pos in
                (*plan['target_positions'][start:start+valid],*range(plan['contexts'][row],plan['contexts'][row]+N))})
            inputs.extend(tables[group].narrow(0,row,1).narrow(1,column,1) for column in accessed)
        writes=[ports['input_ids'].narrow(0,0,query_rows),ports['positions'].narrow(0,0,query_rows),
            ports['query_start'],ports['seq_lens'],context_slots[group].narrow(0,0,physical),
            ports['context_positions'].narrow(0,0,physical),query_slots[group],
            ports['sample_indices'],ports['sample_pos'],ports['sample_idx_mapping']]
        writes.extend(port.narrow(0,state,1) for port in (ports['temperature'],ports['seeds']) for state in request_state.state_indices)
        node=op.node(WorkloadKind.VECTOR,{'rows':batch,'elements':physical+query_rows},'int32','dspark_prepare_inputs',{
            'source_component':'vllm_0_30_prepare_dflash_inputs','source_operation':'prepare_inputs','source_projection_only':True,
            'group_id':gid,'group_ordinal':group,'kernel_block_size':plan['kernel_block_sizes'][group],
            'layer_group_indices':plan['layer_group_indices'],'grid':plan['source_grid'],'BLOCK_SIZE':plan['source_BLOCK_SIZE'],
            'sample_from_anchor':True,'num_query_per_request':N,'CP_SIZE':1,'no_CG':True,
            'context_positions':plan['context_positions'],'query_positions':plan['query_positions'],
            'declared_query_input_ids':active_ids,'retained_input_tail_start':query_rows,
            'input_ID_position_constructor_bound':True,
            'context_slots':plan['context_slots_by_group'][group],'query_slots':plan['query_slots_by_group'][group],
            'query_slot_padding':(query_rows,max_tokens,-1),'sample_padding':(query_rows,max_requests*N,0,0,-1),
            'seq_lens_padding':(batch,max_requests,0),'query_start_padding':(batch,max_requests+1,query_rows),
            'sample_position_rule':'raw_query_pos+1','sample_idx_mapping_dtype':'int32',
            'slot_mapping_scope':'explicit source block-table projection; native physical pages unbound',
            'source_constructor_and_table_staging_bound':False,'normal_capacity_accepted':False,'values_computed':False})
        trace.call_effect(node,tuple(inputs),writes=tuple(writes))
    state=_DSparkPrepareBuffers(trace,plan['limits'],hidden_size,plan['group_ids'],plan['layer_group_indices'],
        plan['kernel_block_sizes'],shared,context_slots,query_slots,tables,
        (step_id,) if prior_buffers is None else (*prior_buffers.step_ids,step_id),attention_ports,input_values,position_values)
    trace._dspark_prepare_buffers=state
    return state


def validate_dspark_idle_request_state(trace,state,temperature,seeds):
    if (type(state) is not _VLLM030PostUpdateState or state.trace is not trace
            or '_v2_post_update_state' not in vars(trace) or trace._v2_post_update_state is not state
            or set(state.request_ids)!=set(state.finished_request_ids)
            or len(state.last_sampled_values)!=state.limits[0]
            or type(state.last_sampled_values[0]) is not int or not 0<=state.last_sampled_values[0]<state.limits[2]):
        raise ValueError('DSpark idle requires finished retained state with known last-sampled row0')
    for port,shape,dtype in ((state.last_sampled,(state.limits[0],1),'int64'),
            (temperature,(state.limits[0],),'fp32'),(seeds,(state.limits[0],),'int64')):
        trace._check_tensor(port)
        if port.shape!=shape or port.dtype!=dtype or not port.is_contiguous:
            raise ValueError('DSpark retained idle state/sampling ports changed their native capacity or dtype')
    if (state.last_sampled.view.allocation_id not in trace._persistent_ids
            or trace._allocations[state.last_sampled.view.allocation_id].freed_event is not None
            or trace._writers[state.last_sampled.view.allocation_id] is None):
        raise ValueError('DSpark retained idle last-sampled backing is no longer live')
    return state.last_sampled_values[0]


def dspark_dummy_prepare_inputs_forward(trace,plan,*,target_positions,target_query_start_loc,idx_mapping,
        hidden_size,noise_token_id,step_id,pool_name,prior_buffers=None,_allocate_only=False,
        request_state=None,temperature=None,seeds=None):
    """Real dummy propose counters/prepare; it does not call post_update.

    Never-admitted workers read constructor cells. Admitted workers read the
    original request/sampling state row0, including historical nonzero IDs.
    """
    if (type(plan) is not dict or 'dummy_run' not in plan or plan['dummy_run'] is not True
            or type(_allocate_only) is not bool or type(noise_token_id) is not int or noise_token_id<0):
        raise ValueError('DSpark dummy preparation requires its explicit null-table zero-state source plan')
    for port,shape,dtype in ((target_positions,(6,),'int64'),(target_query_start_loc,(2,),'int32'),(idx_mapping,(1,),'int64')):
        trace._check_tensor(port)
        if port.shape!=shape or port.dtype!=dtype:
            raise ValueError('DSpark dummy target input metadata differs from its actual six-row source')
    maximum,tokens,length,capacity,N=plan['limits']
    if request_state is None:
        if temperature is not None or seeds is not None:
            raise ValueError('DSpark idle sampling ports require their explicit retained request state')
        anchor=0
    else:
        if prior_buffers is None or request_state.limits[0]!=maximum:
            raise ValueError('DSpark admitted idle cannot replace its original preparation buffers or capacity')
        anchor=validate_dspark_idle_request_state(trace,request_state,temperature,seeds)
    shared,cslots,qslots,tables,attention,input_values,position_values=\
        _dspark_prepare_storage(trace,plan,hidden_size,pool_name,prior_buffers)
    ports=dict(shared)
    if request_state is None:
        if any(role not in ports for role in ('dummy-last-sampled','dummy-temperature','dummy-seeds')):
            raise ValueError('DSpark admitted idle cannot invent replacement constructor-zero request state')
        last=ports['dummy-last-sampled'];temperature=ports['dummy-temperature'];seeds=ports['dummy-seeds']
    else:last=request_state.last_sampled
    validate_dspark_prepare_ports(trace,plan,state_indices=(0,),target_positions=target_positions,
        target_query_start_loc=target_query_start_loc,idx_mapping=idx_mapping,temperature=temperature,
        seeds=seeds,hidden_size=hidden_size,noise_token_id=noise_token_id,step_id=step_id,
        pool_name=pool_name,prior_buffers=prior_buffers)
    if _allocate_only and prior_buffers is not None:return prior_buffers
    steps=() if prior_buffers is None else prior_buffers.step_ids
    if not _allocate_only:
        op=SymbolicOperator(trace,'DSpark-dummy-propose',pool_name)
        counters=[]
        for role,value in (('sampled',1),('rejected',0)):
            port=trace.tensor(trace.unique_name('dummy-'+role),(1,),'int32')
            node=op.node(WorkloadKind.ELEMENTWISE,{'elements':1},'int32','elementwise',{
                'operation':'fill','reads':0,'writes':1,'source_constant':value,
                'source_component':'vllm_0_30_dummy_propose_counters','source_dummy_run':True})
            trace.call(node,(),shape=(1,),dtype='int32',out=port);counters.append(port)
        for group,gid in enumerate(plan['group_ids']):
            zero=SymbolicOperator(trace,f'DSpark-dummy-table-{gid}',pool_name)
            active_table=tables[group].narrow(0,0,1)
            node=zero.node(WorkloadKind.ELEMENTWISE,{'elements':tables[group].shape[1]},'int32','elementwise',{
                'operation':'fill','reads':0,'writes':1,'source_constant':0,'source_dummy_run':True,
                'source_component':'vllm_0_30_dummy_BlockTables'})
            trace.call(node,(),shape=active_table.shape,dtype='int32',out=active_table)
            writes=(ports['input_ids'].narrow(0,0,N),ports['positions'].narrow(0,0,N),ports['query_start'],
                ports['seq_lens'],cslots[group].narrow(0,0,6),ports['context_positions'].narrow(0,0,6),qslots[group],
                ports['sample_indices'],ports['sample_pos'],ports['sample_idx_mapping'],
                ports['temperature'].narrow(0,0,1),ports['seeds'].narrow(0,0,1))
            node=op.node(WorkloadKind.VECTOR,{'rows':1,'elements':6+N},'int32','dspark_prepare_inputs',{
                'source_component':'vllm_0_30_prepare_dflash_inputs','source_operation':'prepare_inputs',
                'source_dummy_run':True,'group_id':gid,'group_ordinal':group,'grid':plan['source_grid'],
                'BLOCK_SIZE':plan['source_BLOCK_SIZE'],'num_query_per_request':N,'sample_from_anchor':True,
                'query_positions':plan['query_positions'],'context_positions':plan['context_positions'],
                'declared_query_input_ids':(anchor,*((noise_token_id,)*(N-1))),
                'idle_request_state_source':'never_admitted_constructor' if request_state is None else 'retained_request_state',
                'context_slots':plan['context_slots_by_group'][group],'query_slots':plan['query_slots_by_group'][group],
                'query_slot_padding':(N,tokens,-1),'native_request_post_update_called':False,
                'normal_capacity_accepted':False,'source_projection_only':True})
            trace.call_effect(node,(target_positions,target_query_start_loc,idx_mapping,*counters,last.narrow(0,0,1),
                temperature.narrow(0,0,1),seeds.narrow(0,0,1),tables[group]),writes=writes)
        input_values=(anchor,*((noise_token_id,)*(N-1)),*input_values[N:])
        position_values=(*plan['query_positions'],*position_values[N:]);steps=(*steps,step_id)
    buffers=_DSparkPrepareBuffers(trace,plan['limits'],hidden_size,plan['group_ids'],plan['layer_group_indices'],
        plan['kernel_block_sizes'],shared,cslots,qslots,tables,steps,attention,input_values,position_values)
    trace._dspark_prepare_buffers=buffers
    return buffers


def dspark_attention_metadata_forward(trace, buffers, plan, *, step_id, pool_name):
    """Build active sparse-SWA request mapping and validity on owned buffers.

    The no-CG profile has no attention-token padding. Constructor initialization,
    native metadata-builder selection and ATen internal workspace remain unbound.
    """
    from .workload_graph import WorkloadEdge
    if (not isinstance(trace,ForwardTrace) or trace._recorded_graph is not None
            or type(buffers) is not _DSparkPrepareBuffers or buffers.trace is not trace
            or '_dspark_prepare_buffers' not in vars(trace) or trace._dspark_prepare_buffers is not buffers
            or type(step_id) is not str or not step_id or not buffers.step_ids or buffers.step_ids[-1]!=step_id
            or type(pool_name) is not str or not pool_name or type(plan) is not dict):
        raise ValueError('DSpark attention metadata requires its current completed prepare revision')
    fields=('dummy_run','target_query_lengths','target_positions','rejected_counts','group_ids','layer_group_indices',
        'kernel_block_sizes','block_table_ids','limits')
    if any(key not in plan for key in fields) or type(plan['limits']) is not tuple or len(plan['limits'])!=5:
        raise ValueError('DSpark attention metadata requires its complete validated prepare plan')
    maximum,tokens,length,capacity,N=plan['limits']
    expected=dspark_prepare_input_plan(**{key:plan[key] for key in fields if key!='limits'},
        max_requests=maximum,max_tokens=tokens,max_model_len=length,cache_capacity_tokens=capacity,num_query_per_request=N)
    if (plan!=expected or buffers.limits!=plan['limits'] or buffers.group_ids!=plan['group_ids']
            or buffers.layer_group_indices!=plan['layer_group_indices'] or buffers.kernel_block_sizes!=plan['kernel_block_sizes']
            or len(buffers.attention_ports)!=len(buffers.group_ids) or len(buffers.query_slots)!=len(buffers.group_ids)):
        raise ValueError('DSpark attention metadata differs from its current prepare groups and limits')
    batch=len(plan['target_query_lengths']);rows=batch*N;ports=dict(buffers.shared)
    starts=ports['query_start']
    for port,shape,dtype in ((starts,(maximum+1,),'int32'),
            *((slot,(tokens,),'int64') for slot in buffers.query_slots),
            *((port,(tokens,),dtype) for pair in buffers.attention_ports for port,dtype in zip(pair,('int32','bool'),strict=True))):
        trace._check_tensor(port)
        if (port.shape!=shape or port.dtype!=dtype or not port.is_contiguous
                or port.view.allocation_id not in trace._persistent_ids
                or trace._allocations[port.view.allocation_id].freed_event is not None):
            raise ValueError('DSpark attention metadata requires its live typed persistent buffers')
    for slot,pair in zip(buffers.query_slots,buffers.attention_ports,strict=True):
        writer=trace._writers[slot.view.allocation_id]
        if writer is None or trace._nodes[writer].attributes['source_component']!='vllm_0_30_prepare_dflash_inputs':
            raise ValueError('DSpark attention metadata requires actual prepared query slots')
        for port in pair:
            writer=trace._writers[port.view.allocation_id]
            if writer is not None and trace._nodes[writer].attributes['source_step_id']==step_id:
                raise ValueError('DSpark attention metadata was already built for this prepare step')
    op=SymbolicOperator(trace,'DSpark-attention-metadata',pool_name);result=[];previous=None
    for gid,slot,(mapping,validity) in zip(buffers.group_ids,buffers.query_slots,buffers.attention_ports,strict=True):
        active_mapping=mapping.narrow(0,0,rows);active_validity=validity.narrow(0,0,rows)
        attrs={'source_component':'vllm_0_30_dspark_attention_metadata','source_step_id':step_id,
            'group_id':gid,'source_projection_only':True,'native_provider_bound':False,
            'source_constructor_bound':False,'normal_capacity_accepted':False,'values_computed':False}
        node=op.node(WorkloadKind.VECTOR,{'elements':rows,'rows':rows},'int32','dspark_token_request_mapping',{
            **attrs,'source_operation':'token_request_mapping','grid':(-(-rows//256),),'BLOCK_SIZE':256,
            'num_warps':4,'num_requests':batch,'num_mapped_tokens':rows,'num_output_tokens':rows,
            'boundary_search':'device_query_start_loc','fixed_kernel_launch':True})
        current=trace.call_effect(node,(starts.narrow(0,0,batch+1),),writes=(active_mapping,))
        if previous is not None:trace._edges.append(WorkloadEdge(previous,current,'barrier'))
        node=op.node(WorkloadKind.VECTOR,{'elements':rows,'rows':rows},'int64','dspark_slot_validity',{
            **attrs,'source_operation':'slot_mapping_ge_zero','source_scalar':0,'fixed_kernel_launch':False,
            'ATen_internal_launch_and_workspace_bound':False})
        predicate=trace.call(node,(slot.narrow(0,0,rows),),shape=(rows,),dtype='bool',controls=(active_mapping,))
        node=op.node(WorkloadKind.VECTOR,{'elements':rows,'rows':rows},'bool','dspark_validity_copy',{
            **attrs,'source_operation':'validity_copy','fixed_kernel_launch':False,
            'ATen_internal_launch_and_workspace_bound':False})
        previous=trace.call_effect(node,(predicate,),writes=(active_validity,))
        del predicate
        result.append((active_mapping,active_validity))
    return tuple(result)
