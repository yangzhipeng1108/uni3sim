"""V4 model/decoder caller reference and a separate observation replay.

Original Python source controls the model/state loop and selected primitive
programs. Independent typed math, payload and caller-port receipts remain
separate from complete internal-storage and actual provider acceptance.
"""

import ast
from functools import wraps
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

from .qwen3_source import _Source, _Tensor
from .v4_partial_source import PartialStateSource
from .v4_finish_source import FinishSource
from .v4_ffn_source import FFN_SOURCE_FILES, MoESource
from .v4_output_source import OUTPUT_SOURCE_FILES, OutputProjectionSource
from .v4_indexer_source import INDEXER_SOURCE_FILES, IndexerWorkSource
from .v4_attention_source import AttentionMathSource
from .v4_mhc_source import MHC_SOURCE_FILES, MHCSource
from .v4_norm_source import NORM_SOURCE_FILES, QKVNormSource, FinalRMSNormSource
from .v4_target_source import TARGET_SOURCE_FILES, TargetSource, argmax_source_reference
from .v4_fp8_source import FP8LinearSource, fp8_quantizer_reference
from .v4_prepare_source import prepare_work_reference
from .v4_terminal_storage import terminal_storage_reference
from .v4_decode_storage import DECODE_STORAGE_SOURCE_FILES, decode_index_storage_reference
from .v4_expert_storage import expert_storage_reference
from .v4_indexer_storage import indexer_decode_storage_reference
from .v4_sparse_storage import SPARSE_STORAGE_SOURCE_FILES, sparse_prefill_storage_reference, sparse_decode_view_reference


class _Calls:
    """Store scalar layouts/identity tokens only; no frames/tensor handles."""
    def __init__(self, tensor_storage):
        self.tensor_storage = tensor_storage
        self.rows, self.storage_ids, self.storage_counts, self.counts, self.order = [], {}, {}, {}, {}

    def freeze(self, rank, value):
        if value is None:
            return None
        if isinstance(value, tuple):
            return tuple(self.freeze(rank, item) for item in value)
        storage = self.tensor_storage(value)
        key = rank, storage
        if key not in self.storage_ids:
            if rank not in self.storage_counts:
                self.storage_counts[rank] = 0
            self.storage_ids[key] = self.storage_counts[rank]
            self.storage_counts[rank] += 1
        if isinstance(value, _Tensor):
            strides, offset = value.strides, value.storage_offset
        else:
            strides = tuple(step // value.view.element_bytes for step in value.view.strides_bytes)
            offset = value.view.offset_bytes // value.view.element_bytes
        return {"shape": tuple(value.shape), "dtype": value.dtype, "storage": self.storage_ids[key],
            "strides_elements": strides, "offset_elements": offset}

    def enter(self, rank, role, inputs):
        key = rank, role
        if key not in self.counts:
            self.counts[key] = 0
        ordinal = self.counts[key]
        self.counts[key] += 1
        if rank not in self.order:
            self.order[rank] = 0
        row = {"rank": rank, "role": role, "ordinal": ordinal, "order": self.order[rank],
            "inputs": self.freeze(rank, inputs)}
        self.order[rank] += 1
        self.rows.append(row)
        return len(self.rows) - 1

    def leave(self, index, output):
        row = self.rows[index]
        row["output"] = self.freeze(row["rank"], output)

    def bind(self, rank, role, function):
        @wraps(function)
        def call(*args):
            index = self.enter(rank, role, args)
            output = function(*args)
            self.leave(index, output)
            return output
        return call


class observe_v4_source_calls:
    """Evaluation-only call hook for a replay, never memory measurements.

    The records retain no frames or tensors. Reading frame.f_locals can
    itself retain old locals in CPython, however, so the observed replay's
    lifetimes must never feed activation/capacity prediction. The caller
    uses an unobserved trace and requires exact graph equality separately.
    """
    def __init__(self):
        from src.serving.workload import traced_models as models, traced_operators as ops
        from src.serving.workload.traced_forward import ForwardTrace, ForwardCollectiveGroup
        self.specs = {function.__code__: (role, parameters) for function, role, parameters in (
            (models.DeepseekV4Model.__call__, "model", ("input_ids", "positions")),
            (models.DeepseekV4DecoderLayer.__call__, "decoder", ("x", "positions", "input_ids", "post_mix", "res_mix", "residual")),
            (models.DeepseekV4Attention.__call__, "attention", ("positions", "hidden_states", "llama_4_scaling")),
            (models.DeepseekV4AttentionInput._run_parallel_input_projections, "input_projections", ("hidden_states",)),
            (ops.SymbolicV4QKVNorm.__call__, "qkv_norm", ("projection",)),
            (models.DeepseekV4Attention._prepare_and_attn, "prepare_attention", ("hidden_states", "qr", "kv", "qr_scale",
                "kv_score", "indexer_kv_score", "indexer_weights", "positions", "o_padded")),
            (models.DeepseekV4OutputProjection.__call__, "output_projection", ("o", "positions")),
            (ops.SymbolicV4InverseRope.__call__, "inverse_rope", ("o", "positions", "rotary_cache")),
            (ops.SymbolicV4GroupedOutput.__call__, "grouped_output_projection", ("values", "scales")),
            (ops.SymbolicBlockFP8Linear.__call__, "block_projection", ("x",)),
            (ops.SymbolicLinear.__call__, "plain_projection", ("x",)),
            (ops.SymbolicV4KVInsert.__call__, "kv_insert", ("q", "kv", "positions")),
            (ops.SymbolicV4Compressor.__call__, "compressor", ("kv_score", "positions")),
            (ops.SymbolicV4ShortTopK.__call__, "short_topk", ("positions",)),
            (ops.SymbolicV4IndexerQuery.__call__, "indexer_query", ("q", "weights", "positions")),
            (ops.SymbolicV4ScoredTopK.__call__, "scored_topk", ("q", "weights")),
            (ops.SymbolicV4SparseAttention.__call__, "sparse_attention", ("q", "positions", "out")),
            (models.DeepseekV4MoE.__call__, "ffn", ("hidden_states", "input_ids")),
            (models.DeepseekV4MoERunner.__call__, "ffn_runner", ("hidden_states", "router_logits", "input_ids")),
            (models.DeepseekV4MLP.__call__, "ffn_shared_mlp", ("x",)),
            (ops.SymbolicSiluAndMul.__call__, "ffn_shared_activation", ("x",)),
            (ops.SymbolicV4Routing.__call__, "ffn_route", ("hidden_states", "router_logits", "input_ids")),
            (ops.SymbolicV4RoutedExperts.__call__, "ffn_routed", ("x", "topk_weights", "topk_ids")),
            (ops.SymbolicAdd.__call__, "ffn_add", ("left", "right")),
            (ForwardCollectiveGroup.call, "ffn_all_reduce", ("tensor",)),
            (ops.SymbolicMHC.__call__, "mhc", ("x", "residual", "post_mix", "res_mix")),
            (ops.SymbolicVocabParallelEmbedding.__call__, "embedding", ("ids",)),
            (ops.SymbolicRMSNorm.__call__, "norm", ("x", "residual")),
            (ops.SymbolicLogitsRowSelect.__call__, "row_select", ("hidden_states", "indices")),
            (ops.SymbolicTPLogitsProcessor.__call__, "logits", ("hidden_states",)),
            (ops.SymbolicGreedy.__call__, "greedy", ("logits",)))}
        self.specs[ForwardTrace.call.__code__] = "compressor_stage", ()
        self.calls = _Calls(lambda tensor: tensor.view.allocation_id)
        self.pending = {}
        self.call_nodes = {}

    def _profile(self, frame, event, result):
        if frame.f_code not in self.specs or event not in ("call", "return"):
            return
        if event == "call":
            role, parameters = self.specs[frame.f_code]
            if role == "compressor_stage":
                node = frame.f_locals["node"]
                trace = frame.f_locals["self"]
                identifier = trace.qualify(node.node_id)
                for index in self.pending.values():
                    if index not in self.call_nodes:
                        self.call_nodes[index] = set()
                    self.call_nodes[index].add((identifier, node.node_id))
                if node.variant not in ("dsv4_compressor_state_cache", "dsv4_compressed_state_cache"):
                    return
                prefix = "indexer" if ":indexer-compressor:" in node.node_id else "main"
                stage = ("save_partial" if node.variant == "dsv4_compressor_state_cache" else
                    node.attributes["finish_stage"] if "finish_stage" in node.attributes else "finish")
                values = frame.f_locals["inputs"][:3 if stage == "save_partial" else 2]
                index = self.calls.enter(frame.f_locals["self"].rank, prefix + "_" + stage, values)
                self.calls.rows[index]["node_id"] = frame.f_locals["self"].qualify(node.node_id)
                self.call_nodes[index] = {(identifier, node.node_id)}
                if stage == "save_partial":
                    operands = {"kv": values[0], "score": values[1], "state_cache": frame.f_locals["out"]}
                    self.calls.rows[index]["operand_layouts"] = {name: None if value is None else {
                        "shape": value.shape, "dtype": value.dtype,
                        "strides_elements": tuple(stride // value.view.element_bytes for stride in value.view.strides_bytes),
                        "offset_elements": value.view.offset_bytes // value.view.element_bytes}
                        for name, value in operands.items()}
                self.pending[id(frame)] = index
                return
            if role in ("block_projection", "plain_projection"):
                roles = {":fused-wqa-wkv": "fused_qkv_projection", ":compressor-kv-score": "compressor_projection",
                    ":indexer-kv-score": "indexer_projection", ":indexer-weights": "indexer_weights_projection",
                    ":wq-b": "query_projection", ":indexer-wq-b": "indexer_query_projection",
                    ":ffn:router": "ffn_router", ":ffn:shared:gate-up": "ffn_shared_gate_up",
                    ":ffn:shared:down": "ffn_shared_down", ":attn:out:wo-b": "output_linear_projection"}
                matches = [label for suffix, label in roles.items() if frame.f_locals["self"].name.endswith(suffix)]
                if not matches:
                    return
                role, = matches
            if role in ("ffn_add", "ffn_shared_activation", "ffn_all_reduce"):
                name = frame.f_locals["name"] if role == "ffn_all_reduce" else frame.f_locals["self"].name
                if role == "ffn_all_reduce" and name.endswith(":attn:out:all-reduce"):
                    role = "output_all_reduce"
                suffix = {"ffn_add": ":ffn:shared-add", "ffn_shared_activation": ":ffn:shared:silu-mul",
                    "ffn_all_reduce": ":ffn:all-reduce", "output_all_reduce": ":attn:out:all-reduce"}[role]
                if not name.endswith(suffix):
                    return
            values = tuple(frame.f_locals[name] for name in parameters)
            rank = next(value.trace.rank for value in values if value is not None)
            if role == "mhc":
                role = frame.f_locals["self"].variant
            if role == "compressor":
                role = "indexer_compressor" if frame.f_locals["self"].name.endswith(":indexer-compressor") else "main_compressor"
            self.pending[id(frame)] = self.calls.enter(rank, role, values)
            if role == "ffn_shared_activation":
                self.calls.rows[self.pending[id(frame)]]["activation_parameters"] = {
                    "swiglu_limit": frame.f_locals["self"].swiglu_limit}
        else:
            if id(frame) not in self.pending:
                # Only explicitly selected attention input linears are observed.
                if self.specs[frame.f_code][0] in ("block_projection", "plain_projection", "compressor_stage",
                        "ffn_add", "ffn_shared_activation", "ffn_all_reduce"):
                    return
                raise ValueError("source caller observation return has no matching entry")
            index = self.pending.pop(id(frame))
            if self.specs[frame.f_code][0] == "mhc":
                self.calls.rows[index]["node_id"] = frame.f_locals["self"].trace.qualify(frame.f_locals["node"].node_id)
            self.calls.leave(index, result)

    def __enter__(self):
        if sys.getprofile() is not None:
            raise ValueError("source caller observation cannot replace an existing Python profile hook")
        sys.setprofile(self._profile)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.setprofile(None)
        if exc_type is None and self.pending:
            raise ValueError("source caller observation ended with unfinished calls")


def _parameter(fields, name):
    expression = fields.expression(name, "nn.Parameter")
    empty = expression.args[0]
    if not isinstance(empty, ast.Call) or ast.unparse(empty.func) != "torch.empty":
        raise ValueError("source mHC parameter allocation binding changed")
    dtype, = (item.value for item in empty.keywords if item.arg == "dtype")
    if ast.unparse(dtype) != "torch.float32":
        raise ValueError("source mHC parameters must remain FP32")
    sizes = tuple(fields.value(argument) for argument in empty.args)
    return _Tensor(sizes[0] if len(sizes) == 1 and isinstance(sizes[0], tuple) else sizes, "fp32")


class _UnresolvedParallelEnable:
    def __bool__(self):
        raise ValueError("source requires an undeclared multistream threshold")


class _UnusedThreshold:
    def __ge__(self, tokens):
        # aux_streams=None short-circuits the original helper before this
        # predicate is used; no arbitrary threshold or runtime value is chosen.
        return _UnresolvedParallelEnable()


class _Weight:
    def __init__(self, shape, role=None, layer=None):
        self.shape, self.role, self.layer, self.dtype = tuple(shape), role, layer, "bf16"

    @property
    def T(self):
        return _Weight(tuple(reversed(self.shape)), self.role, self.layer)


class _ForwardPort(SimpleNamespace):
    """An independent reference callable, never an installed module/provider."""
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _ShortTopKMutation:
    """Project the original kernel's written buffer as a symbolic output.

    The source kernel itself returns None. The observation adapter exposes
    its mutated active-row view, matching the symbolic mutation interface.
    No kernel execution, index values or physical launch is claimed.
    """
    def __init__(self, calls, rank):
        self.calls, self.rank = calls, rank

    def __getitem__(self, grid):
        def launch(buffer, positions, *, TOP_K, COMPRESS_RATIO, PADDED_TOP_K, num_warps):
            rows, = grid
            if (positions.shape != (rows,) or positions.dtype != "int64" or rows > buffer.shape[0]
                    or buffer.dtype != "int32" or buffer.shape[1] != TOP_K or COMPRESS_RATIO != 4
                    or PADDED_TOP_K != 1 << (TOP_K - 1).bit_length() or num_warps != 8):
                raise ValueError("source short-context top-k mutation geometry changed")
            self.calls.bind(self.rank, "short_topk", lambda pos: buffer[:rows, :])(positions)
        return launch


class _CPUValues:
    """Request-sized integer metadata for the genuine CPU boundary helper."""
    def __init__(self, values):
        self.values = tuple(values)

    def __getitem__(self, index):
        value = self.values[index]
        return _CPUValues(value) if isinstance(index, slice) else value

    def __sub__(self, other):
        return _CPUValues(left - right for left, right in zip(self.values, other.values, strict=True))

    def tolist(self):
        return list(self.values)


class _LogicalPagedCache:
    """Source requires rank3; only the declared flattened rows are bound.

    Page count/size, padding, native capacity and physical page sharing are
    deliberately absent. A source method attempting to read them fails.
    """
    ndim, dtype = 3, "uint8"

    def __init__(self, logical_rows):
        self.logical_rows = logical_rows


class _CompressorSource:
    """Genuine compressor prologue/dispatch with logical mutation ports."""
    def __init__(self, source, fields, calls, rank, role, layer, *, query_lengths, kv_lengths,
                 cache_capacity_tokens, graph_runtime_mode, partial_work, finish_work):
        self.source, self.calls, self.rank = source, calls, rank
        self.prefix = "main" if role == "main_compressor" else "indexer"
        if graph_runtime_mode not in ("NONE", "PIECEWISE", "FULL"):
            raise ValueError("source compressor requires an explicit graph runtime mode")
        rows = sum(query_lengths)
        compressor = SimpleNamespace()
        for name in ("head_dim", "compress_ratio", "coff", "overlap", "rope_head_dim", "rms_norm_eps",
                     "_quant_block", "_token_stride", "_scale_dim"):
            setattr(compressor, name, fields.value(fields.expression("self." + name)))
        state_fields = source.fields("CompressorStateCache", {"compress_ratio": compressor.compress_ratio})
        block = state_fields.value(state_fields.expression("self.block_size"))
        if (type(cache_capacity_tokens) is not int or cache_capacity_tokens < max(kv_lengths)
                or cache_capacity_tokens % compressor.compress_ratio or cache_capacity_tokens % block):
            raise ValueError("source compressor logical cache capacity must cover explicit lengths and complete blocks")
        width = compressor.coff * compressor.head_dim
        if ast.unparse(fields.expression("state_dtype")) != "torch.float32":
            raise ValueError("source compressor state/APE declaration must remain FP32")
        _, state_keywords, _ = fields.call("self.state_cache", "CompressorStateCache")
        ape = fields.expression("self.ape", "nn.Parameter").args[0]
        if (ast.unparse(state_keywords["dtype"]) != "state_dtype" or not isinstance(ape, ast.Call)
                or ast.unparse(ape.func) != "torch.empty"):
            raise ValueError("source compressor state/APE typed allocation changed")
        ape_dtype, = (item.value for item in ape.keywords if item.arg == "dtype")
        if ast.unparse(ape_dtype) != "state_dtype":
            raise ValueError("source APE allocation must retain its declared FP32 dtype")
        state = _Tensor((cache_capacity_tokens // block, block, fields.keyword(state_keywords, "state_dim")), "fp32")
        output = _Tensor((cache_capacity_tokens // compressor.compress_ratio,
            compressor._token_stride + compressor._scale_dim), "uint8")
        state_prefix, kv_prefix = f"{role}-{layer}-state", f"{role}-{layer}-kv"
        compressor.state_cache = SimpleNamespace(prefix=state_prefix, kv_cache=state)
        compressor.k_cache_prefix = kv_prefix
        compressor._static_forward_context = {kv_prefix: SimpleNamespace(kv_cache=_LogicalPagedCache(output))}
        compressor.ape = _Tensor(fields.value(ape.args[0]), "fp32")
        compressor.norm = SimpleNamespace(weight=_Weight((compressor.head_dim,)))
        compressor.use_fp4_cache, compressor._compress_scratch = False, None
        platform = SimpleNamespace(is_cuda=lambda: True, is_rocm=lambda: False, is_xpu=lambda: False)
        compressor._use_two_stage_fused_compressor = source.function("compressor.py", "_prefer_two_stage_compressor",
            metadata_globals={"current_platform": platform})()
        starts = [0]
        for length in query_lengths:
            starts.append(starts[-1] + length)
        boundary = source.function("compressor.py", "_get_c128_boundary")(SimpleNamespace(
            seq_lens_cpu_upper_bound=_CPUValues(kv_lengths), query_start_loc_cpu=_CPUValues(starts)))
        slot_mapping = SimpleNamespace(shape=(rows,))
        metadata = {state_prefix: SimpleNamespace(token_to_req_indices=object(), slot_mapping=slot_mapping,
            block_table=object(), block_size=block, c128_boundary=boundary),
            kv_prefix: SimpleNamespace(slot_mapping=object())}
        partial_program = PartialStateSource(source)
        finish_program = FinishSource(source, head_dim=compressor.head_dim, ratio=compressor.compress_ratio,
            overlap=compressor.overlap, rope_dim=compressor.rope_head_dim, quant_block=compressor._quant_block,
            record_bytes=compressor._token_stride + compressor._scale_dim,
            query_lengths=query_lengths, kv_lengths=kv_lengths)

        def finish_math(stage):
            work = finish_program.analyze(stage)
            work.update({"rank": rank, "layer": layer, "role": self.prefix + "_" + stage})
            finish_work.append(work)

        def mutation(stage, inputs, target):
            # The real kernels return None; these ports expose the written
            # logical storage, exactly as ForwardTrace.call(out=...) does.
            index = calls.enter(rank, self.prefix + "_" + stage, inputs)
            calls.leave(index, target)
            return index

        def save(**arguments):
            kv, score, positions = (arguments[name] for name in ("kv", "score", "positions"))
            if (kv.dtype != "fp32" or score.dtype != "fp32" or kv.shape != (rows, width)
                    or score.shape != kv.shape or kv.storage is not score.storage
                    or arguments["state_cache"] is not state or arguments["block_size"] != block
                    or arguments["state_width"] != width or arguments["pdl_kwargs"] != {"launch_pdl": False}):
                raise ValueError("source save-partial must retain FP32 aliased KV/score and separate producer")
            work = partial_program.analyze(arguments)
            work.update({"rank": rank, "layer": layer, "role": self.prefix + "_save_partial"})
            partial_work.append(work)
            index = mutation("save_partial", (kv, score, positions), state)
            calls.rows[index]["operand_layouts"] = work["operand_layouts"]

        def finish(**arguments):
            if arguments["state_cache"] is not state:
                raise ValueError("source finish must consume this compressor's FP32 state")
            finish_math("finish")
            mutation("finish", (state, arguments["positions"]), output)

        def store(**arguments):
            compressed = arguments["compressed_kv"]
            if compressed.dtype != "fp32" or compressed.shape != (rows, compressor.head_dim):
                raise ValueError("source C128 store requires the fresh full-row FP32 intermediate")
            finish_math("store")
            mutation("store", (compressed, arguments["positions"]), output)

        block_class = source.classes["SparseAttnCompressC128Block8Kernel"]
        state_block, = (ast.literal_eval(node.value) for node in block_class.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "state_block_size" for target in node.targets))
        block_port = _ForwardPort(state_block_size=state_block, dispatch=lambda **geometry: geometry)
        allocate = source.forward("SparseAttnCompressC128Block8Kernel", "__call__", metadata_globals={
            "torch": SimpleNamespace(float32="fp32", empty=lambda shape, *, dtype, device: _Tensor(shape, dtype))}).__get__(block_port)
        def compress(**arguments):
            _, launch_arguments, compressed = allocate(**arguments)
            if launch_arguments[0] is not state or launch_arguments[-1] is not compressed:
                raise ValueError("source C128 compress launch ports changed")
            finish_math("compress")
            return calls.bind(rank, self.prefix + "_compress", lambda cache, pos: compressed)(state, arguments["positions"])
        block_port.forward = compress
        cute_type = type("ReferenceCompressorDispatch", (), {"__call__": source.forward("SparseAttnCompressorCuteDSL", "__call__")})
        cute = cute_type()
        cute.c4_store, cute.c128_compress, cute.c128_store = finish, block_port, store
        triton = source.function("fused_compress_quant_cache.py", "compress_norm_rope_store_triton",
            metadata_globals={"_FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL": finish})
        globals = {"get_forward_context": lambda: SimpleNamespace(attn_metadata=metadata,
                cudagraph_runtime_mode=graph_runtime_mode), "cast": lambda annotation, value: value,
            "CompressorMetadata": object, "Any": object, "current_platform": platform,
            "CUDAGraphMode": SimpleNamespace(FULL="FULL"), "torch": SimpleNamespace(uint8="uint8", float8_e4m3fn="fp8"),
            "_SAVE_PARTIAL_STATES_KERNEL": save, "compress_norm_rope_store_triton": triton}
        self.forward = source.forward("DeepseekCompressor", metadata_globals=globals, metadata_imports={
            (1, "nvidia.ops.sparse_attn_compress_cutedsl", "_SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL"): cute}).__get__(compressor)


class _AttentionSource:
    """Original attention/prepare/indexer control; kernel internals are opaque."""
    def __init__(self, source, calls, config, tp_size, rank, projection_work, *, query_lengths,
                 kv_lengths, request_phases, max_batched_tokens, cache_capacity_tokens, partial_work, finish_work,
                 output_source=None, indexer_source=None, attention_math_source=None, norm_source=None,
                 input_fp8_work=None):
        self.source, self.calls, self.config, self.tp_size, self.rank = source, calls, config, tp_size, rank
        self.projection_work = projection_work
        self.partial_work = partial_work
        self.finish_work = finish_work
        self.output_source, self.rotary_tables = output_source, {}
        self.indexer_source = indexer_source
        self.attention_math_source = attention_math_source
        self.norm_source = norm_source
        self.input_fp8_work = input_fp8_work
        self.input_fp8_source = None if input_fp8_work is None else FP8LinearSource(source)
        self.query_lengths, self.kv_lengths, self.phases = query_lengths, kv_lengths, request_phases
        self.cache_capacity_tokens = cache_capacity_tokens
        self.topk_buffer = _Tensor((max_batched_tokens, config["index_topk"]), "int32")
        self.parallel = source.function("multi_stream_utils.py", "execute_in_parallel")
        self.maybe_parallel = source.function("multi_stream_utils.py", "maybe_execute_in_parallel")
        self.fused_projection = source.forward("DeepseekV4Attention", "_fused_wqa_wkv_gemm")
        self.padded = source.forward("DeepseekV4FlashMLAAttention", "get_padded_num_q_heads")

    def fp8_projection(self, layer, role, x, columns):
        if self.input_fp8_source is not None:
            self.input_fp8_work.append({"rank": self.rank, "layer": layer, "role": role,
                **self.input_fp8_source.analyze(x, columns)})

    def layer(self, layer):
        source, config, calls, rank = self.source, self.config, self.calls, self.rank
        fields = source.fields("DeepseekV4Attention", {"config": config, "layer_id": layer},
            tensor_parallel_size=self.tp_size)
        if ast.unparse(fields.expression("layer_id")) != "extract_layer_index(prefix)":
            raise ValueError("source attention layer-index binding changed")
        attention = SimpleNamespace()
        for name in ("hidden_size", "n_local_heads", "q_lora_rank", "head_dim", "compress_ratio", "eps"):
            setattr(attention, name, fields.value(fields.expression("self." + name)))
        attention.padded_heads = self.padded(None, attention.n_local_heads)
        attention.aux_stream_list, attention.ln_events = None, (None,) * 4
        args, keywords, _ = fields.call("self.fused_wqa_wkv", "MergedColumnParallelLinear")
        if (fields.keyword(keywords, "bias") or not fields.keyword(keywords, "disable_tp")):
            raise NotImplementedError("reference fused Q/KV projection requires replicated bias-free input")
        inside, outside = args[0], sum(args[1])

        def linear(x):
            if x.shape[-1] != inside or x.dtype != "bf16":
                raise ValueError("source fused Q/KV constructor/input disagree")
            self.fp8_projection(layer, "fused_qkv_projection", x, outside)
            return _Tensor((x.shape[0], outside)), None
        attention.fused_wqa_wkv = calls.bind(rank, "fused_qkv_projection", linear)
        attention._fused_wqa_wkv_gemm = self.fused_projection.__get__(attention)

        def compressor(owner_fields, role):
            _, arguments, _ = owner_fields.call("self.compressor", "DeepseekCompressor")
            params = {"config": config, "use_fp4_cache": False, **{key: owner_fields.keyword(arguments, key)
                for key in ("hidden_size", "head_dim", "compress_ratio")}}
            comp_fields = source.fields("DeepseekCompressor", params)
            arguments, kw, _ = comp_fields.call("self.fused_wkv_wgate", "MergedColumnParallelLinear")
            if (comp_fields.keyword(kw, "quant_config") is not None or not comp_fields.keyword(kw, "disable_tp")
                    or comp_fields.keyword(kw, "return_bias")):
                raise NotImplementedError("reference compressor mm requires ordinary replicated BF16 weights")
            if "params_dtype" in kw and ast.unparse(kw["params_dtype"]) not in ("None", "torch.bfloat16"):
                raise NotImplementedError("reference compressor weight storage must remain BF16")
            port_role = "main_compressor" if role == "compressor_projection" else "indexer_compressor"
            width = sum(arguments[1])
            body = _CompressorSource(source, comp_fields, calls, rank, port_role, layer,
                query_lengths=self.query_lengths, kv_lengths=self.kv_lengths,
                cache_capacity_tokens=self.cache_capacity_tokens, graph_runtime_mode="NONE", partial_work=self.partial_work,
                finish_work=self.finish_work)
            def invoke(kv_score, positions, rotary):
                if (kv_score.dtype != "fp32" or kv_score.shape != (sum(self.query_lengths), width)
                        or positions.dtype != "int64" or positions.shape != (sum(self.query_lengths),)):
                    raise ValueError("source compressor port requires FP32 projected KV/score and INT64 positions")
                return calls.bind(rank, port_role, lambda value, pos: body.forward(value, pos, rotary))(kv_score, positions)
            return _ForwardPort(fused_wkv_wgate=SimpleNamespace(weight=_Weight((width, arguments[0]), role, layer)),
                forward=invoke)

        attention.compressor = (None if attention.compress_ratio <= 1 else
            compressor(fields, "compressor_projection"))
        attention.indexer = None
        if attention.compress_ratio == 4:
            index_fields = source.fields("DeepseekV4Indexer", {"config": config, "hidden_size": attention.hidden_size,
                "q_lora_rank": attention.q_lora_rank, "compress_ratio": attention.compress_ratio})
            args, keywords, _ = index_fields.call("self.weights_proj", "ReplicatedLinear")
            if index_fields.keyword(keywords, "quant_config") is not None or index_fields.keyword(keywords, "bias"):
                raise NotImplementedError("reference indexer weights require unquantized bias-free BF16")
            index_inside, index_outside = args

            def weights(x):
                if x.shape[-1] != index_inside or x.dtype != "bf16":
                    raise ValueError("source indexer weights constructor/input disagree")
                rows = x.shape[0]
                read = 2 * (rows * index_inside + index_inside * index_outside)
                write = 2 * rows * index_outside
                self.projection_work.append({"rank": rank, "layer": layer, "role": "indexer_weights_projection",
                    "shape": {"b": 1, "m": rows, "n": index_outside, "k": index_inside},
                    "input_dtype": "bf16", "weight_dtype": "bf16", "output_dtype": "bf16",
                    "tensor_flops": 2 * rows * index_inside * index_outside,
                    "read_bytes": read, "write_bytes": write, "compulsory_bytes": read + write})
                return _Tensor((x.shape[0], index_outside)), None
            attention.indexer = _ForwardPort(
                weights_proj=calls.bind(rank, "indexer_weights_projection", weights),
                compressor=compressor(index_fields, "indexer_projection"))

        def mm(x, weight, *, out_dtype):
            if x.dtype != "bf16" or weight.dtype != "bf16" or out_dtype != "fp32" or x.shape[-1] != weight.shape[0]:
                raise ValueError("source compressor projection must keep BF16 A/B and FP32 C")
            m, k, n = x.shape[0], x.shape[1], weight.shape[1]
            self.projection_work.append({"rank": rank, "layer": weight.layer, "role": weight.role,
                "shape": {"b": 1, "m": m, "n": n, "k": k}, "input_dtype": x.dtype,
                "weight_dtype": weight.dtype, "output_dtype": out_dtype,
                "tensor_flops": 2 * m * n * k,
                "source_a_bytes": m * k * 2, "source_b_bytes": k * n * 2, "write_bytes": m * n * 4,
                "read_bytes": (m * k + k * n) * 2,
                "compulsory_bytes": (m * k + k * n) * 2 + m * n * 4})
            return calls.bind(rank, weight.role, lambda value: (_Tensor((value.shape[0], weight.shape[1]), "fp32"), None))(x)[0]

        def norm(qr, kv, qr_weight, kv_weight, epsilon):
            if qr.shape[-1] != qr_weight.shape[0] or kv.shape[-1] != kv_weight.shape[0] or epsilon <= 0:
                raise ValueError("source fused Q/KV norm constructor/input dimensions disagree")
            if self.norm_source is not None:
                return self.norm_source.run(rank, layer, qr, kv, qr_weight, kv_weight, epsilon)
            return _Tensor(qr.shape, qr.dtype), _Tensor(kv.shape, kv.dtype)
        for name in ("q_norm", "kv_norm"):
            norm_args, _, _ = fields.call("self." + name, "RMSNorm")
            setattr(attention, name, SimpleNamespace(weight=SimpleNamespace(data=_Tensor((norm_args[0],), "bf16"))))
        globals = {"torch": SimpleNamespace(mm=mm, float32="fp32",
                empty=lambda shape, *, dtype, device: _Tensor(shape, dtype)),
            "execute_in_parallel": self.parallel,
            "envs": SimpleNamespace(VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=_UnusedThreshold()),
            "fused_q_kv_rmsnorm": norm}
        # Compile each helper with metadata callbacks, never import a torch
        # module or install a fake runtime/SDK. The original no-aux helper
        # executes its own sequential branch before touching CUDA objects.
        project = source.forward("DeepseekV4Attention", "_run_parallel_input_projections", metadata_globals=globals).__get__(attention)
        split = source.forward("DeepseekV4Attention", "_split_qkv_and_norm", metadata_globals=globals).__get__(attention)
        attention._run_parallel_input_projections = calls.bind(rank, "input_projections", project)
        attention._split_qkv_and_norm = calls.bind(rank, "qkv_norm", split)
        self.prepare(attention, fields)
        if self.output_source is None:
            def unavailable_output(*args):
                raise ValueError("input-only source reference has no output-projection program")
            attention._o_proj = unavailable_output
        else:
            attention._o_proj = self.output_source.layer(layer, fields, attention.rotary_emb.cos_sin_cache)
        forward = source.forward("DeepseekV4Attention", metadata_globals=globals).__get__(attention)
        return calls.bind(rank, "attention", forward)

    def prepare(self, attention, fields):
        """Bind genuine prepare/indexer methods to explicit metadata ports.

        Only request lengths/phases select the indexer short/scored branch.
        Rotary/cache storage, mutation dependencies and primitive math still
        need separate comparison; they are excluded from these caller ports.
        """
        source, calls, rank = self.source, self.calls, self.rank
        args, keywords, _ = fields.call("self.wq_b", "ColumnParallelLinear")
        if fields.keyword(keywords, "bias") or fields.keyword(keywords, "return_bias"):
            raise NotImplementedError("source query projection requires bias-free tensor return")
        inside, outside = args
        if outside % self.tp_size:
            raise ValueError("source query projection is not evenly TP-sharded")
        def query(x):
            if x.dtype != "bf16" or x.shape[-1] != inside:
                raise ValueError("source Q projection input differs from its constructor")
            self.fp8_projection(fields.parameters["layer_id"], "query_projection", x, outside // self.tp_size)
            return _Tensor((x.shape[0], outside // self.tp_size)), None
        query_port = calls.bind(rank, "query_projection", query)
        attention.wq_b = lambda x: query_port(x)[0]
        attention._wq_b_proj = source.forward("DeepseekV4Attention", "_wq_b_proj").__get__(attention)
        # These objects only identify the opaque rotary argument; no table
        # shape, sharing or numerical rotation is inferred by this reference.
        rope_kind = "swa" if attention.compress_ratio <= 1 else "compressed"
        if rope_kind not in self.rotary_tables:
            self.rotary_tables[rope_kind] = _Tensor((self.config["max_position_embeddings"], self.config["qk_rope_head_dim"]), "fp32")
        attention.rotary_emb = SimpleNamespace(cos_sin_cache=self.rotary_tables[rope_kind])
        attention.indexer_rotary_emb = attention.rotary_emb
        insert = calls.bind(rank, "kv_insert", lambda q, kv, pos:
            _Tensor((q.shape[0], attention.padded_heads, attention.head_dim), q.dtype))
        attention._fused_qnorm_rope_kv_insert = lambda q, kv, positions, metadata: insert(q, kv, positions)
        metadata = {}
        globals = {"get_forward_context": lambda: SimpleNamespace(attn_metadata=metadata),
            "execute_in_parallel": self.parallel, "maybe_execute_in_parallel": self.maybe_parallel}
        if attention.indexer is not None:
            indexer = attention.indexer
            index_fields = source.fields("DeepseekV4Indexer", {"config": self.config,
                "hidden_size": attention.hidden_size, "q_lora_rank": attention.q_lora_rank,
                "compress_ratio": attention.compress_ratio})
            for name in ("topk_tokens", "n_head", "head_dim", "compress_ratio", "softmax_scale"):
                setattr(indexer, name, index_fields.value(index_fields.expression("self." + name)))
            indexer.use_fp4_kv, indexer.aux_stream, indexer.ln_events = False, None, (None, None)
            indexer.topk_indices_buffer = self.topk_buffer
            indexer.k_cache = SimpleNamespace(prefix="declared-indexer-metadata")
            metadata[indexer.k_cache.prefix] = SimpleNamespace(max_seq_len=max(self.kv_lengths),
                num_decode_tokens=sum(q for q, phase in zip(self.query_lengths, self.phases, strict=True) if phase == "decode"),
                num_prefill_tokens=sum(q for q, phase in zip(self.query_lengths, self.phases, strict=True) if phase == "prefill"))
            args, keywords, _ = index_fields.call("self.wq_b", "ReplicatedLinear")
            if index_fields.keyword(keywords, "bias"):
                raise NotImplementedError("source indexer query requires a bias-free replicated projection")
            index_inside, index_outside = args
            def index_query(x):
                if x.dtype != "bf16" or x.shape[-1] != index_inside:
                    raise ValueError("source indexer Q projection input differs from constructor")
                self.fp8_projection(fields.parameters["layer_id"], "indexer_query_projection", x, index_outside)
                return _Tensor((x.shape[0], index_outside)), None
            indexer.wq_b = calls.bind(rank, "indexer_query_projection", index_query)
            indexer._wq_b_proj = source.forward("DeepseekV4Indexer", "_wq_b_proj").__get__(indexer)
            quant = calls.bind(rank, "indexer_query", lambda q, weights, pos:
                (_Tensor(q.shape, "fp8"), _Tensor(weights.shape, "fp32")))
            def quantize(pos, q, rotary, weights, scale, head_scale, *, use_fp4):
                if use_fp4 or scale != indexer.softmax_scale or head_scale != indexer.n_head ** -0.5:
                    raise NotImplementedError("source query metadata port requires the declared FP8 scale branch")
                return quant(q, weights, pos)
            indexer.forward = source.forward("DeepseekV4Indexer", metadata_globals={**globals,
                "cast": lambda annotation, value: value, "Any": object,
                "torch": SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: False)),
                "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
                "_fill_short_context_topk_indices": _ShortTopKMutation(calls, rank),
                "fused_indexer_q_rope_quant": quantize}).__get__(indexer)
            def score(q, weights):
                if self.indexer_source is not None:
                    self.indexer_source.record(rank, fields.parameters["layer_id"],
                        self.query_lengths, self.kv_lengths, self.phases)
                return None
            scored = calls.bind(rank, "scored_topk", score)
            indexer.indexer_op = lambda hidden, q, unused, weights: scored(q, weights)
        def attention_math(q, positions, out):
            if self.attention_math_source is not None:
                self.attention_math_source.record(rank, fields.parameters["layer_id"], attention.compress_ratio,
                    q.shape[1], self.query_lengths, self.kv_lengths, self.phases)
            return None
        sparse = calls.bind(rank, "sparse_attention", attention_math)
        attention.forward_mqa = lambda q, kv, positions, out: sparse(q, positions, out)
        attention._sparse_indexer_and_attn = source.forward("DeepseekV4Attention", "_sparse_indexer_and_attn").__get__(attention)
        prepare = source.forward("DeepseekV4Attention", "_prepare_and_attn", metadata_globals=globals).__get__(attention)
        attention._prepare_and_attn_fn = calls.bind(rank, "prepare_attention", prepare)


def v4_model_source_reference(upstream_root, config_path, *, query_lengths, kv_lengths, request_phases,
                              max_batched_tokens, cache_capacity_tokens, tensor_parallel_size, input_token_ids,
                              indexer_selection_source=None, source_use_deep_gemm=None, multiprocessors=None,
                              torch_include=None, flashinfer_source_root=None, source_max_model_len=None):
    """Execute original full-layer model/decoder control flow independently.

    Expected ports and mixed-mm work do not invoke production model
    adapters, builders, lowering, predictors or tensor/runtime libraries.
    Compressor state/scratch IO, save work and active-value finish math are
    source-derived. Physical pages, metadata traffic, replicated issue work
    and numerical output remain outside those logical projections.
    """
    config = json.loads(Path(config_path).read_text())
    if (not query_lengths or len(query_lengths) != len(kv_lengths) or len(query_lengths) != len(request_phases)
            or any(type(q) is not int or type(kv) is not int or q <= 0 or kv < q
                for q, kv in zip(query_lengths, kv_lengths, strict=True))
            or any(phase not in ("prefill", "decode") for phase in request_phases)
            or any(q != 1 for q, phase in zip(query_lengths, request_phases, strict=True) if phase == "decode")
            or type(max_batched_tokens) is not int or max_batched_tokens < sum(query_lengths)):
        raise ValueError("V4 caller reference requires explicit aligned query/KV lengths, phases and batch capacity")
    tokens, requests = sum(query_lengths), len(query_lengths)
    if (config["model_type"] != "deepseek_v4" or config["torch_dtype"] != "bfloat16"
            or tensor_parallel_size not in (4, 8) or type(tokens) is not int or tokens <= 0
            or type(requests) is not int or not 0 < requests <= tokens):
        raise ValueError("V4 caller reference requires BF16 TP4/8 and explicit positive token/request counts")
    source = _Source(upstream_root, model_sources={
        "model.py": "vllm/models/deepseek_v4/nvidia/model.py",
        "interfaces.py": "vllm/model_executor/models/interfaces.py",
        "attention.py": "vllm/models/deepseek_v4/attention.py",
        "compressor.py": "vllm/models/deepseek_v4/compressor.py",
        "sparse_attn_compress_cutedsl.py": "vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py",
        "fused_compress_quant_cache.py": "vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py",
        "save_partial_states.py": "vllm/models/deepseek_v4/common/ops/save_partial_states.py",
        "flashmla.py": "vllm/models/deepseek_v4/nvidia/flashmla.py",
        "multi_stream_utils.py": "vllm/utils/multi_stream_utils.py", **FFN_SOURCE_FILES, **OUTPUT_SOURCE_FILES,
        **INDEXER_SOURCE_FILES, **MHC_SOURCE_FILES, **NORM_SOURCE_FILES, **TARGET_SOURCE_FILES, **DECODE_STORAGE_SOURCE_FILES,
        **SPARSE_STORAGE_SOURCE_FILES})
    source.load_target_sources()
    source.validate_row_selection()
    calls = _Calls(lambda tensor: tensor.storage)
    fields = source.fields("DeepseekV4DecoderLayer", {"config": config})
    h = fields.value(fields.expression("self.hidden_size"))
    c = fields.value(fields.expression("self.hc_mult"))
    norm_args, _, _ = fields.call("self.attn_norm", "RMSNorm")
    if norm_args[0] != h:
        raise ValueError("source decoder normalization width changed")
    model_fields = source.fields("DeepseekV4Model", {"config": config})
    factory, = (node for node in ast.walk(model_fields.function) if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "make_layers")
    layers = model_fields.value(factory.args[0])
    if not isinstance(factory.args[1], ast.Lambda) or not isinstance(factory.args[1].body, ast.Call):
        raise ValueError("source model layer factory is unsupported")
    if ast.unparse(factory.args[1].body.func) != "DeepseekV4DecoderLayer":
        raise ValueError("source model decoder factory changed")
    embedding_args, _, _ = model_fields.call("self.embed_tokens", "VocabParallelEmbedding")
    model_norm_args, _, _ = model_fields.call("self.norm", "RMSNorm")
    metadata_globals = {"torch": SimpleNamespace(float32="fp32", int32="int32", int64="int64"),
        "SamplerOutput": SimpleNamespace}

    projection_work, partial_work, finish_work, ffn_linear_work, ffn_routed_work, fp8_work, vector_work = [], [], [], [], [], [], []
    output_matrix_work, inverse_work, output_fp8_work, input_fp8_work = [], [], [], []
    indexer_work = []
    indexer_source = IndexerWorkSource(source, config, indexer_work)
    attention_math_work = []
    attention_math_source = AttentionMathSource(source, config, attention_math_work,
        indexer_selection_source=indexer_selection_source)
    mhc_work = []
    qkv_norm_work = []
    final_norm_work = []
    target_work = []
    routing_object_work = []
    norm_source = QKVNormSource(source, qkv_norm_work)
    final_norm_source = FinalRMSNormSource(source, final_norm_work)
    for rank in range(tensor_parallel_size):
        mhc_source = MHCSource(source, calls, config, rank, mhc_work, multiprocessors=multiprocessors,
            source_use_deep_gemm=source_use_deep_gemm)
        output_source = OutputProjectionSource(source, calls, rank, tensor_parallel_size, output_matrix_work, inverse_work, output_fp8_work)
        attention_source = _AttentionSource(source, calls, config, tensor_parallel_size, rank, projection_work,
            query_lengths=query_lengths, kv_lengths=kv_lengths, request_phases=request_phases,
            max_batched_tokens=max_batched_tokens, cache_capacity_tokens=cache_capacity_tokens, partial_work=partial_work,
            finish_work=finish_work, output_source=output_source, indexer_source=indexer_source,
            attention_math_source=attention_math_source, norm_source=norm_source, input_fp8_work=input_fp8_work)
        ffn_source = MoESource(source, calls, config, tensor_parallel_size, rank, ffn_linear_work, ffn_routed_work, fp8_work, vector_work)
        globals_for_model = {**metadata_globals,
            "mhc_pre_broadcast_tilelang": mhc_source.broadcast,
            "mhc_fused_post_pre_tilelang": mhc_source.fused,
            "mhc_post_tilelang": mhc_source.post,
            "hc_head_fused_kernel_tilelang": mhc_source.head}
        model_type = type("ReferenceV4Model", (), {"__call__": source.forward("DeepseekV4Model", metadata_globals=globals_for_model)})
        decoder_type = type("ReferenceV4Decoder", (), {"__call__": source.forward("DeepseekV4DecoderLayer", metadata_globals=globals_for_model)})
        model = model_type()
        model.use_mega_moe, model.use_sequence_parallel, model._mtp_hidden_buffer = False, False, None
        model.start_layer, model.end_layer = 0, layers
        model.aux_hidden_state_layers, model._aux_upstream_total_cached = (), 0
        model.rms_norm_eps, model.hc_eps = config["rms_norm_eps"], config["hc_eps"]
        for name in ("hc_head_fn", "hc_head_scale", "hc_head_base"):
            setattr(model, name, _parameter(model_fields, "self." + name))
        model.embed_input_ids = source.forward("DeepseekV4Model", "embed_input_ids").__get__(model)
        model.collect_remote_aux_hidden_states = source.forward("EagleModelMixin", "collect_remote_aux_hidden_states").__get__(model)
        model.embed_tokens = calls.bind(rank, "embedding", lambda ids: _Tensor((tokens, embedding_args[1])))
        model.layers = []
        for layer in range(layers):
            decoder = decoder_type()
            decoder.use_sequence_parallel = False
            for name in ("rms_norm_eps", "hc_mult", "hc_sinkhorn_iters", "hc_eps", "hc_post_alpha"):
                setattr(decoder, name, fields.value(fields.expression("self." + name)))
            decoder.hc_attn_fn_broadcast = _Tensor((c * (c + 2), h), "fp32")
            for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"):
                setattr(decoder, name, _parameter(fields, "self." + name))
            weight = SimpleNamespace(data=_Tensor((h,), "bf16"))
            decoder.attn_norm = decoder.ffn_norm = SimpleNamespace(weight=weight, variance_epsilon=norm_args[1])
            decoder.attn = attention_source.layer(layer)
            decoder.ffn = ffn_source.layer(layer)
            model.layers.append(calls.bind(rank, "decoder", decoder))
        def normalize(x, residual):
            if residual is not None or x.shape[-1] != model_norm_args[0]:
                raise ValueError("reference final norm constructor/input width disagrees")
            return final_norm_source.run(rank, x, model_norm_args[0], model_norm_args[1])
        model.norm = calls.bind(rank, "norm", normalize)
        # Normalize the optional residual at the boundary just like the actual
        # SymbolicRMSNorm signature, without adding a wrapper to production.
        norm = model.norm
        model.norm = lambda x: norm(x, None)
        hidden = calls.bind(rank, "model", lambda ids, positions: model(ids, positions, None))(
            _Tensor((tokens,), "int32"), _Tensor((tokens,), "int64"))
        routing_object_work.extend(ffn_source.routing_object_work)
        target = TargetSource(source, calls, config, tensor_parallel_size, rank, target_work, query_lengths)
        selected = calls.bind(rank, "row_select", target.select)(
            hidden, _Tensor((requests,), "int32"))
        logits = calls.bind(rank, "logits", target.logits)(selected)
        calls.bind(rank, "greedy", target.greedy)(logits)
    def binding(file, cls, method, callee=None):
        function = source.method(cls, method)
        lines = ([function.lineno] if callee is None else [node.lineno for node in ast.walk(function)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == callee])
        if not lines:
            raise ValueError("source caller binding is missing: " + cls + "." + method + ":" + str(callee))
        return {"path": str(source.paths[file]), "symbol": cls + "." + method,
            "callee": callee, "lines": lines}

    def function_binding(file, name, callee):
        function, = (node for node in source.trees[file].body
            if isinstance(node, ast.FunctionDef) and node.name == name)
        lines = [node.lineno for node in ast.walk(function)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == callee]
        if not lines:
            raise ValueError("source output call is missing: " + name + ":" + callee)
        return {"path": str(source.paths[file]), "symbol": name, "callee": callee, "lines": lines}

    bindings = {"model": binding("model.py", "DeepseekV4Model", "forward"),
        "decoder": binding("model.py", "DeepseekV4Model", "forward", "layer"),
        "mhc_pre": binding("model.py", "DeepseekV4DecoderLayer", "forward", "mhc_pre_broadcast_tilelang"),
        "mhc_post_pre": binding("model.py", "DeepseekV4DecoderLayer", "forward", "mhc_fused_post_pre_tilelang"),
        "attention": binding("model.py", "DeepseekV4DecoderLayer", "forward", "self.attn"),
        "input_projections": binding("attention.py", "DeepseekV4Attention", "_run_parallel_input_projections"),
        "fused_qkv_projection": binding("attention.py", "DeepseekV4Attention", "_fused_wqa_wkv_gemm", "self.fused_wqa_wkv"),
        "compressor_projection": binding("attention.py", "DeepseekV4Attention", "_run_parallel_input_projections", "torch.mm"),
        "indexer_projection": binding("attention.py", "DeepseekV4Attention", "_run_parallel_input_projections", "torch.mm"),
        "indexer_weights_projection": binding("attention.py", "DeepseekV4Attention", "_run_parallel_input_projections", "indexer.weights_proj"),
        "qkv_norm": binding("attention.py", "DeepseekV4Attention", "_split_qkv_and_norm"),
        "prepare_attention": binding("attention.py", "DeepseekV4Attention", "forward", "self._prepare_and_attn_fn"),
        "query_projection": binding("attention.py", "DeepseekV4Attention", "_wq_b_proj", "self.wq_b"),
        "kv_insert": binding("attention.py", "DeepseekV4Attention", "_prepare_and_attn", "self._fused_qnorm_rope_kv_insert"),
        "main_compressor": binding("attention.py", "DeepseekV4Attention", "_prepare_and_attn", "compressor"),
        "indexer_compressor": binding("attention.py", "DeepseekV4Indexer", "forward", "compressor"),
        "short_topk": binding("attention.py", "DeepseekV4Indexer", "forward"),
        "indexer_query_projection": binding("attention.py", "DeepseekV4Indexer", "_wq_b_proj", "self.wq_b"),
        "indexer_query": binding("attention.py", "DeepseekV4Indexer", "forward", "fused_indexer_q_rope_quant"),
        "scored_topk": binding("attention.py", "DeepseekV4Attention", "_sparse_indexer_and_attn", "self.indexer.indexer_op"),
        "sparse_attention": binding("attention.py", "DeepseekV4Attention", "_sparse_indexer_and_attn", "self.forward_mqa"),
        **{prefix + "_save_partial": binding("compressor.py", "DeepseekCompressor", "forward", "_SAVE_PARTIAL_STATES_KERNEL")
            for prefix in ("main", "indexer")},
        **{prefix + "_finish": binding("compressor.py", "DeepseekCompressor", "forward", "compress_norm_rope_store_fn")
            for prefix in ("main", "indexer")},
        "main_compress": binding("sparse_attn_compress_cutedsl.py", "SparseAttnCompressorCuteDSL", "__call__", "self.c128_compress"),
        "main_store": binding("sparse_attn_compress_cutedsl.py", "SparseAttnCompressorCuteDSL", "__call__", "self.c128_store"),
        "save_partial_program": binding("save_partial_states.py", "SavePartialStatesKernel", "kernel"),
        "output_projection": binding("attention.py", "DeepseekV4Attention", "forward", "self._o_proj"),
        "inverse_rope": function_binding("o_proj.py", "deep_gemm_fp8_o_proj", "fused_inv_rope_fp8_quant"),
        "grouped_output_projection": function_binding("o_proj.py", "deep_gemm_fp8_o_proj", "fp8_einsum"),
        "output_linear_projection": function_binding("o_proj.py", "deep_gemm_fp8_o_proj", "wo_b"),
        "output_all_reduce": binding("linear.py", "RowParallelLinear", "forward", "tensor_model_parallel_all_reduce"),
        "ffn": binding("model.py", "DeepseekV4DecoderLayer", "forward", "self.ffn"),
        "ffn_runner": binding("model.py", "DeepseekV4MoE", "_forward_fused_moe", "self.experts"),
        "ffn_router": binding("moe_runner.py", "MoERunner", "_forward_impl", "self.gate"),
        "ffn_shared_mlp": binding("shared_experts.py", "SharedExperts", "forward", "self._layer"),
        "ffn_shared_gate_up": binding("model.py", "DeepseekV4MLP", "forward", "self.gate_up_proj"),
        "ffn_shared_activation": binding("model.py", "DeepseekV4MLP", "forward", "self.act_fn"),
        "ffn_shared_down": binding("model.py", "DeepseekV4MLP", "forward", "self.down_proj"),
        "ffn_route": binding("moe_runner.py", "MoERunner", "_apply_quant_method", "self.router.select_experts"),
        "ffn_routed": binding("moe_runner.py", "MoERunner", "_apply_quant_method", "self.routed_experts.forward_modular"),
        "ffn_add": binding("moe_runner.py", "MoERunner", "forward"),
        "ffn_all_reduce": binding("moe_runner.py", "MoERunner", "_maybe_reduce_final_output", "tensor_model_parallel_all_reduce"),
        "mhc_post": binding("model.py", "DeepseekV4Model", "forward", "mhc_post_tilelang"),
        "hc_head": binding("model.py", "DeepseekV4Model", "forward", "hc_head_fused_kernel_tilelang"),
        "embedding": binding("model.py", "DeepseekV4Model", "embed_input_ids", "self.embed_tokens"),
        "norm": binding("model.py", "DeepseekV4Model", "forward", "self.norm"),
        "row_select": binding("gpu_model_runner.py", "GPUModelRunner", "execute_model"),
        "logits": binding("model.py", "DeepseekV4ForCausalLM", "compute_logits", "self.logits_processor"),
        "greedy": binding("sampler.py", "Sampler", "forward")}
    result = {"calls": calls.rows, "source_bindings": bindings, "mixed_projection_work": projection_work,
        "partial_state_work": partial_work,
        "finish_work": finish_work,
        "ffn_linear_work": ffn_linear_work, "ffn_routed_work": ffn_routed_work,
        "ffn_marlin_typed_source_binding": ffn_source.marlin.bindings,
        "ffn_fp8_dataflow_work": fp8_work,
        "input_fp8_work": input_fp8_work,
        "ffn_vector_work": vector_work, "ffn_vector_source_binding": ffn_source.vector.source_binding,
        "output_matrix_work": output_matrix_work, "inverse_rope_work": inverse_work, "output_fp8_work": output_fp8_work,
        "indexer_work": indexer_work, "indexer_work_source_binding": indexer_source.binding,
        "attention_math_work": attention_math_work, "attention_math_source_binding": attention_math_source.binding,
        "mhc_work": mhc_work,
        "qkv_norm_work": qkv_norm_work,
        "final_norm_work": final_norm_work,
        "target_work": target_work,
        "routing_object_work": routing_object_work,
        "argmax_source_work": None if torch_include is None else argmax_source_reference(torch_include, target_work),
        "source_paths": {name: str(path) for name, path in source.paths.items()},
        "source_methods": source.bindings, "layers": layers,
        "scope": "complete source layer-stack and declared logical primitive programs; per-family work, ports and limitations are recorded in their own accounts",
        "declared_inputs": {"TP": tensor_parallel_size, "PP": 1, "SP": 1, "EP": False, "MTP": False,
            "aux_hidden": False, "mega_moe": False, "all_greedy": True,
            "attention_aux_streams": None, "cuda_graph_capture": False,
            "shared_experts_aux_stream": None, "dbo": False, "shared_experts": 1,
            "fused_output_already_reduced": False, "gate_provider_bound": False,
            "routed_activation_gemm1_overrides": {"clamp_limit": None, "alpha": None, "beta": None},
            "activation_profile": "BF16 source clamp, alpha1/beta0; scalar and packed forms retained, no numerical equivalence assumed",
            "query_lengths": query_lengths, "kv_lengths": kv_lengths, "request_phases": request_phases,
            "indexer_selection_source": indexer_selection_source,
            "source_use_deep_gemm": source_use_deep_gemm, "specification_multiprocessors": multiprocessors,
            "source_max_model_len": source_max_model_len,
            "indexer_decode_topk_backend": "auto",
            "max_batched_tokens": max_batched_tokens,
            "cache_capacity_tokens": cache_capacity_tokens, "compressor_graph_runtime_mode": "NONE",
            "cache_projection": "declared logical rows only; source rank3 requirement retained, physical page axes unbound",
            "compressor_mutation_projection": "source kernels return None; ports expose written logical storage",
            "partial_state_scope": "genuine wrapper strides and abstract kernel FP32 additions/masked payload; all active slots valid/nonnegative, INT64 metadata, no chosen data values or physical address",
            "short_topk_mutation_projection": "source returns None; expose written active-row buffer as symbolic output",
            "multistream_threshold": "unresolved_irrelevant_in_original_no_aux_helper"},
        "unchecked": ("complete join of source-internal temporary allocation/alias/lifetime receipts to automatic storage records",
            "consecutive logical request/cache state requires a separate explicit sequence receipt"),
        "outside_first_stage": ("actual scheduled physical metadata/block tables, provider/compiler/private workspace and numerical outputs",
            "comparison/index instruction issue cost, normal capacity and MTP"),
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}
    if flashinfer_source_root is not None:
        result["fp8_quantizer_reference"] = fp8_quantizer_reference(result, source.root, flashinfer_source_root)
    result["prepare_work_reference"] = prepare_work_reference(config, source.root, tokens=tokens,
        kv_lengths=kv_lengths, token_ids=input_token_ids, tp=tensor_parallel_size)
    result["terminal_storage_work"] = terminal_storage_reference(source, config, result)
    result["decode_index_storage_work"] = decode_index_storage_reference(source, config, result["declared_inputs"])
    result["expert_storage_work"] = expert_storage_reference(source.root, config, result["declared_inputs"])
    result["indexer_decode_storage_work"] = indexer_decode_storage_reference(source, config, result["declared_inputs"])
    result["sparse_prefill_storage_work"] = sparse_prefill_storage_reference(source, config, result["declared_inputs"])
    result["sparse_decode_view_work"] = sparse_decode_view_reference(source, config, result["declared_inputs"])
    return result


def compare_v4_input_projection_work(reference, plan, hardware):
    """Compare actual owner demands with independent source mm work.

    The expected side has already been generated exclusively from source
    calls/constructor geometry. Only the actual side consumes production
    KernelSpecs/resource inputs. No duration or predicted time is read.
    """
    from src.kernel.kernel_base import create_predictor
    from src.kernel.kernel_spec import KernelKind
    from src.kernel.predictors.gemm import bf16_fp32_gemm_logical_work, unquantized_bf16_gemm_logical_work
    from src.serving.kernel_lowering.kernel_plan import hardware_for_call
    expected = {(row["rank"], row["layer"], row["role"]): row for row in reference["mixed_projection_work"]}
    if len(expected) != len(reference["mixed_projection_work"]):
        raise ValueError("source mixed projection work must retain distinct invocation identities")
    actual = {}
    for call in plan.calls:
        matches = [re.search(r"(?:^|:)layer-(\d+):attn:(compressor-kv-score|indexer-kv-score|indexer-weights)(?::\d+)?$", name)
            for name in call.source_node_ids]
        matches = [match for match in matches if match is not None]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError("source mm comparison cannot collapse multiple projections into one kernel")
        match = matches[0]
        role = {"compressor-kv-score": "compressor_projection", "indexer-kv-score": "indexer_projection",
            "indexer-weights": "indexer_weights_projection"}[match[2]]
        key = call.rank, int(match[1]), role
        if key in actual:
            raise ValueError("actual mixed projection has duplicate lowered calls")
        kernel = call.kernel_spec
        output_dtype = kernel.dtype if kernel.attributes["output_dtype"] is None else kernel.attributes["output_dtype"]
        if (kernel.kind is not KernelKind.GEMM or kernel.variant != "gemm" or kernel.dtype != "bf16"
                or output_dtype != ("bf16" if role == "indexer_weights_projection" else "fp32")):
            actual[key] = {"valid_contract": False, "kind": kernel.kind.value,
                "input_dtype": kernel.dtype, "call_id": call.call_id}
            continue
        if kernel.launch is None and output_dtype == "fp32":
            work = bf16_fp32_gemm_logical_work(kernel)
            compulsory = work["logical_bytes"]
            traffic = {"read": compulsory - work["write_bytes"], "write": work["write_bytes"]}
            tensor_flops = work["demand"]["compute_tensor"]
            native_bound = False
        elif (kernel.launch is None and "mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] == "unquantized_bf16_flops_bytes_v1"):
            work = unquantized_bf16_gemm_logical_work(kernel)
            traffic = {access: sum(item["bytes"] for item in work["operands"] if item["access"] == access)
                for access in ("read", "write")}
            compulsory = sum(traffic.values())
            tensor_flops = work["mathematical_tensor_flops"]
            native_bound = False
        else:
            owner_input, compulsory = create_predictor(kernel.kind).resource_input(kernel, hardware_for_call(call, hardware))
            traffic = {kind: sum(node.bytes for path in owner_input.memory_paths if path.access_kind == kind
                for node in path.nodes if node.resource_role == "dram") for kind in ("read", "write")}
            tensor_flops = owner_input.demand_per_resource["compute_tensor"]
            native_bound = kernel.execution_support == "supported" and kernel.launch is not None
        actual[key] = {"valid_contract": True, "shape": {name: kernel.shape[name] for name in ("b", "m", "n", "k")},
            "input_dtype": kernel.dtype, "weight_dtype": (kernel.dtype if role == "indexer_weights_projection"
                else kernel.attributes["weight_dtype"]),
            "output_dtype": output_dtype,
            "tensor_flops": tensor_flops, "native_resource_program_bound": native_bound,
            "read_bytes": traffic["read"], "write_bytes": traffic["write"], "compulsory_bytes": compulsory,
            "call_id": call.call_id}
    checks = []
    fields = ("shape", "input_dtype", "weight_dtype", "output_dtype", "tensor_flops", "read_bytes", "write_bytes", "compulsory_bytes")
    for key in sorted(set(expected) | set(actual)):
        if key not in expected or key not in actual:
            differences = ["unexpected" if key not in expected else "missing"]
        elif not actual[key]["valid_contract"]:
            differences = ["mixed_output_contract"]
        else:
            differences = [name for name in fields if expected[key][name] != actual[key][name]]
        checks.append({"rank": key[0], "layer": key[1], "role": key[2],
            "status": "matched" if not differences else "mismatched", "differences": differences,
            "expected": expected[key] if key in expected else None, "actual": actual[key] if key in actual else None,
            "source": reference["source_bindings"][key[2]] if key in expected else None})
    return {"scope": "unbatched BF16 A/B to FP32 compressor C or BF16 indexer-weight C, source geometry, 2MNK and unique typed payload",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "individual_actual_operand_read_split_exported": False,
        "limitations": ("actual owner exposes combined A+B reads and C writes, not a per-operand traffic split",
            "library replay, local operands, epilogue and allocator internals excluded", "not observed DRAM or provider/launch acceptance"),
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}


def compare_v4_partial_state_work(reference, plan, hardware):
    """Compare genuine source lane work/layout with actual owner demands."""
    from src.kernel.kernel_base import create_predictor
    from src.kernel.kernel_spec import KernelKind
    from src.kernel.predictors.kv_cache_update import dsv4_partial_state_tensor_objects
    from src.serving.kernel_lowering.kernel_plan import hardware_for_call

    expected = {(row["rank"], row["layer"], row["role"]): row for row in reference["partial_state_work"]}
    if len(expected) != len(reference["partial_state_work"]):
        raise ValueError("source partial-state work must retain distinct invocation identities")
    actual = {}
    for call in plan.calls:
        kernel = call.kernel_spec
        if kernel.variant != "dsv4_compressor_state_cache":
            continue
        matches = [re.search(r"(?:^|:)layer-(\d+):attn:(compressor|indexer-compressor):save-partial(?::\d+)?$", name)
            for name in call.source_node_ids]
        matches = [match for match in matches if match is not None]
        if len(matches) != 1:
            raise ValueError("partial-state work requires one identifiable source save per lowered call")
        match = matches[0]
        key = call.rank, int(match[1]), "main_save_partial" if match[2] == "compressor" else "indexer_save_partial"
        if key in actual:
            raise ValueError("actual partial-state work has duplicate lowered calls")
        if kernel.kind is not KernelKind.KV_CACHE or kernel.dtype != "fp32":
            actual[key] = {"valid_contract": False, "kind": kernel.kind.value, "dtype": kernel.dtype}
            continue
        resources, compulsory = create_predictor(kernel.kind).resource_input(kernel, hardware_for_call(call, hardware))
        traffic = {kind: sum(node.bytes for path in resources.memory_paths if path.access_kind == kind
            for node in path.nodes if node.resource_role == "dram") for kind in ("read", "write")}
        objects = dsv4_partial_state_tensor_objects(kernel)
        actual[key] = {"valid_contract": True, "split_views": objects["split_views"], "writes": objects["writes"],
            "source_read_bytes": objects["source_read_bytes"], "source_write_bytes": objects["source_write_bytes"],
            "score_ape_fp32_additions": objects["score_ape_fp32_additions"],
            "resource_read_bytes": traffic["read"], "resource_write_bytes": traffic["write"],
            "resource_fp32_additions": resources.demand_per_resource["compute_fma"],
            "resource_compulsory_bytes": compulsory, "call_id": call.call_id,
            "unverified_integer_ALU_demand": resources.demand_per_resource["compute_alu"]}
    checks = []
    for key in sorted(set(expected) | set(actual)):
        if key not in expected or key not in actual:
            differences = ["unexpected" if key not in expected else "missing"]
        elif not actual[key]["valid_contract"]:
            differences = ["FP32_partial_state_contract"]
        else:
            e, a = expected[key], actual[key]
            differences = [field for field in ("split_views", "writes", "source_read_bytes", "source_write_bytes",
                "score_ape_fp32_additions") if e[field] != a[field]]
            for field, value in (("resource_read_bytes", e["source_read_bytes"]),
                    ("resource_write_bytes", e["source_write_bytes"]),
                    ("resource_fp32_additions", e["score_ape_fp32_additions"]),
                    ("resource_compulsory_bytes", e["source_read_bytes"] + e["source_write_bytes"])):
                if a[field] != value:
                    differences.append(field)
        checks.append({"rank": key[0], "layer": key[1], "role": key[2],
            "status": "matched" if not differences else "mismatched", "differences": differences,
            "expected": expected[key] if key in expected else None, "actual": actual[key] if key in actual else None,
            "source": reference["source_bindings"]["save_partial_program"] if key in expected else None})
    return {"scope": "all-valid FP32 save-partial split layout, state columns, source additions and requested payload",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "integer_instruction_cost_accepted": False, "physical_cache_or_provider_accepted": False,
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}


def compare_v4_finish_work(reference, plan, hardware):
    """Compare source active-value math and data payload; metadata excluded."""
    from src.kernel.kernel_base import create_predictor
    from src.kernel.kernel_spec import KernelKind
    from src.kernel.predictors.kv_cache_update import dsv4_finish_logical_work, dsv4_finish_partial_state_input
    from src.serving.kernel_lowering.kernel_plan import hardware_for_call

    expected = {(row["rank"], row["layer"], row["role"]): row for row in reference["finish_work"]}
    if len(expected) != len(reference["finish_work"]):
        raise ValueError("source finish work must retain distinct stage identities")
    actual = {}
    for call in plan.calls:
        kernel = call.kernel_spec
        if kernel.variant != "dsv4_compressed_state_cache":
            continue
        matches = [re.search(r"(?:^|:)layer-(\d+):attn:(compressor|indexer-compressor):finish(?::(compress|store))?(?::\d+)?$", name)
            for name in call.source_node_ids]
        matches = [match for match in matches if match is not None]
        if len(matches) != 1:
            raise ValueError("finish work requires one identifiable source stage per lowered call")
        match = matches[0]
        stage = "finish" if match[3] is None else match[3]
        key = call.rank, int(match[1]), ("main" if match[2] == "compressor" else "indexer") + "_" + stage
        if key in actual:
            raise ValueError("actual finish work has duplicate lowered stage calls")
        if kernel.kind is not KernelKind.KV_CACHE:
            actual[key] = {"valid_contract": False, "kind": kernel.kind.value}
            continue
        work = dsv4_finish_logical_work(kernel)
        state = dsv4_finish_partial_state_input(kernel)
        resources, _ = create_predictor(kernel.kind).resource_input(kernel, hardware_for_call(call, hardware))
        traffic = {kind: sum(node.bytes for path in resources.memory_paths if path.access_kind == kind
            for node in path.nodes if node.resource_role == "dram") for kind in ("read", "write")}
        temporary = work["source_temporary"]
        actual[key] = {"valid_contract": True, "stage": "finish" if work["stage"] is None else work["stage"],
            "state_dimension": work["state_dimension"], "completed_states": work["completed_states"],
            "valid_window_rows": work["valid_window_rows"], "demand": work["demand"],
            "state_half_selector": work["state_half_selector"],
            "resource_demand": {name: resources.demand_per_resource[name] for name in work["demand"]},
            "data_read_bytes": traffic["read"] - work["metadata_reads"]["baseline_read_bytes"],
            "data_write_bytes": traffic["write"],
            "state_input_dtype": None if state is None else state["dtype"],
            "state_payload_read_bytes": 0 if state is None else state["payload_read_bytes"],
            "scratch_shape": None if temporary is None else temporary["shape"],
            "scratch_extent_bytes": None if temporary is None else temporary["source_extent_bytes"],
            "excluded_metadata_read_bytes": work["metadata_reads"]["baseline_read_bytes"], "call_id": call.call_id}
    checks = []
    fields = ("stage", "state_dimension", "completed_states", "valid_window_rows", "state_half_selector", "demand", "data_read_bytes",
        "data_write_bytes", "state_input_dtype", "state_payload_read_bytes", "scratch_shape", "scratch_extent_bytes")
    for key in sorted(set(expected) | set(actual)):
        if key not in expected or key not in actual:
            differences = ["unexpected" if key not in expected else "missing"]
        elif not actual[key]["valid_contract"]:
            differences = ["finish_contract"]
        else:
            differences = [field for field in fields if expected[key][field] != actual[key][field]]
            if expected[key]["demand"] != actual[key]["resource_demand"]:
                differences.append("resource_demand")
        checks.append({"rank": key[0], "layer": key[1], "role": key[2],
            "status": "matched" if not differences else "mismatched", "differences": differences,
            "expected": expected[key] if key in expected else None, "actual": actual[key] if key in actual else None,
            "source": expected[key]["source_expressions"] if key in expected else None})
    return {"scope": "active-value source finish FP32 math, logical minimum reductions and data payload; actual metadata bytes subtracted only on actual side",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0,
        "checks": checks, "metadata_traffic_checked": False, "replicated_or_masked_instruction_cost_accepted": False,
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}


def compare_v4_compressor_dependencies(reference, observation, graph):
    """Derive state/scratch RAW dependencies from independent source IO.

    Only save/finish/compress/store ports participate. Physical cache pages,
    sparse consumers, rotary parameters and WAR/WAW effects remain outside
    this scoped check. Expected edges never read the production graph.
    """
    roles = {prefix + "_" + stage for prefix in ("main", "indexer")
        for stage in ("save_partial", "finish", "compress", "store")}
    writers, expected = {}, []
    for row in reference["calls"]:
        if row["role"] not in roles:
            continue
        key = row["rank"], row["role"], row["ordinal"]
        for storage in {value["storage"] for value in row["inputs"]}:
            storage_key = row["rank"], storage
            if storage_key in writers:
                expected.append((writers[storage_key], key))
        writers[row["rank"], row["output"]["storage"]] = key
    actual = {(row["rank"], row["role"], row["ordinal"]): row["node_id"]
        for row in observation.calls.rows if row["role"] in roles}
    data_edges = {(edge.source, edge.destination) for edge in graph.edges if edge.kind == "data"}
    checks = []
    for producer, consumer in expected:
        missing = producer not in actual or consumer not in actual
        pair = None if missing else (actual[producer], actual[consumer])
        matched = not missing and pair in data_edges
        checks.append({"producer": producer, "consumer": consumer,
            "expected": "direct data dependency from source storage writer to reader",
            "actual": pair, "status": "matched" if matched else "mismatched",
            "differences": [] if matched else ["missing_call" if missing else "missing_data_edge"],
            "source": {"producer": reference["source_bindings"][producer[1]],
                "consumer": reference["source_bindings"][consumer[1]]}})
    return {"scope": "source compressor logical state and C128 scratch RAW edges only",
        "total": len(checks), "matched": sum(row["status"] == "matched" for row in checks),
        "mismatched": sum(row["status"] != "matched" for row in checks), "not_checked": 0,
        "not_applicable": not checks, "checks": checks,
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}


def compare_v4_model_source_calls(reference, observation, *, primitive_nodes):
    expected = {(row["rank"], row["role"], row["ordinal"]): row for row in reference["calls"]}
    actual = {(row["rank"], row["role"], row["ordinal"]): row for row in observation.calls.rows}
    checks = []
    for key in sorted(set(expected) | set(actual)):
        if key not in expected or key not in actual:
            differences = ["unexpected" if key not in expected else "missing"]
        else:
            fields = ("order", "inputs", "output", *(name for name in ("operand_layouts", "activation_parameters")
                if name in expected[key]))
            differences = [field for field in fields if field not in actual[key] or expected[key][field] != actual[key][field]]
        checks.append({"rank": key[0], "role": key[1], "ordinal": key[2],
            "status": "matched" if not differences else "mismatched", "differences": differences,
            "source": reference["source_bindings"][key[1]] if key in expected else None,
            "expected": expected[key] if key in expected else None,
            "actual": actual[key] if key in actual else None})
    return {"scope": reference["scope"], "reference_calls": len(expected), "observed_calls": len(actual),
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "primitive_internal_checks": {"total": primitive_nodes, "matched": 0, "mismatched": 0, "not_checked": primitive_nodes},
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}


def compare_v4_source_dependencies(reference, observation, graph):
    """Source caller storage hazards must survive in the actual graph.

    Expected hazards use only the original source's serial tensor ports.
    Actual node associations come from the separate metadata replay. This
    checks caller-level RAW/WAR/WAW, not hidden custom-op memory accesses.
    """
    composite = {"model", "decoder", "attention", "input_projections", "prepare_attention",
        "output_projection", "ffn", "ffn_runner", "ffn_shared_mlp", "main_compressor", "indexer_compressor"}

    def tensors(value):
        if value is None:
            return ()
        if isinstance(value, tuple):
            return tuple(port for item in value for port in tensors(item))
        return (value,)

    writers, readers, expected = {}, {}, set()
    for row in reference["calls"]:
        if row["role"] in composite:
            continue
        key = row["rank"], row["role"], row["ordinal"]
        read = {(row["rank"], port["storage"]) for port in tensors(row["inputs"])}
        write = {(row["rank"], port["storage"]) for port in tensors(row["output"])}
        for storage in read:
            if storage in writers:
                expected.add((writers[storage], key, "RAW", storage))
        for storage in write:
            if storage in writers:
                expected.add((writers[storage], key, "WAW", storage))
            if storage in readers:
                expected.update((prior, key, "WAR", storage) for prior in readers[storage] if prior != key)
        for storage in read:
            if storage not in readers:
                readers[storage] = set()
            readers[storage].add(key)
        for storage in write:
            writers[storage], readers[storage] = key, set()
    nodes = {node.node_id for node in graph.nodes}
    incoming = {name: set() for name in nodes}
    for edge in graph.edges:
        if edge.kind == "data":
            incoming[edge.destination].add(edge.source)
    actual = {}
    for index, row in enumerate(observation.calls.rows):
        if row["role"] in composite:
            continue
        key = row["rank"], row["role"], row["ordinal"]
        if key in actual:
            raise ValueError("source dependency observation repeats a caller")
        actual[key] = set()
        if index in observation.call_nodes:
            for qualified, original in observation.call_nodes[index]:
                if qualified in nodes:
                    actual[key].add(qualified)
                elif original in nodes:
                    actual[key].add(original)  # joined TP collective

    def reaches(producers, consumers):
        pending, seen = list(consumers), set()
        while pending:
            current = pending.pop()
            if current in producers:
                return True
            if current not in seen:
                seen.add(current)
                pending.extend(incoming[current] - seen)
        return False

    checks = []
    for producer, consumer, hazard, storage in sorted(expected):
        valid = producer in actual and consumer in actual and bool(actual[producer]) and bool(actual[consumer])
        matched = valid and reaches(actual[producer], actual[consumer])
        checks.append({"producer": producer, "consumer": consumer, "hazard": hazard, "storage": storage,
            "status": "matched" if matched else "mismatched",
            "differences": [] if matched else ["missing_data_path" if valid else "missing_caller_nodes"],
            "expected": "source serial storage hazard preserved by data dependencies",
            "actual": None if not valid else {"producer_nodes": sorted(actual[producer]), "consumer_nodes": sorted(actual[consumer])},
            "source": {"producer": reference["source_bindings"][producer[1]], "consumer": reference["source_bindings"][consumer[1]]}})
    return {"scope": "source caller-level storage RAW/WAR/WAW paths; custom-op internals and cross-step storage excluded",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}
