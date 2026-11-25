"""Independent active-value finish work from genuine source expressions.

Only shape metadata and operation counts are evaluated. Reduction trees
use their mathematical minimum; replicated/masked backend instructions,
metadata traffic, physical pages and numerical output are separate scopes.
"""

import ast
from collections import Counter
from math import prod

from .qwen3_source import _Fields


class _Math:
    def __init__(self, values, *, parameters=None):
        self.values, self.counts = values, Counter()
        self.parameters = {} if parameters is None else parameters

    def broadcast(self, left, right):
        width = max(len(left), len(right))
        a, b = (1,) * (width - len(left)) + left, (1,) * (width - len(right)) + right
        if any(x != y and x != 1 and y != 1 for x, y in zip(a, b, strict=True)):
            raise ValueError("source mathematical operands cannot broadcast")
        return tuple(max(x, y) for x, y in zip(a, b, strict=True))

    def evaluate(self, node):
        if isinstance(node, ast.Name):
            return self.values[node.id]
        if isinstance(node, ast.Constant):
            return ()
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            # CuTe fragments are evaluated per scalar rmem element, with
            # enclosing source loop geometry applied by FinishSource.
            if self.values[node.value.id] != ():
                raise ValueError("source array fragment requires an explicit scalar-element projection")
            return ()
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            shape = self.evaluate(node.operand)
            self.counts["typed_negate"] += prod(shape)
            return shape
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            shape = self.broadcast(self.evaluate(node.left), self.evaluate(node.right))
            self.counts["compute_fma"] += prod(shape)
            return shape
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name in ("tl.reshape", "tl.broadcast_to"):
                old = self.evaluate(node.args[0])
                if not isinstance(node.args[1], ast.Tuple):
                    raise ValueError("source math shape must be an explicit metadata tuple")
                new = tuple(self.parameters[value.id] if isinstance(value, ast.Name)
                    else ast.literal_eval(value) for value in node.args[1].elts)
                if any(type(width) is not int or width <= 0 for width in new):
                    raise ValueError("source math shape dimensions must be positive integers")
                if (name == "tl.reshape" and prod(old) != prod(new)
                        or name == "tl.broadcast_to" and self.broadcast(old, new) != new):
                    raise ValueError("source math reshape/broadcast disagrees with its metadata")
                return new
            if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
                if len(node.args) != 1 or ast.unparse(node.args[0]) != "Float32":
                    raise ValueError("finish math cast is outside its declared FP32 conversion scope")
                shape = self.evaluate(node.func.value)
                self.counts["typed_conversion"] += prod(shape)
                return shape
            if name == "Float32":
                return self.evaluate(node.args[0])
            if name in ("cute.arch.shuffle_sync_bfly",):
                return self.evaluate(node.args[0])
            if name in ("tl.sum", "tl.max", "tl.softmax"):
                shape = self.evaluate(node.args[0])
                axis, = (ast.literal_eval(item.value) for item in node.keywords if item.arg in ("axis", "dim"))
                axis %= len(shape)
                reduced = (*shape[:axis], *shape[axis + 1:])
                reductions = (shape[axis] - 1) * prod(reduced)
                if name == "tl.max":
                    self.counts["typed_maximum"] += reductions
                else:
                    self.counts["compute_fma"] += reductions
                if name == "tl.softmax":
                    self.counts["compute_fma"] += 2 * prod(shape)
                    self.counts["compute_sfu"] += prod(shape)
                    self.counts["typed_maximum"] += reductions
                    return shape
                return reduced
            unary = {"cute.math.exp2", "cute.math.rsqrt", "tl.exp2", "tl.math.exp2", "tl.log2", "tl.rsqrt"}
            typed = {"cute.arch.fmax": "maximum", "cutlass.min": "minimum", "tl.maximum": "maximum",
                "tl.abs": "absolute", "tl.ceil": "ceil", "tl.clamp": "clamp"}
            if name in unary or name in typed:
                shape = ()
                for argument in node.args:
                    shape = self.broadcast(shape, self.evaluate(argument))
                self.counts["compute_sfu" if name in unary else "typed_" + typed[name]] += prod(shape)
                return shape
        raise ValueError("unsupported source mathematical expression: " + ast.unparse(node))


class _Fragments:
    def __init__(self, function, *, path, symbol):
        self.function, self.bindings = function, []
        self.path, self.symbol = path, symbol
        metadata = {"active", "group_idx", "nope_blocks", "group_lane", "has_kv_slot_idx", "has_position",
            "lane_id", "pos", "Int64", "tid", "row", "const_expr", "i", "state_block_size", "final_lane",
            "has_req_idx", "warp_id", "slot_id", "position", "COMPRESS_RATIO", "kv_slot_idx"}
        for node in ast.walk(function):
            if isinstance(node, ast.If):
                names = {value.id for value in ast.walk(node.test) if isinstance(value, ast.Name)}
                if not names <= metadata:
                    raise ValueError("finish math requires an undeclared value-dependent source branch")

    def find(self, target, *, contains=None, augmented=False):
        nodes = [node for node in ast.walk(self.function) if isinstance(node, ast.AugAssign if augmented else ast.Assign)
            and any(ast.unparse(value) == target for value in ((node.target,) if augmented else node.targets))
            and (contains is None or contains in {value.id for value in ast.walk(node.value) if isinstance(value, ast.Name)})]
        if len(nodes) != 1:
            raise ValueError("source mathematical assignment is ambiguous/missing: " + target)
        node = nodes[0]
        self.bindings.append({"target": target, "line": node.lineno, "expression": ast.unparse(node.value),
            "augmented_add": augmented, "path": self.path, "symbol": self.symbol})
        if augmented and not isinstance(node.op, ast.Add):
            raise ValueError("source reduction update must remain an addition")
        return node.value

    def scalar(self, target, *, contains=None, augmented=False, include_update=True):
        expression = self.find(target, contains=contains, augmented=augmented)
        values = {node.id: () for node in ast.walk(expression) if isinstance(node, ast.Name)}
        trace = _Math(values)
        trace.evaluate(expression)
        if augmented and include_update:
            trace.counts["compute_fma"] += 1
        return trace.counts


def _class_literal(source, cls, name):
    value, = (node.value for node in source.classes[cls].body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
    return ast.literal_eval(value)


class FinishSource:
    def __init__(self, source, *, head_dim, ratio, overlap, rope_dim, quant_block, record_bytes, query_lengths, kv_lengths):
        if (head_dim, ratio) not in ((512, 4), (512, 128), (128, 4)):
            raise ValueError("finish reference requires main C4/C128 or FP8 indexer C4")
        self.source, self.head, self.ratio, self.rope = source, head_dim, ratio, rope_dim
        self.quant, self.record = quant_block, record_bytes
        self.tokens, self.buckets = sum(query_lengths), {}
        if type(overlap) is not bool or overlap != (ratio == 4):
            raise ValueError("source finish overlap is outside its declared C4/C128 domain")
        if head_dim == 512 and ratio == 4:
            function = source.method("SparseAttnCompressNormRopeStoreC4Kernel", "kernel")
            fields = _Fields(function, {"compile_key": {"compress_ratio": ratio, "overlap": overlap}})
            window = fields.value(fields.expression("window"))
            start, = (node.value for node in ast.walk(function) if isinstance(node, ast.Assign)
                and any(ast.unparse(target) == "start" for target in node.targets))
            selector, = (node.value for node in ast.walk(function) if isinstance(node, ast.Assign)
                and any(ast.unparse(target) == "head_offset" for target in node.targets))
            if (ast.unparse(start) != "position - Int64(window - 1)"
                    or ast.unparse(selector) != "Int64(row // compress_ratio * head_dim)"):
                raise ValueError("source C4 state window/half selection changed")
        elif ratio == 128:
            function = source.method("SparseAttnCompressC128Block8Kernel", "kernel")
            window = _class_literal(source, "SparseAttnCompressC128Block8Kernel", "compress_ratio")
            if window != ratio:
                raise ValueError("source C128 window differs from its declared ratio")
        else:
            function, = (node for node in source.trees["fused_compress_quant_cache.py"].body
                if isinstance(node, ast.FunctionDef) and node.name == "_fused_kv_compress_norm_rope_insert_indexer_attn")
            fields = _Fields(function, {"OVERLAP": overlap, "COMPRESS_RATIO": ratio})
            tokens = fields.expression("tokens")
            if not isinstance(tokens, ast.Call) or ast.unparse(tokens.func) != "tl.arange" or ast.literal_eval(tokens.args[0]) != 0:
                raise ValueError("source indexer state window is not an explicit zero-based range")
            window = fields.value(tokens.args[1])
            if ast.unparse(fields.expression("head_offset")) != "(tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE":
                raise ValueError("source indexer half selection changed")
        for q, new in zip(query_lengths, kv_lengths, strict=True):
            old = new - q
            count = new // ratio - old // ratio
            first = (old // ratio + 1) * ratio
            short = min(count, max(0, (window - 1 - first) // ratio + 1))
            if short > 1:
                raise ValueError("source finish window requires unsupported additional overlap")
            for width, number in ((first, short), (window, count - short)):
                if number:
                    if width not in self.buckets:
                        self.buckets[width] = 0
                    self.buckets[width] += number
        self.states = sum(self.buckets.values())
        self.rows = sum(width * count for width, count in self.buckets.items())
        self.fragments = []

    def cute(self, cls):
        fragments = _Fragments(self.source.method(cls, "kernel"),
            path=str(self.source.paths["sparse_attn_compress_cutedsl.py"]), symbol=cls + ".kernel")
        self.fragments.append(fragments)
        return fragments

    def analyze(self, stage):
        if stage not in ("finish", "compress", "store") or self.ratio != 128 and stage != "finish":
            raise ValueError("source finish stage conflicts with its compressor ratio")
        self.fragments = []
        counts = Counter()
        def add(values, units):
            for name, value in values.items():
                counts[name] += value * units
        compress, store = stage != "store", stage != "compress"
        h, states = self.head, self.states
        values, outputs = self.rows * h, states * h
        if compress and self.head == 512 and self.ratio == 4:
            f = self.cute("SparseAttnCompressNormRopeStoreC4Kernel")
            for target, keyword in (("old_scale", None), ("new_scale", None),
                    ("local_sum[e]", "old_scale"), ("local_product[e]", "old_scale")):
                add(f.scalar(target, contains=keyword), values)
            add(f.scalar("x[e]", contains="local_product"), outputs)
        elif compress and self.ratio == 128:
            cls = "SparseAttnCompressC128Block8Kernel"
            f = self.cute(cls)
            groups = _class_literal(self.source, cls, "num_warps")
            steps = _class_literal(self.source, cls, "final_reduce_steps")
            if groups * _class_literal(self.source, cls, "rows_per_warp") != self.ratio:
                raise ValueError("source C128 row partition changed")
            add(f.scalar("exp_score"), values)
            add(f.scalar("local_sum[e]", contains="exp_score", augmented=True), values)
            add(f.scalar("local_product[e]", contains="exp_score", augmented=True), values)
            for target in ("scale", "global_sum", "global_product"):
                add(f.scalar(target), groups * outputs)
            for target in ("global_sum", "global_product"):
                add(f.scalar(target, augmented=True), groups * steps * outputs)
            divide, = (node.value for node in ast.walk(f.function) if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Div)
                and ast.unparse(node.value.left) == "global_product" and ast.unparse(node.value.right) == "global_sum")
            trace = _Math({"global_product": (), "global_sum": ()})
            trace.evaluate(divide)
            add(trace.counts, outputs)
        indexer = self.head == 128
        if indexer:
            function, = (node for node in self.source.trees["fused_compress_quant_cache.py"].body
                if isinstance(node, ast.FunctionDef) and node.name == "_fused_kv_compress_norm_rope_insert_indexer_attn")
            f = _Fragments(function, path=str(self.source.paths["fused_compress_quant_cache.py"]), symbol=function.name)
            self.fragments.append(f)
            for rows, number in self.buckets.items():
                trace = _Math({"score": (rows, h), "kv": (rows, h), "compressed_kv": (h,), "HEAD_SIZE": (),
                    "rms_norm_eps": (), "rms_w": (h,), "variance": (), "rrms": (), "even": (self.rope // 2,),
                    "odd": (self.rope // 2,), "cos_v": (self.rope // 2,), "sin_v": (self.rope // 2,),
                    "result_bf16": (h,), "absmax": (), "INV_FP8_MAX": (), "raw_scale": (),
                    "exponent": (), "inv_scale": ()})
                for target, keyword in (("score", "score"), ("compressed_kv", None), ("variance", None),
                        ("rrms", None), ("normed", None), ("new_even", None), ("new_odd", None),
                        ("raw_scale", None), ("exponent", None), ("inv_scale", None),
                        ("x_scaled", None), ("scale_val", None)):
                    trace.evaluate(f.find(target, contains=keyword))
                add(trace.counts, number)
        elif store:
            cls = "SparseAttnCompressNormRopeStoreC4Kernel" if self.ratio == 4 else "SparseAttnNormRopeStoreKernel"
            f = self.cute(cls)
            if self.ratio == 4:
                add(f.scalar("local_sumsq", contains="x", augmented=True, include_update=False), outputs)
                add(f.scalar("rrms_shared[0]"), states)
                add(f.scalar("x[e]", contains="rrms"), outputs)
            else:
                expression = f.find("local_sumsq")
                if not isinstance(expression, ast.BinOp) or not isinstance(expression.op, ast.Add):
                    raise ValueError("source C128 norm square pair changed")
                for square in (expression.left, expression.right):
                    trace = _Math({"x0": (), "x1": ()})
                    trace.evaluate(square)
                    add(trace.counts, outputs // 2)
                add(f.scalar("rrms"), states)
                for target in ("x0", "x1"):
                    add(f.scalar(target, contains="rrms"), outputs // 2)
            counts["compute_fma"] += states * (h - 1)  # useful sum tree, not replicated shuffle issue work
            for target in ("real", "imag"):
                add(f.scalar(target), states * self.rope // 2)
            nope = h - self.rope
            add(f.scalar("scale_raw"), states * nope // self.quant)
            for target in ("y0", "y1"):
                add(f.scalar(target), states * nope // 2)
        state_read = values * 2 * 4 if compress else 0
        scratch_read = outputs * 4 if store and self.ratio == 128 else 0
        scratch_write = outputs * 4 if compress and self.ratio == 128 else 0
        data_read = state_read + scratch_read + (states * (h * 2 + self.rope * 4) if store else 0)
        return {"stage": stage, "state_dimension": h, "completed_states": states, "valid_window_rows": self.rows,
            "state_half_selector": "window_row // 4" if self.ratio == 4 else "0",
            "demand": {name: counts[name] for name in ("compute_fma", "compute_sfu", "compute_alu", "compute_reduction", "compute_tensor")},
            "data_read_bytes": data_read, "data_write_bytes": scratch_write + (states * self.record if store else 0),
            "state_payload_read_bytes": state_read, "state_input_dtype": "fp32" if compress else None,
            "scratch_shape": [self.tokens, h] if self.ratio == 128 else None,
            "scratch_extent_bytes": self.tokens * h * 4 if self.ratio == 128 else None,
            "source_expressions": [binding for f in self.fragments for binding in f.bindings],
            "counted_typed_actions_in_selected_expressions": {name: value for name, value in counts.items() if name.startswith("typed_")},
            "metadata_traffic_checked": False,
            "scope": "active-value logical arithmetic and typed payload, minimum reductions; not padded/replicated issue work, metadata traffic, numerical equality or provider acceptance"}
