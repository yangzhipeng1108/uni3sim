"""Independent Qwen3 caller-interface reference from genuine source methods.

This executes extracted Python control flow on reference metadata only, not
Torch, vLLM modules, Midlayer workload builders, lowering or predictors. It
does not model numerical values, allocation lifetimes, kernel internals,
compiler behavior, physical KV pages or an installed runtime/provider.
"""

import ast
from copy import deepcopy
from itertools import islice
import json
from math import prod
from pathlib import Path
from types import SimpleNamespace


class _WithoutAnnotations(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.returns, node.decorator_list = None, []
        return self.generic_visit(node)

    def visit_arg(self, node):
        node.annotation = None
        return node

    def visit_AnnAssign(self, node):
        return ast.copy_location(ast.Pass() if node.value is None else ast.Assign(
            targets=[node.target], value=self.visit(node.value)), node)


class _MetadataImports(ast.NodeTransformer):
    """Bind explicit primitive ports without importing SDK/runtime modules.

    Only declared (relative level, module, symbol) imports are admitted;
    source control flow and the import's local name remain unchanged.
    """
    def __init__(self, bindings):
        self.bindings, self.used = bindings, set()

    def visit_Import(self, node):
        raise ValueError("undeclared module import in metadata source method")

    def visit_ImportFrom(self, node):
        assignments = []
        for alias in node.names:
            key = node.level, node.module, alias.name
            if key not in self.bindings:
                raise ValueError("undeclared primitive import in metadata source method: " + repr(key))
            self.used.add(key)
            assignments.append(ast.copy_location(ast.Assign(
                targets=[ast.Name(id=alias.name if alias.asname is None else alias.asname, ctx=ast.Store())],
                value=ast.Subscript(value=ast.Name(id="__source_metadata_imports__", ctx=ast.Load()),
                    slice=ast.Tuple(elts=[ast.Constant(value=item) for item in key], ctx=ast.Load()), ctx=ast.Load())), node))
        return assignments


class _Source:
    def __init__(self, root, *, model_sources=None):
        self.root = Path(root)
        directory = Path(root) / "vllm/model_executor/models"
        self.paths = ({name: directory / name for name in ("qwen2.py", "qwen3.py", "interfaces.py")}
            if model_sources is None else {name: self.root / relative for name, relative in model_sources.items()})
        self.trees = {name: ast.parse(path.read_text()) for name, path in self.paths.items()}
        self.classes = {node.name: node for tree in self.trees.values() for node in tree.body
            if isinstance(node, ast.ClassDef)}
        self.bindings = []

    def load_target_sources(self):
        for name, relative in (
                ("logits_processor.py", "vllm/model_executor/layers/logits_processor.py"),
                ("sampler.py", "vllm/v1/sample/sampler.py"),
                ("gpu_model_runner.py", "vllm/v1/worker/gpu_model_runner.py")):
            path = self.root / relative
            tree = ast.parse(path.read_text())
            self.paths[name], self.trees[name] = path, tree
            self.classes.update({node.name: node for node in tree.body if isinstance(node, ast.ClassDef)})

    def validate_row_selection(self):
        prepare = self.method("GPUModelRunner", "_prepare_inputs")
        branch, = (node for node in ast.walk(prepare) if isinstance(node, ast.If)
            and ast.unparse(node.test) == "not use_spec_decode")
        indices, = (node for node in branch.body if isinstance(node, ast.Assign)
            and any(ast.unparse(target) == "logits_indices" for target in node.targets))
        if ast.unparse(indices.value) != "query_start_loc[1:] - 1":
            raise ValueError("source ordinary-text logits row indices changed")
        execute = self.method("GPUModelRunner", "execute_model")
        gathers = [node for node in ast.walk(execute) if isinstance(node, ast.Assign)
            and any(ast.unparse(target) == "sample_hidden_states" for target in node.targets)]
        if not gathers or any(ast.unparse(node.value) != "hidden_states[logits_indices]" for node in gathers):
            raise ValueError("source logits hidden-state materialization changed")
        self.bindings.extend({"symbol": "GPUModelRunner." + method, "line": line}
            for method, line in (("_prepare_inputs.logits_indices", indices.lineno),
                ("execute_model.sample_hidden_states", gathers[0].lineno)))

    def validate_model_bindings(self):
        """Reject changed inheritance/decoder wiring before executing bodies."""
        model = self.classes["Qwen3Model"]
        if tuple(ast.unparse(base) for base in model.bases) != ("Qwen2Model",):
            raise ValueError("source Qwen3Model no longer inherits the referenced Qwen2Model")
        delegates = [node for node in ast.walk(self.method("Qwen3Model", "__init__"))
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "super().__init__"]
        if (len(delegates) != 1 or len(delegates[0].args) != 0 or
                {item.arg: ast.unparse(item.value) for item in delegates[0].keywords} != {
                    "vllm_config": "vllm_config", "prefix": "prefix", "decoder_layer_type": "Qwen3DecoderLayer"}):
            raise ValueError("source Qwen3Model decoder binding changed")
        aliases = [item for node in self.trees["qwen3.py"].body if isinstance(node, ast.ImportFrom)
            and node.level == 1 and node.module == "qwen2" for item in node.names
            if item.asname == "Qwen3MLP"]
        if len(aliases) != 1 or aliases[0].name != "Qwen2MLP":
            raise ValueError("source Qwen3MLP no longer aliases the referenced Qwen2MLP")

    def method(self, cls, method):
        function = next(node for node in self.classes[cls].body if isinstance(node, ast.FunctionDef) and node.name == method)
        return function

    def forward(self, cls, method="forward", *, metadata_globals=None, metadata_imports=None):
        return self._compile(self.method(cls, method), cls + "." + method, metadata_globals, metadata_imports)

    def function(self, file, name, *, metadata_globals=None, metadata_imports=None):
        original, = (node for node in self.trees[file].body if isinstance(node, ast.FunctionDef) and node.name == name)
        return self._compile(original, name, metadata_globals, metadata_imports)

    def metadata_constructor(self, cls, *, metadata_globals, metadata_base=None):
        """Execute constructor geometry on a plain metadata object.

        The genuine body retains rebinding/branches and zero-argument super.
        Its default base is object; an explicit metadata base can accept a
        CustomOp's initialization arguments. Framework registration stays
        outside this reference, and leaves are explicit metadata ports.
        """
        constructor = _WithoutAnnotations().visit(deepcopy(self.method(cls, "__init__")))
        if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(constructor)):
            raise ValueError("metadata constructor cannot import a runtime/module")
        definition = ast.ClassDef(name=cls, bases=[] if metadata_base is None else [ast.Name(
            id="__metadata_constructor_base__", ctx=ast.Load())], keywords=[], body=[constructor], decorator_list=[])
        tree = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
        namespace = dict(metadata_globals)
        if metadata_base is not None:
            namespace["__metadata_constructor_base__"] = metadata_base
        exec(compile(tree, f"<upstream-metadata-constructor:{cls}>", "exec"), namespace)
        self.bindings.append({"symbol": cls + ".__init__", "line": constructor.lineno,
            "base_scope": "metadata-only constructor base; framework bookkeeping/registration excluded"})
        return namespace[cls]

    def _compile(self, original, symbol, metadata_globals, metadata_imports=None):
        function = _WithoutAnnotations().visit(deepcopy(original))
        if metadata_imports is not None:
            imports = _MetadataImports(metadata_imports)
            function = imports.visit(function)
            if imports.used != set(metadata_imports):
                raise ValueError("declared primitive import is absent from source method")
        tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        # PP1 is a declared reference input. No fake SDK/module is imported.
        namespace = {"get_pp_group": lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True), "islice": islice}
        if metadata_globals is not None:
            namespace.update(metadata_globals)
        if metadata_imports is not None:
            namespace["__source_metadata_imports__"] = metadata_imports
        exec(compile(tree, f"<upstream:{symbol}>", "exec"), namespace)
        self.bindings.append({"symbol": symbol, "line": original.lineno,
            **({"metadata_primitive_imports": [{"level": level, "module": module, "symbol": name}
                for level, module, name in metadata_imports]} if metadata_imports is not None else {})})
        return namespace[original.name]

    def fields(self, cls, parameters, *, tensor_parallel_size=1):
        return _Fields(self.method(cls, "__init__"), parameters, tensor_parallel_size=tensor_parallel_size)

    def default_parameters(self, cls, parameters):
        """Read literal constructor defaults; do not run a module constructor."""
        function = self.method(cls, "__init__")
        names = function.args.args[len(function.args.args) - len(function.args.defaults):]
        defaults = {name.arg: ast.literal_eval(value) for name, value in zip(names, function.args.defaults, strict=True)}
        return {**defaults, **parameters}


class _Fields:
    """Evaluate only constructor scalar metadata, never constructors/ops."""
    def __init__(self, function, parameters, *, tensor_parallel_size=1):
        if type(tensor_parallel_size) is not int or tensor_parallel_size <= 0:
            raise ValueError("source constructor requires a positive declared tensor parallel size")
        self.tensor_parallel_size = tensor_parallel_size
        self.function, self.parameters = function, parameters
        self.assignments = []

        def collect(statements, guards):
            for node in statements:
                if isinstance(node, ast.Assign):
                    self.assignments.append((node, guards))
                elif isinstance(node, ast.If):
                    collect(node.body, (*guards, (node.test, True)))
                    collect(node.orelse, (*guards, (node.test, False)))
        collect(function.body, ())

    def expression(self, name, constructor=None):
        matches = [node.value for node, guards in self.assignments if any(ast.unparse(target) == name for target in node.targets)
            and (constructor is None or isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == constructor)
            and all(bool(self.value(test)) == branch for test, branch in guards)]
        if len(matches) != 1:
            raise ValueError(f"source constructor binding is ambiguous/missing: {name}/{constructor}")
        return matches[0]

    def value(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            values = [self.value(child) for child in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.Name):
            return self.parameters[node.id] if node.id in self.parameters else self.value(self.expression(node.id))
        if isinstance(node, ast.Attribute):
            if ast.unparse(node.value) == "self":
                return self.value(self.expression("self." + node.attr))
            return self.value(node.value)[node.attr]
        if isinstance(node, ast.Subscript):
            return self.value(node.value)[self.value(node.slice)]
        if isinstance(node, ast.BinOp):
            left, right = self.value(node.left), self.value(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Pow):
                return left ** right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not self.value(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self.value(node.operand)
        if isinstance(node, ast.IfExp):
            return self.value(node.body if self.value(node.test) else node.orelse)
        if isinstance(node, ast.BoolOp):
            for child in node.values:
                result = self.value(child)
                if (isinstance(node.op, ast.Or) and result) or (isinstance(node.op, ast.And) and not result):
                    return result
            return result
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = self.value(node.left), self.value(node.comparators[0])
            if isinstance(node.ops[0], ast.Eq):
                return left == right
            if isinstance(node.ops[0], ast.Is):
                return left is right
            if isinstance(node.ops[0], ast.IsNot):
                return left is not right
            if isinstance(node.ops[0], ast.NotEq):
                return left != right
            if isinstance(node.ops[0], ast.GtE):
                return left >= right
            if isinstance(node.ops[0], ast.Gt):
                return left > right
            if isinstance(node.ops[0], ast.Lt):
                return left < right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "get_tensor_model_parallel_world_size" and not node.args:
                return self.tensor_parallel_size
            if name == "get_pp_group" and not node.args:
                return {"is_first_rank": True, "is_last_rank": True}
            if name == "int" and len(node.args) == 1:
                value = self.value(node.args[0])
                if type(value) not in (int, bool):
                    raise ValueError("source integer metadata cast requires an integer or boolean")
                return int(value)
            if name == "getattr" and len(node.args) == 3:
                config, field = self.value(node.args[0]), self.value(node.args[1])
                return config[field] if field in config else self.value(node.args[2])
            if name in ("max", "min"):
                values = tuple(self.value(child) for child in node.args)
                return max(values) if name == "max" else min(values)
        raise NotImplementedError("unsupported upstream constructor metadata: " + ast.unparse(node))

    def call(self, name, constructor):
        call = self.expression(name, constructor)
        return tuple(self.value(node) for node in call.args), {
            item.arg: item.value for item in call.keywords}, call.lineno

    def keyword(self, keywords, name):
        return self.value(keywords[name])


class _Storage:
    def __init__(self):
        self.writer, self.readers = None, set()


class _Tensor:
    """Reference shape and logical storage identity, with no payload/liveness."""
    @staticmethod
    def _dense_strides(shape):
        dense, step = [], 1
        for width in reversed(shape):
            dense.append(step)
            step *= width
        return tuple(reversed(dense))

    def __init__(self, shape, dtype="bf16", storage=None, *, target_ops=None, strides=None, storage_offset=0):
        self.shape, self.dtype = tuple(shape), dtype
        self.storage = _Storage() if storage is None else storage
        self.target_ops = target_ops
        self.strides = self._dense_strides(self.shape) if strides is None else tuple(strides)
        if (len(self.strides) != len(self.shape) or any(type(s) is not int or s <= 0 for s in self.strides)
                or type(storage_offset) is not int or storage_offset < 0):
            raise ValueError("reference layout requires positive element strides and nonnegative offset")
        self.storage_offset = storage_offset

    def stride(self, dim):
        return self.strides[dim]

    def is_contiguous(self):
        return all(size == 1 or stride == expected for size, stride, expected in
            zip(self.shape, self.strides, self._dense_strides(self.shape), strict=True))

    def transpose(self, first, second):
        shape, strides = list(self.shape), list(self.strides)
        shape[first], shape[second] = shape[second], shape[first]
        strides[first], strides[second] = strides[second], strides[first]
        return _Tensor(tuple(shape), self.dtype, self.storage, strides=tuple(strides), storage_offset=self.storage_offset)

    def flatten(self, start_dim):
        if start_dim != 1:
            raise ValueError("reference flatten currently requires a token-major trailing group")
        return self.view(self.shape[0], prod(self.shape[1:]))

    def as_strided(self, shape, strides):
        if self.ndim != 1 or not self.is_contiguous():
            raise ValueError("reference as_strided requires an explicit flat backing")
        span = 1 + sum((width - 1) * step for width, step in zip(shape, strides, strict=True))
        if span > self.numel():
            raise ValueError("reference strided view exceeds its declared backing")
        return _Tensor(shape, self.dtype, self.storage, strides=strides, storage_offset=self.storage_offset)

    def split(self, sizes, dim=-1):
        if dim != -1 or sum(sizes) != self.shape[-1]:
            raise ValueError("source reference split disagrees with constructor geometry")
        outputs, offset = [], self.storage_offset
        for size in sizes:
            outputs.append(_Tensor((*self.shape[:-1], size), self.dtype, self.storage,
                target_ops=self.target_ops, strides=self.strides, storage_offset=offset))
            offset += size * self.strides[-1]
        return tuple(outputs)

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        if shape.count(-1) == 1:
            specified = prod(value for value in shape if value != -1)
            if specified <= 0 or prod(self.shape) % specified:
                raise ValueError("source reference inferred reshape is not integral")
            shape = tuple(prod(self.shape) // specified if value == -1 else value for value in shape)
        if prod(shape) != prod(self.shape) or any(value <= 0 for value in shape):
            raise ValueError("unsupported source reference reshape")
        dense = self._dense_strides(self.shape)
        strides = self._dense_strides(shape)
        if tuple(shape) == self.shape:
            strides = self.strides
        elif any(size != 1 and old != expected for size, old, expected in zip(self.shape, self.strides, dense, strict=True)):
            if shape[0] != self.shape[0] or self.strides[1:] != dense[1:]:
                raise ValueError("source reference cannot view a noncontiguous layout across rows")
            strides = (self.strides[0], *strides[1:])
        return _Tensor(shape, self.dtype, self.storage, target_ops=self.target_ops,
            strides=strides, storage_offset=self.storage_offset)

    def dim(self):
        return len(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        return prod(self.shape)

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def __bool__(self):
        raise ValueError("reference metadata cannot choose an undeclared tensor-value branch")

    @property
    def device(self):
        return "reference_metadata_only"

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) == len(self.shape) and all(isinstance(axis, slice) for axis in item):
            shape = []
            for width, axis in zip(self.shape, item, strict=True):
                start, end, step = axis.indices(width)
                if step != 1 or start != 0 or end <= start:
                    raise NotImplementedError("reference slice only binds nonempty leading unit-step views")
                shape.append(end)
            return _Tensor(shape, self.dtype, self.storage, target_ops=self.target_ops,
                strides=self.strides, storage_offset=self.storage_offset)
        if (not isinstance(item, tuple) or len(item) != 2 or item[0] is not Ellipsis
                or not isinstance(item[1], slice) or item[1].start is not None
                or item[1].step is not None or item[1].stop != self.shape[-1]):
            raise NotImplementedError("reference logits slice requires an unpadded full vocabulary")
        return self.view(*self.shape)

    def to(self, dtype):
        if dtype == self.dtype:
            return self
        if self.target_ops is None:
            raise NotImplementedError("source dtype conversion outside the declared target suffix")
        return self.target_ops.cast(self, dtype)

    def long(self):
        return self.to("int64")

    def argmax(self, dim, keepdim=False):
        if dim != -1 or keepdim or self.target_ops is None:
            raise NotImplementedError("reference only binds row-wise greedy argmax")
        return self.target_ops.argmax(self)

    def unsqueeze(self, dim):
        if dim != -1:
            raise NotImplementedError("reference only binds trailing sampled-token singleton view")
        return self.view(*self.shape, 1)

    def squeeze(self, dim):
        if dim != -1 or self.shape[-1] != 1:
            raise ValueError("reference squeeze requires a trailing singleton")
        return _Tensor(self.shape[:-1], self.dtype, self.storage, strides=self.strides[:-1],
            storage_offset=self.storage_offset)


class _Ledger:
    def __init__(self):
        self.calls, self.names = [], set()

    def emit(self, name, kind, shape, dtype, variant, inputs, outputs, attributes):
        if name in self.names:
            raise ValueError("source repeats a caller boundary without a distinct invocation identity: " + name)
        self.names.add(name)
        dependencies = {tensor.storage.writer for tensor in inputs}
        for tensor in outputs:
            dependencies.add(tensor.storage.writer)
            dependencies.update(tensor.storage.readers)
        dependencies.discard(None)
        self.calls.append({"call": name, "kind": kind, "shape": shape, "dtype": dtype, "variant": variant,
            "inputs": tuple((tensor.shape, tensor.dtype) for tensor in inputs),
            "outputs": tuple((tensor.shape, tensor.dtype) for tensor in outputs), "attributes": attributes,
            "parents": tuple(sorted(dependencies)), "output_alias_inputs": tuple(
                tuple(output.storage is tensor.storage for tensor in inputs) for output in outputs)})
        for tensor in inputs:
            tensor.storage.readers.add(name)
        for tensor in outputs:
            tensor.storage.writer, tensor.storage.readers = name, set()


class _Leaf:
    def __init__(self, ledger, name, operation, *geometry):
        self.ledger, self.name, self.operation, self.geometry = ledger, name, operation, geometry

    def __call__(self, *inputs):
        x = inputs[0]
        rows = prod(x.shape[:-1])
        attributes = {}
        if self.operation in ("linear", "lm_head"):
            inside, outside, bias = self.geometry
            if x.shape[-1] != inside:
                raise ValueError("source linear constructor/input geometry disagree")
            output = _Tensor((*x.shape[:-1], outside), x.dtype)
            # The workload descriptor distinguishes a one-row output head
            # from matrix-matrix projections; both retain the same linear
            # caller semantics and exact input/output geometry.
            variant = "gemv" if self.operation == "lm_head" and rows == 1 else "gemm"
            # Module-owned weights are read-only ports, with constructor-derived
            # geometry and storage independent of both activations and outputs.
            self.ledger.emit(self.name, self.operation, {"b": 1, "m": rows, "n": outside, "k": inside},
                x.dtype, variant, (*inputs, _Tensor((outside, inside), x.dtype)), (output,), {"has_bias": bias})
            return output, None
        if self.operation == "norm":
            width, epsilon = self.geometry
            if x.shape[-1] != width:
                raise ValueError("source norm constructor/input geometry disagree")
            outputs = inputs if len(inputs) == 2 else (_Tensor(x.shape, x.dtype),)
            self.ledger.emit(self.name, "norm", {"rows": rows, "hidden": width}, x.dtype,
                "add_rmsnorm" if len(inputs) == 2 else "rmsnorm",
                (*inputs, _Tensor((width,), x.dtype)), outputs, {"epsilon": epsilon})
            return outputs if len(inputs) == 2 else outputs[0]
        if self.operation == "activation":
            width = x.shape[-1] // 2
            output = _Tensor((*x.shape[:-1], width), x.dtype)
            self.ledger.emit(self.name, "activation", {"elements": rows * width, "tok": rows, "intermediate": width},
                x.dtype, "silu", inputs, (output,), attributes)
            return output
        if self.operation == "embedding":
            width, vocab_size = self.geometry
            count = prod(x.shape)
            name = self.name
            # VocabParallelEmbedding.forward's TP1 branch passes input_.long()
            # to UnquantizedEmbeddingMethod.embedding/F.embedding. Scheduled
            # INT32 IDs therefore materialize INT64 before the gather; this is
            # a declared primitive port contract, not provider acceptance.
            if x.dtype == "int32":
                widened = _Tensor(x.shape, "int64")
                self.ledger.emit(name, "vector", {"rows": count, "elements": count},
                    "int32", "vllm_embedding_id_widen", (x,), (widened,), {
                        "source_component": "vllm_0_30_symbolic_embedding",
                        "embedding_source": "vllm_0_30_unquantized_tp1"})
                x, name = widened, name + ":1"
            elif x.dtype != "int64":
                raise ValueError("reference embedding requires INT32/INT64 source IDs")
            output = _Tensor((count, width))
            self.ledger.emit(name, "gather", {"elements": count * width, "rows": count, "hidden": width,
                "source_rows": -(-vocab_size // 64) * 64}, "bf16", "embedding", (x,), (output,), {
                    "index_bytes": 8, "embedding_source": "vllm_0_30_unquantized_tp1",
                    "model_execution_version": "vllm_0_30_0", "source_operation": "embedding"})
            return output
        if self.operation == "rotary":
            heads, kv_heads, width, cache = self.geometry
            positions, query, key = inputs
            count = positions.shape[0]
            self.ledger.emit(self.name, "rotary", {"elements": count * (heads + kv_heads) * width,
                "seq_len": count, "num_query_heads": heads, "num_kv_heads": kv_heads, "head_dim": width},
                "bf16", "rotary", (*inputs, cache), (query, key), attributes)
            return query, key
        if self.operation == "attention":
            heads, kv_heads, width, queries, contexts, slots, cache = self.geometry
            q, k, v = inputs
            count = sum(queries)
            self.ledger.emit(self.name + ":kv-update", "kv_update", {"elements": count * kv_heads * width,
                "seq_len": count, "num_kv_heads": kv_heads, "head_dim": width}, "bf16", "paged_kv_cache",
                (k, v), (cache,), {"symbolic_slot_mapping": slots})
            output = _Tensor((count, heads * width))
            self.ledger.emit(self.name, "flash_attn", {"batch": len(queries), "q_len": max(queries), "kv_len": max(contexts),
                "head_dim": width, "num_query_heads": heads, "num_kv_heads": kv_heads}, "bf16",
                "decode" if all(length == 1 for length in queries) else "prefill", (*inputs, cache), (output,),
                {"causal": True, "q_lens": queries, "kv_lens": contexts})
            return output
        raise NotImplementedError("unbound reference primitive " + self.operation)


class _TargetOps:
    """Typed metadata callbacks called by extracted Sampler methods only."""
    def __init__(self, ledger, rows, vocab):
        self.ledger, self.rows, self.vocab = ledger, rows, vocab

    def cast(self, value, dtype):
        if (value.dtype, dtype) == ("bf16", "fp32"):
            name, variant, attributes = "target-sampler", "vllm_logits_fp32_cast", {}
        elif (value.dtype, dtype) == ("int64", "int32"):
            name, variant, attributes = "target-sampler:2", "vllm_plain_greedy_sample", {"stage": "output_int32"}
        else:
            raise NotImplementedError("reference target suffix dtype conversion changed")
        output = _Tensor(value.shape, dtype, target_ops=self)
        self.ledger.emit(name, "vector", {"elements": prod(value.shape), "rows": self.rows, "vocab": self.vocab},
            value.dtype, variant, (value,), (output,), attributes)
        return output

    def argmax(self, value):
        if value.dtype != "fp32" or value.shape != (self.rows, self.vocab):
            raise ValueError("reference greedy argmax requires FP32 full-vocabulary row logits")
        output = _Tensor((self.rows,), "int64", target_ops=self)
        self.ledger.emit("target-sampler:1", "vector", {"elements": prod(value.shape), "rows": self.rows, "vocab": self.vocab},
            value.dtype, "vllm_plain_greedy_sample", (value,), (output,), {"stage": "argmax"})
        return output


def _target_source_forward(source, ledger, root, hidden, queries, config):
    """Bind worker selection and execute original logits/greedy wrapper code.

    Only torch dtype *names* are supplied to the detached source methods;
    no torch module is imported/replaced and no numerical operation occurs.
    Head GEMM and argmax are declared logical primitive boundaries, as are
    the model's other leaf operators. Provider internals remain excluded.
    """
    source.load_target_sources()
    source.validate_row_selection()
    root_fields = source.fields("Qwen3ForCausalLM", {"config": config})
    head, _, _ = root_fields.call("self.lm_head", "ParallelLMHead")
    args, keywords, _ = root_fields.call("self.logits_processor", "LogitsProcessor")
    if len(args) != 1 or keywords:
        raise NotImplementedError("reference requires ordinary logits-processor construction")
    fields = source.fields("LogitsProcessor", source.default_parameters("LogitsProcessor", {"vocab_size": args[0]}))
    processor = type("ReferenceLogitsProcessor", (), {"__call__": source.forward("LogitsProcessor")})()
    for name in ("logits_as_input", "scale", "soft_cap", "org_vocab_size"):
        setattr(processor, name, fields.value(fields.expression("self." + name)))
    processor.head_dtype = None  # declared model dtype, no runtime head override
    for name in ("_get_logits", "_apply_head"):
        setattr(processor, name, source.forward("LogitsProcessor", name).__get__(processor))

    def apply_head(layer, states, bias=None):
        if bias is not None:
            raise NotImplementedError("reference head is bias-free")
        return _Leaf(ledger, "lm-head", "lm_head", head[1], head[0], False)(states)[0]

    root.lm_head = SimpleNamespace(tp_size=1, quant_method=SimpleNamespace(apply=apply_head))
    root.logits_processor = processor
    root.compute_logits = source.forward("Qwen3ForCausalLM", "compute_logits").__get__(root)
    rows = len(queries)
    selected, indices = _Tensor((rows, hidden.shape[-1])), _Tensor((rows,), "int32")
    offsets, end = [], 0
    for count in queries:
        end += count
        offsets.append(end - 1)
    ledger.emit("logits-boundary-gather", "vector", {"elements": prod(selected.shape), "rows": rows,
        "hidden": hidden.shape[-1], "source_rows": sum(queries)}, "bf16", "vllm_logits_row_select", (hidden, indices), (selected,),
        {"index_source": "query_start_loc_int32", "selected_indices": tuple(offsets)})
    logits = root.compute_logits(selected)
    logits.target_ops = _TargetOps(ledger, rows, head[0])
    metadata_globals = {"torch": SimpleNamespace(float32="fp32", int32="int32", int64="int64"), "SamplerOutput": SimpleNamespace}
    sampler = type("ReferenceSampler", (), {"__call__": source.forward("Sampler", metadata_globals=metadata_globals)})()
    sampler.logprobs_mode = "raw_logprobs"
    for name in ("sample", "apply_logits_processors"):
        setattr(sampler, name, source.forward("Sampler", name, metadata_globals=metadata_globals).__get__(sampler))
    for name in ("greedy_sample", "apply_penalties"):
        setattr(sampler, name, source.forward("Sampler", name, metadata_globals=metadata_globals))
    metadata = SimpleNamespace(all_greedy=True, all_random=False, no_penalties=True,
        max_num_logprobs=None, logprob_token_ids=None, allowed_token_ids_mask=None, bad_words_token_ids={},
        output_token_ids=tuple(() for _ in queries), spec_token_ids=tuple(() for _ in queries),
        logitsprocs=SimpleNamespace(non_argmax_invariant=()), thinking_budget_state_holder=None)
    return sampler(logits, metadata).sampled_token_ids


def qwen3_source_reference(upstream_root, config_path, *, query_lengths, kv_lengths, slot_mapping, cache_capacity_tokens):
    """Execute the original model/decoder/attention/MLP forward bodies.

    Primitive references cover caller shapes/dtypes/aliases/dependencies and
    the declared CUDA in-place/logical-cache boundary, not library internals.
    Constructor geometry is extracted independently from upstream AST.
    """
    config = json.loads(Path(config_path).read_text())
    if (config["model_type"] != "qwen3" or config["torch_dtype"] != "bfloat16" or config["hidden_act"] != "silu"
            or config["rope_scaling"] is not None or config["sliding_window"] is not None
            or config["attention_bias"] or ("quantization_config" in config and config["quantization_config"] is not None)):
        raise NotImplementedError("source reference requires ordinary dense BF16/default-RoPE Qwen3")
    source, ledger = _Source(upstream_root), _Ledger()
    source.validate_model_bindings()
    caller_types = {name: type("Reference" + name, (), {"__call__": source.forward(name)})
        for name in ("Qwen2MLP", "Qwen3Attention", "Qwen3DecoderLayer", "Qwen2Model", "Qwen3ForCausalLM")}
    model_fields = source.fields("Qwen2Model", {"config": config})
    make_layers, = (node for node in ast.walk(model_fields.function) if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "make_layers")
    factory = make_layers.args[1]
    if (not isinstance(factory, ast.Lambda) or not isinstance(factory.body, ast.Call)
            or ast.unparse(factory.body.func) != "decoder_layer_type" or factory.body.args
            or {item.arg: ast.unparse(item.value) for item in factory.body.keywords} != {
                "config": "config", "cache_config": "cache_config", "quant_config": "quant_config", "prefix": "prefix"}):
        raise ValueError("source make_layers decoder factory changed")
    layers = model_fields.value(make_layers.args[0])
    queries, contexts, slots = tuple(query_lengths), tuple(kv_lengths), tuple(slot_mapping)
    rows = sum(queries)
    if (not queries or len(queries) != len(contexts) or any(type(q) is not int or q <= 0 for q in queries)
            or any(type(k) is not int or k < q for q, k in zip(queries, contexts, strict=True))
            or type(cache_capacity_tokens) is not int or cache_capacity_tokens < sum(contexts)
            or len(slots) != rows or len(set(slots)) != rows
            or any(type(slot) is not int or not 0 <= slot < cache_capacity_tokens for slot in slots)):
        raise ValueError("reference requires aligned queries/total KV lengths and valid logical cache slots")
    rope = None
    model = caller_types["Qwen2Model"]()
    model.start_layer, model.end_layer = 0, layers
    model.aux_hidden_state_layers, model._aux_upstream_total_cached = (), 0
    model.embed_input_ids = source.forward("Qwen2Model", "embed_input_ids").__get__(model)
    for helper in ("_maybe_add_hidden_state", "collect_remote_aux_hidden_states"):
        setattr(model, helper, source.forward("EagleModelMixin", helper).__get__(model))
    embedding, _, _ = model_fields.call("self.embed_tokens", "VocabParallelEmbedding")
    model.embed_tokens = _Leaf(ledger, "embedding", "embedding", embedding[1], embedding[0])
    norm, norm_kw, _ = model_fields.call("self.norm", "RMSNorm")
    model.norm = _Leaf(ledger, "final-norm", "norm", norm[0], model_fields.keyword(norm_kw, "eps"))
    model.layers = []
    for layer_index in range(layers):
        prefix = f"layer-{layer_index}"
        decoder_fields = source.fields("Qwen3DecoderLayer", {"config": config,
            "AttentionType": {"DECODER": "decoder", "ENCODER_ONLY": "encoder_only"}, "per_layer_sliding_window": None})
        _, attention_kw, _ = decoder_fields.call("self.self_attn", "Qwen3Attention")
        if (decoder_fields.keyword(attention_kw, "attn_type") != "decoder"
                or decoder_fields.keyword(attention_kw, "dual_chunk_attention_config") is not None
                or decoder_fields.keyword(attention_kw, "per_layer_sliding_window") is not None):
            raise NotImplementedError("reference requires causal attention without dual-chunk/sliding-window branches")
        attention_parameters = {key: decoder_fields.keyword(attention_kw, key)
            for key in ("hidden_size", "num_heads", "num_kv_heads", "head_dim", "qkv_bias", "rms_norm_eps", "max_position", "attn_type")}
        attention_parameters.update(Attention="decoder_attention", EncoderOnlyAttention="encoder_only_attention",
            AttentionType={"DECODER": "decoder", "ENCODER_ONLY": "encoder_only"})
        attention_fields = source.fields("Qwen3Attention", attention_parameters)
        attention = caller_types["Qwen3Attention"]()
        attention.head_dim = attention_fields.value(attention_fields.expression("self.head_dim"))
        attention.q_size = attention_fields.value(attention_fields.expression("self.q_size"))
        attention.kv_size = attention_fields.value(attention_fields.expression("self.kv_size"))
        args, keywords, _ = attention_fields.call("self.qkv_proj", "QKVParallelLinear")
        attention.qkv_proj = _Leaf(ledger, prefix + ":qkv", "linear", args[0], (args[2] + 2 * args[3]) * args[1],
            attention_fields.keyword(keywords, "bias"))
        args, keywords, _ = attention_fields.call("self.o_proj", "RowParallelLinear")
        attention.o_proj = _Leaf(ledger, prefix + ":o-proj", "linear", *args, attention_fields.keyword(keywords, "bias"))
        for role, label in (("q_norm", "q-norm"), ("k_norm", "k-norm")):
            args, keywords, _ = attention_fields.call("self." + role, "RMSNorm")
            setattr(attention, role, _Leaf(ledger, prefix + ":" + label, "norm", args[0], attention_fields.keyword(keywords, "eps")))
        heads = attention_fields.value(attention_fields.expression("self.num_heads"))
        kv_heads = attention_fields.value(attention_fields.expression("self.num_kv_heads"))
        rope_args, rope_kw, _ = attention_fields.call("self.rotary_emb", "get_rope")
        rope_shape = (attention_fields.keyword(rope_kw, "max_position"), rope_args[0])
        if max(contexts) > rope_shape[0]:
            raise ValueError("reference context exceeds the source RoPE cache")
        if rope is None:
            rope = _Tensor(rope_shape)
        elif rope.shape != rope_shape:
            raise NotImplementedError("reference requires shared default RoPE geometry across layers")
        attention.rotary_emb = _Leaf(ledger, prefix + ":rotary", "rotary", heads, kv_heads, attention.head_dim, rope)
        if attention_fields.value(attention_fields.expression("attn_cls")) != "decoder_attention":
            raise NotImplementedError("reference only binds the source decoder attention class")
        attn_args, attn_kw, _ = attention_fields.call("self.attn", "attn_cls")
        attn_heads, attn_width, attn_scale = attn_args
        attn_kv_heads = attention_fields.keyword(attn_kw, "num_kv_heads")
        if (attn_scale != attn_width ** -0.5 or attention_fields.keyword(attn_kw, "attn_type") != "decoder"):
            raise NotImplementedError("reference requires unmodified default causal attention scaling")
        cache = _Tensor((2, cache_capacity_tokens, attn_kv_heads, attn_width))
        attention.attn = _Leaf(ledger, prefix + ":attention", "attention", attn_heads, attn_kv_heads, attn_width,
            queries, contexts, tuple(slot_mapping), cache)
        decoder = caller_types["Qwen3DecoderLayer"]()
        decoder.self_attn = attention
        for role, label in (("input_layernorm", "input-norm"), ("post_attention_layernorm", "post-attn-norm")):
            args, keywords, _ = decoder_fields.call("self." + role, "RMSNorm")
            setattr(decoder, role, _Leaf(ledger, prefix + ":" + label, "norm", args[0], decoder_fields.keyword(keywords, "eps")))
        _, mlp_kw, _ = decoder_fields.call("self.mlp", "Qwen3MLP")
        mlp_parameters = {key: decoder_fields.keyword(mlp_kw, key) for key in ("hidden_size", "intermediate_size", "hidden_act")}
        if mlp_parameters["hidden_act"] != "silu":
            raise NotImplementedError("source MLP binding requires SiluAndMul")
        mlp_fields = source.fields("Qwen2MLP", mlp_parameters)
        mlp = caller_types["Qwen2MLP"]()
        args, keywords, _ = mlp_fields.call("self.gate_up_proj", "MergedColumnParallelLinear")
        mlp.gate_up_proj = _Leaf(ledger, prefix + ":gate-up", "linear", args[0], sum(args[1]), mlp_fields.keyword(keywords, "bias"))
        args, keywords, _ = mlp_fields.call("self.down_proj", "RowParallelLinear")
        mlp.down_proj = _Leaf(ledger, prefix + ":down", "linear", *args, mlp_fields.keyword(keywords, "bias"))
        mlp_fields.expression("self.act_fn", "SiluAndMul")
        mlp.act_fn = _Leaf(ledger, prefix + ":silu-mul", "activation")
        decoder.mlp = mlp
        model.layers.append(decoder)
    root = caller_types["Qwen3ForCausalLM"]()
    root.model = model
    hidden = root(_Tensor((rows,), "int32"), _Tensor((rows,), "int64"))
    forward_calls = len(ledger.calls)
    sampled = _target_source_forward(source, ledger, root, hidden, queries, config)
    if sampled.shape != (len(queries), 1) or sampled.dtype != "int32":
        raise ValueError("source target output must be INT32 [requests,1]")
    return {"calls": ledger.calls, "source_methods": source.bindings,
        "upstream_forward_driven_calls": forward_calls, "declared_suffix_calls": len(ledger.calls) - forward_calls,
        "target_suffix_source_bound": True,
        "source_paths": {name: str(path) for name, path in source.paths.items()},
        "layers": layers, "scope": "source-bound model/logits/plain-greedy caller interfaces and worker row selection; logical primitive contracts, not provider internals",
        "reference_execution": "extracted_upstream_python_methods_on_independent_metadata_no_tensor_values",
        "declared_inputs": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1, "aux_hidden_states": False,
            "mtp": False, "dtype": "bf16", "norm_and_rope": "declared_cuda_inplace",
            "all_greedy": True, "requested_logprobs": False, "penalties": False,
            "custom_logits_processors": False, "allowed_token_mask": False, "bad_words": False,
            "thinking_budget_state": False, "head_dtype_override": None},
        "complete_worker_memory_accepted": False, "runtime_accepted": False}


def compare_qwen3_source_graph(reference, trace):
    """Compare every emitted caller boundary, including extras/missing calls."""
    graph, views = trace.graph, {view.view_id: view for view in trace.views}
    expected = {trace.qualify(row["call"]): row for row in reference["calls"]}
    if len(expected) != len(reference["calls"]):
        raise ValueError("reference caller identities must be unique")
    actual = {node.node_id: node for node in graph.nodes}
    checks = []
    for name in sorted(set(expected) | set(actual)):
        if name not in expected or name not in actual:
            checks.append({"call": name, "status": "mismatched", "differences": ["unexpected" if name not in expected else "missing"]})
            continue
        row, node = expected[name], actual[name]
        inputs = tuple(views[value] for value in node.attributes["symbolic_tensor_inputs"])
        outputs = tuple(views[value] for value in node.attributes["symbolic_tensor_outputs"])
        fields = {"kind": node.kind.value == row["kind"], "shape": dict(node.shape) == row["shape"],
            "dtype": node.dtype == row["dtype"], "variant": node.variant == row["variant"],
            "inputs": tuple((view.shape, view.dtype) for view in inputs) == row["inputs"],
            "outputs": tuple((view.shape, view.dtype) for view in outputs) == row["outputs"],
            "parents": graph.predecessors(name) == tuple(sorted(trace.qualify(parent) for parent in row["parents"])),
            "alias": tuple(tuple(output.allocation_id == value.allocation_id for value in inputs) for output in outputs) == row["output_alias_inputs"],
            "attributes": all(key in node.attributes and node.attributes[key] == value for key, value in row["attributes"].items())}
        checks.append({"call": name, "status": "matched" if all(fields.values()) else "mismatched",
            "differences": [key for key, matched in fields.items() if not matched]})
    return {"scope": reference["scope"], "reference_calls": len(expected), "actual_calls": len(actual), "checks": checks,
        "upstream_forward_driven_calls": reference["upstream_forward_driven_calls"],
        "declared_suffix_calls": reference["declared_suffix_calls"],
        "target_suffix_source_bound": reference["target_suffix_source_bound"],
        "total": len(checks), "matched": sum(row["status"] == "matched" for row in checks),
        "mismatched": sum(row["status"] != "matched" for row in checks), "not_checked": 0,
        "runtime_accepted": False, "complete_worker_memory_accepted": False}
