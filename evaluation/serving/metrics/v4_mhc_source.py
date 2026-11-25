"""Genuine mHC wrappers and their independent logical stage accounts."""

from contextlib import contextmanager
import ast
from types import SimpleNamespace

from evaluation.serving.simulation.v4_flash_mhc_reference import (
    _fused_config, _validate_prenorm_selection, mhc_stage_reference)

from .qwen3_source import _Tensor


MHC_SOURCE_FILES = {
    "mhc_tilelang.py": "vllm/model_executor/kernels/mhc/tilelang.py",
    "mhc_kernels.py": "vllm/model_executor/kernels/mhc/tilelang_kernels.py",
}


def _canonical(value):
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


class MHCSource:
    def __init__(self, source, calls, raw, rank, rows, *, multiprocessors, source_use_deep_gemm):
        if type(source_use_deep_gemm) is not bool or type(multiprocessors) is not int or multiprocessors <= 0:
            raise ValueError("mHC reference requires declared environment flag and specification SM count")
        self.source, self.calls, self.raw, self.rank, self.rows = source, calls, raw, rank, rows
        self.use_deep_gemm, self.forced, self.active, self.counts = source_use_deep_gemm, False, None, {}
        symbols = {"mhc_pre": "mhc_pre_broadcast_tilelang", "mhc_post_pre": "mhc_fused_post_pre_tilelang",
            "mhc_post": "mhc_post_tilelang", "hc_head": "hc_head_fused_kernel_tilelang"}
        self.bindings = {role: {"symbol": symbol, "line": next(node.lineno
            for node in source.trees["mhc_tilelang.py"].body if isinstance(node, ast.FunctionDef) and node.name == symbol)}
            for role, symbol in symbols.items()}
        wrapper = source.paths["mhc_tilelang.py"].read_text()
        _validate_prenorm_selection((source.root / "vllm/utils/deep_gemm.py").read_text(), wrapper)
        self.split_config = _fused_config(source.paths["mhc_kernels.py"].read_text())
        torch = SimpleNamespace(float32="fp32", bfloat16="bf16", empty=self.empty,
            empty_like=lambda tensor: _Tensor(tensor.shape, tensor.dtype),
            cuda=SimpleNamespace(get_device_properties=lambda device: SimpleNamespace(multi_processor_count=multiprocessors)))
        divide = lambda left, right: -(-left // right)
        self.compute_split = source.function("mhc_kernels.py", "compute_num_split",
            metadata_globals={"torch": torch, "cdiv": divide})
        self.prenorm = source.function("mhc_tilelang.py", "_hc_prenorm_gemm_outputs",
            metadata_globals={"torch": torch, "cdiv": divide}, metadata_imports={
                (0, "vllm.model_executor.kernels.mhc.tilelang_kernels", "compute_num_split"): self.compute_split,
                (0, "vllm.utils.deep_gemm", "is_deep_gemm_supported"): self.supported,
                (0, "vllm.utils.deep_gemm", "tf32_hc_prenorm_gemm"): self.deep_prenorm,
                (0, "vllm.model_executor.kernels.mhc.tilelang_kernels", "_HC_PRENORM_GEMM_TILELANG_KERNEL"): self.tile_prenorm})
        globals = {"torch": torch, "_hc_prenorm_gemm_outputs": self.prenorm_call}
        module = "vllm.model_executor.kernels.mhc.tilelang_kernels"
        self.broadcast_raw = source.function("mhc_tilelang.py", "mhc_pre_broadcast_tilelang",
            metadata_globals=globals, metadata_imports={(0, module, "_MHC_PRE_BIG_FUSE_TILELANG_KERNEL"): self.norm})
        self.fused_raw = source.function("mhc_tilelang.py", "mhc_fused_post_pre_tilelang",
            metadata_globals=globals, metadata_imports={
                (0, module, "_MHC_FUSED_TILELANG_KERNEL"): self.fused_stage,
                (0, module, "_MHC_POST_TILELANG_KERNEL"): self.post_stage,
                (0, module, "_MHC_PRE_BIG_FUSE_TILELANG_KERNEL"): self.norm,
                (0, module, "mhc_fused_post_pre_split_config"): self.split_config})
        self.post_raw = source.function("mhc_tilelang.py", "mhc_post_tilelang", metadata_globals=globals,
            metadata_imports={(0, module, "_MHC_POST_TILELANG_KERNEL"): self.post_stage})
        self.head_raw = source.function("mhc_tilelang.py", "hc_head_fused_kernel_tilelang", metadata_globals=globals,
            metadata_imports={(0, module, "_HC_HEAD_FUSED_TILELANG_KERNEL"): self.head_stage})

    @staticmethod
    def empty(*shape, dtype, device=None):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return _Tensor(shape, dtype)

    def supported(self):
        if not self.use_deep_gemm or self.forced:
            # forced broadcast has (... or not use_tilelang_fallback)=True.
            # The left operand cannot alter this source Boolean result.
            return False
        raise ValueError("nonfused mHC requires unbound has_deep_gemm/platform support")

    def prenorm_call(self, x, fn, *, hidden_size, hc_mult, use_tilelang_fallback=True):
        old = self.forced
        self.forced = not use_tilelang_fallback
        try:
            return self.prenorm(x, fn, hidden_size=hidden_size, hc_mult=hc_mult,
                use_tilelang_fallback=use_tilelang_fallback)
        finally:
            self.forced = old

    @contextmanager
    def invocation(self, role):
        if self.active is not None:
            raise ValueError("mHC reference cannot nest wrapper invocations")
        if role not in self.counts:
            self.counts[role] = 0
        ordinal = self.counts[role]
        self.counts[role] += 1
        row = {"rank": self.rank, "role": role, "ordinal": ordinal, "stages": [],
            "source": {"path": str(self.source.paths["mhc_tilelang.py"]),
                **self.bindings[role],
                "scope": "original wrapper control, metadata tensors and logical device stage ports; provider unbound"}}
        self.active = row
        try:
            yield
            self.rows.append(row)
        finally:
            self.active = None

    def emit(self, stage, tokens, *, splits=1, tile=None):
        if self.active is None:
            raise ValueError("mHC device port has no original wrapper invocation")
        work = mhc_stage_reference(self.raw, stage, tokens, splits=splits, projection_tile=tile)
        self.active["stages"].append(work)

    def deep_prenorm(self, x, fn, out, sqrsum, splits):
        if (x.dtype != "bf16" or fn.dtype != "fp32" or out.dtype != "fp32" or sqrsum.dtype != "fp32"
                or out.shape != (splits, x.shape[0], fn.shape[0]) or sqrsum.shape != (splits, x.shape[0])
                or x.shape[-1] != fn.shape[-1]):
            raise ValueError("mHC prenorm partials must remain FP32")
        stage = "mhc_prenorm_broadcast" if self.forced else "mhc_prenorm"
        self.emit(stage, x.shape[0], splits=splits)

    def tile_prenorm(self, x, fn, out, sqrsum, hidden, mult):
        if (x.dtype != "bf16" or fn.dtype != "fp32" or out.dtype != "fp32" or sqrsum.dtype != "fp32"
                or out.shape != (1, x.shape[0], fn.shape[0]) or sqrsum.shape != (1, x.shape[0])
                or x.shape[-1] != fn.shape[-1]):
            raise ValueError("TileLang mHC prenorm writes one FP32 partial")
        self.emit("mhc_prenorm", x.shape[0])

    def fused_stage(self, comb, residual, post, x, fn, mult, hidden, mix):
        tokens = x.shape[0]
        config = self.split_config(tokens, hidden, mult)
        if config is None:
            raise ValueError("mHC fused device port received a nonfused source shape")
        tile, splits, unused_threads = config
        self.emit("mhc_post_prenorm", tokens, splits=splits, tile=tile)
        return _Tensor((splits, tokens, mix), "fp32"), _Tensor((splits, tokens), "fp32"), _Tensor(residual.shape)

    def post_stage(self, comb, residual, post, x, out, mult, hidden):
        if out.shape != residual.shape or out.dtype != "bf16":
            raise ValueError("mHC post output must be a BF16 residual")
        self.emit("mhc_post", x.shape[0])

    def norm(self, mixes, sqrsum, scale, base, residual, post, comb, layer_input,
             rms_eps, pre_eps, sinkhorn_eps, post_multiplier, repeats, *, residual_out=None, norm_weight=None, norm_eps=1e-6):
        t, h, c = layer_input.shape[0], self.raw["hidden_size"], self.raw["hc_mult"]
        if (mixes.dtype != "fp32" or sqrsum.dtype != "fp32" or post.dtype != "fp32" or comb.dtype != "fp32"
                or layer_input.dtype != "bf16" or repeats != self.raw["hc_sinkhorn_iters"] or norm_weight is None
                or mixes.shape != (mixes.shape[0], t, c * (c + 2)) or sqrsum.shape != (mixes.shape[0], t)
                or post.shape != (t, c) or comb.shape != (t, c * c) or layer_input.shape != (t, h)
                or norm_weight.shape != (h,) or norm_weight.dtype != "bf16"):
            raise ValueError("mHC norm source partial/output types or Sinkhorn count changed")
        stage = "mhc_pre_norm_broadcast" if residual_out is not None else "mhc_pre_norm"
        self.emit(stage, layer_input.shape[0], splits=mixes.shape[0])

    def head_stage(self, residual, fn, scale, base, out, hidden, rms_eps, hc_eps, mult):
        self.emit("hc_head", residual.shape[0])

    def broadcast(self, x, fn, scale, base, *scalars, **options):
        def run(x, residual, post, comb):
            with self.invocation("mhc_pre"):
                return self.broadcast_raw(x, fn, scale, base, *scalars, **options)
        return self.calls.bind(self.rank, "mhc_pre", run)(x, None, None, None)

    def fused(self, x, residual, post, comb, *scalars, **options):
        def run(x, residual, post, comb):
            with self.invocation("mhc_post_pre"):
                return self.fused_raw(x, residual, post, comb, *scalars, **options)
        return self.calls.bind(self.rank, "mhc_post_pre", run)(x, residual, post, comb)

    def post(self, x, residual, post, comb):
        def run(x, residual, post, comb):
            with self.invocation("mhc_post"):
                return self.post_raw(x, residual, post, comb)
        return self.calls.bind(self.rank, "mhc_post", run)(x, residual, post, comb)

    def head(self, x, *scalars):
        def run(x, residual, post, comb):
            with self.invocation("hc_head"):
                return self.head_raw(x, *scalars)
        return self.calls.bind(self.rank, "hc_head", run)(x, None, None, None)


def compare_v4_mhc_work(reference, observation, plan, hardware):
    from src.kernel.kernel_base import create_predictor
    from src.kernel.predictors.vector import mhc_stage_logical_work
    from src.serving.kernel_lowering.kernel_plan import hardware_for_call

    names = ("stage", "operands", "read_bytes", "write_bytes", "demand", "semantic_fields", "views",
        "unpriced_typed_actions", "nonlinear_function_counts")
    expected, observed = {}, {}
    for row in reference["mhc_work"]:
        key = row["rank"], row["role"], row["ordinal"]
        if key in expected:
            raise ValueError("source repeats a mHC invocation identity")
        expected[key] = row
    roles = {"mhc_pre", "mhc_post_pre", "mhc_post", "hc_head"}
    for row in observation.calls.rows:
        if row["role"] in roles:
            key = row["rank"], row["role"], row["ordinal"]
            if key in observed:
                raise ValueError("actual repeats a mHC invocation identity")
            observed[key] = row["node_id"]
    checks = []
    for key in sorted(set(expected) | set(observed)):
        calls = ([] if key not in observed else [call for call in plan.calls if call.rank == key[0]
            and observed[key] in call.source_node_ids])
        stages = [] if key not in expected else expected[key]["stages"]
        count = max(len(calls), len(stages), 1)
        for index in range(count):
            want = None if index >= len(stages) else _canonical({name: stages[index][name] for name in names})
            got = None
            if index < len(calls):
                call = calls[index]
                work = mhc_stage_logical_work(call.kernel_spec)
                got = _canonical({name: work[name] for name in names})
                # The prenorm helper includes the unused vector group,
                # while the independent scalar/MAC ledger omits it.
                # Only an exact zero is representation-equivalent.
                if "compute_vector" in got["demand"] and got["demand"]["compute_vector"] == 0:
                    del got["demand"]["compute_vector"]
                if call.kernel_spec.variant not in ("mhc_prenorm", "mhc_prenorm_broadcast"):
                    owner, _ = create_predictor(call.kernel_spec.kind).resource_input(call.kernel_spec, hardware_for_call(call, hardware))
                    got["demand"] = {name: owner.demand_per_resource[name] for name in work["demand"]}
                    for access in ("read", "write"):
                        got[access + "_bytes"] = sum(node.bytes for path in owner.memory_paths if path.access_kind == access
                            for node in path.nodes if node.resource_role == "l2_cache")
            differences = (["unexpected"] if want is None else ["missing"] if got is None else
                [name for name in names if want[name] != got[name]])
            if index and index < len(calls) and calls[index - 1].call_id not in calls[index].dependencies:
                differences.append("previous_stage_dependency")
            checks.append({"rank": key[0], "role": key[1], "ordinal": key[2], "stage_index": index,
                "expected": want, "actual": got, "differences": differences,
                "status": "mismatched" if differences else "matched",
                "source": expected[key]["source"] if key in expected else None})
    return {"scope": "original mHC wrappers, stage order and logical typed math/payload; no native provider proof",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "limitations": ("stage algebra/unique operands, not compiler or physical traffic",
            "global scratch mapping, allocation lifetimes and numerical function implementations remain separate",
            "forced DeepGEMM broadcast is a source call, not installed provider acceptance"),
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}
