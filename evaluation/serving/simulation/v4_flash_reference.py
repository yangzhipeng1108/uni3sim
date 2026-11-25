"""Independent vLLM source reference for Flash alignment.

This module intentionally imports only the standard library. Expected facts
come from the raw official config and upstream source, never from a candidate
workload, lowering, launch selector or predictor. Projection coverage is the
first implemented part of the full-model reference; unimplemented facets are
explicit requirements, not an implicit pass.
"""

import ast
import hashlib
import json
from pathlib import Path
import re

from .v4_flash_mhc_reference import attach_mhc_accounts, mhc_wrapper_object_reference
from .v4_flash_routing_reference import routing_source_reference, routing_tensor_reference
from .v4_flash_indexer_reference import indexer_baseline_reference
from .v4_flash_attention_reference import attach_attention_output_regions


_ATTENTION = "vllm/models/deepseek_v4/attention.py"
_MODEL = "vllm/models/deepseek_v4/nvidia/model.py"
_COMPRESSOR = "vllm/models/deepseek_v4/compressor.py"
_VOCAB = "vllm/model_executor/layers/vocab_parallel_embedding.py"
_GATE_LINEAR = "vllm/model_executor/layers/fused_moe/router/gate_linear.py"
_LL_BF16 = "vllm/model_executor/kernels/linear/cute_dsl/ll_bf16.py"


def _official_config_identity(raw):
    # Identity-only data stays in its existing unique owner. Do not import
    # or execute its parser or derive any expected operator from that code.
    owner = Path(__file__).resolve().parents[3] / "src/serving/workload/model_config.py"
    tree = ast.parse(owner.read_text(encoding="utf-8"))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "_TARGET_MODEL_PROVENANCE"
                           for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("official identity data owner is missing or ambiguous")
    identity = ast.literal_eval(assignments[0].value)["deepseek_v4"]
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if digest != identity["canonical_config_sha256"]:
        raise ValueError("independent Flash reference requires the exact official config")
    return {"identity_owner": str(owner), "identity_data": identity}


def _definition(tree, name):
    current = tree
    for component in name.split("."):
        matches = [node for node in current.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == component]
        if len(matches) != 1:
            raise ValueError(f"source definition missing or ambiguous: {name}")
        current = matches[0]
    return current


def _constructor(root, relative, owner, member, callee):
    path = root / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definition = _definition(tree, owner + ".__init__")
    matches = []
    for node in ast.walk(definition):
        if isinstance(node, ast.Assign) and any(
                ast.unparse(target) == "self." + member for target in node.targets):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == callee:
                matches.append(node)
    if len(matches) != 1:
        raise ValueError(f"source constructor missing or ambiguous: {owner}.{member}")
    node = matches[0]
    return {"path": str(path), "line": node.lineno, "end_line": node.end_lineno,
            "symbol": owner + ".__init__." + member, "expression": ast.unparse(node),
            "arguments": [ast.unparse(arg) for arg in node.value.args],
            "keywords": {kw.arg: ast.unparse(kw.value) for kw in node.value.keywords}}


def _require_keywords(anchor, expected):
    for name, value in expected.items():
        if name not in anchor["keywords"] or anchor["keywords"][name] != value:
            raise ValueError(f"upstream contract changed: {anchor['symbol']}.{name}")


def _function_anchor(root, relative, symbol):
    path = root / relative
    node = _definition(ast.parse(path.read_text(encoding="utf-8")), symbol)
    return {"path": str(path), "line": node.lineno, "end_line": node.end_lineno, "symbol": symbol}


def _constexpr_body(source, symbol, value):
    """Specialize an explicitly declared boolean template condition.

    Only brace-delimited ``if constexpr (symbol)`` (with optional else) is
    accepted. Other conditions remain source text; they are not guessed.
    """
    if type(value) is not bool:
        raise ValueError("source constexpr specialization requires an explicit boolean")
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    marker = re.compile(r"if\s+constexpr\s*\(\s*" + re.escape(symbol) + r"\s*\)")

    def block(start):
        while start < len(source) and source[start].isspace():
            start += 1
        if start == len(source) or source[start] != "{":
            raise ValueError("source constexpr specialization requires an explicit block")
        end, depth = start + 1, 1
        while depth and end < len(source):
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        if depth:
            raise ValueError("source constexpr specialization block is unterminated")
        return source[start + 1:end - 1], end

    while (match := marker.search(source)) is not None:
        true_body, end = block(match.end())
        tail = end
        while tail < len(source) and source[tail].isspace():
            tail += 1
        false_body = ""
        if source[tail:tail + 4] == "else":
            false_body, end = block(tail + 4)
        replacement = true_body if value else false_body
        source = source[:match.start()] + replacement + source[end:]
    return source


def _false_constexpr_body(source, symbol):
    return _constexpr_body(source, symbol, False)


def _cpp_anchor(root, relative, signature):
    path = root / relative
    text = path.read_text(encoding="utf-8")
    start = text.index(signature)
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        if end == len(text):
            raise ValueError("unterminated upstream C++ function")
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return {"path": str(path), "line": text[:start].count("\n") + 1,
            "end_line": text[:end].count("\n") + 1, "symbol": signature,
            "source": text[start:end]}


def _argmax_anchor(include):
    header = Path(include) / "ATen/native/SharedReduceOps.h"
    version_file = Path(include).parent / "version.py"
    tree = ast.parse(version_file.read_text(encoding="utf-8"))
    versions = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
                and isinstance(node.value, ast.Constant)]
    if len(versions) != 1 or not versions[0].startswith("2.11."):
        raise ValueError("V4 argmax source reference requires a genuine Torch 2.11 header")
    text = header.read_text(encoding="utf-8")
    start = text.index("struct GreaterOrNan {")
    end = text.index("\n};", start) + 3
    body = text[start:end]
    compact = re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", body))
    expected = ("structGreaterOrNan{C10_DEVICEbooloperator()(scalar_ta,scalar_tb,int64_tidx_a,int64_tidx_b)const{"
                "if(at::_isnan(a)){if(at::_isnan(b)){returnidx_a<idx_b;}returntrue;}"
                "return(a==b)?idx_a<idx_b:(a>b);}};")
    if compact != expected:
        raise ValueError("ATen GreaterOrNan source changed; independently rederive pair order")
    return {"path": str(header), "line": text[:start].count("\n") + 1,
            "symbol": "at::native::detail::GreaterOrNan", "source": body,
            "torch_version": versions[0], "version_source": str(version_file),
            "scope": "header semantics only, not loaded provider/compiler/runtime"}


def argmax_pair_source_work(rows, vocab):
    """Minimal pair tree and typed branches of the anchored GreaterOrNan.

    Values select among these branches; shape does not select one branch or
    establish padded/library work. This is shared by both source references.
    """
    if any(type(value) is not int or value <= 0 for value in (rows, vocab)):
        raise ValueError("argmax source work requires positive row/vocabulary extents")
    merges = rows * (vocab - 1)
    return {"logical_pair_merges": merges, "value_dtype": "fp32", "index_dtype": "int64",
        "maximum_fp32_predicates": 3 * merges,
        "logical_primitive_count_bounds": {"fp32_isnan": [merges, 2 * merges],
            "fp32_equal": [0, merges], "fp32_greater": [0, merges], "int64_less": [0, merges],
            "pair_choice": [merges, merges]},
        "typed_primitive_resource_mapping_complete": False, "compiler_or_actual_library_work_complete": False,
        "branches_per_pair_merge": [
            {"condition": "a_is_nan_and_b_is_nan", "fp32_isnan": 2, "fp32_equal": 0,
             "fp32_greater": 0, "int64_less": 1, "pair_choice": 1},
            {"condition": "a_is_nan_and_b_is_not_nan", "fp32_isnan": 2, "fp32_equal": 0,
             "fp32_greater": 0, "int64_less": 0, "pair_choice": 1},
            {"condition": "a_is_not_nan_and_a_equals_b", "fp32_isnan": 1, "fp32_equal": 1,
             "fp32_greater": 0, "int64_less": 1, "pair_choice": 1},
            {"condition": "a_is_not_nan_and_a_differs_from_b_including_nan_b", "fp32_isnan": 1,
             "fp32_equal": 1, "fp32_greater": 1, "int64_less": 0, "pair_choice": 1}]}


def _flashinfer_quantizer_anchor(root):
    """Read a real installed SDK's source without loading a CUDA provider."""
    root = Path(root)
    metadata = ast.parse((root / "_build_meta.py").read_text(encoding="utf-8"))
    values = {target.id: ast.literal_eval(node.value) for node in metadata.body if isinstance(node, ast.Assign)
              for target in node.targets if isinstance(target, ast.Name) and target.id in ("__version__", "__git_version__")}
    if set(values) != {"__version__", "__git_version__"} or any(not isinstance(value, str) or not value for value in values.values()):
        raise ValueError("FlashInfer source binding requires genuine package build metadata")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9.+-]*", values["__version__"]) or not re.fullmatch(r"[0-9a-f]{40}", values["__git_version__"]):
        raise ValueError("FlashInfer source binding requires a release version and full source revision")
    relative = "data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh"
    quant = _cpp_anchor(root, relative, "__global__ void scale_1x128_kernel(")
    warp_max = _cpp_anchor(root, relative, "__forceinline__ __device__ T find_max_elem_in_warp(")
    run = _cpp_anchor(root, relative, "void fp8_gemm_run(__nv_bfloat16 const* mat_a, __nv_fp8_e4m3* fp8_mat_a,")
    runner_relative = relative.replace("fp8_blockscale_gemm_kernel.cuh", "fp8_blockscale_gemm.cu")
    header_relative = relative.replace("fp8_blockscale_gemm_kernel.cuh", "fp8_blockscale_gemm.h")
    runner_class = _cpp_anchor(root, header_relative, "class CutlassFp8BlockScaleGemmRunner : public")
    runner = _cpp_anchor(root, runner_relative,
        "void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::gemm(")
    binding_relative = "data/csrc/fp8_blockscale_gemm_sm90_binding.cu"
    runner_constructor = _cpp_anchor(root, binding_relative, "Fp8BlockScaleGemmRunner()")
    binding = _cpp_anchor(root, binding_relative,
        "kernels::CutlassFp8BlockScaleGemmRunnerInterface* selectRunner(")
    workspace_binding = _cpp_anchor(root, binding_relative, "int64_t getWorkspaceSize(")
    workspace_runner = _cpp_anchor(root, runner_relative,
        "size_t CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::getWorkspaceSizeBase(")
    padded_relative = "data/csrc/nv_internal/tensorrt_llm/deep_gemm/scheduler.cuh"
    padded_offset = _cpp_anchor(root, padded_relative,
        "__host__ __device__ __forceinline__ T_offset compute_padded_offset(")
    configure = _cpp_anchor(root, binding_relative, "void configureWorkspace(")
    workspace_source = re.sub(r"\s+", "", workspace_binding["source"] + workspace_runner["source"])
    for fragment in ("runner_bf16_bf16_->getWorkspaceSizeBase", "runner_bf16_fp8_->getWorkspaceSizeBase",
        "runner_fp8_fp8_->getWorkspaceSizeBase", "std::max(max_shape_m_4_align_,int64_t(div_up(max_shape_m,4)*4))", "if(expected_m_==0)"):
        if fragment not in workspace_source:
            raise ValueError("FlashInfer cached workspace state contract changed")
    reserve_source = re.sub(r"\s+", "", workspace_runner["source"])
    for fragment in ("deep_gemm::compute_padded_offset(max_shape_m,num_problems)",
        "max_shape_m_4_align_*shape_k*sizeof(__nv_fp8_e4m3)",
        "max_shape_m_32_align_padded_*div_up(shape_k,128)*sizeof(float)",
        "num_problems*shape_n*shape_k*sizeof(__nv_fp8_e4m3)",
        "num_problems*div_up(shape_k,128)*div_up(shape_n,128)*sizeof(float)"):
        if fragment not in reserve_source:
            raise ValueError("FlashInfer workspace reserve arithmetic changed")
    padded_source = re.sub(r"\s+", "", padded_offset["source"])
    if ("alignment=32;" not in padded_source
            or "return(offset+problem_idx*(alignment-1))/alignment*alignment;" not in padded_source):
        raise ValueError("FlashInfer workspace padded-offset arithmetic changed")
    if any(name + "->configureWorkspace(workspace_ptr)" not in re.sub(r"\s+", "", configure["source"])
           for name in ("runner_bf16_bf16_", "runner_bf16_fp8_", "runner_fp8_fp8_")):
        raise ValueError("FlashInfer workspace shared backing changed")
    sdk_wrapper = _definition(ast.parse((root / "gemm/gemm_base.py").read_text()), "fp8_blockscale_gemm_sm90")
    cached_runner = _definition(ast.parse((root / "gemm/gemm_base.py").read_text()), "get_fp8_blockscale_gemm_runner_sm90")
    if [ast.unparse(decorator) for decorator in cached_runner.decorator_list] != ["functools.cache"]:
        raise ValueError("FlashInfer runner cache lifetime changed")
    arguments = cached_runner.args
    if (arguments.posonlyargs or arguments.args or arguments.kwonlyargs
            or arguments.vararg is not None or arguments.kwarg is not None):
        raise ValueError("FlashInfer cached runner instance must not be keyed by layer/shape/device arguments")
    if "return module.init()" not in ast.unparse(cached_runner):
        raise ValueError("FlashInfer cached wrapper must retain one initialized runner object")
    if "int64_tmax_shape_m_4_align_=0;" not in re.sub(r"\s+", "", runner_class["source"]):
        raise ValueError("FlashInfer dtype runners must share the same initial maximum-M state")
    construction = re.sub(r"\s+", "", runner_constructor["source"])
    for left, right in (("__nv_bfloat16", "__nv_bfloat16"), ("__nv_bfloat16", "__nv_fp8_e4m3"), ("__nv_fp8_e4m3", "__nv_fp8_e4m3")):
        if f"CutlassFp8BlockScaleGemmRunner<{left},{right},__nv_bfloat16>" not in construction:
            raise ValueError("FlashInfer binding must construct all three cached dtype runners")
    sdk_source = ast.unparse(sdk_wrapper)
    for expression in ("out = torch.empty(M, N, dtype=out_dtype, device=input.device)",
        "workspace_size = runner.get_workspace_size(M, N, K)",
        "workspace = torch.empty(workspace_size, dtype=torch.uint8, device=input.device)", "runner.configure_workspace(workspace)"):
        if expression not in sdk_source:
            raise ValueError("FlashInfer output/workspace allocation source changed")
    if "elseif(!input_is_fp8&&weight_is_fp8)" not in re.sub(r"\s+", "", binding["source"]):
        raise ValueError("FlashInfer BF16/FP8 runner binding changed")
    runner_source = re.sub(r"\s+", "", runner["source"])
    for fragment in ("internal_quantize_a=!std::is_same_v<ElementA,__nv_fp8_e4m3>",
                     "internal_quantize_b=!std::is_same_v<ElementB,__nv_fp8_e4m3>",
                     "ifconstexpr(internal_quantize_a&&!internal_quantize_b)",
                     "fp8_mat_a=reinterpret_cast<__nv_fp8_e4m3*>(ws_ptr)",
                     "ws_ptr+=max_shape_m_4_align_*shape_k*sizeof(__nv_fp8_e4m3)",
                     "per_token_per_128c_scales=reinterpret_cast<float*>(ws_ptr)"):
        if fragment not in runner_source:
            raise ValueError("FlashInfer runner dtype/quantization dispatch changed")
    compact = re.sub(r"\s+", "", quant["source"])
    for fragment in ("scales_along_dim_x=div_up(dim_x,128)", "stride_scale_dim_y=div_up(dim_y,4)*4",
                     "ScaleTypevalue_1=ScaleType(input_frag2[i].x)*scale", "output_line[lane_id]=OutputType(value_1)",
                     "scales[(size_t)scales_idx_x*stride_scale_dim_y+scales_idx_y]=ScaleType(1.f/scale)",
                     "ScaleTypescale=amax!=InputType(0.f)?448.f/ScaleType(amax):1.f;"):
        if fragment not in compact:
            raise ValueError("FlashInfer scale_1x128 source changed; rederive its typed operand account")
    if "if(internal_quantize_a)" not in re.sub(r"\s+", "", run["source"]):
        raise ValueError("FlashInfer in-call activation quantizer condition changed")
    run_program = re.sub(r"\s+", "", run["source"])
    if run_program.index("scale_1x128_kernel<<<") > run_program.index("fp8_gemm_run(fp8_mat_a,"):
        raise ValueError("FlashInfer quantizer must precede the FP8 matrix application")
    warp_program = re.sub(r"\s+", "", warp_max["source"])
    if ("for(intoffset=16;offset>0;offset/=2)" not in warp_program
            or "value=T(std::max(float(value),__shfl_down_sync(0xFFFFFFFF,float(value),offset)))" not in warp_program
            or "value=T(__shfl_sync(0xffffffff,float(value),0))" not in warp_program):
        raise ValueError("FlashInfer BF16-amax warp max program changed")
    matrix_sources = _flashinfer_dense_matrix_anchors(root, relative)
    return {"definitions": [quant, warp_max, run, runner, runner_class, runner_constructor, binding, workspace_binding, workspace_runner, padded_offset, configure,
                _function_anchor(root, "gemm/gemm_base.py", "fp8_blockscale_gemm_sm90"),
                _function_anchor(root, "gemm/gemm_base.py", "get_fp8_blockscale_gemm_runner_sm90")], "package_root": str(root),
            "version": values["__version__"], "package_revision": values["__git_version__"],
            "source_sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
            "runner_source_sha256": hashlib.sha256((root / runner_relative).read_bytes()).hexdigest(),
            "runner_header_sha256": hashlib.sha256((root / header_relative).read_bytes()).hexdigest(),
            "binding_source_sha256": hashlib.sha256((root / binding_relative).read_bytes()).hexdigest(),
            "workspace_padding_source_sha256": hashlib.sha256((root / padded_relative).read_bytes()).hexdigest(),
            "python_wrapper_source_sha256": hashlib.sha256((root / "gemm/gemm_base.py").read_bytes()).hexdigest(),
            "build_metadata_sha256": hashlib.sha256((root / "_build_meta.py").read_bytes()).hexdigest(),
            "dense_matrix_storage": matrix_sources,
            "actual_provider_bound": False,
            "source_conditions": {"input_dtype": "bf16", "weight_dtype": "fp8",
                                  "internal_quantize_a": True, "internal_quantize_b": False,
                                  "workspace_allocation_capacity_bound": False},
            "scope": "installed SDK source program only; not selected vLLM0.30 provider, binary or runtime compatibility"}


def _flashinfer_dense_matrix_anchors(root, kernel_relative):
    """Bind all SM90 normal-dense matrix entry points and tensor arguments.

    Explicit source tensor temporaries are distinct from compiler spill/local
    memory, shared staging and JIT/compiler allocations. The latter are not
    accepted as known zero physical traffic by this source-only contract.
    """
    deep = "data/csrc/nv_internal/tensorrt_llm/deep_gemm/"
    signatures = (
        (kernel_relative, "void fp8_gemm_run(__nv_fp8_e4m3* mat_a, int ld_a,"),
        (kernel_relative, "void gemm_dispatch_old(void* mat_a, int ld_a,"),
        (kernel_relative, "void gemm_dispatch(void* mat_a, int ld_a,"),
        (kernel_relative, "static void run(ElementA* gmem_a, int ld_a, ElementB* gmem_b, int ld_b, ElementD* gmem_d,"),
        (kernel_relative, "cooperative_1x128_by_128x128_fp8_gemm_kernel("),
        (deep + "fp8_gemm.cuh", "void runGemm(cudaKernel_t kernel, void* mat_a, int ld_a,"),
        (deep + "fp8_gemm.cuh", "void runGemmSwapAB(cudaKernel_t kernel, void* mat_a, int ld_a,"),
        (deep + "fp8_gemm_impl.cuh", "fp8_gemm_kernel(__nv_bfloat16* gmem_d,"),
        (deep + "fp8_gemm_impl.cuh", "fp8_gemm_kernel_swapAB("),
        (deep + "scheduler.cuh", "struct NormalScheduler {"),
        (deep + "scheduler.cuh", "struct NormalSchedulerSwapAB {"))
    definitions = [_cpp_anchor(root, path, signature) for path, signature in signatures]
    programs = [re.sub(r"/\*.*?\*/|//[^\n]*", "", item["source"], flags=re.S) for item in definitions]
    compact = [re.sub(r"\s+", "", program) for program in programs]
    pointer_ports = (
        ["mat_a", "mat_b", "mat_d", "scales_a", "scales_b"],
        ["mat_a", "mat_b", "mat_d", "scales_a", "scales_b"],
        ["mat_a", "mat_b", "mat_d", "scales_a", "scales_b"],
        ["gmem_a", "gmem_b", "gmem_d", "scales_a", "scales_b"],
        ["gmem_d", "scales_b"],
        ["mat_a", "mat_b", "mat_d", "scales_a", "scales_b", "grouped_layout"],
        ["mat_a", "mat_b", "mat_d", "scales_a", "scales_b", "grouped_layout"],
        ["gmem_d", "scales_b"], ["gmem_d", "scales_a"], [], [])
    for program, expected_ports in zip(programs, pointer_ports, strict=True):
        header = program[:program.index("{")]
        if re.findall(r"\*\s*(?:const\s+)?(\w+)", header) != expected_ports:
            raise ValueError("FlashInfer dense matrix pointer ports changed; rederive storage")
    for source in programs:
        if re.search(r"\b(?:cudaMalloc\w*|malloc|calloc|realloc)\s*\(|\bnew\b", source):
            raise ValueError("FlashInfer dense callee now allocates source storage; rederive matrix temporaries")
    required = (
        (0, ("if(getDeepGemmEnabled())", "gemm_dispatch(", "gemm_dispatch_old(")),
        (1, ("GemmType::run(", "reinterpret_cast<__nv_bfloat16*>(mat_d)")),
        (2, ("if(shape_m>=m_threshold)", "m_threshold=32;", "num_problems=1;", "deep_gemm::runGemm(",
             "deep_gemm::runGemmSwapAB(", "static_cast<int*>(nullptr)", "deep_gemm::GemmType::Normal")),
        (3, ("usingProblemVisitor=PlainGemmProblemVisitor<TILE_M,TILE_N>;",
             "Kernel<<<grid_size,BLOCK_SIZE,smem_size,stream>>>(gmem_d,ld_d,scales_b,problem_input,",
             "make_2d_tma_a_desc(gmem_a,", "make_2d_tma_b_desc(gmem_b,", "make_2d_tma_scales_a_desc(scales_a,")),
        (4, ("extern__shared__", "gmem_d_this_block=", "&gmem_d_this_block[")),
        (5, ("cudaLaunchKernelEx(&config,kernel,reinterpret_cast<__nv_bfloat16*>(mat_d),scales_b,input,",
             "make_2d_tma_d_desc(reinterpret_cast<__nv_bfloat16*>(mat_d),",
             "input.grouped_layout=grouped_layout;", "config.dynamicSmemBytes=smem_size;")),
        (6, ("cudaLaunchKernelEx(&config,kernel,reinterpret_cast<__nv_bfloat16*>(mat_d),scales_a,input,",
             "make_2d_tma_d_desc_swapAB(reinterpret_cast<__nv_bfloat16*>(mat_d),",
             "input.grouped_layout=grouped_layout;", "config.dynamicSmemBytes=smem_size;")),
        (7, ("extern__shared__", "cute::SM90_TMA_STORE_2D::copy(&tensor_map_d,smem_d,n_block_idx*BLOCK_N,scheduler.get_global_m_idx(m_block_idx));")),
        (8, ("extern__shared__", "cute::SM90_TMA_STORE_2D::copy(&tensor_map_d,smem_d,m_block_idx*BLOCK_M,scheduler.get_global_n_idx(n_block_idx));")),
        (9, ("staticconstexprGemmTypegemm_type=GemmType::Normal;",)),
        (10, ("staticconstexprGemmTypegemm_type=GemmType::Normal;",)))
    for index, fragments in required:
        if any(fragment not in compact[index] for fragment in fragments):
            raise ValueError("FlashInfer SM90 dense matrix tensor/storage program changed")
    if compact[2].count("static_cast<int*>(nullptr)") != 2:
        raise ValueError("FlashInfer normal dense dispatch needs null grouped layout in both orientations")
    paths = sorted({item["path"] for item in definitions})
    return {"definitions": definitions,
            "source_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths},
            "scope": "SM90 normal dense source tensor ports/allocations only; H, external DeepGEMM provider, compiler spill, physical traffic and runtime remain unbound"}


def _source_anchors(root, torch_include, flashinfer_source_root):
    anchors = {}
    for role, relative, owner, member, callee in (
        ("fused-wqa-wkv", _ATTENTION, "DeepseekV4Attention", "fused_wqa_wkv", "MergedColumnParallelLinear"),
        ("wq-b", _ATTENTION, "DeepseekV4Attention", "wq_b", "ColumnParallelLinear"),
        ("wo-a", _ATTENTION, "DeepseekV4Attention", "wo_a", "ColumnParallelLinear"),
        ("wo-b", _ATTENTION, "DeepseekV4Attention", "wo_b", "RowParallelLinear"),
        ("indexer-wq-b", _ATTENTION, "DeepseekV4Indexer", "wq_b", "ReplicatedLinear"),
        ("indexer-weights-proj", _ATTENTION, "DeepseekV4Indexer", "weights_proj", "ReplicatedLinear"),
        ("compressor-kv-score", _COMPRESSOR, "DeepseekCompressor", "fused_wkv_wgate", "MergedColumnParallelLinear"),
        ("router", _MODEL, "DeepseekV4MoE", "gate", "GateLinear"),
        ("embedding-constructor", _MODEL, "DeepseekV4Model", "embed_tokens", "VocabParallelEmbedding"),
        ("final-norm-constructor", _MODEL, "DeepseekV4Model", "norm", "RMSNorm"),
        ("attention-sink", _ATTENTION, "DeepseekV4Attention", "attn_sink", "nn.Parameter"),
        ("shared-experts:gate-up", _MODEL, "DeepseekV4MLP", "gate_up_proj", "MergedColumnParallelLinear"),
        ("shared-experts:down-proj", _MODEL, "DeepseekV4MLP", "down_proj", "RowParallelLinear"),
        ("lm-head", _MODEL, "DeepseekV4ForCausalLM", "lm_head", "ParallelLMHead"),
    ):
        anchors[role] = _constructor(root, relative, owner, member, callee)
    for role in ("fused-wqa-wkv", "wq-b", "wo-b", "indexer-wq-b",
                 "shared-experts:gate-up", "shared-experts:down-proj"):
        _require_keywords(anchors[role], {"quant_config": "quant_config"})
    _require_keywords(anchors["fused-wqa-wkv"], {"disable_tp": "True"})
    _require_keywords(anchors["compressor-kv-score"], {"quant_config": "None", "disable_tp": "True"})
    _require_keywords(anchors["indexer-weights-proj"], {"quant_config": "None"})
    _require_keywords(anchors["router"], {"out_dtype": "torch.float32"})
    _require_keywords(anchors["embedding-constructor"], {"quant_config": "quant_config"})
    if anchors["final-norm-constructor"]["arguments"] != ["config.hidden_size", "self.rms_norm_eps"]:
        raise ValueError("V4 final norm no longer forwards the configured epsilon")
    model_init = re.sub(r"\s+", "", ast.unparse(_definition(ast.parse((root / _MODEL).read_text()), "DeepseekV4Model.__init__")))
    if "self.rms_norm_eps=config.rms_norm_eps" not in model_init:
        raise ValueError("V4 final norm epsilon config origin changed")
    norm_class_tree = ast.parse((root / "vllm/model_executor/layers/layernorm.py").read_text())
    if "self.variance_epsilon=eps" not in re.sub(r"\s+", "", ast.unparse(_definition(norm_class_tree, "RMSNorm.__init__"))):
        raise ValueError("RMSNorm no longer preserves the caller epsilon")
    if anchors["attention-sink"]["arguments"] != ["torch.full((self.padded_heads,), -float('inf'), dtype=torch.float32)"]:
        raise ValueError("V4 FP32 padded sink parameter source changed")
    if "quant_config" in anchors["lm-head"]["keywords"]:
        raise ValueError("V4 head quantizer omission changed; rederive its storage")
    vocab_path = root / _VOCAB
    vocab_tree = ast.parse(vocab_path.read_text(encoding="utf-8"))
    vocab_init = ast.unparse(_definition(vocab_tree, "VocabParallelEmbedding.__init__"))
    required_embedding = (
        "self.tp_size > 1 and current_platform.is_cuda() and isinstance(quant_method, UnquantizedEmbeddingMethod)",
        "quant_method = UnquantizedEmbeddingMethod()")
    if any(fragment not in vocab_init for fragment in required_embedding):
        raise ValueError("embedding CUDA unquantized source eligibility changed")
    quant_anchor = _function_anchor(root, "vllm/model_executor/layers/quantization/fp8.py", "Fp8Config.get_quant_method")
    anchors["embedding-quant-method"] = quant_anchor
    anchors["embedding-cuda"] = _cpp_anchor(root, "csrc/libtorch_stable/vocab_parallel_embedding_kernels.cu",
                                          "__global__ void vocab_parallel_embedding_kernel(")
    embedding_cpp = re.sub(r"\s+", "", anchors["embedding-cuda"]["source"])
    for fragment in ("static_cast<int64_t>(input_ids[token])", "id>=org_vocab_start_index&&id<org_vocab_end_index",
                     "id>=added_vocab_start_index&&id<added_vocab_end_index", "out_row[v]=weight_row[v]",
                     "out_row[v]=zero"):
        if fragment not in embedding_cpp:
            raise ValueError("embedding copy/range source changed; rederive its role account")
    embedding_wrapper_relative = "vllm/_custom_ops.py"
    embedding_wrapper = _definition(ast.parse((root / embedding_wrapper_relative).read_text()), "vocab_parallel_embedding")
    wrapper_body = re.sub(r"\s+", "", ast.unparse(embedding_wrapper))
    for fragment in ("out=torch.empty(input_ids.shape[0],weight.shape[1],dtype=weight.dtype,device=weight.device)",
                     "torch.ops._C.vocab_parallel_embedding(out,input_ids,weight,", "returnout"):
        if fragment not in wrapper_body:
            raise ValueError("embedding fresh output source program changed")
    wrapper_calls = [ast.unparse(node.func) for node in ast.walk(embedding_wrapper) if isinstance(node, ast.Call)]
    if (sorted(wrapper_calls) != ["torch.empty", "torch.ops._C.vocab_parallel_embedding"]
            or len(embedding_wrapper.body) != 4
            or not isinstance(embedding_wrapper.body[1], ast.Assign)
            or [ast.unparse(target) for target in embedding_wrapper.body[1].targets] != ["out"]):
        raise ValueError("embedding wrapper has additional allocation or tensor actions")
    fused_embedding_branches = [node for node in ast.walk(_definition(vocab_tree, "VocabParallelEmbedding.forward"))
                               if isinstance(node, ast.If) and ast.unparse(node.test) == "self.use_fused_embedding"]
    if (len(fused_embedding_branches) != 1 or sorted(ast.unparse(node.func)
            for statement in fused_embedding_branches[0].body for node in ast.walk(statement)
            if isinstance(node, ast.Call)) != ["input_.reshape", "ops.vocab_parallel_embedding", "output_parallel.view"]):
        raise ValueError("embedding fused branch has additional source tensor materializations")
    embedding_forward = re.sub(r"\s+", "", ast.unparse(_definition(vocab_tree, "VocabParallelEmbedding.forward")))
    for fragment in ("ifself.use_fused_embedding:",
                     "input_ifinput_.ndim==1elseinput_.reshape(-1)",
                     "returntensor_model_parallel_all_reduce(output_parallel)"):
        if fragment not in embedding_forward:
            raise ValueError("embedding source tensor forwarding changed")
    anchors["embedding-tensor-dataflow"] = {
        "definitions": [_function_anchor(root, embedding_wrapper_relative, "vocab_parallel_embedding"),
                        _function_anchor(root, _VOCAB, "VocabParallelEmbedding.forward")],
        "kernel": anchors["embedding-cuda"],
        "scope": "TP4 CUDA fused, one-dimensional contiguous INT32 IDs, contiguous BF16 weight; collective internals/compiled spills/runtime separate"}
    padding = [node.value for node in vocab_tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "DEFAULT_VOCAB_PADDING_SIZE"
                       for target in node.targets)]
    if len(padding) != 1 or ast.literal_eval(padding[0]) != 64:
        raise ValueError("embedding default vocabulary padding changed")
    for name in ("ParallelLMHead", "VocabParallelEmbedding"):
        init = _definition(vocab_tree, name + ".__init__")
        defaults = dict(zip((arg.arg for arg in init.args.args[-len(init.args.defaults):]),
                            (ast.unparse(default) for default in init.args.defaults)))
        if defaults["quant_config"] != "None" or defaults["params_dtype"] != "None":
            raise ValueError("unquantized vocabulary constructor defaults changed")
        if name == "VocabParallelEmbedding" and (defaults["org_num_embeddings"] != "None"
                                                or defaults["padding_size"] != "DEFAULT_VOCAB_PADDING_SIZE"):
            raise ValueError("embedding no-added-vocabulary/default-padding source condition changed")
        anchors[name] = {"path": str(vocab_path), "line": init.lineno,
                         "end_line": init.end_lineno, "symbol": name + ".__init__",
                         "expression": ast.unparse(init)}
    for name, relative, symbol in (
        ("decoder-forward", _MODEL, "DeepseekV4DecoderLayer.forward"),
        ("model-forward", _MODEL, "DeepseekV4Model.forward"),
        ("attention-forward", _ATTENTION, "DeepseekV4Attention.forward"),
        ("compressor-forward", _COMPRESSOR, "DeepseekCompressor.forward"),
        ("indexer-forward", _ATTENTION, "DeepseekV4Indexer.forward"),
        ("ffn-forward", _MODEL, "DeepseekV4MoE.forward"),
        ("shared-forward", _MODEL, "DeepseekV4MLP.forward"),
        ("embedding-forward", _VOCAB, "VocabParallelEmbedding.forward"),
        ("tp-all-reduce", "vllm/distributed/communication_op.py", "tensor_model_parallel_all_reduce"),
        ("tp-all-gather", "vllm/distributed/communication_op.py", "tensor_model_parallel_all_gather"),
        ("tp-group-reduce", "vllm/distributed/parallel_state.py", "GroupCoordinator.all_reduce"),
        ("tp-group-gather", "vllm/distributed/parallel_state.py", "GroupCoordinator.all_gather"),
        ("tp-gather-layout-cuda", "vllm/distributed/device_communicators/cuda_communicator.py", "CudaCommunicator.all_gather"),
        ("tp-gather-layout-base", "vllm/distributed/device_communicators/base_device_communicator.py", "DeviceCommunicatorBase.all_gather"),
        ("logits-gather", "vllm/model_executor/layers/logits_processor.py", "LogitsProcessor._gather_logits"),
        ("sampler-forward", "vllm/v1/sample/sampler.py", "Sampler.forward"),
        ("greedy", "vllm/v1/sample/sampler.py", "Sampler.greedy_sample"),
        ("logits", "vllm/model_executor/layers/logits_processor.py", "LogitsProcessor._get_logits"),
        ("row-selection", "vllm/v1/worker/gpu_model_runner.py", "GPUModelRunner.execute_model"),
        ("padded-heads", "vllm/models/deepseek_v4/nvidia/flashmla.py", "DeepseekV4FlashMLAAttention.get_padded_num_q_heads"),
        ("indexer-q-quant", "vllm/models/deepseek_v4/common/ops/fused_indexer_q.py", "fused_indexer_q_rope_quant"),
        ("inverse-rope-quant", "vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py", "FusedInvRopeFP8QuantKernel.kernel"),
        ("indexer-causal-ends", "vllm/v1/attention/backends/mla/indexer.py", "BuildPrefillChunkMetadataKernel.kernel"),
        ("V4-rotary-cache", "vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py", "DeepseekV4ScalingRotaryEmbedding.__init__"),
        ("save-partial", "vllm/models/deepseek_v4/common/ops/save_partial_states.py", "SavePartialStatesKernel.kernel"),
        ("clamped-activation", "vllm/model_executor/layers/activation.py", "SiluAndMulWithClamp.forward_native"),
        ("qkv-norm", "vllm/models/common/ops/fused_qk_rmsnorm.py", "FusedQKVRMSNormKernel.kernel"),
        ("C128-finish-stages", "vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py",
         "split_kv_compress_norm_rope_insert_sparse_attn_cutedsl"),
        ("C128-finish-compress", "vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py",
         "SparseAttnCompressC128Block8Kernel.kernel"),
        ("C128-finish-temporary", "vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py",
         "SparseAttnCompressC128Block8Kernel.__call__"),
        ("C4-finish", "vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py",
         "SparseAttnCompressNormRopeStoreC4Kernel.kernel"),
        ("C128-finish-store", "vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py",
         "SparseAttnNormRopeStoreKernel.kernel"),
        ("indexer-finish", "vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py",
         "_fused_kv_compress_norm_rope_insert_indexer_attn"),
        ("prefill-gather", "vllm/models/deepseek_v4/common/ops/cache_utils.py", "dequantize_and_gather_k_cache"),
        ("fp8-linear-input-quant", "vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py",
         "Fp8BlockScaledMMLinearKernel.apply_weights"),
        ("final-rmsnorm", "vllm/ir/ops/layernorm.py", "rms_norm"),
        ("swa-cache-layout", "vllm/v1/attention/backends/mla/sparse_swa.py", "DeepseekV4SWACache.__init__"),
        ("prefill-combine-indices", "vllm/models/deepseek_v4/common/ops/cache_utils.py", "combine_topk_swa_indices"),
    ):
        anchors[name] = _function_anchor(root, relative, symbol)
    anchors["argmax-pair"] = None if torch_include is None else _argmax_anchor(torch_include)
    head_definitions = []
    for relative, symbol, fragments in (
        ("vllm/model_executor/layers/logits_processor.py", "LogitsProcessor._apply_head", (
            "ifself.head_dtypeisNoneorself.head_dtype==hidden_states.dtype:",
            "returnlm_head.quant_method.apply(lm_head,hidden_states,bias=embedding_bias)")),
        (_VOCAB, "UnquantizedEmbeddingMethod.apply", (
            "returnlinear_batch_invariant(x,layer.weight,bias)", "returndispatch_unquantized_gemm()(layer,x,layer.weight,bias)")),
        ("vllm/model_executor/layers/utils.py", "default_unquantized_gemm", (
            "returntorch.nn.functional.linear(x,weight,bias)",)),
        ("vllm/model_executor/determinism/batch_invariant.py", "linear_batch_invariant", (
            "output=matmul_batch_invariant(input,weight.t())", "ifbiasisnotNone:", "returnoutput")),
        ("vllm/model_executor/determinism/batch_invariant.py", "matmul_batch_invariant", (
            "ifa.ndim==2andb.ndim==2:", "result=mm_batch_invariant(a,b)", "ifoutisnotNone:", "returnresult")),
        ("vllm/model_executor/determinism/batch_invariant.py", "mm_batch_invariant", (
            "ifa.device.type=='xpu':", "returnmatmul_persistent(a,b,bias=bias)")),
        ("vllm/model_executor/determinism/batch_invariant.py", "matmul_persistent", (
            "(M,K)=a.shape", "(K,N)=b.shape", "dtype=a.dtype", "c=torch.empty((M,N),device=a.device,dtype=dtype)", "returnc")),
        ("vllm/config/model.py", "_get_head_dtype", (
            "elifhead_dtypeisNone:", "ifrunner_type=='pooling':", "returntorch.float32", "returndtype")),
    ):
        definition = _definition(ast.parse((root / relative).read_text()), symbol)
        program = re.sub(r"\s+", "", ast.unparse(definition))
        if any(fragment not in program for fragment in fragments):
            raise ValueError("unquantized vocabulary head caller changed: " + symbol)
        head_definitions.append(_function_anchor(root, relative, symbol))
    dispatch_relative = "vllm/model_executor/layers/utils.py"
    dispatch_tree = ast.parse((root / dispatch_relative).read_text())
    dispatch = _definition(dispatch_tree, "dispatch_unquantized_gemm")
    if ast.literal_eval(dispatch.args.defaults[0]) != "auto":
        raise ValueError("unquantized embedding no-argument dispatch default changed")
    backend_tables = [node.value for node in dispatch_tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "_FLASHINFER_BF16_BACKENDS" for target in node.targets)]
    if len(backend_tables) != 1 or not isinstance(backend_tables[0], ast.Dict) or "auto" in [ast.literal_eval(key) for key in backend_tables[0].keys]:
        raise ValueError("unquantized embedding auto dispatch now selects an explicit BF16 backend")
    head_definitions.append(_function_anchor(root, dispatch_relative, "dispatch_unquantized_gemm"))
    head_schemas = {}
    for name, schema in (("linear", "linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor"),
                         ("t", "t(Tensor(a) self) -> Tensor(a)")):
        if torch_include is None:
            head_schemas[name] = None
            continue
        header = Path(torch_include) / f"ATen/ops/{name}_ops.h"
        text = header.read_text()
        if schema not in text:
            raise ValueError("Torch vocabulary head functional/view schema changed: " + name)
        head_schemas[name] = {"path": str(header), "line": text[:text.index(schema)].count("\n") + 1,
                              "schema": schema, "version_binding": "argmax-pair.torch_version"}
    anchors["bf16-lm-head-objects"] = {"definitions": head_definitions,
        "aten_linear_schema": head_schemas["linear"], "aten_weight_t_schema": head_schemas["t"],
        "scope": "logical caller tensor/parameter objects for both source CUDA branches; native scratch, output strides, numerical equivalence and provider unbound"}
    for name in ("tp-gather-layout-cuda", "tp-gather-layout-base"):
        relative, symbol = (("vllm/distributed/device_communicators/cuda_communicator.py", "CudaCommunicator.all_gather")
                            if name.endswith("cuda") else
                            ("vllm/distributed/device_communicators/base_device_communicator.py", "DeviceCommunicatorBase.all_gather"))
        definition = _definition(ast.parse((root / relative).read_text()), symbol)
        program = re.sub(r"\s+", "", ast.unparse(definition))
        for expected in ("output_tensor.reshape((self.world_size,)+input_size)",
                         "output_tensor.movedim(0,dim)",
                         "input_size[:dim]+(self.world_size*input_size[dim],)+input_size[dim+1:]"):
            if expected not in program:
                raise ValueError("last-dimension TP gather materialization source changed")
    compressor_weight = anchors["compressor-kv-score"]["keywords"]
    if (any(compressor_weight[name] != value for name, value in
            (("quant_config", "None"), ("bias", "False"), ("disable_tp", "True")))
            or "params_dtype" in compressor_weight):
        raise ValueError("compressor unquantized model-dtype replicated weight source changed")
    parallel_projections = _definition(ast.parse((root / _ATTENTION).read_text()), "DeepseekV4Attention._run_parallel_input_projections")
    mm_calls = [node for node in ast.walk(parallel_projections) if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.mm"]
    if (len(mm_calls) != 2 or any(ast.unparse(node.args[0]) != "hidden_states"
            or [(keyword.arg, ast.unparse(keyword.value)) for keyword in node.keywords] != [("out_dtype", "torch.float32")]
            for node in mm_calls)
            or sorted(ast.unparse(node.args[1]) for node in mm_calls) != [
                "compressor.fused_wkv_wgate.weight.T", "indexer.compressor.fused_wkv_wgate.weight.T"]):
        raise ValueError("compressor BF16-input / FP32-output projection source changed")
    anchors["compressor-projection-dataflow"] = {
        "projections": _function_anchor(root, _ATTENTION, "DeepseekV4Attention._run_parallel_input_projections"),
        "weight_constructor": anchors["compressor-kv-score"],
        "decoder_input": _function_anchor(root, _MODEL, "DeepseekV4DecoderLayer.forward"),
        "scope": "source shared layer-input, BF16 parameter transpose views and explicit FP32 mm outputs; no native/provider binding"}
    gate_constructor = anchors["router"]["keywords"]
    if (gate_constructor["bias"] != "False" or gate_constructor["out_dtype"] != "torch.float32"
            or "params_dtype" in gate_constructor or "force_fp32_compute" in gate_constructor):
        raise ValueError("Flash gate model-dtype/FP32-output constructor changed")
    gate_tree = ast.parse((root / _GATE_LINEAR).read_text())
    gate_init = _definition(gate_tree, "GateLinear.__init__")
    defaults = dict(zip((arg.arg for arg in gate_init.args.args[-len(gate_init.args.defaults):]),
                        (ast.unparse(value) for value in gate_init.args.defaults)))
    if defaults["params_dtype"] != "None" or defaults["force_fp32_compute"] != "False":
        raise ValueError("Flash gate parameter dtype defaults changed")
    assignments = {ast.unparse(target): node.value for node in ast.walk(gate_init) if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Attribute)}
    for target in ("self.allow_fp32_router_gemm", "self.allow_bf16x3_router_gemm"):
        expression = assignments[target]
        if (not isinstance(expression, ast.BoolOp) or not isinstance(expression.op, ast.And)
                or not any(ast.unparse(value) == "self.weight.dtype == torch.float32" for value in expression.values)):
            raise ValueError("FP32-weight-only gate programs no longer excluded by BF16 source weight")
    gate_forward = _definition(gate_tree, "GateLinear.forward")
    guards = [node for node in gate_forward.body if isinstance(node, ast.If)]
    if (ast.unparse(guards[0].test) != "self.allow_ll_bf16_gemm and x.shape[0] <= 16 and (x.dtype == torch.bfloat16)"
            or ast.unparse(guards[3].test) != "self.allow_cublas_router_gemm and x.dtype == torch.bfloat16"):
        raise ValueError("BF16 gate conditional program eligibility changed")
    if ([ast.unparse(node) for node in guards[0].body if not isinstance(node, ast.ImportFrom)] != [
            "output = ll_bf16_gemm(x, self.weight)", "return (output, None)"]
            or [ast.unparse(node) for node in guards[3].body] != [
                "output = torch.mm(x, self.weight.T, out_dtype=torch.float32)", "return (output, None)"]):
        raise ValueError("BF16 gate caller input/output transformations changed")
    gate_mm = [node for node in ast.walk(guards[3]) if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.mm"]
    if (len(gate_mm) != 1 or ast.unparse(gate_mm[0]) != "torch.mm(x, self.weight.T, out_dtype=torch.float32)"
            or ast.unparse(assignments["self.allow_cublas_router_gemm"]) != (
                "self._router_gemm_cublas_capable and self.weight.dtype == torch.bfloat16 and (self.out_dtype == torch.float32)")):
        raise ValueError("BF16 gate cuBLAS source dtype/transpose changed")
    if (ast.unparse(assignments["self._router_gemm_no_bias"]) != "not bias"
            or ast.unparse(assignments["self._router_gemm_cublas_capable"]) != (
                "(current_platform.is_cuda() or current_platform.is_rocm()) and self._router_gemm_no_bias")):
        raise ValueError("source CUDA BF16 gate no longer guarantees the mixed-output mm branch")
    ll_tree = ast.parse((root / _LL_BF16).read_text())
    ll_call = _definition(ll_tree, "LLBf16Gemm.__call__")
    for symbol in ("ll_bf16_gemm", "LLBf16Gemm.__call__"):
        function = _definition(ll_tree, symbol)
        defaults = dict(zip((arg.arg for arg in function.args.args[-len(function.args.defaults):]),
                            (ast.unparse(value) for value in function.args.defaults)))
        if defaults["output_dtype"] != "torch.float32":
            raise ValueError("ll_bf16 gate output dtype default changed")
    ll_outputs = [node for node in ast.walk(ll_call) if isinstance(node, ast.Assign)
                  and any(ast.unparse(target) == "output" for target in node.targets)]
    if len(ll_outputs) != 1 or ast.unparse(ll_outputs[0]) != "output = torch.empty(M, N, dtype=output_dtype, device=hidden_states.device)":
        raise ValueError("ll_bf16 gate fresh output source changed")
    if [ast.unparse(node) for node in ast.walk(ll_call) if isinstance(node, ast.Return)] != ["return output"]:
        raise ValueError("ll_bf16 gate returns a different output object")
    anchors["gate-projection-dataflow"] = {
        "constructor": anchors["router"], "dispatch": _function_anchor(root, _GATE_LINEAR, "GateLinear.forward"),
        "eligibility": _function_anchor(root, _GATE_LINEAR, "GateLinear.__init__"),
        "ll_output": _function_anchor(root, _LL_BF16, "LLBf16Gemm.__call__"),
        "ll_forward": _function_anchor(root, _LL_BF16, "ll_bf16_gemm"),
        "scope": "common source tensor contract of eligible BF16 gate programs; selected provider, compiler and library scratch remain separate"}
    index_weight = anchors["indexer-weights-proj"]["keywords"]
    if index_weight["quant_config"] != "None" or index_weight["bias"] != "False" or "params_dtype" in index_weight:
        raise ValueError("indexer weights projection unquantized BF16 constructor changed")
    utility_relative = "vllm/model_executor/layers/utils.py"
    utility_tree = ast.parse((root / utility_relative).read_text())
    default_linear = _definition(utility_tree, "default_unquantized_gemm")
    if [ast.unparse(node) for node in default_linear.body] != ["return torch.nn.functional.linear(x, weight, bias)"]:
        raise ValueError("default unquantized linear tensor program changed")
    unquantized_apply = _definition(ast.parse((root / "vllm/model_executor/layers/linear.py").read_text()), "UnquantizedLinearMethod.apply")
    if [ast.unparse(node) for node in unquantized_apply.body] != [
            "if envs.VLLM_BATCH_INVARIANT and (current_platform.is_cuda_alike() or current_platform.is_xpu()):\n    return linear_batch_invariant(x, layer.weight, bias)",
            "return self._gemm_impl(layer, x, layer.weight, bias)"]:
        raise ValueError("unquantized BF16 linear branch program changed")
    backends = [node.value for node in utility_tree.body if isinstance(node, ast.Assign)
                and any(ast.unparse(target) == "_FLASHINFER_BF16_BACKENDS" for target in node.targets)]
    if len(backends) != 1 or not isinstance(backends[0], ast.Dict) or [ast.literal_eval(key) for key in backends[0].keys] != ["flashinfer_cutedsl"]:
        raise ValueError("additional unquantized BF16 CUDA backend needs an independent tensor account")
    implement = _definition(utility_tree, "_can_use_flashinfer_cutedsl_bf16")
    if ast.unparse(implement.body[0]) != ("if not (current_platform.is_cuda() and current_platform.is_device_capability_family(100)):\n    return False"):
        raise ValueError("SM100-only FlashInfer BF16 program exclusion no longer holds for SM90")
    backend_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in backends[0].values[0].keywords}
    if backend_keywords["can_implement"] != "_can_use_flashinfer_cutedsl_bf16":
        raise ValueError("FlashInfer BF16 backend architecture guard binding changed")
    flashinfer_impl = _definition(utility_tree, "cuda_flashinfer_bf16_gemm_impl")
    if ast.unparse(flashinfer_impl.body[1]) != ("if not backend_spec.can_implement(x, weight, bias):\n    return torch.nn.functional.linear(x, weight, bias)"):
        raise ValueError("ineligible FlashInfer BF16 path no longer uses the default linear source program")
    invariant_relative = "vllm/model_executor/determinism/batch_invariant.py"
    invariant_tree = ast.parse((root / invariant_relative).read_text())
    invariant_linear = _definition(invariant_tree, "linear_batch_invariant")
    if [ast.unparse(node) for node in invariant_linear.body] != [
            "output = matmul_batch_invariant(input, weight.t())", "if bias is not None:\n    output = output + bias", "return output"]:
        raise ValueError("bias-free batch-invariant linear tensor program changed")
    persistent = _definition(invariant_tree, "matmul_persistent")
    output_allocations = [node for node in persistent.body if isinstance(node, ast.Assign)
                          and any(ast.unparse(target) == "c" for target in node.targets)]
    if len(output_allocations) != 1 or ast.unparse(output_allocations[0]) != "c = torch.empty((M, N), device=a.device, dtype=dtype)":
        raise ValueError("batch-invariant BF16 matrix output allocation changed")
    if "dtype = a.dtype" not in [ast.unparse(node) for node in persistent.body]:
        raise ValueError("batch-invariant matrix output dtype changed")
    if [ast.unparse(node) for node in persistent.body if isinstance(node, ast.Return)] != ["return c"]:
        raise ValueError("batch-invariant matrix no longer returns its fresh output")
    invariant_mm = _definition(invariant_tree, "mm_batch_invariant")
    if ast.unparse(invariant_mm.body[-1]) != "return matmul_persistent(a, b, bias=bias)":
        raise ValueError("CUDA batch-invariant mm callee changed")
    invariant_matmul = _definition(invariant_tree, "matmul_batch_invariant")
    matrix_path = invariant_matmul.body[0]
    if (not isinstance(matrix_path, ast.If)
            or ast.dump(matrix_path.test) != ast.dump(ast.parse("a.ndim == 2 and b.ndim == 2", mode="eval").body)
            or [ast.unparse(node) for node in matrix_path.body] != ["result = mm_batch_invariant(a, b)",
                "if out is not None:\n    out.copy_(result)\n    return out", "return result"]
            or {argument.arg: ast.unparse(default) for argument, default in
                zip(invariant_matmul.args.kwonlyargs, invariant_matmul.args.kw_defaults, strict=True)} != {"out": "None"}):
        raise ValueError("two-dimensional batch-invariant matrix path changed")
    indexer_forward = _definition(ast.parse((root / _ATTENTION).read_text()), "DeepseekV4Indexer.forward")
    short = [node for node in ast.walk(indexer_forward) if isinstance(node, ast.If)
             and "indexer_metadata.max_seq_len // self.compress_ratio <= self.topk_tokens" in ast.unparse(node.test)]
    if len(short) != 1 or ast.unparse(short[0].body[-1]) != "return (None, None, None)":
        raise ValueError("short-context indexer no longer drops its precomputed weights output")
    if any(isinstance(node, ast.Name) and node.id == "indexer_weights"
           for statement in short[0].body for node in ast.walk(statement)):
        raise ValueError("short-context indexer now reads its weights output")
    index_closure = [node for node in ast.walk(parallel_projections) if isinstance(node, ast.FunctionDef) and node.name == "indexer_weights_proj"]
    if len(index_closure) != 1 or [ast.unparse(node) for node in index_closure[0].body] != [
            "(weights, _) = indexer.weights_proj(hidden_states)", "return weights"]:
        raise ValueError("indexer BF16 weights projection caller transformations changed")
    anchors["indexer-weights-dataflow"] = {
        "projection": anchors["indexer-weights-proj"],
        "input_projection_caller": _function_anchor(root, _ATTENTION, "DeepseekV4Attention._run_parallel_input_projections"),
        "indexer_consumer": _function_anchor(root, _ATTENTION, "DeepseekV4Indexer.forward"),
        "default_linear": _function_anchor(root, utility_relative, "default_unquantized_gemm"),
        "cuda_dispatch": _function_anchor(root, utility_relative, "dispatch_unquantized_gemm"),
        "flashinfer_exclusion": _function_anchor(root, utility_relative, "_can_use_flashinfer_cutedsl_bf16"),
        "invariant_linear": _function_anchor(root, invariant_relative, "linear_batch_invariant"),
        "invariant_matmul": _function_anchor(root, invariant_relative, "matmul_persistent"),
        "scope": "common BF16 projection tensor contract on declared CUDA SM90; actual batch-invariant flag, provider and physical allocator separate"}
    activation_relative = "vllm/model_executor/layers/activation.py"
    activation_tree = ast.parse((root / activation_relative).read_text())
    activation_cuda = _definition(activation_tree, "SiluAndMulWithClamp.forward_cuda")
    if [ast.unparse(node) for node in activation_cuda.body] != [
            "d = x.shape[-1] // 2", "output_shape = x.shape[:-1] + (d,)",
            "out = torch.empty(output_shape, dtype=x.dtype, device=x.device)",
            "self.op(out, x, self.swiglu_limit, self.alpha, self.beta)", "return out"]:
        raise ValueError("clamped shared activation CUDA allocation/source operands changed")
    activation_init = _definition(activation_tree, "SiluAndMulWithClamp.__init__")
    defaults = dict(zip((arg.arg for arg in activation_init.args.args[-len(activation_init.args.defaults):]),
                        (ast.unparse(value) for value in activation_init.args.defaults)))
    if defaults != {"alpha": "1.0", "beta": "0.0"}:
        raise ValueError("shared activation source alpha/beta defaults changed")
    if not any(isinstance(node, ast.Assign) and ast.unparse(node) == "self.op = torch.ops._C.silu_and_mul_with_clamp"
               for node in ast.walk(activation_init)):
        raise ValueError("shared activation CUDA custom-op source binding changed")
    activation_launcher = _cpp_anchor(root, "csrc/libtorch_stable/activation_kernels.cu", "void silu_and_mul_clamp(")
    launcher = re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", activation_launcher["source"]))
    if launcher.split("{", 1)[1] != ("LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel,vllm::packed_silu_kernel,"
                                     "true,true,(float)limit,(float)alpha,(float)beta);}"):
        raise ValueError("shared activation CUDA launch clamp/source operands changed")
    activation_kernel = _cpp_anchor(root, "csrc/libtorch_stable/activation_kernels.cu", "__global__ void act_and_mul_kernel(")
    compact = re.sub(r"\s+", "", activation_kernel["source"])
    for fragment in ("constscalar_t*__restrict__input", "input+token_idx*2*d", "y_ptr=x_ptr+d", "out+token_idx*d"):
        if fragment not in compact:
            raise ValueError("shared activation input/output pointer geometry changed")
    anchors["shared-activation-cuda-dataflow"] = {
        "wrapper": _function_anchor(root, activation_relative, "SiluAndMulWithClamp.forward_cuda"),
        "caller": anchors["shared-forward"], "kernel": activation_kernel, "launcher": activation_launcher,
        "scope": "explicit CUDA custom-op source allocation and pointers; no forward_native temporary equivalence or selected runtime provider"}
    wrapper_relative = "vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py"
    wrapper = _definition(ast.parse((root / wrapper_relative).read_text()), "Fp8BlockScaledMMLinearKernel.apply_weights")
    wrapper_source = ast.unparse(wrapper)
    if ("input_2d = x.view(-1, x.shape[-1])" not in wrapper_source
            or "return output.to(dtype=out_dtype).view(*output_shape)" not in wrapper_source):
        raise ValueError("FP8 caller input/return view source changed")
    output_relative = "vllm/model_executor/kernels/linear/scaled_mm/deep_gemm.py"
    output_body = _definition(ast.parse((root / output_relative).read_text()), "DeepGemmFp8BlockScaledMMKernel.apply_block_scaled_mm")
    if "output = torch.empty((A.shape[0], B.shape[0]), dtype=out_dtype, device=A.device)" not in ast.unparse(output_body):
        raise ValueError("DeepGEMM source caller output allocation changed")
    disabled_quant = [node for node in ast.walk(wrapper) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "self.apply_input_quant"]
    if len(disabled_quant) != 1 or [ast.unparse(node) for node in disabled_quant[0].orelse] != [
            "q_input = input_2d", "input_scale = input_scale if input_scale is not None else input_2d.new_empty(1)"]:
        raise ValueError("FP8 wrapper unused scale placeholder source changed")
    flashinfer_relative = "vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py"
    flashinfer_tree = ast.parse((root / flashinfer_relative).read_text())
    flashinfer_class = _definition(flashinfer_tree, "FlashInferFp8BlockScaledMMKernel")
    quant_flags = [node for node in flashinfer_class.body if isinstance(node, ast.AnnAssign)
                   and ast.unparse(node.target) == "apply_input_quant"]
    if len(quant_flags) != 1 or ast.literal_eval(quant_flags[0].value) is not False:
        raise ValueError("FlashInfer wrapper BF16-input source flag changed")
    callee = _definition(flashinfer_tree, "FlashInferFp8BlockScaledMMKernel.apply_block_scaled_mm")
    returns = [node.value for node in callee.body if isinstance(node, ast.Return)]
    if (len(returns) != 1 or not isinstance(returns[0], ast.Call)
            or ast.unparse(returns[0].func) != "torch.ops.vllm.flashinfer_fp8_blockscale_gemm"
            or [ast.unparse(arg) for arg in returns[0].args] != ["A", "B", "Bs"]):
        raise ValueError("FlashInfer source must not consume the unused As placeholder")
    anchors["fp8-linear-input-quant"]["wrapper_storage"] = {
        "source": _function_anchor(root, wrapper_relative, "Fp8BlockScaledMMLinearKernel.apply_weights"),
        "callee": _function_anchor(root, flashinfer_relative, "FlashInferFp8BlockScaledMMKernel.apply_block_scaled_mm"),
        "scope": "conditional source objects, not selected provider or allocation capacity"}
    split_definition = _definition(ast.parse((root / _ATTENTION).read_text()), "DeepseekV4Attention._split_qkv_and_norm")
    split_calls = [node for node in ast.walk(split_definition) if isinstance(node, ast.Call)
                   and ast.unparse(node.func) == "qr_kv.split"]
    if (len(split_calls) != 1 or [ast.unparse(arg) for arg in split_calls[0].args] != ["[self.q_lora_rank, self.head_dim]"]
            or [(keyword.arg, ast.unparse(keyword.value)) for keyword in split_calls[0].keywords] != [("dim", "-1")]):
        raise ValueError("V4 Q/KV split view source changed")
    if [ast.unparse(node) for node in split_definition.body if isinstance(node, ast.Return)] != ["return (qr, None, kv)"]:
        raise ValueError("NVIDIA Q/KV norm must forward separate normalized tensors without a prequantized Q scale")
    norm_relative = "vllm/models/common/ops/fused_qk_rmsnorm.py"
    norm_wrapper = _definition(ast.parse((root / norm_relative).read_text()), "fused_q_kv_rmsnorm")
    empty_calls = [node for node in ast.walk(norm_wrapper) if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.empty"]
    if [ast.unparse(node) for node in empty_calls] != [
            "torch.empty(qr.shape, dtype=qr.dtype, device=qr.device)",
            "torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)"]:
        raise ValueError("Q/KV norm must allocate separate dense output tensors")
    if [ast.unparse(node) for node in norm_wrapper.body if isinstance(node, ast.Return)] != ["return (qr_out, kv_out)"]:
        raise ValueError("Q/KV norm output identities changed")
    anchors["qkv-norm-dataflow"] = {
        "split": _function_anchor(root, _ATTENTION, "DeepseekV4Attention._split_qkv_and_norm"),
        "wrapper": _function_anchor(root, norm_relative, "fused_q_kv_rmsnorm"),
        "consumers": _function_anchor(root, _ATTENTION, "DeepseekV4Attention._prepare_and_attn"),
        "scope": "source split views, separate normalized outputs and typed source consumers; not actual allocator/runtime"}
    sampler_tree = ast.parse((root / "vllm/v1/sample/sampler.py").read_text())
    sampler_forward = _definition(sampler_tree, "Sampler.forward")
    sampler_assignments = [ast.unparse(node) for node in ast.walk(sampler_forward) if isinstance(node, ast.Assign)]
    for statement in ("logits = logits.to(torch.float32)", "sampled = sampled.long()", "sampled = sampled.to(torch.int32)"):
        if sampler_assignments.count(statement) != 1:
            raise ValueError("plain greedy sampler cast/alias source changed")
    greedy = _definition(sampler_tree, "Sampler.greedy_sample")
    if [ast.unparse(node) for node in greedy.body if isinstance(node, ast.Return)] != ["return logits.argmax(dim=-1).view(-1)"]:
        raise ValueError("greedy argmax result/view source changed")
    outputs = [node for node in ast.walk(sampler_forward) if isinstance(node, ast.keyword)
               and node.arg == "sampled_token_ids" and ast.unparse(node.value) == "sampled.unsqueeze(-1)"]
    if len(outputs) != 1:
        raise ValueError("sampler output INT32 unsqueeze view source changed")
    runner_relative = "vllm/model_executor/layers/fused_moe/runner/moe_runner.py"
    runner_forward = _definition(ast.parse((root / runner_relative).read_text()), "MoERunner.forward")
    shared_add_branches = [node for node in ast.walk(runner_forward) if isinstance(node, ast.If)
                           and ast.unparse(node.test) == "shared_output is not None"]
    if len(shared_add_branches) != 1 or len(shared_add_branches[0].body) != 1:
        raise ValueError("modular MoE shared-add branch changed")
    shared_add_statement = shared_add_branches[0].body[0]
    if ast.unparse(shared_add_statement) != "result = shared_output + fused_output":
        raise ValueError("modular MoE shared-add must allocate an out-of-place result")
    modular_forward = _definition(ast.parse((root / _MODEL).read_text()), "DeepseekV4MoE._forward_fused_moe")
    if ("final_hidden_states = self.experts(" not in ast.unparse(modular_forward)
            or "return final_hidden_states.view(org_shape)" not in ast.unparse(modular_forward)):
        raise ValueError("V4 modular MoE result forwarding changed")
    anchors["modular-shared-add"] = {
        "runner": _function_anchor(root, runner_relative, "MoERunner.forward"),
        "caller": _function_anchor(root, _MODEL, "DeepseekV4MoE._forward_fused_moe"),
        "line": shared_add_statement.lineno,
        "source": ast.unparse(shared_add_statement),
        "scope": "modular branch only; mega-MoE uses a different addition program"}
    compressor_definition = _definition(ast.parse((root / _COMPRESSOR).read_text()), "DeepseekCompressor.forward")
    splits = [node for node in ast.walk(compressor_definition) if isinstance(node, ast.Call)
              and ast.unparse(node.func) == "kv_score.split"]
    if (len(splits) != 1 or ast.unparse(splits[0].args[0]) != "[self.coff * self.head_dim, self.coff * self.head_dim]"
            or [(keyword.arg, ast.unparse(keyword.value)) for keyword in splits[0].keywords] != [("dim", "-1")]):
        raise ValueError("compressor FP32 projection split/view interface changed")
    save_relative = "vllm/models/deepseek_v4/common/ops/save_partial_states.py"
    save_definition = _definition(ast.parse((root / save_relative).read_text()), "SavePartialStatesKernel.kernel")
    compact_save = re.sub(r"\s+", "", ast.unparse(save_definition))
    for fragment in ("ifslot_id<0:return", "block_idx=slot_id//block_size", "pos_in_block=slot_id%block_size",
                     "ape_row=position%COMPRESS_RATIO", "base_ptr+STATE_WIDTH+block", "score+ape"):
        if fragment not in compact_save:
            raise ValueError("save-partial view/slot/score-state source changed")
    invalid_guards = [node for node in ast.walk(save_definition) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "slot_id < 0"]
    if len(invalid_guards) != 1:
        raise ValueError("save-partial invalid-slot early return changed")
    loads_before_guard = [node for node in ast.walk(save_definition) if isinstance(node, ast.Call)
                          and ast.unparse(node.func) == "tl.load" and node.lineno < invalid_guards[0].lineno]
    if len(loads_before_guard) != 1 or ast.unparse(loads_before_guard[0].args[0]) != "slot_mapping_ptr + token_idx":
        raise ValueError("save-partial now accesses inputs before the invalid-slot guard")
    producers = [node for node in ast.walk(compressor_definition) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == "_SAVE_PARTIAL_STATES_KERNEL"]
    consumers = [node for node in ast.walk(compressor_definition) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == "compress_norm_rope_store_fn"]
    if len(producers) != 1 or len(consumers) != 1 or producers[0].lineno >= consumers[0].lineno:
        raise ValueError("save-partial no longer precedes compression/finish")
    producer_arguments = {keyword.arg: ast.unparse(keyword.value) for keyword in producers[0].keywords}
    for name, expected in (("kv", "kv"), ("score", "score"), ("ape", "self.ape"),
                           ("positions", "positions"), ("slot_mapping", "slot_mapping"), ("state_width", "state_width")):
        if producer_arguments[name] != expected:
            raise ValueError("save-partial tensor object argument changed: " + name)
    before_save_calls = sorted(ast.unparse(node.func) for node in ast.walk(compressor_definition)
                               if isinstance(node, ast.Call) and node.lineno < producers[0].lineno)
    if before_save_calls != sorted(("kv_score.split", "get_forward_context", "isinstance", "cast",
                                   "current_platform.is_rocm", "current_platform.is_xpu")):
        raise ValueError("save-partial caller has new operations requiring a materialization review")
    for name in ("kv", "score"):
        writes = [node for node in ast.walk(compressor_definition) if isinstance(node, ast.Name)
                  and isinstance(node.ctx, ast.Store) and node.id == name and node.lineno < producers[0].lineno]
        if len(writes) != 1:
            raise ValueError("save-partial split view is replaced before the source call: " + name)
    for call in producers + consumers:
        options = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
        if options["state_cache"] != "state_cache" or options["pdl_kwargs"] != "pdl_kwargs":
            raise ValueError("compressor producer/consumer state or ordering arguments diverged")
    compact_forward = re.sub(r"\s+", "", ast.unparse(compressor_definition))
    if ("state_cache=self.state_cache.kv_cache" not in compact_forward
            or "{'launch_pdl':False}" not in compact_forward):
        raise ValueError("compressor persistent state ownership/PDL ordering changed")
    compressor_init = re.sub(r"\s+", "", ast.unparse(_definition(
        ast.parse((root / _COMPRESSOR).read_text()), "DeepseekCompressor.__init__")))
    if any(fragment not in compressor_init for fragment in (
            "self.overlap=compress_ratio==4", "self.coff=1+self.overlap", "state_dtype=torch.float32")):
        raise ValueError("compressor C4 overlap width or FP32 state dtype changed")
    anchors["compressor-state-dataflow"] = {
        "split": {"path": str(root / _COMPRESSOR), "line": splits[0].lineno, "expression": ast.unparse(splits[0])},
        "producer": anchors["save-partial"], "consumer": anchors["compressor-forward"],
        "scope": "projection views, source valid-slot producer and persistent module identity; actual state pages and cross-step mapping remain unbound"}
    frame_definitions = []
    for relative, symbol, fragments in (
        (_COMPRESSOR, "CompressorStateCache.bind_kv_cache", ("self.kv_cache=kv_cache.squeeze(1)",)),
        (_COMPRESSOR, "DeepseekCompressor.__init__", (
            "self.ape=nn.Parameter(torch.empty((compress_ratio,self.coff*self.head_dim),dtype=state_dtype,device=self.device),requires_grad=False)",)),
        (_COMPRESSOR, "DeepseekCompressor.forward", (
            "token_to_req_indices=state_metadata.token_to_req_indices", "slot_mapping=state_metadata.slot_mapping",
            "num_actual=slot_mapping.shape[0]", "state_cache=self.state_cache.kv_cache",
            "state_width=state_cache.shape[-1]//2")),
        ("vllm/models/deepseek_v4/common/ops/save_partial_states.py", "SavePartialStatesKernel.__call__", (
            "num_actual=slot_mapping.shape[0]", "head_size=kv.shape[-1]", "return((num_actual,),dict(",
            "kv_stride=kv.stride(0)", "score_stride=score.stride(0)", "state_cache_stride0=state_cache.stride(0)",
            "state_cache_stride1=state_cache.stride(1)")),
    ):
        definition = _definition(ast.parse((root / relative).read_text()), symbol)
        compact = re.sub(r"\s+", "", ast.unparse(definition))
        if any(fragment not in compact for fragment in fragments):
            raise ValueError("save-partial caller/view/materialization source changed: " + symbol)
        if symbol == "CompressorStateCache.bind_kv_cache" and (
                len(definition.body) != 1 or ast.unparse(definition.body[0]) != "self.kv_cache = kv_cache.squeeze(1)"):
            raise ValueError("compressor cache binding no longer consists of the single aliasing squeeze")
        if symbol in ("CompressorStateCache.bind_kv_cache", "SavePartialStatesKernel.__call__"):
            if any(isinstance(node, ast.Call) and ast.unparse(node.func) in
                   ("torch.empty", "torch.empty_like", "torch.zeros", "torch.zeros_like", "torch.tensor", "torch.clone")
                   for node in ast.walk(definition)):
                raise ValueError("save-partial source caller unexpectedly allocates global tensor storage")
        frame_definitions.append(_function_anchor(root, relative, symbol))
    anchors["compressor-state-dataflow"]["tensor_frame"] = {
        "definitions": frame_definitions,
        "scope": "original save-partial caller/view/parameter objects only; actual page alias, reuse and runtime remain unbound"}
    cute_path = "vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py"
    triton_path = "vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py"
    finish_definitions = []
    inventories = {
        "DeepseekCompressor.forward": {"_SAVE_PARTIAL_STATES_KERNEL": 1, "cast": 2, "compress_norm_rope_store_fn": 1,
            "current_platform.is_cuda": 2, "current_platform.is_rocm": 1, "current_platform.is_xpu": 1,
            "dict": 1, "get_forward_context": 1, "getattr": 1, "isinstance": 1, "kv_score.split": 1},
        "SparseAttnCompressC128Block8Kernel.__call__": {"self.dispatch": 1, "torch.empty": 1},
        "SparseAttnCompressorCuteDSL.__call__": {"ValueError": 4, "positions.numel": 1,
            "self.c128_compress": 1, "self.c128_store": 1, "self.c128_store_full": 1, "self.c4_store": 1, "self.c4_store_full": 1},
        "compress_norm_rope_store_triton": {"_FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL": 1,
            "block_table.stride": 1, "cos_sin_cache.stride": 1, "kernel[num_actual,]": 1,
            "kv_cache.stride": 1, "state_cache.stride": 2, "triton.next_power_of_2": 1},
        "FusedKVCompressNormRopeInsertIndexerTritonKernel.__call__": {"block_table.stride": 1,
            "cos_sin_cache.stride": 1, "dict": 1, "kv_cache.stride": 1, "state_cache.stride": 2, "triton.next_power_of_2": 1}}
    for relative, symbol, fragments in (
        (_COMPRESSOR, "DeepseekCompressor.forward", (
            "cos_sin_cache=rotary_emb.cos_sin_cache", "k_cache_metadata=cast(Any,attn_metadata[self.k_cache_prefix])",
            "k_cache_layer=self._static_forward_context[self.k_cache_prefix]", "kv_cache=k_cache_layer.kv_cache",
            "rms_norm_weight=self.norm.weight", "rms_norm_eps=self.rms_norm_eps")),
        (cute_path, "SparseAttnCompressC128Block8Kernel.__call__", (
            "ifcompressed_kvisNone:compressed_kv=torch.empty((num_actual,head_dim),dtype=torch.float32,device=state_cache.device)",
            "return(compile_key,launch_args,compressed_kv)")),
        (cute_path, "SparseAttnCompressorCuteDSL.__call__", (
            "compressed_kv=self.c128_compress(", "num_actual=num_actual", "kv_slot_mapping=k_cache_metadata.slot_mapping",
            "self.c128_store(compressed_kv=compressed_kv,")),
        (triton_path, "compress_norm_rope_store_triton", ("_FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL(",)),
        (triton_path, "FusedKVCompressNormRopeInsertIndexerTritonKernel.__call__", (
            "k_cache_ptr=kv_cache", "kv_slot_mapping_ptr=k_cache_metadata.slot_mapping",
            "kv_cache_block_size=kv_cache.shape[1]", "KV_BLOCK_STRIDE=kv_cache.stride(0)")),
    ):
        definition = _definition(ast.parse((root / relative).read_text()), symbol)
        compact = re.sub(r"\s+", "", ast.unparse(definition))
        if any(fragment not in compact for fragment in fragments):
            raise ValueError("compressor finish caller/temp/cache source changed: " + symbol)
        allocations = [ast.unparse(node.func) for node in ast.walk(definition) if isinstance(node, ast.Call)
                       and ast.unparse(node.func).startswith("torch.")]
        if allocations != (["torch.empty"] if symbol == "SparseAttnCompressC128Block8Kernel.__call__" else []):
            raise ValueError("compressor finish caller global allocation inventory changed: " + symbol)
        names = [ast.unparse(node.func) for node in ast.walk(definition) if isinstance(node, ast.Call)]
        if {name: names.count(name) for name in set(names)} != inventories[symbol]:
            raise ValueError("compressor finish caller has unreviewed operations: " + symbol)
        if symbol == "SparseAttnCompressC128Block8Kernel.__call__" and (
                "launch_args=(state_cache,token_to_req_indices,positions,slot_mapping,block_table,compressed_kv)" not in compact):
            raise ValueError("C128 producer no longer passes its returned FP32 object to the kernel")
        if symbol == "SparseAttnCompressorCuteDSL.__call__":
            for call in (node for node in ast.walk(definition) if isinstance(node, ast.Call)
                         and ast.unparse(node.func) in ("self.c4_store", "self.c128_compress", "self.c128_store")):
                arguments = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
                required = {"positions": "positions", "slot_mapping": "slot_mapping"}
                if ast.unparse(call.func) == "self.c128_compress":
                    required.update(state_cache="state_cache", num_actual="num_actual")
                    if "compressed_kv" in arguments:
                        raise ValueError("C128 source caller no longer requests its own fresh temporary")
                else:
                    required.update(kv_cache="kv_cache", kv_slot_mapping="k_cache_metadata.slot_mapping",
                                    rms_norm_weight="rms_norm_weight", cos_sin_cache="cos_sin_cache")
                    required["compressed_kv" if ast.unparse(call.func) == "self.c128_store" else "state_cache"] = (
                        "compressed_kv" if ast.unparse(call.func) == "self.c128_store" else "state_cache")
                if any(name not in arguments or arguments[name] != value for name, value in required.items()):
                    raise ValueError("compressor finish stage no longer shares its declared source tensor arguments")
        finish_definitions.append(_function_anchor(root, relative, symbol))
    for relative, symbol, fragment in (
        ("vllm/model_executor/warmup/jit_warmup.py", "kernel_launcher", "returnself.launch(launch_spec,{})"),
        ("vllm/model_executor/warmup/jit_warmup_cutedsl_helper.py", "VllmCuTeDSLJitKernel.launch",
         "returnlaunch_spec[2]iflen(launch_spec)==3elseresult"),
    ):
        definition = _definition(ast.parse((root / relative).read_text()), symbol)
        if fragment not in re.sub(r"\s+", "", ast.unparse(definition)):
            raise ValueError("CuTe source launch no longer returns the declared temporary object")
        finish_definitions.append(_function_anchor(root, relative, symbol))
    for symbol, expected in (("DeepseekV4Attention.__init__", "self.prefix"),
                             ("DeepseekV4Indexer.__init__", "self.k_cache.prefix")):
        definition = _definition(ast.parse((root / _ATTENTION).read_text()), symbol)
        constructor, = [node for node in ast.walk(definition) if isinstance(node, ast.Call)
                        and ast.unparse(node.func) == "DeepseekCompressor"]
        arguments = {keyword.arg: ast.unparse(keyword.value) for keyword in constructor.keywords}
        if arguments["k_cache_prefix"] != expected:
            raise ValueError("compressor destination cache module owner changed: " + symbol)
        finish_definitions.append(_function_anchor(root, _ATTENTION, symbol))
    attention_tree = ast.parse((root / _ATTENTION).read_text())
    prepare = _definition(attention_tree, "DeepseekV4Attention._prepare_and_attn")
    indexer_and_attention = _definition(attention_tree, "DeepseekV4Attention._sparse_indexer_and_attn")
    join, = [node for node in ast.walk(prepare) if isinstance(node, ast.Call) and ast.unparse(node.func) == "execute_in_parallel"]
    dispatch, = [node for node in ast.walk(prepare) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == "self._sparse_indexer_and_attn"]
    indexer_call, = [node for node in ast.walk(indexer_and_attention) if isinstance(node, ast.Call)
                    and ast.unparse(node.func) == "self.indexer.indexer_op"]
    if (join.lineno >= dispatch.lineno or [ast.unparse(arg) for arg in indexer_call.args]
            != ["hidden_states", "q_quant", "None", "index_weights"]):
        raise ValueError("indexer no longer follows the prepare join or has new main-cache input")
    finish_definitions.extend(_function_anchor(root, _ATTENTION, symbol) for symbol in (
        "DeepseekV4Attention._prepare_and_attn", "DeepseekV4Attention._sparse_indexer_and_attn"))
    anchors["compressor-state-dataflow"]["finish_frame"] = {
        "definitions": finish_definitions, "caller_call_inventories": inventories,
        "scope": "CUDA packed FP8 main/indexer caller objects and C128 fresh FP32 result; symbolic cache pages, not actual allocator/provider"}
    window_programs = []
    for relative, symbol, fragments in (
        ("vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py", "SparseAttnCompressNormRopeStoreC4Kernel.kernel", (
            "window=(1+int(overlap))*compress_ratio", "start=position-Int64(window-1)",
            "block_index=pos//block_size", "block_offset=pos-block_index*block_size",
            "head_offset=Int64(row//compress_ratio*head_dim)", "ifpos>=Int64(0):")),
        ("vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py", "SparseAttnCompressC128Block8Kernel.kernel", (
            "start=position-Int64(compress_ratio-1)", "first_block_index=start//Int64(state_block_size)",
            "warp_block_index=first_block_index+(warp_id*2).to(Int64)", "block_offset=Int64(i-state_block_size)",
            "row_tensor=state_cache[block_number_i32.to(Int64),block_offset,None]")),
        ("vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py", "_fused_kv_compress_norm_rope_insert_indexer_attn", (
            "start=position-(1+OVERLAP)*COMPRESS_RATIO+1", "pos=start+tokens", "mask_pos=pos>=0",
            "block_indices=pos//block_size", "block_offsets=pos%block_size",
            "head_offset=(tokens>=COMPRESS_RATIO).to(tl.int32)*HEAD_SIZE")),
    ):
        definition = _definition(ast.parse((root / relative).read_text()), symbol)
        body = re.sub(r"\s+", "", ast.unparse(definition))
        if any(fragment not in body for fragment in fragments):
            raise ValueError("compressor state window/page/half source changed: " + symbol)
        window_programs.append({"path": str(root / relative), "symbol": symbol, "line": definition.lineno,
                                "checked_expressions": list(fragments)})
    c128 = _definition(ast.parse(Path(window_programs[1]["path"]).read_text()), "SparseAttnCompressC128Block8Kernel")
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in c128.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ("num_warps", "rows_per_warp", "state_block_size", "compress_ratio")}
    if constants != {"num_warps": 8, "rows_per_warp": 16, "state_block_size": 8, "compress_ratio": 128}:
        raise ValueError("C128 state page/row partition changed")
    anchors["compressor-state-dataflow"]["window_programs"] = window_programs
    anchors["compressor-state-dataflow"]["C128_row_partition"] = constants
    metadata_init = _definition(ast.parse((root / _COMPRESSOR).read_text()), "CompressorMetadataBuilder.__init__")
    metadata_build = _definition(ast.parse((root / _COMPRESSOR).read_text()), "CompressorMetadataBuilder.build")
    if ("self.block_size=mla_spec.block_size" not in re.sub(r"\s+", "", ast.unparse(metadata_init))
            or "block_size=self.block_size" not in re.sub(r"\s+", "", ast.unparse(metadata_build))
            or "block_size=state_metadata.block_size" not in compact_forward):
        raise ValueError("compressor state block size no longer comes from resolved metadata")
    for call in producers + consumers:
        if not any(keyword.arg == "block_size" and ast.unparse(keyword.value) == "block_size" for keyword in call.keywords):
            raise ValueError("compressor save/finish must share the resolved metadata block size")
    unifier = _definition(ast.parse((root / "vllm/v1/core/kv_cache_utils.py").read_text()), "unify_kv_cache_spec_page_size")
    if "new_block_size=layer_spec.block_size*ratio" not in re.sub(r"\s+", "", ast.unparse(unifier)):
        raise ValueError("state block-size unification no longer preserves constructor multiples")
    c128_dispatch = _definition(ast.parse(Path(window_programs[0]["path"]).read_text()), "SparseAttnCompressorCuteDSL.__call__")
    if "ifblock_size!=self.c128_compress.state_block_size:raiseValueError" not in re.sub(r"\s+", "", ast.unparse(c128_dispatch)):
        raise ValueError("C128 dispatcher no longer requires the block8 source domain")
    anchors["compressor-state-dataflow"]["resolved_state_block_size"] = {"definitions": [
        _function_anchor(root, relative, symbol) for relative, symbol in (
            (_COMPRESSOR, "CompressorMetadataBuilder.__init__"),
            (_COMPRESSOR, "CompressorMetadataBuilder.build"),
            (_COMPRESSOR, "CompressorStateCache.__init__"),
            ("vllm/v1/core/kv_cache_utils.py", "unify_kv_cache_spec_page_size"),
            ("vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py", "SparseAttnCompressorCuteDSL.__call__"),
            ("vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py", "split_kv_compress_norm_rope_insert_sparse_attn_cutedsl"))],
        "C4_constructor_default_not_selected_metadata": True,
        "scope": "C4 address program uses resolved metadata B; C128 supported CuTe split requires B=8; no normal allocator/layout acceptance"}
    metadata_guards = []
    for relative, symbol, condition in (
        (window_programs[0]["path"], "SparseAttnCompressNormRopeStoreC4Kernel.kernel.device_kernel", "completed boundaries only"),
        (window_programs[1]["path"], "SparseAttnCompressC128Block8Kernel.kernel.device_kernel", "before boundary guard"),
        (window_programs[2]["path"], "_fused_kv_compress_norm_rope_insert_indexer_attn", "completed boundaries only"),
    ):
        definition = _definition(ast.parse(Path(relative).read_text()), symbol)
        reads = [node for node in ast.walk(definition) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "req_idx" for target in node.targets)
                 and ast.unparse(node.value) in (
                     "token_to_req_indices[token_idx]", "tl.load(token_to_req_indices_ptr + token_idx)")]
        if len(reads) != 1:
            raise ValueError("finish request-index read missing or ambiguous: " + symbol)
        read = reads[0]
        if ".device_kernel" in symbol:
            active_guards = [node for node in definition.body if isinstance(node, ast.If)
                             and ast.unparse(node.test) == "active"]
            if len(active_guards) != 1:
                raise ValueError("finish active boundary guard changed: " + symbol)
            guard = active_guards[0]
            if condition == "completed boundaries only":
                valid = any(node is read for node in ast.walk(guard))
            else:
                valid = read.lineno < guard.lineno and not any(node is read for node in ast.walk(guard))
                enclosing = [node for node in definition.body if isinstance(node, ast.If)
                             and any(item is read for item in ast.walk(node))]
                valid = valid and len(enclosing) == 1 and ast.unparse(enclosing[0].test) == "lane_id == 0 and has_req_idx"
        else:
            guards = [node for node in definition.body if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "(position + 1) % COMPRESS_RATIO != 0"]
            valid = len(guards) == 1 and guards[0].end_lineno < read.lineno
            valid = valid and len(guards[0].body) == 1 and isinstance(guards[0].body[0], ast.Return)
            guard = guards[0] if guards else None
        if not valid:
            raise ValueError("finish request-index boundary condition changed: " + symbol)
        metadata_guards.append({"path": relative, "symbol": symbol, "read_line": read.lineno,
            "guard_line": guard.lineno, "read": ast.unparse(read), "condition": condition})
    anchors["compressor-state-dataflow"]["metadata_request_guards"] = metadata_guards
    floating_programs = []
    for relative, symbol, fragments in (
        (window_programs[0]["path"], "SparseAttnCompressNormRopeStoreC4Kernel.kernel.device_kernel", (
            "local_sum=cute.make_rmem_tensor((elems_per_lane,),Float32)",
            "local_product=cute.make_rmem_tensor((elems_per_lane,),Float32)",
            "new_max=cute.arch.fmax(local_max[e],score)", "x[e]=local_product[e]/local_sum[e]",
            "local_sum[e]=local_sum[e]*old_scale+new_scale")),
        (window_programs[1]["path"], "SparseAttnCompressC128Block8Kernel.kernel.device_kernel", (
            "score_vals=cute.make_rmem_tensor(row_layout,Float32)",
            "local_sum=cute.make_rmem_tensor((elems_per_lane,),Float32)",
            "local_sum[e]+=exp_score", "local_product[e]+=kv_vals[i,e]*exp_score",
            "(score_vals[i,e]-local_max[e])*Float32(rcp_ln2)",
            "(local_warp_max-global_max)*Float32(rcp_ln2)", "global_product/global_sum")),
        (window_programs[1]["path"], "SparseAttnNormRopeStoreKernel.kernel.device_kernel", (
            "weight0=rms_norm_weight[elem0].to(Float32)", "local_sumsq=x0*x0+x1*x1",
            "total/Float32(head_dim)+rms_norm_eps", "abs0=cute.math.absf(q0)",
            "absmax/Float32(fp8_max)", "cute.arch.fmax(q0*inv_scale,Float32(-fp8_max))")),
        (window_programs[2]["path"], "_fused_kv_compress_norm_rope_insert_indexer_attn", (
            "score=tl.softmax(score,dim=0)", "compressed_kv=tl.sum(kv*score,axis=0)",
            "variance=tl.sum(compressed_kv*compressed_kv,axis=0)/HEAD_SIZE",
            "rrms=tl.rsqrt(variance+rms_norm_eps)", "normed=compressed_kv*rrms*rms_w")),
    ):
        definition = _definition(ast.parse(Path(relative).read_text()), symbol)
        body = re.sub(r"\s+", "", ast.unparse(definition))
        if any(fragment not in body for fragment in fragments):
            raise ValueError("finish floating expression/type contract changed: " + symbol)
        floating_programs.append({"path": relative, "symbol": symbol, "line": definition.lineno,
            "checked_expressions": list(fragments)})
    anchors["compressor-state-dataflow"]["floating_programs"] = floating_programs
    slot_relative = "vllm/v1/worker/block_table.py"
    slot_tree = ast.parse((root / slot_relative).read_text())
    slot_kernel = _definition(slot_tree, "ComputeSlotMappingKernel.kernel")
    slot_body = re.sub(r"\s+", "", ast.unparse(slot_kernel))
    for fragment in ("virtual_block_size=KV_CACHE_BLOCK_SIZE*TOTAL_CP_WORLD_SIZE",
                     "virtual_block_indices=pos//virtual_block_size",
                     "virtual_block_offsets=pos-virtual_block_indices*virtual_block_size",
                     "block_indices=virtual_block_indices*BLOCKS_PER_KV_BLOCK+local_block_offsets//block_size",
                     "slot_offsets=local_block_offsets%block_size", "slot_ids=block_numbers*block_size+slot_offsets",
                     "slot_ids=tl.where(is_local,slot_ids,PAD_ID)", "range(num_tokens,max_num_tokens,BLOCK_SIZE)"):
        if fragment not in slot_body:
            raise ValueError("source slot/page mapping program changed")
    slot_init = re.sub(r"\s+", "", ast.unparse(_definition(slot_tree, "BlockTable.__init__")))
    for fragment in ("self.blocks_per_kv_block=block_size//kernel_block_size", "dtype=torch.int32", "dtype=torch.int64"):
        if fragment not in slot_init:
            raise ValueError("source block-table hybrid relation or storage dtype changed")
    metadata_builder = _definition(ast.parse((root / _COMPRESSOR).read_text()), "CompressorMetadataBuilder.build")
    metadata_body = re.sub(r"\s+", "", ast.unparse(metadata_builder))
    for fragment in ("block_table=common_attn_metadata.block_table_tensor.clamp_(min=0)",
                     "slot_mapping=common_attn_metadata.slot_mapping", "block_size=self.block_size"):
        if fragment not in metadata_body:
            raise ValueError("compressor slot/table forwarding or in-place normalization changed")
    pad_relative = "vllm/v1/attention/backends/utils.py"
    pad_constants = [ast.literal_eval(node.value) for node in ast.parse((root / pad_relative).read_text()).body
                     if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "PAD_SLOT_ID" for target in node.targets)]
    if pad_constants != [-1]:
        raise ValueError("source padded slot sentinel changed")
    anchors["compressor-state-dataflow"]["slot_mapping"] = {
        "kernel": _function_anchor(root, slot_relative, "ComputeSlotMappingKernel.kernel"),
        "constructor": _function_anchor(root, slot_relative, "BlockTable.__init__"),
        "forwarding": _function_anchor(root, _COMPRESSOR, "CompressorMetadataBuilder.build"),
        "scope": "CP1 positive-size/divisible-block algebra with nonnegative addressed table IDs; clamp extent, actual data and preparation work remain unbound"}
    allocation_relative = "vllm/v1/worker/utils.py"
    allocation = _definition(ast.parse((root / allocation_relative).read_text()), "allocate_kv_cache")
    allocation_body = re.sub(r"\s+", "", ast.unparse(allocation))
    if ("buf=torch.zeros(buf_size,dtype=torch.int8,device=device)" not in allocation_body
            or "views=create_kv_cache_views(buf,spec,num_blocks,layout,tensor,kernel_block_size=kernel_block_size)" not in allocation_body):
        raise ValueError("source cache groups no longer use the same zeroed byte backing")
    views_relative = "vllm/v1/kv_cache_interface.py"
    views = _definition(ast.parse((root / views_relative).read_text()), "create_kv_cache_views")
    view_body = re.sub(r"\s+", "", ast.unparse(views))
    for fragment in ("torch.as_strided(raw,size=logical_shape,stride=strides,storage_offset=raw.storage_offset()+kv_cache_tensor.offset)",
                     "cache_logical=view_5d[layer_idx]", "cache_logical=cache_logical.view(dtype)"):
        if fragment not in view_body:
            raise ValueError("source cache view offset/strides/dtype alias program changed")
    bind = _definition(ast.parse((root / _COMPRESSOR).read_text()), "CompressorStateCache.bind_kv_cache")
    if "self.kv_cache = kv_cache.squeeze(1)" not in ast.unparse(bind):
        raise ValueError("compressor state binding no longer uses a squeeze view")
    anchors["compressor-state-dataflow"]["shared_cache_views"] = {
        "allocation": _function_anchor(root, allocation_relative, "allocate_kv_cache"),
        "views": _function_anchor(root, views_relative, "create_kv_cache_views"),
        "bind": _function_anchor(root, _COMPRESSOR, "CompressorStateCache.bind_kv_cache"),
        "scope": "one shared backing and source offset/strides/views; selected layout, live allocation ownership and interference remain unbound"}
    coordinator_relative = "vllm/v1/core/kv_cache_coordinator.py"
    packing_relative = "vllm/v1/core/kv_cache_utils.py"
    layout_relative = "vllm/v1/kv_cache_layout.py"
    indexer_backend_relative = "vllm/v1/attention/backends/mla/indexer.py"
    for relative, symbol, fragments in (
        (coordinator_relative, "KVCacheCoordinator.__init__", (
            "self.block_pool=BlockPool(", "block_pool=self.block_pool", "kv_cache_group_id=i")),
        (packing_relative, "_get_kv_cache_bytes_per_block", (
            "bytes_per_block=max((sum((_get_per_layer_spec(group,layer_name).page_size_bytesforlayer_nameingroup.layer_names))forgroupinkv_cache_groups))",)),
        (packing_relative, "get_kv_cache_config_from_groups", (
            "interleaved_block_stride=bytes_per_blockiflayout.is_block_outermostelseNone",
            "byte_offset=0", "fixed_strides=(None,interleaved_block_stride,None,None,None)",
            "offset=byte_offset*max(layer_stride,spec.page_size_bytes)//spec.page_size_bytes",
            "byte_offset+=len(layer_names)*spec.page_size_bytes")),
        (indexer_backend_relative, "DeepseekV4IndexerBackend.supported_kv_cache_layouts", (
            "return(KVCacheLayout.BLHNC,KVCacheLayout.BLNHC)",)),
        (views_relative, "create_kv_cache_views", (
            "ratio=shape_bytes[0]//num_blocks", "ifblock_stride!=dense_page_size:",
            "assertblock_stride%ratio==0", "block_stride//=ratio")),
    ):
        body = re.sub(r"\s+", "", ast.unparse(_definition(ast.parse((root / relative).read_text()), symbol)))
        if any(fragment not in body for fragment in fragments):
            raise ValueError("source group-overlay/manager-block packing changed: " + symbol)
    anchors["compressor-state-dataflow"]["group_overlay"] = {
        "definitions": [_function_anchor(root, relative, symbol) for relative, symbol in (
            (coordinator_relative, "KVCacheCoordinator.__init__"),
            (packing_relative, "_get_kv_cache_bytes_per_block"),
            (packing_relative, "get_kv_cache_config_from_groups"),
            (layout_relative, "KVCacheLayout"),
            (views_relative, "compute_layout_strides"),
            (views_relative, "create_kv_cache_views"),
            (indexer_backend_relative, "DeepseekV4IndexerBackend.supported_kv_cache_layouts"))],
        "scope": "generic block-outer one-head packing; retained page references and exclusive writes are preconditions, actual layout/groups/pages and unpinned reads remain unbound"}
    packed_relative = "vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py"
    index_finish_relative = "vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py"
    gather_relative = "vllm/models/deepseek_v4/common/ops/cache_utils.py"
    destination_definitions = (
        (packed_relative, "SparseAttnCompressNormRopeStoreC4Kernel.kernel.device_kernel"),
        (packed_relative, "SparseAttnNormRopeStoreKernel.kernel.device_kernel"),
        (index_finish_relative, "_fused_kv_compress_norm_rope_insert_indexer_attn"),
        (index_finish_relative, "FusedKVCompressNormRopeInsertIndexerTritonKernel.__call__"),
        (gather_relative, "DequantizeAndGatherKCacheKernel.kernel"),
        (packed_relative, "SparseAttnCompressorCuteDSL.__call__"),
    )
    for relative, symbol, fragments in (
        (packed_relative, destination_definitions[0][1], (
            "value_base=page*k_cache.stride[0]+kv_offset*Int64(token_stride)",
            "kv_cache_block_size*Int64(token_stride)", "kv_offset*Int64(scale_dim)",
            "k_cache.iterator[scale_base+Int64(nope_blocks)]=Uint8(0)")),
        (packed_relative, destination_definitions[1][1], (
            "value_base=page*k_cache.stride[0]+kv_offset*Int64(token_stride)",
            "scale_row_offset=static_block_size*Int64(token_stride)", "kv_offset*Int64(scale_dim)")),
        (index_finish_relative, destination_definitions[2][1], (
            "cache_block_ptr=k_cache_ptr+kv_block_idx.to(tl.int64)*KV_BLOCK_STRIDE",
            "fp8_ptr=cache_block_ptr+kv_pos_in_block*TOKEN_STRIDE",
            "kv_cache_block_size*TOKEN_STRIDE", "kv_pos_in_block*SCALE_DIM",
            "tl.store(scale_ptr.to(tl.pointer_type(tl.float32)),scale_val)")),
        (index_finish_relative, destination_definitions[3][1], (
            "kv_slot_mapping_ptr=k_cache_metadata.slot_mapping", "kv_cache_block_size=kv_cache.shape[1]",
            "KV_BLOCK_STRIDE=kv_cache.stride(0)")),
        (gather_relative, destination_definitions[4][1], (
            "physical_block_idx.to(tl.int64)*block_stride", "token_data_ptr=cache_block_ptr+pos_in_block*token_data_size",
            "cache_block_size*token_data_size", "pos_in_block*scale_dim")),
        (packed_relative, destination_definitions[5][1], (
            "self.c4_store(", "self.c128_store(", "kv_slot_mapping=k_cache_metadata.slot_mapping")),
    ):
        body = re.sub(r"\s+", "", ast.unparse(_definition(ast.parse((root / relative).read_text()), symbol)))
        if any(fragment not in body for fragment in fragments):
            raise ValueError("V4 packed destination/gather address source changed: " + symbol)
    anchors["compressor-state-dataflow"]["destination_cache_addresses"] = {
        "definitions": [_function_anchor(root, relative, symbol) for relative, symbol in destination_definitions],
        "scope": "conditional byte addresses, packed data/scale regions and separate destination metadata; actual values/pages and FlashMLA internals remain unbound"}
    pool_relative = "vllm/v1/core/block_pool.py"
    pool_tree = ast.parse((root / pool_relative).read_text())
    for symbol, fragments in (
        ("BlockPool.__init__", ("KVCacheBlock(idx,pool=self)foridxinrange(num_gpu_blocks)",
            "self.null_block=self.free_block_queue.popleft()", "self.null_block.is_null=True")),
        ("BlockPool.get_new_blocks", ("self.free_block_queue.popleft_n(num_blocks)", "assertblock.ref_cnt==0", "block.ref_cnt+=1")),
        ("BlockPool.is_block_writable", ("returnnotblock.is_nullandblock.ref_cnt==1and(block.block_hashisNone)",)),
        ("BlockPool.touch", ("ifblock.ref_cnt==0and(notblock.is_null):", "self.free_block_queue.remove(block)", "block.ref_cnt+=1")),
        ("BlockPool.free_blocks", ("block.ref_cnt-=1", "ifblock.ref_cnt==0and(notblock.is_null):")),
        ("BlockPool.get_cached_block", ("make_block_hash_with_group_id(block_hash,group_id)",)),
        ("BlockPool.unpin_blocks", ("block.ref_cnt-=1", "self._reuse_watchers[block.block_id]=on_reuse", "ifblock.ref_cnt==0:")),
    ):
        body = re.sub(r"\s+", "", ast.unparse(_definition(pool_tree, symbol)))
        if any(fragment not in body for fragment in fragments):
            raise ValueError("source cache lifetime rule changed: " + symbol)
    new_blocks = _definition(pool_tree, "BlockPool.get_new_blocks")
    notify = [node for node in ast.walk(new_blocks) if isinstance(node, ast.Call) and ast.unparse(node.func) == "self._notify_reuse"]
    increments = [node for node in ast.walk(new_blocks) if isinstance(node, ast.AugAssign) and ast.unparse(node.target) == "block.ref_cnt"]
    if len(notify) != 1 or not increments or notify[0].lineno >= min(node.lineno for node in increments):
        raise ValueError("source unpin reuse notification no longer precedes new ownership")
    utility_relative = "vllm/v1/core/kv_cache_utils.py"
    pack = _definition(ast.parse((root / utility_relative).read_text()), "make_block_hash_with_group_id")
    if "block_hash+group_id.to_bytes(4,'big',signed=False)" not in re.sub(r"\s+", "", ast.unparse(pack)):
        raise ValueError("source cache group hash identity changed")
    manager_relative = "vllm/v1/core/single_type_kv_cache_manager.py"
    manager_tree = ast.parse((root / manager_relative).read_text())
    for symbol, fragments in (
        ("SlidingWindowManager.get_num_skipped_tokens", ("returnmax(0,num_computed_tokens-self.sliding_window+1-self.extra_retained_tokens)",)),
        ("SingleTypeKVCacheManager.remove_skipped_blocks", ("num_skipped_blocks=num_skipped_tokens//self.block_size",
            "num_skipped_blocks=min(num_skipped_blocks,len(blocks))", "self._remove_blocks_in_range(request_id,0,num_skipped_blocks)")),
    ):
        body = re.sub(r"\s+", "", ast.unparse(_definition(manager_tree, symbol)))
        if any(fragment not in body for fragment in fragments):
            raise ValueError("source state retention/free-prefix rule changed")
    registrations = [node for node in ast.walk(manager_tree) if isinstance(node, ast.Call)
                     and ast.unparse(node.func) == "KVCacheSpecRegistry.register" and len(node.args) >= 2
                     and ast.unparse(node.args[0]) == "SlidingWindowMLASpec" and ast.unparse(node.args[1]) == "SlidingWindowManager"]
    if len(registrations) != 1:
        raise ValueError("default sliding-window MLA manager registration changed")
    anchors["compressor-state-dataflow"]["lifetime_retention"] = {
        "pool": [_function_anchor(root, pool_relative, symbol) for symbol in
            ("BlockPool.get_new_blocks", "BlockPool.touch", "BlockPool.is_block_writable", "BlockPool.free_blocks", "BlockPool.unpin_blocks")],
        "group_identity": _function_anchor(root, utility_relative, "make_block_hash_with_group_id"),
        "retention": _function_anchor(root, manager_relative, "SlidingWindowManager.get_num_skipped_tokens"),
        "free_prefix": _function_anchor(root, manager_relative, "SingleTypeKVCacheManager.remove_skipped_blocks"),
        "scope": "logical ref-count/hash/null/reuse rules and nonnegative-extra retention bound; no actual pool state, GPU reader fence or normal capacity acceptance"}
    allocation_definitions = []
    for symbol, fragments in (
        ("SingleTypeKVCacheManager.get_num_blocks_to_allocate", ("ifrequest_idinself.num_cached_block:", "assertlen(new_computed_blocks)==0")),
        ("SingleTypeKVCacheManager.allocate_new_blocks", ("ifrequest_idinself._partial_hit_reqs:",
            "cow_block=self.block_pool.get_new_blocks(1)[0]", "self._apply_cow(request_id,block_idx,source_block,cow_block)",
            "req_blocks.extend(new_blocks)")),
        ("SingleTypeKVCacheManager._apply_cow", ("req_blocks[block_idx]=cow_block",
            "self._pending_cow_copies.append((source_block,cow_block))", "cow_block.ref_cnt+=1")),
        ("SingleTypeKVCacheManager.take_pending_cow_copies", ("pending_copies=self._pending_cow_copies",
            "self._pending_cow_copies=[]", "returnpending_copies")),
    ):
        definition = _definition(manager_tree, symbol)
        compact = re.sub(r"\s+", "", ast.unparse(definition))
        if any(fragment not in compact for fragment in fragments):
            raise ValueError("source cache append/CoW transition changed: " + symbol)
        allocation_definitions.append(_function_anchor(root, manager_relative, symbol))
    anchors["compressor-state-dataflow"]["allocation_transition"] = {
        "definitions": allocation_definitions,
        "scope": "append and partial-hit CoW are distinct source branches; draining copy descriptors does not prove execution or completion"}
    copy_definitions = []
    for relative, symbol, fragments in (
        ("vllm/v1/worker/utils.py", "copy_kv_cache_blocks_inplace", (
            "indices_np=np.array(kv_cache_block_copies,dtype=np.int64)",
            "key=(cache.device,cache.data_ptr())", "ifkeyinseen:continue", "seen.add(key)",
            "(src,dst)=indices.unbind(dim=1)", "divmod(cache.shape[0],num_blocks)",
            "assertremainder==0", "storage_key=(cache.device,storage.data_ptr())",
            "cache.stride(0)*cache.element_size()*kernel_blocks_per_block",
            "ifstorage.nbytes()==num_blocks*scheduler_block_stride:",
            "ifstorage_keyincopied_storages:continue", "copied_storages.add(storage_key)",
            "blocks.set_(storage)", "blocks.view(num_blocks,-1)",
            "cache.unflatten(0,(num_blocks,kernel_blocks_per_block))", "blocks[dst]=blocks[src]")),
        ("vllm/v1/core/kv_cache_manager.py", "KVCacheManager.take_kv_cache_block_copies", (
            "pending_copies.extend(mgr.take_pending_cow_copies())", "forblockinpair", "return(copies,retained_blocks)")),
        ("vllm/v1/core/sched/scheduler.py", "Scheduler._free_cow_retained_blocks", (
            "ifnotself.defer_block_freeorfence_seq<=self.processed_step_seq:",
            "self.kv_cache_manager.block_pool.free_blocks(blocks)",
            "self.deferred_frees.append((fence_seq,blocks[::-1]))")),
    ):
        definition = _definition(ast.parse((root / relative).read_text()), symbol)
        compact = re.sub(r"\s+", "", ast.unparse(definition))
        if any(fragment not in compact for fragment in fragments):
            raise ValueError("worker CoW copy/storage or retention source changed: " + symbol)
        copy_definitions.append(_function_anchor(root, relative, symbol))
    runner_relative = "vllm/v1/worker/gpu_model_runner.py"
    runner_tree = ast.parse((root / runner_relative).read_text())
    for symbol, names in (("GPUModelRunner._update_states", ("self._zero_block_ids", "copy_kv_cache_blocks_inplace")),
                          ("GPUModelRunner.execute_model", ("self._update_states", "self._prepare_inputs", "self._model_forward"))):
        definition = _definition(runner_tree, symbol)
        positions = []
        for name in names:
            calls = [node for node in ast.walk(definition) if isinstance(node, ast.Call) and ast.unparse(node.func) == name]
            if len(calls) != 1:
                raise ValueError("worker CoW source call inventory changed: " + name)
            positions.append(calls[0].lineno)
        if positions != sorted(positions):
            raise ValueError("worker CoW copy is no longer submitted before input preparation/forward")
        copy_definitions.append(_function_anchor(root, runner_relative, symbol))
    anchors["compressor-state-dataflow"]["worker_copy_program"] = {
        "definitions": copy_definitions,
        "scope": "conditional storage/view copy and Python submission order only; actual descriptors, H2D transport and GPU completion remain unbound"}
    swa_tree = ast.parse((root / "vllm/v1/attention/backends/mla/sparse_swa.py").read_text(encoding="utf-8"))
    swa_init = _definition(swa_tree, "DeepseekV4SWACache.__init__")
    swa_defaults = dict(zip((arg.arg for arg in swa_init.args.args[-len(swa_init.args.defaults):]),
                            (ast.literal_eval(default) for default in swa_init.args.defaults)))
    if swa_defaults["packed_bytes_per_token"] != 584:
        raise ValueError("V4 SWA cache packed row layout changed")
    anchors["swa-cache-layout"]["constructor"] = _constructor(
        root, _ATTENTION, "DeepseekV4Attention", "swa_cache_layer", "DeepseekV4SWACache")
    if "packed_bytes_per_token" in anchors["swa-cache-layout"]["constructor"]["keywords"]:
        raise ValueError("V4 SWA cache constructor overrides the default packed row size")
    anchors["swa-cache-layout"]["record_bytes"] = swa_defaults["packed_bytes_per_token"]
    anchors["flashinfer-input-quant"] = (None if flashinfer_source_root is None
                                         else _flashinfer_quantizer_anchor(flashinfer_source_root))
    anchors["clamped-activation"] = clamped_activation_source_reference(root)
    moe_root = "vllm/model_executor/layers/fused_moe/"
    anchors["marlin-moe-source"] = {"definitions": [
        _function_anchor(root, moe_root + filename, symbol) for filename, symbol in (
            ("experts/marlin_moe.py", "_fused_marlin_moe"),
            ("experts/marlin_moe.py", "fused_marlin_moe"),
            ("experts/marlin_moe.py", "MarlinExperts.apply"),
            ("experts/marlin_moe.py", "MarlinExperts.moe_sum"),
            ("expert_map_manager.py", "determine_expert_map"),
            ("modular_kernel.py", "FusedMoEExpertsModular.activation"),
            ("activation.py", "silu_and_mul_with_clamp"),
            ("activation.py", "apply_moe_activation"),
            ("utils.py", "_resize_cache"))],
        "activation_cuda_source": "clamped-activation",
        "reduce_cuda_source": _cpp_anchor(root, "csrc/libtorch_stable/moe/moe_align_sum_kernels.cu",
                                           "__global__ void moe_sum_vec_dynamic_kernel("),
        "scale_postprocess": [_function_anchor(root,
            "vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py", symbol)
            for symbol in ("prepare_moe_mxfp4_layer_for_marlin", "mxfp4_marlin_process_scales")],
        "scope": "conditional modular Marlin EP-off W4A16 source, not loaded backend or checkpoint routing"}
    reduce_relative = "csrc/libtorch_stable/moe/moe_align_sum_kernels.cu"
    reduce_forms = []
    for symbol, expressions in (
        ("__global__ void moe_sum_vec_kernel(", ("floatacc[VEC]", "acc[j]=0.f", "k<TOPK", "acc[j]+=static_cast<float>(packed.val[j])")),
        ("__global__ void moe_sum_vec_dynamic_kernel(", ("floatacc[VEC]", "acc[j]=0.f", "k<topk", "acc[j]+=static_cast<float>(packed.val[j])")),
        ("__global__ void moe_sum_scalar_kernel(", ("floatx=0.f", "k<topk", "x+=static_cast<float>(VLLM_LDG(")),
    ):
        form = _cpp_anchor(root, reduce_relative, symbol)
        compact = re.sub(r"\s+", "", form["source"])
        if any(expression not in compact for expression in expressions):
            raise ValueError("Marlin FP32 zero-seeded expert reduction source changed")
        reduce_forms.append(form)
    anchors["marlin-moe-source"]["reduce_cuda_source_forms"] = reduce_forms
    marlin_template = root / "csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h"
    template_text = marlin_template.read_text()
    matmul_source = _cpp_anchor(root, "csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h", "auto matmul =")
    compact_matmul = re.sub(r"\s+", "", matmul_source["source"])
    if ("for(intj=0;j<4;j++)" not in compact_matmul
            or "for(inti=0;i<thread_m_blocks;i++)" not in compact_matmul
            or compact_matmul.count("mma_trans<a_type_id,use_fp16_accum>") != 1
            or compact_matmul.count("mma<a_type_id,use_fp16_accum>") != 2
            or "m_block_size_8?8:(16*thread_m_blocks)" not in re.sub(r"\s+", "", template_text)):
        raise ValueError("Marlin M-padded BF16 source MMA geometry changed")
    mma_relative = "csrc/libtorch_stable/quantization/marlin/marlin_mma.h"
    mma_sources = [_cpp_anchor(root, mma_relative, signature) for signature in
                   ("__device__ inline void mma(", "__device__ inline void mma_trans(")]
    for form in mma_sources:
        bf16 = form["source"].split("std::is_same<scalar_t, nv_bfloat16>::value", 1)[1].split("} else if constexpr", 1)[0]
        if (bf16.count("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32") != 1
                or bf16.count("asm volatile(") != 1):
            raise ValueError("Marlin BF16 regular/transposed source MMA instruction changed")
    anchors["marlin-moe-source"]["padded_tensor"] = {
        "matmul": matmul_source, "MMA": mma_sources,
        "scope": "BF16 source arithmetic under declared histogram; M8 transposes operands and has no extra16-row padding. Actual compiler/provider and routing remain unbound"}
    tile_source_forms = []
    for signature, fragments in (
        ("auto fetch_to_shared =", ("if(pred)", "row<block_num_valid_tokens", "cp_async4_pred(",
            "sorted_row*a_gl_stride+a_gl_rd_col+a_gl_rd_delta_o*a_off")),
        ("auto thread_block_reduce =", ("threads/b_sh_stride_threads/2", "for(inti=red_off;i>0;i/=2)",
            "if(i<red_off)", "c_rd[k]+c_wr[k]", "if(red_idx==0)", "+=c_rd[j]",
            "j+=(m_block_size_8?2:1)")),
    ):
        form = _cpp_anchor(root, "csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h", signature)
        program = re.sub(r"\s+", "", form["source"])
        if any(fragment not in program for fragment in fragments):
            raise ValueError("Marlin conditional A-load/shared-reduction source changed")
        if signature == "auto thread_block_reduce =" and "block_num_valid_tokens" in program:
            raise ValueError("Marlin shared reduction acquired a live-row predicate")
        tile_source_forms.append(form)
    if any(fragment not in re.sub(r"\s+", "", template_text) for fragment in (
            "a_gl_stride=prob_k/(is_a_8bit?16:8)", "b_sh_stride_threads=b_sh_stride/b_thread_vecs",
            "b_type.size_bits()==4?1:2", "thread_block_reduce();")):
        raise ValueError("Marlin conditional BF16 tile geometry changed")
    anchors["marlin-moe-source"]["conditional_tile_work"] = {"definitions": tile_source_forms,
        "scope": "all-admissible-configuration source A loads and padded shared FP32 addition sites; no configuration selection, point price or physical DRAM assertion"}
    dequant_relative = "csrc/libtorch_stable/quantization/marlin/dequant.h"
    if "constexprboolhas_zp=b_type==vllm::kU4||b_type==vllm::kU8;" not in re.sub(r"\s+", "", template_text):
        raise ValueError("Marlin E2M1 zero-point predicate changed")
    flags = re.search(r"constexpr bool dequant_skip_flop\s*=([^;]+);", template_text)
    if flags is None:
        raise ValueError("Marlin dequant skip predicate is missing")
    flag_expression = re.sub(r"\s+", "", flags.group(1))
    substitutions = {
        "std::is_same<scalar_t,nv_bfloat16>::value": "True",
        "b_type==vllm::kFE4M3fn": "False", "s_type==vllm::kFE8M0fnu": "True",
        "b_type==vllm::kFE2M1f": "True", "s_type==vllm::kFE4M3fn": "False",
        "b_type==vllm::kU8": "False", "is_a_8bit": "False", "has_zp": "False"}
    for expression, value in substitutions.items():
        flag_expression = flag_expression.replace(expression, value)
    flag_expression = flag_expression.replace("||", " or ").replace("&&", " and ").replace("!", " not ").strip()
    for zero_point_float in (False, True):
        flags_ast = ast.parse(flag_expression.replace("is_zp_float", str(zero_point_float)), mode="eval")
        if (any(not isinstance(node, (ast.Expression, ast.Constant, ast.BoolOp, ast.UnaryOp, ast.Or, ast.And, ast.Not))
                for node in ast.walk(flags_ast))
                or eval(compile(flags_ast, "Marlin source dequant predicate", "eval"), {"__builtins__": {}}) is not False):
            raise ValueError("Marlin BF16/E2M1/E8M0 source no longer takes bias-multiply dequant")
    dequant_forms = []
    for signature, fragments in (
        ("__device__ inline void dequant<nv_bfloat162, vllm::kFE2M1f.id(), true>(", (
            "BF16_EXPONENT=8", "FP4_EXPONENT=2", "MASK=0x70007000", "q<<=4",
            "frag_b[1]=*reinterpret_cast<constnv_bfloat162*>(&Out1)",
            "frag_b[0]=*reinterpret_cast<constnv_bfloat162*>(&Out2)")),
        ("__device__ inline void dequant<nv_bfloat162, vllm::kFE2M1f.id(), false>(", (
            "dequant<nv_bfloat162,vllm::kFE2M1f.id(),true>(q,frag_b)",
            "(1<<(BF16_EXPONENT-1))-(1<<(FP4_EXPONENT-1))",
            "frag_b[1]=__hmul2(frag_b[1],bias_reg)", "frag_b[0]=__hmul2(frag_b[0],bias_reg)")),
        ("__device__ inline void dequant_fp8_scales<nv_bfloat162, vllm::kFE8M0fnu.id()>(", (
            "Out1=(q&0xFF00FF00)>>1", "q<<=7", "Out2=q&0x7F807F80",
            "frag_b[1]=*reinterpret_cast<constnv_bfloat162*>(&Out1)",
            "frag_b[0]=*reinterpret_cast<constnv_bfloat162*>(&Out2)")),
    ):
        form = _cpp_anchor(root, dequant_relative, signature)
        if any(fragment not in re.sub(r"\s+", "", form["source"]) for fragment in fragments):
            raise ValueError("Marlin typed E2M1/E8M0 register dequant source changed")
        dequant_forms.append(form)
    scale_form = _cpp_anchor(root, "csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h", "__device__ inline void scale(")
    if (scale_form["source"].count("__hmul2(") != 2
            or compact_matmul.count("dequant_fp8_scales<c_scalar_t2,s_type_id>") != 2
            or compact_matmul.count("dequant_data(b_quant_") != 2
            or compact_matmul.count("scale<a_type_id>") != 2):
        raise ValueError("Marlin dequant per-matmul scale/multiply sites changed")
    anchors["marlin-moe-source"]["register_dequant"] = {
        "definitions": dequant_forms, "multiply": scale_form,
        "scope": "thread-local BF16 source program and scale-byte domains; actual checkpoint values, BF16 intrinsic lowering and packed logical-coordinate mapping remain unbound"}
    repack_relative = "csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu"
    repack_forms = [_cpp_anchor(root, repack_relative, signature) for signature in
                    ("auto fetch_to_shared =", "auto repack_tile =", "torch::stable::Tensor gptq_marlin_repack(")]
    repack_compact = re.sub(r"\s+", "", repack_forms[1]["source"])
    if any(fragment not in repack_compact for fragment in (
            "inttc_row=(th_id%4)*(is_a_8bit?4:2)", "tc_offsets[4]={0,1,8,9}",
            "intpack_idx[8]={0,2,4,6,1,3,5,7}", "out_ptr[out_offset+th_id*4+warp_id]=res")):
        raise ValueError("Marlin checkpoint weight repack scalar mapping changed")
    repack_wrapper = re.sub(r"\s+", "", repack_forms[2]["source"])
    if ("b_q_weight.scalar_type()==torch::headeronly::ScalarType::Int" not in repack_wrapper
            or "{size_k/marlin::tile_size,size_n*marlin::tile_size/pack_factor}" not in repack_wrapper):
        raise ValueError("Marlin prepared weight word dtype/shape changed")
    common_header = re.sub(r"\s+", "", (root / "csrc/libtorch_stable/quantization/marlin/marlin.cuh").read_text())
    if any(fragment not in common_header for fragment in (
            "staticconstexprinttile_size=16;", "staticconstexprinttile_k_size=tile_size;",
            "staticconstexprinttile_n_size=tile_k_size*4;")):
        raise ValueError("Marlin checkpoint repack tile constants changed")
    repack_host_relative = "vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py"
    repack_host = _definition(ast.parse((root / repack_host_relative).read_text()), "_repack_marlin_experts")
    repack_host_body = re.sub(r"\s+", "", ast.unparse(repack_host))
    if any(fragment not in repack_host_body for fragment in (
            "qweight=weight[i].view(torch.int32).T.contiguous()", "out[i]=marlin_qweight",
            "num_experts=weight.shape[0]", "num_bits=4", "is_a_8bit=is_a_8bit",
            "(num_experts,*marlin_qweight.shape)", "dtype=marlin_qweight.dtype")):
        raise ValueError("Marlin expert host repack/view/copy program changed")
    anchors["marlin-moe-source"]["checkpoint_repack"] = {
        "CUDA_source": repack_forms,
        "host_definitions": [_function_anchor(root, relative, symbol) for relative, symbol in (
            (repack_host_relative, "_repack_marlin_experts"),
            (repack_host_relative, "prepare_moe_mxfp4_layer_for_marlin"),
            (repack_host_relative, "mxfp4_marlin_process_scales"),
            ("vllm/model_executor/layers/quantization/utils/marlin_utils.py", "get_scale_perms"),
            ("vllm/model_executor/layers/quantization/utils/marlin_utils.py", "marlin_permute_scales"))],
        "scope": "load-time UINT8 checkpoint to INT32 prepared storage, weight/scale index programs; subsequent MMA reader and actual checkpoint values/runtime unbound"}
    anchors["marlin-moe-source"]["weight_operand_layout"] = None
    if flashinfer_source_root is not None:
        traits_relative = "data/cutlass/include/cute/atom/mma_traits_sm80.hpp"
        layout_base = _cpp_anchor(Path(flashinfer_source_root), traits_relative,
                                 "struct MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>")
        layout_bf16 = _cpp_anchor(Path(flashinfer_source_root), traits_relative,
                                 "struct MMA_Traits<SM80_16x8x16_F32BF16BF16F32_TN>")
        compact_layout = re.sub(r"\s+", "", layout_base["source"])
        for fragment in (
                "ALayout=Layout<Shape<Shape<_4,_8>,Shape<_2,_2,_2>>,Stride<Stride<_32,_1>,Stride<_16,_8,_128>>>",
                "BLayout=Layout<Shape<Shape<_4,_8>,Shape<_2,_2>>,Stride<Stride<_16,_1>,Stride<_8,_64>>>"):
            if fragment not in compact_layout:
                raise ValueError("CUTLASS source BF16 MMA operand layout changed")
        if ("MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>" not in layout_bf16["source"]
                or "using ValTypeA = bfloat16_t;" not in layout_bf16["source"]
                or "using ValTypeB = bfloat16_t;" not in layout_bf16["source"]):
            raise ValueError("CUTLASS BF16 source no longer inherits the checked operand layout")
        transposed_bf16 = mma_sources[1]["source"].split("std::is_same<scalar_t, nv_bfloat16>::value", 1)[1].split("} else if constexpr", 1)[0]
        if '"r"(b[0]),"r"(b2[0]),"r"(b[1]),"r"(b2[1]),"r"(a[0]),"r"(a[1])' not in re.sub(r"\s+", "", transposed_bf16):
            raise ValueError("Marlin M8 weight-A register order changed")
        anchors["marlin-moe-source"]["weight_operand_layout"] = {
            "reference_layouts": [layout_base, layout_bf16],
            "vllm_MMA_register_binding": mma_sources,
            "vllm_accesses": [_cpp_anchor(root, "csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h", signature)
                              for signature in ("auto fetch_to_shared =", "auto fetch_to_registers =")],
            "reference_only_not_target_provider": True,
            "scope": "matching source PTX opcode operand-coordinate contract and packed/scales loads; no target CUTLASS provider, numerical MMA or GPU execution acceptance"}
    marlin_cpp = marlin_template.read_text(encoding="utf-8")
    anchors["marlin-moe-source"]["weight_multiply_source"] = []
    for fragment in ("Cdtype::num2num2(Cdtype::float2num(topk_weight_tmp))",
                     "sh_red_half2[a] = __hmul2(sh_red_half2[a], topk_weight_score)"):
        offset = marlin_cpp.index(fragment)
        anchors["marlin-moe-source"]["weight_multiply_source"].append({
            "path": str(marlin_template), "line": marlin_cpp[:offset].count("\n") + 1,
            "expression": fragment, "scope": "source BF16 conversion and packed multiply; not compiler or issue counts"})
    for relative, symbol, fragments in (
        (moe_root + "expert_map_manager.py", "determine_expert_map", ("ifep_size==1:", "return(global_num_experts,None,None)")),
        (moe_root + "activation.py", "silu_and_mul_with_clamp", (
            "eliftopk_idsisnotNoneandexpert_mapisnotNone:",
            "torch.ops._C.silu_and_mul_with_clamp(output,input,clamp_limit,1.0,0.0)")),
        (moe_root + "experts/marlin_moe.py", "_fused_marlin_moe", (
            "intermediate_cache13,(M*num_topk,w13_num_shards*N)",
            "intermediate_cache13,(M*num_topk,K)",
            "mul_topk_weights=notapply_router_weight_on_input", "use_atomic_add=False")),
        (moe_root + "experts/marlin_moe.py", "MarlinExperts.moe_sum", ("ops.moe_sum(input,output)",)),
        ("vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py", "mxfp4_marlin_process_scales", (
            "marlin_scales.to(torch.float8_e8m0fnu)",)),
    ):
        body = re.sub(r"\s+", "", ast.unparse(_definition(ast.parse((root / relative).read_text(encoding="utf-8")), symbol)))
        if any(fragment not in body for fragment in fragments):
            raise ValueError(f"Marlin routed source program changed: {relative}:{symbol}")
    marlin_module = ast.parse((root / (moe_root + "experts/marlin_moe.py")).read_text(encoding="utf-8"))
    buffer_definitions = (
        (moe_root + "experts/marlin_moe.py", "MarlinExperts.workspace_shapes"),
        (moe_root + "experts/marlin_moe.py", "MarlinExperts.apply"),
        (moe_root + "modular_kernel.py", "FusedMoEExpertsModular.workspace_dtype"),
        (moe_root + "modular_kernel.py", "FusedMoEKernelModularImpl._allocate_buffers"),
        (moe_root + "modular_kernel.py", "FusedMoEKernelModularImpl._fused_experts"),
        (moe_root + "modular_kernel.py", "FusedMoEKernelModularImpl.apply"),
        (moe_root + "topk_weight_and_reduce.py", "TopKWeightAndReduceNoOP.apply"),
        (moe_root + "utils.py", "_resize_cache"),
        ("vllm/v1/worker/workspace.py", "WorkspaceManager.get_simultaneous"),
    )
    for relative, symbol, fragments in (
        (moe_root + "experts/marlin_moe.py", "MarlinExperts.workspace_shapes", (
            "workspace1=(M*topk,max(N,K))", "workspace2=(M*topk*max(2*N,K),)", "output=(M,K)")),
        (moe_root + "experts/marlin_moe.py", "MarlinExperts.apply", (
            "intermediate_cache13=workspace2,intermediate_cache2=workspace13", "output=output")),
        (moe_root + "modular_kernel.py", "FusedMoEExpertsModular.workspace_dtype", ("returnact_dtype",)),
        (moe_root + "modular_kernel.py", "FusedMoEKernelModularImpl._allocate_buffers", (
            "max_shape_size=max(prod(workspace13_shape),prod(fused_out_shape))",
            "current_workspace_manager().get_simultaneous(((max_shape_size,),workspace_dtype),(workspace2_shape,workspace_dtype))",
            "workspace13=_resize_cache(common_workspace,workspace13_shape)", "fused_out=_resize_cache(common_workspace,fused_out_shape)")),
        (moe_root + "modular_kernel.py", "FusedMoEKernelModularImpl._fused_experts", (
            "output_alias.shape==fused_out.shape", "output_alias.dtype==fused_out.dtype", "output_alias.device==fused_out.device",
            "output_alias.is_contiguous()", "elifuse_output_alias:fused_out=output_alias", "returnfused_out")),
        (moe_root + "modular_kernel.py", "FusedMoEKernelModularImpl.apply", (
            "output=torch.empty_like(hidden_states)", "output_alias=output")),
        (moe_root + "topk_weight_and_reduce.py", "TopKWeightAndReduceNoOP.apply", ("ifoutputisfused_expert_output:returnoutput",)),
        (moe_root + "utils.py", "_resize_cache", ("returnx.flatten()[:prod(v)].view(*v)",)),
        ("vllm/v1/worker/workspace.py", "WorkspaceManager.get_simultaneous", (
            "aligned_bytes=[round_up(actual,256)foractualinactual_bytes]", "offsets=list(accumulate([0]+aligned_bytes[:-1]))",
            "current_workspace[offsets[i]:offsets[i]+actual_bytes[i]].view(shapes_and_dtypes[i][1]).reshape(shapes_and_dtypes[i][0])")),
    ):
        body = re.sub(r"\s+", "", ast.unparse(_definition(ast.parse((root / relative).read_text()), symbol)))
        if any(fragment not in body for fragment in fragments):
            raise ValueError("Marlin modular buffer/alias program changed: " + symbol)
    anchors["marlin-moe-source"]["modular_buffers"] = {
        "definitions": [_function_anchor(root, relative, symbol) for relative, symbol in buffer_definitions],
        "scope": "conditional contiguous Standard BF16 modular program; requested arena views and output identity, not actual lane capacity, GPU fence or GEMM FP32 scratch"}
    frame_definitions = []
    for relative, symbol, fragments in (
        (moe_root + "prepare_finalize/no_dp_ep.py", "MoEPrepareAndFinalizeNoDPEPModular.prepare", (
            "(a1q,a1q_scale)=_quantize_input(a1,quant_config,defer_input_quant)", "return(a1q,a1q_scale,None,None,None)")),
        (moe_root + "prepare_finalize/no_dp_ep.py", "MoEPrepareAndFinalizeNoDPEPModular.finalize", (
            "weight_and_reduce_impl.apply(output=output,fused_expert_output=fused_expert_output,")),
        (moe_root + "prepare_finalize/no_dp_ep.py", "_quantize_input", (
            "ifdefer_input_quant:return(a1,None)", "moe_kernel_quantize_input(a1,input_sf,quant_dtype=quant_config.quant_dtype,")),
        (moe_root + "modular_kernel.py", "FusedMoEKernelModularImpl._prepare", (
            "topk_ids=topk_idsif_expert_topk_idsisNoneelse_expert_topk_ids",
            "topk_weights=topk_weightsif_expert_topk_weightsisNoneelse_expert_topk_weights")),
        (moe_root + "modular_kernel.py", "FusedMoEKernelModularImpl._finalize", (
            "self.prepare_finalize.finalize(output,fused_out,topk_weights,topk_ids,apply_router_weight_on_input,", "returnoutput")),
        (moe_root + "experts/marlin_moe.py", "MarlinExperts.finalize_weight_and_reduce_impl", (
            "returnTopKWeightAndReduceNoOP()",)),
        (moe_root + "moe_align_block_size.py", "moe_align_block_size", (
            "sorted_ids=torch.empty((max_num_tokens_padded,),dtype=torch.int32,device=topk_ids.device)",
            "expert_ids=torch.empty((max_num_m_blocks,),dtype=torch.int32,device=topk_ids.device)",
            "num_tokens_post_pad=torch.empty(1,dtype=torch.int32,device=topk_ids.device)")),
    ):
        tree = ast.parse((root / relative).read_text())
        definition = _definition(tree, symbol)
        compact = re.sub(r"\s+", "", ast.unparse(definition))
        if any(fragment not in compact for fragment in fragments):
            raise ValueError("Marlin modular prepare/finalize source identity changed: " + symbol)
        frame_definitions.append(_function_anchor(root, relative, symbol))
    quant_relative = moe_root + "utils.py"
    quant_input = _definition(ast.parse((root / quant_relative).read_text()), "moe_kernel_quantize_input")
    tail = quant_input.body[-1]
    while isinstance(tail, ast.If) and len(tail.orelse) == 1 and isinstance(tail.orelse[0], ast.If):
        tail = tail.orelse[0]
    if (not isinstance(tail, ast.If) or len(tail.orelse) != 1
            or ast.unparse(tail.orelse[0]) != "return (A, A_scale)"):
        raise ValueError("Marlin unquantized prepare input no longer aliases its source tensor")
    frame_definitions.append(_function_anchor(root, quant_relative, "moe_kernel_quantize_input"))
    cpp_frame = _cpp_anchor(root, "csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu",
                            "torch::stable::Tensor moe_wna16_marlin_gemm(")
    cpp_compact = re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", cpp_frame["source"]))
    allocations = re.findall(r"(\w+)\s*=\s*torch::stable::(new_\w+)\s*\(", cpp_frame["source"])
    if allocations != [(name, "new_empty") for name in
                       ("a_scales", "c", "c_tmp", "c_tmp", "global_scale", "b_bias", "b_zeros")]:
        raise ValueError("Marlin matrix source allocation inventory changed")
    for fragment in ("if(c_or_none.has_value()){c=c_or_none.value();", "returnc;",
        "a_scales=torch::stable::new_empty(a,{0},kFloat)", "global_scale=torch::stable::new_empty(a,{0},kFloat)",
        "b_bias=torch::stable::new_empty(a,{0},c_dtype)", "b_zeros=torch::stable::new_empty(a,{0},c_dtype)"):
        if fragment not in cpp_compact:
            raise ValueError("Marlin matrix return/zero-extent object source changed")
    alignment_frame = _cpp_anchor(root, "csrc/libtorch_stable/moe/moe_align_sum_kernels.cu", "void moe_align_block_size(")
    aligned = re.sub(r"\s+", "", alignment_frame["source"])
    for fragment in ("expert_map=torch::stable::new_empty(topk_ids,{0},torch::headeronly::ScalarType::Int)",
        "cumsum_buffer=torch::stable::new_empty(topk_ids,{num_experts+1},torch::headeronly::ScalarType::Int)"):
        if fragment not in aligned:
            raise ValueError("Marlin alignment temporary source inventory changed")
    anchors["marlin-moe-source"]["tensor_frame"] = {"definitions": frame_definitions + [cpp_frame, alignment_frame],
        "scope": "conditional synchronous no-DP/EP BF16 caller identity and source temporary inventory; not selected scratch accesses, lane capacity, provider or GPU fences"}
    inner = _definition(marlin_module, "_fused_marlin_moe")
    gemms = sorted((node for node in ast.walk(inner) if isinstance(node, ast.Call)
                    and ast.unparse(node.func) == "ops.moe_wna16_marlin_gemm"), key=lambda node: node.lineno)
    activations = [node for node in ast.walk(inner) if isinstance(node, ast.Call)
                   and ast.unparse(node.func) in ("apply_moe_activation", "activation_func")]
    outer = _definition(marlin_module, "fused_marlin_moe")
    core = [node for node in ast.walk(outer) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "_fused_marlin_moe"]
    reduce_calls = [node for node in ast.walk(outer) if isinstance(node, ast.Call)
                    and ast.unparse(node.func) == "moe_sum"]
    if (len(gemms) != 2 or len(activations) != 2
            or not all(gemms[0].lineno < node.lineno < gemms[1].lineno for node in activations)
            or len(core) != 1 or len(reduce_calls) != 1 or core[0].lineno >= reduce_calls[0].lineno):
        raise ValueError("Marlin W13 -> activation -> W2 -> moe_sum source ordering changed")
    for gemm in gemms:
        options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in gemm.keywords
                   if keyword.arg in ("use_fp32_reduce", "use_atomic_add")}
        if options != {"use_fp32_reduce": True, "use_atomic_add": False}:
            raise ValueError("Marlin source FP32 reduction/scratch condition changed")
        if any(keyword.arg in ("thread_k", "thread_n", "blocks_per_sm") for keyword in gemm.keywords):
            raise ValueError("Marlin source no longer uses default auto thread configuration")
    ops_relative = "csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu"
    scratch_request = _cpp_anchor(root, ops_relative, "if (use_fp32_reduce && !use_atomic_add)")
    scratch_body = re.sub(r"\s+", "", scratch_request["source"])
    for expression in ("(long)size_n*sorted_token_ids.size(0)",
                       "(long)sms*4*moe_block_size*MARLIN_NAMESPACE_NAME::max_thread_n",
                       "if(moe_block_size==8)max_c_tmp_size*=2;",
                       "c_tmp=torch::stable::new_empty(a,{max_c_tmp_size},kFloat)"):
        if expression not in scratch_body:
            raise ValueError("Marlin FP32 scratch source allocation changed")
    ops_text = (root / ops_relative).read_text()
    thread_candidates = {}
    for family in ("small", "large"):
        declaration = re.search(r"thread_config_t " + family + r"_batch_thread_configs\[\]\s*=\s*\{(.*?)\};", ops_text, re.S)
        if declaration is None:
            raise ValueError("Marlin auto thread configuration source is missing")
        tuples = [list(map(int, match)) for match in re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}", declaration.group(1))]
        if tuples != {"small": [[128, 128, 256], [64, 128, 128], [128, 64, 128]],
                      "large": [[64, 256, 256], [64, 128, 128], [128, 64, 128]]}[family]:
            raise ValueError("Marlin source thread candidate set changed")
        thread_candidates[family] = tuples
    header = root / "csrc/libtorch_stable/quantization/marlin/marlin.cuh"
    constant = re.search(r"static constexpr int max_thread_n\s*=\s*(\d+)\s*;", header.read_text())
    if constant is None or int(constant.group(1)) != 256:
        raise ValueError("Marlin scratch max_thread_n source constant changed")
    reduction = _cpp_anchor(root, "csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h",
                            "auto global_reduce_fp32 =")
    reduce_body = re.sub(r"\s+", "", reduction["source"])
    for expression in ("if(!first)", "if(!last)", "float*frag_c_ptr", "C_tmp[c_cur_offset+active_threads*k+threadIdx.x]"):
        if expression not in reduce_body:
            raise ValueError("Marlin conditional FP32 scratch read/write program changed")
    if "if(slice_count>1&&!use_atomic_add)" not in re.sub(r"\s+", "", marlin_cpp):
        raise ValueError("Marlin scratch no longer requires a multi-block slice")
    compact_kernel = re.sub(r"\s+", "", marlin_cpp)
    for fragment in ("intglobal_mn_tiles=parallel*n_tiles;", "part2_mn_tiles=global_mn_tiles%gridDim.x;",
                     "if(part2_mn_tiles*3<=gridDim.x)part2_mn_tiles+=gridDim.x;",
                     "intiters=div_ceil(k_tiles*part2_mn_tiles,gridDim.x);",
                     "global_reduce_fp32(slice_idx==0,last);", "slice_row=0;",
                     "ifconstexpr(group_blocks>=thread_k_blocks)"):
        if fragment not in compact_kernel:
            raise ValueError("Marlin conditional scratch partition program changed")
    compact_host = re.sub(r"\s+", "", ops_text)
    for fragment in ("allow_count=max(min(allow_count,4),1);", "allow_count=max(min(allow_count,2),1);",
                     "max(prob_n/th_config.thread_n*prob_m*top_k*4/sms,1)", "cudaFuncGetAttributes(&attr,kernel)"):
        if fragment not in compact_host:
            raise ValueError("Marlin auto configuration superset conditions changed")
    anchors["marlin-moe-source"]["fp32_scratch"] = {"allocation": scratch_request,
        "reduction": reduction, "constant": {"path": str(header), "line": header.read_text()[:constant.start()].count("\n") + 1,
            "max_thread_n": int(constant.group(1))},
        "sm_count_source": "cudaDevAttrMultiProcessorCount; static scenario uses declared H200132 SM specification",
        "thread_config_candidates": thread_candidates,
        "partition": _cpp_anchor(root, "csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h", "auto init_part2_slice ="),
        "auto_config": _cpp_anchor(root, ops_relative, "exec_config_t determine_exec_config("),
        "scope": "source extent, dtype and conditional accesses, not allocator capacity or accessed-byte totals"}
    workspace_calls = [node for node in ast.walk(inner) if isinstance(node, ast.Call)
                       and ast.unparse(node.func) == "marlin_make_workspace_new"]
    if (len(workspace_calls) != 1 or workspace_calls[0].lineno >= gemms[0].lineno
            or len(workspace_calls[0].args) != 2 or ast.literal_eval(workspace_calls[0].args[1]) != 4
            or workspace_calls[0].keywords):
        raise ValueError("Marlin default workspace initialization changed")
    apply = _definition(marlin_module, "MarlinExperts.apply")
    no_lora = [node for node in apply.body if isinstance(node, ast.If) and ast.unparse(node.test) == "ctx is None"]
    calls = [node for branch in no_lora for node in ast.walk(branch) if isinstance(node, ast.Call)
             and ast.unparse(node.func) == "fused_marlin_moe"]
    if len(calls) != 1 or any(keyword.arg == "workspace" for keyword in calls[0].keywords):
        raise ValueError("Marlin no-LoRA caller now binds a nondefault workspace")
    for function in (outer, inner):
        defaults = {argument.arg: value for argument, value in zip(function.args.args[-len(function.args.defaults):], function.args.defaults)}
        if "workspace" not in defaults or ast.literal_eval(defaults["workspace"]) is not None:
            raise ValueError("Marlin workspace default changed")
    workspace_relative = "vllm/model_executor/layers/quantization/utils/marlin_utils.py"
    workspace = _function_anchor(root, workspace_relative, "marlin_make_workspace_new")
    workspace_definition = _definition(ast.parse((root / workspace_relative).read_text()), "marlin_make_workspace_new")
    workspace_body = re.sub(r"\s+", "", ast.unparse(workspace_definition))
    if any(fragment not in workspace_body for fragment in (
            "existing:torch.Tensor|None=None", "sms=num_compute_units(device.index)", "size=sms*max_blocks_per_sm",
            "returntorch.zeros(size,dtype=torch.int,device=device,requires_grad=False)")):
        raise ValueError("Marlin default workspace is no longer a zeroed INT32 SM-count vector")
    anchors["marlin-moe-source"]["workspace_zero"] = {"definition": workspace,
        "inner_call_line": workspace_calls[0].lineno, "default_no_lora_workspace": None,
        "scope": "one default INT32 zero initialization before W13, shared by W13/W2; no lock/access or allocator proof"}
    for gemm, expected_c, expected_topk, expected_rows in zip(gemms, ("intermediate_cache1", "output"), ("num_topk", "1"), ("M", "M * num_topk")):
        if ast.unparse(gemm.args[1]) != expected_c:
            raise ValueError("Marlin GEMM no longer receives its supplied output view")
        options = {keyword.arg: ast.unparse(keyword.value) for keyword in gemm.keywords}
        if options["top_k"] != expected_topk or options["size_m"] != expected_rows:
            raise ValueError("Marlin W13/W2 input row divisor or row domain changed")
        if [ast.unparse(arg) for arg in gemm.args[9:12]] != ["sorted_token_ids", "expert_ids", "num_tokens_post_padded"]:
            raise ValueError("Marlin GEMMs no longer consume the same routing metadata")
    metadata_expressions = (
        "int num_tokens_past_padded = num_tokens_past_padded_ptr[0];",
        "int parallel = num_tokens_past_padded / moe_block_size;",
        "if (par_id >= parallel) return;", "expert_id = expert_ids_ptr[block_id];",
        "(block_id * moe_block_size / 4 + threadIdx.x)", "threadIdx.x < moe_block_size / 4",
        "if (idx < prob_m_top_k) local_count++;",
        "sh_rd_block_sorted_ids[threadIdx.x] = idx / top_k;",
        "row < block_num_valid_tokens", "int64_t sorted_row = sh_block_sorted_ids[row];",
        "int64_t true_idx = sorted_row * c_gl_stride + c_gl_wr % c_gl_stride;")
    metadata_anchors = []
    for expression in metadata_expressions:
        if expression not in marlin_cpp:
            raise ValueError("Marlin active-prefix metadata read/row mapping source changed")
        offset = marlin_cpp.index(expression)
        metadata_anchors.append({"path": str(marlin_template), "line": marlin_cpp[:offset].count("\n") + 1,
                                 "expression": expression})
    anchors["marlin-moe-source"]["metadata_consumers"] = {
        "expressions": metadata_anchors, "gemm_lines": [node.lineno for node in gemms],
        "scope": "INT32 active-prefix unique input payload and valid-row mapping; no CTA/split replay or checkpoint routing binding"}
    anchors["marlin-moe-source"]["serial_stage_order"] = {
        "inner_gemm_lines": [node.lineno for node in gemms],
        "inner_activation_lines": sorted(node.lineno for node in activations),
        "outer_core_line": core[0].lineno, "outer_reduce_line": reduce_calls[0].lineno,
        "not_actual_launch_count": True}
    anchors["marlin-moe-source"]["alignment_python"] = _function_anchor(
        root, moe_root + "moe_align_block_size.py", "moe_align_block_size")
    align_relative = "csrc/libtorch_stable/moe/moe_align_sum_kernels.cu"
    anchors["marlin-moe-source"]["alignment_cuda"] = [_cpp_anchor(root, align_relative, signature) for signature in (
        "__device__ __forceinline__ int get_local_expert_id(",
        "__device__ void _moe_align_block_size(", "__device__ void _count_and_sort_expert_tokens(",
        "__device__ void _moe_align_block_size_small_batch_expert(", "void moe_align_block_size(")]
    for index, fragments in (
        (1, ("sorted_token_ids[sorted_token_ids_offset+it]=numel;",
             "cumsum[cumsum_offset+expert_id]=cumsum_val;",
             "expert_ids[expert_ids_offset+i]=inactive_expert_id;")),
        (2, ("atomicAdd(&cumsum_buffer[(model_offset*(num_experts+1))+expert_id],1)",
             "sorted_token_ids[max_num_tokens_padded*model_offset+rank_post_pad]=i;")),
        (3, ("sorted_token_ids[sorted_token_ids_offset+it]=numel;",
             "sorted_token_ids[sorted_token_ids_offset+rank_post_pad]=i;")),
        (4, ("(topk_ids.numel()<1024)&&(num_experts<=64)",
             "{num_experts+1}", "padded_num_experts<1024", "align_kernel<<<2,threads,"))):
        compact = re.sub(r"\s+", "", anchors["marlin-moe-source"]["alignment_cuda"][index]["source"])
        if any(fragment not in compact for fragment in fragments):
            raise ValueError("Marlin alignment dispatch or initialization/sort source program changed")
    align_python = re.sub(r"\s+", "", ast.unparse(_definition(
        ast.parse((root / (moe_root + "moe_align_block_size.py")).read_text()), "moe_align_block_size")))
    for fragment in ("max_num_tokens_padded=topk_ids.numel()+num_experts*(block_size-1)",
                     "iftopk_ids.numel()<num_experts:", "topk_ids.numel()*block_size", "dtype=torch.int32"):
        if fragment not in align_python:
            raise ValueError("Marlin alignment source allocation geometry changed")
    host_source = re.sub(r"\s+", "", ast.unparse(outer))
    if ("forblock_size_min[8,16,32,48,64]:" not in host_source
            or "M*topk/E/block_size_m<0.9" not in host_source
            or "ignore_invalid_experts=True" not in host_source):
        raise ValueError("Marlin source M-block/valid expert branch changed")
    anchors["final-rmsnorm"]["related_definitions"] = [
        _function_anchor(root, relative, symbol) for relative, symbol in (
            ("vllm/model_executor/layers/layernorm.py", "RMSNorm.forward_cuda"),
            ("vllm/model_executor/layers/layernorm.py", "RMSNorm.forward_native"),
            ("vllm/kernels/vllm_c.py", "rms_norm"))]
    anchors["final-rmsnorm"]["related_definitions"].append(_cpp_anchor(root,
        "csrc/libtorch_stable/layernorm_kernels.cu", "__global__ void rms_norm_kernel("))
    for relative, symbol, fragments in (
        ("vllm/ir/ops/layernorm.py", "rms_norm", ("x=x.to(torch.float32)",
            "x_var=xifvariance_sizeisNoneelsex[...,:variance_size]", "variance=x_var.pow(2).mean(dim=-1,keepdim=True)",
            "x=x*torch.rsqrt(variance+epsilon)", "x=x.to(weight.dtype)*weight", "returnx.to(orig_dtype)")),
        ("vllm/platforms/cuda.py", "CudaPlatformBase.get_default_ir_op_priority", (
            "default=['native']ifusing_inductorelse['vllm_c','native']",)),
    ):
        # The CUDA class is named CudaPlatformBase in this source snapshot.
        definition = _definition(ast.parse((root / relative).read_text()), symbol)
        program = re.sub(r"\s+", "", ast.unparse(definition))
        if any(fragment not in program for fragment in fragments):
            raise ValueError("final norm source object/dispatch program changed")
        anchors["final-rmsnorm"]["related_definitions"].append(_function_anchor(root, relative, symbol))
    oink_tree = ast.parse((root / "vllm/kernels/oink_ops.py").read_text())
    oink_gate = [node.value for node in oink_tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "OINK_AVAILABLE" for target in node.targets)]
    if len(oink_gate) != 1 or "current_platform.has_device_capability(100)" not in ast.unparse(oink_gate[0]):
        raise ValueError("Oink RMSNorm is no longer excluded on the declared H200 SM90")
    norm_entry = _cpp_anchor(root, "csrc/libtorch_stable/layernorm_kernels.cu", "void rms_norm(")
    entry_program = re.sub(r"\s+", "", norm_entry["source"])
    if "if(input.stride(-1)!=1){input=torch::stable::contiguous(input);}" not in entry_program:
        raise ValueError("CUDA RMSNorm input-copy guard changed")
    anchors["final-rmsnorm"]["related_definitions"].append(norm_entry)
    batch_norm_relative = "vllm/model_executor/determinism/batch_invariant.py"
    for symbol, fragments in (
        ("rms_norm_batch_invariant", ("weight=weight.contiguous()", "input_2d=input_2d.contiguous()",
            "output=torch.empty_like(input_2d)", "BLOCK_SIZE=1024", "returnoutput.reshape(original_shape)")),
        ("_rms_norm_kernel", ("forcol_offsetinrange(0,n_cols,BLOCK_SIZE)",
            "sum_sq+=tl.sum(tl.where(mask,sq_vals,0.0))", "rms=tl.sqrt(mean_sq+eps)", "inv_rms=1.0/rms",
            "output_f32=output_f32*weight.to(tl.float32)")),
    ):
        definition = _definition(ast.parse((root / batch_norm_relative).read_text()), symbol)
        program = re.sub(r"\s+", "", ast.unparse(definition))
        if any(fragment not in program for fragment in fragments):
            raise ValueError("batch-invariant RMSNorm source math/storage program changed")
        if symbol == "_rms_norm_kernel" and program.count("vals=tl.load(row_start_ptr+col_idx,mask=mask,other=0.0)") != 2:
            raise ValueError("batch-invariant RMSNorm input no longer has two source reads")
        anchors["final-rmsnorm"]["related_definitions"].append(_function_anchor(root, batch_norm_relative, symbol))
    anchors["mhc-source"] = {"definitions": [
        _function_anchor(root, "vllm/model_executor/kernels/mhc/" + filename, symbol)
        for filename, symbol in (
            ("tilelang.py", "_hc_prenorm_gemm_outputs"),
            ("tilelang.py", "mhc_pre_broadcast_tilelang"),
            ("tilelang.py", "mhc_fused_post_pre_tilelang"),
            ("tilelang_kernels.py", "compute_num_split"),
            ("tilelang_kernels.py", "mhc_fused_post_pre_split_config"),
            ("tilelang_kernels.py", "mhc_pre_big_fuse_with_norm_tilelang"),
            ("tilelang_kernels.py", "mhc_pre_big_fuse_broadcast_with_norm_tilelang"),
            ("tilelang_kernels.py", "mhc_fused_tilelang"),
            ("tilelang_kernels.py", "mhc_post_tilelang"),
            ("tilelang_kernels.py", "hc_prenorm_gemm_tilelang"),
            ("tilelang_kernels.py", "hc_prenorm_gemm_block_m_tilelang"),
            ("tilelang_kernels.py", "HcPrenormGemmTileLangKernel.dispatch"),
            ("tilelang_kernels.py", "hc_head_fuse_tilelang"))],
        "backend_support": _function_anchor(root, "vllm/utils/deep_gemm.py", "is_deep_gemm_supported"),
        "forced_broadcast_callee": _function_anchor(root, "vllm/utils/deep_gemm.py", "tf32_hc_prenorm_gemm"),
        "hardware_split_bound": "H200132 SM: max(min(132//ceil(T/64),ceil(H/64)//4),1); configs/hardware/nvidia_h200.yaml SM specification",
        "scope": "source logical MAC/scalar groups and unique operands; not native/provider binding"}
    mhc_kernel_relative = "vllm/model_executor/kernels/mhc/tilelang_kernels.py"
    post_initial_multiply = []
    for symbol, fragments in (
        ("mhc_post_tilelang", ("c_local=T.alloc_fragment(hc,T.float32)",
            "d_local=T.alloc_fragment(h_blk,T.float32)", "x_local=T.alloc_fragment((hc,h_blk),T.float32)",
            "x_local[i_hco,i1_h]=c_local[i_hco]*d_local[i1_h]",
            "x_local[i_hco,i1_h]+=a_local[i_hci,i_hco]*b_local[i_hci,i1_h]")),
        ("mhc_fused_tilelang", ("pm=T.alloc_local((hc,),T.float32)",
            "new_r=T.alloc_local((hc,),T.float32)", "n_tiles=n_out//tile_n",
            "new_r[j]=pm[j]*x_in[i_n,h_idx]", "new_r[j]+=cm[k,j]*residual_in[i_n,k,h_idx]",
            "acc[n]+=weight_t[i_nt*tile_n+n,j,h_idx]*new_r[j]")),
    ):
        definition = _definition(ast.parse((root / mhc_kernel_relative).read_text()), symbol)
        body = re.sub(r"\s+", "", ast.unparse(definition))
        if any(fragment not in body for fragment in fragments):
            raise ValueError("mHC post FP32 initial multiply or projection repetition changed: " + symbol)
        post_initial_multiply.append({"path": str(root / mhc_kernel_relative), "symbol": symbol,
            "line": definition.lineno, "checked_expressions": list(fragments)})
    anchors["mhc-source"]["post_initial_multiply"] = post_initial_multiply
    scalar_programs = []
    for symbol, fragments in (
        ("mhc_pre_big_fuse_with_norm_tilelang", (
            "rms=T.alloc_fragment(1,T.float32)", "mixes=T.alloc_fragment(hc_mult3,T.float32)",
            "cm=T.alloc_fragment((hc_mult,hc_mult),T.float32)", "T.reduce_max(cm,row_max,dim=1)",
            "cm[j,k]=T.exp(cm[j,k]-row_max[j])", "cm[j,k]=cm[j,k]/row_sum[j]+hc_sinkhorn_eps",
            "cm[j,k]=cm[j,k]/(col_sum[k]+hc_sinkhorn_eps)",
            "T.reduce_sum(sumsq_per_pos,sumsq,dim=0)")),
        ("mhc_pre_big_fuse_broadcast_with_norm_tilelang", (
            "rms=T.alloc_fragment(1,T.float32)", "mixes=T.alloc_fragment(hc_mult3,T.float32)",
            "cm=T.alloc_fragment((hc_mult,hc_mult),T.float32)", "T.reduce_max(cm,row_max,dim=1)",
            "cm[j,k]=T.exp(cm[j,k]-row_max[j])", "cm[j,k]=cm[j,k]/row_sum[j]+hc_sinkhorn_eps",
            "T.reduce_sum(sumsq_per_pos,sumsq,dim=0)")),
        ("hc_head_fuse_tilelang", (
            "rsqrt_val=T.alloc_fragment(1,T.float32)",
            "rsqrt_val[0]=T.rsqrt(sqrsum_r[0]/hc_dim+rms_eps)",
            "T.sigmoid(mixes_r[m]*rsqrt_val[0]*hc_scale[0]+hc_base[m])+hc_eps")),
    ):
        definition = _definition(ast.parse((root / mhc_kernel_relative).read_text()), symbol)
        body = re.sub(r"\s+", "", ast.unparse(definition))
        if any(fragment not in body for fragment in fragments):
            raise ValueError("mHC floating scalar or nonlinear function source changed: " + symbol)
        scalar_programs.append({"path": str(root / mhc_kernel_relative), "symbol": symbol,
            "line": definition.lineno, "checked_expressions": list(fragments)})
    anchors["mhc-source"]["scalar_fp32_math"] = scalar_programs
    decoder = _definition(ast.parse((root / _MODEL).read_text()), "DeepseekV4DecoderLayer.forward")
    state_calls = [node for node in ast.walk(decoder) if isinstance(node, ast.Call)
                   and ast.unparse(node.func) == "mhc_fused_post_pre_tilelang"]
    if len(state_calls) != 2 or any([ast.unparse(arg) for arg in node.args[:4]] != ["x", "residual", "post_mix", "res_mix"] for node in state_calls):
        raise ValueError("decoder mHC stream/coefficient argument bindings changed")
    mhc_relative = "vllm/model_executor/kernels/mhc/tilelang.py"
    mhc_tree = ast.parse((root / mhc_relative).read_text())
    for symbol, expression in (
            ("mhc_pre_broadcast_tilelang", "(residual_out, post_mix.unsqueeze(-1), comb_mix.view(num_tokens, hc_mult, hc_mult), layer_input)"),
            ("mhc_fused_post_pre_tilelang", "(residual_cur.view(*outer_shape, hc_mult, hidden_size), post_mix_cur.view(*outer_shape, hc_mult, 1), comb_mix_cur.view(*outer_shape, hc_mult, hc_mult), layer_input_cur.view(*outer_shape, hidden_size))")):
        returns = [node for node in _definition(mhc_tree, symbol).body if isinstance(node, ast.Return)]
        if len(returns) != 1 or ast.dump(returns[0].value) != ast.dump(ast.parse(expression, mode="eval").body):
            raise ValueError("mHC returned tensor tuple/view geometry changed")
    post = _definition(mhc_tree, "mhc_post_tilelang")
    post_calls = [node for node in ast.walk(post) if isinstance(node, ast.Call) and ast.unparse(node.func) == "_MHC_POST_TILELANG_KERNEL"]
    if len(post_calls) != 1 or [ast.unparse(arg) for arg in post_calls[0].args[:5]] != ["comb_res_mix", "residual", "post_layer_mix.squeeze(-1)", "x", "out"]:
        raise ValueError("final mHC post tensor bindings changed")
    anchors["mhc-state-bindings"] = {"decoder": anchors["decoder-forward"], "model": anchors["model-forward"],
        "wrappers": [_function_anchor(root, mhc_relative, symbol) for symbol in (
            "mhc_pre_broadcast_tilelang", "mhc_fused_post_pre_tilelang", "mhc_post_tilelang", "hc_head_fused_kernel_tilelang")],
        "scope": "source returned/input stream tensor ports and direct dependencies; not complete wrapper parameter/temporary or actual runtime/provider acceptance"}
    model_tree = ast.parse((root / _MODEL).read_text())
    parameter_bindings = []
    for owner, descriptors in (("DeepseekV4DecoderLayer", (
            ("hc_attn_fn", "(mix_hc, hc_dim)"), ("hc_ffn_fn", "(mix_hc, hc_dim)"),
            ("hc_attn_base", "mix_hc"), ("hc_ffn_base", "mix_hc"),
            ("hc_attn_scale", "3"), ("hc_ffn_scale", "3"))),
            ("DeepseekV4Model", (("hc_head_fn", "self.hc_mult, self.hc_dim"),
                                  ("hc_head_base", "self.hc_mult"), ("hc_head_scale", "1")))):
        init = _definition(model_tree, owner + ".__init__")
        for member, shape in descriptors:
            assignments = [node for node in ast.walk(init) if isinstance(node, ast.Assign)
                           and any(ast.unparse(target) == "self." + member for target in node.targets)]
            if len(assignments) != 1:
                raise ValueError("mHC parameter allocation missing or ambiguous")
            allocation = assignments[0].value.args[0]
            if (ast.unparse(assignments[0].value.func) != "nn.Parameter" or ast.unparse(allocation.func) != "torch.empty"
                    or ", ".join(ast.unparse(arg) for arg in allocation.args) != shape
                    or {keyword.arg: ast.unparse(keyword.value) for keyword in allocation.keywords} != {"dtype": "torch.float32"}):
                raise ValueError("mHC FP32 parameter allocation dtype/shape changed")
            parameter_bindings.append(_constructor(root, _MODEL, owner, member, "nn.Parameter"))
    decoder_init = _definition(model_tree, "DeepseekV4DecoderLayer.__init__")
    normalized_init = [ast.unparse(node) for node in decoder_init.body if isinstance(node, ast.Assign)]
    if ("mix_hc = (2 + self.hc_mult) * self.hc_mult" not in normalized_init
            or "hc_dim = self.hc_mult * self.hidden_size" not in normalized_init
            or "self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)" not in normalized_init
            or "self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)" not in normalized_init):
        raise ValueError("mHC parameter dimension or model-dtype gamma source changed")
    model_init = _definition(model_tree, "DeepseekV4Model.__init__")
    if "self.hc_dim = self.hc_mult * config.hidden_size" not in [ast.unparse(node) for node in model_init.body if isinstance(node, ast.Assign)]:
        raise ValueError("mHC head parameter flattening geometry changed")
    norm_init = _definition(ast.parse((root / "vllm/model_executor/layers/layernorm.py").read_text()), "RMSNorm.__init__")
    norm_statements = [ast.unparse(node) for node in ast.walk(norm_init) if isinstance(node, ast.Assign)]
    if ("weight_dtype = dtype or torch.get_default_dtype()" not in norm_statements
            or "self.weight = torch.ones(hidden_size, dtype=weight_dtype)" not in norm_statements):
        raise ValueError("mHC standard gamma dtype/allocation source changed")
    broadcast_prepare = _definition(model_tree, "DeepseekV4Model.finalize_mhc_broadcast_weights")
    compact_prepare = re.sub(r"\s+", "", ast.unparse(broadcast_prepare))
    if "layer.hc_attn_fn.detach().view(-1,layer.hc_mult,layer.hidden_size).sum(dim=1)" not in compact_prepare:
        raise ValueError("mHC broadcast projection weight preparation changed")
    kernel_relative = "vllm/model_executor/kernels/mhc/tilelang_kernels.py"
    fused_launcher = _definition(ast.parse((root / kernel_relative).read_text()), "MhcFusedTileLangKernel.__call__")
    fused_allocations = [ast.unparse(node) for node in fused_launcher.body if isinstance(node, ast.Assign)
                         and any(ast.unparse(target) in ("yp_out", "rp_out", "residual_out") for target in node.targets)]
    if fused_allocations != ["yp_out = residual_in.new_empty((n_splits, num_tokens, hc_mult3), dtype=torch.float32)",
            "rp_out = residual_in.new_empty((n_splits, num_tokens), dtype=torch.float32)", "residual_out = residual_in.new_empty(residual_in.shape)"]:
        raise ValueError("mHC fused launcher tensor outputs/partials changed")
    prenorm = _definition(mhc_tree, "_hc_prenorm_gemm_outputs")
    prenorm_allocations = [node for node in ast.walk(prenorm) if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.empty"]
    if len(prenorm_allocations) != 2 or any({keyword.arg: ast.unparse(keyword.value) for keyword in node.keywords} != {"dtype": "torch.float32", "device": "x.device"} for node in prenorm_allocations):
        raise ValueError("mHC prenorm FP32 temporary allocation source changed")
    if [[ast.unparse(arg) for arg in node.args] for node in prenorm_allocations] != [
            ["n_splits", "num_tokens", "fn.shape[0]"], ["n_splits", "num_tokens"]]:
        raise ValueError("mHC prenorm temporary geometry changed")
    active_wrappers = [prenorm] + [_definition(mhc_tree, symbol) for symbol in (
        "mhc_pre_broadcast_tilelang", "mhc_fused_post_pre_tilelang", "mhc_post_tilelang", "hc_head_fused_kernel_tilelang")]
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and (
            node.func.attr == "float" or node.func.attr == "to" and any(ast.unparse(arg) == "torch.float32" for arg in node.args))
            for function in active_wrappers for node in ast.walk(function)):
        raise ValueError("mHC active caller now materializes an FP32 input/parameter cast")
    load = _definition(model_tree, "DeepseekV4ForCausalLM.load_weights")
    after_load = _definition(model_tree, "DeepseekV4ForCausalLM.process_weights_after_loading")
    if (not any(isinstance(node, ast.Call) and ast.unparse(node.func) == "self.process_weights_after_loading" for node in ast.walk(load))
            or not any(isinstance(node, ast.Call) and ast.unparse(node.func) == "self.model.finalize_mhc_broadcast_weights" for node in ast.walk(after_load))):
        raise ValueError("mHC broadcast weights no longer finalized by standard weight loading")
    anchors["mhc-wrapper-objects"] = {"state_bindings": anchors["mhc-state-bindings"], "parameters": parameter_bindings,
        "broadcast_weight_preparation": _function_anchor(root, _MODEL, "DeepseekV4Model.finalize_mhc_broadcast_weights"),
        "fused_launcher": _function_anchor(root, kernel_relative, "MhcFusedTileLangKernel.__call__"),
        "prenorm": _function_anchor(root, mhc_relative, "_hc_prenorm_gemm_outputs"),
        "prenorm_adapter": _function_anchor(root, "vllm/utils/deep_gemm.py", "tf32_hc_prenorm_gemm"),
        "scope": "source caller parameters, views, returned and exposed temporary objects with formal split count; internal compiler/library scratch and actual provider not bound"}
    route_relative = "vllm/model_executor/layers/fused_moe/router/dsv4_topk.py"
    route_tree = ast.parse((root / route_relative).read_text(encoding="utf-8"))
    kernels = [node for node in ast.walk(route_tree) if isinstance(node, ast.FunctionDef)
               and node.name == "_dsv4_topk_kernel"]
    if len(kernels) != 1:
        raise ValueError("V4 conditional Triton routing kernel is missing or ambiguous")
    route_kernel = kernels[0]
    compact_route = re.sub(r"\s+", "", ast.unparse(route_kernel))
    for fragment in ("tl.static_range(6)", "tl.arange(0,8)", "tl.where(logits>20.0,logits,tl.log(1.0+tl.exp(logits)))",
                     "tl.where(current==current,current,-1e+30)", "tl.min(candidate,axis=0)",
                     "tl.where(expert_offsets==expert_id,weights,0.0)",
                     "tl.where(weight_sum>0.0,weight_sum,1.0)",
                     "selected_weights=tl.zeros([8],dtype=tl.float32)",
                     "selected_ids=tl.zeros([8],dtype=tl.int32)",
                     "max_value=tl.max(current,axis=0)",
                     "candidate=tl.where(current==max_value,expert_offsets,NUM_EXPERTS)",
                     "expert_id=tl.min(candidate,axis=0).to(tl.int32)",
                     "selected_weight=tl.sum(tl.where(expert_offsets==expert_id,weights,0.0),axis=0)",
                     "weight_sum=tl.sum(selected_weights,axis=0)"):
        if fragment not in compact_route:
            raise ValueError("V4 Triton routing source program changed: " + fragment)
    anchors["v4-route-source"] = {
        "triton_kernel": {"path": str(root / route_relative), "line": route_kernel.lineno,
                          "end_line": route_kernel.end_lineno, "symbol": route_kernel.name},
        "triton_wrapper": _function_anchor(root, route_relative, "dsv4_topk"),
        "dispatch": _function_anchor(root, "vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py", "fused_topk_bias"),
        "hash_cuda": _cpp_anchor(root, "csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu", "void dsv4HashTopkSoftplusSqrt("),
        "scope": "declared normal modular pure-text FP32/INT32 top6 source, not observed runtime dispatch"}
    route_wrapper = "vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py"
    router_base = "vllm/model_executor/layers/fused_moe/router/base_router.py"
    runner_relative = "vllm/model_executor/layers/fused_moe/runner/moe_runner.py"
    anchors["route-tensor-dataflow"] = _function_anchor(root, route_wrapper, "fused_topk_bias")
    anchors["route-tensor-dataflow"]["related_definitions"] = [
        _function_anchor(root, relative, symbol) for relative, symbol in (
            (route_relative, "dsv4_topk"), (route_wrapper, "FusedTopKBiasRouter._compute_routing"),
            (router_base, "BaseRouter.__init__"), (router_base, "BaseRouter._select_experts"),
            ("vllm/model_executor/layers/fused_moe/router/fused_moe_router.py", "FusedMoERouter.__init__"),
            ("vllm/model_executor/layers/fused_moe/router/fused_moe_router.py", "FusedMoERouter.select_experts"),
            (router_base, "BaseRouter._apply_eplb_mapping"), (router_base, "BaseRouter._convert_indices_dtype"),
            (runner_relative, "MoERunner._apply_quant_method"),
            ("vllm/config/parallel.py", "ParallelConfig._validate_parallel_config"),
            (_MODEL, "DeepseekV4MoE.__init__"))]
    hash_launcher = _cpp_anchor(root, "csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu",
                                "void topkGatingSoftplusSqrtKernelLauncher(")
    anchors["route-tensor-dataflow"]["hash_launcher"] = hash_launcher
    launcher_source = re.sub(r"\s+", "", hash_launcher["source"])
    fast_start = launcher_source.index("if(use_hash&&topk==6&&renormalize&&")
    fast_end = launcher_source.index("return;", fast_start)
    fast_body = launcher_source[fast_start:fast_end]
    if "launchDsv4HashTopk<IndType,HashIndType>" not in fast_body or "token_expert_indices" in fast_body:
        raise ValueError("V4 hash fast path now uses the wrapper temporary")
    for relative, symbol, fragments in (
        (route_wrapper, "fused_topk_bias", (
            "input_tokens.dtype!=hash_indices_table.dtype", "input_tokens=input_tokens.to(dtype=hash_indices_table.dtype)",
            "topk_weights=torch.empty(M,topk,dtype=torch.float32,device=hidden_states.device)",
            "token_expert_indices=torch.empty(M,topk,dtype=torch.int32,device=hidden_states.device)")),
        (route_relative, "dsv4_topk", ("shape=(num_tokens,_TOPK)",
            "topk_weights=gating_output.new_empty(shape,dtype=torch.float32)", "topk_ids=gating_output.new_empty(shape,dtype=indices_dtype)",
            "return(topk_weights,topk_ids)")),
        (router_base, "BaseRouter._apply_eplb_mapping", ("ifself.eplb_stateisnotNone:", "returntopk_ids")),
        (router_base, "BaseRouter._convert_indices_dtype", ("ifindices_typeisnotNoneandtopk_ids.dtype!=indices_type:", "returntopk_ids")),
        (runner_relative, "MoERunner._apply_quant_method", (
            "topk_weights=topk_weights,topk_ids=topk_ids",)),
        ("vllm/config/parallel.py", "ParallelConfig._validate_parallel_config", (
            "ifself.enable_eplb:", "ifnotself.enable_expert_parallel:")),
        (_MODEL, "DeepseekV4MoE.__init__", (
            "self.hash_indices_dtype=torch.int64ifself.use_mega_moeelsetorch.int32", "dtype=self.hash_indices_dtype")),
    ):
        body = re.sub(r"\s+", "", ast.unparse(_definition(ast.parse((root / relative).read_text()), symbol)))
        if any(fragment not in body for fragment in fragments):
            raise ValueError("V4 routing tensor source contract changed: " + symbol)
    route_runner = _definition(ast.parse((root / runner_relative).read_text()), "MoERunner._apply_quant_method")
    route_returns = [node for node in ast.walk(route_runner) if isinstance(node, ast.Assign)
                     and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "self.router.select_experts"]
    if len(route_returns) != 1 or [ast.unparse(node) for node in route_returns[0].targets[0].elts] != ["topk_weights", "topk_ids"]:
        raise ValueError("V4 modular router return objects changed")
    base_init = _definition(ast.parse((root / router_base).read_text()), "BaseRouter.__init__")
    captures = [node for node in ast.walk(base_init) if isinstance(node, ast.AnnAssign) and ast.unparse(node.target) == "self.capture_fn"]
    if len(captures) != 1 or ast.literal_eval(captures[0].value) is not None:
        raise ValueError("V4 normal router no longer defaults to no external capture hook")
    replay_init = _definition(ast.parse((root / "vllm/model_executor/layers/fused_moe/router/fused_moe_router.py").read_text()), "FusedMoERouter.__init__")
    replay = [node for node in ast.walk(replay_init) if isinstance(node, ast.AnnAssign) and ast.unparse(node.target) == "self._routing_replay_out"]
    if len(replay) != 1 or ast.literal_eval(replay[0].value) is not None:
        raise ValueError("V4 normal router no longer defaults to no routing replay copy")
    indexer_relative = "vllm/model_executor/layers/sparse_attn_indexer.py"
    indexer_tree = ast.parse((root / indexer_relative).read_text(encoding="utf-8"))
    indexer_function = _definition(indexer_tree, "sparse_attn_indexer")
    cuda_forward = _definition(indexer_tree, "SparseAttnIndexer.forward_cuda")
    calls = [node for node in ast.walk(cuda_forward) if isinstance(node, ast.Call)
             and ast.unparse(node.func) == "torch.ops.vllm.sparse_attn_indexer"]
    if len(calls) != 1:
        raise ValueError("indexer CUDA forwarding source changed")
    arguments = {parameter.arg: argument for parameter, argument in zip(indexer_function.args.args, calls[0].args)}
    if "skip_topk_buffer_clear" not in arguments or ast.literal_eval(arguments["skip_topk_buffer_clear"]) is not False:
        raise ValueError("indexer source no longer requires sentinel initialization")
    clears = [node for node in ast.walk(indexer_function) if isinstance(node, ast.If)
              and ast.unparse(node.test) == "not skip_topk_buffer_clear"]
    if len(clears) != 1 or "topk_indices_buffer[:hidden_states.shape[0]]=-1" not in re.sub(r"\s+", "", ast.unparse(clears[0])):
        raise ValueError("indexer source sentinel write region changed")
    anchors["indexer-baseline"] = {"definitions": [
        _function_anchor(root, relative, symbol) for relative, symbol in (
            (indexer_relative, "sparse_attn_indexer"), (indexer_relative, "SparseAttnIndexer.forward_cuda"),
            ("tests/kernels/attention/test_deepgemm_attention.py", "_ref_fp8_mqa_logits"),
            ("tests/kernels/attention/test_deepgemm_attention.py", "_ref_fp8_fp4_paged_mqa_logits"),
            ("tests/kernels/test_deepseek_v4_cpu_kernels.py", "_reference_paged_mqa_logits"),
            ("vllm/model_executor/layers/indexer_topk.py", "SparseIndexerTopk._resolve_auto"),
            ("vllm/model_executor/layers/indexer_topk.py", "SparseIndexerTopk.forward"))],
        "scope": "official mathematical reference and wrapper payload/order; not GPU instruction order, selected provider, gather/chunk/selector complete service"}
    score_interfaces = []
    for symbol, bounds in (("fp8_fp4_mqa_logits", ["cu_seqlen_ks", "cu_seqlen_ke"]),
                           ("fp8_fp4_paged_mqa_logits", ["seq_lens"])):
        score_calls = [node for node in ast.walk(indexer_function) if isinstance(node, ast.Call)
                       and ast.unparse(node.func) == symbol]
        if len(score_calls) != 1 or [ast.unparse(arg) for arg in score_calls[0].args[3:3 + len(bounds)]] != bounds:
            raise ValueError("indexer score metadata input interface changed")
        options = {keyword.arg: ast.unparse(keyword.value) for keyword in score_calls[0].keywords}
        if options["clean_logits"] != "False":
            raise ValueError("indexer now requests inactive score cleanup")
        score_interfaces.append({"symbol": symbol, "call_line": score_calls[0].lineno, "bound_inputs": bounds,
                                 "clean_logits": False})
    for symbol in ("ops.top_k_per_row_prefill", "get_indexer_topk(topk_backend)"):
        selections = [node for node in ast.walk(indexer_function) if isinstance(node, ast.Call)
                      and ast.unparse(node.func) == symbol]
        if len(selections) != 1 or ast.unparse(selections[0].args[0]) != "logits":
            raise ValueError("indexer selection no longer consumes the score tensor directly")
    deep_gemm_relative = "vllm/utils/deep_gemm.py"
    deep_gemm_module = ast.parse((root / deep_gemm_relative).read_text())
    for symbol, dimensions in (("fp8_fp4_mqa_logits", "[M, N]"),
                               ("fp8_fp4_paged_mqa_logits", "[B * next_n, max_model_len]")):
        description = ast.get_docstring(_definition(deep_gemm_module, symbol))
        if description is None or dimensions not in description or "torch.float32" not in description:
            raise ValueError("indexer wrapper FP32 logical score interface changed")
    anchors["indexer-baseline"]["score_materialization"] = {
        "calls": score_interfaces, "documented_interfaces": [
            _function_anchor(root, deep_gemm_relative, symbol)
            for symbol in ("fp8_fp4_mqa_logits", "fp8_fp4_paged_mqa_logits")],
        "scope": "source wrapper metadata inputs and documented FP32 logical output; physical stride, inactive-cell stores and implementation remain unbound"}
    metadata_relative = "vllm/v1/attention/backends/mla/indexer.py"
    metadata_tree = ast.parse((root / metadata_relative).read_text(encoding="utf-8"))
    metadata_build = _definition(metadata_tree, "DeepseekV32IndexerMetadataBuilder.build")
    chunk_calls = [node for node in ast.walk(metadata_build) if isinstance(node, ast.Call)
                   and ast.unparse(node.func) == "build_prefill_chunk_metadata"]
    if len(chunk_calls) != 1 or [ast.unparse(keyword.value) for keyword in chunk_calls[0].keywords
                                if keyword.arg == "skip_kv_gather"] != ["query_slice.start > 0"]:
        raise ValueError("indexer query-subchunk gather-reuse condition changed")
    indexer_text = re.sub(r"\s+", "", ast.unparse(indexer_function))
    if ("notchunk.skip_kv_gatherandchunk.local_total_seq_lens>0" not in indexer_text
            or "k_scale.view(torch.float32).squeeze(-1)" not in indexer_text):
        raise ValueError("indexer gather dispatch or FP32 scale view source changed")
    shapes_text = re.sub(r"\s+", "", ast.unparse(_definition(indexer_tree, "_gather_workspace_shapes")))
    if "((total_seq_lens,4),torch.uint8)" not in shapes_text:
        raise ValueError("indexer FP8 gather scale backing changed")
    anchors["indexer-baseline"]["prefill_gather"] = {"definitions": [
        _function_anchor(root, relative, symbol) for relative, symbol in (
            (indexer_relative, "_gather_workspace_shapes"),
            (metadata_relative, "DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks"),
            (metadata_relative, "DeepseekV32IndexerMetadataBuilder.build"),
            (metadata_relative, "build_prefill_chunk_metadata"))],
        "copy_kernel": _cpp_anchor(root, "csrc/libtorch_stable/cache_kernels.cu", "__global__ void cp_gather_indexer_k_quant_cache_kernel("),
        "wrapper": _cpp_anchor(root, "csrc/libtorch_stable/cache_kernels.cu", "void cp_gather_indexer_k_quant_cache("),
        "scope": "CP1 FP8 128-wide payload copied once per request context; query subchunks reuse workspace; metadata/peak capacity/compiler/native binding excluded"}
    sampler_relative = "csrc/libtorch_stable/sampler.cu"
    prefill_wrapper = _cpp_anchor(root, sampler_relative, "void top_k_per_row_prefill(")
    prefill_kernel = _cpp_anchor(root, sampler_relative,
        "static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowPrefill(")
    shortcut = _cpp_anchor(root, sampler_relative, "if (rowLen <= topK)")
    specialized_shortcut = _false_constexpr_body(shortcut["source"], "multipleBlocksPerRow")
    if re.search(r"\b(?:logits|outLogits)\b", specialized_shortcut) or "return;" not in specialized_shortcut:
        raise ValueError("prefill top-k shortcut now accesses scores or no longer returns")
    topk_job = _cpp_anchor(root, sampler_relative,
        "template <int kNumThreadsPerBlock, int kNumBins, bool useRadixSort,")
    if ("bool multipleBlocksPerRow = false" not in topk_job["source"]
            or "topKPerRowJob<kNumThreadsPerBlock,kNumBins,useRadixSort>(" not in re.sub(r"\s+", "", prefill_kernel["source"])):
        raise ValueError("prefill top-k no longer uses unsplit topKPerRowJob")
    anchors["indexer-baseline"]["prefill_selection"] = {"wrapper": prefill_wrapper,
        "kernel": prefill_kernel, "job": topk_job, "shortcut": shortcut,
        "specialization": "multipleBlocksPerRow=false; no logits/outLogits access in the complete rowLen<=topK branch",
        "scope": "source row-bound shortcut and natural/-1 indices; no native/compiler instruction or runtime-provider proof"}
    barrier_definitions = (
        (_ATTENTION, "DeepseekV4Attention._sparse_indexer_and_attn",
         "self.indexer.indexer_op", "self.forward_mqa"),
        (indexer_relative, "sparse_attn_indexer", "ops.top_k_per_row_prefill", "fp8_fp4_paged_mqa_logits"),
        ("vllm/models/deepseek_v4/nvidia/flashmla.py", "DeepseekV4FlashMLAAttention.forward_mqa",
         "self._forward_prefill", "self._forward_decode"))
    anchors["indexer-call-barriers"] = {"definitions": [],
        "scope": "source-call return/join and prefill-before-decode order; not a compiler or measured GPU stream observation"}
    for relative, symbol, before, after in barrier_definitions:
        definition = _definition(ast.parse((root / relative).read_text(encoding="utf-8")), symbol)
        positions = {name: [node.lineno for node in ast.walk(definition)
                           if isinstance(node, ast.Call) and ast.unparse(node.func) == name]
                     for name in (before, after)}
        if len(positions[before]) != 1 or len(positions[after]) != 1 or positions[before][0] >= positions[after][0]:
            raise ValueError("V4 source-call phase barrier changed: " + symbol)
        anchors["indexer-call-barriers"]["definitions"].append({
            **_function_anchor(root, relative, symbol), "before": before, "after": after,
            "before_line": positions[before][0], "after_line": positions[after][0]})
    compact_hash = re.sub(r"\s+", "", anchors["v4-route-source"]["hash_cuda"]["source"])
    for fragment in ("lane<6&&!is_pad_row", "tid2eid[token_id*6+lane]", "__expf(-fabsf(x))",
                     "if(isnan(weight)){weight=0.f;}", "mask=16;mask>0;mask>>=1",
                     "weight*routed_scaling_factor/(weight_sum>0.f?weight_sum:1.f)",
                     "floatweight=0.f", "floatweight_sum=weight",
                     "weight_sum+=VLLM_SHFL_XOR_SYNC(weight_sum,mask)"):
        if fragment not in compact_hash:
            raise ValueError("V4 CUDA hash routing source program changed: " + fragment)
    anchors["inverse-rope-quant"]["related_definitions"] = [
        _function_anchor(root, "vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py", symbol)
        for symbol in ("fused_inv_rope_fp8_quant", "_fused_inv_rope_fp8_quant_kernel_impl")]
    o_project = "vllm/models/deepseek_v4/nvidia/ops/o_proj.py"
    anchors["o-projection-objects"] = _function_anchor(root, o_project, "deep_gemm_fp8_o_proj")
    anchors["attention-output-regions"] = {"definitions": [
        _function_anchor(root, relative, symbol) for relative, symbol in (
            (_ATTENTION, "DeepseekV4Attention.forward"),
            ("vllm/models/deepseek_v4/nvidia/flashmla.py", "DeepseekV4FlashMLAAttention.forward_mqa"),
            ("vllm/models/deepseek_v4/nvidia/flashmla.py", "DeepseekV4FlashMLAAttention._forward_prefill"),
            ("vllm/models/deepseek_v4/nvidia/flashmla.py", "DeepseekV4FlashMLAAttention._forward_decode"))],
        "scope": "caller-owned empty output and supplied out views; matching FlashMLA callee internals remain unverified"}
    for definition, fragments in zip(anchors["attention-output-regions"]["definitions"], (
            ("o_padded=torch.empty((num_tokens,self.padded_heads,self.head_dim)", "o=o_padded[:,:self.n_local_heads,:]"),
            ("output=output[num_decode_tokens:]", "output=output[:num_decode_tokens]"),
            ("out=output[query_start:query_end]",),
            ("out=output.unsqueeze(1)",)), strict=True):
        function = _definition(ast.parse(Path(definition["path"]).read_text(encoding="utf-8")), definition["symbol"])
        compact = re.sub(r"\s+", "", ast.unparse(function))
        if any(fragment not in compact for fragment in fragments):
            raise ValueError("V4 attention supplied-output source program changed")
        if definition["symbol"].endswith(("._forward_prefill", "._forward_decode")):
            decode = definition["symbol"].endswith("._forward_decode")
            callee = "flash_mla_with_kvcache" if decode else "flash_mla_sparse_fwd"
            calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == callee]
            output_loads = [node for node in ast.walk(function) if isinstance(node, ast.Name)
                            and node.id == "output" and isinstance(node.ctx, ast.Load)]
            expected_out = ast.parse("output.unsqueeze(1)" if decode else "output[query_start:query_end]", mode="eval").body
            if len(calls) != 1 or len(output_loads) != 1 or not any(keyword.arg == "out"
                    and ast.dump(keyword.value) == ast.dump(expected_out) for keyword in calls[0].keywords):
                raise ValueError("V4 attention callee must receive its unchanged supplied output view")
            if decode and any(isinstance(node, ast.Name) and node.id == "out" and isinstance(node.ctx, ast.Load)
                              for node in ast.walk(function)):
                raise ValueError("V4 decode callee return is no longer ignored")
    anchors["o-projection-objects"]["related_definitions"] = [
        _function_anchor(root, "vllm/model_executor/layers/quantization/utils/fp8_utils.py", symbol)
        for symbol in ("deepgemm_post_process_fp8_weight_block", "deepgemm_post_process_weight_scale_block", "_upcast_e8m0_to_fp32")]
    for relative, symbol, required in (
        (o_project, "deep_gemm_fp8_o_proj", ("z = torch.empty((o.shape[0], n_groups, o_lora_rank), device=o.device, dtype=torch.bfloat16)",
            "return wo_b(z.flatten(1))", "'bhr,hdr->bhd'", "recipe=einsum_recipe")),
        ("vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py", "fused_inv_rope_fp8_quant",
            ("output = out_buf.transpose(0, 1)", "scales = scale_buf.transpose(0, 1) if quantize else scale_buf")),
        ("vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py", "_fused_inv_rope_fp8_quant_kernel_impl",
            ("(n_groups, num_tokens, d)", "(scale_inner * tma_aligned_T, 1, tma_aligned_T)",
             "n_groups * scale_inner * tma_aligned_T", "grid = (tma_aligned_T, n_groups * heads_per_group)")),
        (_ATTENTION, "DeepseekV4Attention.forward", ("o = o_padded[:, :self.n_local_heads, :]",)),
        ("vllm/model_executor/layers/quantization/utils/fp8_utils.py", "deepgemm_post_process_weight_scale_block",
            ("ws = _upcast_e8m0_to_fp32(ws)", "transform_sf_into_required_layout")),
    ):
        source = ast.unparse(_definition(ast.parse((root / relative).read_text()), symbol))
        if any(expression not in source for expression in required):
            raise ValueError(f"upstream O projection object contract changed: {symbol}")
    anchors.update(prepare_source_bindings(root))
    anchors["prefill-gather"]["related_definitions"] = [
        _function_anchor(root, relative, symbol) for relative, symbol in (
            ("vllm/models/deepseek_v4/nvidia/flashmla.py", "DeepseekV4FlashMLAAttention._forward_prefill"),
            ("vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py", "DequantGatherKCacheKernel.kernel"),
            ("vllm/models/deepseek_v4/common/ops/cache_utils.py", "DequantizeAndGatherKCacheKernel.kernel"),
            ("vllm/model_executor/layers/sparse_attn_indexer.py", "sparse_attn_indexer"))]
    cache_relative = "vllm/models/deepseek_v4/common/ops/cache_utils.py"
    anchors["prefill-combine-indices"]["related_definitions"] = [
        _function_anchor(root, cache_relative, symbol) for symbol in (
            "CombineTopkSwaIndicesKernel.kernel", "CombineTopkSwaIndicesKernel.__call__")]
    cache_tree = ast.parse((root / cache_relative).read_text(encoding="utf-8"))
    short_symbol = "_fill_short_context_topk_indices"
    anchors["short-context-topk-dataflow"] = _function_anchor(root, _ATTENTION, short_symbol)
    anchors["short-context-topk-dataflow"]["related_definitions"] = [
        _function_anchor(root, relative, symbol) for relative, symbol in (
            (_MODEL, "DeepseekV4Model.__init__"), (_MODEL, "DeepseekV4DecoderLayer.__init__"),
            (_ATTENTION, "DeepseekV4Attention.__init__"), (_ATTENTION, "DeepseekV4Indexer.__init__"),
            (_ATTENTION, "DeepseekV4Indexer.forward"),
            ("vllm/models/deepseek_v4/nvidia/flashmla.py", "DeepseekV4FlashMLAAttention._forward_prefill"),
            ("vllm/models/deepseek_v4/nvidia/flashmla.py", "DeepseekV4FlashMLAAttention._forward_decode"))]
    c128_relative = "vllm/models/deepseek_v4/sparse_mla.py"
    anchors["c128-metadata-width"] = _function_anchor(root, c128_relative, "DeepseekV4SparseMLAMetadataBuilder._build_c128a_metadata")
    anchors["c128-metadata-width"]["related_definitions"] = [
        _function_anchor(root, c128_relative, symbol) for symbol in
        ("DeepseekV4SparseMLAMetadataBuilder.__init__", "build_c128a_topk_metadata", "BuildC128ATopkMetadataKernel.kernel")]
    c128_tree = ast.parse((root / c128_relative).read_text())
    alignments = [ast.literal_eval(node.value) for node in c128_tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "_C128A_TOPK_ALIGNMENT" for target in node.targets)]
    if alignments != [128]:
        raise ValueError("C128 source metadata alignment changed")
    anchors["c128-metadata-width"]["alignment"] = alignments[0]
    for symbol, fragments in (
        ("DeepseekV4SparseMLAMetadataBuilder._build_c128a_metadata", (
            "triton.next_power_of_2(max(cm.max_seq_len//self.compress_ratio,1))",
            "self.c128a_max_compressed", "active_topk_width>=cm.max_seq_len//self.compress_ratio",
            "max_compressed_tokens=active_topk_width")),
        ("DeepseekV4SparseMLAMetadataBuilder.__init__", (
            "cdiv(self.model_config.max_model_len,self.compress_ratio)",
            "cdiv(c128a_max_compressed,_C128A_TOPK_ALIGNMENT)*_C128A_TOPK_ALIGNMENT")),
    ):
        body = re.sub(r"\s+", "", ast.unparse(_definition(ast.parse((root / c128_relative).read_text()), symbol)))
        if any(fragment not in body for fragment in fragments):
            raise ValueError("C128 constructor/active metadata width source changed: " + symbol)
    fill = _definition(ast.parse((root / _ATTENTION).read_text()), short_symbol)
    fill_source = re.sub(r"\s+", "", ast.unparse(fill))
    for fragment in ("num_compressed=(tl.load(positions+row)+1)//COMPRESS_RATIO",
                     "output+row*TOP_K+offsets", "tl.where(offsets<num_compressed,offsets,-1)", "mask=offsets<TOP_K"):
        if fragment not in fill_source:
            raise ValueError("V4 short-context supplied-output fill changed")
    source_calls = [ast.unparse(node.func) for node in ast.walk(fill) if isinstance(node, ast.Call)]
    if sorted(source_calls) != sorted(["tl.program_id", "tl.arange", "tl.load", "tl.store", "tl.where"]):
        raise ValueError("V4 short-context fill has additional global/local source services")
    for relative, symbol, fragments in (
        (_MODEL, "DeepseekV4Model.__init__", (
            "self.topk_indices_buffer=torch.empty(vllm_config.scheduler_config.max_num_batched_tokens,config.index_topk,dtype=torch.int32)",
            "topk_indices_buffer=self.topk_indices_buffer")),
        (_MODEL, "DeepseekV4DecoderLayer.__init__", ("topk_indices_buffer=topk_indices_buffer",)),
        (_ATTENTION, "DeepseekV4Attention.__init__", ("self.topk_indices_buffer=topk_indices_buffer", "topk_indices_buffer=topk_indices_buffer")),
        (_ATTENTION, "DeepseekV4Indexer.__init__", ("self.topk_indices_buffer=topk_indices_buffer",)),
        (_ATTENTION, "DeepseekV4Indexer.forward", ("self.topk_indices_buffer,positions,TOP_K=self.topk_tokens", "return(None,None,None)")),
        ("vllm/models/deepseek_v4/nvidia/flashmla.py", "DeepseekV4FlashMLAAttention._forward_prefill", (
            "topk_indices=self.topk_indices_buffer[num_decode_tokens:]", "topk_indices=topk_indices[:num_prefill_tokens]")),
    ):
        body = re.sub(r"\s+", "", ast.unparse(_definition(ast.parse((root / relative).read_text()), symbol)))
        if any(fragment not in body for fragment in fragments):
            raise ValueError("V4 model-owned short-context buffer binding changed: " + symbol)
    anchors["decode-c4-indices"] = _function_anchor(root, cache_relative, "compute_global_topk_indices_and_lens")
    anchors["decode-c4-indices"]["related_definitions"] = [
        _function_anchor(root, relative, symbol) for relative, symbol in (
            (cache_relative, "ComputeGlobalTopkIndicesAndLensKernel.kernel"),
            (cache_relative, "ComputeGlobalTopkIndicesAndLensKernel.__call__"),
            ("vllm/models/deepseek_v4/nvidia/flashmla.py", "DeepseekV4FlashMLAAttention._forward_decode"))]
    decode_relative = "vllm/models/deepseek_v4/nvidia/flashmla.py"
    decode_definition = _definition(ast.parse((root / decode_relative).read_text()), "DeepseekV4FlashMLAAttention._forward_decode")
    decode_calls = [node for node in ast.walk(decode_definition) if isinstance(node, ast.Call)
                    and ast.unparse(node.func) == "flash_mla_with_kvcache"]
    if len(decode_calls) != 1:
        raise ValueError("V4 decode attention caller changed")
    decode_arguments = {keyword.arg: ast.unparse(keyword.value) for keyword in decode_calls[0].keywords}
    if any(name not in decode_arguments or decode_arguments[name] != value for name, value in (
            ("indices", "swa_indices"), ("topk_length", "swa_lens"),
            ("extra_indices_in_kvcache", "topk_indices"), ("extra_topk_length", "topk_lens"))):
        raise ValueError("V4 decode attention no longer receives both index/length tiers")
    anchors["decode-attention-index-inputs"] = _function_anchor(root, decode_relative, "DeepseekV4FlashMLAAttention._forward_decode")
    anchors["decode-attention-index-inputs"]["related_definitions"] = []
    for relative, symbol, fields in (
            ("vllm/v1/attention/backends/mla/sparse_swa.py", "DeepseekSparseSWAMetadataBuilder.__init__",
             ("self.decode_swa_indices", "self.decode_swa_lens")),
            ("vllm/models/deepseek_v4/sparse_mla.py", "DeepseekV4SparseMLAMetadataBuilder.__init__",
             ("self.c128a_global_decode_buffer", "self.c128a_decode_lens_buffer"))):
        definition = _definition(ast.parse((root / relative).read_text()), symbol)
        for field in fields:
            allocations = [node.value for node in ast.walk(definition) if isinstance(node, ast.Assign)
                           and any(ast.unparse(target) == field for target in node.targets)]
            if (len(allocations) != 1 or not isinstance(allocations[0], ast.Call)
                    or not any(keyword.arg == "dtype" and ast.unparse(keyword.value) == "torch.int32"
                               for keyword in allocations[0].keywords)):
                raise ValueError("V4 decode index/length storage is not explicitly INT32: " + field)
        anchors["decode-attention-index-inputs"]["related_definitions"].append(_function_anchor(root, relative, symbol))
    for symbol, fragments in (
        ("compute_global_topk_indices_and_lens", ("global_topk_indices=torch.empty_like(topk_indices)",
            "topk_lens=torch.empty(num_tokens,dtype=torch.int32,device=topk_indices.device)",
            "return(global_topk_indices,topk_lens)")),
        ("ComputeGlobalTopkIndicesAndLensKernel.kernel", (
            "is_valid_token=tl.load(is_valid_token_ptr+token_idx)", "req_idx=tl.load(token_to_req_indices_ptr+token_idx)",
            "mask=offset<topk", "is_valid=local_idx>=0", "mask=mask&is_valid",
            "slot_ids=block_numbers*block_size+block_offsets", "slot_ids=tl.where(is_valid,slot_ids,-1)",
            "count+=tl.sum(is_valid.to(tl.int32),axis=0)", "tl.where(is_valid_token,count,0)")),
    ):
        body = re.sub(r"\s+", "", ast.unparse(_definition(cache_tree, symbol)))
        if any(fragment not in body for fragment in fragments):
            raise ValueError("V4 C4 decode index materialization source changed: " + symbol)
    decoder_tree = ast.parse((root / "vllm/models/deepseek_v4/nvidia/flashmla.py").read_text())
    decoder = re.sub(r"\s+", "", ast.unparse(_definition(decoder_tree, "DeepseekV4FlashMLAAttention._forward_decode")))
    for fragment in ("ifself.compress_ratio==4:", "self.topk_indices_buffer[:num_decode_tokens]",
                     "global_indices.view(num_decode_tokens,1,-1)", "extra_topk_length=topk_lens"):
        if fragment not in decoder:
            raise ValueError("V4 per-layer C4 decode helper binding changed")
    index_class = _definition(cache_tree, "ComputeGlobalTopkIndicesAndLensKernel")
    index_block_sizes = [ast.literal_eval(node.value) for node in index_class.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "TRITON_BLOCK_SIZE" for target in node.targets)]
    if index_block_sizes != [1024]:
        raise ValueError("V4 C4 decode index source vector extent changed")
    workers = _definition(cache_tree, "CombineTopkSwaIndicesKernel")
    worker_values = [ast.literal_eval(node.value) for node in workers.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "NUM_WORKERS" for target in node.targets)]
    alignment_values = [ast.literal_eval(node.value) for node in cache_tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "_SPARSE_PREFILL_TOPK_ALIGNMENT" for target in node.targets)]
    if worker_values != [256] or alignment_values != [128]:
        raise ValueError("V4 prefill index worker count or alignment changed")
    combine = ast.unparse(_definition(cache_tree, "combine_topk_swa_indices"))
    if "combined_indices.fill_(-1)" not in combine:
        raise ValueError("V4 prefill caller-buffer full fill changed")
    anchors["prefill-gather"]["causal_topk"] = _cpp_anchor(root, "csrc/libtorch_stable/sampler.cu",
        "static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowPrefill(")
    anchors["prefill-gather"]["causal_topk_job"] = _cpp_anchor(root, "csrc/libtorch_stable/sampler.cu",
        "static __device__ void topKPerRowJob(")
    finish_relative = "vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py"
    finish_tree = ast.parse((root / finish_relative).read_text(encoding="utf-8"))
    c128 = _definition(finish_tree, "SparseAttnCompressC128Block8Kernel")
    constants = {target.id: ast.literal_eval(node.value) for node in c128.body if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name)
                 and target.id in ("head_tile", "rows_per_warp", "num_warps", "final_reduce_steps")}
    if constants != {"head_tile": 64, "rows_per_warp": 16, "num_warps": 8, "final_reduce_steps": 3}:
        raise ValueError("C128 two-pass geometry changed; rederive its source work")
    for name in ("SparseAttnCompressNormRopeStoreC4Kernel", "SparseAttnNormRopeStoreKernel"):
        minimum = [ast.literal_eval(node.value) for node in _definition(finish_tree, name).body
                   if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "min_scale"
                                                          for target in node.targets)]
        if minimum != [1e-4]:
            raise ValueError("main finish raw-scale lower bound changed")
    for relative, symbol, fragments in (
        (finish_relative, "SparseAttnCompressNormRopeStoreC4Kernel.kernel", (
            "local_sum[e]=local_sum[e]*old_scale+new_scale", "local_product[e]=local_product[e]*old_scale+kv*new_scale",
            "scale_raw=cute.arch.fmax(Float32(min_scale),absmax/Float32(fp8_max))")),
        (finish_relative, "SparseAttnNormRopeStoreKernel.kernel", (
            "scale_raw=cute.arch.fmax(Float32(min_scale),absmax/Float32(fp8_max))",
            "ue8m0=bits+Uint32(8388607)>>Uint32(23)&Uint32(255)",
            "inv_scale=_recast_val(Uint32(254)-ue8m0<<Uint32(23),Float32)")),
        (finish_relative, "SparseAttnCompressC128Block8Kernel.__call__", (
            "compressed_kv=torch.empty((num_actual,head_dim),dtype=torch.float32,device=state_cache.device)",)),
        ("vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py", "_fused_kv_compress_norm_rope_insert_indexer_attn", (
            "result_bf16=result.to(tl.bfloat16).to(tl.float32)", "absmax=tl.maximum(absmax,0.0001)",
            "raw_scale=absmax*INV_FP8_MAX", "exponent=tl.ceil(tl.log2(raw_scale))",
            "inv_scale=tl.exp2(-exponent)", "scale_val=tl.exp2(exponent)")),
    ):
        compact = re.sub(r"\s+", "", ast.unparse(_definition(ast.parse((root / relative).read_text(encoding="utf-8")), symbol)))
        missing = [fragment for fragment in fragments if fragment not in compact]
        if missing:
            raise ValueError(f"finish source program changed: {relative}:{symbol}: {missing}")
    anchors["fp8-stage-forms"] = fp8_quantizer_source_bindings(root)
    row_relative = "vllm/v1/attention/backends/utils.py"
    row_definition = _definition(ast.parse((root / row_relative).read_text()), "reorder_batch_to_split_decodes_and_prefills")
    compact = re.sub(r"\s+", "", ast.unparse(row_definition))
    for expression in ("has_context=num_computed_tokens_np>0", "is_below_threshold=num_scheduled_tokens_np<=decode_threshold",
        "req_regions[is_pure_prefill]=3", "needs_swap=req_regions!=target_regions", "orig_indices=np.where(needs_swap)[0]",
        "sorted_order=np.argsort(req_regions[needs_swap],kind='stable')", "input_batch.swap_states(src,dst)"):
        if expression not in compact:
            raise ValueError("vLLM source row packing contract changed")
    split_definition = _definition(ast.parse((root / row_relative).read_text()), "split_decodes_and_prefills")
    if [ast.literal_eval(value) for value in split_definition.args.defaults] != [1, False, True]:
        raise ValueError("vLLM attention split defaults changed")
    split_source = re.sub(r"\s+", "", ast.unparse(split_definition))
    for expression in ("max_query_len<=decode_threshold", "query_lens[0].item()>decode_threshold",
        "is_prefill=query_lens>decode_threshold", "first_prefill=is_prefill.int().argmax(dim=-1).item()",
        "num_decode_tokens=query_start_loc[first_prefill].item()"):
        if expression not in split_source:
            raise ValueError("vLLM attention query-prefix split changed")
    anchors["model-input-row-order"] = {**_function_anchor(root, row_relative, "reorder_batch_to_split_decodes_and_prefills"),
        "related_definitions": [_function_anchor(root, row_relative, "split_decodes_and_prefills"),
            _function_anchor(root, "vllm/v1/worker/gpu_model_runner.py", "GPUModelRunner._prepare_inputs")],
        "scope": "explicit initial batch and source query-prefix split; short extends require prompt lengths, not runtime InputBatch receipt"}
    return anchors


def flash_cases(prefill_tokens):
    """External scenario definition, independent of the candidate artifact."""
    if type(prefill_tokens) is not int or prefill_tokens < 1:
        raise ValueError("prefill_tokens must be a positive integer")
    return [
        {"phase": "prefill", "request_ids": ["request"], "requests": [{"query": prefill_tokens, "computed": 0}]},
        {"phase": "decode1", "request_ids": ["request"], "requests": [{"query": 1, "computed": prefill_tokens}]},
        {"phase": "decode2", "request_ids": ["request"], "requests": [{"query": 1, "computed": prefill_tokens + 1}]},
        {"phase": "mixed", "request_ids": ["new-request", "request"], "requests": [{"query": prefill_tokens, "computed": 0},
                                           {"query": 1, "computed": prefill_tokens + 2}]},
    ]


def validate_flash_source_cases(cases, max_sequence_length):
    """Validate independent inputs using the existing reference case format.

    A nonzero initial prefix declares pre-existing state, not its physical
    pages or values. Consecutive appearances must still conserve tokens.
    """
    if not isinstance(cases, (list, tuple)) or len(cases) != 4:
        raise ValueError("source cases require the four consecutive Flash phases")
    normalized, prior_ends = [], {}
    for case, phase in zip(cases, ("prefill", "decode1", "decode2", "mixed"), strict=True):
        if (not isinstance(case, dict) or set(case) != {"phase", "request_ids", "requests"}
                or case["phase"] != phase):
            raise ValueError("source case fields and phase order must match the declared Flash profile")
        ids, requests = case["request_ids"], case["requests"]
        if (not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not item for item in ids)
                or len(set(ids)) != len(ids) or not isinstance(requests, list) or len(requests) != len(ids)):
            raise ValueError("source cases require unique request IDs and matching nonempty request rows")
        for request_id, request in zip(ids, requests, strict=True):
            if (not isinstance(request, dict) or not {"query", "computed"}.issubset(request)
                    or set(request) - {"query", "computed", "prompt_tokens"}
                    or any(type(value) is not int for value in request.values())
                    or request["query"] <= 0 or request["computed"] < 0
                    or request["query"] + request["computed"] > max_sequence_length
                    or ("prompt_tokens" in request and not 0 < request["prompt_tokens"] <= max_sequence_length)):
                raise ValueError("source request tokens require valid integers within the declared context")
            if request_id in prior_ends and request["computed"] != prior_ends[request_id]:
                raise ValueError("consecutive source request prefixes must conserve tokens")
            prior_ends[request_id] = request["computed"] + request["query"]
        normalized.append({"phase": phase, "request_ids": list(ids), "requests": [dict(request) for request in requests]})
    return normalized


def validate_flash_token_ids(token_ids, cases, vocab_size):
    """Validate externally declared synthetic inputs, never candidate echoes."""
    if not isinstance(token_ids, dict) or set(token_ids) != {case["phase"] for case in cases}:
        raise ValueError("independent embedding input requires exactly the declared phases")
    for case in cases:
        values = token_ids[case["phase"]]
        count = sum(request["query"] for request in case["requests"])
        if (not isinstance(values, list) or len(values) != count
                or any(type(value) is not int or not 0 <= value < vocab_size for value in values)):
            raise ValueError("independent embedding IDs require exact row counts and valid integer vocabulary IDs")
    return {phase: list(values) for phase, values in token_ids.items()}


def flash_model_input_cases(cases):
    """Independent decode-prefix permutation of the explicit initial batches.

    vLLM sorts only misplaced regions, preserving already correct slots.
    This is not a claim about unobserved persistent InputBatch state.
    """
    execution, orders = [], {}
    for case in cases:
        requests = case["requests"]
        regions = []
        for request in requests:
            if request["computed"] == 0:
                regions.append(3)
            elif request["query"] > 1:
                regions.append(2)
            elif "prompt_tokens" in request:
                regions.append(0 if request["computed"] >= request["prompt_tokens"] else 1)
            else:
                regions.append(0)  # The ordinary decode domain declared by flash_cases.
        destinations = [index for index, region in enumerate(regions) if region != sorted(regions)[index]]
        permutation = list(range(len(requests)))
        for destination, source in zip(destinations, sorted(destinations, key=lambda index: regions[index]), strict=True):
            permutation[destination] = source
        offsets, cursor = [], 0
        for request in requests:
            offsets.append(cursor)
            cursor += request["query"]
        rows = [row for index in permutation for row in range(offsets[index], offsets[index] + requests[index]["query"])]
        ordered_requests = [requests[index] for index in permutation]
        request_ids = [case["request_ids"][index] for index in permutation]
        boundary = next((index for index, request in enumerate(ordered_requests) if request["query"] > 1), len(requests))
        attention_phases = ["decode"] * boundary + ["prefill"] * (len(requests) - boundary)
        if any("prompt_tokens" in request for request in requests) and not all("prompt_tokens" in request for request in requests):
            raise ValueError("independent prompt lengths must cover the whole declared batch")
        prompt_lengths = [request["prompt_tokens"] for request in requests] if all("prompt_tokens" in request for request in requests) else None
        execution.append({**case, "requests": ordered_requests, "request_ids": request_ids, "attention_phases": attention_phases})
        orders[case["phase"]] = {"program": "vllm_0_30_decode_prefix_model_input", "decode_threshold": 1,
            "initial_request_ids": case["request_ids"], "request_permutation": permutation, "token_row_permutation": rows,
            "model_request_ids": request_ids, "model_query_lengths": [request["query"] for request in ordered_requests],
            "model_attention_phases": attention_phases, "model_reorder_regions": [regions[index] for index in permutation],
            "initial_prompt_lengths": prompt_lengths,
            "model_positions": [position for request in ordered_requests for position in range(request["computed"], request["computed"] + request["query"])],
            "initial_order_source": "explicit caller ScheduleStep order; not observed persistent InputBatch",
            "output_request_ids": request_ids, "extra_GPU_permutation_copy": False, "actual_runtime_bound": False}
    return execution, orders


def attach_embedding_accounts(raw, cases, operators, token_ids):
    """CUDA TP source role bytes and copy predicates, independently of lowering.

    Source ID loads repeat per thread/visit; the input object below is unique
    logical storage, not that instruction count or measured DRAM traffic.
    """
    if token_ids is None:
        return
    padded = ((raw["vocab_size"] + 63) // 64) * 64
    if padded % 4:
        raise ValueError("padded embedding vocabulary must divide over TP4")
    shard, hidden = padded // 4, raw["hidden_size"]
    for record in operators:
        if record["variant"] != "embedding":
            continue
        phase = record["node_id"].split(":", 1)[0]
        values, rank = token_ids[phase], record["rank"]
        start = min(rank * shard, raw["vocab_size"])
        end = min((rank + 1) * shard, raw["vocab_size"])
        owned = sum(start <= value < end for value in values)
        rows = len(values)
        record["account_source"] = "embedding-cuda"
        record["lowering_account"] = [{"kind": "VECTOR", "variant": "vocab_parallel_embedding", "dtype": "bf16",
            "required_shape": {"rows": rows, "hidden": hidden}}]
        record["math_account"] = {"mode": "logical_copy_program", "semantic_fields": {
            "org_vocab_interval": [start, end], "added_vocab_interval": None, "owned_token_rows": owned,
            "source_program": {
                "input_id_internal_widening": "single-kernel scalar static_cast<int64_t>; no global INT64 copy",
                "range_comparisons_per_thread_token_visit": 4,
                "weight_copy_or_zero_output": "every output vector is stored on both branches",
                "token_loop_stride": "grid_y", "vector_loop_stride": "grid_x * 256"}}}
        record["byte_account"] = {"mode": "logical_operands", "operands": [
            {"role": "input_ids", "dtype": "int32", "elements": rows, "bytes": 4 * rows, "access": "read"},
            {"role": "owned_weight_rows", "dtype": "bf16", "elements": owned * hidden,
             "bytes": 2 * owned * hidden, "access": "conditional_read"},
            {"role": "partial_output", "dtype": "bf16", "elements": rows * hidden,
             "bytes": 2 * rows * hidden, "access": "write"}]}


def _attach_embedding_tensor_dataflow(raw, operators, model_token_ids):
    """Fresh local fused output, no global mask, direct TP reduction read."""
    if model_token_ids is None:
        return
    by_id = {record["node_id"]: record for record in operators}
    hidden = raw["hidden_size"]
    shard_rows = ((raw["vocab_size"] + 63) // 64 * 64) // 4
    for record in operators:
        if record["kind"] != "gather" or record["variant"] != "embedding":
            continue
        key, rank = record["node_id"], record["rank"]
        phase = key.split(":", 1)[0]
        rows = record["shape"]["rows"]
        ids = model_token_ids[phase]
        consumer = phase + ":token-embedding:tp-all-reduce"
        output = {"output_role": "partial_embedding", "dtype": "bf16", "shape": [rows, hidden],
                  "strides_elements": [hidden, 1], "offset_elements": 0}
        flow = {
            "program": "vllm_0_30_cuda_tp_embedding_objects",
            "input_ids": {"owner": "model input_ids", "dtype": "int32", "shape": [rows],
                "strides_elements": [1], "offset_elements": 0, "values": ids, "mutated": False,
                "global_widen_or_reshape_copy": False},
            "weight": {"module": "model.embed_tokens", "dtype": "bf16", "shape": [shard_rows, hidden],
                "read_only": True, "contiguous": True, "rank_local_shard": True},
            "output": output, "fresh_output": True, "aliases_inputs_or_weight": False,
            "all_output_elements_written": True, "unowned_rows_written_zero": True,
            "empty_initialization_write_bytes": 0, "source_tensor_temporaries": [],
            "global_mask_or_shift_tensors": False, "consumer": consumer,
            "output_lifetime": "through TP reduction input read",
            "scope": "fused CUDA source tensor objects only; collective internals, compiler spills and allocator/runtime separate"}
        all_partials = [phase + ":token-embedding" + (f":tp-rank-{lane}" if lane else "") for lane in range(4)]
        ports = [{"producer": parent, **output} for parent in all_partials]
        for parent, port in zip(all_partials, ports, strict=True):
            by_id[parent]["attributes"]["source_tensor_outputs"] = [{name: value for name, value in port.items() if name != "producer"}]
        by_id[consumer]["attributes"]["source_tensor_inputs"] = ports
        record["attributes"]["source_dataflow"] = flow
        record["source_alias_mode"] = "embedding_objects"
        record["source_alias_account"] = {
            "declaration": flow, "direct_input_producers": [], "direct_consumers": [consumer],
            "output_tensor_roles": [output],
            "collective_input_producers": sorted(all_partials), "collective_input_ports": ports,
            "collective_consumer": {"kind": "collective", "operation": "all_reduce", "reduction": "sum",
                "participants": list(range(4)), "bytes": 2 * rows * hidden},
            "lowered_embedding_compatible": True, "collective_reads_after_output": True,
        }


def flashinfer_quantizer_source_program(rows, columns):
    """Independent derivation from scale_1x128 and its BF16 warp helper.

    Eight-lane groups, binary-search overloads and tail groups are not silently
    substituted. The FP8 source template used here has complete 128-value groups.
    """
    if type(rows) is not int or rows < 1 or type(columns) is not int or columns < 1 or columns % 128:
        raise ValueError("independent scale_1x128 source requires full groups")
    groups = rows * (columns // 128)
    stride = ((rows + 3) // 4) * 4
    return {
        "typed_steps": ["bf16_abs_and_local_max", "fp32_warp_max_then_bf16_round_each_step",
                        "broadcast_lane0_bf16_amax", "fp32_conditional_scale",
                        "bf16_to_fp32_value_multiply", "e4m3_cast_store", "fp32_inverse_scale_store"],
        "source_expression_counts": {
            "bf16_abs": rows * columns, "bf16_local_max": groups * 32 * 2 * 2,
            "fp32_warp_max": groups * 32 * 5, "amax_zero_comparisons": groups * 32,
            "fp32_value_multiply": rows * columns, "fp32_inverse_scale_divide": groups},
        "forward_scale_divide": {"predicate": "bf16_amax != bf16(0)", "true_expressions_per_group": 32,
                                 "false_expressions_per_group": 0, "true_group_count": None,
                                 "true_expression": "448.f / fp32(amax)", "false_value": "1.f"},
        "scale_layout": {"dtype": "fp32", "logical_shape": [columns // 128, rows],
                         "strides_elements": [stride, 1], "logical_written_elements": groups,
                         "required_address_span_elements": (columns // 128 - 1) * stride + rows,
                         "padded_extent_elements": columns // 128 * stride,
                         "padding_written_by_quantizer": False, "actual_allocation_capacity_bound": False},
        "scope": "full-group SDK source expressions and logical scale addressing; not compiler issue counts or provider dispatch",
    }


def c128_metadata_width_reference(max_seq_len, source_max_model_len):
    """Source active next-power-of-two width, capped by constructor capacity."""
    if type(max_seq_len) is not int or max_seq_len <= 0:
        raise ValueError("independent C128 width needs positive sequence extent")
    count = max_seq_len // 128
    power = 1
    while power < max(1, count):
        power *= 2
    upper = max(128, power)
    if source_max_model_len is None:
        minimum_states = (max_seq_len + 127) // 128
        lower = min(upper, ((minimum_states + 127) // 128) * 128)
        if lower != upper:
            raise ValueError("independent C128 source requires explicit max_model_len for a unique width")
        return upper
    if type(source_max_model_len) is not int or source_max_model_len < max_seq_len:
        raise ValueError("independent source max_model_len must cover the full batch")
    maximum_states = (source_max_model_len + 127) // 128
    return min(upper, ((maximum_states + 127) // 128) * 128)


def _prefill_indices_reference(raw, requests, ratio, compressed_pairs, swa_pairs, *, c128_width=None):
    """Independent global materializations from the pure-text helper source."""
    tokens = sum(request["query"] for request in requests)
    if ratio == 1:
        topk = 0
    elif ratio == 4:
        topk = raw["index_topk"]
    elif ratio == 128:
        topk = (c128_metadata_width_reference(max(request["computed"] + request["query"] for request in requests), None)
                if c128_width is None else c128_width)
    else:
        raise ValueError("unknown source compression ratio")
    capacity = ((topk + raw["sliding_window"] + 127) // 128) * 128
    active = compressed_pairs + swa_pairs
    metadata = len(requests) * 256 * 20
    writes = 4 * tokens * capacity + 4 * active + 4 * tokens
    reads = metadata + 4 * compressed_pairs + 4 * active + 4 * tokens
    return {
        "source": "vllm/models/deepseek_v4/common/ops/cache_utils.py:combine_topk_swa_indices/CombineTopkSwaIndicesKernel.kernel",
        "combined_capacity_per_query": capacity,
        "active_entries": {"compressed": compressed_pairs, "swa": swa_pairs},
        "fill": {"dtype": "int32", "value": -1, "write_bytes": 4 * tokens * capacity},
        "combine": {"metadata_dtype": "int32", "workers_per_request": 256,
                    "metadata_read_role_bytes": metadata, "input_topk_read_bytes": 4 * compressed_pairs,
                    "active_indices_write_bytes": 4 * active, "lengths_write_bytes": 4 * tokens},
        "attention_consumer": {"indices_dtype": "int32", "indices_read_bytes": 4 * active,
                               "lengths_dtype": "int32", "lengths_read_bytes": 4 * tokens},
        "source_read_role_bytes": reads, "source_write_role_bytes": writes,
        "dependencies": ["fill_all_combined_indices", "combine_topk_and_swa", "flash_mla_reads_indices_and_lengths"],
        "indices_fill_and_active_store_alias": True,
        "integer_control_and_actual_launch_services_complete": False,
        "scope": "pure-text causal all-valid source program; fill and overwrite are both global writes, compiler/native traffic and integer issue service separate"}


def _decode_c4_indices_reference(raw, requests, *, all_completed_states):
    """Independent full-width helper uses, not attention/provider completion."""
    tokens = sum(request["query"] for request in requests)
    capacity = raw["index_topk"]
    nonnegative = sum(min(capacity, position // 4) for request in requests
                      for position in range(request["computed"] + 1, request["computed"] + request["query"] + 1))
    local_bytes, length_bytes = tokens * capacity * 4, tokens * 4
    exact_count = nonnegative if all_completed_states else None
    block_reads = nonnegative * 4 if all_completed_states else None
    return {
        "source": "vllm/models/deepseek_v4/common/ops/cache_utils.py:compute_global_topk_indices_and_lens/ComputeGlobalTopkIndicesAndLensKernel.kernel",
        "condition": "C4 decode; pure text; all scheduled tokens valid; exact short-context or explicit scored-capacity projection",
        "tokens": tokens, "topk_capacity": capacity, "nonnegative_input_entries": exact_count,
        "nonnegative_input_entry_capacity": nonnegative,
        "producer": {"local_indices_dtype": "int32", "local_indices_read_bytes": local_bytes,
                     "token_to_request_dtype": "int32", "token_to_request_read_bytes": length_bytes,
                     "is_valid_token_dtype": "bool", "is_valid_token_read_bytes": tokens,
                     "block_table_dtype": "int32", "block_table_read_role_bytes": block_reads,
                     "block_table_read_role_bytes_per_nonnegative_entry": 4,
                     "global_indices_dtype": "int32", "global_indices_write_bytes": local_bytes,
                     "lengths_dtype": "int32", "lengths_write_bytes": length_bytes},
        "views": {"global_indices_shape": [tokens, capacity], "global_indices_strides_elements": [capacity, 1],
                  "consumer_shape": [tokens, 1, capacity], "consumer_view_allocates": False,
                  "local_and_global_indices_alias": False, "lengths_are_fresh": True,
                  "empty_allocations_are_zero_initialized": False},
        "source_program": {"slot": "block_table[request, local_index // block_size] * block_size + local_index % block_size",
                           "negative_index_output": -1, "block_table_load_mask": "offset < topk and local_index >= 0",
                           "invalid_token_masks_only_final_length": True,
                           "padded_source_lane_extent": tokens * ((capacity + 1023) // 1024) * 1024,
                           "nonnegative_comparisons_per_padded_source_lane": 1,
                           "length": "count(local_index >= 0) if is_valid_token else 0"},
        "attention_consumer": {"indices_dtype": "int32", "indices_read_bytes": 4 * nonnegative,
                               "lengths_dtype": "int32", "lengths_read_bytes": length_bytes},
        "source_read_role_bytes": local_bytes + 4 * nonnegative + 2 * length_bytes + tokens + (0 if block_reads is None else block_reads),
        "source_write_role_bytes": local_bytes + length_bytes,
        "scored_block_table_reads_priced": False,
        "consumer_index_read_scope": "existing causal selected-capacity projection; not observed IDs",
        "dependencies": ["map_full_local_index_rows_and_count_lengths", "flash_mla_reads_global_indices_and_lengths"],
        "actual_selected_ids_or_page_table_bound": False, "integer_control_or_native_services_complete": False,
        "scope": "known C4 full-width materialization plus existing candidate-capacity consumer projection; scored page-table reads remain symbolic, not zero or complete attention service"}


def _decode_attention_index_inputs_reference(raw, requests, ratio, capacity):
    """Independent active-prefix payload for both decode attention tiers."""
    tokens = sum(request["query"] for request in requests)
    swa = sum(min(raw["sliding_window"], position) for request in requests
              for position in range(request["computed"] + 1, request["computed"] + request["query"] + 1))
    compressed = (sum(min(capacity, position // ratio) for request in requests
        for position in range(request["computed"] + 1, request["computed"] + request["query"] + 1)) if ratio > 1 else 0)
    uses = [("swa_indices", swa), ("swa_lengths", tokens)]
    if ratio > 1:
        uses += [("compressed_indices", compressed), ("compressed_lengths", tokens)]
    operands = [{"role": name, "dtype": "int32", "elements": count, "bytes": 4 * count, "access": "read"}
                for name, count in uses]
    return {"operands": operands, "source_logical_consumer_read_bytes": sum(operand["bytes"] for operand in operands),
        "already_in_c4_materialization_consumer_bytes": 4 * (compressed + tokens) if ratio == 4 else 0,
        "indices_shared_across_query_heads": True, "lengths_shared_across_query_heads": True,
        "selected_ids_or_actual_callee_loads_bound": False,
        "scope": "all-valid causal active-prefix input payload projection; not native loads, cache traffic or complete attention work"}


def _prefill_gather_reference(raw, requests, ratio, *, c128_width=None):
    """Independent useful dequant work and both source backend alternatives."""
    width, rope = raw["head_dim"], raw["qk_rope_head_dim"]
    nope = width - rope
    gathered_compressed = sum((r["computed"] + r["query"]) // ratio for r in requests) if ratio > 1 else 0
    gathered_swa = sum(r["query"] + min(r["computed"], raw["sliding_window"] - 1) for r in requests)
    rows = gathered_compressed + gathered_swa
    selected = sum(min(raw["index_topk"], p // 4) if ratio == 4 else p // ratio
                   for r in requests for p in range(r["computed"] + 1, r["computed"] + r["query"] + 1)) if ratio > 1 else 0
    swa = sum(min(raw["sliding_window"], p) for r in requests
              for p in range(r["computed"] + 1, r["computed"] + r["query"] + 1))
    heads = 64 if raw["num_attention_heads"] // 4 <= 64 else 128
    data_bytes, blocks, physical_scales = nope + rope * 2, nope // 64, width // 64
    indices = _prefill_indices_reference(raw, requests, ratio, selected, swa, c128_width=c128_width)
    tokens = sum(request["query"] for request in requests)
    # The declared baseline projects packed-record footprints and K/V
    # mathematical workspace role reads. Native reloads and gather helper
    # metadata remain outside this scoped component check.
    baseline_bytes = (4 * tokens * heads * width + rows * (data_bytes + physical_scales)
        + 2 * rows * width + 4 * width * (selected + swa) * (heads // 64)
        + indices["source_read_role_bytes"] + indices["source_write_role_bytes"] + 4 * heads)
    return {"gathered_rows": {"compressed": gathered_compressed, "swa": gathered_swa},
        "pair_capacities": {"compressed": selected, "swa": swa},
        "common_useful_fp_multiply_values": rows * nope,
        "active_qk_pv_tensor_flops": (selected + swa) * heads * width * 4,
        "workspace": {"dtype": "bf16", "written_elements": rows * width, "write_bytes": rows * width * 2,
                      "score_and_value_alias_same_storage": True, "separate_k_and_v_allocations": False},
        "backend_alternatives": [
            {"condition": "has_cutedsl()", "backend": "cutedsl",
             "input_record_bytes": rows * (data_bytes + physical_scales), "scale_bytes_loaded": rows * physical_scales,
             "source_multiply_dtype": "bf16", "source_multiply_values": rows * width,
             "overwritten_tail_multiply_values": rows * rope, "scale_exp2_values": 0,
             "scale_construction": "UE8M0 integer bits into BF16x2"},
            {"condition": "not has_cutedsl()", "backend": "triton",
             "input_record_bytes": rows * (data_bytes + blocks), "scale_bytes_loaded": rows * blocks,
             "source_multiply_dtype": "fp32", "source_multiply_values": rows * nope,
             "overwritten_tail_multiply_values": 0, "scale_exp2_values": rows * blocks,
             "scale_construction": "FP32 exp2(encoded_scale-127)"}],
        "source_operation_order": (["gather_compressed"] if ratio > 1 else [])
            + ["gather_swa", "combine_topk_swa_indices", "flash_mla_sparse_fwd"],
        "selected_backend": None, "numerical_equivalence_asserted": False,
        "index_materialization": indices, "baseline_logical_bytes": baseline_bytes}


def _indexer_q_backend_reference(tokens, heads, width, rotary):
    """Two CUDA source programs, never one inferred provider or cost sum.

    CuTe uses BF16x2 abs/max and exponent-bit construction. Triton widens
    rounded Q to FP32 and computes div_rn/log2/ceil/exp2. These counts cover
    useful live scalar arithmetic, excluding duplicated/padded lane work.
    """
    rows = tokens * heads
    triton = {"fp32_multiply": rows * (2 * rotary + 3), "fp32_add_subtract": rows * rotary,
              "fp32_divide": rows * (width + 1), "fp32_abs": rows * width,
              "fp32_maximum": rows * width, "fp32_log2": rows, "fp32_ceil": rows,
              "fp32_exp2": rows, "bf16_rotary_roundtrips": rows * rotary,
              "fp8_conversions": rows * width}
    cute = {"fp32_multiply": rows * (2 * rotary + width + 3), "fp32_add_subtract": rows * rotary,
            "fp32_divide": 0, "fp32_log2": 0, "fp32_ceil": 0, "fp32_exp2": 0,
            "bf16_rotary_roundtrips": rows * rotary, "fp8_conversions": rows * width}
    return {"mathematical_primitives": triton,
        "mathematical_primitives_condition": "CUDA and not has_cutedsl(); Triton FP8 branch",
        "backend_alternatives": [
            {"backend": "cutedsl", "condition": "CUDA and has_cutedsl(); FP8 indexer Q",
             "source": "vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py:IndexerQFp8Kernel.kernel",
             "mathematical_primitives": cute,
             "absmax_dtype": "bf16x2 reduction, then FP32 pair max and epsilon clamp",
             "scale_construction": "FP32 epsilon max times reciprocal448; integer mantissa bump/exponent bits",
             "quantization": "FP32 multiply by bit-constructed inverse scale",
             "weight_expression": "(weights * FP32(host_softmax_scale * host_head_scale)) * q_scale",
             "host_scale_multiplications_per_call": 1},
            {"backend": "triton", "condition": "CUDA and not has_cutedsl(); FP8 indexer Q",
             "source": "vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:FusedIndexerQRopeQuantTritonKernel.kernel",
             "mathematical_primitives": triton,
             "absmax_dtype": "fp32 after BF16 rotary roundtrip",
             "scale_construction": "FP32 div_rn followed by log2/ceil/exp2",
             "quantization": "FP32 div_rn per output value",
             "weight_expression": "((weights * q_scale) * softmax_scale) * head_scale",
             "host_scale_multiplications_per_call": 0}],
        "selected_backend": None, "numerical_equivalence_asserted": False}


def _indexer_q_roofline_reference(tokens):
    """Fixed Flash indexer geometry, ideal arithmetic model, no GPU issue rate.

    The two source programs have different useful arithmetic, but the H200
    arithmetic/byte baseline is memory dominated for each. Native div.rn,
    comparisons, conversions and integer work are not proven by that fact.
    """
    groups = tokens * 64
    def demand(arithmetic, special):
        return {"compute_tensor": 0, "compute_vector": 0, "compute_reduction": 0,
                "compute_fma": arithmetic, "compute_sfu": special, "compute_alu": 0}
    return {"logical_bytes": tokens * (2 * 64 * 128 + 2 * 64 + 8 + 4 * 64 + 64 * 128 + 4 * 64),
        "source_programs": {
            "arithmetic_definition": "one FLOP per FP32 add/subtract/multiply/divide; log2 and exp2 are generic SFU groups, not native instructions",
            "demands_by_backend": {"cutedsl": demand(groups * (3 * 64 + 128 + 3), 0),
                                   "triton": demand(groups * (3 * 64 + 128 + 4), 2 * groups)},
            "selected_backend": None,
            "comparison_conversion_rounding_and_integer_issue_priced": False,
            "actual_provider_bound": False, "all_source_branches_memory_dominated": True},
        "scope": "common ideal mathematical FLOPs/bytes result only when every CUDA source alternative is memory dominated; not a hardware latency bound or native instruction model"}


def clamped_activation_source_reference(root):
    """Bind scalar/packed act_first+clamp expressions without issuing GPU work.

    Compare complete specialized expression bodies, so an additional math
    statement cannot pass merely because older fragments remain present.
    No scalar/packed numerical equivalence or runtime selection is assumed.
    """
    relative = "csrc/libtorch_stable/activation_kernels.cu"
    symbols = ("__device__ __forceinline__ scalar_t compute(", "__device__ __forceinline__ T silu_kernel(",
        "__device__ __forceinline__ packed_t packed_compute(", "__device__ __forceinline__ packed_t packed_silu_kernel(")
    bodies = (
        "scalar_t gate=x;scalar_t up=y;gate=(scalar_t)fminf((float)gate,limit);"
        "up=(scalar_t)fmaxf(fminf((float)up,limit),-limit);"
        "return(scalar_t)(ACT_FN(gate,alpha)*((float)up+beta));",
        "return(T)(((float)x)/(1.0f+expf((float)-x*alpha)));",
        "packed_t gate=x;packed_t up=y;float2 u=cast_to_float2(up);float2 g=cast_to_float2(gate);"
        "g.x=fminf(g.x,limit);g.y=fminf(g.y,limit);u.x=fmaxf(fminf(u.x,limit),-limit);"
        "u.y=fmaxf(fminf(u.y,limit),-limit);gate=cast_to_packed<packed_t>(g);"
        "float2 activated=cast_to_float2(PACKED_ACT_FN(gate,alpha));activated.x*=u.x+beta;"
        "activated.y*=u.y+beta;return cast_to_packed<packed_t>(activated);",
        "float2 fval=cast_to_float2(val);fval.x=fval.x/(1.0f+expf(-fval.x*alpha));"
        "fval.y=fval.y/(1.0f+expf(-fval.y*alpha));return cast_to_packed<packed_t>(fval);",
    )
    forms = []
    for symbol, expected in zip(symbols, bodies, strict=True):
        form = _cpp_anchor(Path(root), relative, symbol)
        text = _constexpr_body(_constexpr_body(form["source"], "act_first", True), "HAS_CLAMP", True)
        body = text[text.index("{") + 1:text.rindex("}")]
        if re.sub(r"\s+", "", body) != re.sub(r"\s+", "", expected):
            raise ValueError("clamped activation typed source arithmetic changed: " + symbol)
        forms.append(form)
    host = _cpp_anchor(Path(root), relative, "void silu_and_mul_clamp(")
    if "LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel,vllm::packed_silu_kernel,true,true,(float)limit,(float)alpha,(float)beta)" not in re.sub(
            r"\s+", "", re.sub(r"/\*.*?\*/|//[^\n]*", "", host["source"], flags=re.S)):
        raise ValueError("clamped activation host no longer selects act_first and HAS_CLAMP")
    return {**forms[0], "related_definitions": [forms[1], host, forms[2], forms[3]]}


def clamped_activation_arithmetic_reference(elements, dtype):
    """Independent scalar/packed source counts; no instruction proxy rates."""
    return {
        "mathematical_primitives": {"fp32_multiply": elements * 2, "fp32_add": elements * 2,
            "fp32_divide": elements, "fp32_exp": elements, "sign_negate": elements * 2,
            "fp32_minmax": elements * 3},
        "baseline_demand": {"compute_fma": elements * 5, "compute_alu": 0, "compute_sfu": elements},
        "source_negation_programs": {
            "scalar": {"value_dtype": dtype, "value_negations": elements,
                       "limit_dtype": "fp32", "limit_negations": elements},
            "packed": {"value_dtype": "fp32", "value_negations": elements,
                       "limit_dtype": "fp32", "limit_negations": elements}, "selected_form": None},
        "unpriced_source_actions": ["typed sign negations", "FP32 fminf/fmaxf",
            "typed conversions", "compiler division lowering and instruction services"],
    }


def _marlin_moe_reference(raw, tokens, tp):
    """Live matrix/activation/reduce work independent of routing values."""
    hidden, width, topk = raw["hidden_size"], raw["moe_intermediate_size"] // tp, raw["num_experts_per_tok"]
    if raw["moe_intermediate_size"] % tp or hidden % 64 or width % 32:
        raise ValueError("independent V4 Marlin reference requires exact TP partition and source-admissible N64/K32 matrices")
    routes, values = tokens * topk, tokens * topk * width
    activation = clamped_activation_arithmetic_reference(values, "bf16")

    def use(role, dtype, count, access, stage):
        return {"role": role, "dtype": dtype, "elements": count,
                "bytes": count * (4 if dtype == "fp32" else 2), "access": access, "stage": stage}

    request0 = 2 * max(routes * max(width, hidden), tokens * hidden)
    request1 = 2 * routes * max(2 * width, hidden)
    offset1 = ((request0 + 255) // 256) * 256
    reference = {
        "source_condition": "CUDA modular Marlin W4A16, EP-off, no LoRA, SILU with clamp; input_dtype=None/BF16, apply_router_weight_on_input=False; declared valid INT32 topk_ids, no token mask",
        "routed_rows": routes,
        "mathematical_primitives": {
            "useful_w13_tensor_flops": routes * hidden * width * 4,
            "useful_w2_tensor_flops": routes * width * hidden * 2,
            "activation_fp32_multiply": values * 2, "activation_fp32_add": values * 2,
            "activation_fp32_divide": values, "activation_fp32_exp": values,
            "activation_sign_negate": values * 2, "activation_fp32_minmax": values * 3,
            "activation_intermediate_bf16_rounding_values": values,
            "router_weight_bf16_multiply": routes * hidden,
            "minimal_topk_fp32_add": tokens * hidden * (topk - 1),
            "source_topk_fp32_add": routes * hidden},
        "activation_resource_groups": activation["baseline_demand"],
        "activation_negation_programs": activation["source_negation_programs"],
        "activation_unpriced_source_actions": activation["unpriced_source_actions"],
        "router_weight_source_program": {
            "operation": "multiply", "input_dtype": "bf16", "weight_input_dtype": "fp32",
            "weight_converted_to": "bf16", "output_dtype": "bf16",
            "mathematical_flops": routes * hidden, "source_packed_lanes": 2,
            "mapped_resource_role": None, "native_issue_service_priced": False,
            "scope": "typed BF16 mathematical multiply; conversion, rounding and packed issue service not priced by INT32 or inferred FP32 throughput"},
        "dequant_source_program": {
            "condition": "BF16/MXFP4 E2M1 with E8M0 group32 scales; dequant_skip_flop=False; no zero point",
            "host_scale_preparation": "checkpoint E8M0 -> params_dtype BF16 -> Marlin permutation -> E8M0",
            "host_scale_group4_permutation": [0, 2, 1, 3], "host_preparation_is_per_forward_work": False,
            "packed_fragment_prepare": "b_quant_1=q; b_quant_0=q<<8",
            "dequant_call_nibble_offsets": [8, 24, 12, 28],
            "weight_bit_placement": "bf16_bits = ((nibble & 8) << 12) | ((nibble & 7) << 6)",
            "weight_bias_multiply": {"dtype": "bf16", "packed_lanes": 2, "bias_power_of_two": 126},
            "group_scale_decode": {"storage_dtype": "e8m0", "output_dtype": "bf16", "byte_to_bits": "scale_byte << 7",
                "source_word_byte_order": [0, 2, 1, 3],
                "exceptional_bytes": [
                    {"byte": 0, "source_value": "zero", "generic_E8M0_value": "2^-127"},
                    {"byte": 255, "source_value": "positive_infinity", "generic_E8M0_value": "NaN"}]},
            "group_scale_multiply": {"dtype": "bf16", "packed_lanes": 2},
            "ordered_value_program": ["integer bit placement", "BF16 bias multiply", "BF16 group-scale multiply", "BF16 MMA operand"],
            "per_weight_value_bf16_multiply_sites": 2, "scale_decode_values_per_unique_group_scale": 8,
            "global_dequant_tensor_materializations": [], "fragment_storage": "thread-local FragB/FragS values",
            "generic_MXFP4_numerical_equivalence_asserted": False,
            "checkpoint_values_or_BF16_intrinsic_lowering_bound": False,
            "mapped_resource_role": None, "native_issue_service_priced": False},
        "operands": [
            use("hidden_row_uses", "bf16", routes * hidden, "read", "w13"),
            use("gate_up", "bf16", 2 * values, "write", "w13"),
            use("gate_up", "bf16", 2 * values, "read", "activation"),
            use("activated", "bf16", values, "write", "activation"),
            use("activated", "bf16", values, "read", "w2"),
            use("selected_weights", "fp32", routes, "read", "w2"),
            use("weighted_expert_output", "bf16", routes * hidden, "write", "w2"),
            use("weighted_expert_output", "bf16", routes * hidden, "read", "reduce"),
            use("reduced_output", "bf16", tokens * hidden, "write", "reduce")],
        "weight_payload_per_active_expert": {
            "packed_e2m1_dtype": "int32", "packed_e2m1_bytes": hidden * width * 3 // 2,
            "group_size": 32, "scale_dtype": "fp8_e8m0", "scale_element_bytes": 1,
            "scale_elements": (hidden * 2 * width + width * hidden) // 32},
        "checkpoint_repack_source_program": {
            "condition": "BF16 activations, E2M1 packed checkpoint rows, group32; N%64=0,K%32=0; contiguous final checkpoint dimension",
            "checkpoint_weight_dtype": "uint8", "checkpoint_scale_storage_dtype": "uint8",
            "prepared_weight_dtype": "int32", "prepared_scale_dtype": "fp8_e8m0",
            "values_per_prepared_word": 8, "weight_value_bits": 4,
            "host_weight_view": "weight[expert].view(int32).T.contiguous()",
            "repack_tile_shape_N_K": [64, 16], "prepared_words_per_tile": 128,
            "word_offset": "(tile_k*(N//64)+tile_n)*128+lane*4+warp",
            "source_n": "tile_n*64+warp*16+lane//4+((j//2)%2)*8",
            "source_k": "tile_k*16+2*(lane%4)+j//4+(j%2)*8",
            "coordinate_domain": "warp in[0,4),lane in[0,32),word nibble j in[0,8)",
            "scale_reorder_l": "(p//4)*4+[0,2,1,3][p%4]",
            "scale_transposed_index_grouped64": "(l//64)*64+(l%64)//8+8*(l%8)",
            "scale_transposed_index_single32": "(l//32)*32+2*((l%32)//8)+(l%2)+8*((l%8)//2)",
            "scale_checkpoint_coordinates": "n=t%N,group_k=t//N",
            "matrices": [{"stage": stage, "N": n, "K": k,
                "checkpoint_weight_shape": [raw["n_routed_experts"], n, k // 2],
                "host_qweight_shape": [k // 8, n],
                "prepared_weight_shape": [raw["n_routed_experts"], k // 16, 2 * n],
                "source_prepared_weight_tensor_bytes": raw["n_routed_experts"] * n * k // 2,
                "checkpoint_scale_shape": [raw["n_routed_experts"], n, k // 32],
                "prepared_scale_shape": [raw["n_routed_experts"], k // 32, n],
                "scale_base_permutation": "grouped64" if k > 32 else "single32"}
                for stage, n, k in (("w13", 2 * width, hidden), ("w2_with_router_weight", hidden, width))],
            "expert_views_share_final_tensor": True, "expert_regions_disjoint": True,
            "final_weight_aliases_checkpoint_or_per_expert_result": False,
            "preparation_is_per_forward_work": False, "added_forward_payload_bytes": 0,
            "MMA_fragment_reader_or_actual_checkpoint_values_bound": False},
        "weight_reader_source_program": {
            "condition": "BF16/E2M1/E8M0 group32 source, contiguous prepared parameters and valid expert/slice indices",
            "copy_unit_bytes": 16, "packed_weight_tensor_dtype": "int32", "scale_tensor_dtype": "fp8_e8m0",
            "configurations_TK_TN_threads": [[128, 128, 256], [64, 256, 256], [64, 128, 128], [128, 64, 128]],
            "iterations": "I=TN*TK//(32*threads)",
            "global_B_int4_index": "expert*N*K//32+(slice_row*TK//16+stage_row)*(N//2)+slice_col*(TN//2)+stage_col",
            "shared_B_int4_read": "(TN//2)*q+tid+(tid//(TN//2))*((TN//2)*(I-1))",
            "warp_n": "(tid//32)%(TN//64)", "warp_k": "(tid//32)//(TN//64)",
            "subtile_n": "slice_col*TN+warp_n*64+j*16", "subtile_k": "slice_row*TK+16*(warp_k*I+q)",
            "regular_weight_B_n": "subtile_n+8*half+lane//4",
            "regular_weight_B_k": "subtile_k+2*(lane%4)+(v%2)+8*(v//2)",
            "M8_weight_A_n": "subtile_n+lane//4+8*((v//2)%2)",
            "M8_weight_A_k": "subtile_k+2*(lane%4)+(v%2)+8*(v//4)",
            "M8_weight_A_word_order": [["frag_b0", 0], ["frag_b1", 0], ["frag_b0", 1], ["frag_b1", 1]],
            "MMA_opcode": "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            "weight_operand_dtype": "bf16", "accumulator_dtype": "fp32",
            "operand_layout_source": "CUTLASS SM80_16x8x16_F32BF16BF16F32_TN A/B traits; same source PTX opcode",
            "packed_read_bytes_per_matrix_M_block": "N*K//2", "scale_read_bytes_per_matrix_M_block": "N*K//32",
            "prefetch_count": "min(R,S-1)+max(R-S+1,0)=R for assigned K tiles R>=0 and stages S>=2",
            "selected_configuration_or_GPU_order_bound": False,
            "activation_ldsm_output_or_numerical_MMA_bound": False,
            "cache_or_physical_DRAM_transactions_bound": False},
        "semantic_fields": {
            "activation_result_rounding_dtype": "bf16", "rounding_before_up_multiply": True,
            "alpha": 1.0, "beta": 0.0, "clamp_limit": raw["swiglu_limit"],
            "router_weight_applied_on_input": False, "router_weight_multiply_dtype": "bf16",
            "router_weight_converted_from_fp32_before_multiply": True,
            "w2_result_rounding_before_router_weight": "bf16",
            "topk_accumulator_dtype": "fp32", "topk_accumulator_initial_value": 0.0,
            "topk_source_add_includes_zero_seed": True,
            "reduced_output_dtype": "bf16",
            "activation_intermediate_is_global_allocation": False},
        "source_views": [
            {"role": "gate_up", "base": "intermediate_cache13", "shape": [routes, 2 * width], "new_allocation": False},
            {"role": "weighted_expert_output", "base": "intermediate_cache13", "shape": [routes, hidden], "new_allocation": False},
            {"role": "activated", "base": "intermediate_cache2", "shape": [routes, width], "new_allocation": False}],
        "modular_buffer_program": {
            "condition": "FusedMoEKernelModularImpl.apply; Standard contiguous 2D BF16 input; EP-off/no LoRA; source output alias matches shape/dtype/device",
            "workspace_dtype": "bf16", "workspace_manager_backing_dtype": "uint8",
            "simultaneous_requests_share_backing": True, "request_alignment_bytes": 256,
            "requests": [
                {"role": "common_workspace", "offset_bytes": 0, "elements": request0 // 2, "bytes": request0},
                {"role": "workspace2", "offset_bytes": offset1, "elements": request1 // 2, "bytes": request1}],
            "requested_workspace_bytes": offset1 + (request1 + 255) // 256 * 256,
            "requested_bytes_are_peak_capacity_or_traffic": False,
            "view_regions": [
                {"role": "activated", "offset_bytes": 0, "bytes": routes * width * 2,
                 "shape": [routes, width], "strides_elements": [width, 1]},
                {"role": "gate_up", "offset_bytes": offset1, "bytes": routes * width * 4,
                 "shape": [routes, 2 * width], "strides_elements": [2 * width, 1]},
                {"role": "weighted_expert_output", "offset_bytes": offset1, "bytes": routes * hidden * 2,
                 "shape": [routes, hidden], "strides_elements": [hidden, 1]},
                {"role": "provisional_fused_out", "offset_bytes": 0, "bytes": tokens * hidden * 2,
                 "shape": [tokens, hidden], "strides_elements": [hidden, 1]}],
            "resize_flatten_slice_view_allocates": False,
            "gate_up_and_weighted_expert_output_overlap": True,
            "gate_up_read_finishes_before_weighted_output_write": True,
            "workspace13_used_as_intermediate_cache2": True,
            "workspace2_used_as_intermediate_cache13": True,
            "provisional_fused_out_replaced_before_write": True,
            "output": {"dtype": "bf16", "shape": [tokens, hidden], "fresh_empty_like_hidden_states": True,
                       "aliases_workspace_or_input": False, "source_finalizer_receives_same_output_object": True},
            "empty_initialization_write_bytes": 0, "finalize_self_copy_bytes": 0,
            "actual_workspace_lane_capacity_addresses_or_GPU_order_bound": False,
            "scope": "source buffer requests/views and serial overwrite order; routing buffers, FP32 GEMM scratch, dequant temporaries, concurrent lane ownership and runtime remain separate"},
        "source_stage_order": ["w13", "activation", "w2_with_router_weight", "topk_reduce"],
        "declared_routing_is_checkpoint_assignment": False,
        "actual_provider_bound": False, "actual_workspace_capacity_bound": False}
    arithmetic = reference["mathematical_primitives"]
    stage_math = (
        {"compute_tensor": arithmetic["useful_w13_tensor_flops"]},
        reference["activation_resource_groups"],
        {"compute_tensor": arithmetic["useful_w2_tensor_flops"]},
        {"compute_fma": arithmetic["source_topk_fp32_add"]})
    weights = ((hidden * width, hidden * 2 * width // 32), (0, 0),
               (hidden * width // 2, width * hidden // 32), (0, 0))
    templates = []
    for name, operand_stage, math_work, (packed, scales) in zip(reference["source_stage_order"],
            ("w13", "activation", "w2", "reduce"), stage_math, weights):
        demand = dict.fromkeys(("compute_tensor", "compute_vector", "compute_reduction",
                              "compute_fma", "compute_alu", "compute_sfu"), 0)
        demand.update(math_work)
        typed = [item for item in reference["operands"] if item["stage"] == operand_stage]
        templates.append({"stage": name, "demand": demand,
            "unpriced_typed_math": ([reference["router_weight_source_program"]]
                if name == "w2_with_router_weight" else []),
            "nonweight_read_write_bytes": {access: sum(item["bytes"] for item in typed if item["access"] == access)
                                           for access in ("read", "write")},
            "weight_payload_per_active_expert_bytes": {"packed_e2m1": packed, "fp8_e8m0_scales": scales}})
    reference["roofline_stage_templates"] = templates
    # Independent wrapper allocation geometry. E=256 selects the two-call
    # general CUDA program even for a single token; small M alone does not
    # select the <=64-expert specialization.
    experts = raw["n_routed_experts"]
    block = next((tile for tile in (8, 16, 32, 48, 64)
                  if tokens * topk / experts / tile < 0.9), 64)
    sorted_capacity = routes + experts * (block - 1)
    if routes < experts:
        sorted_capacity = min(sorted_capacity, routes * block)
    expert_capacity = (sorted_capacity + block - 1) // block
    small = routes < 1024 and experts <= 64
    reference["routing_alignment"] = {"block_size_m": block, "sorted_ids_capacity": sorted_capacity,
        "expert_ids_capacity": expert_capacity, "post_pad_elements": 1,
        "cumsum_elements": 0 if small else experts + 1,
        "index_dtype": "int32", "sorted_sentinel": routes, "inactive_expert_sentinel": -1,
        "all_topk_ids_valid": True, "expert_map": None, "pad_sorted_ids": False,
        "program": "small_batch_expert" if small else "count_fill_then_sort",
        "source_sort_global_atomic_adds": 0 if small else routes,
        "sorted_initialization_and_overwrite_are_distinct_writes": True,
        "source_allocations_are_actual_live_capacity": False,
        "scope": "logical source payload, including unique prefix-boundary reads and per-route atomic read/modify/write; not compiled loads, atomic service time or DRAM traffic"}
    prefix_work = [("routing_align_small", 8 * routes, 4 * (sorted_capacity + expert_capacity + 1 + routes))] if small else [
        ("routing_count_fill", 4 * (routes + experts + 1), 4 * (sorted_capacity + expert_capacity + experts + 2)),
        ("routing_sort", 8 * routes, 8 * routes)]
    prefix = [{"stage": name, "demand": dict.fromkeys(templates[0]["demand"], 0),
        "unpriced_typed_math": [],
        "nonweight_read_write_bytes": {"read": read, "write": write},
        "weight_payload_per_active_expert_bytes": {"packed_e2m1": 0, "fp8_e8m0_scales": 0}}
        for name, read, write in prefix_work]
    workspace = {"stage": "workspace_zero", "demand": dict.fromkeys(templates[0]["demand"], 0),
        "unpriced_typed_math": [],
        "nonweight_read_write_bytes": {"read": 0, "write": 0},
        "weight_payload_per_active_expert_bytes": {"packed_e2m1": 0, "fp8_e8m0_scales": 0}}
    reference["roofline_stage_templates"] = prefix + [workspace] + templates
    for stage in reference["roofline_stage_templates"]:
        stage["read_write_bytes_per_sm"] = {"read": 0, "write": 16 if stage["stage"] == "workspace_zero" else 0}
        matrix = stage["stage"] in ("w13", "w2_with_router_weight")
        stage["metadata_read_bytes_per_active_m_block"] = (block + 1) * 4 if matrix else 0
        stage["metadata_read_bytes_per_stage"] = 4 if matrix else 0
    reference["gemm_metadata"] = {
        "index_dtype": "int32", "element_bytes": 4,
        "active_m_blocks_rule": "sum(ceil(count_e / block_size_m)) for declared valid expert histogram",
        "sorted_ids_elements_per_active_m_block": block,
        "expert_ids_elements_per_active_m_block": 1, "post_pad_elements_per_gemm": 1,
        "sorted_ids_backing_elements": sorted_capacity, "expert_ids_backing_elements": expert_capacity,
        "sorted_ids_producer": "routing_align_small" if small else "routing_sort",
        "expert_ids_and_post_pad_producer": "routing_align_small" if small else "routing_count_fill",
        "consumers": ["w13", "w2_with_router_weight"],
        "padding_sentinel": routes, "allocation_tail_is_read": False,
        "input_row_mappings": [
            {"stage": "w13", "top_k": topk, "input_rows": tokens, "output_rows": routes,
             "input_row": "sorted_route_id // top_k", "output_row": "sorted_route_id"},
            {"stage": "w2_with_router_weight", "top_k": 1, "input_rows": routes, "output_rows": routes,
             "input_row": "sorted_route_id // top_k", "output_row": "sorted_route_id"}],
        "valid_route_id_domain": [0, routes],
        "valid_rows_count_expression": "count(sorted_route_id < prob_m * top_k) per active m block",
        "activation_read_guard": "row < block_num_valid_tokens", "output_scatter_guard": "row < block_num_valid_tokens",
        "within_expert_order_binding_asserted": False,
        "scope": "unique per-GEMM active-prefix input payload and source row mapping under declared routing; not CTA/split replay, DRAM traffic or checkpoint expert IDs"}
    reference["source_stage_order"] = [stage["stage"] for stage in reference["roofline_stage_templates"]]
    reference["workspace_zero"] = {"condition": "modular Marlin no-LoRA default workspace=None; existing=None",
        "dtype": "int32", "elements_per_sm": 4, "bytes_per_sm": 16, "initial_value": 0,
        "initialization_count_per_moe_call": 1, "shared_by": ["w13", "w2_with_router_weight"],
        "following_lock_accesses_bound": False, "actual_allocator_binding_verified": False}
    reference["fp32_scratch_templates"] = []
    for name, output_columns in (("w13", width * 2), ("w2_with_router_weight", hidden)):
        reference["fp32_scratch_templates"].append({"stage": name, "dtype": "fp32", "element_bytes": 4,
            "allocation_condition": "use_fp32_reduce && !use_atomic_add", "use_fp32_reduce": True, "use_atomic_add": False,
            "sorted_capacity_term_elements": output_columns * sorted_capacity,
            "per_sm_cap_elements": block * 1024, "max_thread_n": 256,
            "m8_allocation_factor": 2 if block == 8 else 1,
            "allocation_rule": "min(sorted_capacity_term_elements, sm_count * per_sm_cap_elements) * m8_allocation_factor",
            "initialization": "empty, not zero-filled", "producer_dtype": "fp32", "consumer_dtype": "fp32",
            "access_condition": "slice_count > 1 && !use_atomic_add",
            "reads_previous_partial_if": "!first", "writes_partial_if": "!last",
            "read_write_bytes": None, "additional_fp32_additions": None,
            "fp32_values_per_n_column_by_valid_rows": [None] + [
                8 if block == 8 else sum(2 * max(0, min(8, live_rows - base))
                    for base in range(0, block, 16)) for live_rows in range(1, block + 1)],
            "partial_access_rule": "V=thread_n*values_per_n_column[valid_rows]; read=4*V if !first else 0; write=4*V if !last else 0; FP32_add=V if !first else 0",
            "thread_n_source_candidates": [64, 128] if block <= 16 else [64, 128, 256],
            "per_slice_active_lane_tail_guard_bound": True,
            "selected_slice_census_bound": False,
            "lifetime": "local tensor of one moe_wna16_marlin_gemm call; W13 and W2 source uses are serial",
            "physical_allocation_or_cross_stage_reuse_bound": False,
            "scope": "requested source extent and conditional typed reduction; extent is not bytes touched or normal capacity"})
    buffers = [("sorted_token_ids", sorted_capacity, "alignment through W2"),
               ("expert_ids", expert_capacity, "alignment through W2"),
               ("num_tokens_post_padded", 1, "alignment through W2")]
    if not small:
        buffers.append(("cumsum", experts + 1, "local to alignment custom-op enqueue"))
    empty_objects = [{"stage": "alignment", "role": "expert_map", "dtype": "int32", "elements": 0,
        "initialization_write_bytes": 0, "payload_access_bytes": 0, "lifetime": "local to the custom-op enqueue"}]
    for name in ("w13", "w2_with_router_weight"):
        for role, dtype in (("a_scales", "fp32"), ("global_scale", "fp32"), ("b_bias", "bf16"), ("b_zeros", "bf16")):
            empty_objects.append({"stage": name, "role": role, "dtype": dtype, "elements": 0,
                "initialization_write_bytes": 0, "payload_access_bytes": 0, "lifetime": "local to the custom-op enqueue"})
    reference["tensor_object_program"] = {
        "condition": "successful CUDA Standard no-DP/EP modular Marlin W4A16; no LoRA, bias, zero-point or input quantization; valid routes",
        "prepare": {"hidden_states_alias": True,
                    "activation_scale": "None or borrowed prepare scale, unused by Marlin W4A16",
                    "topk_weights_alias": True, "topk_ids_alias": True,
                    "expert_tokens_meta": None, "expert_map": None,
                    "apply_router_weight_on_input": False, "global_input_copy_bytes": 0},
        "routing_buffers": [{"role": name, "dtype": "int32", "shape": [count], "fresh": True,
            "initialization": "empty; source alignment kernels initialize consumed regions", "lifetime": lifetime}
            for name, count, lifetime in buffers],
        "zero_extent_objects": empty_objects,
        "storage_program_bindings": ["modular_buffer_program", "workspace_zero", "fp32_scratch_templates"],
        "parameter_program_bindings": ["checkpoint_repack_source_program", "weight_payload_per_active_expert"],
        "GEMM_returns_supplied_C": ["w13", "w2_with_router_weight"],
        "activation_input_view": {"base": "gate_up", "shape": [routes, 2 * width],
            "strides_elements": [2 * width, 1], "offset_elements": 0, "allocates": False},
        "reduction_input_view": {"base": "weighted_expert_output", "shape": [tokens, topk, hidden],
            "strides_elements": [topk * hidden, hidden, 1], "offset_elements": 0, "allocates": False},
        "activation_input_view_allocates": False, "reduction_input_view_allocates": False,
        "output": {"dtype": "bf16", "shape": [tokens, hidden], "fresh_empty_like_hidden_states": True,
                   "aliases_input_parameters_routes_or_workspace": False,
                   "written_by": "topk_reduce", "finalizer_returns_same_object": True},
        "buffer_lifetimes": [
            {"role": "gate_up", "producer": "w13", "last_reader": "activation"},
            {"role": "activated", "producer": "activation", "last_reader": "w2_with_router_weight"},
            {"role": "weighted_expert_output", "producer": "w2_with_router_weight", "last_reader": "topk_reduce"}],
        "FP32_scratch": "fresh per-GEMM local tensor, serial W13/W2 source lifetimes; conditional accesses stay parameterized",
        "register_dequant_global_tensors": [], "finalize_copy_bytes": 0,
        "arena_base_capacity_lane_and_GPU_fences_bound": False,
        "scope": "source tensor identity/views/temporaries/lifetimes; not selected slice traffic, actual allocator or provider"}
    return reference


def _marlin_scratch_reference(source, sm_count):
    """Evaluate host allocation geometry only, independently of traffic/pricing."""
    allocations = []
    for program in source["fp32_scratch_templates"]:
        upper = program["per_sm_cap_elements"] * sm_count
        extent = min(upper, program["sorted_capacity_term_elements"])
        elements = extent * program["m8_allocation_factor"]
        allocations.append({"stage": program["stage"], "requested_elements": elements,
                            "requested_bytes": 4 * elements, "dtype": "fp32"})
    return {"sm_count": sm_count, "sm_count_source": "declared HWSpec, not runtime device observation",
        "stages": allocations, "requested_bytes_counted_as_traffic": False,
        "aggregate_peak_live_bytes": None, "actual_allocator_binding_verified": False,
        "scope": "per-GEMM source requests only; no touched-byte, live allocator capacity or provider acceptance"}


def flashinfer_workspace_reference(rows, columns, output_columns):
    """Independent dense SDK request/offset program, not a runtime choice.

    getWorkspaceSizeBase computes the 32-row extent for the current M,
    whereas its 4-row state is monotone. The BF16/BF16 request dominates
    the binding maximum even though runGemm selects BF16/FP8. The unused
    weight reservation therefore exists without any weight-copy traffic.
    """
    m4 = rows + (-rows % 4)
    m32 = rows + (-rows % 32)
    k_groups = columns // 128
    b_scale_elements = ((output_columns + 127) // 128) * k_groups
    b_reserve = output_columns * columns + 4 * b_scale_elements
    scale_reserve = 4 * m32 * k_groups
    return {
        "condition": "installed FlashInfer SM90 dense BF16 A / FP8 B source branch; no provider selection",
        "state": {"parameter": "H", "definition": "max(prior_max_shape_m_4_align, ceil(M/4)*4)",
                  "prior_domain": "nonnegative multiple of 4", "current_aligned_rows": m4,
                  "H_value": None, "expected_m_used_by_dense_gemm": False},
        "allocation": {"dtype": "uint8", "operation": "torch.empty", "initialized": False,
                       "initialization_write_bytes": 0, "request_bytes": {"H_coefficient": columns,
                           "constant": scale_reserve + b_reserve},
                       "request_rule": "maximum of BF16/BF16, BF16/FP8 and FP8/FP8 requests",
                       "shared_base_for_three_runners": True, "allocator_capacity_bytes": None},
        "runner_request_bytes": {"bf16_bf16": {"H_coefficient": columns, "constant": scale_reserve + b_reserve},
                                 "bf16_fp8": {"H_coefficient": columns, "constant": scale_reserve},
                                 "fp8_fp8": {"H_coefficient": 0, "constant": 0}},
        "activation_view": {"dtype": "fp8", "offset_bytes": 0, "shape": [rows, columns],
                            "strides_elements": [columns, 1], "reserved_bytes": {"H_coefficient": columns, "constant": 0},
                            "quantizer_write_bytes": rows * columns, "matmul_unique_read_bytes": rows * columns,
                            "reserved_padding_written_by_quantizer": False},
        "scale_view": {"dtype": "fp32", "offset_bytes": {"H_coefficient": columns, "constant": 0},
                       "shape": [k_groups, rows], "strides_elements": [m4, 1],
                       "reserved_bytes": scale_reserve, "address_span_bytes": 4 * ((k_groups - 1) * m4 + rows),
                       "quantizer_write_bytes": 4 * k_groups * rows, "matmul_unique_read_bytes": 4 * k_groups * rows,
                       "padding_written_by_quantizer": False},
        "unused_reservation_bytes": b_reserve,
        "borrowed_operands": ["FP8 weight", "FP32 weight scales"],
        "source_lifetime": "fresh Python local workspace is passed to cached runner and used by quantizer then matmul",
        "source_substage_order": ["quantize_bf16_and_write_scales", "packed_fp8_matmul"],
        "dense_matrix_storage": {
            "condition": "SM90 normal dense BF16/FP8 SDK source with ENABLE_FP8_BLOCK_SCALE, COMPILE_HOPPER_TMA_GEMMS and without PLACEHOLDER_KERNELS, after in-call activation quantization",
            "ports": [
                {"role": "A", "storage": "workspace", "dtype": "fp8", "offset_bytes": 0,
                 "shape": [rows, columns], "strides_elements": [columns, 1], "access": "read"},
                {"role": "As", "storage": "workspace", "dtype": "fp32",
                 "offset_bytes": {"H_coefficient": columns, "constant": 0},
                 "shape": [k_groups, rows], "strides_elements": [m4, 1], "access": "read"},
                {"role": "B", "storage": "borrowed_weight", "dtype": "fp8", "offset_bytes": 0,
                 "shape": [output_columns, columns], "strides_elements": [columns, 1], "access": "read"},
                {"role": "Bs", "storage": "borrowed_weight_scales", "dtype": "fp32", "offset_bytes": 0,
                 "shape": [(output_columns + 127) // 128, k_groups], "strides_elements": [k_groups, 1], "access": "read"},
                {"role": "D", "storage": "caller_output", "dtype": "bf16", "offset_bytes": 0,
                 "shape": [rows, output_columns], "strides_elements": [output_columns, 1], "access": "write"}],
            "dispatch_alternatives": [
                {"condition": "getDeepGemmEnabled() == false", "entry": "Fp8Gemm::run"},
                {"condition": "getDeepGemmEnabled() == true and M >= 32", "entry": "deep_gemm::runGemm"},
                {"condition": "getDeepGemmEnabled() == true and M < 32", "entry": "deep_gemm::runGemmSwapAB"}],
            "additional_source_global_tensor_bytes": 0,
            "descriptors": "host values passed to kernel; refer to the five existing tensor ports",
            "grouped_layout_pointer": None,
            "staging_and_barriers": "shared memory, not an additional global tensor",
            "accumulators": "FP32 register fragments, not a global partial-result tensor",
            "H_value_provider_compiler_spills_or_physical_traffic_bound": False},
        "stream_completion_or_allocator_reuse_proven": False, "actual_provider_bound": False,
        "allocation_request_is_traffic": False,
        "scope": "conditional source workspace and normal-dense tensor ports; not compiler spill, runtime allocation or normal capacity"}


def fp8_linear_wrapper_reference(rows, columns, output_columns):
    """Conditional wrapper storage roles from upstream apply_weights/callee.

    A BF16 empty As is an unused Python argument of the FlashInfer path;
    it is not an initialized tensor or the internal FP32 quantizer scales.
    No runtime class is inferred from this source program enumeration.
    """
    return {
        "source_symbol": "Fp8BlockScaledMMLinearKernel.apply_weights",
        "input_view": {"dtype": "bf16", "shape": [rows, columns], "aliases_input": True, "allocates": False},
        "output": {"dtype": "bf16", "shape": [rows, output_columns],
                   "to_dtype_allocates_if_mm_returns_configured_dtype": False, "final_view_allocates": False, "bias": None},
        "program_alternatives": [
            {"condition": "apply_input_quant=True; dynamic full-group FP32-scale source template",
             "q_input": {"dtype": "fp8", "shape": [rows, columns], "aliases_input": False},
             "As": {"role": "quantizer_activation_scales", "dtype": "fp32",
                    "logical_shape": [rows, columns // 128], "read_by_matmul": True},
             "quantizer": "separate producer; FP8 values and FP32 scales are read by matmul"},
            {"condition": "FlashInferFp8BlockScaledMMKernel.apply_input_quant=False",
             "q_input": {"dtype": "bf16", "shape": [rows, columns], "aliases_input": True},
             "As": {"role": "unused_wrapper_argument", "read_by_matmul": False,
                    "when_input_scale_present": "alias existing input_scale; no new allocation",
                    "when_input_scale_absent": {"operation": "input_2d.new_empty(1)", "dtype": "bf16", "shape": [1],
                        "logical_storage_bytes": 2, "initialized": False, "logical_read_bytes": 0, "logical_write_bytes": 0}},
             "quantizer": "inside SDK; its FP32 activation scales are a different object from As"}],
        "actual_wrapper_selected": None, "actual_provider_bound": False,
        "scope": "conditional no-bias vLLM BF16-output wrapper objects; not caller identity, packed-scale runtime, library scratch, peak capacity or extra payload traffic"}


def prepare_source_bindings(root):
    """Existing V4 prepare source contracts, shared by both graph adapters."""
    anchors = {"indexer-q-quant": _function_anchor(root,
        "vllm/models/deepseek_v4/common/ops/fused_indexer_q.py", "fused_indexer_q_rope_quant")}
    anchors["indexer-q-quant"]["related_definitions"] = [
        _function_anchor(root, relative, symbol) for relative, symbol in (
            ("vllm/models/deepseek_v4/common/ops/fused_indexer_q.py", "FusedIndexerQRopeQuantTritonKernel.kernel"),
            ("vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py", "_load_q_and_rope"),
            ("vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py", "IndexerQFp8Kernel.kernel"),
            ("vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py", "IndexerQFp8Kernel.__call__"))]
    # Fail closed if source dispatch/scale algebra no longer supports the
    # conditional work accounts below. Do not execute either GPU backend.
    index_q_relative = "vllm/models/deepseek_v4/common/ops/fused_indexer_q.py"
    cute_q_relative = "vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py"
    for relative, symbol, fragments in (
        (index_q_relative, "fused_indexer_q_rope_quant", ("ifhas_cutedsl():", "_INDEXER_Q_FP8_KERNEL(")),
        (index_q_relative, "FusedIndexerQRopeQuantTritonKernel.kernel", (
            "tl.div_rn(tl.maximum(amax,0.0001),FP8_MAX)",
            "tl.math.exp2(tl.math.ceil(tl.math.log2(index_q_scale)))", "tl.div_rn(x_nope,index_q_scale)",
            "index_weights*=index_q_scale", "index_weights*=index_weights_softmax_scale",
            "index_weights*=index_weights_head_scale")),
        (cute_q_relative, "IndexerQFp8Kernel.kernel", (
            "_bf16x2_abs(q_bf16x2[i,0])", "_bf16x2_max(amax_bf16x2,_bf16x2_abs(q_bf16x2[i,j]))",
            "cute_utils.fmax(amax,Float32(0.0001))*Float32(1.0/448.0)",
            "bits+Uint32(8388607)", "scale_exp<<Uint32(23)",
            "inv_scale_bits=Uint32(254)-scale_exp<<Uint32(23)", "q0*inv_fp8_scale",
            "weights[token_id,head_id].to(Float32)*scale*fp8_scale")),
        (cute_q_relative, "IndexerQFp8Kernel.__call__", ("float(weights_softmax_scale*weights_head_scale)",)),
    ):
        body = ast.unparse(_definition(ast.parse((root / relative).read_text(encoding="utf-8")), symbol))
        compact = re.sub(r"\s+", "", body)
        missing = [fragment for fragment in fragments if fragment not in compact]
        if missing:
            raise ValueError(f"indexer Q source program changed: {relative}:{symbol}: {missing}")
    anchors["qnorm-rope-insert"] = _cpp_anchor(
        root, "csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu",
        "__device__ __forceinline__ void processDeepseekV4Slot(")
    anchors["qnorm-rope-insert"]["related_definitions"] = [
        _cpp_anchor(root, "csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu", signature)
        for signature in ("__device__ __forceinline__ float warpSum(",
                          "__device__ __forceinline__ float warp4MaxAbs(")]
    insert_relative = "csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu"
    anchors["swa-insert-objects"] = _function_anchor(root, _ATTENTION, "DeepseekV4Attention._fused_qnorm_rope_kv_insert")
    anchors["swa-insert-objects"]["related_definitions"] = [
        _cpp_anchor(root, insert_relative, "torch::stable::Tensor fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert("),
        anchors["qnorm-rope-insert"],
        _function_anchor(root, "vllm/models/deepseek_v4/nvidia/flashmla.py", "DeepseekV4FlashMLAAttention.forward_mqa"),
        _function_anchor(root, "vllm/models/deepseek_v4/nvidia/flashmla.py", "DeepseekV4FlashMLAAttention._forward_decode")]
    insert_body = re.sub(r"\s+", "", anchors["swa-insert-objects"]["related_definitions"][0]["source"])
    if any(fragment not in insert_body for fragment in (
            "autoq_out=torch::stable::new_empty(", "returnq_out;", "k_cache.stride(0)",
            "num_tokens_insert<=num_tokens_full")):
        raise ValueError("V4 padded Q allocation/source cache view contract changed")
    slot_body = re.sub(r"\s+", "", anchors["qnorm-rope-insert"]["source"])
    if any(fragment not in slot_body for fragment in (
            "if(isPadQ)", "uint4constzero4={0u,0u,0u,0u};", "if(slot_id>=0)", "pos_in_block*token_data_bytes",
            "sumOfSquares+=elements[i]*elements[i]", "warpSum<float>(sumOfSquares)",
            "floatconstexponent=ceilf(log2f(absmax/kFp8Max))", "exponent+127.0f",
            "static_cast<int64_t>(cache_block_size)*token_data_bytes+pos_in_block*scale_bytes",
            "token_scale_ptr[kNumQuantBlocks]=0;", "Converter::convert(Converter::convert(elements[i]))")):
        raise ValueError("V4 packed SWA address/padding/rounding contract changed")
    insert_constants = re.sub(r"\s+", "", (root / insert_relative).read_text())
    if any(fragment not in insert_constants for fragment in (
            "constexprintkHeadDim=512;", "constexprintkRopeDim=64;", "constexprintkNopeDim=kHeadDim-kRopeDim;",
            "constexprintkQuantBlock=64;", "constexprintkNumQuantBlocks=kNopeDim/kQuantBlock;",
            "constexprintkScaleBytesPerToken=kNumQuantBlocks+1;", "constexprintkTokenDataBytes=kNopeDim+kRopeDim*2;")):
        raise ValueError("V4 packed SWA byte layout constants changed")
    bindings = re.sub(r"\s+", "", (root / "csrc/libtorch_stable/torch_bindings.cpp").read_text())
    schema_start = bindings.index('"fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert("')
    if '"boolapply_q_norm=True,boolkv_mxfp8=False)->Tensor"' not in bindings[schema_start:schema_start + 500]:
        raise ValueError("V4 packed SWA op schema defaults changed")
    short = _definition(ast.parse((root / _ATTENTION).read_text()), "_fill_short_context_topk_indices")
    compact = [re.sub(r"\s+", "", ast.unparse(node)) for node in short.body]
    if compact != ["row=tl.program_id(0)", "offsets=tl.arange(0,PADDED_TOP_K)",
            "num_compressed=(tl.load(positions+row)+1)//COMPRESS_RATIO",
            "tl.store(output+row*TOP_K+offsets,tl.where(offsets<num_compressed,offsets,-1),mask=offsets<TOP_K)"]:
        raise ValueError("V4 short-context fill source changed")
    anchors["short-topk"] = _function_anchor(root, _ATTENTION, "_fill_short_context_topk_indices")
    embedding = _cpp_anchor(root, "csrc/libtorch_stable/vocab_parallel_embedding_kernels.cu",
        "__global__ void vocab_parallel_embedding_kernel(")
    text = re.sub(r"\s+", "", embedding["source"])
    for expression in ("static_cast<int64_t>(input_ids[token])", "id>=org_vocab_start_index&&id<org_vocab_end_index",
            "id>=added_vocab_start_index&&id<added_vocab_end_index", "out_row[v]=weight_row[v]", "out_row[v]=zero"):
        if expression not in text:
            raise ValueError("V4 embedding copy/source range program changed")
    anchors["embedding-cuda"] = embedding
    return anchors


def fp8_quantizer_source_bindings(root):
    """Source-only conditional quantizer definitions; never imports a provider."""
    linear_root = "vllm/model_executor/kernels/linear/scaled_mm/"
    quant_relative = "vllm/model_executor/layers/quantization/utils/fp8_utils.py"
    cpp_relative = "csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu"
    bindings = [
        _function_anchor(root, relative, symbol) for relative, symbol in (
            (linear_root + "BlockScaledMMLinearKernel.py", "Fp8BlockScaledMMLinearKernel.apply_weights"),
            (linear_root + "flashinfer.py", "FlashInferFp8BlockScaledMMKernel.apply_block_scaled_mm"),
            (linear_root + "flashinfer.py", "_dynamic_flashinfer_deepgemm_blockscale_gemm_impl"),
            (linear_root + "deep_gemm.py", "DeepGemmFp8BlockScaledMMKernel.__init__"),
            ("vllm/utils/deep_gemm.py", "DeepGemmQuantScaleFMT.init_oracle_cache"),
            ("vllm/model_executor/layers/quantization/input_quant_fp8.py", "QuantFP8.forward_cuda"),
            (quant_relative, "per_token_group_quant_fp8"), (quant_relative, "_per_token_group_quant_fp8_colmajor"))]
    for relative, symbol, fragments in (
        (linear_root + "flashinfer.py", "_dynamic_flashinfer_deepgemm_blockscale_gemm_impl", (
            "ifenvs.VLLM_BATCH_INVARIANT:", "condition=input.shape[0]<32", "column_major_scales=True")),
        (quant_relative, "per_token_group_quant_fp8", (
            "x.is_contiguous()", "torch.ops._C.per_token_group_fp8_quant(", "dtype=torch.float32",
            "tma_aligned_m=get_tma_aligned_size(m,4)")),
        (quant_relative, "_per_token_group_quant_fp8", (
            "scale_raw=_absmax*(1.0/fp8_max)", "y_s=tl.math.exp2(tl.ceil(tl.log2(scale_raw)))ifuse_ue8m0elsescale_raw",
            "y_q=tl.clamp(y/y_s,fp8_min,fp8_max).to(y_q_ptr.dtype.element_ty)")),
        (quant_relative, "_per_token_group_quant_fp8_colmajor", (
            "scale_raw=_absmax*(1.0/fp8_max)", "y_s=tl.math.exp2(tl.ceil(tl.log2(scale_raw)))ifuse_ue8m0elsescale_raw",
            "y_q=tl.clamp(y/y_s,fp8_min,fp8_max).to(y_q_ptr.dtype.element_ty)")),
    ):
        compact = re.sub(r"\s+", "", ast.unparse(_definition(ast.parse((root / relative).read_text()), symbol)))
        if any(fragment not in compact for fragment in fragments):
            raise ValueError("FP8 conditional source-stage program changed")
    cpp = [_cpp_anchor(root, cpp_relative, signature) for signature in (
        "__device__ __forceinline__ float ComputeGroupScale(", "__device__ __forceinline__ void QuantizeGroup(",
        "__global__ void per_token_group_quant_8bit_kernel(")]
    scale_source, value_source, kernel_source = (re.sub(r"\s+", "", item["source"]) for item in cpp)
    if ("floaty_s=local_absmax/max_8bit;" not in scale_source
            or "exp2f(ceilf(log2f(fmaxf(fabsf(y_s),1e-10f))))" not in scale_source
            or "fminf(fmaxf(static_cast<float>(src)/y_s,min_8bit),max_8bit)" not in value_source
            or "threads_per_group=16" not in kernel_source):
        raise ValueError("CUDA FP8 group scale/divide/clamp source changed")
    return {"definitions": bindings + cpp,
        "scope": "source-normalized quantizer -> packed matmul graph; call packaging alternatives, not selected runtime class or launch",
        "scale_storage_scope": "Hopper FP32 or FP32-ceil-UE8M0 scale tensors; packed UE8M0 oracle path is for device families100/120",
        "native_dispatch_bound": False, "finite_precision_equivalence_asserted": False}


def external_fp8_quantizer_reference(rows, columns):
    """Independent FP32-scale CUDA/Triton source programs for full groups."""
    count, groups = rows * columns, rows * (columns // 128)
    programs = []
    for backend in ("cuda", "triton"):
        for rounded in (False, True):
            programs.append({"backend": backend, "scale_ue8m0": rounded,
                "scale_expression": (("exp2(ceil(log2(max(abs(raw_scale),1e-10))))" if backend == "cuda"
                    else "exp2(ceil(log2(raw_scale)))") if rounded else "raw_scale"),
                "raw_scale_expression": "max(abs(x),eps)/448" if backend == "cuda" else "max(abs(x),eps)*(1/448)",
                "nonarithmetic_primitives": {"fp32_abs": count + (groups if rounded and backend == "cuda" else 0),
                    "fp32_max_comparisons": (128 * groups) + (groups if rounded and backend == "cuda" else 0),
                    "fp32_clamp_comparisons": 2 * count, "scale_ceil": groups if rounded else 0},
                "demand": {"compute_tensor": 0, "compute_vector": 0, "compute_fma": groups * 129,
                    "compute_reduction": 0, "compute_alu": 0, "compute_sfu": 2 * groups if rounded else 0}})
    return {"operands": [
        {"role": "input", "dtype": "bf16", "elements": count, "bytes": 2 * count, "access": "read"},
        {"role": "quantized_input", "dtype": "fp8", "elements": count, "bytes": count, "access": "write"},
        {"role": "activation_scales", "dtype": "fp32", "elements": groups, "bytes": groups * 4, "access": "write"}],
        "logical_bytes": count * 3 + groups * 4,
        "source_read_write_bytes": {"read": 2 * count, "write": count + groups * 4},
        "source_alternatives": programs, "selected_backend": None, "selected_scale_ue8m0": None, "selected_tma_alignment": None,
        "scale_layouts": [{"tma_aligned": aligned, "logical_shape": [rows, columns // 128],
            "strides_elements": [1, ((rows + 3) // 4) * 4 if aligned else rows],
            "address_span_bytes": 4 * ((columns // 128 - 1) * (((rows + 3) // 4) * 4 if aligned else rows) + rows),
            "padding_written": False} for aligned in (False, True)],
        "value_program": {"input_widening": "bf16 -> fp32", "eps": 1e-10,
            "quantize": "clamp(fp32(input) / fp32(scale), -448, 448) -> e4m3", "fp32_value_divisions": count,
            "fp32_abs": count, "fp32_clamp_comparisons": 2 * count, "fp32_to_fp8_values": count,
            "global_input_role_read_values": count},
        "backend_conditions": [
            {"backend": "cuda", "condition": "input.is_contiguous() on CUDA", "threads_per_group": 16,
                "local_and_shuffle_max_expressions": count + groups * 16 * 4,
                "scale_divide_expressions": groups * 16,
                "rounded_scale_log2_exp2_expressions_each_if_enabled": groups * 16,
                "shared_bf16_store_values": count, "shared_bf16_load_values": count},
            {"backend": "triton", "condition": "otherwise; last-dimension groups remain contiguous",
                "input_values_reused_in_ssa": count, "shared_memory_program_bound": False}],
        "actual_provider_bound": False, "actual_allocator_binding_verified": False, "numerical_equivalence_asserted": False,
        "arithmetic_scope": "one FLOP per FP32 add/subtract/multiply/divide; SFU log2/exp2; comparisons, abs/ceil, source lane replication and typed conversion issue unpriced"}


def attach_fp8_stage_forms(operators, projections):
    """Normalize the two independently source-bound call packagings.

    This proves a quantizer-to-matmul logical graph. It does not choose the
    runtime class, quantizer backend, scale rounding or actual GPU launches.
    """
    by_id = {record["node_id"]: record for record in projections}
    for record in operators:
        if record["kind"] not in ("linear", "router", "lm_head") or record["lowering_account"] is not None:
            continue
        projection = by_id[record["node_id"]]
        if projection["weight_dtype"] != "fp8" or projection["role"] == "wo-a":
            continue
        b, m, k = (projection["shape"][name] for name in ("b", "m", "k"))
        work = external_fp8_quantizer_reference(b * m, k)
        gemm = {"kind": "GEMM", "variant": record["variant"], "dtype": "bf16"}
        record["lowering_forms"] = {
            "normal_form": ["BF16 -> FP8 values and FP32 scales", "FP8 matmul -> BF16"],
            "external": [{"kind": "VECTOR", "variant": "fp8_group_quant", "dtype": "bf16",
                "required_shape": {"element_count": b * m * k, "rows": b * m, "columns": k},
                "external_quant_work": work}, gemm],
            "in_call": [gemm],
            "unbound_native_conditions": ["selected runtime class/provider", "VLLM_BATCH_INVARIANT", "input contiguity", "scale rounding", "TMA alignment", "compiler and launch"]}


def _state_slot_mapping_reference():
    """CP1 algebra of ComputeSlotMappingKernel plus compressor forwarding."""
    return {
        "context_parallel_size": 1, "context_parallel_rank": 0, "mode_condition": "BlockTable uses TOKEN_TO_KV_SLOT",
        "block_table_dtype": "int32", "positions_dtype": "int64", "slot_dtype": "int64",
        "allocation_kernel_relation": "allocation_block_size = kernel_block_size * blocks_per_allocation_block",
        "table_index": "position // kernel_block_size", "slot_offset": "position % kernel_block_size",
        "slot": "int64(block_table[request_index, table_index]) * kernel_block_size + slot_offset",
        "hybrid_block_splitting_equivalence": "CP1; positive sizes; allocation block size divisible by kernel block size",
        "compressor_table_forwarding": "same common block table after in-place clamp_(min=0)",
        "compressor_slot_forwarding": "same common slot_mapping tensor; no copy",
        "allocation_transition": {
            "ordinary_growth": "req_blocks.extend(new_blocks)",
            "partial_hit": "pop pending partial-hit entry; allocate private CoW block; replace source tail before appending",
            "cow_table_write": "req_blocks[block_idx] = cow_block",
            "cow_copy": "queue (source_block, cow_block); source hit-ref retained and destination receives one extra ref",
            "old_physical_prefix_ids_unconditionally_preserved": False,
            "running_request_new_hit_rule": "request in num_cached_block requires len(new_computed_blocks) == 0",
            "pending_copy_drain": "return pending list and replace queue with an empty list",
            "dequeue_implies_GPU_copy_complete": False,
            "worker_copy_program": {
                "condition": "valid nonnegative scheduler block IDs, nonconflicting destinations, positive num_blocks and exact kernel-block divisibility",
                "indices_dtype": "int64", "actual_copy_descriptors": None,
                "view_dedup_key": ["cache.device", "cache.data_ptr()"],
                "kernel_blocks_per_scheduler_block": "cache.shape[0] // num_blocks",
                "scheduler_block_stride_bytes": "cache.stride(0) * cache.element_size() * kernel_blocks_per_scheduler_block",
                "whole_storage_condition": "storage.nbytes() == num_blocks * scheduler_block_stride_bytes",
                "whole_storage_dedup_key": ["cache.device", "storage.data_ptr()"],
                "whole_storage_rows": "uint8 storage view [num_blocks, storage.nbytes() // num_blocks]",
                "whole_storage_row_bytes": "storage.nbytes() // num_blocks",
                "fallback_rows": "cache.unflatten(0, (num_blocks, kernel_blocks_per_scheduler_block))",
                "fallback_row_bytes": "kernel_blocks_per_scheduler_block * product(cache.shape[1:]) * cache.element_size()",
                "fallback_deduplicates_by_storage": False,
                "source_value_program": "temporary = blocks[src]; blocks[dst] = temporary",
                "rhs_advanced_indexing_materializes": True, "rhs_aliases_cache_storage": False,
                "source_read_bytes_per_copied_unit": "copy_count * row_bytes",
                "temporary_write_bytes_per_copied_unit": "copy_count * row_bytes",
                "temporary_read_bytes_per_copied_unit": "copy_count * row_bytes",
                "destination_write_bytes_per_copied_unit": "copy_count * row_bytes",
                "H2D_index_transport_and_actual_peak_capacity_bound": False,
                "worker_call_order": ["zero_new_blocks_if_requested", "copy_kv_cache_blocks_inplace", "_prepare_inputs", "_model_forward"],
                "python_submission_order_implies_GPU_completion": False,
                "retained_endpoints": "both source and destination",
                "retention_release": "free if not defer_block_free or fence_seq <= processed_step_seq; otherwise enqueue fence with reversed endpoints",
                "actual_tables_copy_events_or_completion_bound": False},
            "actual_partial_hits_copies_or_completion": None},
        "table_normalization_preserves_mapping_if": "every table entry addressed by a valid state read/write is nonnegative before clamp",
        "table_normalization_preserves_mapping_unconditionally": False,
        "cuda_graph_tail_slot": -1, "cuda_graph_tail_range": "[num_tokens, max_num_tokens)",
        "metadata_preparation_inside_model_forward": False,
        "cache_views": {"single_zeroed_int8_backing": True, "groups_may_overlay_bytes": True,
            "layer_block_byte_origin": "tensor.offset + layer_index * layer_stride + physical_block * block_stride",
            "dtype_reinterpretation_allocates": False, "compressor_bind_squeeze_allocates": False,
            "group_overlay_program": {
                "condition": "generic device pool, no HiSparse/GLM-special packing; V4 indexer layouts BLHNC or BLNHC; one head per V4 cache view",
                "block_outer_layouts": ["BLHNC", "BLNHC"],
                "pool_block_bytes": "P = max_group(sum_layer(page_size_bytes))",
                "group_origin_bytes": 0,
                "spec_region_offset_bytes": "sum_previous_specs(number_of_layers * page_size_bytes)",
                "layer_stride_bytes": "page_size_bytes",
                "manager_block_stride_bytes": "P",
                "layer_page_interval_bytes": "[b*P + spec_region_offset + layer_index*page_size, b*P + spec_region_offset + (layer_index+1)*page_size)",
                "all_device_managers_share_one_BlockPool": True,
                "distinct_live_allocations_have_distinct_nonnull_manager_ids": True,
                "same_group_distinct_layers_have_disjoint_page_intervals": True,
                "distinct_groups_same_manager_id_may_overlap": True,
                "nonnull_positive_ref_count_alone_proves_exclusive_write": False,
                "kernel_split_ratio": "R = manager_block_size // kernel_block_size",
                "kernel_block_id": "b * R + subblock; 0 <= subblock < R",
                "split_requires_dense_manager_page": True,
                "split_dense_guard": "manager_block_stride == dense_page_size; manager_block_stride % R == 0",
                "kernel_block_stride_bytes": "manager_block_stride // R",
                "ownership_domain": "balanced pool references; retain each read page through its read, mutate only writable pages; no unpinned borrowed reads",
                "selected_layout_groups_pages_or_execution_order_bound": False,
                "scope": "conditional source packing and noninterference rules; not actual allocation, unpin safety, GPU fences or capacity"},
            "selected_layout_or_live_page_ownership": None},
        "lifetime_rules": {
            "null_block_id": 0, "null_block_allocatable": False,
            "new_block_precondition": "free queue; ref_cnt == 0", "new_block_ref_count": 1,
            "writable_predicate": "not is_null and ref_cnt == 1 and block_hash is None",
            "positive_ref_count_implies_writable": False, "nonnegative_slot_implies_writable": False,
            "touch": "remove zero-ref non-null cached block from free queue, then increment reference",
            "free": "decrement one reference; queue non-null block only at ref_cnt == 0",
            "cache_identity": "block_hash bytes + unsigned 4-byte big-endian cache group ID",
            "unpin": "zero-ref content may remain readable until reuse; callback precedes new handout",
            "reference_domain": "balanced references and valid queue/hash state; callbacks do not establish GPU reader fences",
            "actual_pool_state_or_reuse_order_bound": False,
            "scope": "source metadata lifetime rules; not live allocation, GPU reader fence, capacity or runtime acceptance"},
        "table_extent_and_preparation_traffic": None, "actual_table_or_slots_bound": False,
        "scope": "source CP1 slot/page consistency on declared valid table domain; metadata preparation, actual pages and runtime remain separate"}


def _partial_state_view_reference(tokens, layer, component, dimension, ratio):
    """torch.mm FP32 result split into two non-copying row-strided views."""
    width = dimension * (1 + (ratio == 4))
    return {
        "state_owner": {"layer": layer, "component": component, "scope": "local TP rank module instance"},
        "projection_dtype": "fp32", "projection_shape": [tokens, width * 2],
        "split_dimension": -1, "split_allocates": False,
        "projection_views": [
            {"role": "kv", "shape": [tokens, width], "strides_elements": [width * 2, 1], "storage_offset_elements": 0},
            {"role": "score", "shape": [tokens, width], "strides_elements": [width * 2, 1], "storage_offset_elements": width}],
        "views_share_projection_storage": True, "state_dtype": "fp32", "ape_dtype": "fp32",
        "state_logical_record_elements": width * 2,
        "score_state_offset": "state_cache.shape[-1] // 2",
        "state_row_address": "state_cache_ptr + (slot_id // block_size) * state_cache_stride0 + (slot_id % block_size) * state_cache_stride1",
        "score_update": "score + ape[position % compress_ratio]", "compress_ratio": ratio,
        "position_dtype": "int64", "slot_dtype": "int64", "invalid_slot_predicate": "slot_id < 0",
        "invalid_slot_action": "return before projection, position, APE or state accesses",
        "write_before_finish": True, "launch_pdl": False,
        "slot_mapping_source": _state_slot_mapping_reference(),
        "runtime_slot_mapping": None, "physical_state_strides_and_page_alias": None,
        "scope": "source split views and persistent state producer; not physical pages, allocator or cross-step slot binding"}


def _partial_state_window_reference(requests, layer, component, dimension, ratio):
    window = (2 if ratio == 4 else 1) * ratio
    records = []
    for request in requests:
        old, new = request["computed"], request["computed"] + request["query"]
        completed = []
        for position in range(old, new):
            if (position + 1) % ratio:
                continue
            end = position + 1
            begin = max(0, end - window)
            cut = min(end, max(begin, old))
            completed.append({"boundary_end_exclusive": end, "positions": [begin, end],
                "window_rows": [window - (end - begin), window], "prior_step_positions": [begin, cut],
                "current_step_positions": [max(old, begin), end]})
        block = 4 if ratio == 4 else 8
        skipped_without_extra = max(0, old - window + 1)
        free_end_bound = skipped_without_extra - skipped_without_extra % block
        records.append({"write_positions": [old, new], "completed_windows": completed,
            "retention_guarantee": {"committed_prefix": old,
                "free_prefix_end_upper_bound": free_end_bound,
                "metadata_page_floor_rule": "floor(max(0, committed_prefix - window_rows + 1) / B) * B; B = state_metadata.block_size",
                "bound_domain": "C4 B is a positive multiple of 4; supported C128 B = 8",
                "upper_bound_selects_actual_metadata_B": False,
                "guaranteed_prior_retained_positions": [free_end_bound, old],
                "extra_retained_tokens": None, "extra_retained_tokens_domain": "nonnegative",
                "bounds_are_actual_eviction": False,
                "condition": "SlidingWindowManager frees only complete skipped blocks from the old committed prefix; allocated-range capping can only retain more"}})
    return {
        "state_owner": {"layer": layer, "component": component, "scope": "local TP rank module instance"},
        "state_dtype": "fp32", "head_dim": dimension, "compress_ratio": ratio,
        "state_page_rows": None if ratio == 4 else 8,
        "state_page_rows_source": {
            "constructor_default": 4 if ratio == 4 else 8,
            "address_parameter": "state_metadata.block_size",
            "resolved_C4_domain": "positive multiple of 4 from constructor and source integer block-size scaling",
            "C4_selected_value_bound": False,
            "C128_supported_program_requires_8": True},
        "window_rows": window,
        "request_windows": records, "state_value_half": "window_row // 4" if ratio == 4 else "0",
        "value_column_offset": "state_value_half * head_dim",
        "score_column_offset": "state_width + state_value_half * head_dim",
        "page_lookup": "block_table[request_index, absolute_position // state_metadata.block_size]",
        "page_offset": "absolute_position % state_metadata.block_size", "position_mask": "absolute_position >= 0",
        "boundary_guard": "(position + 1) % compress_ratio == 0",
        "input_domain": "nonnegative positions; valid state/destination slots; B = state_metadata.block_size; C4 B is a positive multiple of 4, supported C128 B = 8; state slots equal block_table page * B + position % B; valid nonoverlapping state records",
        "physical_pages_or_strides_selected": False, "actual_runtime_mapping_bound": False,
        "slot_mapping_source": _state_slot_mapping_reference(),
        "scope": "source valid-domain FP32 state windows and logical prior/current-step partition; not actual page alias, allocator or provider acceptance"}


def _attach_shared_add_dataflow(raw, cases, operators):
    """Logical tensor-object contract of the modular runner's out-of-place add.

    This operation borrows both complete producer tensors; it creates no
    input views or temporary tensor. Producer internals and actual allocator
    addresses remain their own obligations, not prerequisites to this add.
    """
    by_id = {record["node_id"]: record for record in operators}
    tokens_by_phase = {case["phase"]: sum(request["query"] for request in case["requests"])
                       for case in cases}
    for record in operators:
        if record["kind"] != "elementwise" or record["variant"] != "add":
            continue
        node_id = record["node_id"]
        prefix, rank_suffix = node_id.rsplit(":shared-add:", 1)
        shared = prefix + ":shared-experts:down-proj:" + rank_suffix
        routed = prefix + ":routed-experts:" + rank_suffix
        consumer = prefix + ":moe-output:tp-all-reduce"
        flow = {
            "program": "vllm_0_30_modular_shared_add",
            "expression": "shared_output + fused_output",
            "inputs": [{"role": "shared_output", "producer": shared},
                       {"role": "fused_output", "producer": routed}],
            "dtype": "bf16", "shape": [tokens_by_phase[node_id.split(":", 1)[0]], raw["hidden_size"]],
            "input_views": "borrow producer tensors without creating a view",
            "mutates_inputs": False, "output_storage": "new tensor distinct from both inputs",
            "temporary_tensors": [], "input_lifetime": "through add completion",
            "output_lifetime": "through combined-output TP all-reduce", "consumer": consumer,
            "scope": "logical source tensor objects; allocator addresses and library scratch are separate"}
        record["source"] = "modular-shared-add"
        record["attributes"].update(source_symbol="vllm.MoERunner.forward", source_dataflow=flow)
        elements = record["shape"]["elements"]
        output_calls = [{"source_node_ids": [producer], "kind": kind, "variant": variant,
            "rank": record["rank"], "dtype": "bf16",
            "shape": {**by_id[producer]["shape"], "dtype_bytes": 2}}
            for producer, kind, variant in ((shared, "GEMM", "gemm"), (routed, "FUSED_MOE", "routed_experts"))]
        record["source_alias_account"] = {
            "declaration": flow, "direct_input_producers": sorted([shared, routed]),
            "producer_outputs": [{"node_id": key, **{field: by_id[key][field]
                for field in ("rank", "kind", "shape", "dtype")}} for key in (shared, routed)],
            "direct_consumers": [consumer],
            "lowered_calls": [{"kind": "VECTOR", "variant": "elementwise", "rank": record["rank"],
                "dtype": "bf16", "shape": {"element_count": elements, "elements": elements, "dtype_bytes": 2},
                "attributes": {"operation": "add", "reads": 2, "writes": 1},
                "dependencies": sorted(output_calls, key=lambda entry: json.dumps(entry, sort_keys=True))}]}


def _attach_greedy_tensor_dataflow(raw, cases, operators):
    """Independent source tensor copies/views for the complete greedy tail."""
    by_id = {record["node_id"]: record for record in operators}
    case_by_phase = {case["phase"]: case for case in cases}
    roles = {"logits-boundary-select": "row_select", "lm-head:tp-gather-reorder": "gather_reorder", "target-sampler-logits-fp32": "cast",
             "target-sampler-argmax": "argmax", "target-sampler-output-int32": "output_int32"}
    for record in operators:
        base = record["node_id"].split(":tp-rank-", 1)[0]
        phase, role = base.split(":", 1)
        if role not in roles:
            continue
        stage = roles[role]
        suffix = "" if record["rank"] == 0 else f":tp-rank-{record['rank']}"
        requests = case_by_phase[phase]["requests"]
        tokens, rows = sum(request["query"] for request in requests), len(requests)
        hidden, vocab = raw["hidden_size"], raw["vocab_size"]
        ends, cursor = [], 0
        for request in requests:
            cursor += request["query"]
            ends.append(cursor - 1)
        producer_role, consumer_role = {
            "row_select": ("final-norm", "lm-head"),
            "gather_reorder": ("lm-head:tp-all-gather", "target-sampler-logits-fp32"),
            "cast": ("lm-head:tp-gather-reorder" if rows > 1 and vocab // 4 > 1 else "lm-head:tp-all-gather", "target-sampler-argmax"),
            "argmax": ("target-sampler-logits-fp32", "target-sampler-output-int32"),
            "output_int32": ("target-sampler-argmax", None)}[stage]
        producer = phase + ":" + producer_role + ("" if producer_role == "lm-head:tp-all-gather" else suffix)
        consumer = None if consumer_role is None else phase + ":" + consumer_role + suffix
        if stage == "row_select":
            in_dtype, out_dtype = "bf16", "bf16"
            in_shape, out_shape = [tokens, hidden], [rows, hidden]
            views = []
            selection = {"indices_dtype": "int32", "indices": ends,
                         "advanced_indexing_allocates_even_for_identity_indices": True}
            attributes = {"model_execution_version": "vllm_0_30_0", "index_source": "query_start_loc_int32",
                          "selected_indices": ends}
            dependency_kind, dependency_variant = "VECTOR", "rmsnorm"
            dependency_shape = {"rows": tokens, "hidden": hidden, "dtype_bytes": 2, "tok": tokens}
        elif stage == "gather_reorder":
            shard = vocab // 4
            in_dtype = out_dtype = "bf16"
            in_shape, out_shape = [4, rows, shard], [rows, vocab]
            views = [{"operation": "movedim(0,1)", "aliases": "input", "allocates": False,
                      "shape": [rows, 4, shard], "strides_elements": [shard, rows * shard, 1], "offset_elements": 0}]
            selection = None
            attributes = {"model_execution_version": "vllm_0_30_0"}
            dependency_kind, dependency_variant, dependency_shape = "COMMUNICATION", None, {}
        elif stage == "cast":
            in_dtype, out_dtype = "bf16", "fp32"
            in_shape = out_shape = [rows, vocab]
            views = [{"operation": "vocabulary slice", "aliases": "input", "allocates": False,
                      "shape": in_shape, "strides_elements": [vocab, 1], "offset_elements": 0}]
            selection = None
            attributes = {"sampling_source": "vllm_0_30_plain_sampler_bf16"}
            if rows > 1 and vocab // 4 > 1:
                dependency_kind, dependency_variant = "VECTOR", "vllm_logits_gather_reorder"
                dependency_shape = {"element_count": rows * vocab, "elements": rows * vocab,
                                    "rows": rows, "vocab": vocab, "tensor_parallel_size": 4, "dtype_bytes": 2}
            else:
                dependency_kind, dependency_variant, dependency_shape = "COMMUNICATION", None, {}
        elif stage == "argmax":
            in_dtype, out_dtype = "fp32", "int64"
            in_shape, out_shape = [rows, vocab], [rows]
            views = [{"operation": "view(-1)", "aliases": "argmax_result", "allocates": False,
                      "shape": out_shape, "strides_elements": [1], "offset_elements": 0}]
            selection = None
            attributes = {"sampling_source": "vllm_0_30_plain_greedy_bf16", "stage": "argmax"}
            dependency_kind, dependency_variant = "VECTOR", "vllm_logits_fp32_cast"
            dependency_shape = {"rows": rows, "vocab": vocab, "element_count": rows * vocab,
                                "dtype_bytes": 2, "elements": rows * vocab}
        elif stage == "output_int32":
            in_dtype, out_dtype = "int64", "int32"
            in_shape, out_shape = [rows], [rows, 1]
            views = [
                {"operation": "long()", "aliases": "input", "allocates": False,
                 "shape": in_shape, "strides_elements": [1], "offset_elements": 0},
                {"operation": "unsqueeze(-1)", "aliases": "int32_cast_result", "allocates": False,
                 "shape": out_shape, "strides_elements": [1, 1], "offset_elements": 0}]
            selection = None
            attributes = {"sampling_source": "vllm_0_30_plain_greedy_bf16", "stage": "output_int32"}
            dependency_kind, dependency_variant = "VECTOR", "vllm_plain_greedy_sample"
            dependency_shape = {"rows": rows, "vocab": vocab, "element_count": rows * vocab,
                                "dtype_bytes": 4, "elements": rows * vocab}
        else:
            raise ValueError("unknown independent greedy tensor flow stage")
        flow = {"program": "vllm_0_30_plain_greedy_tensor_flow", "stage": stage,
            "input": {"producer": producer, "dtype": in_dtype, "shape": in_shape},
            "output": {"dtype": out_dtype, "shape": out_shape, "distinct_from_input": True},
            "views": views, "selection": selection, "temporary_tensors": [], "mutates_input": False,
            "input_lifetime": "through operation completion", "consumer": consumer,
            "output_lifetime": "exported sampler output" if consumer is None else "through consumer read",
            "scope": "logical source tensor objects; library scratch, allocator addresses and runtime binding are separate"}
        record["attributes"]["source_dataflow"] = flow
        width = {"bf16": 2, "fp32": 4, "int64": 8}[in_dtype]
        shape = {key: value for key, value in record["shape"].items() if key != "elements"}
        shape.update(element_count=record["shape"]["elements"], dtype_bytes=width, elements=record["shape"]["elements"])
        source = by_id[producer]
        record["source_alias_account"] = {
            "declaration": flow, "direct_input_producers": [producer],
            "producer_outputs": [{"node_id": producer, **{field: source[field]
                for field in ("rank", "kind", "shape", "dtype")}}],
            "direct_consumers": [] if consumer is None else [consumer],
            "lowered_calls": [{"kind": "VECTOR", "variant": record["variant"], "rank": record["rank"],
                "dtype": in_dtype, "shape": shape, "attributes": attributes,
                "dependencies": [{"source_node_ids": [producer], "kind": dependency_kind,
                    "variant": dependency_variant, "rank": source["rank"], "dtype": source["dtype"],
                    "shape": dependency_shape}]}]}


def _attach_qkv_norm_dataflow(raw, operators):
    """Separate split views, fresh normalized tensors and their typed readers."""
    by_id = {record["node_id"]: record for record in operators}
    q, d, heads = raw["q_lora_rank"], raw["head_dim"], raw["num_attention_heads"] // 4
    for record in operators:
        if record["variant"] != "dsv4_q_kv_rmsnorm":
            continue
        base = record["node_id"].split(":tp-rank-", 1)[0]
        prefix = base.rsplit(":", 1)[0]
        suffix = "" if record["rank"] == 0 else f":tp-rank-{record['rank']}"
        fused, wq, swa, index = [prefix + ":" + role + suffix for role in
                                ("fused-wqa-wkv", "wq-b", "swa-kv-update", "indexer-wq-b")]
        tokens = record["shape"]["rows"]
        qr_consumers = [wq] + ([index] if index in by_id else [])
        ready = [prefix + ":" + role + suffix for role in
                 ("compressor-kv-score", "indexer-weights-proj", "indexer-compressor-kv-score")
                 if prefix + ":" + role + suffix in by_id]
        record["required_ancestors"] = sorted([fused, *ready])
        for role in ("compressor-state-update", "indexer-compressor-state-update"):
            state = prefix + ":" + role + suffix
            if state in by_id:
                by_id[state]["required_ancestors"].append(record["node_id"])
        flow = {
            "program": "vllm_0_30_q_kv_rmsnorm",
            "projection": {"producer": fused, "dtype": "bf16", "shape": [tokens, q + d], "strides_elements": [q + d, 1]},
            "split_allocates": False,
            "split_views": [
                {"role": "qr", "shape": [tokens, q], "strides_elements": [q + d, 1], "offset_elements": 0},
                {"role": "kv", "shape": [tokens, d], "strides_elements": [q + d, 1], "offset_elements": q}],
            "gamma": [{"role": "q_norm.weight", "dtype": "bf16", "shape": [q], "contiguous": True},
                      {"role": "kv_norm.weight", "dtype": "bf16", "shape": [d], "contiguous": True}],
            "outputs": [
                {"role": "qr", "dtype": "bf16", "shape": [tokens, q], "strides_elements": [q, 1],
                 "offset_elements": 0, "consumers": qr_consumers},
                {"role": "kv", "dtype": "bf16", "shape": [tokens, d], "strides_elements": [d, 1],
                 "offset_elements": 0, "consumers": [swa]}],
            "outputs_allocate_separately": True, "outputs_alias_input": False, "mutates_input": False,
            "qr_scale": None, "epsilon": raw["rms_norm_eps"], "normalization_dtype": "fp32",
            "rounding": "one BF16 store after gamma multiplication", "global_temporaries": [],
            "scope": "source split and normalization tensor objects; physical allocator, source gamma values and runtime are separate"}
        record["attributes"]["source_dataflow"] = flow
        record["source"] = "qkv-norm-dataflow"
        qr_input = [{"producer": record["node_id"], "output_role": "qr", "dtype": "bf16",
                     "shape": [tokens, q], "strides_elements": [q, 1], "offset_elements": 0}]
        consumer_inputs = {consumer: qr_input for consumer in qr_consumers}
        consumer_inputs[swa] = [
            {"producer": wq, "output_role": "output", "dtype": "bf16", "source_shape": [tokens, heads * d],
             "shape": [tokens, heads, d], "strides_elements": [heads * d, d, 1], "offset_elements": 0, "view_allocates": False},
            {"producer": record["node_id"], "output_role": "kv", "dtype": "bf16", "shape": [tokens, d],
             "strides_elements": [d, 1], "offset_elements": 0}]
        for consumer, inputs in consumer_inputs.items():
            by_id[consumer]["attributes"]["source_tensor_inputs"] = inputs
        successors = sorted(item["node_id"] for item in operators if record["node_id"] in item["required_ancestors"])
        producers = [fused, *ready]
        record["source_alias_account"] = {
            "declaration": flow, "direct_input_producers": sorted(producers),
            "producer_outputs": [{"node_id": producer, **{field: by_id[producer][field]
                for field in ("rank", "kind", "shape", "dtype")}} for producer in producers],
            "direct_consumers": successors, "consumer_tensor_inputs": consumer_inputs,
            "lowered_calls": [{"kind": "VECTOR", "variant": record["variant"], "rank": record["rank"],
                "dtype": "bf16", "shape": {**record["shape"], "dtype_bytes": 2}, "attributes": {},
                "dependencies": sorted([{"source_node_ids": [producer], "kind": "GEMM", "variant": "gemm",
                    "rank": record["rank"], "dtype": "bf16", "shape": {**by_id[producer]["shape"], "dtype_bytes": 2}}
                    for producer in producers], key=lambda entry: json.dumps(entry, sort_keys=True))}]}


def _attach_attention_input_projection_dataflow(raw, operators):
    """Bind the shared mHC layer-input and the unquantized FP32 compressor C."""
    by_id = {record["node_id"]: record for record in operators}
    roles = {"fused-wqa-wkv", "compressor-kv-score", "indexer-weights-proj", "indexer-compressor-kv-score"}
    hidden = raw["hidden_size"]
    for record in operators:
        if record["kind"] != "linear":
            continue
        base = record["node_id"].split(":tp-rank-", 1)[0]
        parts = base.split(":")
        if len(parts) != 3 or parts[2] not in roles:
            continue
        phase, layer, role = parts
        suffix = "" if record["rank"] == 0 else f":tp-rank-{record['rank']}"
        prefix = phase + ":" + layer + ":"
        parent = prefix + ("mhc-pre" if layer == "layer-0" else "mhc-post-pre-attn") + suffix
        rows, columns = record["shape"]["m"], record["shape"]["n"]
        input_port = {"producer": parent, "output_role": "layer_input", "dtype": "bf16",
                      "shape": [rows, hidden], "strides_elements": [hidden, 1], "offset_elements": 0}
        record["attributes"]["source_tensor_inputs"] = [input_port]
        if role not in ("compressor-kv-score", "indexer-compressor-kv-score"):
            continue
        component = "compressor" if role == "compressor-kv-score" else "indexer.compressor"
        consumer = prefix + ("compressor-state-update" if role == "compressor-kv-score"
                             else "indexer-compressor-state-update") + suffix
        flow = {"program": "vllm_0_30_bf16_compressor_projection", "input": input_port,
            "weight": {"parameter": component + ".fused_wkv_wgate.weight", "dtype": "bf16",
                "base_shape": [columns, hidden], "base_strides_elements": [hidden, 1],
                "transpose_shape": [hidden, columns], "transpose_strides_elements": [1, hidden], "transpose_allocates": False},
            "output": {"dtype": "fp32", "shape": [rows, columns], "strides_elements": [columns, 1],
                "offset_elements": 0, "aliases_input_or_weight": False},
            "mm_out_dtype": "fp32", "input_cast_to_fp32": False, "weight_cast_to_fp32": False,
            "activation_quantizer": None, "mutates_inputs": False, "caller_temporaries": [], "consumer": consumer,
            "scope": "source BF16 A/B, weight transpose view and fresh FP32 C; not library scratch, physical allocator or runtime"}
        record["source"] = "compressor-projection-dataflow"
        record["attributes"]["source_dataflow"] = flow
        consumer_inputs = [{"producer": record["node_id"], "output_role": "output", "dtype": "fp32",
                            "shape": [rows, columns], "strides_elements": [columns, 1], "offset_elements": 0}]
        by_id[consumer]["attributes"]["source_tensor_inputs"] = consumer_inputs
        record["source_alias_dependency_mode"] = "mhc_layer_input_completion"
        record["source_alias_account"] = {
            "declaration": flow, "direct_input_producers": [parent],
            "producer_outputs": [{"node_id": parent, **{field: by_id[parent][field] for field in ("rank", "kind", "shape", "dtype")}}],
            "direct_consumers": sorted(item["node_id"] for item in operators if record["node_id"] in item["required_ancestors"]),
            "consumer_tensor_inputs": {consumer: consumer_inputs},
            "lowered_calls": [{"kind": "GEMM", "variant": "gemm", "rank": record["rank"], "dtype": "bf16",
                "shape": {**record["shape"], "dtype_bytes": 2},
                "attributes": {"output_dtype": "fp32", "weight_dtype": "bf16", "weight_group_size": 0},
                "dependencies": [{"source_node_id": parent, "rank": record["rank"], "terminal_source_call": True,
                    "layer_input_writes": [{"role": "layer_input", "dtype": "bf16", "shape": [rows, hidden],
                                            "bytes": rows * hidden * 2, "access": "write"}]}]}]}


def _attach_shared_activation_dataflow(raw, operators):
    """Source CUDA wrapper allocates only output; gate/up are kernel pointers."""
    by_id = {record["node_id"]: record for record in operators}
    for record in operators:
        if record["kind"] != "activation" or ":shared-experts:silu-mul:" not in record["node_id"]:
            continue
        prefix, suffix = record["node_id"].rsplit(":shared-experts:silu-mul:", 1)
        producer, consumer = [prefix + ":shared-experts:" + role + ":" + suffix for role in ("gate-up", "down-proj")]
        rows, width = record["shape"]["tok"], record["shape"]["intermediate"]
        flow = {"program": "vllm_0_30_cuda_clamped_shared_activation",
            "input": {"producer": producer, "output_role": "output", "dtype": "bf16", "shape": [rows, 2 * width],
                      "strides_elements": [2 * width, 1], "offset_elements": 0},
            "kernel_input_slices": [
                {"role": "gate", "shape": [rows, width], "strides_elements": [2 * width, 1], "offset_elements": 0},
                {"role": "up", "shape": [rows, width], "strides_elements": [2 * width, 1], "offset_elements": width}],
            "slices_are_kernel_pointer_offsets": True, "slice_tensor_allocations": [],
            "output": {"dtype": "bf16", "shape": [rows, width], "strides_elements": [width, 1], "offset_elements": 0,
                       "aliases_input": False},
            "output_allocation": "torch.empty", "mutates_input": False, "temporary_tensors": [],
            "gate_clamp": {"min": None, "max": raw["swiglu_limit"]},
            "up_clamp": {"min": -raw["swiglu_limit"], "max": raw["swiglu_limit"]}, "alpha": 1.0, "beta": 0.0,
            "consumer": consumer,
            "scope": "CUDA custom-op source tensor contract; not forward_native tensor temporaries, selected runtime provider or physical allocator"}
        record["source"] = "shared-activation-cuda-dataflow"
        record["attributes"]["source_dataflow"] = flow
        consumer_inputs = [{"producer": record["node_id"], "output_role": "output", "dtype": "bf16",
                            "shape": [rows, width], "strides_elements": [width, 1], "offset_elements": 0}]
        by_id[consumer]["attributes"]["source_tensor_inputs"] = consumer_inputs
        record["source_alias_account"] = {"declaration": flow, "direct_input_producers": [producer],
            "producer_outputs": [{"node_id": producer, **{field: by_id[producer][field] for field in ("rank", "kind", "shape", "dtype")}}],
            "direct_consumers": [consumer], "consumer_tensor_inputs": {consumer: consumer_inputs},
            "lowered_calls": [{"kind": "VECTOR", "variant": "silu", "rank": record["rank"], "dtype": "bf16",
                "shape": {**record["shape"], "element_count": rows * width, "dtype_bytes": 2},
                "attributes": {"swiglu_limit": raw["swiglu_limit"]},
                "dependencies": [{"source_node_ids": [producer], "kind": "GEMM", "variant": "gemm", "rank": record["rank"],
                                  "dtype": "bf16", "shape": {**by_id[producer]["shape"], "dtype_bytes": 2}}]}]}


def _attach_gate_projection_dataflow(raw, operators):
    """Bind router logits without assuming which eligible CUDA gate is loaded."""
    by_id = {record["node_id"]: record for record in operators}
    for record in operators:
        if record["kind"] != "router":
            continue
        prefix, suffix = record["node_id"].rsplit(":router:", 1)
        rank = record["rank"]
        parent = prefix + ":mhc-post-pre-ffn" + (f":tp-rank-{rank}" if rank else "")
        consumer = prefix + ":route-selection-weights:" + suffix
        rows, hidden, experts = record["shape"]["m"], raw["hidden_size"], raw["n_routed_experts"]
        flow = {"program": "vllm_0_30_gate_linear_fp32_output",
            "input": {"producer": parent, "output_role": "layer_input", "dtype": "bf16", "shape": [rows, hidden],
                      "strides_elements": [hidden, 1], "offset_elements": 0},
            "weight": {"parameter": "gate.weight", "dtype": "bf16", "base_shape": [experts, hidden], "base_strides_elements": [hidden, 1]},
            "eligible_source_programs": ["cute_ll_bf16", "torch_mm_out_dtype"] if rows <= 16 else ["torch_mm_out_dtype"],
            "selected_source_program": None,
            "cute_ll_condition": "allow_ll_bf16_gemm and rows <= 16 and input.dtype == bf16",
            "program_weight_inputs": {
                "cute_ll_bf16": {"shape": [experts, hidden], "strides_elements": [hidden, 1], "allocates": False},
                "torch_mm_out_dtype": {"shape": [hidden, experts], "strides_elements": [1, hidden], "allocates": False}},
            "output": {"dtype": "fp32", "shape": [rows, experts], "strides_elements": [experts, 1], "offset_elements": 0,
                       "aliases_input_or_weight": False},
            "input_cast_to_fp32": False, "weight_cast_to_fp32": False, "activation_quantizer": None,
            "bias": None, "caller_temporaries": [], "consumer": consumer,
            "scope": "common BF16 gate inputs and fresh FP32 logits for eligible CUDA source programs; selected runtime provider, compilation cache and library scratch separate"}
        record["source"] = "gate-projection-dataflow"
        record["attributes"]["source_dataflow"] = flow
        consumer_inputs = [{"producer": record["node_id"], "output_role": "logits", "dtype": "fp32",
                            "shape": [rows, experts], "strides_elements": [experts, 1], "offset_elements": 0}]
        by_id[consumer]["attributes"]["source_tensor_inputs"] = consumer_inputs
        record["source_alias_dependency_mode"] = "mhc_layer_input_completion"
        record["source_alias_account"] = {"declaration": flow, "direct_input_producers": [parent],
            "producer_outputs": [{"node_id": parent, **{field: by_id[parent][field] for field in ("rank", "kind", "shape", "dtype")}}],
            "direct_consumers": [consumer], "consumer_tensor_inputs": {consumer: consumer_inputs},
            "lowered_calls": [{"kind": "GEMM", "variant": "gemm", "rank": rank, "dtype": "bf16",
                "shape": {**record["shape"], "dtype_bytes": 2},
                "attributes": {"output_dtype": "fp32", "weight_dtype": "bf16", "weight_group_size": 0},
                "dependencies": [{"source_node_id": parent, "rank": rank, "terminal_source_call": True,
                    "layer_input_writes": [{"role": "layer_input", "dtype": "bf16", "shape": [rows, hidden],
                                            "bytes": rows * hidden * 2, "access": "write"}]}]}]}


def _attach_indexer_weights_dataflow(raw, cases, operators):
    """Separate a precomputed weight tensor's numerical consumer from ready joins."""
    by_id = {record["node_id"]: record for record in operators}
    case_by_phase = {case["phase"]: case for case in cases}
    hidden, heads, dimension = raw["hidden_size"], raw["index_n_heads"], raw["index_head_dim"]
    for record in operators:
        if record["kind"] != "linear" or ":indexer-weights-proj" not in record["node_id"]:
            continue
        base = record["node_id"].split(":tp-rank-", 1)[0]
        phase, layer, role = base.split(":")
        suffix = f":tp-rank-{record['rank']}" if record["rank"] else ""
        prefix = phase + ":" + layer + ":"
        requests = case_by_phase[phase]["requests"]
        short = max(request["computed"] + request["query"] for request in requests) // 4 <= raw["index_topk"]
        parent, ready = prefix + "mhc-post-pre-attn" + suffix, prefix + "q-kv-rmsnorm" + suffix
        consumer = None if short else prefix + "indexer-q-rope-quant" + suffix
        rows = record["shape"]["m"]
        flow = {"program": "vllm_0_30_indexer_weights_projection",
            "input": {"producer": parent, "output_role": "layer_input", "dtype": "bf16", "shape": [rows, hidden],
                      "strides_elements": [hidden, 1], "offset_elements": 0},
            "weight": {"parameter": "indexer.weights_proj.weight", "dtype": "bf16", "shape": [heads, hidden], "strides_elements": [hidden, 1]},
            "eligible_source_programs": ["torch_F_linear", "batch_invariant_matmul"],
            "selected_source_program": None, "batch_invariant_flag": None,
            "source_hardware_domain": "CUDA SM90; SM100-only FlashInfer BF16 path ineligible",
            "weight_inputs": {"torch_F_linear": "base_NK", "batch_invariant_matmul": "weight.t() view_KN"},
            "weight_transpose_allocates": False,
            "output": {"dtype": "bf16", "shape": [rows, heads], "strides_elements": [heads, 1], "offset_elements": 0,
                       "aliases_input_or_weight": False},
            "bias": None, "activation_quantizer": None, "caller_temporaries": [], "short_context_eager": short,
            "consumer": consumer, "source_output_use": "unused_by_short_context_return" if short else "fused_indexer_q_weights",
            "readiness_consumer": ready, "readiness_edge_reads_output": False,
            "scope": "source BF16 replicated projection, tensor use versus stream readiness; runtime backend, library scratch and physical allocator separate"}
        record["source"] = "indexer-weights-dataflow"
        record["attributes"]["source_dataflow"] = flow
        consumer_inputs = {}
        if consumer is not None:
            q_projection = prefix + "indexer-wq-b" + suffix
            ports = [{"producer": q_projection, "output_role": "output", "dtype": "bf16", "source_shape": [rows, heads * dimension],
                      "shape": [rows, heads, dimension], "strides_elements": [heads * dimension, dimension, 1],
                      "offset_elements": 0, "view_allocates": False},
                     {"producer": record["node_id"], "output_role": "output", "dtype": "bf16", "shape": [rows, heads],
                      "strides_elements": [heads, 1], "offset_elements": 0}]
            by_id[consumer]["attributes"]["source_tensor_inputs"] = ports
            consumer_inputs[consumer] = ports
        record["source_alias_dependency_mode"] = "mhc_layer_input_completion"
        record["source_alias_account"] = {"declaration": flow, "direct_input_producers": [parent],
            "producer_outputs": [{"node_id": parent, **{field: by_id[parent][field] for field in ("rank", "kind", "shape", "dtype")}}],
            "direct_consumers": sorted([ready] + ([] if consumer is None else [consumer])), "consumer_tensor_inputs": consumer_inputs,
            "lowered_calls": [{"kind": "GEMM", "variant": "gemm", "rank": record["rank"], "dtype": "bf16",
                "shape": {**record["shape"], "dtype_bytes": 2},
                # Homogeneous GEMM carries BF16 in its dtype; the public
                # contract adds explicit unquantized weight fields only
                # for mixed output. Projection checks still verify B's
                # effective dtype independently of this representation.
                "attributes": {"output_dtype": None},
                "dependencies": [{"source_node_id": parent, "rank": record["rank"], "terminal_source_call": True,
                    "layer_input_writes": [{"role": "layer_input", "dtype": "bf16", "shape": [rows, hidden],
                                            "bytes": rows * hidden * 2, "access": "write"}]}]}]}


def _attach_mhc_state_bindings(raw, cases, operators):
    """Bind the decoder's three carried tensors independently of sublayer output."""
    h, c = raw["hidden_size"], raw["hc_mult"]
    by_id = {record["node_id"]: record for record in operators}
    rows_by_phase = {case["phase"]: sum(request["query"] for request in case["requests"]) for case in cases}
    selected = [record for record in operators if record["kind"] == "vector"
                and record["variant"] in ("mhc_pre", "mhc_post_pre", "mhc_post", "hc_head")]

    def tensor(role, dtype, shape, strides):
        return {"output_role": role, "dtype": dtype, "shape": shape, "strides_elements": strides, "offset_elements": 0}

    def outputs(rows, variant):
        residual = tensor("residual", "bf16", [rows, c, h], [c * h, h, 1])
        normalized = tensor("layer_input", "bf16", [rows, h], [h, 1])
        if variant == "mhc_post":
            return [residual]
        if variant == "hc_head":
            return [normalized]
        return [residual, tensor("post_mix", "fp32", [rows, c, 1], [c, 1, 1]),
                tensor("res_mix", "fp32", [rows, c, c], [c * c, c, 1]), normalized]

    for record in selected:
        record["attributes"]["source_tensor_outputs"] = outputs(rows_by_phase[record["node_id"].split(":", 1)[0]], record["variant"])
    for record in selected:
        phase, base = record["node_id"].split(":", 1)
        base = base.split(":tp-rank-", 1)[0]
        rows, rank = rows_by_phase[phase], record["rank"]
        suffix = f":tp-rank-{rank}" if rank else ""
        variant = record["variant"]
        state = None
        if variant == "mhc_pre":
            sublayer = phase + ":token-embedding:tp-all-reduce"
        elif variant == "hc_head":
            sublayer = phase + ":mhc-post" + suffix
        elif variant == "mhc_post":
            layer = raw["num_hidden_layers"] - 1
            sublayer = f"{phase}:layer-{layer}:moe-output:tp-all-reduce"
            state = f"{phase}:layer-{layer}:mhc-post-pre-ffn" + suffix
        else:
            layer_name, role = base.split(":")
            layer = int(layer_name.removeprefix("layer-"))
            if role == "mhc-post-pre-ffn":
                sublayer = f"{phase}:layer-{layer}:wo-b:tp-all-reduce"
                state = f"{phase}:layer-{layer}:" + ("mhc-pre" if layer == 0 else "mhc-post-pre-attn") + suffix
            elif role == "mhc-post-pre-attn" and layer > 0:
                sublayer = f"{phase}:layer-{layer - 1}:moe-output:tp-all-reduce"
                state = f"{phase}:layer-{layer - 1}:mhc-post-pre-ffn" + suffix
            else:
                raise ValueError("unknown source mHC recurrence role")
        if variant == "hc_head":
            ports = [{"producer": sublayer, **outputs(rows, "mhc_post")[0]}]
        else:
            ports = [{"producer": sublayer, "output_role": "rank_output", "collective_rank": rank,
                      "dtype": "bf16", "shape": [rows, h], "strides_elements": [h, 1], "offset_elements": 0}]
            if state is not None:
                ports.extend({"producer": state, **port} for port in outputs(rows, "mhc_post_pre")[:3])
        record["attributes"]["source_tensor_inputs"] = ports
        predecessors = sorted({port["producer"] for port in ports})
        record["required_ancestors"] = predecessors
        record["input_binding_source"] = "mhc-state-bindings"
        record["source_input_binding_account"] = {"direct_input_producers": predecessors,
            "input_tensor_ports": ports,
            "producer_tensor_outputs": [{"producer": parent, "outputs": by_id[parent]["attributes"]["source_tensor_outputs"]}
                for parent in predecessors if by_id[parent]["kind"] != "collective"],
            "external_lowered_producers": [{"source_node_id": parent, "terminal_source_call": True} for parent in predecessors]}


def _attach_o_projection_dataflow(raw, cases, operators):
    """Source o_padded slice, grouped FP8/scales views and token-major C."""
    by_id = {record["node_id"]: record for record in operators}
    case_map = {case["phase"]: case for case in cases}
    heads, dim, groups = raw["num_attention_heads"] // 4, raw["head_dim"], raw["o_groups"] // 4
    padded = 64 if heads <= 64 else 128
    width, rank_width = heads * dim // groups, raw["o_lora_rank"]
    for record in operators:
        if record["variant"] != "dsv4_inv_rope_fp8_quant":
            continue
        phase = record["node_id"].split(":", 1)[0]
        requests = case_map[phase]["requests"]
        rows = sum(request["query"] for request in requests)
        phases = case_map[phase]["attention_phases"]
        decode_rows = sum(request["query"] for request, mode in zip(requests, phases, strict=True) if mode == "decode")
        aligned, blocks = -(-rows // 4) * 4, width // 128
        inverse, wo_a_id = record["node_id"], record["node_id"].replace(":wo-a-inv-rope-quant", ":wo-a")
        wo_b_id = wo_a_id.replace(":wo-a", ":wo-b")
        wo_a, wo_b = by_id[wo_a_id], by_id[wo_b_id]
        inputs = []
        for parent in record["required_ancestors"]:
            producer = by_id[parent]
            prefill = producer["variant"] == "dsv4_sparse_prefill"
            count = sum(request["query"] for request, mode in zip(requests, phases, strict=True) if (mode == "prefill") == prefill)
            start = decode_rows if producer["variant"] == "dsv4_sparse_prefill" else 0
            output = {"output_role": "attention_output_region", "dtype": "bf16", "base": "attention.o_padded",
                "base_shape": [rows, padded, dim], "shape": [count, padded, dim],
                "strides_elements": [padded * dim, dim, 1], "offset_elements": start * padded * dim}
            producer["attributes"]["source_tensor_outputs"] = [output]
            inputs.append({"producer": parent, **output, "shape": [count, heads, dim]})
        # Source forward calls prefill before decode, although row storage
        # retains the scheduler's decode-prefix order.
        inputs.sort(key=lambda port: 0 if by_id[port["producer"]]["variant"] == "dsv4_sparse_prefill" else 1)
        flow = {"program": "vllm_0_30_inverse_rope_output_objects", "hardware_domain": "CUDA SM90",
            "source_input": {"base": "attention.o_padded", "base_shape": [rows, padded, dim],
                "shape": [rows, heads, dim], "strides_elements": [padded * dim, dim, 1], "offset_elements": 0, "slice_allocates": False},
            "positions": {"dtype": "int64", "shape": [rows]},
            "cos_sin_cache": {"dtype": "fp32", "row_width": raw["qk_rope_head_dim"]},
            "value_buffer": {"dtype": "fp8", "shape": [groups, rows, width], "strides_elements": [rows * width, width, 1],
                "initialized_by_allocation": False, "aliases_input": False},
            "scale_buffer": {"dtype": "fp32", "backing_elements": groups * blocks * aligned,
                "shape": [groups, rows, blocks], "strides_elements": [blocks * aligned, 1, aligned],
                "initialized_by_allocation": False, "padding_rows": aligned - rows,
                "padding_elements_written_zero": groups * blocks * (aligned - rows)},
            "returned_transposes_allocate": False, "global_fp32_input_copy": False,
            "kernel_intermediates": "SSA/register values; no caller global intermediate", "quantize": True,
            "tma_aligned_scales": False, "mutates_input": False,
            "scope": "source explicit tensors, views and padding stores; not physical allocator, compiler scratch or runtime"}
        record["attributes"].update(source_dataflow=flow, source_tensor_inputs=inputs, source_tensor_outputs=[
            {"output_role": "o_projection_input", "dtype": "fp8", "shape": [rows, groups, width],
             "strides_elements": [width, rows * width, 1], "offset_elements": 0},
            {"output_role": "scales", "dtype": "fp32", "shape": [rows, groups, blocks],
             "strides_elements": [1, blocks * aligned, aligned], "offset_elements": 0}])
        wo_a["attributes"].update(source_dataflow={"program": "vllm_0_30_fp8_o_projection_objects",
            "equation": "bhr,hdr->bhd", "recipe": [1, 128, 128],
            "weight": {"dtype": "fp8", "shape": [groups, rank_width, width]},
            "weight_scales": {"dtype": "fp32", "shape": [groups, -(-rank_width // 128), blocks],
                "source_parameter": "weight_scale if present else weight_scale_inv"},
            "output_buffer": {"dtype": "bf16", "shape": [rows, groups, rank_width],
                "strides_elements": [groups * rank_width, rank_width, 1], "initialized_by_allocation": False, "aliases_inputs": False},
            "consumer_flatten": {"shape": [rows, groups * rank_width], "strides_elements": [groups * rank_width, 1], "allocates": False},
            "extra_quantizer": False, "mutates_inputs": False, "selected_runtime_backend": None,
            "scope": "source FP8 einsum caller objects; library internal scratch, compiler and actual provider separate"},
            batch_layout="einsum_bhr_hdr_bhd", a_batch_stride_elements=rows * width,
            b_batch_stride_elements=width * rank_width, c_batch_stride_elements=rank_width,
            source_tensor_inputs=[{"producer": inverse, **port} for port in record["attributes"]["source_tensor_outputs"]],
            source_tensor_outputs=[{"output_role": "output", "dtype": "bf16", "shape": [rows, groups, rank_width],
                "strides_elements": [groups * rank_width, rank_width, 1], "offset_elements": 0}])
        wo_b["attributes"]["source_tensor_inputs"] = [{"producer": wo_a_id, "output_role": "output", "dtype": "bf16",
            "base_shape": [rows, groups, rank_width], "shape": [rows, groups * rank_width],
            "strides_elements": [groups * rank_width, 1], "offset_elements": 0, "view_allocates": False}]
        for target in (record, wo_a):
            parents = sorted(target["required_ancestors"])
            consumers = sorted(item["node_id"] for item in operators if target["node_id"] in item["required_ancestors"])
            vector = target is record
            attributes = {"quant_group_size": 128} if vector else {
                "batch_layout": "einsum_bhr_hdr_bhd", "a_batch_stride_elements": rows * width,
                "b_batch_stride_elements": width * rank_width, "c_batch_stride_elements": rank_width,
                "weight_dtype": "fp8", "weight_group_size": 128, "output_dtype": None}
            shape = {**target["shape"], "dtype_bytes": 1 if vector else 2}
            if vector:
                shape["element_count"] = rows * heads * dim
            target["source"] = "o-projection-objects"
            target["source_alias_dependency_mode"] = "source_completion"
            target["source_alias_account"] = {"declaration": target["attributes"]["source_dataflow"],
                "direct_input_producers": parents,
                "producer_outputs": [{"node_id": parent, **{field: by_id[parent][field] for field in ("rank", "kind", "shape", "dtype")}} for parent in parents],
                "direct_consumers": consumers, "input_tensor_ports": target["attributes"]["source_tensor_inputs"],
                "output_tensor_roles": target["attributes"]["source_tensor_outputs"],
                "consumer_tensor_inputs": {consumer: by_id[consumer]["attributes"]["source_tensor_inputs"] for consumer in consumers},
                "lowered_calls": [{"kind": "VECTOR" if vector else "GEMM", "variant": target["variant"], "rank": target["rank"],
                    "dtype": target["dtype"], "shape": shape, "attributes": attributes,
                    "dependencies": [{"source_node_id": parent, "terminal_source_call": True} for parent in parents]}]}


def _attach_mhc_wrapper_alias_accounts(raw, cases, operators):
    """Source caller objects plus independently checked stream and typed-stage bindings."""
    by_id = {record["node_id"]: record for record in operators}
    rows_by_phase = {case["phase"]: sum(request["query"] for request in case["requests"]) for case in cases}
    selected = [record for record in operators if "source_input_binding_account" in record]
    for record in selected:
        node_id, variant = record["node_id"], record["variant"]
        phase = node_id.split(":", 1)[0]
        group = "head" if variant == "hc_head" else "post" if variant == "mhc_post" else "ffn" if ":mhc-post-pre-ffn" in node_id else "attention"
        record["attributes"]["source_dataflow"] = mhc_wrapper_object_reference(raw, rows_by_phase[phase], variant, group)
        record["source"] = "mhc-wrapper-objects"
        if variant in ("mhc_pre", "mhc_post_pre"):
            for consumer in operators:
                if node_id not in consumer["required_ancestors"] or consumer["kind"] not in ("router", "fused_moe", "linear"):
                    continue
                if consumer["kind"] == "linear" and ":shared-experts:gate-up:" not in consumer["node_id"]:
                    continue  # Attention entry ports were attached by their own source contract.
                consumer["attributes"]["source_tensor_inputs"] = [{"producer": node_id, "output_role": "layer_input", "dtype": "bf16",
                    "shape": [rows_by_phase[phase], raw["hidden_size"]], "strides_elements": [raw["hidden_size"], 1], "offset_elements": 0}]
        elif variant == "hc_head":
            consumer = phase + ":final-norm" + (f":tp-rank-{record['rank']}" if record["rank"] else "")
            by_id[consumer]["attributes"]["source_tensor_inputs"] = [{"producer": node_id, "output_role": "layer_input", "dtype": "bf16",
                "shape": [rows_by_phase[phase], raw["hidden_size"]], "strides_elements": [raw["hidden_size"], 1], "offset_elements": 0}]
    for record in selected:
        consumers = sorted(item["node_id"] for item in operators if record["node_id"] in item["required_ancestors"])
        consumer_inputs = {consumer: by_id[consumer]["attributes"]["source_tensor_inputs"] for consumer in consumers}
        if "mhc_nonfused_forms" in record:
            conditional = record["mhc_nonfused_forms"]
            forms = [form["stages"] for form in conditional["forms"]
                     if conditional["selected_backend"] is None or form["backend"] == conditional["selected_backend"]]
        else:
            forms = [record["lowering_account"]]
        tensor_forms = [[{"kind": stage["kind"], "variant": stage["variant"], "dtype": stage["dtype"],
                          "rank": record["rank"], "required_shape": stage["required_shape"],
                          "mhc_stage_work": stage["mhc_stage_work"]} for stage in form] for form in forms]
        record["source_alias_mode"] = "mhc_wrapper_objects"
        record["source_alias_tensor_forms"] = tensor_forms
        record["source_alias_account"] = {"declaration": record["attributes"]["source_dataflow"],
            "source_stream_bindings": record["source_input_binding_account"], "direct_consumers": consumers,
            "consumer_tensor_inputs": consumer_inputs, "lowered_tensor_form_compatible": True}


def short_topk_source_reference(rows, width):
    """Independent scalar/source tensor account of the supplied-output fill."""
    padded = 2 ** ((width - 1).bit_length())
    return {"source_symbol": "deepseek_v4._fill_short_context_topk_indices",
        "positions": {"dtype": "int64", "shape": [rows], "read_bytes": rows * 8, "read_only": True},
        "output": {"dtype": "int32", "shape": [rows, width], "strides_elements": [width, 1],
                   "write_bytes": rows * width * 4, "supplied_output_pointer": True,
                   "reads_previous_output": False, "writes_sentinels_too": True},
        "program": {"compressed_count": "(position + 1) // 4", "value": "column if column < compressed_count else -1",
                    "store_mask": "column < topk_capacity", "source_vector_width": padded,
                    "logical_position_adds": rows, "logical_compression_divides": rows,
                    "logical_value_comparisons": rows * width, "logical_value_selects": rows * width,
                    "padded_source_value_comparisons": rows * padded},
        "per_call_tensor_allocations": [], "global_tensor_temporaries": [],
        "local_values": ["offsets", "compressed_count", "selected_index_or_sentinel"],
        "actual_compiler_or_runtime_bound": False,
        "scope": "supplied-output source fill; source tensor temporaries only, not register spills, allocator addresses or native instructions"}


def partial_state_tensor_reference(rows, width):
    """Independent save_partial caller payload; actual cache alias remains open."""
    return {"projection_dtype": "fp32", "projection_shape": [rows, width * 2],
        "split_views": [{"role": role, "dtype": "fp32", "shape": [rows, width],
            "strides_elements": [width * 2, 1], "offset_elements": offset, "allocates": False}
            for role, offset in (("kv", 0), ("score", width))],
        "ape_read_dtype": "fp32", "positions_dtype": "int64", "slot_dtype": "int64",
        "source_read_bytes": 4 * rows * (width * 2 + width) + 16 * rows,
        "source_write_bytes": 4 * rows * width * 2, "score_ape_fp32_additions": rows * width,
        "writes": [{"role": "kv_state", "dtype": "fp32", "record_offset_elements": 0, "elements_per_row": width},
                   {"role": "score_state", "dtype": "fp32", "record_offset_elements": width, "elements_per_row": width}],
        "state_write_in_place": True, "allocates_state_cache": False, "returns_tensor": False,
        "global_tensor_temporaries": [], "pdl_grid_dependency_used": False,
        "physical_page_alias_or_cross_step_reuse_bound": False,
        "scope": "all-valid source payload and FP32 caller objects only; persistent page alias, reuse, history and actual provider remain unchecked"}


def finish_partial_state_reference(dimension, ratio, stage, rows):
    if stage == "store":
        return None
    width = dimension * (1 + (ratio == 4))
    return {"dtype": "fp32", "head_dim": dimension, "compress_ratio": ratio,
        "state_record_elements": width * 2, "state_width": width,
        "state_address_program": {
            "block_rows_source": "state_metadata.block_size", "block_rows": None if ratio == 4 else 8,
            "constructor_default": 4 if ratio == 4 else 8,
            "C4_domain": "positive multiple of 4", "C128_supported_program_requires_8": True,
            "page_index": "absolute_position // block_rows", "row_offset": "absolute_position % block_rows",
            "same_block_rows_as_save_partial": True},
        "value_columns": "(window_row // 4) * head_dim + column" if ratio == 4 else "column",
        "score_columns": "state_width + value_column", "valid_window_rows": rows,
        "payload_read_bytes": rows * dimension * 2 * 4,
        "cache_mutated_by_finish": False, "global_cast_or_copy_before_read": False,
        "scope": "source persistent KV/score reads only; C128 store-stage compressed_kv and cache metadata are separate operands"}


def _attach_partial_state_caller_frames(operators, sources):
    """Bind the caller only with complete original-source frame definitions."""
    if "tensor_frame" not in sources["compressor-state-dataflow"]:
        return
    for record in operators:
        if record["variant"] != "dsv4_compressor_state_cache":
            continue
        flow = record["attributes"]["source_dataflow"]
        rows, width = record["shape"]["seq_len"], record["shape"]["state_bytes"] // 8
        projection_input, = record["attributes"]["source_tensor_inputs"]
        parent = projection_input["producer"]
        projection = {"producer": parent, "output_role": "output", "dtype": "fp32", "shape": [rows, 2 * width],
                      "strides_elements": [2 * width, 1], "offset_elements": 0}
        frame = {
            "program": "vllm_0_30_save_partial_caller_objects", "projection": projection,
            "views": [{"base": parent, "role": role, "dtype": "fp32", "shape": [rows, width],
                       "strides_elements": [width * 2, 1], "offset_elements": offset, "allocates": False}
                      for role, offset in (("kv", 0), ("score", width))],
            "ape": {"module": flow["state_owner"], "parameter": "ape", "dtype": "fp32",
                    "shape": [flow["compress_ratio"], width], "row": "position % compress_ratio"},
            "state_binding": {"module": flow["state_owner"], "source": "self.state_cache.kv_cache",
                "bind_input_shape": "[num_blocks,1,metadata_block_size,state_record_elements]",
                "view": "squeeze(1)", "allocates": False, "dtype": "fp32",
                "state_record_elements": 2 * width, "strides_elements": ["state_cache_stride0", "state_cache_stride1", 1],
                "storage_offset_and_backing": "inherited from supplied cache tensor; may share backing with other cache groups",
                "page_row": "divmod(slot_mapping[token_index], metadata_block_size)",
                "column_writes": [[0, width], [width, 2 * width]], "mutated_in_place": True},
            "positions": {"dtype": "int64", "elements": rows},
            "state_slots": {"dtype": "int64", "elements": rows},
            "global_tensor_temporaries": [], "returns_tensor": False,
            "source_order": ["projection", "split_views", "save_partial", "finish_if_dispatched"],
            "launch_pdl": False, "actual_pages_reuse_GPU_fences_or_provider_bound": False}
        record["partial_state_frame"] = frame
        readers = [other for other in operators if record["node_id"] in other["required_ancestors"]]
        record["source_alias_mode"] = "partial_state_writer_objects"
        record["source_alias_account"] = {
            "frame": frame, "projection_input": [projection],
            "projection_output": {name: value for name, value in projection.items() if name not in ("producer", "output_role")}
                | {"aliases_input_or_weight": False},
            "state_output": record["attributes"]["source_partial_state_output"],
            "source_slot_page_program": flow["slot_mapping_source"],
            "direct_input_producers": sorted(record["required_ancestors"]),
            "direct_consumers": sorted(other["node_id"] for other in readers),
            "finish_state_inputs": {other["node_id"]: other["attributes"]["source_partial_state_input"]
                for other in readers if other["variant"] == "dsv4_compressed_state_cache"},
            "lowered_writer_form_compatible": True,
            "projection_ready_before_write": True,
            "input_readiness": {source: True for source in sorted(record["required_ancestors"])},
            "scope": "source views and in-place writes for symbolic backing/strides; actual page allocation and cross-step reuse remain separate obligations"}


def _attach_partial_state_ports(raw, operators):
    """Typed module-state edges, kept separate from unresolved page/reuse facts."""
    for record in operators:
        if record["variant"] != "dsv4_compressor_state_cache":
            continue
        flow = record["attributes"]["source_dataflow"]
        rows, width = record["shape"]["seq_len"], record["shape"]["state_bytes"] // 8
        component = flow["state_owner"]["component"]
        if component == "attention.indexer.compressor":
            dimension = raw["index_head_dim"]
        elif component == "attention.compressor":
            dimension = raw["head_dim"]
        else:
            raise ValueError("unknown independent partial-state component: " + component)
        if width != dimension * (1 + (flow["compress_ratio"] == 4)):
            raise ValueError("independent partial-state component has an inconsistent width")
        output = {"output_role": "partial_state_cache", "dtype": "fp32", "state_owner": flow["state_owner"],
            "state_record_elements": 2 * width, "value_offset_elements": 0, "score_offset_elements": width,
            "existing_persistent_storage": True, "returns_tensor": False,
            "physical_page_alias_or_cross_step_reuse_bound": False}
        record["attributes"]["source_partial_state_output"] = output
        record["partial_state_objects"] = partial_state_tensor_reference(rows, width)
        readers = [other for other in operators if other["variant"] == "dsv4_compressed_state_cache"
                   and record["node_id"] in other["required_ancestors"]]
        for reader in readers:
            ratio = reader["shape"]["compress_ratio"]
            if reader["rank"] != record["rank"] or reader["attributes"]["partial_state_owner"] != flow["state_owner"]:
                raise ValueError("independent finish state owner/rank mismatch")
            reader["attributes"]["source_partial_state_input"] = {"producer": record["node_id"], **output,
                "read_only": True, "global_dtype_cast_allocates": False,
                "C128_partial_state_reader_stage": "compress" if ratio == 128 else None}
            for stage in reader["lowering_account"]:
                source_stage = stage["finish_stage"] if "finish_stage" in stage else None
                stage["partial_state_input"] = finish_partial_state_reference(dimension, ratio, source_stage,
                    stage["finish_work"]["valid_window_rows"])


def _attach_finish_caller_objects(raw, cases, operators, sources):
    """Independent source caller/cache object program, with symbolic pages."""
    if "finish_frame" not in sources["compressor-state-dataflow"]:
        return
    by_id = {record["node_id"]: record for record in operators}
    case_map = {case["phase"]: case for case in cases}
    ports = {record["node_id"]: [] for record in operators}
    selected = []
    for record in operators:
        if record["variant"] != "dsv4_compressed_state_cache":
            continue
        selected.append(record)
        attrs = record["attributes"]
        state = attrs["source_partial_state_input"]
        ratio = record["shape"]["compress_ratio"]
        component = attrs["partial_state_owner"]["component"]
        if component == "attention.indexer.compressor":
            indexer = True
        elif component == "attention.compressor":
            indexer = False
        else:
            raise ValueError("independent finish caller has unknown compressor owner: " + component)
        dim = raw["index_head_dim"] if indexer else raw["head_dim"]
        case = case_map[record["node_id"].split(":", 1)[0]]
        positions = [position for request in case["requests"]
                     for position in range(request["computed"], request["computed"] + request["query"])]
        rows = len(positions)
        active = [row for row, position in enumerate(positions) if (position + 1) % ratio == 0]
        owner = {**attrs["partial_state_owner"], "component": "attention.indexer.k_cache" if indexer else "attention"}
        output = {"output_role": "compressed_cache", "dtype": "uint8", "cache_owner": owner,
            "storage": "existing bound module kv_cache", "mutated_in_place": True, "returns_tensor": False,
            "slot_source": "k_cache_metadata.slot_mapping", "slot_dtype": "int64",
            "state_slots_are_destination_slots": False, "written_token_rows": active,
            "page_rows": "kv_cache.shape[1]", "page_stride_bytes": "kv_cache.stride(0)",
            "backing": "inherited cache binding; may overlay other cache groups",
            "actual_pages_or_cross_step_reuse_bound": False}
        temporary = [] if ratio == 4 else [{"object": record["node_id"] + ":compressed-kv", "role": "compressed_kv",
            "dtype": "fp32", "shape": [rows, dim], "strides_elements": [dim, 1], "offset_elements": 0,
            "allocation": "fresh torch.empty", "allocation_initializes_payload": False,
            "aliases_inputs_parameters_or_cache": False, "written_token_rows": active, "read_token_rows": active,
            "producer_stage": "compress", "consumer_stage": "store", "lifetime": "compress result through store enqueue"}]
        stages = ([{"stage": "compress", "state_input": state, "output_object": temporary[0]["object"]},
                   {"stage": "store", "input_object": temporary[0]["object"], "output_cache": output}]
                  if ratio == 128 else [{"stage": None, "state_input": state, "output_cache": output}])
        frame = {"program": "vllm_0_30_finish_caller_objects", "state_input": state,
            "condition": "CUDA packed FP8; real token rows, nonnegative state slots and valid destination slots at completed boundaries; no DP/graph padding",
            "norm_weight": {"module": attrs["partial_state_owner"], "parameter": "norm.weight", "dtype": "bf16", "shape": [dim]},
            "rms_norm_eps": raw["rms_norm_eps"],
            "rotary_cache": {"source": "rotary_emb.cos_sin_cache", "dtype": "fp32", "row_width": raw["qk_rope_head_dim"],
                             "row": "(position // compress_ratio) * compress_ratio"},
            "metadata": {"positions": {"dtype": "int64", "values": positions, "source": "same model positions tensor as save_partial"},
                "state_slots": {"dtype": "int64", "source": "state_metadata.slot_mapping"},
                "destination_slots": {"dtype": "int64", "source": "k_cache_metadata.slot_mapping"},
                "request_indices": {"dtype": "int32", "source": "state_metadata.token_to_req_indices"},
                "state_block_table": {"dtype": "int32", "source": "state_metadata.block_table"}},
            "output_cache": output, "global_tensor_temporaries": temporary, "stages": stages,
            "state_mutated_by_finish": False, "returns_tensor": False, "launch_pdl": False,
            "actual_allocator_GPU_fences_or_provider_bound": False}
        attrs["source_finish_frame"] = frame
        attrs["source_compressed_cache_output"] = output
        readers = [other for other in operators if record["node_id"] in other["required_ancestors"]]
        for reader in readers:
            allowed = (("dsv4_indexer_prefill", "dsv4_indexer_decode", "dsv4_short_context_topk")
                       if indexer else ("dsv4_sparse_prefill", "dsv4_sparse_decode", "dsv4_indexer_prefill", "dsv4_indexer_decode"))
            if reader["rank"] != record["rank"] or reader["variant"] not in allowed:
                raise ValueError("independent finish caller requires source-bound same-rank cache readers")
            order_only = (reader["variant"] == "dsv4_short_context_topk" or not indexer
                          and reader["variant"] in ("dsv4_indexer_prefill", "dsv4_indexer_decode"))
            access = "source order only; no cache payload read" if order_only else "read cache"
            ports[reader["node_id"]].append({"producer": record["node_id"], **output, "consumer_access": access})
    for key, values in ports.items():
        if values:
            by_id[key]["attributes"]["source_compressed_cache_inputs"] = sorted(values, key=lambda port: port["producer"])
    fields = ("stage", "source_temporary", "store_program", "conversion_value_counts", "metadata_reads")
    for record in selected:
        frame = record["attributes"]["source_finish_frame"]
        readers = [other for other in operators if record["node_id"] in other["required_ancestors"]]
        record["source_alias_mode"] = "finish_caller_objects"
        record["source_alias_account"] = {
            "frame": frame, "output_cache": frame["output_cache"],
            "partial_state_output": by_id[frame["state_input"]["producer"]]["attributes"]["source_partial_state_output"],
            "direct_input_producers": sorted(record["required_ancestors"]),
            "direct_consumers": sorted(reader["node_id"] for reader in readers),
            "consumer_cache_inputs": {reader["node_id"]: reader["attributes"]["source_compressed_cache_inputs"] for reader in readers},
            "source_stage_programs": [{name: stage["finish_work"][name] for name in fields} for stage in record["lowering_account"]],
            "lowered_stage_forms_compatible": True, "state_ready_before_finish": True,
            "store_reads_compress_result": True, "cache_readers_follow_final_store": True}


def swa_insert_tensor_reference(rows, heads, padded):
    """Independent packed-cache/Q object program; B and S remain source parameters."""
    return {"program": "vllm_0_30_packed_swa_insert_objects",
        "condition": "eager dict metadata, BF16 Q/KV, uint8 fp8_ds_mla cache, no DP padding, all slots nonnegative",
        "q_input": {"dtype": "bf16", "shape": [rows, heads, 512], "read_only": True},
        "kv_input": {"dtype": "bf16", "shape": [rows, 512], "read_only": True},
        "metadata": {"positions": {"dtype": "int64", "shape": [rows]},
            "slots": {"dtype": "int64", "shape": [rows], "nonnegative": True},
            "cos_sin": {"dtype": "fp32", "row_width": 64}},
        "q_output": {"dtype": "bf16", "shape": [rows, padded, 512], "strides_elements": [padded * 512, 512, 1],
            "fresh": True, "aliases_inputs_or_cache": False, "every_element_written": True,
            "zero_padding_heads": [heads, padded], "allocation_initialization_bytes": 0},
        "cache": {"dtype": "uint8", "existing_storage": True, "wrapper_view_allocates": False,
            "wrapper_view": "[num_blocks, -1]", "read_payload_bytes": 0, "write_payload_bytes": rows * (448 + 128 + 8),
            "block_base_bytes": "(slot // B) * S", "data_offset_bytes": "(slot % B) * 576",
            "rope_offset_bytes": "(slot % B) * 576 + 448",
            "scale_offset_bytes": "B * 576 + (slot % B) * 8", "B": "cache_block_size", "S": "cache.stride(0) in bytes",
            "nope_bytes": 448, "rope_bf16_bytes": 128, "real_scale_bytes": 7, "zero_scale_padding_bytes": 1,
            "source_prequant_rounding": "BF16 roundtrip before absmax/quantization",
            "allocated_or_zeroed_by_this_call": False, "physical_slots_or_page_stride_bound": False},
        "apply_q_norm": True, "kv_mxfp8": False, "global_tensor_temporaries": [],
        "kernel_local_arrays_are_global_temporaries": False,
        "scope": "source caller objects and conditional cache address program; not page allocation, cross-step ownership, compiler spills or runtime provider"}


def swa_insert_work_reference(rows, heads, padded):
    """Independent count of the retained BF16/packed-SWA slot program."""
    q = rows * heads * 512
    scalar = (q * 3 + rows * heads * (32 * 5 + 2)
        + rows * (heads + 1) * 64 * 3 + rows * (32 + 448 + 28))
    return {"objects": swa_insert_tensor_reference(rows, heads, padded),
        "demand": {"compute_tensor": 0, "compute_reduction": 0, "compute_fma": scalar,
            "compute_alu": 0, "compute_sfu": rows * (heads + 32 * 2)},
        "typed_actions": {"bf16_to_fp32": rows * (heads + 2) * 512,
            "fp32_to_bf16": rows * (heads * 512 + 512 + 64), "fp32_to_fp8": rows * 448,
            "fp32_to_uint8": rows * 7, "fp32_abs": rows * 512,
            "fp32_maximum": rows * (512 + 32 * 2 + 32 + 448 + 28),
            "fp32_minimum": rows * (448 + 28), "fp32_ceil": rows * 32, "fp32_negate": rows * 32},
        "read_write_bytes": {"read": rows * (heads * 512 * 2 + 512 * 2 + 8 + 8 + 64 * 4),
            "write": rows * (padded * 512 * 2 + 448 + 64 * 2 + 8)}}


def _attach_swa_insert_dataflow(raw, cases, operators):
    """Bind fresh full Q output to the source's decode-prefix/prefill-suffix views."""
    by_id = {record["node_id"]: record for record in operators}
    case_by_phase = {case["phase"]: case for case in cases}
    for record in operators:
        if record["variant"] != "dsv4_qnorm_rope_kv_insert":
            continue
        rows, heads, padded = (record["shape"][name] for name in ("seq_len", "num_query_heads", "padded_query_heads"))
        inputs = record["attributes"]["source_tensor_inputs"]
        output = {"output_role": "padded_q", "dtype": "bf16", "shape": [rows, padded, 512],
                  "strides_elements": [padded * 512, 512, 1], "offset_elements": 0}
        flow = {"program": "vllm_0_30_packed_swa_insert_io", "q_input": inputs[0], "kv_input": inputs[1],
            "inputs_mutated": False, "output": output, "q_output_fresh": True, "q_padding_heads_written_zero": [heads, padded],
            "epsilon": raw["rms_norm_eps"], "apply_q_norm": True, "kv_mxfp8": False,
            "cache_owner": "same rank-local layer swa_cache_layer.kv_cache", "cache_view_allocates": False,
            "cache_mutated_in_place": True, "cache_read_by_attention_instead_of_input_kv": True,
            "positions": {"dtype": "int64", "shape": [rows]}, "slot_mapping": {"dtype": "int64", "shape": [rows], "all_valid": True},
            "source_global_tensor_temporaries": [], "cache_page_ownership_or_runtime_bound": False}
        record["attributes"].update(source_dataflow=flow, source_tensor_outputs=[output])
        record["source"] = "swa-insert-objects"
        record["source_alias_mode"] = "swa_insert_objects"
        record["swa_insert_objects"] = swa_insert_tensor_reference(rows, heads, padded)
        case = case_by_phase[record["node_id"].split(":", 1)[0]]
        decode_rows = sum(request["query"] for request, mode in zip(case["requests"], case["attention_phases"], strict=True) if mode == "decode")
        readers = [other for other in operators if record["node_id"] in other["required_ancestors"]
                   and other["kind"] == "eager_attn" and other["variant"] in ("dsv4_sparse_decode", "dsv4_sparse_prefill")]
        reader_views = {}
        for reader in readers:
            decode = reader["variant"] == "dsv4_sparse_decode"
            count = sum(request["query"] for request, mode in zip(case["requests"], case["attention_phases"], strict=True) if (mode == "decode") == decode)
            reader_views[reader["node_id"]] = {"producer": record["node_id"], **output,
                "base_shape": [rows, padded, 512], "shape": [count, padded, 512],
                "offset_elements": (0 if decode else decode_rows) * padded * 512, "view_allocates": False,
                "decode_unsqueeze_allocates": False if decode else None,
                "cache_owner": "same rank-local layer swa_cache_layer.kv_cache", "input_kv_read_directly": False}
            reader["attributes"]["source_query_input"] = reader_views[reader["node_id"]]
        record["source_alias_account"] = {"declaration": flow, "direct_input_producers": sorted(record["required_ancestors"]),
            "input_tensor_ports": inputs, "output_tensor_roles": [output],
            "producer_outputs": [{"node_id": port["producer"], **{field: by_id[port["producer"]][field]
                for field in ("rank", "kind", "shape", "dtype")}} for port in inputs],
            "direct_consumers": sorted(other["node_id"] for other in operators if record["node_id"] in other["required_ancestors"]),
            "consumer_query_views": reader_views, "all_readers_after_insert": True,
            "lowered_objects_compatible": True,
            "lowered_readiness": sorted([{"source_node_id": port["producer"], "terminal_source_call": True}
                for port in inputs], key=lambda item: json.dumps(item, sort_keys=True))}


def _attach_route_dataflow(raw, operators, model_token_ids):
    """Returned FP32 weights/INT32 IDs bind directly to modular expert input."""
    by_id = {record["node_id"]: record for record in operators}
    for record in operators:
        if record["variant"] != "dsv4_moe_route":
            continue
        rows, experts = record["shape"]["rows"], raw["n_routed_experts"]
        node_id, rank = record["node_id"], record["rank"]
        phase = node_id.split(":", 1)[0]
        parent, = record["required_ancestors"]
        hashed = record["attributes"]["expert_id_selection"] == "hash_table"
        source = "vllm_0_30_cuda_hash_top6_pure_text" if hashed else "vllm_0_30_triton_top6_pure_text"
        readers = [other for other in operators if node_id in other["required_ancestors"]]
        consumer, = readers
        if consumer["kind"] != "fused_moe" or consumer["rank"] != rank:
            raise ValueError("independent routing has an unexpected expert consumer")
        logits = {"producer": parent, "output_role": "logits", "dtype": "fp32", "shape": [rows, experts],
                  "strides_elements": [experts, 1], "offset_elements": 0}
        by_id[parent]["attributes"]["source_tensor_outputs"] = [
            {name: value for name, value in logits.items() if name != "producer"}]
        outputs = [{"output_role": role, "dtype": dtype, "shape": [rows, 6],
                    "strides_elements": [6, 1], "offset_elements": 0}
                   for role, dtype in (("topk_weights", "fp32"), ("topk_ids", "int32"))]
        flow = {"program": "vllm_0_30_modular_route_tensor_io", "routing_source": source,
            "logits": logits, "hidden_states_payload_read": False,
            "shape_only_hidden_input": by_id[parent]["attributes"]["source_tensor_inputs"][0],
            "allocation_device": "rank-local hidden_states.device",
            "input_ids": {"owner": "model input_ids", "dtype": "int32", "shape": [rows],
                "values": None if model_token_ids is None else model_token_ids[phase], "aliases_embedding_input_ids": True,
                "dtype_conversion_allocates": False, "read_by_kernel": hashed},
            "parameter": {"owner": "gate.tid2eid" if hashed else "gate.e_score_correction_bias",
                          "scope": "same rank-local layer module as logits producer",
                          "dtype": "int32" if hashed else "fp32", "shape": [raw["vocab_size"], 6] if hashed else [experts],
                          "data_view_allocates": False},
            "outputs": outputs, "fresh_output_tensors": True, "outputs_alias_inputs_or_each_other": False,
            "consumer": consumer["node_id"], "outputs_live_through": "modular expert consumer",
            "postprocess": {"eplb_mapping": False, "ids_dtype_copy": False, "fused_shared_columns": 0, "capture_hook": None, "routing_replay_copy": False},
            "actual_allocator_or_runtime_bound": False}
        record["attributes"]["source_dataflow"] = flow
        record["attributes"]["source_tensor_outputs"] = outputs
        ports = [{"producer": node_id, **output} for output in outputs]
        consumer["attributes"]["source_routing_inputs"] = ports
        record["source"] = "route-tensor-dataflow"
        record["source_alias_mode"] = "v4_route_objects"
        record["route_tensor_objects"] = routing_tensor_reference(rows, experts, hashed)
        record["source_alias_account"] = {"declaration": flow, "direct_input_producers": [parent],
            "logits_producer_outputs": by_id[parent]["attributes"]["source_tensor_outputs"],
            "input_tensor_ports": [logits], "output_tensor_roles": outputs,
            "direct_consumers": [consumer["node_id"]], "consumer_routing_inputs": {consumer["node_id"]: ports},
            "model_input_ids": [None if model_token_ids is None else model_token_ids[phase]],
            "lowered_tensor_form_compatible": True,
            "lowered_readiness": [{"source_node_id": parent, "terminal_source_call": True}],
            "expert_reads_after_route": True, "expert_routing_types_compatible": True}


def _attach_short_topk_dataflow(raw, cases, operators):
    """One source model buffer per rank; only its fully written active views."""
    by_id = {record["node_id"]: record for record in operators}
    case_map = {case["phase"]: case for case in cases}
    previous = {}
    for record in operators:
        if record["variant"] != "dsv4_short_context_topk":
            continue
        node_id, rank = record["node_id"], record["rank"]
        phase = node_id.split(":", 1)[0]
        case = case_map[phase]
        rows = sum(request["query"] for request in case["requests"])
        width = raw["index_topk"]
        positions = [position for request in case["requests"]
                     for position in range(request["computed"], request["computed"] + request["query"])]
        decode_rows = sum(request["query"] for request, mode in zip(case["requests"], case["attention_phases"], strict=True) if mode == "decode")
        readers = sorted(other["node_id"] for other in operators if node_id in other["required_ancestors"])
        old_readers = previous[(phase, rank)] if (phase, rank) in previous else []
        storage = {"owner": "model.topk_indices_buffer", "rank": rank,
                   "dtype": "int32", "shape": ["max_num_batched_tokens", width],
                   "strides_elements": [width, 1], "capacity_rows_at_least": rows,
                   "allocation": "torch.empty once per model instance", "initialization_write_bytes": 0}
        flow = {"program": "vllm_0_30_short_context_topk_buffer", "buffer": storage,
            "positions": {"dtype": "int64", "shape": [rows], "values": positions,
                          "source": "declared source model input order", "read_only": True},
            "write_view": {"shape": [rows, width], "offset_elements": 0, "strides_elements": [width, 1]},
            "writes_all_active_entries": True, "reads_old_buffer_contents": False,
            "per_call_tensor_allocations": [], "return_values": [None, None, None],
            "previous_layer_readers": old_readers,
            "buffer_lifetime": "model instance; active prefix overwritten before each use",
            "actual_allocator_or_runtime_bound": False}
        ports = {}
        for reader in readers:
            consumer = by_id[reader]
            mode = consumer["variant"]
            if mode not in ("dsv4_sparse_prefill", "dsv4_sparse_decode"):
                raise ValueError("independent short topk buffer has an unexpected consumer")
            requests = [request for request, request_mode in zip(case["requests"], case["attention_phases"], strict=True)
                        if request_mode == ("decode" if mode == "dsv4_sparse_decode" else "prefill")]
            count = sum(request["query"] for request in requests)
            ports[reader] = {"producer": node_id, "buffer": storage, "dtype": "int32",
                "shape": [count, width], "strides_elements": [width, 1],
                "offset_elements": (0 if mode == "dsv4_sparse_decode" else decode_rows) * width,
                "slice_allocates": False,
                "use": "local-to-global helper input" if mode == "dsv4_sparse_decode" else "combine_topk_swa_indices input"}
            consumer["attributes"]["source_topk_input"] = ports[reader]
        parent, = record["required_ancestors"]
        record["attributes"]["source_dataflow"] = flow
        record["source"] = "short-context-topk-dataflow"
        record["source_alias_mode"] = "short_context_topk_buffer"
        record["short_fill_source_work"] = short_topk_source_reference(rows, width)
        record["source_alias_account"] = {"declaration": flow,
            "direct_input_producers": [parent], "cache_readiness_producer": {"node_id": parent,
                **{field: by_id[parent][field] for field in ("rank", "kind", "shape", "dtype")}},
            "direct_consumers": readers, "consumer_input_views": ports, "model_position_inputs": [positions],
            "lowered_tensor_form_compatible": True,
            "lowered_readiness": [{"source_node_id": parent, "terminal_source_call": True}],
            "consumer_reads_after_fill": True, "previous_reads_before_reuse": True}
        previous[(phase, rank)] = readers


def _attach_marlin_caller_objects(operators, projections, sources):
    """Independent complete source-frame IO/temporary account for modular MoE."""
    if "tensor_frame" not in sources["marlin-moe-source"]:
        return
    by_id = {record["node_id"]: record for record in operators}
    projection_by_id = {record["node_id"]: record for record in projections}
    program_fields = ("tensor_object_program", "modular_buffer_program", "workspace_zero",
                      "fp32_scratch_templates", "checkpoint_repack_source_program",
                      "weight_payload_per_active_expert", "routing_alignment", "gemm_metadata", "source_stage_order",
                      "source_views", "dequant_source_program", "semantic_fields", "source_condition",
                      "actual_provider_bound", "actual_workspace_capacity_bound", "declared_routing_is_checkpoint_assignment")
    for record in operators:
        if record["kind"] != "fused_moe" or record["variant"] != "full":
            continue
        route_id, = [parent for parent in record["required_ancestors"] if by_id[parent]["variant"] == "dsv4_moe_route"]
        route = by_id[route_id]
        hidden = route["attributes"]["source_dataflow"]["shape_only_hidden_input"]
        logits_parent, = route["required_ancestors"]
        layer = projection_by_id[logits_parent]["layer"]
        parameter_module = f"model.layers.{layer}.ffn.experts.routed_experts"
        readers = [item for item in operators if record["node_id"] in item["required_ancestors"]]
        consumer, = readers
        tokens, width = record["shape"]["tokens"], record["shape"]["hidden"]
        output = {"output_role": "output", "dtype": "bf16", "shape": [tokens, width],
                  "strides_elements": [width, 1], "offset_elements": 0}
        route_ports = record["attributes"]["source_routing_inputs"]
        declaration = {"program": "vllm_0_30_modular_marlin_tensor_io",
            "condition": "CUDA no-DP/EP synchronous modular Marlin W4A16, no LoRA; valid top6 routes",
            "parameter_module": parameter_module, "hidden": hidden, "routing": route_ports, "output": output,
            "inputs_mutated": False, "output_aliases_inputs_routes_parameters_or_arena": False,
            "input_reference_lifetime": "through modular call", "output_consumer": consumer["node_id"],
            "temporary_program": "marlin_source_work.tensor_object_program",
            "actual_allocator_lane_fence_or_provider_bound": False}
        record["attributes"]["source_tensor_inputs"] = [hidden]
        record["attributes"]["source_tensor_outputs"] = [output]
        record["attributes"]["source_moe_io"] = declaration
        record["source_alias_mode"] = "modular_marlin_objects"
        record["source_alias_account"] = {
            "declaration": declaration, "parameter_module": parameter_module,
            "direct_input_producers": sorted(record["required_ancestors"]),
            "producer_outputs": {parent: by_id[parent]["attributes"]["source_tensor_outputs"]
                for parent in record["required_ancestors"]},
            "input_tensor_ports": [hidden], "routing_tensor_ports": route_ports,
            "output_tensor_roles": [output], "direct_consumers": [consumer["node_id"]],
            "consumer_tensor_inputs": {consumer["node_id"]: consumer["attributes"]["source_tensor_inputs"]},
            "source_programs": {name: record["source_work"][name] for name in program_fields},
            "scratch_requests": record["scratch_allocation"],
            "scratch_runtime_claims": {"aggregate_peak_live_bytes": None, "actual_allocator_binding_verified": False},
            "declared_expert_axis_preserved": True,
            "lowered_tensor_form_compatible": True,
            "lowered_input_readiness": [{"source_node_id": parent, "terminal_source_call": True}
                for parent in sorted(record["required_ancestors"])],
            "consumer_reads_after_output": True}


def _attach_lm_head_objects(raw, operators, sources):
    """Independent unquantized head caller objects, across both CUDA paths."""
    if sources["bf16-lm-head-objects"]["aten_linear_schema"] is None:
        return
    by_id = {record["node_id"]: record for record in operators}
    for record in operators:
        if record["kind"] != "lm_head":
            continue
        node_id, rank = record["node_id"], record["rank"]
        phase = node_id.split(":", 1)[0]
        rows, columns, hidden = (record["shape"][name] for name in ("m", "n", "k"))
        suffix = "" if rank == 0 else f":tp-rank-{rank}"
        parent, consumer = phase + ":logits-boundary-select" + suffix, phase + ":lm-head:tp-all-gather"
        input_port = {"output_role": "sample_hidden_states", "dtype": "bf16", "shape": [rows, hidden]}
        output_port = {"output_role": "local_logits", "dtype": "bf16", "shape": [rows, columns]}
        flow = {"program": "vllm_0_30_bf16_lm_head_objects", "input": {"producer": parent, **input_port},
            "weight": {"module": "lm_head", "parameter": "weight", "dtype": "bf16", "shape": [columns, hidden],
                "scope": "rank-local persistent parameter", "forward_copy": False},
            "output": output_port, "consumer": consumer, "inputs_and_weight_mutated": False,
            "output_aliases_input_or_weight": False,
            "source_branches": [
                {"condition": "not VLLM_BATCH_INVARIANT", "program": "default_unquantized_gemm -> functional.linear", "parameter_views": []},
                {"condition": "VLLM_BATCH_INVARIANT; CUDA", "program": "linear_batch_invariant -> matmul_batch_invariant -> mm_batch_invariant -> matmul_persistent",
                 "parameter_views": [{"operation": "weight.t()", "shape": [hidden, columns], "allocates": False}]}],
            "source_wrapper_temporary_tensors": [], "output_lifetime": "through TP all-gather input read",
            "selected_runtime_backend": None, "numerical_equivalence_asserted": False,
            "native_library_scratch_allocator_or_output_strides_bound": False}
        by_id[parent]["attributes"]["source_tensor_outputs"] = [input_port]
        record["attributes"].update(weight_dtype="bf16", weight_group_size=0, has_bias=False,
            source_lm_head_io=flow, source_tensor_inputs=[flow["input"]], source_tensor_outputs=[output_port])
        participants = by_id[consumer]["attributes"]["participants"]
        if hidden != raw["hidden_size"] or columns * len(participants) != raw["vocab_size"]:
            raise ValueError("independent BF16 head must partition the complete configured vocabulary")
        head_ids = [phase + ":lm-head" + ("" if lane == 0 else f":tp-rank-{lane}") for lane in participants]
        ports = [{"producer": head, **output_port} for head in head_ids]
        by_id[consumer]["attributes"]["source_tensor_inputs"] = ports
        record["source_alias_mode"] = "bf16_lm_head_objects"
        record["source_alias_account"] = {"declaration": flow, "direct_input_producers": [parent],
            "input_tensor_ports": [flow["input"]], "producer_output_ports": [input_port], "output_tensor_roles": [output_port],
            "direct_consumers": [consumer], "gather_input_producers": head_ids, "gather_input_ports": ports,
            "lowered_head_compatible": True, "head_reads_after_row_selection": True, "gather_reads_after_all_heads": True}


def _attach_final_norm_objects(raw, operators):
    """Independent source requests for all H200 final RMSNorm alternatives."""
    by_id = {record["node_id"]: record for record in operators}
    for record in operators:
        if record["kind"] != "norm" or ":final-norm" not in record["node_id"]:
            continue
        node_id = record["node_id"]
        parent, = record["required_ancestors"]
        reader, = [item for item in operators if node_id in item["required_ancestors"]]
        rows, width = record["shape"]["rows"], raw["hidden_size"]
        input_port = {"producer": parent, "output_role": "layer_input", "dtype": "bf16",
            "shape": [rows, width], "strides_elements": [width, 1], "offset_elements": 0}
        output = {"output_role": "normalized_hidden_states", "dtype": "bf16", "shape": [rows, width]}
        temporaries = [{"role": role, "dtype": dtype, "shape": shape} for role, dtype, shape in (
            ("input_fp32", "fp32", [rows, width]), ("squared_input", "fp32", [rows, width]),
            ("variance", "fp32", [rows, 1]), ("variance_plus_epsilon", "fp32", [rows, 1]),
            ("inverse_rms", "fp32", [rows, 1]), ("normalized_fp32", "fp32", [rows, width]),
            ("normalized_bf16", "bf16", [rows, width]))]
        flow = {"program": "vllm_0_30_final_rmsnorm_objects", "input": input_port,
            "weight": {"module": "model.norm", "dtype": "bf16", "shape": [width], "read_only": True},
            "epsilon": raw["rms_norm_eps"], "residual": None, "variance_size": None,
            "input_mutated": False, "output": output, "fresh_output": True, "output_aliases_input_or_weight": False,
            "source_branches": [
                {"implementation": "ir_reference", "tensor_temporaries": temporaries,
                 "variance_input_aliases_input_fp32": True, "final_to_bf16_returns_existing_output": True},
                {"implementation": "vllm_c_cuda", "tensor_temporaries": [], "fresh_output_allocation": "torch.empty(x.shape)"},
                {"implementation": "batch_invariant_triton", "tensor_temporaries": [],
                 "input_reshape_allocates": False, "output_reshape_allocates": False,
                 "conditional_copies": [
                     {"condition": "input_2d is not contiguous", "dtype": "bf16", "shape": [rows, width]},
                     {"condition": "weight is not contiguous", "dtype": "bf16", "shape": [width]}]}],
            "consumer": reader["node_id"], "selected_implementation": None,
            "numeric_equivalence_compiler_spills_or_allocator_bound": False,
            "scope": "H200 CUDA source caller objects; no external IR plugins or overrides; source requests, not compiled allocation/peak traffic"}
        record["attributes"].update(source_dataflow=flow, source_tensor_outputs=[output])
        record["attributes"]["source_tensor_inputs"] = [input_port]
        reader["attributes"]["source_tensor_inputs"] = [{"producer": node_id, **output}]
        record["source_alias_dependency_mode"] = "source_completion"
        record["source_alias_account"] = {"declaration": flow,
            "direct_input_producers": [parent], "producer_outputs": [{"node_id": parent,
                **{field: by_id[parent][field] for field in ("rank", "kind", "shape", "dtype")}}],
            "direct_consumers": [reader["node_id"]], "input_tensor_ports": [input_port], "output_tensor_roles": [output],
            "consumer_tensor_inputs": {reader["node_id"]: reader["attributes"]["source_tensor_inputs"]},
            "lowered_calls": [{"kind": "VECTOR", "variant": "rmsnorm", "rank": record["rank"], "dtype": "bf16",
                "shape": {"rows": rows, "hidden": width, "dtype_bytes": 2, "tok": rows}, "attributes": {"epsilon": raw["rms_norm_eps"]},
                "dependencies": [{"source_node_id": parent, "terminal_source_call": True}]}]}


def _attach_fp8_linear_objects(operators, projections):
    """Bind all caller I/O to the independently enumerated FP8 source forms."""
    projection_by_id = {projection["node_id"]: projection for projection in projections}
    by_id = {record["node_id"]: record for record in operators}
    selected = [record for record in operators if record["kind"] == "linear" and record["variant"] == "gemm"
        and projection_by_id[record["node_id"]]["weight_dtype"] == "fp8"]
    selected_ids = {record["node_id"] for record in selected}
    cohorts = {}
    for record in selected:
        identity = (record["node_id"].split(":", 1)[0], record["rank"])
        if identity not in cohorts:
            cohorts[identity] = []
        cohorts[identity].append(record)
    for record in selected:
        rows, columns, outputs = (record["shape"][name] for name in ("m", "k", "n"))
        record["attributes"]["source_tensor_outputs"] = [{"output_role": "output", "dtype": "bf16", "shape": [rows, outputs],
            "strides_elements": [outputs, 1], "offset_elements": 0}]
        record["attributes"]["source_linear_io"] = {"program": "vllm_0_30_block_fp8_linear_io",
            "input_view": {"dtype": "bf16", "shape": [rows, columns], "strides_elements": [columns, 1], "allocates": False},
            "output": {"dtype": "bf16", "shape": [rows, outputs], "strides_elements": [outputs, 1],
                "aliases_input_or_parameters": False}, "bias": None, "mutates_inputs": False,
            "global_fp32_input_copy": False, "return_view_allocates": False,
            "temporary_lifetimes": {"external_FP8_input_and_FP32_scales_when_apply_input_quant": "through matmul return",
                "unused_new_As_when_input_scale_absent": "through wrapper call; never read or initialized"},
            "input_lifetime": "through wrapper read", "output_lifetime": "through consumer read", "selected_runtime_wrapper": None,
            "scope": "source caller objects and conditional wrapper program; actual library scratch, physical allocator and runtime separate"}
        phase = record["node_id"].split(":", 1)[0]
        cohort = cohorts[(phase, record["rank"])]
        values = sorted({item["shape"]["m"] for item in cohort})
        entry = f"{phase}:fp8-sdk-runner:rank-{record['rank']}:entry-H"
        record["attributes"]["source_linear_io"]["conditional_sdk_history"] = {
            "condition": "the current source form uses the installed SDK SM90 cached runner",
            "invocation": phase, "rank": record["rank"],
            "cache_function": "get_fp8_blockscale_gemm_runner_sm90", "cache_arguments": [],
            "instance_scope": "one rank-process source program, shared across layers and K/N",
            "cohort_calls": len(cohort), "cohort_M_values": values,
            "input_port": entry, "output_port": f"{phase}:fp8-sdk-runner:rank-{record['rank']}:exit-H",
            "entry_domain": "nonnegative int64 multiple of 4", "entry_value": None,
            "current_H": ({"operation": "max", "arguments": [entry, (rows + 3) // 4 * 4]}
                          if len(values) == 1 else None),
            "nonuniform_rows_require_source_call_order": len(values) != 1,
            "exit_rule": "max(entry_H, align4(M) for cohort calls using the SDK); entry_H when none use it",
            "updates_all_dtype_runners": ["bf16_bf16", "bf16_fp8", "fp8_fp8"],
            "workspace_H_binding": "flashinfer_workspace_source_program.state.H = current_H",
            "workspace_fresh_per_wrapper_call": True,
            "actual_initial_state_provider_or_allocator_bound": False}
    for consumer in operators:
        parents = [by_id[parent] for parent in consumer["required_ancestors"] if parent in selected_ids]
        if not parents or "source_tensor_inputs" in consumer["attributes"]:
            continue
        if consumer["kind"] == "elementwise" and consumer["variant"] == "add":
            rows, columns = parents[0]["shape"]["m"], parents[0]["shape"]["n"]
            consumer["attributes"]["source_tensor_inputs"] = [{"producer": operand["producer"], "output_role": "output", "dtype": "bf16",
                "shape": [rows, columns], "strides_elements": [columns, 1], "offset_elements": 0}
                for operand in consumer["attributes"]["source_dataflow"]["inputs"]]
        elif consumer["kind"] in ("norm", "activation", "collective"):
            consumer["attributes"]["source_tensor_inputs"] = [{"producer": parent["node_id"], **parent["attributes"]["source_tensor_outputs"][0]}
                for parent in sorted(parents, key=lambda parent: (parent["rank"], parent["node_id"]))]
        else:
            raise ValueError("independent FP8 reference has an unbound consumer tensor role")
    for record in selected:
        parents = sorted(record["required_ancestors"])
        consumers = sorted(item["node_id"] for item in operators if record["node_id"] in item["required_ancestors"])
        projection = projection_by_id[record["node_id"]]
        role_paths = {"fused-wqa-wkv": "attn.fused_wqa_wkv", "wq-b": "attn.wq_b", "wo-b": "attn.wo_b",
            "indexer-wq-b": "attn.indexer.wq_b", "shared-experts:gate-up": "ffn.shared_experts.gate_up_proj",
            "shared-experts:down-proj": "ffn.shared_experts.down_proj"}
        record["source_alias_mode"] = "fp8_linear_objects"
        record["source_alias_account"] = {"declaration": record["attributes"]["source_linear_io"],
            "parameter_module": f"model.layers.{projection['layer']}.{role_paths[projection['role']]}",
            "input_tensor_ports": record["attributes"]["source_tensor_inputs"],
            "output_tensor_roles": record["attributes"]["source_tensor_outputs"],
            "direct_input_producers": parents,
            "producer_outputs": [{"node_id": parent, **{field: by_id[parent][field] for field in ("rank", "kind", "shape", "dtype")}} for parent in parents],
            "direct_consumers": consumers,
            "consumer_tensor_inputs": {consumer: by_id[consumer]["attributes"]["source_tensor_inputs"] for consumer in consumers},
            "external_lowered_producers": [{"source_node_id": parent, "terminal_source_call": True} for parent in parents],
            "lowered_tensor_form_compatible": True,
            "wrapper_source_program": projection_by_id[record["node_id"]]["linear_wrapper_source_program"]}


def _attach_compressor_state_transitions(cases, operators):
    """Logical source updates; never equate unknown old/new physical tables."""
    case_map = {case['phase']: case for case in cases}
    for record in operators:
        if record['variant'] != 'dsv4_compressor_state_cache':
            continue
        case = case_map[record['node_id'].split(':', 1)[0]]
        cursor, updates = 0, []
        for request_id, request in zip(case['request_ids'], case['requests'], strict=True):
            old, count = request['computed'], request['query']
            updates.append({'request_id': request_id, 'input_prefix': old, 'output_prefix': old + count,
                'model_token_rows': [cursor, cursor + count], 'write_positions': [old, old + count]})
            cursor += count
        flow = record['attributes']['source_dataflow']
        record['attributes']['source_cache_transition'] = {
            'program': 'vllm_0_30_compressor_partial_state_lineage', 'cache': {'rank': record['rank'], **flow['state_owner']},
            'input_port': record['node_id'] + ':state-entry', 'output_port': record['node_id'] + ':state-exit',
            'state_dtype': 'fp32', 'compress_ratio': flow['compress_ratio'], 'request_updates': updates,
            'write_semantics': 'source valid-slot stores of KV and score+APE; other logical positions not written by this kernel',
            'metadata_preparation': 'external source cache-manager transition; may append/free/CoW before this model call',
            'physical_table_transition': None, 'prefix_cache_hit_observation': None,
            'actual_copy_completion_and_retained_content_bound': False}


def compressor_state_source_links_reference(operators, cases):
    history, links = {}, []
    for case in cases:
        records = sorted((record for record in operators if record['variant'] == 'dsv4_compressor_state_cache'
                          and record['node_id'].startswith(case['phase'] + ':')), key=lambda record: record['node_id'])
        for record in records:
            transition = record['attributes']['source_cache_transition']
            cache = transition['cache']
            identity = cache['rank'], cache['layer'], cache['component']
            if identity not in history:
                history[identity] = {'exit': None, 'requests': {}}
            prior = history[identity]
            requests = []
            for update in transition['request_updates']:
                request_id = update['request_id']
                if request_id in prior['requests']:
                    prefix = prior['requests'][request_id]
                    if prefix != update['input_prefix']:
                        raise ValueError('independent cache logical prefix has a gap')
                    initial = None
                else:
                    prefix = None
                    initial = {'prefix': [0, update['input_prefix']], 'values': None,
                        'prefix_cache_hit_observation': None, 'admission_copy_transition': None}
                requests.append({'request_id': request_id, 'previous_output_prefix': prefix,
                    'input_prefix': update['input_prefix'], 'output_prefix': update['output_prefix'], 'initial_context': initial})
                prior['requests'][request_id] = update['output_prefix']
            links.append({'invocation': case['phase'], 'source_node_id': record['node_id'], 'cache': cache,
                'input_port': transition['input_port'], 'output_port': transition['output_port'],
                'previous_output_port': prior['exit'], 'requests': requests,
                'between_step_physical_transition': None, 'retained_content_or_copy_completion_proven': False})
            prior['exit'] = transition['output_port']
    return links


def fp8_runner_source_links_reference(operators, cases):
    """Independent source invocation order and rank-process cache identity."""
    links, last = [], {}
    for case in cases:
        phase = case["phase"]
        ranks = sorted({record["rank"] for record in operators
            if record["node_id"].startswith(phase + ":") and "source_alias_mode" in record
            and record["source_alias_mode"] == "fp8_linear_objects"})
        for rank in ranks:
            entry = f"{phase}:fp8-sdk-runner:rank-{rank}:entry-H"
            exit_port = f"{phase}:fp8-sdk-runner:rank-{rank}:exit-H"
            links.append({"invocation": phase, "rank": rank, "input_port": entry, "output_port": exit_port,
                "previous_output_port": last[rank] if rank in last else None,
                "initial_parameter": (None if rank in last else {"name": f"rank-{rank}:initial-sdk-H",
                    "domain": "nonnegative int64 multiple of 4", "value": None})})
            last[rank] = exit_port
    return links


def _attach_fp8_callee_accounts(operators, projections, sources):
    """Close the supplied SDK source form for every legal symbolic entry H.

    No numeric initial state is invented. The external-DeepGEMM form and
    nonuniform-row call order still need their own independent callee proof.
    """
    sdk = sources["flashinfer-input-quant"]
    if sdk is None or "dense_matrix_storage" not in sdk:
        return
    by_id = {record["node_id"]: record for record in projections}
    for record in operators:
        if "source_alias_mode" not in record or record["source_alias_mode"] != "fp8_linear_objects":
            continue
        history = record["attributes"]["source_linear_io"]["conditional_sdk_history"]
        if history["nonuniform_rows_require_source_call_order"]:
            continue
        workspace = by_id[record["node_id"]]["flashinfer_workspace_source_program"]
        record["source_callee_account"] = {
            "source_form": "installed_SDK_SM90_in_call_quantization",
            "caller_source_contract_matched": True, "runner_source_links_matched": True,
            "runner_history": history, "workspace_source_program": workspace,
            "quantizer_then_matrix": ["quantize_bf16_and_write_scales", "packed_fp8_matmul"],
            "workspace_source_lifetime": workspace["source_lifetime"],
            "numeric_initial_H_provider_spills_and_allocator_accepted": False}


def _required_dataflow_obligations(operators):
    """Expose every known operator's unresolved alias and cache obligation.

    A source subprogram agreement is not a proof of the entire operator's
    tensor/physical-state contract. These obligations cannot pass from a
    candidate boolean declaration, missing reference value or DAG reachability.
    """
    obligations = []
    for record in operators:
        if "source_alias_account" in record:
            missing_alias = []
        elif record["variant"] == "dsv4_compressor_state_cache":
            missing_alias = ["state_page_alias_rules_and_reuse_lifetime", "source_metadata_constructor_slot_table_consistency"]
        elif record["variant"] == "dsv4_compressed_state_cache":
            missing_alias = ["state_output_shared_page_alias_and_interference", "remaining_materialization_and_buffer_lifetimes"]
        else:
            missing_alias = ["input_tensor_roles_and_producer_bindings", "output_tensor_roles_and_consumer_bindings",
                "view_base_offsets_shapes_and_strides", "temporary_materialization_and_lifetimes"]
        obligations.append({"subject": record["node_id"], "facet": "tensor_alias_and_temporary_materialization",
            "source": record["source"],
            "expected": "independent complete tensor input/output, view alias and temporary materialization account",
            "missing": missing_alias,
            **({"source_alias_account": record["source_alias_account"]}
               if "source_alias_account" in record else {})})
        if "source_alias_mode" in record and record["source_alias_mode"] == "fp8_linear_objects":
            obligations.append({"subject": record["node_id"], "facet": "fp8_callee_materialization",
                "source": "fp8-linear-input-quant",
                "expected": "conditional callee temporary allocation, source lifetime and access ranges; caller objects alone are insufficient",
                "known_source": {"flashinfer_workspace_dtype": "uint8", "workspace_allocation": "empty, not zero-filled",
                    "workspace_request": "max of BF16/BF16, BF16/FP8 and FP8/FP8 runner getWorkspaceSizeBase",
                    "runner_M_state": "max_shape_m_4_align_ retains maximum previous/current aligned M; expected_m_ initializes on first Base request and is unused by dense gemm",
                    "workspace_extents_and_access_ranges": "conditional affine-H reserve and current-M active views compared in projection flashinfer_workspace_source_program",
                    "dense_matrix_source_tensor_ports": "SM90 legacy and both JIT orientations use the existing five tensor ports with no extra source global tensor; checked in projection program when SDK source is supplied",
                    "workspace_extent_is_traffic": False},
                "missing": ["declared_initial_or_prior_cached_runner_M_state", "matrix_callee_internal_scratch_or_zero_scratch_proof",
                    "DeepGEMM_internal_temporary_source_or_zero_scratch_proof"],
                **({"source_callee_account": record["source_callee_account"]}
                   if "source_callee_account" in record else {})})
        if record["kind"] in ("kv_update", "eager_attn"):
            if record["variant"] in ("dsv4_compressor_state_cache", "dsv4_compressed_state_cache"):
                missing_cache = ["source_metadata_constructor_slot_table_consistency", "state_output_page_alias_and_overwrite_order",
                    "output_slot_store_ranges_and_cache_manager_reuse"]
            else:
                missing_cache = ["source_page_slot_address_rules_and_input_domains", "cache_read_write_ranges",
                    "state_continuity_from_previous_phase_or_declared_initial_state"]
            obligations.append({"subject": record["node_id"], "facet": "persistent_cache_and_cross_step_state",
                "source": record["source"],
                "expected": "independent cache object, source page/slot address rules and valid domains, read/write ranges and cross-step continuity account",
                "missing": missing_cache})
    return obligations


def build_flash_reference(config_path, source_root, *, prefill_tokens, torch_include=None, flashinfer_source_root=None,
                          embedding_token_ids=None, v4_source_max_model_len=None, source_cases=None,
                          v4_source_use_deep_gemm=None):
    """Derive every dense projection in all 43 layers and the vocabulary head.

    The independent inventory is not yet the complete model alignment gate:
    the remaining required operator families/facets are retained explicitly.
    """
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if v4_source_use_deep_gemm is not None and type(v4_source_use_deep_gemm) is not bool:
        raise TypeError("independent V4 DeepGEMM env declaration must be boolean or None")
    identity = _official_config_identity(raw)
    if v4_source_max_model_len is not None:
        if (type(v4_source_max_model_len) is not int or v4_source_max_model_len <= 0
                or v4_source_max_model_len > raw["max_position_embeddings"]):
            raise ValueError("independent source max_model_len must fit official context extent")
    if (raw["model_type"] != "deepseek_v4" or raw["num_hidden_layers"] != 43
            or raw["hidden_size"] != 4096 or raw["torch_dtype"] != "bfloat16"):
        raise ValueError("reference requires the exact Flash BF16 activation geometry")
    quant = raw["quantization_config"]
    if (quant["quant_method"] != "fp8" or quant["weight_block_size"] != [128, 128]
            or quant["scale_fmt"] != "ue8m0" or quant["activation_scheme"] != "dynamic"):
        raise ValueError("Flash reference requires dynamic block128 FP8 ue8m0")
    anchors = _source_anchors(Path(source_root), torch_include, flashinfer_source_root)
    records = []
    cases = (flash_cases(prefill_tokens) if source_cases is None else validate_flash_source_cases(
        source_cases, raw["max_position_embeddings"] if v4_source_max_model_len is None else v4_source_max_model_len))
    model_cases, model_orders = flash_model_input_cases(cases)
    token_ids = (None if embedding_token_ids is None else
                 validate_flash_token_ids(embedding_token_ids, cases, raw["vocab_size"]))
    model_token_ids = None if token_ids is None else {
        phase: [token_ids[phase][row] for row in order["token_row_permutation"]]
        for phase, order in model_orders.items()}
    h, d, q = raw["hidden_size"], raw["head_dim"], raw["q_lora_rank"]
    tp = 4
    heads, groups = raw["num_attention_heads"], raw["o_groups"]
    # Shared MLP block-aligned ColumnParallelLinear rounds full block counts
    # before TP split (DeepseekV4MLP.__init__), not arbitrary element chunks.
    shared_i = ((raw["moe_intermediate_size"] * raw["n_shared_experts"] // 128 + tp - 1) // tp) * 128

    def projection(case, layer, role, b, m, n, k, weight, output="bf16"):
        for rank in range(tp):
            rank_explicit = role == "router" or role.startswith("shared-experts:")
            prefix = case["phase"] + ":" + ("" if layer is None else f"layer-{layer}:")
            suffix = f":tp-rank-{rank}" if rank or rank_explicit else ""
            anchor = "compressor-kv-score" if role == "indexer-compressor-kv-score" else role
            shape = {"b": b, "m": m, "n": n, "k": k}
            weight_payload = b * n * k * (2 if weight == "bf16" else 1)
            output_bytes = b * m * n * (4 if output == "fp32" else 2)
            records.append({
                "node_id": prefix + role + suffix, "rank": rank, "layer": layer,
                "role": role, "source": anchor, "shape": shape,
                "input_dtype": "bf16", "weight_dtype": weight, "output_dtype": output,
                "weight_format": "unquantized" if weight == "bf16" else "e4m3_block128x128_ue8m0",
                "mathematical_flops": 2 * b * m * n * k,
                "operand_payload_bytes": {"matmul_input": b * m * k * (2 if weight == "bf16" else 1),
                                          "weight_payload": weight_payload, "output": output_bytes},
                "applicability": "TP4 PP1 EP-off MTP-off non-MegaMoE pure text; explicit source branch",
                "byte_scope": "typed matmul payloads (FP8 A/B for quantized matmul), excludes scale buffers/quantization producers/reloads; not HBM traffic",
            })
            if weight == "fp8":
                records[-1]["linear_wrapper_source_program"] = (
                    None if role == "wo-a" else fp8_linear_wrapper_reference(b * m, k, n))
            if weight == "fp8" and role != "wo-a":
                records[-1]["flashinfer_workspace_source_program"] = flashinfer_workspace_reference(b * m, k, n)
                elements, scale_bytes = b * m * k, 4 * b * m * ((k + 127) // 128)
                records[-1]["input_quantizer_operands"] = {
                    "input_dtype": "bf16", "quantized_dtype": "fp8", "scale_dtype": "fp32", "group_size": 128,
                    "bf16_input_read_bytes": 2 * elements, "fp8_output_write_bytes": elements,
                    "scale_output_write_bytes": scale_bytes}
                records[-1]["input_quantizer_source_program"] = flashinfer_quantizer_source_program(b * m, k)

    for case in cases:
        tokens = sum(request["query"] for request in case["requests"])
        for layer in range(raw["num_hidden_layers"]):
            ratio = max(1, raw["compress_ratios"][layer])
            projection(case, layer, "fused-wqa-wkv", 1, tokens, q + d, h, "fp8")
            projection(case, layer, "wq-b", 1, tokens, heads // tp * d, q, "fp8")
            projection(case, layer, "wo-a", groups // tp, tokens, raw["o_lora_rank"], heads * d // groups, "fp8")
            projection(case, layer, "wo-b", 1, tokens, h, groups // tp * raw["o_lora_rank"], "fp8")
            projection(case, layer, "router", 1, tokens, raw["n_routed_experts"], h, "bf16", "fp32")
            projection(case, layer, "shared-experts:gate-up", 1, tokens, 2 * shared_i, h, "fp8")
            projection(case, layer, "shared-experts:down-proj", 1, tokens, h, shared_i, "fp8")
            if ratio > 1:
                projection(case, layer, "compressor-kv-score", 1, tokens, 2 * (1 + (ratio == 4)) * d, h, "bf16", "fp32")
            if ratio == 4:
                projection(case, layer, "indexer-weights-proj", 1, tokens, raw["index_n_heads"], h, "bf16")
                projection(case, layer, "indexer-compressor-kv-score", 1, tokens, 4 * raw["index_head_dim"], h, "bf16", "fp32")
                # Source skips indexer Q projection entirely when every
                # sequence fits topk: compressed_seq_len <= index_topk.
                if any((request["computed"] + request["query"]) // 4 > raw["index_topk"] for request in case["requests"]):
                    projection(case, layer, "indexer-wq-b", 1, tokens, raw["index_n_heads"] * raw["index_head_dim"], q, "fp8")
        padded_vocab = ((raw["vocab_size"] + 63) // 64) * 64
        if padded_vocab % tp:
            raise ValueError("vocabulary shard is not divisible by TP4")
        projection(case, None, "lm-head", 1, len(case["requests"]), padded_vocab // tp, h, "bf16")
    operators = _flash_operators(raw, model_cases, records, v4_source_max_model_len=v4_source_max_model_len)
    for record in operators:
        if record["kind"] == "fused_moe":
            record["source_work_unchecked_fields"] = (
                ["weight_reader_source_program"] if anchors["marlin-moe-source"]["weight_operand_layout"] is None else [])
    _attach_simple_accounts(raw, model_cases, operators, records)
    attach_fp8_stage_forms(operators, records)
    attach_embedding_accounts(raw, model_cases, operators, model_token_ids)
    _attach_embedding_tensor_dataflow(raw, operators, model_token_ids)
    attach_mhc_accounts(raw, model_cases, operators, Path(source_root), source_use_deep_gemm=v4_source_use_deep_gemm)
    _attach_shared_add_dataflow(raw, model_cases, operators)
    _attach_greedy_tensor_dataflow(raw, model_cases, operators)
    _attach_qkv_norm_dataflow(raw, operators)
    _attach_attention_input_projection_dataflow(raw, operators)
    _attach_shared_activation_dataflow(raw, operators)
    _attach_gate_projection_dataflow(raw, operators)
    _attach_indexer_weights_dataflow(raw, model_cases, operators)
    _attach_o_projection_dataflow(raw, model_cases, operators)
    _attach_mhc_state_bindings(raw, model_cases, operators)
    _attach_mhc_wrapper_alias_accounts(raw, model_cases, operators)
    _attach_fp8_linear_objects(operators, records)
    _attach_lm_head_objects(raw, operators, anchors)
    _attach_final_norm_objects(raw, operators)
    _attach_fp8_callee_accounts(operators, records, anchors)
    _attach_short_topk_dataflow(raw, model_cases, operators)
    _attach_route_dataflow(raw, operators, model_token_ids)
    _attach_marlin_caller_objects(operators, records, anchors)
    _attach_swa_insert_dataflow(raw, model_cases, operators)
    attach_attention_output_regions(raw, model_cases, operators)
    _attach_partial_state_ports(raw, operators)
    _attach_partial_state_caller_frames(operators, anchors)
    _attach_finish_caller_objects(raw, model_cases, operators, anchors)
    _attach_compressor_state_transitions(model_cases, operators)
    for record in operators:
        phase = record["node_id"].split(":", 1)[0]
        if (record["variant"] in ("embedding", "vllm_logits_row_select")
                or ":target-sampler-output-int32" in record["node_id"]):
            record["attributes"]["source_model_input_order"] = model_orders[phase]
            if record["variant"] == "embedding" and model_token_ids is not None:
                record["attributes"]["embedding_token_ids"] = model_token_ids[phase]
    initial_prefixes = {}
    for case in cases:
        for request_id, request in zip(case["request_ids"], case["requests"], strict=True):
            if request_id not in initial_prefixes:
                initial_prefixes[request_id] = request["computed"]
    return {"scope": "DeepSeek-V4 Flash independent static alignment; graph and projection checkpoint",
            "config_path": str(config_path), "source_root": str(source_root), "sources": anchors,
            "config_identity": identity,
            "candidate_requirements": {
                "model_id": identity["identity_data"]["model_id"],
                "model_revision": identity["identity_data"]["revision"],
                "hardware_id": "nvidia_h200", "mtp": "off", "model_execution_version": "vllm_0_30_0",
                "target_sampling_source": "vllm_0_30_plain_greedy_bf16",
                "v4_routing_source": "vllm_0_30_modular_cuda_top6_pure_text",
                "v4_eager_metadata_source": "vllm_0_30_eager_dict_all_valid",
                "v4_source_max_model_len": v4_source_max_model_len, "source_phase_requests_supplied": source_cases is not None,
                **({"v4_source_use_deep_gemm": v4_source_use_deep_gemm} if v4_source_use_deep_gemm is not None else {})},
            "parallel_requirements": {"tensor_parallel_size": 4, "pipeline_parallel_size": 1,
                "enable_expert_parallel": False, "sequence_parallel_size": 1, "context_parallel_size": 1, "data_parallel_size": 1},
            "execution": {"hardware": "nvidia_h200", "tp": tp, "pp": 1, "ep": False, "mtp": False},
            "cases": cases, "initial_request_prefixes": initial_prefixes,
            "model_input_cases": model_cases, "model_input_orders": model_orders,
            "embedding_inputs": token_ids, "projections": records, "operators": operators,
            "required_dataflow_obligations": _required_dataflow_obligations(operators),
            "fp8_runner_source_links": fp8_runner_source_links_reference(operators, model_cases),
            "compressor_state_source_links": compressor_state_source_links_reference(operators, model_cases),
            "remaining_required_facets": [
                "nonprojection_math_comparisons_and_bytes",
                "FP8_scales_and_activation_quantization_storage",
                "C4_C128_long_context_and_fusion_branch_matrix",
            ],
            "limitations": ["explicit routing assumptions are not observed expert assignments",
                            "matched conditional FP8 callee recipes do not bind the target backend selection or every provider alternative; the full FP8 quantization/branch facet remains open",
                            "predicate/index instruction costs are outside the argmax FLOPs/bytes baseline",
                            "GPU launch/provider/runtime, performance accuracy, Serving capacity and MTP not accepted"]}


def _flash_operators(raw, cases, projections, *, v4_source_max_model_len=None):
    """Independent logical graph from decoder/attention/FFN call structure.

    Names adapt semantic roles to the candidate's public IDs; enumeration,
    branches and dimensions depend exclusively on source/config/scenario.
    Required predecessor reachability is weaker than tensor alias alignment,
    which remains a separate unfinished facet.
    """
    tp, h, d, hc = 4, raw["hidden_size"], raw["head_dim"], raw["hc_mult"]
    heads = raw["num_attention_heads"] // tp
    padded = 64 if heads <= 64 else 128
    index_heads, index_d = raw["index_n_heads"], raw["index_head_dim"]
    topk, window = raw["index_topk"], raw["sliding_window"]
    experts, selected = raw["n_routed_experts"], raw["num_experts_per_tok"]
    i = raw["moe_intermediate_size"] // tp
    inventory = []

    def identity(phase, layer, role, rank):
        explicit = role in ("router", "route-selection-weights", "routed-experts", "shared-add") or role.startswith("shared-experts:")
        return phase + ":" + ("" if layer is None else f"layer-{layer}:") + role + (
            f":tp-rank-{rank}" if rank or explicit else "")

    for case in cases:
        phase = case["phase"]
        requests = case["requests"]
        c128_width = c128_metadata_width_reference(
            max(request["computed"] + request["query"] for request in requests), v4_source_max_model_len)
        short_context = max(request["computed"] + request["query"] for request in requests) // 4 <= topk
        tokens = sum(request["query"] for request in requests)
        samples = len(requests)

        def add(layer, role, rank, kind, variant, shape, dtype, source, predecessors, attributes=None):
            record = {"node_id": identity(phase, layer, role, rank), "rank": rank,
                      "kind": kind, "variant": variant, "shape": shape, "dtype": dtype,
                      "source": source, "required_ancestors": sorted(predecessors),
                      "attributes": {} if attributes is None else attributes}
            inventory.append(record)
            return record["node_id"]

        def collective(layer, role, operation, inputs, byte_count):
            source = ("logits" if operation == "all_gather" else "embedding-forward" if layer is None
                      else "attention-forward" if role.startswith("wo-b:") else "ffn-forward")
            return add(layer, role, 0, "collective", None, {}, None, source,
                       inputs, {"operation": operation, "participants": list(range(tp)), "bytes": byte_count,
                                "bytes_semantics": "per_rank_input"})

        roots = [add(None, "token-embedding", rank, "gather", "embedding",
                     {"elements": tokens * h, "rows": tokens, "hidden": h}, "bf16", "embedding-forward", [])
                 for rank in range(tp)]
        previous = collective(None, "token-embedding:tp-all-reduce", "all_reduce", roots, 2 * tokens * h)
        mhc_shape = {"elements": tokens * hc * h, "hidden": h, "hc_mult": hc, "dtype_bytes": 2}
        for layer in range(raw["num_hidden_layers"]):
            ratio = max(1, raw["compress_ratios"][layer])
            layer_records = {record["node_id"]: record for record in projections if record["layer"] == layer
                             and record["node_id"].startswith(phase + ":")}

            def project(role, rank, predecessors):
                record = layer_records[identity(phase, layer, role, rank)]
                return add(layer, role, rank, "router" if role == "router" else "linear",
                           "batched_gemm" if role == "wo-a" else "gemm", record["shape"], "bf16",
                           record["source"], predecessors)

            wo_outputs, ffn_inputs = [], []
            layer_state = []
            for rank in range(tp):
                pre = add(layer, "mhc-pre" if layer == 0 else "mhc-post-pre-attn", rank, "vector",
                          "mhc_pre" if layer == 0 else "mhc_post_pre", mhc_shape, "bf16", "decoder-forward", [previous])
                fused = project("fused-wqa-wkv", rank, [pre])
                aux = []
                if ratio > 1:
                    aux.append(project("compressor-kv-score", rank, [pre]))
                if ratio == 4:
                    aux += [project("indexer-weights-proj", rank, [pre]),
                            project("indexer-compressor-kv-score", rank, [pre])]
                norm = add(layer, "q-kv-rmsnorm", rank, "norm", "dsv4_q_kv_rmsnorm",
                           {"rows": tokens, "q_hidden": raw["q_lora_rank"], "kv_hidden": d}, "bf16", "attention-forward", [fused])
                wq = project("wq-b", rank, [norm])
                index_q = None
                if ratio == 4 and not short_context:
                    index_projection = project("indexer-wq-b", rank, [norm])
                    index_q = add(layer, "indexer-q-rope-quant", rank, "vector", "dsv4_indexer_q_rope_quant",
                                  {"elements": tokens * index_heads * index_d, "seq_len": tokens,
                                   "num_query_heads": index_heads, "head_dim": index_d, "rope_dim": raw["qk_rope_head_dim"]},
                                  "bf16", "indexer-q-quant", [index_projection, aux[1]])
                    # Both CUDA source programs fold scale into weights,
                    # but their arithmetic and fold association differ.
                    groups = tokens * index_heads
                    inventory[-1]["source_work"] = {
                        **_indexer_q_backend_reference(tokens, index_heads, index_d, raw["qk_rope_head_dim"]),
                        "typed_payload_bytes": {"projected_q": 2 * groups * index_d,
                            "indexer_weights": 2 * groups, "positions": 8 * tokens,
                            "rotary_cache": 4 * tokens * raw["qk_rope_head_dim"],
                            "quantized_q": groups * index_d, "folded_weights": 4 * groups},
                        "q_scale_has_separate_buffer": False}
                    inventory[-1]["roofline_baseline"] = _indexer_q_roofline_reference(tokens)
                swa = add(layer, "swa-kv-update", rank, "kv_update", "dsv4_qnorm_rope_kv_insert",
                          {"elements": tokens * (heads + 1) * d, "seq_len": tokens, "num_query_heads": heads,
                           "padded_query_heads": padded, "head_dim": d, "rope_dim": raw["qk_rope_head_dim"],
                           "cache_token_bytes": 584}, "bf16", "attention-forward", [wq, norm])
                attention_inputs = [swa]
                if ratio > 1:
                    state_bytes = 2 * (1 + (ratio == 4)) * d * 4
                    state = add(layer, "compressor-state-update", rank, "kv_update", "dsv4_compressor_state_cache",
                                {"elements": tokens * state_bytes, "seq_len": tokens, "state_bytes": state_bytes, "dtype_bytes": 4},
                                "fp32", "compressor-forward", [aux[0]],
                                {"source_dataflow": _partial_state_view_reference(tokens, layer, "attention.compressor", d, ratio)})
                    completed = sum((request["computed"] + request["query"]) // ratio - request["computed"] // ratio for request in requests)
                    if completed or ratio == 4:
                        state = add(layer, "compressor-kv-update", rank, "kv_update", "dsv4_compressed_state_cache",
                                    {"elements": completed * 584, "seq_len": tokens, "compress_ratio": ratio, "state_bytes": 584, "dtype_bytes": 1},
                                    "uint8", "compressor-forward", [state],
                                    {"state_old_lens": [request["computed"] for request in requests],
                                     "state_new_lens": [request["computed"] + request["query"] for request in requests],
                                     "partial_state_owner": {"layer": layer, "component": "attention.compressor", "scope": "local TP rank module instance"},
                                     "partial_state_windows": _partial_state_window_reference(requests, layer, "attention.compressor", d, ratio)})
                    attention_inputs.append(state)
                if ratio == 4:
                    index_state_bytes = 2 * 2 * index_d * 4
                    idx_state = add(layer, "indexer-compressor-state-update", rank, "kv_update", "dsv4_compressor_state_cache",
                                    {"elements": tokens * index_state_bytes, "seq_len": tokens, "state_bytes": index_state_bytes, "dtype_bytes": 4},
                                    "fp32", "compressor-forward", [aux[2]],
                                    {"source_dataflow": _partial_state_view_reference(tokens, layer, "attention.indexer.compressor", index_d, 4)})
                    # Pure-text default FP8 index cache: 128 data + one FP32 scale.
                    idx_cache = add(layer, "indexer-k-cache-update", rank, "kv_update", "dsv4_compressed_state_cache",
                                    {"elements": completed * (index_d + 4), "seq_len": tokens, "compress_ratio": 4,
                                     "state_bytes": index_d + 4, "dtype_bytes": 1}, "uint8", "compressor-forward", [idx_state],
                                    {"state_old_lens": [request["computed"] for request in requests],
                                     "state_new_lens": [request["computed"] + request["query"] for request in requests],
                                     "partial_state_owner": {"layer": layer, "component": "attention.indexer.compressor", "scope": "local TP rank module instance"},
                                     "partial_state_windows": _partial_state_window_reference(requests, layer, "attention.indexer.compressor", index_d, 4)})
                    if short_context:
                        attention_inputs.append(add(layer, "indexer-short-context-topk", rank, "vector", "dsv4_short_context_topk",
                            {"elements": tokens * topk, "seq_len": tokens, "topk_capacity": topk, "compress_ratio": 4, "dtype_bytes": 4},
                            "int32", "indexer-forward", [idx_cache]))
                attention_outputs = []
                indexer_records, attention_records = [], []
                for mode in ("prefill", "decode"):
                    subset = [request for request, request_phase in zip(requests, case["attention_phases"], strict=True) if request_phase == mode]
                    if not subset:
                        continue
                    # The declared short scenarios contain at most one request per mode.
                    lens = [request["computed"] + request["query"] for request in subset]
                    counts = [length // ratio if ratio > 1 else 0 for length in lens]
                    selected_counts = [min(topk, count) if ratio == 4 else count for count in counts]
                    query_count = sum(request["query"] for request in subset)
                    phase_inputs = list(attention_inputs)
                    if ratio == 4 and not short_context:
                        indexer = add(layer, "indexer-score-topk-" + mode, rank, "eager_attn", "dsv4_indexer_" + mode,
                                      {"batch": len(subset), "q_len": max(request["query"] for request in subset),
                                       "kv_len": max(lens), "num_query_heads": index_heads, "head_dim": index_d,
                                       "topk": min(topk, max(counts)), "compress_ratio": 4, "dtype_bytes": 4},
                                      "int32", "indexer-causal-ends", [index_q, idx_cache, *attention_inputs])
                        pairs = sum((request["computed"] + offset + 1) // 4
                                    for request in subset for offset in range(request["query"]))
                        inventory[-1]["source_work"] = {"active_qk_tensor_flops": 2 * pairs * index_heads * index_d}
                        inventory[-1]["indexer_baseline"] = indexer_baseline_reference(raw, subset, mode)
                        indexer_records.append(inventory[-1])
                        phase_inputs.append(indexer)
                    attention_outputs.append(add(layer, mode + "-sparse-attention", rank, "eager_attn", "dsv4_sparse_" + mode,
                        {"batch": len(subset), "q_len": max(request["query"] for request in subset), "kv_len": max(lens), "num_query_heads": padded,
                         "qk_head_dim": d, "v_head_dim": d, "topk": max(selected_counts), "window_size": window,
                         "compress_ratio": ratio, "compressed_state_bytes": 584,
                         "swa_token_bytes": d + raw["qk_rope_head_dim"] + d // 64, "dtype_bytes": 2},
                        "bf16", "attention-forward", phase_inputs))
                    attention_records.append(inventory[-1])
                    if ratio == 128:
                        inventory[-1]["attributes"]["topk_capacity"] = c128_width
                    if mode == "prefill":
                        inventory[-1]["source_work"] = _prefill_gather_reference(raw, subset, ratio, c128_width=c128_width)
                        inventory[-1]["source_work_source"] = "prefill-gather"
                    else:
                        inventory[-1]["decode_attention_index_inputs"] = _decode_attention_index_inputs_reference(
                            raw, subset, ratio, c128_width if ratio == 128 else topk)
                        if ratio == 4:
                            inventory[-1]["decode_index_materialization"] = _decode_c4_indices_reference(raw, subset, all_completed_states=short_context)
                # One indexer_op finishes prefill and decode before the
                # subsequent forward_mqa call. Both wrappers order prefill
                # before decode. These are source-call barriers, independent
                # of an observed compiler or selected GPU provider.
                for previous_indexer, current_indexer in zip(indexer_records, indexer_records[1:]):
                    current_indexer["required_ancestors"].append(previous_indexer["node_id"])
                if indexer_records:
                    for attention_record in attention_records:
                        joined = indexer_records[-1]["node_id"]
                        if joined not in attention_record["required_ancestors"]:
                            attention_record["required_ancestors"].append(joined)
                for previous_attention, current_attention in zip(attention_records, attention_records[1:]):
                    current_attention["required_ancestors"].append(previous_attention["node_id"])
                for record in indexer_records + attention_records:
                    record["dependency_source"] = "indexer-call-barriers"
                quant = add(layer, "wo-a-inv-rope-quant", rank, "vector", "dsv4_inv_rope_fp8_quant",
                            {"elements": tokens * heads * d, "seq_len": tokens, "num_query_heads": heads,
                             "head_dim": d, "rope_dim": raw["qk_rope_head_dim"]}, "fp8", "attention-forward", attention_outputs)
                wo_a = project("wo-a", rank, [quant])
                wo_outputs.append(project("wo-b", rank, [wo_a]))
                layer_state.append(pre)
            attention_reduce = collective(layer, "wo-b:tp-all-reduce", "all_reduce", wo_outputs, 2 * tokens * h)
            for rank in range(tp):
                ffn = add(layer, "mhc-post-pre-ffn", rank, "vector", "mhc_post_pre", mhc_shape,
                          "bf16", "decoder-forward", [attention_reduce, layer_state[rank]])
                gate = project("router", rank, [ffn])
                route = add(layer, "route-selection-weights", rank, "vector", "dsv4_moe_route",
                            {"elements": tokens * experts, "rows": tokens, "experts": experts, "topk": selected},
                            "fp32", "ffn-forward", [gate],
                            {"expert_id_selection": "hash_table" if layer < raw["num_hash_layers"] else "score_topk",
                             "routed_scaling_factor": raw["routed_scaling_factor"]})
                routed = add(layer, "routed-experts", rank, "fused_moe", "full",
                             {"tokens": tokens, "experts": experts, "topk": selected, "hidden": h, "intermediate": i},
                             "bf16", "ffn-forward", [route, ffn],
                             {"swiglu_limit": raw["swiglu_limit"], "routed_weight_dtype": "mxfp4", "output_is_reduced": False})
                inventory[-1]["source_work"] = _marlin_moe_reference(raw, tokens, tp)
                # This reference declares H200 SXM (132 SM), as specified in
                # configs/hardware/nvidia_h200.yaml; no device is observed.
                inventory[-1]["scratch_allocation"] = _marlin_scratch_reference(inventory[-1]["source_work"], 132)
                inventory[-1]["workspace_zero_bytes"] = 132 * 16
                inventory[-1]["source_work_source"] = "marlin-moe-source"
                up = project("shared-experts:gate-up", rank, [ffn])
                activation = add(layer, "shared-experts:silu-mul", rank, "activation", "silu",
                                 {"elements": tokens * i, "tok": tokens, "intermediate": i}, "bf16", "shared-forward", [up],
                                 {"swiglu_limit": raw["swiglu_limit"]})
                down = project("shared-experts:down-proj", rank, [activation])
                ffn_inputs.append(add(layer, "shared-add", rank, "elementwise", "add", {"elements": tokens * h},
                                      "bf16", "ffn-forward", [routed, down]))
            previous = collective(layer, "moe-output:tp-all-reduce", "all_reduce", ffn_inputs, 2 * tokens * h)
        head_inputs = []
        for rank in range(tp):
            post = add(None, "mhc-post", rank, "vector", "mhc_post", mhc_shape, "bf16", "model-forward", [previous])
            hc_head = add(None, "hc-head", rank, "vector", "hc_head", mhc_shape, "bf16", "model-forward", [post])
            norm = add(None, "final-norm", rank, "norm", "rmsnorm", {"rows": tokens, "hidden": h}, "bf16", "model-forward", [hc_head])
            row_select = add(None, "logits-boundary-select", rank, "vector", "vllm_logits_row_select",
                             {"elements": samples * h, "rows": samples, "hidden": h, "source_rows": tokens}, "bf16", "row-selection", [norm])
            head_inputs.append(add(None, "lm-head", rank, "lm_head", "gemv" if samples == 1 else "gemm",
                                   {"b": 1, "m": samples, "n": raw["vocab_size"] // tp, "k": h}, "bf16", "lm-head", [row_select]))
        gather = collective(None, "lm-head:tp-all-gather", "all_gather", head_inputs, 2 * samples * raw["vocab_size"] // tp)
        for rank in range(tp):
            logits = gather
            if samples > 1 and raw["vocab_size"] // tp > 1:
                logits = add(None, "lm-head:tp-gather-reorder", rank, "vector", "vllm_logits_gather_reorder",
                    {"elements": samples * raw["vocab_size"], "rows": samples,
                     "vocab": raw["vocab_size"], "tensor_parallel_size": tp},
                    "bf16", "tp-gather-layout-cuda", [gather], {"model_execution_version": "vllm_0_30_0"})
            cast = add(None, "target-sampler-logits-fp32", rank, "vector", "vllm_logits_fp32_cast",
                       {"elements": samples * raw["vocab_size"], "rows": samples, "vocab": raw["vocab_size"]}, "bf16", "sampler-forward", [logits])
            argmax = add(None, "target-sampler-argmax", rank, "vector", "vllm_plain_greedy_sample",
                         {"elements": samples * raw["vocab_size"], "rows": samples, "vocab": raw["vocab_size"]}, "fp32", "greedy", [cast])
            add(None, "target-sampler-output-int32", rank, "vector", "vllm_plain_greedy_sample",
                {"elements": samples, "rows": samples, "vocab": raw["vocab_size"]}, "int64", "sampler-forward", [argmax])
    return inventory


def finish_metadata_reference(tokens, states, rows, dimension, ratio, stage):
    """Independent typed metadata operands, retaining the declared byte scope."""
    operands = []

    def add(part, name, dtype, elements, condition):
        operands.append({"stage": part, "role": name, "dtype": dtype, "elements": elements,
            "bytes": elements * {"int32": 4, "int64": 8}[dtype], "condition": condition})

    if stage != "store":
        add("compress", "state_slots", "int64", tokens, "all valid scheduled tokens")
        add("compress", "positions", "int64", tokens, "all valid scheduled tokens")
        if ratio == 4 and dimension == 512:
            add("compress", "destination_slots", "int64", tokens, "before boundary guard")
        add("compress", "request_indices", "int32", tokens if ratio == 128 else states,
            "before boundary guard" if ratio == 128 else "completed boundaries only")
        add("compress", "state_block_table", "int32", rows, "existing valid-window row/split operand projection")
    if stage != "compress":
        if ratio == 128:
            add("store", "state_slots", "int64", tokens, "all valid scheduled tokens")
            add("store", "positions", "int64", tokens, "all valid scheduled tokens")
            add("store", "destination_slots", "int64", tokens, "before boundary guard")
        elif dimension == 128:
            add("store", "destination_slots", "int64", states, "completed boundaries only")
    return {"roles": operands, "baseline_read_bytes": sum(operand["bytes"] for operand in operands),
        "scope": "all-valid baseline metadata operand projection; source load replication, physical pages and observed traffic separate"}


def finish_source_reference(raw, requests, *, ratio, indexer, stage):
    """Independent active-value ledger for the three declared finish programs.

    The CuTe main kernels and Triton indexer differ in softmax, scale rounding
    and quantization order. Expected counts use only source geometry and the
    externally declared requests, not KernelSpec or a production descriptor.
    """
    dim = raw["index_head_dim"] if indexer else raw["head_dim"]
    boundaries = [list(range((request["computed"] // ratio + 1) * ratio,
                             request["computed"] + request["query"] + 1, ratio)) for request in requests]
    n = sum(len(ends) for ends in boundaries)
    window = 8 if ratio == 4 else 128
    r = sum(min(end, window) for ends in boundaries for end in ends)
    v, e = r * dim, n * dim
    count = {name: {} for name in ("compute_fma", "compute_alu", "compute_reduction", "compute_sfu", "compute_tensor")}
    multiply_add, transcendental = (count[name] for name in ("compute_fma", "compute_sfu"))
    typed_actions = {}

    def unpriced(name, operation, elements):
        typed_actions[name] = {"dtype": "fp32", "operation": operation, "elements": elements}

    if stage != "store":
        if ratio == 128:
            unpriced("local_max", "maximum", v)
            unpriced("eight_lane_merge_max", "maximum", e * 8 * 3)
            multiply_add.update({"score_subtract": v, "score_log2e_multiply": v,
                "merge_max_subtract": n * dim * 8, "merge_log2e_multiply": n * dim * 8,
                "compressed_value_divide": e})
            multiply_add.update({"local_exp_sum_add": v, "weighted_value_multiply_add": 2 * v,
                "merge_sum_product_rescale_multiply": n * dim * 8 * 2})
            multiply_add.update({"eight_lane_merge_sum": e * 8 * 3,
                "eight_lane_merge_product": e * 8 * 3})
            transcendental.update({"local_score_exp2": v, "merge_rescale_exp2": n * dim * 8})
        elif indexer:
            multiply_add.update({"softmax_score_subtract": v, "softmax_normalize_divide": v})
            multiply_add["weighted_value_multiply"] = v
            unpriced("softmax_max", "maximum", (r - n) * dim)
            multiply_add.update({"softmax_sum": (r - n) * dim,
                "weighted_value_sum": (r - n) * dim})
            transcendental["softmax_exp"] = v
        else:
            unpriced("online_max", "maximum", v)
            multiply_add.update({"old_new_score_subtract": v * 2,
                "old_new_log2e_multiply": v * 2, "compressed_value_divide": e})
            multiply_add.update({"online_sum_multiply_add": v * 2, "online_product_rescale_multiply": v,
                "weighted_value_multiply_add": v * 2})
            transcendental["online_old_new_exp2"] = v * 2
    if stage != "compress":
        q, b = (n * raw["index_head_dim"], n) if indexer else (n * (raw["head_dim"] - raw["qk_rope_head_dim"]), n * 7)
        rope = n * raw["qk_rope_head_dim"]
        multiply_add.update({"norm_square": e, "norm_rrms_multiply": e, "norm_gamma_multiply": e,
            "rope_multiply": 2 * rope, "rope_add_subtract": rope, "quant_value_scale_multiply": q})
        multiply_add.update({"norm_sum": n * (dim - 1), "norm_variance_divide": n, "norm_eps_add": n})
        unpriced("quant_absmax", "maximum", q - b)
        unpriced("quant_abs", "absolute", q)
        unpriced("quant_lower_clamp", "maximum", q)
        unpriced("quant_upper_clamp", "minimum", q)
        unpriced("quant_scale_minimum_max", "maximum", b)
        transcendental["norm_rsqrt"] = n
        if indexer:
            multiply_add["quant_raw_scale_multiply"] = b
            unpriced("quant_exponent_ceil", "ceil", b)
            unpriced("quant_inverse_exponent_negate", "negate", b)
            transcendental.update({"quant_scale_log2": b, "quant_inverse_scale_exp2": b, "quant_stored_scale_exp2": b})
        else:
            multiply_add["quant_raw_scale_divide"] = b
    temporary = None
    if ratio == 128:
        touched, offset = [], 0
        for request, ends in zip(requests, boundaries, strict=True):
            touched.extend(offset + boundary - request["computed"] - 1 for boundary in ends)
            offset += request["query"]
        temporary = {"role": "compressed_kv", "dtype": "fp32", "shape": [offset, dim],
            "source_extent_bytes": offset * dim * 4, "initialized_by_allocation": False,
            "written_token_rows": touched, "read_token_rows": touched, "producer_stage": "compress", "consumer_stage": "store",
            "source_alias": "same compressed_kv tensor argument", "actual_allocator_binding_verified": False}
    repeated = None
    if not indexer and stage != "compress":
        threads = 256 if ratio == 128 else dim // 8
        scale_lanes = 32 if ratio == 128 else 8
        norm_adds = threads * (1 + 5 + 3) if ratio == 128 else dim + threads * 5 + threads // 32
        repeated = {"norm_rsqrt_expressions": n * (threads if ratio == 128 else 1),
            "norm_sum_add_expressions": n * norm_adds, "quant_scale_threads_per_block": scale_lanes,
            "raw_scale_divide_expressions": n * 7 * scale_lanes,
            "quant_scale_minimum_max_expressions": n * 7 * scale_lanes,
            "ue8m0_integer_expressions": n * 7 * scale_lanes * 5,
            "quant_max_expressions": n * 7 * (32 * (1 + 5) if ratio == 128 else 8 * (8 + 3)),
            "scope": "source scalar expressions, not compiler instructions or baseline replicated service"}
    return {"stage": stage, "state_dimension": dim, "completed_states": n, "valid_window_rows": r,
        "boundary_ends_by_request": boundaries, "window_rows_per_boundary": window,
        "state_half_selector": "window_row // 4" if ratio == 4 else "0", "arithmetic_groups": count,
        "demand": {name: sum(operations.values()) for name, operations in count.items()},
        "unpriced_typed_actions": typed_actions,
        "store_program": None if stage == "compress" else {
            "normalization_input_dtype": "fp32", "gamma_storage_dtype": "bf16", "normalization_math_dtype": "fp32",
            "rope_math_dtype": "fp32", "rotary_position": "boundary_end_exclusive - compress_ratio",
            "quant_pre_cast": "fp32 -> bf16 -> fp32", "fp8_value_dtype": "e4m3",
            "quantized_values_per_state": raw["index_head_dim"] if indexer else raw["head_dim"] - raw["qk_rope_head_dim"],
            "quant_group_size": 128 if indexer else 64,
            "quantization_position": "after RoPE on all 128 values" if indexer else "NoPE branch only; RoPE branch stored BF16",
            "raw_scale_expression": "max(absmax, 1e-4) * (1 / 448)" if indexer else "max(1e-4, absmax / 448)",
            "scale_rounding": "ceil(log2(raw_scale))" if indexer else "((fp32_bits + 0x7fffff) >> 23) & 0xff",
            "inverse_scale_expression": "exp2(-exponent)" if indexer else "reinterpret_fp32((254 - ue8m0) << 23)",
            "stored_scale_expression": "exp2(exponent)" if indexer else "ue8m0 byte",
            "fp8_conversion": "tl.float8e4nv" if indexer else "_fp32x2_to_fp8e4m3x2",
            "scale_storage_dtype": "fp32" if indexer else "uint8_ue8m0",
            "ue8m0_bit_operations_per_block": 0 if indexer else 5,
            "cache_address_program": {
                "condition": "all valid completed boundaries; uint8 packed main/indexer cache",
                "slot_source": "k_cache_metadata.slot_mapping", "slot_dtype": "int64",
                "state_slot_tensor_used_for_output_address": False,
                "block_rows_source": "kv_cache.shape[1]", "block_stride_source": "kv_cache.stride(0) in bytes",
                "block_rows_symbol": "B", "block_stride_symbol": "S",
                "data_stride_bytes": dim if indexer else dim + raw["qk_rope_head_dim"],
                "scale_stride_bytes": 4 if indexer else 8,
                "page_index": "destination_slot // B", "row_offset": "destination_slot % B",
                "page_base_bytes": "(destination_slot // B) * S",
                "data_base_bytes": "page_base + (destination_slot % B) * D",
                "scale_base_bytes": "page_base + B * D + (destination_slot % B) * Q",
                "per_token_AoS_layout_assumed": False,
                "per_state_regions": ([
                    {"role": "fp8_values", "base": "data_base", "offset_bytes": 0, "bytes": dim, "dtype": "e4m3"},
                    {"role": "scale", "base": "scale_base", "offset_bytes": 0, "bytes": 4, "dtype": "fp32"}]
                    if indexer else [
                    {"role": "nope_values", "base": "data_base", "offset_bytes": 0,
                     "bytes": dim - raw["qk_rope_head_dim"], "dtype": "e4m3"},
                    {"role": "rope_values", "base": "data_base", "offset_bytes": dim - raw["qk_rope_head_dim"],
                     "bytes": 2 * raw["qk_rope_head_dim"], "dtype": "bf16"},
                    {"role": "scales", "base": "scale_base", "offset_bytes": 0, "bytes": 7, "dtype": "uint8_ue8m0"},
                    {"role": "scale_padding", "base": "scale_base", "offset_bytes": 7, "bytes": 1, "dtype": "uint8", "value": 0}]),
                "page_domain": "B positive; S >= B*(D+Q); byte-addressable cache with source store alignment",
                "metadata_values_or_physical_pages_bound": False}},
        "conversion_value_counts": None if stage == "compress" else {
            "gamma_bf16_to_fp32": e, "pre_quant_fp32_to_bf16": n * (dim if indexer else dim - raw["qk_rope_head_dim"]),
            "pre_quant_bf16_to_fp32": n * (dim if indexer else dim - raw["qk_rope_head_dim"]),
            "fp32_to_fp8": n * (dim if indexer else dim - raw["qk_rope_head_dim"]),
            "rope_fp32_to_bf16": 0 if indexer else n * raw["qk_rope_head_dim"]},
        "metadata_reads": finish_metadata_reference(sum(request["query"] for request in requests), n, r, dim, ratio, stage),
        "source_replication": repeated,
        "source_temporary": temporary,
        "indexer_vector_domain": None if not indexer or stage == "store" else {
            "score_cells": n * 8 * dim, "active_score_cells": r * dim,
            "masked_negative_position_cells": (n * 8 - r) * dim, "nonrotary_identity_rope_values": n * 64,
            "nonrotary_identity_rope_arithmetic": n * 64 * 3},
        "scope": "active-value arithmetic/byte Roofline; minimum norm/quant trees, explicit C128 merge; not native issue counts",
        "unpriced_source_work": ["FP32 maximum/minimum, absolute, ceil and sign actions; not INT32 ALU operations",
            "compiler mapping and dependency service of floating reductions and division", "BF16/FP8 typed conversions", "main UE8M0 integer bit construction",
            "normalization/quantizer lane replication", "masked indexer vector work and identity RoPE",
            "address, guards, barriers and shared-memory instructions"]}


def _attach_simple_accounts(raw, cases, operators, projections):
    """Source math/typed global payloads; no production formula is imported.

    Ordinary float operations are distinguished from semantic comparisons,
    integer preparation and library instruction work. Accounts not yet
    derived stay None on every instance, exposing the real unchecked count.
    """
    case_map = {case["phase"]: case for case in cases}
    projection_map = {projection["node_id"]: projection for projection in projections}
    roles = ("compute_tensor", "compute_fma", "compute_reduction", "compute_sfu", "compute_alu")

    def demand(**operations):
        result = {role: 0 for role in roles}
        if not set(operations).issubset(result):
            raise ValueError("unknown independent arithmetic role")
        result.update(operations)
        return result

    def operand(role, dtype, count, width, access):
        return {"role": role, "dtype": dtype, "elements": count, "bytes": width * count, "access": access}

    for record in operators:
        case = case_map[record["node_id"].split(":", 1)[0]]
        tokens, rows = sum(request["query"] for request in case["requests"]), len(case["requests"])
        variant = record["variant"]
        record["math_account"] = None
        record["byte_account"] = None
        record["lowering_account"] = None
        if variant in ("dsv4_sparse_prefill", "dsv4_sparse_decode"):
            padded_heads = 64 if raw["num_attention_heads"] // 4 <= 64 else 128
            record["attention_inputs"] = {
                "attention_sink": {"dtype": "fp32", "elements": padded_heads, "bytes": 4 * padded_heads, "access": "read",
                    "scope": "unique per-layer/per-rank parameter input, not per-query reload count"},
                "sink_initialization": "padded heads initialized to -inf; live slots loaded from checkpoint",
                "checkpoint_content_bound": False, "numerical_callee_work_bound": False, "actual_provider_bound": False}
        if record["kind"] in ("linear", "router", "lm_head"):
            projection = projection_map[record["node_id"]]
            if projection["weight_dtype"] != "fp8" or variant == "batched_gemm":
                # Plain BF16 and wo_a's separately prepared FP8 input each
                # have one matrix application in this declared source case.
                record["lowering_account"] = [{"kind": "GEMM", "variant": variant, "dtype": record["dtype"]}]
            else:
                # DeepGEMM's inherited apply_weights quantizes separately;
                # FlashInfer subclasses can accept BF16 directly. Shape
                # alone does not bind the provider/quantization placement.
                record["lowering_source"] = "fp8-linear-input-quant"
                record["lowering_source_condition"] = "apply_input_quant and the selected block-FP8 source kernel; extra fp8_group_quant is not a duplicate matrix application"
        elif record["kind"] == "fused_moe" and variant == "full":
            # The source modular caller packages alignment, both GEMMs,
            # activation and top-k reduction in one expert application. A
            # logical FUSED_MOE call represents that entire source program,
            # not one CUDA launch. Its inner ordered stages are compared
            # separately through source_work/roofline_stage_templates.
            record["lowering_source"] = "marlin-moe-source"
            record["lowering_account"] = [{
                "kind": "FUSED_MOE", "variant": "routed_experts", "dtype": "bf16",
                "required_shape": {**record["shape"], "dtype_bytes": 2},
                "required_attributes": {"expert_parallel_size": 1,
                    "weight_group_size": 32, "routed_weight_dtype": "mxfp4",
                    "swiglu_limit": raw["swiglu_limit"]}}]
        elif record["kind"] == "collective":
            record["lowering_account"] = [{"kind": "COMMUNICATION", "variant": None, "dtype": None}]
            operation = record["attributes"]["operation"]
            if operation not in ("all_reduce", "all_gather"):
                raise ValueError("Flash source inventory has an unrecognized collective")
            size, ranks = record["attributes"]["bytes"], record["attributes"]["participants"]
            record["account_source"] = "tp-all-reduce" if operation == "all_reduce" else "tp-all-gather"
            record["math_account"] = {"mode": "collective_semantics", "operation": operation,
                "participants": ranks, "reduction": "sum" if operation == "all_reduce" else None,
                "input_dtype": "bf16", "minimum_global_sum_adds": (len(ranks) - 1) * (size // 2) if operation == "all_reduce" else 0,
                "scope": "global mathematical sum/gather; reduction order, collective scratch and library instructions unbound"}
            record["byte_account"] = {"mode": "collective_payload", "per_rank_input_bytes": size,
                "per_rank_output_bytes": size if operation == "all_reduce" else size * len(ranks),
                "bytes_semantics": "per_rank_input",
                "scope": "logical tensor payloads; not local buffer visits, actual network traffic or NCCL algorithm"}
        elif variant in ("dsv4_q_kv_rmsnorm", "dsv4_short_context_topk", "silu", "add", "rmsnorm",
                         "vllm_logits_fp32_cast", "vllm_logits_gather_reorder", "vllm_plain_greedy_sample", "vllm_logits_row_select",
                         "dsv4_indexer_q_rope_quant", "mhc_post", "hc_head"):
            record["lowering_account"] = [{"kind": "VECTOR", "variant": "elementwise" if variant == "add" else variant,
                                           "dtype": record["dtype"]}]
            if variant == "vllm_logits_gather_reorder":
                record["lowering_account"][0].update(required_shape={**record["shape"], "dtype_bytes": 2},
                    required_attributes={"model_execution_version": "vllm_0_30_0"}, unbound_library_call=True)
        elif variant == "dsv4_moe_route":
            hashed = record["attributes"]["expert_id_selection"] == "hash_table"
            work = routing_source_reference(raw, tokens, hashed)
            record["account_source"] = "v4-route-source"
            record["route_source_work"] = work
            record["lowering_account"] = [{"kind": "VECTOR", "variant": variant, "dtype": "fp32",
                "required_shape": {"rows": tokens, "experts": raw["n_routed_experts"], "topk": raw["num_experts_per_tok"]}}]
            record["math_account"] = {"demand": work["demand"]}
            record["byte_account"] = {"mode": "modeled_l2_bytes", **work["source_role_read_write_bytes"]}
        elif variant in ("dsv4_indexer_prefill", "dsv4_indexer_decode"):
            work = record["indexer_baseline"]
            record["account_source"] = "indexer-baseline"
            record["lowering_account"] = [{"kind": "SPARSE_ATTN_INDEXER", "variant": variant, "dtype": "int32"}]
            record["math_account"] = {"demand": work["demand"]}
            record["byte_account"] = {"mode": "modeled_l2_bytes", **work["read_write_bytes"]}
        elif variant == "dsv4_inv_rope_fp8_quant":
            record["lowering_source"] = "inverse-rope-quant"
            record["lowering_account"] = [{"kind": "VECTOR", "variant": variant, "dtype": "fp8"}]
        elif variant == "dsv4_compressor_state_cache":
            record["lowering_account"] = [{"kind": "KV_CACHE", "variant": variant, "dtype": "fp32"}]
        elif variant == "dsv4_qnorm_rope_kv_insert":
            record["lowering_account"] = [{"kind": "KV_CACHE", "variant": variant, "dtype": "bf16"}]
        elif variant == "dsv4_compressed_state_cache":
            ratio = record["shape"]["compress_ratio"]
            width = raw["index_head_dim"] if record["shape"]["state_bytes"] == 132 else raw["head_dim"]
            completed_ends = [end for request in case["requests"]
                              for end in range((request["computed"] // ratio + 1) * ratio,
                                               request["computed"] + request["query"] + 1, ratio)]
            window_rows = sum(min(2 * ratio if ratio == 4 else ratio, end) for end in completed_ends)
            count = len(completed_ends)
            frame_read = 2 * 4 * width * window_rows + 4 * window_rows
            norm_rope_read = count * (2 * width + 4 * raw["qk_rope_head_dim"])
            if record["shape"]["compress_ratio"] == 128:
                states = sum((request["computed"] + request["query"]) // 128 - request["computed"] // 128
                             for request in case["requests"])
                record["lowering_source"] = "C128-finish-stages"
                record["lowering_account"] = [
                    {"kind": "KV_CACHE", "variant": variant, "dtype": "fp32", "finish_stage": "compress",
                     "element_count": states * raw["head_dim"], "dtype_bytes": 4,
                     "output_write_bytes": states * raw["head_dim"] * 4,
                     "exp_calls": states * raw["head_dim"] * (128 + 8),
                     "source_payload_bytes": {"read": frame_read + 20 * tokens, "write": count * width * 4}},
                    {"kind": "KV_CACHE", "variant": variant, "dtype": "uint8", "finish_stage": "store",
                     "element_count": states * 584, "dtype_bytes": 1, "output_write_bytes": states * 584,
                     "source_sfu_calls": states, "sfu_source": "C128-finish-store",
                     "quantization_actions": {
                         "quant_abs": {"dtype": "fp32", "operation": "absolute", "elements": states * 448},
                         "quant_lower_clamp": {"dtype": "fp32", "operation": "maximum", "elements": states * 448},
                         "quant_upper_clamp": {"dtype": "fp32", "operation": "minimum", "elements": states * 448}},
                     "source_payload_bytes": {"read": count * width * 4 + 24 * tokens + norm_rope_read,
                                              "write": count * record["shape"]["state_bytes"]}}]
                record["byte_account"] = {"mode": "source_stages"}
            else:
                state_dim = 128 if record["shape"]["state_bytes"] == 132 else 512
                boundaries = [end for request in case["requests"]
                              for end in range((request["computed"] // 4 + 1) * 4,
                                               request["computed"] + request["query"] + 1, 4)]
                # C4 first boundary has four valid rows, later ones eight.
                # Main CuTe is online old/new rescaling (two exponentials
                # per value/row); indexer Triton softmax is one. Its FP32
                # scale store requires a third SFU beyond log2/inv-exp2.
                values = sum(min(8, end) for end in boundaries) * state_dim
                source = "indexer-finish" if state_dim == 128 else "C4-finish"
                sfu = values + 4 * len(boundaries) if state_dim == 128 else 2 * values + len(boundaries)
                record["lowering_account"] = [{"kind": "KV_CACHE", "variant": variant, "dtype": "uint8",
                                               "source_sfu_calls": sfu, "sfu_source": source,
                                               "quantization_actions": {
                                                   "quant_abs": {"dtype": "fp32", "operation": "absolute", "elements": count * (128 if state_dim == 128 else 448)},
                                                   "quant_lower_clamp": {"dtype": "fp32", "operation": "maximum", "elements": count * (128 if state_dim == 128 else 448)},
                                                   "quant_upper_clamp": {"dtype": "fp32", "operation": "minimum", "elements": count * (128 if state_dim == 128 else 448)}}}]
                guard_read = 16 * tokens + 12 * count if state_dim == 128 else 24 * tokens + 4 * count
                record["byte_account"] = {"mode": "modeled_l2_bytes", "read": frame_read + guard_read + norm_rope_read,
                    "write": count * record["shape"]["state_bytes"],
                    "scope": "FP32 KV/score windows, boundary metadata, BF16 gamma, FP32 rotary row and packed cache stores; all slots valid, operand role uses not observed DRAM"}
            record["account_source"] = "indexer-finish" if width == 128 else "C4-finish" if ratio == 4 else "C128-finish-stages"
            for stage in record["lowering_account"]:
                source_stage = stage["finish_stage"] if "finish_stage" in stage else None
                stage["finish_work"] = finish_source_reference(raw, case["requests"], ratio=ratio,
                    indexer=width == raw["index_head_dim"], stage=source_stage)
                if ratio == 4:
                    stage["source_payload_bytes"] = {name: record["byte_account"][name] for name in ("read", "write")}
            record["math_account"] = {"mode": "source_stages"}
            record["byte_account"] = {"mode": "source_stages"}
        if record["kind"] in ("linear", "router", "lm_head", "collective"):
            continue
        if variant == "dsv4_qnorm_rope_kv_insert":
            record["account_source"] = "qnorm-rope-insert"
            heads, width, rope = raw["num_attention_heads"] // 4, raw["head_dim"], raw["qk_rope_head_dim"]
            q_values = tokens * heads * width
            # Source logical scalar groups: Q square/accumulate/scale and
            # one per-head RMS scalar; full warp-sum adds; per-lane KV
            # absmax/scale work. Compiler replication of the Q RMS scalar
            # and integer address/control are excluded, not inferred.
            record["math_account"] = {"demand": demand(
                compute_fma=2 * q_values + q_values + 2 * tokens * heads
                    + tokens * heads * 32 * 5
                    + 3 * tokens * (heads + 1) * rope + tokens * (448 + 32 + 28),
                compute_reduction=0, compute_alu=0,
                compute_sfu=tokens * heads + tokens * 32 * 2)}
            padded = 64 if heads <= 64 else 128
            record["byte_account"] = {"mode": "modeled_l2_bytes",
                "read": tokens * (heads * width * 2 + width * 2 + 8 + 8 + rope * 4),
                "write": tokens * (padded * width * 2 + 448 + 64 * 2 + 8),
                "scope": "unique live BF16 Q/KV, INT64 position/slot, FP32 trig row; BF16 Q padding and packed448-FP8/64-BF16/8-byte-scale cache record; repeated per-head metadata reads and physical residency separate"}
        elif variant == "dsv4_inv_rope_fp8_quant":
            heads, width = raw["num_attention_heads"] // 4, raw["head_dim"]
            rope = raw["qk_rope_head_dim"]
            values = tokens * heads * width
            blocks = values // 128
            scale_backing = ((tokens + 3) // 4 * 4) * heads * width // 128
            record["account_source"] = "inverse-rope-quant"
            # Logical selected inverse rotation: two multiplies and add/sub
            # per rope value. Scalar quantization counts abs, logical max
            # reduction, epsilon max, ceil, divide, clamp and FP8 conversion;
            # compiler eager evaluation of both rotation alternatives and
            # predicate/address work are outside these mathematical groups.
            record["math_account"] = {"demand": demand(
                compute_fma=3 * tokens * heads * rope + blocks + values,
                compute_alu=0, compute_sfu=2 * blocks)}
            record["byte_account"] = {"mode": "modeled_l2_bytes",
                "read": 2 * values + 8 * tokens + 4 * tokens * rope,
                "write": values + 4 * scale_backing,
                "scope": "unique BF16 source, INT64 positions, FP32 rotary rows, live FP8 output and complete FP32 scale backing including zero padding; partner/per-head reloads and physical DRAM separate"}
        elif variant == "dsv4_q_kv_rmsnorm":
            width = raw["q_lora_rank"] + raw["head_dim"]
            elements = tokens * width
            record["account_source"] = "qkv-norm"
            record["math_account"] = {"demand": demand(compute_fma=3 * elements + 4 * tokens,
                compute_reduction=elements - 2 * tokens, compute_sfu=2 * tokens, compute_alu=0)}
            record["byte_account"] = {"mode": "modeled_l2_bytes", "read": 2 * elements + 2 * width, "write": 2 * elements}
            record["byte_account"]["scope"] = "unique activation tensors plus one unique gamma vector; not per-row parameter reloads or observed traffic"
        elif variant == "rmsnorm":
            hidden = raw["hidden_size"]
            count = tokens * hidden
            # Independent source loop census, including masked zero lanes.
            chunk_count = len(range(0, hidden, 1024))
            padded_count = tokens * sum(1024 for _ in range(0, hidden, 1024))
            record["account_source"] = "final-rmsnorm"
            # Minimal mathematical sum, independent of CUB's thread tree
            # and zero-initialized accumulator additions. The IR reference
            # and vllm_c have different rounding before gamma multiplication.
            record["math_account"] = {"primitive_fields": {
                "square_multiply": count, "sum_add": tokens * (hidden - 1),
                "mean_divide": tokens, "epsilon_add": tokens, "rsqrt": tokens,
                "normalize_multiply": count, "gamma_multiply": count},
                "semantic_fields": {
                    "epsilon": raw["rms_norm_eps"],
                    "source_programs": [
                        {"implementation": "ir_reference", "condition": "IR rms_norm reference; weighted BF16, variance_size=None",
                         "variance_dtype": "fp32", "normalized_value_rounding_before_gamma": "bf16",
                         "gamma_multiply_dtype": "bf16", "output_dtype": "bf16"},
                        {"implementation": "vllm_c", "condition": "IR selects vllm_c rms_norm; weighted BF16, variance_size=None",
                         "variance_dtype": "fp32", "normalized_value_rounding_before_gamma": None,
                         "gamma_multiply_dtype": "fp32", "output_dtype": "bf16"},
                        {"implementation": "batch_invariant_triton", "condition": "CUDA VLLM_BATCH_INVARIANT; 2D weighted BF16, residual=None",
                         "variance_dtype": "fp32", "normalized_value_rounding_before_gamma": None,
                         "gamma_multiply_dtype": "fp32", "output_dtype": "bf16", "block_columns": 1024,
                         "source_float_actions": {"square_multiply": padded_count,
                             "chunk_sum_add": tokens * chunk_count * (1024 - 1),
                             "zero_seeded_chunk_accumulate_add": tokens * chunk_count, "mean_divide": tokens,
                             "epsilon_add": tokens, "sqrt": tokens, "inverse_divide": tokens,
                             "normalize_multiply": padded_count, "gamma_multiply": padded_count},
                         "kernel_source_payload_bytes": {"input_read": 4 * count, "weight_read": 2 * count, "output_write": 2 * count},
                         "wrapper_copies": [
                             {"condition": "input_2d is not contiguous", "dtype": "bf16", "read_bytes": 2 * count, "write_bytes": 2 * count},
                             {"condition": "weight is not contiguous", "dtype": "bf16", "read_bytes": 2 * hidden, "write_bytes": 2 * hidden}],
                         "global_fp32_temporary_tensors": [], "selected_or_compiled": False}],
                    "selected_implementation": None, "numerical_equivalence_asserted": False}}
            record["byte_account"] = {"mode": "logical_operands", "operands": [
                operand("hidden_states", "bf16", count, 2, "read"),
                operand("norm_weight", "bf16", hidden, 2, "read"),
                operand("normalized_hidden_states", "bf16", count, 2, "write")],
                "scope": "unique BF16 tensors; repeated input/gamma reads and provider temporaries remain separate"}
        elif variant == "dsv4_compressor_state_cache":
            state_bytes = record["shape"]["state_bytes"]
            values = tokens * state_bytes // 4
            record["account_source"] = "save-partial"
            record["math_account"] = {"demand": demand(compute_fma=values // 2, compute_alu=tokens)}
            record["byte_account"] = {"mode": "modeled_l2_bytes", "read": values * 6 + 16 * tokens, "write": values * 4}
            record["byte_account"]["scope"] = "FP32 projection and APE row use per active token, plus INT64 position/slot; repeated APE rows are role uses, not unique footprint"
        elif variant == "dsv4_short_context_topk":
            entries = tokens * raw["index_topk"]
            record["account_source"] = "indexer-forward"
            record["math_account"] = {"demand": demand(compute_alu=2 * tokens + 2 * entries)}
            record["byte_account"] = {"mode": "modeled_l2_bytes", "read": 8 * tokens, "write": 4 * entries}
            record["byte_account"]["scope"] = "one logical position per row and the full INT32 topk capacity store"
        elif record["kind"] == "activation" and variant == "silu":
            elements = tokens * (raw["moe_intermediate_size"] // 4)
            activation = clamped_activation_arithmetic_reference(elements, "bf16")
            record["account_source"] = "clamped-activation"
            record["math_account"] = {"demand": demand(**activation["baseline_demand"]),
                "semantic_fields": {"activation_result_rounding_dtype": "bf16", "rounding_before_up_multiply": True,
                                    "temporary_is_register_value": True, "alpha": 1.0, "beta": 0.0,
                                    **{name: activation[name] for name in ("mathematical_primitives", "source_negation_programs", "unpriced_source_actions")}}}
            record["byte_account"] = {"mode": "modeled_l2_bytes", "read": 4 * elements, "write": 2 * elements}
            record["byte_account"]["scope"] = "BF16 gate/up read and output store; intermediate BF16 rounding is in registers without a global allocation"
        elif record["kind"] == "elementwise" and variant == "add":
            elements = tokens * raw["hidden_size"]
            record["account_source"] = "ffn-forward"
            record["math_account"] = {"demand": demand(compute_fma=elements)}
            record["byte_account"] = {"mode": "modeled_l2_bytes", "read": 4 * elements, "write": 2 * elements}
            record["byte_account"]["scope"] = "two BF16 input payloads and one BF16 output; DRAM cache behavior is a separate prediction"
        elif variant == "vllm_logits_gather_reorder":
            tp, vocab = record["shape"]["tensor_parallel_size"], raw["vocab_size"]
            shard, count = vocab // tp, rows * vocab
            record["account_source"] = "tp-gather-layout-cuda"
            record["math_account"] = {"demand": demand(), "semantic_fields": {
                "input_shape": [tp, rows, shard], "input_strides_elements": [rows * shard, shard, 1],
                "movedim_shape": [rows, tp, shard], "movedim_strides_elements": [shard, rows * shard, 1],
                "output_shape": [rows, vocab], "output_strides_elements": [vocab, 1],
                "output_aliases_collective_backing": False, "input_mutated": False}}
            record["byte_account"] = {"mode": "logical_operands", "logical_bytes": 4 * count, "operands": [
                operand("rank_major_logits", "bf16", count, 2, "read"),
                operand("row_major_logits", "bf16", count, 2, "write")]}
        elif variant in ("vllm_logits_fp32_cast", "vllm_plain_greedy_sample", "vllm_logits_row_select"):
            record["account_source"] = record["source"]
            record["math_account"] = {"demand": demand()}
            if variant == "vllm_logits_fp32_cast":
                count = rows * raw["vocab_size"]
                operands = [operand("gathered_logits", "bf16", count, 2, "read"),
                            operand("sampler_fp32_logits", "fp32", count, 4, "write")]
            elif variant == "vllm_logits_row_select":
                count = rows * raw["hidden_size"]
                operands = [operand("selected_target_hidden_source_rows", "bf16", count, 2, "read"),
                            operand("logits_indices", "int32", rows, 4, "read"),
                            operand("sample_hidden_states", "bf16", count, 2, "write")]
            elif record["node_id"].split(":tp-rank-", 1)[0].endswith(":target-sampler-argmax"):
                operands = [operand("processed_fp32_logits", "fp32", rows * raw["vocab_size"], 4, "read"),
                            operand("greedy_ids", "int64", rows, 8, "write")]
                record["math_account"]["argmax_pair"] = argmax_pair_source_work(rows, raw["vocab_size"])
            else:
                operands = [operand("greedy_ids", "int64", rows, 8, "read"),
                            operand("sampler_output_ids", "int32", rows, 4, "write")]
                record["math_account"]["semantic_fields"] = {
                    "sampled_long_aliases_int64_argmax": True, "output_unsqueeze_is_view": True}
            record["byte_account"] = {"mode": "logical_operands", "operands": operands}
            record["byte_account"]["scope"] = "typed logical copy/reduction operands; library temporaries, replay and physical DRAM are not inferred"
        elif variant == "dsv4_indexer_q_rope_quant":
            groups = tokens * raw["index_n_heads"]
            record["account_source"] = "indexer-q-quant"
            record["math_account"] = {"primitive_fields": record["source_work"]["mathematical_primitives"],
                "primitive_condition": record["source_work"]["mathematical_primitives_condition"]}
            record["byte_account"] = {"mode": "logical_operands", "operands": [
                operand("projected_q", "bf16", groups * raw["index_head_dim"], 2, "read"),
                operand("indexer_weights", "bf16", groups, 2, "read"),
                operand("positions", "int64", tokens, 8, "read"),
                operand("rotary_cache", "fp32", tokens * raw["qk_rope_head_dim"], 4, "read"),
                operand("quantized_q", "fp8", groups * raw["index_head_dim"], 1, "write"),
                operand("folded_weights", "fp32", groups, 4, "write")]}
            record["byte_account"]["scope"] = "unique Q, weights, positions and FP32 rotary row payload; per-head reloads and physical residency remain separate"
