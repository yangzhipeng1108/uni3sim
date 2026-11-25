"""Independent mHC source accounts; no candidate/production imports.

Counts describe logical MAC/scalar groups and unique global operands, not
compiler instructions or measured cache traffic. Nonfused prenorm retains
its backend condition instead of assuming runtime DeepGEMM availability.
"""

import ast
import math


def _validate_prenorm_selection(support_source, wrapper_source):
    """Bind the Boolean proof to the actual source, without importing vLLM."""
    def function(source, name):
        matches = [node for node in ast.parse(source).body
                   if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(matches) != 1:
            raise ValueError("mHC source selector is missing or ambiguous")
        return matches[0]

    support = function(support_source, "is_deep_gemm_supported")
    returns = [node.value for node in support.body if isinstance(node, ast.Return)]
    expected = ast.parse("envs.VLLM_USE_DEEP_GEMM and has_deep_gemm() and is_supported_arch", mode="eval").body
    if len(returns) != 1 or ast.dump(returns[0]) != ast.dump(expected):
        raise ValueError("mHC DeepGEMM support Boolean program changed")
    helper = function(wrapper_source, "_hc_prenorm_gemm_outputs")
    selectors = [node.value for node in helper.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "use_deep_gemm" for target in node.targets)]
    expected = ast.parse("is_deep_gemm_supported() or not use_tilelang_fallback", mode="eval").body
    if len(selectors) != 1 or ast.dump(selectors[0]) != ast.dump(expected):
        raise ValueError("mHC prenorm fallback Boolean program changed")
    defaults = dict(zip((arg.arg for arg in helper.args.kwonlyargs), helper.args.kw_defaults, strict=True))
    if ("use_tilelang_fallback" not in defaults or not isinstance(defaults["use_tilelang_fallback"], ast.Constant)
            or defaults["use_tilelang_fallback"].value is not True):
        raise ValueError("mHC optional prenorm no longer allows fallback by default")
    fused = function(wrapper_source, "mhc_fused_post_pre_tilelang")
    ordinary_calls = [node for node in ast.walk(fused) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == "_hc_prenorm_gemm_outputs"]
    if len(ordinary_calls) != 1 or any(keyword.arg == "use_tilelang_fallback" for keyword in ordinary_calls[0].keywords):
        raise ValueError("mHC nonfused prenorm changed its fallback argument")
    broadcast = function(wrapper_source, "mhc_pre_broadcast_tilelang")
    calls = [node for node in ast.walk(broadcast) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_hc_prenorm_gemm_outputs"]
    if len(calls) != 1 or not any(keyword.arg == "use_tilelang_fallback"
            and isinstance(keyword.value, ast.Constant) and keyword.value.value is False for keyword in calls[0].keywords):
        raise ValueError("mHC broadcast no longer forces DeepGEMM")


def _fused_config(source):
    """Execute only the upstream pure integer dispatch function and literals."""
    tree = ast.parse(source)
    name = "mhc_fused_post_pre_split_config"
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError("mHC split-config source is missing or ambiguous")
    function = matches[0]
    if any(isinstance(node, (ast.Call, ast.Import, ast.ImportFrom)) for node in ast.walk(function)):
        raise ValueError("mHC split-config is no longer pure integer dispatch")
    literals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                        "_FUSED_POST_PRE_N_THR", "_FUSED_POST_PRE_MAX_TOKENS"):
                    literals[target.id] = ast.literal_eval(node.value)
    if len(literals) != 2:
        raise ValueError("mHC split-config literal constants changed")
    function.decorator_list = []
    namespace = {"__builtins__": {}, "int": int, "tuple": tuple, **literals}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "upstream_mhc_split_config", "exec"), namespace)
    return namespace[name]


def mhc_stage_reference(raw, stage, tokens, *, splits=1, projection_tile=None):
    """Derive one source-stage ledger from raw config and declared splits."""
    h, c = raw["hidden_size"], raw["hc_mult"]
    n = c * (c + 2)
    operands = []

    def tensor(role, dtype, shape, access):
        operands.append({"role": role, "dtype": dtype, "shape": list(shape),
                         "bytes": math.prod(shape) * {"bf16": 2, "fp32": 4}[dtype], "access": access})

    def post_inputs():
        tensor("sublayer_output", "bf16", [tokens, h], "read")
        tensor("residual_input", "bf16", [tokens, c, h], "read")
        tensor("post_coefficients", "fp32", [tokens, c], "read")
        tensor("residual_coefficients", "fp32", [tokens, c, c], "read")
        tensor("residual_output", "bf16", [tokens, c, h], "write")

    fma = alu = sfu = 0
    typed_actions = {}
    nonlinear = {"sigmoid": 0, "exp": 0, "rsqrt": 0}
    views = []
    if stage in ("mhc_prenorm", "mhc_prenorm_broadcast"):
        width = h if stage.endswith("broadcast") else c * h
        tensor("input", "bf16", [tokens, width], "read")
        tensor("weight", "fp32", [n, width], "read")
        tensor("projection_partial", "fp32", [splits, tokens, n], "write")
        tensor("sqrsum_partial", "fp32", [splits, tokens], "write")
        fma = tokens * width * (n + 1) * 2
        semantics = {"projection_partial_dtype": "fp32", "sqrsum_partial_dtype": "fp32",
                     "projection_input": "BF16 global values converted to FP32", "global_fp32_input_copy": False}
    elif stage in ("mhc_post", "mhc_post_prenorm"):
        post_inputs()
        repetitions = 1 if stage == "mhc_post" else n // projection_tile
        fma = repetitions * tokens * c * h * (2 * c + 1)
        if stage == "mhc_post":
            semantics = {"post_mapping_accumulator_dtype": "fp32", "residual_output_dtype": "bf16",
                         "post_initial_multiply_dtype": "fp32", "post_initial_multiply_values": tokens * c * h}
        else:
            tensor("projection_weight", "fp32", [n, c, h], "read")
            tensor("projection_partial", "fp32", [splits, tokens, n], "write")
            tensor("sqrsum_partial", "fp32", [splits, tokens], "write")
            fma += tokens * c * h * (2 * n + 2)
            semantics = {"projection_partial_dtype": "fp32", "sqrsum_partial_dtype": "fp32",
                         "residual_output_dtype": "bf16", "projection_uses_pre_store_fp32_residual": True,
                         "sqrsum_uses_pre_store_fp32_residual": True, "post_mapping_projection_tiles": repetitions,
                         "post_initial_multiply_dtype": "fp32", "post_initial_multiply_values": repetitions * tokens * c * h}
    elif stage in ("mhc_pre_norm", "mhc_pre_norm_broadcast"):
        broadcast = stage.endswith("broadcast")
        tensor("projection_partial", "fp32", [splits, tokens, n], "read")
        tensor("sqrsum_partial", "fp32", [splits, tokens], "read")
        tensor("residual_input", "bf16", [tokens, h] if broadcast else [tokens, c, h], "read")
        tensor("norm_weight", "bf16", [h], "read")
        tensor("hc_scale", "fp32", [3], "read")
        tensor("hc_base", "fp32", [n], "read")
        tensor("post_coefficients", "fp32", [tokens, c], "write")
        tensor("residual_coefficients_flat", "fp32", [tokens, c * c], "write")
        tensor("layer_input", "bf16", [tokens, h], "write")
        if broadcast:
            tensor("residual_output", "bf16", [tokens, c, h], "write")
        # collapse MACs and the two gamma/rsqrt multiplications per output.
        fma = tokens * h * (2 * c + 2)
        # Preserve original source function counts separately from ordinary
        # FP32 arithmetic. Initial row max is an unpriced typed comparison.
        rounds = raw["hc_sinkhorn_iters"]
        sinkhorn = c * (c - 1) + c * c
        for unused in range(rounds):
            sinkhorn += 2 * c * (c - 1) + 4 * c * c
        gate = 3 * c + 3 * c + 2 * c * c
        alu = tokens * (splits * (n + 1) + n + gate + sinkhorn + 4)
        alu += tokens * (2 * h + math.gcd(1024, h) - 1)
        maxima = tokens * c * (c - 1)
        fma += alu - maxima
        alu = 0
        typed_actions["sinkhorn_initial_row_max"] = {"dtype": "fp32", "operation": "maximum", "elements": maxima}
        nonlinear = {"sigmoid": tokens * c * 2, "exp": tokens * c * c, "rsqrt": tokens * 2}
        sfu = tokens * (c + c + c * c + 2)
        semantics = {"projection_partial_dtype": "fp32", "sqrsum_partial_dtype": "fp32",
                     "post_coefficients_dtype": "fp32", "residual_coefficients_dtype": "fp32",
                     "layer_input_dtype": "bf16", "collapse_sumsq_before_shared_bf16_rounding": True,
                     "gamma_multiply_reads_shared_bf16": True, "shared_intermediate_is_global_allocation": False}
        views = [{"base_role": "post_coefficients", "shape": [tokens, c, 1], "operation": "unsqueeze" if broadcast else "view", "new_allocation": False},
                 {"base_role": "residual_coefficients_flat", "shape": [tokens, c, c], "operation": "view", "new_allocation": False}]
    elif stage == "hc_head":
        tensor("residual_input", "bf16", [tokens, c, h], "read")
        tensor("projection_weight", "fp32", [c, c * h], "read")
        tensor("hc_scale", "fp32", [1], "read")
        tensor("hc_base", "fp32", [c], "read")
        tensor("layer_input", "bf16", [tokens, h], "write")
        fma = tokens * c * h * (2 * c + 2 + 2)
        fma += tokens * (4 * c + 2)
        sfu = tokens * (c + 1)
        nonlinear = {"sigmoid": tokens * c, "exp": 0, "rsqrt": tokens}
        semantics = {"projection_and_gate_dtype": "fp32", "layer_input_dtype": "bf16",
                     "mixes_and_rsqrt_are_global_allocations": False, "residual_source_passes": 2}
    else:
        raise ValueError("unknown independent mHC stage")
    return {"stage": stage, "operands": operands,
            "read_bytes": sum(value["bytes"] for value in operands if value["access"] == "read"),
            "write_bytes": sum(value["bytes"] for value in operands if value["access"] == "write"),
            "demand": {"compute_fma": fma, "compute_alu": alu, "compute_sfu": sfu,
                       "compute_tensor": 0, "compute_reduction": 0},
            "semantic_fields": semantics, "views": views,
            "unpriced_typed_actions": typed_actions, "nonlinear_function_counts": nonlinear}


def mhc_nonfused_stage_forms(raw, tokens, *, multiprocessors, source_use_deep_gemm=None):
    """Both source programs, without asserting runtime backend availability.

    tilelang.py runs post before _hc_prenorm_gemm_outputs on BF16
    residual_cur. The helper either partitions K with compute_num_split or
    emits one TileLang partial. Neither device capability nor the split
    shape proves that VLLM_USE_DEEP_GEMM and has_deep_gemm() are true.
    """
    if type(tokens) is not int or type(multiprocessors) is not int or tokens <= 0 or multiprocessors <= 0:
        raise ValueError("nonfused mHC requires positive rows and hardware SM count")
    if source_use_deep_gemm is not None and type(source_use_deep_gemm) is not bool:
        raise TypeError("independent mHC env declaration must be boolean or None")
    width = raw["hidden_size"] * raw["hc_mult"]
    splits = max(min(multiprocessors // ((tokens + 63) // 64), ((width + 63) // 64) // 4), 1)
    result = []
    for backend, count, condition in (
            ("deep_gemm", splits,
             "VLLM_USE_DEEP_GEMM and has_deep_gemm() and current_platform.support_deep_gemm()"),
            ("tilelang", 1,
             "not (VLLM_USE_DEEP_GEMM and has_deep_gemm() and current_platform.support_deep_gemm())")):
        stages = []
        for name in ("mhc_post", "mhc_prenorm", "mhc_pre_norm"):
            shape = {"hidden": raw["hidden_size"], "hc_mult": raw["hc_mult"]}
            if name != "mhc_post":
                shape["n_splits"] = count
            if name == "mhc_pre_norm":
                shape["sinkhorn_iters"] = raw["hc_sinkhorn_iters"]
            stages.append({"kind": "VECTOR", "variant": name, "dtype": "bf16",
                           "required_shape": shape,
                           "mhc_stage_work": mhc_stage_reference(raw, name, tokens, splits=count)})
        result.append({"backend_condition": condition, "backend": backend, "stages": stages})
    selected = "tilelang" if source_use_deep_gemm is False else None
    return {"forms": result, "selected_backend": selected, "actual_provider_bound": False,
            "source_env_declaration": {
                "VLLM_USE_DEEP_GEMM": source_use_deep_gemm,
                "scope": "declared process-lifetime value before cached dispatch; not a runtime observation",
                "has_deep_gemm_bound": False, "platform_support_bound": False,
                "forced_broadcast_uses_deep_gemm": True},
            "scope": "conditional source math/typed stages; source form compatibility does not bind a runtime branch",
            "residual_materialization": {
                "dtype": "bf16", "shape": [tokens, raw["hc_mult"], raw["hidden_size"]],
                "writer_stage": "mhc_post", "reader_stages": ["mhc_prenorm", "mhc_pre_norm"],
                "prenorm_input_view": [tokens, width], "view_allocates": False,
                "projection_reads_pre_store_fp32_residual": False,
                "actual_allocator_identity_bound": False}}


def mhc_wrapper_object_reference(raw, rows, variant, group):
    """Caller tensor objects; split count is a source parameter, not a selected provider."""
    h, c = raw["hidden_size"], raw["hc_mult"]
    n = c * (c + 2)
    parameters, views, objects, returned = [], [], [], []

    def parameter(name, dtype, shape, numeric_read=True):
        return {"name": name, "dtype": dtype, "shape": shape, "contiguous": True, "numeric_read": numeric_read}

    def allocation(name, dtype, shape, lifetime, owner="wrapper"):
        return {"name": name, "dtype": dtype, "shape": shape, "storage": "fresh_contiguous",
                "initialized_by_allocation": False, "lifetime": lifetime, "owner": owner}

    if variant in ("mhc_pre", "mhc_post_pre"):
        stem = "hc_attn" if group == "attention" else "hc_ffn"
        parameters = [parameter(stem + "_fn", "fp32", [n, c * h], variant != "mhc_pre"),
                      parameter(stem + "_scale", "fp32", [3]), parameter(stem + "_base", "fp32", [n]),
                      parameter(("attn_norm" if group == "attention" else "ffn_norm") + ".weight", "bf16", [h])]
        if variant == "mhc_pre":
            parameters.append(parameter("hc_attn_fn_broadcast", "fp32", [n, h]))
        else:
            views = [{"name": "residual_flat", "base": "residual", "shape": [rows, c, h]},
                     {"name": "x_flat", "base": "sublayer_output", "shape": [rows, h]},
                     {"name": "post_layer_mix_flat", "base": "post_mix", "shape": [rows, c]},
                     {"name": "comb_res_mix_flat", "base": "res_mix", "shape": [rows, c, c]},
                     {"name": "fused_weight_view", "base": stem + "_fn", "shape": [n, c, h], "condition": "fused_config is not None"},
                     {"name": "prenorm_residual_view", "base": "residual_result", "shape": [rows, c * h], "condition": "fused_config is None"}]
        objects = [allocation("residual_result", "bf16", [rows, c, h], "returned",
                              "wrapper" if variant == "mhc_pre" else "fused_launcher_or_wrapper"),
                   allocation("post_result", "fp32", [rows, c], "returned"),
                   allocation("comb_result", "fp32", [rows, c * c], "returned"),
                   allocation("layer_input_result", "bf16", [rows, h], "returned"),
                   allocation("projection_partial", "fp32", ["source_n_splits", rows, n], "until pre_norm completion", "prenorm_helper_or_fused_launcher"),
                   allocation("sqrsum_partial", "fp32", ["source_n_splits", rows], "until pre_norm completion", "prenorm_helper_or_fused_launcher")]
        returned = [{"role": role, "base": base, "shape": shape} for role, base, shape in (
            ("residual", "residual_result", [rows, c, h]), ("post_mix", "post_result", [rows, c, 1]),
            ("res_mix", "comb_result", [rows, c, c]), ("layer_input", "layer_input_result", [rows, h]))]
    elif variant == "mhc_post":
        views = [{"name": "post_layer_mix_squeezed", "base": "post_mix", "shape": [rows, c]}]
        objects = [allocation("residual_result", "bf16", [rows, c, h], "returned")]
        returned = [{"role": "residual", "base": "residual_result", "shape": [rows, c, h]}]
    elif variant == "hc_head":
        parameters = [parameter("hc_head_fn", "fp32", [c, c * h]), parameter("hc_head_scale", "fp32", [1]), parameter("hc_head_base", "fp32", [c])]
        objects = [allocation("layer_input_result", "bf16", [rows, h], "returned")]
        returned = [{"role": "layer_input", "base": "layer_input_result", "shape": [rows, h]}]
    else:
        raise ValueError("unknown independent mHC wrapper object program")
    if variant == "mhc_pre":
        programs = [{"condition": "always", "stages": ["prenorm", "pre_norm"],
                     "prenorm": "DeepGEMM required: use_tilelang_fallback=False"}]
    elif variant == "mhc_post_pre":
        programs = [{"condition": "fused_config is not None", "stages": ["post_prenorm", "pre_norm"]},
                    {"condition": "fused_config is None", "stages": ["post", "prenorm", "pre_norm"],
                     "prenorm": "DeepGEMM if is_deep_gemm_supported(), else TileLang with source_n_splits=1"}]
    else:
        programs = [{"condition": "always", "stages": [variant]}]
    return {"program": "vllm_0_30_mhc_wrapper_objects", "variant": variant, "parameter_group": group,
            "parameters": parameters, "input_views": views, "all_views_allocate_storage": False,
            "objects": objects, "returned": returned, "return_views_allocate_storage": False,
            "source_programs": programs,
            "mutates_inputs": False, "caller_fp32_input_copy": False, "norm_weight_copy_in_declared_model": False,
            "partial_split_parameter": "source backend/config determines source_n_splits; caller n_splits argument is ignored",
            "broadcast_weight_prepared_before_forward": variant == "mhc_pre", "selected_runtime_backend": None,
            "scope": "source caller tensor objects with formal split count; compiler/library scratch, physical allocator and actual provider separate"}


def attach_mhc_accounts(raw, cases, operators, source_root, *, source_use_deep_gemm=None):
    _validate_prenorm_selection((source_root / "vllm/utils/deep_gemm.py").read_text(encoding="utf-8"),
        (source_root / "vllm/model_executor/kernels/mhc/tilelang.py").read_text(encoding="utf-8"))
    path = source_root / "vllm/model_executor/kernels/mhc/tilelang_kernels.py"
    fused_config = _fused_config(path.read_text(encoding="utf-8"))
    case_map = {case["phase"]: case for case in cases}
    for record in operators:
        variant = record["variant"]
        if variant not in ("mhc_pre", "mhc_post_pre", "mhc_post", "hc_head"):
            continue
        if source_use_deep_gemm is not None and variant in ("mhc_pre", "mhc_post_pre"):
            record["attributes"]["source_use_deep_gemm"] = source_use_deep_gemm
        case = case_map[record["node_id"].split(":", 1)[0]]
        tokens = sum(request["query"] for request in case["requests"])
        config = fused_config(tokens, raw["hidden_size"], raw["hc_mult"])
        if variant == "mhc_post_pre" and config is None:
            # Derive both branches independently. Candidate shape can be
            # checked against a conditional form; it is never evidence for
            # which runtime backend is available or selected.
            record["account_source"] = record["lowering_source"] = "mhc-source"
            record["lowering_account"] = None
            record["mhc_nonfused_forms"] = mhc_nonfused_stage_forms(
                raw, tokens, multiprocessors=132, source_use_deep_gemm=source_use_deep_gemm)
            record["math_account"] = {"mode": "source_stages"}
            record["byte_account"] = {"mode": "source_stages"}
            continue
        if variant == "mhc_pre":
            # Broadcast forces use_tilelang_fallback=False. Derive the
            # split allocation using the declared H200132-SM specification,
            # never a GPU query or assumed provider installation.
            grid = (tokens + 63) // 64
            k_blocks = (raw["hidden_size"] + 63) // 64
            splits, tile = max(min(132 // grid, k_blocks // 4), 1), None
            names = ("mhc_prenorm_broadcast", "mhc_pre_norm_broadcast")
        elif variant == "mhc_post_pre":
            tile, splits, unused_threads = config
            names = ("mhc_post_prenorm", "mhc_pre_norm")
        else:
            splits, tile, names = 1, None, (variant,)
        record["account_source"] = "mhc-source"
        record["lowering_source"] = "mhc-source"
        record["lowering_account"] = []
        for name in names:
            work = mhc_stage_reference(raw, name, tokens, splits=splits, projection_tile=tile)
            shape = {"hidden": raw["hidden_size"], "hc_mult": raw["hc_mult"]}
            if name in ("mhc_prenorm_broadcast", "mhc_pre_norm_broadcast", "mhc_post_prenorm", "mhc_pre_norm"):
                shape["n_splits"] = splits
            if name == "mhc_post_prenorm":
                shape["tile_n"] = tile
            record["lowering_account"].append({"kind": "VECTOR", "variant": name, "dtype": "bf16",
                                                "required_shape": shape, "mhc_stage_work": work})
        record["math_account"] = {"mode": "source_stages"}
        record["byte_account"] = {"mode": "source_stages"}
