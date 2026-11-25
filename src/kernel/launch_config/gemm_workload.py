"""Canonical static workload identity and domain derivation for GEMM launch facts."""

import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, NamedTuple

from src.kernel.launch_config.selection_primitives import (
    require_boolean,
    require_dtype,
    require_non_empty_string,
    require_positive_integer,
    require_shape,
    require_variant,
)


class LaunchDomainModel(NamedTuple):
    """One launch-domain Serving model's static deployment identity."""

    model_type: str
    tensor_parallel_size: int
    expert_parallel_size: int
    model_path: str
    model_config_sha256: str


# The static Serving deployment the GEMM workload domain is derived for: each
# model's config semantics, (tensor, expert) parallel width, and canonical
# config provenance, plus the vLLM Scheduler contract the workload keys were
# captured under.  The dataset projection and the A800 launch package both
# need them, so they live with the shared derivation.
LAUNCH_DOMAIN_MODELS = {
    "Qwen3-4B": LaunchDomainModel(
        model_type="qwen3",
        tensor_parallel_size=1,
        expert_parallel_size=1,
        model_path="${MODELS_ROOT}/Qwen3-4B",
        model_config_sha256=(
            "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a"),
    ),
    "Qwen3-8B": LaunchDomainModel(
        model_type="qwen3",
        tensor_parallel_size=1,
        expert_parallel_size=1,
        model_path="${MODELS_ROOT}/Qwen3-8B",
        model_config_sha256=(
            "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30"),
    ),
    "Llama-3.1-8B": LaunchDomainModel(
        model_type="llama",
        tensor_parallel_size=1,
        expert_parallel_size=1,
        model_path="${MODELS_ROOT}/Llama-3.1-8B-Instruct",
        model_config_sha256=(
            "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e"),
    ),
    "Qwen3-32B": LaunchDomainModel(
        model_type="qwen3",
        tensor_parallel_size=2,
        expert_parallel_size=1,
        model_path="${MODELS_ROOT}/Qwen3-32B",
        model_config_sha256=(
            "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb"),
    ),
    "Qwen3-30B-A3B": LaunchDomainModel(
        model_type="qwen3_moe",
        tensor_parallel_size=2,
        expert_parallel_size=2,
        model_path="${MODELS_ROOT}/Qwen3-30B-A3B",
        model_config_sha256=(
            "2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297"),
    ),
}
SERVING_MODEL_PARALLELISM = {
    model_id: (model.tensor_parallel_size, model.expert_parallel_size)
    for model_id, model in LAUNCH_DOMAIN_MODELS.items()
}
SERVING_SCHEDULER_CONTRACT = {
    "backend": "vllm",
    "scheduler_version": "0.11.0",
    "scheduler_source_commit": "b8b302cde434df8c9289a2b465406b47ebab1c2d",
    "max_batched_tokens": 8192,
    "max_requests": 256,
    "enable_chunked_prefill": True,
    "mtp_num_speculative_tokens": 0,
}


GEMM_WORKLOAD_KEY_FIELDS = (
    "dtype",
    "variant",
    "b",
    "m",
    "n",
    "k",
    "a_layout",
    "b_layout",
    "c_layout",
    "trans_a",
    "trans_b",
    "epilogue",
    "has_bias",
    "position",
    "tensor_parallel_size",
    "expert_parallel_size",
)

GEMM_LAUNCH_SELECTION_KEY_FIELDS = GEMM_WORKLOAD_KEY_FIELDS[:13]
GEMM_SEMANTIC_CONTEXT_FIELDS = GEMM_WORKLOAD_KEY_FIELDS[13:]

_LAYOUTS = frozenset({"row_major"})
_EPILOGUES = frozenset({"none", "bias"})
_DENSE_POSITIONS = (
    "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head")
_MOE_POSITIONS = ("qkv_proj", "o_proj", "router", "lm_head")
_TOKEN_POSITIONS = frozenset({
    "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "router",
})
_CUBLASLT_STAGES_PATTERN = re.compile(r"(8|16|32|64|128)x([1-9]|10|AUTO)")
_CUBLASLT_STAGE_DEPTHS = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "AUTO": None,
}


def parse_cublaslt_stages(stages: str) -> tuple[int, int | None]:
    """Project a documented cuBLASLt stages enum into K/depth semantics."""

    if not isinstance(stages, str):
        raise TypeError("cuBLASLt stages must be a string")
    match = _CUBLASLT_STAGES_PATTERN.fullmatch(stages)
    if match is None:
        raise ValueError(
            "cuBLASLt stages must match "
            "^(8|16|32|64|128)x(1..10|AUTO)$")
    tile_k = int(match.group(1))
    depth_text = match.group(2)
    return tile_k, _CUBLASLT_STAGE_DEPTHS[depth_text]


def analytical_cublas_tiling(rows: int, n: int, k: int) -> tuple[int, int, int, int]:
    """Return the analytical cuBLAS tile (tile_m, tile_n, tile_k, depth).

    This is the tiler the projected analytical launch policy uses for one
    GEMM problem of ``rows`` output rows when no launch was captured; a
    strided-batched GEMM applies it to each batch problem.
    """

    tile_m = 32 if rows <= 32 else (64 if rows <= 64 else 128)
    tile_n = 64 if n <= 64 else 128
    tile_k = 32 if k <= 32 else 64
    depth = 2 if rows <= 16 else 3
    return tile_m, tile_n, tile_k, depth


def cublaslt_main_block_threads(tile_m: int, tile_n: int) -> int:
    """Return the SM80 thread domain encoded by the selected tile class.

    CUDA 12.2's captured rank-zero launches use one warp for the 16x16 WMMA
    tile, eight warps for the two 32-Ki-element CTA tiles, and four warps for
    every other retained tile.  This is part of the physical launch program:
    it depends only on the selected logical tile, never on the workload shape
    or its origin.
    """

    threads_by_tile = {
        (16, 16): 32,
        (32, 32): 128,
        (32, 64): 128,
        (32, 128): 128,
        (64, 64): 128,
        (64, 128): 128,
        (64, 256): 128,
        (128, 64): 128,
        (128, 128): 128,
        (128, 256): 256,
        (256, 128): 256,
    }
    tile = (tile_m, tile_n)
    if tile not in threads_by_tile:
        raise ValueError(f"unsupported cuBLASLt logical tile {tile!r}")
    return threads_by_tile[tile]


def reject_timing_evidence_fields(value: object, context: str) -> None:
    """Reject timing evidence while accepting explicit false consumption flags."""

    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{context} contains a non-string field")
            normalized_key = key.lower()
            forbidden_token = next((
                token for token in ("duration", "cycle", "timing")
                if token in normalized_key
            ), None)
            non_consumption_flag = (
                normalized_key in {
                    "duration_consumed", "cycle_consumed", "timing_consumed",
                }
                and item is False
            )
            if forbidden_token is not None and not non_consumption_flag:
                raise ValueError(
                    f"{context} contains forbidden {forbidden_token} field "
                    f"{key!r}")
            reject_timing_evidence_fields(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_timing_evidence_fields(item, f"{context}[{index}]")


def gemm_workload_key(
        shape: Mapping[str, int], dtype: str, variant: str,
        attributes: Mapping[str, object]) -> str:
    """Serialize the complete static GEMM launch-selection input."""

    required_attributes = {
        "a_layout", "b_layout", "c_layout", "trans_a", "trans_b",
        "epilogue", "has_bias", "position", "tensor_parallel_size",
        "expert_parallel_size",
    }
    missing_attributes = required_attributes - set(attributes)
    if missing_attributes:
        raise ValueError(
            "GEMM launch workload is missing attributes: "
            f"{sorted(missing_attributes)}")
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    normalized_dtype = require_dtype(dtype, ("bf16", "fp16", "fp32"))
    execution_variant = require_variant(
        variant, ("gemm", "gemv", "batched_gemm"), "GEMM")
    layouts = tuple(
        require_non_empty_string(attributes, name)
        for name in ("a_layout", "b_layout", "c_layout")
    )
    if any(layout not in _LAYOUTS for layout in layouts):
        raise ValueError(
            f"unsupported GEMM layouts {layouts!r}; expected row_major")
    trans_a = require_boolean(attributes, "trans_a")
    trans_b = require_boolean(attributes, "trans_b")
    epilogue = require_non_empty_string(attributes, "epilogue")
    if epilogue not in _EPILOGUES:
        raise ValueError(
            f"unsupported GEMM epilogue {epilogue!r}; expected "
            f"{sorted(_EPILOGUES)}")
    has_bias = require_boolean(attributes, "has_bias")
    if has_bias != (epilogue == "bias"):
        raise ValueError("GEMM has_bias conflicts with epilogue")
    position = require_non_empty_string(attributes, "position")
    if position == "generic":
        raise ValueError("GEMM position must identify the logical operation")
    tensor_parallel_size = require_positive_integer(
        attributes, "tensor_parallel_size")
    expert_parallel_size = require_positive_integer(
        attributes, "expert_parallel_size")

    return json.dumps([
        normalized_dtype,
        execution_variant,
        b,
        m,
        n,
        k,
        layouts[0],
        layouts[1],
        layouts[2],
        trans_a,
        trans_b,
        epilogue,
        has_bias,
        position,
        tensor_parallel_size,
        expert_parallel_size,
    ], separators=(",", ":"))


def validate_gemm_workload_key(key: str) -> None:
    """Reject non-canonical or malformed serialized workload identities."""

    if not isinstance(key, str) or not key:
        raise TypeError("GEMM workload key must be a non-empty string")
    values = json.loads(key)
    if not isinstance(values, list) or len(values) != len(
            GEMM_WORKLOAD_KEY_FIELDS):
        raise ValueError(
            "GEMM workload key must contain exactly "
            f"{len(GEMM_WORKLOAD_KEY_FIELDS)} fields")
    (dtype, variant, b, m, n, k, a_layout, b_layout, c_layout, trans_a,
     trans_b, epilogue, has_bias, position, tensor_parallel_size,
     expert_parallel_size) = values
    rebuilt = gemm_workload_key(
        {"b": b, "m": m, "n": n, "k": k},
        dtype,
        variant,
        {
            "a_layout": a_layout,
            "b_layout": b_layout,
            "c_layout": c_layout,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "epilogue": epilogue,
            "has_bias": has_bias,
            "position": position,
            "tensor_parallel_size": tensor_parallel_size,
            "expert_parallel_size": expert_parallel_size,
        },
    )
    if rebuilt != key:
        raise ValueError("GEMM workload key is not canonical")


def gemm_launch_selection_key(
        shape: Mapping[str, int], dtype: str, variant: str,
        attributes: Mapping[str, object]) -> tuple[object, ...]:
    """Return the physical GEMM selector identity without semantic context."""

    required_attributes = {
        "a_layout", "b_layout", "c_layout", "trans_a", "trans_b",
        "epilogue", "has_bias",
    }
    missing_attributes = required_attributes - set(attributes)
    if missing_attributes:
        raise ValueError(
            "GEMM launch selection is missing attributes: "
            f"{sorted(missing_attributes)}")
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    normalized_dtype = require_dtype(dtype, ("bf16", "fp16", "fp32"))
    execution_variant = require_variant(
        variant, ("gemm", "gemv", "batched_gemm"), "GEMM")
    layouts = tuple(
        require_non_empty_string(attributes, name)
        for name in ("a_layout", "b_layout", "c_layout")
    )
    if any(layout not in _LAYOUTS for layout in layouts):
        raise ValueError(
            f"unsupported GEMM layouts {layouts!r}; expected row_major")
    trans_a = require_boolean(attributes, "trans_a")
    trans_b = require_boolean(attributes, "trans_b")
    epilogue = require_non_empty_string(attributes, "epilogue")
    if epilogue not in _EPILOGUES:
        raise ValueError(
            f"unsupported GEMM epilogue {epilogue!r}; expected "
            f"{sorted(_EPILOGUES)}")
    has_bias = require_boolean(attributes, "has_bias")
    if has_bias != (epilogue == "bias"):
        raise ValueError("GEMM has_bias conflicts with epilogue")
    return (
        normalized_dtype, execution_variant, b, m, n, k,
        layouts[0], layouts[1], layouts[2], trans_a, trans_b, epilogue,
        has_bias,
    )


def gemm_launch_selection_key_from_workload_key(
        workload_key: str) -> tuple[object, ...]:
    """Project one validated semantic workload key onto physical fields."""

    validate_gemm_workload_key(workload_key)
    values = json.loads(workload_key)
    return tuple(values[:len(GEMM_LAUNCH_SELECTION_KEY_FIELDS)])


def gemm_physical_operation_key(workload_key: str) -> str:
    """Project a canonical semantic key to the selector's physical identity."""

    validate_gemm_workload_key(workload_key)
    values = json.loads(workload_key)
    return json.dumps([
        values[3], values[4], values[5], values[0], values[12],
    ], separators=(",", ":"))


def _model_positions(
        model: Mapping[str, object], model_id: str) -> tuple[str, ...]:
    if model["num_experts"] == 0:
        if model["model_type"] not in ("qwen3", "llama"):
            raise ValueError(
                f"model {model_id} has unsupported dense model_type "
                f"{model['model_type']!r}")
        return _DENSE_POSITIONS
    if model["model_type"] == "qwen3_moe":
        return _MOE_POSITIONS
    raise ValueError(
        f"model {model_id} has unsupported MoE model_type "
        f"{model['model_type']!r}")


def _model_position_bias(model: Mapping[str, object], position: str) -> bool:
    if position in ("qkv_proj", "o_proj"):
        return model["attention_bias"]
    if position in ("gate_up_proj", "down_proj"):
        return model["mlp_bias"]
    if position in ("router", "lm_head"):
        return False
    raise ValueError(f"unsupported GEMM position {position!r}")


def _config_positive_integer(
        config: Mapping[str, object], field: str) -> int:
    value = config[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"model config {field} must be an integer")
    if value <= 0:
        raise ValueError(f"model config {field} must be positive")
    return value


def _load_launch_model_semantics(
        model_path: str, model_id: str) -> dict[str, object]:
    config_path = Path(model_path) / "config.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if not isinstance(config, dict):
        raise TypeError("model config root must be a mapping")
    model_type = config["model_type"]
    if model_id not in LAUNCH_DOMAIN_MODELS:
        raise ValueError(f"unsupported launch-domain model {model_id!r}")
    expected_type = LAUNCH_DOMAIN_MODELS[model_id].model_type
    if model_type != expected_type:
        raise ValueError(
            f"model {model_id} must have model_type "
            f"{expected_type!r}, got {model_type!r}")
    architectures = config["architectures"]
    expected_architectures = {
        "qwen3": ["Qwen3ForCausalLM"],
        "llama": ["LlamaForCausalLM"],
        "qwen3_moe": ["Qwen3MoeForCausalLM"],
    }
    if architectures != expected_architectures[model_type]:
        raise ValueError(
            f"model {model_id} architecture differs from its launch contract")
    dtype_fields = [
        field for field in ("dtype", "torch_dtype") if field in config]
    if not dtype_fields:
        raise ValueError("model config requires dtype or torch_dtype")
    dtype_names = {config[field] for field in dtype_fields}
    if dtype_names != {"bfloat16"}:
        raise ValueError(
            f"model {model_id} launch domain requires bfloat16, got "
            f"{sorted(dtype_names)}")
    hidden_size = _config_positive_integer(config, "hidden_size")
    intermediate_size = _config_positive_integer(config, "intermediate_size")
    num_attention_heads = _config_positive_integer(
        config, "num_attention_heads")
    num_key_value_heads = _config_positive_integer(
        config, "num_key_value_heads")
    vocab_size = _config_positive_integer(config, "vocab_size")
    if model_type == "llama":
        if hidden_size % num_attention_heads:
            raise ValueError(
                "Llama hidden_size must be divisible by num_attention_heads")
        head_dim = hidden_size // num_attention_heads
    elif model_type in ("qwen3", "qwen3_moe"):
        head_dim = _config_positive_integer(config, "head_dim")
    else:
        raise AssertionError(f"unhandled model_type {model_type!r}")
    if num_attention_heads % num_key_value_heads:
        raise ValueError(
            "num_attention_heads must be divisible by num_key_value_heads")
    attention_bias = config["attention_bias"]
    if not isinstance(attention_bias, bool):
        raise TypeError("model attention_bias must be boolean")
    if model_type in ("qwen3", "qwen3_moe"):
        # Qwen3's MLP projection bias is fixed off by the architecture and is
        # not represented in its canonical Hugging Face config.  Normalize
        # that explicit model-type contract at this input boundary while
        # rejecting a contradictory optional field when one is supplied.
        if "mlp_bias" in config and config["mlp_bias"] is not False:
            raise ValueError("Qwen3 model config cannot enable mlp_bias")
        mlp_bias = False
    elif model_type == "llama":
        mlp_bias = config["mlp_bias"]
        if not isinstance(mlp_bias, bool):
            raise TypeError("Llama model mlp_bias must be boolean")
    else:
        raise AssertionError(f"unhandled model_type {model_type!r}")
    if model_type == "qwen3_moe":
        num_experts = _config_positive_integer(config, "num_experts")
    elif model_type in ("qwen3", "llama"):
        num_experts = 0
    else:
        raise AssertionError(f"unhandled model_type {model_type!r}")
    return {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "compute_dtype": "bf16",
        "weight_dtype": "bf16",
        "model_type": model_type,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "num_experts": num_experts,
        "attention_bias": attention_bias,
        "mlp_bias": mlp_bias,
    }


def _launch_gemm_dimensions(
        model: Mapping[str, object], position: str,
        tensor_parallel_size: int) -> tuple[int, int]:
    hidden_size = model["hidden_size"]
    intermediate_size = model["intermediate_size"]
    num_attention_heads = model["num_attention_heads"]
    num_key_value_heads = model["num_key_value_heads"]
    head_dim = model["head_dim"]
    if position == "qkv_proj":
        if num_attention_heads % tensor_parallel_size:
            raise ValueError(
                "qkv_proj query heads must be divisible by tensor parallel size")
        local_query_heads = num_attention_heads // tensor_parallel_size
        if (num_key_value_heads >= tensor_parallel_size
                and num_key_value_heads % tensor_parallel_size == 0):
            local_kv_heads = num_key_value_heads // tensor_parallel_size
        elif (num_key_value_heads < tensor_parallel_size
              and tensor_parallel_size % num_key_value_heads == 0):
            local_kv_heads = 1
        else:
            raise ValueError(
                "KV heads and tensor parallel size must divide exactly")
        return (
            (local_query_heads + 2 * local_kv_heads) * head_dim,
            hidden_size,
        )
    if position == "o_proj":
        total_k = num_attention_heads * head_dim
        if total_k % tensor_parallel_size:
            raise ValueError("o_proj K must be divisible by tensor parallel size")
        return hidden_size, total_k // tensor_parallel_size
    if position == "gate_up_proj":
        total_n = 2 * intermediate_size
        if total_n % tensor_parallel_size:
            raise ValueError(
                "gate_up_proj N must be divisible by tensor parallel size")
        return total_n // tensor_parallel_size, hidden_size
    if position == "down_proj":
        if intermediate_size % tensor_parallel_size:
            raise ValueError(
                "down_proj K must be divisible by tensor parallel size")
        return hidden_size, intermediate_size // tensor_parallel_size
    if position == "lm_head":
        padding = 64
        padded_vocab = (
            (model["vocab_size"] + padding - 1) // padding * padding)
        if padded_vocab % tensor_parallel_size:
            raise ValueError(
                "padded lm_head vocabulary must be divisible by tensor "
                "parallel size")
        return padded_vocab // tensor_parallel_size, hidden_size
    if position == "router":
        if model["num_experts"] <= 0:
            raise ValueError("router position requires a MoE model")
        return model["num_experts"], hidden_size
    raise ValueError(f"unsupported GEMM position {position!r}")


def derive_model_gemm_workload_domain(
        model_path: str, model_id: str, *, tensor_parallel_size: int,
        expert_parallel_size: int, max_batched_tokens: int,
        max_requests: int) -> dict[str, object]:
    """Derive one model's complete static launch-selector GEMM key domain."""

    for value, context in (
            (tensor_parallel_size, "tensor_parallel_size"),
            (expert_parallel_size, "expert_parallel_size"),
            (max_batched_tokens, "max_batched_tokens"),
            (max_requests, "max_requests")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{context} must be an integer")
        if value <= 0:
            raise ValueError(f"{context} must be positive")
    model = _load_launch_model_semantics(model_path, model_id)
    positions = _model_positions(model, model_id)
    keys = []
    physical_operations = set()
    position_domains = []
    for position in positions:
        if position in _TOKEN_POSITIONS:
            maximum_m = max_batched_tokens
            variant_contract = "gemm"
        elif position == "lm_head":
            maximum_m = max_requests
            variant_contract = "m1_gemv_else_gemm"
        else:
            raise AssertionError(f"unhandled GEMM position {position!r}")
        n, k = _launch_gemm_dimensions(
            model, position, tensor_parallel_size)
        has_bias = _model_position_bias(model, position)
        for m in range(1, maximum_m + 1):
            if position == "lm_head" and m == 1:
                variant = "gemv"
            else:
                variant = "gemm"
            key = gemm_workload_key(
                {"b": 1, "m": m, "n": n, "k": k},
                model["compute_dtype"],
                variant,
                {
                    "a_layout": "row_major",
                    "b_layout": "row_major",
                    "c_layout": "row_major",
                    "trans_a": False,
                    "trans_b": True,
                    "epilogue": "bias" if has_bias else "none",
                    "has_bias": has_bias,
                    "position": position,
                    "tensor_parallel_size": tensor_parallel_size,
                    "expert_parallel_size": expert_parallel_size,
                },
            )
            keys.append(key)
            physical_operations.add((
                m, n, k, model["compute_dtype"], has_bias))
        position_domains.append({
            "position": position,
            "m_min": 1,
            "m_max": maximum_m,
            "n": n,
            "k": k,
            "dtype": model["compute_dtype"],
            "has_bias": has_bias,
            "variant_contract": variant_contract,
            "semantic_key_count": maximum_m,
        })
    return {
        "config_sha256": model["config_sha256"],
        "compute_dtype": model["compute_dtype"],
        "weight_dtype": model["weight_dtype"],
        "model_type": model["model_type"],
        "workload_keys": tuple(keys),
        "physical_operations": physical_operations,
        "positions": position_domains,
    }
