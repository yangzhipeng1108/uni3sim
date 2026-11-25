"""Canonical static workload identity for FlashAttention launch facts."""

import json
from typing import Mapping

from src.kernel.launch_config.selection_primitives import (
    require_boolean,
    require_dtype,
    require_non_empty_string,
    require_shape,
    require_variant,
)


def flash_attn_workload_key(
        shape: Mapping[str, int], dtype: str, variant: str,
        attributes: Mapping[str, object]) -> str:
    """Serialize the complete static FlashAttention launch-selection input."""

    required_attributes = {
        "is_decode", "causal", "kv_layout", "kv_dist", "q_lens",
        "kv_lens",
    }
    missing_attributes = required_attributes - set(attributes)
    if missing_attributes:
        raise ValueError(
            "FLASH_ATTN launch workload is missing attributes: "
            f"{sorted(missing_attributes)}")
    batch, q_len, kv_len, head_dim, query_heads, kv_heads = require_shape(
        shape, (
            "batch", "q_len", "kv_len", "head_dim",
            "num_query_heads", "num_kv_heads"))
    normalized_dtype = require_dtype(dtype, ("bf16",))
    execution_variant = require_variant(
        variant, ("prefill", "decode"), "FLASH_ATTN")
    is_decode = require_boolean(attributes, "is_decode")
    if is_decode != (execution_variant == "decode"):
        raise ValueError("FLASH_ATTN is_decode conflicts with variant")
    causal = require_boolean(attributes, "causal")
    kv_layout = require_non_empty_string(attributes, "kv_layout")
    kv_dist = require_non_empty_string(attributes, "kv_dist")

    lengths = {}
    for name, maximum in (("q_lens", q_len), ("kv_lens", kv_len)):
        values = attributes[name]
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"FLASH_ATTN {name} must be a list or tuple")
        if len(values) != batch:
            raise ValueError(
                f"FLASH_ATTN {name} length must equal batch: "
                f"expected={batch}, actual={len(values)}")
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in values):
            raise TypeError(f"FLASH_ATTN {name} must contain integers")
        if any(value <= 0 or value > maximum for value in values):
            raise ValueError(
                f"FLASH_ATTN {name} must be positive and at most {maximum}")
        if max(values) != maximum:
            raise ValueError(
                f"FLASH_ATTN {name} maximum must equal shape maximum "
                f"{maximum}")
        lengths[name] = tuple(values)
    if any(query > key_value for query, key_value in zip(
            lengths["q_lens"], lengths["kv_lens"])):
        raise ValueError("FLASH_ATTN requires every q_lens value <= kv_lens")

    return json.dumps([
        normalized_dtype,
        execution_variant,
        batch,
        q_len,
        kv_len,
        head_dim,
        query_heads,
        kv_heads,
        causal,
        kv_layout,
        kv_dist,
        list(lengths["q_lens"]),
        list(lengths["kv_lens"]),
    ], separators=(",", ":"))


def validate_flash_attn_workload_key(key: str) -> None:
    """Reject non-canonical or malformed serialized workload identities."""

    if not isinstance(key, str) or not key:
        raise TypeError("FLASH_ATTN workload key must be a non-empty string")
    values = json.loads(key)
    if not isinstance(values, list) or len(values) != 13:
        raise ValueError(
            "FLASH_ATTN workload key must contain exactly 13 fields")
    (dtype, variant, batch, q_len, kv_len, head_dim, query_heads,
     kv_heads, causal, kv_layout, kv_dist, q_lens, kv_lens) = values
    rebuilt = flash_attn_workload_key(
        {
            "batch": batch,
            "q_len": q_len,
            "kv_len": kv_len,
            "head_dim": head_dim,
            "num_query_heads": query_heads,
            "num_kv_heads": kv_heads,
        },
        dtype,
        variant,
        {
            "is_decode": variant == "decode",
            "causal": causal,
            "kv_layout": kv_layout,
            "kv_dist": kv_dist,
            "q_lens": q_lens,
            "kv_lens": kv_lens,
        },
    )
    if rebuilt != key:
        raise ValueError("FLASH_ATTN workload key is not canonical")
