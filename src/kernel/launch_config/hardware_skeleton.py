"""Shared launch-selection skeleton; hardware packages bind it with data.

Several selectors the dispatcher resolves per hardware are not
hardware-specific in their logic, only in their data: a backend either runs a
separate setup kernel, an output-buffer fill, a typed Cast or a reduce, or it
does not, and when it does the launch has one shape whose geometry is an
observed fact.  The logic lives here once.  Each hardware package binds it
with its own declaration -- the geometry it observed, the recipe selector it
owns -- so a package states *what* its hardware does rather than restating
*how* to select it.

Binding happens in the hardware package, not by looking up ``hardware_id``
here: a projected hardware (H200 over the A800 package) must keep resolving
exactly the functions its projected package binds.
"""

from typing import Callable, Mapping, Tuple

from src.kernel.launch_config.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import (
    ceil_div,
    require_dtype,
    require_shape,
)


def select_flash_attn_forward_partition(
        shape: Mapping[str, int], variant: str,
        attributes: Mapping[str, object]) -> tuple:
    """A backend that runs one fused attention call per layer partitions none.

    An empty result means the logical call executes whole.
    """

    del shape, variant, attributes
    return ()


def select_flash_attn_setup_launches(
        shape: Mapping[str, int], dtype: str, variant: str,
        attributes: Mapping[str, object], *, hardware) -> tuple:
    """A backend with no separate host-side setup kernel emits none."""

    del shape, dtype, variant, attributes, hardware
    return ()


def normalize_flash_attn_execution_attributes(
        shape: Mapping[str, int],
        attributes: Mapping[str, object]) -> Mapping[str, object]:
    """A backend forwards the public FlashAttention attributes unchanged."""

    del shape
    return dict(attributes)


def select_no_flash_attn_output_fill_launches(
        shape: Mapping[str, int], dtype: str, attributes: Mapping[str, object],
        hardware) -> tuple:
    """A backend that emits no separate output-buffer fill selects none."""

    del shape, dtype, attributes, hardware
    return ()


def flash_attn_output_fill_selector(
        *, values_per_cta: int, threads_per_cta: int) -> Callable:
    """Bind the output-buffer zero-fill to one backend's observed geometry.

    The fill covers the whole output buffer of the call (all actual tokens
    times query heads times head dimension) with one nullary elementwise
    launch.  ``values_per_cta`` and ``threads_per_cta`` are the observed launch
    facts of that backend; no duration participates.
    """

    for name, value in (("values_per_cta", values_per_cta),
                        ("threads_per_cta", threads_per_cta)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"output fill {name} must be a positive integer")

    def select_flash_attn_output_fill_launches(
            shape: Mapping[str, int], dtype: str,
            attributes: Mapping[str, object],
            hardware) -> Tuple[Tuple[str, int, LaunchConfig], ...]:
        del attributes, hardware
        batch, q_len, head_dim, query_heads = require_shape(
            shape, ("batch", "q_len", "head_dim", "num_query_heads"))
        require_dtype(dtype, ("bf16", "fp16"))
        elements = batch * q_len * query_heads * head_dim
        return (
            (dtype, elements, LaunchConfig(
                algorithm="observed_flash_attention_output_fill:elementwise",
                grid=(ceil_div(elements, values_per_cta), 1, 1),
                block=(threads_per_cta, 1, 1),
                execution_variant="elementwise",
            )),
        )

    return select_flash_attn_output_fill_launches


def no_typed_cast_roles_selector(recipe_selector: Callable) -> Callable:
    """Bind "this FusedMoE recipe has no typed Cast" to a package's recipe.

    The package's own recipe is still resolved so a malformed shape fails
    exactly where it did in the hardware-local body.
    """

    def select_fused_moe_cast_roles(
            shape: Mapping[str, int], *, hardware) -> Tuple[str, ...]:
        recipe_selector(shape, hardware=hardware)
        return ()

    return select_fused_moe_cast_roles


def select_no_fused_moe_reduce_work_fractions(
        shape: Mapping[str, int], *, hardware) -> Tuple[float, ...]:
    """A retained FusedMoE recipe with no ``moe_reduce`` invocation.

    The full FusedMoE shape is still required so a malformed request fails.
    """

    require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    del hardware
    return ()
