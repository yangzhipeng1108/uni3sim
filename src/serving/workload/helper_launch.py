"""Helper/glue launch emission contract support for the workload builder.

The three call points in ``docs/serving_system.md`` §11 emit
launches whose COUNT comes from a vLLM source predicate or, where the vendor
dispatch layer measurably diverges from that predicate, from the versioned
discrete emission observation validated against the canonical alignment
traces.  Nothing here carries a duration: emission only declares that a
launch happens, and pricing reuses the existing kernel families.

Fail-closed rules (docs/serving_system.md §11):
- The observation file validates exactly.  Unknown top-level fields, unknown
  item fields, unknown key fields, or a non-integer count raise on load.
- Call points in ``OBSERVATION_REQUIRED`` must resolve from the observation;
  a missing entry raises instead of silently falling back to the
  source-predicate count, so a deleted registry cannot quietly undo a
  documented divergence.
- A resolved count of ``0`` is legal only because the observation entry
  carries an explicit authorization note; that note is what separates
  "measured zero" from "nobody wrote this branch".
"""

from __future__ import annotations

from pathlib import Path

import yaml

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OBSERVATION_PATH = (
    _REPOSITORY_ROOT / "configs" / "validation" / "serving"
    / "helper_emission_observation.yaml")

_SCHEMA_VERSION = 1
_ITEM_FIELDS = frozenset({
    "key", "observed_count", "authorization", "evidence"})
_KEY_FIELDS = frozenset({"hardware_id", "call_point", "phase", "model_type"})

# Call points whose count must come from the versioned observation because
# the vendor dispatch layer diverges from the vLLM source predicate.  A
# divergence is consumed from the registry, never re-derived ad hoc.
_OBSERVATION_REQUIRED = frozenset({
    ("nvidia_a800", "token_embedding_gather", "prefill"),
})

# docs/serving_system.md §11 call point 3 (model output boundary):
# ``sample_hidden_states =
# hidden_states[logits_indices]`` gathers one row per target logit row before
# the lm_head projection (nvidia_a800 vllm/v1/worker/gpu_model_runner.py:2348,
# metax_c500 vllm/v1/worker/gpu_model_runner.py:4372,
# ascend_910b vllm_ascend/worker/model_runner_v1.py:1405).
BOUNDARY_LOGITS_GATHER_SOURCES = {
    "nvidia_a800": "vllm.v1.worker.gpu_model_runner.sample_hidden_states",
    "metax_c500": "vllm.v1.worker.gpu_model_runner.sample_hidden_states",
    "ascend_910b": "vllm_ascend.worker.model_runner_v1.sample_hidden_states",
    "nvidia_h200": "vllm.v1.worker.gpu_model_runner.sample_hidden_states",
}

# docs/serving_system.md §11 call point 2 (MoE routing mask): per-card
# emission law for the routing-mask helper
# launches.  The vLLM source predicate (``expert_map is not None``) is a
# binary fact equivalent to ``enable_expert_parallel``; the per-card emission
# set records WHERE each call point is modelled:
# - nvidia_a800 emits the expert-map index lookup here because the A800
#   FUSED_MOE recipe (select_fused_moe_recipe) contains no expert-map
#   variant.  If that recipe ever gains one, this workload-side emission must
#   be removed or the call point would be double counted.
# - metax_c500 emits nothing here: the C500 FUSED_MOE recipe already emits
#   ``moe_expert_map_lookup`` for the same source line, so a second emission
#   would double count.
# - ascend_910b emits all four kernels of the 4-kernel mask string
#   (Index→NotEqual→Cast→Mul, token_dispatcher.py:344-348).  NotEqual and
#   Cast carry their own call points since the CANN elementwise domain
#   gained a source-pinned compare/conversion pricing family: the emission
#   predicate stays the vLLM source predicate, while the launch count of
#   complete mask strings is additionally emitted as the router run
#   contract.  Keeping the four call points in one emission set preserves
#   the single-source rule: adding one without the others would double
#   count the string.
MOE_ROUTING_MASK_EMISSIONS = {
    "nvidia_a800": ("moe_expert_map_index",),
    "metax_c500": (),
    "ascend_910b": (
        "router_mask_index",
        "router_mask_notequal",
        "router_mask_cast",
        "router_mask_mul"),
    # H200 projects the A800 FUSED_MOE recipe (no expert-map variant), so it
    # emits the expert-map index lookup here exactly as A800 does.
    "nvidia_h200": ("moe_expert_map_index",),
}

ROUTER_MASK_SOURCES = {
    "nvidia_a800": (
        "vllm.model_executor.layers.fused_moe.moe_align_block_size"),
    "metax_c500": (
        "vllm.model_executor.layers.fused_moe.moe_align_block_size"),
    "ascend_910b": "vllm_ascend.ops.fused_moe.token_dispatcher",
    "nvidia_h200": (
        "vllm.model_executor.layers.fused_moe.moe_align_block_size"),
}

# The 910B trace-side counter observes the router mask as one contiguous
# four-kernel string (Index -> NotEqual -> Cast -> Mul) per MoE layer.  A
# card whose emission set contains all four call points additionally emits
# that string count as the router run contract, stamped on the ROUTER node
# by the workload builder.  The set membership is structural, so a card that
# emits only part of the string can never claim a run count.
ROUTER_MASK_STRING_CALL_POINTS = frozenset({
    "router_mask_index",
    "router_mask_notequal",
    "router_mask_cast",
    "router_mask_mul",
})
ROUTER_MASK_RUN_CLASS = "class2_router_run"

# docs/serving_system.md §11 call point 3 (model input boundary): the
# scheduler-step input token lookup
# (nvidia_a800 vllm/v1/worker/gpu_model_runner.py:2330 prepares
# logits_indices on the same runner path that embeds the step input).
TOKEN_EMBEDDING_SOURCES = {
    "nvidia_a800": "vllm.v1.worker.gpu_model_runner",
    "metax_c500": "vllm.v1.worker.gpu_model_runner",
    "ascend_910b": "vllm_ascend.worker.model_runner_v1",
    "nvidia_h200": "vllm.v1.worker.gpu_model_runner",
}

# Marker stamped on the first AF residual transfer when the step-head input
# embedding is suppressed by the versioned emission observation: the residual
# buffer then originates in an earlier capture window and the transfer
# anchors its ready edge on the source pool's last in-window producer.  The
# lowering-side AF boundary validation accepts that anchor only with this
# marker, so a graph that lost its embedding producer for any other reason
# still fails closed.
RESIDUAL_ANCHOR_OBSERVATION_SUPPRESSED = "observation_suppressed_step_input"

_observation_cache: dict | None = None


def _validated_observation(document) -> dict:
    if not isinstance(document, dict):
        raise ValueError(
            "helper emission observation must be a mapping at top level")
    if set(document) != {"version", "items"}:
        raise ValueError(
            "helper emission observation top-level fields must be exactly "
            f"{{version, items}}; found {sorted(document)}")
    if document["version"] != _SCHEMA_VERSION:
        raise ValueError(
            "unsupported helper emission observation schema version: "
            f"{document['version']!r}")
    if not isinstance(document["items"], list):
        raise ValueError("helper emission observation items must be a list")
    table = {}
    for item in document["items"]:
        if not isinstance(item, dict) or set(item) != _ITEM_FIELDS:
            raise ValueError(
                "helper emission observation item fields must be exactly "
                f"{sorted(_ITEM_FIELDS)}; found {sorted(item)}")
        key = item["key"]
        if not isinstance(key, dict) or set(key) != _KEY_FIELDS:
            raise ValueError(
                "helper emission observation item key fields must be exactly "
                f"{sorted(_KEY_FIELDS)}; found {sorted(key)}")
        for field in _KEY_FIELDS:
            if not isinstance(key[field], str) or not key[field]:
                raise ValueError(
                    f"helper emission observation key field {field!r} must "
                    "be a non-empty string")
        if key["phase"] not in ("prefill", "mixed", "decode"):
            raise ValueError(
                f"unknown helper emission observation phase: "
                f"{key['phase']!r}")
        count = item["observed_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                "helper emission observed_count must be a non-negative "
                f"integer; found {count!r}")
        if not isinstance(item["authorization"], str) or not (
                item["authorization"].strip()):
            raise ValueError(
                "helper emission observation authorization must be a "
                "non-empty string")
        scope_key = (key["hardware_id"], key["call_point"], key["phase"])
        model_type = key["model_type"]
        if scope_key not in table:
            table[scope_key] = {}
        if model_type in table[scope_key]:
            raise ValueError(
                "duplicate helper emission observation item: "
                f"{scope_key} model_type {model_type!r}")
        table[scope_key][model_type] = item
    return table


def _load_observation() -> dict:
    global _observation_cache
    if _observation_cache is None:
        if not OBSERVATION_PATH.is_file():
            raise ValueError(
                f"helper emission observation not found: {OBSERVATION_PATH}")
        with open(OBSERVATION_PATH, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        _observation_cache = _validated_observation(document)
    return _observation_cache


def helper_emission_count(hardware_id, call_point, phase, source_count,
                          model_type):
    """Resolve one helper call-point count.

    ``source_count`` is the count the vLLM source predicate yields.  When the
    versioned observation records a divergence for this (hardware, call
    point, phase, model type), the observed count wins and the entry's
    authorization is what licenses it; call points declared in
    ``OBSERVATION_REQUIRED`` must have such an entry.  ``model_type`` is the
    versioned model config discriminator (e.g. ``qwen3`` vs ``qwen3_moe``):
    an entry scoped to one model type overrides only that model type, an
    ``any``-scoped entry overrides every model type, and a concurrent
    ``any`` plus model-scoped entry for one call point is ambiguous, so the
    lookup rejects it.  A model type with no matching entry keeps the source
    predicate count — the registered divergence simply does not apply to it.
    """

    if hardware_id not in TOKEN_EMBEDDING_SOURCES:
        raise ValueError(f"unknown hardware id: {hardware_id!r}")
    if phase not in ("prefill", "mixed", "decode"):
        raise ValueError(f"unknown emission phase: {phase!r}")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError(
            "helper emission model_type must be a non-empty string")
    if isinstance(source_count, bool) or not isinstance(source_count, int) \
            or source_count < 0:
        raise ValueError(
            f"source predicate count must be a non-negative integer; found "
            f"{source_count!r}")
    table = _load_observation()
    scope_key = (hardware_id, call_point, phase)
    if scope_key not in table:
        if scope_key in _OBSERVATION_REQUIRED:
            raise ValueError(
                "helper emission call point requires a versioned "
                f"observation entry but none is registered: {scope_key}")
        return source_count
    scoped = table[scope_key]
    if "any" in scoped and model_type in scoped:
        raise ValueError(
            "ambiguous helper emission observation: both an any-scope and a "
            f"model-scoped entry exist for {scope_key}")
    if model_type in scoped:
        return scoped[model_type]["observed_count"]
    if "any" in scoped:
        return scoped["any"]["observed_count"]
    return source_count


def boundary_logits_gather_source(hardware_id) -> str:
    if hardware_id not in BOUNDARY_LOGITS_GATHER_SOURCES:
        raise ValueError(f"unknown hardware id: {hardware_id!r}")
    return BOUNDARY_LOGITS_GATHER_SOURCES[hardware_id]


def moe_routing_mask_emissions(hardware_id) -> tuple[str, ...]:
    """§11 call point 2 (MoE routing mask) this card emits from the workload layer."""

    if hardware_id not in MOE_ROUTING_MASK_EMISSIONS:
        raise ValueError(f"unknown hardware id: {hardware_id!r}")
    return MOE_ROUTING_MASK_EMISSIONS[hardware_id]


def moe_routing_mask_string_complete(hardware_id) -> bool:
    """Whether this card's mask emission set forms the whole 4-kernel string."""

    if hardware_id not in MOE_ROUTING_MASK_EMISSIONS:
        raise ValueError(f"unknown hardware id: {hardware_id!r}")
    return ROUTER_MASK_STRING_CALL_POINTS <= frozenset(
        MOE_ROUTING_MASK_EMISSIONS[hardware_id])


def router_mask_source(hardware_id) -> str:
    if hardware_id not in ROUTER_MASK_SOURCES:
        raise ValueError(f"unknown hardware id: {hardware_id!r}")
    return ROUTER_MASK_SOURCES[hardware_id]
