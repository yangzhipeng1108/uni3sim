"""Deterministic partial-order checks for production kernel predictions.

This evaluator deliberately consumes the same launch selector and public
``predict`` entry point as Serving.  It does not contain a predictor formula,
duration observation, or correction.  A sequence changes one explicit
workload axis while retaining the physical controls recorded in its manifest.
"""

from __future__ import annotations

from evaluation.json_shape import file_sha256

import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, is_dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from evaluation.kernel.dataset import (
    get_hardware_config_path,
    load_kernel_dataset,
    parse_kernel_evaluation_timing,
)
from src.kernel.fused_moe_physical_specs import build_fused_moe_physical_specs
from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import (
    KernelKind,
    KernelSpec,
    PAGED_KV_CACHE_ATTRIBUTES,
)
from src.kernel.launch_config.gemm_launch_selection import (
    gemm_launch_software_revision,
    load_gemm_launch_selection,
)
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.serving.workload.model_config import ModelConfig, load_model_config
from src.serving.workload.shape_inference import infer_gemm_dimensions
from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HARDWARE_IDS = EVALUATED_HARDWARE_IDS
FAMILIES = ("gemm", "flash_attn", "fused_moe", "unfused_moe", "vector")
DEFAULT_EPSILON_NS = 1e-9
_BASE_EXTENTS = (
    1, 2, 3, 4, 8, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257,
    512, 1024, 2048, 4096,
)
_SMOKE_EXTENTS = (1, 2, 16, 17, 64, 65, 128)
_GEMM_SELECTION_ATTRIBUTES = {
    "a_layout": "row_major",
    "b_layout": "row_major",
    "c_layout": "row_major",
    "trans_a": False,
    "trans_b": True,
    "epilogue": "none",
    "has_bias": False,
}


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=root, check=True,
        capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _json_value(value: Any) -> Any:
    """Serialize frozen production contracts without weakening their schema."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(name): _json_value(item)
            for name, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if hasattr(value, "value"):
        return _json_value(value.value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _launch_summary(launch) -> Mapping[str, Any]:
    return {
        "algorithm": launch.algorithm,
        "grid": list(launch.grid),
        "grid_size": launch.grid_size,
        "block": list(launch.block),
        "block_size": launch.block_size,
        "tiling": None if launch.tiling is None else {
            "tile_shape": dict(launch.tiling.tile_shape),
            "pipeline_depth": launch.tiling.pipeline_depth,
            "split_k": launch.tiling.split_k,
            "num_kv_splits": launch.tiling.num_kv_splits,
        },
        "cluster": None if launch.cluster is None else list(launch.cluster),
        "parameters": _json_value(launch.parameters),
        "execution_variant": launch.execution_variant,
        "source_program_type": (
            None if launch.source_program is None
            else type(launch.source_program).__name__),
        "source_program": (
            None if launch.source_program is None
            else _json_value(launch.source_program)),
    }


def _prediction_summary(prediction) -> Mapping[str, Any]:
    diagnostics = prediction.diagnostics
    if diagnostics is None:
        raise RuntimeError("order checking requires Forward diagnostics")
    memory = []
    for path in diagnostics.memory:
        memory.append({
            "path_name": path.path_name,
            "initial_transactions": path.initial_transactions,
            "steady_time_ns": path.steady_time_s * 1e9,
            "nodes": [
                {
                    "resource_role": node.resource_role,
                    "transactions": node.transactions,
                    "bytes": node.bytes,
                    "capacity_time_ns": node.capacity_time_s * 1e9,
                }
                for node in path.nodes
            ],
        })
    pipeline = None
    if diagnostics.pipeline is not None:
        pipeline = {
            "execution_time_ns": diagnostics.pipeline.execution_time_s * 1e9,
            "initiation_interval_ns": (
                diagnostics.pipeline.initiation_interval_s * 1e9),
            "stage_service_ns": {
                name: value * 1e9
                for name, value in diagnostics.pipeline.stage_service_s.items()
            },
            "critical_path": list(diagnostics.pipeline.critical_path),
            "source_schedule": [
                {
                    "task_name": task.task_name,
                    "worker": task.worker,
                    "repeat_count": task.repeat_count,
                }
                for task in diagnostics.pipeline.tasks
            ],
        }
    return {
        "total_ns": prediction.t_total_ns,
        "launch_dispatch_ns": prediction.t_launch_s * 1e9,
        "bottleneck_group": prediction.bottleneck_group,
        "time_per_group_ns": {
            name: value * 1e9
            for name, value in prediction.t_per_group.items()
        },
        "time_per_resource_ns": {
            name: value * 1e9
            for name, value in prediction.t_per_resource.items()
        },
        "compute_time_by_role_ns": {
            name: value * 1e9
            for name, value in diagnostics.compute_time_s.items()
        },
        "memory": memory,
        "pipeline": pipeline,
        "critical_path": list(diagnostics.critical_path),
        "support_level": diagnostics.support_level,
        "limitations": list(diagnostics.limitations),
    }


def _walk_json(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _model_path_by_sha(root: Path) -> Mapping[str, Path]:
    result = {}
    for path in sorted((root / "configs/model").glob("*.json")):
        digest = file_sha256(path)
        if digest in result:
            raise ValueError(
                "two active Serving configs have the same config SHA256: "
                f"{path} and {result[digest]}")
        result[digest] = path
    return result


def discover_serving_models(root: Path) -> tuple[Mapping[str, Any], ...]:
    """Discover configs and deployed TP/EP facts from active config owners."""

    by_sha = _model_path_by_sha(root)
    deployments: dict[str, dict[tuple[int, int], set[str]]] = {
        digest: {} for digest in by_sha
    }
    source_files = sorted((root / "configs/evaluation/experiments").glob("*.json"))
    for source in source_files:
        with source.open(encoding="utf-8") as stream:
            document = json.load(stream)
        for mapping in _walk_json(document):
            required = {
                "model_config_sha256", "tensor_parallel_size",
                "expert_parallel_size",
            }
            if not required <= set(mapping):
                continue
            digest = mapping["model_config_sha256"]
            if digest not in by_sha:
                continue
            tp = mapping["tensor_parallel_size"]
            ep = mapping["expert_parallel_size"]
            if (isinstance(tp, bool) or not isinstance(tp, int) or tp <= 0
                    or isinstance(ep, bool) or not isinstance(ep, int) or ep <= 0):
                raise ValueError(
                    f"invalid active deployment parallelism in {source}")
            parallelism = (tp, ep)
            if parallelism not in deployments[digest]:
                deployments[digest][parallelism] = set()
            deployments[digest][parallelism].add(str(source.relative_to(root)))

    result = []
    for digest, path in sorted(by_sha.items(), key=lambda item: item[1].name):
        # Inventory binds content identities only. Semantic validation belongs
        # to the selected model's load in build_scan_plan; an unrelated draft
        # configuration must not prevent scanning a supported target model.
        facts = sorted(deployments[digest].items())
        if not facts:
            # A config in configs/model is active input by repository contract.
            # TP=EP=1 is the semantic baseline when no deployment artifact pins
            # a larger physical sharding domain.
            facts = [(
                (1, 1), {"configs/model inventory (unsharded baseline)"})]
        result.append({
            "model_path": str(path.relative_to(root)),
            "model_name": path.stem,
            "config_sha256": digest,
            "deployments": [
                {
                    "tensor_parallel_size": parallelism[0],
                    "expert_parallel_size": parallelism[1],
                    "sources": sorted(sources),
                }
                for parallelism, sources in facts
            ],
        })
    return tuple(result)


def _extents(scan_size: str, observed: Sequence[int]) -> tuple[int, ...]:
    if scan_size == "smoke":
        return _SMOKE_EXTENTS
    elif scan_size == "full":
        base = _BASE_EXTENTS
    else:
        raise ValueError("scan_size must be 'smoke' or 'full'")
    values = set(base)
    values.update(value for value in observed if value > 0)
    return tuple(sorted(values))


@lru_cache(maxsize=None)
def _gemm_launch_m_index(
        hardware_id: str,
        ) -> Mapping[tuple[str, str, int, int, int], tuple[int, ...]]:
    """Index canonical GEMM launch observations by the physical M domain."""

    if hardware_id not in HARDWARE_IDS:
        raise ValueError(f"unsupported GEMM launch-data hardware {hardware_id!r}")
    entries = load_gemm_launch_selection(
        hardware_id,
        software_revision=gemm_launch_software_revision(hardware_id))
    values_by_key: dict[tuple[str, str, int, int, int], list[int]] = {}
    for selection_key in entries:
        (
            dtype, variant, batch, m, n, k, a_layout, b_layout, c_layout,
            trans_a, trans_b, epilogue, has_bias,
        ) = selection_key
        if (
                a_layout != _GEMM_SELECTION_ATTRIBUTES["a_layout"]
                or b_layout != _GEMM_SELECTION_ATTRIBUTES["b_layout"]
                or c_layout != _GEMM_SELECTION_ATTRIBUTES["c_layout"]
                or trans_a != _GEMM_SELECTION_ATTRIBUTES["trans_a"]
                or trans_b != _GEMM_SELECTION_ATTRIBUTES["trans_b"]
                or epilogue != _GEMM_SELECTION_ATTRIBUTES["epilogue"]
                or has_bias != _GEMM_SELECTION_ATTRIBUTES["has_bias"]):
            continue
        physical_key = (dtype, variant, batch, n, k)
        if physical_key not in values_by_key:
            values_by_key[physical_key] = []
        values_by_key[physical_key].append(m)
    return {
        physical_key: tuple(sorted(values))
        for physical_key, values in values_by_key.items()
    }


def _gemm_launch_m_values(
        hardware_id: str, dtype: str, batch: int, n: int, k: int,
        ) -> tuple[int, ...]:
    """Return only M values with an exact injectable canonical launch."""

    physical_key = (dtype, "gemm", batch, n, k)
    index = _gemm_launch_m_index(hardware_id)
    if physical_key not in index:
        return ()
    return index[physical_key]


def _observed_m_values(root: Path) -> tuple[int, ...]:
    values = set()
    for path in sorted((root / "configs/evaluation/experiments").glob("*.json")):
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
        for mapping in _walk_json(document):
            if "capture_sizes" not in mapping:
                continue
            capture_sizes = mapping["capture_sizes"]
            if not isinstance(capture_sizes, list):
                raise TypeError(f"capture_sizes in {path} must be a list")
            for value in capture_sizes:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"capture_sizes in {path} must be positive integers")
                values.add(value)
    return tuple(sorted(values))


def _gemm_positions(model: ModelConfig) -> tuple[str, ...]:
    common = ("o_proj", "gate_up_proj", "down_proj", "lm_head")
    if model.attention_layout == "standard_gqa":
        positions = ("qkv_proj",) + common
        if model.num_experts:
            positions += ("router",)
        return positions
    if model.attention_layout == "mla":
        return ("fused_qkv_a_proj", "q_b_proj", "kv_b_proj") + common
    if model.attention_layout == "dsa_mla":
        return (
            "fused_qkv_a_proj", "q_a_proj", "q_b_proj",
            "kv_a_proj_with_mqa", "kv_b_proj", "indexer_wq_b",
            "indexer_wk", "indexer_weights_proj",
            "indexer_wk_weights_proj",
        ) + common
    raise AssertionError(f"unhandled attention layout {model.attention_layout!r}")


def _sequence_id(*parts: str) -> str:
    return "|".join(parts)


def _base_case(
    *, sequence_id: str, family: str, hardware_id: str,
    model_name: str, model_path: str, tp: int, ep: int,
    axis: str, axis_value: int, controls: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "sequence_id": sequence_id,
        "family": family,
        "hardware_id": hardware_id,
        "model_name": model_name,
        "model_path": model_path,
        "tensor_parallel_size": tp,
        "expert_parallel_size": ep,
        "axis": axis,
        "axis_value": axis_value,
        "controls": _json_value(controls),
        "recipe": _json_value(recipe),
    }


def _append_case(
    cases: list[Mapping[str, Any]], sequences: dict[str, Mapping[str, Any]],
    case: Mapping[str, Any],
) -> None:
    sequence_id = case["sequence_id"]
    if sequence_id not in sequences:
        sequences[sequence_id] = {
            "sequence_id": sequence_id,
            "family": case["family"],
            "hardware_id": case["hardware_id"],
            "model_name": case["model_name"],
            "model_path": case["model_path"],
            "tensor_parallel_size": case["tensor_parallel_size"],
            "expert_parallel_size": case["expert_parallel_size"],
            "axis": case["axis"],
            "controls": case["controls"],
        }
    cases.append(case)


def build_scan_plan(
    *, root: Path = REPOSITORY_ROOT, hardware_ids: Sequence[str] = HARDWARE_IDS,
    families: Sequence[str] = FAMILIES, model_names: Sequence[str] = (),
    scan_size: str = "full",
) -> Mapping[str, Any]:
    """Build source-derived physical cases without executing a predictor."""

    requested_hardware = tuple(hardware_ids)
    if not requested_hardware or any(item not in HARDWARE_IDS for item in requested_hardware):
        raise ValueError(f"hardware_ids must be selected from {HARDWARE_IDS}")
    requested_families = tuple(families)
    if not requested_families or any(item not in FAMILIES for item in requested_families):
        raise ValueError(f"families must be selected from {FAMILIES}")
    active_models = discover_serving_models(root)
    selected = []
    selected_names = set(model_names)
    for entry in active_models:
        if selected_names and entry["model_name"] not in selected_names:
            continue
        selected.append(entry)
    if selected_names:
        found = {entry["model_name"] for entry in selected}
        missing = selected_names - found
        if missing:
            raise ValueError(f"unknown active Serving model(s): {sorted(missing)}")
    observed = _observed_m_values(root)
    extents = _extents(scan_size, observed)
    scan_m_values = set(extents)
    cases: list[Mapping[str, Any]] = []
    sequences: dict[str, Mapping[str, Any]] = {}

    for entry in selected:
        model_path = root / entry["model_path"]
        model = load_model_config(model_path)
        for deployment in entry["deployments"]:
            tp = deployment["tensor_parallel_size"]
            ep = deployment["expert_parallel_size"]
            for hardware_id in requested_hardware:
                prefix = (hardware_id, entry["model_name"], f"tp{tp}", f"ep{ep}")
                if "gemm" in requested_families:
                    for position in _gemm_positions(model):
                        try:
                            _, n, k = infer_gemm_dimensions(
                                model, position, 1, tp_size=tp)
                        except ValueError:
                            # The active deployment is not legal for this model
                            # position; it cannot produce a physical Serving GEMM.
                            continue
                        exact_m_values = _gemm_launch_m_values(
                            hardware_id, model.compute_dtype, 1, n, k)
                        exact_m_values = tuple(
                            m for m in exact_m_values if m in scan_m_values)
                        if not exact_m_values:
                            continue
                        sequence_id = _sequence_id(*prefix, "gemm", position, "m")
                        controls = {
                            "kind": "GEMM", "position": position,
                            "dtype": model.compute_dtype, "b": 1,
                            "n": n, "k": k, "variant": "gemm",
                        }
                        for m in exact_m_values:
                            recipe = {
                                "kind": "gemm", "shape": {"b": 1, "m": m, "n": n, "k": k},
                                "dtype": model.compute_dtype, "variant": "gemm",
                                "selection_attributes": {
                                    **_GEMM_SELECTION_ATTRIBUTES,
                                    "launch_policy": "injected_launch",
                                },
                                "prediction_attributes": {},
                                "position": position,
                            }
                            _append_case(cases, sequences, _base_case(
                                sequence_id=sequence_id, family="gemm", hardware_id=hardware_id,
                                model_name=entry["model_name"], model_path=entry["model_path"],
                                tp=tp, ep=ep, axis="m", axis_value=m,
                                controls=controls, recipe=recipe))

                if "flash_attn" in requested_families:
                    if model.attention_layout == "standard_gqa":
                        q_heads = model.num_attention_heads // tp
                        kv_heads = model.kv_heads_per_tensor_parallel_rank(tp)
                        fa_axes = (
                            ("q_len", extents, 1, 4096),
                            ("kv_len", extents, 1, None),
                            ("batch", extents, 1, 512),
                        )
                        for axis, values, fixed_q, fixed_kv in fa_axes:
                            sequence_id = _sequence_id(*prefix, "flash_attn", "decode" if axis != "q_len" else "prefill", axis)
                            axis_values = values
                            if axis == "q_len":
                                # Q=1 is a decode-shaped Serving dispatch;
                                # compare it on the decode axis rather than
                                # forging a prefill KernelSpec with decode ABI.
                                axis_values = tuple(item for item in values if item > 1)
                            for value in axis_values:
                                if axis == "q_len":
                                    q_len, kv_len, batch = value, max(value, fixed_kv), 1
                                    variant = "prefill"
                                elif axis == "kv_len":
                                    q_len, kv_len, batch = fixed_q, max(value, fixed_q), 1
                                    variant = "decode"
                                else:
                                    q_len, kv_len, batch = 1, fixed_kv, value
                                    variant = "decode"
                                shape = {
                                    "batch": batch, "q_len": q_len, "kv_len": kv_len,
                                    "head_dim": model.head_dim, "num_query_heads": q_heads,
                                    "num_kv_heads": kv_heads,
                                }
                                selection_attributes = {
                                    "causal": True, "kv_layout": "paged_varlen",
                                    "kv_dist": "uniform", "fa_ksteps": None,
                                    "fa_eff_mfrac": 1.0, "q_lens": tuple([q_len] * batch),
                                    "kv_lens": tuple([kv_len] * batch),
                                    "launch_policy": "analytical_launch",
                                    "split_policy": "serving", "is_decode": variant == "decode",
                                }
                                controls = {
                                    "kind": "FLASH_ATTN", "variant": variant,
                                    "dtype": model.compute_dtype, "axis": axis,
                                    "head_dim": model.head_dim, "num_query_heads": q_heads,
                                    "num_kv_heads": kv_heads, "kv_layout": "paged_varlen",
                                }
                                recipe = {
                                    "kind": "flash_attn", "shape": shape,
                                    "dtype": model.compute_dtype, "variant": variant,
                                    "selection_attributes": selection_attributes,
                                    "prediction_attributes": {
                                        "causal": True, "kv_layout": "paged_varlen",
                                        "kv_dist": "uniform", "fa_ksteps": None,
                                        "fa_eff_mfrac": 1.0, "q_lens": tuple([q_len] * batch),
                                        "kv_lens": tuple([kv_len] * batch),
                                    },
                                }
                                _append_case(cases, sequences, _base_case(
                                    sequence_id=sequence_id, family="flash_attn", hardware_id=hardware_id,
                                    model_name=entry["model_name"], model_path=entry["model_path"],
                                    tp=tp, ep=ep, axis=axis, axis_value=value,
                                    controls=controls, recipe=recipe))
                    elif model.model_type == "kimi_k2":
                        # Kimi's pinned MLA lowering has a source-backed
                        # A800 prefill/decode pair.  Other hardware packages
                        # fail closed because they do not own this target ABI.
                        query_heads = model.num_attention_heads // tp
                        mla_axes = (
                            ("q_len", tuple(item for item in extents if item > 1), "mla_prefill"),
                            ("kv_len", extents, "mla_decode"),
                            ("batch", extents, "mla_decode"),
                        )
                        for axis, values, variant in mla_axes:
                            sequence_id = _sequence_id(*prefix, "flash_attn", variant, axis)
                            for value in values:
                                if axis == "q_len":
                                    batch, q_len, kv_len = 1, value, max(value, 4096)
                                    qk_head_dim, value_head_dim = 192, 128
                                elif axis == "kv_len":
                                    batch, q_len, kv_len = 1, 1, max(value, 1)
                                    qk_head_dim, value_head_dim = 576, 512
                                else:
                                    batch, q_len, kv_len = value, 1, 512
                                    qk_head_dim, value_head_dim = 576, 512
                                shape = {
                                    "batch": batch, "q_len": q_len, "kv_len": kv_len,
                                    "num_query_heads": query_heads,
                                    "qk_head_dim": qk_head_dim,
                                    "v_head_dim": value_head_dim,
                                    "kv_lora_rank": model.kv_lora_rank,
                                }
                                attributes = {
                                    "causal": True, "kv_layout": "mla", "kv_dist": "uniform",
                                    "q_lens": tuple([q_len] * batch),
                                    "kv_lens": tuple([kv_len] * batch),
                                }
                                recipe = {
                                    "kind": "flash_attn", "shape": shape,
                                    "dtype": model.compute_dtype, "variant": variant,
                                    "selection_attributes": attributes,
                                    "prediction_attributes": attributes,
                                }
                                controls = {
                                    "kind": "FLASH_ATTN", "variant": variant,
                                    "dtype": model.compute_dtype, "axis": axis,
                                    "num_query_heads": query_heads, "kv_lora_rank": model.kv_lora_rank,
                                    "kv_layout": "mla",
                                }
                                _append_case(cases, sequences, _base_case(
                                    sequence_id=sequence_id, family="flash_attn", hardware_id=hardware_id,
                                    model_name=entry["model_name"], model_path=entry["model_path"],
                                    tp=tp, ep=ep, axis=axis, axis_value=value,
                                    controls=controls, recipe=recipe))

                if "fused_moe" in requested_families and model.num_experts:
                    # Current production GPU/NPU FusedMoE launch contracts are
                    # source-backed for the BF16 standard-GQA domain.  MLA/DSA
                    # INT4 routes are represented as fail-closed unsupported.
                    sequence_id = _sequence_id(*prefix, "fused_moe", "tokens")
                    for tokens in extents:
                        shape = {
                            "tokens": tokens, "experts": model.num_experts,
                            "topk": model.num_experts_per_tok, "hidden": model.hidden_size,
                            "intermediate": model.moe_intermediate_size,
                        }
                        counts = _balanced_counts(tokens * model.num_experts_per_tok, model.num_experts)
                        owned = _owned_experts(model.num_experts, ep, 0)
                        local_counts = tuple(
                            count if expert in owned else 0
                            for expert, count in enumerate(counts))
                        recipe = {
                            "kind": "fused_moe", "shape": shape,
                            "dtype": model.compute_dtype,
                            "prediction_attributes": {
                                "tokens_per_expert": counts,
                                "local_tokens_per_expert": local_counts,
                                "expert_parallel_size": ep, "expert_rank": 0,
                                "owned_experts": owned,
                                "expert_weight_tile_rows": None,
                            },
                            "selection_attributes": {"launch_policy": "analytical_launch"},
                            "source_supported": (
                                model.attention_layout == "standard_gqa"
                                and model.compute_dtype == "bf16"),
                        }
                        controls = {
                            "kind": "FUSED_MOE", "dtype": model.compute_dtype,
                            "experts": model.num_experts, "topk": model.num_experts_per_tok,
                            "hidden": model.hidden_size, "intermediate": model.moe_intermediate_size,
                            "expert_parallel_size": ep, "expert_rank": 0,
                        }
                        _append_case(cases, sequences, _base_case(
                            sequence_id=sequence_id, family="fused_moe", hardware_id=hardware_id,
                            model_name=entry["model_name"], model_path=entry["model_path"],
                            tp=tp, ep=ep, axis="tokens", axis_value=tokens,
                            controls=controls, recipe=recipe))

                if "unfused_moe" in requested_families and model.num_experts:
                    sequence_id = _sequence_id(*prefix, "unfused_moe", "tokens")
                    for tokens in extents:
                        shape = {
                            "tokens": tokens, "experts": model.num_experts,
                            "topk": model.num_experts_per_tok, "hidden": model.hidden_size,
                            "intermediate": model.moe_intermediate_size,
                        }
                        recipe = {
                            "kind": "unfused_moe", "shape": shape,
                            "dtype": model.compute_dtype,
                            "prediction_attributes": {
                                "tokens_per_expert": _balanced_counts(
                                    tokens * model.num_experts_per_tok, model.num_experts),
                                "expert_weight_tile_rows": None,
                            },
                        }
                        controls = {
                            "kind": "UNFUSED_MOE", "dtype": model.compute_dtype,
                            "experts": model.num_experts, "topk": model.num_experts_per_tok,
                            "hidden": model.hidden_size, "intermediate": model.moe_intermediate_size,
                        }
                        _append_case(cases, sequences, _base_case(
                            sequence_id=sequence_id, family="unfused_moe", hardware_id=hardware_id,
                            model_name=entry["model_name"], model_path=entry["model_path"],
                            tp=tp, ep=ep, axis="tokens", axis_value=tokens,
                            controls=controls, recipe=recipe))

                if "vector" in requested_families:
                    _add_vector_cases(
                        cases=cases, sequences=sequences, entry=entry,
                        model=model, hardware_id=hardware_id, tp=tp, ep=ep,
                        extents=extents)

    return {
        "schema_version": 1,
        "scan_size": scan_size,
        "models": selected,
        "observed_m_values": list(observed),
        "base_extents": list(extents),
        "sequences": tuple(
            sequences[key] for key in sorted(sequences)),
        "cases": tuple(sorted(cases, key=_case_sort_key)),
    }


def _balanced_counts(total: int, experts: int) -> tuple[int, ...]:
    if total <= 0 or experts <= 0:
        raise ValueError("routing total and expert count must be positive")
    quotient, remainder = divmod(total, experts)
    return tuple(quotient + (1 if index < remainder else 0) for index in range(experts))


def _owned_experts(experts: int, ep: int, rank: int) -> tuple[int, ...]:
    if ep <= 0 or experts % ep or not 0 <= rank < ep:
        raise ValueError("invalid expert-parallel placement")
    per_rank = experts // ep
    return tuple(range(rank * per_rank, (rank + 1) * per_rank))


def _add_vector_cases(
    *, cases: list[Mapping[str, Any]], sequences: dict[str, Mapping[str, Any]],
    entry: Mapping[str, Any], model: ModelConfig, hardware_id: str,
    tp: int, ep: int, extents: Sequence[int],
) -> None:
    """Emit one-factor vector and KV-cache sequences for Serving variants."""

    prefix = (hardware_id, entry["model_name"], f"tp{tp}", f"ep{ep}", "vector")
    q_heads = model.num_attention_heads // tp
    if model.attention_layout == "standard_gqa":
        kv_heads = model.kv_heads_per_tensor_parallel_rank(tp)
    else:
        kv_heads = 1
    variants = (
        ("rmsnorm", "rows", lambda value: (
            KernelKind.VECTOR, "rmsnorm", {"rows": value, "hidden": model.hidden_size}, {}, {})),
        ("activation", "tok", lambda value: (
            KernelKind.VECTOR, "silu", {
                "tok": value, "intermediate": model.intermediate_size,
                "element_count": value * model.intermediate_size,
            }, {}, {})),
        ("elementwise", "element_count", lambda value: (
            KernelKind.VECTOR, "elementwise", {"element_count": value * model.hidden_size},
            {"reads": 2, "writes": 1, "operation": "add"},
            {"reads": 2, "writes": 1, "operation": "add"})),
    )
    if model.attention_layout == "standard_gqa":
        variants += (("rotary", "seq_len", lambda value: (
            KernelKind.VECTOR, "rotary", {
                "seq_len": value, "num_query_heads": q_heads,
                "num_kv_heads": kv_heads, "head_dim": model.head_dim,
                "element_count": value * (q_heads + kv_heads) * model.head_dim,
            }, {}, {})),)
        variants += (("kv_cache", "seq_len", lambda value: (
            KernelKind.KV_CACHE, "paged_kv_cache", {
                "element_count": value * kv_heads * model.head_dim,
                "seq_len": value, "num_kv_heads": kv_heads,
                "head_dim": model.head_dim,
            }, dict(PAGED_KV_CACHE_ATTRIBUTES), dict(PAGED_KV_CACHE_ATTRIBUTES))),)
    for label, axis, builder in variants:
        sequence_id = _sequence_id(*prefix, label, axis)
        for value in extents:
            kind, variant, shape, selection_attributes, prediction_attributes = builder(value)
            recipe = {
                "kind": "vector" if kind is KernelKind.VECTOR else "kv_cache",
                "shape": shape, "dtype": model.compute_dtype, "variant": variant,
                "selection_attributes": {
                    "launch_policy": "analytical_launch", **selection_attributes},
                "prediction_attributes": prediction_attributes,
            }
            controls = {
                "kind": kind.value, "variant": variant, "dtype": model.compute_dtype,
                "axis": axis, "model_hidden": model.hidden_size,
                "num_query_heads": q_heads, "num_kv_heads": kv_heads,
                "head_dim": model.head_dim,
            }
            _append_case(cases, sequences, _base_case(
                sequence_id=sequence_id, family="vector", hardware_id=hardware_id,
                model_name=entry["model_name"], model_path=entry["model_path"],
                tp=tp, ep=ep, axis=axis, axis_value=value,
                controls=controls, recipe=recipe))
    if model.attention_layout in ("mla", "dsa_mla"):
        _add_target_vector_cases(
            cases=cases, sequences=sequences, entry=entry, model=model,
            hardware_id=hardware_id, tp=tp, ep=ep, extents=extents,
            prefix=prefix, query_heads=q_heads)


def _add_target_vector_cases(
    *, cases: list[Mapping[str, Any]], sequences: dict[str, Mapping[str, Any]],
    entry: Mapping[str, Any], model: ModelConfig, hardware_id: str,
    tp: int, ep: int, extents: Sequence[int], prefix: tuple[str, ...],
    query_heads: int,
) -> None:
    if model.target_semantics is None:
        raise ValueError("target vector scan requires target model semantics")
    semantics = model.target_semantics
    rope_attributes = {
        "rope_layout": semantics.rope_layout,
        "rope_type": semantics.rope_type,
        "rope_theta": semantics.rope_theta,
        "rope_factor": semantics.rope_factor,
        "rope_original_max_position_embeddings": semantics.rope_original_max_position_embeddings,
        "rope_beta_fast": semantics.rope_beta_fast,
        "rope_beta_slow": semantics.rope_beta_slow,
        "rope_mscale": semantics.rope_mscale,
        "rope_mscale_all_dim": semantics.rope_mscale_all_dim,
    }
    compressed_dim = model.kv_lora_rank + model.qk_rope_head_dim
    target_variants = (
        ("mla_rope", "seq_len", "vector", "mla_rope", model.compute_dtype),
        ("mla_kv_cache", "seq_len", "kv_cache", "mla_kv_cache", model.compute_dtype),
    )
    for label, axis, kind, variant, dtype in target_variants:
        sequence_id = _sequence_id(*prefix, label, axis)
        for value in extents:
            if kind == "vector":
                shape = {
                    "element_count": value * (query_heads + 1) * model.qk_rope_head_dim,
                    "seq_len": value, "num_query_heads": query_heads,
                    "num_key_heads": 1, "head_dim": model.qk_rope_head_dim,
                }
                selection_attributes = rope_attributes
                prediction_attributes = rope_attributes
            elif kind == "kv_cache":
                shape = {
                    "element_count": value * compressed_dim, "seq_len": value,
                    "compressed_dim": compressed_dim,
                }
                selection_attributes = {
                    "cache_layout": "paged_mla", "cache_write_semantics": "append"}
                prediction_attributes = selection_attributes
            else:
                raise AssertionError(f"unhandled target vector kind {kind!r}")
            recipe = {
                "kind": kind, "shape": shape, "dtype": dtype, "variant": variant,
                "selection_attributes": selection_attributes,
                "prediction_attributes": prediction_attributes,
            }
            controls = {
                "kind": "VECTOR" if kind == "vector" else "KV_CACHE",
                "variant": variant, "dtype": dtype, "axis": axis,
                "num_query_heads": query_heads, "compressed_dim": compressed_dim,
            }
            _append_case(cases, sequences, _base_case(
                sequence_id=sequence_id, family="vector", hardware_id=hardware_id,
                model_name=entry["model_name"], model_path=entry["model_path"],
                tp=tp, ep=ep, axis=axis, axis_value=value,
                controls=controls, recipe=recipe))


def _case_sort_key(case: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        case["hardware_id"], case["family"], case["model_name"],
        case["tensor_parallel_size"], case["expert_parallel_size"],
        case["sequence_id"], case["axis_value"],
    )


def _unsupported_result(case: Mapping[str, Any], reason: str) -> list[Mapping[str, Any]]:
    return [{
        **case,
        "physical_label": "unsupported",
        "status": "unsupported",
        "unsupported_reason": reason,
    }]


def _is_fail_closed_unsupported(exc: Exception) -> bool:
    """Recognize only explicit source/contract absence, never arbitrary bugs."""

    if isinstance(exc, NotImplementedError):
        return True
    if not isinstance(exc, ValueError):
        return False
    message = str(exc)
    return (
        message.startswith("unsupported launch dtype ")
        or "has no request size" in message
        or message.startswith("uncharacterized ")
        or message == "CANN GEMM output bytes exhaust the operand DRAM path"
    )


def _build_specs(case: Mapping[str, Any], hardware: HWSpec) -> tuple[KernelSpec, ...]:
    recipe = case["recipe"]
    kind = recipe["kind"]
    shape = recipe["shape"]
    dtype = recipe["dtype"]
    if kind == "gemm":
        launch = select_launch_config(
            case["hardware_id"], KernelKind.GEMM, shape, dtype,
            recipe["variant"], recipe["selection_attributes"], hardware=hardware)
        return (KernelSpec(
            KernelKind.GEMM, case["hardware_id"], shape, dtype=dtype,
            variant=launch.execution_variant, launch=launch,
            attributes=recipe["prediction_attributes"]),)
    if kind == "flash_attn":
        launch = select_launch_config(
            case["hardware_id"], KernelKind.FLASH_ATTN, shape, dtype,
            recipe["variant"], recipe["selection_attributes"], hardware=hardware)
        return (KernelSpec(
            KernelKind.FLASH_ATTN, case["hardware_id"], shape, dtype=dtype,
            variant=launch.execution_variant, launch=launch,
            attributes=recipe["prediction_attributes"]),)
    if kind == "fused_moe":
        if not recipe["source_supported"]:
            raise NotImplementedError(
                "active target-model routed MoE has no cross-hardware "
                "source-backed FusedMoE launch selector")
        prediction_attributes = dict(recipe["prediction_attributes"])
        for name in (
                "tokens_per_expert", "local_tokens_per_expert",
                "owned_experts"):
            prediction_attributes[name] = tuple(prediction_attributes[name])
        return build_fused_moe_physical_specs(
            hardware_id=case["hardware_id"], shape=shape, dtype=dtype,
            prediction_attributes=prediction_attributes,
            selection_attributes=recipe["selection_attributes"], hardware=hardware)
    if kind == "unfused_moe":
        return (KernelSpec(
            KernelKind.UNFUSED_MOE, case["hardware_id"], shape, dtype=dtype,
            variant="decomposed", launch=None,
            attributes=recipe["prediction_attributes"],
            execution_support="unsupported",
            unsupported_reason="launch_selector_unavailable"),)
    if kind == "vector":
        launch = select_launch_config(
            case["hardware_id"], KernelKind.VECTOR, shape, dtype,
            recipe["variant"], recipe["selection_attributes"], hardware=hardware)
        return (KernelSpec(
            KernelKind.VECTOR, case["hardware_id"], shape, dtype=dtype,
            variant=launch.execution_variant, launch=launch,
            attributes=recipe["prediction_attributes"]),)
    if kind == "kv_cache":
        launch = select_launch_config(
            case["hardware_id"], KernelKind.KV_CACHE, shape, dtype,
            recipe["variant"], recipe["selection_attributes"], hardware=hardware)
        return (KernelSpec(
            KernelKind.KV_CACHE, case["hardware_id"], shape, dtype=dtype,
            variant=launch.execution_variant, launch=launch,
            attributes=recipe["prediction_attributes"]),)
    raise AssertionError(f"unhandled order-check recipe {kind!r}")


def _predict_case(case: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Worker entry point.  Every supported row calls the public predict API."""

    try:
        hardware = HWSpec(get_hardware_config_path(case["hardware_id"]))
        specs = _build_specs(case, hardware)
    except Exception as exc:
        if _is_fail_closed_unsupported(exc):
            return _unsupported_result(case, str(exc))
        return [{
            **case, "physical_label": "construction", "status": "failed",
            "error_type": type(exc).__name__, "error": str(exc),
        }]
    results = []
    for ordinal, spec in enumerate(specs):
        label = f"{ordinal}:{spec.kind.value}:{spec.variant}"
        try:
            prediction = predict(spec, hardware, "midlayer")
        except Exception as exc:  # retained in manifest; parent exits nonzero
            if _is_fail_closed_unsupported(exc):
                results.append({
                    **case, "physical_label": label, "status": "unsupported",
                    "unsupported_reason": str(exc), "kernel_spec": _json_value(spec),
                })
                continue
            results.append({
                **case, "physical_label": label, "status": "failed",
                "error_type": type(exc).__name__, "error": str(exc),
                "kernel_spec": _json_value(spec),
            })
            continue
        results.append({
            **case, "physical_label": label, "status": "success",
            "kernel_spec": _json_value(spec), "launch": _launch_summary(spec.launch),
            "prediction": _prediction_summary(prediction),
        })
    return results


def _run_cases(cases: Sequence[Mapping[str, Any]], jobs: int) -> list[Mapping[str, Any]]:
    if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= 48:
        raise ValueError("jobs must be an integer in [1, 48]")
    ordered = sorted(cases, key=_case_sort_key)
    if jobs == 1:
        nested = [_predict_case(case) for case in ordered]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            nested = list(executor.map(_predict_case, ordered))
    result = [row for rows in nested for row in rows]
    return sorted(result, key=_result_sort_key)


def _result_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return _case_sort_key(row) + (row["physical_label"],)


def _gemm_boundary_cases(
    base_results: Sequence[Mapping[str, Any]],
    existing_cases: Sequence[Mapping[str, Any]], scan_size: str,
) -> tuple[Mapping[str, Any], ...]:
    """Avoid analytical GEMM boundary expansion outside canonical launch data."""

    del base_results, existing_cases, scan_size
    return ()


def _mapping_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> Mapping[str, Any]:
    keys = sorted(set(before) | set(after))
    result = {}
    for key in keys:
        before_value = before[key] if key in before else None
        after_value = after[key] if key in after else None
        if before_value != after_value:
            result[key] = {"before": before_value, "after": after_value}
    return result


def _memory_bytes(summary: Mapping[str, Any]) -> float:
    total = 0.0
    for path in summary["memory"]:
        for node in path["nodes"]:
            total += float(node["bytes"])
    return total


def _predictor_source_function(row: Mapping[str, Any]) -> str:
    """Return the actual Forward owner for a diagnostic component."""

    kind = row["kernel_spec"]["kind"]
    if kind == "GEMM":
        return "src/kernel/predictors/gemm.py:GEMMPredictor.midlayer"
    if kind == "FLASH_ATTN":
        return "src/kernel/predictors/flash_attn.py:FlashAttentionPredictor.midlayer"
    if kind == "FUSED_MOE":
        return "src/kernel/predictors/fused_moe.py:FusedMoEPredictor.midlayer"
    if kind == "UNFUSED_MOE":
        return "src/kernel/predictors/unfused_moe.py:UnfusedMoEPredictor.midlayer"
    if kind == "VECTOR":
        return "src/kernel/predictors/vector.py:VectorPredictor.midlayer"
    if kind == "KV_CACHE":
        return "src/kernel/predictors/kv_cache_update.py:KVCacheUpdatePredictor.midlayer"
    raise AssertionError(f"unhandled diagnostic KernelSpec kind {kind!r}")


def _root_cause(before: Mapping[str, Any], after: Mapping[str, Any]) -> Mapping[str, Any]:
    launch_diff = _mapping_diff(before["launch"], after["launch"])
    if set(launch_diff) == {"source_program"}:
        return {
            "classification": "source program transition",
            "trigger_fields": ["launch.source_program"],
            "source_function": _source_function(after),
            "mechanism": "the production source program changed",
        }
    if launch_diff:
        return {
            "classification": "launch transition",
            "trigger_fields": sorted(launch_diff),
            "source_function": _source_function(after),
            "mechanism": "the production launch selector changed physical launch fields",
        }
    before_prediction = before["prediction"]
    after_prediction = after["prediction"]
    before_bytes = _memory_bytes(before_prediction)
    after_bytes = _memory_bytes(after_prediction)
    if after_bytes < before_bytes:
        return {
            "classification": "traffic",
            "trigger_fields": ["diagnostics.memory.nodes.bytes"],
            "source_function": _predictor_source_function(after),
            "mechanism": "diagnostic memory traffic decreases with the workload axis",
        }
    before_pipeline = before_prediction["pipeline"]
    after_pipeline = after_prediction["pipeline"]
    if (before_pipeline is not None and after_pipeline is not None
            and before_pipeline["source_schedule"]
            != after_pipeline["source_schedule"]):
        return {
            "classification": "source program transition",
            "trigger_fields": ["diagnostics.pipeline.source_schedule"],
            "source_function": _predictor_source_function(after),
            "mechanism": (
                "the source-derived physical worker schedule changed"),
        }
    if before_pipeline is not None and after_pipeline is not None and (
            after_pipeline["execution_time_ns"] < before_pipeline["execution_time_ns"]):
        return {
            "classification": "pipeline",
            "trigger_fields": ["diagnostics.pipeline.execution_time_ns"],
            "source_function": _predictor_source_function(after),
            "mechanism": "the encoded pipeline execution component decreases",
        }
    resource_diff = _mapping_diff(
        before_prediction["time_per_resource_ns"],
        after_prediction["time_per_resource_ns"])
    decreased = [
        name for name, values in resource_diff.items()
        if values["before"] is not None and values["after"] is not None
        and values["after"] < values["before"]
    ]
    if decreased:
        return {
            "classification": "resource/residency",
            "trigger_fields": decreased,
            "source_function": _predictor_source_function(after),
            "mechanism": "a Forward resource-time component decreases without a launch change",
        }
    if after_prediction["launch_dispatch_ns"] < before_prediction["launch_dispatch_ns"]:
        return {
            "classification": "runtime envelope",
            "trigger_fields": ["launch_dispatch_ns"],
            "source_function": _predictor_source_function(after),
            "mechanism": "the runtime dispatch envelope decreases",
        }
    return {
        "classification": "input/lowering",
        "trigger_fields": ["KernelSpec/diagnostics total"],
        "source_function": "evaluation/kernel/prediction_order.py:_build_specs",
        "mechanism": "no decreasing launch, traffic, pipeline, or resource component was isolated",
    }


def _source_function(row: Mapping[str, Any]) -> str:
    hardware = row["hardware_id"]
    family = row["family"]
    if family == "gemm":
        return f"src/kernel/launch_config/{hardware}/gemm_launch.py:select_gemm_launch"
    if family == "flash_attn":
        return f"src/kernel/launch_config/{hardware}/flash_attn_launch.py:select_flash_attn_launch"
    if family == "fused_moe":
        return f"src/kernel/launch_config/{hardware}/fused_moe_launch.py:select_fused_moe_launch"
    if family == "vector":
        return f"src/kernel/launch_config/{hardware}/vector_launch.py:select_vector_launch"
    return "src/kernel/predictors/unfused_moe.py:UnfusedMoEPredictor.midlayer"


def compare_ordered_points(
    rows: Sequence[Mapping[str, Any]], *, epsilon_ns: float = DEFAULT_EPSILON_NS,
) -> list[Mapping[str, Any]]:
    """Compare adjacent successful points of one physical partial-order chain."""

    if not math.isfinite(epsilon_ns) or epsilon_ns < 0:
        raise ValueError("epsilon_ns must be finite and non-negative")
    violations = []
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["status"] != "success":
            continue
        key = (row["sequence_id"], row["physical_label"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)
    for key in sorted(grouped):
        points = sorted(grouped[key], key=lambda row: row["axis_value"])
        for before, after in zip(points, points[1:]):
            if before["axis_value"] >= after["axis_value"]:
                raise ValueError("partial-order sequence must have strictly increasing axis values")
            before_ns = before["prediction"]["total_ns"]
            after_ns = after["prediction"]["total_ns"]
            delta_ns = after_ns - before_ns
            if delta_ns >= -epsilon_ns:
                continue
            cause = _root_cause(before, after)
            violations.append({
                "sequence_id": before["sequence_id"],
                "physical_label": before["physical_label"],
                "hardware_id": before["hardware_id"],
                "family": before["family"],
                "model_name": before["model_name"],
                "axis": before["axis"],
                "work_delta": after["axis_value"] - before["axis_value"],
                "before_axis_value": before["axis_value"],
                "after_axis_value": after["axis_value"],
                "before_ns": before_ns,
                "after_ns": after_ns,
                "delta_ns": delta_ns,
                "drop_pct": 100.0 * (-delta_ns) / before_ns,
                "before_kernel_spec": before["kernel_spec"],
                "after_kernel_spec": after["kernel_spec"],
                "before_launch": before["launch"],
                "after_launch": after["launch"],
                "launch_diff": _mapping_diff(before["launch"], after["launch"]),
                "prediction_diff": {
                    "total_ns": {"before": before_ns, "after": after_ns},
                    "launch_dispatch_ns": {
                        "before": before["prediction"]["launch_dispatch_ns"],
                        "after": after["prediction"]["launch_dispatch_ns"],
                    },
                    "bottleneck_group": {
                        "before": before["prediction"]["bottleneck_group"],
                        "after": after["prediction"]["bottleneck_group"],
                    },
                    "time_per_group_ns": _mapping_diff(
                        before["prediction"]["time_per_group_ns"],
                        after["prediction"]["time_per_group_ns"]),
                    "time_per_resource_ns": _mapping_diff(
                        before["prediction"]["time_per_resource_ns"],
                        after["prediction"]["time_per_resource_ns"]),
                    "compute_time_by_role_ns": _mapping_diff(
                        before["prediction"]["compute_time_by_role_ns"],
                        after["prediction"]["compute_time_by_role_ns"]),
                    "memory": {
                        "before": before["prediction"]["memory"],
                        "after": after["prediction"]["memory"],
                        "total_bytes": {
                            "before": _memory_bytes(before["prediction"]),
                            "after": _memory_bytes(after["prediction"]),
                        },
                    },
                    "pipeline": {
                        "before": before["prediction"]["pipeline"],
                        "after": after["prediction"]["pipeline"],
                    },
                    "critical_path": {
                        "before": before["prediction"]["critical_path"],
                        "after": after["prediction"]["critical_path"],
                    },
                },
                "root_cause": cause,
            })
    return sorted(violations, key=lambda row: (-row["delta_ns"], row["sequence_id"], row["physical_label"]))


def _counts(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    return {
        "success": sum(row["status"] == "success" for row in rows),
        "unsupported": sum(row["status"] == "unsupported" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
    }


def _unsupported_records(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Preserve every fail-closed input and reason outside the success path."""

    return [
        row for row in sorted(rows, key=_result_sort_key)
        if row["status"] == "unsupported"
    ]


def _unsupported_summaries(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    counts: dict[tuple[str, str, str], int] = {}
    for record in records:
        key = (
            record["hardware_id"], record["family"],
            record["unsupported_reason"],
        )
        if key not in counts:
            counts[key] = 0
        counts[key] += 1
    return [
        {
            "hardware_id": hardware_id,
            "family": family,
            "reason": reason,
            "count": counts[(hardware_id, family, reason)],
        }
        for hardware_id, family, reason in sorted(counts)
    ]


def _summary_rows(rows: Sequence[Mapping[str, Any]], violations: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result = []
    for hardware in HARDWARE_IDS:
        for family in FAMILIES:
            scoped = [row for row in rows if row["hardware_id"] == hardware and row["family"] == family]
            if not scoped:
                continue
            scoped_violations = [
                row for row in violations
                if row["hardware_id"] == hardware and row["family"] == family]
            counts = _counts(scoped)
            maximum_drop = 0.0
            if scoped_violations:
                maximum_drop = max(-row["delta_ns"] for row in scoped_violations)
            result.append({
                "hardware_id": hardware,
                "family": family,
                "point_count": len(scoped),
                "sequence_count": len({row["sequence_id"] for row in scoped}),
                "success_count": counts["success"],
                "unsupported_count": counts["unsupported"],
                "failed_count": counts["failed"],
                "violation_count": len(scoped_violations),
                "maximum_drop_ns": maximum_drop,
            })
    return result


def _sequence_lowering_source(sequence: Mapping[str, Any]) -> str:
    family = sequence["family"]
    controls = sequence["controls"]
    if family == "gemm":
        return (
            "src.serving.workload.shape_inference.infer_gemm_dimensions"
            f" ({controls['position']})")
    if family == "flash_attn":
        return "Serving FlashAttention lowering with source-backed launch selection"
    if family == "fused_moe":
        return "src.kernel.fused_moe_physical_specs.build_fused_moe_physical_specs"
    if family == "unfused_moe":
        return "explicit fail-closed UNFUSED_MOE KernelSpec"
    if family == "vector":
        return "Serving vector/KV-cache lowering with source-backed launch selection"
    raise AssertionError(f"unhandled order-check family {family!r}")


def _shape_design(
    sequence: Mapping[str, Any], cases: Sequence[Mapping[str, Any]],
    initial_axis_values: set[int], boundary_axis_values: set[int],
) -> Mapping[str, Any]:
    """Describe the one-factor shape chain without reproducing predictor math."""

    shapes = [case["recipe"]["shape"] for case in cases]
    field_names = sorted({name for shape in shapes for name in shape})
    fixed_shape = {}
    varying_shape_fields = []
    for name in field_names:
        values = [shape[name] for shape in shapes]
        if all(value == values[0] for value in values[1:]):
            fixed_shape[name] = values[0]
        else:
            varying_shape_fields.append(name)
    selection_attributes = [
        (
            case["recipe"]["selection_attributes"]
            if "selection_attributes" in case["recipe"] else {}
        )
        for case in cases
    ]
    selection_field_names = sorted({
        name for attributes in selection_attributes for name in attributes
    })
    fixed_selection_attributes = {}
    varying_selection_attributes = []
    for name in selection_field_names:
        if not all(name in attributes for attributes in selection_attributes):
            varying_selection_attributes.append(name)
            continue
        values = [attributes[name] for attributes in selection_attributes]
        if all(value == values[0] for value in values[1:]):
            fixed_selection_attributes[name] = values[0]
        else:
            varying_selection_attributes.append(name)
    axis = sequence["axis"]
    if axis not in varying_shape_fields:
        raise RuntimeError(
            f"order-check sequence {sequence['sequence_id']} does not vary its declared axis {axis!r}")
    return {
        "declared_axis": axis,
        "varying_shape_fields": varying_shape_fields,
        "fixed_shape": fixed_shape,
        "fixed_selection_attributes": fixed_selection_attributes,
        "varying_selection_attributes": varying_selection_attributes,
        "initial_axis_values": sorted(initial_axis_values),
        "derived_boundary_axis_values": sorted(boundary_axis_values),
        "lowering_source": _sequence_lowering_source(sequence),
    }


def _family_axis_coverage(
    sequences: Sequence[Mapping[str, Any]],
    cases_by_sequence: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Mapping[str, Any]:
    """Summarize active Serving topology coverage without a Cartesian sweep."""

    result = {}
    for family in FAMILIES:
        scoped = [sequence for sequence in sequences if sequence["family"] == family]
        if not scoped:
            continue
        summary = {
            "sequence_count": len(scoped),
            "source_models": sorted({sequence["model_name"] for sequence in scoped}),
            "axes": sorted({sequence["axis"] for sequence in scoped}),
        }
        variants = set()
        for sequence in scoped:
            controls = sequence["controls"]
            if "variant" in controls:
                variants.add(controls["variant"])
            elif family == "fused_moe":
                variants.add("fused_moe")
            elif family == "unfused_moe":
                variants.add("unfused_moe")
            else:
                raise AssertionError(f"missing variant for family {family!r}")
        summary["variants"] = sorted(variants)
        if family == "flash_attn":
            summary["head_topologies"] = [
                {"num_query_heads": query_heads, "num_kv_heads": kv_heads}
                for query_heads, kv_heads in sorted({
                    (
                        sequence["controls"]["num_query_heads"],
                        (
                            sequence["controls"]["num_kv_heads"]
                            if "num_kv_heads" in sequence["controls"] else 1
                        ),
                    )
                    for sequence in scoped
                })
            ]
            summary["head_dims"] = sorted({
                (
                    sequence["controls"]["head_dim"]
                    if "head_dim" in sequence["controls"]
                    else sequence["shape_design"]["fixed_shape"]["qk_head_dim"]
                )
                for sequence in scoped
            })
            summary["kv_layouts"] = sorted({
                sequence["controls"]["kv_layout"] for sequence in scoped
            })
            split_policies = set()
            for sequence in scoped:
                for case in cases_by_sequence[sequence["sequence_id"]]:
                    selection_attributes = case["recipe"]["selection_attributes"]
                    if "split_policy" in selection_attributes:
                        split_policies.add(selection_attributes["split_policy"])
                    else:
                        split_policies.add("source_abi_default")
            summary["split_policies"] = sorted(split_policies)
        elif family in ("fused_moe", "unfused_moe"):
            summary["expert_topologies"] = [
                {
                    "experts": experts,
                    "topk": topk,
                    "hidden": hidden,
                    "intermediate": intermediate,
                }
                for experts, topk, hidden, intermediate in sorted({
                    (
                        sequence["controls"]["experts"],
                        sequence["controls"]["topk"],
                        sequence["controls"]["hidden"],
                        sequence["controls"]["intermediate"],
                    )
                    for sequence in scoped
                })
            ]
            max_tokens_per_expert = set()
            for sequence in scoped:
                for case in cases_by_sequence[sequence["sequence_id"]]:
                    counts = case["recipe"]["prediction_attributes"]["tokens_per_expert"]
                    max_tokens_per_expert.add(max(counts))
            summary["max_tokens_per_expert"] = sorted(max_tokens_per_expert)
        result[family] = summary
    return result


def _coverage_manifest(
    plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
    boundary_cases: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Record source-derived cases and the exact extent sequence evaluated."""

    initial_cases_by_sequence: dict[str, list[Mapping[str, Any]]] = {}
    for case in plan["cases"]:
        sequence_id = case["sequence_id"]
        if sequence_id not in initial_cases_by_sequence:
            initial_cases_by_sequence[sequence_id] = []
        initial_cases_by_sequence[sequence_id].append(case)
    boundary_cases_by_sequence: dict[str, list[Mapping[str, Any]]] = {}
    for case in boundary_cases:
        sequence_id = case["sequence_id"]
        if sequence_id not in boundary_cases_by_sequence:
            boundary_cases_by_sequence[sequence_id] = []
        boundary_cases_by_sequence[sequence_id].append(case)
    cases_by_sequence: dict[str, list[Mapping[str, Any]]] = {}
    for sequence_id, cases in initial_cases_by_sequence.items():
        cases_by_sequence[sequence_id] = list(cases)
    for sequence_id, cases in boundary_cases_by_sequence.items():
        if sequence_id not in cases_by_sequence:
            cases_by_sequence[sequence_id] = []
        cases_by_sequence[sequence_id].extend(cases)

    expected_axis_values: dict[str, set[int]] = {}
    for case in tuple(plan["cases"]) + tuple(boundary_cases):
        sequence_id = case["sequence_id"]
        if sequence_id not in expected_axis_values:
            expected_axis_values[sequence_id] = set()
        expected_axis_values[sequence_id].add(case["axis_value"])
    evaluated_axis_values: dict[str, set[int]] = {}
    for row in rows:
        sequence_id = row["sequence_id"]
        if sequence_id not in evaluated_axis_values:
            evaluated_axis_values[sequence_id] = set()
        evaluated_axis_values[sequence_id].add(row["axis_value"])

    sequences = []
    covered_gemm_sequences = 0
    gemm_sequences = []
    for sequence in plan["sequences"]:
        sequence_id = sequence["sequence_id"]
        expected = expected_axis_values[sequence_id]
        evaluated = evaluated_axis_values[sequence_id]
        if not expected <= evaluated:
            raise RuntimeError(
                f"order-check did not evaluate every planned point for {sequence_id}")
        initial_axis_values = {
            case["axis_value"] for case in initial_cases_by_sequence[sequence_id]
        }
        if sequence_id in boundary_cases_by_sequence:
            boundary_axis_values = {
                case["axis_value"] for case in boundary_cases_by_sequence[sequence_id]
            }
        else:
            boundary_axis_values = set()
        detail = {
            **sequence,
            "expected_axis_values": sorted(expected),
            "evaluated_axis_values": sorted(evaluated),
            "shape_design": _shape_design(
                sequence, cases_by_sequence[sequence_id],
                initial_axis_values, boundary_axis_values),
        }
        sequences.append(detail)
        if sequence["family"] == "gemm":
            gemm_sequences.append(detail)
            covered_gemm_sequences += 1

    nk_pairs = sorted({
        (sequence["controls"]["n"], sequence["controls"]["k"])
        for sequence in gemm_sequences
    })
    physical_combinations = {
        (
            sequence["hardware_id"], sequence["model_name"],
            sequence["tensor_parallel_size"],
            sequence["expert_parallel_size"], sequence["controls"]["dtype"],
            sequence["controls"]["b"], sequence["controls"]["n"],
            sequence["controls"]["k"], sequence["controls"]["variant"],
        )
        for sequence in gemm_sequences
    }
    return {
        "models": plan["models"],
        "base_extents": plan["base_extents"],
        "observed_m_values": plan["observed_m_values"],
        "sequence_count": len(sequences),
        "initial_case_count": len(plan["cases"]),
        "derived_boundary_case_count": len(boundary_cases),
        "family_axis_coverage": _family_axis_coverage(
            sequences, cases_by_sequence),
        "gemm_coverage": {
            "dimension_source": (
                "src.serving.workload.shape_inference.infer_gemm_dimensions"),
            "unique_nk_pair_count": len(nk_pairs),
            "unique_nk_pairs": [
                {"n": n, "k": k}
                for n, k in nk_pairs
            ],
            "unique_physical_combination_count": len(physical_combinations),
            "planned_sequence_count": len(gemm_sequences),
            "evaluated_sequence_count": covered_gemm_sequences,
            "coverage_percent": (
                0.0 if not gemm_sequences
                else 100.0 * covered_gemm_sequences / len(gemm_sequences)),
        },
        "sequences": sequences,
    }


def _trend(delta_ns: float) -> str:
    if delta_ns > 0.0:
        return "increases"
    if delta_ns < 0.0:
        return "decreases"
    return "plateaus"


def _canonical_gemm_pairs(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Join exact GEMM observations at the evaluation boundary only.

    Exactness is intentionally narrow: the complete public GEMM semantic key
    is ``(B, M, K, N, dtype)``.  Canonical durations are never passed to the
    predictor, launch selector, hardware spec, or scan-plan construction.
    """

    indexes: dict[str, dict[tuple[int, int, int, int, str], Mapping[str, Any]]] = {}
    for hardware_id in HARDWARE_IDS:
        index: dict[tuple[int, int, int, int, str], Mapping[str, Any]] = {}
        for case_id, entry in load_kernel_dataset(hardware_id, ("gemm",)).items():
            op_info = entry["op_info"]
            if op_info["Type"] != "GEMM":
                raise ValueError(f"canonical GEMM dataset contains non-GEMM {case_id}")
            attributes = op_info["Attr"]
            key = (
                attributes["B"], attributes["M"], attributes["K"],
                attributes["N"], attributes["dtype"],
            )
            if key in index:
                raise ValueError(
                    f"canonical GEMM dataset has duplicate semantic key for {hardware_id}: {key}")
            timing = parse_kernel_evaluation_timing(
                entry, entry["kernels"], hardware_id)
            index[key] = {
                "case_id": case_id,
                "duration_ns": timing.total_duration_ns,
                "granularity": timing.granularity,
                "source": timing.source,
            }
        indexes[hardware_id] = index

    matched: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["status"] != "success" or row["family"] != "gemm":
            continue
        spec = row["kernel_spec"]
        if spec["kind"] != "GEMM":
            continue
        shape = spec["shape"]
        key = (
            shape["b"], shape["m"], shape["k"], shape["n"], spec["dtype"])
        hardware_id = row["hardware_id"]
        if key not in indexes[hardware_id]:
            continue
        group_key = (row["sequence_id"], row["physical_label"])
        if group_key not in matched:
            matched[group_key] = []
        matched[group_key].append({
            "axis_value": row["axis_value"],
            "kernel_spec": spec,
            "prediction_ns": row["prediction"]["total_ns"],
            "canonical": indexes[hardware_id][key],
            "hardware_id": hardware_id,
            "model_name": row["model_name"],
            "axis": row["axis"],
        })

    pairs = []
    for sequence_id, physical_label in sorted(matched):
        points = sorted(matched[(sequence_id, physical_label)], key=lambda row: row["axis_value"])
        for before, after in zip(points, points[1:]):
            prediction_delta_ns = after["prediction_ns"] - before["prediction_ns"]
            canonical_delta_ns = (
                after["canonical"]["duration_ns"] - before["canonical"]["duration_ns"])
            pairs.append({
                "hardware_id": before["hardware_id"],
                "family": "gemm",
                "model_name": before["model_name"],
                "sequence_id": sequence_id,
                "physical_label": physical_label,
                "axis": before["axis"],
                "work_delta": after["axis_value"] - before["axis_value"],
                "before": before,
                "after": after,
                "prediction_delta_ns": prediction_delta_ns,
                "canonical_delta_ns": canonical_delta_ns,
                "prediction_trend": _trend(prediction_delta_ns),
                "canonical_trend": _trend(canonical_delta_ns),
                "trends_agree": _trend(prediction_delta_ns) == _trend(canonical_delta_ns),
            })
    return pairs


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "hardware_id", "family", "model_name", "tensor_parallel_size",
        "expert_parallel_size", "sequence_id", "physical_label", "axis",
        "axis_value", "status", "duration_ns", "launch_algorithm", "launch_grid",
        "launch_tiling", "unsupported_reason", "error_type", "error",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            result = {name: "" for name in fields}
            for name in (
                    "hardware_id", "family", "model_name", "tensor_parallel_size",
                    "expert_parallel_size", "sequence_id", "physical_label", "axis",
                    "axis_value", "status"):
                result[name] = row[name]
            if row["status"] == "success":
                result["duration_ns"] = row["prediction"]["total_ns"]
                result["launch_algorithm"] = row["launch"]["algorithm"]
                result["launch_grid"] = json.dumps(row["launch"]["grid"])
                result["launch_tiling"] = json.dumps(row["launch"]["tiling"], sort_keys=True)
            elif row["status"] == "unsupported":
                result["unsupported_reason"] = row["unsupported_reason"]
            else:
                result["error_type"] = row["error_type"]
                result["error"] = row["error"]
            writer.writerow(result)


def _diagnostic_change_summary(violation: Mapping[str, Any]) -> str:
    """Summarize the diagnostic component that supports the classification."""

    classification = violation["root_cause"]["classification"]
    diff = violation["prediction_diff"]
    if classification == "launch transition":
        fields = ", ".join(violation["launch_diff"])
        return f"launch fields changed: {fields}"
    if classification == "traffic":
        total_bytes = diff["memory"]["total_bytes"]
        return (
            "memory bytes: "
            f"{total_bytes['before']}→{total_bytes['after']}")
    if classification == "source program transition":
        if violation["root_cause"]["trigger_fields"] == [
                "launch.source_program"]:
            return "launch source program changed"
        before = diff["pipeline"]["before"]
        after = diff["pipeline"]["after"]
        if before is None or after is None:
            raise RuntimeError(
                "source-program cause requires pipeline diagnostics")
        return (
            "source worker programs: "
            f"{len(before['source_schedule'])}→"
            f"{len(after['source_schedule'])}")
    if classification == "pipeline":
        before = diff["pipeline"]["before"]
        after = diff["pipeline"]["after"]
        if before is None or after is None:
            raise RuntimeError("pipeline root cause requires pipeline diagnostics")
        return (
            "pipeline execution: "
            f"{before['execution_time_ns']:.6f}→{after['execution_time_ns']:.6f} ns")
    if classification == "resource/residency":
        changed = diff["time_per_resource_ns"]
        decreased = [
            name for name in sorted(changed)
            if changed[name]["after"] < changed[name]["before"]
        ]
        if not decreased:
            raise RuntimeError("resource root cause requires a decreasing resource")
        name = decreased[0]
        values = changed[name]
        return f"resource {name}: {values['before']:.6f}→{values['after']:.6f} ns"
    if classification == "runtime envelope":
        dispatch = diff["launch_dispatch_ns"]
        return (
            "launch dispatch: "
            f"{dispatch['before']:.6f}→{dispatch['after']:.6f} ns")
    if classification == "input/lowering":
        total = diff["total_ns"]
        return (
            "no isolated component; total: "
            f"{total['before']:.6f}→{total['after']:.6f} ns")
    raise AssertionError(f"unhandled root-cause classification {classification!r}")


def _write_report(
    path: Path, manifest: Mapping[str, Any], summaries: Sequence[Mapping[str, Any]],
    violations: Sequence[Mapping[str, Any]], canonical_pairs: Sequence[Mapping[str, Any]],
    unsupported_records: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Kernel prediction order-check report",
        "",
        f"Commit: `{manifest['commit']}`  ",
        f"Method: `{manifest['method']}`; epsilon: `{manifest['epsilon_ns']} ns`.",
        "",
        "## Coverage and outcomes",
        "",
        "| Hardware | Family | Points | Sequences | Success | Unsupported | Failed | Violations | Max drop (ns) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['hardware_id']} | {row['family']} | {row['point_count']} | "
            f"{row['sequence_count']} | {row['success_count']} | "
            f"{row['unsupported_count']} | {row['failed_count']} | "
            f"{row['violation_count']} | {row['maximum_drop_ns']:.6f} |")
    lines.extend([
        "",
        "## Unsupported / fail-closed domains",
        "",
        "Unsupported is a fail-closed launch/predictor domain, not a monotonicity pass. Every such case, including its source-derived recipe, controls, and reason, is retained in `unsupported.json`.",
        "",
        "| Hardware | Family | Fail-closed reason | Inputs |",
        "| --- | --- | --- | ---: |",
    ])
    for row in _unsupported_summaries(unsupported_records):
        lines.append(
            f"| {row['hardware_id']} | {row['family']} | `{row['reason']}` | "
            f"{row['count']} |")
    lines.extend([
        "",
        "## Serving GEMM source coverage",
        "",
    ])
    gemm_coverage = manifest["coverage_manifest"]["gemm_coverage"]
    lines.extend([
        f"Dimension source: `{gemm_coverage['dimension_source']}`.",
        "",
        "| Unique N/K pairs | Unique physical combinations | Planned sequences | Evaluated sequences | Coverage |",
        "| ---: | ---: | ---: | ---: | ---: |",
        f"| {gemm_coverage['unique_nk_pair_count']} | "
        f"{gemm_coverage['unique_physical_combination_count']} | "
        f"{gemm_coverage['planned_sequence_count']} | "
        f"{gemm_coverage['evaluated_sequence_count']} | "
        f"{gemm_coverage['coverage_percent']:.1f}% |",
        "",
        "`coverage_manifest.json` records each source model, deployment TP/EP, raw N/K facts, and the planned/evaluated M sequence for every combination.",
        "",
        "## Other kernel axis coverage",
        "",
        "| Family | Sequences | Axes | Variants | Source models |",
        "| --- | ---: | --- | --- | ---: |",
    ])
    for family in FAMILIES:
        family_coverage = manifest["coverage_manifest"]["family_axis_coverage"]
        if family not in family_coverage:
            continue
        summary = family_coverage[family]
        lines.append(
            f"| {family} | {summary['sequence_count']} | "
            f"{', '.join(summary['axes'])} | {', '.join(summary['variants'])} | "
            f"{len(summary['source_models'])} |")
    lines.extend([
        "",
        "The coverage manifest records per-sequence fixed and varying shape fields, source lowering, derived boundaries, FA topology/layout, MoE expert/top-k/hidden/intermediate variants and routed maximum tokens per expert.",
        "",
        "## Exact canonical GEMM trends",
        "",
    ])
    if not canonical_pairs:
        lines.append("No pair of evaluated GEMM points has an exact canonical semantic match.")
    else:
        lines.extend([
            "These evaluation-only observations match `(B, M, K, N, dtype)` exactly; they do not influence Forward.",
            "",
            "| Hardware | Model / sequence | M | Predicted trend | Canonical trend | Agreement |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for pair in canonical_pairs:
            before = pair["before"]
            after = pair["after"]
            lines.append(
                f"| {pair['hardware_id']} | {pair['model_name']} / {pair['sequence_id']} | "
                f"{before['axis_value']}→{after['axis_value']} | "
                f"{before['prediction_ns']:.6f}→{after['prediction_ns']:.6f} ns ({pair['prediction_trend']}) | "
                f"{before['canonical']['duration_ns']:.6f}→{after['canonical']['duration_ns']:.6f} ns ({pair['canonical_trend']}) | "
                f"{'yes' if pair['trends_agree'] else 'no'} |")
    lines.extend([
        "",
        "Detailed canonical evidence is in `canonical_pairs.json`.",
        "",
        "## Prediction decreases",
        "",
    ])
    if not violations:
        lines.append("No supported adjacent partial-order pair decreased beyond the numerical epsilon.")
    else:
        lines.extend([
            "| Hardware | Family | Kernel | Axis | Before shape/time | After shape/time | Delta (ns) | Drop | Launch change | Diagnostic component change | Root cause |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
        ])
        for row in violations:
            before_shape = json.dumps(row["before_kernel_spec"]["shape"], sort_keys=True)
            after_shape = json.dumps(row["after_kernel_spec"]["shape"], sort_keys=True)
            launch = ", ".join(row["launch_diff"])
            if not launch:
                launch = "none"
            cause = row["root_cause"]
            diagnostic = _diagnostic_change_summary(row)
            lines.append(
                f"| {row['hardware_id']} | {row['family']} | {row['physical_label']} | "
                f"{row['axis']} +{row['work_delta']} | `{before_shape}` / {row['before_ns']:.6f} | "
                f"`{after_shape}` / {row['after_ns']:.6f} | {row['delta_ns']:.6f} | "
                f"{row['drop_pct']:.4f}% | {launch} | {diagnostic} | "
                f"{cause['classification']}: {cause['source_function']} |")
    lines.extend([
        "",
        "Each detailed pair, including complete before/after KernelSpec, normalized launch, model-component diff, and cause evidence, is in `violations.json`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_order_check(
    *, output_dir: Path, hardware_ids: Sequence[str] = HARDWARE_IDS,
    families: Sequence[str] = FAMILIES, model_names: Sequence[str] = (),
    jobs: int = 48, scan_size: str = "full", epsilon_ns: float = DEFAULT_EPSILON_NS,
    root: Path = REPOSITORY_ROOT,
) -> Mapping[str, Any]:
    """Execute a deterministic, multiprocess scan and write report artifacts."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_scan_plan(
        root=root, hardware_ids=hardware_ids, families=families,
        model_names=model_names, scan_size=scan_size)
    initial_cases = plan["cases"]
    rows = _run_cases(initial_cases, jobs)
    boundary_cases = _gemm_boundary_cases(rows, initial_cases, scan_size)
    if boundary_cases:
        rows = sorted(rows + _run_cases(boundary_cases, jobs), key=_result_sort_key)
    violations = compare_ordered_points(rows, epsilon_ns=epsilon_ns)
    canonical_pairs = _canonical_gemm_pairs(rows)
    unsupported_records = _unsupported_records(rows)
    summaries = _summary_rows(rows, violations)
    coverage_manifest = _coverage_manifest(plan, rows, boundary_cases)
    manifest = {
        "schema_version": 4,
        "commit": _git_commit(root),
        "method": "midlayer",
        "epsilon_ns": epsilon_ns,
        "jobs": jobs,
        "scan_size": scan_size,
        "hardware": {
            hardware_id: {
                "config": str(Path(get_hardware_config_path(hardware_id)).relative_to(root)),
                "sha256": _sha256(Path(get_hardware_config_path(hardware_id))),
            }
            for hardware_id in hardware_ids
        },
        "coverage_manifest": coverage_manifest,
        "counts": {
            "case_results": len(rows),
            "sequence_results": len({(row["sequence_id"], row["physical_label"]) for row in rows}),
            "success": _counts(rows)["success"],
            "unsupported": _counts(rows)["unsupported"],
            "unsupported_reason_groups": len(_unsupported_summaries(unsupported_records)),
            "failed": _counts(rows)["failed"],
            "violations": len(violations),
            "canonical_pairs": len(canonical_pairs),
        },
        "summary": summaries,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "coverage_manifest.json", manifest["coverage_manifest"])
    _write_csv(output_dir / "all_sequences.csv", rows)
    _write_json(output_dir / "violations.json", violations)
    _write_json(output_dir / "canonical_pairs.json", canonical_pairs)
    _write_json(output_dir / "unsupported.json", unsupported_records)
    _write_report(
        output_dir / "report.md", manifest, summaries, violations, canonical_pairs,
        unsupported_records)
    return manifest
