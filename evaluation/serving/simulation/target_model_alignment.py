"""Build source-pinned actual-to-Midlayer target-model alignments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.serving.simulation.execution_artifact import (
    structural_lowering_manifest,
)
from src.kernel.hardware_spec import HWSpec
from src.serving.deployment.deployment_config import (
    DeploymentConfig,
    WorkerPoolConfig,
)
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.kernel_lowering.lower_workload import lower_workload
from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest
from src.serving.workload.traced_models import trace_target_model_structural_forward
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import (
    ExpertRoutingInput,
)


_MODEL_SPECS = {
    "glm-5.2": {
        "run_key": "glm_5_2",
        "upstream_model_id": "zai-org/GLM-5.2",
        "dense_layer": 0,
        "moe_layer": 3,
        "official_config_sha256": (
            "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a"
        ),
    },
    "kimi-k2.6": {
        "run_key": "kimi_k2_6",
        "upstream_model_id": "moonshotai/Kimi-K2.6",
        "dense_layer": 0,
        "moe_layer": 1,
        "official_config_sha256": (
            "85825ca6e18cbe539eb83ee09eedfb3f4222265929f06e9f535a6d9364f55899"
        ),
    },
}


def load_target_model_validation_run(path: str | Path) -> Mapping[str, object]:
    """Load and identify one immutable A800 target-model validation run."""

    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError("target-model validation run must be a regular file")
    source_bytes = source.read_bytes()
    try:
        payload = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("target-model validation run must be UTF-8 JSON") from error
    run = _mapping(payload, "target-model validation run")
    _validate_run(run)
    return {
        "payload": run,
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def build_target_model_alignment(
    canonical_model_id: str,
    validation_run: Mapping[str, object],
    *,
    validation_run_sha256: str,
    repository_root: str | Path,
) -> Mapping[str, object]:
    """Build Prefill/Decode partial alignment for one fixed target model."""

    if canonical_model_id not in _MODEL_SPECS:
        raise ValueError(f"unsupported target model: {canonical_model_id!r}")
    _sha256(validation_run_sha256, "validation run SHA256")
    run = _mapping(validation_run, "target-model validation run")
    _validate_run(run)
    spec = _mapping(_MODEL_SPECS[canonical_model_id], "target model spec")
    run_key = spec["run_key"]
    model_run = _mapping(run[run_key], f"validation run {run_key}")
    root = Path(repository_root)
    model = load_model_config(root / "configs/model" / f"{canonical_model_id}.json")
    if model.model_id != spec["upstream_model_id"]:
        raise ValueError("Midlayer model ID disagrees with target-model identity")
    signatures: dict[str, Mapping[str, object]] = {}
    phases = {}
    for execution_order, phase in enumerate(("prefill", "decode")):
        artifact = _build_structural_artifact(
            canonical_model_id,
            phase,
            model,
            root,
        )
        samples = []
        for layer_role, actual_layer, midlayer_layer in (
            ("dense", 0, spec["dense_layer"]),
            ("moe", 1, spec["moe_layer"]),
        ):
            sample = _build_sample(
                canonical_model_id=canonical_model_id,
                phase=phase,
                execution_order=execution_order,
                layer_role=layer_role,
                actual_layer=actual_layer,
                midlayer_layer=midlayer_layer,
                model_run=model_run,
                artifact=artifact,
                signatures=signatures,
                validation_run_sha256=validation_run_sha256,
            )
            samples.append(sample)
        phases[phase] = {"status": "available", "samples": samples}
    phases["mixed"] = {
        "status": "not_observed",
        "reason": (
            "the fixed A800 Transformers slice executes isolated Prefill and "
            "Decode; mixed Scheduler batching remains outside this partial "
            "actual-execution sample"
        ),
    }
    source_identity = _source_identity(
        canonical_model_id,
        model.config_sha256,
        run,
        validation_run_sha256,
    )
    return {
        "schema_version": 4,
        "kind": "sharegpt_alignment_steps",
        "source_identity": source_identity,
        "operator_signatures": signatures,
        "launch_signatures": {},
        "phases": phases,
    }


def _build_structural_artifact(
    canonical_model_id: str,
    phase: str,
    model,
    repository_root: Path,
) -> Mapping[str, object]:
    deployment = build_deployment_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig("main", "colocated", "nvidia_a800"),),
        ),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    if phase == "prefill":
        requests = (ScheduledRequest("actual-slice", "prefill", 3, 0),)
    elif phase == "decode":
        requests = (ScheduledRequest("actual-slice", "decode", 0, 1, 3),)
    else:
        raise ValueError(f"unsupported target-model phase: {phase!r}")
    schedule = ScheduleStep(
        f"alignment-{canonical_model_id}-{phase}",
        "main",
        requests,
        attention_kv_layout="mla",
    )
    spec = _MODEL_SPECS[canonical_model_id]
    source = trace_target_model_structural_forward(model, schedule,
        deployment.worker_pools["main"], ExpertRoutingInput.seeded_assumption(
            f"target_model_alignment_{canonical_model_id}_{phase}_v1", 0),
        hidden_shape=(schedule.token_count, model.hidden_size), hidden_dtype="bf16",
        representative_layers=(spec["dense_layer"], spec["moe_layer"]))
    graph = source.graph
    plan = lower_workload(
        graph,
        deployment,
        {
            "nvidia_a800": HWSpec(
                repository_root / "configs/hardware/nvidia_a800.yaml"),
        },
        source_projection_scope="a800_tp1_target_representative_layers_v1",
    )
    return {**structural_lowering_manifest(schedule, graph, plan),
        "source_scope": source.source_scope}


def _build_sample(
    *,
    canonical_model_id: str,
    phase: str,
    execution_order: int,
    layer_role: str,
    actual_layer: int,
    midlayer_layer: int,
    model_run: Mapping[str, object],
    artifact: Mapping[str, object],
    signatures: dict[str, Mapping[str, object]],
    validation_run_sha256: str,
) -> Mapping[str, object]:
    event_key = f"{phase}_events"
    source_events = _sequence(model_run[event_key], f"{event_key}")
    module_prefix = f"model.layers.{actual_layer}."
    selected_events = tuple(
        _mapping(value, f"{event_key} event")
        for value in source_events
        if _mapping(value, f"{event_key} event")["module"].startswith(
            module_prefix)
    )
    if not selected_events:
        raise ValueError(
            f"actual execution has no {phase} events for layer {actual_layer}")
    actual_events = []
    for order, event in enumerate(selected_events):
        signature = {
            "operation": event["class"],
            "attrs": {"module": event["module"]},
        }
        signature_id = _canonical_sha256(signature)
        signatures[signature_id] = signature
        actual_events.append({
            "order": order,
            "signature_id": signature_id,
            "layer": actual_layer,
            "rank": 0,
            "inputs": event["inputs"],
            "output": event["output"],
        })
    graph = _mapping(artifact["workload_graph"], "structural workload graph")
    prefix = f"alignment-{canonical_model_id}-{phase}:layer-{midlayer_layer}:"
    operators = tuple(
        _mapping(value, "structural workload operator")
        for value in _sequence(graph["nodes"], "structural workload nodes")
        if _mapping(value, "structural workload operator")["node_id"].startswith(
            prefix)
    )
    plan = _mapping(artifact["kernel_plan"], "structural kernel plan")
    calls = tuple(
        _mapping(value, "structural KernelCall")
        for value in _sequence(plan["calls"], "structural KernelCalls")
        if any(
            source.startswith(prefix)
            for source in _sequence(
                _mapping(value, "structural KernelCall")["source_node_ids"],
                "structural KernelCall source nodes",
            )
        )
    )
    if not operators or not calls:
        raise ValueError("target-model alignment selected an empty Midlayer layer")
    _verify_required_semantics(
        canonical_model_id,
        phase,
        layer_role,
        actual_events,
        operators,
        calls,
        model_run,
    )
    contracts = {
        "logits": model_run[f"{phase}_logits"],
        "cache_class": model_run["cache_class"],
        "cache_sequence_length_after_decode": model_run["cache_sequence_length"],
        "cache_layer_0_key_shape_after_decode": model_run[
            "cache_layer_0_key_shape"],
    }
    if canonical_model_id == "glm-5.2":
        contracts.update({
            "indexer_modes": model_run["indexer_modes"],
            "full_indexer_cache_shape": model_run[
                "full_indexer_cache_shape"],
            "shared_indexer_cache_is_empty": model_run[
                "shared_indexer_cache_is_empty"],
        })
    unsupported = [
        {
            "capability": "cross_version_source_equivalence",
            "reason": (
                "the retained execution/source receipt is Transformers plus vLLM0.27.1; "
                "the shared virtual representative layers adapt vLLM0.30. "
                "This artifact proves typed structural lowering only, without asserting "
                "equal instructions, storage, launch selection or numerical output across versions"
            ),
        },
        {
            "capability": "native_vllm_cuda_execution",
            "reason": (
                "fixed vLLM 0.27.1 requires PyTorch 2.13.0 while the isolated "
                "A800 execution environment uses PyTorch 2.7.1+cu126; vLLM "
                "implementation is source-audited and Transformers supplies "
                "the executed control-flow slice"
            ),
        },
        {
            "capability": "hardware_launch_alignment",
            "reason": (
                "the actual slice records module hooks and tensor contracts, "
                "not profiler launch observations; analytical launches remain "
                "Midlayer structural output only"
            ),
        },
    ]
    if canonical_model_id == "glm-5.2":
        unsupported.append({
            "capability": "mtp_actual_upstream_execution",
            "reason": (
                "fixed vLLM 0.27.1 has no registered GLM-Moe-DSA MTP model; "
                "Midlayer's source-backed Scheduler proposer contract is not "
                "represented as an executed upstream layer in this sample"
            ),
        })
        if phase == "prefill":
            unsupported.append({
                "capability": "prefill_attention_branch_equivalence",
                "reason": (
                    "the0.30 DeepseekV32Attention source takes fresh dense MHA for three "
                    "uncached rows within index_topk, including KV expansion and output copy; "
                    "the previous artifact emitted dsa_sparse_prefill. The retained0.27.1 "
                    "receipt does not establish equal branch workload or storage"
                ),
            })
    sample = {
        "sample_id": (
            f"{canonical_model_id}-{phase}-{layer_role}-layer-{midlayer_layer}"
        ),
        "phase": phase,
        "scope": {
            "coverage": "partial",
            "actual_layers": [actual_layer],
            "midlayer_layers": [midlayer_layer],
            "rank_ids": [0],
            "subgraphs": _subgraphs(canonical_model_id, layer_role),
            "omitted": [
                "all non-representative decoder layers",
                "mixed-phase Scheduler batching",
                "multi-rank actual CUDA execution",
                "production weights and numerical output equality",
                "hardware launch capture and duration prediction accuracy",
            ],
        },
        "actual": {
            "framework": "transformers",
            "execution_mode": "a800_cuda_bf16",
            "events": actual_events,
            "contracts": contracts,
        },
        "midlayer": {
            "model_id": _MODEL_SPECS[canonical_model_id]["upstream_model_id"],
            "phase": phase,
            "layer": midlayer_layer,
            "rank": 0,
            "source_scope": artifact["source_scope"],
            "schedule": _mapping(
                artifact["schedule_step"], "structural schedule step"),
            "operators": operators,
            "kernel_calls": calls,
        },
        "comparisons": _comparisons(
            canonical_model_id, phase, layer_role, operators, calls),
        "differences": [],
        "unsupported": unsupported,
        "provenance": {
            "actual_execution": (
                f"validation_run_sha256:{validation_run_sha256}"
            ),
            "production_shape_source": (
                "fixed official config plus audited Transformers/vLLM source"
            ),
            "midlayer_generation": (
                "trace_target_model_structural_forward->lower_workload(structural_scope)->structural_lowering_manifest"
            ),
        },
    }
    return {
        **sample,
        "content_sha256": _canonical_sha256(sample),
    }


def _verify_required_semantics(
    canonical_model_id: str,
    phase: str,
    layer_role: str,
    actual_events: Sequence[Mapping[str, object]],
    operators: Sequence[Mapping[str, object]],
    calls: Sequence[Mapping[str, object]],
    model_run: Mapping[str, object],
) -> None:
    variants = {value["variant"] for value in operators}
    kernel_specs = tuple(
        _mapping(value["kernel_spec"], "target-model KernelSpec")
        for value in calls
    )
    kernel_variants = {value["variant"] for value in kernel_specs}
    required_variants = set()
    if canonical_model_id == "kimi-k2.6":
        required_variants.add("mla_kv_cache")
        required_variants.add(f"mla_{phase}")
    elif canonical_model_id == "glm-5.2":
        required_variants.update({"dsv32_fused_norm_rope", "dsv32_fused_q"})
        required_variants.add("mla_prefill" if phase == "prefill" else "dsa_sparse_decode")
        if layer_role == "dense":
            required_variants.add(f"dsa_indexer_{phase}")
        elif layer_role == "moe":
            if any(value["variant"].startswith("dsa_indexer_") for value in operators):
                raise ValueError("GLM shared representative layer must reuse the existing top-k buffer")
        else:
            raise AssertionError(f"unsupported target layer role: {layer_role}")
        norm, = (value for value in operators if value["variant"] == "dsv32_fused_norm_rope")
        if norm["attributes"]["has_indexer"] != (layer_role == "dense"):
            raise ValueError("GLM representative layer lost its full/shared fused cache branch")
    else:
        raise AssertionError(f"unsupported target model: {canonical_model_id}")
    missing = required_variants - (variants | kernel_variants)
    if missing:
        raise ValueError(
            f"Midlayer target-model slice lacks required variants: {sorted(missing)}")
    if phase == "prefill":
        token_count = 3
    elif phase == "decode":
        token_count = 1
    else:
        raise AssertionError(f"unsupported target phase: {phase}")
    if layer_role == "moe":
        expert_events = tuple(
            value for value in actual_events
            if _mapping(value, "actual MoE event")["inputs"]
            and len(_mapping(value, "actual MoE event")["inputs"]) == 3
        )
        if len(expert_events) != 1:
            raise ValueError("actual MoE slice requires one routed-expert event")
        inputs = _sequence(expert_events[0]["inputs"], "actual expert inputs")
        hidden = _mapping(inputs[0], "actual routed hidden rows")
        topk_ids = _mapping(inputs[1], "actual routed top-k IDs")
        if hidden["shape"][0] != token_count or topk_ids["shape"] != [token_count, 2]:
            raise ValueError("actual MoE routed rows disagree with the tiny config")
        routed_specs = tuple(
            value for value in kernel_specs
            if value["kind"] == "FUSED_MOE"
        )
        if not routed_specs:
            raise ValueError("Midlayer MoE slice lacks a FUSED_MOE KernelSpec")
        if any(value["shape"]["tokens"] != token_count for value in routed_specs):
            raise ValueError("Midlayer MoE routed rows disagree with phase tokens")
        if canonical_model_id == "kimi-k2.6" and not any(
            value["attributes"]["routed_weight_dtype"] == "int4"
            for value in routed_specs
        ):
            raise ValueError("Kimi routed-expert KernelSpec lost INT4 weights")
    if canonical_model_id == "glm-5.2":
        if model_run["indexer_modes"] != ["full", "shared"]:
            raise ValueError("actual GLM slice lost full/shared indexer control flow")
        if model_run["shared_indexer_cache_is_empty"] is not True:
            raise ValueError("actual GLM shared indexer unexpectedly owns a cache")
    if any(value["execution_support"] != "unsupported" or value["launch"] is not None for value in kernel_specs):
        raise ValueError("target structural projection must not claim actual provider/launch support")


def _comparisons(
    canonical_model_id: str,
    phase: str,
    layer_role: str,
    operators: Sequence[Mapping[str, object]],
    calls: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    operator_ids = ",".join(value["node_id"] for value in operators)
    call_ids = ",".join(value["call_id"] for value in calls)
    values = [
        {
            "mechanism": "phase_and_dynamic_token_rows",
            "actual_evidence": (
                f"A800 {phase} hooks preserve the executed token row count"
            ),
            "midlayer_evidence": operator_ids,
            "status": "matched",
        },
        {
            "mechanism": "logical_operator_to_kernel_spec_lowering",
            "actual_evidence": (
                "fixed upstream module class and tensor hook sequence"
            ),
            "midlayer_evidence": call_ids,
            "status": "unsupported",
        },
        {
            "mechanism": "compressed_mla_kv_cache",
            "actual_evidence": (
                "fixed vLLM MLA source plus executed Transformers cache growth"
            ),
            "midlayer_evidence": "KV_CACHE/mla_kv_cache",
            "status": "matched",
        },
    ]
    if canonical_model_id == "glm-5.2":
        if layer_role == "dense":
            index_variant = "dsa_indexer_" + phase
        elif layer_role == "moe":
            index_variant = ("shared indexer elided; fresh dense-MHA backend does not consume top-k"
                if phase == "prefill" else "shared top-k tensor dependency; no separate indexer launch")
        else:
            raise AssertionError(f"unsupported target layer role: {layer_role}")
        values.append({
            "mechanism": "dsa_full_or_shared_indexer",
            "actual_evidence": (
                "executed GLM full/shared indexer modes and fixed sparse MLA source"
            ),
            "midlayer_evidence": index_variant,
            "status": "matched",
        })
        if layer_role == "dense":
            values.append({
                "mechanism": "dsa_fp8_index_cache",
                "actual_evidence": (
                    "retained0.27.1 index-cache source receipt;0.30 fused norm/cache needs separate cross-version comparison"
                ),
                "midlayer_evidence": "VECTOR/dsv32_fused_norm_rope has_indexer=True; typed FP8 cache backing",
                "status": "unsupported",
            })
    elif canonical_model_id == "kimi-k2.6":
        values.append({
            "mechanism": "mla_attention",
            "actual_evidence": (
                "executed DeepseekV3Attention control flow and fixed vLLM MLA source"
            ),
            "midlayer_evidence": f"FLASH_ATTN/mla_{phase}",
            "status": "unsupported",
        })
    else:
        raise AssertionError(f"unsupported target model: {canonical_model_id}")
    if layer_role == "moe":
        values.append({
            "mechanism": "moe_routed_rows",
            "actual_evidence": (
                "actual expert hook receives [tokens,hidden], [tokens,topk] "
                "IDs, and [tokens,topk] weights"
            ),
            "midlayer_evidence": "FUSED_MOE KernelSpec tokens/topk shape",
            "status": "matched",
        })
        if canonical_model_id == "kimi-k2.6":
            values.append({
                "mechanism": "int4_routed_expert_weights",
                "actual_evidence": (
                    "fixed official compressed-tensors group_0 Linear config"
                ),
                "midlayer_evidence": "FUSED_MOE attributes.weight_dtype=int4",
                "status": "matched",
            })
    return values


def _subgraphs(canonical_model_id: str, layer_role: str) -> list[str]:
    values = [
        "attention projections and rotary",
        "compressed MLA KV cache update",
        "attention prefill/decode branch",
        "output projection and residual norms",
    ]
    if canonical_model_id == "glm-5.2":
        if layer_role == "dense":
            values.append("DSA full indexer")
        elif layer_role == "moe":
            values.append("DSA shared-index reuse")
        else:
            raise AssertionError(f"unsupported target layer role: {layer_role}")
    elif canonical_model_id == "kimi-k2.6":
        pass
    else:
        raise AssertionError(f"unsupported target model: {canonical_model_id}")
    if layer_role == "moe":
        values.extend(("MoE routing and routed rows", "shared expert path"))
    elif layer_role == "dense":
        values.append("dense MLP path")
    else:
        raise AssertionError(f"unsupported target layer role: {layer_role}")
    return values


def _source_identity(
    canonical_model_id: str,
    midlayer_config_sha256: str,
    run: Mapping[str, object],
    validation_run_sha256: str,
) -> Mapping[str, object]:
    spec = _mapping(_MODEL_SPECS[canonical_model_id], "target model spec")
    source_checkouts = _mapping(run["source_checkouts"], "validation sources")
    model_checkout = _mapping(
        source_checkouts[spec["run_key"]], "model checkout")
    transformers_checkout = _mapping(
        source_checkouts["transformers"], "Transformers checkout")
    vllm_checkout = _mapping(source_checkouts["vllm"], "vLLM checkout")
    model_files = ({
        "relative_path": "config.json",
        "sha256": model_checkout["config_sha256"],
    },)
    return {
        "canonical_model_id": canonical_model_id,
        "model_id": spec["upstream_model_id"],
        "model_revision": model_checkout["commit"],
        "official_config_sha256": spec["official_config_sha256"],
        "midlayer_config_sha256": midlayer_config_sha256,
        "midlayer_baseline": run["midlayer_baseline"],
        "validation_run_sha256": validation_run_sha256,
        "environment": {
            "device": run["device"],
            "cuda_visible_devices": run["cuda_visible_devices"],
            "python": run["python"],
            "torch": run["torch"],
            "transformers": _mapping(
                run["transformers"], "installed Transformers")["version"],
        },
        "sources": {
            "model": {
                "commit": model_checkout["commit"],
                "version": None,
                "execution_status": "config_loaded_and_transformers_model_executed",
                "files": model_files,
            },
            "transformers": {
                "commit": transformers_checkout["commit"],
                "version": _mapping(
                    run["transformers"], "installed Transformers")["version"],
                "execution_status": "a800_cuda_prefill_decode_executed",
                "files": transformers_checkout["files"],
            },
            "vllm": {
                "commit": vllm_checkout["commit"],
                "version": "0.27.1",
                "execution_status": (
                    "source_audited_not_executed_pytorch_2_13_requirement"
                ),
                "files": vllm_checkout["files"],
            },
        },
    }


def _validate_run(run: Mapping[str, object]) -> None:
    required = {
        "schema_version",
        "pid",
        "midlayer_baseline",
        "python",
        "torch",
        "transformers",
        "device",
        "cuda_visible_devices",
        "source_checkouts",
        "scope",
        "glm_5_2",
        "kimi_k2_6",
    }
    if set(run) != required:
        raise ValueError("target-model validation run fields are incomplete")
    if run["schema_version"] != 1:
        raise ValueError("target-model validation run schema_version must be 1")
    if run["device"] != "NVIDIA A800-SXM4-80GB":
        raise ValueError("target-model validation run requires an A800")
    _nonempty_string(run["cuda_visible_devices"], "CUDA_VISIBLE_DEVICES")
    _git_commit(run["midlayer_baseline"], "Midlayer baseline")
    transformers = _mapping(run["transformers"], "installed Transformers")
    if transformers["version"] != "5.10.2":
        raise ValueError("target-model validation requires Transformers 5.10.2")
    checkouts = _mapping(run["source_checkouts"], "source checkouts")
    expected_commits = {
        "transformers": "0dad7b822255a0ae261ec45ae937371e859ffd1a",
        "vllm": "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac",
        "glm_5_2": "b4734de4facf877f85769a911abafc5283eab3d9",
        "kimi_k2_6": "7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
    }
    for name, expected in expected_commits.items():
        checkout = _mapping(checkouts[name], f"source checkout {name}")
        if checkout["commit"] != expected:
            raise ValueError(f"source checkout {name} commit is not pinned")
    for run_key in ("glm_5_2", "kimi_k2_6"):
        model = _mapping(run[run_key], f"validation model {run_key}")
        if model["cache_sequence_length"] != 4:
            raise ValueError(f"validation model {run_key} cache did not grow to 4")
        for phase in ("prefill", "decode"):
            events = _sequence(model[f"{phase}_events"], f"{run_key} {phase} events")
            if not events:
                raise ValueError(f"validation model {run_key} lacks {phase} hooks")


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be an object")
    return value


def _sequence(value: object, description: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{description} must be an array")
    return value


def _nonempty_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")
    return value


def _sha256(value: object, description: str) -> str:
    text = _nonempty_string(value, description)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{description} must be a lowercase SHA256")
    return text


def _git_commit(value: object, description: str) -> str:
    text = _nonempty_string(value, description)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{description} must be a full lowercase Git commit")
    return text


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
