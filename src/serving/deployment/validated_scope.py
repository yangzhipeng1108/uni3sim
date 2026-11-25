"""The validated target-model deployment scopes as one declarative table.

Three production gates historically restated overlapping subsets of the same
validated deployment facts:

* ``is_validated_target_construction_scope`` -- the public target logical
  construction scope, separate from mathematical forward support;
* ``src.serving.simulation.validate_workload_emission_parallelism`` -- the
  rank-local workload/kernel emission gate;
* ``src.serving.backend.vllm._validate_native_scheduler_scope`` -- the vLLM
  Scheduler control-plane gate.

The gates do not enforce identical granularity.  The differences are
recorded here as explicit per-row fields instead of being re-derived at each
call site; this module documents the drift without adjudicating it:

* The construction gate predates the hardware pins on the A800 production
  and PD EP8 rows: it accepts those topologies on any hardware and never
  inspects the compute dtype (``construction_checks_hardware=False``).  The
  H200 rows carry hardware pins in every gate.
* The emission gate is the only gate that sees pools, so it alone pins the
  pool size (ranks/workers == TP*PP), requires BF16 compute, and gates the
  A800 production rows behind the caller's ``allow_single_pool_tp1_pp``
  flag.  It also derives the DeepSeek-V3 aggregate TP from the config
  identity: the 32-layer variant is TP4-only at emission, while the
  construction gate accepts that identity at TP8 as well.
* The Scheduler gate knows roles instead of deployments or hardware: it pins
  the five pinned config identities (``TARGET_MODEL_CONFIG_SHA256``) and its
  rows deliberately over-approximate -- the PP2/PP4 rows accept every named
  target model in any role, and the PD EP8 row accepts every named target
  model.  Combinations no other gate validates are failed closed afterwards
  by the narrower per-model checks in ``_construct_native_scheduler`` and
  ``_validate_ep_scheduler_model_scope``.
"""

from dataclasses import dataclass

from .deployment_plan import DeploymentPlan
from .deployment_config import worker_role_operator_group, worker_role_phase

from src.serving.workload.model_config import (
    DEEPSEEK_V3_L32_CONFIG_SHA256,
    DEEPSEEK_V4_FLASH_CONFIG_SHA256,
    DEEPSEEK_V4_PRO_CONFIG_SHA256,
    GLM_MOE_DSA_CONFIG_SHA256,
    ModelConfig,
    is_named_target_model_identity,
    is_validated_qwen3_moe_ep2_model_scope,
    is_validated_llama_tp2_model_scope,
    is_validated_qwen3_tp2_host_kv_model_scope,
    is_dense_parallel_model_scope,
    is_qwen3_moe_source_model_scope,
    is_kimi_k2_parallel_model_scope,
)


def is_deepseek_parallel_source_scope(pool, model, *, model_execution_version, symbolic_forward_source, graph_enabled):
    """Versioned complete V3/V4 target TP/DP/EP/PP source geometry only.

    This admits source construction, without accepting operator prices,
    native GPU workers, normal memory budgets or model accuracy.
    """
    from src.serving.workload.model_config import (is_deepseek_v3_parallel_model_scope,
        is_deepseek_v3_data_parallel_model_scope, is_deepseek_v4_parallel_model_scope,
        is_deepseek_v4_data_parallel_model_scope)
    if not isinstance(model, ModelConfig) or type(graph_enabled) is not bool:
        return False
    config = pool.parallel_plan.config
    if (graph_enabled or pool.hardware_id != "nvidia_h200" or pool.role != "colocated"
            or len(pool.ranks) != config.worker_count or config.context_parallel_size != 1
            or config.sequence_parallel_size != 1):
        return False
    if model.model_type == "deepseek_v3":
        if model_execution_version != "vllm_0_27_1" or symbolic_forward_source != "vllm_0_27_1_deepseek_v3_sync_fp8":
            return False
        local, distributed = is_deepseek_v3_parallel_model_scope, is_deepseek_v3_data_parallel_model_scope
    elif model.model_type == "deepseek_v4":
        if model_execution_version != "vllm_0_30_0" or symbolic_forward_source != "vllm_0_30_v4_sync_fp8_ds_mla":
            return False
        local, distributed = is_deepseek_v4_parallel_model_scope, is_deepseek_v4_data_parallel_model_scope
    else:
        return False
    return (distributed(model, config) if config.data_parallel_size > 1 else
        local(model, config.tensor_parallel_size, config.pipeline_parallel_size, expert_parallel=config.enable_expert_parallel))


def is_kimi_k3_runtime_source_scope(pool, model, runtime_inputs):
    """K3 ordinary source geometry and explicit tier, not GPU acceptance."""
    from src.serving.workload.runtime_inputs import RuntimeInputs
    from src.serving.workload.model_config import is_kimi_k3_parallel_model_scope, is_kimi_k3_data_parallel_model_scope
    if not isinstance(runtime_inputs, RuntimeInputs) or not isinstance(model, ModelConfig):
        return False
    config = pool.parallel_plan.config
    if (model.model_type!='kimi_linear' or model.kimi_k3_semantics is None
            or pool.hardware_id != "nvidia_h200" or pool.role != "colocated" or len(pool.ranks) != config.worker_count
            or config.context_parallel_size != 1 or config.sequence_parallel_size != 1
            or pool.parallel_plan.num_layers != model.num_hidden_layers or pool.parallel_plan.num_experts != model.num_experts
            or len(pool.parallel_plan.layer_to_pipeline_stage) != model.num_hidden_layers
            or pool.parallel_plan.layer_to_pipeline_stage != tuple(sorted(pool.parallel_plan.layer_to_pipeline_stage))
            or set(pool.parallel_plan.layer_to_pipeline_stage) != set(range(config.pipeline_parallel_size))
            or runtime_inputs.model_execution_version != "vllm_0_30_0"
            or runtime_inputs.symbolic_forward_source != "vllm_0_30_k3_mxfp4_sync"
            or runtime_inputs.k3_kda_decode_backend=='native' and model.kimi_k3_semantics.kda_num_heads//config.tensor_parallel_size not in (12,24,48,96)
            or runtime_inputs.graph_execution.enabled or runtime_inputs.mtp_draft_sampling_source is not None
            or runtime_inputs.mtp_proposer_execution_source is not None):
        return False
    if config.data_parallel_size > 1:
        return (is_kimi_k3_data_parallel_model_scope(model, config)
            and (runtime_inputs.k3_latent_tail_tier is None if config.enable_expert_parallel
                else runtime_inputs.k3_latent_tail_tier == "ALLREDUCE_OVERLAP"))
    return (is_kimi_k3_parallel_model_scope(model, config.tensor_parallel_size, config.pipeline_parallel_size,
                expert_parallel=config.enable_expert_parallel)
        and (runtime_inputs.k3_latent_tail_tier is None if config.enable_expert_parallel or config.tensor_parallel_size == 1
            else runtime_inputs.k3_latent_tail_tier in ("ALLREDUCE_OVERLAP", "COLUMN_PARALLEL")))


def is_dspark_runtime_source_scope(pool, model, runtime_inputs):
    from src.serving.workload.model_config import is_deepseek_v4_parallel_model_scope, is_deepseek_v4_data_parallel_model_scope
    config = pool.parallel_plan.config
    mapping = pool.parallel_plan.layer_to_pipeline_stage
    return (pool.hardware_id == "nvidia_h200" and pool.role == "colocated"
        and runtime_inputs.symbolic_forward_source == "vllm_0_30_v4_dspark"
        and not runtime_inputs.graph_execution.enabled and config.context_parallel_size == config.sequence_parallel_size == 1
        and len(pool.ranks) == config.worker_count and pool.parallel_plan.num_layers == len(mapping) == model.num_hidden_layers
        and pool.parallel_plan.num_experts == model.num_experts and mapping == tuple(sorted(mapping))
        and set(mapping) == set(range(config.pipeline_parallel_size))
        and (is_deepseek_v4_data_parallel_model_scope(model, config) if config.data_parallel_size > 1 else
            is_deepseek_v4_parallel_model_scope(model, config.tensor_parallel_size, config.pipeline_parallel_size,
                expert_parallel=config.enable_expert_parallel)))


def is_dense_parallel_source_scope(pool, model, *, model_execution_version,
                                   symbolic_forward_source, graph_enabled):
    """General eager CUDA0.30 16-bit dense source; no provider acceptance.

    TP/PP are checked by source geometry rather than a topology allowlist.
    DP replicas are routed as separate pools by the existing deployment owner.
    Captured and other vendor/version programs keep their existing gates.
    """
    config = pool.parallel_plan.config
    return (model_execution_version == "vllm_0_30_0"
        and symbolic_forward_source == "cuda_eager_inplace" and not graph_enabled
        and pool.hardware_id in ("nvidia_a800", "nvidia_h200") and pool.role == "colocated"
        and config.is_rank_local and not config.enable_expert_parallel
        and len(pool.ranks) == config.worker_count
        and model.compute_dtype == model.weight_dtype and model.compute_dtype in ("bf16", "fp16")
        and is_dense_parallel_model_scope(model, config.tensor_parallel_size, config.pipeline_parallel_size))


def is_kimi_k2_rank_local_source_scope(pool, model):
    """Eager H200 K2 text source geometry; GPU provider remains unvalidated."""
    config = pool.parallel_plan.config
    pd_pool = (pool.role in ("prefill", "decode")
        and config.tensor_parallel_size == config.expert_parallel_size == 8
        and config.pipeline_parallel_size == 1 and config.enable_expert_parallel)
    return (pool.hardware_id == "nvidia_h200" and (pool.role == "colocated" or pd_pool)
        and config.is_rank_local and len(pool.ranks) == config.worker_count
        and is_kimi_k2_parallel_model_scope(model, config.tensor_parallel_size,
            config.pipeline_parallel_size, expert_parallel=config.enable_expert_parallel))



def is_kimi_k2_data_parallel_source_scope(pool, model, *, model_execution_version, symbolic_forward_source, graph_enabled):
    """Versioned eager K2 DP source geometry, separate from all pricing gates."""
    from src.serving.workload.model_config import is_kimi_k2_data_parallel_model_scope
    config = pool.parallel_plan.config
    mapping = pool.parallel_plan.layer_to_pipeline_stage
    return (type(graph_enabled) is bool and not graph_enabled
        and model_execution_version == "vllm_0_30_0"
        and symbolic_forward_source == "vllm_0_30_k2_6_int4_sync_explicit_allreduce"
        and pool.hardware_id == "nvidia_h200" and pool.role == "colocated"
        and len(pool.ranks) == config.worker_count and len(mapping) == model.num_hidden_layers
        and mapping == tuple(sorted(mapping)) and set(mapping) == set(range(config.pipeline_parallel_size))
        and is_kimi_k2_data_parallel_model_scope(model, config))

def is_qwen3_moe_rank_local_source_scope(pool, model, *, model_execution_version,
                                        symbolic_forward_source, graph_enabled):
    """Ordinary CUDA0.30 expert TP or stage-local EP, with actual PP owners."""
    from src.serving.parallelism.tp import validate_gqa_parallelism
    config = pool.parallel_plan.config
    if (model_execution_version != "vllm_0_30_0" or symbolic_forward_source != "cuda_eager_inplace"
            or graph_enabled or not is_qwen3_moe_source_model_scope(model)
            or pool.hardware_id not in ("nvidia_a800", "nvidia_h200") or pool.role != "colocated"
            or config.tensor_parallel_size > model.num_experts
            or not config.is_rank_local or len(pool.ranks) != config.worker_count):
        return False
    try:
        validate_gqa_parallelism(model, config)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class ValidatedScope:
    """One validated target-model deployment topology.

    The fields through ``hardware_ids`` state the validated fact itself;
    ``config_sha256`` pins the one identity-sensitive variant (DeepSeek-V3
    TP4 is validated only for the 32-layer local config).  The remaining
    fields record which axes each consuming gate actually enforces, so the
    historical granularity drift is data rather than call-site folklore.
    """

    name: str
    scope_kind: str
    model_types: tuple[str, ...]
    deployment_modes: tuple[str, ...]
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel: bool
    expert_parallel_size: int
    hardware_ids: tuple[str, ...]
    config_sha256: str | None
    # Construction gate: False on the rows that predate the hardware pin.
    construction_checks_hardware: bool
    # Emission gate: False when the gate never consults this row (the
    # fixture layout is accepted for every single-rank pool up front).
    emission_gate: bool
    # Emission gate: the A800 production rows additionally require the
    # caller's allow_single_pool_tp1_pp flag.
    emission_requires_allow_flag: bool
    # Emission gate: identities excluded on this row (the TP8 DeepSeek-V3
    # aggregate row excludes the 32-layer identity, which is TP4-only at
    # emission; the construction gate accepts it at TP8 too).
    emission_excludes_config_sha256: tuple[str, ...]
    # Scheduler gate: None when the Scheduler never accepts this topology.
    # Where the Scheduler accepts a wider model set than the validated one,
    # scheduler_model_types records the Scheduler's own (over-approximate)
    # set; scheduler_roles=None means the Scheduler row is role-free.
    scheduler_model_types: tuple[str, ...] | None
    scheduler_roles: tuple[str, ...] | None
    scheduler_requires_named_identity: bool

    def __post_init__(self) -> None:
        for field in ("name", "scope_kind"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"ValidatedScope.{field} must be non-empty")
        for field in ("model_types", "deployment_modes", "hardware_ids"):
            value = getattr(self, field)
            if not isinstance(value, tuple) or not value or any(
                    not isinstance(item, str) or not item for item in value):
                raise TypeError(
                    f"ValidatedScope.{field} must be a non-empty tuple of "
                    "non-empty strings")
        for field in (
                "tensor_parallel_size", "pipeline_parallel_size",
                "expert_parallel_size"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"ValidatedScope.{field} must be an integer")
            if value < 1:
                raise ValueError(f"ValidatedScope.{field} must be positive")
        for field in (
                "expert_parallel", "construction_checks_hardware",
                "emission_gate", "emission_requires_allow_flag",
                "scheduler_requires_named_identity"):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"ValidatedScope.{field} must be boolean")
        if self.config_sha256 is not None and (
                not isinstance(self.config_sha256, str)
                or len(self.config_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.config_sha256)):
            raise ValueError(
                "ValidatedScope.config_sha256 must be a lowercase SHA256 or "
                "null")
        if not isinstance(self.emission_excludes_config_sha256, tuple):
            raise TypeError(
                "ValidatedScope.emission_excludes_config_sha256 must be a "
                "tuple")
        if self.scheduler_model_types is not None and (
                not isinstance(self.scheduler_model_types, tuple)
                or not self.scheduler_model_types):
            raise TypeError(
                "ValidatedScope.scheduler_model_types must be a non-empty "
                "tuple or null")
        if self.scheduler_roles is not None and (
                not isinstance(self.scheduler_roles, tuple)
                or not self.scheduler_roles):
            raise TypeError(
                "ValidatedScope.scheduler_roles must be a non-empty tuple "
                "or null")


_TARGET_MODEL_TYPES = ("kimi_k2", "glm_moe_dsa", "deepseek_v3")

_VALIDATED_TARGET_SCOPES = (
    # The TP=PP=EP=1 fixture layout: accepted for the fixed target models in
    # colocated and PD deployments on any hardware.  The emission gate never
    # consults this row because it accepts every single-rank pool up front,
    # and the Scheduler gate handles TP=1 outside the target scope.
    ValidatedScope(
        name="fixture_tp1_pp1_ep1",
        scope_kind="fixture",
        model_types=_TARGET_MODEL_TYPES,
        deployment_modes=("colocated", "pd_disagg"),
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel=False,
        expert_parallel_size=1,
        hardware_ids=("nvidia_a800", "nvidia_h200"),
        config_sha256=None,
        construction_checks_hardware=False,
        emission_gate=False,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=None,
        scheduler_roles=None,
        scheduler_requires_named_identity=False,
    ),
    # Source-backed A800 production layout: one colocated TP=EP=8 pool,
    # PP2 for Kimi-K2.6.
    ValidatedScope(
        name="kimi_a800_colocated_tp8_pp2_ep8",
        scope_kind="a800_production",
        model_types=("kimi_k2",),
        deployment_modes=("colocated",),
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
        expert_parallel=True,
        expert_parallel_size=8,
        hardware_ids=("nvidia_a800",),
        config_sha256=None,
        construction_checks_hardware=False,
        emission_gate=True,
        emission_requires_allow_flag=True,
        emission_excludes_config_sha256=(),
        scheduler_model_types=_TARGET_MODEL_TYPES,
        scheduler_roles=None,
        scheduler_requires_named_identity=True,
    ),
    # Source-backed A800 production layout: one colocated TP=EP=8 pool,
    # PP4 for GLM-5.2.
    ValidatedScope(
        name="glm_a800_colocated_tp8_pp4_ep8",
        scope_kind="a800_production",
        model_types=("glm_moe_dsa",),
        deployment_modes=("colocated",),
        tensor_parallel_size=8,
        pipeline_parallel_size=4,
        expert_parallel=True,
        expert_parallel_size=8,
        hardware_ids=("nvidia_a800",),
        config_sha256=None,
        construction_checks_hardware=False,
        emission_gate=True,
        emission_requires_allow_flag=True,
        emission_excludes_config_sha256=(),
        scheduler_model_types=_TARGET_MODEL_TYPES,
        scheduler_roles=None,
        scheduler_requires_named_identity=True,
    ),
    # Kimi-K2.6 / GLM-5.2 PD production layout: one TP=EP=8 pool per role,
    # PP1, on A800 or H200.
    ValidatedScope(
        name="target_pd_tp8_pp1_ep8",
        scope_kind="production_pd",
        model_types=("kimi_k2", "glm_moe_dsa"),
        deployment_modes=("pd_disagg",),
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel=True,
        expert_parallel_size=8,
        hardware_ids=("nvidia_a800", "nvidia_h200"),
        config_sha256=None,
        construction_checks_hardware=False,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=_TARGET_MODEL_TYPES,
        scheduler_roles=("prefill", "decode"),
        scheduler_requires_named_identity=True,
    ),
    # GLM-5.2 H200 aggregate: one colocated TP8/PP1/EP1 pool.
    ValidatedScope(
        name="glm_h200_colocated_tp8_pp1_ep1",
        scope_kind="h200_aggregate",
        model_types=("glm_moe_dsa",),
        deployment_modes=("colocated",),
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel=False,
        expert_parallel_size=1,
        hardware_ids=("nvidia_h200",),
        config_sha256=None,
        construction_checks_hardware=True,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=("glm_moe_dsa",),
        scheduler_roles=("colocated",),
        scheduler_requires_named_identity=True,
    ),
    # Source-only target-off PP candidate. vLLM 0.30 V32 carries hidden and
    # residual between contiguous PP partitions. This row enables static
    # emission only; it does not grant Scheduler or normal-capacity support.
    ValidatedScope(
        name="glm_h200_colocated_tp8_pp2_ep1",
        scope_kind="h200_aggregate",
        model_types=("glm_moe_dsa",),
        deployment_modes=("colocated",),
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
        expert_parallel=False,
        expert_parallel_size=1,
        hardware_ids=("nvidia_h200",),
        config_sha256=GLM_MOE_DSA_CONFIG_SHA256,
        construction_checks_hardware=True,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=None,
        scheduler_roles=None,
        scheduler_requires_named_identity=False,
    ),
    # GLM-5.2 H200 PD: one TP8/PP1/EP1 pool per role.
    ValidatedScope(
        name="glm_h200_pd_tp8_pp1_ep1",
        scope_kind="h200_pd_ep1",
        model_types=("glm_moe_dsa",),
        deployment_modes=("pd_disagg",),
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel=False,
        expert_parallel_size=1,
        hardware_ids=("nvidia_h200",),
        config_sha256=None,
        construction_checks_hardware=True,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=("glm_moe_dsa",),
        scheduler_roles=("prefill", "decode"),
        scheduler_requires_named_identity=True,
    ),
    # Kimi-K2.6 H200 InferenceX-style colocated layout: one TP8/PP1 pool
    # with EP8 (EP=TP).  W4A16 routed-expert residency closes on 8x141GB
    # HBM, so the PP2 A800 production scope is not the only feasible Kimi
    # colocated layout.
    ValidatedScope(
        name="kimi_h200_colocated_tp8_pp1_ep8",
        scope_kind="h200_colocated_ep8",
        model_types=("kimi_k2",),
        deployment_modes=("colocated",),
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel=True,
        expert_parallel_size=8,
        hardware_ids=("nvidia_h200",),
        config_sha256=None,
        construction_checks_hardware=True,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=("kimi_k2",),
        scheduler_roles=("colocated",),
        scheduler_requires_named_identity=True,
    ),
    # DeepSeek-V3 H200 InferenceX-style aggregate: one colocated
    # TP8/PP1/EP1 pool (FP8 weights TP8-sharded fit one H200 per rank).
    # The emission gate derives the aggregate TP from the config identity,
    # so it excludes the 32-layer variant here; the construction gate
    # accepts that identity at TP8 as well.
    ValidatedScope(
        name="dsv3_h200_colocated_tp8_pp1_ep1",
        scope_kind="h200_aggregate",
        model_types=("deepseek_v3",),
        deployment_modes=("colocated",),
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel=False,
        expert_parallel_size=1,
        hardware_ids=("nvidia_h200",),
        config_sha256=None,
        construction_checks_hardware=True,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(DEEPSEEK_V3_L32_CONFIG_SHA256,),
        scheduler_model_types=("deepseek_v3",),
        scheduler_roles=("colocated",),
        scheduler_requires_named_identity=True,
    ),
    # The 32-layer local DeepSeek-V3 variant measured at TP4 on four H200s:
    # one colocated TP4/PP1/EP1 pool running the same rank-local DAG.  The
    # Scheduler row pins the config identity but not the named-target set.
    ValidatedScope(
        name="dsv3_h200_colocated_tp4_pp1_ep1_l32",
        scope_kind="h200_aggregate",
        model_types=("deepseek_v3",),
        deployment_modes=("colocated",),
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        expert_parallel=False,
        expert_parallel_size=1,
        hardware_ids=("nvidia_h200",),
        config_sha256=DEEPSEEK_V3_L32_CONFIG_SHA256,
        construction_checks_hardware=True,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=("deepseek_v3",),
        scheduler_roles=("colocated",),
        scheduler_requires_named_identity=False,
    ),
    # DeepSeek-V3 H200 PD: one TP8/PP1/EP1 pool per role.  The PD transfer
    # payload is the compressed MLA kv_c||k_pe cache only (no DSA index
    # cache), TP-replicated, so the rank-pair transfer is EP-independent.
    ValidatedScope(
        name="dsv3_h200_pd_tp8_pp1_ep1",
        scope_kind="h200_pd_ep1",
        model_types=("deepseek_v3",),
        deployment_modes=("pd_disagg",),
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel=False,
        expert_parallel_size=1,
        hardware_ids=("nvidia_h200",),
        config_sha256=None,
        construction_checks_hardware=True,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=("deepseek_v3",),
        scheduler_roles=("prefill", "decode"),
        scheduler_requires_named_identity=True,
    ),
    # DeepSeek-V3 H20 PD: the same TP8/PP1/EP1 MLA layout as the H200 row.
    # H20 shares the Hopper instruction set and the same MLA kv_c||k_pe
    # payload, so the topology fact carries over; it is NOT calibrated by any
    # observed H20 GEMM launch LUT (see launch_data/nvidia_h20).
    ValidatedScope(
        name="dsv3_h20_pd_tp8_pp1_ep1",
        scope_kind="h20_pd_ep1",
        model_types=("deepseek_v3",),
        deployment_modes=("pd_disagg",),
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel=False,
        expert_parallel_size=1,
        hardware_ids=("nvidia_h20",),
        config_sha256=None,
        construction_checks_hardware=True,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=("deepseek_v3",),
        scheduler_roles=("prefill", "decode"),
        scheduler_requires_named_identity=True,
    ),
    # DeepSeek-V3 H20 colocated: one TP8/PP1/EP1 pool, the aggregate
    # counterpart of the H200 aggregate row (over-approximated exactly the
    # same way, with the per-model checks failing closed afterwards).
    ValidatedScope(
        name="dsv3_h20_colocated_tp8_pp1_ep1",
        scope_kind="h20_aggregate",
        model_types=("deepseek_v3",),
        deployment_modes=("colocated",),
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel=False,
        expert_parallel_size=1,
        hardware_ids=("nvidia_h20",),
        config_sha256=None,
        construction_checks_hardware=True,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=("deepseek_v3",),
        scheduler_roles=("colocated",),
        scheduler_requires_named_identity=True,
    ),
    # DeepSeek-V4 Flash H200 colocated: one TP4/PP1/EP1 pool (FP8
    # block-scaled weights TP4-sharded fit one H200 per rank; MXFP4 routed
    # experts replicated at EP1).  The config identity pins the Flash
    # checkpoint; everything else fails closed.
    ValidatedScope(
        name="dsv4_flash_h200_colocated_tp4_pp1_ep1",
        scope_kind="h200_aggregate",
        model_types=("deepseek_v4",),
        deployment_modes=("colocated",),
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        expert_parallel=False,
        expert_parallel_size=1,
        hardware_ids=("nvidia_h200",),
        config_sha256=DEEPSEEK_V4_FLASH_CONFIG_SHA256,
        construction_checks_hardware=True,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=("deepseek_v4",),
        scheduler_roles=("colocated",),
        scheduler_requires_named_identity=True,
    ),
    # DeepSeek-V4 Pro H200 colocated: one TP8/PP1/EP1 pool of the Pro
    # checkpoint.
    ValidatedScope(
        name="dsv4_pro_h200_colocated_tp8_pp1_ep1",
        scope_kind="h200_aggregate",
        model_types=("deepseek_v4",),
        deployment_modes=("colocated",),
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel=False,
        expert_parallel_size=1,
        hardware_ids=("nvidia_h200",),
        config_sha256=DEEPSEEK_V4_PRO_CONFIG_SHA256,
        construction_checks_hardware=True,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=("deepseek_v4",),
        scheduler_roles=("colocated",),
        scheduler_requires_named_identity=True,
    ),
)

# The source-backed A800 production parallel layout per target model,
# derived from the table rows so the Scheduler EP gate and the workload
# gates cannot restate it separately.  The Scheduler EP gate additionally
# accepts the TP8/PP1 layout (the production_pd and h200_colocated_ep8
# rows) for both models.
TARGET_MODEL_PRODUCTION_PARALLELISM = {
    row.model_types[0]: (
        row.tensor_parallel_size, row.pipeline_parallel_size)
    for row in _VALIDATED_TARGET_SCOPES
    if row.scope_kind == "a800_production"
}


def _scope_integer(value, field: str, gate: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{gate} scope {field} must be an integer")
    if value < 1:
        raise ValueError(f"{gate} scope {field} must be positive")


def _scope_common_fields(
    gate: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel: bool,
) -> None:
    _scope_integer(tensor_parallel_size, "tensor_parallel_size", gate)
    _scope_integer(pipeline_parallel_size, "pipeline_parallel_size", gate)
    if not isinstance(expert_parallel, bool):
        raise TypeError(f"{gate} scope expert_parallel must be boolean")


def is_validated_target_construction_scope(
    model: ModelConfig,
    *,
    deployment_mode: str,
    hardware_id: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel: bool,
    expert_parallel_size: int,
) -> bool:
    """Return whether the target logical workload construction gate accepts.

    The table checks model/config identity against its rows, deployment mode and the
    parallel axes, and the hardware pin only on the rows that carry one for
    this gate.  Pool size, rank locality, and the single-pool colocated rule
    remain the caller's own structural checks.
    """

    if not isinstance(model, ModelConfig):
        raise TypeError("target construction scope requires a ModelConfig")
    for label, value in (
            ("deployment_mode", deployment_mode),
            ("hardware_id", hardware_id)):
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"target construction scope {label} must be non-empty")
    _scope_common_fields(
        "target construction",
        tensor_parallel_size,
        pipeline_parallel_size,
        expert_parallel,
    )
    _scope_integer(
        expert_parallel_size, "expert_parallel_size", "target construction")
    for row in _VALIDATED_TARGET_SCOPES:
        if model.model_type not in row.model_types:
            continue
        if deployment_mode not in row.deployment_modes:
            continue
        if (tensor_parallel_size, pipeline_parallel_size) != (
                row.tensor_parallel_size, row.pipeline_parallel_size):
            continue
        if expert_parallel is not row.expert_parallel:
            continue
        if expert_parallel_size != row.expert_parallel_size:
            continue
        if (row.config_sha256 is not None
                and model.config_sha256 != row.config_sha256):
            continue
        if row.construction_checks_hardware and (
                hardware_id not in row.hardware_ids):
            continue
        return True
    return False


def validated_target_emission_scope(
    model: ModelConfig,
    *,
    deployment_mode: str,
    hardware_id: str,
    single_pool: bool,
    pool_rank_count: int,
    worker_count: int,
    is_rank_local: bool,
    allow_production_pipeline: bool,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel: bool,
    expert_parallel_size: int,
) -> ValidatedScope | None:
    """Return the validated emission scope one target pool matches, if any.

    This is exactly the target-model branch sequence of
    ``simulation.validate_workload_emission_parallelism``: BF16 compute and
    rank locality up front, then the rows with their pool-size
    (ranks/workers == TP*PP), single-pool colocated rule, hardware pins, the
    ``allow_single_pool_tp1_pp`` flag on the A800 production rows, and the
    32-layer exclusion on the TP8 DeepSeek-V3 aggregate row.  Single-rank
    fixture pools and the non-target Qwen3/Llama slices remain the caller's
    own branches.
    """

    if not isinstance(model, ModelConfig):
        raise TypeError("target emission scope requires a ModelConfig")
    for label, value in (
            ("deployment_mode", deployment_mode),
            ("hardware_id", hardware_id)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"target emission scope {label} must be non-empty")
    for label, value in (
            ("single_pool", single_pool),
            ("is_rank_local", is_rank_local),
            ("allow_production_pipeline", allow_production_pipeline)):
        if not isinstance(value, bool):
            raise TypeError(f"target emission scope {label} must be boolean")
    _scope_common_fields(
        "target emission",
        tensor_parallel_size,
        pipeline_parallel_size,
        expert_parallel,
    )
    _scope_integer(expert_parallel_size, "expert_parallel_size",
                   "target emission")
    _scope_integer(pool_rank_count, "pool_rank_count", "target emission")
    _scope_integer(worker_count, "worker_count", "target emission")
    if model.compute_dtype != "bf16" or not is_rank_local:
        return None
    for row in _VALIDATED_TARGET_SCOPES:
        if not row.emission_gate:
            continue
        if model.model_type not in row.model_types:
            continue
        if deployment_mode not in row.deployment_modes:
            continue
        if (tensor_parallel_size, pipeline_parallel_size) != (
                row.tensor_parallel_size, row.pipeline_parallel_size):
            continue
        if expert_parallel is not row.expert_parallel:
            continue
        if expert_parallel_size != row.expert_parallel_size:
            continue
        if hardware_id not in row.hardware_ids:
            continue
        if (row.config_sha256 is not None
                and model.config_sha256 != row.config_sha256):
            continue
        if model.config_sha256 in row.emission_excludes_config_sha256:
            continue
        if row.emission_requires_allow_flag and not allow_production_pipeline:
            continue
        if "colocated" in row.deployment_modes and not single_pool:
            continue
        pool_size = row.tensor_parallel_size * row.pipeline_parallel_size
        if pool_rank_count != pool_size or worker_count != pool_size:
            continue
        return row
    return None


def is_validated_target_scheduler_scope(
    model: ModelConfig,
    *,
    role: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel: bool,
) -> bool:
    """Return whether the vLLM Scheduler gate's target scope accepts.

    This is exactly the ``target_scope`` disjunction of
    ``vllm._validate_native_scheduler_scope``: the Scheduler knows roles
    rather than deployments or hardware, pins the named config identities on
    every row except the 32-layer DeepSeek-V3 TP4 row, and deliberately
    over-approximates (the PP2/PP4 rows accept every named target model in
    any role; the PD EP8 row accepts every named target model).  The
    narrower per-model checks in ``_construct_native_scheduler`` and
    ``_validate_ep_scheduler_model_scope`` fail closed afterwards.
    """

    if not isinstance(model, ModelConfig):
        raise TypeError("target scheduler scope requires a ModelConfig")
    if not isinstance(role, str) or not role:
        raise ValueError("target scheduler scope role must be non-empty")
    _scope_common_fields(
        "target scheduler",
        tensor_parallel_size,
        pipeline_parallel_size,
        expert_parallel,
    )
    named = is_named_target_model_identity(
        model.model_type, model.config_sha256)
    for row in _VALIDATED_TARGET_SCOPES:
        if row.scheduler_model_types is None:
            continue
        if model.model_type not in row.scheduler_model_types:
            continue
        if row.scheduler_roles is not None and role not in row.scheduler_roles:
            continue
        if (tensor_parallel_size, pipeline_parallel_size) != (
                row.tensor_parallel_size, row.pipeline_parallel_size):
            continue
        if expert_parallel is not row.expert_parallel:
            continue
        if row.scheduler_requires_named_identity and not named:
            continue
        if (row.config_sha256 is not None
                and model.config_sha256 != row.config_sha256):
            continue
        return True
    return False



def is_glm_fp8_mtp_rank_local_source_scope(pool, model):
    """Whole-block stage EP cache/source geometry for the declared MTP program."""
    from src.serving.workload.model_config import is_glm_mtp_tensor_model_scope
    config = pool.parallel_plan.config
    mapping = pool.parallel_plan.layer_to_pipeline_stage
    return (pool.hardware_id == "nvidia_h200" and pool.role == "colocated" and config.is_rank_local
        and config.enable_expert_parallel and model.weight_dtype == "fp8"
        and len(pool.ranks) == config.worker_count and pool.parallel_plan.num_experts == model.num_experts
        and pool.parallel_plan.num_layers == model.num_hidden_layers == len(mapping)
        and mapping == tuple(sorted(mapping)) and set(mapping) == set(range(config.pipeline_parallel_size))
        and is_glm_mtp_tensor_model_scope(model, config.tensor_parallel_size, expert_parallel=True))


def is_glm_fp8_rank_local_source_scope(pool, model: ModelConfig) -> bool:
    """Ordinary FP8 TP/PP source, with whole quantization blocks and no EP."""
    from src.serving.workload.model_config import is_glm_fp8_tensor_model_scope
    config = pool.parallel_plan.config
    mapping = pool.parallel_plan.layer_to_pipeline_stage
    return (is_glm_fp8_tensor_model_scope(model, config.tensor_parallel_size)
        and pool.hardware_id == "nvidia_h200" and config.is_rank_local and not config.enable_expert_parallel
        and len(pool.ranks) == config.worker_count and len(mapping) == model.num_hidden_layers
        and config.pipeline_parallel_size <= model.num_hidden_layers
        and mapping == tuple(sorted(mapping)) and set(mapping) == set(range(config.pipeline_parallel_size)))


def is_glm_fp8_data_parallel_source_scope(pool, model: ModelConfig) -> bool:
    """Exact ordinary FP8 target DP placement; source/provider gates remain."""
    from src.serving.workload.model_config import is_glm_fp8_data_parallel_model_scope
    config = pool.parallel_plan.config
    mapping = pool.parallel_plan.layer_to_pipeline_stage
    return (pool.hardware_id == "nvidia_h200" and pool.role == "colocated"
        and is_glm_fp8_data_parallel_model_scope(model, config)
        and len(pool.ranks) == config.worker_count and pool.parallel_plan.num_experts == model.num_experts
        and pool.parallel_plan.num_layers == len(mapping) == model.num_hidden_layers
        and mapping == tuple(sorted(mapping)) and set(mapping) == set(range(config.pipeline_parallel_size)))


def is_glm_bf16_tensor_source_scope(pool, model: ModelConfig) -> bool:
    """Ordinary BF16 GLM source geometry, independent of a TP whitelist."""
    config = pool.parallel_plan.config
    return (model.model_type == "glm_moe_dsa" and model.compute_dtype == model.weight_dtype == "bf16"
        and pool.hardware_id in ("nvidia_a800", "nvidia_h200")
        and len(pool.ranks) == config.tensor_parallel_size and config.pipeline_parallel_size == 1
        and config.is_rank_local
        and not model.num_attention_heads % config.tensor_parallel_size
        and not model.hidden_size % config.tensor_parallel_size
        and not model.intermediate_size % config.tensor_parallel_size
        and (config.enable_expert_parallel or not model.moe_intermediate_size % config.tensor_parallel_size))


def is_glm_bf16_pipeline_source_scope(pool, model: ModelConfig) -> bool:
    """Static placement domain; current attention-program dependencies are runtime checks."""
    config = pool.parallel_plan.config
    if (model.model_type != "glm_moe_dsa" or model.compute_dtype != model.weight_dtype or model.weight_dtype != "bf16"
            or pool.hardware_id not in ("nvidia_a800", "nvidia_h200") or config.pipeline_parallel_size <= 1
            or config.context_parallel_size != 1 or len(pool.ranks) != config.worker_count
            or model.num_attention_heads % config.tensor_parallel_size or model.hidden_size % config.tensor_parallel_size
            or model.intermediate_size % config.tensor_parallel_size
            or (not config.enable_expert_parallel and model.moe_intermediate_size % (config.tensor_parallel_size * config.data_parallel_size))
            or config.sequence_parallel_size not in ((1, config.tensor_parallel_size)
                if config.enable_expert_parallel and config.data_parallel_size > 1 else (1,))):
        return False
    assignments = pool.parallel_plan.layer_to_pipeline_stage
    for stage in range(config.pipeline_parallel_size):
        layers = tuple(index for index, owner in enumerate(assignments) if owner == stage)
        if not layers or layers != tuple(range(layers[0], layers[-1] + 1)):
            return False
    return True


def is_glm_pcp_source_scope(pool, model: ModelConfig) -> bool:
    return pool.parallel_plan.config.context_parallel_size > 1 and is_glm_distributed_source_scope(pool, model)


def is_glm_distributed_source_scope(pool, model: ModelConfig) -> bool:
    """Source-backed BF16 MLA PCP topology; no provider/runtime claim."""
    config = pool.parallel_plan.config
    return (model.model_type == "glm_moe_dsa" and model.attention_layout == "dsa_mla"
        and model.compute_dtype == model.weight_dtype == "bf16"
        and pool.hardware_id in ("nvidia_a800", "nvidia_h200")
        and (config.context_parallel_size > 1 or config.data_parallel_size > 1)
        and config.pipeline_parallel_size == 1
        and (config.sequence_parallel_size == 1 and (not config.enable_expert_parallel or config.data_parallel_size == 1)
            or config.enable_expert_parallel and config.data_parallel_size > 1
                and config.sequence_parallel_size in (1, config.tensor_parallel_size))
        and not model.num_attention_heads % config.tensor_parallel_size
        and (config.enable_expert_parallel
            or not model.moe_intermediate_size % (config.tensor_parallel_size * config.context_parallel_size * config.data_parallel_size))
        and len(pool.ranks) == config.worker_count)


def is_qwen3_moe_dp_sp_source_scope(pool, model: ModelConfig) -> bool:
    """Ordinary0.30 DP source derives internal MoE SP from TP/DP/EP.

    SP1 declares no independent sequence axis; the original explicit SP=TP
    declaration is also supported. Partial SP subgroups are not this source.
    """
    config = pool.parallel_plan.config
    tp, dp = config.tensor_parallel_size, config.data_parallel_size
    return (model.model_type == "qwen3_moe" and model.attention_layout == "standard_gqa"
        and model.compute_dtype == model.weight_dtype == "bf16" and model.num_shared_experts == model.num_dense_layers == 0
        and model.num_mtp_layers == 0 and model.rope_scaling_type == "default" and model.hidden_act == "silu"
        and dp > 1 and config.sequence_parallel_size in ((1, tp) if config.enable_expert_parallel else (1,))
        and ((config.enable_expert_parallel and model.num_experts > 0)
            or (not config.enable_expert_parallel and model.moe_intermediate_size % (tp * dp) == 0)) and config.context_parallel_size == 1
        and config.pipeline_parallel_size <= model.num_hidden_layers
        and len(pool.ranks) == config.worker_count
        and model.num_attention_heads % tp == model.hidden_size % tp == 0
        and (model.num_key_value_heads % tp == 0 or tp % model.num_key_value_heads == 0)
        and pool.hardware_id in ("nvidia_a800", "nvidia_h200"))

def is_gqa_pd_source_scope(deployment: DeploymentPlan, model: ModelConfig) -> bool:
    """Independent GQA source geometry at both PD endpoints.

    Computation uses the shared rank-local Qwen/Llama modules and PP coordinator;
    transport mapping is validated separately, without a pair whitelist.
    """
    if (deployment.mode != "pd_disagg" or model.model_type not in ("qwen3", "qwen3_moe", "llama")
            or model.attention_layout != "standard_gqa"
            or model.compute_dtype != "bf16" or model.weight_dtype != "bf16"):
        return False
    if {worker_role_phase(pool.role) for pool in deployment.worker_pools.values()} != {"prefill", "decode"}:
        return False
    for pool in deployment.worker_pools.values():
        config = pool.parallel_plan.config
        tp = config.tensor_parallel_size
        if model.model_type == "llama" and (
                config.worker_count != 1 or config.enable_expert_parallel):
            return False
        if is_qwen3_moe_dp_sp_source_scope(pool, model):
            continue
        if (pool.hardware_id not in ("nvidia_a800", "nvidia_h200")
                or not config.is_rank_local
                or len(pool.ranks) != config.worker_count
                or config.pipeline_parallel_size > model.num_hidden_layers
                or model.num_attention_heads % tp or model.hidden_size % tp
                or model.intermediate_size % tp
                or (model.num_key_value_heads % tp and tp % model.num_key_value_heads)):
            return False
        if config.enable_expert_parallel and (
                model.model_type != "qwen3_moe" or model.num_experts < 1):
            return False
        if model.model_type == "qwen3_moe" and (
                model.num_shared_experts
                or (not config.enable_expert_parallel and model.moe_intermediate_size % tp)):
            return False
    return True


def is_validated_tp1_tp2_multi_pool_scope(
    deployment: DeploymentPlan,
    model: ModelConfig,
) -> bool:
    """Recognize dense TP1/TP2 pools and the matching TP2 MoE slice."""

    if not isinstance(deployment, DeploymentPlan):
        raise TypeError("TP1/TP2 multi-pool scope requires a DeploymentPlan")
    if not isinstance(model, ModelConfig):
        raise TypeError("TP1/TP2 multi-pool scope requires a ModelConfig")
    pools = tuple(deployment.worker_pools.values())
    if len(pools) < 2 or deployment.mode not in ("colocated", "pd_disagg"):
        return False
    if deployment.mode == "colocated":
        if any(pool.role != "colocated" for pool in pools):
            return False
    elif deployment.mode == "pd_disagg":
        phases = {worker_role_phase(pool.role) for pool in pools}
        if phases != {"prefill", "decode"}:
            return False
        kv_routes = tuple(
            route for route in deployment.routes if route.payload == "kv")
        if not kv_routes:
            return False
        for route in kv_routes:
            source = deployment.worker_pools[route.source_pool]
            destination = deployment.worker_pools[route.destination_pool]
            source_tp = source.parallel_plan.config.tensor_parallel_size
            destination_tp = (
                destination.parallel_plan.config.tensor_parallel_size)
            if (model.num_key_value_heads < max(source_tp, destination_tp)
                    or model.num_key_value_heads % source_tp
                    or model.num_key_value_heads % destination_tp):
                return False
    else:
        raise AssertionError("validated multi-pool deployment mode vanished")
    parallelisms = tuple(pool.parallel_plan.config for pool in pools)
    if not all(
            parallelism.tensor_parallel_size in (1, 2)
            and len(pool.ranks) == parallelism.tensor_parallel_size
            and parallelism.worker_count == parallelism.tensor_parallel_size
            and parallelism.pipeline_parallel_size == 1
            and parallelism.is_rank_local
            and {item.operation for item in pool.collectives} == (
                set() if parallelism.tensor_parallel_size == 1
                else {"all_reduce", "all_gather"})
            for pool, parallelism in zip(
                pools, parallelisms, strict=True)):
        return False
    dense_tp1_tp2 = (
        model.model_type == "qwen3"
        and model.attention_layout == "standard_gqa"
        and model.num_experts == 0
        and model.compute_dtype == "bf16"
        and model.weight_dtype == "bf16"
        and all(
            not parallelism.enable_expert_parallel
            for parallelism in parallelisms)
    )
    ep2 = (
        is_validated_qwen3_moe_ep2_model_scope(model)
        and all(parallelism.tensor_parallel_size == 2
                for parallelism in parallelisms)
        and all(
            parallelism.enable_expert_parallel
            and parallelism.expert_parallel_size == 2
            for parallelism in parallelisms)
    )
    return dense_tp1_tp2 or ep2


def is_validated_standalone_af_tp2_scope(
    deployment: DeploymentPlan,
    model: ModelConfig,
) -> bool:
    """Return whether the exact source-backed standalone AF+TP2 slice applies."""

    if not isinstance(deployment, DeploymentPlan):
        raise TypeError("AF+TP2 scope requires a DeploymentPlan")
    if not isinstance(model, ModelConfig):
        raise TypeError("AF+TP2 scope requires a ModelConfig")
    if deployment.mode != "af_disagg" or len(deployment.worker_pools) != 2:
        return False
    attention_pools = tuple(
        pool for pool in deployment.worker_pools.values()
        if worker_role_operator_group(pool.role) == "attention"
        and worker_role_phase(pool.role) == "colocated"
    )
    ffn_pools = tuple(
        pool for pool in deployment.worker_pools.values()
        if worker_role_operator_group(pool.role) == "ffn"
        and worker_role_phase(pool.role) == "colocated"
    )
    if len(attention_pools) != 1 or len(ffn_pools) != 1:
        return False
    pools = attention_pools + ffn_pools
    for pool in pools:
        parallelism = pool.parallel_plan.config
        if not (
            pool.hardware_id == "nvidia_a800"
            and len(pool.ranks) == 2
            and parallelism.worker_count == 2
            and parallelism.tensor_parallel_size == 2
            and parallelism.expert_parallel_size == 1
            and parallelism.pipeline_parallel_size == 1
            and parallelism.is_rank_local
            and {
                collective.operation for collective in pool.collectives
            } == {"all_reduce", "all_gather"}
            and len(pool.collectives) == 2
        ):
            return False
    attention_pool = attention_pools[0]
    ffn_pool = ffn_pools[0]
    if attention_pool.kv_cache is not None or ffn_pool.kv_cache is not None:
        return False
    expected_routes = {
        (attention_pool.name, ffn_pool.name),
        (ffn_pool.name, attention_pool.name),
    }
    if (
        len(deployment.routes) != 2
        or {
            (route.source_pool, route.destination_pool)
            for route in deployment.routes
        } != expected_routes
        or any(
            route.payload != "activation"
            or route.algorithm != "send_recv"
            or route.link != "nvlink"
            or route.topology != "point_to_point"
            for route in deployment.routes
        )
    ):
        return False
    return (
        (
            model.model_type == "qwen3"
            or is_validated_llama_tp2_model_scope(model)
        )
        and model.attention_layout == "standard_gqa"
        and model.num_experts == 0
        and model.compute_dtype == "bf16"
        and model.weight_dtype == "bf16"
    )



def is_validated_standalone_af_ep2_scope(
    deployment: DeploymentPlan,
    model: ModelConfig,
) -> bool:
    """Return whether the exact source-backed standalone AF+EP2 slice applies."""

    if not isinstance(deployment, DeploymentPlan):
        raise TypeError("AF+EP2 scope requires a DeploymentPlan")
    if not isinstance(model, ModelConfig):
        raise TypeError("AF+EP2 scope requires a ModelConfig")
    if deployment.mode != "af_disagg" or len(deployment.worker_pools) != 2:
        return False
    attention_pools = tuple(
        pool for pool in deployment.worker_pools.values()
        if worker_role_operator_group(pool.role) == "attention"
        and worker_role_phase(pool.role) == "colocated"
    )
    ffn_pools = tuple(
        pool for pool in deployment.worker_pools.values()
        if worker_role_operator_group(pool.role) == "ffn"
        and worker_role_phase(pool.role) == "colocated"
    )
    if len(attention_pools) != 1 or len(ffn_pools) != 1:
        return False
    attention_pool = attention_pools[0]
    ffn_pool = ffn_pools[0]
    if (attention_pool.ranks != (0, 1)
            or ffn_pool.ranks != (2, 3)
            or attention_pool.kv_cache is not None
            or ffn_pool.kv_cache is not None):
        return False
    for pool, expected_ep_size in (
            (attention_pool, 1), (ffn_pool, 2)):
        parallelism = pool.parallel_plan.config
        if not (
            pool.hardware_id == "nvidia_a800"
            and len(pool.ranks) == 2
            and parallelism.worker_count == 2
            and parallelism.tensor_parallel_size == 2
            and parallelism.expert_parallel_size == expected_ep_size
            and parallelism.pipeline_parallel_size == 1
            and parallelism.is_rank_local
            and {
                collective.operation for collective in pool.collectives
            } == {"all_reduce", "all_gather"}
            and len(pool.collectives) == 2
        ):
            return False
    expected_routes = {
        (attention_pool.name, ffn_pool.name),
        (ffn_pool.name, attention_pool.name),
    }
    if (
        len(deployment.routes) != 2
        or {
            (route.source_pool, route.destination_pool)
            for route in deployment.routes
        } != expected_routes
        or any(
            route.payload != "activation"
            or route.algorithm != "send_recv"
            or route.link != "nvlink"
            or route.topology != "point_to_point"
            for route in deployment.routes
        )
    ):
        return False
    return (
        model.architecture == "Qwen3MoeForCausalLM"
        and model.model_type == "qwen3_moe"
        and model.attention_layout == "standard_gqa"
        and model.hidden_size == 128
        and model.intermediate_size == 256
        and model.moe_intermediate_size == 128
        and model.num_attention_heads == 2
        and model.num_key_value_heads == 2
        and model.num_hidden_layers == 2
        and model.head_dim == 64
        and model.vocab_size == 128
        and model.max_position_embeddings == 128
        and model.num_experts == 2
        and model.num_experts_per_tok == 1
        and model.num_dense_layers == 0
        and model.num_shared_experts == 0
        and model.num_mtp_layers == 0
        and model.mtp_architecture == "none"
        and model.routed_scaling_factor == 1.0
        and not model.tie_word_embeddings
        and not model.attention_bias
        and not model.mlp_bias
        and model.qk_norm
        and model.compute_dtype == "bf16"
        and model.weight_dtype == "bf16"
        and model.weight_group_size == 0
        and model.quantized_weight_targets == ()
        and model.quantized_weight_ignore == ()
    )


def is_validated_colocated_tp2_host_kv_scope(
    deployment: DeploymentPlan,
    model: ModelConfig,
) -> bool:
    """Return whether rank-local Host-KV fan-out is source-backed."""

    if not isinstance(deployment, DeploymentPlan):
        raise TypeError("TP2 Host-KV scope requires a DeploymentPlan")
    if not isinstance(model, ModelConfig):
        raise TypeError("TP2 Host-KV scope requires a ModelConfig")
    if deployment.mode != "colocated" or len(deployment.worker_pools) != 1:
        return False
    pool = next(iter(deployment.worker_pools.values()))
    parallelism = pool.parallel_plan.config
    return (
        pool.hardware_id == "nvidia_a800"
        and pool.ranks == (0, 1)
        and pool.kv_cache is not None
        and parallelism.worker_count == 2
        and parallelism.tensor_parallel_size == 2
        and parallelism.pipeline_parallel_size == 1
        and parallelism.expert_parallel_size == 1
        and parallelism.is_rank_local
        and len(pool.collectives) == 2
        and {item.operation for item in pool.collectives}
        == {"all_reduce", "all_gather"}
        and is_validated_qwen3_tp2_host_kv_model_scope(model)
    )
