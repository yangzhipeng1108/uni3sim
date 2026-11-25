"""Production contracts for single-kernel prediction.

These objects deliberately exclude profiler truth such as runtime kernel names,
measured duration, and counters. Those belong to ``evaluation.kernel``.
"""

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from .launch_config import (
    CannGemmLaunchProgram,
    CannGemmTilePipelineProgram,
    FiaCrossDomainLaunchProgram,
    GpuCtaRasterProgram,
    LaunchConfig,
)
from .launch_config.block_fp8_gemm import BLOCK_FP8_GEMM_ALGORITHMS
from .model_contract import (
    PredictionDiagnostics,
    SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER,
    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
    SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
    SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER,
    SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER,
    SOURCE_BOUND_WARP_SLOT2_INSTRUCTIONS_PARAMETER,
    SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER,
    SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER,
    SOURCE_CANN_BF16_ELEMENTWISE_DAG_PARAMETER,
    SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER,
    SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER,
    SOURCE_VLLM_ASCEND_SPLIT_QKV_RMSNORM_ROPE_PROGRAM_PARAMETER,
    SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER,
    SOURCE_METAX_ROPE_VLLM_0_8_5_ABI10_PROGRAM_PARAMETER,
    SOURCE_METAX_ROPE_VLLM_0_9_1_ABI11_PROGRAM_PARAMETER,
    SOURCE_METAX_ROPE_VLLM_0_24_0_ABI13_PROGRAM_PARAMETER,
    SOURCE_VLLM_0_24_0_ACT_AND_MUL_PROGRAM_PARAMETER,
    SOURCE_QK_ROPE_LOOP_PARAMETER,
    SOURCE_RMSNORM_VLLM_0_8_5_MACA_3_7_2_PROGRAM_PARAMETER,
    SOURCE_PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_PARAMETER,
    SOURCE_PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_PARAMETER,
    SOURCE_METAX_BF16_GROUPED_STRIDED_COPY_PARAMETER,
    SOURCE_SCALAR_KV_LOOP_PARAMETER,
    SOURCE_VECTORIZED_KV_COPY_PARAMETER,
    SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER,
)


class _CachedVectorParameterSchemas(list):
    """Avoid recompiling an already validated closed vector schema family."""

    def __init__(self, schemas):
        super().__init__()
        self._schemas = schemas

    def __contains__(self, value):
        return frozenset(value) in self._schemas

    def __iter__(self):
        return iter(())

    def append(self, value):
        del value

    def extend(self, values):
        del values


class _PrunedVectorParameterSchemas:
    """Build only closed schemas that can still equal the input parameters.

    Vector launch validation expresses a closed schema as additive branches.
    A branch that has already introduced a parameter absent from the launch
    being validated cannot become a match later: no branch removes a
    parameter.  Retaining only subsets of the submitted parameter names is
    therefore equivalent to building the full family for membership while
    avoiding its combinatorial intermediate states.
    """

    def __init__(self, parameter_names):
        self._parameter_names = frozenset(parameter_names)
        self._schemas = {frozenset()}

    def __contains__(self, value):
        return frozenset(value) in self._schemas

    def __iter__(self):
        return iter(self._schemas)

    def append(self, value):
        schema = frozenset(value)
        if schema <= self._parameter_names:
            self._schemas.add(schema)

    def extend(self, values):
        for value in values:
            self.append(value)


_VECTOR_PARAMETER_SCHEMA_CACHE = {}


def linear_expert_partition(num_experts: int, world_size: int, rank: int) -> tuple[int, ...]:
    """Contiguous base/remainder placement, including empty expert ranks.

    vLLM0.30 expert_map_manager.determine_expert_map with EPLB disabled
    gives the first remainder ranks one extra expert.
    """
    if (type(num_experts) is not int or num_experts < 1 or type(world_size) is not int or world_size < 1
            or type(rank) is not int or not 0 <= rank < world_size):
        raise ValueError("linear expert placement requires positive experts/world and an in-range rank")
    base, remainder = divmod(num_experts, world_size)
    start = rank * base + min(rank, remainder)
    return tuple(range(start, start + base + (rank < remainder)))


class KernelKind(str, Enum):
    GEMM = "GEMM"
    FLASH_ATTN = "FLASH_ATTN"
    EAGER_ATTN = "EAGER_ATTN"
    SPARSE_ATTN_INDEXER = "SPARSE_ATTN_INDEXER"
    SPARSE_ATTN = "SPARSE_ATTN"
    KV_CACHE = "KV_CACHE"
    FUSED_MOE = "FUSED_MOE"
    UNFUSED_MOE = "UNFUSED_MOE"
    VECTOR = "VECTOR"
    COMMUNICATION = "COMMUNICATION"
    MEMCPY = "MEMCPY"
    KV_OFFLOAD = "KV_OFFLOAD"

    @classmethod
    def parse(cls, value: "KernelKind | str") -> "KernelKind":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or not value.strip():
            raise TypeError("kernel kind must be a non-empty string or KernelKind")
        normalized = value.strip().upper().replace("-", "_")
        aliases = {
            "FLASHATTN": cls.FLASH_ATTN,
            "EAGERATTN": cls.EAGER_ATTN,
            "SPARSEATTNINDEXER": cls.SPARSE_ATTN_INDEXER,
            "SPARSEATTN": cls.SPARSE_ATTN,
            "KVCACHEUPDATE": cls.KV_CACHE,
            "KV_CACHE_UPDATE": cls.KV_CACHE,
            "FUSEDMOE": cls.FUSED_MOE,
            "UNFUSEDMOE": cls.UNFUSED_MOE,
            "NORM": cls.VECTOR,
            "COLLECTIVE": cls.COMMUNICATION,
            "CUSTOM_COMM": cls.COMMUNICATION,
            "ROTARY": cls.VECTOR,
            "ACTIVATION": cls.VECTOR,
            "REDUCE": cls.VECTOR,
            "GATHER": cls.VECTOR,
            "ELEMENTWISE": cls.VECTOR,
            "MOE_SORT": cls.VECTOR,
            "MOE_HISTOGRAM": cls.VECTOR,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"unknown kernel kind: {value!r}") from exc


def canonical_physical_algorithm(
        hardware_id: str, kind: KernelKind, variant: str,
        algorithm: str) -> str:
    """Return the source-neutral physical implementation identity.

    Dataset observations and Serving launch selection can have different
    provenance labels for the same physical launch.  Forward must not branch
    on that provenance, so every KernelSpec crosses this normalization boundary
    before reaching a predictor.
    """

    if hardware_id in ("nvidia_a800", "nvidia_h200") and kind is KernelKind.GEMM:
        if variant in ("gemm", "batched_gemm", "moe_router") and (
                algorithm == "observed_gemm:gemm"
                or algorithm.startswith("cublaslt_injected_gemm")):
            return "nvidia_cublaslt_gemm"
        if variant == "gemv" and algorithm in {
                "observed_gemm:gemv", "cublaslt_injected_gemv"}:
            return "nvidia_cublas_gemv"
    if hardware_id == "metax_c500" and kind is KernelKind.GEMM:
        if variant in ("gemm", "batched_gemm", "moe_router") and algorithm in {
                "mcblas_bf16_gemm", "mcblas_injected_gemm:gemm",
                "observed_mcblas_gemm:gemm"}:
            return "metax_mcblas_gemm"
        if variant == "gemv" and algorithm in {
                "observed_gemm:gemv", "mcblas_gemv", "mcblas_gemv_wave"}:
            return "metax_mcblas_gemv"
    if hardware_id == "ascend_910b" and kind is KernelKind.GEMM:
        if algorithm in {
                "cann_matmul", "cann_injected_matmul",
                "observed_gemm:gemm"}:
            return "ascend_cann_matmul"
        if algorithm == "observed_gemm:gemv":
            return "ascend_cann_gemv"

    aliases = {
        "metax_flash_attn_lut": "metax_flash_attention_generic",
    }
    if algorithm in aliases:
        return aliases[algorithm]
    if algorithm.startswith("observed_metax_"):
        return algorithm[len("observed_"):]
    if (hardware_id == "metax_c500"
            and kind is KernelKind.FLASH_ATTN
            and algorithm.startswith("observed_flash_attention:")):
        return "metax_flash_attention_generic"
    if algorithm.startswith("observed_flash_attention:"):
        return f"flash_attention:{algorithm.split(':', 1)[1]}"
    return algorithm


MOE_FILL_ROLES = (
    "sorted_token_ids",
    "expert_ids",
    "num_tokens_post_padded",
)

MOE_CAST_ROLES = (
    "topk_ids",
    "expert_tokens",
)

PAGED_KV_CACHE_ATTRIBUTES = MappingProxyType({
    "cache_layout": "paged",
    "cache_head_layout": "nhd",
    "kv_cache_dtype": "auto",
    "cache_storage_matches_input": True,
    "slot_mapping_all_valid": True,
    "vector_copy_base_alignment_bytes": 16,
})


@dataclass(frozen=True)
class PhysicalTransferConfig:
    """Complete physical inputs for communication and data movement."""

    algorithm: str
    bytes: int
    direction: str
    link: str
    topology: Optional[str] = None
    tier: Optional[str] = None
    participants: Tuple[int, ...] = ()
    operation: Optional[str] = None
    bytes_semantics: Optional[str] = None
    source_rank: Optional[int] = None
    destination_rank: Optional[int] = None
    ring_order: Tuple[int, ...] = ()
    reduction: Optional[str] = None
    backend: Optional[str] = None
    resource_id: Optional[str] = None
    backend_config_sha256: Optional[str] = None
    # Framework collective implementation when it differs from the default
    # (vLLM) custom all-reduce: SGLang's CustomAllReduceV2, or SGLang's
    # flashinfer all-reduce fused with the following residual RMSNorm.
    implementation: Optional[str] = None
    # In participants order. Equal IDs identify a shared physical NIC, not
    # an assumed device-local link. Bilateral IB uses one node port identity
    # at each end, sharing node TX/RX independently across routes/GPUs.
    participant_nic_ids: Tuple[str, ...] = ()
    # Bilateral GPUDirect bandwidth scenario. The receiver limit is derived
    # from its HWSpec (HBM, PCIe and network), never from workload timings.
    destination_hardware_id: Optional[str] = None
    destination_bandwidth_bytes_s: Optional[float] = None
    # Ordered variable collective segments: all_gatherv input per rank or
    # reduce_scatterv output per rank. bytes is their sum, never a mean rank.
    partition_bytes: Tuple[int, ...] = ()
    # Explicit directed payloads of one switch collective, in participants
    # order. Unlike a per-rank uniform all-to-all extent, bytes is the sum
    # of all nonlocal matrix entries. Empty preserves the existing program.
    traffic_matrix_bytes: Tuple[Tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if (self.destination_hardware_id is None) != (
                self.destination_bandwidth_bytes_s is None):
            raise ValueError("bilateral transfer requires receiver identity and bandwidth")
        if self.destination_hardware_id is not None:
            from src.kernel.hardware_ids import canonical_hardware_id
            canonical_hardware_id(self.destination_hardware_id)
            value = self.destination_bandwidth_bytes_s
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError("receiver bandwidth must be finite and positive")
            if (self.operation != "send_recv" or self.link != "infiniband"
                    or self.backend != "analytical_roofline_v1"):
                raise ValueError("bilateral transfer requires analytical InfiniBand send_recv")
        if self.implementation not in (
                None, "sglang_custom_all_reduce_v2",
                "sglang_flashinfer_allreduce_residual_rmsnorm"):
            raise ValueError(
                f"unknown transfer implementation: {self.implementation!r}")
        if self.implementation is not None and (
                self.operation != "all_reduce" or self.link != "nvlink"):
            raise ValueError(
                "the SGLang custom all-reduce implementation applies only to "
                "NVLink all_reduce")
        for name in ("algorithm", "direction", "link"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"transfer {name} must be non-empty")
            object.__setattr__(self, name, value.strip().lower())
        for name in (
                "topology", "tier", "operation", "bytes_semantics",
                "reduction", "backend"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"transfer {name} cannot be empty")
                object.__setattr__(self, name, value.strip().lower())
        if self.resource_id is not None:
            if (not isinstance(self.resource_id, str)
                    or not self.resource_id.strip()):
                raise ValueError("transfer resource_id cannot be empty")
            object.__setattr__(
                self, "resource_id", self.resource_id.strip())
        if self.backend_config_sha256 is not None:
            if (not isinstance(self.backend_config_sha256, str)
                    or not self.backend_config_sha256.strip()):
                raise ValueError(
                    "transfer backend_config_sha256 cannot be empty")
            object.__setattr__(
                self,
                "backend_config_sha256",
                self.backend_config_sha256.strip(),
            )
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int):
            raise TypeError("transfer bytes must be an integer")
        if self.bytes < 0:
            raise ValueError("transfer bytes cannot be negative")
        object.__setattr__(self, "bytes", self.bytes)
        if not isinstance(self.participants, (list, tuple)):
            raise TypeError("transfer participants must be a list or tuple")
        participants = tuple(self.participants)
        if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in participants):
            raise TypeError("transfer participant ranks must be integers")
        if any(rank < 0 for rank in participants):
            raise ValueError("transfer participant ranks cannot be negative")
        if len(participants) != len(set(participants)):
            raise ValueError("transfer participant ranks must be unique")
        object.__setattr__(self, "participants", participants)
        if not isinstance(self.traffic_matrix_bytes, (tuple, list)):
            raise TypeError("traffic_matrix_bytes must be a sequence of rows")
        matrix = self.traffic_matrix_bytes
        if matrix:
            if (self.backend != "analytical_roofline_v1"
                    or self.algorithm != "pairwise" or self.operation != "all_to_all"
                    or self.link not in ("roce", "pcie") or self.topology != "fully_connected"
                    or self.direction != "collective" or self.bytes_semantics != "aggregate_input"
                    or self.ring_order or self.reduction is not None
                    or self.source_rank is not None or self.destination_rank is not None
                    or self.destination_hardware_id is not None):
                raise ValueError("traffic matrix requires one analytical direct-switch collective")
            if len(participants) < 2 or len(matrix) != len(participants):
                raise ValueError("traffic matrix must cover every collective participant")
            if any(not isinstance(row, (tuple, list)) or len(row) != len(participants)
                   for row in matrix):
                raise ValueError("traffic matrix must be square in participants order")
            if any(type(value) is not int or value < 0 for row in matrix for value in row):
                raise ValueError("traffic matrix entries must be nonnegative integer bytes")
            matrix = tuple(tuple(row) for row in matrix)
            if any(matrix[index][index] != 0 for index in range(len(participants))):
                raise ValueError("traffic matrix diagonal must be zero")
            if self.bytes <= 0 or sum(map(sum, matrix)) != self.bytes:
                raise ValueError("traffic matrix must conserve the complete positive payload")
        object.__setattr__(self, "traffic_matrix_bytes", tuple(matrix))
        if not isinstance(self.partition_bytes, (tuple, list)):
            raise TypeError("collective partition_bytes must be a list or tuple")
        partitions = tuple(self.partition_bytes)
        if any(type(value) is not int or value < 0 for value in partitions):
            raise ValueError("collective partitions must be nonnegative integer byte extents")
        if self.operation in ("all_gatherv", "reduce_scatterv"):
            if len(partitions) != len(participants) or sum(partitions) != self.bytes:
                raise ValueError("variable collective partitions must cover its complete payload in participant order")
            if (self.backend != "analytical_roofline_v1" or self.algorithm != "pairwise"
                    or self.topology != "fully_connected" or self.ring_order or self.link == "roce"):
                raise ValueError("variable collectives require analytical pairwise fully-connected transport without a shared NIC")
        elif partitions:
            raise ValueError("partition_bytes applies only to variable collectives")
        object.__setattr__(self, "partition_bytes", partitions)
        if not isinstance(self.participant_nic_ids, (list, tuple)):
            raise TypeError("transfer participant_nic_ids must be a list or tuple")
        nic_ids = tuple(self.participant_nic_ids)
        if any(not isinstance(value, str) or not value.strip() or value != value.strip()
               for value in nic_ids):
            raise ValueError("transfer NIC IDs must be non-empty strings without surrounding whitespace")
        node_ib = (self.link == "infiniband" and self.destination_hardware_id is not None
            and self.operation == "send_recv" and bool(nic_ids))
        if self.link == "roce" or self.link == "pcie" and matrix or node_ib:
            if (self.backend != "analytical_roofline_v1"
                    or len(nic_ids) != len(participants) or not nic_ids):
                raise ValueError("analytical switch transport requires an explicit port ID for every participant")
        elif nic_ids:
            raise ValueError("participant_nic_ids requires analytical RoCE, bilateral InfiniBand, or a PCIe traffic matrix")
        object.__setattr__(self, "participant_nic_ids", nic_ids)
        for name in ("source_rank", "destination_rank"):
            rank = getattr(self, name)
            if rank is not None:
                if isinstance(rank, bool) or not isinstance(rank, int):
                    raise TypeError(f"transfer {name} must be an integer or null")
                if rank < 0:
                    raise ValueError(f"transfer {name} cannot be negative")
        if not isinstance(self.ring_order, (list, tuple)):
            raise TypeError("transfer ring_order must be a list or tuple")
        ring_order = tuple(self.ring_order)
        if any(isinstance(rank, bool) or not isinstance(rank, int)
               for rank in ring_order):
            raise TypeError("transfer ring_order ranks must be integers")
        if any(rank < 0 for rank in ring_order):
            raise ValueError("transfer ring_order ranks cannot be negative")
        if len(ring_order) != len(set(ring_order)):
            raise ValueError("transfer ring_order ranks must be unique")
        object.__setattr__(self, "ring_order", ring_order)
        if self.backend_config_sha256 is not None:
            digest = self.backend_config_sha256
            if (len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(
                    "transfer backend_config_sha256 must be a lowercase SHA256")


def _immutable_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError("expected a mapping")
    return MappingProxyType(dict(values))


# Storage width of every dtype the compute-shape contract accepts.  This is a
# physical constant of the dtype, so it is declared once and read by every
# consumer instead of being restated as a local map.
DTYPE_BYTES = {
    "fp8": 1,
    "int8": 1,
    "uint8": 1,
    "fp16": 2,
    "bf16": 2,
    "float16": 2,
    "float32": 4,
    "fp32": 4,
    "int32": 4,
    "int64": 8,
    "bool": 1,
}


def _normalize_compute_shape(kind: KernelKind, variant: str,
                             dtype: str,
                             shape: Mapping[str, int],
                             attributes: Mapping[str, Any]
                             ) -> Mapping[str, int]:
    normalized = dict(shape)
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported compute dtype: {dtype!r}")
    expected_dtype_bytes = DTYPE_BYTES[dtype]
    if ("dtype_bytes" in normalized
            and normalized["dtype_bytes"] != expected_dtype_bytes):
        raise ValueError(
            f"shape dtype_bytes={normalized['dtype_bytes']} disagrees with "
            f"dtype {dtype!r}")
    # dtype_bytes is derived from the required dtype contract.  A supplied
    # copy is only a consistency assertion; omission never selects behavior.
    normalized["dtype_bytes"] = expected_dtype_bytes
    if variant == "projected_attention_path":
        return normalized
    if kind is KernelKind.KV_CACHE and variant == "paged_kv_cache":
        has_elements = "elements" in normalized
        has_element_count = "element_count" in normalized
        if has_elements and has_element_count:
            if normalized["elements"] != normalized["element_count"]:
                raise ValueError(
                    "KV_CACHE/paged_kv_cache elements and "
                    "element_count disagree")
        elif has_elements:
            normalized["element_count"] = normalized["elements"]
        elif has_element_count:
            normalized["elements"] = normalized["element_count"]
        structured_fields = ("seq_len", "num_kv_heads", "head_dim")
        present = {name for name in structured_fields if name in normalized}
        if present and present != set(structured_fields):
            missing = set(structured_fields) - present
            raise ValueError(
                "KV_CACHE/paged_kv_cache has a partial structured "
                f"shape; missing {sorted(missing)}")
        if present:
            structured_count = math.prod(
                normalized[name] for name in structured_fields)
            if ("element_count" in normalized
                    and normalized["element_count"] != structured_count):
                raise ValueError(
                    "KV_CACHE/paged_kv_cache element_count "
                    f"{normalized['element_count']} disagrees with "
                    f"structured shape count {structured_count}")
            normalized["element_count"] = structured_count
            normalized["elements"] = structured_count
    elif kind is KernelKind.VECTOR:
        if variant == "fa3_prepare_varlen":
            if set(normalized) != {"batch", "dtype_bytes"}:
                raise ValueError("VECTOR/fa3_prepare_varlen requires only batch")
        elif variant == "flash_mla_metadata":
            fields = {
                "batch", "kv_block_count", "scheduler_partitions",
                "scheduler_metadata_ints", "split_metadata_ints",
                "scheduler_segment_count",
                "scheduler_worker_run_count",
            }
            unknown = set(normalized) - fields - {"dtype_bytes"}
            if unknown:
                raise ValueError(
                    "VECTOR/flash_mla_metadata has unknown shape fields: "
                    f"{sorted(unknown)}")
            if fields.issubset(normalized):
                if (normalized["scheduler_metadata_ints"]
                        != 8 * normalized["scheduler_partitions"]):
                    raise ValueError(
                        "VECTOR/flash_mla_metadata scheduler_metadata_ints "
                        "must equal 8 * scheduler_partitions")
                if (normalized["split_metadata_ints"]
                        != normalized["batch"] + 1):
                    raise ValueError(
                        "VECTOR/flash_mla_metadata split_metadata_ints must "
                        "equal batch + 1")
                if normalized["kv_block_count"] < normalized["batch"]:
                    raise ValueError(
                        "VECTOR/flash_mla_metadata kv_block_count cannot be "
                        "smaller than batch")
                if (normalized["scheduler_segment_count"]
                        < normalized["batch"]):
                    raise ValueError(
                        "VECTOR/flash_mla_metadata scheduler_segment_count "
                        "cannot be smaller than batch")
                if (normalized["scheduler_worker_run_count"] > normalized[
                        "scheduler_segment_count"]
                        or normalized["scheduler_worker_run_count"]
                        > normalized["scheduler_partitions"]):
                    raise ValueError(
                        "VECTOR/flash_mla_metadata worker runs cannot exceed "
                        "the segment or partition count")
        if variant in ("rmsnorm", "add_rmsnorm"):
            has_rows = "rows" in normalized
            has_tokens = "tok" in normalized
            if has_rows and has_tokens:
                if normalized["rows"] != normalized["tok"]:
                    raise ValueError("VECTOR normalization rows and tok disagree")
            elif has_rows:
                normalized["tok"] = normalized["rows"]
            elif has_tokens:
                normalized["rows"] = normalized["tok"]
        if variant == "qkv_rmsnorm_rope":
            fields = {
                "seq_len", "num_query_heads", "num_kv_heads", "head_dim",
                "rotary_dim",
            }
            present = fields & set(normalized)
            if present and present != fields:
                raise ValueError(
                    "VECTOR/qkv_rmsnorm_rope has a partial structured shape; "
                    f"missing {sorted(fields - present)}")
            if present:
                if normalized["rotary_dim"] > normalized["head_dim"]:
                    raise ValueError(
                        "VECTOR/qkv_rmsnorm_rope rotary_dim cannot exceed "
                        "head_dim")
                if normalized["rotary_dim"] % 2:
                    raise ValueError(
                        "VECTOR/qkv_rmsnorm_rope rotary_dim must be even")
                total_heads = (
                    normalized["num_query_heads"]
                    + 2 * normalized["num_kv_heads"])
                structured_count = (
                    normalized["seq_len"] * total_heads
                    * normalized["head_dim"])
                if ("element_count" in normalized
                        and normalized["element_count"] != structured_count):
                    raise ValueError(
                        "VECTOR/qkv_rmsnorm_rope element_count disagrees "
                        "with the structured QKV extent")
                normalized["element_count"] = structured_count
                normalized["elements"] = structured_count
        has_elements = "elements" in normalized
        has_element_count = "element_count" in normalized
        if has_elements and has_element_count:
            if normalized["elements"] != normalized["element_count"]:
                raise ValueError("VECTOR elements and element_count disagree")
        elif has_elements:
            normalized["element_count"] = normalized["elements"]
        elif has_element_count:
            normalized["elements"] = normalized["element_count"]
        structured_fields = {
            "rotary": (
                "seq_len", "num_query_heads", "num_kv_heads", "head_dim"),
            "rope": (
                "seq_len", "num_query_heads", "num_kv_heads", "head_dim",
                "rotary_dim"),
            "activation": ("tok", "intermediate"),
            "silu": ("tok", "intermediate"),
        }
        if variant in structured_fields:
            fields = structured_fields[variant]
            present = {name for name in fields if name in normalized}
            if present and present != set(fields):
                missing = set(fields) - present
                raise ValueError(
                    f"VECTOR/{variant} has a partial structured shape; "
                    f"missing {sorted(missing)}")
            if present:
                if variant == "rotary":
                    structured_count = (
                        normalized["seq_len"]
                        * (normalized["num_query_heads"]
                           + normalized["num_kv_heads"])
                        * normalized["head_dim"]
                    )
                elif variant == "rope":
                    rotary_dim = normalized["rotary_dim"]
                    if rotary_dim > normalized["head_dim"]:
                        raise ValueError(
                            "VECTOR/rope rotary_dim cannot exceed head_dim")
                    if rotary_dim % 2:
                        raise ValueError(
                            "VECTOR/rope rotary_dim must be even")
                    key_heads = (
                        normalized["num_kv_heads"]
                        if attributes["key_present"] else 0)
                    structured_count = (
                        normalized["seq_len"]
                        * (normalized["num_query_heads"] + key_heads)
                        * rotary_dim
                    )
                elif variant in ("activation", "silu"):
                    structured_count = math.prod(
                        normalized[name] for name in fields)
                else:
                    raise ValueError(
                        f"unhandled structured VECTOR variant: {variant!r}")
                if ("element_count" in normalized
                        and normalized["element_count"] != structured_count):
                    raise ValueError(
                        f"VECTOR/{variant} element_count "
                        f"{normalized['element_count']} disagrees with structured "
                        f"shape count {structured_count}")
                normalized["element_count"] = structured_count
                normalized["elements"] = structured_count
    elif kind in (
            KernelKind.GEMM, KernelKind.FLASH_ATTN, KernelKind.EAGER_ATTN,
            KernelKind.SPARSE_ATTN_INDEXER, KernelKind.SPARSE_ATTN,
            KernelKind.KV_CACHE, KernelKind.FUSED_MOE,
            KernelKind.UNFUSED_MOE):
        pass
    else:
        raise ValueError(f"unhandled compute shape kind: {kind.value}")
    return normalized


def _require_shape_fields(kind: KernelKind, variant: str,
                          shape: Mapping[str, int]) -> None:
    if variant == "projected_attention_path":
        from src.kernel.predictors.projected_attention import validate_shape
        validate_shape(shape)
        return
    variants = {
        KernelKind.GEMM: {"gemm", "gemv", "batched_gemm", "moe_router", "kimi_k3_nope_bmm"},
        KernelKind.FLASH_ATTN: {
            "prefill", "decode", "combine", "split_combine",
            "mla_prefill", "mla_decode", "mla_absorbed_prefill", "kimi_k3_nope_context_attention"},
        KernelKind.EAGER_ATTN: {"decomposed"},
        KernelKind.SPARSE_ATTN_INDEXER: {
            "dsa_indexer_prefill", "dsa_indexer_decode",
            "dsa_indexer_mixed", "dsa_index_reuse",
            "dsv4_indexer_prefill", "dsv4_indexer_decode"},
        KernelKind.SPARSE_ATTN: {
            "dsa_sparse_prefill", "dsa_sparse_decode", "dsa_sparse_mixed",
            "dsv4_sparse_prefill", "dsv4_sparse_decode"},
        KernelKind.KV_CACHE: {
            "paged_kv_cache", "mla_kv_cache", "dsa_fp8_index_cache",
            "dsv4_compressed_state_cache", "dsv4_compressor_state_cache",
            "dsv4_qnorm_rope_kv_insert"},
        KernelKind.FUSED_MOE: {"w1", "w2", "routed_experts", "kimi_k3_marlin_experts"},
        KernelKind.UNFUSED_MOE: {"decomposed"},
        KernelKind.VECTOR: {
            "vllm_attention_state_merge",
            "dsv4_moe_route", "dsv32_moe_route", "kimi_moe_route", "kimi_k3_moe_route", "kimi_k3_attn_res", "kimi_k3_situ", "kimi_k3_kda_decode", "kimi_k3_kda_prefill", "kimi_k3_kda_mixed", "kimi_k3_kda_context_prefill", "kimi_k3_nope_prefill",
            "dsv4_short_context_topk", "vllm_embedding_id_widen", "vllm_masked_embedding", "kimi_k3_nope_decode", "kimi_k3_nope_mixed", "kimi_k3_nope_context", "vllm_plain_greedy_sample", "vllm_mtp_position_mask", "vllm_mtp_eager",
            "rmsnorm", "add_rmsnorm", "layernorm", "mla_rope", "elementwise",
            "elementwise_pytorch_vec4", "rotary", "rope",
            "dsv4_inv_rope_fp8_quant", "dsv4_indexer_q_rope_quant",
            "activation", "silu", "reduce", "gather", "softmax", "vocab_parallel_embedding", "vllm_logits_row_select", "vllm_logits_fp32_cast", "vllm_logits_gather_reorder", "vllm_rejection_logits_materialize",
            "dsv32_draft_argmax", "dsv32_draft_id_cast", "dsv32_draft_row_select", "dsv32_draft_buffer_copy", "dsv32_draft_token_stack", "dsv32_draft_id_scatter", "dsv32_draft_prepare_next_ids", "dsv32_draft_prepare_sample_indices", "dsv32_draft_seq_rollback", "dsv32_draft_step_metadata",
                "qkv_rmsnorm_rope",
                "gemm_reduce", "gemm_elementwise", "flash_mla_metadata", "fa3_prepare_varlen",
                "fp8_group_quant",
                "moe_swiglu",
                "moe_init_routing", "moe_init_routing_v2",
                "moe_init_routing_v3", "moe_finalize_routing",
                "moe_compute_expert_tokens", "moe_transpose", "moe_cast",
                "moe_range", "moe_topk", "moe_softmax", "moe_gating",
                "moe_gating_topk", "moe_abs", "moe_token_unpermute",
            "moe_act_and_mul", "moe_silu", "moe_mul", "moe_sum",
            "moe_align",
            "moe_align_opt_128", "moe_align_small_batch",
            "moe_count_and_sort", "moe_reduce", "moe_fill",
            "moe_routing_weight_norm", "moe_routing_weight_scale",
            "moe_routed_id_extent", "moe_expert_map_lookup",
            "moe_router_logit_cast", "moe_routed_id_widen",
            "moe_enable_if",
            "dsv32_fused_norm_rope", "dsv32_fused_q", "dsv4_q_kv_rmsnorm", "dsv32_eh_norm", "dsv32_embed_eh_norm", "dsv32_embed_norm",
            "mhc_pre", "mhc_post_pre", "mhc_prenorm", "mhc_prenorm_broadcast", "mhc_post_prenorm",
            "mhc_pre_norm", "mhc_pre_norm_broadcast", "mhc_post", "hc_head",
        },
    }
    if variant not in variants[kind]:
        raise ValueError(f"unknown {kind.value} variant: {variant!r}")

    required = {
        KernelKind.GEMM: {"b", "m", "n", "k"},
        KernelKind.FLASH_ATTN: {
            "batch", "q_len", "kv_len", "head_dim",
            "num_query_heads", "num_kv_heads"},
        KernelKind.EAGER_ATTN: {
            "batch", "q_len", "kv_len", "head_dim",
            "num_query_heads", "num_kv_heads"},
        KernelKind.SPARSE_ATTN: {
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "topk"},
        KernelKind.FUSED_MOE: {
            "tokens", "experts", "topk", "hidden", "intermediate"},
        KernelKind.UNFUSED_MOE: {
            "tokens", "experts", "topk", "hidden", "intermediate"},
    }
    if kind is KernelKind.FLASH_ATTN and variant in (
            "mla_prefill", "mla_decode", "mla_absorbed_prefill", "kimi_k3_nope_context_attention"):
        missing = {
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "kv_lora_rank",
        } - set(shape)
    elif (kind is KernelKind.SPARSE_ATTN_INDEXER
          and variant == "dsa_index_reuse"):
        missing = {"element_count"} - set(shape)
    elif (kind is KernelKind.SPARSE_ATTN_INDEXER
          and variant in ("dsv4_indexer_prefill", "dsv4_indexer_decode")):
        missing = {
            "batch", "q_len", "kv_len", "num_query_heads", "head_dim",
            "topk", "compress_ratio",
        } - set(shape)
    elif kind is KernelKind.SPARSE_ATTN_INDEXER:
        missing = {
            "batch", "q_len", "kv_len", "num_query_heads", "head_dim",
            "topk",
        } - set(shape)
    elif (kind is KernelKind.SPARSE_ATTN
          and variant in ("dsv4_sparse_prefill", "dsv4_sparse_decode")):
        missing = {
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "topk", "window_size",
            "compress_ratio", "compressed_state_bytes", "swa_token_bytes",
        } - set(shape)
    elif (kind is KernelKind.KV_CACHE
          and variant == "paged_kv_cache"):
        missing = {
            "element_count", "seq_len", "num_kv_heads", "head_dim",
        } - set(shape)
    elif kind is KernelKind.KV_CACHE and variant == "mla_kv_cache":
        missing = {"element_count", "seq_len", "compressed_dim"} - set(shape)
    elif (kind is KernelKind.KV_CACHE
          and variant == "dsa_fp8_index_cache"):
        missing = {
            "element_count", "seq_len", "key_dim",
            "scale_bytes_per_token",
        } - set(shape)
    elif (kind is KernelKind.KV_CACHE
          and variant == "dsv4_compressed_state_cache"):
        missing = {
            "element_count", "seq_len", "compress_ratio", "state_bytes",
        } - set(shape)
    elif (kind is KernelKind.KV_CACHE
          and variant == "dsv4_qnorm_rope_kv_insert"):
        missing = {
            "element_count", "seq_len", "num_query_heads",
            "padded_query_heads", "head_dim", "rope_dim",
            "cache_token_bytes",
        } - set(shape)
    elif (kind is KernelKind.KV_CACHE
          and variant == "dsv4_compressor_state_cache"):
        missing = {
            "element_count", "seq_len", "state_bytes",
        } - set(shape)
    elif kind in required:
        missing = required[kind] - set(shape)
    elif kind is KernelKind.VECTOR and variant in (
            "rmsnorm", "add_rmsnorm", "layernorm"):
        missing = {"hidden", "rows"} - set(shape)
    elif kind is KernelKind.VECTOR and variant == "mla_rope":
        missing = {
            "element_count", "seq_len", "num_query_heads",
            "num_key_heads", "head_dim",
        } - set(shape)
    elif kind is KernelKind.VECTOR and variant == "dsv4_short_context_topk":
        missing = {"element_count", "seq_len", "topk_capacity", "compress_ratio"} - set(shape)
    elif kind is KernelKind.VECTOR and variant in (
            "dsv32_draft_argmax", "dsv32_draft_id_cast", "dsv32_draft_row_select", "dsv32_draft_buffer_copy", "dsv32_draft_token_stack", "dsv32_draft_id_scatter", "dsv32_draft_prepare_next_ids", "dsv32_draft_prepare_sample_indices", "dsv32_draft_seq_rollback", "dsv32_draft_step_metadata"):
        required = {"element_count", "rows", "dtype_bytes"}
        if variant == "dsv32_draft_argmax":
            required.add("vocab")
        elif variant == "dsv32_draft_token_stack":
            required.add("proposal_depth")
        elif variant == "dsv32_draft_id_scatter":
            required.add("destination_rows")
        elif variant == "dsv32_draft_prepare_next_ids":
            required.add("sampled_width")
        elif variant == "dsv32_draft_step_metadata":
            required.add("input_batch_size")
        elif variant in ("dsv32_draft_row_select", "dsv32_draft_buffer_copy"):
            required.add("hidden")
            if variant == "dsv32_draft_row_select":
                required.add("source_rows")
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("draft sampling has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant == "vllm_rejection_logits_materialize":
        required = {"element_count", "rows", "vocab", "source_rows", "dtype_bytes"}
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("rejection materialization has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant == "vllm_plain_greedy_sample":
        required = {"element_count", "rows", "vocab", "dtype_bytes"}
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("plain greedy sampling has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant == "vllm_embedding_id_widen":
        required = {"element_count", "rows", "dtype_bytes"}
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("embedding ID widen has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant == "vllm_masked_embedding":
        required = {"element_count", "rows", "hidden", "dtype_bytes"}
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("masked embedding has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant == "vllm_attention_state_merge":
        required = {"element_count", "rows", "heads", "head_dim", "dtype_bytes"}
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("attention state merge has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant == "kimi_k3_nope_context":
        required = {"element_count", "rows", "heads", "q_lora_rank", "kv_lora_rank", "context_rows", "block_size", "dtype_bytes"}
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("K3 context vector has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant in ("kimi_k3_nope_prefill", "kimi_k3_nope_decode", "kimi_k3_nope_mixed"):
        required = {"element_count", "rows", "heads", "q_lora_rank", "kv_lora_rank", "dtype_bytes"}
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("K3 NoPE prefill has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant in ("kimi_k3_kda_prefill", "kimi_k3_kda_mixed", "kimi_k3_kda_context_prefill"):
        required = {"element_count", "rows", "heads", "head_dim", "dtype_bytes"}
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("K3 KDA prefill has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant == "kimi_k3_kda_decode":
        required = {"element_count", "rows", "heads", "head_dim", "dtype_bytes"}
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("K3 KDA decode has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant == "kimi_k3_situ":
        required = {"element_count", "rows", "intermediate", "dtype_bytes"}
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("K3 SiTU has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant == "kimi_k3_attn_res":
        required = {"element_count", "rows", "hidden", "dtype_bytes"}
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("K3 attention residual has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant == "vllm_mtp_eager":
        required = {"element_count", "rows", "width", "source_rows", "dtype_bytes"}
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("MTP eager operation has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant == "vllm_mtp_position_mask":
        required = {"element_count", "rows", "hidden", "dtype_bytes"}
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("MTP position mask has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant == "vllm_logits_gather_reorder":
        required = {"element_count", "rows", "vocab", "tensor_parallel_size", "dtype_bytes"}
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("logits gather reorder has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant == "vllm_logits_fp32_cast":
        required = {"element_count", "rows", "vocab", "dtype_bytes"}
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("target logits cast has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant == "vllm_logits_row_select":
        required = {"element_count", "rows", "hidden", "source_rows", "dtype_bytes"}
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("target logits row selection has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant == "vocab_parallel_embedding":
        required = {"element_count", "rows", "hidden", "input_id_bytes", "dtype_bytes"}
        missing = required - set(shape)
        if set(shape) - required - {"elements"}:
            raise ValueError("vocab embedding has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant in ("dsv4_moe_route", "dsv32_moe_route", "kimi_moe_route", "kimi_k3_moe_route"):
        missing = {"element_count", "rows", "experts", "topk"} - set(shape)
    elif kind is KernelKind.VECTOR and variant in ("dsv4_inv_rope_fp8_quant", "dsv4_indexer_q_rope_quant"):
        missing = {
            "element_count", "seq_len", "num_query_heads",
            "head_dim", "rope_dim",
        } - set(shape)
    elif kind is KernelKind.VECTOR and variant == "rope":
        missing = {
            "seq_len", "num_query_heads", "num_kv_heads", "head_dim",
            "rotary_dim", "element_count",
        } - set(shape)
    elif kind is KernelKind.VECTOR and variant == "qkv_rmsnorm_rope":
        missing = {
            "seq_len", "num_query_heads", "num_kv_heads", "head_dim",
            "rotary_dim", "element_count",
        } - set(shape)
    elif variant in (
            "elementwise", "elementwise_pytorch_vec4",
            "rotary", "activation", "silu",
            "reduce", "gather", "softmax",
            "gemm_reduce", "gemm_elementwise", "fp8_group_quant"):
        missing = {"element_count"} - set(shape)
    elif kind is KernelKind.VECTOR and variant == "fa3_prepare_varlen":
        missing = {"batch"} - set(shape)
    elif variant == "flash_mla_metadata":
        missing = {
            "batch", "kv_block_count", "scheduler_partitions",
            "scheduler_metadata_ints", "split_metadata_ints",
            "scheduler_segment_count",
            "scheduler_worker_run_count",
        } - set(shape)
    elif kind is KernelKind.VECTOR and variant in ("dsv32_fused_norm_rope", "dsv32_fused_q"):
        required = {"rows", "rope_dim", "index_dim", "dtype_bytes",
                    "position_bytes", "rope_cache_bytes"}
        if variant == "dsv32_fused_norm_rope":
            required |= {"q_hidden", "kv_hidden", "topk", "slot_bytes", "index_affine_bytes"}
        else:
            required |= {"q_heads", "index_heads", "index_weight_bytes", "q_scale_bytes"}
        missing = required - set(shape)
        if set(shape) - required:
            raise ValueError("V32 target fusion has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant == "dsv32_embed_norm":
        required = {"rows", "hidden", "dtype_bytes", "input_id_bytes"}
        missing = required - set(shape)
        if set(shape) - required:
            raise ValueError("V32 target embedding norm has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant in ("dsv32_eh_norm", "dsv32_embed_eh_norm"):
        required = {"rows", "hidden", "dtype_bytes", "position_bytes"}
        if variant == "dsv32_embed_eh_norm":
            required.add("input_id_bytes")
        missing = required - set(shape)
        if set(shape) - required:
            raise ValueError("V32 MTP input norm has unexpected shape fields")
    elif kind is KernelKind.VECTOR and variant == "dsv4_q_kv_rmsnorm":
        missing = {"rows", "q_hidden", "kv_hidden"} - set(shape)
    elif kind is KernelKind.VECTOR and variant in (
            "mhc_pre", "mhc_post_pre", "mhc_prenorm", "mhc_prenorm_broadcast", "mhc_post_prenorm",
            "mhc_pre_norm", "mhc_pre_norm_broadcast", "mhc_post", "hc_head"):
        required_mhc = {"element_count", "hidden", "hc_mult", "dtype_bytes"}
        if variant in (
                "mhc_prenorm", "mhc_prenorm_broadcast", "mhc_post_prenorm",
                "mhc_pre_norm", "mhc_pre_norm_broadcast"):
            required_mhc.add("n_splits")
        if variant == "mhc_post_prenorm":
            required_mhc.add("tile_n")
        if variant in ("mhc_pre_norm", "mhc_pre_norm_broadcast"):
            required_mhc.add("sinkhorn_iters")
        missing = required_mhc - set(shape)
        if set(shape) - required_mhc - {"elements"}:
            raise ValueError(f"VECTOR/{variant} unexpected mHC shape fields")
    elif variant.startswith("moe_"):
        missing = {
            "tokens", "experts", "topk", "hidden", "intermediate"
        } - set(shape)
    else:
        raise ValueError(
            f"unhandled {kind.value} shape contract for {variant!r}")
    if missing:
        raise ValueError(
            f"{kind.value}/{variant} missing shape fields: {sorted(missing)}")
    exact_shapes = {
        (KernelKind.GEMM, "batched_gemm"):
            {"b", "m", "n", "k", "dtype_bytes"},
        (KernelKind.GEMM, "moe_router"):
            {"b", "m", "n", "k", "dtype_bytes"},
        (KernelKind.FLASH_ATTN, "mla_prefill"): {
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "kv_lora_rank", "dtype_bytes",
        },
        (KernelKind.FLASH_ATTN, "kimi_k3_nope_context_attention"): {
            "batch", "q_len", "kv_len", "num_query_heads", "qk_head_dim",
            "v_head_dim", "kv_lora_rank", "dtype_bytes",
        },
        (KernelKind.FLASH_ATTN, "mla_decode"): {
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "kv_lora_rank", "dtype_bytes",
        },
        (KernelKind.FLASH_ATTN, "mla_absorbed_prefill"): {
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "kv_lora_rank", "dtype_bytes",
        },
        (KernelKind.SPARSE_ATTN_INDEXER, "dsa_indexer_prefill"): {
            "batch", "q_len", "kv_len", "num_query_heads", "head_dim",
            "topk", "dtype_bytes",
        },
        (KernelKind.SPARSE_ATTN_INDEXER, "dsa_indexer_decode"): {
            "batch", "q_len", "kv_len", "num_query_heads", "head_dim",
            "topk", "dtype_bytes",
        },
        (KernelKind.SPARSE_ATTN_INDEXER, "dsa_indexer_mixed"): {
            "batch", "q_len", "kv_len", "num_query_heads", "head_dim",
            "topk", "dtype_bytes",
        },
        (KernelKind.SPARSE_ATTN_INDEXER, "dsa_index_reuse"):
            {"element_count", "dtype_bytes"},
        (KernelKind.SPARSE_ATTN_INDEXER, "dsv4_indexer_prefill"): {
            "batch", "q_len", "kv_len", "num_query_heads", "head_dim",
            "topk", "compress_ratio", "dtype_bytes",
        },
        (KernelKind.SPARSE_ATTN_INDEXER, "dsv4_indexer_decode"): {
            "batch", "q_len", "kv_len", "num_query_heads", "head_dim",
            "topk", "compress_ratio", "dtype_bytes",
        },
        (KernelKind.SPARSE_ATTN, "dsa_sparse_prefill"): {
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "topk", "dtype_bytes",
        },
        (KernelKind.SPARSE_ATTN, "dsa_sparse_decode"): {
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "topk", "dtype_bytes",
        },
        (KernelKind.SPARSE_ATTN, "dsv4_sparse_prefill"): {
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "topk", "window_size",
            "compress_ratio", "compressed_state_bytes", "swa_token_bytes",
            "dtype_bytes",
        },
        (KernelKind.SPARSE_ATTN, "dsv4_sparse_decode"): {
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "topk", "window_size",
            "compress_ratio", "compressed_state_bytes", "swa_token_bytes",
            "dtype_bytes",
        },
        (KernelKind.KV_CACHE, "mla_kv_cache"): {
            "element_count", "seq_len", "compressed_dim", "dtype_bytes",
        },
        (KernelKind.KV_CACHE, "paged_kv_cache"): {
            "element_count", "elements", "seq_len", "num_kv_heads",
            "head_dim", "dtype_bytes",
        },
        (KernelKind.KV_CACHE, "dsa_fp8_index_cache"): {
            "element_count", "seq_len", "key_dim", "scale_bytes_per_token",
            "dtype_bytes",
        },
        (KernelKind.KV_CACHE, "dsv4_compressed_state_cache"): {
            "element_count", "seq_len", "compress_ratio", "state_bytes",
            "dtype_bytes",
        },
        (KernelKind.KV_CACHE, "dsv4_qnorm_rope_kv_insert"): {
            "element_count", "seq_len", "num_query_heads",
            "padded_query_heads", "head_dim", "rope_dim",
            "cache_token_bytes", "dtype_bytes",
        },
        (KernelKind.KV_CACHE, "dsv4_compressor_state_cache"): {
            "element_count", "seq_len", "state_bytes", "dtype_bytes",
        },
        (KernelKind.FUSED_MOE, "routed_experts"): {
            "tokens", "experts", "topk", "hidden", "intermediate",
            "dtype_bytes",
        },
        (KernelKind.FUSED_MOE, "kimi_k3_marlin_experts"): {
            "tokens", "experts", "topk", "hidden", "intermediate", "dtype_bytes",
        },
        (KernelKind.VECTOR, "dsv4_q_kv_rmsnorm"): {
            "rows", "q_hidden", "kv_hidden", "dtype_bytes",
        },
        (KernelKind.VECTOR, "dsv4_short_context_topk"): {
            "element_count", "elements", "seq_len", "topk_capacity", "compress_ratio", "dtype_bytes"},
        (KernelKind.VECTOR, "dsv32_moe_route"): {
            "element_count", "elements", "rows", "experts", "topk", "dtype_bytes"},
        (KernelKind.VECTOR, "kimi_moe_route"): {
            "element_count", "elements", "rows", "experts", "topk", "dtype_bytes"},
        (KernelKind.VECTOR, "kimi_k3_moe_route"): {
            "element_count", "elements", "rows", "experts", "topk", "dtype_bytes"},
        (KernelKind.VECTOR, "dsv4_moe_route"): {
            "element_count", "elements", "rows", "experts", "topk", "dtype_bytes"},
        (KernelKind.VECTOR, "layernorm"):
            {"rows", "hidden", "dtype_bytes"},
        (KernelKind.VECTOR, "mla_rope"): {
            "element_count", "elements", "seq_len", "num_query_heads",
            "num_key_heads", "head_dim", "dtype_bytes",
        },
        (KernelKind.VECTOR, "dsv4_inv_rope_fp8_quant"): {
            "element_count", "elements", "seq_len", "num_query_heads",
            "head_dim", "rope_dim", "dtype_bytes",
        },
        (KernelKind.VECTOR, "dsv4_indexer_q_rope_quant"): {
            "element_count", "elements", "seq_len", "num_query_heads",
            "head_dim", "rope_dim", "dtype_bytes",
        },
        (KernelKind.VECTOR, "qkv_rmsnorm_rope"): {
            "element_count", "elements", "seq_len", "num_query_heads",
            "num_kv_heads", "head_dim", "rotary_dim", "dtype_bytes",
        },
    }
    key = (kind, variant)
    if key in exact_shapes:
        unknown = set(shape) - exact_shapes[key]
        if unknown:
            raise ValueError(
                f"{kind.value}/{variant} unknown shape fields: "
                f"{sorted(unknown)}")


def _normalize_request_lengths(
    owner: str,
    normalized: dict[str, Any],
    shape: Mapping[str, int],
) -> None:
    ascend_fia = ("mathematical_program" in normalized
        and normalized["mathematical_program"] == "ascend_0_18_full_fia_softmax_v1")
    native_split_fuse = ("source_program" in normalized
        and normalized["source_program"] == "ascend_0_18_full_fia_split_fuse")
    sparse_fa3 = ("source_program" in normalized
        and normalized["source_program"] == "vllm_0_30_fa3_sparse_mla")
    full_decode_fa3 = (owner == "FLASH_ATTN" and "source_program" in normalized
        and normalized["source_program"] == "vllm_0_30_fa3_gqa_full_decode")
    if full_decode_fa3:
        fields = {"q_lens", "kv_lens", "model_input_q_lens", "model_input_kv_lens",
            "graph_runtime_mode", "graph_model_input_tokens"}
        if not fields.issubset(normalized):
            raise ValueError("FA3 FULL decode requires complete real and physical request metadata")
        if normalized["graph_runtime_mode"] != "FULL":
            raise ValueError("FA3 FULL decode requires its actual FULL runtime mode")
        physical_rows = normalized["graph_model_input_tokens"]
        if type(physical_rows) is not int or physical_rows <= 0:
            raise ValueError("FA3 FULL decode physical Q/O rows must be a positive integer")
        for name in ("q_lens", "kv_lens", "model_input_q_lens", "model_input_kv_lens"):
            raw = normalized[name]
            if not isinstance(raw, (list, tuple)):
                raise TypeError("FA3 FULL decode " + name + " must be a list or tuple")
            values = tuple(raw)
            minimum = 0 if name.startswith("model_input_") else 1
            if (not values or any(type(value) is not int or value < minimum for value in values)):
                raise ValueError("FA3 FULL decode " + name + " has invalid request lengths")
            normalized[name] = values
        real_q, real_kv = normalized["q_lens"], normalized["kv_lens"]
        model_q, model_kv = normalized["model_input_q_lens"], normalized["model_input_kv_lens"]
        real_count, padded_count = len(real_q), shape["batch"]
        if (len(real_kv) != real_count or any(length != 1 for length in real_q)
                or len(model_q) != padded_count or len(model_kv) != padded_count
                or not real_count <= padded_count <= physical_rows
                or physical_rows < sum(real_q)
                or model_q[:real_count] != real_q or model_kv[:real_count] != real_kv
                or any(model_q[real_count:]) or any(model_kv[real_count:])
                or max(model_q) != shape["q_len"] or max(model_kv) != shape["kv_len"]):
            raise ValueError("FA3 FULL decode must preserve real requests, zero request tails and independent physical Q/O rows")
        # GPUModelRunner repeats the last real cuQ endpoint for every padding
        # request. Its final endpoint need not cover all padded Q/O storage.
        return
    model_input_names = {"model_input_q_lens", "model_input_kv_lens"}
    present_model_input_names = model_input_names & set(normalized)
    if present_model_input_names:
        if (present_model_input_names != model_input_names
                or "graph_runtime_mode" not in normalized):
            raise ValueError(
                f"{owner} graph attention domains must be complete: "
                f"present={sorted(present_model_input_names | ({'graph_runtime_mode'} & set(normalized)))}")
        graph_runtime_mode = normalized["graph_runtime_mode"]
        if graph_runtime_mode not in ("NONE", "FULL", "PIECEWISE"):
            raise ValueError(
                f"{owner} graph_runtime_mode is invalid: "
                f"{graph_runtime_mode!r}")
        for name in ("q_lens", "kv_lens"):
            raw_lengths = normalized[name]
            if not isinstance(raw_lengths, (list, tuple)):
                raise TypeError(f"{owner} {name} must be a list or tuple")
            lengths = tuple(raw_lengths)
            if not lengths:
                raise ValueError(f"{owner} {name} cannot be empty")
            if any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in lengths):
                raise ValueError(
                    f"{owner} {name} must contain positive integers")
            maximum_field = "q_len" if name == "q_lens" else "kv_len"
            if max(lengths) > shape[maximum_field]:
                raise ValueError(
                    f"{owner} {name} exceeds model-input "
                    f"{maximum_field}")
            normalized[name] = lengths
        if len(normalized["q_lens"]) != len(normalized["kv_lens"]):
            raise ValueError(f"{owner} logical request lengths must align")
        if any(request_query > request_kv for request_query, request_kv in zip(
                normalized["q_lens"], normalized["kv_lens"])):
            raise ValueError(
                f"{owner} logical request requires q_lens <= kv_lens")
        for name, maximum_field in (
                ("model_input_q_lens", "q_len"),
                ("model_input_kv_lens", "kv_len")):
            raw_lengths = normalized[name]
            if not isinstance(raw_lengths, (list, tuple)):
                raise TypeError(f"{owner} {name} must be a list or tuple")
            lengths = tuple(raw_lengths)
            if len(lengths) != shape["batch"]:
                raise ValueError(
                    f"{owner} {name} must contain one value per model-input "
                    f"request: expected={shape['batch']}, actual={len(lengths)}")
            minimum = 0 if ascend_fia or native_split_fuse else 1
            if any(isinstance(value, bool) or not isinstance(value, int)
                   or value < minimum for value in lengths):
                raise ValueError(
                    f"{owner} {name} must contain positive integers")
            if max(lengths) != shape[maximum_field]:
                raise ValueError(
                    f"{owner} {name} maximum {max(lengths)} disagrees with "
                    f"shape {maximum_field}={shape[maximum_field]}")
            normalized[name] = lengths
        if not (ascend_fia or native_split_fuse) and any(request_query > request_kv for request_query, request_kv in zip(
                normalized["model_input_q_lens"],
                normalized["model_input_kv_lens"])):
            raise ValueError(
                f"{owner} model-input request requires q_lens <= kv_lens")
        if native_split_fuse:
            count = len(normalized["q_lens"])
            q_lens, kv_lens = normalized["model_input_q_lens"], normalized["model_input_kv_lens"]
            fresh = normalized["ascend_fia_prefill_no_cache"]
            dummy = sum(q_lens) != count
            if (graph_runtime_mode != "FULL"
                    or fresh != (normalized["q_lens"] == normalized["kv_lens"])
                    or len(q_lens) != count + int(dummy)
                    or q_lens[:count] != normalized["q_lens"]
                    or kv_lens[:count] != normalized["kv_lens"]
                    or (dummy and kv_lens[-1] != (q_lens[-1] if fresh else 0))):
                raise ValueError("FULL FIA SplitFuse must preserve real requests and its independent source tail")
        if ascend_fia:
            count = len(normalized["q_lens"])
            q_lens, kv_lens = normalized["model_input_q_lens"], normalized["model_input_kv_lens"]
            dummy = sum(q_lens) != count
            fresh = normalized["q_lens"] == normalized["kv_lens"]
            if (graph_runtime_mode != "FULL" or normalized["ascend_fia_prefill_no_cache"] != fresh
                    or q_lens[:count] != normalized["q_lens"] or kv_lens[:count] != normalized["kv_lens"]
                    or len(q_lens) != count + int(dummy)
                    or (dummy and kv_lens[-1] != (q_lens[-1] if fresh else 0))):
                raise ValueError("Ascend FULL FIA must preserve real lengths and its independent source dummy tail")
        if sum(normalized["q_lens"]) > sum(
                normalized["model_input_q_lens"]):
            raise ValueError(
                f"{owner} logical attention rows exceed model-input rows")
        if graph_runtime_mode == "NONE" and (
                normalized["q_lens"] != normalized["model_input_q_lens"]
                or normalized["kv_lens"]
                != normalized["model_input_kv_lens"]):
            raise ValueError(
                f"{owner} eager execution cannot carry graph padding")
        return
    if ascend_fia or native_split_fuse:
        raise ValueError("Ascend FULL FIA requires complete logical and physical request lengths")
    for name, maximum_field in (
            ("q_lens", "q_len"), ("kv_lens", "kv_len")):
        raw_lengths = normalized[name]
        if not isinstance(raw_lengths, (list, tuple)):
            raise TypeError(f"{owner} {name} must be a list or tuple")
        lengths = tuple(raw_lengths)
        if len(lengths) != shape["batch"]:
            raise ValueError(
                f"{owner} {name} must contain one value per request: "
                f"expected={shape['batch']}, actual={len(lengths)}")
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in lengths):
            raise TypeError(f"{owner} {name} must contain integers")
        empty_context = (name == "kv_lens" and (sparse_fa3 or (
            "mathematical_program" in normalized
            and normalized["mathematical_program"] == "bf16_mla_context_empty_rows_softmax_v1")))
        if any(value < 0 if empty_context else value <= 0 for value in lengths):
            raise ValueError(f"{owner} {name} must contain positive values")
        expected_maximum = shape[maximum_field]
        if max(lengths) != expected_maximum:
            raise ValueError(
                f"{owner} {name} maximum {max(lengths)} disagrees with "
                f"shape {maximum_field}={expected_maximum}")
        normalized[name] = lengths
    mathematical_cross_attention = ("mathematical_program" in normalized
        and normalized["mathematical_program"] in ("bf16_mla_context_softmax_v1", "bf16_mla_context_empty_rows_softmax_v1")
        and normalized["causal"] is False)
    mathematical_empty_causal = ("mathematical_program" in normalized
        and normalized["mathematical_program"] == "bf16_mla_causal_empty_rows_softmax_v1"
        and normalized["causal"] is True)
    native_mla_cross_attention = (owner == "FLASH_ATTN" and normalized["causal"] is False
        and "qk_head_dim" in shape and "v_head_dim" in shape
        and (shape["qk_head_dim"], shape["v_head_dim"]) == (192, 128))
    if not (sparse_fa3 or mathematical_cross_attention or mathematical_empty_causal or native_mla_cross_attention) and any(request_query > request_kv for request_query, request_kv in zip(
            normalized["q_lens"], normalized["kv_lens"])):
        raise ValueError(
            f"{owner} each request requires q_lens <= kv_lens: "
            f"q_lens={normalized['q_lens']}, "
            f"kv_lens={normalized['kv_lens']}")


def _normalize_request_length_tuple(
    name: str,
    raw_lengths: Any,
    *,
    allow_zero: bool,
) -> tuple[int, ...]:
    if not isinstance(raw_lengths, (list, tuple)):
        raise TypeError(f"KV_CACHE {name} must be a list or tuple")
    lengths = tuple(raw_lengths)
    if not lengths:
        raise ValueError(f"KV_CACHE {name} cannot be empty")
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in lengths):
        raise TypeError(f"KV_CACHE {name} must contain integers")
    minimum = 0 if allow_zero else 1
    if any(value < minimum for value in lengths):
        raise ValueError(
            f"KV_CACHE {name} must contain values >= {minimum}")
    return lengths


_KDA_CHUNK_VARIANTS = ("kimi_k3_kda_prefill", "kimi_k3_kda_mixed", "kimi_k3_kda_context_prefill")


_KDA_CHUNK_SOURCE_FIELDS = {
    "source_component", "source_stage", "model_execution_version", "source_compiler",
    "source_input_layout", "source_norm_backend", "source_metadata_state",
    "source_initial_state_dtype", "source_output_final_state", "source_out",
    "source_use_qk_l2norm_in_kernel", "tensor_iterator_source", "metadata_dtype",
    "chunk_offsets_dtype", "source_use_gather", "source_selected_configs",
    "source_chunk_indices", "source_chunk_offsets"}


def _is_kda_chunk_source(kind, variant, attributes):
    return (kind is KernelKind.VECTOR and variant in _KDA_CHUNK_VARIANTS
        and "source_component" in attributes
        and attributes["source_component"] == "vllm_0_30_kimi_k3_triton_chunk")


def _validate_kda_chunk_source_launch(variant, shape, dtype, attributes, launch):
    from src.kernel.launch_config.nvidia_h200.kda_chunk_launch import select_kda_chunk_stage_launch
    required = _KDA_CHUNK_SOURCE_FIELDS | {"stage", "query_start_loc", "gate_lower_bound"}
    if required - set(attributes):
        raise ValueError("KDA chunk child requires complete source selection facts")
    if (attributes["stage"] != "chunk" or attributes["metadata_dtype"] != "int32"
            or attributes["chunk_offsets_dtype"] != "int64"
            or attributes["source_use_qk_l2norm_in_kernel"] is not True
            or attributes["source_output_final_state"] is not True
            or attributes["source_out"] != "original_v"
            or type(attributes["source_use_gather"]) is not bool):
        raise ValueError("KDA chunk child requires exact mixed dtype roles and complete source outputs")
    expected = select_kda_chunk_stage_launch(shape, dtype, variant, attributes, None)
    if launch != expected:
        raise ValueError("KDA chunk child requires its exact H200 source launch and execution_variant")



def _normalize_compute_attributes(kind: KernelKind, variant: str,
                                  attributes: Mapping[str, Any],
                                  shape: Mapping[str, int],
                                  compute_dtype: str):
    normalized = dict(attributes)
    if variant == "projected_attention_path":
        from src.kernel.predictors.projected_attention import normalize_attributes
        return normalize_attributes(kind, attributes, shape, compute_dtype)
    optional = set()
    sparse_fa3 = (kind is KernelKind.FLASH_ATTN
        and "source_program" in normalized
        and normalized["source_program"] == "vllm_0_30_fa3_sparse_mla")
    full_decode_fa3 = (kind is KernelKind.FLASH_ATTN
        and "source_program" in normalized
        and normalized["source_program"] == "vllm_0_30_fa3_gqa_full_decode")
    graph_token_domain_fields = {
        "graph_logical_tokens", "graph_model_input_tokens",
        "graph_runtime_mode", "model_input_shape",
    }
    if (kind is KernelKind.FLASH_ATTN
            and variant in (
                "mla_prefill", "mla_decode", "mla_absorbed_prefill", "kimi_k3_nope_context_attention")):
        # The two deployment CUDA-graph facts are part of the MLA contract:
        # vLLM's MLA metadata builders read them (FlashAttnMLA split bound).
        required = {
            "causal", "kv_layout", "kv_dist", "q_lens", "kv_lens",
            "full_cuda_graphs", "max_cudagraph_capture_size",
        }
        # A graph-replayed MLA decode declares its step token domain; the
        # padded requests (query 0, seq_len 0) carry no attention work.
        optional = {
            "graph_logical_tokens", "graph_model_input_tokens",
            "graph_runtime_mode", "attention_backend", "mathematical_program", "source_program",
        }
        if "source_program" in normalized:
            if (not sparse_fa3 or variant != "mla_decode" or compute_dtype != "bf16"
                    or shape["q_len"] != 1
                    or (shape["qk_head_dim"], shape["v_head_dim"], shape["kv_lora_rank"]) != (576, 512, 512)
                    or normalized["causal"] is not True
                    or normalized["full_cuda_graphs"] is not False
                    or normalized["max_cudagraph_capture_size"] != 0
                    or any(name in normalized for name in ("attention_backend", "mathematical_program",
                        "graph_logical_tokens", "graph_model_input_tokens", "graph_runtime_mode"))):
                raise ValueError("sparse FA3 source requires its unpadded BF16 Q1 absorbed-MLA domain")
        if "mathematical_program" in normalized:
            programs = {"mla_decode": ("bf16_mla_decode_softmax_v1", "bf16_mla_causal_empty_rows_softmax_v1"),
                "mla_absorbed_prefill": ("bf16_mla_absorbed_prefill_softmax_v1",),
                "mla_prefill": ("bf16_mla_prefill_softmax_v1", "bf16_mla_context_softmax_v1", "bf16_mla_context_empty_rows_softmax_v1", "bf16_mla_one_shot_prefill_softmax_v1")}
            if variant not in programs or normalized["mathematical_program"] not in programs[variant]:
                raise ValueError("unknown or incompatible MLA mathematical program")
            if normalized["mathematical_program"] == "bf16_mla_one_shot_prefill_softmax_v1":
                required |= {"prefix_cache_threshold", "max_kv_chunk_capacity"}
        if variant in ("mla_prefill", "mla_absorbed_prefill"):
            optional.add("return_softmax_lse")
        elif variant == "kimi_k3_nope_context_attention":
            required.add("return_softmax_lse")
            optional.update({"output_target", "context_start"})
    elif kind is KernelKind.FLASH_ATTN:
        required = {
            "causal", "kv_layout", "kv_dist", "fa_ksteps",
            "fa_eff_mfrac", "q_lens", "kv_lens",
        }
        optional = {
            "model_input_q_lens", "model_input_kv_lens",
            "graph_runtime_mode", "mathematical_program", "ascend_fia_prefill_no_cache", "source_program",
        }
        if variant == "combine":
            optional.add("return_softmax_lse")
        if full_decode_fa3:
            required |= {"model_input_q_lens", "model_input_kv_lens",
                "graph_runtime_mode", "graph_model_input_tokens"}
            if (not required.issubset(normalized)
                    or variant not in ("decode", "combine") or compute_dtype not in ("bf16", "fp16")
                    or shape["q_len"] != 1 or shape["head_dim"] != 128
                    or normalized["causal"] is not True
                    or normalized["kv_layout"] != "paged_varlen"
                    or normalized["kv_dist"] != "uniform"
                    or normalized["graph_runtime_mode"] != "FULL"
                    or "mathematical_program" in normalized):
                raise ValueError("FA3 FULL decode requires its complete BF16/FP16 D128 causal paged-varlen source domain")
        elif sparse_fa3:
            if (variant != "combine" or compute_dtype != "bf16"
                    or shape["q_len"] != 1 or shape["head_dim"] != 512
                    or shape["num_kv_heads"] != 1 or normalized["kv_layout"] != "mla"
                    or normalized["causal"] is not True
                    or normalized["kv_dist"] not in ("uniform", "explicit")
                    or any(name in normalized for name in ("mathematical_program", "model_input_q_lens",
                        "model_input_kv_lens", "graph_runtime_mode"))):
                raise ValueError("sparse FA3 combine requires its unpadded BF16 Q1 source domain")
        elif "source_program" in normalized and (normalized["source_program"] != "ascend_0_18_full_fia_split_fuse"
                or variant not in ("prefill", "decode")
                or compute_dtype != "bf16" or "mathematical_program" in normalized):
            raise ValueError("native FIA source requires a compatible BF16 call without a mathematical program")
        if "mathematical_program" in normalized and (
                normalized["mathematical_program"] not in ("gqa_softmax_v1", "ascend_0_18_full_fia_softmax_v1")
                or variant not in ("prefill", "decode")):
            raise ValueError("unknown or incompatible FLASH_ATTN mathematical program")
        ascend_fia = ("mathematical_program" in normalized
            and normalized["mathematical_program"] == "ascend_0_18_full_fia_softmax_v1")
        native_split_fuse = ("source_program" in normalized
            and normalized["source_program"] == "ascend_0_18_full_fia_split_fuse")
        if ascend_fia or native_split_fuse:
            required.add("ascend_fia_prefill_no_cache")
            optional.remove("ascend_fia_prefill_no_cache")
            if "ascend_fia_prefill_no_cache" not in normalized or type(normalized["ascend_fia_prefill_no_cache"]) is not bool:
                raise ValueError("Ascend FULL FIA requires its explicit source prefill state")
        elif "ascend_fia_prefill_no_cache" in normalized:
            raise ValueError("Ascend FIA state requires its declared source program")
    elif (kind is KernelKind.FUSED_MOE
          and variant == "routed_experts"):
        required = {
            "tokens_per_expert", "local_tokens_per_expert",
            "expert_parallel_size", "expert_rank", "owned_experts",
            "expert_weight_tile_rows", "routed_scaling_factor",
            "routed_weight_dtype", "weight_group_size",
        }
        optional = {"swiglu_limit", "mathematical_program"}
        if ("mathematical_program" in normalized
                and normalized["mathematical_program"] == "sglang_0_5_19_fp8_fused_shared_swiglu_v1"):
            required.add("fused_shared_experts")
        if "swiglu_limit" not in normalized:
            normalized["swiglu_limit"] = None
        limit = normalized["swiglu_limit"]
        if limit is not None and (isinstance(limit, bool)
                or not isinstance(limit, (int, float))
                or not math.isfinite(limit) or limit <= 0):
            raise ValueError("routed_experts swiglu_limit must be finite positive or None")
        if (limit is not None and "routed_weight_dtype" in normalized
                and normalized["routed_weight_dtype"] != "mxfp4"):
            raise ValueError("clamped routed_experts currently requires MXFP4 weights")
        if "mathematical_program" in normalized and required.issubset(normalized):
            program = normalized["mathematical_program"]
            if program == "unquantized_bf16_swiglu_v1":
                valid = (compute_dtype == "bf16" and normalized["routed_weight_dtype"] == "bf16"
                    and normalized["weight_group_size"] == 0 and normalized["expert_parallel_size"] == 1
                    and normalized["expert_weight_tile_rows"] is None and limit is None
                    and normalized["routed_scaling_factor"] == 1.0 and "graph_runtime_mode" not in normalized)
            elif program == "unquantized_bf16_grouped_swiglu_v1":
                valid = (compute_dtype == "bf16" and normalized["routed_weight_dtype"] == "bf16"
                    and normalized["weight_group_size"] == 0
                    and normalized["expert_weight_tile_rows"] is None and limit is None
                    and "graph_runtime_mode" not in normalized)
            elif program == "unquantized_bf16_legacy_grouped_swiglu_v1":
                valid = (compute_dtype == "bf16" and normalized["routed_weight_dtype"] == "bf16"
                    and normalized["weight_group_size"] == 0 and normalized["expert_parallel_size"] == 2
                    and shape["experts"] % 2 == 0 and normalized["routed_scaling_factor"] == 1.0
                    and normalized["expert_weight_tile_rows"] is None and limit is None
                    and "graph_runtime_mode" not in normalized)
            elif program == "block_fp8_swiglu_v1":
                valid = (compute_dtype == "bf16" and normalized["routed_weight_dtype"] == "fp8"
                    and normalized["weight_group_size"] == 0 and normalized["expert_parallel_size"] == 1
                    and normalized["expert_weight_tile_rows"] is None and limit is None
                    and shape["hidden"] % 128 == shape["intermediate"] % 128 == 0
                    and "graph_runtime_mode" not in normalized)
            elif program == "sglang_0_5_19_fp8_fused_shared_swiglu_v1":
                valid = (compute_dtype == "bf16" and normalized["routed_weight_dtype"] == "fp8"
                    and normalized["weight_group_size"] == 0 and normalized["expert_parallel_size"] == 1
                    and normalized["expert_weight_tile_rows"] is None and limit is None
                    and (shape["experts"], shape["topk"], shape["hidden"], shape["intermediate"]) == (256, 8, 7168, 256)
                    and type(normalized["fused_shared_experts"]) is int and normalized["fused_shared_experts"] == 1
                    and normalized["routed_scaling_factor"] == 1.0 and "graph_runtime_mode" not in normalized)
            elif program == "block_fp8_replicated_ep_swiglu_v1":
                valid = (compute_dtype == "bf16" and normalized["routed_weight_dtype"] == "fp8"
                    and normalized["weight_group_size"] == 0 and normalized["expert_parallel_size"] == 8
                    and normalized["expert_weight_tile_rows"] is None and limit is None
                    and (shape["experts"], shape["topk"], shape["hidden"], shape["intermediate"]) == (256, 8, 6144, 2048)
                    and normalized["routed_scaling_factor"] == 2.5 and "graph_runtime_mode" not in normalized)
            elif program == "kimi_ep_marlin_swiglu_v1":
                valid = (compute_dtype == "bf16" and normalized["routed_weight_dtype"] == "int4"
                    and normalized["weight_group_size"] == 32
                    and type(normalized["expert_parallel_size"]) is int
                    and (normalized["expert_parallel_size"] == 1
                        or normalized["expert_parallel_size"] >= 2
                        and shape["experts"] % normalized["expert_parallel_size"] == 0)
                    and normalized["expert_weight_tile_rows"] is None and limit is None
                    and (shape["experts"], shape["topk"], shape["hidden"]) == (384, 8, 7168)
                    and ((normalized["expert_parallel_size"] == 1 and shape["intermediate"] in (2048, 1024, 512, 256))
                        or normalized["expert_parallel_size"] >= 2 and shape["intermediate"] == 2048)
                    and normalized["routed_scaling_factor"] == 1.0 and "graph_runtime_mode" not in normalized)
            elif program == "dsv4_mxfp4_w4a16_swiglu_v1":
                valid = (compute_dtype == "bf16" and normalized["routed_weight_dtype"] == "mxfp4"
                    and normalized["weight_group_size"] == 32 and normalized["expert_parallel_size"] == 1
                    and normalized["expert_weight_tile_rows"] is None and limit == 10.0
                    and (shape["experts"], shape["topk"], shape["hidden"], shape["intermediate"]) == (256, 6, 4096, 256)
                    and "graph_runtime_mode" not in normalized)
            else:
                valid = False
            if not valid:
                raise ValueError("mathematical experts require their exact unpadded source program")
    elif kind is KernelKind.FUSED_MOE and variant == "kimi_k3_marlin_experts":
        required = {
            "tokens_per_expert", "local_tokens_per_expert", "expert_parallel_size",
            "expert_rank", "owned_experts", "expert_weight_tile_rows",
            "activation_beta", "activation_linear_beta", "routed_scaling_factor",
        }
        if (shape["experts"], shape["topk"], shape["hidden"], shape["intermediate"]) != (896, 16, 3584, 3072):
            raise ValueError("K3 Marlin expert source contract requires exact checkpoint geometry at TP1")
        if ("expert_parallel_size" in normalized and normalized["expert_parallel_size"] != 1):
            raise ValueError("K3 Marlin expert source contract requires TP1 EP-off")
        if ("expert_weight_tile_rows" in normalized and normalized["expert_weight_tile_rows"] is not None):
            raise ValueError("K3 Marlin expert source has no assigned physical weight tile rows")
        for key, value in (("activation_beta", 4.0), ("activation_linear_beta", 25.0), ("routed_scaling_factor", 1.0)):
            if key in normalized and (type(normalized[key]) not in (int, float) or normalized[key] != value):
                raise ValueError("K3 Marlin expert source requires configured SiTU beta4/linear_beta25 and scale1")
    elif kind is KernelKind.FUSED_MOE:
        required = {
            "tokens_per_expert",
            "local_tokens_per_expert",
            "expert_parallel_size",
            "expert_rank",
            "owned_experts",
            "expert_weight_tile_rows",
        }
        optional = {
            "routed_scaling_factor", "routed_weight_dtype",
            "weight_group_size",
            # SGLang shared-experts fusion: the shared expert runs inside the
            # routed kernel as expert E with every token routed to it.
            "fused_shared_experts",
        }
    elif kind is KernelKind.UNFUSED_MOE:
        required = {"tokens_per_expert", "expert_weight_tile_rows"}
    elif kind is KernelKind.GEMM and variant == "kimi_k3_nope_bmm":
        required = {"source_position"}
    elif (kind is KernelKind.GEMM and variant == "batched_gemm"):
        required = {
            "batch_layout", "a_batch_stride_elements",
            "b_batch_stride_elements", "c_batch_stride_elements",
            "weight_dtype", "weight_group_size",
        }
        optional.add("mathematical_program")
        if "mathematical_program" in normalized and (
                normalized["mathematical_program"] != "batched_fp8_quantized_input_fp32_scales_v1"
                or compute_dtype != "bf16" or normalized["weight_dtype"] != "fp8"
                or normalized["weight_group_size"] != 128 or shape["k"] % 128 or shape["n"] % 128):
            raise ValueError("mathematical grouped FP8 requires explicit group128 scales and BF16 C")
    elif kind is KernelKind.GEMM and variant == "moe_router":
        required = {
            "score_function", "topk_method", "expert_group_count",
            "topk_group_count", "normalize_topk_probability",
            "weight_dtype", "weight_group_size",
        }
    elif kind is KernelKind.SPARSE_ATTN_INDEXER:
        if variant == "dsa_index_reuse":
            required = {
                "reuse_semantics", "topk_capacity", "topk_index_dtype",
            }
        elif variant in (
                "dsa_indexer_prefill", "dsa_indexer_decode",
                "dsa_indexer_mixed", "dsv4_indexer_prefill",
                "dsv4_indexer_decode"):
            required = {
                "causal", "score_accumulator_dtype", "score_function",
                "cache_layout", "cache_quantization", "q_lens", "kv_lens",
                "request_phases", "topk_capacity", "topk_index_dtype",
                "topk_sentinel",
            }
            if variant in ("dsv4_indexer_prefill", "dsv4_indexer_decode"):
                optional = {"context_parallel_size"}
                if "context_parallel_size" not in normalized:
                    normalized["context_parallel_size"] = None
            else:
                optional = {"mathematical_program", "context_parallel_size", "skip_topk_buffer_clear",
                    "clean_logits", "indexer_topk_backend", "model_execution_version",
                    "source_block_size", "source_block_table_inner_stride", "source_candidate_blocks",
                    "source_context_lens_layout", "source_cost_model", "source_decode_layout",
                    "source_deepgemm_snapshot", "source_indexer_input_stage", "source_indices",
                    "source_input_layout", "source_kv_cache_layout", "source_max_model_len",
                    "source_num_sms", "source_q_scale", "source_schedule_metadata",
                    "source_schedule_shape", "source_tie_refinement_policy", "source_component",
                    "source_parallel_mode", "source_dense_mha_metadata_layer", "source_topk_clear_state",
                    "source_prefill_chunks", "source_chunk_policy", "source_topk_cluster_size", "source_topk_route", "source_sm_budget"}
                for name in ("mathematical_program", "context_parallel_size",
                        "skip_topk_buffer_clear", "source_cost_model"):
                    if name not in normalized:
                        normalized[name] = None
                if normalized["source_cost_model"] not in (None, "vllm_0_30_bounded_stage_roofline_v1"):
                    raise ValueError("unknown DSA source_cost_model")
                if normalized["source_cost_model"] is not None:
                    required |= {"clean_logits", "indexer_topk_backend", "model_execution_version",
                        "source_block_size", "source_block_table_inner_stride", "source_candidate_blocks",
                        "source_context_lens_layout", "source_decode_layout", "source_deepgemm_snapshot",
                        "source_indexer_input_stage", "source_indices", "source_input_layout",
                        "source_kv_cache_layout", "source_max_model_len", "source_num_sms", "source_q_scale",
                        "source_schedule_metadata", "source_schedule_shape", "source_tie_refinement_policy",
                        "source_parallel_mode", "source_dense_mha_metadata_layer", "source_topk_clear_state",
                        "source_sm_budget"}
                    if variant in ("dsa_indexer_prefill", "dsa_indexer_mixed"):
                        required.add("source_chunk_policy")
                    if variant in ("dsa_indexer_decode", "dsa_indexer_mixed"):
                        if "source_topk_route" not in normalized:
                            raise ValueError(
                                "DSA decode TopK route requires its complete launch-bound fields")
                        required.add("source_topk_route")
                        route = normalized["source_topk_route"]
                        if not isinstance(route, dict) or set(route) != {
                                "route", "cluster", "vector_width", "block_threads",
                                "dynamic_shared_bytes", "algorithm"}:
                            raise ValueError(
                                "DSA decode TopK route requires its complete launch-bound fields")
                        if type(route["route"]) is not str:
                            raise ValueError("DSA decode TopK route name must be a string")
                        if route["block_threads"] != 1024:
                            raise ValueError("DSA decode TopK route block threads must equal its source constant 1024")
                        if not isinstance(route["algorithm"], str):
                            raise ValueError("DSA decode TopK route algorithm must be a string")
                        if route["route"] == "cooperative":
                            if (type(route["cluster"]) is not int
                                    or route["cluster"] not in (2, 4, 8)
                                    or route["vector_width"] is not None
                                    or route["dynamic_shared_bytes"] is not None
                                    or route["algorithm"] != (
                                        "vllm_0_30_sm90_dsa_cooperative_topk_cs"
                                        f"{route['cluster']}_k{normalized['topk_capacity']}")):
                                raise ValueError(
                                    "cooperative TopK route facts are inconsistent with its launch")
                        elif route["route"] == "filtered_persistent":
                            if (route["cluster"] is not None
                                    or type(route["vector_width"]) is not int
                                    or route["vector_width"] not in (1, 2, 4)
                                    or type(route["dynamic_shared_bytes"]) is not int
                                    or route["dynamic_shared_bytes"] != 131072
                                    or route["algorithm"] != (
                                        "vllm_0_30_sm90_dsa_filtered_topk_vec"
                                        f"{route['vector_width']}_k{normalized['topk_capacity']}")):
                                raise ValueError(
                                    "filtered persistent TopK route facts are inconsistent with its launch")
                        else:
                            raise ValueError("unknown DSA decode TopK route")
                        if ("source_topk_cluster_size" in normalized
                                and normalized["source_topk_cluster_size"] != route["cluster"]):
                            raise ValueError(
                                "source_topk_cluster_size disagrees with the launch-bound TopK route")
                if "mathematical_program" in normalized and normalized["mathematical_program"] == "dsa_weighted_relu_specdecode_v1":
                    required |= {"num_speculative_tokens", "decode_threshold"}
        else:
            raise AssertionError(
                f"unhandled SPARSE_ATTN_INDEXER variant: {variant!r}")
    elif kind is KernelKind.SPARSE_ATTN:
        required = {
            "causal", "indexer_type", "kv_layout", "kv_dist", "q_lens",
            "kv_lens", "selected_kv_lens", "topk_capacity",
            "topk_index_dtype",
        }
        if variant in ("dsa_sparse_decode", "dsa_sparse_prefill", "dsa_sparse_mixed"):
            optional |= {"model_execution_version", "source_attention_backend", "source_block_size",
                "source_block_stride_rows", "source_component", "source_cost_model",
                "source_extra_cache", "source_flashmla_snapshot", "source_hisparse",
                "source_index_inputs_layout", "source_index_remap_state", "source_index_remap_stream",
                "source_kv_cache_layout", "source_metadata_layout", "source_parallel_mode",
                "source_q_backing_heads", "source_q_input_layout", "source_attn_sink",
                "tensor_iterator_source"}
            if "source_cost_model" not in normalized:
                normalized["source_cost_model"] = None
            if normalized["source_cost_model"] not in (None, "vllm_0_30_bounded_stage_roofline_v1"):
                raise ValueError("unknown DSA sparse source_cost_model")
            if normalized["source_cost_model"] is not None:
                required |= {"model_execution_version", "source_attention_backend", "source_block_size",
                    "source_block_stride_rows", "source_component", "source_extra_cache",
                    "source_flashmla_snapshot", "source_hisparse", "source_index_inputs_layout",
                    "source_index_remap_state", "source_index_remap_stream", "source_kv_cache_layout",
                    "source_metadata_layout", "source_parallel_mode", "source_q_backing_heads",
                    "source_q_input_layout", "source_attn_sink", "tensor_iterator_source",
                    "source_tie_refinement_policy"}
        if variant in ("dsa_sparse_prefill", "dsa_sparse_mixed"):
            optional.add("mathematical_program")
            if "mathematical_program" in normalized and (
                    normalized["mathematical_program"] != "bf16_latent_force_mqa_v1"
                    or compute_dtype != "bf16" or shape["qk_head_dim"] != 576 or shape["v_head_dim"] != 512):
                raise ValueError("DSA prefill math requires its BF16 latent force-MQA program")
        if variant == "dsa_sparse_decode" and "mathematical_program" in normalized:
            program = normalized["mathematical_program"]
            required.add("mathematical_program")
            if program == "bf16_latent_specdecode_v1":
                required |= {"num_speculative_tokens", "decode_threshold"}
            elif program == "bf16_latent_reused_topk_v1":
                required.add("reused_selection_kv_lens")
            else:
                raise ValueError("unknown DSA decode mathematical program")
            if compute_dtype != "bf16" or shape["qk_head_dim"] != 576 or shape["v_head_dim"] != 512:
                raise ValueError("DSA decode math requires BF16 latent576/512 source geometry")
        if variant in ("dsv4_sparse_prefill", "dsv4_sparse_decode"):
            # DeepSeek-V4 attends the SWA window from the full-resolution
            # cache alongside the selected compressed states, so the window
            # lengths are part of the contract.
            required |= {"swa_kv_lens"}
            optional |= {"compressed_selection", "mathematical_program"}
            if "mathematical_program" in normalized:
                program = normalized["mathematical_program"]
                if program not in ("dsv4_sparse_softmax_flops_bytes_v1", "vllm_0_30_dspark_swa_noncausal_softmax_v1"):
                    raise ValueError("unknown V4 sparse mathematical program")
                if program == "vllm_0_30_dspark_swa_noncausal_softmax_v1":
                    required.add("num_speculative_tokens")
                    if (variant != "dsv4_sparse_decode" or compute_dtype != "bf16"
                            or shape["compress_ratio"] != 1 or shape["window_size"] != 128
                            or shape["num_query_heads"] not in (64, 128)
                            or shape["qk_head_dim"] != 512 or shape["v_head_dim"] != 512):
                        raise ValueError("DSpark sparse math requires BF16 512/512 SWA128 uncompressed decode")
            if "compressed_selection" not in normalized:
                normalized["compressed_selection"] = "candidate_upper_bound"
        if variant == "dsv4_sparse_decode":
            # The split-KV decode writes one FP32 partial (value + LSE
            # scratch) per (request, split) for the combine kernel; the
            # per-request split count is a discrete FlashMLA tile-scheduler
            # launch fact (flash_mla_sched_meta.num_splits), so it is part
            # of the contract rather than a predictor estimate.
            if "mathematical_program" in normalized:
                optional.add("split_counts")
                if "split_counts" not in normalized:
                    normalized["split_counts"] = None
            else:
                required |= {"split_counts"}
    elif kind is KernelKind.KV_CACHE:
        if variant == "paged_kv_cache":
            required = {
                "cache_layout", "cache_head_layout", "kv_cache_dtype",
                "cache_storage_matches_input", "slot_mapping_all_valid",
                "vector_copy_base_alignment_bytes",
            }
        elif variant == "mla_kv_cache":
            required = {"cache_layout", "cache_write_semantics"}
        elif variant == "dsa_fp8_index_cache":
            required = {
                "cache_layout", "cache_write_semantics", "quantization",
            }
        elif variant == "dsv4_compressed_state_cache":
            required = {
                "cache_layout", "cache_write_semantics", "quantization",
                "state_old_lens", "state_new_lens",
            }
            optional = {"finish_stage"}
            if "finish_stage" not in normalized:
                normalized["finish_stage"] = None
        elif variant == "dsv4_qnorm_rope_kv_insert":
            required = {
                "cache_layout", "cache_write_semantics", "quantization",
                "slot_mapping_all_valid",
            }
            optional = {"mathematical_program"}
            if "mathematical_program" in normalized:
                if normalized["mathematical_program"] != "vllm_0_30_dspark_partial_context_insert_v1":
                    raise ValueError("unknown DSpark partial context insert mathematical program")
                required.add("valid_token_count")
        elif variant == "dsv4_compressor_state_cache":
            required = {
                "cache_layout", "cache_write_semantics",
            }
        else:
            raise AssertionError(
                f"unhandled KV cache update variant: {variant!r}")
    elif kind is KernelKind.VECTOR and variant in (
            "elementwise", "elementwise_pytorch_vec4"):
        required = {"reads", "writes", "operation"}
        optional = {"helper_launch", "tensor_iterator_source", "copy_dispatch", "mathematical_program"}
        if ("mathematical_program" in normalized
                and normalized["mathematical_program"] not in ("vllm_0_30_fa3_aot_tail_clear", "vllm_0_30_dspark_typed_role_bytes_v1")) and (
                normalized["mathematical_program"] != "dense_bf16_source_flops_bytes_v1"
                or compute_dtype != "bf16" or variant != "elementwise"
                or "operation" not in normalized
                or normalized["operation"] not in ("add", "mul", "unary", "copy", "fill", "read", "cast_bf16_float32")):
            raise ValueError("elementwise source math requires its explicit BF16 program")
        if ("mathematical_program" in normalized and normalized["mathematical_program"] == "vllm_0_30_dspark_typed_role_bytes_v1"
                and (variant != "elementwise" or compute_dtype not in ("bool", "int32", "int64", "fp32", "bf16")
                    or normalized["operation"] not in ("read", "fill"))):
            raise ValueError("DSpark typed payload requires explicit read/write-only roles in their original dtype")
        materialization_fields = {
            "input_materialization",
            "input_outer_groups",
            "input_rows_per_group",
            "input_group_stride_elements",
            "input_row_stride_elements",
        }
        present_materialization_fields = (
            materialization_fields & set(normalized))
        if present_materialization_fields:
            if ("operation" not in normalized
                    or normalized["operation"] != "copy"):
                raise ValueError(
                    "VECTOR grouped input materialization requires copy")
            required |= materialization_fields
    elif kind is KernelKind.VECTOR and variant in ("dsv32_fused_norm_rope", "dsv32_fused_q"):
        required = {"has_indexer", "index_rope_interleave"}
        if variant == "dsv32_fused_norm_rope":
            required |= {"mla_cache_dtype", "cache_slot_scope"}
            if "cache_slot_scope" in normalized and normalized["cache_slot_scope"] == "no_metadata_no_cache_slots":
                required |= {"source_no_attention_metadata", "index_k_output_present", "topk_active_fill"}
        else:
            required.add("quantize_mqa")
    elif kind is KernelKind.VECTOR and variant == "layernorm":
        required = {"affine", "epsilon"}
    elif kind is KernelKind.VECTOR and variant == "mla_rope":
        required = {
            "rope_layout", "rope_type", "rope_theta", "rope_factor",
            "rope_original_max_position_embeddings", "rope_beta_fast",
            "rope_beta_slow", "rope_mscale", "rope_mscale_all_dim",
        }
    elif kind is KernelKind.VECTOR and variant == "dsv4_short_context_topk":
        required = {"position_bytes"}
    elif kind is KernelKind.VECTOR and variant in (
            "dsv32_draft_argmax", "dsv32_draft_id_cast", "dsv32_draft_row_select", "dsv32_draft_buffer_copy", "dsv32_draft_token_stack", "dsv32_draft_id_scatter", "dsv32_draft_prepare_next_ids", "dsv32_draft_prepare_sample_indices", "dsv32_draft_seq_rollback", "dsv32_draft_step_metadata"):
        required = {"sampling_source"}
        if variant == "dsv32_draft_row_select":
            required |= {"selection_role", "selected_indices"}
        elif variant == "dsv32_draft_buffer_copy":
            required.add("copy_role")
        elif variant == "dsv32_draft_id_scatter":
            required.add("selected_indices")
        elif variant in ("dsv32_draft_prepare_sample_indices", "dsv32_draft_seq_rollback"):
            required.add("index_preparation_mode")
        elif variant == "dsv32_draft_step_metadata":
            required.add("positions_alias_output")
    elif kind is KernelKind.VECTOR and variant == "vllm_rejection_logits_materialize":
        required = {"sampling_source", "materialization_stage", "selected_indices"}
    elif kind is KernelKind.VECTOR and variant == "vllm_attention_state_merge":
        required = {"mathematical_program", "output_lse"}
        optional = {"empty_prefix_rows"}
    elif kind is KernelKind.VECTOR and variant == "kimi_k3_nope_context":
        required = {"stage", "slot_mapping", "cache_capacity_tokens", "rms_norm_eps", "context_block_table"}
        optional = {"context_start", "context_length"}
    elif kind is KernelKind.VECTOR and variant in ("kimi_k3_nope_prefill", "kimi_k3_nope_decode", "kimi_k3_nope_mixed"):
        required = {"stage", "slot_mapping", "cache_capacity_tokens", "rms_norm_eps"}
    elif kind is KernelKind.VECTOR and variant in ("kimi_k3_kda_prefill", "kimi_k3_kda_mixed", "kimi_k3_kda_context_prefill"):
        required = {"stage", "state_indices", "state_pool_slots", "query_start_loc", "has_initial_state", "gate_lower_bound"}
        if "source_component" in normalized:
            if not _is_kda_chunk_source(kind, variant, normalized):
                raise ValueError("KDA chunk requires its exact source component")
            required |= _KDA_CHUNK_SOURCE_FIELDS
    elif kind is KernelKind.VECTOR and variant == "kimi_k3_kda_decode":
        required = {"stage", "state_indices", "state_pool_slots", "gate_lower_bound"}
    elif kind is KernelKind.VECTOR and variant == "kimi_k3_situ":
        required = {"activation_beta", "activation_linear_beta"}
    elif kind is KernelKind.VECTOR and variant == "kimi_k3_attn_res":
        required = {"num_blocks", "block_write_idx", "block_capacity", "has_delta", "apply_output_norm"}
    elif kind is KernelKind.VECTOR and variant == "vllm_plain_greedy_sample":
        required = {"sampling_source", "stage"}
        optional = {"source_num_compute_units", "source_max_threads_per_unit"}
    elif kind is KernelKind.VECTOR and variant == "vllm_embedding_id_widen":
        required = {"embedding_source"}
        optional = {"copy_dispatch"}
    elif kind is KernelKind.VECTOR and variant == "vllm_masked_embedding":
        required = {"stage", "vocab_size", "tensor_parallel_size", "tensor_parallel_rank"}
        optional = {"mask_program"}
        if "mask_program" not in normalized:
            normalized["mask_program"] = "vllm_0_30"
    elif kind is KernelKind.VECTOR and variant == "vllm_mtp_eager":
        required = {"operation", "model_execution_version"}
        operation = normalized["operation"]
        if normalized["model_execution_version"] == "sglang_0_5_19":
            optional = {"eagle_source"}
            if "eagle_source" not in normalized:
                normalized["eagle_source"] = None
        if operation == "dspark_swa_indices":
            required |= {"num_speculative_tokens", "kv_lens"}
        elif operation in ("dspark_greedy_verify", "dspark_roundtrip_prepare"):
            required |= {"num_speculative_tokens", "accepted_drafts"}
        if operation == "eagle_tree_verify":
            required.add("accepted_drafts")
        elif operation == "eagle_eh_norm":
            required.add("epsilon")
        elif operation == "eagle_positions":
            optional.add("query_lens")
            if "query_lens" not in normalized:
                normalized["query_lens"] = (shape["width"],) * shape["rows"]
        elif operation == "eagle_topk1_finalize":
            required.add("write_draft_token")
        if operation in ("gather", "scatter"):
            required |= {"index_bytes", "selected_indices"}
        elif operation == "masked_fill":
            required.add("masked_rows")
        elif operation == "rejection":
            required |= {"verified_tokens", "accepted_tokens"}
        elif operation == "prepare_next_padded":
            required |= {"vocab_size", "valid_sampled_counts", "discarded_rows"}
        elif operation == "step_metadata":
            required |= {"block_size", "max_model_len", "block_table_columns"}
    elif kind is KernelKind.VECTOR and variant == "vllm_mtp_position_mask":
        required = {"stage", "model_execution_version", "masked_rows"}
    elif kind is KernelKind.VECTOR and variant == "vllm_logits_gather_reorder":
        required = {"model_execution_version"}
    elif kind is KernelKind.VECTOR and variant == "vllm_logits_fp32_cast":
        required = {"sampling_source"}
        optional = {"copy_dispatch", "copy_input_contiguous"}
    elif kind is KernelKind.VECTOR and variant == "vllm_logits_row_select":
        required = {"model_execution_version", "index_source", "selected_indices"}
        optional = {"disable_index_broadcast"}
    elif kind is KernelKind.VECTOR and variant == "vocab_parallel_embedding":
        required = {"vocab_size", "tensor_parallel_size", "tensor_parallel_rank"}
        optional = {"owned_token_rows"}
        if "owned_token_rows" not in normalized:
            normalized["owned_token_rows"] = None
    elif kind is KernelKind.VECTOR and variant in ("dsv32_moe_route", "kimi_moe_route", "kimi_k3_moe_route"):
        required = {"expert_id_selection", "routed_scaling_factor", "score_function",
                    "expert_group_count", "topk_group_count", "normalize_topk_probability"}
        if variant in ("dsv32_moe_route", "kimi_moe_route"):
            optional = {"mathematical_program"}
            if "mathematical_program" not in normalized:
                normalized["mathematical_program"] = None
    elif kind is KernelKind.VECTOR and variant == "dsv4_moe_route":
        required = {"expert_id_selection", "routed_scaling_factor"}
        optional = {"routing_source"}
        if "routing_source" not in normalized:
            normalized["routing_source"] = None
    elif kind is KernelKind.VECTOR and variant == "dsv4_inv_rope_fp8_quant":
        required = {"quant_group_size"}
    elif kind is KernelKind.VECTOR and variant == "dsv4_indexer_q_rope_quant":
        required = set()
    elif kind is KernelKind.VECTOR and variant == "rope":
        required = {"is_neox", "key_present"}
    elif kind is KernelKind.VECTOR and variant == "qkv_rmsnorm_rope":
        required = {"bias", "is_neox"}
    elif kind is KernelKind.VECTOR and variant == "moe_fill":
        required = {"fill_role"}
    elif kind is KernelKind.VECTOR and variant == "moe_cast":
        required = {"cast_role"}
    elif kind is KernelKind.VECTOR and variant == "moe_swiglu":
        required = {"routed_rows"}
    elif (kind is KernelKind.VECTOR
          and variant == "moe_compute_expert_tokens"):
        required = {"tokens_per_expert"}
    elif kind is KernelKind.VECTOR and variant in (
            "moe_silu", "moe_mul", "moe_sum", "moe_reduce"):
        required = {"work_fraction"}
        if variant == "moe_sum":
            optional = {"routed_weights_dtype"}
            if "routed_weights_dtype" not in normalized:
                normalized["routed_weights_dtype"] = None
            if normalized["routed_weights_dtype"] not in (None, "fp32"):
                raise ValueError("moe_sum routed weights must be explicit FP32 or absent")
    elif kind is KernelKind.VECTOR and variant == "reduce":
        required = {"reduction_factor"}
    elif kind is KernelKind.VECTOR and variant == "gather":
        required = {
            "gather_factor", "index_count", "index_bytes", "scattered"}
        optional = {"materialized_output", "mathematical_program", "index_select_source", "materialization_source"}
        if "materialized_output" not in normalized:
            normalized["materialized_output"] = False
    elif kind is KernelKind.VECTOR and variant == "gemm_reduce":
        required = {"split_k"}
        optional = {"source_gemm_runtime", "source_gemm_shape"}
        if (("source_gemm_runtime" in normalized) != ("source_gemm_shape" in normalized)
                and not (compute_dtype == "bf16" and "source_gemm_shape" in normalized
                    and "source_gemm_runtime" not in normalized)):
            raise ValueError("aggregate reducer requires both runtime and parent GEMM shape")
        if "source_gemm_shape" in normalized:
            parent = normalized["source_gemm_shape"]
            if (not isinstance(parent, Mapping) or set(parent) != {"b", "m", "n", "k"}
                    or any(type(value) is not int or value <= 0 for value in parent.values())
                    or parent["b"] * parent["m"] * parent["n"] != shape["element_count"]):
                raise ValueError("aggregate reducer parent shape disagrees with output elements")
    elif kind is KernelKind.VECTOR and variant == "fp8_group_quant":
        required = {"group_size"}
        optional = {"source_program", "input_resource"}
        if "input_resource" not in normalized:
            normalized["input_resource"] = "dram"
        if normalized["input_resource"] not in ("dram", "buffer"):
            raise ValueError("FP8 quantizer input resource must be dram or buffer")
        if "source_program" not in normalized:
            normalized["source_program"] = None
    elif kind is KernelKind.VECTOR and variant == "gemm_elementwise":
        required = set()
    elif kind in (KernelKind.GEMM, KernelKind.EAGER_ATTN):
        required = set()
        optional = {"helper_launch"}
    elif kind is KernelKind.VECTOR and variant in {
            "rmsnorm", "add_rmsnorm", "rotary", "activation",
            "silu", "softmax", "flash_mla_metadata", "fa3_prepare_varlen",
            "moe_init_routing", "moe_init_routing_v2",
            "moe_init_routing_v3",
            "moe_finalize_routing",
            "moe_transpose", "moe_cast", "moe_range", "moe_topk",
            "moe_softmax", "moe_gating", "moe_gating_topk", "moe_abs",
            "moe_token_unpermute", "moe_act_and_mul",
            "moe_align", "moe_align_opt_128", "moe_align_small_batch",
            "moe_count_and_sort", "moe_routing_weight_norm", "moe_routing_weight_scale",
            "moe_routed_id_extent", "moe_expert_map_lookup",
            "moe_router_logit_cast", "moe_routed_id_widen", "moe_enable_if",
            "dsv32_fused_norm_rope", "dsv32_fused_q", "dsv4_q_kv_rmsnorm", "dsv32_eh_norm", "dsv32_embed_eh_norm", "dsv32_embed_norm",
            "mhc_pre", "mhc_post_pre", "mhc_prenorm", "mhc_prenorm_broadcast", "mhc_post_prenorm",
            "mhc_pre_norm", "mhc_pre_norm_broadcast", "mhc_post", "hc_head",
    }:
        required = set()
        if kind is KernelKind.VECTOR and variant == "moe_align":
            optional = {"alignment_block_size", "alignment_capacity", "tokens_per_expert"}
    else:
        raise ValueError(
            f"unhandled {kind.value}/{variant} attribute contract")
    if kind is KernelKind.VECTOR and variant == "fa3_prepare_varlen":
        optional.update({"source_program", "num_splits_static"})
        # The upper bound is a runtime source argument, distinct from the
        # prepared per-request split result. Unknown bounds remain unpriced.
        if "num_splits_static" not in normalized:
            normalized["num_splits_static"] = None
        split_bound = normalized["num_splits_static"]
        if (split_bound is not None
                and (type(split_bound) is not int or split_bound <= 0)):
            raise ValueError("FA3 prepare num_splits_static must be positive or None")
        if "source_program" in normalized:
            required |= {"use_dynamic_split", "head_swizzle"}
            if (normalized["source_program"] != "vllm_0_30_fa3_full_aot_prepare"
                    or compute_dtype != "int32" or not required.issubset(normalized)
                    or type(normalized["use_dynamic_split"]) is not bool
                    or normalized["use_dynamic_split"] != (shape["batch"] <= 992)
                    or normalized["head_swizzle"] is not True):
                raise ValueError("FA3 FULL AOT prepare requires INT32, exact dynamic-layout flag and head swizzle")
    if kind is KernelKind.VECTOR and variant in ("elementwise", "elementwise_pytorch_vec4"):
        optional.add("mathematical_program")
        if ("mathematical_program" in normalized
                and normalized["mathematical_program"] == "vllm_0_30_fa3_aot_tail_clear"):
            if (variant != "elementwise"
                    or normalized["mathematical_program"] != "vllm_0_30_fa3_aot_tail_clear"
                    or compute_dtype != "int32" or not required.issubset(normalized)
                    or normalized["operation"] != "fill"
                    or type(normalized["reads"]) is not int or normalized["reads"] != 0
                    or type(normalized["writes"]) is not int or normalized["writes"] != 1):
                raise ValueError("FA3 AOT metadata clear requires the declared INT32 fill mathematical program")
    dense_fp16_variants = {"gather", "rmsnorm", "add_rmsnorm", "silu", "rotary"}
    if ((kind is KernelKind.VECTOR and variant in dense_fp16_variants)
            or (kind is KernelKind.KV_CACHE and variant == "paged_kv_cache")):
        optional.add("mathematical_program")
        if ("mathematical_program" in normalized
                and normalized["mathematical_program"] in (
                    "dense_fp16_source_flops_bytes_v1", "dense_bf16_source_flops_bytes_v1")):
            source_dtype = {"dense_fp16_source_flops_bytes_v1": "fp16",
                "dense_bf16_source_flops_bytes_v1": "bf16"}[normalized["mathematical_program"]]
            if compute_dtype != source_dtype:
                raise ValueError(f"dense {source_dtype.upper()} source math requires homogeneous {source_dtype.upper()} storage")
            if variant == "silu" and "swiglu_limit" in normalized and normalized["swiglu_limit"] is not None:
                raise ValueError("dense FP16 source math requires ordinary unclamped SiLU/mul")
            if kind is KernelKind.VECTOR and variant == "rotary":
                rotary_fields = {"seq_len", "head_dim", "num_query_heads", "num_kv_heads"}
                if (not rotary_fields.issubset(shape) or shape["element_count"] != shape["seq_len"]
                        * (shape["num_query_heads"] + shape["num_kv_heads"]) * shape["head_dim"]):
                    raise ValueError("dense FP16 RoPE math requires its complete full-head Q/K source geometry")
            if kind is KernelKind.KV_CACHE:
                if (not required.issubset(normalized) or normalized["kv_cache_dtype"] != "auto"
                        or normalized["cache_storage_matches_input"] is not True
                        or normalized["slot_mapping_all_valid"] is not True):
                    raise ValueError("dense FP16 KV source math requires matching auto cache storage and valid slots")
        elif "mathematical_program" in normalized and variant != "gather":
            raise ValueError("unsupported dense source mathematical program")
    if kind is KernelKind.VECTOR and variant in ("moe_init_routing", "moe_abs", "moe_token_unpermute") and "mathematical_program" in normalized:
        required.update(("mathematical_program", "local_experts", "routed_rows"))
        if (normalized["mathematical_program"] != "ascend_0_18_moe_dispatch_math_v1"
                or "local_experts" not in normalized or type(normalized["local_experts"]) is not int
                or normalized["local_experts"] * 2 != shape["experts"]
                or "routed_rows" not in normalized or type(normalized["routed_rows"]) is not int
                or not 0 <= normalized["routed_rows"] <= shape["tokens"] * shape["topk"]):
            raise ValueError("Ascend0.18 dispatch requires local EP2 expert/count metadata")
    if kind is KernelKind.VECTOR and variant == "moe_topk":
        optional.update(("source_program", "normalize_topk_probability", "mathematical_program"))
        if "source_program" in normalized:
            if (normalized["source_program"] not in ("vllm_0_30_bf16_softmax_topk", "vllm_0_18_bf16_softmax_topk", "vllm_0_24_bf16_softmax_topk", "vllm_0_11_fp32_softmax_topk", "vllm_0_11_fp32_sigmoid_bias_topk",
                    "vllm_0_11_bf16_sigmoid_bias_grouped_topk", "vllm_0_27_1_bf16_sigmoid_bias_grouped_topk", "sglang_0_5_19_fp32_sigmoid_bias_grouped_topk")
                    or "normalize_topk_probability" not in normalized
                    or type(normalized["normalize_topk_probability"]) is not bool):
                raise ValueError("source softmax top-k requires its explicit program and boolean normalization")
            if normalized["source_program"] == "vllm_0_18_bf16_softmax_topk":
                required.add("stage")
                if "stage" not in normalized or normalized["stage"] not in ("route", "owner_mask"):
                    raise ValueError("Ascend softmax routing requires its route or owner_mask stage")
        elif "normalize_topk_probability" in normalized:
            raise ValueError("top-k normalization requires an explicit source program")
        if ("source_program" in normalized
                and normalized["source_program"] in ("vllm_0_11_bf16_sigmoid_bias_grouped_topk", "vllm_0_27_1_bf16_sigmoid_bias_grouped_topk", "sglang_0_5_19_fp32_sigmoid_bias_grouped_topk")):
            grouped_fields = {"expert_group_count", "topk_group_count", "routed_scaling_factor"}
            required.update(grouped_fields | {"mathematical_program"})
            optional.update(grouped_fields)
            if normalized["source_program"] == "sglang_0_5_19_fp32_sigmoid_bias_grouped_topk":
                required.add("fused_shared_experts")
                if "fused_shared_experts" not in normalized or type(normalized["fused_shared_experts"]) is not int or normalized["fused_shared_experts"] != 1:
                    raise ValueError("SGLang grouped routing requires its one fused shared expert output slot")
            if not grouped_fields <= normalized.keys():
                raise ValueError("DSV3 grouped top-k requires explicit group and scaling declarations")
            if (type(normalized["expert_group_count"]) is not int or normalized["expert_group_count"] != 8
                    or type(normalized["topk_group_count"]) is not int or normalized["topk_group_count"] != 4
                    or shape["experts"] != 256 or shape["topk"] != 8):
                raise ValueError("DSV3 grouped top-k requires 256 experts, top8 and 4 of 8 groups")
            scale = normalized["routed_scaling_factor"]
            if (type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0):
                raise ValueError("DSV3 grouped top-k requires a finite positive source scale")
        if "mathematical_program" in normalized:
            if "source_program" not in normalized:
                raise ValueError("mathematical top-k requires an explicit source program")
            program = normalized["source_program"]
            if program in ("vllm_0_30_bf16_softmax_topk", "vllm_0_18_bf16_softmax_topk", "vllm_0_24_bf16_softmax_topk", "vllm_0_11_fp32_softmax_topk"):
                valid = (normalized["mathematical_program"] == ("fp32_softmax_topk_flops_bytes_v1"
                    if program == "vllm_0_11_fp32_softmax_topk" else "bf16_softmax_topk_public_math_v1"
                    if program in ("vllm_0_18_bf16_softmax_topk", "vllm_0_24_bf16_softmax_topk") else "bf16_softmax_topk_flops_bytes_v1")
                    and shape["experts"] in (2, 128))
                if not valid:
                    raise ValueError("mathematical softmax top-k requires the explicit 2/128-expert source program")
            elif program == "vllm_0_11_fp32_sigmoid_bias_topk":
                valid = normalized["mathematical_program"] == "fp32_sigmoid_bias_topk_flops_bytes_v1"
            elif program in ("vllm_0_11_bf16_sigmoid_bias_grouped_topk", "vllm_0_27_1_bf16_sigmoid_bias_grouped_topk", "sglang_0_5_19_fp32_sigmoid_bias_grouped_topk"):
                valid = normalized["mathematical_program"] == ("fp32_sigmoid_bias_grouped_topk_flops_bytes_v1"
                    if program == "sglang_0_5_19_fp32_sigmoid_bias_grouped_topk" else "bf16_sigmoid_bias_grouped_topk_flops_bytes_v1")
            else:
                raise AssertionError("unhandled explicit top-k source")
            if not valid or not 0 < shape["topk"] <= shape["experts"]:
                raise ValueError("mathematical top-k requires its matching source program and valid expert extent")
    if kind is KernelKind.VECTOR and variant in ("rmsnorm", "dsv4_q_kv_rmsnorm"):
        optional.add("epsilon")
        if "epsilon" not in normalized:
            normalized["epsilon"] = None
        epsilon = normalized["epsilon"]
        if epsilon is not None:
            if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float))
                    or not math.isfinite(epsilon) or epsilon <= 0):
                raise ValueError(f"VECTOR/{variant} epsilon must be positive finite or None")
            normalized["epsilon"] = float(epsilon)
    if kind is KernelKind.VECTOR and variant in ("mhc_pre", "mhc_post_pre"):
        required = {"mathematical_program", "sinkhorn_iters"}
        if (not required.issubset(normalized) or normalized["mathematical_program"] != "vllm_mhc_wrapper_math_v1"
                or type(normalized["sinkhorn_iters"]) is not int or normalized["sinkhorn_iters"] < 1):
            raise ValueError("mHC wrapper requires its mathematical program and positive Sinkhorn iterations")
    if kind is KernelKind.VECTOR and variant in ("mhc_prenorm", "mhc_prenorm_broadcast"):
        optional.add("source_prenorm_backend")
        if "source_prenorm_backend" not in normalized:
            normalized["source_prenorm_backend"] = None
        backend = normalized["source_prenorm_backend"]
        if backend not in (None, "tilelang", "deep_gemm"):
            raise ValueError("mHC source_prenorm_backend must be tilelang, deep_gemm or None")
        if backend == "tilelang" and (variant == "mhc_prenorm_broadcast" or shape["n_splits"] != 1):
            raise ValueError("mHC TileLang prenorm requires optional nonbroadcast single-partial source")
    if kind is not KernelKind.FLASH_ATTN:
        present_graph_domain_fields = graph_token_domain_fields & set(
            normalized)
        if present_graph_domain_fields:
            if present_graph_domain_fields != graph_token_domain_fields:
                raise ValueError(
                    f"{kind.value}/{variant} graph token domains must be "
                    f"complete: present={sorted(present_graph_domain_fields)}")
            logical_tokens = normalized["graph_logical_tokens"]
            model_input_tokens = normalized["graph_model_input_tokens"]
            runtime_mode = normalized["graph_runtime_mode"]
            for name, value in (
                    ("graph_logical_tokens", logical_tokens),
                    ("graph_model_input_tokens", model_input_tokens)):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(
                        f"{kind.value}/{variant} {name} must be an integer")
                if value <= 0:
                    raise ValueError(
                        f"{kind.value}/{variant} {name} must be positive")
            if model_input_tokens < logical_tokens:
                raise ValueError(
                    f"{kind.value}/{variant} model-input tokens cannot "
                    "shrink logical work")
            if runtime_mode not in ("NONE", "FULL", "PIECEWISE"):
                raise ValueError(
                    f"{kind.value}/{variant} graph_runtime_mode is invalid")
            if runtime_mode == "NONE" and logical_tokens != model_input_tokens:
                raise ValueError(
                    f"{kind.value}/{variant} eager execution cannot carry "
                    "graph padding")
            model_input_shape = normalized["model_input_shape"]
            if not isinstance(model_input_shape, Mapping):
                raise TypeError(
                    f"{kind.value}/{variant} model_input_shape must be a mapping")
            # ``elements`` is a normalized alias for VECTOR element_count and
            # ``tok`` is synthesized from ``rows`` only for normalization
            # variants.  model-input shape records the pre-normalization
            # launch ABI, so it retains a source ``tok`` for activation/Silu
            # but does not need the normalization-only alias.
            derived_shape_fields = {"dtype_bytes", "elements"}
            if (kind is KernelKind.VECTOR
                    and variant in ("rmsnorm", "add_rmsnorm")):
                derived_shape_fields.add("tok")
            expected_shape_fields = set(shape) - derived_shape_fields
            if set(model_input_shape) != expected_shape_fields:
                raise ValueError(
                    f"{kind.value}/{variant} model_input_shape fields must "
                    "match the effective shape")
            normalized_model_input_shape = {}
            for name, value in model_input_shape.items():
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(
                        f"{kind.value}/{variant} model_input_shape values "
                        "must be integers")
                if value <= 0:
                    raise ValueError(
                        f"{kind.value}/{variant} model_input_shape values "
                        "must be positive")
                if value < shape[name]:
                    raise ValueError(
                        f"{kind.value}/{variant} model_input_shape cannot "
                        "shrink effective dimensions")
                normalized_model_input_shape[name] = value
            normalized["model_input_shape"] = _immutable_mapping(
                normalized_model_input_shape)
            optional |= graph_token_domain_fields
    if kind is KernelKind.VECTOR and variant in ("dsv32_moe_route", "kimi_moe_route"):
        optional.add("source_projection_scope")
        if ("source_projection_scope" in normalized
                and normalized["source_projection_scope"] not in ("a800_tp1_target_representative_layers_v1", "a800_tp8_pp4_ep8_glm_source_v1", "a800_tp8_pp2_ep8_kimi_source_v1", "a800_tp1_kimi_source_v1", "vllm_0_30_glm_pcp_source_v1")):
            raise ValueError("target structural route requires its exact source projection scope")
        if ("source_projection_scope" in normalized
                and normalized["source_projection_scope"] == "a800_tp8_pp4_ep8_glm_source_v1"
                and variant != "dsv32_moe_route"):
            raise ValueError("GLM pipeline source scope cannot authorize another model's router")
        if ("source_projection_scope" in normalized
                and normalized["source_projection_scope"] in ("a800_tp8_pp2_ep8_kimi_source_v1", "a800_tp1_kimi_source_v1")
                and variant != "kimi_moe_route"):
            raise ValueError("Kimi pipeline source scope cannot authorize another model's router")
    if kind is KernelKind.GEMM:
        optional.add("source_projection_scope")
        optional.add("source_gemm_runtime")
        if "source_projection_scope" in normalized:
            if (normalized["source_projection_scope"] not in ("a800_tp1_target_representative_layers_v1", "a800_tp8_pp4_ep8_glm_source_v1", "vllm_0_30_glm_pcp_source_v1")
                    or variant != "gemm" or compute_dtype != "bf16"
                    or "output_dtype" not in normalized or normalized["output_dtype"] != "fp32"):
                raise ValueError("target structural GEMM scope requires plain BF16 A/B and FP32 C")
        # None is the schema-defined homogeneous-output mode, so a dtype
        # change of a normalized KernelSpec keeps its output tied to input.
        # Explicit mixed output must survive such copies without inference.
        optional.add("output_dtype")
        if "output_dtype" not in normalized:
            normalized["output_dtype"] = None
        # Schema-defined beta: ordinary value 0 (numeric or lexical).  The
        # boundary normalizes equivalent representations once so a non-zero
        # beta can never alias the beta=0 row downstream.  A bool is
        # rejected before any ordinary-alias test because ``False == 0``
        # in Python would otherwise let it slip through as the ordinary
        # value; True is likewise not a numeric beta.
        optional.add("beta")
        if "beta" not in normalized:
            normalized["beta"] = 0
        beta = normalized["beta"]
        if isinstance(beta, bool):
            raise ValueError(
                "GEMM beta must be a non-negative finite number, not bool")
        if beta is None or beta == "" or beta == "0" or beta == 0:
            normalized["beta"] = 0
        elif isinstance(beta, str):
            try:
                beta_value = float(beta)
            except ValueError:
                raise ValueError("GEMM beta must be a decimal string") from None
            if not math.isfinite(beta_value) or beta_value < 0:
                raise ValueError("GEMM beta must be a non-negative finite decimal")
            # Preserve integer-valued decimals as ints so downstream key
            # projections (e.g. the H200 v6 typed beta suffix) see "1"
            # rather than "1.0"; the typed key stores a canonical decimal.
            if beta_value.is_integer():
                beta_value = int(beta_value)
            normalized["beta"] = beta_value
        elif isinstance(beta, (int, float)):
            if not math.isfinite(beta) or beta < 0:
                raise ValueError("GEMM beta must be a non-negative finite number")
            if isinstance(beta, float) and beta.is_integer():
                beta = int(beta)
            normalized["beta"] = beta
        else:
            raise ValueError("GEMM beta must be a non-negative finite number or decimal string")
        output_dtype = normalized["output_dtype"]
        if "source_gemm_runtime" in normalized:
            if (normalized["source_gemm_runtime"] != "vllm_0_30_torch_2_13_cu130_cublas13_1_1_fp16_unquantized"
                    or variant != "gemm" or compute_dtype != "fp16" or shape["b"] != 1
                    or output_dtype not in (None, "fp16") or normalized["beta"] != 0
                    or "mathematical_program" in normalized
                    or any(name in normalized for name in ("model_input_shape", "graph_runtime_mode"))):
                raise ValueError("observed FP16 GEMM runtime requires its homogeneous eager0.30 ordinary source")
        if variant == "gemm":
            optional.add("mathematical_program")
            if "mathematical_program" in normalized:
                program = normalized["mathematical_program"]
                if program == "unquantized_fp32_flops_bytes_v1":
                    if compute_dtype != "fp32" or output_dtype not in (None, "fp32") or shape["b"] != 1:
                        raise ValueError("unquantized FP32 mathematical GEMM requires FP32 A/B/C")
                elif program in ("unquantized_fp16_flops_bytes_v1", "unquantized_bf16_flops_bytes_v1"):
                    required.update({"weight_dtype", "weight_group_size", "has_bias"})
                    source_dtype = {"unquantized_fp16_flops_bytes_v1": "fp16",
                        "unquantized_bf16_flops_bytes_v1": "bf16"}[program]
                    if (not {"weight_dtype", "weight_group_size", "has_bias"}.issubset(normalized)
                            or compute_dtype != source_dtype or output_dtype not in (None, source_dtype) or shape["b"] != 1
                            or normalized["weight_dtype"] != source_dtype
                            or type(normalized["weight_group_size"]) is not int or normalized["weight_group_size"] != 0
                            or normalized["has_bias"] is not False):
                        raise ValueError("unquantized FP16 mathematical GEMM requires explicit homogeneous A/B/C, no bias and no weight quantization")
                elif program == "block_fp8_quantized_input_fp32_scales_v1":
                    required.update({"weight_dtype", "weight_group_size", "has_bias"})
                    if (not required.issubset(normalized) or compute_dtype != "fp8" or output_dtype != "bf16"
                            or shape["b"] != 1 or shape["k"] % 128 or shape["n"] % 128
                            or normalized["weight_dtype"] != "fp8" or normalized["weight_group_size"] != 128
                            or normalized["has_bias"] is not False):
                        raise ValueError("prequantized block-FP8 MM requires FP8 A/B, FP32 group128 scales and BF16 C")
                elif program == "packed_fp4_source_v1":
                    required.update({"weight_dtype", "weight_group_size", "has_bias",
                        "output_dtype", "compute_format", "weight_matrix_count",
                        "matrix_row_counts", "input_quantized"})
                    if not required.issubset(normalized):
                        raise ValueError("packed-FP4 source GEMM requires its complete operand program")
                    matrix_count = normalized["weight_matrix_count"]
                    rows = normalized["matrix_row_counts"]
                    if (compute_dtype not in ("bf16", "fp8", "fp32") or output_dtype not in ("bf16", "fp32")
                            or normalized["weight_dtype"] != "mxfp4"
                            or type(normalized["weight_group_size"]) is not int
                            or normalized["weight_group_size"] != 32
                            or normalized["has_bias"] is not False
                            or normalized["compute_format"] not in ("fp8", "fp4")
                            or type(matrix_count) is not int or matrix_count < 1
                            or not isinstance(rows, (tuple, list)) or len(rows) != matrix_count
                            or any(type(value) is not int or value < 1 for value in rows)
                            or sum(rows) != shape["b"] * shape["m"]
                            or shape["k"] % 32
                            or type(normalized["input_quantized"]) is not bool):
                        raise ValueError("packed-FP4 source GEMM requires conserved matrix rows, group32 weights and explicit compute format")
                    normalized["matrix_row_counts"] = tuple(rows)
                elif program in ("mxfp4_bf16_flops_bytes_v1", "mxfp4_fp8_flops_bytes_v1"):
                    required.update({"weight_dtype", "weight_group_size", "has_bias"})
                    expected_dtype = "fp8" if program == "mxfp4_fp8_flops_bytes_v1" else "bf16"
                    if (not required.issubset(normalized) or compute_dtype != expected_dtype
                            or output_dtype not in (("bf16",) if expected_dtype == "fp8" else (None, "bf16")) or shape["b"] != 1 or shape["k"] % 32
                            or normalized["weight_dtype"] != "mxfp4"
                            or type(normalized["weight_group_size"]) is not int
                            or normalized["weight_group_size"] != 32 or normalized["has_bias"] is not False
                            or (expected_dtype == "fp8" and shape["k"] % 128)):
                        raise ValueError("MXFP4 mathematical GEMM requires declared BF16/FP8 A, BF16 C, group32 E2M1 weights/E8M0 scales and no bias")
                elif (program != "block_fp8_fp32_scales_v1"
                        or compute_dtype != "bf16" or output_dtype not in (None, "bf16")
                        or shape["b"] != 1 or shape["k"] % 128):
                    raise ValueError("block-FP8 mathematical GEMM requires BF16 output and whole128 input groups")
        if output_dtype is not None and (not isinstance(output_dtype, str) or output_dtype not in DTYPE_BYTES):
            raise ValueError("GEMM output_dtype must be a supported explicit dtype or None")
        quantized_mixed = ("mathematical_program" in normalized
            and normalized["mathematical_program"] in ("mxfp4_fp8_flops_bytes_v1", "block_fp8_quantized_input_fp32_scales_v1", "packed_fp4_source_v1"))
        if not quantized_mixed and output_dtype is not None and output_dtype != compute_dtype and (compute_dtype, output_dtype, variant) != ("bf16", "fp32", "gemm"):
            raise ValueError("mixed-output GEMM supports BF16 inputs and FP32 output only")
        if not quantized_mixed and output_dtype is not None and output_dtype != compute_dtype:
            optional.update({"weight_dtype", "weight_group_size"})
            if "weight_dtype" not in normalized:
                normalized["weight_dtype"] = "bf16"
            if "weight_group_size" not in normalized:
                normalized["weight_group_size"] = 0
            if (normalized["weight_dtype"] != "bf16"
                    or type(normalized["weight_group_size"]) is not int
                    or normalized["weight_group_size"] != 0):
                raise ValueError("mixed-output GEMM requires unquantized BF16 weights")
    if kind is KernelKind.VECTOR and variant == "silu":
        optional.add("swiglu_limit")
        if "swiglu_limit" not in normalized:
            normalized["swiglu_limit"] = None
        limit = normalized["swiglu_limit"]
        if limit is not None and (isinstance(limit, bool)
                or not isinstance(limit, (int, float))
                or not math.isfinite(limit) or limit <= 0):
            raise ValueError("silu swiglu_limit must be finite positive or None")
    if "packed_weight_storage" in normalized:
        if (kind is not KernelKind.VECTOR
                or variant not in ("gather", "vocab_parallel_embedding", "mhc_pre", "mhc_post_pre", "hc_head")
                or normalized["packed_weight_storage"] != "mxfp4_group32"
                or compute_dtype != "bf16" or "hidden" not in shape or shape["hidden"] % 32):
            raise ValueError("packed weight storage requires a supported BF16 vector source with complete group32 weight rows")
        optional.add("packed_weight_storage")
    missing = required - set(normalized)
    if missing:
        raise ValueError(
            f"{kind.value}/{variant} missing attributes: {sorted(missing)}")
    unknown = set(normalized) - required - optional
    if unknown:
        raise ValueError(
            f"unknown {kind.value}/{variant} attributes: {sorted(unknown)}")
    if "helper_launch" in normalized:
        # Helper/glue emission contract attribute: a workload-node may carry
        # exactly one emission class (docs/serving_system.md §11).  The value is a mapping with one non-empty "class" string; the
        # harness and lowering read it structurally, never by kernel name.
        helper = normalized["helper_launch"]
        if (not isinstance(helper, Mapping)
                or set(helper) != {"class"}
                or not isinstance(helper["class"], str)
                or not helper["class"].strip()):
            raise ValueError(
                f"{kind.value}/{variant} helper_launch must be a mapping "
                "with exactly one non-empty 'class' string")
    if (kind is KernelKind.VECTOR
            and variant in ("elementwise", "elementwise_pytorch_vec4")
            and "input_materialization" in normalized):
        if normalized["input_materialization"] != (
                "grouped_strided_to_contiguous"):
            raise ValueError(
                "VECTOR copy input_materialization must be "
                "grouped_strided_to_contiguous")
        dimensions = {
            name: normalized[name]
            for name in (
                "input_outer_groups", "input_rows_per_group",
                "input_group_stride_elements", "input_row_stride_elements",
            )
        }
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in dimensions.values()):
            raise TypeError(
                "VECTOR grouped input materialization dimensions must be "
                "integers")
        if any(value <= 0 for value in dimensions.values()):
            raise ValueError(
                "VECTOR grouped input materialization dimensions must be "
                "positive")
        copied_elements = (
            dimensions["input_outer_groups"]
            * dimensions["input_rows_per_group"]
            * dimensions["input_row_stride_elements"])
        if copied_elements != shape["element_count"]:
            raise ValueError(
                "VECTOR grouped input materialization dimensions disagree "
                "with element_count")
        if dimensions["input_group_stride_elements"] <= (
                dimensions["input_rows_per_group"]
                * dimensions["input_row_stride_elements"]):
            raise ValueError(
                "VECTOR grouped input materialization requires an "
                "inter-group gap")
    if kind is KernelKind.VECTOR and variant == "moe_swiglu":
        routed_rows = normalized["routed_rows"]
        if isinstance(routed_rows, bool) or not isinstance(routed_rows, int):
            raise TypeError("VECTOR/moe_swiglu routed_rows must be an integer")
        if routed_rows <= 0:
            raise ValueError("VECTOR/moe_swiglu routed_rows must be positive")
        if routed_rows > shape["tokens"] * shape["topk"]:
            raise ValueError(
                "VECTOR/moe_swiglu routed_rows cannot exceed tokens * topk")
    if (kind is KernelKind.VECTOR
            and variant == "moe_compute_expert_tokens"):
        raw_counts = normalized["tokens_per_expert"]
        if (not isinstance(raw_counts, (list, tuple))
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in raw_counts)):
            raise TypeError(
                "VECTOR/moe_compute_expert_tokens tokens_per_expert must "
                "contain integers")
        counts = tuple(raw_counts)
        if (len(counts) != shape["experts"]
                or any(value < 0 for value in counts)):
            raise ValueError(
                "VECTOR/moe_compute_expert_tokens tokens_per_expert must "
                "contain one non-negative count per expert")
        expected = shape["tokens"] * shape["topk"]
        if sum(counts) != expected:
            raise ValueError(
                "VECTOR/moe_compute_expert_tokens tokens_per_expert sum "
                f"{sum(counts)} != tokens*topk {expected}")
        normalized["tokens_per_expert"] = counts
    if kind is KernelKind.FLASH_ATTN:
        if not isinstance(normalized["causal"], bool):
            raise ValueError("FLASH_ATTN causal must be boolean")
        if variant in ("prefill", "decode", "combine", "split_combine"):
            if "return_softmax_lse" in normalized:
                if (variant != "combine" or type(normalized["return_softmax_lse"]) is not bool
                        or shape["head_dim"] != 128 or shape["num_kv_heads"] != shape["num_query_heads"]
                        or normalized["causal"] is not False or normalized["kv_layout"] != "contiguous"
                        or normalized["fa_ksteps"] is not None or normalized["fa_eff_mfrac"] != 1.0
                        or any(name in normalized for name in ("graph_runtime_mode", "model_input_shape",
                            "model_input_q_lens", "model_input_kv_lens", "graph_logical_tokens", "graph_model_input_tokens"))):
                    raise ValueError("final split-prefill LSE requires its exact unpadded128-wide combine interface")
            if (isinstance(normalized["fa_eff_mfrac"], bool)
                    or not isinstance(
                        normalized["fa_eff_mfrac"], (int, float))):
                raise TypeError("FLASH_ATTN fa_eff_mfrac must be numeric")
            fraction = float(normalized["fa_eff_mfrac"])
            if not math.isfinite(fraction) or not 0 < fraction <= 1:
                raise ValueError("FLASH_ATTN fa_eff_mfrac must be in (0, 1]")
            normalized["fa_eff_mfrac"] = fraction
            if normalized["fa_ksteps"] is not None:
                if (isinstance(normalized["fa_ksteps"], bool)
                        or not isinstance(normalized["fa_ksteps"], int)):
                    raise TypeError("FLASH_ATTN fa_ksteps must be an integer")
                steps = normalized["fa_ksteps"]
                if steps <= 0:
                    raise ValueError("FLASH_ATTN fa_ksteps must be positive")
                normalized["fa_ksteps"] = steps
        elif variant in ("mla_prefill", "mla_decode", "mla_absorbed_prefill", "kimi_k3_nope_context_attention"):
            if "return_softmax_lse" in normalized:
                if (variant not in ("mla_prefill", "mla_absorbed_prefill", "kimi_k3_nope_context_attention")
                        or type(normalized["return_softmax_lse"]) is not bool
                        or (variant == "mla_absorbed_prefill" and "mathematical_program" not in normalized)):
                    raise ValueError("MLA return_softmax_lse requires a boolean prefill output contract")
                if "attention_backend" in normalized:
                    raise NotImplementedError("explicit MLA LSE output currently requires the FA3 analytical source path")
            if variant == "kimi_k3_nope_context_attention":
                if "context_start" in normalized:
                    start = normalized["context_start"]
                    if type(start) is not int or not 0 <= start < 2**31:
                        raise ValueError("K3 context attention requires a nonnegative INT32 chunk start")
                if "output_target" in normalized:
                    if normalized["output_target"] not in ("chunk_partial", "context_accumulator"):
                        raise ValueError("K3 context attention requires an explicit valid output target")
                    if (normalized["output_target"] == "context_accumulator"
                            and ("context_start" not in normalized or normalized["context_start"] != 0)):
                        raise ValueError("K3 continuation attention cannot overwrite its context accumulator")
                if (compute_dtype != "bf16" or normalized["causal"] is not False
                        or normalized["return_softmax_lse"] is not True
                        or normalized["full_cuda_graphs"] is not False
                        or normalized["max_cudagraph_capture_size"] != 0
                        or shape["batch"] != 1 or shape["num_query_heads"] != 96
                        or (shape["qk_head_dim"], shape["v_head_dim"], shape["kv_lora_rank"]) != (192, 128, 512)):
                    raise ValueError("K3 context attention requires exact BF16 TP1 noncausal output+FP32 LSE source geometry")
            if ("attention_backend" in normalized
                    and normalized["attention_backend"] != "flashinfer"):
                raise ValueError(
                    "MLA attention_backend must be 'flashinfer' when given")
            if normalized["kv_layout"] != "mla":
                raise ValueError("MLA attention requires kv_layout='mla'")
            if (normalized["kv_dist"] != "uniform"
                    and not (sparse_fa3 and normalized["kv_dist"] == "explicit")):
                raise ValueError("MLA attention requires uniform lengths or an explicit sparse FA3 source")
            if not isinstance(normalized["full_cuda_graphs"], bool):
                raise TypeError("MLA full_cuda_graphs must be boolean")
            capture = normalized["max_cudagraph_capture_size"]
            if (isinstance(capture, bool) or not isinstance(capture, int)
                    or capture < 0):
                raise ValueError(
                    "MLA max_cudagraph_capture_size must be a non-negative "
                    "integer (0 when CUDA graphs are disabled)")
            if normalized["full_cuda_graphs"] and capture == 0:
                raise ValueError(
                    "MLA full CUDA graphs require a positive capture size")
            graph_fields = {
                "graph_logical_tokens", "graph_model_input_tokens",
                "graph_runtime_mode"}
            present = graph_fields & set(normalized)
            if present:
                if present != graph_fields:
                    raise ValueError(
                        "MLA graph token domain must be complete: "
                        f"present={sorted(present)}")
                if variant != "mla_decode":
                    raise ValueError(
                        "only MLA decode carries a graph token domain")
                logical = normalized["graph_logical_tokens"]
                physical = normalized["graph_model_input_tokens"]
                if (any(isinstance(value, bool) or not isinstance(value, int)
                        or value <= 0 for value in (logical, physical))
                        or physical < logical):
                    raise ValueError(
                        "MLA graph token domain must be positive and "
                        "non-shrinking")
                if normalized["graph_runtime_mode"] != "FULL":
                    raise ValueError(
                        "MLA decode padding is a FULL CUDA-graph domain")
                if logical != sum(normalized["q_lens"]):
                    raise ValueError(
                        "MLA graph logical tokens disagree with q_lens")
        else:
            raise AssertionError(
                f"unhandled FLASH_ATTN variant: {variant!r}")
        _normalize_request_lengths("FLASH_ATTN", normalized, shape)
        if (sparse_fa3 and normalized["kv_dist"] == "uniform"
                and len(set(normalized["kv_lens"])) != 1):
            raise ValueError("sparse FA3 uniform label requires equal live lengths")
        if variant == "mla_decode" and "mathematical_program" in normalized:
            if (compute_dtype != "bf16" or normalized["causal"] is not True
                    or shape["num_query_heads"] <= 0
                    or (shape["qk_head_dim"], shape["v_head_dim"], shape["kv_lora_rank"]) != (576, 512, 512)
                    or len(set(normalized["q_lens"])) != 1
                    or any(not 1 <= q <= 128 or (k < q
                        and normalized["mathematical_program"] != "bf16_mla_causal_empty_rows_softmax_v1")
                        for q, k in zip(normalized["q_lens"], normalized["kv_lens"], strict=True))
                    or normalized["full_cuda_graphs"] is not False or normalized["max_cudagraph_capture_size"] != 0
                    or any(name in normalized for name in ("graph_logical_tokens", "graph_model_input_tokens",
                        "graph_runtime_mode", "attention_backend"))):
                raise ValueError("MLA mathematical program requires BF16 unpadded uniform queries<=128 without a provider")
        if variant == "mla_absorbed_prefill" and "mathematical_program" in normalized:
            if (compute_dtype != "bf16" or normalized["causal"] is not True
                    or (shape["qk_head_dim"], shape["v_head_dim"], shape["kv_lora_rank"]) != (576, 512, 512)
                    or "return_softmax_lse" not in normalized or normalized["return_softmax_lse"] is not False
                    or not 0 < sum(k - q for q, k in zip(normalized["q_lens"], normalized["kv_lens"], strict=True)) < 8192
                    or normalized["full_cuda_graphs"] is not False or normalized["max_cudagraph_capture_size"] != 0
                    or any(name in normalized for name in ("graph_logical_tokens", "graph_model_input_tokens",
                        "graph_runtime_mode", "attention_backend"))):
                raise ValueError("absorbed MLA prefill math requires BF16 causal no-LSE, declared prefix sum1..8191 and no provider/graph")
        if variant == "mla_prefill" and "mathematical_program" in normalized:
            context_program = normalized["mathematical_program"] in (
                "bf16_mla_context_softmax_v1", "bf16_mla_context_empty_rows_softmax_v1")
            if (normalized["mathematical_program"] == "bf16_mla_context_empty_rows_softmax_v1"
                    and ((shape["qk_head_dim"], shape["v_head_dim"], shape["kv_lora_rank"]) != (192, 128, 512)
                        or not any(k == 0 for k in normalized["kv_lens"]))):
                raise ValueError("empty-row MLA context requires V3 geometry and at least one declared zero-KV request")
            one_shot = normalized["mathematical_program"] == "bf16_mla_one_shot_prefill_softmax_v1"
            if one_shot and (type(normalized["prefix_cache_threshold"]) is not int or normalized["prefix_cache_threshold"] != 8192
                    or type(normalized["max_kv_chunk_capacity"]) is not int or normalized["max_kv_chunk_capacity"] <= 0
                    or not 8192 <= sum(k - q for q, k in zip(normalized["q_lens"], normalized["kv_lens"]))
                    or sum(normalized["kv_lens"]) > normalized["max_kv_chunk_capacity"]
                    or normalized["return_softmax_lse"] is not False
                    or (shape["qk_head_dim"], shape["v_head_dim"], shape["kv_lora_rank"]) != (192, 128, 512)):
                raise ValueError("MLA one-shot math requires the declared8192 prefix threshold, chunk capacity and no-LSE V3 geometry")
            if (compute_dtype != "bf16" or normalized["causal"] is not (not context_program)
                    or (shape["qk_head_dim"], shape["v_head_dim"], shape["kv_lora_rank"]) not in ((192, 128, 512), (256, 256, 512))
                    or (not context_program and not one_shot and normalized["q_lens"] != normalized["kv_lens"])
                    or normalized["full_cuda_graphs"] is not False or normalized["max_cudagraph_capture_size"] != 0
                    or "return_softmax_lse" not in normalized
                    or type(normalized["return_softmax_lse"]) is not bool
                    or (context_program and not normalized["return_softmax_lse"])
                    or any(name in normalized for name in ("graph_logical_tokens", "graph_model_input_tokens",
                        "graph_runtime_mode", "attention_backend"))):
                raise ValueError("MLA mathematical prefill requires its BF16 fresh/context lengths and LSE contract without a provider")
    elif kind in (KernelKind.FUSED_MOE, KernelKind.UNFUSED_MOE):
        raw_counts = normalized["tokens_per_expert"]
        if (not isinstance(raw_counts, (list, tuple))
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in raw_counts)):
            raise TypeError("tokens_per_expert must contain integers")
        counts = tuple(raw_counts)
        if len(counts) != shape["experts"] or any(value < 0 for value in counts):
            raise ValueError(
                "tokens_per_expert must contain one non-negative count per "
                "expert")
        expected = shape["tokens"] * shape["topk"]
        if sum(counts) != expected:
            raise ValueError(
                f"tokens_per_expert sum {sum(counts)} != tokens*topk "
                f"{expected}")
        normalized["tokens_per_expert"] = counts
        if kind is KernelKind.FUSED_MOE:
            ep_size = normalized["expert_parallel_size"]
            expert_rank = normalized["expert_rank"]
            if (isinstance(ep_size, bool)
                    or not isinstance(ep_size, int)):
                raise TypeError("expert_parallel_size must be an integer")
            if ep_size <= 0:
                raise ValueError(
                    "expert_parallel_size must be positive")
            if (isinstance(expert_rank, bool)
                    or not isinstance(expert_rank, int)):
                raise TypeError("expert_rank must be an integer")
            if not 0 <= expert_rank < ep_size:
                raise ValueError(
                    "expert_rank must be within expert_parallel_size")

            raw_owned = normalized["owned_experts"]
            if (not isinstance(raw_owned, (list, tuple))
                    or any(isinstance(value, bool)
                           or not isinstance(value, int)
                           for value in raw_owned)):
                raise TypeError("owned_experts must contain integers")
            owned = tuple(raw_owned)
            expected_owned = linear_expert_partition(shape["experts"], ep_size, expert_rank)
            if owned != expected_owned:
                raise ValueError(
                    "owned_experts disagrees with the canonical contiguous "
                    "expert-parallel placement")
            normalized["owned_experts"] = owned

            raw_local_counts = normalized["local_tokens_per_expert"]
            if (not isinstance(raw_local_counts, (list, tuple))
                    or any(isinstance(value, bool)
                           or not isinstance(value, int)
                           for value in raw_local_counts)):
                raise TypeError(
                    "local_tokens_per_expert must contain integers")
            local_counts = tuple(raw_local_counts)
            if (len(local_counts) != shape["experts"]
                    or any(value < 0 for value in local_counts)):
                raise ValueError(
                    "local_tokens_per_expert must contain one non-negative "
                    "count per expert")
            expected_local_counts = tuple(
                count if expert_id in owned else 0
                for expert_id, count in enumerate(counts)
            )
            if local_counts != expected_local_counts:
                raise ValueError(
                    "local_tokens_per_expert disagrees with global routing "
                    "and owned_experts")
            normalized["local_tokens_per_expert"] = local_counts
            normalized["expert_parallel_size"] = ep_size
            normalized["expert_rank"] = expert_rank
        elif kind is KernelKind.UNFUSED_MOE:
            local_counts = counts
            owned = tuple(range(shape["experts"]))
        else:
            raise AssertionError(f"unexpected MoE kernel kind: {kind!r}")
        windows = normalized["expert_weight_tile_rows"]
        if windows is not None:
            if (not isinstance(windows, (list, tuple))
                    or any(isinstance(value, bool)
                           or not isinstance(value, (int, float))
                           for value in windows)):
                raise TypeError(
                    "expert_weight_tile_rows must contain numeric values")
            windows = tuple(float(value) for value in windows)
            if len(windows) != shape["experts"]:
                raise ValueError(
                    "expert_weight_tile_rows must contain one value per expert")
            invalid_windows = [
                (count, window)
                for count, window in zip(local_counts, windows)
                if (not math.isfinite(window) or window < 0
                    or (count > 0 and window <= 0))
            ]
            if invalid_windows:
                raise ValueError(
                    "expert_weight_tile_rows must be finite and non-negative, "
                    "and positive for every active expert")
            if (kind is KernelKind.FUSED_MOE
                    and any(windows[expert_id] != 0.0
                            for expert_id in range(shape["experts"])
                            if expert_id not in owned)):
                raise ValueError(
                    "expert_weight_tile_rows must be zero for non-owned "
                    "experts")
            normalized["expert_weight_tile_rows"] = windows
        if kind is KernelKind.FUSED_MOE and "routed_weight_dtype" in normalized:
            routed_dtype = normalized["routed_weight_dtype"]
            if routed_dtype not in ("bf16", "fp8", "int4", "mxfp4"):
                raise ValueError(
                    "FUSED_MOE routed_weight_dtype must be bf16, fp8, int4 "
                    "or mxfp4")
            group_size = normalized["weight_group_size"]
            if isinstance(group_size, bool) or not isinstance(group_size, int):
                raise TypeError("FUSED_MOE weight_group_size must be an integer")
            if ((routed_dtype in ("bf16", "fp8") and group_size != 0)
                    or (routed_dtype == "int4" and group_size <= 0)
                    or (routed_dtype == "mxfp4" and group_size != 32)):
                raise ValueError(
                    "FUSED_MOE weight_group_size disagrees with routed weight "
                    "dtype")
            if "fused_shared_experts" in normalized:
                fused = normalized["fused_shared_experts"]
                if isinstance(fused, bool) or fused != 1:
                    raise ValueError(
                        "FUSED_MOE fused_shared_experts must be 1 (one "
                        "shared expert fused into the routed kernel)")
                if routed_dtype != "fp8":
                    raise ValueError(
                        "FUSED_MOE fused shared experts are source-backed "
                        "only for the FP8 block kernel")
            scaling = normalized["routed_scaling_factor"]
            if (isinstance(scaling, bool)
                    or not isinstance(scaling, (int, float))
                    or not math.isfinite(scaling)
                    or scaling <= 0):
                raise ValueError(
                    "FUSED_MOE routed_scaling_factor must be finite and "
                    "positive")
            normalized["routed_scaling_factor"] = float(scaling)
    elif kind is KernelKind.GEMM and variant == "kimi_k3_nope_bmm":
        position = normalized["source_position"]
        if position not in ("W_UK_T", "W_UV"):
            raise ValueError("K3 NoPE BMM requires an explicit absorption or value projection source")
        expected = (96, 512, 128) if position == "W_UK_T" else (96, 128, 512)
        if (shape["b"], shape["n"], shape["k"]) != expected or shape["m"] <= 0:
            raise ValueError("K3 NoPE BMM requires exact BF16 TP1 checkpoint geometry")
    elif kind is KernelKind.GEMM and variant == "batched_gemm":
        layout = normalized["batch_layout"]
        if layout not in ("strided", "einsum_bhr_hdr_bhd"):
            raise ValueError("GEMM/batched_gemm has an unsupported batch layout")
        if (layout == "einsum_bhr_hdr_bhd") != (normalized["weight_dtype"] == "fp8"):
            raise ValueError("FP8 batched GEMM requires the source einsum_bhr_hdr_bhd layout; BF16 requires strided")
        expected_strides = {
            "a_batch_stride_elements": shape["m"] * shape["k"],
            "b_batch_stride_elements": shape["k"] * shape["n"],
            # Source C is z[T,G,R], not a dense [G,T,R] allocation.
            # The group stride is R; the token stride is G*R.
            "c_batch_stride_elements": shape["n"] if layout == "einsum_bhr_hdr_bhd" else shape["m"] * shape["n"],
        }
        for name, expected in expected_strides.items():
            value = normalized[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"GEMM/batched_gemm {name} must be an integer")
            if value != expected:
                raise ValueError(
                    f"GEMM/batched_gemm {name} must equal {expected}, got "
                    f"{value}")
        if normalized["weight_dtype"] == "bf16":
            if normalized["weight_group_size"] != 0:
                raise ValueError(
                    "GEMM/batched_gemm BF16 weights are unquantized "
                    "(weight_group_size 0)")
        elif normalized["weight_dtype"] == "fp8":
            # DeepSeek-V4 wo_a FP8 einsum (deepseek_v4/nvidia/ops/o_proj.py
            # deep_gemm_fp8_o_proj): 128x128 block-scaled e4m3 weights with
            # dynamic 1x128 activation groups (SM90 recipe (1, 128, 128)).
            if normalized["weight_group_size"] != 128:
                raise ValueError(
                    "GEMM/batched_gemm FP8 weights require 128x128 block "
                    "scales (weight_group_size 128)")
        else:
            raise ValueError(
                "GEMM/batched_gemm requires BF16 or block-scaled FP8 "
                f"weights, got {normalized['weight_dtype']!r}")
    elif kind is KernelKind.GEMM and variant == "moe_router":
        # DeepSeek-V3 routes with sigmoid scores; DeepSeek-V4 uses the
        # sqrtsoftplus scoring of its GateLinear (deepseek_v4 MoE gate).
        if normalized["score_function"] not in ("sigmoid", "sqrtsoftplus"):
            raise ValueError(
                "GEMM/moe_router requires sigmoid or sqrtsoftplus scoring")
        if normalized["topk_method"] != "noaux_tc":
            raise ValueError("GEMM/moe_router requires noaux_tc top-k")
        for name in ("expert_group_count", "topk_group_count"):
            value = normalized[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"GEMM/moe_router {name} must be positive")
        if normalized["topk_group_count"] > normalized["expert_group_count"]:
            raise ValueError(
                "GEMM/moe_router top-k groups cannot exceed expert groups")
        if not isinstance(normalized["normalize_topk_probability"], bool):
            raise TypeError(
                "GEMM/moe_router normalize_topk_probability must be boolean")
        if (normalized["weight_dtype"] != "bf16"
                or normalized["weight_group_size"] != 0):
            raise ValueError("GEMM/moe_router requires unquantized BF16 weights")
    elif kind is KernelKind.SPARSE_ATTN_INDEXER:
        if variant == "dsa_index_reuse":
            if normalized["reuse_semantics"] != "alias":
                raise ValueError(
                    "SPARSE_ATTN_INDEXER reuse requires alias semantics")
        elif variant in (
                "dsa_indexer_prefill", "dsa_indexer_decode",
                "dsa_indexer_mixed", "dsv4_indexer_prefill",
                "dsv4_indexer_decode"):
            if normalized["causal"] is not True:
                raise ValueError("DSA indexer requires causal=True")
            if normalized["score_accumulator_dtype"] != "fp32":
                raise ValueError("DSA indexer requires FP32 score accumulation")
            if normalized["score_function"] != "weighted_dot_product_topk":
                raise ValueError(
                    "DSA indexer requires weighted_dot_product_topk scoring")
            if normalized["cache_layout"] != "paged_mla_indexer":
                raise ValueError("DSA indexer cache layout is invalid")
            if normalized["cache_quantization"] != (
                    "fp8_e4m3_with_ue8m0_scale"):
                raise ValueError("DSA indexer cache quantization is invalid")
            phases = tuple(normalized["request_phases"])
            if (len(phases) != shape["batch"]
                    or any(phase not in ("prefill", "decode")
                           for phase in phases)):
                raise ValueError(
                    "DSA indexer request_phases must contain one valid phase "
                    f"per request: variant={variant!r} batch={shape['batch']} "
                    f"phases={phases!r}")
            expected_phases = {
                "dsa_indexer_prefill": {"prefill"},
                "dsa_indexer_decode": {"decode"},
                "dsa_indexer_mixed": {"prefill", "decode"},
                "dsv4_indexer_prefill": {"prefill"},
                "dsv4_indexer_decode": {"decode"},
            }[variant]
            if set(phases) != expected_phases:
                raise ValueError(
                    f"{variant} request phases disagree with its variant")
            normalized["request_phases"] = phases
            _normalize_request_lengths(
                "SPARSE_ATTN_INDEXER", normalized, shape)
            # The phase split carries no q_len constraint: the decode group
            # absorbs plain single-token decodes and MTP/EAGLE verify windows
            # (q_len > 1 through the absorbed decode path), while the prefill
            # group covers chunked-prefill rows and single-token prefill tail
            # chunks.  Per-request q_lens <= kv_lens is already enforced by
            # _normalize_request_lengths.
        else:
            raise AssertionError(
                f"unhandled SPARSE_ATTN_INDEXER variant: {variant!r}")
        for name in ("topk_capacity",):
            value = normalized[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"SPARSE_ATTN_INDEXER {name} must be positive")
        if normalized["topk_index_dtype"] != "int32":
            raise ValueError("DSA top-k index dtype must be int32")
        if variant != "dsa_index_reuse" and normalized["topk_sentinel"] != -1:
            raise ValueError("DSA indexer top-k sentinel must be -1")
        if variant in ("dsa_indexer_prefill", "dsa_indexer_decode", "dsa_indexer_mixed"):
            program = normalized["mathematical_program"]
            cp, skip_clear = normalized["context_parallel_size"], normalized["skip_topk_buffer_clear"]
            spec_decode = program == "dsa_weighted_relu_specdecode_v1"
            if spec_decode and (type(normalized["num_speculative_tokens"]) is not int or normalized["num_speculative_tokens"] < 1
                    or type(normalized["decode_threshold"]) is not int
                    or normalized["decode_threshold"] != normalized["num_speculative_tokens"] + 1
                    or not any(phase == "decode" and q > 1 for phase, q in zip(normalized["request_phases"], normalized["q_lens"]))
                    or len({q for phase, q in zip(normalized["request_phases"], normalized["q_lens"]) if phase == "decode"}) != 1
                    or any((phase == "decode") != (q <= normalized["decode_threshold"])
                        for phase, q in zip(normalized["request_phases"], normalized["q_lens"]))):
                raise ValueError("DSA specdecode indexer requires its explicit positive depth+1 phase threshold")
            if program is None:
                if cp is not None or skip_clear is not None:
                    raise ValueError("DSA mathematical source fields require an explicit program")
            elif (type(cp) is not int or cp != 1 or type(skip_clear) is not bool or shape["head_dim"] != 128
                    or any(phase == "decode" and q != 1 and not spec_decode
                        for phase, q in zip(normalized["request_phases"], normalized["q_lens"], strict=True))
                    or not (spec_decode or (program == "dsa_weighted_relu_topk_v1" and variant == "dsa_indexer_decode"
                            and all(q == 1 for q in normalized["q_lens"]))
                        or (program == "dsa_weighted_relu_force_mqa_v1"
                            and variant in ("dsa_indexer_prefill", "dsa_indexer_mixed"))
                        or (program == "dsa_weighted_relu_dense_mha_v1" and variant in ("dsa_indexer_prefill", "dsa_indexer_mixed")
                            and "prefill" in normalized["request_phases"]
                            and all(q == kv and kv <= normalized["topk_capacity"]
                                for q, kv, phase in zip(normalized["q_lens"], normalized["kv_lens"],
                                    normalized["request_phases"], strict=True) if phase == "prefill")))):
                raise ValueError("DSA mathematical indexer requires its declared CP1 FP8 program and unpadded single-token decode rows")
        if variant in ("dsv4_indexer_prefill", "dsv4_indexer_decode"):
            cp = normalized["context_parallel_size"]
            if cp is not None and (type(cp) is not int or cp != 1):
                raise ValueError("V4 indexer source context_parallel_size must be explicitly CP1 or unknown")
            if cp == 1 and shape["head_dim"] != 128:
                raise ValueError("V4 CP1 indexer source requires 128-wide FP8 keys and one FP32 scale")
            # The V4 indexer scores one FP8 key per compressed state; with
            # the short-context fast path every candidate is selected, so
            # topk saturates at the compressed candidate count
            # (deepseek_v4/attention.py DeepseekV4Indexer).
            candidate_count = shape["kv_len"] // shape["compress_ratio"]
            if shape["topk"] != min(
                    normalized["topk_capacity"], candidate_count):
                raise ValueError(
                    "DSV4 indexer topk must equal min(topk_capacity, "
                    "compressed candidate count)")
        elif (variant != "dsa_index_reuse"
                and shape["topk"] != min(
                    normalized["topk_capacity"], shape["kv_len"])):
            raise ValueError(
                "DSA indexer topk must equal min(topk_capacity, kv_len)")
    elif kind is KernelKind.SPARSE_ATTN:
        dspark = ("mathematical_program" in normalized and normalized["mathematical_program"]
            == "vllm_0_30_dspark_swa_noncausal_softmax_v1")
        if dspark:
            if (normalized["causal"] is not False
                    or type(normalized["num_speculative_tokens"]) is not int
                    or normalized["num_speculative_tokens"] != 5
                    or any(type(q) is not int or q != 5 for q in normalized["q_lens"])):
                raise ValueError("DSpark sparse math requires its explicit noncausal anchor-plus-four-noise block")
        elif normalized["causal"] is not True:
            raise ValueError("SPARSE_ATTN requires causal=True")
        if normalized["indexer_type"] not in ("full", "shared"):
            raise ValueError("SPARSE_ATTN indexer_type must be full or shared")
        if normalized["kv_layout"] != "mla":
            raise ValueError("SPARSE_ATTN requires kv_layout='mla'")
        if normalized["kv_dist"] != "uniform":
            raise ValueError("SPARSE_ATTN requires kv_dist='uniform'")
        if normalized["topk_index_dtype"] != "int32":
            raise ValueError("SPARSE_ATTN top-k indices must be int32")
        _normalize_request_lengths("SPARSE_ATTN", normalized, shape)
        if variant == "dsa_sparse_decode" and "mathematical_program" in normalized:
            program = normalized["mathematical_program"]
            if program == "bf16_latent_specdecode_v1" and (
                    type(normalized["num_speculative_tokens"]) is not int or normalized["num_speculative_tokens"] < 1
                    or type(normalized["decode_threshold"]) is not int
                    or normalized["decode_threshold"] != normalized["num_speculative_tokens"] + 1
                    or len(set(normalized["q_lens"])) != 1
                    or not 1 < shape["q_len"] <= normalized["decode_threshold"]):
                raise ValueError("DSA specdecode attention requires uniform rows within its positive depth+1 threshold")
            if program == "bf16_latent_reused_topk_v1":
                old = normalized["reused_selection_kv_lens"]
                if (normalized["indexer_type"] != "shared" or any(q != 1 for q in normalized["q_lens"])
                        or not isinstance(old, (tuple, list)) or len(old) != shape["batch"]
                        or any(type(k) is not int or not 0 < k <= current for k, current in zip(old, normalized["kv_lens"]))):
                    raise ValueError("DSA reused selection requires shared q1 and explicit prior causal KV lengths")
                normalized["reused_selection_kv_lens"] = tuple(old)
        selected = tuple(normalized["selected_kv_lens"])
        dsv4_variant = variant in ("dsv4_sparse_prefill", "dsv4_sparse_decode")
        if dsv4_variant:
            ratio = shape["compress_ratio"]
            selection = normalized["compressed_selection"]
            if selection not in ("candidate_upper_bound", "all_completed_states", "declared_causal_topk"):
                raise ValueError("invalid DSV4 compressed_selection")
            if selection == "declared_causal_topk" and ratio != 4:
                raise ValueError("declared causal top-k cardinality requires C4")
            if selection == "all_completed_states" and ratio not in (4, 128):
                raise ValueError("DSV4 completed-state selection requires C4 or C128")
            if ratio == 1:
                dsv4_expected = (0,) * shape["batch"]
            elif ratio == 4:
                capacity = normalized["topk_capacity"]
                if (isinstance(capacity, bool)
                        or not isinstance(capacity, int) or capacity <= 0):
                    raise ValueError(
                        "SPARSE_ATTN topk_capacity must be positive and "
                        "cover topk")
                if selection == "all_completed_states" and max(normalized["kv_lens"]) // ratio > capacity:
                    raise ValueError("DSV4 all-completed selection exceeds short-context threshold")
                # Only completed C4 states exist: combine_topk_swa_indices
                # uses min((pos+1)//4, TOP_K), including the scored path.
                # Unknown IDs do not turn a partial compression window into
                # another valid candidate.
                dsv4_expected = tuple(
                    min(capacity, kv_len // ratio)
                    for kv_len in normalized["kv_lens"])
            elif ratio == 128:
                capacity = normalized["topk_capacity"]
                # C128 attends the FULL compressed range; there is no
                # indexer and no index_topk cap.  The selection count is
                # floor((position+1)/128) (deepseek_v4/sparse_mla.py:404
                # num_compressed; prefill gathers seq_lens//128,
                # flashmla.py _forward_prefill dequantize_and_gather), held
                # in a 128-aligned buffer (sparse_mla.py:39
                # _C128A_TOPK_ALIGNMENT, :155-170 c128a_max_compressed).
                if (isinstance(capacity, bool)
                        or not isinstance(capacity, int) or capacity <= 0
                        or capacity % 128):
                    raise ValueError(
                        "DSV4 C128 topk_capacity must be a positive "
                        "multiple of the 128-wide C128A topk buffer "
                        "alignment")
                dsv4_expected = tuple(
                    min(capacity, kv_len // ratio)
                    for kv_len in normalized["kv_lens"])
            else:
                raise ValueError(
                    f"DSV4 compress_ratio must be one of 1, 4, 128, got "
                    f"{ratio}")
        else:
            dsv4_expected = None
        if (len(selected) != shape["batch"]
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value <= 0 for value in selected)):
            # DSV4 layers with no compressed state to attend (SWA-only
            # layers, or C128 sequences shorter than one compression window)
            # pass an empty selection upstream (flashmla.py forward_mqa;
            # sparse_mla.py:404 num_compressed == 0).
            if not (dsv4_variant
                    and len(selected) == shape["batch"]
                    and all(isinstance(value, int)
                            and not isinstance(value, bool) and value >= 0
                            for value in selected)):
                raise ValueError(
                    "SPARSE_ATTN selected_kv_lens must contain one positive "
                    "integer per request")
        if max(selected) != shape["topk"]:
            raise ValueError(
                "SPARSE_ATTN selected_kv_lens maximum disagrees with topk")
        normalized["selected_kv_lens"] = selected
        capacity = normalized["topk_capacity"]
        if (isinstance(capacity, bool) or not isinstance(capacity, int)
                or capacity <= 0 or shape["topk"] > capacity):
            raise ValueError(
                "SPARSE_ATTN topk_capacity must be positive and cover topk")
        expected_selected = tuple(
            min(capacity, kv_len) for kv_len in normalized["kv_lens"])
        if "reused_selection_kv_lens" in normalized:
            expected_selected = tuple(min(capacity, length) for length in normalized["reused_selection_kv_lens"])
        if dsv4_expected is not None:
            expected_selected = dsv4_expected
        if variant in ("dsv4_sparse_prefill", "dsv4_sparse_decode"):
            if shape["qk_head_dim"] != 512 or shape["v_head_dim"] != 512:
                raise ValueError(
                    "DSV4 sparse QK/V head dimensions must be 512 including RoPE")
            if shape["compressed_state_bytes"] != 584 or shape["swa_token_bytes"] != 584:
                raise ValueError(
                    "DSV4 fp8_ds_mla compressed/SWA record storage must be 584 bytes; "
                    "K/V roles do not duplicate the stored row")
            # DeepSeek-V4 attends the selected compressed states (C4:
            # indexer top-k; C128: the full compressed range) alongside the
            # SWA window from the separate full-resolution cache
            # (deepseek_v4/nvidia/flashmla.py).
            swa = tuple(normalized["swa_kv_lens"])
            expected_swa = (tuple(q + min(shape["window_size"], kv - q)
                for q, kv in zip(normalized["q_lens"], normalized["kv_lens"], strict=True)) if dspark else tuple(
                min(shape["window_size"], kv_len)
                for kv_len in normalized["kv_lens"]))
            if (len(swa) != shape["batch"]
                    or any(isinstance(value, bool)
                           or not isinstance(value, int)
                           or value <= 0 for value in swa)
                    or swa != expected_swa):
                raise ValueError(
                    "DSpark SWA lengths must include the trailing context window and complete noncausal block"
                    if dspark else "DSV4 SPARSE_ATTN swa_kv_lens must equal min(window_size, kv_len) per request")
            normalized["swa_kv_lens"] = swa
        if variant == "dsv4_sparse_decode" and "mathematical_program" in normalized:
            if normalized["split_counts"] is not None:
                raise ValueError("mathematical sparse decode cannot claim native split counts")
        elif variant == "dsv4_sparse_decode":
            splits = tuple(normalized["split_counts"])
            if (len(splits) != shape["batch"]
                    or any(isinstance(value, bool)
                           or not isinstance(value, int) or value <= 0
                           for value in splits)):
                raise ValueError(
                    "DSV4 sparse decode split_counts must contain one "
                    "positive split count per request")
            normalized["split_counts"] = splits
        if selected != expected_selected:
            raise ValueError(
                "SPARSE_ATTN selected_kv_lens disagree with KV lengths and "
                "top-k capacity")
    elif kind is KernelKind.KV_CACHE:
        if variant == "paged_kv_cache":
            layout = normalized["cache_layout"]
            if not isinstance(layout, str) or not layout.strip():
                raise TypeError(
                    "KV_CACHE/paged_kv_cache cache_layout must be a "
                    "non-empty string")
            if layout.strip().lower() != "paged":
                raise NotImplementedError(
                    "KV_CACHE/paged_kv_cache supports only paged "
                    "layout")
            normalized["cache_layout"] = "paged"
            head_layout = normalized["cache_head_layout"]
            if not isinstance(head_layout, str) or not head_layout.strip():
                raise TypeError(
                    "KV_CACHE/paged_kv_cache cache_head_layout must "
                    "be a non-empty string")
            head_layout = head_layout.strip().lower()
            if head_layout not in ("hnd", "nhd"):
                raise NotImplementedError(
                    "KV_CACHE/paged_kv_cache requires hnd or nhd "
                    "head layout")
            normalized["cache_head_layout"] = head_layout
            kv_cache_dtype = normalized["kv_cache_dtype"]
            if (not isinstance(kv_cache_dtype, str)
                    or not kv_cache_dtype.strip()):
                raise TypeError(
                    "KV_CACHE/paged_kv_cache kv_cache_dtype must be "
                    "a non-empty string")
            if kv_cache_dtype.strip().lower() != "auto":
                raise NotImplementedError(
                    "KV_CACHE/paged_kv_cache currently requires "
                    "kv_cache_dtype='auto'")
            normalized["kv_cache_dtype"] = "auto"
            for name in (
                    "cache_storage_matches_input", "slot_mapping_all_valid"):
                if not isinstance(normalized[name], bool):
                    raise TypeError(
                        f"KV_CACHE/paged_kv_cache {name} must be "
                        "boolean")
                if not normalized[name]:
                    raise NotImplementedError(
                        f"KV_CACHE/paged_kv_cache requires "
                        f"{name}=True")
            alignment = normalized["vector_copy_base_alignment_bytes"]
            if (isinstance(alignment, bool)
                    or not isinstance(alignment, int)):
                raise TypeError(
                    "KV_CACHE/paged_kv_cache alignment must be an "
                    "integer")
            if alignment <= 0 or alignment & (alignment - 1):
                raise ValueError(
                    "KV_CACHE/paged_kv_cache alignment must be a "
                    "positive power of two")
            expected_elements = (
                shape["seq_len"] * shape["num_kv_heads"]
                * shape["head_dim"])
        elif variant == "mla_kv_cache":
            if normalized["cache_write_semantics"] != "append":
                raise ValueError("KV_CACHE requires append semantics")
            if normalized["cache_layout"] != "paged_mla":
                raise ValueError(
                    "compressed MLA cache requires paged_mla layout")
            expected_elements = shape["seq_len"] * shape["compressed_dim"]
        elif variant == "dsa_fp8_index_cache":
            if normalized["cache_write_semantics"] != "append":
                raise ValueError("KV_CACHE requires append semantics")
            if normalized["cache_layout"] != "paged_mla_indexer":
                raise ValueError("DSA index cache layout is invalid")
            if normalized["quantization"] != (
                    "fp8_e4m3_with_ue8m0_scale"):
                raise ValueError("DSA index cache quantization is invalid")
            expected_elements = shape["seq_len"] * (
                shape["key_dim"] + shape["scale_bytes_per_token"])
        elif variant == "dsv4_compressed_state_cache":
            if normalized["cache_write_semantics"] != "append":
                raise ValueError("KV_CACHE requires append semantics")
            if normalized["cache_layout"] != "paged_dsv4_compressed":
                raise ValueError("DSV4 compressed state cache layout is invalid")
            quantization = normalized["quantization"]
            if quantization == "fp8_ds_mla":
                # Upstream fp8_ds_mla layout: 448B NoPE + 128B RoPE + 8B FP8
                # scale per compressed state (deepseek_v4/attention.py
                # MLAAttentionSpec state_content_bytes).
                expected_state_bytes = 584
            elif quantization == "fp8_e4m3_with_ue8m0_scale":
                # C4 indexer K cache: 128 e4m3 bytes + one FP32 scale per
                # 128 values (deepseek_v4/attention.py:993-997).
                expected_state_bytes = 132
            else:
                raise ValueError(
                    "DSV4 compressed state cache quantization is invalid: "
                    f"{quantization!r}")
            ratio = shape["compress_ratio"]
            if ratio not in (1, 4, 128):
                raise ValueError(
                    "DSV4 compress_ratio must be one of 1, 4, 128")
            stage = normalized["finish_stage"]
            if stage not in (None, "compress", "store"):
                raise ValueError("DSV4 finish_stage must be None, compress or store")
            if stage is not None and (ratio != 128 or expected_state_bytes != 584):
                raise ValueError("DSV4 split finish stages require C128 main-cache storage")
            if ratio == 1 or (ratio == 128 and expected_state_bytes != 584):
                raise NotImplementedError("DSV4 finish supports C4 main/indexer or C128 main storage")
            if shape["state_bytes"] != expected_state_bytes:
                raise ValueError(
                    f"DSV4 compressed state must be {expected_state_bytes} "
                    f"bytes under {quantization}")
            old_lens = _normalize_request_length_tuple(
                "state_old_lens", normalized["state_old_lens"],
                allow_zero=True)
            new_lens = _normalize_request_length_tuple(
                "state_new_lens", normalized["state_new_lens"],
                allow_zero=False)
            if len(old_lens) != len(new_lens):
                raise ValueError(
                    "DSV4 compressed state old/new request lengths must "
                    "cover the same requests")
            for old_len, new_len in zip(old_lens, new_lens, strict=True):
                if new_len < old_len:
                    raise ValueError(
                        "DSV4 compressed state request lengths must be "
                        "monotonic within a step")
            if sum(new_lens) - sum(old_lens) != shape["seq_len"]:
                raise ValueError(
                    "DSV4 compressed state request length deltas must sum "
                    "to seq_len")
            # Upstream only finishes (RMSNorm + RoPE + FP8-quant + paged
            # store) a compressed state when a request crosses a
            # compress_ratio window boundary: per request
            # floor(new_len/ratio) - floor(old_len/ratio) states
            # (deepseek_v4/compressor.py DeepseekCompressor); a partial
            # window stays in the FP32 partial-state cache until it
            # completes.
            normalized["state_old_lens"] = old_lens
            normalized["state_new_lens"] = new_lens
            # element_count describes this stage's typed output: 512 FP32
            # scratch values for compress, packed record bytes for store.
            expected_elements = sum(
                new_len // ratio - old_len // ratio
                for old_len, new_len in zip(old_lens, new_lens, strict=True)
            ) * (512 if stage == "compress" else shape["state_bytes"])
        elif variant == "dsv4_qnorm_rope_kv_insert":
            partial_context = ("mathematical_program" in normalized and normalized["mathematical_program"]
                == "vllm_0_30_dspark_partial_context_insert_v1")
            if (normalized["cache_write_semantics"] != "append" and not (partial_context
                    and normalized["cache_write_semantics"] == "masked_target_context_overwrite")):
                raise ValueError("DSV4 fused cache requires append semantics or its exact partial DSpark overwrite program")
            if normalized["cache_layout"] != "paged_dsv4_swa":
                raise ValueError("DSV4 fused cache requires paged_dsv4_swa")
            if normalized["quantization"] != "fp8_ds_mla":
                raise ValueError("DSV4 fused cache requires fp8_ds_mla")
            partial_context = ("mathematical_program" in normalized and normalized["mathematical_program"]
                == "vllm_0_30_dspark_partial_context_insert_v1")
            if partial_context:
                valid = normalized["valid_token_count"]
                if (type(valid) is not int or not 1 <= valid <= 6
                        or shape["seq_len"] != 6 or shape["num_query_heads"] not in (8, 16)
                        or shape["padded_query_heads"] != 64
                        or type(normalized["slot_mapping_all_valid"]) is not bool
                        or normalized["slot_mapping_all_valid"] != (valid == 6)):
                    raise ValueError("DSpark partial context insert requires full six rows and explicit valid prefix1..6")
            elif normalized["slot_mapping_all_valid"] is not True:
                raise ValueError("DSV4 fused cache requires valid token slots")
            if (shape["head_dim"] != 512 or shape["rope_dim"] != 64
                    or shape["cache_token_bytes"] != 584):
                raise ValueError("DSV4 fused cache requires 512/64/584 layout")
            heads = shape["num_query_heads"]
            padded = shape["padded_query_heads"]
            if padded not in (8, 16, 32, 64, 128) or not 0 < heads <= padded:
                raise ValueError("DSV4 fused cache has invalid Q head padding")
            expected_elements = shape["seq_len"] * (heads + 1) * 512
        elif variant == "dsv4_compressor_state_cache":
            # FP32 partial kv/score states written once per token
            # (deepseek_v4/compressor.py _SAVE_PARTIAL_STATES_KERNEL):
            # 2*coff*head_dim fp32 values per state — 8192B (main C4),
            # 4096B (main C128) or 2048B (indexer C4, head_dim=128).
            if normalized["cache_write_semantics"] != "append":
                raise ValueError("KV_CACHE requires append semantics")
            if normalized["cache_layout"] != "paged_dsv4_compressor_state":
                raise ValueError(
                    "DSV4 compressor state cache layout is invalid")
            if shape["state_bytes"] not in (2048, 4096, 8192):
                raise ValueError(
                    "DSV4 compressor state must be 2048, 4096 or 8192 bytes")
            expected_elements = shape["seq_len"] * shape["state_bytes"]
        else:
            raise AssertionError(
                f"unhandled KV_CACHE variant: {variant!r}")
        if shape["element_count"] != expected_elements:
            raise ValueError(
                "KV_CACHE element_count disagrees with its typed "
                "cache representation")
    elif kind is KernelKind.VECTOR and variant == "layernorm":
        if not isinstance(normalized["affine"], bool):
            raise TypeError("VECTOR/layernorm affine must be boolean")
        epsilon = normalized["epsilon"]
        if (isinstance(epsilon, bool)
                or not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon)
                or epsilon <= 0):
            raise ValueError("VECTOR/layernorm epsilon must be positive")
        normalized["epsilon"] = float(epsilon)
    elif kind is KernelKind.VECTOR and variant == "mla_rope":
        if normalized["rope_layout"] != "interleaved":
            raise ValueError("VECTOR/mla_rope requires interleaved layout")
        if normalized["rope_type"] not in ("default", "deepseek_yarn"):
            raise ValueError("VECTOR/mla_rope has unsupported rope_type")
        theta = normalized["rope_theta"]
        if (isinstance(theta, bool)
                or not isinstance(theta, (int, float))
                or not math.isfinite(theta) or theta <= 0):
            raise ValueError("VECTOR/mla_rope rope_theta must be positive")
        normalized["rope_theta"] = float(theta)
        expected_elements = (
            shape["seq_len"]
            * (shape["num_query_heads"] + shape["num_key_heads"])
            * shape["head_dim"]
        )
        if shape["element_count"] != expected_elements:
            raise ValueError(
                "VECTOR/mla_rope element_count disagrees with head geometry")
        yarn_names = (
            "rope_factor", "rope_original_max_position_embeddings",
            "rope_beta_fast", "rope_beta_slow", "rope_mscale",
            "rope_mscale_all_dim",
        )
        if normalized["rope_type"] == "default":
            if any(normalized[name] is not None for name in yarn_names):
                raise ValueError("default MLA RoPE cannot carry YaRN fields")
        elif normalized["rope_type"] == "deepseek_yarn":
            if any(normalized[name] is None for name in yarn_names):
                raise ValueError("DeepSeek YaRN requires every scaling field")
            original_max = normalized[
                "rope_original_max_position_embeddings"]
            if (isinstance(original_max, bool)
                    or not isinstance(original_max, int)
                    or original_max <= 0):
                raise ValueError(
                    "DeepSeek YaRN original max position must be positive")
            for name in (
                    "rope_factor", "rope_beta_fast", "rope_beta_slow"):
                value = normalized[name]
                if (isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value <= 0):
                    raise ValueError(
                        f"DeepSeek YaRN {name} must be finite and positive")
                normalized[name] = float(value)
            # DeepSeek-V4 hard-disables the YaRN attention scale
            # (mscale=mscale_all_dim=0 in deepseek_v4 config), so zero is
            # a valid "disabled" value alongside the positive scaling used
            # by DeepSeek-V3.
            for name in ("rope_mscale", "rope_mscale_all_dim"):
                value = normalized[name]
                if (isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0):
                    raise ValueError(
                        f"DeepSeek YaRN {name} must be finite and "
                        "non-negative")
                normalized[name] = float(value)
            if normalized["rope_beta_fast"] <= normalized["rope_beta_slow"]:
                raise ValueError(
                    "DeepSeek YaRN beta_fast must exceed beta_slow")
        else:
            raise AssertionError(
                f"unhandled MLA RoPE type: {normalized['rope_type']!r}")
    elif kind is KernelKind.VECTOR and variant == "dsv4_short_context_topk":
        if (shape["compress_ratio"] != 4
                or isinstance(normalized["position_bytes"], bool)
                or not isinstance(normalized["position_bytes"], int)
                or normalized["position_bytes"] != 8):
            raise ValueError("V4 short-context fill requires C4 and int64 positions")
        if shape["dtype_bytes"] != 4 or shape["element_count"] != shape["seq_len"] * shape["topk_capacity"]:
            raise ValueError("V4 short-context fill requires complete INT32 topk output rows")
    elif kind is KernelKind.VECTOR and variant in (
            "dsv32_draft_argmax", "dsv32_draft_id_cast", "dsv32_draft_row_select", "dsv32_draft_buffer_copy", "dsv32_draft_token_stack", "dsv32_draft_id_scatter", "dsv32_draft_prepare_next_ids", "dsv32_draft_prepare_sample_indices", "dsv32_draft_seq_rollback", "dsv32_draft_step_metadata"):
        if normalized["sampling_source"] == "vllm_0_30_dspark_full_vocab_greedy_bf16":
            if (variant != "dsv32_draft_argmax" or compute_dtype != "bf16"
                    or not 1 <= shape["rows"] <= 16 or shape["vocab"] != 129280):
                raise ValueError("DSpark greedy source requires BF16 argmax over 1..16 full 129280-value rows")
        elif normalized["sampling_source"] != "vllm_0_30_legacy_full_logits_greedy":
            raise ValueError("draft sampling requires an explicit recognized full-logits greedy source")
        width = (shape["vocab"] if variant == "dsv32_draft_argmax" else
                 shape["proposal_depth"] if variant == "dsv32_draft_token_stack" else
                 shape["sampled_width"] if variant == "dsv32_draft_prepare_next_ids" else
                 shape["hidden"] if variant in ("dsv32_draft_row_select", "dsv32_draft_buffer_copy") else 1)
        expected_elements = shape["rows"] * width
        if shape["element_count"] != expected_elements:
            raise ValueError("draft sampling must cover complete rows")
        if (variant in ("dsv32_draft_prepare_sample_indices", "dsv32_draft_seq_rollback")
                and normalized["index_preparation_mode"] not in
                ("ordinary_query_end", "padded_spec", "unbound_metadata")):
            raise ValueError("draft index preparation requires a declared metadata mode")
        if variant == "dsv32_draft_step_metadata":
            if not isinstance(normalized["positions_alias_output"], bool):
                raise TypeError("draft metadata position alias must be boolean")
            if shape["input_batch_size"] < shape["rows"]:
                raise ValueError("draft metadata input batch cannot truncate real rows")
        if variant == "dsv32_draft_token_stack" and shape["proposal_depth"] <= 1:
            raise ValueError("draft token stack requires multiple proposal passes; depth1 returns a view")
        if variant == "dsv32_draft_id_scatter":
            indices = normalized["selected_indices"]
            if not isinstance(indices, (tuple, list)) or len(indices) != shape["rows"]:
                raise ValueError("draft ID scatter requires one destination index per row")
            if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
                raise TypeError("draft ID scatter requires integer indices")
            if (any(not 0 <= index < shape["destination_rows"] for index in indices)
                    or any(left >= right for left, right in zip(indices, indices[1:]))):
                raise ValueError("draft ID scatter indices must increase within destination rows")
            normalized["selected_indices"] = tuple(indices)
        elif variant == "dsv32_draft_row_select":
            if normalized["selection_role"] not in ("logits_hidden", "recurrent_hidden", "positions", "topk_indices"):
                raise ValueError("draft row selection requires an explicit source role")
            indices = normalized["selected_indices"]
            if not isinstance(indices, (tuple, list)) or len(indices) != shape["rows"]:
                raise ValueError("draft row selection requires one index per selected row")
            if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
                raise TypeError("draft row selection requires integer indices")
            if (any(not 0 <= index < shape["source_rows"] for index in indices)
                    or any(left >= right for left, right in zip(indices, indices[1:]))):
                raise ValueError("draft row selection indices must be increasing within source rows")
            if normalized["selection_role"] == "positions" and shape["hidden"] != 1:
                raise ValueError("draft position selection requires scalar positions")
            normalized["selected_indices"] = tuple(indices)
        elif variant == "dsv32_draft_buffer_copy":
            if normalized["copy_role"] not in (
                    "input_ids", "hidden_states", "topk_indices", "target_token_ids_shift", "target_hidden_states", "positions", "slot_mapping"):
                raise ValueError("draft buffer copy requires an explicit source role")
            if normalized["copy_role"] in ("input_ids", "target_token_ids_shift", "positions", "slot_mapping") and shape["hidden"] != 1:
                raise ValueError("draft ID buffer copy requires scalar IDs")
    elif kind is KernelKind.VECTOR and variant == "vllm_rejection_logits_materialize":
        if normalized["sampling_source"] != "vllm_0_30_rejection_raw_logits_bf16":
            raise ValueError("rejection materialization requires an explicit raw-logits source")
        stage = normalized["materialization_stage"]
        if stage not in ("bonus_select", "bonus_raw_cast", "bonus_work_cast",
                         "target_select", "target_cast", "target_clone"):
            raise ValueError("unknown rejection materialization stage")
        if shape["element_count"] != shape["rows"] * shape["vocab"]:
            raise ValueError("rejection materialization must cover complete vocabulary rows")
        indices = normalized["selected_indices"]
        if not isinstance(indices, (tuple, list)):
            raise TypeError("rejection materialization indices must be a sequence")
        if stage in ("bonus_select", "target_select"):
            if (len(indices) != shape["rows"]
                    or any(type(index) is not int for index in indices)):
                raise ValueError("rejection selection requires one integer index per row")
            if (any(not 0 <= index < shape["source_rows"] for index in indices)
                    or any(left >= right for left, right in zip(indices, indices[1:]))):
                raise ValueError("rejection selection requires increasing in-bounds indices")
        elif indices or shape["source_rows"] != shape["rows"]:
            raise ValueError("rejection cast/clone consumes its complete materialized input")
        normalized["selected_indices"] = tuple(indices)
    elif kind is KernelKind.VECTOR and variant == "vllm_attention_state_merge":
        program = normalized["mathematical_program"]
        if program not in ("vllm_0_30_nonempty_lse_merge_flops_bytes_v1", "sglang_0_5_19_empty_prefix_lse_merge_flops_bytes_v1"):
            raise ValueError("attention state merge requires its explicit nonempty mathematical program")
        if program == "sglang_0_5_19_empty_prefix_lse_merge_flops_bytes_v1":
            if ("empty_prefix_rows" not in normalized or type(normalized["empty_prefix_rows"]) is not int
                    or not 1 <= normalized["empty_prefix_rows"] <= shape["rows"]):
                raise ValueError("empty-prefix merge requires its exact positive number of zero-O/negative-infinity-LSE rows")
        elif "empty_prefix_rows" in normalized:
            raise ValueError("nonempty attention merge cannot declare empty prefix rows")
        if type(normalized["output_lse"]) is not bool:
            raise TypeError("attention state merge output_lse must be boolean")
        if shape["element_count"] != shape["rows"] * shape["heads"] * shape["head_dim"]:
            raise ValueError("attention state merge must cover every output element")
    elif kind is KernelKind.VECTOR and variant == "kimi_k3_nope_context":
        stage = normalized["stage"]
        if stage not in ("q_kv_norm", "key_cache_concat", "output_gate", "context_gather", "context_key_concat", "merge",
                         "context_output_copy", "context_lse_copy", "context_partial_merge"):
            raise ValueError("K3 context vector requires an explicit source stage")
        rows, context, heads, block = (shape[name] for name in ("rows", "context_rows", "heads", "block_size"))
        capacity, slots, table = (normalized[name] for name in ("cache_capacity_tokens", "slot_mapping", "context_block_table"))
        chunk_fields = {"context_start", "context_length"} & set(normalized)
        if chunk_fields and chunk_fields != {"context_start", "context_length"}:
            raise ValueError("K3 context chunk requires both start and total length")
        start = normalized["context_start"] if chunk_fields else 0
        total = normalized["context_length"] if chunk_fields else context
        if (type(start) is not int or type(total) is not int or not 0 <= start < total < 2**31
                or start + context > total or start % block):
            raise ValueError("K3 context chunk must be aligned and contained in its complete INT32 prefix")
        if stage in ("context_output_copy", "context_lse_copy", "context_partial_merge"):
            if not chunk_fields:
                raise ValueError("K3 context accumulation requires explicit chunk metadata")
            if ((stage == "context_partial_merge" and start == 0)
                    or (stage in ("context_output_copy", "context_lse_copy") and start != 0)):
                raise ValueError("K3 context continuation must merge; only the first chunk initializes copies")
        if (rows < 512 or context <= 0 or block <= 0 or block >= 2**31 or block & (block - 1)
                or heads != 96 or (shape["q_lora_rank"], shape["kv_lora_rank"]) != (1536, 512)
                or type(capacity) is not int or capacity <= 0 or capacity % block
                or not isinstance(table, (list, tuple)) or len(table) != -(-total // block)
                or any(type(index) is not int or not 0 <= index < min(capacity // block, 2**31) for index in table)
                or len(set(table)) != len(table)
                or not isinstance(slots, (list, tuple)) or len(slots) != rows
                or any(type(slot) is not int or not 0 <= slot < min(capacity, 2**63) for slot in slots)
                or len(set(slots)) != rows):
            raise ValueError("K3 context requires exact TP1 geometry and valid INT32 paged history/distinct INT64 new slots")
        history = {table[position // block] * block + position % block for position in range(total)}
        if history.intersection(slots):
            raise ValueError("K3 context new writes must not overwrite gathered history")
        normalized["slot_mapping"], normalized["context_block_table"] = tuple(slots), tuple(table)
        eps = normalized["rms_norm_eps"]
        if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("K3 context requires a positive finite RMSNorm epsilon")
        elements = {"q_kv_norm": rows * 2048, "key_cache_concat": rows * (heads * 192 + 576),
                    "output_gate": rows * heads * 128, "context_gather": context * 576,
                    "context_key_concat": context * heads * 192, "merge": rows * heads * 128,
                    "context_output_copy": rows * heads * 128, "context_lse_copy": rows * heads,
                    "context_partial_merge": rows * heads * 128}[stage]
        expected_dtype = "fp32" if stage == "context_lse_copy" else "bf16"
        if shape["element_count"] != elements or compute_dtype != expected_dtype:
            raise ValueError("K3 context requires complete BF16 source outputs and FP32 LSE side roles")
    elif kind is KernelKind.VECTOR and variant in ("kimi_k3_nope_prefill", "kimi_k3_nope_decode", "kimi_k3_nope_mixed"):
        stage = normalized["stage"]
        concat_stages = {
            "kimi_k3_nope_prefill": ("key_cache_concat",),
            "kimi_k3_nope_decode": ("query_cache_concat",),
            "kimi_k3_nope_mixed": ("key_cache_concat", "query_cache_concat"),
        }[variant]
        if stage not in ("q_kv_norm", *concat_stages, "output_gate"):
            raise ValueError("K3 NoPE requires an explicit source stage for its prefill/decode branch")
        slots, capacity = normalized["slot_mapping"], normalized["cache_capacity_tokens"]
        if (type(capacity) is not int or capacity <= 0
                or not isinstance(slots, (tuple, list)) or len(slots) != shape["rows"]
                or any(type(slot) is not int or not 0 <= slot < capacity for slot in slots)
                or len(set(slots)) != len(slots)):
            raise ValueError("K3 NoPE prefill requires distinct valid nonnegative INT64 slot mappings")
        normalized["slot_mapping"] = tuple(slots)
        eps = normalized["rms_norm_eps"]
        if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("K3 NoPE prefill requires a positive finite RMSNorm epsilon")
        if (shape["rows"] <= 0 or shape["heads"] not in (12, 96)
                or (shape["q_lora_rank"], shape["kv_lora_rank"]) != (1536, 512)):
            raise ValueError("K3 NoPE prefill requires exact checkpoint latent/head geometry")
        per_row = (2048 if stage == "q_kv_norm" else
                   shape["heads"] * 192 + 576 if stage == "key_cache_concat" else
                   shape["heads"] * 576 if stage == "query_cache_concat" else shape["heads"] * 128)
        if shape["element_count"] != shape["rows"] * per_row:
            raise ValueError("K3 NoPE prefill element count must preserve its complete source outputs")
    elif kind is KernelKind.VECTOR and variant in ("kimi_k3_kda_prefill", "kimi_k3_kda_mixed", "kimi_k3_kda_context_prefill"):
        stage = normalized["stage"]
        if stage not in ("q_convolution", "k_convolution", "v_convolution", "initial_state", "chunk",
                         "final_state_scatter", "output_copy", "output_norm"):
            raise ValueError("K3 KDA prefill requires an explicit source stage")
        indices, slots = normalized["state_indices"], normalized["state_pool_slots"]
        sequences = 2 if variant == "kimi_k3_kda_mixed" else 1
        if (type(slots) is not int or not sequences < slots <= 2**31
                or not isinstance(indices, (tuple, list)) or len(indices) != sequences
                or any(type(index) is not int or not 0 < index < slots for index in indices)
                or len(set(indices)) != sequences):
            raise ValueError("K3 KDA chunk requires distinct positive INT32 state indices, not null/padding")
        normalized["state_indices"] = tuple(indices)
        loc = normalized["query_start_loc"]
        expected_loc = (0, shape["rows"]) if sequences == 1 else (0, 1, shape["rows"])
        if (not isinstance(loc, (tuple, list)) or any(type(value) is not int for value in loc)
                or tuple(loc) != expected_loc):
            raise ValueError("K3 KDA chunk requires complete declared INT32 query segments")
        normalized["query_start_loc"] = tuple(loc)
        flags = normalized["has_initial_state"]
        if sequences == 1:
            valid_flags = flags is (variant == "kimi_k3_kda_context_prefill")
        else:
            valid_flags = (isinstance(flags, (tuple, list)) and len(flags) == 2
                           and all(type(flag) is bool for flag in flags) and tuple(flags) == (True, False))
            if valid_flags:
                normalized["has_initial_state"] = tuple(flags)
        prefill_rows = shape["rows"] - (sequences - 1)
        if (not valid_flags or type(normalized["gate_lower_bound"]) not in (int, float)
                or normalized["gate_lower_bound"] != -5.0
                or not 64 <= prefill_rows < 2**31 - (sequences - 1)
                or (shape["heads"], shape["head_dim"]) != (96, 128)):
            raise ValueError("K3 KDA chunk requires full/tail64-token fresh prefill, 96x128 heads and source gate bound-5")
        elements = (sequences * 96 * 128 * 128 if stage in ("initial_state", "final_state_scatter")
                    else shape["rows"] * 96 * 128)
        if shape["element_count"] != elements:
            raise ValueError("K3 KDA prefill element count must preserve its complete source output")
    elif kind is KernelKind.VECTOR and variant == "kimi_k3_kda_decode":
        if normalized["stage"] not in ("convolution", "recurrent", "output_copy", "output_norm"):
            raise ValueError("K3 KDA decode requires an explicit source stage")
        slots = normalized["state_pool_slots"]
        indices = normalized["state_indices"]
        if (type(slots) is not int or slots <= 1
                or not isinstance(indices, (tuple, list)) or len(indices) != shape["rows"]
                or any(type(index) is not int for index in indices)):
            raise ValueError("K3 KDA decode requires one integer state index per row and a positive pool")
        if any(not 0 < index < slots for index in indices) or len(set(indices)) != len(indices):
            raise ValueError("K3 KDA decode supports distinct positive active state indices only; null/padding is separate")
        normalized["state_indices"] = tuple(indices)
        if (shape["head_dim"] != 128 or shape["heads"] not in (12, 64, 96)
                or shape["rows"] <= 0 or shape["element_count"] != shape["rows"] * shape["heads"] * 128):
            raise ValueError("KDA source math requires 128-wide TP1/8 or explicit 64-head operator geometry and complete rows")
        if type(normalized["gate_lower_bound"]) not in (int, float) or normalized["gate_lower_bound"] != -5.0:
            raise ValueError("K3 KDA decode requires the exact negative gate bound")
    elif kind is KernelKind.VECTOR and variant == "kimi_k3_situ":
        if (type(normalized["activation_beta"]) not in (int, float)
                or type(normalized["activation_linear_beta"]) not in (int, float)
                or normalized["activation_beta"] != 4.0
                or normalized["activation_linear_beta"] != 25.0):
            raise ValueError("exact K3 SiTU requires beta4 and linear beta25")
        if (shape["rows"] <= 0 or shape["intermediate"] <= 0
                or shape["element_count"] != shape["rows"] * shape["intermediate"]):
            raise ValueError("K3 SiTU requires complete positive output rows")
    elif kind is KernelKind.VECTOR and variant == "kimi_k3_attn_res":
        for name in ("num_blocks", "block_write_idx", "block_capacity"):
            if type(normalized[name]) is not int:
                raise TypeError(f"K3 attention residual {name} must be integer")
        for name in ("has_delta", "apply_output_norm"):
            if type(normalized[name]) is not bool:
                raise TypeError(f"K3 attention residual {name} must be boolean")
        if (not 0 <= normalized["num_blocks"] <= normalized["block_capacity"]
                or not -1 <= normalized["block_write_idx"] < normalized["block_capacity"]
                or normalized["block_capacity"] <= 0
                or (normalized["block_write_idx"] >= 0
                    and normalized["block_write_idx"] != normalized["num_blocks"])):
            raise ValueError("K3 attention residual requires valid initialized blocks and append-only writes")
        if (shape["hidden"] != 7168 or shape["rows"] <= 0
                or shape["element_count"] != shape["rows"] * shape["hidden"]):
            raise ValueError("K3 attention residual requires complete exact-checkpoint hidden rows")
    elif kind is KernelKind.VECTOR and variant == "vllm_logits_gather_reorder":
        if (normalized["model_execution_version"] not in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_30_0", "vllm_0_27_1")
                or shape["rows"] <= 1 or shape["tensor_parallel_size"] <= 1
                or shape["vocab"] % shape["tensor_parallel_size"]
                or shape["vocab"] // shape["tensor_parallel_size"] <= 1 or shape["dtype_bytes"] != 2
                or shape["element_count"] != shape["rows"] * shape["vocab"]):
            raise ValueError("logits gather reorder requires complete multirow TP vocabulary at its declared vLLM source")
    elif kind is KernelKind.VECTOR and variant == "vllm_mtp_eager":
        if (normalized["model_execution_version"] not in ("vllm_0_11_0", "vllm_0_27_1", "vllm_0_30_0", "sglang_0_5_19")
                or shape["rows"] <= 0 or shape["width"] <= 0 or shape["source_rows"] <= 0
                or shape["element_count"] != shape["rows"] * shape["width"]):
            raise ValueError("MTP eager operation requires its explicit source version and positive row geometry")
        operations = {"copy", "concat", "gather", "scatter", "argmax", "cast_ids", "stack", "fill", "compare",
            "logical_and", "sum", "cumsum", "unary", "subtract", "promote_subtract", "promote_add", "where", "where_zero", "where_promote",
            "masked_fill", "row_gather", "rejection", "cast_fp32", "prepare_next_padded", "prepare_inputs_padded", "step_metadata",
            "cast_int64", "promote_concat", "eagle_tree_build", "eagle_tree_verify", "eagle_bonus",
            "eagle_max", "eagle_eh_norm", "eagle_softmax", "eagle_positions", "eagle_topk1_partial", "eagle_topk1_finalize",
            "dspark_aux_mean", "dspark_repeat_hc", "dspark_sigmoid", "dspark_swa_indices", "dspark_zero_query", "dspark_greedy_verify", "dspark_roundtrip_prepare"}
        operation = normalized["operation"]
        if operation not in operations:
            raise ValueError("unsupported MTP eager source operation")
        if operation == "dspark_roundtrip_prepare":
            if (normalized["model_execution_version"] != "vllm_0_30_0"
                    or type(normalized["num_speculative_tokens"]) is not int
                    or normalized["num_speculative_tokens"] != 5
                    or type(normalized["accepted_drafts"]) is not int
                    or not 0 <= normalized["accepted_drafts"] <= 5
                    or shape["rows"] != 1 or shape["width"] != 6 or shape["source_rows"] != 6
                    or shape["dtype_bytes"] != 8):
                raise ValueError("DSpark roundtrip metadata requires vLLM0.30 B1 N6 K5, INT64 positions and explicit A0..5")
        if operation == "dspark_greedy_verify":
            if (normalized["model_execution_version"] != "vllm_0_30_0"
                    or type(normalized["num_speculative_tokens"]) is not int
                    or normalized["num_speculative_tokens"] != 5
                    or type(normalized["accepted_drafts"]) is not int
                    or not 0 <= normalized["accepted_drafts"] <= 5
                    or shape["rows"] != 1 or shape["width"] != 6 or shape["source_rows"] != 6
                    or shape["dtype_bytes"] != 4):
                raise ValueError("DSpark greedy prefix projection requires vLLM0.30 B1 K5, INT32 drafts and explicit A0..5")
        if operation == "dspark_zero_query":
            if (normalized["model_execution_version"] != "vllm_0_30_0"
                    or shape["source_rows"] != shape["rows"] or shape["width"] not in (4096, 8192)
                    or shape["dtype_bytes"] != 2):
                raise ValueError("DSpark dummy Q zeros require vLLM0.30 BF16 localheads8/16 x512 and exact rows")
        if operation == "dspark_swa_indices":
            kv_lens = normalized["kv_lens"]
            if (normalized["model_execution_version"] != "vllm_0_30_0"
                    or type(normalized["num_speculative_tokens"]) is not int
                    or normalized["num_speculative_tokens"] != 5
                    or shape["rows"] != 5 * shape["source_rows"] or shape["width"] != 192
                    or shape["dtype_bytes"] != 4
                    or not isinstance(kv_lens, (tuple, list)) or len(kv_lens) != shape["source_rows"]
                    or any(type(kv) is not int or not 5 <= kv < 2**31 for kv in kv_lens)):
                raise ValueError("DSpark SWA indices require vLLM0.30 all-valid block5, width192 and real KV bounds")
            normalized["kv_lens"] = tuple(kv_lens)
        if operation in ("dspark_aux_mean", "dspark_repeat_hc", "dspark_sigmoid"):
            if normalized["model_execution_version"] != "vllm_0_30_0":
                raise ValueError("DSpark mathematical primitives require the explicit vLLM0.30 source")
            expected_source_rows = (4 * shape["rows"] if operation == "dspark_aux_mean"
                else shape["rows"] // 4 if operation == "dspark_repeat_hc" else shape["rows"])
            if (shape["source_rows"] != expected_source_rows
                    or (operation == "dspark_repeat_hc" and shape["rows"] % 4)
                    or shape["dtype_bytes"] != (4 if operation == "dspark_sigmoid" else 2)):
                raise ValueError("DSpark primitives require exact hc4 input/output rows and typed storage")
        sglang = normalized["model_execution_version"] == "sglang_0_5_19"
        eagle_operations = {"cast_int64", "promote_concat", "eagle_tree_build", "eagle_tree_verify",
            "eagle_bonus", "eagle_max", "eagle_eh_norm", "eagle_softmax", "eagle_positions", "eagle_topk1_partial", "eagle_topk1_finalize"}
        if sglang:
            eagle = normalized["eagle_source"] in ("topk1_steps1_eager", "topk1_steps2_eager")
            if normalized["eagle_source"] not in (None, "topk1_steps1_eager", "topk1_steps2_eager"):
                raise ValueError("SGLang typed operations require the declared topk1 one/two-step eager source")
            allowed = {"cumsum", "unary", "gather"} | ({"copy", "fill", "concat", "promote_add"} | eagle_operations if eagle else set())
            if (operation not in allowed
                    or (operation in ("cumsum", "unary") and (shape["width"] != 1 or shape["source_rows"] != shape["rows"]))
                    or (operation == "unary" and compute_dtype not in (("int32", "int64") if eagle else ("int64",)))
                    or (operation == "gather" and (compute_dtype not in (("bf16", "fp32", "int32", "int64") if eagle else ("bf16",))
                        or normalized["index_bytes"] not in ((4, 8) if eagle else (8,))))):
                raise ValueError("SGLang typed source operation, dtype or geometry is outside its declared scope")
            if operation in ("promote_concat", "eagle_tree_build", "eagle_tree_verify", "eagle_bonus"):
                width = 3 if normalized["eagle_source"] == "topk1_steps2_eager" else 2
                if shape["width"] != width:
                    raise ValueError("SGLang EAGLE requires its explicit source chain width")
            if operation == "eagle_tree_verify":
                accepted = normalized["accepted_drafts"]
                if type(accepted) is int:
                    if not 0 <= accepted < shape["width"]:
                        raise ValueError("SGLang chain verify requires a bounded accepted-draft scenario")
                elif (not isinstance(accepted, tuple) or len(accepted) != shape["rows"]
                        or any(type(value) is not int or not 0 <= value < shape["width"] for value in accepted)):
                    raise ValueError("SGLang chain verify requires one bounded accepted-draft count per request")
            if operation == "eagle_positions":
                queries = normalized["query_lens"]
                if (not isinstance(queries, tuple) or len(queries) != shape["rows"]
                        or any(type(query) is not int or query <= 0 for query in queries) or max(queries) != shape["width"]):
                    raise ValueError("SGLang draft positions require complete positive per-request query lengths")
            if operation in ("eagle_topk1_partial", "eagle_topk1_finalize"):
                if normalized["eagle_source"] != "topk1_steps2_eager":
                    raise ValueError("SGLang split topk1 is bound to the declared two-step draft-decode source")
                if operation == "eagle_topk1_finalize" and type(normalized["write_draft_token"]) is not bool:
                    raise ValueError("SGLang split topk1 requires its explicit optional draft-column write")
            if operation == "eagle_eh_norm" and (type(normalized["epsilon"]) not in (int, float)
                    or not math.isfinite(normalized["epsilon"]) or normalized["epsilon"] <= 0):
                raise ValueError("SGLang fused EH norm requires a finite positive source epsilon")
            if operation == "eagle_eh_norm" and (not 256 < shape["width"] <= 8192 or shape["width"] % 256):
                raise ValueError("SGLang fused EH norm requires the supported source hidden geometry")
        elif operation == "cumsum" or operation in eagle_operations:
            raise ValueError("this typed operation is bound only to the declared SGLang source")
        if operation in ("cast_fp32", "prepare_next_padded", "prepare_inputs_padded", "step_metadata") and normalized["model_execution_version"] not in ("vllm_0_27_1", "vllm_0_30_0"):
            raise ValueError("MTP fused preparation and FP32 cast require the explicit0.27.1 or0.30.0 source")
        allowed_dtypes = {
            "copy": ("bf16", "int32", "int64"), "concat": ("int32",),
            "gather": ("bf16", "int32", "int64"), "scatter": ("int32",),
            "argmax": ("bf16", "fp32"), "cast_ids": ("int64",), "stack": ("int64",),
            "fill": ("bool", "int32", "int64"), "compare": ("int32", "int64"),
            "logical_and": ("bool",), "sum": ("bool",), "cumsum": ("int32",), "unary": ("int32", "int64"),
            "subtract": ("int32", "int64"), "promote_subtract": ("int32",), "promote_add": ("int32",),
            "where": ("int32", "int64"), "where_zero": ("int64",), "where_promote": ("int64",),
            "masked_fill": ("int32", "int64"), "row_gather": ("int32",), "rejection": ("int32",),
            "cast_fp32": ("bf16",), "prepare_next_padded": ("int32",),
            "prepare_inputs_padded": ("int32",), "step_metadata": ("int64",),
            "dspark_aux_mean": ("bf16",), "dspark_repeat_hc": ("bf16",), "dspark_sigmoid": ("fp32",),
            "dspark_swa_indices": ("int32",), "dspark_zero_query": ("bf16",),
            "dspark_greedy_verify": ("int32",), "dspark_roundtrip_prepare": ("int64",)}
        if normalized["model_execution_version"] in ("vllm_0_27_1", "vllm_0_30_0"):
            allowed_dtypes["copy"] = (*allowed_dtypes["copy"], "fp32")
        if sglang and normalized["eagle_source"] in ("topk1_steps1_eager", "topk1_steps2_eager"):
            allowed_dtypes.update({"copy": ("bf16", "fp32", "int32", "int64"), "concat": ("int64",),
                "gather": ("bf16", "fp32", "int32", "int64"), "fill": ("bool", "fp32", "int32", "int64"),
                "cast_int64": ("int32",), "promote_concat": ("int32",), "eagle_tree_build": ("int64",),
                "eagle_tree_verify": ("int64",), "eagle_bonus": ("int32",), "eagle_max": ("fp32",),
                "eagle_eh_norm": ("bf16",), "eagle_softmax": ("fp32",), "eagle_positions": ("int32",),
                "eagle_topk1_partial": ("fp32",), "eagle_topk1_finalize": ("fp32",)})
        if sglang and normalized["eagle_source"] == "topk1_steps2_eager":
            allowed_dtypes["fill"] = (*allowed_dtypes["fill"], "bf16")
        if compute_dtype not in allowed_dtypes[operation]:
            raise ValueError("MTP eager operation input dtype disagrees with its declared source")
        if operation in ("gather", "scatter"):
            indices = normalized["selected_indices"]
            if (normalized["index_bytes"] not in (4, 8) or type(normalized["index_bytes"]) is not int
                    or not isinstance(indices, (tuple, list)) or len(indices) != shape["rows"]
                    or any(type(index) is not int or not 0 <= index < shape["source_rows"] for index in indices)):
                raise ValueError("MTP eager indexing requires explicit typed in-bounds row indices")
            normalized["selected_indices"] = tuple(indices)
            if operation == "scatter" and len(set(indices)) != len(indices):
                raise ValueError("MTP eager ID scatter requires distinct destination indices")
        elif operation == "masked_fill":
            if type(normalized["masked_rows"]) is not int or not 0 <= normalized["masked_rows"] <= shape["rows"]:
                raise ValueError("MTP eager mask requires a declared number of selected rows")
        elif operation == "rejection":
            verified, accepted = normalized["verified_tokens"], normalized["accepted_tokens"]
            if (not isinstance(verified, (tuple, list)) or not isinstance(accepted, (tuple, list))
                    or len(verified) != shape["rows"] or len(accepted) != shape["rows"]
                    or any(type(v) is not int or type(a) is not int or not 0 <= a <= v < shape["width"]
                        for v, a in zip(verified, accepted))):
                raise ValueError("MTP rejection requires one explicit valid acceptance assumption per row")
            normalized["verified_tokens"], normalized["accepted_tokens"] = tuple(verified), tuple(accepted)
        elif operation == "prepare_next_padded":
            counts, discarded = normalized["valid_sampled_counts"], normalized["discarded_rows"]
            if (type(normalized["vocab_size"]) is not int or not 0 < normalized["vocab_size"] < 2**31
                    or not isinstance(counts, (tuple, list)) or not isinstance(discarded, (tuple, list))
                    or len(counts) != shape["rows"] or len(discarded) != shape["rows"]
                    or any(type(count) is not int or not 0 <= count <= shape["width"]
                        or type(flag) is not bool or (flag and count != 0) for count, flag in zip(counts, discarded))
                    or shape["source_rows"] != shape["rows"]):
                raise ValueError("MTP padded next-token preparation requires explicit valid/discard row scenarios")
            normalized["valid_sampled_counts"], normalized["discarded_rows"] = tuple(counts), tuple(discarded)
        elif operation == "step_metadata":
            if (any(type(normalized[name]) is not int or normalized[name] <= 0
                    for name in ("block_size", "max_model_len", "block_table_columns"))
                    or normalized["block_table_columns"] < -(-normalized["max_model_len"] // normalized["block_size"])
                    or shape["width"] != 1 or shape["source_rows"] != shape["rows"]):
                raise ValueError("MTP step metadata requires explicit complete block-table geometry")
        elif operation == "prepare_inputs_padded":
            if shape["width"] != 1 or shape["source_rows"] != shape["rows"]:
                raise ValueError("MTP padded input preparation requires one output pair per request")
        if operation in ("cast_fp32", "prepare_next_padded", "prepare_inputs_padded", "step_metadata"):
            if shape["dtype_bytes"] != {"cast_fp32": 2, "prepare_next_padded": 4,
                    "prepare_inputs_padded": 4, "step_metadata": 8}[operation]:
                raise ValueError("MTP source preparation input storage width disagrees with its typed ports")
        if operation in ("cast_ids", "sum", "unary", "promote_subtract", "promote_add", "row_gather", "rejection", "scatter"):
            required_width = {"cast_ids": 8, "sum": 1, "unary": shape["dtype_bytes"],
                "promote_subtract": 4, "promote_add": 4, "row_gather": 4, "rejection": 4, "scatter": 4}[operation]
            if shape["dtype_bytes"] != required_width:
                raise ValueError("MTP eager operation input dtype disagrees with its source")
        if operation == "logical_and" and shape["dtype_bytes"] != 1:
            raise ValueError("MTP logical mask inputs must be boolean")
        if operation == "argmax" and shape["dtype_bytes"] not in (2, 4):
            raise ValueError("MTP argmax input must be BF16 or FP32")
    elif kind is KernelKind.VECTOR and variant == "vllm_mtp_position_mask":
        mask_stages = {"vllm_0_11_0": ("compare", "fill"), "vllm_0_27_1": ("compare", "where")}
        if (normalized["model_execution_version"] not in mask_stages
                or normalized["stage"] not in mask_stages[normalized["model_execution_version"]]
                or type(normalized["masked_rows"]) is not int
                or not 0 <= normalized["masked_rows"] <= shape["rows"]
                or shape["element_count"] != shape["rows"] * shape["hidden"]
                or shape["dtype_bytes"] != (8 if normalized["stage"] == "compare" else 2)):
            raise ValueError("MTP mask requires its version-bound INT64 compare/BF16 fill or where source")
    elif kind is KernelKind.VECTOR and variant == "vllm_logits_fp32_cast":
        if normalized["sampling_source"] not in ("vllm_0_30_plain_sampler_bf16", "vllm_0_11_plain_sampler_bf16", "vllm_0_24_plain_sampler_bf16", "vllm_0_18_plain_sampler_bf16", "vllm_0_30_plain_sampler_fp16", "vllm_0_11_plain_sampler_fp16", "vllm_0_27_1_plain_sampler_bf16", "sglang_0_5_19_logits_processor_bf16"):
            raise ValueError("target logits cast requires an explicit plain sampler source")
        if "copy_input_contiguous" in normalized and type(normalized["copy_input_contiguous"]) is not bool:
            raise TypeError("logits Copy input contiguity must be boolean")
        if "copy_dispatch" in normalized:
            from src.kernel.launch_config.metax_c500.vector_launch import pytorch_logits_cast_launch
            pytorch_logits_cast_launch(shape, compute_dtype, normalized)
            normalized["copy_dispatch"] = MappingProxyType(dict(normalized["copy_dispatch"]))
        if shape["element_count"] != shape["rows"] * shape["vocab"]:
            raise ValueError("target logits cast must cover complete vocabulary rows")
    elif kind is KernelKind.VECTOR and variant == "vllm_plain_greedy_sample":
        if normalized["sampling_source"] not in ("vllm_0_30_plain_greedy_bf16", "vllm_0_11_plain_greedy_bf16", "vllm_0_24_plain_greedy_bf16", "vllm_0_18_plain_greedy_bf16", "vllm_0_30_plain_greedy_fp16", "vllm_0_11_plain_greedy_fp16", "vllm_0_27_1_plain_greedy_bf16", "sglang_0_5_19_plain_greedy_fp32"):
            raise ValueError("plain greedy sampling requires an explicit no-processor/no-logprobs source")
        if normalized["stage"] not in ("argmax", "output_int32"):
            raise ValueError("plain greedy sampling requires an explicit source stage")
        if normalized["sampling_source"] == "sglang_0_5_19_plain_greedy_fp32" and normalized["stage"] != "argmax":
            raise ValueError("SGLang CUDA ordinary greedy returns INT64 argmax without an INT32 output cast")
        expected_elements = shape["rows"] * shape["vocab"] if normalized["stage"] == "argmax" else shape["rows"]
        if shape["rows"] <= 0 or not 0 < shape["vocab"] < 2**31 or shape["element_count"] != expected_elements:
            raise ValueError("plain greedy sampling requires complete vocabulary rows and INT32-safe IDs")
        source_properties = {"source_num_compute_units", "source_max_threads_per_unit"}
        if source_properties & set(normalized):
            if (not source_properties <= set(normalized) or normalized["stage"] != "argmax"
                    or normalized["sampling_source"] != "vllm_0_11_plain_greedy_bf16"
                    or any(type(normalized[key]) is not int or normalized[key] <= 0 for key in source_properties)):
                raise ValueError("argmax source properties require their complete positive0.11 hardware contract")
    elif kind is KernelKind.VECTOR and variant == "vllm_embedding_id_widen":
        if normalized["embedding_source"] not in ("vllm_0_30_unquantized_tp1", "vllm_0_11_unquantized_tp1", "vllm_0_18_unquantized_tp1", "vllm_0_24_unquantized_tp1"):
            raise ValueError("embedding ID widen requires the unquantized TP1 source")
        if shape["rows"] <= 0 or shape["element_count"] != shape["rows"]:
            raise ValueError("embedding ID widen requires one INT32 input ID per row")
        if "copy_dispatch" in normalized:
            if normalized["embedding_source"] != "vllm_0_24_unquantized_tp1":
                raise ValueError("integer Copy dispatch requires its C5000.24 embedding source")
            from src.kernel.launch_config.metax_c500.vector_launch import pytorch_int32_int64_cast_launch
            pytorch_int32_int64_cast_launch(shape["element_count"], normalized, variant)
            normalized["copy_dispatch"] = MappingProxyType(dict(normalized["copy_dispatch"]))
    elif kind is KernelKind.VECTOR and variant == "vllm_masked_embedding":
        if normalized["mask_program"] not in ("vllm_0_11", "vllm_0_24", "vllm_0_27_1", "vllm_0_30", "vllm_ascend_no_added_vocab"):
            raise ValueError("masked embedding requires its supported source mask program")
        for key in ("vocab_size", "tensor_parallel_size", "tensor_parallel_rank"):
            if type(normalized[key]) is not int:
                raise TypeError("masked embedding vocabulary and TP geometry must be integers")
        vocab, tp, rank = (normalized[key] for key in ("vocab_size", "tensor_parallel_size", "tensor_parallel_rank"))
        if not 0 < vocab < 2**31 or tp < 2 or not 0 <= rank < tp or (-(-vocab // 64) * 64) % tp:
            raise ValueError("masked embedding requires complete padded TP vocabulary shards")
        stage = normalized["stage"]
        if stage not in ("mask_ids", "mask_output"):
            raise ValueError("masked embedding requires an explicit source stage")
        expected = shape["rows"] if stage == "mask_ids" else shape["rows"] * shape["hidden"]
        if shape["rows"] <= 0 or shape["hidden"] <= 0 or shape["element_count"] != expected:
            raise ValueError("masked embedding source stage geometry disagrees")
    elif kind is KernelKind.VECTOR and variant == "vllm_logits_row_select":
        if "disable_index_broadcast" in normalized:
            if type(normalized["disable_index_broadcast"]) is not bool:
                raise TypeError("index broadcast flag must be boolean")
            if normalized["model_execution_version"] != "vllm_0_24_0":
                raise ValueError("index broadcast dispatch requires its C5000.24 source")
        if normalized["model_execution_version"] not in ("vllm_0_30_0", "vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_27_1"):
            raise ValueError("target logits row selection requires explicit vLLM source")
        if (normalized["model_execution_version"] == "vllm_0_27_1"
                and normalized["index_source"] != "query_start_loc_int32"):
            raise ValueError("target logits row selection at0.27.1 requires ordinary INT32 query indices")
        if normalized["index_source"] not in (
                "query_start_loc_int32", "query_start_loc_int32_to_int64",
                "spec_decode_metadata_int64", "unbound_spec_metadata"):
            raise ValueError("target logits row selection requires a declared index source")
        if (normalized["index_source"] == "query_start_loc_int32_to_int64"
                and normalized["model_execution_version"] not in ("vllm_0_11_0", "vllm_0_24_0")):
            raise ValueError("materialized query index conversion requires the declared0.11 or0.24 source")
        indices = normalized["selected_indices"]
        if not isinstance(indices, (tuple, list)) or len(indices) != shape["rows"]:
            raise ValueError("target logits row selection requires one index per logit row")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise TypeError("target logits row selection requires integer indices")
        if (shape["element_count"] != shape["rows"] * shape["hidden"]
                or any(not 0 <= index < shape["source_rows"] for index in indices)
                or any(left >= right for left, right in zip(indices, indices[1:]))):
            raise ValueError("target logits row selection must cover increasing in-bounds complete rows")
        normalized["selected_indices"] = tuple(indices)
    elif kind is KernelKind.VECTOR and variant == "vocab_parallel_embedding":
        for name in ("vocab_size", "tensor_parallel_size", "tensor_parallel_rank"):
            if isinstance(normalized[name], bool) or not isinstance(normalized[name], int):
                raise TypeError(f"vocab embedding {name} must be integer")
        owned_rows = normalized["owned_token_rows"]
        if owned_rows is not None and (type(owned_rows) is not int or not 0 <= owned_rows <= shape["rows"]):
            raise ValueError("vocab embedding owned_token_rows must be an integer within scheduled rows")
        if (normalized["vocab_size"] <= 0 or normalized["tensor_parallel_size"] <= 1
                or not 0 <= normalized["tensor_parallel_rank"] < normalized["tensor_parallel_size"]):
            raise ValueError("vocab embedding requires a valid TP shard")
        if (-(-normalized["vocab_size"] // 64) * 64) % normalized["tensor_parallel_size"]:
            raise ValueError("vocab embedding padded vocabulary must divide evenly over TP")
        shard_rows = (-(-normalized["vocab_size"] // 64) * 64) // normalized["tensor_parallel_size"]
        if normalized["tensor_parallel_rank"] * shard_rows >= normalized["vocab_size"] and owned_rows not in (None, 0):
            raise ValueError("vocab embedding empty shard cannot own scheduled rows")
        if (shape["input_id_bytes"] not in (4, 8)
                or shape["element_count"] != shape["rows"] * shape["hidden"]):
            raise ValueError("vocab embedding requires INT32 ids and complete output rows")
    elif kind is KernelKind.VECTOR and variant in ("dsv32_moe_route", "kimi_moe_route", "kimi_k3_moe_route"):
        expected_experts, expected_topk, source_model = {
            "dsv32_moe_route": (256, 8, "GLM V32"),
            "kimi_moe_route": (384, 8, "Kimi"),
            "kimi_k3_moe_route": (896, 16, "Kimi K3"),
        }[variant]
        if (shape["experts"] != expected_experts or shape["topk"] != expected_topk or shape["rows"] <= 0
                or shape["element_count"] != shape["rows"] * shape["experts"]
                or normalized["expert_id_selection"] != "score_topk"
                or normalized["score_function"] != "sigmoid"
                or type(normalized["expert_group_count"]) is not int
                or type(normalized["topk_group_count"]) is not int
                or normalized["expert_group_count"] != 1
                or normalized["topk_group_count"] != 1
                or normalized["normalize_topk_probability"] is not True):
            raise ValueError(f"{source_model} route requires normalized biased sigmoid top{expected_topk} with one group")
        scale = normalized["routed_scaling_factor"]
        if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                or not math.isfinite(scale) or scale <= 0):
            raise ValueError(f"{source_model} route scale must be finite positive")
        if variant == "kimi_k3_moe_route" and scale != 1.0:
            raise ValueError("K3 source route requires its configured scale1")
        if variant in ("dsv32_moe_route", "kimi_moe_route") and normalized["mathematical_program"] not in (
                None, "biased_sigmoid_topk_flops_bytes_v1"):
            raise ValueError("grouped routing requires its declared mathematical program")
    elif kind is KernelKind.VECTOR and variant == "dsv4_moe_route":
        if (shape["experts"] not in (256, 384) or shape["topk"] != 6
                or shape["rows"] <= 0
                or shape["element_count"] != shape["rows"] * shape["experts"]):
            raise ValueError("V4 route requires full logits rows, 256/384 experts and top6")
        if normalized["expert_id_selection"] not in ("hash_table", "score_topk"):
            raise ValueError("V4 route requires hash_table or score_topk ID selection")
        source = normalized["routing_source"]
        if source not in (None, "vllm_0_30_triton_top6_pure_text",
                          "vllm_0_30_cuda_hash_top6_pure_text"):
            raise ValueError("unsupported V4 routing_source")
        if source == "vllm_0_30_triton_top6_pure_text" and normalized["expert_id_selection"] != "score_topk":
            raise ValueError("V4 Triton routing source excludes hash routing")
        if source == "vllm_0_30_cuda_hash_top6_pure_text" and normalized["expert_id_selection"] != "hash_table":
            raise ValueError("V4 CUDA hash routing source requires hash_table selection")
        scale = normalized["routed_scaling_factor"]
        if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                or not math.isfinite(scale) or scale <= 0):
            raise ValueError("V4 route scale must be finite positive")
    elif kind is KernelKind.VECTOR and variant == "dsv4_indexer_q_rope_quant":
        if (shape["head_dim"] != 128 or shape["rope_dim"] != 64
                or shape["element_count"] != shape["seq_len"] * shape["num_query_heads"] * shape["head_dim"]):
            raise ValueError("V4 indexer Q requires full 128-wide heads with 64 trailing rotary dimensions")
    elif kind is KernelKind.VECTOR and variant == "dsv4_inv_rope_fp8_quant":
        # DeepSeek-V4 fused inverse-RoPE + FP8 quantisation feeding the wo_a
        # einsum (deepseek_v4 common/ops/fused_inv_rope_fp8_quant.py): BF16
        # attention output in, e4m3 values plus one FP32 scale per
        # quant_group_size elements out.
        group_size = normalized["quant_group_size"]
        if (isinstance(group_size, bool) or not isinstance(group_size, int)
                or group_size <= 0):
            raise ValueError(
                "VECTOR/dsv4_inv_rope_fp8_quant quant_group_size must be a "
                "positive integer")
        if shape["head_dim"] % group_size:
            raise ValueError(
                "VECTOR/dsv4_inv_rope_fp8_quant quant_group_size must "
                "divide head_dim")
        if not 0 < shape["rope_dim"] <= shape["head_dim"] or shape["rope_dim"] % 2:
            raise ValueError(
                "VECTOR/dsv4_inv_rope_fp8_quant rope_dim must be positive "
                "even and no larger than head_dim")
        expected_elements = (
            shape["seq_len"] * shape["num_query_heads"] * shape["head_dim"])
        if shape["element_count"] != expected_elements:
            raise ValueError(
                "VECTOR/dsv4_inv_rope_fp8_quant element_count disagrees "
                "with head geometry")
    elif kind is KernelKind.VECTOR and variant == "rope":
        for name in ("is_neox", "key_present"):
            if not isinstance(normalized[name], bool):
                raise ValueError(f"VECTOR/rope {name} must be boolean")
    elif kind is KernelKind.VECTOR and variant in ("dsv32_fused_norm_rope", "dsv32_fused_q"):
        if not isinstance(normalized["has_indexer"], bool):
            raise TypeError("V32 target fusion has_indexer must be boolean")
        if normalized["index_rope_interleave"] is not True:
            raise ValueError("bound GLM V32 target fusion requires interleaved index RoPE")
        if variant == "dsv32_fused_norm_rope":
            if normalized["mla_cache_dtype"] != "bf16" or normalized["cache_slot_scope"] not in (
                    "all_rows_valid_slots_projection", "outputs_only_no_cache_slots", "no_metadata_no_cache_slots"):
                raise ValueError("V32 norm/cache fusion requires its explicit BF16 cache/metadata scope")
            if normalized["cache_slot_scope"] == "no_metadata_no_cache_slots":
                valid_fill = ((type(normalized["topk_active_fill"]) is int and normalized["topk_active_fill"] == -1)
                    if normalized["has_indexer"] else normalized["topk_active_fill"] is None)
                if (normalized["source_no_attention_metadata"] is not True
                        or normalized["index_k_output_present"] is not False or not valid_fill):
                    raise ValueError("V32 metadata=None requires absent cache/index-K output and its actual active TopK fill")
        elif normalized["quantize_mqa"] is not False:
            raise ValueError("V32 fused Q supports the explicit unquantized BF16 MQA path")
        if (shape["dtype_bytes"], shape["position_bytes"], shape["rope_cache_bytes"]) != (2, 8, 2):
            raise ValueError("V32 target fusion requires BF16 tensors/cache and int64 positions")
        if shape["rope_dim"] != 64 or shape["index_dim"] != 128:
            raise ValueError("V32 target fusion supports the bound GLM 64-rope/128-index geometry")
        if variant == "dsv32_fused_norm_rope":
            if shape["slot_bytes"] != 8 or shape["index_affine_bytes"] != 4:
                raise ValueError("V32 norm/cache fusion requires int64 slots and FP32 index affine")
        elif shape["index_weight_bytes"] != 2 or shape["q_scale_bytes"] != 4:
            raise ValueError("V32 fused Q requires BF16 index input weights and FP32 Q scale")
    elif kind is KernelKind.VECTOR and variant == "dsv32_embed_norm":
        if shape["dtype_bytes"] != 2 or shape["input_id_bytes"] != 4:
            raise ValueError("V32 target embedding norm requires BF16 and INT32 ids")
    elif kind is KernelKind.VECTOR and variant in ("dsv32_eh_norm", "dsv32_embed_eh_norm"):
        if shape["dtype_bytes"] != 2 or shape["position_bytes"] != 8:
            raise ValueError("V32 MTP input norm requires BF16 and int64 positions")
        if variant == "dsv32_embed_eh_norm" and shape["input_id_bytes"] != 4:
            raise ValueError("V32 fused embedding requires int32 token ids")
    elif kind is KernelKind.VECTOR and variant in (
            "mhc_pre", "mhc_post_pre", "mhc_prenorm", "mhc_prenorm_broadcast", "mhc_post_prenorm",
            "mhc_pre_norm", "mhc_pre_norm_broadcast", "mhc_post", "hc_head"):
        # DeepSeek-V4 hyper-connection streams: one (tokens, hc_mult, hidden)
        # BF16 state per layer (deepseek_v4/nvidia/model.py:1131-1353).
        stream = shape["hc_mult"] * shape["hidden"]
        if shape["dtype_bytes"] != 2:
            raise ValueError("mHC requires BF16 residuals")
        if shape["element_count"] % stream:
            raise ValueError(
                f"VECTOR/{variant} element_count must be an integral "
                "number of (hc_mult, hidden) stream elements")
    elif kind is KernelKind.VECTOR and variant in (
            "elementwise", "elementwise_pytorch_vec4"):
        for name in ("reads", "writes"):
            if isinstance(normalized[name], bool) or not isinstance(
                    normalized[name], int):
                raise TypeError(f"VECTOR {name} must be an integer")
        if normalized["reads"] < 0 or normalized["writes"] < 0:
            raise ValueError("VECTOR reads/writes cannot be negative")
        operation = normalized["operation"]
        if not isinstance(operation, str) or not operation.strip():
            raise TypeError(
                "VECTOR/elementwise operation must be a non-empty string")
        operation = operation.strip().lower()
        expected_io = {
            "add": (2, 1),
            "mul": (2, 1),
            "div": (2, 1),
            # One tensor operand and a Python scalar (x + c, x * c, cumsum).
            "unary": (1, 1),
            # BF16 logits materialized as FP32 (sglang logits_processor).
            "cast_bf16_float32": (1, 1),
            "cast_int32_int64": (1, 1),
            "copy": (1, 1),
            "fill": (0, 1),
            "read": (1, 0),
            "notequal_int32_bool": (1, 1),
            "cast_bool_bf16": (1, 1),
        }
        if operation not in expected_io:
            raise ValueError(
                "VECTOR/elementwise operation must be one of "
                f"{sorted(expected_io)}, got {operation!r}")
        expected_reads, expected_writes = expected_io[operation]
        if (normalized["reads"], normalized["writes"]) != (
                expected_reads, expected_writes):
            raise ValueError(
                "VECTOR/elementwise reads/writes disagree with operation "
                f"{operation!r}: expected="
                f"({expected_reads}, {expected_writes}), actual="
                f"({normalized['reads']}, {normalized['writes']})")
        normalized["operation"] = operation
        if operation == "cast_int32_int64":
            if (compute_dtype != "int32" or "tensor_iterator_source" not in normalized
                    or normalized["tensor_iterator_source"] not in ("pytorch_2_8_sm80", "pytorch_2_8_sm90", "aten_advanced_indexing_long", "pytorch_2_4_maca_3_7_2")):
                raise ValueError("INT32-to-INT64 cast requires its explicit Torch source or logical ATen source")
            if "copy_dispatch" in normalized:
                if normalized["tensor_iterator_source"] != "pytorch_2_4_maca_3_7_2":
                    raise ValueError("integer Copy dispatch requires its C500 Torch source")
                from src.kernel.launch_config.metax_c500.vector_launch import pytorch_int32_int64_cast_launch
                pytorch_int32_int64_cast_launch(shape["element_count"], normalized, variant)
                normalized["copy_dispatch"] = MappingProxyType(dict(normalized["copy_dispatch"]))
        elif "tensor_iterator_source" in normalized:
            raise ValueError("tensor_iterator_source is only supported for the bound INT32-to-INT64 cast")
        elif "copy_dispatch" in normalized:
            raise ValueError("integer Copy dispatch is only supported for INT32-to-INT64 cast")
    elif kind is KernelKind.VECTOR and variant in (
            "moe_silu", "moe_mul", "moe_sum", "moe_reduce"):
        work_fraction = normalized["work_fraction"]
        if (isinstance(work_fraction, bool)
                or not isinstance(work_fraction, (int, float))):
            raise TypeError(
                f"VECTOR/{variant} work_fraction must be numeric")
        work_fraction = float(work_fraction)
        if not math.isfinite(work_fraction) or not 0 < work_fraction <= 1:
            raise ValueError(
                f"VECTOR/{variant} work_fraction must be finite and in "
                "(0, 1]")
        normalized["work_fraction"] = work_fraction
    elif kind is KernelKind.VECTOR and variant == "moe_fill":
        fill_role = normalized["fill_role"]
        if not isinstance(fill_role, str) or not fill_role.strip():
            raise TypeError("VECTOR/moe_fill fill_role must be a non-empty string")
        fill_role = fill_role.strip().lower()
        valid_fill_roles = set(MOE_FILL_ROLES)
        if fill_role not in valid_fill_roles:
            raise ValueError(
                "VECTOR/moe_fill fill_role must be one of "
                f"{sorted(valid_fill_roles)}, got {fill_role!r}")
        normalized["fill_role"] = fill_role
    elif kind is KernelKind.VECTOR and variant == "moe_cast":
        cast_role = normalized["cast_role"]
        if not isinstance(cast_role, str) or not cast_role.strip():
            raise TypeError(
                "VECTOR/moe_cast cast_role must be a non-empty string")
        cast_role = cast_role.strip().lower()
        valid_cast_roles = set(MOE_CAST_ROLES)
        if cast_role not in valid_cast_roles:
            raise ValueError(
                "VECTOR/moe_cast cast_role must be one of "
                f"{sorted(valid_cast_roles)}, got {cast_role!r}")
        normalized["cast_role"] = cast_role
    elif kind is KernelKind.VECTOR and variant == "reduce":
        if (isinstance(normalized["reduction_factor"], bool)
                or not isinstance(normalized["reduction_factor"], int)):
            raise TypeError("VECTOR reduction_factor must be an integer")
        if normalized["reduction_factor"] <= 0:
            raise ValueError("VECTOR reduction_factor must be positive")
    elif kind is KernelKind.VECTOR and variant == "gather":
        for name in ("gather_factor", "index_count", "index_bytes"):
            if isinstance(normalized[name], bool) or not isinstance(
                    normalized[name], int):
                raise TypeError(f"VECTOR {name} must be an integer")
        for name in ("gather_factor", "index_bytes"):
            if normalized[name] <= 0:
                raise ValueError(f"VECTOR {name} must be positive")
        if normalized["index_count"] < 0:
            raise ValueError("VECTOR index_count cannot be negative")
        if not isinstance(normalized["scattered"], bool):
            raise ValueError("VECTOR scattered must be boolean")
        if not isinstance(normalized["materialized_output"], bool):
            raise TypeError("VECTOR gather materialized_output must be boolean")
        if "materialization_source" in normalized:
            source = normalized["materialization_source"]
            if (source not in ("pytorch_2_4_maca_cat_last_dim", "pytorch_2_4_maca_index_select")
                    or compute_dtype != "bf16" or normalized["gather_factor"] != 1
                    or normalized["index_bytes"] != 8 or not normalized["materialized_output"]
                    or "index_select_source" in normalized or "mathematical_program" in normalized
                    or not {"rows", "hidden"}.issubset(shape)
                    or shape["element_count"] != shape["rows"] * shape["hidden"]):
                raise ValueError("Maca materialized gather requires its typed BF16 source")
            if source == "pytorch_2_4_maca_cat_last_dim":
                if normalized["scattered"] or normalized["index_count"] != 0:
                    raise ValueError("Maca cat requires no indices")
            elif (not normalized["scattered"] or normalized["index_count"] != shape["rows"]
                    or "source_rows" not in shape):
                raise ValueError("Maca index_select requires its contiguous source extent and INT64 indices")
        if "index_select_source" in normalized:
            if (normalized["index_select_source"] != "vllm_0_11_embedding" or compute_dtype != "bf16"
                    or normalized["gather_factor"] != 1 or normalized["index_bytes"] != 8
                    or normalized["materialized_output"] is not True
                    or not {"rows", "hidden", "source_rows"}.issubset(shape)
                    or normalized["index_count"] != shape["rows"]
                    or shape["element_count"] != shape["rows"] * shape["hidden"]
                    or "mathematical_program" in normalized):
                raise ValueError("embedding index_select requires its typed BF16/INT64 materialization")
        if "mathematical_program" in normalized:
            if normalized["mathematical_program"] == "materialized_bf16_gather_bytes_v1":
                if (compute_dtype != "bf16" or normalized["gather_factor"] != 1
                        or not normalized["materialized_output"] or normalized["index_bytes"] != 4):
                    raise ValueError("logical BF16 gather requires materialized rows with explicit INT32 indices")
            elif normalized["mathematical_program"] == "materialized_fp32_copy_bytes_v1":
                if (compute_dtype != "fp32" or normalized["gather_factor"] != 1
                        or normalized["index_count"] != 0 or normalized["scattered"]
                        or not normalized["materialized_output"]):
                    raise ValueError("logical FP32 copy requires an unindexed materialized one-to-one copy")
            elif (normalized["mathematical_program"] not in (
                    "dense_fp16_source_flops_bytes_v1", "dense_bf16_source_flops_bytes_v1")
                    or compute_dtype != {"dense_fp16_source_flops_bytes_v1": "fp16",
                        "dense_bf16_source_flops_bytes_v1": "bf16"}[normalized["mathematical_program"]]
                    or normalized["gather_factor"] != 1
                    or normalized["index_bytes"] != 8 or not normalized["materialized_output"]
                    or "rows" not in shape
                    or normalized["index_count"] not in (0, shape["rows"])):
                raise ValueError("dense FP16 gather requires one-to-one materialized source rows and explicit INT64 indices")
    elif kind is KernelKind.VECTOR and variant == "fp8_group_quant":
        group_size = normalized["group_size"]
        if isinstance(group_size, bool) or not isinstance(group_size, int):
            raise TypeError("VECTOR/fp8_group_quant group_size must be an integer")
        if group_size <= 0 or shape["element_count"] % group_size:
            raise ValueError(
                "VECTOR/fp8_group_quant needs whole quantisation groups")
        source_program = normalized["source_program"]
        if source_program not in (None, "vllm_0_30_group_fp8_fp32_scales"):
            raise ValueError("unknown FP8 group quantization source program")
        if source_program is not None:
            if (group_size != 128 or compute_dtype != "bf16" or "rows" not in shape or "columns" not in shape
                    or shape["columns"] % 128 or shape["rows"] * shape["columns"] != shape["element_count"]):
                raise ValueError("vLLM 0.30 FP8 source requires BF16 full row-major 128-value groups")
    elif kind is KernelKind.VECTOR and variant == "gemm_reduce":
        split_k = normalized["split_k"]
        if isinstance(split_k, bool) or not isinstance(split_k, int):
            raise TypeError("VECTOR/gemm_reduce split_k must be an integer")
        if split_k <= 1:
            raise ValueError("VECTOR/gemm_reduce split_k must exceed one")
    return normalized


def _validate_compute_dtype_contract(
    kind: KernelKind,
    variant: str,
    dtype: str,
    shape: Mapping[str, int],
    attributes: Mapping[str, Any],
) -> None:
    if variant == "projected_attention_path":
        # This source program owns mixed operand dtypes in its stage ledger;
        # indexer contracts retain INT32 result type, other stages BF16.
        expected = "int32" if kind is KernelKind.SPARSE_ATTN_INDEXER else "bf16"
        if dtype != expected:
            raise ValueError("projected Attention dtype disagrees with its stage kind")
        return
    if kind is KernelKind.GEMM and variant == "kimi_k3_nope_bmm":
        if dtype != "bf16":
            raise ValueError("K3 NoPE BMM requires BF16 input, weight and output storage")
    elif kind is KernelKind.VECTOR and variant == "dsv4_short_context_topk":
        if dtype != "int32":
            raise ValueError("V4 short-context fill requires INT32 output")
    elif kind is KernelKind.VECTOR and variant == "vllm_plain_greedy_sample":
        expected = "fp32" if attributes["stage"] == "argmax" else "int64"
        if dtype != expected:
            raise ValueError(f"plain greedy sampling {attributes['stage']} requires {expected} input storage")
    elif kind is KernelKind.VECTOR and variant == "vllm_embedding_id_widen":
        if dtype != "int32":
            raise ValueError("embedding ID widen requires INT32 input and INT64 output")
    elif kind is KernelKind.VECTOR and variant == "vllm_masked_embedding":
        allowed = ("int32",) if attributes["stage"] == "mask_ids" else ("bf16", "fp16")
        if dtype not in allowed:
            raise ValueError("masked embedding requires INT32 IDs or BF16/FP16 output")
    elif kind is KernelKind.VECTOR and variant in ("dsv4_moe_route", "dsv32_moe_route", "kimi_k3_moe_route"):
        if dtype != "fp32":
            raise ValueError("K3 route requires FP32 logits" if variant == "kimi_k3_moe_route"
                             else "V4 route requires FP32 logits")
    elif kind is KernelKind.VECTOR and variant in ("dsv32_draft_id_cast", "dsv32_draft_token_stack"):
        if dtype != "int64":
            raise ValueError("draft ID cast/stack requires INT64 input")
    elif kind is KernelKind.VECTOR and variant in (
            "dsv32_draft_id_scatter", "dsv32_draft_prepare_next_ids", "dsv32_draft_prepare_sample_indices", "dsv32_draft_seq_rollback"):
        if dtype != "int32":
            raise ValueError(f"VECTOR/{variant} requires INT32 input and output")
    elif kind is KernelKind.VECTOR and variant == "dsv32_draft_step_metadata":
        if dtype != "int64":
            raise ValueError("draft step metadata requires INT64 position storage")
    elif kind is KernelKind.VECTOR and variant in ("dsv32_draft_row_select", "dsv32_draft_buffer_copy"):
        expected = (("int64" if attributes["selection_role"] == "positions" else
                     "int32" if attributes["selection_role"] == "topk_indices" else "bf16")
                    if variant == "dsv32_draft_row_select" else
                    ("int32" if attributes["copy_role"] in ("input_ids", "topk_indices", "target_token_ids_shift") else
                     "int64" if attributes["copy_role"] in ("positions", "slot_mapping") else "bf16"))
        if dtype != expected:
            raise ValueError(f"draft source role requires {expected} storage")
    elif kind is KernelKind.VECTOR and variant == "vllm_attention_state_merge":
        if dtype != "bf16":
            raise ValueError("attention state merge requires BF16 partial outputs and FP32 LSE")
    elif kind is KernelKind.VECTOR and variant == "kimi_moe_route":
        if dtype != "bf16":
            raise ValueError("Kimi source routing requires BF16 logits")
    elif kind is KernelKind.VECTOR and variant == "vllm_rejection_logits_materialize":
        expected = "fp32" if attributes["materialization_stage"] == "target_clone" else "bf16"
        if dtype != expected:
            raise ValueError(f"rejection materialization requires {expected} input storage")
    elif kind is KernelKind.VECTOR and variant in ("kimi_k3_nope_prefill", "kimi_k3_nope_decode", "kimi_k3_nope_mixed", "kimi_k3_nope_context"):
        expected_dtype = ("fp32" if variant == "kimi_k3_nope_context" and attributes["stage"] == "context_lse_copy" else "bf16")
        if dtype != expected_dtype:
            raise ValueError("K3 NoPE prefill requires BF16 activation/cache storage")
    elif kind is KernelKind.VECTOR and variant in ("kimi_k3_kda_prefill", "kimi_k3_kda_mixed", "kimi_k3_kda_context_prefill"):
        if dtype != "bf16":
            raise ValueError("K3 KDA prefill requires BF16 activations/conv state and FP32 recurrent state")
    elif kind is KernelKind.VECTOR and variant == "kimi_k3_kda_decode":
        if dtype != "bf16":
            raise ValueError("K3 KDA decode requires BF16 activations/conv state and FP32 recurrent state")
    elif kind is KernelKind.VECTOR and variant == "kimi_k3_situ":
        if dtype != "bf16":
            raise ValueError("K3 SiTU requires BF16 storage and FP32 arithmetic")
    elif kind is KernelKind.VECTOR and variant == "kimi_k3_attn_res":
        if dtype != "bf16":
            raise ValueError("K3 attention residual requires BF16 storage and FP32 arithmetic")
    elif kind is KernelKind.VECTOR and variant == "vllm_logits_gather_reorder":
        if dtype not in ("bf16", "fp16"):
            raise ValueError("logits gather reorder requires BF16 or FP16 input and output")
    elif kind is KernelKind.VECTOR and variant == "vllm_logits_fp32_cast":
        expected = "fp16" if attributes["sampling_source"] in (
            "vllm_0_30_plain_sampler_fp16", "vllm_0_11_plain_sampler_fp16") else "bf16"
        if dtype != expected:
            raise ValueError(f"target logits cast requires {expected.upper()} input and FP32 output")
    elif kind is KernelKind.VECTOR and variant in ("vocab_parallel_embedding", "vllm_logits_row_select"):
        if dtype not in ("bf16", "fp16"):
            raise ValueError(f"VECTOR/{variant} requires BF16 or FP16 dtype")
    elif kind is KernelKind.VECTOR and variant == "dsv4_indexer_q_rope_quant":
        if dtype != "bf16":
            raise ValueError("V4 indexer Q requires BF16 input and separate FP8 output")
    elif kind is KernelKind.VECTOR and variant == "dsv4_inv_rope_fp8_quant":
        if dtype != "fp8":
            raise ValueError("V4 inverse RoPE quantization requires FP8 output storage")
    elif kind is KernelKind.VECTOR and variant in (
            "dsv32_fused_norm_rope", "dsv32_fused_q", "dsv4_q_kv_rmsnorm", "dsv32_eh_norm", "dsv32_embed_eh_norm", "dsv32_embed_norm", "dsv32_draft_argmax", "mhc_prenorm", "mhc_prenorm_broadcast",
            "mhc_post_prenorm", "mhc_pre_norm", "mhc_pre_norm_broadcast",
            "mhc_post", "hc_head"):
        if dtype != "bf16":
            raise ValueError(f"VECTOR/{variant} requires BF16 dtype")
    elif kind is KernelKind.SPARSE_ATTN_INDEXER:
        if dtype != "int32":
            raise ValueError("SPARSE_ATTN_INDEXER requires int32 output dtype")
        if (variant != "dsa_index_reuse"
                and shape["topk"] > attributes["topk_capacity"]):
            raise ValueError(
                "SPARSE_ATTN_INDEXER topk cannot exceed topk_capacity")
    elif kind is KernelKind.SPARSE_ATTN:
        if dtype not in ("bf16", "fp16"):
            raise ValueError("SPARSE_ATTN requires BF16 or FP16 compute dtype")
    elif kind is KernelKind.KV_CACHE:
        if variant == "dsa_fp8_index_cache":
            expected_dtype = "uint8"
        elif variant == "dsv4_compressed_state_cache":
            expected_dtype = "fp32" if attributes["finish_stage"] == "compress" else "uint8"
        elif variant == "dsv4_compressor_state_cache":
            expected_dtype = "fp32"
        elif variant in ("mla_kv_cache", "dsv4_qnorm_rope_kv_insert"):
            expected_dtype = "bf16"
        elif variant == "paged_kv_cache":
            if dtype not in ("bf16", "fp16"):
                raise ValueError(
                    "KV_CACHE/paged_kv_cache requires BF16 or FP16 "
                    "dtype")
            expected_dtype = dtype
        else:
            raise AssertionError(
                f"unhandled KV_CACHE variant: {variant!r}")
        if dtype != expected_dtype:
            raise ValueError(
                f"KV_CACHE/{variant} requires {expected_dtype} dtype")
    elif kind is KernelKind.FLASH_ATTN and variant in (
            "mla_prefill", "mla_decode", "mla_absorbed_prefill", "kimi_k3_nope_context_attention"):
        if dtype not in ("bf16", "fp16"):
            raise ValueError("MLA attention requires BF16 or FP16 dtype")
    elif kind is KernelKind.FUSED_MOE and variant == "kimi_k3_marlin_experts":
        if dtype != "bf16":
            raise ValueError("K3 Marlin expert source requires BF16 activations and MXFP4 weights")
        if any(count > shape["tokens"] for count in attributes["tokens_per_expert"]):
            raise ValueError("K3 Marlin top-k selects distinct expert IDs per token")
    elif kind is KernelKind.FUSED_MOE and variant == "routed_experts":
        if dtype not in ("bf16", "fp16"):
            raise ValueError("routed experts require BF16 or FP16 activations")
        routed_dtype = attributes["routed_weight_dtype"]
        if routed_dtype == "int4":
            group_size = attributes["weight_group_size"]
            for dimension in (shape["hidden"], shape["intermediate"]):
                if dimension % group_size:
                    raise ValueError(
                        "INT4 routed-expert group size must divide hidden and "
                        "intermediate dimensions")
        elif routed_dtype == "mxfp4":
            # DeepSeek-V4 experts: E2M1 values with one UE8M0 scale per 32
            # elements along K, Marlin-packed (vLLM 0.30
            # fused_moe/experts/marlin_moe.py fused_marlin_moe).
            if dtype != "bf16":
                raise ValueError("MXFP4 routed experts require BF16 activations")
            group_size = attributes["weight_group_size"]
            if group_size != 32:
                raise ValueError("MXFP4 routed experts require group size 32")
            for dimension in (shape["hidden"], shape["intermediate"]):
                if dimension % group_size:
                    raise ValueError(
                        "MXFP4 group size must divide hidden and intermediate")


def _normalize_launch_parameters(
        kind: KernelKind,
        shape: Mapping[str, int],
        launch: LaunchConfig,
        attributes: Mapping[str, Any] | None = None,
) -> LaunchConfig:
    graph_padding = (
        attributes is not None
        and "graph_logical_tokens" in attributes
        and attributes["graph_logical_tokens"]
        != attributes["graph_model_input_tokens"])
    parameters = dict(launch.parameters)
    if _is_kda_chunk_source(kind, launch.execution_variant, attributes):
        # Exact replay already checked all parameters, grid/block and options.
        return launch
    if (launch.source_program is not None
            and kind not in (
                KernelKind.GEMM, KernelKind.FLASH_ATTN, KernelKind.VECTOR)):
        raise ValueError(
            "source_program is supported only for GEMM, FLASH_ATTN or "
            "VECTOR launches")
    if (kind is KernelKind.VECTOR
            and launch.source_program is not None):
        raise TypeError(
            "VECTOR launches carry no source program: the vLLM-Ascend "
            "QKNormRope fusion is compiled into one split_qkv_rmsnorm_rope "
            "kernel for both the replayed-graph and fallback steps")
    if kind is KernelKind.GEMM:
        if (launch.source_program is not None
                and not isinstance(
                    launch.source_program,
                    (CannGemmLaunchProgram, CannGemmTilePipelineProgram,
                     GpuCtaRasterProgram))):
            raise TypeError(
                "GEMM source_program must be a typed CANN tile program or "
                "GPU CTA raster program")
        required = {
            "compulsory_traffic", "weight_tile_rows", "mte2_tile_bytes",
            "aiv_mixed", "n_major_traffic",
            "split_k_partial_bytes_per_element",
            "split_k_partial_dram_bytes_per_element",
        }
        if "operand_transport_tma" in parameters:
            if parameters["operand_transport_tma"] != 1.0:
                raise ValueError("operand_transport_tma must declare a TMA source")
            required.add("operand_transport_tma")
        if "uniform_split_k_tiles" in parameters:
            if (parameters["uniform_split_k_tiles"] != 1.0
                    or "operand_transport_tma" not in parameters
                    or launch.tiling is None or launch.tiling.split_k is None
                    or launch.tiling.pipeline_depth is None
                    or launch.tiling.split_k <= 1
                    or launch.execution_variant not in ("gemm", "batched_gemm", "moe_router")
                    or shape["k"] % launch.tiling.tile_shape["k"]):
                raise ValueError("uniform split-K tiles require a split TMA K-tile loop")
            required.add("uniform_split_k_tiles")
        instruction_fields = {"tensor_instruction_m", "tensor_instruction_n", "tensor_instruction_k"}
        if instruction_fields & set(parameters):
            if not instruction_fields.issubset(parameters):
                raise ValueError("GEMM Tensor instruction geometry requires all three axes")
            if "operand_transport_tma" not in parameters or launch.tiling is None:
                raise ValueError("GEMM Tensor instruction geometry requires TMA tiling")
            instruction = tuple(parameters[f"tensor_instruction_{axis}"] for axis in ("m", "n", "k"))
            if any(value <= 0 or not value.is_integer() for value in instruction):
                raise ValueError("GEMM Tensor instruction axes must be positive integers")
            required |= instruction_fields
        cluster_fields = {"cluster_output_m_tiles", "cluster_output_n_tiles"}
        if cluster_fields & set(parameters):
            if not cluster_fields.issubset(parameters):
                raise ValueError("GEMM cluster output geometry requires both axes")
            if "operand_transport_tma" not in parameters or launch.cluster is None:
                raise ValueError("cluster operand reuse requires declared TMA transport and cluster")
            if launch.tiling is None:
                raise ValueError("cluster operand reuse requires output tiling")
            cluster_axes = tuple(parameters[name] for name in sorted(cluster_fields))
            if any(value <= 0 or not value.is_integer() for value in cluster_axes):
                raise ValueError("GEMM cluster output axes must be positive integers")
            if math.prod(cluster_axes) != math.prod(launch.cluster):
                raise ValueError("GEMM cluster output geometry does not conserve cluster CTAs")
            if any(grid % cluster for grid, cluster in zip(launch.grid, launch.cluster)):
                raise ValueError("GEMM physical grid must comprise complete clusters")
            required |= cluster_fields
        if (isinstance(launch.source_program, GpuCtaRasterProgram)
                and launch.source_program.is_cluster_grouped):
            if (not cluster_fields.issubset(parameters)
                    or "operand_transport_tma" not in parameters
                    or launch.tiling is None
                    or launch.tiling.split_k != 1
                    or launch.source_program.k_slices != 1):
                raise ValueError(
                    "cluster output raster requires a non-split TMA cluster launch")
        if (launch.execution_variant == "moe_router"
                and "split_k_count_observed" in launch.parameters):
            # An observed-grid launch whose split count the record omits.
            required.add("split_k_count_observed")
        if launch.algorithm == "h200_vllm030_torch213_fp16_observed_gemm":
            required |= {"registers_per_thread", "dynamic_shared_bytes"}
        if launch.algorithm == "h200_vllm030_torch213_fp16_external_aggregate_gemm":
            required |= {"registers_per_thread", "dynamic_shared_bytes", "modeled_split_k",
                "modeled_partial_bytes_per_element", "split_k_count_observed"}
        if launch.algorithm == "h200_cublaslt129_bf16_external_aggregate_gemm":
            required |= {"modeled_split_k", "modeled_partial_bytes_per_element", "split_k_count_observed"}
    elif kind is KernelKind.FLASH_ATTN:
        if (launch.source_program is not None
                and not isinstance(
                    launch.source_program, FiaCrossDomainLaunchProgram)):
            raise TypeError(
                "FLASH_ATTN source_program must be a typed FIA program")
        base_required = {
            "unpadded_flops", "mma_smem_bytes_per_flop",
            "split_scratch_dram_bytes_per_element",
            "page_table_bytes_per_kv_panel",
        }
        source_schedule_required = base_required | {
            "cann_source_m_base_size",
            "cann_source_s2_base_size",
            "cann_source_task_count",
            "cann_source_tensor_logical_operations",
            "cann_source_tensor_shape_efficiency",
            "cann_source_tensor_schedule_efficiency",
            "cann_source_vector_operations",
            "cann_source_vector_main_operations",
            "cann_source_vector_initialization_operations",
            "cann_source_vector_terminal_operations",
            "cann_source_vector_vec1_operations",
            "cann_source_vector_vec2_operations",
            "cann_source_vector_worker_count",
            "cann_source_vector_schedule_efficiency",
            "cann_source_vec1_read_bytes",
            "cann_source_mm2_score_read_bytes",
            "cann_source_vec2_mm2_read_bytes",
            "cann_source_vec2_state_read_bytes",
            "cann_source_flash_decode_terminal_read_bytes",
            "cann_source_vec1_write_bytes",
            "cann_source_nupdate_write_bytes",
            "cann_source_vec2_write_bytes",
            "cann_source_flash_decode_terminal_write_bytes",
            "cann_source_output_init_bytes",
            "cann_source_external_write_bytes",
            "cann_source_vec1_unique_write_bytes",
            "cann_source_vec2_unique_write_bytes",
            "cann_source_aiv_unique_write_bytes",
            "cann_source_mm1_unique_write_bytes",
            "cann_source_mm2_unique_write_bytes",
            "cann_source_aic_unique_write_bytes",
        }
        if set(parameters) == base_required:
            required = base_required
        elif "cann_source_vector_worker_count" in parameters:
            worker_count_value = parameters[
                "cann_source_vector_worker_count"]
            if (not worker_count_value.is_integer()
                    or worker_count_value <= 0):
                raise ValueError(
                    "FLASH_ATTN source Vector worker count must be a positive "
                    "integer")
            worker_count = int(worker_count_value)
            worker_fields = {
                "generic_operations", "exp_operations", "div_operations",
                "exp_invocations", "div_invocations",
            }
            worker_required = {
                f"cann_source_vector_worker_{worker}_{field}"
                for worker in range(worker_count)
                for field in worker_fields
            }
            required = source_schedule_required | worker_required
        else:
            required = source_schedule_required
        if set(parameters) != required:
            raise ValueError(
                "FLASH_ATTN launch parameter schema mismatch: "
                f"actual={sorted(parameters)}")
    elif kind is KernelKind.FUSED_MOE:
        tiling = launch.tiling
        if tiling is None or not {"m", "n"}.issubset(tiling.tile_shape):
            raise ValueError("FUSED_MOE launch requires M and N tiles")
        required = {
            "group_size_m", "padded_rows", "mte2_tile_bytes",
            "weight_tile_rows", "operand_replay_a", "operand_replay_b",
        }
        cann_grouped_k_program = {
            "cann_depth_a1", "cann_depth_b1",
            "cann_step_ka", "cann_step_kb",
            "cann_step_m", "cann_step_n",
            "cann_db_l0a", "cann_db_l0b", "cann_db_l0c",
        }
        present_cann_program = set(parameters) & cann_grouped_k_program
        if present_cann_program:
            missing_cann_program = (
                cann_grouped_k_program - present_cann_program)
            if missing_cann_program:
                raise ValueError(
                    "FUSED_MOE CANN GroupedMatmul K program is incomplete: "
                    f"missing={sorted(missing_cann_program)}")
            required.update(cann_grouped_k_program)
        if "expert_parallel_size" in parameters:
            launch_ep = parameters["expert_parallel_size"]
            if (not launch_ep.is_integer() or launch_ep <= 1):
                raise ValueError(
                    "FUSED_MOE launch expert_parallel_size must be an "
                    "integer greater than one")
            required.add("expert_parallel_size")
        if "dynamic_shared_memory_bytes" in parameters:
            dynamic_shared_memory = parameters["dynamic_shared_memory_bytes"]
            if (not dynamic_shared_memory.is_integer()
                    or dynamic_shared_memory <= 0):
                raise ValueError(
                    "FUSED_MOE dynamic_shared_memory_bytes must be a "
                    "positive integer")
            required.add("dynamic_shared_memory_bytes")
        occupancy_parameters = {"registers_per_thread", "static_shared_bytes"}
        present_occupancy = set(parameters) & occupancy_parameters
        if present_occupancy:
            if (present_occupancy != occupancy_parameters
                    or "dynamic_shared_memory_bytes" not in parameters):
                raise ValueError("FUSED_MOE occupancy launch facts are incomplete")
            for name in occupancy_parameters:
                value = parameters[name]
                if (not value.is_integer() or value < 0
                        or (name == "registers_per_thread" and value == 0)):
                    raise ValueError(f"FUSED_MOE {name} must be a valid integer")
            required.update(occupancy_parameters)
        compiled_program_parameters = {"num_warps", "num_stages"}
        present_compiled_program = (
            set(parameters) & compiled_program_parameters)
        if present_compiled_program:
            # Two deployed configurations can share tile shape, block threads
            # and dynamic shared memory yet compile to different programs, so
            # these two source/compiler facts are recorded together or not at
            # all.
            if present_compiled_program != compiled_program_parameters:
                raise ValueError(
                    "FUSED_MOE compiled-program launch facts are incomplete: "
                    f"missing={sorted(compiled_program_parameters - present_compiled_program)}")
            for name in sorted(compiled_program_parameters):
                value = parameters[name]
                if not value.is_integer() or value <= 0:
                    raise ValueError(
                        f"FUSED_MOE {name} must be a positive integer")
            required.update(compiled_program_parameters)
        routed_weight_program_parameter = (
            SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER)
        if routed_weight_program_parameter in parameters:
            routed_weight_program = parameters[routed_weight_program_parameter]
            if (not routed_weight_program.is_integer()
                    or int(routed_weight_program) not in (0, 1)):
                raise ValueError(
                    "FUSED_MOE source W2 routed-weight program must be "
                    "zero or one")
            required.add(routed_weight_program_parameter)
    elif kind in (
            KernelKind.EAGER_ATTN, KernelKind.UNFUSED_MOE):
        required = set()
    elif kind in (
            KernelKind.SPARSE_ATTN_INDEXER, KernelKind.SPARSE_ATTN):
        required = set()
    elif kind is KernelKind.KV_CACHE:
        if launch.execution_variant == "paged_kv_cache":
            allowed = {
                "kv_cache_slot_l1_bytes", "lowered_sfu_operations",
                SOURCE_VECTORIZED_KV_COPY_PARAMETER,
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
                "source_bound_resident_ctas_per_unit",
                SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER,
                SOURCE_SCALAR_KV_LOOP_PARAMETER,
                SOURCE_PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_PARAMETER,
                "cann_tiling_key",
            }
            if launch.algorithm == "vllm_0_11_sm90_source_kv_cache":
                allowed.update({"registers_per_thread", "static_shared_bytes", "dynamic_shared_bytes"})
            unknown = set(parameters) - allowed
            if unknown:
                raise ValueError(
                    "KV_CACHE/paged_kv_cache launch parameter schema "
                    f"mismatch: unknown={sorted(unknown)}")
            required = set(parameters)
        elif launch.execution_variant in (
                "mla_kv_cache", "dsa_fp8_index_cache",
                "dsv4_compressed_state_cache",
                "dsv4_compressor_state_cache",
                "dsv4_qnorm_rope_kv_insert"):
            finish_launches = {
                "vllm_dsv4_c4_norm_rope_store_cutedsl": (4, 584, None, 1, 64),
                "vllm_dsv4_c4_indexer_norm_rope_store_triton": (4, 132, None, 1, 32),
                "vllm_dsv4_c128_compress_block8_cutedsl": (128, 584, "compress", 8, 256),
                "vllm_dsv4_c128_norm_rope_store_cutedsl": (128, 584, "store", 1, 256),
            }
            if launch.algorithm in finish_launches:
                ratio, record_bytes, stage, grid_multiplier, threads = finish_launches[launch.algorithm]
                if launch.execution_variant != "dsv4_compressed_state_cache":
                    raise ValueError("DSV4 finish launch requires the compressed-state variant")
                if (shape["compress_ratio"] != ratio or shape["state_bytes"] != record_bytes
                        or attributes["finish_stage"] != stage
                        or launch.block != (threads, 1, 1)
                        or (not graph_padding
                            and launch.grid != (shape["seq_len"] * grid_multiplier, 1, 1))):
                    raise ValueError("DSV4 finish launch disagrees with its source stage/storage")
            if launch.algorithm == "vllm_dsv4_c128_two_stage_aggregate_projection":
                required = {"dsv4_finish_store_grid", "dsv4_finish_store_block"}
                if launch.execution_variant != "dsv4_compressed_state_cache":
                    raise ValueError("DSV4 aggregate launch requires the compressed-state variant")
                if (attributes["finish_stage"] is not None
                        or shape["compress_ratio"] != 128
                        or shape["state_bytes"] != 584
                        or launch.grid != (shape["seq_len"] * 8, 1, 1)
                        or launch.block != (256, 1, 1)):
                    raise ValueError("DSV4 C128 aggregate launch must declare both source stages")
                if set(parameters) != required:
                    raise ValueError("DSV4 C128 aggregate store launch fields are incomplete")
                if (parameters["dsv4_finish_store_grid"] != shape["seq_len"]
                        or parameters["dsv4_finish_store_block"] != 256):
                    raise ValueError("DSV4 C128 aggregate store launch disagrees with source")
            else:
                required = set()
        else:
            raise AssertionError(
                "unhandled KV_CACHE launch variant: "
                f"{launch.execution_variant!r}")
    elif kind is KernelKind.VECTOR:
        variant = launch.execution_variant
        if variant == "dsv4_short_context_topk":
            if (launch.algorithm != "vllm_dsv4_short_context_topk_triton"
                    or launch.grid != (shape["seq_len"], 1, 1)
                    or launch.block != (256, 1, 1) or parameters):
                raise ValueError("V4 short-context fill launch disagrees with source")
        vector_schema_cache_key = None
        if variant in ("rmsnorm", "add_rmsnorm"):
            vector_schema_cache_key = (
                variant,
                (shape["hidden"] == 128 or shape["hidden"] % 256 == 0)
                if variant == "rmsnorm" else None,
                frozenset(parameters),
                launch.algorithm,
            )
            if vector_schema_cache_key in _VECTOR_PARAMETER_SCHEMA_CACHE:
                allowed_schemas = _CachedVectorParameterSchemas(
                    _VECTOR_PARAMETER_SCHEMA_CACHE[
                        vector_schema_cache_key])
            else:
                allowed_schemas = _PrunedVectorParameterSchemas(parameters)
        else:
            allowed_schemas = [set()]
        if variant == "fp8_group_quant":
            if launch.algorithm == "vllm_per_token_group_quant_8bit":
                allowed_schemas = [set()]
            elif launch.algorithm.startswith((
                    "vllm_0_30_sm90_group128_fp8_quant_",
                    "vllm_0_30_sm90_silu_mul_group128_fp8_quant_")):
                allowed_schemas = [{"dynamic_shared_bytes"}]
            else:
                allowed_schemas = []
        if variant == "vllm_embedding_id_widen":
            allowed_schemas = [{"registers_per_thread", "values_per_thread"}]
        if variant == "vllm_logits_row_select":
            allowed_schemas = [{"registers_per_thread", "values_per_thread"}]
        if variant == "vllm_logits_fp32_cast":
            allowed_schemas = [{"registers_per_thread", "values_per_thread"}]
        if variant == "vllm_plain_greedy_sample" and attributes is not None and attributes["stage"] == "output_int32":
            allowed_schemas = [{"registers_per_thread", "values_per_thread"}]
        if variant == "vllm_plain_greedy_sample" and attributes is not None and attributes["stage"] == "argmax":
            allowed_schemas = [{"registers_per_thread", "values_per_thread", "input_vector_size",
                "dynamic_shared_bytes", "static_shared_bytes", "global_partial_bytes", "global_semaphore_bytes"}]
        if (variant == "elementwise" and attributes is not None
                and attributes["operation"] == "cast_int32_int64"):
            allowed_schemas = [{"registers_per_thread", "values_per_thread"}]
        if variant == "gather" and attributes is not None and "index_select_source" in attributes:
            allowed_schemas = [{"registers_per_thread", "values_per_thread"}]
        if variant in ("rmsnorm", "add_rmsnorm"):
            allowed_schemas.extend((
                {"input_reread_count"},
                {"input_reread_count", "lowered_sfu_operations"},
            ))
            allowed_schemas.extend(
                schema | {
                    "norm_reduction_tree_flops",
                    "norm_epsilon_add_flops",
                }
                for schema in tuple(allowed_schemas)
            )
            allowed_schemas.extend(
                schema | {"rstd_output_bytes"}
                for schema in tuple(allowed_schemas)
            )
            allowed_schemas.extend(
                schema | {"cann_rmsnorm_rstd_transfer_bytes"}
                for schema in tuple(allowed_schemas)
                if variant == "rmsnorm"
            )
            allowed_schemas.extend(
                schema | {
                    "lowered_vector_input_operations",
                    "lowered_vector_output_operations",
                    "norm_weight_l2_copies",
                }
                for schema in tuple(allowed_schemas)
            )
            allowed_schemas.extend(
                schema | {"lowered_vector_reduce_operations"}
                for schema in tuple(allowed_schemas)
                if "lowered_vector_input_operations" in schema
            )
            allowed_schemas.extend(
                schema | {
                    "cann_tiling_key", "cann_tiling_payload_bytes",
                    "norm_block_factor",
                }
                for schema in tuple(allowed_schemas)
            )
            if variant in ("rmsnorm", "add_rmsnorm"):
                allowed_schemas.extend(
                    schema | {"source_bound_block_reduction"}
                    for schema in tuple(allowed_schemas)
                )
                allowed_schemas.extend(
                    schema | {SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER}
                    for schema in tuple(allowed_schemas)
                    if ("source_bound_block_reduction" in schema
                        or variant == "add_rmsnorm")
                )
            if variant == "add_rmsnorm":
                allowed_schemas.extend(
                    schema | {"source_bound_add_rmsnorm_single_n_dag"}
                    for schema in tuple(allowed_schemas)
                    if "cann_tiling_key" in schema
                )
                allowed_schemas.extend(
                    schema | {"source_bound_add_rmsnorm_row_group_dag"}
                    for schema in tuple(allowed_schemas)
                    if "cann_tiling_key" in schema
                )
                allowed_schemas.extend(
                    schema | {"source_bound_add_rmsnorm_split_d_dag"}
                    for schema in tuple(allowed_schemas)
                    if "cann_tiling_key" in schema
                )
            elif variant == "rmsnorm":
                allowed_schemas.extend(
                    schema | {"source_bound_rmsnorm_single_row_dag"}
                    for schema in tuple(allowed_schemas)
                    if "cann_tiling_key" in schema
                )
                allowed_schemas.extend(
                    schema | {"source_bound_rmsnorm_row_group_dag"}
                    for schema in tuple(allowed_schemas)
                    if "cann_tiling_key" in schema
                )
                allowed_schemas.extend(
                    schema | {"source_bound_rmsnorm_split_d_dag"}
                    for schema in tuple(allowed_schemas)
                    if "cann_tiling_key" in schema
                )
                # CANN 8.5.1 MERGE_N (key 2) publishes its row-group extent
                # and the host tiler's is_performance flag alongside the
                # source identity; both are discrete launch facts the key-2
                # worker program needs and no other key defines.
                allowed_schemas.extend(
                    schema | {
                        "source_bound_rmsnorm_merge_n_dag",
                        "norm_merge_n_row_factor",
                        "norm_merge_n_performance_flag",
                    }
                    for schema in tuple(allowed_schemas)
                    if "cann_tiling_key" in schema
                )
            else:
                raise AssertionError(
                    f"unexpected normalization variant: {variant!r}")
            # GPU launch packages publish the reduction-tree and epsilon
            # work together, while the fixed CANN sources expose only the
            # scalar epsilon add here (ReduceSum is retained as its own
            # dependency service).  Extend this optional source fact after
            # all NORM schema variants have been composed so it remains
            # compatible with rstd/lowered/tiling/DAG fields without
            # weakening the closed schema boundary.
            allowed_schemas.extend(
                schema | {"norm_epsilon_add_flops"}
                for schema in tuple(allowed_schemas)
                if "norm_epsilon_add_flops" not in schema
            )
            allowed_schemas.extend(
                schema | {SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
            if variant in ("rmsnorm", "add_rmsnorm"):
                allowed_schemas.extend(
                    schema | {
                        SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER}
                    for schema in tuple(allowed_schemas)
                )
            allowed_schemas.extend(
                schema | {
                    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
        if variant in ("rotary", "rope"):
            rotary_schemas = (
                {"rotary_cache_path_bytes"},
                {
                    "rotary_cache_path_bytes",
                    "rotary_position_path_bytes",
                },
                {"rotary_cache_path_bytes", "lowered_sfu_operations"},
            )
            rotary_schemas += tuple(
                schema | {"rotary_position_path_bytes"}
                for schema in rotary_schemas
                if "rotary_position_path_bytes" not in schema
            )
            rotary_schemas += tuple(
                schema | {
                    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER}
                for schema in rotary_schemas
            )
            rotary_schemas += tuple(
                schema | {SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER}
                for schema in rotary_schemas
                if SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
                in schema
            )
            allowed_schemas.extend(rotary_schemas)
            allowed_schemas.extend(
                schema | {SOURCE_QK_ROPE_LOOP_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
            for program_parameter in (
                    SOURCE_METAX_ROPE_VLLM_0_8_5_ABI10_PROGRAM_PARAMETER,
                    SOURCE_METAX_ROPE_VLLM_0_9_1_ABI11_PROGRAM_PARAMETER,
                    SOURCE_METAX_ROPE_VLLM_0_24_0_ABI13_PROGRAM_PARAMETER):
                allowed_schemas.extend(
                    schema | {program_parameter}
                    for schema in tuple(allowed_schemas)
                    if SOURCE_QK_ROPE_LOOP_PARAMETER in schema
                )
            allowed_schemas.extend(
                schema | {SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
            atb_rope_schema = {
                "cann_tiling_key",
                "lowered_vector_operations",
                "source_bound_atb_bf16_rope_dag",
                "atb_rope_slice_elements",
                "atb_rope_rotary_coefficient",
                "atb_rope_cos_format",
            }
            atb_rope_schema.add(
                SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER)
            # The public torch_npu RoPE entry has two GatherV3 prefixes
            # before it dispatches the ATB body.  Keep their selected
            # versioned program and typed entry payload in the normalized
            # launch contract: Dataset and Serving must not infer either at
            # prediction time.
            atb_rope_schema.update({
                SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER,
                "cann_rope_gather_v3_real_core_num",
                "cann_rope_gather_v3_selected_row_bytes",
                "cann_rope_gather_v3_tiling_payload_bytes",
            })
            allowed_schemas.append(atb_rope_schema)
            allowed_schemas.append({"cann_tiling_key"})
        if variant == "qkv_rmsnorm_rope":
            allowed_schemas = [{
                SOURCE_VLLM_ASCEND_SPLIT_QKV_RMSNORM_ROPE_PROGRAM_PARAMETER,
                "qk_batch_rows_per_tile",
                "v_batch_rows_per_tile",
                "source_ub_bytes",
            }]
        if variant in ("rotary", "rope"):
            allowed_schemas.append({"lowered_sfu_operations"})
        if variant == "moe_compute_expert_tokens":
            allowed_schemas = [{
                "cann_tiling_key",
                "cann_moe_compute_expert_tokens_mix_aiv_1_0",
            }]
        if variant == "moe_finalize_routing":
            allowed_schemas = [{"cann_tiling_key"}]
        if variant in (
                "moe_init_routing", "moe_init_routing_v2",
                "moe_init_routing_v3"):
            # The CANN 8.5.1 MoeInitRouting manifest declares a zero-AIC,
            # one-AIV MIX_AIV task.  This is a launch topology fact, not a
            # tunable predictor parameter, and it must not be admitted for
            # other vector variants.
            allowed_schemas = [{"cann_moe_init_routing_mix_aiv_1_0"}]
        if variant in ("activation", "silu"):
            allowed_schemas.append({"warp_instructions_lower_bound"})
            allowed_schemas.append({"lowered_vector_operations"})
            allowed_schemas.append({
                "lowered_vector_operations",
                "vector_exp_serial_invocations",
                "vector_div_serial_invocations",
            })
            allowed_schemas.append({
                "lowered_vector_operations",
                "vector_exp_serial_invocations",
                "vector_div_serial_invocations",
                "swiglu_base_row_len",
                "swiglu_base_col_len",
                "swiglu_total_tiles",
                "swiglu_queue_depth",
            })
            allowed_schemas.extend(
                schema | {SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
            allowed_schemas.extend(
                schema | {SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
            allowed_schemas.extend(
                schema | {SOURCE_BOUND_WARP_SLOT2_INSTRUCTIONS_PARAMETER}
                for schema in tuple(allowed_schemas)
                if "warp_instructions_lower_bound" in schema
            )
            allowed_schemas.extend(
                schema | {
                    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
            allowed_schemas.append({
                SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER,
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
                SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER,
            })
            allowed_schemas.append({
                SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER,
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
                SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER,
                "warp_instructions_lower_bound",
            })
        if (variant == "moe_swiglu"
                and launch.algorithm == "aclnn_moe_swiglu"):
            allowed_schemas.append({
                "lowered_vector_operations",
                "vector_exp_serial_invocations",
                "vector_div_serial_invocations",
                "swiglu_base_row_len",
                "swiglu_base_col_len",
                "swiglu_total_tiles",
                "swiglu_queue_depth",
            })
        if variant in ("elementwise", "elementwise_pytorch_vec4"):
            allowed_schemas.append({"lowered_vector_operations"})
            if variant == "elementwise_pytorch_vec4":
                allowed_schemas.append({"warp_instructions"})
            if variant == "elementwise":
                elementwise_tbe_tiling_schema = {
                    "cann_tiling_key",
                    "cann_tiling_payload_bytes",
                    "elementwise_block_factor",
                    "elementwise_ub_factor",
                    "elementwise_input_queue_depth",
                    "elementwise_input_ping_pong_slots",
                    "elementwise_output_queue_depth",
                    SOURCE_CANN_BF16_ELEMENTWISE_DAG_PARAMETER,
                }
                allowed_schemas.extend(
                    schema | elementwise_tbe_tiling_schema
                    for schema in tuple(allowed_schemas)
                )
                allowed_schemas.extend(
                    schema | {
                        SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER}
                    for schema in tuple(allowed_schemas)
                )
                allowed_schemas.extend(
                    schema | {SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER}
                    for schema in tuple(allowed_schemas)
                )
                allowed_schemas.extend(
                    schema | {SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER}
                    for schema in tuple(allowed_schemas)
                )
                allowed_schemas.extend(
                    schema | {
                        SOURCE_PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_PARAMETER}
                    for schema in tuple(allowed_schemas)
                    if SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER in schema
                )
            allowed_schemas.extend(
                schema | {SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER}
                for schema in tuple(allowed_schemas)
                if SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
                in schema
            )
        if (variant == "elementwise"
                and launch.algorithm in (
                    "observed_flash_attention_copy:elementwise",
                    "observed_norm_input_materialization_copy:elementwise")):
            allowed_schemas.append({
                SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER})
        if (variant == "elementwise"
                and launch.algorithm == (
                    "pytorch_2_4_metax_3_7_2_bf16_grouped_strided_copy")):
            allowed_schemas.append({
                SOURCE_METAX_BF16_GROUPED_STRIDED_COPY_PARAMETER,
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
            })
        if variant == "gemm_reduce":
            if launch.algorithm == "h200_cublaslt129_bf16_external_aggregate_reduce":
                allowed_schemas.append({"split_k_partial_bytes_per_element", "split_k_partial_dram_bytes_per_element",
                    "modeled_split_k", "modeled_partial_bytes_per_element", "split_k_count_observed"})
            if launch.algorithm == "h200_vllm030_torch213_fp16_external_aggregate_reduce":
                allowed_schemas.append({"registers_per_thread", "modeled_split_k",
                    "modeled_partial_bytes_per_element", "split_k_count_observed"})
            reducer_schema = {
                "split_k_partial_bytes_per_element",
                "split_k_partial_dram_bytes_per_element",
            }
            allowed_schemas.append(reducer_schema)
            allowed_schemas.append(reducer_schema | {"split_k_count_observed"})
            allowed_schemas.append(reducer_schema | {
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
                "warp_instructions_lower_bound",
            })
        if variant == "rmsnorm":
            allowed_schemas.extend(
                schema | {
                    SOURCE_RMSNORM_VLLM_0_8_5_MACA_3_7_2_PROGRAM_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
        source_task_schemas = []
        for schema in tuple(allowed_schemas):
            eligible = (
                (variant == "rmsnorm"
                 and {
                     "input_reread_count",
                     "norm_reduction_tree_flops",
                     "norm_epsilon_add_flops",
                }.issubset(schema))
                or (variant == "add_rmsnorm"
                    and {
                        "input_reread_count",
                        "norm_reduction_tree_flops",
                        "norm_epsilon_add_flops",
                    }.issubset(schema))
                or (variant in ("rotary", "rope")
                    and ("rotary_position_path_bytes" in schema
                         or SOURCE_QK_ROPE_LOOP_PARAMETER in schema))
                or (variant in ("activation", "silu")
                    and (not schema
                         or SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER
                         in schema))
                or (variant == "elementwise_pytorch_vec4" and not schema)
                or (variant == "elementwise"
                    and SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER in schema)
                or (variant == "add_rmsnorm"
                    and "warp_instructions_lower_bound" in schema)
                or (variant == "rope"
                    and {
                        "warp_instructions",
                        "rotary_position_path_bytes",
                    }.issubset(schema))
                or (variant == "silu"
                    and "warp_instructions_lower_bound" in schema)
                or (variant == "elementwise_pytorch_vec4"
                    and "warp_instructions" in schema)
                or (variant == "elementwise"
                    and launch.algorithm
                    == "observed_flash_attention_copy:elementwise"
                    and "warp_instructions" in schema)
            )
            if eligible:
                source_task_schemas.append(
                    schema | {SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER})
        read_source_required = (
            (variant == "rmsnorm"
             and (shape["hidden"] == 128
                  or shape["hidden"] % 256 == 0))
            or variant == "silu"
            or (variant == "elementwise_pytorch_vec4"
                and shape["element_count"] % 1024 == 0)
            or (variant == "elementwise"
                and launch.algorithm
                == "observed_flash_attention_copy:elementwise")
        )
        allowed_schemas.extend(source_task_schemas)
        if read_source_required:
            read_source_schemas = tuple(
                schema | {
                    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER}
                for schema in source_task_schemas
            )
            allowed_schemas.extend(read_source_schemas)
            if variant == "rmsnorm":
                allowed_schemas.extend(
                    schema | {"norm_weight_l2_copies"}
                    for schema in (
                        *source_task_schemas, *read_source_schemas)
                )
            allowed_schemas.extend(
                schema | {SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER}
                for schema in read_source_schemas
            )
        if variant == "moe_act_and_mul":
            # The gated-activation program reads two operands per element and
            # publishes their per-wavefront request depth; it carries no other
            # launch facts.
            allowed_schemas.append({
                SOURCE_VLLM_0_24_0_ACT_AND_MUL_PROGRAM_PARAMETER,
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
            })
            allowed_schemas.append({
                SOURCE_VLLM_0_24_0_ACT_AND_MUL_PROGRAM_PARAMETER})
        if variant in (
                "rotary", "rope", "activation", "silu"):
            allowed_schemas.extend(
                schema | {
                    SOURCE_PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
        if (variant in ("rmsnorm", "add_rmsnorm", "silu", "rope", "moe_topk", "moe_align",
                        "moe_align_small_batch", "moe_count_and_sort", "moe_act_and_mul", "moe_sum")
                and launch.algorithm == f"vllm_0_11_sm90_source_{variant}"):
            allowed_schemas.append({"registers_per_thread", "static_shared_bytes", "dynamic_shared_bytes"})
        if (variant == "elementwise" and launch.algorithm in (
                "pytorch_2_13_sm90_fill_bf16", "pytorch_2_13_sm90_fill_fp16",
                "pytorch_2_13_sm90_fill_fp32", "pytorch_2_13_sm90_fill_int32",
                "pytorch_2_13_sm90_fill_int64")):
            allowed_schemas.append({"dynamic_shared_bytes"})
        if set(parameters) not in allowed_schemas:
            raise ValueError(
                f"VECTOR/{variant} launch parameter schema mismatch: "
                f"actual={sorted(parameters)}")
        if (vector_schema_cache_key is not None
                and vector_schema_cache_key
                not in _VECTOR_PARAMETER_SCHEMA_CACHE):
            _VECTOR_PARAMETER_SCHEMA_CACHE[vector_schema_cache_key] = (
                frozenset({frozenset(parameters)}))
        required = set(parameters)
    else:
        raise ValueError(
            f"unhandled {kind.value} launch parameter contract")
    if (kind is KernelKind.KV_CACHE and launch.execution_variant == "paged_kv_cache"
            and launch.algorithm == "vllm_0_11_sm90_source_kv_cache"):
        required = {"registers_per_thread", "static_shared_bytes", "dynamic_shared_bytes"}
    missing = required - set(parameters)
    unknown = set(parameters) - required
    if missing or unknown:
        raise ValueError(
            f"{kind.value} launch parameter schema mismatch: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}")
    if any(value < 0 for value in parameters.values()):
        raise ValueError(
            f"{kind.value} launch parameters cannot be negative: "
            f"{parameters}")
    if kind is KernelKind.GEMM:
        for name in (
                "compulsory_traffic", "aiv_mixed", "n_major_traffic"):
            if parameters[name] not in (0.0, 1.0):
                raise ValueError(
                    f"GEMM launch parameter {name} must be 0 or 1")
        partial_bytes = parameters["split_k_partial_bytes_per_element"]
        partial_dram_bytes = parameters[
            "split_k_partial_dram_bytes_per_element"]
        if not partial_bytes.is_integer():
            raise ValueError(
                "GEMM split_k_partial_bytes_per_element must be integral")
        if not partial_dram_bytes.is_integer():
            raise ValueError(
                "GEMM split_k_partial_dram_bytes_per_element must be "
                "integral")
        if partial_dram_bytes not in (0.0, partial_bytes):
            raise ValueError(
                "GEMM split-K partial DRAM width must be zero or equal to "
                "the materialized partial width")
        split_k = None if launch.tiling is None else launch.tiling.split_k
        if split_k is not None:
            if (split_k == 1
                    and (partial_bytes != 0 or partial_dram_bytes != 0)):
                raise ValueError(
                    "non-split GEMM cannot declare a partial element width")
    elif kind is KernelKind.FLASH_ATTN:
        if parameters["unpadded_flops"] not in (0.0, 1.0):
            raise ValueError(
                "FLASH_ATTN launch parameter unpadded_flops must be 0 or 1")
        if parameters["mma_smem_bytes_per_flop"] < 0:
            raise ValueError(
                "FLASH_ATTN mma_smem_bytes_per_flop cannot be negative")
        scratch_dram_width = parameters[
            "split_scratch_dram_bytes_per_element"]
        if (not scratch_dram_width.is_integer()
                or scratch_dram_width not in (0.0, 4.0)):
            raise ValueError(
                "FLASH_ATTN split scratch DRAM width must be zero or the "
                "four-byte FP32 materialized width")
        splits = (
            None if launch.tiling is None
            else launch.tiling.num_kv_splits)
        if splits == 1 and scratch_dram_width != 0.0:
            raise ValueError(
                "non-split FLASH_ATTN cannot declare a scratch DRAM width")
        page_table_bytes = parameters["page_table_bytes_per_kv_panel"]
        if (not page_table_bytes.is_integer()
                or page_table_bytes not in (0.0, 4.0)):
            raise ValueError(
                "FLASH_ATTN page-table access must be zero or one int32 "
                "entry per KV panel")
        if splits == 1 and page_table_bytes != 0.0:
            raise ValueError(
                "non-split FLASH_ATTN cannot declare page-table panel work")
        if "cann_source_task_count" in parameters:
            if (launch.source_program is None
                    and parameters[
                        "cann_source_vector_terminal_operations"] != 0.0):
                raise ValueError(
                    "FLASH_ATTN FlashDecode terminal lifecycle requires a "
                    "typed FIA main program")
            for name in (
                    "cann_source_task_count",
                    "cann_source_tensor_logical_operations",
                    "cann_source_vector_operations",
                    "cann_source_vector_main_operations",
                    "cann_source_vector_initialization_operations",
                    "cann_source_vector_terminal_operations",
                    "cann_source_vector_vec1_operations",
                    "cann_source_vector_vec2_operations",
                    "cann_source_vector_worker_count",
                    "cann_source_vec1_read_bytes",
                    "cann_source_mm2_score_read_bytes",
                    "cann_source_vec2_mm2_read_bytes",
                    "cann_source_vec2_state_read_bytes",
                    "cann_source_flash_decode_terminal_read_bytes",
                    "cann_source_vec1_write_bytes",
                    "cann_source_nupdate_write_bytes",
                    "cann_source_vec2_write_bytes",
                    "cann_source_flash_decode_terminal_write_bytes",
                    "cann_source_output_init_bytes",
                    "cann_source_external_write_bytes",
                    "cann_source_vec1_unique_write_bytes",
                    "cann_source_vec2_unique_write_bytes",
                    "cann_source_aiv_unique_write_bytes",
                    "cann_source_mm1_unique_write_bytes",
                    "cann_source_mm2_unique_write_bytes",
                    "cann_source_aic_unique_write_bytes"):
                if not parameters[name].is_integer():
                    raise ValueError(
                        f"FLASH_ATTN source parameter {name} must be integral")
            for name in (
                    "cann_source_task_count",
                    "cann_source_tensor_logical_operations",
                    "cann_source_vector_operations",
                    "cann_source_vector_main_operations"):
                if parameters[name] <= 0:
                    raise ValueError(
                        f"FLASH_ATTN source parameter {name} must be positive")
            for name in (
                    "cann_source_tensor_shape_efficiency",
                    "cann_source_tensor_schedule_efficiency",
                    "cann_source_vector_schedule_efficiency"):
                if not 0 < parameters[name] <= 1:
                    raise ValueError(
                        f"FLASH_ATTN source parameter {name} must be in "
                        "(0, 1]")
            if not math.isclose(
                    parameters["cann_source_vector_operations"],
                    parameters["cann_source_vector_main_operations"]
                    + parameters[
                        "cann_source_vector_terminal_operations"],
                    rel_tol=0.0, abs_tol=0.0):
                raise ValueError(
                    "FLASH_ATTN source Vector phases do not conserve "
                    "operations")
            if not math.isclose(
                    parameters["cann_source_vector_main_operations"],
                    parameters[
                        "cann_source_vector_initialization_operations"]
                    + parameters["cann_source_vector_vec1_operations"]
                    + parameters["cann_source_vector_vec2_operations"],
                    rel_tol=0.0, abs_tol=0.0):
                raise ValueError(
                    "FLASH_ATTN source Vec1/Vec2 phases do not conserve main "
                    "Vector operations")
            worker_count = int(parameters[
                "cann_source_vector_worker_count"])
            worker_operations = []
            for worker in range(worker_count):
                for service in ("generic", "exp", "div"):
                    operation_name = (
                        f"cann_source_vector_worker_{worker}_{service}_operations")
                    operation_value = parameters[operation_name]
                    if not operation_value.is_integer():
                        raise ValueError(
                            f"FLASH_ATTN source parameter {operation_name} "
                            "must be integral")
                    worker_operations.append(operation_value)
                for service in ("exp", "div"):
                    operation_name = (
                        f"cann_source_vector_worker_{worker}_{service}_operations")
                    invocation_name = (
                        f"cann_source_vector_worker_{worker}_{service}_invocations")
                    invocation_value = parameters[invocation_name]
                    if not invocation_value.is_integer():
                        raise ValueError(
                            f"FLASH_ATTN source parameter {invocation_name} "
                            "must be integral")
                    if ((parameters[operation_name] > 0)
                            != (invocation_value > 0)):
                        raise ValueError(
                            "FLASH_ATTN source worker Exp/Div operations and "
                            "invocations must be positive together")
            if not math.isclose(
                    math.fsum(worker_operations),
                    parameters["cann_source_vector_operations"],
                    rel_tol=0.0, abs_tol=0.0):
                raise ValueError(
                    "FLASH_ATTN source Vector workers do not conserve "
                    "operations")
            if (parameters["cann_source_external_write_bytes"]
                    < parameters["cann_source_output_init_bytes"]):
                raise ValueError(
                    "FLASH_ATTN external writes cannot be smaller than "
                    "output initialization")
            # The distinct AIV-written extent cannot exceed the bytes MTE3
            # actually moves through the same path.
            if (parameters["cann_source_aiv_unique_write_bytes"]
                    > parameters["cann_source_vec1_write_bytes"]
                    + parameters["cann_source_nupdate_write_bytes"]
                    + parameters["cann_source_vec2_write_bytes"]
                    + parameters[
                        "cann_source_flash_decode_terminal_write_bytes"]
                    + parameters["cann_source_output_init_bytes"]):
                raise ValueError(
                    "FLASH_ATTN distinct AIV writes exceed the AIV write "
                    "stream")
            # Likewise for the Fixpipe ring against the bytes the AIC
            # actually stores, which the Vector side reads back in full.
            if (parameters["cann_source_mm1_unique_write_bytes"]
                    > parameters["cann_source_vec1_read_bytes"]
                    or parameters["cann_source_mm2_unique_write_bytes"]
                    > parameters["cann_source_vec2_mm2_read_bytes"]
                    or parameters["cann_source_aic_unique_write_bytes"]
                    != parameters["cann_source_mm1_unique_write_bytes"]
                    + parameters["cann_source_mm2_unique_write_bytes"]):
                raise ValueError(
                    "FLASH_ATTN distinct AIC writes exceed the Fixpipe write "
                    "stream")
    elif kind is KernelKind.FUSED_MOE:
        for name in (
                "group_size_m", "padded_rows",
                "operand_replay_a", "operand_replay_b"):
            if not parameters[name].is_integer():
                raise ValueError(
                    f"FUSED_MOE launch parameter {name} must be integral")
        for name in ("operand_replay_a", "operand_replay_b"):
            if parameters[name] <= 0:
                raise ValueError(
                    f"FUSED_MOE launch parameter {name} must be positive")
    elif kind in (KernelKind.VECTOR, KernelKind.KV_CACHE):
        for name in (
                "input_reread_count", "lowered_sfu_operations",
                "lowered_vector_operations", "lowered_vector_input_operations",
                "lowered_vector_reduce_operations",
                "lowered_vector_output_operations", "norm_weight_l2_copies",
                "norm_epsilon_add_flops", "norm_reduction_tree_flops",
                "rotary_cache_path_bytes", "rotary_position_path_bytes",
                "atb_rope_slice_elements",
                "atb_rope_rotary_coefficient",
                "cann_rope_gather_v3_real_core_num",
                "cann_rope_gather_v3_selected_row_bytes",
                "cann_rope_gather_v3_tiling_payload_bytes",
                "kv_cache_slot_l1_bytes",
                "rstd_output_bytes", "cann_rmsnorm_rstd_transfer_bytes",
                "norm_merge_n_row_factor",
                "cann_tiling_payload_bytes", "warp_instructions",
                "warp_instructions_lower_bound", "norm_block_factor",
                "source_bound_block_reduction",
                SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER,
                SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER,
                SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER,
                SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER,
                SOURCE_QK_ROPE_LOOP_PARAMETER,
                SOURCE_SCALAR_KV_LOOP_PARAMETER,
                SOURCE_VECTORIZED_KV_COPY_PARAMETER,
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
                "source_bound_resident_ctas_per_unit",
                SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER,
                SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER,
                SOURCE_BOUND_WARP_SLOT2_INSTRUCTIONS_PARAMETER,
                SOURCE_RMSNORM_VLLM_0_8_5_MACA_3_7_2_PROGRAM_PARAMETER,
                SOURCE_METAX_ROPE_VLLM_0_8_5_ABI10_PROGRAM_PARAMETER,
                SOURCE_METAX_ROPE_VLLM_0_9_1_ABI11_PROGRAM_PARAMETER,
                SOURCE_METAX_ROPE_VLLM_0_24_0_ABI13_PROGRAM_PARAMETER,
                SOURCE_VLLM_0_24_0_ACT_AND_MUL_PROGRAM_PARAMETER,
                SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER):
            if name in parameters:
                value = parameters[name]
                if not value.is_integer() or value <= 0:
                    raise ValueError(
                        f"VECTOR {name} must be a positive integer")
        if (not graph_padding
                and "norm_epsilon_add_flops" in parameters
                and parameters["norm_epsilon_add_flops"]
                != float(shape["rows"])):
            raise ValueError(
                "VECTOR norm_epsilon_add_flops must equal one scalar add "
                "per normalization row")
        if ("source_bound_block_reduction" in parameters
                and parameters["source_bound_block_reduction"] != 1.0):
            raise ValueError(
                "VECTOR source_bound_block_reduction must equal one")
        if (SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER in parameters
                and parameters[
                    SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER] != 1.0):
            raise ValueError(
                "VECTOR source_bound_gpu_norm_cta_pipeline must equal one")
        if (SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER in parameters
                and launch.execution_variant not in (
                    "rmsnorm", "add_rmsnorm")):
            raise ValueError(
                "VECTOR source_bound_gpu_norm_cta_pipeline requires a GPU "
                "normalization launch")
        if SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER in parameters:
            if parameters[
                    SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER
                    ] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_silu_nonrecurrent_cta_pipeline "
                    "must equal one")
            if (not graph_padding
                    and (launch.execution_variant != "silu"
                    or launch.grid != (shape["tok"], 1, 1)
                    or launch.grid_size < 108
                    or launch.block != (1024, 1, 1))):
                raise ValueError(
                    "VECTOR source_bound_silu_nonrecurrent_cta_pipeline "
                    "requires its BF16 full-device block1024 source domain")
        for source_loop_parameter in (
                SOURCE_QK_ROPE_LOOP_PARAMETER,
                SOURCE_SCALAR_KV_LOOP_PARAMETER,
                SOURCE_VECTORIZED_KV_COPY_PARAMETER):
            if (source_loop_parameter in parameters
                    and parameters[source_loop_parameter] != 1.0):
                raise ValueError(
                    f"VECTOR {source_loop_parameter} must equal one")
        metax_rope_program_markers = tuple(
            parameter for parameter in (
                SOURCE_METAX_ROPE_VLLM_0_8_5_ABI10_PROGRAM_PARAMETER,
                SOURCE_METAX_ROPE_VLLM_0_9_1_ABI11_PROGRAM_PARAMETER,
                SOURCE_METAX_ROPE_VLLM_0_24_0_ABI13_PROGRAM_PARAMETER,
            )
            if parameter in parameters)
        if len(metax_rope_program_markers) > 1:
            raise ValueError(
                "MetaX RoPE versioned program identities are mutually "
                "exclusive")
        if (SOURCE_QK_ROPE_LOOP_PARAMETER in parameters
                and len(metax_rope_program_markers) != 1):
            raise ValueError(
                "MetaX source Q/K RoPE loop requires exactly one "
                "versioned program identity")
        if metax_rope_program_markers:
            if SOURCE_QK_ROPE_LOOP_PARAMETER not in parameters:
                raise ValueError(
                    "MetaX RoPE program identity requires the C500 "
                    "source Q/K loop")
            expected_variant = (
                "rope"
                if metax_rope_program_markers[0]
                == SOURCE_METAX_ROPE_VLLM_0_8_5_ABI10_PROGRAM_PARAMETER
                else "rotary")
            if launch.execution_variant != expected_variant:
                raise ValueError(
                    "MetaX RoPE program identity disagrees with the "
                    f"versioned execution variant: expected "
                    f"{expected_variant!r}")
        if (SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER in parameters
                and parameters[
                    SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER] != 1.0):
            raise ValueError(
                "VECTOR source_bound_memory_dependency_task must equal one")
        if SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER in parameters:
            dimensions = parameters[
                SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER]
            if dimensions not in (3.0, 4.0):
                raise ValueError(
                    "VECTOR source_bound_tensor_iterator_copy_task must "
                    "equal three or four")
            if (variant != "elementwise"
                    or launch.algorithm not in (
                        "observed_flash_attention_copy:elementwise",
                        "observed_norm_input_materialization_copy:"
                        "elementwise")):
                raise ValueError(
                    "VECTOR TensorIterator copy task requires an observed "
                    "FlashAttention or Norm input-materialization "
                    "elementwise copy")
        if (SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER in parameters
                and launch.execution_variant not in (
                    "rmsnorm", "add_rmsnorm", "rotary", "rope", "elementwise",
                    "elementwise_pytorch_vec4", "paged_kv_cache",
                    "moe_act_and_mul", "gemm_reduce")):
            raise ValueError(
                "VECTOR source-bound L2 read warps require a source-audited "
                "RMSNorm, Rotary, Elementwise, gated-activation or KV-cache "
                "launch")
        if ("source_bound_resident_ctas_per_unit" in parameters
                and (launch.execution_variant != "paged_kv_cache"
                     or SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER
                     not in parameters
                     or SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER
                     not in parameters)):
            raise ValueError(
                "VECTOR source-bound resident CTAs require an exact "
                "paged KV-cache source launch")
        if (SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER in parameters
                and SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER
                not in parameters
                and not any(marker in parameters for marker in (
                    SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER,
                    SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER,
                    SOURCE_METAX_BF16_GROUPED_STRIDED_COPY_PARAMETER,
                    SOURCE_QK_ROPE_LOOP_PARAMETER,
                    SOURCE_SCALAR_KV_LOOP_PARAMETER,
                    SOURCE_VECTORIZED_KV_COPY_PARAMETER,
                    SOURCE_VLLM_0_24_0_ACT_AND_MUL_PROGRAM_PARAMETER,
                    "warp_instructions_lower_bound",
                ))):
            raise ValueError(
                "VECTOR launch parameter schema mismatch: source-bound L2 "
                "reads require a source identity")
        if "source_bound_add_rmsnorm_single_n_dag" in parameters:
            if parameters[
                    "source_bound_add_rmsnorm_single_n_dag"] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_add_rmsnorm_single_n_dag must "
                    "equal one")
            if (launch.execution_variant != "add_rmsnorm"
                    or parameters["cann_tiling_key"] != 33.0):
                raise ValueError(
                    "VECTOR source_bound_add_rmsnorm_single_n_dag requires "
                    "AddRmsNorm CANN key 33")
        if "source_bound_add_rmsnorm_row_group_dag" in parameters:
            if parameters[
                    "source_bound_add_rmsnorm_row_group_dag"] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_add_rmsnorm_row_group_dag must "
                    "equal one")
            if (launch.execution_variant != "add_rmsnorm"
                    or parameters["cann_tiling_key"] != 30.0):
                raise ValueError(
                    "VECTOR source_bound_add_rmsnorm_row_group_dag requires "
                    "AddRmsNorm CANN key 30")
        if "source_bound_add_rmsnorm_split_d_dag" in parameters:
            if parameters[
                    "source_bound_add_rmsnorm_split_d_dag"] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_add_rmsnorm_split_d_dag must "
                    "equal one")
            if (launch.execution_variant != "add_rmsnorm"
                    or parameters["cann_tiling_key"] != 31.0):
                raise ValueError(
                    "VECTOR source_bound_add_rmsnorm_split_d_dag requires "
                    "AddRmsNorm CANN key 31")
        if sum(
                name in parameters
                for name in (
                    "source_bound_add_rmsnorm_single_n_dag",
                    "source_bound_add_rmsnorm_row_group_dag",
                    "source_bound_add_rmsnorm_split_d_dag",
                )) > 1:
            raise ValueError(
                "VECTOR AddRMSNorm source DAG identities are mutually "
                "exclusive")
        if "source_bound_rmsnorm_single_row_dag" in parameters:
            if parameters["source_bound_rmsnorm_single_row_dag"] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_rmsnorm_single_row_dag must equal "
                    "one")
            if (launch.execution_variant != "rmsnorm"
                    or parameters["cann_tiling_key"] != 0.0):
                raise ValueError(
                    "VECTOR source_bound_rmsnorm_single_row_dag requires "
                    "RmsNorm CANN key 0")
        if "source_bound_rmsnorm_row_group_dag" in parameters:
            if parameters["source_bound_rmsnorm_row_group_dag"] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_rmsnorm_row_group_dag must equal "
                    "one")
            if (launch.execution_variant != "rmsnorm"
                    or parameters["cann_tiling_key"] != 0.0):
                raise ValueError(
                    "VECTOR source_bound_rmsnorm_row_group_dag requires "
                    "RmsNorm CANN key 0")
        if "source_bound_rmsnorm_split_d_dag" in parameters:
            if parameters["source_bound_rmsnorm_split_d_dag"] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_rmsnorm_split_d_dag must equal "
                    "one")
            if (launch.execution_variant != "rmsnorm"
                    or parameters["cann_tiling_key"] != 1.0):
                raise ValueError(
                    "VECTOR source_bound_rmsnorm_split_d_dag requires "
                    "RmsNorm CANN key 1")
        if "source_bound_rmsnorm_merge_n_dag" in parameters:
            if parameters["source_bound_rmsnorm_merge_n_dag"] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_rmsnorm_merge_n_dag must equal one")
            if (launch.execution_variant != "rmsnorm"
                    or parameters["cann_tiling_key"] != 2.0):
                raise ValueError(
                    "VECTOR source_bound_rmsnorm_merge_n_dag requires "
                    "RmsNorm CANN key 2")
            if parameters["norm_merge_n_row_factor"] < 1.0:
                raise ValueError(
                    "VECTOR RMSNorm MERGE_N row factor must be positive")
            if parameters["norm_merge_n_performance_flag"] not in (0.0, 1.0):
                raise ValueError(
                    "VECTOR RMSNorm MERGE_N is_performance flag must be "
                    "zero or one")
        if sum(
                name in parameters
                for name in (
                    "source_bound_rmsnorm_single_row_dag",
                    "source_bound_rmsnorm_row_group_dag",
                    "source_bound_rmsnorm_split_d_dag",
                    "source_bound_rmsnorm_merge_n_dag",
                )) > 1:
            raise ValueError(
                "VECTOR RMSNorm source DAG identities are mutually "
                "exclusive")
        if (not graph_padding
                and "source_bound_atb_bf16_rope_dag" in parameters):
            if parameters["source_bound_atb_bf16_rope_dag"] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_atb_bf16_rope_dag must equal one")
            if launch.execution_variant not in ("rotary", "rope"):
                raise ValueError(
                    "VECTOR ATB BF16 RoPE DAG requires a RoPE launch")
            if parameters["atb_rope_rotary_coefficient"] != 2.0:
                raise ValueError(
                    "VECTOR ATB BF16 RoPE rotary coefficient must equal two")
            if parameters["atb_rope_cos_format"] != 0.0:
                raise ValueError(
                    "VECTOR ATB BF16 RoPE cos format must equal zero")
            if shape["head_dim"] % 32:
                raise ValueError(
                    "VECTOR ATB BF16 RoPE DAG requires a 32-element aligned "
                    "head dimension")
            if launch.execution_variant == "rope":
                if shape["rotary_dim"] != shape["head_dim"]:
                    raise ValueError(
                        "VECTOR ATB BF16 RoPE DAG requires full-dimension "
                        "semantics")
            expected_key = 24 if shape["seq_len"] >= 64 else 22
            if parameters["cann_tiling_key"] != float(expected_key):
                raise ValueError(
                    "VECTOR ATB BF16 RoPE tiling key disagrees with the "
                    "source selector")
            nominal_slice = 2_048 if expected_key == 24 else 4_096
            expected_slice = (
                nominal_slice // shape["head_dim"] * shape["head_dim"])
            if (expected_slice <= 0
                    or parameters["atb_rope_slice_elements"]
                    != float(expected_slice)):
                raise ValueError(
                    "VECTOR ATB BF16 RoPE slice disagrees with the selected "
                    "source kernel")
            query_elements = shape["num_query_heads"] * shape["head_dim"]
            key_elements = shape["num_kv_heads"] * shape["head_dim"]
            if query_elements % 64 or key_elements % 64:
                raise ValueError(
                    "VECTOR ATB BF16 RoPE DAG requires complete 64-element "
                    "Vector repeats")
            expected_lowered = (
                5 * shape["element_count"]
                + 2 * shape["seq_len"] * min(query_elements, expected_slice))
            worker_count = launch.grid_size
            base_tokens_per_worker, worker_remainder = divmod(
                shape["seq_len"], worker_count)
            second_mask_workers = (
                worker_count
                if expected_key == 24 and base_tokens_per_worker > 1
                else (
                    worker_remainder
                    if expected_key == 24 and base_tokens_per_worker == 1
                    else 0))
            expected_lowered += shape["head_dim"] * (
                worker_count + second_mask_workers)
            query_heads_per_slice = min(
                query_elements, expected_slice) // shape["head_dim"]
            mask_copy_operations = (
                (worker_count + second_mask_workers)
                * (query_heads_per_slice - 1)
                * shape["head_dim"] * 4)
            cos_sin_copy_operations = (
                shape["seq_len"] * 2 * (query_heads_per_slice - 1)
                * shape["head_dim"] * 2)
            half_rotation_copy_operations = (
                2 * shape["seq_len"] * (query_elements + key_elements))
            expected_lowered += (
                mask_copy_operations
                + cos_sin_copy_operations
                + half_rotation_copy_operations)
            if parameters["lowered_vector_operations"] != float(
                    expected_lowered):
                raise ValueError(
                    "VECTOR ATB BF16 RoPE lowered work disagrees with the "
                    "source program")
            gather_fields = {
                SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER,
                SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER,
                "cann_rope_gather_v3_real_core_num",
                "cann_rope_gather_v3_selected_row_bytes",
                "cann_rope_gather_v3_tiling_payload_bytes",
            }
            missing_gather_fields = gather_fields - set(parameters)
            if missing_gather_fields:
                raise ValueError(
                    "VECTOR public ATB RoPE GatherV3 launch is incomplete: "
                    f"missing={sorted(missing_gather_fields)}")
            if parameters[SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER] != 2.0:
                raise ValueError(
                    "VECTOR public ATB RoPE requires exactly two GatherV3 "
                    "prefixes")
            gather_program = parameters[
                SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER]
            if (not gather_program.is_integer()
                    or int(gather_program) != 10_330):
                raise ValueError(
                    "VECTOR public ATB RoPE GatherV3 must select the CANN "
                    "8.5.1 direct-row key 10330")
            gather_workers = parameters["cann_rope_gather_v3_real_core_num"]
            if not gather_workers.is_integer() or gather_workers <= 0:
                raise ValueError(
                    "VECTOR public ATB RoPE GatherV3 realCoreNum must be a "
                    "positive integer")
            selected_row_bytes = parameters[
                "cann_rope_gather_v3_selected_row_bytes"]
            if (not selected_row_bytes.is_integer()
                    or selected_row_bytes != float(
                        shape["seq_len"] * shape["head_dim"] * 2)):
                raise ValueError(
                    "VECTOR public ATB RoPE GatherV3 selected-row bytes "
                    "disagree with its BF16 [T, D] output ABI")
            if parameters["cann_rope_gather_v3_tiling_payload_bytes"] != 192.0:
                raise ValueError(
                    "VECTOR public ATB RoPE GatherV3 requires its fixed "
                    "192-byte CANN entry payload")
        for name in (
                "vector_exp_serial_invocations",
                "vector_div_serial_invocations"):
            if name in parameters:
                value = parameters[name]
                if not value.is_integer() or value <= 0:
                    raise ValueError(
                        f"VECTOR {name} must be a positive integer")
        if "cann_tiling_key" in parameters:
            tiling_key = parameters["cann_tiling_key"]
            if launch.execution_variant == "rmsnorm":
                # rms_norm.cpp:31-34/49-58 binds key 0 to KernelRmsNorm, key 1
                # to KernelRmsNormSplitD and key 2 to KernelRmsNormMergeN.
                allowed_keys = (0, 1, 2)
                key_owner = "RmsNorm"
            elif launch.execution_variant == "add_rmsnorm":
                allowed_keys = (30, 31, 33)
                key_owner = "AddRmsNorm"
            elif launch.execution_variant == "paged_kv_cache":
                allowed_keys = (200_000_000, 202_000_000)
                key_owner = "ReshapeAndCache"
            elif launch.execution_variant in ("rotary", "rope"):
                allowed_keys = (22, 24)
                key_owner = "ATB BF16 RoPE"
            elif launch.execution_variant == "moe_compute_expert_tokens":
                allowed_keys = (1001, 1002, 1003)
                key_owner = "MoeComputeExpertTokens"
            elif launch.execution_variant == "moe_finalize_routing":
                allowed_keys = (20036,)
                key_owner = "MoeFinalizeRoutingV2 BF16 no-bias"
            elif launch.execution_variant == "elementwise":
                allowed_keys = (210_000_000, 210_010_000, 210_020_000)
                key_owner = "TBE elementwise"
            else:
                raise ValueError(
                    "VECTOR cann_tiling_key requires a supported CANN "
                    "operator variant")
            if (not tiling_key.is_integer()
                    or int(tiling_key) not in allowed_keys):
                raise ValueError(
                    f"VECTOR cann_tiling_key must be one of the supported "
                    f"CANN 8.5.1 {key_owner} keys {allowed_keys}")
            if launch.execution_variant in ("rmsnorm", "add_rmsnorm"):
                # ``GET_TILING_DATA`` deserializes the source entry's typed
                # payload before the key-specific program begins.  This is a
                # fail-closed normalized launch fact: 28-byte RmsNorm and
                # 80-byte AddRmsNorm payloads use different independently
                # measured materialization mechanisms.
                payload_bytes = parameters["cann_tiling_payload_bytes"]
                expected_payload_bytes = (
                    28.0 if launch.execution_variant == "rmsnorm" else 80.0)
                if payload_bytes != expected_payload_bytes:
                    raise ValueError(
                        "VECTOR CANN normalization tiling payload disagrees "
                        "with its fixed CANN 8.5.1 source entry: "
                        f"expected={expected_payload_bytes}, "
                        f"actual={payload_bytes}")
            if launch.execution_variant == "moe_compute_expert_tokens":
                from src.kernel.launch_config.ascend_910b.vector_launch import (
                    select_moe_compute_expert_tokens_tiling_key,
                )
                expected_key = select_moe_compute_expert_tokens_tiling_key(
                    shape["tokens"] * shape["topk"], shape["experts"])
                if int(tiling_key) != expected_key:
                    raise ValueError(
                        "VECTOR MoeComputeExpertTokens cann_tiling_key "
                        "disagrees with the source selector: "
                        f"expected={expected_key}, actual={int(tiling_key)}")
                task_topology = parameters[
                    "cann_moe_compute_expert_tokens_mix_aiv_1_0"]
                if task_topology != 1.0:
                    raise ValueError(
                        "VECTOR MoeComputeExpertTokens MIX_AIV 1:0 source "
                        "topology must be exactly one")
            if launch.execution_variant == "elementwise":
                # CANN 8.5.1 ``BroadcastSch<201>`` invokes
                # ``GET_TILING_DATA_WITH_STRUCT(BroadcastOneDimTilingData,
                # ...)``. The fixed source layout contains five int32, two
                # aligned int64, and 16 scalar bytes: 56 B in total.
                if parameters["cann_tiling_payload_bytes"] != 56.0:
                    raise ValueError(
                        "VECTOR CANN elementwise tiling payload disagrees "
                        "with its fixed CANN 8.5.1 BroadcastOneDim source "
                        "entry: expected=56.0, actual="
                        f"{parameters['cann_tiling_payload_bytes']}")
                block_factor = parameters["elementwise_block_factor"]
                ub_factor = parameters["elementwise_ub_factor"]
                input_depth = parameters[
                    "elementwise_input_queue_depth"]
                input_ping_pong_slots = parameters[
                    "elementwise_input_ping_pong_slots"]
                output_depth = parameters[
                    "elementwise_output_queue_depth"]
                for name, value in (
                        ("elementwise_block_factor", block_factor),
                        ("elementwise_ub_factor", ub_factor),
                        ("elementwise_input_queue_depth", input_depth),
                        ("elementwise_input_ping_pong_slots",
                         input_ping_pong_slots),
                        ("elementwise_output_queue_depth", output_depth)):
                    if not value.is_integer() or value <= 0:
                        raise ValueError(
                            f"VECTOR {name} must be a positive integer")
                elements = shape["element_count"]
                if int(block_factor) < int(ub_factor):
                    raise ValueError(
                        "VECTOR elementwise UB factor cannot exceed its "
                        "block factor")
                # Under graph padding the launch is selected on the captured
                # model-input extent while ``shape`` carries logical work, so
                # the two element counts legitimately differ; the same
                # relaxation guards every other shape-derived VECTOR
                # cross-check in this function.
                if (not graph_padding
                        and launch.grid_size != math.ceil(
                            elements / block_factor)):
                    raise ValueError(
                        "VECTOR elementwise block factor disagrees with its "
                        "launch grid")
                expected_depths = {
                    210_000_000: (1, 2, 1),
                    210_010_000: (2, 2, 1),
                    210_020_000: (4, 2, 1),
                }[int(tiling_key)]
                if (int(input_depth), int(input_ping_pong_slots),
                        int(output_depth)) != expected_depths:
                    raise ValueError(
                        "VECTOR elementwise queue/admission depths disagree "
                        "with its CANN tiling key")
        if "split_k_partial_bytes_per_element" in parameters:
            partial_bytes = parameters[
                "split_k_partial_bytes_per_element"]
            if not partial_bytes.is_integer() or partial_bytes <= 0:
                raise ValueError(
                    "VECTOR split_k_partial_bytes_per_element must be a "
                    "positive integer")
            partial_dram_bytes = parameters[
                "split_k_partial_dram_bytes_per_element"]
            if not partial_dram_bytes.is_integer():
                raise ValueError(
                    "VECTOR split_k_partial_dram_bytes_per_element must be "
                    "integral")
            if partial_dram_bytes not in (0.0, partial_bytes):
                raise ValueError(
                    "VECTOR split-K partial DRAM width must be zero or equal "
                    "to the materialized partial width")
    elif kind in (
            KernelKind.EAGER_ATTN, KernelKind.UNFUSED_MOE,
            KernelKind.SPARSE_ATTN_INDEXER, KernelKind.SPARSE_ATTN):
        pass
    else:
        raise ValueError(
            f"unhandled {kind.value} launch parameter validation")
    return launch


@dataclass(frozen=True)
class L2ExecutionContext:
    """Explicit cache history supplied by an execution owner.

    ``same_program_operand_replays`` counts immediately preceding launches
    that reuse this invocation's immutable operand allocations.  It does not
    assert that result buffers reuse an address: a predictor must conservatively
    account for every predecessor result allocation before declaring an operand
    resident.  Dataset capture and Serving lowering may supply different
    contexts, but they retain the same launch program and predictor path.
    """

    same_program_operand_replays: int = 0

    def __post_init__(self) -> None:
        value = self.same_program_operand_replays
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                "L2ExecutionContext same_program_operand_replays must be an "
                "integer")
        if value < 0:
            raise ValueError(
                "L2ExecutionContext same_program_operand_replays must be "
                "nonnegative")


def _matches_maca_integer_cast_launch(launch, elements, attributes, variant):
    from src.kernel.launch_config.metax_c500.vector_launch import pytorch_int32_int64_cast_launch
    return launch is not None and launch == pytorch_int32_int64_cast_launch(elements, attributes, variant)


@dataclass(frozen=True)
class KernelSpec:
    """A complete hardware modeling primitive consumed by predictors."""

    kind: KernelKind
    hardware_id: str
    shape: Mapping[str, int] = field(default_factory=dict)
    dtype: Optional[str] = None
    variant: Optional[str] = None
    launch: Optional[LaunchConfig] = None
    transfer: Optional[PhysicalTransferConfig] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    anonymous_l2_dirty_bytes: int = 0
    execution_support: str = "supported"
    unsupported_reason: Optional[str] = None
    l2_execution_context: L2ExecutionContext = field(
        default_factory=L2ExecutionContext)

    def validate_materialization_launch(self, hardware) -> None:
        """Validate source-selected gather geometry with its hardware input."""
        from src.kernel.launch_config.select_launch_config import select_launch_config
        if (self.kind is not KernelKind.VECTOR or self.variant != "gather"
                or "materialization_source" not in self.attributes):
            raise ValueError("materialization validation requires a declared gather source")
        if not hardware.supports_source_contract("pytorch_2_4_maca_materialized_gather_launch"):
            raise ValueError("materialization source has no hardware launch binding")
        physical = self.attributes["model_input_shape"] if "model_input_shape" in self.attributes else self.shape
        expected = select_launch_config(self.hardware_id, self.kind, physical, self.dtype, self.variant,
            {**self.attributes, "launch_policy": "injected_launch"}, hardware=hardware)
        if self.launch != expected:
            raise ValueError("materialized gather requires its exact source-selected launch")

    def validate_legacy_bf16_external_launch(self, hardware) -> None:
        """Replay exact published CUDA12.9 geometry; modeled partitions stay explicit."""
        from src.kernel.launch_config.nvidia_h200.gemm_launch import select_legacy_bf16_external_aggregate_launch
        main = "h200_cublaslt129_bf16_external_aggregate_gemm"
        reducer = "h200_cublaslt129_bf16_external_aggregate_reduce"
        if (self.hardware_id != "nvidia_h200" or self.dtype != "bf16" or self.launch is None
                or self.execution_support != "supported" or self.launch.tiling is not None
                or self.launch.source_program is not None or "source_gemm_runtime" in self.attributes
                or hardware is not None and hardware.hardware_id != self.hardware_id):
            raise ValueError("legacy BF16 aggregate requires its exact source/hardware/dtype contract")
        reduction = self.kind is KernelKind.VECTOR and self.variant == "gemm_reduce"
        if reduction:
            if self.launch.algorithm != reducer or "source_gemm_shape" not in self.attributes:
                raise ValueError("legacy BF16 reducer requires its parent GEMM shape")
            parent = self.attributes["source_gemm_shape"]
        elif self.kind is KernelKind.GEMM and self.variant == "gemm" and self.launch.algorithm == main:
            parent = self.shape
        else:
            raise ValueError("legacy BF16 aggregate stage disagrees with its algorithm")
        # This profile is the source-bound ordinary row-major A @ B.T
        # mathematical ABI; those fixed selector fields are not tile facts.
        selection = {"a_layout": "row_major", "b_layout": "row_major", "c_layout": "row_major",
            "trans_a": False, "trans_b": True, "has_bias": False, "epilogue": "none",
            **self.attributes, "launch_policy": "injected_launch"}
        expected = select_legacy_bf16_external_aggregate_launch(parent, self.dtype, "gemm",
            selection, hardware, reducer=reduction)
        if self.launch != expected or reduction and self.attributes["split_k"] != expected.parameters["modeled_split_k"]:
            raise ValueError("legacy BF16 aggregate geometry or explicit model assumption differs from its source row")

    def validate_observed_gemm_launch(self, hardware) -> None:
        """Replay an explicitly bound dense source against the supplied HWSpec."""
        if hardware.hardware_id != self.hardware_id:
            raise ValueError("observed GEMM validation requires matching hardware")
        self._validate_bound_fp16_gemm_launch(hardware)

    def _validate_bound_fp16_gemm_launch(self, hardware) -> None:
        """Exact source replay, including explicitly modeled external stages."""
        from src.kernel.launch_config.nvidia_h200.gemm_launch import (
            select_real_fp16_single_gemm_launch, select_real_fp16_external_gemm_launch)
        single = "h200_vllm030_torch213_fp16_observed_gemm"
        main = "h200_vllm030_torch213_fp16_external_aggregate_gemm"
        reduce = "h200_vllm030_torch213_fp16_external_aggregate_reduce"
        if (self.hardware_id != "nvidia_h200" or self.dtype != "fp16"
                or self.execution_support != "supported" or self.launch is None
                or "source_gemm_runtime" not in self.attributes
                or self.attributes["source_gemm_runtime"] != "vllm_0_30_torch_2_13_cu130_cublas13_1_1_fp16_unquantized"
                or self.launch.tiling is not None or self.launch.source_program is not None):
            raise ValueError("observed FP16 GEMM requires its exact H200 runtime/launch contract")
        attrs = {**self.attributes, "launch_policy": "injected_launch", "model_execution_version": "vllm_0_30_0"}
        if self.kind is KernelKind.GEMM and self.variant == "gemm" and self.launch.algorithm in (single, main):
            selector = select_real_fp16_single_gemm_launch if self.launch.algorithm == single else select_real_fp16_external_gemm_launch
            expected = selector(self.shape, self.dtype, self.variant, attrs, hardware)
        elif (self.kind is KernelKind.VECTOR and self.variant == "gemm_reduce"
                and self.launch.algorithm == reduce and "source_gemm_shape" in self.attributes):
            expected = select_real_fp16_external_gemm_launch(self.attributes["source_gemm_shape"],
                self.dtype, "gemm", attrs, hardware, reducer=True)
            if self.attributes["split_k"] != expected.parameters["modeled_split_k"]:
                raise ValueError("aggregate reducer split count must match its explicit model assumption")
        else:
            raise ValueError("observed FP16 GEMM algorithm/kind disagrees with its source stage")
        if self.launch != expected:
            raise ValueError("observed FP16 GEMM launch disagrees with its actual source row")

    def __post_init__(self) -> None:
        kind = KernelKind.parse(self.kind)
        object.__setattr__(self, "kind", kind)
        from src.kernel.hardware_ids import canonical_hardware_id
        hardware_id = canonical_hardware_id(self.hardware_id)
        object.__setattr__(self, "hardware_id", hardware_id)
        if not isinstance(self.shape, Mapping):
            raise TypeError("KernelSpec.shape must be a mapping")
        if not isinstance(self.attributes, Mapping):
            raise TypeError("KernelSpec.attributes must be a mapping")
        if ("mathematical_program" in self.attributes
                and self.attributes["mathematical_program"] in (
                    "dense_bf16_source_flops_bytes_v1", "unquantized_bf16_flops_bytes_v1",
                    "unquantized_bf16_swiglu_v1", "unquantized_bf16_grouped_swiglu_v1",
                    "unquantized_bf16_legacy_grouped_swiglu_v1")
                and (hardware_id != "nvidia_h200" or self.execution_support != "unsupported" or self.launch is not None)):
            raise ValueError("BF16 source math requires an unbound H200 contract; native launch support is separate")
        if (isinstance(self.anonymous_l2_dirty_bytes, bool)
                or not isinstance(self.anonymous_l2_dirty_bytes, int)):
            raise TypeError(
                "KernelSpec anonymous_l2_dirty_bytes must be an integer")
        if self.anonymous_l2_dirty_bytes < 0:
            raise ValueError(
                "KernelSpec anonymous_l2_dirty_bytes must be nonnegative")
        if not isinstance(self.l2_execution_context, L2ExecutionContext):
            raise TypeError(
                "KernelSpec l2_execution_context must be an "
                "L2ExecutionContext")
        invalid_shape_keys = [
            name for name in self.shape if not isinstance(name, str) or not name
        ]
        if invalid_shape_keys:
            raise TypeError(
                f"shape keys must be non-empty strings: {invalid_shape_keys}")
        invalid_shape_types = {
            name: value for name, value in self.shape.items()
            if isinstance(value, bool) or not isinstance(value, int)
        }
        if invalid_shape_types:
            raise TypeError(
                f"shape dimensions must be integers: {invalid_shape_types}")
        normalized_shape = dict(self.shape)
        if any(v < 0 for v in normalized_shape.values()):
            raise ValueError(f"shape dimensions cannot be negative: {normalized_shape}")
        object.__setattr__(self, "shape", _immutable_mapping(normalized_shape))
        object.__setattr__(self, "attributes", _immutable_mapping(self.attributes))
        if (_is_kda_chunk_source(kind, self.variant, self.attributes)
                and (hardware_id != "nvidia_h200" or self.dtype != "bf16"
                    or self.execution_support != "unsupported"
                    or self.unsupported_reason != "typed_service_unavailable" or self.launch is None)):
            raise ValueError("KDA chunk child requires its exact partial H200 BF16 source launch")
        if self.execution_support not in ("supported", "unsupported"):
            raise ValueError(
                "KernelSpec.execution_support must be supported or unsupported")
        if self.execution_support == "supported":
            if self.unsupported_reason is not None:
                raise ValueError(
                    "supported KernelSpec cannot declare unsupported_reason")
        elif self.execution_support == "unsupported":
            if self.unsupported_reason not in ("launch_selector_unavailable", "typed_service_unavailable"):
                raise ValueError(
                    "unsupported KernelSpec requires unsupported_reason="
                    "'launch_selector_unavailable' or 'typed_service_unavailable'")
            bound_k3_partial_launch = (
                kind is KernelKind.VECTOR and self.hardware_id == "nvidia_h200"
                and self.launch is not None and ((self.variant, self.launch.algorithm) in (
                    ("kimi_k3_kda_decode", "vllm_0_30_sm90_kda_packed_decode"),
                    ("kimi_k3_kda_decode", "vllm_0_30_sm90_kda_gated_rmsnorm"),
                    ("kimi_k3_kda_decode", "vllm_0_30_sm90_kda_causal_conv1d_update"),
                    ("kimi_k3_situ", "vllm_0_30_sm90_kimi_k3_situ_scalar"))
                    or (self.variant in ("kimi_k3_kda_prefill", "kimi_k3_kda_mixed", "kimi_k3_kda_context_prefill")
                        and self.launch.algorithm in ("vllm_0_30_sm90_kda_causal_conv1d_fwd",
                            "vllm_0_30_sm90_kda_gather_initial_states", "vllm_0_30_sm90_kda_gated_rmsnorm"))))
            legacy_router_div = (
                kind is KernelKind.VECTOR and self.variant == "elementwise"
                and self.hardware_id == "nvidia_h200" and self.launch is not None
                and self.launch.algorithm == "pytorch_2_8_sm90_moe_selected_weight_div"
                and "operation" in self.attributes and self.attributes["operation"] == "div")
            if self.unsupported_reason == "typed_service_unavailable":
                if _is_kda_chunk_source(kind, self.variant, self.attributes):
                    _validate_kda_chunk_source_launch(self.variant, self.shape, self.dtype, self.attributes, self.launch)
                elif bound_k3_partial_launch:
                    from src.kernel.launch_config.nvidia_h200.vector_launch import select_vector_launch
                    # These exact algorithms uniquely bind the source wrapper;
                    # state indices, bounds and activation parameters stay real
                    # caller facts. Source-component metadata is selection-only.
                    component = {
                        "kimi_k3_situ": "vllm_0_30_kimi_k3_ordinary_mlp",
                        "kimi_k3_kda_decode": "vllm_0_30_kimi_k3_triton_decode",
                        "kimi_k3_kda_prefill": "vllm_0_30_kimi_k3_triton_fresh_prefill",
                        "kimi_k3_kda_mixed": "vllm_0_30_kimi_k3_triton_mixed",
                        "kimi_k3_kda_context_prefill": "vllm_0_30_kimi_k3_triton_context_prefill",
                    }[self.variant]
                    source = {**self.attributes, "source_component": component}
                    expected = select_vector_launch(self.shape, self.dtype, self.variant, source, None)
                    if self.launch != expected:
                        raise ValueError("K3 partial source requires its exact H200 launch")
                elif legacy_router_div:
                    from src.kernel.launch_config.nvidia_h200.moe_router_launch import select_legacy_router_stage
                    source = {**self.attributes, "model_execution_version": "vllm_0_11_0",
                        "source_moe_router_stage": "normalize_div", "source_router_renormalize": True,
                        "source_input_layout": "contiguous_aligned_16"}
                    expected = select_legacy_router_stage(self.shape, self.dtype, self.variant, source, None)
                    if self.launch != expected:
                        raise ValueError("legacy router division requires its exact H200 broadcast launch")
                elif not (
                        kind is KernelKind.VECTOR and self.variant == "vllm_plain_greedy_sample"
                        and self.hardware_id == "nvidia_h200" and self.launch is not None
                        and "stage" in self.attributes and self.attributes["stage"] == "argmax"):
                    raise ValueError("typed-service refusal requires an exact bound H200 source launch")
        else:
            raise AssertionError(
                f"unhandled execution support: {self.execution_support!r}")

        if (kind is KernelKind.VECTOR and self.variant == "elementwise"
                and "operation" in self.attributes and self.attributes["operation"] == "div"):
            if (self.hardware_id != "nvidia_h200" or self.dtype != "fp32"
                    or self.execution_support != "unsupported"
                    or self.unsupported_reason != "typed_service_unavailable"
                    or self.launch is None
                    or self.launch.algorithm != "pytorch_2_8_sm90_moe_selected_weight_div"):
                raise ValueError("router division binds only its typed-service-unavailable H200 launch")

        compute_kinds = {
            KernelKind.GEMM,
            KernelKind.FLASH_ATTN,
            KernelKind.EAGER_ATTN,
            KernelKind.SPARSE_ATTN_INDEXER,
            KernelKind.SPARSE_ATTN,
            KernelKind.KV_CACHE,
            KernelKind.FUSED_MOE,
            KernelKind.UNFUSED_MOE,
            KernelKind.VECTOR,
        }
        transfer_kinds = {
            KernelKind.COMMUNICATION,
            KernelKind.MEMCPY,
            KernelKind.KV_OFFLOAD,
        }
        if kind in compute_kinds:
            if ((self.launch is not None and self.launch.algorithm in (
                    "h200_cublaslt129_bf16_external_aggregate_gemm",
                    "h200_cublaslt129_bf16_external_aggregate_reduce"))
                    or kind is KernelKind.VECTOR and self.variant == "gemm_reduce" and self.dtype == "bf16"
                    and "source_gemm_shape" in self.attributes and "source_gemm_runtime" not in self.attributes):
                self.validate_legacy_bf16_external_launch(None)
            if ((kind is KernelKind.GEMM or kind is KernelKind.VECTOR and self.variant == "gemm_reduce")
                    and ("source_gemm_runtime" in self.attributes or self.launch is not None
                        and self.launch.algorithm in (
                            "h200_vllm030_torch213_fp16_observed_gemm",
                            "h200_vllm030_torch213_fp16_external_aggregate_gemm",
                            "h200_vllm030_torch213_fp16_external_aggregate_reduce"))):
                self._validate_bound_fp16_gemm_launch(None)
            if (kind is KernelKind.VECTOR and self.variant == "dsv32_fused_norm_rope"
                    and self.attributes["cache_slot_scope"] == "no_metadata_no_cache_slots"
                    and (self.hardware_id not in ("nvidia_a800", "nvidia_h200")
                        or self.launch is not None or self.execution_support != "unsupported")):
                raise ValueError("V32 metadata=None fusion requires its unbound NVIDIA mathematical source")
            if (self.hardware_id == "nvidia_rtx_pro_5500"
                    and (self.launch is not None or self.execution_support != "unsupported")):
                raise ValueError("RTX PRO 5500 parameter scenario requires unbound mathematical execution")
            if self.execution_support == "supported" and self.launch is None:
                raise ValueError(
                    f"supported {kind.value} requires a complete LaunchConfig")
            if (self.execution_support == "unsupported" and self.launch is not None
                    and self.unsupported_reason != "typed_service_unavailable"):
                raise ValueError(
                    f"unsupported {kind.value} cannot contain LaunchConfig")
            if self.transfer is not None:
                raise ValueError(f"{kind.value} cannot contain transfer configuration")
            if (self.dtype is None or not isinstance(self.dtype, str)
                    or not self.dtype.strip()):
                raise ValueError(f"{kind.value} requires dtype")
            if (self.variant is None or not isinstance(self.variant, str)
                    or not self.variant.strip()):
                raise ValueError(f"{kind.value} requires variant")
            normalized_variant = self.variant.strip().lower()
            if (normalized_variant == "projected_attention_path"
                    and (self.hardware_id != "nvidia_h200" or self.execution_support != "unsupported"
                         or self.launch is not None or self.transfer is not None)):
                raise ValueError("projected Attention has only an unbound H200 source contract")
            if (kind is KernelKind.VECTOR and normalized_variant == "elementwise"
                    and "mathematical_program" in self.attributes
                    and self.attributes["mathematical_program"] == "vllm_0_30_fa3_aot_tail_clear"
                    and (self.hardware_id != "nvidia_h200" or self.dtype != "int32"
                        or self.execution_support != "unsupported" or self.launch is not None)):
                raise ValueError("FA3 AOT metadata clear has only an unbound H200 INT32 mathematical contract")
            if kind is KernelKind.VECTOR and normalized_variant == "fa3_prepare_varlen":
                from src.kernel.launch_config.nvidia_h200.flash_attn_launch import select_fa3_prepare_varlen_launch
                if (self.hardware_id != "nvidia_h200" or self.dtype != "int32"
                        or "batch" not in self.shape
                        or self.launch != select_fa3_prepare_varlen_launch(self.shape["batch"])):
                    raise ValueError("FA3 prepare requires its H200 INT32 source launch")
            if (kind is KernelKind.KV_CACHE and normalized_variant == "dsv4_qnorm_rope_kv_insert"
                    and "mathematical_program" in self.attributes
                    and self.attributes["mathematical_program"] == "vllm_0_30_dspark_partial_context_insert_v1"
                    and (self.hardware_id != "nvidia_h200" or self.execution_support != "unsupported"
                        or self.launch is not None)):
                raise ValueError("DSpark partial context insert requires its H200 unbound mathematical contract")
            if (kind is KernelKind.VECTOR and normalized_variant == "vllm_mtp_eager"
                    and (self.hardware_id != ("nvidia_h200" if self.attributes["model_execution_version"] in ("vllm_0_27_1", "vllm_0_30_0", "sglang_0_5_19") else "nvidia_a800") or self.execution_support != "unsupported"
                        or self.launch is not None)):
                raise ValueError("Typed eager operations require their version-bound unbound source contract")
            if (kind is KernelKind.VECTOR and normalized_variant == "vllm_mtp_position_mask"
                    and (self.hardware_id != ("nvidia_h200" if self.attributes["model_execution_version"] == "vllm_0_27_1" else "nvidia_a800") or self.execution_support != "unsupported"
                        or self.launch is not None)):
                raise ValueError("MTP position mask requires its version-bound unbound source contract")
            if (kind is KernelKind.VECTOR and normalized_variant in ("moe_init_routing", "moe_abs", "moe_token_unpermute")
                    and "mathematical_program" in self.attributes
                    and (self.hardware_id != "ascend_910b" or self.dtype != "bf16"
                        or self.execution_support != "unsupported" or self.launch is not None)):
                raise ValueError("Ascend0.18 dispatch math requires BF16 and its unbound Ascend source")
            if (kind is KernelKind.VECTOR and normalized_variant == "moe_topk"
                    and "source_program" in self.attributes
                    and self.attributes["source_program"] in ("vllm_0_11_bf16_sigmoid_bias_grouped_topk", "vllm_0_27_1_bf16_sigmoid_bias_grouped_topk")
                    and self.hardware_id != "nvidia_h200"):
                raise ValueError("DSV3 source grouped top-k requires its H200 mathematical contract")
            if (kind is KernelKind.VECTOR and normalized_variant == "moe_topk"
                    and "source_program" in self.attributes
                    and ((self.hardware_id != "nvidia_h200" and (
                            "mathematical_program" not in self.attributes
                            or self.hardware_id not in ("nvidia_a800", "metax_c500", "ascend_910b")))
                        or self.dtype != ("fp32" if self.attributes["source_program"] in ("vllm_0_11_fp32_sigmoid_bias_topk", "vllm_0_11_fp32_softmax_topk", "sglang_0_5_19_fp32_sigmoid_bias_grouped_topk") else "bf16")
                        or self.execution_support != "unsupported" or self.launch is not None)):
                if self.attributes["source_program"] == "vllm_0_30_bf16_softmax_topk":
                    raise ValueError("source softmax top-k requires BF16 and an unbound H200 source contract")
                raise ValueError("source top-k requires its declared logits dtype and an unbound source contract")
            if (kind is KernelKind.VECTOR and normalized_variant == "moe_topk"
                    and "source_program" in self.attributes
                    and self.attributes["source_program"] == "vllm_0_18_bf16_softmax_topk"
                    and (self.hardware_id != "ascend_910b" or "mathematical_program" not in self.attributes)):
                raise ValueError("0.18 softmax top-k requires its Ascend public mathematical source")
            if (kind is KernelKind.VECTOR and normalized_variant == "moe_topk"
                    and "source_program" in self.attributes
                    and self.attributes["source_program"] == "vllm_0_24_bf16_softmax_topk"
                    and (self.hardware_id != "metax_c500" or "mathematical_program" not in self.attributes)):
                raise ValueError("0.24 softmax top-k requires its C500 public mathematical source")
            if (kind is KernelKind.VECTOR and normalized_variant == "moe_topk"
                    and "source_program" in self.attributes
                    and self.attributes["source_program"] == "vllm_0_11_fp32_softmax_topk"
                    and (self.hardware_id not in ("nvidia_a800", "nvidia_h200") or "mathematical_program" not in self.attributes)):
                raise ValueError("0.11 softmax top-k requires its explicit CUDA FP32 mathematical source")
            if (kind is KernelKind.VECTOR and normalized_variant == "moe_topk"
                    and "source_program" in self.attributes
                    and self.attributes["source_program"] == "vllm_0_11_fp32_sigmoid_bias_topk"
                    and "mathematical_program" not in self.attributes):
                raise ValueError("GLM4 sigmoid top-k requires an explicit mathematical projection")
            if (kind is KernelKind.FLASH_ATTN and "mathematical_program" in self.attributes
                    and (self.execution_support != "unsupported" or self.launch is not None
                         or self.dtype not in ("bf16", "fp16")
                         or (normalized_variant in ("prefill", "decode") and (
                             self.attributes["fa_ksteps"] is not None
                             or self.attributes["fa_eff_mfrac"] != 1.0
                             or (self.attributes["mathematical_program"] == "gqa_softmax_v1"
                                 and "model_input_q_lens" in self.attributes and (
                                 normalized_variant != "decode"
                                 or self.attributes["graph_runtime_mode"] != "FULL"
                                 or any(q != 1 for q in self.attributes["q_lens"]))))))):
                raise ValueError("mathematical attention must not claim an actual launch")
            if (kind is KernelKind.FLASH_ATTN and "mathematical_program" in self.attributes
                    and self.attributes["mathematical_program"] == "ascend_0_18_full_fia_softmax_v1"
                    and (self.hardware_id != "ascend_910b" or self.dtype != "bf16"
                        or self.attributes["causal"] is not True or self.attributes["kv_layout"] != "paged_varlen"
                        or self.attributes["kv_dist"] != "uniform")):
                raise ValueError("Ascend0.18 FULL FIA math requires its unbound BF16 910B source")
            if ((kind is KernelKind.VECTOR and normalized_variant in ("gather", "rmsnorm", "add_rmsnorm", "silu", "rotary")
                    or kind is KernelKind.KV_CACHE and normalized_variant == "paged_kv_cache")
                    and "mathematical_program" in self.attributes
                    and (self.launch is not None or self.execution_support != "unsupported")):
                raise ValueError("logical source math must not claim an actual launch")
            if "packed_weight_storage" in self.attributes:
                from src.kernel.hardware_spec import HWSpec
                if (not HWSpec.source_contract_supported("aggregate_gpu_source_roofline", self.hardware_id)
                        or self.launch is not None or self.execution_support != "unsupported"):
                    raise ValueError("packed vector weights require the explicit unbound aggregate GPU source")
            if (kind is KernelKind.GEMM and "mathematical_program" in self.attributes
                    and (self.launch is not None or self.execution_support != "unsupported")):
                raise ValueError("mathematical GEMM must not claim an actual launch")
            if (kind is KernelKind.GEMM and "mathematical_program" in self.attributes
                    and self.attributes["mathematical_program"] in ("block_fp8_quantized_input_fp32_scales_v1", "batched_fp8_quantized_input_fp32_scales_v1")
                    and self.hardware_id not in ("hygon_bw1100", "nvidia_rtx_pro_5500")):
                raise ValueError("prequantized block-FP8 program is scoped to the declared aggregate GPU design")
            if (kind is KernelKind.GEMM and "mathematical_program" in self.attributes
                    and self.attributes["mathematical_program"] in ("mxfp4_bf16_flops_bytes_v1", "mxfp4_fp8_flops_bytes_v1")
                    and self.hardware_id != "intel_agmf039r47"):
                raise ValueError("MXFP4 plain MM program currently binds only the declared FPGA design")
            if (kind is KernelKind.VECTOR and normalized_variant == "vllm_attention_state_merge"
                    and (self.launch is not None or self.execution_support != "unsupported")):
                raise ValueError("mathematical attention state merge must not claim an actual launch")
            if (kind is KernelKind.FLASH_ATTN and "mathematical_program" in self.attributes
                    and self.attributes["mathematical_program"] == "bf16_mla_context_empty_rows_softmax_v1"
                    and self.hardware_id != "nvidia_h200"):
                raise ValueError("empty-row MLA context currently binds only the declared H200 SGLang source")
            if (kind is KernelKind.FUSED_MOE and "mathematical_program" in self.attributes
                    and self.attributes["mathematical_program"] == "sglang_0_5_19_fp8_fused_shared_swiglu_v1"
                    and self.hardware_id != "nvidia_h200"):
                raise ValueError("SGLang0.5.19 mathematical experts require the declared H200 source contract")
            if (kind is KernelKind.FUSED_MOE and "mathematical_program" in self.attributes
                    and self.attributes["mathematical_program"] == "block_fp8_replicated_ep_swiglu_v1"
                    and self.hardware_id != "nvidia_h200"):
                raise ValueError("EP8 mathematical experts require their H200 source contract")
            if (kind is KernelKind.FUSED_MOE and "mathematical_program" in self.attributes
                    and self.attributes["mathematical_program"] == "kimi_ep_marlin_swiglu_v1"
                    and self.hardware_id not in ("nvidia_a800", "nvidia_h200")):
                raise ValueError("Kimi EP mathematical experts require an unbound NVIDIA source contract")
            if (kind is KernelKind.FUSED_MOE and "mathematical_program" in self.attributes
                    and self.attributes["mathematical_program"] == "kimi_ep_marlin_swiglu_v1"
                    and normalized_shape["intermediate"] in (256, 512, 1024) and self.hardware_id != "nvidia_h200"):
                raise ValueError("Kimi TP-sharded mathematical experts require the H200 source domain")
            if (kind is KernelKind.FUSED_MOE and "mathematical_program" in self.attributes
                    and self.attributes["mathematical_program"] == "dsv4_mxfp4_w4a16_swiglu_v1"
                    and self.hardware_id not in ("hygon_bw1100", "nvidia_rtx_pro_5500")):
                raise ValueError("V4 TP8 mathematical experts require their declared aggregate GPU source scope")
            if (kind is KernelKind.FUSED_MOE and "mathematical_program" in self.attributes
                    and (self.launch is not None or self.execution_support != "unsupported")):
                raise ValueError("mathematical experts must not claim an actual launch")
            if (kind is KernelKind.FLASH_ATTN and normalized_variant == "kimi_k3_nope_context_attention"
                    and (self.execution_support != "unsupported" or self.hardware_id != "nvidia_h200")):
                raise ValueError("K3 context attention requires an unbound H200 source contract")
            if (kind is KernelKind.FUSED_MOE and normalized_variant == "kimi_k3_marlin_experts"
                    and (self.execution_support != "unsupported" or self.hardware_id != "nvidia_h200")):
                raise ValueError("K3 Marlin expert source requires an unbound H200 source contract")
            if (kind is KernelKind.VECTOR and normalized_variant in ("kimi_k3_attn_res", "kimi_k3_situ", "kimi_k3_kda_decode", "kimi_k3_kda_prefill", "kimi_k3_kda_mixed", "kimi_k3_kda_context_prefill", "kimi_k3_nope_prefill", "kimi_k3_nope_mixed", "kimi_k3_nope_context", "kimi_k3_moe_route")
                    and (self.execution_support != "unsupported" or self.hardware_id != "nvidia_h200")):
                raise ValueError("K3 source components require an unbound H200 source contract")
            if kind is KernelKind.VECTOR and normalized_variant == "dsv32_embed_norm" and self.hardware_id not in ("nvidia_a800", "nvidia_h200"):
                raise ValueError("V32 target embedding norm requires a supported NVIDIA source contract")
            if (kind is KernelKind.VECTOR and normalized_variant in (
                    "dsv32_draft_argmax", "dsv32_draft_id_cast", "dsv32_draft_row_select", "dsv32_draft_buffer_copy", "dsv32_draft_token_stack", "dsv32_draft_id_scatter", "dsv32_draft_prepare_next_ids", "dsv32_draft_prepare_sample_indices", "dsv32_draft_seq_rollback", "dsv32_draft_step_metadata")
                    and (self.execution_support != "unsupported" or self.hardware_id != "nvidia_h200")):
                raise ValueError("draft sampling requires an unbound H200 source contract")
            if (kind is KernelKind.VECTOR and normalized_variant == "vllm_rejection_logits_materialize"
                    and (self.execution_support != "unsupported" or self.hardware_id != "nvidia_h200")):
                raise ValueError("rejection materialization requires an unbound H200 source contract")
            if (kind is KernelKind.VECTOR and normalized_variant == "vllm_logits_gather_reorder"
                    and (self.execution_support != "unsupported" or self.hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b", "hygon_bw1100", "nvidia_rtx_pro_5500"))):
                raise ValueError("logits gather reorder requires an unbound CUDA-alike source contract")
            if (kind is KernelKind.VECTOR and normalized_variant == "vllm_masked_embedding"
                    and (self.execution_support != "unsupported" or self.hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b"))):
                raise ValueError("masked embedding requires an unbound CUDA-alike source contract")
            unbound_h200_index = (self.hardware_id == "nvidia_h200"
                and self.launch is None and self.execution_support == "unsupported")
            if (kind is KernelKind.VECTOR and normalized_variant == "gather"
                    and "materialization_source" in self.attributes
                    and self.hardware_id != "metax_c500"):
                raise ValueError("Maca materialization source requires C500")
            unbound_a800_index = False
            if (kind is KernelKind.VECTOR and normalized_variant == "gather"
                    and "index_select_source" in self.attributes and self.hardware_id == "nvidia_a800"
                    and self.launch is None and self.execution_support == "unsupported"):
                from src.kernel.launch_config.nvidia_a800.vector_launch import pytorch_embedding_index_select_launch_domain
                physical = self.attributes["model_input_shape"] if "model_input_shape" in self.attributes else self.shape
                unbound_a800_index = not pytorch_embedding_index_select_launch_domain(
                    physical["rows"], physical["hidden"], physical["source_rows"], physical["element_count"])
            metax024_source = (
                ("model_execution_version" in self.attributes and self.attributes["model_execution_version"] == "vllm_0_24_0")
                or ("sampling_source" in self.attributes and self.attributes["sampling_source"] in (
                    "vllm_0_24_plain_sampler_bf16", "vllm_0_24_plain_greedy_bf16"))
                or ("embedding_source" in self.attributes and self.attributes["embedding_source"] == "vllm_0_24_unquantized_tp1")
                or ("mask_program" in self.attributes and self.attributes["mask_program"] == "vllm_0_24"))
            ascend018_source = (
                ("model_execution_version" in self.attributes and self.attributes["model_execution_version"] == "vllm_0_18_0")
                or ("sampling_source" in self.attributes and self.attributes["sampling_source"] in (
                    "vllm_0_18_plain_sampler_bf16", "vllm_0_18_plain_greedy_bf16"))
                or ("embedding_source" in self.attributes and self.attributes["embedding_source"] == "vllm_0_18_unquantized_tp1"))
            if ascend018_source and (self.hardware_id != "ascend_910b" or self.launch is not None
                    or self.execution_support != "unsupported"):
                raise ValueError("vLLM0.18 source requires its unbound Ascend hardware identity")
            if (ascend018_source and normalized_variant in ("vllm_logits_row_select", "vllm_logits_gather_reorder")
                    and self.dtype != "bf16"):
                raise ValueError("vLLM0.18 ordinary logits boundary requires BF16")
            if metax024_source and self.hardware_id != "metax_c500":
                raise ValueError("vLLM0.24 source requires its C500 hardware identity")
            if (kind is KernelKind.VECTOR and normalized_variant == "gather" and "index_select_source" in self.attributes
                    and not unbound_h200_index and not unbound_a800_index):
                physical = (self.attributes["model_input_shape"] if "model_input_shape" in self.attributes else self.shape)
                rows, hidden, vocab = (physical[key] for key in ("rows", "hidden", "source_rows"))
                small = rows <= 16
                if self.hardware_id == "nvidia_h200":
                    from src.kernel.launch_config.nvidia_h200.vector_launch import select_vector_launch
                    if self.launch != select_vector_launch(physical, self.dtype, normalized_variant, self.attributes, None):
                        raise ValueError("embedding index_select requires its exact H200 Torch2.8 SM90 launch")
                elif (self.hardware_id != "nvidia_a800" or self.launch is None
                        or rows <= 0 or vocab <= 0 or hidden <= 0 or hidden % 128
                        or hidden > 108 * 8 * 128
                        or max(2 * vocab * hidden, 2 * rows * hidden, 8 * rows) >= 2**31
                        or self.launch.algorithm != ("pytorch_2_8_sm80_embedding_small_index" if small else "pytorch_2_8_sm80_vectorized_row_gather")
                        or self.launch.grid != ((hidden // 128, 1, 1) if small else (rows, (2 * hidden + 4095) // 4096, 1))
                        or self.launch.block != ((128, 1, 1) if small else (min(256, 32 * ((2 * hidden + 511) // 512)), 1, 1))
                        or self.launch.tiling is not None or self.launch.cluster is not None
                        or self.launch.source_program is not None
                        or dict(self.launch.parameters) != {"registers_per_thread": 24.0,
                                                            "values_per_thread": 1.0 if small else 8.0}):
                    raise ValueError("embedding index_select requires its exact A800 Torch 2.8 launch")
            if (kind is KernelKind.VECTOR and normalized_variant == "elementwise"
                    and self.attributes["operation"] == "cast_int32_int64"):
                elements = self.shape["element_count"]
                if "tensor_iterator_source" not in self.attributes:
                    raise ValueError("INT32-to-INT64 cast requires its explicit Torch source or logical ATen source")
                logical_aten_cast = self.attributes["tensor_iterator_source"] == "aten_advanced_indexing_long"
                maca_cast = self.attributes["tensor_iterator_source"] == "pytorch_2_4_maca_3_7_2"
                sm90_cast = self.attributes["tensor_iterator_source"] == "pytorch_2_8_sm90"
                if sm90_cast:
                    from src.kernel.launch_config.nvidia_h200.vector_launch import pytorch_int32_int64_cast_launch
                    if (self.hardware_id != "nvidia_h200" or self.launch !=
                            pytorch_int32_int64_cast_launch(elements, normalized_variant)):
                        raise ValueError("integer cast requires its exact H200 Torch2.8 SM90 launch")
                if maca_cast and (self.hardware_id != "metax_c500"
                        or not _matches_maca_integer_cast_launch(self.launch, elements, self.attributes, normalized_variant)):
                    raise ValueError("integer cast requires its exact C500 source-selected Torch launch")
                if logical_aten_cast and not (self.hardware_id in ("nvidia_h200", "metax_c500")
                        and self.launch is None and self.execution_support == "unsupported"):
                    raise ValueError("logical ATen integer cast requires an unbound H200 or C500 source")
                if not logical_aten_cast and not maca_cast and not sm90_cast and (self.hardware_id != "nvidia_a800" or self.launch is None
                        or elements <= 0 or 8 * (elements - 1) > 2**31 - 1
                        or self.launch.algorithm != "pytorch_2_8_sm80_int32_to_int64"
                        or self.launch.grid != ((elements + 511) // 512, 1, 1)
                        or self.launch.block != (128, 1, 1)
                        or self.launch.tiling is not None or self.launch.cluster is not None
                        or self.launch.source_program is not None
                        or self.launch.source_program is not None
                        or dict(self.launch.parameters) != {"registers_per_thread": 32.0, "values_per_thread": 4.0}):
                    raise ValueError("integer cast requires its exact A800 Torch 2.8 launch")
            if (kind is KernelKind.VECTOR and normalized_variant == "vllm_embedding_id_widen"
                    and (self.launch is not None or self.execution_support != "unsupported")):
                if self.hardware_id == "metax_c500":
                    if (self.attributes["embedding_source"] != "vllm_0_24_unquantized_tp1"
                            or not _matches_maca_integer_cast_launch(self.launch, self.shape["rows"], self.attributes, normalized_variant)):
                        raise ValueError("embedding ID widen requires its exact C500 source-selected Torch launch")
                elif self.hardware_id == "nvidia_h200":
                    from src.kernel.launch_config.nvidia_h200.vector_launch import pytorch_int32_int64_cast_launch
                    if (self.attributes["embedding_source"] != "vllm_0_11_unquantized_tp1"
                            or self.launch != pytorch_int32_int64_cast_launch(self.shape["rows"], normalized_variant)):
                        raise ValueError("embedding ID widen requires its exact H200 Torch2.8 SM90 launch")
                elif (self.hardware_id != "nvidia_a800"
                        or self.attributes["embedding_source"] != "vllm_0_11_unquantized_tp1"
                        or self.launch is None
                        or 8 * (self.shape["rows"] - 1) > 2**31 - 1
                        or self.launch.algorithm != "pytorch_2_8_sm80_int32_to_int64"
                        or self.launch.grid != ((self.shape["rows"] + 511) // 512, 1, 1)
                        or self.launch.block != (128, 1, 1)
                        or self.launch.tiling is not None
                        or self.launch.cluster is not None
                        or self.launch.source_program is not None
                        or dict(self.launch.parameters) != {
                            "registers_per_thread": 32.0, "values_per_thread": 4.0}):
                    raise ValueError("embedding ID widen requires its exact A800 Torch 2.8 cast launch")
            if (kind is KernelKind.VECTOR and normalized_variant == "vllm_plain_greedy_sample"
                    and {"source_num_compute_units", "source_max_threads_per_unit"} & set(self.attributes)
                    and not (self.hardware_id == "nvidia_h200" and "stage" in self.attributes
                        and self.attributes["stage"] == "argmax"
                        and self.launch is not None and self.execution_support == "unsupported"
                        and self.unsupported_reason == "typed_service_unavailable")):
                raise ValueError("argmax source properties require its exact H200 launch and unbound typed-service refusal")
            if (kind is KernelKind.VECTOR and normalized_variant == "vllm_plain_greedy_sample"
                    and (self.launch is not None or self.execution_support != "unsupported")):
                if self.hardware_id == "nvidia_h200" and self.attributes["stage"] == "argmax":
                    from src.kernel.launch_config.nvidia_h200.vector_launch import pytorch_argmax_source_launch
                    if (self.execution_support != "unsupported" or self.unsupported_reason != "typed_service_unavailable"
                            or self.launch != pytorch_argmax_source_launch(self.shape, self.dtype, self.attributes)):
                        raise ValueError("argmax requires its exact H200 launch and unbound typed-service refusal")
                elif self.hardware_id == "nvidia_h200" and self.attributes["stage"] == "output_int32":
                    from src.kernel.launch_config.nvidia_h200.vector_launch import pytorch_id_narrow_launch
                    if self.launch != pytorch_id_narrow_launch(self.shape, self.dtype, self.attributes):
                        raise ValueError("greedy output cast requires its exact H200 source launch")
                elif self.hardware_id == "nvidia_a800" and self.attributes["stage"] == "output_int32":
                    from src.kernel.launch_config.nvidia_a800.vector_launch import pytorch_id_narrow_launch
                    if self.launch != pytorch_id_narrow_launch(self.shape, self.dtype, self.attributes):
                        raise ValueError("greedy output cast requires its exact A800 source launch")
                elif self.hardware_id == "metax_c500" and self.attributes["stage"] == "output_int32":
                    from src.kernel.launch_config.metax_c500.vector_launch import pytorch_small_id_narrow_launch
                    if self.launch != pytorch_small_id_narrow_launch(self.shape, self.dtype, self.attributes):
                        raise ValueError("greedy output cast requires its exact C500 small-iterator launch")
                else:
                    raise ValueError("plain greedy sampling requires an unbound or exact C500 ID-narrowing source")
            if kind is KernelKind.VECTOR and normalized_variant == "vllm_logits_fp32_cast":
                if self.hardware_id == "nvidia_a800" and (self.launch is not None or self.execution_support != "unsupported"):
                    from src.kernel.launch_config.nvidia_a800.vector_launch import pytorch_logits_cast_launch
                    if self.launch != pytorch_logits_cast_launch(self.shape, self.dtype, self.attributes):
                        raise ValueError("target logits cast requires its exact A800 source launch")
                if self.hardware_id == "nvidia_h200" and (self.launch is not None or self.execution_support != "unsupported"):
                    from src.kernel.launch_config.nvidia_h200.vector_launch import pytorch_logits_cast_launch
                    if self.launch != pytorch_logits_cast_launch(self.shape, self.dtype, self.attributes):
                        raise ValueError("target logits cast requires its exact H200 source launch")
                small_maca_cast = (self.hardware_id == "metax_c500"
                    and self.attributes["sampling_source"] == "vllm_0_24_plain_sampler_bf16"
                    and "copy_input_contiguous" in self.attributes
                    and self.attributes["copy_input_contiguous"] is True
                    and 0 < self.shape["element_count"] < 2048)
                if "copy_dispatch" in self.attributes or small_maca_cast:
                    from src.kernel.launch_config.metax_c500.vector_launch import pytorch_logits_cast_launch
                    if self.hardware_id != "metax_c500":
                        raise ValueError("copy dispatch requires its C500 source identity")
                    expected = pytorch_logits_cast_launch(self.shape, self.dtype, self.attributes)
                    if self.launch is not None and self.launch != expected:
                        raise ValueError("target logits cast requires its exact C500 source launch")
                    if self.launch is None and self.execution_support != "unsupported":
                        raise ValueError("target logits cast requires its exact C500 source launch")
                elif (self.execution_support != "unsupported" and self.hardware_id not in ("nvidia_a800", "nvidia_h200")) or self.hardware_id not in (
                        "nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b", "hygon_bw1100", "nvidia_rtx_pro_5500"):
                    raise ValueError("target logits cast requires an unbound supported source contract")
            if (kind is KernelKind.VECTOR and normalized_variant == "moe_sum"
                    and "routed_weights_dtype" in self.attributes
                    and self.attributes["routed_weights_dtype"] is not None
                    and (self.hardware_id not in ("hygon_bw1100", "nvidia_rtx_pro_5500") or self.execution_support != "unsupported")):
                raise ValueError("weighted moe_sum currently binds only the declared aggregate GPU logical Roofline")
            if (kind is KernelKind.VECTOR and normalized_variant in ("vllm_logits_row_select", "vllm_logits_gather_reorder")
                    and "model_execution_version" in self.attributes
                    and self.attributes["model_execution_version"] == "vllm_0_27_1"
                    and self.hardware_id != "nvidia_h200"):
                raise ValueError("DSV3 logits boundary requires its declared H200 source contract")
            if (kind is KernelKind.VECTOR and normalized_variant == "vllm_logits_row_select"
                    and self.hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b", "hygon_bw1100", "nvidia_rtx_pro_5500")):
                raise ValueError("target logits row selection requires a supported source contract")
            if (kind is KernelKind.VECTOR and normalized_variant == "vllm_logits_row_select"
                    and "disable_index_broadcast" in self.attributes and self.hardware_id != "metax_c500"):
                raise ValueError("index broadcast dispatch requires its C500 hardware identity")
            if (kind is KernelKind.VECTOR and normalized_variant == "vllm_logits_row_select"
                    and (self.launch is not None or self.execution_support != "unsupported")):
                rows, hidden, source_rows = (self.shape[key] for key in ("rows", "hidden", "source_rows"))
                if self.hardware_id == "metax_c500":
                    from src.kernel.launch_config.metax_c500.vector_launch import pytorch_scalar_index_launch
                    expected = pytorch_scalar_index_launch(self.shape, self.dtype, self.attributes)
                    if self.launch != expected:
                        raise ValueError("target row selection requires its exact C500 scalar indexing launch")
                elif self.hardware_id == "nvidia_h200":
                    from src.kernel.launch_config.nvidia_h200.vector_launch import select_vector_launch
                    if self.launch != select_vector_launch(self.shape, self.dtype, normalized_variant, self.attributes, None):
                        raise ValueError("target row selection requires its exact H200 Torch2.8 SM90 launch")
                elif (self.hardware_id != "nvidia_a800" or self.launch is None
                        or self.attributes["model_execution_version"] != "vllm_0_11_0"
                        or self.attributes["index_source"] != "query_start_loc_int32_to_int64"
                        or self.dtype != "bf16" or hidden % 128
                        or max(2 * source_rows * hidden, 2 * rows * hidden, 8 * rows) >= 2**31
                        or self.launch.algorithm != "pytorch_2_8_sm80_vectorized_row_gather"
                        or self.launch.grid != (rows, (2 * hidden + 4095) // 4096, 1)
                        or self.launch.block != (min(256, 32 * ((2 * hidden + 511) // 512)), 1, 1)
                        or self.launch.tiling is not None or self.launch.cluster is not None
                        or self.launch.source_program is not None
                        or dict(self.launch.parameters) != {"registers_per_thread": 24.0, "values_per_thread": 8.0}):
                    raise ValueError("target row selection requires its exact A8000.11 vectorized launch")
            if (kind is KernelKind.VECTOR and self.attributes is not None and "mathematical_program" in self.attributes
                    and self.attributes["mathematical_program"] == "vllm_0_30_dspark_typed_role_bytes_v1"
                    and (self.hardware_id != "nvidia_h200" or self.launch is not None
                        or self.execution_support != "unsupported")):
                raise ValueError("DSpark typed payload is an unbound H200 mathematical contract")
            if (kind is KernelKind.VECTOR and normalized_variant == "vocab_parallel_embedding"
                    and (self.execution_support != "unsupported" or self.hardware_id not in ("nvidia_a800", "nvidia_h200", "hygon_bw1100", "nvidia_rtx_pro_5500"))):
                raise ValueError("vocab embedding requires an unbound NVIDIA CUDA source contract")
            if (kind is KernelKind.VECTOR and normalized_variant in ("dsv32_moe_route", "kimi_moe_route")
                    and (self.execution_support != "unsupported" or (self.hardware_id != "nvidia_h200"
                        and not (self.hardware_id == "nvidia_a800" and "source_projection_scope" in self.attributes
                            and self.attributes["source_projection_scope"] in ("a800_tp1_target_representative_layers_v1", "a800_tp8_pp4_ep8_glm_source_v1", "a800_tp8_pp2_ep8_kimi_source_v1", "a800_tp1_kimi_source_v1", "vllm_0_30_glm_pcp_source_v1"))))):
                source_model = "GLM V32" if normalized_variant == "dsv32_moe_route" else "Kimi"
                raise ValueError(f"{source_model} route requires an unbound H200 source contract")
            if (kind is KernelKind.VECTOR and normalized_variant in ("dsv32_moe_route", "kimi_moe_route")
                    and "source_projection_scope" in self.attributes
                    and (self.hardware_id != "nvidia_a800" or self.launch is not None or self.execution_support != "unsupported")):
                raise ValueError("A800 structural routing cannot claim an actual launch")
            if (kind is KernelKind.VECTOR and normalized_variant == "dsv4_moe_route"
                    and (self.hardware_id not in ("nvidia_h200", "hygon_bw1100", "nvidia_rtx_pro_5500")
                         or (self.hardware_id in ("hygon_bw1100", "nvidia_rtx_pro_5500") and (self.launch is not None or self.execution_support != "unsupported"))
                         or (self.execution_support == "supported"
                             and ("routing_source" not in self.attributes
                                  or self.attributes["routing_source"] is None)))):
                raise ValueError("V4 route requires an unbound H200 source contract or explicit routing source")
            if (kind is KernelKind.VECTOR and normalized_variant in ("mhc_pre", "mhc_post_pre")
                    and (self.launch is not None or self.execution_support != "unsupported")):
                raise ValueError("mHC mathematical wrapper has no native launch or split claim")
            if (kind is KernelKind.VECTOR
                    and normalized_variant in ("mhc_prenorm", "mhc_prenorm_broadcast")
                    and self.execution_support != "unsupported"):
                raise ValueError("mHC DeepGEMM prenorm requires an unbound launch contract")
            if self.launch is not None:
                if self.launch.execution_variant is None:
                    raise ValueError(
                        f"{kind.value} launch requires execution_variant")
                if self.launch.execution_variant != normalized_variant:
                    raise ValueError(
                        f"{kind.value} variant {normalized_variant!r} does not "
                        "match launch execution_variant "
                        f"{self.launch.execution_variant!r}")
            object.__setattr__(self, "dtype", self.dtype.strip().lower())
            object.__setattr__(self, "variant", normalized_variant)
            normalized_attributes = None
            if kind is KernelKind.VECTOR and normalized_variant == "rope":
                # Typed RoPE element work depends on the validated key-presence
                # contract.  Validate that closed attribute schema before
                # deriving the canonical active-element count; all other
                # variants retain the established shape-first boundary.
                normalized_attributes = _normalize_compute_attributes(
                    kind, normalized_variant, self.attributes,
                    normalized_shape, self.dtype.strip().lower())
            normalized_shape = _normalize_compute_shape(
                kind, normalized_variant, self.dtype.strip().lower(),
                normalized_shape,
                (normalized_attributes
                 if normalized_attributes is not None else self.attributes))
            zero_allowed = (
                kind is KernelKind.SPARSE_ATTN
                and normalized_variant in (
                    "dsv4_sparse_prefill", "dsv4_sparse_decode")
                # SWA-only layers never select; C128 layers with
                # kv_len < 128 have no compressed state yet
                # (sparse_mla.py:404 num_compressed == 0).
                and (normalized_shape["compress_ratio"] in (1, 128)
                     or (normalized_shape["compress_ratio"] == 4
                         and "compressed_selection" in self.attributes
                         and self.attributes["compressed_selection"] in ("all_completed_states", "declared_causal_topk"))))
            zero_indexer_topk_allowed = (
                kind is KernelKind.SPARSE_ATTN_INDEXER
                and normalized_variant == "dsv4_indexer_prefill"
                and normalized_shape["compress_ratio"] == 4
                and "context_parallel_size" in self.attributes
                and type(self.attributes["context_parallel_size"]) is int
                and self.attributes["context_parallel_size"] == 1
                and normalized_shape["kv_len"] // normalized_shape["compress_ratio"] == 0)
            zero_element_count_allowed = (
                kind is KernelKind.KV_CACHE
                and normalized_variant == "dsv4_compressed_state_cache")
            sparse_zero_kv_allowed = (kind is KernelKind.FLASH_ATTN
                and normalized_variant in ("mla_decode", "combine")
                and "source_program" in self.attributes
                and self.attributes["source_program"] == "vllm_0_30_fa3_sparse_mla")
            if any(value <= 0 for name, value in normalized_shape.items()
                   if not (sparse_zero_kv_allowed and name == "kv_len" and value == 0)
                   and not ((zero_allowed or zero_indexer_topk_allowed) and name == "topk")
                   and not (zero_element_count_allowed
                            and name == "element_count")):
                raise ValueError(
                    f"compute shape dimensions must be positive: "
                    f"{normalized_shape}")
            _require_shape_fields(kind, normalized_variant, normalized_shape)
            if (kind in (KernelKind.FLASH_ATTN, KernelKind.EAGER_ATTN)
                    and not (
                        kind is KernelKind.FLASH_ATTN
                        and normalized_variant in (
                            "mla_prefill", "mla_decode",
                            "mla_absorbed_prefill", "kimi_k3_nope_context_attention"))):
                query_heads = normalized_shape["num_query_heads"]
                kv_heads = normalized_shape["num_kv_heads"]
                if query_heads < kv_heads or query_heads % kv_heads:
                    raise ValueError(
                        f"{kind.value} num_query_heads must be an integer "
                        "multiple of num_kv_heads")
            if (kind in (KernelKind.FUSED_MOE, KernelKind.UNFUSED_MOE)
                    and normalized_shape["topk"] > normalized_shape["experts"]):
                raise ValueError(f"{kind.value} topk cannot exceed experts")
            if normalized_attributes is None:
                normalized_attributes = _normalize_compute_attributes(
                    kind, normalized_variant, self.attributes,
                    normalized_shape, self.dtype.strip().lower())
            legacy_vector_variants = ("rmsnorm", "add_rmsnorm", "silu", "rope")
            from src.kernel.launch_config.nvidia_h200.source_vector_launch import (
                LEGACY_FUSED_NORM_GEOMETRY, legacy_fused_norm_geometry)
            if self.launch is not None and self.launch.algorithm == LEGACY_FUSED_NORM_GEOMETRY:
                if (self.hardware_id != "nvidia_h200" or kind is not KernelKind.VECTOR
                        or normalized_variant != "add_rmsnorm"
                        or self.launch != legacy_fused_norm_geometry(normalized_shape, self.dtype)):
                    raise ValueError("legacy fused norm geometry requires its exact H200 source launch")
            legacy_moe_variants = ("moe_topk", "moe_align", "moe_align_small_batch",
                                   "moe_count_and_sort", "moe_act_and_mul", "moe_sum")
            if (self.launch is not None and self.launch.algorithm in (
                    *(f"vllm_0_11_sm90_source_{variant}" for variant in legacy_vector_variants),
                    *(f"vllm_0_11_sm90_source_{variant}" for variant in legacy_moe_variants),
                    "vllm_0_11_sm90_source_kv_cache")):
                if self.hardware_id != "nvidia_h200":
                    raise ValueError("SM90 compiled source resources require H200 hardware")
                source_attributes = {"model_execution_version": "vllm_0_11_0",
                                     "source_input_layout": "contiguous_aligned_16"}
                if (kind is KernelKind.VECTOR and normalized_variant in legacy_vector_variants
                        and self.launch.algorithm == f"vllm_0_11_sm90_source_{normalized_variant}"):
                    from src.kernel.launch_config.nvidia_h200.legacy_vector_resources import select_legacy_vector_resources
                    if normalized_variant == "rope":
                        source_attributes["is_neox"] = normalized_attributes["is_neox"]
                    expected_resources = select_legacy_vector_resources(normalized_shape, self.dtype,
                        normalized_variant, source_attributes)
                elif (kind is KernelKind.KV_CACHE and normalized_variant == "paged_kv_cache"
                        and self.launch.algorithm == "vllm_0_11_sm90_source_kv_cache"):
                    from src.kernel.launch_config.nvidia_h200.legacy_vector_resources import select_legacy_vector_resources
                    source_attributes["kv_cache_dtype"] = normalized_attributes["kv_cache_dtype"]
                    expected_resources = select_legacy_vector_resources(normalized_shape, self.dtype,
                        normalized_variant, source_attributes)
                elif (kind is KernelKind.VECTOR and normalized_variant in legacy_moe_variants
                        and self.launch.algorithm == f"vllm_0_11_sm90_source_{normalized_variant}"):
                    from src.kernel.launch_config.nvidia_h200.legacy_vector_resources import select_legacy_moe_aux_resources
                    # This exact0.11 program binds FP32 softmax logits/INT32
                    # routed IDs. The sum's work fraction remains caller-owned.
                    source_attributes.update({"source_topk_id_dtype": "int32",
                        "source_routing": "softmax", "routing_input_dtype": "fp32"})
                    if normalized_variant == "moe_sum":
                        source_attributes["work_fraction"] = normalized_attributes["work_fraction"]
                    expected_resources = select_legacy_moe_aux_resources(normalized_shape, self.dtype,
                        normalized_variant, source_attributes)
                else:
                    raise ValueError("SM90 compiled source algorithm disagrees with kernel kind/variant")
                if dict(self.launch.parameters) != expected_resources:
                    raise ValueError("SM90 resource fields disagree with the fixed compiled source")
            if (kind is KernelKind.FLASH_ATTN and "source_program" in normalized_attributes
                    and normalized_attributes["source_program"] == "vllm_0_30_fa3_sparse_mla"):
                if (self.hardware_id != "nvidia_h200" or self.dtype != "bf16"
                        or self.execution_support != "supported" or self.launch is None
                        or normalized_variant not in ("mla_decode", "combine")
                        or self.launch.algorithm != ("vllm_030_fa3_sm90_sparse_mla_partial"
                            if normalized_variant == "mla_decode" else "vllm_030_fa3_sm90_sparse_mla_combine_partial")
                        or self.launch.tiling is None
                        or "kv_page" not in self.launch.tiling.tile_shape
                        or self.launch.tiling.tile_shape["kv_page"] != 1
                        or "kv_capacity" not in self.launch.tiling.tile_shape
                        or self.launch.tiling.tile_shape["kv_capacity"] < normalized_shape["kv_len"]
                        or self.launch.source_program is not None):
                    raise ValueError("sparse FA3 source requires its H200 page1 library launch")
            if (kind is KernelKind.FLASH_ATTN and "source_program" in normalized_attributes
                    and normalized_attributes["source_program"] == "ascend_0_18_full_fia_split_fuse"):
                if (self.hardware_id != "ascend_910b" or self.launch is None
                        or self.execution_support != "supported"
                        or self.launch.algorithm != ("cann_flash_attention_score:5000000000000200203"
                            if normalized_attributes["ascend_fia_prefill_no_cache"]
                            else "cann_flash_attention_score:5000000000010200203")
                        or self.launch.grid != (24, 1, 1)
                        or self.launch.source_program is not None
                        or "graph_runtime_mode" not in normalized_attributes
                        or normalized_attributes["graph_runtime_mode"] != "FULL"
                        or (normalized_attributes["kv_layout"] != "contiguous"
                            if normalized_attributes["ascend_fia_prefill_no_cache"]
                            else normalized_attributes["kv_layout"] not in ("paged", "paged_varlen"))):
                    raise ValueError("FULL FIA SplitFuse requires its source TND binding")
            if (kind is KernelKind.FLASH_ATTN and normalized_variant == "combine"
                    and "return_softmax_lse" in normalized_attributes):
                if (self.hardware_id != "nvidia_h200" or self.dtype != "bf16"
                        or self.execution_support != "supported" or self.launch is None
                        or self.launch.algorithm != "vllm_fa3_sm90_mla_prefill_combine_partial"
                        or self.launch.tiling is None or self.launch.tiling.num_kv_splits <= 1
                        or self.launch.block != (256, 1, 1)
                        or dict(self.launch.tiling.tile_shape) != {"rows": 8, "value_head": 128}
                        or self.launch.tiling.pipeline_depth != 4):
                    raise ValueError("final split-prefill LSE requires its BF16 H200 FA3 source combine")
            if (kind is KernelKind.FLASH_ATTN and normalized_variant == "mla_prefill"
                    and "return_softmax_lse" in normalized_attributes and normalized_attributes["return_softmax_lse"]
                    and "mathematical_program" not in normalized_attributes
                    and (self.hardware_id != "nvidia_h200" or self.launch is None
                         or self.launch.algorithm != "vllm_fa3_sm90_mla_prefill")):
                raise ValueError("explicit MLA LSE output requires the H200 FA3 analytical source contract")
            if (kind is KernelKind.GEMM
                    and normalized_attributes["output_dtype"] is not None
                    and normalized_attributes["output_dtype"] != self.dtype
                    and self.hardware_id != "nvidia_h200"
                    and not (self.hardware_id == "intel_agmf039r47"
                        and "mathematical_program" in normalized_attributes
                        and normalized_attributes["mathematical_program"] in ("mxfp4_fp8_flops_bytes_v1", "packed_fp4_source_v1"))
                    and not (self.hardware_id in ("hygon_bw1100", "nvidia_rtx_pro_5500") and self.launch is None
                        and self.execution_support == "unsupported")):
                if (self.hardware_id != "nvidia_a800" or self.launch is not None
                        or self.execution_support != "unsupported"
                        or "source_projection_scope" not in normalized_attributes
                        or normalized_attributes["source_projection_scope"] not in ("a800_tp1_target_representative_layers_v1", "a800_tp8_pp4_ep8_glm_source_v1", "vllm_0_30_glm_pcp_source_v1")):
                    raise ValueError("mixed-output GEMM requires H200 or explicit unbound A800 structural scope")
            if (kind is KernelKind.GEMM and "source_projection_scope" in normalized_attributes
                    and (self.hardware_id != "nvidia_a800" or self.launch is not None
                         or self.execution_support != "unsupported")):
                raise ValueError("A800 target structural GEMM cannot claim an actual launch")
            if (kind is KernelKind.VECTOR and normalized_variant == "fp8_group_quant"
                    and normalized_attributes["input_resource"] == "buffer"
                    and (self.hardware_id != "intel_agmf039r47" or self.launch is not None
                        or self.execution_support != "unsupported")):
                raise ValueError("FP8 A-buffer input requires unbound FPGA execution")
            if (kind is KernelKind.VECTOR and normalized_variant == "fp8_group_quant"
                    and normalized_attributes["source_program"] is not None):
                if self.hardware_id != "nvidia_h200" or self.execution_support != "unsupported" or self.launch is not None:
                    raise ValueError("conditional FP8 quantizer source requires unbound H200 execution")
            if kind is KernelKind.VECTOR and normalized_variant == "dsv4_indexer_q_rope_quant":
                if (self.hardware_id not in ("nvidia_h200", "hygon_bw1100", "nvidia_rtx_pro_5500") or self.execution_support != "unsupported" or self.launch is not None):
                    raise ValueError("V4 indexer Q source projection requires unbound H200 execution")
            if kind is KernelKind.VECTOR and normalized_variant == "dsv4_moe_route":
                source_bound = normalized_attributes["routing_source"] is not None
                if source_bound != (self.execution_support == "supported"):
                    raise ValueError("unbound H200 route must remain unsupported; source routing requires a launch")
                if source_bound:
                    hashed = normalized_attributes["expert_id_selection"] == "hash_table"
                    expected_algorithm = ("vllm_dsv4_hash_top6_cuda_source_projection" if hashed
                                          else "vllm_dsv4_top6_triton_source_projection")
                    expected_grid = (-(-normalized_shape["rows"] // 4) if hashed
                                     else normalized_shape["rows"], 1, 1)
                    if (self.launch.algorithm != expected_algorithm
                            or self.launch.grid != expected_grid
                            or self.launch.block != (128 if hashed else 32, 1, 1)):
                        raise ValueError("V4 routing launch disagrees with source geometry")
            if (kind is KernelKind.VECTOR
                    and not _is_kda_chunk_source(kind, normalized_variant, normalized_attributes)
                    and self.launch is not None and self.launch.algorithm in (
                    "pytorch_2_13_sm90_fill_bf16", "pytorch_2_13_sm90_fill_fp16",
                    "pytorch_2_13_sm90_fill_fp32", "pytorch_2_13_sm90_fill_int32",
                    "pytorch_2_13_sm90_fill_int64")):
                elements = normalized_shape["element_count"]
                if (self.hardware_id != "nvidia_h200" or normalized_variant != "elementwise"
                        or self.launch.algorithm != f"pytorch_2_13_sm90_fill_{self.dtype.strip().lower()}"
                        or normalized_attributes["operation"] != "fill"
                        or normalized_attributes["reads"] != 0 or normalized_attributes["writes"] != 1
                        or elements >= 2**31
                        or normalized_shape["dtype_bytes"] * (elements - 1) >= 2**31 - 1
                        or self.launch.grid != (-(-elements // 1024), 1, 1)
                        or self.launch.block != (128, 1, 1)
                        or self.launch.tiling is not None or self.launch.cluster is not None
                        or dict(self.launch.parameters) != {"dynamic_shared_bytes": 0.0}):
                    raise ValueError("Torch2.13 SM90 fill requires its exact H200 source launch and complete write")
            _validate_compute_dtype_contract(
                kind,
                normalized_variant,
                self.dtype.strip().lower(),
                normalized_shape,
                normalized_attributes,
            )
            if (kind is KernelKind.VECTOR and normalized_variant == "vllm_masked_embedding"
                    and normalized_attributes["mask_program"] == "vllm_0_27_1"
                    and (self.hardware_id != "nvidia_h200"
                        or normalized_attributes["tensor_parallel_size"] not in (4, 8)
                        or (normalized_attributes["stage"] == "mask_output" and self.dtype != "bf16"))):
                raise ValueError("vLLM0.27.1 masked embedding requires its unbound H200 BF16 TP4/8 source")
            if (kind is KernelKind.SPARSE_ATTN_INDEXER
                    and normalized_variant in ("dsa_indexer_prefill", "dsa_indexer_decode", "dsa_indexer_mixed")
                    and normalized_attributes["mathematical_program"] is not None
                    and (self.launch is not None or self.execution_support != "unsupported")):
                raise ValueError("DSA mathematical indexer requires an unbound launch contract")
            if (kind is KernelKind.SPARSE_ATTN and "mathematical_program" in normalized_attributes
                    and normalized_attributes["mathematical_program"] == "vllm_0_30_dspark_swa_noncausal_softmax_v1"
                    and self.hardware_id != "nvidia_h200"):
                raise ValueError("DSpark sparse source math is restricted to the declared H200 scenario")
            if (kind is KernelKind.SPARSE_ATTN and "mathematical_program" in normalized_attributes
                    and (self.launch is not None or self.execution_support != "unsupported")):
                raise ValueError("DSA mathematical prefill attention requires an unbound launch contract")
            if self.launch is None:
                object.__setattr__(
                    self, "shape", _immutable_mapping(normalized_shape))
                object.__setattr__(
                    self, "attributes", _immutable_mapping(normalized_attributes))
                return
            if (kind is KernelKind.GEMM
                    and self.launch.algorithm in BLOCK_FP8_GEMM_ALGORITHMS
                    and normalized_attributes["output_dtype"] not in (None, "bf16")):
                raise ValueError("known packed block-FP8 GEMM launch requires BF16 output storage")
            normalized_launch = _normalize_launch_parameters(
                kind, normalized_shape, self.launch, normalized_attributes)
            if (kind is KernelKind.VECTOR and self.hardware_id == "nvidia_h200"
                    and normalized_variant == "dsv4_inv_rope_fp8_quant"
                    and normalized_launch.algorithm == "dsv4_fused_inv_rope_fp8_quant_analytical"):
                aligned_rows = -(-normalized_shape["seq_len"] // 4) * 4
                if (normalized_launch.grid != (aligned_rows, normalized_shape["num_query_heads"], 1)
                        or normalized_launch.block != (32, 1, 1)):
                    raise ValueError("H200 inverse RoPE launch must include padded scale rows and one warp")
            if (kind is KernelKind.VECTOR
                    and normalized_variant == "moe_swiglu"):
                from src.kernel.launch_config.ascend_910b.vector_launch import (
                    validate_cann_moe_swiglu_launch,
                )
                validate_cann_moe_swiglu_launch(
                    hardware_id,
                    self.dtype.strip().lower(),
                    normalized_shape,
                    normalized_attributes,
                    normalized_launch,
                )
            if (kind is KernelKind.VECTOR
                    and "cann_rmsnorm_rstd_transfer_bytes"
                    in normalized_launch.parameters
                    and (hardware_id != "ascend_910b"
                         or self.dtype.strip().lower() != "bf16"
                         or normalized_variant != "rmsnorm")):
                raise ValueError(
                    "CANN RMSNorm rstd transfer geometry requires "
                    "ascend_910b BF16 RMSNorm")
            if (kind is KernelKind.VECTOR
                    and "source_bound_atb_bf16_rope_dag"
                    in normalized_launch.parameters):
                if hardware_id != "ascend_910b" or self.dtype != "bf16":
                    raise ValueError(
                        "VECTOR ATB BF16 RoPE DAG requires ascend_910b BF16")
                if (normalized_variant == "rope"
                        and (not normalized_attributes["is_neox"]
                             or not normalized_attributes["key_present"])):
                    raise ValueError(
                        "VECTOR ATB BF16 RoPE DAG requires NeoX Q+K "
                        "semantics")
            if (kind in (KernelKind.VECTOR, KernelKind.KV_CACHE)
                    and any(parameter in normalized_launch.parameters for parameter in (
                        SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER,
                        SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER,
                    ))):
                from src.kernel.launch_config.nvidia_a800.vector_launch import (
                    validate_source_bound_vector_launch,
                )
                source_launch_shape = normalized_shape
                if "model_input_shape" in normalized_attributes:
                    source_launch_shape = normalized_attributes[
                        "model_input_shape"]
                validate_source_bound_vector_launch(
                    hardware_id,
                    self.dtype.strip().lower(),
                    normalized_variant,
                    source_launch_shape,
                    normalized_attributes,
                    normalized_launch,
                )
            if (kind is KernelKind.VECTOR
                    and normalized_variant in (
                        "elementwise", "elementwise_pytorch_vec4")
                    and "lowered_vector_operations"
                    in normalized_launch.parameters
                    and normalized_attributes["operation"]
                    not in ("add", "mul")):
                raise ValueError(
                    "VECTOR/elementwise lowered_vector_operations requires "
                    "the source-closed add or mul operation")
            physical_algorithm = canonical_physical_algorithm(
                hardware_id,
                kind,
                normalized_variant,
                normalized_launch.algorithm,
            )
            if physical_algorithm != normalized_launch.algorithm:
                normalized_launch = replace(
                    normalized_launch, algorithm=physical_algorithm)
            object.__setattr__(self, "shape", _immutable_mapping(normalized_shape))
            object.__setattr__(
                self, "attributes", _immutable_mapping(normalized_attributes))
            object.__setattr__(self, "launch", normalized_launch)
        elif kind in transfer_kinds:
            if self.execution_support != "supported":
                raise ValueError(
                    f"{kind.value} transfer cannot be declared unsupported")
            if self.transfer is None:
                raise ValueError(f"{kind.value} requires physical transfer configuration")
            if self.launch is not None:
                raise ValueError(f"{kind.value} does not use GPU launch geometry")
            if (self.shape or self.dtype is not None or self.variant is not None
                    or self.attributes):
                raise ValueError(
                    f"{kind.value} accepts only PhysicalTransferConfig fields")
            if kind is KernelKind.COMMUNICATION:
                if self.transfer.tier is not None:
                    raise ValueError("COMMUNICATION does not accept tier")
                if not self.transfer.topology:
                    raise ValueError("COMMUNICATION requires topology")
                if not self.transfer.operation:
                    raise ValueError("COMMUNICATION requires operation")
                if not self.transfer.bytes_semantics:
                    raise ValueError("COMMUNICATION requires bytes_semantics")
                if not self.transfer.backend:
                    raise ValueError("COMMUNICATION requires backend")
                if not self.transfer.resource_id:
                    raise ValueError("COMMUNICATION requires resource_id")
                if self.transfer.bytes <= 0:
                    raise ValueError("COMMUNICATION bytes must be positive")
                if len(self.transfer.participants) < 2:
                    raise ValueError(
                        "COMMUNICATION requires at least two participants")
                operation = self.transfer.operation
                collective_operations = {
                    "all_reduce", "all_gather", "all_to_all", "all_gatherv", "reduce_scatterv"}
                if operation in collective_operations:
                    if self.transfer.direction != "collective":
                        raise ValueError(
                            f"{operation} COMMUNICATION requires collective "
                            "direction")
                    expected_semantics = ("aggregate_input" if operation == "all_gatherv"
                        or self.transfer.traffic_matrix_bytes else "per_rank_input")
                    if self.transfer.bytes_semantics != expected_semantics:
                        raise ValueError(
                            f"{operation} COMMUNICATION requires "
                            f"{expected_semantics} bytes_semantics")
                    if (self.transfer.source_rank is not None
                            or self.transfer.destination_rank is not None):
                        raise ValueError(
                            f"{operation} COMMUNICATION does not accept "
                            "source/destination ranks")
                elif operation == "send_recv":
                    if self.transfer.direction != "point_to_point":
                        raise ValueError(
                            "send_recv COMMUNICATION requires point_to_point "
                            "direction")
                    if self.transfer.bytes_semantics != "payload":
                        raise ValueError(
                            "send_recv COMMUNICATION requires payload "
                            "bytes_semantics")
                    if len(self.transfer.participants) != 2:
                        raise ValueError(
                            "send_recv COMMUNICATION requires two participants")
                    source_rank = self.transfer.source_rank
                    destination_rank = self.transfer.destination_rank
                    if source_rank is None or destination_rank is None:
                        raise ValueError(
                            "send_recv COMMUNICATION requires source and "
                            "destination ranks")
                    if source_rank == destination_rank:
                        raise ValueError(
                            "send_recv COMMUNICATION endpoints must differ")
                    if {source_rank, destination_rank} != set(
                            self.transfer.participants):
                        raise ValueError(
                            "send_recv COMMUNICATION endpoints must exactly "
                            "match participants")
                else:
                    raise ValueError(
                        f"unknown COMMUNICATION operation: {operation!r}")
                if operation in ("all_reduce", "all_gather"):
                    if operation == "all_reduce":
                        collective_layouts = {
                            "fc2": "fully_connected",
                            "fullmesh": "fully_connected",
                            "mesh": "fully_connected",
                            "ring": "ring",
                            "tree": "tree",
                        }
                    elif operation == "all_gather":
                        collective_layouts = {
                            "fc2": "fully_connected",
                            "fullmesh": "fully_connected",
                            "mesh": "fully_connected",
                            "ring": "ring",
                        }
                    else:
                        raise AssertionError(
                            "validated collective operation disappeared")
                    if self.transfer.algorithm not in collective_layouts:
                        raise ValueError(
                            f"{operation} COMMUNICATION currently requires "
                            "the ring, tree, fc2, mesh, or fullmesh algorithm")
                    expected_topology = collective_layouts[
                        self.transfer.algorithm]
                    if self.transfer.topology != expected_topology:
                        raise ValueError(
                            f"{operation} {self.transfer.algorithm} "
                            f"COMMUNICATION requires {expected_topology} "
                            "topology")
                    if self.transfer.algorithm == "ring":
                        if (len(self.transfer.ring_order)
                                != len(self.transfer.participants)
                                or set(self.transfer.ring_order)
                                != set(self.transfer.participants)):
                            raise ValueError(
                                f"{operation} COMMUNICATION ring_order must "
                                "be a permutation of participants")
                    elif self.transfer.algorithm in (
                            "fc2", "mesh", "fullmesh", "tree"):
                        if self.transfer.ring_order:
                            raise ValueError(
                                f"{operation} {self.transfer.algorithm} "
                                "COMMUNICATION does not accept ring_order")
                    else:
                        raise AssertionError(
                            "validated collective algorithm disappeared")
                elif operation in ("all_to_all", "all_gatherv", "reduce_scatterv"):
                    if self.transfer.algorithm != "pairwise":
                        raise ValueError(
                            "all_to_all COMMUNICATION currently requires the "
                            "pairwise algorithm")
                    if self.transfer.topology != "fully_connected":
                        raise ValueError(
                            "all_to_all COMMUNICATION currently requires a "
                            "fully_connected topology")
                    if self.transfer.ring_order:
                        raise ValueError(
                            "all_to_all COMMUNICATION does not accept ring_order")
                    if operation == "all_to_all" and not self.transfer.traffic_matrix_bytes and self.transfer.bytes % len(
                            self.transfer.participants):
                        raise ValueError(
                            "all_to_all per-rank input bytes must divide the "
                            "participant count")
                elif operation == "send_recv":
                    if self.transfer.algorithm != "send_recv":
                        raise ValueError(
                            "send_recv COMMUNICATION requires send_recv "
                            "algorithm")
                    if self.transfer.topology != "point_to_point":
                        raise ValueError(
                            "send_recv COMMUNICATION requires point_to_point "
                            "topology")
                    if self.transfer.ring_order:
                        raise ValueError(
                            "send_recv COMMUNICATION does not accept ring_order")
                else:
                    raise ValueError(
                        f"unhandled COMMUNICATION operation: {operation!r}")
                if operation in ("all_reduce", "reduce_scatterv"):
                    if self.transfer.reduction != "sum":
                        raise ValueError(
                            "all_reduce COMMUNICATION currently requires sum "
                            "reduction")
                elif self.transfer.reduction is not None:
                    raise ValueError(
                        f"{operation} COMMUNICATION does not accept reduction")
                if self.transfer.backend not in (
                        "analytical_roofline_v1", "protobridge"):
                    raise ValueError(
                        "COMMUNICATION backend must be "
                        "analytical_roofline_v1 or protobridge")
                if self.transfer.backend == "analytical_roofline_v1":
                    if (self.transfer.link not in ("nvlink", "metaxlink", "hccs", "infiniband", "roce")
                            and not (self.transfer.link == "pcie" and self.transfer.traffic_matrix_bytes)):
                        raise ValueError(
                            "analytical Roofline COMMUNICATION models only "
                            "NVLink, MetaXLink, HCCS, InfiniBand or RoCE paths")
                    if ((self.transfer.link == "metaxlink" and self.hardware_id != "metax_c500")
                            or (self.transfer.link == "nvlink" and self.hardware_id == "metax_c500")):
                        raise ValueError("analytical MetaXLink requires its C500 hardware identity")
                    if ((self.transfer.link == "hccs" and self.hardware_id != "ascend_910b")
                            or (self.hardware_id == "ascend_910b" and self.transfer.link in ("nvlink", "metaxlink"))):
                        raise ValueError("analytical HCCS requires its Ascend910B hardware identity")
                    if (operation in ("all_reduce", "all_gather")
                            and self.transfer.algorithm != "ring"):
                        raise ValueError(
                            "analytical Roofline all_reduce/all_gather "
                            "requires the ring algorithm")
                    if self.transfer.backend_config_sha256 is not None:
                        raise ValueError(
                            "analytical Roofline COMMUNICATION does not accept "
                            "backend_config_sha256")
                elif self.transfer.backend == "protobridge":
                    if self.transfer.link not in (
                            "nvlink", "metaxlink", "hccs", "infiniband"):
                        raise ValueError(
                            "ProtoBridge COMMUNICATION link must be nvlink, "
                            "metaxlink, hccs, or infiniband")
                    if self.transfer.backend_config_sha256 is None:
                        raise ValueError(
                            "ProtoBridge COMMUNICATION requires a versioned "
                            "backend config SHA256")
                else:
                    raise ValueError(
                        f"unhandled COMMUNICATION backend: "
                        f"{self.transfer.backend!r}")
            elif kind is KernelKind.MEMCPY:
                if (self.transfer.topology is not None
                        or self.transfer.tier is not None
                        or self.transfer.participants
                        or self.transfer.operation is not None
                        or self.transfer.bytes_semantics is not None
                        or self.transfer.source_rank is not None
                        or self.transfer.destination_rank is not None
                        or self.transfer.ring_order
                        or self.transfer.reduction is not None
                        or self.transfer.backend is not None
                        or self.transfer.resource_id is not None
                        or self.transfer.backend_config_sha256 is not None):
                    raise ValueError(
                        "MEMCPY does not accept communication-specific fields")
                if self.transfer.algorithm == "fpga_staging":
                    if (self.hardware_id != "intel_agmf039r47" or self.transfer.link != "ddr"
                            or self.transfer.direction not in (
                                "network_to_ddr", "ddr_to_buffer", "hbm_to_ddr", "ddr_to_network")):
                        raise ValueError("FPGA staging requires its DDR link, Intel design and explicit memory direction")
                    return
                if self.transfer.algorithm not in (
                        "dma", "native_d2d_flash_attention_copy",
                        "stream_copy", "stream_copy_pinned",
                        "stream_copy_pageable"):
                    raise ValueError(
                        "MEMCPY currently models only DMA, the native "
                        "C500 FlashAttention D2D copy or an eager stream "
                        "copy")
                direction = self.transfer.direction
                if direction in ("d2d", "dtod", "device_to_device"):
                    if self.transfer.link != "dram":
                        raise ValueError("device MEMCPY requires the DRAM link")
                    if self.transfer.algorithm in (
                            "stream_copy_pinned", "stream_copy_pageable"):
                        raise ValueError(
                            "a device-to-device stream copy has no host "
                            "memory kind")
                    if (self.transfer.algorithm
                            == "native_d2d_flash_attention_copy"
                            and self.hardware_id != "metax_c500"):
                        raise ValueError(
                            "native FlashAttention D2D copy requires "
                            "metax_c500")
                elif direction in (
                        "h2d", "d2h", "host_to_device", "device_to_host"):
                    if self.transfer.link != "pcie":
                        raise ValueError("host MEMCPY requires the PCIe link")
                    if self.transfer.algorithm == "stream_copy":
                        raise ValueError(
                            "a host stream copy must name pinned or pageable "
                            "host memory")
                    if self.transfer.algorithm not in (
                            "dma", "stream_copy_pinned",
                            "stream_copy_pageable"):
                        raise ValueError(
                            "native FlashAttention D2D copy requires a "
                            "device-to-device transfer")
                else:
                    raise ValueError(
                        f"unknown MEMCPY direction: {self.transfer.direction!r}")
            elif kind is KernelKind.KV_OFFLOAD:
                if (self.transfer.topology is not None
                        or self.transfer.participants
                        or self.transfer.operation is not None
                        or self.transfer.bytes_semantics is not None
                        or self.transfer.source_rank is not None
                        or self.transfer.destination_rank is not None
                        or self.transfer.ring_order
                        or self.transfer.reduction is not None
                        or self.transfer.backend is not None
                        or self.transfer.resource_id is not None
                        or self.transfer.backend_config_sha256 is not None):
                    raise ValueError(
                        "KV_OFFLOAD does not accept communication-specific "
                        "fields")
                if not self.transfer.tier:
                    raise ValueError("KV_OFFLOAD requires tier")
                if self.transfer.algorithm != "dma":
                    raise ValueError("KV_OFFLOAD currently models only DMA")
                if self.transfer.direction not in ("load", "evict"):
                    raise ValueError(
                        "KV_OFFLOAD direction must be 'load' or 'evict'")
                if self.transfer.link not in ("pcie", "nvlink"):
                    raise ValueError(
                        "KV_OFFLOAD link must be 'pcie' or 'nvlink'")
            else:
                raise AssertionError(
                    f"unhandled validated transfer kind: {kind.value}")
        else:
            raise ValueError(f"unhandled kernel kind: {kind.value}")


@dataclass(frozen=True)
class KernelPrediction:
    """Predicted latency and physical breakdown for one complete KernelSpec."""

    t_total_s: float
    t_per_group: Mapping[str, float]
    t_launch_s: float
    bottleneck_group: str
    t_per_resource: Mapping[str, float]
    diagnostics: Optional[PredictionDiagnostics] = None

    def __post_init__(self) -> None:
        if not isinstance(self.t_per_group, Mapping):
            raise TypeError("t_per_group must be a mapping")
        if not isinstance(self.t_per_resource, Mapping):
            raise TypeError("t_per_resource must be a mapping")
        scalar_times = {
            "t_total_s": self.t_total_s,
            "t_launch_s": self.t_launch_s,
        }
        invalid_scalars = {
            name: value for name, value in scalar_times.items()
            if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0)
        }
        if invalid_scalars:
            raise ValueError(
                f"prediction times must be finite and non-negative: "
                f"{invalid_scalars}")
        invalid_breakdown_keys = [
            name
            for mapping in (self.t_per_group, self.t_per_resource)
            for name in mapping
            if not isinstance(name, str) or not name
        ]
        if invalid_breakdown_keys:
            raise TypeError(
                "prediction breakdown keys must be non-empty strings: "
                f"{invalid_breakdown_keys}")
        invalid_breakdown = {
            name: value
            for mapping in (self.t_per_group, self.t_per_resource)
            for name, value in mapping.items()
            if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0)
        }
        if invalid_breakdown:
            raise ValueError(
                "prediction breakdowns must be finite and non-negative: "
                f"{invalid_breakdown}")
        if not isinstance(self.bottleneck_group, str) or not self.bottleneck_group:
            raise ValueError("prediction bottleneck_group must be non-empty")
        if (self.diagnostics is not None
                and not isinstance(self.diagnostics, PredictionDiagnostics)):
            raise TypeError(
                "prediction diagnostics must be PredictionDiagnostics or None")
        object.__setattr__(self, "t_per_group", _immutable_mapping(self.t_per_group))
        object.__setattr__(self, "t_per_resource", _immutable_mapping(self.t_per_resource))

    @property
    def t_total_ns(self) -> float:
        return self.t_total_s * 1e9


# FULL captures every call of the step, so any spec carrying it is a graph
# node.  PIECEWISE does not: the runtime leaves the configured split-op kinds
# eager and captures only the runs between them, and which side of that
# boundary a call falls on is a property of the step's call sequence, not of
# the KernelSpec.  A KernelSpec cannot answer it, so PIECEWISE stays on the
# standalone floor rather than charging a replayed node's span to calls the
# runtime launches one at a time.
_GRAPH_REPLAYED_RUNTIME_MODES = ("FULL",)


def graph_replayed_launch(kernel) -> bool:
    """Whether this spec executes as a node of a replayed CUDA graph.

    Only a spec that carries an explicit graph runtime mode can answer this;
    the canonical kernel dataset carries none and is therefore standalone,
    which is exactly how it is captured.
    """

    attributes = kernel.attributes
    if "graph_runtime_mode" not in attributes:
        return False
    return attributes["graph_runtime_mode"] in _GRAPH_REPLAYED_RUNTIME_MODES



def prediction_from_ns(latency_ns: float, group: str,
                       launch_ns: float = 0.0) -> KernelPrediction:
    seconds = float(latency_ns) / 1e9
    return KernelPrediction(
        t_total_s=seconds,
        t_per_group={group: seconds},
        t_launch_s=float(launch_ns) / 1e9,
        bottleneck_group=group,
        t_per_resource={group: seconds},
    )
