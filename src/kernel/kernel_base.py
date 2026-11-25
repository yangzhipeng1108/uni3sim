"""Pure production predictor interface and one-to-one kind registry."""

from abc import ABC, abstractmethod
from importlib import import_module
from typing import Dict, Mapping, Optional, Tuple, Type

from src.kernel.kernel_spec import (
    KernelKind, KernelPrediction, KernelSpec,
    graph_replayed_launch, prediction_from_ns,
)
from src.kernel.model_contract import PredictionInput


class KernelPredictor(ABC):
    kind: KernelKind

    @abstractmethod
    def midlayer(self, kernel: KernelSpec, hardware) -> KernelPrediction:
        ...

    @abstractmethod
    def roofline(self, kernel: KernelSpec, hardware) -> KernelPrediction:
        ...

    def resource_input(self, kernel: KernelSpec, hardware):
        """Return the analytical resource input and compulsory traffic.

        Compute predictors expose this for Oracle observation normalization and
        ablation. Transfer predictors may leave it unsupported.
        """
        raise NotImplementedError(f"resource input is not available for {self.kind.value}")

    def roofline_without_launch(self, kernel: KernelSpec, hardware) -> KernelPrediction:
        """Price proven logical work without asserting execution support.

        Owners must explicitly implement this for a closed source contract.
        The default preserves the unsupported boundary for every other kind
        and variant; no launch, occupancy or compiler services are inferred.
        """
        raise NotImplementedError(
            f"{kernel.kind.value}/{kernel.variant} unsupported: {kernel.unsupported_reason}")

    def analytical_without_launch(self, kernel: KernelSpec, hardware) -> KernelPrediction:
        """Price an explicitly modeled analytical helper in a Forward run.

        A missing native launch is independent of a known performance model.
        Owners must define the supported analytical work; the default rejects
        unknown work rather than silently omitting it or inventing a launch.
        """
        raise NotImplementedError(
            f"{kernel.kind.value}/{kernel.variant} unsupported: {kernel.unsupported_reason}")

    def forward_resource_input(self, kernel: KernelSpec, hardware):
        """Return the analytical input consumed by the Forward path.

        Most predictors Forward the public source-contract representation.
        A predictor may override this only when Forward uses an equivalent,
        compact representation that is intentionally distinct for contract
        inspection.
        """
        return self.resource_input(kernel, hardware)

    def oracle_resource_input(self, kernel: KernelSpec, hardware):
        """Return the resource input consumed by Oracle normalization.

        Most predictors use their public source-contract representation.
        Predictors with an exactly equivalent compact analytical envelope may
        override this without changing the public ``resource_input`` route.
        """
        return self.resource_input(kernel, hardware)

    def oracle(self, kernel: KernelSpec, hardware, oracle_input) -> KernelPrediction:
        """Price an observation-normalized resource input with this predictor."""
        from src.kernel.formula import PerformanceFormula

        if not isinstance(oracle_input, PredictionInput):
            raise TypeError(
                "oracle requires an observation-normalized PredictionInput, "
                f"got {type(oracle_input).__name__}")
        return PerformanceFormula(hardware).predict_kernel(oracle_input)

    def pipeweave(self, kernel: KernelSpec, hardware,
                  *, model_hardware: Optional[str] = None) -> KernelPrediction:
        del model_hardware
        raise NotImplementedError(f"PipeWeave does not support {self.kind.value}")

    def predict(self, kernel: KernelSpec, hardware,
                method: str = "midlayer", *, oracle_input=None,
                model_hardware: Optional[str] = None) -> KernelPrediction:
        if not isinstance(kernel, KernelSpec):
            raise TypeError(f"predict requires KernelSpec, got {type(kernel).__name__}")
        if kernel.kind is not self.kind:
            raise ValueError(f"{type(self).__name__} cannot predict {kernel.kind.value}")
        hardware_id = hardware.hardware_id
        if kernel.hardware_id != hardware_id:
            raise ValueError(
                f"KernelSpec hardware_id={kernel.hardware_id} does not match HWSpec {hardware_id}")
        from src.kernel.hardware_spec import HWSpec
        if isinstance(hardware, HWSpec) and hardware.roofline_only:
            if method != "roofline":
                raise NotImplementedError("aggregate hardware profile supports roofline only; native execution services are unknown")
            analytical_transfer = (
                kernel.kind is KernelKind.COMMUNICATION
                and kernel.transfer.backend == "analytical_roofline_v1"
                and (kernel.transfer.link == "roce"
                    or kernel.transfer.link == "pcie" and kernel.transfer.traffic_matrix_bytes))
            analytical_transfer = analytical_transfer or (
                kernel.kind is KernelKind.MEMCPY and kernel.transfer.algorithm == "fpga_staging")
            if (not analytical_transfer
                    and (kernel.execution_support != "unsupported" or kernel.launch is not None)):
                raise NotImplementedError("aggregate roofline profile requires mathematical work without a native launch claim")
        if kernel.execution_support == "unsupported":
            if method == "midlayer":
                if oracle_input is not None:
                    raise ValueError("oracle_input is only valid for the Oracle method")
                if model_hardware is not None:
                    raise ValueError("model_hardware is only valid for PipeWeave")
                return self.analytical_without_launch(kernel, hardware)
            if method == "roofline":
                if oracle_input is not None:
                    raise ValueError("oracle_input is only valid for the Oracle method")
                if model_hardware is not None:
                    raise ValueError("model_hardware is only valid for PipeWeave")
                return self.roofline_without_launch(kernel, hardware)
            raise NotImplementedError(
                f"{kernel.kind.value}/{kernel.variant} unsupported: "
                f"{kernel.unsupported_reason}")
        if kernel.execution_support != "supported":
            raise AssertionError(
                f"unhandled KernelSpec execution support: "
                f"{kernel.execution_support!r}")
        if method == "oracle":
            if model_hardware is not None:
                raise ValueError("model_hardware is only valid for PipeWeave")
            return self.oracle(kernel, hardware, oracle_input)
        elif method == "midlayer":
            if oracle_input is not None:
                raise ValueError("oracle_input is only valid for the Oracle method")
            if model_hardware is not None:
                raise ValueError("model_hardware is only valid for PipeWeave")
            return self.midlayer(kernel, hardware)
        elif method == "roofline":
            if oracle_input is not None:
                raise ValueError("oracle_input is only valid for the Oracle method")
            if model_hardware is not None:
                raise ValueError("model_hardware is only valid for PipeWeave")
            return self.roofline(kernel, hardware)
        elif method == "roofline_span_floor":
            if oracle_input is not None:
                raise ValueError("oracle_input is only valid for the Oracle method")
            if model_hardware is not None:
                raise ValueError("model_hardware is only valid for PipeWeave")
            return self._roofline_span_floor(kernel, hardware)
        elif method == "pipeweave":
            if oracle_input is not None:
                raise ValueError("oracle_input is only valid for the Oracle method")
            return self.pipeweave(
                kernel, hardware, model_hardware=model_hardware)
        else:
            raise ValueError(f"unknown prediction method: {method!r}")


    def _roofline_span_floor(self, kernel: KernelSpec, hardware) -> KernelPrediction:
        """Roofline bounded below by the card's measured launch span floor.

        Roofline prices only aggregate demand, so a launch whose demand is
        tiny costs almost nothing, while an independent microbenchmark shows
        that any memory-touching launch spans at least the published
        ``small_kernel_span_floor``.  This method keeps Roofline unchanged as
        a baseline and composes that measured floor by maximum; transfers
        carry no GPU launch and keep their Roofline price.
        """
        from src.kernel.formula import small_kernel_span_floor_s

        prediction = self.roofline(kernel, hardware)
        if kernel.transfer is not None:
            return prediction
        floor_s = small_kernel_span_floor_s(
            hardware, kernel.launch.grid_size, kernel.variant,
            graph_replayed_launch(kernel))
        if prediction.t_total_s >= floor_s:
            return prediction
        return prediction_from_ns(floor_s * 1e9, "launch_span_floor")


_REGISTRY: Dict[KernelKind, Tuple[str, str]] = {
    KernelKind.GEMM: ("gemm", "GEMMPredictor"),
    KernelKind.FLASH_ATTN: ("flash_attn", "FlashAttentionPredictor"),
    KernelKind.EAGER_ATTN: ("eager_attn", "EagerAttentionPredictor"),
    KernelKind.SPARSE_ATTN_INDEXER: (
        "sparse_attn_indexer", "SparseAttentionIndexerPredictor"),
    KernelKind.SPARSE_ATTN: ("sparse_attn", "SparseAttentionPredictor"),
    KernelKind.KV_CACHE: (
        "kv_cache_update", "KVCacheUpdatePredictor"),
    KernelKind.FUSED_MOE: ("fused_moe", "FusedMoEPredictor"),
    KernelKind.UNFUSED_MOE: ("unfused_moe", "UnfusedMoEPredictor"),
    KernelKind.VECTOR: ("vector", "VectorPredictor"),
    KernelKind.COMMUNICATION: ("communication", "CommunicationPredictor"),
    KernelKind.MEMCPY: ("memcpy", "MemcpyPredictor"),
    KernelKind.KV_OFFLOAD: ("kv_offload", "KVOffloadPredictor"),
}


def create_predictor(kernel_kind: KernelKind | str) -> KernelPredictor:
    kind = KernelKind.parse(kernel_kind)
    module_name, class_name = _REGISTRY[kind]
    cls: Type[KernelPredictor] = getattr(
        import_module(f"src.kernel.predictors.{module_name}"), class_name)
    predictor = cls()
    if predictor.kind is not kind:
        raise RuntimeError(f"registry mismatch for {kind.value}: {type(predictor).__name__}")
    return predictor


def predict(kernel_spec: KernelSpec, hardware,
            method: str = "midlayer", *, oracle_input=None,
            model_hardware: Optional[str] = None,
            memory_context=None, memory_bindings=None) -> KernelPrediction:
    if memory_context is not None:
        from src.kernel.memory_backend import MemoryExecutionContext
        if not isinstance(memory_context, MemoryExecutionContext):
            raise TypeError("memory_context requires MemoryExecutionContext")
        memory_context.validate(kernel_spec, hardware, method)
    from src.kernel.memory_access_builder import (
        prediction_memory_bindings, prediction_memory_input_scope,
    )
    from src.kernel.prediction_cache import (
        prediction_cache_scope, _public_resource_input_producer,
    )
    with (prediction_cache_scope(reuse=True), prediction_memory_input_scope(),
            prediction_memory_bindings(kernel_spec, memory_bindings, method),
            _public_resource_input_producer(kernel_spec, hardware, method)):
        return create_predictor(kernel_spec.kind).predict(
            kernel_spec, hardware, method,
            oracle_input=oracle_input, model_hardware=model_hardware)


def registered_kernel_kinds():
    return frozenset(_REGISTRY)


# Families whose prediction depends on part of the KernelSpec only through a
# coarser physical class.  A family listed here exports
# ``prediction_equivalence_attributes``; every other family is identified by
# its exact contract.
_EQUIVALENCE_FAMILIES = frozenset((KernelKind.FUSED_MOE,))
_EQUIVALENCE_DECLARATIONS: Dict[KernelKind, object] = {}


def prediction_equivalence_attributes(kernel_spec: KernelSpec):
    """Return the canonical attributes naming this launch's physical class.

    A predictor that reads a contract field only through reductions coarser
    than the field itself can declare the canonical member of that class, so
    a run-local memo reuses one prediction across every launch the hardware
    cannot tell apart.  ``None`` means the exact contract is the class.
    """

    if not isinstance(kernel_spec, KernelSpec):
        raise TypeError(
            "prediction equivalence requires KernelSpec, got "
            f"{type(kernel_spec).__name__}")
    kind = kernel_spec.kind
    if kind not in _EQUIVALENCE_FAMILIES:
        return None
    if kind in _EQUIVALENCE_DECLARATIONS:
        declaration = _EQUIVALENCE_DECLARATIONS[kind]
    else:
        module_name, _class_name = _REGISTRY[kind]
        module = import_module(f"src.kernel.predictors.{module_name}")
        declaration = module.prediction_equivalence_attributes
        _EQUIVALENCE_DECLARATIONS[kind] = declaration
    attributes = declaration(kernel_spec)
    if attributes is None:
        return None
    if not isinstance(attributes, Mapping):
        raise TypeError(
            "prediction equivalence attributes must be a mapping")
    return attributes


def prediction_equivalence_fields(kernel_spec: KernelSpec):
    """Return field substitutions naming a predictor's exact physical class.

    These values are used only as memo keys. The actual KernelSpec, graph,
    launch, and prediction remain unchanged. Families without a declaration
    retain their complete contract as their key.
    """

    attributes = prediction_equivalence_attributes(kernel_spec)
    if kernel_spec.kind is KernelKind.FLASH_ATTN:
        module = import_module("src.kernel.predictors.flash_attn")
        return module.prediction_equivalence_fields(kernel_spec)
    if attributes is not None:
        return {"attributes": attributes}
    return None


def build_memory_access_program(kernel_spec, hardware, bindings, granularity,
                                *, program_id: str, tile=None):
    """Build untimed logical access work through the kernel owner."""
    from src.kernel.memory_access_builder import build_memory_access_program as build
    return build(kernel_spec, hardware, bindings, granularity,
                 program_id=program_id, tile=tile)
