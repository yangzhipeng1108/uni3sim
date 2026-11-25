"""Complete physical launch geometry for a compute kernel."""

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from .launch_program import LaunchProgram
from .tiling_spec import TilingSpec


def _dim3(name: str, value: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a three-dimensional list or tuple")
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three dimensions")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in value):
        raise TypeError(f"{name} dimensions must be integers, got {value}")
    result = tuple(value)
    if any(v <= 0 for v in result):
        raise ValueError(f"{name} dimensions must be positive, got {result}")
    return result


@dataclass(frozen=True)
class LaunchConfig:
    """Algorithm and geometry chosen before a predictor is invoked."""

    algorithm: str
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    tiling: Optional[TilingSpec] = None
    cluster: Optional[Tuple[int, int, int]] = None
    parameters: Mapping[str, float] = field(default_factory=dict)
    execution_variant: Optional[str] = None
    source_program: Optional[LaunchProgram] = None

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, str) or not self.algorithm.strip():
            raise ValueError("launch algorithm must be non-empty")
        object.__setattr__(self, "algorithm", self.algorithm.strip())
        object.__setattr__(self, "grid", _dim3("grid", self.grid))
        object.__setattr__(self, "block", _dim3("block", self.block))
        if self.cluster is not None:
            object.__setattr__(self, "cluster", _dim3("cluster", self.cluster))
        if self.tiling is not None and not isinstance(self.tiling, TilingSpec):
            raise TypeError("tiling must be a TilingSpec or null")
        if self.execution_variant is not None:
            if not isinstance(self.execution_variant, str):
                raise TypeError("execution_variant must be a string or null")
            variant = self.execution_variant.strip().lower()
            if not variant:
                raise ValueError("execution_variant cannot be empty")
            object.__setattr__(self, "execution_variant", variant)
        if (self.source_program is not None
                and not isinstance(self.source_program, LaunchProgram)):
            raise TypeError(
                "source_program must be a LaunchProgram or null")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("launch parameters must be a mapping")
        invalid_parameter_keys = [
            name for name in self.parameters
            if not isinstance(name, str) or not name
        ]
        if invalid_parameter_keys:
            raise TypeError(
                "launch parameter keys must be non-empty strings: "
                f"{invalid_parameter_keys}")
        invalid_parameter_types = {
            name: value for name, value in self.parameters.items()
            if isinstance(value, bool) or not isinstance(value, (int, float))
        }
        if invalid_parameter_types:
            raise TypeError(
                "launch parameters must be numeric: "
                f"{invalid_parameter_types}")
        parameters = {
            name: float(value) for name, value in self.parameters.items()
        }
        invalid = {
            name: value for name, value in parameters.items()
            if not math.isfinite(value)
        }
        if invalid:
            raise ValueError(f"launch parameters must be finite: {invalid}")
        object.__setattr__(
            self, "parameters", MappingProxyType(parameters))

    @property
    def grid_size(self) -> int:
        x, y, z = self.grid
        return x * y * z

    @property
    def block_size(self) -> int:
        x, y, z = self.block
        return x * y * z


@dataclass(frozen=True)
class FlashAttentionSetupLaunch:
    """One source-defined setup launch emitted before FlashAttention."""

    variant: str
    dtype: str
    shape: Mapping[str, int]
    attributes: Mapping[str, object]
    launch: LaunchConfig

    def __post_init__(self) -> None:
        if not isinstance(self.variant, str) or not self.variant.strip():
            raise ValueError("FlashAttention setup variant must be non-empty")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise ValueError("FlashAttention setup dtype must be non-empty")
        if not isinstance(self.shape, Mapping):
            raise TypeError("FlashAttention setup shape must be a mapping")
        if not isinstance(self.attributes, Mapping):
            raise TypeError(
                "FlashAttention setup attributes must be a mapping")
        if not isinstance(self.launch, LaunchConfig):
            raise TypeError(
                "FlashAttention setup launch must be a LaunchConfig")
        object.__setattr__(self, "variant", self.variant.strip().lower())
        object.__setattr__(self, "dtype", self.dtype.strip().lower())
        object.__setattr__(self, "shape", MappingProxyType(dict(self.shape)))
        object.__setattr__(
            self, "attributes", MappingProxyType(dict(self.attributes)))


@dataclass(frozen=True)
class FlashAttentionForwardPart:
    """One backend pathway of a partitioned logical FlashAttention call.

    ``request_indexes`` selects the logical requests this pathway executes;
    ``output_copy_elements`` and ``output_copy_launch`` describe the
    source-level slice assignment that writes the pathway result into the
    shared attention output buffer (zero and ``None`` when the pathway's
    write-back is selected elsewhere).
    """

    role: str
    variant: str
    request_indexes: Tuple[int, ...]
    output_copy_elements: int
    output_copy_launch: Optional["LaunchConfig"]

    def __post_init__(self) -> None:
        if self.role not in ("prefill", "decode"):
            raise ValueError(
                f"FlashAttention forward part role is invalid: {self.role!r}")
        if self.variant not in ("prefill", "decode"):
            raise ValueError(
                "FlashAttention forward part variant is invalid: "
                f"{self.variant!r}")
        if (not isinstance(self.request_indexes, tuple)
                or not self.request_indexes
                or any(isinstance(index, bool) or not isinstance(index, int)
                       or index < 0 for index in self.request_indexes)
                or len(set(self.request_indexes))
                != len(self.request_indexes)):
            raise ValueError(
                "FlashAttention forward part request indexes are invalid")
        if (isinstance(self.output_copy_elements, bool)
                or not isinstance(self.output_copy_elements, int)
                or self.output_copy_elements < 0):
            raise ValueError(
                "FlashAttention forward part output copy count is invalid")
        if (self.output_copy_elements == 0) != (
                self.output_copy_launch is None):
            raise ValueError(
                "FlashAttention forward part output copy count and launch "
                "must be present together")
        if (self.output_copy_launch is not None
                and not isinstance(self.output_copy_launch, LaunchConfig)):
            raise TypeError(
                "FlashAttention forward part output copy launch must be a "
                "LaunchConfig")
