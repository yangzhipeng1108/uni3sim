"""User configuration for parallel execution inside one worker pool."""

from dataclasses import dataclass


class ParallelConfigError(ValueError):
    """An invalid or unsupported parallel configuration."""


@dataclass(frozen=True)
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enable_expert_parallel: bool = False
    expert_communication_backend: str = "allgather_reducescatter"
    enable_eplb: bool = False
    sequence_parallel_size: int = 1
    context_parallel_size: int = 1
    data_parallel_size: int = 1

    @property
    def is_rank_local(self) -> bool:
        """Whether the plan is rank-local: one sequence, context and data group.

        Every validated emission scope is rank-local within one sequence,
        context and data-parallel group, so the three unit axes are one fact
        about the plan rather than three conditions every caller restates.
        """

        return (
            self.sequence_parallel_size == 1
            and self.context_parallel_size == 1
            and self.data_parallel_size == 1
        )

    def __post_init__(self) -> None:
        for name in (
            "tensor_parallel_size", "pipeline_parallel_size",
            "sequence_parallel_size", "context_parallel_size",
            "data_parallel_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ParallelConfigError(f"{name} must be an integer >= 1")
        if self.sequence_parallel_size > self.tensor_parallel_size:
            raise ParallelConfigError(
                "sequence_parallel_size cannot exceed tensor_parallel_size")
        if self.tensor_parallel_size % self.sequence_parallel_size:
            raise ParallelConfigError(
                "tensor_parallel_size must be divisible by sequence_parallel_size")
        if not isinstance(self.enable_expert_parallel, bool):
            raise ParallelConfigError(
                "enable_expert_parallel must be boolean")
        if not isinstance(self.enable_eplb, bool):
            raise ParallelConfigError("enable_eplb must be boolean")
        if (not isinstance(self.expert_communication_backend, str)
                or not self.expert_communication_backend.strip()):
            raise ParallelConfigError(
                "expert_communication_backend must be a non-empty string")
        normalized_backend = self.expert_communication_backend.strip().lower()
        object.__setattr__(
            self, "expert_communication_backend", normalized_backend)
        if normalized_backend != "allgather_reducescatter":
            raise ParallelConfigError(
                "the pinned vLLM EP contract currently supports only the "
                "allgather_reducescatter backend")
        if self.enable_eplb:
            raise ParallelConfigError(
                "the pinned vLLM EP contract does not yet support EPLB")
        if (self.enable_expert_parallel
                and self.tensor_parallel_size * self.data_parallel_size * self.context_parallel_size == 1):
            raise ParallelConfigError(
                "expert parallelism requires TP * DP * PCP greater than one")

    @property
    def expert_parallel_size(self) -> int:
        """vLLM0.30 EP reuses DP x PCP x TP ranks at one PP stage."""

        if not self.enable_expert_parallel:
            return 1
        return self.tensor_parallel_size * self.data_parallel_size * self.context_parallel_size

    @property
    def worker_count(self) -> int:
        """Physical ranks; sequence parallelism reuses tensor-parallel ranks."""
        return (
            self.tensor_parallel_size * self.pipeline_parallel_size
            * self.context_parallel_size * self.data_parallel_size
        )
