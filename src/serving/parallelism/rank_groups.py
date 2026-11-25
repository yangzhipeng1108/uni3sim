"""Named rank groups for TP/PP/EP/SP/CP/DP."""

from dataclasses import dataclass
from collections import defaultdict
from itertools import product
from typing import Dict, Iterable, Tuple

from .parallel_config import ParallelConfig

RankGroup = Tuple[int, ...]


@dataclass(frozen=True)
class RankGroups:
    world_size: int
    tensor: Tuple[RankGroup, ...]
    pipeline: Tuple[RankGroup, ...]
    expert: Tuple[RankGroup, ...]
    sequence: Tuple[RankGroup, ...]
    context: Tuple[RankGroup, ...]
    data: Tuple[RankGroup, ...]

    def for_rank(self, dimension: str, rank: int) -> RankGroup:
        if not isinstance(dimension, str):
            raise TypeError("parallel dimension must be a string")
        dimensions = {
            "tensor": self.tensor,
            "pipeline": self.pipeline,
            "expert": self.expert,
            "sequence": self.sequence,
            "context": self.context,
            "data": self.data,
            # MoE EP-off dispatch flattens DP and PCP while keeping TP
            # lanes independent. This composes existing physical axes.
            "data_context": (_deduplicate(tuple(sorted({member for context_rank in context_group
                for data_group in self.data if context_rank in data_group for member in data_group}))
                for context_group in self.context) if dimension == "data_context" else ()),
        }
        if dimension not in dimensions:
            raise ValueError(f"unknown parallel dimension: {dimension!r}")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("rank must be an integer")
        if not 0 <= rank < self.world_size:
            raise ValueError(
                f"rank {rank} is outside world size {self.world_size}")
        groups = dimensions[dimension]
        matches = [group for group in groups if rank in group]
        if len(matches) != 1:
            raise ValueError(
                f"rank {rank} belongs to {len(matches)} {dimension} groups")
        return matches[0]


def _deduplicate(groups: Iterable[RankGroup]) -> Tuple[RankGroup, ...]:
    return tuple(dict.fromkeys(tuple(group) for group in groups))


def build_rank_groups(config: ParallelConfig) -> RankGroups:
    """Construct orthogonal groups, with SP explicitly reusing TP ranks."""
    sizes = (
        config.data_parallel_size,
        config.pipeline_parallel_size,
        config.context_parallel_size,
        config.tensor_parallel_size,
    )
    coordinates = list(product(*(range(size) for size in sizes)))
    rank_by_coordinate: Dict[Tuple[int, ...], int] = {
        coordinate: rank for rank, coordinate in enumerate(coordinates)
    }

    def vary_axes(axes: Tuple[int, ...]) -> Tuple[RankGroup, ...]:
        grouped: Dict[Tuple[int, ...], list[int]] = defaultdict(list)
        for coordinate, rank in rank_by_coordinate.items():
            key = tuple(
                value for index, value in enumerate(coordinate)
                if index not in axes
            )
            grouped[key].append(rank)
        return _deduplicate(tuple(values) for values in grouped.values())

    def vary(axis: int) -> Tuple[RankGroup, ...]:
        return vary_axes((axis,))

    tensor = vary(3)
    if config.enable_expert_parallel:
        expert = vary_axes((0, 2, 3))
    else:
        expert = tuple((rank,) for rank in range(config.worker_count))
    sp_size = config.sequence_parallel_size
    sequence = _deduplicate(
        tuple(group[start:start + sp_size])
        for group in tensor
        for start in range(0, len(group), sp_size)
    )
    return RankGroups(
        world_size=config.worker_count,
        tensor=tensor,
        pipeline=vary(1),
        expert=expert,
        sequence=sequence,
        context=vary(2),
        data=vary(0),
    )
