"""Deterministic synthetic request lengths and prefix-hit oracles."""

import random
from dataclasses import dataclass
import math
from typing import Tuple

from .request_stream import RequestSpec, RequestStream


@dataclass(frozen=True)
class SyntheticRequestConfig:
    num_requests: int
    request_rate: float
    input_lengths: Tuple[int, ...]
    output_lengths: Tuple[int, ...]
    prefix_hit_fraction: float = 0.0
    arrival_distribution: str = "poisson"
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.num_requests, bool) or not isinstance(
                self.num_requests, int):
            raise TypeError("num_requests must be an integer")
        if (isinstance(self.request_rate, bool)
                or not isinstance(self.request_rate, (int, float))
                or not math.isfinite(self.request_rate)):
            raise TypeError("request_rate must be a finite number")
        if self.num_requests < 1 or self.request_rate <= 0:
            raise ValueError("num_requests and request_rate must be positive")
        for name, lengths in (
                ("input_lengths", self.input_lengths),
                ("output_lengths", self.output_lengths)):
            if (not isinstance(lengths, (list, tuple))
                    or any(isinstance(length, bool)
                           or not isinstance(length, int)
                           for length in lengths)):
                raise TypeError(f"{name} must contain integers")
        object.__setattr__(self, "input_lengths", tuple(self.input_lengths))
        object.__setattr__(self, "output_lengths", tuple(self.output_lengths))
        if not self.input_lengths or any(length < 1 for length in self.input_lengths):
            raise ValueError("input_lengths must contain positive values")
        if not self.output_lengths or any(length < 0 for length in self.output_lengths):
            raise ValueError("output_lengths must contain non-negative values")
        if (isinstance(self.prefix_hit_fraction, bool)
                or not isinstance(self.prefix_hit_fraction, (int, float))
                or not math.isfinite(self.prefix_hit_fraction)):
            raise TypeError("prefix_hit_fraction must be a finite number")
        if not 0.0 <= self.prefix_hit_fraction <= 1.0:
            raise ValueError("prefix_hit_fraction must be in [0, 1]")
        if not isinstance(self.arrival_distribution, str):
            raise TypeError("arrival_distribution must be a string")
        if self.arrival_distribution not in ("poisson", "constant"):
            raise ValueError("arrival_distribution must be poisson or constant")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")


def generate_synthetic_requests(config: SyntheticRequestConfig) -> RequestStream:
    rng = random.Random(config.seed)
    now = 0.0
    requests = []
    for index in range(config.num_requests):
        if index:
            if config.arrival_distribution == "poisson":
                now += rng.expovariate(config.request_rate)
            elif config.arrival_distribution == "constant":
                now += 1.0 / config.request_rate
            else:
                raise ValueError(
                    "unknown arrival distribution: "
                    f"{config.arrival_distribution!r}")
        input_length = config.input_lengths[index % len(config.input_lengths)]
        requests.append(RequestSpec(
            arrival_time_s=now,
            request_id=f"request-{index}",
            input_tokens=input_length,
            output_tokens=config.output_lengths[index % len(config.output_lengths)],
            prefix_hit_tokens=int(input_length * config.prefix_hit_fraction),
        ))
    return RequestStream(requests)
