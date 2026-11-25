"""Common output contract for synthetic and production request sources."""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Iterator, Tuple


@dataclass(frozen=True)
class RequestSpec:
    arrival_time_s: float
    request_id: str
    input_tokens: int
    output_tokens: int
    prefix_hit_tokens: int = 0
    prompt_token_ids: Tuple[int, ...] | None = None
    output_token_ids: Tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (isinstance(self.arrival_time_s, bool)
                or not isinstance(self.arrival_time_s, (int, float))
                or not math.isfinite(self.arrival_time_s)
                or self.arrival_time_s < 0):
            raise ValueError(
                "request arrival time must be finite and non-negative")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        for name in ("input_tokens", "output_tokens", "prefix_hit_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"request {name} must be an integer")
        if self.input_tokens < 1:
            raise ValueError("request input_tokens must be positive")
        if self.output_tokens < 0:
            raise ValueError("request output_tokens cannot be negative")
        if not 0 <= self.prefix_hit_tokens <= self.input_tokens:
            raise ValueError(
                "request prefix_hit_tokens must be between zero and "
                "input_tokens")

        for name, length in (("prompt_token_ids", self.input_tokens),
                             ("output_token_ids", self.output_tokens)):
            ids = getattr(self, name)
            if ids is not None:
                if not isinstance(ids, tuple):
                    raise TypeError(f"request {name} must be a tuple or null")
                if len(ids) != length:
                    raise ValueError(f"request {name} length disagrees with token count")
                if any(type(token) is not int or token < 0 for token in ids):
                    raise ValueError(f"request {name} requires nonnegative integer IDs")
        if self.output_token_ids is not None and self.prompt_token_ids is None:
            raise ValueError("output token identity requires prompt token identity")
        if self.prompt_token_ids is not None and self.prefix_hit_tokens:
            raise ValueError("token identity cannot be mixed with trace prefix hits")

    def validate_token_identity(self, vocab_size: int, *, require_complete: bool) -> None:
        """Validate optional identity against the selected model vocabulary."""
        if type(vocab_size) is not int or vocab_size < 1:
            raise ValueError("model vocabulary size must be a positive integer")
        for ids in (self.prompt_token_ids, self.output_token_ids):
            if ids is not None and any(token >= vocab_size for token in ids):
                raise ValueError("request token ID outside model vocabulary")
        if require_complete and (self.prompt_token_ids is None or (
                self.output_tokens and self.output_token_ids is None)):
            raise ValueError("token prefix cache requires real prompt and output token IDs")


@dataclass(frozen=True)
class RequestStream:
    requests: Tuple[RequestSpec, ...]

    def __init__(self, requests: Iterable[RequestSpec]):
        supplied = tuple(requests)
        if not supplied:
            raise ValueError("RequestStream requires at least one request")
        if any(not isinstance(request, RequestSpec) for request in supplied):
            raise TypeError("RequestStream requires RequestSpec objects")
        # ``sorted`` is stable, so equal-arrival requests retain source order.
        # That order is observable because the router reserves load after every
        # decision; using request_id as a secondary key would rewrite a
        # production trace's same-time ordering.
        ordered = tuple(sorted(
            supplied,
            key=lambda request: request.arrival_time_s))
        ids = [request.request_id for request in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("request IDs must be unique")
        object.__setattr__(self, "requests", ordered)

    def __iter__(self) -> Iterator[RequestSpec]:
        return iter(self.requests)

    def __len__(self) -> int:
        return len(self.requests)


def request_stream_sha256(requests: RequestStream) -> str:
    """Hash the canonical ordered request-stream contract."""

    if not isinstance(requests, RequestStream):
        raise TypeError("request stream hash requires RequestStream")
    payload = [
        {
            "arrival_time_s": request.arrival_time_s,
            "request_id": request.request_id,
            "input_tokens": request.input_tokens,
            "output_tokens": request.output_tokens,
            "prefix_hit_tokens": request.prefix_hit_tokens,
        }
        for request in requests
    ]
    for record, request in zip(payload, requests):
        if request.prompt_token_ids is not None:
            record["prompt_token_ids"] = request.prompt_token_ids
            record["output_token_ids"] = request.output_token_ids
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
