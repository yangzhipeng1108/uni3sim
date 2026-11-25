"""Strict request-ID normalization for P2P KV transfers."""

import regex as re


_ROLE_ID_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"^(.*)___decode_addr_[^:]+:\d+$",
    r"^(.*)___prefill_addr_[^:]+:\d+___$",
    # vLLM 0.11 appends the local data-parallel rank to completion IDs in
    # scheduler-trace mode, including warmup IDs.  This suffix is local
    # bookkeeping, not part of the P2P transfer identity.
    r"^cmpl-(.*)___decode_addr_[^:]+:\d+-\d+$",
    r"^cmpl-(.*)___prefill_addr_[^:]+:\d+___-\d+$",
    r"^cmpl-(.*)___decode_addr_[^:]+:\d+-\d+-[0-9a-f]+$",
    r"^cmpl-(.*)___prefill_addr_[^:]+:\d+___-\d+-[0-9a-f]+$",
))


def transfer_request_id(request_id: str) -> str:
    """Return the shared tensor identity for the Prefill and Decode role IDs."""
    matches = [pattern.fullmatch(request_id) for pattern in _ROLE_ID_PATTERNS]
    matched = [match for match in matches if match is not None]
    if len(matched) != 1 or not matched[0].group(1):
        raise ValueError(
            f"Request id {request_id} does not satisfy the P2P role-id contract"
        )
    return matched[0].group(1)
