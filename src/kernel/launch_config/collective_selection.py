"""Versioned vendor collective algorithm and protocol selection result."""

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from src.kernel.hardware_ids import canonical_hardware_id




def _nonempty_string(value, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"collective selection {context} must be non-empty")
    return value.strip()


@dataclass(frozen=True)
class CollectiveSelection:
    """One source-bound vendor selection, independent of predictor support."""

    hardware_id: str
    communication_library: str
    library_version: str
    operation: str
    message_bytes: int
    participant_count: int
    topology: str
    vendor_algorithm: str
    vendor_protocol: Optional[str]
    vendor_executor: Optional[str]
    protocol_status: str
    rule_version: str
    evidence_run_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "hardware_id", canonical_hardware_id(self.hardware_id))
        for field_name in (
                "communication_library", "library_version", "operation",
                "topology", "vendor_algorithm", "protocol_status",
                "rule_version"):
            object.__setattr__(
                self,
                field_name,
                _nonempty_string(getattr(self, field_name), field_name),
            )
        if self.operation not in {"all_reduce", "all_gather"}:
            raise ValueError(
                f"unsupported collective selection operation: "
                f"{self.operation!r}")
        for field_name in ("message_bytes", "participant_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"collective selection {field_name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"collective selection {field_name} must be positive")
        if self.participant_count <= 1:
            raise ValueError(
                "collective selection participant_count must exceed one")
        for field_name in ("vendor_protocol", "vendor_executor"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self, field_name, _nonempty_string(value, field_name))
        if self.protocol_status == "selected":
            if self.vendor_protocol is None:
                raise ValueError(
                    "selected collective protocol must name a vendor protocol")
        elif self.protocol_status in {"unobservable", "unresolved"}:
            if self.vendor_protocol is not None:
                raise ValueError(
                    "unavailable collective protocol must remain null")
        else:
            raise ValueError(
                f"unknown collective protocol status: "
                f"{self.protocol_status!r}")
        if not isinstance(self.evidence_run_ids, tuple):
            raise TypeError(
                "collective selection evidence_run_ids must be a tuple")
        normalized_evidence = tuple(
            _nonempty_string(run_id, "evidence run id")
            for run_id in self.evidence_run_ids
        )
        if not normalized_evidence:
            raise ValueError(
                "collective selection requires at least one evidence run")
        if len(set(normalized_evidence)) != len(normalized_evidence):
            raise ValueError(
                "collective selection evidence run ids must be unique")
        object.__setattr__(self, "evidence_run_ids", normalized_evidence)


@dataclass(frozen=True)
class CollectiveRuleInterval:
    """One inclusive message-byte band and the vendor choice observed in it."""

    lower_bytes: int
    upper_bytes: int
    algorithm: str
    protocol: Optional[str]
    executor: Optional[str]
    protocol_status: str


@dataclass(frozen=True)
class CollectiveRuleTier:
    """Bands that apply to messages meeting one eligibility condition.

    A message is eligible when it is at least ``minimum_bytes`` and a multiple
    of ``alignment_bytes``.  Tiers are tried in order; the first eligible tier
    decides, and a message it leaves between bands is still bracketed.
    """

    minimum_bytes: int
    alignment_bytes: int
    intervals: Tuple[CollectiveRuleInterval, ...]


@dataclass(frozen=True)
class CollectiveRuleSet:
    """A profiled vendor collective library on one topology, as data.

    ``tiers``, ``evidence_run_ids`` and ``bracket_messages`` are keyed by
    ``(operation, participant_count)`` and must cover exactly the two
    operations ``all_reduce`` and ``all_gather`` times
    ``participant_counts``.  ``bracket_messages`` is the
    format string raised when a message falls in an unresolved band; it may
    use ``{operation}``, ``{participant_count}`` and ``{message_bytes}``.
    """

    hardware_id: str
    label: str
    library: str
    library_version: str
    topology: str
    rule_version: str
    participant_counts: Tuple[int, ...]
    max_message_bytes: int
    tiers: Mapping[Tuple[str, int], Tuple[CollectiveRuleTier, ...]]
    evidence_run_ids: Mapping[Tuple[str, int], Tuple[str, ...]]
    bracket_messages: Mapping[Tuple[str, int], str]

    def __post_init__(self) -> None:
        keys = {
            (operation, count)
            for operation in ("all_reduce", "all_gather")
            for count in self.participant_counts
        }
        for name in ("tiers", "evidence_run_ids", "bracket_messages"):
            if set(getattr(self, name)) != keys:
                raise ValueError(
                    f"{self.label} collective rules {name} must cover "
                    f"exactly {sorted(keys)}")
        for key, tiers in self.tiers.items():
            if not tiers:
                raise ValueError(
                    f"{self.label} collective rules {key} declare no tier")
            for tier in tiers:
                for band in tier.intervals:
                    if not (0 < band.lower_bytes <= band.upper_bytes):
                        raise ValueError(
                            f"{self.label} collective band {band} is empty")


def _rank_list_text(counts: Tuple[int, ...]) -> str:
    words = [str(count) for count in counts]
    if len(words) <= 2:
        return " or ".join(words)
    return ", ".join(words[:-1]) + ", or " + words[-1]


def select_declared_collective(
    rules: CollectiveRuleSet,
    *,
    communication_library: str,
    library_version: str,
    operation: str,
    message_bytes: int,
    participant_count: int,
    dtype: str,
    reduction: Optional[str],
    in_place: bool,
    topology: str,
) -> CollectiveSelection:
    """Select a profiled vendor collective rule from its declaration."""

    label = rules.label
    if communication_library != rules.library:
        raise ValueError(
            f"{label} collective selector requires {rules.library}, got "
            f"{communication_library!r}")
    if library_version != rules.library_version:
        raise ValueError(
            f"{label} collective selector supports only {rules.library} "
            f"{rules.library_version}, got {library_version!r}")
    if topology != rules.topology:
        raise ValueError(
            f"{label} collective selector supports only {rules.topology!r}")
    if dtype != "bfloat16":
        raise ValueError(f"{label} collective selector supports only bfloat16")
    if operation == "all_reduce":
        if reduction != "sum" or in_place is not True:
            raise ValueError(
                f"{label} {rules.library} all_reduce requires sum and "
                "in_place=true")
    elif operation == "all_gather":
        if reduction is not None or in_place is not False:
            raise ValueError(
                f"{label} {rules.library} all_gather requires null "
                "reduction and in_place=false")
    else:
        raise ValueError(
            f"unsupported {label} {rules.library} operation: {operation!r}")
    if participant_count not in rules.participant_counts:
        raise ValueError(
            f"{label} collective selector supports only "
            f"{_rank_list_text(rules.participant_counts)} ranks")
    if message_bytes % 2 != 0:
        raise ValueError(
            f"{label} bfloat16 message_bytes must be divisible by 2")
    if message_bytes > rules.max_message_bytes:
        raise ValueError(
            f"{label} collective message_bytes exceed the profiled "
            f"{rules.max_message_bytes // (1024 * 1024)} MiB domain")

    key = (operation, participant_count)
    for tier in rules.tiers[key]:
        if (message_bytes < tier.minimum_bytes
                or message_bytes % tier.alignment_bytes != 0):
            continue
        for band in tier.intervals:
            if band.lower_bytes <= message_bytes <= band.upper_bytes:
                return CollectiveSelection(
                    hardware_id=rules.hardware_id,
                    communication_library=rules.library,
                    library_version=rules.library_version,
                    operation=operation,
                    message_bytes=message_bytes,
                    participant_count=participant_count,
                    topology=rules.topology,
                    vendor_algorithm=band.algorithm,
                    vendor_protocol=band.protocol,
                    vendor_executor=band.executor,
                    protocol_status=band.protocol_status,
                    rule_version=rules.rule_version,
                    evidence_run_ids=rules.evidence_run_ids[key],
                )
        break
    raise NotImplementedError(rules.bracket_messages[key].format(
        operation=operation,
        participant_count=participant_count,
        message_bytes=message_bytes))
