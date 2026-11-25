"""Run-local per-rank work and source-memory lifecycle audit."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from src.serving.simulation import SimulationStep, _state_audit_snapshot


class ServingStateAuditAccumulator:
    """Consume each full step once and retain only auditable scalar/hash state."""

    def __init__(self):
        self._ranks = {}
        self._step_count = 0
        self._all_steps_have_source_state = True

    @property
    def step_count(self):
        return self._step_count

    @staticmethod
    def _new_rank():
        return {
            "logical_calls_by_kind": {},
            "communication_calls_by_operation": {},
            "communication_bytes_by_operation": {},
            "source_memory_domains": {},
            "source_allocation_count": 0,
            "source_allocation_bytes": 0,
            "source_freed_allocation_count": 0,
            "source_freed_allocation_bytes": 0,
            "source_live_allocation_count": 0,
            "source_live_allocation_bytes": 0,
            "source_state_chain": hashlib.sha256(),
        }

    def _rank(self, pool, rank):
        key = (pool, rank)
        if key not in self._ranks:
            self._ranks[key] = self._new_rank()
        return self._ranks[key]

    @staticmethod
    def _increment(mapping, key, value):
        mapping[key] = mapping[key] + value if key in mapping else value

    def snapshot(self, step):
        if not isinstance(step, SimulationStep):
            raise TypeError("state audit requires SimulationStep")
        if step.source_state_summary is None:
            self._all_steps_have_source_state = False
        return _state_audit_snapshot(step)

    def consume(self, step):
        self.consume_snapshot(self.snapshot(step))

    def consume_snapshot(self, snapshot):
        required = {"schema_version", "step_id", "pool_name", "source_state_summary", "calls"}
        if not isinstance(snapshot, Mapping) or set(snapshot) != required:
            raise TypeError("state audit snapshot has an invalid contract")
        if snapshot["schema_version"] != "serving_state_audit_step_v1":
            raise ValueError("state audit snapshot has an unknown schema")
        step_id, pool_name = snapshot["step_id"], snapshot["pool_name"]
        if (not isinstance(step_id, str) or not step_id
                or not isinstance(pool_name, str) or not pool_name):
            raise ValueError("state audit snapshot needs step and pool identities")
        summary = snapshot["source_state_summary"]
        if not isinstance(summary, Mapping) or not isinstance(summary["ranks"], Mapping):
            raise TypeError("state audit source summary has an invalid rank mapping")
        calls = snapshot["calls"]
        if not isinstance(calls, tuple):
            raise TypeError("state audit snapshot calls must be a tuple")
        for call in calls:
            if not isinstance(call, tuple) or len(call) != 7:
                raise TypeError("state audit snapshot has an invalid call")
            call_id, call_pool, rank, kind, count, operation, byte_count = call
            if (not isinstance(call_id, str) or not call_id
                    or not isinstance(call_pool, str) or not call_pool
                    or not isinstance(rank, str) or not rank
                    or not isinstance(kind, str) or not kind
                    or isinstance(count, bool) or not isinstance(count, int) or count <= 0
                    or (operation is not None and (not isinstance(operation, str) or not operation))
                    or isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0
                    or (operation is None and byte_count)):
                raise ValueError("state audit snapshot has invalid call facts")

        for rank, facts in summary["ranks"].items():
            rank_state = self._rank(pool_name, rank)
            for domain, domain_facts in facts["memory_domains"].items():
                domains = rank_state["source_memory_domains"]
                if domain not in domains:
                    domains[domain] = {"allocation_count": 0, "allocation_bytes": 0}
                domains[domain]["allocation_count"] += domain_facts["allocation_count"]
                domains[domain]["allocation_bytes"] += domain_facts["allocation_bytes"]
            for field in (
                    "source_allocation_count", "source_allocation_bytes",
                    "source_freed_allocation_count", "source_freed_allocation_bytes",
                    "source_live_allocation_count", "source_live_allocation_bytes"):
                source_field = field.removeprefix("source_")
                rank_state[field] += facts[source_field]
            rank_state["source_state_chain"].update(step_id.encode("utf-8"))
            rank_state["source_state_chain"].update(
                facts["allocation_state_sha256"].encode("ascii"))

        for call_id, call_pool, rank, kind, count, operation, byte_count in calls:
            rank_state = self._rank(call_pool, rank)
            self._increment(rank_state["logical_calls_by_kind"], kind, count)
            if operation is not None:
                self._increment(rank_state["communication_calls_by_operation"], operation, count)
                self._increment(rank_state["communication_bytes_by_operation"], operation,
                    byte_count * count)
        self._step_count += 1

    def manifest(self, result):
        if self._step_count == 0 or not self._all_steps_have_source_state:
            raise ValueError("state audit did not receive complete source state")
        ranks = []
        for (pool, rank), facts in sorted(self._ranks.items()):
            ranks.append({
                "pool_name": pool,
                "rank": int(rank),
                **{name: value for name, value in facts.items() if name != "source_state_chain"},
                "source_state_chain_sha256": facts["source_state_chain"].hexdigest(),
            })
        transfers = tuple({
            "transfer_id": transfer.transfer_id,
            "payload": transfer.payload,
            "source_pool_id": transfer.source_pool_id,
            "destination_pool_id": transfer.destination_pool_id,
            "resource_id": transfer.resource_id,
            "link": transfer.link,
            "reserved_resource_ids": transfer.reserved_resource_ids,
            "byte_count": transfer.byte_count,
        } for transfer in result.schedule_trace.transfers)
        return {
            "schema_version": "serving_state_audit_v1",
            "step_count": self._step_count,
            "all_steps_have_source_state": self._all_steps_have_source_state,
            "ranks": ranks,
            "transfers": transfers,
            "scope": "all logical calls, communication bytes and source allocation lifetimes; no duration fields",
        }
