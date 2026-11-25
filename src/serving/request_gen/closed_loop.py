"""Zero-think-time closed-loop client arrival process.

A closed-loop benchmark (the InferenceX client contract) keeps a fixed
population of clients, each with at most one outstanding request: the
client's next request is submitted the instant its previous request
completes.  This module chains a canonical request sequence through such a
client population.  Client ``c`` owns the interleaved chain
``requests[c::client_count]``; every client's first request arrives at its
canonical time, and each later request is injected at the exact completion
time of the client's previous request on the engine clock. Explicit
``next_request`` policy instead gives the next canonical request to whichever
client completes, preserving physical input order across admissions.
"""

from dataclasses import replace
import math
from typing import Callable, Optional, Tuple

from .request_stream import RequestSpec, RequestStream


class ClosedLoopClients:
    """Inject canonical requests on behalf of zero-think-time clients."""

    def __init__(
        self,
        requests: RequestStream,
        client_count: int,
        arrival_transform: Optional[Callable[[RequestSpec], RequestSpec]] = None,
        *,
        successor_policy: str = "per_client",
    ):
        if not isinstance(requests, RequestStream):
            raise TypeError("closed-loop clients require a RequestStream")
        if (isinstance(client_count, bool)
                or not isinstance(client_count, int) or client_count < 1):
            raise ValueError(
                "closed-loop client count must be a positive integer")
        if arrival_transform is not None and not callable(arrival_transform):
            raise TypeError("closed-loop arrival transform must be callable")
        if successor_policy not in ("per_client", "next_request"):
            raise ValueError("closed-loop successor policy must be per_client or next_request")
        canonical = tuple(requests.requests)
        if (not canonical or len(canonical) < client_count
                or successor_policy == "per_client" and len(canonical) % client_count != 0):
            raise ValueError(
                "closed-loop requires a nonempty request stream whose length "
                f"is a multiple of the client count: {len(canonical)} "
                f"requests, {client_count} clients")
        self._canonical = canonical
        self._client_count = client_count
        self._arrival_transform = arrival_transform
        self._successor_policy = successor_policy
        self._next_request_index = client_count
        self._last_completion_s = 0.0
        self._chains: Tuple[Tuple[RequestSpec, ...], ...] = tuple(
            canonical[client::client_count] for client in range(client_count))
        self._chain_of = {
            request.request_id: client
            for client, chain in enumerate(self._chains)
            for request in chain
        }
        if len(self._chain_of) != len(canonical):
            raise ValueError("closed-loop request IDs must be unique")
        # Index of each chain's next request to inject; the initial request
        # (index 0) is issued through initial_requests().
        self._next_index = [1] * client_count
        self._outstanding: list[Optional[str]] = [None] * client_count
        self._assigned_arrival_s: dict[str, float] = {}
        self._finished: set[str] = set()
        self._last_chain_arrival_s = [math.nan] * client_count

    @property
    def client_count(self) -> int:
        return self._client_count

    @property
    def requests_per_client(self) -> int:
        if self._successor_policy != "per_client":
            raise ValueError("requests per client depends on actual completion order under next_request")
        return len(self._chains[0])

    def _visible(self, request: RequestSpec) -> RequestSpec:
        if self._arrival_transform is None:
            return request
        return self._arrival_transform(request)

    def initial_requests(self) -> Tuple[RequestSpec, ...]:
        """Issue every client's first request at its canonical arrival time."""

        initial = []
        for client, chain in enumerate(self._chains):
            request = chain[0]
            self._assigned_arrival_s[request.request_id] = request.arrival_time_s
            self._outstanding[client] = request.request_id
            self._last_chain_arrival_s[client] = request.arrival_time_s
            initial.append(self._visible(request))
        return tuple(initial)

    def notify_finished(
        self,
        request_id: str,
        finish_time_s: float,
    ) -> Optional[RequestSpec]:
        """Inject the finished client's successor at its completion time.

        Returns the successor request (scheduler-visible form) or ``None``
        when the client's chain is exhausted.  ``finish_time_s`` is the
        engine-clock completion time of ``request_id``; the successor's
        canonical arrival is stamped at exactly that instant, which is the
        zero-think-time contract.
        """

        if not isinstance(request_id, str) or not request_id:
            raise ValueError("finished request ID must be a non-empty string")
        if (isinstance(finish_time_s, bool)
                or not isinstance(finish_time_s, (int, float))
                or not math.isfinite(finish_time_s) or finish_time_s < 0):
            raise ValueError(
                "closed-loop finish time must be finite and non-negative")
        if request_id not in self._chain_of:
            raise ValueError(f"unknown closed-loop request {request_id!r}")
        if request_id in self._finished:
            raise RuntimeError(
                f"closed-loop request {request_id!r} finished more than once")
        client = self._chain_of[request_id]
        if self._outstanding[client] != request_id:
            raise RuntimeError(
                f"closed-loop client {client} finished {request_id!r} while "
                f"{self._outstanding[client]!r} was outstanding")
        if finish_time_s < self._last_chain_arrival_s[client]:
            raise RuntimeError(
                f"closed-loop client {client} finished before its request "
                "arrived")
        if (self._successor_policy == "next_request"
                and finish_time_s < self._last_completion_s):
            raise RuntimeError("next_request requires notifications in causal completion order")
        self._last_completion_s = finish_time_s
        self._finished.add(request_id)
        self._outstanding[client] = None
        if self._successor_policy == "next_request":
            index = self._next_request_index
            if index == len(self._canonical):
                return None
            successor = replace(self._canonical[index], arrival_time_s=finish_time_s)
            self._next_request_index += 1
            self._chain_of[successor.request_id] = client
        else:
            index = self._next_index[client]
            chain = self._chains[client]
            if index == len(chain):
                return None
            successor = replace(chain[index], arrival_time_s=finish_time_s)
            self._next_index[client] += 1
        self._assigned_arrival_s[successor.request_id] = finish_time_s
        self._outstanding[client] = successor.request_id
        self._last_chain_arrival_s[client] = finish_time_s
        return self._visible(successor)

    def successors_of(
        self,
        request_id: str,
        finish_time_s: float,
    ) -> Tuple[RequestSpec, ...]:
        """Successors injected by one finish, fan-out aware.

        This is the single completion contract the single-pool engine
        consumes: it is called exactly once per finished request and must
        return every successor stamped at a causal instant (the finish
        instant itself, or ``finish_time_s +`` a recorded external gap).
        The default client issues exactly one successor per finish (or none
        at chain end); fan-out controllers (e.g. AgentX sessions) override
        this to return every concurrently-arriving successor.

        The legacy ``notify_finished`` hook is kept as the inherited
        single-successor adapter: the engine never calls it directly.
        """

        successor = self.notify_finished(request_id, finish_time_s)
        if successor is None:
            return ()
        return (successor,)

    def notify_finished_batch(
        self,
        finished,
    ):
        """Process a whole step's finish set through ``successors_of``.

        Accepts either a list of ``(request_id, finish_time_s)`` pairs (a
        per-step view) or a single ``(request_id, finish_time_s)`` pair
        passed positionally (the engine's per-finish contract).  Returns
        every successor stamped at its causal instant.
        """

        if (isinstance(finished, tuple) and len(finished) == 2
                and isinstance(finished[0], str)):
            finished = (finished,)
        successors: list = []
        for request_id, finish_time_s in finished:
            successors.extend(self.successors_of(request_id, finish_time_s))
        return successors

    def effective_request_stream(self) -> RequestStream:
        """Rebuild the canonical stream at the injected arrival times.

        Fails closed unless every canonical request was injected exactly
        once, so a truncated schedule can never silently produce metrics.
        """

        if len(self._assigned_arrival_s) != len(self._canonical):
            missing = sorted(
                request.request_id for request in self._canonical
                if request.request_id not in self._assigned_arrival_s)
            raise RuntimeError(
                "closed-loop run did not inject every canonical request: "
                f"{len(missing)} missing, first={missing[:3]}")
        return RequestStream(
            replace(
                request,
                arrival_time_s=self._assigned_arrival_s[request.request_id])
            for request in self._canonical)
