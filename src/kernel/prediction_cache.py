"""Runtime work caches owned by one prediction experiment.

Code, schema and configuration loaders do not belong in this scope. Entries
are created lazily so predictors imported after experiment entry participate.
"""

from collections import OrderedDict
from collections.abc import MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps
from time import perf_counter


_RUNTIME_CACHES = ContextVar("kernel_runtime_caches", default=None)
_PUBLIC_INPUT_OBSERVER = ContextVar("kernel_public_input_observer", default=None)
_PUBLIC_INPUT_PRODUCER = ContextVar("kernel_public_input_producer", default=None)


@contextmanager
def _observe_public_resource_inputs(observer):
    """Observe current successful Forward inputs within one recording run.

    This is a private recording hook, not an input/result memo. The observer
    receives ``(kernel, hardware, value, *, capture_wall_s)`` synchronously,
    before Forward returns. The capture time must be included once alongside
    the observer's own projection cost in recording metadata; the full
    operation remains inside simulation timing.
    It must consume that current invocation's evidence without mutating the
    source input or hardware. No tuple or hardware is retained by this scope.
    """
    if not callable(observer):
        raise TypeError("public resource input observer must be callable")
    registration = (observer, object())
    token = _PUBLIC_INPUT_OBSERVER.set(registration)
    try:
        yield
    finally:
        _PUBLIC_INPUT_OBSERVER.reset(token)


@contextmanager
def _public_resource_input_producer(kernel, hardware, method):
    registration = _PUBLIC_INPUT_OBSERVER.get()
    if registration is None:
        yield
        return
    from src.kernel.hardware_spec import HWSpec
    from src.kernel.kernel_spec import KernelKind, KernelSpec
    eligible = (type(kernel) is KernelSpec and type(hardware) is HWSpec
        and kernel.kind in (KernelKind.FLASH_ATTN, KernelKind.FUSED_MOE)
        and method in ("midlayer", "roofline", "roofline_span_floor"))
    pending = []
    producer = (registration, kernel, hardware, pending) if eligible else None
    token = _PUBLIC_INPUT_PRODUCER.set(producer)
    try:
        yield
    except BaseException:
        raise
    else:
        # Delivery happens only after successful prediction. Reset first so
        # an observer's own resource export cannot recursively publish itself.
        _PUBLIC_INPUT_PRODUCER.reset(token)
        token = None
        if producer is not None and _PUBLIC_INPUT_OBSERVER.get() is registration:
            observer_token = _PUBLIC_INPUT_PRODUCER.set(None)
            try:
                for value, capture_wall_s in pending:
                    registration[0](kernel, hardware, value,
                                    capture_wall_s=capture_wall_s)
            finally:
                _PUBLIC_INPUT_PRODUCER.reset(observer_token)
    finally:
        pending.clear()
        if token is not None:
            _PUBLIC_INPUT_PRODUCER.reset(token)


def _retain_public_resource_input(kernel, hardware, value):
    """Keep only an actual implicit-default public input for this Forward."""
    producer = _PUBLIC_INPUT_PRODUCER.get()
    if (producer is None or producer[0] is not _PUBLIC_INPUT_OBSERVER.get()
            or producer[1] is not kernel or producer[2] is not hardware):
        return value
    started = perf_counter()
    from contextvars import copy_context
    from dataclasses import replace
    import math
    from src.kernel.memory_access_builder import (
        _BindingInvocation, _PREDICTION_MEMORY_BINDINGS,
    )
    context = copy_context()
    if _PREDICTION_MEMORY_BINDINGS not in context:
        return value
    invocation = context[_PREDICTION_MEMORY_BINDINGS]
    if (type(invocation) is not _BindingInvocation or not invocation.implicit
            or invocation.kernel is not kernel):
        return value
    from src.kernel.model_contract import PredictionInput
    if (type(value) is not tuple or len(value) != 2
            or type(value[0]) is not PredictionInput):
        raise TypeError("public resource observer requires the complete canonical input tuple")
    if (type(value[1]) not in (int, float) or not math.isfinite(value[1])
            or value[1] < 0):
        raise ValueError("public resource observer requires finite nonnegative compulsory bytes")
    # Canonical construction validates and copies both mutable outer mappings.
    # The remaining public fields are immutable source-contract records.
    snapshot = (replace(value[0]), value[1])
    producer[3].append((snapshot, perf_counter() - started))
    return value


@contextmanager
def prediction_cache_scope(*, reuse=False):
    """Start an isolated run, or join an explicitly active enclosing run."""
    if reuse and _RUNTIME_CACHES.get() is not None:
        yield
        return
    caches = {}
    token = _RUNTIME_CACHES.set(caches)
    try:
        yield
    finally:
        _RUNTIME_CACHES.reset(token)
        caches.clear()


def runtime_lru_cache(maxsize=128, typed=False):
    """Keep functools' key semantics and capacity within each run."""
    def decorate(builder):
        identity = object()
        factory = lambda: lru_cache(maxsize=maxsize, typed=typed)(builder)
        default = factory()

        def current():
            caches = _RUNTIME_CACHES.get()
            if caches is None:
                return default
            if identity not in caches:
                caches[identity] = factory()
            return caches[identity]

        @wraps(builder)
        def cached(*args, **kwargs):
            return current()(*args, **kwargs)

        cached.cache_info = lambda: current().cache_info()
        cached.cache_clear = lambda: current().cache_clear()
        cached.cache_parameters = default.cache_parameters
        return cached
    return decorate


class RuntimeCache(MutableMapping):
    """Run-local ordered storage for existing caller-managed eviction policies.

    Callers retain their existing locks and bounds. No scope clears another
    run's entries, including the implicit context used by direct helpers.
    """

    def __init__(self):
        self._default = OrderedDict()
        self._identity = object()

    def _current(self):
        caches = _RUNTIME_CACHES.get()
        if caches is None:
            return self._default
        identity = self._identity
        if identity not in caches:
            caches[identity] = OrderedDict()
        return caches[identity]

    def __getitem__(self, key):
        return self._current()[key]

    def __setitem__(self, key, value):
        self._current()[key] = value

    def __delitem__(self, key):
        del self._current()[key]

    def __iter__(self):
        return iter(self._current())

    def __len__(self):
        return len(self._current())

    def __contains__(self, key):
        return key in self._current()

    def clear(self):
        self._current().clear()

    def move_to_end(self, key, last=True):
        self._current().move_to_end(key, last=last)

    def popitem(self, last=True):
        return self._current().popitem(last=last)
