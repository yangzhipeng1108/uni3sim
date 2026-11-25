"""Untimed, immutable memory access contract owned by the kernel layer.

Regions are allocation-relative; views explain bindings and never add an offset
at consumption. Dependencies, not tuple order, define the execution DAG.
"""
from __future__ import annotations

from src.kernel.prediction_cache import runtime_lru_cache

from dataclasses import dataclass, fields
from enum import Enum
from functools import cached_property, lru_cache
from itertools import product
from math import prod
from types import UnionType
from typing import get_args, get_origin, get_type_hints

U64_MAX = 2**64 - 1


@lru_cache(maxsize=256)
def _compile_type_check(annotation):
    """Compile immutable schema metadata; every payload is still checked."""
    if annotation is int:
        def check(value):
            if type(value) is not int or not 0 <= value <= U64_MAX:
                raise ValueError("expected unsigned 64-bit integer")
        return check
    if annotation is str:
        def check(value):
            if type(value) is not str or not value.strip():
                raise ValueError("expected nonempty string")
        return check
    origin = get_origin(annotation)
    if origin is UnionType:
        annotations = get_args(annotation)
        candidates = tuple(_compile_type_check(item) for item in annotations)
        @lru_cache(maxsize=32)
        def matching(value_type):
            # Dispatch on schema types before checking values. Optional None
            # must not create a failed string/int exception on every field.
            # Keep exact primitive types (bool is not an unsigned integer),
            # while ordinary contract classes retain isinstance semantics.
            return tuple(candidate for item,candidate in zip(annotations,candidates)
                if (value_type is item if item in (int,str) else
                    value_type is tuple if get_origin(item) is tuple else issubclass(value_type,item)))
        def check(value):
            for candidate in matching(type(value)):
                try:
                    candidate(value)
                    return
                except (TypeError, ValueError):
                    pass
            raise TypeError(f"value does not satisfy {annotation}")
        return check
    if origin is tuple:
        args = get_args(annotation)
        variadic = len(args) == 2 and args[1] is Ellipsis
        checks = ((_compile_type_check(args[0]),) if variadic else
                  tuple(_compile_type_check(item) for item in args))
        def check(value):
            if type(value) is not tuple:
                raise TypeError("contract sequences must be immutable tuples")
            if variadic:
                for item in value:
                    checks[0](item)
            else:
                if len(value) != len(checks):
                    raise ValueError("tuple arity mismatch")
                for item, predicate in zip(value, checks):
                    predicate(item)
        return check
    def check(value):
        if not isinstance(value, annotation):
            raise TypeError(f"expected {annotation}, got {type(value)}")
    return check


def check_type(value, annotation):
    _compile_type_check(annotation)(value)


@lru_cache(maxsize=128)
def _contract_type_hints(contract_type):
    """Immutable contract schemas are resolved once, independent of payloads."""
    return get_type_hints(contract_type)


@lru_cache(maxsize=128)
def _contract_field_checks(contract_type):
    annotations = _contract_type_hints(contract_type)
    return tuple((field.name, _compile_type_check(annotations[field.name]))
                 for field in fields(contract_type))


class Typed:
    def __post_init__(self):
        for name, predicate in _contract_field_checks(type(self)):
            predicate(getattr(self, name))

    @cached_property
    def _source_identity_template_checked(self):
        """Intrinsic validation of one immutable source-template object."""
        self.__post_init__()
        return True


def _validate_record_schema(record):
    """Check current nested field types before a value-based reuse boundary.

    Dataclass freezing does not prevent object.__setattr__, and Python value
    equality merges distinct schema values such as False and integer zero.
    Check schema fields without constructing or compiling an access program.
    """
    pending = [record]
    seen = set()
    while pending:
        value = pending.pop()
        if isinstance(value, Typed):
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            Typed.__post_init__(value)
            pending.extend(getattr(value, name)
                           for name, _ in _contract_field_checks(type(value)))
        elif type(value) is tuple:
            pending.extend(value)


class Granularity(str, Enum):
    OP = "op"
    KERNEL = "kernel"
    TILE = "tile"


class Semantics(str, Enum):
    LOGICAL = "logical"
    ISSUED = "issued"
    TRANSACTION = "transaction"


class Ordering(str, Enum):
    PARTIAL = "partial_order"
    TOTAL = "total_order"
    UNORDERED = "unordered"


class AddressLevel(str, Enum):
    OBJECT = "object"
    VIRTUAL = "virtual"
    PHYSICAL = "physical"


class Operation(str, Enum):
    READ = "read"
    WRITE = "write"


class EventKind(str, Enum):
    ALLOCATED = "allocated"
    INITIALIZED = "initialized"
    LOGICAL_AVAILABLE = "logical_available"
    COMPUTE_DONE = "compute_done"
    BARRIER = "barrier"
    FREED = "freed"
    ACCEPTED = "accepted"
    BLOCKED = "blocked"
    READ_READY = "read_ready"
    WRITE_VISIBLE = "write_visible"
    RETIRED = "retired"
    WAKEUP = "wakeup"


@dataclass(frozen=True)
class Span(Typed):
    offset_bytes: int
    length_bytes: int

    def __post_init__(self):
        super().__post_init__()
        if self.length_bytes == 0 or self.offset_bytes + self.length_bytes > U64_MAX:
            raise ValueError("empty or overflowing span")


@dataclass(frozen=True)
class Strided(Typed):
    base_offset_bytes: int
    counts: tuple[int, ...]
    strides_bytes: tuple[int, ...]
    element_span_bytes: int

    def __post_init__(self):
        super().__post_init__()
        if (not self.counts or len(self.counts) != len(self.strides_bytes)
                or min(self.counts) == 0 or self.element_span_bytes == 0):
            raise ValueError("invalid strided geometry")
        if region_end(self) > U64_MAX or requested_bytes(self) > U64_MAX:
            raise ValueError("strided overflow")


@dataclass(frozen=True)
class Indexed(Typed):
    """Bounded logical gather/scatter whose index values are unavailable.

    Each of ``index_count`` indices selects a span at ``base + index*stride``;
    values lie in [0, index_bound). The index read is a dependency, not an
    invented sequence of addresses. Duplicate indices remain possible.
    """
    base_offset_bytes: int
    index_count: int
    index_bound: int
    stride_bytes: int
    element_span_bytes: int
    index_access_id: str

    def __post_init__(self):
        super().__post_init__()
        if min(self.index_count, self.index_bound, self.element_span_bytes) == 0:
            raise ValueError("empty indexed geometry")
        if region_end(self) > U64_MAX or requested_bytes(self) > U64_MAX:
            raise ValueError("indexed overflow")

    def domain(self):
        return Strided(self.base_offset_bytes, (self.index_bound,),
                       (self.stride_bytes,), self.element_span_bytes)


class KVComponent(str, Enum):
    KEY = "key"
    VALUE = "value"
    LATENT = "latent"
    SCALE = "scale"
    STATE = "state"


class Indirection(str, Enum):
    PAGED_KV_HND = "paged_kv_hnd"
    PAGED_KV_NHD = "paged_kv_nhd"
    TOKEN_SLOT = "token_slot"


@dataclass(frozen=True)
class Indirect(Typed):
    """Logical payload with an unavailable page/slot address transform.

    Shape and dtype describe actual selected tensor elements, including
    repeated/aliased slot values. Neither a capacity nor a destination extent
    follows from this count. Only nonexecutable logical programs can use it.
    """
    logical_shape: tuple[int, ...]
    dtype: str
    element_bytes: int
    index_access_id: str
    mapping: Indirection
    component: KVComponent
    parent_view_id: str | None = None
    logical_starts: tuple[int, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        from src.kernel.kernel_spec import DTYPE_BYTES
        if (len(self.logical_shape) != 3 or min(self.logical_shape) == 0
                or self.dtype not in DTYPE_BYTES or self.element_bytes != DTYPE_BYTES[self.dtype]):
            raise ValueError("indirect KV payload requires positive token/head/dimension shape and exact dtype")
        if requested_bytes(self) > U64_MAX:
            raise ValueError("indirect payload overflow")
        if ((self.parent_view_id is None)!=(not self.logical_starts)
                or self.logical_starts and len(self.logical_starts)!=len(self.logical_shape)):
            raise ValueError('indirect logical coordinates require their complete parent view')


@dataclass(frozen=True)
class SymbolicExtent(Typed):
    """A positive but unobserved logical cardinality, never its lower bound."""
    symbol_id: str
    minimum: int

    def __post_init__(self):
        super().__post_init__()
        if not self.symbol_id or self.minimum==0:raise ValueError('symbolic extent requires a positive lower bound')


@dataclass(frozen=True)
class Unmapped(Typed):
    """Selected logical tensor payload without a storage-address transform.

    The selection identity denotes semantic coordinates, never a fabricated
    index sequence or compact physical array. Footprint/capacity are unknown.
    """
    logical_shape: tuple[int | SymbolicExtent, ...]
    dtype: str | None
    element_bytes: int | None
    selection_id: str

    def __post_init__(self):
        super().__post_init__()
        from src.kernel.kernel_spec import DTYPE_BYTES
        lower=tuple(value.minimum if isinstance(value,SymbolicExtent) else value for value in self.logical_shape)
        if self.dtype is None or self.element_bytes is None:
            if (self.dtype is not None or self.element_bytes is not None or not self.logical_shape
                    or min(lower)==0 or prod(lower)>U64_MAX):
                raise ValueError('unbound storage encoding requires complete unknown dtype/width and bounded logical elements')
            return
        if (not self.logical_shape or min(lower) == 0 or self.dtype not in DTYPE_BYTES
                or self.element_bytes != DTYPE_BYTES[self.dtype]):
            raise ValueError("unmapped logical payload requires positive shape and exact storage dtype/width")
        if prod(lower)*self.element_bytes > U64_MAX:
            raise ValueError("unmapped payload overflow")


Region = Span | Strided | Indexed | Indirect | Unmapped


def region_end(region: Region) -> int:
    if isinstance(region, (Indirect, Unmapped)):
        raise NotImplementedError("unresolved address extent is unavailable")
    if isinstance(region, Span):
        return region.offset_bytes + region.length_bytes
    if isinstance(region, Strided):
        return (region.base_offset_bytes + region.element_span_bytes
                + sum((n - 1) * s for n, s in zip(region.counts, region.strides_bytes)))
    if isinstance(region, Indexed):
        return (region.base_offset_bytes + (region.index_bound - 1) * region.stride_bytes
                + region.element_span_bytes)
    raise NotImplementedError("unsupported region")


def requested_bytes(region: Region) -> int:
    if isinstance(region, (Indirect, Unmapped)):
        if region.element_bytes is None:raise ValueError('logical storage encoding is unbound; requested bytes are unavailable')
        if any(isinstance(value,SymbolicExtent) for value in region.logical_shape):
            raise ValueError('symbolic logical cardinality is unbound; requested bytes are unavailable')
        return prod(region.logical_shape) * region.element_bytes
    if isinstance(region, Span):
        return region.length_bytes
    if isinstance(region, Strided):
        return prod(region.counts) * region.element_span_bytes
    if isinstance(region, Indexed):
        return region.index_count * region.element_span_bytes
    raise NotImplementedError("unsupported region")


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return tuple(merged)


def _footprint_geometry(region: Region):
    """Canonical interval lattice, without requested-access multiplicity.

    Equal strides and zero strides alias storage. Collapsing them is valid
    for a footprint, while ``requested_bytes`` keeps every occurrence.
    """
    if isinstance(region, Span):
        return region.offset_bytes, region.length_bytes, ()
    if not isinstance(region, Strided):
        raise NotImplementedError("unsupported region")
    width = region.element_span_bytes
    dims = []
    for stride, count in sorted(zip(region.strides_bytes, region.counts)):
        if count == 1 or stride == 0:
            continue
        if dims and dims[-1][0] == stride:
            dims[-1] = (stride, dims[-1][1] + count - 1)
        else:
            dims.append((stride, count))
    remaining = []
    for stride, count in dims:
        if stride <= width:
            width += (count - 1) * stride
        else:
            remaining.append((stride, count))
    return region.base_offset_bytes, width, tuple(remaining)


def region_intervals(region: Region):
    base, width, remaining = _footprint_geometry(region)
    if prod(count for _, count in remaining) > 1_000_000:
        raise NotImplementedError("exact interval expansion exceeds supported budget")
    return merge_intervals(
        (base + sum(i * s for i, (s, _) in zip(indices, remaining)),
         base + sum(i * s for i, (s, _) in zip(indices, remaining)) + width)
        for indices in product(*(range(n) for _, n in remaining)))


def _separated_geometry(width, dims):
    extent = width
    for stride, count in dims:
        if stride < extent:
            return False
        extent += (count - 1) * stride
    return True


def _symbolically_covers(container: Region, target: Region):
    """Prove rectangular subregions and spans without enumerating addresses.

    False means no proof; callers can still use bounded exact expansion.
    A bounding span is never used to fill a hole in a strided region.
    """
    if isinstance(container, Unmapped) or isinstance(target, Unmapped):
        return container == target
    if isinstance(container, Indirect) or isinstance(target, Indirect):
        return False
    if isinstance(container, Indexed):
        # The same index-read snapshot selects the same logical set. It does
        # not initialize its bounding domain or any independent index read.
        return container == target
    if isinstance(target, Indexed):
        target = target.domain()
    base, width, dims = _footprint_geometry(container)
    target_base, target_width, target_dims = _footprint_geometry(target)
    if target_base < base or region_end(target) > region_end(container):
        return False
    if not dims:
        return True
    if not _separated_geometry(width, dims):
        return container == target
    remaining = target_base - base
    coordinates = {}
    for stride, count in reversed(dims):
        index, remaining = divmod(remaining, stride)
        if index >= count:
            return False
        coordinates[stride] = index
    if remaining + target_width > width:
        return False
    counts = dict(dims)
    return all(stride in counts and coordinates[stride] + count <= counts[stride]
               for stride, count in target_dims)


def covers(container: Region, target: Region):
    if isinstance(container, Unmapped) or isinstance(target, Unmapped):
        return container == target
    if isinstance(container, Indirect) or isinstance(target, Indirect):
        return False
    if _symbolically_covers(container, target):
        return True
    if isinstance(container, Indexed):
        return False
    if isinstance(target, Indexed):
        return covers(container, target.domain())
    if (_footprint_geometry(target)[0] < _footprint_geometry(container)[0]
            or region_end(target) > region_end(container)):
        return False
    outer = region_intervals(container)
    return all(any(a <= c and d <= b for a, b in outer)
               for c, d in region_intervals(target))


def _union_covers(regions: tuple[Region, ...], target: Region):
    if isinstance(target, Unmapped):
        return any(region == target for region in regions)
    if isinstance(target, Indirect):
        return False
    if not regions:
        return False
    if any(_symbolically_covers(region, target) for region in regions):
        return True
    # An unresolved writer cannot prove visibility of other addresses.
    exact_regions = tuple(region for region in regions if not isinstance(region, (Indexed, Indirect, Unmapped)))
    if not exact_regions:
        return False
    if isinstance(target, Indexed):
        target = target.domain()
    # Adjacent initialized spans can jointly cover a huge strided read.
    spans = merge_intervals(
        (base, base + width) for base, width, dims in
        (_footprint_geometry(region) for region in exact_regions) if not dims)
    if any(_symbolically_covers(Span(start, end - start), target)
           for start, end in spans):
        return True
    union = merge_intervals(interval for region in exact_regions
                            for interval in region_intervals(region))
    return all(any(a <= c and d <= b for a, b in union)
               for c, d in region_intervals(target))


@dataclass(frozen=True)
class MemoryAllocation(Typed):
    allocation_id: str
    device_id: str
    memory_domain: str
    size_bytes: int | None
    address_level: AddressLevel
    base_address_bytes: int | None
    mapping_ref: str | None
    allocated_event: str
    freed_event: str | None

    def with_lifetime(self, allocation_id: str, allocated_event: str, freed_event: str | None):
        """Rebind validated storage geometry to independently checked event IDs."""
        if type(self) is not MemoryAllocation or "size_bytes" not in self.__dict__:
            raise TypeError("lifetime derivation requires a validated canonical MemoryAllocation")
        check_type(allocation_id, str)
        check_type(allocated_event, str)
        if freed_event is not None:
            check_type(freed_event, str)
        derived = object.__new__(MemoryAllocation)
        derived.__dict__.update(self.__dict__)
        object.__setattr__(derived, "allocation_id", allocation_id)
        object.__setattr__(derived, "allocated_event", allocated_event)
        object.__setattr__(derived, "freed_event", freed_event)
        return derived

    def __post_init__(self):
        super().__post_init__()
        if self.size_bytes is None and self.address_level is not AddressLevel.OBJECT:
            raise ValueError("unknown capacity requires an unaddressed logical object")
        if self.size_bytes == 0:
            raise ValueError("empty allocation")
        if self.address_level is AddressLevel.OBJECT:
            if self.base_address_bytes is not None or self.mapping_ref is not None:
                raise ValueError("object binding must not invent an address mapping")
        elif self.base_address_bytes is None or self.mapping_ref is None:
            raise ValueError("addressed allocation requires base and mapping contract")
        elif self.base_address_bytes + self.size_bytes > U64_MAX:
            raise ValueError("address overflow")


@dataclass(frozen=True)
class TensorView(Typed):
    view_id: str
    allocation_id: str
    offset_bytes: int
    shape: tuple[int, ...]
    strides_bytes: tuple[int, ...]
    dtype: str
    element_bytes: int

    def __post_init__(self):
        super().__post_init__()
        from src.kernel.kernel_spec import DTYPE_BYTES
        if self.dtype not in DTYPE_BYTES or self.element_bytes != DTYPE_BYTES[self.dtype]:
            raise NotImplementedError("unknown or packed dtype storage layout")
        object.__setattr__(self, "_region", Strided(
            self.offset_bytes, self.shape, self.strides_bytes, self.element_bytes))

    def region(self):
        return self._region

    def with_identity(self, view_id: str, allocation_id: str) -> TensorView:
        """Bind validated immutable layout to independently validated IDs."""
        if type(self) is not TensorView or "_region" not in self.__dict__:
            raise TypeError("identity derivation requires a validated canonical TensorView")
        check_type(view_id, str)
        check_type(allocation_id, str)
        derived = object.__new__(TensorView)
        derived.__dict__.update(self.__dict__)
        object.__setattr__(derived, "view_id", view_id)
        object.__setattr__(derived, "allocation_id", allocation_id)
        return derived

    def __getstate__(self):
        return {field.name: getattr(self, field.name) for field in fields(type(self))}

    def __setstate__(self, state):
        names = tuple(field.name for field in fields(type(self)))
        if type(state) is not dict or set(state) != set(names):
            raise ValueError("invalid TensorView pickle fields")
        for name in names:
            object.__setattr__(self, name, state[name])
        self.__post_init__()


@dataclass(frozen=True)
class LogicalCoordinateTerm(Typed):
    axis: int
    coefficient: int

    def __post_init__(self):
        super().__post_init__()
        if self.coefficient == 0:raise ValueError('logical coordinate term requires a positive coefficient')


@dataclass(frozen=True)
class LogicalCoordinateBinding(Typed):
    """Parent coordinate from child logical coordinates; never a byte address."""
    axis: int
    terms: tuple[LogicalCoordinateTerm, ...]
    offset: int = 0
    divisor: int = 1
    modulus: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.divisor == 0 or self.modulus == 0:
            raise ValueError('logical coordinate divisor/modulus must be positive')


@dataclass(frozen=True)
class LogicalSelectionView(Typed):
    """A typed, possibly repeated selection with optional logical membership.

    Shape counts logical selections, never distinct capacity. There are no
    offsets or strides to accidentally resolve to addresses. A root selection
    is authorized by exact logical availability; a child retains its parent.
    A coordinate map relates logical axes only and never resolves storage.
    """
    view_id: str
    allocation_id: str
    shape: tuple[int | SymbolicExtent, ...]
    dtype: str | None
    element_bytes: int | None
    selection_id: str
    parent_view_id: str | None
    coordinate_map: tuple[LogicalCoordinateBinding, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        from src.kernel.kernel_spec import DTYPE_BYTES
        if self.dtype is None and self.element_bytes is None:
            self.region()
            return
        if self.dtype not in DTYPE_BYTES or self.element_bytes != DTYPE_BYTES[self.dtype]:
            raise NotImplementedError("logical selection requires its actual scalar storage type")
        self.region()

    def region(self):
        return Unmapped(self.shape, self.dtype, self.element_bytes, self.selection_id)

    def _validate_parent_coordinates(self,parent_shape):
        unknown=self.unbound_parent_axes if isinstance(self,LogicalPartialSelectionView) else None
        _validate_logical_parent_geometry(self.shape,self.coordinate_map,parent_shape,unknown)


@dataclass(frozen=True)
class LogicalPartialSelectionView(LogicalSelectionView):
    """Known parent coordinates plus explicit unresolved selection axes.

    Keep the original view's wire type unchanged. Only this variant admits
    a partial coordinate map; an unknown coordinate never means zero.
    """
    unbound_parent_axes: tuple[int, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        known=tuple(binding.axis for binding in self.coordinate_map)
        unknown=self.unbound_parent_axes
        if (self.parent_view_id is None or not known or not unknown
                or known!=tuple(sorted(set(known))) or unknown!=tuple(sorted(set(unknown)))
                or set(known)&set(unknown) or sorted((*known,*unknown))!=list(range(len(known)+len(unknown)))):
            raise ValueError('partial logical selection must partition known and unbound parent axes')


@runtime_lru_cache(maxsize=256)
def _validate_logical_parent_geometry(shape,mapping,parent_shape,unbound_axes):
    """Reuse only arithmetic with every consumed geometry value in the key."""
    bound=tuple(binding.axis for binding in mapping)
    expected=tuple(axis for axis in range(len(parent_shape)) if unbound_axes is None or axis not in unbound_axes)
    if bound!=expected or unbound_axes is not None and any(axis>=len(parent_shape) for axis in unbound_axes):
        raise ValueError('logical coordinate map must bind every parent axis exactly once')
    for binding in mapping:
        terms=unique_by(binding.terms,'axis')
        if any(axis>=len(shape) for axis in terms):raise ValueError('unknown child logical coordinate')
        maximum=binding.offset+sum((shape[term.axis]-1)*term.coefficient for term in binding.terms)
        if maximum>U64_MAX:raise ValueError('logical coordinate expression exceeds uint64')
        maximum//=binding.divisor
        if binding.modulus is not None:maximum=min(maximum,binding.modulus-1)
        if maximum>=parent_shape[binding.axis]:raise ValueError('logical coordinate map exceeds parent extent')


def logical_parent_coordinates(view: LogicalSelectionView, coordinates: tuple[int, ...]):
    """Evaluate logical membership; partial maps return None on unbound axes."""
    if not isinstance(view,LogicalSelectionView) or not view.coordinate_map or view.parent_view_id is None:
        raise NotImplementedError('logical parent coordinate mapping is unknown')
    check_type(coordinates,tuple[int,...])
    if len(coordinates) != len(view.shape) or any(index >= extent for index,extent in zip(coordinates,view.shape)):
        raise ValueError('child logical coordinate outside view')
    partial=isinstance(view,LogicalPartialSelectionView)
    result = [None]*(len(view.coordinate_map)+len(view.unbound_parent_axes)) if partial else []
    for binding in view.coordinate_map:
        value = (binding.offset+sum(coordinates[term.axis]*term.coefficient for term in binding.terms))//binding.divisor
        value=value if binding.modulus is None else value%binding.modulus
        if partial:result[binding.axis]=value
        else:result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class PackedTensorView(Typed):
    """Explicit affine nibble layout; regions describe touched storage bytes.

    The inner dimension is contiguous four-bit storage. Outer strides are
    byte-aligned. Vendor permutations must be resolved by the source builder,
    rather than inferred from a logical matrix shape here.
    """
    view_id: str
    allocation_id: str
    offset_bits: int
    shape: tuple[int, ...]
    strides_bits: tuple[int, ...]
    dtype: str
    element_bits: int

    def __post_init__(self):
        super().__post_init__()
        if self.dtype not in ("int4", "mxfp4") or self.element_bits != 4:
            raise NotImplementedError("unsupported packed element encoding")
        if (not self.shape or min(self.shape) == 0
                or len(self.shape) != len(self.strides_bits)
                or self.strides_bits[-1] != 4 or self.offset_bits % 4
                or any(stride % 8 for stride in self.strides_bits[:-1])):
            raise ValueError("packed view requires aligned affine nibble storage")
        self.region()

    def region(self):
        base, bit_offset = divmod(self.offset_bits, 8)
        row_bytes = (bit_offset + self.shape[-1] * self.element_bits + 7) // 8
        if len(self.shape) == 1:
            return Span(base, row_bytes)
        return Strided(base, self.shape[:-1],
                       tuple(stride // 8 for stride in self.strides_bits[:-1]), row_bytes)


@dataclass(frozen=True)
class AccessScope(Typed):
    scope_id: str
    op_ids: tuple[str, ...]
    call_id: str
    tile: tuple[int, ...]
    rank: int
    stream: str
    agent: str


@dataclass(frozen=True)
class EventRef(Typed):
    node_id: str
    kind: EventKind
    allow_skipped_read: bool = False


@dataclass(frozen=True)
class TileWriteSource(Typed):
    tile_program_id: str
    access_id: str


@dataclass(frozen=True)
class ProgramEvent(Typed):
    event_id: str
    kind: EventKind
    allocation_id: str | None
    region: Region | None
    visibility_domain: str | None
    dependencies: tuple[EventRef, ...]
    view_id: str | None = None
    tile_sources: tuple[TileWriteSource, ...] = ()


class ConstantValue(str, Enum):
    ZERO = 'zero'
    NEGATIVE_ONE = 'negative_one'
    NEGATIVE_INFINITY = 'negative_infinity'

@dataclass(frozen=True)
class ReadValuePredicate(Typed):
    """A candidate access executes only when its actual signed scalar read fits."""
    source_access_id: str
    relation: str

    def __post_init__(self):
        super().__post_init__()
        if not self.source_access_id or self.relation!='nonnegative':
            raise ValueError('unknown read-value access predicate')


@dataclass(frozen=True)
class MemoryAccess(Typed):
    access_id: str
    scope_id: str
    allocation_id: str
    view_id: str | None
    operation: Operation
    region: Region
    semantics: Semantics
    entry: str | None
    source_engine: str | None
    policy: str | None
    request_shape_ref: str | None
    visibility_domain: str
    dependencies: tuple[EventRef, ...]
    completion: EventKind
    repeat: int
    evidence_refs: tuple[str, ...]
    constant_value: ConstantValue | None = None
    predicate: ReadValuePredicate | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.constant_value is not None and self.operation is not Operation.WRITE:
            raise ValueError("constant value requires an actual write")
        expected = EventKind.READ_READY if self.operation is Operation.READ else EventKind.WRITE_VISIBLE
        if self.completion is not expected or self.repeat == 0:
            raise ValueError("invalid access completion or repeat")
        unbound_encoding=isinstance(self.region,Unmapped) and self.region.element_bytes is None
        if unbound_encoding and self.semantics is not Semantics.LOGICAL:
            raise ValueError('unbound storage encoding requires logical accesses')
        if not unbound_encoding and requested_bytes(self.region) * self.repeat > U64_MAX:
            raise ValueError("access byte count overflow")
        if self.repeat != 1 and self.operation is not Operation.READ:
            raise NotImplementedError("compressed repeat supports independent reads only")
        if self.semantics is not Semantics.LOGICAL:
            if self.entry is None or self.source_engine is None or not self.evidence_refs:
                raise ValueError("issued/transaction access requires source and entry evidence")
        if self.semantics is Semantics.TRANSACTION and self.request_shape_ref is None:
            raise ValueError("transaction access requires request geometry")


@dataclass(frozen=True)
class Coverage(Typed):
    granularity: Granularity
    semantics: Semantics
    ordering: Ordering
    executable: bool
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class TileAxis(Typed):
    axis_id: str
    extent: int | SymbolicExtent
    block: int
    value_limit_access_id: str | None = None
    value_limit_start: int = 0
    value_limit_inclusive: bool = False
    value_limit_upper_bound: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.extent,SymbolicExtent):
            if self.block!=1 or self.value_limit_access_id is not None:
                raise ValueError('symbolic tile cardinality requires a unit logical axis without a device scalar bound')
            self.extent.__post_init__()
        elif min(self.extent,self.block)==0:
            raise ValueError('tile axis requires positive extent/block')
        if not self.axis_id:
            raise ValueError("tile axis requires a name and positive extent/block")
        if self.value_limit_access_id is None:
            if self.value_limit_start or self.value_limit_inclusive or self.value_limit_upper_bound is not None:
                raise ValueError('scalar loop range requires its actual READ identity')
        elif self.scalar_limit_upper_bound > U64_MAX:
            raise ValueError('scalar loop range exceeds its integer envelope')

    @property
    def scalar_limit_upper_bound(self):
        if self.value_limit_access_id is None:raise ValueError('tile axis has no scalar READ limit')
        return (self.extent+self.value_limit_start-int(self.value_limit_inclusive)
            if self.value_limit_upper_bound is None else self.value_limit_upper_bound)

    @property
    def count(self):
        if isinstance(self.extent,SymbolicExtent):raise ValueError('symbolic tile cardinality is unbound')
        return (self.extent+self.block-1)//self.block


@dataclass(frozen=True)
class TileIndexTerm(Typed):
    axis_id: str
    coefficient: int


@dataclass(frozen=True)
class TileIndexBinding(Typed):
    """Source tile index = floor((offset + affine target indices)/divisor) % modulus."""
    axis_id: str
    terms: tuple[TileIndexTerm, ...]
    offset: int = 0
    divisor: int = 1
    modulus: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.divisor == 0 or self.modulus == 0:
            raise ValueError('tile coordinate divisor/modulus must be positive')


@dataclass(frozen=True)
class TileDependency(Typed):
    access_id: str
    kind: EventKind
    reduction_axes: tuple[str, ...]
    coordinate_bindings: tuple[TileIndexBinding, ...] = ()
    carry_axis: str | None = None
    initial_access_id: str | None = None
    partition_axis: str | None = None
    allow_skipped_read: bool = False


@dataclass(frozen=True)
class TileIndirectTarget(Typed):
    allocation_id: str
    index_access_id: str
    mapping: Indirection
    component: KVComponent
    value_source_access_ids: tuple[str, ...] = ()





@dataclass(frozen=True)
class TileValueIndex(Typed):
    """Selection from an integer load, atomic return or complete row argmax.

    The relation preserves known logical membership. Concrete winners,
    addresses and atomic linearization still require a value-aware consumer.
    """
    source_access_id: str
    value_kind: str

    def __post_init__(self):
        super().__post_init__()
        if self.value_kind not in ('loaded_scalar','atomic_previous','argmax_index'):
            raise ValueError('unknown tile index value kind')


@dataclass(frozen=True)
class TileAtomicAdd(Typed):
    """One indivisible integer RMW; its previous value remains a symbolic result.

    Updates to equal keys are linearizable, without selecting an order among
    independent actors or resolving unknown index-equivalence classes.
    """
    read_access_id: str
    addend: int


@dataclass(frozen=True)
class TileAccessTemplate(Typed):
    access_id: str
    view_id: str
    operation: Operation
    iteration_axes: tuple[str, ...]
    view_axes: tuple[str | None, ...]
    input_ready: tuple[EventRef, ...]
    dependencies: tuple[TileDependency, ...]
    evidence_refs: tuple[str, ...]
    unmapped: bool = False
    indirect_target: TileIndirectTarget | None = None
    view_axis_divisors: tuple[int, ...] = ()
    scope: AccessScope | None = None
    view_axis_padding: tuple[bool, ...] = ()
    constant_value: ConstantValue | None = None
    value_index: TileValueIndex | None = None
    atomic_add: TileAtomicAdd | None = None
    predicate: ReadValuePredicate | None = None


@dataclass(frozen=True)
class TileCausalPrefix(Typed):
    """Clipped integer prefix; omitted offset retains end-aligned causality."""
    axis_id: str
    index_axis: str
    offset: int | None = None
    index_coefficient: int = 1
    divisor: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.index_coefficient == 0 or self.divisor == 0:
            raise ValueError('causal prefix requires positive coefficient and divisor')


@dataclass(frozen=True)
class TilePartition(Typed):
    """Equivalence classes of a logical unit axis; None keeps membership unbound.

    This is a semantic grouping, not an integer metadata tensor or a claim
    about native memory reads. With class_axis_id, members name output class
    coordinates for each context_axes tile tuple; otherwise consumers select
    the equivalence class of their own member. Class-to-member dependencies
    select the member's single producer class. A member axis may have several
    maps only when the actual producer/consumer axes select one unambiguously.
    None never assigns membership.
    """
    axis_id: str
    members: tuple[int, ...] | None = None
    class_axis_id: str | None = None
    context_axes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TileProgram(Typed):
    tile_program_id: str
    scope: AccessScope
    axes: tuple[TileAxis, ...]
    accesses: tuple[TileAccessTemplate, ...]
    visibility_domain: str
    evidence_refs: tuple[str, ...]
    causal_prefixes: tuple[TileCausalPrefix, ...] = ()
    partitions: tuple[TilePartition, ...] = ()


_SOURCE_IDENTITY_FIELDS = {
    AccessScope: frozenset(('scope_id','call_id','rank','stream')),
    MemoryAllocation: frozenset(('allocation_id','allocated_event')),
    TensorView: frozenset(('view_id','allocation_id')),
    LogicalSelectionView: frozenset(('view_id','allocation_id','selection_id','parent_view_id')),
    ProgramEvent: frozenset(('event_id','allocation_id')),
    TileAccessTemplate: frozenset(('view_id','input_ready','scope')),
    TileProgram: frozenset(('tile_program_id','scope','visibility_domain','accesses')),
}


@lru_cache(maxsize=8)
def _source_identity_checks(kind):
    allowed=_SOURCE_IDENTITY_FIELDS[kind]
    schema=_contract_field_checks(kind)
    return ({name:predicate for name,predicate in schema if name in allowed},
        frozenset(name for name,_ in schema))


def _replace_source_identity(value, **changes):
    """Rebind a checked immutable template without rechecking fixed geometry.

    Only these opaque identities/context fields may change. Allocation/view
    geometry and dtype invariants remain exactly those of the fully checked
    template; the other allowed classes have only Typed field invariants.
    Every changed field is checked with its canonical schema predicate, and
    the final MemoryAccessProgram still checks actual bindings and dataflow.
    """
    kind=type(value)
    if kind not in _SOURCE_IDENTITY_FIELDS or not changes.keys()<=_SOURCE_IDENTITY_FIELDS[kind]:
        raise TypeError('source identity rebinding cannot change geometry or unsupported contract fields')
    _ = value._source_identity_template_checked
    checks,field_names=_source_identity_checks(kind)
    for name,changed in changes.items():checks[name](changed)
    payload=value.__dict__.copy()
    del payload['_source_identity_template_checked']
    if kind is TensorView:
        if payload.keys() != field_names | {'_region'}:
            raise TypeError('TensorView template contains undeclared state')
        return value.with_identity(
            changes['view_id'] if 'view_id' in changes else value.view_id,
            changes['allocation_id'] if 'allocation_id' in changes else value.allocation_id)
    if payload.keys()!=field_names:
        raise TypeError('source identity template contains state outside its declared contract')
    payload.update(changes)
    result=object.__new__(kind)
    result.__dict__.update(payload)
    return result


class _ProgramValueHash:
    """Cache an immutable value without a descriptor lock inherited by fork.

    A recycled pool worker may fork while another parent thread hashes a
    program. Python 3.10 cached_property would retain that vanished thread's
    shared lock. Concurrent first reads may safely repeat this pure hash;
    subsequent reads use the same instance dictionary entry directly.
    """

    def __get__(self, program, owner=None):
        if program is None:
            return self
        value = hash(tuple(getattr(program, field.name) for field in fields(program)))
        program.__dict__["_value_hash"] = value
        return value


@dataclass(frozen=True)
class MemoryAccessProgram(Typed):
    version: str
    program_id: str
    hardware_id: str
    binding_id: str
    coverage: Coverage
    allocations: tuple[MemoryAllocation, ...]
    views: tuple[TensorView | PackedTensorView | LogicalSelectionView | LogicalPartialSelectionView, ...]
    scopes: tuple[AccessScope, ...]
    events: tuple[ProgramEvent, ...]
    accesses: tuple[MemoryAccess, ...]
    tile_programs: tuple[TileProgram, ...] = ()

    # Include every declared field; dataclass replacement creates a fresh cache.
    _value_hash = _ProgramValueHash()

    def __hash__(self):
        return self._value_hash

    def __post_init__(self):
        super().__post_init__()
        _validate_memory_access_program(self)


@runtime_lru_cache(maxsize=8)
def _validate_memory_access_program(program):
    """Reuse validation of an identical complete immutable program.

    Construction and source consumption both validate. The complete value,
    including caller/storage bindings and every dependency, is the key; a
    different payload must pass the same checks before it can be reused.
    """
    if program.version != "midlayer.memory-access.v1":
        raise ValueError("unsupported memory access version")
    from src.kernel.hardware_ids import CANONICAL_HARDWARE_IDS
    if program.hardware_id not in CANONICAL_HARDWARE_IDS:
        raise ValueError("unknown hardware ID")
    symbolic=program_symbolic_extents(program)
    checked=_symbolic_program_projection(program,{name:value.minimum for name,value in symbolic.items()}) if symbolic else program
    validate_program(checked)
    validate_tile_programs(checked)


def program_symbolic_extents(program):
    """Collect and validate shared cardinalities without resolving their values."""
    result={};declared=set()
    def remember(value):
        if isinstance(value,SymbolicExtent):
            value.__post_init__()
            if value.symbol_id in result and result[value.symbol_id]!=value:raise ValueError('symbolic extent declaration conflicts')
            result[value.symbol_id]=value
    for family in program.tile_programs:
        for axis in family.axes:
            remember(axis.extent)
            if isinstance(axis.extent,SymbolicExtent):declared.add(axis.extent.symbol_id)
    for view in program.views:
        for value in view.shape:remember(value)
    for event in program.events:
        if isinstance(event.region,Unmapped):
            for value in event.region.logical_shape:remember(value)
    if set(result)!=declared:raise ValueError('symbolic view/event cardinality requires its actual tile axis')
    if not result:return result
    if (program.coverage.executable or program.coverage.semantics is not Semantics.LOGICAL or program.accesses
            or program.coverage.granularity is not Granularity.TILE):
        raise ValueError('unbound cardinality requires nonexecutable compressed logical tiles')
    allocations=unique_by(program.allocations,'allocation_id');views=unique_by(program.views,'view_id')
    for view in program.views:
        if any(isinstance(value,SymbolicExtent) for value in view.shape):
            allocation=allocations[view.allocation_id]
            if type(view) is not LogicalSelectionView or view.coordinate_map or allocation.address_level is not AddressLevel.OBJECT:
                raise ValueError('symbolic selections require logical membership without address claims')
            if view.parent_view_id is None:
                if allocation.size_bytes is not None:
                    raise ValueError('symbolic root selection requires unknown capacity')
            elif (view.parent_view_id not in views or not isinstance(views[view.parent_view_id],TensorView)
                    or (views[view.parent_view_id].allocation_id,views[view.parent_view_id].dtype,views[view.parent_view_id].element_bytes)
                        != (view.allocation_id,view.dtype,view.element_bytes)):
                raise ValueError('symbolic repeated selection requires its actual typed field parent')
    for family in program.tile_programs:
        axes=unique_by(family.axes,'axis_id');unknown={a.axis_id for a in family.axes if isinstance(a.extent,SymbolicExtent)}
        if not unknown and not any(isinstance(value,SymbolicExtent) for access in family.accesses for value in views[access.view_id].shape):continue
        if family.causal_prefixes:raise ValueError('symbolic cardinality cannot assert fixed causal limits')
        for partition in family.partitions:
            if partition.axis_id in unknown or set(partition.context_axes)&unknown or partition.class_axis_id in unknown and partition.members is not None:
                raise ValueError('symbolic class membership must remain unbound')
        for access in family.accesses:
            view=views[access.view_id]
            if access.value_index is not None or access.atomic_add is not None or access.predicate is not None or access.indirect_target is not None:
                raise ValueError('symbolic cardinality requires a consumer before value-dependent native operations')
            for index,(extent,name) in enumerate(zip(view.shape,access.view_axes,strict=True)):
                if isinstance(extent,SymbolicExtent) or name in unknown:
                    if (not isinstance(extent,SymbolicExtent) or name not in unknown or axes[name].extent!=extent or not access.unmapped
                            or access.view_axis_divisors and access.view_axis_divisors[index]!=1
                            or access.view_axis_padding and access.view_axis_padding[index]):
                        raise ValueError('symbolic view dimension must map its exact logical class axis')
            for dependency in access.dependencies:
                if any(binding.axis_id in unknown or any(term.axis_id in unknown for term in binding.terms)
                        for binding in dependency.coordinate_bindings):
                    raise ValueError('unbound symbolic cardinality cannot assert selected class coordinates')
    return result


def _symbolic_program_projection(program,bindings):
    """Structural validation witness only; never returned as the unbound source."""
    from copy import copy
    from dataclasses import replace
    def shape(values):return tuple(bindings[v.symbol_id] if isinstance(v,SymbolicExtent) else v for v in values)
    checked=copy(program)
    object.__setattr__(checked,'views',tuple(replace(v,shape=shape(v.shape)) if any(isinstance(n,SymbolicExtent) for n in v.shape) else v for v in program.views))
    object.__setattr__(checked,'events',tuple(replace(e,region=replace(e.region,logical_shape=shape(e.region.logical_shape)))
        if isinstance(e.region,Unmapped) and any(isinstance(n,SymbolicExtent) for n in e.region.logical_shape) else e for e in program.events))
    object.__setattr__(checked,'tile_programs',tuple(replace(f,axes=tuple(replace(a,extent=bindings[a.extent.symbol_id])
        if isinstance(a.extent,SymbolicExtent) else a for a in f.axes)) for f in program.tile_programs))
    return checked


def bind_symbolic_extents(program,bindings):
    """Bind explicit consumer facts; metadata alone never supplies these values."""
    from dataclasses import replace
    program.__post_init__();symbols=program_symbolic_extents(program)
    if type(bindings) is not dict or set(bindings)!=set(symbols):raise ValueError('bind exactly the declared symbolic extents')
    for name,value in bindings.items():
        check_type(value,int)
        if value<symbols[name].minimum:raise ValueError('symbolic extent is below its declared lower bound')
    result=_symbolic_program_projection(program,bindings)
    return replace(result,coverage=replace(result.coverage,limitations=(*result.coverage.limitations,
        'Consumer-supplied logical cardinalities: '+', '.join(f'{name}={value}' for name,value in sorted(bindings.items())))))


def tile_template_instances(family,template):
    axes = unique_by(family.axes,'axis_id')
    return _tile_template_instances(axes,family,template)


def _tile_template_instances(axes,family,template):
    total=prod(axes[name].count for name in template.iteration_axes)
    for bound in family.causal_prefixes:
        if bound.axis_id in template.iteration_axes:
            if bound.index_axis not in template.iteration_axes:
                raise ValueError('causal tile iteration requires its actual query coordinate')
            keys,queries=axes[bound.axis_id].extent,axes[bound.index_axis].extent
            total=total//(keys*queries)*_causal_prefix_sum(bound,axes)
    return total


def tile_template_requested_bytes(program,family,template):
    views,axes = unique_by(program.views,'view_id'),unique_by(family.axes,'axis_id')
    return _tile_template_requested_bytes(views,axes,family,template)


def _tile_template_requested_bytes(views,axes,family,template):
    view = views[template.view_id]
    if view.element_bytes is None:raise ValueError('logical storage encoding is unbound; requested bytes are unavailable')
    if any(isinstance(value,SymbolicExtent) for value in view.shape):
        raise ValueError('symbolic logical cardinality is unbound; requested bytes are unavailable')
    # Each named view dimension partitions its extent exactly, including its
    # tail. Loop axes absent from the view are actual repeated tile requests.
    total = prod(view.shape)*view.element_bytes
    for name in template.iteration_axes:
        if name not in template.view_axes:
            total *= axes[name].count
    for bound in family.causal_prefixes:
        if bound.axis_id in template.iteration_axes:
            keys,queries=axes[bound.axis_id].extent,axes[bound.index_axis].extent
            total=total//(keys*queries)*_causal_prefix_sum(bound,axes)
    return total


def _tile_template_requested_elements(views,axes,family,template):
    view = views[template.view_id]
    # Each named view dimension partitions its extent exactly, including its
    # tail. Loop axes absent from the view are actual repeated tile requests.
    total = prod(view.shape)
    for name in template.iteration_axes:
        if name not in template.view_axes:
            total *= axes[name].count
    for bound in family.causal_prefixes:
        if bound.axis_id in template.iteration_axes:
            keys,queries=axes[bound.axis_id].extent,axes[bound.index_axis].extent
            total=total//(keys*queries)*_causal_prefix_sum(bound,axes)
    return total


def _causal_prefix_offset(bound,axes):
    return (axes[bound.axis_id].extent-axes[bound.index_axis].extent+1
        if bound.offset is None else bound.offset)


def _causal_prefix_length(bound,axes,index):
    return min(axes[bound.axis_id].extent,max(0,
        (_causal_prefix_offset(bound,axes)+bound.index_coefficient*index)//bound.divisor))


def _causal_prefix_sum(bound,axes):
    """Sum clipped affine floor prefixes in logarithmic time, without rows."""
    queries,keys = axes[bound.index_axis].extent,axes[bound.axis_id].extent
    coefficient,divisor,offset = bound.index_coefficient,bound.divisor,_causal_prefix_offset(bound,axes)
    first = min(queries,max(0,-(-(divisor-offset)//coefficient)))
    saturated = min(queries,max(first,-(-(keys*divisor-offset)//coefficient)))
    count = saturated-first
    constant = offset+coefficient*first
    result = (queries-saturated)*keys
    if count == 0:return result
    # Euclidean floor-sum of (coefficient*i+constant)//divisor, i=0..count-1.
    while True:
        result += count*(count-1)//2*(coefficient//divisor)+count*(constant//divisor)
        coefficient %= divisor;constant %= divisor
        maximum = coefficient*count+constant
        if maximum < divisor:return result
        count,constant = divmod(maximum,divisor)
        divisor,coefficient = coefficient,divisor


def _tile_coordinate_allowed(family,axes,coordinates):
    return all(bound.axis_id not in coordinates or coordinates[bound.axis_id] <
        _causal_prefix_length(bound,axes,coordinates[bound.index_axis])
        for bound in family.causal_prefixes)


def _tile_coordinates(family,template,axes,limits):
    """Bounded enumeration after exact work count, skipping empty prefixes."""
    def visit(index,coordinates):
        if index == len(template.iteration_axes):
            yield coordinates
            return
        name=template.iteration_axes[index];start=0;stop=limits[name] if name in limits else axes[name].count
        for bound in family.causal_prefixes:
            if bound.axis_id not in template.iteration_axes:continue
            if name == bound.index_axis:start=max(start,0,-(-(bound.divisor-_causal_prefix_offset(bound,axes))//bound.index_coefficient))
            if name == bound.axis_id:stop=min(stop,_causal_prefix_length(bound,axes,coordinates[bound.index_axis]))
        for position in range(start,stop):yield from visit(index+1,{**coordinates,name:position})
    yield from visit(0,{})


def _tile_dense_initializers_cover(view,template,views,accesses):
    """Prove a typed tile from dense subregion stores at the same loop coordinates.

    Only loop-invariant contiguous byte intervals are joined. Iterated strides,
    predicates and loop ownership must agree; this cannot join other rows or
    infer an indirect selection. Complexity depends on dependency count.
    """
    if (not isinstance(view,TensorView) or template.indirect_target is not None
            or template.value_index is not None or template.atomic_add is not None):return False
    def interval(selected,axes):
        width=selected.element_bytes
        dimensions=sorted((stride,size) for size,stride,axis in zip(selected.shape,selected.strides_bytes,axes)
            if axis is None and size>1)
        for stride,size in dimensions:
            if stride!=width:return None
            width*=size
        return selected.offset_bytes,selected.offset_bytes+width
    # Covering the bounding interval is sufficient even for a strided read;
    # producer intervals themselves must be dense, so no holes are invented.
    target=(view.offset_bytes,view.offset_bytes+view.element_bytes+sum((size-1)*stride
        for size,stride,axis in zip(view.shape,view.strides_bytes,template.view_axes) if axis is None))
    intervals=[]
    for dependency in template.dependencies:
        writer=accesses[dependency.access_id];selected=views[writer.view_id]
        if (dependency.kind is not EventKind.WRITE_VISIBLE or dependency.reduction_axes
                or dependency.coordinate_bindings or dependency.carry_axis is not None
                or writer.operation is not Operation.WRITE or writer.unmapped!=template.unmapped
                or writer.indirect_target is not None or writer.value_index is not None or writer.atomic_add is not None
                or writer.predicate!=template.predicate or not isinstance(selected,TensorView)
                or (selected.allocation_id,selected.dtype,selected.element_bytes)!=(view.allocation_id,view.dtype,view.element_bytes)
                or len(selected.shape)!=len(view.shape) or writer.view_axes!=template.view_axes
                or not set(writer.iteration_axes)<=set(template.iteration_axes)
                or writer.view_axis_divisors!=template.view_axis_divisors or writer.view_axis_padding!=template.view_axis_padding):continue
        if any(axis is not None and (axis not in writer.iteration_axes or size!=other_size or stride!=other_stride)
                for axis,size,other_size,stride,other_stride in zip(template.view_axes,view.shape,selected.shape,
                    view.strides_bytes,selected.strides_bytes)):continue
        # A producer-only unused loop would multiply stores without identifying
        # the consumer's own initialization, so all producer loops must index a view.
        if not set(writer.iteration_axes)<=set(axis for axis in writer.view_axes if axis is not None):continue
        region=interval(selected,writer.view_axes)
        if region is not None:intervals.append(region)
    cursor=target[0]
    for start,end in sorted(intervals):
        if start>cursor:break
        if end>cursor:cursor=end
    return cursor>=target[1]


def _tile_typed_parent_fields_are_written(view,template,views,accesses,family):
    """A complete row-field partition initializes an unresolved selection.

    Every row of one dense typed parent must be written unconditionally.
    Selected values/addresses stay unresolved; no unrelated allocation or
    partial row can borrow this visibility proof.
    """
    if not isinstance(view,LogicalSelectionView) or not template.unmapped:return False
    parent=view
    while isinstance(parent,LogicalSelectionView) and parent.parent_view_id is not None:
        parent=views[parent.parent_view_id]
    if (not isinstance(parent,TensorView) or len(parent.shape)!=2
            or (parent.dtype,parent.element_bytes)!=(view.dtype,view.element_bytes)
            or parent.strides_bytes!=(parent.shape[1]*parent.element_bytes,parent.element_bytes)):return False
    axes=unique_by(family.axes,'axis_id');intervals=[]
    for dependency in template.dependencies:
        writer=accesses[dependency.access_id];part=views[writer.view_id]
        if (dependency.kind is not EventKind.WRITE_VISIBLE or dependency.coordinate_bindings
                or dependency.carry_axis is not None or dependency.partition_axis is not None
                or dependency.allow_skipped_read or writer.operation is not Operation.WRITE or writer.unmapped
                or writer.predicate is not None or writer.indirect_target is not None
                or writer.value_index is not None or writer.atomic_add is not None
                or len(writer.iteration_axes)!=1 or writer.view_axes!=(writer.iteration_axes[0],None)
                or dependency.reduction_axes!=writer.iteration_axes or writer.view_axis_divisors or writer.view_axis_padding
                or not isinstance(part,TensorView) or len(part.shape)!=2
                or (part.allocation_id,part.dtype,part.element_bytes,part.strides_bytes)
                    !=(parent.allocation_id,parent.dtype,parent.element_bytes,parent.strides_bytes)
                or part.shape[0]!=parent.shape[0]):continue
        axis,=writer.iteration_axes
        if (axes[axis].extent!=parent.shape[0] or axes[axis].value_limit_access_id is not None
                or any(prefix.axis_id==axis for prefix in family.causal_prefixes)):continue
        first=part.offset_bytes-parent.offset_bytes;last=first+part.shape[1]*part.element_bytes
        if not 0<=first<last<=parent.shape[1]*parent.element_bytes:continue
        intervals.append((first,last))
    cursor=0
    for first,last in sorted(intervals):
        if first>cursor:return False
        cursor=max(cursor,last)
    return cursor==parent.shape[1]*parent.element_bytes


def _tile_row_initializers_cover(view,template,views,accesses,family):
    """Complete unconditional scalar stores initialize their own typed row.

    A reader may select a column prefix of that same row. No other row,
    conditional store, dynamic loop, or unknown address is joined here.
    """
    if (not isinstance(view,TensorView) or len(view.shape)!=2 or template.unmapped
            or len(template.iteration_axes)!=2 or template.view_axes!=template.iteration_axes):return False
    axes=unique_by(family.axes,'axis_id');row,column=template.iteration_axes
    for dep in template.dependencies:
        prior=accesses[dep.access_id];part=views[prior.view_id]
        if (dep.kind is not EventKind.WRITE_VISIBLE or dep.coordinate_bindings or dep.carry_axis is not None
                or dep.partition_axis is not None or len(dep.reduction_axes)!=1
                or prior.operation is not Operation.WRITE or prior.unmapped or prior.predicate is not None
                or prior.value_index is not None or prior.atomic_add is not None or prior.indirect_target is not None
                or len(prior.iteration_axes)!=2 or prior.iteration_axes[0]!=row
                or prior.view_axes!=prior.iteration_axes or dep.reduction_axes!=(prior.iteration_axes[1],)
                or not isinstance(part,TensorView) or len(part.shape)!=2
                or (part.allocation_id,part.dtype,part.element_bytes,part.offset_bytes,part.strides_bytes)
                    !=(view.allocation_id,view.dtype,view.element_bytes,view.offset_bytes,view.strides_bytes)
                or part.shape[0]!=view.shape[0] or part.shape[1]<view.shape[1]
                or part.strides_bytes!=(part.shape[1]*part.element_bytes,part.element_bytes)
                or any(axes[name].value_limit_access_id is not None for name in prior.iteration_axes)
                or any(bound.axis_id in prior.iteration_axes for bound in family.causal_prefixes)
                or prior.view_axis_divisors or prior.view_axis_padding):continue
        return True
    return False


def _tile_selected_parent_is_written(view,template,views,accesses):
    """Prove a dynamically selected row read has a complete actual producer.

    Only a full disjoint partition of a two-dimensional logical row parent is
    admitted. This does not resolve the selected row values or their addresses.
    """
    if _tile_dense_initializers_cover(view,template,views,accesses):return True
    # A contained column selection may consume its own same-coordinate tile
    # store through their identical logical parent (e.g. V512 inside K576).
    # This does not join other rows, reinterpret types, or broadcast readiness.
    if isinstance(view,LogicalSelectionView) and view.parent_view_id is not None and template.unmapped:
        for dependency in template.dependencies:
            writer=accesses[dependency.access_id];selected=views[writer.view_id]
            if (dependency.kind is EventKind.WRITE_VISIBLE and not dependency.reduction_axes
                    and not dependency.coordinate_bindings and dependency.carry_axis is None
                    and writer.operation is Operation.WRITE and writer.indirect_target is None and template.indirect_target is None
                    and writer.value_index is None and template.value_index is None and writer.atomic_add is None
                    and writer.predicate==template.predicate
                    and writer.iteration_axes==template.iteration_axes and writer.view_axes==template.view_axes
                    and (writer.view_axis_divisors or (1,)*len(view.shape))==(template.view_axis_divisors or (1,)*len(view.shape))
                    and (writer.view_axis_padding or (False,)*len(view.shape))==(template.view_axis_padding or (False,)*len(view.shape))
                    and writer.unmapped and isinstance(selected,LogicalSelectionView)
                    and selected.parent_view_id==view.parent_view_id and view.coordinate_map and selected.coordinate_map==view.coordinate_map
                    and (selected.allocation_id,selected.dtype,selected.element_bytes)==(view.allocation_id,view.dtype,view.element_bytes)
                    and len(selected.shape)==len(view.shape) and all(read<=written for read,written in zip(view.shape,selected.shape))):return True
    # A full typed initialization also authorizes reads through a contained
    # subview of the same object. The write must run once, before the reads;
    # this does not infer an unresolved selection or an iterated writer union.
    initialized_parent=view
    while (template.value_index is not None and isinstance(initialized_parent,LogicalSelectionView)
            and initialized_parent.parent_view_id is not None):
        initialized_parent=views[initialized_parent.parent_view_id]
    if template.value_index is not None and isinstance(initialized_parent,TensorView):
        from dataclasses import replace
        proof=replace(template,view_id=initialized_parent.view_id,value_index=None,unmapped=False,
            view_axes=(None,)*len(initialized_parent.shape))
        if _tile_dense_initializers_cover(initialized_parent,proof,views,accesses):return True
    if isinstance(initialized_parent,TensorView) and (not template.unmapped or isinstance(view,LogicalSelectionView)):
        for dependency in template.dependencies:
            writer=accesses[dependency.access_id];written=views[writer.view_id]
            if (dependency.kind is EventKind.WRITE_VISIBLE and not dependency.reduction_axes
                    and not dependency.coordinate_bindings and dependency.carry_axis is None
                    and writer.operation is Operation.WRITE and not writer.iteration_axes
                    and not writer.unmapped and writer.indirect_target is None
                    and writer.view_axes==(None,)*len(written.shape) and isinstance(written,TensorView)
                    and (written.allocation_id,written.dtype,written.element_bytes)==(initialized_parent.allocation_id,initialized_parent.dtype,initialized_parent.element_bytes)
                    and covers(written.region(),initialized_parent.region())):return True
    if (not template.unmapped or not isinstance(view,LogicalSelectionView)
            or view.parent_view_id is None):return False
    parent = views[view.parent_view_id]
    if len(parent.shape) != 2 or (parent.dtype,parent.element_bytes) != (view.dtype,view.element_bytes):return False
    intervals = []
    for dependency in template.dependencies:
        if dependency.kind is not EventKind.WRITE_VISIBLE:continue
        writer = accesses[dependency.access_id];selected = views[writer.view_id]
        if (not isinstance(selected,LogicalSelectionView) or selected.parent_view_id != parent.view_id
                or selected.allocation_id != view.allocation_id or selected.dtype != view.dtype
                or selected.element_bytes != view.element_bytes or len(selected.shape) != 2
                or selected.shape[1] != parent.shape[1] or len(selected.coordinate_map) != 2
                or writer.operation is not Operation.WRITE or writer.indirect_target is not None
                or len(writer.iteration_axes) != 1 or writer.view_axes != (writer.iteration_axes[0],None)
                or dependency.reduction_axes != writer.iteration_axes or dependency.coordinate_bindings):continue
        row,column = selected.coordinate_map
        if (row.axis != 0 or row.terms != (LogicalCoordinateTerm(0,1),) or row.divisor != 1 or row.modulus is not None
                or column != LogicalCoordinateBinding(1,(LogicalCoordinateTerm(1,1),))):continue
        intervals.append((row.offset,row.offset+selected.shape[0]))
    covered = 0
    for start,end in sorted(intervals):
        if start != covered:return False
        covered = end
    return covered == parent.shape[0]


def _validate_tile_value_dataflow(program,family,views,axes,accesses):
    """Keep indexed keys and paired RMWs explicit until a value-aware consumer."""
    paired_reads=set()
    for template in family.accesses:
        if template.value_index is None and template.atomic_add is None:continue
        if template.view_id not in views:raise ValueError('value-indexed tile view is absent')
        view=views[template.view_id]
        if template.atomic_add is not None and template.value_index is None:
            # A known single counter still needs an indivisible RMW. Its
            # address is known, while update order and previous values remain
            # unresolved. Do not turn it into independent ordinary stores.
            atomic=template.atomic_add
            if (program.coverage.semantics is not Semantics.LOGICAL or program.coverage.executable
                    or not isinstance(view,TensorView) or prod(view.shape)!=1 or view.dtype not in ('int32','int64')
                    or template.operation is not Operation.WRITE or template.predicate is not None
                    or template.view_axes!=(None,)*len(view.shape) or not template.iteration_axes
                    or any(axes[name].block!=1 for name in template.iteration_axes)
                    or template.constant_value is not None or template.indirect_target is not None
                    or template.view_axis_divisors or template.view_axis_padding or atomic.read_access_id not in accesses):
                raise ValueError('known scalar atomic requires a typed logical integer counter and actual RMW read')
            read=accesses[atomic.read_access_id]
            if (read.operation is not Operation.READ or read.atomic_add is not None or read.value_index is not None
                    or read.view_id!=template.view_id or read.iteration_axes!=template.iteration_axes
                    or read.view_axes!=template.view_axes or read.unmapped!=template.unmapped or read.predicate is not None
                    or read.indirect_target is not None or read.view_axis_divisors or read.view_axis_padding
                    or TileDependency(read.access_id,EventKind.READ_READY,()) not in template.dependencies
                    or read.access_id in paired_reads):
                raise ValueError('known scalar atomic lost its unique matching RMW read')
            if template.unmapped:
                events=unique_by(program.events,'event_id')
                if not any(ref.node_id in events and events[ref.node_id].view_id==view.view_id
                        and events[ref.node_id].tile_sources for ref in read.input_ready):
                    raise ValueError('unaddressed scalar atomic requires its actual prepared field readiness')
            paired_reads.add(read.access_id)
            continue
        scalar=(len(view.shape)==len(template.iteration_axes) and template.view_axes==template.iteration_axes)
        # A scalar integer key may select a whole payload row. Its value
        # relation still does not supply an address, table capacity or a
        # bound on the selected key. Atomics and indexed writes stay scalar.
        payload_read=(template.operation is Operation.READ and template.atomic_add is None
            and template.value_index is not None and template.value_index.value_kind=='loaded_scalar'
            and tuple(name for name in template.view_axes if name is not None)==template.iteration_axes)
        if (program.coverage.semantics is not Semantics.LOGICAL or program.coverage.executable or not template.unmapped
                or not isinstance(view,LogicalSelectionView) or not (scalar or payload_read)
                or not template.iteration_axes
                or any(name not in axes or axes[name].block!=1 for name in template.iteration_axes) or template.view_axis_divisors or template.view_axis_padding
                or template.indirect_target is not None or template.constant_value is not None):
            raise ValueError('value-indexed tiles require unresolved scalar or read-payload logical selections')
        index=template.value_index
        if index is not None:
            if index.source_access_id not in accesses:raise ValueError('tile index value source is absent')
            source=accesses[index.source_access_id]
            if source.view_id not in views:raise ValueError('tile index source view is absent')
            source_view=views[source.view_id]
            if index.value_kind=='argmax_index':
                parent=views[view.parent_view_id] if view.parent_view_id in views else None
                mapping=(LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),)),)
                if (template.operation is not Operation.READ or template.atomic_add is not None
                        or not isinstance(view,LogicalPartialSelectionView) or view.unbound_parent_axes!=(1,)
                        or view.coordinate_map!=mapping or not isinstance(parent,TensorView)
                        or not isinstance(source_view,TensorView) or len(source_view.shape)!=2
                        or source_view.dtype not in ('float32','fp32') or source.operation is not Operation.READ
                        or source.value_index is not None or source.atomic_add is not None or source.predicate is not None
                        or source.unmapped or source.indirect_target is not None or source.view_axis_divisors or source.view_axis_padding
                        or len(template.iteration_axes)!=1 or len(source.iteration_axes)!=2
                        or source.iteration_axes[:1]!=template.iteration_axes or source.view_axes!=source.iteration_axes
                        or any(axes[name].block!=1 for name in source.iteration_axes)
                        or axes[source.iteration_axes[1]].extent!=source_view.shape[1]
                        or parent.shape!=source_view.shape or parent.dtype not in ('int32','int64')
                        or view.shape!=(source_view.shape[0],)
                        or TileDependency(source.access_id,EventKind.READ_READY,source.iteration_axes[1:]) not in template.dependencies):
                    raise ValueError('argmax index must retain its own row and complete floating-point value reduction')
                continue
            if (source.iteration_axes!=template.iteration_axes or source.view_axes!=source.iteration_axes
                    or len(source_view.shape)!=len(source.iteration_axes) or source_view.dtype not in ('int32','int64')):
                raise ValueError('tile index must use its matching actual scalar integer value')
            if index.value_kind=='loaded_scalar':
                if source.operation is not Operation.READ or source.atomic_add is not None:
                    raise ValueError('loaded tile index requires an actual integer read')
                kind=EventKind.READ_READY
            elif index.value_kind=='atomic_previous':
                if source.operation is not Operation.WRITE or source.atomic_add is None:
                    raise ValueError('atomic previous index requires its actual indivisible update')
                kind=EventKind.WRITE_VISIBLE
            else:raise ValueError('unknown tile index value kind')
            if TileDependency(source.access_id,kind,()) not in template.dependencies:
                raise ValueError('tile selection requires the actual index value readiness')
        atomic=template.atomic_add
        if atomic is not None:
            if (template.operation is not Operation.WRITE or index is None or index.value_kind!='loaded_scalar'
                    or view.dtype not in ('int32','int64') or atomic.read_access_id not in accesses):
                raise ValueError('atomic add requires its integer RMW read and selected key')
            read=accesses[atomic.read_access_id]
            if (read.operation is not Operation.READ or read.atomic_add is not None or read.value_index!=index
                    or read.view_id!=template.view_id or read.iteration_axes!=template.iteration_axes or read.view_axes!=template.view_axes
                    or TileDependency(read.access_id,EventKind.READ_READY,()) not in template.dependencies
                    or read.access_id in paired_reads):raise ValueError('atomic add lost its unique matching RMW read')
            paired_reads.add(read.access_id)


def _tile_dependency_partition(family,dependency,prior,target):
    """Resolve a relation from its actual endpoints, never tuple order."""
    candidates=[]
    for partition in family.partitions:
        if partition.axis_id!=dependency.partition_axis:continue
        member,class_axis=partition.axis_id,partition.class_axis_id
        if member in prior.iteration_axes and member in dependency.reduction_axes:
            if (class_axis is None or class_axis in target.iteration_axes
                    or any(binding.axis_id==class_axis for binding in dependency.coordinate_bindings)):
                candidates.append((partition,False))
        elif (class_axis is not None and member in target.iteration_axes
                and member not in prior.iteration_axes and class_axis in prior.iteration_axes
                and class_axis not in target.iteration_axes):
            candidates.append((partition,True))
    if len(candidates)!=1:
        raise ValueError('partition dependency requires one unambiguous actual member/class relation')
    return candidates[0]


def _tile_partition_initializer(view,template,views,accesses,family):
    """Prove a scalar selection from its actual class writer, keeping keys unknown."""
    if _tile_vector_partition_covers(view,template,views,accesses,family):return True
    for dependency in template.dependencies:
        writer=accesses[dependency.access_id];parent=views[writer.view_id]
        if (dependency.kind is not EventKind.WRITE_VISIBLE or writer.operation is not Operation.WRITE
                or writer.unmapped or writer.indirect_target is not None or writer.predicate is not None
                or template.predicate is not None or writer.value_index is not None or writer.atomic_add is not None
                or not isinstance(parent,TensorView) or len(parent.shape)!=1
                or (parent.allocation_id,parent.dtype,parent.element_bytes)!=(view.allocation_id,view.dtype,view.element_bytes)
                or len(writer.iteration_axes)!=1 or writer.view_axes!=writer.iteration_axes):continue
        if dependency.partition_axis is not None:
            partition,inverse=_tile_dependency_partition(family,dependency,writer,template)
            if (inverse and isinstance(view,LogicalSelectionView) and template.unmapped
                    and view.parent_view_id==parent.view_id and not view.coordinate_map
                    and template.iteration_axes==(partition.axis_id,) and template.view_axes==template.iteration_axes
                    and view.shape==(next(axis.extent for axis in family.axes if axis.axis_id==partition.axis_id),)
                    and writer.iteration_axes==(partition.class_axis_id,)):
                return True
        elif (isinstance(view,TensorView) and not template.unmapped and view.shape==(1,)
                and template.view_axes==(None,) and not dependency.reduction_axes
                and dependency.carry_axis is None and len(dependency.coordinate_bindings)==1):
            binding,=dependency.coordinate_bindings
            if (binding.axis_id==writer.iteration_axes[0] and not binding.terms
                    and binding.divisor==1 and binding.modulus is None
                    and binding.offset<parent.shape[0]
                    and view.offset_bytes==parent.offset_bytes+binding.offset*parent.strides_bytes[0]):
                return True
    return False


def _tile_vector_partition_covers(view,template,views,accesses,family):
    """Join complete static vector stores before an unresolved selected read.

    This proves availability only. It never supplies a sort/routing map or
    turns a possible dynamic/predicated store into initialization.
    """
    if (not isinstance(view,LogicalSelectionView) or view.parent_view_id is None
            or not template.unmapped or template.predicate is not None):return False
    parent=views[view.parent_view_id]
    if not isinstance(parent,TensorView) or len(parent.shape)!=1 or parent.strides_bytes!=(parent.element_bytes,):return False
    axes=unique_by(family.axes,'axis_id');intervals=[]
    for dependency in template.dependencies:
        writer=accesses[dependency.access_id];part=views[writer.view_id]
        if (dependency.kind is not EventKind.WRITE_VISIBLE or dependency.coordinate_bindings
                or dependency.carry_axis is not None or dependency.partition_axis is not None
                or writer.operation is not Operation.WRITE or writer.unmapped or writer.predicate is not None
                or writer.value_index is not None or writer.indirect_target is not None or writer.atomic_add is not None
                or not isinstance(part,TensorView) or len(part.shape)!=1
                or (part.allocation_id,part.dtype,part.element_bytes)!=(parent.allocation_id,parent.dtype,parent.element_bytes)
                or part.strides_bytes!=(parent.element_bytes,) or len(writer.iteration_axes)!=1
                or writer.view_axes!=writer.iteration_axes or dependency.reduction_axes!=writer.iteration_axes
                or writer.iteration_axes[0] not in axes):continue
        axis=axes[writer.iteration_axes[0]]
        if (axis.value_limit_access_id is not None or axis.extent!=part.shape[0]
                or writer.view_axis_padding and any(writer.view_axis_padding)
                or writer.view_axis_divisors and writer.view_axis_divisors!=(1,)
                or any(axis.axis_id==bound.axis_id for bound in family.causal_prefixes)):continue
        intervals.append((part.offset_bytes,part.offset_bytes+part.shape[0]*part.element_bytes))
    cursor=parent.offset_bytes;end=cursor+parent.shape[0]*parent.element_bytes
    for start,stop in sorted(intervals):
        if start>cursor:break
        cursor=max(cursor,stop)
    return cursor>=end


def _tile_selected_predecessor_carry(prior,target,initial,dependency,views):
    """Order a single actor's changing-key RMWs without equating their keys.

    The separate full-parent initialization proves first access availability;
    the predecessor edge only orders this actor's iterations. Plain expansion
    still rejects unresolved values, as for every other value-indexed access.
    """
    if target.value_index is None:return False
    view=views[target.view_id]
    return (isinstance(view,LogicalSelectionView) and view.parent_view_id is not None
        and prior.operation is Operation.WRITE and target.operation is Operation.READ
        and initial.operation is Operation.WRITE and dependency.kind is EventKind.WRITE_VISIBLE
        and target.value_index is not None and target.value_index.value_kind=='loaded_scalar'
        and prior.value_index==target.value_index and prior.atomic_add is None and target.atomic_add is None
        and prior.predicate is None and target.predicate is None
        and prior.view_id==target.view_id and prior.unmapped and target.unmapped
        and prior.iteration_axes==target.iteration_axes==(dependency.carry_axis,)
        and prior.view_axes==target.view_axes==target.iteration_axes
        and initial.view_id==view.parent_view_id and not initial.iteration_axes and not initial.unmapped
        and initial.constant_value is ConstantValue.ZERO and initial.view_axes==(None,)
        and isinstance(views[initial.view_id],TensorView)
        and views[initial.view_id].dtype==view.dtype and views[initial.view_id].element_bytes==view.element_bytes
        and TileDependency(initial.access_id,EventKind.WRITE_VISIBLE,()) in target.dependencies
        and not dependency.reduction_axes and not dependency.coordinate_bindings)


def _validate_tile_value_limits(program,family,axes,accesses,views):
    """Keep native scalar loop limits symbolic instead of inventing a shape."""
    limited={axis.axis_id for axis in family.axes if axis.value_limit_access_id is not None}
    if not limited:return
    consumers_by_axis={name:[] for name in limited}
    escaping_writes={name:[] for name in limited}
    # The old loop scanned every access and dependency again for every
    # independent scalar bound. Index the same checks once; preserve each
    # axis's access/dependency order and the original fail-closed predicates.
    for consumer in family.accesses:
        for name in consumer.iteration_axes:
            if name in limited:consumers_by_axis[name].append(consumer)
        if consumer.operation is not Operation.READ:continue
        for dependency in consumer.dependencies:
            if dependency.access_id not in accesses:continue
            producer=accesses[dependency.access_id]
            if producer.operation is not Operation.WRITE:continue
            for name in producer.iteration_axes:
                if (name in limited and (name not in consumer.iteration_axes
                        or name in dependency.reduction_axes
                        or any(binding.axis_id==name for binding in dependency.coordinate_bindings))):
                    escaping_writes[name].append(consumer)
    for axis in family.axes:
        if axis.value_limit_access_id is None:continue
        if program.coverage.executable or axis.block!=1 or axis.value_limit_access_id not in accesses:
            raise ValueError('value-limited tile requires its actual scalar READ and nonexecutable unit axis')
        source=accesses[axis.value_limit_access_id]
        if source.view_id not in views:raise ValueError('tile value-limit scalar view is missing')
        view=views[source.view_id]
        if (source.operation is not Operation.READ or source.iteration_axes or source.view_axes!=(None,)
                or view.shape!=(1,) or view.dtype not in ('int32','int64') or source.predicate is not None
                or source.value_index is not None or source.atomic_add is not None or source.indirect_target is not None
                or max(axis.extent,axis.value_limit_start,axis.scalar_limit_upper_bound)>2**(31 if view.dtype=='int32' else 63)-1):
            raise ValueError('tile value limit requires one actual signed scalar READ and its integer range')
        if (any(axis.axis_id in (bound.axis_id,bound.index_axis) for bound in family.causal_prefixes)
                or any(axis.axis_id in (partition.axis_id,partition.class_axis_id,*partition.context_axes) for partition in family.partitions)):
            raise ValueError('value-limited axes cannot silently change causal or partition membership')
        consumers=consumers_by_axis[axis.axis_id]
        if not consumers:raise ValueError('tile value-limit axis has no actual consumer')
        for consumer in consumers:
            if TileDependency(source.access_id,EventKind.READ_READY,()) not in consumer.dependencies:
                raise ValueError('value-limited tile requires the actual loop-bound scalar READ_READY')
            if any(any(binding.axis_id==axis.axis_id for binding in dependency.coordinate_bindings)
                    for dependency in consumer.dependencies):
                raise ValueError('value-limited loop cannot hide an unresolved predecessor or coordinate selector')
            for dependency in consumer.dependencies:
                if dependency.carry_axis!=axis.axis_id:continue
                if dependency.initial_access_id not in accesses:
                    raise ValueError('value-limited carry requires an unconditional actual state seed')
                seed=accesses[dependency.initial_access_id]
                if (consumer.value_index is not None or consumer.unmapped or consumer.predicate is not None
                        or seed.predicate is not None or seed.unmapped or seed.value_index is not None
                        or seed.indirect_target is not None or seed.atomic_add is not None
                        or not isinstance(views[consumer.view_id],TensorView)
                        or any(axes[name].value_limit_access_id is not None for name in seed.iteration_axes)):
                    raise ValueError('value-limited carry requires an unconditional actual state seed')
                # The ordinary carry validator below additionally proves that
                # the READ, prior WRITE and initial WRITE use the same state.
        for consumer in escaping_writes[axis.axis_id]:
            if not _tile_dense_initializers_cover(views[consumer.view_id],consumer,views,accesses):
                raise ValueError('value-limited writer cannot prove unconditional full-region initialization')


def _tile_value_limit_bindings(program,values,*,required):
    if not isinstance(program,MemoryAccessProgram):raise TypeError('tile scalar bindings require a typed memory program')
    expected={(family.tile_program_id,axis.value_limit_access_id) for family in program.tile_programs
        for axis in family.axes if axis.value_limit_access_id is not None}
    if values is None:
        if required and expected:raise ValueError('tile scalar loop limits are unbound; supply explicit actual READ values')
        return {}
    if type(values) is not dict or set(values)!=expected:
        raise ValueError('tile scalar value bindings must cover exactly their actual source READ identities')
    result={}
    for family in program.tile_programs:
        bound={}
        for axis in family.axes:
            if axis.value_limit_access_id is None:continue
            value=values[family.tile_program_id,axis.value_limit_access_id]
            if type(value) is not int or not -int(axis.value_limit_inclusive)<=value<=axis.scalar_limit_upper_bound:
                raise ValueError('tile scalar loop bound is outside its declared integer envelope')
            bound[axis.axis_id]=min(axis.extent,max(0,value+int(axis.value_limit_inclusive)-axis.value_limit_start))
        result[family.tile_program_id]=bound
    return result


def _tile_limited_instances(family,template,limits):
    total=tile_template_instances(family,template)
    if not limits:return total
    axes=unique_by(family.axes,'axis_id')
    names=tuple(name for name in template.iteration_axes if name in limits)
    return total//prod(axes[name].count for name in names)*prod(limits[name] for name in names)


def _tile_limited_requested_bytes(views,axes,family,template,limits):
    total=_tile_template_requested_bytes(views,axes,family,template)
    names=tuple(name for name in template.iteration_axes if name in limits)
    return total//prod(axes[name].count for name in names)*prod(limits[name] for name in names)


@runtime_lru_cache(maxsize=256)
def _validate_tile_operand_geometry(axis_values,shape,iteration_axes,view_axes,view_divisors,view_padding):
    """Cache only exact immutable loop geometry; bindings remain per call."""
    axes={name:(extent,block) for name,extent,block in axis_values}
    if (len(view_axes)!=len(shape) or len(set(iteration_axes))!=len(iteration_axes)
            or not set(iteration_axes)<=set(axes)
            or len([name for name in view_axes if name is not None])
                !=len({name for name in view_axes if name is not None})):
        raise ValueError('invalid tile loop/view-axis mapping')
    divisors=view_divisors or (1,)*len(shape)
    padding=view_padding or (False,)*len(shape)
    if len(divisors)!=len(shape) or min(divisors)<=0:
        raise ValueError('tile view divisors must define every positive coordinate grouping')
    if len(padding)!=len(shape):
        raise ValueError('tile view padding must define every dimension')
    for dimension,name,divisor,padded in zip(shape,view_axes,divisors,padding,strict=True):
        if name is None:
            if divisor!=1 or padded:raise ValueError('whole tile dimensions cannot have grouping or padding')
        else:
            if name not in iteration_axes or axes[name][1]%divisor:
                raise ValueError('tile view axis must partition its exact tensor dimension with group-aligned blocks')
            extent,block=axes[name]
            expected=(((extent+block-1)//block)*(block//divisor) if padded else (extent+divisor-1)//divisor)
            if dimension!=expected:
                raise ValueError('tile view axis must retain its exact masked or explicitly padded extent')
    return divisors


@runtime_lru_cache(maxsize=32)
def _validate_static_tile_dependency_graph(steps):
    """Prove ordinary completion edges using their complete immutable graph.

    Carry, coordinate and partition edges retain the general validator. Actual
    views, storage, caller scopes and input availability are checked per source.
    """
    accesses={identity:(operation,iteration_axes,dependencies)
        for identity,operation,iteration_axes,dependencies in steps}
    visiting,done=set(),set()
    def visit(identity):
        if identity in visiting:raise ValueError('tile dependency cycle')
        if identity in done:return
        visiting.add(identity)
        _,target_axes,dependencies=accesses[identity]
        for dep in dependencies:
            if dep.access_id not in accesses:raise ValueError('unknown tile dependency')
            operation,prior_axes,_=accesses[dep.access_id]
            expected=EventKind.READ_READY if operation is Operation.READ else EventKind.WRITE_VISIBLE
            if dep.kind is not expected:raise ValueError('tile dependency requires actual completion kind')
            if (len(set(dep.reduction_axes))!=len(dep.reduction_axes)
                    or not set(dep.reduction_axes)<=set(prior_axes)
                    or set(prior_axes)-set(target_axes)!=set(dep.reduction_axes)-set(target_axes)):
                raise ValueError('tile reduction/broadcast dependency must conserve its iteration axes')
            visit(dep.access_id)
        visiting.remove(identity);done.add(identity)
    for identity in accesses:visit(identity)


def validate_input_readiness(view, ref, *, unmapped, views, events, visibility_domain):
    """One declared tensor input, shared by tile export and analytical consumers."""
    if ref.allow_skipped_read:raise ValueError('input availability cannot be an optional read join')
    if ref.node_id not in events or events[ref.node_id].kind is not ref.kind:
        raise ValueError('tile input readiness is not an actual program event')
    event = events[ref.node_id]
    if event.allocation_id != view.allocation_id or event.visibility_domain != visibility_domain:
        raise ValueError('tile input readiness allocation/visibility mismatch')
    if not unmapped and (event.region is None or not covers(event.region,view.region())):
        raise ValueError('tile input readiness must cover its selected tensor')
    if unmapped:
        parent = view
        while isinstance(parent, LogicalSelectionView) and parent.parent_view_id is not None:
            parent = views[parent.parent_view_id]
        if event.view_id is not None:
            field=views[event.view_id]
            if (not isinstance(field,TensorView) or not isinstance(parent,TensorView)
                    or (field.allocation_id,field.dtype,field.element_bytes)!=(parent.allocation_id,parent.dtype,parent.element_bytes)
                    or not covers(field.region(),parent.region())):
                raise ValueError('tile input availability belongs to another typed field')
        if isinstance(event.region, Unmapped):
            expected = (parent.shape, parent.dtype, parent.element_bytes)
            if (event.region.logical_shape,event.region.dtype,event.region.element_bytes) != expected:
                raise ValueError('unresolved tile readiness must cover its actual typed selection parent')
            if isinstance(parent, LogicalSelectionView) and event.region != parent.region():
                raise ValueError('logical selection root requires its exact available selection')
        elif isinstance(view, LogicalSelectionView) and (event.region is None
                or isinstance(parent, LogicalSelectionView) or not covers(event.region, parent.region())):
            raise ValueError('logical selection requires complete parent readiness')


def validate_tile_programs(program):
    if not program.tile_programs:
        if any(event.tile_sources for event in program.events):
            raise ValueError('tile availability requires its actual compressed writer')
        return
    if program.coverage.granularity is not Granularity.TILE or program.coverage.semantics is not Semantics.LOGICAL:
        raise ValueError('compressed tile source requires logical TILE coverage')
    if program.accesses:
        raise ValueError('tile source cannot retain a second coarse access source')
    views,events = unique_by(program.views,'view_id'),unique_by(program.events,'event_id')
    allocations=unique_by(program.allocations,'allocation_id')
    unique_by(program.tile_programs,'tile_program_id')
    for family in program.tile_programs:
        axes,accesses = unique_by(family.axes,'axis_id'),unique_by(family.accesses,'access_id')
        axis_values=tuple((axis.axis_id,axis.extent,axis.block) for axis in family.axes)
        if any(dep.access_id not in accesses for access in family.accesses for dep in access.dependencies):
            raise ValueError('unknown tile dependency')
        _validate_tile_value_dataflow(program,family,views,axes,accesses)
        _validate_tile_value_limits(program,family,axes,accesses,views)
        static_graph=not family.partitions and all(
            dep.carry_axis is None and dep.initial_access_id is None
            and dep.partition_axis is None and not dep.coordinate_bindings
            for access in family.accesses for dep in access.dependencies)
        if not axes or not accesses or not family.evidence_refs:
            raise ValueError('tile loop requires actual axes/accesses and partition evidence')
        used_partitions=set()
        if family.partitions:
            keys=tuple((p.axis_id,p.class_axis_id,p.context_axes) for p in family.partitions)
            if len(set(keys))!=len(keys):raise ValueError('duplicate logical tile partition relation')
            for partition in family.partitions:
                axis_id=partition.axis_id
                if axis_id not in axes or axes[axis_id].block!=1:
                    raise ValueError('logical partition requires an actual unit tile axis')
                if (len(set(partition.context_axes))!=len(partition.context_axes)
                        or not set(partition.context_axes)<=set(axes)
                        or axis_id in partition.context_axes or partition.class_axis_id in partition.context_axes):
                    raise ValueError('logical partition context must name distinct actual other axes')
                if partition.class_axis_id is None:
                    if partition.context_axes:raise ValueError('member-equivalence partition has no class/context axis')
                elif (partition.class_axis_id not in axes or partition.class_axis_id==axis_id
                        or axes[partition.class_axis_id].block!=1):
                    raise ValueError('logical partition requires its actual independent class axis')
                expected_members=axes[axis_id].count*prod(axes[name].count for name in partition.context_axes)
                if partition.members is None:
                    if program.coverage.executable:raise ValueError('unbound tile partition cannot declare executable dependencies')
                elif len(partition.members)!=expected_members:
                    raise ValueError('logical partition must bind every axis member in every context')
                elif partition.class_axis_id is not None and any(value>=axes[partition.class_axis_id].count for value in partition.members):
                    raise ValueError('logical partition member exceeds its declared class range')
        bounded_axes=set();prefix_indices=set()
        for bound in family.causal_prefixes:
            if (bound.axis_id not in axes or bound.index_axis not in axes or bound.axis_id == bound.index_axis
                    or axes[bound.axis_id].block != 1 or axes[bound.index_axis].block != 1
                    or bound.axis_id in bounded_axes|prefix_indices or bound.index_axis in bounded_axes):
                raise ValueError('causal prefix requires distinct nonoverlapping unit query/key axes')
            if _causal_prefix_offset(bound,axes)+bound.index_coefficient*(axes[bound.index_axis].extent-1) > U64_MAX:
                raise ValueError('causal prefix coordinate exceeds uint64')
            bounded_axes.add(bound.axis_id);prefix_indices.add(bound.index_axis)
        visiting,done = set(),set()
        def visit(identity):
            if identity in visiting:raise ValueError('tile dependency cycle')
            if identity in done:return
            visiting.add(identity)
            target = accesses[identity]
            for dep in target.dependencies:
                if dep.access_id not in accesses:raise ValueError('unknown tile dependency')
                prior = accesses[dep.access_id]
                expected = EventKind.READ_READY if prior.operation is Operation.READ else EventKind.WRITE_VISIBLE
                if dep.kind is not expected:raise ValueError('tile dependency requires actual completion kind')
                class_binding=None;selected_class_axis=None
                if dep.partition_axis is not None:
                    partition,inverse=_tile_dependency_partition(family,dep,prior,target)
                    used_partitions.add((partition.axis_id,partition.class_axis_id,partition.context_axes))
                    if (dep.carry_axis is not None
                            or any(binding.axis_id==dep.partition_axis for binding in dep.coordinate_bindings)):
                        raise ValueError('partition reduction must retain its actual producer member axis')
                    if inverse:
                        selected_class_axis=partition.class_axis_id
                        if (dep.coordinate_bindings
                                or set(dep.reduction_axes)&{selected_class_axis,partition.axis_id,*partition.context_axes}
                                or not set(dep.reduction_axes)<=set(prior.iteration_axes)
                                or not (set(prior.iteration_axes)-{selected_class_axis}-set(dep.reduction_axes))<=set(partition.context_axes)
                                or set(target.iteration_axes)-{partition.axis_id}!=set(partition.context_axes)):
                            raise ValueError('partition selection must retain each actual producer/consumer context')
                    elif partition.class_axis_id is None:
                        if dep.partition_axis not in target.iteration_axes:
                            raise ValueError('member-equivalence partition requires the consumer member axis')
                    else:
                        class_axis=partition.class_axis_id
                        if class_axis not in target.iteration_axes:
                            if (len(dep.coordinate_bindings)!=1 or dep.coordinate_bindings[0].axis_id!=class_axis
                                    or dep.coordinate_bindings[0].divisor!=1
                                    or dep.coordinate_bindings[0].modulus is not None
                                    or not 0<=dep.coordinate_bindings[0].offset<axes[class_axis].count):
                                raise ValueError('partition class requires one bounded affine class selector')
                            class_binding=dep.coordinate_bindings[0]
                        producer_context=set(prior.iteration_axes)-{dep.partition_axis}
                        consumer_context=set(target.iteration_axes)-{class_axis}
                        if class_binding is not None:
                            # A distinct consumer loop may select a class,
                            # e.g. update i selects split i+1. It cannot erase
                            # any shared producer row/batch/expert context.
                            consumer_context-={term.axis_id for term in class_binding.terms}-producer_context
                        reduced_context=set(dep.reduction_axes)-{dep.partition_axis}
                        if (class_binding is None and class_axis not in target.iteration_axes or class_axis in prior.iteration_axes
                                or dep.partition_axis in target.iteration_axes or class_binding is None and dep.coordinate_bindings
                                or dep.partition_axis not in dep.reduction_axes or not reduced_context<=set(partition.context_axes)
                                or reduced_context&consumer_context or producer_context-reduced_context!=consumer_context&producer_context
                                or set(partition.context_axes)!=producer_context):
                            raise ValueError('partition classes must retain each actual producer/consumer context')
                if (dep.carry_axis is None) != (dep.initial_access_id is None):
                    raise ValueError('tile carry requires both its loop axis and actual initial producer')
                if dep.carry_axis is not None:
                    if dep.initial_access_id not in accesses:raise ValueError('unknown tile carry initial producer')
                    initial = accesses[dep.initial_access_id]
                    if not _tile_selected_predecessor_carry(prior,target,initial,dep,views) and (dep.kind is not EventKind.WRITE_VISIBLE or prior.operation is not Operation.WRITE
                            or initial.operation is not Operation.WRITE or target.operation is not Operation.READ
                            or dep.carry_axis not in target.iteration_axes or dep.carry_axis not in prior.iteration_axes
                            or dep.carry_axis in target.view_axes or dep.carry_axis in prior.view_axes
                            or initial.iteration_axes != tuple(axis for axis in prior.iteration_axes if axis != dep.carry_axis)
                            or not prior.view_id == initial.view_id == target.view_id
                            or not prior.view_axes == initial.view_axes == target.view_axes
                            or dep.reduction_axes or dep.coordinate_bindings):
                        raise ValueError('tile carry must reload the same state from its previous loop write or actual initial write')
                bindings=unique_by(dep.coordinate_bindings,'axis_id')
                coordinate_axes=set(bindings)-({class_binding.axis_id} if class_binding is not None else set())
                if selected_class_axis is not None:coordinate_axes.add(selected_class_axis)
                if (len(set(dep.reduction_axes)) != len(dep.reduction_axes)
                        or not set(dep.reduction_axes) <= set(prior.iteration_axes)
                        or set(prior.iteration_axes)-set(target.iteration_axes) != (set(dep.reduction_axes)-set(target.iteration_axes))|coordinate_axes
                        or set(target.iteration_axes)&coordinate_axes
                        or set(dep.reduction_axes)&coordinate_axes):
                    raise ValueError('tile reduction/broadcast dependency must conserve its iteration axes')
                for name,binding in bindings.items():
                    terms=unique_by(binding.terms,'axis_id')
                    if not set(terms) <= set(target.iteration_axes) or any(term.coefficient == 0 for term in binding.terms):
                        raise ValueError('tile coordinate binding must use actual target loop indices')
                    maximum=binding.offset+sum((axes[term.axis_id].count-1)*term.coefficient for term in binding.terms)
                    if maximum > U64_MAX:raise ValueError('tile coordinate expression exceeds uint64')
                    maximum//=binding.divisor
                    if binding.modulus is not None:maximum=min(maximum,binding.modulus-1)
                    if maximum >= axes[name].count:
                        raise ValueError('tile coordinate binding exceeds actual producer loop')
                if dep.carry_axis is None:visit(dep.access_id)
                else:visit(dep.initial_access_id)
            visiting.remove(identity);done.add(identity)
        for template in family.accesses:
            constrained = tuple(bound.index_axis for bound in family.causal_prefixes if bound.axis_id in template.iteration_axes)
            if len(set(constrained)) != len(constrained):
                raise ValueError('one access cannot multiply correlated prefixes of the same query axis')
            if template.view_id not in views:raise ValueError('unknown tile operand view')
            if template.scope is not None and template.scope.call_id != family.scope.call_id:
                raise ValueError('tile participant scope must retain its actual caller identity')
            view = views[template.view_id]
            if template.predicate is not None:
                predicate=template.predicate
                if (program.coverage.semantics is not Semantics.LOGICAL or program.coverage.executable
                        or predicate.source_access_id not in accesses):
                    raise ValueError('conditional tile requires its actual logical signed scalar read')
                source=accesses[predicate.source_access_id]
                if source.view_id not in views:raise ValueError('conditional tile read view missing')
                source_view=views[source.view_id]
                references=tuple(dep for dep in template.dependencies if dep.access_id==source.access_id and dep.kind is EventKind.READ_READY)
                if (source.operation is not Operation.READ or source.predicate is not None
                        or source_view.dtype not in ('int32','int64') or not source.iteration_axes
                        or len(source_view.shape)!=len(source.view_axes)
                        or any(name is None or name not in source.iteration_axes or axes[name].block!=1 for name in source.view_axes)
                        or len(references)!=1 or references[0].reduction_axes or references[0].carry_axis is not None
                        or references[0].partition_axis is not None):
                    raise ValueError('conditional tile requires its own actual signed scalar READ_READY')
            for dependency in template.dependencies:
                if dependency.allow_skipped_read:
                    prior=accesses[dependency.access_id]
                    if (dependency.kind is not EventKind.READ_READY or prior.operation is not Operation.READ
                            or prior.predicate is None or template.predicate is not None
                            or dependency.reduction_axes or dependency.coordinate_bindings
                            or dependency.carry_axis is not None or dependency.partition_axis is not None
                            or prior.iteration_axes!=template.iteration_axes
                            or TileDependency(prior.predicate.source_access_id,EventKind.READ_READY,()) not in template.dependencies):
                        raise ValueError('optional read join requires its own unconditional scalar guard and matching tile')
                if dependency.access_id in accesses and accesses[dependency.access_id].predicate is not None:
                    if template.predicate!=accesses[dependency.access_id].predicate and not dependency.allow_skipped_read:
                        raise ValueError('conditional producer readiness requires the same access predicate')
            if template.constant_value is not None:
                if template.operation is not Operation.WRITE:raise ValueError('constant value requires an actual write')
                if template.constant_value is ConstantValue.NEGATIVE_INFINITY and view.dtype not in ('fp16','bf16','fp32','fp64'):
                    raise ValueError('negative infinity requires floating-point storage')
                if template.constant_value is ConstantValue.NEGATIVE_ONE and view.dtype not in ('int8','int32','int64','fp16','bf16','fp32','fp64'):
                    raise ValueError('negative one requires signed storage')
            if isinstance(view, LogicalSelectionView) and not template.unmapped:
                raise ValueError('logical selection tiles must keep their mapping unresolved')
            for bound in family.causal_prefixes:
                if bound.axis_id in template.iteration_axes and (bound.index_axis not in template.iteration_axes
                        or template.iteration_axes.index(bound.index_axis) > template.iteration_axes.index(bound.axis_id)):
                    raise ValueError('causal key iteration requires its earlier query coordinate')
            work=(_tile_template_requested_elements(views,axes,family,template) if view.element_bytes is None
                else _tile_template_requested_bytes(views,axes,family,template))
            if _tile_template_instances(axes,family,template) > U64_MAX or work > U64_MAX:
                raise ValueError('compressed tile work exceeds uint64 contract')
            divisors=_validate_tile_operand_geometry(axis_values,view.shape,template.iteration_axes,
                template.view_axes,template.view_axis_divisors,template.view_axis_padding)
            if template.indirect_target is None and not template.unmapped and allocations[view.allocation_id].size_bytes is None:
                raise ValueError('unknown-capacity tile requires an explicit unresolved selection')
            if template.operation is Operation.READ and not template.input_ready:
                writers=tuple(accesses[dep.access_id] for dep in template.dependencies
                    if dep.access_id in accesses and dep.kind is EventKind.WRITE_VISIBLE
                    and accesses[dep.access_id].view_id == template.view_id
                    and accesses[dep.access_id].indirect_target is None
                    and accesses[dep.access_id].view_axes == template.view_axes
                    and (accesses[dep.access_id].view_axis_divisors or (1,)*len(view.shape)) == divisors)
                if (not writers and not _tile_selected_parent_is_written(view,template,views,accesses)
                        and not _tile_typed_parent_fields_are_written(view,template,views,accesses,family)
                        and not _tile_row_initializers_cover(view,template,views,accesses,family)
                        and not _tile_partition_initializer(view,template,views,accesses,family)):
                    raise ValueError(f'tile read lacks actual initialized/visible selection: {template.access_id}')
            for ref in template.input_ready:
                validate_input_readiness(view,ref,unmapped=template.unmapped,views=views,
                    events=events,visibility_domain=family.visibility_domain)
            if template.unmapped and program.coverage.executable:
                raise ValueError('unknown tile coordinates cannot be submitted to an executable backend')
            if template.indirect_target is not None:
                target=template.indirect_target
                if (template.operation is not Operation.WRITE or program.coverage.executable or len(view.shape) != 3
                        or target.allocation_id not in allocations or target.index_access_id not in accesses):
                    raise ValueError('indirect tile target requires an actual logical write/index/cache object')
                index=accesses[target.index_access_id]
                references=tuple(dep for dep in template.dependencies if dep.access_id==index.access_id and dep.kind is EventKind.READ_READY)
                if (index.operation is not Operation.READ or views[index.view_id].dtype not in ('int32','int64')
                        or len(references)!=1 or references[0].reduction_axes or references[0].carry_axis is not None
                        or references[0].partition_axis is not None or index.view_axes != template.view_axes[:1]):
                    raise ValueError('indirect tile requires its actual integer index tile READ_READY')
                if target.value_source_access_ids:
                    if len(set(target.value_source_access_ids)) != len(target.value_source_access_ids):
                        raise ValueError('indirect value sources must be distinct actual reads')
                    for identity in target.value_source_access_ids:
                        if identity not in accesses:raise ValueError('indirect value source is unknown')
                        source = accesses[identity]
                        if (identity == target.index_access_id or source.operation is not Operation.READ
                                or source.iteration_axes != template.iteration_axes or source.indirect_target is not None
                                or source.view_axes[:1] != template.view_axes[:1]
                                or TileDependency(identity,EventKind.READ_READY,()) not in template.dependencies):
                            raise ValueError('computed indirect store requires each actual token value READ_READY')
                else:
                    payloads=tuple(accesses[dep.access_id] for dep in template.dependencies
                        if dep.access_id in accesses and dep.kind is EventKind.READ_READY
                        and accesses[dep.access_id].view_id == template.view_id)
                    if len(payloads) != 1 or payloads[0].operation is not Operation.READ or payloads[0].view_axes != template.view_axes:
                        raise ValueError('indirect tile requires one actual matching payload READ_READY')
            if not static_graph:visit(template.access_id)
        if static_graph:
            _validate_static_tile_dependency_graph(tuple(
                (access.access_id,access.operation,access.iteration_axes,access.dependencies)
                for access in family.accesses))
        if used_partitions!={(p.axis_id,p.class_axis_id,p.context_axes) for p in family.partitions}:
            raise ValueError('logical tile partitions require their actual dependency consumers')
    _validate_tile_availability(program)


def _validate_tile_availability(program):
    """A field may become available only after its complete static writer.

    The reference denotes all actual writer tiles, not an invented boundary
    initializer. Consumers in other families retain that field identity.
    """
    if not any(event.tile_sources for event in program.events):return
    families=unique_by(program.tile_programs,'tile_program_id');views=unique_by(program.views,'view_id')
    edges={('event',event.event_id):[('event',ref.node_id) for ref in event.dependencies]
        for event in program.events}
    for family in program.tile_programs:
        for access in family.accesses:
            edges['tile',family.tile_program_id,access.access_id]=[
                ('tile',family.tile_program_id,dep.access_id) for dep in access.dependencies if dep.carry_axis is None]
            edges['tile',family.tile_program_id,access.access_id].extend(('event',ref.node_id) for ref in access.input_ready)
    for event in program.events:
        if not event.tile_sources:continue
        if (event.kind not in (EventKind.INITIALIZED,EventKind.LOGICAL_AVAILABLE) or event.view_id not in views
                or len(event.tile_sources)!=1):
            raise ValueError('tile availability requires one actual complete field writer')
        target=views[event.view_id]
        if not isinstance(target,TensorView) or target.allocation_id!=event.allocation_id:
            raise ValueError('tile availability requires its actual typed target field')
        source,=event.tile_sources
        if source.tile_program_id not in families:raise ValueError('tile availability writer family is missing')
        family=families[source.tile_program_id];accesses=unique_by(family.accesses,'access_id')
        if family.visibility_domain!=event.visibility_domain:raise ValueError('tile availability writer visibility differs')
        if source.access_id not in accesses:raise ValueError('tile availability writer access is missing')
        writer=accesses[source.access_id];axes=unique_by(family.axes,'axis_id')
        if (writer.operation is not Operation.WRITE or writer.view_id!=target.view_id
                or writer.predicate is not None or writer.indirect_target is not None
                or writer.value_index is not None or writer.atomic_add is not None
                or writer.view_axis_divisors or writer.view_axis_padding or family.causal_prefixes or family.partitions
                or any(isinstance(axes[name].extent,SymbolicExtent) or axes[name].value_limit_access_id is not None
                    for name in writer.iteration_axes)
                or any(name not in writer.view_axes for name in writer.iteration_axes) and writer.constant_value is None):
            raise ValueError('tile availability requires unconditional complete static field coverage')
        if isinstance(event.region,Unmapped):
            if (event.region.logical_shape,event.region.dtype,event.region.element_bytes)!=(target.shape,target.dtype,target.element_bytes):
                raise ValueError('tile availability logical field shape/encoding mismatch')
        elif event.region!=target.region():
            raise ValueError('tile availability resolved field region mismatch')
        edges['event',event.event_id].append(('tile',family.tile_program_id,writer.access_id))
    visiting=set();done=set()
    def visit(node):
        if node in visiting:raise ValueError('cross-family tile availability cycle')
        if node in done:return
        visiting.add(node)
        for prior in edges[node]:
            # Ordinary event dependencies on flat accesses are checked by
            # validate_program; compressed programs have no flat accesses.
            if prior not in edges:raise ValueError('tile availability dependency is missing')
            visit(prior)
        visiting.remove(node);done.add(node)
    for node in edges:visit(node)


def _tile_bound_coordinates(coordinates,bindings):
    result=dict(coordinates)
    for binding in bindings:
        value=(binding.offset+sum(coordinates[term.axis_id]*term.coefficient for term in binding.terms))//binding.divisor
        result[binding.axis_id]=value if binding.modulus is None else value%binding.modulus
    return result


def _tile_partition_class(partition,axes,coordinates):
    if partition.members is None:raise ValueError('tile partition membership is unbound')
    context=0
    for name in partition.context_axes:
        context=context*axes[name].count+coordinates[name]
    return partition.members[context*axes[partition.axis_id].count+coordinates[partition.axis_id]]


def expand_tile_program(program, *, max_accesses=10000,max_dependencies=100000,value_limits=None):
    """Materialize only a bounded logical example; production stays compressed."""
    from dataclasses import replace
    program.__post_init__()
    if program_symbolic_extents(program):raise ValueError('symbolic logical cardinality is unbound; expansion requires explicit values')
    limits=_tile_value_limit_bindings(program,value_limits,required=True)
    if not program.tile_programs:return program
    if any(access.value_index is not None or access.atomic_add is not None for family in program.tile_programs for access in family.accesses):
        raise ValueError('data-dependent tile index values and atomic linearization require a value-aware consumer; plain DAG expansion would erase semantics')
    if any(partition.members is None for family in program.tile_programs for partition in family.partitions):
        raise ValueError('tile partition membership is unbound; explicit dependency expansion requires its values')
    count = sum(_tile_limited_instances(family,access,limits[family.tile_program_id] if family.tile_program_id in limits else {})
        for family in program.tile_programs for access in family.accesses)
    if count > max_accesses:raise ValueError('tile expansion exceeds explicit access budget')
    check_type(max_accesses,int);check_type(max_dependencies,int)
    # A row reduction can feed every output coordinate in that row. Bound
    # references as well as access count before allocating a quadratic DAG.
    references=sum(len(event.dependencies) for event in program.events)
    for family in program.tile_programs:
        axes=unique_by(family.axes,'axis_id')
        bound=limits[family.tile_program_id] if family.tile_program_id in limits else {}
        for template in family.accesses:
            per_tile=len(template.input_ready)*(2 if template.unmapped and template.operation is Operation.READ else 1)
            per_tile+=int(template.operation is Operation.WRITE)
            per_tile+=sum(prod(bound[name] if name in bound else axes[name].count for name in dependency.reduction_axes) for dependency in template.dependencies)
            references+=_tile_limited_instances(family,template,bound)*per_tile
    by_family=unique_by(program.tile_programs,'tile_program_id')
    for event in program.events:
        for source in event.tile_sources:
            family=by_family[source.tile_program_id]
            writer=unique_by(family.accesses,'access_id')[source.access_id]
            references+=tile_template_instances(family,writer)
    if references > max_dependencies:
        raise ValueError('tile dependency expansion exceeds explicit reference budget (conservative bound)')
    views,allocations = unique_by(program.views,'view_id'),unique_by(program.allocations,'allocation_id')
    accesses,scopes,events = [],list(program.scopes),list(program.events)
    for family in program.tile_programs:
        axes,templates = unique_by(family.axes,'axis_id'),unique_by(family.accesses,'access_id')
        limit=limits[family.tile_program_id] if family.tile_program_id in limits else {}
        def identity(template,coordinates):
            suffix='.'.join(f'{name}{coordinates[name]*axes[name].block}' for name in template.iteration_axes)
            return f'{family.tile_program_id}.{template.access_id}.{suffix}'
        def scalar_identity(template,source_id,coordinates):
            references=tuple(dep for dep in template.dependencies if dep.access_id==source_id and dep.kind is EventKind.READ_READY)
            if (len(references)!=1 or references[0].reduction_axes or references[0].carry_axis is not None
                    or references[0].partition_axis is not None):
                raise ValueError('scalar tile reference requires one actual coordinate-bound READ_READY')
            return identity(templates[source_id],_tile_bound_coordinates(coordinates,references[0].coordinate_bindings))
        for template in family.accesses:
            for coordinates in _tile_coordinates(family,template,axes,limit):
                view=views[template.view_id]
                divisors=template.view_axis_divisors or (1,)*len(view.shape)
                starts=tuple(0 if name is None else coordinates[name]*axes[name].block//divisor
                    for name,divisor in zip(template.view_axes,divisors,strict=True))
                shape=tuple(dimension if name is None else min(axes[name].block//divisor,dimension-start)
                    for dimension,name,start,divisor in zip(view.shape,template.view_axes,starts,divisors,strict=True))
                name=identity(template,coordinates)
                selection=f'{family.tile_program_id}.{view.view_id}.'+'.'.join(str(start) for start in starts)
                region=(Unmapped(shape,view.dtype,view.element_bytes,selection) if template.unmapped else
                    Strided(view.offset_bytes+sum(start*stride for start,stride in zip(starts,view.strides_bytes,strict=True)),
                        shape,view.strides_bytes,view.element_bytes))
                dependencies=list(template.input_ready)
                if template.unmapped and template.operation is Operation.READ:
                    # The compressed source proves an exact logical partition,
                    # while the parent coordinate map stays unknown. Materialize
                    # this tile's availability from its actual parent readiness;
                    # it is never a resolved initialization or an address claim.
                    dependencies=[]
                    for index,ready in enumerate(template.input_ready):
                        available=f'{name}.available{index}'
                        events.append(ProgramEvent(available,EventKind.LOGICAL_AVAILABLE,view.allocation_id,
                            region,family.visibility_domain,(ready,)))
                        dependencies.append(EventRef(available,EventKind.LOGICAL_AVAILABLE))
                for dep in template.dependencies:
                    prior=templates[dep.access_id]
                    carried = None
                    if dep.carry_axis is not None:
                        if coordinates[dep.carry_axis] == 0:prior=templates[dep.initial_access_id]
                        else:carried = coordinates[dep.carry_axis]-1
                    for reductions in product(*(range(limit[axis] if axis in limit else axes[axis].count) for axis in dep.reduction_axes)):
                        merged={**coordinates,**dict(zip(dep.reduction_axes,reductions,strict=True))}
                        if dep.partition_axis is not None:
                            partition,inverse=_tile_dependency_partition(family,dep,prior,template)
                            if inverse:
                                merged[partition.class_axis_id]=_tile_partition_class(partition,axes,coordinates)
                            else:
                                actual_class=_tile_partition_class(partition,axes,merged)
                                expected_class=(_tile_partition_class(partition,axes,coordinates) if partition.class_axis_id is None
                                    else _tile_bound_coordinates(coordinates,dep.coordinate_bindings)[partition.class_axis_id])
                                if actual_class!=expected_class:continue
                        if carried is not None:merged[dep.carry_axis]=carried
                        bound=_tile_bound_coordinates(coordinates,dep.coordinate_bindings)
                        merged.update((binding.axis_id,bound[binding.axis_id]) for binding in dep.coordinate_bindings)
                        if any(axis in limit and merged[axis]>=limit[axis] for axis in prior.iteration_axes):
                            raise ValueError('tile dependency selects an inactive value-limited producer')
                        if not _tile_coordinate_allowed(family,axes,{name:merged[name] for name in prior.iteration_axes}):continue
                        dependencies.append(EventRef(identity(prior,merged),dep.kind,dep.allow_skipped_read))
                allocation_id=view.allocation_id
                if template.indirect_target is not None:
                    target=template.indirect_target
                    allocation_id=target.allocation_id
                    region=Indirect(shape,view.dtype,view.element_bytes,scalar_identity(template,target.index_access_id,coordinates),
                        target.mapping,target.component,
                        view.view_id if target.allocation_id==view.allocation_id and any(axis is not None for axis in template.view_axes[1:]) else None,
                        starts if target.allocation_id==view.allocation_id and any(axis is not None for axis in template.view_axes[1:]) else ())
                allocated=EventRef(allocations[allocation_id].allocated_event,EventKind.ALLOCATED)
                if (template.unmapped and template.operation is Operation.READ and not template.input_ready
                        and (_tile_selected_parent_is_written(view,template,views,templates)
                             or _tile_typed_parent_fields_are_written(view,template,views,templates,family)
                             or _tile_partition_initializer(view,template,views,templates,family))):
                    # A complete producer partition proves visibility of an
                    # unresolved selected row. Preserve all real write/index
                    # dependencies; this is not an invocation initializer.
                    available=f'{name}.producer_selection'
                    events.append(ProgramEvent(available,EventKind.LOGICAL_AVAILABLE,allocation_id,
                        region,family.visibility_domain,tuple(dependencies)))
                    dependencies.append(EventRef(available,EventKind.LOGICAL_AVAILABLE))
                if template.operation is Operation.WRITE and allocated not in dependencies:dependencies.append(allocated)
                scopes.append(replace(family.scope if template.scope is None else template.scope,scope_id=name,tile=tuple(coordinates[axis]*axes[axis].block
                    for axis in template.iteration_axes)))
                accesses.append(MemoryAccess(name,name,allocation_id,None if template.unmapped or template.indirect_target is not None else view.view_id,
                    template.operation,region,Semantics.LOGICAL,None,None,None,None,family.visibility_domain,
                    tuple(dependencies),EventKind.READ_READY if template.operation is Operation.READ else EventKind.WRITE_VISIBLE,
                    1,template.evidence_refs,template.constant_value,
                    None if template.predicate is None else replace(template.predicate,
                        source_access_id=scalar_identity(template,template.predicate.source_access_id,coordinates))))
    coverage=program.coverage
    if any(event.tile_sources for event in events):
        expanded_by_id=unique_by(accesses,'access_id');expanded_events=[]
        for event in events:
            dependencies=list(event.dependencies)
            for source in event.tile_sources:
                family=by_family[source.tile_program_id];axes=unique_by(family.axes,'axis_id')
                writer=unique_by(family.accesses,'access_id')[source.access_id]
                for coordinates in _tile_coordinates(family,writer,axes,{}):
                    suffix='.'.join(f'{name}{coordinates[name]*axes[name].block}' for name in writer.iteration_axes)
                    identity=f'{family.tile_program_id}.{writer.access_id}.{suffix}'
                    if identity not in expanded_by_id:raise ValueError('tile availability lost its actual expanded writer')
                    dependencies.append(EventRef(identity,EventKind.WRITE_VISIBLE))
            expanded_events.append(replace(event,dependencies=tuple(dependencies),tile_sources=()))
        events=expanded_events
    if value_limits:
        supplied='; '.join(f'{family}/{source}={value}' for (family,source),value in sorted(value_limits.items()))
        coverage=replace(coverage,limitations=(*coverage.limitations,'Consumer-supplied scalar READ values for bounded expansion: '+supplied))
    return replace(program,coverage=coverage,scopes=tuple(scopes),events=tuple(events),accesses=tuple(accesses),tile_programs=())


def unique_by(values, field):
    result = {}
    for value in values:
        key = getattr(value, field)
        if key in result:
            raise ValueError(f"duplicate {field}: {key}")
        result[key] = value
    return result


def validate_storage_bindings(storage_allocations,storage_views,storage_events,coverage,
                              *,storage_scopes=(),storage_accesses=()):
    """Validate actual storage/lifetimes without constructing an access graph.

    Detailed programs and direct analytical consumers share these same
    boundary checks. Optional scopes/accesses supply the existing flat graph
    nodes when a complete program is being validated.
    """
    allocations = unique_by(storage_allocations, "allocation_id")
    views = unique_by(storage_views, "view_id")
    scopes = unique_by(storage_scopes, "scope_id")
    events = unique_by(storage_events, "event_id")
    accesses = unique_by(storage_accesses, "access_id")
    if set(events) & set(accesses):
        raise ValueError("event/access identity collision")
    nodes = events | accesses
    for view in views.values():
        if view.allocation_id not in allocations:
            raise ValueError("unknown view allocation")
        if isinstance(view, LogicalSelectionView):
            if coverage.executable or coverage.semantics is not Semantics.LOGICAL or not coverage.limitations:
                raise ValueError("logical selection requires evidenced nonexecutable logical coverage")
            if isinstance(view,LogicalPartialSelectionView):view.__post_init__()
            if view.coordinate_map:
                if view.parent_view_id is None or view.parent_view_id not in views:
                    raise ValueError('logical coordinate map requires its actual parent')
                parent = views[view.parent_view_id]
                view._validate_parent_coordinates(parent.shape)
            seen = {view.view_id}
            parent = view
            while isinstance(parent, LogicalSelectionView) and parent.parent_view_id is not None:
                if parent.parent_view_id not in views or parent.parent_view_id in seen:
                    raise ValueError("logical selection has unknown or cyclic parent")
                parent = views[parent.parent_view_id]
                seen.add(parent.view_id)
                if not isinstance(parent, (TensorView, LogicalSelectionView)):
                    raise NotImplementedError("logical selection requires an unpacked typed parent")
                if (parent.allocation_id, parent.dtype, parent.element_bytes) != (
                        view.allocation_id, view.dtype, view.element_bytes):
                    raise ValueError("logical selection must retain its actual typed parent")
            continue
        capacity = allocations[view.allocation_id].size_bytes
        if capacity is not None and region_end(view.region()) > capacity:
            raise ValueError("view exceeds allocation")
    for allocation in allocations.values():
        if allocation.size_bytes is None and (
                coverage.executable or coverage.semantics is not Semantics.LOGICAL
                or not coverage.limitations):
            raise ValueError("unknown capacity requires evidenced nonexecutable logical coverage")
    ancestors = {}
    visiting = set()

    def visit(node_id):
        if node_id in ancestors:
            return ancestors[node_id]
        if node_id in visiting:
            raise ValueError("dependency cycle")
        visiting.add(node_id)
        result = set()
        node = nodes[node_id]
        if len(set(node.dependencies)) != len(node.dependencies):
            raise ValueError("duplicate dependencies")
        for dep in node.dependencies:
            if dep.node_id not in nodes:
                raise ValueError("unknown dependency")
            target = nodes[dep.node_id]
            valid = ((EventKind.ACCEPTED, target.completion, EventKind.RETIRED)
                     if isinstance(target, MemoryAccess) else (target.kind,))
            if dep.kind not in valid:
                raise ValueError("dependency event kind mismatch")
            if dep.allow_skipped_read:
                if (not isinstance(node,MemoryAccess) or not isinstance(target,MemoryAccess)
                        or target.operation is not Operation.READ or target.predicate is None
                        or node.predicate is not None or dep.kind is not EventKind.READ_READY
                        or EventRef(target.predicate.source_access_id,EventKind.READ_READY) not in node.dependencies):
                    raise ValueError('optional read join requires its own actual scalar guard READ_READY')
            result.add(dep)
            result.update(visit(dep.node_id))
        visiting.remove(node_id)
        ancestors[node_id] = result
        return result

    for node_id in nodes:
        visit(node_id)
    for event in events.values():
        if event.view_id is not None:
            if event.view_id not in views:raise ValueError('availability field view is missing')
            field=views[event.view_id]
            if (event.kind not in (EventKind.INITIALIZED,EventKind.LOGICAL_AVAILABLE)
                    or field.allocation_id!=event.allocation_id):
                raise ValueError('availability field does not match its typed allocation')
        if event.kind in (EventKind.COMPUTE_DONE, EventKind.BARRIER):
            if event.allocation_id is not None or event.region is not None or event.visibility_domain is not None:
                raise ValueError("compute/barrier cannot initialize storage")
        elif event.kind is EventKind.LOGICAL_AVAILABLE:
            if (event.allocation_id not in allocations or not isinstance(event.region, Unmapped)
                    or event.visibility_domain is None or coverage.executable
                    or coverage.semantics is not Semantics.LOGICAL or not coverage.limitations
                    or allocations[event.allocation_id].address_level is not AddressLevel.OBJECT):
                raise ValueError("logical availability requires an evidenced unaddressed nonexecutable logical selection")
            if EventRef(allocations[event.allocation_id].allocated_event, EventKind.ALLOCATED) not in ancestors[event.event_id]:
                raise ValueError("logical availability must follow its own allocation")
        elif event.kind in (EventKind.ALLOCATED, EventKind.INITIALIZED, EventKind.FREED):
            if event.allocation_id not in allocations:
                raise ValueError("lifecycle allocation missing")
            allocation = allocations[event.allocation_id]
            if event.kind is EventKind.INITIALIZED:
                if event.region is None or event.visibility_domain is None:
                    raise ValueError("initialization requires region and visibility domain")
                if isinstance(event.region, (Indexed, Indirect, Unmapped)):
                    raise ValueError("initialization must declare a resolved storage region")
                if allocation.size_bytes is None:
                    raise ValueError("resolved initialization requires known allocation capacity")
                if region_end(event.region) > allocation.size_bytes:
                    raise ValueError("initialization exceeds allocation")
            elif event.region is not None or event.visibility_domain is not None:
                raise ValueError("allocate/free does not imply initialization")
            if event.kind is not EventKind.ALLOCATED and EventRef(allocation.allocated_event, EventKind.ALLOCATED) not in ancestors[event.event_id]:
                raise ValueError("lifecycle must follow allocation")
        else:
            raise ValueError("backend event cannot be caller-declared")
    # Logical tile scratch can declare thousands of independent lifetimes.
    # Index their events once instead of rescanning the complete event set for
    # each allocation; compute/barrier events have no storage lifetime.
    lifecycle_by_allocation = {identity: [] for identity in allocations}
    for event in events.values():
        if event.allocation_id in lifecycle_by_allocation:
            lifecycle_by_allocation[event.allocation_id].append(event)
    for allocation in allocations.values():
        if allocation.allocated_event not in events:
            raise ValueError("allocation event missing")
        event = events[allocation.allocated_event]
        if event.kind is not EventKind.ALLOCATED or event.allocation_id != allocation.allocation_id:
            raise ValueError("allocation event identity mismatch")
        if allocation.freed_event is not None:
            if allocation.freed_event not in events:
                raise ValueError("free event missing")
            event = events[allocation.freed_event]
            if event.kind is not EventKind.FREED or event.allocation_id != allocation.allocation_id:
                raise ValueError("free event identity mismatch")
        if allocation.freed_event is not None:
            for initializer in lifecycle_by_allocation[allocation.allocation_id]:
                if initializer.kind in (EventKind.INITIALIZED, EventKind.LOGICAL_AVAILABLE):
                    if EventRef(initializer.event_id, EventKind.INITIALIZED) not in ancestors[allocation.freed_event]:
                        raise ValueError("free must await initialization")
        lifecycle = lifecycle_by_allocation[allocation.allocation_id]
        if sum(e.kind is EventKind.ALLOCATED for e in lifecycle) != 1:
            raise ValueError("allocation must have a unique lifetime")
        if sum(e.kind is EventKind.FREED for e in lifecycle) != int(allocation.freed_event is not None):
            raise ValueError("unbound free event")
    # OBJECT identities explicitly have no address. Only addressed allocations
    # can overlap in the address space checked below.
    addressed = tuple(allocation for allocation in storage_allocations
        if allocation.address_level is not AddressLevel.OBJECT)
    for left_index, left in enumerate(addressed):
        for right in addressed[left_index + 1:]:
            if (left.address_level is right.address_level and left.address_level is not AddressLevel.OBJECT
                    and left.device_id == right.device_id and left.memory_domain == right.memory_domain
                    and left.mapping_ref == right.mapping_ref
                    and max(left.base_address_bytes, right.base_address_bytes)
                    < min(left.base_address_bytes + left.size_bytes, right.base_address_bytes + right.size_bytes)):
                raise NotImplementedError("overlapping addressed allocations require a shared allocation/view binding")
    return allocations,views,scopes,events,accesses,ancestors


def validate_program(program):
    allocations,views,scopes,events,accesses,ancestors=validate_storage_bindings(
        program.allocations,program.views,program.events,program.coverage,
        storage_scopes=program.scopes,storage_accesses=program.accesses)
    for access in accesses.values():
        if access.allocation_id not in allocations or access.scope_id not in scopes:
            raise ValueError("access references unknown allocation/scope")
        allocation = allocations[access.allocation_id]
        if not isinstance(access.region, (Indirect, Unmapped)) and allocation.size_bytes is None:
            raise ValueError("unknown-capacity object requires an indirect payload")
        if not isinstance(access.region, (Indirect, Unmapped)) and region_end(access.region) > allocation.size_bytes:
            raise ValueError("access region exceeds allocation")
        if access.view_id is not None:
            if access.view_id not in views:
                raise ValueError("unknown view")
            view = views[access.view_id]
            if view.allocation_id != access.allocation_id or not covers(view.region(), access.region):
                raise ValueError("access is outside bound view")
        if access.constant_value in (ConstantValue.NEGATIVE_INFINITY,ConstantValue.NEGATIVE_ONE):
            if access.view_id is not None:dtype = views[access.view_id].dtype
            elif isinstance(access.region,Unmapped):dtype = access.region.dtype
            else:raise ValueError('constant requires an actual typed region/view')
            if access.constant_value is ConstantValue.NEGATIVE_INFINITY and dtype not in ('fp16','bf16','fp32','fp64'):
                raise ValueError('negative infinity requires floating-point storage')
            if access.constant_value is ConstantValue.NEGATIVE_ONE and dtype not in ('int8','int32','int64','fp16','bf16','fp32','fp64'):
                raise ValueError('negative one requires signed storage')
        if access.semantics is not program.coverage.semantics:
            raise ValueError("coverage semantics mismatch")
        deps = ancestors[access.access_id]
        if access.predicate is not None:
            predicate=access.predicate
            if (program.coverage.semantics is not Semantics.LOGICAL or program.coverage.executable
                    or predicate.source_access_id not in accesses):
                raise ValueError('conditional access requires its actual logical signed scalar read')
            source=accesses[predicate.source_access_id]
            dtype=(views[source.view_id].dtype if source.view_id is not None
                else source.region.dtype if isinstance(source.region,(Unmapped,Indirect)) else None)
            if (source.operation is not Operation.READ or source.predicate is not None or dtype not in ('int32','int64')
                    or requested_bytes(source.region)*source.repeat!={'int32':4,'int64':8}[dtype]
                    or EventRef(source.access_id,EventKind.READ_READY) not in access.dependencies):
                raise ValueError('conditional access requires its actual signed scalar READ_READY')
        for dependency in deps:
            if dependency.node_id in accesses and accesses[dependency.node_id].predicate is not None:
                if (access.predicate!=accesses[dependency.node_id].predicate
                        and not (dependency.allow_skipped_read and accesses[dependency.node_id].operation is Operation.READ)):
                    raise ValueError('conditional producer readiness requires the same access predicate')
        if isinstance(access.region, Unmapped):
            if (access.semantics is not Semantics.LOGICAL or program.coverage.executable or access.view_id is not None
                    or allocation.address_level is not AddressLevel.OBJECT or not access.evidence_refs
                    or not program.coverage.limitations):
                raise ValueError("unmapped selection requires evidenced nonexecutable logical object coverage")
        if isinstance(access.region, (Indexed, Indirect)):
            if (access.semantics is not Semantics.LOGICAL or program.coverage.executable
                    or not access.evidence_refs or not program.coverage.limitations):
                raise ValueError("unresolved indices require evidenced nonexecutable logical coverage")
            if isinstance(access.region,Indirect) and access.region.parent_view_id is not None:
                region=access.region
                if region.parent_view_id not in views:raise ValueError('indirect logical parent view is absent')
                parent=views[region.parent_view_id]
                if ((parent.allocation_id,parent.dtype,parent.element_bytes)!=(access.allocation_id,region.dtype,region.element_bytes)
                        or len(parent.shape)!=len(region.logical_shape)
                        or any(start+count>extent for start,count,extent in zip(region.logical_starts,region.logical_shape,parent.shape,strict=True))):
                    raise ValueError('indirect logical coordinates exceed or differ from their actual parent')
            index_id = access.region.index_access_id
            if (index_id not in accesses or accesses[index_id].operation is not Operation.READ
                    or EventRef(index_id, EventKind.READ_READY) not in deps):
                raise ValueError("indexed access must depend on its index read readiness")
            index_read = accesses[index_id]
            if (index_read.view_id not in views
                    or views[index_read.view_id].dtype not in ("int32", "int64")):
                raise ValueError("indexed access requires an integer index view")
            index_width = views[index_read.view_id].element_bytes
            index_count = (access.region.logical_shape[0] if isinstance(access.region, Indirect)
                           else access.region.index_count)
            if requested_bytes(index_read.region) < index_count * index_width:
                raise ValueError("index read does not cover the declared index count")
        if EventRef(allocation.allocated_event, EventKind.ALLOCATED) not in deps:
            raise ValueError("access before allocation")
        if allocation.freed_event is not None:
            if EventRef(access.access_id, EventKind.RETIRED) not in ancestors[allocation.freed_event]:
                raise ValueError("free must await every live request retirement")
            if EventRef(allocation.freed_event, EventKind.FREED) in deps:
                raise ValueError("access after free")
        if isinstance(access.region, Indirect) and access.operation is Operation.READ:
            raise NotImplementedError("indirect reads require caller visibility evidence not yet expressed")
        if access.operation is Operation.READ:
            visible = []
            for dep in deps:
                if dep.kind in (EventKind.INITIALIZED, EventKind.LOGICAL_AVAILABLE):
                    event = events[dep.node_id]
                    if event.allocation_id == access.allocation_id and event.visibility_domain == access.visibility_domain:
                        visible.append(event.region)
                elif dep.kind is EventKind.WRITE_VISIBLE:
                    writer = accesses[dep.node_id]
                    if writer.allocation_id == access.allocation_id and writer.visibility_domain == access.visibility_domain:
                        visible.append(writer.region)
            if not _union_covers(tuple(visible), access.region):
                raise ValueError("read lacks initialized/visible region dependency")
    # Ordering is a claim about memory accesses, not serialization order.
    if program.coverage.ordering is Ordering.UNORDERED:
        if any(dep.node_id in accesses for a in accesses.values() for dep in ancestors[a.access_id]):
            raise ValueError("unordered coverage contains access ordering")
    elif program.coverage.ordering is Ordering.TOTAL:
        if any(a.repeat > 1 for a in accesses.values()):
            raise ValueError("independent repeat occurrences do not declare a total order")
        ids = tuple(accesses)
        for i, left in enumerate(ids):
            for right in ids[i + 1:]:
                if not (any(d.node_id == left for d in ancestors[right]) or any(d.node_id == right for d in ancestors[left])):
                    raise ValueError("total order claim lacks dependency")


@dataclass(frozen=True)
class AccessStatistics(Typed):
    allocation_id: str
    read_requested_bytes: int
    write_requested_bytes: int
    unique_footprint_bytes: int | None
    conditional_read_requested_bytes: int = 0
    conditional_write_requested_bytes: int = 0

    def __post_init__(self):
        super().__post_init__()
        if (self.conditional_read_requested_bytes>self.read_requested_bytes
                or self.conditional_write_requested_bytes>self.write_requested_bytes):
            raise ValueError('conditional request envelope exceeds its candidate bytes')

    @property
    def exact_read_requested_bytes(self):
        return self.read_requested_bytes if self.conditional_read_requested_bytes==0 else None

    @property
    def exact_write_requested_bytes(self):
        return self.write_requested_bytes if self.conditional_write_requested_bytes==0 else None


def statistics(program: MemoryAccessProgram, *, value_limits=None):
    limits=_tile_value_limit_bindings(program,value_limits,required=False)
    result = []
    views = unique_by(program.views,'view_id')
    accesses_by_allocation = {allocation.allocation_id: [] for allocation in program.allocations}
    templates_by_allocation = {allocation.allocation_id: [] for allocation in program.allocations}
    for access in program.accesses:
        if access.allocation_id in accesses_by_allocation:
            accesses_by_allocation[access.allocation_id].append(access)
    for family in program.tile_programs:
        axes = unique_by(family.axes,'axis_id')
        for template in family.accesses:
            view = views[template.view_id]
            target_id = view.allocation_id if template.indirect_target is None else template.indirect_target.allocation_id
            if target_id in templates_by_allocation:
                templates_by_allocation[target_id].append((family,axes,template))
    for allocation in program.allocations:
        reads = writes = conditional_reads = conditional_writes = 0
        regions = []
        for access in accesses_by_allocation[allocation.allocation_id]:
            if access.allocation_id != allocation.allocation_id:
                continue
            count = requested_bytes(access.region) * access.repeat
            if access.operation is Operation.READ:
                reads += count
                if access.predicate is not None:conditional_reads+=count
            elif access.operation is Operation.WRITE:
                writes += count
                if access.predicate is not None:conditional_writes+=count
            else:
                raise ValueError("unknown operation")
            regions.append(access.region)
        for family,axes,template in templates_by_allocation[allocation.allocation_id]:
            view = views[template.view_id]
            target_id=view.allocation_id if template.indirect_target is None else template.indirect_target.allocation_id
            if target_id != allocation.allocation_id:continue
            bound=limits[family.tile_program_id] if family.tile_program_id in limits else {}
            dynamic=any(axes[name].value_limit_access_id is not None for name in template.iteration_axes)
            count = _tile_limited_requested_bytes(views,axes,family,template,bound)
            if dynamic and value_limits is not None and count==0:continue
            conditional=template.predicate is not None or dynamic and value_limits is None
            if template.operation is Operation.READ:
                reads += count
                if conditional:conditional_reads+=count
            else:
                writes += count
                if conditional:conditional_writes+=count
            selected_region=(Strided(view.offset_bytes,tuple(bound[name] if name in bound else size
                for size,name in zip(view.shape,template.view_axes,strict=True)),view.strides_bytes,view.element_bytes)
                if bound and isinstance(view,TensorView) else view.region())
            regions.append(Unmapped(view.shape,view.dtype,view.element_bytes,family.tile_program_id)
                if template.unmapped or template.indirect_target is not None else selected_region)
        unique = None if conditional_reads or conditional_writes else _unique_region_bytes(regions)
        result.append(AccessStatistics(allocation.allocation_id, reads, writes, unique,conditional_reads,conditional_writes))
    return tuple(result)


def _unique_region_bytes(regions):
    if any(isinstance(region, (Indirect, Unmapped)) for region in regions):
        return None
    # Discard provably contained regions, including exact repeated aliases.
    remaining = []
    for region in regions:
        if any(_symbolically_covers(other, region) for other in remaining):
            continue
        remaining = [other for other in remaining
                     if not _symbolically_covers(region, other)]
        remaining.append(region)
    if any(isinstance(region, Indexed) for region in remaining):
        # Requested count does not establish distinct indices or page identity.
        return None
    separated = sorted(remaining, key=lambda region: _footprint_geometry(region)[0])
    if all(region_end(left) <= _footprint_geometry(right)[0]
           for left, right in zip(separated, separated[1:])):
        total = 0
        for region in separated:
            _, width, dims = _footprint_geometry(region)
            if not _separated_geometry(width, dims):
                break
            total += width * prod(count for _, count in dims)
        else:
            return total
    try:
        intervals = merge_intervals(interval for region in remaining
                                    for interval in region_intervals(region))
    except NotImplementedError:
        return None
    return sum(end - start for start, end in intervals)
