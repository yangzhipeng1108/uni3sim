"""Untimed production memory access generation and explicit object bindings."""
from __future__ import annotations

from src.kernel.prediction_cache import runtime_lru_cache

from dataclasses import dataclass
from functools import cached_property, lru_cache
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from collections.abc import Mapping
from dataclasses import fields,is_dataclass
from types import MappingProxyType
from enum import Enum
from operator import attrgetter

from src.kernel.memory_access import (
    AccessScope, AddressLevel, Coverage, EventKind, EventRef, Granularity,
    MemoryAccess, MemoryAccessProgram, MemoryAllocation, Operation, Ordering,
    ProgramEvent, Semantics, Span, Strided, TensorView, LogicalSelectionView, Typed, Unmapped, unique_by,
    TileAxis,TileDependency,TileAccessTemplate,TileProgram,_validate_record_schema,
)

from src.kernel.memory_cache_update import (
    CacheUpdateBindings, build_paged_kv_update, has_paged_kv_source, standalone_cache_update_bindings,
    has_target_cache_source,standalone_target_cache_bindings,build_target_cache_source,target_cache_source_demands,
    has_tensor_only_kv_source,standalone_tensor_kv_bindings,build_tensor_kv_source,tensor_kv_source_work,
)

from src.kernel.memory_eager_attention import (
    EagerAttentionBindings, build_eager_attention, has_eager_attention_source, standalone_eager_attention_bindings,
)

from src.kernel.memory_flash_attention import (
    has_fa3_prefill_source,standalone_fa3_prefill_bindings,build_fa3_prefill_source,
    has_fa3_combine_source,standalone_fa3_combine_bindings,build_fa3_combine_source,
    has_cann_tnd_source,standalone_cann_tnd_bindings,build_cann_tnd_source,
    has_metax_varlen_source,standalone_metax_varlen_bindings,build_metax_varlen_source,
    has_dense_flash_attention_source, standalone_dense_attention_bindings, build_dense_attention_interface,
    has_fa2_split_source,standalone_fa2_split_bindings,build_fa2_split_source,
    has_cann_attention_source,standalone_cann_attention_bindings,build_cann_attention_source,
    has_metax_mla_source,standalone_metax_mla_bindings,build_metax_mla_source,
    has_grouped_mla_source,standalone_grouped_mla_bindings,build_grouped_mla_source,
)

from src.kernel.memory_normalization import (
    has_gpu_normalization_source, standalone_normalization_bindings, build_gpu_normalization,
    has_pim_normalization_source,build_pim_normalization,
    has_public_normalization_source,standalone_public_normalization_bindings,build_public_normalization,
    has_layernorm_source,standalone_layernorm_bindings,build_layernorm_source,
    has_pairnorm_source,standalone_pairnorm_bindings,build_pairnorm_source,
)

from src.kernel.memory_communication import (
    has_communication_source, standalone_communication_bindings, build_communication_interface,
)

from src.kernel.memory_block_fp8 import (
    has_block_fp8_source, standalone_block_fp8_bindings, build_block_fp8_source,
    has_fp8_group_quant_source, standalone_fp8_group_quant_bindings, build_fp8_group_quant_source,
)


from src.kernel.memory_unfused_moe import (
    has_unfused_moe_source, standalone_unfused_moe_bindings, build_unfused_moe_source,
)


from src.kernel.memory_sparse_indexer import (
    SparseIndexerBindings, has_sparse_indexer_source, standalone_sparse_indexer_bindings, build_sparse_indexer_source,
)


from src.kernel.memory_sparse_attention import (
    SparseAttentionBindings, has_sparse_attention_source, standalone_sparse_attention_bindings, build_sparse_attention_source,
)
from src.kernel.memory_fused_moe import (
    FusedMoEBindings, has_fused_moe_source, standalone_fused_moe_bindings, build_fused_moe_source,
)
from src.kernel.memory_fused_moe import has_moe_count_sort_source,standalone_moe_count_sort_bindings,build_moe_count_sort_source
from src.kernel.memory_fused_moe import has_moe_alignment_source,standalone_moe_alignment_bindings,build_moe_alignment_source
from src.kernel.memory_fused_moe_fp8 import has_fp8_fused_moe_source, standalone_fp8_fused_bindings, build_fp8_fused_source
from src.kernel.memory_routed_moe import has_routed_moe_source, standalone_routed_moe_bindings, build_routed_moe_source
from src.kernel.memory_cann_gemm import has_cann_gemm_source, standalone_cann_gemm_bindings, build_cann_gemm_source
from src.kernel.memory_cann_gemm import has_cann_single_core_gemm_source,standalone_cann_single_core_bindings,build_cann_single_core_source
from src.kernel.memory_cann_gemm import has_cann_deterministic_gemm_source,standalone_cann_deterministic_bindings,build_cann_deterministic_source
from src.kernel.memory_cann_normalization import (
    has_cann_normalization_source, standalone_cann_normalization_bindings, build_cann_normalization_source,
)
from src.kernel.memory_rotary import has_gpu_rotary_source,standalone_rotary_bindings,build_gpu_rotary_source
from src.kernel.memory_rotary import has_tensor_only_rotary_source,standalone_tensor_rotary_bindings,build_tensor_rotary_source
from src.kernel.memory_cann_rotary import has_cann_rotary_source,standalone_cann_rotary_bindings,build_cann_rotary_source
from src.kernel.memory_moe_pointwise import has_moe_pointwise_source,standalone_moe_pointwise_bindings,build_moe_pointwise_source
from src.kernel.memory_moe_reduction import has_moe_reduction_source,standalone_moe_reduction_bindings,build_moe_reduction_source
from src.kernel.memory_moe_reduction import has_route_weight_source,standalone_route_weight_bindings,build_route_weight_source
from src.kernel.memory_moe_routing import (has_gpu_topk_source,standalone_gpu_topk_bindings,build_gpu_topk_source,
    has_moe_metadata_source,standalone_moe_metadata_bindings,build_moe_metadata_source)
from src.kernel.memory_mhc import has_mhc_source,standalone_mhc_bindings,build_mhc_source
from src.kernel.memory_reduction import has_generic_reduction_source,standalone_generic_reduction_bindings,build_generic_reduction_source
from src.kernel.memory_fused_rotary import (has_v32_fused_q_source,standalone_v32_fused_q_bindings,build_v32_fused_q_source,
    has_v32_fused_norm_source,standalone_v32_fused_norm_bindings,build_v32_fused_norm_source)
from src.kernel.memory_router import has_simt_router_source,has_simt_router_abi,has_router_replay_mapping,standalone_simt_router_bindings,build_simt_router_source
from src.kernel.memory_router import has_observed_router_abi,standalone_observed_router_bindings,build_observed_router_source,observed_router_source_work
from src.kernel.memory_router import has_public_router_abi,standalone_public_router_bindings,build_public_router_source,public_router_source_work
from src.kernel.memory_router import has_observed_router_reducer_abi,standalone_observed_router_reducer_bindings,build_observed_router_reducer_source,observed_router_reducer_source_work
from src.kernel.memory_moe_fill import has_moe_fill_source,standalone_moe_fill_bindings,build_moe_fill_source
from src.kernel.memory_cann_moe_finalize import has_cann_moe_finalize_source,standalone_cann_moe_finalize_bindings,build_cann_moe_finalize_source
from src.kernel.memory_cann_moe_routing import has_cann_moe_gating_source,standalone_cann_moe_gating_bindings,build_cann_moe_gating_source,has_cann_moe_expert_cast_source
from src.kernel.memory_cann_moe_routing import has_cann_moe_unbound_cast_source,standalone_cann_unbound_cast_bindings,build_cann_unbound_cast_source
from src.kernel.memory_cann_moe_routing import has_cann_moe_init_source,standalone_cann_moe_init_bindings,build_cann_moe_init_source
from src.kernel.memory_cann_moe_routing import has_cann_moe_transpose_source,cann_moe_transpose_source_demands
from src.kernel.memory_cann_moe_routing import is_cann_transpose_alias,standalone_cann_transpose_alias_bindings,build_cann_transpose_alias_source
from src.kernel.memory_cann_moe_routing import has_cann_moe_range_source,standalone_cann_moe_range_bindings,build_cann_moe_range_source
from src.kernel.memory_cann_moe_routing import has_cann_expert_tokens_source,standalone_cann_expert_tokens_bindings,build_cann_expert_tokens_source
from src.kernel.memory_gpu_split_gemm import has_gpu_split_gemm_source,standalone_gpu_split_gemm_bindings,build_gpu_split_gemm_source


@dataclass(frozen=True)
class GEMMWork(Typed):
    m: int
    n: int
    k: int
    dtype: str
    element_bytes: int
    trans_a: bool = False
    trans_b: bool = False
    output_dtype: str | None = None
    batch_count: int = 1
    batched: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.batch_count == 0 or (not self.batched and self.batch_count != 1):
            raise ValueError("GEMM batch count requires an explicit batched ABI")
        if min(self.m, self.n, self.k) == 0:
            raise ValueError("GEMM dimensions must be positive")
        if (self.dtype, self.element_bytes) not in (("bf16", 2), ("fp16", 2), ("fp32", 4), ("int8", 1)):
            raise NotImplementedError("logical GEMM requires homogeneous unquantized input operands")
        if self.output_dtype not in (None, "bf16", "fp16", "fp32", "int8"):
            raise NotImplementedError("logical GEMM output requires an explicit unquantized dtype")

    @property
    def c_dtype(self):
        return self.dtype if self.output_dtype is None else self.output_dtype

    @property
    def c_element_bytes(self):
        from src.kernel.kernel_spec import DTYPE_BYTES
        return DTYPE_BYTES[self.c_dtype]


@dataclass(frozen=True)
class MemoryBindings(Typed):
    gemm_work: GEMMWork
    binding_id: str
    call_id: str
    allocations: tuple[MemoryAllocation, ...]
    views: tuple[TensorView, ...]
    events: tuple[ProgramEvent, ...]
    a_view: str
    b_view: str
    c_view: str
    a_ready: EventRef
    b_ready: EventRef
    rank: int
    stream: str
    visibility_domain: str


def _work_quantity(value):
    """Keep typed immutable coordinates without expanding integer vectors."""
    kind=type(value)
    if kind in (str,int,bool,type(None)):return value,(kind,value)
    if kind is float:return value,(kind,value.hex())
    if isinstance(value,Enum):return value,(kind,_immutable_work_value(value.value))
    if isinstance(value,Mapping):
        if any(type(key) is not str for key in value):raise TypeError('Source quantity names must be strings')
        items=tuple((key,_work_quantity(value[key])) for key in sorted(value))
        return MappingProxyType({key:item for key,(item,_) in items}),(Mapping,tuple((key,key_value) for key,(_,key_value) in items))
    if is_dataclass(value) and not isinstance(value,type):
        if not kind.__dataclass_params__.frozen:raise TypeError('Source records must be immutable')
        items=tuple((field.name,_work_quantity(getattr(value,field.name))) for field in fields(value))
        if any(item is not getattr(value,name) for name,(item,_) in items):
            raise TypeError('Frozen source records cannot contain mutable quantities')
        return value,(kind,tuple((name,key) for name,(_,key) in items))
    if kind is not tuple:raise TypeError('Analytical source quantities must be immutable values')
    if not value:return value,((tuple,()),value)
    if all(type(item) is int for item in value):return value,((tuple,int),value)
    if all(type(row) is tuple and all(type(item) is int for item in row) for row in value):
        return value,((tuple,tuple,int),value)
    items=tuple(_work_quantity(item) for item in value)
    normalized=value if all(item is original for (item,_),original in zip(items,value,strict=True)) else tuple(item for item,_ in items)
    return normalized,(tuple,tuple(key for _,key in items))


def _immutable_work_value(value):return _work_quantity(value)[1]


@dataclass(frozen=True,eq=False)
class MemoryAccessWork(Mapping):
    """An access rule bound to actual operands, without an event graph.

    Static access templates are shared with explicit export. Axis values and
    operand storage are invocation facts; requested-byte projection uses the
    existing tile-rule arithmetic, including repeated reads and tails.
    """
    invocation: object
    axes: tuple[TileAxis, ...]
    accesses: tuple[TileAccessTemplate, ...]
    operation_ids: tuple[str, ...]
    causal_prefixes: tuple = ()
    logical_rows: int | None = None
    source_kernel: object = None
    source_hardware: object = None
    source_evaluator: object = None
    source_with_hardware: bool = False

    @property
    def is_source_projection(self):return self.source_evaluator is not None

    def __post_init__(self):
        if self.is_source_projection:
            from src.kernel.kernel_spec import KernelSpec
            from src.kernel.hardware_spec import HWSpec
            from types import FunctionType
            if (not isinstance(self.source_kernel,KernelSpec) or self.source_hardware is not None
                    and not isinstance(self.source_hardware,HWSpec)
                    or not isinstance(self.source_evaluator,FunctionType)
                    or type(self.source_with_hardware) is not bool
                    or any(type(value) is not tuple or value for value in (
                        self.axes,self.accesses,self.operation_ids,self.causal_prefixes))
                    or self.logical_rows is not None):
                raise TypeError('Analytical work requires its exact source boundary')
            arguments=(self.source_kernel,self.source_hardware,self.invocation) if self.source_with_hardware else (
                self.source_kernel,self.invocation)
            values=self.source_evaluator(*arguments)
            if type(values) is not dict or any(type(key) is not str for key in values):
                raise TypeError('Analytical source must publish named physical quantities')
            ordered=tuple(sorted(values))
            prepared=tuple((key,_work_quantity(values[key])) for key in ordered)
            quantities=MappingProxyType({key:value for key,(value,_) in prepared})
            typed=tuple((key,key_value) for key,(_,key_value) in prepared)
            snapshot=tuple((field.name,getattr(self.invocation,field.name)) for field in fields(self.invocation))
            object.__setattr__(self,'_source_values',quantities)
            object.__setattr__(self,'_source_key',typed)
            object.__setattr__(self,'_source_snapshot',snapshot)
            object.__setattr__(self,'_source_code',self.source_evaluator.__code__)
            object.__setattr__(self,'_source_seal',tuple((field.name,getattr(self,field.name)) for field in fields(self))+
                (('_source_values',quantities),('_source_key',typed),('_source_snapshot',snapshot)))
            return
        logical=isinstance(self.invocation,MemoryInvocation) and self.invocation.logical_hardware is not None
        if (self.source_kernel is not None and not logical or self.source_hardware is not None
                or self.source_with_hardware is not False):
            raise ValueError('Tile access work cannot acquire an undeclared source projection')
        if logical and self.source_kernel is not None:
            self.invocation.assert_logical_source(self.source_kernel)
        if not isinstance(self.invocation,MemoryInvocation):raise TypeError('Memory work requires its checked invocation')
        from src.kernel.memory_access import check_type
        check_type(self.axes,tuple[TileAxis,...]);check_type(self.accesses,tuple[TileAccessTemplate,...])
        check_type(self.operation_ids,tuple[str,...]);check_type(self.logical_rows,int|None)
        if type(self.axes) is not tuple or type(self.accesses) is not tuple or self.causal_prefixes!=():
            raise TypeError('Memory work requires its immutable declared rule')
        axes,patterns,operations=self.invocation.access_rule(self.logical_rows,source_kernel=self.source_kernel)
        if (tuple((a.axis_id,a.extent,a.block) for a in self.axes)!=axes or self.accesses!=patterns
                or self.operation_ids!=operations or any(a.value_limit_access_id is not None
                    or a.value_limit_start!=0 or a.value_limit_inclusive or a.value_limit_upper_bound is not None for a in self.axes)):
            raise ValueError('Memory work differs from its source access rule')

    def consume_source(self,kernel,hardware,evaluator):
        self._assert_source_unchanged()
        if (not self.is_source_projection or kernel!=self.source_kernel
                or hardware is not self.source_hardware or evaluator is not self.source_evaluator
                or evaluator.__code__ is not self._source_code
                or any(getattr(self.invocation,name) is not value for name,value in self._source_snapshot)):
            raise ValueError('Analytical work crossed or changed its actual source boundary')
        return self

    def _assert_source_unchanged(self):
        if (not self.is_source_projection
                or any(getattr(self,name) is not value for name,value in self._source_seal)
                or any(getattr(self.invocation,name) is not value for name,value in self._source_snapshot)):
            raise ValueError('Immutable analytical source work was overwritten')

    def numeric_key(self,**replacements):
        if not self.is_source_projection:raise TypeError('Tile work has no numeric source projection')
        self._assert_source_unchanged()
        if not replacements:return self._source_key
        if not set(replacements)<=set(self):raise ValueError('Unknown analytical source quantity')
        return tuple((key,_immutable_work_value(replacements[key]) if key in replacements else typed)
            for key,typed in self._source_key)

    def __getitem__(self,key):
        if not self.is_source_projection:raise TypeError('Tile work is consumed through its access rules')
        return self._source_values[key]

    def __iter__(self):
        if not self.is_source_projection:raise TypeError('Tile work is consumed through its access rules')
        return iter(self._source_values)

    def __len__(self):
        if not self.is_source_projection:raise TypeError('Tile work is consumed through its access rules')
        return len(self._source_values)

    def __bool__(self):return True

    def __eq__(self,other):
        if self.is_source_projection:
            if isinstance(other,MemoryAccessWork) and not other.is_source_projection:return False
            return Mapping.__eq__(self,other)
        if type(other) is not type(self) or other.is_source_projection:return NotImplemented
        return all(getattr(self,field.name)==getattr(other,field.name) for field in fields(self))

    @cached_property
    def views(self):
        if self.is_source_projection or self.invocation.logical_hardware is not None:return self.invocation.views
        if not isinstance(self.invocation.operand_source,GEMMWork):
            views,_=_scalar_coordinate_views(self.invocation.operand_source,self.invocation.views,
                self.invocation.read_views,self.invocation.write_views)
            return views
        if self.logical_rows is None:return self.invocation.views
        from dataclasses import replace
        return tuple(replace(view,shape=(view.shape[0],self.logical_rows,view.shape[2]))
            if view.view_id in ('A','C') else view for view in self.invocation.views)

    @property
    def allocations(self):return self.invocation.allocations

    @property
    def hardware_id(self):
        return self.source_kernel.hardware_id if self.is_source_projection else self.invocation.kernel.hardware_id

    @property
    def call_id(self):return self.invocation.call_id

    def requested_bytes(self,operation,view_id=None):
        if self.is_source_projection:raise NotImplementedError('Source price projection is not an exported request stream')
        from src.kernel.memory_access import _tile_template_requested_bytes
        if not isinstance(operation,Operation):raise TypeError('Memory work requires its typed direction')
        views={view.view_id:view for view in self.views};axes={axis.axis_id:axis for axis in self.axes}
        selected=tuple(a for a in self.accesses if a.operation is operation
            and (view_id is None or a.view_id==view_id))
        if view_id is not None and not selected:raise ValueError('Operand has no requests in this direction')
        return sum(_tile_template_requested_bytes(views,axes,self,access) for access in selected)

    def unique_bytes(self,operation,view_id=None):
        from src.kernel.memory_access import _unique_region_bytes
        self.requested_bytes(operation,view_id)
        views={view.view_id:view for view in self.views};regions={}
        for access in self.accesses:
            if access.operation is not operation or view_id is not None and access.view_id!=view_id:continue
            view=views[access.view_id]
            if view.allocation_id not in regions:regions[view.allocation_id]=[]
            regions[view.allocation_id].append(view.region())
        counts=tuple(_unique_region_bytes(parts) for parts in regions.values())
        if any(count is None for count in counts):raise NotImplementedError('Unresolved operand selection footprint')
        return sum(counts)


@lru_cache(maxsize=8)
def _gemm_access_patterns(trans_a,trans_b,batched):
    prefix=('batch',) if batched else ()
    reads=(*prefix,'m','n','k');outputs=(*prefix,'m','n')
    return (
        TileAccessTemplate('a','A',Operation.READ,reads,(*prefix,*(('k','m') if trans_a else ('m','k'))),(),(),()),
        TileAccessTemplate('b','B',Operation.READ,reads,(*prefix,*(('n','k') if trans_b else ('k','n'))),(),(),()),
        TileAccessTemplate('c','C',Operation.WRITE,outputs,(*prefix,'m','n'),(),
            (TileDependency('a',EventKind.READ_READY,('k',)),TileDependency('b',EventKind.READ_READY,('k',))),()))


def _gemm_source_tile(kernel,work):
    if kernel.launch.tiling is not None and {'m','n','k'}<=set(kernel.launch.tiling.tile_shape):
        return tuple(int(kernel.launch.tiling.tile_shape[name]) for name in ('m','n','k'))
    return (1,1,work.k)


def build_logical_gemm(work: GEMMWork, hardware_id: str, bindings: MemoryBindings,
                       *, program_id: str, granularity: Granularity,
                       tile: tuple[int, int, int] | None):
    """Logical row-strided C=A@B with explicit caller-selected logical tiling.

    Tile is an algorithmic export decomposition, never a vendor launch claim.
    Inputs may alias. Output/input overlap needs an in-place algorithm and is
    rejected here. Tails retain exact logical extents, with no invented padding.
    """
    if not isinstance(work, GEMMWork) or not isinstance(bindings, MemoryBindings):
        raise TypeError("typed GEMM work and bindings required")
    if bindings.gemm_work != work:
        raise ValueError("binding logical GEMM work mismatch")
    if work.batched:
        return build_logical_batched_gemm(work, hardware_id, bindings,
            program_id=program_id, granularity=granularity, tile=tile)
    from src.kernel.memory_access import check_type, region_intervals
    check_type(granularity, Granularity)
    views = unique_by(bindings.views, "view_id")
    if any(name not in views for name in (bindings.a_view, bindings.b_view, bindings.c_view)):
        raise ValueError("operand view missing")
    a, b, c = (views[name] for name in (bindings.a_view, bindings.b_view, bindings.c_view))
    for view, shape, dtype, width in (
            (a, (work.k, work.m) if work.trans_a else (work.m, work.k), work.dtype, work.element_bytes),
            (b, (work.n, work.k) if work.trans_b else (work.k, work.n), work.dtype, work.element_bytes),
            (c, (work.m, work.n), work.c_dtype, work.c_element_bytes)):
        if (view.shape != shape or view.dtype != dtype or view.element_bytes != width
                or view.strides_bytes[1] != width
                or view.strides_bytes[0] < shape[1] * width):
            raise NotImplementedError("GEMM requires nonoverlapping row-major operand views")
    for operand in (a, b):
        if operand.allocation_id == c.allocation_id and any(
                max(x, p) < min(y, q) for x, y in region_intervals(operand.region())
                for p, q in region_intervals(c.region())):
            raise NotImplementedError("in-place GEMM is not supported by this logical algorithm")
    if granularity is Granularity.TILE:
        check_type(tile, tuple[int, int, int])
        if min(tile) == 0:
            raise ValueError("tile sizes must be positive")
        tm, tn, tk = tile
    elif granularity in (Granularity.OP, Granularity.KERNEL):
        if tile is not None:
            raise ValueError("coarse logical export cannot silently reuse tile traffic")
        tm, tn, tk = work.m, work.n, work.k
    else:
        raise ValueError("unknown granularity")
    if (((work.m + tm - 1) // tm) * ((work.n + tn - 1) // tn)
            * ((work.k + tk - 1) // tk) > 100_000):
        raise NotImplementedError("explicit logical DAG exceeds export budget")
    scopes = []
    accesses = []
    events = list(bindings.events)
    ref = lambda name, kind: EventRef(name, kind)
    allocation_c = unique_by(bindings.allocations, "allocation_id")[c.allocation_id]
    for mi in range(0, work.m, tm):
        for ni in range(0, work.n, tn):
            previous = None
            tile_id = f"{program_id}.m{mi}.n{ni}"
            for ki in range(0, work.k, tk):
                scope_id = f"{tile_id}.k{ki}"
                scopes.append(AccessScope(scope_id, ("gemm",), bindings.call_id,
                                          (mi, ni, ki) if granularity is Granularity.TILE else (),
                                          bindings.rank, bindings.stream, tile_id))
                read_ids = []
                for label, view, i, j, rows, cols, ready, transpose in (
                        ("a", a, mi, ki, min(tm, work.m-mi), min(tk, work.k-ki), bindings.a_ready, work.trans_a),
                        ("b", b, ki, ni, min(tk, work.k-ki), min(tn, work.n-ni), bindings.b_ready, work.trans_b)):
                    if transpose:
                        i, j, rows, cols = j, i, cols, rows
                    read_id = f"{scope_id}.read_{label}"
                    region = Strided(view.offset_bytes + i*view.strides_bytes[0] + j*view.strides_bytes[1],
                                     (rows, cols), view.strides_bytes, view.element_bytes)
                    accesses.append(MemoryAccess(read_id, scope_id, view.allocation_id, view.view_id,
                        Operation.READ, region, Semantics.LOGICAL, None, None, None, None,
                        bindings.visibility_domain, (ready,), EventKind.READ_READY, 1,
                        ("logical:C=A@B;row-major;on-chip-accumulator",)))
                    read_ids.append(read_id)
                compute = f"{scope_id}.compute"
                deps = tuple(ref(name, EventKind.READ_READY) for name in read_ids)
                if previous is not None:
                    deps += (ref(previous, EventKind.COMPUTE_DONE),)
                events.append(ProgramEvent(compute, EventKind.COMPUTE_DONE, None, None, None, deps))
                previous = compute
            write_id = f"{tile_id}.write_c"
            region = Strided(c.offset_bytes + mi*c.strides_bytes[0] + ni*c.strides_bytes[1],
                             (min(tm, work.m-mi), min(tn, work.n-ni)), c.strides_bytes, c.element_bytes)
            accesses.append(MemoryAccess(write_id, scope_id, c.allocation_id, c.view_id,
                Operation.WRITE, region, Semantics.LOGICAL, None, None, None, None,
                bindings.visibility_domain,
                (ref(previous, EventKind.COMPUTE_DONE), ref(allocation_c.allocated_event, EventKind.ALLOCATED)),
                EventKind.WRITE_VISIBLE, 1, ("logical:C=A@B;write-once",)))
    return MemoryAccessProgram("midlayer.memory-access.v1", program_id, hardware_id,
        bindings.binding_id, Coverage(granularity, Semantics.LOGICAL, Ordering.PARTIAL, True,
        ("Logical algorithm only; no vendor instruction, issue schedule or cache traffic claim.",
         "Compute events are untimed dependencies requiring an execution coordinator.")),
        bindings.allocations, bindings.views, tuple(scopes), tuple(events), tuple(accesses))


def build_tiled_gemm(work,hardware_id,bindings, *, program_id,tile,tile_source="caller-declared logical partition"):
    """One compressed logical GEMM loop with exact tails and K reduction.

    No matrix-size tile DAG is allocated. Native launch mapping is a separate
    evidence claim; axis blocks come from the actual selected launch or an
    explicitly requested logical decomposition.
    """
    if bindings.gemm_work != work or len(tile) != 3 or min(tile) <= 0:
        raise ValueError('tile GEMM requires matching work/bindings and positive M/N/K blocks')
    views=unique_by(bindings.views,'view_id')
    a,b,c=(views[name] for name in (bindings.a_view,bindings.b_view,bindings.c_view))
    expected=((work.k,work.m) if work.trans_a else (work.m,work.k),
        (work.n,work.k) if work.trans_b else (work.k,work.n),(work.m,work.n))
    if work.batched:expected=tuple((work.batch_count,*shape) for shape in expected)
    for view,shape,width,dtype in zip((a,b,c),expected,(work.element_bytes,work.element_bytes,work.c_element_bytes),
            (work.dtype,work.dtype,work.c_dtype),strict=True):
        strides=[];stride=width
        for extent in reversed(shape):strides.insert(0,stride);stride*=extent
        dimensions=tuple(zip(view.strides_bytes,shape,strict=True))
        if work.batched and view is a:
            # q_nope.transpose(0,1) interleaves heads inside token rows.
            # Validate disjoint storage dimensions in physical stride order;
            # logical batch-major order need not be physical storage order.
            dimensions=tuple(sorted(dimensions))
            nonoverlap=all(right>=stride*extent for (stride,extent),(right,_) in zip(dimensions,dimensions[1:]))
        else:
            nonoverlap=all(left>=dimension*right for left,dimension,right in zip(
                view.strides_bytes[:-1],view.shape[1:],view.strides_bytes[1:],strict=True))
        if ((view.shape,view.element_bytes,view.dtype) != (shape,width,dtype) or view.strides_bytes[-1] != width
                or not nonoverlap):
            raise ValueError('tile GEMM requires its actual nonoverlapping row-strided typed operand views')
    if c.allocation_id in {a.allocation_id,b.allocation_id}:
        raise NotImplementedError('tile GEMM alias requires a declared in-place algorithm')
    axes=tuple(TileAxis(name,extent,block) for name,extent,block in zip(('m','n','k'),(work.m,work.n,work.k),tile,strict=True))
    prefix=('batch',) if work.batched else ()
    if work.batched:axes=(TileAxis('batch',work.batch_count,1),*axes)
    evidence=(f'Logical C=A@B; {tile_source} partitions exact operands, including tails.',
        'Each output tile accumulates K blocks on chip and writes C once; independent output tiles have no memory-order dependency.')
    reads=(*prefix,'m','n','k');outputs=(*prefix,'m','n')
    from dataclasses import replace
    patterns=_gemm_access_patterns(work.trans_a,work.trans_b,work.batched)
    accesses=tuple(replace(pattern,view_id=view.view_id,evidence_refs=evidence,
        input_ready=(ready,) if ready is not None else ())
        for pattern,view,ready in zip(patterns,(a,b,c),(bindings.a_ready,bindings.b_ready,None),strict=True))
    scope=AccessScope(program_id,('gemm',),bindings.call_id,(),bindings.rank,bindings.stream,'logical-gemm-tile-loop')
    family=TileProgram(program_id,scope,axes,accesses,bindings.visibility_domain,evidence)
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,True,(
            'Compressed algorithmic tiles; tile axes do not assert native CTA issue order, cache hits or physical transactions.',
            'K reduction dependencies refer to actual operand tile reads; the accumulator is on chip, not a fabricated GM scratch tensor.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))


def build_logical_batched_gemm(work, hardware_id, bindings, *, program_id, granularity, tile):
    """C[b]=A[b]@B[b], with distinct strided batches and on-chip accumulation."""
    if tile is not None or granularity not in (Granularity.OP, Granularity.KERNEL):
        raise NotImplementedError("batched GEMM source currently exposes coarse logical access")
    views = unique_by(bindings.views, "view_id")
    a, b, c = (views[name] for name in (bindings.a_view, bindings.b_view, bindings.c_view))
    shapes = ((work.batch_count, work.k, work.m) if work.trans_a else (work.batch_count, work.m, work.k),
              (work.batch_count, work.n, work.k) if work.trans_b else (work.batch_count, work.k, work.n),
              (work.batch_count, work.m, work.n))
    for view, shape, dtype, width in zip((a, b, c), shapes,
            (work.dtype, work.dtype, work.c_dtype), (work.element_bytes, work.element_bytes, work.c_element_bytes)):
        strides = (shape[1] * shape[2] * width, shape[2] * width, width)
        if (view.shape, view.dtype, view.element_bytes, view.strides_bytes) != (shape, dtype, width, strides):
            raise ValueError("batched GEMM requires its declared dense strided batch layout")
    if c.allocation_id in (a.allocation_id, b.allocation_id):
        raise NotImplementedError("batched GEMM in-place output is not a declared algorithm")
    scope_id = f"{program_id}.batches"
    scope = AccessScope(scope_id, ("gemm", "batched"), bindings.call_id, (),
        bindings.rank, bindings.stream, scope_id)
    accesses = []
    for label, view, ready in (("a", a, bindings.a_ready), ("b", b, bindings.b_ready)):
        accesses.append(MemoryAccess(f"{scope_id}.read_{label}", scope_id, view.allocation_id, view.view_id,
            Operation.READ, view.region(), Semantics.LOGICAL, None, None, None, None,
            bindings.visibility_domain, (ready,), EventKind.READ_READY, 1,
            ("logical:C[batch]=A[batch]@B[batch];distinct-weight-batches",)))
    compute_id = f"{scope_id}.compute"
    compute = ProgramEvent(compute_id, EventKind.COMPUTE_DONE, None, None, None,
        tuple(EventRef(access.access_id, EventKind.READ_READY) for access in accesses))
    allocation_c = unique_by(bindings.allocations, "allocation_id")[c.allocation_id]
    accesses.append(MemoryAccess(f"{scope_id}.write_c", scope_id, c.allocation_id, c.view_id,
        Operation.WRITE, c.region(), Semantics.LOGICAL, None, None, None, None,
        bindings.visibility_domain,
        (EventRef(compute_id, EventKind.COMPUTE_DONE), EventRef(allocation_c.allocated_event, EventKind.ALLOCATED)),
        EventKind.WRITE_VISIBLE, 1, ("logical:C[batch]=A[batch]@B[batch];write-once",)))
    return MemoryAccessProgram("midlayer.memory-access.v1", program_id, hardware_id, bindings.binding_id,
        Coverage(granularity, Semantics.LOGICAL, Ordering.PARTIAL, True,
            ("Dense strided logical batches; no vendor issue/cache/physical transaction claim.",)),
        bindings.allocations, bindings.views, (scope,), (*bindings.events, compute), tuple(accesses))


def standalone_bindings(work: GEMMWork, *, call_id: str, binding_id: str,
                        device_id: str, visibility_domain: str, rank: int = 0, stream: str = "offline"):
    """Explicit independent object-address bindings for offline export only."""
    allocations, views = gemm_operand_records(work,binding_id=binding_id,device_id=device_id)
    events=[]
    for name,allocation in zip(('A','B','C'),allocations,strict=True):
        allocated=allocation.allocated_event
        events.append(ProgramEvent(allocated,EventKind.ALLOCATED,allocation.allocation_id,None,None,()))
        if name!='C':
            events.append(ProgramEvent(f"{binding_id}.{name}.initialized",EventKind.INITIALIZED,
                allocation.allocation_id,Span(0,allocation.size_bytes),visibility_domain,
                (EventRef(allocated,EventKind.ALLOCATED),)))
    return MemoryBindings(work, binding_id, call_id, allocations, views, tuple(events),
        "A", "B", "C", EventRef(f"{binding_id}.A.initialized", EventKind.INITIALIZED),
        EventRef(f"{binding_id}.B.initialized", EventKind.INITIALIZED), rank, stream, visibility_domain)


def gemm_operand_records(work,*,binding_id,device_id,weight_parent=None):
    """One operand layout rule, shared by canonical export and actual binding.

    The optional resolver receives the expected B layout before its allocation
    is created. It must return the real parameter parent, never a shape alias.
    """
    allocations,views=[],[]
    for name, shape in (("A", (work.k, work.m) if work.trans_a else (work.m, work.k)),
                        ("B", (work.n, work.k) if work.trans_b else (work.k, work.n)),
                        ("C", (work.m, work.n))):
        if work.batched:
            shape = (work.batch_count, *shape)
        allocation_id = f"{binding_id}.{name}"
        allocated = f"{allocation_id}.allocated"
        dtype = work.c_dtype if name == "C" else work.dtype
        width = work.c_element_bytes if name == "C" else work.element_bytes
        strides = ((shape[1]*shape[2]*width, shape[2]*width, width) if work.batched
                   else (shape[1]*width, width))
        size=(work.batch_count if work.batched else 1)*shape[-2]*shape[-1]*width
        allocation=(weight_parent(allocation_id,shape,strides,dtype,width,size) if name=='B' and weight_parent is not None
            else MemoryAllocation(allocation_id,device_id,'logical-storage',size,AddressLevel.OBJECT,
                None,None,allocated,None))
        allocations.append(allocation)
        views.append(TensorView(name,allocation.allocation_id,0,shape,strides,dtype,width))
    return tuple(allocations),tuple(views)


@lru_cache(maxsize=16)
def _invocation_record_fields(kind):return tuple(field.name for field in fields(kind))


@lru_cache(maxsize=16)
def _invocation_record_reader(kind):
    names = _invocation_record_fields(kind)
    if len(names) < 2:
        return lambda record: tuple(getattr(record, name) for name in names)
    return attrgetter(*names)


@dataclass(frozen=True)
class MemoryInvocation:
    """Immutable actual operands consumed by analytical source rules.

    A detailed binding/program is an explicit export projection. Normal
    pricing consumes the operand geometry directly, including on a cold call.
    """
    kernel: object
    operand_source: object
    binding_id: str
    call_id: str
    allocations: tuple
    views: tuple
    rank: int
    stream: str
    visibility_domain: str = 'gemm-visible'
    scope_initialized: bool = True
    logical_hardware: object = None

    def validate(self):
        from math import prod
        from src.kernel.memory_access import check_type
        from src.kernel.kernel_spec import KernelSpec,KernelKind
        check_type(self.kernel,KernelSpec)
        check_type(self.scope_initialized,bool)
        if self.logical_hardware is not None:
            _validate_logical_invocation(self)
            return
        if self.kernel.kind is KernelKind.FUSED_MOE:
            for value in (self.binding_id,self.call_id,self.stream,self.visibility_domain):check_type(value,str)
            check_type(self.rank,int)
            from src.kernel.memory_fused_moe import validate_native_moe_invocation
            validate_native_moe_invocation(self)
            return
        if self.kernel.kind is not KernelKind.GEMM:
            self._validate_tensor_operands()
            return
        check_type(self.operand_source,GEMMWork)
        for value in (self.binding_id,self.call_id,self.stream,self.visibility_domain):check_type(value,str)
        check_type(self.rank,int)
        if type(self.allocations) is not tuple or type(self.views) is not tuple:
            raise TypeError('Compact GEMM operands must be immutable tuples')
        if len(self.allocations)!=3 or len(self.views)!=3:
            raise ValueError('Compact GEMM requires its three actual operands')
        work=self.operand_source
        for role,parent,view in zip(('A','B','C'),self.allocations,self.views,strict=True):
            check_type(parent,MemoryAllocation);check_type(view,TensorView)
            shape=((work.k,work.m) if work.trans_a else (work.m,work.k)) if role=='A' else (
                ((work.n,work.k) if work.trans_b else (work.k,work.n)) if role=='B' else (work.m,work.n))
            if work.batched:shape=(work.batch_count,*shape)
            dtype,width=(work.c_dtype,work.c_element_bytes) if role=='C' else (work.dtype,work.element_bytes)
            strides=tuple(prod(shape[axis+1:])*width for axis in range(len(shape)))
            if ((view.view_id,view.allocation_id,view.shape,view.strides_bytes,view.dtype,view.element_bytes,view.offset_bytes)
                    !=(role,parent.allocation_id,shape,strides,dtype,width,0)
                    or parent.size_bytes!=prod(shape)*width or parent.memory_domain!='logical-storage'
                    or parent.address_level is not AddressLevel.OBJECT or parent.base_address_bytes is not None
                    or parent.mapping_ref is not None or parent.freed_event is not None):
                raise ValueError('Compact GEMM actual operand differs from its complete source layout')
        if len({p.device_id for p in self.allocations})!=1 or len({p.allocation_id for p in self.allocations})!=3:
            raise ValueError('Compact GEMM requires distinct same-device actual operands')
        pools=({p.allocation_id for p in self.allocations},{v.view_id for v in self.views},set(self.event_names))
        if len(pools[2])!=5 or any(pools[a]&pools[b] for a in range(3) for b in range(a)):
            raise ValueError('Compact GEMM operand/event identities collide')

    def __post_init__(self):
        self.validate()
        # Native grouped validation already calls every operand's intrinsic
        # checker. Ordinary matrix/scalar validation needs the schema check
        # before those records can acquire an identity seal.
        if self.kernel.kind.value!='FUSED_MOE':
            _validate_record_schema((self.allocations,self.views,
                self.operand_source if isinstance(self.operand_source,GEMMWork) else ()))
        object.__setattr__(self,'_source_validation_epoch',memory_prediction_source_epoch())
        object.__setattr__(self,'_validated_operands',tuple(getattr(self,name) for name in self.__dataclass_fields__))
        self._retain_operand_records()

    def _retain_operand_records(self):
        records=(*self.allocations,*self.views)
        if isinstance(self.operand_source,GEMMWork):
            records=(*records,self.operand_source)
        if self.logical_hardware is not None:
            coordinates=tuple(coordinate for view in self.views for coordinate in view.coordinate_map)
            records=(*records,*coordinates,*(term for coordinate in coordinates for term in coordinate.terms))
            object.__setattr__(self,'_validated_logical_roles',(self.read_views,self.write_views))
        object.__setattr__(self,'_validated_records',tuple((record,_invocation_record_reader(type(record))(record))
            for record in records))
        object.__setattr__(self,'_validated_record_types',tuple(type(record) for record in records))

    def __reduce__(self):
        # Process-local field identities are rebuilt at the restore boundary;
        # integer/scalar pickle encoding need not preserve Python identities.
        return _restore_memory_invocation,tuple(getattr(self,name) for name in self.__dataclass_fields__)

    def assert_boundary_unchanged(self):
        if self._source_validation_epoch != memory_prediction_source_epoch():
            raise ValueError("Memory source recipe changed since binding validation")
        if (any(getattr(self,name) is not value for name,value in zip(self.__dataclass_fields__,self._validated_operands,strict=True))
                or any(name in self.__dict__ for name in ('events','input_ready','a_ready','b_ready',
                    'query_view','weight_view','output_view','routing','query_ready','weight_ready','routing_ready',
                    'formatted_weight','formatted_ready'))):
            raise ValueError('Immutable memory invocation boundary was overwritten')
        if '_validated_logical_roles' in self.__dict__ and any(getattr(self,name) is not value
                for name,value in zip(('read_views','write_views'),self._validated_logical_roles,strict=True)):
            raise ValueError('Immutable logical operand roles were overwritten')
        if '_validated_records' in self.__dict__ and any(actual is not value
                for record,snapshot in self._validated_records
                for actual,value in zip(_invocation_record_reader(type(record))(record),snapshot,strict=True)):
            raise ValueError('Immutable source operand record was overwritten')
        if '_validated_record_types' not in self.__dict__:
            # Older snapshots did not seal ordinary matrix/scalar records.
            # Preserve any older seal above, then revalidate before admitting
            # those operands under the current process-local representation.
            self.validate()
            if self.kernel.kind.value!='FUSED_MOE':
                _validate_record_schema((self.allocations,self.views,
                    self.operand_source if isinstance(self.operand_source,GEMMWork) else ()))
            self._retain_operand_records()
        if any(type(record) is not kind for (record,_),kind in
                zip(self._validated_records,self._validated_record_types,strict=True)):
            raise ValueError('Immutable source operand record type was overwritten')


    def prediction_work(self,kernel,hardware):
        self.assert_boundary_unchanged()
        if self.kernel.kind.value=='FUSED_MOE':
            if kernel!=self.kernel:raise ValueError('Grouped memory invocation crossed its source kernel')
            return fused_moe_prediction_work(kernel,hardware,self)
        context=copy_context()
        prepared=context[_PREDICTION_MEMORY_PROGRAMS][3] if _PREDICTION_MEMORY_PROGRAMS in context else None
        key=(id(self),id(kernel),id(hardware))
        if prepared is not None and key in prepared:
            retained,source,device,work=prepared[key]
            if retained is not self or source is not kernel or device is not hardware:
                raise RuntimeError('Memory work identity collision')
            return work
        if self.logical_hardware is not None:self.assert_logical_source(kernel,hardware)
        elif kernel.hardware_id!=hardware.hardware_id or kernel!=self.kernel:
            raise ValueError('Memory invocation crossed its source kernel/hardware')
        # Normalization already owns its captured row transform in its source
        # rule. Applying the generic vector transform here loses logical rows.
        execution=kernel if self.logical_hardware is not None or has_gpu_normalization_source(kernel,hardware) else memory_execution_kernel(kernel,hardware)
        if self.logical_hardware is not None:
            if hardware is not self.logical_hardware or self.operand_source is not self.kernel:
                raise ValueError('Logical invocation crossed its actual source rule')
        elif isinstance(self.operand_source,GEMMWork):
            if not has_gpu_nonsplit_gemm_source(kernel,hardware) or self.operand_source!=_gemm_work_from_kernel(execution):
                raise ValueError('Memory invocation differs from its matrix source ABI')
        elif not (has_scalar_analytical_work(kernel) or has_gpu_normalization_source(kernel,hardware)) or self.operand_source!=execution:
            raise ValueError('Memory invocation differs from its scalar source ABI')
        source=kernel if self.logical_hardware is not None and kernel is not self.kernel else None
        axes,patterns,operations=self.access_rule(source_kernel=source)
        work=MemoryAccessWork(self,tuple(TileAxis(*axis) for axis in axes),patterns,operations,source_kernel=source)
        if prepared is not None:prepared[key]=(self,kernel,hardware,work)
        return work

    def assert_logical_source(self,kernel,hardware=None):
        """Keep actual operands while admitting the existing numerical spec projection."""
        self.assert_boundary_unchanged()
        if self.logical_hardware is None or hardware is not None and hardware is not self.logical_hardware:
            raise ValueError('Logical work crossed its checked source hardware')
        if kernel is self.kernel:return
        if (kernel.kind,kernel.hardware_id,kernel.variant)!=(self.kernel.kind,self.kernel.hardware_id,self.kernel.variant):
            raise ValueError('Logical work crossed its operation source')
        reads,writes=_logical_interface_layouts(kernel,self.logical_hardware)
        expected=(*reads,*writes)
        if (tuple(role for role,_,_,_ in reads)!=self.read_views or tuple(role for role,_,_,_ in writes)!=self.write_views
                or tuple((view.view_id,view.shape,view.dtype,view.element_bytes) for view in self.views)!=expected):
            raise ValueError('Numerical source projection changed its actual logical operands')

    def access_rule(self,logical_rows=None,*,source_kernel=None):
        work=self.operand_source
        if self.logical_hardware is not None:
            if logical_rows is not None:raise ValueError('Logical selections cannot acquire replay axes')
            return _logical_invocation_rule(self,source_kernel)
        if source_kernel is not None:raise ValueError('Only logical work has a numerical source projection')
        if not isinstance(work,GEMMWork) and work.variant in ('rmsnorm','add_rmsnorm'):
            from src.kernel.memory_normalization import normalization_access_patterns,_normalization_row_extents
            logical,rows=_normalization_row_extents(work)
            if logical_rows is not None:
                if logical_rows!=rows:raise ValueError('Normalization execution projection must retain its actual rows')
                logical=rows
            header=('normalization',) if logical==rows else ('normalization',f'logical_rows={logical}',f'execution_rows={rows}')
            return (('row',rows,1),),normalization_access_patterns(work.variant=='add_rmsnorm'),header
        if not isinstance(work,GEMMWork):
            if logical_rows is not None:raise ValueError('Scalar work cannot acquire matrix replay axes')
            views,write_roles=_scalar_coordinate_views(work,self.views,self.read_views,self.write_views)
            shape=next(view.shape for view in views if view.view_id==(write_roles or self.read_views)[0])
            axes=tuple((f'coordinate{index}',dimension,1) for index,dimension in enumerate(shape))
            patterns=_scalar_access_patterns(tuple(a[0] for a in axes),self.read_views,write_roles,
                tuple(len(view.shape) for view in self.views if view.view_id in self.read_views),work.variant=='gemm_reduce')
            header=_gemm_elementwise_header(work) if work.variant=='gemm_elementwise' else (work.variant,)
            return axes,patterns,header
        rows=work.m
        if logical_rows is not None:
            if (not work.batched or logical_rows!=self.kernel.shape['m']
                    or 'model_input_shape' not in self.kernel.attributes
                    or self.kernel.attributes['graph_runtime_mode'] not in ('FULL','PIECEWISE')
                    or self.kernel.attributes['model_input_shape']['m']!=rows or not 0<logical_rows<=rows):
                raise ValueError('Matrix work requires its declared logical replay rows')
            rows=logical_rows
        tile=_gemm_source_tile(self.kernel,work)
        axes=tuple(zip(('m','n','k'),(rows,work.n,work.k),tile,strict=True))
        if work.batched:axes=(('batch',work.batch_count,1),*axes)
        return axes,_gemm_access_patterns(work.trans_a,work.trans_b,work.batched),('gemm',)

    @cached_property
    def event_names(self):
        if self.kernel.kind.value=='FUSED_MOE':
            from src.kernel.memory_fused_moe import native_moe_event_names
            return native_moe_event_names(self)
        if not isinstance(self.operand_source,GEMMWork):
            return tuple(event.event_id for event in self.materialized.events)
        return tuple(name for role,parent in zip(('A','B','C'),self.allocations,strict=True)
            for name in ((parent.allocated_event,) if role=='C' else
                (parent.allocated_event,f'{self.binding_id}.{self.binding_id}.{role}.initialized')))

    @cached_property
    def materialized(self):
        self.validate()
        if self.logical_hardware is not None:
            return _materialize_logical_operands(self.binding_id,self.call_id,self.allocations,self.views,
                self.read_views,self.write_views,self.rank,self.stream,self.visibility_domain)
        if self.kernel.kind.value=='FUSED_MOE':
            from src.kernel.memory_fused_moe import materialize_native_moe_invocation
            return materialize_native_moe_invocation(self)
        if not isinstance(self.operand_source,GEMMWork):return _export_tensor_invocation(self)
        events=[];ready=[]
        for role,parent in zip(('A','B','C'),self.allocations,strict=True):
            events.append(ProgramEvent(parent.allocated_event,EventKind.ALLOCATED,parent.allocation_id,None,None,()))
            if role!='C':
                name=f'{self.binding_id}.{self.binding_id}.{role}.initialized'
                events.append(ProgramEvent(name,EventKind.INITIALIZED,parent.allocation_id,Span(0,parent.size_bytes),
                    self.visibility_domain,(EventRef(parent.allocated_event,EventKind.ALLOCATED),)))
                ready.append(EventRef(name,EventKind.INITIALIZED))
        return MemoryBindings(self.operand_source,self.binding_id,self.call_id,self.allocations,self.views,tuple(events),
            'A','B','C',*ready,self.rank,self.stream,self.visibility_domain)

    @cached_property
    def read_views(self):
        if self.logical_hardware is not None:
            return tuple(role for role,_,_,_ in _logical_interface_layouts(self.kernel,self.logical_hardware)[0])
        if self.kernel.kind.value=='FUSED_MOE':return tuple(view.view_id for view in self.views if view.view_id!='output')
        if self.operand_source.variant in ('rmsnorm','add_rmsnorm'):
            return ('input','residual','weight') if self.operand_source.variant=='add_rmsnorm' else ('input','weight')
        inputs,_=_vector_operand_layouts(self.operand_source)
        return tuple(f'input{i}' for i in range(len(inputs)))

    @cached_property
    def write_views(self):
        if self.logical_hardware is not None:
            return tuple(role for role,_,_,_ in _logical_interface_layouts(self.kernel,self.logical_hardware)[1])
        if self.kernel.kind.value=='FUSED_MOE':return ('output',)
        if self.operand_source.variant in ('rmsnorm','add_rmsnorm'):
            return ('input','residual') if self.operand_source.variant=='add_rmsnorm' else ('output',)
        _,outputs=_vector_operand_layouts(self.operand_source)
        return tuple(f'output{i}' for i in range(len(outputs)))

    def _validate_tensor_operands(self):
        from src.kernel.kernel_spec import KernelSpec
        from src.kernel.memory_access import check_type,region_end
        check_type(self.operand_source,KernelSpec)
        norm=self.kernel.variant in ('rmsnorm','add_rmsnorm') and self.operand_source.variant==self.kernel.variant
        if not norm and (not has_scalar_analytical_work(self.kernel) or not has_scalar_analytical_work(self.operand_source)):
            raise ValueError('Memory invocation requires its declared scalar ABI')
        for value in (self.binding_id,self.call_id,self.stream,self.visibility_domain):check_type(value,str)
        check_type(self.rank,int)
        if type(self.allocations) is not tuple or type(self.views) is not tuple:
            raise TypeError('Memory operands must be immutable tuples')
        for parent in self.allocations:check_type(parent,MemoryAllocation)
        for view in self.views:check_type(view,TensorView|LogicalSelectionView)
        if norm:
            from src.kernel.memory_normalization import validate_normalization_operands
            views,_,_=validate_normalization_operands(self.operand_source,self.views,self.read_views,self.write_views)
            parents=unique_by(self.allocations,'allocation_id')
        else:views,parents=_scalar_operand_bindings(self.operand_source,self)
        roles=(*self.read_views,*self.write_views)
        if set(views)!=set(roles) or {view.allocation_id for view in self.views}!=set(parents):
            raise ValueError('Scalar invocation has unknown operand records')
        if len({p.device_id for p in self.allocations})!=1:
            raise ValueError('Scalar operands crossed their actual device')
        for view in self.views:
            parent=parents[view.allocation_id]
            if parent.freed_event is not None:raise ValueError('Scalar operand storage has ended its lifetime')
            if isinstance(view,TensorView) and parent.size_bytes is not None and region_end(view.region())>parent.size_bytes:
                raise ValueError('Scalar operand exceeds its actual parent capacity')
            if isinstance(view,LogicalSelectionView) and parent.size_bytes is not None:
                raise ValueError('Unresolved scalar selection cannot claim an affine storage capacity')
            if isinstance(view,LogicalSelectionView) and (view.parent_view_id is not None or view.coordinate_map):
                raise NotImplementedError('Direct scalar selection requires its complete standalone logical parent')

    def __getattr__(self,name):
        if name in ('query_view','weight_view','output_view','routing','formatted_weight','formatted_ready',
                'query_ready','weight_ready','routing_ready') and self.kernel.kind.value=='FUSED_MOE':
            if name in ('query_view','weight_view','output_view'):return name.removesuffix('_view')
            if name=='routing':return next(view.allocation_id for view in self.views if view.view_id=='sorted_token_ids')
            if name=='formatted_weight':return None
            if name=='formatted_ready':return ()
            if name in ('query_ready','weight_ready','routing_ready'):
                from src.kernel.memory_fused_moe import native_moe_input_ready
                return native_moe_input_ready(self,'sorted_token_ids' if name=='routing_ready' else name.removesuffix('_ready'))
        if name in ('a_view','b_view','c_view','a_ready','b_ready','events'):
            return getattr(self.materialized,name)
        raise AttributeError(name)

    @property
    def canonical_binding_type(self):
        from src.kernel.kernel_spec import KernelKind
        if self.logical_hardware is not None:return ElementwiseBindings
        if self.kernel.kind is KernelKind.FUSED_MOE:return FusedMoEBindings
        if self.kernel.kind is KernelKind.GEMM:return MemoryBindings
        if self.kernel.kind is KernelKind.VECTOR:return ElementwiseBindings
        raise ValueError('Invocation has no declared source binding type')


_default_source_epoch = None
_source_binding_epoch = None
_default_source_rule = ()
_source_recipe_signature = None
_source_recipe_epoch = None
_SOURCE_RECIPE_SCOPE = ContextVar("midlayer_source_recipe_scope")


def memory_prediction_source_epoch():
    """Return the live Kernel-owned source recipe identity for memo keys.

    The opaque token retains one complete function/method/code signature in
    this owner and has constant-time hashing. It changes conservatively when
    a source/admission rule is replaced or edited. Serving must not recreate
    or introspect Kernel source modules itself.
    """
    context = copy_context()
    if _SOURCE_RECIPE_SCOPE in context:
        return context[_SOURCE_RECIPE_SCOPE]
    return _live_memory_prediction_source_epoch()


@contextmanager
def memory_prediction_source_scope():
    """One synchronous plan owns one checked source implementation snapshot.

    Operand boundaries are still validated per call. Source code is checked
    on entry and exit, rather than rediscovered for every operand/lookup.
    A changed implementation invalidates the whole operation, never a subset.
    """
    if _SOURCE_RECIPE_SCOPE in copy_context():
        yield
        return
    epoch = memory_prediction_source_epoch()
    token = _SOURCE_RECIPE_SCOPE.set(epoch)
    try:
        yield
    finally:
        _SOURCE_RECIPE_SCOPE.reset(token)
        if memory_prediction_source_epoch() is not epoch:
            raise ValueError("Memory source recipe changed during plan prediction")


def _live_memory_prediction_source_epoch():
    import sys
    from importlib import import_module
    from functools import cached_property
    from types import FunctionType

    modules = (
        "src.kernel.memory_access", "src.kernel.memory_access_builder",
        "src.kernel.memory_path_solver", "src.kernel.memory_block_fp8",
        "src.kernel.memory_cache_update", "src.kernel.memory_cann_gemm",
        "src.kernel.memory_cann_mla", "src.kernel.memory_cann_mla_split",
        "src.kernel.memory_cann_moe_finalize",
        "src.kernel.memory_cann_moe_routing",
        "src.kernel.memory_cann_normalization",
        "src.kernel.memory_cann_paged_attention",
        "src.kernel.memory_cann_rotary",
        "src.kernel.memory_communication",
        "src.kernel.memory_eager_attention",
        "src.kernel.memory_flash_attention",
        "src.kernel.memory_fused_moe", "src.kernel.memory_fused_moe_fp8",
        "src.kernel.memory_fused_moe_naive",
        "src.kernel.memory_fused_rotary",
        "src.kernel.memory_gpu_split_gemm",
        "src.kernel.memory_metax_mla_controls",
        "src.kernel.memory_metax_mla_metadata",
        "src.kernel.memory_metax_mla_scratch", "src.kernel.memory_mhc",
        "src.kernel.memory_moe_alignment", "src.kernel.memory_moe_fill",
        "src.kernel.memory_moe_pointwise",
        "src.kernel.memory_moe_reduction",
        "src.kernel.memory_moe_routing",
        "src.kernel.memory_normalization", "src.kernel.memory_reduction",
        "src.kernel.memory_rotary", "src.kernel.memory_routed_moe",
        "src.kernel.memory_router", "src.kernel.memory_sparse_attention",
        "src.kernel.memory_sparse_indexer", "src.kernel.memory_unfused_moe",
        "src.kernel.predictors.vector", "src.kernel.predictors.flash_attn",
    )
    for module_name in modules:
        import_module(module_name)
    signature = []
    for module_name in modules:
        module = sys.modules[module_name]
        for name, value in sorted(vars(module).items()):
            if isinstance(value, FunctionType) and value.__module__ == module_name:
                signature.append((module_name, name, value, value.__code__))
            elif (callable(value) and hasattr(value, "__wrapped__")
                    and isinstance(value.__wrapped__, FunctionType)
                    and value.__wrapped__.__module__ == module_name):
                signature.append((module_name, name, value,
                    value.__wrapped__, value.__wrapped__.__code__))
            elif isinstance(value, type) and value.__module__ == module_name:
                for method_name, method in sorted(vars(value).items()):
                    function = (method.fget if isinstance(method, property) else
                        method.func if isinstance(method, cached_property) else method)
                    if isinstance(function, FunctionType):
                        signature.append((module_name,
                            f"{name}.{method_name}", method, function,
                            function.__code__))
    # The paged-KV builder is imported above but called through this module's
    # binding. Tracking only its defining module misses a replaced dispatch
    # alias and could return an old exact or numeric price without validation.
    if not isinstance(build_paged_kv_update, FunctionType):
        raise TypeError('Paged KV source builder must retain its Python source callable')
    signature.append((__name__, 'build_paged_kv_update', build_paged_kv_update,
                      build_paged_kv_update.__code__))
    global _source_recipe_signature, _source_recipe_epoch
    signature = tuple(signature)
    if signature != _source_recipe_signature:
        _source_recipe_signature = signature
        _source_recipe_epoch = object()
    return _source_recipe_epoch


def _set_source_binding_epoch(epoch):
    """Full source implementation identity, separate from default geometry."""
    global _source_binding_epoch
    _source_binding_epoch = epoch


def _set_default_source_epoch(epoch):
    global _default_source_epoch,_default_source_rule
    if epoch is not _default_source_epoch:
        import sys
        from src.kernel import memory_normalization,memory_cann_moe_routing
        from src.kernel.predictors import vector
        rules=((sys.modules[__name__],(
            'memory_execution_kernel','_memory_execution_kernel','_gemm_work_from_kernel','gemm_operand_records',
            'scalar_operand_records','_vector_operand_layouts','_elementwise_operand_types','_gemm_elementwise_extents')),
            (memory_normalization,('normalization_operand_records','_normalization_row_extents')),
            (memory_cann_moe_routing,('cann_transpose_execution_kernel',)),
            (vector,('_has_graph_padding','_gpu_full_graph_source_program_execution_kernel','_metax_full_graph_norm_execution_kernel',
                '_cann_full_graph_source_program_execution_kernel')))
        _default_source_rule=tuple((module,name,getattr(module,name),getattr(module,name).__code__)
            for module,names in rules for name in names)
    _default_source_epoch = epoch


class DefaultMemoryInvocation(MemoryInvocation):
    """An unchanged default source, before any operand records are requested.

    The caller still resolves real parameters and all non-default storage takes
    the ordinary binding path. A cold source proof uses MemoryInvocation; warm
    calls carry the same checked KernelSpec, rather than recreating its operands.
    """
    def __new__(cls,*args,**kwargs):
        if 'allocations' in kwargs or 'views' in kwargs:return MemoryInvocation(*args,**kwargs)
        return object.__new__(cls)

    def __init__(self,kernel,hardware,binding_id,call_id,device_id,rank,stream,source_epoch,
                 *,weight_parent=None):
        from src.kernel.kernel_spec import KernelSpec,KernelKind
        from src.kernel.hardware_spec import HWSpec
        from src.kernel.memory_access import check_type
        if not isinstance(kernel,KernelSpec) or not isinstance(hardware,HWSpec):
            raise TypeError('Default source requires its checked kernel/hardware')
        for value in (binding_id,call_id,device_id,stream):check_type(value,str)
        check_type(rank,int)
        if source_epoch is None or kernel.hardware_id!=hardware.hardware_id:
            raise ValueError('Default source requires its current caller epoch')
        matrix=kernel.kind is KernelKind.GEMM and kernel.variant in ('gemm','batched_gemm')
        norm=kernel.kind is KernelKind.VECTOR and has_gpu_normalization_source(kernel,hardware)
        if not (matrix or norm or has_scalar_analytical_work(kernel)):
            raise ValueError('Default invocation has no unchanged analytical operand rule')
        if weight_parent is not None:
            check_type(weight_parent,MemoryAllocation)
            weight_parent.__post_init__()
            size=kernel.shape['k']*kernel.shape['n']*kernel.shape['dtype_bytes']
            if kernel.variant=='batched_gemm':size*=kernel.shape['b']
            if (not matrix or weight_parent.size_bytes!=size or weight_parent.device_id!=device_id
                    or weight_parent.memory_domain!='logical-storage' or weight_parent.freed_event is not None
                    or weight_parent.address_level is not AddressLevel.OBJECT
                    or weight_parent.base_address_bytes is not None or weight_parent.mapping_ref is not None
                    or weight_parent.allocated_event!=weight_parent.allocation_id+'.allocated'):
                raise ValueError('Default matrix parameter differs from its actual source extent/device/lifetime')
            reserved=('A','B','C',binding_id+'.A',binding_id+'.C',binding_id+'.A.allocated',binding_id+'.C.allocated',
                binding_id+'.'+binding_id+'.A.initialized',binding_id+'.'+binding_id+'.B.initialized')
            if weight_parent.allocation_id in reserved or weight_parent.allocated_event in reserved:
                raise ValueError('Default parameter collides with its operand/event namespace')
        visibility='gemm-visible' if matrix else 'norm-visible' if norm else 'elementwise-visible'
        values=(kernel,hardware,binding_id,call_id,device_id,rank,stream,source_epoch,weight_parent,visibility,_default_source_rule)
        names=('kernel','_hardware','binding_id','call_id','device_id','rank','stream','_source_epoch','_weight_parent','visibility_domain','_source_rule')
        self.__dict__.update(zip(names,values,strict=True))
        self.__dict__.update(logical_hardware=None,scope_initialized=True,_default_seal=tuple(zip(names,values,strict=True)),
            _weight_seal=None if weight_parent is None else tuple(getattr(weight_parent,name)
                for name in _invocation_record_fields(MemoryAllocation)))

    def _assert_source_rule(self):
        if any(getattr(module,name) is not function or function.__code__ is not code
                for module,name,function,code in self._source_rule):
            raise ValueError('Default operand rule changed since the actual caller boundary')

    def assert_boundary_unchanged(self):
        if (any(self.__dict__[name] is not value for name,value in self._default_seal)
                or self.logical_hardware is not None or self.scope_initialized is not True
                or any(name in self.__dict__ for name in ('allocations','views','events','operand_source','input_ready',
                    'a_ready','b_ready','query_ready','weight_ready','routing_ready','read_views','write_views'))):
            raise ValueError('Default source boundary was overwritten')
        if self._weight_parent is not None and any(getattr(self._weight_parent,name) is not value for name,value in
                zip(_invocation_record_fields(MemoryAllocation),self._weight_seal,strict=True)):
            raise ValueError('Actual default parameter was overwritten')
        if '_resolved_value' in self.__dict__:
            if self.__dict__['resolved'] is not self._resolved_value:raise ValueError('Resolved default source was overwritten')
            if '_resolved_records' in self.__dict__ and any(getattr(record,name) is not value for record,values in self._resolved_records
                    for name,value in zip(_invocation_record_fields(type(record)),values,strict=True)):
                raise ValueError('Actual resolved default operand was overwritten')
            self._resolved_value.assert_boundary_unchanged()

    @cached_property
    def resolved(self):
        self.assert_boundary_unchanged()
        self._assert_source_rule()
        if self._source_epoch is not _default_source_epoch:
            raise ValueError('Default source recipe changed since its actual caller binding')
        norm=has_gpu_normalization_source(self.kernel,self._hardware)
        execution=self.kernel if norm else memory_execution_kernel(self.kernel,self._hardware)
        if self.kernel.kind.value=='GEMM':
            work=_gemm_work_from_kernel(execution)
            parents,views=gemm_operand_records(work,binding_id=self.binding_id,device_id=self.device_id,
                weight_parent=None if self._weight_parent is None else lambda *args:self._weight_parent)
        else:
            work=execution
            if norm:
                from src.kernel.memory_normalization import normalization_operand_records
                parents,views,_,_=normalization_operand_records(work,binding_id=self.binding_id,device_id=self.device_id)
            else:parents,views,_,_=scalar_operand_records(work,binding_id=self.binding_id,device_id=self.device_id)
        result=MemoryInvocation(self.kernel,work,self.binding_id,self.call_id,parents,views,self.rank,self.stream,self.visibility_domain)
        self.__dict__['_resolved_value']=result
        return result

    def validate(self):
        self.assert_boundary_unchanged()
        self.resolved.validate()

    def prediction_work(self,kernel,hardware):
        self.assert_boundary_unchanged()
        self._assert_source_rule()
        if kernel!=self.kernel:raise ValueError('Default source crossed its kernel')
        return self.resolved.prediction_work(kernel,hardware)

    @cached_property
    def materialized(self):return self.resolved.materialized

    @property
    def read_views(self):return self.resolved.read_views

    @property
    def write_views(self):return self.resolved.write_views

    def __getattr__(self,name):
        if name in ('operand_source','allocations','views','events','a_view','b_view','c_view','a_ready','b_ready'):
            return getattr(self.resolved,name)
        return super().__getattr__(name)

    def __reduce__(self):return _restore_resolved_memory_invocation,(self.resolved,)


def _restore_resolved_memory_invocation(value):
    if type(value) is not MemoryInvocation:raise TypeError('Serialized default source requires canonical operands')
    return value


def _restore_memory_invocation(*values):
    return MemoryInvocation(*values)


def _gemm_work_from_kernel(kernel):
    batched = kernel.variant == "batched_gemm"
    if kernel.variant == "moe_router":
        # The router's dtype names its result; both projection operands are
        # BF16, matching the existing target selector and traffic contract.
        output_dtype = kernel.dtype if kernel.attributes["output_dtype"] is None else kernel.attributes["output_dtype"]
        return GEMMWork(kernel.shape["b"] * kernel.shape["m"],kernel.shape["n"],
            kernel.shape["k"],"bf16",2,trans_b=True,output_dtype=output_dtype)
    return GEMMWork(kernel.shape["m"] if batched else kernel.shape["b"] * kernel.shape["m"],
        kernel.shape["n"], kernel.shape["k"], kernel.dtype, kernel.shape["dtype_bytes"],
        output_dtype=kernel.attributes["output_dtype"],
        batch_count=kernel.shape["b"] if batched else 1, batched=batched)


def build_memory_access_program(kernel, hardware, bindings: MemoryBindings | TransferBindings | ElementwiseBindings | CacheUpdateBindings | EagerAttentionBindings | SparseIndexerBindings | SparseAttentionBindings | FusedMoEBindings | None,
                                granularity: Granularity, *, program_id: str,
                                tile: tuple[int, int, int] | None = None):
    context=copy_context()
    if (_PREDICTION_MEMORY_PROGRAMS not in context or tile is not None
            or type(program_id) is not str):
        return _build_memory_access_program(kernel,hardware,bindings,granularity,program_id=program_id,tile=tile)
    from src.kernel.kernel_spec import KernelSpec
    if not isinstance(kernel,KernelSpec):
        return _build_memory_access_program(kernel,hardware,bindings,granularity,program_id=program_id,tile=tile)
    effective,call_scope=_prediction_bindings(kernel,bindings)
    invocation=context[_PREDICTION_MEMORY_BINDINGS] if _PREDICTION_MEMORY_BINDINGS in context else None
    root=invocation.kernel if invocation is not None else None
    prepared=context[_PREDICTION_MEMORY_PROGRAMS][0]
    for prior_root,prior_kernel,prior_hardware,prior_bindings,prior_scope,explicit,prior_granularity,prior_id,program in prepared:
        same_kernel=(kernel is prior_kernel or root is not None and root is prior_root and kernel==prior_kernel)
        if (same_kernel and hardware is prior_hardware and effective is prior_bindings
                and call_scope==prior_scope and (bindings is not None)==explicit
                and granularity is prior_granularity and program_id==prior_id):
            return program
    program=_build_memory_access_program(kernel,hardware,bindings,granularity,program_id=program_id,tile=tile)
    prepared.append((root,kernel,hardware,effective,call_scope,bindings is not None,granularity,program_id,program))
    if len(prepared)>4:del prepared[0]
    return program


def _build_memory_access_program(kernel, hardware, bindings,
                                granularity: Granularity, *, program_id: str,
                                tile: tuple[int, int, int] | None = None):
    from src.kernel.kernel_spec import KernelKind, KernelSpec
    from src.kernel.hardware_spec import HWSpec
    if not isinstance(kernel, KernelSpec):
        raise TypeError("KernelSpec required")
    if not isinstance(hardware, HWSpec):
        raise TypeError("HWSpec required")
    if kernel.hardware_id != hardware.hardware_id:
        raise ValueError("hardware identity mismatch")
    if kernel.kind is KernelKind.GEMM or has_layernorm_source(kernel,hardware) or has_pairnorm_source(kernel,hardware) or has_public_normalization_source(kernel,hardware):
        kernel = memory_execution_kernel(kernel, hardware)
    if has_cann_moe_transpose_source(kernel,hardware):
        from src.kernel.memory_cann_moe_routing import cann_transpose_execution_kernel
        kernel=cann_transpose_execution_kernel(kernel)
    bindings, call_scope = _prediction_bindings(kernel, bindings)
    if isinstance(bindings,MemoryAccessWork) and bindings.is_source_projection:
        bindings.consume_source(kernel,hardware,bindings.source_evaluator)
        bindings=bindings.invocation
    if isinstance(bindings,(MemoryInvocation,LogicalMemoryInvocation)):
        # Export is the only place the invocation becomes a complete binding.
        bindings=bindings.materialized
    if has_mathematical_gemm_source(kernel):
        if bindings is None:
            bindings = standalone_bindings(_gemm_work_from_kernel(kernel), call_id=program_id,
                binding_id=program_id, device_id="logical-local-device", visibility_domain="mathematical-gemm")
        if not isinstance(bindings, MemoryBindings):
            raise TypeError("mathematical GEMM requires typed matrix operand bindings")
        work, expected = bindings.gemm_work, _gemm_work_from_kernel(kernel)
        fields = ('m', 'n', 'k', 'dtype', 'element_bytes', 'c_dtype', 'c_element_bytes', 'batch_count', 'batched')
        if any(getattr(work, name) != getattr(expected, name) for name in fields):
            raise ValueError("mathematical GEMM source dimensions or storage differ from its KernelSpec")
        from dataclasses import replace
        program = build_logical_gemm(work, hardware.hardware_id, bindings,
            program_id=program_id, granularity=granularity, tile=tile)
        return replace(program, coverage=replace(program.coverage, executable=False,
            limitations=(*program.coverage.limitations,
                "Mathematical operand roles only; native provider, launch and workspace remain unbound.")))
    if has_mathematical_memory_source(kernel):
        return build_mathematical_memory_source(kernel, hardware, bindings, granularity, program_id=program_id)
    if has_logical_interface_source(kernel,hardware) and not has_fa3_prefill_source(kernel):
        if tile is not None:
            raise NotImplementedError('Logical interface summaries do not define native tile overrides')
        if bindings is None:
            bindings = standalone_logical_interface_bindings(kernel,hardware=hardware,call_id=program_id,binding_id=program_id)
        return build_logical_interface(kernel,bindings,granularity,hardware=hardware,program_id=program_id)
    if (isinstance(bindings,ElementwiseBindings) and kernel.kind is not KernelKind.COMMUNICATION
            and not has_target_cache_source(kernel) and not has_metax_varlen_source(kernel) and not has_tensor_only_kv_source(kernel,hardware)
            and not has_moe_count_sort_source(kernel,hardware)
            and not has_moe_alignment_source(kernel,hardware)
            and not has_pairnorm_source(kernel,hardware)
            and not has_generic_reduction_source(kernel,hardware)
            and not has_observed_router_abi(kernel)
            and not has_public_router_abi(kernel)
            and not has_observed_router_reducer_abi(kernel)
            and not (kernel.kind is KernelKind.VECTOR and kernel.variant=='gemm_reduce')
            and kernel.variant!='gemm_elementwise'
            and not has_v32_fused_q_source(kernel,hardware)
            and not has_v32_fused_norm_source(kernel,hardware)
            and not has_dense_flash_attention_source(kernel)
            and not has_fa2_split_source(kernel)
            and not has_fa3_combine_source(kernel)
            and not has_cann_moe_unbound_cast_source(kernel,hardware)
            and not (has_fp8_group_quant_source(kernel))
            and any(isinstance(view,LogicalSelectionView) for view in bindings.views)):
        from src.kernel.memory_cann_mla import has_cann_mla_source
        if not has_cann_mla_source(kernel):
            raise TypeError('logical caller selections require a declared numeric communication, target-cache, C500 varlen, router or flat quantizer ABI')
    if call_scope is not None:
        program_id = f"{call_scope}.{program_id}"
    if has_v32_fused_norm_source(kernel,hardware):
        if tile is not None:raise ValueError('V32 fused norm uses source row tasks rather than matrix tile overrides')
        if bindings is None:bindings=standalone_v32_fused_norm_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_v32_fused_norm_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_v32_fused_q_source(kernel,hardware):
        if tile is not None:raise ValueError('V32 fused Q uses source head tasks rather than matrix tile overrides')
        if bindings is None:bindings=standalone_v32_fused_q_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_v32_fused_q_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_observed_router_reducer_abi(kernel):
        if tile is not None:raise ValueError('Observed reducer cardinality has no physical tile override')
        if bindings is None:bindings=standalone_observed_router_reducer_bindings(kernel,call_id=program_id,binding_id=program_id)
        program=build_observed_router_reducer_source(kernel,hardware,bindings,granularity,program_id=program_id)
        observed_router_reducer_source_work(program,expected_kernel=kernel,hardware=hardware,expected_bindings=bindings)
        return program
    if has_observed_router_abi(kernel):
        if tile is not None:raise ValueError('Observed router tile and split cardinality are unobserved')
        if bindings is None:bindings=standalone_observed_router_bindings(kernel,call_id=program_id,binding_id=program_id)
        program=build_observed_router_source(kernel,hardware,bindings,granularity,program_id=program_id)
        observed_router_source_work(program,expected_kernel=kernel,hardware=hardware,expected_bindings=bindings)
        return program
    if has_simt_router_source(kernel,hardware):
        if tile is not None:raise ValueError('SIMT router uses its source vector/warp partition, not a matrix tile override')
        if bindings is None:bindings=standalone_simt_router_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_simt_router_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_public_router_abi(kernel):
        if tile is not None:raise ValueError('Public router has scalar dot-product tiles, without a native override')
        if bindings is None:bindings=standalone_public_router_bindings(kernel,call_id=program_id,binding_id=program_id)
        program=build_public_router_source(kernel,hardware,bindings,granularity,program_id=program_id)
        public_router_source_work(program,expected_kernel=kernel,hardware=hardware,expected_bindings=bindings)
        return program
    if has_gpu_split_gemm_source(kernel,hardware):
        if tile is not None:
            raise NotImplementedError('split-K source has no issued tile export')
        if bindings is None:
            bindings = standalone_gpu_split_gemm_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_gpu_split_gemm_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_cann_deterministic_gemm_source(kernel):
        if tile is not None:
            raise NotImplementedError('CANN deterministic tile addresses remain unbound')
        if bindings is None:
            bindings = standalone_cann_deterministic_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_cann_deterministic_source(kernel,bindings,granularity,program_id=program_id)
    if has_cann_single_core_gemm_source(kernel):
        if tile is not None:
            raise NotImplementedError('CANN single-core split-K tile addresses remain unbound')
        if bindings is None:
            bindings = standalone_cann_single_core_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_cann_single_core_source(kernel,bindings,granularity,program_id=program_id)
    if has_cann_gemm_source(kernel):
        if tile is not None:
            raise NotImplementedError('CANN matrix source has no bound native tile addresses')
        if bindings is None:
            bindings = standalone_cann_gemm_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_cann_gemm_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_routed_moe_source(kernel,hardware):
        if tile is not None:
            raise NotImplementedError('routed wrapper source has no tile issue trace')
        if bindings is None:
            bindings = standalone_routed_moe_bindings(kernel,call_id=program_id,binding_id=program_id)
        program=build_routed_moe_source(kernel,hardware,bindings,granularity,program_id=program_id)
        from src.kernel.memory_routed_moe import routed_moe_source_demands
        routed_moe_source_demands(program,expected_kernel=kernel,expected_hardware=hardware,expected_bindings=bindings)
        return program
    if has_fp8_fused_moe_source(kernel):
        if tile is not None:
            raise NotImplementedError('FP8 grouped source has no tile issue trace')
        if bindings is None:
            bindings = standalone_fp8_fused_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_fp8_fused_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_fused_moe_source(kernel, hardware):
        if tile is not None:
            raise NotImplementedError('grouped MoE source has no tile issue trace')
        if bindings is None:
            bindings = standalone_fused_moe_bindings(kernel,hardware,call_id=program_id,binding_id=program_id)
        return build_fused_moe_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_sparse_attention_source(kernel):
        if tile is not None:
            raise NotImplementedError("sparse attention has no bound tile address trace")
        if bindings is None:
            bindings = standalone_sparse_attention_bindings(kernel, call_id=program_id, binding_id=program_id)
        program = build_sparse_attention_source(kernel, hardware, bindings, granularity, program_id=program_id)
        if kernel.variant == 'dsv4_sparse_prefill':
            from src.kernel.memory_sparse_attention import v4_prefill_tile_demands
            v4_prefill_tile_demands(program,expected_kernel=kernel,expected_bindings=bindings)
        return program
    if kernel.kind is KernelKind.SPARSE_ATTN_INDEXER and kernel.variant.startswith('dsv4_'):
        from src.kernel.memory_sparse_indexer import V4_INDEXER_SOURCE_GAP
        raise NotImplementedError(V4_INDEXER_SOURCE_GAP)
    if has_sparse_indexer_source(kernel):
        if tile is not None:
            raise NotImplementedError("sparse indexer has no address/tile trace")
        if bindings is None:
            bindings = standalone_sparse_indexer_bindings(kernel, call_id=program_id, binding_id=program_id)
        return build_sparse_indexer_source(kernel, hardware, bindings, granularity, program_id=program_id)
    if kernel.kind is KernelKind.UNFUSED_MOE:
        if not has_unfused_moe_source(kernel, hardware):
            raise NotImplementedError("unfused-MoE source requires its supported compatibility primitive domain")
        if tile is not None:
            raise NotImplementedError("unfused-MoE primitive source has no tile issue order")
        if bindings is None:
            bindings = standalone_unfused_moe_bindings(kernel, call_id=program_id, binding_id=program_id)
        return build_unfused_moe_source(kernel, hardware, bindings, granularity, program_id=program_id)
    if has_fp8_group_quant_source(kernel):
        if tile is not None:
            raise NotImplementedError("FP8 group quantizer has no tile issue trace")
        if bindings is None:
            bindings = standalone_fp8_group_quant_bindings(kernel, call_id=program_id, binding_id=program_id)
        return build_fp8_group_quant_source(kernel, hardware, bindings, granularity, program_id=program_id)
    if has_block_fp8_source(kernel):
        if tile is not None:
            raise NotImplementedError("block-FP8 interface has no declared TMA tile issue order")
        if bindings is None:
            bindings = standalone_block_fp8_bindings(kernel, call_id=program_id, binding_id=program_id)
        return build_block_fp8_source(kernel, hardware, bindings, granularity, program_id=program_id)
    if kernel.kind is KernelKind.COMMUNICATION:
        if not has_communication_source(kernel):
            raise NotImplementedError("collective source does not yet express fused residual/normalization side effects")
        if tile is not None:
            raise NotImplementedError("collective interface has no packet/tile issue order")
        if bindings is None:
            bindings = standalone_communication_bindings(kernel, call_id=program_id, binding_id=program_id)
        return build_communication_interface(kernel, hardware, bindings, granularity, program_id=program_id)
    if kernel.kind is KernelKind.FLASH_ATTN:
        if has_cann_tnd_source(kernel):
            if tile is not None:
                raise NotImplementedError('CANN TND native tile addresses remain unknown')
            if bindings is None:
                bindings = standalone_cann_tnd_bindings(kernel,call_id=program_id,binding_id=program_id)
            return build_cann_tnd_source(kernel,bindings,granularity,program_id=program_id)
        if has_metax_varlen_source(kernel):
            if tile is not None:
                raise NotImplementedError('C500 varlen has no native tile address mapping')
            if bindings is None:
                bindings = standalone_metax_varlen_bindings(kernel,call_id=program_id,binding_id=program_id)
            return build_metax_varlen_source(kernel,bindings,granularity,program_id=program_id)
        from src.kernel.memory_cann_mla import has_cann_mla_source,standalone_cann_mla_bindings,build_cann_mla_source
        if has_cann_mla_source(kernel):
            if tile is not None:raise NotImplementedError('CANN MLA source retains its selected logical work blocks')
            if bindings is None:bindings=standalone_cann_mla_bindings(kernel,hardware,call_id=program_id,binding_id=program_id)
            return build_cann_mla_source(kernel,hardware,bindings,program_id=program_id)
        if has_cann_attention_source(kernel):
            if tile is not None:
                raise NotImplementedError('CANN attention native tile addresses remain unknown')
            if bindings is None:
                bindings = standalone_cann_attention_bindings(kernel,hardware,call_id=program_id,binding_id=program_id)
            return build_cann_attention_source(kernel,hardware,bindings,granularity,program_id=program_id)
        if has_fa3_combine_source(kernel):
            if tile is not None:raise NotImplementedError('FA3 combine retains its native row/value tiles')
            if bindings is None:
                bindings=standalone_fa3_combine_bindings(kernel,hardware,call_id=program_id,binding_id=program_id)
            program=build_fa3_combine_source(kernel,hardware,bindings,granularity,program_id=program_id)
            from src.kernel.memory_flash_attention import fa3_combine_source_demands
            fa3_combine_source_demands(program,expected_kernel=kernel,hardware=hardware,expected_bindings=bindings)
            return program
        if has_fa3_prefill_source(kernel):
            if tile is not None:raise NotImplementedError('FA3 native tile addresses remain unbound')
            if bindings is None:
                bindings = standalone_fa3_prefill_bindings(kernel,call_id=program_id,binding_id=program_id)
            program=build_fa3_prefill_source(kernel,bindings,granularity,program_id=program_id)
            from src.kernel.memory_flash_attention import fa3_prefill_source_demands
            fa3_prefill_source_demands(program,expected_kernel=kernel,expected_bindings=bindings)
            return program
        if has_fa2_split_source(kernel):
            if tile is not None:
                raise NotImplementedError('FA2 split native tile addresses remain unknown')
            if bindings is None:
                bindings = standalone_fa2_split_bindings(kernel,call_id=program_id,binding_id=program_id)
            return build_fa2_split_source(kernel,bindings,granularity,program_id=program_id)
        if has_grouped_mla_source(kernel):
            if tile is not None:
                raise NotImplementedError('Grouped MLA native tile/page mapping remains pending')
            if bindings is None:
                bindings = standalone_grouped_mla_bindings(kernel,call_id=program_id,binding_id=program_id)
            return build_grouped_mla_source(kernel,bindings,granularity,program_id=program_id)
        if has_metax_mla_source(kernel):
            if tile is not None:
                raise NotImplementedError('MetaX MLA has no native tile-address mapping')
            if bindings is None:
                bindings = standalone_metax_mla_bindings(kernel,call_id=program_id,binding_id=program_id)
            return build_metax_mla_source(kernel,bindings,granularity,program_id=program_id)
        if not has_dense_flash_attention_source(kernel):
            raise NotImplementedError("FlashAttention source requires the declared dense external tensor domain")
        if tile is not None:
            raise NotImplementedError("dense external tensor interface has no declared tile issue trace")
        if bindings is None:
            bindings = standalone_dense_attention_bindings(kernel, call_id=program_id, binding_id=program_id)
        return build_dense_attention_interface(kernel, hardware, bindings, granularity, program_id=program_id)
    if kernel.kind is KernelKind.EAGER_ATTN:
        if not has_eager_attention_source(kernel, hardware) or kernel.execution_support != "supported":
            raise NotImplementedError("Eager source requires its declared compatibility primitive domain")
        if tile is not None:
            raise NotImplementedError("Eager source has no declared tile issue order")
        if bindings is None:
            bindings = standalone_eager_attention_bindings(kernel, call_id=program_id, binding_id=program_id)
        return build_eager_attention(kernel, hardware, bindings, granularity, program_id=program_id)
    if has_target_cache_source(kernel):
        if tile is not None:raise NotImplementedError('Target cache accepts its semantic token tiles, not native address tiles')
        if bindings is None:bindings = standalone_target_cache_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_target_cache_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_tensor_only_kv_source(kernel,hardware):
        if tile is not None:raise NotImplementedError('Tensor-only KV has no native bank/page tile mapping')
        if bindings is None:bindings=standalone_tensor_kv_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_tensor_kv_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if kernel.kind is KernelKind.KV_CACHE and kernel.variant == "paged_kv_cache":
        if not has_paged_kv_source(kernel, hardware) or kernel.execution_support != "supported":
            raise NotImplementedError("paged KV source needs the declared serving ABI")
        if tile is not None:
            raise NotImplementedError("unknown page mapping cannot declare a tile address trace")
        if bindings is None:
            bindings = standalone_cache_update_bindings(kernel, hardware, call_id=program_id, binding_id=program_id)
        return build_paged_kv_update(kernel, hardware, bindings, granularity, program_id=program_id)
    if has_cann_moe_transpose_source(kernel,hardware) and is_cann_transpose_alias(kernel):
        if tile is not None:raise ValueError('A zero-memory view has no native memory tile override')
        if bindings is None:bindings=standalone_cann_transpose_alias_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_cann_transpose_alias_source(kernel,bindings,granularity,program_id=program_id)
    if has_cann_moe_range_source(kernel,hardware):
        if tile is not None:
            raise NotImplementedError('CANN Range native tile addresses are unknown')
        if bindings is None:
            bindings = standalone_cann_moe_range_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_cann_moe_range_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_cann_expert_tokens_source(kernel,hardware):
        if tile is not None:
            raise NotImplementedError('ComputeExpertTokens has no issued tile mapping')
        if bindings is None:
            bindings = standalone_cann_expert_tokens_bindings(kernel,hardware,call_id=program_id,binding_id=program_id)
        return build_cann_expert_tokens_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_cann_moe_init_source(kernel,hardware):
        if tile is not None:
            raise NotImplementedError('Legacy InitRouting has no issued tile mapping')
        if bindings is None:
            bindings = standalone_cann_moe_init_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_cann_moe_init_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_cann_moe_gating_source(kernel,hardware):
        if tile is not None:
            raise NotImplementedError('CANN gating top-k has no issued tile trace')
        if bindings is None:
            bindings = standalone_cann_moe_gating_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_cann_moe_gating_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_cann_moe_finalize_source(kernel,hardware):
        if tile is not None:
            raise NotImplementedError('CANN finalization has no issued tile trace')
        if bindings is None:
            bindings = standalone_cann_moe_finalize_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_cann_moe_finalize_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_moe_fill_source(kernel,hardware):
        if tile is not None:
            raise NotImplementedError('MoE fill has no issued tile trace')
        from src.kernel.memory_moe_fill import FillMemoryInvocation
        if isinstance(bindings,FillMemoryInvocation):
            bindings.work(kernel,hardware)
            bindings=bindings.materialized
        if bindings is None:
            bindings = standalone_moe_fill_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_moe_fill_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_moe_alignment_source(kernel,hardware):
        if tile is not None:raise NotImplementedError('Alignment exposes logical metadata tiles, not native issue overrides')
        if bindings is None:bindings=standalone_moe_alignment_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_moe_alignment_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_moe_count_sort_source(kernel,hardware):
        if tile is not None:raise NotImplementedError('Count-and-sort exposes logical route tiles, not an issued tile override')
        if bindings is None:bindings=standalone_moe_count_sort_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_moe_count_sort_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_gpu_topk_source(kernel,hardware):
        if tile is not None:raise NotImplementedError('GPU top-k exposes logical row tiles, not native issue overrides')
        if bindings is None:bindings=standalone_gpu_topk_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_gpu_topk_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_route_weight_source(kernel,hardware):
        if tile is not None:raise NotImplementedError('Public route weights expose row tiles, not native issue overrides')
        if bindings is None:bindings=standalone_route_weight_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_route_weight_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_moe_reduction_source(kernel,hardware):
        if tile is not None:
            raise NotImplementedError('MoE reduction has no issued tile trace')
        if bindings is None:
            bindings = standalone_moe_reduction_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_moe_reduction_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_moe_pointwise_source(kernel,hardware):
        if tile is not None:
            raise NotImplementedError('MoE pointwise has no issued tile trace')
        if bindings is None:
            bindings = standalone_moe_pointwise_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_moe_pointwise_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_cann_rotary_source(kernel,hardware):
        if tile is not None:raise NotImplementedError('CANN rotary uses source-selected slices, not an issued tile override')
        if bindings is None:bindings=standalone_cann_rotary_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_cann_rotary_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_tensor_only_rotary_source(kernel,hardware):
        if tile is not None:raise NotImplementedError('Tensor-only rotary has no native issue tile override')
        if bindings is None:
            bindings=standalone_tensor_rotary_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_tensor_rotary_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_gpu_rotary_source(kernel,hardware):
        if tile is not None:
            raise NotImplementedError('GPU RoPE does not export issued tile transactions')
        if bindings is None:
            bindings = standalone_rotary_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_gpu_rotary_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_cann_normalization_source(kernel,hardware):
        if tile is not None:
            raise NotImplementedError('CANN normalization source has no AIV tile issue trace')
        if bindings is None:
            bindings = standalone_cann_normalization_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_cann_normalization_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_pim_normalization_source(kernel,hardware):
        if tile is not None:raise NotImplementedError('PIM row source has no native bank issue tile override')
        if bindings is None:bindings=standalone_normalization_bindings(kernel,hardware,call_id=program_id,binding_id=program_id)
        return build_pim_normalization(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_public_normalization_source(kernel,hardware):
        if tile is not None:raise NotImplementedError('Public normalization has no native tile override')
        if bindings is None:
            bindings=standalone_public_normalization_bindings(kernel,hardware,call_id=program_id,binding_id=program_id)
        return build_public_normalization(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_generic_reduction_source(kernel,hardware):
        if tile is not None:raise ValueError('Generic reduction uses its actual input/output group axes')
        if bindings is None:bindings=standalone_generic_reduction_bindings(kernel,call_id=program_id,binding_id=program_id)
        program=build_generic_reduction_source(kernel,hardware,bindings,granularity,program_id=program_id)
        from src.kernel.memory_reduction import generic_reduction_source_work
        generic_reduction_source_work(program,expected_kernel=kernel,expected_bindings=bindings)
        return program
    if has_mhc_source(kernel,hardware):
        if tile is not None:raise ValueError('mHC uses its source row/hidden/projection partitions')
        if bindings is None:bindings=standalone_mhc_bindings(kernel,call_id=program_id,binding_id=program_id)
        program=build_mhc_source(kernel,hardware,bindings,granularity,program_id=program_id)
        from src.kernel.memory_mhc import mhc_source_work
        mhc_source_work(program,expected_kernel=kernel,expected_bindings=bindings)
        return program
    if has_moe_metadata_source(kernel,hardware):
        if tile is not None:raise ValueError('MoE metadata uses its declared scalar/reduction partitions')
        if bindings is None:bindings=standalone_moe_metadata_bindings(kernel,call_id=program_id,binding_id=program_id)
        program=build_moe_metadata_source(kernel,hardware,bindings,granularity,program_id=program_id)
        from src.kernel.memory_moe_routing import validate_moe_metadata_bindings
        validate_moe_metadata_bindings(program,expected_kernel=kernel,expected_bindings=bindings)
        return program
    if has_pairnorm_source(kernel,hardware):
        if tile is not None:raise ValueError('Paired normalization uses its source row tasks')
        if bindings is None:bindings=standalone_pairnorm_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_pairnorm_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if has_layernorm_source(kernel,hardware):
        if tile is not None:raise ValueError('LayerNorm uses its declared semantic row partition')
        if bindings is None:bindings=standalone_layernorm_bindings(kernel,call_id=program_id,binding_id=program_id)
        program=build_layernorm_source(kernel,hardware,bindings,granularity,program_id=program_id)
        from src.kernel.memory_normalization import layernorm_source_work
        layernorm_source_work(program,expected_kernel=kernel,expected_bindings=bindings)
        return program
    if has_gpu_normalization_source(kernel, hardware):
        if tile is not None:
            raise NotImplementedError("normalization source has no declared tile issue order")
        if bindings is None:
            bindings = standalone_normalization_bindings(kernel,hardware, call_id=program_id, binding_id=program_id)
        return build_gpu_normalization(kernel, hardware, bindings, granularity, program_id=program_id)
    if has_cann_moe_unbound_cast_source(kernel,hardware):
        if tile is not None:raise ValueError('Unknown-storage cast has no issued tile override')
        if bindings is None:bindings=standalone_cann_unbound_cast_bindings(kernel,call_id=program_id,binding_id=program_id)
        return build_cann_unbound_cast_source(kernel,hardware,bindings,granularity,program_id=program_id)
    if (kernel.kind is KernelKind.VECTOR
            and (has_cann_moe_transpose_source(kernel,hardware) or has_cann_moe_expert_cast_source(kernel,hardware) or has_moe_abs_source(kernel,hardware) or kernel.variant in ("flash_mla_metadata", "elementwise", "elementwise_pytorch_vec4", "gemm_reduce", "gemm_elementwise", "activation", "silu", "moe_act_and_mul", "moe_swiglu", "softmax", "dsv4_short_context_topk"))):
        if kernel.execution_support != "supported":
            raise NotImplementedError("unsupported elementwise execution")
        if tile is not None or granularity not in (Granularity.OP, Granularity.KERNEL, Granularity.TILE):
            raise ValueError("scalar source selects its semantic partition; explicit tile override is unavailable")
        if bindings is None:
            bindings = standalone_elementwise_bindings(kernel, call_id=program_id,
                                                        binding_id=program_id)
        return build_logical_elementwise(kernel, bindings, granularity, program_id=program_id)
    if kernel.kind in (KernelKind.MEMCPY, KernelKind.KV_OFFLOAD):
        if kernel.execution_support != "supported":
            raise NotImplementedError("unsupported transfer execution")
        if tile is not None or granularity not in (Granularity.OP, Granularity.KERNEL, Granularity.TILE):
            raise ValueError("byte-copy source uses independent semantic byte coordinates")
        if bindings is None:
            bindings = standalone_transfer_bindings(kernel, call_id=program_id,
                                                     binding_id=program_id)
        return build_logical_transfer(kernel, bindings, granularity, program_id=program_id)
    if kernel.kind is not KernelKind.GEMM or kernel.execution_support != "supported":
        raise NotImplementedError("logical production export currently supports GEMM only")
    # Positive source-domain declaration: output dtype alone cannot establish
    # storage widths. In particular H200 block-FP8 launches output BF16 while
    # reading FP8/scale objects. Unknown launches must not become BF16 exports.
    unquantized = {
        "nvidia_a800": {"nvidia_cublaslt_gemm", "nvidia_cublas_gemv",
                         "cublas_analytical_gemm", "cublas_analytical_gemv",
                         "cublas_analytical_gemm:batched_gemm", "nvidia_cublaslt_gemm:batched_gemm",
                         "cublas_analytical_gemm:moe_router", "nvidia_cublaslt_gemm:moe_router"},
        "nvidia_h200": {"nvidia_cublaslt_gemm", "nvidia_cublas_gemv",
                         "h200_cublaslt129_injected_gemm",
                         "h200_vllm030_torch213_fp16_observed_gemm",
                         "cublas_analytical_gemm", "cublas_analytical_gemv",
                         "cublas_analytical_gemm:batched_gemm", "nvidia_cublaslt_gemm:batched_gemm",
                         "cublas_analytical_gemm:moe_router", "nvidia_cublaslt_gemm:moe_router"},
        "metax_c500": {"metax_mcblas_gemm", "metax_mcblas_gemv", "mcblas_analytical_gemm",
                         "mcblas_gemv", "mcblas_gemv_wave", "mcblas_injected_gemm:gemm"},
        "ascend_910b": {"ascend_cann_matmul", "ascend_cann_gemv", "cann_matmul_v3"},
        # Compatibility simulators describe scalar operand storage directly;
        # int8 here is an integer tensor ABI, not packed payload plus scales.
        "groq_lpu": {"observed_gemm:gemm", "observed_gemm:gemv"},
        "samsung_hbm_pim": {"observed_gemm:gemm", "observed_gemm:gemv"},
        "ucb_gemmini": {"observed_gemm:gemm", "observed_gemm:gemv"},
    }
    # The scalar GEMV contract already declares its unpacked operands. An
    # opaque untiled launch does not change their logical storage geometry.
    scalar_gemv=(kernel.variant=='gemv' and kernel.shape['b']*kernel.shape['m']==1
        and kernel.launch.tiling is None and kernel.dtype in ('bf16','fp16','fp32'))
    if not scalar_gemv and (kernel.hardware_id not in unquantized or kernel.launch.algorithm not in unquantized[kernel.hardware_id]):
        raise NotImplementedError("quantized or unknown GEMM launch storage export is unsupported")
    if kernel.variant not in ("gemm", "gemv", "batched_gemm", "moe_router"):
        raise NotImplementedError("specialized GEMM variant requires a complete operand/epilogue export")
    if kernel.variant == "moe_router" and not has_gpu_nonsplit_gemm_source(kernel,hardware):
        raise NotImplementedError("router source requires a known nonsplit BF16 projection launch")
    if kernel.variant == "batched_gemm" and kernel.attributes["weight_dtype"] != "bf16":
        raise NotImplementedError("quantized batched GEMM requires payload and scale objects")
    if kernel.dtype not in ("bf16", "fp16", "fp32", "int8"):
        raise NotImplementedError("quantized GEMM export requires additional objects")
    if bindings is None:
        work = _gemm_work_from_kernel(kernel)
        bindings = standalone_bindings(work, call_id=program_id, binding_id=program_id,
                                       device_id="logical-local-device", visibility_domain="gemm-visible")
    if not isinstance(bindings, MemoryBindings):
        raise TypeError("typed operand bindings with explicit logical GEMM semantics required")
    work = bindings.gemm_work
    expected = _gemm_work_from_kernel(kernel)
    if (work.m, work.n, work.k, work.dtype, work.element_bytes, work.batch_count, work.batched) != (
            expected.m, expected.n, expected.k, expected.dtype, expected.element_bytes,
            expected.batch_count, expected.batched):
        raise ValueError("KernelSpec and binding logical GEMM dimensions/dtype mismatch")
    output_dtype = kernel.dtype if kernel.attributes["output_dtype"] is None else kernel.attributes["output_dtype"]
    if work.c_dtype != output_dtype:
        raise ValueError("KernelSpec and binding GEMM output storage mismatch")
    if kernel.variant=='moe_router' and (work.trans_a,work.trans_b)!=(False,True):
        raise ValueError('Router source requires its actual [expert,K] weight parent')
    if tile is not None:
        selected_tile=tile;source='explicit caller logical partition'
    elif kernel.launch.tiling is not None and {'m','n','k'} <= set(kernel.launch.tiling.tile_shape):
        selected_tile=_gemm_source_tile(kernel,work)
        source='LaunchConfig.tiling.tile_shape(m,n,k)'
    else:
        selected_tile=(1,1,work.k)
        source='C[row,column] scalar dot-product over the complete K reduction; native launch decomposition unavailable'
    program = build_tiled_gemm(work,hardware.hardware_id,bindings,program_id=program_id,tile=selected_tile,tile_source=source)
    if "model_input_shape" in kernel.attributes:
        from dataclasses import replace
        physical = kernel.attributes["model_input_shape"]
        resolved = ({"b", "m", "n", "k"} <= set(physical)
                    and all(physical[axis] == kernel.shape[axis] for axis in ("b", "m", "n", "k")))
        limitation = ("Captured GEMM uses declared matrix input extent; padding values are unknown, not zero-work."
                      if resolved else "Captured non-row or incomplete GEMM mapping remains unresolved; declared logical interface only.")
        program = replace(program, coverage=replace(program.coverage,
            limitations=program.coverage.limitations + (limitation,)))
    return program


def gemm_memory_access_program(kernel, hardware):
    """Invocation-local logical operands, shared by prediction and export.

    The normalized scalar GEMM ABI supplies logical storage extents. This
    does not invent a vendor issue order, padded operand storage, residency,
    or persistent allocation identity across Serving calls.
    """
    return build_memory_access_program(kernel, hardware, None, Granularity.KERNEL,
                                       program_id="gemm.operands")


def gemm_prediction_source(kernel,hardware):
    """Consume actual matrix work directly; explicit graph API stays explicit."""
    actual,_=_prediction_bindings(kernel,None)
    if isinstance(actual,MemoryInvocation):
        return actual.prediction_work(kernel if actual.logical_hardware is not None else actual.kernel,hardware)
    if actual is None and has_logical_interface_source(kernel,hardware):
        return logical_interface_memory_access_program(kernel,hardware)
    if actual is None and kernel.variant in ('gemm','batched_gemm') and has_gpu_nonsplit_gemm_source(kernel,hardware):
        execution=memory_execution_kernel(kernel,hardware);work=_gemm_work_from_kernel(execution)
        parents,views=gemm_operand_records(work,binding_id='gemm.analytical',device_id='logical-local-device')
        invocation=MemoryInvocation(kernel,work,'gemm.analytical','gemm.analytical',parents,views,0,'offline')
        invocation = _own_implicit_memory_source(kernel, invocation)
        return invocation.prediction_work(kernel,hardware)
    return gemm_memory_access_program(kernel,hardware)


@dataclass(frozen=True)
class ElementwiseBindings(Typed):
    """Scalar tensor operand identities; read views may also be written."""
    binding_id: str
    call_id: str
    allocations: tuple[MemoryAllocation, ...]
    views: tuple[TensorView | LogicalSelectionView, ...]
    events: tuple[ProgramEvent, ...]
    read_views: tuple[str, ...]
    write_views: tuple[str, ...]
    input_ready: tuple[EventRef, ...]
    rank: int
    stream: str
    visibility_domain: str


def _elementwise_operand_types(kernel):
    """Scalar source ABI, including storage-changing compare/cast operations."""
    operation = kernel.attributes["operation"]
    if operation == "cast_bf16_float32":
        return (("bf16", 2),), (("fp32", 4),)
    if operation == "notequal_int32_bool":
        return (("int32", 4),), (("bool", 1),)
    if operation == "cast_bool_bf16":
        return (("bool", 1),), (("bf16", 2),)
    scalar = (kernel.dtype, kernel.shape["dtype_bytes"])
    if operation in ("add", "mul"):
        return (scalar, scalar), (scalar,)
    if operation in ("unary", "copy"):
        return (scalar,), (scalar,)
    if operation == "fill":
        return (), (scalar,)
    if operation == "read":
        return (scalar,), ()
    raise ValueError(f"unknown scalar elementwise operation: {operation!r}")


def _gemm_elementwise_extents(kernel):
    logical=kernel.shape['element_count']
    if 'model_input_shape' not in kernel.attributes:return logical,logical
    physical=kernel.attributes['model_input_shape']
    if set(physical)!={'element_count'}:raise ValueError('GEMM epilogue replay requires its exact scalar geometry')
    execution=physical['element_count']
    if execution<logical or (execution!=logical and (
            kernel.attributes['graph_runtime_mode'] not in ('FULL','PIECEWISE')
            or execution*kernel.attributes['graph_logical_tokens']!=logical*kernel.attributes['graph_model_input_tokens'])):
        raise ValueError('GEMM epilogue replay differs from its graph coordinate mapping')
    return logical,execution


def _gemm_elementwise_header(kernel):
    logical,execution=_gemm_elementwise_extents(kernel)
    return ('gemm_elementwise',) if logical==execution else (
        'gemm_elementwise',f'logical_elements={logical}',f'execution_elements={execution}')


def _vector_operand_layouts(kernel):
    """Untimed tensor ABI for scalar maps and external split-K reduction."""
    if kernel.variant == 'moe_transpose':
        operand = ((kernel.shape['tokens'],kernel.shape['topk']),'int32',4)
        return (operand,),(operand,)
    if kernel.variant == 'flash_mla_metadata':
        from src.kernel.memory_flash_attention import flash_mla_metadata_operand_layouts
        return flash_mla_metadata_operand_layouts(kernel)
    if kernel.variant == 'moe_cast' and kernel.attributes['cast_role'] == 'expert_tokens':
        return ((((kernel.shape['experts'],),'int32',4),),(((kernel.shape['experts'],),'int64',8),))
    if kernel.variant == 'moe_abs':
        operand = ((kernel.shape['tokens']*kernel.shape['topk'],),'int32',4)
        return (operand,),(operand,)
    if kernel.variant == "dsv4_short_context_topk":
        rows,capacity = kernel.shape["seq_len"],kernel.shape["topk_capacity"]
        return (((rows,),"int64",8),), (((rows,capacity),"int32",4),)
    if kernel.variant in ("moe_act_and_mul", "moe_swiglu"):
        rows = (kernel.shape["tokens"] * kernel.shape["topk"] if kernel.variant == "moe_act_and_mul"
                else kernel.attributes["routed_rows"])
        operand = ((rows, kernel.shape["intermediate"]), "bf16", 2)
        return (operand, operand), (operand,)
    elements = kernel.shape["element_count"]
    if kernel.variant in ("activation", "silu", "moe_act_and_mul", "moe_swiglu"):
        shape = (kernel.shape["tok"], kernel.shape["intermediate"]) if "tok" in kernel.shape else (elements,)
        operand = (shape, kernel.dtype, kernel.shape["dtype_bytes"])
        return (operand, operand), (operand,)
    if kernel.variant in ("elementwise", "elementwise_pytorch_vec4"):
        inputs, outputs = _elementwise_operand_types(kernel)
        if "input_materialization" in kernel.attributes:
            shape = (kernel.attributes["input_outer_groups"],
                     kernel.attributes["input_rows_per_group"],
                     kernel.attributes["input_row_stride_elements"])
            return (tuple((shape, dtype, width) for dtype, width in inputs),
                    tuple(((elements,), dtype, width) for dtype, width in outputs))
        return (tuple(((elements,), dtype, width) for dtype, width in inputs),
                tuple(((elements,), dtype, width) for dtype, width in outputs))
    if kernel.variant=='gemm_elementwise':_,elements=_gemm_elementwise_extents(kernel)
    output = ((elements,), kernel.dtype, kernel.shape["dtype_bytes"])
    if kernel.variant == "softmax" and "rows" in kernel.shape:
        operand = ((kernel.shape["rows"], kernel.shape["hidden"]), kernel.dtype, kernel.shape["dtype_bytes"])
        return (operand,), (operand,)
    if kernel.variant in ("gemm_elementwise", "softmax"):
        return (output,), (output,)
    if kernel.variant == "gemm_reduce":
        if kernel.launch.algorithm not in {
                "observed_gemm_aux:gemm_reduce", "observed_mcblas_gemm_aux:gemm_reduce",
                "logical_gemm_aux:gemm_reduce"}:
            raise NotImplementedError("split-K reducer needs a declared external partial ABI")
        split_k = kernel.attributes["split_k"]
        if split_k <= 1:
            raise ValueError("split-K reducer requires split_k greater than one")
        width = int(kernel.launch.parameters["split_k_partial_bytes_per_element"])
        if width == kernel.shape["dtype_bytes"]:
            partial = ((split_k, elements), kernel.dtype, width)
        elif width == 4 and kernel.dtype in ("bf16", "fp16", "fp32"):
            partial = ((split_k, elements), "fp32", 4)
        else:
            # Width is declared, scalar encoding is not. Preserve actual
            # materialized storage bytes without inventing a float dtype.
            partial = ((split_k, elements, width), "uint8", 1)
        return (partial,), (output,)
    raise ValueError(f"unknown scalar/reducer tensor ABI: {kernel.variant!r}")


def scalar_operand_records(kernel, *, binding_id, device_id):
    """Scalar layouts shared by normal invocation and explicit graph export."""
    if (kernel.variant in ('moe_act_and_mul','moe_swiglu')
            or kernel.variant in ('activation','silu') and 'tok' in kernel.shape):
        operands, _ = _vector_operand_layouts(kernel)
        shape, dtype, width = operands[0]
        rows, columns = shape
        input_id, output_id = f"{binding_id}.input", f"{binding_id}.output"
        size = rows * 2 * columns * width
        allocations = (
            MemoryAllocation(input_id, device_id, "logical-storage", size,
                AddressLevel.OBJECT, None, None, f"{input_id}.allocated", None),
            MemoryAllocation(output_id, device_id, "logical-storage", size // 2,
                AddressLevel.OBJECT, None, None, f"{output_id}.allocated", None))
        views = (TensorView("input0", input_id, 0, shape, (2*columns*width, width), dtype, width),
                 TensorView("input1", input_id, columns*width, shape, (2*columns*width, width), dtype, width),
                 TensorView("output0", output_id, 0, shape, (columns*width, width), dtype, width))
        return allocations,views,('input0','input1'),('output0',)
    input_layouts, output_layouts = _vector_operand_layouts(kernel)
    allocations, views = [], []
    read_views, write_views = [], []
    for direction, operands in (("input", input_layouts), ("output", output_layouts)):
        for index, (shape, dtype, width) in enumerate(operands):
            view_id = f"{direction}{index}"
            allocation_id = f"{binding_id}.{view_id}"
            allocated = f"{allocation_id}.allocated"
            strides, size = [], width
            for dimension in reversed(shape):
                strides.append(size)
                size *= dimension
            if direction == 'input' and kernel.variant == 'moe_transpose':
                view = TensorView(view_id,allocation_id,0,shape,(4,shape[0]*4),dtype,width)
            elif direction == "input" and "input_materialization" in kernel.attributes:
                strides = (kernel.attributes["input_group_stride_elements"] * width,
                           kernel.attributes["input_row_stride_elements"] * width, width)
                view = TensorView(view_id, allocation_id, 0, shape, strides, dtype, width)
                from src.kernel.memory_access import region_end
                size = region_end(view.region())
            else:
                view = TensorView(view_id, allocation_id, 0, shape, tuple(reversed(strides)), dtype, width)
            allocations.append(MemoryAllocation(allocation_id, device_id, "logical-storage",
                size, AddressLevel.OBJECT, None, None, allocated, None))
            views.append(view)
            if direction == "input":read_views.append(view_id)
            else:write_views.append(view_id)
    return tuple(allocations),tuple(views),tuple(read_views),tuple(write_views)


def standalone_elementwise_bindings(kernel, *, call_id, binding_id, device_id="logical-local-device", rank=0, stream="offline"):
    """Explicit event projection of the same scalar operand records."""
    allocations,views,reads,writes=scalar_operand_records(kernel,binding_id=binding_id,device_id=device_id)
    packed=len(allocations)!=len(views)
    events=[];ready=[]
    for parent in allocations:
        events.append(ProgramEvent(parent.allocated_event,EventKind.ALLOCATED,parent.allocation_id,None,None,()))
        inputs=tuple(view for view in views if view.view_id in reads and view.allocation_id==parent.allocation_id)
        if inputs and not packed:
            initialized=parent.allocation_id+'.initialized'
            events.append(ProgramEvent(initialized,EventKind.INITIALIZED,parent.allocation_id,inputs[0].region(),
                'elementwise-visible',(EventRef(parent.allocated_event,EventKind.ALLOCATED),)))
            ready.append(EventRef(initialized,EventKind.INITIALIZED))
    if packed:
        parent=allocations[0];initialized=parent.allocation_id+'.initialized'
        events.append(ProgramEvent(initialized,EventKind.INITIALIZED,parent.allocation_id,Span(0,parent.size_bytes),
            'elementwise-visible',(EventRef(parent.allocated_event,EventKind.ALLOCATED),)))
        ready=[EventRef(initialized,EventKind.INITIALIZED)]*len(reads)
    return ElementwiseBindings(binding_id,call_id,allocations,views,tuple(events),reads,writes,tuple(ready),
        rank,stream,'elementwise-visible')


def _export_tensor_invocation(invocation):
    from dataclasses import replace
    if invocation.operand_source.variant in ('rmsnorm','add_rmsnorm'):
        binding=standalone_normalization_bindings(invocation.operand_source,None,call_id=invocation.call_id,
            binding_id=invocation.binding_id,device_id=invocation.allocations[0].device_id,rank=invocation.rank,stream=invocation.stream)
    else:
        binding=standalone_elementwise_bindings(invocation.operand_source,call_id=invocation.call_id,
            binding_id=invocation.binding_id,device_id=invocation.allocations[0].device_id,
            rank=invocation.rank,stream=invocation.stream)
    actual={view.view_id:view for view in invocation.views}
    parents={p.allocation_id:p for p in invocation.allocations}
    replacements={view.allocation_id:parents[actual[view.view_id].allocation_id] for view in binding.views}
    names={event.event_id:(replacements[event.allocation_id].allocated_event if event.kind is EventKind.ALLOCATED
        else (invocation.binding_id+'.' if invocation.scope_initialized else '')+event.event_id) for event in binding.events}
    selected={ready.node_id:actual[role] for role,ready in zip(binding.read_views,binding.input_ready,strict=True)
        if isinstance(actual[role],LogicalSelectionView)}
    def ref(value):return EventRef(names[value.node_id],EventKind.LOGICAL_AVAILABLE if value.node_id in selected else value.kind)
    events={}
    for event in binding.events:
        projected=replace(event,event_id=names[event.event_id],allocation_id=replacements[event.allocation_id].allocation_id,
            dependencies=tuple(ref(value) for value in event.dependencies))
        if event.event_id in selected:
            projected=replace(projected,kind=EventKind.LOGICAL_AVAILABLE,region=selected[event.event_id].region())
        if projected.event_id in events and events[projected.event_id]!=projected:
            raise ValueError('Shared scalar producer event records disagree')
        events[projected.event_id]=projected
    return replace(binding,allocations=invocation.allocations,views=invocation.views,events=tuple(events.values()),
        input_ready=tuple(ref(value) for value in binding.input_ready),visibility_domain=invocation.visibility_domain)


def _scalar_operand_bindings(kernel, bindings):
    """Shared scalar operand ABI and alias validation for prediction and export."""
    if not isinstance(bindings,(ElementwiseBindings,MemoryInvocation)):
        raise TypeError("typed scalar elementwise bindings required")
    if any(not isinstance(view,TensorView) and not (isinstance(view,LogicalSelectionView)
            and (kernel.variant=='gemm_elementwise' or kernel.variant=='gemm_reduce' and view.view_id in bindings.read_views))
            for view in bindings.views):
        raise TypeError('scalar elementwise operands require explicit tensor layout bindings')
    input_layouts, output_layouts = _vector_operand_layouts(kernel)
    if (len(bindings.read_views) != len(input_layouts)
            or len(bindings.write_views) != len(output_layouts)
            or isinstance(bindings,ElementwiseBindings) and len(bindings.input_ready) != len(input_layouts)):
        raise ValueError("elementwise binding operand count mismatch")
    # Scalar maps can retain row/column coordinates and a padded row stride;
    # their scalar ABI still fixes total elements and each operand encoding.
    coordinate_shape=None
    if kernel.variant in ('elementwise','elementwise_pytorch_vec4'):
        from math import prod
        coordinate_shape=bindings.views[0].shape
    views = unique_by(bindings.views, "view_id")
    allocations = unique_by(bindings.allocations, "allocation_id")
    if kernel.variant == 'moe_abs' and ({views[name].allocation_id for name in bindings.read_views}
            & {views[name].allocation_id for name in bindings.write_views}):
        raise ValueError('MoE torch.abs creates a distinct INT32 routing-index output')
    if kernel.variant == 'moe_cast' and ({views[name].allocation_id for name in bindings.read_views}
            & {views[name].allocation_id for name in bindings.write_views}):
        raise ValueError('Expert-token integer widening requires a distinct output')
    for direction, names, operands in ((Operation.READ, bindings.read_views, input_layouts),
                                      (Operation.WRITE, bindings.write_views, output_layouts)):
        for name, (shape, dtype, width) in zip(names, operands):
            if name not in views:
                raise ValueError("elementwise operand view missing")
            view = views[name]
            shape_matches=(view.shape==shape or coordinate_shape is not None
                and view.shape==coordinate_shape and len(shape)==1 and prod(view.shape)==shape[0])
            if (not shape_matches
                    or view.dtype != dtype or view.element_bytes != width
                    or isinstance(view,TensorView) and view.strides_bytes[-1] < width):
                raise ValueError("elementwise binding storage ABI mismatch")
            if view.shape!=shape and isinstance(view,TensorView):
                span=width
                for dimension,stride in zip(reversed(view.shape),reversed(view.strides_bytes),strict=True):
                    if dimension>1 and stride<span:
                        raise ValueError('Scalar coordinate reshape cannot overlap logical elements')
                    span+=(dimension-1)*stride
            if kernel.variant == "dsv4_short_context_topk":
                strides = (8,) if direction is Operation.READ else (shape[1]*4,4)
                if view.strides_bytes != strides:
                    raise ValueError("short-context fill requires contiguous positions/output rows")
            if direction is Operation.READ and "input_materialization" in kernel.attributes:
                expected_strides = (kernel.attributes["input_group_stride_elements"] * width,
                                    kernel.attributes["input_row_stride_elements"] * width, width)
                if view.strides_bytes != expected_strides:
                    raise ValueError("grouped materialization input stride mismatch")
    from src.kernel.memory_access import region_end
    for output_name in bindings.write_views:
        output = views[output_name]
        for input_name in bindings.read_views:
            operand = views[input_name]
            if operand.allocation_id != output.allocation_id:
                continue
            if isinstance(operand,LogicalSelectionView) or isinstance(output,LogicalSelectionView):
                raise NotImplementedError('Scalar in-place alias needs a declared logical coordinate mapping')
            same_geometry = (operand.offset_bytes, operand.shape, operand.strides_bytes,
                             operand.element_bytes) == (output.offset_bytes, output.shape,
                                                        output.strides_bytes, output.element_bytes)
            if (not same_geometry and max(operand.offset_bytes, output.offset_bytes)
                    < min(region_end(operand.region()), region_end(output.region()))):
                raise NotImplementedError("partially overlapping scalar operands require a source-defined schedule")
    return views, allocations


def build_logical_elementwise(kernel, bindings, granularity, *, program_id):
    """One logical scalar pass with all input dependencies before publication."""
    views, allocations = _scalar_operand_bindings(kernel, bindings)
    op_ids = ('moe_cast','expert_tokens') if kernel.variant == 'moe_cast' else (kernel.variant,)
    if kernel.variant=='gemm_elementwise':op_ids=_gemm_elementwise_header(kernel)
    scope = AccessScope(f"{program_id}.scope", op_ids, bindings.call_id,
                        (), bindings.rank, bindings.stream, "logical-elementwise")
    accesses, dependencies = [], []
    abi = kernel.attributes["operation"] if kernel.variant in (
        "elementwise", "elementwise_pytorch_vec4") else kernel.variant
    evidence = (f"src/kernel/kernel_spec.py:VECTOR scalar/reducer ABI {abi}",)
    if kernel.variant == "dsv4_short_context_topk":
        evidence = ("vLLM0.30 deepseek_v4/attention.py:_fill_short_context_topk_indices int64 position load and masked full INT32 output row",)
    elif kernel.variant == 'moe_abs':
        evidence = ('vLLM-Ascend MoE public-call sorted_indices=torch.abs(expanded_row_idx)',
            'CANN8.5.1 Abs<int32> unary input/output interface; AIV tiling and transactions remain separate projections')
    elif kernel.variant == 'flash_mla_metadata':
        evidence = ('Legacy FlashMLA get_mla_metadata: INT32 cache_seqlens[B], scheduler[P,8], num_splits[B+1]',
            'Logical complete interface extents do not expose scheduler values, native scratch, padding or consumer aliases')
    elif kernel.variant == 'moe_transpose':
        evidence = ('Ascend fused_moe_runner._routing: arange(T*K,INT32).view(K,T).transpose(0,1).contiguous()',
            'Logical transposed source and contiguous destination; no compiler issue/transaction order or allocator-padding claim')
    elif kernel.variant == 'moe_cast':
        evidence = ('Ascend public expert-token INT32 metadata is converted to INT64 for GroupedMatmulV5, including legacy cumulative counts',
            'Existing source launch selects cast_role=expert_tokens; topk_ids dtype is not inferred from its role name')
    # A pointwise scalar coordinate is an exact algorithmic work unit, not a
    # claim about native CTA/vector issue shape. Split-K's unit is instead a
    # complete reduction over the external partials for one output coordinate.
    tiled_variants = ("elementwise", "elementwise_pytorch_vec4", "activation", "silu",
                      "moe_act_and_mul", "moe_swiglu", "gemm_reduce", "gemm_elementwise",
                      "moe_abs", "moe_transpose", "moe_cast")
    if kernel.variant in tiled_variants:
        program = build_scalar_tiles(kernel, bindings, views, scope, evidence, program_id=program_id)
        from src.kernel.memory_path_solver import scalar_tile_source_work
        scalar_tile_source_work(program)
        if kernel.variant=='gemm_elementwise':gemm_elementwise_source_work(program,expected_kernel=kernel,expected_bindings=bindings)
        if kernel.variant == "moe_transpose":cann_moe_transpose_source_demands(program)
        return program
    if kernel.variant == "softmax" and "rows" in kernel.shape:
        return build_softmax_row_tiles(kernel, bindings, scope, program_id=program_id)
    if kernel.variant == "softmax" and "rows" not in kernel.shape:
        return build_softmax_partition_tiles(kernel,bindings,scope,program_id=program_id)
    if kernel.variant == "dsv4_short_context_topk":
        return build_topk_fill_row_tiles(kernel, bindings, scope, evidence, program_id=program_id)
    if kernel.variant == 'flash_mla_metadata':
        from src.kernel.memory_flash_attention import build_flash_mla_metadata_tiles
        return build_flash_mla_metadata_tiles(kernel,bindings,scope,evidence,program_id=program_id)
    declared_events = unique_by(bindings.events,'event_id')
    for index, (name, ready) in enumerate(zip(bindings.read_views, bindings.input_ready)):
        view = views[name]
        region = view.region()
        if ready.kind is EventKind.LOGICAL_AVAILABLE:
            available = declared_events[ready.node_id]
            if (available.kind is not EventKind.LOGICAL_AVAILABLE or available.allocation_id != view.allocation_id
                    or not isinstance(available.region,Unmapped)
                    or (available.region.logical_shape,available.region.dtype,available.region.element_bytes)
                        != (view.shape,view.dtype,view.element_bytes)):
                raise ValueError('Scalar logical availability must match its actual typed input selection')
            region = available.region
        access_id = f"{program_id}.read{index}"
        accesses.append(MemoryAccess(access_id, scope.scope_id, view.allocation_id,None if isinstance(region,Unmapped) else name,
            Operation.READ,region, Semantics.LOGICAL, None, None, None, None,
            bindings.visibility_domain, (ready,), EventKind.READ_READY, 1, evidence))
        dependencies.append(EventRef(access_id, EventKind.READ_READY))
    compute = ProgramEvent(f"{program_id}.compute", EventKind.COMPUTE_DONE,
                           None, None, None, tuple(dependencies))
    for index, name in enumerate(bindings.write_views):
        view = views[name]
        if view.allocation_id not in allocations:
            raise ValueError("elementwise output allocation missing")
        accesses.append(MemoryAccess(f"{program_id}.write{index}", scope.scope_id,
            view.allocation_id, name, Operation.WRITE, view.region(), Semantics.LOGICAL,
            None, None, None, None, bindings.visibility_domain,
            (EventRef(compute.event_id, EventKind.COMPUTE_DONE),
             EventRef(allocations[view.allocation_id].allocated_event, EventKind.ALLOCATED)),
            EventKind.WRITE_VISIBLE, 1, evidence))
    extra_limits = (("Original token-row packing is unavailable; separate logical operands do not prove physical aliasing.",)
                    if kernel.variant in ("activation", "silu") and "tok" not in kernel.shape else ())
    if kernel.variant == "softmax":
        extra_limits += ("Logical softmax primitive interface; reduction axis, compiler passes and local accumulator storage are unavailable.",)
    if kernel.variant == "dsv4_short_context_topk":
        extra_limits += ("Short-context output includes sentinel entries; position values and selected indices remain unknown.",
                         "One logical position read per row; Triton lane issue/sector multiplicity and helper services are not inferred.")
    executable = (all(allocation.size_bytes is not None for allocation in bindings.allocations)
        and not any(isinstance(access.region,Unmapped) for access in accesses))
    if not executable:
        extra_limits += ('Unknown-capacity caller parents retain nonexecutable logical interface offsets; no native allocation bound is inferred.',)
    program = MemoryAccessProgram("midlayer.memory-access.v1", program_id, kernel.hardware_id,
        bindings.binding_id, Coverage(granularity, Semantics.LOGICAL, Ordering.PARTIAL, executable,
            extra_limits + ("Logical scalar ABI only; no issue order, cache traffic or physical transaction claim.",
             "Whole-view completion events do not specify a hardware-wide barrier.")),
        bindings.allocations, bindings.views, (scope,), bindings.events + (compute,), tuple(accesses))
    if kernel.variant == 'flash_mla_metadata':
        from src.kernel.memory_flash_attention import flash_mla_metadata_source_demands
        flash_mla_metadata_source_demands(program)
    if kernel.variant == 'moe_transpose':
        cann_moe_transpose_source_demands(program)
    return program



def build_topk_fill_row_tiles(kernel, bindings, scope, evidence, *, program_id):
    """The integer position authorizes the complete actual row, sentinels included."""
    from dataclasses import replace
    from src.kernel.memory_path_solver import short_context_topk_source_demands
    views = unique_by(bindings.views, "view_id")
    allocations = unique_by(bindings.allocations, "allocation_id")
    events = unique_by(bindings.events, "event_id")
    source, = bindings.read_views
    destination, = bindings.write_views
    ready, = bindings.input_ready
    read_unmapped = (allocations[views[source].allocation_id].size_bytes is None
        or isinstance(events[ready.node_id].region, Unmapped))
    write_unmapped = allocations[views[destination].allocation_id].size_bytes is None
    evidence += ("One independent query row: its INT64 position determines valid compressed indices and the full sentinel tail.",)
    accesses = (
        TileAccessTemplate("position", source, Operation.READ, ("row",), ("row",),
            (ready,), (), evidence, unmapped=read_unmapped),
        TileAccessTemplate("indices", destination, Operation.WRITE, ("row",), ("row", None),
            (), (TileDependency("position", EventKind.READ_READY, ()),), evidence, unmapped=write_unmapped),)
    scope = replace(scope, op_ids=("dsv4_short_context_topk",str(kernel.shape["compress_ratio"])), agent="logical-short-context-fill")
    family = TileProgram(program_id, scope, (TileAxis("row", kernel.shape["seq_len"], 1),),
        accesses, bindings.visibility_domain, evidence)
    program = MemoryAccessProgram("midlayer.memory-access.v1", program_id, kernel.hardware_id,
        bindings.binding_id, Coverage(Granularity.TILE, Semantics.LOGICAL, Ordering.PARTIAL,
            not (read_unmapped or write_unmapped), (
                "Full output row includes sentinel entries; actual position values and resulting indices remain unknown.",
                "Logical row work does not infer issued lane multiplicity, transactions, or compiler scratch.",)),
        bindings.allocations, bindings.views, (), bindings.events, (), (family,))
    short_context_topk_source_demands(program,expected_kernel=kernel)
    return program


def build_softmax_partition_tiles(kernel,bindings,scope, *, program_id):
    """Expose exact flat requests and the unbound within-group reduction."""
    from dataclasses import replace
    from src.kernel.memory_access import TilePartition
    from src.kernel.memory_path_solver import softmax_source_work
    source,=bindings.read_views;destination,=bindings.write_views;ready,=bindings.input_ready
    views=unique_by(bindings.views,'view_id')
    allocations=unique_by(bindings.allocations,'allocation_id');events=unique_by(bindings.events,'event_id')
    read_unmapped=allocations[views[source].allocation_id].size_bytes is None or isinstance(events[ready.node_id].region,Unmapped)
    write_unmapped=allocations[views[destination].allocation_id].size_bytes is None
    scope=replace(scope,agent='logical-softmax-partition')
    evidence=('VECTOR/softmax flat ABI declares element_count but not normalization rows or membership.',
        'Each output scalar requires the complete input equivalence class containing that scalar; class membership stays unbound.',
        'Inputs are read once in the logical interface and reused by their own group; native passes and scratch remain separate projections.')
    accesses=(TileAccessTemplate('input',source,Operation.READ,('element',),('element',),(ready,),(),evidence,unmapped=read_unmapped),
        TileAccessTemplate('output',destination,Operation.WRITE,('element',),('element',),(),
            (TileDependency('input',EventKind.READ_READY,('element',),partition_axis='element'),),evidence,unmapped=write_unmapped))
    family=TileProgram(program_id,scope,(TileAxis('element',kernel.shape['element_count'],1),),accesses,
        bindings.visibility_domain,evidence,partitions=(TilePartition('element'),))
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Flat normalization groups are unbound; dependency expansion and execution require actual membership.',
            'Logical input/output volumes are exact without assuming one global row, independent scalars, or equal row widths.',)),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))
    softmax_source_work(program)
    return program


def build_softmax_row_tiles(kernel, bindings, scope, *, program_id):
    """Each logical softmax work unit consumes and publishes one whole row."""
    from dataclasses import replace
    from src.kernel.memory_path_solver import softmax_row_tile_work
    scope = replace(scope, agent="logical-softmax-rows")
    evidence = (
        "VECTOR/softmax explicit rows/hidden ABI: each output row is normalized by its own complete input row.",
        "Eager caller preserves B*H*Q rows and KV columns; logical row work does not infer native CTA or compiler passes.",)
    views = unique_by(bindings.views, "view_id")
    allocations = unique_by(bindings.allocations, "allocation_id")
    events = unique_by(bindings.events, "event_id")
    source, = bindings.read_views
    destination, = bindings.write_views
    ready, = bindings.input_ready
    read_unmapped = (allocations[views[source].allocation_id].size_bytes is None
                     or isinstance(events[ready.node_id].region, Unmapped))
    write_unmapped = allocations[views[destination].allocation_id].size_bytes is None
    accesses = (
        TileAccessTemplate("input", source, Operation.READ, ("row",), ("row", None),
            (ready,), (), evidence, unmapped=read_unmapped),
        TileAccessTemplate("output", destination, Operation.WRITE, ("row",), ("row", None),
            (), (TileDependency("input", EventKind.READ_READY, ()),), evidence, unmapped=write_unmapped),)
    family = TileProgram(program_id, scope, (TileAxis("row", kernel.shape["rows"], 1),),
        accesses, bindings.visibility_domain, evidence)
    program = MemoryAccessProgram("midlayer.memory-access.v1", program_id, kernel.hardware_id,
        bindings.binding_id, Coverage(Granularity.TILE, Semantics.LOGICAL, Ordering.PARTIAL,
            not (read_unmapped or write_unmapped), (
                "Independent complete row reductions; intermediate maxima, exponentials and sums stay on chip in the logical interface.",
                "Native passes, local scratch, issue width and physical traffic remain separate implementation projections.",)),
        bindings.allocations, bindings.views, (), bindings.events, (), (family,))
    softmax_row_tile_work(program)
    return program


def _scalar_coordinate_views(kernel,views,reads,writes):
    if 'input_materialization' not in kernel.attributes:return views,writes
    from dataclasses import replace
    from math import prod
    indexed={view.view_id:view for view in views};shape=indexed[reads[0]].shape
    group,rows,columns=shape;extra=[]
    for name in writes:
        output=indexed[name]
        if output.shape!=(prod(shape),):raise ValueError('Grouped materialization must conserve output elements')
        stride=output.strides_bytes[0]
        extra.append(replace(output,view_id=name+'.coordinates',shape=shape,
            strides_bytes=(rows*columns*stride,columns*stride,stride)))
    return (*views,*extra),tuple(view.view_id for view in extra)


@runtime_lru_cache(maxsize=64)
def _scalar_access_patterns(names,reads,writes,input_ranks,reducer):
    accesses=[];dependencies=[]
    for index,(name,rank) in enumerate(zip(reads,input_ranks,strict=True)):
        mapping=(None,*names,*((None,) if rank==3 else ())) if reducer else names
        identity=f'read{index}'
        accesses.append(TileAccessTemplate(identity,name,Operation.READ,names,mapping,(),(),()))
        dependencies.append(TileDependency(identity,EventKind.READ_READY,()))
    for index,name in enumerate(writes):
        accesses.append(TileAccessTemplate(f'write{index}',name,Operation.WRITE,names,names,(),tuple(dependencies),()))
    return tuple(accesses)


def build_scalar_tiles(kernel, bindings, views, scope, evidence, *, program_id):
    """Compressed independent map coordinates or complete split-K reductions."""
    from dataclasses import replace
    coordinates,write_roles=_scalar_coordinate_views(kernel,bindings.views,bindings.read_views,bindings.write_views)
    if coordinates is not bindings.views:
        bindings=replace(bindings,views=coordinates,write_views=write_roles)
        evidence+=("Grouped materialization: exact affine group/row/column destination packing; source group gaps are preserved.",)
    views=unique_by(bindings.views,'view_id')
    shape=views[(bindings.write_views or bindings.read_views)[0]].shape
    axes=tuple(TileAxis(f'coordinate{index}',dimension,1) for index,dimension in enumerate(shape))
    names=tuple(axis.axis_id for axis in axes)
    evidence+=("Scalar source ABI: one independent output coordinate per semantic tile; no native CTA/issue-size claim.",)
    if kernel.variant=='gemm_reduce':
        evidence+=("External split-K partial tensor: each output coordinate consumes all split partials; accumulation stays on chip.",)
    allocations=unique_by(bindings.allocations,'allocation_id');events=unique_by(bindings.events,'event_id')
    patterns=_scalar_access_patterns(names,bindings.read_views,bindings.write_views,
        tuple(len(views[name].shape) for name in bindings.read_views),kernel.variant=='gemm_reduce')
    ready=dict(zip(bindings.read_views,bindings.input_ready,strict=True));accesses=[]
    for pattern in patterns:
        view=views[pattern.view_id]
        if kernel.variant=='gemm_reduce' and pattern.operation is Operation.READ:
            if view.shape[1:2]!=shape:raise ValueError('split-K scalar tile requires its actual partial/output coordinates')
        elif view.shape!=shape:raise ValueError('pointwise tile must conserve operand coordinates')
        unmapped=allocations[view.allocation_id].size_bytes is None
        if pattern.operation is Operation.READ:
            ref=ready[view.view_id]
            unmapped=unmapped or isinstance(events[ref.node_id].region,Unmapped)
            pattern=replace(pattern,input_ready=(ref,))
        accesses.append(replace(pattern,evidence_refs=evidence,unmapped=unmapped))
    executable = not any(access.unmapped for access in accesses)
    limitations = ("Compressed independent scalar work units; physical vector width, cache traffic and issue order are separate projections.",)
    if kernel.variant in ("activation", "silu") and "tok" not in kernel.shape:
        limitations += ("Original token-row packing is unavailable; separate logical operands do not prove physical aliasing.",)
    family = TileProgram(program_id, scope, axes, tuple(accesses), bindings.visibility_domain, evidence)
    return MemoryAccessProgram("midlayer.memory-access.v1", program_id, kernel.hardware_id,
        bindings.binding_id, Coverage(Granularity.TILE, Semantics.LOGICAL, Ordering.PARTIAL, executable, limitations),
        bindings.allocations, bindings.views, (), bindings.events, (), (family,))


def elementwise_memory_access_program(kernel, hardware):
    return build_memory_access_program(kernel, hardware, None, Granularity.KERNEL,
                                       program_id="vector.elementwise")


def has_scalar_analytical_work(kernel):
    from src.kernel.kernel_spec import KernelKind
    return (kernel.kind is KernelKind.VECTOR and kernel.execution_support == 'supported'
        and kernel.variant in ('elementwise','elementwise_pytorch_vec4','activation','silu',
            'moe_act_and_mul','moe_swiglu','gemm_reduce','gemm_elementwise')
        and not has_observed_router_reducer_abi(kernel)
        and not _has_external_aggregate_interface(kernel))


def scalar_prediction_work(kernel, hardware, bindings=None):
    """Consume the scalar source ABI without constructing its export tile DAG."""
    from math import prod
    from src.kernel.memory_access import validate_storage_bindings, validate_input_readiness
    if kernel.hardware_id != hardware.hardware_id or not has_scalar_analytical_work(kernel):
        raise ValueError('Scalar analytical work requires its supported kernel/hardware ABI')
    actual, _ = _prediction_bindings(kernel, bindings)
    execution = memory_execution_kernel(kernel, hardware)
    if actual is None:
        parents,operands,_,_=scalar_operand_records(execution,binding_id='scalar.analytical',device_id='logical-local-device')
        actual=MemoryInvocation(kernel,execution,'scalar.analytical','scalar.analytical',parents,operands,0,'offline',
            'elementwise-visible',False)
        actual = _own_implicit_memory_source(kernel, actual)
    if isinstance(actual,MemoryInvocation):
        if kernel!=actual.kernel and kernel!=actual.operand_source:
            raise ValueError('Scalar consumer crossed its declared execution transform')
        work=actual.prediction_work(actual.kernel,hardware)
        from src.kernel.memory_path_solver import scalar_tile_source_work
        scalar_tile_source_work(work)
        return _scalar_work_projection(execution,{v.view_id:v for v in actual.views},actual.read_views,actual.write_views)
    views, allocations = _scalar_operand_bindings(execution, actual)
    actual.__post_init__()
    events = unique_by(actual.events,'event_id')
    if any(view.allocation_id not in allocations for view in actual.views):
        raise ValueError('unknown view allocation')
    if any(ready.node_id not in events for ready in actual.input_ready):
        raise ValueError('Scalar input readiness is missing')
    # Match the scalar export's executable envelope. Unknown records outside
    # its actual roles cannot silently turn a resolved source nonexecutable.
    executable=not (any(allocations[views[role].allocation_id].size_bytes is None
        or isinstance(events[ready.node_id].region,Unmapped)
        for role,ready in zip(actual.read_views,actual.input_ready,strict=True))
        or any(allocations[views[role].allocation_id].size_bytes is None for role in actual.write_views))
    validate_storage_bindings(actual.allocations,actual.views,actual.events,
        Coverage(Granularity.KERNEL,Semantics.LOGICAL,Ordering.PARTIAL,executable,
            ('Scalar tensor ABI; independent coordinate dependencies are materialized only for explicit export.',)))
    if any(event.tile_sources for event in actual.events):
        raise ValueError('Independent scalar input cannot depend on its own export tile writer')
    for role, ready in zip(actual.read_views,actual.input_ready,strict=True):
        view=views[role]
        unmapped=(allocations[view.allocation_id].size_bytes is None
            or isinstance(events[ready.node_id].region,Unmapped))
        validate_input_readiness(view,ready,unmapped=unmapped,views=views,
            events=events,visibility_domain=actual.visibility_domain)
    return _scalar_work_projection(execution,views,actual.read_views,actual.write_views)


def _scalar_work_projection(execution,views,read_roles,write_roles):
    from math import prod
    reads=tuple(views[name] for name in read_roles)
    writes=tuple(views[name] for name in write_roles)
    shape=(writes or reads)[0].shape
    if execution.variant=='gemm_reduce':
        if any(view.shape[1:2]!=shape for view in reads):
            raise ValueError('Scalar reducer must retain its complete partial/output coordinates')
    elif 'input_materialization' in execution.attributes:
        source_shape=reads[0].shape
        if (len(source_shape)!=3 or any(view.shape!=source_shape for view in reads)
                or any(view.shape!=(prod(source_shape),) for view in writes)):
            raise ValueError('Grouped materialization must conserve output elements')
    elif any(view.shape!=shape for view in (*reads,*writes)):
        raise ValueError('Scalar work must conserve every operand coordinate')
    read=sum(prod(view.shape)*view.element_bytes for view in reads)
    write=sum(prod(view.shape)*view.element_bytes for view in writes)
    logical, executed=(_gemm_elementwise_extents(execution) if execution.variant=='gemm_elementwise'
        else (prod(shape),prod(shape)))
    header=_gemm_elementwise_header(execution) if execution.variant=='gemm_elementwise' else (execution.variant,)
    operands=(*reads,*writes)
    layouts=tuple((view.shape,view.dtype,view.element_bytes,
        ('tensor',view.offset_bytes,view.strides_bytes) if isinstance(view,TensorView)
        else ('selection',view.coordinate_map)) for view in operands)
    aliases=tuple(tuple(a.allocation_id==b.allocation_id for b in operands) for a in operands)
    return {'read_bytes':read,'write_bytes':write,'logical_elements':logical,
        'legacy_read_bytes':read*logical//executed,'legacy_write_bytes':write*logical//executed,
        'numeric_projection':('scalar-source',header,shape,read,write,layouts,aliases)}


def gemm_aux_memory_access_program(kernel, hardware):
    return build_memory_access_program(kernel, hardware, None, Granularity.KERNEL,
                                       program_id="vector.gemm_aux")


def gemm_elementwise_source_work(program,*,expected_kernel,expected_bindings=None):
    """Actual scalar replay requests and the retained logical-coordinate price."""
    from src.kernel.kernel_spec import KernelKind
    from src.kernel.memory_path_solver import scalar_tile_source_work
    kernel=expected_kernel;work=scalar_tile_source_work(program);family,=program.tile_programs
    if (kernel.kind is not KernelKind.VECTOR or kernel.variant!='gemm_elementwise'
            or kernel.execution_support!='supported' or kernel.hardware_id!=program.hardware_id
            or family.scope.op_ids!=_gemm_elementwise_header(kernel)):
        raise ValueError('GEMM epilogue source differs from its actual kernel/replay mapping')
    logical,execution=_gemm_elementwise_extents(kernel)
    if (work['shape']!=(execution,) or len(work['reads'])!=1 or len(work['writes'])!=1
            or any(not isinstance(view,(TensorView,LogicalSelectionView)) or (view.dtype,view.element_bytes)!=(kernel.dtype,kernel.shape['dtype_bytes'])
                for view in (*work['reads'],*work['writes']))):
        raise ValueError('GEMM epilogue requires its full typed replay input/output')
    if expected_bindings is not None:
        b=expected_bindings
        if isinstance(b,MemoryInvocation):b=b.materialized
        if (not isinstance(b,ElementwiseBindings) or program.binding_id!=b.binding_id
                or (family.scope.call_id,family.scope.rank,family.scope.stream,family.visibility_domain)!=(b.call_id,b.rank,b.stream,b.visibility_domain)
                or unique_by(program.allocations,'allocation_id')!=unique_by(b.allocations,'allocation_id')
                or unique_by(program.views,'view_id')!=unique_by(b.views,'view_id')
                or unique_by(program.events,'event_id')!=unique_by(b.events,'event_id')
                or family.accesses[0].input_ready!=b.input_ready):
            raise ValueError('GEMM epilogue source differs from actual caller bindings/readiness')
    return {'logical_elements':logical,'execution_elements':execution,
        'read_bytes':work['read_bytes'],'write_bytes':work['write_bytes'],
        'legacy_read_bytes':work['read_bytes']*logical//execution,'legacy_write_bytes':work['write_bytes']*logical//execution}


@dataclass(frozen=True)
class TransferBindings(Typed):
    """Caller-declared allocation identities for the byte-copy ABI."""
    binding_id: str
    call_id: str
    allocations: tuple[MemoryAllocation, ...]
    views: tuple[TensorView, ...]
    events: tuple[ProgramEvent, ...]
    source_view: str | None
    destination_view: str | None
    source_ready: EventRef | None
    rank: int
    stream: str
    visibility_domain: str


def _transfer_storage_domains(kernel):
    from src.kernel.kernel_spec import KernelKind
    if kernel.kind is KernelKind.MEMCPY:
        if kernel.transfer.algorithm == "fpga_staging":
            domains = {
                "network_to_ddr": ("network-payload", "fpga-ddr-storage"),
                "ddr_to_buffer": ("fpga-ddr-storage", "fpga-buffer-storage"),
                "hbm_to_ddr": ("device-storage", "fpga-ddr-storage"),
                "ddr_to_network": ("fpga-ddr-storage", "network-payload"),
            }
            return domains[kernel.transfer.direction]
        if kernel.transfer.algorithm == "stream_copy_pinned":
            host = "host-pinned-storage"
        elif kernel.transfer.algorithm == "stream_copy_pageable":
            host = "host-pageable-storage"
        else:
            host = "host-storage"
        if kernel.transfer.direction in ("d2d", "dtod", "device_to_device"):
            return "device-storage", "device-storage"
        if kernel.transfer.direction in ("h2d", "host_to_device"):
            return host, "device-storage"
        if kernel.transfer.direction in ("d2h", "device_to_host"):
            return "device-storage", host
        raise ValueError("unknown MEMCPY direction")
    if kernel.kind is KernelKind.KV_OFFLOAD:
        tier = f"kv-tier:{kernel.transfer.tier}"
        if kernel.transfer.direction == "load":
            return tier, "device-storage"
        if kernel.transfer.direction == "evict":
            return "device-storage", tier
        raise ValueError("unknown KV_OFFLOAD direction")
    raise TypeError("byte-copy storage domains require MEMCPY or KV_OFFLOAD")


def standalone_transfer_bindings(kernel, *, call_id: str, binding_id: str):
    """Invocation-local object bindings, without persistent identity claims."""
    source_domain, destination_domain = _transfer_storage_domains(kernel)
    byte_count = kernel.transfer.bytes
    allocations, views, events = [], [], []
    if byte_count:
        for name, domain in (("source", source_domain), ("destination", destination_domain)):
            allocation_id = f"{binding_id}.{name}"
            allocated = f"{allocation_id}.allocated"
            device_id = "logical-local-device" if domain == "device-storage" else f"logical-{domain}"
            allocations.append(MemoryAllocation(allocation_id, device_id, domain,
                byte_count, AddressLevel.OBJECT, None, None, allocated, None))
            views.append(TensorView(name, allocation_id, 0, (byte_count,), (1,), "uint8", 1))
            events.append(ProgramEvent(allocated, EventKind.ALLOCATED, allocation_id, None, None, ()))
        events.append(ProgramEvent(f"{binding_id}.source.initialized", EventKind.INITIALIZED,
            f"{binding_id}.source", Span(0, byte_count), "copy-visible",
            (EventRef(f"{binding_id}.source.allocated", EventKind.ALLOCATED),)))
    return TransferBindings(binding_id, call_id, tuple(allocations), tuple(views), tuple(events),
        "source" if byte_count else None, "destination" if byte_count else None,
        EventRef(f"{binding_id}.source.initialized", EventKind.INITIALIZED) if byte_count else None,
        0, "analytical-copy", "copy-visible")


def build_logical_transfer(kernel, bindings: TransferBindings, granularity: Granularity,
                           *, program_id: str):
    """Byte-copy dataflow from the explicit PhysicalTransferConfig payload.

    This does not call a predictor or turn link traffic into an address trace.
    Every copied byte is read at the source and made visible at the destination;
    bus service counts that payload once, independently of the storage ledger.
    """
    from src.kernel.memory_access import check_type
    if not isinstance(bindings, TransferBindings):
        raise TypeError("byte copies require TransferBindings")
    check_type(granularity, Granularity)
    if granularity not in (Granularity.OP, Granularity.KERNEL, Granularity.TILE):
        raise ValueError("unknown byte-copy logical granularity")
    source_domain, destination_domain = _transfer_storage_domains(kernel)
    byte_count = kernel.transfer.bytes
    scopes = (AccessScope(f"{program_id}.copy", (kernel.kind.value,), bindings.call_id,
                          (), bindings.rank, bindings.stream, f"{program_id}.payload"),)
    families = ()
    if byte_count == 0:
        if (bindings.allocations or bindings.views or bindings.events
                or bindings.source_view is not None or bindings.destination_view is not None
                or bindings.source_ready is not None):
            raise ValueError("zero-payload copy must not fabricate storage or accesses")
    else:
        views = unique_by(bindings.views, "view_id")
        allocations = unique_by(bindings.allocations, "allocation_id")
        if bindings.source_view not in views or bindings.destination_view not in views:
            raise ValueError("copy endpoint view missing")
        source, destination = views[bindings.source_view], views[bindings.destination_view]
        for view, domain in ((source, source_domain), (destination, destination_domain)):
            if (view.shape != (byte_count,) or view.strides_bytes != (1,)
                    or view.dtype != "uint8" or view.element_bytes != 1):
                raise ValueError("copy endpoint view must match the byte payload")
            if view.allocation_id not in allocations:
                raise ValueError("copy endpoint allocation missing")
            if allocations[view.allocation_id].memory_domain != domain:
                raise ValueError("copy endpoint storage domain disagrees with direction/tier")
        if bindings.source_ready is None:
            raise ValueError("copy requires source readiness")
        if (source.allocation_id == destination.allocation_id
                and max(source.offset_bytes, destination.offset_bytes)
                < min(source.offset_bytes + byte_count, destination.offset_bytes + byte_count)):
            raise NotImplementedError("overlapping byte copy needs a source-declared in-place algorithm")
        evidence = ("PhysicalTransferConfig byte-copy ABI: independent byte coordinates preserve exact endpoint offsets and per-byte read-before-publication.",
                    "Semantic bytes are not inferred DMA descriptors, packets, sectors or native issued work blocks.")
        read = TileAccessTemplate('source',source.view_id,Operation.READ,('byte',),('byte',),
            (bindings.source_ready,),(),evidence)
        write = TileAccessTemplate('destination',destination.view_id,Operation.WRITE,('byte',),('byte',),(),
            (TileDependency('source',EventKind.READ_READY,()),),evidence)
        families = (TileProgram(program_id,scopes[0],(TileAxis('byte',byte_count,1),),(read,write),
            bindings.visibility_domain,evidence),)
    return MemoryAccessProgram("midlayer.memory-access.v1", program_id, kernel.hardware_id,
        bindings.binding_id, Coverage(Granularity.TILE, Semantics.LOGICAL, Ordering.PARTIAL, True,
            ("Logical byte-copy ABI; no DMA issue order, cache or transaction geometry claim.",)),
        bindings.allocations, bindings.views, () if families else scopes, bindings.events, (), families)


def transfer_memory_access_program(kernel, hardware):
    """Canonical source consumed by the default transfer predictors."""
    return build_memory_access_program(kernel, hardware, None, Granularity.KERNEL,
                                       program_id=f"{kernel.kind.value}.payload")


def has_gpu_nonsplit_gemm_source(kernel, hardware):
    """Migration domain for ordinary unquantized single-output GPU GEMM."""
    return hardware.kernel_memory_model == 'gpu_mshr' and has_gpu_nonsplit_gemm_abi(kernel)


def has_gpu_nonsplit_gemm_abi(kernel):
    """Declared matrix interface, also usable before Serving binds hardware."""
    if kernel.execution_support != "supported" or kernel.launch is None:
        return False
    from src.kernel.kernel_spec import KernelKind
    router=(kernel.kind is KernelKind.GEMM and kernel.variant == "moe_router"
        and kernel.hardware_id in ("nvidia_a800","nvidia_h200") and kernel.dtype in ("bf16","fp32")
        and has_router_replay_mapping(kernel)
        and kernel.attributes["weight_dtype"] == "bf16" and kernel.attributes["weight_group_size"] == 0
        and kernel.attributes["output_dtype"] in (None,kernel.dtype)
        and kernel.launch.algorithm in ("cublas_analytical_gemm:moe_router","nvidia_cublaslt_gemm:moe_router"))
    return (kernel.kind is KernelKind.GEMM and (kernel.variant in ("gemm", "batched_gemm") or router)
        and (kernel.variant != "batched_gemm" or kernel.attributes["weight_dtype"] == "bf16")
        and kernel.dtype in ("bf16", "fp16", "fp32")
        and (router or kernel.launch.algorithm in ("nvidia_cublaslt_gemm", "cublas_analytical_gemm",
                                        "metax_mcblas_gemm", "mcblas_analytical_gemm", "mcblas_injected_gemm:gemm",
                                        "cublas_analytical_gemm:batched_gemm", "nvidia_cublaslt_gemm:batched_gemm"))
        and kernel.launch.tiling is not None and kernel.launch.tiling.split_k == 1
        and kernel.launch.parameters["split_k_partial_bytes_per_element"] == 0
        and kernel.launch.parameters["split_k_partial_dram_bytes_per_element"] == 0)


def has_moe_abs_source(kernel,hardware):
    from src.kernel.kernel_spec import KernelKind
    return (kernel.kind is KernelKind.VECTOR and kernel.variant == 'moe_abs'
        and kernel.execution_support == 'supported' and hardware.kernel_memory_model == 'npu_tile')


def has_default_memory_source(kernel, hardware):
    """Explicit migration domain; this is not a production support downgrade."""
    from src.kernel.kernel_spec import KernelKind
    if has_mathematical_gemm_source(kernel) or has_mathematical_memory_source(kernel):
        return True
    if kernel.execution_support != "supported":
        return False
    if has_logical_interface_source(kernel,hardware):
        return True
    return _has_native_memory_source(kernel,hardware)


def _has_native_memory_source(kernel,hardware):
    """Native admission after the public logical interface was excluded."""
    from src.kernel.kernel_spec import KernelKind
    if kernel.kind is KernelKind.FUSED_MOE:
        return has_routed_moe_source(kernel,hardware) or has_fp8_fused_moe_source(kernel) or has_fused_moe_source(kernel,hardware)
    if kernel.kind is KernelKind.SPARSE_ATTN:
        return has_sparse_attention_source(kernel)
    if kernel.kind is KernelKind.SPARSE_ATTN_INDEXER:
        return has_sparse_indexer_source(kernel)
    if kernel.kind is KernelKind.UNFUSED_MOE:
        return has_unfused_moe_source(kernel, hardware)
    if kernel.kind is KernelKind.COMMUNICATION:
        return has_communication_source(kernel)
    if kernel.kind is KernelKind.FLASH_ATTN:
        from src.kernel.memory_cann_mla import has_cann_mla_source
        return has_cann_mla_source(kernel) or has_fa3_combine_source(kernel) or has_fa3_prefill_source(kernel) or has_cann_tnd_source(kernel) or has_metax_varlen_source(kernel) or has_cann_attention_source(kernel) or has_fa2_split_source(kernel) or has_grouped_mla_source(kernel) or has_metax_mla_source(kernel) or has_dense_flash_attention_source(kernel)
    if kernel.kind is KernelKind.EAGER_ATTN:
        return has_eager_attention_source(kernel, hardware)
    if kernel.kind is KernelKind.KV_CACHE:
        return has_target_cache_source(kernel) or has_paged_kv_source(kernel, hardware) or has_tensor_only_kv_source(kernel,hardware)
    if kernel.kind in (KernelKind.MEMCPY, KernelKind.KV_OFFLOAD):
        return True
    if kernel.kind is KernelKind.VECTOR:
        if has_public_normalization_source(kernel,hardware):return True
        if has_cann_moe_range_source(kernel,hardware):
            return True
        if has_cann_expert_tokens_source(kernel,hardware):
            return True
        if has_cann_moe_transpose_source(kernel,hardware):
            return True
        if has_cann_moe_init_source(kernel,hardware):
            return True
        if has_cann_moe_expert_cast_source(kernel,hardware):
            return True
        if has_cann_moe_unbound_cast_source(kernel,hardware):return True
        if has_cann_moe_gating_source(kernel,hardware):
            return True
        if has_cann_moe_finalize_source(kernel,hardware):
            return True
        if has_moe_fill_source(kernel,hardware):
            return True
        if has_moe_alignment_source(kernel,hardware):return True
        if has_moe_count_sort_source(kernel,hardware):return True
        if has_gpu_topk_source(kernel,hardware) or has_moe_metadata_source(kernel,hardware) or has_mhc_source(kernel,hardware) or has_generic_reduction_source(kernel,hardware) or has_v32_fused_q_source(kernel,hardware) or has_v32_fused_norm_source(kernel,hardware):return True
        if has_route_weight_source(kernel,hardware):return True
        if has_moe_reduction_source(kernel,hardware):
            return True
        if has_moe_pointwise_source(kernel,hardware):
            return True
        if has_cann_rotary_source(kernel,hardware):
            return True
        if has_tensor_only_rotary_source(kernel,hardware):
            return True
        if has_gpu_rotary_source(kernel,hardware):
            return True
        if has_moe_abs_source(kernel,hardware):
            return True
        return has_fp8_group_quant_source(kernel) or has_pairnorm_source(kernel,hardware) or has_layernorm_source(kernel,hardware) or has_cann_normalization_source(kernel,hardware) or has_pim_normalization_source(kernel,hardware) or has_gpu_normalization_source(kernel, hardware) or kernel.variant in ("flash_mla_metadata", "elementwise", "elementwise_pytorch_vec4", "gemm_reduce", "gemm_elementwise", "activation", "silu", "moe_act_and_mul", "moe_swiglu", "softmax", "dsv4_short_context_topk")
    if kernel.kind is KernelKind.GEMM:
        if has_public_router_abi(kernel):return True
        if kernel.hardware_id=='ascend_910b':kernel=memory_execution_kernel(kernel,hardware)
        return (has_observed_router_abi(kernel) or has_simt_router_source(kernel,hardware) or has_gpu_split_gemm_source(kernel,hardware) or has_cann_gemm_source(kernel) or has_cann_single_core_gemm_source(kernel) or has_cann_deterministic_gemm_source(kernel) or has_block_fp8_source(kernel) or has_gpu_nonsplit_gemm_source(kernel, hardware)
                or (kernel.attributes["output_dtype"] is not None
                 and kernel.attributes["output_dtype"] != kernel.dtype)
                or kernel.variant == "gemv"
                or (kernel.variant == "gemm" and hardware.kernel_memory_model in (
                    "pim_bank", "systolic_scratchpad")))
    return False


@dataclass(frozen=True,eq=False)
class MemorySourceAdmission:
    """A checked source domain for one exact call, not a source/result cache."""
    kernel: object
    hardware: object

    def __post_init__(self):
        from src.kernel.kernel_spec import KernelSpec
        from src.kernel.hardware_spec import HWSpec
        if not isinstance(self.kernel,KernelSpec) or not isinstance(self.hardware,HWSpec):
            raise TypeError('Memory admission requires KernelSpec/HWSpec')
        logical=(self.kernel.execution_support=='supported'
            and has_logical_interface_source(self.kernel,self.hardware))
        supported=has_mathematical_gemm_source(self.kernel) or has_mathematical_memory_source(self.kernel) or logical or (self.kernel.execution_support=='supported'
            and _has_native_memory_source(self.kernel,self.hardware))
        object.__setattr__(self,'supported',supported)
        object.__setattr__(self,'logical',logical and not has_fa3_prefill_source(self.kernel))
        object.__setattr__(self,'_seal',(self.kernel,self.hardware,self.supported,self.logical))
        object.__setattr__(self,'_recipe',(
            has_logical_interface_source,has_logical_interface_source.__code__,
            _has_native_memory_source,_has_native_memory_source.__code__))

    def assert_source(self,kernel,hardware):
        if (kernel is not self.kernel or hardware is not self.hardware
                or any(getattr(self,name) is not value for name,value in zip(
                    ('kernel','hardware','supported','logical'),self._seal,strict=True))):
            raise ValueError('Memory source admission crossed or changed its call boundary')
        logical,logical_code,native,native_code=self._recipe
        if (logical is not has_logical_interface_source or logical_code is not has_logical_interface_source.__code__
                or native is not _has_native_memory_source or native_code is not _has_native_memory_source.__code__):
            raise ValueError('Memory source admission recipe changed')


def memory_execution_kernel(kernel, hardware):
    """Resolve exact captured extents for an immutable spec/hardware pair."""
    context=copy_context()
    if _PREDICTION_MEMORY_PROGRAMS not in context:
        return _memory_execution_kernel(kernel,hardware)
    prepared=context[_PREDICTION_MEMORY_PROGRAMS][1]
    identity=(id(kernel),id(hardware))
    if identity in prepared:
        prior_kernel,prior_hardware,execution=prepared[identity]
        if kernel is not prior_kernel or hardware is not prior_hardware:
            raise RuntimeError('prediction execution-spec identity collision')
        prepared.move_to_end(identity)
        return execution
    execution=_memory_execution_kernel(kernel,hardware)
    prepared[identity]=(kernel,hardware,execution)
    if len(prepared)>context[_PREDICTION_MEMORY_PROGRAMS][2]:prepared.popitem(last=False)
    return execution


def _memory_execution_kernel(kernel, hardware):
    """Resolve declared captured GEMM operands and existing vector extents."""
    from src.kernel.kernel_spec import KernelKind
    if kernel.kind is KernelKind.GEMM and kernel.hardware_id=='ascend_910b':
        from src.kernel.memory_cann_gemm import cann_split_execution_kernel
        kernel=cann_split_execution_kernel(kernel)
    if (kernel.kind is KernelKind.GEMM and kernel.variant in ("gemm","moe_router","batched_gemm")
            and (has_gpu_nonsplit_gemm_source(kernel, hardware) or has_block_fp8_source(kernel)
                 or has_gpu_split_gemm_source(kernel, hardware)
                 or (kernel.hardware_id == "nvidia_h200"
                     and kernel.launch is not None
                     and kernel.launch.algorithm == "h200_cublaslt129_injected_gemm"))
            and "model_input_shape" in kernel.attributes
            and kernel.attributes["graph_runtime_mode"] in ("FULL", "PIECEWISE")
            and kernel.attributes["graph_logical_tokens"] < kernel.attributes["graph_model_input_tokens"]):
        from dataclasses import replace
        physical = kernel.attributes["model_input_shape"]
        if not {"b", "m", "n", "k"} <= set(physical):
            # This source subdomain has no complete capture descriptor.
            # Preserve its declared logical interface and existing support.
            return kernel
        if any(physical[axis] != kernel.shape[axis] for axis in ("b", "n", "k")):
            # Other graph mappings need a separate operand-layout contract.
            return kernel
        if physical["m"] < kernel.shape["m"]:
            raise ValueError("captured GEMM source cannot shrink its logical row extent")
        # CUDA graph replay executes the captured BLAS launch at the same
        # input/output addresses. Logical request rows do not shrink those
        # tensor operands; padding values are unknown, never assumed zero.
        attributes = dict(kernel.attributes)
        attributes["graph_logical_tokens"] = attributes["graph_model_input_tokens"]
        if kernel.variant=='batched_gemm':
            # The existing BF16 ABI is dense strided batches. Rebind its
            # execution descriptor to the captured row extent, leaving the
            # original logical KernelSpec and selected launch untouched.
            attributes['a_batch_stride_elements']=physical['m']*physical['k']
            attributes['b_batch_stride_elements']=physical['k']*physical['n']
            attributes['c_batch_stride_elements']=physical['m']*physical['n']
        return replace(kernel, shape=dict(physical), attributes=attributes)
    if kernel.kind is KernelKind.VECTOR:
        from src.kernel.predictors.vector import (
            _gpu_full_graph_source_program_execution_kernel,
            _metax_full_graph_norm_execution_kernel,
            _cann_full_graph_source_program_execution_kernel,
        )
        kernel = _gpu_full_graph_source_program_execution_kernel(kernel, hardware)
        kernel = _metax_full_graph_norm_execution_kernel(kernel, hardware)
        kernel = _cann_full_graph_source_program_execution_kernel(kernel, hardware)
        from src.kernel.memory_cann_moe_routing import cann_transpose_execution_kernel
        kernel=cann_transpose_execution_kernel(kernel)
    return kernel


def batched_gemm_logical_price_program(program,*,expected_kernel):
    """Keep the prior logical-row charge as a projection of full replay.

    The exported/bound source covers every captured row. This restricted
    source is only the existing analytical pricing domain, with the actual
    parent strides and capacities retained rather than a second byte formula.
    """
    from dataclasses import replace
    from src.kernel.memory_access import unique_by
    from src.kernel.memory_path_solver import logical_gemm_operand_geometry
    kernel=expected_kernel
    if kernel.variant!='batched_gemm':raise ValueError('Batched GEMM price projection requires its actual variant')
    work=logical_gemm_operand_geometry(program,batched=True)
    b,m,n,k=(kernel.shape[name] for name in ('b','m','n','k'))
    if (work[0],work[2],work[3])!=(b,n,k):raise ValueError('Batched GEMM price differs from its execution source geometry')
    if work[1]==m:return program
    if ('model_input_shape' not in kernel.attributes or kernel.attributes['graph_runtime_mode'] not in ('FULL','PIECEWISE')
            or kernel.attributes['model_input_shape']['m']!=work[1] or work[1]<m):
        raise ValueError('Batched GEMM price requires a declared complete replay-row mapping')
    if isinstance(program,MemoryAccessWork):
        return replace(program,axes=tuple(replace(axis,extent=m) if axis.axis_id=='m' else axis for axis in program.axes),
            logical_rows=m)
    views=unique_by(program.views,'view_id');family,=program.tile_programs
    roles={access.access_id:views[access.view_id] for access in family.accesses}
    updates={roles[name].view_id:replace(roles[name],shape=(b,m,roles[name].shape[2])) for name in ('a','c')}
    projected=replace(program,views=tuple(updates[v.view_id] if v.view_id in updates else v for v in program.views),
        tile_programs=(replace(family,axes=tuple(replace(axis,extent=m) if axis.axis_id=='m' else axis for axis in family.axes)),),
        coverage=replace(program.coverage,limitations=(*program.coverage.limitations,
            'Logical-row pricing projection only; complete captured batch storage and requests are exported separately.')))
    if logical_gemm_operand_geometry(projected,batched=True)!=(b,m,n,k,*work[4:]):
        raise ValueError('Batched GEMM logical-row projection lost its actual typed operands')
    return projected


def default_memory_bindings(kernel, hardware, *, call_id, binding_id, rank, stream, device_id=None, analytical=False,
                            _admission=None):
    """Caller-scoped bindings compiled from the same real source ABI."""
    from dataclasses import replace
    from src.kernel.kernel_spec import KernelKind
    if type(analytical) is not bool:raise TypeError('Memory source representation requires a boolean request')
    if device_id is not None and (type(device_id) is not str or not device_id.strip()):
        raise ValueError("binding device requires a nonempty identity")
    admission=MemorySourceAdmission(kernel,hardware) if _admission is None else _admission
    if not isinstance(admission,MemorySourceAdmission):raise TypeError('Memory source requires typed admission')
    admission.assert_source(kernel,hardware)
    target_device="logical-local-device" if device_id is None else device_id
    if has_mathematical_gemm_source(kernel):
        return standalone_bindings(_gemm_work_from_kernel(kernel), call_id=call_id, binding_id=binding_id,
            device_id=target_device, visibility_domain="mathematical-gemm", rank=rank, stream=stream)
    if has_mathematical_memory_source(kernel):
        return standalone_mathematical_bindings(kernel, call_id=call_id, binding_id=binding_id,
            rank=rank, stream=stream, device_id=target_device)
    if admission.logical:
        if kernel.hardware_id != hardware.hardware_id:
            raise ValueError('Logical interface hardware identity mismatch')
        if analytical:
            return logical_memory_invocation(kernel,hardware,binding_id=binding_id,call_id=call_id,
                device_id=target_device,rank=rank,stream=stream)
        return standalone_logical_interface_bindings(kernel,hardware=hardware,call_id=call_id,binding_id=binding_id,
            device_id=target_device,rank=rank,stream=stream)
    if not admission.supported:
        raise NotImplementedError("production memory source migration is pending for this domain")
    kernel = memory_execution_kernel(kernel, hardware)
    if kernel.kind is KernelKind.FUSED_MOE:
        from src.kernel.memory_fused_moe import has_native_grouped_moe_analytical_work,native_moe_operand_records
        if analytical and has_native_grouped_moe_analytical_work(kernel,hardware):
            header,parents,views=native_moe_operand_records(kernel,binding_id=binding_id,device_id=target_device)
            return MemoryInvocation(kernel,header,binding_id,call_id,parents,views,rank,stream,'moe-visible',False)
        if has_routed_moe_source(kernel,hardware):
            binding = standalone_routed_moe_bindings(kernel,call_id=call_id,binding_id=binding_id)
        else:
            binding = (standalone_fp8_fused_bindings(kernel,call_id=call_id,binding_id=binding_id)
                       if has_fp8_fused_moe_source(kernel) else standalone_fused_moe_bindings(kernel,hardware,call_id=call_id,binding_id=binding_id))
    elif kernel.kind is KernelKind.SPARSE_ATTN:
        binding = standalone_sparse_attention_bindings(kernel, call_id=call_id, binding_id=binding_id)
    elif kernel.kind is KernelKind.SPARSE_ATTN_INDEXER:
        binding = standalone_sparse_indexer_bindings(kernel, call_id=call_id, binding_id=binding_id)
    elif kernel.kind is KernelKind.UNFUSED_MOE:
        binding = standalone_unfused_moe_bindings(kernel, call_id=call_id, binding_id=binding_id)
    elif has_fp8_group_quant_source(kernel):
        binding = standalone_fp8_group_quant_bindings(kernel, call_id=call_id, binding_id=binding_id)
    elif has_block_fp8_source(kernel):
        binding = standalone_block_fp8_bindings(kernel, call_id=call_id, binding_id=binding_id)
    elif kernel.kind is KernelKind.COMMUNICATION:
        binding = standalone_communication_bindings(kernel, call_id=call_id, binding_id=binding_id,
            device_id=device_id,rank=rank,stream=stream)
    elif kernel.kind is KernelKind.FLASH_ATTN:
        from src.kernel.memory_cann_mla import has_cann_mla_source,standalone_cann_mla_bindings
        binding = (standalone_cann_mla_bindings(kernel,hardware,call_id=call_id,binding_id=binding_id)
            if has_cann_mla_source(kernel) else standalone_fa3_combine_bindings(kernel,hardware,call_id=call_id,binding_id=binding_id)
            if has_fa3_combine_source(kernel) else standalone_fa3_prefill_bindings(kernel,call_id=call_id,binding_id=binding_id)
            if has_fa3_prefill_source(kernel) else standalone_cann_tnd_bindings(kernel,call_id=call_id,binding_id=binding_id)
            if has_cann_tnd_source(kernel) else standalone_metax_varlen_bindings(kernel,call_id=call_id,binding_id=binding_id)
            if has_metax_varlen_source(kernel) else standalone_cann_attention_bindings(kernel,hardware,call_id=call_id,binding_id=binding_id)
            if has_cann_attention_source(kernel) else standalone_fa2_split_bindings(kernel,call_id=call_id,binding_id=binding_id)
            if has_fa2_split_source(kernel) else standalone_grouped_mla_bindings(kernel,call_id=call_id,binding_id=binding_id)
            if has_grouped_mla_source(kernel) else standalone_metax_mla_bindings(kernel,call_id=call_id,binding_id=binding_id)
            if has_metax_mla_source(kernel) else standalone_dense_attention_bindings(kernel,call_id=call_id,binding_id=binding_id))
    elif kernel.kind is KernelKind.EAGER_ATTN:
        binding = standalone_eager_attention_bindings(kernel, call_id=call_id, binding_id=binding_id)
    elif kernel.kind is KernelKind.KV_CACHE:
        binding = (standalone_target_cache_bindings(kernel,call_id=call_id,binding_id=binding_id) if has_target_cache_source(kernel)
            else standalone_tensor_kv_bindings(kernel,call_id=call_id,binding_id=binding_id) if has_tensor_only_kv_source(kernel,hardware)
            else standalone_cache_update_bindings(kernel,hardware,call_id=call_id,binding_id=binding_id))
    elif kernel.kind in (KernelKind.MEMCPY, KernelKind.KV_OFFLOAD):
        binding = standalone_transfer_bindings(kernel, call_id=call_id, binding_id=binding_id)
    elif has_cann_moe_transpose_source(kernel,hardware) and is_cann_transpose_alias(kernel):
        binding=standalone_cann_transpose_alias_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_cann_moe_unbound_cast_source(kernel,hardware):
        binding=standalone_cann_unbound_cast_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_cann_moe_range_source(kernel,hardware):
        binding = standalone_cann_moe_range_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_cann_expert_tokens_source(kernel,hardware):
        binding = standalone_cann_expert_tokens_bindings(kernel,hardware,call_id=call_id,binding_id=binding_id)
    elif has_cann_moe_init_source(kernel,hardware):
        binding = standalone_cann_moe_init_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_cann_moe_gating_source(kernel,hardware):
        binding = standalone_cann_moe_gating_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_cann_moe_finalize_source(kernel,hardware):
        binding = standalone_cann_moe_finalize_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_moe_fill_source(kernel,hardware):
        binding = standalone_moe_fill_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_moe_alignment_source(kernel,hardware):
        binding=standalone_moe_alignment_bindings(kernel,call_id=call_id,binding_id=binding_id,
            device_id=target_device,rank=rank,stream=stream)
    elif has_moe_count_sort_source(kernel,hardware):
        binding=standalone_moe_count_sort_bindings(kernel,call_id=call_id,binding_id=binding_id,
            device_id=target_device,rank=rank,stream=stream)
    elif has_gpu_topk_source(kernel,hardware):
        binding=standalone_gpu_topk_bindings(kernel,call_id=call_id,binding_id=binding_id,
            device_id=target_device,rank=rank,stream=stream)
    elif has_route_weight_source(kernel,hardware):
        binding=standalone_route_weight_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_moe_reduction_source(kernel,hardware):
        binding = standalone_moe_reduction_bindings(kernel,call_id=call_id,binding_id=binding_id,
            device_id=target_device,rank=rank,stream=stream)
    elif has_moe_pointwise_source(kernel,hardware):
        binding = standalone_moe_pointwise_bindings(kernel,call_id=call_id,binding_id=binding_id,
            device_id=target_device,rank=rank,stream=stream)
    elif has_cann_rotary_source(kernel,hardware):
        binding=standalone_cann_rotary_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_tensor_only_rotary_source(kernel,hardware):
        binding=standalone_tensor_rotary_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_gpu_rotary_source(kernel,hardware):
        binding = standalone_rotary_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_cann_normalization_source(kernel,hardware):
        binding = standalone_cann_normalization_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_pim_normalization_source(kernel,hardware):
        binding=standalone_normalization_bindings(kernel,hardware,call_id=call_id,binding_id=binding_id)
    elif has_public_normalization_source(kernel,hardware):
        binding=standalone_public_normalization_bindings(kernel,hardware,call_id=call_id,binding_id=binding_id)
    elif has_generic_reduction_source(kernel,hardware):
        binding=standalone_generic_reduction_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_mhc_source(kernel,hardware):
        binding=standalone_mhc_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_moe_metadata_source(kernel,hardware):
        binding=standalone_moe_metadata_bindings(kernel,call_id=call_id,binding_id=binding_id,
            device_id=target_device,rank=rank,stream=stream)
    elif has_v32_fused_norm_source(kernel,hardware):
        binding=standalone_v32_fused_norm_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_v32_fused_q_source(kernel,hardware):
        binding=standalone_v32_fused_q_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_pairnorm_source(kernel,hardware):
        binding=standalone_pairnorm_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_layernorm_source(kernel,hardware):
        binding=standalone_layernorm_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_gpu_normalization_source(kernel, hardware):
        binding = standalone_normalization_bindings(kernel,hardware, call_id=call_id, binding_id=binding_id,
            device_id=target_device,rank=rank,stream=stream)
    elif has_observed_router_reducer_abi(kernel):
        binding=standalone_observed_router_reducer_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif kernel.kind is KernelKind.VECTOR:
        binding = standalone_elementwise_bindings(kernel, call_id=call_id, binding_id=binding_id,
            device_id=target_device,rank=rank,stream=stream)
    elif has_observed_router_abi(kernel):
        binding=standalone_observed_router_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_simt_router_source(kernel,hardware):
        binding=standalone_simt_router_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_public_router_abi(kernel):
        binding=standalone_public_router_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_gpu_split_gemm_source(kernel,hardware):
        binding = standalone_gpu_split_gemm_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_cann_deterministic_gemm_source(kernel):
        binding = standalone_cann_deterministic_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_cann_single_core_gemm_source(kernel):
        binding = standalone_cann_single_core_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif has_cann_gemm_source(kernel):
        binding = standalone_cann_gemm_bindings(kernel,call_id=call_id,binding_id=binding_id)
    elif kernel.kind is KernelKind.GEMM:
        work = _gemm_work_from_kernel(kernel)
        binding = standalone_bindings(work, call_id=call_id, binding_id=binding_id,
            device_id=target_device, visibility_domain="gemm-visible",rank=rank,stream=stream)
    else:
        raise AssertionError("default source domain dispatch is incomplete")
    changes={}
    if binding.rank!=rank:changes['rank']=rank
    if binding.stream!=stream:changes['stream']=stream
    if device_id is not None and any(a.device_id!=device_id for a in binding.allocations):
        changes['allocations']=tuple(a if a.device_id==device_id else replace(a,device_id=device_id)
            for a in binding.allocations)
    return replace(binding,**changes) if changes else binding


def _validated_scalar_program(kernel, program, hardware):
    """Revalidate the actual caller source before prediction memo lookup."""
    if kernel.variant=='reduce':
        from src.kernel.memory_reduction import generic_reduction_source_work
        generic_reduction_source_work(program,expected_kernel=kernel)
    if kernel.variant=='gemm_elementwise':gemm_elementwise_source_work(program,expected_kernel=kernel)
    if kernel.variant in ('mhc_post','hc_head','mhc_post_prenorm','mhc_pre_norm','mhc_pre_norm_broadcast'):
        from src.kernel.memory_mhc import mhc_source_work
        mhc_source_work(program,expected_kernel=kernel)
    if kernel.variant in ('moe_router_logit_cast','moe_routed_id_widen','moe_routed_id_extent','moe_expert_map_lookup'):
        from src.kernel.memory_moe_routing import moe_metadata_source_work
        moe_metadata_source_work(program,expected_kernel=kernel,hardware=hardware)
    if kernel.variant in ("elementwise", "elementwise_pytorch_vec4", "activation", "silu",
                          "moe_act_and_mul", "moe_swiglu", "gemm_reduce", "gemm_elementwise"):
        from src.kernel.memory_path_solver import scalar_tile_source_work
        scalar_tile_source_work(program)
    return program


def prediction_memory_numeric_projection(kernel, hardware, program):
    """Declare numeric reuse only for these validated geometry projections.

    Actual storage/readiness are checked before this function. These models
    consume the returned source work and the KernelSpec/HWSpec, rather than
    caller names or allocation IDs. Other sources keep their exact binding key.
    """
    execution=memory_execution_kernel(kernel,hardware)
    if has_gpu_split_gemm_source(execution,hardware):
        from src.kernel.memory_gpu_split_gemm import gpu_split_gemm_source_demands
        return ('gpu-split-matrix',gpu_split_gemm_source_demands(
            program,expected_kernel=execution))
    if has_gpu_nonsplit_gemm_source(execution,hardware):
        from src.kernel.memory_path_solver import logical_gemm_operand_geometry
        return ('gpu-nonsplit-matrix',logical_gemm_operand_geometry(
            program,batched=execution.variant=='batched_gemm'))
    if has_gpu_normalization_source(execution,hardware):
        from src.kernel.memory_path_solver import normalization_source_bytes
        return ('gpu-normalization',normalization_source_bytes(
            program,expected_kernel=execution))
    if has_gpu_rotary_source(execution,hardware):
        from src.kernel.memory_rotary import rotary_source_demands
        return ('gpu-rotary',rotary_source_demands(program))
    if has_fa2_split_source(execution):
        from src.kernel.memory_flash_attention import fa2_split_source_demands
        return ('fa2-split',fa2_split_source_demands(program))
    if has_moe_alignment_source(execution,hardware):
        from src.kernel.memory_fused_moe import moe_alignment_source_demands
        return ('moe-alignment',moe_alignment_source_demands(program,expected_kernel=execution))
    if has_gpu_topk_source(execution,hardware):
        from src.kernel.memory_moe_routing import gpu_topk_source_demands
        return ('gpu-topk',gpu_topk_source_demands(program,expected_kernel=execution))
    if has_moe_count_sort_source(execution,hardware):
        from src.kernel.memory_fused_moe import moe_count_sort_source_demands
        return ('moe-count-sort',moe_count_sort_source_demands(program,expected_kernel=execution))
    if has_moe_reduction_source(execution,hardware):
        from src.kernel.memory_moe_reduction import moe_reduction_source_demands
        return ('moe-reduction',moe_reduction_source_demands(program,expected_kernel=execution))
    if has_moe_metadata_source(execution,hardware):
        from src.kernel.memory_moe_routing import moe_metadata_source_work
        return ('moe-metadata',moe_metadata_source_work(program,expected_kernel=execution,hardware=hardware))
    if has_route_weight_source(execution,hardware):
        from src.kernel.memory_moe_reduction import route_weight_source_demands
        return ('route-weight',route_weight_source_demands(program,expected_kernel=execution))
    if has_communication_source(execution):
        from src.kernel.memory_path_solver import collective_source_work
        # Both existing backends consume this exact payload/participant/op
        # projection. Caller names do not affect either backend's price.
        return ('collective',collective_source_work(program))
    if (len(program.tile_programs)==1
            and program.tile_programs[0].scope.agent=='logical-elementwise'):
        from src.kernel.memory_path_solver import scalar_tile_source_work
        work=scalar_tile_source_work(program)
        operands=(*work['reads'],*work['writes'])
        layouts=tuple((v.shape,v.dtype,v.element_bytes,
            ('tensor',v.offset_bytes,v.strides_bytes) if isinstance(v,TensorView)
            else ('selection',v.coordinate_map)) for v in operands)
        aliases=tuple(tuple(a.allocation_id==b.allocation_id for b in operands) for a in operands)
        return ('scalar-source',work['scope'].op_ids,work['shape'],work['read_bytes'],
            work['write_bytes'],layouts,aliases)
    if (has_paged_kv_source(execution,hardware) and not has_target_cache_source(execution)):
        from src.kernel.memory_path_solver import logical_paged_kv_update_bytes
        return ('paged-kv-update',logical_paged_kv_update_bytes(program))
    from src.kernel.kernel_spec import KernelKind
    if kernel.kind is KernelKind.FUSED_MOE and has_fused_moe_source(execution,hardware):
        from src.kernel.memory_path_solver import fused_moe_source_demands
        from src.kernel.kernel_base import prediction_equivalence_attributes
        work=fused_moe_source_demands(program)
        if prediction_equivalence_attributes(kernel) is not None:
            # The existing predictor declares expert relabelling invariant
            # for these recipes. Preserve global/local count correlation and
            # every other validated work field; order-sensitive recipes keep
            # their exact routing. Caller identities never enter this price.
            global_counts,local_counts=work['routing']
            work={**work,'routing':tuple(sorted(zip(global_counts,local_counts,strict=True)))}
        return ('fused-moe-source-work',work)
    if has_cann_single_core_gemm_source(execution):
        from src.kernel.memory_cann_gemm import cann_single_core_source_demands
        return ('cann-single-core',cann_single_core_source_demands(program))
    if has_cann_deterministic_gemm_source(execution):
        from src.kernel.memory_cann_gemm import cann_deterministic_source_demands
        return ('cann-deterministic',cann_deterministic_source_demands(program))
    if has_cann_gemm_source(execution):
        from src.kernel.memory_cann_gemm import cann_gemm_source_demands
        return ('cann-matrix',cann_gemm_source_demands(program))
    if has_cann_normalization_source(execution,hardware):
        from src.kernel.memory_cann_normalization import cann_normalization_source_demands
        return ('cann-normalization',cann_normalization_source_demands(program))
    if has_cann_tnd_source(execution):
        # All consumed quantities, including native state and mask counts,
        # are retained. Caller storage IDs do not enter this analytical price.
        from src.kernel.memory_flash_attention import cann_tnd_source_demands
        return ('cann-tnd',cann_tnd_source_demands(program))
    return None


def bound_prediction_numeric_reuse_key(kernel, hardware, bindings, method):
    """Validate this caller before sharing an existing declared numeric result.

    Only explicitly declared source projections participate. A fresh actual
    program is validated before every new binding can reuse a price. Matrix,
    normalization and scalar keys also retain the alpha-renamed caller
    structure. Paged KV consumes only its validated payload/index byte counts
    after source validation; its exact KernelSpec/HWSpec are keyed by the memo.
    """
    if (method not in ('midlayer','roofline','roofline_span_floor')
            or type(bindings) not in (MemoryBindings,ElementwiseBindings,CacheUpdateBindings)):
        return None
    execution = memory_execution_kernel(kernel,hardware)
    from src.kernel.kernel_spec import KernelKind
    logical = (has_logical_interface_source(kernel,hardware)
        and (kernel.kind is KernelKind.GEMM and kernel.variant == 'gemm'
            or has_scalar_analytical_work(kernel)))
    if not (has_gpu_nonsplit_gemm_source(execution,hardware)
            or has_gpu_normalization_source(execution,hardware)
            or has_scalar_analytical_work(execution) or logical
            or (type(bindings) is CacheUpdateBindings
                and has_paged_kv_source(execution,hardware)
                and not has_target_cache_source(execution))):
        return None
    program = bound_prediction_memory_program(kernel,hardware,bindings,method)
    projection = prediction_memory_numeric_projection(kernel,hardware,program)
    if projection is None and logical:
        from src.kernel.memory_path_solver import logical_interface_source_bytes
        projection = ('logical-public-interface',logical_interface_source_bytes(
            program,expected_kernel=kernel,hardware=hardware))
    if projection is None:
        return None
    if type(bindings) is CacheUpdateBindings:
        # The paged-KV predictor reads this source only through these three
        # byte quantities. Layout, launch and services remain in KernelSpec
        # and HWSpec. The builder above checked this caller's own storage,
        # mapping, aliases and readiness; no source program is shared here.
        return _immutable_work_value(projection)
    pools = tuple(tuple(getattr(item,field) for item in records) for records,field in (
        (bindings.allocations,'allocation_id'),(bindings.views,'view_id'),(bindings.events,'event_id')))
    if any(len(set(pool)) != len(pool) for pool in pools):
        raise ValueError('Numeric source reuse requires unique operand and event identities')
    # A source tensor may name its view and allocation identically. Typed
    # field roles select their namespace, so cross-namespace overlap is safe.
    allocations,views,events = ({name:index for index,name in enumerate(pool)} for pool in pools)

    def structure(value,field):
        kind = type(value)
        if kind is str:
            if field == 'binding_id':return ('binding',)
            if field == 'call_id':return ('call',)
            if field == 'allocation_id' and value in allocations:return ('allocation',allocations[value])
            if field in ('event_id','node_id','allocated_event','freed_event') and value in events:
                return ('event',events[value])
            if (field in ('view_id','parent_view_id','read_views','write_views') or field.endswith('_view')) and value in views:
                return ('view',views[value])
            if field == 'visibility_domain' and value == bindings.visibility_domain:
                return ('local-visibility',)
            return ('literal',value)
        if kind is tuple:return tuple(structure(item,field) for item in value)
        if is_dataclass(value) and not isinstance(value,type):
            return (kind,tuple((item.name,structure(getattr(value,item.name),item.name)) for item in fields(value)))
        return _immutable_work_value(value)

    return (_immutable_work_value(projection),structure(bindings,''))


def bound_prediction_memory_program(kernel, hardware, bindings, method):
    """Validate/compile a cached call's actual typed source, without pricing."""
    from src.kernel.kernel_spec import KernelKind
    if isinstance(bindings,MemoryAccessWork) and bindings.is_source_projection:
        bindings.consume_source(kernel,hardware,bindings.source_evaluator)
        bindings=bindings.invocation
    if isinstance(bindings,(MemoryInvocation,LogicalMemoryInvocation)):
        bindings.prediction_work(kernel,hardware)
        bindings=bindings.materialized
    if has_mathematical_gemm_source(kernel) or has_mathematical_memory_source(kernel) or has_logical_interface_source(kernel,hardware):
        if method not in ('midlayer','roofline','roofline_span_floor'):
            raise NotImplementedError('Logical interface bindings require an analytical prediction method')
        return build_memory_access_program(kernel,hardware,bindings,Granularity.KERNEL,program_id='bound.logical')
    execution = memory_execution_kernel(kernel, hardware)
    with prediction_memory_bindings(kernel, bindings, method):
        if has_v32_fused_norm_source(execution,hardware):
            from src.kernel.memory_fused_rotary import v32_fused_norm_source_work
            program=elementwise_memory_access_program(execution,hardware)
            v32_fused_norm_source_work(program,expected_kernel=execution,hardware=hardware,expected_bindings=bindings)
            return program
        if has_v32_fused_q_source(execution,hardware):
            from src.kernel.memory_fused_rotary import v32_fused_q_source_work
            program=elementwise_memory_access_program(execution,hardware)
            v32_fused_q_source_work(program,expected_kernel=execution,hardware=hardware,expected_bindings=bindings)
            return program
        if has_observed_router_reducer_abi(execution):
            program=gemm_aux_memory_access_program(execution,hardware)
            observed_router_reducer_source_work(program,expected_kernel=execution,hardware=hardware,expected_bindings=bindings)
            return program
        if has_observed_router_abi(execution):
            program=gemm_memory_access_program(execution,hardware)
            observed_router_source_work(program,expected_kernel=execution,hardware=hardware,expected_bindings=bindings)
            return program
        if has_simt_router_source(execution,hardware):
            from src.kernel.memory_router import simt_router_source_work
            program=gemm_memory_access_program(execution,hardware)
            simt_router_source_work(program,expected_kernel=execution,hardware=hardware,expected_bindings=bindings)
            return program
        if has_public_router_abi(execution):
            program=gemm_memory_access_program(execution,hardware)
            public_router_source_work(program,expected_kernel=execution,hardware=hardware,expected_bindings=bindings)
            return program
        if has_generic_reduction_source(execution,hardware):
            from src.kernel.memory_reduction import generic_reduction_source_work
            program=elementwise_memory_access_program(execution,hardware)
            generic_reduction_source_work(program,expected_kernel=execution,expected_bindings=bindings)
            return program
        if has_moe_metadata_source(execution,hardware):
            from src.kernel.memory_moe_routing import moe_metadata_source_work
            program=elementwise_memory_access_program(execution,hardware)
            moe_metadata_source_work(program,expected_kernel=execution,hardware=hardware,expected_bindings=bindings)
            return program
        if kernel.kind is KernelKind.FUSED_MOE:
            program = fused_moe_memory_access_program(execution,hardware)
            if has_routed_moe_source(execution,hardware):
                from src.kernel.memory_routed_moe import routed_moe_source_demands
                routed_moe_source_demands(program,expected_kernel=execution,expected_hardware=hardware,expected_bindings=bindings)
            elif has_fp8_fused_moe_source(execution):
                from src.kernel.memory_fused_moe_fp8 import fp8_fused_source_demands
                fp8_fused_source_demands(program,expected_kernel=execution)
            else:
                from src.kernel.memory_path_solver import fused_moe_source_demands
                fused_moe_source_demands(program)
            return program
        if kernel.kind is KernelKind.SPARSE_ATTN:
            from src.kernel.memory_path_solver import sparse_attention_source_demands,v4_sparse_source_demands
            program = sparse_attention_memory_access_program(execution,hardware)
            if execution.variant.startswith('dsv4_'):
                v4_sparse_source_demands(program,expected_kernel=execution,expected_bindings=bindings)
            else:sparse_attention_source_demands(program,expected_kernel=execution,expected_bindings=bindings)
            return program
        if kernel.kind is KernelKind.SPARSE_ATTN_INDEXER:
            from src.kernel.memory_path_solver import sparse_indexer_source_demands
            program = sparse_indexer_memory_access_program(execution,hardware)
            sparse_indexer_source_demands(program,expected_kernel=execution)
            return program
        if kernel.kind is KernelKind.UNFUSED_MOE:
            from src.kernel.memory_path_solver import unfused_moe_primitive_demands
            program = unfused_moe_memory_access_program(execution, hardware)
            unfused_moe_primitive_demands(program)
            return program
        if kernel.kind is KernelKind.COMMUNICATION:
            from src.kernel.memory_path_solver import collective_source_work
            program = communication_memory_access_program(execution, hardware)
            collective_source_work(program)
            return program
        if has_cann_moe_range_source(kernel,hardware):
            from src.kernel.memory_cann_moe_routing import cann_moe_range_source_demands
            program = cann_moe_range_memory_access_program(execution,hardware)
            cann_moe_range_source_demands(program)
            return program
        if kernel.kind is KernelKind.FLASH_ATTN:
            from src.kernel.memory_cann_mla import has_cann_mla_source,cann_mla_source_demands
            if has_cann_mla_source(execution):
                program=build_memory_access_program(execution,hardware,None,Granularity.TILE,program_id='cann.mla')
                cann_mla_source_demands(program,expected_kernel=execution,hardware=hardware)
                return program
            if has_cann_tnd_source(execution):
                from src.kernel.memory_flash_attention import cann_tnd_source_demands
                program = cann_tnd_memory_access_program(execution,hardware)
                cann_tnd_source_demands(program)
                return program
            if has_metax_varlen_source(execution):
                from src.kernel.memory_flash_attention import metax_varlen_source_demands
                program = metax_varlen_memory_access_program(execution,hardware)
                metax_varlen_source_demands(program)
                return program
            if has_cann_attention_source(execution):
                from src.kernel.memory_flash_attention import cann_attention_source_demands
                program = cann_attention_memory_access_program(execution,hardware)
                cann_attention_source_demands(program,expected_kernel=execution)
                return program
            if has_fa3_combine_source(execution):
                from src.kernel.memory_flash_attention import fa3_combine_source_demands
                program=fa3_combine_memory_access_program(execution,hardware)
                fa3_combine_source_demands(program,expected_kernel=execution,hardware=hardware,expected_bindings=bindings)
                return program
            if has_fa3_prefill_source(execution):
                from src.kernel.memory_flash_attention import fa3_prefill_source_demands
                program = fa3_prefill_memory_access_program(execution,hardware)
                fa3_prefill_source_demands(program,expected_kernel=execution,expected_bindings=bindings)
                return program
            if has_fa2_split_source(execution):
                from src.kernel.memory_flash_attention import fa2_split_source_demands
                program = fa2_split_memory_access_program(execution,hardware)
                fa2_split_source_demands(program)
                return program
            if has_grouped_mla_source(execution):
                from src.kernel.memory_flash_attention import grouped_mla_source_demands
                program = grouped_mla_memory_access_program(execution,hardware)
                grouped_mla_source_demands(program)
                return program
            if has_metax_mla_source(execution):
                from src.kernel.memory_flash_attention import metax_mla_source_demands
                program = metax_mla_memory_access_program(execution,hardware)
                metax_mla_source_demands(program,expected_kernel=execution)
                return program
            from src.kernel.memory_path_solver import dense_attention_interface_demands
            program = dense_attention_memory_access_program(execution,hardware)
            dense_attention_interface_demands(program)
            return program
        if kernel.kind is KernelKind.EAGER_ATTN:
            from src.kernel.memory_path_solver import eager_attention_interface_demands
            program=eager_attention_memory_access_program(execution, hardware)
            eager_attention_interface_demands(program)
            return program
        if kernel.kind is KernelKind.KV_CACHE:
            if has_tensor_only_kv_source(execution,hardware):
                program=paged_kv_memory_access_program(execution,hardware)
                tensor_kv_source_work(program,expected_kernel=execution)
                return program
            if has_target_cache_source(execution):
                program = target_cache_memory_access_program(execution,hardware)
                target_cache_source_demands(program,expected_kernel=execution)
                return program
            from src.kernel.memory_path_solver import logical_paged_kv_update_bytes
            program=paged_kv_memory_access_program(execution,hardware)
            logical_paged_kv_update_bytes(program)
            return program
        if kernel.kind in (KernelKind.MEMCPY, KernelKind.KV_OFFLOAD):
            from src.kernel.memory_path_solver import logical_byte_copy_payload
            program = transfer_memory_access_program(execution, hardware)
            logical_byte_copy_payload(program)
            return program
        if kernel.kind is KernelKind.GEMM:
            if has_gpu_split_gemm_source(execution,hardware):
                from src.kernel.memory_gpu_split_gemm import gpu_split_gemm_source_demands
                program = gemm_memory_access_program(execution,hardware)
                gpu_split_gemm_source_demands(program,expected_kernel=execution)
                return program
            if has_cann_deterministic_gemm_source(execution):
                from src.kernel.memory_cann_gemm import cann_deterministic_source_demands
                program = gemm_memory_access_program(execution,hardware)
                cann_deterministic_source_demands(program)
                return program
            if has_cann_single_core_gemm_source(execution):
                from src.kernel.memory_cann_gemm import cann_single_core_source_demands
                program = gemm_memory_access_program(execution,hardware)
                cann_single_core_source_demands(program)
                return program
            if has_cann_gemm_source(execution):
                from src.kernel.memory_cann_gemm import cann_gemm_source_demands
                program = gemm_memory_access_program(execution,hardware)
                cann_gemm_source_demands(program)
                return program
            if has_block_fp8_source(execution):
                from src.kernel.memory_path_solver import block_fp8_source_demands
                program = block_fp8_memory_access_program(execution, hardware)
                block_fp8_source_demands(program)
                return program
            if has_gpu_nonsplit_gemm_source(execution,hardware):
                from src.kernel.memory_path_solver import logical_gemm_operand_geometry
                program=gemm_memory_access_program(execution,hardware)
                logical_gemm_operand_geometry(program,batched=execution.variant == 'batched_gemm')
                return program
            if (kernel.attributes["output_dtype"] is not None
                    and kernel.attributes["output_dtype"] != kernel.dtype):
                return build_memory_access_program(execution, hardware, None, Granularity.KERNEL,
                                                   program_id="gemm.mixed_output")
            from src.kernel.memory_path_solver import logical_gemm_operand_geometry
            program = gemm_memory_access_program(execution, hardware)
            logical_gemm_operand_geometry(program,batched=execution.variant == 'batched_gemm')
            return program
        if kernel.kind is KernelKind.VECTOR:
            if execution.variant == "dsv4_short_context_topk":
                from src.kernel.memory_path_solver import short_context_topk_source_demands
                program = elementwise_memory_access_program(execution, hardware)
                short_context_topk_source_demands(program,expected_kernel=execution)
                return program
            if execution.variant == "softmax":
                from src.kernel.memory_path_solver import softmax_source_work
                program = elementwise_memory_access_program(execution, hardware)
                softmax_source_work(program)
                return program
            if has_cann_expert_tokens_source(execution,hardware):
                from src.kernel.memory_cann_moe_routing import cann_expert_tokens_source_demands
                program = cann_expert_tokens_memory_access_program(execution,hardware)
                cann_expert_tokens_source_demands(program)
                return program
            if has_cann_moe_transpose_source(execution,hardware):
                program = elementwise_memory_access_program(execution,hardware)
                cann_moe_transpose_source_demands(program)
                return program
            if has_cann_moe_init_source(execution,hardware):
                from src.kernel.memory_cann_moe_routing import cann_moe_init_source_demands
                program = cann_moe_init_memory_access_program(execution,hardware)
                cann_moe_init_source_demands(program)
                return program
            if kernel.variant == 'flash_mla_metadata':
                from src.kernel.memory_flash_attention import flash_mla_metadata_source_demands
                program = elementwise_memory_access_program(execution,hardware)
                flash_mla_metadata_source_demands(program)
                return program
            if has_cann_moe_expert_cast_source(execution,hardware):
                from src.kernel.memory_cann_moe_routing import cann_moe_expert_cast_source_demands
                program = elementwise_memory_access_program(execution,hardware)
                cann_moe_expert_cast_source_demands(program,expected_kernel=execution)
                return program
            if has_cann_moe_unbound_cast_source(execution,hardware):
                from src.kernel.memory_cann_moe_routing import cann_unbound_cast_source_demands
                program=elementwise_memory_access_program(execution,hardware)
                cann_unbound_cast_source_demands(program,expected_kernel=execution)
                return program
            if has_cann_moe_gating_source(execution,hardware):
                from src.kernel.memory_cann_moe_routing import cann_moe_gating_source_demands
                program = cann_moe_gating_memory_access_program(execution,hardware)
                cann_moe_gating_source_demands(program)
                return program
            if has_cann_moe_finalize_source(execution,hardware):
                from src.kernel.memory_cann_moe_finalize import cann_moe_finalize_source_demands
                program = cann_moe_finalize_memory_access_program(execution,hardware)
                cann_moe_finalize_source_demands(program)
                return program
            if has_moe_fill_source(execution,hardware):
                from src.kernel.memory_moe_fill import moe_fill_source_demands
                program = moe_fill_memory_access_program(execution,hardware)
                moe_fill_source_demands(program)
                return program
            if has_moe_alignment_source(execution,hardware):
                from src.kernel.memory_fused_moe import moe_alignment_source_demands
                program=moe_alignment_memory_access_program(execution,hardware)
                moe_alignment_source_demands(program,expected_kernel=execution)
                return program
            if has_moe_count_sort_source(execution,hardware):
                from src.kernel.memory_fused_moe import moe_count_sort_source_demands
                program=moe_count_sort_memory_access_program(execution,hardware)
                moe_count_sort_source_demands(program,expected_kernel=execution)
                return program
            if has_gpu_topk_source(execution,hardware):
                from src.kernel.memory_moe_routing import gpu_topk_source_demands
                program=gpu_topk_memory_access_program(execution,hardware)
                gpu_topk_source_demands(program,expected_kernel=execution)
                return program
            if has_route_weight_source(execution,hardware):
                from src.kernel.memory_moe_reduction import route_weight_source_demands
                program=route_weight_memory_access_program(execution,hardware)
                route_weight_source_demands(program,expected_kernel=execution)
                return program
            if has_moe_reduction_source(execution,hardware):
                from src.kernel.memory_moe_reduction import moe_reduction_source_demands
                program = moe_reduction_memory_access_program(execution,hardware)
                moe_reduction_source_demands(program,expected_kernel=execution)
                return program
            if has_moe_pointwise_source(execution,hardware):
                from src.kernel.memory_moe_pointwise import moe_pointwise_source_demands
                program = moe_pointwise_memory_access_program(execution,hardware)
                moe_pointwise_source_demands(program)
                return program
            if has_cann_rotary_source(execution,hardware):
                from src.kernel.memory_cann_rotary import cann_rotary_source_demands
                program=rotary_memory_access_program(execution,hardware)
                cann_rotary_source_demands(program,expected_kernel=execution)
                return program
            if has_tensor_only_rotary_source(execution,hardware):
                from src.kernel.memory_rotary import rotary_source_demands
                program=rotary_memory_access_program(execution,hardware)
                rotary_source_demands(program)
                return program
            if has_gpu_rotary_source(execution,hardware):
                from src.kernel.memory_rotary import rotary_source_demands
                program = rotary_memory_access_program(execution,hardware)
                rotary_source_demands(program)
                return program
            if has_cann_normalization_source(execution,hardware):
                from src.kernel.memory_cann_normalization import cann_normalization_source_demands
                program = normalization_memory_access_program(execution,hardware)
                cann_normalization_source_demands(program)
                return program
            if has_fp8_group_quant_source(execution):
                from src.kernel.memory_path_solver import fp8_group_quant_source_demands
                program = fp8_group_quant_memory_access_program(execution, hardware)
                fp8_group_quant_source_demands(program)
                return program
            if has_pim_normalization_source(execution,hardware):
                from src.kernel.memory_normalization import pim_normalization_source_work
                program=normalization_memory_access_program(execution,hardware)
                pim_normalization_source_work(program,expected_kernel=execution)
                return program
            if has_pairnorm_source(execution,hardware):
                from src.kernel.memory_normalization import pairnorm_source_work
                program=normalization_memory_access_program(execution,hardware)
                pairnorm_source_work(program,expected_kernel=execution)
                return program
            if has_layernorm_source(execution,hardware):
                from src.kernel.memory_normalization import layernorm_source_work
                program=normalization_memory_access_program(execution,hardware)
                layernorm_source_work(program,expected_kernel=execution)
                return program
            if has_public_normalization_source(execution,hardware):
                from src.kernel.memory_normalization import public_normalization_source_work
                program=normalization_memory_access_program(execution,hardware)
                public_normalization_source_work(program,expected_kernel=execution,hardware=hardware)
                return program
            if has_gpu_normalization_source(execution, hardware):
                from src.kernel.memory_path_solver import normalization_source_bytes
                program=normalization_memory_access_program(execution,hardware)
                normalization_source_bytes(program,expected_kernel=execution)
                return program
            if has_moe_abs_source(execution,hardware):
                from src.kernel.memory_path_solver import moe_abs_source_work
                program = elementwise_memory_access_program(execution,hardware)
                moe_abs_source_work(program)
                return program
            if kernel.variant in ("gemm_reduce", "gemm_elementwise"):
                return _validated_scalar_program(execution, gemm_aux_memory_access_program(execution, hardware), hardware)
            return _validated_scalar_program(execution, elementwise_memory_access_program(execution, hardware), hardware)
        raise NotImplementedError("kernel has no default typed source")


@dataclass
class _BindingInvocation:
    kernel: object
    bindings: object
    consumed: bool = False
    implicit: bool = False


_PREDICTION_MEMORY_BINDINGS = ContextVar("midlayer_prediction_memory_bindings")
_PREDICTION_MEMORY_PROGRAMS = ContextVar("midlayer_prediction_memory_programs")


@contextmanager
def prediction_memory_program_reuse(*, execution_cache=None):
    """Reuse a complete source only inside one memo validation/pricing call.

    Every new invocation compiles and validates its actual caller. A numeric
    miss can then consume that same immutable program instead of rebuilding
    it. Original invocation/storage/hardware identities, complete execution
    KernelSpec and all builder options must match. Physical-extent rewrites
    can create equal immutable specs within that same invocation; nothing
    reuses another frame's storage/readiness.

    The same scope also reuses deterministic physical-extent resolution for
    an identical original KernelSpec/HWSpec object pair. A run-owned optional
    cache retains only that deterministic transformation; actual source
    bindings and programs are always fresh per invocation.
    """
    if execution_cache is not None and type(execution_cache) is not OrderedDict:
        raise TypeError('prediction execution reuse requires its owned ordered cache')
    token=_PREDICTION_MEMORY_PROGRAMS.set(([],OrderedDict() if execution_cache is None else execution_cache,
        4 if execution_cache is None else 256,{}))
    try:
        yield
    finally:
        _PREDICTION_MEMORY_PROGRAMS.reset(token)


@contextmanager
def prediction_memory_input_scope():
    """Own one analytical input for the duration of a public prediction.

    A Serving memo already owns this scope while it validates and prices a
    call. Standalone prediction owns the same lifecycle, rather than making
    each internal byte/shape query construct a separate source description.
    Nothing survives the prediction or substitutes for a caller boundary.
    """
    if _PREDICTION_MEMORY_PROGRAMS in copy_context():
        yield
    else:
        with prediction_memory_program_reuse():
            yield


@contextmanager
def _standalone_resource_input_scope(kernel):
    """Own implicit operands for one otherwise unbound resource export.

    A caller's bindings (including an intentional empty context) stay in
    charge. Only a standalone call establishes its own existing analytical
    scopes, and both scopes are restored when construction raises or returns.
    """
    if _PREDICTION_MEMORY_BINDINGS in copy_context():
        yield
    else:
        with prediction_memory_bindings(kernel, None, 'roofline'), prediction_memory_input_scope():
            yield


def _own_implicit_memory_source(kernel, actual):
    context = copy_context()
    if _PREDICTION_MEMORY_BINDINGS not in context:
        return actual
    invocation = context[_PREDICTION_MEMORY_BINDINGS]
    if invocation is None or not invocation.implicit:
        return actual
    if (kernel.kind, kernel.hardware_id, kernel.variant) != (
            invocation.kernel.kind, invocation.kernel.hardware_id, invocation.kernel.variant):
        raise ValueError('Generated analytical input crossed its prediction operation')
    if invocation.bindings is None:
        invocation.bindings = actual
    elif invocation.bindings is not actual:
        raise ValueError('One prediction constructed two different analytical inputs')
    return actual


@contextmanager
def prediction_memory_bindings(kernel, bindings, method):
    """Scope explicit untimed source bindings to one default prediction.

    This routes existing typed source bindings, not an external backend or
    a timing/session context. An unmigrated path cannot silently ignore them.
    """
    from src.kernel.kernel_spec import KernelKind, KernelSpec
    invocation = (_BindingInvocation(kernel, None, implicit=True)
                  if bindings is None and method in ('midlayer','roofline','roofline_span_floor') else None)
    if isinstance(bindings,(MemoryInvocation,LogicalMemoryInvocation)):
        if bindings.kernel!=kernel:raise ValueError('Memory invocation crossed its source kernel')
    if bindings is not None:
        if not isinstance(kernel, KernelSpec):
            raise TypeError("memory bindings require KernelSpec")
        if method not in ("midlayer", "roofline", "roofline_span_floor"):
            raise NotImplementedError("explicit source bindings are defined for default analytical methods only")
        binding_types = {KernelKind.VECTOR: (ElementwiseBindings,MemoryInvocation),
                         KernelKind.MEMCPY: TransferBindings, KernelKind.KV_OFFLOAD: TransferBindings,
                         KernelKind.KV_CACHE: (ElementwiseBindings,CacheUpdateBindings), KernelKind.EAGER_ATTN: EagerAttentionBindings,
                         KernelKind.FLASH_ATTN: ElementwiseBindings, KernelKind.COMMUNICATION: ElementwiseBindings,
                         KernelKind.UNFUSED_MOE: ElementwiseBindings, KernelKind.SPARSE_ATTN_INDEXER: SparseIndexerBindings, KernelKind.SPARSE_ATTN: SparseAttentionBindings,
                         KernelKind.FUSED_MOE: FusedMoEBindings}
        if kernel.kind is KernelKind.GEMM:
            from src.kernel.memory_gpu_split_gemm import has_gpu_split_gemm_abi
            from src.kernel.memory_cann_gemm import cann_split_execution_kernel
            # Only GEMM admission needs the captured matrix transform.
            cann_execution = cann_split_execution_kernel(kernel)
            split_binding = has_gpu_split_gemm_abi(kernel) or has_observed_router_abi(kernel) or has_public_tensor_gemm_source(kernel)
            binding_type = (ElementwiseBindings if has_public_router_abi(kernel) or has_simt_router_abi(kernel)
                or split_binding or has_block_fp8_source(kernel) or has_cann_gemm_source(cann_execution)
                or has_cann_single_core_gemm_source(cann_execution) or has_cann_deterministic_gemm_source(cann_execution)
                else (MemoryBindings,MemoryInvocation))
        elif kernel.kind is KernelKind.FUSED_MOE:
            binding_type = (ElementwiseBindings if has_fp8_fused_moe_source(kernel)
                or kernel.variant == 'routed_experts' else (FusedMoEBindings,MemoryInvocation))
        elif kernel.kind in binding_types:
            binding_type = binding_types[kernel.kind]
        else:
            raise NotImplementedError("kernel family has no production memory binding consumer yet")
        actual=bindings
        if isinstance(bindings,MemoryAccessWork) and bindings.is_source_projection:
            bindings.consume_source(kernel,bindings.source_hardware,bindings.source_evaluator)
            actual=bindings.invocation
        if not (isinstance(actual,binding_type) or isinstance(actual,(MemoryInvocation,LogicalMemoryInvocation))
                and actual.logical_hardware is not None and issubclass(ElementwiseBindings,binding_type)):
            from src.kernel.memory_moe_fill import FillMemoryInvocation
            if not (isinstance(bindings,FillMemoryInvocation) and kernel.kind is KernelKind.VECTOR
                    and kernel.variant=='moe_fill'):
                raise TypeError("memory bindings do not match the kernel family")
        if isinstance(actual,Typed):
            _validate_record_schema(actual)
        elif isinstance(actual,MemoryInvocation):
            actual.assert_boundary_unchanged()
        invocation = _BindingInvocation(kernel, bindings)
    token = _PREDICTION_MEMORY_BINDINGS.set(invocation)
    try:
        yield
        if invocation is not None and not invocation.implicit and not invocation.consumed:
            raise NotImplementedError(
                f"explicit memory bindings were not consumed by this production path: "
                f"{kernel.kind.value}/{kernel.variant} on {kernel.hardware_id} "
                f"({type(invocation.bindings).__name__}, method={method}, "
                f"execution_support={kernel.execution_support}, dtype={kernel.dtype}, "
                f"program={kernel.attributes['mathematical_program'] if 'mathematical_program' in kernel.attributes else None})")
    finally:
        _PREDICTION_MEMORY_BINDINGS.reset(token)


def _prediction_bindings(kernel, bindings):
    if bindings is not None:
        return bindings, None
    context = copy_context()
    if _PREDICTION_MEMORY_BINDINGS not in context:
        return None, None
    invocation = context[_PREDICTION_MEMORY_BINDINGS]
    if invocation is None:
        return None, None
    from src.kernel.kernel_spec import KernelKind
    if (kernel.kind is KernelKind.KV_CACHE
            and kernel.variant == "dsv4_compressed_state_cache"
            and invocation.kernel.variant == kernel.variant
            and invocation.kernel.attributes["finish_stage"] is None
            and kernel.attributes["finish_stage"] in ("compress", "store")):
        from src.kernel.memory_cache_update import (
            dsv4_c128_stage_bindings, standalone_target_cache_bindings)
        if invocation.bindings is None:
            invocation.bindings = standalone_target_cache_bindings(
                invocation.kernel, call_id="c128.aggregate", binding_id="c128.aggregate")
        actual = dsv4_c128_stage_bindings(invocation.kernel, kernel, invocation.bindings)
        invocation.consumed = True
        return actual, actual.call_id
    if invocation.bindings is None:
        return None, None
    if (kernel.kind, kernel.hardware_id, kernel.variant) != (
            invocation.kernel.kind, invocation.kernel.hardware_id, invocation.kernel.variant):
        raise ValueError("prediction memory bindings crossed a kernel/hardware/variant boundary")
    # Actual operand shape, storage type, alias and readiness validation is
    # performed by the single builder on the execution KernelSpec. This also
    # catches graph-shape rewrites whose caller bindings have the wrong extent.
    invocation.consumed = True
    return invocation.bindings, invocation.bindings.call_id


def paged_kv_memory_access_program(kernel, hardware):
    return build_memory_access_program(kernel, hardware, None, Granularity.KERNEL,
                                       program_id="kv_cache.paged_update")


def eager_attention_memory_access_program(kernel, hardware):
    return build_memory_access_program(kernel, hardware, None, Granularity.KERNEL, program_id="eager_attention.primitives")


def dense_attention_memory_access_program(kernel, hardware):
    return build_memory_access_program(kernel, hardware, None, Granularity.KERNEL,
                                      program_id="flash_attention.dense_interfaces")


def _source_prediction_work(kernel,hardware,actual,evaluator,*,with_hardware=False):
    actual = _own_implicit_memory_source(kernel, actual)
    if isinstance(actual,MemoryAccessWork):
        return actual.consume_source(kernel,hardware,evaluator)
    return MemoryAccessWork(actual,(),(),(),source_kernel=kernel,source_hardware=hardware,
        source_evaluator=evaluator,source_with_hardware=with_hardware)


def dense_attention_prediction_work(kernel,hardware,bindings=None):
    """Source-owned analytical demand; detailed graphs remain explicit export."""
    from src.kernel.memory_flash_attention import has_dense_ragged_analytical_work,dense_ragged_prediction_work
    if has_dense_ragged_analytical_work(kernel):
        actual,_=_prediction_bindings(kernel,bindings)
        if actual is None:
            actual=standalone_dense_attention_bindings(kernel,call_id='attention.analytical',binding_id='attention.analytical')
        return _source_prediction_work(kernel,hardware,actual,dense_ragged_prediction_work,with_hardware=True)
    from src.kernel.memory_path_solver import dense_attention_interface_demands
    return dense_attention_interface_demands(dense_attention_memory_access_program(kernel,hardware))


def metax_mla_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id='flash_attention.metax_mla')


def metax_mla_prediction_work(kernel,hardware,bindings=None):
    from src.kernel.memory_flash_attention import metax_mla_prediction_work as evaluate
    if kernel.hardware_id!=hardware.hardware_id:raise ValueError('MetaX MLA hardware identity mismatch')
    actual,_=_prediction_bindings(kernel,bindings)
    if actual is None:actual=standalone_metax_mla_bindings(kernel,call_id='mla',binding_id='mla')
    return _source_prediction_work(kernel,hardware,actual,evaluate)


def grouped_mla_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id='flash_attention.grouped_mla')


def grouped_mla_prediction_work(kernel,hardware,bindings=None):
    from src.kernel.memory_flash_attention import grouped_mla_prediction_work as evaluate
    if kernel.hardware_id!=hardware.hardware_id:raise ValueError('Grouped MLA hardware identity mismatch')
    actual,_=_prediction_bindings(kernel,bindings)
    if actual is None:actual=standalone_grouped_mla_bindings(kernel,call_id='mla',binding_id='mla')
    return _source_prediction_work(kernel,hardware,actual,evaluate)


def normalization_prediction_source(kernel,hardware):
    actual,_=_prediction_bindings(kernel,None)
    if isinstance(actual,MemoryInvocation):
        work=actual.prediction_work(actual.kernel,hardware)
        if kernel!=actual.kernel:
            if kernel!=memory_execution_kernel(actual.kernel,hardware):
                raise ValueError('Normalization consumer crossed its declared replay transform')
            # The existing Vector model prices its physical execution spec.
            # Keep all actual rows; only the declared price header changes.
            from dataclasses import replace
            rows=next(axis.extent for axis in work.axes if axis.axis_id=='row')
            return replace(work,logical_rows=rows,operation_ids=('normalization',))
        return work
    if actual is None and has_gpu_normalization_source(kernel,hardware):
        from src.kernel.memory_normalization import normalization_operand_records
        execution=kernel
        parents,views,_,_=normalization_operand_records(execution,binding_id='normalization.analytical',device_id='logical-local-device')
        invocation=MemoryInvocation(kernel,execution,'normalization.analytical','normalization.analytical',parents,views,0,'offline','norm-visible',False)
        invocation = _own_implicit_memory_source(kernel, invocation)
        return invocation.prediction_work(kernel,hardware)
    return normalization_memory_access_program(kernel,hardware)


def normalization_memory_access_program(kernel, hardware):
    return build_memory_access_program(kernel, hardware, None, Granularity.KERNEL,
                                      program_id="vector.normalization")


def rotary_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id='vector.rotary')


def cann_moe_init_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id='vector.cann_moe_init')


def cann_expert_tokens_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id='vector.cann_expert_tokens')


def moe_pointwise_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id='vector.moe_pointwise')


def moe_count_sort_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id='vector.moe_count_sort')


def route_weight_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id='vector.route_weights')


def gpu_topk_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id='vector.gpu_topk')


def moe_reduction_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id='vector.moe_reduction')


def moe_fill_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id='vector.moe_fill')


def moe_fill_prediction_work(kernel,hardware,bindings=None):
    from src.kernel.memory_moe_fill import FillMemoryInvocation,moe_fill_source_demands
    actual,_=_prediction_bindings(kernel,bindings)
    if actual is None:
        actual=FillMemoryInvocation(kernel,'fill.analytical','fill.analytical',
            'logical-local-device',0,'offline')
    if isinstance(actual,FillMemoryInvocation):return actual.work(kernel,hardware)
    program=build_memory_access_program(kernel,hardware,actual,Granularity.KERNEL,program_id='vector.moe_fill')
    return moe_fill_source_demands(program)


def cann_moe_finalize_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id='vector.cann_moe_finalize')


def cann_moe_gating_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id='vector.cann_moe_gating')


def communication_memory_access_program(kernel, hardware):
    return build_memory_access_program(kernel, hardware, None, Granularity.KERNEL,
                                      program_id="communication.participants")


def block_fp8_memory_access_program(kernel, hardware):
    return build_memory_access_program(kernel, hardware, None, Granularity.KERNEL,
                                      program_id="gemm.block_fp8")


def fp8_group_quant_memory_access_program(kernel, hardware):
    return build_memory_access_program(kernel, hardware, None, Granularity.KERNEL,
                                      program_id="vector.fp8_group_quant")


def unfused_moe_memory_access_program(kernel, hardware):
    return build_memory_access_program(kernel, hardware, None, Granularity.KERNEL,
                                      program_id="unfused_moe.primitives")


def sparse_indexer_memory_access_program(kernel, hardware):
    return build_memory_access_program(kernel, hardware, None, Granularity.KERNEL,
                                      program_id="sparse_indexer.interfaces")


def sparse_attention_memory_access_program(kernel, hardware):
    return build_memory_access_program(kernel, hardware, None, Granularity.KERNEL,
                                      program_id="sparse_attention.interfaces")


def fused_moe_memory_access_program(kernel, hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id='fused_moe.projection')


def fa2_split_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id="fa2.payload")


def fa2_split_prediction_work(kernel,hardware=None,bindings=None):
    from src.kernel.memory_flash_attention import fa2_split_prediction_work as evaluate
    if hardware is not None and kernel.hardware_id!=hardware.hardware_id:
        raise ValueError('FA2 split hardware identity mismatch')
    actual,_=_prediction_bindings(kernel,bindings)
    if actual is None:
        actual=standalone_fa2_split_bindings(kernel,call_id='fa2.payload',binding_id='fa2.payload')
    return _source_prediction_work(kernel,hardware,actual,evaluate)


def cann_attention_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id="cann.attention")


def metax_varlen_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id="metax.varlen")


def metax_varlen_prediction_work(kernel,hardware,bindings=None):
    from src.kernel.memory_flash_attention import metax_varlen_prediction_work as evaluate
    if kernel.hardware_id!=hardware.hardware_id:raise ValueError('C500 varlen hardware identity mismatch')
    actual,_=_prediction_bindings(kernel,bindings)
    if actual is None:
        actual=standalone_metax_varlen_bindings(kernel,call_id='metax.varlen',binding_id='metax.varlen')
    return _source_prediction_work(kernel,hardware,actual,evaluate)


def cann_tnd_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id="cann.tnd")


def cann_moe_range_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id="cann.range")


def target_cache_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.TILE,program_id='kv_cache.token_slot')


def fa3_prefill_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id='flash_attention.fa3_prefill')


def fa3_prefill_prediction_work(kernel,hardware,bindings=None):
    from src.kernel.memory_flash_attention import fa3_prefill_prediction_work as evaluate
    actual,_=_prediction_bindings(kernel,bindings)
    if actual is None:
        actual=standalone_fa3_prefill_bindings(kernel,call_id='fa3.prefill',binding_id='fa3.prefill')
    return _source_prediction_work(kernel,hardware,actual,evaluate,with_hardware=True)


def moe_alignment_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.TILE,program_id="vector.moe_alignment")


def moe_alignment_prediction_work(kernel,hardware,bindings=None):
    """Ordinary alignment's public analytical demand, without its scan graph."""
    from src.kernel.memory_moe_alignment import regular_alignment_bindings,regular_alignment_prediction_work
    if (kernel.hardware_id!=hardware.hardware_id or kernel.variant!='moe_align'
            or not has_moe_alignment_source(kernel,hardware)):
        raise ValueError('Analytical ordinary alignment requires its existing supported source ABI')
    actual,_=_prediction_bindings(kernel,bindings)
    if actual is None:
        actual=regular_alignment_bindings(kernel,call_id='vector.moe_alignment',binding_id='vector.moe_alignment')
    return _source_prediction_work(kernel,hardware,actual,regular_alignment_prediction_work)


def fused_moe_prediction_work(kernel,hardware,bindings=None):
    """Native grouped projection's directly evaluated analytical demand."""
    from src.kernel.memory_fused_moe import native_grouped_moe_prediction_work
    actual,_=_prediction_bindings(kernel,bindings)
    if actual is None:
        actual=default_memory_bindings(kernel,hardware,call_id='moe.analytical',binding_id='moe.analytical',
            rank=0,stream='offline',analytical=True)
    return _source_prediction_work(kernel,hardware,actual,native_grouped_moe_prediction_work,with_hardware=True)


def fa3_combine_memory_access_program(kernel,hardware):
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,program_id='flash_attention.fa3_combine')


def fa3_combine_prediction_work(kernel,hardware,bindings=None):
    from src.kernel.memory_flash_attention import fa3_combine_prediction_work as evaluate
    actual,_=_prediction_bindings(kernel,bindings)
    if actual is None:
        actual=standalone_fa3_combine_bindings(kernel,hardware,call_id='fa3.combine',binding_id='fa3.combine')
    return _source_prediction_work(kernel,hardware,actual,evaluate,with_hardware=True)


_LOGICAL_MOE_VARIANTS=(
    'moe_init_routing','moe_init_routing_v2','moe_init_routing_v3',
    'moe_gating_topk','moe_token_unpermute','moe_finalize_routing',
    'moe_transpose','moe_cast','moe_range','moe_topk','moe_silu','moe_mul',
    'moe_sum','moe_reduce','moe_align','moe_align_small_batch',
    'moe_routing_weight_norm','moe_routing_weight_scale','moe_count_and_sort','moe_fill')


def has_public_attention_source(kernel):
    from src.kernel.kernel_spec import KernelKind
    if (kernel.kind is not KernelKind.FLASH_ATTN or kernel.execution_support!='supported'
            or kernel.variant not in ('prefill','decode')):return False
    from src.kernel.memory_cann_mla import has_cann_mla_source
    return not any((has_cann_mla_source(kernel),has_cann_tnd_source(kernel),has_metax_varlen_source(kernel),
        has_cann_attention_source(kernel),has_fa2_split_source(kernel),has_dense_flash_attention_source(kernel),
        has_grouped_mla_source(kernel),has_metax_mla_source(kernel)))


def public_attention_dimensions(kernel):
    """Declared mathematical QK/V channels; native storage is separate."""
    from src.kernel.predictors.flash_attn import _grouped_mla_tile,_cann_mla_dimensions,_metax_mla_dimensions
    grouped=_grouped_mla_tile(kernel)
    if grouped is not None:return 'grouped',grouped[-2],grouped[-1]
    cann=_cann_mla_dimensions(kernel)
    if cann is not None:return 'cann_mla',*cann
    metax=_metax_mla_dimensions(kernel)
    if metax is not None:return 'metax_mla',*metax
    return 'qkv',kernel.shape['head_dim'],kernel.shape['head_dim']


def has_public_tensor_gemm_source(kernel,hardware=None):
    """Mathematical fallback interface, without asserting vendor input storage.

    The existing nominal price remains a separate conditional projection.
    More specific typed/native operand sources always take precedence.
    """
    from src.kernel.kernel_spec import KernelKind
    from src.kernel.memory_gpu_split_gemm import has_gpu_split_gemm_abi
    from src.kernel.memory_cann_gemm import (has_cann_gemm_source,cann_split_execution_kernel,
        has_cann_single_core_gemm_source,has_cann_deterministic_gemm_source)
    kernel=cann_split_execution_kernel(kernel)
    return (kernel.kind is KernelKind.GEMM and kernel.variant in ('gemm','batched_gemm')
        and kernel.execution_support=='supported'
        and kernel.attributes['output_dtype'] in (None,kernel.dtype)
        and kernel.hardware_id not in ('groq_lpu','samsung_hbm_pim','ucb_gemmini')
        and not any((has_gpu_nonsplit_gemm_abi(kernel),has_gpu_split_gemm_abi(kernel),
            has_cann_gemm_source(kernel),has_cann_single_core_gemm_source(kernel),
            has_cann_deterministic_gemm_source(kernel),has_block_fp8_source(kernel))))


def _has_external_aggregate_interface(kernel):
    """Explicit modeled storage; no native partial-buffer ABI is asserted."""
    from src.kernel.kernel_spec import KernelKind
    return (kernel.execution_support == 'supported' and kernel.launch is not None
        and kernel.hardware_id == 'nvidia_h200'
        and (kernel.kind, kernel.variant, kernel.dtype, kernel.launch.algorithm) in (
            (KernelKind.GEMM, 'gemm', 'fp16', 'h200_vllm030_torch213_fp16_external_aggregate_gemm'),
            (KernelKind.VECTOR, 'gemm_reduce', 'fp16', 'h200_vllm030_torch213_fp16_external_aggregate_reduce'),
            (KernelKind.GEMM, 'gemm', 'bf16', 'h200_cublaslt129_bf16_external_aggregate_gemm'),
            (KernelKind.VECTOR, 'gemm_reduce', 'bf16', 'h200_cublaslt129_bf16_external_aggregate_reduce')))


def has_logical_interface_source(kernel,hardware=None):
    """Explicit mathematical interfaces; independent of native/pricing admission."""
    from src.kernel.kernel_spec import KernelKind
    if kernel.execution_support!='supported':return False
    if _has_external_aggregate_interface(kernel):return True
    if has_public_tensor_gemm_source(kernel,hardware):return True
    if has_public_attention_source(kernel):return True
    if kernel.kind is KernelKind.VECTOR and kernel.variant in _LOGICAL_MOE_VARIANTS:
        # Keep the richer source whenever it is admitted. The mathematical
        # interface closes only the former unbound numerical branch.
        return hardware is None or not any(source(kernel,hardware) for source in (
            has_cann_moe_init_source,has_cann_moe_gating_source,has_cann_moe_finalize_source,
            has_cann_moe_transpose_source,has_cann_moe_expert_cast_source,has_cann_moe_unbound_cast_source,
            has_cann_moe_range_source,has_gpu_topk_source,has_moe_pointwise_source,
            has_moe_reduction_source,has_moe_alignment_source,has_moe_count_sort_source,
            has_route_weight_source,has_moe_fill_source))
    return (
        kernel.kind is KernelKind.FLASH_ATTN and (kernel.variant in ('mla_decode','mla_absorbed_prefill','mla_prefill')
            or kernel.launch.algorithm=='vllm_fa3_sm90_mla_decode'
            or kernel.variant=='combine' and not any((has_fa3_combine_source(kernel),has_fa2_split_source(kernel),
                has_grouped_mla_source(kernel),has_metax_mla_source(kernel))))
        or kernel.kind is KernelKind.VECTOR and kernel.variant in (
            'gather','mla_rope','qkv_rmsnorm_rope','moe_softmax','moe_gating',
            'moe_align_opt_128','moe_enable_if','dsv4_inv_rope_fp8_quant'))


def logical_interface_position_type(hardware):
    model=hardware.kernel_memory_model
    if model=='gpu_mshr':return 'int64',8
    if model=='npu_tile':
        if not hardware.aic_aiv_separated:raise ValueError('npu_tile Vector ABI requires separated AIC/AIV execution')
        return 'int32',4
    if model in ('pim_bank','systolic_scratchpad'):return None,None
    raise ValueError('Unknown kernel memory model for the serving position ABI')


def logical_interface_priced_by_source(kernel):
    from src.kernel.kernel_spec import KernelKind
    return (has_public_tensor_gemm_source(kernel) or has_public_attention_source(kernel) or kernel.kind is KernelKind.VECTOR and kernel.variant in (
        'gather','mla_rope','qkv_rmsnorm_rope','dsv4_inv_rope_fp8_quant',
        'moe_softmax','moe_gating','moe_align_opt_128','moe_enable_if',*_LOGICAL_MOE_VARIANTS)
        or kernel.kind is KernelKind.FLASH_ATTN and (kernel.variant in ('mla_decode','mla_prefill','mla_absorbed_prefill')
            or kernel.launch.algorithm=='vllm_fa3_sm90_mla_decode'
            or kernel.variant=='combine' and has_logical_interface_source(kernel)))


def _logical_interface_read_repeat(kernel,role):
    if kernel.variant=='mla_decode' and kernel.launch.algorithm!='vllm_fa3_sm90_mla_decode':
        lengths=kernel.attributes['q_lens']
        for index,queries in enumerate(lengths):
            names=('compressed_kv','key_rope') if len(lengths)==1 else (
                f'compressed_kv.request{index}',f'key_rope.request{index}')
            if role in names:return max(queries,1)
    return kernel.launch.grid_size if kernel.variant=='qkv_rmsnorm_rope' and role in ('q_gamma','k_gamma') else 1


def _logical_interface_layouts(kernel,hardware=None):
    """Normal contract quantities only; unknown storage remains explicitly opaque."""
    from src.kernel.kernel_spec import KernelKind
    s=kernel.shape;dtype=kernel.dtype;width=s['dtype_bytes'];v=kernel.variant
    if not has_logical_interface_source(kernel,hardware):raise ValueError('No declared logical interface for this kernel')
    if _has_external_aggregate_interface(kernel):
        physical = kernel.shape if kernel.kind is KernelKind.GEMM else kernel.attributes['source_gemm_shape']
        b,m,n,k = (physical[name] for name in ('b','m','n','k'))
        splits = int(kernel.launch.parameters['modeled_split_k'])
        partial = ('modeled_partials',(splits,b,m,n),'fp32',4)
        if kernel.kind is KernelKind.GEMM:
            return (('A',(b,m,k),dtype,width),('B',(k,n),dtype,width)),(partial,)
        return (partial,),(('logical_result',(b,m,n),dtype,width),)
    if kernel.kind is KernelKind.GEMM:
        b,m,n,k=(s[name] for name in ('b','m','n','k'))
        # Only mathematical index sets are known for opaque launches. In
        # particular BF16 output does not prove BF16 A/B or exclude scales.
        batched=v=='batched_gemm'
        reads=(('A',(b,m,k),None,None),('B',(b,k,n) if batched else (k,n),None,None))
        writes=(('logical_result',(b,m,n),dtype,width),)
        partial=kernel.launch.parameters['split_k_partial_bytes_per_element']
        if partial:
            if partial!=int(partial):raise ValueError('Declared GEMM partial storage needs integral dimensions')
            if kernel.launch.tiling is not None:
                splits=kernel.launch.tiling.split_k
                if splits is None:raise ValueError('Declared GEMM partial storage needs integral dimensions')
                writes+=(('partials',(splits,b*m*n,int(partial)),'uint8',1),)
            # Without retained split dimensions this interface cannot create
            # a scratch extent. The obligation remains explicitly unbound;
            # logical_result is already a mathematical, not physical C, role.
        return reads,writes
    if kernel.kind is KernelKind.FLASH_ATTN:
        if has_public_attention_source(kernel):
            mode,qk,value=public_attention_dimensions(kernel)
            from src.kernel.predictors.flash_attn import _cann_fia_appended_slot_execution_kernel,_metax_full_graph_execution_kernel
            kernel=_cann_fia_appended_slot_execution_kernel(_metax_full_graph_execution_kernel(kernel))
            s=kernel.shape
            from src.kernel.predictors.flash_attn import _request_lengths
            query_lengths, kv_lengths = _request_lengths(kernel)
            q=sum(query_lengths);kv=sum(kv_lengths)
            h=s['num_query_heads'];kvh=s['num_kv_heads'];splits=kernel.launch.tiling.num_kv_splits
            reads=(('query',(q,h,qk),dtype,width),('key',(kv,kvh,qk),dtype,width),('value',(kv,kvh,value),dtype,width))
            if mode=='grouped' or (mode=='qkv' and splits>1):
                writes=(('partial_output',(q,h,splits,value),'fp32',4),('partial_lse',(q,h,splits),'fp32',4))
            else:
                writes=(('output',(q,h,value),dtype,width),)
                from src.kernel.predictors.flash_attn import (_has_metax_q64_prefill_pipeline,
                    _METAX_372_Q128_PREFILL_ALGORITHM,_METAX_372_Q128_SPLIT_PREFILL_ALGORITHM)
                if _has_metax_q64_prefill_pipeline(kernel) or kernel.launch.algorithm in (
                        _METAX_372_Q128_PREFILL_ALGORITHM,_METAX_372_Q128_SPLIT_PREFILL_ALGORITHM):
                    writes+=(('lse',(q,h),'fp32',4),)
            return reads,writes
        if kernel.launch.algorithm=='vllm_fa3_sm90_mla_prefill':
            tile=kernel.launch.tiling.tile_shape;q=sum(kernel.attributes['q_lens']);kv=sum(kernel.attributes['kv_lens']);h=s['num_query_heads']
            return (('query',(q,h,tile['qk_head']),dtype,width),('key',(kv,h,tile['qk_head']),dtype,width),
                ('value',(kv,h,tile['value_head']),dtype,width)),(('output',(q,h,tile['value_head']),dtype,width),)
        if v=='combine':
            from src.kernel.predictors.flash_attn import (
                _metax_full_graph_execution_kernel,_fa2_full_graph_combine_execution_kernel,_has_graph_padding)
            kernel=_fa2_full_graph_combine_execution_kernel(_metax_full_graph_execution_kernel(kernel))
            q=sum(kernel.attributes['q_lens']);h=s['num_query_heads'];d=s['head_dim']
            splits=kernel.launch.tiling.num_kv_splits
            if splits is None or splits<1:raise ValueError('Logical combine requires its declared split count')
            writes=(('output',(q,h,d),dtype,width),)
            # FA2's split reducer publishes the final FP32 LSE as well as O.
            # This is a launch ABI fact, including logical descriptors on
            # devices outside the narrower native source admission.
            if (not _has_graph_padding(kernel)
                    and kernel.launch.algorithm in ('flash_attn_splitkv_combine','triton_grouped_mla_stage2')):
                writes+=(('lse',(q,h),'fp32',4),)
            return (('partial_output',(q,h,splits,d),'fp32',4),('partial_lse',(q,h,splits),'fp32',4)),writes
        if v=='mla_prefill':
            q=sum(kernel.attributes['q_lens']);kv=sum(kernel.attributes['kv_lens'])
            h=s['num_query_heads'];d=s['qk_head_dim']
            return (('query',(q,h,d),dtype,width),('key',(kv,h,d),dtype,width),
                    ('padded_value',(kv,h,d),dtype,width)),(('padded_output',(q,h,d),dtype,width),)
        q=sum(kernel.attributes['q_lens'])
        h=s['num_query_heads']
        if kernel.launch.algorithm=='vllm_fa3_sm90_mla_decode':
            rank=kernel.launch.tiling.tile_shape['value_head'];rope=kernel.launch.tiling.tile_shape['qk_head']
            query_width=rank+rope;value_width=rank
        else:
            rank=s['kv_lora_rank'];rope=s['qk_head_dim']-rank
            query_width=s['qk_head_dim'];value_width=s['v_head_dim']
        if rope<0:raise ValueError('MLA query width cannot be smaller than compressed KV rank')
        reads=[];q_lens=kernel.attributes['q_lens'];kv_lens=kernel.attributes['kv_lens']
        for index,(queries,kv) in enumerate(zip(q_lens,kv_lens)):
            suffix='' if len(q_lens)==1 else f'.request{index}'
            reads.extend((('query'+suffix,(queries,h,query_width),dtype,width),
                          ('compressed_kv'+suffix,(kv,rank),dtype,width)))
            if rope:reads.append(('key_rope'+suffix,(kv,rope),dtype,width))
        splits=kernel.launch.tiling.num_kv_splits
        if kernel.launch.algorithm=='vllm_fa3_sm90_mla_decode':
            from src.kernel.memory_flash_attention import fa3_mla_decode_partition_blocks
            tiles=fa3_mla_decode_partition_blocks(q_lens,kv_lens,h,splits,kernel.launch.grid[0],kernel.launch.tiling.tile_shape['kv'])
            active=sum(count>0 for count in tiles)
            writes=(('partial_output',(active,h,value_width),'fp32',4),('partial_lse',(active,h),'fp32',4))
        elif v=='mla_decode' and splits is not None and splits>1:
            writes=(('partial_output',(q,h,splits,value_width),'fp32',4),('partial_lse',(q,h,splits),'fp32',4))
        else:writes=(('output',(q,h,value_width),dtype,width),)
        return tuple(reads),writes
    if v=='gather':
        a=kernel.attributes;elements=s['element_count']
        reads=(('selected_values',(elements*a['gather_factor'],),dtype,width),
               ('indices',(a['index_count'],a['index_bytes']),'uint8',1))
        writes=(('output',(elements,),dtype,width),) if a['materialized_output'] else ()
        return reads,writes
    if v in ('mla_rope','qkv_rmsnorm_rope'):
        t,h,d=s['seq_len'],s['num_query_heads'],s['head_dim']
        k=s['num_key_heads'] if v=='mla_rope' else s['num_kv_heads']
        rotary=d if v=='mla_rope' else s['rotary_dim']
        position_type=('int64',8) if v=='qkv_rmsnorm_rope' else (
            (None,None) if hardware is None else logical_interface_position_type(hardware))
        reads=[('query',(t,h,d),dtype,width),('key',(t,k,d),dtype,width),('cos_sin',(t,rotary),dtype,width)]
        if hardware is None or position_type[1] is not None:reads.append(('positions',(t,),*position_type))
        writes=[('query_output',(t,h,d),dtype,width),('key_output',(t,k,d),dtype,width)]
        if v=='qkv_rmsnorm_rope':
            reads.extend((('value',(t,k,d),dtype,width),('q_gamma',(d,),dtype,width),('k_gamma',(d,),dtype,width)))
            writes.append(('value_output',(t,k,d),dtype,width))
            if kernel.attributes['bias']:reads.append(('bias',(h+k,d),dtype,width))
        return tuple(reads),tuple(writes)
    if v=='dsv4_inv_rope_fp8_quant':
        t,h,d=s['seq_len'],s['num_query_heads'],s['head_dim'];group=kernel.attributes['quant_group_size']
        position_type=(None,None) if hardware is None else logical_interface_position_type(hardware)
        reads=[('input',(t,h,d),'bf16',2),('cos_sin',(t,s['rope_dim']),'fp32',4)]
        if hardware is None or position_type[1] is not None:reads.append(('positions',(t,),*position_type))
        padded_tokens=-(-t//4)*4
        return tuple(reads),(('quantized',(t,h,d),'fp8',1),('scales',(padded_tokens,h,d//group),'fp32',4))
    t,e,k=s['tokens'],s['experts'],s['topk']
    if v in ('moe_softmax','moe_gating','moe_topk'):
        # These legacy contracts expose dense score work, not a top-k index ABI.
        return (('scores',(t,e),dtype,width),),(('score_results',(t,e),dtype,width),)
    if v in ('moe_align_opt_128','moe_align','moe_align_small_batch','moe_count_and_sort'):
        # Portable INT32 routing-reference state: expert offsets and the
        # (expert, token) route pair. This is the existing aggregate model's
        # logical representation, not the vendor's packed native ABI.
        return (('expert_state',(e,),'int32',4),('route_pairs',(t*k,2),'int32',4)),(
            ('expert_results',(e,),'int32',4),('sorted_route_pairs',(t*k,2),'int32',4))
    if v=='moe_enable_if':
        return (('expert_state',(e,),'int32',4),('route_state',(t*k,2),'int32',4)),(
            ('expert_result',(e,),'int32',4),('route_result',(t*k,2),'int32',4))
    h=s['hidden'];i=s['intermediate'];r=t*k
    if v in ('moe_init_routing','moe_init_routing_v2','moe_init_routing_v3'):
        reads=(('input',(t,h),'bf16',2),('route_ids',(r,),'int32',4))
        writes=(('routed',(r,h),'bf16',2),('route_output',(r,),'int32',4))
        if v=='moe_init_routing':
            return (*reads,('route_order',(r,),'int32',4)),(*writes,('route_order_output',(r,),'int32',4))
        if v=='moe_init_routing_v3':writes+= (('route_inverse',(r,),'int32',4),)
        return reads,(*writes,('expert_offsets',(e,),'int64',8))
    if v=='moe_gating_topk':
        return (('scores',(t,e),'fp32',4),),(
            ('router_output',(t,e),'fp32',4),('weights',(t,k),'fp32',4),('ids',(t,k),'int32',4))
    if v in ('moe_token_unpermute','moe_finalize_routing'):
        scale=('fp32',4) if v=='moe_token_unpermute' else ('bf16',2)
        return (('routed',(r,h),'bf16',2),('scales',(r,),*scale),('indices',(r,),'int32',4)),(
            ('output',(t,h),'bf16',2),)
    if v=='moe_transpose':
        return (('input',(t,k),'int32',4),),(('output',(k,t),'int32',4),)
    if v=='moe_cast':
        role=kernel.attributes['cast_role']
        if role=='topk_ids':count=r
        elif role=='expert_tokens':count=e
        else:raise ValueError('Unknown routing metadata cast role')
        return (('input',(count,),'int32',4),),(('output',(count,),'int32',4),)
    if v=='moe_range':
        return (('expert_state',(e,),'int32',4),('route_state',(r,),'int32',4)),(
            ('expert_result',(e,),'int32',4),('route_result',(r,),'int32',4))
    if v in ('moe_silu','moe_mul','moe_sum','moe_reduce'):
        total=r*i if v in ('moe_silu','moe_mul') else t*h
        selected=total*kernel.attributes['work_fraction']
        if selected!=int(selected) or selected<=0:
            raise ValueError('Logical iterator slice requires a positive integral element count')
        selected=int(selected)
        if v in ('moe_sum','moe_reduce'):
            return (('routed',(selected,k),'bf16',2),),(('output',(selected,),'bf16',2),)
        roles=('gate',) if v=='moe_silu' else ('silu','up')
        return tuple((role,(selected,),'bf16',2) for role in roles),(('output',(selected,),'bf16',2),)
    if v=='moe_routing_weight_norm':
        return (('weights',(t,k),'fp32',4),),(('sum',(t,),'fp32',4),)
    if v=='moe_routing_weight_scale':
        return (('weights',(t,k),'fp32',4),('sum',(t,),'fp32',4)),(('output',(t,k),'fp32',4),)
    if v=='moe_fill':
        block=16 if t<=e else 64
        padded=r+e*(block-1);role=kernel.attributes['fill_role']
        if role=='sorted_token_ids':count=padded
        elif role=='expert_ids':count=-(-padded//block)
        elif role=='num_tokens_post_padded':count=1
        else:raise ValueError('Unknown routing fill role')
        return (),((role,(count,),'int32',4),)
    raise AssertionError('Unreachable logical interface variant')


@dataclass(frozen=True, eq=False)
class LogicalMemoryInvocation:
    """Public role geometry and caller scope, without a storage/event graph.

    These interfaces declare invocation-boundary availability, not physical
    producer storage. Native producer registration remains owned by Serving.
    The original canonical binding is constructed only for explicit export.
    """
    kernel: object
    logical_hardware: object
    binding_id: str
    call_id: str
    device_id: str
    rank: int
    stream: str

    visibility_domain = 'logical-interface-visible'

    def __post_init__(self):
        from src.kernel.kernel_spec import KernelSpec, DTYPE_BYTES
        from src.kernel.hardware_spec import HWSpec
        from src.kernel.memory_access import check_type,U64_MAX
        from math import prod
        check_type(self.kernel, KernelSpec)
        check_type(self.logical_hardware, HWSpec)
        for value in (self.binding_id,self.call_id,self.device_id,self.stream):check_type(value,str)
        check_type(self.rank,int)
        if self.kernel.hardware_id!=self.logical_hardware.hardware_id:
            raise ValueError('Logical interface hardware identity mismatch')
        reads,writes=_logical_interface_layouts(self.kernel,self.logical_hardware)
        roles=tuple(role for role,_,_,_ in (*reads,*writes))
        if len(set(roles))!=len(roles):raise ValueError('Logical source roles must be unique')
        for role,shape,dtype,width in (*reads,*writes):
            if (type(role) is not str or not role or type(shape) is not tuple or not shape):
                raise ValueError('Logical source requires declared positive role dimensions')
            for dimension in shape:check_type(dimension,int)
            if min(shape)==0 or prod(shape)*(1 if width is None else width)>U64_MAX:
                raise ValueError('Logical source requires positive bounded payloads')
            if width is not None:check_type(width,int)
            if (dtype,width)!=(None,None) and (dtype not in DTYPE_BYTES or width!=DTYPE_BYTES[dtype]):
                raise NotImplementedError('Logical selection requires its actual scalar storage type')
        grouped=(has_public_attention_source(self.kernel)
            and public_attention_dimensions(self.kernel)[0]=='grouped')
        object.__setattr__(self,'layouts',(reads,writes))
        object.__setattr__(self,'read_views',tuple(role for role,_,_,_ in reads))
        object.__setattr__(self,'write_views',tuple(role for role,_,_,_ in writes))
        object.__setattr__(self,'grouped_alias',grouped)
        object.__setattr__(self,'_layout_recipe',(_logical_interface_layouts,_logical_interface_layouts.__code__))
        names=(*self.__dataclass_fields__,'layouts','read_views','write_views','grouped_alias','visibility_domain','_layout_recipe')
        object.__setattr__(self,'_source_seal',tuple((name,getattr(self,name)) for name in names))

    def assert_boundary_unchanged(self):
        if any(getattr(self,name) is not value for name,value in self._source_seal):
            raise ValueError('Immutable logical source boundary was overwritten')
        recipe,code=self._layout_recipe
        if recipe is not _logical_interface_layouts or code is not _logical_interface_layouts.__code__:
            raise ValueError('Logical source layout recipe changed after admission')
        if '_export_snapshot' in self.__dict__:
            exported,records=self._export_snapshot
            if (self.__dict__['materialized'] is not exported or any(
                    getattr(record,name) is not value for record,values in records for name,value in values)):
                raise ValueError('Exported logical source operand was overwritten')

    def assert_logical_source(self,kernel,hardware=None):
        self.assert_boundary_unchanged()
        if hardware is not None and hardware is not self.logical_hardware:
            from src.kernel.hardware_spec import HWSpec
            if (not isinstance(hardware,HWSpec)
                    or hardware.hardware_id!=self.logical_hardware.hardware_id
                    or _logical_interface_layouts(self.kernel,hardware)!=self.layouts):
                raise ValueError('Logical work crossed its source hardware')
        if kernel is self.kernel:return
        if ((kernel.kind,kernel.hardware_id,kernel.variant)!=(self.kernel.kind,self.kernel.hardware_id,self.kernel.variant)
                or _logical_interface_layouts(kernel,self.logical_hardware)!=self.layouts):
            raise ValueError('Numerical source projection changed its actual logical operands')

    def prediction_work(self,kernel,hardware):
        self.assert_logical_source(kernel,hardware)
        return self

    def requested_bytes(self,operation):
        from math import prod
        self.assert_boundary_unchanged()
        if not isinstance(operation,Operation):raise TypeError('Logical work requires a typed direction')
        layouts=self.layouts[0 if operation is Operation.READ else 1]
        if any(width is None for _,_,_,width in layouts):
            raise NotImplementedError('Logical source encoding must be declared before byte pricing')
        return sum(prod(shape)*width*(_logical_interface_read_repeat(self.kernel,role)
            if operation is Operation.READ else 1) for role,shape,_,width in layouts)

    def unique_bytes(self,operation):
        self.assert_boundary_unchanged()
        if not isinstance(operation,Operation):raise TypeError('Logical work requires a typed direction')
        if self.layouts[0 if operation is Operation.READ else 1]:
            raise NotImplementedError('Unresolved operand selection footprint')
        return 0

    def allocation_id(self,role):
        return self.binding_id+'.'+('key' if self.grouped_alias and role=='value' else role)

    @property
    def allocation_ids(self):
        return {self.allocation_id(role) for role in (*self.read_views,*self.write_views)}

    @cached_property
    def materialized(self):
        self.assert_boundary_unchanged()
        allocations,views=logical_operand_records(self.kernel,self.logical_hardware,
            binding_id=self.binding_id,device_id=self.device_id)
        exported=_materialize_logical_operands(self.binding_id,self.call_id,allocations,views,
            self.read_views,self.write_views,self.rank,self.stream,self.visibility_domain)
        coordinates=tuple(coordinate for view in views for coordinate in view.coordinate_map)
        records=(exported,*allocations,*views,*coordinates,*(term for c in coordinates for term in c.terms))
        object.__setattr__(self,'_export_snapshot',(exported,tuple((record,tuple(
            (field.name,getattr(record,field.name)) for field in fields(record))) for record in records)))
        return exported

    @property
    def allocations(self):return self.materialized.allocations

    @property
    def views(self):return self.materialized.views


def logical_memory_invocation(kernel,hardware,*,binding_id,call_id,device_id,rank,stream):
    return LogicalMemoryInvocation(kernel,hardware,binding_id,call_id,device_id,rank,stream)


def logical_operand_records(kernel,hardware,*,binding_id,device_id):
    """One public selection rule shared by direct work and explicit export."""
    reads,writes=_logical_interface_layouts(kernel,hardware)
    allocations=[];views=[]
    for role,shape,dtype,width in (*reads,*writes):
        identity=binding_id+'.'+role
        alias=(has_public_attention_source(kernel) and public_attention_dimensions(kernel)[0]=='grouped' and role=='value')
        if alias:
            from src.kernel.memory_access import LogicalCoordinateBinding,LogicalCoordinateTerm
            parent=next(view for view in views if view.view_id=='key')
            view=LogicalSelectionView(role,parent.allocation_id,shape,dtype,width,identity+'.selection','key',tuple(
                LogicalCoordinateBinding(axis,(LogicalCoordinateTerm(axis,1),)) for axis in range(3)))
        else:
            allocations.append(MemoryAllocation(identity,device_id,'logical-storage',None,
                AddressLevel.OBJECT,None,None,identity+'.allocated',None))
            view=LogicalSelectionView(role,identity,shape,dtype,width,identity+'.selection',None)
        views.append(view)
    return tuple(allocations),tuple(views)


def _validate_logical_invocation(invocation):
    from src.kernel.hardware_spec import HWSpec
    from src.kernel.memory_access import check_type,LogicalCoordinateBinding,LogicalCoordinateTerm
    check_type(invocation.logical_hardware,HWSpec)
    if (invocation.kernel.hardware_id!=invocation.logical_hardware.hardware_id
            or invocation.operand_source is not invocation.kernel
            or invocation.visibility_domain!='logical-interface-visible' or invocation.scope_initialized is not True):
        raise ValueError('Logical invocation differs from its source boundary')
    for value in (invocation.binding_id,invocation.call_id,invocation.stream):check_type(value,str)
    check_type(invocation.rank,int)
    if type(invocation.allocations) is not tuple or type(invocation.views) is not tuple:
        raise TypeError('Logical operands must be immutable records')
    reads,writes=_logical_interface_layouts(invocation.kernel,invocation.logical_hardware)
    for name,layouts in (('read_views',reads),('write_views',writes)):
        roles=tuple(role for role,_,_,_ in layouts)
        if name in invocation.__dict__:
            if getattr(invocation,name)!=roles:raise ValueError('Logical operand roles differ from their source')
        else:object.__setattr__(invocation,name,roles)
    views=unique_by(invocation.views,'view_id');parents=unique_by(invocation.allocations,'allocation_id')
    if set(views)!={role for role,_,_,_ in (*reads,*writes)} or set(parents)!={view.allocation_id for view in views.values()}:
        raise ValueError('Logical invocation has undeclared or missing operands')
    grouped=has_public_attention_source(invocation.kernel) and public_attention_dimensions(invocation.kernel)[0]=='grouped'
    for role,shape,dtype,width in (*reads,*writes):
        view=views[role];check_type(view,LogicalSelectionView);view.__post_init__()
        expected_parent=invocation.binding_id+('.key' if grouped and role=='value' else '.'+role)
        parent_view='key' if grouped and role=='value' else None
        coordinates=tuple(LogicalCoordinateBinding(axis,(LogicalCoordinateTerm(axis,1),)) for axis in range(3)) if parent_view else ()
        if ((view.shape,view.dtype,view.element_bytes,view.allocation_id,view.selection_id,view.parent_view_id,view.coordinate_map)
                !=(shape,dtype,width,expected_parent,invocation.binding_id+'.'+role+'.selection',parent_view,coordinates)):
            raise ValueError('Logical operand differs from its source selection or actual alias')
    if len({parent.device_id for parent in parents.values()})!=1:
        raise ValueError('Logical operand parents crossed devices')
    for parent in parents.values():
        check_type(parent,MemoryAllocation);parent.__post_init__()
        if (parent.memory_domain,parent.size_bytes,parent.address_level,parent.base_address_bytes,parent.mapping_ref,
                parent.allocated_event,parent.freed_event)!=('logical-storage',None,AddressLevel.OBJECT,None,None,parent.allocation_id+'.allocated',None):
            raise ValueError('Logical selection cannot acquire an unproved physical layout or lifetime')


@runtime_lru_cache(maxsize=128)
def _logical_access_patterns(reads,writes):
    """Static role/rank and dependency metadata; no dimensions or byte values."""
    accesses=[]
    for role,rank,repeated in reads:
        accesses.append(TileAccessTemplate('read.'+role,role,Operation.READ,('repeat.'+role,) if repeated else (),
            (None,)*rank,(),(),()))
    dependencies=tuple(TileDependency(access.access_id,EventKind.READ_READY,access.iteration_axes) for access in accesses)
    for role,rank in writes:
        accesses.append(TileAccessTemplate('write.'+role,role,Operation.WRITE,(),(None,)*rank,(),dependencies,()))
    return tuple(accesses)


def _logical_invocation_rule(invocation,source_kernel=None):
    kernel=invocation.kernel if source_kernel is None else source_kernel
    views={view.view_id:view for view in invocation.views}
    repeats=tuple((role,_logical_interface_read_repeat(kernel,role)) for role in invocation.read_views)
    axes=tuple(('repeat.'+role,repeat,1) for role,repeat in repeats if repeat!=1)
    patterns=_logical_access_patterns(tuple((role,len(views[role].shape),repeat!=1) for role,repeat in repeats),
        tuple((role,len(views[role].shape)) for role in invocation.write_views))
    return axes,patterns,(invocation.kernel.kind.value,invocation.kernel.variant)


def _materialize_logical_operands(binding_id,call_id,allocations,views,reads,writes,rank,stream,visibility_domain):
    events=[];ready=[];parents=unique_by(allocations,'allocation_id');allocated=set()
    for view in views:
        parent=parents[view.allocation_id]
        if parent.allocation_id not in allocated:
            events.append(ProgramEvent(parent.allocated_event,EventKind.ALLOCATED,parent.allocation_id,None,None,()))
            allocated.add(parent.allocation_id)
        if view.view_id in reads:
            event=ProgramEvent(binding_id+'.'+view.view_id+'.ready',EventKind.LOGICAL_AVAILABLE,
                view.allocation_id,view.region(),visibility_domain,(EventRef(parent.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,allocations,views,tuple(events),
        reads,writes,tuple(ready),rank,stream,visibility_domain)


def standalone_logical_interface_bindings(kernel,*,call_id,binding_id,hardware=None,
        device_id="logical-local-device",rank=0,stream="offline"):
    allocations,views=logical_operand_records(kernel,hardware,binding_id=binding_id,device_id=device_id)
    reads,writes=_logical_interface_layouts(kernel,hardware)
    return _materialize_logical_operands(binding_id,call_id,allocations,views,
        tuple(role for role,_,_,_ in reads),tuple(role for role,_,_,_ in writes),rank,stream,'logical-interface-visible')


def build_logical_interface(kernel,bindings,granularity,*,program_id,hardware=None):
    """Non-executable logical work; native issue/address fidelity is separate."""
    from src.kernel.kernel_spec import KernelKind
    if granularity not in (Granularity.OP,Granularity.KERNEL):
        raise NotImplementedError('This interface supports op/kernel summaries; native tile partition is not declared')
    if not isinstance(bindings,ElementwiseBindings):raise TypeError('Logical interface requires typed object bindings')
    reads,writes=_logical_interface_layouts(kernel,hardware)
    expected=(*reads,*writes);views=unique_by(bindings.views,'view_id')
    if (bindings.read_views!=tuple(item[0] for item in reads)
            or bindings.write_views!=tuple(item[0] for item in writes)
            or len(bindings.input_ready)!=len(reads) or set(views)!={item[0] for item in expected}):
        raise ValueError('Logical interface operand roles/readiness differ from the kernel contract')
    for role,shape,dtype,width in expected:
        view=views[role]
        if (view.shape,view.dtype,view.element_bytes)!=(shape,dtype,width):
            raise ValueError(f'Logical interface operand shape or encoding differs from its kernel: {role}: actual={(view.shape,view.dtype,view.element_bytes)}, expected={(shape,dtype,width)}')
    scope=AccessScope(program_id+'.scope',(kernel.kind.value,kernel.variant),bindings.call_id,(),
        bindings.rank,bindings.stream,'logical-public-interface')
    evidence=('src/kernel/kernel_spec.py: explicit variant shape and attribute contract.',
        'Mathematical interface only; unknown encoding/selection is preserved, without native traffic or timing inference.')
    accesses=[];parents=[]
    for index,role in enumerate(bindings.read_views):
        view=views[role];identity=program_id+'.read.'+role
        accesses.append(MemoryAccess(identity,scope.scope_id,view.allocation_id,None,Operation.READ,view.region(),
            Semantics.LOGICAL,None,None,None,None,bindings.visibility_domain,(bindings.input_ready[index],),
            EventKind.READ_READY,_logical_interface_read_repeat(kernel,role),evidence))
        parents.append(EventRef(identity,EventKind.READ_READY))
    for role in bindings.write_views:
        view=views[role]
        allocation=unique_by(bindings.allocations,'allocation_id')[view.allocation_id]
        accesses.append(MemoryAccess(program_id+'.write.'+role,scope.scope_id,view.allocation_id,None,
            Operation.WRITE,view.region(),Semantics.LOGICAL,None,None,None,None,bindings.visibility_domain,
            (*tuple(parents),EventRef(allocation.allocated_event,EventKind.ALLOCATED)),EventKind.WRITE_VISIBLE,1,evidence))
    limitations=('Logical object selections only; no addresses, contiguous layout, capacity, physical alias or native tiles are inferred.',
        'Opaque metadata/position encodings and index membership remain unknown; no native padding/scratch or exact traffic claim.',
        'Read-to-write dependencies describe the whole mathematical call; this is not an executable scheduling model.',
        'Known typed roles may feed the common byte projection; opaque roles require a declared pricing projection.')
    if _has_external_aggregate_interface(kernel):
        limitations+=('Partial count and FP32 partial encoding are explicit aggregate model assumptions, not observed effective K partitions or native buffer layout.',
            'Main owns modeled partial writes only; reducer owns modeled partial reads and one final result write in the declared input/output dtype.',
            'No physical partial-buffer allocation, native alias or source ABI is established by these modeled selections.')
    elif kernel.kind is KernelKind.GEMM:
        limitations+=('A/B storage encodings, scales and native scratch are unbound. Logical result is the mathematical GEMM result, not a claim that this launch publishes final C.',
            'Existing nominal-width GEMM pricing is conditional model work, not requested bytes of the opaque input selections.')
        if kernel.launch.tiling is None and kernel.launch.parameters['split_k_partial_bytes_per_element']:
            limitations+=('Observed partial output has unknown split count, K intervals and scratch extent; no partial allocation or zero-scratch claim is made.',)
    if kernel.variant in ('moe_align_opt_128','moe_enable_if'):
        limitations+=('INT32 expert state and two-word route records are the portable logical reference representation; native packed/compacted metadata widths and membership are not established by this summary.',)
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(granularity,Semantics.LOGICAL,Ordering.PARTIAL,False,limitations),bindings.allocations,
        bindings.views,(scope,),bindings.events,tuple(accesses))


def logical_interface_memory_access_program(kernel,hardware):
    actual,_=_prediction_bindings(kernel,None)
    if actual is None:
        actual=default_memory_bindings(kernel,hardware,call_id='logical.analytical',binding_id='logical.analytical',
            rank=0,stream='offline',analytical=True)
        actual = _own_implicit_memory_source(kernel, actual)
    if isinstance(actual,(MemoryInvocation,LogicalMemoryInvocation)) and actual.logical_hardware is not None:
        return actual.prediction_work(kernel,hardware)
    return build_memory_access_program(kernel,hardware,None,Granularity.KERNEL,
        program_id='vector.logical-interface')


def has_mathematical_gemm_source(kernel):
    """Existing unpacked FP16/FP32 matrix programs; no library launch is inferred."""
    from src.kernel.kernel_spec import KernelKind
    return (kernel.kind is KernelKind.GEMM and kernel.variant == 'gemm'
        and kernel.execution_support == 'unsupported' and kernel.launch is None
        and 'mathematical_program' in kernel.attributes
        and kernel.attributes['mathematical_program'] in ('unquantized_fp32_flops_bytes_v1',
            'unquantized_fp16_flops_bytes_v1', 'unquantized_bf16_flops_bytes_v1'))


def has_mathematical_memory_source(kernel):
    """Public mathematical operands, without claiming a native launch."""
    from src.kernel.kernel_spec import KernelKind
    if kernel.execution_support != 'unsupported' or kernel.launch is not None:
        return False
    if (kernel.hardware_id == 'nvidia_h200' and kernel.kind is KernelKind.VECTOR
            and kernel.variant == 'silu' and kernel.attributes['swiglu_limit'] is not None):
        return True
    if 'mathematical_program' not in kernel.attributes:
        return False
    program = kernel.attributes['mathematical_program']
    if kernel.kind is KernelKind.VECTOR and program in ('dense_fp16_source_flops_bytes_v1', 'dense_bf16_source_flops_bytes_v1'):
        return kernel.variant in ('rmsnorm','add_rmsnorm','silu','elementwise')
    if kernel.kind is KernelKind.FLASH_ATTN:
        return program in ('gqa_softmax_v1', 'bf16_mla_decode_softmax_v1',
            'bf16_mla_causal_empty_rows_softmax_v1', 'bf16_mla_absorbed_prefill_softmax_v1')
    return (kernel.kind is KernelKind.VECTOR and kernel.variant == 'moe_topk'
        and kernel.attributes['source_program'] in ('vllm_0_11_fp32_softmax_topk',
            'vllm_0_30_bf16_softmax_topk', 'vllm_0_24_bf16_softmax_topk',
            'vllm_0_18_bf16_softmax_topk')
        and not (kernel.attributes['source_program'] == 'vllm_0_18_bf16_softmax_topk'
                 and kernel.attributes['stage'] == 'owner_mask'))


def mathematical_operand_layouts(kernel):
    from src.kernel.kernel_spec import KernelKind
    if not has_mathematical_memory_source(kernel):
        raise ValueError('No mathematical memory source for this program')
    if kernel.kind is KernelKind.VECTOR and kernel.variant in ('rmsnorm','add_rmsnorm'):
        shape = (kernel.shape['rows'],kernel.shape['hidden'])
        reads = (('input',shape,kernel.dtype,False),)
        if kernel.variant == 'add_rmsnorm':
            reads += (('residual',shape,kernel.dtype,False),)
        reads += (('weight',(shape[1],),kernel.dtype,False),)
        writes = reads[:2] if kernel.variant == 'add_rmsnorm' else (('output',shape,kernel.dtype,False),)
        return reads,writes
    if kernel.kind is KernelKind.FLASH_ATTN:
        q, kv = sum(kernel.attributes['q_lens']), sum(kernel.attributes['kv_lens'])
        heads = kernel.shape['num_query_heads']
        if kernel.attributes['mathematical_program'] == 'gqa_softmax_v1':
            dim, kvheads = kernel.shape['head_dim'], kernel.shape['num_kv_heads']
            return (('query', (q, heads, dim), kernel.dtype, False),
                ('key', (kv, kvheads, dim), kernel.dtype, True),
                ('value', (kv, kvheads, dim), kernel.dtype, True)), (
                ('output', (q, heads, dim), kernel.dtype, False),)
        qk, value = kernel.shape['qk_head_dim'], kernel.shape['v_head_dim']
        writes = (('output', (q, heads, value), 'bf16', False),)
        if kernel.variant == 'mla_decode':
            writes += (('lse', (q, heads, 1), 'fp32', False),)
        return (('query', (q, heads, qk), 'bf16', False),
                ('compressed_kv', (kv, qk), 'bf16', True)), writes
    rows, experts, topk = (kernel.shape[name] for name in ('tokens', 'experts', 'topk'))
    source = kernel.attributes['source_program']
    dtype = 'bf16' if source == 'vllm_0_18_bf16_softmax_topk' else 'fp32'
    outputs = (('weights', (rows, topk), dtype, False),
               ('expert_ids', (rows, topk), 'int32', False))
    if source != 'vllm_0_18_bf16_softmax_topk':
        outputs += (('source_rows', (rows, topk), 'int32', False),)
    if source == 'vllm_0_11_fp32_softmax_topk' and kernel.attributes['normalize_topk_probability']:
        outputs += (('raw_weights', (rows, topk), 'fp32', False),
                    ('selected_weight_sum', (rows, 1), 'fp32', False))
    return (('logits', (rows, experts), kernel.dtype, False),), outputs


def standalone_mathematical_bindings(kernel, *, call_id, binding_id, rank=0,
                                      stream='offline', device_id='logical-local-device'):
    from math import prod
    from src.kernel.kernel_spec import DTYPE_BYTES
    if kernel.kind.value == 'VECTOR' and kernel.variant in ('silu', 'elementwise'):
        return standalone_elementwise_bindings(kernel,call_id=call_id,binding_id=binding_id,
            rank=rank,stream=stream,device_id=device_id)
    reads, writes = mathematical_operand_layouts(kernel)
    allocations, views, events, ready = [], [], [], []
    for role, shape, dtype, opaque in (*reads, *writes):
        if role in {view.view_id for view in views}:
            continue  # The fused norm writes its exact input and residual objects.
        width = DTYPE_BYTES[dtype]
        identity = f'{binding_id}.{role}'
        allocation = MemoryAllocation(identity, device_id, 'mathematical-public-operand',
            None if opaque else prod(shape) * width, AddressLevel.OBJECT, None, None,
            identity + '.allocated', None)
        view = (LogicalSelectionView(role, identity, shape, dtype, width, identity + '.selection', None)
            if opaque else TensorView(role, identity, 0, shape,
                tuple(prod(shape[i + 1:]) * width for i in range(len(shape))), dtype, width))
        allocations.append(allocation); views.append(view)
        events.append(ProgramEvent(allocation.allocated_event, EventKind.ALLOCATED, identity, None, None, ()))
        if role in {item[0] for item in reads}:
            kind = EventKind.LOGICAL_AVAILABLE if opaque else EventKind.INITIALIZED
            event = ProgramEvent(identity + '.ready', kind, identity, view.region(),
                'mathematical-public', (EventRef(allocation.allocated_event, EventKind.ALLOCATED),))
            events.append(event); ready.append(EventRef(event.event_id, kind))
    return ElementwiseBindings(binding_id, call_id, tuple(allocations), tuple(views), tuple(events),
        tuple(item[0] for item in reads), tuple(item[0] for item in writes), tuple(ready),
        rank, stream, 'mathematical-public')


def build_mathematical_memory_source(kernel, hardware, bindings, granularity, *, program_id):
    from math import prod
    from src.kernel.kernel_spec import DTYPE_BYTES
    if kernel.kind.value == 'VECTOR' and kernel.variant in ('silu', 'elementwise'):
        if not has_mathematical_memory_source(kernel):
            raise ValueError('No mathematical elementwise source for this program')
        if bindings is None:
            bindings = standalone_mathematical_bindings(kernel,call_id=program_id,binding_id=program_id)
        from dataclasses import replace
        program = build_logical_elementwise(kernel,bindings,granularity,program_id=program_id)
        return replace(program,coverage=replace(program.coverage,executable=False,
            limitations=(*program.coverage.limitations,'Mathematical typed scalar roles; native launch and issue service unbound.')))
    reads, writes = mathematical_operand_layouts(kernel)
    if granularity not in (Granularity.OP, Granularity.KERNEL, Granularity.TILE):
        raise ValueError('Mathematical source has no native instruction granularity')
    if bindings is None:
        bindings = standalone_mathematical_bindings(kernel, call_id=program_id, binding_id=program_id)
    if not isinstance(bindings, ElementwiseBindings):
        raise TypeError('Mathematical operands require ElementwiseBindings')
    expected_reads, expected_writes = tuple(item[0] for item in reads), tuple(item[0] for item in writes)
    if (bindings.read_views != expected_reads or set(bindings.write_views) != set(expected_writes)
            or len(bindings.write_views) != len(expected_writes) or len(bindings.input_ready) != len(reads)):
        raise ValueError('Mathematical source input/output/readiness roles differ')
    views = unique_by(bindings.views, 'view_id')
    if set(views) != {*expected_reads, *expected_writes}:
        raise ValueError('Mathematical source has unexpected or missing operands')
    for role, shape, dtype, opaque in (*reads, *writes):
        view = views[role]
        valid_shape = view.shape == shape
        if role == 'lse' and len(set(kernel.attributes['q_lens'])) == 1:
            valid_shape = valid_shape or view.shape == (len(kernel.attributes['q_lens']),
                kernel.shape['num_query_heads'], kernel.attributes['q_lens'][0])
        if (not valid_shape or view.dtype != dtype or view.element_bytes != DTYPE_BYTES[dtype]
                or not isinstance(view, (TensorView, LogicalSelectionView) if opaque else TensorView)):
            raise ValueError(f'Mathematical source {role} storage does not match its typed shape')
    input_parents = {views[role].allocation_id for role in expected_reads}
    output_parents = [views[role].allocation_id for role in expected_writes]
    fused_norm = kernel.kind.value == 'VECTOR' and kernel.variant == 'add_rmsnorm'
    if kernel.kind.value == 'VECTOR' and kernel.variant in ('rmsnorm','add_rmsnorm'):
        from src.kernel.memory_normalization import validate_normalization_operands
        validate_normalization_operands(kernel,bindings.views,bindings.read_views,bindings.write_views)
    if not fused_norm and (input_parents.intersection(output_parents) or len(set(output_parents)) != len(output_parents)):
        raise ValueError('Mathematical source output aliases lack a declared in-place program')
    scope = AccessScope(program_id, ('mathematical_public', kernel.attributes['mathematical_program']),
        bindings.call_id, (), bindings.rank, bindings.stream, 'mathematical-public-operands')
    allocations = unique_by(bindings.allocations, 'allocation_id')
    accesses = []
    evidence = ('Public mathematical source roles and data dependencies; native issue, workspace and page layout are unbound.',)
    def access(identity, role, operation, dependencies):
        view = views[role]
        item = MemoryAccess(program_id + '.' + identity, scope.scope_id, view.allocation_id,
            None if isinstance(view, LogicalSelectionView) else role, operation, view.region(), Semantics.LOGICAL, None, None, None, None,
            bindings.visibility_domain, (*dependencies, EventRef(
                allocations[view.allocation_id].allocated_event, EventKind.ALLOCATED)),
            EventKind.READ_READY if operation is Operation.READ else EventKind.WRITE_VISIBLE, 1, evidence)
        accesses.append(item)
        return EventRef(item.access_id, item.completion)
    inputs = tuple(access(role, role, Operation.READ, (ready,))
        for role, ready in zip(expected_reads, bindings.input_ready, strict=True))
    if fused_norm:
        updated = access('residual_update','residual',Operation.WRITE,inputs[:2])
        access('output','input',Operation.WRITE,(*inputs,updated))
    elif 'raw_weights' in expected_writes:
        raw = access('raw_weights', 'raw_weights', Operation.WRITE, inputs)
        for role in ('expert_ids', 'source_rows'):
            access(role, role, Operation.WRITE, inputs)
        summand = access('sum_input', 'raw_weights', Operation.READ, (raw,))
        summed = access('selected_weight_sum', 'selected_weight_sum', Operation.WRITE, (summand,))
        numerator = access('normalize_input', 'raw_weights', Operation.READ, (raw,))
        denominator = access('normalize_sum', 'selected_weight_sum', Operation.READ, (summed,))
        access('weights', 'weights', Operation.WRITE, (numerator, denominator))
    else:
        for role in expected_writes:
            access(role, role, Operation.WRITE, inputs)
    return MemoryAccessProgram('midlayer.memory-access.v1', program_id, hardware.hardware_id,
        bindings.binding_id, Coverage(Granularity.KERNEL, Semantics.LOGICAL, Ordering.PARTIAL,
            False, evidence), bindings.allocations, bindings.views, (scope,), bindings.events, tuple(accesses))


def mathematical_prediction_memory_bytes(kernel, hardware):
    from src.kernel.memory_path_solver import logical_requested_bytes
    source = build_memory_access_program(kernel, hardware, None, Granularity.KERNEL,
                                        program_id='mathematical.public')
    return sum(logical_requested_bytes(source, operation) for operation in (Operation.READ, Operation.WRITE))
