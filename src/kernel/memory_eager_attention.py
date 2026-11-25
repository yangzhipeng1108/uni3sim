"""Logical primitive interfaces of the supported compatibility Eager workload."""
from dataclasses import dataclass

from src.kernel.memory_access import (
    AccessScope, AddressLevel, Coverage, EventKind, EventRef, Granularity, MemoryAccess,
    MemoryAccessProgram, MemoryAllocation, Operation, Ordering, ProgramEvent, Semantics,
    Span, TensorView, Typed, unique_by,TileAxis,TileAccessTemplate,TileDependency,TileProgram,
)
from src.kernel.kernel_spec import KernelKind


@dataclass(frozen=True)
class EagerAttentionBindings(Typed):
    binding_id: str
    call_id: str
    allocations: tuple[MemoryAllocation, ...]
    views: tuple[TensorView, ...]
    events: tuple[ProgramEvent, ...]
    query_view: str
    key_view: str
    value_view: str
    score_view: str
    probability_view: str
    output_view: str
    query_ready: EventRef
    key_ready: EventRef
    value_ready: EventRef
    rank: int
    stream: str
    visibility_domain: str


def has_eager_attention_source(kernel, hardware):
    return (kernel.kind is KernelKind.EAGER_ATTN and kernel.variant == "decomposed"
            and hardware.kernel_memory_model in ("pim_bank", "systolic_scratchpad"))


def eager_operand_shapes(kernel):
    # This is the actual compatibility primitive shape used by the dataset
    # generators: mm [Q,D,KV,B*HQ] folds its fourth field into matrix M.
    # It is not a GQA/batched-KV storage claim.
    rows = kernel.shape["batch"] * kernel.shape["num_query_heads"] * kernel.shape["q_len"]
    width, length = kernel.shape["head_dim"], kernel.shape["kv_len"]
    return ((rows, width), (width, length), (length, width),
            (rows, length), (rows, length), (rows, width))


def standalone_eager_attention_bindings(kernel, *, call_id, binding_id):
    allocations, views, events = [], [], []
    names = ("query", "key", "value", "score", "probability", "output")
    width = kernel.shape["dtype_bytes"]
    for index, (name, shape) in enumerate(zip(names, eager_operand_shapes(kernel))):
        identity = f"{binding_id}.{name}"
        allocation = MemoryAllocation(identity, "logical-local-device", "logical-storage",
            shape[0]*shape[1]*width, AddressLevel.OBJECT, None, None, f"{identity}.allocated", None)
        allocations.append(allocation)
        views.append(TensorView(name, identity, 0, shape, (shape[1]*width, width), kernel.dtype, width))
        events.append(ProgramEvent(allocation.allocated_event, EventKind.ALLOCATED, identity, None, None, ()))
        if index < 3:
            events.append(ProgramEvent(f"{identity}.initialized", EventKind.INITIALIZED, identity,
                Span(0, allocation.size_bytes), "eager-visible", (EventRef(allocation.allocated_event, EventKind.ALLOCATED),)))
    return EagerAttentionBindings(binding_id, call_id, tuple(allocations), tuple(views), tuple(events), *names,
        *(EventRef(f"{binding_id}.{name}.initialized", EventKind.INITIALIZED) for name in names[:3]),
        0, "offline", "eager-visible")


def build_eager_attention(kernel, hardware, bindings, granularity, *, program_id):
    if not isinstance(bindings, EagerAttentionBindings):
        raise TypeError("Eager attention requires typed Q/K/V/temporary/output bindings")
    if granularity not in (Granularity.OP, Granularity.KERNEL,Granularity.TILE):
        raise ValueError("unknown Eager logical granularity")
    views = unique_by(bindings.views, "view_id")
    names = (bindings.query_view, bindings.key_view, bindings.value_view,
             bindings.score_view, bindings.probability_view, bindings.output_view)
    operands = tuple(views[name] for name in names)
    for view, shape in zip(operands, eager_operand_shapes(kernel)):
        if (view.shape, view.dtype, view.element_bytes) != (shape, kernel.dtype, kernel.shape["dtype_bytes"]):
            raise ValueError("Eager operand/temporary shape or storage type mismatch")
        if view.strides_bytes != (shape[1]*view.element_bytes, view.element_bytes):
            raise NotImplementedError("compatibility Eager projection requires dense primitive interfaces")
    if len({view.allocation_id for view in operands}) != 6:
        raise NotImplementedError("Eager temporary/input alias needs a declared implementation lifecycle")
    query,key,value,score,probability,output=operands;rows,dimension=query.shape;length=key.shape[1]
    evidence=('Compatibility primitive ABI: QK matrix dot products materialize score, row softmax materializes probability, PV dots materialize output.',
        'One scalar output/dot is a semantic work unit; complete K or KV reductions stay on chip, not a native systolic/CTA issue claim.',
        'Probability at each column requires the whole actual score row; PV reads actual written probabilities, not fabricated initializers.')
    def scope(stage):
        return AccessScope(f'{program_id}.{stage}',('eager_attention',stage),bindings.call_id,(),bindings.rank,bindings.stream,'compatibility-primitive-tiles')
    accesses=(
        TileAccessTemplate('qkt.query',query.view_id,Operation.READ,('m','n'),('m',None),(bindings.query_ready,),(),evidence,scope=scope('qkt')),
        TileAccessTemplate('qkt.key',key.view_id,Operation.READ,('m','n'),(None,'n'),(bindings.key_ready,),(),evidence,scope=scope('qkt')),
        TileAccessTemplate('qkt.score',score.view_id,Operation.WRITE,('m','n'),('m','n'),(),(
            TileDependency('qkt.query',EventKind.READ_READY,()),TileDependency('qkt.key',EventKind.READ_READY,())),evidence,scope=scope('qkt')),
        TileAccessTemplate('softmax.score',score.view_id,Operation.READ,('m','n'),('m','n'),(),
            (TileDependency('qkt.score',EventKind.WRITE_VISIBLE,()),),evidence,scope=scope('softmax')),
        TileAccessTemplate('softmax.probability',probability.view_id,Operation.WRITE,('m','n'),('m','n'),(),
            (TileDependency('softmax.score',EventKind.READ_READY,('n',)),),evidence,scope=scope('softmax')),
        TileAccessTemplate('pv.probability',probability.view_id,Operation.READ,('m','n','d'),('m','n'),(),
            (TileDependency('softmax.probability',EventKind.WRITE_VISIBLE,()),),evidence,scope=scope('pv')),
        TileAccessTemplate('pv.value',value.view_id,Operation.READ,('m','n','d'),('n','d'),(bindings.value_ready,),(),evidence,scope=scope('pv')),
        TileAccessTemplate('pv.output',output.view_id,Operation.WRITE,('m','d'),('m','d'),(),(
            TileDependency('pv.probability',EventKind.READ_READY,('n',)),TileDependency('pv.value',EventKind.READ_READY,('n',))),evidence,scope=scope('pv')))
    parent=AccessScope(program_id,('eager_attention','decomposed'),bindings.call_id,(),bindings.rank,bindings.stream,'compatibility-eager-row-reductions')
    family=TileProgram(program_id,parent,(TileAxis('m',rows,1),TileAxis('n',length,1),TileAxis('d',dimension,1)),
        accesses,bindings.visibility_domain,evidence)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,True,(
            'Compatibility folded primitive rows share K/V; this is not a batched/GQA storage mapping.',
            'Scores and probabilities remain distinct materialized temporary parents; row reduction and output-coordinate dependencies are explicit.',
            'PIM bank-column rereads/exp service and systolic unpriced glue remain existing separate projections; int8 is a compatibility storage convention.',
            'No physical bank/page/issued or host inter-stage transfer claim.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))
    from src.kernel.memory_path_solver import eager_attention_interface_demands
    eager_attention_interface_demands(program)
    return program
