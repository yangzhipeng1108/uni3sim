"""Untimed collective participant tensor interfaces, separate from wire traffic."""
from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope, AddressLevel, Coverage, EventKind, EventRef, Granularity,
    MemoryAccess, MemoryAccessProgram, MemoryAllocation, Operation, Ordering,
    ProgramEvent, Semantics, TensorView, LogicalSelectionView, unique_by,TileAxis,TileAccessTemplate,TileDependency,TileProgram,TilePartition,
)


def has_communication_source(kernel):
    return (kernel.kind is KernelKind.COMMUNICATION and kernel.execution_support == "supported"
            and kernel.transfer.implementation != "sglang_flashinfer_allreduce_residual_rmsnorm")


def communication_operand_layouts(kernel):
    transfer = kernel.transfer
    if transfer.operation == "send_recv":
        reads, writes = (transfer.source_rank,), (transfer.destination_rank,)
    elif transfer.operation == "all_gatherv":
        return (tuple((rank, size) for rank, size in zip(transfer.participants, transfer.partition_bytes, strict=True) if size),
            tuple((rank, transfer.bytes) for rank in transfer.participants))
    elif transfer.operation == "reduce_scatterv":
        return (tuple((rank, transfer.bytes) for rank in transfer.participants),
            tuple((rank, size) for rank, size in zip(transfer.participants, transfer.partition_bytes, strict=True) if size))
    elif transfer.operation in ("all_reduce", "all_gather", "all_to_all"):
        reads = writes = transfer.participants
    else:
        raise ValueError("unknown collective interface operation")
    output_bytes = transfer.bytes * len(transfer.participants) if transfer.operation == "all_gather" else transfer.bytes
    return tuple((rank, transfer.bytes) for rank in reads), tuple((rank, output_bytes) for rank in writes)


def standalone_communication_bindings(kernel, *, call_id, binding_id, device_id=None, rank=0, stream="offline"):
    from src.kernel.memory_access_builder import ElementwiseBindings
    allocations, views, events, ready = [], [], [], []
    names = ([], [])
    for side, operands in enumerate(communication_operand_layouts(kernel)):
        for operand_rank, size in operands:
            name = f"rank{operand_rank}.{'input' if side == 0 else 'output'}"
            identity = f"{binding_id}.{name}"
            allocation = MemoryAllocation(identity, f"{kernel.hardware_id}.rank{operand_rank}" if device_id is None else device_id, "logical-communication-tensor",
                size, AddressLevel.OBJECT, None, None, f"{identity}.allocated", None)
            view = TensorView(name, identity, 0, (size,), (1,), "uint8", 1)
            allocations.append(allocation)
            views.append(view)
            names[side].append(name)
            events.append(ProgramEvent(allocation.allocated_event, EventKind.ALLOCATED, identity, None, None, ()))
            if side == 0:
                initialized = f"{identity}.initialized"
                events.append(ProgramEvent(initialized, EventKind.INITIALIZED, identity, view.region(), "collective-visible",
                    (EventRef(allocation.allocated_event, EventKind.ALLOCATED),)))
                ready.append(EventRef(initialized, EventKind.INITIALIZED))
    return ElementwiseBindings(binding_id, call_id, tuple(allocations), tuple(views), tuple(events),
        tuple(names[0]), tuple(names[1]), tuple(ready), rank, stream, "collective-visible")


def build_communication_interface(kernel, hardware, bindings, granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(bindings, ElementwiseBindings):
        raise TypeError("collective interface requires typed participant tensor bindings")
    if granularity not in (Granularity.OP, Granularity.KERNEL, Granularity.TILE):
        raise NotImplementedError("collective interface does not declare packet/tile issue order")
    inputs, outputs = communication_operand_layouts(kernel)
    if (len(bindings.read_views), len(bindings.write_views), len(bindings.input_ready)) != (len(inputs), len(outputs), len(inputs)):
        raise ValueError("collective participant operand/readiness count mismatch")
    views, allocations = unique_by(bindings.views, "view_id"), unique_by(bindings.allocations, "allocation_id")
    all_views = tuple(views[name] for name in (*bindings.read_views, *bindings.write_views))
    if kernel.transfer.operation in ("all_gatherv", "reduce_scatterv"):
        return build_variable_collective_tiles(kernel, bindings, inputs, outputs, all_views, program_id=program_id)
    numeric = kernel.transfer.operation == 'all_reduce' and any(view.dtype != 'uint8' for view in all_views)
    if numeric:
        return build_numeric_all_reduce_tiles(kernel,bindings,inputs,outputs,all_views,program_id=program_id)
    if any(isinstance(view,LogicalSelectionView) for view in all_views):
        raise NotImplementedError('logical participant selections require the numeric all-reduce ABI')
    for view, (_, size) in zip(all_views, (*inputs, *outputs)):
        if (view.shape, view.strides_bytes, view.dtype, view.element_bytes) != ((size,), (1,), "uint8", 1):
            raise ValueError("collective participant payload/storage mismatch")
    read_views, write_views = all_views[:len(inputs)], all_views[len(inputs):]
    if (len({view.allocation_id for view in read_views}) != len(read_views)
            or len({view.allocation_id for view in write_views}) != len(write_views)):
        raise NotImplementedError("peer participant alias requires an independently declared IPC mapping")
    for (destination, _), output in zip(outputs, write_views):
        for (source, _), operand in zip(inputs, read_views):
            if output.allocation_id != operand.allocation_id:
                continue
            if (kernel.transfer.operation != "all_reduce" or source != destination or output != operand):
                raise NotImplementedError("collective overlap needs an explicit participant/phase storage lifecycle")
    if kernel.transfer.operation != 'all_reduce':
        return build_collective_copy_tiles(kernel,bindings,inputs,outputs,read_views,write_views,program_id=program_id)
    return build_untyped_all_reduce_tiles(kernel,bindings,inputs,outputs,read_views,write_views,program_id=program_id)


def build_untyped_all_reduce_tiles(kernel,bindings,inputs,outputs,read_views,write_views, *, program_id):
    """Exact byte requests with an explicitly unknown scalar-word partition."""
    from src.kernel.memory_access import TilePartition
    evidence=('PhysicalTransferConfig all_reduce sum declares complete byte buffers but no arithmetic dtype.',
        'Output byte i requires every participant input byte in its own scalar-word equivalence class; membership remains unbound.',
        'No one-byte arithmetic, whole-tensor barrier, native ring passes or vendor workspace is inferred.')
    def scope(rank,role):
        return AccessScope(f'{program_id}.rank{rank}.{role}',('communication','all_reduce',role),bindings.call_id,(),
            rank,bindings.stream,'participant-unbound-word')
    accesses=[]
    for index,((rank,_),view) in enumerate(zip(inputs,read_views,strict=True)):
        accesses.append(TileAccessTemplate(f'read.rank{rank}',view.view_id,Operation.READ,('byte',),('byte',),
            (bindings.input_ready[index],),(),evidence,scope=scope(rank,'input')))
    required=tuple(TileDependency(access.access_id,EventKind.READ_READY,('byte',),partition_axis='byte') for access in accesses)
    for (rank,_),view in zip(outputs,write_views,strict=True):
        accesses.append(TileAccessTemplate(f'write.rank{rank}',view.view_id,Operation.WRITE,('byte',),('byte',),
            (),required,evidence,scope=scope(rank,'output')))
    parent=AccessScope(program_id,('communication','all_reduce','sum',*(str(rank) for rank in kernel.transfer.participants)),
        bindings.call_id,(),bindings.rank,bindings.stream,'all-reduce-unbound-words')
    family=TileProgram(program_id,parent,(TileAxis('byte',kernel.transfer.bytes,1),),tuple(accesses),
        bindings.visibility_domain,evidence,partitions=(TilePartition('byte'),))
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Byte extents and within-word reduction dependencies are explicit; scalar width and group membership remain unknown.',
            'Dependency expansion requires group membership; arithmetic execution additionally requires an actual numeric dtype.',
            'Wire/topology/protocol prices remain separate from logical participant requests.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))
    from src.kernel.memory_path_solver import collective_source_work
    collective_source_work(program);return program


def typed_all_reduce_bindings(kernel, bindings, dtype):
    """Refine a byte payload with an independently known caller scalar ABI."""
    from dataclasses import replace
    from src.kernel.kernel_spec import DTYPE_BYTES
    if kernel.transfer.operation != 'all_reduce' or kernel.transfer.reduction != 'sum':
        raise ValueError('typed collective reduction requires the explicit all_reduce sum operation')
    if dtype not in ('bf16','fp16','fp32','int8','int32','int64'):
        raise NotImplementedError('all-reduce scalar storage ABI is not declared for this dtype')
    width = DTYPE_BYTES[dtype]
    if kernel.transfer.bytes % width:
        raise ValueError('all-reduce payload must contain complete scalar elements')
    views = []
    for view in bindings.views:
        if isinstance(view,LogicalSelectionView):
            if (view.dtype,view.element_bytes,view.shape) != (dtype,width,(kernel.transfer.bytes//width,)):
                raise ValueError('all-reduce logical producer must supply the exact numeric element ABI')
            views.append(view)
            continue
        if (view.dtype,view.element_bytes,view.shape,view.strides_bytes) != ('uint8',1,(kernel.transfer.bytes,),(1,)):
            raise ValueError('all-reduce dtype refinement requires the original complete byte tensor')
        views.append(replace(view,shape=(kernel.transfer.bytes//width,),strides_bytes=(width,),dtype=dtype,element_bytes=width))
    initialized = {view.allocation_id:view.region() for view in views if view.view_id in bindings.read_views}
    events = tuple(replace(event,region=initialized[event.allocation_id])
        if event.kind is EventKind.INITIALIZED and event.allocation_id in initialized else event for event in bindings.events)
    return replace(bindings,views=tuple(views),events=events)


def build_numeric_all_reduce_tiles(kernel,bindings,inputs,outputs,operands, *, program_id):
    """Every scalar output depends on the matching word from every participant."""
    from src.kernel.kernel_spec import DTYPE_BYTES
    dtype = operands[0].dtype
    if kernel.transfer.reduction != 'sum' or dtype not in ('bf16','fp16','fp32','int8','int32','int64'):
        raise NotImplementedError('all-reduce tiles require a declared supported scalar sum ABI')
    width = DTYPE_BYTES[dtype]
    payload = kernel.transfer.bytes
    if payload % width:
        raise ValueError('all-reduce payload must contain complete scalar elements')
    for view in operands:
        if ((view.shape,view.dtype,view.element_bytes) != ((payload//width,),dtype,width)
                or (isinstance(view,TensorView) and view.strides_bytes != (width,))):
            raise ValueError('all-reduce participants must share the exact numeric element ABI')
    reads,writes = operands[:len(inputs)],operands[len(inputs):]
    if len({view.allocation_id for view in reads}) != len(reads) or len({view.allocation_id for view in writes}) != len(writes):
        raise NotImplementedError('typed all-reduce peer storage requires an actual mapping')
    for (destination,_),output in zip(outputs,writes):
        for (source,_),operand in zip(inputs,reads):
            if output.allocation_id == operand.allocation_id and (source != destination or output.region() != operand.region()):
                raise NotImplementedError('typed all-reduce overlap requires the matching rank and exact word geometry')
    evidence = ('Caller-bound numeric participant tensors independently define scalar type and width; no dtype inferred from byte count or duration.',
        'all_reduce sum: output element i requires input element i of every actual participant, independently of other elements.',
        'Wire bytes/topology/protocol, reduction instruction cost and vendor scratch remain separate projections.')
    accesses = []
    def scope(rank,role):
        return AccessScope(f'{program_id}.rank{rank}.{role}',('communication','all_reduce',role),bindings.call_id,(),
            rank,bindings.stream,'participant-numeric-reduction')
    for index,((rank,_),view) in enumerate(zip(inputs,operands[:len(inputs)],strict=True)):
        accesses.append(TileAccessTemplate(f'read.rank{rank}',view.view_id,Operation.READ,('element',),('element',),
            (bindings.input_ready[index],),(),evidence,unmapped=isinstance(view,LogicalSelectionView),scope=scope(rank,'input')))
    required = tuple(TileDependency(access.access_id,EventKind.READ_READY,()) for access in accesses)
    for (rank,_),view in zip(outputs,operands[len(inputs):],strict=True):
        accesses.append(TileAccessTemplate(f'write.rank{rank}',view.view_id,Operation.WRITE,('element',),('element',),
            (),required,evidence,unmapped=isinstance(view,LogicalSelectionView),scope=scope(rank,'output')))
    parent = AccessScope(program_id,('communication','all_reduce','sum',*tuple(str(rank) for rank in kernel.transfer.participants)),
        bindings.call_id,(),bindings.rank,bindings.stream,'all-reduce-elements')
    family = TileProgram(program_id,parent,(TileAxis('element',payload//width,1),),tuple(accesses),bindings.visibility_domain,evidence)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,not any(isinstance(view,LogicalSelectionView) for view in operands),(
            'Typed per-element scalar sum dependencies; independent elements do not acquire a whole-tensor barrier.',
            'Arithmetic precision/association, wire protocols, workspace, physical peer mapping and fused residual/norm side effects remain separate or pending.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))
    from src.kernel.memory_path_solver import collective_source_work
    collective_source_work(program)
    return program


def build_collective_copy_tiles(kernel,bindings,inputs,outputs,read_views,write_views, *, program_id):
    """Logical peer-segment copies; network packets and wire work stay separate."""
    operation=kernel.transfer.operation; ranks=kernel.transfer.participants
    payload=kernel.transfer.bytes
    chunk=payload//len(ranks) if operation == 'all_to_all' else payload
    evidence=('PhysicalTransferConfig ordered participants define logical gather concatenation and all-to-all peer split order.',
        'Each semantic byte tile preserves source/destination selection and actual contributor READ_READY; no packet/native issue order.',
        'Ring order affects the existing network model, not logical tensor segment placement.')
    views=list(bindings.views); accesses=[]
    def scope(rank,role):
        return AccessScope(f'{program_id}.rank{rank}.{role}',('communication',operation,role),bindings.call_id,(),
            rank,bindings.stream,'participant-logical-byte-tiles')
    def segment(view,start,size,label):
        selected=TensorView(f'{view.view_id}.{label}',view.allocation_id,view.offset_bytes+start,(size,),(1,),'uint8',1)
        views.append(selected);return selected
    reads={}
    for index,((rank,_),view) in enumerate(zip(inputs,read_views,strict=True)):
        peers=ranks if operation == 'all_to_all' else (None,)
        for peer_index,peer in enumerate(peers):
            selected=segment(view,peer_index*chunk,chunk,f'to{peer}') if operation == 'all_to_all' else view
            identity=f'read.rank{rank}.peer{peer}'
            accesses.append(TileAccessTemplate(identity,selected.view_id,Operation.READ,('byte',),('byte',),
                (bindings.input_ready[index],),(),evidence,scope=scope(rank,'input')))
            reads[(rank,peer)]=identity
    for destination_index,((rank,_),view) in enumerate(zip(outputs,write_views,strict=True)):
        contributors=tuple(source for source,_ in inputs)
        for source_index,source in enumerate(contributors):
            if operation == 'all_gather':
                selected=segment(view,source_index*payload,payload,f'from{source}')
            elif operation == 'all_to_all':selected=segment(view,source_index*chunk,chunk,f'from{source}')
            else:selected=view
            read=reads[(source,rank if operation == 'all_to_all' else None)]
            accesses.append(TileAccessTemplate(f'write.rank{rank}.from{source}',selected.view_id,Operation.WRITE,
                ('byte',),('byte',),(),(TileDependency(read,EventKind.READ_READY,()),),evidence,scope=scope(rank,'output')))
    parent=AccessScope(program_id,('communication',operation,*tuple(str(rank) for rank in ranks)),bindings.call_id,(),bindings.rank,bindings.stream,'collective-copy-coordinates')
    family=TileProgram(program_id,parent,(TileAxis('byte',chunk,1),),tuple(accesses),bindings.visibility_domain,evidence)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,True,(
            'Logical participant byte copies and exact peer segments only; wire/protocol overhead and native workspace are separate.',
            'Participant rank scopes/devices survive expansion; no all-rank data barrier for independent copies.',
            'All-reduce arithmetic requires an independently declared numeric element ABI before its full tile migration.')),
        bindings.allocations,tuple(views),(),bindings.events,(),(family,))
    from src.kernel.memory_path_solver import collective_source_work
    collective_source_work(program)
    return program


def build_variable_collective_tiles(kernel, bindings, inputs, outputs, operands, *, program_id):
    """Ordered variable gather or scalar-sum scatter with no empty allocations.

    Each positive segment owns its own compressed coordinate loop. Zero-size
    ranks still belong to the communication group but have no payload operand
    on that side. A byte-only sum retains an unknown word partition.
    """
    from src.kernel.kernel_spec import DTYPE_BYTES
    transfer = kernel.transfer
    gather = transfer.operation == "all_gatherv"
    dtype = operands[0].dtype
    if dtype not in ("uint8", "bf16", "fp16", "fp32", "int8", "int32", "int64"):
        raise NotImplementedError("variable collective requires an explicit scalar storage ABI")
    width = DTYPE_BYTES[dtype]
    if any(size % width for size in transfer.partition_bytes):
        raise ValueError("collective partitions must contain complete scalar elements")
    for view, (_, size) in zip(operands, (*inputs, *outputs), strict=True):
        if (not isinstance(view, TensorView) or view.shape != (size // width,)
                or (view.dtype, view.element_bytes, view.strides_bytes) != (dtype, width, (width,))):
            raise ValueError("variable collective must bind exact contiguous participant operands")
    reads, writes = operands[:len(inputs)], operands[len(inputs):]
    if len({view.allocation_id for view in operands}) != len(operands):
        raise NotImplementedError("variable collective aliases require an explicit participant storage lifecycle")
    read_by_rank = dict(zip((rank for rank, _ in inputs), reads, strict=True))
    write_by_rank = dict(zip((rank for rank, _ in outputs), writes, strict=True))
    ready_by_rank = dict(zip((rank for rank, _ in inputs), bindings.input_ready, strict=True))
    ranks, sizes = transfer.participants, transfer.partition_bytes
    header = ("communication", transfer.operation, "partitioned", str(len(ranks)),
        *(str(rank) for rank in ranks), *(str(size) for size in sizes))
    axis = "byte" if dtype == "uint8" else "element"
    unknown_sum = not gather and dtype == "uint8"
    evidence = ("Ordered participant segment sizes come from the current collective inputs, independently of duration.",
        "Gather preserves source segments; scatter sum requires the corresponding segment from every input rank.",
        "Zero segments own no allocation; native protocol, scratch and reduction arithmetic cost remain separate.")
    views, families, prefix = list(bindings.views), [], 0
    for index, size in enumerate(sizes):
        if not size:
            continue
        accesses = []
        sources = (ranks[index],) if gather else ranks
        destinations = ranks if gather else (ranks[index],)
        for rank in sources:
            parent = read_by_rank[rank]
            view = TensorView(f"{parent.view_id}.partition{index}", parent.allocation_id,
                parent.offset_bytes + (0 if gather else prefix), (size // width,), (width,), dtype, width)
            views.append(view)
            scope = AccessScope(f"{program_id}.partition{index}.rank{rank}.input",
                ("communication", transfer.operation, "input"), bindings.call_id, (), rank, bindings.stream,
                "participant-variable-collective")
            accesses.append(TileAccessTemplate(f"read.rank{rank}", view.view_id, Operation.READ, (axis,), (axis,),
                (ready_by_rank[rank],), (), evidence, scope=scope))
        dependencies = tuple(TileDependency(access.access_id, EventKind.READ_READY,
            (axis,) if unknown_sum else (), partition_axis=axis if unknown_sum else None) for access in accesses)
        for rank in destinations:
            parent = write_by_rank[rank]
            view = TensorView(f"{parent.view_id}.partition{index}", parent.allocation_id,
                parent.offset_bytes + (prefix if gather else 0), (size // width,), (width,), dtype, width)
            views.append(view)
            scope = AccessScope(f"{program_id}.partition{index}.rank{rank}.output",
                ("communication", transfer.operation, "output"), bindings.call_id, (), rank, bindings.stream,
                "participant-variable-collective")
            accesses.append(TileAccessTemplate(f"write.rank{rank}", view.view_id, Operation.WRITE, (axis,), (axis,),
                (), dependencies, evidence, scope=scope))
        scope = AccessScope(f"{program_id}.partition{index}", (*header, str(index)), bindings.call_id, (),
            bindings.rank, bindings.stream, "variable-collective-segments")
        families.append(TileProgram(scope.scope_id, scope, (TileAxis(axis, size // width, 1),), tuple(accesses),
            bindings.visibility_domain, evidence, partitions=(TilePartition(axis),) if unknown_sum else ()))
        prefix += size
    program = MemoryAccessProgram("midlayer.memory-access.v1", program_id, kernel.hardware_id, bindings.binding_id,
        Coverage(Granularity.TILE, Semantics.LOGICAL, Ordering.PARTIAL, not unknown_sum, evidence),
        bindings.allocations, tuple(views), (), bindings.events, (), tuple(families))
    from src.kernel.memory_path_solver import collective_source_work
    collective_source_work(program)
    return program
