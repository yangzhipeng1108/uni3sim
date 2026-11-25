"""Separate BF16 MoE SiLU and multiplication, including opaque iterator slices."""
from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope,AddressLevel,Coverage,EventKind,EventRef,Granularity,
    MemoryAccessProgram,MemoryAllocation,Operation,Ordering,ProgramEvent,Semantics,
    TensorView,Unmapped,unique_by,TileAxis,TileAccessTemplate,TileDependency,TileProgram,
)


def has_moe_pointwise_source(kernel,hardware):
    if (kernel.kind is not KernelKind.VECTOR or kernel.variant not in ('moe_silu','moe_mul')
            or kernel.execution_support != 'supported' or kernel.dtype != 'bf16'
            or hardware.kernel_memory_model != 'gpu_mshr'):
        return False
    selected = kernel.shape['tokens']*kernel.shape['topk']*kernel.shape['intermediate']*kernel.attributes['work_fraction']
    return selected > 0 and selected.is_integer()


def _layout(kernel):
    rows,columns = kernel.shape['tokens']*kernel.shape['topk'],kernel.shape['intermediate']
    selected = int(rows*columns*kernel.attributes['work_fraction'])
    if selected <= 0 or selected != rows*columns*kernel.attributes['work_fraction']:
        raise ValueError('MoE pointwise source requires an integral positive iterator slice')
    inputs = ('gate',) if kernel.variant == 'moe_silu' else ('silu','up')
    return rows,columns,selected,inputs


def _region(view,binding_id,selected):
    if selected == view.shape[0]*view.shape[1]:
        return view.region()
    return Unmapped((selected,),'bf16',2,f'{binding_id}.{view.view_id}.tensor_iterator_slice')


def standalone_moe_pointwise_bindings(kernel, *, call_id,binding_id, device_id="logical-local-device", rank=0, stream="offline"):
    from src.kernel.memory_access_builder import ElementwiseBindings
    rows,columns,selected,inputs = _layout(kernel)
    allocations,views,events,ready = [],[],[],[]
    for role in (*inputs,'output'):
        identity = f'{binding_id}.{role}'
        view = TensorView(role,identity,0,(rows,columns),(columns*2,2),'bf16',2)
        allocation = MemoryAllocation(identity,device_id,'logical-storage',rows*columns*2,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation); views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role in inputs:
            region = _region(view,binding_id,selected)
            kind = EventKind.LOGICAL_AVAILABLE if isinstance(region,Unmapped) else EventKind.INITIALIZED
            event = ProgramEvent(f'{identity}.available',kind,identity,region,'moe-pointwise-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event); ready.append(EventRef(event.event_id,kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        inputs,('output',),tuple(ready),rank,stream,'moe-pointwise-visible')


def build_moe_pointwise_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('MoE pointwise kernels require typed BF16 input/output bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise ValueError('unknown MoE pointwise logical granularity')
    rows,columns,selected,inputs = _layout(kernel)
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (inputs,('output',),len(inputs)):
        raise ValueError('MoE pointwise operand roles mismatch')
    views,allocations = unique_by(bindings.views,'view_id'),unique_by(bindings.allocations,'allocation_id')
    if set(views) != {*inputs,'output'}:
        raise ValueError('MoE pointwise views must be complete and explicit')
    if views['output'].allocation_id in {views[role].allocation_id for role in inputs}:
        raise ValueError('Standalone torch SiLU/multiply allocates a distinct output')
    for view in views.values():
        if ((view.shape,view.dtype,view.element_bytes) != ((rows,columns),'bf16',2)
                or view.strides_bytes[-1] != 2 or view.strides_bytes[0] < columns*2):
            raise ValueError('MoE pointwise source BF16 matrix/stride mismatch')
    evidence = ('MetaX fused_moe_runner:gate/up are W1 halves; intermediate_cache2.copy_(torch.nn.functional.silu(gate)*up)',
        'PyTorch separate SiLU reads gate/writes BF16 result; multiply reads both inputs/writes BF16 product',
        'work_fraction is the explicit source-derived TensorIterator per-invocation slice; missing offsets remain unbound')
    full = selected == rows*columns
    selected_views = (() if full else tuple(TensorView(f'{role}.selection',views[role].allocation_id,0,
        (selected,),(2,),'bf16',2) for role in (*inputs,'output')))
    active = views if full else {role:view for role,view in zip((*inputs,'output'),selected_views,strict=True)}
    axes = (TileAxis('row',rows,1),TileAxis('column',columns,1)) if full else (TileAxis('selected',selected,1),)
    names = tuple(axis.axis_id for axis in axes)
    evidence += ('Independent scalar-coordinate work units; partial slice ordinals retain unknown T/H mapping and no native issue claim.',)
    templates = tuple(TileAccessTemplate(role,active[role].view_id,Operation.READ,names,names,
        (ready,),(),evidence,unmapped=not full) for role,ready in zip(inputs,bindings.input_ready,strict=True))
    output = TileAccessTemplate('output',active['output'].view_id,Operation.WRITE,names,names,(),
        tuple(TileDependency(role,EventKind.READ_READY,()) for role in inputs),evidence,unmapped=not full)
    scope = AccessScope(program_id,('moe_pointwise',kernel.variant),bindings.call_id,(),bindings.rank,bindings.stream,'logical-moe-pointwise')
    family = TileProgram(program_id,scope,axes,(*templates,output),bindings.visibility_domain,evidence)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,
            ('Full BF16 matrix roles retain declared strides; partial TensorIterator scalar ordinals have no guessed parent offsets or physical footprint.',
             'No fabricated cross-call producer identity; separate slices are not proven disjoint.',
             'Native packing/split-order/scratch/issued transactions remain separate or pending.')),
        bindings.allocations,(*bindings.views,*selected_views),(),bindings.events,(),(family,))
    moe_pointwise_source_demands(program)
    return program


def moe_pointwise_source_demands(program):
    if not program.tile_programs or program.accesses:
        raise ValueError('MoE pointwise requires its unique tile source')
    from math import prod
    from src.kernel.memory_path_solver import logical_requested_bytes
    program.__post_init__()
    if len(program.tile_programs) != 1:
        raise ValueError('MoE pointwise requires one actual selection loop')
    family, = program.tile_programs
    if family.scope.op_ids not in (('moe_pointwise','moe_silu'),('moe_pointwise','moe_mul')):
        raise ValueError('MoE pointwise source variant missing')
    inputs = ('gate',) if family.scope.op_ids[1] == 'moe_silu' else ('silu','up')
    templates = unique_by(family.accesses,'access_id'); views = unique_by(program.views,'view_id')
    if set(templates) != {*inputs,'output'}:
        raise ValueError('MoE pointwise source operands incomplete')
    rows,columns = views['output'].shape; selected = prod(views[templates['output'].view_id].shape)
    full = selected == rows*columns
    expected_axes = (('row',rows,1),('column',columns,1)) if full else (('selected',selected,1),)
    if not 0 < selected <= rows*columns or tuple((axis.axis_id,axis.extent,axis.block) for axis in family.axes) != expected_axes:
        raise ValueError('MoE pointwise iterator selection exceeds full parent')
    names = tuple(axis.axis_id for axis in family.axes)
    for role in (*inputs,'output'):
        access = templates[role]; parent = views[role]; active = views[access.view_id]
        if ((parent.shape,parent.dtype,parent.element_bytes) != ((rows,columns),'bf16',2)
                or parent.strides_bytes[-1] != 2 or parent.strides_bytes[0] < columns*2
                or (active.allocation_id,active.dtype,active.element_bytes) != (parent.allocation_id,'bf16',2)
                or active.shape != ((rows,columns) if full else (selected,))
                or access.iteration_axes != names or access.view_axes != names or access.view_axis_divisors
                or access.unmapped != (not full) or access.indirect_target is not None
                or access.operation is not (Operation.WRITE if role == 'output' else Operation.READ)):
            raise ValueError('MoE pointwise source must conserve actual BF16 input/output slice regions')
    if views['output'].allocation_id in {views[role].allocation_id for role in inputs}:
        raise ValueError('MoE pointwise standalone output cannot alias its inputs')
    if templates['output'].dependencies != tuple(TileDependency(role,EventKind.READ_READY,()) for role in inputs):
        raise ValueError('MoE pointwise output lacks its actual input READ_READY causality')
    return {'elements':selected,'full_shape':(rows,columns),'full_invocation':full,
        'read_bytes':logical_requested_bytes(program,Operation.READ),'write_bytes':logical_requested_bytes(program,Operation.WRITE)}
