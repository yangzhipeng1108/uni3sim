"""Primitive interfaces of the existing compatibility unfused-MoE workload."""
from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope, AddressLevel, Coverage, EventKind, EventRef, Granularity,
    MemoryAccessProgram, MemoryAllocation, Operation, Ordering, ProgramEvent, Semantics,
    TensorView, TileAxis, TileAccessTemplate, TileDependency, TileProgram, unique_by, requested_bytes,
)


def has_unfused_moe_source(kernel, hardware):
    return (kernel.kind is KernelKind.UNFUSED_MOE and kernel.variant == 'decomposed'
            and kernel.execution_support == 'supported'
            and hardware.kernel_memory_model in ('pim_bank','systolic_scratchpad'))


def unfused_moe_primitive_layouts(kernel):
    """Actual mm/softmax/elewise shapes; no native top-k gather/scatter claim."""
    tokens, experts, hidden, intermediate = (kernel.shape[name] for name in ('tokens','experts','hidden','intermediate'))
    counts = kernel.attributes['tokens_per_expert']
    return _primitive_layouts(tokens,experts,hidden,intermediate,counts)


def _primitive_layouts(tokens,experts,hidden,intermediate,counts):
    stages = [('router','mm',(('router_input',(tokens,hidden)),('router_weight',(hidden,experts))),
               ('router_logits',(tokens,experts))),
              ('router_softmax','softmax',(('router_logits',(tokens,experts)),),
               ('router_probability',(tokens,experts)))]
    for expert,count in enumerate(counts):
        if count == 0:
            continue
        prefix = f'expert{expert}'
        stages += [(f'{prefix}.w1','mm',((f'{prefix}.input',(count,hidden)),
                    (f'{prefix}.w1_weight',(hidden,2*intermediate))),
                   (f'{prefix}.w1_output',(count,2*intermediate))),
                   (f'{prefix}.pointwise','pointwise',((f'{prefix}.pointwise_lhs',(count,intermediate)),
                    (f'{prefix}.pointwise_rhs',(count,intermediate))),
                   (f'{prefix}.activation',(count,intermediate))),
                   (f'{prefix}.w2','mm',((f'{prefix}.activation',(count,intermediate)),
                    (f'{prefix}.w2_weight',(intermediate,hidden))),
                   (f'{prefix}.output',(count,hidden)))]
    # The generator runs elewise over routed_rows*hidden, not a native
    # top-k reduction producing tokens*hidden. Keep that distinction visible.
    routed = sum(counts)
    stages.append(('output_pointwise','pointwise',(('output_lhs',(routed,hidden)),('output_rhs',(routed,hidden))),
                   ('output',(routed,hidden))))
    return tuple(stages)


def standalone_unfused_moe_bindings(kernel, *, call_id, binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    stages = unfused_moe_primitive_layouts(kernel)
    produced = {name for _,_,_,(name,_) in stages}
    layouts = {}
    for _,_,inputs,output in stages:
        for name,shape in (*inputs,output):
            if name in layouts and layouts[name] != shape:
                raise ValueError('unfused-MoE primitive role geometry changed')
            layouts[name] = shape
    allocations,views,events,ready,reads = [],[],[],[],[]
    width = kernel.shape['dtype_bytes']
    for name,shape in layouts.items():
        identity = f'{binding_id}.{name}'
        view = TensorView(name,identity,0,shape,(shape[1]*width,width),kernel.dtype,width)
        allocation = MemoryAllocation(identity,'logical-local-device','logical-storage',shape[0]*shape[1]*width,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation)
        views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if name not in produced:
            initialized = f'{identity}.initialized'
            events.append(ProgramEvent(initialized,EventKind.INITIALIZED,identity,view.region(),'moe-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),)))
            reads.append(name)
            ready.append(EventRef(initialized,EventKind.INITIALIZED))
    outputs = tuple(name for name in layouts if name in produced)
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),
        tuple(reads),outputs,tuple(ready),0,'offline','moe-visible')


def build_unfused_moe_source(kernel, hardware, bindings, granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not isinstance(bindings,ElementwiseBindings):
        raise TypeError('unfused-MoE requires typed primitive operand bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):
        raise NotImplementedError('unfused-MoE exposes logical tiles, not native bank issue order')
    stages = unfused_moe_primitive_layouts(kernel)
    expected = standalone_unfused_moe_bindings(kernel,call_id=bindings.call_id,binding_id=bindings.binding_id)
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready)) != (
            expected.read_views,expected.write_views,len(expected.input_ready)):
        raise ValueError('unfused-MoE primitive input/output role count mismatch')
    views = unique_by(bindings.views,'view_id')
    if set(views) != {view.view_id for view in expected.views}:
        raise ValueError('unfused-MoE primitive operand role missing')
    for reference in expected.views:
        actual = views[reference.view_id]
        if (actual.shape,actual.strides_bytes,actual.dtype,actual.element_bytes) != (
                reference.shape,reference.strides_bytes,reference.dtype,reference.element_bytes):
            raise ValueError('unfused-MoE primitive operand storage mismatch')
    if len({view.allocation_id for view in views.values()}) != len(views):
        raise NotImplementedError('unfused-MoE primitive alias requires an explicit caller lifecycle')
    family = _primitive_tiles(stages,views,dict(zip(bindings.read_views,bindings.input_ready)),
        program_id,bindings.call_id,bindings.rank,bindings.stream,bindings.visibility_domain)
    program = MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,True,(
            'Logical scalar dots, softmax rows and elementwise work; no native bank/tile issue order.',
            'Routing, expert input gather, W1 packing and final aggregation mappings remain independent boundary inputs.',
            'Final pointwise work covers routed rows, not a native top-k reduction; int8 is scalar compatibility storage without a native quantized MoE ABI.',
            'The static primitive calls are serial; each boundary waits for all preceding outputs, not a hardware barrier.',
            'Timeloop software glue remains unpriced; bank-stream and compulsory-interface prices do not count logical dot rereads.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))
    unfused_moe_tile_demands(program)
    return program


def _primitive_tiles(stages,views,readiness,program_id,call_id,rank,stream,visibility):
    """One compressed primitive DAG, including real materialized intermediates."""
    evidence = ('samsung_hbm_pim.py:_decomp_moe_latency and Timeloop static mm/softmax/elewise calls.',
        'Logical dot products reduce the actual K dimension; softmax publication waits for its complete row.',
        'UniNDP elewise_micro reads two operands per scalar result; unresolved native routing is not inferred.')
    axes,templates,produced = {},[],{}
    tokens,hidden = views['router_input'].shape
    experts = views['router_weight'].shape[1]
    axes.update((('token',tokens),('hidden',hidden),('expert',experts)))
    previous = None
    for index,(name,kind,inputs,(output,_)) in enumerate(stages):
        if name == 'router':
            row,column,reduction = 'token','expert','hidden'
        elif name == 'router_softmax':
            row,column,reduction = 'token','expert',None
        elif name == 'output_pointwise':
            row,column,reduction = 'routed','hidden',None
        else:
            expert = name.split('.')[0]
            row = f'{expert}.row'
            column = f'{expert}.gate' if name.endswith('.w1') else 'hidden' if name.endswith('.w2') else f'{expert}.inner'
            reduction = 'hidden' if name.endswith('.w1') else f'{expert}.inner' if name.endswith('.w2') else None
        out = views[output]
        for axis,extent in zip((row,column),out.shape,strict=True):
            if axis in axes and axes[axis] != extent:raise ValueError('unfused-MoE shared logical axis extent mismatch')
            axes[axis] = extent
        if reduction is not None:
            extent = views[inputs[0][0]].shape[1]
            if reduction in axes and axes[reduction] != extent:raise ValueError('unfused-MoE K dimension mismatch')
            axes[reduction] = extent
        scope = AccessScope(f'{program_id}.{name}',('unfused_moe',name,kind),call_id,(),rank,stream,
            'compatibility-unfused-primitive-tiles')
        loop = (row,reduction,column) if reduction is not None else (row,column)
        reads = []
        for operand,(role,_) in enumerate(inputs):
            mapping = ((row,reduction) if operand == 0 else (reduction,column)) if reduction is not None else (row,column)
            dependencies = []
            if role in produced:
                writer = produced[role]
                if previous is None or previous.access_id != writer.access_id:
                    dependencies.append(TileDependency(writer.access_id,EventKind.WRITE_VISIBLE,
                        tuple(axis for axis in writer.iteration_axes if axis not in loop)))
                ready = ()
            else:
                ready = (readiness[role],)
            if previous is not None:
                boundary = TileDependency(previous.access_id,EventKind.WRITE_VISIBLE,previous.iteration_axes)
                if boundary not in dependencies:dependencies.append(boundary)
            identity = f's{index}.read{operand}'
            templates.append(TileAccessTemplate(identity,role,Operation.READ,loop,mapping,ready,
                tuple(dependencies),evidence,scope=scope))
            reads.append(TileDependency(identity,EventKind.READ_READY,
                (reduction,) if reduction is not None else (column,) if kind == 'softmax' else ()))
        write = TileAccessTemplate(f's{index}.write',output,Operation.WRITE,(row,column),(row,column),(),
            tuple(reads),evidence,scope=scope)
        templates.append(write);produced[output] = write;previous = write
    scope = AccessScope(program_id,('unfused_moe','decomposed'),call_id,(),rank,stream,
        'compatibility-unfused-primitive-tiles')
    return TileProgram(program_id,scope,tuple(TileAxis(name,extent,1) for name,extent in axes.items()),
        tuple(templates),visibility,evidence)


def unfused_moe_tile_demands(program):
    """Validate the sole tile source, then project retained compulsory interfaces."""
    program.__post_init__()
    if len(program.tile_programs) != 1:raise ValueError('unfused-MoE requires one primitive tile pipeline')
    family, = program.tile_programs
    views = unique_by(program.views,'view_id')
    tokens,hidden = views['router_input'].shape
    experts = views['router_weight'].shape[1]
    counts,intermediate = [],None
    for expert in range(experts):
        role = f'expert{expert}.input'
        if role not in views:
            counts.append(0);continue
        count,dimension = views[role].shape
        width = views[f'expert{expert}.activation'].shape[1]
        if dimension != hidden or (intermediate is not None and intermediate != width):
            raise ValueError('unfused-MoE expert geometry mismatch')
        intermediate = width;counts.append(count)
    if intermediate is None:raise ValueError('unfused-MoE requires actual routed work')
    stages = _primitive_layouts(tokens,experts,hidden,intermediate,counts)
    layouts = {name:shape for _,_,inputs,output in stages for name,shape in (*inputs,output)}
    if set(views) != set(layouts) or len({view.allocation_id for view in views.values()}) != len(views):
        raise ValueError('unfused-MoE primitive roles/independent storage mismatch')
    storage = views['router_input'].dtype,views['router_input'].element_bytes
    for name,shape in layouts.items():
        view = views[name]
        if (view.shape,view.dtype,view.element_bytes,view.strides_bytes) != (
                shape,*storage,(shape[1]*storage[1],storage[1])):
            raise ValueError('unfused-MoE primitive storage/geometry mismatch')
    templates = unique_by(family.accesses,'access_id')
    produced = {output[0] for _,_,_,output in stages}
    readiness = {}
    for template in templates.values():
        if template.operation is Operation.READ and template.view_id not in produced:
            if len(template.input_ready) != 1:raise ValueError('unfused-MoE input readiness missing')
            readiness[template.view_id] = template.input_ready[0]
    expected = _primitive_tiles(stages,views,readiness,program.program_id,family.scope.call_id,
        family.scope.rank,family.scope.stream,family.visibility_domain)
    if family != expected:raise ValueError('unfused-MoE tile geometry or input/producer readiness causality changed')
    size = lambda role:requested_bytes(views[role].region())
    primitive_bytes = sum(size(role) for _,_,inputs,output in stages for role,_ in (*inputs,output))
    expert_bytes = sum(size(role) for name,kind,inputs,output in stages if kind == 'mm' and name != 'router'
        for role,_ in (*inputs,output))
    return {'geometry':(tokens,experts,hidden,intermediate),'tokens_per_expert':tuple(counts),
        'expert_mm_bytes':expert_bytes,'primitive_requested_bytes':primitive_bytes}
