"""Source-level fused RoPE tasks, keeping runtime tensor selections unresolved."""
from math import prod

from src.kernel.kernel_spec import KernelKind,DTYPE_BYTES
from src.kernel.memory_access import (
    AccessScope,AddressLevel,Coverage,EventKind,EventRef,Granularity,LogicalSelectionView,
    MemoryAccessProgram,MemoryAllocation,Operation,Ordering,ProgramEvent,Semantics,TensorView,
    TileAxis,TileAccessTemplate,TileDependency,TileProgram,unique_by,
)


def has_v32_fused_q_source(kernel,hardware):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant=='dsv32_fused_q'
        and kernel.execution_support=='supported' and hardware.hardware_id==kernel.hardware_id
        and kernel.dtype=='bf16' and kernel.attributes['quantize_mqa'] is False)


def _row_extents(kernel):
    logical=kernel.shape['rows']
    if 'model_input_shape' not in kernel.attributes:return logical,logical
    physical=kernel.attributes['model_input_shape']
    if (set(physical)!=set(kernel.shape)-{'dtype_bytes'}
            or any(physical[name]!=kernel.shape[name] for name in physical if name!='rows')
            or physical['rows']*kernel.attributes['graph_logical_tokens']
                != logical*kernel.attributes['graph_model_input_tokens']):
        raise ValueError('V32 fused source requires its actual unchanged non-row geometry and declared replay row mapping')
    return logical,physical['rows']


def _layouts(kernel):
    sh=kernel.shape;_,t=_row_extents(kernel);q,i,d,r=(sh[name] for name in ('q_heads','index_heads','index_dim','rope_dim'))
    # Selection objects describe numeric runtime views, not invented dense
    # physical strides or the capacity of their underlying cache/tensor arena.
    inputs={'positions':((t,),'int64',False),'q_pe':((t,q,r),kernel.dtype,True),
        'q_cache':((t,r),kernel.dtype,True),'q_scale':((1,),'fp32',False)}
    outputs={'q_output':((t,q,r),kernel.dtype,True)}
    if kernel.attributes['has_indexer']:
        inputs.update(index_q=((t,i,d),kernel.dtype,True),index_cache=((t,r),kernel.dtype,True),
            index_weights=((t,i),kernel.dtype,True))
        outputs.update(index_output=((t,i,d),'fp8',True),weight_output=((t,i),'fp32',True))
    return inputs,outputs


def _strides(shape,width):
    return tuple(prod(shape[index+1:])*width for index in range(len(shape)))


def standalone_v32_fused_q_bindings(kernel,*,call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    inputs,outputs=_layouts(kernel);allocations=[];views=[];events=[];ready=[]
    for role,(shape,dtype,selection) in {**inputs,**outputs}.items():
        identity=f'{binding_id}.{role}';width=DTYPE_BYTES[dtype]
        allocation=MemoryAllocation(identity,'logical-local-device','v32-q-global',None if selection else prod(shape)*width,
            AddressLevel.OBJECT,None,None,identity+'.allocated',None)
        view=(LogicalSelectionView(role,identity,shape,dtype,width,identity+'.runtime_numeric_selection',None)
            if selection else TensorView(role,identity,0,shape,_strides(shape,width),dtype,width))
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role in inputs:
            kind=EventKind.LOGICAL_AVAILABLE if selection else EventKind.INITIALIZED
            event=ProgramEvent(identity+'.available',kind,identity,view.region(),'v32-q-visible',
                (EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),tuple(inputs),tuple(outputs),
        tuple(ready),0,'offline','v32-q-visible')


def _tiles(kernel,views,ready,scope,visibility):
    sh=kernel.shape;logical,t=_row_extents(kernel);q,i,r=(sh[name] for name in ('q_heads','index_heads','rope_dim'))
    evidence=('vLLM 0.30 deepseek_v32/common/kernels.py _fused_q_kernel BF16 MQA branch.',
        'MQA task handles two heads with an exact odd-head tail. Indexer task handles one full head in registers.',
        'Indexer partner load covers the complete input head, including the unrotated suffix; cosine/sine loads repeat for both channel parities.',
        'Q scale has a source load unused by BF16 output; compiler dead-code elimination is unbound. Skipped indexer and NoPE-pack branches have no logical requests.',
        f'Logical rows {logical}; declared execution rows {t}; replay padding contents are unknown.')
    def access(name,role,operation,loops,mapping,dependencies=()):
        return TileAccessTemplate(name,role,operation,loops,mapping,
            (ready[role],) if operation is Operation.READ else (),dependencies,evidence,
            unmapped=isinstance(views[role],LogicalSelectionView))
    qloops=('token','q_head');axes=[TileAxis('token',t,1),TileAxis('q_head',q,2)]
    accesses=[access('q_position','positions',Operation.READ,qloops,('token',)),
        access('q_cache_read','q_cache',Operation.READ,qloops,('token',None),
            (TileDependency('q_position',EventKind.READ_READY,()),)),
        access('q_scale_read','q_scale',Operation.READ,qloops,(None,)),
        access('q_read','q_pe',Operation.READ,qloops,('token','q_head',None)),
        access('q_write','q_output',Operation.WRITE,qloops,('token','q_head',None),
            (TileDependency('q_read',EventKind.READ_READY,()),TileDependency('q_cache_read',EventKind.READ_READY,())))]
    if kernel.attributes['has_indexer']:
        loops=('token','index_head');axes.extend((TileAxis('index_head',i,1),TileAxis('coefficient_parity',2,1)))
        accesses.extend((access('index_position','positions',Operation.READ,loops,('token',)),
            access('index_cache_read','index_cache',Operation.READ,(*loops,'coefficient_parity'),('token',None),
                (TileDependency('index_position',EventKind.READ_READY,()),)),
            access('index_read','index_q',Operation.READ,loops,('token','index_head',None)),
            access('index_partner','index_q',Operation.READ,loops,('token','index_head',None)),
            access('index_weight_read','index_weights',Operation.READ,loops,('token','index_head'))))
        complete=(TileDependency('index_read',EventKind.READ_READY,()),TileDependency('index_partner',EventKind.READ_READY,()),
            TileDependency('index_cache_read',EventKind.READ_READY,('coefficient_parity',)))
        accesses.extend((access('index_write','index_output',Operation.WRITE,loops,('token','index_head',None),complete),
            access('weight_write','weight_output',Operation.WRITE,loops,('token','index_head'),
                (*complete,TileDependency('index_weight_read',EventKind.READ_READY,())))))
    return TileProgram(scope.scope_id,scope,tuple(axes),tuple(accesses),visibility,evidence)


def build_v32_fused_q_source(kernel,hardware,bindings,granularity,*,program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_v32_fused_q_source(kernel,hardware):raise NotImplementedError('V32 fused Q source requires its BF16 MQA contract')
    if not isinstance(bindings,ElementwiseBindings):raise TypeError('V32 fused Q requires actual typed caller bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('V32 fused Q uses source head tasks')
    inputs,outputs=_layouts(kernel);layouts={**inputs,**outputs};views=unique_by(bindings.views,'view_id')
    allocations=unique_by(bindings.allocations,'allocation_id')
    if (set(views)!=set(layouts) or bindings.read_views!=tuple(inputs) or bindings.write_views!=tuple(outputs)
            or len(bindings.input_ready)!=len(inputs)):
        raise ValueError('V32 fused Q must preserve every actual active input and output role')
    for role,(shape,dtype,_) in layouts.items():
        view=views[role];width=DTYPE_BYTES[dtype]
        if (view.shape,view.dtype,view.element_bytes)!=(shape,dtype,width):
            raise ValueError('V32 fused Q typed geometry differs from its actual kernel')
        if isinstance(view,TensorView) and (view.strides_bytes[-1]!=width or any(left < count*right
                for left,count,right in zip(view.strides_bytes[:-1],view.shape[1:],view.strides_bytes[1:],strict=True))):
            raise ValueError('V32 fused Q requires nonoverlapping caller rows and contiguous inner elements')
    out_ids={views[role].allocation_id for role in outputs}
    if len(out_ids)!=len(outputs) or out_ids&{views[role].allocation_id for role in inputs}:
        raise ValueError('V32 fused Q materializes independent outputs; no in-place ABI is declared')
    if len({allocations[v.allocation_id].device_id for v in views.values()})!=1:
        raise ValueError('V32 fused Q operands must share their actual compute device')
    header=('dsv32_fused_q',f'indexer={kernel.attributes["has_indexer"]}',
        *(f'{name}={value}' for name,value in sorted(kernel.shape.items())))
    scope=AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-v32-fused-q')
    family=_tiles(kernel,views,dict(zip(inputs,bindings.input_ready,strict=True)),scope,bindings.visibility_domain)
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Runtime q/index strides and position-selected cache addresses/capacity remain unresolved unless supplied by the caller.',
            'Logical head tasks come from the source algorithm; they do not certify a synthetic launch or physical CTA issue.',
            'Indexer rotation/amax/quantization stays in registers; no global roped-index or scale scratch is invented.',
            'Logical requests include partner and coefficient replays; retained per-role footprint pricing is not observed DRAM traffic.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))


def v32_fused_q_source_work(program,*,expected_kernel,hardware,expected_bindings=None):
    from src.kernel.memory_access_builder import ElementwiseBindings
    from src.kernel.memory_path_solver import logical_requested_bytes
    program.__post_init__()
    if program.accesses or program.scopes or len(program.tile_programs)!=1:
        raise ValueError('V32 fused Q requires its actual compressed head source')
    family,=program.tile_programs;views=unique_by(program.views,'view_id');templates=unique_by(family.accesses,'access_id')
    inputs,outputs=_layouts(expected_kernel)
    if expected_bindings is None:
        names={'positions':'q_position','q_pe':'q_read','q_cache':'q_cache_read','q_scale':'q_scale_read',
            'index_q':'index_read','index_cache':'index_cache_read','index_weights':'index_weight_read'}
        ready=tuple(templates[names[role]].input_ready[0] for role in inputs)
        expected_bindings=ElementwiseBindings(program.binding_id,family.scope.call_id,program.allocations,program.views,program.events,
            tuple(inputs),tuple(outputs),ready,family.scope.rank,family.scope.stream,family.visibility_domain)
    expected=build_v32_fused_q_source(expected_kernel,hardware,expected_bindings,Granularity.TILE,program_id=program.program_id)
    if program!=expected:raise ValueError('V32 fused Q lost actual head work, replay, causality or caller binding')
    # The existing model prices one declared value footprint per role, while
    # source requests above retain every actual head-task replay. Selection
    # extents do not establish distinct physical cache capacity/addresses.
    footprints={role:prod(views[role].shape)*views[role].element_bytes for role in inputs}
    logical,execution=_row_extents(expected_kernel)
    priced_inputs={role:width if role=='q_scale' else width//execution*logical for role,width in footprints.items()}
    return {'read_bytes':logical_requested_bytes(program,Operation.READ),
        'write_bytes':logical_requested_bytes(program,Operation.WRITE),
        'legacy_read_bytes':sum(priced_inputs.values()),
        'legacy_write_bytes':sum(prod(views[role].shape)*views[role].element_bytes//execution*logical for role in outputs),
        'declared_input_role_bytes':footprints,'selected_cache_unique_bytes':None,
        'logical_rows':logical,'execution_rows':execution}


def has_v32_fused_norm_source(kernel,hardware):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant=='dsv32_fused_norm_rope'
        and kernel.execution_support=='supported' and hardware.hardware_id==kernel.hardware_id
        and kernel.dtype=='bf16' and kernel.attributes['mla_cache_dtype']=='bf16'
        and kernel.attributes['cache_slot_scope']=='all_rows_valid_slots_projection')


def _norm_layouts(kernel):
    sh=kernel.shape;_,t=_row_extents(kernel);q,k,r,d=(sh[n] for n in ('q_hidden','kv_hidden','rope_dim','index_dim'))
    inputs={'positions':((t,),'int64',False),'mla_slots':((t,),'int64',False),
        'q_input':((t,q),'bf16',True),'q_gamma':((q,),'bf16',False),
        'kv_input':((t,k),'bf16',True),'kv_gamma':((k,),'bf16',False),
        'kpe_input':((t,r),'bf16',True),'kpe_cache':((t,r),'bf16',True)}
    outputs={'q_output':((t,q),'bf16',True),'kv_output':((t,k),'bf16',True),
        'kpe_output':((t,r),'bf16',True),'mla_nope':((t,k),'bf16',True),'mla_rope':((t,r),'bf16',True)}
    if kernel.attributes['has_indexer']:
        inputs.update(index_k=((t,d),'bf16',True),index_gamma=((d,),'fp32',False),
            index_beta=((d,),'fp32',False),index_cache=((t,r),'bf16',True),index_slots=((t,),'int64',False))
        outputs.update(index_payload=((t,d),'fp8',True),index_scale=((t,),'fp32',True),
            topk_output=((t,sh['topk']),'int32',True))
    return inputs,outputs


def standalone_v32_fused_norm_bindings(kernel,*,call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    from src.kernel.memory_access import LogicalCoordinateBinding,LogicalCoordinateTerm
    inputs,outputs=_norm_layouts(kernel);allocations=[];views=[];events=[];ready=[]
    for role,(shape,dtype,selection) in {**inputs,**outputs}.items():
        # MLA fields share a logical [slot, NoPE+RoPE] parent. FP8 payload
        # and FP32 scales share their allocation, but not a typed parent:
        # the source packs scales after each block's payload, not per token.
        parent_role='mla_cache' if role in ('mla_nope','mla_rope') else (
            'indexer_cache' if role in ('index_payload','index_scale') else role)
        identity=f'{binding_id}.{parent_role}';width=DTYPE_BYTES[dtype]
        if not any(a.allocation_id==identity for a in allocations):
            allocation=MemoryAllocation(identity,'logical-local-device','v32-norm-global',
                None if selection else prod(shape)*width,AddressLevel.OBJECT,None,None,identity+'.allocated',None)
            allocations.append(allocation)
            events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        coordinates=()
        if parent_role=='mla_cache':
            coordinates=(LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),)),
                LogicalCoordinateBinding(1,(LogicalCoordinateTerm(1,1),),offset=kernel.shape['kv_hidden'] if role=='mla_rope' else 0))
        view=(LogicalSelectionView(role,identity,shape,dtype,width,identity+'.'+role+'.runtime_selection',
                'mla_cache' if parent_role=='mla_cache' else None,coordinates)
            if selection else TensorView(role,identity,0,shape,_strides(shape,width),dtype,width))
        views.append(view)
        if role in inputs:
            kind=EventKind.LOGICAL_AVAILABLE if selection else EventKind.INITIALIZED
            event=ProgramEvent(identity+'.available',kind,identity,view.region(),'v32-norm-visible',
                (EventRef(identity+'.allocated',EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    _,t=_row_extents(kernel)
    views.append(LogicalSelectionView('mla_cache',f'{binding_id}.mla_cache',(t,kernel.shape['kv_hidden']+kernel.shape['rope_dim']),
        'bf16',2,f'{binding_id}.mla_cache.selected_valid_slots',None))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),tuple(inputs),tuple(outputs),
        tuple(ready),0,'offline','v32-norm-visible')


def _norm_tiles(kernel,views,ready,scope,visibility):
    from src.kernel.memory_access import ConstantValue
    _,t=_row_extents(kernel)
    evidence=('vLLM 0.30 deepseek_v32/common/kernels.py _fused_norm_rope_kernel pid 0/1/2/3 and wrapper.',
        'One row per branch task; Q masked power-of-two lanes request only real channels; topk stores use 1024-element tails.',
        'Q and sentinel fill do not depend on slot ownership; BF16 materialized KV and KPE are independent allocations.',
        'Valid-slot projection only. Source-text slot gate/reload and partner/affine/coefficient replays are retained; native DCE/CSE is unbound.',
        'Index FP8 payload and FP32 scale are disjoint fields of one block-packed allocation; no per-token scale offset is invented.',
        'Row normalization, rotation partners, amax and quantization stay in registers; no global scratch is invented.')
    def dep(name,axes=()):return TileDependency(name,EventKind.READ_READY,axes)
    def access(name,role,operation,mapping,dependencies=(),*,loops=('token',),constant=None):
        return TileAccessTemplate(name,role,operation,loops,mapping,
            (ready[role],) if operation is Operation.READ else (),dependencies,evidence,
            unmapped=isinstance(views[role],LogicalSelectionView),constant_value=constant)
    read=Operation.READ;write=Operation.WRITE
    axes=[TileAxis('token',t,1)]
    accesses=[access('q_read','q_input',read,('token',None)),access('q_gamma_read','q_gamma',read,(None,)),
        access('q_write','q_output',write,('token',None),(dep('q_read'),dep('q_gamma_read'))),
        access('kv_gate','mla_slots',read,('token',)),
        # With both materialized pointers present the pid1 gate predicate
        # cannot skip the branch. It does not causally gate these outputs.
        access('kv_read','kv_input',read,('token',None)),access('kv_gamma_read','kv_gamma',read,(None,)),
        access('kv_position','positions',read,('token',)),
        access('kpe_coefficients','kpe_cache',read,('token',None),(dep('kv_position'),)),
        access('kpe_read','kpe_input',read,('token',None)),
        access('kv_write','kv_output',write,('token',None),(dep('kv_read'),dep('kv_gamma_read'))),
        access('kpe_write','kpe_output',write,('token',None),(dep('kpe_read'),dep('kpe_coefficients'))),
        access('mla_address_slot','mla_slots',read,('token',)),
        access('mla_nope_write','mla_nope',write,('token',None),(dep('kv_read'),dep('kv_gamma_read'),dep('mla_address_slot'))),
        access('mla_rope_write','mla_rope',write,('token',None),(dep('kpe_read'),dep('kpe_coefficients'),dep('mla_address_slot'))),
        # pid0 reaches this source-text load even without HAS_INDEXER.
        # An optimizing compiler may remove that unused branch's gate.
        access('index_gate','mla_slots',read,('token',))]
    if kernel.attributes['has_indexer']:
        axes.extend((TileAxis('coefficient_parity',2,1),TileAxis('topk_entry',kernel.shape['topk'],1024)))
        for name,role,mapping in (('index_read','index_k',('token',None)),('index_gamma_read','index_gamma',(None,)),
                ('index_beta_read','index_beta',(None,)),('index_partner','index_k',('token',None)),
                ('index_gamma_partner','index_gamma',(None,)),('index_beta_partner','index_beta',(None,)),
                ('index_position','positions',('token',))):
            accesses.append(access(name,role,read,mapping,(dep('index_gate'),)))
        accesses.append(access('index_coefficients','index_cache',read,('token',None),(dep('index_position'),),
            loops=('token','coefficient_parity')))
        accesses.append(access('index_address_slot','index_slots',read,('token',),(dep('index_gate'),)))
        complete=tuple(dep(n) for n in ('index_read','index_gamma_read','index_beta_read','index_partner',
            'index_gamma_partner','index_beta_partner','index_address_slot'))+(dep('index_coefficients',('coefficient_parity',)),)
        accesses.extend((access('index_payload_write','index_payload',write,('token',None),complete),
            access('index_scale_write','index_scale',write,('token',),complete),
            access('topk_fill','topk_output',write,('token','topk_entry'),loops=('token','topk_entry'),constant=ConstantValue.NEGATIVE_ONE)))
    return TileProgram(scope.scope_id,scope,tuple(axes),tuple(accesses),visibility,evidence)


def build_v32_fused_norm_source(kernel,hardware,bindings,granularity,*,program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_v32_fused_norm_source(kernel,hardware):raise NotImplementedError('V32 norm source requires its BF16 valid-slot contract')
    if not isinstance(bindings,ElementwiseBindings):raise TypeError('V32 norm requires actual typed caller bindings')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('V32 norm uses source row tasks')
    inputs,outputs=_norm_layouts(kernel);views=unique_by(bindings.views,'view_id');allocations=unique_by(bindings.allocations,'allocation_id')
    if (set(views)!=set(inputs)|set(outputs)|{'mla_cache'} or bindings.read_views!=tuple(inputs)
            or bindings.write_views!=tuple(outputs) or len(bindings.input_ready)!=len(inputs)):
        raise ValueError('V32 norm requires every active source role and its shared MLA parent')
    for role,(shape,dtype,_) in {**inputs,**outputs}.items():
        view=views[role]
        if (view.shape,view.dtype,view.element_bytes)!=(shape,dtype,DTYPE_BYTES[dtype]):
            raise ValueError('V32 norm typed geometry differs from its actual kernel')
        if isinstance(view,TensorView) and (view.strides_bytes[-1]!=view.element_bytes or any(left < count*right
                for left,count,right in zip(view.strides_bytes[:-1],view.shape[1:],view.strides_bytes[1:],strict=True))):
            raise ValueError('V32 norm requires nonoverlapping caller rows and contiguous inner elements')
    _,t=_row_extents(kernel);parent=views['mla_cache']
    if (not isinstance(parent,LogicalSelectionView) or parent.parent_view_id is not None
            or (parent.shape,parent.dtype,parent.element_bytes)!=( (t,kernel.shape['kv_hidden']+kernel.shape['rope_dim']),'bf16',2)):
        raise ValueError('V32 norm must retain the actual unresolved selected MLA parent')
    from src.kernel.memory_access import LogicalCoordinateBinding,LogicalCoordinateTerm
    for role,offset in (('mla_nope',0),('mla_rope',kernel.shape['kv_hidden'])):
        view=views[role]
        expected=(LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),)),
            LogicalCoordinateBinding(1,(LogicalCoordinateTerm(1,1),),offset=offset))
        if (not isinstance(view,LogicalSelectionView) or view.allocation_id!=parent.allocation_id
                or view.parent_view_id!='mla_cache' or view.coordinate_map!=expected):
            raise ValueError('V32 norm must retain shared NoPE/RoPE cache fields')
    materialized=[views[r].allocation_id for r in ('q_output','kv_output','kpe_output')]
    cache_ids={parent.allocation_id}
    if kernel.attributes['has_indexer']:
        payload,scale=(views[r] for r in ('index_payload','index_scale'))
        if (not isinstance(payload,LogicalSelectionView) or not isinstance(scale,LogicalSelectionView)
                or payload.allocation_id!=scale.allocation_id or payload.parent_view_id is not None or scale.parent_view_id is not None
                or payload.selection_id==scale.selection_id):
            raise ValueError('V32 norm requires separate typed fields of the same indexer cache allocation')
        cache_ids.add(payload.allocation_id);materialized.append(views['topk_output'].allocation_id)
        if len(cache_ids)!=2:raise ValueError('MLA and indexer caches are distinct allocations')
    if (len(set(materialized))!=len(materialized) or set(materialized)&cache_ids
            or (set(materialized)|cache_ids)&{views[r].allocation_id for r in inputs}):
        raise ValueError('V32 norm materialized outputs, caches and inputs must remain independent')
    if len({allocations[v.allocation_id].device_id for v in views.values()})!=1:raise ValueError('V32 norm requires one actual compute device')
    header=('dsv32_fused_norm_rope',f'indexer={kernel.attributes["has_indexer"]}',
        *(f'{name}={value}' for name,value in sorted(kernel.shape.items())))
    scope=AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,'logical-v32-fused-norm')
    family=_norm_tiles(kernel,views,dict(zip(inputs,bindings.input_ready,strict=True)),scope,bindings.visibility_domain)
    return MemoryAccessProgram('midlayer.memory-access.v1',program_id,hardware.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'BF16 all-valid-slot projection; actual slot values/ownership, runtime strides and selected cache capacity remain unresolved.',
            'Logical source tasks do not bind compiler/native issue; repeated source loads can be optimized.',
            'Separate materialized KV/KPE and shared cache field identities are source facts; scratch remains in registers.',
            'Index payload/scales are block-packed fields of one allocation with unknown block geometry; no physical offsets inferred.',
            'Existing per-role footprint pricing excludes source replays and is not observed DRAM traffic.')),
        bindings.allocations,bindings.views,(),bindings.events,(),(family,))


def v32_fused_norm_source_work(program,*,expected_kernel,hardware,expected_bindings=None):
    from src.kernel.memory_access_builder import ElementwiseBindings
    from src.kernel.memory_path_solver import logical_requested_bytes
    program.__post_init__()
    if program.accesses or program.scopes or len(program.tile_programs)!=1:raise ValueError('V32 norm requires compressed source row branches')
    family,=program.tile_programs;views=unique_by(program.views,'view_id');templates=unique_by(family.accesses,'access_id')
    inputs,outputs=_norm_layouts(expected_kernel)
    if expected_bindings is None:
        names={'positions':'kv_position','mla_slots':'mla_address_slot','q_input':'q_read','q_gamma':'q_gamma_read',
            'kv_input':'kv_read','kv_gamma':'kv_gamma_read','kpe_input':'kpe_read','kpe_cache':'kpe_coefficients',
            'index_k':'index_read','index_gamma':'index_gamma_read','index_beta':'index_beta_read',
            'index_cache':'index_coefficients','index_slots':'index_address_slot'}
        expected_bindings=ElementwiseBindings(program.binding_id,family.scope.call_id,program.allocations,program.views,program.events,
            tuple(inputs),tuple(outputs),tuple(templates[names[r]].input_ready[0] for r in inputs),family.scope.rank,
            family.scope.stream,family.visibility_domain)
    expected=build_v32_fused_norm_source(expected_kernel,hardware,expected_bindings,Granularity.TILE,program_id=program.program_id)
    if program!=expected:raise ValueError('V32 norm lost source branch work, replay, causality or caller binding')
    footprints={role:prod(views[role].shape)*views[role].element_bytes for role in inputs}
    logical,execution=_row_extents(expected_kernel)
    parameters={'q_gamma','kv_gamma','index_gamma','index_beta'}
    return {'read_bytes':logical_requested_bytes(program,Operation.READ),'write_bytes':logical_requested_bytes(program,Operation.WRITE),
        'legacy_read_bytes':sum(width if role in parameters else width//execution*logical for role,width in footprints.items()),
        'legacy_write_bytes':sum(prod(views[role].shape)*views[role].element_bytes//execution*logical for role in outputs),
        'declared_input_role_bytes':footprints,'selected_cache_unique_bytes':None,'logical_rows':logical,'execution_rows':execution}
