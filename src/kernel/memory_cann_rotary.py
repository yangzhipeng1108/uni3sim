"""CANN public RoPE: two GatherV3 launches and logical ATB BF16 slice dataflow.

The existing source-bound launch contract owns keys22/24 and slice geometry.
Private arrays describe logical versions, not simultaneous native UB residency.
The separately retained installed CANN source is an algorithm reference, not
an assertion of byte identity with the existing ATB production-source pin.
"""

from src.kernel.prediction_cache import runtime_lru_cache
from dataclasses import replace
from math import prod

from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    AccessScope,AddressLevel,ConstantValue,Coverage,EventKind,EventRef,Granularity,
    LogicalCoordinateBinding,LogicalCoordinateTerm,LogicalSelectionView,MemoryAccessProgram,
    MemoryAllocation,Operation,Ordering,ProgramEvent,Semantics,TensorView,TileAccessTemplate,
    TileAxis,TileDependency,TileProgram,Unmapped,unique_by,_tile_template_requested_bytes,
    _replace_source_identity,
)


def has_cann_rotary_source(kernel,hardware):
    return (kernel.kind is KernelKind.VECTOR and kernel.variant in ('rotary','rope')
        and kernel.hardware_id=='ascend_910b' and hardware.kernel_memory_model=='npu_tile'
        and kernel.execution_support=='supported' and kernel.dtype=='bf16'
        and 'source_bound_atb_bf16_rope_dag' in kernel.launch.parameters)


def _validate_geometry(g):
    t,q,k,d,slice_elements,workers,gworkers,key=g
    expected_key=24 if t>=64 else 22
    if (min(t,q,d,slice_elements,workers,gworkers)<=0 or k<0 or d%32
            or workers>t or gworkers>t or key!=expected_key
            or slice_elements!=(2048 if key==24 else 4096)//d*d
            or (q*d)%64 or (k*d)%64):
        raise ValueError('CANN rotary geometry requires its actual aligned full-dimension source domain')


def _geometry(kernel):
    p=kernel.launch.parameters;s=kernel.shape
    if (kernel.hardware_id!='ascend_910b' or kernel.kind is not KernelKind.VECTOR
            or kernel.variant not in ('rotary','rope') or kernel.dtype!='bf16'
            or p['source_bound_atb_bf16_rope_dag']!=1.0
            or p['atb_rope_rotary_coefficient']!=2.0 or p['atb_rope_cos_format']!=0.0
            or p['source_cann_rope_index_select_count']!=2.0
            or p['source_cann_rope_gather_v3_program']!=10330.0
            or p['cann_rope_gather_v3_tiling_payload_bytes']!=192.0):
        raise ValueError('CANN rotary requires its source-bound BF16 full-dimension NeoX ABI')
    t,q,k,d=(s[name] for name in ('seq_len','num_query_heads','num_kv_heads','head_dim'))
    if kernel.variant=='rope' and (s['rotary_dim']!=d or not kernel.attributes['is_neox']):
        raise ValueError('ATB rotary source requires full-dimension NeoX pairing')
    if kernel.variant=='rope' and not kernel.attributes['key_present']:k=0
    key=24 if t>=64 else 22;slice_elements=(2048 if key==24 else 4096)//d*d
    if (d%32 or not slice_elements or s['dtype_bytes']!=2 or s['element_count']!=t*(q+k)*d
            or p['cann_tiling_key']!=key or p['atb_rope_slice_elements']!=slice_elements
            or p['cann_rope_gather_v3_selected_row_bytes']!=t*d*2
            or kernel.launch.grid[1:]!=(1,1) or kernel.launch.block!=(1,1,1)):
        raise ValueError('CANN rotary launch slices or public tensor shape disagree')
    workers=kernel.launch.grid_size;gworkers=int(p['cann_rope_gather_v3_real_core_num'])
    if not 0<workers<=t or not 0<gworkers<=t or gworkers!=p['cann_rope_gather_v3_real_core_num']:
        raise ValueError('CANN rotary requires actual active ATB and Gather workers')
    result=t,q,k,d,slice_elements,workers,gworkers,key
    _validate_geometry(result);return result


def _layouts(g):
    t,q,k,d,*_=g
    return (('query',(t,q,d),'bf16',2),*((('key',(t,k,d),'bf16',2),) if k else ()),
        ('positions',(t,),'int32',4),('cos_table',(t,d),'bf16',2),('sin_table',(t,d),'bf16',2),
        ('rope_tiling',(48,),'uint8',1),('cos_tiling',(192,),'uint8',1),('sin_tiling',(192,),'uint8',1))


def _strides(shape,width):
    values=[]
    for size in reversed(shape):values.append(width);width*=size
    return tuple(reversed(values))


def _validate_parents(g,parents):
    if set(parents)!={role for role,*_ in _layouts(g)}:raise ValueError('CANN rotary requires every actual public operand')
    for role,shape,dtype,width in _layouts(g):
        view=parents[role]
        if (not isinstance(view,TensorView) or (view.shape,view.dtype,view.element_bytes)!=(shape,dtype,width)):
            raise ValueError('CANN rotary actual typed shape differs')
        if role in ('query','key'):
            if view.strides_bytes[1:]!=(g[3]*2,2) or view.strides_bytes[0]<shape[1]*g[3]*2:
                raise ValueError('CANN rotary Q/K head or token storage stride differs')
        elif role not in ('cos_table','sin_table') and view.strides_bytes!=_strides(shape,width):
            raise ValueError('CANN rotary position/descriptor storage stride differs')
    if g[2] and parents['query'].allocation_id==parents['key'].allocation_id:
        query,key=parents['query'],parents['key'];stride=query.strides_bytes[0]
        starts=(query.offset_bytes%stride,key.offset_bytes%stride)
        ends=tuple(start+view.shape[1]*g[3]*2 for start,view in zip(starts,(query,key)))
        if key.strides_bytes[0]!=stride or max(ends)>stride or max(starts)<min(ends):
            raise ValueError('CANN rotary in-place Q/K regions overlap or use different packed rows')
    tensor_ids={parents[role].allocation_id for role in (('query','key') if g[2] else ('query',))}
    if any(parents[role].allocation_id in tensor_ids for role in ('positions','cos_table','sin_table','rope_tiling','cos_tiling','sin_tiling')):
        raise ValueError('CANN rotary read-only controls/cache cannot alias its Q/K output')


def standalone_cann_rotary_bindings(kernel, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    g=_geometry(kernel);allocations=[];views=[];events=[];ready=[]
    for role,shape,dtype,width in _layouts(g):
        identity=f'{binding_id}.{role}';selected=role in ('cos_table','sin_table')
        allocation=MemoryAllocation(identity,'logical-local-device','logical-position-selected-cache' if selected else 'logical-storage',
            None if selected else prod(shape)*width,AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view=TensorView(role,identity,0,shape,_strides(shape,width),dtype,width)
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        event=ProgramEvent(f'{identity}.available',EventKind.LOGICAL_AVAILABLE if selected else EventKind.INITIALIZED,
            identity,Unmapped(shape,dtype,width,f'{binding_id}.{role}.position_selected') if selected else view.region(),
            'cann-rotary-visible',(EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
        events.append(event);ready.append(EventRef(event.event_id,event.kind))
    roles=tuple(role for role,*_ in _layouts(g));outputs=('query','key') if g[2] else ('query',)
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),roles,outputs,
        tuple(ready),0,'offline','cann-rotary-visible')


def _recipe(g,parents,ready,scope,visibility):
    return _bound_recipe(g,tuple(sorted(parents.items())),tuple(sorted(ready.items())),scope,visibility)


@runtime_lru_cache(maxsize=8)
def _bound_recipe(g,parent_items,ready_items,scope,visibility):
    parents=dict(parent_items);ready=dict(ready_items)
    # Private value dataflow depends on geometry, not caller names. Share that
    # immutable structure, then bind every public parent, readiness and scope.
    # Consumers still compare the complete actual program with this recipe.
    roles=tuple(role for role,*_ in _layouts(g))
    canonical_parents=tuple((role,replace(parents[role],allocation_id=f'recipe.parent.{role}')) for role in roles)
    canonical_ready=tuple((role,replace(ready[role],node_id=f'recipe.ready.{role}')) for role in roles)
    canonical_scope=replace(scope,scope_id='cann.rotary.recipe',call_id='recipe.call',rank=0,stream='recipe.stream')
    allocations,views,events,families=_cached_recipe(g,canonical_parents,canonical_ready,
        canonical_scope,'recipe.visibility')
    def actual_identity(identity):
        return scope.scope_id+identity[len(canonical_scope.scope_id):]
    allocations=tuple(_replace_source_identity(allocation,allocation_id=actual_identity(allocation.allocation_id),
        allocated_event=actual_identity(allocation.allocated_event)) for allocation in allocations)
    rebound_views=tuple(_replace_source_identity(view,
        allocation_id=parents[view.parent_view_id].allocation_id if view.parent_view_id in parents else actual_identity(view.allocation_id),
        selection_id=actual_identity(view.selection_id)) if isinstance(view,LogicalSelectionView)
        else _replace_source_identity(view,allocation_id=actual_identity(view.allocation_id)) for view in views)
    events=tuple(_replace_source_identity(event,event_id=actual_identity(event.event_id),
        allocation_id=actual_identity(event.allocation_id)) for event in events)
    actual_ready={f'recipe.ready.{role}':ready[role] for role in roles}
    family,=families
    accesses=tuple(_replace_source_identity(access,input_ready=tuple(actual_ready[ref.node_id] for ref in access.input_ready))
        if access.input_ready else access for access in family.accesses)
    return allocations,rebound_views,events,(_replace_source_identity(family,tile_program_id=scope.scope_id,scope=scope,
        visibility_domain=visibility,accesses=accesses),)


@runtime_lru_cache(maxsize=8)
def _cached_recipe(g,parent_items,ready_items,scope,visibility):
    parents=dict(parent_items);ready=dict(ready_items)
    t,q,k,d,slice_elements,workers,gworkers,key=g
    allocations=[];views=[];events=[];axes=[];accesses=[];private_allocations=set()
    evidence=('Source-bound public RoPE launch: two GatherV3/10330 direct-row selections precede the ATB BF16 key22/24 program.',
        'ATB published slice geometry, cos/sin broadcast and cast, two half-rotation copies, three FP32 multiplies, one add and BF16 output conversion.',
        'Logical private versions retain typed value dataflow; native UB pool addresses, instruction issue, bank layout and simultaneous residency are not inferred.')
    def dep(name,kind=EventKind.READ_READY,reduction=()):return TileDependency(name,kind,reduction)
    def axis(name,count,block=1):axes.append(TileAxis(name,count,block));return name
    def buffer(name,shape,dtype='bf16',width=2,domain='logical-ub-versioned'):
        identity=f'{scope.scope_id}.{name}';allocation=MemoryAllocation(identity,'logical-local-device',domain,None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view=TensorView(name,identity,0,shape,_strides(shape,width),dtype,width)
        allocations.append(allocation);private_allocations.add(identity);views.append(view);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        return view
    def sub(name,parent,shape,offset,strides):
        view=TensorView(name,parent.allocation_id,parent.offset_bytes+offset,shape,strides,parent.dtype,parent.element_bytes)
        views.append(view);return view
    def selected(name,parent,shape,mapping):
        view=LogicalSelectionView(name,parent.allocation_id,shape,parent.dtype,parent.element_bytes,
            f'{scope.scope_id}.{name}',parent.view_id,tuple(mapping));views.append(view);return view
    def coord(axis_id,terms=(),offset=0):
        return LogicalCoordinateBinding(axis_id,tuple(LogicalCoordinateTerm(a,b) for a,b in terms),offset=offset)
    def add(name,view,op,loops,mapping,deps=(),input_ready=(),value=None):
        accesses.append(TileAccessTemplate(name,view.view_id,op,loops,mapping,input_ready,deps,evidence,
            unmapped=isinstance(view,LogicalSelectionView) or view.allocation_id in private_allocations,constant_value=value))
    gathered={role:buffer(f'gathered.{role}',(t,d),domain='logical-gm-gather-output') for role in ('cos','sin')}
    gather_outputs=[]
    gather_base,gather_remainder=divmod(t,gworkers)
    gather_groups=tuple(group for group in ((0,gather_base+1,gather_remainder,0),
        (gather_remainder,gather_base,gworkers-gather_remainder,gather_remainder*(gather_base+1))) if group[2])
    for role in ('cos','sin'):
        own_outputs=[]
        for worker,count,worker_count,offset in gather_groups:
            prefix=f'gather.{role}.w{worker}';rows=worker_count*count;row=axis(f'{prefix}.row',rows,count)
            loops=(row,)
            add(f'{prefix}.tiling',parents[f'{role}_tiling'],Operation.READ,loops,(None,),tuple(gather_outputs),(ready[f'{role}_tiling'],))
            position=selected(f'{prefix}.positions',parents['positions'],(rows,),
                (coord(0,((0,1),),offset),))
            add(f'{prefix}.positions',position,Operation.READ,loops,loops, (dep(f'{prefix}.tiling'),),(ready['positions'],))
            index=buffer(f'{prefix}.index_ub',(rows,),'int32',4)
            add(f'{prefix}.index_write',index,Operation.WRITE,loops,loops,(dep(f'{prefix}.positions'),))
            add(f'{prefix}.index_read',index,Operation.READ,loops,loops,(dep(f'{prefix}.index_write',EventKind.WRITE_VISIBLE),))
            table=selected(f'{prefix}.table',parents[f'{role}_table'],(rows,d),
                (coord(0,((0,1),),offset),coord(1,((1,1),))))
            add(f'{prefix}.table',table,Operation.READ,loops,(*loops,None),
                (dep(f'{prefix}.index_read'),),(ready[f'{role}_table'],))
            data=buffer(f'{prefix}.data_ub',(rows,d))
            add(f'{prefix}.data_write',data,Operation.WRITE,loops,(*loops,None),(dep(f'{prefix}.table'),))
            add(f'{prefix}.data_read',data,Operation.READ,loops,(*loops,None),(dep(f'{prefix}.data_write',EventKind.WRITE_VISIBLE),))
            output=selected(f'{prefix}.output',gathered[role],(rows,d),
                (coord(0,((0,1),),offset),coord(1,((1,1),))))
            add(f'{prefix}.output',output,Operation.WRITE,loops,(*loops,None),(dep(f'{prefix}.data_read'),))
            own_outputs.append(dep(f'{prefix}.output',EventKind.WRITE_VISIBLE,loops))
        gather_outputs.extend(own_outputs)
    cache_heads=min(q*d,slice_elements)//d;base,remainder=divmod(t,workers)
    worker_groups=tuple(group for group in ((0,base+1,remainder,0),
        (remainder,base,workers-remainder,remainder*(base+1))) if group[2])
    def vectorize(start_view,start_allocation,start_access,worker_axis,worker_count,owned):
        local={allocation.allocation_id for allocation in allocations[start_allocation:]}
        pitches={view.allocation_id:prod(view.shape)*view.element_bytes for view in views[start_view:]
            if isinstance(view,TensorView) and view.view_id==view.allocation_id.removeprefix(f'{scope.scope_id}.')}
        local_views=set()
        for index in range(start_view,len(views)):
            view=views[index];local_views.add(view.view_id)
            if isinstance(view,TensorView):
                if view.allocation_id not in local:raise ValueError('Worker-local rotary tensor lost its allocation')
                views[index]=replace(view,shape=(worker_count,*view.shape),strides_bytes=(pitches[view.allocation_id],*view.strides_bytes))
            else:
                mapping=tuple(replace(binding,terms=(*((LogicalCoordinateTerm(0,owned),) if binding.axis==0 else ()),
                    *(replace(term,axis=term.axis+1) for term in binding.terms))) for binding in view.coordinate_map)
                views[index]=replace(view,shape=(worker_count,*view.shape),coordinate_map=mapping)
        for index in range(start_access,len(accesses)):
            access=accesses[index]
            accesses[index]=replace(access,iteration_axes=(worker_axis,*access.iteration_axes),
                view_axes=(worker_axis,*access.view_axes) if access.view_id in local_views else access.view_axes)
    for worker,owned,worker_count,worker_offset in worker_groups:
        slots=2 if key==24 and owned>1 else 1;group_prefix=f'atb.w{worker}'
        worker_axis=axis(f'{group_prefix}.worker',worker_count)
        add(f'{group_prefix}.tiling',parents['rope_tiling'],Operation.READ,(worker_axis,),(None,),tuple(gather_outputs),(ready['rope_tiling'],))
        for slot in range(slots):
            start_view,start_allocation,start_access=len(views),len(allocations),len(accesses)
            count=-(-(owned-slot)//slots);prefix=f'atb.w{worker}.slot{slot}';token=axis(f'{prefix}.token',count)
            masks=[];cache_fp32={}
            mask=buffer(f'{prefix}.mask',(cache_heads,d),'fp32',4)
            first=sub(f'{prefix}.mask_first',mask,(1,d),0,mask.strides_bytes)
            add(f'{prefix}.mask_first',first,Operation.WRITE,(),(None,None),(dep(f'{group_prefix}.tiling'),))
            masks.append(dep(f'{prefix}.mask_first',EventKind.WRITE_VISIBLE))
            if cache_heads>1:
                copies=axis(f'{prefix}.mask_copy',cache_heads-1)
                add(f'{prefix}.mask_copy',first,Operation.READ,(copies,),(None,None),tuple(masks))
                tail=sub(f'{prefix}.mask_tail',mask,(cache_heads-1,d),d*4,mask.strides_bytes)
                add(f'{prefix}.mask_tail',tail,Operation.WRITE,(),(None,None),(dep(f'{prefix}.mask_copy',reduction=(copies,)),))
                masks.append(dep(f'{prefix}.mask_tail',EventKind.WRITE_VISIBLE))
            for role in ('cos','sin'):
                inp=selected(f'{prefix}.{role}',gathered[role],(count,d),
                    (coord(0,((0,slots),),worker_offset+slot),coord(1,((1,1),))))
                add(f'{prefix}.{role}',inp,Operation.READ,(token,),(token,None),(*gather_outputs,dep(f'{group_prefix}.tiling')))
                ub=buffer(f'{prefix}.{role}_ub',(count,cache_heads,d));pitch=ub.strides_bytes
                first=sub(f'{prefix}.{role}_first',ub,(count,1,d),0,pitch)
                add(f'{prefix}.{role}_first',first,Operation.WRITE,(token,),(token,None,None),(dep(f'{prefix}.{role}'),))
                initialized=[dep(f'{prefix}.{role}_first',EventKind.WRITE_VISIBLE)]
                if cache_heads>1:
                    copies=axis(f'{prefix}.{role}_copy',cache_heads-1)
                    add(f'{prefix}.{role}_copy',first,Operation.READ,(token,copies),(token,None,None),tuple(initialized))
                    tail=sub(f'{prefix}.{role}_tail',ub,(count,cache_heads-1,d),d*2,pitch)
                    add(f'{prefix}.{role}_tail',tail,Operation.WRITE,(token,),(token,None,None),
                        (dep(f'{prefix}.{role}_copy',reduction=(copies,)),));initialized.append(dep(f'{prefix}.{role}_tail',EventKind.WRITE_VISIBLE))
                add(f'{prefix}.{role}_cast_read',ub,Operation.READ,(token,),(token,None,None),tuple(initialized))
                fp32=buffer(f'{prefix}.{role}_fp32',(count,cache_heads,d),'fp32',4);cache_fp32[role]=fp32
                add(f'{prefix}.{role}_cast_write',fp32,Operation.WRITE,(token,),(token,None,None),(dep(f'{prefix}.{role}_cast_read'),))
            for role,heads in (('query',q),('key',k)):
                if not heads:continue
                full,tail=divmod(heads,cache_heads)
                head_groups=tuple((label,size,slices,start) for label,size,slices,start in
                    (('full',cache_heads,full,0),('tail',tail,1,full*cache_heads)) if size and slices)
                for label,size,slices,start in head_groups:
                    base=f'{prefix}.{role}.{label}';part=axis(f'{base}.slice',slices);loops=(token,part);mapping=(token,part,None,None)
                    original=[];gm=[]
                    for half in range(2):
                        name=f'{base}.h{half}';shape=(count,slices,size,d//2)
                        view=selected(f'{name}.global',parents[role],shape,
                            (coord(0,((0,slots),),worker_offset+slot),coord(1,((1,cache_heads),(2,1)),start),coord(2,((3,1),),half*d//2)))
                        add(f'{name}.global_read',view,Operation.READ,loops,mapping,(dep(f'{group_prefix}.tiling'),),(ready[role],))
                        ub=buffer(f'{name}.original',shape);original.append(ub);gm.append(view)
                        add(f'{name}.original_write',ub,Operation.WRITE,loops,mapping,(dep(f'{name}.global_read'),))
                    for half in range(2):
                        name=f'{base}.h{half}';other=f'{base}.h{1-half}';shape=original[half].shape
                        add(f'{name}.copy_read',original[1-half],Operation.READ,loops,mapping,(dep(f'{other}.original_write',EventKind.WRITE_VISIBLE),))
                        rotated=buffer(f'{name}.rotated',shape)
                        add(f'{name}.copy_write',rotated,Operation.WRITE,loops,mapping,(dep(f'{name}.copy_read'),))
                        states={}
                        for tag,source,writer in (('ori',original[half],f'{name}.original_write'),('rot',rotated,f'{name}.copy_write')):
                            add(f'{name}.{tag}_cast_read',source,Operation.READ,loops,mapping,(dep(writer,EventKind.WRITE_VISIBLE),))
                            state=buffer(f'{name}.{tag}_fp32',shape,'fp32',4);states[tag]=state
                            add(f'{name}.{tag}_cast_write',state,Operation.WRITE,loops,mapping,(dep(f'{name}.{tag}_cast_read'),))
                        mask_half=sub(f'{name}.mask',mask,(size,d//2),half*d//2*4,mask.strides_bytes)
                        add(f'{name}.mask',mask_half,Operation.READ,loops,(None,None),tuple(masks))
                        for tag,operand,writer,extra in (
                                ('ori_cos',states['ori'],f'{name}.ori_cast_write','cos'),
                                ('rot_mask',states['rot'],f'{name}.rot_cast_write','mask'),
                                ('rot_sin',states['rot'],f'{name}.rot_mask_write','sin')):
                            add(f'{name}.{tag}_read',operand,Operation.READ,loops,mapping,(dep(writer,EventKind.WRITE_VISIBLE),))
                            if extra=='mask':extra_ref=dep(f'{name}.mask')
                            else:
                                cache=cache_fp32[extra];selected_cache=sub(f'{name}.{extra}',cache,(count,size,d//2),half*d//2*4,cache.strides_bytes)
                                add(f'{name}.{extra}',selected_cache,Operation.READ,loops,(token,None,None),
                                    (dep(f'{prefix}.{extra}_cast_write',EventKind.WRITE_VISIBLE),));extra_ref=dep(f'{name}.{extra}')
                            add(f'{name}.{tag}_write',operand,Operation.WRITE,loops,mapping,(dep(f'{name}.{tag}_read'),extra_ref))
                        for tag,writer in (('ori',f'{name}.ori_cos_write'),('rot',f'{name}.rot_sin_write')):
                            add(f'{name}.{tag}_final',states[tag],Operation.READ,loops,mapping,(dep(writer,EventKind.WRITE_VISIBLE),))
                        summed=buffer(f'{name}.sum',shape,'fp32',4)
                        add(f'{name}.sum_write',summed,Operation.WRITE,loops,mapping,(dep(f'{name}.ori_final'),dep(f'{name}.rot_final')))
                        add(f'{name}.sum_read',summed,Operation.READ,loops,mapping,(dep(f'{name}.sum_write',EventKind.WRITE_VISIBLE),))
                        output=buffer(f'{name}.output',shape)
                        add(f'{name}.output_cast_write',output,Operation.WRITE,loops,mapping,(dep(f'{name}.sum_read'),))
                        add(f'{name}.output_read',output,Operation.READ,loops,mapping,(dep(f'{name}.output_cast_write',EventKind.WRITE_VISIBLE),))
                        add(f'{name}.global_write',gm[half],Operation.WRITE,loops,mapping,(dep(f'{name}.output_read'),))
            vectorize(start_view,start_allocation,start_access,worker_axis,worker_count,owned)
    family=TileProgram(scope.scope_id,scope,tuple(axes),tuple(accesses),visibility,evidence)
    return tuple(allocations),tuple(views),tuple(events),(family,)


def build_cann_rotary_source(kernel,hardware,bindings,granularity, *, program_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    if not has_cann_rotary_source(kernel,hardware) or not isinstance(bindings,ElementwiseBindings):
        raise ValueError('CANN rotary requires its actual source-bound kernel and typed operands')
    if granularity not in (Granularity.OP,Granularity.KERNEL,Granularity.TILE):raise ValueError('Unknown CANN rotary granularity')
    g=_geometry(kernel);parents=unique_by(bindings.views,'view_id');roles=tuple(role for role,*_ in _layouts(g))
    if g[6]!=min(g[0],int(hardware.resource_by_role('compute_vector').total_units)) or g[5]>hardware.resource_by_role('compute_vector').total_units:
        raise ValueError('CANN rotary active worker geometry differs from hardware')
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(roles,('query','key') if g[2] else ('query',),len(roles)):
        raise ValueError('CANN rotary public inputs, in-place outputs and readiness differ')
    _validate_parents(g,parents)
    scope=AccessScope(program_id,('cann_rotary',*map(str,g)),bindings.call_id,(),bindings.rank,bindings.stream,'logical-cann-rotary')
    a,v,e,f=_recipe(g,parents,dict(zip(roles,bindings.input_ready,strict=True)),scope,bindings.visibility_domain)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Two public GatherV3 prefixes, native source-declared BF16 slice geometry, typed logical UB value versions and in-place Q/K outputs.',
            'Position values and whole cache capacity remain unknown. Private versions do not assert native pool placement or instruction issue.',
            'Existing main cache-row/position pricing is retained separately from the complete logical gather/UB requests.')),
        (*bindings.allocations,*a),(*bindings.views,*v),(scope,),(*bindings.events,*e),(),f)
    cann_rotary_source_demands(program,expected_kernel=kernel);return program


def cann_rotary_source_demands(program, *, expected_kernel=None):
    work=_cann_rotary_source_demands(program)
    if expected_kernel is not None and tuple(map(int,program.scopes[0].op_ids[1:]))!=_geometry(expected_kernel):
        raise ValueError('CANN rotary source differs from actual kernel')
    # Cached validation applies only to this complete immutable program. Return
    # owned dictionaries so a consumer cannot alter another prediction's work.
    return {**work,'totals':dict(work['totals'])}


@runtime_lru_cache(maxsize=8)
def _cann_rotary_source_demands(program):
    program.__post_init__()
    if program.accesses or len(program.scopes)!=1 or len(program.tile_programs)!=1:raise ValueError('CANN rotary requires its complete compressed source')
    scope,=program.scopes;header=scope.op_ids
    if len(header)!=9 or header[0]!='cann_rotary' or scope.agent!='logical-cann-rotary' or program.hardware_id!='ascend_910b':raise ValueError('CANN rotary source header differs')
    g=tuple(map(int,header[1:]));t,q,k,d,slice_elements,workers,gworkers,key=g
    _validate_geometry(g)
    views=unique_by(program.views,'view_id');roles=tuple(role for role,*_ in _layouts(g));parents={role:views[role] for role in roles};ready={}
    _validate_parents(g,parents)
    for access in program.tile_programs[0].accesses:
        if access.input_ready:
            view=views[access.view_id]
            while view.view_id not in parents:view=views[view.parent_view_id]
            role=view.view_id
            if len(access.input_ready)!=1 or role in ready and ready[role]!=access.input_ready[0]:raise ValueError('CANN rotary input readiness differs')
            ready[role]=access.input_ready[0]
    a,v,e,f=_recipe(g,parents,ready,scope,program.tile_programs[0].visibility_domain)
    allocations=unique_by(program.allocations,'allocation_id');events=unique_by(program.events,'event_id')
    if (program.tile_programs!=f or set(views)!={*parents,*(view.view_id for view in v)}
            or any(views[view.view_id]!=view for view in v)
            or any(item.allocation_id not in allocations or allocations[item.allocation_id]!=item for item in a)
            or any(item.event_id not in events or events[item.event_id]!=item for item in e)):
        raise ValueError('CANN rotary lost its actual Gather/half-rotation/cache broadcast/cast READ_READY closure')
    family,=f;axes=unique_by(family.axes,'axis_id')
    amounts={access.access_id:_tile_template_requested_bytes(views,axes,family,access) for access in family.accesses}
    tensor_read=sum(value for name,value in amounts.items() if name.startswith('atb.') and name.endswith('.global_read'))
    tensor_write=sum(value for name,value in amounts.items() if name.startswith('atb.') and name.endswith('.global_write'))
    def slices(heads):return tuple(min(slice_elements,heads*d-start) for start in range(0,heads*d,slice_elements))
    return {'tokens':t,'active_elements':tensor_read//2,'rotary_dim':d,'tensor_read_bytes':tensor_read,'tensor_write_bytes':tensor_write,
        'per_token_position_bytes':t*4,'per_token_cache_row_bytes':t*d*2,
        'gather_read_bytes_per_launch':sum(value for name,value in amounts.items() if name.startswith('gather.cos.') and name.endswith(('.positions','.table'))),
        'gather_write_bytes_per_launch':sum(value for name,value in amounts.items() if name.startswith('gather.cos.') and name.endswith('.output')),
        'query_slices':slices(q),'key_slices':slices(k),'atb_workers':workers,'gather_workers':gworkers,'gather_key':10330,
        'logical_read_bytes':sum(amounts[access.access_id] for access in family.accesses if access.operation is Operation.READ),
        'logical_write_bytes':sum(amounts[access.access_id] for access in family.accesses if access.operation is Operation.WRITE),
        'totals':amounts,'legacy_main_price_projection':True}
