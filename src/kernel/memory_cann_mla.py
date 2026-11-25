"""CANN MLA decode: separate RoPE operands, aliased K/V and actual stacks."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from math import prod

from src.kernel.kernel_spec import KernelKind
from src.kernel.launch_config import FiaCrossDomainLaunchProgram
from src.kernel.memory_access import (
    AccessScope,AddressLevel,ConstantValue,Coverage,EventKind,EventRef,Granularity,
    LogicalCoordinateBinding,LogicalCoordinateTerm,LogicalSelectionView,
    MemoryAccessProgram,MemoryAllocation,Operation,Ordering,ProgramEvent,
    Semantics,TensorView,TileAccessTemplate,TileAxis,TileDependency,TileProgram,
    Unmapped,unique_by,_tile_template_requested_bytes,
)
from src.kernel.memory_flash_attention import dense_strides

_WORK=ContextVar('cann_mla_memory_source_work')


def has_cann_mla_source(kernel):
    from src.kernel.memory_cann_mla_split import split_geometry
    if split_geometry(kernel) is not None:return True
    if (kernel.kind is not KernelKind.FLASH_ATTN or kernel.hardware_id!='ascend_910b'
            or kernel.execution_support!='supported' or kernel.dtype!='bf16' or kernel.variant!='decode'
            or kernel.launch.algorithm!='cann_flash_attention_score:105000000000222220'
            or kernel.attributes['kv_layout']!='mla' or kernel.attributes['causal'] is not False
            or not isinstance(kernel.launch.source_program,FiaCrossDomainLaunchProgram)
            or tuple(kernel.shape[name] for name in ('q_len','head_dim','num_query_heads','num_kv_heads'))!=(1,512,64,1)
            or 'model_input_q_lens' in kernel.attributes or 'model_input_kv_lens' in kernel.attributes):
        return False
    batch=kernel.shape['batch'];length=kernel.shape['kv_len']
    return (tuple(kernel.attributes['q_lens'])==(1,)*batch and tuple(kernel.attributes['kv_lens'])==(length,)*batch
        and 0<length<=int(kernel.launch.parameters['cann_source_s2_base_size'])
        and all(task.is_head_first and task.is_head_last and not task.is_s2_split_core
            for group in kernel.launch.source_program.aic_groups for task in group.tasks))


def _header(kernel,hardware):
    if kernel.launch.algorithm=='cann_flash_attention_score:105000000000322220':
        from src.kernel.memory_cann_mla_split import split_header
        return split_header(kernel,hardware)
    if not has_cann_mla_source(kernel):raise ValueError('CANN MLA source requires its ordinary one-stack ABI')
    mechanism=hardware.fia_tiling_data_materialization_service
    if mechanism is None:raise ValueError('CANN MLA actor argument materialization is unbound')
    ranges=[];seen=set()
    for core,group in enumerate(kernel.launch.source_program.aic_groups):
        tasks=group.tasks
        if (group.aic_group!=core or not tasks or any(task.g_s1!=0 or task.s2!=0 or task.actual_m!=64
                or task.actual_s2!=kernel.shape['kv_len'] for task in tasks)):
            raise ValueError('CANN MLA ordinary source lost its actual head/stack geometry')
        indices=tuple(task.b_n2 for task in tasks)
        if indices!=tuple(range(indices[0],indices[0]+len(indices))) or seen.intersection(indices):
            raise ValueError('CANN MLA ordinary request ranges overlap or are not contiguous')
        seen.update(indices);ranges.append((indices[0],len(indices)))
    if seen!=set(range(kernel.shape['batch'])) or len(ranges)!=kernel.launch.grid_size:
        raise ValueError('CANN MLA ordinary source omitted an actual request or actor')
    return ('cann_mla_ordinary',str(kernel.shape['batch']),str(kernel.shape['kv_len']),
        *(str(mechanism[key]) for key in ('payload_bytes','aiv_per_group','aiv_transfer_bytes')),
        ';'.join(f'{start}:{count}' for start,count in ranges))


def _layouts(header):
    if not (len(header)==7 and header[0]=='cann_mla_ordinary' or len(header)==8 and header[0]=='cann_mla_split'):
        raise ValueError('Unknown CANN MLA source header')
    batch,length,payload,actors,transfer=map(int,header[1:6])
    if min(batch,length,payload)<=0 or header[0]=='cann_mla_ordinary' and length>512 or actors!=2 or transfer<payload:
        raise ValueError('CANN MLA ordinary source geometry differs')
    return (('tiling',(payload,),'uint8',1),('actual_query',(batch,),'int64',8),
        ('actual_key',(batch,),'int64',8),('block_table',(batch,-(-length//128)),'int32',4),
        ('query',(batch,64,512),'bf16',2),('query_rope',(batch,64,64),'bf16',2),
        ('key',(batch*length,512),'bf16',2),('key_rope',(batch*length,64),'bf16',2),
        ('output',(64,batch,1,512),'bf16',2))


def _availability(view):
    return Unmapped(view.shape,view.dtype,view.element_bytes,f'cann_mla.{view.view_id}.logical_operand')


def standalone_cann_mla_bindings(kernel,hardware, *, call_id,binding_id):
    from src.kernel.memory_access_builder import ElementwiseBindings
    header=_header(kernel,hardware);allocations=[];views=[];events=[];ready=[]
    for role,shape,dtype,width in _layouts(header):
        identity=f'{binding_id}.{role}'
        allocation=MemoryAllocation(identity,'logical-local-device','logical-native-tensor',None,
            AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view=(LogicalSelectionView(role,identity,shape,dtype,width,f'cann_mla.{role}.logical_operand',None)
            if role in ('key','key_rope') else TensorView(role,identity,0,shape,dense_strides(shape,width),dtype,width))
        allocations.append(allocation);views.append(view)
        events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        if role!='output':
            event=ProgramEvent(f'{identity}.available',EventKind.LOGICAL_AVAILABLE,identity,_availability(view),
                'cann-mla-visible',(EventRef(allocation.allocated_event,EventKind.ALLOCATED),))
            events.append(event);ready.append(EventRef(event.event_id,event.kind))
    roles=tuple(role for role,*_ in _layouts(header))
    return ElementwiseBindings(binding_id,call_id,tuple(allocations),tuple(views),tuple(events),roles[:-1],
        ('output',),tuple(ready),0,'offline','cann-mla-visible')


def _validate_parents(header,views):
    parents={role:views[role] for role,*_ in _layouts(header)}
    if len({view.allocation_id for view in parents.values()})!=len(parents):
        raise ValueError('CANN MLA independent operands alias; V must reuse the key view only')
    for role,shape,dtype,width in _layouts(header):
        view=parents[role]
        if (view.shape,view.dtype,view.element_bytes)!=(shape,dtype,width):raise ValueError('CANN MLA operand ABI differs')
        if role in ('key','key_rope'):
            if (not isinstance(view,LogicalSelectionView) or view.parent_view_id is not None or view.coordinate_map
                    or view.selection_id!=_availability(view).selection_id):raise ValueError('CANN MLA paged operands require their opaque logical selection')
        elif not isinstance(view,TensorView) or view.offset_bytes!=0 or view.strides_bytes!=dense_strides(shape,width):
            raise ValueError('CANN MLA BNSD/NBSD/length/table storage ABI differs')
    return parents


def _recipe(header,parents,ready,scope,visibility):
    if header[0]=='cann_mla_split':
        from src.kernel.memory_cann_mla_split import split_recipe
        return split_recipe(header,parents,ready,scope,visibility)
    batch,length=map(int,header[1:3]);ranges=tuple(tuple(map(int,item.split(':'))) for item in header[6].split(';'))
    if (sum(count for _,count in ranges)!=batch or any(count<=0 or start!=sum(value for _,value in ranges[:i])
            for i,(start,count) in enumerate(ranges))):raise ValueError('CANN MLA request ownership differs')
    allocations=[];views=[];events=[];families=[]
    evidence=('CANN8.5.1 FIA MLA ordinary BF16 BNSD_NBSD, one selected KV stack per request; source-selected AIC request ranges.',
        'Separate 512 no-PE and 64 RoPE input objects; PV reads the same key-nope storage as QK, never an invented V allocation.',
        'fia_block_vec_nonquant_mla.h SoftmaxFlashV2Compute/AmlaVecCompute: default max/sum, n/coefficient and aMlaSum; final Vec2 reads the FP32 PV result and aMlaSum.',
        'Logical page selections and head16 Vector work blocks preserve exact valid tails; physical addresses, native scalar lookup multiplicity and ring placement are unknown.')
    def coord(axis,terms=(),offset=0):return LogicalCoordinateBinding(axis,tuple(LogicalCoordinateTerm(i,c) for i,c in terms),offset)
    for core,(start,count) in enumerate(ranges):
        identity=f'{scope.scope_id}.aic{core}';accesses=[]
        def child(name,role,shape,mapping):
            parent=parents[role];view=LogicalSelectionView(f'{identity}.{name}',parent.allocation_id,shape,parent.dtype,parent.element_bytes,
                f'cann_mla.{core}.{name}',parent.view_id,mapping);views.append(view);return view
        def scratch(name,shape,dtype='fp32',width=4,domain='logical-ub-state'):
            key=f'{identity}.{name}';allocation=MemoryAllocation(key,'logical-local-device',domain,None,AddressLevel.OBJECT,None,None,f'{key}.allocated',None)
            view=LogicalSelectionView(key,key,shape,dtype,width,f'cann_mla.{core}.{name}',None)
            allocations.append(allocation);views.append(view);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,key,None,None,()));return view
        def dep(name,kind=EventKind.READ_READY,reduction=()):return TileDependency(name,kind,reduction)
        def add(name,view,operation,loops,mapping,dependencies=(),inputs=(),constant=None,access_scope=None,unmapped=True):
            accesses.append(TileAccessTemplate(name,view.view_id,operation,loops,mapping,inputs,dependencies,evidence,
                scope=access_scope,unmapped=unmapped,constant_value=constant))
        actor_controls={}
        for actor in ('aic','aiv0','aiv1'):
            add(f'argument_{actor}',parents['tiling'],Operation.READ,(),(None,),inputs=(ready['tiling'],))
            refs=[]
            for role in ('actual_query','actual_key'):
                view=child(f'{actor}.{role}',role,(count,),(coord(0,((0,1),),start),))
                add(f'{actor}.{role}',view,Operation.READ,('task',),('task',),(dep(f'argument_{actor}'),),(ready[role],));refs.append(dep(f'{actor}.{role}'))
            actor_controls[actor]=tuple(refs)
        clear=scratch('framework_clear',(32896,),'uint8',1,'logical-framework-gm')
        add('framework_clear',clear,Operation.WRITE,(),(None,),constant=ConstantValue.ZERO)
        controls=actor_controls['aic'];query_deps=[]
        for role,width in (('query',512),('query_rope',64)):
            view=child(role,role,(count,64,width),(coord(0,((0,1),),start),coord(1,((1,1),)),coord(2,((2,1),))))
            add(role,view,Operation.READ,('task','head'),('task','head',None),controls,(ready[role],));query_deps.append(dep(role))
        table=child('pages','block_table',(count,-(-length//128)),(coord(0,((0,1),),start),coord(1,((1,1),))))
        for phase in ('qk','pv'):add(f'{phase}.pages',table,Operation.READ,('task',),('task',None),controls,(ready['block_table'],))
        key_deps=[]
        for role,width in (('key',512),('key_rope',64)):
            view=child(role,role,(count,length,width),(coord(0,((0,length),(1,1)),start*length),coord(1,((2,1),))))
            add(role,view,Operation.READ,('task',),('task',None,None),(dep('qk.pages'),*controls),(ready[role],));key_deps.append(dep(role))
        # V is the same no-PE object, selected with the same coordinates.
        key=next(view for view in views if view.view_id==f'{identity}.key')
        add('value',key,Operation.READ,('task',),('task',None,None),(dep('pv.pages'),*controls),(ready['key'],))
        scores=scratch('scores',(count,64,length),domain='logical-gm-workspace')
        probability=scratch('probability',(count,64,length),'bf16',2,'logical-gm-workspace')
        partial=scratch('pv_result',(count,64,512),domain='logical-gm-workspace')
        outputs=child('output','output',(count,64,512),(coord(0,((1,1),)),coord(1,((0,1),),start),coord(2),coord(3,((2,1),))))
        for actor,offset in (('aiv0',0),('aiv1',32)):
            row=f'{actor}.head';default={}
            for name in ('max','sum'):
                key=f'{identity}.{actor}.default_{name}'
                allocation=MemoryAllocation(key,'logical-local-device','logical-ub-state',2048,AddressLevel.OBJECT,None,None,f'{key}.allocated',None)
                allocations.append(allocation);view=TensorView(key,key,0,(512,),(4,),'fp32',4);views.append(view)
                events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,key,None,None,()))
                initializer=replace(scope,scope_id=f'{identity}.{actor}.{name}.initializer',op_ids=('Duplicate','-inf' if name=='max' else '0.0'),agent=f'logical-cann-mla-{actor}-initializer')
                add(f'{actor}.default_{name}',view,Operation.WRITE,(),(None,),(dep(f'argument_{actor}'),),constant=ConstantValue.ZERO if name=='sum' else None,access_scope=initializer,unmapped=False)
                active=TensorView(f'{view.view_id}.active',view.allocation_id,0,(32,),(4,),'fp32',4);views.append(active)
                add(f'{actor}.{name}_read',active,Operation.READ,('task',row),(row,),(dep(f'{actor}.default_{name}',EventKind.WRITE_VISIBLE),),unmapped=False);default[name]=dep(f'{actor}.{name}_read')
            selected={}
            for name,parent in (('scores',scores),('probability',probability),('pv_result',partial),('output',outputs)):
                view=LogicalSelectionView(f'{identity}.{actor}.{name}',parent.allocation_id,(count,32,parent.shape[2]),parent.dtype,parent.element_bytes,
                    f'cann_mla.{core}.{actor}.{name}',parent.view_id,(coord(0,((0,1),)),coord(1,((1,1),),offset),coord(2,((2,1),))))
                views.append(view);selected[name]=view
            # Bind each Vector head16 tile to its own producer head16 tile.
            from src.kernel.memory_access import TileIndexBinding,TileIndexTerm
            head_binding=(TileIndexBinding('head',(TileIndexTerm(row,1),),offset//16),)
            add(f'{actor}.scores_write',selected['scores'],Operation.WRITE,('task',row),('task',row,None),
                (*(TileDependency(role,EventKind.READ_READY,(),head_binding) for role in ('query','query_rope')),
                    *key_deps,dep('framework_clear',EventKind.WRITE_VISIBLE)))
            add(f'{actor}.scores_read',selected['scores'],Operation.READ,('task',row),('task',row,None),
                (dep(f'{actor}.scores_write',EventKind.WRITE_VISIBLE),*actor_controls[actor]))
            states={name:scratch(f'{actor}.{name}',(count,32)) for name in ('max','sum','n','coefficient','amla_sum')}
            state_deps=(dep(f'{actor}.scores_read'),default['max'],default['sum'])
            state_inputs={
                'max':state_deps,
                'sum':(*state_deps,dep(f'{actor}.max_write',EventKind.WRITE_VISIBLE)),
                'n':(dep(f'{actor}.max_write',EventKind.WRITE_VISIBLE),),
                'coefficient':(dep(f'{actor}.max_write',EventKind.WRITE_VISIBLE),dep(f'{actor}.n_write',EventKind.WRITE_VISIBLE)),
                'amla_sum':(dep(f'{actor}.sum_write',EventKind.WRITE_VISIBLE),dep(f'{actor}.n_write',EventKind.WRITE_VISIBLE),dep(f'{actor}.coefficient_write',EventKind.WRITE_VISIBLE)),
            }
            for name in ('max','sum','n','coefficient','amla_sum'):
                add(f'{actor}.{name}_write',states[name],Operation.WRITE,('task',row),('task',row),state_inputs[name])
            add(f'{actor}.probability_write',selected['probability'],Operation.WRITE,('task',row),('task',row,None),
                (*state_deps,dep(f'{actor}.max_write',EventKind.WRITE_VISIBLE),dep(f'{actor}.coefficient_write',EventKind.WRITE_VISIBLE)))
            add(f'{actor}.probability_read',selected['probability'],Operation.READ,('task',row),('task',row,None),
                (dep(f'{actor}.probability_write',EventKind.WRITE_VISIBLE),dep('argument_aic')))
            add(f'{actor}.pv_write',selected['pv_result'],Operation.WRITE,('task',row),('task',row,None),
                (dep(f'{actor}.probability_read'),dep('value')))
            add(f'{actor}.pv_read',selected['pv_result'],Operation.READ,('task',row),('task',row,None),
                (dep(f'{actor}.pv_write',EventKind.WRITE_VISIBLE),*actor_controls[actor]))
            add(f'{actor}.amla_sum_read',states['amla_sum'],Operation.READ,('task',row),('task',row),(dep(f'{actor}.amla_sum_write',EventKind.WRITE_VISIBLE),))
            add(f'{actor}.output',selected['output'],Operation.WRITE,('task',row),('task',row,None),
                (dep(f'{actor}.pv_read'),dep(f'{actor}.amla_sum_read')))
        family_scope=replace(scope,scope_id=identity,agent='logical-cann-mla-ordinary-stacks')
        families.append(TileProgram(identity,family_scope,(TileAxis('task',count,1),TileAxis('head',64,16),
            TileAxis('aiv0.head',32,16),TileAxis('aiv1.head',32,16)),tuple(accesses),visibility,evidence))
    # InitOutputSingleCore partitions the flat NBSD output among all AIVs,
    # then SyncAll precedes every ordinary task. Keep that real global edge
    # in one family; per-core families cannot express its cross-core join.
    init=[];total=batch*64*512;chunk=-(-total//(2*len(ranges)))
    for worker in range(2*len(ranges)):
        begin=worker*chunk;size=min(chunk,total-begin)
        if size<=0:continue
        identity=f'init.aiv{worker}'
        view=LogicalSelectionView(f'{scope.scope_id}.{identity}',parents['output'].allocation_id,(size,),'bf16',2,
            f'cann_mla.{identity}',parents['output'].view_id,(
                LogicalCoordinateBinding(0,(LogicalCoordinateTerm(0,1),),begin,batch*512),
                LogicalCoordinateBinding(1,(LogicalCoordinateTerm(0,1),),begin,512,batch),
                LogicalCoordinateBinding(2,()),
                LogicalCoordinateBinding(3,(LogicalCoordinateTerm(0,1),),begin,1,512)))
        views.append(view)
        initializer=replace(scope,scope_id=f'{scope.scope_id}.{identity}.initializer',op_ids=('InitOutput','0'),agent='logical-cann-mla-output-initializer')
        init.append(TileAccessTemplate(identity,view.view_id,Operation.WRITE,(),(None,),(),
            (TileDependency(f'aic{worker//2}.argument_aiv{worker%2}',EventKind.READ_READY,()),),evidence,
            scope=initializer,unmapped=True,constant_value=ConstantValue.ZERO))
    initialized=tuple(TileDependency(access.access_id,EventKind.WRITE_VISIBLE,()) for access in init)
    merged_axes=[];merged=[]
    for core,family in enumerate(families):
        axes={axis.axis_id:f'aic{core}.{axis.axis_id}' for axis in family.axes}
        identities={access.access_id:f'aic{core}.{access.access_id}' for access in family.accesses}
        merged_axes.extend(replace(axis,axis_id=axes[axis.axis_id]) for axis in family.axes)
        for access in family.accesses:
            deps=tuple(replace(dep,access_id=identities[dep.access_id],
                reduction_axes=tuple(axes[name] for name in dep.reduction_axes),
                coordinate_bindings=tuple(replace(binding,axis_id=axes[binding.axis_id],
                    terms=tuple(replace(term,axis_id=axes[term.axis_id]) for term in binding.terms)) for binding in dep.coordinate_bindings))
                for dep in access.dependencies)
            if not access.access_id.startswith('argument_') and access.access_id!='framework_clear':deps+=initialized
            merged.append(replace(access,access_id=identities[access.access_id],
                iteration_axes=tuple(axes[name] for name in access.iteration_axes),
                view_axes=tuple(axes[name] if name is not None else None for name in access.view_axes),dependencies=deps))
    family=TileProgram(scope.scope_id,scope,tuple(merged_axes),(*init,*merged),visibility,(*evidence,
        'FiaKernelNonQuantMla::IsInitAttentionOutGm/InitOutputSingleCore: supplied actual Q lengths require zero initialization of disjoint flat NBSD output intervals; all AIVs SyncAll before tasks.'))
    return tuple(allocations),tuple(views),tuple(events),(family,)


def build_cann_mla_source(kernel,hardware,bindings, *, program_id):
    header=_header(kernel,hardware);parents=_validate_parents(header,unique_by(bindings.views,'view_id'));roles=tuple(parents)
    if (bindings.read_views,bindings.write_views,len(bindings.input_ready))!=(roles[:-1],('output',),len(roles)-1):
        raise ValueError('CANN MLA bindings omit actual RoPE, cache, control or argument operands')
    ready=dict(zip(bindings.read_views,bindings.input_ready,strict=True))
    scope=AccessScope(program_id,header,bindings.call_id,(),bindings.rank,bindings.stream,
        'logical-cann-mla-split-stacks' if header[0]=='cann_mla_split' else 'logical-cann-mla-ordinary-stacks')
    allocations,views,events,families=_recipe(header,parents,ready,scope,bindings.visibility_domain)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Source-selected MLA: separate RoPE operands, aliased K/V, controls, stack recurrence and actual split-head terminal when selected.',
            'Native page values, ring addresses, transaction issue counts and cache residency remain unknown.') if header[0]=='cann_mla_split' else (
            'Ordinary selected one-stack MLA source: separate RoPE operands, aliased K/V, controls and GM/UB handoffs.',
            'Native page values, ring addresses, transaction issue counts and cache residency remain unknown; split-core recurrence is a separate source.')),
        (*bindings.allocations,*allocations),(*bindings.views,*views),(scope,),(*bindings.events,*events),(),families)
    cann_mla_source_demands(program,expected_kernel=kernel,hardware=hardware);return program


def cann_mla_source_demands(program, *, expected_kernel=None,hardware=None):
    program.__post_init__()
    if len(program.scopes)!=1 or program.accesses or not program.tile_programs:raise ValueError('CANN MLA requires its unique compressed source')
    scope,=program.scopes;header=scope.op_ids;views=unique_by(program.views,'view_id');parents=_validate_parents(header,views)
    if expected_kernel is not None and (hardware is None or _header(expected_kernel,hardware)!=header or program.hardware_id!=expected_kernel.hardware_id):
        raise ValueError('CANN MLA source differs from its actual kernel')
    ready={}
    for family in program.tile_programs:
        for access in family.accesses:
            if access.input_ready:
                if len(access.input_ready)!=1:raise ValueError('CANN MLA requires exact input readiness')
                view=views[access.view_id]
                while view.view_id not in parents:view=views[view.parent_view_id]
                role=view.view_id;ref,=access.input_ready
                if role in ready and ready[role]!=ref:raise ValueError('CANN MLA input availability differs across actors')
                ready[role]=ref
    events=unique_by(program.events,'event_id')
    for role,ref in ready.items():
        if ref.kind is not EventKind.LOGICAL_AVAILABLE or events[ref.node_id].region!=_availability(parents[role]):
            raise ValueError('CANN MLA requires actual selected operand availability')
    allocations,selected,expected_events,families=_recipe(header,parents,ready,scope,program.tile_programs[0].visibility_domain)
    if (program.tile_programs!=families or set(views)!={*parents,*(view.view_id for view in selected)}
            or any(views[view.view_id]!=view for view in selected) or any(item not in program.allocations for item in allocations)
            or any(item not in program.events for item in expected_events)):
        raise ValueError('CANN MLA lost its actual RoPE, aliased value, state or READ_READY dependency')
    if header[0]=='cann_mla_split':
        from src.kernel.memory_cann_mla_split import split_demands
        return split_demands(program,parents)
    totals={}
    for family in families:
        axes=unique_by(family.axes,'axis_id')
        for access in family.accesses:
            amount=_tile_template_requested_bytes(views,axes,family,access)
            name=access.access_id.split('.',1)[1] if access.access_id.startswith('aic') else access.access_id
            totals[name]=amount+(totals[name] if name in totals else 0)
    values={role:prod(view.shape)*view.element_bytes for role,view in parents.items()}
    query=totals['query']+totals['query_rope'];key=totals['key']+totals['key_rope'];value=totals['value']
    output=totals['aiv0.output']+totals['aiv1.output']
    if (query,key,value,output)!=(values['query']+values['query_rope'],values['key']+values['key_rope'],values['key'],values['output']):
        raise ValueError('CANN MLA projected payload does not partition its actual operands')
    return {'query_unique_bytes':query,'key_interface_bytes':key,'value_interface_bytes':value,'kv_unique_bytes':None,
        'write_bytes':output,'priced_interface_bytes':query+key+value+output,'priced_score_elements':(totals['aiv0.scores_write']+totals['aiv1.scores_write'])//4,
        'priced_output_elements':output//2,'priced_task_query_l2_bytes':query,'key_requested_bytes':key,'value_requested_bytes':value,
        'framework_clear_write_bytes':totals['framework_clear'],
        'output_initialization_write_bytes':sum(amount for name,amount in totals.items() if name.startswith('init.aiv')),
        'totals':totals,'value_aliases_key':True,'native_page_addresses_bound':False}


@contextmanager
def cann_mla_source_work(kernel,work):
    token=_WORK.set((kernel,work))
    try:yield
    finally:_WORK.reset(token)


def current_cann_mla_source_work(kernel,hardware):
    try:bound,work=_WORK.get()
    except LookupError:
        from src.kernel.memory_access_builder import build_memory_access_program
        return cann_mla_source_demands(build_memory_access_program(kernel,hardware,None,Granularity.TILE,program_id='cann.mla'),expected_kernel=kernel,hardware=hardware)
    if bound is not kernel:raise ValueError('CANN MLA work belongs to a different invocation')
    return work


def bound_cann_mla_source_work(kernel):
    try:bound,work=_WORK.get()
    except LookupError:raise ValueError('CANN MLA workspace pricing requires its bound production source') from None
    if bound is not kernel:raise ValueError('CANN MLA work belongs to a different invocation')
    return work
