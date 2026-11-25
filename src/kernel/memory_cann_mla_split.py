"""Source-selected CANN MLA stack recurrence and mixed FlashDecode heads.

Logical RMW pairs describe the synchronized INT32 Nupdate/FP32 PV updates
of one accumulator allocation. They do not model atomic instruction issue.
"""
from dataclasses import replace
from math import prod

from src.kernel.kernel_spec import KernelKind
from src.kernel.launch_config import FiaCrossDomainLaunchProgram
from src.kernel.memory_access import (
    AddressLevel,ConstantValue,EventKind,LogicalCoordinateBinding,LogicalCoordinateTerm,
    LogicalSelectionView,MemoryAllocation,Operation,ProgramEvent,TensorView,
    TileAccessTemplate,TileAxis,TileDependency,TileIndexBinding,TileIndexTerm,
    TileProgram,_tile_template_requested_bytes,unique_by,
)


def split_geometry(kernel):
    if (kernel.kind is not KernelKind.FLASH_ATTN or kernel.hardware_id!='ascend_910b'
            or kernel.execution_support!='supported' or kernel.dtype!='bf16' or kernel.variant!='decode'
            or kernel.launch.algorithm!='cann_flash_attention_score:105000000000322220'
            or kernel.attributes['kv_layout']!='mla' or kernel.attributes['causal'] is not False
            or not isinstance(kernel.launch.source_program,FiaCrossDomainLaunchProgram)
            or tuple(kernel.shape[name] for name in ('q_len','head_dim','num_query_heads','num_kv_heads'))!=(1,512,64,1)
            or 'model_input_q_lens' in kernel.attributes or 'model_input_kv_lens' in kernel.attributes):return None
    batch=kernel.shape['batch'];kv=kernel.shape['kv_len'];block=int(kernel.launch.parameters['cann_source_s2_base_size'])
    if (block!=512 or tuple(kernel.attributes['q_lens'])!=(1,)*batch
            or tuple(kernel.attributes['kv_lens'])!=(kv,)*batch):return None
    stacks=-(-kv//block);cursor=0;segments=[];partial={}
    for core,group in enumerate(kernel.launch.source_program.aic_groups):
        if group.aic_group!=core or not group.tasks:return None
        tasks=group.tasks;offset=0;local=[]
        while offset<len(tasks):
            first=tasks[offset];request=first.b_n2;start=first.s2;count=0;split=first.is_s2_split_core
            while offset<len(tasks) and tasks[offset].b_n2==request:
                task=tasks[offset];stack=cursor%stacks
                if (task.b_n2!=cursor//stacks or task.s2!=stack or task.g_s1!=0 or task.actual_m!=64
                        or task.actual_s2!=min(block,kv-stack*block) or task.is_s2_split_core!=split
                        or task.is_head_first!=(count==0)
                        or task.is_head_last!=(offset==len(tasks)-1 or tasks[offset+1].b_n2!=request)):return None
                count+=1;cursor+=1;offset+=1
            if not split and (start!=0 or count!=stacks):return None
            if split:
                if request not in partial:partial[request]=[]
                partial[request].append((start,count))
            segment=(core,request,1,start,count,int(split))
            if local and not split and local[-1][5]==0 and local[-1][1]+local[-1][2]==request:
                old=local[-1];local[-1]=(*old[:2],old[2]+1,*old[3:])
            else:local.append(segment)
        segments.extend(local)
    if cursor!=batch*stacks or len(kernel.launch.source_program.aic_groups)!=kernel.launch.grid_size or not partial:return None
    for parts in partial.values():
        position=0
        if len(parts)<2:return None
        for start,count in parts:
            if start!=position:return None
            position+=count
        if position!=stacks:return None
    return tuple(segments)


def split_header(kernel,hardware):
    segments=split_geometry(kernel)
    if segments is None:raise ValueError('CANN MLA split source requires exact selected mixed-head geometry')
    mechanism=hardware.fia_tiling_data_materialization_service
    if mechanism is None:raise ValueError('CANN MLA actor argument materialization is unbound')
    return ('cann_mla_split',str(kernel.shape['batch']),str(kernel.shape['kv_len']),
        *(str(mechanism[key]) for key in ('payload_bytes','aiv_per_group','aiv_transfer_bytes')),
        str(kernel.launch.grid_size),';'.join(':'.join(map(str,item)) for item in segments))


def split_recipe(header,parents,ready,scope,visibility):
    if len(header)!=8 or header[0]!='cann_mla_split':raise ValueError('Unknown split MLA header')
    batch,kv,payload,actors,transfer,cores=map(int,header[1:7]);stacks=-(-kv//512)
    segments=tuple(tuple(map(int,item.split(':'))) for item in header[7].split(';'))
    if min(batch,kv,payload,cores)<=0 or actors!=2 or transfer<payload:raise ValueError('CANN MLA split ABI differs')
    evidence=('CANN8.5.1 v3 key105000000000322220: actual AIC request/stack intervals; mixed complete and split heads.',
        'fia_block_cube_nonquant_mla.h reloads query_nope/query_rope for each KV stack; PV aliases key_nope.',
        'fia_block_vec_nonquant_mla.h:456-561 carries max/sum/n/coefficient; :656-733 INT32 Nupdate RMW precedes :984-1003 cube FP32 PV RMW on the same accumulator.',
        'Logical synchronized RMW read/write pairs retain bit reinterpretation and dependency; no atomic issue counts, native ring addresses or cache residency are asserted.',
        'Split Vec1 publishes eight FP32 sum/max lanes/head; final Vec2 normalizes FP32 partial O; kernel FlashDecode SyncAll precedes weighted terminal reduction.')
    allocations=[];views=[];events=[];accesses=[];axes=[];parts={};split_heads={}
    def coord(axis,terms=(),offset=0,divisor=1,modulus=None):
        return LogicalCoordinateBinding(axis,tuple(LogicalCoordinateTerm(i,c) for i,c in terms),offset,divisor,modulus)
    def scratch(name,shape,dtype='fp32',width=4,domain='logical-ub-state'):
        identity=f'{scope.scope_id}.{name}'
        allocation=MemoryAllocation(identity,'logical-local-device',domain,None,AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        view=LogicalSelectionView(identity,identity,shape,dtype,width,f'cann_mla_split.{name}',None)
        allocations.append(allocation);views.append(view);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()));return view
    def child(name,parent,shape,mapping,dtype=None,width=None):
        view=LogicalSelectionView(f'{scope.scope_id}.{name}',parent.allocation_id,shape,
            parent.dtype if dtype is None else dtype,parent.element_bytes if width is None else width,
            f'cann_mla_split.{name}',parent.view_id,mapping)
        views.append(view);return view
    def dep(name,kind=EventKind.READ_READY,bindings=(),reduction=()):return TileDependency(name,kind,reduction,bindings)
    def add(name,view,operation,loops=(),mapping=None,deps=(),inputs=(),value=None,symbol=None):
        selected_scope=None if symbol is None else replace(scope,scope_id=f'{scope.scope_id}.{name}.scope',op_ids=('Duplicate',symbol),agent='logical-cann-mla-state-initializer')
        access=TileAccessTemplate(name,view.view_id,operation,loops,(None,)*len(view.shape) if mapping is None else mapping,
            inputs,deps,evidence,scope=selected_scope,unmapped=True,constant_value=value)
        accesses.append(access);return name
    # InitOutputSingleCore clears disjoint flat NBSD ranges, followed by SyncAll.
    initializers=[];total=batch*64*512;chunk=-(-total//(2*cores))
    for worker in range(2*cores):
        begin=worker*chunk;size=min(chunk,total-begin)
        if size<=0:continue
        view=child(f'init.aiv{worker}',parents['output'],(size,),(
            coord(0,((0,1),),begin,batch*512),coord(1,((0,1),),begin,512,batch),coord(2),coord(3,((0,1),),begin,1,512)))
        initializers.append(add(f'init.aiv{worker}',view,Operation.WRITE,value=ConstantValue.ZERO))
    initialized=tuple(dep(name,EventKind.WRITE_VISIBLE) for name in initializers)
    defaults={}
    for core in range(cores):
        for actor in ('aic','aiv0','aiv1'):
            add(f'core{core}.argument_{actor}',parents['tiling'],Operation.READ,inputs=(ready['tiling'],))
        clear=scratch(f'core{core}.framework_clear',(32896,),'uint8',1,'logical-framework-gm')
        add(f'core{core}.framework_clear',clear,Operation.WRITE,value=ConstantValue.ZERO)
        for actor in ('aiv0','aiv1'):
            for state in ('max','sum'):
                name=f'core{core}.{actor}.default_{state}'
                identity=f'{scope.scope_id}.{name}'
                allocation=MemoryAllocation(identity,'logical-local-device','logical-ub-state',2048,AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
                root=TensorView(identity,identity,0,(512,),(4,),'fp32',4)
                allocations.append(allocation);views.append(root);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
                add(name,root,Operation.WRITE,deps=(dep(f'core{core}.argument_{actor}'),),value=ConstantValue.ZERO if state=='sum' else None,symbol='0.0' if state=='sum' else '-inf')
                accesses[-1]=replace(accesses[-1],unmapped=False)
                active=TensorView(f'{identity}.active',identity,0,(32,),(4,),'fp32',4)
                views.append(active);defaults[core,actor,state]=(active,name)
    cursor=0;seen=set()
    for index,segment in enumerate(segments):
        if len(segment)!=6:raise ValueError('CANN MLA split segment arity differs')
        core,request,count,start,length,split=segment
        if (not 0<=core<cores or count<=0 or length<=0 or start<0 or start+length>stacks or split not in (0,1)
                or request*stacks+start!=cursor or split and count!=1 or not split and (start!=0 or length!=stacks)):
            raise ValueError('CANN MLA split request/stack partition differs')
        cursor+=count*length;seen.add(core);prefix=f's{index}';owner=f'fd{request}' if split else prefix
        task=f'{owner}.task';head=f'{owner}.head';middle=f'{prefix}.middle'
        axes.extend((TileAxis(task,count,1),TileAxis(head,64,16)))
        if length>2:axes.append(TileAxis(middle,length-2,1))
        controls={}
        for actor in ('aic','aiv0','aiv1'):
            refs=[]
            for role in ('actual_query','actual_key'):
                view=child(f'{prefix}.{actor}.{role}',parents[role],(count,),(coord(0,((0,1),),request),))
                name=add(f'{prefix}.{actor}.{role}',view,Operation.READ,(task,),(task,),
                    (dep(f'core{core}.argument_{actor}'),*initialized),(ready[role],));refs.append(dep(name))
            controls[actor]=tuple(refs)
        phases=[('first',start,1)]
        if length>2:phases.append(('middle',start+1,length-2))
        if length>1:phases.append(('last',start+length-1,1))
        shared={}
        for phase,stack,phase_count in phases:
            repeat=phase=='middle';loops=(task,middle) if repeat else (task,);columns=512 if repeat else min(512,kv-stack*512)
            for role,width in (('query',512),('query_rope',64)):
                view=child(f'{prefix}.{phase}.{role}',parents[role],(count,64,width),
                    (coord(0,((0,1),),request),coord(1,((1,1),)),coord(2,((2,1),))))
                name=add(f'{prefix}.{phase}.{role}',view,Operation.READ,(*loops,head),(task,head,None),controls['aic'],(ready[role],));shared[phase,role]=name
            pages=-(-columns//128)
            table_shape=(count,phase_count,pages) if repeat else (count,pages)
            mapping=(coord(0,((0,1),),request),coord(1,((1,4),(2,1)),stack*4)) if repeat else (coord(0,((0,1),),request),coord(1,((1,1),),stack*4))
            table=child(f'{prefix}.{phase}.pages',parents['block_table'],table_shape,mapping)
            for stage in ('qk','pv'):
                add(f'{prefix}.{phase}.{stage}.pages',table,Operation.READ,loops,(*loops,None),controls['aic'],(ready['block_table'],))
            for role,width in (('key',512),('key_rope',64)):
                shape=(count,phase_count,columns,width) if repeat else (count,columns,width)
                mapping=(coord(0,((0,kv),(1,512),(2,1)),request*kv+stack*512),coord(1,((3,1),))) if repeat else (
                    coord(0,((0,kv),(1,1)),request*kv+stack*512),coord(1,((2,1),)))
                view=child(f'{prefix}.{phase}.{role}',parents[role],shape,mapping)
                name=add(f'{prefix}.{phase}.{role}',view,Operation.READ,loops,(*loops,None,None),
                    (dep(f'{prefix}.{phase}.qk.pages'),*controls['aic']),(ready[role],));shared[phase,role]=name
                if role=='key':
                    shared[phase,'value']=add(f'{prefix}.{phase}.value',view,Operation.READ,loops,(*loops,None,None),
                        (dep(f'{prefix}.{phase}.pv.pages'),*controls['aic']),(ready['key'],))
        if split:
            if request not in split_heads:split_heads[request]=[]
            split_heads[request].append(index)
        for actor,offset in (('aiv0',0),('aiv1',32)):
            row=f'{owner}.{actor}.head';axes.append(TileAxis(row,32,16));rv=(task,row);pfx=f'{prefix}.{actor}'
            states={name:scratch(f'{pfx}.{name}',(count,32)) for name in ('max','sum','n','coefficient','amla_sum')}
            accumulator=scratch(f'{pfx}.accumulator',(count,32,512),domain='logical-gm-workspace')
            # FP32 is the storage ABI. Nupdate reinterprets its bits for the
            # synchronized native INT32 add, retained explicitly in scope.
            previous=None
            for phase,stack,phase_count in phases:
                repeat=phase=='middle';last=phase==phases[-1][0];p=f'{pfx}.{phase}';loops=(task,middle,row) if repeat else rv
                columns=512 if repeat else min(512,kv-stack*512);sv=rv;mv=(*rv,None)
                shape=(count,phase_count,32,columns) if repeat else (count,32,columns);mapping=(*loops,None)
                scores=scratch(f'{p}.scores',shape,domain='logical-gm-workspace');prob=scratch(f'{p}.probability',shape,'bf16',2,'logical-gm-workspace')
                head_binding=(TileIndexBinding(head,(TileIndexTerm(row,1),),offset//16),)
                add(f'{p}.scores_write',scores,Operation.WRITE,loops,mapping,
                    (*(dep(shared[phase,role],bindings=head_binding) for role in ('query','query_rope')),
                        dep(shared[phase,'key']),dep(shared[phase,'key_rope']),dep(f'core{core}.framework_clear',EventKind.WRITE_VISIBLE)))
                add(f'{p}.scores_read',scores,Operation.READ,loops,mapping,(dep(f'{p}.scores_write',EventKind.WRITE_VISIBLE),*controls[actor]))
                final_binding=(TileIndexBinding(middle,(),phase_count-1),) if repeat else ()
                prior_binding=(TileIndexBinding(middle,(),length-3),) if previous=='middle' else ()
                def carried(name):
                    return (TileDependency(f'{pfx}.middle.{name}',EventKind.WRITE_VISIBLE,(),carry_axis=middle,
                        initial_access_id=f'{pfx}.first.{name}') if repeat else dep(f'{pfx}.{previous}.{name}',EventKind.WRITE_VISIBLE,prior_binding))
                old=[]
                for name in ('max','sum','n','coefficient'):
                    if phase=='first':
                        if name not in ('max','sum'):continue
                        view,writer=defaults[core,actor,name]
                        add(f'{p}.{name}_read',view,Operation.READ,rv,(row,),(dep(writer,EventKind.WRITE_VISIBLE),))
                        accesses[-1]=replace(accesses[-1],unmapped=False)
                    else:add(f'{p}.{name}_read',states[name],Operation.READ,loops,sv,(carried(f'{name}_write'),))
                    old.append(dep(f'{p}.{name}_read'))
                base=(dep(f'{p}.scores_read'),*old)
                for name in ('max','sum','n','coefficient'):
                    deps=base if name=='max' else (*base,dep(f'{p}.max_write',EventKind.WRITE_VISIBLE))
                    add(f'{p}.{name}_write',states[name],Operation.WRITE,loops,sv,deps)
                add(f'{p}.probability_write',prob,Operation.WRITE,loops,mapping,(*base,dep(f'{p}.coefficient_write',EventKind.WRITE_VISIBLE)))
                add(f'{p}.probability_read',prob,Operation.READ,loops,mapping,(dep(f'{p}.probability_write',EventKind.WRITE_VISIBLE),))
                pvdeps=(dep(f'{p}.probability_read'),dep(shared[phase,'value']))
                if phase!='first':
                    update=scratch(f'{p}.nupdate',(count,phase_count,32) if repeat else (count,32),'int32',4)
                    uv=(*loops,) if repeat else rv
                    add(f'{p}.nupdate_write',update,Operation.WRITE,loops,uv,(*old,dep(f'{p}.n_write',EventKind.WRITE_VISIBLE),dep(f'{p}.coefficient_write',EventKind.WRITE_VISIBLE)))
                    add(f'{p}.nupdate_read',update,Operation.READ,loops,uv,(dep(f'{p}.nupdate_write',EventKind.WRITE_VISIBLE),))
                    for suffix,operation,deps in (
                            ('read',Operation.READ,(carried('pv_write'),)),
                            ('write',Operation.WRITE,(dep(f'{p}.nupdate_acc_read'),dep(f'{p}.nupdate_read')))):
                        identity=add(f'{p}.nupdate_acc_{suffix}',accumulator,operation,loops,mv,deps)
                        accesses[-1]=replace(accesses[-1],scope=replace(scope,scope_id=f'{scope.scope_id}.{identity}.scope',
                            op_ids=('synchronized_accumulator_rmw','int32_bit_add'),agent='logical-cann-mla-nupdate'))
                    add(f'{p}.pv_acc_read',accumulator,Operation.READ,loops,mv,(dep(f'{p}.nupdate_acc_write',EventKind.WRITE_VISIBLE),))
                    pvdeps+= (dep(f'{p}.pv_acc_read'),)
                add(f'{p}.pv_write',accumulator,Operation.WRITE,loops,mv,pvdeps)
                if last:
                    add(f'{p}.amla_sum_write',states['amla_sum'],Operation.WRITE,loops,sv,
                        (dep(f'{p}.sum_write',EventKind.WRITE_VISIBLE),dep(f'{p}.n_write',EventKind.WRITE_VISIBLE),dep(f'{p}.coefficient_write',EventKind.WRITE_VISIBLE)))
                    add(f'{pfx}.pv_read',accumulator,Operation.READ,rv,mv,(dep(f'{p}.pv_write',EventKind.WRITE_VISIBLE,final_binding),))
                    add(f'{pfx}.amla_sum_read',states['amla_sum'],Operation.READ,rv,sv,(dep(f'{p}.amla_sum_write',EventKind.WRITE_VISIBLE,final_binding),))
                    output=(scratch(f'{pfx}.partial',(count,32,512),domain='logical-gm-workspace') if split else child(f'{pfx}.output',parents['output'],(count,32,512),
                        (coord(0,((1,1),),offset),coord(1,((0,1),),request),coord(2),coord(3,((2,1),)))))
                    writer=add(f'{pfx}.partial' if split else f'{pfx}.output',output,Operation.WRITE,rv,mv,(dep(f'{pfx}.pv_read'),dep(f'{pfx}.amla_sum_read')))
                    if split:
                        packed={}
                        for name in ('max','sum'):
                            read=add(f'{pfx}.fd_{name}_read',states[name],Operation.READ,rv,sv,(dep(f'{p}.{name}_write',EventKind.WRITE_VISIBLE,final_binding),))
                            gm=scratch(f'{pfx}.fd_{name}',(count,32,8),domain='logical-gm-workspace')
                            write=add(f'{pfx}.fd_{name}_write',gm,Operation.WRITE,rv,(*rv,None),(dep(read),));packed[name]=(gm,write)
                        parts[index,actor]=(output,writer,packed,rv,offset)
                previous=phase
    if cursor!=batch*stacks or seen!=set(range(cores)) or not split_heads:raise ValueError('CANN MLA split domain omitted requests or actors')
    barrier=tuple(dep(access.access_id,EventKind.WRITE_VISIBLE if access.operation is Operation.WRITE else EventKind.READ_READY,
        reduction=access.iteration_axes) for access in accesses if access.access_id.endswith(('.output','.partial','fd_max_write','fd_sum_write'))
            or '.argument_' in access.access_id)
    # Terminal worker assignment is unknown; each actual partial remains distinct.
    for request,indexes in sorted(split_heads.items()):
        if len(indexes)<2:raise ValueError('Split MLA terminal requires all real producers')
        task=f'fd{request}.task';axes.append(TileAxis(task,1,1))
        for actor,offset in (('aiv0',0),('aiv1',32)):
            row=f'fd{request}.{actor}.head';axes.append(TileAxis(row,32,16))
            loops=(task,row);mv=(*loops,None);prefix=f'fd{request}.{actor}';matrices=[];lse={'max':[],'sum':[]}
            for index in indexes:
                output,writer,packed,producer_axes,_=parts[index,actor]
                if producer_axes!=loops:raise ValueError('CANN MLA terminal lost its own split-head axes')
                bindings=()
                matrices.append(add(f'{prefix}.p{index}.partial_read',output,Operation.READ,loops,mv,
                    (dep(writer,EventKind.WRITE_VISIBLE,bindings),dep(f'{prefix}.max_init',EventKind.WRITE_VISIBLE),
                        *(item for item in barrier if item.access_id!=writer))))
                for name,(view,producer) in packed.items():
                    lse[name].append(add(f'{prefix}.p{index}.{name}_read',view,Operation.READ,loops,mv,
                        (dep(producer,EventKind.WRITE_VISIBLE,bindings),dep(f'{prefix}.max_init',EventKind.WRITE_VISIBLE),
                            *(item for item in barrier if item.access_id!=producer))))
            maximum=scratch(f'{prefix}.maximum',(1,32,8));total=scratch(f'{prefix}.total',(1,32,8))
            for name,view,symbol,value in (('max',maximum,'-ConstInfo.FLOAT_MAX',None),('sum',total,'0.0',ConstantValue.ZERO)):
                add(f'{prefix}.{name}_init',view,Operation.WRITE,loops,mv,barrier,value=value,symbol=symbol)
                add(f'{prefix}.{name}_init_read',view,Operation.READ,loops,mv,(dep(f'{prefix}.{name}_init',EventKind.WRITE_VISIBLE),))
            add(f'{prefix}.max_write',maximum,Operation.WRITE,loops,mv,(dep(f'{prefix}.max_init_read'),*(dep(name) for name in lse['max'])))
            add(f'{prefix}.max_read',maximum,Operation.READ,loops,mv,(dep(f'{prefix}.max_write',EventKind.WRITE_VISIBLE),))
            weights=[]
            for index,sum_read,max_read in zip(indexes,lse['sum'],lse['max'],strict=True):
                weight=scratch(f'{prefix}.p{index}.weight',(1,32,8));name=f'{prefix}.p{index}.weight'
                add(f'{name}_write',weight,Operation.WRITE,loops,mv,(dep(sum_read),dep(max_read),dep(f'{prefix}.max_read')))
                read=add(f'{name}_read',weight,Operation.READ,loops,mv,(dep(f'{name}_write',EventKind.WRITE_VISIBLE),));weights.append(read)
            add(f'{prefix}.sum_write',total,Operation.WRITE,loops,mv,(dep(f'{prefix}.sum_init_read'),*(dep(name) for name in weights)))
            add(f'{prefix}.sum_read',total,Operation.READ,loops,mv,(dep(f'{prefix}.sum_write',EventKind.WRITE_VISIBLE),))
            accumulator=scratch(f'{prefix}.accumulator',(1,32,512));previous=None
            for index,weight,matrix in zip(indexes,weights,matrices,strict=True):
                name=f'{prefix}.p{index}';scale=scratch(f'{name}.scale',(1,32,8))
                add(f'{name}.scale_write',scale,Operation.WRITE,loops,mv,(dep(weight),dep(f'{prefix}.sum_read')))
                add(f'{name}.scale_read',scale,Operation.READ,loops,mv,(dep(f'{name}.scale_write',EventKind.WRITE_VISIBLE),))
                deps=(dep(matrix),dep(f'{name}.scale_read'))
                if previous is not None:
                    weighted=scratch(f'{name}.weighted',(1,32,512))
                    add(f'{name}.weighted_write',weighted,Operation.WRITE,loops,mv,deps)
                    add(f'{name}.weighted_read',weighted,Operation.READ,loops,mv,(dep(f'{name}.weighted_write',EventKind.WRITE_VISIBLE),))
                    add(f'{name}.acc_read',accumulator,Operation.READ,loops,mv,(dep(previous,EventKind.WRITE_VISIBLE),))
                    deps=(dep(f'{name}.weighted_read'),dep(f'{name}.acc_read'))
                previous=add(f'{name}.acc_write',accumulator,Operation.WRITE,loops,mv,deps)
            add(f'{prefix}.acc_read',accumulator,Operation.READ,loops,mv,(dep(previous,EventKind.WRITE_VISIBLE),))
            output=child(f'{prefix}.output',parents['output'],(1,32,512),
                (coord(0,((1,1),),offset),coord(1,(),request),coord(2),coord(3,((2,1),))))
            add(f'{prefix}.output',output,Operation.WRITE,loops,mv,(dep(f'{prefix}.acc_read'),))
    selected_axes={}
    for axis in axes:
        if axis.axis_id in selected_axes and selected_axes[axis.axis_id]!=axis:raise ValueError('CANN MLA shared head axis differs')
        selected_axes[axis.axis_id]=axis
    return tuple(allocations),tuple(views),tuple(events),(TileProgram(scope.scope_id,scope,tuple(selected_axes.values()),tuple(accesses),visibility,evidence),)


def split_demands(program,parents):
    header=program.scopes[0].op_ids;batch,kv=map(int,header[1:3]);views=unique_by(program.views,'view_id');totals={}
    family,=program.tile_programs;axes=unique_by(family.axes,'axis_id')
    for access in family.accesses:totals[access.access_id]=_tile_template_requested_bytes(views,axes,family,access)
    def amount(suffix):return sum(value for name,value in totals.items() if name.endswith(suffix))
    unique={role:prod(view.shape)*view.element_bytes for role,view in parents.items()}
    query=unique['query']+unique['query_rope'];key=unique['key']+unique['key_rope'];value=unique['key'];output=amount('.output')
    if (amount('.key')+amount('.key_rope'),amount('.value'),output)!=(key,value,unique['output']):raise ValueError('CANN MLA split payload partition differs')
    return {'query_unique_bytes':query,'key_interface_bytes':key,'value_interface_bytes':value,'kv_unique_bytes':None,
        'write_bytes':output,'priced_interface_bytes':query+key+value+output,'priced_score_elements':amount('.scores_write')//4,
        'priced_output_elements':output//2,'priced_task_query_l2_bytes':amount('.query')+amount('.query_rope'),
        'key_requested_bytes':key,'value_requested_bytes':value,'framework_clear_write_bytes':amount('.framework_clear'),
        'output_initialization_write_bytes':sum(value for name,value in totals.items() if name.startswith('init.aiv')),
        'totals':totals,'value_aliases_key':True,'native_page_addresses_bound':False,'split_head_count':sum(name.endswith('.aiv0.output') and name.startswith('fd') for name in totals),
        'requested_query_bytes':amount('.query')+amount('.query_rope'),'nupdate_rmw_bytes':amount('.nupdate_acc_write')}
