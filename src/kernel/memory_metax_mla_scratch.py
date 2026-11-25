"""Logical shared staging and register-only online carry in C500 MLA main.

Retained XCORE1000 bitcode uses the shared arena first for Q512, then
Q-RoPE64, panel K576/V512, and final BF16/FP32 output exchange. Register
phi carries are logical value storage, never priced as DRAM/shared traffic.
"""
from dataclasses import replace
from src.kernel.memory_access import (
    AddressLevel,ConstantValue,EventKind,LogicalCoordinateBinding,LogicalCoordinateTerm,
    LogicalSelectionView,MemoryAllocation,Operation,ProgramEvent,TensorView,
    TileAccessTemplate,TileDependency,TileIndexBinding,
)


def main_smem_prediction_bytes(work,query_bytes,output_bytes):
    """The existing Q512/Q64, K576/V512 and output-exchange projection."""
    from src.kernel.launch_config.tiling_spec import METAX_MLA_KV_BLOCK
    staged_tokens=work.kv_block_count*METAX_MLA_KV_BLOCK
    # Each Q and output crosses shared storage twice. A padded BF16 KV
    # panel is written once as K576, then read as K576 and leading V512.
    return 2*query_bytes+staged_tokens*(576+576+512)*2+2*output_bytes


def attach_main_scratch(header,scope,family,all_views):
    if header[1]=='combine':return _attach_combine_scratch(header,scope,family,all_views)
    if header[1]!='main':raise ValueError('Unknown MetaX MLA scratch stage')
    known={view.view_id:view for view in all_views};accesses={access.access_id:access for access in family.accesses}
    additions=[];views=[];allocations=[];events=[];axes={axis.axis_id:axis for axis in family.axes}
    evidence=('Retained C500 XCORE1000 BF16 MLA KV16 bitcode: shared Q512 staging is consumed before the same arena is reused for Q-RoPE64; K576 staging is read again as leading V512.',
        'KV-loop FP32 phi max/sum/PV accumulator values are explicit register-only carries, not invented GM buffers. Valid logical elements and masked KV tails only; native swizzle/issued padding and physical arena pool placement remain unresolved.',
        'Final BF16 direct output or FP32 split partial is exchanged through shared storage before its global store. Logical private versions do not imply simultaneous native arena residency.')
    def dep(name,kind=EventKind.READ_READY,bindings=()):return TileDependency(name,kind,(),bindings)
    def scratch(name,shape,dtype,width,domain,allocation=None):
        identity=f'{scope.scope_id}.{name}'
        if allocation is None:
            allocation=MemoryAllocation(identity,'logical-local-device',domain,None,AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
            allocations.append(allocation);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,allocation.allocation_id,None,None,()))
        view=LogicalSelectionView(identity,allocation.allocation_id,shape,dtype,width,f'metax_mla.{name}',None)
        views.append(view);return view,allocation
    def add(name,view,operation,loops,mapping,deps=(),value=None):
        source_scope=replace(scope,scope_id=f'{scope.scope_id}.{name}.scope',agent='logical-metax-mla-shared-state',op_ids=(*header,'shared_and_register_state'))
        additions.append(TileAccessTemplate(name,view.view_id,operation,loops,mapping,(),deps,evidence,scope=source_scope,unmapped=True,constant_value=value))
    payloads=[access for access in family.accesses if access.access_id.endswith('.query') and access.scope is not None]
    for query in payloads:
        prefix=query.access_id.rsplit('.',1)[0];head=f'{prefix}.head';panel=f'{prefix}.kv_panel'
        cache=accesses[f'{prefix}.cache'];length=axes[panel].extent;last=axes[panel].count-1
        qrefs=[];previous=None;arena=None
        for name,width in (('query_nope',512),('query_rope',64)):
            view,arena=scratch(f'{prefix}.{name}',(64,width),'bf16',2,'logical-shared-arena',arena)
            deps=(dep(query.access_id),) if previous is None else (dep(query.access_id),dep(previous))
            add(f'{prefix}.{name}_write',view,Operation.WRITE,(head,),(head,None),deps)
            previous=f'{prefix}.{name}_read';add(previous,view,Operation.READ,(head,),(head,None),(dep(f'{prefix}.{name}_write',EventKind.WRITE_VISIBLE),));qrefs.append(dep(previous))
        identity=f'{scope.scope_id}.{prefix}.cache_shared'
        allocation=MemoryAllocation(identity,'logical-local-device','logical-shared-versioned',None,AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        root=TensorView(identity,identity,0,(length,576),(1152,2),'bf16',2);views.append(root)
        mapping=tuple(LogicalCoordinateBinding(i,(LogicalCoordinateTerm(i,1),)) for i in range(2))
        staged=LogicalSelectionView(f'{identity}.K',identity,(length,576),'bf16',2,f'metax_mla.{prefix}.K_shared',root.view_id,mapping)
        value=LogicalSelectionView(f'{identity}.V',identity,(length,512),'bf16',2,f'metax_mla.{prefix}.V_shared',root.view_id,mapping)
        views.extend((staged,value))
        add(f'{prefix}.cache_shared_write',staged,Operation.WRITE,(panel,),(panel,None),
            (dep(cache.access_id),*(TileDependency(item.access_id,EventKind.READ_READY,(head,)) for item in qrefs)))
        add(f'{prefix}.key_shared_read',staged,Operation.READ,(panel,),(panel,None),(dep(f'{prefix}.cache_shared_write',EventKind.WRITE_VISIBLE),))
        add(f'{prefix}.value_shared_read',value,Operation.READ,(panel,),(panel,None),(dep(f'{prefix}.cache_shared_write',EventKind.WRITE_VISIBLE),))
        states={}
        for state,shape,constant in (('max',(64,),ConstantValue.NEGATIVE_INFINITY),('sum',(64,),ConstantValue.ZERO),('pv',(64,512),ConstantValue.ZERO)):
            view,_=scratch(f'{prefix}.register_{state}',shape,'fp32',4,'logical-register-state');states[state]=view
            rv=(head,) if len(shape)==1 else (head,None)
            add(f'{prefix}.{state}_init',view,Operation.WRITE,(head,),rv,value=constant)
            add(f'{prefix}.{state}_previous',view,Operation.READ,(head,panel),rv,
                (TileDependency(f'{prefix}.{state}_write',EventKind.WRITE_VISIBLE,(),carry_axis=panel,initial_access_id=f'{prefix}.{state}_init'),))
        base=(*qrefs,dep(f'{prefix}.key_shared_read'),dep(f'{prefix}.max_previous'),dep(f'{prefix}.sum_previous'))
        add(f'{prefix}.max_write',states['max'],Operation.WRITE,(head,panel),(head,),base)
        add(f'{prefix}.sum_write',states['sum'],Operation.WRITE,(head,panel),(head,),(*base,dep(f'{prefix}.max_write',EventKind.WRITE_VISIBLE)))
        add(f'{prefix}.pv_write',states['pv'],Operation.WRITE,(head,panel),(head,None),
            (*base,dep(f'{prefix}.max_write',EventKind.WRITE_VISIBLE),dep(f'{prefix}.sum_write',EventKind.WRITE_VISIBLE),dep(f'{prefix}.pv_previous'),dep(f'{prefix}.value_shared_read')))
        final={}
        for state,view in states.items():
            name=f'{prefix}.{state}_final';rv=(head,) if len(view.shape)==1 else (head,None)
            add(name,view,Operation.READ,(head,),rv,(dep(f'{prefix}.{state}_write',EventKind.WRITE_VISIBLE,(TileIndexBinding(panel,(),last),)),));final[state]=dep(name)
        for role in ('output','partial_output','lse','partial_lse'):
            name=f'{prefix}.{role}'
            if name not in accesses:continue
            output=accesses[name]
            if role in ('output','partial_output'):
                parent=known[output.view_id];exchange,_=scratch(f'{prefix}.output_exchange',parent.shape,parent.dtype,parent.element_bytes,'logical-shared-output-exchange')
                add(f'{prefix}.output_shared_write',exchange,Operation.WRITE,(head,),(head,None),(final['pv'],final['sum']))
                add(f'{prefix}.output_shared_read',exchange,Operation.READ,(head,),(head,None),(dep(f'{prefix}.output_shared_write',EventKind.WRITE_VISIBLE),))
                accesses[name]=replace(output,dependencies=(*output.dependencies,dep(f'{prefix}.output_shared_read')))
            else:accesses[name]=replace(output,dependencies=(*output.dependencies,final['max'],final['sum']))
    return tuple(allocations),tuple(views),tuple(events),replace(family,accesses=(*accesses.values(),*additions))


def _attach_combine_scratch(header,scope,family,all_views):
    """Actual sLSE overwrite and register weighted reduction per head tile."""
    known={view.view_id:view for view in all_views};accesses={access.access_id:access for access in family.accesses}
    additions=[];allocations=[];views=[];events=[];axes={axis.axis_id:axis for axis in family.axes}
    evidence=('C500 XCORE1000 combine bitcode uses static sLSE[kMaxSplits][blockM+1]: each active split/head LSE is staged, then overwritten by exp(partial_lse-total_lse) and reread by the partial-O FMA loop.',
        'Native head-block selection is retained; scalar/pool addresses and the unused bank-padding column are not inferred. Register-only total-LSE and weighted-PV accumulator are logical values, not GM/shared price terms.')
    def dep(name,kind=EventKind.READ_READY,reduction=(),bindings=()):return TileDependency(name,kind,reduction,bindings)
    def scratch(name,shape,domain):
        identity=f'{scope.scope_id}.{name}';allocation=MemoryAllocation(identity,'logical-local-device',domain,None,AddressLevel.OBJECT,None,None,f'{identity}.allocated',None)
        allocations.append(allocation);events.append(ProgramEvent(allocation.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        view=LogicalSelectionView(identity,identity,shape,'fp32',4,f'metax_mla.{name}',None);views.append(view);return view
    def add(name,view,operation,loops,mapping,deps=(),value=None):
        source_scope=replace(scope,scope_id=f'{scope.scope_id}.{name}.scope',op_ids=(*header,'shared_and_register_state'),agent='logical-metax-mla-shared-state')
        additions.append(TileAccessTemplate(name,view.view_id,operation,loops,mapping,(),deps,evidence,scope=source_scope,unmapped=True,constant_value=value))
    for load in family.accesses:
        if not load.access_id.endswith('.partial_lse'):continue
        prefix=load.access_id.rsplit('.',1)[0];head=f'{prefix}.head';split=f'{prefix}.split';count=axes[split].extent
        shared=scratch(f'{prefix}.sLSE',(count,64),'logical-shared-combine-state');loops=(split,head)
        add(f'{prefix}.sLSE_write',shared,Operation.WRITE,loops,loops,(dep(load.access_id),))
        add(f'{prefix}.sLSE_read',shared,Operation.READ,loops,loops,(dep(f'{prefix}.sLSE_write',EventKind.WRITE_VISIBLE),))
        total=scratch(f'{prefix}.total_lse',(64,),'logical-register-state')
        add(f'{prefix}.total_lse_write',total,Operation.WRITE,(head,),(head,),(dep(f'{prefix}.sLSE_read',reduction=(split,)),))
        add(f'{prefix}.total_lse_read',total,Operation.READ,(head,),(head,),(dep(f'{prefix}.total_lse_write',EventKind.WRITE_VISIBLE),))
        add(f'{prefix}.weight_write',shared,Operation.WRITE,loops,loops,(dep(f'{prefix}.sLSE_read'),dep(f'{prefix}.total_lse_read')))
        add(f'{prefix}.weight_read',shared,Operation.READ,loops,loops,(dep(f'{prefix}.weight_write',EventKind.WRITE_VISIBLE),))
        accumulator=scratch(f'{prefix}.weighted_pv',(64,512),'logical-register-state')
        add(f'{prefix}.acc_init',accumulator,Operation.WRITE,(head,),(head,None),value=ConstantValue.ZERO)
        add(f'{prefix}.acc_previous',accumulator,Operation.READ,(head,split),(head,None),
            (TileDependency(f'{prefix}.acc_write',EventKind.WRITE_VISIBLE,(),carry_axis=split,initial_access_id=f'{prefix}.acc_init'),))
        add(f'{prefix}.acc_write',accumulator,Operation.WRITE,(head,split),(head,None),
            (dep(f'{prefix}.partial_output'),dep(f'{prefix}.weight_read'),dep(f'{prefix}.acc_previous')))
        add(f'{prefix}.acc_final',accumulator,Operation.READ,(head,),(head,None),
            (dep(f'{prefix}.acc_write',EventKind.WRITE_VISIBLE,bindings=(TileIndexBinding(split,(),count-1),)),))
        for role,read in (('output','acc_final'),('lse','total_lse_read')):
            name=f'{prefix}.{role}';access=accesses[name]
            accesses[name]=replace(access,dependencies=(*access.dependencies,dep(f'{prefix}.{read}')))
    return tuple(allocations),tuple(views),tuple(events),replace(family,accesses=(*accesses.values(),*additions))
