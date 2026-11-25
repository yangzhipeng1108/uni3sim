"""C500 XCORE1000 FlashMLA control reads from retained device bitcode.

The source manifest owns binary/bitcode identity. Page values and native
addresses remain unresolved; exact worker/request/KV-panel ownership does not.
"""
from dataclasses import replace

from src.kernel.memory_access import (
    EventKind,LogicalCoordinateBinding,LogicalCoordinateTerm,LogicalSelectionView,
    Operation,TileAccessTemplate,TileAxis,TileDependency,TileIndexBinding,TileProgram,
)


def control_layouts(kv_lens,stage):
    batch=len(kv_lens)
    if stage=='combine':return (('splits',(batch+1,),'int32',4),)
    if stage!='main':raise ValueError('Unknown MetaX MLA control stage')
    return (('lengths',(batch,),'int32',4),('scheduler',(104,8),'int32',4),
        ('splits',(batch+1,),'int32',4),('block_table',(batch,-(-max(kv_lens)//16)),'int32',4))


def attach_controls(header,parents,control_ready,work,scope,visibility,payload_families):
    """Join exact control readers and payloads in one dependency namespace."""
    stage=header[1];kv_lens=tuple(map(int,header[2].split(',')))
    layouts=control_layouts(kv_lens,stage)
    if len(control_ready)!=len(layouts):raise ValueError('MetaX MLA actual controls are absent')
    ready=dict(zip((role for role,*_ in layouts),control_ready,strict=True));views=[];axes=[];accesses=[]
    evidence=('Retained MACA3.7.2 FlashMLA XCORE1000 BF16 head64/KV16 bitcode, source_manifest.json under memory_unification_mla_ir_20261017.',
        'Main params+456: all104 workers read scheduler fields0..3; only valid workers read field4. Each real request segment reads its INT32 length and adjacent split prefixes.',
        'Main params+352 is the page16 block-table interface; logical panel lookup controls that same panel cache read. Page values and physical native issue multiplicity remain unresolved.',
        'Combine params+472 reads adjacent INT32 split prefixes before deciding whether the request is split; direct requests produce no combine payload.',
        'These control reads do not alter the retained payload/cache/SMEM/DRAM price projections.')
    def coord(axis,terms=(),offset=0):return LogicalCoordinateBinding(axis,tuple(LogicalCoordinateTerm(a,c) for a,c in terms),offset)
    def child(name,role,shape,mapping):
        parent=parents[role];view=LogicalSelectionView(f'{scope.scope_id}.{name}',parent.allocation_id,shape,parent.dtype,parent.element_bytes,
            f'metax_mla.{name}',parent.view_id,mapping);views.append(view);return view
    def add(name,view,loops,mapping,deps=(),inputs=(),divisors=()):
        accesses.append(TileAccessTemplate(name,view.view_id,Operation.READ,loops,mapping,inputs,deps,evidence,unmapped=True,view_axis_divisors=divisors))
    def dep(name,bindings=()):return TileDependency(name,EventKind.READ_READY,(),bindings)
    if stage=='main':
        axes.append(TileAxis('partition',104,1))
        selected=child('scheduler_first4','scheduler',(104,4),(coord(0,((0,1),)),coord(1,((1,1),))))
        add('scheduler_first4',selected,('partition',),('partition',None),inputs=(ready['scheduler'],))
        active=tuple(index for index,segments in enumerate(work.segments_by_partition) if segments)
        if active!=tuple(range(len(active))):raise ValueError('MetaX MLA active workers are not the actual scheduler prefix')
        axes.append(TileAxis('active_partition',len(active),1))
        selected=child('scheduler_fifth','scheduler',(len(active),),(coord(0,((0,1),)),coord(1,(),4)))
        add('scheduler_fifth',selected,('active_partition',),('active_partition',),inputs=(ready['scheduler'],))
    for index,family in enumerate(payload_families):
        prefix=f'p{index}';names={access.access_id:f'{prefix}.{access.access_id}' for access in family.accesses}
        axes_by_name={axis.axis_id:f'{prefix}.{axis.axis_id}' for axis in family.axes}
        axes.extend(replace(axis,axis_id=axes_by_name[axis.axis_id]) for axis in family.axes)
        request=int(family.scope.op_ids[5] if stage=='main' else family.scope.op_ids[4])
        local=[]
        if stage=='main':
            worker=int(family.scope.op_ids[4]);first=int(family.scope.op_ids[7]);blocks=int(family.scope.op_ids[8])
            local.extend((dep('scheduler_first4',(TileIndexBinding('partition',(),worker),)),
                dep('scheduler_fifth',(TileIndexBinding('active_partition',(),worker),))))
            lengths=child(f'{prefix}.lengths','lengths',(1,),(coord(0,(),request),))
            add(f'{prefix}.lengths',lengths,(),(None,),tuple(local),(ready['lengths'],));local.append(dep(f'{prefix}.lengths'))
        else:
            # Actual reducer CTAs also inspect direct-request prefixes. All
            # requests are handled below, including those with no payload.
            worker=None
        splits=child(f'{prefix}.splits','splits',(2,),(coord(0,((0,1),),request),))
        head=axes_by_name['head']
        split_loops=() if stage=='main' else (head,)
        add(f'{prefix}.splits',splits,split_loops,(None,),tuple(local),(ready['splits'],));local.append(dep(f'{prefix}.splits'))
        if stage=='main':
            table=child(f'{prefix}.block_table','block_table',(blocks,),
                (coord(0,(),request),coord(1,((0,1),),first)))
            panel=axes_by_name['kv_panel']
            add(f'{prefix}.block_table',table,(panel,),(panel,),tuple(local),(ready['block_table'],),divisors=(16,))
        for access in family.accesses:
            dependencies=tuple(replace(item,access_id=names[item.access_id],
                reduction_axes=tuple(axes_by_name[axis] for axis in item.reduction_axes)) for item in access.dependencies)
            dependencies+=tuple(local)
            if stage=='main' and access.access_id=='cache':dependencies+=(dep(f'{prefix}.block_table'),)
            accesses.append(replace(access,access_id=names[access.access_id],
                iteration_axes=tuple(axes_by_name[axis] for axis in access.iteration_axes),
                view_axes=tuple(axes_by_name[axis] if axis is not None else None for axis in access.view_axes),dependencies=dependencies,scope=family.scope))
    if stage=='combine':
        from src.kernel.launch_config.tiling_spec import metax_mla_combine_block_m
        block=metax_mla_combine_block_m(len(kv_lens),1,64)
        for request,count in enumerate(work.splits_per_request):
            if count!=1:continue
            axis=f'direct{request}.head';axes.append(TileAxis(axis,64,block))
            view=child(f'direct{request}.splits','splits',(2,),(coord(0,((0,1),),request),))
            add(f'direct{request}.splits',view,(axis,),(None,),inputs=(ready['splits'],))
    if not axes:raise ValueError('MetaX MLA control domain omitted all actual workers')
    family=TileProgram(scope.scope_id,scope,tuple(axes),tuple(accesses),visibility,evidence)
    return tuple(views),(family,)
