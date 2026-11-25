"""C500 XCORE1000 dense-length metadata shared arena and cursor dataflow.

Request membership remains symbolic: the metadata KernelSpec retains only
aggregate scheduler counts, not the request lengths. No request order/indices
are recovered from those aggregates. Private SSA versions are logical register
values; their requests never enter retained GM/shared service prices.
"""
from dataclasses import replace

from src.kernel.memory_access import (
    AddressLevel,ConstantValue,Coverage,EventKind,Granularity,LogicalSelectionView,
    MemoryAccessProgram,MemoryAllocation,Operation,Ordering,ProgramEvent,Semantics,
    TensorView,TileAccessTemplate,TileAxis,TileDependency,TileIndexBinding,TileIndexTerm,
    TilePartition,TileProgram,unique_by,_tile_template_requested_bytes,
)


def _geometry(kernel):
    if (kernel.hardware_id!='metax_c500' or kernel.variant!='flash_mla_metadata' or kernel.dtype!='int32'
            or kernel.launch.algorithm!='observed_flash_mla_metadata:flash_mla_metadata'
            or kernel.launch.grid!=(1,1,1) or kernel.launch.block!=(32,1,1)):
        raise ValueError('C500 metadata requires its actual XCORE1000 dense-length source ABI')
    s=kernel.shape
    g=tuple(s[name] for name in ('batch','scheduler_partitions','scheduler_segment_count','scheduler_worker_run_count','kv_block_count'))
    _validate_geometry(g)
    if s['scheduler_metadata_ints']!=8*g[1] or s['split_metadata_ints']!=g[0]+1:
        raise ValueError('C500 metadata storage disagrees with its scheduler counts')
    return g


def _validate_geometry(g):
    b,p,s,r,blocks=g
    if (min(g)<=0 or blocks<b or not 1<=r<=min(p,s) or s<b or s> b+r-1
            or not 0<=p+b-s<=p):
        raise ValueError('C500 metadata requires consistent positive dense scheduler geometry')


def _components(g,scope,parents,ready,visibility,evidence,device,binding_id):
    b,p,s,r,_=g;v=b+r-1;l=p+b-s
    allocations=[];events=[];views=[];accesses=[];axes=[];partitions=[]
    known=dict(parents)
    def axis(name,count):
        if count<=0:raise ValueError('Empty metadata subsets must be omitted')
        axes.append(TileAxis(name,count,1));return name
    for name,count in (('request',b),('partition',p),('visit',v),('active',r)):
        axis(name,count)
    def view(name,allocation,offset,shape,strides):
        result=TensorView(f'{scope.scope_id}.{name}',allocation,offset,shape,strides,'int32',4)
        views.append(result);known[result.view_id]=result;return result
    def arena(name,words,domain):
        identity=f'{binding_id}.{scope.scope_id}.{name}'
        result=MemoryAllocation(identity,device,domain,words*4,
            AddressLevel.OBJECT,None,None,identity+'.allocated',None)
        allocations.append(result);events.append(ProgramEvent(result.allocated_event,EventKind.ALLOCATED,identity,None,None,()))
        return identity
    def private(name,words):
        identity=arena(name,words,'logical-register-state')
        return view(name,identity,0,(words,),(4,))
    def dep(name,kind=EventKind.READ_READY,reduction=(),bindings=(),member=None,carry=None,initial=None):
        return TileDependency(name,kind,reduction,bindings,carry,initial,member)
    def add(name,selected,operation,loops,mapping,deps=(),*,input_ready=(),constant=None,unmapped=False,domain='shared'):
        participant=replace(scope,scope_id=f'{scope.scope_id}.{name}.scope',agent=f'logical-metax-metadata-{domain}')
        accesses.append(TileAccessTemplate(name,selected.view_id,operation,loops,mapping,input_ready,deps,evidence,
            unmapped=unmapped,scope=participant,constant_value=constant))
    def relation(member,owner):
        partitions.append(TilePartition(member,None,owner))
    def selected(name,parent,member,owner):
        count=next(a.extent for a in axes if a.axis_id==member)
        result=LogicalSelectionView(f'{scope.scope_id}.{name}',parent.allocation_id,(count,),'int32',4,
            f'metax_metadata.{name}',parent.view_id)
        views.append(result);known[result.view_id]=result
        relation(member,owner);return result
    def select_read(name,parent,writer,member,owner,extras=()):
        target=selected(name,parent,member,owner)
        add(name,target,Operation.READ,(member,),(member,),
            (dep(writer,EventKind.WRITE_VISIBLE,member=member),*extras),unmapped=True)
    def class_input(name,kind,member):return dep(name,kind,member=member)
    def class_reduce(name,kind,member):return dep(name,kind,(member,),member=member)
    def fixed(name,index):return (TileIndexBinding(name,(),index),)
    def mapped(name,source,offset=0):return (TileIndexBinding(name,(TileIndexTerm(source,1),),offset),)

    shared=arena('shared',5*b+1,'logical-shared-arena')
    blocks=view('blocks',shared,0,(b,),(4,))
    prefix=view('prefix',shared,4*b,(b+1,),(4,))
    lengths=view('lengths',shared,4*(2*b+1),(b,),(4,))
    begin=view('begin',shared,4*(3*b+1),(b,),(4,))
    end=view('end',shared,4*(4*b+1),(b,),(4,))
    prefix_zero=view('prefix_zero',shared,prefix.offset_bytes,(1,),(4,))
    prefix_tail=view('prefix_tail',shared,prefix.offset_bytes+4,(b,),(4,))
    end_zero=view('begin_B_alias_end_0',shared,end.offset_bytes,(1,),(4,))
    output_prefix=parents['output1']
    output_zero=view('output_prefix_zero',output_prefix.allocation_id,output_prefix.offset_bytes,(1,),(4,))
    output_tail=view('output_prefix_tail',output_prefix.allocation_id,output_prefix.offset_bytes+4,(b,),(4,))
    scheduler=parents['output0']
    fields=view('scheduler_fields',scheduler.allocation_id,scheduler.offset_bytes,(p,5),scheduler.strides_bytes)
    add('length',parents['input0'],Operation.READ,('request',),('request',),input_ready=(ready,),domain='global')
    add('length_init',lengths,Operation.WRITE,('request',),('request',),(dep('length'),))
    add('blocks_init',blocks,Operation.WRITE,('request',),('request',),(dep('length'),))
    add('begin_init',begin,Operation.WRITE,('request',),('request',),constant=ConstantValue.ZERO)
    add('end_init',end,Operation.WRITE,('request',),('request',),(dep('length'),))
    add('prefix_zero_init',prefix_zero,Operation.WRITE,(),(None,),constant=ConstantValue.ZERO)

    budget=private('budget',1)
    add('budget_init',budget,Operation.WRITE,(),(None,),(dep('length',reduction=('request',)),),domain='register')
    add('budget',budget,Operation.READ,('visit',),(None,),(dep('budget_init',EventKind.WRITE_VISIBLE),),domain='register')
    partition_state=private('partition_state',4)
    add('partition_state_init',partition_state,Operation.WRITE,(),(None,),constant=ConstantValue.ZERO,domain='register')
    add('partition_start',partition_state,Operation.READ,('partition',),(None,),(
        dep('partition_state_update',EventKind.WRITE_VISIBLE,carry='partition',initial='partition_state_init'),),domain='register')
    cursor=private('visit_cursor',4)
    add('cursor_init',cursor,Operation.WRITE,(),(None,),constant=ConstantValue.ZERO,domain='register')
    add('cursor',cursor,Operation.READ,('visit',),(None,),(
        dep('cursor_update',EventKind.WRITE_VISIBLE,carry='visit',initial='cursor_init'),),domain='register')
    select_read('blocks_read',blocks,'blocks_init','visit','request',(
        dep('cursor'),dep('budget'),class_input('partition_start',EventKind.READ_READY,'visit')))
    relation('visit','partition')
    add('cursor_update',cursor,Operation.WRITE,('visit',),(None,),(
        dep('cursor'),dep('blocks_read'),dep('budget'),class_input('partition_start',EventKind.READ_READY,'visit')),domain='register')
    add('partition_state_update',partition_state,Operation.WRITE,('partition',),(None,),(
        dep('partition_start'),class_reduce('cursor_update',EventKind.WRITE_VISIBLE,'visit')),domain='register')
    select_read('begin_read',begin,'begin_init','active','request',(
        dep('partition_start',bindings=mapped('partition','active')),))
    relation('active','partition')
    starts=(class_reduce('begin_read',EventKind.READ_READY,'active'),)
    if p>r:
        axis('exhausted',p-r)
        add('begin_B_read',end_zero,Operation.READ,('exhausted',),(None,),(
            dep('end_init',EventKind.WRITE_VISIBLE,bindings=fixed('request',0)),
            dep('partition_start',bindings=mapped('partition','exhausted',r))))
        relation('exhausted','partition')
        starts=(*starts,class_reduce('begin_B_read',EventKind.READ_READY,'exhausted'))

    boundaries=[]
    if l:
        axis('zero_offset',l)
        select_read('previous_length',lengths,'length_init','zero_offset','request',(
            class_input('partition_state_update',EventKind.WRITE_VISIBLE,'zero_offset'),))
        relation('zero_offset','partition')
        # Same request index for lengths[r-1] and end[r-1]. Reuse one map.
        target=LogicalSelectionView(f'{scope.scope_id}.end_read',shared,(l,),'int32',4,'metax_metadata.end_read',end.view_id)
        views.append(target);known[target.view_id]=target
        add('end_read',target,Operation.READ,('zero_offset',),('zero_offset',),(
            class_input('end_init',EventKind.WRITE_VISIBLE,'zero_offset'),dep('previous_length')),unmapped=True)
        boundaries.extend((class_reduce('previous_length',EventKind.READ_READY,'zero_offset'),class_reduce('end_read',EventKind.READ_READY,'zero_offset')))
    if p>l:
        axis('partial_offset',p-l)
        select_read('partial_begin_read',begin,'begin_init','partial_offset','request',(
            class_input('partition_state_update',EventKind.WRITE_VISIBLE,'partial_offset'),))
        relation('partial_offset','partition')
        boundaries.append(class_reduce('partial_begin_read',EventKind.READ_READY,'partial_offset'))
    add('scheduler',fields,Operation.WRITE,('partition',),('partition',None),(
        dep('partition_start'),dep('partition_state_update',EventKind.WRITE_VISIBLE),*starts,*boundaries),domain='global')

    total=private('completed_request_prefix',1)
    add('total_init',total,Operation.WRITE,(),(None,),constant=ConstantValue.ZERO,domain='register')
    add('total_previous',total,Operation.READ,('request',),(None,),(
        dep('total_update',EventKind.WRITE_VISIBLE,carry='request',initial='total_init'),),domain='register')
    add('total_update',total,Operation.WRITE,('request',),(None,),(
        dep('total_previous'),class_reduce('cursor_update',EventKind.WRITE_VISIBLE,'visit')),domain='register')
    add('prefix_write',prefix_tail,Operation.WRITE,('request',),('request',),(dep('total_update',EventKind.WRITE_VISIBLE),))
    # The real final warp synchronization precedes all prefix copy loads.
    joins=(dep('prefix_zero_init',EventKind.WRITE_VISIBLE),dep('prefix_write',EventKind.WRITE_VISIBLE,('request',)))
    add('prefix_zero_read',prefix_zero,Operation.READ,(),(None,),joins)
    add('prefix_read',prefix_tail,Operation.READ,('request',),('request',),joins)
    add('split_zero',output_zero,Operation.WRITE,(),(None,),(dep('prefix_zero_read'),),domain='global')
    add('split',output_tail,Operation.WRITE,('request',),('request',),(dep('prefix_read'),),domain='global')
    # Repeated references to the same relation are one semantic map.
    unique=[]
    for partition in partitions:
        if partition not in unique:unique.append(partition)
    family=TileProgram(scope.scope_id,scope,tuple(axes),tuple(accesses),visibility,evidence,partitions=tuple(unique))
    return tuple(allocations),tuple(views),tuple(events),family


def build_metax_metadata_source(kernel,bindings,scope,evidence,*,program_id):
    g=_geometry(kernel)
    scope=replace(scope,scope_id=program_id,agent='logical-metax-mla-metadata',
        op_ids=('flash_mla_metadata','metax_native',*(str(value) for value in g)))
    evidence=(*evidence,'C500 retained XCORE1000 metadata IR: five INT32 arrays share one 5B+1-word arena; begin[B] aliases end[0].',
        'Thread0 cursor/partition phi values retain only source-required carry; request/visit/partition membership is unbound, not reconstructed from aggregate counts.',
        'Shared prefix reads follow the actual final warp join. Native parameters, instructions, physical addresses and register allocation are not simulated.')
    parents=unique_by(bindings.views,'view_id');bound_allocations=unique_by(bindings.allocations,'allocation_id')
    device=bound_allocations[parents['input0'].allocation_id].device_id
    allocations,views,events,family=_components(g,scope,parents,bindings.input_ready[0],bindings.visibility_domain,evidence,device,bindings.binding_id)
    program=MemoryAccessProgram('midlayer.memory-access.v1',program_id,kernel.hardware_id,bindings.binding_id,
        Coverage(Granularity.TILE,Semantics.LOGICAL,Ordering.PARTIAL,False,(
            'Dynamic positive-length scheduler membership is symbolic; explicit DAG expansion requires values.',
            'Logical register SSA is not native traffic. Existing GM-only pricing and eight-field scheduler price remain explicit.')),
        (*bindings.allocations,*allocations),(*bindings.views,*views),(),(*bindings.events,*events),(),(family,))
    metax_metadata_source_demands(program);return program


def metax_metadata_source_demands(program):
    program.__post_init__()
    if program.hardware_id!='metax_c500' or program.accesses or program.scopes or len(program.tile_programs)!=1:
        raise ValueError('C500 metadata requires its unique compressed INT32 tile source')
    family,=program.tile_programs;header=family.scope.op_ids
    if len(header)!=7 or header[:2]!=('flash_mla_metadata','metax_native'):
        raise ValueError('C500 metadata requires its actual shared/cursor source')
    g=tuple(int(value) for value in header[2:]);_validate_geometry(g);b,p,s,r,_=g
    known=unique_by(program.views,'view_id');parents={role:known[role] for role in ('input0','output0','output1')}
    for role,shape,strides in (('input0',(b,),(4,)),('output0',(p,8),(32,4)),('output1',(b+1,),(4,))):
        parent=parents[role]
        if not isinstance(parent,TensorView) or (parent.shape,parent.strides_bytes,parent.dtype,parent.element_bytes)!=(shape,strides,'int32',4):
            raise ValueError('C500 metadata requires its complete INT32 tensor ABI')
    if len({parent.allocation_id for parent in parents.values()})!=3:raise ValueError('C500 metadata public parents must be distinct')
    accesses=unique_by(family.accesses,'access_id')
    if 'length' not in accesses or len(accesses['length'].input_ready)!=1:raise ValueError('C500 metadata requires actual sequence-length READ_READY')
    all_allocations=unique_by(program.allocations,'allocation_id');all_events=unique_by(program.events,'event_id')
    device=all_allocations[parents['input0'].allocation_id].device_id
    allocations,views,events,expected=_components(g,family.scope,parents,accesses['length'].input_ready[0],family.visibility_domain,family.evidence_refs,device,program.binding_id)
    if family!=expected or set(known)!=set(parents)|{view.view_id for view in views} or any(known[view.view_id]!=view for view in views):
        raise ValueError('C500 metadata lost actual sequence-length READ_READY, shared alias or cursor/partition dependency')
    if any(all_allocations[item.allocation_id]!=item for item in allocations) or any(all_events[item.event_id]!=item for item in events):
        raise ValueError('C500 metadata internal storage/initialization changed')
    axes=unique_by(family.axes,'axis_id')
    def count(operation,domain):
        return sum(_tile_template_requested_bytes(known,axes,family,a) for a in family.accesses
            if a.operation is operation and a.scope.agent==f'logical-metax-metadata-{domain}')
    return {'batch':b,'partitions':p,'read_bytes':count(Operation.READ,'global'),
        'write_bytes':count(Operation.WRITE,'global'),'priced_write_bytes':p*8*4+(b+1)*4,
        'scheduler_written_fields':5,'scheduler_storage_fields':8,
        'shared_read_bytes':count(Operation.READ,'shared'),'shared_write_bytes':count(Operation.WRITE,'shared'),
        'register_read_bytes':count(Operation.READ,'register'),'register_write_bytes':count(Operation.WRITE,'register')}
