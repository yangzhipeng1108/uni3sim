"""Static HMMA operand shared-memory byte facts, never timing values."""
import re

from dataset_pipeline.kernel_dataset.nvidia_h200_startup_issue import _copy_free_callee_work


def analyze_async_group_window(instructions, *, loop_begin, loop_end,
                               tile_m, tile_n, tile_k, threads, buffer_slots,
                               prefetch_panels=None):
    """Close committed A/B group work, without assigning completion times.

    Every path reaching the first K-loop wait must have the same committed
    copy trace. Copy-free scalar calls are summarized by the existing owner.
    PTX/compiler evidence must separately bind the meaning of DEPBAR/commit;
    this routine proves source counts, not undocumented SASS timing, absolute
    shared-slot aliasing, or interwarp readiness.
    """
    geometry=(tile_m,tile_n,tile_k,threads,buffer_slots)
    if (any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in geometry)
            or threads%32 or buffer_slots>4):
        raise ValueError('positive physical geometry and1..4operand slots required')
    asynchronous_matrix = prefetch_panels is not None
    panels = buffer_slots if prefetch_panels is None else prefetch_panels
    if isinstance(panels,bool) or not isinstance(panels,int) or not 1<=panels<=buffer_slots:
        raise ValueError('prefetch panels must fit physical operand slots')
    ins=[(int(r['address'],16),' '.join(r['instruction'].split())) for r in instructions]
    if not ins or any(b[0]<=a[0] for a,b in zip(ins,ins[1:])):
        raise ValueError('ordered unique source addresses required')
    by={a:i for i,(a,_) in enumerate(ins)}
    lo,hi=int(loop_begin,16),int(loop_end,16)
    if lo not in by or hi not in by or lo>hi:
        raise ValueError('closed source interval required')
    begin,end=by[lo],by[hi];body=ins[begin:end+1]
    expected_wait=f'DEPBAR.LE SB0, {hex(2*(panels-1))}'
    waits=[i for i,(_,t) in enumerate(body) if t==expected_wait]
    if len(waits)!=1 or re.fullmatch(
            r'(?:@!?P[0-9]+ )?BRA '+re.escape(hex(lo)),body[-1][1]) is None:
        raise ValueError('source loop requires one panel wait and closed backedge')
    wait,=waits
    if any(re.fullmatch(r'UIADD3 UR\d+, UR\d+, 0x1, URZ',t) is None for _,t in body[:wait]):
        raise ValueError('unsupported work before panel wait')
    matrix_proof = None
    if asynchronous_matrix:
        # Reuse the existing complete-K matrix owner. Only the relative refill
        # edge is new here; no operand decoding or instruction timing is added.
        from dataset_pipeline.kernel_dataset.nvidia_h200_matrix_preparation import analyze_projection
        matrix_proof=analyze_projection('\n'.join(
            r['address']+' '+r['instruction'] for r in instructions),
            tile_m=tile_m,tile_n=tile_n,tile_k=tile_k,threads=threads)
        if matrix_proof['loop_begin']!=hex(lo) or matrix_proof['loop_end']!=hex(hi):
            raise ValueError('matrix and copy loop intervals disagree')
        matrix_waits=[i for i,(_,t) in enumerate(body) if 'WARPGROUP.DEPBAR' in t]
        refills=[i for i,(_,t) in enumerate(body) if 'LDGSTS.' in t and not t.startswith('@!PT ')]
        if len(matrix_waits)!=1 or not refills or matrix_waits[0]>=min(refills):
            raise ValueError('post-issue matrix wait must precede every refill copy')
    def advance(state,address,text):
        if text.startswith('@!PT '):return state
        groups,pending=state
        if 'LDGSTS.' in text:
            if not text.startswith('LDGSTS.E.BYPASS.128 '):
                raise ValueError('unclosed asynchronous copy or predication')
            return groups,pending+(hex(address),)
        if 'LDGDEPBAR' in text:
            if text!='LDGDEPBAR' or not pending:
                raise ValueError('unclosed or empty commit group')
            return groups+(pending,),()
        if asynchronous_matrix and re.fullmatch(r'WARPGROUP\.DEPBAR\.LE gsb0, 0x[0-7]',text):
            return state
        if 'DEPBAR.' in text:
            raise ValueError('additional wait inside copy trace')
        return state
    # Enumerate CFG states only up to first entry; scalar paths converge to
    # identical traces. A copy-bearing cycle or skipped commit fails closed.
    todo=[0];successors={}
    while todo:
        i=todo.pop()
        if i==begin:
            successors[i]=();continue
        if i>=len(ins):raise ValueError('source falls outside export')
        if begin<i<=end:raise ValueError('prologue enters the interior of the K loop')
        if i in successors:continue
        address,text=ins[i]
        if text.startswith('@!PT '):next_nodes=(i+1,)
        else:
            pred=text.startswith('@');tokens=text.split();op=tokens[1] if pred else tokens[0]
            if op=='BRA':
                target=int(tokens[-1],16)
                if target not in by:raise ValueError('missing source branch target')
                next_nodes=(by[target],i+1) if pred else (by[target],)
            elif op=='EXIT':next_nodes=(i+1,) if pred else ()
            elif op=='CALL.REL.NOINC':
                returning=re.fullmatch(r'MOV (R\d+), (0x[0-9a-f]+)',ins[i-1][1]) if i else None
                target=int(tokens[-1],16)
                if (pred or returning is None or target not in by or i+1>=len(ins)
                        or ins[0][0]+int(returning[2],16)!=ins[i+1][0]):
                    raise ValueError('unclosed group prologue call')
                summary=_copy_free_callee_work(ins,by[target],returning[1],ins[0][0])
                if summary is None:raise ValueError('unsupported group prologue callee')
                last=by[int(summary['return_address'],16)]
                if any('DEPBAR' in t for _,t in ins[by[target]:last+1]):
                    raise ValueError('callee changes async group state')
                next_nodes=(i+1,)
            elif op.startswith(('CALL','RET','BRX','JMP','JMX','BSSY','BSYNC','BREAK','BRK','SSY','SYNC','PRET')):
                raise ValueError('unsupported source control flow')
            else:next_nodes=(i+1,)
        successors[i]=next_nodes;todo.extend(next_nodes)
    predecessors={i:set() for i in successors}
    for i,nodes in successors.items():
        for n in nodes:predecessors[n].add(i)
    ancestors=set();todo=[begin]
    if begin not in predecessors:raise ValueError('unreachable prologue group trace')
    while todo:
        i=todo.pop()
        if i in ancestors:continue
        ancestors.add(i);todo.extend(predecessors[i])
    todo=[(0,((),()))];seen=set();entries=set()
    while todo:
        i,state=todo.pop()
        if i not in ancestors or (i,state) in seen:continue
        seen.add((i,state))
        if i==begin:
            entries.add(state);continue
        if len(state[0])>2*panels or len(state[1])*16*threads>2*(tile_m+tile_n)*tile_k:
            raise ValueError('prologue copy cycle or excess work')
        state=advance(state,*ins[i]);todo.extend((n,state) for n in successors[i])
    if len(entries)!=1:raise ValueError('ambiguous or unreachable prologue group trace')
    prologue,pending=entries.pop()
    loop_state=((),())
    for address,text in body[wait+1:-1]:
        if re.search(r'\b(?:BRA|BRX|CALL|RET|EXIT)(?:\.|\s)',text) and not text.startswith('@!PT '):
            raise ValueError('one straight-line K loop required')
        loop_state=advance(loop_state,address,text)
    loop_groups,loop_pending=loop_state
    expected=(2*tile_m*tile_k,2*tile_n*tile_k)
    sizes=lambda groups:tuple(len(g)*16*threads for g in groups)
    if pending or loop_pending or sizes(loop_groups)!=expected or sizes(prologue)!=expected*panels:
        raise ValueError('committed A/B work does not close the source panel window')
    result=dict(status='bound_static_async_group_window',buffer_slots=buffer_slots,
        groups_per_panel=2,prologue_group_bytes=sizes(prologue),loop_group_bytes=sizes(loop_groups),
        prologue_copy_addresses=prologue,loop_copy_addresses=loop_groups,
        wait_address=hex(body[wait][0]),wait_instruction=expected_wait,
        newer_groups_permitted_pending=2*(panels-1),
        oldest_committed_panel_groups_required_complete=2,
        cfg_states_to_first_wait=len(seen),all_paths_reaching_wait_have_same_copy_trace=True,
        absolute_shared_slot_alias_proven=False,measured_ready_times=None,
        limitations='Group work/order only. Completion interpretation requires separate PTX/compiler binding. No all-window drain, latency, issue price, CTA skew or queue policy is inferred.')
    if asynchronous_matrix:
        result.update(prefetch_panels=panels,
            matrix_launch_capacity=matrix_proof['matrix_launch_capacity'],
            refill_after_matrix_wait=True,
            matrix_wait_address=hex(body[matrix_waits[0]][0]),
            first_refill_address=hex(body[min(refills)][0]))
    return result


def analyze_shared_address_banks(instructions, *, loop_begin, loop_end, threads,
                                 bank_count, bank_word_bytes):
    """Prove one source address slice and its untimed matrix-row bank map.

    Conditional CFG edges remain possible; paths ending before the observed
    LDSM cannot define its registers. Only a small integer/LOP3 expression
    subset is evaluated. Copy-free straight-line call clobbers are explicit.
    Unknown definitions, reaching-definition ambiguity and unsupported
    expressions fail closed. This is not a bank latency or port-rate model.
    """
    if (any(isinstance(v,bool) or not isinstance(v,int) for v in
            (threads,bank_count,bank_word_bytes)) or
            threads<=0 or threads%32 or bank_count!=32 or bank_word_bytes!=4):
        raise ValueError('positive warp geometry and documented32x4B bank layout required')
    ins=[(int(r['address'],16),' '.join(r['instruction'].split())) for r in instructions]
    if not ins or any(b[0]<=a[0] for a,b in zip(ins,ins[1:])):
        raise ValueError('ordered unique source addresses required')
    by={a:i for i,(a,_) in enumerate(ins)}
    lo,hi=int(loop_begin,16),int(loop_end,16)
    if lo not in by or hi not in by or lo>hi:raise ValueError('closed source interval required')
    reads=[]
    for i,(a,t) in enumerate(ins):
        if not lo<=a<=hi or t.startswith('@!PT '):continue
        m=re.fullmatch(r'LDSM\.16\.M88(?:\.MT88)?\.([124]) R\d+, \[(R\d+)(?:\+(UR\d+))?(?:\+(0x[0-9a-f]+))?\]',t)
        if m:reads.append((i,int(m[1]),m[2],m[3],0 if m[4] is None else int(m[4],16)))
        elif 'LDSM.' in t:raise ValueError('unsupported matrix shared address')
    if not reads:raise ValueError('source loop has no shared matrix reads')
    target=reads[0][0];base=ins[0][0];call_writes={};successors={}
    def parts(text):
        tokens=text.split();predicated=text.startswith('@')
        return predicated,tokens[1] if predicated else tokens[0],' '.join(tokens[2:] if predicated else tokens[1:])
    def destinations(text):
        _,op,args=parts(text);m=re.match(r'((?:UR|R)\d+),',args)
        if m is None or op.startswith('RET'):return set()
        matrix_load=re.fullmatch(r'LDSM\.16\.M88(?:\.MT88)?\.([124])',op)
        if matrix_load is not None:width=int(matrix_load[1])
        elif op=='HMMA.16816.F32.BF16':width=4
        elif op.startswith(('IMAD.WIDE','UIMAD.WIDE','F2I.U64')):width=2
        elif op.startswith(('LD','ULD')):width=4 if '.128' in op else 2 if '.64' in op else 1
        elif op=='CS2R':width=2
        elif op.startswith(('IMAD','UIMAD','IADD','VIADD','LEA','LOP3','MOV','UMOV','S2R','S2UR',
                            'SHF','USHF','IABS','I2F','F2I','MUFU','SEL','USEL','SGXT','VIMNMX','UIADD','ULEA',
                            'PRMT','P2R')) or op=='CS2R.32':width=1
        else:raise ValueError('unknown source register destination width: '+op)
        register=m[1];prefix='UR' if register.startswith('UR') else 'R';number=int(register[len(prefix):])
        return {prefix+str(number+k) for k in range(width)}
    reachable=set();pending=[0]
    while pending:
        i=pending.pop()
        if i in reachable:continue
        reachable.add(i)
        if i==target:successors[i]=();continue
        address,text=ins[i];pred,op,args=parts(text);next_nodes=[]
        if op=='BRA':
            address=int(args,16)
            if address not in by:raise ValueError('missing source branch target')
            next_nodes.append(by[address])
            if pred:next_nodes.append(i+1)
        elif op=='EXIT':
            if pred:next_nodes.append(i+1)
        elif op=='CALL.REL.NOINC':
            returning=re.fullmatch(r'MOV (R\d+), (0x[0-9a-f]+)',ins[i-1][1]) if i else None
            address=int(args,16)
            if pred or returning is None or address not in by or i+1>=len(ins) or base+int(returning[2],16)!=ins[i+1][0]:
                raise ValueError('unclosed shared-address call')
            summary=_copy_free_callee_work(ins,by[address],returning[1],base)
            if summary is None:raise ValueError('unsupported shared-address callee')
            end=by[int(summary['return_address'],16)]
            call_writes[i]=set().union(*(destinations(t) for _,t in ins[by[address]:end+1]))
            next_nodes.append(i+1)
        elif op.startswith(('CALL','RET','BRX','JMP','JMX','BSSY','BSYNC','BREAK','BRK','SSY','SYNC','PRET')):
            raise ValueError('unsupported source control flow')
        else:next_nodes.append(i+1)
        if any(n>=len(ins) for n in next_nodes):raise ValueError('source falls outside export')
        successors[i]=tuple(next_nodes);pending.extend(next_nodes)
    if target not in reachable:raise ValueError('matrix read is unreachable')
    predecessors={i:set() for i in reachable}
    for i,nodes in successors.items():
        for n in nodes:predecessors[n].add(i)
    ancestors=set();pending=[target]
    while pending:
        i=pending.pop()
        if i in ancestors:continue
        ancestors.add(i);pending.extend(predecessors[i])
    def reaching(register,point):
        definitions=set();visited=set();todo=list(predecessors[point])
        while todo:
            i=todo.pop()
            if i in visited or i not in ancestors:continue
            visited.add(i)
            writes=call_writes[i] if i in call_writes else destinations(ins[i][1])
            if register in writes:
                definitions.add(i)
                if not parts(ins[i][1])[0]:continue
            if not predecessors[i]:definitions.add(None)
            todo.extend(predecessors[i])
        if len(definitions)!=1 or None in definitions:raise ValueError('ambiguous or undefined shared address register: '+register)
        definition,=definitions
        if definition in call_writes:raise ValueError('shared address depends on an opaque callee result')
        return definition
    trace=set();cache={};resolving=set();mask=(1<<32)-1;period=bank_count*bank_word_bytes
    def value(register,point,modulus):
        if register in ('RZ','URZ'):return (0,)*threads
        if not re.fullmatch(r'(?:UR|R)\d+',register):return (int(register,0)%modulus,)*threads
        key=(register,point,modulus)
        if key in cache:return cache[key]
        if key in resolving:raise ValueError('cyclic shared address expression')
        resolving.add(key);j=reaching(register,point);trace.add(j);pred,op,args=parts(ins[j][1]);operands=[v.strip() for v in args.split(',')]
        if pred:raise ValueError('predicated shared address definition')
        if op=='S2R' and operands[1]=='SR_TID.X':result=tuple(i%modulus for i in range(threads))
        elif op in ('MOV','UMOV'):result=value(operands[1],j,modulus)
        elif op in ('ULEA','LEA') and len(operands)==4:
            shift=int(operands[3],0)
            a=(0,)*threads if (1<<shift)%modulus==0 else value(operands[1],j,modulus)
            b=value(operands[2],j,modulus);result=tuple(((x<<shift)+y)%modulus for x,y in zip(a,b))
        elif op in ('IMAD','IMAD.U32','IMAD.SHL.U32','IMAD.IADD') and len(operands)==4:
            if op=='IMAD.IADD' and operands[2]!='0x1':raise ValueError('unbound IMAD.IADD multiplier')
            a,b,c=(value(v,j,modulus) for v in operands[1:4]);result=tuple((x*y+z)%modulus for x,y,z in zip(a,b,c))
        elif op=='IADD3' and len(operands)==4:
            a,b,c=(value(v,j,modulus) for v in operands[1:4]);result=tuple((x+y+z)%modulus for x,y,z in zip(a,b,c))
        elif (op=='SHF.L.U32' and len(operands)==4 and operands[3]=='RZ'
              and re.fullmatch(r'0x[0-9a-f]+',operands[2]) and int(operands[2],16)<32):
            # Local sm90 compiler binding: x<<n with 0<=n<32 becomes
            # SHF.L.U32 d,x,n,RZ. Other funnel halves/modes stay unbound.
            shift=int(operands[2],16);a=value(operands[1],j,modulus)
            result=tuple((x<<shift)%modulus for x in a)
        elif op=='LOP3.LUT' and operands[-1] in ('PT','!PT'):
            a,b,c=(value(v,j,modulus) for v in operands[1:4]);lut=int(operands[4],0);result=[]
            for x,y,z in zip(a,b,c):
                word=0
                for bit in range(8):
                    if lut&(1<<bit):word|=(x if bit&4 else ~x)&(y if bit&2 else ~y)&(z if bit&1 else ~z)
                result.append((word&mask)%modulus)
            result=tuple(result)
        else:raise ValueError('unsupported shared address expression: '+ins[j][1])
        cache[key]=result;resolving.remove(key);return result
    profiles=[];gp_bases={r[2] for r in reads};uniform_bases={r[3] for r in reads if r[3] is not None}
    for i,(a,t) in enumerate(ins):
        if not lo<=a<=hi or t.startswith('@!PT '):continue
        writes=destinations(t)
        if writes&gp_bases or i>target and writes&uniform_bases:raise ValueError('shared base changes within the source read interval')
    for index,matrices,register,uniform,offset in reads:
        addresses=value(register,target,1<<32)
        if uniform is not None and any(value(uniform,target,period)):raise ValueError('uniform shared base has an unclosed bank remainder')
        if any((a+offset)%16 for a in addresses):raise ValueError('shared matrix row is not naturally16B aligned')
        groups=[]
        for warp in range(threads//32):
            for matrix in range(matrices):
                words=[set() for _ in range(bank_count)]
                for lane in range(8*matrix,8*(matrix+1)):
                    row=addresses[warp*32+lane]+offset
                    for byte in range(0,16,bank_word_bytes):
                        word=(row+byte)//bank_word_bytes;words[word%bank_count].add(word)
                groups.append(dict(warp=warp,matrix=matrix,max_distinct_words_per_bank=max(map(len,words)),
                                   occupied_banks=sum(bool(w) for w in words)))
        profiles.append(dict(address=hex(ins[index][0]),base_register=register,uniform_register=uniform,
                             constant_offset=offset,matrices=matrices,
                             row_addresses_without_bank_aligned_uniform_bytes=[a+offset for a in addresses],
                             uniform_bank_remainder_bytes=0,row_groups=groups))
    return dict(status='bound_static_shared_bank_addresses',threads=threads,bank_count=bank_count,
        bank_word_bytes=bank_word_bytes,bank_period_bytes=period,profiles=profiles,
        reaching_definition_slice=[dict(address=hex(ins[i][0]),instruction=ins[i][1]) for i in sorted(trace)],
        cfg_reachable_nodes=len(reachable),cfg_nodes_reaching_first_read=len(ancestors),
        source_paths_exiting_before_first_read_pruned=True,
        measured_bank_conflicts=None,dependency_latency_cycles=None,
        limitations='Static 128B row-group bank word multiplicity only. Uniform base is closed modulo 128B, not as an absolute address. No measured wavefront, loaded port contention, interwarp service or latency is inferred.')


def analyze_register_consumption(instructions, *, tile_m, tile_n, tile_k,
                                 threads, physical_k_tiles):
    """Close one straight-line LDSM/HMMA fragment interval without pricing it.

    SM80-style m16n8k16 BF16 uses four A, two B and four C/D 32-bit
    registers per lane. LDSM x1/x2/x4 defines one/two/four registers.
    Register versions, rather than register numbers alone, identify ownership.
    MOV and exact zero-product IMAD.MOV preserve value origins through aliases;
    address arithmetic may not transform a tracked matrix fragment. Joint path
    counts retain overlap between load readiness and accumulator dependencies.
    Unsupported instructions in the interval fail closed; this deliberately
    does not claim a general SASS dependency decoder or scoreboard latency.
    """
    geometry=(tile_m,tile_n,tile_k,threads,physical_k_tiles)
    if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in geometry) or threads%32:
        raise ValueError('positive integral warp geometry required')
    body=[(int(i['address'],16),i['instruction']) for i in instructions
          if not i['instruction'].startswith('@!PT ')]
    if not body or any(b[0]<=a[0] for a,b in zip(body,body[1:])):
        raise ValueError('ordered unique instruction addresses required')
    branch=re.fullmatch(r'(?:@!?P[0-9]+ )?BRA (0x[0-9a-f]+)',body[-1][1])
    if branch is None or int(branch[1],16)!=body[0][0] or any(
            t.startswith('@') or re.search(r'\b(?:BRA|BRX|CALL|RET|EXIT)(?:\.|\s)',t)
            for _,t in body[:-1]):
        raise ValueError('one unpredicated straight-line loop body required')
    loads=[i for i,(_,t) in enumerate(body) if t.startswith('LDSM.')]
    matrix=[i for i,(_,t) in enumerate(body) if 'HMMA.' in t]
    if not loads or not matrix or matrix[0]<loads[0] or any('HGMMA.' in t for _,t in body):
        raise ValueError('shared definitions must precede matrix consumption')
    first,last=loads[0],matrix[-1]
    waits=[i for i,(_,t) in enumerate(body[:first]) if t.startswith('DEPBAR.LE SB0, ')]
    barriers=[i for i,(_,t) in enumerate(body[:first]) if t=='BAR.SYNC.DEFER_BLOCKING 0x0']
    if loads[-1]>last or not waits or not barriers or max(waits)>=max(barriers):
        raise ValueError('operand-ready wait/barrier and consumed loads required')
    events=[];live={};origins={};last_read={};edges=[];live_in=set();definitions={}
    for address,text in body[first:last+1]:
        if text=='NOP':continue
        load=re.fullmatch(r'LDSM\.16\.M88(?:\.MT88)?\.([124]) R(\d+), (\[[^\]]+\])',text)
        mma=re.fullmatch(r'HMMA\.16816\.F32\.BF16 R(\d+), R(\d+), R(\d+), R(\d+)',text)
        move=re.fullmatch(r'(?:MOV R(\d+), R(\d+)|IMAD\.MOV\.U32 R(\d+), RZ, RZ, R(\d+))',text)
        index=len(events)
        copied_origin=None
        if load is not None:
            width,base=int(load[1]),int(load[2]);writes=tuple(range(base,base+width))
            if base%width:raise ValueError('unaligned LDSM register tuple')
            event=dict(address=hex(address),kind='ldsm',writes=writes,reads=(),shared_address=load[3])
            definitions[index]={r:[] for r in writes}
        elif mma is not None:
            d,a,b,c=map(int,mma.groups());writes=tuple(range(d,d+4))
            ab=tuple(range(a,a+4))+tuple(range(b,b+2));acc=tuple(range(c,c+4))
            producers=set()
            for register in ab:
                if register not in origins or origins[register][0]!='ldsm':
                    raise ValueError('HMMA A/B requires current LDSM register definitions')
                _,producer,source_register=origins[register]
                definitions[producer][source_register].append(index);producers.add(producer)
            for producer in sorted(producers):edges.append(dict(producer=producer,consumer=index,kind='ldsm_to_hmma_raw'))
            for register in acc:
                if register not in origins:live_in.add(register)
                elif origins[register][0]=='live_in':live_in.add(origins[register][2])
                elif origins[register][0]!='hmma':
                    raise ValueError('accumulator aliases an unconsumed shared fragment')
                else:edges.append(dict(producer=origins[register][1],consumer=index,kind='accumulator_raw'))
            event=dict(address=hex(address),kind='hmma',writes=writes,reads=ab+acc,
                       operand_a=tuple(range(a,a+4)),operand_b=tuple(range(b,b+2)),accumulator=acc)
            for register in event['reads']:
                if register in live and events[live[register]]['kind']=='register_copy':
                    edges.append(dict(producer=live[register],consumer=index,kind='register_transport_raw'))
                last_read[register]=index
        elif move is not None:
            d,source=map(int,move.groups()[:2] if move[1] is not None else move.groups()[2:])
            copied_origin=origins[source] if source in origins else ('live_in',None,source)
            if source in live:edges.append(dict(producer=live[source],consumer=index,kind='register_transport_raw'))
            if copied_origin[0] in ('ldsm','hmma'):
                edges.append(dict(producer=copied_origin[1],consumer=index,
                    kind='ldsm_to_register_copy_raw' if copied_origin[0]=='ldsm' else 'accumulator_transport_raw'))
            last_read[source]=index;writes=(d,)
            event=dict(address=hex(address),kind='register_copy',writes=writes,reads=(source,),
                       copied_origin=copied_origin)
        elif text in ('BAR.SYNC.DEFER_BLOCKING 0x0','LDGDEPBAR') or text.startswith('LDGSTS.E.BYPASS.128 '):
            # Existing copy-ring closure owns destination slot safety. These
            # instructions define no register fragments; their latency is
            # omitted from this register-only dependency lower bound.
            writes=();event=dict(address=hex(address),kind='nonregister_memory',writes=(),reads=())
        elif (re.fullmatch(r'(?:ISETP\.[A-Z0-9.]+|PLOP3\.LUT) P\d+, .+',text)
              or re.fullmatch(r'UISETP\.[A-Z0-9.]+ UP\d+, .+',text)
              or re.fullmatch(r'(?:UIADD3(?:\.X)?|USEL) UR\d+, .+',text)):
            reads=tuple(int(r) for r in re.findall(r'\bR(\d+)\b',text))
            if any(r in origins and origins[r][0] in ('ldsm','hmma') for r in reads):
                raise ValueError('predicate/uniform operation reads a tracked fragment')
            # No general-purpose destination. Uniform/predicate issue is not
            # included in the conservative warp data-instruction count.
            writes=();event=dict(address=hex(address),kind='predicate_or_uniform',writes=(),reads=reads)
        else:
            scalar=re.fullmatch(r'(SEL|IMAD\.WIDE\.U32|IMAD|IMAD\.X|IMAD\.IADD|IMAD\.U32|IADD3|LEA|MOV) R(\d+), (.+)',text)
            if scalar is None:raise ValueError('fragment interval contains unsupported instruction: '+text)
            if scalar[1]=='MOV' and re.fullmatch(r'UR\d+',scalar[3]) is None:
                raise ValueError('fragment interval contains unsupported instruction: '+text)
            reads=tuple(int(r) for r in re.findall(r'\bR(\d+)\b',scalar[3]))
            if scalar[1]=='IMAD.WIDE.U32':
                wide=re.fullmatch(r'R(\d+)',scalar[3].split(',')[-1].strip())
                if wide is not None:reads+=(int(wide[1])+1,)
            if any(r in origins and origins[r][0] in ('ldsm','hmma') for r in reads):
                raise ValueError('scalar operation reads a tracked fragment')
            base=int(scalar[2]);width=2 if scalar[1]=='IMAD.WIDE.U32' else 1
            writes=tuple(range(base,base+width))
            event=dict(address=hex(address),kind='scalar',writes=writes,reads=reads)
        for register in writes:
            if register in live:
                edges.append(dict(producer=live[register],consumer=index,kind='register_redefinition_waw'))
            if register in last_read and last_read[register]!=index:
                edges.append(dict(producer=last_read[register],consumer=index,kind='register_reuse_war'))
            live[register]=index
            origins[register]=copied_origin if copied_origin is not None else (event['kind'],index,register)
        events.append(event)
    if any(not uses for registers in definitions.values() for uses in registers.values()):
        raise ValueError('every loaded register version must have a reachable matrix consumer')
    if any(r not in origins or origins[r][0]!='hmma' for r in live_in):
        raise ValueError('loop-carried accumulator must be produced at interval exit')
    for _,text in body[:first]+body[last+1:-1]:
        destination=re.match(r'(\S+) R(\d+),',text)
        if destination is not None:
            base=int(destination[2]);width=2 if '.WIDE.' in destination[1] or destination[1].endswith('.64') else 1
            if set(range(base,base+width)) & live_in:
                raise ValueError('accumulator register is redefined across fragment boundary')
    written={r for event in events for r in event['writes']}
    if any(int(r) in written for event in events if event['kind']=='ldsm'
           for r in re.findall(r'\bR(\d+)\b',event['shared_address'])):
        raise ValueError('shared address aliases fragment-interval register definitions')
    uniform_written={int(m[1]) for _,t in body[first:last+1]
                     if (m:=re.match(r'(?:UIADD3(?:\.X)?|USEL) UR(\d+),',t)) is not None}
    if any(int(r) in uniform_written for event in events if event['kind']=='ldsm'
           for r in re.findall(r'\bUR(\d+)\b',event['shared_address'])):
        raise ValueError('shared address uniform base changes inside fragment interval')
    copies=[t for _,t in body if t.startswith('LDGSTS.')]
    if any(not t.split()[0].endswith('.128') for t in copies):raise ValueError('unknown copy width')
    operations=sum(e['kind']=='hmma' for e in events)*4096*(threads//32)
    if operations!=2*tile_m*tile_n*tile_k*physical_k_tiles:
        raise ValueError('complete physical K matrix work required')
    if len(copies)*16*threads!=2*(tile_m+tile_n)*tile_k*physical_k_tiles:
        raise ValueError('complete padded A/B destination work required')
    for index,registers in definitions.items():
        events[index]['register_consumers']={str(r):uses for r,uses in registers.items()}
    # Warp issue order is a zero-weight edge here. Count dependency depth
    # with only one edge family having unit weight, all other service free.
    # These are symbolic latency coefficients, never measured cycle counts.
    depths={}
    for name,kinds in (('load_to_use',('ldsm_to_hmma_raw','ldsm_to_register_copy_raw')),
                       ('accumulator',('accumulator_raw','accumulator_transport_raw'))):
        levels=[]
        for index in range(len(events)):
            levels.append(max([levels[-1] if levels else 0]+[
                levels[e['producer']]+1 for e in edges if e['consumer']==index and e['kind'] in kinds]))
        depths[name]=max(levels)
    depths['accumulator_result_ready']=depths['accumulator']+1
    def frontier(pairs):
        return {p for p in pairs if not any(q!=p and q[0]>=p[0] and q[1]>=p[1] for q in pairs)}
    # Keep joint dependency counts: separately maximal load and accumulator
    # paths generally overlap, so their individual maxima cannot be added.
    path_counts=[]
    for index in range(len(events)):
        pairs=set(path_counts[-1]) if path_counts else {(0,0)}
        for edge in edges:
            if edge['consumer']!=index:continue
            dl=int(edge['kind'] in ('ldsm_to_hmma_raw','ldsm_to_register_copy_raw'))
            dh=int(edge['kind'] in ('accumulator_raw','accumulator_transport_raw'))
            pairs.update((a+dl,b+dh) for a,b in path_counts[edge['producer']])
        path_counts.append(frontier(pairs))
    ready_counts=frontier({(a,b+1) for i,e in enumerate(events) if e['kind']=='hmma'
                           for a,b in path_counts[i]})
    return dict(status='bound_straight_line_register_consumption',events=events,edges=edges,
        loop_carried_accumulator_registers=sorted(live_in),
        ldsm_count=len(definitions),hmma_count=sum(e['kind']=='hmma' for e in events),
        load_defined_registers=sum(len(r) for r in definitions.values()),
        interval_warp_instruction_lower_count=sum(e['kind'] in ('ldsm','hmma') for e in events),
        shared_read_bytes_per_cta=sum(len(r) for r in definitions.values())*4*threads,
        async_destination_bytes_per_cta=len(copies)*16*threads,
        physical_k_tiles_per_static_loop=physical_k_tiles,
        symbolic_serial_dependency_depths=depths,
        joint_load_accumulator_result_path_counts=[dict(load_to_use=a,hmma_result=b)
                                                  for a,b in sorted(ready_counts)],
        dependency_latency_cycles=None,issue_service_cycles=None,
        limitations=('Ready wait/barrier dominates the interval; exact copy-group/slot ownership remains with the existing source ring proof. '
                     'WAR records operand register use before overwrite, not measured scoreboard completion. '
                     'No native instruction stall bits, bank conflicts or HMMA dependency latency are inferred. '
                     'Joint counts are a source precedence lower bound, not a complete instruction schedule.'))


def analyze_projection(source, *, tile_m, tile_n, tile_k, threads):
    if threads%32 or min(tile_m,tile_n,tile_k,threads)<=0:
        raise ValueError('positive warp-aligned physical geometry required')
    instructions=[]
    for line in source.splitlines():
        match=re.match(r'\s*(0x[0-9a-f]+)\s+(.*)',line)
        if match:
            instructions.append((int(match[1],16),' '.join(match[2].split())))
    if not instructions or len({a for a,_ in instructions})!=len(instructions):
        raise ValueError('unique complete SASS addresses required')
    by_address={address:i for i,(address,_) in enumerate(instructions)}
    loops=[]
    for end,(address,text) in enumerate(instructions):
        branch=re.fullmatch(r'(?:@!?P[0-9]+ )?BRA (0x[0-9a-f]+)',text)
        if branch is None or int(branch[1],16)>=address:
            continue
        target=int(branch[1],16)
        starts=[i for i,(a,_) in enumerate(instructions) if a==target]
        if len(starts)!=1:
            continue
        begin,=starts
        body=instructions[begin:end+1]
        if not any('HMMA.' in t for _,t in body):
            continue
        loops.append((begin,end,body))
    # Reject an outer interval containing another matrix loop.
    loops=[x for x in loops if not any(x[0]<=y[0] and y[1]<x[1] for y in loops)]
    if len(loops)!=1:
        raise ValueError('one straight-line matrix K loop required')
    begin,end,body=loops[0]
    read=write=hmma=0
    for i,(address,text) in enumerate(body):
        if text.startswith('@!PT '):
            continue
        if i==len(body)-1:
            continue
        if 'HGMMA.' in text or re.search(r'\b(?:BRA|BRX|CALL|RET|EXIT)\b',text):
            raise ValueError('unsupported control flow or Tensor family')
        if text.startswith('@') and re.search(r'\b(?:HMMA|LDSM|LDS|LDGSTS|STS|STSM)\b',text):
            raise ValueError('predicated data movement is not statically closed')
        opcode=text.split()[0]
        if opcode=='HMMA.16816.F32.BF16':
            hmma+=1
        elif opcode.startswith('HMMA.'):
            raise ValueError('unknown HMMA instruction semantics')
        elif opcode.startswith('LDSM.'):
            match=re.fullmatch(r'LDSM\.16\.M88(?:\.MT88)?\.([124])',opcode)
            if match is None:
                raise ValueError('unknown LDSM matrix extent')
            read+=int(match[1])*8*8*2*(threads//32)
        elif opcode.startswith('LDGSTS.'):
            if not opcode.endswith('.128'):
                raise ValueError('unknown asynchronous copy width')
            write+=16*threads
        elif opcode.startswith(('LDS','STS')):
            raise ValueError('unclosed shared instruction in operand loop')
    operations=hmma*4096*(threads//32)
    unit=2*tile_m*tile_n*tile_k
    if not operations or operations%unit:
        raise ValueError('matrix work does not match complete physical K tiles')
    steps=operations//unit
    if read%steps or write%steps:
        raise ValueError('shared work cannot normalize to a K tile')
    read//=steps
    write//=steps
    if write!=2*(tile_m+tile_n)*tile_k:
        raise ValueError('copy destination work does not conserve padded A/B tiles')
    prologue_write=0
    for index,(address,text) in enumerate(instructions[:begin]):
        if text.startswith('@!PT '):
            continue
        opcode=text.split()[0]
        if 'LDGSTS.' in text:
            if text.startswith('@') or not opcode.endswith('.128'):
                raise ValueError('unclosed prologue copy')
            prologue_write+=16*threads
        elif opcode=='CALL.REL.NOINC':
            if index==0 or index+1>=len(instructions):
                raise ValueError('unclosed prologue helper')
            returning=re.fullmatch(r'MOV (R[0-9]+), (0x[0-9a-f]+)',instructions[index-1][1])
            target=int(text.split()[-1],16)
            base=instructions[0][0]
            if returning is None or target not in by_address:
                raise ValueError('unclosed prologue helper return')
            register,offset=returning.groups()
            if base+int(offset,16)!=instructions[index+1][0]:
                raise ValueError('prologue helper return is not caller successor')
            callee=_copy_free_callee_work(instructions,by_address[target],register,base)
            if callee is None:
                raise ValueError('prologue helper control or return is unsupported')
            last=by_address[int(callee['return_address'],16)]
            if any(re.search(r'\b(?:LDGSTS|LDSM|LDS|STS|STSM|HMMA|HGMMA)(?:\.|\s)',t)
                   for _,t in instructions[by_address[target]:last+1]):
                raise ValueError('prologue helper contains shared or Tensor work')
        elif 'CALL' in text:
            raise ValueError('unclosed predicated or unsupported helper')
        elif re.search(r'\b(?:LDSM|LDS|STS|STSM|HMMA|HGMMA)(?:\.|\s)',text):
            raise ValueError('unclosed prologue shared or Tensor work')
    if prologue_write!=write:
        raise ValueError('one-panel prologue must fill one complete operand panel')
    output_read=output_write=0
    exited=False
    for _,text in instructions[end+1:]:
        if text.startswith('@!PT '):
            continue
        if text=='EXIT':
            exited=True
            break
        if text.startswith('@') and text.endswith(' EXIT'):
            # Full output tile is a conservative lane-count bound after a
            # masked-thread return; do not claim those lanes are observed live.
            continue
        opcode=text.split()[0]
        if opcode=='STSM.16.M88.4':
            output_write+=512*(threads//32)
        elif opcode=='LDS.128':
            output_read+=16*threads
        elif re.search(r'\b(?:LDGSTS|LDSM|LDS|STS|STSM|HMMA|HGMMA|BRA|CALL|RET)(?:\.|\s)',text):
            raise ValueError('unclosed output shared/Tensor/control work')
    if not exited or output_write!=2*tile_m*tile_n or output_read!=output_write:
        raise ValueError('one full BF16 output tile must be rearranged once')
    return dict(status='bound_hmma_operand_shared_work',read_bytes_per_k_cta=read,
                write_bytes_per_k_cta=write,prologue_write_bytes_per_cta=prologue_write,
                output_shared_read_bytes_per_cta_upper_bound=output_read,
                output_shared_write_bytes_per_cta=output_write,
                output_read_scope='full-tile upper bound after any masked-thread exit',
                loop_begin=hex(instructions[begin][0]),loop_end=hex(instructions[end][0]),
                physical_k_tiles_per_static_loop=steps,
                instruction_body=[dict(address=hex(a),instruction=t) for a,t in body])
