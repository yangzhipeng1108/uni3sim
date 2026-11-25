"""Static post-operand-wait issue work before the first HGMMA of each K tile.

This closes one linear, one-physical-K loop. Predicated scalar instructions
and all instruction/dependency latencies are omitted. No duration is read.
"""
import re


def analyze_projection(source, *, tile_m, tile_n, tile_k, threads):
    if (any(isinstance(v,bool) or not isinstance(v,int) or v<=0
            for v in (tile_m,tile_n,tile_k,threads)) or threads % 128):
        raise ValueError('complete warpgroup geometry required')
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
        if branch is None:
            continue
        target=int(branch[1],16)
        if target>=address or target not in by_address:
            continue
        begin=by_address[target]
        if any('HGMMA.' in t for _,t in instructions[begin:end+1]):
            loops.append((begin,end))
    if len(loops)!=1:
        raise ValueError('one linear HGMMA K loop required')
    begin,end=loops[0]
    body=instructions[begin:end+1]
    if (not body[0][1].startswith('DEPBAR.LE SB0, ')
            or body[1][1]!='BAR.SYNC.DEFER_BLOCKING 0x0'):
        raise ValueError('operand wait and CTA barrier must dominate preparation')
    matrix_indices=[]
    operations=0
    for i,(_,text) in enumerate(body):
        if text.startswith('@!PT '):
            continue
        opcode=text.split()[1] if text.startswith('@') else text.split()[0]
        if i<len(body)-1 and opcode.startswith(('BRA','BRX','CALL','RET','EXIT',
                'JMP','JMX','BSSY','BSYNC','BREAK','BRK','SSY','SYNC','PRET')):
            raise ValueError('nonlinear preparation loop is unbound')
        if opcode.startswith('HMMA.'):
            raise ValueError('mixed Tensor families are unbound')
        if opcode.startswith('HGMMA.'):
            match=re.fullmatch(r'HGMMA\.(\d+)x(\d+)x(\d+)\.F32\.BF16',opcode)
            if match is None or text.startswith('@'):
                raise ValueError('unclosed HGMMA work')
            m,n,k=map(int,match.groups())
            operations+=2*m*n*k*(threads//128)
            matrix_indices.append(i)
    if (operations!=2*tile_m*tile_n*tile_k or not matrix_indices
            or any(re.search(r'\b(?:HMMA|HGMMA)\.',text)
                   for _,text in instructions[:begin]+instructions[end+1:])):
        raise ValueError('one complete physical K tile must close the matrix work')
    # The wait and barrier establish readiness; their issue may overlap the
    # last outstanding operand returns. Charge only work strictly after both.
    prefix=body[2:matrix_indices[0]]
    waits=[(i,re.fullmatch(r'WARPGROUP\.DEPBAR\.LE gsb0, 0x([0-7])',text))
           for i,(_,text) in enumerate(body) if 'WARPGROUP.DEPBAR' in text]
    commits=[i for i,(_,text) in enumerate(body) if 'HGMMA.' in text and 'gsb' in text]
    if (len(waits)!=1 or waits[0][1] is None or waits[0][0]<=matrix_indices[-1]
            or commits!=[matrix_indices[-1]] or not body[commits[0]][1].endswith(', gsb0')):
        raise ValueError('one committed matrix group and its post-issue wait required')
    # wait_group N permits N old groups before the following iteration.
    # The next group can be issued before that wait, hence N+1 launch slots.
    matrix_launch_capacity=int(waits[0][1][1],16)+1
    for _,text in prefix:
        if text.startswith('@!PT '):
            continue
        opcode=text.split()[1] if text.startswith('@') else text.split()[0]
        if opcode.startswith(('LD','ST','RED','ATOM','CALL','BRA','BRX','RET','EXIT')):
            raise ValueError('preparation cannot hide memory or control work')
    count=sum(not text.startswith('@') for _,text in prefix)
    if count<=0:
        raise ValueError('positive preparation issue work required')
    return dict(status='bound_post_wait_matrix_preparation_issue',
        unpredicated_warp_instructions_per_k=count,
        matrix_launch_capacity=matrix_launch_capacity,
        loop_begin=hex(instructions[begin][0]),loop_end=hex(instructions[end][0]),
        first_matrix_address=hex(body[matrix_indices[0]][0]),
        physical_k_tiles_per_static_loop=1,predicated_instructions_charged=False,
        dependency_or_barrier_latency_charged=False,
        operand_wait_and_barrier=[dict(address=hex(a),instruction=t) for a,t in body[:2]],
        preparation=[dict(address=hex(a),instruction=t) for a,t in prefix])
