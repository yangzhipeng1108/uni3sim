"""Static entry-issue lower bound before a compiled MoE's first operand copy.

No timing fields, production measurements or predictor imports. The graph
permits every conditional edge and charges only unpredicated instructions.
Verified copy-free straight-line callees contribute their actual work on
each call edge. Predicated work and instruction/dependency latencies remain
omitted, so this is a work-count lower bound, not complete startup time.
"""
import argparse
import heapq
import json
import re
from pathlib import Path


class UnsupportedStartupGraph(ValueError):
    """The export does not close a required control-flow fact."""


def _copy_free_callee_work(instructions, start, register, base):
    """Count a copy-free straight-line call with a preserved return address.

    Even a predicated callee copy invalidates first-copy ordering: its
    predicate is not known here. Unsupported calls/branches remain unbound.
    The returned work includes RET, while CALL stays on the caller edge.
    """
    aliases={register};count=0
    for address,text in instructions[start:]:
        tokens=text.split();predicated=text.startswith('@')
        count+=not predicated
        opcode=tokens[1] if predicated else tokens[0]
        operands=' '.join(tokens[2:] if predicated else tokens[1:])
        if opcode=='RET.REL.NODEC':
            returning=operands.split()
            if (not predicated and len(returning)==2
                    and returning[0] in aliases and returning[1]==hex(base)):
                return dict(unpredicated_warp_instructions=count,
                    entry_address=hex(instructions[start][0]),
                    return_address=hex(address))
            return None
        if opcode.startswith(('LDGSTS','BRA','BRX','JMP','JMX','CALL','EXIT',
                'RET','BSSY','BSYNC','BREAK','BRK','SSY','SYNC','PRET',
                'RPCMOV','RTT')):
            return None
        destination=operands.split(',')[0]
        if not re.fullmatch(r'R[0-9]+',destination):continue
        copied=(re.fullmatch(r'MOV (R[0-9]+), (R[0-9]+)',text)
            or re.fullmatch(r'IMAD.MOV.U32 (R[0-9]+), RZ, RZ, (R[0-9]+)',text))
        source_is_alias=copied is not None and copied[2] in aliases
        width=4 if '.128' in opcode else 2 if any(
            x in opcode for x in ('.64','.U64','.WIDE')) else 1
        for offset in range(width):
            aliases.discard('R'+str(int(destination[1:])+offset))
        if source_is_alias and not predicated:aliases.add(destination)
    return None


def minimum_startup_issue(instructions):
    if not instructions or len({a for a,_ in instructions})!=len(instructions):
        raise UnsupportedStartupGraph('empty or duplicate instruction addresses')
    by_address={address:i for i,(address,_) in enumerate(instructions)}
    base=instructions[0][0]
    pending=[(0,0,())];visited=set();callees={}
    while pending:
        count,index,path=heapq.heappop(pending)
        if index in visited:continue
        visited.add(index)
        address,text=instructions[index]
        predicated=text.startswith('@')
        opcode=text.split()[1] if predicated else text.split()[0]
        if opcode.startswith('LDGSTS.'):
            return dict(minimum_unpredicated_startup_warp_instructions=count,
                entry_address=hex(base),first_copy_address=hex(address),
                minimum_path=[dict(address=hex(instructions[i][0]),
                    instruction=instructions[i][1]) for i in path],
                callees_on_minimum_path=[dict(callsite_address=hex(instructions[i][0]),
                    **callees[i]) for i in path if i in callees],
                predicated_instructions_charged=False,callee_body_charged=True)
        cost=0 if predicated else 1
        next_indices=[]
        if opcode=='BRA':
            target=int(text.split()[-1],16)
            if target not in by_address:
                raise UnsupportedStartupGraph('branch target missing from export')
            next_indices.append(by_address[target])
            if predicated:next_indices.append(index+1)
        elif opcode=='EXIT':
            if predicated:next_indices.append(index+1)
        elif opcode=='CALL.REL.NOINC':
            if predicated or index==0 or index+1==len(instructions):
                raise UnsupportedStartupGraph('unclosed call')
            returning=re.fullmatch(r'MOV (R[0-9]+), (0x[0-9a-f]+)',instructions[index-1][1])
            target=int(text.split()[-1],16)
            if returning is None or target not in by_address:
                raise UnsupportedStartupGraph('unknown explicit return register')
            register,offset=returning.groups()
            if base+int(offset,16)!=instructions[index+1][0]:
                raise UnsupportedStartupGraph('explicit return is not caller successor')
            callee=_copy_free_callee_work(instructions,by_address[target],register,base)
            if callee is None:
                raise UnsupportedStartupGraph('copy-free callee return is unbound')
            # Re-solve the CFG with complete call-edge work. Adding helper
            # counts to the old caller-only shortest path is not necessarily
            # a lower bound: another arm can avoid an expensive call.
            callees[index]=callee
            cost+=callee['unpredicated_warp_instructions']
            next_indices.append(index+1)
        elif opcode.startswith(('CALL','JMP','RET','BRX','JMX','BSSY','BSYNC',
                'BREAK','BRK','SSY','SYNC','PRET','RPCMOV','RTT')):
            raise UnsupportedStartupGraph('unsupported control flow')
        else:
            next_indices.append(index+1)
        for successor in next_indices:
            if successor<len(instructions):
                heapq.heappush(pending,(count+cost,successor,path+(index,)))
    raise UnsupportedStartupGraph('no asynchronous operand copy reachable')


def analyze(source):
    sections=[];current=[]
    for line in source.splitlines():
        if line.startswith('Kernel Name'):
            if current:sections.append(current)
            current=[line]
        else:current.append(line)
    sections.append(current)
    out=[]
    for section in sections:
        if not any('HMMA.' in x or 'HGMMA.' in x for x in section):continue
        instructions=[]
        for line in section:
            matched=re.match(r'\s*(0x[0-9a-f]+)\s+(.*)',line)
            if matched:instructions.append((int(matched[1],16),' '.join(matched[2].split())))
        try:
            result=minimum_startup_issue(instructions)
        except UnsupportedStartupGraph as exc:
            result=dict(status='unbound',reason=str(exc))
        else:
            result['status']='bound_conservative_startup_issue'
        out.append(result)
    return dict(production_or_evaluation_duration_consumed=False,
        scope='CFG shortest total entry work before any first LDGSTS, including verified copy-free straight-line callees; conditional arms optimistic, predicated instructions and all dependency latency omitted',
        projections=out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('source',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    result=analyze(args.source.read_text());args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(status=x['status'],count=x['minimum_unpredicated_startup_warp_instructions'] if x['status']=='bound_conservative_startup_issue' else None) for x in result['projections']]))
