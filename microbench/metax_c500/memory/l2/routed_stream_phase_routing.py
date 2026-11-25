"""Write the routing description routed_stream_phase.cpp replays for one
canonical C500 MoE case: top-k ids from the dataset runner's deterministic heap
over tok_per_expert (scripts/kernel_dataset/metax_c500/fused_moe_runner.py at
c8cf3f318, runner sha256 436bee88...), then moe_align_block_size (routes of each
expert in ascending flat index, padded to BM).  Reads only the case shape; no
duration or traffic observation is consumed.

Usage: python routed_stream_phase_routing.py <case_id> <out.bin>
"""
import heapq, json, sys, struct
import zstandard
import numpy as np
case = sys.argv[1]; out = sys.argv[2]
d = json.loads(zstandard.ZstdDecompressor().stream_reader(open('kernel_dataset/metax_c500/fused_moe.json.zst','rb')).read())
attr = d[case]['op_info']['Attr']
T, E, K_, H, I = attr['TOK'], attr['E'], attr['TOPK'], attr['H'], attr['INT']
counts = list(attr['tok_per_expert'])
heap = [(-c, e) for e, c in enumerate(counts) if c]; heapq.heapify(heap); rows = []
for _ in range(T):
    sel = [heapq.heappop(heap) for _ in range(K_)]; rows.append([e for _, e in sel])
    for nc, e in sel:
        if nc + 1 < 0: heapq.heappush(heap, (nc + 1, e))
flat = np.array(rows, dtype=np.int64).reshape(-1)
BM = 64 if T > E else 16
assert BM == 64
per = [[] for _ in range(E)]
for i, e in enumerate(flat): per[e].append(i)
sorted_ids = []; expert_of_block = []
for e in range(E):
    c = len(per[e]); pad = -(-c // BM) * BM
    seg = per[e] + [T * K_] * (pad - c); sorted_ids += seg; expert_of_block += [e] * (pad // BM)
ntpp = len(sorted_ids); EM = T * K_ + E * (BM - 1)
with open(out, 'wb') as f:
    f.write(struct.pack('<9q', T, E, K_, H, I, BM, ntpp, EM, len(expert_of_block)))
    np.array(sorted_ids, dtype=np.int32).tofile(f); np.array(expert_of_block, dtype=np.int32).tofile(f)
print(case, 'T',T,'E',E,'topk',K_,'H',H,'I',I,'ntpp',ntpp,'EM',EM,'blocks',len(expert_of_block))
