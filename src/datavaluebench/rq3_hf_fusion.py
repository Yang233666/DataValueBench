"""Reusable DataValueBench benchmark implementation."""
import numpy as np
CHANNELS=('structured','lexical_bm25','semantic_bge')
def fuse_channels(channels):
    if not channels or any(c not in CHANNELS for c in channels):raise ValueError('unknown or empty channel')
    union={}
    for channel in CHANNELS:
        if channel not in channels:continue
        selected=list(channels[channel])
        if len(set(selected))!=len(selected):raise ValueError('duplicate channel candidate')
        for rank,candidate in enumerate(selected,1):
            item=union.setdefault(int(candidate),{'candidate_position':int(candidate),'rrf_score':0.,**{c+'_rank':None for c in CHANNELS}})
            item['rrf_score']+=1/(60+rank);item[channel+'_rank']=rank
    ordered=sorted(union.values(),key=lambda r:(-r['rrf_score'],r['candidate_position']))
    return [dict(r,fused_rank=i) for i,r in enumerate(ordered,1)]
