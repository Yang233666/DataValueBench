"""Reusable DataValueBench benchmark implementation."""
import numpy as np
from .rq3_ranking import bge01_kernel, id_sort_key, l2_normalize


def _validate(ids, k):
    if k not in (5,10,20) or k > len(ids) or len(ids) != len(set(ids)):
        raise ValueError("invalid frozen requested depth or candidate identities")


def mmr_prefix(candidate_ids, quality, vectors, *, lam, k):
    ids=list(candidate_ids);_validate(ids,k)
    if lam not in {0.,.25,.5,.75,1.}:
        raise ValueError("lambda outside frozen grid")
    q=np.asarray(quality,dtype=np.float64)
    if q.shape != (len(ids),) or not np.isfinite(q).all():raise ValueError("invalid quality")
    similarity=bge01_kernel(vectors);chosen=[];remaining=set(range(len(ids)))
    while len(chosen)<k:
        values={i:lam*float(q[i])-(1.-lam)*max((float(similarity[i,j]) for j in chosen),default=0.)
                for i in remaining}
        best=min(remaining,key=lambda i:(-values[i],id_sort_key(ids[i])))
        chosen.append(best);remaining.remove(best)
    return [ids[i] for i in chosen]


def facility_location_prefix(candidate_ids, vectors, *, k):
    ids=list(candidate_ids);_validate(ids,k)
    similarity=bge01_kernel(vectors);current=np.zeros(len(ids),dtype=np.float64)
    chosen=[];remaining=set(range(len(ids)))
    while len(chosen)<k:
        gains={i:float(np.maximum(current,similarity[:,i]).sum()-current.sum()) for i in remaining}
        best=min(remaining,key=lambda i:(-gains[i],id_sort_key(ids[i])))
        chosen.append(best);current=np.maximum(current,similarity[:,best]);remaining.remove(best)
    return [ids[i] for i in chosen]


def farthest_first_prefix(candidate_ids, quality, vectors, *, k):
    ids=list(candidate_ids);_validate(ids,k);q=np.asarray(quality,dtype=np.float64)
    normalized=l2_normalize(vectors)
    if q.shape != (len(ids),) or not np.isfinite(q).all():raise ValueError("invalid quality")
    first=min(range(len(ids)),key=lambda i:(-float(q[i]),id_sort_key(ids[i])))
    chosen=[first];remaining=set(range(len(ids)))-{first}
    while len(chosen)<k:
        radii={i:min(float(np.linalg.norm(normalized[i]-normalized[j])) for j in chosen) for i in remaining}
        best=min(remaining,key=lambda i:(-radii[i],id_sort_key(ids[i])))
        chosen.append(best);remaining.remove(best)
    return [ids[i] for i in chosen]
