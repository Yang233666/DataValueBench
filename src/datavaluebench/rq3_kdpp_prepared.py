"""Reusable DataValueBench benchmark implementation."""
import numpy as np
from .rq3_ranking import kdpp_kernel,id_sort_key

class PreparedKDPP:
    def __init__(self,candidate_ids,quality,vectors):
        self.ids=list(candidate_ids);self.quality=np.asarray(quality,dtype=np.float64).copy()
        if len(self.ids)!=len(set(self.ids)) or self.quality.shape!=(len(self.ids),):
            raise ValueError('Complete unique candidates and aligned quality required')
        self.kernel,self.audit=kdpp_kernel(self.quality,vectors)
        self.rank=int(np.linalg.matrix_rank(self.kernel))
        self.kernel.setflags(write=False)
    def sample(self,*,k,seed):
        if k not in (5,10,20):raise ValueError('k is outside the frozen set')
        if self.rank<k:raise ValueError('BLOCKED: k exceeds validated DPP kernel rank')
        from dppy.finite_dpps import FiniteDPP
        sampler=FiniteDPP('likelihood',L=self.kernel.copy())
        indices=sampler.sample_exact_k_dpp(size=k,mode='GS',random_state=np.random.RandomState(seed))
        if len(indices)!=len(set(indices)) or len(indices)!=k:raise RuntimeError('Invalid exact sample')
        indices=sorted(indices,key=lambda i:(-float(self.quality[i]),id_sort_key(self.ids[i])))
        return [self.ids[i] for i in indices],dict(self.audit)
