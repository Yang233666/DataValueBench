"""Reusable DataValueBench benchmark implementation."""
import numpy as np
from numba import njit
from datavaluebench.reference import U1Reference
@njit(cache=True,fastmath=False,parallel=False)
def _evaluate(selected,folds,counts,sums,totals,observations,flow_positions,values,offsets):
    if len(selected)==0:return 0.
    nplayers,nflows=counts.shape;total_utility=0.
    for fold in range(5):
        train_count=0;train_sum=0.;fc=np.zeros(nflows,np.int64);fs=np.zeros(nflows,np.float64)
        for i in selected:
            if folds[i]==fold:continue
            train_count+=observations[i];train_sum+=totals[i]
            for j in range(nflows):fc[j]+=counts[i,j];fs[j]+=sums[i,j]
        if train_count==0:continue
        global_mean=train_sum/train_count
        means=np.empty(nflows,np.float64)
        for j in range(nflows):means[j]=fs[j]/fc[j] if fc[j]>0 else global_mean
        for i in range(nplayers):
            if folds[i]!=fold:continue
            error=0.
            for row in range(offsets[i],offsets[i+1]):error+=abs(values[row]-means[flow_positions[row]])
            total_utility+=1.-error/observations[i]
    return total_utility/nplayers
class ExactU1Oracle:
    def __init__(self,records,players):
        self.reference=U1Reference(records,players);self.players=self.reference.players;e=[self.reference.evidence[x] for x in self.players]
        self.folds=np.array([x.fold_id for x in e],np.int64);self.counts=np.array([x.flow_counts for x in e]);self.sums=np.array([x.flow_sums for x in e]);self.totals=np.array([x.total for x in e]);self.observations=np.array([x.count for x in e],np.int64)
        self.offsets=np.concatenate(([0],np.cumsum(self.observations)));self.values=np.concatenate([x.values for x in e]);self.positions=np.concatenate([np.array([self.reference.flow_position[int(f)] for f in x.flow_ids],np.int64) for x in e])
    def __call__(self,indices):
        selected=np.array(sorted(set(int(i) for i in indices)),np.int64)
        if np.any(selected<0) or np.any(selected>=len(self.players)):raise ValueError('unknown player position')
        return float(_evaluate(selected,self.folds,self.counts,self.sums,self.totals,self.observations,self.positions,self.values,self.offsets))
