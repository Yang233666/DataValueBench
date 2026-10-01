"""Reusable DataValueBench benchmark implementation."""
import hashlib
import numpy as np

def cluster_draw(analysis_id,regime_id,condition_id,replicate,count):
    if any(not isinstance(x,str) or not x or '|' in x for x in (analysis_id,regime_id,condition_id)):
        raise ValueError('Explicit unambiguous frozen seed identifiers required')
    if type(replicate) is not int or replicate<0 or type(count) is not int or count<1:raise ValueError('Invalid replicate/count')
    payload=f'DVBench-v2-RQ1-BOOTSTRAP|{analysis_id}|{regime_id}|{condition_id}|{replicate}|seed=20260903'
    seed=int.from_bytes(hashlib.sha256(payload.encode('utf-8')).digest()[:16],'big')
    return np.random.Generator(np.random.PCG64DXSM(seed)).integers(0,count,size=count,dtype=np.int64)

class ReplicatedPredictiveMetrics:
    """Integer weights mean literal row replication, including tie multiplicities."""
    def __init__(self,target,prediction):
        self.y=np.asarray(target,dtype=np.float64);self.p=np.asarray(prediction,dtype=np.float64)
        if self.y.ndim!=1 or not len(self.y) or self.y.shape!=self.p.shape or not np.isfinite(self.y).all() or not np.isfinite(self.p).all():raise ValueError('Complete finite pairs required')
        self.error=self.p-self.y
        self.y_unique,self.y_inverse=np.unique(self.y,return_inverse=True)
        self.p_unique,self.p_inverse=np.unique(self.p,return_inverse=True)
    @staticmethod
    def ranks(inverse,groups,weights):
        mass=np.bincount(inverse,weights=weights,minlength=groups)
        ends=np.cumsum(mass)
        return (ends-(mass-1)/2)[inverse]
    def evaluate(self,weights):
        w=np.asarray(weights)
        if w.shape!=self.y.shape or not np.issubdtype(w.dtype,np.integer) or np.any(w<0) or w.sum()<1:raise ValueError('Nonnegative integer replication weights required')
        total=int(w.sum());mean=float(np.dot(w,self.y)/total);variance=float(np.dot(w,(self.y-mean)**2))
        yr=self.ranks(self.y_inverse,len(self.y_unique),w);pr=self.ranks(self.p_inverse,len(self.p_unique),w);mid=(total+1)/2
        yc=yr-mid;pc=pr-mid;yv=float(np.dot(w,yc*yc));pv=float(np.dot(w,pc*pc))
        return {'MAE':float(np.dot(w,np.abs(self.error))/total),
                'R2':float(1-np.dot(w,self.error*self.error)/variance) if np.ptp(self.y[w>0])>0 else None,
                'Spearman':float(np.dot(w,yc*pc)/np.sqrt(yv*pv)) if yv>0 and pv>0 else None}
