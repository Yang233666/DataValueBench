"""Reusable DataValueBench benchmark implementation."""
import numpy as np
from scipy import sparse

def source_batch(transition,source_positions,dangling_positions,*,batch_size=16):
    A=sparse.csr_array(transition,dtype=np.float64);N=A.shape[0]
    if A.shape!=(N,N) or not np.isfinite(A.data).all() or np.any(A.data<0):raise ValueError('Invalid frozen stochastic transition')
    sources=np.asarray(source_positions,dtype=np.int64);dangling=np.asarray(dangling_positions,dtype=np.int64)
    if len(set(sources.tolist()))!=len(sources) or np.any(sources<0) or np.any(sources>=N):raise ValueError('Invalid source identities')
    outputs=[];iterations=[]
    for begin in range(0,len(sources),batch_size):
        positions=sources[begin:begin+batch_size];B=len(positions);p=np.zeros((N,B),dtype=np.float64);p[positions,np.arange(B)]=1.;x=np.full((N,B),1./N,dtype=np.float64);done=np.zeros(B,dtype=bool);result=np.empty_like(x);steps=np.zeros(B,dtype=np.int64)
        for iteration in range(1,1001):
            previous=x;mass=np.array([sum(previous[dangling,j]) for j in range(B)]);x=.85*(A.T@previous+mass[None,:]*p)+(1.-.85)*p
            # Preserve NumPy's contiguous vector summation used by nx.pagerank.
            error=np.array([np.ascontiguousarray(np.absolute(x[:,j]-previous[:,j])).sum() for j in range(B)])
            just=(error<N*1e-12)&~done
            result[:,just]=x[:,just];steps[just]=iteration;done|=just
            if done.all():break
        if not done.all():raise RuntimeError('PPR did not converge within frozen1000iterations')
        outputs.append(result.T.copy());iterations.extend(steps.tolist())
    return np.concatenate(outputs),iterations
