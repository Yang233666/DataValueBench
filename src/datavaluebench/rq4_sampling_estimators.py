"""Reusable DataValueBench benchmark implementation."""
import hashlib
import numpy as np
from datavaluebench.rq4_beta_weights import beta_cardinality_weights
SEEDS=(20260903,20260904,20260905,20260906,20260907)
MC_BUDGETS=(50,100,200,500,1000,2000)
BETA_BUDGETS=(100,200,500,1000,2000)
BANZHAF_BUDGETS=(250,500,1000,2000,5000)

def rng_for(component,seed):
    if seed not in SEEDS:raise ValueError('frozen seed required')
    digest=hashlib.sha256(f'DVBench-v2-RQ4-{component}|{seed}'.encode()).digest()
    return np.random.Generator(np.random.PCG64DXSM(int.from_bytes(digest[:16],'big')))

def permutation_stream(m,seed):
    rng=rng_for('PERMUTATIONS',seed)
    return np.array([rng.permutation(m) for _ in range(2000)],dtype=np.int64)

def msr_stream(m,seed):
    return rng_for('BANZHAF-MSR',seed).binomial(1,.5,size=(5000,m))

def permutation_estimates(oracle,stream):
    n,m=stream.shape
    if n!=2000:raise ValueError('complete frozen permutation stream required')
    weights=np.asarray(beta_cardinality_weights(m));mc=np.zeros(m);beta=np.zeros(m);results={};cache={}
    def utility(ids):
        key=tuple(sorted(ids))
        if key not in cache:cache[key]=float(oracle(key))
        return cache[key]
    for index,permutation in enumerate(stream,1):
        selected=[];previous=utility(())
        for k,player in enumerate(permutation):
            selected.append(int(player));value=utility(selected);delta=value-previous
            mc[player]+=delta;beta[player]+=m*weights[k]*delta;previous=value
        if index in MC_BUDGETS:results[('MonteCarloShapley',index)]=mc.copy()/index
        if index in BETA_BUDGETS:results[('BetaShapley',index)]=beta.copy()/index
    return results,{'logical_utility_calls':n*(m+1),'physical_cache_computations':len(cache),'shared_marginal_reuse_between_methods':True}

def msr_estimates(oracle,stream):
    n,m=stream.shape
    if n!=5000:raise ValueError('complete frozen MSR stream required')
    cache={};values=[]
    for row in stream:
        key=tuple(int(x) for x in np.flatnonzero(row))
        if key not in cache:cache[key]=float(oracle(key))
        values.append(cache[key])
    y=np.asarray(values);results={}
    for budget in BANZHAF_BUDGETS:
        design=stream[:budget];count=design.sum(axis=0);excluded=budget-count
        positive=np.sum(design*y[:budget,None],axis=0);negative=np.sum((1-design)*y[:budget,None],axis=0)
        valid=(count>0)&(excluded>0);value=np.full(m,np.nan)
        value[valid]=positive[valid]/count[valid]-negative[valid]/excluded[valid]
        results[budget]={'values':value,'included_count':count,'excluded_count':excluded,'status':['COMPLETE' if x else 'INSUFFICIENT_MSR_PARTITION' for x in valid]}
    return results,y,{'logical_utility_calls':n,'physical_cache_computations':len(cache)}
