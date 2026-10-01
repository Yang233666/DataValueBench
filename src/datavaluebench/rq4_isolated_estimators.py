"""Reusable DataValueBench benchmark implementation."""
import numpy as np
from datavaluebench.rq4_beta_weights import beta_cardinality_weights
from datavaluebench.rq4_sampling_estimators import MC_BUDGETS,BETA_BUDGETS

def isolated_permutation_estimate(oracle,stream,method,budget):
    if method not in ('MonteCarloShapley','BetaShapley'):raise ValueError('unknown permutation estimator')
    if budget not in (MC_BUDGETS if method=='MonteCarloShapley' else BETA_BUDGETS):raise ValueError('unfrozen budget')
    m=stream.shape[1];value=np.zeros(m);cache={};logical=0;weights=beta_cardinality_weights(m)
    def utility(indices):
        nonlocal logical
        logical+=1;key=tuple(sorted(indices))
        if key not in cache:cache[key]=float(oracle(key))
        return cache[key]
    for permutation in stream[:budget]:
        selected=[];previous=utility(())
        for k,player in enumerate(permutation):
            selected.append(int(player));current=utility(selected);marginal=current-previous
            value[player]+=marginal if method=='MonteCarloShapley' else m*weights[k]*marginal
            previous=current
    return value/budget,{'logical_utility_queries':logical,'unique_coalition_queries':len(cache),'physical_cold_cache_computations':len(cache),'cache_hits':logical-len(cache)}
