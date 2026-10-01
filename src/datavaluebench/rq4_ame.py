"""Reusable DataValueBench benchmark implementation."""
import time
from importlib.metadata import version
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import LassoCV

PROPORTIONS=(0.2,0.4,0.6,0.8)
SEEDS=(20260903,20260904,20260905,20260906,20260907)
BUDGETS=(100,250,500,1000)


def subset_design(players,seed,budget):
    if players not in (12,85) or seed not in SEEDS or budget not in BUDGETS:
        raise ValueError('frozen AME players/seed/budget required')
    rng=np.random.RandomState(seed)
    streams=[rng.binomial(1,p,size=(1000,players)) for p in PROPORTIONS]
    return np.vstack([x[:budget] for x in streams]),rng


def fit_design(subsets,responses,rng):
    if version('scikit-learn')!='1.7.2':raise RuntimeError('AME requires scikit-learn==1.7.2')
    normalized=zscore(subsets,axis=1)
    normalized[np.isnan(normalized)]=0
    centered=responses-np.mean(responses)
    estimator=LassoCV(random_state=rng)
    started=time.perf_counter();estimator.fit(X=normalized,y=centered)
    return estimator.coef_,{'fit_seconds':time.perf_counter()-started,'alpha':float(estimator.alpha_),
                           'constructor_defaults':{k:v for k,v in estimator.get_params().items() if k!='random_state'}}


def ame_oracle(oracle,players,seed,budget):
    subsets,rng=subset_design(players,seed,budget)
    cache={};responses=[];singletons=0
    for row in subsets:
        indices=tuple(int(i) for i in row.nonzero()[0])
        if indices not in cache:cache[indices]=float(oracle(indices))
        responses.append(cache[indices]);singletons+=indices==(0,)
    responses=np.asarray(responses,dtype=np.float64)
    if not np.isfinite(responses).all():raise ValueError('nonfinite frozen utility')
    coefficients,fit=fit_design(subsets,responses,rng)
    return coefficients,subsets,responses,{'logical_utility_calls':4*budget,'physical_cache_computations':len(cache),
        'singleton_first_player_rows_evaluated':singletons,'all_rows_receive_actual_oracle_response':True,**fit}
