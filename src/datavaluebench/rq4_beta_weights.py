"""Reusable DataValueBench benchmark implementation."""
from functools import lru_cache
import numpy as np
from scipy.special import beta
@lru_cache(maxsize=None)
def beta_cardinality_weights(m):
    if not isinstance(m,int) or m<1:raise ValueError('positive game size required')
    raw=[beta(k+1,m-k-1+4)/beta(k+1,m-k) for k in range(m)]
    weights=np.array(raw)/np.sum(raw)
    return tuple(float(x) for x in weights)
