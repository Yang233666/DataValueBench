"""Equivalent forest tree construction in parallel; deterministic serial prediction."""
from datavaluebench.rq1_tree_estimators import random_forest

def fit_parallel_forest(config,seed,x,y,threads):
    if threads<1:raise ValueError('positive execution thread count required')
    model=random_forest(config,seed,threads=threads).fit(x,y)
    # sklearn's parallel prediction reduces trees in arrival order. Keep the
    # frozen deterministic sequential sum; only independent tree fits parallelize.
    model.n_jobs=1
    return model
