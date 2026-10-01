"""Reusable DataValueBench benchmark implementation."""
import numpy as np

def interval_report(draws, reason):
    x=np.asarray(draws,dtype=np.float64)
    if x.ndim!=1 or not len(x) or np.isinf(x).any():
        raise ValueError('Prescribed vector required; infinite values are an engineering error')
    defined=int(np.isfinite(x).sum());missing=len(x)-defined
    return {'ci_status':'UNDEFINED_BOOTSTRAP_STATISTIC' if missing else 'DEFINED',
            'percentile_95_CI':None if missing else np.percentile(x,[2.5,97.5],method='linear').tolist(),
            'total_prescribed_draws':len(x),'defined_draws':defined,'undefined_draws':missing,
            'undefined_fraction':missing/len(x),'undefined_reason':reason if missing else None}
