"""Reusable DataValueBench benchmark implementation."""
import numpy as np
from scipy.stats import spearmanr,pearsonr,kendalltau

def predictive_metrics(target,prediction):
    y=np.asarray(target,dtype=np.float64);p=np.asarray(prediction,dtype=np.float64)
    if y.ndim!=1 or y.shape!=p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Complete finite target/prediction pairs required; no silent row deletion')
    error=p-y;variance=float(np.square(y-y.mean()).sum());both_vary=len(y)>1 and np.ptp(y)>0 and np.ptp(p)>0
    result={'MAE':float(np.abs(error).mean()),'R2':float(1-np.square(error).sum()/variance) if np.ptp(y)>0 else None,
            'RMSE':float(np.sqrt(np.square(error).mean())),
            'Spearman':float(spearmanr(y,p).statistic) if both_vary else None,
            'Pearson':float(pearsonr(y,p).statistic) if both_vary else None,
            'Kendall':float(kendalltau(y,p,variant='b',method='auto').statistic) if both_vary else None}
    for name,value in result.items():
        if value is not None and not np.isfinite(value):result[name]=None
    return {'rows':len(y),'values':result,'status':{name:'DEFINED' if value is not None else 'UNDEFINED' for name,value in result.items()},
            'correlation_convention':'Pinned SciPy: tie-aware Spearman, Pearson, ordinary Kendall tau-b; undefined is not zero'}

def status_decomposition(target,prediction,statuses,reasons):
    from collections import Counter
    y=np.asarray(target,dtype=float);p=np.asarray(prediction,dtype=float);status=np.asarray(statuses,dtype=object)
    if y.shape!=p.shape or len(status)!=len(y) or len(reasons)!=len(y) or not len(y):raise ValueError('Aligned complete status rows required')
    allowed={'NATIVE','FALLBACK','MIXED_COMPONENT_FALLBACK','UNKNOWN_ID_ZERO_ENCODING'}
    if not set(status)<=allowed or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('This complete-output evaluator cannot silently remove undefined rows')
    native=status=='NATIVE';degraded=~native;counts=Counter(status)
    return {'defined_rate':1.,'native_rate':float(native.mean()),'degraded_rate':float(degraded.mean()),'undefined_rate':0.,
            'exact_status_counts':dict(counts),'exact_status_rates':{k:v/len(y) for k,v in counts.items()},
            'fallback_reason_counts':dict(Counter(x for x in reasons if isinstance(x,str) and x)),
            'subsets':{name:{'rows':int(mask.sum()),'MAE':float(np.abs(p[mask]-y[mask]).mean()) if mask.any() else None,
                'RMSE':float(np.sqrt(np.square(p[mask]-y[mask]).mean())) if mask.any() else None} for name,mask in [('native',native),('degraded',degraded)]}}
