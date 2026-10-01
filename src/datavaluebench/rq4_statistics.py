"""Reusable DataValueBench benchmark implementation."""
import hashlib,itertools,json
import numpy as np
from scipy.stats import spearmanr,kendalltau

def order(values,ids):return np.array(sorted(range(len(ids)),key=lambda i:(-float(values[i]),ids[i])),dtype=np.int64)
def ranks(values,ids):
 result=np.empty(len(ids),dtype=np.int64);result[order(values,ids)]=np.arange(1,len(ids)+1);return result

def agreement_metrics(a,b,ids,ks=(5,10)):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 if a.shape!=b.shape or len(a)!=len(ids) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('finite aligned vectors required')
 constant=np.ptp(a)==0 or np.ptp(b)==0
 ra,rb=ranks(a,ids),ranks(b,ids);oa,ob=order(a,ids),order(b,ids);delta=abs(ra-rb)
 result={'spearman':None if constant else float(spearmanr(a,b).statistic),'kendall_tau_b':None if constant else float(kendalltau(a,b,variant='b').statistic),'correlation_status':'CONSTANT_VECTOR' if constant else 'DEFINED','mean_absolute_rank_change':float(delta.mean()),'maximum_absolute_rank_change':int(delta.max()),'exact_rank_matches':int(np.sum(ra==rb)),'sign_agreement':float(np.mean(np.sign(a)==np.sign(b))),'sign_errors':int(np.sum(np.sign(a)!=np.sign(b))),'strict_zero_crossings':int(np.sum(a*b<0)),'pairwise_rank_reversals':sum((ra[i]-ra[j])*(rb[i]-rb[j])<0 for i,j in itertools.combinations(range(len(ids)),2))}
 result['pairwise_rank_reversals']=int(result['pairwise_rank_reversals'])
 for k in ks:
  if k>len(ids):continue
  for name,x,y in [('top',oa[:k],ob[:k]),('bottom',oa[-k:],ob[-k:])]:
   intersection=len(set(x)&set(y));result[f'{name}_{k}_intersection_count']=intersection;result[f'{name}_{k}_overlap_fraction']=intersection/k
 return result

def same_rule_metrics(approx,exact,ids):
 a=np.asarray(approx);e=np.asarray(exact);error=abs(a-e)
 return dict(agreement_metrics(a,e,ids,ks=(3,5)),mae=float(error.mean()),median_absolute_error=float(np.median(error)),rmse=float(np.sqrt(np.mean(error**2))),maximum_absolute_error=float(error.max()))

def bootstrap_indices():
 out=[]
 for b in range(10000):
  payload={'master_seed':20260903,'namespace':'DVBench-v2-RQ4-EXACTNESS-BOOTSTRAP-v1.0','replicate':b}
  seed=int.from_bytes(hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()[:16],'big')
  out.append(np.random.Generator(np.random.PCG64DXSM(seed)).integers(0,5,size=5))
 return np.asarray(out,dtype=np.int64)

def consequence_curves(oracle,ranking):
 ranking=list(map(int,ranking));m=len(ranking)
 if len(set(ranking))!=m:raise ValueError('ranking must preserve each player exactly once')
 add=np.array([oracle(ranking[:k]) for k in range(m+1)],dtype=np.float64)
 low=add[::-1].copy();high=np.array([oracle(ranking[k:]) for k in range(m+1)],dtype=np.float64)
 full=float(add[-1]);curves={};aucs={}
 for name,y in [('addition',add),('low_removal',low),('high_removal',high)]:
  norm=y/full if full>0 else np.full_like(y,np.nan);curves[name]={'raw':y,'normalized':norm}
  raw=float(np.sum((y[:-1]+y[1:])/2)/m)
  aucs[name+'_auc_raw']=raw;aucs[name+'_auc_normalized']=raw/full if full>0 else None
 return curves,dict(aucs,full_utility=full,normalization_status='DEFINED' if full>0 else 'NONPOSITIVE_FULL_UTILITY')
