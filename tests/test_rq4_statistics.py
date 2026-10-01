import numpy as np
from datavaluebench.rq4_statistics import agreement_metrics,same_rule_metrics,consequence_curves,bootstrap_indices

def test_same_rule_and_ties():
 ids=list('abcde');a=np.array([1,1,2,3,4]);r=same_rule_metrics(a,a,ids)
 assert r['mae']==0 and r['exact_rank_matches']==5 and r['top_3_overlap_fraction']==1
 r=agreement_metrics(np.ones(5),a,ids);assert r['spearman'] is None and r['correlation_status']=='CONSTANT_VECTOR'

def test_complete_nonmonotone_curves_not_clipped():
 oracle=lambda ids:0 if not ids else (2 if len(ids)==1 else 1)
 curves,auc=consequence_curves(oracle,[0,1]);assert np.array_equal(curves['addition']['normalized'],[0,2,1])
 assert auc['addition_auc_normalized']==1.25 and np.array_equal(curves['low_removal']['raw'],[1,2,0])

def test_bootstrap_repeats_and_keeps_cluster_size():
 a=bootstrap_indices();assert a.shape==(10000,5) and a.min()==0 and a.max()==4 and np.array_equal(a,bootstrap_indices())
