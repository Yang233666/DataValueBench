import hashlib
import numpy as np
import pytest
from scipy.stats import spearmanr
from datavaluebench.rq1_bootstrap_mechanics import ReplicatedPredictiveMetrics,cluster_draw

def test_exact_literal_replication_reference_random_and_tie_heavy():
    rng=np.random.default_rng(44)
    for tied in (False,True):
        for size in (3,10,53,201):
            y=rng.normal(size=size);p=rng.normal(size=size)
            if tied:y=np.round(y);p=np.round(p)
            metric=ReplicatedPredictiveMetrics(y,p)
            for repeat in range(12):
                w=rng.integers(0,8,size=size,dtype=np.int64)
                if not w.sum():w[0]=1
                yy=np.repeat(y,w);pp=np.repeat(p,w);actual=metric.evaluate(w)
                assert actual['MAE']==pytest.approx(np.abs(yy-pp).mean(),abs=1e-12,rel=0)
                var=np.square(yy-yy.mean()).sum()
                if var:assert actual['R2']==pytest.approx(1-np.square(pp-yy).sum()/var,abs=1e-12,rel=0)
                else:assert actual['R2'] is None
                if np.ptp(yy) and np.ptp(pp):assert actual['Spearman']==pytest.approx(spearmanr(yy,pp).statistic,abs=1e-12,rel=0)
                else:assert actual['Spearman'] is None

def test_workflow_seed_matches_reference_uniform_choice_and_order():
    payload=b'DVBench-v2-RQ1-BOOTSTRAP|WORKFLOW_SELECTION|GT|GT80|7|seed=20260903'
    seed=int.from_bytes(hashlib.sha256(payload).digest()[:16],'big')
    expected=np.random.Generator(np.random.PCG64DXSM(seed)).choice(6,size=6,replace=True)
    assert np.array_equal(cluster_draw('WORKFLOW_SELECTION','GT','GT80',7,6),expected)
    cluster_draw('WORKFLOW_SELECTION','GD','GD',8,73)
    assert np.array_equal(cluster_draw('WORKFLOW_SELECTION','GT','GT80',7,6),expected)

def test_missing_identity_fractional_weights_and_zero_mass_rejected():
    with pytest.raises(ValueError):cluster_draw('','G0','G0',0,3)
    metric=ReplicatedPredictiveMetrics([0.,1.],[1.,0.])
    for weights in ([0,0],[1.,.5],[-1,2]):
        with pytest.raises(ValueError):metric.evaluate(weights)
