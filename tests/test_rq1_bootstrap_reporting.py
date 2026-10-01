import numpy as np
import pytest
from datavaluebench.rq1_bootstrap_reporting import interval_report

def test_one_undefined_draw_cannot_be_removed():
    x=np.arange(10000,dtype=float);x[-1]=np.nan
    r=interval_report(x,'DEGENERATE_STATISTIC')
    assert r['ci_status']=='UNDEFINED_BOOTSTRAP_STATISTIC'
    assert r['percentile_95_CI'] is None and r['undefined_draws']==1
    assert np.isnan(x[-1]) and len(x)==10000

def test_all_prescribed_defined_draws_use_frozen_percentiles():
    x=np.arange(2000,dtype=float)
    assert interval_report(x,None)['percentile_95_CI']==np.percentile(x,[2.5,97.5],method='linear').tolist()

def test_infinity_is_not_silently_reclassified_as_scientific_undefined():
    with pytest.raises(ValueError):interval_report([1,np.inf],'BAD_NUMERICS')
