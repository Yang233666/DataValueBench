import numpy as np
import pytest
from datavaluebench.rq4_ame import subset_design,ame_oracle,PROPORTIONS


def test_frozen_streams_and_prefixes_exact():
    rng=np.random.RandomState(20260903)
    streams=[rng.binomial(1,p,(1000,12)) for p in PROPORTIONS]
    for budget in [100,250,500,1000]:
        actual,state=subset_design(12,20260903,budget)
        assert np.array_equal(actual,np.vstack([s[:budget] for s in streams]))
        assert np.array_equal(state.get_state()[1],rng.get_state()[1])


def test_actual_singleton_response_and_empty_coalition():
    values,design,y,audit=ame_oracle(lambda ids:0.0 if not ids else float(sum(i+1 for i in ids)),12,20260903,100)
    singleton=(design[:,0]==1)&(design.sum(axis=1)==1)
    assert singleton.sum()==5 and np.all(y[singleton]==1)
    assert np.all(y[design.sum(axis=1)==0]==0)
    assert audit['logical_utility_calls']==400 and audit['singleton_first_player_rows_evaluated']==5
    assert np.isfinite(values).all()
