import numpy as np,pytest
from datavaluebench.rq3_kdpp_prepared import PreparedKDPP
from datavaluebench.rq3_ranking import exact_kdpp_sample

def test_exact_all_seeds_depths_with_duplicate_vector_rank_deficiency():
 rng=np.random.RandomState(20260903);x=rng.normal(size=(20,16));x[-4:]=x[:4];ids=[f'Exact/{i}' for i in range(20)];q=np.linspace(0,1,20);prepared=PreparedKDPP(ids,q,x)
 for k in (5,10):
  for seed in range(20260903,20260908):assert prepared.sample(k=k,seed=seed)==exact_kdpp_sample(ids,q,x,k=k,seed=seed)
 assert prepared.rank<20
 with pytest.raises(ValueError,match='exceeds'):prepared.sample(k=20,seed=20260903)
 assert not prepared.kernel.flags.writeable
