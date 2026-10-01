import numpy as np,pytest
from scipy import sparse
from datavaluebench.rq2_exact_retrieval import BM25Index,complete_ranking,sparse_dot_scores,dense_dot_scores,exhaustive_maxsim
from datavaluebench.construction import exact_bm25

def test_bm25_independent_reference_repetition_oov_and_complete_tail():
    documents=['Alpha alpha b-c','alpha gamma','b-c gamma','unmatched',''];model=BM25Index().fit(documents)
    for query in ['alpha','alpha alpha b-c','GAMMA alpha missing','missing','']:
        actual=model.score(query);expected=exact_bm25(documents,query);np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12);np.testing.assert_array_equal(actual,model.score(query))
    np.testing.assert_array_equal(model.score('alpha alpha'),2*model.score('alpha'))
    ids=['é/z','Z/2','A/1','a/0','A/2'];positions,_=complete_ranking(ids,model.score('missing'));assert [ids[i] for i in positions]==sorted(ids)
    with pytest.raises(TypeError):model.score({'query':'alpha','qrel':1})

def test_exhaustive_scores_manual_maxsim_and_negative_token_values():
    d=np.array([[1.,0.],[0.,1.],[-1.,0.]],np.float32);q=np.array([[1.,0.],[0.,1.]],np.float32);offsets=np.array([0,2,3]);scores=exhaustive_maxsim(q,d,offsets);np.testing.assert_array_equal(scores,[2.,-1.])
    rng=np.random.default_rng(5);x=rng.normal(size=(100,7));v=rng.normal(size=7);np.testing.assert_allclose(sparse_dot_scores(v[None],sparse.csr_matrix(x)),dense_dot_scores(v,x),rtol=0,atol=1e-14)
    with pytest.raises(ValueError):complete_ranking(['a','a'],[1,0])
    with pytest.raises(ValueError):exhaustive_maxsim(q,d,[0,2,2,3])

def test_random_identity_only_and_fixed_rrf_ties():
    import hashlib
    from datavaluebench.rq2_exact_retrieval import random_order,fixed_bm25_bge_rrf
    ids=['b','a','c'];position,keys=random_order(ids);assert [ids[i] for i in position]==sorted(ids,key=lambda s:(hashlib.sha256(('DVBench-v2-RQ2-RANDOM|20260903|'+s).encode()).hexdigest(),s))
    reversed_ids=ids[::-1];other,_=random_order(reversed_ids);assert [ids[i] for i in position]==[reversed_ids[i] for i in other]
    rank,scores,ra,rb=fixed_bm25_bge_rrf(ids,[0,1,2],[1,0,2]);assert rank.tolist()==[1,0,2];np.testing.assert_array_equal(scores,[1/61+1/62,1/61+1/62,2/63])
    with pytest.raises(ValueError):fixed_bm25_bge_rrf(ids,[0,1],[1,0])

def test_reranker_exact_top100_replacement_and_tail():
    from datavaluebench.rq2_exact_retrieval import rerank_top100_preserve_tail
    ids=[f'doc{i:04d}' for i in range(1000)];rrf=np.arange(999,-1,-1);scores=np.zeros(100);out=rerank_top100_preserve_tail(ids,rrf,scores)
    np.testing.assert_array_equal(out[:100],np.arange(900,1000));np.testing.assert_array_equal(out[100:],rrf[100:])
    with pytest.raises(ValueError):rerank_top100_preserve_tail(ids,rrf,np.zeros(99))
