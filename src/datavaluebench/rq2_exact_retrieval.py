"""Reusable DataValueBench benchmark implementation."""
import math,re
from collections import Counter
import numpy as np
from scipy import sparse
TOKEN=re.compile(r'(?u)\b[\w\-:]+\b')

def complete_ranking(dataset_ids,scores):
    ids=np.asarray(dataset_ids,dtype=str);scores=np.asarray(scores)
    if scores.shape!=(len(ids),) or len(set(ids))!=len(ids) or not np.isfinite(scores).all():raise ValueError('incomplete/nonfinite scores or duplicate identities')
    positions=np.lexsort((ids,-scores));return positions,scores[positions]

class BM25Index:
    def fit(self,documents):
        if any(not isinstance(t,str) for t in documents):raise TypeError('documents must be exact strings')
        tokens=[TOKEN.findall(t.lower()) for t in documents];self.lengths=np.array(list(map(len,tokens)),dtype=np.float64);self.avgdl=float(self.lengths.mean());self.rows=len(tokens)
        if not self.rows or self.avgdl<=0:raise ValueError('undefined empty BM25 corpus')
        self.vocabulary={t:i for i,t in enumerate(sorted({x for row in tokens for x in row}))};rows=[];cols=[];counts=[]
        for i,row in enumerate(tokens):
            for token,count in Counter(row).items():rows.append(i);cols.append(self.vocabulary[token]);counts.append(count)
        tf=sparse.csc_matrix((np.asarray(counts,np.float64),(rows,cols)),shape=(self.rows,len(self.vocabulary)));self.df=np.diff(tf.indptr);self.idf=np.array([math.log(1.+(self.rows-int(df)+.5)/(int(df)+.5)) for df in self.df]);weight=tf.copy()
        for j in range(tf.shape[1]):
            sl=slice(tf.indptr[j],tf.indptr[j+1]);d=tf.indices[sl];v=tf.data[sl];weight.data[sl]=self.idf[j]*v*2.5/(v+1.5*(1.-.75+.75*self.lengths[d]/self.avgdl))
        self.weights=weight;return self
    def score(self,query):
        if not isinstance(query,str):raise TypeError('TrackB accepts exact NL string only')
        output=np.zeros(self.rows,np.float64)
        for token in TOKEN.findall(query.lower()):
            j=self.vocabulary.get(token)
            if j is None:continue
            sl=slice(self.weights.indptr[j],self.weights.indptr[j+1]);output[self.weights.indices[sl]]+=self.weights.data[sl]
        return output

def sparse_dot_scores(query,documents):
    q=sparse.csr_matrix(query)
    if q.shape!=(1,documents.shape[1]):raise ValueError('query dimension mismatch')
    result=(q@documents.T).toarray().ravel()
    if len(result)!=documents.shape[0] or not np.isfinite(result).all():raise ValueError('incomplete scores')
    return result

def dense_dot_scores(query,documents):
    q=np.asarray(query)
    if q.shape!=(documents.shape[1],):raise ValueError('query dimension mismatch')
    result=documents@q
    if len(result)!=len(documents) or not np.isfinite(result).all():raise ValueError('incomplete scores')
    return result

def exhaustive_maxsim(query,document_vectors,offsets):
    """Full uncompressed ragged document scan, preserving every document score."""
    q=np.asarray(query);d=np.asarray(document_vectors);offsets=np.asarray(offsets)
    if q.ndim!=2 or d.ndim!=2 or q.shape[1]!=d.shape[1] or offsets.ndim!=1 or offsets[0]!=0 or offsets[-1]!=len(d) or np.any(np.diff(offsets)<=0):raise ValueError('invalid complete ragged token representation')
    out=np.empty(len(offsets)-1,dtype=np.result_type(q,d))
    for i,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):out[i]=np.max(q@d[int(a):int(b)].T,axis=1).sum()
    if not np.isfinite(out).all():raise ValueError('nonfinite MaxSim score')
    return out

def random_order(dataset_ids):
    import hashlib
    if len(set(dataset_ids))!=len(dataset_ids):raise ValueError('duplicate corpus identities')
    keys=[hashlib.sha256(('DVBench-v2-RQ2-RANDOM|20260903|'+uid).encode('utf-8')).hexdigest() for uid in dataset_ids]
    return sorted(range(len(dataset_ids)),key=lambda i:(keys[i],dataset_ids[i])),keys

def fixed_bm25_bge_rrf(dataset_ids,bm25_positions,bge_positions):
    n=len(dataset_ids);a=np.asarray(bm25_positions);b=np.asarray(bge_positions)
    if len(set(dataset_ids))!=n or a.shape!=(n,) or b.shape!=(n,) or not np.array_equal(np.sort(a),np.arange(n)) or not np.array_equal(np.sort(b),np.arange(n)):raise ValueError('RRF requires both complete corpus rankings')
    ra=np.empty(n,np.int64);rb=np.empty(n,np.int64);ra[a]=np.arange(1,n+1);rb[b]=np.arange(1,n+1);scores=1./(60+ra)+1./(60+rb);positions=np.lexsort((np.asarray(dataset_ids,dtype=str),np.minimum(ra,rb),-scores));return positions,scores,ra,rb

def rerank_top100_preserve_tail(dataset_ids,rrf_positions,top100_scores):
    ids=np.asarray(dataset_ids,dtype=str);first=np.asarray(rrf_positions);scores=np.asarray(top100_scores)
    if len(ids)<100 or len(set(ids))!=len(ids) or not np.array_equal(np.sort(first),np.arange(len(ids))) or scores.shape!=(100,) or not np.isfinite(scores).all():raise ValueError('frozen reranker requires100uniqueheadscores and completeRRFinput')
    head=first[:100];reranked=head[np.lexsort((ids[head],-scores))];result=np.concatenate([reranked,first[100:]])
    assert np.array_equal(result[100:],first[100:]);return result
