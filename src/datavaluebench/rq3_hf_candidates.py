"""Reusable DataValueBench benchmark implementation."""
from collections import Counter
import re
import numpy as np
from scipy import sparse
FIELDS=('task_categories','task_contexts','tags','language','license')
TOKEN=re.compile(r'(?u)\b[\w\-:]+\b')

def structured_matrices(frame):
    matrices=[]
    for field in FIELDS:
        sets=[sorted(set(v)) for v in frame[field+'_values']]
        vocab={v:i for i,v in enumerate(sorted({v for row in sets for v in row}))}
        rows=[];cols=[]
        for i,row in enumerate(sets):
            rows.extend([i]*len(row));cols.extend(vocab[v] for v in row)
        matrix=sparse.csr_matrix((np.ones(len(rows),dtype=np.int32),(rows,cols)),shape=(len(frame),len(vocab)))
        matrices.append((matrix,np.array([len(row) for row in sets],dtype=np.int32),vocab))
    return matrices


def structured_scores(matrices,positions):
    n=matrices[0][0].shape[0];out=np.zeros((len(positions),n),dtype=np.float64)
    for matrix,sizes,vocab in matrices:
        inter=(matrix[positions]@matrix.T).toarray()
        union=sizes[positions,None]+sizes[None,:]-inter
        out+=np.divide(inter,union,out=np.zeros(inter.shape,dtype=np.float64),where=union!=0)
    return out/5


def bm25_matrices(texts):
    tokens=[TOKEN.findall(t.lower()) for t in texts]
    vocab={t:i for i,t in enumerate(sorted({t for row in tokens for t in row}))}
    lengths=np.array([len(row) for row in tokens],dtype=np.float64)
    if np.any(lengths==0):raise ValueError('empty BM25 document')
    rows=[];cols=[];data=[];df=np.zeros(len(vocab),dtype=np.int64)
    for i,row in enumerate(tokens):
        for token,count in Counter(row).items():
            rows.append(i);cols.append(vocab[token]);data.append(float(count));df[vocab[token]]+=1
    query=sparse.csr_matrix((data,(rows,cols)),shape=(len(texts),len(vocab)),dtype=np.float64)
    idf=np.log(1+(len(texts)-df+.5)/(df+.5))
    denominator=query.data+1.5*(.25+.75*lengths[np.repeat(np.arange(len(texts)),np.diff(query.indptr))]/lengths.mean())
    docs=sparse.csr_matrix((idf[query.indices]*query.data*2.5/denominator,query.indices,query.indptr),shape=query.shape)
    return query,docs,{'vocabulary':vocab,'document_frequencies':df,'document_lengths':lengths,'avgdl':float(lengths.mean())}


def exact_topk(scores,self_position,k=1000):
    """Partial selection with exact boundary tie completion; indices are sorted IDs."""
    if k>=len(scores) or not np.isfinite(scores).all():raise ValueError('invalid exact top-k request')
    work=np.asarray(scores).copy();work[self_position]=-np.inf
    cutoff=np.partition(work,len(work)-k)[len(work)-k]
    greater=np.flatnonzero(work>cutoff);ties=np.flatnonzero(work==cutoff)
    chosen=np.concatenate((greater,ties[:k-len(greater)]))
    return chosen[np.lexsort((chosen,-work[chosen]))]
