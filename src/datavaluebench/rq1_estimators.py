"""Reusable DataValueBench benchmark implementation."""
import json
import numpy as np,pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler,normalize,OneHotEncoder
from sklearn.linear_model import Ridge
from importlib.metadata import version
from datavaluebench.construction import global_mean,grouped_mean,task_flow_additive_mean
BASELINES={'GM':'GlobalMean','TD_MEAN':'TaskDatasetMean','FLOW_MEAN':'FlowMean','TF_ADD':'TaskFlowAdditiveMean','FREQ':'FrequencyBaseline'}
ALPHAS=(.0001,.001,.01,.1,1.,10.,100.,1000.)
D_FEATURES=('log1p_NumberOfInstances','log1p_NumberOfFeatures','log1p_NumberOfClasses','NumericFeatureFraction','SymbolicFeatureFraction','MissingValueFraction','InstancesWithMissingFraction','MajorityClassFraction','MinorityClassFraction','log1p_MaxNominalAttDistinctValues')
def baseline(condition,train,test):
 train=train.sort_values('observation_uid',kind='stable')
 if not len(train):raise ValueError('empty legal training set')
 if 'predictive_accuracy' in test:raise ValueError('test outcomes may not enter estimator')
 if condition=='GM':values=global_mean(train,test);reason=np.full(len(test),None,dtype=object);status=np.full(len(test),'NATIVE',dtype=object)
 elif condition=='TF_ADD':
  values,raw=task_flow_additive_mean(train,test);status=np.where(raw=='none','NATIVE',np.where(raw=='unseen_task_global_mean+unseen_flow_zero_effect','FALLBACK','MIXED_COMPONENT_FALLBACK'));reason=np.where(raw=='none',None,raw)
 else:
  keys={'TD_MEAN':['task_id'],'FLOW_MEAN':['flow_id'],'FREQ':['task_id','flow_id']}[condition];values,raw=grouped_mean(train,test,keys);status=np.where(raw=='none','NATIVE','FALLBACK');reason=np.where(raw=='none',None,{'TD_MEAN':'GLOBAL_MEAN_UNSEEN_TASKDATASET','FLOW_MEAN':'GLOBAL_MEAN_UNSEEN_FLOW','FREQ':'GLOBAL_MEAN_UNSEEN_TASKFLOW_PAIR'}[condition])
 return values,status,reason
class Representation:
 def __init__(self,condition,dataset_features,workflow_text,minilm_ids,minilm_vectors):
  self.condition=condition;self.features=dataset_features.set_index('dataset_id')[list(D_FEATURES)];self.text=workflow_text.set_index('flow_id').canonical_text;self.minilm={int(uid):i for i,uid in enumerate(minilm_ids)};self.vectors=minilm_vectors
 def fit(self,train):
  if version('scikit-learn')!='1.7.2':raise RuntimeError('frozen scikit-learn1.7.2 required')
  self.train_dataset_ids=sorted(train.dataset_id.unique());self.train_flow_ids=sorted(train.flow_id.unique())
  if self.condition=='ID':self.encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float64).fit(train[['task_dataset_uid','flow_uid']]);return self
  if self.condition.startswith('D'):
   raw=self.features.reindex(self.train_dataset_ids).to_numpy();self.all_missing=np.isnan(raw).all(axis=0);self.medians=np.array([0. if self.all_missing[j] else np.nanmedian(raw[:,j]) for j in range(10)]);filled=np.where(np.isnan(raw),self.medians,raw);self.scaler=StandardScaler().fit(filled)
  if 'TFIDF' in self.condition:
   texts=sorted(set(self.text.reindex(self.train_flow_ids)));self.vectorizer=TfidfVectorizer(lowercase=True,analyzer='word',token_pattern=r'(?u)\b\w\w+\b',ngram_range=(1,2),min_df=1,max_df=1.,use_idf=True,smooth_idf=True,sublinear_tf=True,norm='l2',max_features=None,dtype=np.float64).fit(texts)
  return self
 def transform(self,frame):
  if self.condition=='ID':return self.encoder.transform(frame[['task_dataset_uid','flow_uid']])
  blocks=[]
  if self.condition.startswith('D'):
   raw=self.features.reindex(frame.dataset_id).to_numpy();filled=np.where(np.isnan(raw),self.medians,raw);blocks.append(normalize(self.scaler.transform(filled),norm='l2'))
  if 'TFIDF' in self.condition:blocks.append(self.vectorizer.transform(self.text.reindex(frame.flow_id).tolist()))
  if 'MINILM' in self.condition:blocks.append(self.vectors[[self.minilm[int(x)] for x in frame.flow_id]].astype(np.float64))
  if not blocks:raise ValueError('unknown frozen representation')
  return sparse.hstack(blocks,format='csr') if any(sparse.issparse(b) for b in blocks) else np.hstack(blocks)
def ridge(alpha):
 if alpha not in ALPHAS:raise ValueError('outside frozen alpha grid')
 return Ridge(alpha=alpha,fit_intercept=True,solver='lsqr',tol=1e-4,max_iter=None)
def choose_unique_config(records):
 best=min(r['macro_inner_mae'] for r in records);tied=[r for r in records if abs(r['macro_inner_mae']-best)<=1e-12]
 if len(tied)!=1:raise ValueError('PARAMETER_UNSPECIFIED: RQ1 canonical config_id tie priority absent from Method Card; multiple tied configurations require scientific authority')
 return tied[0]
def config_id(alpha):return json.dumps({'alpha':alpha},sort_keys=True,separators=(',',':'))
