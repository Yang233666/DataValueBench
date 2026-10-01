"""Reusable DataValueBench benchmark implementation."""
from itertools import product
from importlib.metadata import version
import json
import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from datavaluebench.rq1_estimators import Representation

LGBM_GRID=tuple(dict(zip(('n_estimators','learning_rate','num_leaves','min_child_samples'),v)) for v in product((200,500),(.03,.10),(15,31,63),(10,30)))
RF_GRID=tuple(dict(zip(('n_estimators','max_depth','min_samples_leaf','max_features'),v)) for v in product((300,600),(None,20),(1,5,20),(1.0,'sqrt')))
FINAL_SEEDS=(20260903,20260904,20260905,20260906,20260907)

def configuration_id(config):
    return json.dumps(config,sort_keys=True,separators=(',',':'))

class TreeRepresentation(Representation):
    """Uses the same frozen TFIDF as Ridge, but never scales dataset features."""
    def __init__(self,condition,dataset_features,workflow_text,minilm_ids,minilm_vectors,hybrid=False):
        super().__init__(condition,dataset_features,workflow_text,minilm_ids,minilm_vectors)
        self.hybrid=hybrid
    def fit(self,train):
        if 'predictive_accuracy' in train:
            raise ValueError('representation fitting accepts identities/features only')
        # Parent fits the training-only medians and TFIDF. Its scaler is unused.
        super().fit(train)
        if self.hybrid:
            self.encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float32).fit(train[['task_dataset_uid','flow_uid']])
        return self
    def transform(self,frame):
        if 'predictive_accuracy' in frame:
            raise ValueError('representation transform accepts identities/features only')
        blocks=[]
        if self.hybrid:
            blocks.append(self.encoder.transform(frame[['task_dataset_uid','flow_uid']]))
        if self.condition.startswith('D'):
            raw=self.features.reindex(frame.dataset_id).to_numpy()
            blocks.append(np.where(np.isnan(raw),self.medians,raw).astype(np.float32))
        if 'TFIDF' in self.condition:
            blocks.append(self.vectorizer.transform(self.text.reindex(frame.flow_id).tolist()))
        if 'MINILM' in self.condition:
            blocks.append(self.vectors[[self.minilm[int(x)] for x in frame.flow_id]])
        if not blocks:raise ValueError('empty representation')
        return sparse.hstack(blocks,format='csr') if any(sparse.issparse(b) for b in blocks) else np.hstack(blocks)

def lightgbm(config,threads=1):
    if version('lightgbm')!='4.6.0':raise RuntimeError('frozen LightGBM4.6.0 required')
    if config not in LGBM_GRID:raise ValueError('outside frozen grid')
    from lightgbm import LGBMRegressor
    return LGBMRegressor(boosting_type='gbdt',objective='regression',max_depth=-1,subsample=1.,subsample_freq=0,colsample_bytree=1.,reg_alpha=0.,reg_lambda=0.,deterministic=True,force_col_wise=True,zero_as_missing=False,verbosity=-1,n_jobs=threads,random_state=20260903,**config)

def random_forest(config,seed,threads=1):
    if version('scikit-learn')!='1.7.2':raise RuntimeError('frozen sklearn1.7.2 required')
    if config not in RF_GRID or seed not in FINAL_SEEDS:raise ValueError('outside frozen grid/seeds')
    return RandomForestRegressor(criterion='squared_error',bootstrap=True,oob_score=False,min_samples_split=2,max_leaf_nodes=None,ccp_alpha=0.,n_jobs=threads,random_state=seed,**config)
