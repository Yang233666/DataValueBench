import numpy as np,pandas as pd,pytest
from datavaluebench.rq1_estimators import baseline,Representation,D_FEATURES,choose_unique_config

def toy():
 return pd.DataFrame({'observation_uid':['b','a','c'],'task_id':[1,1,2],'flow_id':[10,20,10],'task_dataset_uid':['t1','t1','t2'],'flow_uid':['f10','f20','f10'],'dataset_id':[1,1,2],'predictive_accuracy':[.2,.6,.9]})
def test_baselines_reference_cold_fallback_and_target_boundary():
 train=toy();test=pd.DataFrame({'task_id':[1,9,1],'flow_id':[10,10,99]});mean=np.mean([.2,.6,.9]);expected={'GM':[mean]*3,'TD_MEAN':[.4,mean,.4],'FLOW_MEAN':[.55,.55,mean],'FREQ':[.2,mean,mean],'TF_ADD':[.3,mean-.1,.4]}
 for method,values in expected.items():
  pred,status,reason=baseline(method,train,test);np.testing.assert_allclose(pred,values,rtol=0,atol=1e-15);assert np.array_equal(pred,baseline(method,train.iloc[::-1],test)[0])
 with pytest.raises(ValueError):baseline('GM',train,test.assign(predictive_accuracy=0))
def test_representation_training_unique_identity_and_vocabulary():
 train=toy();features=pd.DataFrame({'dataset_id':[1,2,3],**{k:[1.,3.,1e9] for k in D_FEATURES}});texts=pd.DataFrame({'flow_id':[10,20,99],'canonical_text':['alpha beta','gamma delta','heldout sentinel']});vectors=np.eye(3,dtype=np.float32);r=Representation('D+W-TFIDF',features,texts,[10,20,99],vectors).fit(train);assert np.array_equal(r.medians,np.full(10,2.));assert np.array_equal(r.scaler.mean_,np.full(10,2.));assert 'heldout' not in r.vectorizer.vocabulary_;assert r.transform(train).shape[0]==3
 i=Representation('ID',features,texts,[10,20,99],vectors).fit(train);unknown=train.iloc[:1].assign(task_dataset_uid='unknown',flow_uid='unknown');assert i.transform(unknown).nnz==0;assert len(i.encoder.categories_)==2
 with pytest.raises(ValueError,match='tie priority'):choose_unique_config([{'macro_inner_mae':.1},{'macro_inner_mae':.1+1e-13}])
