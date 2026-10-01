"""Reusable DataValueBench benchmark implementation."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from datavaluebench.rq1_estimators import D_FEATURES
PARAMETERS=dict(perform_qr=True,max_outer_iter=100,max_inner_iter_init=1000,max_inner_iter_final=1000,lsqr_inner_init_tol=1e-15,lsqr_smart_tol=True,lsqr_smart_obj_min=1e-5,init_option=0,stop_relRes=-1,stop_relDiff=-1,stop_relResDiff=-1,verbose=False)
REFERENCE_SHA='c9637ef529d22d93d4336bb75457ab062fbd722cafbf2a792915c302393b9a1c'

def load_reference(root):
    root=Path(root);manifest=json.loads((root/'PINNED_SOURCE_MANIFEST.json').read_text())
    for name,record in manifest.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==record['sha256']
    path=root/'Python/algorithms/AltMin.py';assert hashlib.sha256(path.read_bytes()).hexdigest()==REFERENCE_SHA
    source_root=str(root/'Python')
    if source_root not in sys.path:sys.path.insert(0,source_root)
    spec=importlib.util.spec_from_file_location('dvb_pinned_altmin',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def solve(reference,X,omega,rank,A,B,seed):
    if rank not in (1,2,4,8):raise ValueError('outside frozen rank grid')
    if seed not in range(20260903,20260908):raise ValueError('outside frozen final seeds')
    assert np.isfinite(X).all() and np.isfinite(A).all() and np.isfinite(B).all()
    assert np.all(X[omega==0]==0) and omega.sum()>0
    capture={};lsqr_iterations=[];old_profile=sys.getprofile();old_rng=np.random.get_state()
    def profile(frame,event,arg):
        if event=='return' and frame.f_code is reference.AltMin.__code__:
            capture['U']=frame.f_locals['U_best'].copy();capture['V']=frame.f_locals['V_best'].copy()
        if event=='return' and frame.f_code is reference.sp_linalg.lsqr.__code__:
            lsqr_iterations.append(int(arg[2]))
    start=time.perf_counter()
    try:
        np.random.seed(seed);sys.setprofile(profile)
        result,iterations,converged,residuals=reference.AltMin(X,omega,rank,A,B,**PARAMETERS)
    finally:
        sys.setprofile(old_profile);np.random.set_state(old_rng)
    reconstructed=A@capture['U']@capture['V'].T@B.T
    assert np.array_equal(result,reconstructed) and np.isfinite(result).all()
    assert iterations==100 and not converged and len(lsqr_iterations)==200
    return result,{'rank':rank,'seed':seed,'outer_iterations':iterations,'lsqr_iterations':lsqr_iterations,'residuals':residuals[1:],'best_training_residual':min(residuals[1:]),'independent_factor_reconstruction_bitwise_equal':True,'wall_seconds':time.perf_counter()-start,'U':capture['U'],'V':capture['V']}

class IMCInputs:
    def __init__(self,cohort_identities,dataset_features,minilm_ids,minilm_vectors):
        if 'predictive_accuracy' in cohort_identities:raise ValueError('universe accepts identities only')
        identity=cohort_identities[['task_dataset_uid','dataset_id']].drop_duplicates().sort_values('task_dataset_uid');flow=cohort_identities[['flow_uid','flow_id']].drop_duplicates().sort_values('flow_uid')
        assert len(identity)==85 and len(flow)==1005
        self.tasks=identity.task_dataset_uid.tolist();self.flows=flow.flow_uid.tolist();self.task_positions={x:i for i,x in enumerate(self.tasks)};self.flow_positions={x:i for i,x in enumerate(self.flows)}
        self.rawA=dataset_features.set_index('dataset_id').reindex(identity.dataset_id)[list(D_FEATURES)].to_numpy(dtype=np.float64)
        positions={int(x):i for i,x in enumerate(minilm_ids)};self.rawB=np.asarray(minilm_vectors[[positions[int(x)] for x in flow.flow_id]],dtype=np.float64)
    def fit(self,train):
        grouped=train.groupby(['task_dataset_uid','flow_uid'],sort=True).predictive_accuracy.agg(['mean','size']);self.X=np.zeros((85,1005));self.omega=np.zeros_like(self.X,dtype=np.int8);self.counts=np.zeros_like(self.X,dtype=np.int64)
        for (t,f),r in grouped.iterrows():
            i,j=self.task_positions[t],self.flow_positions[f];self.X[i,j]=r['mean'];self.omega[i,j]=1;self.counts[i,j]=int(r['size'])
        self.training_task_positions=sorted({self.task_positions[t] for t in train.task_dataset_uid});self.training_flow_positions=sorted({self.flow_positions[f] for f in train.flow_uid})
        raw=self.rawA[self.training_task_positions];self.all_missing=np.isnan(raw).all(axis=0);self.medians=np.array([0. if self.all_missing[j] else np.nanmedian(raw[:,j]) for j in range(10)])
        filled=np.where(np.isnan(self.rawA),self.medians,self.rawA);self.row_scaler=StandardScaler().fit(filled[self.training_task_positions]);self.column_scaler=StandardScaler().fit(self.rawB[self.training_flow_positions])
        row=self.row_scaler.transform(filled);column=self.column_scaler.transform(self.rawB);column[:,self.column_scaler.var_==0]=0.
        self.A=np.column_stack([row,np.ones(85)]);self.B=np.column_stack([column,np.ones(1005)])
        assert self.counts.sum()==len(train) and self.A.shape==(85,11) and self.B.shape==(1005,385)
        return self
    def predictions(self,matrix,test):
        if 'predictive_accuracy' in test:raise ValueError('test outcomes forbidden')
        return matrix[[self.task_positions[t] for t in test.task_dataset_uid],[self.flow_positions[f] for f in test.flow_uid]]
