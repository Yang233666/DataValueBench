"""Offline input/support validation and deterministic released-result summaries.

This module never trains, downloads, constructs an index, or calls a model API.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,sys

def require(ok,message):
    if not ok:raise ValueError(message)

def read(root,path):return json.loads((root/path).read_text())
def lines(root,path):return [json.loads(x) for x in (root/path).read_text().splitlines() if x]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()

def integrity(root):
    entries=read(root,'checksums.json')['files']
    for name,expected in entries.items():
        p=root/name
        require(p.is_file(),'Missing required file: '+name)
        require(sha(p)==expected,'Checksum mismatch: '+name)
    return {'status':'PASS','files_verified':len(entries)}

def load_support(root,ecosystem):
    """Reconstruct condition membership in its stored order from Parquet tables."""
    import pandas as pd
    sources=pd.read_parquet(root/f'data/rq3/{ecosystem}/sources.parquet')
    members=pd.read_parquet(root/f'results/rq3/{ecosystem}/common_support.parquet')
    result=read(root,f'results/rq3/{ecosystem}/support_statistics.json')
    for i,condition in enumerate(result['conditions']):
        part=members[members.condition_index==i].sort_values('membership_position')
        require(part.membership_position.tolist()==list(range(len(part))),'Noncontiguous membership order')
        condition['common_source_ids']=sources.source_id.iloc[part.source_position].tolist()
    return result

def rq1(root):
    import pandas as pd
    scope=read(root,'configs/rq1/scope.json');frames=[pd.read_csv(p) for p in sorted((root/'results/rq1').glob('*_metrics.csv'))]
    metrics=pd.concat(frames,ignore_index=True)
    require(set(metrics.condition)=={'G0','GT80','GD','GF'},'Unexpected RQ1 regime')
    require(not metrics.duplicated(['condition','method_id','metric']).any(),'Duplicate RQ1 result')
    require(set(metrics.method_id)==set(scope['conditions']),'RQ1 method membership mismatch')
    gm=metrics[(metrics.condition=='GT80')&(metrics.method_id=='GM')&(metrics.metric=='Spearman')]
    require(len(gm)==1 and gm.point_estimate.isna().all(),'Constant GlobalMean correlation must remain undefined')
    intervals=[row for p in (root/'results/rq1').glob('*_intervals.json') for row in read(root,p)['intervals']]
    undef=next(r for r in intervals if (r['condition'],r['method_id'],r['metric'])==('GT80','GM','Spearman'))
    require(undef['point_estimate'] is None and undef['defined_draws']==0 and undef['undefined_draws']==2000,'Undefined bootstrap support changed')
    obs=pd.read_parquet(root/'data/rq1/observation_identities.parquet')
    require(len(obs)==106907 and obs.observation_uid.nunique()==106907,'Observation identity mismatch')
    require((obs.task_id.nunique(),obs.dataset_id.nunique(),obs.flow_id.nunique(),obs.run_id.nunique())==(85,85,1005,105807),'Historical cohort identity mismatch')
    features=pd.read_parquet(root/'data/rq1/dataset_features.parquet')
    require(set(features.dataset_id)==set(obs.dataset_id),'Intrinsic feature identity mismatch')
    import numpy as np
    vectors=np.load(root/'data/rq1/workflow_vectors.npy',allow_pickle=False)
    vector_ids=pd.read_parquet(root/'data/rq1/workflow_vector_ids.parquet')
    require(vectors.shape==(1005,384) and str(vectors.dtype)=='float32' and np.isfinite(vectors).all(),'MiniLM feature schema mismatch')
    require(set(vector_ids.flow_id)==set(obs.flow_id),'MiniLM workflow identities mismatch')
    split_counts={}
    for kind in ['outer','inner']:
        t=pd.read_parquet(root/f'data/rq1/{kind}_splits.parquet')
        require(t.observation_position.between(0,len(obs)-1).all(),'Unknown observation position')
        key='condition' if kind=='outer' else 'outer_condition'
        require(set(t[key])==set(scope['regimes']),'Split scope mismatch');split_counts[kind]=len(t)
    return {'status':'PASS','regimes':scope['regimes'],'methods':len(scope['conditions']),'metric_rows':len(metrics),'observation_count':len(obs),'split_rows':split_counts,'undefined_correlation_preserved':True,'mode':'bundled-data/result verification; no estimator refitting'}

def rq2(root):
    import pandas as pd
    latent=lines(root,'data/rq2/latent_intents.jsonl');queries=lines(root,'data/rq2/track_b_queries.jsonl');na=lines(root,'data/rq2/unrealized_track_b_slots.jsonl');tracka=lines(root,'data/rq2/track_a_queries.jsonl')
    lids={r['intent_id'] for r in latent};qids={r['intent_id'] for r in queries};nids={r['intent_id'] for r in na}
    require((len(latent),len(lids),len(queries),len(qids),len(na),len(nids))==(500,500,475,475,25,25),'Workload cardinality mismatch')
    require(lids==qids|nids and not qids&nids,'Workload partition mismatch')
    require(len({' '.join(r['nl_query'].casefold().split()) for r in queries})==475,'Duplicate natural-language query')
    require(set(read(root,'data/rq2/common_support.json'))==qids,'Common support mismatch')
    require({r['intent_id'] for r in tracka}==lids,'Track A intent mismatch')
    require(all(r['reason'] for r in na),'Unrealized slot missing reason')
    census=pd.read_parquet(root/'data/rq2/eligible_intents.parquet')
    require(len(census)==278035 and census.intent_id.nunique()==278035 and lids<=set(census.intent_id),'Eligible intent census mismatch')
    from .rq2_query_construction import intent_record
    for row in latent:
        conditions=json.loads(row['canonical_intent_json'])['conditions']
        rebuilt=intent_record([(c['field'],c['value']) for c in conditions],row['support'])
        require(rebuilt['intent_id']==row['intent_id'] and rebuilt['intent_sha256']==row['intent_sha256'],'Intent identity mismatch')
    qrels=pd.read_parquet(root/'data/rq2/automatic_qrels.parquet')
    require(set(qrels.intent_id)==lids and set(qrels.relevance)=={1},'Automatic qrel schema/support mismatch')
    support=qrels.groupby('intent_id').dataset_id.nunique().to_dict()
    require(all(support[r['intent_id']]==r['support'] for r in latent),'Qrel counts differ from intent supports')
    frame=pd.read_csv(root/'results/rq2/per_intent_metrics.csv',float_precision='round_trip');expected=read(root,'results/rq2/effectiveness.json')
    require(len(frame)==5700 and not frame.duplicated(['method_id','intent_id']).any(),'Per-intent key mismatch')
    summaries=[]
    for method,values in expected['methods'].items():
        part=frame[frame.method_id==method];require(len(part)==475 and set(part.intent_id)==qids,'Method support differs: '+method)
        means={metric:math.fsum(part[metric].astype(float))/475 for metric in ['ndcg_at_10','map']}
        require(all(abs(means[k]-values[k])<1e-14 for k in means),'Aggregate mismatch: '+method)
        summaries.append({'method':method,'support':475,**means})
    require(len(summaries)==12,'Unexpected method count')
    require(sum(r['complexity']==1 for r in queries)==275 and sum(r['complexity']==2 for r in queries)==200,'Query complexity coverage mismatch')
    return {'status':'PASS','latent_intents':500,'common_support':475,'unrealized_track_b_slots':25,'qrel_rows':len(qrels),'effectiveness':summaries,'mode':'macro means recomputed from released per-intent outputs; no retrieval or remote generation'}

def rq3(root):
    import pandas as pd
    import pyarrow.parquet as pq
    reports={}
    for eco,n in [('openml',6408),('huggingface',100000)]:
        source=pd.read_parquet(root/f'data/rq3/{eco}/sources.parquet')
        require(len(source)==n and source.source_id.nunique()==n and source.source_position.tolist()==list(range(n)),'Source dictionary mismatch')
        s=load_support(root,eco);primary=read(root,f'results/rq3/{eco}/primary_metrics.json')
        require(primary['comparison_sample_size']==n and len(primary['rows'])==33,'Primary scope mismatch')
        require('BGE-M3' in {r['method'] for r in primary['rows']},'BGE-M3 label missing')
        for condition in s['conditions']:
            require(len(condition['common_source_ids'])==len(set(condition['common_source_ids']))==condition['comparison_sample_size'],'Common support mismatch')
        c=next(x for x in s['conditions'] if x['candidate_pool']=='C100_v2' and x['requested_depth']==10)
        require(len(c['common_source_ids'])==n,'Primary common support mismatch')
        count=0;seen=set()
        for p in sorted((root/f'data/rq3/{eco}').glob('candidates*.parquet')):
            table=pq.read_table(p)
            for row in table.to_pylist():
                sid=row['source_position'];ids=row['candidate_positions']
                require(sid not in seen and len(ids)==len(set(ids))==200 and sid not in ids,'Candidate order/membership invalid')
                require(all(0<=i<n for i in ids),'Candidate outside ecosystem')
                seen.add(sid);count+=1
        require(count==n,'Incomplete candidate source coverage')
        reports[eco]={'sources':n,'conditions':len(s['conditions']),'primary_endpoint_values':33,'candidate_identities':count*200,'structural_NA_preserved':True}
    return {'status':'PASS','ecosystems':reports,'mode':'frozen results, complete candidate membership and applicability verification; no ranking/model execution'}

def rq4(root):
    import pandas as pd
    from .rq4_statistics import agreement_metrics
    players=pd.read_parquet(root/'data/rq4/players.parquet');ids=sorted(players.player_uid)
    require(len(ids)==len(set(ids))==85 and sorted(players.groupby('fold_id').size())==[17]*5,'Player/fold membership mismatch')
    frames={u:pd.read_csv(root/f'results/rq4/{u.lower()}_player_values.csv',float_precision='round_trip') for u in ['U0','U1']}
    comparison=read(root,'results/rq4/utility_comparisons.json')['comparisons'];summaries=[]
    for saved in comparison:
        rule=saved['rule_a'];vectors=[]
        for u in ['U0','U1']:
            f=frames[u];part=f[f.rule==rule].set_index('player_uid')
            require(set(part.index)==set(ids) and len(part)==85,'Player result support mismatch')
            vectors.append(part.reindex(ids).primary_value.to_numpy())
        recomputed=agreement_metrics(*vectors,ids)
        for key,value in recomputed.items():
            target='sign_changes' if key=='sign_errors' else key
            if target not in saved:continue
            expected=saved[target]
            if isinstance(value,(int,float)) and not isinstance(value,bool):require(abs(value-expected)<1e-12,'Fixed-game comparison mismatch: '+rule+' '+target)
            else:require(value==expected,'Fixed-game categorical mismatch')
        summaries.append({'rule':rule,**recomputed})
    require(len(summaries)==6,'Expected six fixed-game contribution rules')
    return {'status':'PASS','players':85,'comparisons':summaries,'mode':'existing same-rule comparison implementation replayed from frozen player values; no contribution estimation'}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);parser.add_argument('--rq',choices=['all','rq1','rq2','rq3','rq4'],default='all');parser.add_argument('--output',type=Path);parser.add_argument('--checksums-only',action='store_true');args=parser.parse_args()
    try:
        result={'integrity':integrity(args.root)}
        if not args.checksums_only:
            for name in ['rq1','rq2','rq3','rq4'] if args.rq=='all' else [args.rq]:result[name]=globals()[name](args.root)
        text=json.dumps(result,indent=2,allow_nan=False)+'\n'
        if args.output:
            args.output.mkdir(parents=True,exist_ok=False);(args.output/'verification.json').write_text(text)
        print(text,end='')
    except (ValueError,KeyError,FileNotFoundError,ImportError) as exc:
        parser.exit(2,f'Verification failed: {exc}\n')
if __name__=='__main__':main()
