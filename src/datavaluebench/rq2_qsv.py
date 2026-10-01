"""Reusable DataValueBench benchmark implementation."""
import hashlib,json,math
from datetime import datetime
DIMENSIONS=('fidelity','clarity','naturalness')
LABELS={'tags':'Topic or tag','task_categories':'Task category','task_contexts':'Task context','language':'Language','license':'License'}
REASONS={'fidelity':('MISSING_CONSTRAINT','ADDED_CONSTRAINT','WRONG_VALUE','LOGICAL_RELATION_CHANGED','SEMANTIC_DRIFT','OTHER_FIDELITY'),'clarity':('AMBIGUOUS_SCOPE','AMBIGUOUS_VALUE','AMBIGUOUS_RELATION','DATASET_SEARCH_INTENT_UNCLEAR','OTHER_CLARITY'),'naturalness':('SCHEMA_LIKE','DATABASE_LIKE','UNGRAMMATICAL','UNNATURAL_QUERY','OTHER_NATURALNESS')}
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def candidate_identity(row):
    return 'rq2qc_'+digest({'intent_id':row['intent_id'],'generation_attempt':row['generation_attempt'],'candidate_index':row['candidate_index'],'query_text_sha256':hashlib.sha256(row['query'].encode()).hexdigest()})
def presentation_order(rows,validator):
    if not isinstance(validator,str) or not validator.strip():raise ValueError('real anonymous validator ID required')
    return sorted(rows,key=lambda r:(digest({'validator_anonymous_id':validator,'candidate_uid':r['candidate_uid'],'presentation_seed':20260903}),r['candidate_uid']))
def public_item(row):
    return {'conditions':[{'label':LABELS[c['field']],'value':c['value']} for c in row['conditions']],'query':row['query']}
def validate_label(label):
    for dim in DIMENSIONS:
        value=label.get(dim+'_label');codes=label.get(dim+'_reason_codes')
        if value not in ('PASS','FAIL') or not isinstance(codes,list) or len(set(codes))!=len(codes):raise ValueError('invalid binary label/reasons')
        if any(c not in REASONS[dim] for c in codes):raise ValueError('wrong dimension reason')
        if (value=='FAIL' and not codes) or (value=='PASS' and codes):raise ValueError('FAIL requires reason; PASS has no reason')
        if any(c.startswith('OTHER_') for c in codes) and not label.get('optional_note','').strip():raise ValueError('OTHER requires a note')
    start=datetime.fromisoformat(label['annotation_start_timestamp']);end=datetime.fromisoformat(label['annotation_end_timestamp'])
    if start.tzinfo is None or end.tzinfo is None or end<start:raise ValueError('invalid UTC-aware annotation interval')
    if label.get('instruction_version')!='v1.0':raise ValueError('wrong instruction version')
    return label

def agreement(rows,candidate_ids,validators):
    if len(validators)!=2 or len(set(validators))!=2:raise ValueError('exactly two independent validators')
    indexed={}
    for row in rows:
        validate_label(row);key=(row['candidate_uid'],row['validator_anonymous_id'])
        if key in indexed:raise ValueError('duplicate locked initial label')
        if key[0] not in candidate_ids or key[1] not in validators:raise ValueError('label outside assignments')
        indexed[key]=row
    complete=[c for c in candidate_ids if all((c,v) in indexed for v in validators)]
    report={'expected_candidates':len(candidate_ids),'complete_paired_candidates':len(complete),'missing_initial_labels':len(candidate_ids)*2-len(indexed),'human_systematic_interpretation_review':'REQUIRED','pilot_pass':False}
    if report['missing_initial_labels']:return dict(report,status='INCOMPLETE_ANNOTATION')
    raw={};kappa={};disputes=[]
    for dim in DIMENSIONS:
        a=[indexed[c,validators[0]][dim+'_label'] for c in complete];b=[indexed[c,validators[1]][dim+'_label'] for c in complete]
        po=sum(x==y for x,y in zip(a,b))/len(a);pa=a.count('PASS')/len(a);pb=b.count('PASS')/len(b);pe=pa*pb+(1-pa)*(1-pb)
        raw[dim]=po;kappa[dim]=None if pe==1 else (po-pe)/(1-pe)
    for c in complete:
        dims=[d for d in DIMENSIONS if indexed[c,validators[0]][d+'_label']!=indexed[c,validators[1]][d+'_label']]
        if dims:disputes.append({'candidate_uid':c,'disputed_dimensions':dims})
    exact=(len(complete)-len(disputes))/len(complete)
    numerical=raw['fidelity']>=.90 and raw['clarity']>=.85 and raw['naturalness']>=.85 and exact>=.80
    report.update(raw_agreement=raw,cohen_kappa=kappa,candidate_exact_agreement=exact,adjudication_queue=disputes,numerical_gate_pass=numerical)
    report['status']='INCOMPLETE_ANNOTATION' if report['missing_initial_labels'] else ('HUMAN_SYSTEMATIC_INTERPRETATION_REVIEW_REQUIRED' if numerical else 'ANNOTATION_PILOT_FAILED')
    return report
