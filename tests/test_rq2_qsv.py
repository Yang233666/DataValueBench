import pytest
from datavaluebench.rq2_qsv import candidate_identity,public_item,presentation_order,agreement,validate_label

def test_identity_includes_exact_text_and_blind_view():
 r={'intent_id':'toy','generation_attempt':0,'candidate_index':1,'query':'Toy query','conditions':[{'field':'language','value':'zh'}]}
 assert candidate_identity(r)!=candidate_identity(dict(r,query='toy query'))
 assert public_item(r)=={'query':'Toy query','conditions':[{'label':'Language','value':'zh'}]}
 assert set(public_item(r))=={'conditions','query'}

def test_order_is_independent_of_input_order():
 rows=[{'candidate_uid':str(i)} for i in range(108)]
 assert presentation_order(rows,'test_A')==presentation_order(rows[::-1],'test_A')
 assert presentation_order(rows,'test_A')!=presentation_order(rows,'test_B')

def toy(v):
 return dict(candidate_uid='toy',validator_anonymous_id=v,instruction_version='v1.0',annotation_start_timestamp='2026-09-13T00:00:00+00:00',annotation_end_timestamp='2026-09-13T00:00:01+00:00',**{d+'_label':'PASS' for d in ('fidelity','clarity','naturalness')},**{d+'_reason_codes':[] for d in ('fidelity','clarity','naturalness')})
def test_missing_and_no_variation_never_automatic_human_gate_pass():
 assert agreement([],['toy'],['A','B'])['status']=='INCOMPLETE_ANNOTATION'
 r=agreement([toy('A'),toy('B')],['toy'],['A','B'])
 assert r['numerical_gate_pass'] and not r['pilot_pass'] and r['cohen_kappa']['fidelity'] is None
 with pytest.raises(ValueError):agreement([toy('A'),toy('A')],['toy'],['A','B'])
def test_reason_and_adjudication():
 b=toy('B');b['clarity_label']='FAIL'
 with pytest.raises(ValueError):validate_label(b)
 b['clarity_reason_codes']=['AMBIGUOUS_SCOPE']
 r=agreement([toy('A'),b],['toy'],['A','B'])
 assert r['adjudication_queue']==[{'candidate_uid':'toy','disputed_dimensions':['clarity']}]
