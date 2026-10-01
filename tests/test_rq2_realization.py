"""Bounded synthetic checks for the relocated pure query-construction rules."""
import copy
import pytest
from datavaluebench.rq2_realization import select_replacements,replace_exhausted,resolve_global_duplicates

def rows():
    return [dict(intent_id=f'i{i}',intent_sha256=f'h{i}',sampling_key=f'{i}',support=20,group='tags',support_stratum='Low',stratum_sampling_rank=i+1,original_slot_intent_id=f'i{i}') for i in range(4)]

def test_replacement_never_reuses_previously_used_identity():
    census=rows();active=census[:2];maps,new,after,used=select_replacements(census,active,['i1','i0'],{'i0','i1'}, {'i0':'i0','i1':'i1'})
    assert [m['failed_intent_id'] for m in maps]==['i0','i1']
    assert [r['intent_id'] for r in new]==['i2','i3']
    assert [r['original_slot_intent_id'] for r in after]==['i0','i1']
    assert used=={'i0','i1','i2','i3'}
    with pytest.raises(ValueError,match='stratum exhausted'):
        select_replacements(census,active,['i0'],used)

def test_exhausted_slot_stays_explicit_without_zero_score():
    census=rows()[:1];na={};attempt={'i0':3}
    maps,active,used=replace_exhausted(census,census,{},attempt,{'i0'},na)
    assert maps==[] and active==census and used=={'i0'}
    assert na['i0']['reason']=='QUERY_REALIZATION_STRATUM_EXHAUSTED'
    assert na['i0']['remaining_unused_candidates']==[]
    assert 'score' not in na['i0']

def test_duplicate_resolution_keeps_earlier_sampling_identity():
    census=rows();byid={r['intent_id']:r for r in census}
    selected={r['intent_id']:dict(nl_query=q,generation_attempt=0,final_authority='ASTRA_SEMANTIC_VALIDATION') for r,q in zip(census[:2],['Dataset need','  DATASET   NEED '])}
    attempt={};before=copy.deepcopy(selected)
    conflicts=resolve_global_duplicates(selected,byid,attempt,{})
    assert list(selected)==['i0'] and selected['i0']==before['i0']
    assert attempt=={'i1':1} and conflicts[0]['semantic_validity_unchanged']
