import hashlib,pytest
from datavaluebench.rq2_query_generation import messages,generation_seed,validate_response

def test_seed_first63bits_and_message_firewall():
 h='a'*64;expected=int.from_bytes(hashlib.sha256(f'DVBench-v2-RQ2-QGEN|20260903|{h}|attempt=0'.encode()).digest(),'big')>>(256-63)
 assert generation_seed(h,0)==expected
 result=messages({'conditions':[{'field':'language','value':'zh'}],'support':999,'qrel':'SECRET'})
 assert result[1]['content']=='Structured dataset-search intent:\n\nConstraint 1:\nlanguage: zh\n\nGenerate the three queries.'
 assert 'SECRET' not in str(result)

def test_format_only_does_not_substitute_human_judgment():
 assert validate_response('{"queries":["a","b","c"]}')==['a','b','c']
 for raw in ['{"queries":["a","A ","c"]}','{"queries":["a","b"]}','{"queries":["a","b","c"],"extra":1}']:
  with pytest.raises(ValueError):validate_response(raw)
