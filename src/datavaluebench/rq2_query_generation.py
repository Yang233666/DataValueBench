"""Reusable DataValueBench benchmark implementation."""
import hashlib,json,re
LABELS={'tags':'topic or tag','task_categories':'task category','task_contexts':'task context','language':'language','license':'license'}
SYSTEM='You generate natural-language search queries for a dataset retrieval benchmark. Convert the supplied structured dataset-search intent into exactly three concise English queries that a researcher could plausibly type when searching for datasets. Preserve every supplied constraint and do not add any constraint. Do not mention database schemas, internal field names, SQL, Boolean operators, JSON, benchmark construction, or relevance labels. Natural mentions of the supplied values are allowed and should not be artificially hidden. Each query must be a standalone dataset-search information need, semantically faithful, unambiguous, and natural. Return only one valid JSON object with exactly one key "queries", whose value is an array of exactly three different strings. Do not output explanations or any other text.'

def messages(intent):
    conditions=intent['conditions']
    blocks=[f'Constraint {i}:\n{LABELS[c["field"]]}: {c["value"]}' for i,c in enumerate(conditions,1)]
    return [{'role':'system','content':SYSTEM},{'role':'user','content':'Structured dataset-search intent:\n\n'+'\n\n'.join(blocks)+'\n\nGenerate the three queries.'}]

def generation_seed(intent_sha256,attempt):
    if attempt not in (0,1,2):raise ValueError('frozen attempt exhausted')
    digest=hashlib.sha256(f'DVBench-v2-RQ2-QGEN|20260903|{intent_sha256}|attempt={attempt}'.encode()).digest()
    return int.from_bytes(digest[:8],'big')>>1

def validate_response(response):
    obj=json.loads(response)
    if not isinstance(obj,dict) or set(obj)!= {'queries'} or not isinstance(obj['queries'],list) or len(obj['queries'])!=3:raise ValueError('FORMAT_FAIL')
    values=obj['queries']
    if any(not isinstance(x,str) or not x.strip() or '\n' in x or '\r' in x for x in values):raise ValueError('FORMAT_FAIL')
    normalized=[re.sub(r'\s+',' ',x.casefold()).strip() for x in values]
    if len(set(normalized))!=3:raise ValueError('FORMAT_FAIL')
    return values
