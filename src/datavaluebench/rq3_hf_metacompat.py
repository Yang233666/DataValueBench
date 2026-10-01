"""Reusable DataValueBench benchmark implementation."""
from datavaluebench.reference import _tokens
FIELDS=('task_contexts','language','license','name_tokens','tags')
COMPONENTS=('task_similarity','language_similarity','license_similarity','name_similarity','tag_similarity')
class HFMetaCompat:
    def __init__(self,records):
        self.values={r['dataset_id']:tuple(frozenset(_tokens(r[f])) for f in FIELDS) for r in records}
    def components(self,source,candidate):
        a=self.values[source];b=self.values[candidate]
        values=[len(x&y)/len(x|y) if x and y else 0.0 for x,y in zip(a,b)]
        result=dict(zip(COMPONENTS,values))
        result['metacompat']=.35*values[0]+.20*values[1]+.15*values[2]+.15*values[3]+.15*values[4]
        return result
