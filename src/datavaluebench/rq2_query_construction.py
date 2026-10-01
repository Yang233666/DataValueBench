"""Reusable DataValueBench benchmark implementation."""
import hashlib,json,math
FIELDS=('tags','task_categories','task_contexts','language','license')
STRATA=('low-support','medium-support','high-support')

def intent_record(conditions,support):
    conditions=sorted(conditions,key=lambda x:FIELDS.index(x[0]))
    obj={'conditions':[{'field':f,'value':v} for f,v in conditions],'operator':'AND'}
    encoded=json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    digest=hashlib.sha256(encoded.encode()).hexdigest()
    return {'intent_id':'rq2i_'+digest[:16],'intent_sha256':digest,'canonical_intent_json':encoded,
            'complexity':len(conditions),'group':'|'.join(f for f,v in conditions),'support':support,
            'sampling_key':hashlib.sha256(('DVBench-v2-RQ2-SAMPLE|20260903|'+digest).encode()).hexdigest()}


def hamilton_quotas(capacities,total,base):
    if sum(capacities)<total:raise ValueError('insufficient eligible intent capacity')
    quotas=[min(base,n) for n in capacities]
    while sum(quotas)<total:
        remaining=total-sum(quotas);capacity=[n-q for n,q in zip(capacities,quotas)]
        weights=[math.sqrt(n) for n in capacity];den=sum(weights)
        ideal=[remaining*w/den for w in weights]
        for i,x in enumerate(ideal):quotas[i]+=min(capacity[i],math.floor(x))
        remaining=total-sum(quotas)
        for i in sorted(range(len(quotas)),key=lambda i:(-(ideal[i]-math.floor(ideal[i])),i)):
            if not remaining:break
            if quotas[i]<capacities[i]:quotas[i]+=1;remaining-=1
    return quotas


def stratum_quotas(capacities,total):
    quotas=[total//3+(i<total%3) for i in range(3)]
    quotas=[min(q,c) for q,c in zip(quotas,capacities)]
    while sum(quotas)<total:
        choices=[i for i in range(3) if quotas[i]<capacities[i]]
        if not choices:raise ValueError('query-construction stratum exhausted')
        chosen=min(choices,key=lambda i:(-(capacities[i]-quotas[i]),i));quotas[chosen]+=1
    return quotas
