"""Deterministic duplicate/replacement rules; no remote calls or file writes."""
import re,json,hashlib

def require(value,message):
    if not value:raise ValueError(message)

NA='NOT_APPLICABLE_QUERY_REALIZATION_STRATUM_EXHAUSTED'
REASON='QUERY_REALIZATION_STRATUM_EXHAUSTED'

def sampling_order(row):
    # Original builder uses stable key sort on support/hash-sorted stratum.
    return row['sampling_key'], row['support'], row['intent_sha256']

def select_replacements(census, active, failures, used, slot_origins=None):
    """Pure sequential selector; also applies to recursive rounds (never frees used IDs)."""
    by_id = {x['intent_id']: x for x in census}
    active_ids = {x['intent_id'] for x in active}
    require(len(active_ids) == len(active), 'duplicate active identities')
    require(active_ids <= set(used), 'active workload missing from historical used set')
    require(len(failures) == len(set(failures)) and set(failures) <= active_ids, 'invalid failed slots')
    origins = slot_origins or {i: i for i in active_ids}
    ordered = sorted(failures, key=lambda i: sampling_order(by_id[origins[i]]))
    accumulated = set(used)
    mappings, replacements = [], []
    for failed in ordered:
        old = by_id[failed]
        pool = sorted((x for x in census if x['group'] == old['group'] and x['support_stratum'] == old['support_stratum'] and x['intent_id'] not in accumulated), key=sampling_order)
        require(bool(pool), 'query-construction stratum exhausted')
        new = dict(pool[0])
        mappings.append(dict(failed_intent_id=failed, original_slot_intent_id=origins[failed],
            failed_field_or_field_pair=old['group'], failed_support_stratum=old['support_stratum'],
            failed_sampling_key=old['sampling_key'], failed_sampling_rank=old['stratum_sampling_rank'],
            original_slot_sampling_key=by_id[origins[failed]]['sampling_key'],
            replacement_intent_id=new['intent_id'], replacement_intent_sha256=new['intent_sha256'],
            field_or_field_pair=new['group'], support_stratum=new['support_stratum'],
            sampling_key=new['sampling_key'], stratum_sampling_rank=new['stratum_sampling_rank'],
            used_count_before=len(accumulated), previously_unused=new['intent_id'] not in accumulated,
            used_set_sha256_before=digest(json.dumps(sorted(accumulated), separators=(',', ':')).encode()),
            available_same_stratum_count=len(pool), generation_attempt=0))
        accumulated.add(new['intent_id'])
        replacements.append(new)
    replace = {m['failed_intent_id']: n for m,n in zip(mappings,replacements)}
    adjusted = []
    for row in active:
        new = dict(replace.get(row['intent_id'], row))
        new['original_slot_intent_id'] = origins[row['intent_id']]
        new['historical_stage_a_selected'] = bool(new.pop('selected', new.get('historical_stage_a_selected', False)))
        new['active_workload_member'] = True
        adjusted.append(new)
    require(len(adjusted) == len({x['intent_id'] for x in adjusted}) == len(active), 'active workload size/uniqueness changed')
    return mappings, replacements, adjusted, accumulated

def norm(q):return re.sub(r'\s+',' ',q.casefold()).strip()

def resolve_global_duplicates(selected,byid,next_attempt,rule_binding):
    """Supersede selection roles only; semantic evidence remains immutable."""
    seen={};duplicates=[]
    for iid in sorted(selected,key=lambda i:sampling_order(byid[i])):
        current=selected[iid];key=norm(current['nl_query'])
        if key not in seen:seen[key]=iid;continue
        earlier=seen[key];attempt=current['generation_attempt']
        require(attempt in (0,1,2),'invalid selected generation attempt')
        duplicates.append(dict(earlier=earlier,later=iid,normalization=key,
            status='VALID_BUT_SUPERSEDED_BY_GLOBAL_DUPLICATE_RULE',
            event='HUMAN_VALID_QUERY_SUPERSEDED_BY_DUPLICATE_RULE' if current['final_authority']=='HUMAN' else 'ASTRA_VALID_QUERY_SUPERSEDED_BY_DUPLICATE_RULE',
            reason='QUERY_IDENTITY_COLLISION',semantic_validity_unchanged=True,
            superseded_selection=dict(current),preserved_selection=dict(selected[earlier]),
            earlier_sampling_order=list(sampling_order(byid[earlier])),later_sampling_order=list(sampling_order(byid[iid])),
            next_action='NEXT_GENERATION_ATTEMPT' if attempt<2 else 'SAME_STRATUM_REPLACEMENT',
            next_generation_attempt=attempt+1 if attempt<2 else None,frozen_duplicate_rule=rule_binding))
        # 3 is an exhaustion sentinel, never a generation request.
        next_attempt[iid]=attempt+1
    for record in duplicates:selected.pop(record['later'])
    return duplicates

def replace_exhausted(census,active,selected,next_attempt,used,structural_na=None):
    failures=[r['intent_id'] for r in active if r['intent_id'] not in selected and r['intent_id'] not in (structural_na or {}) and next_attempt.get(r['intent_id'],0)>2]
    if not failures:return [],active,used
    origins={r['intent_id']:r['original_slot_intent_id'] for r in active}
    if structural_na is None:
        mappings,newrows,active,used=select_replacements(census,active,failures,used,origins)
    else:
        byid={r['intent_id']:r for r in census};mappings=[];newrows=[]
        for iid in sorted(failures,key=lambda i:sampling_order(byid[origins[i]])):
            old=byid[iid]
            stratum=[r for r in census if (r['group'],r['support_stratum'])==(old['group'],old['support_stratum'])]
            pool=sorted([r for r in stratum if r['intent_id'] not in used],key=sampling_order)
            if not pool:
                structural_na[iid]=dict(intent_id=iid,original_slot_intent_id=origins[iid],status=NA,reason=REASON,field_or_field_pair=old['group'],support_stratum=old['support_stratum'],support=old['support'],completed_generation_attempt=2,remaining_unused_candidates=[],eligible_census_count=len(stratum),historically_used_count=sum(r['intent_id'] in used for r in stratum),used_set_sha256=digest(canonical(sorted(used))))
                continue
            current_origins={r['intent_id']:r['original_slot_intent_id'] for r in active}
            m,n,active,used=select_replacements(census,active,[iid],used,current_origins)
            mappings.extend(m);newrows.extend(n)
    for r in newrows:next_attempt[r['intent_id']]=0
    return mappings,active,used

def lexical(query,conditions):
    tokens=lambda s:re.findall(r'[^\W_]+',s.lower(),flags=re.UNICODE)
    qt=set(tokens(query));values=[];mentions=[]
    for c in conditions:
        vt=tokens(c['value']);require(bool(vt),'empty lexical value token set')
        values.append(len(qt&set(vt))/len(set(vt)));mentions.append(int((' '+' '.join(vt)+' ') in (' '+' '.join(tokens(query))+' ')))
    score=sum(values)/len(values)
    return dict(lex_overlap=score,lex_overlap_bin='Low' if score<1/3 else 'Medium' if score<2/3 else 'High',exact_value_mentions=mentions,condition_token_recall=values)

def digest(b):return hashlib.sha256(b).hexdigest()

def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
