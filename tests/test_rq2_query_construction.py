from datavaluebench.rq2_query_construction import hamilton_quotas,stratum_quotas,intent_record
import pytest


def test_capacity_capped_hamilton_and_field_ties():
 assert hamilton_quotas([100,100,100,100,100],300,30)==[60]*5
 assert hamilton_quotas([1,1,1000],10,1)==[1,1,8]
 assert hamilton_quotas([100,100,100],5,0)==[2,2,1]
 with pytest.raises(ValueError):hamilton_quotas([1,2],4,1)


def test_stratum_redistribution_and_order():
 assert stratum_quotas([100,100,100],5)==[2,2,1]
 assert stratum_quotas([0,10,20],10)==[0,3,7]


def test_intent_hash_is_order_independent_and_field_sensitive():
 a=intent_record([('language','en'),('tags','text')],20)
 assert a==intent_record([('tags','text'),('language','en')],20)
 assert a['intent_id']!=intent_record([('task_contexts','text'),('language','en')],20)['intent_id']
