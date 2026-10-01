import copy
import pytest
from datavaluebench.rq1_workflow_text import canonical_workflow_text


def test_exact_fields_normalization_and_leakage_exclusion():
    flow={"id":"999","name":" Ａ\t Model ","class_name":"Class","external_version":" 1\n2 ",
          "description":"SECRET_PERFORMANCE","uploader":"PRIVATE_AUTHOR","parameter":[
              {"name":"z","default_value":[]},{"name":"a","default_value":" x\n y "}]}
    expected="FLOW_CLASS: Class\nFLOW_NAME: A Model\nFLOW_VERSION: 1 2\nPARAM: ROOT::a=x y\nPARAM: ROOT::z=<MISSING>"
    assert canonical_workflow_text({"flow":flow})==expected
    assert not canonical_workflow_text(flow).endswith("\n")


def test_component_preorder_and_dictionary_parameter_order_independence():
    flow={"id":"1","name":"root","parameter":[{"name":"b","default_value":"2"},{"name":"a","default_value":"1"}],
          "component":[{"identifier":"z","flow":{"id":"2","name":"last"}},
                       {"identifier":"a","flow":{"id":"3","name":"first","component":{"identifier":"c","flow":{"id":"4","name":"nested"}}}}]}
    expected=canonical_workflow_text(flow);shuffled=dict(reversed(list(copy.deepcopy(flow).items())))
    shuffled["parameter"].reverse();shuffled["component"].reverse()
    assert canonical_workflow_text(shuffled)==expected
    assert expected.index("COMPONENT_PATH: ROOT/a\n")<expected.index("COMPONENT_PATH: ROOT/a/c\n")<expected.index("COMPONENT_PATH: ROOT/z\n")


def test_cycles_rejected_and_shared_noncyclic_component_allowed():
    root={"id":"1","name":"root"};root["component"]={"identifier":"cycle","flow":root}
    with pytest.raises(ValueError,match="cycle"):canonical_workflow_text(root)
    child={"id":"2","name":"child"};root["component"]=[{"identifier":"a","flow":child},{"identifier":"b","flow":child}]
    assert canonical_workflow_text(root).count("COMPONENT_NAME: child")==2


def test_missing_fields_and_no_parameter_value_inference():
    text=canonical_workflow_text({"id":"1","parameter":{"name":"flag","default_value":[],"description":"default true"}})
    assert text=="FLOW_CLASS: \nFLOW_NAME: \nFLOW_VERSION: \nPARAM: ROOT::flag=<MISSING>"
    with pytest.raises(ValueError):canonical_workflow_text({"parameter":{"name":"x","default_value":{"unmapped":"field"}}})


def test_collision_subtrees_preserve_multiplicity_and_ignore_ids_positions():
    child={"identifier":"preproc","flow":{"id":"1","name":"z","description":"excluded"}}
    earlier={"identifier":"preproc","flow":{"id":"999","name":"a"}}
    root={"name":"root","component":[child,earlier,copy.deepcopy(child)]}
    text=canonical_workflow_text(root)
    root["component"].reverse()
    root["component"][0]["flow"]["id"]="777"
    root["component"][0]["flow"]["description"]="different excluded evidence"
    assert canonical_workflow_text(root)==text
    assert text.count("COMPONENT_PATH: ROOT/preproc")==3
    assert text.count("COMPONENT_NAME: z")==2
    assert text.index("COMPONENT_NAME: a")<text.index("COMPONENT_NAME: z")


def test_collision_secondary_key_is_recursive_and_dictionary_order_independent():
    a={"name":"same","component":[{"identifier":"inner","flow":{"name":"a"}}]}
    z={"name":"same","component":[{"identifier":"inner","flow":{"name":"z"}}]}
    root={"component":[{"identifier":"dup","flow":z},{"identifier":"dup","flow":a}]}
    expected=canonical_workflow_text(root)
    root["component"].reverse()
    root["component"][0]["flow"]=dict(reversed(list(a.items())))
    assert canonical_workflow_text(root)==expected
    assert expected.index("COMPONENT_NAME: a")<expected.index("COMPONENT_NAME: z")
