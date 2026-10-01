import pytest,networkx as nx
from datavaluebench.rq3_pinned_ppr import PinnedPPRGraph

def test_actual_pinned_reference_restart_and_canonical_ties():
 nodes=sorted(['Dataset::HF/A','Dataset::HF/a','Dataset::HF/isolated','Tag::x']);edges=[('Dataset::HF/A','Tag::x'),('Dataset::HF/a','Tag::x')];model=PinnedPPRGraph(nodes,edges);g=nx.Graph();g.add_nodes_from(nodes);g.add_edges_from(edges);restart={n:float(n=='Dataset::HF/A') for n in nodes};expected=nx.pagerank(g,alpha=.85,personalization=restart,dangling=restart,max_iter=1000,tol=1e-12,weight=None);actual=model.scores('HF/A',['HF/a','HF/isolated']);assert actual=={k:expected['Dataset::'+k] for k in actual};assert model.scores('HF/A',['HF/a','HF/isolated'])==actual;assert model.order('HF/isolated',['HF/a','HF/A'])==['HF/A','HF/a'];assert nx.is_frozen(model.graph)

def test_pinned_version_and_full_graph_identity_fail_closed(monkeypatch):
 with pytest.raises(ValueError):PinnedPPRGraph(['Dataset::a'],[('Dataset::a','Unknown::b')])
 monkeypatch.setattr(nx,'__version__','3.4.2')
 with pytest.raises(RuntimeError):PinnedPPRGraph(['Dataset::a'],[])
