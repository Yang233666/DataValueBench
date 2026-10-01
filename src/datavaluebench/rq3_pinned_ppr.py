"""Reusable DataValueBench benchmark implementation."""
import hashlib,inspect
import networkx as nx
from datavaluebench.rq3_ranking import score_order
class PinnedPPRGraph:
    def __init__(self,nodes,edges):
        if nx.__version__!='3.6.1':raise RuntimeError('Pinned PPR requires isolated NetworkX3.6.1; do not inherit another environment version')
        if len(nodes)!=len(set(nodes)) or list(nodes)!=sorted(nodes):raise ValueError('Frozen canonical graph node order required')
        self.graph=nx.Graph();self.graph.add_nodes_from(nodes);self.graph.add_edges_from(edges)
        if list(self.graph)!=list(nodes):raise ValueError('Edge endpoint absent from frozen node universe')
        nx.freeze(self.graph)
        from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_scipy
        self.reference={'networkx':nx.__version__,'pagerank_source_sha256':hashlib.sha256(inspect.getsource(_pagerank_scipy).encode()).hexdigest(),'alpha':.85,'max_iter':1000,'tol':1e-12,'weight':None,'graph_nodes':len(self.graph),'graph_edges':self.graph.number_of_edges(),'backend':'unmodified pinned nx.pagerank; only immutable graph object reused'}
    def scores(self,source_id,candidate_ids):
        source='Dataset::'+str(source_id)
        if source not in self.graph or source_id in candidate_ids or len(candidate_ids)!=len(set(candidate_ids)):raise ValueError('Invalid frozen source/candidate identities')
        nodes=['Dataset::'+str(c) for c in candidate_ids]
        if any(n not in self.graph for n in nodes):raise ValueError('Candidate missing from global graph')
        restart={n:float(n==source) for n in self.graph}
        scores=nx.pagerank(self.graph,alpha=.85,personalization=restart,dangling=restart,max_iter=1000,tol=1e-12,weight=None)
        return {c:scores[n] for c,n in zip(candidate_ids,nodes)}
    def order(self,source_id,candidate_ids):return score_order(self.scores(source_id,candidate_ids))
