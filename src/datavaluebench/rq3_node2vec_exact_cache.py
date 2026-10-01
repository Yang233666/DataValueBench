"""Reusable DataValueBench benchmark implementation."""
import itertools,random
import numpy as np
from node2vec import Node2Vec
class SharedPreviousProbabilities:
    def __init__(self,probabilities,neighbors):self.probabilities=probabilities;self.neighbors=frozenset(neighbors)
    def __getitem__(self,previous):
        if previous not in self.neighbors:raise KeyError(previous)
        return self.probabilities
class ExactCachedNode2Vec(Node2Vec):
    def _precompute_probabilities(self):
        if self.p!=1 or self.q!=1 or self.sampling_strategy or self.workers!=1:raise ValueError('cache is valid only for frozen p=q=1,workers=1,no sampling override')
        if self.graph.is_directed() or self.graph.is_multigraph():raise ValueError('frozen graph must be simple undirected')
        if any(float(data.get(self.weight_key,1))!=1 for _,_,data in self.graph.edges(data=True)):raise ValueError('frozen graph must have binary unit weights')
        # The reference inserts each source followed by first encountered neighbors.
        # Do not replace this insertion sequence by a sorted-node iteration.
        for source in self.graph.nodes():
            self.d_graph[source]
            for neighbor in self.graph.neighbors(source):self.d_graph[neighbor]
        for source in self.graph.nodes():
            neighbors=list(self.graph.neighbors(source));weights=np.ones(len(neighbors),dtype=np.float64)
            probabilities=weights/weights.sum() if neighbors else weights
            probabilities.flags.writeable=False
            self.d_graph[source].update({self.NEIGHBORS_KEY:neighbors,self.FIRST_TRAVEL_KEY:probabilities,self.PROBABILITIES_KEY:SharedPreviousProbabilities(probabilities,neighbors),'cached_cumulative':list(itertools.accumulate(probabilities))})
    def _generate_walks(self):
        walks=[]
        for _ in range(self.num_walks):
            roots=list(self.d_graph.keys());random.shuffle(roots)
            for source in roots:
                walk=[source]
                while len(walk)<self.walk_length:
                    item=self.d_graph[walk[-1]];options=item[self.NEIGHBORS_KEY]
                    if not options:break
                    walk.append(random.choices(options,cum_weights=item['cached_cumulative'])[0])
                walks.append(list(map(str,walk)))
        return walks

def train_exact_cached(graph,nodes,seed):
    from unittest.mock import patch
    from datavaluebench.rq3_ranking import train_node2vec_embeddings
    # Process-local substitution only of the above proven cache implementation.
    # The production training wrapper supplies every frozen Gensim argument.
    with patch('node2vec.Node2Vec',ExactCachedNode2Vec):
        return train_node2vec_embeddings(graph,nodes,seed=seed)
