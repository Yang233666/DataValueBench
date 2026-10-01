"""Reusable DataValueBench benchmark implementation."""
from __future__ import annotations

from collections import Counter
import math
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _apply_pairs(syn0, syn1, centers, contexts, negatives, update, total, learning_rate):
    dimension = syn0.shape[1]
    for pair in range(len(centers)):
        alpha = max(learning_rate * .0001, learning_rate * (1. - update / total))
        update += 1
        input_row = contexts[pair]
        before = syn0[input_row].copy()
        error = np.zeros(dimension, dtype=np.float64)
        for target_number in range(negatives.shape[1] + 1):
            target = centers[pair] if target_number == 0 else negatives[pair, target_number - 1]
            label = 1. if target_number == 0 else 0.
            dot = float(np.dot(before, syn1[target]))
            sigmoid = 1. / (1. + math.exp(-max(-20., min(20., dot))))
            gradient = (label - sigmoid) * alpha
            for coordinate in range(dimension):
                error[coordinate] += gradient * syn1[target, coordinate]
                syn1[target, coordinate] += gradient * before[coordinate]
        for coordinate in range(dimension):
            syn0[input_row, coordinate] += error[coordinate]
    return update


def _draw_negatives(rng, context_types, distributions, count):
    # Generator.choice(p=...) uses this exact cumsum/normalization/searchsorted
    # path and one random double per target. Batching preserves the flat stream.
    uniforms = rng.random((len(context_types), count))
    result = np.empty(uniforms.shape, dtype=np.int64)
    for kind, (eligible_indices, cdf) in distributions.items():
        mask = context_types == kind
        result[mask] = eligible_indices[np.searchsorted(cdf, uniforms[mask], side="right")]
    return result


def train_metapath2vecpp_exact_accelerated(walks, *, embedding_dim, window, negative,
                                         initial_learning_rate, epochs, seed,
                                         batch_pairs=16384, progress=None):
    from .rq3_ranking import id_sort_key, typed_node_type, _type_negative_distribution, l2_normalize

    if not walks or min(embedding_dim, window, negative, epochs, batch_pairs) <= 0:
        raise ValueError("invalid metapath2vec++ corpus/configuration")
    vocabulary = sorted({str(node) for walk in walks for node in walk}, key=id_sort_key)
    index = {node: i for i, node in enumerate(vocabulary)}
    counts = Counter(str(node) for walk in walks for node in walk)
    type_names = sorted({typed_node_type(node) for node in vocabulary})
    type_index = {name: i for i, name in enumerate(type_names)}
    types = np.array([type_index[typed_node_type(node)] for node in vocabulary], dtype=np.int64)
    distributions = {}
    for name in type_names:
        eligible, weights = _type_negative_distribution(name, vocabulary, counts)
        cdf = weights.cumsum()
        cdf /= cdf[-1]
        distributions[type_index[name]] = (np.array([index[node] for node in eligible], dtype=np.int64), cdf)
    rng = np.random.default_rng(seed)
    syn0 = rng.uniform(-.5 / embedding_dim, .5 / embedding_dim, (len(vocabulary), embedding_dim))
    syn1 = np.zeros_like(syn0)
    encoded = [np.array([index[str(node)] for node in walk], dtype=np.int64) for walk in walks]
    patterns = {}
    for length in {len(walk) for walk in encoded}:
        pairs = [(center, context) for center in range(length)
                 for context in range(max(0, center-window), min(length, center+window+1)) if context != center]
        patterns[length] = (np.array([p[0] for p in pairs], dtype=np.int64),
                            np.array([p[1] for p in pairs], dtype=np.int64))
    total = max(1, sum(len(patterns[len(walk)][0]) for walk in encoded) * epochs)
    center_buffer = np.empty(batch_pairs, dtype=np.int64)
    context_buffer = np.empty(batch_pairs, dtype=np.int64)
    update, filled = 0, 0
    def apply(n):
        nonlocal update
        centers, contexts = center_buffer[:n], context_buffer[:n]
        negatives = _draw_negatives(rng, types[contexts], distributions, negative)
        update = _apply_pairs(syn0, syn1, centers, contexts, negatives, update, total, initial_learning_rate)
    for epoch in range(epochs):
        for walk_number, walk in enumerate(encoded):
            center_positions, context_positions = patterns[len(walk)]
            start = 0
            while start < len(center_positions):
                take = min(batch_pairs - filled, len(center_positions) - start)
                center_buffer[filled:filled+take] = walk[center_positions[start:start+take]]
                context_buffer[filled:filled+take] = walk[context_positions[start:start+take]]
                start += take
                filled += take
                if filled == batch_pairs:
                    apply(filled)
                    filled = 0
            if progress is not None and (walk_number % 1024 == 0 or walk_number + 1 == len(encoded)):
                progress(epoch=epoch, walks_complete=walk_number+1, walks_total=len(encoded),
                         pairs_complete=update, pairs_total=total)
    if filled:
        apply(filled)
    normalized = l2_normalize(syn0)
    return {node: normalized[i] for i, node in enumerate(vocabulary)}


def acceleration_reference_check():
    """Bounded check in the actual methods executable before production use."""
    from importlib.metadata import version
    from .rq3_ranking import train_metapath2vecpp_reference
    walks = [["Dataset::1", "ClassBin::0", "Dataset::2", "ClassBin::0", "Dataset::1"],
             ["Dataset::2", "ClassBin::1", "Dataset::1", "ClassBin::1", "Dataset::2"]]
    rows = []
    for seed in range(20260903, 20260908):
        kwargs = dict(embedding_dim=128, window=7, negative=5,
                      initial_learning_rate=.025, epochs=2, seed=seed)
        reference = train_metapath2vecpp_reference(walks, **kwargs)
        for batch in (7, 16384):
            observed = train_metapath2vecpp_exact_accelerated(walks, batch_pairs=batch, **kwargs)
            rows.append({"seed": seed, "batch_pairs": batch,
                         "bitwise_equal": all(np.array_equal(reference[node], observed[node]) for node in reference)})
    return {"status": "PASS" if all(row["bitwise_equal"] for row in rows) else "FAILED",
            "checks": rows, "fastmath": False, "parallel_updates": False,
            "versions": {name: version(name) for name in ("numpy", "scipy", "numba", "llvmlite")}}
