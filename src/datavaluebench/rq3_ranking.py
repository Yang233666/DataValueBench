"""Reusable DataValueBench benchmark implementation."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from datavaluebench.construction import BM25_TOKEN_PATTERN, exact_bm25


RANDOM_SEED = 20260903
FINAL_SEEDS = tuple(range(20260903, 20260908))


def id_sort_key(value: Any) -> tuple[int, Any]:
    return (0, int(value)) if isinstance(value, (int, np.integer)) else (1, str(value))


def random_order(ecosystem_id: str, source_id: Any, candidate_ids: Sequence[Any]) -> list[Any]:
    """Frozen deterministic Random ordering (the method uses no runtime RNG)."""
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")

    def key(candidate_id: Any) -> tuple[str, tuple[int, Any]]:
        payload = (
            f"DVBench-v2-RQ3-RANDOM|{RANDOM_SEED}|"
            f"{ecosystem_id}|{source_id}|{candidate_id}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), id_sort_key(candidate_id)

    return sorted(candidate_ids, key=key)


def score_order(scores: Mapping[Any, float]) -> list[Any]:
    if any(not math.isfinite(float(score)) for score in scores.values()):
        raise ValueError("scores must be finite")
    return sorted(scores, key=lambda item: (-float(scores[item]), id_sort_key(item)))


def bm25_source_scores(
    corpus_ids: Sequence[Any], canonical_texts: Sequence[str], source_id: Any
) -> dict[Any, float]:
    """Score against the full frozen ecosystem index, before pool restriction."""
    if len(corpus_ids) != len(canonical_texts) or len(set(corpus_ids)) != len(corpus_ids):
        raise ValueError("corpus IDs/texts must be aligned and unique")
    position = {dataset_id: i for i, dataset_id in enumerate(corpus_ids)}
    if source_id not in position:
        raise KeyError(source_id)
    values = exact_bm25(canonical_texts, canonical_texts[position[source_id]], k1=1.5, b=0.75)
    return {dataset_id: float(values[i]) for i, dataset_id in enumerate(corpus_ids)}


def build_bm25_corpus(corpus_ids: Sequence[Any], canonical_texts: Sequence[str]) -> dict[str, Any]:
    """Build the frozen exact BM25 corpus representation once per ecosystem."""
    if len(corpus_ids) != len(canonical_texts) or len(set(corpus_ids)) != len(corpus_ids):
        raise ValueError("corpus IDs/texts must be aligned and unique")
    from scipy import sparse

    tokenized = [BM25_TOKEN_PATTERN.findall(str(text).lower()) for text in canonical_texts]
    vocabulary = {token: i for i, token in enumerate(sorted({token for row in tokenized for token in row}))}
    lengths = np.array([len(row) for row in tokenized], dtype=np.float64)
    if np.any(lengths == 0):
        raise ValueError("empty canonical BM25 document")
    row_indices: list[int] = []
    column_indices: list[int] = []
    frequencies: list[float] = []
    document_frequency = np.zeros(len(vocabulary), dtype=np.int64)
    for row_index, row in enumerate(tokenized):
        for token, count in Counter(row).items():
            row_indices.append(row_index)
            column_indices.append(vocabulary[token])
            frequencies.append(float(count))
            document_frequency[vocabulary[token]] += 1
    term_frequency = sparse.csr_matrix(
        (frequencies, (row_indices, column_indices)),
        shape=(len(tokenized), len(vocabulary)), dtype=np.float64,
    )
    repeated_rows = np.repeat(np.arange(len(tokenized)), np.diff(term_frequency.indptr))
    inverse_document_frequency = np.log(
        1.0 + (len(tokenized) - document_frequency + 0.5) / (document_frequency + 0.5)
    )
    denominator = term_frequency.data + 1.5 * (
        0.25 + 0.75 * lengths[repeated_rows] / lengths.mean()
    )
    document_weights = sparse.csr_matrix(
        (
            inverse_document_frequency[term_frequency.indices]
            * term_frequency.data * 2.5 / denominator,
            term_frequency.indices,
            term_frequency.indptr,
        ),
        shape=term_frequency.shape,
    )
    return {
        "corpus_ids": tuple(corpus_ids),
        "positions": {dataset_id: i for i, dataset_id in enumerate(corpus_ids)},
        "queries": term_frequency,
        "document_weights": document_weights,
    }


def bm25_candidate_scores(
    corpus: Mapping[str, Any], source_id: Any, candidate_ids: Sequence[Any]
) -> dict[Any, float]:
    positions = corpus["positions"]
    if source_id not in positions or any(candidate not in positions for candidate in candidate_ids):
        raise KeyError("source/candidate absent from frozen BM25 corpus")
    candidate_positions = [positions[candidate] for candidate in candidate_ids]
    values = (
        corpus["queries"][positions[source_id]]
        @ corpus["document_weights"][candidate_positions].T
    ).toarray()[0]
    return {candidate: float(values[i]) for i, candidate in enumerate(candidate_ids)}


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("embedding matrix must be finite and two-dimensional")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms == 0):
        raise ValueError("zero-norm embedding")
    return matrix.astype(np.float64) / norms[:, None]


def dense_source_scores(
    dataset_ids: Sequence[Any], vectors: np.ndarray, source_id: Any
) -> dict[Any, float]:
    # This arithmetic deliberately mirrors the accepted M2 candidate builder:
    # stored FP32 rows, FP32 norm/division, then FP32 inner product.
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("embedding matrix must be finite and two-dimensional")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("zero-norm embedding")
    normalized = matrix / norms[:, None]
    position = {dataset_id: i for i, dataset_id in enumerate(dataset_ids)}
    if len(position) != len(dataset_ids) or source_id not in position:
        raise ValueError("dataset ID mapping is invalid")
    scores = normalized @ normalized[position[source_id]]
    return {dataset_id: float(scores[i]) for i, dataset_id in enumerate(dataset_ids)}


def ppr_source_order(
    nodes: Sequence[str], edges: Sequence[tuple[str, str]], source: str, candidates: Sequence[str]
) -> tuple[list[str], dict[str, float]]:
    """Use the frozen NetworkX 3.6.1 reference solver directly.

    This avoids treating the local sparse implementation as an unvalidated
    production acceleration.
    """
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    scores = nx.pagerank(
        graph, alpha=0.85,
        personalization={node: float(node == source) for node in nodes},
        dangling={node: float(node == source) for node in nodes},
        max_iter=1000, tol=1e-12, weight=None,
    )
    restricted = {candidate: scores[candidate] for candidate in candidates}
    return score_order(restricted), restricted


def node2vec_walks(
    nodes: Sequence[Any],
    edges: Sequence[tuple[Any, Any]],
    *,
    walk_length: int,
    num_walks: int,
    p: float,
    q: float,
    seed: int,
) -> list[list[Any]]:
    """Deterministic node2vec reference walks with canonical neighbor ordering."""
    if walk_length < 1 or num_walks < 1 or p <= 0 or q <= 0:
        raise ValueError("invalid node2vec walk parameter")
    ordered = sorted(set(nodes), key=id_sort_key)
    adjacency: dict[Any, set[Any]] = {node: set() for node in ordered}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    rng = np.random.default_rng(seed)
    walks: list[list[Any]] = []
    for _ in range(num_walks):
        roots = list(ordered)
        rng.shuffle(roots)
        for root in roots:
            walk = [root]
            while len(walk) < walk_length and adjacency[walk[-1]]:
                options = sorted(adjacency[walk[-1]], key=id_sort_key)
                if len(walk) == 1:
                    probabilities = np.full(len(options), 1.0 / len(options))
                else:
                    previous = walk[-2]
                    weights = np.array([
                        1.0 / p if nxt == previous else (
                            1.0 if nxt in adjacency[previous] else 1.0 / q
                        )
                        for nxt in options
                    ], dtype=np.float64)
                    probabilities = weights / weights.sum()
                walk.append(options[int(rng.choice(len(options), p=probabilities))])
            walks.append(walk)
    return walks


def train_node2vec_embeddings(
    graph: Any,
    nodes: Sequence[Any],
    *,
    seed: int,
) -> dict[Any, np.ndarray]:
    """Run the frozen node2vec 0.5.0 -> Gensim 4.3.3 training path.

    This is deliberately a thin, explicit wrapper around the approved package;
    it does not implement an alternate walker or embedding algorithm.  The
    process-level hash seed is part of the frozen configuration because Gensim
    receives the Python built-in ``hash`` function.
    """
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required for frozen node2vec training")
    from node2vec import Node2Vec

    model = Node2Vec(
        graph,
        dimensions=128,
        walk_length=80,
        num_walks=10,
        p=1.0,
        q=1.0,
        weight_key="weight",
        workers=1,
        sampling_strategy=None,
        quiet=True,
        temp_folder=None,
        seed=seed,
    ).fit(
        alpha=0.025,
        window=10,
        min_count=1,
        max_vocab_size=None,
        sample=0.0,
        seed=seed,
        workers=1,
        min_alpha=0.0001,
        sg=1,
        hs=0,
        negative=5,
        ns_exponent=0.75,
        cbow_mean=1,
        hashfxn=hash,
        epochs=5,
        null_word=0,
        trim_rule=None,
        sorted_vocab=1,
        batch_words=10000,
        compute_loss=False,
        callbacks=(),
        comment=None,
        max_final_vocab=None,
        shrink_windows=True,
    )
    ordered = sorted(nodes, key=id_sort_key)
    missing = [node for node in ordered if str(node) not in model.wv]
    if missing:
        raise RuntimeError(f"node2vec vocabulary missing expected nodes: {missing}")
    matrix = np.vstack([np.asarray(model.wv[str(node)], dtype=np.float32) for node in ordered])
    normalized = l2_normalize(matrix)
    vectors = {node: normalized[i].copy() for i, node in enumerate(ordered)}
    if any(vector.shape != (128,) or not np.isfinite(vector).all() or not np.any(vector) for vector in vectors.values()):
        raise RuntimeError("node2vec emitted an invalid embedding vector")
    return vectors


def typed_node_type(node: str) -> str:
    if "::" not in node:
        raise ValueError(f"untyped node: {node}")
    return node.split("::", 1)[0]


def metapath_walks(
    dataset_nodes: Sequence[str],
    typed_edges: Sequence[tuple[str, str]],
    metapaths: Sequence[tuple[str, str, str]],
    *,
    walk_length: int,
    walks_per_dataset_per_metapath: int,
    seed: int,
    progress=None,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Generate deterministic dataset-rooted, schema-constrained HIN walks."""
    if walk_length < 1 or walks_per_dataset_per_metapath < 1:
        raise ValueError("invalid metapath walk parameter")
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in typed_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    rng = np.random.default_rng(seed)
    walks: list[list[str]] = []
    omissions: list[dict[str, Any]] = []
    option_cache = {}
    for root_number, root in enumerate(sorted(dataset_nodes, key=id_sort_key), 1):
        if typed_node_type(root) != "Dataset":
            raise ValueError("all walk roots must be Dataset nodes")
        for schema in metapaths:
            if len(schema) != 3 or schema[0] != "Dataset" or schema[2] != "Dataset":
                raise ValueError(f"invalid frozen dataset-returning schema: {schema}")
            for repetition in range(walks_per_dataset_per_metapath):
                walk = [root]
                while len(walk) < walk_length:
                    expected_type = schema[len(walk) % 2]
                    option_key = (walk[-1], expected_type)
                    if option_key not in option_cache:
                        option_cache[option_key] = sorted(
                            (node for node in adjacency[walk[-1]] if typed_node_type(node) == expected_type),
                            key=id_sort_key,
                        )
                    options = option_cache[option_key]
                    if not options:
                        omissions.append({
                            "dataset_node": root,
                            "schema": "-".join(schema),
                            "repetition": repetition,
                            "stopped_at_length": len(walk),
                        })
                        walk = []
                        break
                    walk.append(options[int(rng.integers(len(options)))])
                if walk:
                    walks.append(walk)
        if progress is not None and (root_number % 128 == 0 or root_number == len(dataset_nodes)):
            progress(roots_complete=root_number, roots_total=len(dataset_nodes))
    return walks, omissions


def type_aware_negative_sample(
    positive_context: str,
    vocabulary: Sequence[str],
    counts: Mapping[str, int],
    *,
    count: int,
    rng: np.random.Generator,
) -> list[str]:
    """Official metapath2vec++ unigram^0.75 sampling within context type."""
    node_type = typed_node_type(positive_context)
    eligible, weights = _type_negative_distribution(node_type, vocabulary, counts)
    return [eligible[int(i)] for i in rng.choice(len(eligible), size=count, p=weights)]


def _type_negative_distribution(node_type, vocabulary, counts):
    """Immutable sampling inputs; constructing them consumes no random state."""
    eligible = sorted(
        (node for node in vocabulary if typed_node_type(node) == node_type), key=id_sort_key
    )
    if not eligible:
        raise ValueError(f"no negative-sampling vocabulary for type {node_type}")
    weights = np.array([float(counts[node]) ** 0.75 for node in eligible], dtype=np.float64)
    weights /= weights.sum()
    return eligible, weights


def train_metapath2vecpp_reference(
    walks: Sequence[Sequence[str]],
    *,
    embedding_dim: int,
    window: int,
    negative: int,
    initial_learning_rate: float,
    epochs: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Deterministic SGNS with type-aware negatives for reference/smoke use.

    It follows the authors' fixed-window, unigram^0.75, skip-gram updates.  The
    implementation is intentionally simple and is not asserted to be a
    production-scale acceleration.
    """
    if not walks or min(embedding_dim, window, negative, epochs) <= 0:
        raise ValueError("invalid metapath2vec++ corpus/configuration")
    vocabulary = sorted({str(node) for walk in walks for node in walk}, key=id_sort_key)
    index = {node: i for i, node in enumerate(vocabulary)}
    counts = Counter(str(node) for walk in walks for node in walk)
    negative_distributions = {
        node_type: _type_negative_distribution(node_type, vocabulary, counts)
        for node_type in sorted({typed_node_type(node) for node in vocabulary})
    }
    rng = np.random.default_rng(seed)
    syn0 = rng.uniform(-0.5 / embedding_dim, 0.5 / embedding_dim, (len(vocabulary), embedding_dim))
    syn1 = np.zeros_like(syn0)
    pair_count = sum(
        min(window, position) + min(window, len(walk) - position - 1)
        for walk in walks for position in range(len(walk))
    )
    total = max(1, pair_count * epochs)
    update = 0
    for _ in range(epochs):
        for walk in walks:
            tokens = [str(node) for node in walk]
            for center_position, center in enumerate(tokens):
                left = max(0, center_position - window)
                right = min(len(tokens), center_position + window + 1)
                for context_position in range(left, right):
                    if context_position == center_position:
                        continue
                    context = tokens[context_position]
                    alpha = max(
                        initial_learning_rate * 0.0001,
                        initial_learning_rate * (1.0 - update / total),
                    )
                    update += 1
                    eligible, weights = negative_distributions[typed_node_type(context)]
                    # The authors' ++ branch retains target == word collisions.
                    # That rejection exists only in their non-++ branch.
                    targets = [(center, 1.0)] + [
                        (eligible[int(i)], 0.0)
                        for i in rng.choice(len(eligible), size=negative, p=weights)
                    ]
                    input_row = index[context]
                    error = np.zeros(embedding_dim, dtype=np.float64)
                    input_before = syn0[input_row].copy()
                    for target, label in targets:
                        output_row = index[target]
                        dot = float(input_before @ syn1[output_row])
                        sigmoid = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, dot))))
                        gradient = (label - sigmoid) * alpha
                        error += gradient * syn1[output_row]
                        syn1[output_row] += gradient * input_before
                    syn0[input_row] += error
    normalized = l2_normalize(syn0)
    return {node: normalized[i] for i, node in enumerate(vocabulary)}


def parse_word2vec_text(path: Path) -> dict[str, np.ndarray]:
    """Parse the text vector format emitted by the authors' reference binary."""
    with path.open("r", encoding="utf-8") as handle:
        count, dimension = map(int, handle.readline().split())
        result: dict[str, np.ndarray] = {}
        for line in handle:
            pieces = line.split()
            if not pieces:
                continue
            result[pieces[0]] = np.asarray(pieces[1:], dtype=np.float64)
    if len(result) != count or any(vector.shape != (dimension,) for vector in result.values()):
        raise ValueError("invalid reference vector file")
    return result


def bge01_kernel(vectors: np.ndarray) -> np.ndarray:
    normalized = l2_normalize(vectors)
    return np.clip((1.0 + normalized @ normalized.T) / 2.0, 0.0, 1.0)


def mmr_order(
    candidate_ids: Sequence[Any], quality: Sequence[float], vectors: np.ndarray, *, lam: float
) -> list[Any]:
    if lam not in {0.0, 0.25, 0.5, 0.75, 1.0}:
        raise ValueError("lambda is outside the frozen grid")
    ids = list(candidate_ids)
    q = np.asarray(quality, dtype=np.float64)
    if len(ids) != len(q) or q.shape != (len(ids),) or not np.isfinite(q).all():
        raise ValueError("invalid quality vector")
    similarity = bge01_kernel(vectors)
    chosen: list[int] = []
    remaining = set(range(len(ids)))
    while remaining:
        values = {}
        for i in remaining:
            redundancy = max((float(similarity[i, j]) for j in chosen), default=0.0)
            values[i] = lam * float(q[i]) - (1.0 - lam) * redundancy
        best = min(remaining, key=lambda i: (-values[i], id_sort_key(ids[i])))
        chosen.append(best)
        remaining.remove(best)
    return [ids[i] for i in chosen]


def facility_location_order(candidate_ids: Sequence[Any], vectors: np.ndarray) -> list[Any]:
    ids = list(candidate_ids)
    similarity = bge01_kernel(vectors)
    current = np.zeros(len(ids), dtype=np.float64)
    chosen: list[int] = []
    remaining = set(range(len(ids)))
    while remaining:
        gains = {
            i: float(np.maximum(current, similarity[:, i]).sum() - current.sum())
            for i in remaining
        }
        best = min(remaining, key=lambda i: (-gains[i], id_sort_key(ids[i])))
        chosen.append(best)
        current = np.maximum(current, similarity[:, best])
        remaining.remove(best)
    return [ids[i] for i in chosen]


def farthest_first_order(
    candidate_ids: Sequence[Any], quality: Sequence[float], vectors: np.ndarray
) -> list[Any]:
    ids = list(candidate_ids)
    q = np.asarray(quality, dtype=np.float64)
    normalized = l2_normalize(vectors)
    if len(ids) != len(q) or not np.isfinite(q).all():
        raise ValueError("invalid quality vector")
    first = min(range(len(ids)), key=lambda i: (-float(q[i]), id_sort_key(ids[i])))
    chosen = [first]
    remaining = set(range(len(ids))) - {first}
    while remaining:
        radii = {
            i: min(float(np.linalg.norm(normalized[i] - normalized[j])) for j in chosen)
            for i in remaining
        }
        best = min(remaining, key=lambda i: (-radii[i], id_sort_key(ids[i])))
        chosen.append(best)
        remaining.remove(best)
    return [ids[i] for i in chosen]


def kdpp_kernel(quality: Sequence[float], vectors: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    q = 1.0 + np.asarray(quality, dtype=np.float64)
    if np.any(q < 1.0) or np.any(q > 2.0) or not np.isfinite(q).all():
        raise ValueError("MetaCompat quality must be finite in [0,1]")
    x = l2_normalize(vectors)
    b = q[:, None] * x
    raw = b @ b.T
    kernel = (raw + raw.T) / 2.0
    values, vectors_eig = np.linalg.eigh(kernel)
    minimum = float(values.min(initial=0.0))
    if minimum < -1e-10:
        raise ValueError("PILOT_FAILED: k-DPP kernel materially non-PSD")
    correction = 0.0
    if minimum < 0.0:
        corrected = np.maximum(values, 0.0)
        rebuilt = (vectors_eig * corrected) @ vectors_eig.T
        correction = float(np.linalg.norm(rebuilt - kernel))
        kernel = (rebuilt + rebuilt.T) / 2.0
    return kernel, {"minimum_eigenvalue": minimum, "psd_correction_frobenius_norm": correction}


def exact_kdpp_sample(
    candidate_ids: Sequence[Any],
    quality: Sequence[float],
    vectors: np.ndarray,
    *,
    k: int,
    seed: int,
) -> tuple[list[Any], dict[str, float]]:
    """Draw with the frozen DPPy 0.3.3 finite exact k-DPP GS path."""
    if k not in {5, 10, 20}:
        raise ValueError("k is outside the frozen set")
    kernel, audit = kdpp_kernel(quality, vectors)
    if k > np.linalg.matrix_rank(kernel):
        raise ValueError("BLOCKED: k exceeds validated DPP kernel rank")
    from dppy.finite_dpps import FiniteDPP

    sampler = FiniteDPP("likelihood", **{"L": kernel})
    selected = sampler.sample_exact_k_dpp(
        size=k, mode="GS", random_state=np.random.RandomState(seed)
    )
    if len(selected) != k or len(set(selected)) != k:
        raise RuntimeError("DPPy returned an invalid fixed-cardinality sample")
    ids = list(candidate_ids)
    # DPP membership is unordered; the frozen presentation order is quality
    # descending and then dataset ID ascending, and cannot change membership.
    selected_ids = [ids[i] for i in selected]
    quality_by_id = {dataset_id: float(quality[i]) for i, dataset_id in enumerate(ids)}
    selected_ids.sort(key=lambda item: (-quality_by_id[item], id_sort_key(item)))
    return selected_ids, audit


def semantic_ild(vectors: np.ndarray) -> float:
    normalized = l2_normalize(vectors)
    if len(normalized) < 2:
        raise ValueError("SemanticILD requires k >= 2")
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    upper = np.triu_indices(len(normalized), 1)
    return float(np.mean(1.0 - similarity[upper], dtype=np.float64))


def openml_structured_ild(coordinates: np.ndarray) -> float:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("OpenML StructuredILD requires k>=2 and three coordinates")
    distances = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            terms = []
            for a, b in zip(values[left], values[right]):
                if np.isnan(a) and np.isnan(b):
                    terms.append(0.0)
                elif np.isnan(a) or np.isnan(b):
                    terms.append(1.0)
                else:
                    terms.append(abs(float(a) - float(b)))
            distances.append(float(np.mean(terms, dtype=np.float64)))
    return float(np.mean(distances, dtype=np.float64))
