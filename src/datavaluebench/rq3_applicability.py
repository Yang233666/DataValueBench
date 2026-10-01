"""Reusable DataValueBench benchmark implementation."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .reference import openml_metacompat
from .rq3_ranking import kdpp_kernel

APPLICABLE = "APPLICABLE"
STRUCTURAL_NA = "NOT_APPLICABLE_STRUCTURAL_RANK_DEFICIENCY"
AUTHORITY_RELATIVE = "configs/rq3/applicability.json"
AUTHORITY_ID = "RQ3-k-DPP-applicability"
KEYS = ["source_dataset_id", "method", "method_operating_condition",
        "candidate_pool", "requested_depth", "replicate_seed"]
SOURCE_KEYS = KEYS[:-1]
METRICS = ["MetaCompat", "GTE_SemanticILD", "StructuredILD"]
STOCHASTIC = {"node2vec", "metapath2vec++", "k-DPP"}


def canonical_identity(value):
    """Preserve HF strings exactly; retain canonical integer OpenML identities."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, str) and value:
        return value
    raise ValueError("canonical identity must be a nonempty HF string or OpenML integer")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def authority_binding(root):
    path = Path(root) / AUTHORITY_RELATIVE
    return {"specification_id": AUTHORITY_ID, "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def classify_kernel(candidate_ids, quality, vectors):
    """Audit exactly the production kernel and its existing numerical rank."""
    ids = [canonical_identity(value) for value in candidate_ids]
    vectors = np.asarray(vectors)
    if not ids or len(ids) != len(set(ids)) or len(ids) != len(quality) or len(ids) != len(vectors):
        raise ValueError("incomplete or duplicate candidate identities")
    kernel, audit = kdpp_kernel(quality, vectors)
    rank = int(np.linalg.matrix_rank(kernel))
    return {"rank": rank, **audit,
            "rank_algorithm": "numpy.linalg.matrix_rank(L, tol=None, hermitian=False)",
            "numpy_version": np.__version__, "kernel_dtype": str(kernel.dtype),
            "kernel_sha256": hashlib.sha256(np.ascontiguousarray(kernel).tobytes()).hexdigest(),
            "candidate_ids": ids, "candidate_ids_sha256": digest_json(ids),
            "quality_sha256": digest_json([float(value) for value in quality]),
            "candidate_vectors_sha256": hashlib.sha256(np.ascontiguousarray(vectors).tobytes()).hexdigest(),
            "exact_distinct_stored_vectors": int(np.unique(vectors, axis=0).shape[0])}


def audit_openml_kdpp(source_ids, meta, candidates, bge_ids, bge_vectors, conditions,
                      seeds, progress=None):
    """Materialize every requested source/pool/depth, without sampling."""
    from threadpoolctl import threadpool_limits

    source_ids = [int(value) for value in source_ids]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate source identities")
    positions = {int(value): i for i, value in enumerate(bge_ids)}
    if len(positions) != len(bge_ids):
        raise ValueError("duplicate BGE identities")
    required = {}
    for cell in conditions:
        if cell["method"] == "k-DPP":
            required.setdefault((cell["candidate_pool"], cell["candidate_pool_size"]), set()).add(cell["requested_depth"])
    grouped = {int(s): g.sort_values("fused_rank", kind="stable")
               for s, g in candidates.groupby("source_dataset_id", sort=False)}
    rows, kernels = [], []
    # This is an audit only. The exact sampler retains its production path.
    with threadpool_limits(limits=1):
        for number, source in enumerate(sorted(source_ids), 1):
            pool = grouped[source]
            for (pool_name, size), depths in sorted(required.items()):
                part = pool.head(size)
                ids = part.candidate_dataset_id.astype(int).tolist()
                if len(ids) != size or source in ids or part.fused_rank.tolist() != list(range(1, size + 1)):
                    raise ValueError(f"Incomplete/noncanonical {pool_name} for source {source}")
                vectors = np.vstack([bge_vectors[positions[d]] for d in ids])
                quality = [openml_metacompat(meta[source], meta[d])["metacompat"] for d in ids]
                evidence = classify_kernel(ids, quality, vectors)
                base = {"ecosystem": "OpenML", "ecosystem_id": "OML", "source_dataset_id": source,
                        "method": "k-DPP", "method_operating_condition": "canonical",
                        "candidate_pool": pool_name, "candidate_pool_size": size, **evidence}
                kernels.append({**base, "required_depths": sorted(depths)})
                for depth in sorted(depths):
                    applicable = evidence["rank"] >= depth
                    rows.append({**base, "requested_depth": depth,
                                 "applicability_status": APPLICABLE if applicable else STRUCTURAL_NA,
                                 "reason": "rank(L) >= requested_k" if applicable else "rank(L) < requested_k",
                                 "replicate_seeds": list(seeds), "sampling_attempted": False})
            if progress is not None:
                progress(number, len(source_ids))
    na = [row for row in rows if row["applicability_status"] == STRUCTURAL_NA]
    counts = []
    for pool, depth in sorted({(r["candidate_pool"], r["requested_depth"]) for r in rows}):
        selected = [r for r in rows if (r["candidate_pool"], r["requested_depth"]) == (pool, depth)]
        n = sum(r["applicability_status"] == APPLICABLE for r in selected)
        counts.append({"candidate_pool": pool, "requested_depth": depth,
                       "expected_sources": len(selected), "applicable_sources": n,
                       "structural_na_sources": len(selected) - n,
                       "applicability_rate": n / len(selected)})
    return {"status": "PASS", "source_count": len(source_ids), "kernels_checked": len(kernels),
            "condition_count": len(rows), "kernel_audits": kernels, "conditions": rows,
            "not_applicable_source_conditions": na, "not_applicable_seed_cells": len(na) * len(seeds),
            "coverage": counts, "effectiveness_metrics_computed": False}


def logical_tuple(values):
    return tuple(None if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))
                 else value for value in values)


def expected_keys(sources, cells, seeds):
    return [(canonical_identity(source), cell["method"], cell["method_operating_condition"],
             cell["candidate_pool"], cell["requested_depth"], seed)
            for cell in cells for source in sources
            for seed in (seeds if cell["method"] in STOCHASTIC else (None,))]


def na_keys(report, seeds):
    return {(row["source_dataset_id"], "k-DPP", "canonical", row["candidate_pool"],
             row["requested_depth"], seed)
            for row in report["conditions"] if row["applicability_status"] == STRUCTURAL_NA
            for seed in seeds}


def validate_accounting(metrics, rankings, accounting, report, sources, cells, seeds):
    """Reject missing outputs, fabricated N/A, imputation, and absent rankings."""
    expected = expected_keys(sources, cells, seeds)
    na = na_keys(report, seeds)
    observed = [logical_tuple(row) for row in metrics[KEYS].itertuples(index=False, name=None)]
    accounted = [logical_tuple(row) for row in accounting[KEYS].itertuples(index=False, name=None)]
    ranked = {logical_tuple(row) for row in rankings[KEYS].itertuples(index=False, name=None)}
    checks = {
        "expected_identities_unique": len(expected) == len(set(expected)),
        "every_expected_cell_accounted": Counter(accounted) == Counter(expected),
        "all_applicable_metrics_complete": set(observed) == set(expected) - na and len(observed) == len(set(observed)),
        "all_applicable_rankings_present": ranked == set(expected) - na,
        "no_na_metrics_or_rankings": not (set(observed) & na or ranked & na),
    }
    correct = True
    for row in accounting.to_dict("records"):
        key = logical_tuple([row[k] for k in KEYS])
        wanted = STRUCTURAL_NA if key in na else "COMPLETE"
        correct &= row.get("execution_status") == wanted
        if key in na:
            correct &= row.get("reason") == "rank(L) < requested_k" and not row.get("sampling_attempted", True)
    checks["execution_status_matches_independent_applicability"] = bool(correct)
    return checks


def source_metric_values(metrics, seeds):
    """Check all frozen replicates before the SE-card within-source mean."""
    output = []
    for key, frame in metrics.groupby(SOURCE_KEYS, dropna=False, sort=True):
        observed = [logical_tuple([v])[0] for v in frame.replicate_seed]
        wanted = list(seeds) if key[1] in STOCHASTIC else [None]
        if Counter(observed) != Counter(wanted):
            raise ValueError(f"incomplete/duplicate source replicates: {key}")
        values = frame[METRICS].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite applicable metric")
        output.append({**dict(zip(SOURCE_KEYS, key)), **frame[METRICS].mean().to_dict(),
                       "seed_count": len(wanted)})
    return pd.DataFrame(output, columns=SOURCE_KEYS + METRICS + ["seed_count"])


def common_source_support(source_values, pool, depth, method_conditions):
    """An explicit comparison always uses exactly the methods' intersection."""
    if not method_conditions:
        raise ValueError("empty comparison")
    support = None
    for method, operating in method_conditions:
        rows = source_values[(source_values.candidate_pool == pool) &
                             (source_values.requested_depth == depth) &
                             (source_values.method == method) &
                             (source_values.method_operating_condition == operating)]
        ids = {canonical_identity(value) for value in rows.source_dataset_id}
        support = ids if support is None else support & ids
    return sorted(support)


def descriptive_and_common_summaries(metrics, report, sources, cells, seeds):
    """Diagnostic macro summaries; inferential procedures consume the same support helper."""
    values = source_metric_values(metrics, seeds)
    descriptive, comparative = [], []
    descriptive_keys = ["method", "method_operating_condition", "candidate_pool", "requested_depth"]
    for key in sorted({tuple(cell[name] for name in descriptive_keys) for cell in cells}):
        frame = values
        for name, value in zip(descriptive_keys, key):
            frame = frame[frame[name] == value]
        descriptive.append({**dict(zip(descriptive_keys, key)),
                            "expected_sources": len(sources), "applicable_sources": len(frame),
                            "applicability_rate": len(frame) / len(sources),
                            "metrics": frame[METRICS].mean().to_dict() if len(frame) else None})
    for pool, depth in sorted({(c["candidate_pool"], c["requested_depth"]) for c in cells}):
        active = sorted({(c["method"], c["method_operating_condition"]) for c in cells
                         if c["candidate_pool"] == pool and c["requested_depth"] == depth
                         and (c["method"] != "MMR" or c["method_operating_condition"] == "lambda=0.50")})
        support = common_source_support(values, pool, depth, active)
        for method, operating in active:
            frame = values[(values.candidate_pool == pool) & (values.requested_depth == depth) &
                           (values.method == method) & (values.method_operating_condition == operating) &
                           values.source_dataset_id.isin(support)]
            comparative.append({"method": method, "method_operating_condition": operating,
                                "candidate_pool": pool, "requested_depth": depth,
                                "comparison": "ALL_ACTIVE_CANONICAL_METHODS", "comparison_sample_size": len(support),
                                "source_ids_sha256": digest_json(support), "source_ids": support,
                                "metrics": frame[METRICS].mean().to_dict() if support else None})
    return {"evidence_status": "PILOT_DIAGNOSTIC_ONLY", "source_seed_aggregation": "MEAN_WITHIN_SOURCE",
            "applicability_coverage": report["coverage"], "method_descriptive": descriptive,
            "all_method_common_support": comparative, "na_imputation": False}
