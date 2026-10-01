"""Reusable DataValueBench benchmark implementation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from datavaluebench.reference import canonical_uid


OPENML_SOURCE_SHA256 = "54d131d7071b920e8ef33016b4986abff9d642e673b3f345ce3bca9ec2875a04"
SPLIT_SEED = 20260903
INNER_SPLIT_SEED = 20260913
FINAL_SEEDS = tuple(range(20260903, 20260908))
BM25_TOKEN_PATTERN = re.compile(r"(?u)\b[\w\-:]+\b")


def _id_sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, np.integer)):
        return (0, int(value))
    return (1, str(value))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rq1_observation_uid(source_row_index: int, source_sha256: str = OPENML_SOURCE_SHA256) -> str:
    return canonical_uid(
        "rq1obs_",
        {
            "artifact_sha256": source_sha256,
            "namespace": "DVBench-v2-RQ1-OBS-v1.0",
            "source_row_index": int(source_row_index),
        },
    )


def rq4_observation_uid(source_row_index: int, source_sha256: str = OPENML_SOURCE_SHA256) -> str:
    return canonical_uid(
        "rq4obs_",
        {
            "artifact_sha256": source_sha256,
            "namespace": "DVBench-v2-RQ4-OBS-v1.0",
            "source_row_index": int(source_row_index),
        },
    )


def rq1_task_dataset_uid(task_id: int, dataset_id: int) -> str:
    return canonical_uid(
        "rq1td_",
        {
            "dataset_id": int(dataset_id),
            "namespace": "DVBench-v2-RQ1-TASKDATASET-v1.0",
            "task_id": int(task_id),
        },
    )


def rq1_flow_uid(flow_id: int) -> str:
    return canonical_uid(
        "rq1flow_",
        {"flow_id": int(flow_id), "namespace": "DVBench-v2-RQ1-FLOW-v1.0"},
    )


def build_rq1_cohort(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "original_row_index", "run_id", "task_id", "resolved_dataset_id",
        "flow_id", "upload_time", "value", "function",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"missing authoritative columns: {missing}")
    frame = source[list(required)].copy()
    if len(frame) != 106_907:
        raise ValueError(f"expected 106907 rows, observed {len(frame)}")
    expected_counts = {
        "task_id": 85, "resolved_dataset_id": 85, "flow_id": 1005, "run_id": 105807,
    }
    for field, expected in expected_counts.items():
        observed = int(frame[field].nunique(dropna=False))
        if observed != expected:
            raise ValueError(f"{field}: expected {expected}, observed {observed}")
    pairs = frame[["task_id", "resolved_dataset_id"]].drop_duplicates()
    if len(pairs) != 85 or pairs["task_id"].nunique() != 85 or pairs["resolved_dataset_id"].nunique() != 85:
        raise ValueError("task-dataset relation is not one-to-one")
    if set(frame["function"].astype(str)) != {"predictive_accuracy"}:
        raise ValueError("target function is not exactly predictive_accuracy")
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all() or not frame["value"].between(0.0, 1.0).all():
        raise ValueError("predictive accuracy is not finite in [0,1]")
    if frame["original_row_index"].duplicated().any():
        raise ValueError("source row indexes are not unique")
    frame = frame.rename(
        columns={
            "original_row_index": "source_row_index",
            "resolved_dataset_id": "dataset_id",
            "upload_time": "evaluation_timestamp",
            "value": "predictive_accuracy",
        }
    )
    for field in ("source_row_index", "run_id", "task_id", "dataset_id", "flow_id"):
        frame[field] = frame[field].astype("int64")
    frame["evaluation_timestamp"] = pd.to_datetime(frame["evaluation_timestamp"], utc=True)
    frame["observation_uid"] = frame["source_row_index"].map(rq1_observation_uid)
    frame["task_dataset_uid"] = [
        rq1_task_dataset_uid(t, d)
        for t, d in frame[["task_id", "dataset_id"]].itertuples(index=False, name=None)
    ]
    frame["flow_uid"] = frame["flow_id"].map(rq1_flow_uid)
    return frame.sort_values("observation_uid", kind="mergesort").reset_index(drop=True)


def _balanced_group_folds(
    frame: pd.DataFrame,
    group_field: str,
    group_uid_field: str,
    regime_id: str,
    capacity: int | None,
) -> dict[Any, int]:
    grouping_fields = [group_field] if group_field == group_uid_field else [group_field, group_uid_field]
    groups = (
        frame.groupby(grouping_fields, as_index=False)
        .size()
        .rename(columns={"size": "support"})
    )
    groups["tie_key"] = groups[group_uid_field].map(
        lambda uid: sha256_text(f"DVBench-v2-RQ1-BALANCE|{SPLIT_SEED}|{regime_id}|{uid}")
    )
    groups = groups.sort_values(
        ["support", "tie_key", group_uid_field], ascending=[False, True, True], kind="mergesort"
    )
    loads = [0] * 5
    counts = [0] * 5
    answer: dict[Any, int] = {}
    for row in groups.itertuples(index=False):
        eligible = [fold for fold in range(5) if capacity is None or counts[fold] < capacity]
        fold = min(eligible, key=lambda item: (loads[item], counts[item], item))
        answer[getattr(row, group_field)] = fold
        loads[fold] += int(row.support)
        counts[fold] += 1
    return answer


def _long_fivefold(frame: pd.DataFrame, regime: str, group_fold: pd.Series) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold in range(5):
        role = np.where(group_fold.to_numpy() == fold, "TEST", "TRAIN")
        parts.append(pd.DataFrame({
            "regime": regime,
            "condition": regime,
            "fold_id": fold,
            "observation_uid": frame["observation_uid"].to_numpy(),
            "role": role,
        }))
    return pd.concat(parts, ignore_index=True)


def _temporal_blocks(frame: pd.DataFrame) -> tuple[pd.Series, list[dict[str, Any]]]:
    grouped = frame.groupby("evaluation_timestamp", sort=True).size()
    cumulative = grouped.cumsum().to_numpy()
    timestamps = grouped.index.to_list()
    boundaries: list[int] = []
    receipts: list[dict[str, Any]] = []
    previous = 0
    for fraction in (0.20, 0.40, 0.60, 0.80):
        target = fraction * len(frame)
        candidates = range(max(previous, 0), len(cumulative) - 1)
        index = min(candidates, key=lambda i: (abs(int(cumulative[i]) - target), i))
        boundaries.append(index)
        previous = index + 1
        receipts.append({
            "fraction": fraction,
            "timestamp": timestamps[index].isoformat(),
            "cumulative_rows": int(cumulative[index]),
            "target_rows": target,
        })
    timestamp_block: dict[pd.Timestamp, int] = {}
    for index, timestamp in enumerate(timestamps):
        timestamp_block[timestamp] = 1 + sum(index > boundary for boundary in boundaries)
    return frame["evaluation_timestamp"].map(timestamp_block).astype("int8"), receipts


def build_rq1_splits(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Run identity is the actual group; its UID must not depend on one arbitrary row.
    run_uids = {run: canonical_uid("rq1run_", {"namespace": "DVBench-v2-RQ1-RUN-v1.0", "run_id": int(run)}) for run in frame["run_id"].unique()}
    g0_frame = frame.assign(run_uid=frame["run_id"].map(run_uids))
    g0_map = _balanced_group_folds(g0_frame, "run_id", "run_uid", "G0", None)
    gd_map = _balanced_group_folds(frame, "task_dataset_uid", "task_dataset_uid", "GD", 17)
    gf_map = _balanced_group_folds(frame, "flow_uid", "flow_uid", "GF", 201)
    pieces = [
        _long_fivefold(frame, "G0", frame["run_id"].map(g0_map)),
        _long_fivefold(frame, "GD", frame["task_dataset_uid"].map(gd_map)),
        _long_fivefold(frame, "GF", frame["flow_uid"].map(gf_map)),
    ]
    block, boundary_receipt = _temporal_blocks(frame)
    for train_through, condition in zip((1, 2, 3, 4), ("GT20", "GT40", "GT60", "GT80")):
        roles = np.where(block <= train_through, "TRAIN", np.where(block == train_through + 1, "TEST", "UNUSED"))
        pieces.append(pd.DataFrame({
            "regime": "GT", "condition": condition, "fold_id": 0,
            "observation_uid": frame["observation_uid"].to_numpy(), "role": roles,
        }))
    gd_fold = frame["task_dataset_uid"].map(gd_map).to_numpy()
    gf_fold = frame["flow_uid"].map(gf_map).to_numpy()
    gdf_stats: list[dict[str, Any]] = []
    for fold in range(5):
        held_task = gd_fold == fold
        held_flow = gf_fold == fold
        roles = np.where(held_task & held_flow, "TEST", np.where(~held_task & ~held_flow, "TRAIN", "EMBARGO"))
        pieces.append(pd.DataFrame({
            "regime": "GDF", "condition": "GDF", "fold_id": fold,
            "observation_uid": frame["observation_uid"].to_numpy(), "role": roles,
        }))
        test = frame.loc[roles == "TEST"]
        gdf_stats.append({
            "fold_id": fold,
            "test_rows": len(test),
            "test_task_datasets": int(test["task_dataset_uid"].nunique()),
            "test_flows": int(test["flow_uid"].nunique()),
            "train_rows": int(np.sum(roles == "TRAIN")),
            "embargo_rows": int(np.sum(roles == "EMBARGO")),
        })
    drr_fold = frame["observation_uid"].map(
        lambda uid: int(sha256_text(f"DVBench-v2-RQ1-ROW-RANDOM|{SPLIT_SEED}|{uid}"), 16)
    ).rank(method="first").astype("int64").sub(1).mod(5)
    pieces.append(_long_fivefold(frame, "DRR", drr_fold))
    split = pd.concat(pieces, ignore_index=True)
    gdf_union = int(split.query("regime == 'GDF' and role == 'TEST'")["observation_uid"].nunique())
    support_pass = all(
        row["test_rows"] >= 1000 and row["test_task_datasets"] >= 10 and row["test_flows"] >= 50
        for row in gdf_stats
    ) and gdf_union >= 10_000
    report = {
        "temporal_boundaries": boundary_receipt,
        "gdf_folds": gdf_stats,
        "gdf_union_test_rows": gdf_union,
        "gdf_support_pass": support_pass,
        "split_rows": len(split),
    }
    return split.sort_values(["regime", "condition", "fold_id", "observation_uid"], kind="mergesort"), report


def global_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.full(len(test), float(train["predictive_accuracy"].mean()), dtype=np.float64)


def grouped_mean(train: pd.DataFrame, test: pd.DataFrame, keys: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    global_value = float(train["predictive_accuracy"].mean())
    means = train.groupby(list(keys))["predictive_accuracy"].mean()
    test_keys = [tuple(row) for row in test[list(keys)].to_numpy()]
    if len(keys) == 1:
        test_keys = [row[0] for row in test_keys]
    values = np.array([means.get(key, global_value) for key in test_keys], dtype=np.float64)
    fallback = np.array(["none" if key in means.index else "global_mean" for key in test_keys], dtype=object)
    return values, fallback


def task_flow_additive_mean(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    global_value = float(train["predictive_accuracy"].mean())
    task_mean = train.groupby("task_id")["predictive_accuracy"].mean()
    residual = train["predictive_accuracy"] - train["task_id"].map(task_mean).fillna(global_value)
    flow_effect = residual.groupby(train["flow_id"]).mean()
    predictions = (
        test["task_id"].map(task_mean).fillna(global_value)
        + test["flow_id"].map(flow_effect).fillna(0.0)
    ).to_numpy(dtype=np.float64)
    unseen_task = ~test["task_id"].isin(task_mean.index).to_numpy()
    unseen_flow = ~test["flow_id"].isin(flow_effect.index).to_numpy()
    fallback = np.full(len(test), "none", dtype=object)
    fallback[unseen_task] = "unseen_task_global_mean"
    fallback[unseen_flow & ~unseen_task] = "unseen_flow_zero_effect"
    fallback[unseen_flow & unseen_task] = "unseen_task_global_mean+unseen_flow_zero_effect"
    return predictions, fallback


def make_identity_onehot_ridge(alpha: float):
    """Frozen sklearn pipeline; alpha must come from the frozen nested grid."""
    allowed = {10.0 ** exponent for exponent in range(-4, 4)}
    if float(alpha) not in allowed:
        raise ValueError("alpha is outside the frozen 1e-4..1e3 grid")
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    transform = ColumnTransformer(
        [("identity", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64), ["task_id", "flow_id"])],
        remainder="drop",
    )
    return Pipeline([
        ("identity", transform),
        ("ridge", Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr", tol=1e-4, max_iter=None)),
    ])


def make_representation_ridge(alpha: float):
    allowed = {10.0 ** exponent for exponent in range(-4, 4)}
    if float(alpha) not in allowed:
        raise ValueError("alpha is outside the frozen 1e-4..1e3 grid")
    from sklearn.linear_model import Ridge
    return Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr", tol=1e-4, max_iter=None)


def exact_bm25(documents: Sequence[str], query: str, *, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    tokenized = [BM25_TOKEN_PATTERN.findall(str(doc).lower()) for doc in documents]
    query_tokens = BM25_TOKEN_PATTERN.findall(str(query).lower())
    lengths = np.array([len(doc) for doc in tokenized], dtype=np.float64)
    avgdl = float(lengths.mean()) if len(lengths) else 0.0
    doc_counts = [Counter(doc) for doc in tokenized]
    scores = np.zeros(len(documents), dtype=np.float64)
    n_docs = len(documents)
    for token in query_tokens:  # repeated query tokens intentionally repeat contribution
        df = sum(token in counts for counts in doc_counts)
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        for i, counts in enumerate(doc_counts):
            tf = counts.get(token, 0)
            if not tf:
                continue
            denominator = tf + k1 * (1.0 - b + b * lengths[i] / avgdl)
            scores[i] += idf * tf * (k1 + 1.0) / denominator
    return scores


def reciprocal_rank_fusion(branches: Mapping[str, Sequence[Any]], *, k: int = 60) -> list[tuple[Any, float]]:
    totals: defaultdict[Any, float] = defaultdict(float)
    for ranking in branches.values():
        if len(set(ranking)) != len(ranking):
            raise ValueError("a branch ranking contains duplicate candidate IDs")
        for rank, candidate in enumerate(ranking, start=1):
            totals[candidate] += 1.0 / (k + rank)
    return sorted(totals.items(), key=lambda item: (-item[1], _id_sort_key(item[0])))


def exact_topk(candidate_ids: Sequence[Any], scores: Sequence[float], k: int, *, self_id: Any | None = None) -> list[Any]:
    if len(candidate_ids) != len(scores):
        raise ValueError("candidate IDs and scores must align")
    rows = [(candidate, float(score)) for candidate, score in zip(candidate_ids, scores) if candidate != self_id]
    if any(not math.isfinite(score) for _, score in rows):
        raise ValueError("non-finite exact retrieval score")
    return [candidate for candidate, _ in sorted(rows, key=lambda item: (-item[1], _id_sort_key(item[0])))[:k]]


def literal_candidate_prefixes(ranking: Sequence[Any]) -> dict[str, list[Any]]:
    if len(ranking) < 200:
        raise ValueError("C200 requires at least 200 candidates")
    c200 = list(ranking[:200])
    return {f"C{n}": c200[:n] for n in (20, 50, 100, 200)}


def ppr_power_iteration(
    nodes: Sequence[str], edges: Sequence[tuple[str, str]], source: str,
    *, alpha: float = 0.85, max_iter: int = 1000, tol: float = 1e-12,
) -> dict[str, float]:
    ordered = list(nodes)
    if source not in ordered:
        raise KeyError(source)
    pos = {node: index for index, node in enumerate(ordered)}
    neighbors: list[set[int]] = [set() for _ in ordered]
    for left, right in edges:
        if left == right:
            continue
        neighbors[pos[left]].add(pos[right])
        neighbors[pos[right]].add(pos[left])
    personalization = np.zeros(len(ordered), dtype=np.float64)
    personalization[pos[source]] = 1.0
    rank = personalization.copy()
    for _ in range(max_iter):
        nxt = (1.0 - alpha) * personalization
        dangling_mass = 0.0
        for index, value in enumerate(rank):
            if neighbors[index]:
                share = alpha * value / len(neighbors[index])
                for target in neighbors[index]:
                    nxt[target] += share
            else:
                dangling_mass += value
        nxt += alpha * dangling_mass * personalization
        if np.abs(nxt - rank).sum() < len(ordered) * tol:
            return {node: float(nxt[index]) for index, node in enumerate(ordered)}
        rank = nxt
    raise RuntimeError("PPR failed to converge within frozen max_iter=1000")


OPENML_QUALITY_COLUMNS = {
    "instances": "quality_NumberOfInstances",
    "features": "quality_NumberOfFeatures",
    "classes": "quality_NumberOfClasses",
}

RQ1_SOURCE_QUALITIES = (
    "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses",
    "NumberOfNumericFeatures", "NumberOfSymbolicFeatures", "NumberOfMissingValues",
    "NumberOfInstancesWithMissingValues", "MajorityClassSize", "MinorityClassSize",
    "MaxNominalAttDistinctValues",
)


def build_rq1_dataset_intrinsic(dataset_ids: Sequence[int], openml_metadata: pd.DataFrame) -> pd.DataFrame:
    """Build the frozen ten source and ten derived OpenML qualities, preserving NA."""
    renamed = {f"quality_{name}": name for name in RQ1_SOURCE_QUALITIES}
    required = {"did", *renamed}
    missing = sorted(required.difference(openml_metadata.columns))
    if missing:
        raise ValueError(f"missing intrinsic quality fields: {missing}")
    metadata = openml_metadata[["did", *renamed]].rename(columns=renamed).copy()
    for field in RQ1_SOURCE_QUALITIES:
        metadata[field] = _valid_nonnegative(metadata[field])
    output = pd.DataFrame({"dataset_id": sorted(set(int(item) for item in dataset_ids))})
    output = output.merge(metadata, left_on="dataset_id", right_on="did", how="left", validate="one_to_one").drop(columns="did")
    n, p, c = output.NumberOfInstances, output.NumberOfFeatures, output.NumberOfClasses
    pn, ps = output.NumberOfNumericFeatures, output.NumberOfSymbolicFeatures
    mv, mi = output.NumberOfMissingValues, output.NumberOfInstancesWithMissingValues
    maj, minority = output.MajorityClassSize, output.MinorityClassSize
    mnd = output.MaxNominalAttDistinctValues
    output["log1p_NumberOfInstances"] = np.log1p(n)
    output["log1p_NumberOfFeatures"] = np.log1p(p)
    output["log1p_NumberOfClasses"] = np.log1p(c)
    output["NumericFeatureFraction"] = pn.div(p).where(p > 0)
    output["SymbolicFeatureFraction"] = ps.div(p).where(p > 0)
    output["MissingValueFraction"] = mv.div(n * p).where((n > 0) & (p > 0))
    output["InstancesWithMissingFraction"] = mi.div(n).where(n > 0)
    output["MajorityClassFraction"] = maj.div(n).where(n > 0)
    output["MinorityClassFraction"] = minority.div(n).where(n > 0)
    output["log1p_MaxNominalAttDistinctValues"] = np.log1p(mnd)
    return output


def _valid_nonnegative(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    return values.where(np.isfinite(values) & (values >= 0.0))


def build_openmlval_representation(source: pd.DataFrame) -> pd.DataFrame:
    required = {"did", "name", *OPENML_QUALITY_COLUMNS.values()}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"missing OpenML-Val fields: {missing}")
    if len(source) != 6408 or source["did"].nunique(dropna=False) != 6408:
        raise ValueError("OpenML-Val must contain exactly 6408 unique source IDs")
    output = pd.DataFrame({"dataset_id": source["did"].astype("int64"), "name": source["name"].fillna("").astype(str)})
    for short, column in OPENML_QUALITY_COLUMNS.items():
        value = _valid_nonnegative(source[column])
        transformed = np.log1p(value)
        finite = transformed.dropna()
        if finite.empty:
            normalized = pd.Series(np.nan, index=output.index, dtype="float64")
        elif float(finite.max()) == float(finite.min()):
            normalized = transformed.notna().astype("float64") * 0.5
            normalized = normalized.where(transformed.notna())
        else:
            normalized = (transformed - float(finite.min())) / (float(finite.max()) - float(finite.min()))
        output[short] = value
        output[f"z_{short}"] = normalized
    output["canonical_text"] = output.apply(openml_canonical_text, axis=1)
    return output.sort_values("dataset_id", kind="mergesort").reset_index(drop=True)


def _lossless_number(value: Any) -> str | None:
    if pd.isna(value):
        return None
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return repr(number)


def openml_canonical_text(row: Mapping[str, Any]) -> str:
    lines: list[str] = []
    name = str(row.get("name", "")).strip()
    if name:
        lines.append(f"Name: {name}")
    for field, label in (("instances", "Number of instances"), ("features", "Number of features"), ("classes", "Number of classes")):
        serialized = _lossless_number(row.get(field))
        if serialized is not None:
            lines.append(f"{label}: {serialized}")
    return "\n".join(lines)


def structured_similarity(source: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    scores = []
    for field in ("z_instances", "z_features", "z_classes"):
        left, right = source.get(field), candidate.get(field)
        scores.append(0.0 if pd.isna(left) or pd.isna(right) else 1.0 - abs(float(left) - float(right)))
    return float(np.mean(scores, dtype=np.float64))


def build_openml_graph(representation: pd.DataFrame) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Build the frozen typed directed reciprocal OpenML HIN.

    Missing/invalid normalized coordinates intentionally create no incidence
    edge, but every dataset and all thirty typed bin nodes are retained.
    """
    nodes = [f"Dataset::{int(value)}" for value in representation["dataset_id"]]
    nodes.extend(f"{prefix}::{index}" for prefix in ("InstanceBin", "FeatureBin", "ClassBin") for index in range(10))
    edges: list[tuple[str, str, str]] = []
    for row in representation.itertuples(index=False):
        dataset = f"Dataset::{int(row.dataset_id)}"
        for field, prefix, forward_type, reverse_type in (
            ("z_instances", "InstanceBin", "HAS_INSTANCE_BIN", "INSTANCE_BIN_OF"),
            ("z_features", "FeatureBin", "HAS_FEATURE_BIN", "FEATURE_BIN_OF"),
            ("z_classes", "ClassBin", "HAS_CLASS_BIN", "CLASS_BIN_OF"),
        ):
            value = getattr(row, field)
            if pd.isna(value):
                continue
            bin_id = min(int(math.floor(10.0 * float(value))), 9)
            bin_node = f"{prefix}::{bin_id}"
            edges.append((dataset, bin_node, forward_type))
            edges.append((bin_node, dataset, reverse_type))
    return sorted(nodes), sorted(edges)


def mmr_select(relevance: Mapping[Any, float], similarity: Mapping[tuple[Any, Any], float], k: int, lam: float) -> list[Any]:
    remaining = set(relevance)
    chosen: list[Any] = []
    while remaining and len(chosen) < k:
        def objective(candidate: Any) -> tuple[float, str]:
            redundancy = max((similarity.get((candidate, item), similarity.get((item, candidate), 0.0)) for item in chosen), default=0.0)
            return (lam * relevance[candidate] - (1.0 - lam) * redundancy, _id_sort_key(candidate))
        best = sorted(remaining, key=lambda item: (-objective(item)[0], objective(item)[1]))[0]
        chosen.append(best)
        remaining.remove(best)
    return chosen


def facility_location_select(candidates: Sequence[Any], similarity: Mapping[tuple[Any, Any], float], k: int) -> list[Any]:
    universe = list(candidates)
    chosen: list[Any] = []
    current = {item: 0.0 for item in universe}
    while len(chosen) < min(k, len(universe)):
        gains = {}
        for candidate in universe:
            if candidate in chosen:
                continue
            gains[candidate] = sum(max(current[item], similarity.get((item, candidate), similarity.get((candidate, item), 0.0))) - current[item] for item in universe)
        best = sorted(gains, key=lambda item: (-gains[item], _id_sort_key(item)))[0]
        chosen.append(best)
        for item in universe:
            current[item] = max(current[item], similarity.get((item, best), similarity.get((best, item), 0.0)))
    return chosen


def farthest_first_select(candidates: Sequence[Any], similarity: Mapping[tuple[Any, Any], float], k: int) -> list[Any]:
    ordered = sorted(candidates, key=_id_sort_key)
    if not ordered:
        return []
    chosen = [ordered[0]]
    while len(chosen) < min(k, len(ordered)):
        remaining = [item for item in ordered if item not in chosen]
        best = sorted(
            remaining,
            key=lambda item: (-min(1.0 - similarity.get((item, selected), similarity.get((selected, item), 0.0)) for selected in chosen), _id_sort_key(item)),
        )[0]
        chosen.append(best)
    return chosen


def exact_rule_values(utilities: Sequence[float], players: Sequence[str]) -> dict[str, dict[str, float]]:
    m = len(players)
    if len(utilities) != 1 << m:
        raise ValueError("utility vector must contain every coalition in binary-mask order")
    output = {rule: {player: 0.0 for player in players} for rule in ("standalone", "loo", "shapley", "banzhaf", "beta_4_1")}
    full = (1 << m) - 1
    for i, player in enumerate(players):
        bit = 1 << i
        output["standalone"][player] = float(utilities[bit])
        output["loo"][player] = float(utilities[full] - utilities[full ^ bit])
        for mask in range(1 << m):
            if mask & bit:
                continue
            size = mask.bit_count()
            marginal = float(utilities[mask | bit] - utilities[mask])
            output["shapley"][player] += marginal / (m * math.comb(m - 1, size))
            output["banzhaf"][player] += marginal / (2 ** (m - 1))
            beta_weight = beta_cardinality_weights(m)[size] / math.comb(m - 1, size)
            output["beta_4_1"][player] += marginal * beta_weight
    return output


@lru_cache(maxsize=None)
def beta_cardinality_weights(m: int) -> tuple[float, ...]:
    """Frozen Beta-binomial q_k for alpha=4, beta=1."""
    raw = []
    for k in range(m):
        log_beta_ratio = (
            math.lgamma(k + 1) + math.lgamma(m - k + 3) - math.lgamma(m + 4)
            - (math.lgamma(1) + math.lgamma(4) - math.lgamma(5))
        )
        raw.append(math.comb(m - 1, k) * math.exp(log_beta_ratio))
    total = math.fsum(raw)
    return tuple(value / total for value in raw)


def rank_consequence(values: Mapping[str, float], utility, k_values: Sequence[int]) -> list[dict[str, Any]]:
    descending = sorted(values, key=lambda uid: (-values[uid], uid))
    ascending = list(reversed(descending))
    rows: list[dict[str, Any]] = []
    for k in k_values:
        rows.append({"direction": "addition", "k": int(k), "utility": float(utility(descending[:k]))})
        rows.append({"direction": "removal", "k": int(k), "utility": float(utility(descending[k:]))})
        rows.append({"direction": "reverse_addition", "k": int(k), "utility": float(utility(ascending[:k]))})
    return rows


def monte_carlo_shapley(utility, players: Sequence[str], permutations: int, rng: np.random.Generator) -> dict[str, float]:
    if permutations <= 0:
        raise ValueError("permutation budget must be supplied by frozen authority")
    totals = {player: 0.0 for player in players}
    ordered = np.asarray(list(players), dtype=object)
    for _ in range(permutations):
        permutation = rng.permutation(ordered).tolist()
        coalition: list[str] = []
        before = float(utility(coalition))
        for player in permutation:
            coalition.append(str(player))
            after = float(utility(coalition))
            totals[str(player)] += after - before
            before = after
    return {player: value / permutations for player, value in totals.items()}


def monte_carlo_banzhaf(utility, players: Sequence[str], samples: int, rng: np.random.Generator) -> dict[str, float]:
    if samples <= 0:
        raise ValueError("sample budget must be supplied by frozen authority")
    totals = {player: 0.0 for player in players}
    for player in players:
        others = [item for item in players if item != player]
        for _ in range(samples):
            mask = rng.integers(0, 2, size=len(others), dtype=np.int8)
            coalition = [item for item, keep in zip(others, mask) if keep]
            totals[player] += float(utility([*coalition, player])) - float(utility(coalition))
        totals[player] /= samples
    return totals


def monte_carlo_beta41(utility, players: Sequence[str], samples: int, rng: np.random.Generator) -> dict[str, float]:
    if samples <= 0:
        raise ValueError("sample budget must be supplied by frozen authority")
    q = np.asarray(beta_cardinality_weights(len(players)), dtype=np.float64)
    totals = {player: 0.0 for player in players}
    for player in players:
        others = np.asarray([item for item in players if item != player], dtype=object)
        for _ in range(samples):
            size = int(rng.choice(len(players), p=q))
            coalition = rng.choice(others, size=size, replace=False).tolist() if size else []
            totals[player] += float(utility([*coalition, player])) - float(utility(coalition))
        totals[player] /= samples
    return totals


def validate_nested_split(frame: pd.DataFrame, split: pd.DataFrame, regime: str, fold_id: int) -> bool:
    subset = split[(split["regime"] == regime) & (split["fold_id"] == fold_id)]
    if subset["observation_uid"].duplicated().any():
        return False
    merged = frame.merge(subset[["observation_uid", "role"]], on="observation_uid", validate="one_to_one")
    train, test = merged[merged.role == "TRAIN"], merged[merged.role == "TEST"]
    if regime == "G0":
        return set(train.run_id).isdisjoint(test.run_id)
    if regime == "GD":
        return set(train.task_dataset_uid).isdisjoint(test.task_dataset_uid)
    if regime == "GF":
        return set(train.flow_uid).isdisjoint(test.flow_uid)
    if regime == "GDF":
        return set(train.task_dataset_uid).isdisjoint(test.task_dataset_uid) and set(train.flow_uid).isdisjoint(test.flow_uid)
    return True


def _inner_assign_groups(
    frame: pd.DataFrame,
    group_field: str,
    regime: str,
    condition: str,
    outer_fold: int,
    capacities: Sequence[int] | None,
) -> pd.Series:
    groups = frame.groupby(group_field, as_index=False).size().rename(columns={"size": "support"})
    groups["tie_key"] = groups[group_field].map(
        lambda uid: sha256_text(
            f"DVBench-v2-RQ1-INNER|{regime}|{condition}|{outer_fold}|{uid}|{INNER_SPLIT_SEED}"
        )
    )
    groups = groups.sort_values(["support", "tie_key", group_field], ascending=[False, True, True], kind="mergesort")
    loads = [0, 0, 0]
    counts = [0, 0, 0]
    mapping: dict[Any, int] = {}
    for row in groups.itertuples(index=False):
        eligible = [fold for fold in range(3) if capacities is None or counts[fold] < capacities[fold]]
        fold = min(eligible, key=lambda item: (loads[item], counts[item], item))
        mapping[getattr(row, group_field)] = fold
        loads[fold] += int(row.support)
        counts[fold] += 1
    return frame[group_field].map(mapping).astype("int8")


def build_rq1_inner_splits(
    frame: pd.DataFrame,
    outer_split: pd.DataFrame,
    regime: str,
    condition: str,
    outer_fold: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outer = outer_split[
        (outer_split["regime"] == regime)
        & (outer_split["condition"] == condition)
        & (outer_split["fold_id"] == outer_fold)
        & (outer_split["role"] == "TRAIN")
    ][["observation_uid"]]
    training = frame.merge(outer, on="observation_uid", validate="one_to_one")
    roles_by_fold: list[np.ndarray] = []
    if regime == "GT":
        grouped = training.groupby("evaluation_timestamp", sort=True).size()
        cumulative = grouped.cumsum().to_numpy()
        stamps = grouped.index.to_list()
        boundaries = []
        previous = 0
        for fraction in (0.25, 0.50, 0.75):
            target = fraction * len(training)
            idx = min(range(previous, len(cumulative) - 1), key=lambda i: (abs(int(cumulative[i]) - target), i))
            boundaries.append(idx)
            previous = idx + 1
        mapping = {stamp: 1 + sum(index > boundary for boundary in boundaries) for index, stamp in enumerate(stamps)}
        block = training["evaluation_timestamp"].map(mapping).to_numpy()
        for fold in range(3):
            roles_by_fold.append(np.where(block <= fold + 1, "TRAIN", np.where(block == fold + 2, "VALIDATION", "UNUSED")))
    elif regime == "GDF":
        task_fold = _inner_assign_groups(training, "task_dataset_uid", regime + "-TD", condition, outer_fold, (23, 23, 22)).to_numpy()
        flow_fold = _inner_assign_groups(training, "flow_uid", regime + "-FLOW", condition, outer_fold, (268, 268, 268)).to_numpy()
        for fold in range(3):
            held_task, held_flow = task_fold == fold, flow_fold == fold
            roles_by_fold.append(np.where(held_task & held_flow, "VALIDATION", np.where(~held_task & ~held_flow, "TRAIN", "EMBARGO")))
    else:
        field, capacities = {
            "G0": ("run_id", None), "GD": ("task_dataset_uid", (23, 23, 22)),
            "GF": ("flow_uid", (268, 268, 268)),
        }[regime]
        assignment = _inner_assign_groups(training, field, regime, condition, outer_fold, capacities).to_numpy()
        roles_by_fold = [np.where(assignment == fold, "VALIDATION", "TRAIN") for fold in range(3)]
    parts, stats = [], []
    for fold, roles in enumerate(roles_by_fold):
        part = pd.DataFrame({
            "outer_regime": regime, "outer_condition": condition, "outer_fold_id": outer_fold,
            "inner_fold_id": fold, "observation_uid": training["observation_uid"].to_numpy(), "role": roles,
        })
        validation = training.loc[roles == "VALIDATION"]
        record = {"inner_fold_id": fold, "training_rows": int(np.sum(roles == "TRAIN")), "validation_rows": len(validation), "embargo_rows": int(np.sum(roles == "EMBARGO"))}
        if regime == "GDF":
            record.update({"validation_task_datasets": int(validation.task_dataset_uid.nunique()), "validation_flows": int(validation.flow_uid.nunique())})
        if regime == "GT" and len(validation):
            inner_train = training.loc[roles == "TRAIN"]
            record["strict_chronology"] = bool(inner_train.evaluation_timestamp.max() < validation.evaluation_timestamp.min())
        stats.append(record)
        parts.append(part)
    support_pass = True
    if regime == "GDF":
        support_pass = all(item["validation_rows"] >= 500 and item["validation_task_datasets"] >= 5 and item["validation_flows"] >= 25 for item in stats)
    return pd.concat(parts, ignore_index=True), {"regime": regime, "condition": condition, "outer_fold_id": outer_fold, "folds": stats, "support_pass": support_pass}
