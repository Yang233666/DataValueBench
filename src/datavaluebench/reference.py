"""Reusable DataValueBench benchmark implementation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


RQ4_SEED = 20260903
RQ4_FOLD_COUNT = 5
RQ4_FOLD_CAPACITY = 17
U1_EQUIVALENCE_TOLERANCE = 1e-12
U1_INTERACTION_THRESHOLD = 1e-10


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_uid(prefix: str, payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return prefix + _sha256_text(serialized)


def rq1_task_dataset_uid(task_id: int, dataset_id: int) -> str:
    return canonical_uid(
        "rq1td_",
        {
            "dataset_id": int(dataset_id),
            "namespace": "DVBench-v2-RQ1-TASKDATASET-v1.0",
            "task_id": int(task_id),
        },
    )


def rq4_player_uid(task_id: int, dataset_id: int) -> str:
    return canonical_uid(
        "rq4pl_",
        {
            "dataset_id": int(dataset_id),
            "namespace": "DVBench-v2-RQ4-PLAYER-v1.0",
            "task_id": int(task_id),
        },
    )


def rq1_observation_uid(row_index: int) -> str:
    return canonical_uid(
        "rq1obs_",
        {
            "namespace": "DVBench-v2-RQ1-OBS-v1.0",
            "row_index": int(row_index),
        },
    )


def rq4_observation_uid(row_index: int) -> str:
    return canonical_uid(
        "rq4obs_",
        {
            "namespace": "DVBench-v2-RQ4-OBS-v1.0",
            "row_index": int(row_index),
        },
    )


def build_player_registry(records: pd.DataFrame) -> pd.DataFrame:
    pairs = records[["task_id", "resolved_dataset_id"]].drop_duplicates().copy()
    if len(pairs) != 85:
        raise ValueError(f"expected 85 task-dataset players, observed {len(pairs)}")
    if pairs["task_id"].nunique() != 85 or pairs["resolved_dataset_id"].nunique() != 85:
        raise ValueError("task-dataset relation is not exactly one-to-one")
    counts = records.groupby("task_id", sort=False).size()
    pairs["task_id"] = pairs["task_id"].astype("int64")
    pairs["resolved_dataset_id"] = pairs["resolved_dataset_id"].astype("int64")
    pairs["rq1_task_dataset_uid"] = [
        rq1_task_dataset_uid(t, d)
        for t, d in pairs[["task_id", "resolved_dataset_id"]].itertuples(
            index=False, name=None
        )
    ]
    pairs["player_uid"] = [
        rq4_player_uid(t, d)
        for t, d in pairs[["task_id", "resolved_dataset_id"]].itertuples(
            index=False, name=None
        )
    ]
    pairs["observation_count"] = pairs["task_id"].map(counts).astype("int64")
    pairs["fold_tie_key"] = pairs["player_uid"].map(
        lambda uid: _sha256_text(
            f"DVBench-v2-RQ4-CROSSFIT|{RQ4_SEED}|{uid}"
        )
    )
    assignment_order = pairs.sort_values(
        ["observation_count", "fold_tie_key", "player_uid"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    fold_observations = [0] * RQ4_FOLD_COUNT
    fold_players = [0] * RQ4_FOLD_COUNT
    assignments: dict[str, int] = {}
    for row in assignment_order.itertuples(index=False):
        eligible = [
            fold
            for fold in range(RQ4_FOLD_COUNT)
            if fold_players[fold] < RQ4_FOLD_CAPACITY
        ]
        selected = min(
            eligible,
            key=lambda fold: (
                fold_observations[fold], fold_players[fold], fold
            ),
        )
        assignments[row.player_uid] = selected
        fold_observations[selected] += int(row.observation_count)
        fold_players[selected] += 1
    pairs["fold_id"] = pairs["player_uid"].map(assignments).astype("int64")
    pairs = pairs.sort_values("player_uid", kind="mergesort").reset_index(drop=True)
    pairs.insert(0, "canonical_player_order", np.arange(1, len(pairs) + 1))
    return pairs


def build_crosswalk(records: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    task_map = players.set_index("task_id")
    output = records[
        ["original_row_index", "task_id", "resolved_dataset_id", "flow_id", "run_id"]
    ].copy()
    output["rq1_observation_uid"] = output["original_row_index"].map(
        rq1_observation_uid
    )
    output["rq4_observation_uid"] = output["original_row_index"].map(
        rq4_observation_uid
    )
    output["rq1_task_dataset_uid"] = output["task_id"].map(
        task_map["rq1_task_dataset_uid"]
    )
    output["rq4_player_uid"] = output["task_id"].map(task_map["player_uid"])
    output["rq4_fold_id"] = output["task_id"].map(task_map["fold_id"])
    if output[["rq1_task_dataset_uid", "rq4_player_uid", "rq4_fold_id"]].isna().any().any():
        raise ValueError("crosswalk contains orphan observations")
    return output.sort_values("original_row_index", kind="mergesort").reset_index(drop=True)


@dataclass(frozen=True)
class PlayerEvidence:
    flow_ids: np.ndarray
    values: np.ndarray
    fold_id: int
    count: int
    total: float
    flow_counts: np.ndarray
    flow_sums: np.ndarray


class U1Reference:
    """Direct and sufficient-stat implementations of frozen RQ4 U1."""

    def __init__(self, records: pd.DataFrame, players: pd.DataFrame):
        task_to_uid = players.set_index("task_id")["player_uid"].to_dict()
        task_to_fold = players.set_index("task_id")["fold_id"].to_dict()
        frame = records[["task_id", "flow_id", "value"]].copy()
        frame["player_uid"] = frame["task_id"].map(task_to_uid)
        frame["fold_id"] = frame["task_id"].map(task_to_fold)
        self.players = tuple(sorted(players["player_uid"].astype(str)))
        self.fold_by_player = {
            str(row.player_uid): int(row.fold_id)
            for row in players.itertuples(index=False)
        }
        self.flow_ids = tuple(sorted(int(value) for value in frame["flow_id"].unique()))
        flow_position = {flow_id: pos for pos, flow_id in enumerate(self.flow_ids)}
        self.flow_position = flow_position
        self.evidence: dict[str, PlayerEvidence] = {}
        for player_uid, group in frame.groupby("player_uid", sort=False):
            flows = group["flow_id"].to_numpy(dtype=np.int64)
            values = group["value"].to_numpy(dtype=np.float64)
            positions = np.fromiter(
                (flow_position[int(flow_id)] for flow_id in flows),
                dtype=np.int64,
                count=len(flows),
            )
            flow_counts = np.bincount(
                positions, minlength=len(flow_position)
            ).astype(np.int64)
            flow_sums = np.bincount(
                positions, weights=values, minlength=len(flow_position)
            ).astype(np.float64)
            self.evidence[str(player_uid)] = PlayerEvidence(
                flow_ids=flows,
                values=values,
                fold_id=int(group["fold_id"].iloc[0]),
                count=len(values),
                total=float(values.sum(dtype=np.float64)),
                flow_counts=flow_counts,
                flow_sums=flow_sums,
            )

    def _coalition(self, coalition: Iterable[str]) -> frozenset[str]:
        result = frozenset(str(uid) for uid in coalition)
        unknown = result.difference(self.players)
        if unknown:
            raise KeyError(f"unknown player UIDs: {sorted(unknown)}")
        return result

    def direct(self, coalition: Iterable[str]) -> float:
        selected = self._coalition(coalition)
        if not selected:
            return 0.0
        heldout_utilities: list[float] = []
        for fold in range(RQ4_FOLD_COUNT):
            training = [uid for uid in selected if self.fold_by_player[uid] != fold]
            heldout = [uid for uid in self.players if self.fold_by_player[uid] == fold]
            if not training:
                heldout_utilities.extend([0.0] * len(heldout))
                continue
            training_flows = np.concatenate(
                [self.evidence[uid].flow_ids for uid in training]
            )
            training_values = np.concatenate(
                [self.evidence[uid].values for uid in training]
            )
            global_mean = float(training_values.mean(dtype=np.float64))
            flow_counts: dict[int, int] = {}
            flow_sums: dict[int, float] = {}
            for flow_id, value in zip(training_flows, training_values):
                key = int(flow_id)
                flow_counts[key] = flow_counts.get(key, 0) + 1
                flow_sums[key] = flow_sums.get(key, 0.0) + float(value)
            flow_means = {
                flow_id: flow_sums[flow_id] / flow_counts[flow_id]
                for flow_id in flow_counts
            }
            for uid in heldout:
                evidence = self.evidence[uid]
                predictions = np.fromiter(
                    (
                        flow_means.get(int(flow_id), global_mean)
                        for flow_id in evidence.flow_ids
                    ),
                    dtype=np.float64,
                    count=evidence.count,
                )
                mae = float(
                    np.abs(evidence.values - predictions).mean(dtype=np.float64)
                )
                heldout_utilities.append(1.0 - mae)
        return float(np.mean(heldout_utilities, dtype=np.float64))

    def sufficient(self, coalition: Iterable[str]) -> float:
        selected = self._coalition(coalition)
        if not selected:
            return 0.0
        heldout_utilities: list[float] = []
        for fold in range(RQ4_FOLD_COUNT):
            training = [uid for uid in selected if self.fold_by_player[uid] != fold]
            heldout = [uid for uid in self.players if self.fold_by_player[uid] == fold]
            if not training:
                heldout_utilities.extend([0.0] * len(heldout))
                continue
            total_count = sum(self.evidence[uid].count for uid in training)
            total_sum = sum(self.evidence[uid].total for uid in training)
            global_mean = total_sum / total_count
            flow_counts = np.sum(
                [self.evidence[uid].flow_counts for uid in training],
                axis=0,
                dtype=np.int64,
            )
            flow_sums = np.sum(
                [self.evidence[uid].flow_sums for uid in training],
                axis=0,
                dtype=np.float64,
            )
            flow_means = np.divide(
                flow_sums,
                flow_counts,
                out=np.full_like(flow_sums, global_mean),
                where=flow_counts > 0,
            )
            for uid in heldout:
                evidence = self.evidence[uid]
                positions = np.fromiter(
                    (self.flow_position[int(flow_id)] for flow_id in evidence.flow_ids),
                    dtype=np.int64,
                    count=evidence.count,
                )
                mae = float(
                    np.abs(evidence.values - flow_means[positions]).mean(
                        dtype=np.float64
                    )
                )
                heldout_utilities.append(1.0 - mae)
        return float(np.mean(heldout_utilities, dtype=np.float64))


def u0_player_scores(records: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    scores = (
        records.groupby(["task_id", "resolved_dataset_id"], as_index=False)["value"]
        .agg(observation_count="size", u0_player_score="mean")
        .merge(
            players[
                ["task_id", "resolved_dataset_id", "player_uid", "canonical_player_order"]
            ],
            on=["task_id", "resolved_dataset_id"],
            validate="one_to_one",
        )
    )
    return scores.sort_values("player_uid", kind="mergesort").reset_index(drop=True)


def task_flow_additive_mean(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    global_mean = float(train["value"].mean())
    task_mean = train.groupby("task_id")["value"].mean()
    residual = train["value"] - train["task_id"].map(task_mean).fillna(global_mean)
    flow_effect = residual.groupby(train["flow_id"]).mean()
    predictions = (
        test["task_id"].map(task_mean).fillna(global_mean)
        + test["flow_id"].map(flow_effect).fillna(0.0)
    ).to_numpy(dtype=np.float64)
    fallback = np.full(len(test), "none", dtype=object)
    unseen_task = ~test["task_id"].isin(task_mean.index).to_numpy()
    unseen_flow = ~test["flow_id"].isin(flow_effect.index).to_numpy()
    fallback[unseen_task] = "unseen_task_global_mean"
    fallback[unseen_flow & ~unseen_task] = "unseen_flow_zero_effect"
    fallback[unseen_flow & unseen_task] = "unseen_task_global_mean+unseen_flow_zero_effect"
    return predictions, fallback


def frequency_baseline(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    global_mean = float(train["value"].mean())
    means = train.groupby(["task_id", "flow_id"])["value"].mean()
    keys = list(map(tuple, test[["task_id", "flow_id"]].to_numpy()))
    predictions = np.array([means.get(key, global_mean) for key in keys], dtype=np.float64)
    fallback = np.array(
        ["none" if key in means.index else "unseen_task_flow_global_mean" for key in keys],
        dtype=object,
    )
    return predictions, fallback


def _tokens(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(
        token
        for token in str(value).replace(",", "|").split("|")
        if token and token != "unknown"
    )


def historical_jaccard(left: Any, right: Any) -> float:
    left_set, right_set = set(_tokens(left)), set(_tokens(right))
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def hf_metacompat(source: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    components = {
        "task_similarity": historical_jaccard(
            source.get("task_contexts"), candidate.get("task_contexts")
        ),
        "language_similarity": historical_jaccard(
            source.get("language"), candidate.get("language")
        ),
        "license_similarity": historical_jaccard(
            source.get("license"), candidate.get("license")
        ),
        "name_similarity": historical_jaccard(
            source.get("name_tokens"), candidate.get("name_tokens")
        ),
        "tag_similarity": historical_jaccard(source.get("tags"), candidate.get("tags")),
    }
    components["metacompat"] = (
        0.35 * components["task_similarity"]
        + 0.20 * components["language_similarity"]
        + 0.15 * components["license_similarity"]
        + 0.15 * components["name_similarity"]
        + 0.15 * components["tag_similarity"]
    )
    return components


def openml_historical_bin(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if numeric <= 10:
        return "tiny"
    if numeric <= 100:
        return "small"
    if numeric <= 1000:
        return "medium"
    if numeric <= 10000:
        return "large"
    return "xlarge"


def openml_metacompat(
    source: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    components = {
        "name_similarity": historical_jaccard(
            source.get("name_tokens"), candidate.get("name_tokens")
        ),
        "instance_bin_match": float(
            source.get("instance_bin") == candidate.get("instance_bin")
        ),
        "feature_bin_match": float(
            source.get("feature_bin") == candidate.get("feature_bin")
        ),
        "class_bin_match": float(source.get("class_bin") == candidate.get("class_bin")),
    }
    components["metacompat"] = (
        0.40 * components["name_similarity"]
        + 0.20 * components["instance_bin_match"]
        + 0.20 * components["feature_bin_match"]
        + 0.20 * components["class_bin_match"]
    )
    return components


def popularity_scores(corpus: pd.DataFrame) -> np.ndarray:
    downloads = np.log1p(
        pd.to_numeric(corpus["downloads"], errors="coerce").fillna(0).clip(lower=0)
    ).to_numpy(dtype=np.float64)
    likes = np.log1p(
        pd.to_numeric(corpus["likes"], errors="coerce").fillna(0).clip(lower=0)
    ).to_numpy(dtype=np.float64)
    combined = 0.7 * downloads + 0.3 * likes
    if len(combined) == 0 or np.nanmax(combined) <= np.nanmin(combined):
        return np.zeros_like(combined)
    return (combined - np.nanmin(combined)) / (
        np.nanmax(combined) - np.nanmin(combined)
    )


COMPLETENESS_FIELDS = (
    "author",
    "tags",
    "task_contexts",
    "task_categories",
    "language",
    "license",
    "downloads",
    "likes",
    "size_categories",
)


def metadata_completeness_scores(corpus: pd.DataFrame) -> np.ndarray:
    masks: list[np.ndarray] = []
    missing_tokens = {"", "nan", "none", "[]", "{}"}
    for field in COMPLETENESS_FIELDS:
        values = corpus[field].fillna("").astype(str).str.strip().str.lower()
        masks.append((~values.isin(missing_tokens)).to_numpy(dtype=np.float64))
    return np.vstack(masks).mean(axis=0, dtype=np.float64)


class SeedNamespaceError(ValueError):
    pass


def namespaced_rng(namespace: str, raw_seed: int) -> np.random.Generator:
    parts = namespace.split("/")
    if len(parts) != 4 or parts[0] != "DVBench-v2" or parts[1] not in {
        "RQ1", "RQ2", "RQ3", "RQ4", "MASTER"
    } or not parts[2] or not parts[3]:
        raise SeedNamespaceError(
            "namespace must match DVBench-v2/<RQ>/<COMPONENT>/<PURPOSE>"
        )
    if not isinstance(raw_seed, (int, np.integer)):
        raise SeedNamespaceError("raw seed must be an explicit integer")
    digest = hashlib.sha256(
        f"{namespace}|raw_seed={int(raw_seed)}".encode("utf-8")
    ).digest()
    entropy = np.frombuffer(digest, dtype=np.uint32).tolist()
    return np.random.default_rng(np.random.SeedSequence(entropy))


class ForbiddenLineageError(ValueError):
    pass


SHARED_RAW_CLASSES = {
    "shared_openml_raw",
    "shared_hf_raw",
    "rq1_rq4_provenance_crosswalk",
}
SHARED_INTRINSIC_CLASSES = {"shared_hf_intrinsic_representation"}
DERIVED_CLASSES = {"training", "calibration", "result", "ranking", "human_label"}


def validate_lineage_edge(
    source_rq: str,
    target_rq: str,
    artifact_class: str,
    *,
    source_sha256: str | None = None,
    expected_source_sha256: str | None = None,
) -> bool:
    source_rq, target_rq = source_rq.upper(), target_rq.upper()
    if source_rq == target_rq:
        return True
    if artifact_class in DERIVED_CLASSES:
        raise ForbiddenLineageError(
            f"derived artifact class {artifact_class} cannot cross {source_rq}->{target_rq}"
        )
    allowed = False
    if artifact_class == "shared_openml_raw":
        allowed = {source_rq, target_rq} == {"RQ1", "RQ4"}
    elif artifact_class == "rq1_rq4_provenance_crosswalk":
        allowed = {source_rq, target_rq} == {"RQ1", "RQ4"}
    elif artifact_class in {"shared_hf_raw", "shared_hf_intrinsic_representation"}:
        allowed = {source_rq, target_rq} == {"RQ2", "RQ3"}
    if not allowed:
        raise ForbiddenLineageError(
            f"artifact class {artifact_class} is not allowed on {source_rq}->{target_rq}"
        )
    if artifact_class in SHARED_INTRINSIC_CLASSES:
        if not source_sha256 or source_sha256 != expected_source_sha256:
            raise ForbiddenLineageError("shared intrinsic representation hash mismatch")
    return True


def representative_coalition_battery(players: Sequence[str], folds: Mapping[str, int]) -> list[set[str]]:
    ordered = sorted(str(uid) for uid in players)
    battery: list[set[str]] = [set(), {ordered[0]}, {ordered[-1]}, set(ordered)]
    for fold in range(RQ4_FOLD_COUNT):
        battery.append({uid for uid in ordered if folds[uid] == fold})
    for size in (2, 5, 12, 24, 42, 63):
        battery.append(set(ordered[:size]))
    return battery


def interaction_groups(players: Sequence[str]) -> list[list[str]]:
    ordered = sorted(
        (str(uid) for uid in players),
        key=lambda uid: (
            _sha256_text(
                f"DVBench-v2-RQ4-INTERACTION-SANITY|{RQ4_SEED}|{uid}"
            ),
            uid,
        ),
    )[:24]
    return [ordered[offset : offset + 8] for offset in (0, 8, 16)]


def exact_reference_subsets(players: pd.DataFrame) -> list[dict[str, Any]]:
    ordered = players.copy()
    ordered["quartile_tie_key"] = ordered["player_uid"].map(
        lambda uid: _sha256_text(
            f"DVBench-v2-RQ4-EXACT-QUARTILE|{RQ4_SEED}|{uid}"
        )
    )
    ordered = ordered.sort_values(
        ["observation_count", "quartile_tie_key", "player_uid"], kind="mergesort"
    ).reset_index(drop=True)
    quartiles = [ordered.iloc[:22], ordered.iloc[22:43], ordered.iloc[43:64], ordered.iloc[64:]]
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for replicate in range(5):
        salt = 0
        while True:
            chosen: list[str] = []
            for quartile in quartiles:
                ranked = sorted(
                    quartile["player_uid"].astype(str),
                    key=lambda uid: (
                        _sha256_text(
                            f"DVBench-v2-RQ4-EXACT-SUBSET|{RQ4_SEED}|{replicate}|{salt}|{uid}"
                        ),
                        uid,
                    ),
                )
                chosen.extend(ranked[:3])
            key = tuple(sorted(chosen))
            if key not in seen:
                seen.add(key)
                results.append(
                    {"replicate": replicate, "salt": salt, "player_uids": list(key)}
                )
                break
            salt += 1
    return results
