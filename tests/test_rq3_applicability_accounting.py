import copy

import numpy as np
import pandas as pd
import pytest

from datavaluebench.rq3_applicability import (
    APPLICABLE, STRUCTURAL_NA, KEYS, METRICS, classify_kernel,
    validate_accounting, source_metric_values, common_source_support,
    descriptive_and_common_summaries,
)


SEEDS = (20260903, 20260904, 20260905, 20260906, 20260907)


def accounting_fixture():
    cells = [{"method": method, "method_operating_condition": "canonical",
              "candidate_pool": "C20_v2", "candidate_pool_size": 20, "requested_depth": 10}
             for method in ("Random", "k-DPP")]
    conditions = [{"source_dataset_id": s, "candidate_pool": "C20_v2", "requested_depth": 10,
                   "applicability_status": STRUCTURAL_NA if s == 1 else APPLICABLE}
                  for s in (1, 2)]
    metrics, rankings, accounting = [], [], []
    for cell in cells:
        for source in (1, 2):
            for seed in SEEDS if cell["method"] == "k-DPP" else (None,):
                key = {name: cell[name] for name in KEYS if name not in {"source_dataset_id", "replicate_seed"}}
                key.update(source_dataset_id=source, replicate_seed=seed)
                na = source == 1 and cell["method"] == "k-DPP"
                accounting.append({**key, "execution_status": STRUCTURAL_NA if na else "COMPLETE",
                                   "reason": "rank(L) < requested_k" if na else None, "sampling_attempted": not na})
                if not na:
                    metrics.append({**key, **{name: float(source) for name in METRICS}})
                    rankings.append({**key, "rank": 1, "candidate_dataset_id": 20})
    return pd.DataFrame(metrics), pd.DataFrame(rankings), pd.DataFrame(accounting), {"conditions": conditions, "coverage": []}, [1, 2], cells, SEEDS


def test_complete_applicable_and_na_cells_pass_without_imputation():
    assert all(validate_accounting(*accounting_fixture()).values())


@pytest.mark.parametrize("mutation", ["missing_accounting", "missing_metrics", "missing_rankings", "duplicate", "fake_na", "imputed", "sampled_na"])
def test_corrupt_or_partial_cells_fail(mutation):
    metrics, rankings, accounting, report, sources, cells, seeds = accounting_fixture()
    if mutation == "missing_accounting": accounting = accounting.iloc[1:]
    if mutation == "missing_metrics": metrics = metrics.iloc[1:]
    if mutation == "missing_rankings": rankings = rankings.iloc[1:]
    if mutation == "duplicate": metrics = pd.concat([metrics, metrics.iloc[:1]])
    if mutation == "fake_na": accounting.loc[0, "execution_status"] = STRUCTURAL_NA
    if mutation == "imputed":
        fake = accounting[accounting.execution_status == STRUCTURAL_NA].iloc[0].to_dict()
        fake.update({name: 0. for name in METRICS})
        metrics = pd.concat([metrics, pd.DataFrame([fake])])
    if mutation == "sampled_na": accounting.loc[accounting.execution_status == STRUCTURAL_NA, "sampling_attempted"] = True
    assert not all(validate_accounting(metrics, rankings, accounting, report, sources, cells, seeds).values())


def test_descriptions_keep_coverage_and_comparisons_use_intersection():
    metrics, _, _, report, sources, cells, seeds = accounting_fixture()
    values = source_metric_values(metrics, seeds)
    assert len(values) == 3  # Five seeds produce one source value.
    assert common_source_support(values, "C20_v2", 10, [("Random", "canonical"), ("k-DPP", "canonical")]) == [2]
    summary = descriptive_and_common_summaries(metrics, report, sources, cells, seeds)
    descriptions = {row["method"]: row for row in summary["method_descriptive"]}
    assert descriptions["Random"]["applicable_sources"] == 2
    assert descriptions["k-DPP"]["applicability_rate"] == .5
    assert descriptions["Random"]["metrics"]["MetaCompat"] == 1.5
    assert all(row["comparison_sample_size"] == 1 and row["metrics"]["MetaCompat"] == 2.
               for row in summary["all_method_common_support"])


def test_incomplete_seed_replicates_cannot_enter_summaries():
    metrics, *_ = accounting_fixture()
    with pytest.raises(ValueError, match="source replicates"):
        source_metric_values(metrics.iloc[:-1], SEEDS)


def test_kernel_rank_evidence_preserves_frozen_kernel_and_rejects_invalid_ids():
    vectors = np.tile(np.eye(3), (7, 1))[:20]
    evidence = classify_kernel(range(20), np.ones(20), vectors)
    assert evidence["rank"] == 3
    assert evidence["exact_distinct_stored_vectors"] == 3
    assert len(evidence["kernel_sha256"]) == 64
    with pytest.raises(ValueError, match="duplicate candidate"):
        classify_kernel([1] * 20, np.ones(20), vectors)


def test_zero_applicable_method_has_explicit_zero_coverage_and_no_metric():
    metrics, _, _, report, sources, cells, seeds = accounting_fixture()
    metrics = metrics[metrics.method != "k-DPP"]
    summary = descriptive_and_common_summaries(metrics, report, sources, cells, seeds)
    dpp = next(row for row in summary["method_descriptive"] if row["method"] == "k-DPP")
    assert dpp["applicable_sources"] == 0 and dpp["metrics"] is None
    assert all(row["comparison_sample_size"] == 0 and row["metrics"] is None
               for row in summary["all_method_common_support"])
