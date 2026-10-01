# DataValueBench

DataValueBench evaluates datasets in four parallel settings: workflow performance estimation, metadata-based dataset retrieval, dataset alternative ranking, and task-level utility contribution. Each setting has its own information boundary, comparison support and endpoints. There is no universal DataValue score or pooled cross-setting leaderboard.

## What DataValueBench evaluates

| Setting | Evaluation | Released scope |
|---|---|---|
| RQ1 · Workflow Performance Estimation | Estimate historical workflow accuracy | G0, GT80, GD and GF; 20 methods/conditions |
| RQ2 · Metadata-Based Dataset Retrieval | Structured and natural-language queries with automatic metadata-derived relevance | 100,000-record corpus; 500 latent intents; 475 realized Track B intents used for common comparison support |
| RQ3 · Dataset Alternative Ranking | Metadata compatibility and diversity | Separate OpenML and Hugging Face ecosystems; primary C100, k=10 |
| RQ4 · Task-Level Utility Contribution | Compare six contribution rules under two utility structures | Two fixed 85-player games |

## Quick start

Core verification is tested on Linux with Python 3.10.12. Graph reference tests use a separate Python 3.11 environment. Setup downloads Python dependencies into a local environment; verification is CPU-only and offline, with no model inference.

```bash
(
  bash scripts/setup.sh
  bash scripts/run_tests.sh
  bash scripts/verify_release.sh
)
```

To use an existing environment, set `PYTHON_BIN` to its interpreter and run `bash scripts/setup.sh --check`. Graph reference tests use a separate pinned environment; see [reproducibility](docs/REPRODUCIBILITY.md).

## Repository structure

| Directory | Contents |
|---|---|
| [data/](data/README.md) | Workloads, qrels, identity dictionaries, splits, candidates and player definitions |
| [results/](results/README.md) | Frozen per-intent/per-player outputs, aggregates, uncertainty and support statistics |
| [configs/](configs/README.md) | Scientific method, evaluation and index settings |
| [models/](models/README.md) | Exact checkpoint identities, model roles and upstream terms |
| [prompts/](prompts/README.md) | Query-generation and semantic-validation prompts/schema |
| [src/datavaluebench/](src/datavaluebench/) | Reusable benchmark implementations and verification loaders |
| [scripts/](scripts/) | Setup, tests, release verification and per-setting entry points |
| [tests/](tests/) | Correctness and reference tests |
| [docs/](docs/BENCHMARK.md) | Benchmark definitions, data boundaries and reproduction instructions |

## Data

Bundled inputs include 106,907 historical observation identities, 85 task/dataset identities, 1,005 workflow identities, exact reported split memberships, the exact intrinsic features used in the reported results and frozen MiniLM vectors, all 500 latent intents and structured queries, 475 realized natural-language queries, automatic qrels, 25 explicitly unrealized slots, complete RQ3 C200 candidate memberships, and 85 RQ4 player/fold definitions with U0 utility inputs. Smaller RQ3 candidate pools are literal ordered prefixes.

The release separates author-created benchmark inputs and derived results from upstream dataset content. Historical accuracy observations, full metadata text and other external inputs with unresolved redistribution terms are documented with exact identities in [the data guide](docs/DATA.md). The package does not refresh or substitute those inputs. HF sources use hash-based identifiers; one credential-shaped upstream lookup name is omitted without dropping the corresponding source identity.

## Models

[Exact revisions and roles](models/model_manifest.json) cover MiniLM, BGE-M3, Multilingual E5, GTE multilingual, SPLADE, ColBERTv2, Qwen3-14B and the recorded Astra validator. BGE-M3 constructs RQ3 candidates and supplies its ranking baseline; GTE supplies the separate SemanticILD evaluation space. Model weights are obtained from their providers only when doing upstream inference. They are unnecessary for the quick start.

## Verifying the released benchmark results

The following commands replay the supported downstream verification and reconstruction steps from the released frozen inputs and results. They do not refit estimators, rebuild retrieval indexes, rerun retrieval, or estimate new contribution values.

```bash
(
  bash scripts/reproduce_rq1.sh --output outputs/rq1
  bash scripts/reproduce_rq2.sh --output outputs/rq2
  bash scripts/reproduce_rq3.sh --output outputs/rq3
  bash scripts/reproduce_rq4.sh --output outputs/rq4
)
```

Each command writes `verification.json` into a new output directory and fails on missing or changed inputs. RQ1 verifies the reported results, split support and undefined correlations. RQ2 recomputes macro retrieval scores from 5,700 released per-intent rows. RQ3 verifies separate result/support tables and all candidate identities. RQ4 replays the existing same-rule comparison code from frozen player values. These commands do not refit models, rerun retrieval or estimate new contributions. See [RQ1](docs/RQ1.md), [RQ2](docs/RQ2.md), [RQ3](docs/RQ3.md) and [RQ4](docs/RQ4.md).

## Verification

`checksums.json` covers released inputs, results, code and documentation. `verify_release.sh` checks file hashes, schemas and scientific support invariants. Core tests cover deterministic construction, undefined statistics, query semantics and contribution comparisons; optional graph tests check exact k-DPP and pinned PPR. [Schema documentation](docs/DATA.md) explains joins and normalized support storage.

## Full reconstruction

Three modes are distinct: **bundled-data/result verification** is supported; **reproduction from frozen inputs** supports the released downstream calculations and candidate/support inspection; **full upstream reconstruction** requires additional exact historical inputs, model/index construction and method-specific environments. Exact public archive URLs for some historical inputs are not established. No turnkey end-to-end rebuild is claimed. [Reproducibility instructions](docs/REPRODUCIBILITY.md) identify these dependencies without silently using latest data or models.

## Scope and limitations

RQ1 excludes GDF because its prespecified support requirement is unmet, and excludes computational efficiency. GT80 GlobalMean Spearman remains undefined. RQ2's 25 unrealized slots never receive zero retrieval scores; query fidelity validation is distinct from human relevance. RQ2 human relevance and secondary reranking, and RQ3 human suitability, are not reported. RQ3 structural N/A remains explicit; objective/evaluation overlap is disclosed. RQ4 is descriptive fixed-game analysis, with no population, causal, economic-value or full-85-player exact U1 Shapley claim.

## Citation

Use [CITATION.cff](CITATION.cff) for the author and project metadata. No unassigned DOI is supplied.

## License

Software, source code, scripts, configurations, and repository documentation are licensed under the MIT License. Original DataValueBench benchmark data and derived result collections are licensed under CC BY 4.0 to the extent that the authors hold the applicable rights. Third-party datasets, metadata, model weights, and other upstream materials remain subject to their original licenses and terms and are not relicensed by DataValueBench.

See the [MIT software license](LICENSE), [benchmark data and results license](LICENSE-DATA.md), and [third-party notices](THIRD_PARTY_NOTICES.md).
