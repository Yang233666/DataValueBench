# Reproducibility modes

DataValueBench provides three reproducibility paths. The released artifact is designed to verify reported benchmark results and reconstruct supported analyses from frozen inputs, while full upstream regeneration depends on external snapshots and provider-specific resources.

| Mode | Supported work | Additional prerequisites |
|---|---|---|
| Bundled-data/result verification | Hashes, input schemas, identities, split membership, candidate ordering, result support | Core CPU environment |
| Reproduction from released frozen inputs | RQ2 aggregate scores from per-intent outputs; RQ4 same-rule comparisons from player values; RQ3 support reconstruction | Same core environment; no models |
| Full upstream reconstruction | Estimator refitting, retrieval/index construction, graph/ranking execution, utility/contribution estimation | Exact external snapshots, provider terms/access, model revisions, method-specific environments |

Run the shell entry points from any working directory; they resolve their repository root. `PYTHON_BIN` selects an existing interpreter. `setup.sh --check` checks versions without installation. Normal setup creates a fresh local `.venv` and installs the package in editable mode. It refuses to overwrite an existing environment. Setup downloads dependencies only; verification downloads nothing. Results are printed as JSON or written with `--output` to a new directory. Failure exits nonzero.

Core verification is tested on Linux with Python 3.10.12, with pins in `requirements/core.txt`. Core reference tests exclude two graph modules with additional pins. Graph reference tests use a separate Python 3.11 environment (NetworkX 3.6.1 requires a compatible installer environment):

```bash
(
  python3.11 -m venv .venv-graph
  .venv-graph/bin/python -m pip install -r requirements/graph.txt
  PYTHON_BIN="$PWD/.venv-graph/bin/python" bash scripts/run_tests.sh graph
)
```

Historical execution used archived pinned graph dependencies; no version guard is relaxed. Clean installation on every operating system is not guaranteed. The supported quick path is Linux. Torch/model packages are unnecessary for that path.

`data/external_inputs.json` records external input dependencies, exact hashes, provider sources, and affected stages. For an available original snapshot, verify SHA256 before supplying it to a reusable implementation. A live provider listing is not a deterministic way to recreate a historical snapshot. Full upstream regeneration follows the external input and environment requirements documented in `data/external_inputs.json`; the release does not automatically acquire restricted upstream snapshots. Do not replace exact retrieval with approximate indexes or mathematical N/A with a fallback method.

Reusable APIs live under `datavaluebench`: `rq1_estimators`, `rq1_tree_estimators`, `rq1_imc`, `rq2_query_construction`, `rq2_query_generation`, `rq2_exact_retrieval`, `rq3_ranking`, `rq3_hf_candidates`, `rq3_applicability`, `reference`, `rq4_sampling_estimators`, `rq4_ame`, and `rq4_statistics`. They preserve the original scientific implementation; full orchestration must respect the per-setting information boundaries and exact configurations. Reference tests exercise bounded synthetic examples, not benchmark training.
