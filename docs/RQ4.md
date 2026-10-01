# RQ4 · Fixed-game contribution assessment

The 85 players are historical OpenML task/dataset pairs, ordered by `player_uid`. `players.parquet` contains resolved task/dataset IDs, observation counts and the exact result-blind five-fold assignment (17 players per fold). `u0_player_utilities.parquet` holds the fixed per-player historical mean accuracies.

U0 assigns zero to the empty coalition and the mean of fixed per-player accuracies to a nonempty coalition. U1 is five-fold cross-fitted flow-conditioned prediction: training-coalition flow means, global training mean fallback for unseen flows, and held-out-player macro(1−MAE). A fold with no coalition training observations contributes zero. U1 uses exact historical observation values, not RQ1 predictions or a learned utility model.

Six rules are compared across the same two games: StandaloneUtility, LeaveOneOut, MonteCarloShapley, BetaShapley, DataBanzhaf and AME. Stored result identifiers retain their original exact rule names (including `Banzhaf` where used). BetaShapley uses alpha=4, beta=1. Seeds, budgets and AME's four inclusion probabilities/LassoCV settings are in `configs/rq4/methods_and_utilities.json`. Scores and sign information are preserved, including negative values.

`bash scripts/reproduce_rq4.sh` calls the existing `agreement_metrics` implementation on the released aligned player values. It verifies Spearman, Kendall tau-b, rank changes, exact rank matches, sign changes and Top-5/Top-10/bottom overlaps against saved comparisons. These are deterministic descriptive summaries of fixed games, not a new experiment or inferential test.

No full-85-player exact U1 Shapley claim is made. Exact U0 and local exact checks in the implementation have their stated scopes only. This analysis provides no repeated-game, population-level, causal or economic-value inference. Full utility/contribution reruns need the exact external historical observation values; bundled comparison verification does not.
