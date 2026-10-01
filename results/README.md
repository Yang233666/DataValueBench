# Released frozen results

`rq1/` contains metric estimates and bootstrap intervals for the four reported regimes. `rq2/` contains all 5,700 per-intent method rows, aggregates, pairwise statistics and query-realization coverage. `rq3/openml/` and `rq3/huggingface/` contain separate primary endpoints, common-support membership, applicability summaries, MMR sensitivity and paired/Holm statistics. `rq4/` contains fixed-game player scores, ranks, signs and same-rule comparisons.

Inputs and identity dictionaries are in `data/`. Numerical values are preserved; reader-facing RQ3 method labels use BGE-M3. There is no cross-setting pooled score. Undefined and not-applicable values remain explicit. The reproduction scripts validate these outputs and replay supported downstream calculations; they do not fit new models or generate scientific results.

Original DataValueBench derived result collections under `results/` are covered by [Creative Commons Attribution 4.0 International (CC BY 4.0)](../LICENSE-DATA.md), only to the extent that the authors hold the applicable rights. Third-party or upstream elements remain subject to their original licenses and terms and are not relicensed by DataValueBench.
