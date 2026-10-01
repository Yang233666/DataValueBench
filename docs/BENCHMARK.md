# Benchmark definitions

The four settings evaluate different objects and must not be collapsed into one score. RQ1 predicts historical workflow performance under defined generalization regimes. RQ2 ranks a fixed metadata corpus for paired structured/natural-language intents using automatic metadata-derived relevance. RQ3 ranks alternatives conditional on a source dataset, separately in two ecosystems. RQ4 assesses contributions within two explicitly fixed games.

The scope configurations retain stable result identities for provenance: `RQ1-FR-001`, `RQ2-FR-AUTO-001`, `RQ3-FR-OML-001` and `RQ3-FR-HF-001`. These identify the reported results, not software versions.

All methods within a comparison share the stated evaluation support. Undefined correlations, unrealized queries and mathematical structural inapplicability are different concepts. None is silently zero-imputed. Inputs and results are frozen; new provider responses are not replacements for a historical snapshot.

Implementation constants, scientific seed namespaces, utility condition IDs and exact model names remain unchanged even when they contain version tokens. Reader-facing condition labels follow the manuscript. The primary RQ3 condition is displayed as `C100`; the machine-readable stable condition identifier `C100_v2` is retained as an implementation identity, not the public condition name. `C20_hist` and `C20_v2` are retained because they are explicit sensitivity-condition labels in the reported analysis. ColBERTv2 and all-MiniLM-L6-v2 are genuine model identities. SHA-based namespaces must retain every byte to preserve IDs and RNG streams.

See the four setting guides and machine-readable configurations for details. No deferred human study is represented as completed evidence.
