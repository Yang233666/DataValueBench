# Data, schemas and access

## Release categories

| Category | Included or external | Basis and purpose |
|---|---|---|
| A · Author-created | Included: latent intents, paired queries, workload/split assignments, player/fold definitions | Necessary benchmark construction inputs; no personal annotations |
| B · Derived benchmark data | Included: qrels, normalized identity maps, the exact intrinsic features used in the reported results, fixed MiniLM vectors, candidate memberships, fixed U0 inputs, frozen result/support statistics | Author-created transformations or outputs; no underlying raw dataset rows or model weights |
| C · Redistributable upstream content | No blanket raw upstream corpus release | Public provider availability is not assumed to grant redistribution rights |
| D · Upstream content with unresolved redistribution basis | External: accuracy observation values, workflow text, full metadata snapshots | Exact identity, source, acquisition limitation and dependent stage in `data/external_inputs.json` |
| E · Large caches/indexes/intermediates | External: retrieval indexes, GTE/BGE/graph caches, full top-1,000 channel lists and fused unions | Unnecessary for normal verification; exact recorded index hashes/settings retained |

The 106,907-row observation identity table is included, but its upstream accuracy/timestamp/function values are not. This preserves row multiplicity and resolved IDs while separating redistribution of historical measurements. The full observation snapshot is about 9.7 MB; its exclusion is a terms decision, not a size decision. There is currently no guaranteed public URL for the exact snapshot. OpenML task/flow/run IDs enable source lookup, but a refreshed response is not a substitute for a recorded hash.

## Input schemas and joins

- **RQ1 observation identities:** one row per `observation_uid`; `source_row_index`, task/dataset/flow/run IDs and exact task/flow UIDs. Table order defines zero-based `observation_position`.
- **RQ1 splits:** `outer_splits.parquet` has regime/condition/fold/role and observation position; `inner_splits.parquet` additionally has outer and inner fold IDs. Joining positions to the identity table recovers every original ordered member. Only G0, GT80, GD and GF are selected. No split is regenerated.
- **RQ1 features:** `dataset_features.parquet` contains the ten exact derived intrinsic features used in the reported RQ1 analysis. `workflow_vectors.npy` is the recorded float32 MiniLM matrix; `workflow_vector_ids.parquet` binds row order and truncation details. These small derived inputs are bundled; raw workflow descriptions remain external.
- **RQ2 workload:** `latent_intents.jsonl` has 500 unique intent IDs, exact predicate identities, support/stratum and sampling metadata. `track_a_queries.jsonl` binds structured conditions. `track_b_queries.jsonl` has 475 unique realized queries, query hashes, construction attempts and validation categories, with private artifact paths removed.
- **RQ2 N/A and qrels:** `unrealized_track_b_slots.jsonl` holds 25 explicitly unrealized slots with reasons. `automatic_qrels.parquet` has 158,126 positive intent/dataset pairs (`relevance=1`) across the 500 intents. This is automatic metadata-derived relevance. Workload memberships and exact common-support IDs are separate inputs. Unrealized queries never receive synthetic zero scores.
- **RQ3 sources:** separate tables for 6,408 OpenML and 100,000 HF sources, with zero-based `source_position`. OpenML `source_id` is the exact integer dataset ID. HF `source_id` is SHA256 of the original UTF-8 canonical dataset ID. `upstream_dataset_id` provides a lookup name except one credential-shaped upstream name, whose lookup cell is null. No source is dropped. Hashes are not re-sorted: original source order is preserved.
- **RQ3 candidates:** one row per source, with 200 ordered `candidate_positions` referring to that ecosystem's source dictionary. Smaller nested pools are prefixes. HF is partitioned into four contiguous 25,000-source files. All 21,281,600 identities and their ordering are retained; top-channel scores and large fused-union intermediates are not needed to inspect candidate support.
- **RQ4 players:** 85 player UIDs with resolved task/dataset IDs, exact order, observation counts and fold assignments. `u0_player_utilities.parquet` contains the fixed per-player mean-accuracy inputs. U1 needs the separate exact historical observation snapshot, listed externally.

## Result schemas

RQ1 CSVs key points by condition/method/metric; JSON intervals preserve undefined values and draw counts. RQ2 per-intent CSVs key results by method/intent; aggregate, bootstrap, pairwise/Holm, stratified and workload statistics remain inspectable JSON/CSV. RQ4 tables key fixed-game scores/ranks/signs by utility/rule/player; comparisons retain the same-rule endpoints.

RQ3 support storage is normalized without dropping evidence. `support_statistics.json` retains condition statistics. `common_support.parquet` contains `condition_index`, `membership_position` and `source_position`. Join through the source table and sort by membership position to reconstruct the exact original support list. The public `load_support` loader performs this operation. Numeric values, nulls and ordering were round-trip checked; the RQ3 display label BGE-M3 is used throughout.

## External acquisition

For each absent numerical input, `external_inputs.json` records SHA256, size, provider source and dependent step. For archived retrieval indexes, `configs/rq2/index_files.json` records component hashes, and index configurations record dtype, lengths, pooling and normalization. Model/provider identities are separate in `models/model_manifest.json`.

Exact historical archives must be obtained from the benchmark authors where redistribution terms permit. Where no public archive route exists, full reconstruction remains conditional. Do not query a live API and claim the result is the historical corpus. Original providers are [OpenML](https://www.openml.org/) and [Hugging Face datasets](https://huggingface.co/datasets). Respect [OpenML terms](https://docs.openml.org/intro/terms/) and [Hugging Face terms](https://huggingface.co/terms-of-service); individual resource licenses still apply. No access credentials belong in the repository.

The complete eligible RQ2 census (`eligible_intents.parquet`, 278,035 intents) is also included for construction inspection. Its `selected` flag denotes the initial 500 before prespecified replacements; the current workload is exclusively `latent_intents.jsonl`. This distinction does not change current evaluation support. Pure duplicate-resolution, replacement and lexical-diagnostic functions are in `rq2_realization.py`.

External RQ3 graph/cache identities are in `configs/rq3/graph_inputs.json`; they are required only for upstream graph-method replay, not bundled support verification.
