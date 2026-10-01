# Models

No model weights are distributed. [model_manifest.json](model_manifest.json) records exact checkpoint revisions, roles, access requirements and links to the upstream license at that revision. Normal bundled-data/result verification uses no model inference, downloads, GPU or hosted service.

| Model | Role | Upstream license recorded at the pinned revision |
|---|---|---|
| all-MiniLM-L6-v2 | RQ1 workflow text features | Apache-2.0 |
| Multilingual E5 base | RQ2 retrieval | MIT |
| BGE-M3 | RQ2 retrieval; RQ3 candidate construction and ranking baseline | MIT |
| GTE multilingual base | RQ2 retrieval; RQ3 SemanticILD evaluation | Apache-2.0 |
| SPLADE cocondenser ensembledistil | RQ2 sparse retrieval | CC-BY-NC-SA-4.0 |
| ColBERTv2 | RQ2 late-interaction retrieval | MIT |
| Qwen3-14B | RQ2 query generation | Apache-2.0 |
| GPT-6 Astra (recorded provider name) | RQ2 full-workload query-fidelity validation | Hosted-service terms |

GTE's external code revision is pinned separately as `Alibaba-NLP/new-impl`. BGE-M3 candidate/ranking geometry and GTE evaluation geometry are deliberately distinct. The BGE-M3 baseline is dense inner-product ranking using its pinned representation.

The Astra record binds the provider name and reasoning effort, not an immutable weights revision. Availability of that historical service is not guaranteed. Released final queries remove the need to repeat query generation or validation. Validation establishes query-to-intent fidelity; it is not human dataset relevance.

RQ1 estimator families and RQ3 graph embedding methods are learned from task-local inputs. Their training configurations, seeds and pinned implementations are included; fitted estimators and embedding caches are excluded. AME fits a LassoCV response model using actual utility-oracle responses; it does not define a new learned utility.
