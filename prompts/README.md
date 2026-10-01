# Query construction prompts

`rq2/query_generation.txt` is the exact system prompt. `datavaluebench.rq2_query_generation.messages` constructs the exact user prompt from field/value constraints; `generation_seed` preserves the original SHA-derived seed namespace. Sampling parameters and model revision are in `configs/rq2/query_generation.json` and `models/model_manifest.json`.

`rq2/semantic_validation.txt` and `rq2/judgment_schema.json` are the exact semantic-fidelity prompt and machine schema. No annotator records or model conversation logs are included. The bundled 475 realized queries are the evaluation input, so reproducing released aggregates does not call either model.
