# Model and implementation identities

Use [the model inventory](../models/README.md) and [machine-readable manifest](../models/model_manifest.json) for checkpoint revisions, roles and terms. [RQ1 configurations](../configs/rq1/estimators.json), [RQ3 methods](../configs/rq3/methods.json) and [RQ4 methods](../configs/rq4/methods_and_utilities.json) bind learned benchmark implementations and seeds. A checkpoint revision and an inference implementation revision are distinct; GTE records both.

Verification is CPU-only and does not invoke these models. Qwen generation used BF16 GPU inference; dense/sparse/token-vector construction requires the original method-specific environments and potentially substantial CPU/GPU resources. Astra validation requires hosted access if repeated. No full-run runtime or minimum hardware claim is inferred from the small verification path.
