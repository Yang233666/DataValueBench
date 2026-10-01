"""Reusable DataValueBench benchmark implementation."""
from __future__ import annotations

import hashlib
import json
import numpy as np

REPLICATES = {"AUTO": 2000, "HUMAN": 10000}
MASTER_SEED = 20260903
NAMESPACE = "DVBench-v2-RQ3-BOOTSTRAP-v1.0"


def bootstrap_indices(ecosystem, scope, replicate, source_count):
    if ecosystem not in ("HF", "OML") or scope not in REPLICATES:
        raise ValueError("invalid frozen ecosystem/scope")
    if type(replicate) is not int or not 0 <= replicate < REPLICATES[scope] or source_count < 1:
        raise ValueError("invalid replicate or source count")
    payload = dict(ecosystem=ecosystem, master_seed=MASTER_SEED, namespace=NAMESPACE,
                   replicate=replicate, scope=scope)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    seed128 = int.from_bytes(hashlib.sha256(encoded).digest()[:16], "big")
    rng = np.random.Generator(np.random.PCG64DXSM(seed128))
    return rng.integers(0, source_count, size=source_count, dtype=np.int64)


def paired_effect(method_values, anchor_values, source_ids, *, ecosystem, scope,
                  confirmatory=False):
    """No independent method resampling; use the same indices for effects/nulls."""
    if ecosystem not in ("HF", "OML") or scope not in REPLICATES:
        raise ValueError("invalid frozen ecosystem/scope")
    ids = list(source_ids)
    if not ids or ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("nonempty unique canonical ascending source support required")
    if ecosystem == "OML" and any(type(x) is not int for x in ids):
        raise ValueError("OpenML source IDs must be canonical integers")
    if ecosystem == "HF" and any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("HF source IDs must be canonical strings")
    method, anchor = np.asarray(method_values, dtype=np.float64), np.asarray(anchor_values, dtype=np.float64)
    if method.shape != (len(ids),) or anchor.shape != method.shape or not (np.isfinite(method).all() and np.isfinite(anchor).all()):
        raise ValueError("complete finite values on the registered common support required")
    delta = method - anchor
    observed = float(delta.mean())
    centered = delta - observed
    means, null_means = [], []
    design_sha = hashlib.sha256()
    for b in range(REPLICATES[scope]):
        indices = bootstrap_indices(ecosystem, scope, b, len(ids))
        design_sha.update(indices.astype("<i8", copy=False).tobytes())
        means.append(float(delta[indices].mean()))
        if confirmatory:
            null_means.append(float(centered[indices].mean()))
    ci = np.percentile(means, [2.5, 97.5], method="linear").tolist()
    result = {"ecosystem": ecosystem, "scope": scope, "comparison_sample_size": len(ids),
              "source_ids_sha256": hashlib.sha256(json.dumps(ids,ensure_ascii=False,separators=(",", ":")).encode()).hexdigest(),
              "method_macro_mean": float(method.mean()), "anchor_macro_mean": float(anchor.mean()),
              "paired_mean_difference": observed, "paired_median_difference": float(np.median(delta)),
              "paired_ci95": ci, "win_source_proportion": float(np.mean(delta > 0)),
              "tie_source_proportion": float(np.mean(delta == 0)),
              "loss_source_proportion": float(np.mean(delta < 0)),
              "bootstrap_design": {"numpy_version": np.__version__, "bit_generator": "PCG64DXSM",
                  "namespace": NAMESPACE, "master_seed": MASTER_SEED, "replicates": REPLICATES[scope],
                  "index_arrays_sha256": design_sha.hexdigest(), "index_byte_encoding": "concatenated little-endian int64 in replicate order",
                  "percentile_method": "linear", "replicate_seed_encoding": "canonical compact UTF-8 JSON; SHA256 first 16 bytes unsigned big endian"}}
    if confirmatory:
        result["raw_bootstrap_p"] = (1 + sum(abs(x) >= abs(observed) for x in null_means))/(REPLICATES[scope]+1)
    return result


def holm_family(pvalues):
    """Adjust exactly the supplied predeclared active comparisons in one family."""
    if any(not isinstance(k,str) or not np.isfinite(v) or not 0 <= v <= 1 for k,v in pvalues.items()):
        raise ValueError("finite active comparison p-values required; no artificial N/A p-values")
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    result, running = {}, 0.
    for i, key in enumerate(ordered):
        running = max(running, min(1., (len(ordered)-i)*pvalues[key]))
        result[key] = {"raw_bootstrap_p": float(pvalues[key]), "holm_adjusted_p": float(running),
                       "holm_reject_alpha_0_05": bool(running <= .05)}
    return result
