"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
import math
from api import Verdict


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def check_data_batch(payload, ctx):
    """
    Pillar: checks
    Faults: volume_spike, volume_drop, null_spike, freshness_lag, distribution_shift
    """
    batch_id = payload.get("batch_id")
    prof = ctx.tools.batch_profile(batch_id)
    if not isinstance(prof, dict) or "error" in prof:
        return Verdict(alert=False, pillar="checks")

    row_count = prof.get("row_count", 0)
    null_rate = prof.get("null_rate", {}).get("customer_id", 0.0)
    mean_amount = prof.get("mean_amount", 0.0)
    staleness_min = prof.get("staleness_min", 0.0)

    base = ctx.baseline

    # 1. Single-dimensional threshold checks
    if row_count < base.get("row_count_min", 435.4732):
        return Verdict(alert=True, pillar="checks", reason=f"volume_drop: row_count={row_count}")
    if row_count > base.get("row_count_max", 561.2948):
        return Verdict(alert=True, pillar="checks", reason=f"volume_spike: row_count={row_count}")
    if null_rate > base.get("null_rate_max", 0.0109):
        return Verdict(alert=True, pillar="checks", reason=f"null_spike: null_rate={null_rate}")
    if mean_amount < base.get("mean_amount_min", 72.7645) or mean_amount > base.get("mean_amount_max", 90.6053):
        return Verdict(alert=True, pillar="checks", reason=f"distribution_shift: mean_amount={mean_amount}")
    if staleness_min > 7.0:
        return Verdict(alert=True, pillar="checks", reason=f"freshness_lag: staleness_min={staleness_min}")

    # 2. Multi-signal Composite Distance (Euclidean z-score norm)
    mu_r, sigma_r = 498.384, 20.970
    mu_m, sigma_m = 81.6849, 2.9735
    null_max = base.get("null_rate_max", 0.0109)
    stale_max = base.get("staleness_min_max", 8.418)

    norm2 = math.sqrt(
        ((row_count - mu_r) / sigma_r) ** 2
        + ((mean_amount - mu_m) / sigma_m) ** 2
        + (null_rate / null_max) ** 2
        + (staleness_min / stale_max) ** 2
    )

    if norm2 > 3.15:
        return Verdict(alert=True, pillar="checks", reason=f"multi_signal_anomaly: norm2={norm2:.2f}")

    return Verdict(alert=False, pillar="checks")


def check_contract_checkpoint(payload, ctx):
    """
    Pillar: contracts
    Faults: type_violation, schema_break, sla_violation
    """
    contract_id = payload.get("contract_id")
    batch_id = payload.get("checkpoint_batch_id")
    diff = ctx.tools.contract_diff(contract_id, batch_id)
    if not isinstance(diff, dict) or "error" in diff:
        return Verdict(alert=False, pillar="contracts")

    violations = diff.get("violations", [])
    if violations:
        return Verdict(alert=True, pillar="contracts", reason=f"contract_violations: {violations}")

    freshness_delay = diff.get("freshness_delay_min", 0.0)
    sla_freshness = payload.get("declared_sla", {}).get("freshness_min", 15)
    baseline_delay_max = ctx.baseline.get("freshness_delay_max_min", 11.1141)

    if freshness_delay > sla_freshness or freshness_delay > baseline_delay_max:
        return Verdict(alert=True, pillar="contracts", reason=f"sla_violation: freshness_delay={freshness_delay}")

    return Verdict(alert=False, pillar="contracts")


def check_lineage_run(payload, ctx):
    """
    Pillar: lineage
    Faults: orphan_output, missing_upstream, runtime_anomaly
    """
    run_id = payload.get("run_id")
    slice_data = ctx.tools.lineage_graph_slice(run_id)
    if not isinstance(slice_data, dict) or "error" in slice_data:
        return Verdict(alert=False, pillar="lineage")

    downstream_count = slice_data.get("actual_downstream_count", 0)
    if downstream_count == 0:
        return Verdict(alert=True, pillar="lineage", reason="orphan_output: actual_downstream_count=0")

    upstream = set(slice_data.get("actual_upstream", []))
    if "raw.orders" not in upstream or "raw.customers" not in upstream:
        return Verdict(alert=True, pillar="lineage", reason=f"missing_upstream: {slice_data.get('actual_upstream')}")

    duration_ms = slice_data.get("duration_ms", 0.0)
    if duration_ms > ctx.baseline.get("lineage_duration_ms_max", 5134.9804):
        return Verdict(alert=True, pillar="lineage", reason=f"runtime_anomaly: duration_ms={duration_ms}")

    return Verdict(alert=False, pillar="lineage")


def check_feature_materialization(payload, ctx):
    """
    Pillar: ai_infra
    Faults: feature_skew (Training-Serving feature drift)
    """
    feature_view = payload.get("feature_view")
    batch_id = payload.get("batch_id")
    drift = ctx.tools.feature_drift(feature_view, batch_id)
    if not isinstance(drift, dict) or "error" in drift:
        return Verdict(alert=False, pillar="ai_infra")

    mean_shift_sigma = drift.get("mean_shift_sigma")
    if mean_shift_sigma is None:
        serve_mean = drift.get("serve_mean", 0.0)
        train_mean = drift.get("train_mean", 0.0)
        train_std = drift.get("train_std", 1.0)
        mean_shift_sigma = abs(serve_mean - train_mean) / train_std

    if mean_shift_sigma > 0.65:
        return Verdict(alert=True, pillar="ai_infra", reason=f"feature_skew: mean_shift_sigma={mean_shift_sigma}")

    return Verdict(alert=False, pillar="ai_infra")


def check_embedding_batch(payload, ctx):
    """
    Pillar: ai_infra
    Faults: embedding_drift, corpus_staleness
    """
    corpus = payload.get("corpus")
    batch_id = payload.get("chunk_batch_id")
    drift = ctx.tools.embedding_drift(corpus, batch_id)
    if not isinstance(drift, dict) or "error" in drift:
        return Verdict(alert=False, pillar="ai_infra")

    centroid_shift = drift.get("centroid_shift", 0.0)
    avg_doc_age_days = drift.get("avg_doc_age_days", 0.0)

    if centroid_shift > 0.039:
        return Verdict(alert=True, pillar="ai_infra", reason=f"embedding_drift: centroid_shift={centroid_shift}")

    if avg_doc_age_days > 41.5:
        return Verdict(alert=True, pillar="ai_infra", reason=f"corpus_staleness: avg_doc_age_days={avg_doc_age_days}")

    return Verdict(alert=False, pillar="ai_infra")
