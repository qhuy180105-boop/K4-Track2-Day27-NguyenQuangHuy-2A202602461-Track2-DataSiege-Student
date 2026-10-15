# Reflection (≤1 page)

### Which fault types were hardest to catch, and why?

1. **Subtle Multi-Metric Distribution Shifts (`data_batch`):**
   - Mild parameter drifts where individual column metrics (e.g., mean order total, row count, or item count) only deviate slightly (Z-score between $1.5\sigma$ and $2.5\sigma$). In isolation, none of these metrics violate standard 3-sigma univariate quality gates.
   - To catch them without inducing false alarms, we had to construct a multi-dimensional composite Euclidean distance ($L_2$ norm of standard score vectors across metrics: `norm2 > 3.15`), projecting orthogonal subtle shifts into a single discriminative anomaly signal.

2. **In-Distribution Latency & Staleness Anomalies (`lineage_run`, `contract_checkpoint`):**
   - In the unseen Private evaluation stream, subtle latency faults (e.g., run duration around $\approx 4,400\text{ ms}$) and subtle document staleness ($\approx 35\text{ days}$) fall completely within the empirical envelope of the clean stream (where clean durations reach up to $4,805\text{ ms}$ and clean max doc age reaches $37.5\text{ days}$).
   - Because the scoring penalty for false alarms is severe ($-30 \times \text{FPR}$, whereas missed detections only lose $0.5 \times$ credit), setting a threshold aggressive enough to capture these in-distribution outliers creates false positives on clean runs. Discerning such anomalies on a single-event basis without historical lineage context creates a fundamental signal-to-noise barrier.

3. **Subtle Feature & Embedding Drift (`feature_materialization`, `embedding_batch`):**
   - While structural issues (schema mismatches, dimensionality shifts, NaN/null embeddings) are deterministic and free to verify, subtle semantic drift or KS distribution shifts require metered tool invocations (`drift_ks_test`, `centroid_cosine_similarity`). Establishing the precise cutoff ($p < 0.005$ and $\Delta\text{cosine} > 0.12$) was critical to avoid false alarms on natural batch variance.

---

### What would you change about your cost/coverage tradeoff, if you had another pass?

1. **Dynamic Adaptive Budget Governor via `ctx.state`:**
   - In our final strategy, we invoked one metered tool per eligible event (e.g., `contract_catalog_lookup`, `upstream_status_probe`, `drift_ks_test`, `centroid_cosine_similarity`). In the Public phase, this resulted in 240 credits spent against a 220 budget ($9.09\%$ overage, costing $1.82$ score penalty, but capturing $100\%$ TPR for $+50$ score). In Private, the 300-credit budget easily covered this with $0.0\%$ overage.
   - If granted another iteration, we would implement a two-stage triage gate:
     - Stage 1 (Free): Compute instant statistical deviations using `ctx.baseline`. If an event is unambiguously clean ($Z < 1.2$) or unambiguously corrupt (missing fields, dimension mismatch), decide immediately without calling tools.
     - Stage 2 (Metered): Only spend budget on events lying in the ambiguous gray zone ($1.5 \le Z \le 3.0$), keeping budget overage strictly at zero across all phase lengths.

2. **Online Welford Streaming Baselines:**
   - Instead of treating `ctx.baseline` as a static snapshot, we would use `ctx.state` to update online running statistics (Welford's algorithm for rolling mean and variance) across verified clean batches. This would allow the detector to adapt to non-stationary baseline drift without raising false alarms.
