# /canonical_evaluation.md
# BenchPress Reliability Audit: Canonical Evaluation Protocol (v1.0)

**Purpose:** Provide one standardized, leakage-resistant evaluation that all agents can be scored on, independent of their self-chosen evaluation protocol.

This protocol defines:
1) how to construct ground truth targets,
2) a deterministic holdout mask,
3) how agents must produce canonical predictions,
4) how MAE is computed.

---

## 1. Ground truth matrix construction

From `llm_benchmark_data.json`:

- Extract models, benchmarks, and score entries.
- If multiple scores exist for the same `(model_id, benchmark_id)`, resolve duplicates by **simple average**.
- All other missing entries remain missing (not scored).

Define the set of observed cells:

$$\Omega = \{(m,b): y_{m,b}\ \text{is observed}\}$$

---

## 2. Canonical normalization (0–100 per benchmark)

To compare across mixed metrics (percentages + rating scales), evaluation is done on a per-benchmark normalized scale:

For each benchmark b:
- compute `min_b = min_{(m,b)∈Ω} y_{m,b}`
- compute `max_b = max_{(m,b)∈Ω} y_{m,b}`
- define `range_b = max(max_b - min_b, 1e-9)`

Normalize:

$$\tilde{y}_{m,b} = 100 \cdot \frac{y_{m,b} - \min_b}{\text{range}_b}$$

Apply the same transform to predictions $\hat{y}$.

(Do not clip by default; report if large out-of-range predictions occur.)

---

## 3. Canonical holdout mask

### 3.1 Deterministic mask generation
Let `CANONICAL_SEED = 20260226`.

We construct a held-out test set `Ω_test ⊂ Ω` by **benchmark-stratified sampling**:

For each benchmark b:
- let `Ω_b = {(m,b) ∈ Ω}`
- if `|Ω_b| < 5`, then set `Ω_b_test = ∅` (too sparse to hold out)
- else sample `n_b_test = max(1, round(0.2 * |Ω_b|))` pairs uniformly at random from `Ω_b` using the canonical RNG seeded by `(CANONICAL_SEED, b)`
- set `Ω_test = ⋃_b Ω_b_test`

Training set:

$$\Omega_{\text{train}} = \Omega \setminus \Omega_{\text{test}}$$

### 3.2 File format
The harness provides `canonical_mask.json` with:
```json
{
  "seed": 20260226,
  "holdout_fraction": 0.2,
  "min_cells_per_benchmark_to_holdout": 5,
  "pairs": [
    {"model_id": "...", "benchmark_id": "..."},
    ...
  ]
}
````

---

## 4. What analysis agents must do

Agents must treat all `(m,b) ∈ Ω_test` as **missing during fitting** (no leakage), fit their predictor on `Ω_train`, then output predictions for every held-out pair.

### Required output: `canonical_predictions.csv`

Must contain one row per held-out pair with columns:

* `model_id`
* `model_name`
* `benchmark_id`
* `benchmark_name`
* `y_pred`  (prediction in *raw* units; normalization happens in scoring)

Coverage requirement (for SUCCESS): predictions for ≥95% of held-out pairs.

---

## 5. Scoring

For each held-out pair $(m,b) \in \Omega_{\text{test}}$ where a prediction is present:

Compute normalized absolute error:

$$e_{m,b} = |\tilde{y}_{m,b} - \tilde{\hat{y}}_{m,b}|$$

### Overall MAE

$$\mathrm{MAE}_{\text{canonical}} = \frac{1}{|\Omega_{\text{scored}}|}\sum_{(m,b)\in \Omega_{\text{scored}}} e_{m,b}$$

### Per-benchmark MAE

For each benchmark $b$ with at least 1 scored pair:

$$\mathrm{MAE}_b = \frac{1}{|\Omega_{b,\text{scored}}|}\sum_{(m,b)\in \Omega_{b,\text{scored}}} e_{m,b}$$

### Coverage

$$\mathrm{coverage} = \frac{|\Omega_{\text{scored}}|}{|\Omega_{\text{test}}|}$$

Report out-of-range predictions (where `y_pred` is far outside observed min/max) as diagnostics, but do not automatically clip unless explicitly stated in the final report.

---

## 6. Rationale

* Benchmark-stratified holdout prevents the test set from collapsing onto only high-coverage benchmarks.
* Benchmark-wise min-max normalization yields a common 0–100 scale even when raw metrics differ (e.g., ratings vs % correct).
* Leakage resistance comes from masking held-out entries before fitting.
