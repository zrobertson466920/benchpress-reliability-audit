# BenchPress Reliability Audit: Final Report

**Analysis date:** 2026-02-28
**K = 50 agents** (opus-4.6, runs 01–50)
**Two independent reliability evaluators** (opus-4.6-reliability_run01, _run02)
**All statistics independently computed and verified from raw agent outputs.**

---

## 1. SUCCESS/FAILURE Breakdown (Section 6.1)

| Metric | Value |
|:-------|------:|
| Total agents (K) | 50 |
| SUCCESS (S) | 48 (96.0%) |
| FAILURE | 2 (4.0%) |

**Failures:** Runs 16 and 21 — both missing `results_summary.json`.
All 48 SUCCESS agents achieved 100% canonical prediction coverage (196/196 test pairs).

---

## 2. Canonical MAE Distribution (Section 3)

| Statistic | Value |
|:----------|------:|
| Mean | 15.08 |
| Median | 15.58 |
| Std | 2.30 |
| Min | 5.68 (run07) |
| Max | 18.95 |
| Q25 | 14.49 |
| Q75 | 16.41 |

| MAE Bin | Count | Fraction |
|:--------|------:|---------:|
| < 5 | 0 | 0.0% |
| 5–10 | 2 | 4.2% |
| 10–20 | 46 | 95.8% |
| ≥ 20 | 0 | 0.0% |

The canonical MAE distribution is tightly concentrated in the 10–20 range.
Only two outliers (runs 07 and 35) achieved MAE < 10.

---

## 3. Hypothesis Table

| Hypothesis | Criterion | Observed | Pass/Fail |
|:-----------|:----------|:---------|:---------:|
| H1: Rank convergence | ≥75% effective rank ≤ 3 | 41.7% (20/48) | **FAIL** |
| H2: Subset partial convergence | 0.2 < mean Jaccard < 0.6 | mean J = 0.145 | **FAIL** |
| H3: Prediction feasibility | ≥80% canonical MAE < 10 | 4.2% (2/48) | **FAIL** |
| H4: Preprocessing as primary fork | Top ΔW query = preprocessing | Top = MAE threshold | **FAIL** |
| H5: Qualitative robustness | ≥90% support low-rank claim | 47.9% mention low-rank | **FAIL** |

**All five pre-registered hypotheses fail.**

---

## 4. Hypothesis Details

### H1 — Rank Convergence
Effective rank distribution (n=48): mean=8.4, median=5, min=1, max=20.

| Rank | Count | % |
|:-----|------:|--:|
| 1 | 4 | 8.3% |
| 2 | 8 | 16.7% |
| 3 | 8 | 16.7% |
| 4 | 2 | 4.2% |
| ≥5 | 26 | 54.2% |

Only 41.7% report rank ≤ 3. The wide spread (1–20) reflects genuine methodological
divergence: agents using strict 90% variance thresholds report high ranks (10–20),
while agents using elbow/gap heuristics or practical prediction quality report low ranks
(1–3). The concept "effective rank" is operationally ambiguous without a fixed criterion.

### H2 — Benchmark Subset Convergence
48 agents with benchmark selections, subset sizes 5–8 (mean 6.3).
Mean pairwise Jaccard = 0.145 (below the 0.2 lower bound).

Most commonly selected benchmarks:

| Benchmark | Count | % |
|:----------|------:|--:|
| ARC-AGI-2 | 23 | 48% |
| HMMT Feb 2025 | 22 | 46% |
| SWE-bench Verified | 19 | 40% |
| AIME 2025 | 17 | 35% |
| LiveCodeBench | 15 | 31% |
| MMLU-Pro | 14 | 29% |

There is partial thematic convergence (challenging reasoning/coding benchmarks preferred)
but insufficient overlap at the individual benchmark level to reach mean Jaccard > 0.2.

### H3 — Prediction Feasibility
Only 2/48 agents (4.2%) achieve canonical MAE < 10 on the normalized 0–100 scale.
The task is inherently difficult: predicting held-out benchmarks from only k=5 revealed
scores per model, with 54–66% overall data missingness. The prediction is far from the
pre-registered feasibility threshold.

### H4 — Preprocessing as Primary Fork
Both evaluators independently identify canonical MAE < 10 as the top fork:

| Evaluator | Top fork | ΔW | Preprocessing rank |
|:----------|:---------|---:|:-------------------|
| 01 | T1_mae_lt10 | 0.00973 | #3 (T2_filtered_bench, ΔW=0.00576) |
| 02 | T1_Q2 (MAE<10) | 0.00347 | #2 (T2_Q1 aggressive_filtering, ΔW=0.00156) |

The top fork is tautological — the 2 agents with low MAE are maximally distinctive
because they are extreme outliers on the primary performance metric, not because they
represent a distinct analytical school. Preprocessing/filtering ranks #2–3 in both
evaluators but with much smaller ΔW.

### H5 — Qualitative Robustness
No tier-3 query achieves the ≥90% threshold needed to test the claim "the matrix is
strongly low-rank and benchmark performance is predictably structured."

| Query (Eval 01) | YES | NO | % YES |
|:----------------|----:|---:|------:|
| T3_mentions_lowrank | 23 | 25 | 47.9% |
| T3_mentions_dominant | 26 | 22 | 54.2% |
| T3_mentions_elo | 29 | 19 | 60.4% |
| T3_mentions_sparse | 25 | 23 | 52.1% |
| T3_mentions_scale | 17 | 31 | 35.4% |

The "low-rank" claim is genuinely contested — agents that use 90% variance thresholds
correctly note the matrix requires 10–20 components, while those using prediction-oriented
criteria note that rank 2–3 suffices for reasonable predictions.

Per the analysis plan: H5 is reported as **"not testable under prereg constraints"**
since no deterministic query achieves adequate variance for the ≥90% threshold.

---

## 5. Mutual Evaluation & TVD-MI (Section 5)

### 5.1 Welfare (independently replicated from raw response matrices)

| Metric | Evaluator 01 | Evaluator 02 |
|:-------|:-------------|:-------------|
| Overall W | 0.120319 | 0.111042 |
| Welfare std | 0.020738 | 0.022672 |
| Welfare min | 0.064255 | 0.065957 |
| Welfare max | 0.157340 | 0.160638 |
| Replication match | r=1.000, max diff<1e-6 | r=1.000, max diff<1e-6 |

### 5.2 Permutation Test (independently replicated, n=5000)

| Metric | Evaluator 01 | Evaluator 02 |
|:-------|:-------------|:-------------|
| W_obs | 0.120319 | 0.111042 |
| Null mean | 0.117629 | 0.103997 |
| Null std | 0.001740 | 0.002168 |
| p-value | 0.069 | **0.002** |
| z-score | 1.55 | **3.25** |

Evaluator 01's result is marginal (p=0.069), consistent with their own reported p=0.071.
Evaluator 02 shows a substantially stronger signal (p=0.002, z=3.25), indicating that
its query set captures more structured cross-agent variation. The difference likely
reflects query design quality: evaluator 02's queries may better separate genuine
methodological forks from noise.

### 5.3 Fork Contributions (ΔW)

**Evaluator 01 top 5:**

| Rank | Query | Tier | ΔW |
|:-----|:------|:----:|---:|
| 1 | T1_mae_lt10 (canonical MAE < 10) | 1 | 0.00973 |
| 2 | T2_canonical_mae_10_20 (MAE in [10,20)) | 2 | 0.00709 |
| 3 | T2_filtered_bench (filtered preprocessing) | 2 | 0.00576 |
| 4 | T1_missing_gt06 (missing fraction > 0.6) | 1 | 0.00388 |
| 5 | T2_minmax_norm (min-max normalization) | 2 | 0.00368 |

**Evaluator 02 top 5:**

| Rank | Query | Tier | ΔW |
|:-----|:------|:----:|---:|
| 1 | T1_Q2 (canonical MAE < 10) | 1 | 0.00347 |
| 2 | T2_Q1 (aggressive filtering) | 2 | 0.00156 |
| 3 | T1_Q5 (missing fraction 0.4–0.6) | 1 | 0.00114 |
| 4 | T4_Q2 (selected includes coding) | 4 | −0.00059 |
| 5 | T4_Q3 (selected includes agentic) | 4 | −0.00091 |

---

## 6. Secondary Analyses

### 6.2 Fork-Conditioned Outcomes

**Top fork (MAE < 10): YES = 2, NO = 46**
- MAE: YES mean = 7.54, NO mean = 15.40, Cohen's d = −4.56 (trivially large — tautological)
- Effective rank: YES mean = 10.5, NO mean = 8.35, Cohen's d = 0.32 (small)

**Preprocessing fork (filtered benchmarks): YES = 45, NO = 3**
- MAE: YES mean = 15.05, NO mean = 15.50, Cohen's d = −0.19 (negligible)
- Nearly all agents filter benchmarks; the 3 non-filterers show no meaningful difference.

### 6.3 Cross-Evaluator Robustness

| Metric | Value |
|:-------|------:|
| Welfare vector correlation | r = 0.60 |
| Common agents | 48 |

Both evaluators rank canonical MAE threshold as the #1 fork and preprocessing/filtering
in the top 3. The welfare correlation of 0.60 indicates moderate agreement on which
agents are distinctive, despite fully independent query design. The difference in
permutation p-values (0.069 vs 0.002) suggests evaluator 02's queries are more
informative, possibly due to better balance of YES/NO rates or more discriminating
tier-2 methodology queries.

---

## 7. Interpretation

### Key Findings
1. **High success rate** (96%) but **uniformly mediocre prediction quality** (median MAE ~15.6
   on 0–100 scale). The task is harder than anticipated.
2. **No rank convergence** — "effective rank" is operationally ambiguous. The bimodal
   distribution (low-rank vs. high-rank reports) maps directly onto whether agents
   optimize for variance explained (high rank) or prediction utility (low rank).
3. **Low subset overlap** (Jaccard 0.145) — agents select diverse benchmarks even when
   thematic preferences (reasoning, coding) partially align.
4. **The primary fork is performance itself**, not a methodological choice. The 2 agents
   with MAE < 10 are outliers, not a distinct school.
5. **"Low-rank" is genuinely contested** — it depends on the criterion (90% variance → rank
   16; prediction quality → rank 2–3), and both perspectives are defensible.
6. **Mutual evaluation shows weak but real structure** — evaluator 02's permutation test
   (p=0.002) confirms agents share more information than chance, while evaluator 01's
   marginal result (p=0.069) suggests the signal is sensitive to query design.

### Limitations
- **Finite Q noise**: 20 binary queries provide limited resolution for fork detection.
- **Single model family**: All agents are opus-4.6; inductive bias is shared.
- **Tautological top fork**: The MAE < 10 query dominates ΔW because it isolates 2
  extreme outliers, not a meaningful analytical fork.
- **H5 untestable**: No deterministic query achieves the ≥90% threshold needed;
  the claim is genuinely split ~50/50 among agents.
- **Cross-evaluator permutation divergence**: p=0.069 vs p=0.002 highlights sensitivity
  of the mutual evaluation framework to query design choices.
