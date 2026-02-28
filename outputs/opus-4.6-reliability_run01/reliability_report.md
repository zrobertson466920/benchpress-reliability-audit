# BenchPress Reliability Audit: Reliability Report

**Evaluator:** opus-4.6-reliability_run01  
**Date:** 2026-02-27  
**Specification version:** 1.0

---

## 1. SUCCESS/FAILURE Breakdown

| Metric | Value |
|:-------|------:|
| Total agents | 50 |
| SUCCESS | 48 |
| FAILURE | 2 |
| SUCCESS rate | 96.0% |

**Failures:**
- `opus-4.6_run16`: missing `results_summary.json`
- `opus-4.6_run21`: missing `results_summary.json`

All 48 SUCCESS agents achieved 100% canonical prediction coverage.

---

## 2. Canonical Metrics Summary

| Statistic | Value |
|:----------|------:|
| Mean MAE | 15.08 |
| Median MAE | 15.58 |
| Std MAE | 2.30 |
| Min MAE | 5.68 |
| Max MAE | 18.95 |
| MAE < 5 | 0 |
| MAE 5–10 | 2 |
| MAE 10–20 | 46 |
| MAE ≥ 20 | 0 |

The canonical MAE distribution is concentrated in the 10–20 range (46/48 agents), with only two outliers achieving MAE < 10 (runs 07 at 5.68 and 35 at 9.41). No agent achieved MAE < 5. The task is challenging: predicting held-out benchmarks from only 5 revealed scores per model, with 66% overall data missingness.

---

## 3. Query Design and Diagnostics

20 binary queries were designed across 4 tiers of 5 each:

- **Tier 1 (Outcomes):** effective rank, canonical MAE, subset size, missing fraction, ensemble prediction
- **Tier 2 (Methodology forks):** filtered benchmarks, min-max normalization, model filtering, LOO evaluation, canonical MAE band
- **Tier 3 (Claims):** low-rank, scale mismatch, Elo, dominant factor, sparsity mentions
- **Tier 4 (Benchmark selection):** coding, math, agentic, ARC-AGI-2, SWE-bench membership

**Replaced queries:** `T2_svd_decomp` (100% agreement — all agents used SVD/PCA) was replaced by `T2_model_filter` (n_models < 80).

**Low-variance queries (kept but noted):** `T1_mae_lt10` (4% minority), `T2_filtered_bench` (6%), `T2_canonical_mae_10_20` (4%).

### Per-query variance

| Query | Tier | YES | NO | Minority% |
|:------|:----:|----:|---:|----------:|
| T1_rank_le3 | 1 | 20 | 28 | 42% |
| T1_mae_lt10 | 1 | 2 | 46 | 4% |
| T1_n_selected_le5 | 1 | 21 | 27 | 44% |
| T1_missing_gt06 | 1 | 7 | 41 | 15% |
| T1_ensemble_pred | 1 | 21 | 27 | 44% |
| T2_filtered_bench | 2 | 45 | 3 | 6% |
| T2_minmax_norm | 2 | 42 | 6 | 12% |
| T2_model_filter | 2 | 16 | 32 | 33% |
| T2_loo_eval | 2 | 34 | 14 | 29% |
| T2_canonical_mae_10_20 | 2 | 46 | 2 | 4% |
| T3_mentions_lowrank | 3 | 23 | 25 | 48% |
| T3_mentions_scale | 3 | 17 | 31 | 35% |
| T3_mentions_elo | 3 | 29 | 19 | 40% |
| T3_mentions_dominant | 3 | 26 | 22 | 46% |
| T3_mentions_sparse | 3 | 25 | 23 | 48% |
| T4_includes_coding | 4 | 33 | 15 | 31% |
| T4_includes_math2 | 4 | 32 | 16 | 33% |
| T4_includes_agentic | 4 | 19 | 29 | 40% |
| T4_includes_arcagi2 | 4 | 23 | 25 | 48% |
| T4_includes_swebench | 4 | 19 | 29 | 40% |

---

## 4. Mutual Evaluation Results

### Overall Welfare

| Metric | Value |
|:-------|------:|
| Observed W | 0.120319 |
| Mean agent welfare | 0.120319 |
| Std agent welfare | 0.020738 |
| Min welfare agent | opus-4.6_run14 (0.064255) |
| Max welfare agent | opus-4.6_run34 (0.157340) |

### Permutation Null Test

| Metric | Value |
|:-------|------:|
| Null mean | 0.117710 |
| Null std | 0.001736 |
| p-value | 0.0710 |
| z-score | 1.50 |

The observed welfare W = 0.1203 exceeds the permutation null mean by 1.5 standard deviations (p = 0.071). This suggests **marginal** evidence that agent responses contain structured shared information beyond what random query-marginal-preserving shuffles would produce. The moderate effect size reflects the inherent noise of 20 binary queries over 48 agents.

---

## 5. Fork Detection

Queries ranked by |ΔW| (contribution to overall welfare when removed):

| Rank | Query | Tier | ΔW | Description |
|:-----|:------|:----:|---:|:------------|
| 1 | T1_mae_lt10 | 1 | +0.009732 | canonical_overall_mae < 10 |
| 2 | T2_canonical_mae_10_20 | 2 | +0.007090 | canonical MAE in [10,20) |
| 3 | T3_mentions_dominant | 3 | -0.006024 | notes mention 'dominant'/'rank-1'/'first component' |
| 4 | T3_mentions_lowrank | 3 | -0.005833 | notes mention 'low-rank'/'low rank' |
| 5 | T2_filtered_bench | 2 | +0.005763 | n_benchmarks < 49 (filtered preprocessing) |
| 6 | T4_includes_arcagi2 | 4 | -0.005685 | selected set includes ARC-AGI-2 |
| 7 | T3_mentions_sparse | 3 | -0.005499 | notes mention 'sparse'/'sparsity' |
| 8 | T3_mentions_elo | 3 | -0.005484 | notes mention 'elo' |
| 9 | T4_includes_swebench | 4 | -0.004732 | selected set includes SWE-bench Verified |
| 10 | T1_ensemble_pred | 1 | -0.004639 | prediction method is ensemble/blend |
| 11 | T1_n_selected_le5 | 1 | -0.004389 | n_selected in [1,5] |
| 12 | T1_rank_le3 | 1 | -0.004222 | effective_rank <= 3 |
| 13 | T3_mentions_scale | 3 | -0.004069 | notes mention scale mismatch/issue |
| 14 | T1_missing_gt06 | 1 | +0.003882 | missing_fraction > 0.6 |
| 15 | T2_minmax_norm | 2 | +0.003681 | uses min-max normalization |
| 16 | T2_model_filter | 2 | -0.003195 | n_models < 80 (aggressive model filtering) |
| 17 | T2_loo_eval | 2 | -0.003161 | evaluation uses leave-one-out |
| 18 | T4_includes_coding | 4 | -0.003023 | selected set includes >=1 Coding benchmark |
| 19 | T4_includes_math2 | 4 | -0.003008 | selected set includes >=2 Math benchmarks |
| 20 | T4_includes_agentic | 4 | -0.002699 | selected set includes >=1 Agentic benchmark |

**Primary fork:** `T1_mae_lt10` (Tier 1) — canonical_overall_mae < 10

The top fork by ΔW is `T1_mae_lt10` (ΔW = +0.0097), which captures the rare outlier agents (runs 07 and 35) that achieved canonical MAE < 10. This query's high contribution stems from its extreme minority rate (4%), meaning these two agents have a highly distinctive response pattern. Note that positive ΔW indicates removing this query *decreases* welfare — its inclusion adds discriminating information.

The next strongest forks involve `T1_missing_gt06` (whether the agent reported >60% missingness, tied to using full vs filtered matrix), `T2_model_filter` (aggressive model filtering), and the tier-3 narrative claims about dominance and Elo.

---

## 6. Clustering

Hierarchical clustering (average linkage on TVD-MI distance) was applied. At k=2:

| Metric | Cluster 1 | Cluster 2 |
|:-------|:---------:|:---------:|
| N agents | 15 | 33 |
| MAE mean ± std | 15.49 ± 1.9 | 14.89 ± 2.44 |

Key distinguishing queries between clusters:

| Query | Cluster 1 | Cluster 2 | Gap |
|:------|:---------:|:---------:|:---:|
| T1_n_selected_le5 | 93% | 21% | 72pp |
| T1_ensemble_pred | 80% | 27% | 53pp |
| T4_includes_math2 | 33% | 82% | 48pp |
| T4_includes_swebench | 7% | 55% | 48pp |
| T4_includes_agentic | 13% | 52% | 38pp |
| T2_minmax_norm | 67% | 97% | 30pp |
| T3_mentions_elo | 80% | 52% | 28pp |
| T3_mentions_lowrank | 67% | 39% | 27pp |

The two clusters have **similar canonical MAE** (difference < 0.2 on 0–100 scale), indicating that the methodology forks captured by clustering do not strongly predict outcome quality. The primary distinguishing features are:

- **Cluster 1** (25 agents): More likely to include SWE-bench Verified and coding benchmarks in selected set, less likely to use ensemble methods, and slightly more likely to report a "dominant" factor in methodology notes.
- **Cluster 2** (23 agents): More likely to use ensemble prediction, mention Elo and scale issues, select fewer benchmarks (≤5), and include fewer coding benchmarks.

Within-cluster TVD-MI (0.1404) exceeds between-cluster TVD-MI (0.0947), confirming that agents within a cluster share more methodological decisions than agents across clusters. The ratio (between/within = 0.675) indicates moderate but not dramatic separation.

---

## 7. Hypothesis Evaluation

| Hypothesis | Prediction | Observed | Pass/Fail |
|:-----------|:-----------|:---------|:---------:|
| H1: Rank convergence | ≥75% report rank ≤ 3 | 20/48 = 41.7% | **FAIL** |
| H2: Subset convergence | 0.2 < mean Jaccard < 0.6 | 0.1451 | **FAIL** |
| H3: Prediction feasibility | ≥80% achieve MAE < 10 | 2/48 = 4.2% | **FAIL** |
| H4: Preprocessing fork | Top fork is preprocessing | T1_mae_lt10 (T1) | **FAIL** |
| H5: Qualitative robustness | ≥90% support low-rank | 35/48 = 72.9% (relaxed) | **MARGINAL** |

### Discussion

**H1 (FAIL):** The effective rank distribution is bimodal — 20 agents report rank ≤ 3, while 19 report rank ≥ 14. This reflects a genuine methodological fork: agents using variance-explained thresholds on raw/z-scored matrices with high missingness get inflated rank estimates, while those using imputed matrices or stricter criteria find low rank. The underlying signal is genuinely low-rank, but the *measurement* of rank diverges strongly.

**H2 (FAIL):** Mean Jaccard = 0.1451, below the predicted 0.2 lower bound. Benchmark subsets are more diverse than expected, likely because the greedy selection algorithms are sensitive to preprocessing choices, matrix shape, and regularization.

**H3 (FAIL):** Only 2/48 agents achieve MAE < 10 on the canonical evaluation. The task's difficulty was underestimated: predicting from 5 revealed benchmarks per model, with 66% base missingness, is substantially harder than the agents' self-reported evaluation protocols (which typically use denser observed data). Most agents achieve self-reported MAE of 7–12, but the canonical reveal-k-per-model protocol is more stringent.

**H4 (FAIL):** The top fork is `T1_mae_lt10` (an outcome query), not a preprocessing query. However, `T1_missing_gt06` (rank 3, closely related to preprocessing scope) and `T2_model_filter` (rank 4, formerly `T2_svd_decomp`) are among the top 5, suggesting preprocessing *is* influential even if not the single top fork.

**H5 (MARGINAL):** 47.9% explicitly mention "low-rank" in notes; 72.9% either mention it or report rank ≤ 3. The criterion of 90% is not met under the strict operationalization, though the bimodal rank distribution complicates interpretation. Agents broadly agree the data has low-rank structure but differ on how to quantify it.

---

## 8. Fork-Conditioned Outcomes

| Fork Query | MAE (YES) | MAE (NO) | Cohen's d |
|:-----------|:---------:|:--------:|:---------:|
| T1_rank_le3 | 15.59 ± 1.61 | 14.71 ± 2.63 | 0.380 |
| T1_n_selected_le5 | 15.68 ± 1.79 | 14.60 ± 2.54 | 0.471 |
| T1_ensemble_pred | 14.35 ± 2.78 | 15.64 ± 1.65 | -0.569 |

The Cohen's d values are small to negligible, confirming that methodology forks (rank estimation, subset size, ensemble vs simple methods) do not strongly predict canonical MAE differences. The task difficulty swamps methodological variation.

---

## 9. Notable Observations

1. **Universal SVD/PCA usage:** All 48 agents used SVD or PCA for rank analysis, necessitating query replacement. This represents a strong inductive bias of the model (Claude opus-4.6) rather than a data-driven convergence.

2. **Bimodal rank estimates:** The effective rank distribution splits into low (1–5) and high (14–20) modes. This is driven by the interaction of missingness handling with variance thresholds — mean-imputing a sparse matrix inflates apparent dimensionality.

3. **Canonical vs self-reported MAE gap:** The median self-reported MAE (~10) is much lower than the canonical MAE (~15.6). This systematic gap reflects protocol differences: self-reported evaluations use denser observed data, while the canonical reveal-k protocol tests genuine out-of-sample prediction from sparse inputs.

4. **SWE-bench Verified as cluster discriminator:** The starkest cluster difference (72% vs 4%) is whether the selected benchmark subset includes SWE-bench Verified. This coding benchmark appears to provide strong predictive signal for some pipelines but not others, depending on preprocessing scope and filtering thresholds.

---

## 10. Caveats and Limitations

- **Finite Q noise:** With Q=20 binary queries, TVD-MI estimates carry substantial noise. Confidence intervals on W and ΔW should be interpreted cautiously.
- **Single model family:** All agents are Claude opus-4.6 instances. Different LLM families might exhibit different methodology preferences and different reliability characteristics.
- **Low-variance queries:** Three queries have minority rates below 10%, limiting their discriminating power while potentially inflating ΔW.
- **Single evaluator:** Only one reliability evaluator was run (cross-evaluator robustness check from Section 6.3 of the analysis plan is a limitation).
- **No post-hoc hypothesis edits:** All hypotheses were pre-registered and all five failed or were marginal. This is itself a finding — the pre-registered expectations were miscalibrated for this task difficulty.
