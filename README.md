# BenchPress Reliability Audit (preregistered)

50 independent Claude Opus 4.6 agents analyzed the same LLM benchmark dataset.
Two evaluator agents measured inter-agent agreement via TVD-MI i.e. no ground-truth
judging. All hypotheses, metrics, and analysis plans were frozen before execution.

**Original post:** <a href="https://x.com/DimitrisPapail/status/2026531440414925307" target="_blank">BenchPress - "You Don't Need to Run Every Eval"</a> (Dimitris Papailiopoulos)

**Pre-registration:** <a href="experiment_protocol.md" target="_blank">`experiment_protocol.md`</a> | <a href="https://osf.io/x36uk" target="_blank">OSF</a>

## Normalization alignment (read this first)

BenchPress reports **7.25% MedAPE** (median absolute percentage error in raw
mixed-scale space) and **MedianAE = 4.73** (raw score points). Our canonical
evaluation reports **median MAE ≈ 15.6** on a per-benchmark normalized 0–100
scale. These look incompatible. They mostly aren't.

Converting Dimitris's MedianAE 4.73 to the canonical 0–100 scale gives an
observation-weighted estimate of **~9.5 normalized points**. Converting his
MedAPE 7.25% gives **~11.4 normalized points**. Our canonical holdout is also
harder: it reveals only 5 scores per model (hiding ~78% of observed data vs
Dimitris's 50%), which extrapolates to +2–4 additional error points based on
his own hiding-fraction sweep. Accounting for both normalization and holdout
difficulty, the expected canonical MAE for Dimitris's pipeline is approximately
**12–15 normalized points** which is in the range of what our 48 agents produced.

The headline gap is mostly a scale artifact compounded by a harder holdout
protocol. The substantive findings are about what is and isn't robust once
you standardize the measurement.

## Key findings

1. 48/50 agents produced required outputs (96%).
2. Median MAE ~ 15.6; only 2/48 agents achieved MAE < 10. The 2 outliers
   approximate Dimitris's optimized pipeline. Prediction quality is comparable
   to BenchPress under canonical normalization.
3. All 5 preregistered hypotheses failed.
4. Effective rank is criterion-dependent:
   - 90% variance thresholds produce effective rank 10–20
   - Prediction-utility criteria produce effective rank 2–3
5. Low convergence on benchmark subset selection: mean pairwise Jaccard ≈ 0.145.
6. Cross-agent structure is weak but detectable, and query-sensitive:
   - Evaluator 01: W = 0.120, p = 0.069
   - Evaluator 02: W = 0.111, p = 0.002
   - The evaluator with *lower* welfare has the *lower* p-value because its queries
     extract less apparent structure but more of it is signal.

## What this means

1. Set up a normalization and holdout protocol before debating "replication."
   Most of the 7% vs 15.6 gap is a scale artifact.

2. Rank depends on what you optimize (variance explained vs prediction utility).
   "Low-rank" is not a single claim.

3. Agents succeed at the task yet disagree on which benchmark subsets matter.
   High execution reliability does not imply convergence of interpretation.

4. The evaluator extracting less total variation achieved stronger statistical
   significance. Measurement quality and quantity trade off.

## Results at a glance

| Metric | Value |
|:-------|:------|
| Analysis agents | 50 (48 SUCCESS, 2 FAILURE) |
| Reliability evaluators | 2 |
| Conversation traces preserved | 48 (4 early traces lost to harness bug) |
| Total cost | ~$115 USD |
| Canonical MAE median | 15.6 (normalized 0–100) |
| Canonical MAE range | 5.7 – 18.9 |
| Mean pairwise Jaccard | 0.145 |
| Cross-evaluator welfare correlation | r = 0.60 |

## Key parameters

| Parameter | Value | Source |
|:----------|:------|:-------|
| K (agents) | 50 | `experiment_protocol.md` §3.1 |
| Model | `claude-opus-4-6` | `experiment_protocol.md` §3.1 |
| Temperature | 1.0 | `experiment_protocol.md` §3.1 |
| Max turns | 12 | `experiment_protocol.md` §3.1 |
| Exec timeout | 120s | `experiment_protocol.md` §3.1 |
| Canonical seed | 20260226 | `shared/canonical_evaluation.md` §3.1 |
| Holdout fraction | 20% | `shared/canonical_evaluation.md` §3.1 |
| Binary queries | 20 | `shared/reliability_specification.md` §Step 3 |

## Hypothesis outcomes

| # | Hypothesis | Criterion | Observed | Result |
|:--|:-----------|:----------|:---------|:------:|
| H1 | Rank convergence | ≥75% effective rank ≤ 3 | 41.7% | FAIL |
| H2 | Subset partial convergence | 0.2 < mean Jaccard < 0.6 | J = 0.145 | FAIL |
| H3 | Prediction feasibility | ≥80% canonical MAE < 10 | 4.2% | FAIL |
| H4 | Preprocessing as primary fork | Top ΔW = preprocessing | Top = MAE threshold | FAIL |
| H5 | Qualitative robustness | ≥90% support low-rank | 47.9% | FAIL |

## Limitations

- **Normalization comparison is approximate.** Converting between MedAPE (relative)
  and normalized MAE (absolute after rescaling) requires distributional assumptions.
  The estimates above assume uniform error across benchmarks; actual error
  distributions are benchmark-dependent.
- **Finite Q noise.** 20 binary queries provide limited resolution for fork detection.
- **Single model family.** All agents are opus-4.6; inductive bias is shared.
- **Tautological top fork.** The MAE < 10 query dominates ΔW because it isolates
  2 extreme outliers, not a meaningful analytical school.
- **H5 untestable under prereg constraints.** No deterministic query achieves the
  ≥90% threshold needed; the low-rank claim is genuinely split ~50/50.
- **Cross-evaluator p-value divergence.** p = 0.069 vs p = 0.002 highlights
  sensitivity of mutual evaluation to query design choices.

## Directory structure

```
benchpress-reliability-audit/
├── README.md                    # This file
├── experiment_protocol.md       # Pre-registered protocol (frozen)
├── experiment_results.jsonl     # Per-run metadata (cost, turns, timing)
├── trace_manifest.json          # Status of all 54 runs
│
├── shared/                      # Input files (identical across all runs)
│   ├── analysis_plan.md
│   ├── benchpress_specification.md
│   ├── canonical_evaluation.md
│   ├── canonical_mask.json
│   ├── llm_benchmark_data.json
│   └── reliability_specification.md
│
├── traces/                      # Sanitized conversation traces
│   ├── opus-4.6_run05.json … opus-4.6_run50.json
│   ├── opus-4.6-reliability_run01.json
│   └── opus-4.6-reliability_run02.json
│
├── outputs/                     # Per-run output files
│   ├── opus-4.6_run01/ … opus-4.6_run50/
│   ├── opus-4.6-reliability_run01/
│   └── opus-4.6-reliability_run02/
│
└── scripts/                     # Reproduction code
    ├── run_experiment.py
    ├── run_full_experiment.sh
    ├── agent_core.py
    ├── dependencies.py
    ├── generate_canonical_mask.py
    ├── retrieve_traces.sh
    └── requirements.txt
```

## Missing traces

Four early analysis agent traces (runs 01–04) were lost due to a harness bug
where the first five runs shared a single conversation ID. All five runs'
**output files are preserved** — only the conversation logs are missing.
Two additional reliability evaluator entries in `experiment_results.jsonl`
are zero-cost failed starts. See `trace_manifest.json` for the complete inventory.

## Data

`shared/llm_benchmark_data.json` contains the same 83-model × 49-benchmark
matrix used in BenchPress (1,375 observed cells, 33.8% fill rate). The schema
was intentionally undocumented to agents — schema discovery is a measured
degree of freedom.

## Reproduction

```bash
cd scripts/
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."

# Generate canonical mask
python generate_canonical_mask.py \
    --data ../shared/llm_benchmark_data.json \
    --output ../shared/canonical_mask.json

# Run analysis agents
python run_experiment.py \
    --spec ../shared/benchpress_specification.md \
    --runs 50 --batch-size 5 \
    --data-dir ../shared \
    --project ../results

# Run reliability evaluators
python run_experiment.py \
    --spec ../shared/reliability_specification.md \
    --runs 2 \
    --data-dir ../shared \
    --project ../results
```

## Checksums

```
DATA_SHA256 = 255ea00914119403032f90f9568e9b6236483ff8b6858f18c22b62fd5bebe449
CANONICAL_MASK_SHA256 = 2a572935a067134718f207d59a1de29cb4f0aefbe94f81148da6b79bb896091c
```

