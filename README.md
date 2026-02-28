# BenchPress Reliability Audit

Pre-registered experiment measuring the reliability of LLM-agent data analysis
through mutual evaluation. 50 independent Claude Opus 4.6 agents analyzed the
same LLM benchmark performance dataset, then two reliability evaluator agents
assessed inter-agent agreement using a TVD-MI mechanism — without ground-truth
judging.

**Paper:** [BenchPress](https://x.com/DimitrisPapail/status/2026531440414925307)  
**Pre-registration:** [`experiment_protocol.md`](experiment_protocol.md)

## Results at a Glance

- **50 analysis agents** completed (48 produced required outputs)
- **2 reliability evaluator agents** ran mutual evaluation
- **48 conversation traces** preserved (4 early traces lost to a harness bug)
- **Total cost:** ~$115 USD at Claude Opus 4.6 pricing

## Directory Structure

```
benchpress-reliability-audit/
├── README.md                    # This file
├── experiment_protocol.md       # Pre-registered protocol (frozen)
├── experiment_results.jsonl     # Per-run metadata (cost, turns, timing)
├── trace_manifest.json          # Status of all 54 runs (ok/lost/failed)
│
├── shared/                      # Input files (identical across all runs)
│   ├── analysis_plan.md         # Hypotheses and analysis plan
│   ├── benchpress_specification.md  # Analysis agent task spec
│   ├── canonical_evaluation.md  # Holdout split + scoring spec
│   ├── canonical_mask.json      # Deterministic holdout mask
│   ├── llm_benchmark_data.json  # Input dataset
│   └── reliability_specification.md  # Mutual evaluation spec
│
├── traces/                      # Sanitized conversation traces
│   ├── opus-4.6_run05.json      # Analysis agents (46 of 50)
│   ├── opus-4.6_run06.json
│   ├── ...
│   ├── opus-4.6-reliability_run01.json  # Reliability evaluators (2)
│   └── opus-4.6-reliability_run02.json
│
├── outputs/                     # Per-run output files (deduplicated)
│   ├── opus-4.6_run01/          # 50 analysis agent outputs
│   │   ├── results_summary.json
│   │   ├── prediction_results.json
│   │   ├── canonical_predictions.csv
│   │   ├── selected_benchmarks.json
│   │   ├── singular_values.json
│   │   ├── performance_matrix.csv
│   │   ├── cleaned_matrix.csv
│   │   └── scratch.py
│   ├── ...
│   ├── opus-4.6-reliability_run01/  # Reliability evaluator outputs
│   │   ├── reliability_summary.json
│   │   ├── reliability_report.md
│   │   ├── tvdmi_matrix.csv
│   │   ├── welfare.csv
│   │   ├── clusters.json
│   │   └── ...
│   └── opus-4.6-reliability_run02/
│
└── scripts/                     # Reproduction code
    ├── run_experiment.py        # Experiment runner
    ├── run_full_experiment.sh   # Full pipeline script
    ├── agent_core.py            # Headless agent loop
    ├── dependencies.py          # Distilled runtime
    ├── generate_canonical_mask.py  # Holdout mask generator
    ├── retrieve_traces.sh       # Trace packaging
    └── requirements.txt
```

## Key Parameters

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

## Missing Traces

Four early analysis agent traces (runs 01–04) were lost due to a harness bug
where the first five runs shared a single conversation ID. The trace file was
overwritten by each successive run, leaving only run 05's trace. All five runs'
**output files are preserved** — only the conversation logs are missing.

Two additional reliability evaluator entries in `experiment_results.jsonl` are
zero-cost failed starts (the harness retried successfully). These are documented
in `trace_manifest.json` with status `failed_start`.

See `trace_manifest.json` for the complete inventory.

## Data

`shared/llm_benchmark_data.json` contains LLM benchmark performance data. The
schema was intentionally undocumented to agents — schema discovery is a measured
degree of freedom.

## Reproduction

To re-run the experiment from scratch:

```bash
cd scripts/
pip install -r requirements.txt

# Configure API keys
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