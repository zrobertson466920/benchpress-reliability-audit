# BenchPress Reliability Audit: Pre-registration

Pre-registered experiment protocol for measuring the reliability of LLM-agent
data analysis through mutual evaluation.

## Overview

This experiment runs K=50 independent LLM agents on the same data analysis task
(characterizing low-rank structure in LLM benchmark performance data), then
evaluates inter-agent agreement using a TVD-MI mutual evaluation mechanism —
without human/subjective judging.

## Frozen Artifacts

The following files constitute the pre-registration and must not be modified
after the timestamp:

| File | Purpose |
|:-----|:--------|
| `experiment_protocol.md` | Run parameters, inclusion/exclusion rules |
| `benchpress_specification.md` | Analysis agent task specification |
| `canonical_evaluation.md` | Canonical reveal-k split + scoring |
| `reliability_specification.md` | Mutual evaluation (TVD-MI) specification |
| `analysis_plan.md` | Hypotheses and analysis plan |

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd benchpress-replication-prereg

# Install dependencies
pip install -r requirements.txt

# Configure API keys (one of):
# Option A: environment variable
export ANTHROPIC_API_KEY="sk-ant-..."

# Option B: config.py (git-ignored)
cat > config.py << 'EOF'
ANTHROPIC_API_KEYS = {
    "research": "sk-ant-...",
}
ANTHROPIC_MODEL = {
    "research": "claude-opus-4-6",
}
MAX_TOKENS = 16384
EOF
```

No `PYTHONPATH` manipulation needed — all dependencies are self-contained.

## Running the Experiment

### Step 1: Generate the canonical mask

```bash
python generate_canonical_mask.py \
    --data llm_benchmark_data.json \
    --output canonical_mask.json
```

### Step 2: Run analysis agents

```bash
# Sequential (1 at a time)
python run_experiment.py \
    --spec benchpress_specification.md \
    --runs 50 \
    --data-dir . \
    --project ./results

# Parallel (5 at a time)
python run_experiment.py \
    --spec benchpress_specification.md \
    --runs 50 \
    --batch-size 5 \
    --data-dir . \
    --project ./results \
    --quiet
```

The harness supports resume: if interrupted, re-running the same command skips
already-completed agents.

### Step 3: Run reliability evaluator

After all 50 agents complete, run the reliability evaluator against the
collected outputs:

```bash
python run_experiment.py \
    --spec reliability_specification.md \
    --runs 2 \
    --data-dir . \
    --project ./results
```

The evaluator agent reads analysis agent outputs from `./results/` and has
read-only access to the same data files. Two independent runs are recommended
for cross-evaluator robustness (see `analysis_plan.md` §6.3).

### Step 4: Retrieve traces (VM workflow)

```bash
bash retrieve_traces.sh ./results
# Creates benchpress_traces_YYYYMMDD_HHMMSS.tar.gz
# Transfer with scp/rsync as printed
```

## Key Parameters

| Parameter | Value | Source |
|:----------|:------|:-------|
| K (agents) | 50 | `experiment_protocol.md` §3.1 |
| Model | `claude-opus-4-6` | `experiment_protocol.md` §3.1 |
| Max turns | 12 | `experiment_protocol.md` §3.1 |
| Exec timeout | 120s | `experiment_protocol.md` §3.1 |
| Canonical seed | 20260226 | `canonical_evaluation.md` §3.1 |
| Reveal k (per eval model) | 5 | `canonical_evaluation.md` §3.1 |
| Evaluated models | 12 | `canonical_evaluation.md` §3.1 |
| Min cells per evaluated model | 15 | `canonical_evaluation.md` §3.1 |
| Binary queries | 20 | `reliability_specification.md` §Step 3 |

## Data

`llm_benchmark_data.json` contains LLM benchmark performance data. The schema
is intentionally undocumented to agents — schema discovery is a measured degree
of freedom.

## File Manifest

All code is self-contained — no external imports beyond `requirements.txt`.

| File | Role |
|:-----|:-----|
| `dependencies.py` | Distilled runtime (LLM, edits, conversation storage) |
| `agent_core.py` | Headless agent loop |
| `run_experiment.py` | Experiment runner (sequential/parallel) |
| `generate_canonical_mask.py` | Deterministic reveal-k mask generator |
| `retrieve_traces.sh` | Package traces for transfer off VM |
| `config.py` | API keys (git-ignored, create manually) |

Checksums (fill in before publishing):
- `DATA_SHA256 = 255ea00914119403032f90f9568e9b6236483ff8b6858f18c22b62fd5bebe449`
- `CANONICAL_MASK_SHA256 = f8432b73d5e96aa7664e1a350326ffc9724849461bf2333771425d445a8d139d`