
# /experiment_protocol.md
# BenchPress Reliability Audit: Experiment Protocol (v1.0)

**Author:** Zachary Robertson  
**Date:** 2026-02-26  
**Status:** Pre-registered (POC completed; excluded from analysis)

---

## 1. Artifacts frozen at preregistration time

This preregistration consists of the following frozen files:

1. `/benchpress_specification.md` (analysis-agent task spec)
2. `/reliability_specification.md` (mutual-evaluation reliability spec)
3. `/canonical_evaluation.md` (canonical split + scoring)
4. `/analysis_plan.md` (hypotheses + analysis)
5. `/experiment_protocol.md` (this file)

No other task-relevant instructions are provided to agents.

---

## 2. Dataset provenance & versioning

The analysis agents receive `llm_benchmark_data.json` exactly as provided (unmodified). The preregistration should include a deterministic identifier (fill in when publishing):
- `DATA_SHA256 = 255ea00914119403032f90f9568e9b6236483ff8b6858f18c22b62fd5bebe449`
- `CANONICAL_MASK_SHA256 = f8432b73d5e96aa7664e1a350326ffc9724849461bf2333771425d445a8d139d`

---

## 3. Agent population and run parameters

### 3.1 Main runs
- **K (target):** 50 agents
- **Model:** Claude Opus 4.6 (`claude-opus-4-6`)
- **Temperature:** 1.0 (Anthropic SDK default)
- **Top-p:** 1.0 (Anthropic default)
- **Max turns:** 12
- **Execution timeout:** 120s per `scratch.py` run
- **Tools:** local python execution only (no internet)
- **Prompt:** agent receives only `/benchpress_specification.md` as user message (plus files in working directory)

### 3.2 Independence guarantees
Each agent runs in an isolated directory with:
- fresh filesystem state
- only the provided input files
- no access to other agents’ directories or outputs
- no persistent memory between runs

Agents may not see:
- the POC trace
- any other agent outputs
- any prior BenchPress writeups

---

## 4. Inclusion/exclusion rules

### 4.1 What counts as “completed”
An agent is counted as **SUCCESS** iff it produces:
- `results_summary.json` with required keys
- `canonical_predictions.csv` with required columns
- ≥95% canonical coverage (as defined in `/reliability_specification.md`)

Otherwise, **FAILURE**.

### 4.2 Reruns
No reruns are allowed except for *infrastructure failures* that are clearly non-agentic, defined as:
- corrupted file copy into working directory
- harness crash before the agent receives the spec
- empty output directory due to IO failure

Reruns due to:
- timeouts,
- JSON formatting errors,
- incorrect results,
are **not** allowed.

**Resume policy:** When running the K=50 batch in multiple sessions, the harness may skip already-completed agent directories (resume). This is not a "rerun" — it simply avoids re-executing agents whose outputs already exist. The harness checks for completion by verifying an output directory contains the required files.

---

## 5. POC disclosure (excluded from K=50)

One POC run was executed before preregistration to validate that the specification is feasible within the turn budget and to identify major forks.

POC highlights:
- Dropped benchmarks/models under sparse coverage thresholds and normalized scores to [0,100] 
- Selected 8 benchmarks and achieved overall MAE ≈ 4.953 on a normalized [0,100] scale in its own evaluation 
- Reported effective rank in the 1–3 range depending on criterion and emphasized a dominant rank-1 factor 

The POC output is excluded from all aggregate statistics and mutual-evaluation analysis.

---

## 6. Logging & release plan

For each agent run, the harness saves:
- full conversation trace
- all produced files
- stdout/stderr logs from `scratch.py`

Release (planned):
- all SUCCESS/FAILURE traces
- all artifacts produced by agents
- evaluation scripts for canonical scoring and reliability analysis
- the preregistration files listed in Section 1

---

## 7. Three-specification architecture

The audit involves three distinct specification types, executed in sequence:

| # | Artifact | Type | Runs | When |
|:--|:---------|:-----|:-----|:-----|
| 1 | `generate_canonical_mask.py` | Deterministic script (not an agent) | 1 | Before experiment; produces `canonical_mask.json` |
| 2 | `benchpress_specification.md` | Analysis agent task spec | K = 50 | Main experiment; each agent runs independently |
| 3 | `reliability_specification.md` | Reliability evaluator agent spec | 1–2 | After all analysis agents complete |

The canonical mask script is a fixed Python program with no stochastic elements beyond the pre-registered seed. The analysis agents are LLM agents with temperature 1.0 (Anthropic SDK default). The reliability evaluator is also an LLM agent, but runs once (or twice for cross-evaluator robustness) with read-only access to all analysis agent outputs.

---

## 8. Primary endpoints

The experiment’s primary endpoints are computed in `/analysis_plan.md`:
- convergence on rank claims
- convergence / dispersion in benchmark subset selection
- canonical prediction performance distribution
- TVD-MI welfare + fork structure from mutual evaluation