#!/bin/bash
set -euo pipefail

cd ~/projects/benchpress-reliability-audit
source .venv/bin/activate

# API key must be set before running
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set"
    echo "Run: export ANTHROPIC_API_KEY='sk-ant-...'"
    exit 1
fi

echo "=========================================="
echo "Step 1: K=50 analysis agents"
echo "=========================================="
python run_experiment.py \
    --spec benchpress_specification.md \
    --runs 50 \
    --batch-size 5 \
    --data-dir . \
    --project ./results

echo ""
echo "=========================================="
echo "Step 2: Sanity check"
echo "=========================================="
python -c "
import json, pandas as pd, os
mask = json.loads(open('canonical_mask.json').read())
success, fail = 0, 0
for d in sorted(os.listdir('results')):
    if not os.path.isdir(f'results/{d}'): continue
    rpath = f'results/{d}/results_summary.json'
    cpath = f'results/{d}/canonical_predictions.csv'
    if not os.path.exists(rpath) or not os.path.exists(cpath):
        print(f'  FAIL {d}: missing files')
        fail += 1
        continue
    try:
        r = json.loads(open(rpath).read())
        p = pd.read_csv(cpath)
        cov = p['y_pred'].notna().sum() / len(mask['pairs'])
        rank = r['rank_analysis']['effective_rank']
        mae = r['prediction']['overall_mae']
        status = 'SUCCESS' if cov >= 0.95 else 'FAIL'
        if status == 'FAIL': fail += 1
        else: success += 1
        print(f'  {status} {d}: rank={rank}, MAE={mae:.2f}, coverage={cov:.1%}')
    except Exception as e:
        print(f'  FAIL {d}: {e}')
        fail += 1
print(f'\nTotal: {success} SUCCESS, {fail} FAIL')
"

echo ""
echo "=========================================="
echo "Step 3: Reliability evaluator (2 runs)"
echo "=========================================="
python run_experiment.py \
    --spec reliability_specification.md \
    --runs 2 \
    --data-dir . \
    --project ./results

echo ""
echo "=========================================="
echo "Step 4: Package traces"
echo "=========================================="
bash retrieve_traces.sh ./results

echo ""
echo "=========================================="
echo "DONE"
echo "=========================================="