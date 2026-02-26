#!/usr/bin/env bash
# retrieve_traces.sh — Package experiment traces for transfer off VM
#
# Usage:
#   bash retrieve_traces.sh [results_dir] [output_archive]
#
# Defaults:
#   results_dir  = ./results
#   output_archive = benchpress_traces_$(date +%Y%m%d_%H%M%S).tar.gz

set -euo pipefail

RESULTS_DIR="${1:-./results}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="${2:-benchpress_traces_${TIMESTAMP}.tar.gz}"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR" >&2
    exit 1
fi

# Count completed runs
N_RUNS=$(find "$RESULTS_DIR" -name "results_summary.json" | wc -l | tr -d ' ')
N_DIRS=$(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')

echo "=== BenchPress Trace Retrieval ==="
echo "  Results dir:  $RESULTS_DIR"
echo "  Agent dirs:   $N_DIRS"
echo "  With results: $N_RUNS"
echo ""

# Include: all agent output dirs + experiment_results.jsonl + conversations/
INCLUDE_PATTERNS=(
    "$RESULTS_DIR"
)

# Also grab conversations if they exist
if [ -d "./conversations" ]; then
    INCLUDE_PATTERNS+=("./conversations")
    N_CONVS=$(find ./conversations -name "*.json" | wc -l | tr -d ' ')
    echo "  Conversations: $N_CONVS"
fi

echo ""
echo "Archiving to: $ARCHIVE"

tar -czf "$ARCHIVE" "${INCLUDE_PATTERNS[@]}"

SIZE=$(du -h "$ARCHIVE" | cut -f1)
echo "Done: $ARCHIVE ($SIZE)"
echo ""
echo "Transfer with:"
echo "  scp $ARCHIVE user@local:~/benchpress/"
echo "  # or"
echo "  rsync -avz $ARCHIVE user@local:~/benchpress/"
