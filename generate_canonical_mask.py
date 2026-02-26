#!/usr/bin/env python3
"""
Generate canonical_mask.json for the BenchPress Reliability Audit.

Implements the deterministic holdout mask defined in canonical_evaluation.md:
- CANONICAL_SEED = 20260226
- Benchmark-stratified sampling: 20% holdout per benchmark
- Minimum 5 observed cells per benchmark to hold out any
- Per-benchmark RNG seeded by (CANONICAL_SEED, benchmark_id)

Usage:
    python generate_canonical_mask.py --data llm_benchmark_data.json --output canonical_mask.json
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict


CANONICAL_SEED = 20260226
HOLDOUT_FRACTION = 0.2
MIN_CELLS_PER_BENCHMARK = 5


def extract_observed_pairs(data):
    """
    Extract all observed (model_id, benchmark_id) pairs from the dataset.

    Returns:
        pairs_by_benchmark: dict mapping benchmark_id -> list of (model_id, benchmark_id)
    """
    pairs_by_benchmark = defaultdict(list)

    # The schema is unknown a priori — try common structures
    # Structure 1: list of entries with model/benchmark/score fields
    if isinstance(data, list):
        for entry in data:
            mid = entry.get("model_id") or entry.get("model")
            bid = entry.get("benchmark_id") or entry.get("benchmark")
            score = entry.get("score") or entry.get("value")
            if mid is not None and bid is not None and score is not None:
                pairs_by_benchmark[str(bid)].append((str(mid), str(bid)))
        if pairs_by_benchmark:
            return pairs_by_benchmark

    # Structure 2: nested dict {model_id: {benchmark_id: score, ...}, ...}
    if isinstance(data, dict):
        # Check if values are dicts (model -> benchmarks mapping)
        sample_val = next(iter(data.values()), None)
        if isinstance(sample_val, dict):
            for mid, benchmarks in data.items():
                if isinstance(benchmarks, dict):
                    for bid, score in benchmarks.items():
                        if score is not None:
                            try:
                                float(score)
                                pairs_by_benchmark[str(bid)].append((str(mid), str(bid)))
                            except (ValueError, TypeError):
                                continue
            if pairs_by_benchmark:
                return pairs_by_benchmark

        # Structure 3: {benchmark_id: {model_id: score, ...}, ...}
        # (transpose of structure 2)
        if isinstance(sample_val, dict):
            for bid, models in data.items():
                if isinstance(models, dict):
                    for mid, score in models.items():
                        if score is not None:
                            try:
                                float(score)
                                pairs_by_benchmark[str(bid)].append((str(mid), str(bid)))
                            except (ValueError, TypeError):
                                continue
            if pairs_by_benchmark:
                return pairs_by_benchmark

        # Structure 4: top-level key wrapping a list
        for key in ["data", "results", "entries", "scores"]:
            if key in data and isinstance(data[key], list):
                return extract_observed_pairs(data[key])

        # Structure 5: top-level key wrapping a dict of models
        for key in ["models", "data", "results"]:
            if key in data and isinstance(data[key], dict):
                return extract_observed_pairs(data[key])

    return pairs_by_benchmark


def make_benchmark_seed(canonical_seed, benchmark_id):
    """
    Create a deterministic integer seed from (CANONICAL_SEED, benchmark_id).
    Uses SHA-256 to avoid collisions from simple concatenation.
    """
    h = hashlib.sha256(f"{canonical_seed}:{benchmark_id}".encode()).hexdigest()
    return int(h[:16], 16)


def generate_mask(data):
    """
    Generate the canonical holdout mask.

    Returns:
        dict with keys: seed, holdout_fraction, min_cells_per_benchmark_to_holdout, pairs
    """
    pairs_by_benchmark = extract_observed_pairs(data)

    if not pairs_by_benchmark:
        print("ERROR: Could not extract any (model, benchmark) pairs from data.", file=sys.stderr)
        print("Please check the data schema and update extract_observed_pairs().", file=sys.stderr)
        sys.exit(1)

    holdout_pairs = []
    stats = {"total_benchmarks": len(pairs_by_benchmark), "benchmarks_held_out": 0, "benchmarks_skipped": 0}

    for bid in sorted(pairs_by_benchmark.keys()):
        omega_b = pairs_by_benchmark[bid]
        # Deduplicate (same model_id, benchmark_id may appear multiple times)
        omega_b = list(set(omega_b))

        if len(omega_b) < MIN_CELLS_PER_BENCHMARK:
            stats["benchmarks_skipped"] += 1
            continue

        n_test = max(1, round(HOLDOUT_FRACTION * len(omega_b)))

        # Seed RNG per-benchmark for determinism
        rng = random.Random(make_benchmark_seed(CANONICAL_SEED, bid))
        selected = rng.sample(omega_b, n_test)

        for mid, bid_ in selected:
            holdout_pairs.append({"model_id": mid, "benchmark_id": bid_})

        stats["benchmarks_held_out"] += 1

    print(f"Mask generation complete:", file=sys.stderr)
    print(f"  Total benchmarks: {stats['total_benchmarks']}", file=sys.stderr)
    print(f"  Benchmarks with holdout: {stats['benchmarks_held_out']}", file=sys.stderr)
    print(f"  Benchmarks skipped (< {MIN_CELLS_PER_BENCHMARK} cells): {stats['benchmarks_skipped']}", file=sys.stderr)
    print(f"  Total held-out pairs: {len(holdout_pairs)}", file=sys.stderr)

    return {
        "seed": CANONICAL_SEED,
        "holdout_fraction": HOLDOUT_FRACTION,
        "min_cells_per_benchmark_to_holdout": MIN_CELLS_PER_BENCHMARK,
        "pairs": holdout_pairs,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate canonical holdout mask for BenchPress audit")
    parser.add_argument("--data", required=True, help="Path to llm_benchmark_data.json")
    parser.add_argument("--output", default="canonical_mask.json", help="Output path (default: canonical_mask.json)")
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    mask = generate_mask(data)

    with open(args.output, "w") as f:
        json.dump(mask, f, indent=2)

    print(f"Saved {args.output} ({len(mask['pairs'])} held-out pairs)", file=sys.stderr)


if __name__ == "__main__":
    main()