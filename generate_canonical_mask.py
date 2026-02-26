#!/usr/bin/env python3
"""
Generate canonical_mask.json for the BenchPress Reliability Audit.

Implements the deterministic reveal-k-per-model mask defined in canonical_evaluation.md:
- CANONICAL_SEED = 20260226
- REVEAL_K = 5 benchmarks revealed per evaluated model
- N_EVAL_MODELS = 12 evaluated models (sampled deterministically from eligible models)
- MIN_CELLS_PER_MODEL_TO_EVAL = 15 observed cells required for eligibility
- Per-model RNG seeded by (CANONICAL_SEED, model_id)

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
REVEAL_K = 5
N_EVAL_MODELS = 12
MIN_CELLS_PER_MODEL_TO_EVAL = 15


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
                                pairs_by_benchmark[str(bid)].append(
                                    (str(mid), str(bid))
                                )
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
                                pairs_by_benchmark[str(bid)].append(
                                    (str(mid), str(bid))
                                )
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


def make_model_seed(canonical_seed, model_id):
    """
    Create a deterministic integer seed from (CANONICAL_SEED, model_id).
    Uses SHA-256 to avoid collisions from simple concatenation.
    """
    h = hashlib.sha256(f"{canonical_seed}:{model_id}".encode()).hexdigest()
    return int(h[:16], 16)


def generate_mask(data):
    """
    Generate the canonical reveal-k-per-model mask.

    Returns:
        dict with keys:
          - seed
          - reveal_k
          - n_eval_models
          - min_cells_per_model_to_eval
          - eval_models
          - revealed
          - pairs  (held-out pairs across eval_models)
    """
    pairs_by_benchmark = extract_observed_pairs(data)

    if not pairs_by_benchmark:
        print(
            "ERROR: Could not extract any (model, benchmark) pairs from data.",
            file=sys.stderr,
        )
        print(
            "Please check the data schema and update extract_observed_pairs().",
            file=sys.stderr,
        )
        sys.exit(1)

    # Invert to model -> set(benchmark_id)
    benchmarks_by_model = defaultdict(set)
    for bid, omega_b in pairs_by_benchmark.items():
        for mid, bid_ in set(omega_b):
            benchmarks_by_model[str(mid)].add(str(bid_))

    eligible_models = sorted(
        [
            mid
            for mid, bset in benchmarks_by_model.items()
            if len(bset) >= MIN_CELLS_PER_MODEL_TO_EVAL
        ]
    )
    if not eligible_models:
        print(
            f"ERROR: No eligible models found with >= {MIN_CELLS_PER_MODEL_TO_EVAL} observed cells.",
            file=sys.stderr,
        )
        sys.exit(1)

    rng_models = random.Random(CANONICAL_SEED)
    n_eval = min(N_EVAL_MODELS, len(eligible_models))
    eval_models = sorted(rng_models.sample(eligible_models, n_eval))

    revealed = []
    holdout_pairs = []
    stats = {
        "eligible_models": len(eligible_models),
        "n_eval_models": len(eval_models),
        "total_heldout_pairs": 0,
    }

    for mid in eval_models:
        observed_benchmarks = sorted(list(benchmarks_by_model[mid]))

        rng_reveal = random.Random(make_model_seed(CANONICAL_SEED, mid))
        reveal_benchmarks = sorted(rng_reveal.sample(observed_benchmarks, REVEAL_K))

        revealed.append({"model_id": mid, "benchmark_ids": reveal_benchmarks})

        # Hold out all other observed benchmarks for this model
        holdout_benchmarks = [
            b for b in observed_benchmarks if b not in set(reveal_benchmarks)
        ]
        for bid in holdout_benchmarks:
            holdout_pairs.append({"model_id": mid, "benchmark_id": bid})

    stats["total_heldout_pairs"] = len(holdout_pairs)

    print("Mask generation complete:", file=sys.stderr)
    print(
        f"  Eligible models (>= {MIN_CELLS_PER_MODEL_TO_EVAL} cells): {stats['eligible_models']}",
        file=sys.stderr,
    )
    print(f"  Evaluated models: {stats['n_eval_models']}", file=sys.stderr)
    print(f"  Reveal K: {REVEAL_K}", file=sys.stderr)
    print(f"  Total held-out pairs: {stats['total_heldout_pairs']}", file=sys.stderr)

    return {
        "seed": CANONICAL_SEED,
        "reveal_k": REVEAL_K,
        "n_eval_models": len(eval_models),
        "min_cells_per_model_to_eval": MIN_CELLS_PER_MODEL_TO_EVAL,
        "eval_models": eval_models,
        "revealed": revealed,
        "pairs": holdout_pairs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate canonical reveal-k mask for BenchPress audit"
    )
    parser.add_argument("--data", required=True, help="Path to llm_benchmark_data.json")
    parser.add_argument(
        "--output",
        default="canonical_mask.json",
        help="Output path (default: canonical_mask.json)",
    )
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    mask = generate_mask(data)

    with open(args.output, "w") as f:
        json.dump(mask, f, indent=2)

    print(f"Saved {args.output} ({len(mask['pairs'])} held-out pairs)", file=sys.stderr)


if __name__ == "__main__":
    main()
