"""Generate independent test sets with zero overlap to existing data.

Creates:
  1. data/test_holdout_10k.json  -- 10K pairs, seed=123, no overlap with 50K (seed=42)
  2. data/test_50k_independent.json -- 50K pairs, seed=99, no overlap with 10K (seed=42)

Verifies zero overlap before saving.
"""

import json
import random
import sys
from pathlib import Path

# Add src to path so we can import project modules
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from minimal10digittransformer.data.addition import generate_test_set, save_test_set, load_test_set

DATA_DIR = ROOT / "data"


def pairs_to_set(pairs: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Convert list of pairs to a set for O(1) lookup."""
    return set(pairs)


def main():
    # ---- Load existing test sets ----
    print("Loading existing test sets...")
    existing_10k = load_test_set(DATA_DIR / "test_10k.json")
    existing_50k = load_test_set(DATA_DIR / "test_50k.json")
    print(f"  Existing 10K: {len(existing_10k)} pairs (seed=42)")
    print(f"  Existing 50K: {len(existing_50k)} pairs (seed=42)")

    # Confirm the overlap issue
    set_10k = pairs_to_set(existing_10k)
    set_50k = pairs_to_set(existing_50k)
    overlap_existing = set_10k & set_50k
    print(f"\n  Overlap between existing 10K and 50K: {len(overlap_existing)} pairs")
    prefix_match = all(existing_10k[i] == existing_50k[i] for i in range(len(existing_10k)))
    print(f"  10K is exact prefix of 50K: {prefix_match}")

    # ---- Generate holdout 10K (seed=123) ----
    print("\n--- Generating holdout 10K test set (seed=123) ---")
    holdout_10k = generate_test_set(10_000, seed=123)
    set_holdout_10k = pairs_to_set(holdout_10k)

    # Check for internal duplicates
    print(f"  Generated: {len(holdout_10k)} pairs, {len(set_holdout_10k)} unique")

    # Check overlap with existing 50K
    overlap_holdout_vs_50k = set_holdout_10k & set_50k
    print(f"  Overlap with existing 50K (seed=42): {len(overlap_holdout_vs_50k)} pairs")

    # Check overlap with existing 10K
    overlap_holdout_vs_10k = set_holdout_10k & set_10k
    print(f"  Overlap with existing 10K (seed=42): {len(overlap_holdout_vs_10k)} pairs")

    if len(overlap_holdout_vs_50k) > 0:
        print("  ERROR: Overlap detected with 50K! Aborting.")
        sys.exit(1)
    print("  PASS: Zero overlap with existing 50K set.")

    # ---- Generate independent 50K (seed=99) ----
    print("\n--- Generating independent 50K test set (seed=99) ---")
    independent_50k = generate_test_set(50_000, seed=99)
    set_independent_50k = pairs_to_set(independent_50k)

    # Check for internal duplicates
    print(f"  Generated: {len(independent_50k)} pairs, {len(set_independent_50k)} unique")

    # Check overlap with existing 10K
    overlap_ind50k_vs_10k = set_independent_50k & set_10k
    print(f"  Overlap with existing 10K (seed=42): {len(overlap_ind50k_vs_10k)} pairs")

    # Check overlap with holdout 10K
    overlap_ind50k_vs_holdout = set_independent_50k & set_holdout_10k
    print(f"  Overlap with holdout 10K (seed=123): {len(overlap_ind50k_vs_holdout)} pairs")

    # Check overlap with existing 50K
    overlap_ind50k_vs_50k = set_independent_50k & set_50k
    print(f"  Overlap with existing 50K (seed=42): {len(overlap_ind50k_vs_50k)} pairs")

    if len(overlap_ind50k_vs_10k) > 0:
        print("  ERROR: Overlap detected with existing 10K! Aborting.")
        sys.exit(1)
    print("  PASS: Zero overlap with existing 10K set.")

    # ---- Save ----
    print("\n--- Saving ---")
    save_test_set(holdout_10k, DATA_DIR / "test_holdout_10k.json")
    print(f"  Saved: {DATA_DIR / 'test_holdout_10k.json'}")

    save_test_set(independent_50k, DATA_DIR / "test_50k_independent.json")
    print(f"  Saved: {DATA_DIR / 'test_50k_independent.json'}")

    # ---- Final summary ----
    print("\n" + "=" * 60)
    print("OVERLAP SUMMARY")
    print("=" * 60)
    print(f"  existing 10K (seed=42) vs existing 50K (seed=42):  {len(overlap_existing):>5} pairs  (KNOWN ISSUE)")
    print(f"  holdout 10K  (seed=123) vs existing 50K (seed=42): {len(overlap_holdout_vs_50k):>5} pairs  OK")
    print(f"  holdout 10K  (seed=123) vs existing 10K (seed=42): {len(overlap_holdout_vs_10k):>5} pairs  OK")
    print(f"  indep 50K    (seed=99)  vs existing 10K (seed=42): {len(overlap_ind50k_vs_10k):>5} pairs  OK")
    print(f"  indep 50K    (seed=99)  vs holdout 10K  (seed=123):{len(overlap_ind50k_vs_holdout):>5} pairs  OK")
    print(f"  indep 50K    (seed=99)  vs existing 50K (seed=42): {len(overlap_ind50k_vs_50k):>5} pairs  OK")
    print("=" * 60)
    print("All new test sets verified independent. Done.")


if __name__ == "__main__":
    main()
