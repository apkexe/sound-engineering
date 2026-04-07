"""
Build Evaluation CSV from Downloaded MIPPIA Pairs
===================================================
Creates a balanced evaluation CSV with:
  - Positive pairs: original <-> similar (true matches from the dataset)
  - Negative pairs: original_i <-> similar_j where i != j (random unrelated)

Usage:
    python data/build_mippia_eval.py
    python data/build_mippia_eval.py --audio_dir data/mippia_full --neg_ratio 2
"""

import argparse
import random
from pathlib import Path

import pandas as pd


def find_available_pairs(audio_dir: Path):
    """Find all complete pairs (both original and similar exist)."""
    pairs = {}
    for f in sorted(audio_dir.iterdir()):
        if f.is_file() and f.suffix in (".wav", ".mp3"):
            name = f.stem
            parts = name.rsplit("_", 1)
            if len(parts) == 2:
                pid, role = parts
                if pid not in pairs:
                    pairs[pid] = {}
                pairs[pid][role] = str(f)

    complete = {
        pid: roles for pid, roles in pairs.items()
        if "original" in roles and "similar" in roles
    }
    return complete


def build_eval_csv(audio_dir: str, output: str, neg_ratio: int = 1, seed: int = 42):
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        print(f"ERROR: {audio_dir} does not exist. Run download_all_mippia.py first.")
        return

    pairs = find_available_pairs(audio_dir)
    print(f"Found {len(pairs)} complete pairs in {audio_dir}")

    if len(pairs) < 2:
        print("ERROR: Need at least 2 pairs to build negatives.")
        return

    rows = []

    # Positive pairs: original <-> similar
    for pid, roles in sorted(pairs.items()):
        rows.append({
            "track_a": roles["original"],
            "track_b": roles["similar"],
            "label": 1,
            "pair_id": pid,
            "pair_type": "positive",
        })

    n_pos = len(rows)
    print(f"Positive pairs: {n_pos}")

    # Negative pairs: cross-pair combinations
    rng = random.Random(seed)
    pair_ids = sorted(pairs.keys())
    negatives = []

    for pid in pair_ids:
        # Pick neg_ratio different pairs to create negatives
        others = [p for p in pair_ids if p != pid]
        chosen = rng.sample(others, min(neg_ratio, len(others)))
        for other_pid in chosen:
            negatives.append({
                "track_a": pairs[pid]["original"],
                "track_b": pairs[other_pid]["similar"],
                "label": 0,
                "pair_id": f"{pid}_vs_{other_pid}",
                "pair_type": "negative",
            })

    # Balance: keep at most n_pos * neg_ratio negatives
    max_neg = n_pos * neg_ratio
    if len(negatives) > max_neg:
        negatives = rng.sample(negatives, max_neg)

    rows.extend(negatives)
    print(f"Negative pairs: {len(negatives)}")
    print(f"Total pairs: {len(rows)}")

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.to_csv(output, index=False)
    print(f"Saved to {output}")

    # Summary
    print(f"\nLabel distribution:")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="data/mippia_full")
    parser.add_argument("--output", default="data/eval_pairs_mippia.csv")
    parser.add_argument("--neg_ratio", type=int, default=1,
                        help="Number of negatives per positive pair")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_eval_csv(args.audio_dir, args.output, args.neg_ratio, args.seed)
