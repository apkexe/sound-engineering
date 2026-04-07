"""
Create a fixed 10-pair evaluation subset (5 positive + 5 negative) for fast iteration.
Seed is fixed so the subset is always the same.
"""
import pandas as pd
from pathlib import Path

def main():
    src = Path("data/eval_pairs_mippia.csv")
    df = pd.read_csv(src)

    pos = df[df["label"] == 1].sample(5, random_state=42)
    neg = df[df["label"] == 0].sample(5, random_state=42)
    subset = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    out = Path("experiments/subset_10pairs.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out, index=False)
    print(f"Created {out} with {len(subset)} pairs ({len(pos)} pos, {len(neg)} neg)")
    print(subset[["track_a", "track_b", "label"]].to_string(index=False))

if __name__ == "__main__":
    main()
