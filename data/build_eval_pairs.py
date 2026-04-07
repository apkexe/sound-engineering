"""
Build evaluation pairs CSV from downloaded MIPPIA-style files.

Expected filename pattern in --audio_dir:
  <pair_id>_original.mp3
  <pair_id>_similar.mp3

Creates positives:
  (pair_id_original, pair_id_similar, label=1)

Creates negatives by mismatching originals/similars across different pair_ids:
  (pair_i_original, pair_j_similar, label=0)

Usage:
  python data/build_eval_pairs.py --audio_dir data/mippia --out_csv data/eval_pairs.csv --neg_ratio 1.0
"""

import argparse
import random
from pathlib import Path

import pandas as pd


SUPPORTED_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def _collect_pairs(audio_dir: Path):
    originals = {}
    similars = {}

    for p in audio_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        stem = p.stem
        if stem.endswith("_original"):
            pair_id = stem[: -len("_original")]
            originals[pair_id] = str(p)
        elif stem.endswith("_similar"):
            pair_id = stem[: -len("_similar")]
            similars[pair_id] = str(p)

    common_ids = sorted(set(originals) & set(similars))
    return common_ids, originals, similars


def build_eval_pairs(audio_dir: str, out_csv: str, neg_ratio: float = 1.0, seed: int = 42):
    rng = random.Random(seed)

    audio_dir_p = Path(audio_dir)
    ids, originals, similars = _collect_pairs(audio_dir_p)
    if len(ids) < 2:
        raise ValueError("Need at least 2 complete pairs to generate negatives.")

    rows = []

    # Positives
    for pair_id in ids:
        rows.append({
            "track_a": originals[pair_id],
            "track_b": similars[pair_id],
            "label": 1,
            "pair_type": "positive",
        })

    # Negatives
    n_pos = len(ids)
    n_neg = int(round(n_pos * neg_ratio))

    for _ in range(n_neg):
        a_id = rng.choice(ids)
        b_id = rng.choice(ids)
        while b_id == a_id:
            b_id = rng.choice(ids)

        rows.append({
            "track_a": originals[a_id],
            "track_b": similars[b_id],
            "label": 0,
            "pair_type": "negative",
        })

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Saved {len(df)} rows to {out_csv}")
    print(f"Positives: {(df['label'] == 1).sum()} | Negatives: {(df['label'] == 0).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build balanced eval pairs CSV")
    parser.add_argument("--audio_dir", required=True, help="Directory with *_original.mp3 and *_similar.mp3")
    parser.add_argument("--out_csv", default="data/eval_pairs.csv", help="Output CSV path")
    parser.add_argument("--neg_ratio", type=float, default=1.0, help="Negative-to-positive ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_eval_pairs(args.audio_dir, args.out_csv, args.neg_ratio, args.seed)
