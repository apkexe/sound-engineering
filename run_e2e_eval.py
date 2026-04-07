"""
End-to-End MIPPIA Evaluation
==============================
Runs the full pipeline: build eval CSV → evaluate → generate report.

Usage:
    python run_e2e_eval.py
    python run_e2e_eval.py --max_pairs 20        # quick baseline on first 20 pairs
    python run_e2e_eval.py --audio_dir data/mippia_full --neg_ratio 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.build_mippia_eval import build_eval_csv
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="End-to-end MIPPIA evaluation")
    parser.add_argument("--audio_dir", default="data/mippia_full")
    parser.add_argument("--eval_csv", default="data/eval_pairs_mippia.csv")
    parser.add_argument("--output", default="results/mippia_e2e_results.json")
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.805)
    parser.add_argument("--max_pairs", type=int, default=0,
                        help="Limit total pairs for quick baseline (0=all)")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    # Step 1: Build eval CSV
    print("=" * 60)
    print("STEP 1: Building evaluation pairs CSV")
    print("=" * 60)
    build_eval_csv(args.audio_dir, args.eval_csv, args.neg_ratio)

    # Step 2: Check how many pairs we have
    import pandas as pd
    df = pd.read_csv(args.eval_csv)

    # Optionally limit pairs for a quick run
    if args.max_pairs > 0 and len(df) > args.max_pairs:
        # Keep balanced: half pos, half neg
        pos = df[df["label"] == 1].head(args.max_pairs // 2)
        neg = df[df["label"] == 0].head(args.max_pairs // 2)
        df = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
        limited_csv = args.eval_csv.replace(".csv", f"_top{args.max_pairs}.csv")
        df.to_csv(limited_csv, index=False)
        args.eval_csv = limited_csv
        print(f"\nLimited to {len(df)} pairs for quick baseline")

    n_total = len(df)
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    print(f"\nReady to evaluate {n_total} pairs ({n_pos} positive, {n_neg} negative)")

    if n_total == 0:
        print("ERROR: No pairs to evaluate. Check audio_dir.")
        return

    # Step 3: Run evaluation
    print("\n" + "=" * 60)
    print("STEP 2: Running MAIA evaluation pipeline")
    print("=" * 60)

    start_time = time.time()
    results_df = evaluate(
        pairs_csv=args.eval_csv,
        output_path=args.output,
        threshold=args.threshold,
    )
    elapsed = time.time() - start_time

    print(f"\nTotal evaluation time: {elapsed:.1f}s")
    print(f"Average per pair: {elapsed / max(n_total, 1):.1f}s")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
