"""
Run full 15-branch pipeline on the 10-pair subset and derive both:
  - 13-branch baseline scores (weights zeroed for dmax+tonnetz, renormalized)
  - 15-branch improved scores (full pipeline)

This avoids running the expensive pipeline twice.
"""
import json, sys, time, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["MAIA_SKIP_SRCSEP"] = "1"  # Skip source separation for speed

import numpy as np
import pandas as pd
from pipeline import compare_tracks

SUBSET_CSV = "experiments/subset_10pairs.csv"
OUT_DIR = Path("experiments/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 15-branch weights (current)
W15 = {
    "semantic": 0.13, "melodic": 0.05, "structural": 0.02,
    "artifact_diff": 0.01, "ssm": 0.01, "spectral_corr": 0.01,
    "rhythm": 0.19, "mert": 0.04, "stem_combined": 0.02,
    "cqt": 0.01, "qmax": 0.12, "clap_multiscale": 0.12,
    "panns": 0.17, "dmax": 0.04, "tonnetz": 0.06,
}

# 13-branch baseline: zero out dmax & tonnetz, redistribute proportionally
W13 = {k: v for k, v in W15.items() if k not in ("dmax", "tonnetz")}
w13_sum = sum(W13.values())
W13 = {k: round(v / w13_sum, 6) for k, v in W13.items()}

# Mapping from result dict keys to weight keys
SCORE_KEY_MAP = {
    "semantic": "semantic_similarity",
    "melodic": "melodic_alignment",
    "structural": "structural_correspondence",
    "artifact_diff": "artifact_diff",
    "ssm": "ssm_similarity",
    "spectral_corr": "spectral_correlation",
    "rhythm": "rhythm_similarity",
    "mert": "mert_similarity",
    "stem_combined": "stem_combined",
    "cqt": "cqt_similarity",
    "qmax": "qmax_score",
    "clap_multiscale": "clap_multiscale",
    "panns": "panns_similarity",
    "dmax": "dmax_score",
    "tonnetz": "tonnetz_similarity",
}


def recompute_score(result_dict, weights):
    """Recompute attribution score with different weights from sub-scores."""
    total = 0.0
    for wk, wv in weights.items():
        sk = SCORE_KEY_MAP[wk]
        total += wv * result_dict.get(sk, 0.5)
    return float(np.clip(total, 0.0, 1.0))


def summarize(pairs, name, threshold=0.72):
    """Compute summary metrics from a list of pair results."""
    pos = [p for p in pairs if p["true_label"] == 1]
    neg = [p for p in pairs if p["true_label"] == 0]

    mean_pos = np.mean([p["score"] for p in pos])
    mean_neg = np.mean([p["score"] for p in neg])

    predictions = [int(p["score"] >= threshold) for p in pairs]
    labels = [p["true_label"] for p in pairs]
    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct / len(pairs)

    tp = sum(p == 1 and l == 1 for p, l in zip(predictions, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(predictions, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(predictions, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(predictions, labels))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "experiment": name,
        "accuracy": round(accuracy, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "threshold": threshold,
        "mean_score_pos": round(mean_pos, 4),
        "mean_score_neg": round(mean_neg, 4),
        "score_gap": round(mean_pos - mean_neg, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def main():
    df = pd.read_csv(SUBSET_CSV)
    print(f"\nRunning 15-branch pipeline on {len(df)} pairs...")
    print(f"(Source separation: SKIPPED)")
    print(f"{'='*60}\n")

    raw_results = []
    t0 = time.time()

    for i, row in df.iterrows():
        pair_id = row.get("pair_id", f"pair_{i}")
        print(f"[{i+1}/{len(df)}] {pair_id}: {Path(row.track_a).name} vs {Path(row.track_b).name}")
        t1 = time.time()
        try:
            result = compare_tracks(row.track_a, row.track_b, verbose=False)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        elapsed_pair = time.time() - t1
        result["pair_id"] = pair_id
        result["true_label"] = int(row.label)
        result["elapsed_s"] = round(elapsed_pair, 1)
        raw_results.append(result)
        print(f"  score={result['attribution_score']:.4f}  label={row.label}  ({elapsed_pair:.1f}s)")

    total_elapsed = time.time() - t0
    print(f"\nTotal time: {total_elapsed:.1f}s\n")

    # Save raw 15-branch results
    with open(OUT_DIR / "subset_15branch_raw.json", "w") as f:
        json.dump(raw_results, f, indent=2)

    # Derive both sets of scores (recomputed from sub-scores, not from pipeline output)
    pairs_13 = []
    pairs_15 = []
    for r in raw_results:
        s13 = recompute_score(r, W13)
        s15 = recompute_score(r, W15)
        pairs_13.append({"pair_id": r["pair_id"], "true_label": r["true_label"], "score": s13})
        pairs_15.append({"pair_id": r["pair_id"], "true_label": r["true_label"], "score": s15})

    sum13 = summarize(pairs_13, "baseline_13branch")
    sum15 = summarize(pairs_15, "improved_15branch")

    print("=" * 60)
    print("  BASELINE (13-branch, no dmax/tonnetz)")
    print("=" * 60)
    for k, v in sum13.items():
        print(f"  {k}: {v}")

    print(f"\n{'='*60}")
    print("  IMPROVED (15-branch, +dmax +tonnetz)")
    print("=" * 60)
    for k, v in sum15.items():
        print(f"  {k}: {v}")

    improvement = sum15["score_gap"] - sum13["score_gap"]
    print(f"\n--- Improvement ---")
    print(f"  Score gap: {sum13['score_gap']:.4f} -> {sum15['score_gap']:.4f} ({improvement:+.4f})")
    print(f"  Accuracy:  {sum13['accuracy']:.4f} -> {sum15['accuracy']:.4f}")

    # Per-pair comparison
    print(f"\n--- Per-Pair Comparison ---")
    print(f"  {'pair':<15} {'label':>5}  {'13br':>6}  {'15br':>6}  {'delta':>7}")
    for p13, p15 in zip(pairs_13, pairs_15):
        delta = p15["score"] - p13["score"]
        print(f"  {p13['pair_id']:<15} {p13['true_label']:>5}  {p13['score']:.4f}  {p15['score']:.4f}  {delta:+.4f}")

    # Save summaries
    comparison = {
        "baseline_13branch": sum13,
        "improved_15branch": sum15,
        "improvement_gap": round(improvement, 4),
        "per_pair": [
            {"pair_id": p13["pair_id"], "label": p13["true_label"],
             "score_13br": round(p13["score"], 4), "score_15br": round(p15["score"], 4),
             "delta": round(p15["score"] - p13["score"], 4)}
            for p13, p15 in zip(pairs_13, pairs_15)
        ],
        "total_elapsed_s": round(total_elapsed, 1),
    }
    with open(OUT_DIR / "subset_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nResults saved to experiments/results/")


if __name__ == "__main__":
    main()
