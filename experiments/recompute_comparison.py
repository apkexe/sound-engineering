"""
Recompute 13-branch vs 15-branch comparison from raw sub-scores.
Uses the corrected weights (sum=1.0) regardless of what weights the pipeline used.

Run after experiments/run_subset_eval.py completes:
    python experiments/recompute_comparison.py
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

RAW_PATH = Path("experiments/results/subset_15branch_raw.json")
OUT_PATH = Path("experiments/results/subset_comparison.json")

# Corrected 15-branch weights (sum = 1.0)
W15 = {
    "semantic": 0.13, "melodic": 0.05, "structural": 0.02,
    "artifact_diff": 0.01, "ssm": 0.01, "spectral_corr": 0.01,
    "rhythm": 0.19, "mert": 0.04, "stem_combined": 0.02,
    "cqt": 0.01, "qmax": 0.12, "clap_multiscale": 0.12,
    "panns": 0.17, "dmax": 0.04, "tonnetz": 0.06,
}

# 13-branch: zero out dmax & tonnetz, renormalize
W13 = {k: v for k, v in W15.items() if k not in ("dmax", "tonnetz")}
w13_sum = sum(W13.values())
W13 = {k: round(v / w13_sum, 6) for k, v in W13.items()}

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
    total = 0.0
    for wk, wv in weights.items():
        sk = SCORE_KEY_MAP[wk]
        total += wv * result_dict.get(sk, 0.5)
    return float(np.clip(total, 0.0, 1.0))


def summarize(pairs, name, threshold=0.72):
    pos = [p for p in pairs if p["true_label"] == 1]
    neg = [p for p in pairs if p["true_label"] == 0]
    mean_pos = np.mean([p["score"] for p in pos])
    mean_neg = np.mean([p["score"] for p in neg])
    predictions = [int(p["score"] >= threshold) for p in pairs]
    labels = [p["true_label"] for p in pairs]
    correct = sum(p == l for p, l in zip(predictions, labels))
    tp = sum(p == 1 and l == 1 for p, l in zip(predictions, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(predictions, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(predictions, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(predictions, labels))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "experiment": name,
        "accuracy": round(correct / len(pairs), 4),
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "threshold": threshold,
        "mean_score_pos": round(float(mean_pos), 4),
        "mean_score_neg": round(float(mean_neg), 4),
        "score_gap": round(float(mean_pos - mean_neg), 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def main():
    if not RAW_PATH.exists():
        print(f"Raw results not found at {RAW_PATH}. Run run_subset_eval.py first.")
        return

    with open(RAW_PATH) as f:
        raw = json.load(f)

    print(f"Loaded {len(raw)} pair results from {RAW_PATH}")
    print(f"W15 sum = {sum(W15.values()):.4f}")
    print(f"W13 sum = {sum(W13.values()):.4f}")

    pairs_13, pairs_15 = [], []
    for r in raw:
        s13 = recompute_score(r, W13)
        s15 = recompute_score(r, W15)
        pairs_13.append({"pair_id": r["pair_id"], "true_label": r["true_label"], "score": s13})
        pairs_15.append({"pair_id": r["pair_id"], "true_label": r["true_label"], "score": s15})

    sum13 = summarize(pairs_13, "baseline_13branch")
    sum15 = summarize(pairs_15, "improved_15branch")

    print(f"\n{'='*60}")
    print(f"  BASELINE (13-branch, no dmax/tonnetz)")
    print(f"{'='*60}")
    for k, v in sum13.items():
        print(f"  {k}: {v}")

    print(f"\n{'='*60}")
    print(f"  IMPROVED (15-branch, +dmax +tonnetz)")
    print(f"{'='*60}")
    for k, v in sum15.items():
        print(f"  {k}: {v}")

    improvement = sum15["score_gap"] - sum13["score_gap"]
    print(f"\n--- Improvement ---")
    print(f"  Score gap: {sum13['score_gap']:.4f} -> {sum15['score_gap']:.4f} ({improvement:+.4f})")
    print(f"  Accuracy:  {sum13['accuracy']:.4f} -> {sum15['accuracy']:.4f}")

    print(f"\n--- Per-Pair Comparison ---")
    print(f"  {'pair':<15} {'label':>5}  {'13br':>6}  {'15br':>6}  {'delta':>7}")
    for p13, p15 in zip(pairs_13, pairs_15):
        delta = p15["score"] - p13["score"]
        print(f"  {p13['pair_id']:<15} {p13['true_label']:>5}  {p13['score']:.4f}  {p15['score']:.4f}  {delta:+.4f}")

    # Sub-score analysis
    print(f"\n--- Sub-Score Means (pos vs neg) ---")
    pos_results = [r for r in raw if r["true_label"] == 1]
    neg_results = [r for r in raw if r["true_label"] == 0]
    for wk in sorted(W15.keys(), key=lambda k: W15[k], reverse=True):
        sk = SCORE_KEY_MAP[wk]
        pos_mean = np.mean([r.get(sk, 0.5) for r in pos_results])
        neg_mean = np.mean([r.get(sk, 0.5) for r in neg_results])
        gap = pos_mean - neg_mean
        print(f"  {wk:<20} w={W15[wk]:.2f}  pos={pos_mean:.4f}  neg={neg_mean:.4f}  gap={gap:+.4f}")

    comparison = {
        "baseline_13branch": sum13,
        "improved_15branch": sum15,
        "improvement_gap": round(improvement, 4),
        "weights_15branch": W15,
        "weights_13branch": W13,
        "per_pair": [
            {"pair_id": p13["pair_id"], "label": p13["true_label"],
             "score_13br": round(p13["score"], 4), "score_15br": round(p15["score"], 4),
             "delta": round(p15["score"] - p13["score"], 4)}
            for p13, p15 in zip(pairs_13, pairs_15)
        ],
    }
    with open(OUT_PATH, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
