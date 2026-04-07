"""
Extended pipeline that adds experimental features from GitHub repo research.
Wraps the existing compare_tracks() and adds new scoring branches.

New techniques tested:
  1. Dmax (ChromaCoverId/Serra09) — cumulative cross-recurrence score
  2. CENS chroma similarity (da-tacos/acoss) — smoothed chroma comparison
  3. Tonnetz harmonic similarity (da-tacos) — 6-D tonal space comparison
  4. Spectral contrast similarity (da-tacos) — sub-band peak/valley comparison
  5. Spectral flux onset similarity (madmom/acoss) — enhanced onset detection
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np

from src.features.sota_features import (
    dmax_similarity,
    cens_chroma_similarity,
    tonnetz_similarity,
    spectral_contrast_similarity,
    spectral_flux_onset_similarity,
)


def compute_experimental_features(track_a: str, track_b: str) -> dict:
    """Compute the 5 experimental features for a pair of tracks."""
    sr = 22050
    y_a, _ = librosa.load(track_a, sr=sr, mono=True)
    y_b, _ = librosa.load(track_b, sr=sr, mono=True)

    results = {}

    # 1. Dmax
    try:
        dmax_res = dmax_similarity(y_a, sr, y_b, sr)
        results["dmax_score"] = round(dmax_res["dmax_score"], 4)
    except Exception as e:
        results["dmax_score"] = 0.5
        results["dmax_error"] = str(e)

    # 2. CENS chroma
    try:
        results["cens_similarity"] = round(cens_chroma_similarity(y_a, sr, y_b, sr), 4)
    except Exception as e:
        results["cens_similarity"] = 0.5
        results["cens_error"] = str(e)

    # 3. Tonnetz
    try:
        results["tonnetz_similarity"] = round(tonnetz_similarity(y_a, sr, y_b, sr), 4)
    except Exception as e:
        results["tonnetz_similarity"] = 0.5
        results["tonnetz_error"] = str(e)

    # 4. Spectral contrast
    try:
        results["spectral_contrast"] = round(spectral_contrast_similarity(y_a, sr, y_b, sr), 4)
    except Exception as e:
        results["spectral_contrast"] = 0.5
        results["spectral_contrast_error"] = str(e)

    # 5. Spectral flux onset
    try:
        results["spectral_flux_onset"] = round(spectral_flux_onset_similarity(y_a, sr, y_b, sr), 4)
    except Exception as e:
        results["spectral_flux_onset"] = 0.5
        results["spectral_flux_error"] = str(e)

    return results


def evaluate_experimental(subset_csv: str, output_path: str):
    """Run experimental features on the subset and save results."""
    import pandas as pd

    df = pd.read_csv(subset_csv)
    all_results = []

    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] {Path(row.track_a).name} vs {Path(row.track_b).name}")
        feats = compute_experimental_features(row.track_a, row.track_b)
        feats["track_a"] = row.track_a
        feats["track_b"] = row.track_b
        feats["true_label"] = int(row.label)
        all_results.append(feats)

    # Compute mean positive/negative for each feature
    feature_names = ["dmax_score", "cens_similarity", "tonnetz_similarity",
                     "spectral_contrast", "spectral_flux_onset"]

    print("\n" + "="*60)
    print("EXPERIMENTAL FEATURES — Discrimination Analysis")
    print("="*60)
    print(f"{'Feature':<25} {'Pos Mean':>10} {'Neg Mean':>10} {'Gap':>10} {'Signal':>10}")
    print("-"*65)

    improvements = {}
    for feat in feature_names:
        pos_vals = [r[feat] for r in all_results if r["true_label"] == 1]
        neg_vals = [r[feat] for r in all_results if r["true_label"] == 0]
        pos_mean = np.mean(pos_vals)
        neg_mean = np.mean(neg_vals)
        gap = pos_mean - neg_mean
        signal = "STRONG" if gap > 0.03 else "Moderate" if gap > 0.01 else "Weak" if gap > 0 else "NONE"
        print(f"{feat:<25} {pos_mean:>10.4f} {neg_mean:>10.4f} {gap:>+10.4f} {signal:>10}")
        improvements[feat] = {"pos_mean": round(pos_mean, 4), "neg_mean": round(neg_mean, 4),
                              "gap": round(gap, 4), "signal": signal}

    output = {"pairs": all_results, "discrimination": improvements}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return improvements


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="experiments/subset_10pairs.csv")
    parser.add_argument("--output", default="experiments/results/experimental_features.json")
    args = parser.parse_args()
    evaluate_experimental(args.subset, args.output)
