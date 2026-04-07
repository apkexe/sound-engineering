"""
Batch Evaluation Script
========================
Evaluates MAIA on a set of labeled pairs and prints a full report.

Usage:
    python evaluate.py --pairs_csv data/eval_pairs.csv --output results.json

CSV format (eval_pairs.csv):
    track_a,track_b,label
    data/mippia/001_original.mp3,data/mippia/001_similar.mp3,1
    data/mippia/001_original.mp3,data/mippia/002_original.mp3,0
    ...

label: 1 = Track B is attributed to Track A, 0 = unrelated
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import compare_tracks
from src.model.calibration import ScoreCalibrator


def evaluate(
    pairs_csv: str,
    output_path: str = None,
    threshold: float = 0.829,
    calibration_model: str = None,
):
    df = pd.read_csv(pairs_csv)
    required_cols = {"track_a", "track_b", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required_cols}")

    calibrator = ScoreCalibrator.load(calibration_model) if calibration_model else None

    results = []
    skipped = 0
    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] {Path(row.track_a).name} vs {Path(row.track_b).name}")
        try:
            result = compare_tracks(row.track_a, row.track_b)
        except Exception as e:
            print(f"  ERROR: {e} — skipping pair")
            skipped += 1
            continue
        result["track_a"] = row.track_a
        result["track_b"] = row.track_b
        result["true_label"] = int(row.label)
        if calibrator:
            result["calibrated_score"] = round(calibrator.predict_proba(result), 4)
            decision_score = result["calibrated_score"]
        else:
            decision_score = result["attribution_score"]

        result["predicted"] = int(decision_score >= threshold)
        result["correct"] = result["predicted"] == result["true_label"]
        results.append(result)

    if skipped:
        print(f"\nSkipped {skipped} pairs due to errors")

    results_df = pd.DataFrame(results)

    # --- Metrics ---
    accuracy = results_df["correct"].mean()
    positives = results_df[results_df["true_label"] == 1]
    negatives = results_df[results_df["true_label"] == 0]

    tp = ((results_df["true_label"] == 1) & (results_df["predicted"] == 1)).sum()
    fp = ((results_df["true_label"] == 0) & (results_df["predicted"] == 1)).sum()
    fn = ((results_df["true_label"] == 1) & (results_df["predicted"] == 0)).sum()
    tn = ((results_df["true_label"] == 0) & (results_df["predicted"] == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("\n" + "=" * 50)
    print("MAIA Evaluation Report")
    print("=" * 50)
    print(f"Total pairs    : {len(results_df)}")
    print(f"Threshold      : {threshold}")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"\nTP={tp}  FP={fp}  FN={fn}  TN={tn}")
    if calibrator:
        print("Calibration    : enabled")
    print(f"\nMean score (attributed)  : {positives['attribution_score'].mean():.4f}")
    print(f"Mean score (unrelated)   : {negatives['attribution_score'].mean():.4f}")
    if calibrator:
        print(f"Mean calibrated (attr)   : {positives['calibrated_score'].mean():.4f}")
        print(f"Mean calibrated (unrel)  : {negatives['calibrated_score'].mean():.4f}")
    print(f"\nSub-component means:")
    for col in [
        "semantic_similarity", "melodic_alignment", "structural_correspondence",
        "ai_artifact_score", "artifact_diff", "ssm_similarity", "spectral_correlation",
        "rhythm_similarity", "mert_similarity", "stem_combined",
        "vocal_similarity", "drum_similarity", "bass_similarity", "other_similarity",
        "cqt_similarity", "qmax_score", "clap_multiscale", "panns_similarity",
        "dmax_score", "tonnetz_similarity",
    ]:
        if col in positives.columns:
            print(f"  {col}: attributed={positives[col].mean():.4f}  unrelated={negatives[col].mean():.4f}")

    # Save
    if output_path:
        output = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "threshold": threshold,
            "calibration_model": calibration_model,
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
            "pairs": results,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MAIA on labeled pairs")
    parser.add_argument("--pairs_csv", required=True, help="CSV with track_a, track_b, label")
    parser.add_argument("--output", default="results.json", help="Output JSON path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.829,
        help="Decision threshold (0.829 is the Exp 6 optimum — 85%% accuracy, zero false positives)",
    )
    parser.add_argument("--calibration_model", default=None, help="Optional calibration model JSON")
    args = parser.parse_args()
    evaluate(args.pairs_csv, args.output, args.threshold, args.calibration_model)
