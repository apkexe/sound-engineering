"""
Train GBDT or Logistic Regression Calibrator on MAIA Sub-Scores
================================================================
Uses evaluation results to learn an optimal decision boundary
from the 13 sub-component scores.

Usage:
    python train_calibrator.py --input results/exp5_all_improvements.json --method gbdt
    python train_calibrator.py --input results/exp5_all_improvements.json --method logistic
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score

from src.model.calibration import ScoreCalibrator, FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/mippia_full_results.json")
    parser.add_argument("--output", default=None,
                        help="Output path (default: models/calibrator_<method>.pkl or .json)")
    parser.add_argument("--method", choices=["gbdt", "logistic"], default="gbdt")
    args = parser.parse_args()

    if args.output is None:
        ext = ".pkl" if args.method == "gbdt" else ".json"
        args.output = f"models/calibrator_{args.method}{ext}"

    with open(args.input) as f:
        data = json.load(f)

    # Support both "pairs" (old format) and "results" (new format) keys
    pairs = data.get("pairs", data.get("results", []))
    print(f"Training on {len(pairs)} pairs")
    print(f"Method: {args.method}")
    print(f"Features ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}")

    # Show label distribution
    labels = [p["true_label"] for p in pairs]
    print(f"  Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")

    # Build rows with feature names matching FEATURE_NAMES
    rows = []
    for p in pairs:
        row = {name: float(p.get(name, 0.5)) for name in FEATURE_NAMES}
        row["true_label"] = p["true_label"]
        rows.append(row)

    # Train calibrator
    calibrator = ScoreCalibrator.fit(rows, method=args.method)

    # Cross-validation
    X = np.array([ScoreCalibrator._feature_vector(r) for r in rows])
    y = np.array([r["true_label"] for r in rows])
    n_per_class = min(sum(y == 0), sum(y == 1))
    cv_folds = min(5, n_per_class)
    if cv_folds >= 2:
        cv_acc = cross_val_score(calibrator.model, X, y, cv=cv_folds, scoring="accuracy")
        cv_f1 = cross_val_score(calibrator.model, X, y, cv=cv_folds, scoring="f1")
        print(f"\n{cv_folds}-fold CV accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
        print(f"{cv_folds}-fold CV F1:       {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    else:
        print(f"\nToo few samples per class ({n_per_class}) for cross-validation")

    # Show feature importances / coefficients
    if args.method == "gbdt":
        importances = calibrator.model.feature_importances_
        print(f"\nGBDT feature importances:")
        for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
            print(f"  {name:30s}: {imp:.4f}")
    else:
        coef = calibrator.model.coef_[0]
        print(f"\nLogistic regression coefficients:")
        for name, c in zip(FEATURE_NAMES, coef):
            print(f"  {name:30s}: {c:+.4f}")
        print(f"  {'intercept':30s}: {calibrator.model.intercept_[0]:+.4f}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(args.output)
    print(f"\nCalibrator saved to {args.output}")

    # Quick test: predict on training data
    correct_heur = 0
    correct_cal = 0
    for r, p in zip(rows, pairs):
        cal_score = calibrator.predict_proba(r)
        heur_score = p.get("attribution_score", 0.5)
        label = r["true_label"]
        correct_cal += (int(cal_score >= 0.5) == label)
        correct_heur += (int(heur_score >= 0.5) == label)

    print(f"\nHeuristic train accuracy:   {correct_heur}/{len(rows)} = {correct_heur/len(rows)*100:.1f}%")
    print(f"Calibrated train accuracy:  {correct_cal}/{len(rows)} = {correct_cal/len(rows)*100:.1f}%")


if __name__ == "__main__":
    main()
