"""
Train a calibration model for MAIA sub-scores.

Input can be either:
1) JSON output from evaluate.py (contains "pairs" list), or
2) CSV file with columns:
   semantic_similarity, melodic_alignment, structural_correspondence,
   ai_artifact_score, attribution_score, true_label

Usage:
  python train_calibration.py --input results.json --output models/calibrator.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.model.calibration import ScoreCalibrator


def _load_rows(input_path: str):
    p = Path(input_path)
    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("pairs", [])
        if not rows:
            raise ValueError("No 'pairs' found in JSON input.")
        return rows

    df = pd.read_csv(p)
    required = {
        "semantic_similarity",
        "melodic_alignment",
        "structural_correspondence",
        "ai_artifact_score",
        "attribution_score",
        "true_label",
    }
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required}")
    return df.to_dict(orient="records")


def main(input_path: str, output_path: str, c_value: float):
    rows = _load_rows(input_path)
    calibrator = ScoreCalibrator.fit(rows, c_value=c_value)

    probs = [calibrator.predict_proba(r) for r in rows]
    labels = [int(r["true_label"]) for r in rows]
    preds = [int(p >= 0.5) for p in probs]

    auc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds)

    calibrator.save(output_path)
    print(f"Saved calibration model to: {output_path}")
    print(f"Train ROC-AUC: {auc:.4f}")
    print(f"Train F1@0.5:  {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAIA score calibrator")
    parser.add_argument("--input", required=True, help="Path to evaluate.py JSON output or feature CSV")
    parser.add_argument("--output", default="models/calibrator.json", help="Output calibration model path")
    parser.add_argument("--c", type=float, default=1.0, help="Inverse regularization strength for logistic regression")
    args = parser.parse_args()

    main(args.input, args.output, args.c)
