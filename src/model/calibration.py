"""
Calibration layer for MAIA attribution scores.

Supports two calibrator modes:
  - LogisticRegression (default, for small datasets)
  - GradientBoosting (GBDT, for larger datasets with non-linear interactions)

Trains on all sub-scores from the 13-branch attribution pipeline.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


# All 13 sub-score features the pipeline produces
FEATURE_NAMES = [
    "semantic_similarity",
    "melodic_alignment",
    "structural_correspondence",
    "artifact_diff",
    "ssm_similarity",
    "spectral_correlation",
    "rhythm_similarity",
    "mert_similarity",
    "stem_combined",
    "cqt_similarity",
    "qmax_score",
    "clap_multiscale",
    "panns_similarity",
]


@dataclass
class ScoreCalibrator:
    model: object  # LogisticRegression or GradientBoostingClassifier
    model_type: str = "logistic"

    @staticmethod
    def _feature_vector(row: Dict) -> List[float]:
        return [float(row.get(name, 0.5)) for name in FEATURE_NAMES]

    @classmethod
    def fit(cls, rows: List[Dict], method: str = "gbdt",
            c_value: float = 1.0, random_state: int = 42):
        if not rows:
            raise ValueError("No training rows provided for calibration.")

        x = np.array([cls._feature_vector(r) for r in rows], dtype=np.float64)
        y = np.array([int(r["true_label"]) for r in rows], dtype=np.int32)

        if len(np.unique(y)) < 2:
            raise ValueError("Calibration requires both positive and negative labels.")

        if method == "gbdt":
            model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=3,
                subsample=0.8,
                random_state=random_state,
            )
            model.fit(x, y)
            return cls(model=model, model_type="gbdt")
        else:
            model = LogisticRegression(C=c_value, max_iter=1000,
                                       random_state=random_state)
            model.fit(x, y)
            return cls(model=model, model_type="logistic")

    def predict_proba(self, row: Dict) -> float:
        x = np.array([self._feature_vector(row)], dtype=np.float64)
        return float(self.model.predict_proba(x)[0, 1])

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.model_type == "gbdt":
            import pickle
            with open(path, "wb") as f:
                pickle.dump({"model_type": "gbdt", "model": self.model,
                             "feature_names": FEATURE_NAMES}, f)
        else:
            out = {
                "model_type": "logistic",
                "feature_names": FEATURE_NAMES,
                "coef": self.model.coef_[0].tolist(),
                "intercept": float(self.model.intercept_[0]),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)

    @classmethod
    def load(cls, path: str):
        p = Path(path)
        if p.suffix == ".pkl" or p.suffix == ".pickle":
            import pickle
            with open(path, "rb") as f:
                payload = pickle.load(f)
            return cls(model=payload["model"], model_type="gbdt")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if payload.get("model_type") == "gbdt":
            raise ValueError("GBDT models must be saved as .pkl files")

        coef = np.array(payload["coef"], dtype=np.float64)
        intercept = float(payload["intercept"])

        model = LogisticRegression()
        model.classes_ = np.array([0, 1], dtype=np.int32)
        model.coef_ = coef.reshape(1, -1)
        model.intercept_ = np.array([intercept], dtype=np.float64)
        model.n_features_in_ = coef.shape[0]

        return cls(model=model, model_type="logistic")
