"""
Robust validation and overfitting audit for MAIA result files.

What this script does:
1) Evaluates fixed-threshold baseline metrics.
2) Runs leakage-safe threshold tuning (nested CV on score only).
3) Runs leakage-safe regularized logistic calibration (nested CV on sub-scores).
4) Flags suspicious metrics (train-test gaps, extreme imbalance behavior).

Usage:
    python experiments/robust_validation.py --results_json results/mippia_full_results.json
    python experiments/robust_validation.py --train_json train.json --test_json test.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "semantic_similarity",
    "melodic_alignment",
    "structural_correspondence",
    "ai_artifact_score",
    "ssm_similarity",
    "spectral_correlation",
    "rhythm_similarity",
    "attribution_score",
]


def load_rows(path: str) -> List[Dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "pairs" not in payload:
        raise ValueError(f"Invalid results JSON format: {path}")
    return payload["pairs"]


def to_xy(rows: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.array([int(r["true_label"]) for r in rows], dtype=np.int32)
    raw = np.array([float(r.get("attribution_score", 0.5)) for r in rows], dtype=np.float64)
    x = np.array([[float(r.get(f, 0.5)) for f in FEATURES] for r in rows], dtype=np.float64)
    return x, y, raw


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str = "balanced_accuracy",
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pb = y_pred[idx]
        if metric_name == "accuracy":
            v = accuracy_score(yb, pb)
        elif metric_name == "f1":
            v = f1_score(yb, pb, zero_division=0)
        else:
            v = balanced_accuracy_score(yb, pb)
        vals.append(float(v))

    return {
        "metric": metric_name,
        "mean": float(np.mean(vals)),
        "ci95_low": float(np.percentile(vals, 2.5)),
        "ci95_high": float(np.percentile(vals, 97.5)),
        "ci95_width": float(np.percentile(vals, 97.5) - np.percentile(vals, 2.5)),
    }


def threshold_sweep(raw: np.ndarray, y: np.ndarray, threshold_grid: np.ndarray) -> Dict[str, float]:
    best_t = float(threshold_grid[0])
    best_ba = -1.0
    best_acc = 0.0
    best_f1 = 0.0

    for t in threshold_grid:
        p = (raw >= t).astype(np.int32)
        ba = float(balanced_accuracy_score(y, p))
        if ba > best_ba:
            best_ba = ba
            best_t = float(t)
            best_acc = float(accuracy_score(y, p))
            best_f1 = float(f1_score(y, p, zero_division=0))

    return {
        "best_threshold": best_t,
        "best_balanced_accuracy": best_ba,
        "best_accuracy": best_acc,
        "best_f1": best_f1,
    }


def repeated_holdout_threshold_audit(
    raw: np.ndarray,
    y: np.ndarray,
    threshold_grid: np.ndarray,
    n_runs: int = 200,
    test_size: float = 0.3,
) -> Dict[str, float]:
    train_bal = []
    test_bal = []
    train_f1 = []
    test_f1 = []
    tuned_thresholds = []

    for s in range(n_runs):
        tr, te = train_test_split(
            np.arange(len(y)), test_size=test_size, stratify=y, random_state=1337 + s
        )
        raw_tr, y_tr = raw[tr], y[tr]
        raw_te, y_te = raw[te], y[te]

        best_t = float(threshold_grid[0])
        best_ba = -1.0
        for t in threshold_grid:
            p_tr = (raw_tr >= t).astype(np.int32)
            ba_tr = float(balanced_accuracy_score(y_tr, p_tr))
            if ba_tr > best_ba:
                best_ba = ba_tr
                best_t = float(t)

        tuned_thresholds.append(best_t)

        p_tr = (raw_tr >= best_t).astype(np.int32)
        p_te = (raw_te >= best_t).astype(np.int32)

        train_bal.append(float(balanced_accuracy_score(y_tr, p_tr)))
        test_bal.append(float(balanced_accuracy_score(y_te, p_te)))
        train_f1.append(float(f1_score(y_tr, p_tr, zero_division=0)))
        test_f1.append(float(f1_score(y_te, p_te, zero_division=0)))

    train_bal_arr = np.array(train_bal, dtype=np.float64)
    test_bal_arr = np.array(test_bal, dtype=np.float64)
    train_f1_arr = np.array(train_f1, dtype=np.float64)
    test_f1_arr = np.array(test_f1, dtype=np.float64)
    tuned_t_arr = np.array(tuned_thresholds, dtype=np.float64)

    return {
        "n_runs": int(n_runs),
        "test_size": float(test_size),
        "train_balanced_accuracy_mean": float(train_bal_arr.mean()),
        "train_balanced_accuracy_std": float(train_bal_arr.std(ddof=1)),
        "test_balanced_accuracy_mean": float(test_bal_arr.mean()),
        "test_balanced_accuracy_std": float(test_bal_arr.std(ddof=1)),
        "train_test_balanced_accuracy_gap_mean": float((train_bal_arr - test_bal_arr).mean()),
        "train_f1_mean": float(train_f1_arr.mean()),
        "test_f1_mean": float(test_f1_arr.mean()),
        "train_test_f1_gap_mean": float((train_f1_arr - test_f1_arr).mean()),
        "tuned_threshold_mean": float(tuned_t_arr.mean()),
        "tuned_threshold_std": float(tuned_t_arr.std(ddof=1)),
        "tuned_threshold_p10": float(np.percentile(tuned_t_arr, 10)),
        "tuned_threshold_p90": float(np.percentile(tuned_t_arr, 90)),
    }


def nested_cv_threshold_only(raw: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, float]:
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pred = np.zeros_like(y)

    for tr, te in outer.split(raw, y):
        r_tr, y_tr = raw[tr], y[tr]
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

        best_t = 0.5
        best_score = -1.0
        for t in np.linspace(0.5, 0.95, 181):
            inner_scores = []
            for i_tr, i_va in inner.split(r_tr, y_tr):
                p = (r_tr[i_va] >= t).astype(np.int32)
                inner_scores.append(balanced_accuracy_score(y_tr[i_va], p))
            s = float(np.mean(inner_scores))
            if s > best_score:
                best_score = s
                best_t = float(t)

        pred[te] = (raw[te] >= best_t).astype(np.int32)

    out = metrics(y, pred)
    out["model"] = "nested_cv_threshold_only"
    return out


def nested_cv_logistic(x: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, float]:
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probs = np.zeros(len(y), dtype=np.float64)

    for tr, te in outer.split(x, y):
        x_tr, y_tr = x[tr], y[tr]
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

        best_c = 1.0
        best_score = -1.0
        for c in [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
            inner_scores = []
            for i_tr, i_va in inner.split(x_tr, y_tr):
                clf = Pipeline([
                    ("sc", StandardScaler()),
                    ("lr", LogisticRegression(C=c, max_iter=4000, class_weight="balanced", random_state=42)),
                ])
                clf.fit(x_tr[i_tr], y_tr[i_tr])
                p = clf.predict(x_tr[i_va])
                inner_scores.append(balanced_accuracy_score(y_tr[i_va], p))
            s = float(np.mean(inner_scores))
            if s > best_score:
                best_score = s
                best_c = float(c)

        clf = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=best_c, max_iter=4000, class_weight="balanced", random_state=42)),
        ])
        clf.fit(x_tr, y_tr)
        probs[te] = clf.predict_proba(x[te])[:, 1]

    pred = (probs >= 0.5).astype(np.int32)
    out = metrics(y, pred)
    out["model"] = "nested_cv_logistic"
    return out


def suspicious_warnings(train_m: Dict[str, float], test_m: Dict[str, float]) -> List[str]:
    warnings = []
    if train_m["accuracy"] >= 0.9 and test_m["accuracy"] <= 0.6:
        warnings.append("Potential overfitting: train accuracy >= 90% but test <= 60%.")
    if train_m["f1"] - test_m["f1"] >= 0.2:
        warnings.append("Potential overfitting: train-test F1 gap >= 0.20.")
    if test_m["recall"] >= 0.95 and test_m["precision"] <= 0.55:
        warnings.append("Suspicious operating point: very high recall with low precision (many false positives).")
    if test_m["precision"] >= 0.95 and test_m["recall"] <= 0.55:
        warnings.append("Suspicious operating point: very high precision with low recall (many false negatives).")
    return warnings


def suspicious_warnings_single_file(
    baseline_threshold: float,
    threshold_scan: Dict[str, float],
    holdout_audit: Dict[str, float],
    baseline_ci: Dict[str, float],
) -> List[str]:
    warnings = []

    if abs(threshold_scan["best_threshold"] - baseline_threshold) >= 0.05:
        warnings.append(
            "Threshold drift: best threshold is far from default; calibration may be unstable across datasets."
        )

    if holdout_audit["train_test_balanced_accuracy_gap_mean"] >= 0.08:
        warnings.append(
            "Potential overfitting: repeated train-test balanced-accuracy gap >= 0.08."
        )

    if holdout_audit["test_balanced_accuracy_std"] >= 0.06:
        warnings.append(
            "Potential instability: test balanced-accuracy variance is high across random splits."
        )

    if holdout_audit["tuned_threshold_std"] >= 0.035:
        warnings.append(
            "Threshold instability: tuned threshold variance is high; operating point may not be robust."
        )

    if baseline_ci["ci95_width"] >= 0.15:
        warnings.append(
            "High uncertainty: baseline metric confidence interval is wide; more data is recommended."
        )

    return warnings


def main():
    parser = argparse.ArgumentParser(description="Robust validation and overfitting audit")
    parser.add_argument("--results_json", default=None, help="Single JSON for nested CV analysis")
    parser.add_argument("--train_json", default=None, help="Train JSON for explicit train/test gap audit")
    parser.add_argument("--test_json", default=None, help="Test JSON for explicit train/test gap audit")
    parser.add_argument("--baseline_threshold", type=float, default=0.829, help="Baseline fixed threshold (Exp 6 optimum)")
    parser.add_argument("--output", default="experiments/results/robust_validation_report.json")
    parser.add_argument("--n_holdout_runs", type=int, default=200, help="Repeated holdout runs for stability audit")
    args = parser.parse_args()

    report = {}

    if args.results_json:
        rows = load_rows(args.results_json)
        x, y, raw = to_xy(rows)
        threshold_grid = np.linspace(max(0.5, args.baseline_threshold - 0.12), min(0.95, args.baseline_threshold + 0.12), 241)

        baseline_pred = (raw >= args.baseline_threshold).astype(np.int32)
        baseline = metrics(y, baseline_pred)
        baseline["model"] = f"fixed_threshold_{args.baseline_threshold}"
        baseline_ci = bootstrap_metric_ci(y, baseline_pred, metric_name="balanced_accuracy")

        thr_scan = threshold_sweep(raw, y, threshold_grid)
        holdout_audit = repeated_holdout_threshold_audit(
            raw=raw,
            y=y,
            threshold_grid=threshold_grid,
            n_runs=args.n_holdout_runs,
            test_size=0.3,
        )
        single_file_warnings = suspicious_warnings_single_file(
            baseline_threshold=args.baseline_threshold,
            threshold_scan=thr_scan,
            holdout_audit=holdout_audit,
            baseline_ci=baseline_ci,
        )

        thr_nested = nested_cv_threshold_only(raw, y)
        log_nested = nested_cv_logistic(x, y)

        report["single_file_analysis"] = {
            "input": args.results_json,
            "n_pairs": int(len(rows)),
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
            "baseline": baseline,
            "baseline_balanced_accuracy_ci95_bootstrap": baseline_ci,
            "threshold_sweep": thr_scan,
            "repeated_holdout_threshold_audit": holdout_audit,
            "nested_threshold_only": thr_nested,
            "nested_logistic": log_nested,
            "warnings": single_file_warnings,
        }

    if args.train_json and args.test_json:
        train_rows = load_rows(args.train_json)
        test_rows = load_rows(args.test_json)

        _, y_tr, raw_tr = to_xy(train_rows)
        _, y_te, raw_te = to_xy(test_rows)

        train_m = metrics(y_tr, (raw_tr >= args.baseline_threshold).astype(np.int32))
        test_m = metrics(y_te, (raw_te >= args.baseline_threshold).astype(np.int32))
        warns = suspicious_warnings(train_m, test_m)

        report["train_test_gap_analysis"] = {
            "train_input": args.train_json,
            "test_input": args.test_json,
            "baseline_threshold": args.baseline_threshold,
            "train_metrics": train_m,
            "test_metrics": test_m,
            "warnings": warns,
        }

    if not report:
        raise ValueError("Provide either --results_json or (--train_json and --test_json).")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved report: {out_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
