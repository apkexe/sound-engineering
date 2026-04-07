"""
Experiment runner for comparing pipeline configurations.
Runs evaluation on the 10-pair subset with different settings and saves results.

Usage:
    python experiments/run_experiment.py --name baseline
    python experiments/run_experiment.py --name hpcp --config hpcp
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import evaluate


def run_experiment(name: str, subset_csv: str, threshold: float = 0.72):
    """Run evaluation and save structured results."""
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'='*60}")

    t0 = time.time()
    results_df = evaluate(
        pairs_csv=subset_csv,
        output_path=str(out_path),
        threshold=threshold,
    )
    elapsed = time.time() - t0

    # Read back to get metrics
    with open(out_path) as f:
        data = json.load(f)

    summary = {
        "experiment": name,
        "accuracy": data["accuracy"],
        "precision": data["precision"],
        "recall": data["recall"],
        "f1": data["f1"],
        "threshold": threshold,
        "elapsed_seconds": round(elapsed, 1),
        "mean_score_pos": round(
            sum(p["attribution_score"] for p in data["pairs"] if p["true_label"] == 1)
            / max(sum(1 for p in data["pairs"] if p["true_label"] == 1), 1), 4),
        "mean_score_neg": round(
            sum(p["attribution_score"] for p in data["pairs"] if p["true_label"] == 0)
            / max(sum(1 for p in data["pairs"] if p["true_label"] == 0), 1), 4),
    }
    summary["score_gap"] = round(summary["mean_score_pos"] - summary["mean_score_neg"], 4)

    summary_path = out_dir / f"{name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n--- {name} Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  Saved to: {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--subset", default="experiments/subset_10pairs.csv")
    parser.add_argument("--threshold", type=float, default=0.72)
    args = parser.parse_args()

    run_experiment(args.name, args.subset, args.threshold)


if __name__ == "__main__":
    main()
