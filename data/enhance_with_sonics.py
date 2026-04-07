"""
SONICS enhancement utilities for MAIA.

This script supports two actions:
1) Download a small SONICS fake-song subset to local disk.
2) Augment an existing eval pairs CSV with SONICS-based hard negatives.

Why this exists:
- MIPPIA is ideal for attribution labels.
- SONICS adds diverse AI-generated negatives for robustness checks.

Examples:
  # Download 200 SONICS fake tracks
  python data/enhance_with_sonics.py download --out_dir data/sonics_subset --num_samples 200

  # Augment existing MIPPIA eval pairs with SONICS negatives
  python data/enhance_with_sonics.py augment \
    --base_pairs_csv data/reports/eval_pairs.csv \
    --sonics_csv data/sonics_subset/sonics_subset.csv \
    --out_csv data/reports/eval_pairs_sonics.csv \
    --neg_ratio 0.5
"""

import argparse
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download


SONICS_REPO = "awsaf49/sonics"


def download_sonics_subset(
    out_dir: str,
    num_samples: int = 500,
    seed: int = 42,
    split: str = "train",
    only_fake: bool = True,
):
    rng = random.Random(seed)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(SONICS_REPO, split=split, streaming=True)

    selected = []
    seen_ids = set()

    for row in ds:
        target = row.get("target", None)
        if only_fake and target != 1:
            continue

        row_id = row.get("id", None)
        filepath = row.get("filepath", None)
        if row_id is None or not filepath:
            continue
        if row_id in seen_ids:
            continue

        seen_ids.add(row_id)
        selected.append(
            {
                "id": int(row_id),
                "filepath": filepath,
                "duration": row.get("duration", None),
                "source": row.get("source", ""),
                "algorithm": row.get("algorithm", ""),
                "label": row.get("label", ""),
                "target": int(row.get("target", 0)),
                "split": row.get("split", split),
            }
        )

        if len(selected) >= num_samples:
            break

    if not selected:
        raise ValueError("No SONICS rows selected. Check connectivity and dataset filters.")

    # Shuffle selected rows for source diversity.
    rng.shuffle(selected)

    local_rows = []
    ok_downloads = 0
    for i, row in enumerate(selected, start=1):
        rel = row["filepath"]
        row["local_path"] = ""
        try:
            cached_path = hf_hub_download(
                repo_id=SONICS_REPO,
                repo_type="dataset",
                filename=rel,
            )
            src = Path(cached_path)
            local_name = f"sonics_{row['id']}_{src.name}"
            dst = out_dir_p / local_name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())

            row["local_path"] = str(dst)
            ok_downloads += 1
        except Exception as exc:
            print(f"[WARN] Could not download {rel}: {exc}")

        local_rows.append(row)

        if i % 25 == 0:
            print(f"Downloaded {i}/{len(selected)} SONICS tracks...")

    if not local_rows:
        raise ValueError("No SONICS rows were selected. Could not build SONICS manifest.")

    out_csv = out_dir_p / "sonics_subset.csv"
    pd.DataFrame(local_rows).to_csv(out_csv, index=False)
    print(f"Saved SONICS subset CSV: {out_csv}")
    print(f"Selected rows: {len(local_rows)}")
    print(f"Downloaded files: {ok_downloads}")
    if ok_downloads == 0:
        print("[INFO] No files were directly downloadable via repo paths. ")
        print("[INFO] You can still use this CSV as a manifest and map filepath via --sonics_root in augment mode.")


def augment_pairs_with_sonics(
    base_pairs_csv: str,
    sonics_csv: str,
    out_csv: str,
    sonics_root: str = None,
    neg_ratio: float = 0.5,
    seed: int = 42,
):
    rng = random.Random(seed)

    base_df = pd.read_csv(base_pairs_csv)
    required_cols = {"track_a", "track_b", "label"}
    if not required_cols.issubset(base_df.columns):
        raise ValueError(f"Base pairs CSV missing required columns: {required_cols}")

    sonics_df = pd.read_csv(sonics_csv)
    if "local_path" not in sonics_df.columns:
        raise ValueError("SONICS CSV must contain 'local_path'.")

    positives = base_df[base_df["label"] == 1]
    if positives.empty:
        raise ValueError("Base pairs CSV has no positive rows to anchor SONICS negatives.")

    sonics_paths = []
    root = Path(sonics_root) if sonics_root else None
    for _, row in sonics_df.iterrows():
        local_path = str(row.get("local_path", "") or "").strip()
        if local_path and Path(local_path).exists():
            sonics_paths.append(local_path)
            continue

        if root is not None:
            rel = str(row.get("filepath", "") or "").strip()
            if rel:
                candidate = root / rel
                if candidate.exists():
                    sonics_paths.append(str(candidate))

    if not sonics_paths:
        raise ValueError(
            "No valid SONICS audio paths found. Provide downloaded local_path entries "
            "or pass --sonics_root pointing to a local SONICS mirror."
        )

    n_pos = len(positives)
    n_new_neg = max(1, int(round(n_pos * neg_ratio)))

    rows = []
    for _ in range(n_new_neg):
        base_row = positives.sample(n=1, random_state=rng.randint(0, 10_000_000)).iloc[0]
        sonics_path = rng.choice(sonics_paths)

        rows.append(
            {
                "track_a": base_row["track_a"],
                "track_b": sonics_path,
                "label": 0,
                "pair_type": "negative_sonics",
            }
        )

    aug_df = pd.concat([base_df, pd.DataFrame(rows)], ignore_index=True)
    aug_df = aug_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    aug_df.to_csv(out_csv, index=False)

    print(f"Saved augmented pairs: {out_csv}")
    print(f"Base rows: {len(base_df)} | Added SONICS negatives: {len(rows)} | Total: {len(aug_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SONICS enhancement for MAIA")
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download", help="Download SONICS subset")
    p_download.add_argument("--out_dir", default="data/sonics_subset", help="Output directory")
    p_download.add_argument("--num_samples", type=int, default=500, help="Number of SONICS tracks")
    p_download.add_argument("--seed", type=int, default=42)
    p_download.add_argument("--split", default="train")
    p_download.add_argument("--include_real", action="store_true", help="Include non-fake rows")

    p_aug = sub.add_parser("augment", help="Augment eval pairs with SONICS negatives")
    p_aug.add_argument("--base_pairs_csv", required=True)
    p_aug.add_argument("--sonics_csv", required=True)
    p_aug.add_argument("--out_csv", required=True)
    p_aug.add_argument("--sonics_root", default=None, help="Optional local root to resolve SONICS filepath entries")
    p_aug.add_argument("--neg_ratio", type=float, default=0.5)
    p_aug.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "download":
        download_sonics_subset(
            out_dir=args.out_dir,
            num_samples=args.num_samples,
            seed=args.seed,
            split=args.split,
            only_fake=not args.include_real,
        )
    elif args.command == "augment":
        augment_pairs_with_sonics(
            base_pairs_csv=args.base_pairs_csv,
            sonics_csv=args.sonics_csv,
            out_csv=args.out_csv,
            sonics_root=args.sonics_root,
            neg_ratio=args.neg_ratio,
            seed=args.seed,
        )
