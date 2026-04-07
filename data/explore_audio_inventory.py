"""
Day 1 data exploration utility.

Scans an audio directory and produces:
1) file-level metadata CSV
2) pair-level summary for *_original/*_similar files
3) compact JSON report with dataset statistics

Usage:
  python data/explore_audio_inventory.py --audio_dir data/mippia --out_dir data/reports
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

SUPPORTED_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def _audio_files(audio_dir: Path):
    for p in audio_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def _probe(path: Path):
    try:
        y, sr = librosa.load(path, sr=22050, mono=True)
        duration = float(librosa.get_duration(y=y, sr=sr))
        rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        zcr = float(librosa.feature.zero_crossing_rate(y=y).mean()) if len(y) else 0.0
        return {
            "path": str(path),
            "name": path.name,
            "ext": path.suffix.lower(),
            "duration_sec": duration,
            "sr": sr,
            "rms": rms,
            "peak": peak,
            "zcr": zcr,
            "ok": True,
            "error": "",
        }
    except Exception as exc:
        return {
            "path": str(path),
            "name": path.name,
            "ext": path.suffix.lower(),
            "duration_sec": np.nan,
            "sr": np.nan,
            "rms": np.nan,
            "peak": np.nan,
            "zcr": np.nan,
            "ok": False,
            "error": str(exc),
        }


def _pair_id_and_role(filename: str):
    stem = Path(filename).stem
    if stem.endswith("_original"):
        return stem[: -len("_original")], "original"
    if stem.endswith("_similar"):
        return stem[: -len("_similar")], "similar"
    return None, None


def build_reports(audio_dir: str, out_dir: str):
    audio_dir_p = Path(audio_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    rows = [_probe(p) for p in _audio_files(audio_dir_p)]
    if not rows:
        raise ValueError(f"No audio files found in {audio_dir}")

    files_df = pd.DataFrame(rows)
    files_csv = out_dir_p / "audio_files_profile.csv"
    files_df.to_csv(files_csv, index=False)

    # Pair-level summary for MIPPIA naming convention.
    pairs = {}
    for _, row in files_df.iterrows():
        pair_id, role = _pair_id_and_role(row["name"])
        if pair_id is None:
            continue
        if pair_id not in pairs:
            pairs[pair_id] = {"pair_id": pair_id}
        pairs[pair_id][f"{role}_path"] = row["path"]
        pairs[pair_id][f"{role}_duration_sec"] = row["duration_sec"]

    pairs_df = pd.DataFrame(list(pairs.values()))
    if not pairs_df.empty:
        pairs_df["complete_pair"] = pairs_df[["original_path", "similar_path"]].notna().all(axis=1)
        pairs_df["duration_delta_sec"] = (
            pairs_df["similar_duration_sec"] - pairs_df["original_duration_sec"]
        ).abs()
    pairs_csv = out_dir_p / "mippia_pair_summary.csv"
    pairs_df.to_csv(pairs_csv, index=False)

    ok_df = files_df[files_df["ok"] == True]
    report = {
        "audio_dir": str(audio_dir_p),
        "num_files": int(len(files_df)),
        "num_ok": int(ok_df.shape[0]),
        "num_failed": int((files_df["ok"] == False).sum()),
        "duration_sec": {
            "mean": float(ok_df["duration_sec"].mean()) if not ok_df.empty else 0.0,
            "median": float(ok_df["duration_sec"].median()) if not ok_df.empty else 0.0,
            "p10": float(ok_df["duration_sec"].quantile(0.10)) if not ok_df.empty else 0.0,
            "p90": float(ok_df["duration_sec"].quantile(0.90)) if not ok_df.empty else 0.0,
        },
        "rms": {
            "mean": float(ok_df["rms"].mean()) if not ok_df.empty else 0.0,
            "median": float(ok_df["rms"].median()) if not ok_df.empty else 0.0,
        },
        "mippia_pairs": {
            "detected": int(len(pairs_df)),
            "complete": int(pairs_df.get("complete_pair", pd.Series([], dtype=bool)).sum()) if not pairs_df.empty else 0,
        },
        "outputs": {
            "file_profile_csv": str(files_csv),
            "pair_summary_csv": str(pairs_csv),
        },
    }

    report_json = out_dir_p / "day1_data_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore local audio inventory for Day 1")
    parser.add_argument("--audio_dir", required=True, help="Directory with audio files")
    parser.add_argument("--out_dir", default="data/reports", help="Output directory for Day 1 reports")
    args = parser.parse_args()

    build_reports(args.audio_dir, args.out_dir)
