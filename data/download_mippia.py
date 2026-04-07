"""
MIPPIA/SMP Dataset Downloader
==============================
Downloads the Similar Music Pairs dataset for training and evaluation.

The MIPPIA dataset provides original high-quality audio via YouTube URLs.
Required: yt-dlp, pandas, ffmpeg

Usage:
    python data/download_mippia.py --out_dir data/mippia --max_pairs 100

This script is adapted from the official MIPPIA download.py:
    https://github.com/Mippia/smp_dataset
"""

import argparse
import subprocess
import json
from pathlib import Path
import shutil

import pandas as pd


def download_pair(row: dict, out_dir: Path) -> bool:
    """
    Download a single audio pair using yt-dlp.
    Saves original and similar track as {id}_original.mp3 and {id}_similar.mp3.
    """
    pair_id = str(row.get("id", "unknown"))
    url_orig = row.get("original_url", "")
    url_sim = row.get("similar_url", "")

    success = True
    ytdlp_cmd = ["yt-dlp"] if shutil.which("yt-dlp") else ["python", "-m", "yt_dlp"]
    for url, suffix in [(url_orig, "original"), (url_sim, "similar")]:
        if not url:
            continue
        out_path = out_dir / f"{pair_id}_{suffix}.%(ext)s"
        cmd = ytdlp_cmd + [
            "-x",                          # extract audio
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "--output", str(out_path),
            "--no-playlist",
            "--quiet",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f"  [WARN] Failed to download {suffix} for pair {pair_id}: {url}")
            success = False

    return success


def main(metadata_csv: str, out_dir: str, max_pairs: int = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv)
    print(f"Loaded {len(df)} pairs from {metadata_csv}")

    if max_pairs:
        df = df.head(max_pairs)
        print(f"Limiting to {max_pairs} pairs")

    success_count = 0
    for _, row in df.iterrows():
        ok = download_pair(row.to_dict(), out_dir)
        if ok:
            success_count += 1

    print(f"\nDownloaded {success_count}/{len(df)} pairs to {out_dir}/")
    print("You can now use these files with compare_tracks()")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MIPPIA/SMP audio pairs")
    parser.add_argument("--metadata", default="data/mippia_metadata.csv",
                        help="Path to the MIPPIA metadata CSV")
    parser.add_argument("--out_dir", default="data/mippia",
                        help="Output directory for downloaded audio")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Limit number of pairs to download (for testing)")
    args = parser.parse_args()
    main(args.metadata, args.out_dir, args.max_pairs)
