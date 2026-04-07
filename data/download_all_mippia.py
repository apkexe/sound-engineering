"""
Download All MIPPIA Pairs
==========================
Downloads all 70 unique pairs from the SMP dataset using yt-dlp.
Saves as WAV files for maximum quality during analysis.

Usage:
    python data/download_all_mippia.py
    python data/download_all_mippia.py --max_pairs 10   # partial download
    python data/download_all_mippia.py --format mp3      # smaller files
"""

import argparse
import subprocess
import shutil
import sys
from pathlib import Path

import pandas as pd


def get_ytdlp_cmd():
    """Find yt-dlp executable or fall back to python -m yt_dlp."""
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    return [sys.executable, "-m", "yt_dlp"]


def download_audio(url: str, out_path: str, audio_format: str = "wav") -> bool:
    """Download a single audio file from YouTube."""
    cmd = get_ytdlp_cmd() + [
        "-x",
        "--audio-format", audio_format,
        "--audio-quality", "0",
        "--output", out_path,
        "--no-playlist",
        "--no-overwrites",
        "--quiet",
        "--no-warnings",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def main():
    parser = argparse.ArgumentParser(description="Download all MIPPIA pairs")
    parser.add_argument("--csv", default="data/smp_dataset/Final_dataset_pairs.csv")
    parser.add_argument("--out_dir", default="data/mippia_full")
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--format", default="wav", choices=["wav", "mp3"])
    parser.add_argument("--skip_existing", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    pairs = df.drop_duplicates("pair_number")[
        ["pair_number", "ori_title", "comp_title", "ori_link", "comp_link", "relation"]
    ].sort_values("pair_number")

    if args.max_pairs:
        pairs = pairs.head(args.max_pairs)

    total = len(pairs)
    print(f"Downloading {total} pairs to {out_dir}/")
    print(f"Format: {args.format}")
    print()

    success = 0
    failed_pairs = []

    for idx, (_, row) in enumerate(pairs.iterrows()):
        pid = int(row.pair_number)
        print(f"[{idx+1}/{total}] Pair {pid}: {row.ori_title[:50]} vs {row.comp_title[:50]}")

        pair_ok = True
        for url, suffix in [(row.ori_link, "original"), (row.comp_link, "similar")]:
            fname = f"{pid}_{suffix}.{args.format}"
            fpath = out_dir / fname

            if args.skip_existing and fpath.exists() and fpath.stat().st_size > 1000:
                print(f"  {suffix}: already exists, skipping")
                continue

            out_template = str(out_dir / f"{pid}_{suffix}.%(ext)s")
            ok = download_audio(url, out_template, args.format)
            if ok:
                # yt-dlp may produce the file already
                actual = out_dir / fname
                if actual.exists():
                    print(f"  {suffix}: OK ({actual.stat().st_size / 1024:.0f} KB)")
                else:
                    # Check for alternative extensions
                    found = list(out_dir.glob(f"{pid}_{suffix}.*"))
                    if found:
                        print(f"  {suffix}: OK as {found[0].name}")
                    else:
                        print(f"  {suffix}: WARN - file not found after download")
                        pair_ok = False
            else:
                print(f"  {suffix}: FAILED ({url})")
                pair_ok = False

        if pair_ok:
            success += 1
        else:
            failed_pairs.append(pid)

    print(f"\n{'='*50}")
    print(f"Download complete: {success}/{total} pairs successful")
    if failed_pairs:
        print(f"Failed pairs: {failed_pairs}")

    # Save a manifest
    manifest = out_dir / "manifest.txt"
    files = sorted(out_dir.glob(f"*.{args.format}"))
    manifest.write_text("\n".join(f.name for f in files))
    print(f"Manifest: {manifest} ({len(files)} files)")


if __name__ == "__main__":
    main()
