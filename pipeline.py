"""
MAIA — Multi-scale Attribution Intelligence Architecture
=========================================================
Main entry point for AI-generated song attribution scoring.

Usage:
    from pipeline import compare_tracks
    score = compare_tracks("path/to/original.mp3", "path/to/candidate.mp3")
    print(score)  # float in [0, 1], higher = more likely an AI attribution of original

CLI:
    python pipeline.py --track_a original.mp3 --track_b candidate.mp3

SOTA References:
    - Rahman et al. "SONICS" (ICLR 2025, arXiv:2408.14080) — long-range SSM,
      rhythmic/pitch/dynamic artifact detection
    - Comanducci et al. "FakeMusicCaps" (arXiv:2409.10684) — mel-spectrogram
      cross-correlation
    - Wu et al. CLAP (ICASSP 2023) — semantic audio embeddings
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np

from src.features.temporal_sampler import MultiScaleTemporalSampler
from src.features.spectral import SpectralFeatureExtractor
from src.features.embeddings import CLAPEmbedder
from src.features.mert_embeddings import MERTEmbedder
from src.features.artifacts import AIArtifactDetector
from src.features.sota_features import (
    compare_ssm,
    mel_cross_correlation,
    rhythm_similarity,
    cqt_chroma_oti_similarity,
    qmax_similarity,
)
from src.features.source_separation import separate_track, stem_similarity
from src.features.panns_embeddings import panns_similarity
from src.model.attribution import AttributionScorer

# Module-level model caches for expensive models
_clap_instance = None
_mert_instance = None

def _get_clap():
    global _clap_instance
    if _clap_instance is None:
        _clap_instance = CLAPEmbedder()
    return _clap_instance

def _get_mert():
    global _mert_instance
    if _mert_instance is None:
        _mert_instance = MERTEmbedder()
    return _mert_instance


def compare_tracks(track_a: str, track_b: str, verbose: bool = False) -> dict:
    """
    Compare two audio tracks and return an attribution score.

    Parameters
    ----------
    track_a : str
        Path to the original (or reference) audio file.
    track_b : str
        Path to the candidate (potentially AI-generated) audio file.
    verbose : bool
        If True, print intermediate scores.

    Returns
    -------
    dict with keys:
        - 'attribution_score' (float, 0–1): Overall likelihood Track B is derived from Track A
        - 'semantic_similarity' (float, 0–1): CLAP embedding cosine similarity
        - 'melodic_alignment' (float, 0–1): DTW chroma alignment score
        - 'structural_correspondence' (float, 0–1): Section boundary alignment
        - 'ai_artifact_score' (float, 0–1): Strength of AI generation artifacts in Track B
        - 'verdict' (str): Human-readable verdict
    """
    sampler = MultiScaleTemporalSampler()
    spectral = SpectralFeatureExtractor()
    clap = _get_clap()
    mert = _get_mert()
    artifact_detector = AIArtifactDetector()
    scorer = AttributionScorer()

    # Step 1: Multi-scale temporal sampling
    windows_a = sampler.sample(track_a)
    windows_b = sampler.sample(track_b)

    if verbose:
        print(f"[Track A] Detected {len(windows_a)} structural windows")
        print(f"[Track B] Detected {len(windows_b)} structural windows")

    # Step 2: Feature extraction per window
    feats_a = spectral.extract_batch(windows_a)
    feats_b = spectral.extract_batch(windows_b)

    # Step 3: CLAP semantic embeddings (full-track aware)
    emb_a = clap.embed_windows(windows_a)
    emb_b = clap.embed_windows(windows_b)

    # Step 3b: Multi-scale CLAP temporal alignment
    if verbose:
        print("[CLAP] Computing multi-scale temporal alignment…")
    clap_multiscale = clap.multi_scale_similarity(windows_a, windows_b)

    # Step 4: AI artifact analysis on BOTH tracks
    artifact_score_a = artifact_detector.score(windows_a)
    artifact_score = artifact_detector.score(windows_b)

    # Step 5: SOTA-inspired full-track features
    # Load full audio for cross-track comparisons (SSM, spectro, rhythm)
    sr = 22050
    y_a, _ = librosa.load(track_a, sr=sr, mono=True)
    y_b, _ = librosa.load(track_b, sr=sr, mono=True)

    if verbose:
        print("[SOTA] Computing self-similarity matrix comparison…")
    ssm_sim = compare_ssm(y_a, sr, y_b, sr, feature="chroma")

    if verbose:
        print("[SOTA] Computing mel-spectrogram cross-correlation…")
    spec_corr = mel_cross_correlation(y_a, sr, y_b, sr)

    if verbose:
        print("[SOTA] Computing rhythm similarity…")
    rhythm_sim = rhythm_similarity(y_a, sr, y_b, sr)

    if verbose:
        print("[SOTA] Computing CQT chroma + OTI similarity…")
    cqt_result = cqt_chroma_oti_similarity(y_a, sr, y_b, sr)

    if verbose:
        print("[SOTA] Computing Qmax / cross-recurrence plot…")
    qmax_result = qmax_similarity(y_a, sr, y_b, sr)


    # Step 6b: MERT music understanding similarity
    if verbose:
        print("[MERT] Computing music embedding similarity…")
    mert_sim = mert.similarity(y_a, sr, y_b, sr)

    # Step 6c: Source separation + per-stem comparison
    import os
    if os.getenv("MAIA_SKIP_SRCSEP", "0") == "1":
        stem_scores = {"stem_combined": 0.5, "vocal_similarity": 0.5,
                       "drum_similarity": 0.5, "bass_similarity": 0.5,
                       "other_similarity": 0.5}
        if verbose:
            print("[SrcSep] SKIPPED (MAIA_SKIP_SRCSEP=1)")
    else:
        if verbose:
            print("[SrcSep] Separating tracks into stems (vocals/drums/bass/other)…")
        stems_a = separate_track(track_a, sr=sr)
        stems_b = separate_track(track_b, sr=sr)
        stem_scores = stem_similarity(stems_a, stems_b, sr=sr)
        if verbose:
            for k, v in stem_scores.items():
                print(f"  {k}: {v:.4f}")

    # Step 6d: PANNs perceptual embedding similarity
    if verbose:
        print("[PANNs] Computing perceptual embedding similarity…")
    panns_sim = panns_similarity(y_a, sr, y_b, sr)

    # Step 7: Attribution scoring (13 branches)
    result = scorer.score(
        feats_a=feats_a,
        feats_b=feats_b,
        emb_a=emb_a,
        emb_b=emb_b,
        artifact_score=artifact_score,
        artifact_score_a=artifact_score_a,
        ssm_similarity=ssm_sim,
        spectral_corr=spec_corr,
        rhythm_similarity=rhythm_sim,
        mert_similarity=mert_sim,
        stem_scores=stem_scores,
        cqt_similarity=cqt_result["cqt_similarity"],
        qmax_score=qmax_result["qmax_score"],
        clap_multiscale=clap_multiscale,
        panns_similarity=panns_sim,
    )

    if verbose:
        print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAIA: AI Song Attribution Scorer")
    parser.add_argument("--track_a", required=True, help="Path to the original track")
    parser.add_argument("--track_b", required=True, help="Path to the candidate track")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = compare_tracks(args.track_a, args.track_b, verbose=args.verbose)

    print("\n=== MAIA Attribution Report ===")
    print(f"Attribution Score   : {result['attribution_score']:.4f}")
    print(f"Semantic Similarity : {result['semantic_similarity']:.4f}")
    print(f"Melodic Alignment   : {result['melodic_alignment']:.4f}")
    print(f"Structural Match    : {result['structural_correspondence']:.4f}")
    print(f"AI Artifact Score   : {result['ai_artifact_score']:.4f}")
    print(f"SSM Similarity      : {result['ssm_similarity']:.4f}")
    print(f"Spectral Corr.      : {result['spectral_correlation']:.4f}")
    print(f"Rhythm Similarity   : {result['rhythm_similarity']:.4f}")
    print(f"MERT Similarity     : {result['mert_similarity']:.4f}")
    print(f"Verdict             : {result['verdict']}")
