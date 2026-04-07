"""
Attribution Scorer
==================
Combines all feature signals into a final Attribution Score.

The Attribution Score answers: "How likely is Track B a synthetic
derivative (attribution) of Track A?"

Score components:
  1. semantic_similarity    — CLAP cosine similarity (how "musically alike")
  2. melodic_alignment      — DTW distance on chroma features (melodic trace)
  3. structural_correspondence — Section boundary temporal alignment
  4. ai_artifact_score      — Strength of AI generation artifacts
  5. ssm_similarity         — Long-Range Self-Similarity Matrix comparison
                              (SONICS: SpecTTTra insight, Rahman et al. ICLR 2025)
  6. spectral_correlation   — Mel-spectrogram cross-correlation
                              (FakeMusicCaps: Comanducci et al. 2024)
  7. rhythm_similarity      — Onset/tempo pattern comparison
                              (SONICS: AI has predictable rhythms)

Final score formula:
  S = w1 * semantic + w2 * melodic + w3 * structural + w4 * artifact_boost
      + w5 * ssm + w6 * spectral + w7 * rhythm
  where artifact_boost = artifact_score * semantic  (artifacts only count
  if there's also genuine similarity — prevents false positives on random
  AI-generated audio that happens to sound AI-like)

Weights are empirically tuned and informed by SOTA ablation studies.

References:
  - Rahman et al. "SONICS: Synthetic Or Not — Identifying Counterfeit Songs"
    ICLR 2025 (arXiv:2408.14080)
  - Comanducci et al. "FakeMusicCaps" (arXiv:2409.10684)
  - Wu et al. CLAP (ICASSP 2023)
"""

from typing import List

import numpy as np
from dtaidistance import dtw

from src.features.spectral import WindowFeatures


# Default weights (sum to 1.0)
# These are the exact Exp 6 weights that achieved 85% accuracy (17/20, zero FP)
# at threshold 0.829 on the full 13-branch pipeline.
#
# Note: A gap-proportional reweighting was tested post-Exp 6 (boosting rhythm
# to 0.22, panns to 0.19, etc.) — it increased score gap from 0.040 to 0.066
# but REDUCED accuracy from 85% to 80%. Classification accuracy is what matters,
# so we use the original Exp 6 weights that produced the best result.
DEFAULT_WEIGHTS = {
    "semantic": 0.07,           # CLAP cosine similarity (gap=+0.080)
    "melodic": 0.04,            # DTW chroma alignment (gap=+0.029)
    "structural": 0.03,         # Section boundary alignment (gap=+0.013)
    "artifact_diff": 0.02,      # |artifact_A - artifact_B| (gap=+0.007)
    "ssm": 0.03,                # Self-similarity matrix comparison (gap=+0.005)
    "spectral_corr": 0.02,      # Mel-spectrogram cross-correlation (gap=+0.002)
    "rhythm": 0.12,             # Onset/tempo pattern — strongest signal (gap=+0.109)
    "mert": 0.09,               # MERT v1-330M (gap=+0.018)
    "stem_combined": 0.25,      # Per-stem comparison via HTDemucs (gap=+0.008)
    "cqt": 0.08,                # CQT chroma + OTI (gap=+0.005)
    "qmax": 0.07,               # Qmax cross-recurrence (gap=+0.075)
    "clap_multiscale": 0.10,    # Multi-scale CLAP temporal (gap=+0.071)
    "panns": 0.08,              # PANNs CNN14 perceptual (gap=+0.042)
}
# Note: Dmax and tonnetz were tested but excluded — they degraded accuracy
# from 70% to 50% on the 10-pair controlled comparison (§25).

VERDICT_THRESHOLDS = {
    0.75: "Strong AI Attribution — Track B is very likely a synthetic derivative of Track A",
    0.55: "Probable AI Attribution — Track B shows significant similarities to Track A",
    0.35: "Possible Relationship — Tracks share some features but attribution is uncertain",
    0.00: "Unlikely Attribution — Tracks appear unrelated",
}


class AttributionScorer:
    """
    Computes the final Attribution Score from extracted features and embeddings.
    """

    def __init__(self, weights: dict = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self.last_best_key_shift = 0

    def score(
        self,
        feats_a: List[WindowFeatures],
        feats_b: List[WindowFeatures],
        emb_a: np.ndarray,
        emb_b: np.ndarray,
        artifact_score: float,
        artifact_score_a: float = None,
        ssm_similarity: float = None,
        spectral_corr: float = None,
        rhythm_similarity: float = None,
        mert_similarity: float = None,
        stem_scores: dict = None,
        cqt_similarity: float = None,
        qmax_score: float = None,
        clap_multiscale: float = None,
        panns_similarity: float = None,
        dmax_score: float = None,
        tonnetz_similarity: float = None,
    ) -> dict:
        """
        Compute all sub-scores and the final attribution score.

        Parameters
        ----------
        feats_a, feats_b : per-window spectral features
        emb_a, emb_b     : track-level CLAP embeddings (L2-normalized)
        artifact_score   : global AI artifact score for Track B
        ssm_similarity   : SOTA: self-similarity matrix comparison score
        spectral_corr    : SOTA: mel-spectrogram cross-correlation score
        rhythm_similarity: SOTA: onset/tempo pattern similarity score

        Returns
        -------
        dict with all sub-scores, final attribution_score, and verdict.
        """
        # 1. Semantic similarity via CLAP embeddings
        semantic = self._semantic_similarity(emb_a, emb_b)

        # 2. Melodic alignment via DTW on chroma features
        melodic = self._melodic_alignment(feats_a, feats_b)

        # 3. Structural correspondence
        structural = self._structural_correspondence(feats_a, feats_b)

        # 4. Artifact difference: |artifact_A - artifact_B|
        # Truly attributed pairs should have SIMILAR artifact profiles
        # (both AI-generated), so low diff → attribution signal.
        # We invert: artifact_diff_score = 1 - |diff| so higher = more similar.
        art_a = artifact_score_a if artifact_score_a is not None else artifact_score
        artifact_diff = 1.0 - abs(art_a - artifact_score)

        # 5-7. SOTA-inspired branches (default to 0.5 if not computed)
        ssm_val = ssm_similarity if ssm_similarity is not None else 0.5
        spec_val = spectral_corr if spectral_corr is not None else 0.5
        rhythm_val = rhythm_similarity if rhythm_similarity is not None else 0.5
        mert_val = mert_similarity if mert_similarity is not None else 0.5
        stem_val = stem_scores.get("stem_combined", 0.5) if stem_scores else 0.5
        cqt_val = cqt_similarity if cqt_similarity is not None else 0.5
        qmax_val = qmax_score if qmax_score is not None else 0.5
        clap_ms_val = clap_multiscale if clap_multiscale is not None else 0.5
        panns_val = panns_similarity if panns_similarity is not None else 0.5
        dmax_val = dmax_score if dmax_score is not None else 0.5
        tonnetz_val = tonnetz_similarity if tonnetz_similarity is not None else 0.5

        # Final weighted combination
        w = self.weights
        final = (
            w["semantic"] * semantic
            + w["melodic"] * melodic
            + w["structural"] * structural
            + w.get("artifact_diff", w.get("artifact_boost", 0.0)) * artifact_diff
            + w.get("ssm", 0.0) * ssm_val
            + w.get("spectral_corr", 0.0) * spec_val
            + w.get("rhythm", 0.0) * rhythm_val
            + w.get("mert", 0.0) * mert_val
            + w.get("stem_combined", 0.0) * stem_val
            + w.get("cqt", 0.0) * cqt_val
            + w.get("qmax", 0.0) * qmax_val
            + w.get("clap_multiscale", 0.0) * clap_ms_val
            + w.get("panns", 0.0) * panns_val
            + w.get("dmax", 0.0) * dmax_val
            + w.get("tonnetz", 0.0) * tonnetz_val
        )
        final = float(np.clip(final, 0.0, 1.0))

        verdict = self._get_verdict(final)

        return {
            "attribution_score": round(final, 4),
            "semantic_similarity": round(semantic, 4),
            "melodic_alignment": round(melodic, 4),
            "best_chroma_key_shift": int(self.last_best_key_shift),
            "structural_correspondence": round(structural, 4),
            "ai_artifact_score": round(artifact_score, 4),
            "ai_artifact_score_a": round(art_a, 4),
            "artifact_diff": round(artifact_diff, 4),
            "ssm_similarity": round(ssm_val, 4),
            "spectral_correlation": round(spec_val, 4),
            "rhythm_similarity": round(rhythm_val, 4),
            "mert_similarity": round(mert_val, 4),
            "stem_combined": round(stem_val, 4),
            "vocal_similarity": round(stem_scores.get("vocal_similarity", 0.5), 4) if stem_scores else 0.5,
            "drum_similarity": round(stem_scores.get("drum_similarity", 0.5), 4) if stem_scores else 0.5,
            "bass_similarity": round(stem_scores.get("bass_similarity", 0.5), 4) if stem_scores else 0.5,
            "other_similarity": round(stem_scores.get("other_similarity", 0.5), 4) if stem_scores else 0.5,
            "cqt_similarity": round(cqt_val, 4),
            "qmax_score": round(qmax_val, 4),
            "clap_multiscale": round(clap_ms_val, 4),
            "panns_similarity": round(panns_val, 4),
            "dmax_score": round(dmax_val, 4),
            "tonnetz_similarity": round(tonnetz_val, 4),
            "verdict": verdict,
        }

    # ------------------------------------------------------------------
    # Sub-scorers
    # ------------------------------------------------------------------

    def _semantic_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized CLAP embeddings. Maps to [0, 1]."""
        raw = float(np.dot(emb_a, emb_b))
        # CLAP cosine similarity is in [-1, 1]; rescale to [0, 1]
        return (raw + 1.0) / 2.0

    def _melodic_alignment(
        self,
        feats_a: List[WindowFeatures],
        feats_b: List[WindowFeatures],
    ) -> float:
        """
        Compute melodic similarity via DTW on chroma mean vectors.

        Each track is represented as a sequence of per-window chroma_mean vectors.
        DTW handles tempo differences and different numbers of sections.
        Returns a similarity score in [0, 1] (1 = identical melodies).
        """
        if not feats_a or not feats_b:
            return 0.0

        # Build chroma sequences (N, 12) for each track.
        # We then test all 12 circular key shifts on Track B chroma and keep
        # the best DTW similarity, making alignment robust to transposition.
        seq_a = np.array([f.chroma_mean for f in feats_a], dtype=np.float64)
        seq_b = np.array([f.chroma_mean for f in feats_b], dtype=np.float64)

        flat_a = seq_a.flatten()
        flat_a = (flat_a - flat_a.min()) / (flat_a.max() - flat_a.min() + 1e-8)

        best_similarity = 0.0
        best_shift = 0
        for shift in range(12):
            shifted_b = np.roll(seq_b, shift=shift, axis=1)
            flat_b = shifted_b.flatten()
            flat_b = (flat_b - flat_b.min()) / (flat_b.max() - flat_b.min() + 1e-8)

            dtw_dist = dtw.distance(flat_a, flat_b)
            similarity = float(np.exp(-dtw_dist / 3.0))
            if similarity > best_similarity:
                best_similarity = similarity
                best_shift = shift

        self.last_best_key_shift = best_shift
        return float(np.clip(best_similarity, 0.0, 1.0))

    def _structural_correspondence(
        self,
        feats_a: List[WindowFeatures],
        feats_b: List[WindowFeatures],
    ) -> float:
        """
        Measure how well the structural section boundaries align
        proportionally between the two tracks.

        AI-generated derivatives often preserve the original's song structure
        (verse at ~0.15, chorus at ~0.35, etc. of total duration).
        We align section midpoints proportionally and measure Euclidean distance.
        Returns a similarity score in [0, 1].
        """
        if not feats_a or not feats_b:
            return 0.0

        def get_relative_midpoints(feats):
            """Midpoints of each section, normalized to [0, 1] by total duration."""
            starts = [f.start_sec for f in feats]
            ends = [f.end_sec for f in feats]
            min_start = min(starts)
            duration = max(ends) - min_start + 1e-8
            midpoints = [(((s + e) / 2) - min_start) / duration for s, e in zip(starts, ends)]
            return np.array(sorted(midpoints))

        mids_a = get_relative_midpoints(feats_a)
        mids_b = get_relative_midpoints(feats_b)

        # Align by padding the shorter one with zeros/ones at boundaries
        n = max(len(mids_a), len(mids_b))
        mids_a = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(mids_a)), mids_a)
        mids_b = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(mids_b)), mids_b)

        # Mean absolute difference in section midpoint positions
        mad = float(np.mean(np.abs(mids_a - mids_b)))
        # Convert to similarity: 0 diff → 1.0, 0.5 diff → 0.0
        similarity = max(0.0, 1.0 - 2.0 * mad)
        return float(similarity)

    def _get_verdict(self, score: float) -> str:
        for threshold, verdict in sorted(VERDICT_THRESHOLDS.items(), reverse=True):
            if score >= threshold:
                return verdict
        return VERDICT_THRESHOLDS[0.00]
