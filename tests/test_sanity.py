"""
Quick sanity check that all pipeline imports and scorer math work correctly.
Does NOT load heavy models — just verifies code structure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_imports():
    """All modules import without error."""
    from src.features.temporal_sampler import MultiScaleTemporalSampler
    from src.features.spectral import SpectralFeatureExtractor
    from src.features.embeddings import CLAPEmbedder
    from src.features.mert_embeddings import MERTEmbedder
    from src.features.artifacts import AIArtifactDetector
    from src.features.sota_features import (
        compare_ssm, mel_cross_correlation, rhythm_similarity,
        cqt_chroma_oti_similarity, qmax_similarity,
        dmax_similarity, tonnetz_similarity,
    )
    from src.features.source_separation import separate_track, stem_similarity
    from src.features.panns_embeddings import panns_similarity
    from src.model.attribution import AttributionScorer, DEFAULT_WEIGHTS
    from src.model.calibration import ScoreCalibrator
    print("[OK] All imports successful")


def test_weights_sum():
    """Weights sum to 1.0."""
    from src.model.attribution import DEFAULT_WEIGHTS
    total = sum(DEFAULT_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"
    print(f"[OK] Weights sum = {total:.6f}")


def test_scorer_with_dummy():
    """Scorer produces valid output with dummy inputs."""
    import numpy as np
    from src.model.attribution import AttributionScorer

    scorer = AttributionScorer()

    # Minimal dummy inputs
    class FakeWindow:
        chroma_mean = np.random.rand(12)
        onset_density = 5.0
        rms_mean = 0.1
        spectral_centroid_mean = 2000.0
        start_sec = 0.0
        end_sec = 10.0

    result = scorer.score(
        feats_a=[FakeWindow()],
        feats_b=[FakeWindow()],
        emb_a=np.random.randn(512).astype(np.float32),
        emb_b=np.random.randn(512).astype(np.float32),
        artifact_score=0.5,
        artifact_score_a=0.45,
        ssm_similarity=0.8,
        spectral_corr=0.9,
        rhythm_similarity=0.7,
        mert_similarity=0.85,
        stem_scores={"stem_combined": 0.78, "vocal_similarity": 0.8,
                     "drum_similarity": 0.7, "bass_similarity": 0.75,
                     "other_similarity": 0.8},
        cqt_similarity=0.95,
        qmax_score=0.12,
        clap_multiscale=0.88,
        panns_similarity=0.82,
    )

    assert 0.0 <= result["attribution_score"] <= 1.0
    assert "verdict" in result
    assert len(result) > 13  # Should have all sub-scores

    # Check all expected keys exist
    expected_keys = [
        "attribution_score", "semantic_similarity", "melodic_alignment",
        "structural_correspondence", "ai_artifact_score", "ssm_similarity",
        "spectral_correlation", "rhythm_similarity", "mert_similarity",
        "stem_combined", "cqt_similarity", "qmax_score", "clap_multiscale",
        "panns_similarity", "verdict",
    ]
    for k in expected_keys:
        assert k in result, f"Missing key: {k}"

    print(f"[OK] Scorer output: score={result['attribution_score']:.4f}, "
          f"verdict='{result['verdict'][:30]}...'")
    print(f"[OK] All {len(expected_keys)} expected keys present")


def test_branch_count():
    """Verify we have exactly 13 branches."""
    from src.model.attribution import DEFAULT_WEIGHTS
    assert len(DEFAULT_WEIGHTS) == 13, f"Expected 13 branches, got {len(DEFAULT_WEIGHTS)}"
    print(f"[OK] Branch count = {len(DEFAULT_WEIGHTS)}")


if __name__ == "__main__":
    test_imports()
    test_weights_sum()
    test_branch_count()
    test_scorer_with_dummy()
    print("\n=== All validation checks passed ===")
