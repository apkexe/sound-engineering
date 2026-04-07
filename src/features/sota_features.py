"""
SOTA-Inspired Feature Extractors
=================================
Implements state-of-the-art techniques for AI music detection and pairwise
attribution, grounded in recent literature:

  - Rahman et al. "SONICS: Synthetic Or Not — Identifying Counterfeit Songs"
    ICLR 2025. arXiv:2408.14080
  - Comanducci et al. "FakeMusicCaps: Detection and Attribution of Synthetic
    Music Generated via Text-to-Music Models", arXiv:2409.10684
  - Wu et al. "Large-Scale Contrastive Language-Audio Pretraining with Feature
    Fusion" (CLAP), ICASSP 2023

Key additions:
  1. Long-Range Self-Similarity (LRSS) — compares self-similarity matrices
     (SSMs) between two tracks to capture structural correspondence at the
     spectrogram level (SpecTTTra insight).
  2. Mel-Spectrogram Cross-Correlation — direct spectral comparison between
     paired tracks (FakeMusicCaps ResNet18+Spec insight).
  3. Onset/Rhythm Complexity — AI generators produce more predictable beat
     patterns (SONICS analysis of true/false positives).
  4. Pitch Variability — AI songs exhibit limited pitch contour variation
     (SONICS result analysis).
"""

import librosa
import numpy as np


# ── Long-Range Self-Similarity Matrix ──────────────────────────────────

def compute_ssm(y: np.ndarray, sr: int, feature: str = "chroma",
                hop_length: int = 512) -> np.ndarray:
    """
    Compute the self-similarity matrix (SSM) of an audio signal.

    The SSM captures long-range temporal dependencies: repeated verses,
    choruses, and rhythmic patterns produce bright diagonals. SONICS
    (Rahman et al., ICLR 2025) shows that real songs maintain consistent
    SSMs while AI-generated songs often fail at long-range coherence.

    Parameters
    ----------
    y : 1-D audio signal.
    sr : sample rate.
    feature : "chroma" or "mfcc".
    hop_length : hop size for feature extraction.

    Returns
    -------
    ssm : (T, T) cosine-similarity matrix, values in [-1, 1].
    """
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode="constant")

    if feature == "chroma":
        feat = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    else:
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

    # Down-sample to manageable size (max ~100 frames) to keep O(N^2) tractable
    # Reduced from 200 to 100: SSM showed <0.002 discriminative gap in baseline,
    # so trading resolution for ~4x speed improvement is worthwhile.
    max_frames = 100
    if feat.shape[1] > max_frames:
        indices = np.linspace(0, feat.shape[1] - 1, max_frames, dtype=int)
        feat = feat[:, indices]

    # L2 normalise columns
    norms = np.linalg.norm(feat, axis=0, keepdims=True) + 1e-8
    feat_norm = feat / norms

    # Cosine similarity matrix
    ssm = feat_norm.T @ feat_norm  # (T, T)
    return ssm


def compare_ssm(y_a: np.ndarray, sr_a: int,
                y_b: np.ndarray, sr_b: int,
                feature: str = "chroma") -> float:
    """
    Compare self-similarity matrices of two tracks.

    If Track B is derived from Track A, their SSMs should share a similar
    block/diagonal structure—even if key, tempo, or timbre differ.
    We compare SSMs via normalized Frobenius inner product.

    Returns a similarity score in [0, 1].
    """
    ssm_a = compute_ssm(y_a, sr_a, feature)
    ssm_b = compute_ssm(y_b, sr_b, feature)

    # Resize to same dimension
    n = min(ssm_a.shape[0], ssm_b.shape[0], 100)
    ssm_a_r = _resize_matrix(ssm_a, n)
    ssm_b_r = _resize_matrix(ssm_b, n)

    # Normalised Frobenius inner product
    norm_a = np.linalg.norm(ssm_a_r, "fro") + 1e-8
    norm_b = np.linalg.norm(ssm_b_r, "fro") + 1e-8
    similarity = float(np.sum(ssm_a_r * ssm_b_r) / (norm_a * norm_b))

    return float(np.clip((similarity + 1) / 2.0, 0.0, 1.0))


def _resize_matrix(m: np.ndarray, n: int) -> np.ndarray:
    """Bi-linear resize of a square matrix to (n, n)."""
    from numpy import interp as np_interp
    old = np.linspace(0, 1, m.shape[0])
    new = np.linspace(0, 1, n)
    # Row-wise interpolation
    rows = np.array([np.interp(new, old, m[i, :]) for i in range(m.shape[0])])
    # Column-wise interpolation
    out = np.array([np.interp(new, old, rows[:, j]) for j in range(n)]).T
    return out


# ── Mel-Spectrogram Cross-Correlation ──────────────────────────────────

def mel_cross_correlation(y_a: np.ndarray, sr_a: int,
                          y_b: np.ndarray, sr_b: int,
                          n_mels: int = 128, hop_length: int = 512) -> float:
    """
    Compare mel-spectrograms of two tracks via cosine similarity of their
    frequency-averaged profiles. Inspired by ResNet18+Spec baseline from
    FakeMusicCaps (Comanducci et al., 2024) which showed that simple
    spectrogram-based classifiers achieve strong detection performance.

    We compute the mean mel-frequency profile (spectral envelope) and the
    mean temporal envelope for each track, then combine their cosine
    similarities.

    Returns a score in [0, 1].
    """
    mel_a = _safe_mel(y_a, sr_a, n_mels, hop_length)
    mel_b = _safe_mel(y_b, sr_b, n_mels, hop_length)

    # Frequency profile: average across time → (n_mels,)
    freq_a = mel_a.mean(axis=1)
    freq_b = mel_b.mean(axis=1)
    freq_sim = _cosine_sim(freq_a, freq_b)

    # Temporal envelope: average across frequency, then resample to same length
    temp_a = mel_a.mean(axis=0)
    temp_b = mel_b.mean(axis=0)
    n = min(len(temp_a), len(temp_b), 200)
    temp_a_r = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(temp_a)), temp_a)
    temp_b_r = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(temp_b)), temp_b)
    temp_sim = _cosine_sim(temp_a_r, temp_b_r)

    # Combine: frequency profile is more stable, weight it higher
    combined = 0.6 * freq_sim + 0.4 * temp_sim
    return float(np.clip(combined, 0.0, 1.0))


def _safe_mel(y, sr, n_mels, hop_length):
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode="constant")
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    return librosa.power_to_db(mel, ref=np.max)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    sim = float(np.dot(a, b) / (na * nb))
    return (sim + 1.0) / 2.0  # map [-1,1] → [0,1]


# ── Onset / Rhythm Complexity ──────────────────────────────────────────

def rhythm_similarity(y_a: np.ndarray, sr_a: int,
                      y_b: np.ndarray, sr_b: int) -> float:
    """
    Compare rhythmic patterns between two tracks.

    SONICS (Rahman et al., 2025) found that AI-generated songs exhibit more
    predictable rhythmic structures. We compare:
      1. Onset strength envelope correlation
      2. Tempo ratio

    Returns a similarity score in [0, 1].
    """
    # Onset strength envelopes
    onset_a = _onset_envelope(y_a, sr_a)
    onset_b = _onset_envelope(y_b, sr_b)

    # Resample to same length and correlate
    n = min(len(onset_a), len(onset_b), 200)
    oa = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(onset_a)), onset_a)
    ob = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(onset_b)), onset_b)
    onset_corr = _cosine_sim(oa, ob)

    # Tempo similarity
    tempo_a = _estimate_tempo(y_a, sr_a)
    tempo_b = _estimate_tempo(y_b, sr_b)
    # Tempo ratio handles double/half time
    ratio = tempo_a / (tempo_b + 1e-8)
    # Check if close to 1x, 2x, or 0.5x (common in derivatives)
    tempo_diffs = [abs(ratio - r) for r in [0.5, 1.0, 2.0]]
    tempo_sim = float(np.exp(-min(tempo_diffs) * 5.0))

    return float(np.clip(0.6 * onset_corr + 0.4 * tempo_sim, 0.0, 1.0))


def _onset_envelope(y, sr):
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode="constant")
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)


def _estimate_tempo(y, sr):
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode="constant")
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo) if np.isscalar(tempo) else float(tempo[0])


# ── Pitch Variability ─────────────────────────────────────────────────

def pitch_variability_score(y: np.ndarray, sr: int) -> float:
    """
    Measure pitch contour variability. SONICS (Rahman et al., 2025) analysis
    shows AI-generated songs have limited pitch variability and less expressive
    vocal pitch contours compared to real music.

    Higher score → more variability (more human-like).
    Returns [0, 1].
    """
    if len(y) < 4096:
        y = np.pad(y, (0, 4096 - len(y)), mode="constant")

    # Extract pitch using pyin (probabilistic YIN)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"), sr=sr,
        hop_length=512,
    )
    # Keep only voiced frames
    voiced = f0[voiced_flag]
    if len(voiced) < 5:
        return 0.5

    # Convert to semitones relative to median
    median_f0 = np.median(voiced)
    semitones = 12 * np.log2(voiced / (median_f0 + 1e-8) + 1e-8)

    # Variability metrics
    pitch_range = float(np.ptp(semitones))  # total range in semitones
    pitch_std = float(np.std(semitones))

    # Typical ranges: real music 8–24 semitones, AI 4–12 semitones
    range_score = float(np.clip(pitch_range / 24.0, 0.0, 1.0))
    std_score = float(np.clip(pitch_std / 4.0, 0.0, 1.0))

    return float(0.5 * range_score + 0.5 * std_score)


# ── Dynamic Range Analysis ─────────────────────────────────────────────

def dynamic_range_score(y: np.ndarray, sr: int) -> float:
    """
    Measure dynamic range variability. AI generators tend to produce audio
    with more compressed dynamics (SONICS finding: AI lacks the 'dynamic
    variation and unexpected changes' of real music).

    Higher score → more dynamic range (more human-like).
    Returns [0, 1].
    """
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode="constant")

    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    if len(rms) < 2:
        return 0.5

    rms_db = librosa.amplitude_to_db(rms + 1e-8)

    # Dynamic range = difference between 95th and 5th percentile
    dr = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))

    # Natural music: ~15–40 dB dynamic range; AI: ~5–15 dB
    return float(np.clip(dr / 30.0, 0.0, 1.0))


# ── CQT-based Chroma + Optimal Transposition Index (OTI) ──────────────

def cqt_chroma_oti_similarity(y_a: np.ndarray, sr_a: int,
                               y_b: np.ndarray, sr_b: int,
                               hop_length: int = 512) -> dict:
    """
    CQT-based chroma comparison with Optimal Transposition Index (OTI).

    The standard in cover song identification systems. CQT provides
    logarithmic frequency resolution matching musical pitch — superior
    to mel-based chroma for tonal comparison.

    OTI finds the key shift (0-11 semitones) that maximizes similarity
    between the two chroma sequences, handling cases where an AI model
    transposes the original.

    Returns dict with:
      - cqt_similarity: best cosine similarity across all transpositions
      - optimal_transposition: the shift that maximized similarity
    """
    chroma_a = _safe_cqt_chroma(y_a, sr_a, hop_length)
    chroma_b = _safe_cqt_chroma(y_b, sr_b, hop_length)

    # Global chroma profiles (average over time)
    prof_a = chroma_a.mean(axis=1)  # (12,)
    prof_b = chroma_b.mean(axis=1)  # (12,)

    # Test all 12 transpositions
    best_sim = -2.0
    best_shift = 0
    for shift in range(12):
        shifted_b = np.roll(prof_b, shift)
        sim = float(np.dot(prof_a, shifted_b) / (
            np.linalg.norm(prof_a) * np.linalg.norm(shifted_b) + 1e-8
        ))
        if sim > best_sim:
            best_sim = sim
            best_shift = shift

    return {
        "cqt_similarity": float(np.clip((best_sim + 1.0) / 2.0, 0.0, 1.0)),
        "optimal_transposition": best_shift,
    }


def _safe_cqt_chroma(y, sr, hop_length):
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode="constant")
    return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)


# ── Qmax / Cross-Recurrence Plot ──────────────────────────────────────

def qmax_similarity(y_a: np.ndarray, sr_a: int,
                    y_b: np.ndarray, sr_b: int,
                    hop_length: int = 512,
                    max_frames: int = 200) -> dict:
    """
    Classic cover song detection via cross-recurrence plot (CRP) analysis.

    Computes a cross-recurrence matrix between the CQT chroma sequences
    of two tracks and extracts the Qmax score — the length of the longest
    near-diagonal stripe, normalized by the shorter sequence length.

    This captures melodic/harmonic similarity despite tempo changes,
    orchestration changes, and key transposition.

    The OTI from cqt_chroma_oti_similarity is applied first to align keys.

    Parameters
    ----------
    max_frames : int
        Max frames to keep per track (for speed). CRP is O(N*M).

    Returns
    -------
    dict with:
      - qmax_score: normalized longest diagonal coverage [0, 1]
      - crp_density: fraction of recurrence points (overall similarity)
    """
    chroma_a = _safe_cqt_chroma(y_a, sr_a, hop_length)
    chroma_b = _safe_cqt_chroma(y_b, sr_b, hop_length)

    # Downsample if too long
    if chroma_a.shape[1] > max_frames:
        idx = np.linspace(0, chroma_a.shape[1] - 1, max_frames, dtype=int)
        chroma_a = chroma_a[:, idx]
    if chroma_b.shape[1] > max_frames:
        idx = np.linspace(0, chroma_b.shape[1] - 1, max_frames, dtype=int)
        chroma_b = chroma_b[:, idx]

    # Apply OTI: find best transposition
    prof_a = chroma_a.mean(axis=1)
    prof_b = chroma_b.mean(axis=1)
    best_shift = 0
    best_sim = -2.0
    for s in range(12):
        sim = float(np.dot(prof_a, np.roll(prof_b, s)) / (
            np.linalg.norm(prof_a) * np.linalg.norm(np.roll(prof_b, s)) + 1e-8
        ))
        if sim > best_sim:
            best_sim = sim
            best_shift = s
    chroma_b = np.roll(chroma_b, best_shift, axis=0)

    # L2-normalize columns
    na = np.linalg.norm(chroma_a, axis=0, keepdims=True) + 1e-8
    nb = np.linalg.norm(chroma_b, axis=0, keepdims=True) + 1e-8
    chroma_a = chroma_a / na
    chroma_b = chroma_b / nb

    # Cross-recurrence matrix: cosine similarity between all frame pairs
    crm = chroma_a.T @ chroma_b  # (Na, Nb), values in [-1, 1]

    # Binary recurrence: threshold at 0.8 cosine similarity
    threshold = 0.8
    crp = (crm >= threshold).astype(np.float32)

    # CRP density: fraction of recurrence points
    crp_density = float(crp.mean())

    # Qmax: longest near-diagonal stripe
    # We search diagonals with small slope tolerance (handles tempo changes)
    Na, Nb = crp.shape
    min_len = min(Na, Nb)
    longest = 0

    # Main diagonals and near-diagonals (offset by up to 10% of min length)
    max_offset = max(1, min_len // 10)
    for offset in range(-max_offset, max_offset + 1):
        diag = np.diag(crp, k=offset)
        # Find longest consecutive run of 1s
        run = 0
        for v in diag:
            if v > 0:
                run += 1
                longest = max(longest, run)
            else:
                run = 0

    qmax_score = float(longest / (min_len + 1e-8))
    qmax_score = float(np.clip(qmax_score, 0.0, 1.0))

    return {
        "qmax_score": qmax_score,
        "crp_density": float(np.clip(crp_density * 5.0, 0.0, 1.0)),  # scale up
    }


# ── Dmax Measure (ChromaCoverId / Serra 2009, Chen 2017) ──────────────

def dmax_similarity(y_a: np.ndarray, sr_a: int,
                    y_b: np.ndarray, sr_b: int,
                    hop_length: int = 512,
                    max_frames: int = 200) -> dict:
    """
    Dmax cover song similarity using cumulative score matrix.

    Source: albincorreya/ChromaCoverId
    Papers:
      - Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence
        quantification for cover song identification. New J. Physics.
      - Chen, N., Li, W., & Xiao, H. (2017). Fusing similarity functions
        for cover song identification. Multimedia Tools & Applications.

    Unlike Qmax (longest near-diagonal stripe), Dmax accumulates similarity
    scores along diagonals using dynamic programming, making it more robust
    to local gaps and tempo micro-variations.

    Returns dict with dmax_score in [0, 1].
    """
    chroma_a = _safe_cqt_chroma(y_a, sr_a, hop_length)
    chroma_b = _safe_cqt_chroma(y_b, sr_b, hop_length)

    # Downsample
    if chroma_a.shape[1] > max_frames:
        idx = np.linspace(0, chroma_a.shape[1] - 1, max_frames, dtype=int)
        chroma_a = chroma_a[:, idx]
    if chroma_b.shape[1] > max_frames:
        idx = np.linspace(0, chroma_b.shape[1] - 1, max_frames, dtype=int)
        chroma_b = chroma_b[:, idx]

    # Apply OTI
    prof_a = chroma_a.mean(axis=1)
    prof_b = chroma_b.mean(axis=1)
    best_shift, best_sim = 0, -2.0
    for s in range(12):
        sim = float(np.dot(prof_a, np.roll(prof_b, s)) / (
            np.linalg.norm(prof_a) * np.linalg.norm(np.roll(prof_b, s)) + 1e-8))
        if sim > best_sim:
            best_sim = sim
            best_shift = s
    chroma_b = np.roll(chroma_b, best_shift, axis=0)

    # L2-normalize
    na = np.linalg.norm(chroma_a, axis=0, keepdims=True) + 1e-8
    nb = np.linalg.norm(chroma_b, axis=0, keepdims=True) + 1e-8
    chroma_a = chroma_a / na
    chroma_b = chroma_b / nb

    # Similarity matrix
    sim_matrix = chroma_a.T @ chroma_b  # (Na, Nb)

    # Cumulative score matrix (dynamic programming along diagonals)
    Na, Nb = sim_matrix.shape
    D = np.zeros_like(sim_matrix)

    for i in range(Na):
        for j in range(Nb):
            val = sim_matrix[i, j]
            if val > 0:
                prev = 0.0
                if i > 0 and j > 0:
                    prev = max(prev, D[i-1, j-1])
                if i > 1 and j > 0:
                    prev = max(prev, D[i-2, j-1])
                if i > 0 and j > 1:
                    prev = max(prev, D[i-1, j-2])
                D[i, j] = val + prev

    dmax = float(D.max())
    min_len = min(Na, Nb)
    dmax_score = float(np.clip(dmax / (min_len + 1e-8), 0.0, 1.0))

    return {"dmax_score": dmax_score}


# ── CENS Chroma Similarity (da-tacos/acoss) ───────────────────────────

def cens_chroma_similarity(y_a: np.ndarray, sr_a: int,
                           y_b: np.ndarray, sr_b: int,
                           hop_length: int = 512) -> float:
    """
    Chroma Energy Normalized Statistics (CENS) similarity.

    Source: da-tacos/acoss feature extraction pipeline.
    Paper: Müller, M. (2007). "Information Retrieval for Music and Motion"

    CENS applies L1 normalization and statistical smoothing to chroma features,
    making them more robust to local tempo variations, articulation, and
    dynamics. This is the standard in cover song identification research.

    Returns similarity score in [0, 1].
    """
    cens_a = _safe_cens(y_a, sr_a, hop_length)
    cens_b = _safe_cens(y_b, sr_b, hop_length)

    # Average over time to get global chroma profile
    prof_a = cens_a.mean(axis=1)
    prof_b = cens_b.mean(axis=1)

    # OTI: test all 12 transpositions
    best_sim = -2.0
    for shift in range(12):
        shifted = np.roll(prof_b, shift)
        sim = float(np.dot(prof_a, shifted) / (
            np.linalg.norm(prof_a) * np.linalg.norm(shifted) + 1e-8))
        best_sim = max(best_sim, sim)

    return float(np.clip((best_sim + 1.0) / 2.0, 0.0, 1.0))


def _safe_cens(y, sr, hop_length):
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode="constant")
    return librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)


# ── Tonnetz Harmonic Similarity ───────────────────────────────────────

def tonnetz_similarity(y_a: np.ndarray, sr_a: int,
                       y_b: np.ndarray, sr_b: int) -> float:
    """
    Tonal centroid (tonnetz) feature similarity.

    Source: Inspired by da-tacos/acoss feature set and musicological analysis.
    Paper: Harte, C., Sandler, M. & Gasser, M. (2006). "Detecting Harmonic
    Change in Musical Audio." ACM Multimedia.

    Tonnetz encodes pitch content in a 6-D tonal space (fifths, minor thirds,
    major thirds axes) which captures harmonic relationships that chroma alone
    misses. Two versions of the same song should occupy similar tonnetz regions
    even if transposed.

    Returns similarity in [0,1].
    """
    t_a = _safe_tonnetz(y_a, sr_a)
    t_b = _safe_tonnetz(y_b, sr_b)

    # Global tonnetz profile (6-D)
    prof_a = t_a.mean(axis=1)
    prof_b = t_b.mean(axis=1)

    sim = float(np.dot(prof_a, prof_b) / (
        np.linalg.norm(prof_a) * np.linalg.norm(prof_b) + 1e-8))
    return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))


def _safe_tonnetz(y, sr):
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode="constant")
    return librosa.feature.tonnetz(y=y, sr=sr)


# ── Spectral Contrast Similarity ─────────────────────────────────────

def spectral_contrast_similarity(y_a: np.ndarray, sr_a: int,
                                  y_b: np.ndarray, sr_b: int,
                                  hop_length: int = 512) -> float:
    """
    Spectral contrast feature comparison.

    Source: da-tacos/acoss feature set.
    Paper: Jiang, D.-N. et al. (2002). "Music type classification by spectral
    contrast feature." IEEE Intl Conf on Multimedia & Expo.

    Captures the difference between peaks and valleys in a spectrogram across
    frequency sub-bands. AI-generated music may have different spectral contrast
    profiles than real music, and derivatives should match the original.

    Returns similarity in [0,1].
    """
    sc_a = _safe_spectral_contrast(y_a, sr_a, hop_length)
    sc_b = _safe_spectral_contrast(y_b, sr_b, hop_length)

    # Average over time
    prof_a = sc_a.mean(axis=1)
    prof_b = sc_b.mean(axis=1)

    sim = float(np.dot(prof_a, prof_b) / (
        np.linalg.norm(prof_a) * np.linalg.norm(prof_b) + 1e-8))
    return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))


def _safe_spectral_contrast(y, sr, hop_length):
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode="constant")
    return librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)


# ── Enhanced Onset / Spectral Flux Similarity ─────────────────────────

def spectral_flux_onset_similarity(y_a: np.ndarray, sr_a: int,
                                    y_b: np.ndarray, sr_b: int) -> float:
    """
    Spectral flux-based onset similarity.

    Source: Inspired by madmom/acoss onset detection pipeline.
    Paper: Böck, S. et al. (2012). "Evaluating the Online Capabilities of
    Onset Detection Methods." ISMIR.

    Uses spectral flux (frame-to-frame spectral difference) as the novelty
    function instead of librosa's default complex domain. This is more
    sensitive to timbral onsets and produces cleaner onset patterns.

    Compares novelty function envelopes between tracks.
    Returns [0,1].
    """
    flux_a = _spectral_flux(y_a, sr_a)
    flux_b = _spectral_flux(y_b, sr_b)

    # Resample to common length and correlate
    n = min(len(flux_a), len(flux_b), 200)
    fa = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(flux_a)), flux_a)
    fb = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(flux_b)), flux_b)

    return _cosine_sim(fa, fb)


def _spectral_flux(y, sr, hop_length=512, n_fft=2048):
    """Compute spectral flux novelty function."""
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)), mode="constant")
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # Half-wave rectified first-order difference
    diff = np.diff(S, axis=1)
    diff = np.maximum(0, diff)  # only positive changes (onsets)
    flux = diff.sum(axis=0)
    return flux
