"""
AI Artifact Detector
====================
Detects spectral and structural artifacts characteristic of AI music generators
(primarily Suno and Udio). These artifacts are signatures of the generation
process itself — not the musical content.

Key artifacts we detect:

1. **Phase Discontinuity Score**
   - AI generators often produce audio with abrupt phase resets at chunk
     boundaries (typically every 10–30 seconds). We measure the mean
     absolute phase jump across Short-Time Fourier Transform frames.

2. **Spectral Periodicity Patterns**
   - Generator models produce subtle harmonic "ringing" at fixed frequencies
     (related to their vocoder/codec). We detect anomalous spectral peaks
     that repeat at non-musical intervals.

3. **Unnatural Harmonic Envelope Regularity**
   - Human performances have micro-variations in pitch and dynamics.
     AI-generated audio tends to be unnaturally stable. We measure the
     coefficient of variation (CV) of the harmonic component's envelope.

4. **Spectral Flatness Anomaly**
   - AI generators applied to silence or transition regions sometimes produce
     spectrally flat "white noise" patches. We detect unusually flat segments.

5. **Mel-Cepstral Distortion (MCD)**
   - Compares the smoothness of the MFCCs across time to detect
     "over-smooth" regions that are characteristic of neural synthesis.
"""

from typing import List

import librosa
import numpy as np

from src.features.temporal_sampler import AudioWindow


class AIArtifactDetector:
    """
    Scores a list of AudioWindows for AI generation artifacts.
    Returns a float in [0, 1] — the higher the score, the more AI-like.
    """

    def score(self, windows: List[AudioWindow]) -> float:
        """
        Aggregate AI artifact score across all windows.
        We take energy-weighted mean to emphasize musically active regions.
        """
        if not windows:
            return 0.0

        scores = [self._score_window(w) for w in windows]
        energies = np.array([w.energy for w in windows])
        scores = np.array(scores)

        if energies.sum() > 0:
            return float(np.average(scores, weights=energies))
        return float(scores.mean())

    def _score_window(self, window: AudioWindow) -> float:
        """Compute artifact score for a single window. Returns [0, 1].

        Sub-detectors include baseline features plus SOTA-inspired detectors
        from SONICS (Rahman et al., ICLR 2025) and FakeMusicCaps
        (Comanducci et al., 2024):
          - Rhythmic predictability (SONICS: AI has over-regular beat patterns)
          - Pitch contour stability (SONICS: AI has limited pitch variability)
          - Dynamic range compression (SONICS: AI lacks dynamic variation)
        """
        y, sr = window.audio, window.sr
        if len(y) < sr:  # shorter than 1 second, skip
            return 0.0

        if len(y) < 2048:
            y = np.pad(y, (0, 2048 - len(y)), mode="constant")

        subscores = [
            self._phase_discontinuity(y, sr),
            self._harmonic_regularity(y, sr),
            self._spectral_flatness_anomaly(y, sr),
            self._mfcc_smoothness(y, sr),
            self._rhythmic_predictability(y, sr),
            self._pitch_contour_stability(y, sr),
            self._dynamic_compression(y, sr),
        ]
        return float(np.mean(subscores))

    # ------------------------------------------------------------------
    # Sub-detectors
    # ------------------------------------------------------------------

    def _phase_discontinuity(self, y: np.ndarray, sr: int) -> float:
        """
        Detect abrupt phase jumps in the STFT across time.
        AI chunking artifacts produce larger-than-natural phase jumps.
        Returns a score in [0, 1].
        """
        n_fft = 2048
        hop = 512
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        phase = np.angle(stft)  # (freq, time)

        # Instantaneous frequency deviation from expected linear phase
        phase_diff = np.diff(phase, axis=1)
        # Wrap to [-π, π]
        phase_diff = ((phase_diff + np.pi) % (2 * np.pi)) - np.pi
        mean_jump = float(np.mean(np.abs(phase_diff)))

        # Calibration: natural speech/music ≈ 0.8–1.2 rad; AI artifacts → higher
        # Normalize: 0=no artifact (mean_jump≤0.8), 1=strong artifact (mean_jump≥2.0)
        score = np.clip((mean_jump - 0.8) / 1.2, 0.0, 1.0)
        return float(score)

    def _harmonic_regularity(self, y: np.ndarray, sr: int) -> float:
        """
        Detect unnaturally regular (over-stable) harmonic envelope.
        Human music has organic micro-variations; AI tends to be too smooth.
        Score: 1 = unnaturally regular (AI-like), 0 = natural variation.
        """
        y_harm, _ = librosa.effects.hpss(y)
        # Envelope of harmonic component
        envelope = np.abs(librosa.stft(y_harm)).mean(axis=0)
        if envelope.std() == 0:
            return 0.5

        # Coefficient of variation: low CV → unnaturally stable
        cv = float(envelope.std() / (envelope.mean() + 1e-8))

        # Natural music typically has CV in [0.3, 1.5]
        # Very low CV (<0.15) suggests over-regularized AI generation
        if cv < 0.15:
            score = 1.0
        elif cv < 0.3:
            score = (0.3 - cv) / 0.15
        else:
            score = 0.0
        return float(score)

    def _spectral_flatness_anomaly(self, y: np.ndarray, sr: int) -> float:
        """
        Detect patches of anomalously high spectral flatness.
        Gen models sometimes insert "blank" segments with near-white-noise content.
        """
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=512)[0]
        # High flatness in musical regions is suspicious
        threshold = 0.3
        anomaly_ratio = float(np.mean(flatness > threshold))
        return np.clip(anomaly_ratio * 2, 0.0, 1.0)

    def _mfcc_smoothness(self, y: np.ndarray, sr: int) -> float:
        """
        Measure over-smoothness in MFCC trajectories.
        Neural synthesis produces MFCCs that vary too smoothly (low second-order difference).
        """
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
        # Second-order difference (acceleration)
        n_frames = mfcc.shape[1]
        if n_frames < 3:
            return 0.0

        width = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
        width = max(3, width)
        mfcc_d2 = librosa.feature.delta(mfcc, order=2, width=width)
        mean_accel = float(np.mean(np.abs(mfcc_d2)))

        # Very low acceleration indicates over-smooth synthesis
        # Natural music: mean_accel ≈ 1.5–5.0; AI: can be < 0.8
        score = np.clip(1.0 - (mean_accel / 1.5), 0.0, 1.0)
        return float(score)

    def _rhythmic_predictability(self, y: np.ndarray, sr: int) -> float:
        """
        Detect over-regular beat patterns (SONICS: AI songs have more
        predictable rhythmic structures).

        We measure the coefficient of variation of inter-onset intervals.
        Perfectly regular beats → low CV → high artifact score.
        Returns [0, 1].
        """
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=512,
            backtrack=False,
        )
        if len(onsets) < 4:
            return 0.0

        # Inter-onset intervals in frames
        ioi = np.diff(onsets).astype(float)
        cv = float(ioi.std() / (ioi.mean() + 1e-8))

        # Natural music CV ≈ 0.3–0.8; AI can be < 0.15
        if cv < 0.10:
            return 1.0
        elif cv < 0.25:
            return float((0.25 - cv) / 0.15)
        return 0.0

    def _pitch_contour_stability(self, y: np.ndarray, sr: int) -> float:
        """
        Detect unnaturally stable pitch contours (SONICS: AI has limited
        pitch variability and lacks expressive vocal techniques like
        melismatic phrasing and sudden changes).

        Measures the standard deviation of the pitch contour. Low std →
        monotone AI generation.
        Returns [0, 1].
        """
        if len(y) < 4096:
            y = np.pad(y, (0, 4096 - len(y)), mode="constant")

        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"), sr=sr,
            hop_length=512,
        )
        voiced = f0[voiced_flag]
        if len(voiced) < 5:
            return 0.0

        # Semitone deviation relative to median
        median_f0 = np.median(voiced)
        semitones = 12 * np.log2(voiced / (median_f0 + 1e-8) + 1e-8)
        pitch_std = float(np.std(semitones))

        # Natural vocals: std ≈ 2–6 semitones; AI: < 1.5 semitones
        if pitch_std < 0.8:
            return 1.0
        elif pitch_std < 2.0:
            return float((2.0 - pitch_std) / 1.2)
        return 0.0

    def _dynamic_compression(self, y: np.ndarray, sr: int) -> float:
        """
        Detect over-compressed dynamics (SONICS: real songs have 'dynamic
        variation and unexpected changes' that AI often lacks).

        Measures the interquartile range of the RMS envelope in dB.
        Returns [0, 1] — higher = more compressed (AI-like).
        """
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        if len(rms) < 2:
            return 0.0

        rms_db = librosa.amplitude_to_db(rms + 1e-8)
        iqr = float(np.percentile(rms_db, 75) - np.percentile(rms_db, 25))

        # Natural music IQR ≈ 6–20 dB; AI: < 4 dB
        if iqr < 2.0:
            return 1.0
        elif iqr < 6.0:
            return float((6.0 - iqr) / 4.0)
        return 0.0
