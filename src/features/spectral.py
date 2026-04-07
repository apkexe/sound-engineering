"""
Spectral Feature Extractor
===========================
Extracts a rich feature vector per AudioWindow, covering:

  - Chroma (12-bin pitch class profile) — melodic/harmonic fingerprint
  - MFCC (13 coefficients + delta + delta-delta) — timbral texture
  - Mel-spectrogram statistics — frequency content distribution
  - Spectral centroid, bandwidth, rolloff, flatness — spectral shape
  - Zero-crossing rate — noisiness
  - Tempo (BPM) — rhythmic fingerprint
  - Harmonic-to-Noise Ratio (HNR) — voice/instrument quality
  - RMS energy envelope — dynamics profile

All time-varying features are summarized with mean + std across frames,
giving a fixed-size vector per window.
"""

from dataclasses import dataclass
from typing import List

import librosa
import numpy as np

from src.features.temporal_sampler import AudioWindow


@dataclass
class WindowFeatures:
    """Fixed-size feature vector for a single AudioWindow."""
    chroma_mean: np.ndarray       # (12,)
    chroma_std: np.ndarray        # (12,)
    mfcc_mean: np.ndarray         # (39,) — 13 coeff + 13 delta + 13 delta2
    mfcc_std: np.ndarray          # (39,)
    mel_mean: np.ndarray          # (128,)
    mel_std: np.ndarray           # (128,)
    spectral_mean: np.ndarray     # (5,) — centroid, bandwidth, rolloff, flatness, zcr
    spectral_std: np.ndarray      # (5,)
    tempo: float
    hnr: float
    energy: float
    # Metadata
    section_label: str
    start_sec: float
    end_sec: float

    def to_vector(self) -> np.ndarray:
        """Flatten all features into a single 1D numpy array."""
        return np.concatenate([
            self.chroma_mean, self.chroma_std,
            self.mfcc_mean, self.mfcc_std,
            self.mel_mean, self.mel_std,
            self.spectral_mean, self.spectral_std,
            [self.tempo, self.hnr, self.energy],
        ])


class SpectralFeatureExtractor:
    """
    Extracts WindowFeatures from a list of AudioWindows.
    """

    def __init__(self, n_mfcc: int = 13, n_mels: int = 128, hop_length: int = 512):
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length

    def extract(self, window: AudioWindow) -> WindowFeatures:
        y, sr = window.audio, window.sr

        if len(y) == 0:
            return self._zero_features(window)

        # Guard against tiny sections from segmentation. Some librosa features
        # require minimum frame lengths; pad very short clips to stable size.
        if len(y) < 2048:
            y = np.pad(y, (0, 2048 - len(y)), mode="constant")

        hop = self.hop_length

        # --- Chroma ---
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
        chroma_mean = chroma.mean(axis=1)
        chroma_std = chroma.std(axis=1)

        # --- MFCC + deltas ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=hop)
        n_frames = mfcc.shape[1]
        if n_frames < 3:
            mfcc_d = np.zeros_like(mfcc)
            mfcc_d2 = np.zeros_like(mfcc)
        else:
            # librosa requires odd width <= number of frames
            width = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
            width = max(3, width)
            mfcc_d = librosa.feature.delta(mfcc, width=width)
            mfcc_d2 = librosa.feature.delta(mfcc, order=2, width=width)
        mfcc_full = np.vstack([mfcc, mfcc_d, mfcc_d2])  # (39, T)
        mfcc_mean = mfcc_full.mean(axis=1)
        mfcc_std = mfcc_full.std(axis=1)

        # --- Mel spectrogram ---
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels, hop_length=hop
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_mean = mel_db.mean(axis=1)
        mel_std = mel_db.std(axis=1)

        # --- Spectral shape features ---
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop)

        spectral_stack = np.vstack([centroid, bandwidth, rolloff, flatness, zcr])
        spectral_mean = spectral_stack.mean(axis=1)
        spectral_std = spectral_stack.std(axis=1)

        # --- Tempo ---
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo) if np.isscalar(tempo) else float(tempo[0])

        # --- Harmonic-to-Noise Ratio (approximation) ---
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_rms = float(np.sqrt(np.mean(y_harm ** 2)) + 1e-8)
        perc_rms = float(np.sqrt(np.mean(y_perc ** 2)) + 1e-8)
        hnr = harm_rms / (harm_rms + perc_rms)  # [0, 1], higher = more harmonic

        return WindowFeatures(
            chroma_mean=chroma_mean,
            chroma_std=chroma_std,
            mfcc_mean=mfcc_mean,
            mfcc_std=mfcc_std,
            mel_mean=mel_mean,
            mel_std=mel_std,
            spectral_mean=spectral_mean,
            spectral_std=spectral_std,
            tempo=tempo,
            hnr=hnr,
            energy=window.energy,
            section_label=window.section_label,
            start_sec=window.start_sec,
            end_sec=window.end_sec,
        )

    def extract_batch(self, windows: List[AudioWindow]) -> List[WindowFeatures]:
        return [self.extract(w) for w in windows]

    def _zero_features(self, window: AudioWindow) -> WindowFeatures:
        return WindowFeatures(
            chroma_mean=np.zeros(12),
            chroma_std=np.zeros(12),
            mfcc_mean=np.zeros(39),
            mfcc_std=np.zeros(39),
            mel_mean=np.zeros(self.n_mels),
            mel_std=np.zeros(self.n_mels),
            spectral_mean=np.zeros(5),
            spectral_std=np.zeros(5),
            tempo=0.0,
            hnr=0.0,
            energy=window.energy,
            section_label=window.section_label,
            start_sec=window.start_sec,
            end_sec=window.end_sec,
        )
