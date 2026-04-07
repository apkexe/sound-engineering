"""
Multi-Scale Temporal Sampler
============================
Solves the "window-bias" problem by detecting musical structure
and sampling one representative window per structural section.

Instead of naive approaches:
  - ❌ First 30 seconds only
  - ❌ Random N windows

We use structure-aware sampling:
  - ✅ Detect section boundaries (intro/verse/chorus/bridge/outro)
  - ✅ Sample the most representative (highest energy) 10s chunk per section
  - ✅ Ensures full-song coverage regardless of track length
"""

from dataclasses import dataclass
from typing import List

import librosa
import numpy as np


WINDOW_DURATION = 10.0   # seconds per window
SR = 22050               # target sample rate


@dataclass
class AudioWindow:
    """A time-bounded excerpt of an audio track."""
    audio: np.ndarray       # shape: (samples,)
    sr: int
    start_sec: float
    end_sec: float
    section_label: str      # e.g. "section_0", "intro", etc.
    energy: float           # RMS energy of this window (for attention weighting)


class MultiScaleTemporalSampler:
    """
    Loads an audio file, detects structural sections, and returns
    a list of AudioWindow objects — one representative window per section.
    """

    def __init__(
        self,
        window_duration: float = WINDOW_DURATION,
        sr: int = SR,
        max_sections: int = 8,
        min_sections: int = 2,
    ):
        self.window_duration = window_duration
        self.sr = sr
        self.max_sections = max_sections
        self.min_sections = min_sections

    def sample(self, audio_path: str) -> List[AudioWindow]:
        """
        Load audio and return structure-aware windows.

        Parameters
        ----------
        audio_path : str
            Path to audio file (mp3, wav, flac, etc.)

        Returns
        -------
        List[AudioWindow], sorted by start time.
        """
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        boundaries_sec = self._detect_boundaries(y, sr, duration)
        windows = self._extract_windows(y, sr, boundaries_sec)
        if not windows:
            rms = float(np.sqrt(np.mean(y ** 2))) if len(y) > 0 else 0.0
            windows = [
                AudioWindow(
                    audio=y,
                    sr=sr,
                    start_sec=0.0,
                    end_sec=duration,
                    section_label="full_track",
                    energy=rms,
                )
            ]
        return windows

    def _detect_boundaries(
        self, y: np.ndarray, sr: int, duration: float
    ) -> List[float]:
        """
        Detect structural section boundaries.

        For tracks longer than 60s we use efficient uniform segmentation
        (~1 section per 30s) to avoid O(T²) recurrence matrix costs.
        For shorter tracks we use agglomerative MFCC clustering.
        """
        n_sections = min(
            self.max_sections,
            max(self.min_sections, int(duration / 30))
        )

        # For long tracks, skip expensive recurrence matrix and use
        # uniform time boundaries. This keeps processing under ~10s
        # even for 5-minute songs.
        if duration > 60:
            step = duration / n_sections
            boundaries = [i * step for i in range(n_sections)] + [duration]
            return boundaries

        # Short tracks: use agglomerative MFCC segmentation
        hop_length = 512
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        n_frames = mfcc.shape[1]
        if n_frames < 3:
            return [0.0, duration]

        width = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
        width = max(3, width)
        mfcc_delta = librosa.feature.delta(mfcc, width=width)
        features = np.vstack([mfcc, mfcc_delta])

        frames = librosa.segment.agglomerative(features, n_sections)
        times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

        boundaries = sorted(set([0.0] + list(times) + [duration]))
        return boundaries

    def _extract_windows(
        self, y: np.ndarray, sr: int, boundaries_sec: List[float]
    ) -> List[AudioWindow]:
        """
        For each section defined by consecutive boundaries,
        extract the highest-energy window_duration-long excerpt.
        """
        windows = []
        for i in range(len(boundaries_sec) - 1):
            section_start = boundaries_sec[i]
            section_end = boundaries_sec[i + 1]
            section_duration = section_end - section_start

            # Skip degenerate boundaries from segmentation jitter.
            if section_duration < 1.0:
                continue

            # If section is shorter than window, take the whole section
            if section_duration <= self.window_duration:
                win_start = section_start
                win_end = section_end
            else:
                # Slide a window through section, pick highest RMS
                win_start, win_end = self._best_energy_window(
                    y, sr, section_start, section_end
                )

            start_sample = int(win_start * sr)
            end_sample = int(win_end * sr)
            audio_chunk = y[start_sample:end_sample]

            rms = float(np.sqrt(np.mean(audio_chunk ** 2))) if len(audio_chunk) > 0 else 0.0

            windows.append(AudioWindow(
                audio=audio_chunk,
                sr=sr,
                start_sec=win_start,
                end_sec=win_end,
                section_label=f"section_{i}",
                energy=rms,
            ))
        return windows

    def _best_energy_window(
        self,
        y: np.ndarray,
        sr: int,
        section_start: float,
        section_end: float,
    ) -> tuple:
        """
        Slide a window of `window_duration` through [section_start, section_end]
        with 50% overlap and return the (start, end) of the highest-RMS window.
        """
        step = self.window_duration / 2
        best_rms = -1.0
        best_start = section_start

        t = section_start
        while t + self.window_duration <= section_end:
            s = int(t * sr)
            e = int((t + self.window_duration) * sr)
            chunk = y[s:e]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms > best_rms:
                best_rms = rms
                best_start = t
            t += step

        return best_start, best_start + self.window_duration
