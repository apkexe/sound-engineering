"""
MERT Music Embedding Extractor
================================
Uses the MERT (Music Understanding) model from m-a-p to extract
music-specific embeddings that complement CLAP's language-audio space.

MERT is pre-trained on 160K hours of music with acoustic and musical
token prediction, giving it strong representations for pitch, rhythm,
timbre, and harmonic content.

Model: m-a-p/MERT-v1-330M (330M params, 1024-D)

Reference:
  Li et al. "MERT: Acoustic Music Understanding Model with Large-Scale
  Self-supervised Training" (ICLR 2024)
"""

import importlib
import os
from typing import List

import librosa
import numpy as np

from src.features.temporal_sampler import AudioWindow

MERT_MODEL_ID = "m-a-p/MERT-v1-330M"
MERT_SR = 24000  # MERT expects 24kHz


class MERTEmbedder:
    """
    Extracts music embeddings using MERT. Produces 1024-D vectors
    from the mean of the last hidden state (330M model).
    """

    def __init__(self, model_id: str = MERT_MODEL_ID, device: str = None):
        self.model = None
        self.processor = None
        self.use_fallback = False

        if os.getenv("MAIA_DISABLE_MERT", "0") == "1":
            self.use_fallback = True
            return

        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            AutoModel = getattr(transformers, "AutoModel")
            Wav2Vec2FeatureExtractor = getattr(transformers, "Wav2Vec2FeatureExtractor")

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device

            print(f"[MERTEmbedder] Loading {model_id}...")
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_id, trust_remote_code=True
            ).to(device)
            self.model.eval()
            print("[MERTEmbedder] Model ready.")
        except Exception as exc:
            self.use_fallback = True
            print(f"[MERTEmbedder] Fallback mode (unavailable: {exc})")

    def embed_track(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute a single MERT embedding for a full audio signal.

        Parameters
        ----------
        y : mono audio signal
        sr : sample rate

        Returns
        -------
        np.ndarray of shape (1024,) for 330M model, L2-normalized.
        """
        if self.use_fallback:
            return self._fallback(y, sr)

        torch = importlib.import_module("torch")

        # Resample to MERT's expected rate
        if sr != MERT_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=MERT_SR)

        # MERT can handle up to ~30s at a time; take a representative 30s chunk
        max_samples = MERT_SR * 30
        if len(y) > max_samples:
            # Take the middle 30s (most representative)
            start = (len(y) - max_samples) // 2
            y = y[start:start + max_samples]

        inputs = self.processor(
            y, sampling_rate=MERT_SR, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use last hidden state, mean-pooled across time
            hidden = outputs.last_hidden_state  # (1, T, 1024) for 330M
            embed = hidden.mean(dim=1).cpu().numpy()[0]  # (768,)

        norm = np.linalg.norm(embed)
        if norm > 0:
            embed = embed / norm
        return embed

    def similarity(self, y_a: np.ndarray, sr_a: int,
                   y_b: np.ndarray, sr_b: int) -> float:
        """
        Compute cosine similarity between MERT embeddings of two tracks.
        Returns value in [0, 1].
        """
        emb_a = self.embed_track(y_a, sr_a)
        emb_b = self.embed_track(y_b, sr_b)
        raw = float(np.dot(emb_a, emb_b))
        return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))

    def _fallback(self, y: np.ndarray, sr: int) -> np.ndarray:
        """1024-D fallback from chroma + MFCC + onset statistics."""
        if len(y) < 2048:
            y = np.pad(y, (0, 2048 - len(y)), mode="constant")

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)

        base = np.concatenate([
            chroma.mean(axis=1), chroma.std(axis=1),       # 24
            mfcc.mean(axis=1), mfcc.std(axis=1),           # 40
            [onset.mean(), onset.std(), onset.max()],       # 3
        ])  # 67-D

        repeats = int(np.ceil(1024 / len(base)))
        embed = np.tile(base, repeats)[:1024].astype(np.float32)
        norm = np.linalg.norm(embed)
        if norm > 0:
            embed = embed / norm
        return embed
