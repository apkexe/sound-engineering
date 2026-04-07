"""
CLAP Embedding Extractor
========================
Uses Microsoft's CLAP (Contrastive Language-Audio Pre-training) model
to extract semantic audio embeddings per window.

CLAP gives us a rich, pre-trained representation that captures high-level
musical semantics (genre, mood, instrumentation) — going beyond low-level
spectral features. This is critical for detecting semantic similarity
between an original and its AI derivative across variations in timbre/key.

Model used: laion/larger_clap_music_and_speech
"""

from typing import List
import importlib
import os

import numpy as np
import librosa

from src.features.temporal_sampler import AudioWindow

MODEL_ID = "laion/larger_clap_music_and_speech"


class CLAPEmbedder:
    """
    Wraps the CLAP model to produce audio embeddings per window.
    Embeddings are L2-normalized and ready for cosine similarity.
    """

    def __init__(self, model_id: str = MODEL_ID, device: str = None):
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None
        self.use_fallback = False

        if os.getenv("MAIA_DISABLE_CLAP", "0") == "1":
            self.use_fallback = True
            self._load_error = "disabled by MAIA_DISABLE_CLAP=1"
            print("[CLAPEmbedder] CLAP disabled via env var; using fallback embedding.")
            return

        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            ClapModel = getattr(transformers, "ClapModel")
            ClapProcessor = getattr(transformers, "ClapProcessor")

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"[CLAPEmbedder] Loading {model_id} on {self.device}...")
            self.processor = ClapProcessor.from_pretrained(model_id)
            self.model = ClapModel.from_pretrained(model_id).to(self.device)
            self.model.eval()
            print("[CLAPEmbedder] Model ready.")
        except Exception as exc:
            # Keep pipeline runnable offline and during setup; this fallback can be
            # replaced by CLAP once dependencies/model weights are available.
            self.use_fallback = True
            self._load_error = str(exc)
            print(
                "[CLAPEmbedder] Falling back to handcrafted embedding "
                f"(CLAP unavailable: {self._load_error})"
            )

    def embed_window(self, window: AudioWindow) -> np.ndarray:
        """
        Compute a single CLAP audio embedding for an AudioWindow.

        Returns
        -------
        np.ndarray of shape (512,), L2-normalized.
        """
        if self.use_fallback:
            return self._fallback_embed(window)

        torch = importlib.import_module("torch")

        # CLAP expects 48 kHz – resample if needed
        audio = window.audio
        sr = window.sr
        if sr != 48000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
            sr = 48000

        # CLAP processor expects sampling_rate; resample if needed
        inputs = self.processor(
            audio=audio,
            sampling_rate=sr,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.model.get_audio_features(**inputs)
            # Newer transformers returns BaseModelOutputWithPooling
            if hasattr(out, 'pooler_output'):
                embed = out.pooler_output.cpu().numpy()[0]
            else:
                embed = out.cpu().numpy()[0]

        # L2 normalize
        norm = np.linalg.norm(embed)
        if norm > 0:
            embed = embed / norm
        return embed

    def _fallback_embed(self, window: AudioWindow) -> np.ndarray:
        """
        Lightweight 512-D embedding using mel/chroma statistics.
        This keeps the full attribution pipeline operational when CLAP
        cannot be loaded (no internet or missing heavy deps).
        """
        y = window.audio
        sr = window.sr
        if len(y) == 0:
            return np.zeros(512, dtype=np.float32)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)

        base = np.concatenate([
            mel_db.mean(axis=1),
            mel_db.std(axis=1),
            chroma.mean(axis=1),
            chroma.std(axis=1),
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
        ])

        # Repeat/truncate to 512 for compatibility with downstream code.
        repeats = int(np.ceil(512 / len(base)))
        embed = np.tile(base, repeats)[:512].astype(np.float32)
        norm = np.linalg.norm(embed)
        if norm > 0:
            embed = embed / norm
        return embed

    def embed_windows(self, windows: List[AudioWindow]) -> np.ndarray:
        """
        Compute CLAP embeddings for a list of windows.
        Returns energy-weighted mean embedding (attention pooling).

        Returns
        -------
        np.ndarray of shape (512,), the aggregated track-level embedding.
        """
        if not windows:
            return np.zeros(512)

        embeddings = []
        energies = []
        for w in windows:
            emb = self.embed_window(w)
            embeddings.append(emb)
            energies.append(w.energy)

        # Attention pooling: weight each window's embedding by its energy
        embeddings = np.array(embeddings)  # (N, 512)
        energies = np.array(energies)
        if energies.sum() > 0:
            weights = energies / energies.sum()
        else:
            weights = np.ones(len(energies)) / len(energies)

        aggregated = (embeddings * weights[:, None]).sum(axis=0)
        # Re-normalize
        norm = np.linalg.norm(aggregated)
        if norm > 0:
            aggregated = aggregated / norm
        return aggregated

    def cosine_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized embeddings."""
        return float(np.dot(emb_a, emb_b))

    def multi_scale_similarity(self, windows_a: List[AudioWindow],
                                windows_b: List[AudioWindow]) -> float:
        """
        Multi-scale CLAP temporal alignment score.

        Instead of a single global embedding, this computes CLAP embeddings
        per window and finds the best temporal alignment between the two
        sequences using a window-level cosine similarity matrix.

        This captures structural similarity (e.g., chorus matches chorus)
        that a single global embedding misses.

        Returns a similarity score in [0, 1].
        """
        if not windows_a or not windows_b:
            return 0.5

        embs_a = np.array([self.embed_window(w) for w in windows_a])
        embs_b = np.array([self.embed_window(w) for w in windows_b])

        # Cosine similarity matrix between all window pairs
        sim_matrix = embs_a @ embs_b.T  # (Na, Nb)

        # Best match per window in A (how well each section of A is covered in B)
        best_per_a = sim_matrix.max(axis=1).mean()
        # Best match per window in B
        best_per_b = sim_matrix.max(axis=0).mean()

        # Symmetric score
        score = (best_per_a + best_per_b) / 2.0
        return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))
