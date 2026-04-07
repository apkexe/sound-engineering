"""
PANNs (Pretrained Audio Neural Networks) Perceptual Embeddings
==============================================================
Extracts 2048-D perceptual audio embeddings using CNN14 pretrained on AudioSet.

PANNs captures general audio characteristics (timbre, texture, environmental
sounds) that complement CLAP (semantic) and MERT (music-specific) embeddings.

Reference:
  Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for
  Audio Pattern Recognition", IEEE/ACM TASLP, 2020.
"""

import numpy as np
import librosa

_panns_model = None
_panns_sr = 32000  # PANNs expects 32 kHz


def _get_model():
    """Lazy-load PANNs CNN14 model (downloads checkpoint on first use)."""
    global _panns_model
    if _panns_model is None:
        try:
            from panns_inference import AudioTagging
            _panns_model = AudioTagging(checkpoint_path=None, device="cpu")
            print("[PANNs] CNN14 model ready.")
        except Exception as e:
            print(f"[PANNs] Failed to load model: {e}")
            _panns_model = "FAILED"
    return _panns_model


def extract_panns_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract a 2048-D PANNs embedding from an audio signal.

    Parameters
    ----------
    audio : np.ndarray, shape (N,)
        Mono audio waveform.
    sr : int
        Sample rate of the audio.

    Returns
    -------
    np.ndarray, shape (2048,)
        L2-normalized PANNs embedding.
    """
    model = _get_model()

    if model == "FAILED" or model is None:
        return np.zeros(2048, dtype=np.float32)

    # Resample to 32 kHz if needed
    if sr != _panns_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=_panns_sr)

    # PANNs expects (batch, samples) float32
    audio_input = audio[np.newaxis, :].astype(np.float32)

    try:
        _clipwise, embedding = model.inference(audio_input)
        emb = embedding[0]  # shape (2048,)
        # L2-normalize
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        return emb.astype(np.float32)
    except Exception as e:
        print(f"[PANNs] Inference failed: {e}")
        return np.zeros(2048, dtype=np.float32)


def panns_similarity(audio_a: np.ndarray, sr_a: int,
                     audio_b: np.ndarray, sr_b: int) -> float:
    """
    Compute cosine similarity between PANNs embeddings of two audio tracks.

    Returns
    -------
    float in [0, 1] — 1 means perceptually identical.
    """
    emb_a = extract_panns_embedding(audio_a, sr_a)
    emb_b = extract_panns_embedding(audio_b, sr_b)

    # Both are L2-normalized, so dot product = cosine similarity
    raw = float(np.dot(emb_a, emb_b))
    # Map from [-1, 1] to [0, 1]
    return (raw + 1.0) / 2.0
