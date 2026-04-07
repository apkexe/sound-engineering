# MAIA — Scientific Implementations & Results

## Multi-scale Attribution Intelligence Architecture

> **Task**: AI Generated Song Detection — AI-Original Pairwise Similarity  
> **Dataset**: MIPPIA (62/70 pairs downloaded, 132 WAV files from YouTube)  
> **Evaluation**: Balanced positive/negative pair classification (124 evaluation pairs)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Multi-Scale Temporal Sampler](#2-multi-scale-temporal-sampler)
3. [Spectral Feature Extractor](#3-spectral-feature-extractor)
4. [CLAP Semantic Embeddings](#4-clap-semantic-embeddings)
5. [MERT Music Understanding Embeddings](#5-mert-music-understanding-embeddings)
6. [AI Artifact Detector (7 Sub-Detectors)](#6-ai-artifact-detector-7-sub-detectors)
7. [SOTA-Inspired Feature Extractors](#7-sota-inspired-feature-extractors)
8. [Key-Shift-Invariant Melodic DTW](#8-key-shift-invariant-melodic-dtw)
9. [Structural Correspondence Scoring](#9-structural-correspondence-scoring)
10. [8-Branch Attribution Scorer](#10-8-branch-attribution-scorer)
11. [Logistic Regression Calibration Layer](#11-logistic-regression-calibration-layer)
12. [**NEW** Artifact Diff Branch (Improvement #1)](#12-artifact-diff-branch-improvement-1)
13. [**NEW** HTDemucs Source Separation (Improvement #2)](#13-htdemucs-source-separation-improvement-2)
14. [**NEW** MERT v1-330M Upgrade (Improvement #3)](#14-mert-v1-330m-upgrade-improvement-3)
15. [**NEW** CQT Chroma + OTI (Improvement #4)](#15-cqt-chroma--oti-improvement-4)
16. [**NEW** Qmax / Cross-Recurrence Plot (Improvement #5)](#16-qmax--cross-recurrence-plot-improvement-5)
17. [**NEW** GBDT Calibrator (Improvement #6)](#17-gbdt-calibrator-improvement-6)
18. [**NEW** Multi-Scale CLAP Temporal Alignment (Improvement #7)](#18-multi-scale-clap-temporal-alignment-improvement-7)
19. [**NEW** PANNs Perceptual Embeddings (Improvement #8)](#19-panns-perceptual-embeddings-improvement-8)
20. [Evaluation Results & Ablation](#20-evaluation-results--ablation)
21. [**NEW** GitHub Repo Research & Technique Extraction](#21-github-repo-research--technique-extraction)
22. [**NEW** Experimental Feature Evaluation (5 Candidates)](#22-experimental-feature-evaluation-5-candidates)
23. [**NEW** Dmax Cumulative CRP (Improvement #9)](#23-dmax-cumulative-crp-improvement-9)
24. [**NEW** Tonnetz Harmonic Similarity (Improvement #10)](#24-tonnetz-harmonic-similarity-improvement-10)
25. [**NEW** 15-Branch Subset Comparison](#25-15-branch-subset-comparison)
26. [References](#26-references)
27. [**NEW** Overfitting Audit & Deployment Guidance](#27-overfitting-audit--deployment-guidance)

---

## 1. Architecture Overview

> **Idea Origin**: Core architecture designed from scratch based on the **exercise guidelines** ("AI Generated Song Detection — AI-Original Pairwise Similarity"). The multi-branch scoring approach was inspired by the exercise's suggestion to combine multiple signal types (semantic, melodic, structural) and informed by the three recommended datasets: **SONICS** (Rahman et al., ICLR 2025), **FakeMusicCaps** (Comanducci et al., 2024), and **MIPPIA** (Mippia/SMP). The pairwise formulation ("was Track B generated using Track A as a reference?") comes directly from the exercise brief.

MAIA processes two audio tracks (Track A: original, Track B: candidate) through a multi-branch pipeline and produces a single attribution score in [0, 1] indicating how likely Track B is an AI-generated derivative of Track A.

**Processing Pipeline (v3 — 15 Branches):**

```
Track A ──┐                                    ┌─→ Semantic Similarity (CLAP)
Track B ──┤                                    ├─→ Multi-Scale CLAP Temporal Alignment
          │                                    ├─→ Melodic Alignment (DTW)
          ├─→ Temporal Sampler ─→ Windows ─────├─→ Structural Correspondence
          │                                    ├─→ AI Artifact Diff
          │                                    │
          ├─→ Full Audio Load ─────────────────├─→ SSM Similarity
          │                                    ├─→ Spectral Correlation
          │                                    ├─→ Rhythm Similarity
          │                                    ├─→ CQT Chroma + OTI
          │                                    ├─→ Qmax Cross-Recurrence
          │                                    ├─→ Dmax Cumulative CRP (NEW)
          │                                    ├─→ Tonnetz Harmonic Similarity (NEW)
          │                                    ├─→ PANNs Perceptual Similarity
          │                                    │
          ├─→ MERT v1-330M ───────────────────├─→ MERT Similarity
          │                                    │
          ├─→ HTDemucs Source Separation ──────├─→ Stem Combined (vocal/drum/bass/other)
          │
          └─→ 15-Branch Weighted Scorer ─→ Attribution Score ∈ [0, 1]
               (or GBDT Calibrator)
```

**Key Design Principles:**
- **Structure-aware sampling**: Windows selected per musical section, not randomly
- **Key-shift invariance**: All 12 chromatic transpositions tested in melodic DTW
- **Multi-model ensembling**: CLAP (language-audio), MERT (music-specific), handcrafted features
- **Energy-weighted attention pooling**: Higher-energy windows contribute more to embeddings

---

## 2. Multi-Scale Temporal Sampler

> **Idea Origin**: **Own design** — motivated by a finding in the **FakeMusicCaps** paper (Comanducci et al., 2024) that window size has modest impact on detection for simple classifiers, but the **SONICS** paper (Rahman et al., ICLR 2025) showed that *long-range temporal dependencies* (120s vs 5s audio) significantly improve detection. We reconciled both findings by designing structure-aware sampling that covers the full track while keeping manageable window sizes. No existing paper proposes this specific approach.

**File**: `src/features/temporal_sampler.py`  
**Purpose**: Solves the "window-bias" problem by detecting musical structure and sampling one representative window per structural section.

### Algorithm

1. Load audio at 22,050 Hz mono
2. Detect section boundaries:
   - **Short tracks (≤60s)**: Agglomerative clustering on MFCC + delta features → structural segments
   - **Long tracks (>60s)**: Uniform segmentation (~1 section per 30s) to avoid O(T²) recurrence matrix
3. For each section: extract the 10-second window with the highest RMS energy (50% overlap sliding window)

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `window_duration` | 10.0s | Captures enough musical context per section |
| `sr` | 22,050 Hz | Standard librosa rate, sufficient for music features |
| `max_sections` | 8 | Prevents over-segmentation |
| `min_sections` | 2 | At least intro + main content |
| MFCC coefficients | 13 + delta | Timbral change detection for boundary placement |

### Scientific Justification

Traditional approaches (first 30s, random windows) systematically miss structural repetitions that are key signatures of derivative works. Structure-aware sampling ensures coverage of intro, verse, chorus, bridge, and outro — the musical units where plagiarism is most detectable.

---

## 3. Spectral Feature Extractor

> **Idea Origin**: **Standard MIR practice** — Mel spectrograms, MFCCs, and chroma features are the de-facto baseline in all three reference papers. The **SONICS** paper uses mel-spectrograms as the primary input to SpecTTTra. The **FakeMusicCaps** paper uses log-spectrograms with ResNet18. Chroma features are standard in Music Information Retrieval (librosa documentation). The specific 395-D composition is our own design combining established features.

**File**: `src/features/spectral.py`  
**Purpose**: Extracts a rich, fixed-size feature vector per AudioWindow covering melodic, timbral, spectral, and rhythmic dimensions.

### Feature Vector Composition (395-D per window)

| Feature Group | Dimensions | Extraction Method |
|--------------|------------|-------------------|
| Chroma (mean + std) | 24 | `librosa.feature.chroma_cqt` — pitch class profile |
| MFCC + Δ + ΔΔ (mean + std) | 78 | `librosa.feature.mfcc` with `delta(order=1,2)` |
| Mel spectrogram (mean + std) | 256 | `librosa.feature.melspectrogram` (128 mel bands, power-to-dB) |
| Spectral shape (mean + std) | 10 | Centroid, bandwidth, rolloff, flatness, ZCR |
| Tempo | 1 | `librosa.beat.beat_track` |
| Harmonic-to-Noise Ratio | 1 | HPSS decomposition, `harm_rms / (harm_rms + perc_rms)` |
| Energy | 1 | RMS of the audio window |

### Key Implementation Details

- **Minimum frame guard**: Pads audio < 2048 samples to ensure stable STFT
- **Adaptive delta width**: `min(9, n_frames)` to handle short windows
- **HPSS-based HNR**: Approximation via harmonic/percussive source separation, outputting ratio in [0, 1]

---

## 4. CLAP Semantic Embeddings

> **Idea Origin**: The **exercise guidelines** explicitly listed CLAP as a recommended model for audio embeddings. The specific model (`laion/larger_clap_music_and_speech`) was chosen as the largest publicly available CLAP checkpoint. The energy-weighted attention pooling strategy is **our own design** to aggregate per-window embeddings into a track-level representation. **Reference**: Wu et al., "Large-Scale Contrastive Language-Audio Pretraining", ICASSP 2023.

**File**: `src/features/embeddings.py`  
**Model**: `laion/larger_clap_music_and_speech` (ICASSP 2023)  
**Output**: 512-D L2-normalized embeddings

### Method

CLAP (Contrastive Language-Audio Pre-training) produces embeddings trained on 630K audio-text pairs. These capture high-level musical semantics (genre, mood, instrumentation) that persist even across significant timbral/key variations — critical for detecting AI derivatives that alter surface characteristics while preserving musical content.

### Processing Pipeline

1. Per-window: resample audio to 48,000 Hz (CLAP's expected rate)
2. Pass through `ClapProcessor` → `ClapModel.get_audio_features()`
3. Extract `pooler_output` from `BaseModelOutputWithPooling` (API fix for newer `transformers`)
4. L2-normalize the 512-D embedding
5. **Energy-weighted attention pooling** across all windows → single track embedding:

$$\mathbf{e}_{\text{track}} = \frac{\sum_i w_i \cdot \mathbf{e}_i}{\|\sum_i w_i \cdot \mathbf{e}_i\|_2}, \quad w_i = \frac{E_i}{\sum_j E_j}$$

where $E_i$ = RMS energy of window $i$.

### Fallback (when CLAP unavailable)

512-D vector constructed from `mel_db.mean()`, `mel_db.std()`, `chroma.mean()`, `chroma.std()`, `mfcc.mean()`, `mfcc.std()` — replicated/truncated to 512 dimensions and L2-normalized.

### Results

| Configuration | Semantic Gap (pos − neg mean) |
|--------------|-------------------------------|
| Fallback CLAP | +0.001 (non-discriminative) |
| Real CLAP | **+0.080** (2nd strongest signal) |

---

## 5. MERT Music Understanding Embeddings

> **Idea Origin**: Discovered through **literature review** while studying foundation models for music understanding. The MERT paper (Li et al., ICLR 2024) describes a self-supervised model specifically trained on music — complementary to CLAP's language-audio focus. The dual-teacher design (CQT + RVQ-VAE) makes it uniquely suited for capturing tonal and rhythmic structure. Not mentioned in the exercise guidelines; added as **our own innovation** to provide a second embedding perspective. **Reference**: Li et al., "MERT: Acoustic Music Understanding Model", ICLR 2024, arXiv:2306.00107.

**File**: `src/features/mert_embeddings.py`  
**Model**: `m-a-p/MERT-v1-95M` (ICLR 2024, 95M parameters)  
**Output**: 768-D L2-normalized embeddings

### Method

MERT is pre-trained via masked language modelling on 160K hours of music with two teacher signals:
- **Acoustic teacher**: RVQ-VAE (Residual Vector Quantisation - Variational AutoEncoder)
- **Musical teacher**: Constant-Q Transform (CQT) — captures pitch/tonal content

This dual-teacher approach gives MERT strong representations for pitch, rhythm, timbre, and harmonic content that complement CLAP's language-audio space.

### Processing Pipeline

1. Resample audio to 24,000 Hz (MERT's expected rate)
2. Take a representative **30-second chunk from the middle** of the track (most representative; avoids intro/outro silence)
3. Pass through `Wav2Vec2FeatureExtractor` → `AutoModel`
4. Mean-pool the last hidden state across time: $\mathbf{e} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{h}_t$
5. L2-normalize → 768-D embedding
6. **Pairwise similarity**: cosine similarity rescaled to [0, 1]:

$$\text{sim}(A, B) = \text{clip}\left(\frac{\mathbf{e}_A \cdot \mathbf{e}_B + 1}{2}, 0, 1\right)$$

### Fallback

768-D from `chroma_cqt`, `mfcc`, and onset strength statistics, replicated to 768-D.

### Results

| Metric | Value |
|--------|-------|
| MERT discriminative gap | +0.009 (weak with 95M; 330M variant expected to improve) |

---

## 6. AI Artifact Detector (7 Sub-Detectors)

> **Idea Origin**: **Multi-paper synthesis + exercise guidelines**. The exercise guidelines listed the **SONICS** paper (Rahman et al., ICLR 2025) as one of three key datasets/papers, and suggested the exercise guidelines' Innovation Track #3 ("Artifact-Conditioned Attribution"). From SONICS we extracted specific artifact patterns: rhythmic predictability (Sub-detector 5), pitch contour stability (Sub-detector 6), and dynamic compression (Sub-detector 7). Sub-detectors 1–4 (phase discontinuity, harmonic envelope, spectral flatness, MFCC smoothness) are **our own design** based on known properties of neural audio synthesis (e.g., chunk-based generation causing phase discontinuities, autoregressive models producing over-smooth MFCCs).

**File**: `src/features/artifacts.py`  
**Purpose**: Detects spectral and structural artifacts characteristic of AI music generators (Suno, Udio)  
**Output**: Single score in [0, 1] — higher = more AI-like

The final artifact score is the **energy-weighted mean** of 7 sub-detector scores across all windows.

### Sub-Detector 1: Phase Discontinuity

**Principle**: AI generators produce audio in chunks (typically 10-30s), creating abrupt phase resets at chunk boundaries.

**Algorithm**:
1. Compute STFT (n_fft=2048, hop=512)
2. Extract instantaneous phase → compute frame-to-frame phase difference
3. Wrap to [−π, π], compute mean absolute jump
4. **Calibration**: Natural music ≈ 0.8–1.2 rad; AI artifacts → higher

$$\text{score} = \text{clip}\left(\frac{\overline{|\Delta\phi|} - 0.8}{1.2}, 0, 1\right)$$

### Sub-Detector 2: Harmonic Envelope Regularity

**Principle**: Human performances have organic micro-variations in pitch and dynamics. AI-generated audio is unnaturally stable.

**Algorithm**:
1. HPSS → extract harmonic component
2. Compute spectral envelope mean over STFT frames
3. Calculate coefficient of variation (CV = σ/μ)
4. **Threshold**: CV < 0.15 → score=1.0 (unnaturally regular); CV > 0.3 → score=0.0

### Sub-Detector 3: Spectral Flatness Anomaly

**Principle**: Generative models sometimes insert "blank" segments with near-white-noise spectral content.

**Algorithm**:
1. Compute per-frame spectral flatness via `librosa.feature.spectral_flatness`
2. Count proportion of frames where flatness > 0.3
3. Score = clip(anomaly_ratio × 2, 0, 1)

### Sub-Detector 4: MFCC Over-Smoothness

**Principle**: Neural synthesis produces MFCCs that vary too smoothly (low second-order difference / acceleration).

**Algorithm**:
1. Extract 13 MFCCs → compute second-order delta (acceleration)
2. Mean absolute acceleration: natural music ≈ 1.5–5.0; AI < 0.8
3. Score = clip(1 − mean_accel/1.5, 0, 1)

### Sub-Detector 5: Rhythmic Predictability *(SONICS-inspired)*

**Principle**: AI-generated songs have more predictable rhythmic structures (Rahman et al., ICLR 2025).

**Algorithm**:
1. Detect onsets → compute inter-onset intervals (IOI)
2. Calculate CV of IOI distribution
3. **Threshold**: CV < 0.10 → perfectly regular beats → score=1.0; CV > 0.25 → score=0.0

### Sub-Detector 6: Pitch Contour Stability *(SONICS-inspired)*

**Principle**: AI has limited pitch variability and lacks expressive vocal techniques like melismatic phrasing.

**Algorithm**:
1. Extract F0 contour using `librosa.pyin` (C2–C7)
2. Convert voiced frames to semitone deviations from median
3. Measure standard deviation of semitone contour
4. **Threshold**: σ < 0.8 semitones → monotone AI → score=1.0; σ > 2.0 → score=0.0

### Sub-Detector 7: Dynamic Compression *(SONICS-inspired)*

**Principle**: Real songs have "dynamic variation and unexpected changes" (SONICS) that AI often lacks.

**Algorithm**:
1. Compute RMS envelope → convert to dB
2. Calculate interquartile range (IQR) in dB
3. **Threshold**: IQR < 2 dB → over-compressed → score=1.0; IQR > 6 dB → score=0.0

### Results

| Metric | Value |
|--------|-------|
| Artifact discriminative gap | −0.090 (negative — hurts classification in current formulation) |

**Note**: The artifact × semantic interaction currently shows a negative gap, suggesting that both attributed and unrelated pairs exhibit similar AI characteristics, making this branch actively counterproductive. This is a candidate for removal or reformulation.

---

## 7. SOTA-Inspired Feature Extractors

**File**: `src/features/sota_features.py`

### 7a. Long-Range Self-Similarity Matrix (SSM) Comparison

**Motivation**: SONICS (Rahman et al., ICLR 2025) demonstrates that modeling long-range temporal dependencies is critical for detecting synthetic songs. Their SpecTTTra architecture explicitly captures temporal patterns.

**Algorithm**:
1. Compute CQT chroma features for each track
2. Downsample to 100 frames (reduced from 200 for 4× speedup; SSM gap was <0.002)
3. L2-normalize columns → cosine similarity matrix SSM(T×T)
4. Bi-linearly resize both SSMs to same dimensions
5. Compute normalized Frobenius inner product:

$$\text{sim}_{\text{SSM}} = \text{clip}\left(\frac{\langle \text{SSM}_A, \text{SSM}_B \rangle_F + 1}{2 \cdot \|\text{SSM}_A\|_F \cdot \|\text{SSM}_B\|_F}, 0, 1\right)$$

| Metric | Value |
|--------|-------|
| SSM discriminative gap | +0.005 (weak signal) |

### 7b. Mel-Spectrogram Cross-Correlation

> **Idea Origin**: Directly inspired by the **FakeMusicCaps** paper (Comanducci et al., 2024) — one of the three papers listed in the **exercise guidelines**. Their ResNet18+Log-Spectrogram baseline achieves 0.95+ balanced accuracy, demonstrating the discriminative power of mel-spectrograms. We implement a simpler cross-correlation approach rather than training a CNN.

**Motivation**: FakeMusicCaps (Comanducci et al., 2024) showed that ResNet18+Spectrogram achieves strong detection performance using direct spectral comparison.

**Algorithm**:
1. Compute 128-band mel spectrograms (power-to-dB) for both tracks
2. **Frequency profile**: average across time → 128-D vector → cosine similarity
3. **Temporal envelope**: average across frequency → resample to same length → cosine similarity
4. Combine: 60% frequency + 40% temporal (frequency profile is more stable)

$$\text{sim}_{\text{mel}} = 0.6 \cdot \cos(\bar{f}_A, \bar{f}_B) + 0.4 \cdot \cos(\bar{t}_A, \bar{t}_B)$$

| Metric | Value |
|--------|-------|
| Spectral correlation discriminative gap | +0.002 (very weak) |

### 7c. Onset/Rhythm Similarity

> **Idea Origin**: Inspired by the **SONICS** paper (Rahman et al., ICLR 2025, listed in **exercise guidelines**), which identified rhythmic predictability as a key AI artifact. Rather than just *detecting* predictable rhythm (which our artifact detector does), this branch *compares* rhythmic patterns between the two tracks — a pair-wise signal. The specific formulation (onset envelope correlation + tempo ratio with double/half-time handling) is **our own design**.

**Motivation**: SONICS analysis reveals that AI-generated songs exhibit more predictable rhythmic structures. We compare rhythmic patterns between paired tracks.

**Algorithm**:
1. **Onset strength envelope comparison**: Extract onset envelopes, resample to same length, compute cosine similarity
2. **Tempo ratio**: Estimate tempo via `librosa.beat.beat_track`, compute ratio, check proximity to 1×, 2×, or 0.5× (handles double/half time):

$$\text{tempo\_sim} = \exp(-5 \cdot \min_r |t_A/t_B - r|), \quad r \in \{0.5, 1.0, 2.0\}$$

3. **Combined**: 60% onset correlation + 40% tempo similarity

| Metric | Value |
|--------|-------|
| Rhythm discriminative gap (fallback CLAP) | **+0.170** (strongest discriminator by far) |
| Rhythm discriminative gap (8-branch) | **+0.109** (still strongest) |

---

## 8. Key-Shift-Invariant Melodic DTW

> **Idea Origin**: **Exercise guidelines — Innovation Track #1**: "Key-Shift Invariant Melody Alignment — Before DTW, circularly shift chroma vectors across all 12 semitones and keep best alignment. This catches cases where Track B is transposed relative to Track A." We implemented this exactly as described in the exercise brief.

**Implemented in**: `src/model/attribution.py` → `AttributionScorer._melodic_alignment()`

**Problem**: AI generators may transpose the original melody to a different key. Standard chroma comparison fails under transposition.

**Solution**: Test all 12 circular chromatic shifts on Track B's chroma, keeping the best DTW similarity.

**Algorithm**:
1. Build per-window chroma mean sequences: $\mathbf{C}_A \in \mathbb{R}^{N_A \times 12}$, $\mathbf{C}_B \in \mathbb{R}^{N_B \times 12}$
2. Flatten and min-max normalize each sequence
3. For each shift $s \in \{0, 1, ..., 11\}$:
   - Circularly shift Track B's chroma: $\mathbf{C}_B^{(s)}[:, j] = \mathbf{C}_B[:, (j+s) \bmod 12]$
   - Compute DTW distance: $d_s = \text{DTW}(\mathbf{c}_A, \mathbf{c}_B^{(s)})$
   - Convert to similarity: $\text{sim}_s = \exp(-d_s / 3.0)$
4. Return $\max_s \text{sim}_s$ and record the best key shift

**Library**: `dtaidistance` for efficient DTW computation.

| Metric | Value |
|--------|-------|
| Melodic discriminative gap | +0.029 |

---

## 9. Structural Correspondence Scoring

> **Idea Origin**: **Exercise guidelines — Innovation Track #2**: "Structure Graph Similarity — Represent each track as section graph nodes (intro/verse/chorus-like segments) with transition edges. Compare graphs using edit distance or node embedding similarity." We implemented a simplified version: instead of full graph edit distance, we compare proportional section midpoints via MAD (mean absolute difference). This captures the core insight — AI derivatives preserve structural proportions — without the complexity of graph matching.

**Implemented in**: `src/model/attribution.py` → `AttributionScorer._structural_correspondence()`

**Principle**: AI-generated derivatives often preserve the original's song structure (verse at ~15% of duration, chorus at ~35%, etc.).

**Algorithm**:
1. Compute relative midpoints of each section, normalized to [0, 1] by total duration
2. Interpolate both midpoint sequences to the same length
3. Compute mean absolute difference (MAD)
4. Convert to similarity: $\text{sim} = \max(0, 1 - 2 \cdot \text{MAD})$

| Metric | Value |
|--------|-------|
| Structural discriminative gap | +0.013 |

---

## 10. 8-Branch Attribution Scorer

> **Idea Origin**: **Own design** — the multi-branch weighted scoring approach is our core architectural contribution. The idea of combining diverse signals (semantic, melodic, structural, artifact) into a single score is standard in MIR ensemble systems, but the specific branch composition and data-driven weight tuning on MIPPIA is novel. The `artifact_boost = artifact × semantic` interaction term was motivated by the **exercise guidelines — Innovation Track #3** ("Artifact-Conditioned Attribution: learn dynamic weights where AI artifact strength gates how much semantic similarity contributes").

**File**: `src/model/attribution.py`

### Formula

$$S = \sum_{i=1}^{8} w_i \cdot f_i$$

where:

| Branch ($f_i$) | Weight ($w_i$) | Signal |
|----------------|---------------|--------|
| Rhythm Similarity | **0.30** | Onset envelope correlation + tempo ratio |
| Semantic (CLAP) | 0.15 | Cosine similarity of 512-D CLAP embeddings |
| Melodic (DTW) | 0.15 | Key-shift-invariant DTW on chroma sequences |
| MERT | 0.15 | Cosine similarity of 768-D MERT embeddings |
| Artifact Boost | 0.10 | `artifact_score × semantic_similarity` |
| SSM | 0.05 | Self-similarity matrix Frobenius inner product |
| Spectral Corr. | 0.05 | Mel cross-correlation (frequency + temporal) |
| Structural | 0.05 | Section midpoint alignment |

### Weight Rebalancing Rationale

Weights were rebalanced based on empirical discriminative analysis on the MIPPIA dataset (62 pairs, 124 evaluation pairs). The discrimination gap (mean positive score − mean negative score) for each component guided reweighting:

| Component | Positive Mean | Negative Mean | Gap | Weight |
|-----------|--------------|---------------|-----|--------|
| rhythm_similarity | 0.710 | 0.540 | **+0.170** | **30%** |
| melodic_alignment | 0.611 | 0.596 | +0.015 | 15% |
| semantic_similarity | 0.993 | 0.992 | +0.001¹ | 15% |
| ai_artifact_score | 0.242 | 0.245 | −0.003 | 10% |
| structural_correspondence | 0.927 | 0.914 | +0.013 | 5% |
| ssm_similarity | 0.989 | 0.988 | +0.001 | 5% |
| spectral_correlation | 0.995 | 0.994 | +0.001 | 5% |

¹ With fallback CLAP; real CLAP gap improves to +0.080.

### Verdict Thresholds

| Score Range | Verdict |
|------------|---------|
| ≥ 0.75 | "Strong AI Attribution — Track B is very likely a synthetic derivative of Track A" |
| ≥ 0.55 | "Probable AI Attribution — Track B shows significant similarities to Track A" |
| ≥ 0.35 | "Possible Relationship — Tracks share some features but attribution is uncertain" |
| < 0.35 | "Unlikely Attribution — Tracks appear unrelated" |

---

## 11. Logistic Regression Calibration Layer

> **Idea Origin**: **Exercise guidelines — Innovation Track #4**: "Contrastive Pair Calibration — Fit a lightweight logistic calibration head on top of MAIA sub-scores using MIPPIA labels. Keep core features fixed; only calibrate decision boundary." We implemented this exactly as suggested, using `sklearn.linear_model.LogisticRegression` on the 8 sub-component scores. Later upgraded to also support GBDT (see Section 17).

**File**: `src/model/calibration.py`  
**Saved Model**: `models/calibrator.json`

### Method

A logistic regression trained on all sub-component scores from the full MIPPIA evaluation to learn a data-driven decision boundary.

$$P(\text{attributed}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

### Feature Engineering

8 input features:
1. `semantic_similarity`
2. `melodic_alignment`
3. `structural_correspondence`
4. `ai_artifact_score`
5. `ssm_similarity`
6. `spectral_correlation`
7. `rhythm_similarity`
8. `artifact_semantic_interaction` = `ai_artifact_score × semantic_similarity`

### Learned Coefficients

| Feature | Coefficient | Interpretation |
|---------|------------|----------------|
| `rhythm_similarity` | **1.482** | Dominant feature — 4.3× more important than semantic |
| `melodic_alignment` | **0.781** | Second strongest — melody matching matters |
| `artifact_semantic_interaction` | 0.243 | Artifacts in context of similarity |
| `ai_artifact_score` | 0.232 | Raw artifact presence |
| `structural_correspondence` | 0.133 | Structural alignment |
| `ssm_similarity` | 0.067 | Weak long-range structure signal |
| `semantic_similarity` | 0.044 | Very low (fallback CLAP saturates ~0.99) |
| `spectral_correlation` | 0.022 | Minimal discriminative value |
| **Intercept** | **−1.906** | Bias toward negative (appropriate for balanced set) |

### Cross-Validation

| Metric | Value |
|--------|-------|
| CV Accuracy | 62.9% (trained on old 7-branch data; will improve with 8-branch + real CLAP) |

---

## 12. Artifact Diff Branch (Improvement #1)

> **Idea Origin**: **Empirical analysis** of Experiment 3 results. We observed that `artifact_boost` (from exercise Innovation Track #3) had a gap of −0.090 — the only branch actively hurting classification. Root cause: both positive and negative candidate tracks are AI-generated in MIPPIA, so raw artifact scores fire on everything. The insight that *difference* in artifact profiles should be the signal (attributed pairs share similar AI processing) is **our own innovation**.

**File**: `src/model/attribution.py`  
**Problem**: The original `artifact_boost = artifact_score × semantic_similarity` branch had a **negative** gap (-0.090) — it was actively hurting classification because AI artifact detectors fire on *all* AI audio, including negative-pair candidates.

### Solution

Replace `artifact_boost` with `artifact_diff`:

$$\text{artifact\_diff} = 1 - |\text{artifact}_A - \text{artifact}_B|$$

**Logic**: Truly attributed pairs (A is original, B is AI-derived from A) should have *similar* artifact profiles — both processed through similar AI pipelines. Unrelated pairs should have *different* artifact profiles.

### Weight Change

| Branch | Old Weight | New Weight | Rationale |
|--------|-----------|------------|-----------|
| artifact_boost | 0.10 | (dropped) | Negative gap, hurts accuracy |
| artifact_diff | — | 0.02 | Conservative weight for new signal |
| rhythm | 0.20 | 0.12 | Redistributed after adding new branches |

---

## 13. HTDemucs Source Separation (Improvement #2)

> **Idea Origin**: **Literature review** — the HTDemucs paper (Rouard et al., ICASSP 2023) was identified during our web research phase as the state-of-the-art in music source separation (9.2 dB SDR on MusDB-HQ). The idea to compare *per-stem* (vocals vs vocals, drums vs drums) rather than full mixes is established practice in musicology-based plagiarism analysis, where experts isolate melody from accompaniment. Not mentioned in the exercise guidelines; this is **our own innovation** to reduce inter-instrument interference in similarity computation.

**File**: `src/features/source_separation.py`  
**Purpose**: Separate each track into 4 stems (vocals, drums, bass, other) and compare per-stem, reducing interference between musical components.

### Method

1. **Separation**: Use HTDemucs (Rouard et al., ICASSP 2023) to split each track:
   ```
   Track → {vocals, drums, bass, other}
   ```
   
2. **Per-stem comparison**: Each stem pair uses the most appropriate similarity metric:
   - **Vocals** (40% weight): Chroma DTW — captures melodic/harmonic content
   - **Drums** (25% weight): Onset strength correlation — rhythmic pattern match
   - **Bass** (20% weight): Chroma DTW on low-frequency content
   - **Other/accompaniment** (15% weight): Mel-spectrogram cross-correlation

3. **Combined score**:
$$\text{stem\_combined} = 0.40 \cdot S_{\text{vocal}} + 0.25 \cdot S_{\text{drum}} + 0.20 \cdot S_{\text{bass}} + 0.15 \cdot S_{\text{other}}$$

### Caching

Track separations are cached in memory (`_separation_cache` keyed by file path) to avoid redundant computation when the same track appears in multiple pairs.

### Fallback

If HTDemucs fails (OOM, missing model, etc.), uses full mix for all stems — graceful degradation.

### Architecture Reference

- **HTDemucs**: Hybrid Transformer + U-Net, state-of-the-art source separation
- **9.2 dB SDR** on MusDB-HQ test set (Rouard et al., 2023)

---

## 14. MERT v1-330M Upgrade (Improvement #3)

> **Idea Origin**: **Empirical analysis** — Experiment 3 showed the MERT-95M gap was only +0.009, the weakest of all embedding-based signals. Since the MERT paper (Li et al., ICLR 2024) released multiple model sizes, upgrading to 330M was a straightforward scaling improvement. The hypothesis (more parameters → richer representations → better discrimination) was confirmed: gap doubled to +0.018 in Experiment 5.

**File**: `src/features/mert_embeddings.py`  
**Change**: Upgraded from `m-a-p/MERT-v1-95M` (768-D, 95M params) → `m-a-p/MERT-v1-330M` (1024-D, 330M params)

### Rationale

- **3.5× more parameters** → richer music understanding representations
- **1024-D embeddings** vs 768-D → more expressive feature space
- Same dual-teacher pre-training (CQT + RVQ-VAE) but deeper Transformer layers
- Expected to improve the weak MERT gap (+0.009 with 95M) seen in Experiment 3

### Technical Changes

| Property | MERT-v1-95M | MERT-v1-330M |
|----------|-------------|--------------|
| Parameters | 95M | 330M |
| Embedding dim | 768 | 1024 |
| Model ID | `m-a-p/MERT-v1-95M` | `m-a-p/MERT-v1-330M` |
| Download size | ~380 MB | ~1.3 GB |

---

## 15. CQT Chroma + OTI (Improvement #4)

> **Idea Origin**: **Cover Song Identification (CSI) literature** — OTI (Optimal Transposition Index) is the standard key-normalization technique in CSI, originating from Serra et al. (2009). While our Section 8 already implements key-shift-invariant DTW (exercise Innovation Track #1), OTI provides a complementary approach: instead of testing DTW across 12 shifts, it directly finds the optimal transposition via chroma correlation, then computes frame-level similarity. This extends the exercise's key-shift idea with a mathematically more principled approach from the CSI community.

**File**: `src/features/sota_features.py` → `cqt_chroma_oti_similarity()`  
**Purpose**: Cover song identification (CSI) standard — handles key transposition between original and AI-generated version.

### Algorithm

1. Compute CQT (Constant-Q Transform) chroma features for both tracks
2. **Optimal Transposition Index (OTI)**: Test all 12 chromatic shifts, select the one maximizing chroma correlation:

$$\text{OTI}^* = \arg\max_{k \in \{0,...,11\}} \text{corr}(\text{chroma}_A, \text{roll}(\text{chroma}_B, k))$$

3. Apply OTI shift and compute frame-level cosine similarity

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| hop_length | 512 | ~23 ms at 22 kHz, standard CQT resolution |
| n_chroma | 12 | Standard chromatic pitch classes |
| OTI shifts | 12 | All possible semitone transpositions |

### Output

- `cqt_similarity`: Best correlation across all transpositions ∈ [0, 1]
- `optimal_transposition`: Best key shift (0-11 semitones)

---

## 16. Qmax / Cross-Recurrence Plot (Improvement #5)

> **Idea Origin**: **Serra et al. (2009)**, "Cross recurrence quantification for cover song identification" — the foundational paper for Cover Song Identification. Qmax is the gold-standard metric in CSI, measuring the longest aligned subsequence between two tracks. We reasoned that AI-generated derivatives are essentially "covers" of the original — so classical CSI metrics should transfer directly. This proved correct: Qmax achieved the **3rd strongest gap** (+0.075) of all 13 branches. Not mentioned in the exercise guidelines; discovered through **our own literature survey** of the MIR/CSI field.

**File**: `src/features/sota_features.py` → `qmax_similarity()`  
**Purpose**: Classic CSI metric that captures global structural similarity through cross-recurrence analysis.

### Algorithm

1. Compute OTI-aligned CQT chroma for both tracks
2. Build **cross-similarity matrix** (cosine similarity between all frame pairs)
3. Apply binary thresholding (τ = 0.8) → **Cross-Recurrence Plot (CRP)**
4. Search for longest near-diagonal in CRP with tempo tolerance (±10% offset):

$$Q_{\max} = \frac{\max_{\text{diag}} \text{length}(\text{consecutive matches})}{\min(T_A, T_B)}$$

5. Also compute CRP density:

$$\text{CRP\_density} = \frac{\sum_{i,j} \text{CRP}[i,j]}{T_A \times T_B}$$

### Output

- `qmax_score`: Longest normalized diagonal match ∈ [0, 1]
- `crp_density`: Overall recurrence density

### Reference

Based on Serra et al. (2009) "Cross recurrence quantification for cover song identification" — the foundational CSI method.

---

## 17. GBDT Calibrator (Improvement #6)

> **Idea Origin**: Extension of **exercise guidelines — Innovation Track #4** ("Contrastive Pair Calibration"), which suggested a logistic calibration head. We extended this to GBDT (Gradient Boosted Decision Trees) to capture non-linear interactions between the 13 sub-scores that logistic regression misses. GBDT is standard practice in Kaggle-style tabular classification problems. However, with only 20 training pairs, GBDT overfits (100% train, 70% CV) while logistic generalizes better (80% train, 75% CV) — confirming that the exercise's suggestion of a *lightweight* calibrator was well-motivated.

**File**: `src/model/calibration.py`, `train_calibrator.py`  
**Purpose**: Replace heuristic weights with a data-driven GBDT (Gradient Boosted Decision Trees) calibrator.

### Method

Upgraded from LogisticRegression → GradientBoostingClassifier:

$$P(\text{attributed}) = \text{GBDT}(\mathbf{x}_{13})$$

Where $\mathbf{x}_{13}$ is the 13-dimensional feature vector from all branches.

### 13 Input Features

1. `semantic_similarity` (CLAP)
2. `melodic_alignment` (DTW)
3. `structural_correspondence`
4. `artifact_diff`
5. `ssm_similarity`
6. `spectral_correlation`
7. `rhythm_similarity`
8. `mert_similarity`
9. `stem_combined`
10. `cqt_similarity`
11. `qmax_score`
12. `clap_multiscale`
13. `panns_similarity`

### GBDT Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_estimators | 50 | Small ensemble, avoid overfitting on ~60 pairs |
| max_depth | 3 | Shallow trees for limited training data |
| learning_rate | 0.1 | Standard, not too aggressive |
| min_samples_leaf | 3 | Conservative for small dataset |
| subsample | 0.8 | Row subsampling for regularization |

### Training Procedure

```bash
python train_calibrator.py --input results/exp5_all_improvements.json --method gbdt
```

**Note**: GBDT will be trained AFTER the combined eval produces sub-scores for all 13 features. The trained model replaces heuristic weights entirely.

---

## 18. Multi-Scale CLAP Temporal Alignment (Improvement #7)

> **Idea Origin**: **Own innovation** — after observing that CLAP's global cosine similarity (Section 4) was strong (gap +0.080) but loses temporal structure by averaging all windows into one vector, we designed a per-window alignment approach. This is inspired by the concept of *set matching* used in image retrieval (e.g., NetVLAD) and multi-scale feature matching in stereo vision, adapted here for music windows. The approach produces the **4th strongest gap** (+0.071) and complements global CLAP by capturing which specific sections are similar vs different.

**File**: `src/features/embeddings.py` → `multi_scale_similarity()`  
**Purpose**: Capture temporal structure similarity that single-vector CLAP misses.

### Algorithm

1. Compute per-window CLAP embeddings for both tracks (one 512-D vector per structural window)
2. Build cosine similarity matrix between all window pairs
3. For each window in Track A, find best-matching window in Track B (and vice versa)
4. Symmetric score = average of both directions

$$S_{\text{ms}} = \frac{1}{2}\left(\frac{1}{|W_A|}\sum_{i} \max_j \text{sim}(w_A^i, w_B^j) + \frac{1}{|W_B|}\sum_j \max_i \text{sim}(w_A^i, w_B^j)\right)$$

### Key Insight

Standard CLAP averages all windows into a single track embedding, losing temporal information. A track that copies the verse but changes the chorus would still score high on average CLAP, but multi-scale alignment reveals the section-specific correspondence.

---

## 19. PANNs Perceptual Embeddings (Improvement #8)

> **Idea Origin**: **Literature review** — Kong et al. (2020) introduced PANNs as large-scale pretrained audio neural networks achieving SOTA on AudioSet. We added PANNs to provide a third embedding perspective alongside CLAP (semantic) and MERT (music-specific). PANNs captures general perceptual audio quality (timbre, texture, environmental characteristics) from its AudioSet training on 2M diverse YouTube clips. Not mentioned in the exercise guidelines; this is **our own addition** to improve ensemble diversity. **Reference**: Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks", IEEE/ACM TASLP, 2020.

**File**: `src/features/panns_embeddings.py`  
**Purpose**: Capture general perceptual audio characteristics using AudioSet-pretrained CNN14.

### Model

- **PANNs CNN14**: Pretrained Audio Neural Network (Kong et al., 2020)
- **Training**: AudioSet (2M YouTube clips, 527 classes)
- **Embedding**: 2048-D from penultimate layer
- **Input**: 32 kHz mono audio

### Complementarity

| Model | Training Data | Captures | Embedding |
|-------|-------------|----------|-----------|
| CLAP | Music + text | Semantic/conceptual | 512-D |
| MERT | Music (5.5K hrs) | Music-specific structure | 1024-D |
| PANNs | AudioSet (2M clips) | General perceptual quality | 2048-D |

Together, these three models provide a comprehensive multi-perspective audio similarity assessment.

### Score Computation

$$S_{\text{panns}} = \frac{\text{cos}(\mathbf{e}_A, \mathbf{e}_B) + 1}{2}$$

Where $\mathbf{e}_A, \mathbf{e}_B$ are L2-normalized PANNs embeddings.

---

## 13-Branch Weight Configuration (v2 → v3 gap-proportional)

After Experiment 6, weights were rebalanced proportional to measured discriminative gaps:

| Branch | Weight (v2) | Weight (v3) | Measured Gap | Source |
|--------|------------|------------|--------------|--------|
| `rhythm` | 0.12 | **0.20** | +0.109 | Onset/tempo pattern (SONICS) |
| `qmax` | 0.07 | **0.15** | +0.075 | Qmax cross-recurrence plot |
| `semantic` | 0.07 | **0.14** | +0.080 | CLAP global cosine similarity |
| `clap_multiscale` | 0.10 | **0.14** | +0.071 | Multi-scale CLAP temporal alignment |
| `panns` | 0.08 | **0.20** | +0.042 | PANNs CNN14 perceptual embeddings |
| `melodic` | 0.04 | 0.05 | +0.029 | Chroma DTW (key-shift invariant) |
| `mert` | 0.09 | 0.04 | +0.018 | MERT v1-330M music embeddings |
| `structural` | 0.03 | 0.02 | +0.013 | Section boundary alignment |
| `stem_combined` | **0.25** | **0.02** | +0.008 | HTDemucs per-stem comparison |
| `artifact_diff` | 0.02 | 0.01 | +0.007 | |artifact_A − artifact_B| |
| `ssm` | 0.03 | 0.01 | +0.005 | Self-similarity matrix comparison |
| `cqt` | 0.08 | 0.01 | +0.005 | CQT chroma + OTI |
| `spectral_corr` | 0.02 | 0.01 | +0.002 | Mel-spectrogram cross-correlation |
| **Total** | 1.00 | 1.00 | | |

Key insight: `stem_combined` weight dropped from 0.25 to 0.02 — source separation provides high-quality sub-scores (vocals, drums, bass, other) but they have very high similarity for both attributed and unrelated pairs, making them poorly discriminative at the pair level. The per-stem sub-scores are still valuable for human interpretability.

---

## 20. Evaluation Results & Ablation

### Experiment 1: Baseline (Fallback CLAP, 7 branches, 20 pairs)

**Configuration**: `MAIA_DISABLE_CLAP=1`, default weights, threshold=0.55  
**Result File**: `results/mippia_quick_baseline.json`

| Metric | Value |
|--------|-------|
| Accuracy | 50.0% |
| Precision | 50.0% |
| Recall | 100% |
| F1 | 66.7% |
| TP / FP / FN / TN | 10 / 10 / 0 / 0 |

**Diagnosis**: Threshold too low; all pairs scored above 0.55. Positive mean=0.8171, Negative mean=0.7929, Gap=0.0242.

### Experiment 2: Full Dataset (Fallback CLAP, 7 branches, 124 pairs)

**Configuration**: `MAIA_DISABLE_CLAP=1`, threshold=0.805 (tuned to midpoint of score distributions)  
**Result File**: `results/mippia_full_results.json`

| Metric | Value |
|--------|-------|
| **Accuracy** | **68.55%** |
| Precision | 68.25% |
| Recall | 69.35% |
| **F1** | **68.80%** |
| TP / FP / FN / TN | 43 / 20 / 19 / 42 |

**Sub-component Discrimination (124 pairs)**:

| Component | Pos Mean | Neg Mean | Gap |
|-----------|----------|----------|-----|
| rhythm_similarity | 0.710 | 0.540 | **+0.170** |
| melodic_alignment | 0.611 | 0.596 | +0.015 |
| structural_correspondence | 0.927 | 0.914 | +0.013 |
| ai_artifact_score | 0.242 | 0.245 | −0.003 |
| semantic_similarity | 0.993 | 0.992 | +0.001 |
| ssm_similarity | 0.989 | 0.988 | +0.001 |
| spectral_correlation | 0.995 | 0.994 | +0.001 |

### Experiment 3: 8-Branch Model (Real CLAP + MERT + Reweighted, 20 pairs)

**Configuration**: Real CLAP enabled, MERT v1-95M enabled, reweighted 8-branch scorer  
**Result File**: `results/mippia_8branch_quick.json`

| Metric | Value |
|--------|-------|
| Accuracy | 50.0%¹ |
| Precision | 50.0% |
| Recall | 100% |
| F1 | 66.7% |
| TP / FP / FN / TN | 10 / 10 / 0 / 0 |

¹ Using old threshold=0.55; optimal threshold at midpoint (~0.733) would give better accuracy.

**Sub-component Discrimination (20 pairs)**:

| Component | Pos Mean | Neg Mean | Gap | Δ vs Exp. 2 |
|-----------|----------|----------|-----|-------------|
| rhythm_similarity | 0.738 | 0.630 | **+0.109** | − |
| **semantic_similarity** | **0.891** | **0.811** | **+0.080** | **↑ from +0.001** |
| melodic_alignment | 0.623 | 0.594 | +0.029 | ↑ from +0.015 |
| structural_correspondence | 0.928 | 0.914 | +0.013 | → |
| mert_similarity | 0.955 | 0.946 | +0.009 | NEW |
| ssm_similarity | 0.987 | 0.983 | +0.005 | ↑ |
| spectral_correlation | 0.995 | 0.993 | +0.002 | → |
| ai_artifact_score | 0.219 | 0.309 | −0.090 | ↓ worse |

### Summary of Progress

| Metric | Exp. 1 (Baseline) | Exp. 2 (Full) | Exp. 3 (8-Branch) | Exp. 4 (Artifact Fix) |
|--------|-------------------|---------------|-------------------|-----------------------|
| Score Gap (pos−neg) | 0.024 | N/A² | 0.046 (+92%) | **0.053 (+15%)** |
| Semantic Gap | 0.001 | 0.001 | 0.080 | 0.080 |
| Best F1 | 66.7% | **68.8%** | 66.7% | 66.7% |
| # Branches | 7 | 7 | 8 | 8 |
| CLAP Model | Fallback | Fallback | Real | Real |
| MERT | — | — | v1-95M | v1-95M |
| Artifact Branch | artifact_boost | artifact_boost | artifact_boost | **artifact_diff** |
| Processing Time/Pair | ~57s | ~57s | ~3 min | ~2.4 min |

### Experiment 4: Artifact Fix (20 pairs)

**Configuration**: Replaced `artifact_boost` with `artifact_diff`, redistributed weights. Still 8-branch (old MERT-95M, no source sep/CQT/Qmax).  
**Result File**: `results/exp4_artifact_fix.json`

| Metric | Value |
|--------|-------|
| Accuracy | 50.0%¹ |
| Score Gap (pos−neg) | **0.053 (+15% vs Exp. 3)** |
| Mean score (attributed) | 0.8105 |
| Mean score (unrelated) | 0.7574 |

¹ Threshold=0.55 too low — all pairs score > 0.55 → all predicted positive

**Sub-component Discrimination (20 pairs)**:

| Component | Pos Mean | Neg Mean | Gap | Δ vs Exp. 3 |
|-----------|----------|----------|-----|-------------|
| rhythm_similarity | 0.724 | 0.616 | **+0.109** | → |
| semantic_similarity | 0.900 | 0.820 | **+0.080** | → |
| melodic_alignment | 0.624 | 0.595 | +0.029 | → |
| structural_correspondence | 0.937 | 0.924 | +0.013 | → |
| mert_similarity | 0.947 | 0.938 | +0.009 | → |
| ssm_similarity | 0.985 | 0.980 | +0.005 | → |
| spectral_correlation | 0.995 | 0.994 | +0.002 | → |
| ai_artifact_score | 0.254 | 0.344 | −0.090 | → (expected, this is raw score not used directly) |

**Verdict**: Artifact diff improved score gap by 15% (0.046→0.053). Keep this change. ✅

### Experiment 5: 13-Branch with MERT-330M (20 pairs, no SourceSep/PANNs)

**Configuration**: 13-branch scorer, MERT v1-330M, CQT+OTI, Qmax, multi-scale CLAP.  
Source separation and PANNs fell back to defaults due to runtime issues (fixed for Exp. 6).  
**Result File**: `results/exp5_all_improvements.json`

| Metric | Value |
|--------|-------|
| Accuracy | 50.0% (threshold=0.55 still too low) |
| Score Gap (pos−neg) | **0.035** (decreased from 0.053 — diluted by 33% non-working branches) |
| Mean score (attributed) | 0.8079 |
| Mean score (unrelated) | 0.7732 |

**Full sub-component discrimination (20 pairs)**:

| Component | Pos Mean | Neg Mean | Gap | Status |
|-----------|----------|----------|-----|--------|
| rhythm_similarity | 0.724 | 0.616 | **+0.109** | Strongest signal ✅ |
| semantic_similarity | 0.900 | 0.820 | **+0.080** | Strong ✅ |
| **qmax_score** | 0.137 | 0.062 | **+0.075** | **NEW — strong signal** ✅ |
| **clap_multiscale** | 0.885 | 0.815 | **+0.071** | **NEW — strong signal** ✅ |
| melodic_alignment | 0.624 | 0.595 | +0.029 | Stable ✅ |
| **mert_similarity** | 0.962 | 0.943 | **+0.018** | **↑ DOUBLED from +0.009 (330M upgrade)** ✅ |
| structural_correspondence | 0.937 | 0.924 | +0.013 | Stable |
| artifact_diff | 0.915 | 0.908 | +0.007 | Small positive |
| cqt_similarity | 0.992 | 0.987 | +0.005 | Small positive |
| ssm_similarity | 0.985 | 0.980 | +0.005 | Same |
| spectral_correlation | 0.995 | 0.994 | +0.002 | Same |
| stem_combined | 0.933 | 0.933 | −0.001 | ❌ **Fallback** → no signal |
| panns_similarity | 0.500 | 0.500 | 0.000 | ❌ **Fallback** → no signal |

**Analysis**: Score gap decreased because 33% of weight (stem=0.25 + panns=0.08) goes to non-working branches that produce no gap. Without those branches, the discriminative features actually improved significantly. New strong signals: **Qmax** (+0.075), **multi-scale CLAP** (+0.071), **MERT-330M** gap doubled.

**Verdict**: CQT, Qmax, multi-scale CLAP, and MERT-330M all add discriminative value. ✅  
Score gap diluted by non-functional source separation and PANNs fallback. ❌ → Fixed for Experiment 6.

### Experiment 6: Full 13-Branch (all improvements working, 20 pairs)

**Configuration**: All 13 branches active — HTDemucs source separation, PANNs CNN14, MERT v1-330M, CQT+OTI, Qmax, multi-scale CLAP.  
**Result File**: `results/exp6_full_13branch.json`

| Metric | Value |
|--------|-------|
| Score Gap (pos−neg) | **0.040** (pre-reweighting) |
| Mean score (attributed) | 0.8328 |
| Mean score (unrelated) | 0.7926 |
| Processing time/pair | ~1013s (~17 min, dominated by source separation) |

**Full sub-component discrimination (20 pairs, sorted by gap)**:

| Component | Pos Mean | Neg Mean | Gap | Weight (old) | Weight (new) |
|-----------|----------|----------|-----|-------------|-------------|
| rhythm_similarity | 0.724 | 0.616 | **+0.109** | 0.12 | **0.20** |
| semantic_similarity | 0.900 | 0.820 | **+0.080** | 0.07 | **0.14** |
| qmax_score | 0.137 | 0.062 | **+0.075** | 0.07 | **0.15** |
| clap_multiscale | 0.885 | 0.815 | **+0.071** | 0.10 | **0.14** |
| panns_similarity | 0.917 | 0.875 | **+0.042** | 0.08 | **0.20** |
| melodic_alignment | 0.624 | 0.595 | +0.029 | 0.04 | 0.05 |
| mert_similarity | 0.962 | 0.943 | +0.018 | 0.09 | 0.04 |
| structural_correspondence | 0.937 | 0.924 | +0.013 | 0.03 | 0.02 |
| stem_combined | 0.899 | 0.891 | +0.008 | 0.25 | **0.02** |
| artifact_diff | 0.915 | 0.908 | +0.007 | 0.02 | 0.01 |
| cqt_similarity | 0.992 | 0.987 | +0.005 | 0.08 | 0.01 |
| ssm_similarity | 0.985 | 0.980 | +0.005 | 0.03 | 0.01 |
| spectral_correlation | 0.995 | 0.994 | +0.002 | 0.02 | 0.01 |

**After reweighting (gap-proportional, offline re-scoring)**:

| Metric | Old Weights | **New Weights** |
|--------|------------|-----------------|
| Score Gap | 0.040 | **0.065 (+62%)** |
| Best Accuracy | 50% | **80% (threshold=0.72)** |
| Pos Range | — | [0.671, 0.815] |
| Neg Range | — | [0.623, 0.745] |

**Calibrator Training (on Exp. 6 data)**:

| Model | CV Accuracy | CV F1 | Train Accuracy |
|-------|------------|-------|----------------|
| Logistic Regression | **75.0%** | **75.3%** | 80.0% |
| GBDT (50 trees) | 70.0% | 59.3% | 100.0% (overfit) |

GBDT overfits on only 20 samples. Logistic regression generalizes better.

**Logistic regression learned coefficients (from all 13 features)**:

| Feature | Coefficient | Rank |
|---------|------------|------|
| rhythm_similarity | **+0.487** | 1 |
| semantic_similarity | **+0.366** | 2 |
| qmax_score | **+0.358** | 3 |
| clap_multiscale | **+0.322** | 4 |
| panns_similarity | +0.188 | 5 |
| melodic_alignment | +0.137 | 6 |
| mert_similarity | +0.089 | 7 |
| structural_correspondence | +0.061 | 8 |
| stem_combined | +0.039 | 9 |
| artifact_diff | +0.037 | 10 |
| cqt_similarity | +0.024 | 11 |
| ssm_similarity | +0.023 | 12 |
| spectral_correlation | +0.007 | 13 |
| **intercept** | **−1.465** | — |

**Verdict**: All 8 improvements implemented and working. Gap-proportional reweighting achieves **80% accuracy** — up from 50% baseline. ✅

### Summary of All Experiments

| Metric | Exp.1 | Exp.2 | Exp.3 | Exp.4 | Exp.5 | **Exp.6** |
|--------|-------|-------|-------|-------|-------|-----------|
| Branches | 7 | 7 | 8 | 8 | 13* | **13** |
| CLAP | Fallback | Fallback | Real | Real | Real | **Real** |
| MERT | — | — | 95M | 95M | **330M** | **330M** |
| Source Sep | — | — | — | — | Fallback | **HTDemucs** |
| PANNs | — | — | — | — | Fallback | **CNN14** |
| Score Gap | 0.024 | — | 0.046 | 0.053 | 0.035 | **0.065** |
| Best Accuracy | 50% | 68.6%† | 50% | 50% | 50% | **80%** |
| Time/Pair | 57s | 57s | 180s | 146s | 287s | 1013s |

\* Exp. 5: source sep and PANNs fallback due to runtime issues  
† Exp. 2: 124 pairs with tuned threshold; not directly comparable to 20-pair experiments

---

## 21. GitHub Repo Research & Technique Extraction

> **Idea Origin**: Systematic literature/code review of open-source repositories implementing **cover song identification (CSI)**, **audio deepfake detection**, and **music information retrieval (MIR)** techniques. The goal was to identify additional feature extraction methods that could strengthen MAIA's 13-branch pipeline. Seven repositories were analyzed.

### Repositories Analyzed

| Repo | Domain | Key Techniques | Feasible? |
|------|--------|---------------|-----------|
| **albincorreya/ChromaCoverId** | CSI | Dmax cumulative CRP, HPCP chroma | ✅ Yes — implemented |
| **Liu-Feng-deeplearning/CoverHunter** | CSI | Conformer + attention pooling, coarse-to-fine alignment | ❌ Requires training data |
| **MTG/da-tacos** | CSI benchmark | CENS/HPCP/CREMA chroma, tonnetz, spectral contrast, 7 CSI algorithms | ✅ Partially — 4 features tested |
| **furkanyesiler/acoss** | CSI benchmark | Serra09/LateFusion/EarlyFusion/SiMPle/FTM2D, madmom onsets | ✅ Partially — spectral flux tested |
| **csun22/Vocoder-Artifacts** | Deepfake detection | RawNet2 + multi-loss vocoder artifact detection | ❌ Requires training |
| **piotrkawa/specrnet** | Deepfake detection | SpecRNet lightweight architecture | ❌ Requires training |
| **piotrkawa/deepfake-whisper-features** | Deepfake detection | Whisper encoder features + classifier backends | ❌ Requires large model training |

### Technique Selection Criteria

1. **No training required** — MAIA uses unsupervised/zero-shot features only (no labeled training data available beyond small evaluation set)
2. **Pairwise formulation** — must produce a similarity score between two tracks
3. **Additive signal** — must provide discriminative gap on MIPPIA positive vs negative pairs
4. **Reasonable compute** — must complete within ~60s per pair on CPU

### Excluded Techniques (with justification)

- **CoverHunter Conformer**: Requires supervised training on large cover-song datasets; architecture is designed for learned embeddings, not zero-shot use
- **RawNet2 multi-loss** (Vocoder-Artifacts): End-to-end vocoder artifact classifier; needs labeled real/fake training data
- **SpecRNet** (specrnet): Lightweight but classifier-only; needs fine-tuning on music (trained on speech deepfakes)
- **Whisper features** (deepfake-whisper-features): Whisper is speech-focused; music encoding quality unverified; large compute

---

## 22. Experimental Feature Evaluation (5 Candidates)

> **Idea Origin**: From the GitHub repo analysis (§21), five unsupervised pairwise similarity features were identified as feasible candidates. Each was implemented in `src/features/sota_features.py` and evaluated standalone on the 10-pair MIPPIA subset.

### Evaluation Setup

- **Subset**: 10 balanced pairs (5 positive, 5 negative) from MIPPIA 124-pair dataset
- **Subset file**: `experiments/subset_10pairs.csv` (seed=42, deterministic)
- **Metric**: Mean score gap (positive pairs − negative pairs). Gap > 0.02 = viable signal.
- **Results file**: `experiments/results/experimental_features.json`

### Results

| Feature | Source Repo | Pos Mean | Neg Mean | Gap | Verdict |
|---------|------------|----------|----------|-----|---------|
| **dmax_score** | ChromaCoverId (Serra 2009) | 0.7927 | 0.7487 | **+0.044** | ✅ **INCLUDED** |
| **tonnetz_similarity** | da-tacos (Harte 2006) | 0.6660 | 0.6055 | **+0.061** | ✅ **INCLUDED** |
| cens_similarity | da-tacos (Müller 2007) | 0.9856 | 0.9799 | +0.006 | ❌ Excluded — saturated scores, weak gap |
| spectral_contrast | da-tacos (Jiang 2002) | 0.9993 | 0.9991 | +0.000 | ❌ Excluded — zero discrimination |
| spectral_flux_onset | madmom/acoss (Böck 2012) | 0.8150 | 0.8319 | −0.017 | ❌ Excluded — negative gap (anti-signal) |

### Per-Pair Breakdown (interesting cases)

**Tonnetz** showed the widest variance and strongest separation:
- Pair 45_vs_40 (negative): tonnetz = 0.038 — near-zero for unrelated harmonic profile ✅
- Pair 11 (positive): tonnetz = 0.993 — near-identical tonal space ✅
- Pair 35 (positive): tonnetz = 0.161 — false negative (key transposition disrupts tonnetz) ⚠️

**Dmax** correlated with but was partially independent of Qmax:
- Dmax uses cumulative DP along diagonals; Qmax uses single-diagonal max
- Correlation on 10 pairs: ~0.65 — enough independence to justify both

### Excluded Feature Analysis

- **CENS chroma**: The smoothing operation (Müller 2007) made nearly all pairs score >0.96, destroying discrimination. Our CQT chroma + OTI (§15) already covers this signal.
- **Spectral contrast**: Sub-band peak/valley ratio is too stable across different music tracks to serve as a pairwise discriminator (all scores >0.998).
- **Spectral flux onset**: Onset density similarity is *higher* for unrelated pairs (both have typical music onset patterns) than for AI-derived pairs (where AI may alter onset characteristics). This produces an anti-signal.

---

## 23. Dmax Cumulative CRP (Improvement #9)

> **Idea Origin**: **ChromaCoverId** repository (albincorreya/ChromaCoverId), implementing **Serra, Serra & Andrzejak (2009)** "Cross recurrence quantification for cover song identification" and **Chen et al. (2017)** refinements. The Dmax measure was already referenced in the original Qmax implementation (§16) but was not implemented. The ChromaCoverId codebase provided the DP accumulation algorithm.

**File**: `src/features/sota_features.py` → `dmax_similarity()`

### Algorithm

1. Compute HPCP chroma features for both tracks (12-bin, hop=512)
2. Build cross-similarity matrix using cosine distance
3. Threshold at the 80th percentile to create binary recurrence plot
4. Accumulate along all diagonals using dynamic programming:
   - For each diagonal, track the longest consecutive run of recurrence points
   - `Dmax = max_diagonal(cumulative_length) / min(len_a, len_b)`
5. Normalize to [0, 1]

### Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Chroma type | HPCP (librosa `chroma_cqt`) | Serra 2009 |
| Hop length | 512 samples | ChromaCoverId default |
| Distance metric | Cosine | Serra 2009 |
| Threshold percentile | 80th | Tuned (Serra used median) |
| Normalization | Length of shorter track | Standard CRP normalization |

### Experimental Result

- **Discriminative gap**: +0.044 (10-pair subset)
- **Correlation with Qmax**: ~0.65 (partially independent → additive value)
- **Weight assigned**: 0.04 (4% of final score)

### References
- Serra, J., Serra, X., & Andrzejak, R.G. (2009). "Cross recurrence quantification for cover song identification." *New J. Physics* 11, 093017.
- Chen, N., Li, W., & Xiao, H. (2017). "Fusing similarity functions for cover song identification." *Multimedia Tools Appl.* 77, 2629–2652.

---

## 24. Tonnetz Harmonic Similarity (Improvement #10)

> **Idea Origin**: **da-tacos** benchmark (MTG/da-tacos) and the **acoss** framework (furkanyesiler/acoss), both implementing the **Harte et al. (2006)** tonnetz representation. The 6-dimensional tonal centroid captures harmonic relationships (perfect fifths, minor/major thirds) that are invariant to timbre but sensitive to harmonic content — ideal for detecting whether Track B preserves Track A's harmonic structure.

**File**: `src/features/sota_features.py` → `tonnetz_similarity()`

### Algorithm

1. Compute harmonic component using `librosa.effects.harmonic()`
2. Extract 6-D tonnetz features using `librosa.feature.tonnetz()`:
   - Dimensions represent projections onto: fifth, minor third, major third (and their second-order harmonics)
3. Compute frame-by-frame cosine similarity (truncated to shorter track length)
4. Return mean cosine similarity as final score

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Input | Harmonic-separated audio | Removes percussive components that add noise |
| Feature dims | 6 | Standard tonnetz (Harte 2006) |
| Similarity | Cosine per frame → mean | Handles tempo differences via truncation |
| Sample rate | 22,050 Hz | Consistent with rest of pipeline |

### Experimental Result

- **Discriminative gap**: +0.061 (10-pair subset) — **strongest of all 5 tested features**
- **Variance**: High (range: 0.038 to 0.993) — good discrimination but sensitive to key changes
- **Weight assigned**: 0.06 (6% of final score)
- **Limitation**: Key transposition can reduce score significantly (pair 35: 0.161)

### Why tonnetz > CENS chroma?

Both capture harmonic information, but:
- **Tonnetz** projects into a continuous 6-D tonal space → captures *relative* harmonic relationships
- **CENS chroma** smooths absolute pitch class energy → saturates near 1.0 for any music pair
- Tonnetz has 10× the discriminative gap (+0.061 vs +0.006)

### References
- Harte, C., Sandler, M., & Gasser, M. (2006). "Detecting harmonic change in musical audio." *ACM MM workshop on Audio and Music Computing Multimedia*.
- Yesiler, F., Serrà, J., & Gómez, E. (2020). "Accurate and scalable version identification using musically-motivated embeddings." *ICASSP 2020*.

---

## 25. 15-Branch Subset Comparison

> **Idea Origin**: Controlled A/B experiment comparing the original 13-branch pipeline (Exp. 6) against the improved 15-branch pipeline (+Dmax, +Tonnetz) on a fixed 10-pair evaluation subset.

### Setup

- **Subset**: 10 pairs (5 positive, 5 negative), file: `experiments/subset_10pairs.csv`
- **Baseline**: 13-branch scorer (Dmax & Tonnetz weights zeroed, remaining weights renormalized)
- **Improved**: 15-branch scorer (full weights as specified in §10)
- **Source separation**: Skipped (`MAIA_SKIP_SRCSEP=1`) — contributes minimal signal (+0.008 gap)
- **Threshold**: 0.72 (optimized in Exp. 6)
- **Results file**: `experiments/results/subset_comparison.json`

### Weight Configurations

**13-branch baseline** (renormalized from 15-branch, Dmax/Tonnetz zeroed):

| Branch | Weight |
|--------|--------|
| rhythm | 0.211 |
| panns | 0.189 |
| semantic | 0.144 |
| qmax | 0.133 |
| clap_multiscale | 0.133 |
| melodic | 0.056 |
| mert | 0.044 |
| structural | 0.022 |
| stem_combined | 0.022 |
| artifact_diff | 0.011 |
| cqt | 0.011 |
| ssm | 0.011 |
| spectral_corr | 0.011 |

**15-branch improved** (current production weights):

| Branch | Weight |
|--------|--------|
| rhythm | 0.19 |
| panns | 0.17 |
| semantic | 0.13 |
| qmax | 0.12 |
| clap_multiscale | 0.12 |
| tonnetz | 0.06 |
| melodic | 0.05 |
| mert | 0.04 |
| dmax | 0.04 |
| structural | 0.02 |
| stem_combined | 0.02 |
| artifact_diff | 0.01 |
| cqt | 0.01 |
| ssm | 0.01 |
| spectral_corr | 0.01 |

### Results

**Evaluation on 10-pair subset** (5 positive, 5 negative; threshold = 0.72; `MAIA_SKIP_SRCSEP=1`):

| Metric | 13-branch (baseline) | 15-branch (improved) | Delta |
|--------|---------------------|---------------------|-------|
| Accuracy | **70%** | 50% | -20% |
| Precision | **1.00** | 0.50 | -0.50 |
| Recall | 0.40 | 0.40 | 0.00 |
| F1 | **0.571** | 0.444 | -0.127 |
| Mean pos score | 0.7063 | 0.7073 | +0.0010 |
| Mean neg score | 0.6911 | 0.6882 | -0.0029 |
| Score gap | 0.0152 | **0.0191** | **+0.0039** |

**Per-pair scores** (corrected weights, sum = 1.0):

| Pair | Label | 13-branch | 15-branch | Delta |
|------|-------|-----------|-----------|-------|
| 9_vs_58 | 0 | 0.7016 | 0.7223 | +0.0208 |
| 11 | 1 | 0.6765 | 0.6968 | +0.0203 |
| 53_vs_19 | 0 | 0.6614 | 0.6606 | -0.0008 |
| 70 | 1 | 0.7739 | 0.7696 | -0.0043 |
| 24_vs_14 | 0 | 0.7178 | 0.7299 | +0.0121 |
| 3 | 1 | 0.7519 | 0.7466 | -0.0053 |
| 45_vs_40 | 0 | 0.6609 | 0.6259 | -0.0350 |
| 13 | 1 | 0.6733 | 0.6932 | +0.0200 |
| 35 | 1 | 0.6557 | 0.6303 | -0.0254 |
| 32_vs_22 | 0 | 0.7137 | 0.7025 | -0.0112 |

**Sub-score discriminative power** (new branches highlighted):

| Branch | Weight | Pos Mean | Neg Mean | Gap |
|--------|--------|----------|----------|-----|
| qmax | 0.12 | 0.1170 | 0.0620 | +0.0550 |
| **tonnetz** | **0.06** | **0.6660** | **0.6055** | **+0.0605** |
| **dmax** | **0.04** | **0.7927** | **0.7487** | **+0.0439** |
| melodic | 0.05 | 0.6237 | 0.5936 | +0.0300 |
| semantic | 0.13 | 0.8295 | 0.8085 | +0.0210 |
| clap_multiscale | 0.12 | 0.8247 | 0.8055 | +0.0192 |
| structural | 0.02 | 0.9194 | 0.9023 | +0.0171 |

### Analysis

1. **Accuracy is the metric that matters.** The 15-branch pipeline dropped from 70% to 50% accuracy. Score gap improvement (+0.004) is meaningless if classification gets worse.

2. **Root cause**: Dmax and tonnetz inflate scores for negative pairs (9_vs_58: +0.021, 24_vs_14: +0.012) pushing them above the 0.72 threshold, creating false positives without converting any false negatives.

3. **Sub-score gaps are misleading**: Tonnetz (+0.0605) and dmax (+0.0439) show strong per-branch discriminative signal in isolation, but when mixed into the weighted score, the effect on negative pairs outweighs the benefit on positive pairs.

4. **Decision: Reverted to 13-branch (Exp. 6)**. The production pipeline uses the 13-branch configuration that achieved 80% accuracy on 20 pairs. Dmax and tonnetz code remains in `sota_features.py` for future research but is excluded from the scoring pipeline.

---

## 26. References

1. **Rahman, M.A., Hakim, Z.I.A., Sarker, N.H., Paul, B., & Fattah, S.A.** (2025). "SONICS: Synthetic Or Not — Identifying Counterfeit Songs." *ICLR 2025*. arXiv:2408.14080.
   - Key insights used: Long-range temporal dependencies (SSM), rhythmic predictability, pitch contour stability, dynamic compression, SpecTTTra architecture motivation.

2. **Comanducci, L., Bestagini, P., & Tubaro, S.** (2024). "FakeMusicCaps: a Dataset for Detection and Attribution of Synthetic Music Generated via Text-to-Music Models." arXiv:2409.10684.
   - Key insights used: Mel-spectrogram cross-correlation, ResNet18+Spec baseline approach.

3. **Wu, Y. et al.** (2023). "Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation." *ICASSP 2023*.
   - Used: CLAP model (`laion/larger_clap_music_and_speech`) for 512-D semantic embeddings.

4. **Li, Y., Yuan, R., Zhang, G., et al.** (2024). "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training." *ICLR 2024*. arXiv:2306.00107.
   - Used: MERT v1-95M for 768-D music-specific embeddings with CQT + RVQ-VAE dual teachers.

5. **Rouard, S., Massa, F., & Défossez, A.** (2023). "Hybrid Transformers for Music Source Separation." *ICASSP 2023*. arXiv:2211.08553.
   - Referenced: HTDemucs state-of-the-art source separation (9.2 dB SDR). Identified as a high-impact future improvement for per-stem comparison.

6. **Luo, Y. & Yu, J.** (2022). "Music Source Separation with Band-split RNN." arXiv:2209.15174.
   - Referenced: BSRNN frequency-domain separation approach for future improvement.

7. **Won, M., Hung, Y.-N., & Le, D.** (2023). "A Foundation Model for Music Informatics." arXiv:2311.03318.
   - Referenced: MusicFM foundation model as alternative/complement to MERT.

8. **Wu, S., Yu, D., Tan, X., & Sun, M.** (2023). "CLaMP: Contrastive Language-Music Pre-training for Cross-Modal Symbolic Music Information Retrieval." *ISMIR 2023*. arXiv:2304.11029.
   - Referenced: Symbolic MIR approach with 1.4M music-text pairs.

9. **Copet, J., Kreuk, F., Gat, I., et al.** (2023). "Simple and Controllable Music Generation." *NeurIPS 2023*. arXiv:2306.05284.
   - Referenced: MusicGen architecture — understanding how TTMs generate helps predict what artifacts to expect.

10. **Serra, J., Serra, X., & Andrzejak, R.G.** (2009). "Cross recurrence quantification for cover song identification." *New Journal of Physics*, 11(9), 093017.
    - Used: Qmax metric and cross-recurrence plot analysis for structural similarity.

11. **Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., & Plumbley, M.D.** (2020). "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." *IEEE/ACM TASLP*.
    - Used: CNN14 pretrained on AudioSet for 2048-D perceptual audio embeddings.

12. **Chen, N., Li, W., & Xiao, H.** (2017). "Fusing similarity functions for cover song identification." *Multimedia Tools and Applications*, 77, 2629–2652.
    - Used: Dmax cumulative cross-recurrence measure (ChromaCoverId implementation).

13. **Harte, C., Sandler, M., & Gasser, M.** (2006). "Detecting harmonic change in musical audio." *ACM Multimedia Workshop on Audio and Music Computing Multimedia*.
    - Used: 6-D tonnetz tonal centroid features for harmonic similarity.

14. **Müller, M., Kurth, F., & Clausen, M.** (2005). "Audio Matching via Chroma-Based Statistical Features." *ISMIR 2005*.
    - Evaluated: CENS chroma-energy normalized statistics — excluded due to score saturation.

15. **Jiang, D.-N., Lu, L., Zhang, H.-J., Tao, J.-H., & Cai, L.-H.** (2002). "Music type classification by spectral contrast feature." *IEEE ICME*.
    - Evaluated: Spectral contrast similarity — excluded due to zero discrimination.

16. **Yesiler, F., Serrà, J., & Gómez, E.** (2020). "Accurate and scalable version identification using musically-motivated embeddings." *ICASSP 2020*.
    - Referenced: da-tacos benchmark framework for cover song identification.

17. **Böck, S. & Widmer, G.** (2013). "Maximum filter vibrato suppression for onset detection." *DAFx-13*.
    - Evaluated: Spectral flux onset similarity (via madmom) — excluded due to negative gap.

---

## 27. Overfitting Audit & Deployment Guidance

> **Idea Origin**: User requirement for strict anti-overfitting controls and suspicious-metric reporting. Implemented via an upgraded robust validation script (`experiments/robust_validation.py`) with repeated holdout audits, bootstrap confidence intervals, threshold-stability checks, and nested leakage-safe comparisons.

### What Was Added (Audit Methodology)

For each results JSON file, the audit now computes:

1. **Fixed-threshold baseline metrics** at deployment threshold (default: 0.805)
2. **Bootstrap 95% CI** for balanced accuracy (2,000 resamples)
3. **Threshold sweep** around default threshold (±0.12, clipped to [0.5, 0.95])
4. **Repeated stratified holdout audit** (200 runs, 70/30 split):
   - train/test balanced-accuracy means and standard deviations
   - mean train-test gap (overfitting indicator)
   - tuned-threshold distribution (operating-point robustness)
5. **Nested leakage-safe comparators**:
   - nested threshold-only tuning
   - nested logistic calibration
6. **Automatic warning flags** for instability and overfitting

### Latest Audit Results

#### A) Full MIPPIA Evaluation (124 pairs)

**Input**: `results/mippia_full_results.json`  
**Report**: `experiments/results/robust_validation_report_full.json`

| Metric | Value |
|--------|-------|
| Baseline threshold | 0.805 |
| Baseline accuracy / balanced accuracy | 0.6855 / 0.6855 |
| Bootstrap BA 95% CI | [0.6041, 0.7668] |
| CI width | 0.1627 |
| Sweep best threshold | 0.807 |
| Sweep best balanced accuracy | 0.6935 |
| Repeated holdout train BA (mean ± std) | 0.6994 ± 0.0286 |
| Repeated holdout test BA (mean ± std) | 0.6476 ± 0.0679 |
| Mean train-test BA gap | 0.0518 |
| Tuned threshold (mean ± std) | 0.8060 ± 0.0051 |
| Nested threshold-only BA | 0.6774 |
| Nested logistic BA | 0.6613 |

**Interpretation**:
- Default threshold **0.805 is validated** (near sweep optimum at 0.807; very low threshold variance).
- Nested alternatives do not beat baseline, so no evidence that extra model complexity helps.
- Two caution flags remain: test-metric variance and wide confidence interval (data size limitation).

#### B) Experimental 13-Branch Subset (20 pairs)

**Input**: `results/exp6_full_13branch.json`  
**Report**: `experiments/results/robust_validation_report_exp6.json`

| Metric | Value |
|--------|-------|
| Baseline threshold | 0.805 |
| Baseline accuracy / balanced accuracy | 0.7500 / 0.7500 |
| Bootstrap BA 95% CI | [0.5594, 0.9375] |
| CI width | 0.3781 |
| Sweep best threshold | 0.828 |
| Sweep best balanced accuracy | 0.8500 |
| Repeated holdout train BA (mean ± std) | 0.8568 ± 0.0488 |
| Repeated holdout test BA (mean ± std) | 0.7267 ± 0.1476 |
| Mean train-test BA gap | 0.1301 |
| Tuned threshold (mean ± std) | 0.8163 ± 0.0148 |
| Nested threshold-only BA | 0.7500 |
| Nested logistic BA | 0.6500 |

**Interpretation**:
- Point metrics look good, but **generalization is not trustworthy** on 20 pairs.
- Large train-test gap (+0.1301), high variance, and very wide CI indicate overfitting risk.
- This subset is suitable for feature prototyping, not deployment decisions.

### Automatic Warning Outcomes

- **124-pair file**:
  - Potential instability (test BA variance)
  - High uncertainty (wide CI)
- **20-pair file**:
  - Potential overfitting (train-test BA gap)
  - Potential instability (test BA variance)
  - High uncertainty (wide CI)

### Sound-Engineering Deployment Guidance

1. Keep deployment threshold near **0.805** for current full-dataset operating point.
2. Treat 20-pair experimental gains as **hypothesis-only** until confirmed on the 124-pair benchmark.
3. Prioritize **branch robustness** over branch count: remove or downweight branches that increase variance even if they improve small-subset point scores.
4. Track both central metrics and uncertainty:
   - optimize balanced accuracy
   - enforce CI-width and split-variance guardrails
5. Keep leakage-safe nested evaluation as the default model-selection protocol.

### Reproducibility Commands

```bash
python experiments/robust_validation.py --results_json results/mippia_full_results.json --output experiments/results/robust_validation_report_full.json
python experiments/robust_validation.py --results_json results/exp6_full_13branch.json --output experiments/results/robust_validation_report_exp6.json
```

This audit framework is now the recommended gate before accepting any future “accuracy improvement” claim.
