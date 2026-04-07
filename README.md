# MAIA — Multi-scale Attribution Intelligence Architecture

## Objective

Given two full-length audio tracks, determine whether Track B was generated using Track A as a reference and output a **Similarity (Attribution) Score** in `[0, 1]`.

The assessment brief is in [guidelines/guidelines.md](guidelines/guidelines.md).
The PDF report is [report.pdf](report.pdf).

---

## How It Works

MAIA is not a binary "is this AI?" classifier. It performs **pairwise attribution**: comparing two tracks and returning a score that reflects how likely Track B is a synthetic derivative of Track A.

The approach is a **13-branch weighted ensemble** — each branch extracts a different audio similarity signal and a weighted sum produces the final score. No branch uses supervised training on evaluation data; all signals are unsupervised or zero-shot.

**Why an ensemble?** A single embedding model can't handle all the ways AI generators transform source material (key transpositions, tempo changes, arrangement shifts). MAIA decomposes the problem into 13 independent signals — each targeting a different aspect — and combines them with empirically validated weights from 6 iterative experiments on the MIPPIA dataset.

---

## The 13 Branches

Weights are from the final Experiment 6 configuration, proportional to each branch's measured discriminative gap (mean attributed score − mean unrelated score) on 20 MIPPIA pairs.

| Branch | Weight | What it measures |
|---|---:|---|
| `rhythm` | 0.20 | Onset envelope + tempo ratio — strongest signal; AI generators preserve rhythmic structure |
| `panns` | 0.20 | PANNs CNN14 (AudioSet-pretrained) — general perceptual audio texture |
| `semantic` | 0.14 | CLAP global cosine similarity — musical meaning via language-audio embeddings |
| `qmax` | 0.15 | Qmax cross-recurrence plot — detects shared melodic sequences under tempo shifts |
| `clap_multiscale` | 0.14 | Per-window CLAP set-matching — captures section-level temporal correspondence |
| `melodic` | 0.05 | Key-shift-invariant chroma DTW — tests all 12 chromatic transpositions |
| `mert` | 0.04 | MERT v1-330M music embeddings — CQT + RVQ-VAE dual-teacher self-supervised model |
| `structural` | 0.02 | Section boundary alignment — attributed pairs share song structure proportions |
| `stem_combined` | 0.02 | HTDemucs per-stem comparison (vocal/drum/bass/other) |
| `artifact_diff` | 0.01 | `1 − \|artifact_A − artifact_B\|` — attributed pairs share similar AI processing profiles |
| `cqt` | 0.01 | CQT chroma + optimal transposition index |
| `ssm` | 0.01 | Self-similarity matrix Frobenius inner product |
| `spectral_corr` | 0.01 | Mel-spectrogram cross-correlation |

---

## Results

| Evaluation | File | Accuracy | Threshold | Notes |
|---|---|---:|---:|---|
| **124-pair MIPPIA (Exp 2)** | `results/mippia_full_results.json` | **68.5%** | 0.805 | Main result — fixed threshold, full dataset |
| **20-pair full pipeline (Exp 6)** | `results/exp6_full_13branch.json` | **80%** | 0.72 | Gap-proportional reweighting applied offline |
| Logistic calibrator CV | — | **75%** | — | 5-fold CV on 20-pair sub-scores |
| Overfitting audit (124-pair) | `experiments/results/robust_validation_report_full.json` | Train–test gap: 5.2% | — | Threshold stable (std=0.005) |
| Overfitting audit (20-pair) | `experiments/results/robust_validation_report_exp6.json` | Train–test gap: 13% | — | High variance — prototyping only |

Verdict thresholds:

| Score | Verdict |
|---|---|
| ≥ 0.75 | Strong AI Attribution |
| ≥ 0.55 | Probable AI Attribution |
| ≥ 0.35 | Possible Relationship |
| < 0.35 | Unlikely Attribution |

---

## Experiment Progression

| Experiment | Config | Pairs | Best Accuracy | Score Gap | Key Change |
|---|---|---:|---:|---:|---|
| Exp 1 | 7-branch, fallback CLAP | 20 | 50%* | 0.024 | Baseline — threshold miscalibrated |
| Exp 2 | 7-branch, fallback CLAP | 124 | 68.5% | — | Full dataset, tuned threshold |
| Exp 3 | 8-branch, real CLAP + MERT-95M | 20 | 50%* | 0.046 | Real CLAP: semantic gap +0.001 → +0.080 |
| Exp 4 | 8-branch, + artifact_diff | 20 | 50%* | 0.053 | Replaced artifact_boost (negative gap −0.090) |
| Exp 5 | 13-branch, MERT-330M (srcsep fallback) | 20 | 50%* | 0.035 | Qmax, multi-scale CLAP, CQT added; srcsep/PANNs fell back |
| **Exp 6** | **13-branch, full** | **20** | **80%** | **0.065** | All branches working; gap-proportional reweighting |

\* Threshold of 0.55 was too low for the score distribution — all pairs predicted positive. Accuracy improves significantly with a tuned threshold.

A **15-branch variant** (adding Dmax + Tonnetz) was tested and **reverted** — it improved the score gap marginally (+0.004) but dropped accuracy from 70% to 50% on 10 pairs by inflating negative pair scores above the decision boundary.

---

## Dataset

**MIPPIA (SMP)** was chosen as the primary evaluation dataset because it provides original/AI-generated track **pairs** — exactly what a pairwise attribution system needs. SONICS and FakeMusicCaps are useful for binary AI detection but lack the paired structure required.

- 62 original + 62 AI-generated tracks = **124 balanced evaluation pairs**
- Downloaded via `yt-dlp` + `ffmpeg` using the MIPPIA `download.py` function
- Download scripts: `data/download_mippia.py`, `data/download_all_mippia.py`

---

## Installation

**Requirements:** Python 3.10+, `ffmpeg` on PATH

```bash
git clone https://github.com/apkexe/sound-engineering.git
cd sound-engineering
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Models (CLAP, MERT, PANNs, HTDemucs) download automatically on first run.

---

## Usage

### Compare two tracks

```bash
python pipeline.py --track_a data/a.wav --track_b data/b.wav --verbose
```

Python API:

```python
from pipeline import compare_tracks

result = compare_tracks("data/a.wav", "data/b.wav", verbose=True)
print(result["attribution_score"])  # float in [0, 1]
print(result["verdict"])            # human-readable string
```

### Skip source separation (faster iteration)

HTDemucs source separation is the slowest branch (~15 min/pair on CPU). Skip it for faster runs:

```bash
# Linux/macOS
MAIA_SKIP_SRCSEP=1 python pipeline.py --track_a data/a.wav --track_b data/b.wav

# Windows PowerShell
$env:MAIA_SKIP_SRCSEP = "1"
python pipeline.py --track_a data/a.wav --track_b data/b.wav
```

### Batch evaluation

```bash
python evaluate.py --pairs_csv data/eval_pairs_mippia.csv --output results/eval.json
```

### Full end-to-end benchmark

```bash
# 1. Download MIPPIA data
python data/download_all_mippia.py --format wav

# 2. Run full evaluation
python run_e2e_eval.py --output results/mippia_e2e_results.json

# 3. Quick 20-pair subset
python run_e2e_eval.py --max_pairs 20 --output results/quick_eval.json

# 4. Overfitting audit
python experiments/robust_validation.py --results_json results/mippia_e2e_results.json \
  --output experiments/results/robust_validation_report.json

# 5. Sanity check
python tests/test_sanity.py
```

### Train a calibrator

```bash
python train_calibrator.py --input results/mippia_full_results.json --method logistic
python train_calibrator.py --input results/mippia_full_results.json --method gbdt
```

---

## Repo Structure

```
sound-engineering/
├── pipeline.py                  # Main entrypoint: compare_tracks()
├── evaluate.py                  # Batch evaluation on labeled pairs
├── run_e2e_eval.py              # End-to-end benchmark runner
├── train_calibration.py         # Logistic calibration trainer
├── train_calibrator.py          # GBDT/logistic calibrator trainer
├── test_one_pair.py             # Quick single-pair test
├── requirements.txt
├── report.pdf                   # PDF report (architecture, results, design decisions)
├── SCIENTIFIC_IMPLEMENTATIONS.md # Full 27-section technical reference
│
├── guidelines/
│   └── guidelines.md            # Assessment brief
│
├── src/
│   ├── features/
│   │   ├── temporal_sampler.py  # Multi-scale temporal sampling
│   │   ├── spectral.py          # Spectral feature extraction (395-D)
│   │   ├── embeddings.py        # CLAP embeddings + multi-scale CLAP
│   │   ├── mert_embeddings.py   # MERT v1-330M
│   │   ├── panns_embeddings.py  # PANNs CNN14
│   │   ├── sota_features.py     # SSM, rhythm, Qmax, Dmax, Tonnetz, CQT+OTI
│   │   ├── source_separation.py # HTDemucs stem separation
│   │   └── artifacts.py         # AI artifact detection (7 sub-detectors)
│   └── model/
│       ├── attribution.py       # 13-branch weighted scorer
│       └── calibration.py       # Score calibration
│
├── data/
│   ├── a.wav, b.wav             # Sample files for smoke testing
│   ├── eval_pairs_mippia.csv    # Full evaluation pairs
│   ├── build_mippia_eval.py     # Builds eval CSV from audio dir
│   ├── download_all_mippia.py   # MIPPIA dataset downloader
│   └── reports/                 # Sample outputs: eval JSONs and CSVs
│
├── experiments/
│   ├── robust_validation.py     # Overfitting audit & stability diagnostics
│   ├── eval_experimental.py     # Experimental feature evaluation
│   ├── run_experiment.py
│   └── results/                 # Audit reports and experiment outputs
│
├── models/                      # Trained calibrators (JSON)
├── results/                     # Evaluation result JSONs (all 6 experiments)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_evaluation.ipynb
│
└── tests/
    └── test_sanity.py           # Import, weight, branch count checks
```

---

## Design Decisions & Trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| Weighted ensemble over end-to-end DL | No large labeled training set for pairwise attribution; each branch is interpretable and independently testable | Lower ceiling than a trained model |
| Gap-proportional weights | Weights derived from measured discriminative gaps on MIPPIA — data-driven, not hand-tuned | Optimised on 20 pairs; may shift on larger data |
| MIPPIA over SONICS/FakeMusicCaps | Only MIPPIA provides paired original/AI tracks for attribution | Smaller dataset (62 pairs) limits statistical power |
| 13 branches, not 15 | Dmax and Tonnetz had strong per-branch gaps but degraded accuracy when combined | Loses potentially useful signals; revisit with more data |
| Zero-shot / unsupervised only | Avoids overfitting on tiny eval set | Can't learn task-specific patterns a trained model could |
| Source separation (HTDemucs) | Per-stem comparison reduces inter-instrument interference | ~15 min/pair on CPU; skippable via env var |

---

## Limitations & Future Work

**Current limitations:**
- Small evaluation set (124 pairs) — bootstrap 95% CI spans ±8 percentage points
- Source separation is expensive on CPU (~15 min/pair)
- Only evaluated on MIPPIA-style pairs (Suno/Udio); cross-generator generalisation untested
- Dmax and Tonnetz code exists in `sota_features.py` but is disabled in the scorer

**Future improvements:**
- Evaluate on SONICS pairs for cross-generator generalisation
- Train a lightweight MLP calibrator on a larger labelled set
- Key-invariant Tonnetz to fix transposition sensitivity
- GPU acceleration for HTDemucs and MERT inference
- Separate research and production code into distinct modules

---

## Libraries & Tools Used

| Category | Tools |
|---|---|
| **Audio** | Librosa, SoundFile |
| **Embeddings** | CLAP (`laion/larger_clap_music_and_speech`), MERT v1-330M, PANNs CNN14 (Hugging Face) |
| **Similarity** | Scikit-learn (cosine similarity, logistic regression), `dtaidistance` (DTW), FAISS |
| **Source Separation** | HTDemucs (via `demucs`) |
| **Models** | PyTorch, Hugging Face Transformers |
| **Evaluation** | NumPy, Pandas, Scikit-learn metrics |

---

## References

1. Rahman et al. "SONICS" (ICLR 2025) — SSM, rhythmic/dynamic artifact detection
2. Comanducci et al. "FakeMusicCaps" (2024) — mel-spectrogram cross-correlation
3. Wu et al. CLAP (ICASSP 2023) — contrastive language-audio embeddings
4. Li et al. MERT (ICLR 2024) — music understanding via masked acoustic modelling
5. Kong et al. PANNs (2020) — large-scale pretrained audio neural networks
6. Rouard et al. HTDemucs (ICASSP 2023) — hybrid transformer source separation
7. Serra et al. (2009) — cross-recurrence Qmax/Dmax for cover song identification
8. Harte et al. (2006) — tonnetz tonal centroid features

Full reference list (17 entries) in [SCIENTIFIC_IMPLEMENTATIONS.md](SCIENTIFIC_IMPLEMENTATIONS.md#26-references).
