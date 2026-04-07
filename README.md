# MAIA

> Multi-scale Attribution Intelligence Architecture  
> Orfium hiring assessment repository for pairwise AI-generated song attribution

---

## What This Repo Does

MAIA answers a pairwise attribution question:

**Was Track B generated using Track A as a reference?**

This repository is not a generic "is this AI music?" classifier. It compares two tracks and returns an attribution score in `[0, 1]` together with interpretable sub-scores.

The assessment document used to guide the implementation is included here: [IN-Hiring Assesment - AI Generated song detection-050326-091206.pdf](IN-Hiring%20Assesment%20-%20AI%20Generated%20song%20detection-050326-091206.pdf)

The current supported production path in this repo is a **13-branch weighted scorer** built on:

- structure-aware temporal sampling
- CLAP semantic embeddings
- MERT music embeddings
- PANNs perceptual embeddings
- rhythmic, structural, spectral, and artifact-based similarity signals
- an explicit validation workflow with repeated holdout, nested CV, and suspicious-metric warnings

---

## Current Repo State

This README reflects the repository as it exists now.

- Main inference entrypoint: [pipeline.py](pipeline.py)
- Batch evaluation entrypoint: [evaluate.py](evaluate.py)
- End-to-end benchmark runner: [run_e2e_eval.py](run_e2e_eval.py)
- Validation and overfitting audit: [experiments/robust_validation.py](experiments/robust_validation.py)
- Detailed technical write-up: [SCIENTIFIC_IMPLEMENTATIONS.md](SCIENTIFIC_IMPLEMENTATIONS.md)
- Packaging notes: [GITHUB_DEPLOYMENT.md](GITHUB_DEPLOYMENT.md)
- Sanity checks: [tests/test_sanity.py](tests/test_sanity.py)

The repository also contains research artifacts and exploratory scripts under [experiments](experiments), but the primary supported path is the 13-branch MAIA pipeline.

---

## Main Results In Repo

These numbers come from result artifacts currently present in the repository.

| Evaluation | Artifact | Result |
|---|---|---|
| Full MIPPIA evaluation | [results/mippia_full_results.json](results/mippia_full_results.json) | Accuracy / balanced accuracy: `0.6855` at threshold `0.805` |
| Robust audit on full eval | [experiments/results/robust_validation_report_full.json](experiments/results/robust_validation_report_full.json) | Stable threshold near `0.805`, moderate uncertainty, no obvious leakage signal |
| 20-pair experimental run | [results/exp6_full_13branch.json](results/exp6_full_13branch.json) | Accuracy: `0.75` at threshold `0.805` |
| Robust audit on 20-pair run | [experiments/results/robust_validation_report_exp6.json](experiments/results/robust_validation_report_exp6.json) | High variance; useful for experimentation, not strong deployment evidence |

Important distinction:

- The **124-pair evaluation** is the more credible benchmark in this repo.
- The **20-pair experiments** are useful for iteration, but they are much more variance-sensitive.

---

## Production Configuration

The scorer implementation in [src/model/attribution.py](src/model/attribution.py) currently uses **13 active branches**.

Active production weights:

| Branch | Weight |
|---|---:|
| `semantic` | 0.15 |
| `melodic` | 0.06 |
| `structural` | 0.02 |
| `artifact_diff` | 0.01 |
| `ssm` | 0.01 |
| `spectral_corr` | 0.01 |
| `rhythm` | 0.22 |
| `mert` | 0.04 |
| `stem_combined` | 0.02 |
| `cqt` | 0.01 |
| `qmax` | 0.13 |
| `clap_multiscale` | 0.13 |
| `panns` | 0.19 |

Research-only fields like `dmax_score` and `tonnetz_similarity` may appear in exploratory scripts or output shape discussions, but they are **not active production weights** in the default configuration.

Validated evaluation threshold:

- `0.805` in [evaluate.py](evaluate.py) and [run_e2e_eval.py](run_e2e_eval.py)

Human-readable verdict thresholds from the scorer:

- `>= 0.75`: Strong AI Attribution
- `>= 0.55`: Probable AI Attribution
- `>= 0.35`: Possible Relationship
- `< 0.35`: Unlikely Attribution

---

## Installation

### Requirements

- Python 3.10+
- `ffmpeg` available on `PATH`
- enough disk for downloaded models and local audio data

### Setup

```bash
git clone <repo-url>
cd orfium

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Notes:

- model downloads happen lazily at runtime
- PANNs weights may need to exist in the standard user cache path if not auto-downloaded
- large datasets, reports, caches, and local artifacts are intentionally ignored by [.gitignore](.gitignore)

---

## Quick Start

### Single Pair Inference

The repository includes small local sample files [data/a.wav](data/a.wav) and [data/b.wav](data/b.wav) for a smoke test.

```bash
python pipeline.py --track_a data/a.wav --track_b data/b.wav --verbose
```

You can also call the API directly:

```python
from pipeline import compare_tracks

result = compare_tracks("data/a.wav", "data/b.wav", verbose=True)
print(result["attribution_score"])
print(result["verdict"])
```

Representative output keys:

```json
{
  "attribution_score": 0.6038,
  "semantic_similarity": 0.7421,
  "melodic_alignment": 0.5117,
  "best_chroma_key_shift": 2,
  "structural_correspondence": 0.8423,
  "ai_artifact_score": 0.3172,
  "artifact_diff": 0.95,
  "ssm_similarity": 0.8,
  "spectral_correlation": 0.9,
  "rhythm_similarity": 0.7,
  "mert_similarity": 0.85,
  "stem_combined": 0.78,
  "cqt_similarity": 0.95,
  "qmax_score": 0.12,
  "clap_multiscale": 0.88,
  "panns_similarity": 0.82,
  "verdict": "Probable AI Attribution — Track B shows significant similarities to Track A"
}
```

### Faster Iteration Without Source Separation

Source separation is the most expensive part of the pipeline and can be skipped for faster experiments.

```bash
# Windows PowerShell
$env:MAIA_SKIP_SRCSEP = "1"
python pipeline.py --track_a data/a.wav --track_b data/b.wav

# Linux/macOS
MAIA_SKIP_SRCSEP=1 python pipeline.py --track_a data/a.wav --track_b data/b.wav
```

---

## Batch Evaluation

Evaluate a CSV with columns `track_a`, `track_b`, `label`:

```bash
python evaluate.py --pairs_csv data/eval_pairs_mippia.csv --output results/eval.json --threshold 0.805
```

Optional calibrated evaluation:

```bash
python evaluate.py --pairs_csv data/eval_pairs_mippia.csv --output results/eval_calibrated.json --threshold 0.805 --calibration_model models/calibrator_logistic.json
```

The default threshold in the current repo is `0.805`.

---

## End-to-End Benchmark Flow

The main reproducible benchmark path is:

### 1. Download data

```bash
python data/download_all_mippia.py --format wav
```

### 2. Run the end-to-end evaluation

```bash
python run_e2e_eval.py --output results/mippia_e2e_results.json --threshold 0.805
```

This script internally rebuilds the evaluation CSV and then runs batch evaluation.

### 3. Quick benchmark on a smaller subset

```bash
python run_e2e_eval.py --max_pairs 20 --output results/quick_eval.json --threshold 0.805
```

### 4. Run the overfitting audit

```bash
python experiments/robust_validation.py --results_json results/mippia_e2e_results.json --output experiments/results/robust_validation_report.json
```

### 5. Run the sanity check

```bash
python tests/test_sanity.py
```

---

## Validation And Overfitting Checks

The repo includes an explicit audit path in [experiments/robust_validation.py](experiments/robust_validation.py).

What it currently does:

- fixed-threshold baseline evaluation
- bootstrap confidence interval for balanced accuracy
- threshold sweep around the deployment threshold
- repeated stratified holdout audit
- nested threshold-only comparison
- nested logistic calibration comparison
- warning generation for suspicious instability and overfitting patterns

Current practical interpretation of the repo’s results:

- [results/mippia_full_results.json](results/mippia_full_results.json) is the best available evidence in the repo
- threshold `0.805` is close to the best threshold found by sweep on full data
- the 20-pair experiments are useful for exploration, but they are not strong enough to justify stronger deployment claims on their own

Suspicious-metric examples this repo is designed to catch:

- very large train/test gaps
- very high train performance with weak test performance
- unstable threshold selection across repeated splits
- wide confidence intervals from too little evidence

---

## Repo Structure

```text
orfium/
├── pipeline.py
├── evaluate.py
├── run_e2e_eval.py
├── README.md
├── SCIENTIFIC_IMPLEMENTATIONS.md
├── GITHUB_DEPLOYMENT.md
├── requirements.txt
├── .gitignore
├── IN-Hiring Assesment - AI Generated song detection-050326-091206.pdf
│
├── src/
│   ├── features/
│   │   ├── temporal_sampler.py
│   │   ├── spectral.py
│   │   ├── embeddings.py
│   │   ├── mert_embeddings.py
│   │   ├── panns_embeddings.py
│   │   ├── sota_features.py
│   │   ├── source_separation.py
│   │   └── artifacts.py
│   └── model/
│       ├── attribution.py
│       └── calibration.py
│
├── data/
│   ├── a.wav
│   ├── b.wav
│   ├── build_eval_pairs.py
│   ├── build_mippia_eval.py
│   ├── download_all_mippia.py
│   ├── download_mippia.py
│   ├── enhance_with_sonics.py
│   ├── explore_audio_inventory.py
│   ├── eval_pairs_mippia.csv
│   ├── eval_pairs_mippia_top20.csv
│   └── ... local datasets ignored by git
│
├── experiments/
│   ├── robust_validation.py
│   ├── eval_experimental.py
│   ├── run_experiment.py
│   ├── run_subset_eval.py
│   ├── recompute_comparison.py
│   ├── create_subset.py
│   ├── subset_10pairs.csv
│   └── results/
│
├── models/
│   ├── calibrator.json
│   ├── calibrator_logistic.json
│   └── calibrator_gbdt.pkl
│
├── results/
└── tests/
    └── test_sanity.py
```

Ignored local-heavy folders such as downloaded datasets, caches, virtual environments, and reports are handled by [.gitignore](.gitignore).

---

## Sanity Check

Run the lightweight integrity test:

```bash
python tests/test_sanity.py
```

This validates:

- imports resolve
- scorer weights sum to `1.0`
- branch count is `13`
- scorer output shape is valid

---

## Important Limitations

- this is still a small-data evaluation setting
- the repo mixes production code and research experiments in one workspace
- source separation is expensive on CPU
- not every research feature visible in the wider codebase is active in the production weighting
- current evidence is strongest on MIPPIA-style evaluation, not broad cross-generator generalization

---

## Related Documents In Repo

- [SCIENTIFIC_IMPLEMENTATIONS.md](SCIENTIFIC_IMPLEMENTATIONS.md): full technical rationale, experiments, and literature mapping
- [GITHUB_DEPLOYMENT.md](GITHUB_DEPLOYMENT.md): packaging and upload guidance

---

## References

Key references used in the implementation are summarized in [SCIENTIFIC_IMPLEMENTATIONS.md](SCIENTIFIC_IMPLEMENTATIONS.md#26-references), including:

- SONICS
- FakeMusicCaps
- CLAP
- MERT
- PANNs
- HTDemucs
- cover song identification work based on cross-recurrence methods
