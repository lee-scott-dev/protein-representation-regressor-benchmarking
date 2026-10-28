# Imputing Missing DMS Variant-Effect Scores: Benchmarking Protein Language Models and Interpretable Biological Features

Deep mutational scanning (DMS) measures the functional effects of many protein variants at once, but the resulting sequence-function maps are rarely complete. This repository contains a pipeline (produced for an MSc dissertation) for benchmarking methods that impute the missing scores.

The approach is simple: an **encoder** turns each single-residue variant into a numerical representation, and a **regressor** maps that representation to a functional score. Every encoder-regressor pair is evaluated with 5-fold cross-validation on the variants that have measured scores, and compared against a naïve positional-mean baseline.

## Supported methods

**Protein language model encoders**

| Family | Source | Module |
|---|---|---|
| ESM-1, ESM-1b, ESM-1v, ESM-2 (8M to 15B) | FAIR (`fair-esm`) | `esm_encoder.py` |
| ESM-C (300M, 600M) | EvolutionaryScale | `esmc_encoder.py` |
| ESM-3 open | EvolutionaryScale | `esm3_encoder.py` |
| ProtBERT | ProtTrans | `protbert_encoder.py` |
| ProtT5 | ProtTrans | `prott5_encoder.py` |

**Hand-crafted representations**

- **One-hot** variant descriptor (`onehot_encoder.py`)
- **BOHN** (`bohn.py`): BLOSUM62 substitution scores (B) + one-hot (OH) descriptors + a naïve (N) within-fold positional-mean feature, falling back to the overall mean for unseen positions

**Regressors** (scikit-learn): Linear, Lasso, ElasticNet, SGD, PLS, Gaussian process, Kernel ridge, SVR, KNN, Decision tree, Random forest, Extra trees, AdaBoost and Gradient boosting. Each can be hyperparameter-tuned from a grid via `hyperparameter_tuning.py`.

**Baselines**: a naïve positional-mean baseline (`naive_baseline.py`) and a BLOSUM62 assessment against functional scores (`blosum.py`).

See the tables in `embedding_and_imputation_pipeline.ipynb` for the exact calling strings of every supported model.

## Repository layout

```
.
├── embedding_and_imputation_pipeline.ipynb   # (*) Main benchmarking pipeline for representations and regressors
├── exploratory_data_analysis.ipynb           # Dataset validation and EDA
├── evaluation.ipynb                          # Results heatmaps and per-representation plots
├── esm15b_convertor.ipynb                    # Converts per-variant .pt files to the pipeline's .pkl format
├── data_loader.py                # Loads wild type and scores for a dataset
├── informatics.py                # EDA summaries and plotting helpers
├── sequence_function_heatmap.py  # Sequence-function heatmap plotting
├── onehot_encoder.py             # One-hot feature representations
├── blosum.py                     # BLOSUM62 scores and correlations 
├── bohn.py                       # BOHN feature representations
├── esm_encoder.py                # ESM-1 / 1b / 1v / 2 (fair-esm) representations
├── esmc_encoder.py               # ESM-C representations
├── esm3_encoder.py               # ESM-3 open representations
├── protbert_encoder.py           # ProtBERT representations
├── prott5_encoder.py             # ProtT5 representations
├── naive_baseline.py             # Positional-mean baseline calculation
└── hyperparameter_tuning.py      # Grid-search tuning for regressors
```

## Environment setup

Python 3.11 was used (3.11.15). Versions are in `requirements.txt`:

Seeds are fixed (`42`) for NumPy, PyTorch and the cross-validation splits.

## Input format

Ensure each protein is a folder under `datasets/` (e.g. `datasets/HRAS/`) containing:

| File | Contents |
|---|---|
| `wild_type.fasta` | Reference (unmutated) sequence |
| `all_variants.fasta` | Every single-residue variant (e.g. `M1A`) with its full mutant sequence. Clean this file if removing outliers. |
| `scores.csv` | Measured functional score, one row per scored variant. Clean this file if removing outliers. |

Variant names follow `<WT residue><position><mutant residue>`, so `T2A` is wild-type T at position 2 mutated to A. The datasets used here were **HRAS**, **P53** and **PTEN**; to use other proteins, add a folder and change the `datasets` list or other occurences within the notebooks.