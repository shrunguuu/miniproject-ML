# Reproduce & Extend: Latent Feature Extraction for Musical Genres from Raw Audio

UE24CS352A Machine Learning – Mini-Project

**Reference:** A. Sawhney, V. Vasavada, W. Wang, *Latent Feature Extraction for Musical Genres from Raw Audio*, NeurIPS 2018 workshop (Stanford CS229 project).

**Task:** classify 4 GTZAN genres (classical, jazz, metal, pop) directly from raw audio, with no hand-made features.

- **Part 1 – Reproduce.** We rebuilt the paper's pipeline and its three models: a two-layer neural network, a vanilla autoencoder, and the Deep Softmax Autoencoder (DSAE).
- **Part 2 – Extend.** We added a **1-D convolutional network (CNN)** trained on exactly the same inputs, split and metrics.

## Results at a glance

Test accuracy on the paper's segment-level split, mean ± std over 3 seeds:

| Model | Reported | Ours |
|---|---|---|
| Two-layer NN | 36.4 % | 39.8 ± 1.3 % |
| Deep Softmax Autoencoder | 65.3 % | 61.7 ± 1.2 % |
| **1-D CNN (extension)** | – | **87.3 ± 0.7 %** |

The reproduction lands within about 3.5 points of the paper. The CNN beats the paper's best model by about 25 points.

All tables are in [`results/tables.md`](results/tables.md), and all figures are in [`results/figures/`](results/figures).

## Repository layout

```
src/
  data.py      GTZAN loading, average pooling, segmenting, train/dev/test split
  models.py    TwoLayerNN, VanillaAutoencoder, DeepSoftmaxAutoencoder (reference), CNN1D (extension)
  train.py     train + evaluate one model (accuracy, per-class P/R/F1, confusion matrix)
  run_all.py   every experiment in the report
  analyze.py   tables (reported vs ours) + figures
  demo.py      live demo: predict the genre of a song
demo.ipynb     notebook walkthrough of the whole project (for the live demo)
results/
  runs/        one JSON per run (config, per-epoch history, metrics)
  checkpoints/ trained weights (used by demo.py and analyze.py)
  figures/     fig1 raw PCA, fig3 curves, fig4 confusion, fig5/6 PCA of encodings, fig7 comparison
  tables.md    all result tables
writeup/       two-page write-up (PDF) and its source
```

## Setup

The code needs Python 3.10 or newer. A GPU is not needed: everything was run on a 2-core CPU.

```bash
pip install -r requirements.txt
```

**Data.** Get the GTZAN Genre Collection, either from Kaggle (`carlthome/gtzan-genre-collection` or `andradaolteanu/gtzan-dataset-music-genre-classification`) or from any mirror of the original `genres.tar.gz`. Put the four genre folders here:

```
data/genres/classical/*.wav   data/genres/jazz/*.wav
data/genres/metal/*.wav       data/genres/pop/*.wav
```

**Data cleaning:** `src/data.py` checks every file and skips any it cannot decode (or that is much shorter than 30 s), printing which ones. The Kaggle copy `andradaolteanu/gtzan-dataset-music-genre-classification` has one corrupted file, `jazz/jazz.00054.wav`; with that copy you get 399 songs (7,980 segments, test set 980), so your numbers can differ slightly from ours. Our copy had no bad files, so all 400 songs were used.

If your copy holds `.au` files, convert them first, for example with `sox x.au x.wav` or `ffmpeg -i x.au x.wav`. We used the copy committed in `github.com/EsratMaria/MusicGenreRecogniton` (`genres/`), which has the same 100 × 30 s clips per genre as the original.

## Run

**Notebook (VS Code / Jupyter):** open `demo.ipynb` and Run All. It walks through the data, the models, short live training runs (which do not overwrite saved results), the full results and live song prediction (~2 min).

**Command line:**

```bash
python src/data.py                                       # builds data/processed_gtzan4.npz (8000 x 500), ~10 s
python src/train.py --model deep_softmax_ae --seed 0     # one model (two_layer_nn | vanilla_ae | deep_softmax_ae | cnn1d)
python src/run_all.py                                    # all 13 runs, ~1 h on CPU
python src/analyze.py                                    # tables + figures from results/runs
python src/demo.py                                       # live demo on a random held-out song
python src/demo.py data/genres/jazz/jazz.00007.wav       # ...or on any audio file
```

`train.py --split song` uses the stricter song-level split described below. Add `--no-save` to train without overwriting the saved checkpoints and results (e.g. `python src/train.py --model cnn1d --epochs 3 --no-save` for a 1-minute live demo).

## What exactly was reproduced

| Item | Paper / authors' code | This repo |
|---|---|---|
| Data | GTZAN, 4 genres × 100 songs | same |
| Preprocessing | librosa at 22,050 Hz, first 600,000 samples, average-pool 40, then 20 segments × 500 → 8,000 × 500 | same |
| Split | random 6000 / 1000 / 1000 over **segments** | same, with seeds 0, 1 and 2 |
| Two-layer NN | 500 → 128 (tanh) → 4, softmax cross-entropy | same |
| Autoencoder | 500-256-192-128-**64**-128-192-256-500 | same, with activations taken from the authors' code |
| DSAE classifier | 64 → 32 → 16 → 4 on the bottleneck | same |
| Loss (Eq. 3) | γ·MSE + (1−γ)·CE, γ = 0.9 | same |
| Optimiser | Adam, lr 1e-4, batch 512, dropout keep-prob 0.9 | same; 200 epochs for the NN, 500 for the autoencoders (Fig. 3) |
| Metrics | accuracy, per-class precision, recall and F1, confusion matrix, PCA plots | same |

**Model selection.** For every model we evaluate the checkpoint with the best dev accuracy. `tables.md` (Table 1b) also gives last-epoch numbers.

## Extension: 1-D CNN

The architecture is four blocks of Conv1d (kernel 9) → BatchNorm → ReLU → MaxPool, with 32, 64, 128 and 128 channels. These are followed by global average pooling, dropout 0.3 and a linear layer to 4 classes. The model has 0.24 M parameters, against 0.42 M for the DSAE. It is trained with Adam (lr 1e-3, batch 64, weight decay 1e-4) for 40 epochs.

**Why this model.** A segment of raw audio is a time series in which the same patterns (onsets, periodicities, timbre) can appear at any time offset. An MLP sees each of the 500 input positions as an unrelated feature, so it has to relearn every pattern at every position, and it overfits. A CNN shares its filters across time and pools them, which makes it translation-invariant. It learns what amounts to a filter bank directly from the waveform, and it does so with fewer parameters. None of the paper's experiments uses a convolutional model.

## Key findings

1. **The reproduction holds.** The two-layer NN gives 39.8 % (paper 36.4 %) and the DSAE gives 61.7 % (paper 65.3 %). The per-class pattern also matches: classical is easiest, and metal and pop are confused with each other.
2. **The reconstruction branch does nothing in this design.** The decoder ends in a ReLU, as in the authors' code, so it cannot output the negative half of a waveform. It collapses to all zeros: test MSE is 0.01173, against an input variance of 0.01132. As a result, the DSAE behaves like a deep MLP classifier. Its large train–test gap (97 % train vs 60 % test at the last epoch) is ordinary overfitting.
3. **The CNN is far better:** 87.3 % test accuracy and macro-F1 0.87, against 0.62 for the DSAE. It is better on every genre, especially metal and pop.
4. **Data leakage check.** The paper's split puts segments of the same song into both train and test. On a song-level split, where no song appears in two sets, the CNN drops to 78.8 %. That is still well above the DSAE (64.6 %) and the two-layer NN (45.6 %).
