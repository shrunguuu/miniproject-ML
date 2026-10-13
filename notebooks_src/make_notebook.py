"""Generates demo.ipynb (the walkthrough notebook) in the project root."""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Music genre from raw audio – Reproduce & Extend
**Reference:** Sawhney, Vasavada & Wang, *Latent Feature Extraction for Musical Genres from Raw Audio* (NeurIPS 2018 workshop)

This notebook walks through the whole project:
1. Data: GTZAN, 4 genres → 8,000 raw-audio vectors of length 500
2. Models: the paper's two-layer NN, vanilla autoencoder and Deep Softmax Autoencoder (DSAE), plus our **1-D CNN**
3. Live training (short runs that do **not** overwrite the saved results)
4. Full results: reported vs ours
5. Live prediction on a song

The full 13-run experiment takes about 1 hour (`python src/run_all.py`), so its trained models and results are loaded from `results/`."""),
code("""import os, sys, json
sys.path.insert(0, "src")          # our code lives in src/
import numpy as np, torch, matplotlib.pyplot as plt
from IPython.display import Markdown, Image, Audio, display

from data import GENRES, build_dataset, make_split
from models import MODELS
from train import run
torch.set_num_threads(os.cpu_count() or 1)
print("torch", torch.__version__, "| genres:", GENRES)"""),

md("## 1. Data\nLoad each song at 22,050 Hz, keep 600,000 samples, average-pool by 40 (≈551 Hz), cut 20 segments of 500 → 8,000 examples. Random 6000/1000/1000 split over segments, as in the paper."),
code("""X, y, song = build_dataset()       # first run reads the .wav files (~10 s), then it is cached
tr, dv, te = make_split(y, song, "segment", seed=0)
print("X:", X.shape, "| examples per genre:", dict(zip(GENRES, np.bincount(y))))
print("train / dev / test:", len(tr), len(dv), len(te))"""),
code("""# One 500-sample input segment per genre: this raw waveform is all the models see
fig, axes = plt.subplots(1, 4, figsize=(14, 2.6), sharey=True)
for g, ax in enumerate(axes):
    ax.plot(X[np.where(y == g)[0][0]], lw=0.8, color="#2a78d6")
    ax.set_title(GENRES[g]); ax.set_xlabel("sample (≈551 Hz)")
plt.tight_layout(); plt.show()"""),
code("""# Listen to a song from the dataset
Audio("data/genres/metal/metal.00010.wav")"""),

md("## 2. Models\nThe first three follow the paper (Fig. 2 and the authors' code). The 1-D CNN is our extension: convolutions share filters over time, so a pattern is recognised wherever it occurs in the clip."),
code("""for name, cls in MODELS.items():
    m = cls()
    print(f"{name:16s} {sum(p.numel() for p in m.parameters()):>8,} parameters")
print()
print(MODELS["cnn1d"]())"""),

md("## 3. Live training\nShort runs with `save=False`, so the saved checkpoints and results are not touched.\n\n**Baseline (paper's two-layer NN, full 200 epochs, ~15 s):**"),
code("""res_nn = run("two_layer_nn", seed=0, save=False)"""),
md("**Our 1-D CNN, 3 epochs only (~1–2 min; the full run uses 40 epochs):**"),
code("""res_cnn = run("cnn1d", seed=0, epochs=3, save=False)"""),
code("""print(f"Two-layer NN  test accuracy: {100*res_nn['test_acc']:.1f} %   (paper: 36.4 %)")
print(f"1-D CNN (3 ep) test accuracy: {100*res_cnn['test_acc']:.1f} %   (full 40-epoch run: 87.3 %)")"""),

md("## 4. Full results (3 seeds, loaded from `results/`)"),
code("""display(Markdown(open("results/tables.md", encoding="utf-8").read()))"""),
code("""display(Image("results/figures/fig7_model_comparison.png"))"""),
code("""display(Image("results/figures/fig4_confusion.png"))"""),
code("""display(Image("results/figures/fig6_pca_logits4.png"))"""),
md("**Finding the paper does not report:** the decoder ends in a ReLU, so it cannot output negative amplitudes. Its output collapses to zero, which means the reconstruction loss does nothing."),
code("""from train import predict
ae = MODELS["vanilla_ae"](); ae.load_state_dict(torch.load("results/checkpoints/vanilla_ae_segment_s0.pt")); ae.eval()
recon = predict(ae, torch.from_numpy(X[te]))["recon"].numpy()
print("max decoder output:", recon.max(), "| share of input samples < 0:", round(float((X < 0).mean()), 3))"""),

md("## 5. Live prediction on a song\nEach model predicts all 20 segments of the song; the song's genre is the majority vote."),
code("""from demo import segments_from_file

def predict_song(path):
    x = torch.from_numpy(segments_from_file(path))
    for name in ("two_layer_nn", "deep_softmax_ae", "cnn1d"):
        m = MODELS[name](); m.load_state_dict(torch.load(f"results/checkpoints/{name}_segment_s0.pt")); m.eval()
        with torch.no_grad():
            p = torch.softmax(m(x)["logits"], 1).numpy()
        votes = np.bincount(p.argmax(1), minlength=4)
        print(f"{name:16s} -> {GENRES[votes.argmax()]:9s} votes {votes.tolist()}  "
              + "  ".join(f"{g}:{v:.2f}" for g, v in zip(GENRES, p.mean(0))))"""),
code("""song_path = "data/genres/jazz/jazz.00007.wav"     # change to any GTZAN file (or your own .wav)
display(Audio(song_path))
predict_song(song_path)"""),
code("""predict_song("data/genres/metal/metal.00010.wav")"""),
md("### Conclusions\n1. The paper reproduces: the DSAE gets 61.7 % vs 65.3 % reported, and the baseline 39.8 % vs 36.4 %.\n2. The DSAE's reconstruction loss is inactive (ReLU output), so its gains come from depth alone.\n3. The 1-D CNN reaches **87.3 %** test accuracy (78.8 % with no song shared across splits): **much better** than every reference model."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                                 "language_info": {"name": "python"}})
nbf.write(nb, os.path.join(ROOT, "demo.ipynb"))
print("wrote demo.ipynb")
