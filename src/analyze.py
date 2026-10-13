"""Aggregate run results into tables and regenerate every figure.

    python src/analyze.py
Reads results/runs/*.json and results/checkpoints/*.pt, writes results/summary.json,
results/tables.md and results/figures/*.png.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from data import GENRES, ROOT, build_dataset, make_split
from models import MODELS
from train import predict

RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

# Numbers reported in the reference paper (Table 1 and Table 2).
REPORTED = {
    "two_layer_nn": {"train_acc": 0.520, "dev_acc": 0.381, "test_acc": 0.364},
    "deep_softmax_ae": {"train_acc": 0.949, "dev_acc": 0.641, "test_acc": 0.653,
                        "per_class": {"classical": (0.783, 0.775, 0.779), "jazz": (0.606, 0.627, 0.616),
                                      "metal": (0.515, 0.554, 0.5337), "pop": (0.670, 0.608, 0.638)}},
}
NAMES = {"two_layer_nn": "Two-layer NN", "deep_softmax_ae": "Deep Softmax AE",
         "cnn1d": "1-D CNN (ours)", "vanilla_ae": "Vanilla AE"}

# Style: validated reference palette (dataviz skill); 3 categorical slots max per chart.
SURF, INK, INK2, GRID, BG_PTS = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#cfcec8"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.titlesize": 10, "axes.titleweight": "bold", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def load_runs():
    runs = {}
    for f in glob.glob(os.path.join(RES, "runs", "*.json")):
        r = json.load(open(f))
        runs[(r["model"], r["split"], r["seed"])] = r
    return runs


def agg(runs, model, split, key):
    vals = [r[key] for (m, s, _), r in runs.items() if m == model and s == split and key in r]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else None


def agg_class(runs, model, split, genre, metric):
    vals = [r["per_class"][genre][metric] for (m, s, _), r in runs.items() if m == model and s == split]
    return float(np.mean(vals)), float(np.std(vals))


def pct(x):
    return f"{100 * x[0]:.1f} ± {100 * x[1]:.1f}" if x[2] > 1 else f"{100 * x[0]:.1f}"


def tables(runs):
    summary, lines = {}, []
    clf = ["two_layer_nn", "deep_softmax_ae", "cnn1d"]
    for split in ("segment", "song"):
        summary[split] = {m: {k: agg(runs, m, split, k) for k in
                              ("train_acc", "dev_acc", "test_acc", "macro_f1", "final_epoch_test_acc")}
                          for m in clf}
        for m in clf:
            summary[split][m]["per_class"] = {g: {mt: agg_class(runs, m, split, g, mt)
                                                  for mt in ("precision", "recall", "f1")} for g in GENRES}

    lines.append("## Table 1 – Classification accuracy (%), segment-level split (as in paper), mean ± std over 3 seeds\n")
    lines.append("| Model | Train (reported) | Train (ours) | Dev (reported) | Dev (ours) | Test (reported) | Test (ours) |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in clf:
        rep = REPORTED.get(m, {})
        row = [NAMES[m]]
        for k in ("train_acc", "dev_acc", "test_acc"):
            row += [f"{100 * rep[k]:.1f}" if k in rep else "–", pct(summary["segment"][m][k])]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\nOurs = checkpoint with best dev accuracy, evaluated without dropout.\n")
    lines.append("## Table 1b – Same runs read at the last epoch (the paper's apparent protocol; train = running "
                 "accuracy with dropout on, as logged in the authors' code)\n")
    lines.append("| Model | Epochs | Train (ours) | Dev (ours) | Test (ours) |")
    lines.append("|---|---|---|---|---|")
    for m in clf:
        rs = [r for (mm, s, _), r in runs.items() if mm == m and s == "segment"]
        tr = [r["history"]["train_acc"][-1] for r in rs]
        dv = [r["history"]["dev_acc"][-1] for r in rs]
        te = [r["final_epoch_test_acc"] for r in rs]
        f = lambda v: f"{100 * np.mean(v):.1f} ± {100 * np.std(v):.1f}"
        summary["segment"][m]["last_epoch"] = {"train": float(np.mean(tr)), "dev": float(np.mean(dv)),
                                               "test": float(np.mean(te))}
        lines.append(f"| {NAMES[m]} | {rs[0]['config']['epochs']} | {f(tr)} | {f(dv)} | {f(te)} |")
    va = runs.get(("vanilla_ae", "segment", 0))
    if va:
        X, _, _ = build_dataset()
        summary["vanilla_ae_test_mse"] = va["test_mse"]
        summary["input_variance"] = float(X.var())
        lines.append(f"\nVanilla AE test reconstruction MSE = {va['test_mse']:.5f} vs. input variance "
                     f"{X.var():.5f} (the ReLU output layer cannot produce negative amplitudes).\n")

    lines.append("\n## Table 2 – Deep Softmax AE per-class test metrics, reported vs ours (mean over 3 seeds)\n")
    lines.append("| Genre | Precision (rep.) | Precision (ours) | Recall (rep.) | Recall (ours) | F1 (rep.) | F1 (ours) |")
    lines.append("|---|---|---|---|---|---|---|")
    for g in GENRES:
        rp = REPORTED["deep_softmax_ae"]["per_class"][g]
        ours = summary["segment"]["deep_softmax_ae"]["per_class"][g]
        lines.append(f"| {g} | {rp[0]:.3f} | {ours['precision'][0]:.3f} | {rp[1]:.3f} | {ours['recall'][0]:.3f} "
                     f"| {rp[2]:.3f} | {ours['f1'][0]:.3f} |")

    lines.append("\n## Table 3 – Per-class test F1, all classifiers (segment split, mean over 3 seeds)\n")
    lines.append("| Model | " + " | ".join(GENRES) + " | Macro F1 |")
    lines.append("|---|" + "---|" * (len(GENRES) + 1))
    for m in clf:
        pc = summary["segment"][m]["per_class"]
        lines.append(f"| {NAMES[m]} | " + " | ".join(f"{pc[g]['f1'][0]:.3f}" for g in GENRES)
                     + f" | {summary['segment'][m]['macro_f1'][0]:.3f} |")

    lines.append("\n## Table 4 – Test accuracy (%) on segment split vs stricter song-level split (seed 0 for song split)\n")
    lines.append("| Model | Segment split (paper protocol) | Song-level split (no song in two sets) |")
    lines.append("|---|---|---|")
    for m in clf:
        lines.append(f"| {NAMES[m]} | {pct(summary['segment'][m]['test_acc'])} | {pct(summary['song'][m]['test_acc'])} |")

    with open(os.path.join(RES, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(RES, "tables.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary


def load_model(name, split="segment", seed=0):
    m = MODELS[name]()
    m.load_state_dict(torch.load(os.path.join(RES, "checkpoints", f"{name}_{split}_s{seed}.pt")))
    return m.eval()


def facet_pca(rows, path, suptitle):
    """rows: list of (row_title, 2-D points, labels). One panel per genre (genre highlighted, rest grey)."""
    fig, axes = plt.subplots(len(rows), 4, figsize=(11, 2.55 * len(rows)), squeeze=False)
    for r, (title, P, lab) in enumerate(rows):
        for c, g in enumerate(GENRES):
            ax = axes[r, c]
            ax.scatter(P[lab != c, 0], P[lab != c, 1], s=4, c=BG_PTS, linewidths=0, rasterized=True)
            ax.scatter(P[lab == c, 0], P[lab == c, 1], s=6, c=S1, linewidths=0, alpha=0.8, rasterized=True)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(g, color=INK)
            if c == 0:
                ax.set_ylabel(title, color=INK, fontsize=9, fontweight="bold")
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figures(runs, summary):
    X, y, song = build_dataset()
    _, dv, te = make_split(y, song, "segment", 0)
    held = np.concatenate([dv, te])
    Xh, yh = torch.from_numpy(X[held]), y[held]

    def pca2(Z):
        return PCA(2, random_state=0).fit_transform(np.asarray(Z))

    # Fig 1: raw data
    facet_pca([("Raw 500-d input", pca2(Xh), yh)], os.path.join(FIG, "fig1_raw_pca.png"),
              "PCA of raw 500-d audio segments (held-out dev+test): genres overlap")

    # Fig 5: 64-d bottlenecks
    zv = predict(load_model("vanilla_ae"), Xh)["z"]
    zd = predict(load_model("deep_softmax_ae"), Xh)["z"]
    facet_pca([("Vanilla AE", pca2(zv), yh), ("Deep Softmax AE", pca2(zd), yh)],
              os.path.join(FIG, "fig5_pca_bottleneck64.png"), "PCA of 64-d bottleneck encodings")

    # Fig 6: 4-d logits (+ CNN)
    rows = [(NAMES[m], pca2(predict(load_model(m), Xh)["logits"]), yh)
            for m in ("two_layer_nn", "deep_softmax_ae", "cnn1d")]
    facet_pca(rows, os.path.join(FIG, "fig6_pca_logits4.png"), "PCA of 4-d logits used as genre encodings")

    # Fig 3: training curves (seed 0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, m in zip(axes, ("deep_softmax_ae", "cnn1d")):
        h = runs[(m, "segment", 0)]["history"]
        ep = np.arange(1, len(h["train_acc"]) + 1)
        ax.plot(ep, h["train_acc"], color=S1, lw=2)
        ax.plot(ep, h["dev_acc"], color=S2, lw=2)
        ax.text(ep[-1], h["train_acc"][-1] + 0.02, "train", color=INK2, ha="right", va="bottom")
        ax.text(ep[-1], h["dev_acc"][-1] - 0.03, "dev", color=INK2, ha="right", va="top")
        ax.set_ylim(0, 1.02), ax.set_xlabel("epoch"), ax.set_ylabel("accuracy")
        ax.set_title(NAMES[m] + " – accuracy per epoch", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_training_curves.png"), dpi=160)
    plt.close(fig)

    # Fig 4: confusion matrices (seed 0, test)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, m in zip(axes, ("deep_softmax_ae", "cnn1d")):
        cm = np.array(runs[(m, "segment", 0)]["confusion"], dtype=float)
        cm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.55 else INK, fontsize=9)
        ax.set_xticks(range(4), GENRES), ax.set_yticks(range(4), GENRES)
        ax.set_xlabel("predicted"), ax.set_ylabel("true")
        ax.set_title(NAMES[m] + " (test, row-normalised)", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_confusion.png"), dpi=160)
    plt.close(fig)

    # Fig 7: model comparison, test accuracy, segment vs song split
    models = ["two_layer_nn", "deep_softmax_ae", "cnn1d"]
    xs = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for k, (split, col, lab) in enumerate((("segment", S1, "segment split (paper protocol)"),
                                           ("song", S2, "song-level split"))):
        vals = [summary[split][m]["test_acc"] for m in models]
        mean = [100 * v[0] for v in vals]
        err = [100 * v[1] for v in vals]
        bars = ax.bar(xs + (k - 0.5) * (w + 0.02), mean, w, color=col, label=lab,
                      yerr=err if split == "segment" else None, error_kw=dict(ecolor=INK2, lw=1, capsize=3))
        for i, (b, v) in enumerate(zip(bars, mean)):
            top = max(v + err[i] if split == "segment" else v, 100 * (REPORTED.get(models[i], {}).get("test_acc") or 0))
            ax.text(b.get_x() + b.get_width() / 2, top + 2, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    rep = [REPORTED.get(m, {}).get("test_acc") for m in models]
    for i, r in enumerate(rep):
        if r:
            ax.plot([i - 0.42, i + 0.42], [100 * r] * 2, color=INK, lw=1.2, ls="--")
    ax.plot([], [], color=INK, lw=1.2, ls="--", label="reported in paper")
    ax.axhline(25, color=INK2, lw=0.8, ls=":")
    ax.text(-0.62, 26, "chance (25%)", color=INK2, fontsize=8, ha="left")
    ax.set_xticks(xs, [NAMES[m] for m in models]), ax.set_ylabel("test accuracy (%)"), ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Test accuracy by model", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_model_comparison.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    runs = load_runs()
    s = tables(runs)
    figures(runs, s)
    print("figures written to", FIG)
