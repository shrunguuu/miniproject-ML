"""Train / evaluate one model.

Example:
    python src/train.py --model deep_softmax_ae --split segment --seed 0
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from data import GENRES, ROOT, build_dataset, make_split
from models import MODELS

# Hyperparameters. Reference models use the paper's values (Fig. 2, Sec. 4.3, Sec. 5.1):
# Adam, lr 1e-4, batch 512, gamma 0.9, dropout keep-prob 0.9. Epoch budgets follow
# the authors' code (200) / Fig. 3 x-axis (~500).
CONFIG = {
    "two_layer_nn":    dict(lr=1e-4, batch=512, epochs=200, wd=0.0),
    "vanilla_ae":      dict(lr=1e-4, batch=512, epochs=500, wd=0.0),
    "deep_softmax_ae": dict(lr=1e-4, batch=512, epochs=500, wd=0.0, gamma=0.9),
    "cnn1d":           dict(lr=1e-3, batch=64, epochs=40, wd=1e-4),
}
RESULTS = os.path.join(ROOT, "results")


def loss_fn(name, out, x, y, cfg):
    if name == "vanilla_ae":
        return F.mse_loss(out["recon"], x)
    if name == "deep_softmax_ae":  # Eq. 3: gamma * reconstruction + (1 - gamma) * cross-entropy
        g = cfg["gamma"]
        return g * F.mse_loss(out["recon"], x) + (1 - g) * F.cross_entropy(out["logits"], y)
    return F.cross_entropy(out["logits"], y)


@torch.no_grad()
def predict(model, X, bs=1000):
    model.eval()
    outs = [model(X[i:i + bs]) for i in range(0, len(X), bs)]
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def run(name, split="segment", seed=0, epochs=None, verbose=True, save=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = dict(CONFIG[name])
    if epochs:
        cfg["epochs"] = epochs
    X, y, song = build_dataset()
    tr, dv, te = make_split(y, song, split, seed)
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    Xtr, ytr, Xdv, ydv, Xte, yte = X[tr], y[tr], X[dv], y[dv], X[te], y[te]

    model = MODELS[name]()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    is_clf = name != "vanilla_ae"
    hist = {"train_acc": [], "dev_acc": [], "train_loss": []}
    best = (-1.0, None, 0)
    t0 = time.time()
    g = torch.Generator().manual_seed(seed)
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        correct, tot_loss = 0, 0.0
        for i in range(0, len(perm), cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            out = model(Xtr[b])
            loss = loss_fn(name, out, Xtr[b], ytr[b], cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(b)
            if is_clf:
                correct += (out["logits"].argmax(1) == ytr[b]).sum().item()
        hist["train_loss"].append(tot_loss / len(Xtr))
        if is_clf:
            dev_acc = (predict(model, Xdv)["logits"].argmax(1) == ydv).float().mean().item()
            hist["train_acc"].append(correct / len(Xtr))  # running accuracy, dropout on (as in the paper's curves)
            hist["dev_acc"].append(dev_acc)
            if dev_acc > best[0]:
                best = (dev_acc, copy.deepcopy(model.state_dict()), ep + 1)
        if verbose and (ep + 1) % max(1, cfg["epochs"] // 10) == 0:
            msg = f"[{name}|{split}|s{seed}] epoch {ep+1}/{cfg['epochs']} loss {hist['train_loss'][-1]:.4f}"
            if is_clf:
                msg += f" train {hist['train_acc'][-1]:.3f} dev {hist['dev_acc'][-1]:.3f}"
            print(msg, flush=True)

    res = {"model": name, "split": split, "seed": seed, "config": cfg,
           "train_seconds": round(time.time() - t0, 1), "history": hist,
           "n_params": sum(p.numel() for p in model.parameters())}
    if is_clf:
        final_test = (predict(model, Xte)["logits"].argmax(1) == yte).float().mean().item()
        model.load_state_dict(best[1])  # checkpoint with best dev accuracy
        res["best_epoch"] = best[2]
        res["final_epoch_test_acc"] = final_test
        for part, (Xp, yp) in {"train": (Xtr, ytr), "dev": (Xdv, ydv), "test": (Xte, yte)}.items():
            pred = predict(model, Xp)["logits"].argmax(1).numpy()
            res[f"{part}_acc"] = accuracy_score(yp.numpy(), pred)
        pred = predict(model, Xte)["logits"].argmax(1).numpy()
        p, r, f, _ = precision_recall_fscore_support(yte.numpy(), pred, labels=range(4), zero_division=0)
        res["per_class"] = {gname: {"precision": p[i], "recall": r[i], "f1": f[i]} for i, gname in enumerate(GENRES)}
        res["macro_f1"] = float(f.mean())
        res["confusion"] = confusion_matrix(yte.numpy(), pred, labels=range(4)).tolist()
    else:
        res["test_mse"] = F.mse_loss(predict(model, Xte)["recon"], Xte).item()

    tag = f"{name}_{split}_s{seed}"
    if save:  # --no-save leaves the saved checkpoints and results untouched (useful for demos)
        os.makedirs(os.path.join(RESULTS, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(RESULTS, "runs"), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(RESULTS, "checkpoints", tag + ".pt"))
        with open(os.path.join(RESULTS, "runs", tag + ".json"), "w") as fh:
            json.dump(res, fh, default=float)
    if verbose and is_clf:
        print(f"[{tag}] best dev epoch {res['best_epoch']}  train {res['train_acc']:.3f}  "
              f"dev {res['dev_acc']:.3f}  test {res['test_acc']:.3f}  macroF1 {res['macro_f1']:.3f}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--split", choices=["segment", "song"], default="segment")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-save", action="store_true", help="do not overwrite saved checkpoints/results")
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    run(a.model, a.split, a.seed, a.epochs, save=not a.no_save)
