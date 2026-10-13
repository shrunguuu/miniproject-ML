"""Live demo: predict the genre of a song with the trained models.

    python src/demo.py                       # random held-out test song
    python src/demo.py path/to/song.wav      # any audio file librosa can read

The song is cut into the same 20 x 500-sample pooled segments used in training;
each model predicts every segment and the song label is the majority vote.
"""
import os
import sys

import numpy as np
import torch

from data import GENRES, KEEP, N_SEGMENTS, POOL, ROOT, SEG_LEN, build_dataset, make_split
from models import MODELS

CKPT = os.path.join(ROOT, "results", "checkpoints")


def segments_from_file(path):
    import librosa
    audio, _ = librosa.load(path, sr=22050)
    audio = np.pad(audio[:KEEP], (0, max(0, KEEP - len(audio))))
    pooled = audio.reshape(-1, POOL).mean(1)
    return np.stack([pooled[j * SEG_LEN:(j + 1) * SEG_LEN] for j in range(N_SEGMENTS)]).astype(np.float32)


def main():
    if len(sys.argv) > 1:
        segs, truth, name = segments_from_file(sys.argv[1]), None, sys.argv[1]
    else:  # a random song that has segments in the test split
        X, y, song = build_dataset()
        _, _, te = make_split(y, song, "segment", 0)
        sid = np.random.choice(np.unique(song[te]))
        idx = np.where(song == sid)[0]
        segs, truth = X[idx], GENRES[y[idx][0]]
        name = f"GTZAN {truth}.{sid % 100:05d}"
    print(f"\nSong: {name}" + (f"   (true genre: {truth})" if truth else ""))
    x = torch.from_numpy(segs)
    for m in ("two_layer_nn", "deep_softmax_ae", "cnn1d"):
        model = MODELS[m]()
        model.load_state_dict(torch.load(os.path.join(CKPT, f"{m}_segment_s0.pt")))
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(x)["logits"], 1).numpy()
        votes = np.bincount(probs.argmax(1), minlength=4)
        mean_p = probs.mean(0)
        pred = GENRES[int(votes.argmax())]
        detail = "  ".join(f"{g}:{p:.2f}" for g, p in zip(GENRES, mean_p))
        print(f"  {m:16s} -> {pred:9s}  votes {votes.tolist()}  mean prob [{detail}]")


if __name__ == "__main__":
    main()
