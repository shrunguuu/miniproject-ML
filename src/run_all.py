"""Run every experiment in the report (about 1 hour on a 2-core CPU)."""
import torch
from train import run

torch.set_num_threads(2)
JOBS = [("vanilla_ae", "segment", 0)]
JOBS += [(m, "segment", s) for s in (0, 1, 2) for m in ("two_layer_nn", "deep_softmax_ae", "cnn1d")]
JOBS += [(m, "song", 0) for m in ("two_layer_nn", "deep_softmax_ae", "cnn1d")]

if __name__ == "__main__":
    for m, split, s in JOBS:
        run(m, split, s)
