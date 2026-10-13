## Table 1 – Classification accuracy (%), segment-level split (as in paper), mean ± std over 3 seeds

| Model | Train (reported) | Train (ours) | Dev (reported) | Dev (ours) | Test (reported) | Test (ours) |
|---|---|---|---|---|---|---|
| Two-layer NN | 52.0 | 51.9 ± 0.9 | 38.1 | 39.4 ± 0.6 | 36.4 | 39.8 ± 1.3 |
| Deep Softmax AE | 94.9 | 88.7 ± 2.4 | 64.1 | 63.0 ± 1.5 | 65.3 | 61.7 ± 1.2 |
| 1-D CNN (ours) | – | 94.6 ± 1.8 | – | 87.4 ± 0.8 | – | 87.3 ± 0.7 |

Ours = checkpoint with best dev accuracy, evaluated without dropout.

## Table 1b – Same runs read at the last epoch (the paper's apparent protocol; train = running accuracy with dropout on, as logged in the authors' code)

| Model | Epochs | Train (ours) | Dev (ours) | Test (ours) |
|---|---|---|---|---|
| Two-layer NN | 200 | 53.4 ± 0.4 | 38.1 ± 0.8 | 39.8 ± 1.3 |
| Deep Softmax AE | 500 | 97.0 ± 0.8 | 59.5 ± 2.0 | 60.1 ± 1.6 |
| 1-D CNN (ours) | 40 | 95.1 ± 0.1 | 78.5 ± 3.9 | 79.0 ± 5.2 |

Vanilla AE test reconstruction MSE = 0.01173 vs. input variance 0.01132 (the ReLU output layer cannot produce negative amplitudes).


## Table 2 – Deep Softmax AE per-class test metrics, reported vs ours (mean over 3 seeds)

| Genre | Precision (rep.) | Precision (ours) | Recall (rep.) | Recall (ours) | F1 (rep.) | F1 (ours) |
|---|---|---|---|---|---|---|
| classical | 0.783 | 0.764 | 0.775 | 0.770 | 0.779 | 0.766 |
| jazz | 0.606 | 0.580 | 0.627 | 0.571 | 0.616 | 0.575 |
| metal | 0.515 | 0.514 | 0.554 | 0.523 | 0.534 | 0.518 |
| pop | 0.670 | 0.621 | 0.608 | 0.608 | 0.638 | 0.613 |

## Table 3 – Per-class test F1, all classifiers (segment split, mean over 3 seeds)

| Model | classical | jazz | metal | pop | Macro F1 |
|---|---|---|---|---|---|
| Two-layer NN | 0.562 | 0.279 | 0.328 | 0.343 | 0.378 |
| Deep Softmax AE | 0.766 | 0.575 | 0.518 | 0.613 | 0.618 |
| 1-D CNN (ours) | 0.892 | 0.845 | 0.879 | 0.872 | 0.872 |

## Table 4 – Test accuracy (%) on segment split vs stricter song-level split (seed 0 for song split)

| Model | Segment split (paper protocol) | Song-level split (no song in two sets) |
|---|---|---|
| Two-layer NN | 39.8 ± 1.3 | 45.6 |
| Deep Softmax AE | 61.7 ± 1.2 | 64.6 |
| 1-D CNN (ours) | 87.3 ± 0.7 | 78.8 |
