"""Models. The first three follow the reference paper (Fig. 2 and the authors'
TensorFlow code); CNN1D is our extension model."""
import torch
import torch.nn as nn


def xavier(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


# ---------------------------------------------------------------- reference
class TwoLayerNN(nn.Module):
    """Paper Sec. 4.1: one hidden layer of 128 units, tanh, softmax output."""

    def __init__(self, d_in=500, n_classes=4):
        super().__init__()
        self.hidden = nn.Linear(d_in, 128)
        self.out = nn.Linear(128, n_classes)
        self.apply(xavier)

    def forward(self, x):
        return {"logits": self.out(torch.tanh(self.hidden(x)))}


class Encoder(nn.Module):
    """500 -> 256 -> 192 -> 128 -> 64 (activations as in the authors' code)."""

    def __init__(self, d_in=500, p_drop=0.1):
        super().__init__()
        self.l1, self.l2 = nn.Linear(d_in, 256), nn.Linear(256, 192)
        self.l3, self.l4 = nn.Linear(192, 128), nn.Linear(128, 64)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        a = torch.sigmoid(self.l1(x))
        a = self.drop(torch.tanh(self.l2(a)))
        a = torch.relu(self.l3(a))
        return torch.relu(self.l4(a))


class Decoder(nn.Module):
    """64 -> 128 -> 192 -> 256 -> 500."""

    def __init__(self, d_out=500, p_drop=0.1):
        super().__init__()
        self.l1, self.l2 = nn.Linear(64, 128), nn.Linear(128, 192)
        self.l3, self.l4 = nn.Linear(192, 256), nn.Linear(256, d_out)
        self.drop = nn.Dropout(p_drop)

    def forward(self, z):
        a = torch.sigmoid(self.l1(z))
        a = self.drop(torch.sigmoid(self.l2(a)))
        a = torch.relu(self.l3(a))
        return torch.relu(self.l4(a))


class VanillaAutoencoder(nn.Module):
    """Paper Sec. 4.2 (deep version, top half of Fig. 2). Reconstruction only."""

    def __init__(self, d_in=500, p_drop=0.1):
        super().__init__()
        self.encoder, self.decoder = Encoder(d_in, p_drop), Decoder(d_in, p_drop)
        self.apply(xavier)

    def forward(self, x):
        z = self.encoder(x)
        return {"z": z, "recon": self.decoder(z)}


class DeepSoftmaxAutoencoder(nn.Module):
    """Paper Sec. 4.3 / Fig. 2: autoencoder + classifier 64 -> 32 -> 16 -> 4 on the bottleneck."""

    def __init__(self, d_in=500, n_classes=4, p_drop=0.1):
        super().__init__()
        self.encoder, self.decoder = Encoder(d_in, p_drop), Decoder(d_in, p_drop)
        self.c1, self.c2, self.c3 = nn.Linear(64, 32), nn.Linear(32, 16), nn.Linear(16, n_classes)
        self.apply(xavier)

    def forward(self, x):
        z = self.encoder(x)
        h = torch.tanh(self.c2(torch.tanh(self.c1(z))))
        return {"z": z, "recon": self.decoder(z), "logits": self.c3(h)}


# ---------------------------------------------------------------- extension
class CNN1D(nn.Module):
    """1-D convolutional network on the same 500-sample raw waveform.

    Four conv blocks (kernel 9) with batch norm, ReLU and max-pooling, then
    global average pooling over time and a linear softmax layer.  Weight
    sharing across time makes it translation-invariant, which an MLP on raw
    samples is not.  ~0.24M parameters (the Deep Softmax Autoencoder has ~0.42M).
    """

    def __init__(self, n_classes=4, p_drop=0.3):
        super().__init__()

        def block(cin, cout, pool=True):
            layers = [nn.Conv1d(cin, cout, 9, padding=4), nn.BatchNorm1d(cout), nn.ReLU()]
            if pool:
                layers.append(nn.MaxPool1d(2))
            return layers

        self.features = nn.Sequential(*block(1, 32), *block(32, 64), *block(64, 128), *block(128, 128, pool=False))
        self.drop = nn.Dropout(p_drop)
        self.out = nn.Linear(128, n_classes)
        self.apply(xavier)

    def forward(self, x):
        h = self.features(x.unsqueeze(1)).mean(dim=2)  # global average pool -> (B, 128)
        return {"z": h, "logits": self.out(self.drop(h))}


MODELS = {
    "two_layer_nn": TwoLayerNN,
    "vanilla_ae": VanillaAutoencoder,
    "deep_softmax_ae": DeepSoftmaxAutoencoder,
    "cnn1d": CNN1D,
}
