import torch
import torch.nn as nn
from typing import List

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
}

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], activation="relu", dropout=0.0):
        super().__init__()
        dims = [input_dim] + hidden_dims
        enc_layers = []
        for i in range(len(dims)-1):
            enc_layers.append(nn.Linear(dims[i], dims[i+1]))
            if dropout and dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            enc_layers.append(ACTIVATIONS[activation]())
        self.encoder = nn.Sequential(*enc_layers)

        # decoder: mirror
        dec_dims = list(reversed(dims))
        dec_layers = []
        for i in range(len(dec_dims)-1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i+1]))
            if i < len(dec_dims)-2:  # utolsó réteg legyen lineáris (rekonstrukció)
                if dropout and dropout > 0:
                    dec_layers.append(nn.Dropout(dropout))
                dec_layers.append(ACTIVATIONS[activation]())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

