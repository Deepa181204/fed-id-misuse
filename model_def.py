# model_def.py
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Small autoencoder with flexible input_dim.
    We'll construct with input_dim = number of features at runtime.
    Architecture matches the earlier example: input -> 32 -> 8 -> 4 -> decode.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
