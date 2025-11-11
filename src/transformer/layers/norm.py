import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.eps = 1e-5

        # Trainable parameters applied to the computed normalization.
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # Mean and variance across the feature dimension `emb_dim`.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize the input by mean and standard deviation (`var` is squared
        # standard deviation). Adding `eps` prevents division by zero.
        norm = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm + self.shift
