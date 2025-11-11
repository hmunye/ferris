import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, input):
        # Standard deviation across the feature dimension.
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize the input by mean and std (var is squared std). Epsilon
        # prevents division by zero.
        norm = (input - mean) / torch.sqrt(var + self.eps)

        # Apply scaling and shifting (trainable parameters)
        return self.scale * norm + self.shift
