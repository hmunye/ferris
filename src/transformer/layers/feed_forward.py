import torch.nn as nn

from .gelu import GELU


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Linear layer to project the input embeddings to a larger space and
        # enable the model to learn more complex patterns, GELU activation
        # function is applied to introduce non-linearity, second linear layer to
        # compress the embeddings back to the original `n_embd`, allowing the
        # model to make predictions or pass the transformed embeddings to the
        # next layer.
        self.layers = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )

    def forward(self, x):
        return self.layers(x)
