import torch.nn as nn

from .attention import MultiHeadAttention
from .norm import LayerNorm
from .feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_len=cfg.context_len,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias,
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        # Using residual (shortcut) connections to improve gradient flow and
        # avoid vanishing gradients.
        #
        # First sub-layer (attention layer).
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        # Add back the original input (residual connection).
        x = x + shortcut

        # Second sub-layer (feed-forward network).
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        # Add back the original input (residual connection).
        x = x + shortcut

        return x
