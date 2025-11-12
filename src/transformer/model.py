import torch
import torch.nn as nn

from .layers import TransformerBlock, LayerNorm


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)

        # Encodes the position of each token in the sequence (since transformers
        # do not inherently process sequences in-order) into a dense vector
        # embedding.
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.emb_dim)

        self.drop_emb = nn.Dropout(cfg.drop_rate)

        # Sequential stack of transformer blocks (decoder layers).
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # Convert token indices to dense token and position embeddings.
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)

        # Apply normalization to the final output of the transformer layers.
        # Stabilizes the learning process.
        x = self.final_norm(x)

        # Project the final embeddings back to the vocabulary size. This will
        # produce logits for each token in the vocabulary (to predict the next
        # token).
        logits = self.out_head(x)

        return logits
