from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_len: int = 256
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False
    num_epochs: int = 10


cfg = GPTConfig()
