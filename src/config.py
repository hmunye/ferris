from dataclasses import dataclass


@dataclass
class GPTConfig:
    n_vocab: int = 50257
    n_ctx: int = 256
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False
    n_epoch: int = 10


cfg = GPTConfig()
