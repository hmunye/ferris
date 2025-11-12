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
    n_epoch: int = 15
    n_batch: int = 32
    eval_freq: int = 5
    eval_iter: int = 5
    lr: float = 5e-4
    weight_decay: float = 0.1


cfg = GPTConfig()
