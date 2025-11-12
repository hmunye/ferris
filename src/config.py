from dataclasses import dataclass


@dataclass
class GPTConfig:
    n_vocab: int = 50257
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False

    n_epoch: int = 10
    n_batch: int = 12
    eval_freq: int = 5
    eval_iter: int = 5
    lr_init: float = 3e-05
    lr_min: float = 1e-6
    lr_peak: float = 5e-4
    weight_decay: float = 0.1

    top_k: int = 25
    temp: float = 1.0


cfg = GPTConfig()
