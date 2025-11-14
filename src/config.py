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
    n_batch: int = 16
    n_grad_acc: int = 4  # effective batch size = 64
    eval_freq: int = 2000
    eval_iter: int = 200
    lr_init: float = 3e-5
    lr_peak: float = 5e-4
    lr_min: float = 3e-5
    weight_decay: float = 0.01

    top_k: int = 25
    temp: float = 1.0

    n_workers: int = 16
    n_val_workers: int = 4


cfg = GPTConfig()
