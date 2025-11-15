from dataclasses import dataclass


@dataclass
class GPTConfig:
    n_vocab: int = 50257  # size of BPE vocabulary
    n_ctx: int = 1024  # context/window size (max sequence length)
    n_embd: int = 768  # embedding dimension
    n_head: int = 12  # attention heads
    n_layer: int = 12  # transformer layers (blocks)
    drop_rate: float = 0.1  # dropout probability
    qkv_bias: bool = False  # include bias in QKV projections

    n_epoch: int = 10  # total epochs (iterations over train dataset)
    n_batch: int = 16  # batch size per iteration
    n_grad_acc: int = (
        4  # gradient accumulation steps (16 * 4 = 64 effective batch size)
    )

    eval_freq: int = 5000  # steps before model evaluation (train/val loss)
    eval_iter: int = 200  # batches to compute train/val loss over
    sample_freq: int = 10000  # steps before sample text generation

    lr_init: float = 3e-5  # initial learning rate
    lr_peak: float = 5e-4  # peak learning rate for warmup
    lr_min: float = 3e-5  # minimum learning rate for cosine decay
    weight_decay: float = 0.01  # L2 regularization

    temp: float = 1.4  # softmax temperature (higher = more random)
    top_k: int = 25  # top-k filtering during generation

    n_workers: int = 16  # data loader workers for training dataset
    n_val_workers: int = 8  # data loader workers for validation dataset


cfg = GPTConfig()
