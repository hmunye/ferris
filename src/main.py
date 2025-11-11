import torch
import tiktoken

from transformer import GPTModel
from util import encode_text, decode_tokens
from data import create_dataloader


def main():
    torch.manual_seed(123)

    # GPT-2 124M parameters
    cfg = {
        "vocab_size": 50257,  # Byte Pair Encoding (BPE) vocabulary size
        "context_len": 256,  # Maximum input tokens via positional embeddings
        "emb_dim": 768,  # Dimensionality of embedding space (token vectors)
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of transformer layers (blocks)
        "drop_rate": 0.1,  # Dropout rate applied to hidden units
        "qkv_bias": False,  # Whether to include bias term in QKV projections
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    file_path = "datasets/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader(
        train_data,
        batch_size=2,
        max_length=cfg["context_len"],
        stride=cfg["context_len"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader(
        val_data,
        batch_size=2,
        max_length=cfg["context_len"],
        stride=cfg["context_len"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    model = GPTModel(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print("training loss:", train_loss)
    print("validation loss:", val_loss)


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    loader_len = len(data_loader)

    if loader_len == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = loader_len
    else:
        num_batches = min(num_batches, loader_len)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss
        else:
            break

    # Average the loss over all batches.
    return total_loss / num_batches


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )

    return loss


def generate_text(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        # Disable gradient tracking.
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        probas = torch.softmax(logits, dim=-1)
        next_idx = torch.argmax(probas, dim=-1, keepdim=True)

        idx = torch.cat((idx, next_idx), dim=-1)

    return idx


if __name__ == "__main__":
    main()
