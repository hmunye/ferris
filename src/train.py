import torch
import tiktoken
import time
import os
import requests

from dataset import create_dataloader
from transformer import GPTModel
from util import encode_text, decode_tokens
from config import cfg


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"PyTorch version: {torch.__version__}")
    print(f"using {device}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

        capability = torch.cuda.get_device_capability()

        # Volta (7.0+), Turing (7.5+), Ampere (8.0+), Hopper (9.0+).
        if capability[0] >= 7:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"
            print("using tensor cores")
        else:
            print("tensor cores not supported, using default precision")

    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader, val_loader = prepare_data_loaders(cfg=cfg, tokenizer=tokenizer)

    model = GPTModel(cfg)
    model = torch.compile(model)
    model.to(device).to(torch.bfloat16)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    start_ctx = "Every effort moves you"

    train_losses, val_losses, track_tokens = [], [], []
    total_tokens, global_step, last_tokens = 0, -1, 0

    # Variables for cumulative average tokens/sec.
    cumulative_tokens, cumulative_time = 0.0, 0.0

    use_cuda = device.type == "cuda"

    if use_cuda:
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        # Ensure all prior CUDA operations are done.
        torch.cuda.synchronize()
        # Start the timer for the first interval.
        t_start.record()
    else:
        # Start the timer for the first interval.
        t0 = time.time()

    for epoch in range(cfg.n_epoch):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # Forward and backward pass.
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # Calculate loss gradients.
            loss.backward()
            # Update model weights using loss gradients.
            optimizer.step()

            total_tokens += input_batch.numel()

            if global_step % cfg.eval_freq == 0:
                # End timing for the current interval.
                if use_cuda:
                    t_end.record()
                    # Wait for all CUDA ops to complete.
                    torch.cuda.synchronize()
                    ms_elapsed = t_start.elapsed_time(t_end) / 1000
                    # Reset timer for the next interval.
                    t_start.record()
                else:
                    ms_elapsed = time.time() - t0
                    # Reset timer for the next interval.
                    t0 = time.time()

                # Calculate tokens processed in this interval.
                tokens_interval = total_tokens - last_tokens
                last_tokens = total_tokens
                tps = tokens_interval / ms_elapsed if ms_elapsed > 0 else 0

                # Update cumulative counters (skip first interval).
                if global_step:  # False when global_step == 0
                    cumulative_tokens += tokens_interval
                    cumulative_time += ms_elapsed

                # Compute cumulative average tokens/sec (skip first interval).
                avg_tps = (
                    cumulative_tokens / cumulative_time if cumulative_time > 0 else 0
                )

                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, cfg.eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens.append(total_tokens)

                print(
                    f"ep {epoch+1}, step {global_step:06d}, "
                    f"train: {train_loss:.3f}, val: {val_loss:.3f}, "
                    f"step tok/sec: {round(tps)}, avg tok/sec: {round(avg_tps)}"
                )

        # Print sample output after each epoch.
        print_sample(model, tokenizer, device, start_ctx)

        # Print memory stats after each epoch.
        if torch.cuda.is_available():
            device = torch.cuda.current_device()

            allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
            reserved_gb = torch.cuda.memory_reserved(device) / 1024**3

            print(f"\nallocated memory: {allocated_gb:.4f} GB")
            print(f"reserved memory: {reserved_gb:.4f} GB\n")

    # Save the model and optimizer parameters.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "checkpoints/model.pth",
    )


def prepare_data_loaders(cfg, tokenizer):
    print("preparing data loaders...")

    file_path = "data/middlemarch.txt"
    url = "https://www.gutenberg.org/cache/epub/145/pg145.txt"

    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader(
        text_data[:split_idx],
        tokenizer,
        batch_size=2,
        max_len=cfg.n_ctx,
        stride=cfg.n_ctx,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader(
        text_data[split_idx:],
        tokenizer,
        batch_size=2,
        max_len=cfg.n_ctx,
        stride=cfg.n_ctx,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    print("data prepared\n")

    return train_loader, val_loader


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)

    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )

    return loss


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
            total_loss += loss.item()
        else:
            break

    # Average the loss over all batches.
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # Disables dropout.
    model.eval()

    # Disables gradient tracking.
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    model.train()
    return train_loss, val_loss


def print_sample(model, tokenizer, device, start_ctx):
    # Disables dropout.
    model.eval()

    context_size = model.pos_emb.weight.shape[0]
    encoded = encode_text(start_ctx, tokenizer).to(device)

    # Disables gradient tracking.
    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            temp=1.4,
            top_k=25,
        )

    # Compact the print format, replacing newlines.
    decoded_text = decode_tokens(token_ids, tokenizer).replace("\n", " ")
    print(f"{decoded_text}\n")

    model.train()


def generate(
    model, idx, max_new_tokens, context_size, temp=0.0, top_k=None, eos_id=None
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        # Disable gradient tracking.
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if top_k is not None:
            # Apply top-k sampling to restrict sampled tokens to the top-k most
            # likely.
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]

            # Mask other tokens with `-inf`.
            logits = torch.where(
                condition=logits < min_val,
                input=torch.tensor(float("-inf")).to(logits.device),
                other=logits,
            )

        if temp > 0.0:
            # Apply temperature scaling.
            logits /= temp
            probs = torch.softmax(logits, dim=-1)

            # Probabilistic sampling.
            next_idx = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding.
            next_idx = torch.argmax(probs, dim=-1, keepdim=True)

        if next_idx == eos_id:
            break

        # Append the predicted token to the input.
        idx = torch.cat((idx, next_idx), dim=-1)

    return idx


if __name__ == "__main__":
    train()
