import torch
import tiktoken
import time
import math
import sys
import re
import argparse

from dataset import prepare_data_loaders
from transformer import GPTModel
from util import encode_text, decode_tokens
from config import cfg


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: '{device}'")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

        capability = torch.cuda.get_device_capability()

        # Volta (7.0+), Turing (7.5+), Ampere (8.0+), Hopper (9.0+).
        if capability[0] >= 7:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"
            print("Using tensor cores")
        else:
            print("Tensor cores not supported, using default precision")

    tokenizer = tiktoken.get_encoding("gpt2")

    print(f"Using {cfg.n_workers} worker procs for training loader")
    print(f"Using {cfg.n_val_workers} worker procs for validation loader")

    print("Preparing data loaders...")
    train_loader, val_loader = prepare_data_loaders(cfg=cfg, tokenizer=tokenizer)
    print("Data loaders prepared")

    # Determine if training is being resumed.
    checkpoint_file = args.file

    if checkpoint_file:
        pattern = r"checkpoints/model_epoch_(\d+)_step_(\d+)_val_loss_([\d.]+)\.pth"
        match = re.match(pattern, checkpoint_file)

        if match:
            resume_epoch = int(match.group(1)) - 1
            global_step = int(match.group(2))
            # Keep track of best validation loss when saving model checkpoints.
            best_val_loss = float(match.group(3))

            checkpoint = torch.load(checkpoint_file, map_location=device)

            model = GPTModel(cfg)
            model = torch.compile(model)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device).to(torch.bfloat16)

            optimizer = torch.optim.AdamW(
                model.parameters(), weight_decay=cfg.weight_decay
            )
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            print(f"ERROR: Failed to parse checkpoint file '{checkpoint_file}'", file=sys.stderr)
            quit(1)
    else:
        resume_epoch = 0
        global_step = -1
        # Keep track of best validation loss when saving model checkpoints.
        best_val_loss = float("inf")

        model = GPTModel(cfg)
        model = torch.compile(model)
        model.to(device).to(torch.bfloat16)

        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=cfg.weight_decay)

    total_tokens, last_tokens = 0, 0
    cumulative_tokens, cumulative_time = 0.0, 0.0

    # Total training iterations.
    n_steps = len(train_loader) * cfg.n_epoch - resume_epoch
    n_warmup = int(0.1 * n_steps)

    # Learning rate linear warmup stabilizes training by gradually increasing
    # the learning rate from an initial value to a peak during "warmup" phase.
    lr_increment = (cfg.lr_peak - cfg.lr_init) / n_warmup

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

    # Not zeroing gradients if resuming.
    if not checkpoint_file:
        print("Starting training...\n")
        # Clear initialized gradients so they can be accumulated.
        optimizer.zero_grad()
    else:
        print(
            f"Resuming training with epoch: {resume_epoch+1}, step: {global_step:06d}, best_val_loss: {best_val_loss:.3f}\n"
        )

    for epoch in range(resume_epoch, cfg.n_epoch):
        model.train()

        for input_batch, target_batch in train_loader:
            global_step += 1

            if global_step < n_warmup:
                # Update learning rate during warmup phase.
                lr = cfg.lr_init + global_step * lr_increment
            else:
                # Cosine annealing after warmup phase to modulate the learning
                # rate throughout training.
                progress = (global_step - n_warmup) / (n_steps - n_warmup)
                lr = cfg.lr_min + (cfg.lr_peak - cfg.lr_min) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward and backward pass.
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # Calculate loss gradients.
            loss.backward()

            # Apply gradient clipping each iteration to avoid exploding
            # gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update model weights using loss gradients with gradient
            # accumulation.
            if global_step % cfg.n_grad_acc == 0:
                optimizer.step()
                optimizer.zero_grad()

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

                print(
                    f"Ep {epoch+1}, Step {global_step:06d}, "
                    f"Train: {train_loss:.3f}, Val: {val_loss:.3f}, "
                    f"Step tok/sec: {round(tps)}, Avg tok/sec: {round(avg_tps)}"
                )

                if val_loss < best_val_loss:
                    # Save the model and optimizer parameters.
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        f"checkpoints/model_epoch_{epoch+1}_step_{global_step:06d}_val_loss_{val_loss:.3f}.pth",
                    )

                    print(
                        f"Saved checkpoint: model_epoch_{epoch+1}_step_{global_step:06d}_val_loss_{val_loss:.3f}"
                    )
                    best_val_loss = val_loss

            if global_step % cfg.sample_freq == 0:
                # Print sample output for observation.
                print_sample(model, tokenizer, device, "But I think I don't like", cfg)

    # Save the final model and optimizer parameters.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "checkpoints/final_model_owt.pth",
    )

    print("Training complete: 'final_model_owt.pth' saved")


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


def print_sample(model, tokenizer, device, start_ctx, cfg):
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
            temp=cfg.temp,
            top_k=cfg.top_k,
        )

    # Print in compacted format, replacing newlines.
    decoded_text = decode_tokens(token_ids, tokenizer).replace("\n", "\\n")
    print(f"\nSample generation: {decoded_text}\n")

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
    parser = argparse.ArgumentParser(description="GPT Model Training")
    parser.add_argument(
        "--file",
        type=str,
        help="resumes training given a model checkpoint file",
    )

    args = parser.parse_args()

    train_model(args)
