import time
import math
import torch
import tiktoken
import argparse

from dataset import prepare_instruct_data_loaders
from train import calc_loss_batch, evaluate_model, generate
from util import encode_text, decode_tokens
from config import ft_cfg as cfg
from transformer import GPTModel, replace_linear_with_lora


def ift(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Apple Silicon.
    if torch.backends.mps.is_available():
        device = torch.device("mps")

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

    print(
        f"\nUsing {cfg.n_workers + cfg.n_test_workers + cfg.n_val_workers} worker procs for data loaders\n"
    )

    print("Preparing data loaders...")

    tokenizer = tiktoken.get_encoding("gpt2")
    eos_id = tokenizer.eot_token

    source = ""
    if "lima" in args.source:
        source = "lima"

    if "concat" in args.source:
        source = "concat"

    train_loader, val_loader = prepare_instruct_data_loaders(
        cfg,
        tokenizer,
        args.source,
        eos_id,
        device,
        allowed_max_len=cfg.n_ctx,
        data_source=source,
    )

    print("Data loaders prepared\n")

    # Determine if further fine-tuning is being done.
    checkpoint_file = args.file

    model = GPTModel(cfg)
    model = torch.compile(model)

    # Freeze original model parameters.
    for param in model.parameters():
        param.requires_grad = False

    if checkpoint_file:
        print("Fine-tuning existing model...")
        checkpoint = torch.load(
            checkpoint_file,
            map_location=device,
        )
        # Using LoRA for fine-tuning, which results in less trainable parameters,
        # while not affecting the current model's weights.
        replace_linear_with_lora(model, cfg.rank, cfg.alpha)

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Fine-tuning foundation model...")
        checkpoint = torch.load(
            "checkpoints/foundation_model.pth",
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        # Using LoRA for fine-tuning, which results in less trainable parameters,
        # while not affecting the current model's weights.
        replace_linear_with_lora(model, cfg.rank, cfg.alpha)

    model.to(device).to(torch.bfloat16)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr_init,
        weight_decay=cfg.weight_decay,
    )

    global_step = -1
    total_tokens, last_tokens = 0, 0
    cumulative_tokens, cumulative_time = 0.0, 0.0

    # Total training iterations.
    n_steps = len(train_loader) * cfg.n_epoch
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

    # Clear initialized gradients so they can be accumulated.
    optimizer.zero_grad()

    try:
        for epoch in range(cfg.n_epoch):
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
                        cumulative_tokens / cumulative_time
                        if cumulative_time > 0
                        else 0
                    )

                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, cfg.eval_iter
                    )

                    print(
                        f"Ep {epoch+1}, Step {global_step:07d}, "
                        f"Train: {train_loss:.3f}, Val: {val_loss:.3f}, "
                        f"Step tok/sec: {round(tps)}, Avg tok/sec: {round(avg_tps)}"
                    )

                if global_step % cfg.sample_freq == 0:
                    instruction_text = (
                        "Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request."
                        "\n\n### Instruction:\nIdentify a programming language suitable for game development."
                    )
                    # Print sample output for observation.
                    print_sample(model, tokenizer, device, instruction_text, cfg)

        if args.output:
            output_path = args.output
        else:
            output_path = "fine_tuned/model_instruct.pth"

        # Save the final model and optimizer parameters.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            output_path,
        )

        print(f"Training complete: '{output_path}' saved")
    except (EOFError, KeyboardInterrupt):
        output_path = "fine_tuned/model_instruct_interrupt.pth"

        print(f"Keyboard interrupt: saving current model as: '{output_path}'")

        # Save the current model and optimizer parameters.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            output_path,
        )


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
            max_new_tokens=35,
            context_size=context_size,
            temp=cfg.temp,
            top_k=cfg.top_k,
        )

    decoded_text = decode_tokens(token_ids, tokenizer)
    response_text = decoded_text[len(start_ctx) :].strip()
    print(f"\n\x1b[1;34mSample response\x1b[0m: {response_text}\n")

    model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Model Instruction Fine-Tuning")

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="path to the source dataset file",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="path to the model checkpoint file for resuming fine-tuning",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="path to save the fine-tuned model",
    )

    args = parser.parse_args()

    # Instruction fine-tuning.
    ift(args)
