import torch
import tiktoken

from dataset import create_dataloader
from transformer import GPTModel
from util import encode_text, decode_tokens
from config import cfg


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(cfg)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    tokenizer = tiktoken.get_encoding("gpt2")

    eval_freq = 5
    eval_iter = 5
    start_ctx = "Every effort moves you"

    train_loader, val_loader = prepare_data_loaders(cfg=cfg, tokenizer=tokenizer)

    train_losses, val_losses, seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(cfg.n_epoch):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # Calculate loss gradients.
            loss.backward()

            # Update model weights using loss gradients.
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                seen.append(tokens_seen)

                print(
                    f"ep {epoch+1} (step {global_step:06d}): "
                    f"train loss {train_loss:.3f}"
                    f"val loss {val_loss:.3f}"
                )

        # Print sample output after each epoch.
        print_sample(model, tokenizer, device, start_ctx)

    # Save the model and optimizer parameters.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "checkpoints/model.pth",
    )


def prepare_data_loaders(cfg, tokenizer):
    file_path = "data/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader(
        train_data,
        tokenizer,
        batch_size=2,
        max_len=cfg.n_ctx,
        stride=cfg.n_ctx,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader(
        val_data,
        tokenizer,
        batch_size=2,
        max_len=cfg.n_ctx,
        stride=cfg.n_ctx,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

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
            total_loss += loss
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
            max_new_tokens=15,
            context_size=context_size,
            temp=1.4,
            top_k=25,
        )

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
