import torch

def main():
    import tiktoken
    from transformer import GPTModel

    tokenizer = tiktoken.get_encoding("gpt2")

    # GPT-2 124M parameters
    cfg = {
        "vocab_size": 50257,  # Byte Pair Encoding (BPE) vocabulary size
        "context_len": 1024,  # Maximum input tokens via positional embeddings
        "emb_dim": 768,  # Dimensionality of embedding space (token vectors)
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of transformer layers (blocks)
        "drop_rate": 0.1,  # Dropout rate applied to hidden units
        "qkv_bias": False,  # Whether to include bias term in QKV projections
    }

    # torch.manual_seed(123)

    model = GPTModel(cfg)

    start_ctx = "Hello, I am"
    encoded = tokenizer.encode(start_ctx)

    print("input:", start_ctx)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("tensor shape:", encoded_tensor.shape)

    model.eval()
    out = generate_text(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=cfg["context_len"],
    )

    print("output:", out)
    print("output len:", len(out[0]))

    print("decoded:", tokenizer.decode(out.squeeze(0).tolist()))


def generate_text(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, next_idx), dim=-1)

    return idx


if __name__ == "__main__":
    main()
