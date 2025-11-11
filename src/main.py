def main():
    import torch
    import tiktoken
    from transformer import GPTModel

    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Byte Pair Encoding (BPE) vocabulary size
        "context_len": 1024,  # Maximum input tokens via positional embeddings
        "emb_dim": 768,  # Dimensionality of embedding space (token vectors)
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of transformer layers (blocks)
        "drop_rate": 0.1,  # Dropout rate applied to hidden units
        "qkv_bias": False,  # Whether to include bias term in QKV projections
    }

    model = GPTModel(GPT_CONFIG_124M)
    logits = model(batch)

    print("shape:", logits.shape)
    print(logits)


if __name__ == "__main__":
    main()
