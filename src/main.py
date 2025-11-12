import torch
import tiktoken

from transformer import GPTModel
from util import encode_text, decode_tokens, generate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GPT-2 124M parameters
    cfg = {
        "vocab_size": 50257,  # Byte Pair Encoding (BPE) vocabulary size
        "context_len": 256,  # Maximum input tokens via positional embeddings
        "emb_dim": 768,  # Dimensionality of embedding space (token vectors)
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of transformer layers (blocks)
        "drop_rate": 0.1,  # Dropout rate applied to hidden units
        "qkv_bias": False,  # Whether to include bias term in QKV projections
        "num_epochs": 10,
    }

    checkpoint = torch.load("checkpoints/model.pth", map_location=device)

    model = GPTModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    while True:
        try:
            prompt = input("prompt: ")

            token_ids = generate(
                model=model,
                idx=encode_text(prompt, tokenizer).to(device),
                max_new_tokens=30,
                context_size=cfg["context_len"],
                top_k=25,
                temp=1.0,
            )

            print("output:", decode_tokens(token_ids, tokenizer))
        except (EOFError, KeyboardInterrupt):
            break


if __name__ == "__main__":
    main()
