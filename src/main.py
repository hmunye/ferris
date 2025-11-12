import torch
import tiktoken

from transformer import GPTModel
from util import encode_text, decode_tokens, generate
from config import cfg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                context_size=cfg.n_ctx,
                top_k=25,
                temp=1.0,
            )

            print("output:", decode_tokens(token_ids, tokenizer))
        except (EOFError, KeyboardInterrupt):
            break


if __name__ == "__main__":
    main()
