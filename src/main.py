import torch
import tiktoken
import time

from transformer import GPTModel
from util import encode_text
from config import cfg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("checkpoints/model.pth", map_location=device)

    model = GPTModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    max_new_tokens = 30

    while True:
        try:
            prompt = input("prompt: ")

            idx = encode_text(prompt, tokenizer).to(device)

            print(f"output: {prompt}", end="", flush=True)

            # Disable gradient tracking.
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    token_id = generate(
                        model=model,
                        idx=idx,
                        context_size=cfg.n_ctx,
                        top_k=25,
                        temp=1.0,
                    )

                    if token_id is None:
                        break
                    else:
                        decoded_text = tokenizer.decode([token_id])

                        print(decoded_text, sep="", end="", flush=True)

                        # Append the predicted token to the input.
                        idx = torch.cat((idx, token_id), dim=-1)

                        time.sleep(0.01 * len(decoded_text))

            print()
        except (EOFError, KeyboardInterrupt):
            break


def generate(model, idx, context_size, temp=0.0, top_k=None, eos_id=None):
    idx_cond = idx[:, -context_size:]

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
        next_idx = torch.argmax(logits, dim=-1, keepdim=True)

    if next_idx == eos_id:
        return None

    return next_idx


if __name__ == "__main__":
    main()
