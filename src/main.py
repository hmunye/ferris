import torch
import tiktoken
import time
import argparse

from transformer import GPTModel, replace_linear_with_lora
from util import encode_text, decode_tokens
from config import pt_cfg as cfg


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Apple Silicon.
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Using device: '{device}'")

    checkpoint = torch.load(
        args.file,
        map_location=device,
    )

    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()

        # Volta (7.0+), Turing (7.5+), Ampere (8.0+), Hopper (9.0+).
        if capability[0] >= 7:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"

    model = GPTModel(cfg)
    model = torch.compile(model)

    replace_linear_with_lora(model, cfg.rank, cfg.alpha)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).to(torch.bfloat16)

    tokenizer = tiktoken.get_encoding("gpt2")

    eos_id = tokenizer.eot_token
    max_new_tokens = 1000

    model.eval()

    while True:
        try:
            print("❯ ", end="", flush=True)
            prompt = input()
            instruction_text = (
                f"Below is an instruction that describes a task."
                f"Write a response that appropriately completes the request."
                f"\n\n### Instruction:\n{prompt}"
            )

            encoded = encode_text(instruction_text, tokenizer).to(device)

            # Disable gradient tracking.
            with torch.inference_mode():
                for _ in range(max_new_tokens):
                    token_id = generate_next_token(
                        model=model,
                        idx=encoded,
                        context_size=cfg.n_ctx,
                        top_k=cfg.top_k,
                        temp=cfg.temp,
                        eos_id=eos_id,
                    )

                    if token_id is None:
                        break
                    else:
                        decoded_text = decode_tokens(token_id, tokenizer)
                        print(decoded_text, sep="", end="", flush=True)
                        # Append the predicted token to the input.
                        encoded = torch.cat((encoded, token_id), dim=-1)

                        time.sleep(0.01 * len(decoded_text))

            print()
        except (EOFError, KeyboardInterrupt):
            break


def generate_next_token(model, idx, context_size, temp=0.0, top_k=None, eos_id=None):
    # Truncate context.
    idx_cond = idx[:, -context_size:]

    logits = model(idx_cond)

    # Last token logits contain the prediction.
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

    if next_idx.item() == eos_id:
        return None

    return next_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Model Inference")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="path to the model file to run inference",
    )

    args = parser.parse_args()
    main(args)
