import torch


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


def encode_text(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # Add `batch` dimension.
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def decode_tokens(token_ids, tokenizer):
    # Remove `batch` dimension.
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
