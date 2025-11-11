import torch


def encode_text(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # Add `batch` dimension.
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def decode_tokens(token_ids, tokenizer):
    # Remove `batch` dimension.
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
