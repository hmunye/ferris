import torch


def encode_text(txt, tokenizer):
    encoded = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
    # Add `batch` dimension.
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def decode_tokens(token_ids, tokenizer):
    # Remove `batch` dimension.
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
