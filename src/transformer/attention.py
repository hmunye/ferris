import torch


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        super().__init__()

        self.d_out = d_out
        self.num_heads = num_heads

        self.head_dim = d_out // num_heads

        # Initialize weight matrices for query, key, and value projections.
        self.w_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

        # Linear projection to combine results from all attention heads.
        self.out_proj = torch.nn.Linear(d_out, d_out)

        # Dropout layer to help reduce overfitting during training.
        self.dropout = torch.nn.Dropout(dropout)

        # Causal mask to ensure tokens only attend to the current and previous
        # tokens.
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length), diagonal=1
            ),  # upper triangular matrix
        )

    def forward(self, input):
        # Batch size, number of tokens, and the input dimension.
        b, num_tokens, d_in = input.shape

        # Project the input sequence into queries, keys, and values using the
        # respective weights.
        k = self.w_k(input)  # shape: (b, num_tokens, d_out)
        q = self.w_q(input)  # shape: (b, num_tokens, d_out)
        v = self.w_v(input)  # shape: (b, num_tokens, d_out)

        # Reshape `k`, `q`, `v` tensors to separate the attention heads.
        k = torch.transpose(
            k.view(b, num_tokens, self.num_heads, self.head_dim), 1, 2
        )  # shape: (b, num_heads, num_tokens, head_dim)
        q = torch.transpose(
            q.view(b, num_tokens, self.num_heads, self.head_dim), 1, 2
        )  # shape: (b, num_heads, num_tokens, head_dim)
        v = torch.transpose(
            v.view(b, num_tokens, self.num_heads, self.head_dim), 1, 2
        )  # shape: (b, num_heads, num_tokens, head_dim)

        # Compute attention scores (dot-product) for each attention head.
        attn_scores = q @ k.transpose(2, 3)

        # Mask is truncated to the number of tokens.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Apply mask to attention scores, setting the future tokens to `-inf`,
        # which makes the softmax of those positions effectively zero. Performed
        # in-place.
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Normalize the attention scores using softmax to get attention weights.
        attn_weights = torch.nn.functional.softmax(
            attn_scores / k.shape[-1] ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        # Compute the context vectors by multiplying the attention weights with
        # the value matrix.
        context = (attn_weights @ v).transpose(
            1, 2
        )  # shape: (b, num_tokens, num_heads, head_dim)

        # Flatten the heads back together and get the final output.
        context = context.contiguous().view(b, num_tokens, self.d_out)

        # Linearly project the concatenated context vectors back to the original
        # output dimension.
        return self.out_proj(context)
