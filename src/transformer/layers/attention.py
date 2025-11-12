# TODO: Maybe add KV cache for more efficient inference. Look into GQA/MLA/SWA
# over regular Multi-Head Attention (MHA).

import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, num_heads, qkv_bias=False):
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        super().__init__()

        self.d_out = d_out
        self.context_len = context_len
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Initialize weight matrices for query, key, and value projections.
        # self.w_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        # self.w_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        # self.w_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)

        # Linear projection to combine results from all attention heads.
        self.proj = nn.Linear(d_out, d_out)

        # Dropout layer to help reduce overfitting during training.
        # self.dropout = nn.Dropout(dropout)
        self.dropout = dropout

        # Causal mask to ensure tokens only attend to the current and previous
        # tokens.
        #
        # self.register_buffer(
        #     "mask",
        #     torch.triu(
        #         torch.ones(context_len, context_len), diagonal=1
        #     ),  # upper triangular matrix
        # )

    def forward(self, x):
        # Batch size, number of tokens, and the embedding dimension.
        b, num_tokens, d_in = x.shape

        # Project the input sequence into queries, keys, and values using the
        # respective weights.
        #
        # k = self.w_k(x)  # shape: (b, num_tokens, d_out)
        # q = self.w_q(x)  # shape: (b, num_tokens, d_out)
        # v = self.w_v(x)  # shape: (b, num_tokens, d_out)
        # (b, num_tokens, d_in) --> (b, num_tokens, 3 * d_in)
        qkv = self.qkv(x)

        # Reshape q, k, v tensors to separate the attention heads.
        #
        # k = torch.transpose(
        #     k.view(batch, num_tokens, self.num_heads, self.head_dim), 1, 2
        # )  # shape: (b, num_heads, num_tokens, head_dim)
        # q = torch.transpose(
        #     q.view(batch, num_tokens, self.num_heads, self.head_dim), 1, 2
        # )  # shape: (b, num_heads, num_tokens, head_dim)
        # v = torch.transpose(
        #     v.view(batch, num_tokens, self.num_heads, self.head_dim), 1, 2
        # )  # shape: (b, num_heads, num_tokens, head_dim)
        #
        # (b, num_tokens, 3 * d_in) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(b, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv

        # Compute attention scores (dot-product) for each attention head.
        # attn_scores = q @ k.transpose(2, 3)

        # Mask is truncated to the number of tokens.
        # mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Apply mask to attention scores, setting the future tokens to `-inf`,
        # which makes the softmax of those positions effectively zero. Performed
        # in-place.
        # attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Normalize the attention scores using softmax to get attention weights.
        # attn_weights = nn.functional.softmax(attn_scores / k.shape[-1] ** 0.5, dim=-1)
        # attn_weights = self.dropout(attn_weights)

        # Compute the context vectors by multiplying the attention weights with
        # the value matrix.
        # context = (attn_weights @ v).transpose(
        #     1, 2
        # )  # shape: (b, num_tokens, num_heads, head_dim)

        # Flatten the heads back together and get the final output.
        # context = context.contiguous().view(b, num_tokens, self.d_out)

        # PyTorch's `scaled_dot_product_attention`, which implements a
        # memory-optimized version of self-attention called `FlashAttention`.
        context_vec = nn.functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            dropout_p=self.dropout,
            is_causal=True,
        )

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = (
            context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        )

        # Linearly project the concatenated context vectors back to the original
        # output dimension.
        return self.proj(context_vec)
