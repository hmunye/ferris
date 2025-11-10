import torch

class SelfAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out

        self.w_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, input):
        b, num_tokens, d_in = input.shape

        k = self.w_k(input)
        q = self.w_q(input)
        v = self.w_v(input)

        attn_scores = q @ k.transpose(1, 2)
        # Performed in-place.
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], 
            -torch.inf
        )

        attn_weights = self.dropout(torch.nn.functional.softmax(
            attn_scores / k.shape[-1]**0.5, dim=-1
        ))

        context = attn_weights @ v
        return context
