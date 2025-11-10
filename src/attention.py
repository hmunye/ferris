import torch

class SelfAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.w_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, input):
        k = self.w_k(input)
        q = self.w_q(input)
        v = self.w_v(input)

        attn_scores = q @ k.T
        attn_weights = torch.nn.functional.softmax(
            attn_scores / k.shape[-1]**0.5, dim=-1
        )

        context = attn_weights @ v
        return context
