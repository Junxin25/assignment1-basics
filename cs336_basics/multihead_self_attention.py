import torch
import torch.nn as nn
from einops import rearrange, einsum

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT) 

from RoPE import RotaryPositionalEmbedding as RoPE

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        factory_kwargs = {'device': torch.device('cpu')}
        self.W_q, self.W_k, self.W_v, self.W_o = (nn.Parameter(torch.empty(d_model, d_model, **factory_kwargs)) for _ in range(4))

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))  # (1, 1, max_seq_len, max_seq_len)

        if use_rope:
            self.rope = RoPE(theta=10000.0, d_k=self.d_k, max_seq_len=2048, device=None)
        
    def forward(self,
                 x: Float[Tensor, "batch seq_len d_model"], 
                 token_position: Int[Tensor, "batch seq_len"] = None) -> Float[Tensor, "batch seq_len d_model"]:
        # batch_size, seq_len, d_model = x.shape
        # assert d_model == self.d_model
        # assert seq_len <= 2048

        # Q、K、V 投影
        q = einsum(x, self.W_q, 'b s d, d e -> b s e')
        k = einsum(x, self.W_k, 'b s d, d e -> b s e')
        v = einsum(x, self.W_v, 'b s d, d e -> b s e')

        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)  # (batch_size, num_heads, seq_len, d_k)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)

        if self.use_rope:
            if token_position is not None:
                assert token_position.shape == x.shape[:-1]
                q = self.rope(q, token_position)
                k = self.rope(k, token_position)
        
        attn_scores = einsum(q, k, 'b h i d, b h j d -> b h i j') / (self.d_k ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = einsum(attn_weights, v, 'b h i j, b h j d -> b h i d')

        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        output = einsum(attn_output, self.W_o, 'b s d, d e -> b s e')    #(batch_size, seq_len, d_model)
        return output