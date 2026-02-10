import  torch
import torch.nn as nn
import math
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT) 

from softmax import softmax
from typing import Optional



# Q.shape = (batch_size, seq_len_q, d_k)
# K.shape = (batch_size, seq_len_k, d_k)
# V.shape = (batch_size, seq_len_v, d_v)
def scale_dot_product_attention(Q: torch.Tensor,
                                K: torch.Tensor,
                                V: torch.Tensor,
                                mask: Optional[torch.Tensor] = None,
                                ) -> torch.Tensor:
    d_k = Q.shape[-1]
    scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attention = softmax(dim=-1)(scores)
    output = torch.matmul(attention, V)
    return output

if __name__ == "__main__":
    batch_size = 2
    seq_len_q = 3
    seq_len_kv = 4
    d_k = 5
    d_v = 6

    Q = torch.randn(batch_size, seq_len_q, d_k)
    K = torch.randn(batch_size, seq_len_kv, d_k)
    V = torch.randn(batch_size, seq_len_kv, d_v)

    output = scale_dot_product_attention(Q, K, V)
    print("Output:")
    print(output.shape)  # Should be (batch_size, seq_len_q, d_v)
