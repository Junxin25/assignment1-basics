import torch
import torch.nn as nn

class embedding(nn.Module):
    def __init__(self, num_embeddings: int,
                  embedding_dim: int,
                  device: torch.device | None = None,
                  dtype: torch.dtype | None = None):
        super(embedding, self).__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02)


    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]