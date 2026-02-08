import torch
import torch.nn as nn
from torch.nn import init

class linear(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ):
        super().__init__()
        self.W = nn.Parameter(torch.empty((in_features, out_features), device=device, dtype=dtype))
        init.trunc_normal_(self.W, mean=0.0, std=0.02)

    def forward(self, x):
        return x @ self.W
    
if __name__ == "__main__":
    # test
    lin = linear(4, 5)
    x = torch.randn((2, 4))
    print(lin(x).shape)