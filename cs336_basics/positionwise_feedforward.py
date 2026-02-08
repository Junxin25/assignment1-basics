import torch
import torch.nn as nn
import math

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model

        self.d_ff = self.choose_d_ff(d_model)
        self.weight1 = nn.Parameter(torch.empty((d_model, self.d_ff), device=device, dtype=dtype))
        self.weight2 = nn.Parameter(torch.empty((self.d_ff, d_model), device=device, dtype=dtype))
        self.weight3 = nn.Parameter(torch.empty((d_model, self.d_ff), device=device, dtype=dtype))
        
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)
        nn.init.xavier_uniform_(self.weight3)

    def choose_d_ff(self, d_model, multiple=64):
        raw = (8.0 / 3.0) * d_model
        return int(math.ceil(raw / multiple) * multiple)

    def forward(self, x):
        a = x @ self.weight1
        b = x @ self.weight3
        g = (a * torch.sigmoid(a)) * b
        y = g @ self.weight2
        return y
    
if __name__ == "__main__":
    d_model = 512
    d_ff = 2048
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, d_model)
    device = torch.device("cuda:0")  # 或 "cpu"
    x = x.to(device)
    ff = PositionwiseFeedForward(d_model, device=device).to(device)
    output = ff(x)
    print(output.shape)  # Should be (batch_size, seq_len, d_model)