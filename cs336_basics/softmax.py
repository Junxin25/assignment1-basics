import torch
import torch.nn as nn

class softmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # raise NotImplementedError
        x = x - torch.max(x, dim=self.dim, keepdim=True).values
        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x, dim=self.dim, keepdim=True)
    
if __name__ == "__main__":
    batch_size = 2
    seq_len = 3
    d_k = 4
    x = torch.randn(batch_size, seq_len, d_k)
    print("Input:")
    print(x)

    softmax_layer = softmax(dim=-1)
    output = softmax_layer(x)
    print("Output:")
    print(output)