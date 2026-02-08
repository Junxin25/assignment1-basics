import torch
import torch.nn as nn

class RMSNorm(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty((d_model,), device=device, dtype=dtype))
        nn.init.ones_(self.weight)


    # x = shape=(batch, seq_len, hidden_dim), dtype=torch.float32
    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS_norm = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        result = x / RMS_norm * self.weight
        return result.to(in_dtype)

if __name__ == "__main__":
    # Example usage
    d_model = 512
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, d_model)
    rmsnorm = RMSNorm(d_model)
    output = rmsnorm(x)
    print(output.shape)  # Should be (batch_size, seq_len, d_model)