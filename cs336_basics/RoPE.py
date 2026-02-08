import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
        assert d_k % 2 == 0
        d_half = d_k // 2

        k = torch.arange(d_half, device=device, dtype=torch.float32)    #(d_half,)
        inv_freq = theta ** (-2 * k / d_k)    #(d_half,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)    #(max_seq_len,)
        
        ## Compute cos and sin for all positions and dimensions in advance
        angles = pos[:, None] * inv_freq[None, :]   #(max_seq_len, d_half)
        cos = torch.cos(angles)    #(max_seq_len, d_half)
        sin = torch.sin(angles)    #(max_seq_len, d_half)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
    
    def forward(self, x: torch.Tensor, token_position: torch.Tensor):
        # x: (batch_size, seq_len, d_k)
        # token_position: (batch_size, seq_len)
        batch_size, seq_len, d_k = x.shape
        assert d_k == self.d_k
        assert seq_len <= self.max_seq_len

        cos = self.cos_cached[token_position]   #(batch_size, seq_len, d_half)
        sin = self.sin_cached[token_position]   #(batch_size, seq_len, d_half

        x_in_dtype = x.dtype
        x = x.to(torch.float32)    #(batch_size, seq_len, d_k)

        x_pair = x.view(batch_size, seq_len, d_k // 2, 2)   #(batch_size, seq_len, d_half, 2)
        x_even = x_pair[..., 0]    #(batch_size, seq_len, d_half)
        x_odd = x_pair[..., 1]     #(batch_size, seq_len, d_half)

        y_even = x_even * cos - x_odd * sin   #(batch_size, seq_len, d_half)
        y_odd = x_even * sin + x_odd * cos    #(batch_size, seq_len, d_half)

        y = torch.stack((y_even, y_odd), dim=-1).view(batch_size, seq_len, d_k)    #(batch_size, seq_len, d_k)
        y = y.to(x_in_dtype)
        return y
    
if __name__ == "__main__":
    theta = 10000.0
    d_k = 512
    max_seq_len = 2048
    batch_size = 2
    seq_len = 10

    device = torch.device("cuda:0")  # 或 "cpu"
    x = torch.randn(batch_size, seq_len, d_k, device=device)
    token_position = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)   #(batch_size, seq_len)

    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=device).to(device)
    output = rope(x, token_position)
    print(output.shape)  # Should be (batch_size, seq_len, d_k)