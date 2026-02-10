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
        *batch_dims, seq_len, d_k = x.shape
        d_half = d_k // 2

        if token_position is None:
            base = torch.arange(seq_len, device=x.device, dtype=torch.long)   #(seq_len,)
            token_position = base.view(*([1] * (x.ndim - 2)), seq_len).expand(*x.shape[:-1])    #(1, seq_len)
        else: 
            # assert token_position.shape == (*batch_dims, seq_len)
            token_position = token_position.to(device=x.device, dtype=torch.long)
            while token_position.ndim < x.ndim - 1:
                token_position = token_position.unsqueeze(1)   #(batch_size, seq_len, 1)
            token_position = token_position.expand(*x.shape[:-1])    #(batch_size, seq_len)

        cos = self.cos_cached[token_position]   #(batch_size, seq_len, d_half)
        sin = self.sin_cached[token_position]   #(batch_size, seq_len, d_half

        x_in_dtype = x.dtype
        x = x.to(torch.float32)    #(batch_size, seq_len, d_k)

        x_pair = x.view(*x.shape[:-1], d_k // 2, 2)   #(*batch_dims, d_half, 2)
        x_even = x_pair[..., 0]    #(batch_size, seq_len, d_half)
        x_odd = x_pair[..., 1]     #(batch_size, seq_len, d_half)

        y_even = x_even * cos - x_odd * sin   #(batch_size, seq_len, d_half)
        y_odd = x_even * sin + x_odd * cos    #(batch_size, seq_len, d_half)

        y = torch.stack((y_even, y_odd), dim=-1).view(*x.shape)    #(batch_size, seq_len, d_k)
        y = y.to(x_in_dtype)
        return y
    
if __name__ == "__main__":
    #multiheaded test
    batch_size = 2
    heads = 3
    seq_len = 4
    d_k = 8
    theta = 10000.0
    max_seq_len = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device).to(device)
    x = torch.randn(batch_size, heads, seq_len, d_k, device=device)
    token_position = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device=device)
    y = rope(x, token_position)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)