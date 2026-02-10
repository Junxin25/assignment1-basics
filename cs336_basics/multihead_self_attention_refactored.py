
class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 max_seq_len: int,
                 rope_theta: float = 10000.0,
                 use_rope: bool = False,
                 device: torch.device = torch.device('cpu')):