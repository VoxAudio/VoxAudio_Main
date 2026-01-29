# embeddings.py
import torch
import torch.nn as nn

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, dim, frequency_embedding_size=256, max_period=10000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.dim = dim
        self.max_period = max_period
        assert dim % 2 == 0, 'dim must be even.'

        # 使用 torch.no_grad() + register_buffer
        with torch.no_grad():
            freqs = 1.0 / (10000 ** (torch.arange(0, frequency_embedding_size, 2, dtype=torch.float32) / frequency_embedding_size))
            freq_scale = 10000 / max_period  # 保持频率对齐
            freqs = freq_scale * freqs  # (freq_size//2,)
        # print(freqs)
        self.register_buffer('freqs', freqs, persistent=False)

    def timestep_embedding(self, t):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :return: an (N, D) Tensor of positional embeddings.
        """
        if len(t.shape) == 1:
            args = t[:, None].float() * self.freqs[None, :]  # (N, D//2)
        elif len(t.shape) == 2:
            args = t[:, :, None].float() * self.freqs[None, None, :]  # (N, D//2)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (N, D)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t).to(t.dtype)  # (N, D)
        t_emb = self.mlp(t_freq)  # (N, dim)
        return t_emb
