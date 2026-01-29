import torch
from torch import nn
from torch.nn import functional as F
import math


class ChannelLastConv1d(nn.Conv1d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x

class CausalChannelLastConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros'):
        
        # 1. 首先计算因果卷积所需要的总 padding
        # 这个 padding 会被应用到输入的左侧
        self.total_padding = (kernel_size - 1) * dilation
        
        # 2. 调用父类 nn.Conv1d 的构造函数
        # 我们将计算好的 padding 值传递给它，让父类来管理参数和卷积的准备工作
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.total_padding, # 将 padding 传递给父类
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x_permuted = x.permute(0, 2, 1)
        
        out_permuted = super().forward(x_permuted)
        
        if self.total_padding > 0:
            out_causal = out_permuted[..., :-self.total_padding]
        else:
            out_causal = out_permuted
            
        out = out_causal.permute(0, 2, 1)
        
        return out
        
# https://github.com/Stability-AI/sd3-ref
class MLP(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ConvMLP(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ChannelLastConv1d(dim,
                                    hidden_dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)
        self.w2 = ChannelLastConv1d(hidden_dim,
                                    dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)
        self.w3 = ChannelLastConv1d(dim,
                                    hidden_dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class CausalConvMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        dilation: int = 1, # 引入 dilation 参数用于扩张因果卷积
    ):
        """
        Args:
            dim (int): 输入和输出维度 (通道数)。
            hidden_dim (int): 隐藏层维度。
            multiple_of (int): 确保隐藏维度是此值的倍数。
            kernel_size (int): 卷积核大小。
            dilation (int): 扩张率 (用于扩张因果卷积)。
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = CausalChannelLastConv1d(dim,
                                          hidden_dim,
                                          bias=False,
                                          kernel_size=kernel_size,
                                          dilation=dilation)
        self.w2 = CausalChannelLastConv1d(hidden_dim,
                                          dim,
                                          bias=False,
                                          kernel_size=kernel_size,
                                          dilation=dilation)
        self.w3 = CausalChannelLastConv1d(dim,
                                          hidden_dim,
                                          bias=False,
                                          kernel_size=kernel_size,
                                          dilation=dilation)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class ChunkChannelLastConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 chunk_size: int = 32, 
                 **kwargs):
        
        super().__init__(
            in_channels, out_channels, kernel_size, 
            **kwargs 
        )
        
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, L, C)
        B, L, C = x.shape
        
        # 1. Zero padding to make length L divisible by chunk_size
        remainder = L % self.chunk_size
        padding_len = 0
        x_padded = x
        
        if remainder != 0:
            padding_len = self.chunk_size - remainder
            padding = torch.zeros((B, padding_len, C), dtype=x.dtype, device=x.device)
            x_padded = torch.cat([x, padding], dim=1)
        
        L_padded = L + padding_len

        # 2. Reshape for chunking: (B, L_padded, C) -> (B * num_chunks, chunk_size, C)
        num_chunks = L_padded // self.chunk_size
        x_reshaped = x_padded.reshape(B * num_chunks, self.chunk_size, C)
        
        # 3. Permute (Channel-Last -> Channel-First): 
        # (B * num_chunks, chunk_size, C) -> (B * num_chunks, C, chunk_size)
        x_permuted = x_reshaped.permute(0, 2, 1)
        # 4. Apply nn.Conv1d's forward
        out_permuted = super().forward(x_permuted) 
        # 5. Permute back (Channel-First -> Channel-Last):
        # (B * num_chunks, C_out, L_out_chunk) -> (B * num_chunks, L_out_chunk, C_out)
        out_reshaped = out_permuted.permute(0, 2, 1)
        
        # 6. Reshape back to original Batch shape
        L_out_chunk = out_reshaped.shape[1]
        L_out_padded = L_out_chunk * num_chunks
        
        out_padded = out_reshaped.reshape(B, L_out_padded, self.out_channels)
        
        # 7. Crop the zero padding added earlier
        out = out_padded[:, :L, :]
        return out


class ChunkConvMLP(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
        chunk_size: int = 32
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ChunkChannelLastConv1d(dim,
                                    hidden_dim,
                                    kernel_size=kernel_size,
                                    chunk_size=chunk_size,
                                    bias=False,
                                    padding=padding)
        self.w2 = ChunkChannelLastConv1d(hidden_dim,
                                    dim,
                                    kernel_size=kernel_size,
                                    chunk_size=chunk_size,
                                    bias=False,
                                    padding=padding)
        self.w3 = ChunkChannelLastConv1d(dim,
                                    hidden_dim,
                                    kernel_size=kernel_size,
                                    chunk_size=chunk_size,
                                    bias=False,
                                    padding=padding)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
