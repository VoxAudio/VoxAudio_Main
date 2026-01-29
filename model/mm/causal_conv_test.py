import torch
import torch.nn as nn


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
        
        self.total_padding = (kernel_size - 1) * dilation
        
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