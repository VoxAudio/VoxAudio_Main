# Reference: # https://github.com/bytedance/Make-An-Audio-2

import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5, norm_fn=torch.log10):
    return norm_fn(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes, norm_fn):
    output = dynamic_range_compression_torch(magnitudes, norm_fn=norm_fn)
    return output


class STFTConverter(nn.Module):

    def __init__(
        self,
        *,
        sampling_rate: float = 16_000,
        n_fft: int = 1024,
        num_mels: int = 128,
        hop_size: int = 256,
        win_size: int = 1024,
        fmin: float = 0,
        fmax: float = 8_000,
        norm_fn=torch.log,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.norm_fn = norm_fn

        mel = librosa_mel_fn(sr=self.sampling_rate,
                             n_fft=self.n_fft,
                             n_mels=self.num_mels,
                             fmin=self.fmin,
                             fmax=self.fmax)
        mel_basis = torch.from_numpy(mel).float()
        hann_window = torch.hann_window(self.win_size)

        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('hann_window', hann_window)

    @property
    def device(self):
        return self.hann_window.device

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # input: batch_size * length
        bs = waveform.shape[0]
        waveform = waveform.clamp(min=-1., max=1.)

        spec = torch.stft(waveform,
                          self.n_fft,
                          hop_length=self.hop_size,
                          win_length=self.win_size,
                          window=self.hann_window,
                          center=True,
                          pad_mode='reflect',
                          normalized=False,
                          onesided=True,
                          return_complex=True)

        spec = torch.view_as_real(spec)

        power = spec.pow(2).sum(-1)
        angle = torch.atan2(spec[..., 1], spec[..., 0])

        power = torch.log10(power.clamp(min=1e-5))
        spec = torch.stack([power, angle], dim=-1)

        spec = rearrange(spec, 'b f t c -> b c f t', b=bs)

        return spec

    def invert(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        bs = spec.shape[0]

        spec = rearrange(spec, 'b c f t -> b f t c', b=bs).contiguous()
        power = spec[..., 0]
        power = 10**power

        unit_vector = torch.stack([
            torch.cos(spec[..., 1]),
            torch.sin(spec[..., 1]),
        ], dim=-1)

        spec = torch.sqrt(power) * unit_vector

        # spec = rearrange(spec, '(b c) f t -> b f t c', b=bs).contiguous()
        spec = torch.view_as_complex(spec)

        waveform = torch.istft(
            spec,
            self.n_fft,
            length=length,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=True,
            normalized=False,
            onesided=True,
            return_complex=False,
        )

        return waveform
