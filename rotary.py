import torch
from torch import nn

class RotaryPositionEncoder(nn.Module):
    def __init__(self, hidden_size, num_attention_heads) -> None:
        super().__init__()
        self.register_buffer('freq_cis', self._freqs_cis(hidden_size // num_attention_heads, 512))
        self.num_attention_heads = num_attention_heads

    def _freqs_cis(self, dim, seq_len, theta=10000.0):
        freqs = 1.0 / (theta ** torch.arange(0, dim, 2).float().div(dim))
        t = torch.arange(seq_len)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return torch.view_as_real(freqs_cis)

    def apply_rotary_qk(self, position, *args):
        pos = position.clip(0, self.freq_cis.shape[0] - 1)
        freqs = self.freq_cis[pos]
        bn, seq, heads, dim = args[0].shape
        args_ = [x.reshape(bn, seq, heads // 2, 2, dim // 2, 2) for x in args]
        freqs = freqs[:, :, None, :, :]
        outputs = [self._complex_mul(x, freqs).view(y.shape) for x, y in zip(args_, args)]
        return outputs

    def apply_rotary_v(self, position, value):
        _value = value.view(*value.shape[:2], self.num_attention_heads, -1)
        output = self.apply_rotary_qk(position, _value)[0]
        output = output.view(*output.shape[:2], -1)
        return output

    def _complex_mul(self, x, y):
        real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
        imag = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
        return torch.stack([real, imag], -1)