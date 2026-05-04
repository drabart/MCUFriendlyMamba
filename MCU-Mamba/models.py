import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_raw import MambaRaw

class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.linear_in(x)  # [B, T, H]
        x = self.linear(x)  # [B, T, H]
        x = x.transpose(1, 2)  # [B, H, T]
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

class ResidualMamba(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, x):
        return x + self.mamba(x)  # residual keeps gradients healthy

class TinyMamba(nn.Module):
    def __init__( self, input_dim=57, d_model=64, d_state=16, d_conv=4, expand=2, output_size=6):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, d_model)
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.linear_in(x)  # [B, T, H]
        x = self.mamba(x)  # [B, T, H]
        x = x.transpose(1, 2)  # [B, H, T]
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class TinyMambaMulti(nn.Module):
    def __init__( self, input_dim=57, d_model=64, d_state=16, d_conv=4, expand=2, n_layers=1, output_size=6):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, d_model)
        self.mamba_layers = nn.Sequential(
            *[
                ResidualMamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_layers)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.linear_in(x)  # [B, T, H]
        x = self.mamba_layers(x)  # [B, T, H]
        x = x.transpose(1, 2)  # [B, H, T]
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

class TinyMambaHAR(nn.Module):
    def __init__(self, input_dim, d_model=64, d_state=16, d_conv=4, expand=2, output_size=6):
        super().__init__()

        self.linear_in = nn.Linear(input_dim, d_model, bias=False)
        self.mamba = MambaRaw(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, output_size, bias=False)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.mamba(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return x
