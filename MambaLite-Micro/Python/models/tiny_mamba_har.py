import torch.nn as nn
import torch

from .mamba_raw import MambaRaw


class TinyMambaHAR(nn.Module):
    def __init__(self, input_dim, mamba_channel_width, num_classes=6):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

        self.linear_in = nn.Linear(input_dim, mamba_channel_width)
        self.mamba = MambaRaw(d_model=mamba_channel_width, d_state=16, expand=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(mamba_channel_width, num_classes)

    def forward(self, x):
        x = self.quant(x)
        x = self.linear_in(x)
        x = self.mamba(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


def build_model(input_dim, mamba_channel_width, num_classes=6, device=None):
    model = TinyMambaHAR(input_dim=input_dim, mamba_channel_width=mamba_channel_width, num_classes=num_classes)
    if device is not None:
        model = model.to(device)
    return model