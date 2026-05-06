import torch
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.nn import QuantIdentity, QuantLinear
from brevitas_custom_layers import QuantMambaBlock

# class Linear(nn.Module):
#     def __init__(self, input_dim, output_dim, d_model=64):
#         super().__init__()
#         self.linear_in = nn.Linear(input_dim, d_model)
#         self.linear = nn.Linear(d_model, d_model)
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.classifier = nn.Linear(d_model, output_dim)

#     def forward(self, x):
#         x = self.linear_in(x)  # [B, T, H]
#         x = self.linear(x)  # [B, T, H]
#         x = x.transpose(1, 2)  # [B, H, T]
#         x = self.pool(x).squeeze(-1)
#         return self.classifier(x)

class TinyMamba(nn.Module):
    def __init__(self, input_dim, d_model=64, d_state=16, d_conv=4, expand=2, output_size=6):
        super().__init__()

        self.linear_in = qnn.QuantLinear(input_dim, d_model, weight_bit_width=4, bias=False)
        self.mamba = QuantMambaBlock(
            d_model=d_model,
            d_inner=d_model * expand,
            d_state=d_state,
            conv_kernel=d_conv,
            bit_width=4,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = qnn.QuantLinear(d_model, output_size, weight_bit_width=4, bias=False)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.mamba(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return x
