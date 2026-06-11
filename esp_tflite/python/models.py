"""Linear model variant replacing Mamba with a linear layer for MCU inference."""
import torch
import torch.nn as nn
from mamba_raw import MambaBlock

class HARMamba(nn.Module):
    """HAR model with MAMBA block (non-quantized by default).
    
    Uses standard PyTorch layers for the main model. Quantization can be applied
    at export time via TFLite post-training quantization (PTQ).
    
    Args:
        input_dim: Input feature dimension (e.g., 57 for HAR)
        d_model: Model dimension (default: 64)
        d_state: SSM state dimension (default: 16)
        d_conv: Convolution kernel size (default: 4)
        expand: Expansion factor for d_inner (default: 2)
        output_size: Number of output classes (default: 6)
    """
    def __init__(self, input_dim, d_model=64, d_state=16, d_conv=4, expand=2, output_size=6):
        super().__init__()
        
        # Store architecture parameters
        self.input_dim = input_dim
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.output_size = output_size

        self.linear_in = nn.Linear(input_dim, d_model, bias=False)
        self.mamba = MambaBlock(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, output_size, bias=False)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, seq_len, input_dim]
        
        Returns:
            Output logits [batch, output_size]
        """
        x = self.linear_in(x)      # [batch, seq_len, d_model]
        x = self.mamba(x)          # [batch, seq_len, d_model]
        x = x.transpose(1, 2)      # [batch, d_model, seq_len]
        x = self.pool(x).squeeze(-1)  # [batch, d_model]
        x = self.classifier(x)     # [batch, output_size]
        return x