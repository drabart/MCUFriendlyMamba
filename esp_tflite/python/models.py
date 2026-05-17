"""Linear model variant replacing Mamba with a linear layer for MCU inference."""
import torch.nn as nn
from mamba_raw import MambaBlock

class HARLinear(nn.Module):
    """Simplified model with linear layer replacing mamba block.
    
    Uses standard PyTorch layers. Quantization is applied at TFLite level using PTQ.
    
    Input shape: (batch, 10, 57) - HAR dataset
    Output shape: (batch, 6) - 6 activity classes
    """
    def __init__(self, input_dim, d_model=64, output_size=6):
        super().__init__()
        
        # Input projection: (10, 57) -> (10, d_model)
        self.linear_in = nn.Linear(input_dim, d_model, bias=False)
        
        # Middle linear layer (replaces mamba): (10, d_model) -> (10, d_model)
        # Reshape to (batch*10, d_model), apply linear, reshape back
        self.middle_linear = nn.Linear(d_model, d_model, bias=False)
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Output classifier
        self.classifier = nn.Linear(d_model, output_size, bias=False)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 10, 57) - time series data
        
        Returns:
            logits: (batch, 6) - activity class predictions
        """
        # x: (batch, 10, 57)
        x = self.linear_in(x)  # (batch, 10, d_model)
        
        # Apply middle linear layer per timestep
        batch, time, d_model = x.shape
        x = x.reshape(batch * time, d_model)  # (batch*10, d_model)
        x = self.middle_linear(x)  # (batch*10, d_model)
        x = x.reshape(batch, time, d_model)  # (batch, 10, d_model)
        
        # Global average pooling
        x = x.transpose(1, 2)  # (batch, d_model, 10)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        
        # Classification
        x = self.classifier(x)  # (batch, 6)
        return x


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

        # Non-quantized input layer
        self.linear_in = nn.Linear(input_dim, d_model, bias=False)
        
        # Non-quantized MAMBA block
        # self.mamba = MambaBlock(
        #     d_model=d_model, 
        #     d_state=d_state, 
        #     d_conv=d_conv, 
        #     expand=expand
        # )
        self.mamba = MambaBlock(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Non-quantized output layer
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

