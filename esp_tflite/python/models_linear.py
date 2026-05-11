"""Linear model variant replacing Mamba with a linear layer for MCU inference."""
import torch
import torch.nn as nn


class TinyLinear(nn.Module):
    """Simplified model with linear layer replacing mamba block.
    
    Uses standard PyTorch layers. Quantization is applied at TFLite level using PTQ.
    
    Input shape: (batch, 10, 57) - HAR dataset
    Output shape: (batch, 6) - 6 activity classes
    """
    def __init__(self, input_dim, d_model=64, output_size=6, bit_width=8):
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
