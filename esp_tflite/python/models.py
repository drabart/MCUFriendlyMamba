"""Linear model variant replacing Mamba with a linear layer for MCU inference."""
import torch
import torch.nn as nn
from mamba_raw import MambaBlock
from mamba_step import MambaBlockStep

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


class HARMambaStep(nn.Module):
    """HAR model with single-step MAMBA block for token-by-token inference.
    
    Uses MambaBlockStep for efficient step-wise inference without loop unrolling.
    Suitable for streaming or MCU inference where tokens arrive sequentially.
    
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
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv

        # Input layer
        self.linear_in = nn.Linear(input_dim, d_model, bias=False)
        
        # Single-step MAMBA block
        self.mamba = MambaBlockStep(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layer
        self.classifier = nn.Linear(d_model, output_size, bias=False)

    def initialize_states(self, batch_size, device):
        """Initialize conv_state and ssm_state for a given batch size."""
        # conv_state: (batch, d_inner, d_conv - 1) - cache of previous tokens
        conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv - 1, device=device)
        
        # ssm_state: (batch, d_inner, d_state) - hidden state of SSM
        ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=device)
        
        return conv_state, ssm_state

    def forward_step(self, x_t, conv_state, ssm_state):
        """Process a single token and update states.
        
        Args:
            x_t: Single token [batch, 1, input_dim]
            conv_state: Conv cache [batch, d_inner, d_conv - 1]
            ssm_state: SSM hidden state [batch, d_inner, d_state]
        
        Returns:
            out_t: Output for this token [batch, 1, d_model]
            conv_state_next: Updated conv cache
            ssm_state_next: Updated SSM state
        """
        # Project input
        x_t = self.linear_in(x_t)  # [batch, 1, d_model]
        
        # Step the mamba block
        y_t, conv_state_next, ssm_state_next = self.mamba(x_t, conv_state, ssm_state)
        
        return y_t, conv_state_next, ssm_state_next

    def forward(self, x, return_states=False):
        """Process a sequence step by step.
        
        Args:
            x: Input sequence [batch, seq_len, input_dim]
            return_states: If True, return final states along with output
        
        Returns:
            output: Classification output [batch, output_size]
            (optional) conv_state, ssm_state: Final states if return_states=True
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize states
        conv_state, ssm_state = self.initialize_states(batch_size, device)
        
        # Collect outputs for all timesteps
        ys = []
        
        # Process sequence step by step
        for t in range(seq_len):
            x_t = x[:, t:t+1, :]  # [batch, 1, input_dim]
            y_t, conv_state, ssm_state = self.forward_step(x_t, conv_state, ssm_state)
            ys.append(y_t)
        
        # Stack outputs: [batch, seq_len, d_model]
        y_seq = torch.cat(ys, dim=1)
        
        # Global average pooling
        y_seq = y_seq.transpose(1, 2)  # [batch, d_model, seq_len]
        y_pooled = self.pool(y_seq).squeeze(-1)  # [batch, d_model]
        
        # Classification
        output = self.classifier(y_pooled)  # [batch, output_size]
        
        if return_states:
            return output, conv_state, ssm_state
        return output
