"""Analyze output shapes in HARMamba model, unwinding Mamba block internals.

This script traces through the model architecture and reports:
- Tensor shapes at each layer (batch=1 only)
- Data flow through linear, conv, and SSM components
- Internal Mamba block structure breakdown
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from models import HARMamba


@dataclass
class LayerInfo:
    """Information about a layer's output."""
    name: str
    layer_type: str
    output_shape: Tuple[int, ...]
    depth: int = 0  # For nested layers
    parent: Optional[str] = None
    
    def indent_str(self) -> str:
        return "  " * self.depth


class OutputShapeAnalyzer:
    """Analyzes output shapes through HARMamba model with Mamba internals unwound."""
    
    def __init__(self, model: HARMamba, batch_size: int = 1, seq_len: int = 10, input_features: int = 57):
        self.model = model
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.input_features = input_features
        self.layers: List[LayerInfo] = []
    
    def analyze(self) -> List[LayerInfo]:
        """Run shape analysis through entire model."""
        self.layers = []
        
        # Input
        x = torch.randn(self.batch_size, self.seq_len, self.input_features)
        self.layers.append(LayerInfo(
            name="input",
            layer_type="Input",
            output_shape=tuple(x.shape),
            depth=0
        ))
        
        # Linear projection
        with torch.no_grad():
            x = self.model.linear_in(x)
        self.layers.append(LayerInfo(
            name="linear_in",
            layer_type="Linear",
            output_shape=tuple(x.shape),
            depth=0
        ))
        
        # Unwind Mamba block
        self._analyze_mamba(x)
        
        # Pool
        with torch.no_grad():
            x_for_pool = x.transpose(1, 2)
            x_pooled = self.model.pool(x_for_pool).squeeze(-1)
        self.layers.append(LayerInfo(
            name="global_avg_pool",
            layer_type="AdaptiveAvgPool1d",
            output_shape=tuple(x_pooled.shape),
            depth=0
        ))
        
        # Classifier
        with torch.no_grad():
            x_out = self.model.classifier(x_pooled)
        self.layers.append(LayerInfo(
            name="classifier",
            layer_type="Linear",
            output_shape=tuple(x_out.shape),
            depth=0
        ))
        
        return self.layers
    
    def _analyze_mamba(self, x: torch.Tensor):
        """Unwind Mamba block and trace shapes through internal components."""
        
        # Get Mamba block
        mamba = self.model.mamba
        
        self.layers.append(LayerInfo(
            name="mamba_block",
            layer_type="Mamba (Internal Structure)",
            output_shape=tuple(x.shape),
            depth=0
        ))
        
        with torch.no_grad():
            # State projection: d_model -> d_inner
            state = mamba.state_proj(x)
            self.layers.append(LayerInfo(
                name="  ├─ state_proj",
                layer_type="Linear (d_model→d_inner)",
                output_shape=tuple(state.shape),
                depth=1,
                parent="mamba_block"
            ))
            
            # Gate projection: d_model -> d_inner
            gate = mamba.gate_proj(x)
            self.layers.append(LayerInfo(
                name="  ├─ gate_proj",
                layer_type="Linear (d_model→d_inner)",
                output_shape=tuple(gate.shape),
                depth=1,
                parent="mamba_block"
            ))
            
            # Conv1d preparation: reshape for conv (batch, seq_len, d_inner) -> (batch, d_inner, seq_len)
            state_for_conv = state.transpose(1, 2)
            self.layers.append(LayerInfo(
                name="  ├─ transpose_for_conv",
                layer_type="Transpose",
                output_shape=tuple(state_for_conv.shape),
                depth=1,
                parent="mamba_block"
            ))
            
            # Conv1d: (batch, d_inner, seq_len) -> (batch, d_inner, seq_len)
            state_conv = mamba.conv1d(state_for_conv)[:, :, :self.seq_len]
            self.layers.append(LayerInfo(
                name="  ├─ conv1d",
                layer_type="Conv1d (kernel_size=4, groups=d_inner)",
                output_shape=tuple(state_conv.shape),
                depth=1,
                parent="mamba_block"
            ))
            
            # Reshape back: (batch, d_inner, seq_len) -> (batch, seq_len, d_inner)
            state_after_conv = state_conv.transpose(1, 2)
            self.layers.append(LayerInfo(
                name="  ├─ transpose_after_conv",
                layer_type="Transpose",
                output_shape=tuple(state_after_conv.shape),
                depth=1,
                parent="mamba_block"
            ))
            
            # Activation (ReLU)
            state_act = torch.nn.functional.relu(state_after_conv)
            self.layers.append(LayerInfo(
                name="  ├─ relu",
                layer_type="ReLU",
                output_shape=tuple(state_act.shape),
                depth=1,
                parent="mamba_block"
            ))
            
            # Selective Scan (SSM)
            self.layers.append(LayerInfo(
                name="  ├─ selective_scan",
                layer_type="SelectiveScanSequential",
                output_shape=tuple(state_act.shape),
                depth=1,
                parent="mamba_block"
            ))
            
            # Unwind SelectiveScan internals
            ssm = mamba.ssm
            d_state = ssm.A_log.shape[1]
            d_inner = ssm.A_log.shape[0]
            
            self.layers.append(LayerInfo(
                name="  │  ├─ x_proj_B (d_inner→d_state)",
                layer_type="Linear",
                output_shape=(self.batch_size, self.seq_len, d_state),
                depth=2,
                parent="selective_scan"
            ))
            
            self.layers.append(LayerInfo(
                name="  │  ├─ x_proj_C (d_inner→d_state)",
                layer_type="Linear",
                output_shape=(self.batch_size, self.seq_len, d_state),
                depth=2,
                parent="selective_scan"
            ))
            
            dt_size = ssm.x_proj_dt.out_features
            self.layers.append(LayerInfo(
                name="  │  ├─ x_proj_dt (d_inner→dt_size)",
                layer_type="Linear",
                output_shape=(self.batch_size, self.seq_len, dt_size),
                depth=2,
                parent="selective_scan"
            ))
            
            self.layers.append(LayerInfo(
                name="  │  ├─ dt_proj (dt_size→d_inner)",
                layer_type="Linear",
                output_shape=(self.batch_size, self.seq_len, d_inner),
                depth=2,
                parent="selective_scan"
            ))
            
            self.layers.append(LayerInfo(
                name="  │  ├─ softplus(dt)",
                layer_type="Softplus",
                output_shape=(self.batch_size, self.seq_len, d_inner),
                depth=2,
                parent="selective_scan"
            ))
            
            self.layers.append(LayerInfo(
                name="  │  └─ ssm_loop (per-timestep)",
                layer_type="Sequential Loop",
                output_shape=(self.batch_size, self.seq_len, d_inner),
                depth=2,
                parent="selective_scan"
            ))
            
            self.layers.append(LayerInfo(
                name="  │     (hidden state: batch×d_inner×d_state)",
                layer_type="State",
                output_shape=(self.batch_size, d_inner, d_state),
                depth=3,
                parent="ssm_loop"
            ))
            
            # Run SSM to get output
            y_ssm = mamba.ssm(state_act)
            
            # Gate activation (ReLU)
            gate_act = torch.nn.functional.relu(gate)
            self.layers.append(LayerInfo(
                name="  ├─ gate_relu",
                layer_type="ReLU",
                output_shape=tuple(gate_act.shape),
                depth=1,
                parent="mamba_block"
            ))
            
            # Gate multiplication
            y_gated = y_ssm * gate_act
            self.layers.append(LayerInfo(
                name="  ├─ gate_multiply",
                layer_type="Element-wise Multiply",
                output_shape=tuple(y_gated.shape),
                depth=1,
                parent="mamba_block"
            ))
            
            # Out projection: d_inner -> d_model
            y_out = mamba.out_proj(y_gated)
            self.layers.append(LayerInfo(
                name="  └─ out_proj",
                layer_type="Linear (d_inner→d_model)",
                output_shape=tuple(y_out.shape),
                depth=1,
                parent="mamba_block"
            ))


def print_analysis(layers: List[LayerInfo]):
    """Print shape analysis results."""
    
    print("\n" + "=" * 100)
    print("HARMamba OUTPUT SHAPES (Mamba Block Unwound)")
    print("=" * 100)
    
    print(f"\nBatch Size: 1 | Sequence Length: 10 | Input Features: 57")
    
    print("\n" + "-" * 100)
    print("DATA FLOW")
    print("-" * 100)
    
    for layer in layers:
        indent = layer.indent_str()
        shape_str = " × ".join(str(d) for d in layer.output_shape)
        print(f"{indent}{layer.name:<40} {shape_str:>30}")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    # Create model with default parameters
    model = HARMamba(
        input_dim=57,      # HAR dataset
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        output_size=6      # 6 activity classes
    )
    
    model.eval()
    
    # Run analysis (batch_size=1 only)
    print("\n" + "#" * 100)
    print("HARMamba OUTPUT SHAPE ANALYSIS")
    print("#" * 100)
    
    analyzer = OutputShapeAnalyzer(model, batch_size=1, seq_len=10, input_features=57)
    layers = analyzer.analyze()
    
    print_analysis(layers)
    
    print("\nAnalysis complete!")
