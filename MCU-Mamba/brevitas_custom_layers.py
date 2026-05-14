import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from brevitas.nn import QuantIdentity, QuantLinear

class QuantConv1dDepthwise(nn.Module):
    """
    Quantized depthwise 1D convolution for MAMBA.

    Each channel has its own filter (groups=C). Supports causal padding
    for autoregressive models.

    Args:
        in_channels: Number of input channels (= output channels for depthwise)
        kernel_size: Size of the convolution kernel (default: 4 for MAMBA)
        bias: Whether to include a bias term (default: True)
        causal: If True, use left-only padding for causal convolution
        bit_width: Bit width for quantization (default: 8)
        return_quant_tensor: Whether to return QuantTensor (default: True)
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 4,
        bias: bool = True,
        causal: bool = True,
        bit_width: int = 8,
        return_quant_tensor: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.causal = causal

        # Depthwise convolution: groups = in_channels
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=bias,
            padding=0  # We handle padding manually for causal conv
        )

        # Weight quantization (will be extracted during export)
        self.weight_scale = nn.Parameter(torch.ones(1), requires_grad=False)

        # Output quantization
        self.output_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor
        )

    def forward(self, x):
        """
        Forward pass with optional causal padding.

        Args:
            x: Input tensor [B, C, L] (batch, channels, length)

        Returns:
            Output tensor [B, C, L] with same shape (if causal)
        """
        # Extract value if QuantTensor
        if hasattr(x, 'value'):
            x = x.value

        # Apply causal padding (left-only)
        if self.causal:
            x = F.pad(x, (self.kernel_size - 1, 0))

        # Apply depthwise convolution
        out = self.conv(x)

        # Quantize output
        return self.output_quant(out)
    

class QuantSiLU(nn.Module):
    """
    Quantized SiLU (Swish) activation with LUT export support.

    SiLU(x) = x * sigmoid(x)

    For INT8 input, this uses a 256-entry lookup table for exact
    integer-only execution.

    Args:
        bit_width: Bit width for quantization (default: 8)
        return_quant_tensor: Whether to return QuantTensor (default: True)
    """

    def __init__(
        self,
        bit_width: int = 8,
        return_quant_tensor: bool = True,
    ):
        super().__init__()
        self.bit_width = bit_width

        # Output quantization
        self.output_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor
        )

    def forward(self, x):
        """
        Apply SiLU activation.

        During training: standard SiLU in FP32
        During export: scale is captured for LUT generation

        Args:
            x: Input tensor (QuantTensor or Tensor)

        Returns:
            Output tensor with SiLU applied and quantized
        """
        # Extract value if QuantTensor
        if hasattr(x, 'value'):
            x = x.value

        # Apply SiLU (Swish) activation
        out = F.silu(x)

        # Quantize output
        return self.output_quant(out)


class QuantSSM(nn.Module):
    """
    Quantized State Space Model (SSM) core for MAMBA.

    Implements the selective state space recurrence:
        h[t] = dA * h[t-1] + dB' * x[t]
        y[t] = C[t] * h[t] + D * x[t]

    Where:
        - dA = exp(dt * A) (discretized state transition)
        - dB' = dt * B * s_x * phi1(dt * A) (discretized input matrix)
        - C is input-dependent (computed from x_proj)

    Args:
        d_inner: Inner dimension (number of channels M)
        d_state: State dimension (D)
        dt_rank: Rank of dt projection (default: d_inner // 16)
        bit_width: Bit width for quantization (default: 8)
        return_quant_tensor: Whether to return QuantTensor (default: True)
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int = 16,
        dt_rank: int = None,
        bit_width: int = 8,
        return_quant_tensor: bool = True,
    ):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else max(1, d_inner // 16)

        # SSM parameters (learnable)
        # A_log: Log of A parameter (A = -exp(A_log), so A is negative)
        self.A_log = nn.Parameter(torch.randn(d_state, d_inner) * 0.1)

        # D: Skip connection coefficient
        self.D = nn.Parameter(torch.ones(d_inner))

        # Separate projections for dt, B, C
        self.dt_proj_in = QuantLinear(
            d_inner, self.dt_rank,
            bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )
        self.B_proj = QuantLinear(
            d_inner, d_state,
            bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )
        self.C_proj = QuantLinear(
            d_inner, d_state,
            bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )

        # Quantization for dt, B, C projections
        self.dt_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        self.B_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        self.C_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # dt_proj: Projects dt_input to full dt (d_inner dimensional)
        self.dt_proj = QuantLinear(
            self.dt_rank, d_inner,
            bias=True,  # Important: bias initializes dt range
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )

        # Initialize dt_proj bias for appropriate dt range [0.001, 0.1]
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Initialize bias to produce dt values in target range
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Output quantization
        self.output_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor
        )

    @property
    def A(self):
        """Compute A matrix from A_log (A is always negative)."""
        return -torch.exp(self.A_log)

    def forward(self, x, z=None):
        """
        Forward pass through SSM.

        Args:
            x: Input tensor [B, L, M] or [B, M] for single timestep
            z: Optional gate input [B, L, M] for SiLU gating

        Returns:
            y: Output tensor [B, L, M] or [B, M]
        """
        # Extract value if QuantTensor
        if hasattr(x, 'value'):
            x = x.value
        if z is not None and hasattr(z, 'value'):
            z = z.value

        # Handle single timestep input
        single_step = x.dim() == 2
        if single_step:
            x = x.unsqueeze(1)  # [B, 1, M]
            if z is not None:
                z = z.unsqueeze(1)

        B, L, M = x.shape
        D = self.d_state

        # Project x to get dt_input, B_ssm, C_ssm
        x_flat = x.reshape(B * L, M)
        
        dt_input = self.dt_proj_in(x_flat)
        dt_input = self.dt_quant(dt_input.reshape(B, L, self.dt_rank))
        if hasattr(dt_input, 'value'):
            dt_input = dt_input.value
        
        B_ssm = self.B_proj(x_flat)
        B_ssm = self.B_quant(B_ssm.reshape(B, L, D))
        if hasattr(B_ssm, 'value'):
            B_ssm = B_ssm.value
        
        C_ssm = self.C_proj(x_flat)
        C_ssm = self.C_quant(C_ssm.reshape(B, L, D))
        if hasattr(C_ssm, 'value'):
            C_ssm = C_ssm.value

        # Project dt_input to full dt and apply softplus
        dt = self.dt_proj(dt_input.reshape(B * L, self.dt_rank))  # [B*L, M]
        dt = F.softplus(dt).reshape(B, L, M)  # [B, L, M]

        # Compute A (negative exponential)
        A = self.A  # [D, M]

        # print("A shape:", A.shape)
        # print("dt shape:", dt.shape)
        # print("B_ssm shape:", B_ssm.shape)
        # print("C_ssm shape:", C_ssm.shape)

        # Initialize state
        h = torch.zeros(B, M, D, device=x.device, dtype=x.dtype)

        # print("Initial h shape:", h.shape)
        # print("D shape:", self.D.shape)

        # Discretization and scan (sequential over time)
        y_list = []
        for t in range(L):
            dt_t = dt[:, t, :]  # [B, M]
            x_t = x[:, t, :]  # [B, M]
            B_t = B_ssm[:, t, :]  # [B, D]
            C_t = C_ssm[:, t, :]  # [B, D]

            # Discretize: dA = exp(dt * A), dB' = dt * B (simplified)
            # dA: [B, M] x [D, M] -> broadcast to [B, M, D]
            dA = torch.exp(dt_t.unsqueeze(-1) * A.T.unsqueeze(0))  # [B, M, D]

            # dB' = dt * B (simplified, without phi1 for training)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # [B, M, D]

            # State update: h = dA * h + dB' * x
            h = dA * h + dB * x_t.unsqueeze(-1)  # [B, M, D]

            # Output: y = C * h + D * x
            y_t = torch.sum(h * C_t.unsqueeze(1), dim=-1)  # [B, M]
            y_t = y_t + self.D * x_t

            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)  # [B, L, M]

        # print("Raw SSM output y shape:", y.shape)

        # Apply SiLU gate if z is provided
        if z is not None:
            gate = F.silu(z)

            # print("Gate shape:", gate.shape)

            y = y * gate

        # Remove time dimension if single step
        if single_step:
            y = y.squeeze(1)

        # Quantize output
        return self.output_quant(y)


class QuantMambaBlock(nn.Module):
    """
    Complete quantized MAMBA block.

    Architecture:
        x_in -> in_proj -> [x, z] split
                            |
                            v
                      conv1d -> SiLU -> x_proj -> dt_proj -> SSM
                            |                                 |
                            +----------> SiLU gate <----------+
                                              |
                                              v
                                         out_proj -> x_out

    Args:
        d_model: Model dimension (input/output)
        d_inner: Inner dimension (default: 2 * d_model)
        d_state: SSM state dimension (default: 16)
        conv_kernel: Conv1d kernel size (default: 4)
        bit_width: Bit width for quantization (default: 8)
        return_quant_tensor: Whether to return QuantTensor (default: True)
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int = None,
        d_state: int = 16,
        conv_kernel: int = 4,
        bit_width: int = 8,
        return_quant_tensor: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner if d_inner is not None else 2 * d_model
        self.d_state = d_state
        self.kernel_size = conv_kernel

        # State and gate projections (both take d_model as input)
        self.state_proj = QuantLinear(
            d_model, self.d_inner,
            bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )
        self.gate_proj = QuantLinear(
            d_model, self.d_inner,
            bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )

        # Quantization for state and gate branches
        self.state_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )
        self.gate_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Conv1d on x branch
        self.conv1d = QuantConv1dDepthwise(
            in_channels=self.d_inner,
            kernel_size=conv_kernel,
            causal=True,
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # SiLU after conv1d
        self.silu = QuantSiLU(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # SSM core
        self.ssm = QuantSSM(
            d_inner=self.d_inner,
            d_state=d_state,
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Output projection: d_inner -> d_model
        self.out_proj = QuantLinear(
            self.d_inner, d_model,
            bias=False,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )
        self.output_quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor
        )

    def forward(self, x):
        """
        Forward pass through MAMBA block.

        Args:
            x: Input tensor [B, L, d_model]

        Returns:
            Output tensor [B, L, d_model]
        """
        # Extract value if QuantTensor
        if hasattr(x, 'value'):
            x = x.value

        B, L, _ = x.shape

        # Apply state_proj and gate_proj to input
        x_branch = self.state_proj(x.reshape(B * L, self.d_model))
        x_branch = self.state_quant(x_branch.reshape(B, L, self.d_inner))
        if hasattr(x_branch, 'value'):
            x_branch = x_branch.value

        z_branch = self.gate_proj(x.reshape(B * L, self.d_model))
        z_branch = self.gate_quant(z_branch.reshape(B, L, self.d_inner))
        if hasattr(z_branch, 'value'):
            z_branch = z_branch.value

        # x branch: conv1d -> SiLU -> SSM
        # Transpose for conv1d: [B, L, M] -> [B, M, L]
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)
        if hasattr(x_branch, 'value'):
            x_branch = x_branch.value
        x_branch = x_branch.transpose(1, 2)  # Back to [B, L, M]

        # print("After conv1d, x_branch shape:", x_branch.shape)

        x_branch = self.silu(x_branch)
        if hasattr(x_branch, 'value'):
            x_branch = x_branch.value

        # SSM with z gate
        y = self.ssm(x_branch, z_branch)
        if hasattr(y, 'value'):
            y = y.value

        # Output projection
        out = self.out_proj(y.reshape(B * L, self.d_inner))
        out = self.output_quant(out.reshape(B, L, self.d_model))

        return out
    