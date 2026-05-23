import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class SelectiveScanSequential(nn.Module):
    """Selective Scan operator optimized for TFLite conversion and quantization."""

    def __init__(self, d_inner, d_state, dt_size):
        super().__init__()
        self.x_proj_B = nn.Linear(d_inner, d_state, bias=False)
        self.x_proj_C = nn.Linear(d_inner, d_state, bias=False)
        self.x_proj_dt = nn.Linear(d_inner, dt_size, bias=False)
        self.dt_proj = nn.Linear(dt_size, d_inner, bias=False)

        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, state):
        b, l, e = state.shape

        B = self.x_proj_B(state)
        C = self.x_proj_C(state)

        dt = self.x_proj_dt(state)
        dt = self.dt_proj(dt)
        # dt = F.softplus(dt)
        dt = F.relu(dt)

        A = -torch.exp(self.A_log)

        n = A.shape[1]

        # Reshape D for clean broadcasting without dynamic slicing in the loop
        # D shape: (channels,) -> (1, 1, channels)
        D_proj = self.D.view(1, 1, e)

        # Pre-reshape inputs to avoid repetitive unsqueezing inside the loop
        # delta: (b, l, e) -> (b, l, e, 1)
        delta_expanded = dt.unsqueeze(-1)
        
        # B and C: (b, l, n) -> (b, l, 1, n)
        B_expanded = B.unsqueeze(2)
        C_expanded = C.unsqueeze(2)
        
        # u: (b, l, e) -> (b, l, e, 1)
        u_expanded = state.unsqueeze(-1)

        # Initialize hidden state
        h = torch.zeros(b, e, n, device=state.device, dtype=state.dtype)
        
        # Use a list to collect outputs instead of in-place tensor assignment
        ys_list = []

        # Reshape A for explicit broadcasting: (1, e, n)
        A_reshaped = A.unsqueeze(0)

        for t in range(l):
            # Grab slices for timestep t (slicing along dim 1 is TFLite friendly)
            delta_t = delta_expanded[:, t]  # (b, e, 1)
            B_t = B_expanded[:, t]          # (b, 1, n)
            C_t = C_expanded[:, t]          # (b, 1, n)
            u_t = u_expanded[:, t]          # (b, e, 1)

            # Compute A_bar and B_bar using explicit broadcasting
            A_bar = torch.exp(delta_t * A_reshaped)    # (b, e, n)
            B_bar = delta_t * B_t           # (b, e, n)

            # Update hidden state
            h = A_bar * h + B_bar * u_t     # (b, e, n)

            # Compute output y for timestep t
            y_t = torch.sum(h * C_t, dim=-1) # (b, e)
            
            ys_list.append(y_t)

        # Stack along the time dimension: (b, l, e)
        ys = torch.stack(ys_list, dim=1)

        # Add the direct feedthrough path outside the loop (fully vectorized)
        ys = ys + D_proj * state

        return ys


class MambaBlock(nn.Module):
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2, dt_size=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.state_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.gate_proj = nn.Linear(d_model, self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=False
        )

        self.ssm = SelectiveScanSequential(d_inner=self.d_inner, d_state=self.d_state, dt_size=dt_size)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        b, l, d = x.shape

        state = self.state_proj(x)
        gate = self.gate_proj(x)

        state = state.transpose(1, 2)
        state = self.conv1d(state)[:, :, :l]
        
        # state = F.silu(state)
        state = F.relu(state)
        
        state = state.transpose(1, 2)

        y = self.ssm(state)

        # gate = F.silu(gate)
        gate = F.relu(gate)
        y = y * gate

        return self.out_proj(y)
