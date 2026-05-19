import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveScanStep(nn.Module):
    """Single-step Selective Scan for TFLite inference without loop unrolling."""

    def __init__(self, d_inner, d_state, dt_size):
        super().__init__()
        self.x_proj_B = nn.Linear(d_inner, d_state, bias=False)
        self.x_proj_C = nn.Linear(d_inner, d_state, bias=False)
        self.x_proj_dt = nn.Linear(d_inner, dt_size, bias=False)
        self.dt_proj = nn.Linear(dt_size, d_inner, bias=False)

        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, state_t, h_prev):
        # state_t: (b, 1, e) -> Current token input
        # h_prev:  (b, e, n) -> Previous SSM hidden state
        
        b, _, e = state_t.shape

        B = self.x_proj_B(state_t)        # (b, 1, n)
        C = self.x_proj_C(state_t)        # (b, 1, n)

        dt = self.x_proj_dt(state_t)      # (b, 1, dt_size)
        dt = self.dt_proj(dt)             # (b, 1, e)
        dt = F.softplus(dt)               # (b, 1, e)

        A = -torch.exp(self.A_log)        # (e, n)

        # Reshape for broadcasting
        delta_t = dt.squeeze(1).unsqueeze(-1)  # (b, e, 1)
        B_t = B.squeeze(1).unsqueeze(1)        # (b, 1, n)
        C_t = C.squeeze(1).unsqueeze(1)        # (b, 1, n)
        u_t = state_t.squeeze(1).unsqueeze(-1) # (b, e, 1)
        A_reshaped = A.unsqueeze(0)            # (1, e, n)

        # Compute A_bar and B_bar for the single timestep
        A_bar = torch.exp(delta_t * A_reshaped) # (b, e, n)
        B_bar = delta_t * B_t                   # (b, e, n)

        # Compute new state and output
        h_next = A_bar * h_prev + B_bar * u_t   # (b, e, n)
        y_t = torch.sum(h_next * C_t, dim=-1)   # (b, e)

        # Feedthrough path
        D_proj = self.D.view(1, e)              # (1, e)
        y_t = y_t + D_proj * state_t.squeeze(1) # (b, e)

        return y_t.unsqueeze(1), h_next         # (b, 1, e), (b, e, n)


class MambaBlockStep(nn.Module):
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2, dt_size=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv

        self.state_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.gate_proj = nn.Linear(d_model, self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, bias=False
        )

        # Assuming SelectiveScanMock was a typo in your code for SelectiveScanSequential
        self.ssm = SelectiveScanStep(d_inner=self.d_inner, d_state=self.d_state, dt_size=dt_size)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x_t, conv_state, ssm_state):
        # x_t:        (b, 1, d_model)
        # conv_state: (b, d_inner, d_conv - 1) -> Cache of past tokens
        # ssm_state:  (b, d_inner, n)          -> Hidden state of SSM

        state_t = self.state_proj(x_t) # (b, 1, d_inner)
        gate_t = self.gate_proj(x_t)   # (b, 1, d_inner)

        state_t = state_t.transpose(1, 2) # (b, d_inner, 1)
        
        # 1. Slide the Conv1d cache
        # Append the new token to the history cache. Shape becomes (b, d_inner, d_conv)
        conv_input = torch.cat([conv_state, state_t], dim=2) 
        
        # The new state is everything except the oldest token
        conv_state_next = conv_input[:, :, 1:] 
        
        # Run 1D Conv over exactly `d_conv` tokens -> outputs exactly 1 token.
        # No padding needed here since we provided the exact history window.
        state_conv = F.conv1d(conv_input, self.conv1d.weight, bias=None, groups=self.d_inner)
        
        state_conv = F.silu(state_conv)
        state_conv = state_conv.transpose(1, 2) # (b, 1, d_inner)

        # 2. Step the SSM
        y_t, ssm_state_next = self.ssm(state_conv, ssm_state)

        # 3. Gate and output
        gate_t = F.silu(gate_t)
        y_t = y_t * gate_t

        out_t = self.out_proj(y_t) # (b, 1, d_model)

        return out_t, conv_state_next, ssm_state_next