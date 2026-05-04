import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MambaRaw(nn.Module):
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2):
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

        self.x_proj_B = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.x_proj_C = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.x_proj_dt = nn.Linear(self.d_inner, 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=False)

        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        b, l, d = x.shape

        state = self.state_proj(x)
        gate = self.gate_proj(x)

        state = state.transpose(1, 2)
        state = self.conv1d(state)[:, :, :l]
        
        state = F.silu(state)
        
        state = state.transpose(1, 2)

        B = self.x_proj_B(state)
        C = self.x_proj_C(state)

        # LoRA module
        dt = self.x_proj_dt(state)
        dt = self.dt_proj(dt)
        dt = F.softplus(dt)

        A = -torch.exp(self.A_log)
        y = selective_scan_sequential(state, dt, A, B, C, self.D)

        gate = F.silu(gate)
        y = y * gate

        return self.out_proj(y)


def selective_scan_sequential(u, delta, A, B, C, D):
    b, l, e = u.shape
    n = A.shape[1]

    h = torch.zeros(b, e, n, device=u.device, dtype=u.dtype)
    ys = torch.empty(b, l, e, device=u.device, dtype=u.dtype)

    for t in range(l):
        delta_t = delta[:, t, :, None]
        A_bar = torch.exp(delta_t * A)

        B_t = B[:, t, :].unsqueeze(1)
        B_bar = delta_t * B_t

        h = A_bar * h + B_bar * u[:, t, :, None]

        C_t = C[:, t, :].unsqueeze(1)
        y = (h * C_t).sum(dim=-1) + D[None, :] * u[:, t, :]

        ys[:, t, :] = y

    return ys