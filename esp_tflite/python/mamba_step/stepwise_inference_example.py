"""Stepwise inference: decompose trained model into 3 separate models.

Loads best_model.pt (HARMamba) and extracts weights into:
1. PreSSM: Input projection + state/gate + conv
2. StepSSM: Single timestep SSM with persistent state (called 10x)
3. PostSSM: Gate multiply + out_proj + pool + classify
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import HARMamba
from data import load_har_data


class PreSSMModule(nn.Module):
    """Pre-SSM: input projection through conv1d."""
    def __init__(self, input_dim=57, d_model=64, d_inner=128, d_conv=4):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, d_model, bias=False)
        self.state_proj = nn.Linear(d_model, d_inner, bias=False)
        self.gate_proj = nn.Linear(d_model, d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, 
                                 padding=d_conv-1, bias=False)
    
    def forward(self, x):
        x = self.linear_in(x)  # (B, T, d_model)
        state = self.state_proj(x)  # (B, T, d_inner)
        gate = self.gate_proj(x)  # (B, T, d_inner)
        
        state = state.transpose(1, 2)  # (B, d_inner, T)
        state = self.conv1d(state)[:, :, :x.shape[1]]
        state = state.transpose(1, 2)  # (B, T, d_inner)
        state = F.relu(state)
        
        return state, gate


class StepSSMModule(nn.Module):
    """Single-timestep SSM with persistent hidden state."""
    def __init__(self, d_inner=128, d_state=16, dt_size=8):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        
        self.x_proj_B = nn.Linear(d_inner, d_state, bias=False)
        self.x_proj_C = nn.Linear(d_inner, d_state, bias=False)
        self.x_proj_dt = nn.Linear(d_inner, dt_size, bias=False)
        self.dt_proj = nn.Linear(dt_size, d_inner, bias=False)
        
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))
        
        self.hidden_state = None
    
    def reset_state(self, batch_size, device, dtype):
        self.hidden_state = torch.zeros(batch_size, self.d_inner, self.d_state,
                                        device=device, dtype=dtype)
    
    def forward(self, x_t):
        """Process single timestep (B, d_inner)."""
        B = self.x_proj_B(x_t)  # (B, d_state)
        C = self.x_proj_C(x_t)  # (B, d_state)
        dt = self.x_proj_dt(x_t)
        dt = self.dt_proj(dt)
        dt = F.relu(dt)
        
        A = -torch.exp(self.A_log)
        
        # Reshape for broadcasting
        x_expanded = x_t.unsqueeze(-1)  # (B, d_inner, 1)
        dt_expanded = dt.unsqueeze(-1)  # (B, d_inner, 1)
        B_expanded = B.unsqueeze(1)  # (B, 1, d_state)
        C_expanded = C.unsqueeze(1)  # (B, 1, d_state)
        
        A_bar = torch.exp(dt_expanded * A.unsqueeze(0))  # (B, d_inner, d_state)
        B_bar = dt_expanded * B_expanded  # (B, d_inner, d_state)
        
        self.hidden_state = A_bar * self.hidden_state + B_bar * x_expanded
        
        y_t = torch.sum(self.hidden_state * C_expanded, dim=-1)  # (B, d_inner)
        y_t = y_t + self.D.view(1, -1) * x_t
        
        return y_t


class PostSSMModule(nn.Module):
    """Post-SSM: gate, out_proj, pool, classify."""
    def __init__(self, d_model=64, d_inner=128, output_size=6):
        super().__init__()
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, output_size, bias=False)
    
    def forward(self, y, gate):
        gate = F.relu(gate)
        y = y * gate  # (B, T, d_inner)
        y = self.out_proj(y)  # (B, T, d_model)
        y = y.transpose(1, 2)  # (B, d_model, T)
        y = self.pool(y).squeeze(-1)  # (B, d_model)
        logits = self.classifier(y)  # (B, output_size)
        return logits


def load_trained_weights(pre_ssm, step_ssm, post_ssm, model_path):
    """Extract weights from trained HARMamba model into 3 modules."""
    print(f"Loading weights from {model_path}...")
    
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    
    # Pre-SSM weights
    pre_ssm.linear_in.weight.data = state_dict['linear_in.weight']
    pre_ssm.state_proj.weight.data = state_dict['mamba.state_proj.weight']
    pre_ssm.gate_proj.weight.data = state_dict['mamba.gate_proj.weight']
    pre_ssm.conv1d.weight.data = state_dict['mamba.conv1d.weight']
    
    # Step-SSM weights
    step_ssm.x_proj_B.weight.data = state_dict['mamba.ssm.x_proj_B.weight']
    step_ssm.x_proj_C.weight.data = state_dict['mamba.ssm.x_proj_C.weight']
    step_ssm.x_proj_dt.weight.data = state_dict['mamba.ssm.x_proj_dt.weight']
    step_ssm.dt_proj.weight.data = state_dict['mamba.ssm.dt_proj.weight']
    step_ssm.A_log.data = state_dict['mamba.ssm.A_log']
    step_ssm.D.data = state_dict['mamba.ssm.D']
    
    # Post-SSM weights
    post_ssm.out_proj.weight.data = state_dict['mamba.out_proj.weight']
    post_ssm.classifier.weight.data = state_dict['classifier.weight']
    
    print("✓ Weights loaded\n")


def inference_stepwise(pre_ssm, step_ssm, post_ssm, x):
    """Run inference using 3 separate models."""
    with torch.no_grad():
        # Step 1: Pre-SSM
        state, gate = pre_ssm(x)  # (B, T, d_inner)
        
        # Step 2: Step-by-step SSM
        batch_size, seq_len = x.shape[0], x.shape[1]
        step_ssm.reset_state(batch_size, x.device, x.dtype)
        
        ys = []
        for t in range(seq_len):
            y_t = step_ssm(state[:, t, :])
            ys.append(y_t)
        
        y_ssm = torch.stack(ys, dim=1)  # (B, T, d_inner)
        
        # Step 3: Post-SSM
        logits = post_ssm(y_ssm, gate)
        
        return logits


def main():
    print("=" * 80)
    print("STEPWISE SSM INFERENCE - FULL TEST SET EVALUATION")
    print("=" * 80 + "\n")
    
    # Setup paths
    script_dir = Path(__file__).parent.parent
    model_path = script_dir / "models" / "best_model.pt"
    
    # Create models
    pre_ssm = PreSSMModule(input_dim=57, d_model=64, d_inner=128, d_conv=4)
    step_ssm = StepSSMModule(d_inner=128, d_state=16)
    post_ssm = PostSSMModule(d_model=64, d_inner=128, output_size=6)
    
    # Load weights from trained model
    load_trained_weights(pre_ssm, step_ssm, post_ssm, str(model_path))
    
    # Set to eval mode
    pre_ssm.eval()
    step_ssm.eval()
    post_ssm.eval()
    
    # Load HAR test data (one level above esp_tflite)
    data_dir = Path(__file__).parent.parent.parent.parent / "UCI HAR Dataset"
    print(f"Loading HAR data from {data_dir}...")
    _, _, test_ds = load_har_data(str(data_dir))
    
    print(f"✓ Data loaded")
    print(f"  Test set size: {len(test_ds)}\n")
    
    # Load reference model
    har_model = HARMamba(input_dim=57, d_model=64, d_state=16, d_conv=4, 
                         expand=2, output_size=6)
    state_dict = torch.load(str(model_path), map_location='cpu', weights_only=True)
    har_model.load_state_dict(state_dict)
    har_model.eval()
    
    # Tracking
    num_classes = 6
    ref_correct = 0
    step_correct = 0
    both_correct = 0
    total_diff = 0.0
    max_diff = 0.0
    
    confusion_ref = torch.zeros(num_classes, num_classes, dtype=torch.long)
    confusion_step = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_ds):
            x = x.unsqueeze(0)  # (1, T, features)
            
            # Reference model
            logits_ref = har_model(x)
            pred_ref = logits_ref.argmax(dim=-1).item()
            
            # Stepwise model
            logits_step = inference_stepwise(pre_ssm, step_ssm, post_ssm, x)
            pred_step = logits_step.argmax(dim=-1).item()
            
            # Track accuracy
            if pred_ref == y:
                ref_correct += 1
            if pred_step == y:
                step_correct += 1
            if pred_ref == y and pred_step == y:
                both_correct += 1
            
            # Track confusion
            confusion_ref[y, pred_ref] += 1
            confusion_step[y, pred_step] += 1
            
            # Track numerical differences
            diff = (logits_ref - logits_step).abs().max().item()
            total_diff += diff
            max_diff = max(max_diff, diff)
    
    # Calculate metrics
    total = len(test_ds)
    ref_acc = ref_correct / total
    step_acc = step_correct / total
    mean_diff = total_diff / total
    
    print(f"\n{'=' * 80}")
    print("TEST SET RESULTS")
    print(f"{'=' * 80}")
    print(f"Total samples: {total}")
    print(f"\nReference Model Accuracy: {ref_acc:.4f} ({ref_correct}/{total})")
    print(f"Stepwise Model Accuracy:  {step_acc:.4f} ({step_correct}/{total})")
    print(f"Both Correct:             {both_correct}/{total}\n")
    
    print(f"Numerical Differences:")
    print(f"  Mean difference: {mean_diff:.8f}")
    print(f"  Max difference:  {max_diff:.8f}\n")
    
    # Per-class accuracy
    print(f"Per-class Accuracy (Reference Model):")
    for c in range(num_classes):
        acc = confusion_ref[c, c].item() / confusion_ref[c].sum().item() if confusion_ref[c].sum() > 0 else 0
        print(f"  Class {c}: {acc:.4f}")
    
    print(f"\nPer-class Accuracy (Stepwise Model):")
    for c in range(num_classes):
        acc = confusion_step[c, c].item() / confusion_step[c].sum().item() if confusion_step[c].sum() > 0 else 0
        print(f"  Class {c}: {acc:.4f}")
    
    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
