"""Stepwise inference: decompose trained model into 3 separate models.

Loads best_model_har.pt / best_model_kws.pt and extracts weights into:
1. PreSSM: Input projection + state/gate + conv
2. StepSSM: Single timestep SSM with persistent state
3. PostSSM: Gate multiply + out_proj + pool + classify
Supports both HAR and KWS datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import json
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import HARMamba
from data import load_har_data, load_speechcommands_data


def load_model_metadata(model_path, dataset):
    """Load metadata_{dataset}.json from the model directory to get architecture params."""
    model_dir = Path(model_path).parent
    metadata_file = model_dir / f"metadata_{dataset}.json"
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"metadata_{dataset}.json not found in {model_dir}")
    
    with open(metadata_file, 'r') as f:
        return json.load(f)


class PreSSMModule(nn.Module):
    """Pre-SSM: input projection through conv1d."""
    def __init__(self, input_dim, d_model, d_inner, d_conv):
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
    """Single-timestep SSM: takes hidden state as input, returns output and updated state."""
    def __init__(self, d_inner, d_state, dt_size):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        
        self.x_proj_B = nn.Linear(d_inner, d_state, bias=False)
        self.x_proj_C = nn.Linear(d_inner, d_state, bias=False)
        self.x_proj_dt = nn.Linear(d_inner, dt_size, bias=False)
        self.dt_proj = nn.Linear(dt_size, d_inner, bias=False)
        
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.A  = (-torch.exp(self.A_log.data)).unsqueeze(0)
        self.D = nn.Parameter(torch.ones(d_inner))
    
    def forward(self, x_t, hidden_state):
        """Process single timestep.
        
        Args:
            x_t: (B, d_inner) input for this timestep
            hidden_state: (B, d_inner, d_state) current hidden state
        
        Returns:
            y_t: (B, d_inner) output
            hidden_state_new: (B, d_inner, d_state) updated hidden state
        """
        B = self.x_proj_B(x_t)  # (B, d_state)
        C = self.x_proj_C(x_t)  # (B, d_state)
        dt = self.x_proj_dt(x_t)
        dt = self.dt_proj(dt)
        dt = F.relu(dt)
        
        # Reshape for broadcasting
        x_expanded = x_t.unsqueeze(-1)  # (B, d_inner, 1)
        dt_expanded = dt.unsqueeze(-1)  # (B, d_inner, 1)
        B_expanded = B.unsqueeze(1)  # (B, 1, d_state)
        C_expanded = C.unsqueeze(1)  # (B, 1, d_state)
        
        A_bar = torch.exp(dt_expanded * self.A)  # (B, d_inner, d_state)
        B_bar = dt_expanded * B_expanded  # (B, d_inner, d_state)
        
        hidden_state_new = A_bar * hidden_state + B_bar * x_expanded
        
        y_t = torch.sum(hidden_state_new * C_expanded, dim=-1)  # (B, d_inner)
        y_t = y_t + self.D.view(1, -1) * x_t
        
        return y_t, hidden_state_new


class PostSSMModule(nn.Module):
    """Post-SSM: gate, out_proj, pool, classify."""
    def __init__(self, d_model, d_inner, output_size):
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
    
    # Update pre-computed A buffer after loading A_log
    step_ssm.A.copy_(-torch.exp(step_ssm.A_log.data))
    
    # Post-SSM weights
    post_ssm.out_proj.weight.data = state_dict['mamba.out_proj.weight']
    post_ssm.classifier.weight.data = state_dict['classifier.weight']


def inference_stepwise(pre_ssm, step_ssm, post_ssm, x):
    """Run inference using 3 separate models."""
    with torch.no_grad():
        # Step 1: Pre-SSM
        state, gate = pre_ssm(x)  # (B, T, d_inner)
        
        # Step 2: Step-by-step SSM
        batch_size, seq_len = x.shape[0], x.shape[1]
        hidden_state = torch.zeros(batch_size, step_ssm.d_inner, step_ssm.d_state,
                                   device=x.device, dtype=x.dtype)
        
        ys = []
        for t in range(seq_len):
            y_t, hidden_state = step_ssm(state[:, t, :], hidden_state)
            ys.append(y_t)
        
        y_ssm = torch.stack(ys, dim=1)  # (B, T, d_inner)
        
        # Step 3: Post-SSM
        logits = post_ssm(y_ssm, gate)
        
        return logits


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load trained model and extract weights into 3 stepwise models for inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained PyTorch model (default: ../models/best_model_har.pt or best_model_kws.pt)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["har", "kws"],
        default="har",
        help="Dataset type (har or kws, default: har)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to dataset root directory",
    )
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent.parent
    
    # Determine model path
    if args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = script_dir / model_path
    else:
        model_path = script_dir / "models" / f"best_model_{args.dataset}.pt"
    
    # Determine dataset directory
    if args.dataset_dir:
        data_dir = Path(args.dataset_dir)
    else:
        # Default paths
        if args.dataset == "har":
            data_dir = Path(__file__).parent.parent.parent.parent / "UCI HAR Dataset"
        else:  # kws
            data_dir = Path(__file__).parent.parent.parent.parent / "SpeechCommands"
    
    print(f"Using model: {model_path}")
    print(f"Using dataset: {args.dataset} at {data_dir}")
    
    # Load metadata to get model architecture
    metadata = load_model_metadata(str(model_path), args.dataset)
    input_dim = metadata.get('input_dim')
    d_model = metadata.get('d_model')
    output_size = metadata.get('output_size')
    seq_len = metadata.get('input_shape')[1]  # Sequence length
    d_inner = d_model * 2  # Standard expansion
    d_state = 16
    d_conv = 4
    
    print(f"Model architecture: input_dim={input_dim}, d_model={d_model}, output_size={output_size}, seq_len={seq_len}")
    
    # Create models with metadata-based dimensions
    pre_ssm = PreSSMModule(input_dim=input_dim, d_model=d_model, d_inner=d_inner, d_conv=d_conv)
    step_ssm = StepSSMModule(d_inner=d_inner, d_state=d_state)
    post_ssm = PostSSMModule(d_model=d_model, d_inner=d_inner, output_size=output_size)
    
    # Load weights from trained model
    load_trained_weights(pre_ssm, step_ssm, post_ssm, str(model_path))
    
    # Set to eval mode
    pre_ssm.eval()
    step_ssm.eval()
    post_ssm.eval()
    
    # Load dataset - handle both HAR and KWS
    if args.dataset == "har":
        _, _, test_ds = load_har_data(str(data_dir))
    else:  # kws
        _, _, test_ds = load_speechcommands_data(str(data_dir), "../models/audio_preprocessor_float.tflite")
    
    # Load reference model with correct dimensions
    har_model = HARMamba(input_dim=input_dim, d_model=d_model, d_state=d_state, d_conv=d_conv, 
                         expand=2, output_size=output_size)
    state_dict = torch.load(str(model_path), map_location='cpu', weights_only=True)
    har_model.load_state_dict(state_dict)
    har_model.eval()
    
    # Tracking
    num_classes = output_size
    ref_correct = 0
    step_correct = 0
    both_correct = 0
    total_diff = 0.0
    max_diff = 0.0
    
    confusion_ref = torch.zeros(num_classes, num_classes, dtype=torch.long)
    confusion_step = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    print(f"Evaluating {len(test_ds)} samples...")
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
            
            if (idx + 1) % 500 == 0:
                print(f"  {idx + 1}/{len(test_ds)}")
    
    # Calculate metrics
    total = len(test_ds)
    ref_acc = ref_correct / total
    step_acc = step_correct / total
    mean_diff = total_diff / total
    
    print(f"\nResults: {total} samples")
    print(f"Reference Accuracy:  {ref_acc:.4f} ({ref_correct}/{total})")
    print(f"Stepwise Accuracy:   {step_acc:.4f} ({step_correct}/{total})")
    print(f"Max output diff:     {max_diff:.8f}")


if __name__ == "__main__":
    main()
