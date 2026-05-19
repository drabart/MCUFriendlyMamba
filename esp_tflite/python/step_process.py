"""Load and process the best model as mamba_step for inference."""
import torch
import os
import json
from models import HARMambaStep

# Load metadata to get model architecture parameters
metadata_path = os.path.join(os.path.dirname(__file__), "models", "metadata.json")
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Create model with the same architecture as the saved state_dict
mamba_step = HARMambaStep(
    input_dim=metadata['input_dim'],
    d_model=metadata['d_model'],
    output_size=metadata['output_size']
)

# Load the state_dict
model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.pt")
state_dict = torch.load(model_path)

# Map state_dict from HARMamba to HARMambaStep
# The SSM core (A_log, D) and projections are compatible between the two
mapped_state_dict = {}
for key, value in state_dict.items():
    # Direct compatible keys (linear layers)
    if key.startswith('linear_in.') or key.startswith('classifier.'):
        mapped_state_dict[key] = value
    # SSM core parameters are compatible
    elif key.startswith('mamba.ssm.'):
        mapped_state_dict[key] = value

mamba_step.load_state_dict(mapped_state_dict, strict=False)

# Set to evaluation mode
mamba_step.eval()

print(f"Model loaded from {model_path}")
print(f"Model structure:\n{mamba_step}")
