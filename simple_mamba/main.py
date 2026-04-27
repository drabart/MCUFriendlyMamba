import torch
from simple_mamba import MambaBlock


# Define block parameters
dim = 512
hidden_dim = 128
heads = 8
in_channels = 3
out_channels = 3
kernel_size = 3

# Create an instance of MambaBlock
mamba_block = MambaBlock(
    dim, hidden_dim, heads, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, state_dim=64, depth=1, heads=2
)

# Create a sample input tensor
x = torch.randn(1, dim, dim)

# Pass the tensor through the MambaBlock
output = mamba_block(x)
print("Output shape:", output.shape)