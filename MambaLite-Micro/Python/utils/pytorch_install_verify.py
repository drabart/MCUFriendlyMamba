import torch

print(f'Detected: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU found')