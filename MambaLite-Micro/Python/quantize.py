import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.tiny_mamba_har import TinyMambaHAR
from utils.data import load_har_data
from utils.eval_utils import evaluate

def quantize_static(model, calibration_dataloader, max_batches):
    # 1. Set the backend (use 'qnnpack' for ARM/Mobile, 'fbgemm' for x86)
    backend = 'fbgemm' 
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    
    # 2. Prepare the model (inserts observers into your Linear/Conv1d layers)
    # The observers will be recursively inserted into your MambaRaw sub-modules
    torch.quantization.prepare(model, inplace=True)
    
    # 3. Calibration Loop
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(calibration_dataloader):
            if i >= max_batches:
                break
            # FIX: Only pass 'data' to the model, discard the 'target'
            model(data)
            
    # 4. Convert the model (replaces observers with quantized modules)
    torch.quantization.convert(model, inplace=True)
    return model

def test(model, test_loader):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, torch.device("cpu"))

    print(f"[Test] Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Offline Hadamard/KLT prep for MambaLite-Micro")
    parser.add_argument("--checkpoint", type=str, default="./Python/models/mamba_har_model.sd.pt")
    parser.add_argument("--data-dir", type=str, default="../UCI HAR Dataset")
    parser.add_argument("--output", type=str, default="./Python/models/mamba_har_model_quantized.sd.pt")
    parser.add_argument("--channel-width", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--max-batches", type=int, default=8)

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    input_model = TinyMambaHAR(input_dim=57, mamba_channel_width=args.channel_width, num_classes=args.num_classes).cpu().eval()
    state = torch.load(args.checkpoint, map_location="cpu")
    input_model.load_state_dict(state)

    x_train, y_train, _, _ = load_har_data(args.data_dir)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=False)

    model = quantize_static(input_model, loader, max_batches=args.max_batches)
    torch.save(model.state_dict(), args.output)

    _, _, X_test, y_test = load_har_data(args.data_dir)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)
    test(model, test_loader)        

if __name__ == "__main__":
    main()
