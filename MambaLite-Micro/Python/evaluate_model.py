import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.data import load_har_data
from utils.eval_utils import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TinyMambaHAR model on the HAR test split")
    parser.add_argument("--model", type=str, default="./Python/models/mamba_har_model.model.pt")
    parser.add_argument("--data-dir", type=str, default="../UCI HAR Dataset")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cpu")

    _, _, X_test, y_test = load_har_data(args.data_dir)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size)

    model = torch.load(args.model, map_location=device, weights_only=False)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("Loaded model:", args.model)
    print(f"[Test] Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")

if __name__ == "__main__":
    main()