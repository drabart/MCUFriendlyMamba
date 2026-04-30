import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.data import load_har_data
from utils.eval_utils import evaluate
from models.tiny_mamba_har import build_model

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TinyMambaHAR checkpoint on the HAR test split")
    parser.add_argument("--checkpoint", type=str, default="./Python/mamba_har_model_offline_rotated.pt")
    parser.add_argument("--data-dir", type=str, default="../UCI HAR Dataset")
    parser.add_argument("--input-dim", type=int, default=57)
    parser.add_argument("--channel-width", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, X_test, y_test = load_har_data(args.data_dir)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size)

    model = build_model(
        input_dim=args.input_dim,
        mamba_channel_width=args.channel_width,
        num_classes=args.num_classes,
        device=device,
    )
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("Loaded checkpoint:", args.checkpoint)
    print(f"[Test] Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")


if __name__ == "__main__":
    main()