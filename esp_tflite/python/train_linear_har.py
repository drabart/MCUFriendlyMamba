"""Training script for linear HAR model."""
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import json
import os

from data import load_har_data
from models import HARLinear, HARMamba

def train(model, device, train_loader, optimizer, epoch, print_stats=False, log_interval=10, dry_run=False):
    """Train for one epoch."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if print_stats and batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if dry_run:
                break


def test(model, device, test_loader, print_stats=False):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    if print_stats:
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * accuracy,
            )
        )
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Train linear HAR model")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Path to UCI HAR dataset",
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=64,
        help="Model dimension",
    )
    parser.add_argument(
        "--bit-width",
        type=int,
        default=8,
        help="Quantization bit width",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Output directory for trained model",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading HAR dataset...")
    train_ds, val_ds, test_ds = load_har_data(args.dataset_dir)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Create model
    print("Creating model...")
    # model = HARLinear(
    #     input_dim=57,
    #     d_model=args.d_model,
    #     output_size=6,
    # ).to(device)
    model = HARMamba(
        input_dim=57,
        d_model=args.d_model,
        output_size=6,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    print("Starting training...")
    best_accuracy = 0
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, print_stats=True)
        val_accuracy = test(model, device, val_loader, print_stats=True)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with accuracy: {best_accuracy:.4f}")
        
        scheduler.step()

    # Test on final test set
    print("\nFinal test set evaluation:")
    model.load_state_dict(torch.load(best_model_path))
    test_accuracy = test(model, device, test_loader, print_stats=True)

    # Save training metadata
    metadata = {
        "dataset": "UCI HAR",
        "input_shape": [1, 10, 57],
        "output_shape": [1, 6],
        "d_model": args.d_model,
        "bit_width": args.bit_width,
        "epochs_trained": args.epochs,
        "best_val_accuracy": best_accuracy,
        "test_accuracy": test_accuracy,
    }
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved to: {best_model_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
