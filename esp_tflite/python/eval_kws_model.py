"""Quick evaluation of best_model_kws.pt on validation and test sets."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

from data import load_speechcommands_data
from models import HARMamba


def test(model, device, test_loader, dataset_name=""):
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

    print(
        f"{dataset_name}: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * accuracy:.2f}%)"
    )
    return accuracy


def main():
    # Configuration
    dataset_dir = "/home/drabart/Documents/ResearchProject/SpeechCommands"
    model_path = "./models/best_model_kws.pt"
    batch_size = 32

    # Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading SpeechCommands (KWS) dataset from {dataset_dir}...")
    train_ds, val_ds, test_ds = load_speechcommands_data(dataset_dir)
    
    # Get input/output dimensions from first sample
    sample_x, _ = train_ds[0]
    input_dim = int(sample_x.shape[1])
    output_size = 35

    # Create data loaders
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Load model
    print(f"Loading model from {model_path}...")
    model = HARMamba(
        input_dim=input_dim,
        d_model=64,
        output_size=output_size,
    ).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✓ Model loaded successfully")
    else:
        print(f"✗ Model file not found: {model_path}")
        return

    # Evaluate
    print("\n" + "="*60)
    print("Evaluating best_model_kws.pt")
    print("="*60)
    val_acc = test(model, device, val_loader, "Validation set")
    test_acc = test(model, device, test_loader, "Test set     ")
    
    print("="*60)
    print(f"\nSummary:")
    print(f"  Validation accuracy: {100.0 * val_acc:.2f}%")
    print(f"  Test accuracy:       {100.0 * test_acc:.2f}%")


if __name__ == "__main__":
    main()
