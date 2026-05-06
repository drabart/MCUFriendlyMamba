# Originally copied and modified from:
# https://github.com/pytorch/examples/blob/main/mnist/main.py
# under the following license:  BSD-3-Clause license

from __future__ import print_function
import argparse
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from brevitas.export import export_onnx_qcdq
import onnxruntime as ort
import numpy as np
import os
import time
from data import load_har_data, load_mnist_data, load_speechcommands_data
from models import TinyMamba


def train(model, device, train_loader, optimizer, epoch, print_stats=False, log_interval=10, dry_run=False):
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
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if (print_stats):
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
    return correct / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    # Model settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="har",
        help="What dataset to train on ('mnist', 'har' or 'kws') (default: 'har')",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Path to the dataset directory",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mamba-orig",
        help="What model to train ('mamba-orig' or 'mamba-raw') (default: 'mamba-orig')",
    )
    # Training settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--validate-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for validating (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        metavar="LR",
        help="learning rate (default: 0.005)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--quantize", 
        action="store_true", 
        default=False, 
        help="enables model quantization"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        default=True,
        help="For Saving the current Model in ONNX format",
    )
    parser.add_argument(
        "--full-test-onnx",
        action="store_true",
        default=False,
        help="Test the onnx exported model fully against the pytorch model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dataset_type = args.dataset.lower()
    dataset_dir = args.dataset_dir
    print(f"Training with dataset: {dataset_type}")

    train_kwargs = {"batch_size": args.batch_size, "shuffle": True}
    validate_kwargs = {"batch_size": args.batch_size}
    validate_single_kwargs = {"batch_size": 1}
    test_kwargs = {"batch_size": args.validate_batch_size}
    if use_cuda:
        if dataset_type == "kws":
            num_workers = 12
        else:
            num_workers = 1
        cuda_kwargs = {"num_workers": num_workers, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        validate_kwargs.update(cuda_kwargs)
        validate_single_kwargs.update(cuda_kwargs)

    # Improve DataLoader throughput when workers are used
    if train_kwargs.get("num_workers", 0) > 0:
        train_kwargs.setdefault("persistent_workers", True)
        train_kwargs.setdefault("prefetch_factor", 2)
    if validate_kwargs.get("num_workers", 0) > 0:
        validate_kwargs.setdefault("persistent_workers", True)
    if test_kwargs.get("num_workers", 0) > 0:
        test_kwargs.setdefault("persistent_workers", True)


    if dataset_type == "mnist":
        output_size = 10
        input_dim = 28
        d_model = 8
        train_ds, val_ds, test_ds = load_mnist_data(dataset_dir)
    elif dataset_type == "har":
        output_size = 6
        input_dim = 57

        d_model = 64
        d_state = 32
        d_conv = 4
        expand = 2
        train_ds, val_ds, test_ds = load_har_data(dataset_dir)
    elif dataset_type == "kws":
        output_size = 35
        input_dim = 40

        d_model = 64
        d_state = 32
        d_conv = 4
        expand = 2
        train_ds, val_ds, test_ds = load_speechcommands_data(dataset_dir)
    else:
        sys.exit(f"Unknown dataset: {dataset_type}. Choose 'mnist', 'kws' or 'har'")

    train_loader = torch.utils.data.DataLoader(train_ds, **train_kwargs)
    _ = torch.utils.data.DataLoader(test_ds, **test_kwargs)  # Test loader is not used yet
    validate_loader = torch.utils.data.DataLoader(val_ds, **validate_kwargs)
    validate_loader_single = torch.utils.data.DataLoader(val_ds, **validate_single_kwargs)

    model_type = args.model.lower()
    print(f"Training model: {model_type}, hidden dim: {d_model}")

    match model_type:
        case "mamba-orig":
            model = TinyMamba(input_dim=input_dim,d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, output_size=output_size).to(device)
        case "mamba-raw":
            model = TinyMamba(input_dim=input_dim,d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, output_size=output_size).to(device)
        case _:
            sys.exit(
                "Please specify a correct model with the environment variable MODEL"
            )

    model_name = f"{dataset_type}-{model_type}"
    dry_run_name = ""
    model_dir = "./models"
    if args.dry_run:
        dry_run_name = "-dry"
        model_dir = "/tmp"

    onnx_path = os.path.join(model_dir, model_name + dry_run_name + ".onnx")
    pt_path = os.path.join(model_dir, model_name + dry_run_name + ".pt")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            True,
            log_interval=args.log_interval,
            dry_run=args.dry_run,
        )
        test(model, device, validate_loader, print_stats=True)
        scheduler.step()
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} duration: {epoch_time:.2f}s")
        if args.dry_run:
            break

    if args.export_onnx:
        # --- Export to ONNX (QCDQ) ---
        print("Exporting model")
        model.eval()
        inputs, _ = next(iter(validate_loader_single))
        dummy_input = inputs.to(device)

        export_onnx_qcdq(
            model, 
            args=dummy_input, 
            export_path=onnx_path, 
            opset_version=13,
        )

        # --- Verify Accuracy of the Exported ONNX Model ---
        def evaluate_onnx(onnx_path, loader):
            sess = ort.InferenceSession(onnx_path)
            input_name = sess.get_inputs()[0].name
            correct = 0
            for data, target in loader:
                # ensure numpy float32 on CPU
                input_data = data.cpu().numpy().astype(np.float32)
                output = sess.run(None, {input_name: input_data})[0]
                pred = np.argmax(output, axis=1)
                correct += np.sum(pred == target.numpy())
            return 100. * correct / len(loader.dataset)

        print(f"Accuracy AFTER export (ONNX Runtime): {evaluate_onnx(onnx_path, validate_loader_single):.2f}%")

    # for name, module in model.named_modules():
    #     print(f"{name}: {module}")

if __name__ == "__main__":
    main()
