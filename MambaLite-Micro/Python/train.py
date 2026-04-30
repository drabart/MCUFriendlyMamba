#
# train.py
# Sample training script for HAR task.
#
# Copyright (c) 2025 MambaLite-Micro Authors
# Licensed under the MIT License.

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from utils.data import load_har_data
from utils.eval_utils import evaluate
from models.tiny_mamba_har import build_model
from utils.train_utils import train

def main():
    data_dir = r'../UCI HAR Dataset'
    batch_size = 64
    epochs = 20
    lr = 1e-3
    # hidden_dim = 128
    model_save_path = "./Python/models/mamba_har_model.sd.pt"
    debug_model_save_path = "./Python/models/mamba_har_model.model.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, y_train, X_test, y_test = load_har_data(data_dir)
    val_len = int(0.2 * len(X_train))
    train_len = len(X_train) - val_len
    train_ds, val_ds = random_split(TensorDataset(X_train, y_train), [train_len, val_len])
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = build_model(input_dim=57, mamba_channel_width=64, num_classes=6, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            torch.save(model, debug_model_save_path)
            print(f"✅ Saved new best model to {model_save_path}")
            print(f"✅ Saved debug model to {debug_model_save_path}")

    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[Final Test] Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")

if __name__ == "__main__":
    main()
