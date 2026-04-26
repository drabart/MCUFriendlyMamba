#
# train.py
# Sample training script for HAR task.
#
# Copyright (c) 2025 MambaLite-Micro Authors
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
import numpy as np
from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm.modules.mamba2_simple import Mamba2Simple

# class TinyMambaHAR(nn.Module):
#     def __init__(self, input_dim, mamba_channel_width, num_classes=6):
#         super().__init__()
#         self.linear_in = nn.Linear(input_dim, mamba_channel_width)
#         self.conv = nn.Sequential(
#             nn.Conv1d(input_dim, mamba_channel_width, 4, padding='same'),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Conv1d(mamba_channel_width, mamba_channel_width, 4, padding='same'),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1) # Squashes T=10 down to 1
#         )
#         # self.mamba = Mamba(d_model=mamba_channel_width, d_state=1, expand=1)
#         # self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.classifier = nn.Linear(mamba_channel_width, num_classes)

#     def forward(self, x):
#         x = self.linear_in(x)      # [B, T, H]
#         x = self.conv(x)           # [B, H, 1]
#         # x = self.mamba(x)         # [B, T, H]
#         # x, _ = self.rnn(x)
#         x = x.transpose(1, 2)      # [B, H, T]
#         x = self.pool(x).squeeze(-1)
#         return self.classifier(x)

class TinyMambaHAR(nn.Module):
    def __init__(self, input_dim, mamba_channel_width, num_classes=6):
        super().__init__()
        # input_dim = 57, mamba_channel_width = 57 (based on your setup)
        
        self.conv = nn.Sequential(
            # Input dim is 57. Kernel is 4.
            nn.Conv1d(input_dim, mamba_channel_width, 4, padding='same'),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(mamba_channel_width, mamba_channel_width, 4, padding='same'),
            nn.ReLU()
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(mamba_channel_width, num_classes)

    def forward(self, x):
        # 1. x arrives as [Batch, 10, 57] 
        # 2. Swap dims for Conv1d: [Batch, 57, 10]
        x = x.transpose(1, 2) 
        
        # 3. Run through Convolutions
        x = self.conv(x)           # Output: [Batch, 57, 10]
        
        # 4. Global Average Pool: [Batch, 57, 10] -> [Batch, 57, 1]
        x = self.pool(x)
        
        # 5. Remove the last dim and classify
        x = x.squeeze(-1)          # Output: [Batch, 57]
        return self.classifier(x)

def load_har_data(data_dir):
    def load_txt(file_path):
        return pd.read_csv(file_path, sep=r'\s+', engine='python', header=None).values

    X_train = load_txt(os.path.join(data_dir, 'train', 'X_train.txt'))
    y_train = load_txt(os.path.join(data_dir, 'train', 'y_train.txt')).squeeze() - 1
    X_test = load_txt(os.path.join(data_dir, 'test', 'X_test.txt'))
    y_test = load_txt(os.path.join(data_dir, 'test', 'y_test.txt')).squeeze() - 1

    def prepare(X):
        X = F.pad(torch.tensor(X, dtype=torch.float32), (0, 570 - 561))
        return X.view(-1, 10, 57)

    X_train = prepare(X_train)
    X_test = prepare(X_test)

    # print(X_train.shape, type(X_train), X_train.dtype)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, y_train, X_test, y_test

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in dataloader:
        xb, yb = xb.to(device='cuda'), yb.to(device='cuda')
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device='cuda'), yb.to(device='cuda')
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return total_loss / len(dataloader), correct / total

def main():
    data_dir = r'UCI HAR Dataset'
    batch_size = 64
    epochs = 20
    lr = 1e-3
    # hidden_dim = 128
    model_save_path = "MambaLite-Micro/Python/mamba_har_model.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, y_train, X_test, y_test = load_har_data(data_dir)
    val_len = int(0.2 * len(X_train))
    train_len = len(X_train) - val_len
    train_ds, val_ds = random_split(TensorDataset(X_train, y_train), [train_len, val_len])
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = TinyMambaHAR(input_dim=57, mamba_channel_width=64, num_classes=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Saved new best model to {model_save_path}")

    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"[Final Test] Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")

if __name__ == "__main__":
    main()
