import torch


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device=device), yb.to(device=device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return total_loss / len(dataloader), correct / total