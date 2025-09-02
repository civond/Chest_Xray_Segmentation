import torch
from tqdm import tqdm

def val_fn(device, loader, model, loss_fn):
    model.eval()
    loop = tqdm(loader, desc="Validating", leave=False)

    total_loss = 0
    total_iou = 0
    num_batches = 0

    with torch.no_grad():
        for data, labels in loop:
            data = data.to(device)
            labels = labels.to(device)

            # Forward pass
            predictions = model(data)['out']  # Shape: [B, C, H, W]
            loss = loss_fn(predictions, labels)
            total_loss += loss.item()

            # Get predicted class per pixel
            preds = torch.argmax(predictions, dim=1)  # [B, H, W]
            
            # Compute IoU for each batch
            intersection = torch.logical_and(preds == 1, labels == 1).sum(dim=(1,2))
            union = torch.logical_or(preds == 1, labels == 1).sum(dim=(1,2))

            # Avoid division by zero
            batch_iou = torch.mean((intersection.float() + 1e-6) / (union.float() + 1e-6))
            total_iou += batch_iou.item()
            num_batches += 1

    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / num_batches
    return avg_loss, avg_iou