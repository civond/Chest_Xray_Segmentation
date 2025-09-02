import torch
from tqdm import tqdm

def dice_score(preds, targets, threshold=0.5, eps=1e-6):
    """
    Calculate Dice score for binary segmentation.
    preds: raw model outputs (logits)
    targets: ground truth masks (0 or 1)
    """
    # Apply sigmoid to logits
    preds = torch.sigmoid(preds)
    # Binarize predictions
    preds = (preds > threshold).float()
    
    # Flatten
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    dice = (2. * intersection + eps) / (preds.sum() + targets.sum() + eps)
    
    return dice

def train_fn(device, loader, model, optimizer, loss_fn, scaler):
    model.train()  # make sure model is in training mode
    loop = tqdm(loader)
    
    # Track loss and dice score
    total_loss = 0
    total_dice = 0

    # Main loop
    for batch_idx, (data, labels) in enumerate(loop):
        data = data.to(device)
        labels = labels.to(device).float() # for BCE


        # forward pass
        with torch.amp.autocast('cuda'):
            #print(type(model(data))) <--- sanity check
            predictions = model(data)  # <-- extract tensor
            loss = loss_fn(predictions, labels)

        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        dice = dice_score(predictions, labels)
        total_loss += loss.item()
        total_dice += dice.item()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader) 

    return avg_loss, avg_dice