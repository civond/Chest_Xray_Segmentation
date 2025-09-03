import torch
from tqdm import tqdm
from utils.dice_score import dice_score

def val_fn(device, loader, model, loss_fn):
    model.eval()
    loop = tqdm(loader, desc="Validating", leave=False)

    # Track loss and dice score
    total_loss = 0
    total_dice = 0

    with torch.no_grad():
        for data, labels, filenames in loop:
            data = data.to(device)
            labels = labels.to(device).float() # for BCE

            # Forward pass
            predictions = model(data)
            loss = loss_fn(predictions, labels)
            total_loss += loss.item()

            dice = dice_score(predictions, labels)
            total_dice += dice.item()
        
    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader) 
    # update tqdm loop
    loop.set_postfix(loss=loss.item())

    return avg_loss, avg_dice