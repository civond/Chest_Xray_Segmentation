import torch
from tqdm import tqdm
from utils.dice_score import dice_score

def train_fn(device, loader, model, optimizer, loss_fn, scaler):
    model.train()  # make sure model is in training mode
    loop = tqdm(loader)
    
    # Track loss and dice score
    total_loss = 0
    total_dice = 0

    # Main loop
    for batch_idx, (data, labels, filenames) in enumerate(loop):
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