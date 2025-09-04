import segmentation_models_pytorch as smp
import tqdm
import cv2
import numpy as np
import torch.nn as nn

from utils.checkpoints import *
from utils.dice_score import dice_score
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def inference(save_dir, device, loader, model, loss_fn):

    model.eval()  # Make sure model is in evaluation mode
    loop = tqdm.tqdm(loader, desc="Running inference")

    total_loss = 0
    total_dice = 0

    with torch.no_grad():
        for data, labels, filenames in loop:
            data = data.to(device)
            labels = labels.to(device).float() # for BCE

            # Forward pass
            predictions = model(data)
            preds = torch.sigmoid(predictions) > 0.5
            preds = preds.cpu().numpy()

            loss = loss_fn(predictions, labels)
            total_loss += loss.item()
            dice = dice_score(predictions, labels)
            total_dice += dice.item()

            for i in range(preds.shape[0]):
                mask = preds[i, 0, :, :]  # Remove batch & channel dims
                mask = (mask * 255).astype("uint8")

                #save_path = os.path.join(save_dir, filenames[i])
                #cv2.imwrite(save_path, mask)

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    loop.set_postfix(loss=loss.item())

    return avg_loss, avg_dice