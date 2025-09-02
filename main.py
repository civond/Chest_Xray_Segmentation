import torch
import torch.optim as optim
import torch.nn as nn
import segmentation_models_pytorch as smp

from utils.create_transforms import create_transforms
from utils.get_loader import get_loader
from utils.train import train_fn
from utils.validation import val_fn
from utils.checkpoints import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_WORKERS = 4
IMAGE_HEIGHT = 300  
IMAGE_WIDTH = 300
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN = True

TRAIN_DATA_DIR = "./data_train/"
VALID_DATA_DIR = "./data_valid/"

CHECKPOINT_PATH = "my_checkpoint.pth.tar"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load model from SMP with efficientnetb3 backbone (input size 300x300)
    model = smp.Unet('efficientnet-b3', 
                 encoder_weights='imagenet',
                 classes=1,
                 activation=None)
    model.to(device)

    transforms = create_transforms(IMAGE_HEIGHT, IMAGE_WIDTH, TRAIN)

    # Create train and validation loaders
    train_loader = get_loader(TRAIN_DATA_DIR, 
                              BATCH_SIZE, 
                              transforms, 
                              NUM_WORKERS, 
                              TRAIN, 
                              PIN_MEMORY)
    
    valid_loader = get_loader(VALID_DATA_DIR, 
                              BATCH_SIZE, 
                              transforms, 
                              NUM_WORKERS, 
                              TRAIN, 
                              PIN_MEMORY)
        
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(device="cuda")

    if TRAIN == True:
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch: {epoch}")

            # Train
            train_loss, train_dice = train_fn(device, train_loader, model, optimizer, loss_fn, scaler)
            print(f"Train loss: {train_loss}")
            print(f"Train dice: {train_dice}")
            
            # Validation step
            [valid_loss, valid_dice] = val_fn(device, valid_loader, model, loss_fn)
            print(f"Valid loss: {valid_loss}")
            print(f"Valid dice: {valid_dice}")

            # Save checkpoint
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            save_checkpoint(checkpoint)
            
    elif TRAIN == False:
        print("Interence")
        load_checkpoint(CHECKPOINT_PATH, model)


if __name__ == "__main__":
    main()