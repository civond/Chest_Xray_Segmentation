import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import segmentation_models_pytorch as smp
import toml
import argparse

from utils.create_transforms import create_transforms, create_inference_transform
from utils.get_loader import get_loader, get_inference_loader
from utils.train import train_fn
from utils.validation import val_fn
from utils.checkpoints import *
from utils.inference import inference

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    # First positional argument: mode
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "inference"],
        help="Operation mode: 'train' or 'inference'"
    )
    
    # Second positional argument: config
    parser.add_argument(
        "config_file", 
        type=str,
        help="Path to TOML configuration file"
    )
    return parser.parse_args()

def main():
    # Load TOML config
    args = parse_args()
    mode = args.mode
    config_path = args.config_file
    config = toml.load(config_path)

    if mode.lower() == "train":
        print("Running training loop...")

        # Access TOML values
        train_dir = config["Paths"]["train_data_dir"]
        valid_dir = config["Paths"]["valid_data_dir"]
        checkpoint_path = config["Paths"]["checkpoint_path"]

        learning_rate = config["Hyperparameters"]["learning_rate"]
        batch_size = config["Hyperparameters"]["batch_size"]
        num_epochs = config["Hyperparameters"]["num_epochs"]
        num_workers = config["Hyperparameters"]["num_workers"]
        image_height = config["Hyperparameters"]["image_height"]
        image_width = config["Hyperparameters"]["image_width"]
        pin_memory = config["Hyperparameters"]["pin_memory"]
        load_model = config["Hyperparameters"]["load_model"]
        train = config["Hyperparameters"]["train"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", device)
        """
        Load model from SMP with efficientnetb3 backbone (input size 300x300)

        model = smp.Unet('efficientnet-b3',
                    in_channels=1, 
                    encoder_weights=None,
                    classes=1,
                    activation=None)
        """
        model = smp.Unet('efficientnet-b3', 
                    encoder_weights='imagenet',
                    classes=1,
                    activation=None)
        model.to(device)

        transforms = create_transforms(image_height, image_width, train)

        # Create train and validation loaders
        train_loader = get_loader(train_dir, 
                                batch_size, 
                                transforms, 
                                num_workers, 
                                train, 
                                pin_memory)
        
        valid_loader = get_loader(valid_dir, 
                                batch_size, 
                                transforms, 
                                num_workers, 
                                train, 
                                pin_memory)
            
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scaler = torch.amp.GradScaler(device="cuda")

        training_loss = []
        training_dice = []
        validation_loss = []
        validation_dice = []

        for epoch in range(num_epochs):
            print(f"\nEpoch: {epoch}")

            # Train
            train_loss, train_dice = train_fn(device, train_loader, model, optimizer, loss_fn, scaler)
            training_loss.append(train_loss)
            training_dice.append(train_dice)
            print(f"Train loss: {train_loss}")
            print(f"Train dice: {train_dice}")
            
            # Validation step
            [valid_loss, valid_dice] = val_fn(device, valid_loader, model, loss_fn)
            validation_loss.append(valid_loss)
            validation_dice.append(valid_dice)
            print(f"Valid loss: {valid_loss}")
            print(f"Valid dice: {valid_dice}")

            # Save checkpoint
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            save_checkpoint(checkpoint)

        df = pd.DataFrame({
            "training_loss": training_loss,
            "training_dice": training_dice,
            "validation_loss": validation_loss,
            "validation_dice": validation_dice
        })
        df.to_csv("training_metrics.csv", index=False)
        
    elif mode.lower() == "inference":
        print("Running inference...")
        test_dir = config["Paths"]["test_data_dir"]
        save_dir = config["Paths"]["save_dir"]
        checkpoint_path = config["Paths"]["checkpoint_path"]
        batch_size = config["Hyperparameters"]["batch_size"]
        
        num_workers = config["Hyperparameters"]["num_workers"]
        image_height = config["Hyperparameters"]["image_height"]
        image_width = config["Hyperparameters"]["image_width"]
        pin_memory = config["Hyperparameters"]["pin_memory"]
        train = config["Hyperparameters"]["train"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Import model and load weights
        model = smp.Unet('efficientnet-b3', 
                    encoder_weights='imagenet',
                    classes=1,
                    activation=None)
        load_checkpoint(checkpoint_path, model)
        model.to(device)
        loss_fn = nn.BCEWithLogitsLoss()

        inference_transform = create_inference_transform(image_height, image_width)
        inference_loader = get_inference_loader(test_dir, 
                                batch_size, 
                                inference_transform, 
                                num_workers, 
                                train, 
                                pin_memory)
        test_loss, test_dice = inference(save_dir, device, inference_loader, model, loss_fn)
        print(f"test loss: {test_loss}")
        print(f"test dice: {test_dice}")
    

if __name__ == "__main__":
    main()