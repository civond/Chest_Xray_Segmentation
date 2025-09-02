import torch
import datetime

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("-> Saving checkpoint.")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model):
    print("-> Loading checkpoint.")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    #checkpoint = torch.load(checkpoint_path, map_location="gpu")
    model.load_state_dict(checkpoint["state_dict"])