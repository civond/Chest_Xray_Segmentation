from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torch

class ImageDataset(Dataset):
    def __init__(self, data_dir, train=True, transform = None):
        """
        data_dir: ./data
        This function expects subfolders:
            ./data/images/
            ./data/masks/
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.transform = transform
        
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".png")])
        self.masks = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(".png")])
        assert len(self.images) == len(self.masks), "Number of images and masks must match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load image
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load mask
        mask_name = self.masks[index]
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = mask // 255

        if image.shape[:2] != mask.shape[:2]:
                mask = np.array(Image.fromarray(mask).resize((image.shape[1], image.shape[0])))

        # Apply transform
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.unsqueeze(0) # This will transform mask dimmensions from [height, width] --> [1, height, width]

        return image, mask, mask_name
