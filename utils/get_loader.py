from utils.dataset import ImageDataset
from torch.utils.data import DataLoader

def get_loader(data_dir,
               batch_size,
               transform,
               num_workers=4,
               train=True,
               pin_memory=True):
    
    dataset = ImageDataset(
        data_dir, 
        train=train,
        transform=transform
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    return loader