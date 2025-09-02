import albumentations as A
from albumentations.pytorch import ToTensorV2
#from utils.get_loader import get_loader


# Create transforms object
def create_transforms(IMAGE_HEIGHT, IMAGE_WIDTH, TRAIN):
    if TRAIN == True:
        transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=35, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})
        
    if TRAIN == False:
        transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})
        
    return transform