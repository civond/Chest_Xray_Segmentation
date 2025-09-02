

import torch
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt

print('hello')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet('efficientnet-b3', 
                 encoder_weights='imagenet',
                 classes=1,
                 activation=None)
    
model.to(device)
print(model)

"""
import torch
import torch.optim as optim
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import cv2

preprocessing_fn = get_preprocessing_fn("efficientnet-b3", pretrained="imagenet")

print(preprocessing_fn)

path = "./data/images/COVID-1.png"
img = cv2.imread(path)
print(img.shape)"""