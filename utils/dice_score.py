import torch

def dice_score(preds, targets, threshold=0.5, eps=1e-6):
    """
    Calculate Dice score for binary segmentation.
    preds: raw model outputs (logits)
    targets: ground truth masks (0 or 1)
    """
    # Apply sigmoid to logits
    preds = torch.sigmoid(preds)
    # Binarize predictions
    preds = (preds > threshold).float()
    
    # Flatten
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    dice = (2. * intersection + eps) / (preds.sum() + targets.sum() + eps)
    
    return dice