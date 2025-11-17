"""
Image preprocessing and augmentation transforms.
"""

from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image


def get_train_transform(image_size=224):
    """Augmentation pipeline for contrastive training."""
    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.6, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform(image_size=224):
    """Simple resize and normalize for validation."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class AlbumentationsWrapper:
    """Wraps albumentations transforms for use with PIL images."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        result = self.transform(image=image)
        return result["image"]
