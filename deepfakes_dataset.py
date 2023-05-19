import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2 
import numpy as np

import uuid
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate

from transforms.albu import IsotropicResize

class DeepFakesDataset(Dataset):
    def __init__(self, images_paths, labels, image_size = 224, mode='train'):
        self.x = images_paths
        self.y = labels
        self.n_samples = len(images_paths)
        self.mode = mode
        self.image_size = image_size

    def create_train_transforms(self, size):
        return Compose([
            OneOf([
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
            GaussNoise(p=0.3),
            #GaussianBlur(blur_limit=3, p=0.05),
            HorizontalFlip(),
            ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]
        )
    def create_val_transforms(self, size):
        return Compose([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ])
    
    def __getitem__(self, index):
        image_path = self.x[index]
        image = cv2.imread(image_path)
        
        if self.mode == 'train':
            transform = self.create_train_transforms(self.image_size)
        else:
            transform = self.create_val_transforms(self.image_size)

        image = transform(image=image)["image"]
        label = self.y[index]

        return torch.tensor(image, dtype=torch.float), torch.tensor(label)

    
    def __len__(self):
        return self.n_samples
