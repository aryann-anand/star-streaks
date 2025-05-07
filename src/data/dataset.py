# src/data/dataset.py
import os
import numpy as np
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

class SpaceObjectDataset(Dataset):
    def __init__(self, config_path, train=True, transform=None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set paths and parameters
        self.raw_images_path = self.config['dataset']['raw_images_path']
        self.img_size = self.config['dataset']['img_size']
        self.transform = transform
        
        # Get image paths
        self.image_paths = list(Path(self.raw_images_path).glob('*.tiff'))
        
        # Split into train and test sets
        train_size = int(len(self.image_paths) * self.config['dataset']['train_ratio'])
        if train:
            self.image_paths = self.image_paths[:train_size]
        else:
            self.image_paths = self.image_paths[train_size:]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        
        # Normalize 16-bit to 8-bit
        img = (img / 65535.0 * 255.0).astype(np.uint8)
        
        # Resize image to required size
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Apply transformations if specified
        if self.transform:
            img = self.transform(img)
        
        # For now, return only the image (labels will be added later)
        return img, str(img_path)

def create_dataloader(config_path, train=True, batch_size=None, transform=None):
    # Create dataset
    dataset = SpaceObjectDataset(config_path, train, transform)
    
    # Load configuration if batch_size is not provided
    if batch_size is None:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        batch_size = config['dataset']['batch_size']
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader
