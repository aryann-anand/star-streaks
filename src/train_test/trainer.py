# src/train_test/trainer.py
import os
import yaml
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO

class Trainer:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set paths and parameters
        self.raw_images_path = self.config['dataset']['raw_images_path']
        self.img_size = self.config['dataset']['img_size']
        self.batch_size = self.config['dataset']['batch_size']
        self.epochs = self.config['training']['epochs']
        self.device = self.config['training']['device']
        
        # Create output directories
        self.output_dir = Path('output')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self, dataset_yaml_path):
        """
        Train the YOLOv8 model
        
        Args:
            dataset_yaml_path: Path to dataset YAML file
            
        Returns:
            Training results and training time
        """
        # Create YOLO model
        model = YOLO(f"{self.config['model']['name']}.pt")
        
        # Start timer
        start_time = time.time()
        
        # Train the model
        results = model.train(
            data=dataset_yaml_path,
            epochs=self.epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            device=self.device,
            workers=4,
            val=True,
            verbose=True,
            project=str(self.output_dir),
            name='train'
        )
        
        # Calculate training time
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        return results, training_time, model
