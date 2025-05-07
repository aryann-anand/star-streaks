# src/data/annotation.py
import os
import numpy as np
import cv2
from pathlib import Path
import yaml
import json
import random

class DataAnnotator:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set paths and parameters
        self.raw_images_path = self.config['dataset']['raw_images_path']
        self.output_dir = Path('output')
        self.annotations_dir = self.output_dir / 'annotations'
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Class names
        self.class_names = self.config['model']['classes']
    
    def detect_objects(self, image):
        """
        Detect streaks and stars in an image using image processing techniques
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            List of detected objects with class and bounding box
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding to better separate objects from background
        # This is critical for astronomical images with varying background noise
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize detected objects
        detected_objects = []
        
        # Process each contour
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip very small contours (noise)
            if w < 3 or h < 3:
                continue
            
            # Calculate aspect ratio
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            
            # Determine class (streak or star) based on aspect ratio and size
            # Streaks typically have high aspect ratios (elongated)
            # Stars typically have aspect ratios close to 1 (circular)
            class_id = 0 if aspect_ratio > 3.0 else 1  # 0: streak, 1: star
            
            # Add to detected objects
            detected_objects.append({
                'class_id': class_id,
                'box': [x, y, w, h]
            })
        
        return detected_objects
    
    def annotate_image(self, image_path):
        """
        Automatically annotate an image to detect streaks and stars
        
        Args:
            image_path: Path to the image
            
        Returns:
            Annotations in YOLO format
        """
        # Load image
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        
        # Normalize 16-bit to 8-bit
        img_8bit = (img / 65535.0 * 255.0).astype(np.uint8)
        
        # Detect objects
        detected_objects = self.detect_objects(img_8bit)
        
        # Image dimensions
        img_height, img_width = img_8bit.shape[:2]
        
        # Initialize annotations
        annotations = []
        
        # Convert detections to YOLO format
        for obj in detected_objects:
            x, y, w, h = obj['box']
            
            # Calculate center coordinates
            x_center = x + w / 2
            y_center = y + h / 2
            
            # Normalize coordinates (YOLO format)
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = w / img_width
            height_norm = h / img_height
            
            # Add annotation
            annotations.append({
                'class_id': obj['class_id'],
                'x_center': x_center_norm,
                'y_center': y_center_norm,
                'width': width_norm,
                'height': height_norm
            })
        
        return annotations
    
    def annotate_all_images(self):
        """
        Annotate all raw images and save annotations
        
        Returns:
            Dictionary mapping image paths to annotations
        """
        # Get all image paths
        image_paths = list(Path(self.raw_images_path).glob('*.tiff'))
        
        # Initialize results
        annotation_results = {}
        
        # Process each image
        for img_path in image_paths:
            print(f"Annotating {img_path}...")
            
            # Annotate image
            annotations = self.annotate_image(img_path)
            
            # Save annotations to YOLO format file
            yolo_annotations_path = self.annotations_dir / f"{img_path.stem}.txt"
            with open(yolo_annotations_path, 'w') as f:
                for annotation in annotations:
                    f.write(f"{annotation['class_id']} {annotation['x_center']} {annotation['y_center']} {annotation['width']} {annotation['height']}\n")
            
            # Save to results
            annotation_results[str(img_path)] = annotations
        
        # Save all annotations to JSON file
        all_annotations_path = self.annotations_dir / 'all_annotations.json'
        with open(all_annotations_path, 'w') as f:
            json.dump(annotation_results, f, indent=4)
        
        print(f"Annotations saved to {self.annotations_dir}")
        
        return annotation_results
    
    def create_dataset_yaml(self):
        """
        Create YAML file for training YOLOv8
        
        Returns:
            Path to the dataset YAML file
        """
        # Get all image paths
        image_paths = list(Path(self.raw_images_path).glob('*.tiff'))
        
        # Shuffle images
        random.shuffle(image_paths)
        
        # Split into train and validation sets
        train_size = int(len(image_paths) * self.config['dataset']['train_ratio'])
        train_paths = image_paths[:train_size]
        val_paths = image_paths[train_size:]
        
        # Create dataset YAML
        dataset_yaml = {
            'path': str(Path(self.raw_images_path).absolute()),
            'train': [str(p.relative_to(self.raw_images_path)) for p in train_paths],
            'val': [str(p.relative_to(self.raw_images_path)) for p in val_paths],
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        # Save YAML file
        dataset_yaml_path = self.output_dir / 'dataset.yaml'
        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f)
        
        print(f"Dataset YAML saved to {dataset_yaml_path}")
        
        return str(dataset_yaml_path)
