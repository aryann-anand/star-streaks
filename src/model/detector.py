# src/model/detector.py
import torch
import yaml
from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path
import time
import supervision as sv

class SpaceObjectDetector:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model
        self.model_name = self.config['model']['name']
        self.confidence_threshold = self.config['model']['confidence_threshold']
        self.iou_threshold = self.config['model']['iou_threshold']
        self.img_size = self.config['dataset']['img_size']
        self.device = self.config['training']['device']
        
        # Class names
        self.class_names = self.config['model']['classes']
        
        # Initialize YOLOv8 model
        self.model = YOLO(f"{self.model_name}.pt")
        
        # Set model parameters
        self.model.conf = self.confidence_threshold
        self.model.iou = self.iou_threshold
        
    def train(self, dataset_yaml_path, epochs=None):
        """
        Train the YOLOv8 model
        
        Args:
            dataset_yaml_path: Path to dataset YAML file
            epochs: Number of epochs (if None, use config value)
        
        Returns:
            Training results
        """
        if epochs is None:
            epochs = self.config['training']['epochs']
        
        # Set training parameters
        results = self.model.train(
            data=dataset_yaml_path,
            epochs=epochs,
            imgsz=self.img_size,
            batch=self.config['dataset']['batch_size'],
            device=self.device,
            workers=4,
            val=True,
            verbose=True
        )
        
        return results
    
    def evaluate(self, dataset_yaml_path):
        """
        Evaluate the YOLOv8 model
        
        Args:
            dataset_yaml_path: Path to dataset YAML file
        
        Returns:
            Evaluation results
        """
        results = self.model.val(
            data=dataset_yaml_path,
            imgsz=self.img_size,
            batch=self.config['dataset']['batch_size'],
            device=self.device,
            verbose=True
        )
        
        return results
    
    def predict(self, image, return_centroids=True):
        """
        Perform object detection on an image
        
        Args:
            image: Input image (numpy array)
            return_centroids: Whether to return centroids
        
        Returns:
            Detections and optionally centroids
        """
        # Start timer
        start_time = time.time()
        
        # Perform prediction
        results = self.model(image, size=self.img_size, verbose=False)
        
        # Calculate computation time
        computation_time = time.time() - start_time
        
        # Process results
        detections = []
        centroids = []
        
        for result in results:
            # Convert to supervision Detections for easier processing
            boxes = sv.Detections.from_ultralytics(result)
            
            # Process each detection
            for i, (xyxy, _, confidence, class_id, _) in enumerate(boxes):
                x1, y1, x2, y2 = xyxy
                class_name = self.class_names[int(class_id)]
                
                detection = {
                    'box': [float(x1), float(y1), float(x2), float(y2)],
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'confidence': float(confidence)
                }
                detections.append(detection)
                
                if return_centroids:
                    # Calculate centroid
                    centroid_x = (x1 + x2) / 2
                    centroid_y = (y1 + y2) / 2
                    centroids.append({
                        'class_id': int(class_id),
                        'class_name': class_name,
                        'centroid': [float(centroid_x), float(centroid_y)]
                    })
        
        return {
            'detections': detections,
            'centroids': centroids if return_centroids else None,
            'computation_time': computation_time
        }
    
    def save_model(self, save_path):
        """
        Save the trained model
        
        Args:
            save_path: Path to save the model
        """
        self.model.export(format='onnx', save_dir=save_path)
    
    def load_model(self, model_path):
        """
        Load a trained model
        
        Args:
            model_path: Path to the trained model
        """
        self.model = YOLO(model_path)
        self.model.conf = self.confidence_threshold
        self.model.iou = self.iou_threshold
