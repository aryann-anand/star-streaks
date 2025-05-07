# src/train_test/evaluator.py
import os
import yaml
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import json
from ultralytics import YOLO

class Evaluator:
    def __init__(self, config_path, model=None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set paths and parameters
        self.raw_images_path = self.config['dataset']['raw_images_path']
        self.img_size = self.config['dataset']['img_size']
        self.batch_size = self.config['dataset']['batch_size']
        self.device = self.config['training']['device']
        self.class_names = self.config['model']['classes']
        
        # Create output directories
        self.output_dir = Path('output')
        self.results_dir = self.output_dir / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set model
        self.model = model if model else None
    
    def load_model(self, model_path):
        """
        Load a trained model
        
        Args:
            model_path: Path to the trained model
        """
        self.model = YOLO(model_path)
        self.model.conf = self.config['model']['confidence_threshold']
        self.model.iou = self.config['model']['iou_threshold']
    
    def evaluate(self, dataset_yaml_path):
        """
        Evaluate the YOLOv8 model
        
        Args:
            dataset_yaml_path: Path to dataset YAML file
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Start timer
        start_time = time.time()
        
        # Evaluate the model
        results = self.model.val(
            data=dataset_yaml_path,
            imgsz=self.img_size,
            batch=self.batch_size,
            device=self.device,
            verbose=True,
            project=str(self.output_dir),
            name='val'
        )
        
        # Calculate evaluation time
        evaluation_time = time.time() - start_time
        
        # Extract metrics
        metrics = {
            'map50': float(results.box.map50),
            'map75': float(results.box.map75),
            'map': float(results.box.map),
            'precision': float(results.box.precision),
            'recall': float(results.box.recall),
            'evaluation_time': evaluation_time
        }
        
        # Save metrics
        metrics_path = self.results_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Evaluation completed in {evaluation_time:.2f} seconds")
        print(f"Metrics saved to {metrics_path}")
        
        return metrics
    
    def predict_on_images(self, image_paths):
        """
        Run prediction on a list of images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        results = []
        total_time = 0
        
        for img_path in image_paths:
            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            
            # Normalize 16-bit to 8-bit
            img_8bit = (img / 65535.0 * 255.0).astype(np.uint8)
            
            # Start timer
            start_time = time.time()
            
            # Perform prediction
            yolo_results = self.model(img_8bit, size=self.img_size, verbose=False)
            
            # Calculate computation time
            computation_time = time.time() - start_time
            total_time += computation_time
            
            # Process results
            detections = []
            centroids = []
            
            for result in yolo_results:
                # Get boxes
                if not result.boxes:
                    continue
                    
                # Get boxes, classes, and scores
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                
                for box, cls, score in zip(boxes, classes, scores):
                    x1, y1, x2, y2 = box
                    class_id = int(cls)
                    class_name = self.class_names[class_id]
                    detection = {
                        'box': [float(x1), float(y1), float(x2), float(y2)],
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': float(score)
                    }
                    detections.append(detection)
                    
                    # Calculate centroid
                    centroid_x = (x1 + x2) / 2
                    centroid_y = (y1 + y2) / 2
                    centroids.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'centroid': [float(centroid_x), float(centroid_y)]
                    })
            
            result = {
                'image_path': str(img_path),
                'detections': detections,
                'centroids': centroids,
                'computation_time': computation_time
            }
            results.append(result)
        
        # Calculate average computation time
        avg_time = total_time / len(image_paths) if image_paths else 0
        fps = len(image_paths) / total_time if total_time > 0 else 0
        
        # Save results
        summary = {
            'num_images': len(image_paths),
            'total_computation_time': total_time,
            'avg_computation_time': avg_time,
            'fps': fps
        }
        
        # Save summary
        summary_path = self.results_dir / 'prediction_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"Prediction completed for {len(image_paths)} images")
        print(f"Average computation time: {avg_time:.4f} seconds per image")
        print(f"Frames per second (FPS): {fps:.2f}")
        
        return results, summary
