# main.py
import os
import yaml
from pathlib import Path
import torch
import time
import numpy as np
import cv2

# Import project modules
from src.data.annotation import DataAnnotator
from src.model.detector import SpaceObjectDetector
from src.train_test.trainer import Trainer
from src.train_test.evaluator import Evaluator
from src.utils.visualization import Visualizer
from src.utils.metrics import MetricsEvaluator

def create_config():
    """Create configuration file if it doesn't exist"""
    config_dir = Path('config')
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / 'config.yaml'
    if not config_path.exists():
        config = {
            'dataset': {
                'raw_images_path': 'Datasets/Raw_Images',
                'train_ratio': 0.8,
                'img_size': 640,
                'batch_size': 8
            },
            'model': {
                'name': 'yolov8s',
                'classes': ['streak', 'star'],
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45
            },
            'training': {
                'epochs': 50,
                'learning_rate': 0.01,
                'weight_decay': 0.0005,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'evaluation': {
                'save_results': True,
                'save_path': 'results'
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
    
    return str(config_path)

def main():
    # Set up output directories
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration file
    config_path = create_config()
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("===== Space Object Detection using YOLOv8 =====")
    
    # Step 1: Annotate images
    print("\n----- Step 1: Annotating Images -----")
    annotator = DataAnnotator(config_path)
    annotations = annotator.annotate_all_images()
    dataset_yaml_path = annotator.create_dataset_yaml()
    
    # Step 2: Train model
    print("\n----- Step 2: Training Model -----")
    trainer = Trainer(config_path)
    results, training_time, trained_model = trainer.train(dataset_yaml_path)
    
    # Step 3: Evaluate model
    print("\n----- Step 3: Evaluating Model -----")
    evaluator = Evaluator(config_path, model=trained_model)
    metrics = evaluator.evaluate(dataset_yaml_path)
    
    # Step 4: Run inference on test images
    print("\n----- Step 4: Running Inference on Test Images -----")
    # Get test images
    test_dir = Path(config['dataset']['raw_images_path'])
    image_paths = list(test_dir.glob('*.tiff'))
    train_size = int(len(image_paths) * config['dataset']['train_ratio'])
    test_paths = image_paths[train_size:]
    
    # Create visualizer
    visualizer = Visualizer(class_names=config['model']['classes'])
    
    # Run inference
    prediction_results, summary = evaluator.predict_on_images(test_paths)
    
    # Visualize results
    print("\n----- Step 5: Visualizing Results -----")
    for i, result in enumerate(prediction_results):
        # Load image
        img_path = result['image_path']
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_8bit = (img / 65535.0 * 255.0).astype(np.uint8)
        
        # Visualize detections
        vis_img = visualizer.visualize_detections(
            img_8bit,
            result,
            show_centroids=True,
            save_path=f"output/results/detection_{i+1}.jpg"
        )
    
    # Print final results
    print("\n===== Results Summary =====")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"mAP@0.5: {metrics['map50']:.4f}")
    print(f"mAP@0.75: {metrics['map75']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['map']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Average inference time: {summary['avg_computation_time']:.4f} seconds per image")
    print(f"Frames per second (FPS): {summary['fps']:.2f}")
    print("\nDetection results have been saved to output/results/")

if __name__ == "__main__":
    main()
