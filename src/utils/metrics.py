# src/utils/metrics.py
import numpy as np
import json
from pathlib import Path

class MetricsEvaluator:
    def __init__(self):
        self.results = {}
    
    def calculate_metrics(self, predictions, ground_truth=None):
        """
        Calculate metrics from prediction results
        
        Args:
            predictions: List of prediction results
            ground_truth: Ground truth annotations (optional)
            
        Returns:
            Dictionary of metrics
        """
        # Calculate timing metrics
        total_time = sum(pred['computation_time'] for pred in predictions)
        avg_time = total_time / len(predictions) if predictions else 0
        fps = len(predictions) / total_time if total_time > 0 else 0
        
        # Count detections by class
        streak_count = 0
        star_count = 0
        
        for pred in predictions:
            for det in pred['detections']:
                if det['class_name'] == 'streak':
                    streak_count += 1
                elif det['class_name'] == 'star':
                    star_count += 1
        
        # Create metrics dictionary
        metrics = {
            'timing': {
                'total_computation_time': total_time,
                'avg_computation_time': avg_time,
                'fps': fps
            },
            'detections': {
                'total': len([det for pred in predictions for det in pred['detections']]),
                'streak_count': streak_count,
                'star_count': star_count
            }
        }
        
        # Add ground truth metrics if available
        if ground_truth:
            # Calculate precision, recall, F1 score
            # This would require matching predictions with ground truth
            pass
        
        return metrics
    
    def save_metrics(self, metrics, save_path):
        """
        Save metrics to a JSON file
        
        Args:
            metrics: Metrics dictionary
            save_path: Path to save the metrics
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Metrics saved to {save_path}")
