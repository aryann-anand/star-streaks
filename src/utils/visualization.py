# src/utils/visualization.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class Visualizer:
    def __init__(self, class_names=None):
        self.class_names = class_names or ['streak', 'star']
        self.colors = {
            'streak': (0, 0, 255),  # Red for streaks
            'star': (0, 255, 0)     # Green for stars
        }
    
    def visualize_detections(self, image, detections, show_centroids=True, save_path=None):
        """
        Visualize object detections on an image
        
        Args:
            image: Input image (numpy array)
            detections: Detection results from the model
            show_centroids: Whether to show centroids
            save_path: Path to save the visualization
        
        Returns:
            Visualization image
        """
        # Make a copy of the image to draw on
        vis_image = image.copy()
        
        # Convert grayscale to RGB if needed
        if len(vis_image.shape) == 2 or vis_image.shape[2] == 1:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Draw bounding boxes and labels
        for det in detections['detections']:
            x1, y1, x2, y2 = det['box']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_name = det['class_name']
            confidence = det['confidence']
            color = self.colors[class_name]
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw centroids if requested
        if show_centroids and detections['centroids']:
            for centroid_info in detections['centroids']:
                cx, cy = centroid_info['centroid']
                cx, cy = int(cx), int(cy)
                class_name = centroid_info['class_name']
                color = self.colors[class_name]
                
                # Draw centroid
                cv2.circle(vis_image, (cx, cy), 5, color, -1)
                
                # Draw centroid coordinates
                label = f"({cx}, {cy})"
                cv2.putText(vis_image, label, (cx + 10, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save the visualization if requested
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path, vis_image)
        
        return vis_image
