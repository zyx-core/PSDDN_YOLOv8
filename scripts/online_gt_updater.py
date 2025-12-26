"""
Online GT Updating Module for PSDDN

Implements the iterative refinement of pseudo ground truth boxes
using model predictions during training.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json


class OnlineGTUpdater:
    """
    Handles online updating of pseudo GT boxes during training.
    
    Algorithm:
    1. Run inference on training set
    2. For each pseudo GT box, find best prediction
    3. Select prediction if: high confidence AND smaller size than initial d(g, NNg)
    4. Update label files
    """
    
    def __init__(
        self,
        labels_dir: str,
        initial_nn_distances_file: str,
        confidence_threshold: float = 0.5,
        update_frequency: int = 10
    ):
        """
        Initialize GT updater.
        
        Args:
            labels_dir: Directory containing YOLO label files
            initial_nn_distances_file: JSON file with initial NN distances per image
            confidence_threshold: Minimum confidence for replacement
            update_frequency: Update every N epochs
        """
        self.labels_dir = Path(labels_dir)
        self.confidence_threshold = confidence_threshold
        self.update_frequency = update_frequency
        
        # Load initial NN distances
        with open(initial_nn_distances_file, 'r') as f:
            self.initial_nn_distances = json.load(f)
    
    def update_gt(
        self,
        model,
        image_paths: List[str],
        device: str = 'cuda'
    ) -> Dict[str, int]:
        """
        Perform online GT update using current model.
        
        Args:
            model: Trained YOLO model
            image_paths: List of training image paths
            device: Device to run inference on
        
        Returns:
            Dictionary with update statistics
        """
        model.eval()
        stats = {
            'total_boxes': 0,
            'updated_boxes': 0,
            'skipped_no_predictions': 0,
            'skipped_low_confidence': 0,
            'skipped_large_size': 0
        }
        
        with torch.no_grad():
            for img_path in image_paths:
                img_name = Path(img_path).stem
                label_file = self.labels_dir / f"{img_name}.txt"
                
                if not label_file.exists():
                    continue
                
                # Load current pseudo GT
                current_gt = self.load_labels(label_file)
                if len(current_gt) == 0:
                    continue
                
                stats['total_boxes'] += len(current_gt)
                
                # Run inference
                results = model(img_path)
                predictions = results[0].boxes  # First image
                
                if len(predictions) == 0:
                    stats['skipped_no_predictions'] += len(current_gt)
                    continue
                
                # Get initial NN distances for this image
                nn_distances = self.initial_nn_distances.get(img_name, [])
                
                # Update each GT box
                updated_gt = []
                for i, gt_box in enumerate(current_gt):
                    # Find best matching prediction
                    best_pred = self.find_best_prediction(
                        gt_box,
                        predictions,
                        nn_distances[i] if i < len(nn_distances) else None
                    )
                    
                    if best_pred is not None:
                        updated_gt.append(best_pred)
                        stats['updated_boxes'] += 1
                    else:
                        updated_gt.append(gt_box)
                
                # Save updated labels
                self.save_labels(label_file, updated_gt)
        
        model.train()
        return stats
    
    def find_best_prediction(
        self,
        gt_box: np.ndarray,
        predictions,
        initial_nn_distance: float = None
    ) -> np.ndarray:
        """
        Find best prediction to replace GT box.
        
        Criteria:
        1. Confidence > threshold
        2. Box size (width or height) < initial NN distance
        3. Highest confidence among valid candidates
        
        Args:
            gt_box: Current GT box [class_id, x_center, y_center, width, height]
            predictions: Model predictions (boxes object from ultralytics)
            initial_nn_distance: Initial NN distance for size constraint
        
        Returns:
            Best prediction box or None if no valid candidate
        """
        if len(predictions) == 0:
            return None
        
        # Convert predictions to numpy
        pred_boxes = predictions.xyxyn.cpu().numpy()  # Normalized xyxy
        pred_confs = predictions.conf.cpu().numpy()
        
        # Filter by confidence
        valid_mask = pred_confs > self.confidence_threshold
        
        if initial_nn_distance is not None:
            # Convert to xywh for size check
            pred_widths = pred_boxes[:, 2] - pred_boxes[:, 0]
            pred_heights = pred_boxes[:, 3] - pred_boxes[:, 1]
            
            # Size constraint: width or height < initial NN distance
            size_mask = (pred_widths < initial_nn_distance) | (pred_heights < initial_nn_distance)
            valid_mask = valid_mask & size_mask
        
        if not valid_mask.any():
            return None
        
        # Select highest confidence among valid
        valid_indices = np.where(valid_mask)[0]
        best_idx = valid_indices[np.argmax(pred_confs[valid_indices])]
        
        # Convert best prediction to YOLO format
        best_box_xyxy = pred_boxes[best_idx]
        x_center = (best_box_xyxy[0] + best_box_xyxy[2]) / 2
        y_center = (best_box_xyxy[1] + best_box_xyxy[3]) / 2
        width = best_box_xyxy[2] - best_box_xyxy[0]
        height = best_box_xyxy[3] - best_box_xyxy[1]
        
        return np.array([gt_box[0], x_center, y_center, width, height])
    
    def load_labels(self, label_file: Path) -> List[np.ndarray]:
        """Load YOLO format labels from file."""
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append(np.array([float(x) for x in parts]))
        return labels
    
    def save_labels(self, label_file: Path, labels: List[np.ndarray]):
        """Save YOLO format labels to file."""
        with open(label_file, 'w') as f:
            for label in labels:
                f.write(f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")


def create_nn_distances_file(
    annotations_file: str,
    output_file: str,
    format: str = 'json'
):
    """
    Create a file storing initial NN distances for each image.
    
    This should be run after pseudo_gt_init.py to save the NN distances
    for use during online GT updating.
    
    Args:
        annotations_file: Original point annotations file
        output_file: Output JSON file path
        format: Input format ('json' or 'csv')
    """
    from pseudo_gt_init import load_point_annotations, calculate_nn_distances
    
    annotations = load_point_annotations(annotations_file, format=format)
    nn_distances_dict = {}
    
    for img_name, points in annotations.items():
        nn_distances = calculate_nn_distances(points)
        nn_distances_dict[img_name] = nn_distances.tolist()
    
    with open(output_file, 'w') as f:
        json.dump(nn_distances_dict, f, indent=2)
    
    print(f"Saved NN distances to {output_file}")


if __name__ == "__main__":
    print("Online GT Updater module loaded successfully")
