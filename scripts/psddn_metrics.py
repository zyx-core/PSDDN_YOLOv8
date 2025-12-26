"""
PSDDN Custom Evaluation Metrics

Implements crowd counting metrics and point-supervised detection metrics:
1. MAE (Mean Absolute Error) for counting
2. MSE (Mean Squared Error) for counting  
3. AP (Average Precision) with relaxed point-supervised criteria
"""

import torch
import numpy as np
from typing import List, Tuple, Dict


def calculate_counting_metrics(pred_counts: List[int], gt_counts: List[int]) -> Dict[str, float]:
    """
    Calculate MAE and MSE for crowd counting.
    
    Args:
        pred_counts: List of predicted counts per image
        gt_counts: List of ground truth counts per image
    
    Returns:
        Dictionary with 'MAE' and 'MSE' keys
    """
    pred_counts = np.array(pred_counts)
    gt_counts = np.array(gt_counts)
    
    mae = np.mean(np.abs(pred_counts - gt_counts))
    mse = np.mean((pred_counts - gt_counts) ** 2)
    
    return {
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(np.sqrt(mse))
    }


def calculate_point_supervised_ap(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_points: torch.Tensor,
    nn_distances: torch.Tensor,
    center_threshold: float = 20.0,
    size_ratio: float = 1.2,
    iou_thresholds: List[float] = [0.5, 0.75]
) -> Dict[str, float]:
    """
    Calculate Average Precision with point-supervised criteria.
    
    A detection is considered correct if:
    1. Center distance between predicted box center and GT point <= center_threshold
    2. Box width or height <= size_ratio * d(g, NNg)
    
    Args:
        pred_boxes: Predicted boxes (N, 4) in xyxy format
        pred_scores: Confidence scores (N,)
        gt_points: Ground truth points (M, 2)
        nn_distances: Nearest neighbor distances for each GT point (M,)
        center_threshold: Maximum center distance in pixels (default: 20)
        size_ratio: Maximum size ratio (default: 1.2)
        iou_thresholds: List of IoU thresholds for AP calculation
    
    Returns:
        Dictionary with AP scores at different thresholds
    """
    if len(pred_boxes) == 0 or len(gt_points) == 0:
        return {f'AP@{int(t*100)}': 0.0 for t in iou_thresholds}
    
    # Convert boxes to centers and sizes
    pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2  # (N, 2)
    pred_widths = pred_boxes[:, 2] - pred_boxes[:, 0]  # (N,)
    pred_heights = pred_boxes[:, 3] - pred_boxes[:, 1]  # (N,)
    
    # Calculate distances from each prediction to each GT point
    # pred_centers: (N, 2), gt_points: (M, 2)
    # distances: (N, M)
    distances = torch.cdist(pred_centers, gt_points)  # Euclidean distance
    
    # Find closest GT point for each prediction
    min_distances, closest_gt_idx = distances.min(dim=1)  # (N,), (N,)
    
    # Get NN distance for closest GT point
    closest_nn_dist = nn_distances[closest_gt_idx]  # (N,)
    
    # Check criteria
    # Criterion 1: Center distance <= threshold
    center_criterion = min_distances <= center_threshold
    
    # Criterion 2: Size <= size_ratio * d(g, NNg)
    size_criterion = (pred_widths <= size_ratio * closest_nn_dist) | \
                     (pred_heights <= size_ratio * closest_nn_dist)
    
    # Combined: both criteria must be satisfied
    is_correct = center_criterion & size_criterion
    
    # Sort by confidence (descending)
    sorted_indices = torch.argsort(pred_scores, descending=True)
    is_correct_sorted = is_correct[sorted_indices]
    
    # Calculate precision and recall
    tp = torch.cumsum(is_correct_sorted.float(), dim=0)
    fp = torch.cumsum((~is_correct_sorted).float(), dim=0)
    
    recall = tp / len(gt_points)
    precision = tp / (tp + fp)
    
    # Calculate AP (area under precision-recall curve)
    # Use 11-point interpolation
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        if torch.sum(recall >= t) == 0:
            p = 0
        else:
            p = torch.max(precision[recall >= t])
        ap += p / 11
    
    return {
        'AP': float(ap),
        'Precision': float(precision[-1]) if len(precision) > 0 else 0.0,
        'Recall': float(recall[-1]) if len(recall) > 0 else 0.0,
        'TP': int(tp[-1]) if len(tp) > 0 else 0,
        'FP': int(fp[-1]) if len(fp) > 0 else 0,
        'GT': len(gt_points)
    }


def evaluate_psddn(
    predictions: List[Dict],
    ground_truth: List[Dict],
    center_threshold: float = 20.0,
    size_ratio: float = 1.2
) -> Dict[str, float]:
    """
    Complete PSDDN evaluation on a dataset.
    
    Args:
        predictions: List of dicts with keys 'boxes', 'scores' for each image
        ground_truth: List of dicts with keys 'points', 'nn_distances' for each image
        center_threshold: Center distance threshold in pixels
        size_ratio: Size ratio threshold
    
    Returns:
        Dictionary with all metrics (MAE, MSE, AP, etc.)
    """
    pred_counts = []
    gt_counts = []
    all_ap_metrics = []
    
    for pred, gt in zip(predictions, ground_truth):
        # Counting metrics
        pred_counts.append(len(pred['boxes']))
        gt_counts.append(len(gt['points']))
        
        # Detection metrics
        if len(pred['boxes']) > 0 and len(gt['points']) > 0:
            ap_metrics = calculate_point_supervised_ap(
                pred_boxes=torch.tensor(pred['boxes']),
                pred_scores=torch.tensor(pred['scores']),
                gt_points=torch.tensor(gt['points']),
                nn_distances=torch.tensor(gt['nn_distances']),
                center_threshold=center_threshold,
                size_ratio=size_ratio
            )
            all_ap_metrics.append(ap_metrics)
    
    # Aggregate counting metrics
    counting_metrics = calculate_counting_metrics(pred_counts, gt_counts)
    
    # Aggregate detection metrics
    if all_ap_metrics:
        mean_ap = np.mean([m['AP'] for m in all_ap_metrics])
        mean_precision = np.mean([m['Precision'] for m in all_ap_metrics])
        mean_recall = np.mean([m['Recall'] for m in all_ap_metrics])
        total_tp = sum([m['TP'] for m in all_ap_metrics])
        total_fp = sum([m['FP'] for m in all_ap_metrics])
        total_gt = sum([m['GT'] for m in all_ap_metrics])
    else:
        mean_ap = 0.0
        mean_precision = 0.0
        mean_recall = 0.0
        total_tp = 0
        total_fp = 0
        total_gt = 0
    
    return {
        **counting_metrics,
        'AP': mean_ap,
        'Precision': mean_precision,
        'Recall': mean_recall,
        'TP': total_tp,
        'FP': total_fp,
        'GT': total_gt
    }


if __name__ == "__main__":
    # Example usage
    print("Testing PSDDN evaluation metrics...")
    
    # Dummy data
    predictions = [
        {
            'boxes': [[10, 10, 30, 30], [50, 50, 70, 70]],
            'scores': [0.9, 0.8]
        }
    ]
    
    ground_truth = [
        {
            'points': [[20, 20], [60, 60]],
            'nn_distances': [40, 40]
        }
    ]
    
    metrics = evaluate_psddn(predictions, ground_truth)
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
