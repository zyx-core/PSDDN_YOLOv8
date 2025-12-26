"""
PSDDN Inference Script

Run inference on test set and generate crowd counting results.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import cv2
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO
from psddn_metrics import evaluate_psddn


def run_inference(
    model_path: str,
    test_images_dir: str,
    output_dir: str,
    conf_threshold: float = 0.5,
    save_visualizations: bool = True
) -> Dict:
    """
    Run inference on test set and save results.
    
    Args:
        model_path: Path to trained PSDDN model weights
        test_images_dir: Directory containing test images
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
        save_visualizations: Whether to save visualized predictions
    
    Returns:
        Dictionary with predictions and statistics
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_visualizations:
        vis_dir = output_path / "visualizations"
        vis_dir.mkdir(exist_ok=True)
    
    # Get test images
    test_images = list(Path(test_images_dir).glob("*.jpg")) + \
                  list(Path(test_images_dir).glob("*.png"))
    
    print(f"Found {len(test_images)} test images")
    
    # Run inference
    all_predictions = []
    all_counts = []
    
    for img_path in tqdm(test_images, desc="Running inference"):
        # Predict
        results = model(str(img_path), conf=conf_threshold, verbose=False)
        result = results[0]
        
        # Extract predictions
        boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) > 0 else np.array([])
        scores = result.boxes.conf.cpu().numpy() if len(result.boxes) > 0 else np.array([])
        
        # Count
        count = len(boxes)
        all_counts.append(count)
        
        # Store predictions
        all_predictions.append({
            'image': img_path.name,
            'count': count,
            'boxes': boxes.tolist(),
            'scores': scores.tolist()
        })
        
        # Visualize
        if save_visualizations:
            vis_img = visualize_predictions(
                str(img_path),
                boxes,
                scores,
                count
            )
            cv2.imwrite(str(vis_dir / img_path.name), vis_img)
    
    # Save predictions
    predictions_file = output_path / "predictions.json"
    with open(predictions_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    print(f"\nSaved predictions to {predictions_file}")
    
    # Statistics
    stats = {
        'total_images': len(test_images),
        'total_detections': sum(all_counts),
        'mean_count': np.mean(all_counts),
        'std_count': np.std(all_counts),
        'min_count': int(np.min(all_counts)),
        'max_count': int(np.max(all_counts))
    }
    
    return {
        'predictions': all_predictions,
        'statistics': stats
    }


def visualize_predictions(
    image_path: str,
    boxes: np.ndarray,
    scores: np.ndarray,
    count: int
) -> np.ndarray:
    """
    Visualize predictions on image.
    
    Args:
        image_path: Path to image
        boxes: Predicted boxes (N, 4) in xyxy format
        scores: Confidence scores (N,)
        count: Total count
    
    Returns:
        Visualized image
    """
    img = cv2.imread(image_path)
    
    # Draw boxes
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        
        # Color based on confidence
        color = (0, int(255 * score), int(255 * (1 - score)))
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw confidence
        cv2.putText(
            img,
            f"{score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    
    # Draw count
    cv2.putText(
        img,
        f"Count: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )
    
    return img


def evaluate_with_ground_truth(
    predictions_file: str,
    ground_truth_file: str,
    output_file: str
):
    """
    Evaluate predictions against ground truth.
    
    Args:
        predictions_file: JSON file with predictions
        ground_truth_file: JSON file with ground truth points
        output_file: Output file for metrics
    """
    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions_data = json.load(f)
    
    # Load ground truth
    with open(ground_truth_file, 'r') as f:
        gt_data = json.load(f)
    
    # Convert to evaluation format
    predictions = []
    ground_truth = []
    
    for pred in predictions_data:
        img_name = Path(pred['image']).stem
        
        predictions.append({
            'boxes': pred['boxes'],
            'scores': pred['scores']
        })
        
        # Find matching GT
        if img_name in gt_data:
            gt_points = gt_data[img_name]
            # Calculate NN distances (simplified - should use actual NN distances)
            nn_distances = [50.0] * len(gt_points)  # Placeholder
            
            ground_truth.append({
                'points': gt_points,
                'nn_distances': nn_distances
            })
        else:
            ground_truth.append({
                'points': [],
                'nn_distances': []
            })
    
    # Evaluate
    metrics = evaluate_psddn(predictions, ground_truth)
    
    # Save metrics
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nEvaluation Metrics:")
    print(f"  MAE: {metrics['MAE']:.2f}")
    print(f"  MSE: {metrics['MSE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  AP: {metrics['AP']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    
    print(f"\nSaved metrics to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='PSDDN Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--images', type=str, required=True, help='Test images directory')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualizations')
    parser.add_argument('--gt', type=str, help='Ground truth file for evaluation')
    
    args = parser.parse_args()
    
    # Run inference
    results = run_inference(
        model_path=args.model,
        test_images_dir=args.images,
        output_dir=args.output,
        conf_threshold=args.conf,
        save_visualizations=not args.no_vis
    )
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Inference Statistics")
    print("=" * 60)
    for key, value in results['statistics'].items():
        print(f"  {key}: {value}")
    
    # Evaluate if GT provided
    if args.gt:
        print("\n" + "=" * 60)
        print("Evaluation")
        print("=" * 60)
        evaluate_with_ground_truth(
            predictions_file=f"{args.output}/predictions.json",
            ground_truth_file=args.gt,
            output_file=f"{args.output}/metrics.json"
        )


if __name__ == "__main__":
    main()
