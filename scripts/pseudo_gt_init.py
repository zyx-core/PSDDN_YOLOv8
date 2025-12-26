"""
Pseudo Ground Truth Initialization Script for PSDDN

This script converts raw point annotations (head centers) into initial pseudo 
bounding box ground truth (g0) for YOLOv8 training.

Input: Point annotations in JSON/CSV format with (x, y) coordinates
Output: YOLO format label files (.txt) with normalized bounding boxes

Algorithm:
1. For each head point g, calculate distance to nearest neighbor: d(g, NNg)
2. Initialize pseudo GT box: g0 = (xc, yc, d, d) - centered at point with side length = NN distance
3. Convert to YOLO format: class_id x_norm y_norm w_norm h_norm
"""

import json
import csv
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree
import argparse


def load_point_annotations(annotation_file, format='json'):
    """
    Load point annotations from file.
    
    Args:
        annotation_file: Path to annotation file
        format: 'json' or 'csv'
    
    Returns:
        dict: {image_name: [(x1, y1), (x2, y2), ...]}
    """
    annotations = {}
    
    if format == 'json':
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            # Expected format: {"image1.jpg": [[x1, y1], [x2, y2], ...], ...}
            for img_name, points in data.items():
                annotations[img_name] = [(p[0], p[1]) for p in points]
    
    elif format == 'csv':
        # Expected format: image_name, x, y (one point per line)
        with open(annotation_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header if present
            for row in reader:
                img_name, x, y = row[0], float(row[1]), float(row[2])
                if img_name not in annotations:
                    annotations[img_name] = []
                annotations[img_name].append((x, y))
    
    return annotations


def calculate_nn_distances(points):
    """
    Calculate nearest neighbor distance for each point.
    
    Args:
        points: List of (x, y) tuples
    
    Returns:
        List of distances to nearest neighbor for each point
    """
    if len(points) < 2:
        # If only one point or no points, use a default distance
        return [50.0] * len(points)  # Default box size
    
    points_array = np.array(points)
    kdtree = KDTree(points_array)
    
    # Query for 2 nearest neighbors (first is the point itself)
    distances, indices = kdtree.query(points_array, k=2)
    
    # Return distance to nearest neighbor (index 1, not 0 which is self)
    nn_distances = distances[:, 1]
    
    return nn_distances


def generate_pseudo_gt_boxes(points, nn_distances, img_width, img_height):
    """
    Generate pseudo GT boxes from points and NN distances.
    
    Args:
        points: List of (x, y) tuples (absolute coordinates)
        nn_distances: List of NN distances for each point
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        List of YOLO format boxes: [(class_id, x_norm, y_norm, w_norm, h_norm), ...]
    """
    yolo_boxes = []
    
    for (x, y), d in zip(points, nn_distances):
        # Center coordinates (already at point location)
        xc = x
        yc = y
        
        # Box width and height = NN distance
        w = d
        h = d
        
        # Normalize to [0, 1] for YOLO format
        x_norm = xc / img_width
        y_norm = yc / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        
        # Class ID = 0 (single class: "head")
        class_id = 0
        
        yolo_boxes.append((class_id, x_norm, y_norm, w_norm, h_norm))
    
    return yolo_boxes


def save_yolo_labels(yolo_boxes, output_file):
    """
    Save YOLO format labels to file.
    
    Args:
        yolo_boxes: List of (class_id, x_norm, y_norm, w_norm, h_norm)
        output_file: Path to output .txt file
    """
    with open(output_file, 'w') as f:
        for box in yolo_boxes:
            class_id, x, y, w, h = box
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def process_dataset(annotation_file, image_dir, output_dir, img_width=1920, img_height=1080, format='json'):
    """
    Process entire dataset to generate pseudo GT labels.
    
    Args:
        annotation_file: Path to point annotations
        image_dir: Directory containing images
        output_dir: Directory to save YOLO labels
        img_width: Default image width (if not reading from actual images)
        img_height: Default image height
        format: Annotation file format ('json' or 'csv')
    """
    # Load annotations
    print(f"Loading annotations from {annotation_file}...")
    annotations = load_point_annotations(annotation_file, format=format)
    print(f"Loaded annotations for {len(annotations)} images")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    all_nn_distances = []
    
    for img_name, points in annotations.items():
        if len(points) == 0:
            print(f"Warning: No points for {img_name}, skipping...")
            continue
        
        # Calculate NN distances
        nn_distances = calculate_nn_distances(points)
        all_nn_distances.extend(nn_distances)
        
        # Generate pseudo GT boxes
        yolo_boxes = generate_pseudo_gt_boxes(points, nn_distances, img_width, img_height)
        
        # Save to file
        label_file = output_path / f"{Path(img_name).stem}.txt"
        save_yolo_labels(yolo_boxes, label_file)
        
        print(f"Processed {img_name}: {len(points)} points -> {len(yolo_boxes)} boxes")
    
    # Print statistics
    if all_nn_distances:
        print(f"\n=== Statistics ===")
        print(f"Total points: {len(all_nn_distances)}")
        print(f"Mean NN distance: {np.mean(all_nn_distances):.2f}")
        print(f"Std NN distance: {np.std(all_nn_distances):.2f}")
        print(f"Min NN distance: {np.min(all_nn_distances):.2f}")
        print(f"Max NN distance: {np.max(all_nn_distances):.2f}")
    
    print(f"\nPseudo GT labels saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo GT from point annotations')
    parser.add_argument('--annotations', type=str, required=True, help='Path to annotation file')
    parser.add_argument('--images', type=str, required=True, help='Path to images directory')
    parser.add_argument('--output', type=str, required=True, help='Path to output labels directory')
    parser.add_argument('--format', type=str, default='json', choices=['json', 'csv'], 
                        help='Annotation file format')
    parser.add_argument('--img-width', type=int, default=1920, help='Image width in pixels')
    parser.add_argument('--img-height', type=int, default=1080, help='Image height in pixels')
    
    args = parser.parse_args()
    
    process_dataset(
        annotation_file=args.annotations,
        image_dir=args.images,
        output_dir=args.output,
        img_width=args.img_width,
        img_height=args.img_height,
        format=args.format
    )


if __name__ == "__main__":
    main()
