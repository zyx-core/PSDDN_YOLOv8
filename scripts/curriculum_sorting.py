"""
Curriculum Learning Dataset Sorting Script for PSDDN

This script calculates training difficulty (TL) for each image based on the 
distribution of nearest neighbor distances and sorts the dataset into folds 
for curriculum learning.

Algorithm:
1. Calculate global mean (μ) and std (σ) of all d(g, NNg) across training set
2. For each image, calculate difficulty: TL = 1 - (1/|G|) * Σ Φ(dg | μ, σ)
3. Sort images by TL (ascending = easiest first)
4. Split into Z=3 folds: I1 (easiest), I2 (medium), I3 (hardest)
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import norm
import argparse


def load_pseudo_gt_labels(labels_dir):
    """
    Load pseudo GT labels and extract box sizes.
    
    Args:
        labels_dir: Directory containing YOLO format .txt files
    
    Returns:
        dict: {image_name: [w1, w2, ...]} (normalized widths as proxy for NN distances)
    """
    labels_path = Path(labels_dir)
    image_boxes = {}
    
    for label_file in labels_path.glob('*.txt'):
        img_name = label_file.stem
        widths = []
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    # YOLO format: class_id x_center y_center width height
                    w = float(parts[3])
                    widths.append(w)
        
        if widths:
            image_boxes[img_name] = widths
    
    return image_boxes


def calculate_global_statistics(image_boxes):
    """
    Calculate global mean and std of box widths (proxy for NN distances).
    
    Args:
        image_boxes: dict of {image_name: [w1, w2, ...]}
    
    Returns:
        tuple: (mean, std)
    """
    all_widths = []
    for widths in image_boxes.values():
        all_widths.extend(widths)
    
    mean = np.mean(all_widths)
    std = np.std(all_widths)
    
    return mean, std


def calculate_image_difficulty(widths, mean, std):
    """
    Calculate training difficulty TL for an image.
    
    TL = 1 - (1/|G|) * Σ Φ(dg | μ, σ)
    
    Where Φ is the CDF of normal distribution. Images with medium-sized boxes
    (close to mean) have low difficulty (easier to learn).
    
    Args:
        widths: List of box widths for this image
        mean: Global mean width
        std: Global std width
    
    Returns:
        float: Difficulty score (lower = easier)
    """
    if len(widths) == 0:
        return 1.0  # Maximum difficulty for empty images
    
    # Calculate CDF values for each width
    cdf_values = norm.cdf(widths, loc=mean, scale=std)
    
    # Average CDF value
    avg_cdf = np.mean(cdf_values)
    
    # Difficulty: 1 - avg_cdf
    # Images with widths close to mean have high CDF (~0.5), thus low difficulty
    difficulty = 1 - avg_cdf
    
    return difficulty


def sort_and_split_dataset(image_boxes, mean, std, num_folds=3):
    """
    Sort images by difficulty and split into folds for curriculum learning.
    
    Args:
        image_boxes: dict of {image_name: [w1, w2, ...]}
        mean: Global mean width
        std: Global std width
        num_folds: Number of folds to create (default: 3)
    
    Returns:
        list: [fold1_images, fold2_images, fold3_images]
    """
    # Calculate difficulty for each image
    image_difficulties = []
    
    for img_name, widths in image_boxes.items():
        difficulty = calculate_image_difficulty(widths, mean, std)
        image_difficulties.append((img_name, difficulty))
    
    # Sort by difficulty (ascending = easiest first)
    image_difficulties.sort(key=lambda x: x[1])
    
    # Split into folds
    total_images = len(image_difficulties)
    fold_size = total_images // num_folds
    
    folds = []
    for i in range(num_folds):
        start_idx = i * fold_size
        if i == num_folds - 1:
            # Last fold gets remaining images
            end_idx = total_images
        else:
            end_idx = (i + 1) * fold_size
        
        fold_images = [img_name for img_name, _ in image_difficulties[start_idx:end_idx]]
        folds.append(fold_images)
    
    return folds, image_difficulties


def save_folds(folds, output_dir):
    """
    Save fold information to JSON files.
    
    Args:
        folds: List of fold image lists
        output_dir: Directory to save fold files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, fold in enumerate(folds, 1):
        fold_file = output_path / f"fold_{i}.json"
        with open(fold_file, 'w') as f:
            json.dump(fold, f, indent=2)
        print(f"Fold {i}: {len(fold)} images -> {fold_file}")


def save_difficulty_report(image_difficulties, mean, std, output_file):
    """
    Save detailed difficulty report.
    
    Args:
        image_difficulties: List of (image_name, difficulty) tuples
        mean: Global mean
        std: Global std
        output_file: Path to output report
    """
    with open(output_file, 'w') as f:
        f.write("=== Curriculum Learning Difficulty Report ===\n\n")
        f.write(f"Global Statistics:\n")
        f.write(f"  Mean NN distance (width): {mean:.6f}\n")
        f.write(f"  Std NN distance (width): {std:.6f}\n\n")
        f.write(f"Image Difficulties (sorted easiest to hardest):\n")
        f.write(f"{'Image Name':<40} {'Difficulty':<12}\n")
        f.write("-" * 52 + "\n")
        
        for img_name, difficulty in image_difficulties:
            f.write(f"{img_name:<40} {difficulty:.6f}\n")
    
    print(f"Difficulty report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Sort dataset for curriculum learning')
    parser.add_argument('--labels', type=str, required=True, 
                        help='Path to pseudo GT labels directory')
    parser.add_argument('--output', type=str, required=True, 
                        help='Path to output directory for folds')
    parser.add_argument('--num-folds', type=int, default=3, 
                        help='Number of folds to create')
    
    args = parser.parse_args()
    
    # Load labels
    print(f"Loading pseudo GT labels from {args.labels}...")
    image_boxes = load_pseudo_gt_labels(args.labels)
    print(f"Loaded {len(image_boxes)} images")
    
    # Calculate global statistics
    mean, std = calculate_global_statistics(image_boxes)
    print(f"\nGlobal Statistics:")
    print(f"  Mean: {mean:.6f}")
    print(f"  Std: {std:.6f}")
    
    # Sort and split
    print(f"\nSorting images by difficulty...")
    folds, image_difficulties = sort_and_split_dataset(image_boxes, mean, std, args.num_folds)
    
    # Save folds
    print(f"\nSaving folds to {args.output}...")
    save_folds(folds, args.output)
    
    # Save report
    report_file = Path(args.output) / "difficulty_report.txt"
    save_difficulty_report(image_difficulties, mean, std, report_file)
    
    print(f"\n✓ Curriculum learning dataset prepared successfully!")


if __name__ == "__main__":
    main()
