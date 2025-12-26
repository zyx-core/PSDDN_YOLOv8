"""
Test script for Phase 2 data preparation scripts.

This script demonstrates the complete Phase 2 workflow:
1. Generate pseudo GT from point annotations
2. Sort dataset for curriculum learning
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pseudo_gt_init import process_dataset
from curriculum_sorting import (
    load_pseudo_gt_labels,
    calculate_global_statistics,
    sort_and_split_dataset,
    save_folds,
    save_difficulty_report
)


def test_phase2():
    """Run complete Phase 2 test."""
    
    print("=" * 60)
    print("PHASE 2 TEST: Data Preparation and Pseudo GT Generation")
    print("=" * 60)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    annotations_file = base_dir / "data" / "example_annotations.json"
    images_dir = base_dir / "data" / "images"  # Not used in this test
    labels_dir = base_dir / "data" / "labels"
    folds_dir = base_dir / "data" / "folds"
    
    # Step 1: Generate Pseudo GT
    print("\n[Step 1] Generating Pseudo GT from point annotations...")
    print("-" * 60)
    
    process_dataset(
        annotation_file=str(annotations_file),
        image_dir=str(images_dir),
        output_dir=str(labels_dir),
        img_width=640,
        img_height=480,
        format='json'
    )
    
    # Step 2: Curriculum Sorting
    print("\n[Step 2] Sorting dataset for curriculum learning...")
    print("-" * 60)
    
    # Load labels
    image_boxes = load_pseudo_gt_labels(labels_dir)
    print(f"Loaded {len(image_boxes)} images")
    
    # Calculate statistics
    mean, std = calculate_global_statistics(image_boxes)
    print(f"Global Statistics: Mean={mean:.6f}, Std={std:.6f}")
    
    # Sort and split
    folds, image_difficulties = sort_and_split_dataset(image_boxes, mean, std, num_folds=3)
    
    # Save folds
    save_folds(folds, folds_dir)
    
    # Save report
    report_file = folds_dir / "difficulty_report.txt"
    save_difficulty_report(image_difficulties, mean, std, report_file)
    
    print("\n" + "=" * 60)
    print("PHASE 2 TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - Pseudo GT labels: {labels_dir}")
    print(f"  - Curriculum folds: {folds_dir}")
    print(f"  - Difficulty report: {report_file}")


if __name__ == "__main__":
    test_phase2()
