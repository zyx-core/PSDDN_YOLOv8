"""
Helper script to analyze crowd counting metrics (MAE, MSE) for a trained PSDDN model.
"""

import argparse
from pathlib import Path
from scripts.psddn_metrics import calculate_crowd_metrics, save_metrics_report

def main():
    parser = argparse.ArgumentParser(description='Analyze crowd counting metrics')
    parser.add_argument('--model', type=str, required=True, help='Path to trained YOLO model (.pt)')
    parser.add_argument('--images', type=str, required=True, help='Path to test images directory')
    parser.add_argument('--gt', type=str, required=True, help='Path to ground truth JSON file')
    parser.add_argument('--output', type=str, default='runs/psddn/metrics_report', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing model: {args.model}")
    print(f"Dataset images: {args.images}")
    print(f"Ground truth: {args.gt}")
    
    # Calculate metrics
    metrics = calculate_crowd_metrics(
        model_path=args.model,
        test_images_dir=args.images,
        gt_json_path=args.gt,
        conf_threshold=args.conf
    )
    
    # Save report
    report_file = output_dir / "counting_report.txt"
    save_metrics_report(metrics, str(report_file))
    
    print(f"\n[DONE] Results saved to {output_dir}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")

if __name__ == "__main__":
    main()
