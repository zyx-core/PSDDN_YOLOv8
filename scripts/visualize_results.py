"""
Results Visualization and Report Generation for PSDDN

Creates comprehensive visualizations and performance reports.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
import seaborn as sns


def plot_count_comparison(
    predictions_file: str,
    ground_truth_file: str,
    output_file: str
):
    """
    Plot predicted vs ground truth counts.
    
    Args:
        predictions_file: JSON file with predictions
        ground_truth_file: JSON file with ground truth
        output_file: Output image file
    """
    # Load data
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    # Extract counts
    pred_counts = [p['count'] for p in predictions]
    gt_counts = [len(ground_truth.get(Path(p['image']).stem, [])) for p in predictions]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(gt_counts, pred_counts, alpha=0.6)
    axes[0].plot([0, max(gt_counts)], [0, max(gt_counts)], 'r--', label='Perfect prediction')
    axes[0].set_xlabel('Ground Truth Count')
    axes[0].set_ylabel('Predicted Count')
    axes[0].set_title('Predicted vs Ground Truth Counts')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error distribution
    errors = np.array(pred_counts) - np.array(gt_counts)
    axes[1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', label='Zero error')
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved count comparison plot to {output_file}")


def plot_metrics_summary(
    metrics_file: str,
    output_file: str
):
    """
    Plot summary of evaluation metrics.
    
    Args:
        metrics_file: JSON file with metrics
        output_file: Output image file
    """
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Counting metrics
    counting_metrics = ['MAE', 'MSE', 'RMSE']
    counting_values = [metrics.get(m, 0) for m in counting_metrics]
    
    axes[0, 0].bar(counting_metrics, counting_values, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0, 0].set_title('Counting Metrics')
    axes[0, 0].set_ylabel('Error')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Detection metrics
    detection_metrics = ['Precision', 'Recall', 'AP']
    detection_values = [metrics.get(m, 0) for m in detection_metrics]
    
    axes[0, 1].bar(detection_metrics, detection_values, color=['#9b59b6', '#f39c12', '#1abc9c'])
    axes[0, 1].set_title('Detection Metrics')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Confusion-like metrics
    tp = metrics.get('TP', 0)
    fp = metrics.get('FP', 0)
    fn = metrics.get('GT', 0) - tp
    
    confusion_data = [tp, fp, fn]
    confusion_labels = ['True Positives', 'False Positives', 'False Negatives']
    colors = ['#27ae60', '#e67e22', '#c0392b']
    
    axes[1, 0].pie(confusion_data, labels=confusion_labels, autopct='%1.1f%%', colors=colors)
    axes[1, 0].set_title('Detection Breakdown')
    
    # Summary text
    axes[1, 1].axis('off')
    summary_text = f"""
    PSDDN Evaluation Summary
    ========================
    
    Counting Performance:
      MAE:  {metrics.get('MAE', 0):.2f}
      MSE:  {metrics.get('MSE', 0):.2f}
      RMSE: {metrics.get('RMSE', 0):.2f}
    
    Detection Performance:
      AP:        {metrics.get('AP', 0):.4f}
      Precision: {metrics.get('Precision', 0):.4f}
      Recall:    {metrics.get('Recall', 0):.4f}
    
    Detection Counts:
      True Positives:  {tp}
      False Positives: {fp}
      False Negatives: {fn}
      Ground Truth:    {metrics.get('GT', 0)}
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved metrics summary plot to {output_file}")


def generate_report(
    predictions_file: str,
    metrics_file: str,
    output_dir: str
):
    """
    Generate comprehensive HTML report.
    
    Args:
        predictions_file: JSON file with predictions
        metrics_file: JSON file with metrics
        output_dir: Output directory for report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Generate plots
    # (Plots would be generated here and saved)
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PSDDN Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            .metric {{ font-size: 24px; font-weight: bold; color: #27ae60; }}
        </style>
    </head>
    <body>
        <h1>PSDDN Crowd Counting Evaluation Report</h1>
        
        <h2>Summary</h2>
        <p>Total images evaluated: {len(predictions)}</p>
        
        <h2>Counting Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>MAE (Mean Absolute Error)</td><td class="metric">{metrics.get('MAE', 0):.2f}</td></tr>
            <tr><td>MSE (Mean Squared Error)</td><td class="metric">{metrics.get('MSE', 0):.2f}</td></tr>
            <tr><td>RMSE (Root MSE)</td><td class="metric">{metrics.get('RMSE', 0):.2f}</td></tr>
        </table>
        
        <h2>Detection Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Average Precision (AP)</td><td class="metric">{metrics.get('AP', 0):.4f}</td></tr>
            <tr><td>Precision</td><td class="metric">{metrics.get('Precision', 0):.4f}</td></tr>
            <tr><td>Recall</td><td class="metric">{metrics.get('Recall', 0):.4f}</td></tr>
        </table>
        
        <h2>Detection Breakdown</h2>
        <table>
            <tr><th>Category</th><th>Count</th></tr>
            <tr><td>True Positives</td><td>{metrics.get('TP', 0)}</td></tr>
            <tr><td>False Positives</td><td>{metrics.get('FP', 0)}</td></tr>
            <tr><td>False Negatives</td><td>{metrics.get('GT', 0) - metrics.get('TP', 0)}</td></tr>
            <tr><td>Total Ground Truth</td><td>{metrics.get('GT', 0)}</td></tr>
        </table>
    </body>
    </html>
    """
    
    report_file = output_path / "report.html"
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    print(f"Generated HTML report: {report_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate PSDDN visualization and report')
    parser.add_argument('--predictions', type=str, required=True, help='Predictions JSON file')
    parser.add_argument('--metrics', type=str, required=True, help='Metrics JSON file')
    parser.add_argument('--gt', type=str, help='Ground truth JSON file')
    parser.add_argument('--output', type=str, default='report', help='Output directory')
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    if args.gt:
        plot_count_comparison(
            args.predictions,
            args.gt,
            str(output_path / "count_comparison.png")
        )
    
    plot_metrics_summary(
        args.metrics,
        str(output_path / "metrics_summary.png")
    )
    
    # Generate report
    generate_report(
        args.predictions,
        args.metrics,
        args.output
    )
    
    print(f"\nReport generated in {args.output}/")
