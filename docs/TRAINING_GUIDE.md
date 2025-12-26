# PSDDN Training Guide

This guide explains how to train a PSDDN model for crowd counting using the custom trainer and scripts.

## Prerequisites

1. **Dataset Prepared**: Point annotations converted to pseudo GT using Phase 2 scripts
2. **Curriculum Folds**: Dataset sorted into 3 folds using `curriculum_sorting.py`
3. **YOLOv8 Environment**: Ultralytics installed with PSDDN modifications

## Training Steps

### Step 1: Prepare Data Configuration

Create a `data.yaml` file for your dataset:

```yaml
# data.yaml
path: /path/to/dataset  # Dataset root directory
train: images/train  # Train images (relative to 'path')
val: images/val  # Val images (relative to 'path')

# Classes
names:
  0: head

# Number of classes
nc: 1
```

### Step 2: Run Training

```python
from scripts.psddn_trainer import train_psddn

train_psddn(
    data_yaml='data/crowd_counting.yaml',
    folds_dir='data/folds',  # Directory with fold_1.json, fold_2.json, fold_3.json
    model='yolov8n.yaml',  # Or use pretrained: 'yolov8n.pt'
    epochs=100,
    batch=16,
    imgsz=640,
    project='runs/psddn',
    name='exp1'
)
```

### Step 3: Monitor Training

The trainer will:
- **Stage 1 (Epochs 0-29)**: Train on Fold 1 (easiest images)
- **Stage 2 (Epochs 30-59)**: Add Fold 2 (medium difficulty)
- **Stage 3 (Epochs 60-99)**: Add Fold 3 (hardest images)
- **GT Updates**: Every 10 epochs, refine pseudo GT boxes using model predictions

## Training Configuration

### Curriculum Learning

Modify fold epochs in `psddn_trainer.py`:

```python
self.fold_epochs = [30, 30, 40]  # Epochs per stage
```

### Online GT Updating

Adjust update frequency:

```python
self.gt_update_interval = 10  # Update every 10 epochs
```

### Loss Weights

Modify in training args:

```python
train_psddn(
    ...,
    box=7.5,  # Box loss weight
    cls=0.5,  # Classification loss weight
)
```

## Evaluation

After training, evaluate using custom metrics:

```python
from scripts.psddn_metrics import evaluate_psddn

# Prepare predictions and ground truth
predictions = [
    {'boxes': [...], 'scores': [...]},  # Per image
    ...
]

ground_truth = [
    {'points': [...], 'nn_distances': [...]},  # Per image
    ...
]

metrics = evaluate_psddn(predictions, ground_truth)
print(f"MAE: {metrics['MAE']:.2f}")
print(f"MSE: {metrics['MSE']:.2f}")
print(f"AP: {metrics['AP']:.4f}")
```

## Advanced Usage

### Custom Loss Parameters

Modify 3-sigma rule bands in `psddn_loss.py`:

```python
loss_size = self.calculate_size_loss_3sigma(
    pred_sizes, target_sizes, target_centers[:, 1], weight,
    num_bands=10  # Adjust number of horizontal bands
)
```

### Manual GT Updating

Use `OnlineGTUpdater` directly:

```python
from scripts.online_gt_updater import OnlineGTUpdater

updater = OnlineGTUpdater(
    labels_dir='data/labels',
    initial_nn_distances_file='data/nn_distances.json',
    confidence_threshold=0.5
)

# Update GT using trained model
stats = updater.update_gt(model, image_paths)
print(f"Updated {stats['updated_boxes']} / {stats['total_boxes']} boxes")
```

## Troubleshooting

### Issue: Loss not decreasing
- Check that PSDDN loss is being used (see logs for "Using PSDDN custom loss")
- Verify pseudo GT boxes are reasonable (visualize with `plot_labels`)
- Try lower learning rate or longer warmup

### Issue: Curriculum not working
- Ensure fold files exist in `folds_dir`
- Check fold file format (JSON list of image names)
- Verify image names match between folds and dataset

### Issue: GT updates failing
- Ensure `nn_distances.json` file exists
- Check model is producing predictions (run inference test)
- Verify label file paths are correct

## Next Steps

After training:
1. Run final evaluation on test set
2. Visualize predictions vs ground truth
3. Analyze failure cases
4. Fine-tune hyperparameters if needed
