# PSDDN Training Plan - ShanghaiTech Dataset

## Overview
This guide provides step-by-step instructions for training PSDDN on the ShanghaiTech crowd counting dataset.

## Dataset Information

**ShanghaiTech Dataset** has two parts:
- **Part A**: 482 images (300 train, 182 test) - Dense crowds, complex scenes
- **Part B**: 716 images (400 train, 316 test) - Sparse crowds, simpler scenes

**Recommendation**: Start with **Part B** (easier, faster training)

---

## Step 1: Download ShanghaiTech Dataset

### Option A: Manual Download
1. Go to: https://github.com/desenzhou/ShanghaiTechDataset
2. Download the dataset
3. Extract to: `C:\Users\arsha\PSDDN_YOLOv8\data\shanghaitech\`

### Option B: Using Git
```bash
cd C:\Users\arsha\PSDDN_YOLOv8\data
git clone https://github.com/desenzhou/ShanghaiTechDataset.git shanghaitech
```

**Expected structure:**
```
data/shanghaitech/
├── part_A/
│   ├── train_data/
│   │   ├── images/
│   │   └── ground-truth/  (MAT files with points)
│   └── test_data/
│       ├── images/
│       └── ground-truth/
└── part_B/
    ├── train_data/
    └── test_data/
```

---

## Step 2: Convert MAT Annotations to JSON

ShanghaiTech uses MATLAB `.mat` files. We need to convert them to JSON.

Create `scripts/convert_shanghaitech.py`:

```python
"""Convert ShanghaiTech MAT annotations to JSON format."""
import scipy.io as sio
import json
from pathlib import Path
import argparse

def convert_mat_to_json(mat_dir, output_file):
    """Convert all MAT files in directory to single JSON file."""
    annotations = {}
    
    mat_files = list(Path(mat_dir).glob("*.mat"))
    print(f"Found {len(mat_files)} MAT files")
    
    for mat_file in mat_files:
        # Load MAT file
        mat = sio.loadmat(str(mat_file))
        points = mat['image_info'][0, 0]['location'][0, 0]
        
        # Image name (remove GT_ prefix and .mat extension)
        img_name = mat_file.stem.replace('GT_', '')
        
        # Convert to list of [x, y] coordinates
        point_list = points.tolist()
        annotations[img_name] = point_list
        
        print(f"  {img_name}: {len(point_list)} points")
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\nSaved annotations to {output_file}")
    print(f"Total images: {len(annotations)}")
    print(f"Total annotations: {sum(len(pts) for pts in annotations.values())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat-dir', required=True, help='Directory with MAT files')
    parser.add_argument('--output', required=True, help='Output JSON file')
    args = parser.parse_args()
    
    convert_mat_to_json(args.mat_dir, args.output)
```

**Run conversion:**
```bash
cd C:\Users\arsha\PSDDN_YOLOv8

# Convert Part B training annotations
python scripts/convert_shanghaitech.py ^
    --mat-dir data/shanghaitech/part_B/train_data/ground-truth ^
    --output data/shanghaitech/part_B_train_annotations.json

# Convert Part B test annotations
python scripts/convert_shanghaitech.py ^
    --mat-dir data/shanghaitech/part_B/test_data/ground-truth ^
    --output data/shanghaitech/part_B_test_annotations.json
```

---

## Step 3: Generate Pseudo GT Labels

```bash
# Generate pseudo GT boxes from point annotations
python scripts/pseudo_gt_init.py ^
    --annotations data/shanghaitech/part_B_train_annotations.json ^
    --images data/shanghaitech/part_B/train_data/images ^
    --output data/shanghaitech/labels/train ^
    --img-width 1024 ^
    --img-height 768 ^
    --format json
```

**Expected output:**
- Creates YOLO format labels in `data/shanghaitech/labels/train/`
- One `.txt` file per image with format: `0 x_center y_center width height`

---

## Step 4: Create Curriculum Folds

```bash
# Sort dataset by difficulty and create 3 folds
python scripts/curriculum_sorting.py ^
    --labels data/shanghaitech/labels/train ^
    --output data/shanghaitech/folds ^
    --num-folds 3
```

**Expected output:**
- `data/shanghaitech/folds/fold_1.json` (easy images)
- `data/shanghaitech/folds/fold_2.json` (medium images)
- `data/shanghaitech/folds/fold_3.json` (hard images)
- `data/shanghaitech/folds/difficulty_report.txt`

---

## Step 5: Create Data Configuration

Create `data/shanghaitech/shanghaitech_partB.yaml`:

```yaml
# ShanghaiTech Part B Dataset Configuration
path: C:\Users\arsha\PSDDN_YOLOv8\data\shanghaitech
train: part_B/train_data/images
val: part_B/test_data/images

# Classes
names:
  0: head

# Number of classes
nc: 1
```

---

## Step 6: Create Training Script

Create `train_shanghaitech.py`:

```python
"""Train PSDDN on ShanghaiTech Part B dataset."""
import sys
sys.path.insert(0, 'C:/Users/arsha/PSDDN_YOLOv8/ultralytics_repo')
sys.path.insert(0, 'C:/Users/arsha/PSDDN_YOLOv8/scripts')

from psddn_trainer import train_psddn

# Training configuration
train_psddn(
    data_yaml='data/shanghaitech/shanghaitech_partB.yaml',
    folds_dir='data/shanghaitech/folds',
    model='yolov8n.yaml',  # Start from scratch
    # model='yolov8n.pt',  # Or use pretrained weights
    epochs=100,
    batch=8,  # Adjust based on GPU memory
    imgsz=640,
    project='runs/psddn_shanghaitech',
    name='partB_exp1',
    # Additional hyperparameters
    lr0=0.01,  # Initial learning rate
    warmup_epochs=3,
    box=7.5,  # Box loss weight
    cls=0.5,  # Classification loss weight
    device=0,  # GPU 0 (use 'cpu' if no GPU)
)
```

---

## Step 7: Start Training

```bash
cd C:\Users\arsha\PSDDN_YOLOv8

# Start training
python train_shanghaitech.py
```

**Training will:**
- Stage 1 (Epochs 0-29): Train on Fold 1 (easiest 133 images)
- Stage 2 (Epochs 30-59): Add Fold 2 (267 images total)
- Stage 3 (Epochs 60-99): All images (400 images)
- Update pseudo GT every 10 epochs

**Monitor training:**
- Logs: `runs/psddn_shanghaitech/partB_exp1/`
- Weights: `runs/psddn_shanghaitech/partB_exp1/weights/best.pt`
- Metrics: Check terminal output or TensorBoard

---

## Step 8: Run Inference on Test Set

```bash
# Run inference
python scripts/inference.py ^
    --model runs/psddn_shanghaitech/partB_exp1/weights/best.pt ^
    --images data/shanghaitech/part_B/test_data/images ^
    --output results/shanghaitech_partB ^
    --conf 0.5 ^
    --gt data/shanghaitech/part_B_test_annotations.json
```

---

## Step 9: Generate Evaluation Report

```bash
# Generate visualizations and report
python scripts/visualize_results.py ^
    --predictions results/shanghaitech_partB/predictions.json ^
    --metrics results/shanghaitech_partB/metrics.json ^
    --gt data/shanghaitech/part_B_test_annotations.json ^
    --output report/shanghaitech_partB
```

**View report:**
Open `report/shanghaitech_partB/report.html` in browser

---

## Expected Results (ShanghaiTech Part B)

Based on PSDDN paper benchmarks:
- **MAE**: ~7-10 (lower is better)
- **MSE**: ~12-15 (lower is better)
- **Training time**: ~2-4 hours on GPU

---

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce batch size
```python
batch=4,  # or batch=2
```

### Issue: Training too slow
**Solution**: Use smaller image size
```python
imgsz=512,  # instead of 640
```

### Issue: Poor accuracy
**Solutions**:
1. Train longer: `epochs=150`
2. Use pretrained weights: `model='yolov8n.pt'`
3. Adjust learning rate: `lr0=0.005`

### Issue: scipy not installed
```bash
pip install scipy
```

---

## Advanced: Training on Part A (Dense Crowds)

After Part B, try Part A:

```bash
# Convert Part A annotations
python scripts/convert_shanghaitech.py ^
    --mat-dir data/shanghaitech/part_A/train_data/ground-truth ^
    --output data/shanghaitech/part_A_train_annotations.json

# Generate pseudo GT (Part A has different image sizes)
python scripts/pseudo_gt_init.py ^
    --annotations data/shanghaitech/part_A_train_annotations.json ^
    --images data/shanghaitech/part_A/train_data/images ^
    --output data/shanghaitech/labels_A/train ^
    --format json

# Create curriculum folds
python scripts/curriculum_sorting.py ^
    --labels data/shanghaitech/labels_A/train ^
    --output data/shanghaitech/folds_A

# Train (may need smaller batch size)
# Update train_shanghaitech.py to use Part A config
```

---

## Summary Timeline

| Step | Task | Time |
|------|------|------|
| 1 | Download dataset | 5-10 min |
| 2 | Convert annotations | 2 min |
| 3 | Generate pseudo GT | 5 min |
| 4 | Create curriculum | 2 min |
| 5 | Setup config | 2 min |
| 6-7 | Training | 2-4 hours |
| 8 | Inference | 5 min |
| 9 | Report | 2 min |
| **Total** | | **~3-5 hours** |

---

## Next Steps After Training

1. **Analyze results**: Check which images have high errors
2. **Fine-tune**: Adjust hyperparameters based on results
3. **Compare**: Try Part A for denser crowds
4. **Deploy**: Use trained model for real-world applications

Good luck with training! 🚀
