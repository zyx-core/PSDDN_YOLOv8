# PSDDN Loss Integration Guide

## How to Use PSDDN Loss in Training

The PSDDN custom loss has been implemented in `ultralytics/utils/psddn_loss.py`. To use it in your YOLOv8 training, you need to modify the model initialization to use `PSDDNDetectionLoss` instead of the default `v8DetectionLoss`.

### Method 1: Modify the Model Class (Recommended)

Edit `ultralytics/models/yolo/detect/train.py` or create a custom training script:

```python
from ultralytics import YOLO
from ultralytics.utils.psddn_loss import PSDDNDetectionLoss

# Load model
model = YOLO('yolov8n.yaml')

# Replace the loss function before training
# This needs to be done after model initialization but before training
# You'll need to modify the DetectionTrainer class to use PSDDNDetectionLoss

# In ultralytics/models/yolo/detect/train.py, modify the DetectionTrainer class:
# Change:
#   self.loss = v8DetectionLoss(self.model)
# To:
#   from ultralytics.utils.psddn_loss import PSDDNDetectionLoss
#   self.loss = PSDDNDetectionLoss(self.model)
```

### Method 2: Create Custom Trainer

Create a custom trainer class that inherits from `DetectionTrainer`:

```python
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.psddn_loss import PSDDNDetectionLoss

class PSDDNTrainer(DetectionTrainer):
    def get_loss(self, model):
        """Override to use PSDDN loss instead of default v8DetectionLoss."""
        return PSDDNDetectionLoss(model)

# Use custom trainer
model = YOLO('yolov8n.yaml')
model.train(
    data='path/to/data.yaml',
    epochs=100,
    trainer=PSDDNTrainer
)
```

### Method 3: Direct Modification (Quick Test)

For quick testing, directly modify `ultralytics/models/yolo/detect/train.py`:

**File**: `ultralytics_repo/ultralytics/models/yolo/detect/train.py`

**Find** (around line 20-30):
```python
from ultralytics.utils.loss import v8DetectionLoss
```

**Replace with**:
```python
from ultralytics.utils.psddn_loss import PSDDNDetectionLoss as v8DetectionLoss
```

This aliases `PSDDNDetectionLoss` as `v8DetectionLoss`, so the rest of the code works without changes.

## Loss Components

The PSDDN loss consists of:

1. **Center Loss (l_xy)**: MSE between predicted box center and ground truth point
   - Ensures the box is centered at the annotated head point
   
2. **Size Loss (l_w, l_h)**: 3-sigma rule constraint
   - Groups boxes by horizontal bands (y-coordinate)
   - Calculates local mean (μ) and std (σ) of box sizes in each band
   - Penalizes boxes outside μ ± 3σ range
   - Enforces perspective consistency (nearby objects have similar sizes)

3. **Classification Loss**: Standard BCE (unchanged from YOLOv8)

## Next Steps

1. Test the loss with dummy data: `python scripts/test_psddn_loss.py`
2. Modify the trainer to use PSDDN loss (choose one of the methods above)
3. Prepare your dataset using Phase 2 scripts
4. Start training with curriculum learning (Phase 4)
