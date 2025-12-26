# Quick Fix for Colab Import Error

## Problem
```
ModuleNotFoundError: No module named 'ultralytics.utils.psddn_loss'
```

## Solution

Add this cell **after Step 2** (after cloning repositories):

```python
# Copy PSDDN loss to installed ultralytics package
import site
import shutil
from pathlib import Path

# Find installed ultralytics location
ultralytics_path = Path(site.getsitepackages()[0]) / 'ultralytics' / 'utils'
source_file = Path('/content/PSDDN_YOLOv8/ultralytics_repo/ultralytics/utils/psddn_loss.py')

# Copy psddn_loss.py
if source_file.exists():
    shutil.copy(source_file, ultralytics_path / 'psddn_loss.py')
    print(f"✅ Copied psddn_loss.py to {ultralytics_path}")
else:
    print("❌ psddn_loss.py not found!")
```

Then continue with the rest of the notebook.

## Alternative: Manual Fix

If the above doesn't work, replace Step 8 (Training) with:

```python
# Alternative: Use sys.path instead of copying
import sys
sys.path.insert(0, '/content/PSDDN_YOLOv8/ultralytics_repo')

# Now import will work
from ultralytics.utils.psddn_loss import PSDDNDetectionLoss
from ultralytics.models.yolo.detect import DetectionTrainer

# Continue with training...
```

## Complete Fixed Step 2

Replace the Step 2 cell with this:

```python
# Clone your PSDDN implementation
!git clone https://github.com/zyx-core/PSDDN_YOLOv8.git
%cd PSDDN_YOLOv8

# Clone Ultralytics YOLOv8
!git clone https://github.com/ultralytics/ultralytics.git ultralytics_repo

# IMPORTANT: Copy custom PSDDN loss to installed ultralytics
import site
import shutil
from pathlib import Path

ultralytics_path = Path(site.getsitepackages()[0]) / 'ultralytics' / 'utils'
source_file = Path('/content/PSDDN_YOLOv8/ultralytics_repo/ultralytics/utils/psddn_loss.py')

if source_file.exists():
    shutil.copy(source_file, ultralytics_path / 'psddn_loss.py')
    print(f"✅ Copied psddn_loss.py to {ultralytics_path}")
else:
    print("❌ Error: psddn_loss.py not found!")

print("✅ Repository cloned and configured!")
```

This will fix the import error!
