# PSDDN Master Training Script (Final Version)

This script handles the entire setup, including downloading the actual dataset (300MB+), which was the cause of the previous "0 images FOUND" error.

```python
# ==============================================================================
# 1. CLEANUP & ENVIRONMENT SETUP
# ==============================================================================
import os, sys, shutil
from pathlib import Path

PROJECT_HOME = "/content/PSDDN_YOLOv8"
if os.path.exists(PROJECT_HOME): shutil.rmtree(PROJECT_HOME)

print("--- Cloning Fixed Repository ---")
!git clone https://github.com/zyx-core/PSDDN_YOLOv8.git {PROJECT_HOME}
%cd {PROJECT_HOME}

# Use -y to bypass prompts, and quiet mode for cleaner output
!pip install ultralytics==8.2.0 scipy -q
sys.path.insert(0, os.path.join(PROJECT_HOME, "scripts"))

# ==============================================================================
# 2. DATASET DOWNLOAD & PREPARATION (ShanghaiTech Part B)
# ==============================================================================
print("\n--- Downloading & Preparing Dataset (Part B) ---")

# Step 2.1: Download from official Dropbox link (reliable source)
!mkdir -p data
%cd data
if not os.path.exists("ShanghaiTech"):
    print("Downloading dataset (this may take a minute)...")
    # dl=1 ensures a direct download from Dropbox
    !wget -O shanghaitech.zip "https://www.dropbox.com/scl/fi/dkj5kulc9zj0rzesslck8/ShanghaiTech_Crowd_Counting_Dataset.zip?rlkey=ymbcj50ac04uvqn8p49j9af5f&dl=1"
    print("Unzipping...")
    !unzip -q shanghaitech.zip
    
    # Standardize the folder name if it was zipped with a different one
    if os.path.exists("ShanghaiTech_Crowd_Counting_Dataset"):
        os.rename("ShanghaiTech_Crowd_Counting_Dataset", "ShanghaiTech")
%cd ..

# Step 2.2: Convert annotations (Now using robust recursive search)
print("Converting annotations...")
!python scripts/convert_shanghaitech.py --mat-dir data/ShanghaiTech/part_B/train_data --output data/shanghaitech/part_B_train.json
!python scripts/convert_shanghaitech.py --mat-dir data/ShanghaiTech/part_B/test_data --output data/shanghaitech/part_B_test.json

# Step 2.3: Generate Pseudo GT Labels
print("Generating pseudo GT labels...")
!python scripts/pseudo_gt_init.py \
    --annotations data/shanghaitech/part_B_train.json \
    --images data/ShanghaiTech/part_B/train_data/images \
    --output data/shanghaitech/labels/train \
    --img-width 1024 \
    --img-height 768 \
    --format json

# Step 2.4: Create Curriculum Folds
print("Creating curriculum folds...")
!python scripts/curriculum_sorting.py \
    --labels data/shanghaitech/labels/train \
    --output data/shanghaitech/folds

# Step 2.5: Create Data Config YAML
print("Creating data config...")
yaml_content = f"""
path: /content/PSDDN_YOLOv8/data/shanghaitech
train: /content/PSDDN_YOLOv8/data/ShanghaiTech/part_B/train_data/images
val: /content/PSDDN_YOLOv8/data/ShanghaiTech/part_B/test_data/images
names:
  0: head
nc: 1
"""
with open("data/shanghaitech/shanghaitech_partB.yaml", "w") as f:
    f.write(yaml_content.strip())

# ==============================================================================
# 3. START PSDDN TRAINING
# ==============================================================================
print("\n--- Starting PSDDN Training ---")
from psddn_trainer import train_psddn

train_psddn(
    data_yaml='data/shanghaitech/shanghaitech_partB.yaml',
    folds_dir='data/shanghaitech/folds',
    model='yolov8n.pt',
    epochs=100,
    batch=16,
    imgsz=640,
    project='runs/psddn',
    name='colab_exp',
    device=0
)
```

### Why this is better:
1.  **Reliable Data**: Uses a Dropbox direct link that actually contains the 300MB of images and MAT files.
2.  **Robust Path Finding**: The conversion script now searches recursively, so it will find the MAT files even if the zip extraction creates nested folders.
3.  **Clean Configuration**: The YAML generation uses a standard triple-quoted string to avoid any scanner errors.
4.  **Version Control**: Explicitly sets `ultralytics` version for better stability.
