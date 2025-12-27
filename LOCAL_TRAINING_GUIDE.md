# PSDDN Local Training Guide (Windows)

Follow these steps to train PSDDN on your local machine using your NVIDIA GPU.

## 🛠️ Prerequisites
- **Python 3.8 - 3.11** installed.
- **Git** installed.
- **NVIDIA GPU** with latest drivers.

---

## 🚀 Setup Instructions

### 1. Initialize Environment
Open PowerShell in the project root and run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\setup_local.ps1
```

### 2. Download the Dataset
Run the simplified Python downloader to get ShanghaiTech Part B:
```powershell
.\venv\Scripts\python.exe scripts/download_data.py
```

### 3. Preparation
Process the data for training:
```powershell
# Convert MAT to JSON
.\venv\Scripts\python.exe scripts/convert_shanghaitech.py --mat-dir data/ShanghaiTech/part_B/train_data --output data/shanghaitech/part_B_train.json

# Generate Pseudo GT
.\venv\Scripts\python.exe scripts/pseudo_gt_init.py --annotations data/shanghaitech/part_B_train.json --images data/ShanghaiTech/part_B/train_data/images --output data/shanghaitech/labels/train --img-width 1024 --img-height 768 --format json

# Create Curriculum Folds
.\venv\Scripts\python.exe scripts/curriculum_sorting.py --labels data/shanghaitech/labels/train --output data/shanghaitech/folds
```

### 4. Start Training 🚀
```powershell
.\venv\Scripts\python.exe train_shanghaitech.py
```

---

## 💡 Tips for 4GB VRAM
If you get **Out of Memory (OOM)** errors:
1. Open `train_shanghaitech.py`.
2. Reduce `batch=8` to `batch=4` or `batch=2`.
3. Reduce `imgsz=640` to `imgsz=512`.

---

## 📊 Viewing Results
Once training is done, results will be in `runs/psddn_shanghaitech/`.
To visualize:
```powershell
.\venv\Scripts\python.exe scripts/visualize_results.py --results runs/psddn_shanghaitech/partB_exp1
```
