"""Train PSDDN on a mini subset (20 images) for testing."""
import sys
import os

# Ensure we can import from local folders
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'ultralytics_repo'))
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))

import torch
from psddn_trainer import train_psddn

if __name__ == '__main__':
    print("Starting Mini-Training (20 images, 10 epochs)...")
    print("Settings: Batch=4, Workers=0 (Safe Mode)")
    
    train_psddn(
        data_yaml='data/shanghaitech/mini.yaml',  # Triggers 'mini' mode (20 images)
        folds_dir='data/shanghaitech/folds',
        project='runs/psddn_shanghaitech',
        name='partB_mini',
        epochs=10,        # Quick run
        imgsz=640,
        batch=4,          # Safe for GPU memory
        workers=0,        # Fixes Windows hang/progress bar issue
        
        # Hyperparameters
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=0,
        box=7.5,
        cls=0.5,
        device=0 if torch.cuda.is_available() else 'cpu',
    )

    print("\n" + "="*60)
    print("Mini-Training COMPLETED!")
    print("="*60)
