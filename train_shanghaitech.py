"""Train PSDDN on ShanghaiTech Part B dataset."""
import sys
sys.path.insert(0, 'C:/Users/arsha/PSDDN_YOLOv8/ultralytics_repo')
sys.path.insert(0, 'C:/Users/arsha/PSDDN_YOLOv8/scripts')
import torch
from ultralytics.utils import LOGGER

from psddn_trainer import train_psddn

# Training configuration for ShanghaiTech Part B
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    train_psddn(
        data_yaml='data/shanghaitech/shanghaitech_partB.yaml',
        folds_dir='data/shanghaitech/folds',
        project='runs/psddn_shanghaitech',
        name='partB',
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        workers=args.workers,
        
        # Hyperparameters
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,  # Box loss weight
        cls=0.5,  # Classification loss weight
        device=0 if torch.cuda.is_available() else 'cpu',
    )

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
