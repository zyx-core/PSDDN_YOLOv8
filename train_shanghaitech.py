"""Train PSDDN on ShanghaiTech Part B dataset."""
import sys
sys.path.insert(0, 'C:/Users/arsha/PSDDN_YOLOv8/ultralytics_repo')
sys.path.insert(0, 'C:/Users/arsha/PSDDN_YOLOv8/scripts')

from psddn_trainer import train_psddn

# Training configuration for ShanghaiTech Part B
train_psddn(
    data_yaml='data/shanghaitech/shanghaitech_partB.yaml',
    folds_dir='data/shanghaitech/folds',
    model='yolov8n.yaml',  # Start from scratch
    # model='yolov8n.pt',  # Or use pretrained weights (uncomment to use)
    epochs=100,
    batch=8,  # Adjust based on GPU memory (reduce to 4 or 2 if OOM)
    imgsz=640,
    project='runs/psddn_shanghaitech',
    name='partB_exp1',
    
    # Hyperparameters
    lr0=0.01,  # Initial learning rate
    warmup_epochs=3,
    box=7.5,  # Box loss weight
    cls=0.5,  # Classification loss weight
    device=0,  # GPU 0 (use 'cpu' if no GPU)
    
    # Optional: Enable mixed precision for faster training
    # amp=True,
    
    # Optional: Save checkpoint every N epochs
    # save_period=10,
)

print("\n" + "="*60)
print("Training completed!")
print("="*60)
print(f"Best weights: runs/psddn_shanghaitech/partB_exp1/weights/best.pt")
print(f"Last weights: runs/psddn_shanghaitech/partB_exp1/weights/last.pt")
print("\nNext steps:")
print("1. Run inference: python scripts/inference.py --model runs/psddn_shanghaitech/partB_exp1/weights/best.pt ...")
print("2. Generate report: python scripts/visualize_results.py ...")
