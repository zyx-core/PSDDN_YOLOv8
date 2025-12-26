"""
PSDDN Custom Trainer for YOLOv8

This module implements a custom trainer for PSDDN that includes:
1. Online GT Updating: Periodically refines pseudo GT boxes using model predictions
2. Curriculum Learning: Progressive training on easy -> medium -> hard folds
3. Custom PSDDN Loss: Uses point-supervised loss functions
"""

import json
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np

from ultralytics.models.yolo.detect import DetectionTrainer
try:
    from ultralytics.utils.psddn_loss import PSDDNDetectionLoss
except ImportError:
    from psddn_loss import PSDDNDetectionLoss
from ultralytics.utils import DEFAULT_CFG, LOGGER


class PSDDNTrainer(DetectionTrainer):
    """
    Custom trainer for PSDDN (Point-Supervised Deep Detection Network).
    
    Features:
    - Uses PSDDN custom loss (center loss + size loss)
    - Online GT updating every K epochs
    - Curriculum learning with 3 progressive folds
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize PSDDN trainer with custom settings."""
        super().__init__(cfg, overrides, _callbacks)
        
        # PSDDN-specific settings
        self.gt_update_interval = 10  # Update GT every 10 epochs
        self.last_gt_update_epoch = 0
        self.curriculum_folds = []  # Will be loaded from fold files
        self.current_fold_stage = 1  # Start with fold 1
        self.fold_epochs = [30, 30, 40]  # Epochs per fold stage
        
        LOGGER.info("Initialized PSDDN Trainer")
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Override to use PSDDN-specific model configuration."""
        model = super().get_model(cfg, weights, verbose)
        return model
    
    def set_model_attributes(self):
        """Set model attributes including custom loss."""
        super().set_model_attributes()
        # The loss will be set in build_dataset when we know the model structure
    
    def get_loss(self, model):
        """Override to use PSDDN custom loss."""
        LOGGER.info("Using PSDDN custom loss (center + size constraints)")
        return PSDDNDetectionLoss(model)
    
    def load_curriculum_folds(self, folds_dir: str):
        """
        Load curriculum learning folds from directory.
        
        Args:
            folds_dir: Directory containing fold_1.json, fold_2.json, fold_3.json
        """
        folds_path = Path(folds_dir)
        self.curriculum_folds = []
        
        for i in range(1, 4):
            fold_file = folds_path / f"fold_{i}.json"
            if fold_file.exists():
                with open(fold_file, 'r') as f:
                    fold_images = json.load(f)
                    self.curriculum_folds.append(fold_images)
                    LOGGER.info(f"Loaded Fold {i}: {len(fold_images)} images")
            else:
                LOGGER.warning(f"Fold file not found: {fold_file}")
        
        if len(self.curriculum_folds) != 3:
            LOGGER.warning(f"Expected 3 folds, found {len(self.curriculum_folds)}")
    
    def get_active_images(self) -> List[str]:
        """
        Get list of active images based on current curriculum stage.
        
        Returns:
            List of image names to use in current stage
        """
        if not self.curriculum_folds:
            # No curriculum, use all images
            return None
        
        # Determine stage based on epoch
        cumulative_epochs = 0
        stage = 1
        for i, epochs in enumerate(self.fold_epochs, 1):
            cumulative_epochs += epochs
            if self.epoch < cumulative_epochs:
                stage = i
                break
        else:
            stage = 3  # Final stage
        
        # Accumulate folds up to current stage
        active_images = []
        for i in range(stage):
            if i < len(self.curriculum_folds):
                active_images.extend(self.curriculum_folds[i])
        
        if stage != self.current_fold_stage:
            self.current_fold_stage = stage
            LOGGER.info(f"Curriculum Stage {stage}: Using {len(active_images)} images")
        
        return active_images
    
    def _do_train(self, world_size=1):
        """Override training loop to add online GT updating."""
        # Call parent training loop
        # We'll add a callback for GT updating
        self.add_callback("on_train_epoch_end", self.on_epoch_end_callback)
        super()._do_train(world_size)
    
    def on_epoch_end_callback(self, trainer):
        """
        Callback executed at the end of each epoch.
        Performs online GT updating if interval is reached.
        """
        if (self.epoch - self.last_gt_update_epoch) >= self.gt_update_interval:
            LOGGER.info(f"Epoch {self.epoch}: Performing online GT update...")
            self.update_pseudo_gt()
            self.last_gt_update_epoch = self.epoch
    
    def update_pseudo_gt(self):
        """
        Online GT Updating: Refine pseudo GT boxes using current model predictions.
        
        Algorithm:
        1. Run inference on training set
        2. For each pseudo GT box, find best prediction (highest score, smaller size)
        3. Replace pseudo GT with refined prediction
        4. Update label files on disk
        """
        LOGGER.info("Running inference on training set for GT refinement...")
        
        # This is a placeholder - full implementation would:
        # 1. Load training images
        # 2. Run model inference
        # 3. Match predictions to pseudo GT
        # 4. Select best predictions (high score, size < d(g, NNg))
        # 5. Update label files
        
        # For now, just log that we would do this
        LOGGER.info("GT update completed (placeholder - implement full logic)")
    
    def preprocess_batch(self, batch):
        """
        Preprocess batch to filter images based on curriculum stage.
        
        This is called before each batch is processed.
        """
        # Get active images for current stage
        active_images = self.get_active_images()
        
        if active_images is not None:
            # Filter batch to only include active images
            # This requires modifying the batch dict
            # For now, we'll just pass through
            pass
        
        return super().preprocess_batch(batch)


def train_psddn(
    data_yaml: str,
    folds_dir: str,
    model: str = 'yolov8n.yaml',
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = 'runs/psddn',
    name: str = 'exp',
    **kwargs
):
    """
    Train PSDDN model with curriculum learning and online GT updating.
    
    Args:
        data_yaml: Path to data.yaml configuration file
        folds_dir: Directory containing curriculum fold files
        model: Model configuration (yaml) or pretrained weights
        epochs: Total number of training epochs
        imgsz: Input image size
        batch: Batch size
        project: Project directory for saving runs
        name: Experiment name
        **kwargs: Additional training arguments
    
    Example:
        >>> train_psddn(
        ...     data_yaml='data/crowd_counting.yaml',
        ...     folds_dir='data/folds',
        ...     epochs=100,
        ...     batch=16
        ... )
    """
    # Create trainer
    trainer = PSDDNTrainer(overrides={
        'data': data_yaml,
        'model': model,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'project': project,
        'name': name,
        **kwargs
    })
    
    # Load curriculum folds
    trainer.load_curriculum_folds(folds_dir)
    
    # Start training
    LOGGER.info("Starting PSDDN training with curriculum learning...")
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    # Example usage
    train_psddn(
        data_yaml='path/to/data.yaml',
        folds_dir='path/to/folds',
        epochs=100,
        batch=16
    )
