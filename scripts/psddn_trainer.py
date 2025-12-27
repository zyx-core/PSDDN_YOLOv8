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
try:
    from online_gt_updater import OnlineGTUpdater
except ImportError:
    # Handle if script is imported differently
    from scripts.online_gt_updater import OnlineGTUpdater


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
        # Fix for resume: Ensure 'epochs' override persists after check_resume
        if overrides and 'epochs' in overrides:
            self.args.epochs = overrides['epochs']
        
        # PSDDN-specific settings
        self.gt_update_interval = 10  # Update GT every 10 epochs
        self.last_gt_update_epoch = 0
        self.curriculum_folds = []  # Will be loaded from fold files
        self.current_fold_stage = 1  # Start with fold 1
        self.fold_epochs = [30, 30, 40]  # Epochs per fold stage
        
        # Initialize GT Updater if file provided
        self.nn_distances_file = overrides.get('nn_distances', None) if overrides else None
        if self.nn_distances_file and Path(self.nn_distances_file).exists():
            self.gt_updater = OnlineGTUpdater(
                labels_dir=overrides.get('labels_dir', 'data/labels'),
                initial_nn_distances_file=self.nn_distances_file,
                confidence_threshold=overrides.get('conf_refine', 0.5)
            )
            LOGGER.info(f"OnlineGTUpdater initialized with {self.nn_distances_file}")
        else:
            self.gt_updater = None
            if self.nn_distances_file:
                LOGGER.warning(f"NN distances file NOT found: {self.nn_distances_file}. Refinement disabled.")
        
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
        Load curriculum learning folds from directory with robust name matching.
        
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
                    
                    if not fold_images:
                        LOGGER.warning(f"Fold file {fold_file.name} is empty!")
                    
                    # Robust name handling: Keep just stems and handle 'processed_' prefix
                    normalized_images = []
                    for img in fold_images:
                        stem = Path(img).stem.replace('processed_', '')
                        normalized_images.append(stem)
                    
                    # MINI MODE: Limit to 20 images if config name contains 'mini'
                    if 'mini' in str(self.args.data):
                        LOGGER.info("MINI MODE ENABLED: Using first 20 images only.")
                        normalized_images = normalized_images[:20]
                        
                    self.curriculum_folds.append(normalized_images)
                    LOGGER.info(f"Loaded Fold {i}: {len(normalized_images)} entries")
            else:
                LOGGER.error(f"CRITICAL: Fold file not found: {fold_file}")
                # Add empty list to maintain index alignment
                self.curriculum_folds.append([])
        
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
        super()._do_train()
    
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
        """
        if not self.gt_updater:
            LOGGER.warning("OnlineGTUpdater not initialized (missing nn_distances file). Skipping refinement.")
            return

        LOGGER.info("Running Online GT Refinement...")
        
        # Get image paths for the current curriculum stage
        active_stems = self.get_active_images()
        if not active_stems:
            # Fallback to all images in training loader
            image_paths = self.train_loader.dataset.im_files
        else:
            # Filter only active images
            image_paths = [
                im for im in self.train_loader.dataset.im_files 
                if Path(im).stem.replace('processed_', '') in active_stems
            ]

        if not image_paths:
            LOGGER.warning("No images found for GT refinement!")
            return

        # Perform update
        stats = self.gt_updater.update_gt(
            model=self.model,
            image_paths=image_paths,
            device=next(self.model.parameters()).device
        )
        
        LOGGER.info(f"GT update completed. Results: {stats}")
    
    def preprocess_batch(self, batch):
        """
        Preprocess batch to filter images based on curriculum stage.
        
        This is called before each batch is processed.
        """
        # Note: Filtering at the batch level is inefficient in YOLOv8
        # because the dataloader already prepared the batch.
        # Instead, we filter at the dataset creation level (not implemented here yet)
        # or we filter the loss based on image filenames.
        
        return super().preprocess_batch(batch)

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build PSDDN dataset with curriculum filtering."""
        from ultralytics.data.dataset import YOLODataset
        
        dataset = YOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls,
            stride=int(self.stride),
            pad=0.0,
            data=self.data,
            classes=self.args.classes,
            task=self.args.task
        )
        
        if mode == "train" and self.curriculum_folds:
            active_stems = self.get_active_images()
            if active_stems:
                LOGGER.info(f"Filtering dataset to {len(active_stems)} active images...")
                # Filter im_files and labels
                filtered_im_files = []
                filtered_labels = []
                for im, lb in zip(dataset.im_files, dataset.labels):
                    stem = Path(im).stem.replace('processed_', '')
                    if stem in active_stems:
                        filtered_im_files.append(im)
                        filtered_labels.append(lb)
                
                dataset.im_files = filtered_im_files
                dataset.labels = filtered_labels
                LOGGER.info(f"Dataset filtered: {len(dataset.im_files)} images remaining")
                
        return dataset


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
    
    # Sanity check
    total_images = sum(len(f) for f in trainer.curriculum_folds)
    if total_images == 0:
        LOGGER.error("CRITICAL ERROR: No images found in curriculum folds. Training will likely fail.")
        LOGGER.error(f"Check if {folds_dir} contains valid fold_1.json, etc.")
    
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
