"""
PSDDN Custom Loss Implementation for YOLOv8

This module implements the Point-Supervised Deep Detection Network (PSDDN) loss functions
for crowd counting with point annotations.

Key Components:
1. PSDDNBboxLoss: Custom bbox loss with center and size constraints
2. PSDDNDetectionLoss: Modified v8DetectionLoss for point supervision
"""

import torch
import torch.nn as nn
from typing import Tuple

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


class PSDDNBboxLoss(nn.Module):
    """
    Custom bounding box loss for PSDDN with locally-constrained regression.
    
    Implements:
    1. Center Loss (l_xy): MSE between predicted center and ground truth point
    2. Size Loss (l_w, l_h): 3-sigma rule to constrain box sizes within local distribution
    """
    
    def __init__(self, reg_max: int = 16):
        """Initialize PSDDN bbox loss."""
        super().__init__()
        self.reg_max = reg_max
        self.dfl_loss = None  # We don't use DFL for PSDDN
        
    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,  # These are pseudo GT boxes (xyxy format)
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        gt_points: torch.Tensor = None,  # Ground truth head points (x, y)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PSDDN bbox loss.
        
        Args:
            pred_dist: Predicted distribution (not used in PSDDN)
            pred_bboxes: Predicted boxes in xyxy format (b, h*w, 4)
            anchor_points: Anchor points (h*w, 2)
            target_bboxes: Pseudo GT boxes in xyxy format (b, h*w, 4)
            target_scores: Target scores (b, h*w, num_classes)
            target_scores_sum: Sum of target scores
            fg_mask: Foreground mask (b, h*w)
            gt_points: Ground truth points (b, h*w, 2) - optional
            
        Returns:
            loss_bbox: Combined bbox loss
            loss_dfl: DFL loss (0 for PSDDN)
        """
        if not fg_mask.any():
            return torch.tensor(0.0, device=pred_bboxes.device), torch.tensor(0.0, device=pred_bboxes.device)
        
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        # Extract predicted and target boxes for positive samples
        pred_boxes_pos = pred_bboxes[fg_mask]  # (N, 4) xyxy
        target_boxes_pos = target_bboxes[fg_mask]  # (N, 4) xyxy
        
        # Convert xyxy to xywh for easier center/size extraction
        pred_xywh = self.xyxy2xywh(pred_boxes_pos)
        target_xywh = self.xyxy2xywh(target_boxes_pos)
        
        # 1. Center Loss (l_xy): MSE on center coordinates
        pred_centers = pred_xywh[:, :2]  # (N, 2)
        target_centers = target_xywh[:, :2]  # (N, 2)
        
        loss_center = ((pred_centers - target_centers) ** 2).sum(dim=1, keepdim=True)  # (N, 1)
        loss_center = (loss_center * weight).sum() / target_scores_sum
        
        # 2. Size Loss (l_w, l_h): 3-sigma rule
        pred_sizes = pred_xywh[:, 2:]  # (N, 2) - width, height
        target_sizes = target_xywh[:, 2:]  # (N, 2)
        
        # Calculate local statistics for size constraints
        # Group by horizontal bands (approximate - use y-coordinate bins)
        loss_size = self.calculate_size_loss_3sigma(
            pred_sizes, target_sizes, target_centers[:, 1], weight
        )
        loss_size = loss_size / target_scores_sum
        
        # Combine losses
        loss_bbox = loss_center + loss_size
        
        # No DFL loss for PSDDN
        loss_dfl = torch.tensor(0.0, device=pred_bboxes.device)
        
        return loss_bbox, loss_dfl
    
    @staticmethod
    def xyxy2xywh(boxes: torch.Tensor) -> torch.Tensor:
        """Convert xyxy format to xywh (center format)."""
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        w = x2 - x1
        h = y2 - y1
        xc = x1 + w / 2
        yc = y1 + h / 2
        return torch.stack([xc, yc, w, h], dim=-1)
    
    def calculate_size_loss_3sigma(
        self,
        pred_sizes: torch.Tensor,
        target_sizes: torch.Tensor,
        y_coords: torch.Tensor,
        weight: torch.Tensor,
        num_bands: int = 10
    ) -> torch.Tensor:
        """
        Calculate size loss using 3-sigma rule.
        
        Boxes in the same horizontal band should have similar sizes.
        Penalize predictions that fall outside μ ± 3σ of local distribution.
        
        Args:
            pred_sizes: Predicted widths and heights (N, 2)
            target_sizes: Target widths and heights (N, 2)
            y_coords: Y-coordinates for grouping into bands (N,)
            weight: Sample weights (N, 1)
            num_bands: Number of horizontal bands to divide image into
            
        Returns:
            Size loss
        """
        device = pred_sizes.device
        N = pred_sizes.shape[0]
        
        if N == 0:
            return torch.tensor(0.0, device=device)
        
        # Normalize y_coords to [0, 1] and assign to bands
        y_min, y_max = y_coords.min(), y_coords.max()
        if y_max - y_min < 1e-6:
            # All points in same band
            band_indices = torch.zeros(N, dtype=torch.long, device=device)
        else:
            y_norm = (y_coords - y_min) / (y_max - y_min + 1e-6)
            band_indices = (y_norm * num_bands).long().clamp(0, num_bands - 1)
        
        loss_size = torch.zeros(N, 1, device=device)
        
        # For each band, calculate local statistics and apply 3-sigma rule
        for band_idx in range(num_bands):
            mask = band_indices == band_idx
            if not mask.any():
                continue
            
            # Get target sizes in this band
            band_target_sizes = target_sizes[mask]  # (M, 2)
            
            if band_target_sizes.shape[0] < 2:
                # Not enough samples for statistics, skip constraint
                continue
            
            # Calculate mean and std for width and height separately
            mean_w = band_target_sizes[:, 0].mean()
            std_w = band_target_sizes[:, 0].std() + 1e-6
            mean_h = band_target_sizes[:, 1].mean()
            std_h = band_target_sizes[:, 1].std() + 1e-6
            
            # Get predicted sizes in this band
            band_pred_sizes = pred_sizes[mask]  # (M, 2)
            
            # Apply 3-sigma rule
            # l_w = (pred_w - (μ_w + 3σ_w))^2 if pred_w > μ_w + 3σ_w
            # l_w = ((μ_w - 3σ_w) - pred_w)^2 if pred_w < μ_w - 3σ_w
            # l_w = 0 otherwise
            
            upper_bound_w = mean_w + 3 * std_w
            lower_bound_w = torch.clamp(mean_w - 3 * std_w, min=0.0)
            upper_bound_h = mean_h + 3 * std_h
            lower_bound_h = torch.clamp(mean_h - 3 * std_h, min=0.0)
            
            # Width loss
            loss_w = torch.zeros_like(band_pred_sizes[:, 0:1])
            mask_w_upper = band_pred_sizes[:, 0] > upper_bound_w
            mask_w_lower = band_pred_sizes[:, 0] < lower_bound_w
            loss_w[mask_w_upper] = (band_pred_sizes[mask_w_upper, 0:1] - upper_bound_w) ** 2
            loss_w[mask_w_lower] = (lower_bound_w - band_pred_sizes[mask_w_lower, 0:1]) ** 2
            
            # Height loss
            loss_h = torch.zeros_like(band_pred_sizes[:, 1:2])
            mask_h_upper = band_pred_sizes[:, 1] > upper_bound_h
            mask_h_lower = band_pred_sizes[:, 1] < lower_bound_h
            loss_h[mask_h_upper] = (band_pred_sizes[mask_h_upper, 1:2] - upper_bound_h) ** 2
            loss_h[mask_h_lower] = (lower_bound_h - band_pred_sizes[mask_h_lower, 1:2]) ** 2
            
            # Combine width and height loss
            loss_size[mask] = loss_w + loss_h
        
        # Weighted sum
        return (loss_size * weight).sum()


class PSDDNDetectionLoss:
    """
    PSDDN Detection Loss - Modified v8DetectionLoss for point supervision.
    
    Replaces standard bbox loss with PSDDN-specific losses:
    - Center loss: Ensures predicted center matches ground truth point
    - Size loss: Constrains box sizes using local 3-sigma rule
    """
    
    def __init__(self, model, tal_topk: int = 10):
        """Initialize PSDDN detection loss."""
        # Import here to avoid circular dependency
        from ultralytics.utils.loss import v8DetectionLoss
        
        # Initialize base detection loss
        self.base_loss = v8DetectionLoss(model, tal_topk=tal_topk)
        
        # Replace bbox_loss with PSDDN version
        self.base_loss.bbox_loss = PSDDNBboxLoss(self.base_loss.reg_max).to(self.base_loss.device)
        
    def __call__(self, preds, batch):
        """Forward pass - delegates to base loss with PSDDN bbox loss."""
        return self.base_loss(preds, batch)
    
    def __getattr__(self, name):
        """Delegate attribute access to base_loss."""
        return getattr(self.base_loss, name)
