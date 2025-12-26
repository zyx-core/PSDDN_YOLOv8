"""
Test script for PSDDN custom loss implementation.

This script verifies that the PSDDN loss functions work correctly
with dummy data before using them in actual training.
"""

import sys
from pathlib import Path
import torch

# Add ultralytics to path
sys.path.insert(0, str(Path(__file__).parent.parent / "ultralytics_repo"))

from ultralytics.utils.psddn_loss import PSDDNBboxLoss


def test_psddn_bbox_loss():
    """Test PSDDN bbox loss with dummy data."""
    print("=" * 60)
    print("Testing PSDDN Bbox Loss")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Create loss module
    loss_fn = PSDDNBboxLoss(reg_max=16).to(device)
    
    # Create dummy data
    batch_size = 2
    num_anchors = 100
    
    # Predicted boxes (xyxy format) - requires grad for backward test
    pred_bboxes = torch.rand(batch_size, num_anchors, 4, device=device, requires_grad=True) * 100
    pred_bboxes_data = pred_bboxes.detach().clone()
    pred_bboxes_data[:, :, 2:] += pred_bboxes_data[:, :, :2]  # Ensure x2 > x1, y2 > y1
    pred_bboxes = pred_bboxes_data.requires_grad_(True)
    
    # Target boxes (pseudo GT in xyxy format)
    target_bboxes = torch.rand(batch_size, num_anchors, 4, device=device) * 100
    target_bboxes[:, :, 2:] += target_bboxes[:, :, :2]
    
    # Anchor points
    anchor_points = torch.rand(num_anchors, 2, device=device) * 100
    
    # Predicted distribution (not used in PSDDN)
    pred_dist = torch.rand(batch_size, num_anchors, 64, device=device)
    
    # Target scores (confidence)
    target_scores = torch.rand(batch_size, num_anchors, 1, device=device)
    target_scores_sum = target_scores.sum()
    
    # Foreground mask (only first 50 anchors are positive)
    fg_mask = torch.zeros(batch_size, num_anchors, dtype=torch.bool, device=device)
    fg_mask[:, :50] = True
    
    print(f"Batch size: {batch_size}")
    print(f"Num anchors: {num_anchors}")
    print(f"Num positive samples: {fg_mask.sum().item()}\n")
    
    # Forward pass
    try:
        loss_bbox, loss_dfl = loss_fn(
            pred_dist=pred_dist,
            pred_bboxes=pred_bboxes,
            anchor_points=anchor_points,
            target_bboxes=target_bboxes,
            target_scores=target_scores,
            target_scores_sum=target_scores_sum,
            fg_mask=fg_mask
        )
        
        print(f"✓ Loss computation successful!")
        print(f"  Bbox loss: {loss_bbox.item():.4f}")
        print(f"  DFL loss: {loss_dfl.item():.4f}")
        print(f"  Total loss: {(loss_bbox + loss_dfl).item():.4f}\n")
        
        # Test backward pass
        total_loss = loss_bbox + loss_dfl
        total_loss.backward()
        print(f"✓ Backward pass successful!\n")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during loss computation:")
        print(f"  {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_3sigma_rule():
    """Test the 3-sigma size constraint specifically."""
    print("=" * 60)
    print("Testing 3-Sigma Size Constraint")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = PSDDNBboxLoss(reg_max=16).to(device)
    
    # Create controlled test case
    # All targets have width=10, height=10 (std will be ~0)
    N = 20
    pred_sizes = torch.tensor([
        [10.0, 10.0],  # Within bounds
        [10.5, 10.5],  # Within bounds
        [50.0, 50.0],  # Way outside bounds (should be penalized)
        [2.0, 2.0],    # Way outside bounds (should be penalized)
    ] + [[10.0, 10.0]] * (N - 4), device=device)
    
    target_sizes = torch.ones(N, 2, device=device) * 10.0
    y_coords = torch.linspace(0, 100, N, device=device)
    weight = torch.ones(N, 1, device=device)
    
    loss = loss_fn.calculate_size_loss_3sigma(
        pred_sizes, target_sizes, y_coords, weight, num_bands=1
    )
    
    print(f"Target sizes: all [10.0, 10.0]")
    print(f"Predicted sizes (first 4): {pred_sizes[:4].tolist()}")
    print(f"Size loss: {loss.item():.4f}")
    print(f"✓ 3-sigma constraint test completed\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PSDDN Loss Implementation Test Suite")
    print("=" * 60 + "\n")
    
    # Test 1: Basic bbox loss
    test1_pass = test_psddn_bbox_loss()
    
    # Test 2: 3-sigma rule
    test_3sigma_rule()
    
    print("=" * 60)
    if test1_pass:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED. Please review the errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
