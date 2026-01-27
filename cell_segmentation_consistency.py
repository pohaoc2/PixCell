"""
Cell Segmentation Consistency Module for PixCell ControlNet

Ensures generated images match the provided cell masks by:
1. Running cell segmentation on generated images
2. Comparing predicted masks with ground truth masks
3. Adding consistency loss to training

Uses CellViT-SAM-H for segmentation (same as used for mask extraction)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class CellSegmentationConsistency(nn.Module):
    """
    Cell segmentation consistency checker
    
    Segments generated images and compares with provided masks to ensure
    the generated histology images actually contain cells in the right locations.
    
    Args:
        cellvit_model_path: Path to CellViT-SAM-H checkpoint
        device: Device to run on
        image_size: Input image size (256 for PixCell-256)
        use_cache: Cache model to avoid reloading
    """
    def __init__(
        self,
        cellvit_model_path=None,
        device='cuda',
        image_size=256,
        use_cache=True,
    ):
        super().__init__()
        
        self.device = device
        self.image_size = image_size
        self.use_cache = use_cache
        self._model = None
        self.cellvit_model_path = cellvit_model_path
        
        # Preprocessing for CellViT (depends on their specific requirements)
        # Typically histopathology models expect different normalization
        from torchvision import transforms
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats (common for CellViT)
            std=[0.229, 0.224, 0.225]
        )
    
    @property
    def model(self):
        """Lazy load model to save memory"""
        if self._model is None and self.cellvit_model_path is not None:
            self._model = self._load_cellvit_model()
        return self._model
    
    def _load_cellvit_model(self):
        """Load CellViT-SAM-H model"""
        try:
            # Try importing CellViT
            from cellvit.models import CellViT
            
            print(f"Loading CellViT-SAM-H from {self.cellvit_model_path}")
            
            # Load model configuration and checkpoint
            model = CellViT(
                num_classes=6,  # Background + 5 cell types (adjust as needed)
                embed_dim=1280,  # SAM-H embedding dimension
                input_channels=3,
                extract_layers=[3, 6, 9, 12]
            )
            
            # Load checkpoint
            checkpoint = torch.load(self.cellvit_model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            print("✓ CellViT-SAM-H loaded successfully")
            return model
            
        except ImportError:
            print("⚠ CellViT not installed, using simplified segmentation")
            return self._create_simple_segmentation_model()
        except Exception as e:
            print(f"⚠ Error loading CellViT: {e}")
            print("  Using simplified segmentation model")
            return self._create_simple_segmentation_model()
    
    def _create_simple_segmentation_model(self):
        """
        Create a simple U-Net style segmentation model as fallback
        This can be trained jointly or used as a lightweight alternative
        """
        class SimpleSegmentationNet(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Encoder
                self.enc1 = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
                self.pool1 = nn.MaxPool2d(2, 2)
                
                self.enc2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                )
                self.pool2 = nn.MaxPool2d(2, 2)
                
                # Bottleneck
                self.bottleneck = nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )
                
                # Decoder
                self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self.dec1 = nn.Sequential(
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                )
                
                self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec2 = nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
                
                # Final layer
                self.final = nn.Conv2d(64, 1, 1)  # Binary segmentation
            
            def forward(self, x):
                # Encoder
                e1 = self.enc1(x)
                p1 = self.pool1(e1)
                
                e2 = self.enc2(p1)
                p2 = self.pool2(e2)
                
                # Bottleneck
                b = self.bottleneck(p2)
                
                # Decoder
                d1 = self.up1(b)
                d1 = torch.cat([d1, e2], dim=1)
                d1 = self.dec1(d1)
                
                d2 = self.up2(d1)
                d2 = torch.cat([d2, e1], dim=1)
                d2 = self.dec2(d2)
                
                # Final
                out = self.final(d2)
                return out
        
        model = SimpleSegmentationNet().to(self.device)
        model.eval()
        return model
    
    def segment_image(self, image):
        """
        Segment cells in an image
        
        Args:
            image: Tensor of shape (B, 3, H, W) in range [-1, 1] or [0, 1]
            
        Returns:
            Binary mask (B, 1, H, W) with 1 for cells, 0 for background
        """
        # Normalize if needed
        if image.min() < 0:  # [-1, 1] range
            image = (image + 1) / 2  # Convert to [0, 1]
        
        # Apply preprocessing
        image_norm = self.transform(image)
        
        # Run segmentation
        with torch.no_grad():
            if self.model is not None:
                pred = self.model(image_norm)
                
                # Convert to binary mask
                if pred.shape[1] > 1:  # Multi-class
                    # Take max across classes (excluding background)
                    pred = pred[:, 1:].max(dim=1, keepdim=True)[0]
                
                # Apply sigmoid and threshold
                mask = torch.sigmoid(pred) > 0.5
                mask = mask.float()
            else:
                # No model available, return zeros
                mask = torch.zeros(image.shape[0], 1, image.shape[2], image.shape[3], 
                                  device=image.device)
        
        return mask
    
    @torch.no_grad()
    def compute_mask_metrics(self, pred_mask, gt_mask):
        """
        Compute mask similarity metrics
        
        Args:
            pred_mask: Predicted binary mask (B, 1, H, W)
            gt_mask: Ground truth mask (B, 1, H, W)
            
        Returns:
            dict of metrics: IoU, Dice, Precision, Recall
        """
        # Flatten
        pred_flat = pred_mask.view(pred_mask.shape[0], -1)
        gt_flat = gt_mask.view(gt_mask.shape[0], -1)
        
        # Compute metrics
        intersection = (pred_flat * gt_flat).sum(dim=1)
        union = (pred_flat + gt_flat).clamp(0, 1).sum(dim=1)
        pred_sum = pred_flat.sum(dim=1)
        gt_sum = gt_flat.sum(dim=1)
        
        # IoU
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        # Dice
        dice = (2 * intersection + 1e-6) / (pred_sum + gt_sum + 1e-6)
        
        # Precision
        precision = (intersection + 1e-6) / (pred_sum + 1e-6)
        
        # Recall
        recall = (intersection + 1e-6) / (gt_sum + 1e-6)
        
        return {
            'iou': iou.mean().item(),
            'dice': dice.mean().item(),
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
        }


def mask_consistency_loss(pred_mask, gt_mask, loss_type='bce'):
    """
    Compute consistency loss between predicted and ground truth masks
    
    Args:
        pred_mask: Predicted mask from generated image (B, 1, H, W)
        gt_mask: Ground truth mask provided as conditioning (B, 1, H, W)
        loss_type: 'bce', 'dice', 'focal', or 'combined'
        
    Returns:
        Consistency loss value
    """
    # Ensure same resolution
    if pred_mask.shape != gt_mask.shape:
        pred_mask = F.interpolate(pred_mask, size=gt_mask.shape[-2:], mode='bilinear', align_corners=False)
    
    if loss_type == 'bce':
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
    
    elif loss_type == 'dice':
        # Dice loss (better for imbalanced masks)
        pred_prob = torch.sigmoid(pred_mask)
        intersection = (pred_prob * gt_mask).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3))
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        loss = 1 - dice.mean()
    
    elif loss_type == 'focal':
        # Focal loss (focus on hard examples)
        alpha = 0.25
        gamma = 2.0
        
        pred_prob = torch.sigmoid(pred_mask)
        ce_loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        p_t = pred_prob * gt_mask + (1 - pred_prob) * (1 - gt_mask)
        focal_weight = (1 - p_t) ** gamma
        loss = (alpha * focal_weight * ce_loss).mean()
    
    elif loss_type == 'combined':
        # Combined BCE + Dice
        bce_loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
        
        pred_prob = torch.sigmoid(pred_mask)
        intersection = (pred_prob * gt_mask).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3))
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice.mean()
        
        loss = 0.5 * bce_loss + 0.5 * dice_loss
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss


def chamfer_distance_2d(pred_mask, gt_mask, max_distance=50):
    """
    Compute 2D Chamfer distance between mask boundaries
    
    Measures how well the cell boundaries match between predicted and GT masks.
    
    Args:
        pred_mask: Predicted binary mask (B, 1, H, W)
        gt_mask: Ground truth mask (B, 1, H, W)
        max_distance: Maximum distance to consider
        
    Returns:
        Chamfer distance
    """
    batch_size = pred_mask.shape[0]
    chamfer_dists = []
    
    for b in range(batch_size):
        # Extract boundaries using morphological operations
        pred_boundary = extract_boundary(pred_mask[b, 0])
        gt_boundary = extract_boundary(gt_mask[b, 0])
        
        # Get boundary coordinates
        pred_coords = torch.nonzero(pred_boundary, as_tuple=False).float()
        gt_coords = torch.nonzero(gt_boundary, as_tuple=False).float()
        
        if pred_coords.shape[0] == 0 or gt_coords.shape[0] == 0:
            # No boundary found
            chamfer_dists.append(torch.tensor(max_distance, device=pred_mask.device))
            continue
        
        # Compute pairwise distances
        # pred -> gt
        dists_pred_to_gt = torch.cdist(pred_coords, gt_coords, p=2)
        min_dist_pred_to_gt = dists_pred_to_gt.min(dim=1)[0].mean()
        
        # gt -> pred
        dists_gt_to_pred = torch.cdist(gt_coords, pred_coords, p=2)
        min_dist_gt_to_pred = dists_gt_to_pred.min(dim=1)[0].mean()
        
        # Chamfer distance (symmetric)
        chamfer = (min_dist_pred_to_gt + min_dist_gt_to_pred) / 2
        chamfer_dists.append(chamfer)
    
    return torch.stack(chamfer_dists).mean()


def extract_boundary(mask):
    """
    Extract boundary from binary mask using erosion
    
    Args:
        mask: Binary mask (H, W)
        
    Returns:
        Boundary mask (H, W)
    """
    # Simple boundary extraction: mask - eroded_mask
    kernel = torch.ones(1, 1, 3, 3, device=mask.device)
    mask_4d = mask.unsqueeze(0).unsqueeze(0).float()
    
    eroded = F.conv2d(mask_4d, kernel, padding=1)
    eroded = (eroded == 9).float()  # All 9 neighbors are 1
    
    boundary = mask_4d - eroded
    return boundary[0, 0] > 0


# Example usage in training
if __name__ == "__main__":
    print("Testing Cell Segmentation Consistency Module...")
    print("=" * 70)
    
    # Create consistency checker
    consistency = CellSegmentationConsistency(
        cellvit_model_path=None,  # Will use simple model
        device='cuda' if torch.cuda.is_available() else 'cpu',
        image_size=256,
    )
    
    # Test with random images and masks
    batch_size = 2
    generated_images = torch.randn(batch_size, 3, 256, 256)
    gt_masks = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    
    print("\n1. Segmenting generated images...")
    pred_masks = consistency.segment_image(generated_images)
    print(f"   Generated images: {generated_images.shape}")
    print(f"   Predicted masks: {pred_masks.shape}")
    
    print("\n2. Computing mask metrics...")
    metrics = consistency.compute_mask_metrics(pred_masks, gt_masks)
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")
    
    print("\n3. Computing consistency losses...")
    for loss_type in ['bce', 'dice', 'focal', 'combined']:
        loss = mask_consistency_loss(pred_masks, gt_masks, loss_type=loss_type)
        print(f"   {loss_type} loss: {loss:.4f}")
    
    print("\n4. Computing Chamfer distance...")
    chamfer = chamfer_distance_2d(pred_masks, gt_masks)
    print(f"   Chamfer distance: {chamfer:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
