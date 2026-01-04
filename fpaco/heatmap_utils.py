import numpy as np
import torch

def generate_gaussian_heatmap(h, w, boxes, scores=None, sigma=15):
    """
    Step 3: Generate heatmap from BBoxes.
    Args:
        h, w: Height and Width of the output heatmap.
        boxes: Tensor/List of [x1, y1, x2, y2].
        scores: Optional confidence scores for each box.
        sigma: Standard deviation for the Gaussian.
    Returns:
        heatmap: [H, W] np.float32 array.
    """
    heatmap = np.zeros((h, w), dtype=np.float32)
    if len(boxes) == 0:
        return heatmap

    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        score = scores[i].item() if scores is not None else 1.0
        
        # Distance squared to center
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        # Gaussian blob
        blob = np.exp(-dist_sq / (2 * sigma**2))
        
        # Aggregate using maximum
        heatmap = np.maximum(heatmap, blob * score)
        
    return heatmap

def boxes_to_mask_heatmap(h, w, boxes, scores=None):
    """
    Alternative: Hard mask for BBox area.
    """
    heatmap = np.zeros((h, w), dtype=np.float32)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        score = scores[i].item() if scores is not None else 1.0
        heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], score)
    return heatmap
