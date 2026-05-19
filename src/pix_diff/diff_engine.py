"""Pixel difference engine for generating diff frames."""

import numpy as np
import cv2
from enum import Enum
from typing import Optional, Callable, Tuple

from .metrics import per_channel_metric, get_metric_range


class DiffMode(Enum):
    """Modes for pixel difference visualization."""
    GRAYSCALE = "grayscale"
    COLOR = "color"


def compute_diff(frame1: np.ndarray, frame2: np.ndarray, 
                 mode: DiffMode, threshold: int = 0,
                 metric_fn: Optional[Callable] = None,
                 metric_name: str = 'per_channel',
                 feather: int = 0) -> np.ndarray:
    """
    Compute pixel difference between two frames.
    
    Args:
        frame1: Previous frame (H, W, 3) uint8 BGR
        frame2: Current frame (H, W, 3) uint8 BGR
        mode: DiffMode.GRAYSCALE or DiffMode.COLOR
        threshold: Minimum difference to consider changed (0-255)
        metric_fn: Optional metric function. Defaults to per_channel_metric
        metric_name: Name of metric for range normalization
        feather: Gaussian blur radius for smoothing edges (0=off)
    
    Returns:
        Diff frame (H, W, 3) uint8 BGR
    """
    # Ensure same shape
    if frame1.shape != frame2.shape:
        raise ValueError(f"Frame shapes don't match: {frame1.shape} vs {frame2.shape}")
    
    # Use provided metric or default
    if metric_fn is None:
        metric_fn = per_channel_metric
    
    # Get magnitude and changed mask from metric
    magnitude, changed_mask = metric_fn(frame1, frame2, threshold)
    
    # Apply feathering (Gaussian blur) to soften edges
    soft_mask = _apply_feather(changed_mask, feather)
    
    if mode == DiffMode.GRAYSCALE:
        return _grayscale_mode(magnitude, soft_mask, metric_name)
    elif mode == DiffMode.COLOR:
        return _color_mode(frame2, soft_mask)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _apply_feather(changed_mask: np.ndarray, feather: int) -> np.ndarray:
    """
    Apply Gaussian blur to the change mask for soft edges.
    
    Args:
        changed_mask: Boolean mask (H, W)
        feather: Blur radius in pixels (0=no blur)
    
    Returns:
        Soft mask (H, W) float32 in range [0.0, 1.0]
    """
    if feather <= 0:
        return changed_mask.astype(np.float32)
    
    # Convert bool to float32 (0.0 or 1.0)
    mask_float = changed_mask.astype(np.float32)
    
    # Apply Gaussian blur
    # Kernel size = radius*2+1 (must be odd)
    kernel_size = feather * 2 + 1
    blurred = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), 0)
    
    return blurred


def _grayscale_mode(magnitude: np.ndarray, soft_mask: np.ndarray,
                    metric_name: str = 'per_channel') -> np.ndarray:
    """
    Mode 1: Changed pixels are white with intensity based on difference.
    Greater difference = brighter white. Unchanged = black.
    
    Intensity is normalized based on the metric's expected range.
    """
    # Get metric range for normalization
    min_val, max_val = get_metric_range(metric_name)
    
    # Normalize magnitude to 0-255 range
    # intensity = 255 * (1 - magnitude / max_val)
    # So max difference → black (0), min difference → white (255)
    normalized = np.clip(magnitude / max_val, 0, 1)
    intensity = 255 * (1 - normalized)
    
    # Apply soft mask: feathered edges fade smoothly
    intensity = intensity * soft_mask
    
    # Convert to uint8
    intensity = intensity.astype(np.uint8)
    
    # Stack to 3-channel grayscale (BGR)
    result = np.stack([intensity, intensity, intensity], axis=2)
    return result


def _color_mode(frame2: np.ndarray, soft_mask: np.ndarray) -> np.ndarray:
    """
    Mode 2: Changed pixels keep original frame 2 color.
    Unchanged pixels are black.
    
    With feathering, edges transition smoothly.
    """
    # Apply soft mask: pixel values scale by mask intensity
    # This creates smooth transitions at boundaries
    result = frame2.astype(np.float32) * soft_mask[:, :, np.newaxis]
    return result.astype(np.uint8)
