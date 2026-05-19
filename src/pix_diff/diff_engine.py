"""Pixel difference engine for generating diff frames."""

import numpy as np
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
                 metric_name: str = 'per_channel') -> np.ndarray:
    """
    Compute pixel difference between two frames.
    
    Args:
        frame1: Previous frame (H, W, 3) uint8 BGR
        frame2: Current frame (H, W, 3) uint8 BGR
        mode: DiffMode.GRAYSCALE or DiffMode.COLOR
        threshold: Minimum difference to consider changed (0-255)
        metric_fn: Optional metric function. Defaults to per_channel_metric
        metric_name: Name of metric for range normalization
    
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
    
    if mode == DiffMode.GRAYSCALE:
        return _grayscale_mode(magnitude, changed_mask, metric_name)
    elif mode == DiffMode.COLOR:
        return _color_mode(frame2, changed_mask)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _grayscale_mode(magnitude: np.ndarray, changed_mask: np.ndarray,
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
    
    # Apply mask: unchanged pixels stay black
    intensity = np.where(changed_mask, intensity, 0)
    
    # Convert to uint8
    intensity = intensity.astype(np.uint8)
    
    # Stack to 3-channel grayscale (BGR)
    result = np.stack([intensity, intensity, intensity], axis=2)
    return result


def _color_mode(frame2: np.ndarray, changed_mask: np.ndarray) -> np.ndarray:
    """
    Mode 2: Changed pixels keep original frame 2 color.
    Unchanged pixels are black.
    """
    # Apply mask: keep frame2 where changed, black where unchanged
    result = np.where(changed_mask[:, :, np.newaxis], frame2, 0)
    return result.astype(np.uint8)
