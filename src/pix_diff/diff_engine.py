"""Pixel difference engine for generating diff frames."""

import numpy as np
from enum import Enum


class DiffMode(Enum):
    """Modes for pixel difference visualization."""
    GRAYSCALE = "grayscale"
    COLOR = "color"


def compute_diff(frame1: np.ndarray, frame2: np.ndarray, 
                 mode: DiffMode, threshold: int = 0) -> np.ndarray:
    """
    Compute pixel difference between two frames.
    
    Args:
        frame1: Previous frame (H, W, 3) uint8 BGR
        frame2: Current frame (H, W, 3) uint8 BGR
        mode: DiffMode.GRAYSCALE or DiffMode.COLOR
        threshold: Minimum per-channel difference to consider changed (0-255)
    
    Returns:
        Diff frame (H, W, 3) uint8 BGR
    """
    # Ensure same shape
    if frame1.shape != frame2.shape:
        raise ValueError(f"Frame shapes don't match: {frame1.shape} vs {frame2.shape}")
    
    # Cast to int16 to avoid overflow on subtraction
    f1 = frame1.astype(np.int16)
    f2 = frame2.astype(np.int16)
    
    # Per-channel absolute difference
    diff = np.abs(f2 - f1)  # (H, W, 3) int16
    
    # Create mask: pixels changed if ANY channel exceeds threshold
    changed_mask = np.any(diff > threshold, axis=2)  # (H, W) bool
    
    if mode == DiffMode.GRAYSCALE:
        return _grayscale_mode(f2, diff, changed_mask)
    elif mode == DiffMode.COLOR:
        return _color_mode(f2, changed_mask)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _grayscale_mode(frame2: np.ndarray, diff: np.ndarray, 
                    changed_mask: np.ndarray) -> np.ndarray:
    """
    Mode 1: Changed pixels are white with intensity based on difference.
    Greater difference = brighter white. Unchanged = black.
    
    Uses per-channel average difference inverted: 
    avg_diff=0 (no change) → black, avg_diff=255 (max change) → white
    """
    # Average per-channel difference
    avg_diff = np.mean(diff, axis=2)  # (H, W) float64
    
    # Invert: greater difference = brighter (255 - diff would make greater diff = darker)
    # Actually user said: "greater difference lesser the brightness"
    # So: max diff (255) → black (0), min diff (0) → white (255)
    # intensity = 255 - avg_diff
    intensity = 255 - avg_diff
    
    # Apply mask: unchanged pixels stay black
    intensity = np.where(changed_mask, intensity, 0)
    
    # Clip and convert to uint8
    intensity = np.clip(intensity, 0, 255).astype(np.uint8)
    
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
