"""Pixel comparison metrics for diff computation."""

import numpy as np
import cv2
from typing import Callable, Tuple

from .gpu_backend import get_array_module, is_cuda_available, to_cpu

# Rec.709 luminance coefficients
REC709_COEFFS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Type alias for metric functions
MetricFn = Callable[[np.ndarray, np.ndarray, int], Tuple[np.ndarray, np.ndarray]]


def per_channel_metric(frame1: np.ndarray, frame2: np.ndarray, 
                       threshold: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-channel absolute difference (original behavior).
    
    Returns:
        magnitude: Average per-channel difference (H, W)
        changed: Boolean mask (H, W)
    """
    f1 = frame1.astype(np.int16)
    f2 = frame2.astype(np.int16)
    diff = np.abs(f2 - f1)  # (H, W, 3)
    
    magnitude = np.mean(diff, axis=2).astype(np.float32)
    changed = np.any(diff > threshold, axis=2)
    
    return magnitude, changed


def euclidean_metric(frame1: np.ndarray, frame2: np.ndarray,
                     threshold: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    3D Euclidean distance in RGB space.
    
    Range: 0 to sqrt(3*255^2) ≈ 441.67
    """
    f1 = frame1.astype(np.float32)
    f2 = frame2.astype(np.float32)
    diff = f2 - f1
    
    magnitude = np.sqrt(np.sum(diff ** 2, axis=2))
    changed = magnitude > threshold
    
    return magnitude, changed


def luminance_metric(frame1: np.ndarray, frame2: np.ndarray,
                     threshold: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rec.709 luminance difference only.
    
    Range: 0 to 255
    """
    coeffs = REC709_COEFFS
    
    f1 = frame1.astype(np.float32)
    f2 = frame2.astype(np.float32)
    
    y1 = np.sum(f1 * coeffs, axis=2)
    y2 = np.sum(f2 * coeffs, axis=2)
    
    magnitude = np.abs(y2 - y1)
    changed = magnitude > threshold
    
    return magnitude, changed


def weighted_rgb_metric(frame1: np.ndarray, frame2: np.ndarray,
                        threshold: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Luminance-weighted RGB difference (Rec.709 weights).
    
    Range: 0 to 255
    """
    coeffs = REC709_COEFFS
    
    f1 = frame1.astype(np.float32)
    f2 = frame2.astype(np.float32)
    
    diff = np.abs(f2 - f1)
    weighted = diff * coeffs
    
    magnitude = np.sum(weighted, axis=2)
    changed = magnitude > threshold
    
    return magnitude, changed


def rgb_to_lab_cpu(rgb: np.ndarray) -> np.ndarray:
    """Convert BGR (uint8) to Lab (float32) on CPU using OpenCV."""
    # OpenCV cvtColor expects BGR input, outputs Lab with:
    # L: 0-100, a: -128 to 127, b: -128 to 127
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2Lab).astype(np.float32)
    return lab


def rgb_to_lab_gpu(rgb: np.ndarray) -> np.ndarray:
    """Convert BGR (uint8) to Lab (float32) on GPU using CuPy."""
    import cupy as cp
    
    # Convert to float32 and normalize to 0-1
    rgb_f = rgb.astype(cp.float32) / 255.0
    
    # BGR to XYZ (D65) matrix
    # Source: sRGB D65
    mat = cp.array([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ], dtype=cp.float32)
    
    # Apply matrix
    xyz = cp.dot(rgb_f, mat.T)
    
    # D65 white point
    xn, yn, zn = 95.047, 100.0, 108.883
    
    # Normalize by white point
    xyz[:, :, 0] /= xn
    xyz[:, :, 1] /= yn
    xyz[:, :, 2] /= zn
    
    # XYZ to Lab
    def f(t):
        return cp.where(t > 0.008856, 
                       cp.power(t, 1.0/3.0),
                       7.787 * t + 16.0/116.0)
    
    fx = f(xyz[:, :, 0])
    fy = f(xyz[:, :, 1])
    fz = f(xyz[:, :, 2])
    
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    
    lab = cp.stack([L, a, b], axis=2)
    return lab


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Route to CPU or GPU implementation."""
    if is_cuda_available():
        try:
            return rgb_to_lab_gpu(rgb)
        except Exception:
            # Fall back to CPU if GPU fails
            pass
    return rgb_to_lab_cpu(rgb)


def delta_e_cie76_metric(frame1: np.ndarray, frame2: np.ndarray,
                         threshold: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    CIE76 Delta E (Euclidean distance in Lab space).
    
    Range: 0 to ~100+
    Threshold interpretation: perceptual difference units
    """
    lab1 = rgb_to_lab(frame1)
    lab2 = rgb_to_lab(frame2)
    
    diff = lab2 - lab1
    magnitude = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Convert threshold to float for comparison
    changed = magnitude > float(threshold)
    
    return magnitude, changed


def delta_e_cie94_metric(frame1: np.ndarray, frame2: np.ndarray,
                         threshold: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full CIE94 Delta E with graphic arts parametric weights.
    
    Uses kL=1, kC=1, kH=1 (graphic arts standard).
    
    Range: 0 to ~100+
    Threshold interpretation: perceptual difference units
    """
    lab1 = rgb_to_lab(frame1)
    lab2 = rgb_to_lab(frame2)
    
    # Convert to CPU for complex math if on GPU
    if hasattr(lab1, 'get'):
        lab1 = lab1.get()
        lab2 = lab2.get()
    
    L1, a1, b1 = lab1[:, :, 0], lab1[:, :, 1], lab1[:, :, 2]
    L2, a2, b2 = lab2[:, :, 0], lab2[:, :, 1], lab2[:, :, 2]
    
    # Delta L
    dL = L2 - L1
    
    # C1, C2 (chroma)
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    dC = C2 - C1
    
    # Delta H
    da = a2 - a1
    db = b2 - b1
    dH_sq = da**2 + db**2 - dC**2
    dH_sq = np.maximum(dH_sq, 0)  # Numerical safety
    dH = np.sqrt(dH_sq)
    
    # Parametric weights
    kL, kC, kH = 1.0, 1.0, 1.0
    
    # Sl, Sc, Sh
    Sl = 1.0
    Sc = 1.0 + 0.045 * C1
    Sh = 1.0 + 0.015 * C1
    
    # CIE94 formula
    term1 = (dL / (kL * Sl))**2
    term2 = (dC / (kC * Sc))**2
    term3 = (dH / (kH * Sh))**2
    
    magnitude = np.sqrt(term1 + term2 + term3)
    changed = magnitude > float(threshold)
    
    return magnitude, changed


# Registry of available metrics
METRIC_REGISTRY = {
    'per_channel': per_channel_metric,
    'euclidean': euclidean_metric,
    'luminance': luminance_metric,
    'weighted_rgb': weighted_rgb_metric,
    'delta_e_cie76': delta_e_cie76_metric,
    'delta_e_cie94': delta_e_cie94_metric,
}


def get_metric(name: str) -> MetricFn:
    """Get metric function by name."""
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}. Available: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[name]


def get_metric_range(name: str) -> Tuple[float, float]:
    """Get expected range for a metric."""
    ranges = {
        'per_channel': (0.0, 255.0),
        'euclidean': (0.0, 441.67),
        'luminance': (0.0, 255.0),
        'weighted_rgb': (0.0, 255.0),
        'delta_e_cie76': (0.0, 100.0),
        'delta_e_cie94': (0.0, 100.0),
    }
    return ranges.get(name, (0.0, 255.0))
