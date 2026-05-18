"""GPU/CUDA abstraction layer with automatic fallback to CPU."""

import numpy as np
from typing import Optional

# Try to import CuPy
try:
    import cupy as cp
    from cupy.cuda import memory_hooks
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    cp = None


def is_cuda_available() -> bool:
    """Check if CUDA acceleration is available."""
    return HAS_CUDA


def get_array_module():
    """Get the array module (cupy or numpy)."""
    return cp if HAS_CUDA else np


def to_gpu(array: np.ndarray) -> np.ndarray:
    """Transfer array to GPU. No-op if CUDA unavailable."""
    if HAS_CUDA:
        return cp.asarray(array)
    return array


def to_cpu(array) -> np.ndarray:
    """Transfer array from GPU to CPU. No-op if already on CPU."""
    if HAS_CUDA and hasattr(array, 'get'):
        return array.get()
    return array


def get_vram_info() -> Optional[tuple]:
    """Get GPU VRAM info: (free_bytes, total_bytes). Returns None if no CUDA."""
    if not HAS_CUDA:
        return None
    try:
        mem_info = cp.cuda.Device().mem_info
        return (mem_info[0], mem_info[1])  # free, total
    except Exception:
        return None


def estimate_batch_size(frame_shape: tuple, safety_factor: float = 3.0) -> int:
    """
    Estimate optimal batch size based on available VRAM.
    
    Args:
        frame_shape: (H, W, C) of a single frame
        safety_factor: multiplier for memory safety margin
    
    Returns:
        Recommended batch size (minimum 1)
    """
    if not HAS_CUDA:
        return 1
    
    vram_info = get_vram_info()
    if vram_info is None:
        return 1
    
    free_bytes, total_bytes = vram_info
    
    # Calculate memory per batch element:
    # - 2 input frames (int16 for subtraction): 2 * H * W * C * 2 bytes
    # - 1 output frame (uint8): H * W * C * 1 bytes
    # - Intermediate arrays (diff, mask): ~2 * H * W * C * 4 bytes
    h, w, c = frame_shape
    bytes_per_element = (2 * h * w * c * 2) + (h * w * c * 1) + (2 * h * w * c * 4)
    
    # Available with safety margin
    usable_bytes = free_bytes / safety_factor
    
    batch_size = max(1, int(usable_bytes / bytes_per_element))
    
    # Cap at reasonable maximum to avoid pipeline stalls
    return min(batch_size, 32)
