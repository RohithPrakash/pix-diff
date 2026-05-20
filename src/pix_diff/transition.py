"""Transition effect controller for smooth crossfade from original to diff."""

import numpy as np
from typing import Optional, Tuple


class TransitionController:
    """
    Manages temporal crossfade from original video to pix-diff visualization.
    
    Three phases:
    1. Original: Show original frames (alpha=0)
    2. Transition: Smooth ease-in-out blend (alpha 0→1)
    3. Diff: Show pure diff frames (alpha=1)
    """
    
    def __init__(self, fps: float, delay_sec: float = 0.0, duration_sec: float = 3.0):
        """
        Initialize transition controller.
        
        Args:
            fps: Video frames per second
            delay_sec: Seconds to show original before transition starts
            duration_sec: Duration of transition in seconds
        """
        self.fps = fps
        self._duration_sec = duration_sec
        self.delay_frames = int(delay_sec * fps)
        self.duration_frames = max(1, int(duration_sec * fps))
        self.total_transition_frames = self.delay_frames + self.duration_frames
    
    def get_alpha(self, frame_idx: int) -> float:
        """
        Get blend alpha for a given frame index.
        
        Args:
            frame_idx: Index of diff frame (0 to k-2)
        
        Returns:
            Alpha value 0.0 (original) to 1.0 (diff)
        """
        if frame_idx < self.delay_frames:
            # Original phase
            return 0.0
        elif frame_idx >= self.total_transition_frames:
            # Diff phase
            return 1.0
        else:
            # Transition phase: smooth ease-in-out
            t = (frame_idx - self.delay_frames) / self.duration_frames
            return self._ease_in_out(t)
    
    def _ease_in_out(self, t: float) -> float:
        """
        Smoothstep ease-in-out function.
        
        Args:
            t: Input 0.0 to 1.0
        
        Returns:
            Smoothed value 0.0 to 1.0
        """
        # Clamp to [0, 1]
        t = max(0.0, min(1.0, t))
        # Smoothstep: 3t² - 2t³
        return t * t * (3.0 - 2.0 * t)
    
    def blend(self, original: np.ndarray, diff: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Blend original and diff frames based on transition alpha.
        
        Args:
            original: Original video frame (H, W, 3) uint8
            diff: Diff visualization frame (H, W, 3) uint8
            frame_idx: Frame index for alpha calculation
        
        Returns:
            Blended frame (H, W, 3) uint8
        """
        alpha = self.get_alpha(frame_idx)
        
        if alpha <= 0.0:
            return original
        elif alpha >= 1.0:
            return diff
        else:
            # Linear blend: original * (1-alpha) + diff * alpha
            # Convert to float for precision, then back to uint8
            blended = original.astype(np.float32) * (1.0 - alpha) + diff.astype(np.float32) * alpha
            return np.clip(blended, 0, 255).astype(np.uint8)
    
    def is_active(self) -> bool:
        """Check if transition is active (duration > 0)."""
        return self._duration_sec > 0
    
    def __repr__(self) -> str:
        delay_sec = self.delay_frames / self.fps
        duration_sec = self.duration_frames / self.fps
        return f"TransitionController(delay={delay_sec:.1f}s, duration={duration_sec:.1f}s)"
