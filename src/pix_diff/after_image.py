"""After-image (motion trail) effect with fading persistence."""

import numpy as np
from enum import Enum
from typing import Optional, Tuple


class FadeMode(Enum):
    """Fade behavior for after-image trails."""
    EXPONENTIAL = "exponential"  # Multiplicative decay per frame
    FIXED = "fixed"              # Linear fade over fixed duration


class TrailMode(Enum):
    """How new changes are composited with existing trails."""
    MAX = "max"         # Keep the brightest/most intense
    ADDITIVE = "additive"  # Accumulate brightness
    REPLACE = "replace"    # Overwrite with latest


class AfterImageAccumulator:
    """
    Maintains persistent motion trails that fade over time.
    
    Trails are composited over the original diff frame output,
    creating visual echoes of pixel changes.
    """
    
    def __init__(self, frame_shape: Tuple[int, int, int],
                 fade_mode: FadeMode = FadeMode.EXPONENTIAL,
                 fade_duration: int = 10,
                 decay_factor: float = 0.85,
                 trail_mode: TrailMode = TrailMode.MAX):
        """
        Initialize after-image accumulator.
        
        Args:
            frame_shape: (H, W, C) of video frames
            fade_mode: EXPONENTIAL or FIXED
            fade_duration: Frames for fixed mode fade (default: 10)
            decay_factor: Multiplier for exponential decay (default: 0.85)
            trail_mode: How to composite new changes (default: MAX)
        """
        self.frame_shape = frame_shape
        self.fade_mode = fade_mode
        self.fade_duration = max(1, fade_duration)
        self.decay_factor = np.clip(decay_factor, 0.0, 1.0)
        self.trail_mode = trail_mode
        
        # Internal trail buffer in float32 for smooth fading
        self._trail_buffer = np.zeros(frame_shape, dtype=np.float32)
        
        # For fixed duration mode: track age and original value of each pixel's trail
        if fade_mode == FadeMode.FIXED:
            self._age_buffer = np.zeros(frame_shape[:2], dtype=np.int32)
            self._original_buffer = np.zeros(frame_shape, dtype=np.float32)
        else:
            self._age_buffer = None
            self._original_buffer = None
    
    def process(self, diff_frame: np.ndarray) -> np.ndarray:
        """
        Composite after-image trails onto a diff frame.
        
        Args:
            diff_frame: Current diff frame (H, W, 3) uint8
        
        Returns:
            Composited frame with trails (H, W, 3) uint8
        """
        # Convert diff to float32 for accumulation
        diff_float = diff_frame.astype(np.float32)
        
        # Apply fade to existing trails first
        self._apply_fade()
        
        # Update trail buffer with new changes
        self._update_trails(diff_float)
        
        # Composite: max of diff frame and trails
        # This ensures the current diff is always visible
        result = np.maximum(diff_float, self._trail_buffer)
        
        # Clip and convert back to uint8
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _update_trails(self, diff_float: np.ndarray):
        """Update trail buffer with new pixel changes."""
        # Find changed pixels (non-zero in diff frame)
        changed_mask = np.any(diff_float > 0, axis=2)
        
        if self.trail_mode == TrailMode.MAX:
            # Keep the maximum value between existing trail and new diff
            improved_mask = np.any(diff_float > self._trail_buffer, axis=2)
            self._trail_buffer = np.maximum(self._trail_buffer, diff_float)
            # Reset age and store original for newly changed pixels in fixed mode
            if self._age_buffer is not None:
                reset_mask = changed_mask | improved_mask
                self._age_buffer[reset_mask] = 0
                self._original_buffer[reset_mask] = self._trail_buffer[reset_mask]
                
        elif self.trail_mode == TrailMode.ADDITIVE:
            # Add new diff to existing trails (with saturation)
            self._trail_buffer += diff_float * changed_mask[:, :, np.newaxis]
            # Reset age and store original for newly changed pixels
            if self._age_buffer is not None:
                self._age_buffer[changed_mask] = 0
                self._original_buffer[changed_mask] = self._trail_buffer[changed_mask]
                
        elif self.trail_mode == TrailMode.REPLACE:
            # Overwrite trail with latest diff where changed
            self._trail_buffer = np.where(
                changed_mask[:, :, np.newaxis],
                diff_float,
                self._trail_buffer
            )
            # Reset age and store original for newly changed pixels
            if self._age_buffer is not None:
                self._age_buffer[changed_mask] = 0
                self._original_buffer[changed_mask] = diff_float[changed_mask]
    
    def _apply_fade(self):
        """Apply fade/decay to trails."""
        if self.fade_mode == FadeMode.EXPONENTIAL:
            # Multiplicative decay: trail *= decay_factor
            self._trail_buffer *= self.decay_factor
            
        elif self.fade_mode == FadeMode.FIXED:
            # Linear fade from original value: intensity = original * max(0, 1 - age/duration)
            self._age_buffer += 1
            
            # Calculate fade factor: 1.0 at age 0, 0.0 at age >= duration
            fade_factor = np.clip(1.0 - (self._age_buffer / self.fade_duration), 0, 1)
            
            # Apply fade per pixel using original values
            fade_3ch = fade_factor[:, :, np.newaxis]
            self._trail_buffer = self._original_buffer * fade_3ch
        
        # Zero out very faint trails to prevent accumulation of rounding errors
        self._trail_buffer[self._trail_buffer < 1.0] = 0
    
    def reset(self):
        """Clear all trails."""
        self._trail_buffer.fill(0)
        if self._age_buffer is not None:
            self._age_buffer.fill(0)
        if self._original_buffer is not None:
            self._original_buffer.fill(0)
    
    @property
    def has_trails(self) -> bool:
        """Check if any active trails exist."""
        return np.any(self._trail_buffer > 0)
