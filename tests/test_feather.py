"""Tests for feathered edges in diff engine."""

import pytest
import numpy as np

from pix_diff.diff_engine import compute_diff, DiffMode, _apply_feather


class TestApplyFeather:
    """Test feathering function."""
    
    def test_feather_zero(self):
        """Feather=0 should return binary mask unchanged."""
        mask = np.array([[False, True], [True, False]])
        result = _apply_feather(mask, 0)
        
        assert result.dtype == np.float32
        assert np.array_equal(result, mask.astype(np.float32))
    
    def test_feather_positive(self):
        """Feather > 0 should blur edges."""
        # Create a mask with a larger True region for better blur testing
        mask = np.zeros((20, 20), dtype=bool)
        mask[8:12, 8:12] = True  # 4x4 square
        
        result = _apply_feather(mask, 2)
        
        assert result.dtype == np.float32
        assert result.shape == (20, 20)
        # Center should still be strong
        assert result[10, 10] > 0.5
        # Edges should fade to 0
        assert result[0, 0] == 0.0
        assert result[19, 19] == 0.0
        # Some intermediate values should exist (smoothed)
        assert np.any((result > 0.0) & (result < 1.0))
    
    def test_feather_preserves_shape(self):
        """Feather should preserve input shape."""
        mask = np.zeros((20, 30), dtype=bool)
        mask[10:15, 10:20] = True
        
        result = _apply_feather(mask, 3)
        assert result.shape == (20, 30)


class TestComputeDiffWithFeather:
    """Test diff computation with feathering."""
    
    def test_grayscale_with_feather(self):
        """Grayscale mode with feather should have soft edges."""
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2[3:7, 3:7] = 255  # White square
        
        # Without feather (sharp edges)
        result_sharp = compute_diff(frame1, frame2, DiffMode.GRAYSCALE, feather=0)
        
        # With feather (soft edges)
        result_soft = compute_diff(frame1, frame2, DiffMode.GRAYSCALE, feather=2)
        
        # Both should show the changed region
        assert np.any(result_soft > 0)
        
        # Soft result should have intermediate values at edges
        # (not just 0 and max)
        unique_values = len(np.unique(result_soft))
        assert unique_values > 2  # More than just black/white
    
    def test_color_with_feather(self):
        """Color mode with feather should have soft edges."""
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2[3:7, 3:7] = [0, 255, 0]  # Green square in BGR
        
        result = compute_diff(frame1, frame2, DiffMode.COLOR, feather=2)
        
        # Should have intermediate values at edges
        # Check red/blue channels (should be 0 everywhere)
        assert np.all(result[:, :, 0] == 0)  # Blue channel
        assert np.all(result[:, :, 2] == 0)  # Red channel
        
        # Green channel should have gradient
        green_channel = result[:, :, 1]
        unique_green = len(np.unique(green_channel))
        assert unique_green > 2  # More than just 0 and 255
    
    def test_feather_zero_same_as_before(self):
        """Feather=0 should produce same result as no feather."""
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2[3:7, 3:7] = 255
        
        result_default = compute_diff(frame1, frame2, DiffMode.GRAYSCALE)
        result_feather0 = compute_diff(frame1, frame2, DiffMode.GRAYSCALE, feather=0)
        
        assert np.array_equal(result_default, result_feather0)
    
    def test_feather_preserves_center_intensity(self):
        """Center of large changed region should remain at full intensity."""
        frame1 = np.zeros((30, 30, 3), dtype=np.uint8)
        frame2 = np.zeros((30, 30, 3), dtype=np.uint8)
        frame2[10:20, 10:20] = 128  # Gray square (partial difference)
        
        result = compute_diff(frame1, frame2, DiffMode.GRAYSCALE, feather=2)
        
        # Center should still be bright (128/255 diff → ~127 intensity)
        center = result[15, 15, 0]
        assert center > 100  # Should be reasonably bright
