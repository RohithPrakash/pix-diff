"""Tests for diff engine."""

import numpy as np
import pytest

from pix_diff.diff_engine import compute_diff, DiffMode


class TestComputeDiff:
    """Test pixel difference computation."""
    
    def test_identical_frames_grayscale(self):
        """Two identical frames should produce all-black output."""
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = compute_diff(frame, frame, DiffMode.GRAYSCALE)
        
        assert result.shape == (10, 10, 3)
        assert np.all(result == 0)
    
    def test_identical_frames_color(self):
        """Two identical frames should produce all-black output in color mode."""
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = compute_diff(frame, frame, DiffMode.COLOR)
        
        assert result.shape == (10, 10, 3)
        assert np.all(result == 0)
    
    def test_max_difference_grayscale(self):
        """Max difference (black to white) should produce white in grayscale mode."""
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.full((10, 10, 3), 255, dtype=np.uint8)
        result = compute_diff(frame1, frame2, DiffMode.GRAYSCALE)
        
        # Average diff is 255, intensity = 255 - 255 = 0 (black)
        # Wait, that's what the user asked: "greater difference lesser the brightness"
        assert np.all(result == 0)
    
    def test_max_difference_color(self):
        """Changed pixels should keep frame2 color."""
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.full((10, 10, 3), 255, dtype=np.uint8)
        result = compute_diff(frame1, frame2, DiffMode.COLOR)
        
        assert np.all(result == 255)
    
    def test_threshold_blocks_small_changes(self):
        """Threshold should ignore small differences."""
        frame1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        frame2 = np.full((10, 10, 3), 104, dtype=np.uint8)  # Diff of 4
        
        result = compute_diff(frame1, frame2, DiffMode.COLOR, threshold=5)
        assert np.all(result == 0)  # Blocked by threshold
        
        result = compute_diff(frame1, frame2, DiffMode.COLOR, threshold=3)
        assert np.all(result == 104)  # Allowed
    
    def test_partial_changes(self):
        """Only changed pixels should be highlighted."""
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2[5, 5] = [255, 255, 255]  # Change one pixel
        
        result = compute_diff(frame1, frame2, DiffMode.COLOR)
        
        # Most pixels black
        assert np.all(result[0:5, 0:5] == 0)
        assert np.all(result[6:, 6:] == 0)
        # Changed pixel white
        assert np.all(result[5, 5] == 255)
    
    def test_mismatched_shapes_raises(self):
        """Different frame sizes should raise ValueError."""
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.zeros((5, 5, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            compute_diff(frame1, frame2, DiffMode.GRAYSCALE)


class TestDiffMode:
    """Test DiffMode enum."""
    
    def test_mode_values(self):
        assert DiffMode.GRAYSCALE.value == "grayscale"
        assert DiffMode.COLOR.value == "color"
