"""Tests for after-image module."""

import pytest
import numpy as np

from pix_diff.after_image import AfterImageAccumulator, FadeMode, TrailMode


class TestAfterImageAccumulator:
    """Test after-image motion trail functionality."""
    
    def test_init_default(self):
        """Test default initialization."""
        acc = AfterImageAccumulator((10, 10, 3))
        
        assert acc.frame_shape == (10, 10, 3)
        assert acc.fade_mode == FadeMode.EXPONENTIAL
        assert acc.fade_duration == 10
        assert acc.decay_factor == 0.85
        assert acc.trail_mode == TrailMode.MAX
        assert not acc.has_trails
    
    def test_init_fixed_mode(self):
        """Test fixed fade mode initialization."""
        acc = AfterImageAccumulator(
            (10, 10, 3),
            fade_mode=FadeMode.FIXED,
            fade_duration=5
        )
        
        assert acc.fade_mode == FadeMode.FIXED
        assert acc._age_buffer is not None
        assert acc._age_buffer.shape == (10, 10)
    
    def test_process_no_change(self):
        """Processing black diff should return black."""
        acc = AfterImageAccumulator((10, 10, 3))
        diff = np.zeros((10, 10, 3), dtype=np.uint8)
        
        result = acc.process(diff)
        assert result.shape == (10, 10, 3)
        assert np.all(result == 0)
        assert not acc.has_trails
    
    def test_exponential_fade(self):
        """Test that trails fade exponentially."""
        acc = AfterImageAccumulator(
            (10, 10, 3),
            fade_mode=FadeMode.EXPONENTIAL,
            decay_factor=0.5
        )
        
        # Create a bright diff frame
        diff = np.full((10, 10, 3), 255, dtype=np.uint8)
        
        # Process once - trail should be at full strength
        result1 = acc.process(diff)
        assert np.all(result1 == 255)
        
        # Process black frame - trail should fade
        black = np.zeros((10, 10, 3), dtype=np.uint8)
        result2 = acc.process(black)
        
        # Trail should have decayed to 255 * 0.5 = 127.5
        expected = 127
        assert np.all(result2 >= expected - 1)
        assert np.all(result2 <= expected + 1)
    
    def test_fixed_fade(self):
        """Test that trails fade linearly over fixed duration."""
        acc = AfterImageAccumulator(
            (10, 10, 3),
            fade_mode=FadeMode.FIXED,
            fade_duration=4
        )
        
        diff = np.full((10, 10, 3), 200, dtype=np.uint8)
        black = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Frame 0: full brightness
        result0 = acc.process(diff)
        assert np.all(result0 == 200)
        
        # Frame 1: 3/4 brightness (linear fade from original 200)
        result1 = acc.process(black)
        expected1 = 200 * (3/4)
        assert np.all(result1 >= expected1 - 2)
        assert np.all(result1 <= expected1 + 2)
        
        # Frame 2: 2/4 brightness (linear fade from original 200)
        result2 = acc.process(black)
        expected2 = 200 * (2/4)
        assert np.all(result2 >= expected2 - 2)
        assert np.all(result2 <= expected2 + 2)
        
        # Frame 3: 1/4 brightness (linear fade from original 200)
        result3 = acc.process(black)
        expected3 = 200 * (1/4)
        assert np.all(result3 >= expected3 - 2)
        assert np.all(result3 <= expected3 + 2)
        
        # Frame 4: should be gone (age >= duration)
        result4 = acc.process(black)
        assert np.all(result4 == 0)
    
    def test_composite_with_diff(self):
        """Trails should be composited over current diff frame."""
        acc = AfterImageAccumulator(
            (10, 10, 3),
            fade_mode=FadeMode.EXPONENTIAL,
            decay_factor=0.5
        )
        
        # First frame: bright white
        diff1 = np.full((10, 10, 3), 255, dtype=np.uint8)
        result1 = acc.process(diff1)
        assert np.all(result1 == 255)
        
        # Second frame: dimmer diff at different location
        diff2 = np.zeros((10, 10, 3), dtype=np.uint8)
        diff2[0:5, 0:5] = 100
        result2 = acc.process(diff2)
        
        # Trail from first frame should still be visible everywhere
        # (dimmed by 0.5 to 127.5)
        # Since trail (127.5) > diff (100 or 0), trail dominates
        assert np.all(result2 >= 125)
    
    def test_trail_mode_max(self):
        """MAX mode should keep brightest trail."""
        acc = AfterImageAccumulator(
            (10, 10, 3),
            trail_mode=TrailMode.MAX,
            fade_mode=FadeMode.EXPONENTIAL,
            decay_factor=1.0  # No fade for simplicity
        )
        
        diff1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        acc.process(diff1)
        
        diff2 = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = acc.process(diff2)
        
        # Should show the brighter value
        assert np.all(result == 200)
    
    def test_trail_mode_additive(self):
        """ADDITIVE mode should accumulate brightness."""
        acc = AfterImageAccumulator(
            (10, 10, 3),
            trail_mode=TrailMode.ADDITIVE,
            fade_mode=FadeMode.EXPONENTIAL,
            decay_factor=1.0  # No fade for simplicity
        )
        
        diff1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        acc.process(diff1)
        
        diff2 = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = acc.process(diff2)
        
        # Trail: 100 (from first) + 100 (from second) = 200
        # Composite: max(100, 200) = 200
        assert np.all(result == 200)
    
    def test_trail_mode_replace(self):
        """REPLACE mode should overwrite with latest."""
        acc = AfterImageAccumulator(
            (10, 10, 3),
            trail_mode=TrailMode.REPLACE,
            fade_mode=FadeMode.EXPONENTIAL,
            decay_factor=1.0  # No fade for simplicity
        )
        
        diff1 = np.full((10, 10, 3), 200, dtype=np.uint8)
        acc.process(diff1)
        
        diff2 = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = acc.process(diff2)
        
        # Should show the latest value
        assert np.all(result == 100)
    
    def test_reset(self):
        """Reset should clear all trails."""
        acc = AfterImageAccumulator((10, 10, 3))
        
        diff = np.full((10, 10, 3), 255, dtype=np.uint8)
        acc.process(diff)
        assert acc.has_trails
        
        acc.reset()
        assert not acc.has_trails
        
        black = np.zeros((10, 10, 3), dtype=np.uint8)
        result = acc.process(black)
        assert np.all(result == 0)


class TestFadeModes:
    """Test fade mode enum values."""
    
    def test_values(self):
        assert FadeMode.EXPONENTIAL.value == "exponential"
        assert FadeMode.FIXED.value == "fixed"


class TestTrailModes:
    """Test trail mode enum values."""
    
    def test_values(self):
        assert TrailMode.MAX.value == "max"
        assert TrailMode.ADDITIVE.value == "additive"
        assert TrailMode.REPLACE.value == "replace"
