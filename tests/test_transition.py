"""Tests for transition effect."""

import pytest
import numpy as np

from pix_diff.transition import TransitionController


class TestTransitionController:
    """Test transition controller functionality."""
    
    def test_init_default(self):
        """Test default initialization."""
        tc = TransitionController(fps=30.0)
        assert tc.fps == 30.0
        assert tc.delay_frames == 0
        assert tc.duration_frames == 90  # 3s * 30fps
        assert tc.total_transition_frames == 90
    
    def test_init_custom(self):
        """Test custom delay and duration."""
        tc = TransitionController(fps=30.0, delay_sec=2.0, duration_sec=5.0)
        assert tc.delay_frames == 60
        assert tc.duration_frames == 150
        assert tc.total_transition_frames == 210
    
    def test_alpha_original_phase(self):
        """Alpha should be 0 during original phase."""
        tc = TransitionController(fps=30.0, delay_sec=2.0, duration_sec=3.0)
        
        # Before delay
        assert tc.get_alpha(0) == 0.0
        assert tc.get_alpha(30) == 0.0
        assert tc.get_alpha(59) == 0.0
    
    def test_alpha_diff_phase(self):
        """Alpha should be 1 after transition completes."""
        tc = TransitionController(fps=30.0, delay_sec=2.0, duration_sec=3.0)
        
        # After transition
        assert tc.get_alpha(150) == 1.0
        assert tc.get_alpha(200) == 1.0
    
    def test_alpha_transition_phase(self):
        """Alpha should be between 0 and 1 during transition."""
        tc = TransitionController(fps=30.0, delay_sec=1.0, duration_sec=2.0)
        
        # During transition
        alpha = tc.get_alpha(45)  # 1.5s in (75% through transition)
        assert 0.0 < alpha < 1.0
    
    def test_ease_in_out_curve(self):
        """Test smoothstep curve properties."""
        tc = TransitionController(fps=30.0)
        
        # At t=0, alpha=0
        assert tc._ease_in_out(0.0) == 0.0
        # At t=1, alpha=1
        assert tc._ease_in_out(1.0) == 1.0
        # At t=0.5, alpha should be 0.5 (symmetric)
        assert tc._ease_in_out(0.5) == 0.5
        # Should be smooth (derivative zero at boundaries)
        assert tc._ease_in_out(0.01) > 0.0
        assert tc._ease_in_out(0.99) < 1.0
    
    def test_blend_original(self):
        """Blend with alpha=0 should return original."""
        tc = TransitionController(fps=30.0)
        original = np.full((10, 10, 3), 100, dtype=np.uint8)
        diff = np.full((10, 10, 3), 200, dtype=np.uint8)
        
        result = tc.blend(original, diff, 0)
        assert np.array_equal(result, original)
    
    def test_blend_diff(self):
        """Blend with alpha=1 should return diff."""
        tc = TransitionController(fps=30.0, delay_sec=0.0, duration_sec=1.0)
        original = np.full((10, 10, 3), 100, dtype=np.uint8)
        diff = np.full((10, 10, 3), 200, dtype=np.uint8)
        
        result = tc.blend(original, diff, 50)  # After 1s at 50fps
        assert np.array_equal(result, diff)
    
    def test_blend_midpoint(self):
        """Blend at midpoint should mix both."""
        tc = TransitionController(fps=30.0, delay_sec=0.0, duration_sec=2.0)
        original = np.full((10, 10, 3), 100, dtype=np.uint8)
        diff = np.full((10, 10, 3), 200, dtype=np.uint8)
        
        result = tc.blend(original, diff, 30)  # 1s in
        # Should be between original and diff
        assert np.all(result > 100)
        assert np.all(result < 200)
    
    def test_is_active(self):
        """Test is_active method."""
        tc_active = TransitionController(fps=30.0, duration_sec=3.0)
        tc_inactive = TransitionController(fps=30.0, duration_sec=0.0)
        
        assert tc_active.is_active() is True
        # When duration is 0, transition is not active (no effect)
        # Note: duration_frames will be 1 due to max(1, ...), but duration_sec=0 means disabled
        assert tc_inactive.is_active() is False
    
    def test_repr(self):
        """Test string representation."""
        tc = TransitionController(fps=30.0, delay_sec=2.0, duration_sec=5.0)
        assert "delay=2.0s" in repr(tc)
        assert "duration=5.0s" in repr(tc)
