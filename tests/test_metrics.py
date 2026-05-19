"""Tests for pixel comparison metrics."""

import pytest
import numpy as np

from pix_diff.metrics import (
    per_channel_metric, euclidean_metric, luminance_metric,
    weighted_rgb_metric, delta_e_cie76_metric, delta_e_cie94_metric,
    get_metric, get_metric_range, METRIC_REGISTRY
)


class TestPerChannelMetric:
    """Test per-channel absolute difference metric."""
    
    def test_identical_frames(self):
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        mag, changed = per_channel_metric(frame, frame)
        
        assert mag.shape == (10, 10)
        assert changed.shape == (10, 10)
        assert np.all(mag == 0)
        assert np.all(changed == False)
    
    def test_max_difference(self):
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.full((10, 10, 3), 255, dtype=np.uint8)
        mag, changed = per_channel_metric(frame1, frame2)
        
        assert np.all(mag == 255)
        assert np.all(changed == True)
    
    def test_threshold(self):
        frame1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        frame2 = np.full((10, 10, 3), 105, dtype=np.uint8)
        
        mag, changed = per_channel_metric(frame1, frame2, threshold=10)
        assert np.all(changed == False)
        
        mag, changed = per_channel_metric(frame1, frame2, threshold=3)
        assert np.all(changed == True)
    
    def test_partial_change(self):
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2[5, 5] = [255, 255, 255]
        
        mag, changed = per_channel_metric(frame1, frame2)
        assert changed[5, 5] == True
        assert np.all(changed[0:5, 0:5] == False)


class TestEuclideanMetric:
    """Test RGB Euclidean distance metric."""
    
    def test_identical_frames(self):
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        mag, changed = euclidean_metric(frame, frame)
        
        assert np.all(mag == 0)
        assert np.all(changed == False)
    
    def test_max_difference(self):
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.full((10, 10, 3), 255, dtype=np.uint8)
        mag, changed = euclidean_metric(frame1, frame2)
        
        expected = np.sqrt(3 * 255**2)
        assert np.allclose(mag, expected, atol=0.1)
        assert np.all(changed == True)
    
    def test_single_channel(self):
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2[:, :, 0] = 100  # Only B channel changes
        
        mag, changed = euclidean_metric(frame1, frame2)
        assert np.allclose(mag, 100.0, atol=0.1)


class TestLuminanceMetric:
    """Test Rec.709 luminance metric."""
    
    def test_identical_frames(self):
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        mag, changed = luminance_metric(frame, frame)
        
        assert np.all(mag == 0)
        assert np.all(changed == False)
    
    def test_white_difference(self):
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.full((10, 10, 3), 255, dtype=np.uint8)
        mag, changed = luminance_metric(frame1, frame2)
        
        # Luminance of white = 255
        assert np.allclose(mag, 255.0, atol=0.1)
    
    def test_rec709_weights(self):
        # Green contributes most to luminance
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2[:, :, 1] = 100  # Only G channel
        
        mag, changed = luminance_metric(frame1, frame2)
        expected = 100 * 0.7152
        assert np.allclose(mag, expected, atol=0.1)


class TestWeightedRGBMetric:
    """Test weighted RGB metric."""
    
    def test_identical_frames(self):
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        mag, changed = weighted_rgb_metric(frame, frame)
        
        assert np.all(mag == 0)
        assert np.all(changed == False)
    
    def test_white_difference(self):
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.full((10, 10, 3), 255, dtype=np.uint8)
        mag, changed = weighted_rgb_metric(frame1, frame2)
        
        # Sum of weights = 1.0, so max diff = 255
        assert np.allclose(mag, 255.0, atol=0.1)


class TestDeltaECIE76:
    """Test CIE76 Delta E metric."""
    
    def test_identical_frames(self):
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        mag, changed = delta_e_cie76_metric(frame, frame)
        
        assert np.all(mag == 0)
        assert np.all(changed == False)
    
    def test_perceptual_difference(self):
        # Gray vs white should have significant but not max Delta E
        frame1 = np.full((10, 10, 3), 128, dtype=np.uint8)
        frame2 = np.full((10, 10, 3), 255, dtype=np.uint8)
        mag, changed = delta_e_cie76_metric(frame1, frame2)
        
        # Should be changed
        assert np.all(changed == True)
        # Magnitude should be reasonable (not 0, not extremely high)
        assert np.all(mag > 0)
        assert np.all(mag < 200)
    
    def test_threshold_float(self):
        # Delta E threshold is in perceptual units
        frame1 = np.full((10, 10, 3), 128, dtype=np.uint8)
        frame2 = np.full((10, 10, 3), 130, dtype=np.uint8)
        
        mag, changed = delta_e_cie76_metric(frame1, frame2, threshold=50)
        # Small difference should be below threshold
        assert np.all(changed == False)


class TestDeltaECIE94:
    """Test CIE94 Delta E metric."""
    
    def test_identical_frames(self):
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        mag, changed = delta_e_cie94_metric(frame, frame)
        
        assert np.all(mag == 0)
        assert np.all(changed == False)
    
    def test_perceptual_difference(self):
        frame1 = np.full((10, 10, 3), 128, dtype=np.uint8)
        frame2 = np.full((10, 10, 3), 255, dtype=np.uint8)
        mag, changed = delta_e_cie94_metric(frame1, frame2)
        
        assert np.all(changed == True)
        assert np.all(mag > 0)


class TestMetricRegistry:
    """Test metric registry functions."""
    
    def test_get_metric_valid(self):
        metric_fn = get_metric('euclidean')
        assert metric_fn == euclidean_metric
    
    def test_get_metric_invalid(self):
        with pytest.raises(ValueError):
            get_metric('invalid')
    
    def test_get_metric_range(self):
        assert get_metric_range('per_channel') == (0.0, 255.0)
        assert get_metric_range('euclidean') == (0.0, 441.67)
        assert get_metric_range('delta_e_cie76') == (0.0, 100.0)
    
    def test_metric_registry_keys(self):
        expected = {'per_channel', 'euclidean', 'luminance', 
                   'weighted_rgb', 'delta_e_cie76', 'delta_e_cie94'}
        assert set(METRIC_REGISTRY.keys()) == expected
