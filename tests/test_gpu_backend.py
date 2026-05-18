"""Tests for GPU backend."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from pix_diff.gpu_backend import (
    is_cuda_available, get_array_module, to_gpu, to_cpu,
    estimate_batch_size
)


class TestGpuBackend:
    """Test GPU backend functionality."""
    
    def test_is_cuda_available_no_cupy(self):
        """Should return False when CuPy not installed."""
        with patch('pix_diff.gpu_backend.HAS_CUDA', False):
            assert is_cuda_available() is False
    
    def test_get_array_module_cpu(self):
        """Should return numpy when CUDA unavailable."""
        with patch('pix_diff.gpu_backend.HAS_CUDA', False):
            xp = get_array_module()
            assert xp == np
    
    def test_to_gpu_no_cuda(self):
        """to_gpu should be no-op when CUDA unavailable."""
        arr = np.array([1, 2, 3])
        result = to_gpu(arr)
        assert result is arr
        assert isinstance(result, np.ndarray)
    
    def test_to_cpu_no_cuda(self):
        """to_cpu should be no-op when CUDA unavailable."""
        arr = np.array([1, 2, 3])
        result = to_cpu(arr)
        assert result is arr
    
    def test_estimate_batch_size_no_cuda(self):
        """Should return 1 when CUDA unavailable."""
        with patch('pix_diff.gpu_backend.HAS_CUDA', False):
            batch_size = estimate_batch_size((1080, 1920, 3))
            assert batch_size == 1
    
    def test_estimate_batch_size_with_cuda(self):
        """Should calculate batch size based on VRAM."""
        mock_mem_info = MagicMock()
        mock_mem_info.__getitem__ = MagicMock(side_effect=lambda x: [8*1024*1024*1024, 16*1024*1024*1024][x])
        
        with patch('pix_diff.gpu_backend.HAS_CUDA', True):
            with patch('pix_diff.gpu_backend.cp') as mock_cp:
                mock_cp.cuda.Device.return_value.mem_info = mock_mem_info
                batch_size = estimate_batch_size((1080, 1920, 3))
                assert batch_size > 1
                assert batch_size <= 32
    
    def test_estimate_batch_size_capped(self):
        """Should cap batch size at 32."""
        mock_mem_info = MagicMock()
        mock_mem_info.__getitem__ = MagicMock(side_effect=lambda x: [100*1024*1024*1024, 128*1024*1024*1024][x])
        
        with patch('pix_diff.gpu_backend.HAS_CUDA', True):
            with patch('pix_diff.gpu_backend.cp') as mock_cp:
                mock_cp.cuda.Device.return_value.mem_info = mock_mem_info
                batch_size = estimate_batch_size((480, 640, 3))
                assert batch_size == 32
