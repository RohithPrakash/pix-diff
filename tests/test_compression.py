"""Tests for compression module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from pix_diff.compression import (
    detect_ffmpeg, validate_codec, CODEC_MAP,
    FFmpegWriter, AutoVideoWriter
)


class TestDetectFFmpeg:
    """Test FFmpeg detection."""
    
    def test_detect_ffmpeg_found(self):
        """Should return path when ffmpeg is available."""
        with patch('shutil.which', return_value='/usr/bin/ffmpeg'):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                result = detect_ffmpeg()
                assert result == '/usr/bin/ffmpeg'
    
    def test_detect_ffmpeg_not_found(self):
        """Should return None when ffmpeg is not installed."""
        with patch('shutil.which', return_value=None):
            result = detect_ffmpeg()
            assert result is None
    
    def test_detect_ffmpeg_failure(self):
        """Should return None when ffmpeg execution fails."""
        with patch('shutil.which', return_value='/usr/bin/ffmpeg'):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 1
                result = detect_ffmpeg()
                assert result is None


class TestValidateCodec:
    """Test codec validation."""
    
    def test_valid_codecs(self):
        for codec in ['h264', 'h265', 'vp9', 'av1']:
            assert validate_codec(codec) == codec
    
    def test_hevc_alias(self):
        assert validate_codec('hevc') == 'h265'
    
    def test_invalid_codec(self):
        with pytest.raises(ValueError):
            validate_codec('invalid')


class TestAutoVideoWriter:
    """Test auto video writer with fallback."""
    
    def test_ffmpeg_fallback_to_opencv(self, tmp_path):
        """Should fall back to OpenCV when FFmpeg unavailable."""
        with patch('pix_diff.compression.detect_ffmpeg', return_value=None):
            output = tmp_path / "test.mp4"
            writer = AutoVideoWriter(str(output), 30.0, 640, 480, codec='h264')
            
            assert not writer.using_ffmpeg
            writer.release()
    
    def test_ffmpeg_writer_with_mock(self, tmp_path):
        """Should use FFmpeg when available."""
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        
        with patch('pix_diff.compression.detect_ffmpeg', return_value='/usr/bin/ffmpeg'):
            with patch('subprocess.Popen', return_value=mock_process):
                output = tmp_path / "test.mp4"
                writer = AutoVideoWriter(str(output), 30.0, 640, 480, codec='h264')
                
                assert writer.using_ffmpeg
                
                # Test write
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                writer.write(frame)
                mock_process.stdin.write.assert_called_once()
                
                writer.release()
