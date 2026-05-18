"""Video I/O utilities for reading frames and writing output videos."""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional

# Re-export AutoVideoWriter for convenience
from .compression import AutoVideoWriter as VideoWriter


class VideoMetadata:
    """Container for video metadata."""
    
    def __init__(self, fps: float, width: int, height: int, 
                 total_frames: int, duration_sec: float, codec: str):
        self.fps = fps
        self.width = width
        self.height = height
        self.total_frames = total_frames
        self.duration_sec = duration_sec
        self.codec = codec
    
    def __repr__(self) -> str:
        return (f"VideoMetadata(fps={self.fps:.2f}, {self.width}x{self.height}, "
                f"frames={self.total_frames}, duration={self.duration_sec:.2f}s, "
                f"codec={self.codec})")


class VideoReader:
    """Reads video frames using OpenCV with metadata extraction."""
    
    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        
        self.metadata = self._extract_metadata()
    
    def _extract_metadata(self) -> VideoMetadata:
        """Extract video metadata from OpenCV capture."""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Handle cases where FPS might be 0 or invalid
        if fps <= 0:
            fps = 30.0  # Default fallback
        
        duration_sec = total_frames / fps if fps > 0 else 0
        
        # Get codec as string
        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        
        return VideoMetadata(fps, width, height, total_frames, duration_sec, codec)
    
    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames as numpy arrays (BGR format)."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
    
    def get_frame_count(self) -> int:
        """Return total frame count."""
        return self.metadata.total_frames
    
    def release(self):
        """Release video capture resources."""
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class OpenCVVideoWriter:
    """Legacy OpenCV video writer (kept for reference)."""
    
    def __init__(self, output_path: str, fps: float, width: int, height: int,
                 codec: str = 'mp4v'):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(self.path), fourcc, fps, (width, height)
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {output_path}")
    
    def write(self, frame: np.ndarray):
        """Write a single frame."""
        self.writer.write(frame)
    
    def release(self):
        """Release video writer resources."""
        self.writer.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# Default VideoWriter now uses AutoVideoWriter with FFmpeg fallback
# Already imported on line 9
