"""Video compression with FFmpeg and fallback to OpenCV."""

import subprocess
import shutil
import numpy as np
from pathlib import Path
from typing import Optional
import cv2


# Valid codecs mapping: name -> ffmpeg codec string
CODEC_MAP = {
    'h264': 'libx264',
    'h265': 'libx265', 
    'vp9': 'libvpx-vp9',
    'av1': 'libaom-av1',
}

# OpenCV fallback codecs
OPENCV_CODEC_MAP = {
    'h264': 'mp4v',
    'h265': 'mp4v',  # OpenCV doesn't support H.265 well
    'vp9': 'mp4v',
    'av1': 'mp4v',
}


def detect_ffmpeg() -> Optional[str]:
    """Detect FFmpeg installation. Returns path or None."""
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        try:
            result = subprocess.run(
                [ffmpeg_path, '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return ffmpeg_path
        except Exception:
            pass
    return None


def validate_codec(codec: str) -> str:
    """Validate and normalize codec name."""
    codec = codec.lower().replace('hevc', 'h265')
    if codec not in CODEC_MAP:
        raise ValueError(f"Unsupported codec: {codec}. Choose from: {list(CODEC_MAP.keys())}")
    return codec


class FFmpegWriter:
    """Write video using FFmpeg subprocess with advanced codec support."""
    
    def __init__(self, output_path: str, fps: float, width: int, height: int,
                 codec: str = 'h264', crf: int = 23, preset: str = 'medium'):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        ffmpeg_path = detect_ffmpeg()
        if not ffmpeg_path:
            raise RuntimeError("FFmpeg not found")
        
        codec_str = CODEC_MAP.get(codec, 'libx264')
        
        # Build FFmpeg command
        cmd = [
            ffmpeg_path,
            '-y',  # Overwrite output
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',  # Read from stdin
            '-c:v', codec_str,
            '-crf', str(crf),
        ]
        
        # Add preset for supported codecs
        if codec in ['h264', 'h265']:
            cmd.extend(['-preset', preset])
        
        # Pixel format for compatibility
        cmd.extend(['-pix_fmt', 'yuv420p'])
        
        # VP9 and AV1 need special handling
        if codec == 'vp9':
            cmd.extend(['-b:v', '0'])  # Constant quality mode
        elif codec == 'av1':
            cmd.extend(['-cpu-used', '4'])  # Speed/quality tradeoff
        
        cmd.append(str(self.path))
        
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        self._closed = False
    
    def write(self, frame: np.ndarray):
        """Write a frame (BGR, uint8)."""
        if self._closed:
            raise RuntimeError("Writer is closed")
        self.process.stdin.write(frame.tobytes())
    
    def release(self):
        """Release resources and finalize video."""
        if self._closed:
            return
        self._closed = True
        
        if self.process.stdin:
            self.process.stdin.close()
        
        # Wait for FFmpeg to finish
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class AutoVideoWriter:
    """Automatically choose best available writer: FFmpeg or OpenCV fallback."""
    
    def __init__(self, output_path: str, fps: float, width: int, height: int,
                 codec: str = 'h264', crf: int = 23, preset: str = 'medium'):
        self.path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.crf = crf
        self.preset = preset
        
        # Try FFmpeg first
        self._using_ffmpeg = False
        self._writer = None
        
        try:
            self._writer = FFmpegWriter(
                output_path, fps, width, height,
                codec=codec, crf=crf, preset=preset
            )
            self._using_ffmpeg = True
            print(f"Using FFmpeg with {codec} codec (CRF={crf}, preset={preset})")
        except RuntimeError:
            # Fallback to OpenCV
            opencv_codec = OPENCV_CODEC_MAP.get(codec, 'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*opencv_codec)
            self._writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not self._writer.isOpened():
                raise RuntimeError(f"Failed to create video writer: {output_path}")
            
            print(f"Using OpenCV fallback with {opencv_codec} codec (compression unavailable)")
    
    @property
    def using_ffmpeg(self) -> bool:
        return self._using_ffmpeg
    
    def write(self, frame: np.ndarray):
        """Write a frame."""
        self._writer.write(frame)
    
    def release(self):
        """Release resources."""
        if self._writer:
            self._writer.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
