"""pix-diff: Generate pixel difference visualization from video."""

__version__ = "0.6.0"

from .diff_engine import DiffMode, compute_diff
from .video import VideoReader, VideoWriter
from .compression import AutoVideoWriter, validate_codec
from .gpu_backend import is_cuda_available
from .after_image import AfterImageAccumulator, FadeMode, TrailMode
from .main import process_video

__all__ = [
    "DiffMode", "compute_diff", "VideoReader", "VideoWriter",
    "AutoVideoWriter", "validate_codec", "is_cuda_available",
    "AfterImageAccumulator", "FadeMode", "TrailMode", "process_video"
]
